"""Measured, label-blind inference adapters for the MiSO v4 selection study."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
from scipy.signal import resample_poly
import torch

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prompt_for(row, *, generated_json=False):
    options = "\n".join(f"{LETTERS[i]}. {choice}" for i, choice in enumerate(row["choices"]))
    contract = ("Return a JSON object with a probabilities array in option order and a choice letter. "
                "Probabilities must sum to one. Do not include an explanation.") if generated_json else (
                "Choose the best option. Reply with exactly its single capital letter and nothing else. Do not explain or think aloud.")
    return f"{row['question']}\n\nOptions:\n{options}\n\n{contract}"


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, Mapping):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


def decode_media(root, items, policy):
    import soundfile as sf
    images, audios = [], []
    for item in items:
        path = root / item["path"]
        if item["modality"] == "image":
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            image.thumbnail((policy["image_max_side"], policy["image_max_side"]), Image.Resampling.LANCZOS)
            images.append(image)
        elif item["modality"] == "audio":
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
            audio = audio.mean(axis=1)
            if len(audio) / sample_rate > policy["audio_max_seconds"]:
                raise ValueError("Audio exceeds frozen duration limit; do not silently truncate")
            target_rate = policy["audio_sample_rate"]
            if sample_rate != target_rate:
                gcd = math.gcd(sample_rate, target_rate)
                audio = resample_poly(audio, target_rate // gcd, sample_rate // gcd).astype(np.float32)
            audios.append(audio)
        else:
            raise ValueError("Unsupported media type")
    return images, audios


class Backbone:
    def __init__(self, spec):
        import transformers as tr
        self.spec, self.key = spec, spec["key"]
        download = json.loads((Path("/cache/study/downloads") / (self.key + ".json")).read_text())
        if download["revision"] != spec["revision"] or download["model"] != spec["id"]:
            raise ValueError("Cached checkpoint identity differs from frozen candidate")
        self.snapshot = download["snapshot"]
        path = self.snapshot
        self.family = "gemma" if self.key.startswith("gemma") else "qwen" if self.key.startswith("qwen") else self.key
        self.processor = tr.AutoProcessor.from_pretrained(path, trust_remote_code=self.family in {"minicpmo45", "phi4mm"}, local_files_only=True)
        kwargs = dict(torch_dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
                      attn_implementation="sdpa", output_loading_info=True)
        if self.key.startswith("gemma4"):
            cls = tr.Gemma4UnifiedForConditionalGeneration if self.key == "gemma4-12b" else tr.Gemma4ForConditionalGeneration
        elif self.key.startswith("qwen25"):
            cls = tr.Qwen2_5OmniThinkerForConditionalGeneration
        elif self.key.startswith("qwen3"):
            cls = tr.Qwen3OmniMoeThinkerForConditionalGeneration
        elif self.key == "minicpmo45":
            cls = tr.AutoModel
            kwargs.update(trust_remote_code=True, init_vision=True, init_audio=True, init_tts=False)
        elif self.key == "phi4mm":
            cls = tr.AutoModelForCausalLM
            kwargs.update(trust_remote_code=True)
        else:
            raise ValueError("Unsupported model adapter")
        self.model, self.loading = cls.from_pretrained(path, **kwargs)
        missing = self.loading.get("missing_keys", [])
        if missing:
            raise ValueError(f"Missing pretrained tensors would produce an invalid comparison: {missing[:10]}")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = self.processor.tokenizer
        self.token_ids = []
        for letter in LETTERS[:10]:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Candidate letter is not one token for {self.key}: {letter}")
            self.token_ids.append(ids[0])
        self.feature = None
        self.head = self.model.get_output_embeddings()
        self.head_hook = self.head.register_forward_pre_hook(self.capture_feature)
        self.events = []
        self.encoder_hooks = []
        chosen = []
        encoder_names = {"vision_tower", "audio_tower", "visual", "vpm", "apm", "vision_model", "audio_model", "embed_vision", "embed_audio"}
        for name, module in self.model.named_modules():
            if name.split(".")[-1] in encoder_names and not any(name.startswith(x + ".") for x in chosen):
                chosen.append(name)
                self.encoder_hooks.append(module.register_forward_pre_hook(self.start_encoder(name)))
                self.encoder_hooks.append(module.register_forward_hook(self.end_encoder(name)))
        self.encoder_names = chosen

    def capture_feature(self, module, inputs):
        self.feature = inputs[0][:, -1, :].detach()

    def start_encoder(self, name):
        def before(module, inputs):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            self.events.append((name, start, end))
        return before

    def end_encoder(self, name):
        def after(module, inputs, output):
            for event_name, start, end in reversed(self.events):
                if event_name == name:
                    end.record(); break
        return after

    def prepare(self, prompt, images, audios):
        if self.family == "phi4mm":
            markers = "".join(f"<|image_{i + 1}|>" for i in range(len(images)))
            markers += "".join(f"<|audio_{i + 1}|>" for i in range(len(audios)))
            text = f"<|user|>\n{markers}{prompt}<|end|>\n<|assistant|>\n"
            kwargs = {"text": text, "return_tensors": "pt"}
            if images: kwargs["images"] = images
            if audios: kwargs["audios"] = [(a, 16000) for a in audios]
            return self.processor(**kwargs), text
        if self.family == "minicpmo45":
            content = "\n".join([*("<image>./</image>" for _ in images), *("<audio>./</audio>" for _ in audios), prompt])
            text = self.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                        add_generation_prompt=True, use_tts_template=bool(audios), enable_thinking=False)
            data = self.processor([text], [images], [audios], [[0] * len(audios)],
                                  max_slice_nums=9, return_tensors="pt", max_length=8192)
            data.pop("image_sizes", None)
            if "position_ids" not in data:
                data["position_ids"] = torch.arange(data["input_ids"].shape[-1])[None, :]
            return data, text
        content = [{"type": "image", "image": image} for image in images]
        content.extend({"type": "audio", "audio": audio} for audio in audios)
        content.append({"type": "text", "text": prompt})
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
        if images: kwargs["images"] = images
        if audios: kwargs.update(audio=audios, sampling_rate=16000)
        return self.processor(**kwargs), text

    def forward(self, inputs):
        if self.family == "minicpmo45":
            return self.model(data=inputs, use_cache=False, return_dict=True, logits_to_keep=1)
        options = dict(use_cache=False, return_dict=True)
        params = inspect.signature(self.model.forward).parameters
        if "logits_to_keep" in params: options["logits_to_keep"] = 1
        elif "num_logits_to_keep" in params: options["num_logits_to_keep"] = 1
        return self.model(**inputs, **options)

    def infer(self, root, row, policy):
        # Deliberately receive only question, choices, and media. No target,
        # transcript, source intent, annotation box, or row ID enters the prompt.
        start = time.perf_counter()
        images, audios = decode_media(root, row["media"], policy)
        decoded = time.perf_counter()
        prompt = prompt_for(row)
        token_started = time.perf_counter()
        text_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        tokenization_seconds = time.perf_counter() - token_started
        # The isolated text-tokenizer probe is excluded from request latency;
        # processor timing includes the actual tokenizer invocation.
        processed_start = time.perf_counter()
        inputs, rendered = self.prepare(prompt, images, audios)
        processed = time.perf_counter()
        seq_length = inputs["input_ids"].shape[-1]
        if seq_length > policy["max_input_tokens"]:
            raise ValueError(f"Input exceeds frozen token ceiling: {seq_length}")
        inputs = move(inputs, "cuda")
        torch.cuda.synchronize()
        transferred = time.perf_counter()
        self.events = []; self.feature = None
        with torch.inference_mode():
            output = self.forward(inputs)
            torch.cuda.synchronize()
            forwarded = time.perf_counter()
            logits = output.logits[0, -1].float()
            selected = logits[self.token_ids[:len(row["choices"])]]
            probabilities = selected.softmax(dim=-1)
            mass = (torch.logsumexp(selected, dim=-1) - torch.logsumexp(logits, dim=-1)).exp()
            top_token = int(logits.argmax())
            values = selected.cpu().numpy()
            probs = probabilities.cpu().numpy()
            allowed_mass = float(mass.cpu())
            # Feature transfer belongs to research extraction, not the serving
            # path. Keep it out of request timing while recording it separately.
            torch.cuda.synchronize()
            scored = time.perf_counter()
            if self.feature is None:
                raise ValueError("Output-head feature hook did not fire")
            feature = self.feature[0].float().cpu().numpy()
            torch.cuda.synchronize()
        encoder_ms = defaultdict_sum([(name, begin.elapsed_time(end)) for name, begin, end in self.events])
        timings = {"decode_ms": (decoded - start) * 1000, "text_tokenizer_probe_ms": tokenization_seconds * 1000,
                   "processor_ms": (processed - processed_start) * 1000, "h2d_ms": (transferred - processed) * 1000,
                   "forward_ms": (forwarded - transferred) * 1000, "score_and_d2h_ms": (scored - forwarded) * 1000,
                   "request_ms": ((decoded - start) + (scored - processed_start)) * 1000,
                   "research_feature_copy_ms": (time.perf_counter() - scored) * 1000,
                   "encoder_gpu_ms": encoder_ms}
        return {"status": "ok", "logits": values.tolist(), "probabilities": probs.tolist(),
                "allowed_vocabulary_mass": allowed_mass, "unconstrained_top_token": self.tokenizer.decode([top_token]),
                "unconstrained_is_candidate": top_token in self.token_ids[:len(row["choices"])],
                "tokens": {"plain_question_and_choices": len(text_tokens), "processed_input": int(seq_length)},
                "timings": timings}, feature


def defaultdict_sum(pairs):
    result = {}
    for name, value in pairs:
        result[name] = result.get(name, 0.0) + float(value)
    return result


def run_candidate(root, key, phase, output):
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(20260924)
    protocol = json.loads((root / "evals/v4-selection-protocol-v1.json").read_text())
    registry = json.loads((root / "evals/v4-candidates-v1.json").read_text())
    spec = next(m for m in registry["models"] if m["key"] == key)
    if spec["gated"]: raise ValueError("No access to gated candidate")
    manifest_path = root / "evals/manifests/v4_selection_v1.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    for row in rows:
        for item in row["media"]:
            if digest(root / item["path"]) != item["sha256"]:
                raise ValueError("Media hash mismatch before inference")
    selected = [r for r in rows if (r["split"] == "confirmation") == (phase == "confirmation")]
    if phase == "smoke":
        selected = [next(r for r in rows if r["split"] == "train" and r["track"] == track) for track in protocol["tracks"]]
    elif phase not in {"development", "confirmation"}:
        raise ValueError("Unknown phase")
    selected.sort(key=lambda r: hashlib.sha256(("order:" + r["id"]).encode()).hexdigest())
    source_hashes = {name: digest(root / name) for name in ["evals/v4-selection-protocol-v1.json", "evals/v4-candidates-v1.json", "evals/manifests/v4_selection_v1.jsonl", "mmso/backbone_study.py"]}
    begun = {"key": key, "phase": phase, "expected": len(selected), "model": spec,
             "source_hashes": source_hashes, "hardware": {"name": torch.cuda.get_device_name(),
             "memory_bytes": torch.cuda.get_device_properties(0).total_memory, "torch": torch.__version__, "cuda": torch.version.cuda},
             "versions": {x: importlib.metadata.version(x) for x in ["transformers", "accelerate", "numpy", "librosa", "soundfile"]}}
    (output / "started.json").write_text(json.dumps(begun, indent=2) + "\n")
    model = Backbone(spec)
    load_seconds = time.perf_counter() - started
    parameters = sum(p.numel() for p in model.model.parameters())
    warmup_start = time.perf_counter()
    warmups = []
    for track in protocol["tracks"]:
        row = next(r for r in rows if r["split"] == "train" and r["track"] == track)
        try:
            model.infer(root, {k: row[k] for k in ("question", "choices", "media")}, protocol["input_policy"])
            warmups.append({"track": track, "status": "ok"})
        except Exception as error:
            warmups.append({"track": track, "status": "error", "error": str(error)})
            torch.cuda.empty_cache()
    warmup_seconds = time.perf_counter() - warmup_start
    torch.cuda.reset_peak_memory_stats()
    records, features, feature_ids = [], [], []
    for row in selected:
        if time.perf_counter() - started > 3350:
            raise TimeoutError("Internal study timeout; partial predictions retained")
        try:
            result, feature = model.infer(root, {k: row[k] for k in ("question", "choices", "media")}, protocol["input_policy"])
            features.append(feature); feature_ids.append(row["id"])
        except Exception as error:
            result = {"status": "error", "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}
            torch.cuda.empty_cache()
        records.append({"id": row["id"], "track": row["track"], "split": row["split"], **result})
        with (output / "predictions.jsonl").open("a") as stream:
            stream.write(json.dumps(records[-1], allow_nan=False) + "\n")
        if len(records) % 25 == 0 or phase == "smoke":
            print(json.dumps({"key": key, "phase": phase, "completed": len(records), "expected": len(selected), "last_status": result["status"]}), flush=True)
    if features:
        np.savez_compressed(output / "features.npz", ids=np.asarray(feature_ids), features=np.stack(features))
    summary = {**begun, "status": "completed", "completed": len(records), "errors": sum(r["status"] != "ok" for r in records),
               "load_seconds": load_seconds, "warmup_seconds": warmup_seconds, "warmups": warmups,
               "function_seconds": time.perf_counter() - started, "loaded_parameters": parameters,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "encoder_modules": model.encoder_names,
               "loading_info": {k: v for k, v in model.loading.items() if k != "error_msgs"},
               "features_sha256": digest(output / "features.npz") if features else None,
               "predictions_sha256": digest(output / "predictions.jsonl"), "accuracy_computed_remotely": False}
    (output / "result.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary
