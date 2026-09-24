"""A process-cached, serialized inference runtime for the published checkpoint."""
from __future__ import annotations

import json
from copy import deepcopy
import threading
import time
from uuid import uuid4

import numpy as np
from safetensors.torch import load_file
import torch

from ..artifacts import sha256
from ..audio import choose_device
from ..joint_model import Vocabulary, encode_requests
from .media import decode_audio, decode_image, MAX_AUDIO_BYTES, MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, SAMPLE_RATES
from .schema import (MODEL_ID, DecisionRequest, DecisionResponse, NoulQuestion, ScoreQuestion)
from .registry import (CHECKPOINT_SHA256, CONFIG_SHA256, ModelRegistration, V2_REGISTRATION)


def format_result(question, candidates, probabilities, threshold):
    ids = [c["id"] for c in candidates]
    values = np.asarray(probabilities, dtype=float)
    best = int(values.argmax())
    tied = np.flatnonzero(np.isclose(values, values[best], atol=1e-7, rtol=0)).tolist()
    prediction = ids[best] if len(tied) == 1 else None
    abstained = prediction is None or float(values[best]) < threshold
    # Equal probabilities have equal ranks; candidate IDs only break display-order ties.
    order = sorted(range(len(ids)), key=lambda i: (-values[i], ids[i]))
    ranking = [{"id": ids[i], "probability": float(values[i]),
                "rank": 1 + int(np.sum(values > values[i] + 1e-7))} for i in order]
    result = {"type": question.type, "probabilities": dict(zip(ids, values.tolist())), "ranking": ranking,
              "prediction": prediction, "decision": None if abstained else prediction,
              "abstained": abstained, "ties": sorted(ids[i] for i in tied) if len(tied) > 1 else [],
              "confidence": float(values[best])}
    if isinstance(question, NoulQuestion):
        result.update(probability_true=result["probabilities"]["true"],
                      value=None if abstained else prediction == "true")
    elif isinstance(question, ScoreQuestion):
        rubric = np.asarray([level.value for level in question.levels])
        expected = float(np.dot(values, rubric))
        result.update(expected_value=expected, value=None if abstained else expected,
                      range=[float(rubric.min()), float(rubric.max())])
    return result


class NativeRuntime:
    def __init__(self, device="auto", registration: ModelRegistration = V2_REGISTRATION, *, inference_lock=None):
        self.registration = registration
        self.checkpoint = registration.checkpoint_path()
        self.checkpoint_sha256 = sha256(self.checkpoint)
        if self.checkpoint_sha256 != registration.checkpoint_sha256:
            raise ValueError(f"The registered {registration.id} checkpoint fingerprint does not match")
        config_path = self.checkpoint.with_name("config.json")
        self.config_sha256 = sha256(config_path)
        if self.config_sha256 != registration.config_sha256:
            raise ValueError(f"The registered {registration.id} configuration fingerprint does not match")
        self.config = json.loads(config_path.read_text())
        if self.config["architecture"] != registration.architecture or self.config.get("mode") != "full":
            raise ValueError("Unsupported registered checkpoint architecture or modality mode")
        self.temperature = self.config["temperature"]
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Invalid checkpoint calibration temperature")
        self.vocabulary = Vocabulary(self.config["vocabulary"])
        self.device = choose_device(device)
        self.model = registration.factory(self.config, len(self.vocabulary.tokens))
        if not all(callable(getattr(self.model, method, None)) for method in ("encode_observations", "decide")):
            raise ValueError("Registered factory must implement the native observation/candidate interface")
        self.model.load_state_dict(load_file(str(self.checkpoint)))
        self.model.to(self.device).eval()
        self.lock = inference_lock if inference_lock is not None else threading.Lock()

    def model_card(self):
        card = {"id": self.registration.id, "display_name": "MiSO " + self.registration.id.rsplit("-", 1)[-1],
                "object": "model", "checkpoint_sha256": self.checkpoint_sha256,
                "config_sha256": self.config_sha256, "architecture": self.config["architecture"],
                "parameters": sum(p.numel() for p in self.model.parameters()), "device": str(self.device),
                "scope": self.registration.scope if self.registration.scope is not None else self.config["scope"], "status": "experimental",
                "capabilities": {"input_modalities": ["image", "audio", "text"],
                    "output_types": ["choice", "noul", "score", "ranking"],
                    "required_media": {"image": 1, "audio": 1},
                    "text_role": "Each text block supplies a referenced question; no chat history or free context",
                    "image": {"media_types": ["image/png", "image/jpeg"], "max_bytes": MAX_IMAGE_BYTES,
                              "max_pixels": MAX_IMAGE_PIXELS, "max_edge": 4096, "model_size": [128, 128],
                              "layout": "generated 2x2 symbol panel", "animation": False},
                    "audio": {"media_types": ["audio/wav"], "encoding": "PCM16", "channels": 1,
                              "sample_rates": list(SAMPLE_RATES), "max_bytes": MAX_AUDIO_BYTES,
                              "max_seconds": 1.0, "model_sample_rate": 16000, "streaming": False,
                              "words": ["down", "go", "left", "no", "right", "stop", "up", "yes"]},
                    "text": {"vocabulary": self.vocabulary.tokens[2:], "question_max_tokens": 24,
                             "candidate_max_tokens": 8, "max_questions": 32,
                             "candidate_count": [2, 32], "tokenizer": "lowercase ASCII words; punctuation is ignored"},
                    "remote_media_urls": False, "video": False, "generation": False, "tool_execution": False},
                "calibration": {"temperature": self.temperature,
                    "confidence": "maximum temperature-scaled categorical probability; not epistemic uncertainty",
                    "scope": self.registration.calibration_scope},
                "known_limitations": list(self.registration.known_limitations)}
        if self.registration.measured_results is not None:
            card["measured_results"] = deepcopy(dict(self.registration.measured_results))
        return card

    def decide(self, request: DecisionRequest):
        if request.model != self.registration.id:
            raise ValueError("Request model does not match this loaded runtime")
        neural = request.neural_requests()
        question, candidates, mask = encode_requests(neural, self.vocabulary)
        # Only one device inference at a time; each HTTP request has fresh observation tensors.
        with self.lock, torch.inference_mode():
            image_block = next(b for b in request.input if b.type == "image")
            audio_block = next(b for b in request.input if b.type == "audio")
            image, image_info = decode_image(image_block.source)
            audio, audio_info = decode_audio(audio_block.source)
            visual, acoustic = self.model.encode_observations(image[None].to(self.device), audio[None].to(self.device))
            count = len(neural)
            logits = self.model.decide(visual.expand(count, -1, -1), acoustic.expand(count, -1),
                                      question.to(self.device), candidates.to(self.device), mask.to(self.device))
            probabilities = (logits / self.temperature).softmax(-1).cpu().numpy()
        results = {}
        for (name, typed), item, p in zip(request.questions.items(), neural, probabilities):
            results[name] = format_result(typed, item["candidates"], p[:len(item["candidates"])], request.abstain_threshold)
        return DecisionResponse(id="dec_" + uuid4().hex, created=int(time.time()), model=self.registration.id,
            checkpoint_sha256=self.checkpoint_sha256, results=results,
            input_summary={"image": image_info, "audio": audio_info, "questions": len(neural)},
            semantics={"confidence": "max categorical probability after temperature scaling",
                       "ranking": "descending probability, conditional on the supplied candidate set",
                       "score": "expected caller-supplied rubric value; not a separately trained regression head",
                       "noul": "P(yes) from the yes/no candidate distribution",
                       "abstention": "decision/value are null for ties or confidence below abstain_threshold",
                       "temperature": self.temperature, "device": str(self.device)})
