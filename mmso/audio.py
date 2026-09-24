"""From-scratch log-mel CNN controls for real speech and sound clips."""
from __future__ import annotations

import json
import math
import resource
import sys
import time
import wave

import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from scipy.signal import resample_poly
from scipy.special import logsumexp, softmax
from safetensors.torch import load_file, save_file

from .artifacts import ROOT, local_media_path, provenance, read_manifest, sha256, validate_manifest, write_json, write_manifest
from .metrics import categorical_metrics, clustered_accuracy_interval


def read_audio(path, sample_rate=16000):
    with wave.open(str(path)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("The pilot supports PCM16 WAV only")
        sr, channels = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.
    if channels > 1:
        x = x.reshape(-1, channels).mean(1)
    if sr != sample_rate:
        divisor = math.gcd(sr, sample_rate)
        x = resample_poly(x, sample_rate // divisor, sr // divisor).astype(np.float32)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Empty or nonfinite waveform")
    return torch.from_numpy(x.copy())


class LogMel:
    """CPU preprocessing; symmetric windows use the full offline clip, not streaming."""
    def __init__(self, sample_rate=16000, n_fft=400, hop=160, bands=64):
        self.sample_rate, self.n_fft, self.hop = sample_rate, n_fft, hop
        mel_max = 2595 * math.log10(1 + sample_rate / 2 / 700)
        hz = 700 * (10 ** (torch.linspace(0, mel_max, bands + 2) / 2595) - 1)
        frequencies = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        left = (frequencies[None, :] - hz[:-2, None]) / (hz[1:-1, None] - hz[:-2, None])
        right = (hz[2:, None] - frequencies[None, :]) / (hz[2:, None] - hz[1:-1, None])
        self.filters = torch.minimum(left, right).clamp_min(0)
        self.window = torch.hann_window(n_fft)

    def __call__(self, waveform):
        if len(waveform) < self.n_fft:
            waveform = torch.nn.functional.pad(waveform, (0, self.n_fft - len(waveform)))
        spectrum = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop,
                              window=self.window, return_complex=True, center=True).abs().square()
        features = (self.filters @ spectrum).clamp_min(1e-10).log()
        features = (features - features.mean()) / features.std().clamp_min(1e-5)
        # Three stride-2 blocks followed by 4-bin pooling require multiples of 32
        # for MPS adaptive pooling. Zero is the normalized mean, not extra signal.
        features = torch.nn.functional.pad(features, (0, (-features.shape[-1]) % 32))
        return features.unsqueeze(0)


class AudioCNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten())
        self.classifier = nn.Sequential(nn.Linear(1024, 64), nn.GELU(), nn.Linear(64, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def choose_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError("Supported devices are cpu, mps, or cuda")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS requested but unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(requested)


def fit_temperature(logits, targets):
    z = np.asarray(logits, dtype=float); y = np.asarray(targets, dtype=int)
    def loss(log_t):
        scaled = z / np.exp(log_t)
        return float((logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]).mean())
    result = minimize_scalar(loss, bounds=(-3., 3.), method="bounded")
    # Calibration fit may safely choose T=1 if its bounded optimization is worse.
    return float(np.exp(result.x)) if result.success and result.fun < loss(0.) else 1.


def features_for_rows(rows, seconds, frontend):
    length = int(frontend.sample_rate * seconds)
    features = []
    for r in rows:
        signal = read_audio(local_media_path(ROOT, r["media"][0]["path"]))
        signal = torch.nn.functional.pad(signal[:length], (0, max(0, length-len(signal))))
        features.append(frontend(signal))
    return torch.stack(features)


@torch.inference_mode()
def infer_logits(model, x, device, batch=64):
    model.eval()
    return torch.cat([model(part.to(device)).cpu() for part in x.split(batch)]).numpy()


def benchmark_audio(model, rows, seconds, frontend, device, temperature):
    sample = np.random.default_rng(57).choice(len(rows), min(32, len(rows)), replace=False)
    length = int(frontend.sample_rate * seconds)
    def predict(row):
        signal = read_audio(local_media_path(ROOT, row["media"][0]["path"]))
        signal = torch.nn.functional.pad(signal[:length], (0, max(0, length-len(signal))))
        with torch.inference_mode():
            return (model(frontend(signal).unsqueeze(0).to(device)) / temperature).softmax(-1).cpu()
    for i in sample[:5]:
        predict(rows[i])
    times = []
    for i in sample:
        synchronize(device); start = time.perf_counter()
        predict(rows[i]); synchronize(device)
        times.append(time.perf_counter() - start)
    durations = [rows[i]["media"][0]["duration_seconds"] for i in sample]
    return {"timed_examples": len(times), "warm_p50_ms": float(np.median(times)*1000),
            "warm_p95_ms": float(np.quantile(times,.95)*1000), "mean_processing_realtime_factor": float(np.mean(np.array(times)/durations)),
            "mean_observed_duration_seconds": float(np.mean(durations)), "batch_size": 1,
            "includes": ["warm file read", "PCM decode", "resample", "log-mel", "model", "temperature and softmax", "device synchronization"],
            "endpointing_delay_ms": None, "cold_process_latency_ms": None,
            "streaming": False, "note": "Offline complete clips. Listening duration is not processing latency."}


def run_audio(dataset, run_id, device="auto", epochs=16, max_train_seconds=120, seed=20260923):
    if dataset not in {"speech_keywords", "sound_events"}:
        raise ValueError("Unknown audio dataset")
    report_dir = ROOT / "reports" / run_id
    if report_dir.exists():
        raise ValueError("Run IDs are immutable; select a new ID")
    if epochs < 1 or max_train_seconds <= 0:
        raise ValueError("Positive training budget required")
    start_total = time.perf_counter()
    torch.set_num_threads(4); torch.manual_seed(seed)
    selected_device = choose_device(device)
    manifest = ROOT / "evals/manifests" / f"{dataset}.jsonl"
    rows = read_manifest(manifest); audit = validate_manifest(rows)
    labels = rows[0]["labels"]
    seconds = 1 if dataset == "speech_keywords" else 5
    frontend = LogMel()
    split_rows = {s: [r for r in rows if r["split"] == s] for s in ["train","dev","calibration","test"]}
    if not all(split_rows.values()):
        raise ValueError("All four data roles are required")
    data_start = time.perf_counter()
    # Only train/development data is prepared before recipe selection.
    x_train = features_for_rows(split_rows["train"], seconds, frontend)
    x_dev = features_for_rows(split_rows["dev"], seconds, frontend)
    y_train = torch.tensor([labels.index(r["target"]) for r in split_rows["train"]])
    y_dev = np.array([labels.index(r["target"]) for r in split_rows["dev"]])
    preparation_seconds = time.perf_counter() - data_start
    model = AudioCNN(len(labels))
    # Fixed weights/input validate actual CPU/MPS forward agreement before training.
    model.eval(); reference = model(x_train[:2]).detach().numpy()
    model.to(selected_device)
    observed = infer_logits(model, x_train[:2], selected_device)
    backend_difference = float(np.max(np.abs(reference-observed)))
    if backend_difference > 2e-4:
        raise ValueError(f"CPU/device discrepancy {backend_difference}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    generator = torch.Generator().manual_seed(seed)
    history = []; train_seconds = 0.; selection_seconds = 0.; best = float("inf"); best_state = None; best_epoch = 0
    for epoch in range(1, epochs+1):
        model.train(); order = torch.randperm(len(x_train), generator=generator)
        losses = []; completed_steps = 0
        for indices in order.split(32):
            if train_seconds >= max_train_seconds:
                break
            synchronize(selected_device); t = time.perf_counter()
            inputs = x_train[indices].to(selected_device); targets = y_train[indices].to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward(); optimizer.step(); synchronize(selected_device)
            train_seconds += time.perf_counter() - t
            losses.append(float(loss.detach().cpu())); completed_steps += 1
        if not completed_steps:
            break
        t = time.perf_counter()
        dev_logits = infer_logits(model, x_dev, selected_device)
        dev_metrics = categorical_metrics(y_dev, softmax(dev_logits,axis=1), labels)
        selection_seconds += time.perf_counter() - t
        history.append({"epoch": epoch,"steps": completed_steps,"mean_train_loss": float(np.mean(losses)),
                        "dev_nll": dev_metrics["nll"],"dev_accuracy": dev_metrics["accuracy"],"train_seconds":train_seconds})
        print(json.dumps({"dataset":dataset,**history[-1]}),flush=True)
        if dev_metrics["nll"] < best:
            best = dev_metrics["nll"]; best_epoch = epoch
            best_state = {k:v.detach().cpu().clone().contiguous() for k,v in model.state_dict().items()}
    if best_state is None:
        raise ValueError("No completed training update")
    model.load_state_dict(best_state)
    # Freeze model, fit T on calibration only, then load/evaluate the test once.
    x_cal = features_for_rows(split_rows["calibration"],seconds,frontend)
    y_cal = np.array([labels.index(r["target"]) for r in split_rows["calibration"]])
    temperature = fit_temperature(infer_logits(model,x_cal,selected_device),y_cal)
    x_test = features_for_rows(split_rows["test"],seconds,frontend)
    y_test = np.array([labels.index(r["target"]) for r in split_rows["test"]])
    test_logits = infer_logits(model,x_test,selected_device)
    raw = softmax(test_logits,axis=1); calibrated = softmax(test_logits/temperature,axis=1)
    prior = np.bincount(y_train.numpy(),minlength=len(labels)).astype(float); prior /= prior.sum()
    checkpoint = ROOT / "artifacts" / run_id / "model.safetensors"
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    save_file(best_state,str(checkpoint))
    restored = AudioCNN(len(labels)).to(selected_device)
    restored.load_state_dict(load_file(str(checkpoint)))
    restored_logits = infer_logits(restored,x_test[:2],selected_device)
    if not np.allclose(restored_logits,test_logits[:2],atol=2e-4,rtol=0):
        raise ValueError("Checkpoint reload changed predictions")
    configuration = {"architecture":"AudioCNN-v1","labels":labels,"sample_rate":16000,"seconds":seconds,
                     "frontend":{"n_fft":400,"hop":160,"mel_bands":64,"normalization":"per-clip mean/std", "offline_centered_stft":True,"time_pad_multiple":32},
                     "seed":seed,"max_epochs":epochs,"max_train_seconds":max_train_seconds,"chosen_epoch":best_epoch,
                     "optimizer":"AdamW","learning_rate":1e-3,"weight_decay":.01,"batch_size":32,"temperature":temperature}
    write_json(checkpoint.with_name("config.json"),configuration)
    timings = benchmark_audio(restored,split_rows["test"],seconds,frontend,selected_device,temperature)
    write_manifest(report_dir / "predictions.jsonl", [
        {"id":r["id"],"target":r["target"],"raw_probabilities":raw[i].tolist(),"probabilities":calibrated[i].tolist()}
        for i,r in enumerate(split_rows["test"])])
    report = {"run_id":run_id,"dataset":dataset,"status":"completed","kind":"categorical_audio",
        "scope":"single-seed implementation baseline, not architecture search or multimodal proof",
        "provenance":provenance(manifest),"dataset_audit":audit,"configuration":configuration,
        "device":str(selected_device),"torch_version":torch.__version__,"parameters":sum(p.numel() for p in model.parameters()),
        "checkpoint":str(checkpoint.relative_to(ROOT)),"checkpoint_sha256":sha256(checkpoint),"cpu_device_max_abs_logit_difference":backend_difference,
        "test_raw":categorical_metrics(y_test,raw,labels),"test_calibrated":categorical_metrics(y_test,calibrated,labels),
        "test_prior":categorical_metrics(y_test,np.tile(prior,(len(y_test),1)),labels),
        "accuracy_interval":clustered_accuracy_interval(raw.argmax(1)==y_test,[r["group_id"] for r in split_rows["test"]]),
        "history":history,"timing":{"feature_preparation_train_dev_seconds":preparation_seconds,
            "synchronized_optimizer_step_seconds":train_seconds,"dev_selection_seconds":selection_seconds,
            "total_wall_seconds":time.perf_counter()-start_total,"inference":timings},
        "memory":{"process_high_water_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024),
                  "mps_driver_end_bytes":torch.mps.driver_allocated_memory() if selected_device.type=="mps" else None},
        "limitations":["Custom pilot split, not official leaderboard protocol","Single training seed","Offline audio; no endpointing, unknown-class or overlapping-event evidence","Test outcomes do not feed model selection"]}
    write_json(report_dir / "report.json",report)
    print(json.dumps({"run_id":run_id,"accuracy":report["test_raw"]["accuracy"],"macro_f1":report["test_raw"]["macro_f1"],
                      "report":str((report_dir/"report.json").relative_to(ROOT))}),flush=True)
    return report


def predict_audio(checkpoint, audio_path, device="auto"):
    checkpoint = ROOT / checkpoint
    config = json.loads(checkpoint.with_name("config.json").read_text())
    chosen = choose_device(device); model=AudioCNN(len(config["labels"])).to(chosen)
    model.load_state_dict(load_file(str(checkpoint))); model.eval()
    signal=read_audio(audio_path); length=int(config["sample_rate"]*config["seconds"])
    signal=torch.nn.functional.pad(signal[:length],(0,max(0,length-len(signal))))
    with torch.inference_mode():
        p=(model(LogMel()(signal).unsqueeze(0).to(chosen))/config["temperature"]).softmax(-1).cpu().numpy()[0]
    return {"prediction":config["labels"][int(p.argmax())],"probabilities":dict(zip(config["labels"],p.tolist())),
            "input_scope":"offline clip; truncated/padded to checkpoint duration", "duration_limit_seconds":config["seconds"]}
