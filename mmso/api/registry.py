"""Static serving registrations. HTTP requests select IDs, never checkpoint paths."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import PurePosixPath
import re
from types import MappingProxyType
from typing import Callable, Mapping

from torch import nn

from . import MODEL_ID
from ..artifacts import ROOT
from ..joint_model import NativeDecisionModel

# Preserve the historical v2 identity; importing these from runtime.py remains supported.
CHECKPOINT_SHA256 = "9fa7f2c3b129340977a7bf7413044b27d5a4fb4c55eec748df289f8bc122a343"
CONFIG_SHA256 = "59839e234b2c63eb61430f984356ffbf2a4e05bb125fa73df5a3111547fc9759"


def make_native_v1(config, vocabulary_size):
    return NativeDecisionModel(vocabulary_size, config["width"], config["layers"])


@dataclass(frozen=True)
class ModelRegistration:
    id: str
    checkpoint: str
    checkpoint_sha256: str
    config_sha256: str
    architecture: str
    factory: Callable[[dict, int], nn.Module]
    scope: str | None = None
    calibration_scope: str = "No calibration evaluation is recorded in this registration; temperature comes from the fingerprinted configuration"
    known_limitations: tuple[str, ...] = ()
    measured_results: Mapping | None = None

    def checkpoint_path(self):
        relative = PurePosixPath(self.checkpoint)
        if (relative.is_absolute() or ".." in relative.parts or relative.parts[:1] != ("artifacts",)
                or relative.suffix != ".safetensors"):
            raise ValueError("Registered checkpoints must be repository-relative .safetensors under artifacts/")
        path = (ROOT / self.checkpoint).resolve()
        artifact_root = (ROOT / "artifacts").resolve()
        if not artifact_root.is_relative_to(ROOT.resolve()) or not path.is_relative_to(artifact_root):
            raise ValueError("Registered checkpoint escapes artifacts/")
        config_path = path.with_name("config.json")
        if not config_path.resolve().is_relative_to((ROOT / "artifacts").resolve()):
            raise ValueError("Registered configuration escapes artifacts/")
        return path


V2_REGISTRATION = ModelRegistration(
    id=MODEL_ID,
    checkpoint="artifacts/joint-full-v2/model.safetensors",
    checkpoint_sha256=CHECKPOINT_SHA256,
    config_sha256=CONFIG_SHA256,
    architecture="native-decision-v1",
    factory=make_native_v1,
    calibration_scope="Fitted on the recorded joint-panel calibration partition; arbitrary rubrics and candidate subsets are not separately calibrated",
    known_limitations=(
        "No demonstrated real-browser or general image understanding",
        "45.66% on the exposed held-out compositional split; 72.92% in distribution",
        "Noul and numeric Score are explicit transformations of the same candidate distribution",
        "Changing the candidate set changes probabilities and confidence",
    ),
)

V3_REGISTRATION = ModelRegistration(
    id="mmso-joint-v3",
    checkpoint="artifacts/optimization-primitive-s24-v1/model.safetensors",
    checkpoint_sha256="4284d30920615018b500716d406236d919f5e38e268fdc5490c6f559af612bfa",
    config_sha256="c61ae17776932f842c0c03d67e740e252e82f9bc4b511cb7349e05dc9b82d94a",
    architecture="native-decision-v1",
    factory=make_native_v1,
    calibration_scope="Temperature fitted on separate familiar/compositional calibration panels after development selection; arbitrary rubrics, candidate subsets and new task families remain uncalibrated",
    known_limitations=(
        "Eight spoken keywords, generated 2x2 panels and a limited learned question vocabulary",
        "No demonstrated native real-browser or general image understanding; OCR control results are separate",
        "The composition-gap identities were previously known; confirmation voices and images are fresh",
        "Only two recipe seeds; substantial remaining errors and seed variability",
        "Confidence is categorical concentration, not epistemic uncertainty or an unknown-input detector",
        "Noul and numeric Score transform the same candidate distribution; changing candidates changes probabilities",
    ),
    measured_results={
        "report": "reports/optimization-v1/README.md",
        "familiar_accuracy": 0.8131510416666666,
        "known_composition_gap_accuracy": 0.669921875,
        "paraphrase_accuracy": 0.8170572916666666,
        "fresh_speakers": 58,
        "fresh_recordings": 256,
        "primary_questions_per_slice": 1536,
        "seeds_per_recipe": 2,
        "selection": "Nominated on development before confirmation; inference has no temporary primitive heads",
    },
)

# Explicit reviewed versions; preserve v2 as the compatibility default.
MODEL_REGISTRY = MappingProxyType({MODEL_ID: V2_REGISTRATION, V3_REGISTRATION.id: V3_REGISTRATION})


def registry_snapshot(registry=None):
    entries = dict(MODEL_REGISTRY if registry is None else registry)
    if MODEL_ID not in entries:
        raise ValueError(f"The registry must retain the default model {MODEL_ID}")
    for name, registration in entries.items():
        if not isinstance(registration, ModelRegistration) or registration.id != name:
            raise ValueError("Registry keys must match their ModelRegistration IDs")
        if re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", name) is None:
            raise ValueError("Invalid registered model ID")
        for fingerprint in (registration.checkpoint_sha256, registration.config_sha256):
            if re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None:
                raise ValueError("Each registered artifact requires a SHA256 fingerprint")
        if not registration.architecture or not callable(registration.factory):
            raise ValueError("Each registration requires an architecture and a local model factory")
        registration.checkpoint_path()
        if registration.measured_results is not None:
            metadata = deepcopy(dict(registration.measured_results))
            json.dumps(metadata, allow_nan=False)
            entries[name] = replace(registration, measured_results=metadata)
    return MappingProxyType(entries)
