"""Bounded, in-memory media ingestion. No remote URLs or server file paths."""
from __future__ import annotations

import base64
import binascii
from io import BytesIO
import math
import warnings
import wave

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.signal import resample_poly
import torch

from .schema import Base64Source
from ..audio import LogMel

MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_AUDIO_BYTES = 128 * 1024
MAX_IMAGE_PIXELS = 2048 * 2048
SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)


def inline_bytes(source, kind):
    if isinstance(source, Base64Source):
        mime, encoded = source.media_type, source.data
    else:
        header, separator, encoded = source.url.partition(",")
        if not separator or not header.startswith("data:") or not header.endswith(";base64"):
            raise ValueError("Only inline base64 data URLs are accepted; remote URLs and paths are unsupported")
        mime = header[5:-7]
    expected = {"image/png", "image/jpeg"} if kind == "image" else {"audio/wav"}
    if mime not in expected:
        raise ValueError(f"Unsupported {kind} media type: use {', '.join(sorted(expected))}")
    limit = MAX_IMAGE_BYTES if kind == "image" else MAX_AUDIO_BYTES
    if len(encoded) > 4 * math.ceil(limit / 3):
        raise ValueError(f"{kind} exceeds its {limit}-byte decoded limit")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Invalid base64 media data") from None
    if not data or len(data) > limit:
        raise ValueError(f"{kind} is empty or exceeds its {limit}-byte decoded limit")
    return data, mime


def decode_image(source):
    data, mime = inline_bytes(source, "image")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                expected = {"image/png": "PNG", "image/jpeg": "JPEG"}[mime]
                if image.format != expected:
                    raise ValueError("Image bytes do not match the declared media type")
                width, height = image.size
                if width * height > MAX_IMAGE_PIXELS or max(width, height) > 4096:
                    raise ValueError("Image exceeds 4,194,304 pixels or a 4096-pixel edge")
                if getattr(image, "is_animated", False):
                    raise ValueError("Animated images are unsupported")
                image.verify()
            with Image.open(BytesIO(data)) as image:
                pixels = np.array(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)).copy()
        tensor = torch.from_numpy(pixels).permute(2, 0, 1).float() / 127.5 - 1
        return tensor, {"original_size": [width, height], "model_size": [128, 128], "media_type": mime}
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError,
            Image.DecompressionBombWarning):
        raise ValueError("Invalid or oversized PNG/JPEG image") from None


def decode_audio(source):
    data, _ = inline_bytes(source, "audio")
    try:
        with wave.open(BytesIO(data), "rb") as audio:
            rate, count = audio.getframerate(), audio.getnframes()
            if audio.getsampwidth() != 2 or audio.getnchannels() != 1 or audio.getcomptype() != "NONE":
                raise ValueError("Audio must be uncompressed mono PCM16 WAV")
            if rate not in SAMPLE_RATES:
                raise ValueError(f"Unsupported sample rate; use one of {list(SAMPLE_RATES)} Hz")
            if not 0 < count <= rate:
                raise ValueError("Audio must contain more than zero and at most 1.0 seconds; clips are never truncated")
            frames = audio.readframes(count)
            if len(frames) != count * 2:
                raise ValueError("Truncated WAV data")
    except (wave.Error, EOFError):
        raise ValueError("Invalid PCM16 WAV audio") from None
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)
    waveform = torch.from_numpy(samples.copy())
    waveform = torch.nn.functional.pad(waveform, (0, 16000 - len(waveform)))
    return LogMel()(waveform), {"duration_seconds": count / rate, "sample_rate": rate,
                                "model_sample_rate": 16000, "padded_to_seconds": 1.0,
                                "channels": 1, "streaming": False}
