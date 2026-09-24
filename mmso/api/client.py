"""A dependency-free synchronous client. Local paths are read by the caller."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from . import MODEL_ID


class APIError(RuntimeError):
    def __init__(self, status, error):
        self.status, self.error = status, error
        super().__init__(f"HTTP {status}: {error.get('message', error)}")


def media_block(path, kind, media_type):
    """Encode a caller-local file; no file path is sent to the server."""
    return {"type": kind, "source": {"type": "base64", "media_type": media_type,
            "data": base64.b64encode(Path(path).read_bytes()).decode("ascii")}}


def _text_block(block, text_id):
    if set(block) != {"type", "text"} or not isinstance(block["text"], str):
        raise ValueError("Text content must contain only type and text")
    return {"type": "text", "id": text_id, "text": block["text"]}


def from_openai_content(content, *, text_id="prompt"):
    """Normalize the documented inline Responses image/text + Chat audio subset.

    This accepts a list of content blocks, not messages or a provider request.
    Reference the optional text block with question_ref=text_id in our API.
    """
    if not isinstance(content, list):
        raise ValueError("Supply a content-block list, not messages or chat history")
    output = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Each content block must be an object")
        kind = block.get("type")
        if kind == "input_text":
            output.append(_text_block(block, text_id))
        elif kind == "input_image" and set(block) == {"type", "image_url"}:
            url = block["image_url"]
            if not isinstance(url, str) or not url.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
                raise ValueError("Only inline PNG/JPEG base64 image data URLs are supported")
            output.append({"type": "image", "source": {"type": "data_url", "url": url}})
        elif kind == "input_audio" and set(block) == {"type", "input_audio"}:
            audio = block["input_audio"]
            if not isinstance(audio, dict) or set(audio) != {"data", "format"} or audio["format"] != "wav" or not isinstance(audio["data"], str):
                raise ValueError("Only input_audio with base64 data and format='wav' is supported")
            output.append({"type": "audio", "source": {"type": "base64", "media_type": "audio/wav", "data": audio["data"]}})
        else:
            raise ValueError("Unsupported OpenAI content block or extra fields; accepted types are input_text, inline input_image, and WAV input_audio")
    if sum(block["type"] == "text" for block in output) > 1:
        raise ValueError("This adapter accepts at most one text block; use named native text blocks for multiple questions")
    return output


def from_claude_content(content, *, text_id="prompt"):
    """Normalize Claude text/base64-image blocks; add a native audio block separately."""
    if not isinstance(content, list):
        raise ValueError("Supply a content-block list, not messages or chat history")
    output = []
    for block in content:
        if not isinstance(block, dict):
            raise ValueError("Each content block must be an object")
        if block.get("type") == "text":
            output.append(_text_block(block, text_id))
        elif block.get("type") == "image" and set(block) == {"type", "source"}:
            source = block["source"]
            if (not isinstance(source, dict) or set(source) != {"type", "media_type", "data"}
                    or source["type"] != "base64" or not isinstance(source["media_type"], str)
                    or source["media_type"] not in {"image/png", "image/jpeg"}
                    or not isinstance(source["data"], str)):
                raise ValueError("Only Claude base64 PNG/JPEG image sources are supported")
            output.append({"type": "image", "source": dict(source)})
        else:
            raise ValueError("Unsupported Claude content block or extra fields; accepted types are text and base64 image")
    if sum(block["type"] == "text" for block in output) > 1:
        raise ValueError("This adapter accepts at most one text block; use named native text blocks for multiple questions")
    return output


class Client:
    def __init__(self, base_url="http://127.0.0.1:8000", api_key=None, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.api_key = os.environ.get("MMSO_API_KEY", "") if api_key is None else api_key
        self.timeout = timeout

    def _request(self, path, payload=None):
        headers = {"Accept": "application/json"}
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        if data is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = Request(self.base_url + path, data=data, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as error:
            try:
                body = json.loads(error.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                body = {"error": {"code": "http_error", "message": "Server returned a non-JSON error response"}}
            raise APIError(error.code, body.get("error", body)) from None

    def models(self):
        return self._request("/v1/models")

    def decide(self, *, input, questions, model=MODEL_ID, abstain_threshold=0.0):
        return self._request("/v1/decisions", {"model": model, "input": input, "questions": questions,
                                             "abstain_threshold": abstain_threshold})
