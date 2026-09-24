"""HTTP contract tests use the actual published CPU checkpoint, never an oracle."""
import base64
import copy
from io import BytesIO
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import URLError
from urllib.request import urlopen
import wave

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import torch

from mmso.api.app import create_app, MAX_REQUEST_BYTES
from mmso.api.client import APIError, Client, from_claude_content, from_openai_content
from mmso.api.runtime import CONFIG_SHA256, CHECKPOINT_SHA256, format_result
from mmso.api.schema import DecisionRequest, DecisionResponse
from mmso.artifacts import ROOT
from mmso.joint_model import predict_joint


def wav_bytes(seconds=1, rate=16000, channels=1, width=2):
    frames = int(seconds * rate)
    values = (8000 * np.sin(2 * np.pi * 230 * np.arange(frames) / rate)).astype("<i2")
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(width)
        audio.setframerate(rate)
        audio.writeframes(values.tobytes() if channels == 1 else np.repeat(values[:, None], channels, axis=1).tobytes())
    return output.getvalue()


def block(kind, data, media_type):
    return {"type": kind, "source": {"type": "base64", "media_type": media_type,
            "data": base64.b64encode(data).decode()}}


def payload():
    return {"model": "mmso-joint-v2",
        "input": [block("image", (ROOT / "examples/joint-panel.png").read_bytes(), "image/png"),
                  block("audio", wav_bytes(), "audio/wav"),
                  {"type": "text", "id": "presence", "text": "is the spoken command on the screen"}],
        "questions": {
            "color": {"type": "choice", "question": "what color marks the spoken command",
                      "choices": [{"id": c, "text": c} for c in ["red", "green", "blue", "yellow", "not present"]]},
            "ranked": {"type": "ranking", "question": "where is the spoken command on the screen",
                       "choices": [{"id": str(i), "text": c} for i, c in enumerate(["top left", "top right", "bottom left", "bottom right", "not present"])]},
            "present": {"type": "noul", "question_ref": "presence"},
            "score": {"type": "score", "question_ref": "presence", "levels": [
                {"id": "no", "text": "no", "value": -2}, {"id": "yes", "text": "yes", "value": 7}]}},
        "abstain_threshold": 0.0}


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.client = TestClient(create_app(device="cpu", api_key=""))
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def post(self, body):
        return self.client.post("/v1/decisions", json=body)

    def test_real_checkpoint_http_predictions_match_existing_inference(self):
        body = payload()
        response = self.post(body)
        self.assertEqual(response.status_code, 200, response.text)
        actual = DecisionResponse.model_validate(response.json())
        request = DecisionRequest.model_validate(body)
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "audio.wav"
            audio.write_bytes(wav_bytes())
            expected = predict_joint(ROOT / "artifacts/joint-full-v2/model.safetensors",
                ROOT / "examples/joint-panel.png", audio, request.neural_requests(), device="cpu")
        self.assertEqual(actual.checkpoint_sha256, CHECKPOINT_SHA256)
        for result, reference in zip(actual.results.values(), expected["answers"]):
            self.assertEqual(result.prediction, reference["prediction"])
            for key, probability in reference["probabilities"].items():
                self.assertAlmostEqual(result.probabilities[key], probability, places=6)
            self.assertAlmostEqual(sum(result.probabilities.values()), 1, places=6)
            self.assertEqual([entry.probability for entry in result.ranking],
                             sorted(result.probabilities.values(), reverse=True))
        truth = actual.results["present"].probability_true
        self.assertAlmostEqual(actual.results["score"].expected_value, -2 * (1 - truth) + 7 * truth, places=5)
        self.assertEqual(actual.results["score"].range, [-2, 7])
        self.assertEqual(actual.input_summary["audio"]["duration_seconds"], 1.0)

    def test_one_runtime_is_reused_and_questions_are_isolated(self):
        runtime = self.client.app.state.runtime
        body = payload()
        together = self.post(body).json()["results"]["color"]
        body["questions"] = {"color": body["questions"]["color"]}
        body["input"] = body["input"][:2]
        alone = self.post(body).json()["results"]["color"]
        for key, probability in together["probabilities"].items():
            self.assertAlmostEqual(probability, alone["probabilities"][key], places=5)
        self.assertIs(runtime, self.client.app.state.runtime)

    def test_candidate_permutation_and_ids_do_not_change_semantics(self):
        body = payload()
        original = self.post(body).json()["results"]["color"]["probabilities"]
        body["questions"]["color"]["choices"].reverse()
        for candidate in body["questions"]["color"]["choices"]:
            candidate["id"] = "opaque:" + candidate["id"]
        actual = self.post(body).json()["results"]["color"]["probabilities"]
        for key, probability in original.items():
            self.assertAlmostEqual(probability, actual["opaque:" + key], places=6)

    def test_abstention_preserves_distribution_and_nulls_typed_decisions(self):
        body = payload()
        body["abstain_threshold"] = 1.0
        results = self.post(body).json()["results"]
        for result in results.values():
            self.assertTrue(result["abstained"])
            self.assertIsNone(result["decision"])
        self.assertIsNone(results["present"]["value"])
        self.assertIsNone(results["score"]["value"])
        self.assertIsInstance(results["score"]["expected_value"], float)

    def test_tied_candidates_abstain_and_share_rank(self):
        question = DecisionRequest.model_validate(payload()).questions["score"]
        result = format_result(question, [{"id": "no"}, {"id": "yes"}], [.5, .5], 0)
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["value"])
        self.assertEqual(result["ties"], ["no", "yes"])
        self.assertEqual([r["rank"] for r in result["ranking"]], [1, 1])

    def test_data_urls_jpeg_and_short_resampled_audio(self):
        body = payload()
        image = BytesIO()
        with Image.open(ROOT / "examples/joint-panel.png") as original:
            original.convert("RGB").save(image, format="JPEG")
        body["input"][0]["source"] = {"type": "data_url", "url": "data:image/jpeg;base64," + base64.b64encode(image.getvalue()).decode()}
        body["input"][1] = block("audio", wav_bytes(seconds=.5, rate=44100), "audio/wav")
        response = self.post(body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["input_summary"]["audio"]["duration_seconds"], .5)

    def test_missing_duplicate_unknown_and_unused_blocks_are_rejected(self):
        cases = []
        body = payload();body["input"].pop(0);cases.append(body)
        body = payload();body["input"].append(copy.deepcopy(body["input"][0]));cases.append(body)
        body = payload();body["input"][0]["type"] = "video";cases.append(body)
        body = payload();body["input"].append({"type": "text", "id": "context", "text": "red"});cases.append(body)
        body = payload();body["input"].append(copy.deepcopy(body["input"][2]));cases.append(body)
        body = payload();body["questions"]["present"]["question_ref"] = "missing";cases.append(body)
        body = payload();body["questions"]["present"]["question"] = "yes";cases.append(body)
        for body in cases:
            with self.subTest(body_kind=str(body["input"][-1].get("type"))):
                self.assertEqual(self.post(body).status_code, 422)

    def test_chat_roles_context_unknown_fields_and_bad_model_are_explicit(self):
        for field in ["messages", "system", "context", "stream", "response_format"]:
            body = payload();body[field] = "unsupported"
            response = self.post(body)
            self.assertEqual(response.status_code, 422)
            self.assertNotIn(body["input"][0]["source"]["data"], response.text)
        body = payload();body["model"] = "jev"
        self.assertEqual(self.post(body).status_code, 404)

    def test_vocabulary_token_budgets_and_lexical_loss_are_explicit(self):
        for question in ["describe the browser", "red " * 25, "", "what color 123", "what color 🐈", "réd", "!?"]:
            body = payload();body["questions"]["color"]["question"] = question
            self.assertEqual(self.post(body).status_code, 422, question)
        body = payload();body["questions"]["color"]["choices"][0]["text"] = "red " * 9
        self.assertEqual(self.post(body).status_code, 422)

    def test_duplicate_candidates_or_meanings_and_bad_score_are_rejected(self):
        for field, value in [("id", "green"), ("text", "GREEN!")]:
            body = payload();body["questions"]["color"]["choices"][0][field] = value
            self.assertEqual(self.post(body).status_code, 422)
        for value in [float("nan"), float("inf"), 1e300, -2]:
            body = payload();body["questions"]["score"]["levels"][1]["value"] = value
            response = self.client.post("/v1/decisions", content=json.dumps(body), headers={"Content-Type": "application/json"})
            self.assertEqual(response.status_code, 422)

    def test_bad_sources_audio_bounds_and_header_deception_are_rejected(self):
        cases = []
        for url in ["https://example.com/a.png", "file:///etc/passwd", "/tmp/image.png", "data:image/png;base64,!"]:
            body = payload();body["input"][0]["source"] = {"type": "data_url", "url": url};cases.append(body)
        body = payload();body["input"][0]["source"]["data"] = "!";cases.append(body)
        body = payload();body["input"][0]["source"]["media_type"] = "image/jpeg";cases.append(body)
        body = payload();body["input"][0]["source"]["path"] = "/tmp/image.png";cases.append(body)
        for audio in [wav_bytes(1.0001), wav_bytes(channels=2), wav_bytes(rate=16001), wav_bytes(0), wav_bytes()[:-50], b"not wav"]:
            body = payload();body["input"][1] = block("audio", audio, "audio/wav");cases.append(body)
        for body in cases:
            self.assertEqual(self.post(body).status_code, 422)

    def test_large_image_and_http_body_are_rejected_before_inference(self):
        image = BytesIO();Image.new("RGB", (2049, 2048)).save(image, format="PNG")
        body = payload();body["input"][0] = block("image", image.getvalue(), "image/png")
        self.assertEqual(self.post(body).status_code, 422)
        response = self.client.post("/v1/decisions", content=b"x" * (MAX_REQUEST_BYTES + 1))
        self.assertEqual(response.status_code, 413)

    def test_model_capabilities_and_openapi_have_concrete_typed_schemas(self):
        card = self.client.get("/v1/models").json()["data"][0]
        self.assertEqual(card["config_sha256"], CONFIG_SHA256)
        self.assertEqual(card["parameters"], 668097)
        self.assertEqual(card["capabilities"]["audio"]["max_seconds"], 1.0)
        self.assertFalse(card["capabilities"]["generation"])
        spec = self.client.get("/openapi.json").json()
        results = spec["components"]["schemas"]["DecisionResponse"]["properties"]["results"]["additionalProperties"]
        self.assertEqual(results["discriminator"]["propertyName"], "type")
        self.assertEqual(len(results["oneOf"]), 4)
        for path in ["/missing", "/v1/models/missing"]:
            self.assertEqual(self.client.get(path).status_code, 404)
            self.assertIn("error", self.client.get(path).json())
        self.assertEqual(self.client.get("/v1/decisions").status_code, 405)

    def test_optional_bearer_auth(self):
        with TestClient(create_app(device="cpu", api_key="test-local-key")) as client:
            self.assertEqual(client.get("/healthz").status_code, 200)
            self.assertEqual(client.get("/v1/models").status_code, 401)
            self.assertEqual(client.post("/v1/decisions", json=payload()).status_code, 401)
            self.assertEqual(client.get("/v1/models", headers={"Authorization": "Bearer wrong"}).status_code, 401)
            self.assertEqual(client.get("/v1/models", headers=[(b"Authorization", b"Bearer caf\xe9")]).status_code, 401)
            self.assertEqual(client.get("/v1/models", headers={"Authorization": "Bearer test-local-key"}).status_code, 200)


class AdapterTests(unittest.TestCase):
    def test_supported_provider_shapes_map_to_executable_request(self):
        body = payload()
        image, audio = body["input"][:2]
        common = [{"type": "input_image", "image_url": "data:image/png;base64," + image["source"]["data"]},
                  {"type": "input_audio", "input_audio": {"data": audio["source"]["data"], "format": "wav"}},
                  {"type": "input_text", "text": "is the spoken command on the screen"}]
        converted = from_openai_content(common, text_id="presence")
        body["input"] = converted
        self.assertEqual(len(DecisionRequest.model_validate(body).neural_requests()), 4)
        converted = from_claude_content([{"type": "image", "source": image["source"]},
                    {"type": "text", "text": "is the spoken command on the screen"}], text_id="presence")
        body["input"] = converted + [audio]
        self.assertEqual(len(DecisionRequest.model_validate(body).neural_requests()), 4)

    def test_adapters_reject_unsupported_provider_features(self):
        for content in [
            [{"role": "user", "content": "hello"}],
            [{"type": "input_image", "image_url": "https://example.com/image.png"}],
            [{"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "high"}],
            [{"type": "input_audio", "input_audio": {"data": "AA==", "format": "mp3"}}],
            [{"type": "input_text", "text": "yes"}, {"type": "input_text", "text": "no"}],
            {"messages": []},
        ]:
            with self.assertRaises(ValueError):from_openai_content(content)
        with self.assertRaises(ValueError):
            from_claude_content([{"type": "image", "source": {"type": "url", "url": "https://example.com"}}])


class LiveHTTPTests(unittest.TestCase):
    def test_serve_cli_and_python_client_over_real_loopback_http(self):
        with socket.socket() as bound:
            bound.bind(("127.0.0.1", 0));port = bound.getsockname()[1]
        environment = dict(os.environ, MMSO_API_KEY="local-integration-test")
        process = subprocess.Popen([sys.executable, "-m", "mmso", "serve", "--device", "cpu", "--port", str(port)],
                                   cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        base_url = f"http://127.0.0.1:{port}"
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail("API exited during startup: " + process.communicate()[0])
                try:
                    with urlopen(base_url + "/healthz", timeout=.2) as response:
                        if json.load(response)["status"] == "ready":break
                except (URLError, TimeoutError):
                    time.sleep(.05)
            else:
                self.fail("API did not become ready within 30 seconds")
            with self.assertRaises(APIError) as rejected:
                Client(base_url, api_key="wrong").models()
            self.assertEqual(rejected.exception.status, 401)
            client = Client(base_url, api_key="local-integration-test")
            self.assertEqual(client.models()["data"][0]["checkpoint_sha256"], CHECKPOINT_SHA256)
            body = payload()
            actual = client.decide(input=body["input"], questions=body["questions"])
            self.assertEqual(set(actual["results"]), set(body["questions"]))
            DecisionResponse.model_validate(actual)
            # No Content-Length: exercise the actual ASGI streamed-body boundary.
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            try:
                encoded = json.dumps(body).encode()
                headers = {"Content-Type": "application/json", "Authorization": "Bearer local-integration-test"}
                connection.request("POST", "/v1/decisions", body=[encoded[:100], encoded[100:]], headers=headers, encode_chunked=True)
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["model"], "mmso-joint-v2")
                connection.request("POST", "/v1/decisions", body=[b"x" * (MAX_REQUEST_BYTES // 2), b"x" * (MAX_REQUEST_BYTES // 2 + 1)],
                                   headers=headers, encode_chunked=True)
                response = connection.getresponse()
                self.assertEqual(response.status, 413)
                response.read()
            finally:
                connection.close()
        finally:
            process.terminate()
            try:process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.communicate()


if __name__ == "__main__":
    unittest.main()
