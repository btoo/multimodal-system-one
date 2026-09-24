"""Registry tests use actual v2 weights under a test-only second ID on CPU."""
from dataclasses import replace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import torch

from mmso.api import MODEL_ID
from mmso.api.app import create_app
from mmso.api.registry import MODEL_REGISTRY, V2_REGISTRATION, V3_REGISTRATION, make_native_v1
from mmso.api.runtime import NativeRuntime, CHECKPOINT_SHA256, CONFIG_SHA256
from mmso.api.schema import DecisionResponse
from test_api import payload


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_production_registry_adds_confirmed_v3_and_retains_v2_default(self):
        self.assertEqual(list(MODEL_REGISTRY), [MODEL_ID, "mmso-joint-v3"])
        self.assertEqual(MODEL_ID, "mmso-joint-v2")
        self.assertEqual(V2_REGISTRATION.checkpoint_sha256, CHECKPOINT_SHA256)
        self.assertEqual(V2_REGISTRATION.config_sha256, CONFIG_SHA256)
        with TestClient(create_app(device="cpu", api_key="")) as client:
            self.assertEqual([m["id"] for m in client.get("/v1/models").json()["data"]], [MODEL_ID, "mmso-joint-v3"])
            body = payload();body.pop("model")
            response = client.post("/v1/decisions", json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["model"], MODEL_ID)
            body["model"] = "mmso-joint-v3"
            response = client.post("/v1/decisions", json=body)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["model"], "mmso-joint-v3")
            self.assertEqual(response.json()["checkpoint_sha256"], V3_REGISTRATION.checkpoint_sha256)
            self.assertNotEqual(response.json()["checkpoint_sha256"], CHECKPOINT_SHA256)
            self.assertEqual(client.get("/v1/models/mmso-joint-v3").json()["parameters"], 668097)

    def test_two_startup_cached_models_dispatch_and_keep_cards_isolated(self):
        loaded = []

        def factory(config, vocabulary_size):
            loaded.append(config["architecture"])
            return make_native_v1(config, vocabulary_size)

        alias = "test-only-v2-replica"
        metadata = {"fixture_only": True, "no_new_accuracy_result": True}
        registry = {
            MODEL_ID: replace(V2_REGISTRATION, factory=factory),
            alias: replace(V2_REGISTRATION, id=alias, factory=factory,
                scope="Test-only second runtime using unchanged v2 weights", measured_results=metadata,
                known_limitations=("Test fixture; not a production model registration",),
                calibration_scope="Same v2 calibration, test alias only"),
        }
        app = create_app(device="cpu", api_key="", registry=registry)
        # An application captures its registrations; editing the caller's mapping/metadata is not hot reload.
        registry.clear();metadata["fixture_only"] = False
        with TestClient(app) as client:
            first, second = app.state.runtimes[MODEL_ID], app.state.runtimes[alias]
            self.assertIs(app.state.runtime, first)
            self.assertIsNot(first, second)
            self.assertIsNot(first.model, second.model)
            self.assertNotEqual(next(first.model.parameters()).data_ptr(), next(second.model.parameters()).data_ptr())
            self.assertEqual(len(loaded), 2)
            self.assertEqual(client.get("/healthz").json()["loaded_models"], 2)
            cards = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
            self.assertEqual(set(cards), {MODEL_ID, alias})
            self.assertNotIn("measured_results", cards[MODEL_ID])
            self.assertTrue(cards[alias]["measured_results"]["fixture_only"])
            self.assertIn("Test-only", cards[alias]["scope"])
            self.assertEqual(client.get("/v1/models/" + alias).json(), cards[alias])
            with patch.object(first, "decide", wraps=first.decide) as default_call, patch.object(second, "decide", wraps=second.decide) as alias_call:
                body = payload()
                original = client.post("/v1/decisions", json=body)
                body["model"] = alias
                alternate = client.post("/v1/decisions", json=body)
                self.assertEqual(original.status_code, 200, original.text)
                self.assertEqual(alternate.status_code, 200, alternate.text)
                self.assertEqual(default_call.call_count, 1)
                self.assertEqual(alias_call.call_count, 1)
            first_result = DecisionResponse.model_validate(original.json())
            second_result = DecisionResponse.model_validate(alternate.json())
            self.assertEqual(first_result.model, MODEL_ID)
            self.assertEqual(second_result.model, alias)
            self.assertEqual(first_result.results, second_result.results)
            self.assertEqual(second_result.checkpoint_sha256, CHECKPOINT_SHA256)
            self.assertEqual(len(loaded), 2)
            self.assertIs(app.state.runtimes[MODEL_ID], first)
            self.assertIs(app.state.runtimes[alias], second)
            for unknown in ["missing", "mmso-joint-v3", "artifacts/joint-full-v2/model.safetensors"]:
                body["model"] = unknown
                self.assertEqual(client.post("/v1/decisions", json=body).status_code, 404)
            self.assertEqual(client.get("/v1/models/missing").status_code, 404)

    def test_bad_registry_identity_paths_and_fingerprints_fail_closed(self):
        for invalid in [
            {"wrong-key": V2_REGISTRATION, MODEL_ID: V2_REGISTRATION},
            {MODEL_ID: replace(V2_REGISTRATION, checkpoint="/tmp/model.safetensors")},
            {MODEL_ID: replace(V2_REGISTRATION, checkpoint="artifacts/../data/model.safetensors")},
            {MODEL_ID: replace(V2_REGISTRATION, config_sha256="missing")},
        ]:
            with self.assertRaises(ValueError):create_app(device="cpu", registry=invalid)
        bad = replace(V2_REGISTRATION, id="test-corrupt-config", config_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "configuration fingerprint"):
            with TestClient(create_app(device="cpu", registry={MODEL_ID: V2_REGISTRATION, bad.id: bad})):
                self.fail("Invalid registered artifacts must prevent readiness")
        bad = replace(V2_REGISTRATION, checkpoint_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "checkpoint fingerprint"):
            NativeRuntime("cpu", bad)


if __name__ == "__main__":unittest.main()
