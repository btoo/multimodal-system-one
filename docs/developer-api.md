# Local developer API

The MiSO API runs the published native checkpoint on your machine. Model cards include a MiSO display name; historical `mmso-joint-v2` and `mmso-joint-v3` IDs remain stable. It accepts image pixels, a short WAV recording, and question text, then returns named classification distributions, Boolean probabilities, numeric rubric scores, and ranked candidates in one response. It makes no calls to OpenAI, Claude, or Jev.

The API serves two explicit versions, both limited to a small vocabulary, eight spoken keywords, and generated 2×2 symbol panels. The newer **`mmso-joint-v3` scores 81.32% on familiar combinations and 66.99% on the known composition gap**, versus v2's 65.30%/47.14% on identical fresh examples. V2 remains the compatibility default. Arbitrary screenshots and speech can fit a file format without being understood. See the [current accuracy study](../reports/optimization-v1/README.md) and [historical v2 report](../reports/joint-v2/README.md).

## Run it

```bash
uv sync --locked
uv run mmso serve --device cpu
```

The default address is `http://127.0.0.1:8000`. Weights are verified and loaded once during startup. `--device mps` uses Apple Silicon acceleration; `--device auto` chooses it when available. `--threads 2`, `--port 8000`, and `--host 127.0.0.1` are the other defaults. One device inference runs at a time inside each process.

For the recorded public speech + generated panel example, prepare the data if it is not already present:

```bash
uv run mmso prepare speech_keywords
uv run mmso joint-prepare
uv run python examples/api_client.py
# Explicitly use the new confirmed weights:
uv run python examples/api_client.py --model mmso-joint-v3
```

The example returns all four output types from the same audio and image. It uses the same fixed scene as the [earlier joint demo](../examples/joint-demo-input.json), selected before its original evaluation. The example's presence score assigns `no → 0` and `yes → 1`, so its expected value is the estimated probability of presence. The [recorded HTTP response](../examples/api-response.json) predicts absence with probability 0.630, ranks red first for the opposite word with probability 0.887, and returns a presence score of 0.370. This is one illustrative scene, not a new accuracy evaluation.

That saved response is v2. V3 has distinct weights and temperature; its model card and response identify the requested version. For new Python calls, pass `model="mmso-joint-v3"` to `Client.decide`. The [varied-input API report](../reports/api-comparison-v3/README.md) verifies CPU/MPS parity and records all latency rounds, including busy-host tails. Millisecond measurements are not a stable latency guarantee.

To inspect a raw HTTP request:

```bash
uv run python examples/api_client.py --write-request /tmp/mmso-request.json
curl --fail-with-body http://127.0.0.1:8000/v1/decisions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/mmso-request.json
```

`/docs` exposes interactive OpenAPI documentation; `/openapi.json` contains the typed schemas. A [checked-in OpenAPI snapshot](openapi.json) is also available for client tooling. `GET /healthz` returns readiness after the checkpoint has loaded. `GET /v1/models` and `GET /v1/models/mmso-joint-v2` return its capabilities, full learned vocabulary, calibration definition, and pinned weight/configuration hashes.

Set the same `MMSO_API_KEY` environment variable in the server and client processes to enable bearer authentication. Model and decision endpoints then require `Authorization: Bearer …`; the Python client reads that variable automatically. Health and API documentation remain public. Loopback is the default; configure a key before intentionally exposing another interface. This repository does not supply hosted deployment, TLS, accounts, or usage billing.

## Request contract

```json
{
  "model": "mmso-joint-v2",
  "input": [
    {
      "type": "image",
      "source": {"type": "base64", "media_type": "image/png", "data": "<base64 PNG>"}
    },
    {
      "type": "audio",
      "source": {"type": "base64", "media_type": "audio/wav", "data": "<base64 PCM16 WAV>"}
    },
    {"type": "text", "id": "presence", "text": "is the spoken command on the screen"}
  ],
  "questions": {
    "color": {
      "type": "choice",
      "question": "what color marks the spoken command",
      "choices": [
        {"id": "r", "text": "red"},
        {"id": "g", "text": "green"},
        {"id": "b", "text": "blue"},
        {"id": "y", "text": "yellow"},
        {"id": "absent", "text": "not present"}
      ]
    },
    "present": {"type": "noul", "question_ref": "presence"},
    "presence_score": {
      "type": "score",
      "question_ref": "presence",
      "levels": [
        {"id": "absent", "text": "no", "value": 0},
        {"id": "present", "text": "yes", "value": 1}
      ]
    }
  },
  "abstain_threshold": 0.0
}
```

Replace the placeholder data with real base64; the example script does this. A media source can instead use `{"type":"data_url","url":"data:image/png;base64,…"}` or an `audio/wav` data URL. Remote URLs and server-side file paths are rejected.

Every request needs exactly one image and one audio block. Each named question has either direct `question` text or a `question_ref` referencing a text block. Every text block must be referenced. This keeps question text distinct from unsupported free-form context. Questions share encoded audio and image observations and are evaluated independently; other questions and their answers do not become context.

There is no chat history, system role, conversation state, transcript input, arbitrary response schema, tool call, video, streaming, or text generation endpoint. Unknown fields and unsupported blocks produce explicit errors.

## Output semantics

The response has `object: "decision.response"`, a request ID, the model/checkpoint identity, and `results` keyed by the supplied question names. Its OpenAPI schema specifies a discriminated result type for each answer.

| Question type | Request | Result |
|---|---|---|
| `choice` | `choices: [{id, text}, …]` | Probability distribution, most likely `prediction`, thresholded `decision`, complete ranking |
| `ranking` | Same candidate contract as Choice | Candidates sorted by descending probability, with ranks and the same distribution/abstention fields |
| `noul` | Yes/no question | `probability_true = P(yes)`, `value: true/false/null`, and the complete yes/no distribution |
| `score` | `levels: [{id, text, value}, …]` | Distribution over rubric levels, `expected_value = Σ p(level) × value`, and thresholded numeric `value` |

These are projections of **one learned candidate scorer**. Noul supplies candidates `true → yes` and `false → no`. Score uses explicit caller-supplied numeric values; this extends Jev's publicly documented index-based Score convention. It is not a separately trained regression model. Values must be distinct, finite, and between −1,000,000,000 and +1,000,000,000. The current model cannot learn the meaning of a new rubric from its numeric values; descriptions still have to use its learned vocabulary and supported concepts.

For every result:

- `probabilities` sum to approximately one over the supplied candidate set. These are mutually exclusive categorical probabilities, not independent multilabel relevance scores. Changing the set changes normalization and may change confidence.
- `confidence` is the maximum probability after the published temperature scaling. It is not a separate estimate of epistemic uncertainty or out-of-distribution detection. The original calibration does not establish calibration for arbitrary new rubrics or candidate subsets.
- `prediction` is the unique most likely candidate ID, even when a confidence threshold causes abstention. A top tie makes `prediction` null.
- `decision` is null if confidence is below `abstain_threshold` or top candidates tie within an absolute probability tolerance of `1e-7`. `abstained` reports that state. Noul and Score also return `value: null` on abstention while preserving the full distribution and expectation.
- Ranking ties share ranks. IDs break display-order ties only; IDs never enter the neural model. A candidate ID may be arbitrary text, while candidate descriptions are learned text inputs.

`input_summary` records decoded dimensions, audio sample rate/duration, and preprocessing. `semantics` records the inference device and output interpretations. The response is structured JSON because the server assembles typed decisions; the model is not generating JSON tokens.

## Accepted media and text

| Input | Current limit and treatment |
|---|---|
| Whole JSON body | 4 MiB, including base64; enforced for streamed/chunked request bodies too |
| Image | PNG/JPEG, ≤2 MiB decoded, ≤4,194,304 pixels and ≤4096 pixels per edge; converted to RGB and resized to 128×128; animation rejected |
| Audio | Mono uncompressed PCM16 WAV, >0 and ≤1 second, ≤128 KiB; sample rates 8,000 / 16,000 / 22,050 / 24,000 / 44,100 / 48,000 Hz |
| Audio preprocessing | Resampled to 16 kHz, shorter clips zero-padded to one second, then log-mel features; longer clips rejected rather than truncated |
| Questions | 1–32 per request; each ≤24 learned word tokens |
| Candidates | 2–32 per question; each description ≤8 learned word tokens; unique IDs and unique tokenized descriptions |
| Text | ASCII letters, whitespace, and punctuation; lowercase word tokenization ignores punctuation; unsupported numeric/non-ASCII text and unknown vocabulary are rejected |

Those transport limits do not expand the training distribution. The visual encoder uses fixed 2×2 quadrants, and the audio model learned `down`, `go`, `left`, `no`, `right`, `stop`, `up`, and `yes`. The model card lists all 41 usable words (43 embedding entries including two special tokens). There is no silent truncation of question/candidate text or audio at this API boundary.

## Python client and provider content adapters

The small synchronous client uses only Python's standard library and does not depend on any provider SDK:

```python
from mmso.api.client import Client, media_block

client = Client("http://127.0.0.1:8000")
result = client.decide(
    model="mmso-joint-v2",
    input=[
        media_block("panel.png", "image", "image/png"),
        media_block("keyword.wav", "audio", "audio/wav"),
    ],
    questions={"present": {"type": "noul", "question": "is the spoken command on the screen"}},
)
print(result["results"]["present"]["probability_true"])
```

`media_block` reads a file on the **client**, then sends bytes. The server never receives the local path. API failures raise `APIError` with `status` and the structured `error` payload.

Optional adapters make common inline content shapes reusable:

```python
from mmso.api.client import from_openai_content, from_claude_content

blocks = from_openai_content([
    {"type": "input_text", "text": "is the spoken command on the screen"},
    {"type": "input_image", "image_url": "data:image/png;base64," + image_base64},
    {"type": "input_audio", "input_audio": {"data": wav_base64, "format": "wav"}},
], text_id="prompt")
result = client.decide(input=blocks, questions={
    "present": {"type": "noul", "question_ref": "prompt"},
})

# Claude image/text content; add our native audio block separately.
blocks = from_claude_content([
    {"type": "text", "text": "is the spoken command on the screen"},
    {"type": "image", "source": {
        "type": "base64", "media_type": "image/png", "data": image_base64,
    }},
], text_id="prompt") + [media_block("keyword.wav", "audio", "audio/wav")]
```

The OpenAI helper combines the documented **Responses image/text shape** and **Chat Completions audio shape** as a convenience for this API. It does not assert that a single OpenAI endpoint accepts that combined request. Both adapters reject unknown fields, remote URLs, role/message wrappers, unsupported media features, and multiple text blocks. Use native named text blocks for multiple questions. These helpers are not full provider request adapters or OpenAI/Claude SDK compatibility layers.

Contract references, checked September 24, 2026: [OpenAI images and vision](https://developers.openai.com/api/docs/guides/images-vision), [OpenAI audio input](https://developers.openai.com/api/docs/guides/audio-chat-completions), [Claude vision](https://platform.claude.com/docs/en/build-with-claude/vision), and [TypeSafe System One API](https://docs.typesafe.ai/api). Jev's public API uses a text state and typed questions; this API uses native media observations and explicit candidate/rubric descriptions.

## Errors and verification

Errors use `{"error":{"code":"…","message":"…"}}`, with field details for schema errors. Invalid schemas/media/vocabulary return 422, an oversized request returns 413, unknown model/routes return 404, unsupported methods return 405, and missing/incorrect configured credentials return 401. Validation responses omit the submitted base64 media payload. No model prediction is returned when a request is rejected.

```bash
uv run --locked python -m unittest discover -s tests -p 'test_api.py' -v
```

The tests use the actual published CPU checkpoint and a deterministic synthetic WAV, compare HTTP probabilities against the existing native inference implementation, and verify candidate-order/ID invariance, question isolation, scoring/ranking/abstention, media/schema bounds, bearer authentication, OpenAPI result schemas, and provider adapters. A subprocess test launches the real CLI server and uses the Python client over a loopback TCP connection. These establish transport/inference correctness, not new model-accuracy measurements. The recorded public-speech example and separate HTTP benchmark supplement those checks.

The model's weights and configuration are both fingerprinted at startup, so changing its vocabulary, temperature, or weights requires explicitly updating its versioned registration. Training and the API do not share a mutable in-process model.

## Registering another confirmed native model

The static registry in [`mmso/api/registry.py`](../mmso/api/registry.py) contains **`mmso-joint-v2` and `mmso-joint-v3`**. V2 remains the request and Python-client default; explicitly select v3 for the improved weights. `GET /v1/models` lists the registrations actually loaded by that server; `POST /v1/decisions` dispatches using the explicit `model` field and returns that same identity. There are no speculative production aliases, automatic newest-checkpoint discovery, or HTTP registration/checkpoint-path inputs.

A future confirmed checkpoint needs a distinct `ModelRegistration` with a repository-relative `.safetensors` path under `artifacts/`, weight and configuration SHA256 hashes, the expected architecture name, and a local `factory(config, vocabulary_size)`. The factory must provide the existing `encode_observations` / `decide` interface and compatible media/text behavior; changing those input semantics also requires changing and validating the API contract. Each registration can supply its own scope, calibration statement, limitations, and optional `measured_results` metadata. A new registration does not inherit v2's measured accuracy claims.

The app snapshots this registry and eagerly loads every registered runtime at startup. Each runtime owns its model, vocabulary, temperature, and card; a shared inference lock serializes work on the selected device. Any fingerprint or architecture mismatch prevents readiness. Adding a registration requires a restart; it does not change v2's default or fingerprint. The typed Choice/Noul/Score/Ranking response schema stays the same. For library/test callers, `create_app(..., registry=...)` supplies a static registry snapshot, and `app.state.runtime` remains an alias for the historical default runtime alongside `app.state.runtimes`.

Registry tests load two isolated CPU runtimes from the same verified weights under a temporary test-only second ID, prove dispatch and cache reuse, check card isolation, and reject unknown models and corrupt registrations. The second ID is never added to the production registry.

```bash
uv run --locked python -m unittest discover -s tests -p 'test_api*.py' -v
```
