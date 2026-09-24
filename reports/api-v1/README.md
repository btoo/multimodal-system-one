# Developer API verification

The local HTTP API serves the previously evaluated `mmso-joint-v2` checkpoint with Choice, Noul, Score, and Ranking results. Its [developer guide](../../docs/developer-api.md) documents the exact contract and [OpenAPI schema](../../docs/openapi.json).

## Actual HTTP evidence

The fixed public-recording demo produced all four result types through the real local server. Its probabilities matched the saved native prediction within **1.79×10⁻⁷**, including the numeric expectation for the presence score. A threshold of 1 forced abstention while retaining probabilities. The [verification record](demo-verification.json) preserves both responses and the reference/checkpoint hashes.

This is one previously fixed scene and three distinct questions, exposed through four typed views. It proves transport and checkpoint parity, not an additional accuracy benchmark. The separate API suite covers actual CPU inference, malformed and oversized media, schema and vocabulary bounds, authentication, candidate ordering, question isolation, provider content adapters, and a subprocess server/client connection.

## Warm HTTP latency

Measured September 24, 2026 on the Apple M4 Pro MacBook, after the model-training runs finished. The server used **CPU inference with two PyTorch threads**, one loaded checkpoint, and one request at a time. The client reused the same assembled audio/image fixture, opened a new loopback HTTP connection for each request, and requested four typed answers.

| Measurement | Result |
|---|---:|
| Warmup requests, excluded | 5 |
| Measured requests | 32 |
| Median | **4.81 ms** |
| p95 | **5.60 ms** |
| Mean | 4.84 ms |
| Range | 4.11–6.03 ms |

The clock includes request transmission, server validation, media decoding, neural processing, response serialization/transmission, and client JSON decoding. Audio capture, client base64 assembly, server startup, and checkpoint loading are excluded. The recording must already be available. This repeated-fixture measurement is not a streaming latency, concurrency, capacity, or independent-input benchmark.

The older 9.84 ms neural-pipeline figure used a different device and request shape. These are different measurements, so their difference is not an estimate of API overhead or an optimization gain.

[Raw samples and source fingerprints](http-benchmark.json) make the statistics reproducible. The benchmark captured the committed Python sources and dependency lock; the server used the pinned weight and calibration configuration hashes recorded in its model card.

## Reproduce

From the repository checkout, prepare the example data and start the server:

```bash
uv sync --locked
uv run mmso prepare speech_keywords
uv run mmso joint-prepare
uv run mmso serve --device cpu --threads 2
```

In another terminal, create a request and write new evidence to fresh paths:

```bash
uv run python examples/api_client.py --write-request /tmp/mmso-request.json
uv run python scripts/verify_api_demo.py /tmp/mmso-request.json --output /tmp/mmso-parity.json
uv run python scripts/benchmark_api.py /tmp/mmso-request.json --output /tmp/mmso-http-benchmark.json
uv run python scripts/check_api_results.py
```

The last command independently verifies the checked-in schema, historical parity, timing statistics, and source/configuration/checkpoint hashes without requiring raw media or a running server. It does not overwrite the original measurement.
