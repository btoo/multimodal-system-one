# Versioned API correctness and observed speed

`mmso-joint-v3` is the development-nominated, fresh-confirmed primitive-supervision checkpoint. Its inference architecture remains 668,097 parameters. The local API loads v2 and v3 once and dispatches by explicit model ID; v2 remains the compatibility default. Select `model: "mmso-joint-v3"` to use the new weights.

The [accuracy study](../optimization-v1/README.md) passed its predeclared gates in two seeds. V3 scored **81.32% familiar / 66.99% known-composition accuracy**, compared with v2's **65.30% / 47.14%** on identical new voice/image examples. These are eight-keyword/generated-panel decisions, not arbitrary speech or real-screen performance.

## HTTP correctness

Each timing round sent 32 different generated panels and their real recordings, balanced across the eight words and the two slices. Each request returned four typed outputs from three distinct questions. Three varied cases per round were compared directly with native checkpoint inference, outside the timing samples.

CPU HTTP and native probabilities matched exactly. The MPS server differed from native CPU inference by at most **2.39×10⁻⁷**. Model and configuration hashes match the registered artifacts. The 78-test integrated suite passes, including actual CPU/MPS checks, registry isolation, media/schema limits and real HTTP clients.

## Latency varies with host conditions

| Round | Model / backend | Median | p95 | Range |
|---|---|---:|---:|---:|
| Initial varied-input round | v2 / CPU | 5.21 ms | 6.83 ms | 4.26–7.39 ms |
| Initial varied-input round | v3 / CPU | 5.16 ms | 5.50 ms | 4.59–6.04 ms |
| Sequential repeat | v2 / CPU | 15.88 ms | 35.95 ms | 9.30–41.62 ms |
| Sequential repeat | v3 / CPU | 32.55 ms | 145.17 ms | 10.39–290.43 ms |
| Additional backend check | v3 / MPS | 59.08 ms | 111.24 ms | 37.22–187.51 ms |

All rounds used the same 32 scene IDs, five warmup requests, one client request at a time, a fresh loopback connection per request, and two PyTorch CPU threads. The clock includes transport, validation, media decoding/preprocessing, neural inference/synchronization, result serialization, and client JSON decoding. It excludes recording time, client file/base64 assembly, and model startup. The recording must already be available.

The first timing sequence overlapped the integrated test suite. That scheduling overlap was identified before inspecting its numbers, and both rounds were retained. The repeat ran after those tests and training ended, but the wider laptop remained busy. A subsequent read-only check observed load averages around 27/21/18 and about 2.4 GB of swap use. These observations make contention a plausible explanation; they do not isolate its cause or prove which process affected which sample. No unrelated user workloads were stopped.

One MPS comparison was then run under the current host conditions. Its server output was redirected to a local log, unlike the terminal-attached CPU server. Backend, logging destination and timing order are therefore not fully controlled. These results do **not** establish a speedup, a stable sub-100 ms p95, or a production latency guarantee. The same small architecture can run in milliseconds, but the observed busy-host tails must remain visible.

The original [5.60 ms v2 p95](../api-v1/README.md) used one repeated fixture. It is a valid historical measurement with a narrower input sample, not the guaranteed latency for this broader task.

## Evidence and reproduction

- [Initial v2](../api-v2-varied-v1/report.json) and [v3](../api-v3-varied-v1/report.json), including separate overlap-context records.
- [Sequential v2](../api-v2-varied-v2/report.json) and [v3](../api-v3-varied-v2/report.json).
- [MPS v3](../api-v3-mps-v1/report.json) and [host-context record](../api-varied-context-v1.json).
- Each directory includes complete per-request responses, hashes, scene IDs, raw durations, and source fingerprints.

```bash
uv run mmso serve --device cpu --threads 2
# In another terminal, after preparing the recorded confirmation assets:
uv run python scripts/benchmark_model_api.py --model mmso-joint-v3 \
  --checkpoint artifacts/optimization-primitive-s24-v1/model.safetensors \
  --output-dir /tmp/mmso-new-timing --context 'Describe actual host conditions'
uv run python scripts/check_varied_api_results.py
```

A future latency comparison should randomize/interleave model/backend order and record host load before and during each round, preferably on a controlled machine. Accuracy improvements alone do not resolve the remaining latency measurement uncertainty.
