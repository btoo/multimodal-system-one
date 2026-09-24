# First successful Modal L4 pilot

On September 24, 2026, the existing **668,097-parameter v3 model ran on an NVIDIA L4**, then completed 32 training updates and reproduced the final state exactly after restarting from update 16 in a second container. This is infrastructure qualification using already-exposed training examples. It does not establish a new accuracy result or promote new weights.

[Completed Modal app](https://modal.com/apps/btoo/main/ap-R2EIEU2rGMrlIxEtHpwAN9) · [Runner and reproduction guide](../../docs/modal.md)

| Measurement | Observed result |
|---|---:|
| Mac CPU versus CUDA decisions | 64/64 agree |
| Maximum probability difference, Mac CPU versus CUDA | 0.000100553 |
| Maximum probability difference, Linux CPU versus CUDA | 0.000100851 |
| Warm batch-1 inference, median / p95 | 3.64 / 4.72 ms |
| Warm batch-32 inference, median / p95 | 5.61 / 6.24 ms per batch |
| Warm batch-32 throughput | 5,694 decisions/second |
| Uninterrupted 32-update training time | 1.84 seconds |
| Median synchronized optimizer update | 20.07 ms |
| Reference / resume RPC wall time | 19.78 / 17.64 seconds |
| Fresh-container resume, weights and optimizer | Exact match |
| Fresh-container resume, sampler/RNG and losses | Exact match |
| Peak PyTorch allocated GPU memory, reference phase | 142.27 MiB |

## What was measured

The bundle contains 64 public training scenes, expanded to 512 questions, with eight recordings per keyword and generated 2×2 panels. The parity check uses the first 64 question requests. These examples were already in model training; 64/64 is agreement with another backend, **not task accuracy**. The production v3 checkpoint remains unchanged, SHA-256 `4284d30920615018b500716d406236d919f5e38e268fdc5490c6f559af612bfa`.

Inference uses 32 measured batches after five warmups at each batch size. Timers include prepared-tensor indexing, CPU/GPU transfer, forward computation, temperature/softmax, return transfer and synchronization. They exclude PNG/WAV decoding, model loading, startup, RPC and recording duration. These numbers cannot be substituted for the existing end-to-end HTTP benchmark or interpreted as a MacBook speedup.

Training starts from a copy of the existing v3 weights, uses AdamW at 0.0001 with batch size 16, and takes 32 updates. Positive gradients reached image, audio and text parameters. The very low training losses reflect revisiting training data with a trained checkpoint; they are not evidence of generalization improvement. Reference training time includes first-use overhead, gradient checks and synchronized optimizer updates; it excludes preprocessing, checkpoint writes and RPC.

The step-16 checkpoint includes model, optimizer, sampling generator, CPU RNG and CUDA RNG. A **single-use second container** loads that file and repeats updates 17–32. Final tensors, optimizer, sampler/RNG and all losses match the uninterrupted reference exactly. Downloaded checkpoint hashes and tensor comparisons were independently checked locally. This tests a clean container restart; an actual provider preemption was not injected, and the reference function commits its files after completing the phase.

The cloud image uses Python 3.12.10, PyTorch 2.14.0+cu130 and CUDA 13.0. Source revision is `010266476fffd3c5eb4e92ebdb6baace5308b2a9`. Direct numerical dependencies are pinned; the build also resolves transitive CUDA packages, so this is a captured environment rather than a fully hashed Linux dependency lock. The successful run reused the image built during setup.

## Two setup failures retained

1. [Attempt v1](../modal-l4-pilot-v1/failure.json): Modal's file-based CLI imported `modal_pilot`, but its directory was missing from `PYTHONPATH`. The app was stopped and the path corrected. Container initialization can be retried by Modal even when function retries are zero.
2. [Attempt v2](../modal-l4-pilot-v2/failure.json): the first backward pass rejected adaptive average pooling under CUDA strict determinism. The pilot now substitutes equivalent nonoverlapping average pooling for the fixed divisible spatial dimensions. CPU forward/gradient tests pass; GPU probabilities are checked against the original CPU implementation. Parameters and checkpoint keys are unchanged.

## Cost and shutdown

All four apps, including preflight and both failed attempts, were verified **stopped with zero containers**. Their combined wall lifetime was 147 seconds. Applying the L4 rate plus the pilot's maximum CPU/RAM allocation across that entire lifetime gives a rough **$0.038 compute proxy**, with image-build compute and retained Volume storage separate. It is not a final bill or an enforced spending cap.

The [billing snapshot](cost-and-shutdown.json) showed $0.00469021 in the ephemeral-app breakdown while aggregate metered and billed totals remained zero. Those figures had not reconciled, so actual final cost is left unset. Runtime and allocation estimates support the intended sub-$1 scale; we do not claim the run was free. [Modal pricing](https://modal.com/pricing).

## Evidence

- [Uploaded file hashes and data selection](bundle.json)
- [Reference phase, probabilities and timings](reference.json)
- [Fresh-container resumed phase](resume.json)
- [Independent Mac CPU comparison](local-parity.json)
- [Downloaded checkpoint verification](downloaded-checkpoint-verification.json)
- [Run status and configured limits](result.json)

```bash
uv run --locked python scripts/check_cloud_pilot.py
# Optional independent model execution and full checkpoint check:
mkdir -p .research/modal/downloaded
uv run --locked --group cloud modal volume get --profile btoo \
  mmso-pilot-checkpoints-v1 /modal-l4-pilot-v3 .research/modal/downloaded
uv run --locked python scripts/check_cloud_pilot.py --recompute-local \
  --checkpoint-dir .research/modal/downloaded/modal-l4-pilot-v3
```

The downloaded checkpoints remain under ignored `.research/`; full optimizer snapshots are not added to Git. The small evidence reports and source are versioned. All 83 local tests passed after the CUDA and pooling changes.
