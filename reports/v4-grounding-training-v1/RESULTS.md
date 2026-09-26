# Grounding and broader-training results

Development and regression cases below are exposed; use the separate confirmation table for the fixed-checkpoint follow-up. Local and hosted latency boundaries differ.

| Task | Cases | MiniCPM base | Qwen, 0–1 prompt | MiSO mixed adapter | GPT-6 Sol none | Qwen, 0–1000 prompt |
|---|---:|---:|---:|---:|---:|---:|
| screenspot_pro | 117 | 3.42% (4/117) | 0.85% (1/117) | 21.37% (25/117) | 66.67% (78/117) | 11.97% (14/117) |
| omniact | 96 | 11.46% (11/96) | 0.00% (0/96) | 68.75% (66/96) | — | 37.50% (36/96) |
| chartqa | 128 | 84.38% (108/128) | 75.78% (97/128) | 85.16% (109/128) | 39.06% (50/128) | — |
| boolq | 128 | 91.41% (117/128) | 88.28% (113/128) | 91.41% (117/128) | 89.84% (115/128) | — |
| mmau | 128 | 73.44% (94/128) | 71.09% (91/128) | 73.44% (94/128) | — | — |
| mmstar | 120 | 59.17% (71/120) | 72.50% (87/120) | 62.50% (75/120) | — | — |
| mmlu_pro | 112 | 48.21% (54/112) | 50.89% (57/112) | 49.11% (55/112) | — | — |

ChartQA uses the frozen strict relaxed-numeric/exact-text scorer. The separately reported unit/format diagnostic is post hoc and is not a replacement leaderboard metric.

## Coordinate contract and latency

| Model | Point dataset | In-box / cases | Valid normalized JSON | Median ms | p95 ms |
|---|---|---:|---:|---:|---:|
| MiniCPM base | screenspot_pro | 4/117 | 107/117 | 1012.3 | 1507.8 |
| MiniCPM base | omniact | 11/96 | 68/96 | 804.5 | 1212.1 |
| Qwen, 0–1 prompt | screenspot_pro | 1/117 | 2/117 | 1387.8 | 3473.7 |
| Qwen, 0–1 prompt | omniact | 0/96 | 1/96 | 1195.2 | 2896.3 |
| MiSO mixed adapter | screenspot_pro | 25/117 | 117/117 | 931.8 | 1331.3 |
| MiSO mixed adapter | omniact | 66/96 | 96/96 | 788.7 | 1000.9 |
| GPT-6 Sol none | screenspot_pro | 78/117 | 117/117 | 1654.8 | 2339.4 |
| Qwen, 0–1000 prompt | screenspot_pro | 14/117 | 21/117 | 1636.6 | 4478.5 |
| Qwen, 0–1000 prompt | omniact | 36/96 | 42/96 | 1561.0 | 4071.3 |

H100 timings include local media decode, preprocessing, full model generation and CPU output. GPT-6 Sol timings also include network and hosted service work. No matched hosted-serving speed claim is made.

## Fresh confirmation

| Dataset | Cases | Unchanged base | Fixed mixed adapter | Paired delta, 95% group interval |
|---|---:|---:|---:|---|
| screenspot_v2 | 96 | 28.12% | 71.88% | +43.75 pp [+33.33, +54.17] |
| omniact | 96 | 6.25% | 66.67% | +60.42 pp [+42.71, +77.08] |
| chartqa | 128 | 85.94% | 86.72% | +0.78 pp [+0.00, +2.34] |
| boolq | 128 | 91.41% | 92.97% | +1.56 pp [+0.00, +3.91] |

### Paired image/audio regression

On the same exposed controlled image/audio cases: base 54/64 (84.38%), adapter 60/64 (93.75%). This is regression coverage, not new real-world speech/screen confirmation.

### Coordinate latency on confirmation

| Task | Base median ms | Adapter median ms |
|---|---:|---:|
| screenspot_v2 | 643.3 | 733.2 |
| omniact | 563.1 | 682.2 |

These sequential warm measurements do not establish a kernel speedup. Coordinate generation remains autoregressive; improved success did not make every request faster.

## Explicit 0–1000 coordinate control

Same original target boxes and deterministic output conversion; no guessed units. This exploratory prompt-format control was frozen before confirmation calls.

| Partition | Dataset | Correct / cases | Valid output | Accuracy |
|---|---|---:|---:|---:|
| development | omniact | 36/96 | 42/96 | 37.50% |
| development | screenspot_pro | 14/117 | 21/117 | 11.97% |
| confirmation | screenspot_v2 | 36/96 | 37/96 | 37.50% |
| confirmation | omniact | 32/96 | 38/96 | 33.33% |

## Image-content dependence diagnostic

The blank-image condition removes the visible targets. Its score is agreement with the original boxes, not grounding accuracy on a valid visible-target task.

| Model | Clean-image successes | Blank-image reference agreements |
|---|---:|---:|
| base | 27/96 | 1/96 |
| adapted | 69/96 | 2/96 |

This tests image reliance, not absent-target detection or calibrated abstention.
