# Generated v4 follow-up results

Generated from the retained prediction files. Percentages below measure different task distributions; no pooled frontier score is computed.

## Full public audit

| Model | MMAU, 1,000 | MMStar, 1,500 | MMLU-Pro subset, 448 |
|---|---:|---:|---:|
| mini-base | 76.60% (766/1000) | 63.80% (957/1500) | 47.32% (212/448) |
| mini-adapted | 76.40% (764/1000) | 63.40% (951/1500) | 47.10% (211/448) |
| qwen3-base | 78.20% (782/1000) | 70.13% (1052/1500) | 55.36% (248/448) |
| gemma4-12b | 66.00% (660/1000) | 65.00% (975/1500) | 51.56% (231/448) |
| jev | — | — | 79.46% (356/448) |
| openai-sol-none | — | — | 79.24% (355/448) |

A dash means not run on that full benchmark. For OpenAI visual comparisons, use the matched 120 cases below.

## Paired MMStar subset

| Model | Same visual cases | Accuracy | 95% grouped interval |
|---|---:|---:|---|
| mini-base | 120 | 62.50% | 53.78%–71.67% |
| mini-adapted | 120 | 57.50% | 48.74%–66.67% |
| qwen3-base | 120 | 73.33% | 65.00%–81.51% |
| gemma4-12b | 120 | 66.67% | 57.98%–75.21% |
| openai-sol-none | 120 | 78.33% | 70.83%–85.83% |

## Text decision latency

The timing boundaries differ. Local H100 results exclude network/queue; provider references include network and hosted service work. These columns are not a claim of matched end-to-end speed.

| Model | MMLU-Pro accuracy | Median ms | p95 ms | Timing boundary |
|---|---:|---:|---:|---|
| mini-base | 47.32% | 28.89 | 35.45 | Local H100 pipeline, network excluded |
| mini-adapted | 47.10% | 39.19 | 45.84 | Local H100 pipeline, network excluded |
| qwen3-base | 55.36% | 73.86 | 123.35 | Local H100 pipeline, network excluded |
| gemma4-12b | 51.56% | 50.32 | 101.08 | Local H100 pipeline, network excluded |
| jev | 79.46% | 335.87 | 420.09 | API client including network |
| openai-sol-none | 79.24% | 1319.38 | 2053.06 | API client including network |

### Matched text API cost per correct decision

| Reference | All 448 requests, USD | USD per 1,000 correct decisions |
|---|---:|---:|
| jev | 0.010689 | 0.0300 |
| openai-sol-none | 0.279028 | 0.7860 |

Includes spending on wrong answers; uses returned token usage and standard list rates including cache writes. Earlier setup/failures remain in the overall spend ledger. MiSO has no measured hosted-serving cost per correct decision yet; training/study spend and busy-GPU proxies do not substitute for it.

## Reasoning-enabled OpenAI pilot

GPT-6 Astra medium has only 20 selected cases, one per text/visual category. This is a capability smoke check with a 1,024-output-token cap; inspect truncations before interpreting it.

- mmlu_pro: 13/14 correct; completed-answer coverage 100.00%; failures `{}`.
- mmstar: 6/6 correct; completed-answer coverage 100.00%; failures `{}`.

## Same-observation speed

| Questions | Independent ms | Default media reuse ms | Experimental packed ms | Default speedup | Packed speedup |
|---|---:|---:|---:|---:|---:|
| 1 | 122.47 | 121.69 | 122.80 | 1.01× | 1.00× |
| 4 | 488.82 | 213.10 | 127.25 | 2.29× | 3.84× |
| 16 | 1948.22 | 566.80 | 159.95 | 3.44× | 12.18× |

Each cell pools five repeats on each of five observations, with paths interleaved. Default reuse matched all 105 probability vectors in this diagnostic. Packed mode preserved 105/105 top choices but differed by up to 0.0364 in probability.

## Fresh screen diagnostic

On 117 unused screenshots: **51.28% region accuracy**, **0.00% actual point-inside-box accuracy** using the selected grid-cell center. This is not a dedicated GUI grounding head or an official full ScreenSpot-Pro score.

## Paired accuracy changes

Right model minus original MiniCPM base, on identical cases. Bootstrap resamples image/audio/question groups. These are exploratory public-audit comparisons.

| Benchmark | Right model | Cases | Difference | 95% paired interval |
|---|---|---:|---:|---|
| mmstar | mini-adapted | 1500 | -0.40 pp | -1.55 to +0.73 pp |
| mmstar | qwen3-base | 1500 | +6.33 pp | +3.80 to +8.86 pp |
| mmstar | gemma4-12b | 1500 | +1.20 pp | -1.42 to +3.77 pp |
| mmstar | openai-sol-none | 120 | +15.83 pp | +6.61 to +25.21 pp |
| mmau | mini-adapted | 1000 | -0.20 pp | -1.63 to +1.26 pp |
| mmau | qwen3-base | 1000 | +1.60 pp | -1.00 to +4.28 pp |
| mmau | gemma4-12b | 1000 | -10.60 pp | -13.63 to -7.43 pp |
| mmlu_pro | mini-adapted | 448 | -0.22 pp | -2.68 to +2.24 pp |
| mmlu_pro | qwen3-base | 448 | +8.04 pp | +3.57 to +12.47 pp |
| mmlu_pro | gemma4-12b | 448 | +4.24 pp | -0.45 to +9.13 pp |
| mmlu_pro | jev | 448 | +32.14 pp | +27.17 to +37.14 pp |
| mmlu_pro | openai-sol-none | 448 | +31.92 pp | +26.89 to +37.20 pp |
