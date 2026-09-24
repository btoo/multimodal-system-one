# Jev API integration smoke

The real `jev-1.13.0` API completed all 12 original fictional routing cases and matched all 12 expected final actions. Each request contained one Choice, one Noul and one Score question. This establishes that authentication, primitives, validation, policy execution and token accounting work. These obvious engineering fixtures are **not TypeSafe's published benchmark**, independent human evaluation, or evidence of general model quality.

| Measurement | Result |
|---|---:|
| Requests / completed | 12 / 12 |
| Expected final actions matched | 12 / 12 |
| Client-visible median / p95 | 365.42 / 507.71 ms |
| Input tokens across requests | 5,183 |
| Inference API cost at the checked published rate | $0.000217686 total |
| Mean inference cost per completed fixture | $0.0000181405 |
| Configured spending bound | $0.10 |
| Automatic retries | 0 |
| Eligible for a research frontier | No |

Cost is returned input-token usage multiplied by the documented $0.042/M rate; output tokens are uncharged under that rate. It is an inference API cost calculation, not training cost or a finalized account invoice. [TypeSafe model pricing](https://docs.typesafe.ai/models). Timing is from the MacBook client around each HTTP request and includes connection/network/server time. It is a different boundary from MiSO's warm GPU-kernel pilot.

All raw response distributions, provider confidences and usage are retained in [case records](cases.jsonl), alongside hashes of the original requests. [Aggregate results](result.json) preserve the exact initial runner revision. No credentials, account key identifiers or private customer records are included.

MiSO v3 currently marks the same 12 cases unsupported because it cannot consume free-form text state. Its [capability audit](../miso-jev-readiness-v1/README.md) has no invented zero-accuracy or timing result. We cannot place the two on a comparable quality frontier yet. The [comparison plan](../../docs/research/jev-comparison.md) identifies the language-model and benchmark-export work needed next.

```bash
uv run --locked python scripts/check_workflow_results.py
```
