# MiSO workflow comparison readiness

**MiSO v3 cannot currently enter the shared Jev text-workflow track.** All 12 original smoke cases are marked unsupported: the current model requires a raw image/audio pair, has no free-form state input, and uses 43 vocabulary tokens including special tokens. Its registered checkpoint was loaded and checked during the audit. No accuracy, latency or cost point was invented, and no external API calls were made.

A separate direct reproduction confirmed that a text-only request to the existing API returns `422 invalid_request`. Encoding an ordinary refund question also rejects words outside the learned vocabulary. This establishes a capability/contract gap, rather than a measured accuracy deficit relative to Jev.

The [case records](cases.jsonl) retain every input ID and request hash. [The result](result.json) records the checkpoint, unsupported reasons and source hashes. The fixtures are our own obvious routing cases, not TypeSafe's published workflow benchmark. [Comparison design and next model gate](../../docs/research/jev-comparison.md).
