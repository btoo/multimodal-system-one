# V4 research attention adapters

These are the two fixed 512-update, rank-8 attention-adapter experiments. Existing publisher weights were frozen; new query/value projection updates were trained, saved independently and merged in memory for evaluation. The publisher checkpoints in the Modal cache were not overwritten.

Use each `adapter/config.json` for its base model revision, module names, scale, training duration and tensor hash. The loader is [backbone_adapters.py](../../mmso/backbone_adapters.py). The adapters and their calibration must be paired with the exact base revision.

These are noncommercial research artifacts because the experiment includes CC BY-NC 4.0 SLURP audio. Upstream base-model terms continue to apply. Qwen-based artifacts are **Built with Qwen**. [SLURP attribution](https://github.com/pswietojanski/slurp#license), [candidate inventory](../../evals/v4-candidates-v1.json), [complete study](../../reports/v4-selection-v1/README.md).
