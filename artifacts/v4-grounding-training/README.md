# Grounding and mixed-task adapters

These experiments start from a pinned native audio/image/text backbone and train
language-attention LoRA factors on separate GUI, chart, passage-decision and
audio/joint replay data. Configuration, training traces and tensor hashes are
published with the study. Tensor files remain in the owner's local ignored
artifact directory and the existing Modal volume.

This is a research artifact, not a production model registration or a
commercially cleared weight release. Source terms include OmniAct's MIT dataset
card, ChartQA's GPL-3.0 metadata, BoolQ's CC-BY-SA-3.0, and noncommercial SLURP
audio used in replay. Those terms are not replaced by this repository's code
license. Raw source questions and media remain in ignored `data/`.

See `evals/v4-grounding-training-protocol-v1.json` and
`evals/acquisition/v4_grounding_training_v1.json` for source pins and integrity
checks. Do not use public-audit or confirmation examples for training.
