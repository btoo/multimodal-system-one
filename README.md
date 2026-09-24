# Multimodal System One

A research project for a small, natively multimodal model that learns fast, typed decisions from images and language, starting on an Apple M4 Pro with 48 GB of unified memory.

> **Research foundation — work in progress.** No model has been implemented or trained. This repository is the base for a primary-source research review, illustrated architecture proposal, and a later proof of concept.

## Target

Learn a probability distribution over declared answers given observations, a question, and candidate descriptions. The initial study will train a small image–text model end to end on controlled visual scenes, then investigate richer inputs and audio.

The first candidates are a compact joint transformer, text-conditioned FiLM features, and a latent-bottleneck model. They are hypotheses to evaluate, not measured winners.

## Research in progress

- Review current recursive self-improvement and automated-research methodologies, including their evaluators, experiment archives, and evidence of real improvement.
- Compare multimodal architectures, probability losses, calibration methods, and data designs that expose shortcuts.
- Develop original theory and architecture diagrams, with reproducible analytical plots and clear separation from experimental results.
- Specify a bounded, hardware-aware experiment campaign with an independent final evaluation.

Karpathy’s [autoresearch](https://github.com/karpathy/autoresearch) was the initial reference. Its inspected revision and file fingerprints are recorded in [the source audit](docs/research/autoresearch-source.json). The methodology review is being expanded to newer research before selecting the final approach. No upstream training loop has been executed.

## Reproduce the documentation environment

```bash
uv sync --locked --group docs
```

The documentation and research dependencies are locked. Figure tooling and the full research README are being developed; there is no training command yet.

Research started 2026-09-23. Independent project; no affiliation with TypeSafe, Jev, or referenced research teams.
