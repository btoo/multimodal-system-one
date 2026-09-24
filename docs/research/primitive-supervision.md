# Primitive supervision and fresh speech confirmation

The [optimization diagnosis](optimization-review.md) separates three questions: is information present in the representations, can the decision head use it, and are its probabilities useful? The previous factorized model had highly decodable glyph features but failed the final decisions. A raw-loss selector also preferred checkpoints that mainly predicted absence. More parameters did not fix those problems.

This study changes the **training recipe** while retaining the exact 668,097-parameter inference model. It tests whether direct supervision of observable concepts, aligned across audio and vision, helps the shared candidate scorer learn useful joint decisions. It is an independently designed application of the research ideas, not a reproduction of G²D or Jev.

## The controlled intervention

The control learns only the existing candidate cross-entropy objective. The treatment begins with 256 updates of direct primitive supervision and then retains those losses during joint training:

$$L_{\mathrm{warmup}}=L_{\mathrm{visual\ word}}+L_{\mathrm{visual\ color}}+L_{\mathrm{audio\ word}},$$

$$L_{\mathrm{joint}}=L_{\mathrm{decision}}+0.25\left(L_{\mathrm{visual\ word}}+L_{\mathrm{visual\ color}}+L_{\mathrm{audio\ word}}\right).$$

Each cross-entropy averages over its own examples or visual tiles before combination. A temporary eight-way linear word classifier is shared between the visual and acoustic token representations. A separate temporary four-way head predicts color. The shared word classifier encourages compatible concept separation; it does not require the entire audio and visual vectors to be identical. Color information can remain in the visual representation.

Both conditions receive the same original 2,048 training scenes and independently re-paired speech stream. We test two initialization/sampling seeds. Each run has an 8,192-update budget and a 480-second synchronized-training guard. Warmup consumes part of the treatment's budget, so it receives 7,936 joint updates rather than the control's 8,192. The treatment adds 1,548 temporary parameters and additional labels/computation; this is a recipe comparison at matched attempts, not matched FLOPs.

Only training labels enter the auxiliary losses. Development labels measure diagnostics and select a checkpoint. No labels, temporary heads, transcripts, or scene metadata enter retained-model inference. Selected and final neural weights, development probabilities, and treatment readout diagnostics are retained. The trained auxiliary heads are saved separately for audit and excluded from the serving state.

Temporary-head accuracy before/after treatment is a within-treatment diagnostic. It is not a fair encoder comparison against an untrained control head. The separate fixed-readout procedure in the [diagnosis report](../../reports/optimization-diagnosis-v1/README.md) can compare the information in frozen encoders under matched readout training.

## Selection and probabilities

The previous raw-NLL-only rule could prefer a nearly constant absence prediction. Before training, this study instead declares an accuracy gate and selection order common to both conditions:

1. Prefer checkpoints with at least 65% development accuracy on familiar combinations.
2. Within the qualifying group, maximize mean accuracy across familiar and known held-combination slices.
3. Break ties using lower mean raw normalized NLL.

If no checkpoint passes the first gate, retain the best fallback as a failed run for promotion purposes. The rule does not erase raw probability quality: all raw NLL, Brier, calibration, selected/final evidence, and present-target versus absent-target metrics remain available. A single temperature is fitted only on the separate calibration rows after checkpoint selection. A gain against a previous study cannot be attributed solely to supervision because the selection rule also changed; the newly matched control is the causal comparison.

## New human recordings and evaluation boundary

The official [Speech Commands v0.02 test archive](https://www.tensorflow.org/datasets/catalog/speech_commands) is 112.6 MB. Its SHA256 matches the [pinned TensorFlow Datasets checksum blob](https://api.github.com/repos/tensorflow/datasets/git/blobs/ebcb6fa83ebb4f8ab6e780290bf3f82b538b3287). We use a declared subset, not the official benchmark score.

After excluding all 1,750 previously exposed speaker IDs and all prior audio hashes, the archive contains 70 eligible new speakers. Deterministic selection takes 32 distinct speakers per keyword, giving **256 recordings from 58 speakers** overall. Every recording is paired with a new familiar-combination panel and a new held-combination panel. No speaker or media hash overlaps previous manifests.

The color/word combinations held out of training are the same known failure types used in development. The confirmation is fresh in voices and images, not in the identity of the compositional gap. Images remain generated 2×2 panels. The [real-screen study](screen-grounding-v2.md) is a separate control and must not be attributed to this native model.

All four selected checkpoints and a frozen served-v2 reference are evaluated only after the nomination is committed. Both seed outcomes are reported, with paired speaker-cluster intervals conditional on each checkpoint. Two seeds provide a replication check, not a precise estimate of general training variance.

The [frozen protocol](../../evals/optimization-protocol-v1.json) defines accuracy gates before evaluation. Passing those gates still requires inference, latency, and compatibility verification before any serving change. Input vocabulary and real-world scope do not expand merely because the controlled score improves.

## Reproduction

```bash
uv run python scripts/run_optimization_study.py prepare
# Commit source, protocol and manifest before starting a new versioned study.
uv run python scripts/run_optimization_study.py train --run-id optimization-control-s24-v1
uv run python scripts/run_optimization_study.py train --run-id optimization-primitive-s24-v1
uv run python scripts/run_optimization_study.py train --run-id optimization-control-s25-v1
uv run python scripts/run_optimization_study.py train --run-id optimization-primitive-s25-v1
uv run python scripts/run_optimization_study.py nominate
# Commit the nomination before confirmation inference.
uv run python scripts/run_optimization_study.py evaluate --run-id optimization-control-s24-v1
uv run python scripts/run_optimization_study.py evaluate --run-id optimization-primitive-s24-v1
uv run python scripts/run_optimization_study.py evaluate --run-id optimization-control-s25-v1
uv run python scripts/run_optimization_study.py evaluate --run-id optimization-primitive-s25-v1
uv run python scripts/run_optimization_study.py evaluate --run-id joint-full-v2
uv run python scripts/summarize_optimization_results.py
uv run python scripts/check_optimization_results.py
```

Run IDs and recorded outcomes are immutable. A reproduction with new training must define new versioned IDs and a new confirmation policy. The checked-in reports can be verified without retraining or downloading raw media.
