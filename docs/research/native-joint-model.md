# Native question-conditioned decision model

The first unified implementation is `NativeDecisionModel` in [joint_model.py](../../mmso/joint_model.py). It accepts audio, an image, a textual question, and candidate descriptions. One set of weights produces a scalar for each candidate; a masked softmax returns the supplied answer probabilities. The output layer does not have a fixed unit for each answer class.

The [held-out report](../../reports/joint-v2/README.md) now records 72.92% joint accuracy versus about 50% for matched unimodal controls. The held command–color combinations fail at 45.66%. This establishes a useful controlled prototype, not robust compositional or real-browser generalization.

## What is neural, and what is outside the model?

| Component | Implementation |
|---|---|
| Audio input | PCM waveform → fixed log-mel transform → learned acoustic encoder |
| Image input | Pixels → four label-independent quadrants → shared learned visual encoder |
| Language | One learned token/position encoder for both questions and candidate descriptions |
| Fusion | Two shared transformer blocks over four visual, one acoustic, and one question representation |
| Decision | Each candidate independently queries the joint state; one shared scalar scorer |
| Output | Softmax over valid supplied candidates; opaque IDs mapped back by ordinary code |
| Calibration | A scalar temperature fitted on calibration data after model selection |

The model has **668,097 parameters** in the initial width-128 configuration. The acoustic representation is initialized from our earlier speech checkpoint, with its fixed eight-class head removed. Audio, visual, text, fusion, and scoring weights are then trained together. No transcript, audio label, scene graph, question-family ID, oracle answer, or candidate ID enters the forward pass.

The scene generator and its symbolic oracle create training labels and measure correctness. They are in a separate module and are not called by `predict_joint`. The public prediction function accepts files and request text without consulting a dataset manifest.

## Why the first task is controlled

The data pairs real human keyword recordings with generated 2×2 panels containing directional/control symbols. Questions request a color, position, or presence judgment about the spoken command or its opposite. The task defines opposite pairs as left/right, up/down, yes/no, and go/stop. Changing only the question can therefore change the correct answer while the audio, image, and candidate set stay fixed.

The six joint question families are:

- Color of the spoken command.
- Color of its opposite.
- Position of the spoken command.
- Position of its opposite.
- Whether the spoken command is present.
- Whether the spoken command is absent.

Two auxiliary question families ask which word was spoken or which command a specified tile represents. They use the same scoring head. They are reported separately and excluded from the primary six-family joint score.

The controlled task gives us an independent oracle and interventions. It is not evidence that the model can operate a real browser. The current acoustic vocabulary is eight words; the learned text vocabulary has 43 tokens including special tokens. Unknown text tokens and overlong requests raise explicit errors. The image stem is tailored to the generated quadrant layout.

## Data and split controls

The source manifest contains 2,048 training panels, 192 development panels, 128 calibration panels, 192 in-distribution test panels, 192 compositional test panels, and 96 test position interventions. Extra paraphrase questions reuse in-distribution test observations while changing the wording.

Development, calibration, and test speakers were excluded from every partition of the earlier audio baseline. Audio/content hashes, speaker groups, and scene families stay within a split. The compositional slice holds out one command/color combination per command while retaining its individual primitives in training.

All test statistics must identify their slice. Treating related questions or interventions as independent examples would exaggerate the effective sample size; report scene/speaker counts alongside question counts.

## What the development loop taught us

The initial fixed-pair model could memorize its training examples but generalized poorly. Every training audio recording had a single panel partner, and every panel a single spoken-word partner. A frozen visual-feature diagnostic showed that some glyph information was present, yet the full decision model failed to use it well.

The nominated follow-up keeps the model and optimizer fixed but independently re-pairs training words/recordings with panels on every batch. The training oracle recomputes the answer. This prevents stable audio→panel and panel→spoken-word lookup shortcuts. Development six-family accuracy rose from approximately 52% to 73%. This is development evidence from one seed, not a final-test claim or universal causal result.

The first run is preserved under [joint-full-v1](../../reports/joint-full-v1/training.json). The second is under [joint-full-v2](../../reports/joint-full-v2/training.json). The [nomination record](../../evals/joint-nomination-v2.json) pins the chosen checkpoint before held-out evaluation. Matched unimodal controls use the same data-sampling seeds and optimizer-step count.

## Interface and structural properties

```python
from mmso.joint_model import predict_joint

result = predict_joint(
    checkpoint="artifacts/joint-full-v2/model.safetensors",
    image_path="examples/joint-panel.png",
    audio_path="path/to/a/one-second-command.wav",
    requests=[{
        "question": "what color marks the spoken command",
        "candidates": [
            {"id": "choice-a", "text": "red"},
            {"id": "choice-b", "text": "blue"},
            {"id": "choice-c", "text": "green"},
            {"id": "choice-d", "text": "yellow"},
            {"id": "choice-e", "text": "not present"},
        ],
    }],
)
```

Use an exhaustive candidate set for an unconditional classification claim. A subset changes the meaning and normalization of the probabilities. In the generated panel task, include all four colors and `not present` unless intentionally running the easier-choice diagnostic.

Candidates have no slot embeddings or mutual attention. Permuting them therefore only permutes scores up to floating-point tolerance. Adding/removing candidates leaves existing unnormalized scores unchanged, while the softmax denominator changes. Candidate IDs are mapping keys and can be renamed without changing the input tensors. Questions are independent batch items; observation encodings may be reused across questions.

The neural prediction function is a choice scorer. The [HTTP developer API](../developer-api.md) exposes Choice, Noul, Score, and Ranking views over that scorer: Boolean questions use `yes`/`no` candidates; Score computes an expectation over declared rubric values; Ranking sorts candidate probabilities. These serializers do not add new learned tasks. Trained multi-label event outputs remain future work. The generic evaluation library's support for other metrics is not evidence that this checkpoint supports those capabilities.

These properties are checked separately from accuracy. Structural invariance does not establish understanding of novel words, arbitrary rubrics, or unseen task types.

## Scope relative to Jev

This implements an independently designed version of the public *behavioral idea*: observation + question + candidate semantics → direct typed probabilities. It uses supervised probability learning, not a claimed reproduction of Jev's RLCD. The available public description was insufficient to reconstruct Jev's complete architecture/training method.

The next step after this controlled proof is better visual proposals and semantically paired real speech/screens, followed by broader language and temporal inputs. The earlier real-screen grounding failure remains unresolved by success on these generated panels.
