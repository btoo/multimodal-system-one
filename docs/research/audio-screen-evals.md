# Audio, speech, and screen-use evaluation

This is the current application target. **Audio is a core modality.** The earlier image/text shapes experiment remains a mechanics and algorithm control; success there does not establish real-world speech or computer-use capability.

Status: evaluation specification and source shortlist. No data has been downloaded or recorded, no split has been populated, and no evaluator or model has been run. The default first application is spoken commands grounded in screenshots, with sound events evaluated independently. Start with English while recording language coverage explicitly; multilingual performance is a separate claim.

## Four capabilities to evaluate separately

| Suite | Input and example | Required output | Primary evidence |
|---|---|---|---|
| Speech meaning | Audio: “open settings”, “stop”, or a correction | Intent and bounded argument/target class; `unknown`/`wait` when appropriate | Macro F1, joint intent/argument accuracy, per-speaker slices |
| Acoustic events | Audio with a notification, speech, music, or overlapping sounds | Independent probabilities for co-occurring event labels | Macro average precision, per-class recall, false activations on negatives |
| Screen understanding | Screenshot + question about an enabled control, dialog, error, or target element | State labels or target probabilities/coordinates | State F1, grounding hit rate, candidate proposal recall |
| Joint audio/screen decisions | “Close that tab” with competing tabs and a declared reference context | Correct action type and target, or abstention | Joint exact match, paired intervention consistency, accepted-case error |

End-to-end computer/browser use is a fifth, later integration suite. A grounded click is not proof that the requested workflow succeeded. The surrounding agent may still need history, planning, text entry, and recovery.

## Public data first; targeted collection for the missing joint task

| Source | Role | What it does not establish | Access/provenance note |
|---|---|---|---|
| [Speech Commands](https://www.tensorflow.org/datasets/catalog/speech_commands) | Small real-speech pipeline check with short keywords and background noise | Sentence meaning, arbitrary commands, or screen grounding | Preserve official splits and speaker grouping; source card lists CC BY 4.0 |
| [Fluent Speech Commands](https://lorenlugosch.github.io/publication/2019-04-01-pretrain-speech-model) | Compact spoken-intent benchmark | GUI-specific commands or broad spontaneous speech | Use author-linked data and original terms; code license is not the corpus license |
| [SLURP](https://github.com/pswietojanski/slurp) | Richer spoken intent and entities, with recording metadata | Paired screen understanding | Audio and text have different terms; see [source license](https://github.com/pswietojanski/slurp/blob/master/LICENSE.txt) |
| [ESC-50](https://github.com/karolpiczak/ESC-50) | Small environmental-sound diagnostic | Browser-specific notification taxonomy or overlapping event detection | Preserve its source-recording folds; research dataset has noncommercial terms |
| [AudioSet](https://research.google.com/audioset/download.html) | Larger, multi-label sound-event reference | A ready-made local browser-audio corpus | Metadata/features are provided; underlying media availability and rights require separate handling |
| [ScreenSpot-Pro](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding) | High-resolution GUI grounding stress test | Screen-state classification, speech understanding, or workflow success | Keep as an evaluation benchmark; use the [author dataset](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) |
| [Multimodal-Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) | Screenshot/action data with task, website, and domain splits | Spoken instructions or current live-site behavior | Its card is marked OpenRAIL, while the original Mind2Web repository lists CC BY 4.0; resolve the selected release's exact terms before acquisition |
| [WebArena](https://github.com/web-arena-x/webarena) / [OSWorld](https://github.com/xlang-ai/OSWorld-V2) | Later execution-based workflow evaluation | Isolated model quality | Pin environment/task releases and use resettable test environments; do not import leaderboard scores |

The metadata-only shortlist is in [dataset-catalog.json](../../evals/dataset-catalog.json). These datasets are task-specific controls; pairing an unrelated sound clip with a screenshot would not create a meaningful joint label.

The first acquisition should be a small speech benchmark, a manageable acoustic-event subset, and a screen-understanding evaluation slice. Preserve untouched official test partitions. Data used repeatedly to select the model is development data; report public benchmark exposure and known pretrained-checkpoint provenance. Large test sets are not automatically contamination-free.

## The paired data we should collect

Build a small controlled browser/desktop test workspace with ordinary pages and apps, fictitious account data, and reproducible states. Record real utterances intentionally for the dataset. Preserve screen resolution and input timestamps. Capture the state before the decision, the requested target, and an independently checked outcome where an action is executed in the test workspace.

A **200–500 episode pilot** is a practical starting collection target, not a sufficiency guarantee for training or broad accuracy claims. Cover several speakers and at least five distinct app/site layouts if feasible; otherwise label the scope as personal-voice or limited-layout evaluation. Existing public data supplies broader speech/acoustic coverage. Reserve independent groups for development, calibration, and final evaluation; collect additional training examples instead of recycling the final test.

Useful cases include:

- “Click the blue button” with two blue controls; surrounding task context makes one correct or leaves it ambiguous.
- “Close the other tab” with an explicit prior referent; if that referent is absent, the correct behavior is to ask/defer.
- “Don't submit it—open the preview” to distinguish final intent from keyword spotting.
- “Stop” while media is playing, while a download is running, or while no relevant activity is visible.
- A notification sound with a visible incoming-call dialog versus a matching sound inside a video.
- Similar-looking enabled and disabled controls; duplicate button labels; small icons; zoom/theme variations.
- Noise, silence, clipped utterances, late corrections, and background speech. Addressed-to-assistant labels require an observable cue/context, not a guess from acoustics alone.

Use TTS, simulated sounds, and rendered UI variations to expand **training/development** coverage. Keep a human-recorded evaluation slice. Reading a benchmark's text instructions aloud creates an adapted speech-grounding benchmark; it is not the original benchmark and must be reported under a separate name. Do not use teacher transcripts or target labels as inputs to the native-audio arm.

## The model implication

The real-task path should accept **audio + resolution-preserving screen tokens + optional text/history** into trainable fusion and direct decision heads. Modality presence, audio time, and screen time are explicit inputs. Audio-only and image-only requests remain valid task types.

Keep two development tracks:

1. **From-scratch controls:** tiny speech keyword/sound classifiers and the synthetic image/text task teach the mechanics and reveal bugs.
2. **Practical native-input model:** start with compact pretrained visual and acoustic representations, train the fusion and heads, then evaluate adapters or selective unfreezing. Report frozen/trainable parameter counts and inherited pretraining. This retains raw-modality input without claiming all representations were learned from scratch here.

A speech-oriented representation and an environmental-sound representation may serve different needs. Choose or combine them based on the separate evals; a good keyword recognizer is not automatically a good sound-event or semantic speech model.

For screenshots, use a global view plus resolution-preserving tiles/crops. Account for the cost and recall of any proposal mechanism. A model choosing among ground-truth target boxes measures conditional selection, not complete grounding. Evaluate a deployable proposal path separately and include cases where no correct candidate is proposed.

For overlapping acoustic events, use independent sigmoid/BCE outputs. A categorical softmax is appropriate only when labels are mutually exclusive. The original Boolean/Choice/Score interface therefore gains a multi-label event output; no requirement forces event probabilities to sum to one.

## Baselines that identify where a gain comes from

| Baseline | Purpose |
|---|---|
| Class priors / simple rules | Verify the dataset cannot be solved cheaply through imbalance |
| Raw audio only | Speech and acoustic capability; shortcut control for joint tasks |
| Screenshot only | Visual capability; shortcut control for joint tasks |
| ASR transcript + screenshot | Measure the usefulness and failure modes of a speech-to-text bottleneck |
| Human transcript + screenshot | Oracle-transcript diagnostic to separate recognition from downstream decision errors |
| Native audio + screenshot | Proposed practical model |
| Oracle candidates versus generated candidates | Separate proposal failures from decision failures |
| Pixel-only versus pixels + DOM/accessibility | Separate learned visual grounding from extra structured information |

Train matched controls where feasible. Evaluate shuffling, masking, and contradictory modality pairs as additional diagnostics. An ASR cascade is a legitimate strong baseline; native audio must earn its benefit through acoustic cues, robustness, latency, or accuracy.

## Split discipline and labeling

Assign related examples to the same partition using recording/source IDs, speaker, session, utterance variants, screenshot family, UI template, and workflow. Keep all crops, channels, noisy mixes, TTS derivatives, and nearby frames of an original example together. Evaluate unseen speakers and unseen apps independently and jointly; random file splits do not establish either.

Use official splits for public benchmark comparisons. Report a custom grouped split under its own name. The release adapter must document any speaker or source overlap rather than silently claiming disjointness.

For paired tasks, the same instruction should appear in several screen states, and the same screen should support several instructions. Include both interventions that change the correct answer and irrelevant changes that preserve it. Counterbalance target position and label frequency. Candidate descriptions, DOM annotations, filenames, and future action traces must not reveal the correct answer.

Two reviewers should independently label a subset of the pilot, especially negation, ambiguous references, and overlapping events. Report disagreement and adjudication. Store acceptable answer sets where several targets/actions are valid; do not force an arbitrary single target and then call alternatives errors.

## Metrics and timing

Publish a dashboard per suite before considering any combined score. The synthetic normalized NLL scalar does not automatically apply to multi-label events or coordinate grounding.

- **Speech:** macro F1; intent/argument exact match; unknown-intent rejection; errors by speaker, accent/language coverage, noise, and recording device. WER is a secondary diagnostic for the ASR baseline, not the speech-meaning score.
- **Sound events:** macro/micro average precision, per-class recall at a fixed false-positive rate, false activations per hour on annotated negative streams, and event onset/offset error for streaming tasks. Unannotated events are not automatically verified negatives.
- **Screens:** state classification F1; target hit rate; target-size and resolution slices; proposal recall and conditional selection accuracy. Use the benchmark's native metric for official comparisons.
- **Joint decisions:** action-plus-target exact match, intervention consistency, performance relative to unimodal/cascade controls, and risk versus coverage. Report multiple valid answers and ambiguous cases separately.
- **Probability quality:** task-appropriate NLL/BCE/Brier, reliability, and abstention metrics. Multi-label calibration is per label and prevalence-aware.
- **Integration:** actual workflow success, recovery, and mistakes after execution in the resettable test environment. Do not substitute offline step accuracy.

For audio, record observation duration, endpointing delay, post-endpoint processing delay, real-time factor, and chunk p50/p95. A model processing a five-second clip in 80 ms has not responded within 80 ms of speech onset. Streaming evals receive only prefixes and screenshots available at each cutoff; full-clip statistics and future frames must not leak into earlier predictions.

The old 100 ms / 64×64-image target belongs only to the toy workload. Freeze new resource gates after profiling realistic audio durations, screenshot resolutions, candidate counts, and preprocessing. GPU memory and total wall time include encoders and proposal stages, even if their weights are frozen.

## Immediate implementation order

1. Implement a shared manifest and metric interface for the four suites, with explicit provenance and grouped splits.
2. Acquire the smallest useful public subsets after pinning versions/terms; freeze the first eval manifests before tuning.
3. Build one audio baseline and one screen baseline, including the ASR/structured-input comparison arms where relevant.
4. Collect the paired pilot and train/evaluate native fusion with a fixed task definition.
5. Run the architecture/objective search against these evals, then add an execution-based agent test.

No universal accuracy threshold is claimed yet. Baseline results and pilot variance should set task-specific acceptance targets before model selection; the final holdout must remain outside that process.
