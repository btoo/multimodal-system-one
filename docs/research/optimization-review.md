# Optimization review: make perceived concepts usable

The strongest current diagnosis separates **decodable perception** from **usable cross-modal binding**. A fixed linear readout extracts glyph identity from the selected factorized encoder at **97.92% on ID panels and 92.58% on held-color panels**. Yet that checkpoint's decision head answers “not present” to every development color/position question. More visual capacity alone therefore does not address the demonstrated failure. The next controlled intervention is direct primitive supervision with shared audio/glyph concepts, followed by joint learning while retaining that supervision.

This review was made on 2026-09-24. It uses primary papers, existing training/development predictions, and two explicitly bounded CPU diagnostics. It does not reproduce the papers, infer Jev's unpublished recipe, or claim recursive improvement. [Complete diagnosis and provenance](../../reports/optimization-diagnosis-v1/README.md).

## What the saved evidence establishes

The three scale-study nominees were selected after only one or two epochs. Small RGB and factorized nominees choose “not present” for **100%** of both ID and composition color/position requests: present-target accuracy is **0%**, absent-target accuracy **100%**. The larger nominee does the same on all composition requests and 99.48% of ID requests. Their aggregate scores mainly reward an absence shortcut.

The existing `tile_word` auxiliary score is not a pure image-recognition test. It also requires parsing a location question, routing to the right visual token and mapping its contents through the shared candidate scorer. Low accuracy can arise at any of those stages. We fitted equal, temporary ridge readouts to frozen representations to separate them. Every condition used the same 256 training panels, 1,024 training glyphs, training-only standardization and fixed ridge strength 1; every probe was scored on the same existing development panels. No neural weights changed.

| Frozen checkpoint | Glyph ID | Glyph held-color | Color ID / held-color | Audio word |
|---|---:|---:|---:|---:|
| Existing served v2 | 79.69% | 39.19% | 100% / 100% | 83.33% |
| Small RGB nominee | 64.19% | 10.42% | 100% / 100% | 80.73% |
| Larger RGB nominee | 68.36% | 6.12% | 100% / 100% | 81.77% |
| Factorized nominee | **97.92%** | **92.58%** | 100% / 100% | 80.21% |

The RGB features have a marked held-color weakness. The factorized features contain readily decodable glyph information that the joint head does not use successfully. These are distinct failure modes. The ridge classifiers have supervised training of their own and are discarded; their accuracies are neither served-model scores nor general visual capability. Results condition on these selected checkpoints and one small probe-training subset.

## Raw confidence affected checkpoint selection

The prior scale study selected on **raw**, not calibrated, normalized NLL. Scalar temperature fitting happened only after selection. By epoch 32 the factorized run reached 71.61% ID and 50.87% composition development accuracy, but composition normalized NLL rose to 2.0368, versus 0.9626 at its selected epoch 2. The selector rejected later, increasingly overconfident weights. The final epoch's logits and weights were not retained, so a calibration rescue cannot be measured retrospectively.

A positive temperature leaves every argmax unchanged. Calibration might improve probability quality or prevent a future selector from discarding useful rankings, but it cannot turn the observed roughly 51% composition accuracy into reliable binding. Preserve selected and final logits in the next study, apply the same selection rule to treatment and control, and report accuracy and proper losses separately. Never select temperature on confirmation data.

## How the recent methods change the next experiment

### Asymmetric Representation Learning, ICCV 2025

[Wei et al., Improving Multimodal Learning via Imbalanced Learning](https://openaccess.thecvf.com/content/ICCV2025/papers/Wei_Improving_Multimodal_Learning_via_Imbalanced_Learning_ICCV_2025_paper.pdf) challenges the premise that equal modality contributions are optimal. ARL combines modality-specific losses with gradient modulation informed by predictive uncertainty; its authors explicitly acknowledge that single-inference entropy is an imperfect uncertainty estimate. The variance argument in Eq. 13 omits a cross-covariance term, so extending its optimality claim requires an additional assumption about correlated branch outputs. This last observation is our mathematical reading, not an empirical refutation. **Decision:** do not make equal gradient norms the optimization objective or treat high confidence as universal reliability. Read: main-paper §3, experimental setup/ablation, and stated limitation. PDF SHA256: `532de3abdfb3340d2de8b0df81834f32d044bc1f885dfe232e39fa0281b4347e`.

### Gradient-Guided Distillation, ICCV 2025

[Rakib and Bagavathi, G²D](https://arxiv.org/html/2506.21514v1) (arXiv v1, 2025-06-26) retains unimodal feature competence using pretrained teachers, feature and logit distillation, and sequential modality prioritization. Its teachers predict the same downstream labels. It also reports that some balancing baselines depress the stronger modality, and that preserving modality-specific features need not collapse the modality gap. **Decision:** borrow the principle of maintaining useful branch representations, not its complete teacher/logit recipe. Our direct primitive labels are observable from each corresponding modality; many of our joint targets are not. A shared concept classifier encourages common semantics while leaving modality-specific feature dimensions available. Read: §3.1–3.2, Algorithm 1, §4.2 and the multi-modality comparison; no author training code executed.

### Gradient Modulation Projection, 2026

[Li et al., GMP](https://arxiv.org/html/2603.14175v1) (v1, 2026-03-15; authors report AAAI 2026 oral acceptance) separates classification gradients from domain-invariance gradients, then modulates and projects them according to task strength and conflicts. Its setting has source-domain labels, a domain discriminator and pretrained audiovisual encoders; experiments report averages over five runs. Strong source accuracy is insufficient evidence for unseen-domain performance. **Decision:** retain that caution and measure gradient relationships, but defer its intervention. Our current model has no domain-adversarial objective. Adding one and changing optimization together would obscure attribution, and making all visual features color-invariant would remove information needed by color questions. Read: task definition, gradient-decoupling/projection methods and experimental setup.

### BALM, 2026

[Nguyen et al., BALM](https://arxiv.org/html/2603.19718v1) (v1, 2026-03-20; authors report CVPR 2026 acceptance) addresses unequal modality missing rates using context-conditioned feature calibration and gradient rebalancing with auxiliary unimodal heads. Its mechanism includes uneven exposure and missing inputs. **Decision:** defer until a genuine missing-modality experiment exists. Our full joint-training condition always provides audio and image, so unequal missing rates do not explain its present failure. The shared-target unimodal-head assumption also needs care for our cross-modal questions. Read: §3.1–3.4 and the stated emotion-recognition setting; screened for applicability rather than replicated.

### Pareto LoRA, 2026

[Wei et al., Pareto LoRA](https://arxiv.org/html/2606.17296v1) (v1, 2026-06-15) uses gradient conflict and magnitude diagnostics to choose Pareto integration, rescaling or ordinary updates. It studies Emu2 instruction tuning for text/image generation on eight MI300X GPUs, evaluated with a model judge. Table 3 also reports reduced text quality and helpfulness in one task despite improved image metrics. **Decision:** use its diagnostic discipline, not its headline gain or training recipe. Our candidate classifier has neither LoRA parameters nor separate generated image/text objectives. Projection becomes a testable follow-up only if repeated gradient measurements identify a specific conflict worth changing. Read: §3.2–3.3, experimental setup, Tables 1–3 and the authors' discussion of that tradeoff.

These sources disagree about what should be balanced and rely on different targets. Their recency does not resolve our design choice. None establishes that simply amplifying the weaker-looking branch will improve a candidate-conditioned decision model.

## Why primitive supervision fits this task

For a panel of four distinct commands and a uniformly sampled spoken command from eight, the image alone leaves a 1/2 probability that the requested command is absent. Each particular location has probability 1/8. Predicting absence can therefore achieve 50% location accuracy without hearing the request. The opposite-command mapping is bijective, so the same argument applies there. Under the symmetric renderer, knowing the spoken word alone also leaves the panel contents uncertain. Finite samples can deviate from those design probabilities.

Consequently, making an audio-only teacher confidently predict a joint location/color answer would reward a prior or a shortcut. The proposed primitive targets avoid that problem: audio predicts the recorded keyword; each visual token predicts its glyph and color. The same word head can classify audio and visual tokens because those labels refer to the same eight concepts. This is a task-matched hypothesis, not a claimed reproduction of a cited method.

A CPU gradient snapshot on 32 training scenes found that the existing question-conditioned `tile_word` loss also backpropagates through the audio branch. That is allowed by the architecture even though the target does not require audio. Image-branch joint/tile-word gradient cosine was −0.332 for the selected small nominee and +0.726 for the served model. This establishes that routing and relationships vary by checkpoint; one snapshot cannot establish causal modality dominance. Magnitudes also depend on parameterization, even when RMS and parameter-relative norms are reported.

## Concrete falsification protocol

The [executable follow-up protocol](../../evals/optimization-protocol-v1.json), frozen in source commit `558662f`, defines one training recipe against joint-only controls at two fixed seeds: 256 primitive-only warmup updates followed by the unchanged candidate objective plus 0.25 times each normalized primitive loss. Warmup is inside the common 8,192-update budget. Temporary heads are removed for inference; both conditions keep the same 668,097-parameter serving architecture. The accuracy/ID-gate selection rule is new and applies equally to treatment and control. Old-versus-new comparisons therefore do not isolate the curriculum effect.

| Observation | What it would establish or falsify |
|---|---|
| Primitive probes improve, but present-target joint accuracy remains near zero | Perception supervision alone does not make the fusion/scorer use the information. Investigate correspondence/routing next. |
| ID glyph probe improves while held-color glyph probe stays poor | The representation still uses a color shortcut; do not call that compositional perception. |
| Primitive competence falls between warmup and joint training | The retained loss is insufficient to prevent forgetting, or joint optimization interferes. |
| One treatment seed improves but the other regresses | The recipe is not yet supported as stable; publish both instead of selecting a lucky seed. |
| Aggregate accuracy rises only through more correct absent cases | The intended binding capability has not improved. Report present-target and absent-target slices. |
| NLL improves without any ranking/accuracy change | Probability scaling improved; it is not evidence of better perception or binding. |

Use within-treatment temporary-head curves to detect learning and forgetting. An untrained baseline temporary head is not a fair encoder probe; use equally fitted frozen readouts when comparing representations across conditions. Freeze independent confirmation speakers/hashes before training, nominate before opening their predictions, and keep the known-composition-gap limitation explicit. Real screenshots and richer speech remain separate application evaluations; these mechanisms do not establish computer-use readiness.
