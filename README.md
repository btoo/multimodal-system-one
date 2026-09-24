# Multimodal System One

**A research project for learning fast, typed decisions directly from images and language on a MacBook.**

![Project overview: visual observations and language enter a jointly trained model that predicts answer probabilities.](docs/assets/overview.svg)

> **Stage: research and design.** No decision model has been implemented or trained. Architecture choices are hypotheses; all numerical figures below are analytical illustrations or explicitly labeled simulations. The repository currently contains the literature synthesis, original diagrams, a reproducible figure generator, and the proposed experiment protocol. The method now draws on current automated-research and recursive self-improvement work, including September 2026 preprints.

The first goal is a small model whose weights, data, losses, and failure modes we can understand. It should answer bounded questions about observations, return a probability distribution over the declared answers, and support abstention in the surrounding software. The initial target machine is an Apple M4 Pro with 48 GB of unified memory. Training throughput and achievable latency are still unmeasured.

**Recommended starting point:** a compact, jointly trained image–text transformer with a shared candidate scorer and supervised probability losses. Compare it against late fusion and a FiLM-style model; challenge it with a latent bottleneck. Use controlled scenes to establish whether it actually combines modalities before moving to natural images and audio.

This recommendation is an engineering judgment from the sources below. The best architecture for our data and compute budget remains an empirical question.

## Read the project

- [The decision we want to learn](#the-decision-we-want-to-learn)
- [Architecture and alternatives](#architecture-and-alternatives)
- [The probability objective](#the-probability-objective)
- [Data that requires multiple modalities](#data-that-requires-multiple-modalities)
- [Research that improves its own search](#research-that-improves-its-own-search)
- [MacBook implementation plan](#macbook-implementation-plan)
- [Evidence and next milestones](#evidence-and-next-milestones)
- [Annotated reading map](docs/research/reading-map.md), [decision register](docs/research/decisions.md), [experiment specification](docs/research/experiment-plan.md)

## The decision we want to learn

For observations $x$, a question $q$, and candidate descriptions $C=\{c_1,\ldots,c_K\}$, learn:

$$
p_\theta(y=k\mid x,q,C).
$$

Here, **System One** is our operational target: direct, bounded predictions with a short inference path. It is not a claim about human cognition, universal intelligence, or a new mathematical model class.

TypeSafe's Jev motivates the interface: structured questions and probabilistic answers. Its documented input is currently text-only. Its public training overview names RLCD without specifying a reproducible training recipe. This project investigates an independent multimodal design; it does not claim to implement Jev's internals. [Jev interface](https://docs.typesafe.ai/concepts/system-one), [RLCD overview](https://docs.typesafe.ai/introduction/machine-learning-primer).

### What “native multimodal” means here

Images enter as pixel-derived tokens, language as text embeddings, and later audio as spectrogram patches. Their representations interact in the trainable prediction network. The decision loss can update both modality pathways. Separate modality-specific input stems are compatible with this definition.

There are two distinct development tracks:

| Track | What is learned | What it can establish |
|---|---|---|
| **A · Fundamentals** | All small-model weights, from random initialization | Multimodal learning on a controlled task distribution |
| **B · Practical transfer** | A decision head, fusion layers, or adapters over pretrained encoders | Useful decisions on richer inputs, with inherited pretraining knowledge |

We begin with A. Its limited vocabulary and generated scenes do not establish open-domain instruction following. Results from B must be reported separately, including pretrained parameter counts and data provenance.

### Output contract

| Primitive | Learned output | Software representation |
|---|---|---|
| Boolean | Bernoulli probability | `p_true` in `[0, 1]` |
| Choice | Categorical distribution over supplied candidates | Candidate ID plus all probabilities |
| Ordered score | Distribution over ordered rubric levels | Level probabilities and their weighted mean |

Initially, an ordered score can use a categorical head; an ordinal loss is a later comparison. A valid schema is a property of the serializer. Correctness and calibration require separate measurement.

The following is an **illustrative future response**, not a model result:

```json
{
  "type": "choice",
  "probabilities": {"left": 0.12, "right": 0.78, "stay": 0.10},
  "prediction": "right",
  "decision": "abstain",
  "policy": {"min_probability": 0.90},
  "calibration_id": "illustration-only"
}
```

Abstention is separate from the candidate labels. A softmax always allocates its mass somewhere, even when every supplied option is unsuitable. The v1 data contract supplies an exhaustive answer set, with an explicit `none` answer where the task requires it. Open-set rejection is an additional research problem.

## Architecture and alternatives

![Four architecture families: late fusion, FiLM conditioning, early token fusion, and latent bottlenecks.](docs/assets/fusion-families.svg)

The initial study compares four small models with the same observations, candidate semantics, task mix, and evaluation protocol:

| ID | Design | Why include it | Main concern |
|---|---|---|---|
| A0 | Image encoder + text encoder + pooled late fusion | Cheap control for what coarse features can solve | Pooling may erase spatial relationships |
| A1 | CNN features modulated by a text-conditioned FiLM network | Strong, simple conditioning baseline | Visual inductive bias; recomputation for each question |
| A2 | Image and question tokens in a joint transformer | Transparent baseline for learning cross-modal interactions | Token count and data efficiency |
| A3 | Inputs read into a small latent array, then queried | Decouples input length from the main processing depth | Compression may lose small objects or fine details |

FiLM provides a useful conditional-computation precedent; its own compositional tests also show that strong in-distribution results can coexist with a large generalization gap. ViLT supports the simplicity of patch/text interaction, but its published model uses pretrained initialization: it is not proof that our tiny random-initialized model will learn as efficiently. Perceiver IO motivates A3's input/latent/output separation. [FiLM §2 and §4.5](https://arxiv.org/html/1709.07871), [ViLT §3 and §4.5](https://arxiv.org/html/2102.03334), [Perceiver IO §3](https://arxiv.org/html/2107.14795).

### Proposed A2 baseline

![Detailed proposed model: image patches and question tokens interact, then an independent shared scorer evaluates candidate descriptions.](docs/assets/architecture.svg)

1. Split a `64 × 64` RGB image into `8 × 8` patches: 64 visual tokens.
2. Embed question tokens with a small, training-only vocabulary and position embeddings.
3. Add modality and spatial positions; process image and question tokens together using bidirectional attention.
4. Encode each candidate description with the same small text encoder, independently of its slot in the answer list.
5. Use the candidate embedding to attend to the joint image/question representation; apply the same scalar scoring function to every candidate.
6. Mask padding, then apply softmax across the valid candidate scores. Serialize the numerical result.

In compact notation:

$$
H=F_\theta([P_\theta(x_{image});E_\theta(q)]),\qquad
e_k=E_\theta(c_k),
$$
$$
z_k=g_\theta\!\left(e_k,\operatorname{CrossAttn}_\theta(e_k,H,H)\right),
\qquad p_k=\frac{\exp(z_k/T)}{\sum_{j\in valid}\exp(z_j/T)}.
$$

Use $T=1$ during the initial training baseline. Fit any post-training temperature on a dedicated calibration split after selecting the model.

**Starting configuration, subject to a smoke profile:** width 256, six joint blocks, four attention heads, feedforward width 1024, and two shared text-encoding blocks. Cap questions at 32 tokens and candidates at eight tokens each, with 2–8 options. The expected size is on the order of 5–10 million parameters; the implementation must report the exact count. We should also try smaller widths rather than assume this scale is optimal.

### Candidate order should not become evidence

![Candidate permutation diagram: reordering options only reorders their corresponding probabilities.](docs/assets/candidate-equivariance.svg)

The independent shared scorer gives a structural invariant:

$$
p_\theta(x,q,\pi C)=\pi p_\theta(x,q,C).
$$

This holds at deterministic inference when candidates have no slot embeddings, cross-candidate attention, or order-dependent truncation. Word positions *inside* a candidate remain meaningful. Candidate IDs are output keys; descriptive text supplies semantics. Reject duplicate IDs, duplicate semantic options in the controlled grammar, and questions that refer to opaque option positions. For exact score ties, return the tied set or abstain so the final label does not depend on array order.

Set symmetry is a general design principle [Deep Sets](https://arxiv.org/abs/1703.06114). Textual label conditioning is related to GLiClass, but our isolated scorer deliberately differs from its main joint label/context encoder. [GLiClass §2.1](https://arxiv.org/html/2508.07662).

The invariant is about **permutation of the same candidate set**. Adding an option changes the softmax denominator. Independent scores also impose a restrictive choice structure; relational questions about the candidate set may later require a permutation-equivariant set module.

### Several questions and the cost of perception

The baseline evaluates questions as independent batch items. Parameters are shared; questions do not attend to one another. Because A2 fuses the question into the observation representation, its full representation cannot be reused unchanged across new questions. Only question-independent input work can be cached.

A later challenger can encode observations once and let separate query heads read that state. We must test the quality cost of compressing observations before the question is known. Parallel numerical outputs remove sequential answer-token decoding, but input processing and additional questions still cost compute.

![Analytical attention-pair growth for full input attention versus a fixed-size latent bottleneck.](docs/assets/attention-scaling.svg)

The plot counts attention interactions for a simplified six-layer comparison, with 32 latent tokens and eight output queries. It excludes projections, feedforward work, kernels, batching, and memory traffic. It is **not a latency or FLOP benchmark**. The bottleneck pattern follows the complexity decomposition in [Perceiver IO Appendix E.2](https://arxiv.org/html/2107.14795).

### Which other approaches belong in the study?

| Approach | What it contributes | Position in this project |
|---|---|---|
| Contrastive alignment: CLIP / SigLIP | Reusable image/text representations | Transfer baseline or optional pretraining objective |
| ImageBind | Alignment across more than two modalities | Reference for later audio and missing-pair data |
| BLIP-2 | Small learned bridges over frozen encoders | Reference for Track B |
| GLiClass | Direct predictions conditioned on label text | Reference for dynamic answer spaces |
| VL-JEPA | Predict semantic answer embeddings; score without always decoding text | Important alternative objective after the supervised baseline |
| I-JEPA / LeJEPA | Representation learning and explicit collapse prevention | Optional pretraining study when unlabeled data is useful |
| Chameleon / Transfusion | Joint modeling of mixed modalities | Generative extension if the eventual task needs image or text synthesis |

Sources and limitations for each are in the [reading map](docs/research/reading-map.md). We should not add every loss at once. Similarity learning, generative modeling, probability estimation, and action selection solve different problems. An experiment must identify which capability its added complexity is supposed to improve.

## The probability objective

Start with supervised negative log likelihood (NLL):

$$
\mathcal L_{choice}=-\log p_\theta(y\mid x,q,C),
$$
$$
\mathcal L_{boolean}=-y\log p-(1-y)\log(1-p).
$$

Use a fixed task sampling mixture. Compare Brier loss separately before considering an auxiliary weighted combination. For ordered outcomes, retain the full level distribution; a mean alone hides ambiguity.

Log loss and Brier loss are proper probability objectives: their population optima recover the true conditional distribution under the usual realizability/optimization assumptions. Finite data, misspecification, and distribution shift can still leave a trained network miscalibrated. [Proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf).

### Why correctness reward is not a probability target

Let the true chance of an event be $q$, and let an agent randomly choose “yes” with probability $p$. Its expected correctness reward is:

$$
R(p)=qp+(1-q)(1-p).
$$

For $q>1/2$, this is maximized at $p=1$, rather than $p=q$. With $q=0.7$, an optimal action policy always choosing “yes” can coexist with the honest belief “70%.” A policy's action probability and an event's probability have different meanings.

![Analytical comparison: NLL and Brier objectives prefer the true probability while sampled correctness reward prefers a deterministic action.](docs/assets/proper-scoring.svg)

This is our own elementary derivation, not a description of Jev's unpublished algorithm. If labels are directly available, differentiating a proper loss is the simplest baseline. RL becomes relevant for delayed rewards, partial feedback, or interaction-dependent outcomes; the reward still must match the quantity we want to learn. Replacing supervised training with PPO does not itself establish calibration.

The binary illustration uses Brier loss $(p-y)^2$. The experiment ledger uses the sum over all categorical outcomes, which is twice that value for a two-class distribution. The convention changes scale, not the optimum.

### Calibration, uncertainty, and abstention

We will report NLL, Brier score, reliability plots with bin counts, and expected calibration error (ECE) under a declared binning rule. ECE alone can conceal poor discrimination or depend heavily on bins. A model predicting the base rate everywhere can be calibrated and still be unhelpful.

Temperature scaling is a useful low-complexity postprocessing comparison. Fit it on calibration data only; report both raw and adjusted outputs. One temperature is not guaranteed to repair every question family or unfamiliar distribution. [Temperature scaling](https://arxiv.org/html/1706.04599), [uncertainty under shift](https://arxiv.org/abs/1906.02530).

![Synthetic calibration and risk-versus-coverage demonstration, generated from a known probability process rather than a trained model.](docs/assets/calibration.svg)

This illustration samples outcomes from a known Bernoulli process, artificially sharpens predictions, and then reverses that sharpening. It explains the diagnostics; it is not evidence for our proposed model. The risk/coverage panel shows the tradeoff as a confidence threshold rejects more cases. Selective classification and conformal prediction offer further tools, with assumptions and guarantees that must be stated precisely. [Selective classification](https://arxiv.org/abs/1705.08500), [conformal prediction tutorial](https://arxiv.org/abs/2107.07511).

For the first proof, use a simple threshold policy and measure the actual error among accepted examples. Missing evidence, unfamiliar inputs, and ambiguous observations require dedicated evaluation. High softmax probability does not certify familiarity.

## Data that requires multiple modalities

The first dataset will be a deterministic 2D scene generator: colored shapes, positions, and simple occlusions, paired with a small compositional question grammar. All labels come from an independent scene oracle. The model sees rendered pixels and text; scene graphs, seeds, template IDs, and oracle programs stay outside its inputs.

![Counterfactual tests: change the scene or change the question while holding the other fixed.](docs/assets/counterfactuals.svg)

Start with existence, attribute selection, and one-hop spatial relations. Add bounded counting, then noisy or incomplete observations. Long-horizon navigation is a later task because it mixes perception with planning and credit assignment.

Key construction rules:

- **Split by scene family before rendering and question expansion.** Views or paraphrases of the same underlying scene cannot cross splits.
- **Balance labels within question templates.** A text-only baseline should fail on tasks intended to require vision.
- **Use paired interventions.** Change the relevant visual property while holding text fixed; change the question while holding the scene fixed. Include distractor changes that should leave the answer unchanged.
- **Hold out combinations and grammar patterns.** Novel seeds alone do not test compositional generalization. Keep every individual primitive represented in training.
- **Treat observation noise explicitly.** Controlled hidden-state mixtures can supply known conditional event probabilities; arbitrary label corruption and occlusion are not interchangeable.
- **Separate ablations from robust training.** Shuffling a modality diagnoses reliance; a separately trained unimodal control diagnoses information leakage. Missing-modality training requires labels consistent with the information that remains.

CLEVR's diagnostic design and CoGenT splits motivate the controlled stage. Winoground and ARO motivate relation/order counterexamples. Their published results do not transfer automatically to our generated data. [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/), [Winoground](https://arxiv.org/abs/2204.03162), [ARO §3–4](https://arxiv.org/html/2210.01936).

### Data progression

| Stage | Inputs | Purpose |
|---|---|---|
| D0 | Pixels + grammar-based language | Verify learnability and eliminate simple shortcuts |
| D1 | New combinations, paraphrases, corruptions | Diagnose generalization and confidence failures |
| D2 | Natural images and richer language | Track B, starting with a carefully scoped real-data task |
| D3 | Images + text + short audio | Test cases where audio supplies information unavailable in the image |
| D4 | Short synchronized video/audio histories | Temporal decisions, after explicit timing and alignment support |

NLVR2 is a candidate natural-image benchmark, subject to its image-access terms. Audio spectrogram patches offer a tractable input route; an audio stream needs timing and missingness semantics as well as an encoder. [NLVR2](https://lil.nlp.cornell.edu/nlvr/), [AST](https://arxiv.org/abs/2104.01778).

## Research that improves its own search

There are two different optimization targets: **the multimodal model** and **the research procedure that finds better models**. Improving the first is ordinary automated experimentation. Improving the second requires testing whether a changed procedure finds better solutions under the same budget on tasks it was not tuned to.

![Two levels of improvement: model experiments below, independently evaluated research-policy changes above.](docs/assets/two-level-research.svg)

The methodology is a synthesis of current primary research, with small-scale adaptations declared explicitly:

| Source | Evidence and scope | What we adopt |
|---|---|---|
| [AIDE², Sep 22, 2026](https://arxiv.org/abs/2609.26457) | Research-harness rewrites with external-task evaluation; its ignition test is inconclusive | Separate model search from method search; compare research efficiency at equal budgets |
| [AREX, Sep 1 revision](https://arxiv.org/abs/2607.21461) | Research-answer refinement through checks of individual constraints | Maintain verified claims, contradictions, and open questions; perform targeted follow-up searches |
| [RSIAgent, Sep 18 revision](https://arxiv.org/abs/2609.15364) | Environment practice retained as reusable memory, with frozen model weights | Distill evidence from successes and failures; reconcile contradictory lessons |
| [ASI-Evolve, Mar 31](https://arxiv.org/abs/2603.29640) | Automated research over model/data/learning changes | Attach a compact analysis and literature rationale to each experiment |
| [Hyperagents, Mar 19](https://arxiv.org/abs/2603.19461) | Editable task and meta-level agent programs | Version the proposer/selection procedure separately and test transfer before promotion |
| [ShinkaEvolve, maintained through Aug 2026](https://github.com/SakanaAI/ShinkaEvolve) | Executable population-based program search | Retain alternative lineages, detect repeated proposals, and allocate an exploration budget |
| [GEPA](https://arxiv.org/abs/2507.19457) | Reflection-guided prompt optimization | Consider tested changes to research instructions as a later method-level experiment |

These systems are not interchangeable, and some of the papers are very recent preprints. We have inspected methods and source material, not reproduced their reported gains. The [frontier review](docs/research/frontier-methods.md) records dates, reading depth, limitations, and the adoption decision.

**Practical choice:** use an archive-based, evidence-driven experiment process for the proof of concept, with ShinkaEvolve as the preferred runner candidate after a local compatibility check. Keep the initial research policy fixed for comparison. A later meta-study may change that policy, but must use separate task suites and account for both training compute and proposer cost. This avoids confusing a lucky model trial with a better research method.

The [research state](docs/research/research-state.json) already applies claim-by-claim verification to this literature pass. Its unresolved entries remain hypotheses. The [protocol](program.md) specifies how the same discipline will govern future experiments. There is no training or evolution runner in this repository yet.

![Proposed experiment cycle: literature and archive inform a hypothesis, frozen measurements create evidence, and several useful lineages remain available.](docs/assets/research-loop.svg)

Our planned loop makes the following choices explicit:

| Component | Proposed rule |
|---|---|
| Mutable surface | Model and training configuration; one hypothesis per trial |
| Frozen surface | Data semantics, split manifests, oracle, metric code, and resource accounting |
| Search metric | Macro average of per-task NLL normalized by `log(K)` |
| Screening budget | 300 seconds of synchronized optimizer-loop wall time per run |
| Confirmation | Baseline and nominee, five fresh seeds each, 1,800 seconds per run |
| Constraints | Predeclared latency, memory, correctness, and modality-use checks |
| Recording | Source hash, config, seeds, environment, metrics, durations, and failure status |
| Stop rule | Finite campaign cap and explicit research question; no indefinite execution |

Random uniform predictions have normalized NLL 1 for a fixed exhaustive categorical answer set. Report raw NLL and accuracy alongside the scalar; normalization is for cross-task aggregation, not a claim that every task has equal difficulty. Finalists must also survive slice-level checks so the mean cannot hide a failed modality or task.

The proposed first campaign is 12 architecture-screening runs, eight refinement runs, and ten confirmation runs: **6 h 40 min of allocated training time**, excluding setup and evaluation. This is a planned budget, not a runtime estimate or authorization for an unattended run in this research stage. A short profiling pilot must confirm that five-minute screens are informative before freezing this campaign. The optional meta-research campaign has a separate budget and is disabled.

### Keep the final test out of the search

![Data governance diagram: train and development feed model search; calibration fits temperature; a final test is used after the model is frozen.](docs/assets/evaluation-firewall.svg)

Use separate train, development, calibration, and final-test scene families. The search may repeatedly consult development results. Repeatedly choosing models against a “test” makes it part of development, regardless of its filename. [Adaptive data analysis](https://arxiv.org/abs/1506.02629).

Freeze the candidate and its policy before the final evaluation. If final results trigger design changes, retire that test as development evidence and prepare another untouched test for a later confirmatory claim. This is procedural separation, not an access-control guarantee against an agent that can read the filesystem.

The [experiment specification](docs/research/experiment-plan.md) defines leakage checks, candidate permutations, multi-question isolation, missing modalities, scene-level uncertainty estimates, and the complete metric ledger. The [campaign manifest](experiments/campaign.json) is marked `planned`, with execution disabled.

## MacBook implementation plan

Use **PyTorch + MPS** for the first implementation, with CPU reference checks and eager execution. This is a portability and debugging choice, not a measured speed ranking. PyTorch documents MPS GPU training; unsupported operations and numerical behavior must be checked on the actual installed version. [MPS documentation](https://docs.pytorch.org/docs/main/notes/mps.html).

**MLX is a credible alternative** for Apple silicon. Benchmark it later against identical data, model semantics, precision, and timing boundaries if profiling identifies a worthwhile port. For historical comparison, the autoresearch-linked [MPS fork](https://github.com/miolini/autoresearch-macos) and [MLX fork](https://github.com/trevin-creator/autoresearch-mlx) are useful implementation references; neither has been executed here. [MLX](https://github.com/ml-explore/mlx).

Practical defaults:

- Start with float32, standard dense attention, AdamW, and short sequences. Verify lower precision before using it in comparisons.
- Profile full input-to-output latency, including preprocessing and synchronization, at batch size one. Also report throughput and multiple-question scaling separately.
- Fix power mode, record OS/framework versions, and randomize trial order where practical. Laptop thermal drift can distort rankings.
- Record process memory, MPS tensor allocation, and driver allocation with their distinct meanings. The 48 GB physical memory pool is shared with macOS; it is not a 48 GB exclusive GPU budget.
- Choose a conservative working budget in the pilot. A proposed 12 GiB process/driver guardrail is a starting operational limit, not an estimate of required memory.

As a rough parameter-state calculation, 10 million parameters with float32 weights, gradients, and two Adam moments occupy about 160 MB in decimal units before activations, temporary buffers, framework overhead, and input data. Sequence lengths and attention can dominate the rest. This estimate is not a measured memory profile.

### Reproduce the documentation

```bash
uv sync --locked --group docs
uv run --locked --group docs python scripts/render_figures.py
uv run --locked python scripts/check_docs.py
```

The figure generator writes SVGs and PNG previews. Sources are original code, with a fixed random seed for the calibration illustration. Its manifest records the formulas, assumptions, and simulated values. No external image generator or copied paper figures are used.

## Evidence and next milestones

| Item | Current status |
|---|---|
| Primary literature and upstream source review | Documented in the reading map |
| Architecture and loss recommendation | Proposed; untested on our task |
| Original explanatory figures | Reproducible documentation artifacts |
| Task data, evaluator, model, and training loop | Not implemented |
| Trained checkpoints and benchmark results | None |
| Proof of multimodal generalization or calibration | Pending experiments |

Implementation should proceed through four gates:

1. **Make the experiment trustworthy.** Implement the scene oracle, split manifests, metrics, and leakage controls. Verify that probability metrics behave correctly on known distributions.
2. **Make a tiny model learn.** Overfit a small controlled batch; confirm gradients reach both input pathways; run CPU/MPS numerical checks. This proves implementation mechanics only.
3. **Run the bounded architecture study.** Compare A0–A3, preserve failures, and confirm promising changes at a longer budget across fresh seeds.
4. **Publish the actual evidence.** Report all planned slices, calibrated and raw metrics, timings, resource use, counterexamples, and an interactive local demo using a real checkpoint.

The research pass covers the major choices needed for this first proof. It cannot establish an exhaustive optimum across all architectures or guarantee that published large-model gains survive downscaling. The most consequential open questions are whether task-conditioned fusion beats FiLM at this scale, whether a bottleneck preserves spatial evidence, and whether confidence remains useful on unfamiliar combinations.

### Repository contents

```text
README.md                         Research narrative and architecture
program.md                        Evidence-driven evolutionary research protocol
docs/research/frontier-methods.md  Current RSI methods and adoption decisions
docs/research/research-state.json  Verified claims and unresolved constraints
docs/research/reading-map.md       Annotated primary sources and reading depth
docs/research/decisions.md         Proposed choices and falsification criteria
docs/research/experiment-plan.md   Data, metrics, budgets, and acceptance gates
docs/research/autoresearch-source.json  Historical source inspection
docs/research/shinka-source.json   Inspected runner candidate and hashes
docs/assets/                      Original SVG figures and PNG previews
experiments/campaign.json          Disabled, proposed campaign configuration
experiments/results.tsv            Empty result ledger; no measured runs
scripts/render_figures.py          Reproducible diagrams and analytical plots
scripts/check_docs.py              Documentation integrity checks
```

Historical reference: [autoresearch](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117), inspected at `228791f` (March 26, 2026), remains a simple greedy-loop control. Its CUDA language-model implementation was not run. Repository age alone does not establish methodological inferiority.

Research snapshot: **2026-09-23**. This is an independent project. References to Jev, autoresearch, or other model families identify influences, not affiliation or reproduced results.
