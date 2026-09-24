# Reading map

Primary sources consulted on 2026-09-23, with the composition-study and developer-interface additions checked on 2026-09-24. This is a scoped design review, not a claim to cover every relevant publication or to reproduce any paper. “Methods” means targeted sections of full text were inspected; “abstract” means claims are limited to the authors' summary. A model's reported scale or benchmark result is not evidence for the same result on our machine.

The current automated-research sources are assessed separately in [frontier-methods.md](frontier-methods.md). The table below records the model-side evidence and the decision each source informs.

The application-specific speech, sound, and screen datasets are reviewed in [audio-screen-evals.md](audio-screen-evals.md), with source URLs, access notes, and acquisition status in [the dataset catalog](../../evals/dataset-catalog.json).

## Multimodal structure and learning

| Source | Reading depth | Relevant contribution | Limit / project decision |
|---|---|---|---|
| [ViLT](https://arxiv.org/abs/2102.03334) | Methods §3.1 and ablations §4.5 | Lightweight patch/text interaction | Its initialization is pretrained; our random-initialized A2 remains an experiment |
| [FiLM](https://arxiv.org/abs/1709.07871) | Methods §2.1 and CoGenT §4.5 | Text-conditioned affine visual feature modulation | Retain as a serious small-model control; compositional transfer still needs testing |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | Methods §3.1 and complexity Appendix E.2 | Separate input reading, latent processing, and output queries | A3 candidate; compression can discard spatial evidence |
| [Attention Bottlenecks for Multimodal Fusion](https://arxiv.org/abs/2107.00135) | Abstract and official research account | Restricted communication across modalities | Later intermediate-fusion comparison, not mandatory v1 machinery |
| [CLIP](https://arxiv.org/abs/2103.00020) | Abstract | Image/text contrastive representation learning | Useful transfer control; semantic similarity is not calibrated task probability |
| [SigLIP](https://arxiv.org/abs/2303.15343) | Abstract | Pairwise sigmoid alignment objective | Auxiliary pretraining candidate; it does not replace our conditional answer loss |
| [ImageBind](https://arxiv.org/abs/2305.05665) | Abstract and author project page | Image-paired alignment across several modalities | Reference for audio expansion; inherited pretraining is a separate track |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | Abstract | Learned bridge between frozen pretrained encoders | Track B precedent, rather than an all-weights-from-scratch proof |
| [GLiClass](https://arxiv.org/abs/2508.07662) | Methods §2.1 and §2.3 | Classification conditioned on text labels; multiple encoder variants | Borrow label semantics; do not assume its joint label interactions give order invariance |
| [Deep Sets](https://arxiv.org/abs/1703.06114) | Abstract | Symmetry-aware operations on sets | Motivate a structural permutation test; our scorer invariant is derived directly |
| [VL-JEPA](https://arxiv.org/abs/2512.10942) | Methods §2, implementation §3.1–3.2 | Predict answer representations conditioned on vision and a question | A relevant future objective challenger; uses substantial pretrained components |
| [I-JEPA](https://arxiv.org/abs/2301.08243) | Abstract | Predict masked image representations | Potential visual pretraining study; no probability-calibration guarantee |
| [LeJEPA](https://arxiv.org/abs/2511.08544) | Objective §5.1 | Embedding prediction with distribution regularization | Anti-collapse reference if auxiliary latent learning is tested |
| [Chameleon](https://arxiv.org/abs/2405.09818) | Abstract | Early-fusion mixed image/text generation | A wider output goal than the initial decision task |
| [Transfusion](https://arxiv.org/abs/2408.11039) | Abstract | Joint discrete-token and continuous-image generation | Deferred unless the project needs generative outputs |
| [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) | Abstract | Attention over spectrogram patches | Practical later audio input representation |
| [Knowledge distillation](https://arxiv.org/abs/1503.02531) | Abstract | Transfer predictive distributions to a smaller model | Later teacher/student track, with teacher errors and cost measured separately |
| [SmolVLM-256M model card](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) | Model card | An existing small generative VLM comparison candidate | No local speed or quality claim; inspect licensing/version when used |
| [Are Object-Centric Representations Better At Compositional Generalization?](https://arxiv.org/html/2602.16689v1) | Problem setup §3 and experiment design | Separate representation structure, data diversity, downstream size, and compute in controlled composition tests | Motivates our size-versus-visual-prior comparison; their pretrained object slots differ from our hand-specified shape/color transform |
| [Disentanglement of Color and Shape Representations for Continual Learning](https://arxiv.org/abs/2007.06356) | Abstract | Separate color and shape pathways as a representation hypothesis | Different learning setting; our local intervention needs its own evidence |

## Probability quality and evaluation

| Source | Reading depth | Relevant contribution | Limit / project decision |
|---|---|---|---|
| [Gneiting & Raftery: proper scoring rules](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf) | Definitions and categorical examples | Proper probability objectives | Start with NLL; compare Brier directly |
| [Guo et al.: calibration](https://arxiv.org/abs/1706.04599) | Temperature scaling §4.2 | Scalar postprocessing of logits | Fit separately after selection; report raw and adjusted performance |
| [Ovadia et al.: uncertainty under shift](https://arxiv.org/abs/1906.02530) | Abstract | Calibration can degrade outside the training distribution | Include explicit shifted slices |
| [Selective classification](https://arxiv.org/abs/1705.08500) | Abstract | Error/coverage tradeoffs through rejection | Measure accepted-case error; do not import another study's guarantees |
| [SelectiveNet](https://arxiv.org/abs/1901.09192) | Abstract | Jointly learned prediction and rejection | Later comparison against thresholding |
| [Conformal prediction tutorial](https://arxiv.org/abs/2107.07511) | Abstract | Prediction sets with statistical coverage | Optional; standard marginal guarantees require exchangeability and do not imply conditional correctness |
| [Adaptive data analysis](https://arxiv.org/abs/1506.02629) | Abstract | Adaptive reuse can overfit a holdout | Keep a separate final test and track any later reuse |

## Data and diagnostic controls

| Source | Reading depth | Relevant contribution | Limit / project decision |
|---|---|---|---|
| [CLEVR and CoGenT](https://cs.stanford.edu/people/jcjohns/clevr/) | Author dataset page | Oracle-generated visual questions and compositional splits | Build a smaller original 2D task; do not call it the CLEVR benchmark |
| [NLVR / NLVR2](https://lil.nlp.cornell.edu/nlvr/) | Author dataset page | Sentence verification grounded in synthetic or natural image pairs | Candidate real-data extension; honor image access conditions |
| [Winoground](https://arxiv.org/abs/2204.03162) | Abstract | Same-word, different-order visual language contrast | Use relation reversals and swapped-attribute examples |
| [ARO / composition-aware negatives](https://arxiv.org/abs/2210.01936) | Methods §3.2–4 | Retrieval shortcuts and hard negative construction | Ensure plausible alternatives differ in the relevant relation or attribute |

## Platform and interface references

The [developer interface design](developer-interface.md) additionally records current OpenAI and Claude media-block conventions and the Jev HTTP, rubric, and confidence contracts. Those interfaces inform our transport and result shapes; they do not confer their models' capabilities.

| Source | Reading depth | What was established |
|---|---|---|
| [Jev concepts](https://docs.typesafe.ai/concepts/system-one) | Official documentation | Typed probability interface and currently text-only inputs |
| [Jev AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer) | Official documentation | Public training intent; no complete RLCD recipe established |
| [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html) | Official documentation | GPU training backend; actual operator/precision compatibility still needs a local test |
| [MLX](https://github.com/ml-explore/mlx) | Official repository | Apple silicon array/autodiff framework; no comparative performance established here |
| [autoresearch](https://github.com/karpathy/autoresearch/tree/228791fb499afffb54b46200aca536f79142f117) | Source inspection | Historical fixed-budget greedy-loop control; CUDA-specific implementation |
| [autoresearch-macos](https://github.com/miolini/autoresearch-macos) | README | Potential MPS port reference; not locally executed |
| [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) | README | Potential MLX port reference; not locally executed |

## How the review was conducted

1. Identify the separate questions: representation, cross-modal interaction, output semantics, learning objective, calibration, generalization, resource cost, and research-process quality.
2. Search primary papers, author-maintained repositories, and official platform documentation. News and community summaries were discovery leads only.
3. Inspect the methods behind the decisions with the largest implementation consequences. For newer self-improvement work, distinguish model-weight changes from harness or memory changes.
4. Record an applicable claim and a limitation for each source. Do not turn large-scale results into promised laptop performance.
5. Preserve the unresolved hypotheses in [research-state.json](research-state.json) and give them concrete experiments in [experiment-plan.md](experiment-plan.md).

Coverage is sufficient to define the first bounded proof, not to establish a universal optimum. Continuing literature search is justified when a new source changes a candidate, a metric, or a falsification test.
