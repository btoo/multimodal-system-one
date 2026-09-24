# Current automated-research and self-improvement methods

Snapshot: 2026-09-23. This review distinguishes a paper's original publication, its latest inspected revision, repository activity, and actual local execution. Recent publication is not independent replication. No research system reviewed here has been run against our multimodal task.

## Conclusion for this project

Use **evidence-driven evolutionary experimentation** for the model study. Preserve diverse candidates, diagnose failures, measure resource use, and select using fixed evaluations. After this is working, evaluate whether changes to the research policy improve discovery on separate tasks at matched cost.

The implementation candidate is ShinkaEvolve, subject to a local evaluator integration test. The conceptual method also draws on AIDE², ASI-Evolve, AREX, and Hyperagents. This is a declared synthesis, not a claimed reproduction of one paper. A finite, serial GPU queue fits one laptop; a population does not require concurrent training or a swarm of agents.

## What is actually improving?

| Layer | Mutable object | Necessary evidence |
|---|---|---|
| Answer refinement | A draft claim or research answer | Better satisfaction of independently checked constraints |
| Experience reuse | Research notes, procedures, or memory | Transfer to new tasks after freezing the learned material |
| Program evolution | Model/trainer code | Repeated improvement under a fixed task and resource contract |
| Research-method improvement | Proposal, selection, analysis, or context policy | More improvement per fixed budget on held-out task families |
| Model self-improvement | The model's weights and learning procedure | A distinct training/evaluation study; not established by editing harness code |

The initial model study operates at the program-evolution layer. The present literature pass uses constraint checking and retained evidence. Neither constitutes demonstrated recursive acceleration.

## Source assessments

### AIDE² — strongest direct recent methodological match

[Paper, Sep 22, 2026](https://arxiv.org/abs/2609.26457); inspected HTML method, experimental design, discussion, and ignition-test sections.

The authors optimize research-agent code through nested searches and report repeated gains with external-benchmark transfer. Their separate test of whether a discovered agent is a better *outer-loop self-improver* remains inconclusive. This distinction changes our design: evaluate model quality and research efficiency separately. We adopt equal-budget comparisons and a separate transfer suite. The paper is one day old in this snapshot; independent reproduction was not established in this review. We did not locate or verify a complete runnable release during the pass.

### AREX — most relevant to the literature phase

[Paper, inspected v3 Sep 1, 2026](https://arxiv.org/abs/2607.21461); inspected inner/outer-loop and training sections.

AREX audits a provisional research answer against individual constraints and carries verified evidence and unresolved questions into subsequent rounds. Its training also involves agentic mid-training and RL; manually adopting its verification pattern does not reproduce that trained system. We apply the pattern through `research-state.json`: each consequential claim needs a source, an applicability limit, and a follow-up test. Self-reported confidence is not our acceptance criterion.

### RSIAgent — experience as an artifact

[Paper, inspected v2 Sep 18, 2026](https://arxiv.org/abs/2609.15364); inspected memory-update and limitations sections.

The system accumulates environment-specific lessons through exploration while leaving model parameters fixed. It separates experience acquisition from consolidation and recognizes verifier-error propagation as a limitation. For our project, every retained lesson must link to an experiment or a primary source; failure reports remain useful without being promoted to successful outcomes. This is a memory-management precedent, not evidence for better multimodal architecture search.

### ASI-Evolve — analysis and literature priors

[Paper, Mar 31, 2026](https://arxiv.org/abs/2603.29640); inspected researcher/analyzer/cognition/database sections. [Author-linked repository](https://github.com/GAIR-NLP/ASI-Evolve).

The framework combines experiments with a reusable body of prior knowledge and condensed analysis. Its reported applications span architectures, data, and learning algorithms. We adopt the explicit hypothesis→measurement→analysis record. Its results come from different tasks and budgets, so the gains do not predict a laptop result. The paper's sampling-policy comparisons motivate testing selection strategy rather than treating an archive as automatically beneficial.

### Hyperagents — make the research method testable

[Paper, Mar 19, 2026](https://arxiv.org/abs/2603.19461); inspected methods, transfer metric, and cost appendix. [Author-linked repository](https://github.com/facebookresearch/Hyperagents).

Hyperagents allow the task-solving and improvement procedures to be edited. Their improvement-at-budget evaluation suggests a concrete future test: freeze a candidate research policy, let it improve several new tasks, and compare it with the original policy. The reported experiments are much larger than this proof; the cost appendix alone describes tens of millions of self-modification tokens. We adopt separation of task/method versions, not their full experiment scale or an assumption of free compounding gains.

### ShinkaEvolve — preferred executable starting point

[Paper, Sep 17, 2025](https://arxiv.org/abs/2509.19349); [official repository](https://github.com/SakanaAI/ShinkaEvolve), inspected at `9912af12d423504b8d580f4179fd15f5f88b8c50` (Aug 21, 2026). The paper abstract, repository configuration, dependency file, novelty handling, and population documentation were consulted.

ShinkaEvolve exposes an archive, parent selection, novelty checks, and budget controls. We want those capabilities after the evaluator exists. The first configuration should use one evaluator job, a small archive, and exact configuration hashing before any paid novelty judgment. The dependency list is substantial; repository inspection does not establish Mac compatibility. A zero-provider, deterministic integration smoke test is required before live evolution. Its published sample-efficiency gains belong to its own experiments.

### GEPA — optional research-instruction optimization

[Paper](https://arxiv.org/abs/2507.19457); abstract consulted. [Author-linked code](https://github.com/gepa-ai/gepa).

Reflection over execution traces and retention of complementary candidates are relevant when the mutable object is an LLM instruction or prompt. This is a later option for the research-policy layer, evaluated against a fixed policy and a held-out task suite. It does not directly train the small multimodal network or make its probabilities calibrated.

### AlphaEvolve and DGM — useful foundations

[AlphaEvolve, official research account](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/); [DGM, inspected March 2026 revision](https://arxiv.org/abs/2505.22954).

AlphaEvolve motivates evaluator-backed program search; DGM motivates preserving different agent lineages that may become useful stepping stones. These are older sources that remain conceptually relevant. Our archive policy is provisional and needs an equal-budget comparison with a simple greedy method.

### Counterevidence and taxonomy

[Open-ended AI research case studies](https://arxiv.org/abs/2607.27191) report engineering success alongside failure to resolve the central research questions in two evaluated cases. We therefore require an explicit answer to each research question and counterexamples, not just working software or a polished report.

[September RSI survey, inspected Sep 22 revision](https://arxiv.org/abs/2609.11873) helps distinguish autonomy levels. It is a taxonomy and collection of evidence, not an empirical demonstration that all the levels have been solved. We use it for terminology only.

## Changes from the initial autoresearch plan

| Initial idea | Adopted method |
|---|---|
| Follow the current best program | Keep several complementary model families and their ancestry |
| Read a scalar result | Record slice metrics, learning curves, failure cases, and resource costs |
| Propose another edit | Retrieve relevant prior evidence and state a falsifiable hypothesis |
| Treat every win as progress | Repeat promising results with fresh seeds and longer budgets |
| Assume the researcher improves too | Evaluate researcher changes separately on new task families |
| Continue indefinitely | Use a finite campaign and report unresolved questions at its boundary |

No performance gain from these adaptations is claimed yet. “More elaborate” and “more recent” are not substitutes for the equal-budget control.

## Applying the method now

The current pass followed a broad-to-targeted sequence: identify architecture/objective/evaluation families, inspect primary methods for the consequential choices, check contrary evidence, and retain unresolved propositions in a structured state file. The resulting recommendation changes when evidence contradicts it: FiLM remains a serious contender; JEPA remains an objective challenger; a meta-loop must earn its extra cost.

The next stage must produce a correct dataset and evaluator. Once a training trial exists, the methodology can become an executable archive/search loop. Until then, adding a research framework cannot experimentally identify the best model algorithm.
