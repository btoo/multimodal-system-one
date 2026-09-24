# Proposed experiment contract

Status: design only. The implementation, evaluator, and research runner do not yet exist. Numbers below are proposed settings, not measured outcomes. The profiling pilot may revise them once, before the comparative campaign is frozen.

## Research questions

1. Does the model use both pixels and question semantics to predict bounded answers?
2. Which of A0 late fusion, A1 FiLM, A2 joint tokens, and A3 latent fusion gives the best quality within the laptop budget?
3. Are its probabilities useful on held-out scenes, new combinations, and corrupted observations?
4. Does an archive of diverse candidates improve discovery efficiency over a simple greedy loop? This needs its own matched-budget method comparison and is not settled by the architecture campaign.

## Stage 0: establish trustworthy measurement

Implement an original 2D renderer and independent oracle. Start with 3–6 shapes, a fixed palette, and unambiguous center-based spatial relations. Rendering takes scene records; the oracle computes labels from those records without reading the image. The model receives only pixels, question/candidate text, and explicit modality-presence flags.

The data generator must reject ambiguous questions, duplicate semantic answers, and boundary cases that violate the declared relation convention. Use functional question programs internally for label generation; do not expose them to the model. Include at least existence, attribute selection, and one-hop relation tasks. Counting to a fixed maximum is a later score-head slice.

Split scene families first. A family includes the base scene, counterfactual variants, render perturbations, and associated question variants. Assign all members to the same split. Content-hash renders and metadata after excluding split IDs to catch accidental duplication. Fit the tokenizer on training text only. Seed namespaces must be distinct, but seed separation alone is insufficient.

Proposed initial data budget: 20,000 training scene families, 2,000 development, 2,000 calibration, and 4,000 final-test families, with several questions per family. Report exact examples and label/template histograms. This may shrink during the pilot if data-generation or evaluation cost dominates. The final-test family specification is frozen before model selection, with actual final examples kept out of the search context.

## Stage 1: implementation checks

These checks establish mechanics, not model usefulness:

- Fit one tiny batch to near-zero task loss and verify that the image stem and text path receive nonzero gradients on a task requiring both.
- Compare CPU and MPS forward/loss values with explicit tolerances and the same weights. Start in float32.
- Check masked padding, variable candidate counts, batch independence, finite logits, valid normalization, and checkpoint save/reload.
- Check deterministic candidate permutation at evaluation time. The numerical tolerance is proposed as absolute probability difference ≤ `1e-5`, adjusted only if justified by the numerical reference test.
- Check that appending an unrelated question does not change an existing answer beyond the numerical tolerance. No cross-question attention or batch-statistic dependence is allowed.
- Verify metric code on exact distributions: correct deterministic forecasts, uniform forecasts, wrong confident forecasts, and known Bernoulli mixtures.
- Keep exact ties explicit; final-label selection must not become dependent on candidate array order.

## Stage 2: bounded architecture screen

Use four model families, three fixed screening seeds per family, and 300 seconds per run. Choose a neutral baseline configuration for each family rather than starving the convolutional controls to match transformer parameter count. Report both parameter counts and examples processed. The primary comparison is equal time; a finalist follow-up can use matched examples to distinguish throughput gains from statistical efficiency.

Timing begins at the first optimizer step and includes all optimizer work and input loading. A fixed pre-timing inference-only warmup may allocate kernels; it must not update parameters. Synchronize the device at timing boundaries. Allow at most one completed step of budget overshoot and record it. Do not permit variable numbers of uncounted training warmup steps.

Log setup, preprocessing, training, evaluation, and end-to-end elapsed time separately. Cache generation is fixed before trials. Record dtype, framework version, OS, power mode, hardware, and any active resource pressure. Reorder architecture trials across seeds to reduce thermal-order bias.

Proposed resource gates: at most 12 GiB process resident memory and 12 GiB reported MPS driver allocation, tracked separately; at most 100 ms warm p95 input-to-probability latency on the fixed v1 workload. These are feasibility targets to check and freeze in the pilot, not promised performance. All candidates use the same gates after freezing. Record tensor memory too, without equating it to driver/process/physical memory.

## Stage 3: evidence-driven refinement

Allocate eight additional 300-second trials across archived candidates. Record the parent, hypothesis, expected mechanism, configuration diff, and relevant source or failure case. Include a reserved exploratory proposal instead of spending every slot on the current best scalar score. Exact repeat configurations are skipped unless they intentionally add a seed or longer budget.

The archive stores every attempted result, including crashes. The active parent pool retains the best feasible representative of each architecture plus non-dominated quality/latency/memory candidates, subject to a small cap such as 12. This is our proposed policy, not an assertion that it implements ShinkaEvolve's published algorithm verbatim.

A run that changes evaluation code, split semantics, or the time accounting is invalid for this campaign. Fix evaluator defects in a separately versioned benchmark and rerun affected baselines. Invalid or crashed trials have null metrics; zero loss must never encode a failure.

## Stage 4: confirmation and final evaluation

Nominate one model on development evidence and compare it with the preregistered A2 baseline. Run five fresh seeds per configuration for 1,800 seconds each. If A2 itself wins, use the strongest alternative as the paired comparator and label that substitution. Selection of the nominee must happen before confirmation outcomes are inspected.

Report every seed, the paired development difference, scene-cluster uncertainty intervals, and between-seed variation. Five seeds give a limited estimate; a nominal confidence interval is not proof against all selection bias. Require the improvement to be consistent with the practical effect threshold established by baseline repeat noise. If it is inconclusive, report inconclusive and keep the simpler default.

Fit one temperature per declared primitive family on calibration data after the model/training recipe is frozen. If the sample size is too small, use one global temperature rather than fit many parameters. Choose any threshold from a preregistered grid using calibration data and a stated accepted-error objective. Freeze both before the final test.

Evaluate all five confirmed seeds without choosing the best final-test run. A separately predeclared seed supplies the demo checkpoint. If the final test suggests another design change, its findings are exploratory evidence for a new campaign, not a reason to quietly rerun selection.

## Metrics

For task families `t` with `n_t` examples and a declared exhaustive answer set of size `K_i >= 2`:

$$
S=\frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}\frac{1}{n_t}
\sum_{i\in t}\frac{-\log p_\theta(y_i\mid x_i,q_i,C_i)}{\log K_i}.
$$

All families have fixed nonzero evaluation counts. Evaluate BCE as a two-class NLL; an ordered categorical head uses its declared level count. Weights and cardinalities are task metadata, not outcome-dependent scores. Lower is better. Uniform predictions score 1. The metric does not measure open-set detection and is not comparable after changing the task mixture.

Also report:

| Metric | Definition / use |
|---|---|
| Accuracy | Per task, macro mean, and raw totals |
| NLL | Mean negative log probability in nats; stable log-softmax computation |
| Brier | Mean `sum_k (p_k - 1[y=k])²`; document this multiclass normalization |
| ECE | Ten fixed equal-width top-confidence bins; show counts and reliability points |
| Boolean reliability | Bin `p_true` and compare with positive-event frequency |
| Risk / coverage | Accepted-case error versus fraction accepted; empty accepted sets yield undefined risk |
| Score quality | Distribution NLL plus absolute error of expected rubric level |
| Candidate symmetry | Max and percentile mapped probability differences over permutations |
| Query isolation | Existing-query difference after adding independent queries |
| Efficiency | Batch-one cold and warm p50/p95; throughput; multiple-query scaling; memory |

Latency workload: one 64×64 image, 32 padded question tokens, and four eight-token candidates. Use a fixed workload plus a representative variable-length distribution, 50 inference warmups and 500 timed examples, with preprocessing and device synchronization included. Report both. Multiple-question batches of 1, 4, and 16 quantify scaling rather than assuming constant cost.

## Required diagnostic slices

| Slice | Construction | Failure it exposes |
|---|---|---|
| In-distribution | New scene families from training grammar/distribution | Basic generalization |
| New attribute combinations | Disjoint composed tuples, primitives still seen | Memorized color/shape associations |
| New phrasing | Held-out templates and paraphrase structure | Template dependence |
| Relation reversal | Paired image/question interventions | Bag-of-words shortcuts |
| Relevant visual edit | Change one answer-determining property | Insensitivity to pixels |
| Irrelevant edit | Change only distractor properties | Spurious sensitivity |
| Text-only / image-only trained controls | Independent controls with matching data budgets | Unimodal leakage or dataset biases |
| Shuffled modality at evaluation | Shuffle between distinct scenes | Model dependence; interpret alongside controls |
| Candidate permutation | Same descriptions and IDs, changed list order | Position bias or serialization defects |
| Opaque ID replacement | Rename keys without changing descriptions | Key-name shortcut |
| Corruption and missing input | Explicit severity/presence values | Confidence under lost evidence |
| Known uncertainty | Finite hidden-state mixtures with exact outcome probabilities | Honest belief estimation when evidence is ambiguous |

For uncertainty tasks, the oracle computes the distribution conditional on the *observed* input. A hard label from an occluded full scene can be a sampled outcome, but it is not evidence that the visible input uniquely determines the answer. Balanced examples alone do not solve this distinction.

Do not rebalance realized outcomes inside a known-probability mixture: that changes the conditional distribution being estimated. Balance task/template coverage while preserving each mixture's declared sampling process, and report evaluation prevalence.

## Minimum evidence for the first proof

Proposed gates, finalized after pilot and before comparison:

1. At least 90% macro in-distribution accuracy and 75% on the declared novel-combination slice; report per-task scores and chance rates.
2. At least a 10-percentage-point advantage over the strongest trained unimodal control on the designated jointly necessary subset, with scene-cluster uncertainty reported.
3. Structural candidate-order and query-isolation checks pass. Label/ID mapping and tie handling are correct.
4. Raw and calibrated NLL/Brier/reliability are reported on the final test; calibration is not declared solved from ECE alone. Abstention coverage and accepted error are reported even when the desired target fails.
5. The frozen pilot resource gates are met, or the resource failure is reported and the proof remains incomplete.

These thresholds are project acceptance choices, not results from the literature. A failed gate is a useful research result and is retained.

## Research-policy study — a separate later campaign

Once the task evaluator is stable, compare a simple greedy controller with the archive/analysis controller. Use the same base proposer model, task starting points, tool permissions, total training seconds, proposal-call budget, and token/cost caps. Measure improvement relative to each task's starting baseline after a fixed budget, including failed attempts.

Use development *task families* to tune the controller and different final task families to assess transfer. Example families can vary modality, relation grammar, data scarcity, and corruption structure; individual examples alone are not a method-level holdout. Freeze the controller during each evaluation. Compare several independent search seeds. Count researcher tokens and evaluation overhead in addition to GPU time.

Only after a controller repeatedly produces transferable gains should changing that controller become the object of a meta-loop. A new model checkpoint, better prose, or a higher score on one tuned task does not establish improved research ability.
