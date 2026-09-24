# Research protocol

Version 1 — research/design stage. This is a project-specific protocol drawing on the [frontier review](docs/research/frontier-methods.md). It is not executable training software or a claim to reproduce the cited systems.

## Objective

Find a small native image–text decision model whose probabilities, modality use, generalization, and resource costs are measured on a controlled task. Improve the research procedure only through a separately evaluated method-level study.

The present repository contains documentation and figure tools. Do not manufacture benchmark rows, trained weights, or a winner. When implementation begins, build the evaluator and a baseline before integrating an evolution runner.

## Literature and theory phase

1. Read the README, decision register, experiment contract, and current research state.
2. For a consequential claim, identify whether it is a source finding, an analytical derivation, or a project hypothesis.
3. Follow primary sources to the relevant methods and limitations. Record publication/revision dates and reading depth. Treat repository activity as maintenance evidence, not performance evidence.
4. Check contradictory evidence. Preserve uncertainty when a paper's scale, dataset, or evaluation does not match this project.
5. Update the research-state record with a concrete follow-up experiment. A new source is useful if it changes a design decision or its falsification test.
6. Finish this phase with a reviewable design and unresolved questions. Literature alone cannot select an empirically best architecture.

## Implementation phase prerequisites

- The renderer, oracle, split policy, metrics, and timing boundaries exist and pass the mechanical checks in the experiment contract.
- A baseline can train, save, reload, and evaluate on CPU/MPS as applicable.
- A short pilot determines feasible data size, runtime, precision, and resource gates. Freeze them before comparative trials.
- The campaign records task/evaluator hashes and explicit total compute and proposer-call budgets. The default manifest is disabled because no runner exists yet.
- If ShinkaEvolve is adopted, pin a tested revision and run a deterministic, provider-free integration check first. Keep API model selection and spending explicit in the campaign rather than inheriting library defaults.

## Model-search loop

Use a dedicated experiment branch/worktree for candidate changes. Preserve the working tree and other work. An archive is an append-only evidence record; it is not permission to destructively reset unrelated files.

For each permitted trial:

1. **Choose a parent.** Consult the small active archive, not only the most recent winner. Reserve some budget for an alternative family. Exact repeats require an intentional seed/budget reason.
2. **State a hypothesis.** Record the proposed change, expected mechanism, predicted metric effect, uncertainty, and the source or prior run that motivates it.
3. **Implement one attributable change.** Architecture, optimizer, or training configuration can vary. The task semantics, evaluation, split manifests, and resource accounting stay fixed.
4. **Record provenance before execution.** Source/config hashes, parent IDs, campaign version, initialization seed, training-data seed, environment, and budget.
5. **Run one controlled trial.** Use synchronized accounting. Separate setup, training, evaluation, and total elapsed time. Count failed trials against the campaign's resource budget.
6. **Validate the result.** Check finiteness, output invariants, metric consistency, unmodified frozen files, and resource gates. A changed evaluator invalidates the comparison.
7. **Analyze the evidence.** Store improvements, regressions, counterexamples, and likely explanations. An explanation is a hypothesis unless a controlled intervention supports causation.
8. **Update the archive.** Retain successful and failed records. Maintain a small parent pool with family diversity and quality/resource tradeoffs. Do not erase an unpromising branch's evidence.
9. **Confirm before promotion.** A screening winner is provisional. Use the prescribed longer-budget fresh-seed comparison before choosing the model recipe.
10. **Stop at the declared cap.** Report the best supported result, failed gates, and unanswered research questions. Do not silently extend the run or change criteria to manufacture a winner.

## Evidence and memory rules

Every retained lesson links to a run ID or source. Separate facts from interpretations. Reconcile contradictory findings by comparing task distribution, budget, seeds, and environment. Do not carry a hypothesis forward as a verified fact just because it appears in an earlier note.

The research-state file is project data, not global agent memory. Updates here do not modify the user's personal memory files.

## Model selection and final evaluation

Choose the nominee using development data. Fit postprocessing on calibration data after freezing the recipe. Keep final examples out of hypothesis generation. Evaluate every preregistered final seed and report the distribution, not the best seed. A final-test-triggered redesign starts a new evaluation cycle.

The evaluator should be independently invoked by the trusted harness. Hash checks catch accidental mutation but are not a security boundary: candidate code must not be allowed to read protected final data or replace evaluator outputs. Build actual process/input separation before claiming isolation.

## Optional method-level loop

This is disabled for the first model study. To activate it later:

- Define a separate budget and a suite of development and held-out task families.
- Version the proposer instructions, selection rule, analysis policy, and context handling separately from model/trainer code.
- Hold the base proposer model and permissions constant across comparisons. Include training, evaluation, tokens, failed attempts, and end-to-end cost.
- Freeze each candidate research policy while evaluating its ability to improve new tasks.
- Promote a method only when repeated trials support improved discovery efficiency and transfer. Preserve negative results.

Changing the research policy and the model simultaneously would confound the comparison. Neither self-reported confidence nor a single fortunate trial is evidence of recursive improvement.
