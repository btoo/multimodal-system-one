# Decision register

Latest decision: **MiniCPM-o 4.5 is the provisional v4 research backbone**, with Qwen3-Omni as the accuracy reference. The [matched eight-checkpoint investigation](../../reports/v4-selection-v1/README.md) found no release-qualified winner: screen understanding remains below the declared gate. Preserve the direct probability interface and investigate shared-state question branches; a controlled H100 probe preserved 16/16 top decisions with a 6.53× reduction in the prepared language/scoring phase. No v4 model was promoted to the playground.

The table below records the earlier small-model control decisions. The [original baseline results](../../reports/pilot-v1/README.md), [reading map](reading-map.md) and [frontier review](frontier-methods.md) remain historical evidence.

| Decision | Rationale | What could change it |
|---|---|---|
| Make speech/audio and screen evals first-class | Matches the intended computer/browser-use application | Refine task priority from application evidence |
| Keep from-scratch controls and a practical pretrained track | Preserves fundamentals while giving real speech/screens a useful starting point | Compare learning curves, cost, and transfer honestly |
| Start with A2, compare A0/A1/A3 | Simple tokens are easy to inspect; FiLM and bottlenecks are credible alternatives | Equal-budget trials show better quality/resource tradeoffs |
| Use direct candidate probabilities | Matches the bounded decision task | Requirements expand to free-form explanations or generated media |
| Use a shared independent candidate scorer | Permutation equivariance and variable answer sets | Set-relative reasoning needs cross-candidate interaction |
| Use NLL first | Direct outcome supervision and meaningful probabilities | A controlled comparison favors Brier or an ordinal objective |
| Defer auxiliary JEPA/contrastive losses | Their benefit must exceed added training cost | Label scarcity or transfer tests provide a reason to pretrain |
| Defer RL | The first dataset supplies full labels | Delayed outcomes or partial feedback become central |
| Fit temperature only after selection | Avoid tuning the network to calibration data | A separate calibrated-training experiment is preregistered |
| Use PyTorch MPS first | Familiar debugging and CPU reference path | Measured equivalent MLX implementation offers a useful improvement |
| Retain several experiment lineages | Different families can be useful stepping stones | A matched-cost greedy control is equally good with less machinery |
| Use ShinkaEvolve as runner candidate | Source exposes needed archive/budget capabilities | Local smoke test fails or dependencies dominate the small study |
| Hold the research policy fixed initially | Establish a stable baseline before meta-optimization | Multiple task suites exist and a separate method study is budgeted |

## Claims deliberately left open

- A transformer might not be the best small visual reasoning model at the selected data scale.
- A bottleneck's asymptotic advantage might not yield a latency win on this MPS workload.
- Low NLL on generated scenes might not transfer to real photos or unfamiliar instructions.
- Confidence may degrade under new combinations even when accuracy remains useful.
- Archive search might cost more than the gains it produces in a small search space.
- A research method that improves one benchmark might not improve future research efficiency.

## Scope of the synthetic control

The model must show a learnable link from pixel evidence and question semantics to correct bounded answers. Both modalities must matter on deliberately balanced tasks. The report must include failures, resource measurements, and unadjusted probability quality. It need not demonstrate open-world knowledge, arbitrary natural language, real-time robotics, a world model, or self-accelerating AI research.

This is a scope decision for the first experiment, not a limit on the longer-term project.

The application proof is governed by [audio-screen-evals.md](audio-screen-evals.md). It requires real-speech, acoustic-event, visual, and joint evaluations; the synthetic control cannot satisfy it.
