# Benchmark Findings

This document summarizes the main experimental findings from the Crossy Road RL benchmark through environment **v9**.

The goal is not to present a finalized paper narrative, but to consolidate the strongest observations from the completed experiments and use them to guide the next stage of benchmark development.

---

## 1. Observation Representation Strongly Affects Learnability

The clearest representation result comes from comparing **v4** and **v5**.

These environments use the same underlying road-crossing dynamics, but differ in how the state is presented to the agent:

- **v4:** global observation
- **v5:** compact local observation spanning one row behind, the current row, and two rows ahead

### Final 1M-step success rates

| Algorithm | v4 Global | v5 Local2 |
|---|---:|---:|
| PPO | 7.0% | 79.8% |
| TRPO | 6.2% | 94.0% |
| DQN | 10.2% | 68.6% |
| QR-DQN | 12.0% | 56.0% |

All four algorithms improve substantially under the local representation.

This suggests that the larger global state does not necessarily provide a more useful learning signal. Instead, the compact local representation appears to provide a stronger inductive bias by focusing the policy on immediately relevant hazards.

A central result of the benchmark is therefore:

> More information is not necessarily better for reinforcement learning if the additional state increases representational complexity without improving the immediate decision context.

---

## 2. The Best Observation Horizon Depends on the Algorithm

Environment **v6** holds the underlying world fixed while varying the observation horizon.

The tested configurations are:

- **local1:** one row behind, current row, one row ahead
- **local2:** one row behind, current row, two rows ahead
- **local3:** one row behind, current row, three rows ahead
- **global:** full state

### Final 1M-step success rates

| Algorithm | Local1 | Local2 | Local3 | Global |
|---|---:|---:|---:|---:|
| PPO | 49.0% | 79.8% | 64.4% | 7.0% |
| TRPO | 87.8% | 94.0% | 82.4% | 6.2% |
| DQN | 64.0% | 68.6% | 54.6% | 10.2% |
| QR-DQN | 62.2% | 56.0% | 47.4% | 12.0% |

PPO, TRPO, and DQN perform best with **local2**, while QR-DQN performs best with **local1**.

This indicates that observation design interacts with the learning algorithm itself.

There is no single observation horizon that is optimal for every algorithm.

---

## 3. Recurrence Does Not Improve PPO in the Limited-Observation Setting

Recurrent PPO was evaluated on the local1 version of v6 to test whether memory could compensate for the smaller observation horizon.

The optimized recurrent configuration used:

- `MlpLstmPolicy`
- learning rate: `3e-4`
- entropy coefficient: `0.02`
- LSTM hidden size: `64`
- 10 training epochs

### Five-seed recurrent PPO learning curve

| Steps | Success Rate |
|---:|---:|
| 200k | 4.6% ± 4.8% |
| 400k | 8.0% ± 5.0% |
| 600k | 16.6% ± 8.4% |
| 800k | 23.8% ± 14.3% |
| 1M | 18.8% ± 9.9% |

Feedforward PPO reaches **49.0% ± 27.8%** on the same local1 environment.

Under the tested setup, recurrence therefore does not improve PPO.

This experiment is treated as a completed negative result rather than an area requiring further tuning.

---

## 4. Procedural Layout Variation Dramatically Increases Difficulty

Environment **v7** introduces procedural road placement while retaining the local2 observation.

Instead of having road hazards at fixed absolute rows, each episode samples four road rows subject to structural constraints.

Difficulty increases with the order in which hazards are encountered rather than with absolute row index.

This removes a useful regularity from the fixed environment: the agent can no longer infer the type of upcoming challenge purely from absolute position.

### Fixed vs procedural roads at 1M

| Algorithm | v5 Fixed Roads | v7 Procedural Roads | Change |
|---|---:|---:|---:|
| PPO | 79.8% | 23.6% | -56.2 pp |
| TRPO | 94.0% | 48.6% | -45.4 pp |
| DQN | 68.6% | 3.2% | -65.4 pp |
| QR-DQN | 56.0% | 12.6% | -43.4 pp |

Proceduralization causes a large performance drop for every algorithm.

The effect is particularly severe for DQN, which falls from 68.6% to 3.2%.

This suggests that fixed-layout environments allow policies to exploit spatial regularities that disappear once the layout changes between episodes.

---

## 5. Procedural Training Transfers Well to New Seeds from the Same Generator

The v7 models were also evaluated on held-out reset seeds from `10000–10099`.

### 1M-step results

| Algorithm | Standard Eval | Held-out Seeds |
|---|---:|---:|
| PPO | 23.6% | 26.2% |
| TRPO | 48.6% | 49.0% |
| DQN | 3.2% | 2.4% |
| QR-DQN | 12.6% | 14.8% |

There is essentially no performance gap.

This indicates that once trained on the procedural distribution, the learned policies transfer well to previously unseen reset seeds sampled from the **same procedural generator**.

This should not be interpreted as out-of-distribution generalization.

Instead, it shows successful transfer to new samples from the same environment distribution.

---

## 6. Mechanical Diversity Does Not Necessarily Increase Difficulty

Environment **v8** introduces two qualitatively different hazard mechanics:

- roads with moving cars
- rivers with moving platforms

The layout is fixed:

- road at row 1
- road at row 3
- river at row 5
- river at row 7

The observation remains local2 and explicitly encodes row type.

### Final 1M-step results

| Algorithm | Success | Road Collision | Drowning |
|---|---:|---:|---:|
| PPO | 94.0% ± 4.2% | 2.8% | 3.2% |
| TRPO | 100.0% ± 0.0% | 0.0% | 0.0% |
| DQN | 80.6% ± 9.5% | 2.8% | 16.6% |
| QR-DQN | 96.2% ± 2.0% | 1.2% | 2.6% |

Despite requiring two different control strategies, v8 is highly learnable.

TRPO completely solves the task across all five seeds, while PPO and QR-DQN approach saturation.

This demonstrates that simply introducing more mechanics does not necessarily make an RL task harder.

---

## 7. Procedural Mixed Mechanics Are Substantially Harder Than Fixed Mixed Mechanics

Environment **v9** combines:

- procedural hazard placement
- road and river mechanics
- procedural assignment of hazard type
- local2 observation

Each episode contains four hazards, with at least one road and at least one river.

The agent therefore cannot rely on a fixed mapping between absolute row and hazard type.

It must inspect the observed row type and select an appropriate strategy.

### Sanity baselines

| Policy | Success | Avg. Max Row |
|---|---:|---:|
| Random | 0% | 2.12 |
| Mixed-hazard heuristic | 32% | 5.91 |

The heuristic produces both road and river failures, confirming that both mechanics affect task difficulty.

### Final 1M-step results

| Algorithm | Success | Avg. Max Row | Road Collision | Drowning | Timeout |
|---|---:|---:|---:|---:|---:|
| PPO | 50.0% ± 7.6% | 7.30 | 30.8% | 18.6% | 0.6% |
| TRPO | 49.0% ± 6.8% | 7.30 | 29.6% | 20.6% | 0.8% |
| DQN | 46.4% ± 7.8% | 6.99 | 33.2% | 18.6% | 1.8% |
| QR-DQN | 33.6% ± 6.3% | 6.59 | 35.4% | 27.2% | 3.8% |

### Fixed vs procedural mixed mechanics

| Algorithm | v8 Fixed Mixed | v9 Procedural Mixed | Change |
|---|---:|---:|---:|
| PPO | 94.0% | 50.0% | -44.0 pp |
| TRPO | 100.0% | 49.0% | -51.0 pp |
| DQN | 80.6% | 46.4% | -34.2 pp |
| QR-DQN | 96.2% | 33.6% | -62.6 pp |

The large drop from v8 to v9 reinforces the conclusion that **procedural variation is a major source of difficulty**.

---

## 8. Mixed Mechanics Can Actually Improve Performance in a Procedural Environment

The comparison between **v7** and **v9** is especially important.

Both environments use procedural layouts.

The key difference is:

- v7 contains only roads
- v9 contains roads and rivers

### Final 1M-step success rates

| Algorithm | v7 Procedural Roads | v9 Procedural Mixed |
|---|---:|---:|
| PPO | 23.6% | 50.0% |
| TRPO | 48.6% | 49.0% |
| DQN | 3.2% | 46.4% |
| QR-DQN | 12.6% | 33.6% |

Adding a second mechanic does not make the procedural task uniformly harder.

Instead:

- PPO improves substantially
- DQN improves dramatically
- QR-DQN improves
- TRPO remains almost unchanged

A likely explanation is that later road hazards in v7 are especially difficult, while some river hazards in v9 are easier despite requiring a different control rule.

This reinforces the idea that task difficulty depends on the **structure of the mechanics**, not simply on the number of mechanics present.

---

## 9. Algorithms Differ in Sample Efficiency

The v9 learning curves reveal substantial differences in how quickly the algorithms acquire useful policies.

### v9 success rates

| Algorithm | 200k | 400k | 600k | 800k | 1M |
|---|---:|---:|---:|---:|---:|
| PPO | 29.6% | 34.2% | 41.4% | 46.6% | 50.0% |
| TRPO | 28.6% | 35.2% | 42.2% | 44.0% | 49.0% |
| DQN | 3.8% | 14.2% | 25.6% | 40.6% | 46.4% |
| QR-DQN | 0.2% | 15.8% | 20.2% | 27.2% | 33.6% |

PPO and TRPO acquire useful behavior much earlier.

DQN initially performs very poorly, but continues improving and nearly catches PPO and TRPO by 1M steps.

This suggests a meaningful distinction between:

- early sample efficiency
- eventual performance

---

## 10. Failure Modes Differ by Algorithm

The v8/v9 failure-mode analysis separates episode outcomes into:

- success
- road collision
- drowning
- timeout

In v8, most algorithms almost completely eliminate both failure modes.

DQN is the main exception, with a residual **16.6% drowning rate** at 1M.

In v9, failures remain substantial for every algorithm.

At 1M:

| Algorithm | Road Collision | Drowning | Timeout |
|---|---:|---:|---:|
| PPO | 30.8% | 18.6% | 0.6% |
| TRPO | 29.6% | 20.6% | 0.8% |
| DQN | 33.2% | 18.6% | 1.8% |
| QR-DQN | 35.4% | 27.2% | 3.8% |

Road collisions are the dominant residual failure for PPO, TRPO, and DQN.

QR-DQN struggles substantially with both mechanics.

Timeout rates are low at 1M, indicating that the dominant failures come from hazard interaction rather than simple inactivity.

---

# Consolidated 1M-Step Benchmark

| Environment | PPO | TRPO | DQN | QR-DQN |
|---|---:|---:|---:|---:|
| v3 — Simple Fixed Roads | 57.8% ± 24.5% | 74.2% ± 4.1% | 0.0% ± 0.0% | 81.8% ± 5.2% |
| v4 — Harder Global | 7.0% ± 0.7% | 6.2% ± 0.8% | 10.2% ± 3.6% | 12.0% ± 2.8% |
| v5 — Fixed Local2 | 79.8% ± 15.3% | 94.0% ± 5.2% | 68.6% ± 12.5% | 56.0% ± 20.8% |
| v7 — Procedural Roads | 23.6% ± 6.3% | 48.6% ± 23.9% | 3.2% ± 4.4% | 12.6% ± 6.4% |
| v8 — Fixed Mixed | 94.0% ± 4.2% | 100.0% ± 0.0% | 80.6% ± 9.5% | 96.2% ± 2.0% |
| v9 — Procedural Mixed | 50.0% ± 7.6% | 49.0% ± 6.8% | 46.4% ± 7.8% | 33.6% ± 6.3% |

---

# Current Conclusions

The completed experiments support several recurring observations.

## Representation matters

Compact local observations can substantially outperform larger global states.

## More information is not always beneficial

Performance decreases when the observation horizon becomes unnecessarily large.

## Observation design interacts with the algorithm

Different algorithms prefer different context horizons.

## Recurrence is not automatically beneficial

Recurrent PPO did not improve performance under the tested limited-observation setting.

## Procedural variation is a major source of difficulty

Both road-only and mixed-mechanics tasks become substantially harder when hazard placement changes between episodes.

## Mechanical diversity is not equivalent to difficulty

Mixed road/river mechanics are highly learnable in a fixed layout and can even improve performance relative to a procedural road-only environment.

## Algorithm rankings are environment-dependent

No single algorithm dominates across all tasks.

## Learning speed and final performance are distinct

PPO and TRPO often learn useful behavior earlier, while DQN sometimes improves substantially later in training.

## Failure composition provides information beyond success rate

Road collisions, drowning, and timeouts reveal qualitatively different weaknesses that are hidden by aggregate success alone.

---

# Open Questions

The next stage of the benchmark should focus on questions that are not yet resolved by v3–v9.

Promising directions include:

1. **Distribution shift**
   - train under one procedural distribution and evaluate under a modified one

2. **Cross-environment transfer**
   - initialize or evaluate policies across related environments

3. **Stochastic dynamics**
   - vary hazard speeds, transition behavior, or observation reliability

4. **Robustness**
   - test whether successful policies remain stable under controlled perturbations

5. **Longer-horizon generalization**
   - increase environmental scale while preserving learned mechanics

These directions may provide more information than simply adding another incrementally harder fixed environment.

---

# Figures

Current cross-environment figures are stored under:

```text
results/figures/cross_environment/

including:

v4_vs_v5_observation.png
v5_vs_v7_procedural_roads.png
v7_vs_v9_procedural_mechanics.png
v8_vs_v9_procedural_mixed.png
v8_v9_failure_modes.png

