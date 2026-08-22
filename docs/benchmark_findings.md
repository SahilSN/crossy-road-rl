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


---

# Distribution-Shift Robustness

After completing the v9 procedural mixed-mechanics benchmark, we evaluated whether the trained policies remained effective when the evaluation environment differed systematically from the training distribution.

All distribution-shift experiments use the existing v9 models trained under the standard environment. No retraining or fine-tuning is performed.

The purpose is therefore to measure out-of-distribution robustness rather than adaptation.

Two shift dimensions have been evaluated so far:

1. hazard speed
2. hazard composition

---

## 11. Evaluation-Time Hazard Speed Shift

The first distribution-shift experiment modifies the speed of all moving hazards while keeping the rest of the v9 environment unchanged.

The training distribution corresponds to:

```text
speed_scale = 1.0
```

Evaluation is performed at:

```text
0.8x
1.0x
1.2x
1.4x
```

where the multiplier is applied to both:

- road-car velocity
- river-platform velocity

The same five trained seeds for PPO, TRPO, DQN, and QR-DQN are evaluated under every condition.

### Final 1M-step success rates

| Algorithm | 0.8x | 1.0x | 1.2x | 1.4x |
|---|---:|---:|---:|---:|
| PPO | 49.0% ± 4.2% | 50.0% ± 7.6% | 38.6% ± 6.9% | 27.8% ± 5.8% |
| TRPO | 48.4% ± 10.9% | 49.0% ± 6.8% | 40.6% ± 7.5% | 28.6% ± 6.2% |
| DQN | 50.2% ± 9.7% | 46.4% ± 7.8% | 37.0% ± 4.9% | 31.0% ± 5.6% |
| QR-DQN | 32.2% ± 3.3% | 33.6% ± 6.3% | 27.2% ± 8.2% | 27.0% ± 2.4% |

### Main finding

The robustness profile is asymmetric.

Slower hazards produce little degradation:

- PPO: 50.0% -> 49.0%
- TRPO: 49.0% -> 48.4%
- QR-DQN: 33.6% -> 32.2%
- DQN improves from 46.4% to 50.2%

Faster hazards produce a much larger performance drop.

At 1.4x speed:

- PPO falls to 27.8%
- TRPO falls to 28.6%
- DQN falls to 31.0%
- QR-DQN falls to 27.0%

This suggests that policies trained at the standard timing distribution tolerate slower dynamics relatively well, but are substantially less robust when the decision window becomes shorter.

### Algorithm ranking under strong speed shift

The standard v9 ranking is approximately:

```text
PPO ~= TRPO > DQN > QR-DQN
```

At 1.4x speed, the algorithms converge much more closely:

```text
DQN      31.0%
TRPO     28.6%
PPO      27.8%
QR-DQN   27.0%
```

DQN therefore has the highest mean success rate under the strongest tested speed shift, despite not having the highest in-distribution performance.

This suggests that in-distribution success and distribution-shift robustness are not necessarily aligned.

---

## 12. Failure Modes Under Speed Shift

The speed-shift experiment was also analyzed by decomposing episode outcomes into:

- success
- road collision
- drowning
- timeout

The dominant change under faster speeds is an increase in road-collision failures.

For PPO and TRPO especially, the reduction in success from 1.0x to 1.4x is primarily transferred into the road-collision category.

Drowning rates remain comparatively more stable.

This suggests that road crossing is more sensitive to timing compression than river traversal.

A plausible interpretation is that faster cars reduce the available safe crossing interval, invalidating learned timing policies more aggressively.

River traversal also depends on motion timing, but the platform-support mechanic appears comparatively more tolerant to the tested speed changes.

The speed-shift result therefore indicates not only a reduction in aggregate robustness, but also a mechanic-specific failure pattern:

> Faster evaluation-time dynamics primarily disrupt road-crossing timing, while river-handling behavior is comparatively more stable.

---

## 13. Hazard-Composition Shift

The second distribution-shift experiment changes the relative frequency of road and river hazards while preserving:

- total number of hazards
- procedural hazard placement
- standard hazard speeds
- local2 observation
- the same trained policies

Three evaluation conditions are used.

### Standard

The original v9 composition generator.

Possible mixed compositions include:

```text
1 road / 3 rivers
2 roads / 2 rivers
3 roads / 1 river
```

### Road-heavy

Every episode contains exactly:

```text
3 roads
1 river
```

### River-heavy

Every episode contains exactly:

```text
1 road
3 rivers
```

No retraining is performed.

---

## 14. Composition-Shift Results

### Final 1M-step success rates

| Algorithm | Standard | Road-heavy | River-heavy |
|---|---:|---:|---:|
| PPO | 50.0% ± 7.6% | 39.2% ± 5.1% | 57.2% ± 4.9% |
| TRPO | 49.0% ± 6.8% | 43.0% ± 7.8% | 58.4% ± 4.7% |
| DQN | 46.4% ± 7.8% | 34.2% ± 8.0% | 61.6% ± 8.5% |
| QR-DQN | 33.6% ± 6.3% | 20.0% ± 5.4% | 46.4% ± 7.1% |

The direction of the effect is consistent across every algorithm:

```text
road-heavy
    ↓ performance

river-heavy
    ↑ performance
```

---

## 15. Road Hazards Are the More Difficult v9 Mechanic

The composition-shift experiment provides direct evidence that the road mechanic contributes more strongly to v9 difficulty than the river mechanic.

Increasing road frequency decreases success for every algorithm.

Increasing river frequency improves success for every algorithm.

The effect is especially large for the value-based methods.

For DQN:

```text
road-heavy    34.2%
standard      46.4%
river-heavy   61.6%
```

For QR-DQN:

```text
road-heavy    20.0%
standard      33.6%
river-heavy   46.4%
```

This result is consistent with the earlier v9 failure-mode analysis, where road collisions were the most common residual failure for PPO, TRPO, and DQN.

It also helps explain why v9 can outperform v7 despite having two different mechanics.

v7 contains four procedural road hazards, while v9 replaces some of those roads with rivers.

The additional mechanic increases behavioral diversity, but does not necessarily increase overall task difficulty because river hazards are comparatively easier for the trained agents.

---

## 16. Combined Distribution-Shift Findings

The two completed shift experiments test different forms of robustness.

### Dynamics shift

Hazard-speed modification changes how quickly the environment evolves.

> Policies are relatively robust to slower-than-training dynamics but degrade substantially under faster-than-training dynamics.

### Composition shift

Hazard-composition modification changes how frequently each learned mechanic is encountered.

> Policies perform worse as the environment becomes more road-heavy and better as it becomes more river-heavy.

Together, these results reinforce several broader benchmark findings.

### Robustness is asymmetric

A perturbation does not necessarily have equal effects in both directions.

For example:

```text
0.8x speed
≈ standard performance

1.4x speed
<< standard performance
```

### Mechanics contribute unequally to difficulty

Road and river hazards are not interchangeable sources of complexity.

Road hazards are consistently more difficult under the tested v9 configuration.

### In-distribution performance does not completely predict robustness

The highest-performing in-distribution algorithm does not necessarily remain the highest-performing algorithm under shift.

DQN, for example, becomes the strongest mean performer at the most severe tested speed shift.

### Failure-mode analysis is important

Aggregate success rate alone does not explain why performance changes.

The speed-shift experiments show that degradation is disproportionately associated with increased road collisions.

---

# Distribution-Shift Figures

Distribution-shift figures and aggregated results are stored under:

```text
results/figures/distribution_shift/
```

Current outputs include:

```text
v9_speed_shift_robustness.png
v9_speed_shift_failure_modes.png
v9_speed_shift_raw.csv
v9_speed_shift_summary.csv

v9_composition_shift_robustness.png
v9_composition_shift_raw.csv
v9_composition_shift_summary.csv
```

---

# Updated Open Questions

With the initial distribution-shift experiments complete, promising directions now include:

1. **Cross-environment transfer**
   - evaluate whether policies learned in one environment transfer to related environments

2. **Training under broader distributions**
   - determine whether domain randomization improves robustness to speed or composition shift

3. **Mechanic-specific transfer**
   - test whether road or river competence transfers when mechanics are introduced separately

4. **More severe out-of-distribution changes**
   - alter layout constraints or environmental scale rather than only local dynamics

5. **Robustness versus specialization**
   - compare policies optimized for narrow fixed distributions against policies trained under greater procedural diversity

Cross-environment transfer is the most natural next step because it asks a qualitatively different question:

> Do the learned behaviors generalize across related tasks, rather than only across variants of the same v9 environment?


---

## 17. Cross-Environment Transfer Between v8 and v9

We next tested whether policies trained in one mixed-mechanics environment could transfer directly to the other without retraining.

The comparison is between:

- **v8:** fixed mixed-mechanics layout
- **v9:** procedural mixed-mechanics layout

Both environments use the same action space and compatible local2 observation structure, allowing direct checkpoint evaluation.

Two transfer directions were evaluated:

```text
v8 -> v9
fixed -> procedural

v9 -> v8
procedural -> fixed
```

All four algorithms were evaluated across five training seeds using 100 evaluation episodes per seed.

### Transfer success rates

| Direction | PPO | TRPO | DQN | QR-DQN |
|---|---:|---:|---:|---:|
| v8 -> v9 | 11.0% ± 8.3% | 14.0% ± 4.4% | 2.0% ± 1.9% | 0.8% ± 0.8% |
| v9 -> v8 | 70.2% ± 8.0% | 76.2% ± 15.2% | 76.0% ± 10.5% | 53.4% ± 14.2% |

The transfer pattern is strongly asymmetric.

Policies trained on fixed v8 perform very poorly when evaluated on procedural v9:

```text
PPO      11.0%
TRPO     14.0%
DQN       2.0%
QR-DQN    0.8%
```

Relative to their native v8 performance, these policies retain only a small fraction of their original success rate.

By contrast, v9-trained policies retain substantial competence when transferred to fixed v8:

```text
PPO      70.2%
TRPO     76.2%
DQN      76.0%
QR-DQN   53.4%
```

This produces the following native-to-transfer comparison:

| Algorithm | Native v8 | v8 -> v9 | Native v9 | v9 -> v8 |
|---|---:|---:|---:|---:|
| PPO | 94.0% | 11.0% | 50.0% | 70.2% |
| TRPO | 100.0% | 14.0% | 49.0% | 76.2% |
| DQN | 80.6% | 2.0% | 46.4% | 76.0% |
| QR-DQN | 96.2% | 0.8% | 33.6% | 53.4% |

### Interpretation

The results suggest that training under procedural variation produces behavior that is substantially more reusable across related environment configurations.

Fixed-layout policies appear to specialize heavily to the structure encountered during training.

Once the road and river locations become procedural, their performance collapses.

Procedural v9 policies, in contrast, remain effective when transferred to the fixed v8 environment.

A useful summary is:

> Cross-environment transfer is strongly asymmetric: policies trained under procedural variation retain substantial competence on the corresponding fixed environment, while policies trained on the fixed environment generalize poorly to the procedural environment.

This should not be interpreted as evidence that procedural training universally dominates fixed training.

The transfer directions differ in difficulty:

- v8 -> v9 moves from a simpler fixed task to a more variable procedural task
- v9 -> v8 moves from a more variable task to a simpler fixed task

The result therefore supports a claim about **asymmetric generalization**, rather than a symmetric comparison of policy quality.

---

## 18. Transfer Failure Modes

Failure behavior also differs substantially across algorithms in the difficult v8 -> v9 direction.

Mean v8 -> v9 outcomes include:

| Algorithm | Road collision | Drowning | Timeout |
|---|---:|---:|---:|
| PPO | 35.0% | 30.2% | 23.8% |
| TRPO | 48.6% | 37.0% | 0.4% |
| DQN | 28.0% | 39.0% | 31.0% |
| QR-DQN | 20.6% | 26.2% | 52.4% |

These results indicate that fixed-layout specialization does not manifest in exactly the same way for every algorithm.

TRPO continues to make substantial progress in v9 but fails primarily through interactions with the hazards themselves.

QR-DQN, in contrast, frequently stalls or fails to progress effectively, producing a timeout rate above 50%.

DQN exhibits both hazard failures and substantial timeout behavior.

This suggests at least two distinct transfer-failure patterns:

1. **behavioral timing failure**
   - the agent continues traversing the environment but its learned interaction strategy no longer works reliably

2. **structural specialization**
   - the policy appears to rely more strongly on the fixed spatial organization encountered during training and fails to make consistent progress when that organization changes

The v9 -> v8 direction is much healthier overall, with most failures concentrated in road collisions or drowning rather than widespread timeout behavior.

---

## 19. Cross-Environment Transfer Figure

The transfer comparison is visualized in:

```text
results/figures/cross_env_transfer/
    v8_v9_cross_environment_transfer.png
```

The figure separates the two training conditions:

- **Train on Fixed v8**
  - native v8 performance
  - transfer to procedural v9

- **Train on Procedural v9**
  - native v9 performance
  - transfer to fixed v8

The visual asymmetry is large.

Fixed v8 policies transition from high native performance to near-zero or low success on v9, while v9 policies retain substantial success when transferred to v8.

Aggregated transfer data are stored under:

```text
results/cross_env_transfer/
```

including:

```text
cross_env_transfer_raw.csv
cross_env_transfer_summary.csv
cross_env_transfer_retention.csv
```

---

## 20. Updated Generalization Findings

The benchmark now contains three distinct forms of generalization analysis:

### Same-generator held-out seeds

v7 policies show similar performance on standard and held-out reset seeds.

This indicates that changing the random seed within the same procedural generator does not constitute a meaningful distribution shift.

### Evaluation-time distribution shift

v9 policies were evaluated under:

- hazard-speed changes
- hazard-composition changes

These experiments show that robustness depends on the direction and type of shift.

### Cross-environment transfer

v8 and v9 provide a stronger structural generalization test.

The result is strongly asymmetric:

```text
fixed training -> procedural evaluation
poor transfer

procedural training -> fixed evaluation
substantial transfer
```

Together, these experiments suggest that procedural diversity during training is important for learning behavior that remains useful outside a single fixed environment configuration.

At the same time, procedural training does not eliminate robustness limitations: v9 policies still degrade under faster hazards and under road-heavy composition shifts.

This points to an important distinction:

> Procedural training improves structural generalization, but does not guarantee robustness to all forms of distribution shift.


---

## 21. Speed Domain Randomization in v10

After identifying sensitivity to faster hazard dynamics in v9, we introduced a training-time intervention to test whether broader exposure to speed variation could improve robustness.

v10 preserves the v9 environment structure and mechanics, but randomizes the global hazard speed multiplier once per episode during training:

```text
speed_scale ~ Uniform(0.8, 1.2)
```

The sampled scale affects both:

- road-car speeds
- river-platform speeds

Everything else remains unchanged relative to v9, including:

- procedural hazard placement
- hazard composition sampling
- local2 observation
- reward structure
- action space
- road and river mechanics

The purpose of v10 is therefore not to create a new benchmark task, but to test whether moderate training-time domain randomization improves evaluation-time robustness.

No evaluation-time randomization is used.

Instead, v10-trained models are evaluated on fixed v9 environments at:

```text
0.8x
1.0x
1.2x
1.4x
```

This allows direct comparison against the existing v9-trained policies.

---

## 22. v9 vs. v10 Speed Robustness

All four algorithms were trained for 1M steps across five seeds under v10.

The resulting policies were evaluated at the same four speed scales used in the original v9 speed-shift experiment.

### Final 1M-step success rates

| Algorithm | Training | 0.8x | 1.0x | 1.2x | 1.4x |
|---|---|---:|---:|---:|---:|
| PPO | v9 fixed-speed | 49.0% ± 4.2% | 50.0% ± 7.6% | 38.6% ± 6.9% | 27.8% ± 5.8% |
| PPO | v10 randomized | 49.0% ± 2.4% | 48.6% ± 5.7% | 42.4% ± 4.5% | 29.8% ± 2.6% |
| TRPO | v9 fixed-speed | 48.4% ± 10.9% | 49.0% ± 6.8% | 40.6% ± 7.5% | 28.6% ± 6.2% |
| TRPO | v10 randomized | 51.6% ± 5.9% | 53.4% ± 5.7% | 41.6% ± 4.2% | 31.2% ± 3.2% |
| DQN | v9 fixed-speed | 50.2% ± 9.7% | 46.4% ± 7.8% | 37.0% ± 4.9% | 31.0% ± 5.6% |
| DQN | v10 randomized | 47.6% ± 15.3% | 44.8% ± 15.4% | 37.8% ± 11.5% | 29.6% ± 10.3% |
| QR-DQN | v9 fixed-speed | 32.2% ± 3.3% | 33.6% ± 6.3% | 27.2% ± 8.2% | 27.0% ± 2.4% |
| QR-DQN | v10 randomized | 38.0% ± 10.5% | 36.4% ± 12.5% | 29.2% ± 8.0% | 29.6% ± 5.8% |

---

## 23. Domain-Randomization Effect

The absolute change in success rate from v9 to v10 is:

| Algorithm | 0.8x | 1.0x | 1.2x | 1.4x |
|---|---:|---:|---:|---:|
| PPO | +0.0 pp | -1.4 pp | +3.8 pp | +2.0 pp |
| TRPO | +3.2 pp | +4.4 pp | +1.0 pp | +2.6 pp |
| DQN | -2.6 pp | -1.6 pp | +0.8 pp | -1.4 pp |
| QR-DQN | +5.8 pp | +2.8 pp | +2.0 pp | +2.6 pp |

The effect is therefore algorithm-dependent.

### PPO

PPO shows the most intervention-like behavior.

Performance is nearly unchanged at 0.8x, slightly reduced at the nominal 1.0x condition, and improved under faster evaluation dynamics:

```text
1.2x: +3.8 pp
1.4x: +2.0 pp
```

This suggests that broader speed exposure during training improves PPO robustness to timing compression while incurring only a small nominal-speed cost.

### TRPO

TRPO improves across every tested speed condition:

```text
0.8x: +3.2 pp
1.0x: +4.4 pp
1.2x: +1.0 pp
1.4x: +2.6 pp
```

The largest gain occurs at 1.0x rather than under the strongest shift.

This suggests that speed randomization may improve TRPO's policy more generally, rather than acting only as an out-of-distribution robustness intervention.

### DQN

DQN shows no consistent benefit.

Its only improvement is:

```text
1.2x: +0.8 pp
```

while performance decreases slightly at:

```text
0.8x
1.0x
1.4x
```

The v10 DQN results also show substantially larger seed variance than v9.

Speed domain randomization therefore does not appear to be a reliable robustness intervention for DQN under the current setup.

### QR-DQN

QR-DQN improves at every tested speed:

```text
0.8x: +5.8 pp
1.0x: +2.8 pp
1.2x: +2.0 pp
1.4x: +2.6 pp
```

Although the gains are modest and variance increases, the direction of the effect is consistently positive.

This suggests that broader training dynamics may reduce some of QR-DQN's sensitivity to the original narrow speed distribution.

---

## 24. Domain Randomization Improves but Does Not Solve Speed Sensitivity

The combined result is:

> Moderate speed domain randomization provides small robustness gains for PPO, TRPO, and QR-DQN, but does not consistently help DQN.

The intervention does not eliminate the underlying speed-shift problem.

All algorithms still degrade substantially as evaluation speed increases.

For example:

```text
PPO v10:
1.0x    48.6%
1.4x    29.8%

TRPO v10:
1.0x    53.4%
1.4x    31.2%
```

The faster-dynamics regime therefore remains significantly more difficult even after training-time randomization.

The result should be interpreted as a modest mitigation rather than a complete robustness solution.

---

## 25. Failure Modes After Domain Randomization

Road collisions remain the primary failure mode as evaluation speed increases.

For PPO under v10:

```text
road collision rate

1.0x    33.8%
1.2x    41.0%
1.4x    49.2%
```

For TRPO:

```text
1.0x    29.6%
1.2x    35.6%
1.4x    40.2%
```

The same qualitative pattern observed in v9 therefore remains present in v10.

Training on variable speeds reduces performance loss somewhat for several algorithms, but does not remove the underlying road-timing vulnerability.

This reinforces the earlier conclusion that high-speed road traversal is one of the dominant robustness bottlenecks in the procedural mixed-mechanics environment.

---

## 26. Speed Domain-Randomization Figure

The v9-v10 comparison is visualized in:

```text
results/figures/domain_randomization/
    v9_v10_speed_domain_randomization.png
```

The figure contains one panel per algorithm and compares:

```text
v9: fixed-speed training

v10: speed-randomized training
```

across:

```text
0.8x
1.0x
1.2x
1.4x
```

with error bars showing ±1 standard deviation across five training seeds.

Supporting aggregated data are stored in:

```text
results/figures/domain_randomization/
```

including:

```text
v9_v10_speed_comparison_raw.csv
v9_v10_speed_comparison_summary.csv
v9_v10_speed_comparison_effect.csv
v9_v10_speed_domain_randomization.png
```

---

## 27. Updated Robustness Findings

The benchmark now supports several increasingly strong conclusions about generalization and robustness.

### Procedural variation improves structural transfer

v9-trained policies transfer substantially better to fixed v8 than v8-trained policies transfer to procedural v9.

### Faster dynamics remain a major weakness

All algorithms degrade as hazard speed increases beyond the standard training regime.

### Road mechanics dominate speed sensitivity

The increase in failures under faster dynamics is primarily associated with road collisions rather than drowning.

### Composition affects task difficulty strongly

Road-heavy episodes are consistently harder than river-heavy episodes.

### Domain randomization provides partial mitigation

Training across a broader speed range improves robustness for several algorithms, but the effect is modest and algorithm-dependent.

In particular:

```text
PPO      modest positive effect
TRPO     consistent positive effect
QR-DQN   consistent positive effect
DQN      no consistent benefit
```

The resulting picture is therefore not that procedural training or domain randomization automatically solves out-of-distribution robustness.

Instead:

> Broader training distributions improve some forms of generalization, but robustness remains strongly dependent on algorithm, mechanic, and shift type.


---

## 28. Unseen Hazard-Composition Generalization

The earlier composition-shift experiment evaluated:

```text
1 road / 3 rivers
3 roads / 1 river
```

These conditions change the frequency of hazard mechanics but remain within the support of the standard v9 training distribution.

Standard v9 training permits the mixed compositions:

```text
1 road / 3 rivers
2 roads / 2 rivers
3 roads / 1 river
```

To test genuine composition-support extrapolation, two additional evaluation-only conditions were introduced:

```text
all_river = 0 roads / 4 rivers
all_road  = 4 roads / 0 rivers
```

These extreme compositions never occur during standard v9 training.

No retraining or fine-tuning is performed.

The same v9 checkpoints are evaluated directly on the unseen composition conditions.

---

## 29. Full Composition Generalization Ladder

The five evaluated composition conditions are:

```text
All river
0 roads / 4 rivers
        ↓
River-heavy
1 road / 3 rivers
        ↓
Standard mixture
1/3, 2/2, or 3/1
        ↓
Road-heavy
3 roads / 1 river
        ↓
All road
4 roads / 0 rivers
```

Five training seeds are evaluated for PPO, TRPO, DQN, and QR-DQN.

### Final 1M-step success rates

| Algorithm | All river | 1R / 3V | Standard | 3R / 1V | All road |
|---|---:|---:|---:|---:|---:|
| PPO | 61.4% ± 3.0% | 57.2% ± 4.9% | 50.0% ± 7.6% | 39.2% ± 5.1% | 26.0% ± 4.4% |
| TRPO | 62.2% ± 4.9% | 58.4% ± 4.7% | 49.0% ± 6.8% | 43.0% ± 7.8% | 36.2% ± 12.8% |
| DQN | 78.0% ± 7.8% | 61.6% ± 8.5% | 46.4% ± 7.8% | 34.2% ± 8.0% | 25.8% ± 3.3% |
| QR-DQN | 55.0% ± 9.1% | 46.4% ± 7.1% | 33.6% ± 6.3% | 20.0% ± 5.4% | 13.8% ± 7.3% |

The pattern is consistent across all four algorithms:

> Success decreases as the evaluation environment becomes more road-dominated.

---

## 30. Extreme Unseen-Composition Gap

The difference between the two unseen composition endpoints is large:

```text
All-river minus all-road success

PPO       +35.4 percentage points
TRPO      +26.0 percentage points
DQN       +52.2 percentage points
QR-DQN    +41.2 percentage points
```

DQN shows the largest extreme gap.

Its success rate changes from:

```text
all river    78.0%
all road     25.8%
```

QR-DQN also shows a very large difference:

```text
all river    55.0%
all road     13.8%
```

These results extend the earlier road-heavy / river-heavy finding beyond the composition support observed during training.

---

## 31. Failure Modes Across Composition

The failure-mode decomposition reinforces the same conclusion.

As road prevalence increases, road collisions become increasingly dominant.

### PPO

```text
All river      road collision =  0.0%
1R / 3V        road collision = 17.0%
Standard       road collision = 30.8%
3R / 1V        road collision = 55.8%
All road       road collision = 73.2%
```

### TRPO

```text
All river      road collision =  0.0%
1R / 3V        road collision = 13.4%
Standard       road collision = 29.6%
3R / 1V        road collision = 51.2%
All road       road collision = 63.2%
```

### DQN

```text
All river      road collision =  0.0%
1R / 3V        road collision = 15.4%
Standard       road collision = 33.2%
3R / 1V        road collision = 57.4%
All road       road collision = 73.0%
```

### QR-DQN

```text
All river      road collision =  0.0%
1R / 3V        road collision = 18.2%
Standard       road collision = 35.4%
3R / 1V        road collision = 60.8%
All road       road collision = 78.6%
```

Drowning exhibits the complementary pattern and disappears entirely in the all-road environment.

This indicates that the composition effect is not merely an aggregate success-rate artifact.

The underlying failure mechanism shifts systematically with the prevalence of each hazard type.

---

## 32. Interpretation of Unseen Composition Shift

The unseen endpoint results provide stronger evidence that road hazards are the dominant source of difficulty in v9.

The effect extends beyond the compositions encountered during training:

```text
unseen all-river
    easier than mixed training conditions

unseen all-road
    harder than mixed training conditions
```

This indicates that the learned policies can extrapolate to unseen mechanic compositions, but performance depends strongly on which mechanic dominates.

The result is not best interpreted as a generic failure of out-of-distribution generalization.

In fact, the all-river condition is also outside the training support and produces substantially higher success.

Instead, the relevant conclusion is:

> Out-of-support composition alone is not necessarily harmful; performance depends strongly on the intrinsic difficulty of the mechanic that dominates the shifted environment.

This distinction is important because it separates:

```text
distribution distance
```

from:

```text
task difficulty
```

The all-road and all-river environments are both compositionally unseen, yet they produce opposite performance effects.

---

## 33. Composition-OOD Figure

The full composition-generalization result is visualized in:

```text
results/figures/composition_ood/
    v9_composition_ood_ladder.png
```

The figure orders evaluation conditions from all-river to all-road and plots success rate with ±1 standard deviation across five training seeds.

Supporting outputs are stored in:

```text
results/figures/composition_ood/
```

including:

```text
v9_composition_ood_raw.csv
v9_composition_ood_summary.csv
v9_composition_success_ladder.csv
v9_composition_ood_ladder.png
```

One important caveat is that the standard condition is not a fixed two-road / two-river environment.

It is the original v9 mixture over:

```text
1R / 3V
2R / 2V
3R / 1V
```

The composition ladder therefore demonstrates a strong mechanic-composition trend, but should not be interpreted as a strictly linear per-road dose-response experiment.

---

## 34. Updated Distribution-Shift Interpretation

The distribution-shift experiments now separate several distinct phenomena.

### Same-support composition reweighting

```text
river-heavy
road-heavy
```

These conditions remain within v9 training support.

### Out-of-support composition extrapolation

```text
all-river
all-road
```

These conditions introduce compositions never observed during v9 training.

### Dynamics shift

```text
0.8x
1.2x
1.4x hazard speed
```

This changes the temporal dynamics while preserving the underlying mechanics.

The combined findings suggest:

1. **Road hazards are intrinsically harder than river hazards under the current benchmark design.**
2. **Increasing road prevalence reduces performance across all tested algorithms.**
3. **This trend continues outside the composition support seen during training.**
4. **Out-of-support evaluation is not inherently harmful, since all-river is both unseen and easier.**
5. **Distribution shift and task difficulty must therefore be analyzed separately.**

This provides a more precise robustness conclusion than simply labeling all shifted conditions as OOD failures.


---

## 35. Spatial-Layout Generalization

The v9 procedural generator constrains the spatial arrangement of hazards.

During standard training:

- four hazard rows are sampled from rows 1–8,
- at least one hazard must occur in the lower half,
- at least one hazard must occur in the upper half,
- no more than two hazards may appear consecutively.

To evaluate sensitivity to spatial structure, two evaluation-only layout modes were introduced.

### Separated control

The separated condition fixes hazards at:

```text
[1, 3, 6, 8]
```

This arrangement satisfies the original v9 layout constraints and is therefore structurally compatible with the training distribution.

It serves as a control for whether simply fixing the environment to a particular layout causes degradation.

### Clustered OOD

The clustered condition samples from:

```text
[2, 3, 4, 5]
[3, 4, 5, 6]
[4, 5, 6, 7]
```

Each layout contains four consecutive hazard rows.

Such layouts are excluded by the standard v9 generator because training permits at most two consecutive hazards.

The clustered condition therefore represents structural spatial-layout extrapolation outside the training support.

No retraining or fine-tuning is performed.

---

## 36. Spatial-Layout OOD Results

Five training seeds are evaluated for PPO, TRPO, DQN, and QR-DQN.

### Success rate at 1M steps

| Algorithm | Standard | Separated | Clustered OOD |
|---|---:|---:|---:|
| PPO | 50.0% ± 7.6% | 58.0% ± 3.8% | 21.8% ± 6.1% |
| TRPO | 49.0% ± 6.8% | 49.0% ± 9.4% | 26.2% ± 3.2% |
| DQN | 46.4% ± 7.8% | 48.8% ± 4.2% | 22.2% ± 2.4% |
| QR-DQN | 33.6% ± 6.3% | 34.2% ± 8.6% | 17.0% ± 6.8% |

Relative to standard v9:

```text
PPO
Separated      +8.0 pp
Clustered     -28.2 pp

TRPO
Separated       0.0 pp
Clustered     -22.8 pp

DQN
Separated      +2.4 pp
Clustered     -24.2 pp

QR-DQN
Separated      +0.6 pp
Clustered     -16.6 pp
```

The separated condition remains close to the standard v9 baseline for all four algorithms.

In contrast, the clustered OOD condition causes a large degradation across every tested algorithm.

This indicates that the observed failure is not simply caused by evaluating on a fixed spatial arrangement.

Instead, the important factor is the unseen clustering structure.

---

## 37. Failure Modes Under Spatial Clustering

The clustered condition increases both road collisions and drowning.

### PPO

```text
Standard
road collision = 30.8%
drowning       = 18.6%

Clustered
road collision = 51.4%
drowning       = 26.8%
```

### TRPO

```text
Standard
road collision = 29.6%
drowning       = 20.6%

Clustered
road collision = 47.2%
drowning       = 26.2%
```

### DQN

```text
Standard
road collision = 33.2%
drowning       = 18.6%

Clustered
road collision = 49.4%
drowning       = 27.6%
```

### QR-DQN

```text
Standard
road collision = 35.4%
drowning       = 27.2%

Clustered
road collision = 49.0%
drowning       = 29.6%
```

The degradation therefore does not appear to arise from one isolated mechanic.

Instead, consecutive hazards increase failure across both road and river interactions.

---

## 38. Interpretation of Spatial-Layout Shift

The spatial-layout experiment reveals a structural limitation that is distinct from the earlier speed and composition shifts.

The key comparison is:

```text
fixed separated layout
    ≈ standard performance

unseen clustered layout
    << standard performance
```

This suggests that the learned policies tolerate a fixed spatial arrangement when it remains compatible with the structural patterns encountered during training.

However, performance degrades substantially when hazards are arranged into a four-row consecutive cluster excluded by the training generator.

The result therefore supports the conclusion:

> The learned policies are sensitive not only to hazard mechanics and dynamics, but also to the higher-level spatial structure in which hazards are encountered.

Because both road and river failure rates increase under clustering, the effect is best interpreted as a general sequential-hazard difficulty rather than a failure associated with one mechanic alone.

---

## 39. Spatial-Layout OOD Figure

The spatial-layout result is visualized in:

```text
results/figures/layout_ood/
    v9_layout_ood_success.png
```

The grouped bar chart compares:

```text
Standard
Separated
Clustered OOD
```

for PPO, TRPO, DQN, and QR-DQN.

Error bars show ±1 standard deviation across five training seeds.

Supporting outputs are stored in:

```text
results/figures/layout_ood/
    v9_layout_ood_raw.csv
    v9_layout_ood_summary.csv
    v9_layout_ood_success.png
```

---

## 40. Updated Generalization Picture

The v9 benchmark now exposes several distinct forms of generalization.

### Same-distribution reset seeds

Held-out reset seeds drawn from the same procedural generator produce little degradation.

This is not true OOD evaluation.

### Composition reweighting

```text
river-heavy
road-heavy
```

These conditions alter mechanic prevalence while remaining within training support.

### Composition-support extrapolation

```text
all-river
all-road
```

These conditions contain hazard compositions never observed during standard training.

### Dynamics shift

```text
0.8x
1.2x
1.4x hazard speed
```

These conditions change temporal dynamics.

### Spatial-structure extrapolation

```text
clustered four-hazard sequences
```

These layouts violate the maximum-consecutive-hazard constraint used during training.

Taken together, the experiments show that generalization depends strongly on the dimension along which the environment changes.

The policies are relatively tolerant to some within-support changes, but can degrade sharply when evaluation violates structural assumptions embedded in the training distribution.

