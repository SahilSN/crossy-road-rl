# Crossy Road RL

A controlled reinforcement-learning benchmark for studying how algorithm performance changes as a simple Crossy Road-style environment is progressively modified.

## Benchmark Analysis

For a more detailed discussion of the experimental results and cross-environment findings, see [docs/benchmark_findings.md](docs/benchmark_findings.md).

The project is designed around a sequence of environments that vary one or a small number of factors at a time, including:

- environment difficulty,
- observation representation,
- observation horizon,
- procedural layout variation,
- heterogeneous hazard mechanics, and
- combinations of procedural variation and mixed mechanics.

The current benchmark compares:

- **PPO**
- **TRPO**
- **DQN**
- **QR-DQN** (50 quantiles)

Recurrent PPO has also been evaluated separately as an observation/memory ablation.

> **Status:** Experimental benchmark in active development. The results below summarize completed experiments through environment **v9**.

---

## Experimental Protocol

Unless otherwise noted:

- Training checkpoints: **200k, 400k, 600k, 800k, and 1M steps**
- Final benchmark: **5 training seeds (0–4)**
- Evaluation: **100 episodes per checkpoint**
- Standard evaluation seeds: **1000–1099**
- QR-DQN uses **50 quantiles**
- Environments use a discrete 5-action space:
  - wait
  - forward
  - backward
  - left
  - right

The benchmark emphasizes both final performance and learning behavior across training.

---

# Environment Progression

## v3 — Simple Fixed Road Environment

The initial environment is a compact, fully observed road-crossing task.

Key characteristics:

- grid width: 7
- goal row: 5
- two fixed road rows
- 3 cars per lane
- fully observed global state
- maximum episode length: 50 steps

### Final 1M-step results

| Algorithm | Success rate |
|---|---:|
| QR-DQN | **81.8% ± 5.2%** |
| TRPO | **74.2% ± 4.2%** |
| PPO | **57.8% ± 24.5%** |
| DQN | **0.0%** |

DQN converged to a systematic stalling strategy, while PPO exhibited substantial seed variance.

A2C and Recurrent PPO were also tested as exploratory baselines and tended to collapse toward aggressive forward movement.

---

## v4 — Harder Fixed Environment with Global Observation

v4 increases the scale and traffic difficulty while retaining a fully observed global state.

Key characteristics:

- grid width: 9
- goal row: 9
- four road rows: 1, 3, 5, 7
- progressively denser/faster traffic
- alternating lane directions
- maximum episode length: 100 steps
- global observation

### Final 1M-step results

| Algorithm | Success rate |
|---|---:|
| QR-DQN | **12.0% ± 2.8%** |
| DQN | **10.2% ± 3.6%** |
| PPO | **7.0% ± 0.7%** |
| TRPO | **6.2% ± 0.8%** |

Despite the modest increase in environmental complexity, performance collapses for all four algorithms.

This motivated an investigation into whether the difficulty came from the environment itself or from the representation supplied to the agent.

---

## v5 — Local Egocentric Observation

v5 keeps the **same underlying dynamics as v4** but replaces the global representation with a compact local observation.

The agent observes:

- one row behind,
- its current row,
- two rows ahead.

The environment dynamics remain equivalent to v4.

### Final 1M-step results

| Algorithm | Success rate |
|---|---:|
| TRPO | **94.0% ± 5.2%** |
| PPO | **79.8% ± 15.3%** |
| DQN | **68.6% ± 12.5%** |
| QR-DQN | **56.0% ± 20.8%** |

This produces one of the largest effects in the benchmark:

> **Providing less state information substantially improves learning when that information is structured around the agent's local decision context.**

The result suggests that compact local observations can act as a useful inductive bias.

---

## v6 — Observation-Horizon Ablation

v6 holds the underlying v4/v5 world fixed while varying how many rows the agent can observe.

The tested representations are:

- **local1:** current row, one behind, one ahead
- **local2:** current row, one behind, two ahead
- **local3:** current row, one behind, three ahead
- **global:** full v4 observation

### Final 1M-step success rates

| Algorithm | Local1 | Local2 | Local3 | Global |
|---|---:|---:|---:|---:|
| PPO | 49.0% | **79.8%** | 64.4% | 7.0% |
| TRPO | 87.8% | **94.0%** | 82.4% | 6.2% |
| DQN | 64.0% | **68.6%** | 54.6% | 10.2% |
| QR-DQN | **62.2%** | 56.0% | 47.4% | 12.0% |

### Main finding

The optimal observation horizon is algorithm-dependent.

- PPO, TRPO, and DQN perform best with **local2**
- QR-DQN performs best with **local1**
- the global representation is consistently the worst

More information is therefore not necessarily beneficial.

---

# Recurrent PPO Ablation

Recurrent PPO was evaluated on the local1 environment to test whether memory could compensate for the limited observation horizon.

The optimized recurrent configuration used:

- `MlpLstmPolicy`
- learning rate: `3e-4`
- entropy coefficient: `0.02`
- LSTM hidden size: `64`
- 10 epochs

### Five-seed learning curve

| Steps | Success rate |
|---:|---:|
| 200k | 4.6% ± 4.8% |
| 400k | 8.0% ± 5.0% |
| 600k | 16.6% ± 8.4% |
| 800k | **23.8% ± 14.3%** |
| 1M | 18.8% ± 9.9% |

Feedforward PPO on the same local1 task reaches **49.0% ± 27.8%** at 1M.

Under this setup, recurrence does not improve PPO and is therefore treated as a completed negative result rather than a primary benchmark algorithm.

---

# Procedural Environments

## v7 — Procedural Road Layout

v7 keeps the local2 observation but randomizes the road layout every episode.

Each reset samples four road rows from rows 1–8 subject to structural constraints.

Difficulty depends on the **order in which hazards are encountered**, rather than absolute row position.

This prevents the policy from relying on a fixed mapping such as:

```text
row 1 -> road
row 3 -> road
row 5 -> road
row 7 -> road
```

Instead, the agent must interpret the local state of the current episode.

### Final 1M-step results

| Algorithm | Success rate |
|---|---:|
| TRPO | **48.6% ± 23.9%** |
| PPO | **23.6% ± 6.3%** |
| QR-DQN | **12.6% ± 6.4%** |
| DQN | **3.2% ± 4.4%** |

Relative to fixed-layout v5/local2:

| Algorithm | v5 fixed | v7 procedural | Change |
|---|---:|---:|---:|
| TRPO | 94.0% | 48.6% | -45.4 pp |
| PPO | 79.8% | 23.6% | -56.2 pp |
| QR-DQN | 56.0% | 12.6% | -43.4 pp |
| DQN | 68.6% | 3.2% | -65.4 pp |

Procedural variation dramatically increases training difficulty.

### Held-out procedural seeds

Models trained on v7 were also evaluated on a separate set of reset seeds (`10000–10099`).

At 1M:

| Algorithm | Standard eval | Held-out seeds |
|---|---:|---:|
| TRPO | 48.6% | 49.0% |
| PPO | 23.6% | 26.2% |
| QR-DQN | 12.6% | 14.8% |
| DQN | 3.2% | 2.4% |

There is essentially no degradation when sampling previously unseen reset seeds from the **same procedural generator**.

This should not be interpreted as out-of-distribution generalization; rather, it indicates that the learned policies transfer well to new samples from the training environment distribution.

---

# Mixed Mechanics

## v8 — Fixed Road + River Environment

v8 introduces a qualitatively different hazard type while returning to a fixed layout.

Layout:

```text
row 0: start
row 1: road
row 2: safe
row 3: road
row 4: safe
row 5: river
row 6: safe
row 7: river
row 8: safe
row 9: goal
```

### Road mechanic

The agent must avoid moving cars.

### River mechanic

The agent must land on moving platforms.

While standing on a platform, the player is carried horizontally during physics updates.

Failure occurs if:

- the player enters a river without platform support, or
- platform motion carries the player outside the valid horizontal boundary.

The local2 observation encodes:

- `0 = safe`
- `1 = road`
- `2 = river`

The evaluator separately tracks:

- road collisions,
- drowning,
- timeout,
- success.

### Final 1M-step results

| Algorithm | Success | Road collision | Drowning |
|---|---:|---:|---:|
| TRPO | **100.0% ± 0.0%** | 0.0% | 0.0% |
| QR-DQN | **96.2% ± 2.0%** | 1.2% | 2.6% |
| PPO | **94.0% ± 4.2%** | 2.8% | 3.2% |
| DQN | **80.6% ± 9.5%** | 2.8% | 16.6% |

The mixed-mechanics environment is highly learnable.

This provides an important counterexample to the assumption that simply adding qualitatively different mechanics necessarily makes an RL task harder.

---

## v9 — Procedural Mixed Mechanics

v9 combines the two preceding experimental axes:

- procedural hazard placement from v7
- heterogeneous road/river mechanics from v8

Each episode contains four hazards sampled across rows 1–8.

Each hazard is assigned either:

- `road`, or
- `river`

subject to constraints including:

- exactly four hazards,
- at least one road,
- at least one river,
- hazards in both the lower and upper halves of the map,
- no more than two consecutive hazard rows.

The observation remains local2 and explicitly identifies the current row type.

Therefore, the agent cannot simply memorize which mechanic belongs to a particular absolute row. It must select behavior based on the observed state of the current episode.

### Sanity baselines

| Policy | Success | Avg. max row |
|---|---:|---:|
| Random | 0% | 2.12 |
| Mixed-hazard heuristic | 32% | 5.91 |

The heuristic failures are distributed across both mechanics:

- 31 road collisions
- 37 drownings

This confirms that v9 remains solvable while exercising both hazard types.

### Final 1M-step results

| Algorithm | Success | Avg. max row | Road collision | Drowning | Timeout |
|---|---:|---:|---:|---:|---:|
| PPO | **50.0% ± 7.6%** | 7.30 | 30.8% | 18.6% | 0.6% |
| TRPO | **49.0% ± 6.8%** | 7.30 | 29.6% | 20.6% | 0.8% |
| DQN | **46.4% ± 7.8%** | 6.99 | 33.2% | 18.6% | 1.8% |
| QR-DQN | **33.6% ± 6.3%** | 6.59 | 35.4% | 27.2% | 3.8% |

### Learning behavior

At 200k steps:

| Algorithm | Success |
|---|---:|
| PPO | 29.6% |
| TRPO | 28.6% |
| DQN | 3.8% |
| QR-DQN | 0.2% |

At 1M:

| Algorithm | Success |
|---|---:|
| PPO | 50.0% |
| TRPO | 49.0% |
| DQN | 46.4% |
| QR-DQN | 33.6% |

PPO and TRPO learn useful behavior substantially earlier than DQN, but DQN closes most of the gap by 1M steps.

---

# Cross-Environment Findings

## 1. Observation representation can matter more than environment size

The transition from v4 to v5 changes the observation representation while retaining the same underlying world dynamics.

Performance improves dramatically:

| Algorithm | v4 global | v5 local2 |
|---|---:|---:|
| PPO | 7.0% | **79.8%** |
| TRPO | 6.2% | **94.0%** |
| DQN | 10.2% | **68.6%** |
| QR-DQN | 12.0% | **56.0%** |

A compact local representation provides a much stronger learning signal than the larger global observation.

---

## 2. The optimal observation horizon depends on the algorithm

v6 shows that there is no universally optimal amount of local context.

- PPO, TRPO, and DQN prefer local2
- QR-DQN prefers local1
- all four struggle with the global representation

This suggests that representation complexity interacts directly with optimization and algorithm design.

---

## 3. Procedural variation is a major source of difficulty

Comparing the fixed local2 road task (v5) with the procedural road task (v7) produces large performance losses across every algorithm.

Similarly, comparing fixed mixed mechanics (v8) with procedural mixed mechanics (v9):

| Algorithm | v8 fixed mixed | v9 procedural mixed | Change |
|---|---:|---:|---:|
| PPO | 94.0% | 50.0% | -44.0 pp |
| TRPO | 100.0% | 49.0% | -51.0 pp |
| DQN | 80.6% | 46.4% | -34.2 pp |
| QR-DQN | 96.2% | 33.6% | -62.6 pp |

The ability to memorize or specialize to a fixed layout appears to be a major advantage in the fixed environments.

---

## 4. Mechanical diversity does not monotonically increase difficulty

The comparison between v7 and v9 is especially informative.

Both are procedural, but v7 contains only roads while v9 contains both roads and rivers.

| Algorithm | v7 procedural roads | v9 procedural mixed |
|---|---:|---:|
| PPO | 23.6% | **50.0%** |
| TRPO | 48.6% | **49.0%** |
| DQN | 3.2% | **46.4%** |
| QR-DQN | 12.6% | **33.6%** |

Adding a second mechanic does not universally make the task harder.

The structure and difficulty of the underlying hazards matter more than a simple count of mechanics.

---

## 5. Algorithm rankings are environment-dependent

No algorithm dominates every environment.

Examples:

- **QR-DQN** is strongest in v3
- **TRPO** dominates v5 and completely solves v8
- **PPO and TRPO** are nearly tied in v9
- **DQN** ranges from complete failure in some environments to near-PPO/TRPO performance in v9

This makes the environment progression useful for studying differences in learning dynamics rather than only comparing a single final score.

---

# Current Benchmark Summary

Final 1M-step success rates from completed five-seed benchmarks:

| Environment | PPO | TRPO | DQN | QR-DQN |
|---|---:|---:|---:|---:|
| v3 — simple fixed roads | 57.8% | 74.2% | 0.0% | **81.8%** |
| v4 — harder global state | 7.0% | 6.2% | 10.2% | **12.0%** |
| v5 — fixed local2 | 79.8% | **94.0%** | 68.6% | 56.0% |
| v7 — procedural roads | 23.6% | **48.6%** | 3.2% | 12.6% |
| v8 — fixed mixed mechanics | 94.0% | **100.0%** | 80.6% | 96.2% |
| v9 — procedural mixed mechanics | **50.0%** | 49.0% | 46.4% | 33.6% |

v6 is reported separately because it is an observation-horizon ablation rather than a single environment configuration.

---

# Repository Structure

```text
crossyroad_rl/
    env.py          # v3
    env_v4.py       # harder fixed/global environment
    env_v5.py       # local observation
    env_v6.py       # parameterized observation horizon
    env_v7.py       # procedural road layout
    env_v8.py       # fixed road + river mechanics
    env_v9.py       # procedural mixed mechanics

training/
    run_experiment.py
    evaluate_run.py
    aggregate_results.py

results/
    runs/
    benchmark_all_runs.csv
    benchmark_summary.csv

tests/
    ...
```

---

# Reproducing an Experiment

A typical training run is:

```bash
python training/run_experiment.py \
    --env v9 \
    --algorithm ppo \
    --seed 0 \
    --timesteps 1000000 \
    --checkpoint-freq 200000
```

Evaluate the saved checkpoints with:

```bash
python training/evaluate_run.py \
    --env v9 \
    --algorithm ppo \
    --seed 0 \
    --episodes 100 \
    --eval-seed-start 1000
```

Aggregate completed experiments with:

```bash
python training/aggregate_results.py
```

Results are stored under:

```text
results/runs/<environment>/<run_name>/
```

---

# Interpretation

The benchmark so far suggests several broader patterns:

1. **State representation strongly controls learnability.**
2. **More information is not always better.**
3. **Observation requirements differ by RL algorithm.**
4. **Procedural variation can be substantially harder than fixed-layout control.**
5. **Adding mechanics does not necessarily increase difficulty.**
6. **Algorithms differ not only in final performance but also in sample efficiency and failure mode.**
7. **A benchmark should therefore vary both environment structure and representation rather than rely on a single task.**

These conclusions are currently exploratory and are intended to guide the next stage of benchmark development.

---

## Project Status

Completed:

- fixed-road baseline environments
- global/local observation comparison
- observation-horizon ablation
- recurrent PPO ablation
- procedural road environment
- held-out procedural-seed evaluation
- mixed road/river environment
- procedural mixed-mechanics environment
- five-seed PPO/TRPO/DQN/QR-DQN benchmarks through v9

Next steps will focus on consolidating the benchmark and deciding which additional environmental dimension is most informative to vary.
