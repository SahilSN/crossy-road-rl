# Crossy Road RL Benchmark

An experimental reinforcement-learning benchmark for studying how policy performance changes under observation design, procedural generation, mixed mechanics, transfer, and distribution shift.

The project evolves a simple Crossy Road-style environment through several controlled variants and evaluates four reinforcement-learning algorithms:

- PPO
- TRPO
- DQN
- QR-DQN

Experiments use multiple training seeds and checkpoint evaluations through 1M environment steps.

## Benchmark Environments

The main environment progression is:

- **v3** — simple fixed road environment
- **v4** — larger fixed global-observation environment
- **v5** — local egocentric observation
- **v6** — observation-horizon ablation
- **v7** — procedural road layouts
- **v8** — fixed mixed road/river mechanics
- **v9** — procedural mixed-mechanics benchmark
- **v10** — v9 with speed domain randomization during training

v9 serves as the primary final benchmark.

## Main Findings

The experiments show that:

- local observations substantially outperform the original global representation;
- the best observation horizon depends on the RL algorithm;
- procedural generation is a major source of difficulty;
- mixed mechanics alone do not necessarily make a task difficult;
- road hazards are systematically harder than river hazards;
- faster hazard dynamics reduce performance across algorithms;
- unseen hazard compositions can produce large but direction-dependent performance changes;
- unseen clustered spatial layouts cause substantial degradation;
- zero-shot transfer from procedural v9 to fixed v8 is much stronger than transfer from v8 to v9;
- speed domain randomization provides modest, algorithm-dependent robustness gains;
- recurrent PPO did not outperform feed-forward PPO in the tested partial-observation setting.

For detailed results, tables, interpretations, and figures, see:

[`docs/benchmark_findings.md`](docs/benchmark_findings.md)

## Repository Structure

```text
crossyroad_rl/
    env.py
    env_v4.py
    ...
    env_v10.py

training/
    run_experiment.py
    evaluate_run.py
    aggregate_*.py
    plot_*.py

results/
    runs/
    figures/
    benchmark_all_runs.csv
    benchmark_summary.csv

docs/
    benchmark_findings.md
```

## Training

Example:

```bash
python training/run_experiment.py \
    --env v9 \
    --algorithm ppo \
    --seed 0
```

Training checkpoints are saved periodically through 1M steps.

## Evaluation

Example:

```bash
python training/evaluate_run.py \
    --env v9 \
    --algorithm ppo \
    --seed 0 \
    --episodes 100 \
    --eval-seed-start 1000
```

The evaluator also supports the distribution-shift conditions used in the benchmark, including speed, composition, and spatial-layout evaluation modes.

## Results

Final figures are organized under:

```text
results/figures/
```

Key categories include:

```text
cross_environment/
distribution_shift/
cross_env_transfer/
domain_randomization/
composition_ood/
layout_ood/
```

Raw per-run evaluations are preserved under:

```text
results/runs/
```

## Reproducibility

The benchmark uses multiple training seeds and fixed evaluation-seed ranges.

Before reproducing results, install the project dependencies and verify that the desired Stable-Baselines3 / sb3-contrib versions match the original experimental environment.

## Status

The benchmark is experimentally complete.

The repository is retained as a record of the environment progression, trained policies, evaluation results, and generalization analyses.
