#!/bin/bash
set -e

ALGORITHMS=("ppo" "dqn" "qrdqn")
SEEDS=(0 1 2 3 4)

for algorithm in "${ALGORITHMS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo
        echo "=========================================="
        echo "Training ${algorithm}, seed ${seed}"
        echo "=========================================="

        python training/run_experiment.py \
            --algorithm "$algorithm" \
            --seed "$seed" \
            --timesteps 1000000 \
            --checkpoint-freq 200000

        echo
        echo "Evaluating ${algorithm}, seed ${seed}"

        python training/evaluate_run.py \
            --algorithm "$algorithm" \
            --seed "$seed" \
            --episodes 100 \
            --eval-seed-start 1000
    done
done

echo
echo "All benchmark runs complete."
