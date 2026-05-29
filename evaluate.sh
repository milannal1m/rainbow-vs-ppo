#!/bin/bash
#SBATCH --job-name=flappybird-eval
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch evaluate.sh [hyperparams_set]
# Example: sbatch evaluate.sh flappybird_cnn2
HYPERPARAMS=${1:-flappybird_cnn2}

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dqn-flappy-bird-cuda

export HEADLESS=1

python agent.py "$HYPERPARAMS" --evaluate
