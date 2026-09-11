#!/bin/bash
#SBATCH --job-name=rl-eval
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH --time=4:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage (submit from the repo root). Without extra flags this runs --evaluate.
#   sbatch scripts/evaluate.sh flappybird_ppo_tuned
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda scripts/evaluate.sh mario_ppo_tuned --evaluate-levels
HYPERPARAMS=${1:-flappybird_ppo_tuned}
shift || true

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1

python src/agent.py "$HYPERPARAMS" "${@:---evaluate}"
