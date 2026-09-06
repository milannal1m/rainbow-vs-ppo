#!/bin/bash
#SBATCH --job-name=flappybird-eval
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32000
#SBATCH --time=4:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch evaluate.sh [hyperparams_set] [extra agent.py flags...]
#          sbatch evaluate.sh mario_ppo --evaluate-levels
# Example: sbatch evaluate.sh flappybird_cnn2
HYPERPARAMS=${1:-flappybird_cnn2}
shift || true

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
# Mario configs need the python 3.13 env:
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda evaluate.sh mario_rainbow
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1

python src/agent.py "$HYPERPARAMS" "${@:---evaluate}"
