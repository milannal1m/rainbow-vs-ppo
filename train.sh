#!/bin/bash
#SBATCH --job-name=flappybird-dqn
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=112000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch train.sh [hyperparams_set] [extra agent.py flags...]
# Example: sbatch train.sh flappybird_cnn1
#          sbatch train.sh mario_rainbow --resume
HYPERPARAMS=${1:-flappybird_cnn1}
shift || true

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
# Mario configs need the python 3.13 env (gym-super-mario-bros requires >= 3.13):
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda train.sh mario_rainbow
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1

python src/agent.py "$HYPERPARAMS" --train "$@" &
TRAIN_PID=$!

sleep 3600
nvidia-smi

# Auf Training warten
wait $TRAIN_PID


#squeue --me
#squeue -u ul_lmm50 --start
