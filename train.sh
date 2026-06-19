#!/bin/bash
#SBATCH --job-name=flappybird-dqn
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=16000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage: sbatch train.sh [hyperparams_set]
# Example: sbatch train.sh flappybird_cnn1
HYPERPARAMS=${1:-flappybird_cnn1}

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dqn-flappy-bird-cuda

export HEADLESS=1

python src/agent.py "$HYPERPARAMS" --train &
TRAIN_PID=$!

sleep 3600
nvidia-smi

# Auf Training warten
wait $TRAIN_PID


#squeue --me
#squeue -u ul_lmm50 --start
