#!/bin/bash
#SBATCH --job-name=rl-train
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32000
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# USR1 reaches the batch shell 600 s before the wall clock; the trap below forwards it to the
# agent, which stops cleanly and writes its replay buffer. Without the trap bash drops it.
#SBATCH --signal=B:USR1@600

# Usage (submit from the repo root):
#   sbatch scripts/train.sh flappybird_ppo_tuned
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda scripts/train.sh mario_rainbow_tuned --resume
HYPERPARAMS=${1:-flappybird_ppo_tuned}
shift || true

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
# Mario needs the python 3.13 env; gym-super-mario-bros requires it.
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1

python src/agent.py "$HYPERPARAMS" --train "$@" &
TRAIN_PID=$!

trap 'echo "[signal] USR1 -- forwarding to agent $TRAIN_PID"; kill -USR1 "$TRAIN_PID" 2>/dev/null' USR1

# Backgrounded: bash defers a trap until the running foreground command returns, so a
# foreground sleep here would swallow USR1 for up to an hour.
( sleep 3600; nvidia-smi ) &   # GPU snapshot an hour in, for the record

# `wait` returns as soon as a trap runs, so repeat until the child is really gone.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
done
