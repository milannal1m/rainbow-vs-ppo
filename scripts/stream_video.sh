#!/bin/bash
#SBATCH --job-name=stream-video
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4         # 1 for env+policy, ~3 for the x264 encode
#SBATCH --mem=8000                # streaming is O(1) in frames, ~0.4 GB
#SBATCH --time=10:00:00           # partition max; a full FlappyBird episode needs ~6 h
#SBATCH --signal=B:USR1@300       # 5 min of warning, so the mp4 is finalised rather than truncated
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Record a trained policy by streaming frames into ffmpeg, which holds constant memory.
# RecordVideo buffers every frame (0.44 MB each), so a converged FlappyBird episode would need
# ~335 GB; streaming caps out at compute (~100 frames/s) and disk (~75 MB) instead.
#
# Usage (submit from the repo root). Output: runs/<set>/videos/clip_s<seed>_r<reward>.mp4
#   sbatch scripts/stream_video.sh flappybird_ppo_tuned                  # whole episode
#   sbatch scripts/stream_video.sh flappybird_ppo_tuned --seconds 600    # cap the footage
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda scripts/stream_video.sh mario_ppo_tuned --level 1-1
HP=${1:?usage: sbatch scripts/stream_video.sh <hyperparams_set> [record_video.py flags...]}
shift || true

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1
echo "streaming video: set=$HP env=$CONDA_ENV extra_args=$*"

python src/record_video.py "$HP" --stream "$@" &
PID=$!

# Forward the signal so python breaks its write loop and ffmpeg writes the moov atom; a hard
# kill mid-write leaves an unplayable file.
trap 'echo "[signal] USR1 -> asking the recorder to finalise"; kill -USR1 "$PID" 2>/dev/null' USR1

while kill -0 "$PID" 2>/dev/null; do wait "$PID"; done

echo "done. videos:"
ls -lh "runs/$HP/videos/" 2>/dev/null || echo "  (no videos dir -- check the log above)"
