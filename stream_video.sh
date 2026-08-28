#!/bin/bash
#SBATCH --job-name=stream-video
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4         # 1 for env+policy, ~3 for the x264 encode
#SBATCH --mem=8000                # streaming is O(1) in frames (~0.4GB); no 128GB needed
#SBATCH --time=13:00:00           # partition max; a full FlappyBird episode needs ~2h
#SBATCH --signal=B:USR1@300       # USR1 5 min before the limit -> mp4 gets finalised, not truncated
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Record a video of a trained policy by streaming frames into ffmpeg (constant memory).
#
# Usage: sbatch stream_video.sh <hyperparams_set> [extra record_video.py flags...]
#
#   sbatch stream_video.sh flappybird_ppo_tuned_3                    # whole episode
#   sbatch stream_video.sh flappybird_ppo_tuned_3 --seconds 600      # cap at 10 min of footage
#   sbatch stream_video.sh flappybird_ppo_tuned_3 --seed 7
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda stream_video.sh mario_ppo --level 1-1
#
# Output lands in runs/<set>/videos/clip_s<seed>_r<reward>.mp4.
#
# Why not RecordVideo: it buffers every frame in RAM (0.44 MB each), so flappybird_ppo_tuned_3's
# ~25,000 s episode would need ~335 GB. Streaming holds ~0.4 GB regardless of length; the real
# limits become compute (~100 frames/s, so ~2 h for a full episode) and disk (~75 MB).

HP=${1:?usage: sbatch stream_video.sh <hyperparams_set> [record_video.py flags...]}
shift || true

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
# Mario configs need the python 3.13 env:
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda stream_video.sh mario_ppo
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1

echo "streaming video: set=$HP env=$CONDA_ENV extra_args=$* time_limit=${SLURM_JOB_END_TIME:-48h}"

python src/record_video.py "$HP" --stream "$@" &
PID=$!

# Forward the early SLURM signal to python: it breaks its write loop, closes the ffmpeg pipe and
# ffmpeg writes the moov atom. Without this a wall-clock kill leaves an unplayable file.
trap 'echo "[signal] USR1 -> asking the recorder to finalise"; kill -USR1 "$PID" 2>/dev/null' USR1

# `wait` returns early whenever a trap fires, so loop until the child is really gone.
while kill -0 "$PID" 2>/dev/null; do wait "$PID"; done

echo "done. videos:"
ls -lh "runs/$HP/videos/" 2>/dev/null || echo "  (no videos dir -- check the log above)"
