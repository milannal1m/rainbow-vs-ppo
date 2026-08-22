#!/bin/bash
#SBATCH --job-name=flappybird-hpo
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:4              # pack ~12 trials (3/GPU); you are GPU-underutilized
#SBATCH --cpus-per-task=16        # ~2.5 cores/trial — NOT the 64 max
#SBATCH --mem=192000             # ~192GB: 12 Rainbow trials @ ~15GB replay + overhead
#SBATCH --time=48:00:00          # Mario studies need it; override for flappybird (sbatch --time=32:00:00)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Unified Optuna hyperparameter search for one algorithm.
#
# Usage: sbatch hyperparametertune.sh <algorithm> [n_trials] [trials_per_gpu]
#   <algorithm>     one of: dqn | rainbow | ppo
#   n_trials        target total trials in the study        (default 400)
#   trials_per_gpu  concurrent trials packed onto each GPU  (default: rainbow=1, dqn/ppo=3)
#
# Env-var overrides (defaults keep every existing FlappyBird invocation identical):
#   TUNE_ENV   flappybird | mario     which game to tune            (default flappybird)
#   OBJECTIVE  p25 | median | mean    eval statistic to maximise    (default p25)
#
# Mario needs BOTH the python 3.13 conda env and TUNE_ENV -- without TUNE_ENV the workers
# build a FlappyBird config and die on the missing flappy_bird_gymnasium import:
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda,TUNE_ENV=mario,OBJECTIVE=mean \
#          hyperparametertune.sh ppo 400
#
# PPO is replay-free, so request less RAM/CPU for it:
#   sbatch --mem=64000 --cpus-per-task=24 hyperparametertune.sh ppo 400
#
# Results land in runs/hpo/<algorithm>_<jobid>/ (journal, trials.csv, plots, summary.txt);
# the best config is appended to hyperparams.yml as flappybird_<algorithm>_tuned, then:
#   sbatch train.sh flappybird_<algorithm>_tuned

ALGO=${1:-ppo}
N_TRIALS=${2:-400}
TUNE_ENV=${TUNE_ENV:-flappybird}
OBJECTIVE=${OBJECTIVE:-p25}
# Per-algorithm packing default: Rainbow is GPU-compute-bound (C51 + noisy + dueling), so it
# saturates the card and must NOT be overpacked; PPO / vanilla DQN are env-bound and pack well.
if [ -n "$3" ]; then
    TRIALS_PER_GPU=$3
elif [ "$ALGO" = "rainbow" ]; then
    TRIALS_PER_GPU=1
else
    TRIALS_PER_GPU=3
fi

mkdir -p logs

module load devel/miniforge/25.3.1-python-3.12
source "$(conda info --base)/etc/profile.d/conda.sh"
# Mario studies need the python 3.13 env:
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda hyperparametertune.sh rainbow 200
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1
export OMP_NUM_THREADS=2          # keep 12 packed processes from oversubscribing the cores

N_GPUS=${SLURM_GPUS_ON_NODE:-4}
K=$(( N_GPUS * TRIALS_PER_GPU ))
if [ "$TUNE_ENV" = "flappybird" ]; then
    STUDY="${ALGO}_${SLURM_JOB_ID}"          # unchanged, so old journals stay resumable by name
else
    STUDY="${TUNE_ENV}_${ALGO}_${SLURM_JOB_ID}"
fi

echo "HPO: env=$TUNE_ENV algorithm=$ALGO objective=$OBJECTIVE n_trials=$N_TRIALS study=$STUDY workers=$K gpus=$N_GPUS"

# Launch K workers, round-robin pinned across the allocated GPUs. They share one
# file-based Optuna study (JournalStorage is multi-process safe). The first worker is
# given a head start so it creates the study before the rest attach.
for (( i=0; i<K; i++ )); do
    CUDA_VISIBLE_DEVICES=$(( i % N_GPUS )) \
        python src/tune.py --env "$TUNE_ENV" --algorithm "$ALGO" --objective "$OBJECTIVE" \
               --n-trials "$N_TRIALS" --study "$STUDY" &
    if [ "$i" -eq 0 ]; then sleep 15; fi
done

sleep 3600
nvidia-smi   # 1h-in GPU snapshot (matches train.sh convention)

wait

# All workers done → export artifacts + best config (single process, no race).
python src/tune.py --env "$TUNE_ENV" --algorithm "$ALGO" --objective "$OBJECTIVE" \
       --study "$STUDY" --report-only
