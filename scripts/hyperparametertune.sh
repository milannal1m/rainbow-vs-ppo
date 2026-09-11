#!/bin/bash
#SBATCH --job-name=rl-hpo
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=48000              # 4 Rainbow workers need 23-32 GB with the proxy-scaled buffer.
                                 # PPO is replay-free but packs 12 workers and measured 31-39 GB:
                                 # --mem=64000 --cpus-per-task=24 for it.
#SBATCH --time=48:00:00          # Mario needs it; FlappyBird studies finish in ~32 h
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@600      # so the report still runs if the study is cut off mid-flight

# Optuna hyperparameter search for one algorithm, many workers sharing one file-based study.
#
# Usage (submit from the repo root):
#   sbatch scripts/hyperparametertune.sh <dqn|rainbow|ppo> [n_trials] [trials_per_gpu]
#   sbatch --export=ALL,CONDA_ENV=mario-rl-cuda,TUNE_ENV=mario \
#          scripts/hyperparametertune.sh rainbow 200
#
# Env vars: TUNE_ENV (flappybird|mario), OBJECTIVE (p25|median|mean), PROXY_STEPS, FULL_STEPS.
# TUNE_ENV is mandatory for Mario -- without it the workers build a FlappyBird config and die on
# the missing flappy_bird_gymnasium import.
#
# Results land in runs/hpo/<study>/ (journal, trials.csv, plots, summary.txt) and the winner is
# appended to hyperparams.yml as <env>_<algorithm>_tuned.

ALGO=${1:-ppo}
N_TRIALS=${2:-400}
TUNE_ENV=${TUNE_ENV:-flappybird}

# Env-aware so it cannot be forgotten on the command line. FlappyBird needs p25: eval episodes
# there are unbounded, so reward is heavy-tailed and one lucky episode drives the mean. Mario
# needs the mean: across 20 levels with mostly-zero flag rates p25 collapses to ~0.
if [ "$TUNE_ENV" = "mario" ]; then
    OBJECTIVE=${OBJECTIVE:-mean}
else
    OBJECTIVE=${OBJECTIVE:-p25}
fi
PROXY_STEPS=${PROXY_STEPS:-}
FULL_STEPS=${FULL_STEPS:-}

# PPO packs 3/GPU: no replay buffer (~2.9 GB/worker against Rainbow's 16.8 GB at 300k) and far
# less gradient work per env step. Rainbow stays at 1/GPU -- packing 8 once completed 0 of 18
# trials against 4 workers completing 36 of 80, though the cause was memory thrashing rather than
# CPU. With the proxy-scaled buffer 12 workers would now fit, so 1/GPU is conservative rather
# than measured for the current configuration; re-measure before raising it.
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
CONDA_ENV=${CONDA_ENV:-dqn-flappy-bird-cuda}
conda activate "$CONDA_ENV"

export HEADLESS=1
export OMP_NUM_THREADS=2          # keep packed processes from oversubscribing the cores

N_GPUS=${SLURM_GPUS_ON_NODE:-4}
K=$(( N_GPUS * TRIALS_PER_GPU ))
if [ "$TUNE_ENV" = "flappybird" ]; then
    STUDY="${ALGO}_${SLURM_JOB_ID}"          # unprefixed, so old journals stay resumable by name
else
    STUDY="${TUNE_ENV}_${ALGO}_${SLURM_JOB_ID}"
fi

echo "HPO: env=$TUNE_ENV algorithm=$ALGO objective=$OBJECTIVE n_trials=$N_TRIALS study=$STUDY workers=$K gpus=$N_GPUS"

# Round-robin across the allocated GPUs. The first worker gets a head start so it creates the
# study before the rest attach.
for (( i=0; i<K; i++ )); do
    CUDA_VISIBLE_DEVICES=$(( i % N_GPUS )) \
        python src/tune.py --env "$TUNE_ENV" --algorithm "$ALGO" --objective "$OBJECTIVE" \
               --n-trials "$N_TRIALS" --study "$STUDY" \
               ${PROXY_STEPS:+--proxy-steps "$PROXY_STEPS"} &
    if [ "$i" -eq 0 ]; then sleep 15; fi
done

# Write the artifacts from a trap, on USR1 and on EXIT, guarded so it happens once: a job killed
# at the wall clock would otherwise leave nothing but the journal.
REPORT_ARGS=(--env "$TUNE_ENV" --algorithm "$ALGO" --objective "$OBJECTIVE" --study "$STUDY")
[ -n "$FULL_STEPS" ] && REPORT_ARGS+=(--full-steps "$FULL_STEPS")
REPORTED=0
write_report() {
    [ "$REPORTED" = "1" ] && return 0
    REPORTED=1
    echo "[report] writing artifacts for $STUDY"
    python src/tune.py "${REPORT_ARGS[@]}" --report-only || true
}
trap 'echo "[signal] USR1 -- wall clock near, reporting early"; write_report; exit 0' USR1
trap write_report EXIT

sleep 3600 &
wait $!        # backgrounded so the USR1 trap can fire during it
nvidia-smi

wait           # all workers
