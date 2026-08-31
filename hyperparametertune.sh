#!/bin/bash
#SBATCH --job-name=flappybird-hpo
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:4              # pack ~12 trials (3/GPU); you are GPU-underutilized
#SBATCH --cpus-per-task=16        # ~2.5 cores/trial — NOT the 64 max
#SBATCH --mem=192000             # ~192GB: 12 Rainbow trials @ ~15GB replay + overhead
#SBATCH --time=48:00:00          # Mario studies need it; override for flappybird (sbatch --time=32:00:00)
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --signal=B:USR1@600      # USR1 to the batch script 10 min before the wall clock, so the
                                 # report still runs when the study is cut off mid-flight

# Unified Optuna hyperparameter search for one algorithm.
#
# Usage: sbatch hyperparametertune.sh <algorithm> [n_trials] [trials_per_gpu]
#   <algorithm>     one of: dqn | rainbow | ppo
#   n_trials        target total trials in the study        (default 400)
#   trials_per_gpu  concurrent trials packed onto each GPU  (default: rainbow=1, dqn/ppo=3)
#
# Env-var overrides (defaults keep every existing FlappyBird invocation identical):
#   TUNE_ENV   flappybird | mario     which game to tune            (default flappybird)
#   OBJECTIVE  p25 | median | mean    eval statistic to maximise    (default: p25 on
#                                    flappybird, mean on mario -- set per TUNE_ENV below)
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
# Env-aware default so it cannot be forgotten: p25 on FlappyBird (unbounded eval episodes make
# reward heavy-tailed, so the mean is outlier-driven), mean on Mario (with 20 levels and mostly
# zero flag rates, p25 collapses to ~0 and stops discriminating). Override explicitly if needed.
if [ "$TUNE_ENV" = "mario" ]; then
    OBJECTIVE=${OBJECTIVE:-mean}
else
    OBJECTIVE=${OBJECTIVE:-p25}
fi
PROXY_STEPS=${PROXY_STEPS:-}     # override the per-env/algo default
FULL_STEPS=${FULL_STEPS:-}       # max_env_steps written into the exported winner config
# Per-algorithm packing. Rainbow is neither GPU- nor RAM-bound now that trials use a
# proxy-scaled replay buffer (50k on Mario = 2.8 GB/worker, see build_trial_config), so the
# 16 cores of the node are the limit. Measured: 4 vs 8 workers gave the same aggregate throughput
# (13.4 vs 13.6 env steps/s), i.e. packing more only makes each trial slower -- study 6715345 ran
# 8 workers and finished 0 of 18 trials in 48h, while 6736259 ran 4 and completed 36 of 80.
# PPO / vanilla DQN are env-bound and pack fine.
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
               --n-trials "$N_TRIALS" --study "$STUDY" \
               ${PROXY_STEPS:+--proxy-steps "$PROXY_STEPS"} &
    if [ "$i" -eq 0 ]; then sleep 15; fi
done

# Artifacts are written by the LAST line of this script, so a job killed at the wall clock
# used to leave nothing but the journal. Run it from a trap instead: on USR1 (600s before the
# limit) and on EXIT, guarded so it only happens once.
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
nvidia-smi     # 1h-in GPU snapshot (matches train.sh convention)

wait           # all workers
