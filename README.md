# Flappy Bird & Super Mario Bros RL

Training reinforcement-learning agents on **two games** from pixel observations, with the same two algorithms and the same entry point. The game is chosen by the config, not the command.

- **Flappy Bird** — vector and pixel observations.
- **Super Mario Bros (NES)** — a level *generalisation* study: train on a subset of levels, evaluate zero-shot on disjoint subsets. See [Super Mario Bros](#super-mario-bros-nes) below.

Two algorithm families are supported, selected per-config via an `algorithm` key:

- **DQN** (value-based, off-policy): standard DQN, Double DQN, Dueling DQN, CNN-DQN, and a full Rainbow variant (PER, n-step, Noisy Nets, distributional C51) via a plug-and-play network registry.
- **PPO** (policy-gradient, on-policy): actor-critic with GAE, clipped surrogate objective, and an entropy bonus for exploration. Available with an MLP (`ppo`) or a CNN backbone (`ppo_cnn`).

Inspired by [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird) and [dqn_pytorch](https://github.com/johnnycode8/dqn_pytorch).

---

## Installation

### Local (Mac / CPU)

```bash
conda env create -f environment.yml
conda activate fp_dqn
```

### Cluster (CUDA)

```bash
conda env create -f environment-cuda.yml
conda activate dqn-flappy-bird-cuda
```

> **Note:** `torch.cuda.is_available()` returns `False` on login nodes — this is expected. CUDA is only available on compute nodes allocated by SLURM.

### Super Mario Bros (separate env)

`gym-super-mario-bros` 9.1.0 requires **Python ≥ 3.13**, so Mario gets its own environment. The
same source tree serves both games: Mario configs set `env_package: gym_super_mario_bros`, which is
what `BaseAgent.__init__` imports instead of `flappy_bird_gymnasium`.

```bash
conda env create -f environment-mario.yml        # local  (mario-rl)
conda env create -f environment-mario-cuda.yml  # cluster (mario-rl-cuda)
```

> On macOS, conda-forge's `llvm-openmp` and the pip torch wheel each ship a `libomp.dylib`, so
> local runs need `KMP_DUPLICATE_LIB_OK=TRUE`. Cluster runs are unaffected.

---

## Running

The **same entry point runs both DQN and PPO** — the algorithm is chosen by the config, not the command. Each set in `hyperparams.yml` carries an `algorithm` key (`"dqn"` by default, or `"ppo"`); `src/agent.py` dispatches to the right agent automatically. So switching from DQN to PPO is just passing a different config name.

### Train locally

```bash
conda activate fp_dqn

# DQN
python src/agent.py flappybird7 --train

# PPO (CNN backbone)
python src/agent.py flappybird_ppo --train
```

### Test (with display)

```bash
python src/agent.py flappybird7        # DQN
python src/agent.py flappybird_ppo     # PPO
```

### Evaluate (100 greedy episodes; Grad-CAM for CNN configs)

```bash
python src/agent.py flappybird_ppo --evaluate
```

### Submit to cluster

The SLURM scripts take the config name as an argument, so they work for **either algorithm** — just pass a PPO config:

```bash
# Train — default config (flappybird_cnn1)
sbatch train.sh

# Train — specific DQN config
sbatch train.sh flappybird8

# Train — PPO config
sbatch train.sh flappybird_ppo

# Evaluate a trained config
sbatch evaluate.sh flappybird_ppo
```

All available configs are defined in `hyperparams.yml`.

### Adding a PPO config

Create a new set in `hyperparams.yml` with `algorithm: "ppo"`. Key PPO parameters (with `flappybird_ppo` defaults): `network_type` (`ppo` MLP or `ppo_cnn`), `rollout_steps` (2048), `ppo_epochs` (10), `minibatch_size` (64), `clip_eps` (0.2), `gae_lambda` (0.95), `vf_coef` (0.5), `ent_coef` (0.01, the exploration knob), and `max_grad_norm` (0.5). For a CNN config, also set `network_type: "ppo_cnn"`, `frame_stack`, `obs_size`, and `rgb_wrapper`.

---

## Super Mario Bros (NES)

A level-generalisation study: train on a subset of the 32 SMB1 stages, evaluate zero-shot on
disjoint subsets. Configs: `mario_rainbow`, `mario_ppo` (multi-level), `mario_rainbow_1_1` /
`mario_ppo_1_1` (single level, for comparison with the CS224R paper), plus `*_debug` smoke tests.

```bash
conda activate mario-rl                                  # locally: prefix KMP_DUPLICATE_LIB_OK=TRUE
python src/mario_probe.py                                # run FIRST: validates the whole env stack
python src/agent.py mario_ppo_debug --train              # ~3k-step smoke test
python src/agent.py mario_ppo --train                    # real run
python src/agent.py mario_ppo --train --resume           # continue after a wall-clock kill
python src/agent.py mario_ppo --evaluate-levels          # four-tier eval + 10 chronological game runs

# cluster
sbatch --export=ALL,CONDA_ENV=mario-rl-cuda train.sh mario_rainbow
```

**Resume is required, not optional.** At ~100–210 agent steps/s, 10M steps exceeds the 24 h SLURM
wall clock, so a full run spans several jobs. `--resume` restores model, optimizer, RNG state and
counters from `<run>_state.pt`; per-episode metrics are appended to `runs/<run>/episodes.csv`, which
survives a `kill -9`. The replay buffer is *not* checkpointed (~17 GB), so a resumed Rainbow run
gates learning for `resume_refill_steps` while the buffer refills — this is logged, not hidden.

**Evaluation tiers** (defined in `src/mario_levels.py`, split `smb1_stage_holdout`):

| Tier | Levels | Question |
|---|---|---|
| `train` | 20 | did it learn at all? (memorisation ceiling) |
| `tier0` | 2 (5-3, 7-2) | **positive control** — layout twins of trained stages. Should be clearly non-zero |
| `tier1` | 7 | in-distribution generalisation to novel layouts |
| `tier2` | 12 Lost Levels | out-of-distribution (novel layouts *and* mechanics) |
| excluded | 4-4, 7-4, 8-4 | routing mazes; appendix only |

Plus, alongside the per-level tiers, **10 chronological runs of the original game** (`SuperMarioBros-v0`, warps allowed, one episode = one playthrough attempt until game over), reporting the mean and max furthest stage reached, stages cleared, and warps used.

**Warps are enabled and rewarded** (`warp_bonus`, paid per stage skipped). One consequence is baked into the split: every warp zone exits to an X-1 stage, so no X-1 stage may be held out — otherwise an agent training on 1-2 could warp straight into the eval set. `mario_levels._validate()` asserts this at import.

The split is a **frozen protocol commitment**, not a random draw: SMB1 reuses whole level layouts
across five stage pairs, so a naive split would silently measure "same layout, different palette".
`mario_levels._validate()` asserts at import that no twin pair straddles train/tier1 and that every
held-out archetype has a training representative. Tier 0 deliberately *does* straddle two twins —
that is what distinguishes a real negative result from a broken eval harness.

Design rationale, verified upstream-behaviour gotchas, and the decision log live in `mario.md`
(gitignored).

---

## Automatic hyperparameter tuning

`src/tune.py` runs an **Optuna** search (TPE sampler + Hyperband pruning) over any of the three algorithms, using the same agents as normal training. Because a full run takes ~24h, the search trains many **short proxy runs** in parallel and prunes weak ones early, then you run only the winner at full length.

`hyperparametertune.sh` packs several trials onto each of 4 A100s in one SLURM job (training is CPU/env-bound, so the GPUs sit near-idle otherwise):

```bash
# Search one algorithm (dqn | rainbow | ppo). Args: <algorithm> [n_trials] [trials_per_gpu]
sbatch --time=48:00:00 hyperparametertune.sh rainbow 400   # Rainbow: give it 48h (see below)
sbatch hyperparametertune.sh dqn 400

# PPO is replay-free — request less RAM/CPU:
sbatch --mem=64000 --cpus-per-task=24 hyperparametertune.sh ppo 400
```

**Packing (`trials_per_gpu`) defaults per algorithm:** PPO / vanilla DQN are env-bound (GPU near-idle), so they pack **3/GPU**. Rainbow is GPU-compute-bound (C51 + noisy nets + dueling saturate the card), so it packs **1/GPU** — overpacking it just slows every trial. You can override with the 3rd arg.

**Objective:** the **25th-percentile** greedy reward over `--eval-episodes` (default 30) deterministic episodes (`--objective p25|median|mean`, default `p25`). p25 is deliberately robust: it ignores lucky one-off long episodes and penalizes unstable configs (those that collapse on some seeds) — a plain mean gets inflated by a single tail episode and selects for entropy-collapse-prone configs. **What's tuned vs fixed:** fast-acting knobs (learning rate, batch size, PPO `ent_coef`/`clip_eps`/`ppo_epochs`, etc.) are searched; horizon-dependent ones are handled specially — `replay_memory_size` is **fixed** (a short proxy can't fill a large buffer, so it has no signal), while epsilon decay is searched as a **horizon-relative fraction** (`epsilon_frac`) that transfers from the proxy to the full run.

**Fairness across algorithms:** the yardstick is identical for all three (same proxy budget, same p25 eval, same `--n-trials`, same pruner) — that's what makes the comparison valid. Because Rainbow is slower and packs 1/GPU, it completes fewer trials in a fixed wall-clock, so give it the full 48h and check `completed trials` in each `summary.txt`; compare studies at roughly equal completed-trial counts.

**Results** land in `runs/hpo/<study>/`:

| File | Contents |
|---|---|
| `<study>.journal` | the Optuna study — the authoritative record of every config tried |
| `trials.csv` | flat table: params + objective value + state, one row per trial |
| `optimization_history.png`, `param_importances.png`, `parallel_coordinate.png`, `slice.png` | search plots |
| `summary.txt` | best value + config |

The best config is appended to `hyperparams.yml` as `flappybird_<algorithm>_tuned`. Run it at full length with the normal script:

```bash
sbatch train.sh flappybird_ppo_tuned
```

Local smoke test (few tiny trials, CPU/MPS):

```bash
python src/tune.py --algorithm ppo --n-trials 4 --proxy-steps 5000 --eval-episodes 5
python src/tune.py --algorithm ppo --study ppo --report-only
```

Per-worker trials run under `runs/hpo/<study>/trials/` (lightweight — no videos) and never touch top-level `runs/`.

---

## Project Structure

| File | Description |
|---|---|
| `agent.py` | Main entry point. Reads the config's `algorithm` key and dispatches to `DQNAgent` or `PPOAgent`; handles `--train`, `--evaluate`, and test modes. |
| `base_agent.py` | Shared `BaseAgent` base class: environment construction, evaluation, Grad-CAM (`explain()`), and the common `test`/`evaluate` loops. |
| `dqn.py` | DQN network definitions (`DQN`, `DuelingDQN`, `CNNDQN`, Rainbow) and the `optimize` function. New architectures can be added to `NETWORK_REGISTRY`. |
| `dqn_agent.py` | `DQNAgent` — the DQN/Rainbow training loop. |
| `experience_replay.py` | DQN buffers: uniform `ReplayMemory` (local RNG for reproducibility), `PrioritizedReplayMemory` (PER via a sum-tree), and `NStepBuffer` (multi-step returns). |
| `ppo.py` | PPO network definitions (`ActorCritic`, `CNNActorCritic`; `PPO_NETWORK_REGISTRY`) and the `ppo_optimize` step (clipped surrogate + value loss + entropy bonus). |
| `ppo_agent.py` | `PPOAgent` — the PPO training loop (collect rollout → GAE → epochs of minibatch updates). |
| `rollout_buffer.py` | On-policy `RolloutBuffer` with GAE advantage estimation; wiped each iteration. |
| `tune.py` | Optuna hyperparameter search (search spaces, objective, pruning, plots, winner export). The only module that imports Optuna. `--env {flappybird,mario}` selects the game. |
| `utils.py` | Preprocessing pipeline, video recording, sanity check, logging, and the shared `save_graph` (labels adapt to DQN vs PPO and to the game). |
| `env_factory.py` | **The single owner of environment construction** — the one place that builds an env plus its observation pipeline, for either game. Replaces three duplicated wrapper chains. |
| `env_metrics.py` | The per-game "secondary metric" layer (pipes vs pages cleared), episode-length FPS, graph labels, and Grad-CAM action names. Derived from `env_id`, so a config cannot pick the wrong one. |
| `checkpointing.py` | Resume support (`save_run_state` / `load_run_state`: model + optimizer + counters + RNG), the append-only per-episode CSV, and the per-step metric striding that bounds memory on long runs. |
| `mario_levels.py` | Mario level inventory, archetypes, documented layout twins, and the frozen train/eval splits. Stdlib only, so it imports under either conda env. Self-validates the split at import. |
| `mario_env.py` | Mario wrappers (`MarioSanitizeX`, `MarioAreaRebase`, `MarioWarpGuard`, `MarioEpisodeInfo`, no-op/sticky/reward-scale) and `MultiLevelMarioEnv`, which switches level on `reset()`. Imported lazily. |
| `mario_eval.py` | Four-tier per-level evaluation plus the chronological full-game runs; macro aggregates with bootstrap CIs, jackknife sensitivity, 8×4 heatmaps and bar charts. |
| `mario_probe.py` | Pre-flight checks for the Mario stack — **run it before any training**. Each check guards an invariant that can regress (see `mario.md`). |
| `config.py` | Global constants (`RUNS_DIR`, `CHECKPOINT_EVERY`, `HEADLESS`, etc.). Set `HEADLESS=1` as an environment variable to suppress the display (done automatically in `train.sh`/`evaluate.sh`). |
| `hyperparams.yml` | All training configurations. Each named set maps to one experiment; the `algorithm` key selects DQN or PPO. |
| `environment.yml` | Conda environment for local CPU/MPS training (Flappy Bird, python 3.11). |
| `environment-cuda.yml` | Conda environment for cluster GPU training (CUDA 12.4). |
| `environment-mario.yml` / `-cuda.yml` | Mario environments (python 3.13 — required by `gym-super-mario-bros` 9.1). |
| `train.sh` | SLURM training job script. Accepts the config name; set `CONDA_ENV` to pick the environment (defaults to the Flappy Bird one). |
| `evaluate.sh` | SLURM evaluation job script. Accepts the config name as an argument. |
| `hyperparametertune.sh` | SLURM job that packs multiple Optuna trials across 4 GPUs for one algorithm, then exports the best config. |

Training outputs (logs, model checkpoints, graphs, videos) are saved under `runs/<config_name>/`. Hyperparameter-search outputs are saved under `runs/hpo/<study>/`. Mario runs additionally write `episodes.csv` (one row per episode, durable across crashes), `<run>_state.pt` (resume state), and per-level evaluation under `runs/<config_name>/evaluation/mario/`.

---