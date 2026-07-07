# Flappy Bird RL

Training reinforcement-learning agents to play Flappy Bird using vector and pixel observations. Two algorithm families are supported, selected per-config via an `algorithm` key:

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

## Automatic hyperparameter tuning

`src/tune.py` runs an **Optuna** search (TPE sampler + Hyperband pruning) over any of the three algorithms, using the same agents as normal training. Because a full run takes ~24h, the search trains many **short proxy runs** in parallel and prunes weak ones early, then you run only the winner at full length.

`hyperparametertune.sh` packs several trials onto each of 4 A100s in one SLURM job (training is CPU/env-bound, so the GPUs sit near-idle otherwise):

```bash
# Search one algorithm (dqn | rainbow | ppo). Args: <algorithm> [n_trials] [trials_per_gpu]
sbatch hyperparametertune.sh rainbow 400
sbatch hyperparametertune.sh dqn 400

# PPO is replay-free — request less RAM/CPU:
sbatch --mem=64000 --cpus-per-task=24 hyperparametertune.sh ppo 400
```

**Objective:** mean greedy reward over `--eval-episodes` (default 30) deterministic episodes. **What's tuned vs fixed:** fast-acting knobs (learning rate, batch size, PPO `ent_coef`/`clip_eps`/`ppo_epochs`, etc.) are searched; horizon-dependent ones are handled specially — `replay_memory_size` is **fixed** (a short proxy can't fill a large buffer, so it has no signal), while epsilon decay is searched as a **horizon-relative fraction** (`epsilon_frac`) that transfers from the proxy to the full run.

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
| `tune.py` | Optuna hyperparameter search (search spaces, objective, pruning, plots, winner export). The only module that imports Optuna. |
| `utils.py` | Preprocessing pipeline, video recording, sanity check, logging, the shared `save_graph` (labels adapt to DQN vs PPO), and the `flappy_bird_env` render patch. |
| `config.py` | Global constants (`RUNS_DIR`, `CHECKPOINT_EVERY`, `HEADLESS`, etc.). Set `HEADLESS=1` as an environment variable to suppress the display (done automatically in `train.sh`/`evaluate.sh`). |
| `hyperparams.yml` | All training configurations. Each named set maps to one experiment; the `algorithm` key selects DQN or PPO. |
| `environment.yml` | Conda environment for local CPU/MPS training. |
| `environment-cuda.yml` | Conda environment for cluster GPU training (CUDA 12.4). |
| `train.sh` | SLURM training job script for the cluster. Accepts the config name as an argument (works for DQN or PPO). |
| `evaluate.sh` | SLURM evaluation job script. Accepts the config name as an argument. |
| `hyperparametertune.sh` | SLURM job that packs multiple Optuna trials across 4 GPUs for one algorithm, then exports the best config. |

Training outputs (logs, model checkpoints, graphs, videos) are saved under `runs/<config_name>/`. Hyperparameter-search outputs are saved under `runs/hpo/<study>/`.

---