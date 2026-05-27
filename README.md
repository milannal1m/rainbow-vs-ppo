# Flappy Bird DQN

Training a Deep Q-Network agent to play Flappy Bird using vector and pixel observations. Supports standard DQN, Double DQN, Dueling DQN, and CNN-based DQN via a plug-and-play network registry.

Inspired by [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird) and [dqn_pytorch](https://github.com/johnnycode8/dqn_pytorch).

---

## Installation

### Local (Mac / CPU)

```bash
conda env create -f environment.yml
conda activate flappybird-nocnn
```

### Cluster (CUDA)

```bash
conda env create -f environment-cuda.yml
conda activate dqn-flappy-bird-cuda
```

> **Note:** `torch.cuda.is_available()` returns `False` on login nodes — this is expected. CUDA is only available on compute nodes allocated by SLURM.

---

## Running

### Train locally

```bash
conda activate flappybird-nocnn
python agent.py flappybird7 --train
```

### Test (with display)

```bash
python agent.py flappybird7
```

### Submit to cluster

```bash
# Default config (flappybird_cnn1)
sbatch slurm.sh

# Specific config
sbatch slurm.sh flappybird8
```

All available configs are defined in `hyperparams.yml`.

---

## Project Structure

| File | Description |
|---|---|
| `agent.py` | Main entry point. `Agent` class handles training and testing loop. |
| `dqn.py` | Network definitions (`DQN`, `DuelingDQN`, `CNNDQN`) and the `optimize` function. New architectures can be added to `NETWORK_REGISTRY`. |
| `experience_replay.py` | Replay buffer with a local RNG (not global) for reproducibility. |
| `utils.py` | Preprocessing pipeline, video recording, sanity check, logging, and the `flappy_bird_env` render patch. |
| `config.py` | Global constants (`RUNS_DIR`, `CHECKPOINT_EVERY`, `HEADLESS`, etc.). Set `HEADLESS=1` as an environment variable to suppress the display (done automatically in `slurm.sh`). |
| `hyperparams.yml` | All training configurations. Each named set maps to one experiment. |
| `environment.yml` | Conda environment for local CPU/MPS training. |
| `environment-cuda.yml` | Conda environment for cluster GPU training (CUDA 12.4). |
| `slurm.sh` | SLURM job script for the cluster. Accepts the config name as an argument. |

Training outputs (logs, model checkpoints, graphs, videos) are saved under `runs/<config_name>/`.

---