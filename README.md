# Rainbow DQN vs. PPO on Flappy Bird and Super Mario Bros.

A compute-matched comparison of two reinforcement learning methods learning from pixels alone.
Both agents get the same environment, the same observation pipeline, the same reward transform
and the same wall-clock budget; only the algorithm differs. On Mario they train on 20 levels and
are evaluated on 21 held-out ones, 12 of them from *Super Mario Bros.: The Lost Levels*, to
measure generalisation rather than memorisation of a single stage.

## Layout

```
src/            all Python; run everything from the repo root
scripts/        SLURM batch scripts
environments/   conda environment files
hyperparams.yml the four reported configurations
runs/           training artifacts, evaluations, figures
```

| Module | Purpose |
|---|---|
| `agent.py` | CLI entry point: train, evaluate, per-level evaluate |
| `base_agent.py` | Shared agent: config loading, run directories, greedy rollout, evaluation, Grad-CAM |
| `dqn_agent.py` / `ppo_agent.py` | The two training loops; they differ only in `_policy_step` for evaluation |
| `dqn.py` / `ppo.py` | Networks and gradient updates. `NETWORK_REGISTRY` maps a config's `network_type` to a class |
| `experience_replay.py` | Uniform and prioritised replay (sum tree), n-step buffer |
| `rollout_buffer.py` | On-policy rollout buffer with GAE |
| `env_factory.py` | The one place an env plus its observation pipeline is built |
| `env_metrics.py` | Per-game secondary metric (pipes vs. pages), axis labels, frame rate |
| `mario_env.py` | Mario wrappers and the multi-level env |
| `mario_levels.py` | Level inventory, archetypes, the frozen train/eval splits |
| `mario_eval.py` | Per-level evaluation, tier aggregates, figures, videos |
| `mario_probe.py` | Pre-flight checks for the Mario stack |
| `mario_rom.py` / `mario_route_table.py` | Build the route table used for normalised progress |
| `checkpointing.py` | Resume state, per-episode CSV, replay-buffer checkpoint |
| `tune.py` | Optuna hyperparameter search |
| `record_video.py` | Record a policy, streaming frames to ffmpeg |
| `utils.py` | Logging, plots, observation preprocessing, action selection |

## Installation

Two conda environments are needed: Mario requires Python 3.13 (`gym-super-mario-bros`), Flappy
Bird runs on 3.11. Use the `-cuda` files on a GPU node.

```bash
conda env create -f environments/flappybird.yml        # -> dqn-flappy-bird
conda env create -f environments/mario.yml             # -> mario-rl
conda env create -f environments/flappybird-cuda.yml   # -> dqn-flappy-bird-cuda
conda env create -f environments/mario-cuda.yml        # -> mario-rl-cuda
```

On macOS, `export KMP_DUPLICATE_LIB_OK=TRUE` before running anything locally.

## The four reported configurations

All use seed 42 and the same observation pipeline: 4 stacked grayscale frames, 80×80 for Flappy
Bird and 84×84 for Mario, scaled to [0, 1].

| Config | Algorithm | Hidden width | Game |
|---|---|---|---|
| `flappybird_ppo_tuned` | PPO | 512 | Flappy Bird |
| `flappybird_rainbow_tuned` | Rainbow DQN | 256 | Flappy Bird |
| `mario_ppo_tuned` | PPO | 512 | Super Mario Bros. |
| `mario_rainbow_tuned` | Rainbow DQN | 512 | Super Mario Bros. |

## Running

Everything is invoked from the repo root, so relative paths resolve.

```bash
# train
python src/agent.py flappybird_ppo_tuned --train
python src/agent.py mario_rainbow_tuned --train --resume       # continue from <run>_state.pt

# evaluate: N greedy episodes, writes metrics.json + Grad-CAM + chart
python src/agent.py flappybird_ppo_tuned --evaluate --episodes 25
python src/agent.py flappybird_ppo_tuned --evaluate --episodes 10 --resume   # add 10 more

# Mario: per-level evaluation over the tiers, plus chronological full-game runs
python src/agent.py mario_ppo_tuned --evaluate-levels
python src/agent.py mario_ppo_tuned --evaluate-levels train,tier1 --episodes-per-level 30
```

### On the cluster

**Size the allocation before you submit.** The `#SBATCH` lines in `scripts/` are defaults that
suit a Flappy Bird PPO run; they are not right for every job, and memory in particular is not
symmetric between the two algorithms. Rainbow holds a replay buffer of stacked frames, PPO holds
nothing between updates:

| Job | Replay buffer | Measured peak | Sensible `--mem` |
|---|---|---|---|
| Mario PPO | — | 2.9 GB | 16000 |
| Mario Rainbow | 400k x (4,84,84) = 22.7 GB | 36.2 GB | 64000 |
| Flappy Bird PPO | — | ~3 GB | 16000 |
| Flappy Bird Rainbow | 1M x (4,80,80) = 51.4 GB | — | 96000 |

Override what the job needs on the `sbatch` line rather than editing the scripts:

```bash
sbatch scripts/train.sh flappybird_ppo_tuned

sbatch --mem=64000 --time=12:00:00 \
       --export=ALL,CONDA_ENV=mario-rl-cuda scripts/train.sh mario_rainbow_tuned

sbatch --export=ALL,CONDA_ENV=mario-rl-cuda \
       scripts/evaluate.sh mario_ppo_tuned --evaluate-levels
```

Extra flags after the config name are passed through to `agent.py`. Check your quota before
starting a Rainbow run: `--resume` also writes the replay buffer to disk (~23 GB on Mario).

### Evaluation flags

| Flag | Default | Effect |
|---|---|---|
| `--episodes` | 100 | Greedy episodes for `--evaluate`; with `--resume` these are *added* to `evaluation/episodes_eval.csv` |
| `--evaluate-levels [TIERS]` | all four tiers | Per-level Mario evaluation; `train,tier0,tier1,tier2` |
| `--episodes-per-level` | 30 | Episodes per level |
| `--full-game-runs` | 10 | Chronological runs of the original game, warps allowed; 0 to skip |
| `--policy` | `argmax` | `argmax`, `stochastic` or `topk3` (the last two are PPO-only) |
| `--levels` | — | Evaluate only these levels, e.g. `1-1,5-3` |
| `--videos` | `per_world` | `per_world`, `all`, `none`, or a list |
| `--videos-per-level` | 1 | Clips per level, best episodes first |
| `--grad-cam` | `best` | `best`, `all`, `none`, or a list of levels |
| `--grad-cam-frames` / `--grad-cam-stride` | 10 / 25 | The episode stops once the frames are collected, so at most `frames × stride` agent steps are simulated |

### Recording a video

`--evaluate` writes no mp4: a converged Flappy Bird policy survives ~758k frames, and buffering
those in RAM would need ~335 GB. `record_video.py` streams frames straight into ffmpeg instead,
which is constant in memory.

```bash
python src/record_video.py flappybird_ppo_tuned --stream            # whole episode
python src/record_video.py mario_ppo_tuned --seconds 45 --level 1-1
sbatch scripts/stream_video.sh flappybird_ppo_tuned
```

## Environments

### Flappy Bird

`flappy-bird-gymnasium` 0.4.0, played from the rendered frame rather than the 12-dimensional
feature vector. Rewards are already normalised: +1 per pipe, +0.1 per surviving step, −0.5 for
leaving the top of the screen, −1 on collision, so no reward transform is applied.

An episode ends only on collision and the environment registers no step limit, so a competent
policy plays indefinitely; episodes are stopped once the return reaches 10⁵. Wherever that cap
binds, the reported reward measures the cap rather than the policy.

The upstream implementation leaks three variables across episodes (wing animation, ground scroll
offset, flap flag), all of which reach the rendered frame. `FlappyBirdResetFix` resets them, which
is what makes an episode a pure function of its seed.

### Super Mario Bros.

`gym-super-mario-bros` 9.1.0 on `nes-py` 9.0.1, COMPLEX_MOVEMENT (12 actions), 4-frame skip.
Sticky actions (p = 0.25) and up to 30 no-op starts decorrelate the deterministic emulator; both
are active during evaluation as well as training.

**Reward transform.** One agent step spans 4 emulator frames whose rewards are summed. The
environment already clamps each *frame* to its declared range of ±15, so the sum is clipped to
±15 again and divided by 15 — one unit is one maximal frame reward as the environment defines it.
The +50 flag bonus is exempt from that clip and re-added from the info dictionary, because the
per-frame clamp has already reduced it to +15 and no setting above the environment can recover
it. Without the exemption a completed level scores +1.000 against the +0.800 of sustained
running, which asks the agent to travel right rather than to finish. See `ClipScaleReward` in
`mario_env.py`.

**Level splits** (`mario_levels.py`). `smb1_stage_holdout` is the protocol split:

| Tier | n | Contents |
|---|---|---|
| `train` | 20 | Trained on |
| `tier0` | 2 | Layout twins of trained stages — a positive control: if tier1/tier2 are 0 % everywhere, this is what separates a real negative result from a broken harness |
| `tier1` | 7 | Novel SMB1 layouts |
| `tier2` | 12 | *The Lost Levels* — novel layouts and novel mechanics |
| excluded | 3 | Maze stages, where horizontal progress is not monotone in skill |

Layout twins are kept on the same side of the split, except the two pairs deliberately straddled
to form tier 0. No X-1 stage is held out, because warp zones exit to them and warps are enabled.

## Hyperparameter search

```bash
sbatch scripts/hyperparametertune.sh ppo 200
sbatch --export=ALL,CONDA_ENV=mario-rl-cuda,TUNE_ENV=mario \
       scripts/hyperparametertune.sh rainbow 200
```

Optuna with a multivariate TPE sampler and Hyperband pruning; many workers share one file-based
study. Nine parameters are searched for PPO and eight for Rainbow, over ranges identical in both
games. `TUNE_ENV` is mandatory for Mario. Results land in `runs/hpo/<study>/` and the winner is
appended to `hyperparams.yml` as `<env>_<algo>_tuned`.

The objective differs by game and defaults accordingly: the 25th percentile of episode return on
Flappy Bird, where unbounded episodes make the mean outlier-driven, and the mean on Mario, where
mostly-zero flag rates collapse the percentile to zero.

## Checks

`mario_probe.py` runs the Mario stack's invariants — observation shapes, the x-underflow guard,
area rebasing, decorrelation, warp tracking, the level splits, and that both algorithm configs
face an identical task:

```bash
python src/mario_probe.py
python src/mario_probe.py --only spaces,fps
```

`mario_levels.py` additionally validates the split at import: no overlap between tiers, no twin
straddling outside tier 0, every eval archetype represented in training.

## Reproducing the reported results

All four runs use seed 42 on one A100. Flappy Bird fits a single 24-hour job; Mario needs
about 30 hours, which the partition's wall clock does not allow in one go, so it is run in
blocks of 12 + 12 + 6 hours joined by `--resume`.

```bash
# Flappy Bird: one 24 h block each
sbatch                --time=24:00:00 --mem=16000 scripts/train.sh flappybird_ppo_tuned
sbatch                --time=24:00:00 --mem=96000 scripts/train.sh flappybird_rainbow_tuned

# Mario: ~30 h as 12 + 12 + 6, the second and third block with --resume
M="--export=ALL,CONDA_ENV=mario-rl-cuda"
sbatch $M --time=12:00:00 --mem=16000 scripts/train.sh mario_ppo_tuned
sbatch $M --time=12:00:00 --mem=16000 scripts/train.sh mario_ppo_tuned --resume
sbatch $M --time=06:00:00 --mem=16000 scripts/train.sh mario_ppo_tuned --resume

sbatch $M --time=12:00:00 --mem=64000 scripts/train.sh mario_rainbow_tuned
sbatch $M --time=12:00:00 --mem=64000 scripts/train.sh mario_rainbow_tuned --resume
sbatch $M --time=06:00:00 --mem=64000 scripts/train.sh mario_rainbow_tuned --resume

# evaluation
sbatch    scripts/evaluate.sh flappybird_ppo_tuned --evaluate --episodes 25
sbatch $M scripts/evaluate.sh mario_ppo_tuned --evaluate-levels
```

Reported numbers come from the best checkpoint of a run — selected by a greedy probe episode
during training — not from the final weights. Each run is a single seed: the wall-clock budget
did not allow repetition, so no difference carries a variance estimate.

A Mario run does not fit one SLURM job. `--resume` continues from `<run>_state.pt`, which carries
the model, optimizer, RNG state and metric series. Rainbow additionally writes its replay buffer
once at the wall clock, so a resumed run continues with a full buffer instead of refilling for
~1.4 hours; that file is large (~23 GB on Mario) and is not tracked.

## Watching the agents

The clips are committed, so you can just open them.

`runs/<config>/checkpoint_videos/` shows training progress: one greedy episode saved every so
often, named `checkpoint_ep<episode>_r<reward>.mp4`. Sorted by episode you can watch the policy
come together. The reward in the name is where that episode ended, almost always by dying.

`runs/<mario config>/evaluation/mario/videos/` has one clip per level from its best evaluation
episode, `level_<level>_end<stage>_pages<n>.mp4`, where `pages` is how far it got. The held-out
levels carry their Lost Levels id (`level_SuperMarioBros2-...`). `fullgame_best` and
`fullgame_worst` are chronological runs of the original game.

The really long recordings are not in here. A converged Flappy Bird policy does not stop on its
own — one uncapped episode ran 18 hours of game time before it finally died — and files like that
have no business in a git repository. Use `scripts/stream_video.sh` if you want one.

## Credits

The convolutional trunk borrows from two sources and departs from both. Kernel sizes, strides
and channel counts are those of Mnih et al. (2015), which reduces resolution by striding alone;
[yenchenlin/DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird) pools
after every convolution. This implementation pools once, after the first convolution, which is
neither. The Flappy Bird preprocessing otherwise follows yenchenlin.

Games: [flappy-bird-gymnasium](https://github.com/markub3327/flappy-bird-gymnasium) and
[gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros).
