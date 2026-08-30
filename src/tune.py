"""Unified Optuna hyperparameter tuning for DQN / Rainbow / PPO.

This is the ONLY module that knows about Optuna. It drives the existing agents through
their public hooks (`hyperparams` dict injection, `run_name`, `max_env_steps`, the
`report_cb`/`record_video` args on `train()`, and the `record_artifacts` arg on
`evaluate()`); it never modifies them.

Two modes:
  * worker (default): run `study.optimize()` — many workers share one file-based study.
  * `--report-only`: load a finished study, write trials.csv + plots + summary, and append
    the best config to hyperparams.yml as `<env>_<algo>_tuned`
    (`flappybird_<algo>_tuned` for the default env).

Run from the repo root, e.g.:
    python src/tune.py --algorithm ppo --n-trials 40 --proxy-steps 300000
    python src/tune.py --algorithm ppo --study ppo_123 --report-only
    python src/tune.py --env mario --algorithm rainbow --n-trials 60
"""
import os
# Headless rendering for the pixel pipeline (must precede any gym/env import).
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import csv
import sys

import numpy as np
import torch
import yaml
import optuna

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import RUNS_DIR

# Respect the per-worker thread cap set by hyperparametertune.sh so packed trials don't
# oversubscribe the node's cores.
torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))

BASE_SEED = 42
HPO_DIR = os.path.join(RUNS_DIR, "hpo")

# Throttle intermediate pruning reports to at most one per this many env steps. DQN/Rainbow
# would otherwise report per-episode (~50 steps) and, with many workers on a shared journal,
# hammer the file lock — the cause of the Rainbow study starving out. PPO already reports
# per-iteration (~1-4k steps), so this barely affects it.
REPORT_INTERVAL_STEPS = 5000

# Objective statistic over the greedy-eval episodes. p25 (default) is robust to lucky-tail
# episodes AND penalises unstable configs (those dying on >25% of seeds), unlike the mean,
# which the first study inflated 27x on a single lucky episode.
OBJECTIVE_KEYS = {"p25": "eval_reward_p25", "median": "eval_reward_median", "mean": "eval_reward_mean"}

# ── Base configs (fixed keys per algorithm) ──────────────────────────────────────────
# Pixel pipeline matching the hand-tuned flappybird_ppo / flappybird_cnn / flappybird_rainbow
# sets. The tuner overrides the searched keys on top of these.
# Env blocks are merged onto the algorithm base at build time, so the search spaces stay
# per-algorithm while the env is a separate CLI dimension.
_ENV_FLAPPYBIRD = {
    "env_id": "FlappyBird-v0",
    "env_make_params": {"use_lidar": False, "background": None},
    "frame_stack": 4,
    "obs_size": 80,
    "rgb_wrapper": True,
}

# Mario trials use the SAME 20 training levels as the real run. smb1_hpo (4 levels, all
# ground archetype) was unfaithful: it tuned a homogeneous 4-task problem while the run is a
# heterogeneous 20-task one, and the winning config lost to a hand-set LR schedule. Measured on
# two configs over these 20 levels, x_pos_max separates them by 31% at 1M and 52% at 3M, so the
# thinner 150k/level budget still ranks. Selection must only ever read training levels —
# scoring on a held-out tier would be test-set selection.
_ENV_MARIO = {
    "env_id": "SuperMarioBros-v0",
    "env_package": "gym_super_mario_bros",
    "env_make_params": {
        "level_split": "smb1_stage_holdout", "level_set": "train", "level_sampler": "seed_hash",
        "action_set": "COMPLEX_MOVEMENT", "frame_skip": 4,
        "reward_clip": 15.0, "reward_divisor": 15.0,
        "noop_max": 30, "sticky_prob": 0.25,
        "max_episode_steps": 3000, "warp_bonus": 5.0,
    },
    "frame_stack": 4,
    "obs_size": 84,
    "rgb_wrapper": False,
}

ENVS = {"flappybird": _ENV_FLAPPYBIRD, "mario": _ENV_MARIO}

# Env-specific overrides of algorithm-base keys.
ENV_ALGO_OVERRIDES = {
    # Mario's steady-state discounted value is ~59, so FlappyBird's v_max=20 would clamp
    # essentially every C51 target.
    # The four knobs below are FIXED for Mario, not searched: the 56-trial FlappyBird Rainbow
    # study put them last by fANOVA importance (v_max .008, n_step .019, batch_size .044,
    # network_sync_rate .075 -> 15% of variance combined), and with ~29 affordable Mario trials
    # the budget is better spent on sigma_init/lr/per_* (85%). Values chosen to match the best
    # Mario trials so far (t2/t3: batch 32, sync 2000) and the C51 arithmetic (steady-state
    # discounted value ~59, so v_max 100 leaves headroom without wasting atoms).
    ("mario", "rainbow"): {"n_atoms": 101, "v_min": -15.0, "v_max": 100.0,
                           "replay_memory_size": 300000, "per_beta_frames": 5000000,
                           "batch_size": 32, "n_step": 3, "network_sync_rate": 2000},
    ("mario", "dqn"):     {"replay_memory_size": 300000},
    ("mario", "ppo"):     {"rollout_steps": 4096},
}

_ENV = _ENV_FLAPPYBIRD  # backwards-compatible alias

PPO_BASE = {
    "algorithm": "ppo",
    "network_type": "ppo_cnn",
    "hidden_dim": 512,
    "discount_factor_g": 0.99,
    "stop_on_reward": 100000,
    "seed": BASE_SEED,
    "max_grad_norm": 0.5,
    "lr_min": 1e-5,          # floor for linear LR annealing (see build_*_config)
}

DQN_BASE = {
    "algorithm": "dqn",
    "network_type": "cnn_dqn",
    "hidden_dim": 512,
    "discount_factor_g": 0.99,
    "stop_on_reward": 100000,
    "seed": BASE_SEED,
    "enable_double_dqn": True,
    "replay_memory_size": 200000,       # FIXED (proxy-blind): benefit confirmed in full runs
    "epsilon_init": 1.0,
    "epsilon_min": 0.01,                # > 0 so the epsilon_frac schedule is well-defined
    "epsilon_decay": 0.999995,          # fallback if epsilon_frac is absent
    "network_sync_rate": 1000,
    "start_learning_after": 20000,
}

RAINBOW_BASE = {
    "algorithm": "dqn",
    "network_type": "rainbow_cnn_dqn",
    "hidden_dim": 512,
    "discount_factor_g": 0.99,
    "stop_on_reward": 100000,
    "seed": BASE_SEED,
    "replay_memory_size": 1000000,      # FIXED (proxy-blind)
    "batch_size": 32,
    "epsilon_init": 1.0, "epsilon_decay": 1.0, "epsilon_min": 0.0,  # unused: noisy nets
    "network_sync_rate": 1000,
    "start_learning_after": 20000,
    "enable_double_dqn": True, "use_dueling": True, "use_per": True,
    "use_nstep": True, "use_noisy": True, "use_distributional": True,
    "n_atoms": 51, "v_min": -5.0, "v_max": 20.0,
    "per_alpha": 0.5, "per_beta_init": 0.4, "per_beta_frames": 2000000,
    "sigma_init": 0.5,
}

BASE = {"ppo": PPO_BASE, "dqn": DQN_BASE, "rainbow": RAINBOW_BASE}


def suggest_params(trial, algorithm, env="flappybird"):
    """Search space per algorithm. Only fast-acting / horizon-relative knobs are tuned;
    proxy-blind params (e.g. replay_memory_size) stay fixed in the base config."""
    if algorithm == "ppo":
        return {
            "learning_rate_a": trial.suggest_float("learning_rate_a", 1e-5, 1e-3, log=True),
            "ppo_epochs":      trial.suggest_categorical("ppo_epochs", [3, 4, 6, 10]),
            # Floor raised 1e-3 -> 5e-3: the first study drove ent_coef to ~0.001 (entropy
            # collapse -> unstable full runs). Keep it above the collapse regime.
            "ent_coef":        trial.suggest_float("ent_coef", 5e-3, 5e-2, log=True),
            "clip_eps":        trial.suggest_categorical("clip_eps", [0.1, 0.2, 0.3]),
            "gae_lambda":      trial.suggest_float("gae_lambda", 0.9, 0.99),
            "vf_coef":         trial.suggest_float("vf_coef", 0.3, 1.0),
            # Floor raised to 4096: small rollouts (1024) give high-variance advantage
            # estimates -> jittery policy over a long run. Larger rollouts smooth updates.
            "rollout_steps":   trial.suggest_categorical("rollout_steps", [4096, 8192]),
            "minibatch_size":  trial.suggest_categorical("minibatch_size", [64, 128, 256]),
            "hidden_dim":      trial.suggest_categorical("hidden_dim", [256, 512]),
        }
    if algorithm == "dqn":
        return {
            "learning_rate_a":   trial.suggest_float("learning_rate_a", 1e-5, 1e-3, log=True),
            "batch_size":        trial.suggest_categorical("batch_size", [32, 64, 128, 512]),
            "network_sync_rate": trial.suggest_categorical("network_sync_rate", [500, 1000, 2000, 5000]),
            "epsilon_frac":      trial.suggest_float("epsilon_frac", 0.1, 0.6),  # horizon-relative
            "hidden_dim":        trial.suggest_categorical("hidden_dim", [256, 512]),
        }
    if algorithm == "rainbow":
        p = {
            "learning_rate_a":   trial.suggest_float("learning_rate_a", 1e-5, 5e-4, log=True),
            "per_alpha":         trial.suggest_float("per_alpha", 0.3, 0.7),
            "per_beta_init":     trial.suggest_float("per_beta_init", 0.3, 0.6),
            "sigma_init":        trial.suggest_float("sigma_init", 0.3, 0.7),
        }
        if env != "mario":
            # Mario fixes these in ENV_ALGO_OVERRIDES — see the note there.
            p.update({
                "batch_size":        trial.suggest_categorical("batch_size", [32, 64, 128]),
                "network_sync_rate": trial.suggest_categorical("network_sync_rate", [500, 1000, 2000, 8000]),
                "n_step":            trial.suggest_categorical("n_step", [1, 3, 5]),
                # ~59 on Mario vs ~10 on FlappyBird, so one categorical cannot serve both
                "v_max":             trial.suggest_categorical("v_max", [15.0, 20.0, 30.0]),
            })
        return p
    raise ValueError(f"unknown algorithm: {algorithm}")


def _merge_env(cfg, env, algorithm):
    """Layer the env block and any env-specific base overrides onto an algorithm base."""
    cfg.update(ENVS[env])
    cfg.update(ENV_ALGO_OVERRIDES.get((env, algorithm), {}))
    return cfg


def build_trial_config(algorithm, params, proxy_steps, env="flappybird"):
    """Merge searched params onto the base and scale horizon-calibrated knobs to the proxy."""
    cfg = _merge_env(dict(BASE[algorithm]), env, algorithm)
    cfg.update(params)
    cfg["max_env_steps"] = proxy_steps
    if algorithm == "ppo":
        # Anneal LR over the proxy horizon so trials train in the same decaying-LR regime
        # the exported run will use — params are then selected for annealing, not a fixed LR.
        cfg["lr_anneal_steps"] = proxy_steps
    if algorithm in ("dqn", "rainbow"):
        # Keep warmup well under Hyperband's first rung (~proxy/9) so trials actually learn
        # before the first prune decision (the first study pruned everything at ~13k steps).
        cfg["start_learning_after"] = min(cfg.get("start_learning_after", 20000),
                                          max(1000, proxy_steps // 30))
        if cfg.get("use_per"):
            cfg["per_beta_frames"] = proxy_steps  # anneal beta over the proxy horizon
    return cfg


def build_export_config(algorithm, params, full_steps, env="flappybird"):
    """The winner config for a full-length run: full-horizon values, no proxy scaling."""
    cfg = _merge_env(dict(BASE[algorithm]), env, algorithm)
    cfg.update(params)
    if env == "mario":
        # export against the real training split, not the HPO proxy
        cfg["env_make_params"] = dict(cfg["env_make_params"], level_split="smb1_stage_holdout")
    if algorithm == "ppo":
        # Open-ended run: anneal LR over full_steps, then hold at lr_min and keep training.
        # No hard max_env_steps stop (LR floor replaces it).
        cfg["lr_anneal_steps"] = full_steps
    else:
        cfg["max_env_steps"] = full_steps  # needed for epsilon_frac -> decay at full scale
    return cfg


def make_agent(cfg, label, run_name):
    algo = cfg.get("algorithm", "dqn")
    if algo == "ppo":
        from ppo_agent import PPOAgent
        return PPOAgent(label, hyperparams=cfg, run_name=run_name)
    from dqn_agent import DQNAgent  # DQN + Rainbow
    return DQNAgent(label, hyperparams=cfg, run_name=run_name)


def objective(trial, a):
    params = suggest_params(trial, a.algorithm, env=a.env)
    study_name = trial.study.study_name
    seed_means = []

    for si in range(a.search_seeds):
        cfg = build_trial_config(a.algorithm, params, a.proxy_steps, env=a.env)
        cfg["seed"] = BASE_SEED + si
        suffix = f"_s{si}" if a.search_seeds > 1 else ""
        run_name = os.path.join("hpo", study_name, "trials", f"t{trial.number}{suffix}")
        agent = make_agent(cfg, f"t{trial.number}", run_name)
        try:
            last_report = [0]

            def report_cb(step, metric):
                # Intermediate reporting + pruning only in single-seed mode (unambiguous curve).
                # Throttled to REPORT_INTERVAL_STEPS so per-episode DQN/Rainbow calls don't
                # hammer the shared journal lock.
                if a.search_seeds != 1:
                    return
                if step - last_report[0] < REPORT_INTERVAL_STEPS:
                    return
                last_report[0] = step
                trial.report(metric, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            agent.train(report_cb=report_cb, record_video=False)
            metrics = agent.evaluate(num_episodes=a.eval_episodes, record_artifacts=False)
            seed_means.append(metrics[OBJECTIVE_KEYS[a.objective]])
        finally:
            del agent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return float(np.mean(seed_means))


# ── storage helper (works across optuna 3.x / 4.x) ───────────────────────────────────
def make_storage(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:  # optuna >= 4
        from optuna.storages.journal import JournalFileBackend
        backend = JournalFileBackend(path)
    except Exception:  # optuna 3.x
        from optuna.storages import JournalFileStorage
        backend = JournalFileStorage(path)
    return optuna.storages.JournalStorage(backend)


def run_worker(a):
    storage = make_storage(a.storage)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=max(1, a.prune_warmup_steps),
        max_resource=a.proxy_steps,
        reduction_factor=3,
    )
    study = optuna.create_study(
        study_name=a.study, storage=storage,
        # unseeded: workers diverge. n_startup_trials is pure random sampling, so on a
        # ~40-trial Mario study the default 10 would waste a quarter of the budget.
        sampler=optuna.samplers.TPESampler(multivariate=True,
                                          n_startup_trials=a.startup_trials),
        pruner=pruner, direction="maximize", load_if_exists=True,
    )

    def stop_when_reached(study, trial):
        if len(study.get_trials(deepcopy=False)) >= a.n_trials:
            study.stop()

    study.optimize(lambda t: objective(t, a), callbacks=[stop_when_reached], timeout=a.timeout)
    print(f"[worker] done. total trials in study: {len(study.get_trials(deepcopy=False))}")


# ── report-only: artifacts + winner export ───────────────────────────────────────────
def _write_trials_csv(study, path):
    trials = study.get_trials(deepcopy=False)
    param_keys = sorted({k for t in trials for k in t.params})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["number", "state", "value"] + param_keys)
        for t in trials:
            w.writerow([t.number, t.state.name, t.value] + [t.params.get(k, "") for k in param_keys])


def _write_plots(study, study_dir):
    from optuna.visualization.matplotlib import (
        plot_optimization_history, plot_param_importances,
        plot_parallel_coordinate, plot_slice,
    )
    for name, fn in [
        ("optimization_history", plot_optimization_history),
        ("param_importances", plot_param_importances),
        ("parallel_coordinate", plot_parallel_coordinate),
        ("slice", plot_slice),
    ]:
        try:
            ax = fn(study)
            fig = ax.figure if hasattr(ax, "figure") else ax[0].figure
            fig.tight_layout()
            fig.savefig(os.path.join(study_dir, f"{name}.png"), bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[report] skipped {name}: {e}")


def append_config_to_yaml(path, key, cfg, study_name):
    with open(path) as f:
        existing = yaml.safe_load(f) or {}
    final_key, i = key, 2
    while final_key in existing:
        final_key = f"{key}_{i}"
        i += 1
    block = yaml.dump({final_key: cfg}, default_flow_style=False, sort_keys=False)
    with open(path, "a") as f:
        f.write(f"\n# Auto-generated by tune.py from study '{study_name}' (best trial)\n")
        f.write(block)
    return final_key


def run_report(a):
    storage = make_storage(a.storage)
    study = optuna.load_study(study_name=a.study, storage=storage)
    study_dir = os.path.join(HPO_DIR, a.study)
    os.makedirs(study_dir, exist_ok=True)

    _write_trials_csv(study, os.path.join(study_dir, "trials.csv"))
    _write_plots(study, study_dir)

    completed = [t for t in study.get_trials(deepcopy=False)
                 if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("[report] no completed trials — nothing to export.")
        return

    best = study.best_trial
    export_cfg = build_export_config(a.algorithm, best.params, a.full_steps, env=a.env)
    prefix = "flappybird" if a.env == "flappybird" else a.env
    final_key = append_config_to_yaml("hyperparams.yml", f"{prefix}_{a.algorithm}_tuned",
                                      export_cfg, a.study)

    lines = [
        f"study: {a.study}",
        f"algorithm: {a.algorithm}",
        f"completed trials: {len(completed)} / {len(study.get_trials(deepcopy=False))} total",
        f"best value ({a.objective} greedy reward): {best.value:.4f}",
        f"best trial: #{best.number}",
        "best params:",
        *[f"  {k}: {v}" for k, v in best.params.items()],
        f"exported to hyperparams.yml as: {final_key}",
        f"run it with: sbatch train.sh {final_key}",
    ]
    with open(os.path.join(study_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Optuna HPO for DQN / Rainbow / PPO.")
    p.add_argument("--algorithm", required=True, choices=["dqn", "rainbow", "ppo"])
    p.add_argument("--env", default="flappybird", choices=sorted(ENVS),
                   help="which game to tune on (default flappybird, preserving the "
                        "existing study names and export keys)")
    p.add_argument("--n-trials", type=int, default=400, help="target total trials in the study")
    p.add_argument("--proxy-steps", type=int, default=None,
                   help="env steps per trial (proxy budget). Default is algorithm-aware: 300k for "
                        "dqn/rainbow (Rainbow is slow, ~1 trial/GPU — a long proxy starves the trial "
                        "count), 1M for ppo (cheaper per step, benefits from the longer horizon). "
                        "Override explicitly to change.")
    p.add_argument("--eval-episodes", type=int, default=None,
                   help="greedy episodes for the objective. Env-aware default: 30 for "
                        "flappybird, 300 for mario (a 4-level pool makes a 30-episode mean "
                        "noisier than the signal it has to rank)")
    p.add_argument("--objective", default="p25", choices=list(OBJECTIVE_KEYS),
                   help="eval statistic to maximize (p25=robust, default)")
    p.add_argument("--prune-warmup-steps", type=int, default=None,
                   help="steps before Hyperband's first pruning rung. Env-aware default: "
                        "proxy/9 for flappybird, proxy/3 for mario (proxy/9 lands where "
                        "learners and non-learners are still <1 sd apart on Mario)")
    p.add_argument("--full-steps", type=int, default=10_000_000,
                   help="max_env_steps written into the exported winner config")
    p.add_argument("--search-seeds", type=int, default=1,
                   help="seeds averaged per trial (pruning only active when 1)")
    p.add_argument("--startup-trials", type=int, default=None,
                   help="TPE random-sampling warmup (default: 5 for mario, 10 for flappybird)")
    p.add_argument("--study", default=None, help="study name (default: the algorithm)")
    p.add_argument("--storage", default=None,
                   help="journal path (default: runs/hpo/<study>/<study>.journal)")
    p.add_argument("--timeout", type=float, default=None, help="per-worker wall-time cap (s)")
    p.add_argument("--report-only", action="store_true",
                   help="load the study, write artifacts + export winner; do not optimize")
    a = p.parse_args()

    if a.proxy_steps is None:
        if a.env == "mario":
            # Mario needs far more steps before any signal shows, so the proxy-to-full transfer
            # assumption is weaker here than it was for FlappyBird. Since trials now run on all
            # 20 training levels, the per-level budget is what binds: PPO gets 150k/level at 3M,
            # Rainbow only 15k at 300k -- and start_learning_after=20000 alone eats 6.7% of that
            # proxy. 600k doubles it to 30k/level and still yields ~71 trials per 48h study at
            # 4 workers (~25 env steps/s each; 8 workers thrash the replay buffers, see
            # hyperparametertune.sh).
            a.proxy_steps = 3_000_000 if a.algorithm == "ppo" else 600_000
        else:
            a.proxy_steps = 1_000_000 if a.algorithm == "ppo" else 300_000
    if a.eval_episodes is None:
        a.eval_episodes = 300 if a.env == "mario" else 30
    if a.startup_trials is None:
        a.startup_trials = 5 if a.env == "mario" else 10
    if a.prune_warmup_steps is None:
        # Hyperband only pays off if the FIRST rung is cheap: on FlappyBird a rejected Rainbow
        # trial cost 0.15h, so 144 of 200 trials were discarded for 34% of the hours. A Mario
        # rung at proxy//3 costs 8.7h -> 0 of 8 trials pruned, Hyperband inert. Measured
        # crossovers: PPO's proxy is noise below ~300k and reliable at 400-500k (rho .47/.55/.68
        # at 300/400/500k, n=28); Rainbow already identifies the weaker half from ~50k (n=4, so
        # suggestive only, but it is far more sample-efficient: replay + n-step + PER).
        if a.env == "mario" and a.algorithm == "ppo":
            a.prune_warmup_steps = a.proxy_steps // 3      # 1M at the 3M proxy: on the 20-level
            # split two configs are only 7% apart at 500k but 31% apart at 1M, so pruning at 500k
            # would discard good trials. Costs ~96 trials per 48h study instead of ~120.
        else:
            a.prune_warmup_steps = a.proxy_steps // 9      # 33k at Rainbow's 300k; unchanged for flappybird
    if a.study is None:
        # unchanged for flappybird, so existing journals still resume by name
        a.study = a.algorithm if a.env == "flappybird" else f"{a.env}_{a.algorithm}"
    if a.storage is None:
        a.storage = os.path.join(HPO_DIR, a.study, f"{a.study}.journal")

    if a.report_only:
        run_report(a)
    else:
        run_worker(a)


if __name__ == "__main__":
    main()
