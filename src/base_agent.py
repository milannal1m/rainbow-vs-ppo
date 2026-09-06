import gymnasium as gym
import random
import torch
import yaml
import csv
import json
import itertools
import importlib
import os
import numpy as np

from datetime import datetime

from utils import log, save_eval_chart, record_episode
from env_factory import make_env
from env_metrics import make_metric_spec, action_labels_for
from config import RUNS_DIR

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseAgent:
    _eval_aux_label = "Aux metric"

    def __init__(self, hyperparams_set, hyperparams=None, run_name=None):
        # `hyperparams` (a pre-parsed dict) lets the HPO layer inject a trial config
        # without touching hyperparams.yml; when None we load it by name as before.
        if hyperparams is None:
            with open("hyperparams.yml", "r") as f:
                hyperparams = yaml.safe_load(f)[hyperparams_set]
        self.hyperparams = hyperparams

        self.hyperparams_set   = hyperparams_set
        self.env_id            = hyperparams["env_id"]
        self.env_make_params   = hyperparams.get("env_make_params", {})
        self.env_package       = hyperparams.get("env_package", "flappy_bird_gymnasium")
        self.frame_stack       = hyperparams.get("frame_stack", None)
        self.obs_size          = hyperparams.get("obs_size", 80)
        self.rgb_wrapper       = hyperparams.get("rgb_wrapper", False)
        self.hidden_dim        = hyperparams["hidden_dim"]
        self.learning_rate_a   = hyperparams["learning_rate_a"]
        self.discount_factor_g = hyperparams["discount_factor_g"]
        self.stop_on_reward    = hyperparams["stop_on_reward"]
        self.lr_decay_patience = hyperparams.get("lr_decay_patience", None)
        self.start_learning_after = hyperparams.get("start_learning_after", 0)
        self.max_env_steps     = hyperparams.get("max_env_steps", None)  # None = run until killed
        self.seed              = hyperparams.get("seed", None)

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        importlib.import_module(self.env_package)

        # Secondary metric (pipes vs pages), axis label and frame rate. Derived from env_id, so
        # a config cannot pick the wrong one.
        self.metric_spec = make_metric_spec(hyperparams)

        # run_name controls the output subdirectory (defaults to the set name for normal
        # runs); HPO passes e.g. "hpo/<study>/trials/t7" to isolate trials under runs/hpo/.
        self.run_name            = run_name if run_name is not None else hyperparams_set
        base                     = os.path.basename(self.run_name)
        self.RUN_DIR             = os.path.join(RUNS_DIR, self.run_name)
        os.makedirs(self.RUN_DIR, exist_ok=True)
        self.LOG_FILE            = os.path.join(self.RUN_DIR, f"{base}.log")
        self.MODEL_FILE          = os.path.join(self.RUN_DIR, f"{base}.pt")
        self.MODEL_FILE_TRAINING = os.path.join(self.RUN_DIR, f"{base}_best_training.pt")
        self.GRAPH_FILE          = os.path.join(self.RUN_DIR, f"{base}.png")
        self.CHECKPOINT_VIDEO_DIR = os.path.join(self.RUN_DIR, "checkpoint_videos")
        os.makedirs(self.CHECKPOINT_VIDEO_DIR, exist_ok=True)
        # Resume state and the per-episode metric log — Mario runs exceed the 24 h wall clock.
        self.STATE_FILE   = os.path.join(self.RUN_DIR, f"{base}_state.pt")
        self.EPISODES_CSV = os.path.join(self.RUN_DIR, "episodes.csv")
        # The replay buffer is too large to checkpoint, so after a resume learning is gated for
        # this many steps while it refills. Unused by PPO.
        self.resume_refill_steps = hyperparams.get("resume_refill_steps",
                                                   self.start_learning_after)

    def _make_env(self, render_mode=None, levels=None):
        # env_factory owns the wrapper chain; record_episode and save_preprocessed_sanity_check
        # go through the same call.
        return make_env(self.env_id, self.env_make_params, render_mode=render_mode,
                        obs_size=self.obs_size, frame_stack=self.frame_stack,
                        rgb_wrapper=self.rgb_wrapper, levels=levels)

    def train(self):
        raise NotImplementedError

    def _build_model(self, num_states, num_actions):
        raise NotImplementedError

    def _load_policy(self, env):
        raise NotImplementedError

    def _run_episode_greedy(self, env, model, seed, collect_states=False, metric=None):
        """Returns (reward, secondary, length, aux_vals, states).

        `secondary` is metric.value() — pipes for FlappyBird, pages for Mario. aux_vals is
        algorithm-specific: Q-values for DQN, value estimates for PPO. Pass `metric` to read
        `extras()` afterwards; the tuple stays a 5-tuple so existing callers are unaffected.
        """
        raise NotImplementedError

    def _action_label(self, action):
        labels = getattr(self, "_action_labels", None)
        if labels is None:
            return str(action)
        return labels[action] if 0 <= action < len(labels) else str(action)

    def explain(self, num_frames=10, frame_stride=50, levels=None, out_dir=None,
                seed=42, subtitle=None):
        """Grad-CAM figures, one per conv layer, from a single greedy episode.

        levels pins the Mario level (None = the configured set, which is what FlappyBird uses);
        out_dir/subtitle let a caller write several sets side by side without overwriting.
        (Not named `label`: the per-frame action label below would shadow it.)
        """
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import matplotlib.pyplot as plt

        if not self.frame_stack:
            print("explain() only supported for CNN models.")
            return

        env = self._make_env(levels=levels)
        policy = self._load_policy(env)
        # 'flap'/'no-flap' would be silently wrong on all 12 Mario actions.
        self._action_labels = action_labels_for(self.hyperparams, env.action_space.n)

        if not hasattr(policy, 'conv3'):
            print("explain() requires a model with conv1/conv2/conv3 layers.")
            env.close()
            return

        # Collect exactly num_frames states, one every frame_stride steps, then stop the episode.
        # Collecting the whole episode is what OOM-killed job 6709373: a converged FlappyBird
        # policy runs ~757k steps, i.e. ~77 GB of stacked frames for a 10-frame figure.
        _, _, _, _, sampled = self._run_episode_greedy(
            env, policy, seed=seed, collect_states=num_frames, collect_stride=frame_stride)
        env.close()

        if not sampled:
            print("explain(): no frames collected.")
            return
        n = len(sampled)

        explain_dir = out_dir or os.path.join(self.RUN_DIR, "grad_cam")
        os.makedirs(explain_dir, exist_ok=True)

        conv_layers = [("conv1", policy.conv1), ("conv2", policy.conv2), ("conv3", policy.conv3)]

        for layer_name, layer in conv_layers:
            with GradCAM(model=policy, target_layers=[layer]) as cam:
                fig, axes = plt.subplots(n, 5, figsize=(15, n * 3))
                if n == 1:
                    axes = axes[np.newaxis, :]
                fig.suptitle(f"Grad-CAM — {layer_name}"
                             + (f"  ({subtitle})" if subtitle else ""))

                for i, s in enumerate(sampled):
                    inp = s.unsqueeze(0)
                    with torch.no_grad():
                        out = policy(inp)
                        logits = out[0].squeeze() if isinstance(out, tuple) else out.squeeze()
                    action = logits.argmax().item()

                    grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(action)])[0]

                    for f in range(min(4, s.shape[0])):
                        axes[i, f].imshow(s[f].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                        axes[i, f].set_title(f"Frame {f + 1}")
                        axes[i, f].axis('off')

                    frame_rgb = np.stack([s[-1].cpu().numpy()] * 3, axis=-1)
                    cam_img   = show_cam_on_image(frame_rgb, grayscale_cam, use_rgb=True)
                    label = f"{self._action_label(action)}  {logits[action]:.2f}"
                    axes[i, 4].imshow(cam_img)
                    axes[i, 4].set_title(label)
                    axes[i, 4].axis('off')

                fig.tight_layout()
                fig.savefig(os.path.join(explain_dir, f"grad_cam_{layer_name}.png"), bbox_inches='tight')
                plt.close(fig)
                print(f"Grad-CAM saved to {explain_dir}/grad_cam_{layer_name}.png")

    EVAL_ROWS = "episodes_eval.csv"

    def _eval_rows_path(self):
        return os.path.join(self.RUN_DIR, "evaluation", self.EVAL_ROWS)

    def _load_eval_rows(self):
        """Per-episode evaluation rows from a previous call, or []."""
        path = self._eval_rows_path()
        if not os.path.exists(path):
            return []
        with open(path, newline="") as f:
            rows = []
            for r in csv.DictReader(f):
                rows.append({
                    "seed": int(r["seed"]), "reward": float(r["reward"]),
                    "secondary": float(r["secondary"]), "length": int(r["length"]),
                    "aux": float(r["aux"]), "extras": json.loads(r["extras"] or "{}"),
                })
        return rows

    def _save_eval_rows(self, rows):
        path = self._eval_rows_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["seed", "reward", "secondary", "length", "aux",
                                              "extras"])
            w.writeheader()
            for r in rows:
                w.writerow({**{k: r[k] for k in ("seed", "reward", "secondary", "length", "aux")},
                            "extras": json.dumps(r["extras"], default=str)})

    def evaluate(self, num_episodes=25, record_artifacts=True, resume=False):
        # record_artifacts=False is the lightweight HPO path: compute the objective
        # metrics and dump metrics.json, but skip Grad-CAM, video and chart writes.
        # resume=True appends num_episodes MORE episodes to the ones already in
        # evaluation/episodes_eval.csv and re-aggregates over the union, so a run can be
        # topped up later instead of restarted.
        if record_artifacts and self.frame_stack:
            self.explain()

        prior = self._load_eval_rows() if (resume and record_artifacts) else []
        if prior:
            print(f"resuming evaluation: {len(prior)} episodes already recorded, "
                  f"adding {num_episodes}")

        env = self._make_env()
        policy = self._load_policy(env)
        rows = list(prior)

        try:
            for i in range(num_episodes):
                episode = len(prior) + i
                seed = episode + 1
                metric = self.metric_spec.new()
                reward, secondary, length, aux_vals, _ = self._run_episode_greedy(
                    env, policy, seed=seed, metric=metric)
                rows.append({
                    "seed": seed, "reward": float(reward), "secondary": float(secondary),
                    "length": int(length),
                    "aux": float(np.mean(aux_vals)) if aux_vals else 0.0,
                    "extras": metric.extras() or {},
                })
                # written every episode, so an OOM or wall-clock kill keeps what was measured
                if record_artifacts:
                    self._save_eval_rows(rows)
        finally:
            env.close()

        all_rewards = [r["reward"] for r in rows]
        all_pipes   = [r["secondary"] for r in rows]
        all_lengths = [r["length"] for r in rows]
        all_aux     = [r["aux"] for r in rows]
        all_extras  = [r["extras"] for r in rows]
        best_seed   = max(rows, key=lambda r: r["reward"])["seed"]
        n_total     = len(rows)

        lengths_s = [l / self.metric_spec.fps for l in all_lengths]
        metrics = {
            "eval_reward_mean":   float(np.mean(all_rewards)),
            "eval_reward_median": float(np.median(all_rewards)),
            "eval_reward_p25":    float(np.percentile(all_rewards, 25)),  # robust objective
            "eval_reward_std":    float(np.std(all_rewards)),
            # eval_pipes_mean for FlappyBird, eval_pages_mean for Mario
            f"eval_{self.metric_spec.key}_mean": float(np.mean(all_pipes)),
            "eval_length_s_mean": float(np.mean(lengths_s)),
            "n_episodes":         int(n_total),
        }
        # per-env extras (flag rate, death-cause mix); empty for FlappyBird
        metrics.update(self.metric_spec.aggregate(all_extras))
        with open(os.path.join(self.RUN_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        lines = [
            f"Evaluation over {n_total} greedy episodes",
            f"Reward:               mean={np.mean(all_rewards):.2f}  std={np.std(all_rewards):.2f}",
            f"{self.metric_spec.label + ':':22s}mean={np.mean(all_pipes):.2f}  std={np.std(all_pipes):.2f}",
            f"Episode length (s):   mean={np.mean(lengths_s):.2f}  std={np.std(lengths_s):.2f}",
            f"{self._eval_aux_label}:  mean={np.mean(all_aux):.4f}  std={np.std(all_aux):.4f}",
        ]

        if record_artifacts:
            eval_dir = os.path.join(self.RUN_DIR, "evaluation")
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "evaluation.log"), "w") as f:
                f.write("\n".join(lines) + "\n")
            for line in lines:
                print(line)
            save_eval_chart(all_rewards, os.path.join(eval_dir, "evaluation.png"))
            record_episode(policy, self.env_id, self.env_make_params, eval_dir, "evaluation",
                           self.stop_on_reward, best_seed, device,
                           obs_size=self.obs_size, frame_stack=self.frame_stack, rgb_wrapper=self.rgb_wrapper)

        return metrics

    def test(self, render=True):
        env = self._make_env(render_mode='human' if render else None)
        policy = self._load_policy(env)

        try:
            for episode in itertools.count():
                reward, _, _, _, _ = self._run_episode_greedy(env, policy, seed=episode + 1)
                print(f"Episode {episode}: reward = {reward:.2f}")
        finally:
            env.close()
