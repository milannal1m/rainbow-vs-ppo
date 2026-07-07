import gymnasium as gym
import random
import torch
import yaml
import json
import itertools
import importlib
import os
import numpy as np

from datetime import datetime

from utils import log, save_eval_chart, record_episode, preprocess_env, RGBObservationWrapper, FlappyBirdResetFix
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

    def _make_env(self, render_mode=None):
        needs_rgb = self.rgb_wrapper or self.frame_stack
        env = gym.make(self.env_id, render_mode="rgb_array" if needs_rgb else render_mode, **self.env_make_params)
        env = FlappyBirdResetFix(env)
        if self.rgb_wrapper:
            env = RGBObservationWrapper(env)
        if self.frame_stack:
            env = preprocess_env(env, self.obs_size, self.frame_stack)
        if needs_rgb and render_mode == "human":
            from gymnasium.wrappers import HumanRendering
            env = HumanRendering(env)
        return env

    def train(self):
        raise NotImplementedError

    def _build_model(self, num_states, num_actions):
        raise NotImplementedError

    def _load_policy(self, env):
        raise NotImplementedError

    def _run_episode_greedy(self, env, model, seed, collect_states=False):
        """Returns (reward, pipes, length, aux_vals, states).
        aux_vals is algorithm-specific: Q-values for DQN, value estimates for PPO."""
        raise NotImplementedError

    def explain(self, num_frames=10):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import matplotlib.pyplot as plt

        if not self.frame_stack:
            print("explain() only supported for CNN models.")
            return

        env = self._make_env()
        policy = self._load_policy(env)

        if not hasattr(policy, 'conv3'):
            print("explain() requires a model with conv1/conv2/conv3 layers.")
            env.close()
            return

        _, _, _, _, all_states = self._run_episode_greedy(env, policy, seed=42, collect_states=True)
        env.close()

        n = min(num_frames, len(all_states))
        indices = np.linspace(0, len(all_states) - 1, n, dtype=int)
        sampled = [all_states[i] for i in indices]

        explain_dir = os.path.join(self.RUN_DIR, "grad_cam")
        os.makedirs(explain_dir, exist_ok=True)

        conv_layers = [("conv1", policy.conv1), ("conv2", policy.conv2), ("conv3", policy.conv3)]

        for layer_name, layer in conv_layers:
            with GradCAM(model=policy, target_layers=[layer]) as cam:
                fig, axes = plt.subplots(n, 5, figsize=(15, n * 3))
                if n == 1:
                    axes = axes[np.newaxis, :]
                fig.suptitle(f"Grad-CAM — {layer_name}")

                for i, s in enumerate(sampled):
                    inp = s.unsqueeze(0)
                    with torch.no_grad():
                        out = policy(inp)
                        logits = out[0].squeeze() if isinstance(out, tuple) else out.squeeze()
                    action = logits.argmax().item()

                    grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(action)])[0]

                    for f in range(4):
                        axes[i, f].imshow(s[f].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                        axes[i, f].set_title(f"Frame {f + 1}")
                        axes[i, f].axis('off')

                    frame_rgb = np.stack([s[3].cpu().numpy()] * 3, axis=-1)
                    cam_img   = show_cam_on_image(frame_rgb, grayscale_cam, use_rgb=True)
                    label = f"{'flap' if action == 1 else 'no-flap'}  {logits[action]:.2f}"
                    axes[i, 4].imshow(cam_img)
                    axes[i, 4].set_title(label)
                    axes[i, 4].axis('off')

                fig.tight_layout()
                fig.savefig(os.path.join(explain_dir, f"grad_cam_{layer_name}.png"), bbox_inches='tight')
                plt.close(fig)
                print(f"Grad-CAM saved to {explain_dir}/grad_cam_{layer_name}.png")

    def evaluate(self, num_episodes=100, record_artifacts=True):
        # record_artifacts=False is the lightweight HPO path: compute the objective
        # metrics and dump metrics.json, but skip Grad-CAM, video and chart writes.
        if record_artifacts and self.frame_stack:
            self.explain()

        env = self._make_env()
        policy = self._load_policy(env)

        all_rewards, all_pipes, all_lengths, all_aux = [], [], [], []
        best_reward, best_seed = float("-inf"), 1

        try:
            for episode in range(num_episodes):
                seed = episode + 1
                reward, pipes, length, aux_vals, _ = self._run_episode_greedy(env, policy, seed=seed)
                all_rewards.append(reward)
                all_pipes.append(pipes)
                all_lengths.append(length)
                all_aux.append(np.mean(aux_vals) if aux_vals else 0.0)
                if reward > best_reward:
                    best_reward = reward
                    best_seed   = seed
        finally:
            env.close()

        lengths_s = [l / 30 for l in all_lengths]
        metrics = {
            "eval_reward_mean":   float(np.mean(all_rewards)),
            "eval_reward_std":    float(np.std(all_rewards)),
            "eval_pipes_mean":    float(np.mean(all_pipes)),
            "eval_length_s_mean": float(np.mean(lengths_s)),
            "n_episodes":         int(num_episodes),
        }
        with open(os.path.join(self.RUN_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        lines = [
            f"Evaluation over {num_episodes} greedy episodes",
            f"Reward:               mean={np.mean(all_rewards):.2f}  std={np.std(all_rewards):.2f}",
            f"Pipes passed:         mean={np.mean(all_pipes):.2f}  std={np.std(all_pipes):.2f}",
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
