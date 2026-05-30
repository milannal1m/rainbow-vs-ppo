import gymnasium as gym
import random
import torch
import yaml
import argparse
import itertools
import importlib
import os
import numpy as np

from datetime import datetime, timedelta

from experience_replay import ReplayMemory
from dqn import NETWORK_REGISTRY, optimize
from utils import log, save_graph, record_episode, preprocess_env, save_preprocessed_sanity_check, RGBObservationWrapper
from config import DATE_FORMAT, RUNS_DIR, CHECKPOINT_EVERY, REPLAY_MEMORY_SEED, GRAPH_UPDATE_SECONDS, HEADLESS

os.makedirs(RUNS_DIR, exist_ok=True)

if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


class Agent:
    def __init__(self, hyperparams_set):
        with open("hyperparams.yml", "r") as f:
            all_hyperparam_sets = yaml.safe_load(f)
        hyperparams = all_hyperparam_sets[hyperparams_set]

        self.hyperparams_set = hyperparams_set
        self.env_id                 = hyperparams["env_id"]
        self.replay_memory_size     = hyperparams["replay_memory_size"]
        self.batch_size             = hyperparams["batch_size"]
        self.start_learning_after   = hyperparams.get("start_learning_after", 0)
        self.epsilon_init           = hyperparams["epsilon_init"]
        self.epsilon_decay          = hyperparams["epsilon_decay"]
        self.epsilon_min            = hyperparams["epsilon_min"]
        self.network_sync_rate      = hyperparams["network_sync_rate"]
        self.learning_rate_a        = hyperparams["learning_rate_a"]
        self.discount_factor_g      = hyperparams["discount_factor_g"]
        self.stop_on_reward         = hyperparams["stop_on_reward"]
        self.hidden_dim             = hyperparams["hidden_dim"]
        self.enable_double_dqn      = hyperparams.get("enable_double_dqn", False)
        self.network_type           = hyperparams.get("network_type", "dqn")
        self.env_make_params        = hyperparams.get("env_make_params", {})
        self.frame_stack            = hyperparams.get("frame_stack", None)
        self.obs_size               = hyperparams.get("obs_size", 80)
        self.env_package            = hyperparams.get("env_package", "flappy_bird_gymnasium")
        self.rgb_wrapper            = hyperparams.get("rgb_wrapper", False)
        self.lr_decay_patience      = hyperparams.get("lr_decay_patience", None)

        importlib.import_module(self.env_package)

        self.loss_fn = torch.nn.MSELoss()

        self.RUN_DIR = os.path.join(RUNS_DIR, self.hyperparams_set)
        os.makedirs(self.RUN_DIR, exist_ok=True)
        self.LOG_FILE   = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.log")
        self.MODEL_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.pt")
        self.GRAPH_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.png")
        self.CHECKPOINT_VIDEO_DIR = os.path.join(self.RUN_DIR, "checkpoint_videos")
        os.makedirs(self.CHECKPOINT_VIDEO_DIR, exist_ok=True)

    def _make_env(self, render_mode=None):
        if render_mode is None and self.frame_stack:
            render_mode = "rgb_array"
        env = gym.make(self.env_id, render_mode=render_mode, **self.env_make_params)
        if self.rgb_wrapper:
            env = RGBObservationWrapper(env)
        if self.frame_stack:
            env = preprocess_env(env, self.obs_size, self.frame_stack)
        return env

    def _build_model(self, num_states, num_actions):
        cls = NETWORK_REGISTRY[self.network_type]
        return cls(num_states, num_actions, self.hidden_dim).to(device)

    def train(self):
        env = self._make_env()

        num_actions = env.action_space.n
        num_states  = env.observation_space.shape[0]

        policy_dqn = self._build_model(num_states, num_actions)
        target_dqn = self._build_model(num_states, num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=self.lr_decay_patience, min_lr=1e-6
        ) if self.lr_decay_patience else None

        memory  = ReplayMemory(capacity=self.replay_memory_size, seed=REPLAY_MEMORY_SEED)
        epsilon = self.epsilon_init

        rewards_per_episode = []
        pipes_per_episode   = []
        lengths_per_episode = []
        loss_per_step       = []
        q_per_step          = []
        epsilon_history     = []
        step_count          = 0
        best_reward         = float("-inf")
        best_greedy_reward  = float("-inf")

        start_time = datetime.now()
        last_graph_update_time = start_time
        log(f"{start_time.strftime(DATE_FORMAT)}: Training starting...", self.LOG_FILE, mode='w')
        log(f"Device: {device}", self.LOG_FILE)

        if self.frame_stack:
            save_preprocessed_sanity_check(self.env_id, self.env_make_params,
                                           self.obs_size, self.frame_stack, self.RUN_DIR,
                                           rgb_wrapper=self.rgb_wrapper)

        try:
            for episode in itertools.count():
                state, _ = env.reset(seed=episode + 1)
                state = torch.tensor(state, dtype=torch.float32).to(device)

                terminated     = False
                truncated      = False
                episode_reward = 0.0
                episode_pipes  = 0
                episode_length = 0

                while not (terminated or truncated) and episode_reward < self.stop_on_reward:
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

                    new_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    if reward >= 1.0:
                        episode_pipes += 1

                    new_state     = torch.tensor(new_state, dtype=torch.float32).to(device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
                    action_tensor = torch.tensor(action, dtype=torch.int64, device=device)

                    memory.push((state, action_tensor, new_state, reward_tensor, terminated))
                    step_count += 1
                    state = new_state

                    if len(memory) > self.batch_size and step_count > self.start_learning_after:
                        mini_batch = memory.sample(self.batch_size)
                        loss, mean_q = optimize(mini_batch, policy_dqn, target_dqn, optimizer, self.loss_fn,
                                                self.discount_factor_g, self.enable_double_dqn, device)
                        loss_per_step.append(loss)
                        q_per_step.append(mean_q)

                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)

                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                rewards_per_episode.append(episode_reward)
                pipes_per_episode.append(episode_pipes)
                lengths_per_episode.append(episode_length)

                if lr_scheduler:
                    lr_scheduler.step(np.mean(rewards_per_episode[-100:]))

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best reward {best_reward:.2f}.", self.LOG_FILE)

                if episode % CHECKPOINT_EVERY == 0:
                    greedy_reward = record_episode(policy_dqn, self.env_id, self.env_make_params,
                                                   self.CHECKPOINT_VIDEO_DIR, f"checkpoint_ep{episode}",
                                                   self.stop_on_reward, seed=episode + 1, device=device,
                                                   obs_size=self.obs_size, frame_stack=self.frame_stack,
                                                   rgb_wrapper=self.rgb_wrapper)
                    if greedy_reward > best_greedy_reward:
                        best_greedy_reward = greedy_reward
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best greedy reward {best_greedy_reward:.2f}, model saved.", self.LOG_FILE)

                if datetime.now() - last_graph_update_time > timedelta(seconds=GRAPH_UPDATE_SECONDS):
                    save_graph(rewards_per_episode, pipes_per_episode, lengths_per_episode,
                               loss_per_step, q_per_step, epsilon_history, self.GRAPH_FILE)
                    last_graph_update_time = datetime.now()
        finally:
            env.close()

    def explain(self, num_frames=10):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import matplotlib.pyplot as plt

        env = self._make_env()
        num_actions = env.action_space.n
        num_states  = env.observation_space.shape[0]

        policy_dqn = self._build_model(num_states, num_actions)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
        policy_dqn.eval()

        if not hasattr(policy_dqn, 'conv3'):
            print("explain() only supported for CNN models.")
            env.close()
            return

        all_states = []
        state, _ = env.reset(seed=42)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        terminated = False
        truncated  = False
        episode_reward = 0.0

        while not (terminated or truncated) and episode_reward < self.stop_on_reward:
            all_states.append(state.clone())
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = torch.tensor(new_state, dtype=torch.float32).to(device)

        env.close()

        n = min(num_frames, len(all_states))
        indices = np.linspace(0, len(all_states) - 1, n, dtype=int)
        sampled = [all_states[i] for i in indices]

        explain_dir = os.path.join(self.RUN_DIR, "grad_cam")
        os.makedirs(explain_dir, exist_ok=True)

        conv_layers = [("conv1", policy_dqn.conv1), ("conv2", policy_dqn.conv2), ("conv3", policy_dqn.conv3)]

        for layer_name, layer in conv_layers:
            with GradCAM(model=policy_dqn, target_layers=[layer]) as cam:
                fig, axes = plt.subplots(n, 5, figsize=(15, n * 3))
                if n == 1:
                    axes = axes[np.newaxis, :]
                fig.suptitle(f"Grad-CAM — {layer_name}")

                for i, s in enumerate(sampled):
                    inp = s.unsqueeze(0)
                    with torch.no_grad():
                        q_vals = policy_dqn(inp).squeeze()
                    action = q_vals.argmax().item()

                    grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(action)])[0]

                    for f in range(4):
                        axes[i, f].imshow(s[f].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                        axes[i, f].set_title(f"Frame {f + 1}")
                        axes[i, f].axis('off')

                    frame_rgb = np.stack([s[3].cpu().numpy()] * 3, axis=-1)
                    cam_img   = show_cam_on_image(frame_rgb, grayscale_cam, use_rgb=True)
                    label = f"{'flap' if action == 1 else 'no-flap'}  Q={q_vals[action]:.2f}"
                    axes[i, 4].imshow(cam_img)
                    axes[i, 4].set_title(label)
                    axes[i, 4].axis('off')

                fig.tight_layout()
                fig.savefig(os.path.join(explain_dir, f"grad_cam_{layer_name}.png"), bbox_inches='tight')
                plt.close(fig)
                print(f"Grad-CAM saved to {explain_dir}/grad_cam_{layer_name}.png")

    def evaluate(self, num_episodes=100):
        self.explain()

        env = self._make_env()

        num_actions = env.action_space.n
        num_states  = env.observation_space.shape[0]

        policy_dqn = self._build_model(num_states, num_actions)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
        policy_dqn.eval()

        all_rewards, all_pipes, all_lengths, all_q = [], [], [], []

        try:
            for episode in range(num_episodes):
                state, _ = env.reset(seed=episode + 1)
                state = torch.tensor(state, dtype=torch.float32).to(device)

                terminated     = False
                truncated      = False
                episode_reward = 0.0
                episode_pipes  = 0
                episode_length = 0
                episode_q      = []

                while not (terminated or truncated) and episode_reward < self.stop_on_reward:
                    with torch.no_grad():
                        q_vals = policy_dqn(state.unsqueeze(0)).squeeze()
                        episode_q.append(q_vals.max().item())
                        action = q_vals.argmax().item()

                    new_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    if reward >= 1.0:
                        episode_pipes += 1
                    state = torch.tensor(new_state, dtype=torch.float32).to(device)

                all_rewards.append(episode_reward)
                all_pipes.append(episode_pipes)
                all_lengths.append(episode_length)
                all_q.append(np.mean(episode_q) if episode_q else 0.0)
        finally:
            env.close()

        eval_log = os.path.join(self.RUN_DIR, "evaluation.log")
        lengths_s = [l / 30 for l in all_lengths]
        lines = [
            f"Evaluation over {num_episodes} greedy episodes",
            f"Reward:               mean={np.mean(all_rewards):.2f}  std={np.std(all_rewards):.2f}",
            f"Pipes passed:         mean={np.mean(all_pipes):.2f}  std={np.std(all_pipes):.2f}",
            f"Episode length (s):   mean={np.mean(lengths_s):.2f}  std={np.std(lengths_s):.2f}",
            f"Q-value mag:          mean={np.mean(all_q):.4f}  std={np.std(all_q):.4f}",
        ]
        with open(eval_log, "w") as f:
            f.write("\n".join(lines) + "\n")
        for line in lines:
            print(line)

    def test(self, render=True):
        env = self._make_env(render_mode='human' if render else None)

        num_actions = env.action_space.n
        num_states  = env.observation_space.shape[0]

        policy_dqn = self._build_model(num_states, num_actions)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
        policy_dqn.eval()

        try:
            for episode in itertools.count():
                state, _ = env.reset(seed=episode + 1)
                state = torch.tensor(state, dtype=torch.float32).to(device)

                terminated     = False
                truncated      = False
                episode_reward = 0.0

                while not (terminated or truncated) and episode_reward < self.stop_on_reward:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

                    new_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    state = torch.tensor(new_state, dtype=torch.float32).to(device)

                print(f"Episode {episode}: reward = {episode_reward:.2f}")
        finally:
            env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    parser.add_argument('--evaluate', help='Evaluate saved model over 100 greedy episodes', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparams_set=args.hyperparameters)

    if args.train:
        dql.train()
    elif args.evaluate:
        dql.evaluate()
    else:
        dql.test()
