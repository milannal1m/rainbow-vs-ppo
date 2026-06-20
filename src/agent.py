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

from experience_replay import ReplayMemory, PrioritizedReplayMemory, NStepBuffer
from dqn import NETWORK_REGISTRY, optimize, mean_sigma
from utils import log, save_graph, save_eval_chart, record_episode, preprocess_env, save_preprocessed_sanity_check, RGBObservationWrapper, FlappyBirdResetFix
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
        self.seed                   = hyperparams.get("seed", None)

        # Rainbow extension flags
        self.use_noisy          = hyperparams.get("use_noisy", False)
        self.use_per            = hyperparams.get("use_per", False)
        self.use_nstep          = hyperparams.get("use_nstep", False)
        self.n_step             = hyperparams.get("n_step", 3)
        self.use_distributional = hyperparams.get("use_distributional", False)
        self.n_atoms            = hyperparams.get("n_atoms", 51)
        self.v_min              = hyperparams.get("v_min", -10.0)
        self.v_max              = hyperparams.get("v_max", 10.0)
        self.per_alpha          = hyperparams.get("per_alpha", 0.5)
        self.per_beta_init      = hyperparams.get("per_beta_init", 0.4)
        self.per_beta_frames    = hyperparams.get("per_beta_frames", 2000000)
        self.sigma_init         = hyperparams.get("sigma_init", 0.5)
        self.use_dueling        = hyperparams.get("use_dueling", False)

        # Noisy Nets replace epsilon-greedy exploration
        if self.use_noisy:
            self.epsilon_init = 0.0
            self.epsilon_min  = 0.0

        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        importlib.import_module(self.env_package)

        self.RUN_DIR = os.path.join(RUNS_DIR, self.hyperparams_set)
        os.makedirs(self.RUN_DIR, exist_ok=True)
        self.LOG_FILE   = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.log")
        self.MODEL_FILE         = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.pt")
        self.MODEL_FILE_TRAINING = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}_best_training.pt")
        self.GRAPH_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.png")
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

    def _build_model(self, num_states, num_actions):
        cls = NETWORK_REGISTRY[self.network_type]
        if self.network_type == "rainbow_cnn_dqn":
            return cls(
                num_states, num_actions, self.hidden_dim, obs_size=self.obs_size,
                use_noisy=self.use_noisy, sigma_init=self.sigma_init,
                use_dueling=self.use_dueling,
                use_distributional=self.use_distributional,
                n_atoms=self.n_atoms, v_min=self.v_min, v_max=self.v_max,
            ).to(device)
        if self.network_type == "cnn_dqn":
            return cls(num_states, num_actions, self.hidden_dim, obs_size=self.obs_size).to(device)
        return cls(num_states, num_actions, self.hidden_dim).to(device)

    def _load_policy(self, env):
        num_actions = env.action_space.n
        num_states  = env.observation_space.shape[0]
        policy_dqn  = self._build_model(num_states, num_actions)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
        policy_dqn.eval()
        return policy_dqn

    def _run_episode_greedy(self, env, policy_dqn, seed, collect_states=False):
        state, _ = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        terminated = False
        truncated  = False
        episode_reward = 0.0
        episode_pipes  = 0
        episode_length = 0
        episode_q      = []
        states         = []

        while not (terminated or truncated) and episode_reward < self.stop_on_reward:
            if collect_states:
                states.append(state.clone())
            with torch.no_grad():
                q_vals = policy_dqn(state.unsqueeze(0)).squeeze()
            action = q_vals.argmax().item()
            episode_q.append(q_vals.max().item())
            new_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_length += 1
            if reward >= 1.0:
                episode_pipes += 1
            state = torch.tensor(new_state, dtype=torch.float32).to(device)

        return episode_reward, episode_pipes, episode_length, episode_q, states

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

        if self.use_per:
            memory = PrioritizedReplayMemory(self.replay_memory_size, alpha=self.per_alpha)
        else:
            memory = ReplayMemory(capacity=self.replay_memory_size, seed=REPLAY_MEMORY_SEED)

        nstep_buf       = NStepBuffer(self.n_step, self.discount_factor_g, memory) if self.use_nstep else None
        effective_gamma = self.discount_factor_g ** self.n_step if self.use_nstep else self.discount_factor_g

        epsilon = self.epsilon_init
        exploration_label = "Mean σ (Noisy Nets)" if self.use_noisy else "Epsilon"

        rewards_per_episode = []
        pipes_per_episode   = []
        lengths_per_episode = []
        loss_per_step       = []
        q_per_step          = []
        epsilon_history     = []
        step_count          = 0
        total_steps         = 0
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
                    if self.use_noisy:
                        policy_dqn.sample_noise()
                        target_dqn.sample_noise()

                    if not self.use_noisy and random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

                    new_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    if reward >= 1.0:
                        episode_pipes += 1

                    new_state_t   = torch.tensor(new_state, dtype=torch.float32).to(device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32)
                    action_tensor = torch.tensor(action, dtype=torch.int64)

                    # Store on CPU as uint8 for image obs (4x smaller than float32)
                    if self.frame_stack:
                        s_store  = (state       * 255).round().byte().cpu()
                        ns_store = (new_state_t * 255).round().byte().cpu()
                    else:
                        s_store  = state.cpu()
                        ns_store = new_state_t.cpu()
                    transition = (s_store, action_tensor, ns_store, reward_tensor, terminated)
                    if nstep_buf is not None:
                        nstep_buf.push(transition)
                    else:
                        memory.push(transition)

                    step_count  += 1
                    total_steps += 1
                    state = new_state_t

                    if len(memory) > self.batch_size and total_steps > self.start_learning_after:
                        if self.use_per:
                            beta = min(1.0, self.per_beta_init +
                                       total_steps * (1.0 - self.per_beta_init) / self.per_beta_frames)
                            mini_batch, per_idxs, per_weights = memory.sample(self.batch_size, beta=beta)
                            weights_t = torch.tensor(per_weights, dtype=torch.float32).to(device)
                        else:
                            mini_batch = memory.sample(self.batch_size)
                            per_idxs, weights_t = None, None

                        loss, mean_q, td_errors = optimize(
                            mini_batch, policy_dqn, target_dqn, optimizer,
                            effective_gamma, self.enable_double_dqn, device, weights=weights_t
                        )
                        loss_per_step.append(loss)
                        q_per_step.append(mean_q)

                        if self.use_per:
                            memory.update_priorities(per_idxs, td_errors)

                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                    if not self.use_noisy:
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(
                        mean_sigma(policy_dqn) if self.use_noisy else epsilon
                    )

                if nstep_buf is not None:
                    nstep_buf.flush()

                rewards_per_episode.append(episode_reward)
                pipes_per_episode.append(episode_pipes)
                lengths_per_episode.append(episode_length)

                if lr_scheduler:
                    lr_scheduler.step(np.mean(rewards_per_episode[-100:]))

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE_TRAINING)
                    log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best reward {best_reward:.2f}, training model saved.", self.LOG_FILE)

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
                               loss_per_step, q_per_step, epsilon_history, self.GRAPH_FILE,
                               exploration_label=exploration_label)
                    last_graph_update_time = datetime.now()
        finally:
            env.close()

    def explain(self, num_frames=10):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import matplotlib.pyplot as plt

        env = self._make_env()
        policy_dqn = self._load_policy(env)

        if not hasattr(policy_dqn, 'conv3'):
            print("explain() only supported for CNN models.")
            env.close()
            return

        _, _, _, _, all_states = self._run_episode_greedy(env, policy_dqn, seed=42, collect_states=True)
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
        if self.network_type in ("cnn_dqn", "rainbow_cnn_dqn"):
            self.explain()

        eval_dir = os.path.join(self.RUN_DIR, "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        env = self._make_env()
        policy_dqn = self._load_policy(env)

        all_rewards, all_pipes, all_lengths, all_q = [], [], [], []
        best_reward, best_seed = float("-inf"), 1

        try:
            for episode in range(num_episodes):
                seed = episode + 1
                reward, pipes, length, q_vals, _ = self._run_episode_greedy(env, policy_dqn, seed=seed)
                all_rewards.append(reward)
                all_pipes.append(pipes)
                all_lengths.append(length)
                all_q.append(np.mean(q_vals) if q_vals else 0.0)
                if reward > best_reward:
                    best_reward = reward
                    best_seed   = seed
        finally:
            env.close()

        lengths_s = [l / 30 for l in all_lengths]
        lines = [
            f"Evaluation over {num_episodes} greedy episodes",
            f"Reward:               mean={np.mean(all_rewards):.2f}  std={np.std(all_rewards):.2f}",
            f"Pipes passed:         mean={np.mean(all_pipes):.2f}  std={np.std(all_pipes):.2f}",
            f"Episode length (s):   mean={np.mean(lengths_s):.2f}  std={np.std(lengths_s):.2f}",
            f"Q-value mag:          mean={np.mean(all_q):.4f}  std={np.std(all_q):.4f}",
        ]
        with open(os.path.join(eval_dir, "evaluation.log"), "w") as f:
            f.write("\n".join(lines) + "\n")
        for line in lines:
            print(line)

        save_eval_chart(all_rewards, os.path.join(eval_dir, "evaluation.png"))

        record_episode(policy_dqn, self.env_id, self.env_make_params, eval_dir, "evaluation",
                       self.stop_on_reward, best_seed, device,
                       obs_size=self.obs_size, frame_stack=self.frame_stack, rgb_wrapper=self.rgb_wrapper)

    def test(self, render=True):
        env = self._make_env(render_mode='human' if render else None)
        policy_dqn = self._load_policy(env)

        try:
            for episode in itertools.count():
                reward, _, _, _, _ = self._run_episode_greedy(env, policy_dqn, seed=episode + 1)
                print(f"Episode {episode}: reward = {reward:.2f}")
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
