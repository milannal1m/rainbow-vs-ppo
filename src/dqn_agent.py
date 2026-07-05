import random
import torch
import itertools
import os
import numpy as np

from datetime import datetime, timedelta

from base_agent import BaseAgent, device
from experience_replay import ReplayMemory, PrioritizedReplayMemory, NStepBuffer
from dqn import NETWORK_REGISTRY, optimize, mean_sigma
from utils import log, save_graph, record_episode, save_preprocessed_sanity_check
from config import DATE_FORMAT, CHECKPOINT_EVERY, REPLAY_MEMORY_SEED, GRAPH_UPDATE_SECONDS


class DQNAgent(BaseAgent):
    _eval_aux_label = "Q-value mag"

    def __init__(self, hyperparams_set):
        super().__init__(hyperparams_set)

        import yaml
        with open("hyperparams.yml", "r") as f:
            hyperparams = yaml.safe_load(f)[hyperparams_set]

        self.replay_memory_size   = hyperparams["replay_memory_size"]
        self.batch_size           = hyperparams["batch_size"]
        self.epsilon_init         = hyperparams["epsilon_init"]
        self.epsilon_decay        = hyperparams["epsilon_decay"]
        self.epsilon_min          = hyperparams["epsilon_min"]
        self.network_sync_rate    = hyperparams["network_sync_rate"]
        self.enable_double_dqn    = hyperparams.get("enable_double_dqn", False)
        self.network_type         = hyperparams.get("network_type", "dqn")

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

        if self.use_noisy:
            self.epsilon_init = 0.0
            self.epsilon_min  = 0.0

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

                if lr_scheduler and total_steps > self.start_learning_after:
                    lr_before = optimizer.param_groups[0]['lr']
                    lr_scheduler.step(np.mean(rewards_per_episode[-100:]))
                    lr_after = optimizer.param_groups[0]['lr']
                    if lr_after < lr_before:
                        log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: LR reduced {lr_before:.2e} → {lr_after:.2e}", self.LOG_FILE)

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
