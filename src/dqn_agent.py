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
from checkpointing import (EpisodeCSVLogger, episode_row, load_run_state, metric_stride,
                           save_run_state, CHECKPOINT_STATE_SECONDS)
from config import DATE_FORMAT, CHECKPOINT_EVERY, REPLAY_MEMORY_SEED, GRAPH_UPDATE_SECONDS


class DQNAgent(BaseAgent):
    _eval_aux_label = "Q-value mag"

    def __init__(self, hyperparams_set, hyperparams=None, run_name=None):
        super().__init__(hyperparams_set, hyperparams=hyperparams, run_name=run_name)

        hyperparams = self.hyperparams  # resolved by BaseAgent (file or injected dict)

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

        # Horizon-relative epsilon schedule: if `epsilon_frac` is given, reach epsilon_min
        # at that fraction of max_env_steps. This makes a searched schedule transfer between
        # a short proxy run and the full run. Falls back to the raw epsilon_decay otherwise.
        epsilon_frac = hyperparams.get("epsilon_frac", None)
        if epsilon_frac is not None and self.max_env_steps and self.epsilon_min > 0:
            anneal_steps = max(1.0, epsilon_frac * self.max_env_steps)
            self.epsilon_decay = (self.epsilon_min / self.epsilon_init) ** (1.0 / anneal_steps)

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

    def _run_episode_greedy(self, env, policy_dqn, seed, collect_states=False, metric=None):
        state, _ = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        terminated = False
        truncated  = False
        episode_reward = 0.0
        episode_metric = metric if metric is not None else self.metric_spec.new()
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
            new_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            episode_metric.update(reward, info)
            state = torch.tensor(new_state, dtype=torch.float32).to(device)

        return episode_reward, episode_metric.value(), episode_length, episode_q, states

    def train(self, report_cb=None, record_video=True, resume=False):
        # report_cb(step, metric): optional hook (used by HPO) called once per episode;
        #   it may raise to abort the run early (pruning). record_video=False skips the
        #   checkpoint video writes and sanity-check png for lightweight HPO trials.
        #   resume=True continues from <base>_state.pt.
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
        start_episode       = 0
        # The next episode to run. Tracked explicitly rather than as `episode + 1` at save time:
        # the loop can break at the top of an episode that never ran, which would skip it on resume.
        next_episode        = 0
        best_reward         = float("-inf")
        best_greedy_reward  = float("-inf")
        # Stride the per-step series; a 15M-step run would otherwise hold 15M-element lists.
        stride = metric_stride(self.max_env_steps)

        # ── resume ───────────────────────────────────────────────────────────────
        refill_until = 0
        restored = load_run_state(self.STATE_FILE, model=policy_dqn, optimizer=optimizer,
                                  map_location=device) if resume else None
        if restored:
            target_dqn.load_state_dict(policy_dqn.state_dict())
            total_steps        = restored.get("total_steps", 0)
            start_episode      = next_episode = restored.get("episode", 0)
            best_reward        = restored.get("best_reward", float("-inf"))
            best_greedy_reward = restored.get("best_greedy_reward", float("-inf"))
            epsilon            = restored.get("epsilon", epsilon)
            rewards_per_episode = restored.get("rewards_per_episode", [])
            pipes_per_episode   = restored.get("pipes_per_episode", [])
            lengths_per_episode = restored.get("lengths_per_episode", [])
            # buffer is not checkpointed (~17 GB), so gate learning while it refills
            refill_until = total_steps + self.resume_refill_steps

        start_time = datetime.now()
        last_graph_update_time = start_time
        last_state_save_time   = start_time
        log_mode = 'a' if restored else 'w'
        log(f"{start_time.strftime(DATE_FORMAT)}: Training "
            f"{'RESUMING' if restored else 'starting'}...", self.LOG_FILE, mode=log_mode)
        log(f"Device: {device}", self.LOG_FILE)
        if restored:
            log(f"Resumed at step {total_steps}, episode {start_episode}. Replay buffer was NOT "
                f"checkpointed: learning is gated until step {refill_until} while it refills — "
                f"expect a brief off-policy discontinuity here.", self.LOG_FILE)
        csv_logger = EpisodeCSVLogger(self.EPISODES_CSV, resume=bool(restored))

        if self.frame_stack and record_video:
            save_preprocessed_sanity_check(self.env_id, self.env_make_params,
                                           self.obs_size, self.frame_stack, self.RUN_DIR,
                                           rgb_wrapper=self.rgb_wrapper)

        try:
            for episode in itertools.count(start_episode):
                if self.max_env_steps and total_steps >= self.max_env_steps:
                    break
                # seed=episode+1 also picks the Mario level, so resuming at the right index
                # keeps the level sequence continuous.
                state, _ = env.reset(seed=episode + 1)
                state = torch.tensor(state, dtype=torch.float32).to(device)

                terminated     = False
                truncated      = False
                episode_reward = 0.0
                episode_metric = self.metric_spec.new()
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

                    new_state, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    episode_metric.update(reward, info)

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

                    if (len(memory) > self.batch_size
                            and total_steps > self.start_learning_after
                            and total_steps > refill_until):
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
                        if total_steps % stride == 0:
                            loss_per_step.append(loss)
                            q_per_step.append(mean_q)

                        if self.use_per:
                            memory.update_priorities(per_idxs, td_errors)

                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                    if not self.use_noisy:
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    if total_steps % stride == 0:
                        epsilon_history.append(
                            mean_sigma(policy_dqn) if self.use_noisy else epsilon
                        )

                    if self.max_env_steps and total_steps >= self.max_env_steps:
                        break

                if nstep_buf is not None:
                    nstep_buf.flush()

                rewards_per_episode.append(episode_reward)
                pipes_per_episode.append(episode_metric.value())
                lengths_per_episode.append(episode_length)

                csv_logger.log(**episode_row(
                    episode=episode, total_steps=total_steps,
                    wall_s=(datetime.now() - start_time).total_seconds(),
                    algo=self.network_type, return_scaled=episode_reward,
                    length=episode_length, secondary=episode_metric.value(),
                    extras=episode_metric.extras(),
                ))
                next_episode = episode + 1   # this episode is complete and logged

                if report_cb is not None:
                    report_cb(total_steps, float(np.mean(rewards_per_episode[-100:])))

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
                                                   rgb_wrapper=self.rgb_wrapper, record_video=record_video)
                    if greedy_reward > best_greedy_reward:
                        best_greedy_reward = greedy_reward
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best greedy reward {best_greedy_reward:.2f}, model saved.", self.LOG_FILE)

                if datetime.now() - last_graph_update_time > timedelta(seconds=GRAPH_UPDATE_SECONDS):
                    save_graph(rewards_per_episode, pipes_per_episode, lengths_per_episode,
                               loss_per_step, q_per_step, epsilon_history, self.GRAPH_FILE,
                               exploration_label=exploration_label,
                               secondary_label=self.metric_spec.label, fps=self.metric_spec.fps)
                    last_graph_update_time = datetime.now()

                # time-based, so a wall-clock kill loses at most CHECKPOINT_STATE_SECONDS
                if datetime.now() - last_state_save_time > timedelta(seconds=CHECKPOINT_STATE_SECONDS):
                    save_run_state(self.STATE_FILE, model=policy_dqn, optimizer=optimizer,
                                   counters={
                                       "total_steps": total_steps, "episode": next_episode,
                                       "best_reward": best_reward,
                                       "best_greedy_reward": best_greedy_reward,
                                       "epsilon": epsilon,
                                       "rewards_per_episode": rewards_per_episode,
                                       "pipes_per_episode": pipes_per_episode,
                                       "lengths_per_episode": lengths_per_episode,
                                   })
                    last_state_save_time = datetime.now()
        finally:
            # final save, so a clean exit is resumable too
            try:
                save_run_state(self.STATE_FILE, model=policy_dqn, optimizer=optimizer,
                               counters={
                                   "total_steps": total_steps, "episode": next_episode,
                                   "best_reward": best_reward,
                                   "best_greedy_reward": best_greedy_reward,
                                   "epsilon": epsilon,
                                   "rewards_per_episode": rewards_per_episode,
                                   "pipes_per_episode": pipes_per_episode,
                                   "lengths_per_episode": lengths_per_episode,
                               })
            except Exception as exc:  # noqa: BLE001
                print(f"[checkpoint] final save failed: {exc}")
            csv_logger.close()
            env.close()
