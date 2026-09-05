import torch
import itertools
import os
import numpy as np

from datetime import datetime, timedelta

from base_agent import BaseAgent, device
from ppo import PPO_NETWORK_REGISTRY, ppo_optimize
from rollout_buffer import RolloutBuffer
from utils import log, save_graph, record_episode, save_preprocessed_sanity_check
from checkpointing import (EpisodeCSVLogger, episode_row, load_run_state, save_run_state,
                           CHECKPOINT_STATE_SECONDS)
from config import DATE_FORMAT, CHECKPOINT_EVERY, GRAPH_UPDATE_SECONDS


class PPOAgent(BaseAgent):
    _eval_aux_label = "Value estimate"

    def __init__(self, hyperparams_set, hyperparams=None, run_name=None):
        super().__init__(hyperparams_set, hyperparams=hyperparams, run_name=run_name)

        hyperparams = self.hyperparams  # resolved by BaseAgent (file or injected dict)

        self.network_type   = hyperparams.get("network_type", "ppo")
        self.rollout_steps  = hyperparams.get("rollout_steps", 2048)
        self.ppo_epochs     = hyperparams.get("ppo_epochs", 10)
        self.minibatch_size = hyperparams.get("minibatch_size", 64)
        self.clip_eps       = hyperparams.get("clip_eps", 0.2)
        self.gae_lambda     = hyperparams.get("gae_lambda", 0.95)
        self.vf_coef        = hyperparams.get("vf_coef", 0.5)
        self.ent_coef       = hyperparams.get("ent_coef", 0.01)
        self.max_grad_norm  = hyperparams.get("max_grad_norm", 0.5)
        # Linear LR annealing (PPO standard): ramp lr_init -> lr_min over lr_anneal_steps
        # env-steps, then hold at lr_min and keep training (no max_env_steps required).
        # When lr_anneal_steps is unset, fall back to the base ReduceLROnPlateau.
        self.lr_anneal_steps = hyperparams.get("lr_anneal_steps", None)
        self.lr_min          = hyperparams.get("lr_min", 1e-5)

    def _build_model(self, num_states, num_actions):
        cls = PPO_NETWORK_REGISTRY[self.network_type]
        if self.network_type == "ppo_cnn":
            return cls(num_states, num_actions, self.hidden_dim, obs_size=self.obs_size).to(device)
        return cls(num_states, num_actions, self.hidden_dim).to(device)

    def _load_policy(self, env):
        num_actions  = env.action_space.n
        num_states   = env.observation_space.shape[0]
        actor_critic = self._build_model(num_states, num_actions)
        actor_critic.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
        actor_critic.eval()
        return actor_critic

    def _run_episode_greedy(self, env, actor_critic, seed, collect_states=False, metric=None,
                            collect_stride=1):
        state, _ = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        terminated = False
        truncated  = False
        episode_reward = 0.0
        episode_metric = metric if metric is not None else self.metric_spec.new()
        episode_length = 0
        value_estimates = []
        states = []

        # collect_states: False = none, True = every frame (unbounded), int n = at most n frames
        # taken every collect_stride steps, then stop. The int form exists because Grad-CAM needs
        # ~10 frames while a converged FlappyBird episode is ~757k steps -- collecting all of them
        # is ~77 GB and was what OOM-killed job 6709373.
        want = None if collect_states is True else (int(collect_states) if collect_states else 0)

        while not (terminated or truncated) and episode_reward < self.stop_on_reward:
            if collect_states and (want is None or episode_length % collect_stride == 0):
                states.append(state.clone())
                if want is not None and len(states) >= want:
                    break
            with torch.no_grad():
                action, _, _, value = actor_critic.get_action(state.unsqueeze(0), deterministic=True)
            value_estimates.append(value.item())
            new_state, reward, terminated, truncated, info = env.step(action.item())
            episode_reward += reward
            episode_length += 1
            episode_metric.update(reward, info)
            state = torch.tensor(new_state, dtype=torch.float32).to(device)

        return episode_reward, episode_metric.value(), episode_length, value_estimates, states

    def train(self, report_cb=None, record_video=True, resume=False):
        # report_cb(step, metric): optional hook (used by HPO) called once per PPO iteration;
        #   it may raise to abort the run early (pruning). record_video=False skips the
        #   checkpoint video writes and sanity-check png for lightweight HPO trials.
        #   resume=True continues from <base>_state.pt.
        env = self._make_env()

        num_actions  = env.action_space.n
        num_states   = env.observation_space.shape[0]
        actor_critic = self._build_model(num_states, num_actions)

        optimizer = torch.optim.Adam(actor_critic.parameters(), lr=self.learning_rate_a, eps=1e-5)
        # Prefer linear LR annealing for PPO: its reward signal is too noisy for reliable
        # plateau detection. ReduceLROnPlateau is kept only as a fallback when annealing is off.
        use_lr_anneal   = bool(self.lr_anneal_steps)
        lr_floor_logged = False
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=self.lr_decay_patience, min_lr=1e-6
        ) if (self.lr_decay_patience and not use_lr_anneal) else None

        buffer = RolloutBuffer(self.rollout_steps, self.discount_factor_g, self.gae_lambda)

        rewards_per_episode  = []
        pipes_per_episode    = []
        lengths_per_episode  = []
        policy_loss_history  = []
        value_loss_history   = []
        entropy_history      = []
        best_reward          = float("-inf")
        best_greedy_reward   = float("-inf")
        episode              = 0
        global_step          = 0
        # No striding here: PPO appends per iteration, not per step, so a 15M run is ~3.7k points.

        # ── resume ───────────────────────────────────────────────────────────────
        restored = load_run_state(self.STATE_FILE, model=actor_critic, optimizer=optimizer,
                                  map_location=device) if resume else None
        if restored:
            global_step        = restored.get("total_steps", 0)
            episode            = restored.get("episode", 0)
            best_reward        = restored.get("best_reward", float("-inf"))
            best_greedy_reward = restored.get("best_greedy_reward", float("-inf"))
            rewards_per_episode = restored.get("rewards_per_episode", [])
            pipes_per_episode   = restored.get("pipes_per_episode", [])
            lengths_per_episode = restored.get("lengths_per_episode", [])

        start_time = datetime.now()
        last_graph_update_time = start_time
        last_state_save_time   = start_time
        log_mode = 'a' if restored else 'w'
        log(f"{start_time.strftime(DATE_FORMAT)}: Training "
            f"{'RESUMING' if restored else 'starting'}...", self.LOG_FILE, mode=log_mode)
        log(f"Device: {device}", self.LOG_FILE)
        if restored:
            log(f"Resumed at step {global_step}, episode {episode}. PPO is on-policy, so there is "
                f"no replay buffer to refill — the resume is exact apart from the in-flight rollout.",
                self.LOG_FILE)
        csv_logger = EpisodeCSVLogger(self.EPISODES_CSV, resume=bool(restored))

        if self.frame_stack and record_video:
            save_preprocessed_sanity_check(self.env_id, self.env_make_params,
                                           self.obs_size, self.frame_stack, self.RUN_DIR,
                                           rgb_wrapper=self.rgb_wrapper)

        # episode / global_step are set above (0, or resumed). seed=episode+1 also picks the
        # Mario level, so resuming keeps the sequence.
        state, _     = env.reset(seed=episode + 1)
        state        = torch.tensor(state, dtype=torch.float32).to(device)
        ep_reward    = 0.0
        ep_metric    = self.metric_spec.new()
        ep_length    = 0

        try:
            for _ in itertools.count():
                if self.max_env_steps and global_step >= self.max_env_steps:
                    break

                # ── linear LR annealing → floor, then hold ───────────────────
                if use_lr_anneal:
                    frac   = max(0.0, 1.0 - global_step / self.lr_anneal_steps)
                    new_lr = self.lr_min + frac * (self.learning_rate_a - self.lr_min)
                    for pg in optimizer.param_groups:
                        pg['lr'] = new_lr
                    if not lr_floor_logged and frac == 0.0:
                        lr_floor_logged = True
                        log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: LR reached floor {self.lr_min:.2e} (step {global_step}), holding.", self.LOG_FILE)

                buffer.reset()

                # ── collect rollout ──────────────────────────────────────────
                for _ in range(self.rollout_steps):
                    with torch.no_grad():
                        action, log_prob, _, value = actor_critic.get_action(state.unsqueeze(0))

                    next_state, reward, terminated, truncated, info = env.step(action.item())
                    cut_by_reward_cap = ep_reward + reward >= self.stop_on_reward
                    done = terminated or truncated or cut_by_reward_cap

                    ep_reward += reward
                    ep_length += 1
                    ep_metric.update(reward, info)

                    next_state_t = torch.tensor(next_state, dtype=torch.float32).to(device)

                    # Only a real terminal state has zero future value. Truncation (step cap,
                    # reward cap) cuts a running episode, so bootstrap from the final state or the
                    # critic learns that being truncated is worthless.
                    bootstrap_value = 0.0
                    if done and not terminated:
                        with torch.no_grad():
                            _, _, _, bv = actor_critic.get_action(next_state_t.unsqueeze(0))
                        bootstrap_value = bv.squeeze()

                    buffer.push(state, action.squeeze(), reward, log_prob.squeeze(),
                                value.squeeze(), done, bootstrap_value=bootstrap_value)
                    global_step += 1
                    state = next_state_t

                    if done:
                        rewards_per_episode.append(ep_reward)
                        pipes_per_episode.append(ep_metric.value())
                        lengths_per_episode.append(ep_length)

                        csv_logger.log(**episode_row(
                            episode=episode, total_steps=global_step,
                            wall_s=(datetime.now() - start_time).total_seconds(),
                            algo=self.network_type, return_scaled=ep_reward,
                            length=ep_length, secondary=ep_metric.value(),
                            extras=ep_metric.extras(),
                        ))

                        if ep_reward > best_reward:
                            best_reward = ep_reward
                            torch.save(actor_critic.state_dict(), self.MODEL_FILE_TRAINING)
                            log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best reward {best_reward:.2f}, training model saved.", self.LOG_FILE)

                        if episode % CHECKPOINT_EVERY == 0:
                            greedy_reward = record_episode(
                                actor_critic, self.env_id, self.env_make_params,
                                self.CHECKPOINT_VIDEO_DIR, f"checkpoint_ep{episode}",
                                self.stop_on_reward, seed=episode + 1, device=device,
                                obs_size=self.obs_size, frame_stack=self.frame_stack,
                                rgb_wrapper=self.rgb_wrapper, record_video=record_video,
                            )
                            if greedy_reward > best_greedy_reward:
                                best_greedy_reward = greedy_reward
                                torch.save(actor_critic.state_dict(), self.MODEL_FILE)
                                log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best greedy reward {best_greedy_reward:.2f}, model saved.", self.LOG_FILE)

                        episode += 1
                        ep_reward = 0.0
                        ep_metric = self.metric_spec.new()
                        ep_length = 0
                        state, _ = env.reset(seed=episode + 1)
                        state = torch.tensor(state, dtype=torch.float32).to(device)

                # ── bootstrap last value & compute GAE ──────────────────────
                with torch.no_grad():
                    _, _, _, last_value = actor_critic.get_action(state.unsqueeze(0))
                buffer.finalize(last_value.squeeze())

                # ── PPO update epochs ────────────────────────────────────────
                actor_critic.train()
                batch_policy_losses, batch_value_losses, batch_entropies = [], [], []
                for _ in range(self.ppo_epochs):
                    for batch in buffer.get_minibatches(self.minibatch_size, device):
                        _, pol_loss, val_loss, ent = ppo_optimize(
                            actor_critic, optimizer, batch,
                            self.clip_eps, self.vf_coef, self.ent_coef, self.max_grad_norm,
                        )
                        batch_policy_losses.append(pol_loss)
                        batch_value_losses.append(val_loss)
                        batch_entropies.append(ent)
                actor_critic.eval()

                policy_loss_history.append(np.mean(batch_policy_losses))
                value_loss_history.append(np.mean(batch_value_losses))
                entropy_history.append(np.mean(batch_entropies))

                if report_cb is not None and rewards_per_episode:
                    report_cb(global_step, float(np.mean(rewards_per_episode[-100:])))

                if lr_scheduler and rewards_per_episode and global_step > self.start_learning_after:
                    lr_before = optimizer.param_groups[0]['lr']
                    lr_scheduler.step(np.mean(rewards_per_episode[-100:]))
                    lr_after = optimizer.param_groups[0]['lr']
                    if lr_after < lr_before:
                        log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: LR reduced {lr_before:.2e} → {lr_after:.2e}", self.LOG_FILE)

                if datetime.now() - last_graph_update_time > timedelta(seconds=GRAPH_UPDATE_SECONDS):
                    save_graph(
                        rewards_per_episode, pipes_per_episode, lengths_per_episode,
                        policy_loss_history, value_loss_history, entropy_history,
                        self.GRAPH_FILE,
                        loss_label="Policy Loss", aux_label="Value Loss", exploration_label="Entropy",
                        metric_xlabel="PPO iterations", metric_window="100-iter",
                        smooth_exploration=True,
                        secondary_label=self.metric_spec.label, fps=self.metric_spec.fps,
                    )
                    last_graph_update_time = datetime.now()

                # time-based, so a wall-clock kill loses at most CHECKPOINT_STATE_SECONDS
                if datetime.now() - last_state_save_time > timedelta(seconds=CHECKPOINT_STATE_SECONDS):
                    save_run_state(self.STATE_FILE, model=actor_critic, optimizer=optimizer,
                                   counters=self._counters(global_step, episode, best_reward,
                                                           best_greedy_reward,
                                                           rewards_per_episode, pipes_per_episode,
                                                           lengths_per_episode))
                    last_state_save_time = datetime.now()
        finally:
            # final save, so a clean exit is resumable too
            try:
                save_run_state(self.STATE_FILE, model=actor_critic, optimizer=optimizer,
                               counters=self._counters(global_step, episode, best_reward,
                                                       best_greedy_reward, rewards_per_episode,
                                                       pipes_per_episode, lengths_per_episode))
            except Exception as exc:  # noqa: BLE001
                print(f"[checkpoint] final save failed: {exc}")
            csv_logger.close()
            env.close()

    @staticmethod
    def _counters(global_step, episode, best_reward, best_greedy_reward,
                  rewards_per_episode, pipes_per_episode, lengths_per_episode):
        return {
            "total_steps": global_step, "episode": episode,
            "best_reward": best_reward, "best_greedy_reward": best_greedy_reward,
            "rewards_per_episode": rewards_per_episode,
            "pipes_per_episode": pipes_per_episode,
            "lengths_per_episode": lengths_per_episode,
        }
