import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from gymnasium.wrappers import (
    GrayscaleObservation,
    ResizeObservation,
    FrameStackObservation,
    TransformObservation,
    RecordVideo,
)
from gymnasium.spaces import Box


class RGBObservationWrapper(gym.ObservationWrapper):
    _BG_COLOR = np.array([200, 200, 200], dtype=np.uint8)  # flappy-bird-gymnasium FILL_BACKGROUND_COLOR

    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        frame = self.env.render()
        h, w, c = frame.shape
        self.observation_space = Box(0, 255, shape=(h, w, c), dtype=np.uint8)

    def observation(self, _):
        frame = self.env.render()
        bg_mask = np.all(frame == self._BG_COLOR, axis=-1, keepdims=True)
        return np.where(bg_mask, 0, frame)


def preprocess_env(env, obs_size, frame_stack):
    env = ResizeObservation(env, shape=(obs_size, obs_size))
    env = GrayscaleObservation(env, keep_dim=False)
    env = FrameStackObservation(env, frame_stack)
    env = TransformObservation(
        env,
        lambda obs: obs.astype(np.float32) / 255.0,
        observation_space=Box(0.0, 1.0, shape=(frame_stack, obs_size, obs_size), dtype=np.float32),
    )
    return env


def save_preprocessed_sanity_check(env_id, env_make_params, obs_size, frame_stack, run_dir,
                                    rgb_wrapper=False, steps=10):
    sample_dir = os.path.join(run_dir, "sanity_check")
    os.makedirs(sample_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array", **env_make_params)
    if rgb_wrapper:
        env = RGBObservationWrapper(env)
    env = preprocess_env(env, obs_size, frame_stack)
    state, _ = env.reset(seed=0)

    for _ in range(steps):
        state, _, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    env.close()

    fig, axes = plt.subplots(1, frame_stack, figsize=(frame_stack * 3, 3))
    fig.suptitle("Preprocessed agent view (sanity check)")
    for i, ax in enumerate(axes):
        ax.imshow(state[i], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Frame {i + 1}")
        ax.axis("off")

    fig.savefig(os.path.join(sample_dir, "preprocessed_frames.png"), bbox_inches="tight")
    plt.close(fig)


def log(message, log_file, mode='a'):
    print(message)
    with open(log_file, mode) as f:
        f.write(message + '\n')


def _rolling_avg(data, window=100):
    result = np.zeros(len(data))
    for i in range(len(data)):
        result[i] = np.mean(data[max(0, i - window + 1):i + 1])
    return result


def save_graph(rewards_per_episode, pipes_per_episode, lengths_per_episode,
               loss_per_step, q_per_step, epsilon_history, graph_file):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Training metrics")

    ep_x = range(len(rewards_per_episode))
    step_x = range(len(loss_per_step))

    axes[0, 0].set_xlabel("Episodes")
    axes[0, 0].set_ylabel("Mean reward (100-ep)")
    axes[0, 0].plot(_rolling_avg(rewards_per_episode), color="#4C72B0")

    axes[0, 1].set_xlabel("Episodes")
    axes[0, 1].set_ylabel("Pipes passed (100-ep)")
    axes[0, 1].plot(_rolling_avg(pipes_per_episode), color="#55A868")

    axes[0, 2].set_xlabel("Episodes")
    axes[0, 2].set_ylabel("Episode length in s (100-ep)")
    axes[0, 2].plot(_rolling_avg([s / 30 for s in lengths_per_episode]), color="#C44E52")

    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("TD loss (100-step)")
    if loss_per_step:
        axes[1, 0].plot(_rolling_avg(loss_per_step), color="#DD8452")

    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Q-value magnitude (100-step)")
    if q_per_step:
        axes[1, 1].plot(_rolling_avg(q_per_step), color="#8172B2")

    axes[1, 2].set_xlabel("Steps")
    axes[1, 2].set_ylabel("Epsilon")
    axes[1, 2].plot(epsilon_history, color="#937860")

    fig.tight_layout()
    fig.savefig(graph_file)
    plt.close(fig)


def _rename_latest_video(video_dir, new_filename):
    try:
        candidates = [
            os.path.join(video_dir, name)
            for name in os.listdir(video_dir)
            if name.endswith(".mp4")
        ]
        if not candidates:
            return
        latest = max(candidates, key=os.path.getmtime)
        os.replace(latest, os.path.join(video_dir, new_filename))
    except OSError:
        pass


def record_episode(policy_dqn, env_id, env_make_params, video_dir, name_prefix,
                   stop_on_reward, seed, device, obs_size=None, frame_stack=None, rgb_wrapper=False):
    env = gym.make(env_id, render_mode="rgb_array", **env_make_params)
    if rgb_wrapper:
        env = RGBObservationWrapper(env)
    if frame_stack:
        env = preprocess_env(env, obs_size, frame_stack)
    env = RecordVideo(
        env,
        video_dir,
        name_prefix=f"{name_prefix}_tmp",
        episode_trigger=lambda _: True,
        disable_logger=True,
    )

    state, _ = env.reset(seed=seed)
    state = torch.tensor(state, dtype=torch.float32).to(device)
    terminated     = False
    truncated      = False
    episode_reward = 0.0

    while not (terminated or truncated) and episode_reward < stop_on_reward:
        with torch.no_grad():
            action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
        new_state, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        state = torch.tensor(new_state, dtype=torch.float32).to(device)

    env.close()
    _rename_latest_video(video_dir, f"{name_prefix}_r{episode_reward:.2f}.mp4")
    return episode_reward
