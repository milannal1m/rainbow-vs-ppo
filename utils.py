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
    def __init__(self, env):
        super().__init__(env)
        self.env.reset()
        frame = self.env.render()
        h, w, c = frame.shape
        self.observation_space = Box(0, 255, shape=(h, w, c), dtype=np.uint8)

    def observation(self, _):
        return self.env.render()


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


def save_graph(rewards_per_episode, epsilon_history, graph_file):
    fig = plt.figure(1)

    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])

    plt.subplot(121)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.plot(mean_rewards)

    plt.subplot(122)
    plt.xlabel('Steps')
    plt.ylabel('Epsilon Decay')
    plt.plot(epsilon_history)

    plt.subplots_adjust(wspace=1.0, hspace=1.0)
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
