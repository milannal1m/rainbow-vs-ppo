import gymnasium as gym
import random
import torch
import yaml
import argparse
import itertools
import importlib
import os

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

        importlib.import_module(self.env_package)

        self.loss_fn = torch.nn.MSELoss()

        self.RUN_DIR = os.path.join(RUNS_DIR, self.hyperparams_set)
        os.makedirs(self.RUN_DIR, exist_ok=True)
        self.LOG_FILE   = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.log")
        self.MODEL_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.pt")
        self.GRAPH_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.png")
        self.BEST_VIDEO_DIR       = os.path.join(self.RUN_DIR, "best_videos")
        os.makedirs(self.BEST_VIDEO_DIR, exist_ok=True)
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

        memory  = ReplayMemory(capacity=self.replay_memory_size, seed=REPLAY_MEMORY_SEED)
        epsilon = self.epsilon_init

        rewards_per_episode = []
        epsilon_history     = []
        step_count          = 0
        best_reward         = float("-inf")
        best_greedy_reward  = float("-inf")

        start_time = datetime.now()
        last_graph_update_time = start_time
        log(f"{start_time.strftime(DATE_FORMAT)}: Training starting...", self.LOG_FILE, mode='w')

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

                while not (terminated or truncated) and episode_reward < self.stop_on_reward:
                    if random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        with torch.no_grad():
                            action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

                    new_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward

                    new_state     = torch.tensor(new_state, dtype=torch.float32).to(device)
                    reward_tensor = torch.tensor(reward, dtype=torch.float32).to(device)
                    action_tensor = torch.tensor(action, dtype=torch.int64, device=device)

                    memory.push((state, action_tensor, new_state, reward_tensor, terminated))
                    step_count += 1
                    state = new_state

                    if len(memory) > self.batch_size and step_count > self.start_learning_after:
                        mini_batch = memory.sample(self.batch_size)
                        optimize(mini_batch, policy_dqn, target_dqn, optimizer, self.loss_fn,
                                 self.discount_factor_g, self.enable_double_dqn, device)

                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)

                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                rewards_per_episode.append(episode_reward)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    log(f"{datetime.now().strftime(DATE_FORMAT)} Episode {episode}: New best reward {best_reward:.2f}, model saved.", self.LOG_FILE)
                    record_episode(policy_dqn, self.env_id, self.env_make_params,
                                   self.BEST_VIDEO_DIR, f"best_ep{episode}",
                                   self.stop_on_reward, seed=episode + 1, device=device,
                                   obs_size=self.obs_size, frame_stack=self.frame_stack,
                                   rgb_wrapper=self.rgb_wrapper)

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
                    save_graph(rewards_per_episode, epsilon_history, self.GRAPH_FILE)
                    last_graph_update_time = datetime.now()
        finally:
            env.close()

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
    args = parser.parse_args()

    dql = Agent(hyperparams_set=args.hyperparameters)

    if args.train:
        dql.train()
    else:
        dql.test()
