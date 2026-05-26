import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from experience_replay import ReplayMemory
from dqn import DQN, DuelingDQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import gymnasium
import os

DATE_FORMAT = "%Y-%m-%d_%H-%M-%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use("Agg")  # generate plots as images and save

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


class Agent:
    def __init__(self, hyperparams_set):
        with open("hyperparams.yml", "r") as f:
            all_hyperparam_sets = yaml.safe_load(f)
        hyperparams = all_hyperparam_sets[hyperparams_set]

        self.hyperparams_set = hyperparams_set
        # Hyperparameters
        self.env_id             = hyperparams["env_id"]
        self.replay_memory_size = hyperparams["replay_memory_size"]
        self.batch_size         = hyperparams["batch_size"]
        self.epsilon_init       = hyperparams["epsilon_init"]
        self.epsilon_decay      = hyperparams["epsilon_decay"]
        self.epsilon_min        = hyperparams["epsilon_min"]
        self.network_sync_rate  = hyperparams["network_sync_rate"]
        self.learning_rate_a    = hyperparams["learning_rate_a"]
        self.discount_factor_g  = hyperparams["discount_factor_g"]
        self.stop_on_reward     = hyperparams["stop_on_reward"]
        self.fc1_nodes          = hyperparams["fc1_nodes"]
        self.enable_double_dqn  = hyperparams.get("enable_double_dqn", False)
        self.enable_dueling_dqn = hyperparams.get("enable_dueling_dqn", False)

        # Get optional environment-specific parameters, default to empty dict
        self.env_make_params    = hyperparams.get('env_make_params',{}) 

        self.loss_fn = torch.nn.MSELoss()   # Mean Squared Error Loss
        self.optimizer = None               # Will be initialized after creating the DQN model

        # Paths to save run info and models
        self.RUN_DIR = os.path.join(RUNS_DIR, self.hyperparams_set)
        os.makedirs(self.RUN_DIR, exist_ok=True)
        self.LOG_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.log")
        self.MODEL_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.pt")
        self.GRAPH_FILE = os.path.join(self.RUN_DIR, f"{self.hyperparams_set}.png")
        self.BEST_VIDEO_DIR = os.path.join(self.RUN_DIR, "best_videos")
        os.makedirs(self.BEST_VIDEO_DIR, exist_ok=True)
        self.CHECKPOINT_VIDEO_DIR = os.path.join(self.RUN_DIR, "checkpoint_videos")
        os.makedirs(self.CHECKPOINT_VIDEO_DIR, exist_ok=True)

    def _make_env(self, render_mode=None):
        return gym.make(self.env_id, render_mode=render_mode, **self.env_make_params)

    def _rename_latest_video(self, reward_value, episode):
        try:
            candidates = [
                os.path.join(self.BEST_VIDEO_DIR, name)
                for name in os.listdir(self.BEST_VIDEO_DIR)
                if name.endswith(".mp4")
            ]
            if not candidates:
                return
            latest = max(candidates, key=os.path.getmtime)
            reward_tag = f"r{reward_value:.2f}"
            new_name = f"best_ep{episode}_{reward_tag}.mp4"
            new_path = os.path.join(self.BEST_VIDEO_DIR, new_name)
            os.replace(latest, new_path)
        except OSError:
            pass

    def save_best_run_video(self, policy_dqn, episode, seed):
        name_prefix = f"best_ep{episode}_tmp"
        env = self._make_env(render_mode="rgb_array")
        env = RecordVideo(
            env,
            self.BEST_VIDEO_DIR,
            name_prefix=name_prefix,
            episode_trigger=lambda ep: True,
            disable_logger=True,
        )

        state, _ = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        terminated = False
        episode_reward = 0.0

        while not terminated and episode_reward < self.stop_on_reward:
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

            new_state, reward, terminated, _, _ = env.step(action)
            episode_reward += reward
            state = torch.tensor(new_state, dtype=torch.float32).to(device)

        env.close()
        self._rename_latest_video(episode_reward, episode)
        return episode_reward

    def save_checkpoint_video(self, policy_dqn, episode):
        env = self._make_env(render_mode="rgb_array")
        env = RecordVideo(
            env,
            self.CHECKPOINT_VIDEO_DIR,
            name_prefix=f"checkpoint_ep{episode}_tmp",
            episode_trigger=lambda _: True,
            disable_logger=True,
        )
        state, _ = env.reset(seed=episode + 1)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        terminated = False
        episode_reward = 0.0

        while not terminated and episode_reward < self.stop_on_reward:
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
            new_state, reward, terminated, _, _ = env.step(action)
            episode_reward += reward
            state = torch.tensor(new_state, dtype=torch.float32).to(device)

        env.close()
        try:
            candidates = [
                os.path.join(self.CHECKPOINT_VIDEO_DIR, f)
                for f in os.listdir(self.CHECKPOINT_VIDEO_DIR)
                if f.endswith(".mp4")
            ]
            if candidates:
                latest = max(candidates, key=os.path.getmtime)
                new_path = os.path.join(self.CHECKPOINT_VIDEO_DIR, f"checkpoint_ep{episode}_r{episode_reward:.2f}.mp4")
                os.replace(latest, new_path)
        except OSError:
            pass
        return episode_reward

    def run(self, is_training=True, render=False):
        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        env = self._make_env(render_mode='human' if render else None)

        num_actions = env.action_space.n
        num_states = env.observation_space.shape[0]
        if self.enable_dueling_dqn:
            policy_dqn = DuelingDQN(num_states, num_actions, self.fc1_nodes).to(device)
        else:
            policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

        rewards_per_episode = []

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')

            epsilon_history = []

            memory = ReplayMemory(capacity=self.replay_memory_size, seed=23)

            epsilon = self.epsilon_init
            if self.enable_dueling_dqn:
                target_dqn = DuelingDQN(num_states, num_actions, self.fc1_nodes).to(device)
            else:
                target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device)

            target_dqn.load_state_dict(policy_dqn.state_dict())

            # Steps taken, used for syncing target network to policy network
            step_count = 0

            best_reward = float("-inf")
            best_greedy_reward = float("-inf")

            # Initialize optimizer after creating the DQN model
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()


        for episode in itertools.count():
            state, _ = env.reset(seed=episode+1) # reset environment and set seed for reproducibility
            state = torch.tensor(state, dtype=torch.float32).to(device)

            terminated = False
            episode_reward = 0.0

            #Play game until done (1 episode)
            while(not terminated and episode_reward < self.stop_on_reward):
                #Epsilon Greedy Action Selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()

                # Execute action (gymnasium expects a Python int)
                new_state, reward, terminated, _, info = env.step(action)

                # Accumulate Reward
                episode_reward += reward

                #Convert into tensors
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                if is_training:
                    action_tensor = torch.tensor(action, dtype=torch.int64, device=device)
                    memory.push((state, action_tensor, new_state, reward, terminated))
                    step_count += 1

                    # If enough experience has been collected, optimize every step
                    if len(memory) > self.batch_size:
                        mini_batch = memory.sample(self.batch_size)
                        self.optimize(mini_batch, policy_dqn, target_dqn)

                        # Decay epsilon
                        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                        epsilon_history.append(epsilon)

                        # Copy policy network to target network after a certain number of steps
                        if step_count > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_count = 0

                # Move to new state
                state = new_state

            rewards_per_episode.append(episode_reward)

            # Save model if we got a new best reward
            if is_training:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}Episode {episode}: New best reward {best_reward:.2f}, model saved."
                    print(log_message)
                    with open(self.LOG_FILE, "a") as log_file:
                        log_file.write(log_message + "\n")
                    self.save_best_run_video(policy_dqn, episode, seed=episode + 1)

                # Save checkpoint video every 10000 episodes
                if episode % 10000 == 0:
                    greedy_reward = self.save_checkpoint_video(policy_dqn, episode)
                    if greedy_reward > best_greedy_reward:
                        best_greedy_reward = greedy_reward
                        torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                        log_message = f"{datetime.now().strftime(DATE_FORMAT)}Episode {episode}: New best greedy reward {best_greedy_reward:.2f}, model saved."
                        print(log_message)
                        with open(self.LOG_FILE, "a") as log_file:
                            log_file.write(log_message + "\n")

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time
    

    def optimize(self, batch, policy_dqn, target_dqn):
        # For readyblity: Process each transition in the batch separately
        # for state, action, new_state, reward, terminated in batch:

        #     if terminated:
        #         target_q = reward
        #     else:
        #         with torch.no_grad():
        #             target_q = reward + self.discount_factor_g * target_dqn(new_state.unsqueeze(0)).max().item()
        #But we actually want pytorch optimal code:

        states, actions, new_states, rewards, terminations = zip(*batch)

        # Create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations, dtype=torch.bool).to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(new_states).max(dim=1)[1]
                target_q = rewards + (1-terminations.float()) * self.discount_factor_g * target_dqn(new_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            else:
                target_q = rewards + (1-terminations.float()) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
            
        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute loss for whole batch
        loss = self.loss_fn(current_q, target_q)

        #Optimize model
        self.optimizer.zero_grad() # Zero gradients
        loss.backward()            # Backpropagate loss
        self.optimizer.step()      # Update weights


    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs steps (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.xlabel('Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

if __name__ == "__main__":
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparams_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)