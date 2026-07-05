import argparse
import os
import yaml
import torch

from config import RUNS_DIR, HEADLESS

os.makedirs(RUNS_DIR, exist_ok=True)

if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train',    help='Training mode', action='store_true')
    parser.add_argument('--evaluate', help='Evaluate saved model over 100 greedy episodes', action='store_true')
    args = parser.parse_args()

    with open("hyperparams.yml", "r") as f:
        hp = yaml.safe_load(f)[args.hyperparameters]

    algorithm = hp.get("algorithm", "dqn")

    if algorithm == "ppo":
        from ppo_agent import PPOAgent as AgentClass
    else:
        from dqn_agent import DQNAgent as AgentClass

    agent = AgentClass(hyperparams_set=args.hyperparameters)

    if args.train:
        agent.train()
    elif args.evaluate:
        agent.evaluate()
    else:
        agent.test()
