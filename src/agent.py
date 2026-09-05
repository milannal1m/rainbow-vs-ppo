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
    parser.add_argument('--evaluate', help='Evaluate saved model over --episodes greedy episodes', action='store_true')
    parser.add_argument('--episodes', type=int, default=100,
                       help='episodes for --evaluate; with --resume these are ADDED to the ones '
                            'already in evaluation/episodes_eval.csv')
    parser.add_argument('--resume',   help='Continue training from <run>_state.pt (Mario runs '
                                           'exceed the 24h SLURM wall clock)',
                        action='store_true')
    parser.add_argument('--evaluate-levels', metavar='TIERS', nargs='?',
                       const='train,tier0,tier1,tier2',
                       help='Mario per-level evaluation over the given tiers (default all four), '
                            'plus chronological runs of the original game with warps allowed')
    parser.add_argument('--episodes-per-level', type=int, default=30)
    parser.add_argument('--full-game-runs', type=int, default=10,
                       help='chronological original-game runs; 0 to skip')
    parser.add_argument('--policy', default='argmax',
                       choices=('argmax', 'stochastic', 'topk3'),
                       help='action selection at eval time (topk3 = the paper k=3 variant, PPO)')
    parser.add_argument('--levels', default=None,
                       help='Mario: evaluate only these levels (comma-separated), ignoring the '
                            'tier selection. Use to re-record a few levels without a full pass.')
    parser.add_argument('--videos', default='per_world',
                       help="Mario level clips: 'per_world' (default, one per world), 'all' "
                            "(every evaluated level), 'none', or a comma-separated list e.g. 1-1,4-2")
    parser.add_argument('--videos-per-level', type=int, default=1,
                       help='clips per level, best episodes first (default 1)')
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
        agent.train(resume=args.resume)
    elif args.evaluate_levels:
        import mario_eval
        tiers = tuple(t.strip() for t in args.evaluate_levels.split(',') if t.strip())
        if args.videos == 'none':
            vids = ()
        elif args.videos in ('per_world', 'all'):
            vids = args.videos                 # resolved after evaluation, see mario_eval.run
        else:
            vids = tuple(v.strip() for v in args.videos.split(',') if v.strip())
        only = tuple(l.strip() for l in args.levels.split(',') if l.strip()) if args.levels else None
        summary = mario_eval.run(agent, tiers=tiers, levels=only,
                                 episodes_per_level=args.episodes_per_level,
                                 policy_mode=args.policy,
                                 full_game_episodes=args.full_game_runs,
                                 video_levels=vids,
                                 videos_per_level=args.videos_per_level)
        for tier, t in summary["tiers"].items():
            pages = t["pages_macro"]
            print(f"{tier:9s} n={t['n_levels']:2d}  flag={t['flag_rate_macro']:.3f}  "
                  f"pages={'n/a' if pages is None else f'{pages:.2f}'}")
        if "full_game" in summary:
            g = summary["full_game"]
            print(f"\noriginal game ({g['episodes']} runs, warps allowed): furthest stage "
                  f"mean {g['furthest_index_mean']:.2f} (~{g['furthest_stage_mean_equiv']}), "
                  f"max {g['furthest_stage_max']}")
    elif args.evaluate:
        agent.evaluate(num_episodes=args.episodes, resume=args.resume)
    else:
        agent.test()
