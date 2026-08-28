"""Record a length-capped video of a trained policy playing.

A full episode is not recordable once an agent converges: flappybird_ppo_tuned_3 survives
~25,000 s of game time per episode (~758k frames), and RecordVideo buffers every frame in RAM
at ~0.44 MB, i.e. ~335 GB. That is why `--evaluate` writes evaluation.png/.log but no mp4.

--stream pipes frames straight into ffmpeg instead, so memory is O(1) and the only limits are
compute time and disk (~75 MB for a 7-hour FlappyBird episode). Use it for anything long.

Usage:
    python src/record_video.py flappybird_ppo_tuned_3 --seconds 60
    python src/record_video.py flappybird_ppo_tuned_3 --stream            # whole episode
    python src/record_video.py mario_ppo --seconds 45 --seed 7 --level 1-1
"""
import argparse
import os
import shutil
import signal
import sys
import subprocess
import yaml
import torch

from config import RUNS_DIR, HEADLESS

if HEADLESS:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from utils import record_episode

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def _ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    from imageio_ffmpeg import get_ffmpeg_exe   # ships with moviepy, already a dependency
    return get_ffmpeg_exe()


def stream_episode(policy, env, seed, device, out_path, fps, max_steps=None):
    """Greedy episode written frame-by-frame to ffmpeg. Constant memory, unlike RecordVideo."""
    # SLURM sends SIGTERM (or SIGUSR1 with --signal) before SIGKILL. Break the loop on it so the
    # finally-block closes the pipe and ffmpeg writes the moov atom -- a hard kill mid-write
    # leaves an unplayable file.
    stop = {"now": False}

    def _on_signal(signum, _frame):
        stop["now"] = True
        print(f"  [signal {signum}] finalising the file...", flush=True)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, _on_signal)

    state, _ = env.reset(seed=seed)
    frame = env.render()
    h, w = frame.shape[:2]

    proc = subprocess.Popen(
        [_ffmpeg_exe(), "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}",
         "-i", "-", "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
         out_path],
        stdin=subprocess.PIPE)

    state = torch.tensor(state, dtype=torch.float32).to(device)
    reward_total, steps = 0.0, 0
    terminated = truncated = False
    try:
        proc.stdin.write(frame.tobytes())
        while (not (terminated or truncated) and not stop["now"]
               and (max_steps is None or steps < max_steps)):
            with torch.no_grad():
                out = policy(state.unsqueeze(0))
                logits = out[0] if isinstance(out, tuple) else out
                action = logits.squeeze().argmax().item()
            new_state, reward, terminated, truncated, _ = env.step(action)
            reward_total += reward
            steps += 1
            proc.stdin.write(env.render().tobytes())
            state = torch.tensor(new_state, dtype=torch.float32).to(device)
            if steps % (int(fps) * 300) == 0:               # progress every 5 min of footage
                print(f"  {steps/fps/60:6.1f} min recorded, reward {reward_total:.0f}", flush=True)
    finally:
        proc.stdin.close()
        proc.wait()
    return reward_total, steps


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("hyperparameters")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="clip length in real playback seconds (default 60); ignored by --stream "
                        "unless given explicitly")
    p.add_argument("--stream", action="store_true",
                   help="pipe frames to ffmpeg (O(1) memory) instead of buffering in RAM; "
                        "records the WHOLE episode unless --seconds is also passed")
    p.add_argument("--seed", type=int, default=1, help="episode seed")
    p.add_argument("--level", default=None, help="Mario only: pin one level, e.g. 1-1")
    p.add_argument("--out", default=None, help="output dir (default runs/<name>/videos)")
    p.add_argument("--name", default=None, help="filename prefix (default clip_s<seed>)")
    a = p.parse_args()

    with open("hyperparams.yml", "r") as f:
        hp = yaml.safe_load(f)[a.hyperparameters]

    if hp.get("algorithm", "dqn") == "ppo":
        from ppo_agent import PPOAgent as AgentClass
    else:
        from dqn_agent import DQNAgent as AgentClass
    agent = AgentClass(hyperparams_set=a.hyperparameters)

    # _load_policy needs an env only for the observation/action shapes.
    env = agent._make_env()
    try:
        policy = agent._load_policy(env)
    finally:
        env.close()

    fps       = agent.metric_spec.fps
    max_steps = max(1, int(round(a.seconds * fps)))
    out_dir   = a.out or os.path.join(agent.RUN_DIR, "videos")
    os.makedirs(out_dir, exist_ok=True)

    emp = dict(agent.env_make_params or {})
    if a.level:                       # pin a single Mario level for a reproducible clip
        emp["levels"] = [a.level]
        emp.pop("level_split", None)
        emp.pop("level_set", None)

    prefix = a.name or f"clip_s{a.seed}"

    if a.stream:
        cap = max_steps if any(x.startswith("--seconds") for x in sys.argv) else None
        out_path = os.path.join(out_dir, f"{prefix}.mp4")
        print(f"streaming {'the whole episode' if cap is None else f'{a.seconds:.0f}s'} "
              f"from {a.hyperparameters}, seed {a.seed} -> {out_path}")
        from env_factory import make_env
        env = make_env(agent.env_id, emp, render_mode="rgb_array", obs_size=agent.obs_size,
                       frame_stack=agent.frame_stack, rgb_wrapper=agent.rgb_wrapper)
        try:
            reward, steps = stream_episode(policy, env, a.seed, device, out_path, fps, cap)
        finally:
            env.close()
        final = os.path.join(out_dir, f"{prefix}_r{reward:.2f}.mp4")
        os.replace(out_path, final)
        print(f"{steps} frames = {steps/fps/60:.1f} min, reward {reward:.2f} -> {final}")
        return

    print(f"recording {a.seconds:.0f}s ({max_steps} steps at {fps:g} fps) "
          f"from {a.hyperparameters}, seed {a.seed} -> {out_dir}")
    reward = record_episode(
        policy, agent.env_id, emp, out_dir, prefix,
        stop_on_reward=float("inf"), seed=a.seed, device=device,
        obs_size=agent.obs_size, frame_stack=agent.frame_stack,
        rgb_wrapper=agent.rgb_wrapper, max_steps=max_steps)
    print(f"reward over the clip: {reward:.2f}")


if __name__ == "__main__":
    main()
