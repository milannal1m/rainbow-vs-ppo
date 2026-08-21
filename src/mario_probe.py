"""Pre-flight checks for the Mario stack. Run before training and on a new machine.

Each check guards an invariant that can regress. The one-off investigations that shaped the design
are gone — their conclusions live in constants and in mario.md.

    python src/mario_probe.py                     # everything
    python src/mario_probe.py --only spaces,fps
"""
import argparse
import os
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import yaml

import mario_levels as ML
from env_factory import make_env

OK, BAD, WARN = "  OK ", " FAIL", " WARN"
RIGHT_RUN, LEFT_RUN = 3, 8      # COMPLEX_MOVEMENT: right+B, left+B


def _hdr(t):
    print(f"\n{'=' * 76}\n{t}\n{'=' * 76}")


def _emp(**over):
    base = {"action_set": "COMPLEX_MOVEMENT", "frame_skip": 4, "reward_clip": 15.0,
            "reward_divisor": 15.0, "noop_max": 30, "sticky_prob": 0.25,
            "max_episode_steps": 3000, "warp_bonus": 5.0, "level_sampler": "seed_hash"}
    base.update(over)
    return base


def _stack(level, **over):
    """The env exactly as an agent builds it, through env_factory."""
    return make_env("SuperMarioBros-v0", _emp(levels=[level], **over),
                    obs_size=84, frame_stack=4)


def _child(level):
    """Per-level child chain only, so per-FRAME info is visible."""
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    from mario_env import ACTION_SETS
    return JoypadSpace(gym.make(ML.env_id_for(level), render_mode=None),
                       ACTION_SETS["COMPLEX_MOVEMENT"])


# ── checks ───────────────────────────────────────────────────────────────────────────
def check_spaces():
    """Observation/action plumbing — catches a broken install immediately."""
    _hdr("SPACES + INFO")
    from mario_env import action_labels
    env = _stack("1-1")
    obs, _ = env.reset(seed=1)
    fps = env.unwrapped.metadata.get("render_fps")
    shape_ok = obs.shape == (4, 84, 84) and obs.dtype == np.float32 and obs.max() <= 1.0
    acts_ok = env.action_space.n == 12 and len(action_labels()) == 12
    print(f"       obs {obs.shape} {obs.dtype} in [{obs.min():.2f}, {obs.max():.2f}]   "
          f"actions {env.action_space.n}   render_fps {fps}")
    env.close()

    child = _child("1-1")
    _, info = child.reset(seed=1)
    needed = ("x_pos", "x_pos_max", "y_viewport", "player_state", "world", "stage", "area",
              "flag_get", "time", "single_stage")
    missing = [k for k in needed if k not in info]
    child.close()
    print(f"{OK if shape_ok else BAD}  (4,84,84) float32 in [0,1]")
    print(f"{OK if acts_ok else BAD}  Discrete(12) COMPLEX_MOVEMENT")
    print(f"{OK if fps == 15 else BAD}  render_fps 15 (= 60 // frame_skip)")
    print(f"{OK if not missing else BAD}  required info keys" + (f" missing {missing}" if missing else ""))
    return shape_ok and acts_ok and fps == 15 and not missing


def check_x_underflow():
    """Holding LEFT wraps x to 65535, which latches _x_position_max and kills the dense reward
    for the episode. MarioSanitizeX must catch it and be a no-op otherwise."""
    _hdr("X UNDERFLOW GUARD")
    import gymnasium as gym
    from nes_py.wrappers import JoypadSpace
    from mario_env import ACTION_SETS, MarioSanitizeX, MAX_PLAUSIBLE_X

    def run(env_id, action, frames, guard):
        e = JoypadSpace(gym.make(env_id, render_mode=None), ACTION_SETS["COMPLEX_MOVEMENT"])
        if guard:
            e = MarioSanitizeX(e)
        _, i = e.reset(seed=1)
        xs = [i["x_pos"]]
        for _ in range(frames):
            _, _, term, trunc, i = e.step(action)
            xs.append(i["x_pos"])
            if term or trunc:
                break
        out = (max(xs), e.unwrapped._x_position_max, getattr(e, "spurious", 0))
        e.close()
        return out

    lvl = "SuperMarioBros2-1-1-v0"          # wraps after ~8700 frames of run-left
    raw_x, raw_max, _ = run(lvl, LEFT_RUN, 10_000, guard=False)
    fix_x, fix_max, caught = run(lvl, LEFT_RUN, 10_000, guard=True)
    a = run("SuperMarioBros-1-1-v0", RIGHT_RUN, 1500, guard=False)
    b = run("SuperMarioBros-1-1-v0", RIGHT_RUN, 1500, guard=True)

    reproduced = raw_x > MAX_PLAUSIBLE_X or raw_max > MAX_PLAUSIBLE_X
    guarded = fix_x <= MAX_PLAUSIBLE_X and fix_max <= MAX_PLAUSIBLE_X
    noop = a[0] == b[0] and a[1] == b[1] and b[2] == 0
    print(f"       run-left  unguarded max_x={raw_x:6d} env_max={raw_max:6d}")
    print(f"       run-left  guarded   max_x={fix_x:6d} env_max={fix_max:6d} caught={caught}")
    print(f"{OK if reproduced else WARN}  underflow still reproduces without the guard")
    print(f"{OK if guarded else BAD}  guard keeps the counter sane")
    print(f"{OK if noop else BAD}  no-op on forward motion (max_x {a[0]} vs {b[0]})")
    return guarded and noop


def check_area_rebase():
    """After a forward pipe the dense reward stays 0 until x re-exceeds the old area's max.
    Unit-tested against a mock of the upstream rule — no scripted policy reaches 1-2's end pipe."""
    _hdr("AREA REBASE (unit)")
    import gymnasium as gym
    from mario_env import MarioAreaRebase

    class Mock(gym.Env):
        action_space = gym.spaces.Discrete(2)
        observation_space = gym.spaces.Box(0, 255, (1,), dtype=np.uint8)

        def __init__(self, switch=60, total=80):
            self.switch, self.total = switch, total

        def reset(self, *, seed=None, options=None):
            self.t, self.area = 0, 3
            self._x_position = self._x_position_max = 40
            return np.zeros(1, np.uint8), self._info()

        def _info(self):
            return {"area": self.area, "x_pos": self._x_position,
                    "x_pos_max": self._x_position_max, "progress_max": self._x_position_max}

        def step(self, _):
            self.t += 1
            if self.t == self.switch:
                self.area, self._x_position = 4, 40      # forward pipe: x restarts low
            else:
                self._x_position += 3
            r = self._x_position - self._x_position_max   # verbatim upstream _progress_reward
            if r <= 0:
                reward = 0
            else:
                self._x_position_max = self._x_position
                reward = 0 if r > 5 else r
            return np.zeros(1, np.uint8), float(reward), False, self.t >= self.total, self._info()

    def earned(fixed):
        env = MarioAreaRebase(Mock()) if fixed else Mock()
        env.reset(seed=1)
        before = after = 0.0
        switched = False
        while True:
            _, r, term, trunc, i = env.step(0)
            switched = switched or i["area"] != 3
            if switched:
                after += r
            else:
                before += r
            if term or trunc:
                return before, after

    b0, a0 = earned(False)
    b1, a1 = earned(True)
    print(f"       positive reward before / after the area change:")
    print(f"         unfixed {b0:7.1f} / {a0:7.1f}      fixed {b1:7.1f} / {a1:7.1f}")
    ok = a0 == 0.0 and a1 > 0.0 and b0 == b1
    print(f"{OK if ok else BAD}  bug reproduced, fix restores reward, no change before the switch")
    return ok


def check_decorrelation():
    """The emulator restores a backup on reset, so identical seeds give identical rollouts —
    greedy eval would have no variance and Rainbow could memorise one sequence per level. Sticky
    actions are what actually decorrelates; no-op starts barely move the needle here."""
    _hdr("DECORRELATION")

    def ret(noop, sticky, seed):
        env = _stack("1-1", noop_max=noop, sticky_prob=sticky, max_episode_steps=400)
        env.reset(seed=seed)
        total, xmax = 0.0, 0
        for s in range(400):
            _, r, term, trunc, i = env.step(RIGHT_RUN if (s // 8) % 3 else 4)
            total += r
            xmax = max(xmax, i.get("x_pos_max", 0))
            if term or trunc:
                break
        env.close()
        return round(total, 3), xmax

    seeds = range(1, 9)
    bare = [ret(0, 0.0, s) for s in seeds]
    live = [ret(30, 0.25, s) for s in seeds]
    bare_same = len({r for r, _ in bare}) == 1
    live_spread = max(x for _, x in live) - min(x for _, x in live)
    live_distinct = len({r for r, _ in live})
    print(f"       bare (noop=0, sticky=0): {len({r for r, _ in bare})}/8 distinct returns")
    print(f"       noop=30 + sticky=0.25:   {live_distinct}/8 distinct, x_pos_max spread {live_spread} px")
    print(f"{OK if bare_same else WARN}  determinism confirmed without decorrelation (the hazard)")
    print(f"{OK if live_distinct > 1 else BAD}  decorrelation produces real eval variance")
    return live_distinct > 1


def check_warps():
    """Warps are enabled and rewarded: check the tracker sees the stage jump, pays per stage
    skipped, and lets the episode continue."""
    _hdr("WARP TRACKING + BONUS")
    import gymnasium as gym
    from mario_env import MarioWarpTracker, level_index

    class FakeWarp(gym.Wrapper):
        """Rewrites world/stage after `after` steps, simulating a warp-zone exit."""
        def __init__(self, env, after=5, to=(4, 1)):
            super().__init__(env)
            self.after, self.to, self.n = after, to, 0

        def reset(self, **kw):
            self.n = 0
            return self.env.reset(**kw)

        def step(self, a):
            obs, r, term, trunc, i = self.env.step(a)
            self.n += 1
            if self.n >= self.after:
                i = dict(i, world=self.to[0], stage=self.to[1])
            return obs, r, term, trunc, i

    BONUS = 5.0
    env = MarioWarpTracker(FakeWarp(_child("1-2"), after=5, to=(4, 1)), warp_bonus=BONUS)
    env.reset(seed=1)
    ended, bonus_step, flagged = None, None, False
    for s in range(20):
        _, r, term, trunc, i = env.step(1)
        if i.get("warp_skip"):
            bonus_step = (s, i["warp_skip"], r)
        flagged = flagged or bool(i.get("warped"))
        if term or trunc:
            ended = s
            break
    env.close()

    # 1-2 (index 2) -> 4-1 (index 13): a single-stage env treats any change as a warp
    expected_skip = level_index(4, 1) - level_index(1, 2)
    continued = ended is None
    print(f"       warp detected at step {bonus_step[0] if bonus_step else None}, "
          f"skip={bonus_step[1] if bonus_step else 0} (expected {expected_skip}), "
          f"step reward {bonus_step[2] if bonus_step else 0:.1f}")
    ok_skip = bool(bonus_step) and bonus_step[1] == expected_skip
    ok_bonus = bool(bonus_step) and bonus_step[2] >= BONUS * expected_skip
    print(f"{OK if ok_skip else BAD}  stages-skipped counted correctly")
    print(f"{OK if ok_bonus else BAD}  warp bonus paid ({BONUS}/stage)")
    print(f"{OK if continued else BAD}  episode CONTINUES through the warp (not truncated)")
    print(f"{OK if flagged else BAD}  info['warped'] set for the metrics layer")
    return ok_skip and ok_bonus and continued and flagged


def check_game_metric():
    """Unit-test the "how far did it get" tracker.

    No blind scripted policy clears 1-1, so driving this directly is the only way to check
    cross-stage progress, stage-clear counting and warp accounting without a trained agent.
    """
    _hdr("FULL-GAME PROGRESS TRACKER (unit)")
    from mario_eval import GameProgressTracker, _index_to_stage

    def info(w, s, flag=False, warps=0, skipped=0):
        return {"world": w, "stage": s, "flag_get": flag, "warps": warps,
                "stages_skipped": skipped}

    # clear 1-1, advance to 1-2, clear it, then die back at 1-1 on a later life
    t = GameProgressTracker(info(1, 1))
    for step in (info(1, 1), info(1, 1, flag=True), info(1, 1, flag=True),   # flag latches
                 info(1, 2), info(1, 2, flag=True), info(1, 2), info(1, 1)):
        t.update(step)
    normal = (t.best_index == 2 and t.best_stage == "1-2" and t.cleared == 2
              and len(t.visited) == 2)
    print(f"       normal play: furthest={t.best_stage} (idx {t.best_index}) cleared={t.cleared} "
          f"visited={len(t.visited)}")
    print(f"{OK if normal else BAD}  advance tracked, flag latch counted once, regress ignored")

    # a warp from 1-2 to 4-1 must count as real progress
    t = GameProgressTracker(info(1, 1))
    t.update(info(1, 2))
    t.update(info(4, 1, warps=1, skipped=11))
    warped = (t.best_index == 13 and t.best_stage == "4-1" and t.warps == 1
              and t.stages_skipped == 11)
    print(f"       after a warp: furthest={t.best_stage} (idx {t.best_index}) "
          f"warps={t.warps} skipped={t.stages_skipped}")
    print(f"{OK if warped else BAD}  warp counted as progress, skip recorded")

    labels = [_index_to_stage(i) for i in (1, 4, 5, 13, 32)]
    idx_ok = labels == ["1-1", "1-4", "2-1", "4-1", "8-4"]
    print(f"{OK if idx_ok else BAD}  index<->stage mapping {labels}")
    return normal and warped and idx_ok


def check_levels_load():
    """All 32 SMB1 stages plus the tier-2 ids must construct and report their target."""
    _hdr("LEVELS LOAD")
    import gymnasium as gym
    import gym_super_mario_bros  # noqa: F401
    bad = []
    t0 = time.time()
    for level in ML.SMB1_LEVELS:
        w, s = ML.parse_level(level)
        e = gym.make(ML.env_id_for(level), render_mode=None)
        _, i = e.reset(seed=1)
        if (i["world"], i["stage"]) != (w, s) or i["area"] != ML.START_AREA[level]:
            bad.append(level)
        e.close()
    ll_bad = []
    for lvl in ML.get_split()["tier2"]:
        try:
            e = gym.make(lvl, render_mode=None); e.reset(seed=1); e.close()
        except Exception as exc:                       # noqa: BLE001
            ll_bad.append((lvl, type(exc).__name__))
    print(f"       32 SMB1 stages + {len(ML.get_split()['tier2'])} Lost Levels in "
          f"{time.time() - t0:.1f}s")
    print(f"{OK if not bad else BAD}  world/stage/area match START_AREA" + (f" — {bad}" if bad else ""))
    print(f"{OK if not ll_bad else BAD}  tier-2 ids load" + (f" — {ll_bad}" if ll_bad else ""))
    return not bad and not ll_bad


def check_pool():
    """The level must be a pure function of the reset seed, so both algorithms see the same
    sequence, and it must not depend on the global torch/numpy streams."""
    _hdr("LEVEL POOL")
    from mario_env import MultiLevelMarioEnv
    levels = list(ML.split_levels("smb1_small", "train"))

    def seq(torch_seed):
        import torch
        torch.manual_seed(torch_seed)
        np.random.seed(torch_seed)
        env = MultiLevelMarioEnv(levels, sampler="seed_hash")
        out = []
        for ep in range(60):
            env.reset(seed=ep + 1)
            out.append(env.current_level)
        env.close()
        return out

    a, b = seq(0), seq(12345)
    counts = {lv: a.count(lv) for lv in levels}
    spread = max(counts.values()) - min(counts.values())
    print(f"       first 10: {a[:10]}")
    print(f"       histogram over 60 episodes: {counts}")
    print(f"{OK if a == b else BAD}  same sequence under different torch/numpy seeds")
    print(f"{OK if spread <= 12 else WARN}  roughly uniform (max-min {spread})")
    return a == b


def check_split():
    """Print the frozen split and re-assert its constraints."""
    _hdr(f"SPLIT — {ML.DEFAULT_SPLIT}")
    split = ML.get_split()
    for tier in ML.TIERS + ("excluded",):
        levels = split.get(tier, ())
        if not levels:
            continue
        detail = ", ".join(
            f"{l}({ML.LEVELS[l].archetype[:5]}{'/warp' if l in ML.WARP_LEVELS else ''})"
            if l in ML.LEVELS else l for l in levels)
        print(f"   {tier:9s} ({len(levels):2d})  {detail}")
    ML._validate()
    held = set(split["tier1"])
    print(f"\n{OK}  no warp destination held out: {sorted(ML.WARP_DESTINATIONS & held) or 'none'}")
    print(f"{OK}  protocol constraints re-asserted (twins, archetypes, mazes)")
    return True


def check_fps():
    """Throughput, for the compute budget. Re-run on the cluster — a laptop differs."""
    _hdr("THROUGHPUT")
    env = _stack("1-1", noop_max=0, max_episode_steps=10 ** 9)
    env.reset(seed=1)
    t0, n = time.time(), 1200
    for s in range(n):
        _, _, term, trunc, _ = env.step(RIGHT_RUN)
        if term or trunc:
            env.reset(seed=s + 2)
    dt = time.time() - t0
    env.close()
    aps = n / dt
    print(f"       {aps:.0f} agent steps/s env-only ({aps * 4:.0f} NES frames/s)")
    print(f"       ~50% of that with the CNN update in the loop:")
    for b in (5, 10, 15):
        print(f"         {b:2d}M steps ≈ {b * 1e6 / (aps * 0.5) / 3600:5.1f} h "
              f"({'fits' if b * 1e6 / (aps * 0.5) / 3600 < 24 else 'EXCEEDS'} a 24 h job)")
    return True


def check_compare(names):
    """The env block must be identical between the two algorithm configs."""
    _hdr(f"COMPARE — {' vs '.join(names)}")
    with open("hyperparams.yml") as f:
        hp = yaml.safe_load(f)
    keys = ("env_id", "env_package", "env_make_params", "frame_stack", "obs_size", "rgb_wrapper",
            "discount_factor_g", "max_env_steps", "stop_on_reward", "seed")
    cfgs = {n: hp[n] for n in names}
    diffs = [k for k in keys if len({repr(c.get(k)) for c in cfgs.values()}) > 1]
    for k in keys:
        print(f"{BAD if k in diffs else OK}  {k} = {cfgs[names[0]].get(k)!r}")
    if diffs:
        print("\nDIFFERENCES invalidate the algorithm comparison:")
        for k in diffs:
            for n, c in cfgs.items():
                print(f"    {k:20s} {n:22s} {c.get(k)!r}")
    return not diffs


CHECKS = {
    "spaces": check_spaces, "underflow": check_x_underflow, "area": check_area_rebase,
    "decorrelation": check_decorrelation, "warps": check_warps, "game": check_game_metric,
    "levels": check_levels_load, "pool": check_pool, "split": check_split, "fps": check_fps,
}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", help=f"comma-separated subset of: {','.join(CHECKS)}")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   default=["mario_rainbow", "mario_ppo"],
                   help="config pair whose env blocks must match")
    a = p.parse_args()

    names = [n.strip() for n in a.only.split(",")] if a.only else list(CHECKS)
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        print(f"unknown check(s) {unknown}; valid: {list(CHECKS)}", file=sys.stderr)
        return 2

    results = {}
    for n in names:
        try:
            results[n] = bool(CHECKS[n]())
        except Exception:                              # noqa: BLE001 — report, never crash
            import traceback
            traceback.print_exc()
            results[n] = False
    if not a.only:
        try:
            results["compare"] = bool(check_compare(a.compare))
        except Exception:                              # noqa: BLE001
            import traceback
            traceback.print_exc()
            results["compare"] = False

    _hdr("SUMMARY")
    for n, ok in results.items():
        print(f"{OK if ok else BAD}  {n}")
    failed = [n for n, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed"
          + (f" — FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
