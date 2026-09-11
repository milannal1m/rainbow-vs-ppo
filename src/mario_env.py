"""Mario env construction: the game-specific wrappers plus the multi-level env.

Imports gym_super_mario_bros at module scope, so env_factory.make_env imports this lazily from
its Mario branch only -- the package does not exist in the FlappyBird environment. Most wrappers
here work around one specific upstream quirk, named in each class docstring.
"""
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, TimeLimit

import gym_super_mario_bros  # noqa: F401  -- registers the SuperMarioBros* env ids
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from mario_levels import env_id_for

ACTION_SETS = {
    "RIGHT_ONLY": RIGHT_ONLY,              # 5
    "SIMPLE_MOVEMENT": SIMPLE_MOVEMENT,    # 7  (no `down` -> cannot enter vertical pipes)
    "COMPLEX_MOVEMENT": COMPLEX_MOVEMENT,  # 12 (paper parity; `down` needed for pipe routes)
}

NES_FPS = 60

# Salts keep the derived streams independent of each other and of the global RNGs, so every
# stochastic element is a pure function of the reset seed and both algorithms see the same
# episode for a given seed.
_NOOP_SALT   = 0x4E4F4F50  # "NOOP"
_STICKY_SALT = 0x53544B59  # "STKY"
_LEVEL_SALT  = 0x4C564C53  # "LVLS"

# Player state 0x0b is the death animation; y_viewport > 1 is documented upstream as
# "below viewport (i.e. dead, falling down a hole)".
_PLAYER_STATE_DYING = 0x0b


def action_labels(action_set="COMPLEX_MOVEMENT"):
    """Action names for the Grad-CAM panels, which are otherwise hardcoded to flap/no-flap."""
    return [" ".join(buttons) for buttons in ACTION_SETS[action_set]]


def _rng(salt, seed):
    """A generator derived from (salt, seed). Deterministic, and independent of global RNGs."""
    return np.random.default_rng([salt, int(seed) & 0x7FFFFFFF])


# No SMB1 or Lost Levels stage is longer than ~5000 px, so anything past this is an underflow
# wrap of the 16-bit x counter, not a position.
MAX_PLAUSIBLE_X = 8192
# Mario runs at under 4 px/frame. Needed as well as the absolute bound: the underflow also emits
# 255 when only the low byte wrapped, which passes any absolute check. Area transitions are exempt.
MAX_X_JUMP = 32


# ── Upstream-behaviour fixes ─────────────────────────────────────────────────────────
class MarioSanitizeX(gym.Wrapper):
    """Repair `x_pos` when holding LEFT drives it below 0 and the RAM bytes wrap to 65535.

    Not just a metrics problem: `_progress_reward` latches `_x_position_max` before its cap check,
    so one wrap kills the dense progress reward for the rest of the episode. Must wrap the ROM env
    directly, so `self.env.unwrapped` is the SuperMarioBrosEnv.
    """

    def __init__(self, env):
        super().__init__(env)
        self._reset_state()

    def _reset_state(self):
        self._last_x = 0
        self._have_x = False
        self._last_good_max = 0
        self._prev_area = None
        self.spurious = 0

    def _sanitize(self, info):
        u = self.env.unwrapped
        x = info.get("x_pos", 0)
        area = info.get("area")
        area_changed = self._prev_area is not None and area != self._prev_area

        bad = x > MAX_PLAUSIBLE_X or (
            self._have_x and not area_changed and abs(x - self._last_x) > MAX_X_JUMP)

        if bad:
            self.spurious += 1
            x = self._last_x                                # hold the last trusted position
            info["x_pos"] = x
            info["progress"] = x
            u._x_position_max = self._last_good_max          # undo the poisoned high-water mark
        else:
            self._last_x = x
            self._have_x = True
            self._last_good_max = u._x_position_max

        self._prev_area = area
        info["x_pos_max"] = u._x_position_max
        info["progress_max"] = u._x_position_max
        info["x_spurious"] = bad
        return info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_state()
        return obs, self._sanitize(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, self._sanitize(info)


class MarioAreaRebase(gym.Wrapper):
    """Rebase the env's max-x high-water mark when Mario changes area.

    `x_pos` restarts per area after a pipe but `_x_position_max` does not, so the dense reward
    dies until x re-exceeds the old area's max. Marks are kept per area rather than reset, so
    returning from a bonus room does not re-pay earned reward. Must sit below the frame-skip
    wrapper to see every frame.
    """

    def __init__(self, env):
        super().__init__(env)
        self._area_max = {}
        self._prev_area = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._area_max = {}
        self._prev_area = info.get("area")
        self._area_max[self._prev_area] = info.get("x_pos_max", 0)
        info["area_max"] = dict(self._area_max)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        area = info.get("area")

        if self._prev_area is not None and area != self._prev_area:
            # Entering (or returning to) an area: restore that area's own high-water mark, or
            # start from the current x if this is the first visit. The transition step itself
            # forfeits its progress reward -- the env already computed it before we got here.
            restored = self._area_max.get(area, info.get("x_pos", 0))
            self.env.unwrapped._x_position_max = restored
            info["x_pos_max"] = restored
            info["progress_max"] = restored

        self._prev_area = area
        self._area_max[area] = max(self._area_max.get(area, 0), info.get("x_pos_max", 0))
        info["area_max"] = dict(self._area_max)
        return obs, reward, terminated, truncated, info


def level_index(world, stage):
    """Chronological position of a stage in the game: 1-1 -> 1, 1-2 -> 2, ... 8-4 -> 32."""
    return (int(world) - 1) * 4 + int(stage)


class MarioWarpTracker(gym.Wrapper):
    """Detect warp-zone use, pay `warp_bonus` per stage skipped, and let the episode continue.

    A warp is a non-consecutive advance in stage order: normal play steps +1, a warp jumps
    several at once. In a single-stage env any world/stage change counts. Paying per stage
    skipped makes a deeper warp worth more. Backward warps are recorded but not paid.
    """

    def __init__(self, env, warp_bonus=0.0):
        super().__init__(env)
        self.warp_bonus = float(warp_bonus)
        self._prev = None
        self._single_stage = False
        self.warps = 0
        self.stages_skipped = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev = (info["world"], info["stage"])
        self._single_stage = bool(info.get("single_stage"))
        self.warps = 0
        self.stages_skipped = 0
        info.update(warped=False, warp_skip=0, warps=0, stages_skipped=0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        now = (info["world"], info["stage"])
        skipped = 0

        if now != self._prev:
            delta = level_index(*now) - level_index(*self._prev)
            # Single stage: any change is a warp. Full game: only a non-consecutive jump is.
            if self._single_stage or delta > 1:
                skipped = max(0, delta - 1) if not self._single_stage else max(0, delta)
                self.warps += 1
                self.stages_skipped += skipped
                if self.warp_bonus and skipped:
                    reward += self.warp_bonus * skipped
            self._prev = now

        info.update(warped=skipped > 0 or self.warps > 0, warp_skip=skipped,
                    warps=self.warps, stages_skipped=self.stages_skipped)
        return obs, reward, terminated, truncated, info


class NoopResetWrapper(gym.Wrapper):
    """Take a seed-derived number of NOOP frames after reset, to decorrelate episodes.

    The emulator restores a backup on reset, so identical seeds would otherwise give identical
    rollouts. On Mario this decorrelates only weakly -- standing still scrolls nothing, and sticky
    actions do the real work -- but it is free. k is a pure function of the seed, so eval episodes
    stay paired across algorithms.
    """

    def __init__(self, env, noop_max=30, noop_action=0):
        super().__init__(env)
        self.noop_max = int(noop_max)
        self.noop_action = noop_action

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        if self.noop_max <= 0:
            return obs, info

        gen = _rng(_NOOP_SALT, seed) if seed is not None else self.np_random
        k = int(gen.integers(0, self.noop_max + 1))

        for _ in range(k):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                # Should not happen (standing still only burns clock), but never hand back a
                # terminal observation as an initial state.
                obs, info = self.env.reset(seed=seed, options=options)
                break

        info["noops"] = k
        return obs, info


class StickyActionWrapper(gym.Wrapper):
    """Sticky actions (Machado et al.): repeat the previous action with probability p.

    On by default. This is the primary decorrelation for a deterministic emulator — measured at
    ~400 px of trajectory spread against ~3 px for no-op starts. Without it, greedy eval has no
    variance and Rainbow's argmax could memorise one action sequence per level.
    """

    def __init__(self, env, sticky_prob=0.0):
        super().__init__(env)
        self.sticky_prob = float(sticky_prob)
        self._last_action = 0
        self._gen = None

    def reset(self, *, seed=None, options=None):
        self._last_action = 0
        self._gen = _rng(_STICKY_SALT, seed) if seed is not None else self.np_random
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        if self.sticky_prob > 0.0 and self._gen.random() < self.sticky_prob:
            action = self._last_action
        else:
            self._last_action = action
        return self.env.step(action)


class ClipScaleReward(gym.Wrapper):
    """reward -> clip(reward, -clip, +clip) / divisor on the summed agent-step reward.

    Sits above the frame-skip wrapper, so it acts on the sum of the 4 skipped frames. The env has
    already clamped each FRAME to its declared reward_range (-15, +15), so a sum can still carry
    +-60; clip == divisor == 15 bounds an ordinary agent step to [-1, +1], which is what makes
    Rainbow's fixed C51 support viable. MarioEpisodeInfo records the unscaled return.

    completion_unclipped re-adds the one-off +50 flag bonus from
    info["reward_components"]["completion"]. That component is raw while the frame it arrived on
    was already clamped to +15, so no clip setting here can recover it -- the components dict is
    the only route. Without it a completion scores the same +1 as sustained running, which asks
    the agent to travel right rather than to finish. Ordinary steps are untouched, so the
    discounted value gains one ~+3.3 spike per episode instead of being rescaled.
    """

    def __init__(self, env, clip=15.0, divisor=15.0, completion_unclipped=False):
        super().__init__(env)
        self.clip = float(clip)
        self.divisor = float(divisor)
        self.completion_unclipped = bool(completion_unclipped)

    def _scale(self, reward):
        if self.clip is not None:
            reward = max(-self.clip, min(self.clip, reward))
        return reward / self.divisor

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self.completion_unclipped:
            return obs, self._scale(reward), terminated, truncated, info
        # info is the last inner frame's, which on a completion step IS the flag frame: the
        # single-stage env terminates there. The bonus is ADDED, not subtracted out -- `reward`
        # never contained the raw 50, only the +15 the env clamped that frame to.
        bonus = float((info.get("reward_components") or {}).get("completion", 0.0))
        scaled = self._scale(reward)
        if bonus:
            scaled += bonus / self.divisor
        return obs, scaled, terminated, truncated, info


# ── Episode bookkeeping ──────────────────────────────────────────────────────────────
class MarioEpisodeInfo(gym.Wrapper):
    """Accumulate per-episode stats and emit them as info["episode"] on termination.

    Sits below the frame-skip wrapper and below the reward scaling, so it sees every frame and
    `return_raw` is the true unscaled return. `covered_px` sums the per-area spans traversed,
    which stays meaningful across the area transitions where x_pos_max does not.
    """

    def __init__(self, env, level=None):
        super().__init__(env)
        self.level = level
        self._reset_state()

    def _reset_state(self):
        self._raw_return = 0.0
        self._frames = 0
        self._area_span = {}          # area -> [min_x, max_x]
        self._last_info = {}
        self._steps_since_progress = 0
        self._best_covered = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._reset_state()
        self._note(info)
        return obs, info

    def _note(self, info):
        area = info.get("area")
        x = info.get("x_pos", 0)
        span = self._area_span.get(area)
        if span is None:
            self._area_span[area] = [x, x]
        else:
            span[0] = min(span[0], x)
            span[1] = max(span[1], x)
        self._last_info = info

    @property
    def covered_px(self):
        return sum(hi - lo for lo, hi in self._area_span.values())

    def _death_cause(self, terminated, truncated, info):
        """Classify how the episode ended. y_viewport > 1 means falling down a hole and
        player_state 0x0b is the death animation, so pit vs enemy is a read, not a guess."""
        if info.get("flag_get"):
            return "flag"
        if info.get("warped"):
            return "warp_exit"
        if info.get("y_viewport", 1) > 1:
            return "pit"
        if info.get("time", 1) <= 0:
            return "timeout"
        if info.get("player_state") == _PLAYER_STATE_DYING or info.get("is_dead"):
            return "enemy"
        if truncated:
            return "step_cap"
        return "unknown"

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._raw_return += float(reward)
        self._frames += 1
        self._note(info)

        covered = self.covered_px
        if covered > self._best_covered:
            self._best_covered = covered
            self._steps_since_progress = 0
        else:
            self._steps_since_progress += 1

        if terminated or truncated:
            info["episode"] = {
                "level":        self.level,
                "world":        info.get("world"),
                "stage":        info.get("stage"),
                "return_raw":   self._raw_return,
                "frames":       self._frames,
                "x_pos_max":    max((hi for _, hi in self._area_span.values()), default=0),
                "covered_px":   covered,
                "pages":        covered // 256,
                "max_area":     max(self._area_span, default=None),
                "areas":        len(self._area_span),
                "flag_get":     bool(info.get("flag_get")),
                "warped":       bool(info.get("warped")),
                "warps":        int(info.get("warps", 0)),
                "stages_skipped": int(info.get("stages_skipped", 0)),
                "death_cause":  self._death_cause(terminated, truncated, info),
                "time_left":    info.get("time"),
                "coins":        info.get("coins"),
                "score":        info.get("score"),
                "stalled_for":  self._steps_since_progress,
                "area_span":    {a: list(s) for a, s in self._area_span.items()},
            }
        return obs, reward, terminated, truncated, info


# ── Multi-level env ──────────────────────────────────────────────────────────────────
class MultiLevelMarioEnv(gym.Env):
    """Holds one child env per level and switches level on reset().

    A pool is unavoidable: the target stage is written to RAM only in `_skip_start_screen()`
    during __init__, and reset() restores that backup, so an instance cannot be retargeted.
    Children are built lazily; the first eagerly, so the spaces are known at construction.
    Subclasses gym.Env (not Wrapper) so `env.unwrapped` resolves here, which is how the
    evaluator reaches set_level().
    """

    def __init__(self, levels, *, render_mode=None, action_set="COMPLEX_MOVEMENT",
                 version="v0", sampler="seed_hash", frame_skip=4,
                 warp_bonus=0.0):
        if not levels:
            raise ValueError("MultiLevelMarioEnv needs at least one level")
        self.levels = tuple(levels)
        self.render_mode = render_mode
        self.action_set = action_set
        self.version = version
        self.sampler = sampler
        self.warp_bonus = warp_bonus
        self.metadata = {
            "render_modes": ["rgb_array", "human"],
            # One rendered frame per agent step, so this is the physically correct playback rate.
            "render_fps": max(1, NES_FPS // max(1, int(frame_skip))),
        }

        self._children = {}
        self._current_level = self.levels[0]
        self._episodes = 0

        first = self._child(self.levels[0])
        self.observation_space = first.observation_space
        self.action_space = first.action_space

    # -- child management ----------------------------------------------------------
    def _build_child(self, level):
        """Build one fully-wrapped single-stage env.

        Anything needing per-frame info or the ROM env's internals belongs here, below
        MultiLevelMarioEnv: the frame-skip wrapper above only surfaces the last inner info.
        """
        # A level is either a bare "3-2" (SMB1) or a full id like "SuperMarioBros2-C-3-v0".
        env_id = (level if level.startswith("SuperMarioBros")
                  else env_id_for(level, version=self.version))

        env = gym.make(env_id, render_mode="rgb_array")
        env = JoypadSpace(env, ACTION_SETS[self.action_set])
        env = MarioSanitizeX(env)      # must precede everything that reads x_pos
        env = MarioAreaRebase(env)
        env = MarioWarpTracker(env, warp_bonus=self.warp_bonus)
        env = MarioEpisodeInfo(env, level=level)
        return env

    def _child(self, level):
        if level not in self._children:
            self._children[level] = self._build_child(level)
        return self._children[level]

    # -- level selection -----------------------------------------------------------
    def _pick_level(self, seed):
        if len(self.levels) == 1:
            return self.levels[0]
        if self.sampler == "round_robin":
            return self.levels[self._episodes % len(self.levels)]
        if self.sampler == "seed_hash" and seed is not None:
            # Pure function of the reset seed. Both train loops reset with seed=episode+1, so
            # the two algorithms see the same level sequence despite consuming the global RNGs
            # at different rates.
            idx = int(_rng(_LEVEL_SALT, seed).integers(0, len(self.levels)))
            return self.levels[idx]
        return self.levels[int(self.np_random.integers(0, len(self.levels)))]

    @property
    def current_level(self):
        return self._current_level

    # -- gym API -------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        level = self._pick_level(seed)
        self._current_level = level
        self._episodes += 1
        obs, info = self._child(level).reset(seed=seed, options=None)
        info["level"] = level
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._child(self._current_level).step(action)
        info["level"] = self._current_level
        if "episode" in info:
            info["episode"]["level"] = self._current_level
        return obs, reward, terminated, truncated, info

    def render(self):
        # .copy() is load-bearing: nes-py returns a zero-copy view of the screen buffer, so
        # RecordVideo would hold N aliases of it and every mp4 frame would show the final state.
        frame = self._child(self._current_level).render()
        return None if frame is None else frame.copy()

    def close(self):
        # nes-py has a history of double-free on close(); pop first so each child closes once.
        while self._children:
            _, child = self._children.popitem()
            try:
                child.close()
            except Exception:  # noqa: BLE001 -- a failing close must not mask the real error
                pass


# ── Public builder ───────────────────────────────────────────────────────────────────
def make_mario_env(*, levels, render_mode=None, action_set="COMPLEX_MOVEMENT", version="v0",
                   frame_skip=4, reward_clip=15.0, reward_divisor=15.0, noop_max=30,
                   sticky_prob=0.25, max_episode_steps=3000, warp_bonus=0.0,
                   level_sampler="seed_hash",
                   completion_unclipped=False):
    """Assemble the Mario env, up to but not including the observation pipeline.

    env_factory adds preprocess_env on top, so observation handling stays shared with
    FlappyBird. Per-frame concerns live inside each child; agent-step concerns out here.
    """
    env = MultiLevelMarioEnv(
        levels,
        render_mode=render_mode,
        action_set=action_set,
        version=version,
        sampler=level_sampler,
        frame_skip=frame_skip,
        warp_bonus=warp_bonus,
    )
    env = NoopResetWrapper(env, noop_max=noop_max)
    if sticky_prob > 0.0:
        env = StickyActionWrapper(env, sticky_prob=sticky_prob)
    if frame_skip and frame_skip > 1:
        env = MaxAndSkipObservation(env, skip=frame_skip)
    env = ClipScaleReward(env, clip=reward_clip, divisor=reward_divisor,
                          completion_unclipped=completion_unclipped)
    if max_episode_steps:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env
