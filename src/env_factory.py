"""The single place that builds an env plus its observation pipeline.

The wrapper chain used to be spelled out three times (BaseAgent._make_env, record_episode,
save_preprocessed_sanity_check), each applying FlappyBirdResetFix unconditionally — which is why
cartpole1 crashed. Mario is a branch here rather than a parallel pipeline, so preprocess_env stays
shared; mario_env is imported lazily so nes_py is never touched in the FlappyBird env.
"""
import gymnasium as gym

from utils import FlappyBirdResetFix, RGBObservationWrapper, preprocess_env

FLAPPYBIRD = "flappybird"
MARIO      = "mario"
GENERIC    = "generic"

# Keys in env_make_params that configure the Mario wrappers rather than gym.make(). Keeping them
# in that dict (already threaded through every call site) is what lets record_episode and
# save_preprocessed_sanity_check keep their signatures.
MARIO_WRAPPER_KEYS = frozenset({
    "action_set", "version", "frame_skip", "reward_clip", "reward_divisor",
    "noop_max", "sticky_prob", "max_episode_steps", "warp_bonus",
    "level_sampler", "use_gym_make", "completion_unclipped",
})
# Consumed by mario_levels.levels_from_config, not passed to the builder.
MARIO_LEVEL_KEYS = frozenset({"levels", "level_split", "level_set"})
# Not an env parameter at all -- an explicit override for env_kind detection.
KIND_KEY = "env_kind"


def env_kind(env_id, override=None):
    """"mario", "flappybird" or "generic" — generic gets no game-specific wrappers, which is
    what makes plain gymnasium envs like CartPole work."""
    if override:
        return override
    if env_id.startswith("SuperMarioBros"):
        return MARIO
    if env_id.startswith("FlappyBird"):
        return FLAPPYBIRD
    return GENERIC


def split_env_params(env_make_params, kind):
    """Split env_make_params into (gym.make kwargs, wrapper kwargs). For everything except mario
    the whole dict goes to gym.make, so FlappyBird is unchanged."""
    params = dict(env_make_params or {})
    params.pop(KIND_KEY, None)

    if kind != MARIO:
        return params, {}

    wrapper_kw, gym_kw = {}, {}
    for key, value in params.items():
        if key in MARIO_LEVEL_KEYS:
            continue  # resolved separately by mario_levels.levels_from_config
        elif key in MARIO_WRAPPER_KEYS:
            wrapper_kw[key] = value
        else:
            gym_kw[key] = value

    if gym_kw:
        raise ValueError(
            f"unknown Mario env_make_params keys: {sorted(gym_kw)}. "
            f"Known wrapper keys: {sorted(MARIO_WRAPPER_KEYS)}; "
            f"level keys: {sorted(MARIO_LEVEL_KEYS)}"
        )
    return {}, wrapper_kw


def make_env(env_id, env_make_params=None, *, render_mode=None,
             obs_size=80, frame_stack=None, rgb_wrapper=False, kind=None, levels=None):
    """Build an env with its observation pipeline. `levels` overrides the list resolved from
    env_make_params, which is how the evaluator builds a single-stage env from a multi-level config."""
    kind = env_kind(env_id, kind or (env_make_params or {}).get(KIND_KEY))
    needs_rgb = rgb_wrapper or frame_stack
    gym_kw, wrapper_kw = split_env_params(env_make_params, kind)

    if kind == MARIO:
        # Lazy: keeps nes_py off the FlappyBird import path.
        from mario_env import make_mario_env
        from mario_levels import levels_from_config

        if levels is None:
            levels, _ = levels_from_config(env_make_params)
        env = make_mario_env(
            levels=levels,
            render_mode="rgb_array" if needs_rgb else render_mode,
            **wrapper_kw,
        )
    else:
        env = gym.make(env_id, render_mode="rgb_array" if needs_rgb else render_mode, **gym_kw)
        if kind == FLAPPYBIRD:
            env = FlappyBirdResetFix(env)   # conditional: it pokes flappy-bird internals
        if rgb_wrapper:
            env = RGBObservationWrapper(env)

    if frame_stack:
        env = preprocess_env(env, obs_size, frame_stack)
    if needs_rgb and render_mode == "human":
        from gymnasium.wrappers import HumanRendering
        env = HumanRendering(env)
    return env


def describe_env_params(env_make_params, kind=None, env_id=""):
    """One-line provenance for the run log, so a Rainbow-vs-PPO comparison stays auditable."""
    kind = env_kind(env_id, kind or (env_make_params or {}).get(KIND_KEY))
    if kind != MARIO:
        return f"kind={kind}"
    from mario_levels import levels_from_config
    _, provenance = levels_from_config(env_make_params)
    _, wrapper_kw = split_env_params(env_make_params, kind)
    knobs = " ".join(f"{k}={v}" for k, v in sorted(wrapper_kw.items()))
    return f"kind=mario levels={provenance} {knobs}"
