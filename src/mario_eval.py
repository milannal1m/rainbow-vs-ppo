"""Per-level Mario evaluation: the four tiers, aggregates and figures.

BaseAgent.evaluate() pools everything into one number, which cannot answer the question this
study is about, so each level is evaluated separately (mario_levels.SPLITS defines the tiers).

Also plays chronological runs of the original game with warps allowed — the "how far did it get"
number the per-level tiers cannot show.

Eval seeds start at 777_000, disjoint from the training seeds (episode + 1).
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import mario_levels as ML
from env_factory import describe_env_params
from utils import log

EVAL_SEED_BASE = 777_000
POLICY_MODES = ("argmax", "stochastic", "topk3")


# ── one episode ──────────────────────────────────────────────────────────────────────
def _select_action(model, state, mode, device):
    """Action selection for either algorithm. PPO exposes get_action(); DQN returns Q-values.
    stochastic/topk3 are PPO-only (topk3 is the paper's k=3 variant) and fall back to argmax on a
    value net, where the logits are Q-values rather than a policy."""
    is_ppo = hasattr(model, "get_action")
    with torch.no_grad():
        if is_ppo:
            if mode == "argmax":
                action, _, _, value = model.get_action(state.unsqueeze(0), deterministic=True)
                return int(action.item()), float(value.item())
            logits, value = model(state.unsqueeze(0))
            logits = logits.squeeze(0)
            if mode == "topk3":
                k = min(3, logits.numel())
                top_vals, top_idx = torch.topk(logits, k)
                pick = torch.distributions.Categorical(logits=top_vals).sample()
                return int(top_idx[pick].item()), float(value.item())
            pick = torch.distributions.Categorical(logits=logits).sample()
            return int(pick.item()), float(value.item())

        q = model(state.unsqueeze(0)).squeeze(0)
        return int(q.argmax().item()), float(q.max().item())


def _run_episode(agent, env, model, seed, metric, mode, device):
    state, _ = env.reset(seed=seed)
    state = torch.tensor(state, dtype=torch.float32).to(device)
    terminated = truncated = False
    total, length, aux = 0.0, 0, []

    while not (terminated or truncated) and total < agent.stop_on_reward:
        action, aux_val = _select_action(model, state, mode, device)
        aux.append(aux_val)
        new_state, reward, terminated, truncated, info = env.step(action)
        total += reward
        length += 1
        metric.update(reward, info)
        state = torch.tensor(new_state, dtype=torch.float32).to(device)

    return total, length, float(np.mean(aux)) if aux else 0.0


# ── progress normalisation ───────────────────────────────────────────────────────────
def _load_route_table(path=ML.ROUTE_TABLE_PATH):
    """The frozen route table, if one has been measured. Not derived from agent runs — that would
    make the metric depend on the thing being measured. Without it we report pages."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _route_progress(extras, level, route_table):
    """Route Progress in [0,1], or None when the level has no measured route."""
    if not route_table or level not in route_table:
        return None
    route = route_table[level].get("route")
    if not route:
        return None
    # JSON turns the int area keys into strings, so try both.
    spans = extras.get("area_span") or {}
    covered = total = 0
    for entry in route:
        area, x0, x1 = entry["area"], entry["x_min"], entry["x_max"]
        span = max(0, x1 - x0)
        total += span
        seen = spans.get(area, spans.get(str(area)))
        if seen:
            covered += min(max(0, seen[1] - x0), span)
    return (covered / total) if total else None


# ── per-level evaluation ─────────────────────────────────────────────────────────────
def evaluate_levels(agent, levels, *, episodes_per_level=30, seed_base=EVAL_SEED_BASE,
                    policy_mode="argmax", route_table=None, log_file=None, device=None):
    """Evaluate one level at a time. Returns {level: {metric: value}}."""
    from base_agent import device as default_device
    device = device or default_device
    if policy_mode not in POLICY_MODES:
        raise ValueError(f"policy_mode must be one of {POLICY_MODES}, got {policy_mode!r}")

    results = {}
    for idx, level in enumerate(levels):
        env = agent._make_env(levels=[level])
        try:
            model = agent._load_policy(env)
            rows = []
            for ep in range(episodes_per_level):
                metric = agent.metric_spec.new()
                seed = seed_base + 1000 * idx + ep
                total, length, aux = _run_episode(agent, env, model, seed, metric, policy_mode,
                                                  device)
                extras = metric.extras() or {}
                rows.append({
                    "seed": seed, "return_scaled": total, "length": length, "aux": aux,
                    "pages": metric.value(),
                    "return_raw": extras.get("return_raw"),
                    "x_pos_max": extras.get("x_pos_max", 0),
                    "covered_px": extras.get("covered_px", 0),
                    "flag_get": bool(extras.get("flag_get")),
                    "warped": bool(extras.get("warped")),
                    "death_cause": extras.get("death_cause") or "unknown",
                    "route_progress": _route_progress(extras, level, route_table),
                    # kept so progress can be re-normalised later without re-running anything;
                    # the multi-area levels need the per-area spans, not just covered_px
                    "area_span": extras.get("area_span"),
                })
        finally:
            env.close()

        results[level] = _summarise_level(level, rows)
        if log_file:
            r = results[level]
            log(f"  {level:22s} flag={r['flag_rate']:.2f} pages={r['pages_mean']:.2f} "
                f"x_max={r['x_pos_max_mean']:.0f} distinct_x={r['distinct_terminal_x']}/"
                f"{r['n']} deaths={r['death_mix']}", log_file)
    return results


def _summarise_level(level, rows):
    terminal_x = [r["x_pos_max"] for r in rows]
    causes = {}
    for r in rows:
        causes[r["death_cause"]] = causes.get(r["death_cause"], 0) + 1
    rp = [r["route_progress"] for r in rows if r["route_progress"] is not None]

    return {
        "level": level,
        "tier": ML.tier_of(level) or "unassigned",
        "archetype": ML.LEVELS[level].archetype if level in ML.LEVELS else "lost_levels",
        "n": len(rows),
        "flag_rate":      float(np.mean([r["flag_get"] for r in rows])),
        "warp_rate":      float(np.mean([r["warped"] for r in rows])),
        "pages_mean":     float(np.mean([r["pages"] for r in rows])),
        "pages_max":      float(np.max([r["pages"] for r in rows])),
        "x_pos_max_mean": float(np.mean(terminal_x)),
        "x_pos_max_best": int(np.max(terminal_x)),
        # the seed of the furthest episode, so it can be replayed for video
        "best_seed": int(max(rows, key=lambda r: (r["flag_get"], r["x_pos_max"]))["seed"]),
        "return_mean":    float(np.mean([r["return_scaled"] for r in rows])),
        "return_raw_mean": float(np.mean([r["return_raw"] for r in rows
                                          if r["return_raw"] is not None] or [0.0])),
        "length_mean":    float(np.mean([r["length"] for r in rows])),
        "route_progress_mean": float(np.mean(rp)) if rp else None,
        # If this is 1, the decorrelation failed for this level and its CI is meaningless.
        "distinct_terminal_x": int(len(set(terminal_x))),
        "death_mix": causes,
        "episodes": rows,
    }


# ── chronological full-game runs ─────────────────────────────────────────────────────
GAME_ENV_ID = "SuperMarioBros-v0"


class GameProgressTracker:
    """Tracks how far a chronological playthrough got.

    Separate from run_full_game so it can be unit-tested: no scripted policy clears 1-1, so
    driving this directly is the only way to check cross-stage tracking without a trained agent.
    flag_get stays true for several frames, so clears are counted on the rising edge.
    """

    def __init__(self, info):
        from mario_env import level_index
        self._index = level_index
        start = f"{info['world']}-{info['stage']}"
        self.best_index = level_index(info["world"], info["stage"])
        self.best_stage = start
        self.visited = {start}
        self.cleared = 0
        self.warps = 0
        self.stages_skipped = 0
        self._prev_flag = False

    def update(self, info):
        label = f"{info['world']}-{info['stage']}"
        self.visited.add(label)
        idx = self._index(info["world"], info["stage"])
        if idx > self.best_index:
            self.best_index, self.best_stage = idx, label
        self.warps = max(self.warps, int(info.get("warps", 0)))
        self.stages_skipped = max(self.stages_skipped, int(info.get("stages_skipped", 0)))
        flag = bool(info.get("flag_get"))
        if flag and not self._prev_flag:
            self.cleared += 1
        self._prev_flag = flag


def run_full_game(agent, *, episodes=10, seed_base=888_000, policy_mode="argmax",
                  max_steps=20_000, log_file=None, device=None, video_dir=None):
    """Play the original game from 1-1, warps allowed, and report how far it got.

    Uses the full-game env, so clearing a stage advances to the next and a warp really skips
    ahead. One episode runs until game over, i.e. one playthrough attempt. "How far" is the
    chronological stage index (1-1 -> 1, 8-4 -> 32), which counts a warp as real progress.
    """
    from base_agent import device as default_device
    device = device or default_device

    runs = []
    for ep in range(episodes):
        env = agent._make_env(levels=[GAME_ENV_ID])
        try:
            model = agent._load_policy(env)
            state, info = env.reset(seed=seed_base + ep)
            state = torch.tensor(state, dtype=torch.float32).to(device)
            track = GameProgressTracker(info)
            total_ret, steps = 0.0, 0

            for steps in range(1, max_steps + 1):
                action, _ = _select_action(model, state, policy_mode, device)
                new_state, reward, terminated, truncated, info = env.step(action)
                total_ret += reward
                track.update(info)
                state = torch.tensor(new_state, dtype=torch.float32).to(device)
                if terminated or truncated:
                    break
        finally:
            env.close()

        runs.append({
            "seed": seed_base + ep,
            "furthest_index": track.best_index, "furthest_stage": track.best_stage,
            "stages_cleared": track.cleared, "stages_visited": len(track.visited),
            "warps": track.warps, "stages_skipped": track.stages_skipped,
            "return_scaled": total_ret, "steps": steps,
        })
        if log_file:
            r = runs[-1]
            log(f"  run {ep:2d}: furthest {r['furthest_stage']:>4s} (index {r['furthest_index']:2d})"
                f"  cleared={r['stages_cleared']}  visited={r['stages_visited']}"
                f"  warps={r['warps']} skipped={r['stages_skipped']}  steps={r['steps']}", log_file)

    idx = [r["furthest_index"] for r in runs]
    best = max(runs, key=lambda r: r["furthest_index"])
    summary = {
        "episodes": len(runs),
        "policy_mode": policy_mode,
        "furthest_index_mean": float(np.mean(idx)),
        "furthest_index_std": float(np.std(idx)),
        "furthest_index_max": int(max(idx)),
        "furthest_stage_mean_equiv": _index_to_stage(float(np.mean(idx))),
        "furthest_stage_max": best["furthest_stage"],
        "stages_cleared_mean": float(np.mean([r["stages_cleared"] for r in runs])),
        "stages_cleared_max": int(max(r["stages_cleared"] for r in runs)),
        "warp_runs": int(sum(1 for r in runs if r["warps"] > 0)),
        "stages_skipped_mean": float(np.mean([r["stages_skipped"] for r in runs])),
        "runs": runs,
    }
    if video_dir:
        # Best and worst run, so the mp4s bracket what the policy actually does.
        for tag, run in (("best", best), ("worst", min(runs, key=lambda r: r["furthest_index"]))):
            try:
                record_run(agent, level=GAME_ENV_ID, seed=run["seed"], out_dir=video_dir,
                           name=f"fullgame_{tag}", policy_mode=policy_mode, max_steps=max_steps)
            except Exception as exc:                       # noqa: BLE001 -- video is optional
                print(f"[video] full-game {tag} run failed: {exc}")

    if log_file:
        log(f"\nfull game over {len(runs)} runs: furthest stage mean "
            f"{summary['furthest_index_mean']:.2f} (~{summary['furthest_stage_mean_equiv']}) "
            f"max {summary['furthest_stage_max']} (index {summary['furthest_index_max']})", log_file)
        log(f"  stages cleared: mean {summary['stages_cleared_mean']:.2f} "
            f"max {summary['stages_cleared_max']};  runs that warped: "
            f"{summary['warp_runs']}/{len(runs)}", log_file)
    return summary


def record_run(agent, *, level, seed, out_dir, name, policy_mode="argmax",
               max_steps=20_000, device=None):
    """Replay one episode with RecordVideo attached and write an mp4.

    Cheap because a seed reproduces a run exactly: the emulator is deterministic and the no-op
    and sticky draws are pure functions of the seed, so we can pick the run we want from the
    summary and re-play precisely that one instead of recording all of them.
    """
    from gymnasium.wrappers import RecordVideo
    from base_agent import device as default_device
    from utils import _rename_latest_video
    device = device or default_device

    os.makedirs(out_dir, exist_ok=True)
    env = RecordVideo(agent._make_env(levels=[level]), out_dir, name_prefix=f"{name}_tmp",
                      episode_trigger=lambda _: True, disable_logger=True)
    try:
        model = agent._load_policy(env)
        state, _ = env.reset(seed=seed)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        for _ in range(max_steps):
            action, _ = _select_action(model, state, policy_mode, device)
            state_np, _, terminated, truncated, info = env.step(action)
            state = torch.tensor(state_np, dtype=torch.float32).to(device)
            if terminated or truncated:
                break
    finally:
        env.close()

    ep = info.get("episode") or {}
    tag = f"{info.get('world', '?')}-{info.get('stage', '?')}"
    label = f"{name}_end{tag}_pages{ep.get('pages', 0)}"
    _rename_latest_video(out_dir, f"{label}.mp4")
    return os.path.join(out_dir, f"{label}.mp4")


def _index_to_stage(index):
    """Chronological index back to a "w-s" label; accepts a fractional mean."""
    i = max(1, int(round(index)))
    world, stage = (i - 1) // 4 + 1, (i - 1) % 4 + 1
    return f"{world}-{stage}"


def save_full_game_chart(summary, path):
    """Per-run furthest stage with the mean marked."""
    runs = summary["runs"]
    fig, ax = plt.subplots(figsize=(max(6, len(runs) * 0.75), 4.2))
    idx = [r["furthest_index"] for r in runs]
    ax.bar(range(len(runs)), idx, color="#4C72B0")
    ax.axhline(summary["furthest_index_mean"], color="#C44E52", ls="--",
               label=f"mean {summary['furthest_index_mean']:.2f} "
                     f"(~{summary['furthest_stage_mean_equiv']})")
    for i, r in enumerate(runs):
        ax.text(i, r["furthest_index"], r["furthest_stage"]
                + ("*" if r["warps"] else ""), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(runs)), [str(i) for i in range(len(runs))])
    ax.set_xlabel("run  (* = used a warp)")
    ax.set_ylabel("furthest stage (chronological index)")
    ax.set_ylim(0, 33)
    ax.set_yticks([1, 5, 9, 13, 17, 21, 25, 29, 32],
                  ["1-1", "2-1", "3-1", "4-1", "5-1", "6-1", "7-1", "8-1", "8-4"])
    ax.set_title("Original game, chronological — warps allowed")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ── aggregation ──────────────────────────────────────────────────────────────────────
def _bootstrap_ci(values, n_boot=10_000, seed=0):
    """Percentile bootstrap CI of the macro-mean over levels."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n_boot, len(arr)))].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def aggregate(per_level, split_name=ML.DEFAULT_SPLIT):
    """Macro-average over levels per tier, with bootstrap CIs.

    Macro (each level equal) rather than pooling episodes: pooling ignores level clustering,
    understates uncertainty and lets one high-variance level dominate. Non-monotone-x levels are
    excluded from progress means but keep their flag rate.
    """
    out = {"split": split_name, "tiers": {}}
    by_tier = {}
    for level, r in per_level.items():
        by_tier.setdefault(r["tier"], []).append(r)

    for tier, rows in sorted(by_tier.items()):
        progress_rows = [r for r in rows if r["level"] not in ML.NON_MONOTONIC_X]
        flags = [r["flag_rate"] for r in rows]
        pages = [r["pages_mean"] for r in progress_rows]
        rp = [r["route_progress_mean"] for r in progress_rows
              if r["route_progress_mean"] is not None]

        out["tiers"][tier] = {
            "n_levels": len(rows),
            "flag_rate_macro": float(np.mean(flags)),
            "flag_rate_ci": _bootstrap_ci(flags),
            "pages_macro": float(np.mean(pages)) if pages else None,
            "pages_ci": _bootstrap_ci(pages),
            "route_progress_macro": float(np.mean(rp)) if rp else None,
            "x_pos_max_macro": float(np.mean([r["x_pos_max_mean"] for r in progress_rows]))
                               if progress_rows else None,
            "warp_rate_macro": float(np.mean([r["warp_rate"] for r in rows])),
            "levels_excluded_from_progress": sorted(
                {r["level"] for r in rows} & set(ML.NON_MONOTONIC_X)),
            # Every episode ended identically — decorrelation failed, so this level's CI is
            # fiction. Needs n > 1 to mean anything.
            "degenerate_levels": sorted(r["level"] for r in rows
                                        if r["n"] > 1 and r["distinct_terminal_x"] <= 1),
        }

    # The headline contrast: memorisation vs generalisation.
    train = out["tiers"].get("train")
    held = out["tiers"].get("tier1")
    if train and held:
        out["generalisation_gap_flag"] = train["flag_rate_macro"] - held["flag_rate_macro"]
        if train["pages_macro"] and held["pages_macro"] is not None:
            out["generalisation_gap_pages"] = train["pages_macro"] - held["pages_macro"]
            out["transfer_ratio_pages"] = (held["pages_macro"] / train["pages_macro"]
                                           if train["pages_macro"] > 0 else None)
    return out


# ── jackknife sensitivity ────────────────────────────────────────────────────────────
def jackknife_tier(per_level, tier="tier1", key="pages_mean"):
    """Recompute a tier's macro-mean dropping one level at a time — how much the conclusion
    depends on which levels landed in the held-out set. More split draws would cost a training
    run each."""
    rows = [r for r in per_level.values()
            if r["tier"] == tier and r["level"] not in ML.NON_MONOTONIC_X]
    vals = [r[key] for r in rows if r[key] is not None]
    if len(vals) < 3:
        return None
    out = []
    for i, dropped in enumerate(rows):
        rest = [v for j, v in enumerate(vals) if j != i]
        out.append({"dropped": dropped["level"], "macro": float(np.mean(rest))})
    macros = [o["macro"] for o in out]
    return {"key": key, "full": float(np.mean(vals)), "min": min(macros), "max": max(macros),
            "range": max(macros) - min(macros), "leave_one_out": out}


# ── figures ──────────────────────────────────────────────────────────────────────────
def save_level_heatmap(per_level, path, *, value_key="flag_rate", split_name=ML.DEFAULT_SPLIT,
                       title=None):
    """8x4 SMB1 grid; Lost Levels go in the tier-2 bar chart instead.

    One sequential hue for the values, with tier shown by the cell edge rather than a second fill
    colour — that would confound the value scale it sits on.
    """
    grid = np.full((8, 4), np.nan)
    for level, r in per_level.items():
        if level not in ML.LEVELS:
            continue
        w, s = ML.parse_level(level)
        val = r.get(value_key)
        if val is not None:
            grid[w - 1, s - 1] = val

    fig, ax = plt.subplots(figsize=(6.4, 9.6))
    data_max = float(np.nanmax(grid)) if np.isfinite(grid).any() else 0.0
    # Rates are always [0,1]; open-ended keys use the data range but never collapse to ~0, or an
    # all-zero result (the expected zero-shot outcome) renders a 1e-9 colourbar.
    vmax = 1.0 if (value_key.endswith("rate") or data_max <= 0) else data_max
    im = ax.imshow(grid, cmap="viridis", vmin=0.0, vmax=vmax, aspect="auto")

    tier_edge = {"train": None, "tier0": "#F5C518", "tier1": "#E4572E", "excluded": "#8A8A8A"}
    for w in range(8):
        for s in range(4):
            level = ML.level_str(w + 1, s + 1)
            r = per_level.get(level)
            val = grid[w, s]
            if np.isnan(val):
                ax.add_patch(plt.Rectangle((s - .5, w - .5), 1, 1, hatch="///",
                                           fill=False, edgecolor="#BBBBBB", linewidth=0.5))
                label = "n/a"
            else:
                label = f"{val:.2f}"
            tier = (r or {}).get("tier") or ML.tier_of(level, split_name) or "unassigned"
            edge = tier_edge.get(tier)
            if edge:
                ax.add_patch(plt.Rectangle((s - .5, w - .5), 1, 1, fill=False,
                                           edgecolor=edge, linewidth=2.5))
            shade = "white" if (np.isnan(val) or val < vmax * 0.55) else "black"
            ax.text(s, w, f"{level}\n{label}", ha="center", va="center", fontsize=8, color=shade)

    ax.set_xticks(range(4), [f"stage {s}" for s in range(1, 5)])
    ax.set_yticks(range(8), [f"world {w}" for w in range(1, 9)])
    ax.set_title(title or f"SMB1 per-level {value_key}", fontsize=11)
    handles = [plt.Line2D([], [], color=c, lw=2.5, label=t)
               for t, c in tier_edge.items() if c]
    handles.append(plt.Line2D([], [], color="#BBBBBB", lw=0.8, label="not evaluated"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=4,
              frameon=False, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_key)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def save_tier_bars(agg, path, title=None):
    """Per-tier macro flag rate and pages with bootstrap CIs — the headline figure."""
    tiers = [t for t in ML.TIERS if t in agg["tiers"]]
    if not tiers:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, ci_key, label) in zip(axes, [
            ("flag_rate_macro", "flag_rate_ci", "Completion rate (macro over levels)"),
            ("pages_macro", "pages_ci", "Pages cleared (macro over levels)")]):
        vals, los, his = [], [], []
        for t in tiers:
            v = agg["tiers"][t].get(key)
            vals.append(0.0 if v is None else v)
            lo, hi = agg["tiers"][t].get(ci_key) or (None, None)
            los.append(0.0 if lo is None else max(0.0, vals[-1] - lo))
            his.append(0.0 if hi is None else max(0.0, hi - vals[-1]))
        ax.bar(tiers, vals, yerr=[los, his], capsize=4,
               color=["#4C72B0", "#F5C518", "#E4572E", "#55A868"][:len(tiers)])
        ax.set_ylabel(label)
        ax.set_xlabel("tier")
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(title or "Mario generalisation by tier (95% bootstrap CI over levels)")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def save_tier2_bars(per_level, path):
    """Tier-2 per-stage pages — Lost Levels do not fit the 8x4 SMB1 grid."""
    rows = [r for r in per_level.values() if r["tier"] == "tier2"]
    if not rows:
        return
    rows.sort(key=lambda r: r["level"])
    names = [r["level"].replace("SuperMarioBros2-", "LL ").replace("-v0", "") for r in rows]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.8), 4))
    ax.bar(names, [r["pages_mean"] for r in rows], color="#55A868")
    for i, r in enumerate(rows):
        if r["flag_rate"] > 0:
            ax.text(i, r["pages_mean"], f"flag {r['flag_rate']:.0%}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Pages cleared (mean)")
    ax.set_title("Tier 2 — Lost Levels (out-of-distribution)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ── driver ───────────────────────────────────────────────────────────────────────────
def _write_csv(per_level, path):
    fields = ("level", "tier", "archetype", "n", "flag_rate", "pages_mean", "pages_max",
              "x_pos_max_mean", "x_pos_max_best", "route_progress_mean", "return_mean",
              "return_raw_mean", "length_mean", "warp_rate", "distinct_terminal_x", "death_mix")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for level in sorted(per_level, key=lambda l: (l not in ML.LEVELS, l)):
            row = dict(per_level[level])
            row["death_mix"] = ";".join(f"{k}={v}" for k, v in sorted(row["death_mix"].items()))
            w.writerow(row)


def _one_level_per_world(per_level):
    """The lowest-numbered evaluated stage of each SMB1 world.

    Lowest rather than best-performing: a per-world best would pick different stages for
    different algorithms, and the clips would no longer be comparable. Lost Levels ids carry no
    SMB1 world and are skipped — request them explicitly if wanted.
    """
    by_world = {}
    for level in per_level:
        try:
            world, stage = ML.parse_level(level)
        except Exception:                                  # noqa: BLE001 -- non-SMB1 level id
            continue
        if world not in by_world or stage < by_world[world][0]:
            by_world[world] = (stage, level)
    return tuple(lvl for _, (_, lvl) in sorted(by_world.items()))


def run(agent, *, split_name=None, tiers=("train", "tier0", "tier1", "tier2"),
        episodes_per_level=30, policy_mode="argmax", out_dir=None, levels=None,
        full_game_episodes=10, video_levels="per_world", full_game_video=True):
    """Evaluate the requested tiers, write JSON/CSV/figures, return the aggregate.

    Also plays full_game_episodes chronological runs of the original game (0 to skip).
    """
    emp = agent.env_make_params or {}
    split_name = split_name or emp.get("level_split") or ML.DEFAULT_SPLIT
    out_dir = out_dir or os.path.join(agent.RUN_DIR, "evaluation", "mario")
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "evaluation.log")
    video_dir = os.path.join(out_dir, "videos")

    if levels:
        todo = list(levels)
    else:
        todo = []
        for tier in tiers:
            todo.extend(ML.split_levels(split_name, tier))
    if not todo:
        raise ValueError(f"no levels selected (split={split_name}, tiers={tiers})")

    route_table = _load_route_table()
    log(f"Per-level evaluation — split={split_name} tiers={list(tiers)} "
        f"levels={len(todo)} episodes/level={episodes_per_level} policy={policy_mode}",
        log_file, mode='w')
    log(f"env: {describe_env_params(emp, env_id=agent.env_id)}", log_file)
    if route_table is None:
        log("NOTE: no frozen route table (mario_levels.json) — reporting `pages` and raw x_pos_max "
            "instead of normalised Route Progress. See mario.md 5.3.", log_file)

    per_level = evaluate_levels(agent, todo, episodes_per_level=episodes_per_level,
                                policy_mode=policy_mode, route_table=route_table,
                                log_file=log_file)
    agg = aggregate(per_level, split_name)
    agg["jackknife_tier1_pages"] = jackknife_tier(per_level, "tier1", "pages_mean")

    # How far it actually gets in the real game, which the per-level tiers cannot show.
    if full_game_episodes:
        log(f"\nOriginal game, chronological ({full_game_episodes} runs, warps allowed)", log_file)
        agg["full_game"] = run_full_game(agent, episodes=full_game_episodes,
                                         policy_mode=policy_mode, log_file=log_file,
                                         video_dir=video_dir if full_game_video else None)
        save_full_game_chart(agg["full_game"],
                             os.path.join(out_dir, f"full_game{'' if policy_mode == 'argmax' else '_' + policy_mode}.png"))
    agg["policy_mode"] = policy_mode
    agg["episodes_per_level"] = episodes_per_level
    # so a Rainbow-vs-PPO comparison stays auditable after the fact
    agg["env"] = describe_env_params(emp, env_id=agent.env_id)
    agg["checkpoint"] = agent.MODEL_FILE
    agg["seed"] = agent.seed

    suffix = "" if policy_mode == "argmax" else f"_{policy_mode}"
    with open(os.path.join(out_dir, f"per_level{suffix}.json"), "w") as f:
        json.dump(per_level, f, indent=2, default=str)
    with open(os.path.join(out_dir, f"summary{suffix}.json"), "w") as f:
        json.dump(agg, f, indent=2, default=str)
    _write_csv(per_level, os.path.join(out_dir, f"per_level{suffix}.csv"))

    save_level_heatmap(per_level, os.path.join(out_dir, f"heatmap_flag_rate{suffix}.png"),
                       value_key="flag_rate", split_name=split_name,
                       title=f"Completion rate per level ({policy_mode})")
    save_level_heatmap(per_level, os.path.join(out_dir, f"heatmap_pages{suffix}.png"),
                       value_key="pages_mean", split_name=split_name,
                       title=f"Pages cleared per level ({policy_mode})")
    save_tier_bars(agg, os.path.join(out_dir, f"tier_summary{suffix}.png"))
    save_tier2_bars(per_level, os.path.join(out_dir, f"tier2_lost_levels{suffix}.png"))

    # One mp4 per requested level, replaying its best episode (a seed reproduces a run exactly).
    if video_levels == "per_world":
        video_levels = _one_level_per_world(per_level)
        log(f"videos: one per world -> {list(video_levels)}", log_file)
    for level in (video_levels or ()):
        if level not in per_level:
            continue
        try:
            best_seed = per_level[level]["best_seed"]
            record_run(agent, level=level, seed=best_seed, out_dir=video_dir,
                       name=f"level_{level}", policy_mode=policy_mode)
        except Exception as exc:                           # noqa: BLE001 -- video is optional
            print(f"[video] level {level} failed: {exc}")

    log("", log_file)
    for tier, t in agg["tiers"].items():
        log(f"{tier:9s} n={t['n_levels']:2d}  flag={t['flag_rate_macro']:.3f} "
            f"pages={t['pages_macro'] if t['pages_macro'] is None else round(t['pages_macro'], 2)}"
            f"  warp={t['warp_rate_macro']:.3f}"
            + (f"  DEGENERATE(no eval variance): {t['degenerate_levels']}"
               if t["degenerate_levels"] else ""), log_file)
    if "generalisation_gap_flag" in agg:
        log(f"generalisation gap (train - tier1): flag={agg['generalisation_gap_flag']:.3f}"
            + (f" pages={agg['generalisation_gap_pages']:.2f}"
               if "generalisation_gap_pages" in agg else ""), log_file)
    log(f"\nwritten to {out_dir}", log_file)
    return agg
