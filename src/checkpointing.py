"""Resume-from-checkpoint and durable per-episode metrics.

Mario runs do not fit one SLURM job (10M steps is ~26 h against a 24 h wall clock), and before
this a killed job lost everything: only model.state_dict() was written and the metric lists lived
in RAM.

The replay buffer is deliberately NOT checkpointed — 300k transitions is ~17 GB, too much to write
every 15 minutes. A resumed Rainbow run refills from scratch and resume_refill_steps gates learning
until it has; the discontinuity is logged rather than hidden.
"""
import csv
import os
import random

import numpy as np
import torch

# Time-based rather than episode-based: CHECKPOINT_EVERY = 1000 episodes is far too coarse for
# Mario, where one episode can run thousands of steps.
CHECKPOINT_STATE_SECONDS = 900

# The loops append one value per env step, so a 15M-step run would hold three 15M-element lists
# (~1.3 GB) and re-cumsum them every GRAPH_UPDATE_SECONDS. Striding keeps the plots faithful and
# the memory bounded; configs without max_env_steps get stride 1, i.e. the previous behaviour.
METRIC_SERIES_CAP = 200_000


def metric_stride(max_env_steps):
    """Append only every Nth per-step metric sample, so the series stays bounded."""
    if not max_env_steps:
        return 1
    return max(1, int(max_env_steps) // METRIC_SERIES_CAP)


# ── full training state ──────────────────────────────────────────────────────────────
def _rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng(state):
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"])
                            else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # noqa: BLE001 -- a bad RNG restore must not kill a resume
        print(f"[resume] could not restore RNG state ({exc}); continuing with fresh streams")


def save_run_state(path, *, model, optimizer, counters, extra=None):
    """Write the full training state atomically — the SLURM wall clock can kill a job mid-write,
    and a half-written checkpoint replacing a good one would lose the run."""
    payload = {
        "version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "counters": dict(counters),
        "rng": _rng_state(),
        "extra": dict(extra or {}),
    }
    tmp = f"{path}.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_run_state(path, *, model, optimizer, map_location=None):
    """Restore model/optimizer/RNG in place and return the counters dict (None if absent)."""
    if not os.path.exists(path):
        return None
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    _restore_rng(payload.get("rng"))
    counters = dict(payload.get("counters", {}))
    counters["_extra"] = payload.get("extra", {})
    return counters



# ── replay buffer ────────────────────────────────────────────────────────────────────
# Written ONCE, when the wall clock is about to end the job -- never on the periodic cadence.
# 400k Mario transitions are ~23 GB (1M FlappyBird ones ~51 GB): far too much every 15 minutes,
# but ~1 min as a one-off, against the 600 s warning SLURM's --signal gives. Without it a resumed
# Rainbow run starts from an empty buffer and spends ~1.4 h refilling it.

def save_replay_buffer(path, memory, nstep_buf=None):
    """Write the buffer beside the run state. Returns bytes written, or 0 on failure."""
    if not hasattr(memory, "state_dict"):
        return 0
    tmp = f"{path}.tmp"
    try:
        torch.save({"version": 1, "memory": memory.state_dict(),
                    "nstep": nstep_buf.state_dict() if nstep_buf is not None else None}, tmp)
        os.replace(tmp, path)
        return os.path.getsize(path)
    except Exception as exc:  # noqa: BLE001 -- a failed buffer save must not lose the run
        print(f"[checkpoint] replay buffer save failed: {exc}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return 0


def load_replay_buffer(path, memory, nstep_buf=None, map_location=None):
    """Restore the buffer in place. Returns the transition count, or 0 if unavailable.

    A missing or unreadable file is not fatal: the run then refills from scratch, which is the
    behaviour that existed before this.
    """
    if not os.path.exists(path) or not hasattr(memory, "load_state_dict"):
        return 0
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        memory.load_state_dict(payload["memory"])
        if nstep_buf is not None and payload.get("nstep"):
            nstep_buf.load_state_dict(payload["nstep"])
        return len(memory)
    except Exception as exc:  # noqa: BLE001
        print(f"[resume] replay buffer restore failed ({exc}); refilling from scratch")
        return 0


# ── per-episode CSV ──────────────────────────────────────────────────────────────────
class EpisodeCSVLogger:
    """Append-only per-episode log, flushed every row so a kill -9 loses at most one episode."""

    # Superset of columns across both games; FlappyBird leaves the Mario ones empty.
    FIELDS = (
        "episode", "total_steps", "wall_s", "algo", "level",
        "return_scaled", "return_raw", "length", "secondary",
        "x_pos_max", "covered_px", "pages", "flag_get", "death_cause",
        "warped", "time_left", "coins", "score", "areas", "max_area",
    )

    def __init__(self, path, fields=None, resume=False):
        self.path = path
        self.fields = tuple(fields or self.FIELDS)
        # header only on a new file — a resume appends
        write_header = not (resume and os.path.exists(path))
        self._fh = open(path, "a" if resume else "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.fields, extrasaction="ignore")
        if write_header:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, **row):
        self._writer.writerow(row)
        self._fh.flush()

    def close(self):
        try:
            self._fh.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def episode_row(*, episode, total_steps, wall_s, algo, return_scaled, length, secondary,
                extras=None):
    """One CSV row from the loop counters plus the metric's per-episode extras."""
    row = {
        "episode": episode, "total_steps": total_steps, "wall_s": round(wall_s, 1),
        "algo": algo, "return_scaled": round(float(return_scaled), 4),
        "length": length, "secondary": secondary,
    }
    for key in ("level", "return_raw", "x_pos_max", "covered_px", "pages", "flag_get",
                "death_cause", "warped", "time_left", "coins", "score", "areas", "max_area"):
        if extras and key in extras:
            row[key] = extras[key]
    return row
