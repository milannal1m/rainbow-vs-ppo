"""The per-env secondary metric (pipes vs pages), its axis label and its frame rate.

The shared training/eval code used to hardcode FlappyBird specifics in four places: the
`reward >= 1.0` pipe counter, a 30 FPS assumption, the axis label, and the Grad-CAM action names.

Derived from env_id rather than a new YAML key, so a FlappyBird config cannot accidentally get the
Mario metric and a Mario config cannot forget to set one. For FlappyBird every value here matches
the old inline code, and metrics.json keeps eval_pipes_mean.
"""
from env_factory import FLAPPYBIRD, MARIO, env_kind

FLAPPYBIRD_FPS = 30
NES_FPS = 60


class EpisodeMetric:
    """Accumulates one episode's secondary metric. One instance per episode."""

    def update(self, reward, info):
        raise NotImplementedError

    def value(self):
        """The scalar plotted on the training curve and averaged in metrics.json."""
        raise NotImplementedError

    def extras(self):
        """Per-episode side facts, aggregated by MetricSpec.aggregate."""
        return {}


class PipeCounterMetric(EpisodeMetric):
    """FlappyBird: count steps that scored a pipe (the previous inline `reward >= 1.0`)."""

    def __init__(self):
        self.pipes = 0

    def update(self, reward, info):
        if reward >= 1.0:
            self.pipes += 1

    def value(self):
        return self.pipes


class MarioProgressMetric(EpisodeMetric):
    """Mario: pages cleared (256 px = one NES screen), plus the per-episode facts.

    Pages rather than flag rate, which reads flat at 0 for a long time and makes the training
    curve useless; and rather than a normalised fraction, which would need a per-level length
    constant we deliberately do not have (see mario.md). Derived from covered_px, which stays
    meaningful across area transitions where x_pos_max does not.
    """

    _FACT_KEYS = (
        "level", "world", "stage", "return_raw", "x_pos_max", "covered_px", "pages",
        "max_area", "areas", "flag_get", "warped", "death_cause", "time_left",
        "coins", "score", "stalled_for", "warps", "stages_skipped",
        # per-area spans, so progress can be re-normalised later if a route table ever appears
        "area_span",
    )

    def __init__(self):
        self._episode = None
        # fallback for episodes the training loop cuts short, where info["episode"] never arrives
        self._live_covered = 0
        self._live_x_max = 0
        self._live_flag = False
        self._live_info = {}

    def update(self, reward, info):
        self._live_info = info
        area_max = info.get("area_max")
        if area_max:
            self._live_x_max = max(self._live_x_max, max(area_max.values()))
        self._live_covered = max(self._live_covered, info.get("covered_px", self._live_covered))
        if info.get("flag_get"):
            self._live_flag = True
        if "episode" in info:
            self._episode = info["episode"]

    def value(self):
        if self._episode is not None:
            return float(self._episode.get("pages", 0))
        return float(self._live_covered // 256)

    def extras(self):
        if self._episode is not None:
            return {k: self._episode.get(k) for k in self._FACT_KEYS}
        info = self._live_info
        return {
            "level": info.get("level"), "world": info.get("world"), "stage": info.get("stage"),
            "return_raw": None, "x_pos_max": self._live_x_max,
            "covered_px": self._live_covered, "pages": self._live_covered // 256,
            "max_area": info.get("area"), "areas": None, "flag_get": self._live_flag,
            "warped": bool(info.get("warped")), "death_cause": "cut_short",
            "time_left": info.get("time"), "coins": info.get("coins"),
            "score": info.get("score"), "stalled_for": None,
        }


class MetricSpec:
    """How one env family reports its secondary metric."""

    def __init__(self, key, label, fps, factory, aggregator=None):
        self.key = key           # -> metrics.json key f"eval_{key}_mean"
        self.label = label       # -> graph axis label
        self.fps = fps           # -> episode length in seconds
        self._factory = factory
        self._aggregator = aggregator

    def new(self):
        return self._factory()

    def aggregate(self, extras_list):
        if self._aggregator is None:
            return {}
        return self._aggregator([e for e in extras_list if e])

    def __repr__(self):
        return f"MetricSpec(key={self.key!r}, label={self.label!r}, fps={self.fps})"


def _mean(values):
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate_mario(extras_list):
    if not extras_list:
        return {}
    n = len(extras_list)
    out = {
        "eval_flag_rate":       _mean([1.0 if e.get("flag_get") else 0.0 for e in extras_list]),
        "eval_warp_rate":       _mean([1.0 if e.get("warped") else 0.0 for e in extras_list]),
        "eval_stages_skipped_mean": _mean([e.get("stages_skipped") for e in extras_list]),
        "eval_x_pos_max_mean":  _mean([e.get("x_pos_max") for e in extras_list]),
        "eval_covered_px_mean": _mean([e.get("covered_px") for e in extras_list]),
        "eval_return_raw_mean": _mean([e.get("return_raw") for e in extras_list]),
    }
    # The one diagnostic that survives a 0%-success eval: what actually killed it.
    causes = {}
    for e in extras_list:
        cause = e.get("death_cause") or "unknown"
        causes[cause] = causes.get(cause, 0) + 1
    for cause, count in sorted(causes.items()):
        out[f"eval_death_{cause}_frac"] = count / n
    return out


def make_metric_spec(hyperparams):
    """Build the MetricSpec for a config, from its env_id (and frame_skip for Mario)."""
    env_id = hyperparams.get("env_id", "")
    emp = hyperparams.get("env_make_params") or {}
    kind = env_kind(env_id, emp.get("env_kind"))

    if kind == MARIO:
        frame_skip = int(emp.get("frame_skip", 4) or 1)
        return MetricSpec(
            key="pages",
            label="Pages cleared",
            # One agent step is `frame_skip` NES frames at 60 Hz.
            fps=max(1, NES_FPS // frame_skip),
            factory=MarioProgressMetric,
            aggregator=_aggregate_mario,
        )

    # FlappyBird and generic envs keep the historical behaviour and key names.
    return MetricSpec(
        key="pipes",
        label="Pipes passed",
        fps=FLAPPYBIRD_FPS,
        factory=PipeCounterMetric,
    )


def action_labels_for(hyperparams, num_actions):
    """Action names for the Grad-CAM panels — 'flap'/'no-flap' is wrong on all 12 Mario actions."""
    env_id = hyperparams.get("env_id", "")
    emp = hyperparams.get("env_make_params") or {}
    kind = env_kind(env_id, emp.get("env_kind"))

    if kind == MARIO:
        from mario_env import action_labels
        labels = action_labels(emp.get("action_set", "COMPLEX_MOVEMENT"))
        if len(labels) == num_actions:
            return labels
    elif kind == FLAPPYBIRD and num_actions == 2:
        return ["no-flap", "flap"]

    return [f"a{i}" for i in range(num_actions)]
