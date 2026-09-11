"""Mario level inventory, archetypes and the frozen train/eval splits.

Stdlib only, because env_factory imports it under both conda envs while gym_super_mario_bros
exists in only one of them.
"""
import os
from dataclasses import dataclass

# ── Archetypes ───────────────────────────────────────────────────────────────────────
# The canonical SMB1 totals, asserted at import as a checksum on the per-level table below.
GROUND      = "ground"
ATHLETIC    = "athletic"
UNDERGROUND = "underground"
UNDERWATER  = "underwater"
CASTLE      = "castle"

ARCHETYPE_COUNTS = {GROUND: 13, ATHLETIC: 7, UNDERGROUND: 2, UNDERWATER: 2, CASTLE: 8}

# Palette is a world-level property; worlds 3 and 6 are entirely night. Holding out either would
# remove the night palette from training, making the eval test palette rather than layout.
NIGHT_WORLDS = frozenset({3, 6})


@dataclass(frozen=True)
class LevelInfo:
    """Static facts about one SMB1 stage."""
    level: str            # "3-2"
    world: int
    stage: int
    archetype: str
    twin: str | None      # shares ROM level data with this stage (see TWIN_PAIRS)
    warp_zone: bool       # contains a warp zone -> can silently leave the target stage
    maze: bool            # routing maze: x_pos is non-monotone in skill

    @property
    def night(self):
        return self.world in NIGHT_WORLDS

    @property
    def multi_area(self):
        """Advances through a forward area transition, which resets x_pos (see MarioAreaRebase)."""
        return self.level in MULTI_AREA_LEVELS


# ── Layout reuse (the leakage hazard) ────────────────────────────────────────────────
# SMB1 ships several stages twice, the later one reusing the same ROM data with a harder enemy
# set. Straddling a twin pair measures "same layout, different palette" while claiming to measure
# layout generalisation, so SPLITS keeps twins together -- except the two pairs deliberately
# straddled as the tier-0 control. Source: chridd.nfshost.com/smb-reuse
TWIN_PAIRS = (
    ("1-3", "5-3"),
    ("1-4", "6-4"),
    ("2-2", "7-2"),
    ("2-3", "7-3"),
    ("2-4", "4-4"),
)
_TWIN_OF = {a: b for a, b in TWIN_PAIRS} | {b: a for a, b in TWIN_PAIRS}

# Lost Levels C-3 / C-4 are modified copies of 7-3 / 7-4, so C-3 doubles as an OOD twin control.
LOST_LEVELS_TWINS = {"C-3": "7-3", "C-4": "7-4"}

# The only two stages with warp zones.
WARP_LEVELS = frozenset({"1-2", "4-2"})

# Warps are enabled and rewarded, so training on 1-2 or 4-2 can walk into a held-out stage. Every
# warp zone exits to some world's first stage, so treating all X-1 as destinations is conservative
# and needs no per-zone table.
WARP_DESTINATIONS = frozenset(f"{world}-1" for world in range(1, 9))

# Routing mazes: a wrong route loops back, so x_pos is non-monotone in skill and progress
# saturates at the first junction for any memoryless policy. 8-4 also needs specific pipe
# descents and the real Bowser. Excluded from train and from all aggregates.
MAZE_LEVELS = frozenset({"4-4", "7-4", "8-4"})

# Same set, named for the property the metric layer cares about, so aggregation reads as
# "exclude non-monotone levels from progress means" (they still contribute completion rate).
NON_MONOTONIC_X = MAZE_LEVELS

# `area` is a world-cumulative counter, not a per-level sub-area index, and every stage starts at
# its primary area with x_pos == 40 — so the env drops Mario straight into 1-2's underground part.
START_AREA = {
    "1-1": 1, "1-2": 3, "1-3": 4, "1-4": 5,
    "2-1": 1, "2-2": 3, "2-3": 4, "2-4": 5,
    "3-1": 1, "3-2": 2, "3-3": 3, "3-4": 4,
    "4-1": 1, "4-2": 3, "4-3": 4, "4-4": 5,
    "5-1": 1, "5-2": 2, "5-3": 3, "5-4": 4,
    "6-1": 1, "6-2": 2, "6-3": 3, "6-4": 4,
    "7-1": 1, "7-2": 3, "7-3": 4, "7-4": 5,
    "8-1": 1, "8-2": 2, "8-3": 3, "8-4": 4,
}

# Stages spanning two areas, so a forward pipe resets x_pos mid-level (see MarioAreaRebase).
# Derived from START_AREA, not guessed: primary areas run sequentially within a world, so the gap
# in worlds 1/2/4/7 is that world's stage 2 owning an extra (above-ground intro) area. 8-4 is
# added by hand — its maze sub-areas sit above its start area, so no gap reveals them.
# The transition lands on the final stretch to the flagpole, not early in the level.
MULTI_AREA_LEVELS = frozenset({"1-2", "2-2", "4-2", "7-2", "8-4"})

# ── The inventory ────────────────────────────────────────────────────────────────────
# Archetype labels were checked against a frame from each stage; the totals are asserted below.
_ROWS = (
    # level, archetype
    ("1-1", GROUND),      ("1-2", UNDERGROUND), ("1-3", ATHLETIC),    ("1-4", CASTLE),
    ("2-1", GROUND),      ("2-2", UNDERWATER),  ("2-3", ATHLETIC),    ("2-4", CASTLE),
    ("3-1", GROUND),      ("3-2", GROUND),      ("3-3", ATHLETIC),    ("3-4", CASTLE),
    ("4-1", GROUND),      ("4-2", UNDERGROUND), ("4-3", ATHLETIC),    ("4-4", CASTLE),
    ("5-1", GROUND),      ("5-2", GROUND),      ("5-3", ATHLETIC),    ("5-4", CASTLE),
    ("6-1", GROUND),      ("6-2", GROUND),      ("6-3", ATHLETIC),    ("6-4", CASTLE),
    ("7-1", GROUND),      ("7-2", UNDERWATER),  ("7-3", ATHLETIC),    ("7-4", CASTLE),
    ("8-1", GROUND),      ("8-2", GROUND),      ("8-3", GROUND),      ("8-4", CASTLE),
)


def parse_level(level):
    """"3-2" -> (3, 2). Accepts the Lost Levels letter worlds ("A-1" -> (10, 1))."""
    world_s, stage_s = level.split("-")
    world = int(world_s) if world_s.isdigit() else ord(world_s.upper()) - ord("A") + 10
    return world, int(stage_s)


def level_str(world, stage):
    """(3, 2) -> "3-2". Worlds >= 10 use the Lost Levels letter labels."""
    label = str(world) if world <= 9 else chr(ord("A") + world - 10)
    return f"{label}-{stage}"


LEVELS = {}
for _level, _archetype in _ROWS:
    _w, _s = parse_level(_level)
    LEVELS[_level] = LevelInfo(
        level=_level, world=_w, stage=_s, archetype=_archetype,
        twin=_TWIN_OF.get(_level),
        warp_zone=_level in WARP_LEVELS,
        maze=_level in MAZE_LEVELS,
    )

SMB1_LEVELS = tuple(level for level, _ in _ROWS)


# ── Env ids ──────────────────────────────────────────────────────────────────────────
def env_id_for(level, lost_levels=False, version="v0"):
    """"3-2" -> "SuperMarioBros-3-2-v0" (or "SuperMarioBros2-3-2-v0" for Lost Levels)."""
    game = "SuperMarioBros2" if lost_levels else "SuperMarioBros"
    return f"{game}-{level}-{version}"


def is_lost_levels(level_or_id):
    return level_or_id.startswith("SuperMarioBros2-")


# ── The splits ───────────────────────────────────────────────────────────────────────
# Explicit lists, not a seeded shuffle: no random draw satisfies all three constraints at once --
# twins must not straddle train/eval (except the deliberate tier-0 pair), every eval archetype
# needs a training example, and eval difficulty must overlap training rather than sit above it.
# A frozen list also goes into the report verbatim and cannot drift.
#
#   train  memorisation ceiling — did it learn at all?
#   tier0  positive control: layout twins of trained stages. If tier1/tier2 are 0% everywhere,
#          this is what separates a real negative result from a broken eval harness.
#   tier1  generalisation to novel layouts
#   tier2  out-of-distribution: novel layouts and novel mechanics
#
# protocol: True marks splits results are claimed from; those get the full discipline enforced at
# import. The debug/hpo splits are proxies and are exempt from the scientific constraints.
SPLITS = {
    "smb1_stage_holdout": {
        "protocol": True,
        "train": (
            "1-1", "1-2", "1-3", "1-4", "2-1", "2-2", "2-3", "3-1", "3-3", "3-4",
            "4-1", "5-1", "5-4", "6-1", "6-2", "6-4", "7-1", "7-3", "8-1", "8-3",
        ),
        # 5-3 is the harder variant of trained 1-3; 7-2 of trained 2-2.
        "tier0": ("5-3", "7-2"),
        # No X-1 stage here: those are warp destinations, and warps are enabled, so holding one out
        # would let a warp from trained 1-2 walk straight into the held-out set. 5-2 and 8-2 stand
        # in for what were 2-1 and 8-1.
        "tier1": ("2-4", "3-2", "4-2", "4-3", "5-2", "6-3", "8-2"),
        "tier2": (
            # OOD-near (Lost Levels worlds 1-4)
            "SuperMarioBros2-1-1-v0", "SuperMarioBros2-1-3-v0", "SuperMarioBros2-2-1-v0",
            "SuperMarioBros2-3-1-v0", "SuperMarioBros2-4-1-v0", "SuperMarioBros2-4-3-v0",
            # OOD-far (Lost Levels worlds 5-8)
            "SuperMarioBros2-5-1-v0", "SuperMarioBros2-6-1-v0", "SuperMarioBros2-7-1-v0",
            "SuperMarioBros2-8-1-v0",
            # OOD twin control (~ trained 7-3) and an extreme
            "SuperMarioBros2-C-3-v0", "SuperMarioBros2-A-1-v0",
        ),
        "excluded": tuple(sorted(MAZE_LEVELS)),
    },
    # Pre-registered compute-constrained fallback. Same twin discipline: 1-3 trains, 5-3 is the
    # tier0 control; no other twin pair straddles.
    "smb1_small": {
        "protocol": True,
        "train": ("1-1", "1-3", "2-1", "3-1", "4-1", "5-1"),
        "tier0": ("5-3",),
        # again no X-1 stage held out, for the warp-leakage reason above
        "tier1": ("2-3", "5-2", "6-3", "8-2"),
        "tier2": ("SuperMarioBros2-1-1-v0", "SuperMarioBros2-4-1-v0"),
        "excluded": tuple(sorted(MAZE_LEVELS)),
    },
    # Cheap proxy for Optuna trials. Selection may ONLY ever read training levels, so the eval
    # tiers here exist purely so the harness has something to report; no trial may score on them.
    "smb1_hpo": {
        "train": ("1-1", "2-1", "3-1", "4-1"),
        "tier0": (),
        "tier1": ("5-1",),
        "tier2": (),
        "excluded": tuple(sorted(MAZE_LEVELS)),
    },
    "smb1_debug": {
        "train": ("1-1",),
        "tier0": (),
        "tier1": ("1-2",),
        "tier2": (),
        "excluded": tuple(sorted(MAZE_LEVELS)),
    },
}

DEFAULT_SPLIT = "smb1_stage_holdout"
TIERS = ("train", "tier0", "tier1", "tier2")


def get_split(name=DEFAULT_SPLIT):
    """Return the split dict for `name`; raises KeyError with the valid names listed."""
    try:
        return SPLITS[name]
    except KeyError:
        raise KeyError(
            f"unknown level split {name!r}; known splits: {sorted(SPLITS)}"
        ) from None


def split_levels(name=DEFAULT_SPLIT, tier="train"):
    """Levels for one tier of a split. `tier='all'` concatenates train+tier0+tier1+tier2."""
    split = get_split(name)
    if tier == "all":
        out = []
        for t in TIERS:
            out.extend(split.get(t, ()))
        return tuple(out)
    if tier not in TIERS + ("excluded",):
        raise KeyError(f"unknown tier {tier!r}; known tiers: {TIERS + ('all', 'excluded')}")
    return tuple(split.get(tier, ()))


def tier_of(level, name=DEFAULT_SPLIT):
    """Which tier a level belongs to in a given split, or None if it is not in the split."""
    split = get_split(name)
    for tier in TIERS + ("excluded",):
        if level in split.get(tier, ()):
            return tier
    return None


def levels_from_config(env_make_params):
    """Resolve the level list a Mario env should sample from.

    Resolution order, most explicit first:
      1. `levels`: an explicit list (or a single "1-1" string, or "1-1,1-2")
      2. `level_split` + `level_set`  (level_set defaults to "train")
      3. the default split's train tier

    Returns (levels, provenance) where provenance is a short string for the run log, so a
    training run always records WHERE its level list came from.
    """
    p = env_make_params or {}

    explicit = p.get("levels")
    if explicit:
        if isinstance(explicit, str):
            explicit = [s.strip() for s in explicit.split(",") if s.strip()]
        return tuple(explicit), f"explicit ({len(explicit)} levels)"

    split_name = p.get("level_split") or DEFAULT_SPLIT
    level_set = p.get("level_set", "train")
    levels = split_levels(split_name, level_set)
    if not levels:
        raise ValueError(f"split {split_name!r} tier {level_set!r} is empty")
    return levels, f"{split_name}/{level_set} ({len(levels)} levels)"


# ── Route table (progress normalisation) ─────────────────────────────────────────────
# progress_max is raw pixels and x_pos is per-area, so a normalised progress metric would need a
# per-stage reference route. Not measured (see mario.md); mario_eval reads this file if it appears,
# and reports pages otherwise. It must not be derived from agent runs — that would be circular.
# Anchored to this file rather than the cwd, so the table is found whatever directory
# python was invoked from.
ROUTE_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "mario_levels.json")
PAGE_WIDTH = 256  # one NES screen; `pages_cleared` is the normalisation-free fallback metric


def _validate():
    """Import-time checks: cheap, and they catch table edits that break the protocol."""
    assert len(SMB1_LEVELS) == 32, f"expected 32 SMB1 stages, got {len(SMB1_LEVELS)}"

    counts = {}
    for info in LEVELS.values():
        counts[info.archetype] = counts.get(info.archetype, 0) + 1
    assert counts == ARCHETYPE_COUNTS, (
        f"archetype totals {counts} != canonical {ARCHETYPE_COUNTS}; the per-level table is wrong"
    )

    for a, b in TWIN_PAIRS:
        assert LEVELS[a].twin == b and LEVELS[b].twin == a, f"twin link broken for {a}/{b}"

    for name, split in SPLITS.items():
        assigned = [lvl for tier in TIERS + ("excluded",) for lvl in split.get(tier, ())]
        dupes = {lvl for lvl in assigned if assigned.count(lvl) > 1}
        assert not dupes, f"split {name}: {sorted(dupes)} assigned to more than one tier"

        smb1 = [lvl for lvl in assigned if not is_lost_levels(lvl)]
        unknown = [lvl for lvl in smb1 if lvl not in LEVELS]
        assert not unknown, f"split {name}: unknown SMB1 levels {unknown}"

        # No maze stage may be trained on or evaluated -- they are appendix-only.
        for tier in TIERS:
            leaked = MAZE_LEVELS & set(split.get(tier, ()))
            assert not leaked, f"split {name}: maze stage(s) {sorted(leaked)} in tier {tier}"

        # The remaining checks constrain the SCIENCE, so they only apply to splits that results
        # are claimed from. Debug/proxy splits are exempt by design.
        if not split.get("protocol", False):
            continue

        # Twin discipline: a twin pair may only straddle train/eval via tier0 (the deliberate
        # positive control). Any other straddle silently turns generalisation into layout recall.
        train = set(split["train"])
        tier0 = set(split.get("tier0", ()))
        held_out = set(split.get("tier1", ()))
        for a, b in TWIN_PAIRS:
            for x, y in ((a, b), (b, a)):
                assert not (x in train and y in held_out), (
                    f"split {name}: layout twins {x} (train) / {y} (tier1) straddle the split; "
                    f"move {y} to tier0 or keep both on one side"
                )
        for lvl in tier0:
            twin = LEVELS[lvl].twin
            assert twin is not None and twin in train, (
                f"split {name}: tier0 stage {lvl} is only a valid positive control if its layout "
                f"twin is trained on (twin={twin})"
            )

        # Warps are enabled, and every warp zone exits to an X-1 stage, so holding one out would
        # let an agent training on a warp stage (1-2, 4-2) travel into the held-out set.
        leaked = WARP_DESTINATIONS & held_out
        assert not leaked, (
            f"split {name}: {sorted(leaked)} are warp destinations and cannot be held out while "
            f"warps are enabled — an agent training on 1-2 or 4-2 could reach them"
        )

        # Every held-out archetype needs a training representative, else tier1 measures
        # archetype extrapolation rather than layout generalisation.
        trained_archetypes = {LEVELS[lvl].archetype for lvl in train}
        for lvl in held_out:
            arch = LEVELS[lvl].archetype
            assert arch in trained_archetypes, (
                f"split {name}: held-out {lvl} has archetype {arch!r} with no training example"
            )


_validate()
