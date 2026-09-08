"""Build the route table (mario_levels.json) from completed episodes in runs/*/episodes.csv.

`x_pos_max` at `flag_get` IS the level length: touching the flag means the level was traversed
end to end, and the flag is a fixed property of the ROM. This is NOT the circular statistic
rejected in mario.md 5.3 -- that was normalising by the furthest point ANY agent reached.

Levels nobody completed have no measurement; --rom fills those from the ROM object list
(src/mario_rom.py, +-72 px on the 13 levels where both agree). Without --rom they get no entry and
mario_eval reports route_progress=None for them, falling back to `pages`.

Usage:
    python src/mario_route_table.py                    # report measured only
    python src/mario_route_table.py --rom               # measured + ROM for the gaps
    python src/mario_route_table.py --rom --write       # ...and write mario_levels.json
"""
import argparse
import collections
import csv
import glob
import json
import os

from mario_levels import ROUTE_TABLE_PATH, START_AREA

# Termination is detected at the end of a 4-frame skip, so x can overshoot the flag by a few
# frames of movement. A completion further than this from the level's mode is not a normal
# flag contact (sanitiser fallback, spurious flag) and is dropped.
OUTLIER_PX = 64
X_START = 40  # every single-stage env starts its primary area at x_pos == 40


def collect(pattern="runs/*/episodes.csv"):
    """level -> list of (x_pos_max, area) over every flag_get episode found."""
    out = collections.defaultdict(list)
    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row.get("flag_get") != "True" or not row.get("level"):
                    continue
                try:
                    x = int(float(row["x_pos_max"]))
                    area = int(row["max_area"])
                except (TypeError, ValueError):
                    continue
                out[row["level"]].append((x, area))
    return out


def summarise(level, samples):
    xs = [x for x, _ in samples]
    mode = collections.Counter(xs).most_common(1)[0][0]
    kept = [x for x in xs if abs(x - mode) <= OUTLIER_PX]
    areas = collections.Counter(a for _, a in samples)
    area = areas.most_common(1)[0][0]
    return {
        "x_max": mode,
        "n": len(kept),
        "n_dropped": len(xs) - len(kept),
        "spread": max(kept) - min(kept),
        "area": area,
        "areas_seen": dict(areas),
        "start_area_expected": START_AREA.get(level),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", action="store_true", help=f"write {ROUTE_TABLE_PATH}")
    p.add_argument("--glob", default="runs/*/episodes.csv")
    p.add_argument("--rom", action="store_true",
                   help="fill unmeasured levels from the ROM object list (needs the mario env)")
    p.add_argument("--log", nargs="?", const="mario_levels.log", default=None,
                   help="also write a readable px/pages table (default mario_levels.log)")
    a = p.parse_args()

    found = collect(a.glob)
    if not found:
        print(f"no flag_get episodes in {a.glob}")
        return

    table, rows = {}, []
    for level in sorted(found):
        s = summarise(level, found[level])
        table[level] = {
            "route": [{"area": s["area"], "x_min": X_START, "x_max": s["x_max"]}],
            "source": "flag_get episodes in runs/*/episodes.csv",
            "n_completions": s["n"],
            "x_spread": s["spread"],
        }
        rows.append((level, s))

    hdr = f"{'level':6s} {'x_max':>6s} {'pages':>6s} {'n':>6s} {'drop':>5s} {'spread':>7s} {'area':>5s} {'START_AREA':>11s}"
    print(hdr)
    print("-" * len(hdr))
    for level, s in rows:
        flag = "" if s["area"] == s["start_area_expected"] else "  <-- AREA MISMATCH"
        print(f"{level:6s} {s['x_max']:6d} {s['x_max'] / 256:6.2f} {s['n']:6d} "
              f"{s['n_dropped']:5d} {s['spread']:7d} {s['area']:5d} "
              f"{str(s['start_area_expected']):>11s}{flag}")
    measured = set(table)
    if a.rom:
        import mario_rom
        from mario_levels import DEFAULT_SPLIT, SMB1_LEVELS, split_levels
        wanted = list(SMB1_LEVELS) + list(split_levels(DEFAULT_SPLIT, "tier2"))
        # Scan every level, not only the unmeasured ones: the object pointer is what identifies
        # shared ROM data, and that has to be known for measured levels too (2-4 == 5-4).
        print(f"\nROM parse for {len(wanted)} levels "
              f"({len(wanted) - len(measured)} of them unmeasured)")
        failed = []
        for r in mario_rom.scan(wanted):
            if not r.get("ok"):
                failed.append((r["level"], r["reason"])); continue
            entry = table.get(r["level"])
            if entry is None:                      # no completion -> take the ROM length
                entry = table[r["level"]] = {
                    "route": [{"area": r["area"], "x_min": X_START, "x_max": r["length_px"]}],
                    "source": "rom object list (src/mario_rom.py)",
                    "warp_tail": r["warp_tail"],
                }
            else:                                  # measured wins; keep the ROM value for audit
                entry["rom_length_px"] = r["length_px"]
                entry["rom_delta_px"] = r["length_px"] - entry["route"][0]["x_max"]
            expected = START_AREA.get(r["level"])
            if expected is not None and r["area"] != expected:
                print(f"  AREA MISMATCH {r['level']}: parsed {r['area']} != START_AREA {expected}")
            entry["object_pointer"] = f"${r['object_pointer']:04X}"
            entry["enemy_pointer"] = f"${r['enemy_pointer']:04X}"
            entry["warp_tail"] = r["warp_tail"]      # on every entry, so the audit can filter
        for level, why in failed:
            print(f"  FAILED {level}: {why}")
        for level, was, now in propagate_twins(table):
            print(f"  twin upgrade {level}: {was} -> {now} px "
                  f"({(was-X_START)//256} -> {(now-X_START)//256} max_pages)")

    print(f"\n{len(measured)} levels measured from completions"
          + (f", {len(table) - len(measured)} added from the ROM" if a.rom else ""))

    twins = collections.defaultdict(list)
    for level, s in rows:
        twins[s["x_max"]].append(level)
    shared = {x: lv for x, lv in twins.items() if len(lv) > 1}
    if shared:
        print("identical lengths (NOT proof of twinning -- mario_rom.py compares ROM\n       pointers, which is the definitive test; 3-4 matches 1-4/6-4 in length only):")
        for x, lv in sorted(shared.items()):
            print(f"  {x:5d} px: {', '.join(lv)}")

    if a.write:
        with open(ROUTE_TABLE_PATH, "w") as f:
            json.dump(table, f, indent=2, sort_keys=True)
        print(f"\nwritten to {os.path.abspath(ROUTE_TABLE_PATH)}")
        if a.log:
            path, n = write_log(a.log)
            print(f"report for {n} levels -> {os.path.abspath(path)}")
    else:
        print(f"\n(dry run -- pass --write to create {ROUTE_TABLE_PATH})")



def propagate_twins(table):
    """A ROM-identical twin of a completed level has the same length -- take the exact value.

    Identical object AND enemy pointers mean the same level data, so a measured flag position
    transfers. This upgrades 7-2 (tier0) and 2-4 (tier1) from estimate to exact.
    """
    groups = collections.defaultdict(list)
    for level, e in table.items():
        if e.get("object_pointer"):
            groups[(e["object_pointer"], e.get("enemy_pointer"))].append(level)

    upgraded = []
    for members in groups.values():
        if len(members) < 2:
            continue
        exact = {table[m]["route"][0]["x_max"] for m in members
                 if table[m]["source"].startswith("flag_get")}
        if len(exact) != 1:
            continue                                   # none measured, or they disagree
        value = exact.pop()
        donors = sorted(m for m in members if table[m]["source"].startswith("flag_get"))
        for m in members:
            e = table[m]
            if e["source"].startswith("flag_get") or e["route"][0]["x_max"] == value:
                continue
            upgraded.append((m, e["route"][0]["x_max"], value))
            e["rom_length_px"] = e["route"][0]["x_max"]
            e["rom_delta_px"] = e["route"][0]["x_max"] - value
            e["route"][0]["x_max"] = value
            e["source"] = f"flag_get via ROM-identical twin ({', '.join(donors)})"
    return upgraded


# ── human-readable report ────────────────────────────────────────────────────────────
def write_log(path="mario_levels.log", table_path=ROUTE_TABLE_PATH):
    """Dump the route table as a sorted text table: px, pages, and the `pages` metric ceiling.

    The `pages` metric is covered_px // 256 and covered_px = x_max - X_START, so the highest value
    it can reach on a level is (x_max - X_START) // 256 -- that, not x_max / 256, is the divisor
    for a normalised pages score.
    """
    import mario_levels as ML

    with open(table_path) as f:
        table = json.load(f)

    def sort_key(level):
        if level.startswith("SuperMarioBros"):
            return (1,) + ML.parse_level("-".join(level.split("-")[1:-1]))
        return (0,) + ML.parse_level(level)

    # ROM pointer identity is the definitive twin test; group by it where we have it.
    groups = collections.defaultdict(list)
    for level, e in table.items():
        if e.get("object_pointer"):
            groups[e["object_pointer"]].append(level)
    shared = {p: sorted(lv, key=sort_key) for p, lv in groups.items() if len(lv) > 1}
    twin_of = {lv: [o for o in g if o != lv] for g in shared.values() for lv in g}

    lines = [
        "Mario level lengths -- route table report",
        f"generated from {table_path} by src/mario_route_table.py",
        "",
        "length_px = x_max from the route table (flag position). traversable = x_max - "
        f"{X_START} (levels start at x_pos {X_START}).",
        "max_pages = (traversable) // 256 -- the ceiling of the `pages` metric, i.e. the divisor "
        "for a normalised score.",
        "source: 'flag_get' = measured from completed episodes (exact); 'rom' = parsed from the "
        "ROM object list (+-72 px).",
        "",
        f"{'level':24s} {'tier':9s} {'archetype':12s} {'length_px':>9s} {'traversable':>11s} "
        f"{'max_pages':>9s} {'pages_frac':>10s} {'source':>9s}  notes",
        "-" * 132,
    ]
    for level in sorted(table, key=sort_key):
        e = table[level]
        x_max = e["route"][0]["x_max"]
        trav = x_max - X_START
        info = ML.LEVELS.get(level)
        notes = []
        if e.get("warp_tail"):
            notes.append("warp tail: ROM overestimates the flag route")
        if level in twin_of:
            notes.append("same ROM data as " + ", ".join(twin_of[level]))
        if e.get("n_completions"):
            notes.append(f"n={e['n_completions']}, spread {e.get('x_spread', 0)}px")
        lines.append(
            f"{level:24s} {ML.tier_of(level) or 'unassigned':9s} "
            f"{(info.archetype if info else 'lost_levels'):12s} {x_max:9d} {trav:11d} "
            f"{trav // 256:9d} {trav / 256:10.2f} "
            f"{e['source'].split()[0]:>9s}  {'; '.join(notes)}"
        )

    missing = [lv for lv in list(ML.SMB1_LEVELS) + list(ML.split_levels(ML.DEFAULT_SPLIT, "tier2"))
               if lv not in table]
    lines += ["", f"{len(table)} levels in the table; missing: {', '.join(missing) or 'none'}"]
    if shared:
        lines += ["", "shared ROM object lists (identical terrain AND enemy data):"]
        for p, lv in sorted(shared.items()):
            lines.append(f"  {p}: {', '.join(lv)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path, len(table)

if __name__ == "__main__":
    main()
