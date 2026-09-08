"""Level lengths straight out of the ROM object list -- no playthrough needed.

SMB1 stores each area's terrain as 2-byte entries: b0 = xxxxyyyy (column within the page, row),
b1 = n ttttttt (n = advance to the next page, t = object type). 0xFD terminates the list.
An entry with b0 == 0x0D and b1 < 0x20 is an absolute page-set to b1 instead -- both discriminators
are load-bearing: without the b1 < 0x20 test, 1-4/2-2/5-4 read a regular object (b1 0xC4/0xC9/0x45)
as a page jump to 196/201/69.

The area pointer is read from RAM ($E7/$E8), so no ROM pointer table has to be hardcoded; that
needs a live env, hence the mario conda env.

Calibrated against the 14 levels measured from flag_get episodes (see mario_route_table.py):
`max_object_x - X_BIAS` has mean error -0.0 px, sd 35 px, max 72 px over the 13 non-warp levels.

Usage:
    python src/mario_rom.py                 # all SMB1 stages + the tier2 Lost Levels
    python src/mario_rom.py --levels 1-1,8-1
"""
import argparse

import mario_levels as ML

TERMINATOR = 0xFD
PAGE_SET = 0x0D          # b0 value marking an absolute page-set
MAX_PAGE = 0x20          # page numbers fit in 5 bits; above this, b1 is an object type
COLUMN_PX = 16           # one object column
PAGE_PX = ML.PAGE_WIDTH  # 256
X_BIAS = 94              # calibration: max_object_x overshoots the flag by this much
MAX_ENTRIES = 2000

# The object list runs past the flag into the warp zone on these, so a ROM length overestimates
# the route to the flagpole. Prefer a measured value where one exists.
WARP_TAIL = frozenset({"1-2", "4-2"})


def prg_bytes(lost_levels=False):
    from gym_super_mario_bros._roms import smb1_rom_path, smb2jp_rom_path
    path = smb2jp_rom_path() if lost_levels else smb1_rom_path()
    with open(path, "rb") as f:
        raw = f.read()
    return raw[16:16 + 32768]          # 16-byte iNES header, then 32K PRG at $8000


def area_pointer(env_id):
    """(object ptr, enemy ptr, start area) after reset. Needs gym_super_mario_bros."""
    import gymnasium as gym
    import gym_super_mario_bros  # noqa: F401
    from nes_py.wrappers import JoypadSpace
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

    env = JoypadSpace(gym.make(env_id, render_mode="rgb_array"), COMPLEX_MOVEMENT)
    try:
        _, info = env.reset(seed=1)
        ram = env.unwrapped.ram
        # area comes from info, not RAM: $075A is LIVES ($0760+1 is the area), and reading the
        # wrong one silently keys the route table on area 2 everywhere -- which makes
        # _route_progress return 0.0 rather than None.
        return (int(ram[0xE7]) + 256 * int(ram[0xE8]),
                int(ram[0xE9]) + 256 * int(ram[0xEA]),
                int(info["area"]))
    finally:
        env.close()


def parse_area(prg, pointer):
    """Walk one object list. Returns end page, furthest object x, entry count, page-set count."""
    if not 0x8000 <= pointer <= 0xFFFF - 1:
        return {"ok": False, "reason": f"pointer ${pointer:04X} outside PRG"}

    addr, page, max_x, entries, page_sets = pointer, 0, 0, 0, 0
    while entries < MAX_ENTRIES:
        i = addr - 0x8000
        if i + 1 >= len(prg):
            return {"ok": False, "reason": "ran off the end of PRG"}
        b0, b1 = prg[i], prg[i + 1]
        if b0 == TERMINATOR:
            return {"ok": True, "end_page": page, "max_object_x": max_x,
                    "entries": entries, "page_sets": page_sets}
        if b0 == PAGE_SET and b1 < MAX_PAGE:
            page = b1
            page_sets += 1
        else:
            if b1 & 0x80:
                page += 1
            max_x = max(max_x, page * PAGE_PX + (b0 >> 4) * COLUMN_PX)
        entries += 1
        addr += 2
    return {"ok": False, "reason": f"no terminator within {MAX_ENTRIES} entries"}


def length_of(level, prg_cache=None):
    """Estimated level length in pixels, plus the raw parse. `level` is "3-2" or a full env id."""
    lost = ML.is_lost_levels(level)
    env_id = level if level.startswith("SuperMarioBros") else ML.env_id_for(level)
    prg_cache = prg_cache if prg_cache is not None else {}
    if lost not in prg_cache:
        prg_cache[lost] = prg_bytes(lost)

    obj_ptr, enemy_ptr, area = area_pointer(env_id)
    parsed = parse_area(prg_cache[lost], obj_ptr)
    out = {"level": level, "env_id": env_id, "object_pointer": obj_ptr,
           "enemy_pointer": enemy_ptr, "area": area, **parsed}
    if parsed.get("ok"):
        out["length_px"] = parsed["max_object_x"] - X_BIAS
        out["pages"] = out["length_px"] / PAGE_PX
        out["warp_tail"] = level in WARP_TAIL
    return out


def scan(levels):
    cache = {}
    return [length_of(lv, cache) for lv in levels]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--levels", default=None, help="comma-separated; default all SMB1 + tier2")
    a = p.parse_args()

    if a.levels:
        levels = [s.strip() for s in a.levels.split(",") if s.strip()]
    else:
        levels = list(ML.SMB1_LEVELS) + list(ML.split_levels(ML.DEFAULT_SPLIT, "tier2"))

    rows = scan(levels)
    print(f"{'level':24s} {'obj_ptr':>8s} {'end_pg':>6s} {'length':>7s} {'pages':>6s} {'sets':>4s}  note")
    groups = {}
    for r in rows:
        if not r.get("ok"):
            print(f"{r['level']:24s} ${r['object_pointer']:04X} {'':>6s} {'':>7s} {'':>6s} "
                  f"{'':>4s}  FAILED: {r['reason']}")
            continue
        note = "warp tail -- overestimates the flag route" if r["warp_tail"] else ""
        print(f"{r['level']:24s} ${r['object_pointer']:04X} {r['end_page']:6d} "
              f"{r['length_px']:7d} {r['pages']:6.2f} {r['page_sets']:4d}  {note}")
        groups.setdefault(r["object_pointer"], []).append(r["level"])

    dupes = {p_: lv for p_, lv in groups.items() if len(lv) > 1}
    if dupes:
        print("\nshared object lists (identical terrain data):")
        for p_, lv in sorted(dupes.items()):
            print(f"  ${p_:04X}: {', '.join(lv)}")


if __name__ == "__main__":
    main()
