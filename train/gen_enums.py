#!/usr/bin/env python3
"""
Run from repo root after changing the C++ ActionCategory, Step, or
ActionRefZone enums:
    python train/gen_enums.py
Writes train/_enums.py with _CAT_NAMES (action-category int -> short display
name), _STEP_NAMES (ordered step display names), ACTION_CATEGORY_MAX, and
_REF_NAMES / REF_ZONE_MAX (per-action zone_ref block).

The C++ enums in src/classes/action.h (ActionCategory), src/classes/game.h
(Step), and src/classes/gamestate.h (ActionRefZone) are the single source of
truth for the integer values and ordering. The
short human-readable display strings live ONLY here, in _CAT_DISPLAY /
_STEP_DISPLAY below. The generator fails loudly if the C++ enum and the display
map drift apart (a new/renamed C++ category with no display string, or a stale
display entry), so the one place to update on an enum change is this file.
"""
import re, os, sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
ACTION_H  = os.path.join(REPO_ROOT, "src/classes/action.h")
GAME_H    = os.path.join(REPO_ROOT, "src/classes/game.h")
GAMESTATE_H = os.path.join(REPO_ROOT, "src/classes/gamestate.h")
OUT_FILE  = os.path.join(REPO_ROOT, "train/_enums.py")

# C++ ActionCategory enum name -> short display abbreviation (the only place
# these cosmetic strings are defined). Keys must exactly match the C++ enum.
_CAT_DISPLAY = {
    "PASS_PRIORITY": "PASS", "MANA_ABILITY": "MANA",
    "SELECT_ATTACKER": "SEL_ATK", "CONFIRM_ATTACKERS": "CONF_ATK",
    "SELECT_BLOCKER": "SEL_BLK", "CONFIRM_BLOCKERS": "CONF_BLK",
    "ACTIVATE_ABILITY": "ACTIVATE", "CAST_SPELL": "CAST",
    "SELECT_TARGET": "TARGET", "PLAY_LAND": "LAND", "OTHER_CHOICE": "OTHER",
    "MULLIGAN": "MULLIGAN", "BOTTOM_DECK_CARD": "BOTTOM_CARD",
    "MANA_W": "MANA_W", "MANA_U": "MANA_U", "MANA_B": "MANA_B",
    "MANA_R": "MANA_R", "MANA_G": "MANA_G", "MANA_C": "MANA_C",
    "SEARCH_LIBRARY": "SEARCH", "TOP_LIBRARY": "TOP_LIB", "SHUFFLE": "SHUFFLE",
    "PAYING_COSTS": "PAYING", "DIG_CHOICE": "DIG",
    "SIDEBOARD_IN": "SB_IN", "SIDEBOARD_OUT": "SB_OUT",
    "SIDEBOARD_DONE": "SB_DONE",
    # Categories split out of the former OTHER_CHOICE catch-all.
    "SACRIFICE_PERMANENT": "SACRIFICE", "RETURN_PERMANENT": "RETURN",
    "CHOOSE_X": "CHOOSE_X", "DISCARD": "DISCARD", "CHOOSE_MODE": "MODE",
    "CHOOSE_MANA_COLOR": "MANA_COLOR", "PAY_UNLESS": "PAY_UNLESS",
    "NAME_CARD": "NAME_CARD", "CHOOSE_TYPE": "CHOOSE_TYPE",
    "KEEP_LEGEND": "KEEP_LEGEND", "ORDER_TRIGGERS": "ORDER_TRIG",
    "CHOOSE_REPLACEMENT": "REPLACE", "ATTACK_TARGET": "ATK_TGT",
    "BLOCK_TARGET": "BLK_TGT", "OPTIONAL_YESNO": "YES_NO",
    "PLAY_FREE": "PLAY_FREE", "SYLVAN_CHOICE": "SYLVAN",
    "CHOOSE_CARD": "CHOOSE_CARD", "ASSIGN_DAMAGE": "ASSIGN_DMG",
    "COMPANION": "COMPANION",
}

# C++ ActionRefZone enum name -> short display string for the per-action
# zone_ref block (which zone/side the action's referenced entity lives in).
_REF_DISPLAY = {
    "REF_NONE": "-", "REF_SELF_BATTLEFIELD": "own bf",
    "REF_OPP_BATTLEFIELD": "opp bf", "REF_SELF_HAND": "hand",
    "REF_STACK": "stack", "REF_SELF_GY": "own gy", "REF_OPP_GY": "opp gy",
    "REF_SELF_EXILE": "own ex", "REF_OPP_EXILE": "opp ex",
    "REF_PLAYER_SELF": "you", "REF_PLAYER_OPP": "opp",
}

# C++ Step enum name -> display string (the only place these are defined).
_STEP_DISPLAY = {
    "UNTAP": "Untap", "UPKEEP": "Upkeep", "DRAW": "Draw",
    "FIRST_MAIN": "First Main", "BEGIN_COMBAT": "Begin Combat",
    "DECLARE_ATTACKERS": "Declare Atk", "DECLARE_BLOCKERS": "Declare Blk",
    "FIRST_STRIKE_DAMAGE": "First Strike Dmg", "COMBAT_DAMAGE": "Combat Dmg",
    "END_OF_COMBAT": "End Combat", "SECOND_MAIN": "Second Main",
    "END_STEP": "End Step", "CLEANUP": "Cleanup",
}


def _strip_comments(text):
    """Remove // line comments and /* */ block comments from C++ source."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _enum_body(path, header_pattern):
    """Return the comment-stripped body of the enum opened by header_pattern."""
    src = open(path).read()
    m = re.search(header_pattern + r"\s*\{(.*?)\}", src, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"could not find enum {header_pattern!r} in {path}")
    return _strip_comments(m.group(1))


def parse_action_categories():
    """Return {enum_name: int} for ActionCategory (explicit `= N` values)."""
    body = _enum_body(ACTION_H, r"enum\s+class\s+ActionCategory")
    pairs = {m.group(1): int(m.group(2))
             for m in re.finditer(r"(\w+)\s*=\s*(\d+)", body)}
    if not pairs:
        raise RuntimeError("no ActionCategory entries parsed")
    return pairs


def parse_steps():
    """Return the ordered list of Step enum names (implicit sequential values)."""
    body = _enum_body(GAME_H, r"typedef\s+enum\s+Step")
    names = [tok.strip() for tok in body.split(",")]
    return [n for n in names if re.fullmatch(r"[A-Z_][A-Z0-9_]*", n)]


def parse_ref_zones():
    """Return the ordered list of ActionRefZone enum names.

    Values are implicit sequential (only REF_NONE carries an explicit `= 0`),
    so parse by order, stripping any `= N` initializer.
    """
    body = _enum_body(GAMESTATE_H, r"typedef\s+enum\s+ActionRefZone_tag")
    names = []
    for tok in body.split(","):
        name = tok.split("=")[0].strip()
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", name):
            names.append(name)
    if not names or names[0] != "REF_NONE":
        raise RuntimeError("ActionRefZone parse failed (expected REF_NONE first)")
    return names


def _check_coverage(parsed_names, display, what):
    """Fail loudly if the C++ enum and the display map have drifted apart."""
    parsed, mapped = set(parsed_names), set(display)
    missing = parsed - mapped
    stale = mapped - parsed
    if missing:
        raise RuntimeError(
            f"{what}: C++ enum has {sorted(missing)} with no display string — "
            f"add them to the display map in gen_enums.py")
    if stale:
        raise RuntimeError(
            f"{what}: display map has {sorted(stale)} not in the C++ enum — "
            f"remove them from gen_enums.py")


def main():
    cats = parse_action_categories()
    steps = parse_steps()
    refs = parse_ref_zones()
    _check_coverage(cats.keys(), _CAT_DISPLAY, "ActionCategory")
    _check_coverage(steps, _STEP_DISPLAY, "Step")
    _check_coverage(refs, _REF_DISPLAY, "ActionRefZone")

    cat_by_int = sorted(cats.items(), key=lambda kv: kv[1])
    cat_max = max(cats.values())

    lines = [
        "# AUTO-GENERATED by train/gen_enums.py — do not edit manually.",
        "# Integer values/ordering come from the C++ enums in",
        "# src/classes/action.h (ActionCategory) and src/classes/game.h (Step).",
        "# Re-run train/gen_enums.py after changing either enum.",
        "",
        f"ACTION_CATEGORY_MAX = {cat_max}",
        f"N_ACTION_CATEGORIES = {cat_max + 1}",
        "",
        "# ActionCategory value -> short display name.",
        "_CAT_NAMES = {",
    ]
    for name, val in cat_by_int:
        lines.append(f'    {val}: "{_CAT_DISPLAY[name]}",  # {name}')
    lines += [
        "}",
        "",
        "# Step (turn phase) display names, in enum order.",
        "_STEP_NAMES = [",
    ]
    for name in steps:
        lines.append(f'    "{_STEP_DISPLAY[name]}",  # {name}')
    lines += [
        "]",
        "",
        f"REF_ZONE_MAX = {len(refs) - 1}",
        f"N_REF_ZONES = {len(refs)}",
        "",
        "# ActionRefZone value -> short display name (per-action zone_ref block).",
        "_REF_NAMES = {",
    ]
    for val, name in enumerate(refs):
        lines.append(f'    {val}: "{_REF_DISPLAY[name]}",  # {name}')
    lines += ["}", ""]

    with open(OUT_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {OUT_FILE}: {len(cats)} categories, {len(steps)} steps, "
          f"{len(refs)} ref zones")


if __name__ == "__main__":
    main()
