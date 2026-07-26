#!/usr/bin/env python3
"""
Run from repo root after changing the C++ ActionCategory, Step, ActionRefZone,
or MandatoryChoice enums, or the OBS_KEYWORDS array:
    python train/gen_enums.py
Writes train/_enums.py with _CAT_NAMES (action-category int -> short display
name), CAT_<NAME> (name-keyed ActionCategory ints), _STEP_NAMES (ordered step
display names), ACTION_CATEGORY_MAX, _REF_NAMES / REF_ZONE_MAX (per-action
zone_ref block), _MC_NAMES / N_MANDATORY_CHOICES (the MandatoryChoice one-hot in
the global-extras block), _OBS_KEYWORDS / N_OBS_KEYWORDS (the per-permanent
keyword multi-hot order), and the observation layout / size constants
(MAX_ACTIONS, STATE_SIZE, N_CARD_TYPES, PERM_SLOT_SIZE, ...) parsed straight from
the C++ headers so train/env.py never hand-copies them.

The C++ enums in src/classes/action.h (ActionCategory), src/classes/game.h
(Step, MandatoryChoice), and src/classes/gamestate.h (ActionRefZone), plus the
OBS_KEYWORDS string array in src/machine_io.h, are the single source of truth
for the integer values and ordering. The
short human-readable display strings live ONLY here, in _CAT_DISPLAY /
_STEP_DISPLAY / _MC_DISPLAY below. The generator fails loudly if the C++ enum
and the display map drift apart (a new/renamed C++ category with no display
string, or a stale display entry), so the one place to update on an enum change
is this file.
"""
import re, os, sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__)) + "/.."
ACTION_H  = os.path.join(REPO_ROOT, "src/classes/action.h")
GAME_H    = os.path.join(REPO_ROOT, "src/classes/game.h")
GAMESTATE_H = os.path.join(REPO_ROOT, "src/classes/gamestate.h")
MACHINE_IO_H = os.path.join(REPO_ROOT, "src/machine_io.h")
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
    "COMPANION": "COMPANION", "DONT_SHUFFLE": "DONT_SHUFFLE",
    "KEEP_HAND": "KEEP_HAND", "EXILE_FROM_YARD": "EXILE_FROM_YARD",
}

# C++ ActionRefZone enum name -> short display string for the per-action
# zone_ref block (which zone/side the action's referenced entity lives in).
_REF_DISPLAY = {
    "REF_NONE": "-", "REF_SELF_BATTLEFIELD": "own bf",
    "REF_OPP_BATTLEFIELD": "opp bf", "REF_SELF_HAND": "hand",
    "REF_OPP_HAND": "opp hand", "REF_STACK": "stack", "REF_SELF_GY": "own gy", "REF_OPP_GY": "opp gy",
    "REF_SELF_EXILE": "own ex", "REF_OPP_EXILE": "opp ex",
    "REF_PLAYER_SELF": "you", "REF_PLAYER_OPP": "opp",
}

# C++ MandatoryChoice enum name -> display string for the one-hot in the
# state vector's global-extras block (NONE is a real slot at index 0).
_MC_DISPLAY = {
    "NONE": "None",
    "DECLARE_ATTACKERS_CHOICE": "Declare Atk",
    "DECLARE_BLOCKERS_CHOICE": "Declare Blk",
    "CLEANUP_DISCARD": "Cleanup Discard",
    "CHOOSE_ENTITY": "Choose Entity",
    "ASSIGN_COMBAT_DAMAGE_CHOICE": "Assign Dmg",
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


# C++ int constants (a `#define` or `static constexpr int`) to surface into
# _enums.py so train/env.py can import them instead of hand-copying the layout.
# Grouped by the header that owns each; names must match the C++ identifiers
# exactly. parse_int_constant() hard-fails on any missing name (never a default).
_GAMESTATE_INTS = [
    "MAX_ACTIONS", "MAX_CHOICE_DESC", "PERM_COUNTERS_LEN", "PERM_TOKEN_NAME_LEN",
    "MAX_BATTLEFIELD_SLOTS", "MAX_STACK_DISPLAY", "MAX_STACK_MODES",
    "MAX_STACK_TGTS", "MAX_GY_SLOTS", "MAX_HAND_SLOTS",
    "DECKLIST_MAIN_SLOTS", "DECKLIST_SIDE_SLOTS",
]
_GAME_INTS = ["KNOWN_TOP_LIBRARY_SIZE", "ACTION_HISTORY_SIZE"]
_MACHINE_INTS = ["STATE_SIZE", "N_CARD_TYPES", "PERM_SLOT_SIZE", "OPTION_ORDINAL_MAX",
                 "N_ACTION_OBS_BLOCKS", "SIDEBOARD_SWAP_CAP"]


def _strip_comments(text):
    """Remove // line comments and /* */ block comments from C++ source."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def parse_int_constant(path, name):
    """Return the int value of the C++ constant `name` defined in `path`.

    Handles both `#define NAME 123` and `static constexpr int NAME = 123;`
    (also plain `const/constexpr int`). Hard-fails via sys.exit if the constant
    is absent — it NEVER falls back to a default, so a renamed/removed C++
    constant fails the build loudly instead of silently emitting a stale value.
    """
    src = _strip_comments(open(path).read())
    m = re.search(r"#\s*define\s+" + re.escape(name) + r"\s+(\d+)\b", src)
    if not m:
        m = re.search(r"\b" + re.escape(name) + r"\b\s*=\s*(\d+)", src)
    if not m:
        sys.exit(f"gen_enums.py: constant {name!r} not found in {path} — cannot "
                 f"generate train/_enums.py without it (check its name/location "
                 f"in that header; it must be a #define or constexpr int)")
    return int(m.group(1))


def _enum_body(path, header_pattern):
    """Return the comment-stripped body of the enum opened by header_pattern."""
    # Strip comments BEFORE the brace match: a `}` inside a comment (e.g. the
    # `{3}` in a mana-cost comment) would otherwise terminate the non-greedy
    # `\}` early and silently drop every entry after it.
    src = _strip_comments(open(path).read())
    m = re.search(header_pattern + r"\s*\{(.*?)\}", src, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"could not find enum {header_pattern!r} in {path}")
    return m.group(1)


def _split_entries(body):
    """Non-empty, comma-separated entries of a comment-stripped enum body.

    This is the single tokenization used by every enum parser so that the
    per-enum count smoke-check compares like with like (no findall that could
    silently skip a malformed entry).
    """
    return [tok.strip() for tok in body.split(",") if tok.strip()]


def parse_action_categories():
    """Return {enum_name: int} for ActionCategory (explicit `= N` values).

    EVERY entry inside the enum body MUST carry an explicit `= N`. An entry
    written without one is a hard error: the old findall-based parser silently
    dropped it, and if the author also forgot the `_CAT_DISPLAY` string, the
    bidirectional drift check couldn't fire either (the name was absent from
    both sets), so `_enums.py` regenerated wrong without any error. Validating
    every comma-split entry against the explicit-value pattern closes that hole.
    """
    body = _enum_body(ACTION_H, r"enum\s+class\s+ActionCategory")
    entries = _split_entries(body)
    pairs = {}
    for entry in entries:
        m = re.fullmatch(r"(\w+)\s*=\s*(\d+)", entry)
        if not m:
            sys.exit(f"gen_enums.py: ActionCategory entry {entry!r} in {ACTION_H} "
                     f"lacks an explicit '= N' value — every ActionCategory entry "
                     f"must be written 'NAME = <int>' so the generator cannot "
                     f"silently drop it (which would also dodge the display-name "
                     f"drift check and regenerate train/_enums.py wrong)")
        pairs[m.group(1)] = int(m.group(2))
    if not pairs:
        raise RuntimeError("no ActionCategory entries parsed")
    if len(pairs) != len(entries):
        raise RuntimeError(
            f"ActionCategory: parsed {len(pairs)} names but the enum body has "
            f"{len(entries)} entries (duplicate enum name?)")
    return pairs


def _parse_implicit_enum(path, header_pattern, first_expected, what):
    """Return the ordered names of an implicit-sequential enum.

    These enums (Step, MandatoryChoice, ActionRefZone) assign values by position
    (0, 1, 2, …). An explicit `= N` initializer is tolerated ONLY when it agrees
    with the entry's implicit position (e.g. `REF_NONE = 0` at index 0); an
    explicit value that disagrees with the position — which would silently break
    the ordering these parsers assume — is a hard error naming the entry and
    file. A malformed entry is likewise a hard error rather than being skipped.
    """
    body = _enum_body(path, header_pattern)
    entries = _split_entries(body)
    names = []
    for idx, entry in enumerate(entries):
        m = re.fullmatch(r"([A-Za-z_]\w*)\s*(?:=\s*(\d+))?", entry)
        if not m:
            sys.exit(f"gen_enums.py: {what} entry {entry!r} in {path} is malformed "
                     f"(expected 'NAME' with an implicit sequential value)")
        name, val = m.group(1), m.group(2)
        if val is not None and int(val) != idx:
            sys.exit(f"gen_enums.py: {what} entry {entry!r} in {path} has an "
                     f"explicit '= {val}' that disagrees with its implicit "
                     f"sequential position ({idx}) — this enum is parsed by order, "
                     f"so an out-of-sequence explicit value would silently break "
                     f"the mapping; remove the initializer or correct it")
        names.append(name)
    if not names or names[0] != first_expected:
        sys.exit(f"gen_enums.py: {what} parse failed in {path} "
                 f"(expected {first_expected!r} as the first entry)")
    if len(names) != len(entries):
        raise RuntimeError(
            f"{what}: parsed {len(names)} names but the enum body has "
            f"{len(entries)} entries")
    return names


def parse_steps():
    """Return the ordered list of Step enum names (implicit sequential values)."""
    return _parse_implicit_enum(GAME_H, r"typedef\s+enum\s+Step", "UNTAP", "Step")


def parse_ref_zones():
    """Return the ordered list of ActionRefZone enum names (implicit sequential).

    Only REF_NONE carries an explicit `= 0`, which agrees with its position.
    """
    return _parse_implicit_enum(GAMESTATE_H, r"typedef\s+enum\s+ActionRefZone_tag",
                                "REF_NONE", "ActionRefZone")


def parse_mandatory_choices():
    """Return the ordered list of MandatoryChoice enum names (implicit sequential)."""
    return _parse_implicit_enum(GAME_H, r"enum\s+MandatoryChoice", "NONE",
                                "MandatoryChoice")


def parse_obs_keywords():
    """Return the OBS_KEYWORDS strings from machine_io.h, in serialized order."""
    src = _strip_comments(open(MACHINE_IO_H).read())
    m = re.search(r"OBS_KEYWORDS\[\]\s*=\s*\{(.*?)\}\s*;", src, flags=re.DOTALL)
    if not m:
        raise RuntimeError(f"could not find OBS_KEYWORDS array in {MACHINE_IO_H}")
    keywords = re.findall(r'"([^"]+)"', m.group(1))
    if not keywords:
        raise RuntimeError("no OBS_KEYWORDS entries parsed")
    return keywords


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
    mcs = parse_mandatory_choices()
    keywords = parse_obs_keywords()
    _check_coverage(cats.keys(), _CAT_DISPLAY, "ActionCategory")
    _check_coverage(steps, _STEP_DISPLAY, "Step")
    _check_coverage(refs, _REF_DISPLAY, "ActionRefZone")
    _check_coverage(mcs, _MC_DISPLAY, "MandatoryChoice")

    cat_by_int = sorted(cats.items(), key=lambda kv: kv[1])
    cat_max = max(cats.values())

    # Cross-check the hand-maintained C++ ACTION_CATEGORY_MAX against the parsed
    # enum's actual max, so the two can never silently drift.
    cpp_cat_max = parse_int_constant(ACTION_H, "ACTION_CATEGORY_MAX")
    if cpp_cat_max != cat_max:
        sys.exit(f"gen_enums.py: C++ ACTION_CATEGORY_MAX ({cpp_cat_max}) != the "
                 f"parsed ActionCategory enum max ({cat_max}) — update "
                 f"ACTION_CATEGORY_MAX in src/classes/action.h")

    # Observation layout / size constants, parsed from their owning headers.
    layout = []  # (name, value, source-header comment), in emission order
    for name in _GAMESTATE_INTS:
        layout.append((name, parse_int_constant(GAMESTATE_H, name),
                       "src/classes/gamestate.h"))
    for name in _GAME_INTS:
        layout.append((name, parse_int_constant(GAME_H, name),
                       "src/classes/game.h"))
    for name in _MACHINE_INTS:
        layout.append((name, parse_int_constant(MACHINE_IO_H, name),
                       "src/machine_io.h"))

    lines = [
        "# AUTO-GENERATED by train/gen_enums.py — do not edit manually.",
        "# Integer values/ordering come from the C++ enums in",
        "# src/classes/action.h (ActionCategory), src/classes/game.h (Step,",
        "# MandatoryChoice), src/classes/gamestate.h (ActionRefZone), and the",
        "# OBS_KEYWORDS array in src/machine_io.h.",
        "# Re-run train/gen_enums.py after changing any of them.",
        "",
        f"ACTION_CATEGORY_MAX = {cat_max}",
        f"N_ACTION_CATEGORIES = {cat_max + 1}",
        "",
        "# ── Observation layout / size constants (parsed from the C++ headers, the",
        "# single source of truth). train/env.py imports these instead of hand-",
        "# copying the numbers; the header naming each constant follows it. ──",
    ]
    for name, val, src_comment in layout:
        lines.append(f"{name} = {val}  # {src_comment}")
    lines += [
        "",
        "# Name-keyed ActionCategory constants: CAT_<ENUM_NAME> = value, for every",
        "# entry in the C++ ActionCategory enum (src/classes/action.h).",
    ]
    for name, val in cat_by_int:
        lines.append(f"CAT_{name} = {val}")
    lines += [
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
        f"N_MANDATORY_CHOICES = {len(mcs)}",
        "",
        "# MandatoryChoice display names, in enum order (the one-hot in the",
        "# state vector's global-extras block; NONE is a real slot at index 0).",
        "_MC_NAMES = [",
    ]
    for name in mcs:
        lines.append(f'    "{_MC_DISPLAY[name]}",  # {name}')
    lines += [
        "]",
        "",
        f"N_OBS_KEYWORDS = {len(keywords)}",
        "",
        "# Per-permanent keyword multi-hot vocabulary, in serialized order",
        "# (mirrors OBS_KEYWORDS in src/machine_io.h).",
        "_OBS_KEYWORDS = [",
    ]
    for kw in keywords:
        lines.append(f'    "{kw}",')
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
          f"{len(refs)} ref zones, {len(mcs)} mandatory choices, "
          f"{len(keywords)} obs keywords, {len(layout)} layout constants")


if __name__ == "__main__":
    main()
