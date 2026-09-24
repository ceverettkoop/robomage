"""The GUI launcher dialogs' field tables and their one settings file.

Qt-free on purpose: the dialogs (gui_game.NewPlaySessionDialog, gui_main.
NewAnalysisSessionDialog) build their widgets from these tables, and the
default ``make check`` tier (test_cli_spec.py) checks the tables against
cli_spec without instantiating a widget.

Every dialog field IS a command-line flag: the field is keyed by the flag's
argparse dest, starts from the flag's cli_spec default, and is persisted
under that dest. The Play dialog mirrors play.py (``cli_spec.PLAY_TOOL``);
the Analysis dialog mirrors the analysis browser (``analysis.py browse``,
``cli_spec.ANALYSIS_BROWSE_SUB``). Both dialogs remember their last-used
values in ONE file, ``~/.robomage/gui_launcher.json``, one section per dialog
(``{"play": {...}, "analysis": {...}}``). Keys that are not a field of that
dialog, or whose value does not fit the flag, are ignored — an old or
hand-edited file never breaks the launcher.
"""

import json
import os

from cli_spec import (ANALYSIS_BROWSE_SUB, PLAY_TOOL, arg_default, iter_args,
                      play_seat_specs, resolve_play_seats,
                      DEFAULT_PLAY_OPPONENT)

LAUNCHER_CONFIG = os.path.join(os.path.expanduser("~"), ".robomage",
                               "gui_launcher.json")
# The two per-dialog files the launchers wrote before they shared one; read
# (never written) when the shared file has no section for that dialog yet.
_LEGACY_CONFIGS = {
    "play": LAUNCHER_CONFIG,
    "analysis": os.path.join(os.path.expanduser("~"), ".robomage",
                             "gui_analysis_launcher.json"),
}

PLAY_SECTION = "play"
ANALYSIS_SECTION = "analysis"

# Dialog field -> flag dest, per section. Every play.py / browse flag the
# dialog does NOT offer is listed in *_CLI_ONLY with the reason, so the table
# accounts for the whole Sub (test_cli_spec checks both directions).
PLAY_FIELDS = (
    "player_a", "player_b", "deck_a", "deck_b", "on_the_play", "format",
    "human_clock", "hard_timeout", "record_shards",
    "sims", "worlds", "think_time", "search_procs", "match_clock",
    "search_device", "search_xw", "paced",
    "analysis", "analysis_evaluator", "analysis_worlds", "analysis_procs",
    "analysis_cap", "analysis_device", "analysis_xw", "analysis_auto",
)
PLAY_CLI_ONLY = {
    "board": "the dialog IS the GUI board",
    "seed": "a launcher session always draws a fresh engine seed "
            "(File ▸ Open replays a saved one)",
    "binary": "the app's --binary applies to every session it starts",
}
ANALYSIS_FIELDS = (
    "player_a", "player_b", "deck_a", "deck_b", "games", "seed", "format",
    "sims", "worlds", "think_time", "search_procs", "match_clock",
    "search_device", "search_xw",
    "source", "seat", "no_net",
)
ANALYSIS_CLI_ONLY = {
    "board": "the dialog IS the GUI board",
    "binary": "the app's --binary applies to every session it starts",
}

_SECTIONS = {
    PLAY_SECTION: (PLAY_TOOL.subs[0], PLAY_FIELDS),
    ANALYSIS_SECTION: (ANALYSIS_BROWSE_SUB, ANALYSIS_FIELDS),
}


def section_args(section):
    """``{dest: Arg}`` for a section's fields, in field order."""
    sub, fields = _SECTIONS[section]
    by_dest = {a.dest: a for a in iter_args(sub)}
    return {dest: by_dest[dest] for dest in fields}


def section_defaults(section):
    """``{dest: default}`` for a section's fields — the cli_spec defaults.

    The play seats are the RESOLVED defaults (play.py leaves both --player-a
    and --player-b unset, which ``resolve_play_seats`` reads as the human on
    A against the default opponent on B), since a seat combo needs a value."""
    out = {dest: arg_default(a) for dest, a in section_args(section).items()}
    if section == PLAY_SECTION:
        human, opponent = resolve_play_seats(out["player_a"], out["player_b"])
        out["player_a"], out["player_b"] = play_seat_specs(
            human, opponent or DEFAULT_PLAY_OPPONENT)
    return out


def _fits(arg, value):
    """True when a saved ``value`` is a legal value of ``arg``'s flag."""
    if value is None:
        return True
    if arg.kind in ("flag", "bool"):
        return isinstance(value, bool)
    if arg.kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if arg.kind == "float":
        return (isinstance(value, (int, float))
                and not isinstance(value, bool))
    if arg.kind == "choice":
        return value in arg.choices
    return isinstance(value, str)


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def load_section(section, path=None):
    """A dialog's starting values: the cli_spec defaults, overlaid key by key
    with the saved values that are fields of the dialog and fit their flag.
    A saved None is honored (a knob parked on its '(default)' sentinel)."""
    values = section_defaults(section)
    data = _read_json(path or LAUNCHER_CONFIG)
    saved = data.get(section)
    if not isinstance(saved, dict) and path is None:
        legacy = _read_json(_LEGACY_CONFIGS[section])
        saved = legacy if not isinstance(legacy.get(section), dict) else None
    if not isinstance(saved, dict):
        return values
    args = section_args(section)
    for dest, value in saved.items():
        if dest not in args or not _fits(args[dest], value):
            continue
        if value is None and values[dest] is not None:
            continue                         # None only where the flag's unset
        values[dest] = value
    return values


def save_section(section, values, path=None):
    """Persist a dialog's field values (best-effort; other sections kept)."""
    path = path or LAUNCHER_CONFIG
    data = _read_json(path)
    data = {k: v for k, v in data.items() if k in _SECTIONS}
    fields = section_args(section)
    data[section] = {dest: values[dest] for dest in fields if dest in values}
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass                                 # persistence is best-effort
