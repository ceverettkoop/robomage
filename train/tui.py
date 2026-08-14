"""RoboMage TUI launcher.

A Textual control panel that composes and runs train.py / analysis.py / play.py
commands. It imports ONLY cli_spec (the single source of truth for every flag)
and curriculum (itself cli_spec-only), never the heavy ML scripts, so it starts
instantly and can never drift from the real CLIs — change a flag in cli_spec.py
and it shows up here automatically.

Run from the repo root:
    train/.venv/bin/python train/tui.py

Every command suspends the TUI and runs in the real terminal, with its output
simultaneously teed to a per-run log file. The TUI resumes when the command
exits.
"""

import contextlib
import glob
import os
import random
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import (Button, Checkbox, Footer, Header, Input, Label,
                             ListItem, ListView, Select, SelectionList, Static,
                             Tree)

from cli_spec import (ALL_TOOLS, REPO_ROOT, MutexGroup)
# Curriculum plans: stdlib-only module (cli_spec + progress_io), so the launcher
# can list/read/write plan files without pulling in the ML stack.
import curriculum

VENV_PY = sys.executable
_DECKS_DIR = os.path.join(REPO_ROOT, "bin", "resources", "decks")
_CKPT_DIR = os.path.join(REPO_ROOT, "train", "checkpoints")

# Rolling per-command output logs: one file per run, newest 50 kept.
_CMD_LOG_DIR = os.path.join(REPO_ROOT, "train", "logs", "tui_commands")
_CMD_LOG_KEEP = 50
# `script` (util-linux) tees an interactive command's pty output to a file
# without breaking its interactivity. None if unavailable → log header only.
_SCRIPT_BIN = shutil.which("script")


def _open_command_log(argv):
    """Create a fresh log file for one command run and prune old ones.

    Returns an open text file handle (caller closes it) with a header already
    written, or None if the log dir can't be created. Files are named with a
    sortable timestamp so pruning to the newest ``_CMD_LOG_KEEP`` is a simple
    sort-and-trim.
    """
    try:
        os.makedirs(_CMD_LOG_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        f = open(os.path.join(_CMD_LOG_DIR, f"cmd_{stamp}.log"), "w", encoding="utf-8")
    except OSError:
        return None
    f.write(f"# {datetime.now().isoformat(timespec='seconds')}\n")
    f.write("$ " + shlex.join(argv) + "\n\n")
    f.flush()
    _prune_command_logs()
    return f


def _prune_command_logs():
    """Keep only the newest ``_CMD_LOG_KEEP`` command log files."""
    logs = sorted(glob.glob(os.path.join(_CMD_LOG_DIR, "cmd_*.log")))
    for path in logs[:-_CMD_LOG_KEEP]:
        try:
            os.remove(path)
        except OSError:
            pass


# Deck subfolders hidden from the dropdowns: temp/ holds auto-generated test
# decks (see test_harness.py), not_used/ parked development stubs.
_DECK_SCAN_EXCLUDE = frozenset({"temp", "not_used"})


def _grouped_sort_key(rel):
    """Sort decks/checkpoints top-level first, then grouped by subfolder,
    alphabetical within each group."""
    return (rel.count("/"), rel)


def _scan_decks():
    """All .dk decks under decks/ (recursive), as decks/-relative stems.

    Subfolder decks are offered in the 'league/ur_delver' path-relative form
    that train.py and the engine accept alongside top-level stems like
    'delver'. temp/ and not_used/ are excluded."""
    out = []
    for root, dirs, files in os.walk(_DECKS_DIR):
        dirs[:] = sorted(d for d in dirs if d not in _DECK_SCAN_EXCLUDE)
        rel_dir = os.path.relpath(root, _DECKS_DIR).replace(os.sep, "/")
        for fname in files:
            if fname.endswith(".dk"):
                stem = os.path.splitext(fname)[0]
                out.append(stem if rel_dir == "." else f"{rel_dir}/{stem}")
    return sorted(out, key=_grouped_sort_key)


def _scan_league_decks():
    # League roster decks live in decks/league/; reference them as 'league/<stem>'
    # so the engine loads decks/league/<stem>.dk (matches train.league() default).
    return sorted("league/" + os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(_DECKS_DIR, "league", "*.dk")))


def _scan_checkpoints():
    # checkpoints/-relative paths ('gen__final.zip', 'gen__v{steps}.zip', plus any
    # explicit zip) — scan recursively so a nested layout still turns up. They are
    # expanded to a repo-relative path at argv time so the value loads whether the
    # script reads it directly (play) or resolves it.
    out = [os.path.relpath(p, _CKPT_DIR).replace(os.sep, "/")
           for p in glob.glob(os.path.join(_CKPT_DIR, "**", "*.zip"), recursive=True)]
    return sorted(out, key=_grouped_sort_key)


def _scan_az_checkpoints():
    # checkpoints/-relative AZ nets ('az/gen__azfinal.pt', 'az/gen__azv{steps}.pt'),
    # skipping the .ts.pt TorchScript exports (actor runtime artifacts, not loadable nets).
    out = [os.path.relpath(p, _CKPT_DIR).replace(os.sep, "/")
           for p in glob.glob(os.path.join(_CKPT_DIR, "az", "**", "*.pt"),
                              recursive=True)
           if not p.endswith(".ts.pt")]
    return sorted(out, key=_grouped_sort_key)


def _scan_agents():
    """Controller specs for fields consumed by opponents.make_controller.

    The model is the ONE generalist (`gen`), so the AZ entries are the fixed
    'az:gen' (MCTS at play time), 'azraw:gen' (raw policy), and 'mcts:gen'
    (PPO-priored search) specs — the prefix IS the raw-vs-search lever, and
    they resolve through resolve_az_checkpoint/resolve_checkpoint to the
    generalist's best net as training advances. Explicit scanned AZ checkpoint
    paths are offered too, plus every PPO zip (bare, as before)."""
    az = [f"{pfx}gen" for pfx in ("az:", "azraw:", "mcts:")]
    return _scan_checkpoints() + az + _scan_az_checkpoints()


def _scan_curricula():
    """Curriculum plan names under checkpoints/curricula/ (see curriculum.py)."""
    return curriculum.list_plans()


def _generalist_checkpoints():
    """PPO checkpoints the --load field can resume, as checkpoints/-relative
    paths.

    There is one generalist model (`gen`), so this is every gen snapshot
    ('gen__v{steps}.zip'), 'gen__final.zip', and any explicit .zip present —
    i.e. all scanned zips. The choice is deck-independent (the deck the model
    pilots travels as a separate explicit --deck parameter)."""
    return _scan_checkpoints()


def _expand_checkpoint(val):
    """A bare checkpoint filename → repo-relative path that every script can load."""
    if val in _scan_checkpoints() or val in _scan_az_checkpoints():
        return os.path.join("train", "checkpoints", val)
    return val   # a path, a shorthand, an az:/azraw: spec, or 'scripted'


# Suggestion source tagged on each Arg in cli_spec (arg.suggest) → scanner.
# 'checkpoint' stays PPO-zips-only (fields that MaskablePPO.load directly:
# --load resume, --from-ppo); 'agent' adds the az:/azraw:/mcts: gen entries for
# fields that go through make_controller (observe's players, baseline's model);
# 'az_checkpoint' is bare AZ .pt paths.
_SCANNERS = {"deck": _scan_decks, "league_deck": _scan_league_decks,
             "checkpoint": _scan_checkpoints, "agent": _scan_agents,
             "az_checkpoint": _scan_az_checkpoints,
             "curriculum": _scan_curricula}

# TUI-only default overrides for the league form (the common "train the whole
# roster" run): these seed the widgets differently from the CLI Arg defaults but
# leave cli_spec / the command-line untouched. The multipick '--decks' is
# pre-checked separately (see _apply_league_defaults) since it needs the mounted
# widget. '--bo3' is a shared flag (common_args), so we override it here rather
# than flipping its global cli_spec default.
_LEAGUE_TUI_DEFAULTS = {
    "--bo3": True,            # best-of-three on by default
    "--promote-margin": 0,    # 0 disables the snapshot win-rate gate
}


def _suggestions_for(arg):
    fn = _SCANNERS.get(getattr(arg, "suggest", None))
    return fn() if fn else []


class ArgFormMixin:
    """Builds an input form from ``cli_spec`` Sub items and reads it back.

    Shared by the launcher's flat command form and the curriculum screen's
    per-phase form: one Arg -> one row (Checkbox / Select / SelectionList /
    Input), a MutexGroup -> one Select, plus the roster ordering behaviour of a
    ``multi=True`` deck picker. Hosts must call ``_init_form_state()`` and
    provide ``self._sub`` (the Sub being edited, used for the form-level
    defaults) and a ``#fieldhelp`` Static.
    """

    # Seed an az* form's --seed with a fresh random value per load (see
    # _default_for). Hosts that EDIT a stored document rather than launch a
    # one-off command turn this off.
    _RANDOM_AZ_SEED = True

    def _init_form_state(self):
        self._fields = []
        self._help_by_widget = {}   # widget -> help text, for the focus help line

    def _form_rows(self, items):
        """Widget rows for a list of Sub items; populates ``self._fields``."""
        rows = []
        for item in items:
            if isinstance(item, MutexGroup):
                opts = [(a.name, a.name) for a in item.args]
                sel = Select(opts, allow_blank=True, prompt="(neither)", compact=True)
                self._fields.append({"kind": "mutex", "group": item, "widget": sel})
                self._help_by_widget[sel] = item.help
                rows.append(self._row(item.label, sel, required=False))
            else:
                rows.append(self._build_arg(item))
                # A roster multipick gets a live rotation-order readout row below it.
                f = self._fields[-1]
                if f.get("readout") is not None:
                    rows.append(self._row("order", f["readout"], required=False))
        return rows

    def _row(self, name, widget, required, extra=""):
        """Wrap a field as a single compact row: short name label + widget.

        ``extra`` adds CSS classes (e.g. "tall" for multi-line widgets like the
        deck multi-select)."""
        label = Label(f"{name}{'*' if required else ''}", classes="fieldname" + (" req" if required else ""))
        classes = "fieldrow" + (f" {extra}" if extra else "")
        return Horizontal(label, widget, classes=classes)

    def _options_for(self, a):
        """Dropdown options for a suggest-tagged arg (decks/checkpoints)."""
        # --load resumes a specific checkpoint of the one generalist model, so it
        # offers every gen snapshot / gen__final / explicit zip (deck-independent;
        # the piloted deck travels as a separate --deck parameter).
        if a.name == "--load" and a.suggest == "checkpoint":
            return _generalist_checkpoints()
        opts = list(_suggestions_for(a))
        # Fields that also accept the rule-based agent get a 'scripted' option.
        if a.suggest in ("checkpoint", "agent") and (
                a.default == "scripted" or a.name == "--opponent"):
            opts = ["scripted"] + opts
        # AZ fields defaulting to the generalist stem offer 'gen' itself, so the
        # default is selectable (and preselected) rather than only the explicit
        # snapshot paths the scanner finds.
        if a.suggest == "az_checkpoint" and a.default == "gen":
            opts = ["gen"] + opts
        return opts

    def _arg_for_widget(self, widget):
        for f in self._fields:
            if f.get("widget") is widget:
                return f.get("arg")
        return None

    def _default_for(self, a):
        """Effective initial value for an arg's widget, applying TUI-only overrides.

        Currently the league form seeds a few widgets differently from their
        cli_spec defaults (see ``_LEAGUE_TUI_DEFAULTS``); everything else falls
        through to the Arg's own default."""
        if getattr(self._sub, "name", None) == "league" and a.name in _LEAGUE_TUI_DEFAULTS:
            return _LEAGUE_TUI_DEFAULTS[a.name]
        # az* forms: pre-fill --seed with a fresh random value per form load, so
        # repeated TUI launches don't silently replay the fixed cli_spec seed
        # (identical self-play games before any training has happened). CLI
        # defaults are untouched, and the value is visible in the field and the
        # composed-command preview, so every run stays reproducible. A plan
        # editor opts out (_RANDOM_AZ_SEED): merely selecting a phase must not
        # rewrite the saved plan with a new random seed.
        if (self._RANDOM_AZ_SEED and a.name == "--seed"
                and str(getattr(self._sub, "name", "")).startswith("az")):
            return random.randint(1, 999_999)
        return a.default

    def _build_arg(self, a):
        # One row per arg; the (verbose) help shows in #fieldhelp when focused.
        default = self._default_for(a)
        if a.kind == "flag":
            w = Checkbox(value=bool(default), compact=True)
            self._fields.append({"kind": "flag", "arg": a, "widget": w})
        elif a.kind == "choice":
            opts = [(c, c) for c in a.choices]
            kwargs = {"allow_blank": not a.required, "compact": True}
            if default in a.choices:          # only pass a real, valid default
                kwargs["value"] = default
            w = Select(opts, **kwargs)
            self._fields.append({"kind": "choice", "arg": a, "widget": w})
        elif a.suggest and getattr(a, "multi", False):
            # multi-select roster (e.g. league --decks): pick several with space.
            # League pre-checks all decks post-mount (see _apply_league_defaults).
            # `order` is the emitted roster order (== training rotation order); it
            # tracks selection order and is reorderable with [ / ] (see
            # action_roster_earlier / _reconcile_roster_order). `readout` shows it.
            vals = self._options_for(a)
            w = SelectionList(*[(v, v) for v in vals])
            readout = Static("", classes="rosterorder")
            self._fields.append({"kind": "multipick", "arg": a, "widget": w,
                                 "order": [], "readout": readout})
            self._help_by_widget[w] = a.help
            return self._row(a.name, w, a.required, extra="tall")
        elif a.suggest:
            # deck / checkpoint → dropdown of repo contents
            vals = self._options_for(a)
            kwargs = {"allow_blank": True, "compact": True}
            if default in vals:
                kwargs["value"] = default
            w = Select([(v, v) for v in vals], **kwargs)
            self._fields.append({"kind": "pick", "arg": a, "widget": w})
        else:
            itype = "integer" if a.kind == "int" else "text"
            w = Input(value="" if default is None else str(default),
                      type=itype, placeholder=a.name, compact=True)
            self._fields.append({"kind": "value", "arg": a, "widget": w})
        self._help_by_widget[w] = a.help
        return self._row(a.name, w, a.required)

    def _field_by_name(self, name):
        return next((f for f in self._fields
                     if f.get("arg") is not None and f["arg"].name == name), None)

    def _field_by_dest(self, dest):
        return next((f for f in self._fields
                     if f.get("arg") is not None and f["arg"].dest == dest), None)

    # ── reading the form back ────────────────────────────────────────────
    def _field_value(self, f):
        """One field's current value, normalized by field kind.

        flag -> bool; mutex -> the chosen flag NAME or None; choice/pick -> str
        or None; multipick -> the ordered list of picks; value -> the trimmed
        text or None."""
        w = f["widget"]
        kind = f["kind"]
        if kind == "flag":
            return bool(w.value)
        if kind in ("mutex", "choice", "pick"):
            v = w.value
            return v if isinstance(v, str) and v else None
        if kind == "multipick":
            selected = set(w.selected)
            order = f.get("order")
            return ([d for d in order if d in selected] if order is not None
                    else list(w.selected))
        text = w.value.strip()
        return text or None

    def _collect_values(self):
        """``([(field, value)], missing_required_names)`` for the whole form."""
        out, missing = [], []
        for f in self._fields:
            val = self._field_value(f)
            a = f.get("arg")
            if a is not None and a.required and val in (None, [], False):
                missing.append(a.name)
            out.append((f, val))
        return out, missing

    def _apply_values(self, values):
        """Push ``{dest: value}`` back into the mounted widgets (form prefill)."""
        for dest, val in values.items():
            f = self._field_by_dest(dest)
            if f is None:
                continue
            w = f["widget"]
            if f["kind"] == "flag":
                w.value = bool(val)
            elif f["kind"] == "multipick":
                picks = ([v.strip() for v in str(val).split(",") if v.strip()]
                         if not isinstance(val, (list, tuple)) else list(val))
                w.deselect_all()
                for p in picks:
                    with contextlib.suppress(Exception):
                        w.select(p)
                f["order"] = picks
                self._refresh_roster_readout(f)
            elif f["kind"] in ("choice", "pick"):
                with contextlib.suppress(Exception):
                    w.value = val
            else:
                w.value = "" if val is None else str(val)

    # ── roster ordering (== training rotation order) ─────────────────────
    def _roster_field(self):
        """The focused-form's order-tracked multipick field, or None."""
        return next((f for f in self._fields if f.get("readout") is not None), None)

    def _reconcile_roster_order(self, f):
        """Sync f["order"] (the emitted rotation order) with what's checked:
        keep the existing order for survivors (preserving manual [ / ] reorders),
        drop deselected decks, and append newly-checked decks in check order."""
        selected = list(f["widget"].selected)           # check-order
        sel_set = set(selected)
        order = [d for d in f.get("order", []) if d in sel_set]
        order += [d for d in selected if d not in order]
        f["order"] = order
        self._refresh_roster_readout(f)

    def _refresh_roster_readout(self, f):
        order = [d for d in f.get("order", []) if d in set(f["widget"].selected)]
        if order:
            txt = "  ".join(f"{i + 1}. {d.split('/')[-1]}" for i, d in enumerate(order))
        else:
            txt = "(no decks selected)"
        f["readout"].update("rotation → " + txt)

    def _roster_move(self, delta):
        """Move the highlighted deck delta places in the rotation order.

        Returns True when it acted (the roster list had focus), so a host that
        shares the [ / ] keys with another list can fall through."""
        f = self._roster_field()
        if f is None or self.focused is not f["widget"]:
            return False
        w = f["widget"]
        idx = w.highlighted
        if idx is None:
            return False
        val = w.get_option_at_index(idx).value
        order = f.get("order", [])
        if val not in order:
            return False                                 # highlighted deck not checked
        i = order.index(val)
        j = i + delta
        if 0 <= j < len(order):
            order[i], order[j] = order[j], order[i]
            self._refresh_roster_readout(f)
        return True

    def _show_field_help(self, widget):
        """Update the #fieldhelp line for the newly focused widget."""
        help_text = self._help_by_widget.get(widget)
        if help_text is not None:
            self.query_one("#fieldhelp", Static).update(help_text)


class LauncherApp(ArgFormMixin, App):
    CSS = """
    #tree { width: 22; border-right: solid $accent; }
    #right { width: 1fr; }
    #form { height: 1fr; padding: 0 1; }
    .fieldrow { height: 1; margin-bottom: 1; }
    .fieldname { width: 18; color: $text-muted; text-align: right; padding: 0 1 0 0; }
    .req { color: $warning; }
    .fieldrow Input, .fieldrow Select { width: 1fr; }
    .fieldrow.tall { height: auto; }
    .fieldrow.tall SelectionList { width: 1fr; height: auto; max-height: 8; border: round $accent; }
    #fieldhelp { height: 1; padding: 0 1; color: $text-muted; }
    .rosterorder { width: 1fr; color: $accent; }
    #preview { height: auto; max-height: 5; padding: 0 1; color: $text-muted; border-top: solid $accent; }
    """

    TITLE = "robomage"

    BINDINGS = [
        ("r", "run", "Run"),
        ("q", "quit", "Quit"),
        # League roster reorder — only active while the --decks list is focused
        # (see check_action); moves the highlighted deck in the training rotation.
        ("[", "roster_earlier", "◀ order"),
        ("]", "roster_later", "order ▶"),
    ]

    def __init__(self):
        super().__init__()
        self._tool = None
        self._sub = None
        self._init_form_state()

    # ── layout ────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield Tree("Commands", id="tree")
            with Vertical(id="right"):
                yield VerticalScroll(id="form")
                yield Static("", id="fieldhelp")
                yield Static("Select a command on the left.", id="preview")
        yield Footer()

    def on_mount(self):
        tree = self.query_one("#tree", Tree)
        tree.show_root = False
        for tool in ALL_TOOLS:
            node = tree.root.add(tool.key, expand=True)
            for sub in tool.subs:
                leaf = node.add_leaf(sub.name)
                leaf.data = (tool, sub)
        tree.root.expand()

    # ── form building ─────────────────────────────────────────────────────
    async def on_tree_node_selected(self, event: Tree.NodeSelected):
        if not event.node.data:
            return
        tool, sub = event.node.data
        # 'curriculum' is not a flat form: it opens the multi-phase plan builder
        # (the one command whose input is a JSON file rather than a flag list).
        if sub.name == "curriculum":
            self.push_screen(CurriculumScreen())
            return
        await self._load_sub(tool, sub)

    async def _load_sub(self, tool, sub):
        self._tool, self._sub = tool, sub
        form = self.query_one("#form", VerticalScroll)
        await form.remove_children()
        self._init_form_state()
        rows = self._form_rows(sub.items)
        if rows:
            await form.mount(*rows)
        self._apply_league_defaults()
        self.query_one("#fieldhelp", Static).update("")
        self.update_preview()

    def _apply_league_defaults(self):
        """Post-mount league conveniences: pre-check all roster decks, then apply
        the --resume gate. No-op for every other sub."""
        if getattr(self._sub, "name", None) != "league":
            return
        decks = self._field_by_name("--decks")
        if decks is not None and decks["kind"] == "multipick":
            decks["widget"].select_all()   # start with the whole roster checked
            self._reconcile_roster_order(decks)   # seed rotation order + readout
        self._apply_resume_gate()

    # ── league roster ordering (== training rotation order) ─────────────────
    def action_roster_earlier(self):
        if self._roster_move(-1):
            self.update_preview()

    def action_roster_later(self):
        if self._roster_move(1):
            self.update_preview()

    def check_action(self, action, parameters):
        # The roster reorder keys are only meaningful (and only shown in the
        # footer) while a deck multipick list is focused.
        if action in ("roster_earlier", "roster_later"):
            f = self._roster_field()
            return f is not None and self.focused is f["widget"]
        return True

    def _apply_resume_gate(self):
        """League ``--resume`` restores the full run config from the sidecar and
        ignores every other flag, so disable all other fields while it is ticked
        (and _collect emits only ``--resume``)."""
        if getattr(self._sub, "name", None) != "league":
            return
        resume = self._field_by_name("--resume")
        if resume is None:
            return
        on = bool(resume["widget"].value)
        for f in self._fields:
            w = f.get("widget")
            if w is not None and w is not resume["widget"]:
                w.disabled = on

    # ── reactive updates ──────────────────────────────────────────────────
    def on_descendant_focus(self, event):
        self._show_field_help(event.widget)
        # Roster reorder keys ([ / ]) are only live while the list is focused, so
        # re-evaluate check_action to show/hide them in the footer on focus change.
        self.refresh_bindings()

    def on_select_changed(self, event: Select.Changed):
        self.update_preview()

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged):
        f = next((f for f in self._fields if f.get("widget") is event.selection_list), None)
        if f is not None and f.get("readout") is not None:
            self._reconcile_roster_order(f)
        self.update_preview()

    def on_input_changed(self, event: Input.Changed):
        self.update_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed):
        arg = self._arg_for_widget(event.control)
        if arg is not None and arg.name == "--resume":
            self._apply_resume_gate()
        self.update_preview()

    # ── argv composition ──────────────────────────────────────────────────
    def _script_abs(self):
        return os.path.join(REPO_ROOT, self._tool.script)

    # Scripts that take over the whole terminal with their own Textual app:
    # play.py launches the game board (tui_game.py) and tui_analysis.py is the
    # analysis browser. Teeing either through `script` would fill the log with
    # terminal escape sequences, so they run without logging.
    _FULLSCREEN_SCRIPTS = frozenset({"play.py", "tui_analysis.py",
                                     "tui_az_inspect.py"})

    def _is_play_mode(self):
        """True when the selected command is itself a full-screen Textual app
        (see _FULLSCREEN_SCRIPTS) — it runs in the terminal without logging."""
        return bool(self._tool) and (os.path.basename(self._tool.script)
                                     in self._FULLSCREEN_SCRIPTS)

    def _collect(self):
        """Return (argv, missing_required_names) for the selected command."""
        argv = [VENV_PY, self._script_abs()]
        if not self._tool.flat:          # flat tools (play.py, test_harness.py) have no subcommand token
            argv.append(self._sub.name)
        # League --resume ignores all other flags (config restored from the
        # sidecar), so emit just the resume flag for a clean command.
        if getattr(self._sub, "name", None) == "league":
            resume = self._field_by_name("--resume")
            if resume is not None and resume["widget"].value:
                return argv + ["--resume"], []
        values, missing = self._collect_values()
        for f, val in values:
            if f["kind"] == "mutex":
                if val:                  # a real flag name (not the blank sentinel)
                    argv.append(val)
                continue
            a = f["arg"]
            if f["kind"] == "flag":
                if val:
                    argv.append(a.name)
                continue
            if val in (None, []):
                continue
            if f["kind"] == "multipick":   # multi-select roster -> comma-joined
                text = ",".join(val)
            elif f["kind"] == "pick" and a.suggest in ("checkpoint", "agent",
                                                       "az_checkpoint"):
                text = _expand_checkpoint(val)
            else:
                text = val
            if a.is_positional:
                argv.append(text)
            else:
                argv += [a.name, text]
        return argv, missing

    def _preview_text(self, argv):
        rel = [os.path.relpath(a, REPO_ROOT) if a.startswith(REPO_ROOT) else a for a in argv]
        rel[0] = "python"
        return shlex.join(rel)

    def update_preview(self):
        if not self._sub:
            return
        argv, missing = self._collect()
        text = self._preview_text(argv)
        if missing:
            text += f"\n\nmissing required: {', '.join(missing)}"
        banner = ("[runs in terminal]" if self._is_play_mode()
                  else "[runs in terminal — output logged]")
        self.query_one("#preview", Static).update(f"{banner}\n{text}")

    # ── actions ───────────────────────────────────────────────────────────
    def action_run(self):
        if not self._sub:
            return
        argv, missing = self._collect()
        if missing:
            self.notify(f"Missing required: {', '.join(missing)}", severity="error")
            return
        self._run_in_terminal(argv)

    def _run_in_terminal(self, argv):
        """Suspend the TUI, run the command in the real terminal, and tee its
        output to a per-run log file; resume the TUI when the command exits.

        `script` (util-linux) runs the command in a pty — keeping interactive
        commands interactive — while appending everything it shows to logpath
        after our header, so the file mirrors the terminal live. When `script`
        is unavailable the command still runs, but only the header is logged.
        Play mode (the full-screen TUI game board) is exempt — it runs unlogged
        (see _is_play_mode)."""
        logf = _open_command_log(argv) if not self._is_play_mode() else None
        logpath = logf.name if logf else None
        if logf:
            logf.close()
        run_argv = argv
        if logpath and _SCRIPT_BIN:
            run_argv = [_SCRIPT_BIN, "-q", "-a", "-e",
                        "-c", shlex.join(argv), logpath]
        elif logpath:
            self.notify("`script` not found — terminal output will not be logged",
                        severity="warning")
        with self.suspend():
            print("\n$ " + shlex.join(argv) + "\n", flush=True)
            try:
                subprocess.run(run_argv, cwd=REPO_ROOT)
            except KeyboardInterrupt:
                pass
            try:
                input("\n[press Enter to return to the launcher] ")
            except EOFError:
                pass
        if logpath and _SCRIPT_BIN:
            self.notify(f"output logged to {os.path.relpath(logpath, REPO_ROOT)}")


class CurriculumScreen(ArgFormMixin, Screen):
    """Plan builder + status view for multi-phase training curricula.

    The one command whose input is a FILE rather than a flag list, so it gets a
    screen of its own instead of the flat form: the left pane edits the phase
    list (add / remove / reorder, same ``[`` / ``]`` convention as the league
    roster), the right pane is that phase's argument form — generated by the
    shared ``ArgFormMixin`` from the phase kind's own cli_spec Sub, so a league
    phase offers the same deck multipick and an AZ phase the same knobs as the
    flat forms do. Save/Load read and write
    ``train/checkpoints/curricula/<name>.plan.json``; Run/Resume hand
    ``train.py curriculum --plan …`` to the launcher's terminal-suspend path
    like every other command."""

    DEFAULT_CSS = """
    #cleft { width: 46; border-right: solid $accent; }
    #cright { width: 1fr; }
    #phases { height: 1fr; border: round $accent; }
    #cbuttons { height: 6; }
    #cbuttons .cbuttonrow { height: 3; }
    #cbuttons Button { min-width: 8; margin: 0 1 0 0; }
    #phaseform { height: 1fr; padding: 0 1; }
    #cstatus { height: auto; max-height: 10; padding: 0 1; color: $text-muted;
               border-top: solid $accent; }
    """

    BINDINGS = [
        ("escape", "back", "Back"),
        ("ctrl+s", "save", "Save"),
        ("ctrl+r", "run", "Run"),
        # Phase reorder — falls through to the roster reorder while a phase
        # form's deck multipick is focused (see action_move_earlier).
        ("[", "move_earlier", "◀ move"),
        ("]", "move_later", "move ▶"),
    ]

    # A plan is a stored document: selecting an az phase must not rewrite it
    # with a fresh random --seed the way a one-off az launch form does.
    _RANDOM_AZ_SEED = False

    def __init__(self, name_or_path: str = None):
        super().__init__()
        self._init_form_state()
        self._sub = None            # the Sub whose form is currently mounted
        self._phase_idx = None
        self._plan = {"version": curriculum.PLAN_VERSION, "name": "new_plan",
                      "phases": []}
        self._pending_load = name_or_path

    # ── layout ────────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="cleft"):
                yield self._row("plan name", Input(value=self._plan["name"],
                                                   id="planname", compact=True),
                                required=False)
                yield self._row("load plan",
                                Select([(p, p) for p in _scan_curricula()],
                                       id="planpick", allow_blank=True,
                                       compact=True), required=False)
                yield self._row("phase kind",
                                Select([(k, k) for k in curriculum.PHASE_KINDS],
                                       id="kindpick", allow_blank=False,
                                       value=curriculum.PHASE_KINDS[0],
                                       compact=True), required=False)
                yield ListView(id="phases")
                with Vertical(id="cbuttons"):
                    with Horizontal(classes="cbuttonrow"):
                        yield Button("Add", id="add", variant="primary")
                        yield Button("Del", id="del")
                        yield Button("Save", id="save")
                        yield Button("Load", id="load")
                    with Horizontal(classes="cbuttonrow"):
                        yield Button("Run", id="run", variant="success")
                        yield Button("Resume", id="resume")
                        yield Button("Status", id="status")
            with Vertical(id="cright"):
                yield VerticalScroll(id="phaseform")
                yield Static("", id="fieldhelp")
                yield Static("", id="cstatus")
        yield Footer()

    async def on_mount(self):
        if self._pending_load:
            await self._load_plan(self._pending_load)
        else:
            await self._refresh_phase_list()
        self._show_status("Add phases on the left; edit the selected phase's "
                          "arguments on the right. Save writes "
                          f"{os.path.relpath(curriculum.CURRICULA_DIR, REPO_ROOT)}"
                          "/<name>.plan.json")

    # ── plan <-> widgets ──────────────────────────────────────────────────
    async def _refresh_phase_list(self, select=None):
        lv = self.query_one("#phases", ListView)
        await lv.clear()
        for i, phase in enumerate(self._plan["phases"]):
            lv.append(ListItem(Label(self._phase_line(i, phase))))
        if self._plan["phases"]:
            lv.index = max(0, min(len(self._plan["phases"]) - 1,
                                  self._phase_idx if select is None else select))
        else:
            self._phase_idx = None
            await self._clear_form()

    def _phase_line(self, i, phase):
        return f"{i + 1}. {curriculum.describe_phase(phase)}"

    def _refresh_phase_line(self, idx):
        """Update one list row in place (keeps focus, unlike a full rebuild)."""
        lv = self.query_one("#phases", ListView)
        if 0 <= idx < len(lv.children):
            with contextlib.suppress(Exception):
                lv.children[idx].query_one(Label).update(
                    self._phase_line(idx, self._plan["phases"][idx]))

    async def _clear_form(self):
        await self.query_one("#phaseform", VerticalScroll).remove_children()
        self._init_form_state()
        self._sub = None

    async def _load_phase_form(self, idx):
        """Mount the selected phase's argument form and prefill it."""
        self._phase_idx = idx
        phase = self._plan["phases"][idx]
        kind = phase["kind"]
        form = self.query_one("#phaseform", VerticalScroll)
        await form.remove_children()
        self._init_form_state()
        self._sub = curriculum.phase_sub(kind)
        rows = self._form_rows(curriculum.phase_form_args(kind))
        if rows:
            await form.mount(*rows)
        self._apply_values(curriculum.phase_values(phase))
        self.query_one("#fieldhelp", Static).update(
            f"phase {idx + 1}: train.py {kind}")

    def _form_phase(self, kind):
        """The phase dict the mounted form currently describes.

        Only values that DIFFER from the subcommand's own default are stored, so
        a saved plan stays readable and keeps tracking the cli_spec defaults;
        the convenience fields (steps / decks / archetype / …) are always kept
        so a phase line reads like the plan documentation."""
        values, _missing = self._collect_values()
        out = {}
        for f, val in values:
            if f["kind"] == "mutex":
                if val:
                    arg = next(a for a in f["group"].args if a.name == val)
                    out[arg.dest] = True
                continue
            a = f["arg"]
            if f["kind"] == "flag":
                if val:
                    out[a.dest] = True
                continue
            if val in (None, [], ""):
                continue
            named = curriculum.dest_phase_field(kind, a.dest) is not None
            if named or str(val) != str(a.default):
                out[a.dest] = curriculum.coerce_value(a, val)
        return curriculum.make_phase(kind, out)

    def _sync_phase(self):
        """Write the mounted form back into the selected phase."""
        if self._phase_idx is None or self._sub is None:
            return
        idx = self._phase_idx
        self._plan["phases"][idx] = self._form_phase(self._sub.name)
        self._refresh_phase_line(idx)

    # ── events ────────────────────────────────────────────────────────────
    async def on_list_view_highlighted(self, event: ListView.Highlighted):
        idx = event.list_view.index
        if idx is None or idx >= len(self._plan["phases"]) or idx == self._phase_idx:
            return
        await self._load_phase_form(idx)

    def on_descendant_focus(self, event):
        self._show_field_help(event.widget)

    def on_select_changed(self, event: Select.Changed):
        event.stop()
        if event.select.id in ("planpick", "kindpick"):
            return
        self._sync_phase()

    def on_selection_list_selected_changed(self, event):
        event.stop()
        f = next((f for f in self._fields
                  if f.get("widget") is event.selection_list), None)
        if f is not None and f.get("readout") is not None:
            self._reconcile_roster_order(f)
        self._sync_phase()

    def on_input_changed(self, event: Input.Changed):
        event.stop()
        if event.input.id == "planname":
            self._plan["name"] = event.input.value.strip() or "new_plan"
            return
        self._sync_phase()

    def on_checkbox_changed(self, event: Checkbox.Changed):
        event.stop()
        self._sync_phase()

    async def on_button_pressed(self, event: Button.Pressed):
        await self._do(event.button.id)

    async def _do(self, what):
        if what == "add":
            await self.action_add_phase()
        elif what == "del":
            await self.action_del_phase()
        elif what == "save":
            self.action_save()
        elif what == "load":
            await self._load_plan(self.query_one("#planpick", Select).value)
        elif what == "run":
            self._launch(resume=False)
        elif what == "resume":
            self._launch(resume=True)
        elif what == "status":
            self.action_status()

    # ── actions ───────────────────────────────────────────────────────────
    async def action_add_phase(self):
        kind = self.query_one("#kindpick", Select).value
        if not isinstance(kind, str):
            return
        self._plan["phases"].append(curriculum.make_phase(kind, {}))
        await self._refresh_phase_list(select=len(self._plan["phases"]) - 1)

    async def action_del_phase(self):
        if self._phase_idx is None:
            return
        idx = self._phase_idx
        del self._plan["phases"][idx]
        self._phase_idx = None
        await self._refresh_phase_list(select=max(0, idx - 1))

    def _move_phase(self, delta):
        i = self._phase_idx
        if i is None:
            return
        j = i + delta
        phases = self._plan["phases"]
        if not (0 <= j < len(phases)):
            return
        phases[i], phases[j] = phases[j], phases[i]
        self._phase_idx = j
        self._refresh_phase_line(i)
        self._refresh_phase_line(j)
        self.query_one("#phases", ListView).index = j

    def action_move_earlier(self):
        # Inside a phase form's deck multipick the keys keep their league
        # meaning (reorder the rotation); elsewhere they move the phase.
        if not self._roster_move(-1):
            self._move_phase(-1)
        else:
            self._sync_phase()

    def action_move_later(self):
        if not self._roster_move(1):
            self._move_phase(1)
        else:
            self._sync_phase()

    def action_back(self):
        self.app.pop_screen()

    def _plan_path(self):
        name = self.query_one("#planname", Input).value.strip() or "new_plan"
        return curriculum.plan_path(name)

    def action_save(self):
        self._sync_phase()
        try:
            path = curriculum.save_plan(self._plan_path(), self._plan)
        except (curriculum.PlanError, OSError) as exc:
            self.notify(str(exc), severity="error", timeout=10)
            return None
        self.notify(f"saved {os.path.relpath(path, REPO_ROOT)}")
        picker = self.query_one("#planpick", Select)
        picker.set_options([(p, p) for p in _scan_curricula()])
        return path

    async def _load_plan(self, name_or_path):
        if not isinstance(name_or_path, str) or not name_or_path:
            return
        try:
            plan = curriculum.load_plan(curriculum.plan_path(name_or_path))
        except curriculum.PlanError as exc:
            self.notify(str(exc), severity="error", timeout=10)
            return
        self._plan = plan
        self._phase_idx = None
        self.query_one("#planname", Input).value = plan.get("name", "")
        await self._refresh_phase_list(select=0)
        self.action_status()

    def action_status(self):
        """Show the plan's progress sidecar (per-phase state + driver detail)."""
        path = self._plan_path()
        if not os.path.exists(path):
            self._show_status("(save the plan to track its progress)")
            return
        try:
            plan = curriculum.load_plan(path)
        except curriculum.PlanError as exc:
            self._show_status(str(exc))
            return
        self._show_status(curriculum.format_status(
            plan, curriculum.read_progress(path), path))

    def _show_status(self, text):
        self.query_one("#cstatus", Static).update(text)

    def action_run(self):
        self._launch(resume=False)

    def _launch(self, resume: bool):
        """Save the plan, then run it through the launcher's terminal path."""
        path = self.action_save()
        if path is None:
            return
        argv = [VENV_PY, os.path.join(REPO_ROOT, "train", "train.py"),
                "curriculum", "--plan", path]
        if resume:
            argv.append("--resume")
        self.app._run_in_terminal(argv)
        self.action_status()


if __name__ == "__main__":
    LauncherApp().run()
