"""RoboMage TUI launcher.

A Textual control panel that composes and runs train.py / analysis.py / play.py
commands. It imports ONLY cli_spec (the single source of truth for every flag),
never the heavy ML scripts, so it starts instantly and can never drift from the
real CLIs — change a flag in cli_spec.py and it shows up here automatically.

Run from the repo root:
    train/.venv/bin/python train/tui.py

Every command suspends the TUI and runs in the real terminal, with its output
simultaneously teed to a per-run log file. The TUI resumes when the command
exits.
"""

import glob
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Checkbox, Footer, Header, Input, Label,
                             Select, SelectionList, Static, Tree)

from cli_spec import (ALL_TOOLS, REPO_ROOT, MutexGroup)

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
    # checkpoints/-relative paths ('delver__final.zip', 'league/ur_delver__final.zip')
    # — checkpoints of subfolder decks live in the mirrored subfolder, so scan
    # recursively. They are expanded to a repo-relative path at argv time so the
    # value loads whether the script reads it directly (play) or resolves it.
    out = [os.path.relpath(p, _CKPT_DIR).replace(os.sep, "/")
           for p in glob.glob(os.path.join(_CKPT_DIR, "**", "*.zip"), recursive=True)]
    return sorted(out, key=_grouped_sort_key)


def _checkpoints_for_deck(deck):
    """Checkpoints piloting ``deck``, as checkpoints/-relative paths.

    Models are per-deck generalists named '{deck}__final.zip' /
    '{deck}__v{steps}.zip'. A subfolder deck ('league/ur_delver') looks in the
    mirrored checkpoints subfolder first ('league/ur_delver__*.zip' — where
    training saves them), then falls back to a flat top-level '{stem}__*.zip'
    so pre-subfolder checkpoints keep working."""
    ckpts = _scan_checkpoints()
    out = [c for c in ckpts if c.startswith(f"{deck}__")]
    stem = deck.rsplit("/", 1)[-1]
    if stem != deck:
        out += [c for c in ckpts if "/" not in c and c.startswith(f"{stem}__")]
    return out


def _expand_checkpoint(val):
    """A bare checkpoint filename → repo-relative path that every script can load."""
    if val in _scan_checkpoints():
        return os.path.join("train", "checkpoints", val)
    return val   # already a path, a shorthand, or 'scripted' — leave it alone


# Suggestion source tagged on each Arg in cli_spec (arg.suggest) → scanner.
_SCANNERS = {"deck": _scan_decks, "league_deck": _scan_league_decks,
             "checkpoint": _scan_checkpoints}

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


class LauncherApp(App):
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
        self._fields = []
        self._help_by_widget = {}   # widget -> help text, for the focus help line

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
        if event.node.data:
            await self._load_sub(*event.node.data)

    async def _load_sub(self, tool, sub):
        self._tool, self._sub = tool, sub
        form = self.query_one("#form", VerticalScroll)
        await form.remove_children()
        self._fields = []
        self._help_by_widget = {}
        rows = []
        for item in sub.items:
            if isinstance(item, MutexGroup):
                opts = [(a.name, a.name) for a in item.args]
                sel = Select(opts, allow_blank=True, prompt="(neither)", compact=True)
                self._fields.append({"kind": "mutex", "group": item, "widget": sel})
                self._help_by_widget[sel] = "Opponent mode — mutually exclusive (default: neither)"
                rows.append(self._row("opp-mode", sel, required=False))
            else:
                rows.append(self._build_arg(item))
                # A roster multipick gets a live rotation-order readout row below it.
                f = self._fields[-1]
                if f.get("readout") is not None:
                    rows.append(self._row("order", f["readout"], required=False))
        if rows:
            await form.mount(*rows)
        self._apply_league_defaults()
        self.query_one("#fieldhelp", Static).update("")
        self.update_preview()

    def _row(self, name, widget, required, extra=""):
        """Wrap a field as a single compact row: short name label + widget.

        ``extra`` adds CSS classes (e.g. "tall" for multi-line widgets like the
        deck multi-select)."""
        label = Label(f"{name}{'*' if required else ''}", classes="fieldname" + (" req" if required else ""))
        classes = "fieldrow" + (f" {extra}" if extra else "")
        return Horizontal(label, widget, classes=classes)

    def _options_for(self, a):
        """Dropdown options for a suggest-tagged arg (decks/checkpoints)."""
        # --load resumes a specific checkpoint of the selected deck's generalist,
        # so it offers that deck's own pilots (see _deck_checkpoints).
        if a.name == "--load" and a.suggest == "checkpoint":
            return self._deck_checkpoints()
        opts = list(_suggestions_for(a))
        # Fields that also accept the rule-based agent get a 'scripted' option.
        if a.suggest == "checkpoint" and (a.default == "scripted" or a.name == "--opponent"):
            opts = ["scripted"] + opts
        return opts

    def _field_value(self, name):
        """Current string value of a field by arg name, or None if blank/absent."""
        f = next((f for f in self._fields
                  if f.get("arg") is not None and f["arg"].name == name), None)
        if f is None:
            return None
        v = f["widget"].value
        return v if isinstance(v, str) and v else None

    def _deck_checkpoints(self):
        """Checkpoints piloting the currently selected deck.

        Models are per-deck generalists named '{deck}__final.zip' /
        '{deck}__v{steps}.zip', so the --load dropdown offers the chosen deck's
        own pilots (the opponent is irrelevant). Empty until a deck is chosen.
        Subfolder decks look in the mirrored checkpoints subfolder first, with
        a flat-layout fallback (see _checkpoints_for_deck)."""
        deck = self._field_value("--deck")
        if not deck:
            return []
        return _checkpoints_for_deck(deck)

    def _refresh_load_options(self):
        """Re-filter the --load dropdown after a deck/opponent change."""
        field = next((f for f in self._fields
                      if f.get("arg") is not None and f["arg"].name == "--load"), None)
        if field is None:
            return
        sel = field["widget"]
        opts = self._deck_checkpoints()
        keep = sel.value if (isinstance(sel.value, str) and sel.value in opts) else None
        sel.set_options([(v, v) for v in opts])
        if keep is not None:
            sel.value = keep
        else:
            sel.clear()

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
    def _roster_field(self):
        """The league ``--decks`` multipick field, or None when not on it."""
        if getattr(self._sub, "name", None) != "league":
            return None
        f = self._field_by_name("--decks")
        return f if (f is not None and f.get("readout") is not None) else None

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
        """Move the highlighted deck delta places in the rotation order."""
        f = self._roster_field()
        if f is None or self.focused is not f["widget"]:
            return
        w = f["widget"]
        idx = w.highlighted
        if idx is None:
            return
        val = w.get_option_at_index(idx).value
        order = f.get("order", [])
        if val not in order:
            return                                       # highlighted deck not checked
        i = order.index(val)
        j = i + delta
        if 0 <= j < len(order):
            order[i], order[j] = order[j], order[i]
            self._refresh_roster_readout(f)
            self.update_preview()

    def action_roster_earlier(self):
        self._roster_move(-1)

    def action_roster_later(self):
        self._roster_move(1)

    def check_action(self, action, parameters):
        # The roster reorder keys are only meaningful (and only shown in the
        # footer) while the league --decks list is focused.
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
        help_text = self._help_by_widget.get(event.widget)
        if help_text is not None:
            self.query_one("#fieldhelp", Static).update(help_text)
        # Roster reorder keys ([ / ]) are only live while the list is focused, so
        # re-evaluate check_action to show/hide them in the footer on focus change.
        self.refresh_bindings()

    def on_select_changed(self, event: Select.Changed):
        arg = self._arg_for_widget(event.select)
        if arg is not None and arg.name in ("--deck", "--opponent"):
            self._refresh_load_options()
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
    _FULLSCREEN_SCRIPTS = frozenset({"play.py", "tui_analysis.py"})

    def _is_play_mode(self):
        """True when the selected command is itself a full-screen Textual app
        (see _FULLSCREEN_SCRIPTS) — it runs in the terminal without logging."""
        return bool(self._tool) and (os.path.basename(self._tool.script)
                                     in self._FULLSCREEN_SCRIPTS)

    def _collect(self):
        """Return (argv, missing_required_names)."""
        argv = [VENV_PY, self._script_abs()]
        if not self._tool.flat:          # flat tools (play.py, test_harness.py) have no subcommand token
            argv.append(self._sub.name)
        # League --resume ignores all other flags (config restored from the
        # sidecar), so emit just the resume flag for a clean command.
        if getattr(self._sub, "name", None) == "league":
            resume = self._field_by_name("--resume")
            if resume is not None and resume["widget"].value:
                return argv + ["--resume"], []
        missing = []
        for f in self._fields:
            if f["kind"] == "flag":
                if f["widget"].value:
                    argv.append(f["arg"].name)
            elif f["kind"] == "mutex":
                v = f["widget"].value
                if isinstance(v, str) and v:   # real flag name (not the blank sentinel)
                    argv.append(v)
            elif f["kind"] == "choice":
                v = f["widget"].value
                if isinstance(v, str) and v:
                    argv += [f["arg"].name, v]
                elif f["arg"].required:
                    missing.append(f["arg"].name)
            elif f["kind"] == "multipick":   # multi-select roster → comma-joined
                a = f["arg"]
                # Emit in the user-set rotation order (f["order"], reorderable with
                # [ / ]), filtered to what's currently checked; fall back to raw
                # check-order for non-order-tracked multipicks.
                selected = set(f["widget"].selected)
                order = f.get("order")
                if order is not None:
                    vals = [d for d in order if d in selected]
                else:
                    vals = list(f["widget"].selected)
                if vals:
                    argv += [a.name, ",".join(vals)]
                elif a.required:
                    missing.append(a.name)
            elif f["kind"] == "pick":   # dropdown of decks/checkpoints
                a = f["arg"]
                v = f["widget"].value
                if isinstance(v, str) and v:
                    if a.suggest == "checkpoint":
                        v = _expand_checkpoint(v)
                    if a.is_positional:
                        argv.append(v)
                    else:
                        argv += [a.name, v]
                elif a.required:
                    missing.append(a.name)
            else:  # value (free-text Input: numbers, --binary, …)
                a = f["arg"]
                val = f["widget"].value.strip()
                if val:
                    if a.is_positional:
                        argv.append(val)
                    else:
                        argv += [a.name, val]
                elif a.required:
                    missing.append(a.name)
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


if __name__ == "__main__":
    LauncherApp().run()
