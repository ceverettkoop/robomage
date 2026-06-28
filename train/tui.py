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
_REC_DIRS = [os.path.join(REPO_ROOT, "recordings"), REPO_ROOT]

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


def _scan_decks():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(_DECKS_DIR, "*.dk")))


def _scan_league_decks():
    # League roster decks live in decks/league/; reference them as 'league/<stem>'
    # so the engine loads decks/league/<stem>.dk (matches train.league() default).
    return sorted("league/" + os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(_DECKS_DIR, "league", "*.dk")))


def _scan_checkpoints():
    # Basenames make the inline autocomplete nice (type "delver" → completes the
    # filename). They are expanded to a repo-relative path at argv time so the
    # value loads whether the script reads it directly (play) or resolves it.
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(_CKPT_DIR, "*.zip")))


def _expand_checkpoint(val):
    """A bare checkpoint filename → repo-relative path that every script can load."""
    if val in _scan_checkpoints():
        return os.path.join("train", "checkpoints", val)
    return val   # already a path, a shorthand, or 'scripted' — leave it alone


def _scan_recordings():
    out = []
    for d in _REC_DIRS:
        out += glob.glob(os.path.join(d, "*.rmrec"))
    return sorted(os.path.relpath(p, REPO_ROOT) for p in out)


# Suggestion source tagged on each Arg in cli_spec (arg.suggest) → scanner.
_SCANNERS = {"deck": _scan_decks, "league_deck": _scan_league_decks,
             "checkpoint": _scan_checkpoints, "recording": _scan_recordings}


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
    #preview { height: auto; max-height: 5; padding: 0 1; color: $text-muted; border-top: solid $accent; }
    """

    TITLE = "robomage"

    BINDINGS = [
        ("r", "run", "Run"),
        ("q", "quit", "Quit"),
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
        if rows:
            await form.mount(*rows)
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
        """Dropdown options for a suggest-tagged arg (decks/checkpoints/recordings)."""
        # --load resumes training the {deck}_{opponent} matchup, so it only
        # offers checkpoints saved for that matchup (see _matchup_checkpoints).
        if a.name == "--load" and a.suggest == "checkpoint":
            return self._matchup_checkpoints()
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

    def _matchup_checkpoints(self):
        """Checkpoints compatible with the currently selected matchup.

        Training checkpoints are named '{model_deck}_{opp_deck}_final.zip' (and
        '..._{N}_steps.zip'); resuming requires a checkpoint of the same
        matchup, so only filenames prefixed with the chosen '{deck}_{opponent}_'
        are offered. Empty until both deck and opponent are chosen."""
        deck, opp = self._field_value("--deck"), self._field_value("--opponent")
        if not deck or not opp:
            return []
        prefix = f"{deck}_{opp}_"
        return [c for c in _scan_checkpoints() if c.startswith(prefix)]

    def _refresh_load_options(self):
        """Re-filter the --load dropdown after a deck/opponent change."""
        field = next((f for f in self._fields
                      if f.get("arg") is not None and f["arg"].name == "--load"), None)
        if field is None:
            return
        sel = field["widget"]
        opts = self._matchup_checkpoints()
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

    def _build_arg(self, a):
        # One row per arg; the (verbose) help shows in #fieldhelp when focused.
        if a.kind == "flag":
            w = Checkbox(value=bool(a.default), compact=True)
            self._fields.append({"kind": "flag", "arg": a, "widget": w})
        elif a.kind == "choice":
            opts = [(c, c) for c in a.choices]
            kwargs = {"allow_blank": not a.required, "compact": True}
            if a.default in a.choices:        # only pass a real, valid default
                kwargs["value"] = a.default
            w = Select(opts, **kwargs)
            self._fields.append({"kind": "choice", "arg": a, "widget": w})
        elif a.suggest and getattr(a, "multi", False):
            # multi-select roster (e.g. league --decks): pick several with space.
            vals = self._options_for(a)
            w = SelectionList(*[(v, v) for v in vals])
            self._fields.append({"kind": "multipick", "arg": a, "widget": w})
            self._help_by_widget[w] = a.help
            return self._row(a.name, w, a.required, extra="tall")
        elif a.suggest:
            # deck / checkpoint / recording → dropdown of repo contents
            vals = self._options_for(a)
            kwargs = {"allow_blank": True, "compact": True}
            if a.default in vals:
                kwargs["value"] = a.default
            w = Select([(v, v) for v in vals], **kwargs)
            self._fields.append({"kind": "pick", "arg": a, "widget": w})
        else:
            itype = "integer" if a.kind == "int" else "text"
            w = Input(value="" if a.default is None else str(a.default),
                      type=itype, placeholder=a.name, compact=True)
            self._fields.append({"kind": "value", "arg": a, "widget": w})
        self._help_by_widget[w] = a.help
        return self._row(a.name, w, a.required)

    # ── reactive updates ──────────────────────────────────────────────────
    def on_descendant_focus(self, event):
        help_text = self._help_by_widget.get(event.widget)
        if help_text is not None:
            self.query_one("#fieldhelp", Static).update(help_text)

    def on_select_changed(self, event: Select.Changed):
        arg = self._arg_for_widget(event.select)
        if arg is not None and arg.name in ("--deck", "--opponent"):
            self._refresh_load_options()
        self.update_preview()

    def on_selection_list_selected_changed(self, event: SelectionList.SelectedChanged):
        self.update_preview()

    def on_input_changed(self, event: Input.Changed):
        self.update_preview()

    def on_checkbox_changed(self, event: Checkbox.Changed):
        self.update_preview()

    # ── argv composition ──────────────────────────────────────────────────
    def _script_abs(self):
        return os.path.join(REPO_ROOT, self._tool.script)

    def _collect(self):
        """Return (argv, missing_required_names)."""
        argv = [VENV_PY, self._script_abs()]
        if not self._tool.flat:          # flat tools (play.py, test_harness.py) have no subcommand token
            argv.append(self._sub.name)
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
                vals = list(f["widget"].selected)
                if vals:
                    argv += [a.name, ",".join(vals)]
                elif a.required:
                    missing.append(a.name)
            elif f["kind"] == "pick":   # dropdown of decks/checkpoints/recordings
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
        self.query_one("#preview", Static).update(f"[runs in terminal — output logged]\n{text}")

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
        is unavailable the command still runs, but only the header is logged."""
        logf = _open_command_log(argv)
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
