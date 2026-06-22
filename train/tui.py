"""RoboMage TUI launcher.

A Textual control panel that composes and runs train.py / analysis.py / play.py
commands. It imports ONLY cli_spec (the single source of truth for every flag),
never the heavy ML scripts, so it starts instantly and can never drift from the
real CLIs — change a flag in cli_spec.py and it shows up here automatically.

Run from the repo root:
    train/.venv/bin/python train/tui.py

Non-interactive commands stream their output into the log pane (Run/Stop).
Interactive ones (analysis REPLs, play) suspend the TUI and hand over the real
terminal, resuming when they exit.
"""

import asyncio
import glob
import os
import shlex
import signal
import subprocess
import sys

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (Checkbox, Footer, Header, Input, Label, RichLog,
                             Select, Static, Tree)

from cli_spec import (ALL_TOOLS, REPO_ROOT, MutexGroup)

VENV_PY = sys.executable
_DECKS_DIR = os.path.join(REPO_ROOT, "bin", "resources", "decks")
_CKPT_DIR = os.path.join(REPO_ROOT, "train", "checkpoints")
_REC_DIRS = [os.path.join(REPO_ROOT, "recordings"), REPO_ROOT]


def _scan_decks():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(_DECKS_DIR, "*.dk")))


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
_SCANNERS = {"deck": _scan_decks, "checkpoint": _scan_checkpoints, "recording": _scan_recordings}


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
    #fieldhelp { height: 1; padding: 0 1; color: $text-muted; }
    #preview { height: auto; max-height: 5; padding: 0 1; color: $text-muted; border-top: solid $accent; }
    #log { height: 10; border-top: solid $accent; }
    """

    TITLE = "robomage"

    BINDINGS = [
        ("r", "run", "Run"),
        ("s", "stop", "Stop"),
        ("y", "copy_log", "Copy log"),
        ("c", "clear", "Clear log"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self._tool = None
        self._sub = None
        self._fields = []
        self._proc = None
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
                yield RichLog(id="log", markup=False, highlight=False, wrap=True)
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

    def _row(self, name, widget, required):
        """Wrap a field as a single compact row: short name label + widget."""
        label = Label(f"{name}{'*' if required else ''}", classes="fieldname" + (" req" if required else ""))
        return Horizontal(label, widget, classes="fieldrow")

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
        if self._tool.key != "play":     # play.py has no subcommand token
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
        mode = "interactive (runs in terminal)" if self._sub.mode == "interactive" else "capture"
        self.query_one("#preview", Static).update(f"[{mode}]\n{text}")

    # ── actions ───────────────────────────────────────────────────────────
    def action_clear(self):
        self.query_one("#log", RichLog).clear()

    def _log_text(self):
        """The full log buffer as plain text."""
        log = self.query_one("#log", RichLog)
        return "\n".join(strip.text for strip in log.lines)

    def action_copy_log(self):
        """Copy the whole log buffer to the system clipboard (OSC 52)."""
        text = self._log_text()
        log = self.query_one("#log", RichLog)
        if not text:
            log.write("(log is empty — nothing to copy)")
            return
        self.copy_to_clipboard(text)
        log.write(f"copied {len(text.splitlines())} log line(s) to clipboard")

    def action_run(self):
        if not self._sub:
            return
        argv, missing = self._collect()
        log = self.query_one("#log", RichLog)
        if missing:
            log.write(f"Missing required: {', '.join(missing)}")
            return
        if self._sub.mode == "interactive":
            self._run_interactive(argv)
        else:
            self._run_capture(argv)

    def _run_interactive(self, argv):
        log = self.query_one("#log", RichLog)
        with self.suspend():
            print("\n$ " + shlex.join(argv) + "\n", flush=True)
            try:
                subprocess.run(argv, cwd=REPO_ROOT)
            except KeyboardInterrupt:
                pass
            try:
                input("\n[press Enter to return to the launcher] ")
            except EOFError:
                pass
        log.write("returned from interactive command")

    @work(exclusive=True)
    async def _run_capture(self, argv):
        log = self.query_one("#log", RichLog)
        log.write("$ " + shlex.join(argv))
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, cwd=REPO_ROOT,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
                start_new_session=True)
        except Exception as exc:
            log.write(f"failed to start: {exc}")
            return
        self._proc = proc
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                log.write(line.decode("utf-8", "replace").rstrip("\n"))
        finally:
            rc = await proc.wait()
            self._proc = None
            log.write(f"[exit code {rc}]")

    def action_stop(self):
        proc = self._proc
        log = self.query_one("#log", RichLog)
        if not proc or proc.returncode is not None:
            log.write("nothing running")
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        log.write("stop signal sent")


if __name__ == "__main__":
    LauncherApp().run()
