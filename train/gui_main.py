"""RoboMage GUI app shell: one MainWindow, a menu bar, and session management.

The window is created ONCE per process and switches between three central
panes on a QStackedWidget:

* WelcomePane   — app title + "File ▸ New Session" hint (no session).
* PlayPane      — the live game board (gui_game), one per play session.
* BrowserPane   — the retrospective analysis browser (gui_browser), one per
                  analysis session (imported lazily; optional module).

SessionManager owns session build/teardown (teardown-before-build — never two
engine processes at once), the File ▸ Save/Open flows (gui_session_io's
.rmplay/.rmtrace formats), and the smoke-exit plumbing. The play teardown
order is load-bearing and preserved from gui_game: pane.shutdown() (event
filter off → analysis window down FIRST → driver quit → thread join), THEN
env.close(), then the pane is removed.

Entry points:

* ``run(...)``          — play.py --gui: build a play session directly.
* ``run_launcher(...)`` — gui.sh / no arguments: welcome pane with the menus
                          live (File ▸ New Session to begin; no dialog is
                          auto-opened).

Smokes: ``ROBOMAGE_GUI_SMOKE`` / ``ROBOMAGE_ANALYSIS_SMOKE`` keep their
gui_game semantics — the pane auto-plays and emits session_finished, which
(under smoke only) tears the session down and quits the app; the analysis
smoke's exit-1 check lives in ``run``. All confirm/critical dialogs are
suppressed (printed to stderr) while any smoke env var is set so headless
runs can never block on a modal.
"""

import os
import sys
import tempfile
import time
import traceback

import numpy as np

from PySide6.QtCore import Qt, QObject, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                               QVBoxLayout, QHBoxLayout, QStackedWidget,
                               QDialog, QComboBox, QFormLayout,
                               QDialogButtonBox, QMessageBox, QGroupBox,
                               QSpinBox, QCheckBox, QLineEdit, QToolButton,
                               QFileDialog)

import gui_session_io
from game_driver import build_session
from gui_game import (PlayPane, NewPlaySessionDialog, _ensure_app,
                      _analysis_cfg_from, _smoke_n_from_env, _scan_decks,
                      _OPPONENT_PRESETS, _load_launcher_config,
                      _save_launcher_config, _resolve_opponent_spec)

# Where the analysis-session dialog remembers its last-used options.
_ANALYSIS_LAUNCHER_CONFIG = os.path.join(
    os.path.expanduser("~"), ".robomage", "gui_analysis_launcher.json")

_ANALYSIS_DEFAULTS = {
    "model": "gen",
    "opponent": "scripted:hard",
    "deck_a": "league/ur_delver",
    "deck_b": "",
    "n_games": 20,
    "bo3": True,
    "think_time": None,
    "match_clock": None,
    "shards_on": False,
    "shards": "",
    "seat": "A",
    "no_net": False,
}

# Model (value-net) presets for the analysis dialog's editable combo.
_ANALYSIS_MODEL_PRESETS = ["gen", "az:gen", "azraw:gen", "mcts:gen"]

_KEYS_TEXT = """Game board:
  digits          pick the numbered action
  Enter           confirm the highlighted action
  Space           pass priority (when pass is action 0)
  P               autopass to the next upkeep
  arrows          jump to / move in the action menu (from anywhere)
  hold Q, or right-click-hold a card    oracle-text popup
  < / >           narrow / widen the log panel
  F9              toggle the analysis window
                  (F5 analyze, F6 review opponent's last decision,
                   Shift+F5 stop)
  F10             analyze the session's shard recording (when
                  "Record shards" was ticked) in a second window

Game menu (no shortcuts — conceding is deliberate, and confirmed):
  Concede Game    lose this game; in a bo3 the match continues
  Concede Match   end the match now in your opponent's favour

Application:
  Ctrl+N          new play session
  Ctrl+Shift+N    new analysis session
  Ctrl+O          open a saved session
  Ctrl+S          save session        Ctrl+Shift+S  save session as
  Ctrl+Q          quit"""


def _smoke_active():
    """True when any headless smoke env var is set — modal dialogs must be
    suppressed (a modal under offscreen has nothing to dismiss it)."""
    return any(os.environ.get(v) for v in (
        "ROBOMAGE_GUI_SMOKE", "ROBOMAGE_ANALYSIS_SMOKE",
        "ROBOMAGE_GUI_SESSION_SMOKE", "ROBOMAGE_GUI_TRACE_SMOKE"))


def _critical(parent, title, text):
    """QMessageBox.critical, or a stderr line under smoke (never block headless)."""
    if _smoke_active():
        print(f"[{title}] {text}", file=sys.stderr)
        return
    QMessageBox.critical(parent, title, text)


# ── Panes ─────────────────────────────────────────────────────────────────────

class WelcomePane(QWidget):
    """The empty-state central pane: app title + how to begin."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addStretch(2)
        title = QLabel("RoboMage")
        title.setObjectName("welcomeTitle")
        title.setAlignment(Qt.AlignCenter)
        hint = QLabel("File ▸ New Session to begin")
        hint.setObjectName("welcomeHint")
        hint.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)
        lay.addWidget(hint)
        lay.addStretch(3)


# ── New Analysis Session dialog ───────────────────────────────────────────────

class NewAnalysisSessionDialog(QDialog):
    """File ▸ New Session ▸ Analysis…: configure a simulated-games analysis
    session (model/opponent/decks/games/clock knobs), or tick "Browse recorded
    shards instead" to load recorded AZ self-play with no live engine.

    ``options()`` computes the opts dict live (callable without exec — the
    smoke constructs the dialog programmatically); OK validates, persists the
    fields to ~/.robomage/gui_analysis_launcher.json, and accepts."""

    def __init__(self, binary_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RoboMage — New Analysis Session")
        self.setModal(True)
        self._binary = binary_path
        cfg = _load_launcher_config(_ANALYSIS_LAUNCHER_CONFIG)
        decks = _scan_decks()

        def get(key):
            return cfg[key] if key in cfg else _ANALYSIS_DEFAULTS[key]

        # -- simulate group --------------------------------------------------
        form = QFormLayout()
        form.setSpacing(8)

        self._model = QComboBox()
        self._model.setEditable(True)
        self._model.addItems(_ANALYSIS_MODEL_PRESETS)
        self._model.setEditText(get("model") or "gen")
        self._model.setToolTip(
            "The model whose play is analyzed: 'gen' (the generalist), an "
            "az:/azraw:/mcts: search spec, or a checkpoint path. In shard "
            "mode this is the V(s) net (unless 'no net').")
        form.addRow("Model", self._model)

        self._opponent = QComboBox()
        self._opponent.setEditable(True)
        for label, spec in _OPPONENT_PRESETS:
            self._opponent.addItem(label, spec)
        NewPlaySessionDialog._set_combo_spec(self._opponent, get("opponent"))
        self._opponent.setToolTip(
            "Opponent controller for the simulated games: a scripted tier, "
            "'gen', a search spec, or a checkpoint path.")
        form.addRow("Opponent", self._opponent)

        self._deck_a = NewPlaySessionDialog._deck_combo(decks, get("deck_a"))
        self._deck_a.setToolTip("The model's deck.")
        form.addRow("Model deck", self._deck_a)

        self._deck_b = QComboBox()
        self._deck_b.setEditable(True)
        self._deck_b.addItem("")                 # blank = mirror the model deck
        self._deck_b.addItems(decks)
        cur_b = get("deck_b") or ""
        idx = self._deck_b.findText(cur_b)
        if idx >= 0:
            self._deck_b.setCurrentIndex(idx)
        else:
            self._deck_b.setEditText(cur_b)
        self._deck_b.setToolTip(
            "The opponent's deck; leave blank to mirror the model deck.")
        form.addRow("Opponent deck", self._deck_b)

        self._n_games = QSpinBox()
        self._n_games.setRange(0, 500)
        self._n_games.setValue(int(get("n_games") or 0))
        self._n_games.setToolTip("Games to simulate on startup.")
        form.addRow("Games", self._n_games)

        self._bo3 = QCheckBox("Best of three (with sideboarding)")
        self._bo3.setChecked(bool(get("bo3")))
        form.addRow(self._bo3)

        self._think_time = NewPlaySessionDialog._float_field(
            get("think_time"),
            "Wall-clock seconds per search decision (search specs only).")
        self._match_clock = NewPlaySessionDialog._float_field(
            get("match_clock"),
            "Whole-match thinking bank in seconds (chess clock; search specs "
            "only).")
        form.addRow("Think time (s)", self._think_time)
        form.addRow("Match clock (s)", self._match_clock)

        self._sim_box = QGroupBox("Simulate games")
        self._sim_box.setLayout(form)

        # -- shards group ----------------------------------------------------
        self._shards_box = QGroupBox("Browse recorded shards instead")
        self._shards_box.setCheckable(True)
        self._shards_box.setChecked(bool(get("shards_on")))
        self._shards_box.setToolTip(
            "Load recorded AZ self-play (a directory of shard_*.npz, e.g. "
            "train/az_data/gen) instead of simulating — no live engine; "
            "whatif/run stay disabled.")
        sform = QFormLayout()
        sform.setSpacing(8)
        shards_row = QHBoxLayout()
        self._shards_dir = QLineEdit(get("shards") or "")
        self._shards_dir.setToolTip("Directory of shard_*.npz files.")
        browse = QToolButton()
        browse.setText("…")
        browse.clicked.connect(self._browse_shards)
        shards_row.addWidget(self._shards_dir, 1)
        shards_row.addWidget(browse)
        sform.addRow("Shards dir", shards_row)
        self._seat = QComboBox()
        self._seat.addItems(["A", "B"])
        self._seat.setCurrentIndex(1 if get("seat") == "B" else 0)
        self._seat.setToolTip(
            "Viewpoint seat: that seat's searched decisions are the browsable "
            "steps.")
        sform.addRow("Seat", self._seat)
        self._no_net = QCheckBox("No value net — use recorded outcomes (torch-free)")
        self._no_net.setChecked(bool(get("no_net")))
        sform.addRow(self._no_net)
        self._shards_box.setLayout(sform)
        self._shards_box.toggled.connect(self._update_enabled)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Start analysis")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)

        lay = QVBoxLayout(self)
        title = QLabel("New Analysis Session")
        title.setObjectName("launcherTitle")
        subtitle = QLabel("Simulate games and browse them, or open recorded shards.")
        subtitle.setObjectName("launcherSubtitle")
        lay.addWidget(title)
        lay.addWidget(subtitle)
        lay.addWidget(self._sim_box)
        lay.addWidget(self._shards_box)
        lay.addWidget(buttons)
        self.setMinimumWidth(460)
        self._update_enabled()

    def _browse_shards(self):
        path = QFileDialog.getExistingDirectory(self, "Shards directory",
                                                self._shards_dir.text() or ".")
        if path:
            self._shards_dir.setText(path)

    def _update_enabled(self, *_):
        """Shard mode disables the sim fields (they don't apply)."""
        self._sim_box.setEnabled(not self._shards_box.isChecked())

    def options(self):
        """The opts dict the SessionManager/BrowserPane contract expects.
        Computed live so the dialog is inspectable without exec()."""
        shards_on = self._shards_box.isChecked()
        deck_b = self._deck_b.currentText().strip()
        return {
            "model": self._model.currentText().strip() or "gen",
            "opponent": NewPlaySessionDialog._combo_spec(self._opponent),
            "deck_a": self._deck_a.currentText().strip(),
            "deck_b": deck_b or None,
            "n_games": int(self._n_games.value()),
            "bo3": self._bo3.isChecked(),
            "think_time": NewPlaySessionDialog._spin_value(self._think_time),
            "match_clock": NewPlaySessionDialog._spin_value(self._match_clock),
            "shards": (self._shards_dir.text().strip() or None) if shards_on
                      else None,
            "seat": self._seat.currentText(),
            "no_net": self._no_net.isChecked(),
            "binary": self._binary,
        }

    def _on_accept(self):
        opts = self.options()
        if self._shards_box.isChecked():
            if not opts["shards"]:
                QMessageBox.warning(self, "Missing shards directory",
                                    "Pick the directory of shard_*.npz files.")
                return
        else:
            if not opts["deck_a"]:
                QMessageBox.warning(self, "Missing deck",
                                    "Pick the model's deck.")
                return
            if not opts["model"] or not opts["opponent"]:
                QMessageBox.warning(self, "Missing spec",
                                    "Pick a model and an opponent.")
                return
        _save_launcher_config({
            "model": opts["model"], "opponent": opts["opponent"],
            "deck_a": opts["deck_a"], "deck_b": opts["deck_b"] or "",
            "n_games": opts["n_games"], "bo3": opts["bo3"],
            "think_time": opts["think_time"],
            "match_clock": opts["match_clock"],
            "shards_on": self._shards_box.isChecked(),
            "shards": self._shards_dir.text().strip(),
            "seat": opts["seat"], "no_net": opts["no_net"],
        }, _ANALYSIS_LAUNCHER_CONFIG)
        self.accept()


# ── Session management ────────────────────────────────────────────────────────

class SessionManager(QObject):
    """Builds and tears down the window's ONE current session (play or
    analysis) and owns the Save/Open flows. Teardown-before-build: a new
    session never coexists with the old one's engine process."""

    def __init__(self, window, stack, welcome, binary_path):
        super().__init__(window)
        self._window = window
        self._stack = stack
        self._welcome = welcome
        self._binary = binary_path
        self.mode = None                     # None | "play" | "analysis"
        self.pane = None
        self._session = None                 # play mode: game_driver.Session
        self._opts = None                    # opts dict the session was built from
        self._save_path = None
        # Captured before teardown so run()'s exit-1 check survives the pane.
        self.analysis_smoke_ok = None

    # ----- build -----

    def new_play_session(self, opts, replay=None):
        """Build a play session from an options dict (NewPlaySessionDialog
        keys: binary, model_path, human_player, human_deck, model_deck, bo3,
        analysis, human_clock_s, hard_timeout; plus optional engine_seed).
        `replay` is a loaded .rmplay doc — its actions are fast-forwarded
        before the interactive loop."""
        if not self._teardown_current(confirm=True):
            return False
        session = None
        try:
            analysis_cfg = _analysis_cfg_from(opts.get("analysis"))
            session = build_session(
                opts["binary"], opts["model_path"],
                human_player=opts.get("human_player"),
                human_deck=opts["human_deck"], model_deck=opts["model_deck"],
                bo3=opts.get("bo3", True), analysis=analysis_cfg is not None,
                step_pacing=True, engine_seed=opts.get("engine_seed"),
                human_clock_s=opts.get("human_clock_s"),
                hard_timeout=bool(opts.get("hard_timeout", False)))
            session.analysis_cfg = analysis_cfg
            if opts.get("record_shards"):
                from shard_record import default_recording_dir
                session.record_dir = default_recording_dir()
            pane = PlayPane(session,
                            replay_actions=(replay or {}).get("actions"))
        except Exception as exc:                          # noqa: BLE001
            if session is not None:
                try:
                    session.env.close()
                except Exception:                          # noqa: BLE001
                    pass
            _critical(self._window, "Could not start session", str(exc))
            return False
        self._session = session
        self._opts = dict(opts)
        self._install(pane, "play")
        pane.session_finished.connect(self._on_session_finished)
        pane.start()
        self._window._sync_actions()
        return True

    def new_analysis_session(self, opts, traces=None):
        """Build an analysis (browser) session. `opts` is the BrowserPane
        contract dict (see NewAnalysisSessionDialog.options); `traces` is an
        optional (games, provenance) pair to preload (the Open path)."""
        try:
            from gui_browser import BrowserPane
        except ImportError:
            _critical(self._window, "Analysis browser not available",
                      "The analysis browser (train/gui_browser.py) is not "
                      "present in this build.")
            return False
        if not self._teardown_current(confirm=True):
            return False
        try:
            pane = BrowserPane(opts, parent=self._window)
        except Exception as exc:                          # noqa: BLE001
            _critical(self._window, "Could not start analysis session", str(exc))
            return False
        self._opts = dict(opts)
        self._install(pane, "analysis")
        try:
            pane.status.connect(self._window.statusBar().showMessage)
        except Exception:                                  # noqa: BLE001
            pass                                           # signal optional
        try:
            pane.start()
            if traces is not None:
                games, provenance = traces
                pane.load_traces(games, provenance)
        except Exception as exc:                          # noqa: BLE001
            _critical(self._window, "Analysis session failed", str(exc))
            self._teardown_current(confirm=False)
            return False
        self._window._sync_actions()
        return True

    def _install(self, pane, mode):
        self._stack.addWidget(pane)
        self._stack.setCurrentWidget(pane)
        self.pane = pane
        self.mode = mode
        self._save_path = None
        try:
            pane.set_active(True)
        except Exception:                                  # noqa: BLE001
            pass

    # ----- teardown -----

    def _should_confirm(self):
        if self.mode == "play":
            return not getattr(self.pane, "game_over", False)
        if self.mode == "analysis":
            try:
                return bool(self.pane.is_busy())
            except Exception:                              # noqa: BLE001
                return False
        return False

    def _teardown_current(self, confirm=True):
        """Tear the current session down (load-bearing order) and land on the
        welcome pane. Returns False if the user cancelled the confirm."""
        if self.pane is None:
            return True
        if (confirm and not _smoke_active() and self._should_confirm()):
            resp = QMessageBox.question(
                self._window, "End session?",
                "A session is in progress — end it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
            if resp != QMessageBox.StandardButton.Yes:
                return False
        pane, self.pane = self.pane, None
        mode, self.mode = self.mode, None
        session, self._session = self._session, None
        self._opts = None
        self._save_path = None
        if mode == "play":
            # Preserve the analysis smoke verdict across teardown (run() reads
            # it after the pane is gone).
            if getattr(pane, "_analysis", None) is not None:
                self.analysis_smoke_ok = bool(pane._analysis.smoke_ok)
            pane.shutdown()                  # filter → analysis → driver/join
            if session is not None:
                try:
                    session.env.close()      # the HOST closes the env, last
                except Exception:                          # noqa: BLE001
                    pass
        else:
            try:
                pane.shutdown()              # bounded-blocking per contract
            except Exception:                              # noqa: BLE001
                pass
        try:
            pane.set_active(False)
        except Exception:                                  # noqa: BLE001
            pass
        self._stack.removeWidget(pane)
        pane.deleteLater()
        self._stack.setCurrentWidget(self._welcome)
        self._window._sync_actions()
        return True

    def close_current(self, confirm=True):
        return self._teardown_current(confirm=confirm)

    def shutdown(self):
        """Full teardown with no confirm (quit / window close)."""
        self._teardown_current(confirm=False)

    def _on_session_finished(self):
        """PlayPane says the session is over. Under smoke: tear down and quit
        the app (run() then computes the exit code). Interactive: no-op — the
        game-over banner stays up and the menus drive what happens next."""
        if _smoke_n_from_env() is None:
            return
        self.shutdown()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    # ----- save -----

    def save(self, path=None):
        """File ▸ Save Session (Ctrl+S). Play mode → .rmplay replay spec;
        analysis mode → .rmtrace trace bundle. `path=None` prompts (or reuses
        the session's last save path)."""
        if self.pane is None:
            return False
        if path is None:
            path = self._save_path or self._ask_save_path()
            if not path:
                return False
        try:
            if self.mode == "play":
                self._save_play(path)
            else:
                gui_session_io.save_traces(path, self.pane.traces(),
                                           self.pane.provenance())
        except Exception as exc:                          # noqa: BLE001
            _critical(self._window, "Save failed", str(exc))
            return False
        self._save_path = path
        try:
            self._window.statusBar().showMessage(f"Saved {path}", 5000)
        except Exception:                                  # noqa: BLE001
            pass
        return True

    def save_as(self):
        path = self._ask_save_path()
        if not path:
            return False
        return self.save(path)

    def _ask_save_path(self):
        ext = (gui_session_io.PLAY_EXT if self.mode == "play"
               else gui_session_io.TRACE_EXT)
        kind = ("play session" if self.mode == "play" else "analysis session")
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save Session", "",
            f"RoboMage {kind} (*{ext})")
        if not path:
            return None
        if not os.path.splitext(path)[1]:
            path += ext
        return path

    def _save_play(self, path):
        pane, session = self.pane, self._session
        actions = list(pane.driver.action_log)   # UI-thread snapshot
        env = session.env
        human_is_a = not session.opp_is_a
        deck_a = session.human_deck if human_is_a else session.opp_deck
        deck_b = session.opp_deck if human_is_a else session.human_deck
        final_obs = getattr(env, "_obs", None)
        opts = self._opts or {}
        gui_session_io.save_replay(
            path,
            engine_seed=getattr(env, "last_engine_seed", None),
            actions=actions,
            deck_a=deck_a, deck_b=deck_b,
            human_deck=session.human_deck, opp_deck=session.opp_deck,
            human_is_a=human_is_a, bo3=session.bo3,
            opponent_spec=opts.get("model_path") or "scripted",
            analysis=opts.get("analysis"),
            binary=opts.get("binary") or self._binary,
            final_obs=(np.array(final_obs, dtype=np.float32)
                       if final_obs is not None else None),
            in_progress=not getattr(pane, "game_over", False))

    # ----- open -----

    def open_path(self, path):
        """File ▸ Open Session (Ctrl+O) routing: sniff the kind, then load."""
        kind = gui_session_io.sniff_kind(path)
        if kind == "play":
            return self._open_play(path)
        if kind == "trace":
            return self._open_trace(path)
        _critical(self._window, "Could not open file",
                  f"{path} is not a RoboMage session file.")
        return False

    def _ask_open_play_choice(self, doc):
        """'continue' | 'analyze' | None. A method so smokes can stub it."""
        n = len(doc.get("actions") or [])
        box = QMessageBox(self._window)
        box.setWindowTitle("Open Play Session")
        box.setText(f"Saved play session: {doc.get('human_deck')} vs "
                    f"{doc.get('opp_deck')} ({n} actions"
                    f"{', in progress' if doc.get('in_progress') else ''}).")
        cont = box.addButton("Continue playing", QMessageBox.ButtonRole.AcceptRole)
        analyze = box.addButton("Analyze this game…", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is cont:
            return "continue"
        if clicked is analyze:
            return "analyze"
        return None

    def _open_play(self, path):
        try:
            doc = gui_session_io.load_replay(path)
        except Exception as exc:                          # noqa: BLE001
            _critical(self._window, "Could not open session", str(exc))
            return False
        choice = self._ask_open_play_choice(doc)
        if choice is None:
            return False
        binary = doc.get("binary") or self._binary
        if choice == "analyze":
            # Replays the whole game on a fresh env — may take a moment.
            app = QApplication.instance()
            app.setOverrideCursor(Qt.WaitCursor)
            try:
                trace = gui_session_io.trace_from_replay(doc, binary)
            except Exception as exc:                      # noqa: BLE001
                _critical(self._window, "Could not analyze session", str(exc))
                return False
            finally:
                app.restoreOverrideCursor()
            provenance = {
                "model": None, "opponent": doc.get("opponent_spec"),
                "deck_a": doc.get("deck_a"), "deck_b": doc.get("deck_b"),
                "bo3": doc.get("bo3", True), "source": path,
            }
            opts = self._browse_only_opts(provenance, binary)
            return self.new_analysis_session(opts, traces=([trace], provenance))
        # Continue playing: rebuild the session at the saved seed/seat/decks
        # and fast-forward the saved action prefix.
        try:
            model_path = _resolve_opponent_spec(doc.get("opponent_spec")
                                                or "scripted")
        except Exception as exc:                          # noqa: BLE001
            _critical(self._window, "Opponent unavailable", str(exc))
            return False
        opts = dict(binary=binary, model_path=model_path,
                    human_player="A" if doc.get("human_is_a", True) else "B",
                    human_deck=doc["human_deck"], model_deck=doc["opp_deck"],
                    bo3=doc.get("bo3", True), analysis=doc.get("analysis"),
                    engine_seed=doc["engine_seed"])
        if self.new_play_session(opts, replay=doc):
            self._save_path = path
            return True
        return False

    def _open_trace(self, path):
        interp_fn = None
        try:
            import analysis as an
            interp_fn = an._extract_interpretable
        except Exception:                                  # noqa: BLE001
            pass                     # torch-free fallback: empty interp features
        try:
            games, meta = gui_session_io.load_traces(path, interp_fn=interp_fn)
        except Exception as exc:                          # noqa: BLE001
            _critical(self._window, "Could not open session", str(exc))
            return False
        provenance = meta.get("provenance") or {}
        opts = self._browse_only_opts(provenance, self._binary)
        if self.new_analysis_session(opts, traces=(games, provenance)):
            self._save_path = path
            return True
        return False

    @staticmethod
    def _browse_only_opts(provenance, binary):
        """A BrowserPane opts dict for a traces-only (no simulation) session.
        Deliberately carries NO engine keys (model/opponent/shards) — their
        presence is what makes BrowserPane build an engine worker, and an
        opened .rmtrace must never reload a model/env. The provenance dict
        (passed alongside via load_traces) keeps them for display."""
        return {
            "deck_a": provenance.get("deck_a"),
            "deck_b": provenance.get("deck_b"),
            "n_games": 0,
            "bo3": bool(provenance.get("bo3", True)),
            "think_time": None, "match_clock": None,
            "seat": provenance.get("seat", "A"),
            "no_net": False, "binary": binary,
        }


# ── Shard-browser window (Analyze Recording…) ────────────────────────────────

def _torch_available():
    import importlib.util
    return importlib.util.find_spec("torch") is not None


def _recording_value_spec(opponent_spec):
    """The V(s) model spec for browsing a recording, derived from the play
    session's opponent spec: strip the ?knob query; an mcts: wrapper searches
    with the PPO heads, so its value net is the bare PPO spec."""
    base = (opponent_spec or "az:gen").split("?", 1)[0].strip()
    low = base.lower()
    if low.startswith("mcts:"):
        return base[5:] or "gen"
    return base or "az:gen"


class ShardBrowserWindow(QMainWindow):
    """A SECOND top-level window hosting a shard-mode BrowserPane, so a live
    play session's recording can be analyzed WITHOUT tearing the game down
    (the SessionManager owns only the main window's one session). Reads the
    recording directory on open; reopen the window (View ▸ Analyze Recording…)
    to pick up decisions recorded since."""

    def __init__(self, opts, parent=None):
        super().__init__(parent)          # QMainWindow stays top-level
        from gui_browser import BrowserPane
        self.setWindowTitle(f"RoboMage — Recording · {opts.get('shards', '')}")
        self.resize(1180, 860)
        self._pane = BrowserPane(opts, parent=self)
        self.setCentralWidget(self._pane)
        try:
            self._pane.status.connect(self.statusBar().showMessage)
        except Exception:                                  # noqa: BLE001
            pass
        self._pane.start()

    def closeEvent(self, event):
        self._pane.shutdown()
        super().closeEvent(event)


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """The one app window: menu bar + QStackedWidget central (welcome / play /
    browser panes). Created once per process; sessions come and go through the
    SessionManager."""

    def __init__(self, binary_path):
        super().__init__()
        self._binary = binary_path
        self.setWindowTitle("RoboMage")
        self.resize(1280, 940)

        self._stack = QStackedWidget()
        self._welcome = WelcomePane()
        self._stack.addWidget(self._welcome)
        self.setCentralWidget(self._stack)
        self.manager = SessionManager(self, self._stack, self._welcome,
                                      binary_path)
        # Open Analyze-Recording browser windows (independent top-levels; they
        # outlive session switches and are closed with the app).
        self._shard_browsers = []

        self._build_menus()
        self.statusBar()                     # create it (styled by _QSS)
        # Enable/check state is cheap to recompute; a coarse timer covers the
        # signals we don't track individually (action_log growth, analysis
        # window visibility, browser trace arrivals).
        self._sync_timer = QTimer(self)
        self._sync_timer.timeout.connect(self._sync_actions)
        self._sync_timer.start(500)
        self._sync_actions()

    # ----- menus -----

    def _build_menus(self):
        bar = self.menuBar()

        filem = bar.addMenu("&File")
        newm = filem.addMenu("New Session")
        self._act_new_play = self._action(
            "Play…", self._new_play_dialog, "Ctrl+N")
        self._act_new_analysis = self._action(
            "Analysis…", self._new_analysis_dialog, "Ctrl+Shift+N")
        newm.addAction(self._act_new_play)
        newm.addAction(self._act_new_analysis)
        self._act_open = self._action("Open Session…", self._open_dialog,
                                      "Ctrl+O")
        filem.addAction(self._act_open)
        self._act_save = self._action("Save Session",
                                      lambda: self.manager.save(), "Ctrl+S")
        self._act_save_as = self._action("Save Session As…",
                                         self.manager.save_as, "Ctrl+Shift+S")
        filem.addAction(self._act_save)
        filem.addAction(self._act_save_as)
        filem.addSeparator()
        self._act_quit = self._action("Quit", self._quit, "Ctrl+Q")
        filem.addAction(self._act_quit)

        # Game: in-game actions on the live play session (empty of meaning in
        # any other mode, so every entry is disabled there by _sync_actions).
        gamem = bar.addMenu("&Game")
        self._act_concede_game = self._action(
            "Concede Game", lambda: self._concede(False))
        self._act_concede_match = self._action(
            "Concede Match", lambda: self._concede(True))
        gamem.addAction(self._act_concede_game)
        gamem.addAction(self._act_concede_match)

        viewm = bar.addMenu("&View")
        self._act_analysis_win = self._action(
            "Analysis Window", self._toggle_analysis_window, "F9")
        self._act_analysis_win.setCheckable(True)
        viewm.addAction(self._act_analysis_win)
        self._act_analyze_rec = self._action(
            "Analyze Recording…", self._analyze_recording, "F10")
        viewm.addAction(self._act_analyze_rec)
        # No shortcuts on the log-width entries: PlayPane keeps its own
        # widget-scoped </> shortcuts, so menu shortcuts would double-fire.
        self._act_widen = self._action(
            "Widen Log  ( > )", lambda: self._nudge_log(1))
        self._act_narrow = self._action(
            "Narrow Log  ( < )", lambda: self._nudge_log(-1))
        viewm.addAction(self._act_widen)
        viewm.addAction(self._act_narrow)

        helpm = bar.addMenu("&Help")
        helpm.addAction(self._action("Keyboard Reference", self._show_keys))

    def _action(self, text, slot, shortcut=None):
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        return act

    def _sync_actions(self):
        """Recompute every menu action's enable/check state from the current
        session (called on session changes and by the coarse timer)."""
        mode, pane = self.manager.mode, self.manager.pane
        save_ok = False
        if pane is not None:
            if mode == "play":
                save_ok = bool(pane.driver.action_log)
            elif mode == "analysis":
                try:
                    save_ok = bool(pane.traces())
                except Exception:                          # noqa: BLE001
                    save_ok = False
        self._act_save.setEnabled(save_ok)
        self._act_save_as.setEnabled(save_ok)

        has_analysis = (mode == "play" and pane is not None
                        and pane._analysis is not None)
        self._act_analysis_win.setEnabled(has_analysis)
        self._act_analysis_win.setChecked(
            bool(has_analysis and pane._analysis.isVisible()))
        # Concede: only on a live play session, and only for a scope that is
        # still meaningful (single game => "Concede Match" is the same action,
        # so the pane reports it unavailable; both go dead at game over).
        for act, is_match in ((self._act_concede_game, False),
                              (self._act_concede_match, True)):
            act.setEnabled(mode == "play" and pane is not None
                           and pane.can_concede(is_match))
        # Single game: the one entry conceding it IS conceding the match — say
        # so rather than leaving a misleading "Concede Game" label.
        one_game = (mode == "play" and pane is not None
                    and not getattr(pane, "_bo3", True))
        self._act_concede_game.setText(
            "Concede Game (= the match)" if one_game else "Concede Game")

        rec = (getattr(pane, "recorder", None)
               if mode == "play" and pane is not None else None)
        self._act_analyze_rec.setEnabled(
            bool(rec is not None and rec.rows_recorded))
        for act in (self._act_widen, self._act_narrow):
            act.setEnabled(mode == "play")

        title = "RoboMage"
        if pane is not None:
            try:
                title = pane.title() or title
            except Exception:                              # noqa: BLE001
                pass
        if title != self.windowTitle():
            self.setWindowTitle(title)

    # ----- menu slots -----

    def _new_play_dialog(self):
        dlg = NewPlaySessionDialog(self._binary, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.manager.new_play_session(dlg.options())

    def _new_analysis_dialog(self):
        dlg = NewAnalysisSessionDialog(self._binary, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self.manager.new_analysis_session(dlg.options())

    def _open_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "",
            f"RoboMage sessions (*{gui_session_io.PLAY_EXT} "
            f"*{gui_session_io.TRACE_EXT});;All files (*)")
        if path:
            self.manager.open_path(path)

    def _analyze_recording(self):
        """View ▸ Analyze Recording… (F10): flush the play session's shard
        recorder and open its directory in a shard-mode browser window — a
        SECOND window, so the game keeps running (mid-game analysis). Rows of
        the game in progress are included (their z is 0 until it ends); reopen
        to pick up newer decisions."""
        pane = self.manager.pane
        rec = (getattr(pane, "recorder", None)
               if self.manager.mode == "play" else None)
        if rec is None:
            return
        if not rec.rows_recorded:
            self.statusBar().showMessage("No decisions recorded yet.", 5000)
            return
        rec.flush()
        session = self.manager._session
        opts = {
            "shards": rec.out_dir,
            "seat": "A" if session.opp_is_a else "B",     # the opponent's rows
            "model": _recording_value_spec(
                (self.manager._opts or {}).get("model_path")),
            "no_net": not _torch_available(),
            "n_games": 0,                                  # load every match
            "binary": self._binary,
        }
        try:
            win = ShardBrowserWindow(opts, parent=self)
        except Exception as exc:                          # noqa: BLE001
            _critical(self, "Could not open recording", str(exc))
            return
        self._shard_browsers.append(win)
        win.show()

    def _close_shard_browsers(self):
        browsers, self._shard_browsers = self._shard_browsers, []
        for win in browsers:
            try:
                win.close()
            except Exception:                              # noqa: BLE001
                pass

    def _concede(self, match):
        """Game ▸ Concede Game / Concede Match. The pane owns the confirmation
        dialog and the driver call; it returns False when it declined (not a
        play session, already over, or the human cancelled)."""
        pane = self.manager.pane
        if self.manager.mode != "play" or pane is None:
            return
        if pane.concede(match=match):
            self.statusBar().showMessage(
                "Match conceded." if match else "Game conceded.", 5000)
        self._sync_actions()

    def _toggle_analysis_window(self, checked):
        pane = self.manager.pane
        if (self.manager.mode != "play" or pane is None
                or pane._analysis is None):
            return
        if checked:
            pane._analysis.show()
        else:
            pane._analysis.hide()

    def _nudge_log(self, delta):
        pane = self.manager.pane
        if self.manager.mode == "play" and pane is not None:
            pane._nudge_splitter(delta)

    def _show_keys(self):
        QMessageBox.information(self, "Keyboard Reference", _KEYS_TEXT)

    def _quit(self):
        # Confirm-if-active happens inside the teardown (skipped under smoke).
        if not self.manager.close_current(confirm=True):
            return
        self._close_shard_browsers()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event):
        # Window close = quit: full teardown order, env closed by the manager.
        self._close_shard_browsers()
        self.manager.shutdown()
        super().closeEvent(event)


# ── Entry points ──────────────────────────────────────────────────────────────

def run(binary_path, model_path, human_player=None,
        human_deck="delver", model_deck="delver", bo3=True, analysis=False,
        record_shards=False, human_clock_s=None, hard_timeout=False):
    """play.py --gui entry: MainWindow + a play session built directly (no
    dialog). Same signature/semantics as tui_game.run. Returns the exit code;
    the ROBOMAGE_ANALYSIS_SMOKE / record-shards smoke exit-1 checks live
    here.

    `human_clock_s` / `hard_timeout` are play.py's --human-clock /
    --hard-timeout (the launcher dialog has its own fields for them)."""
    app = _ensure_app()
    window = MainWindow(binary_path)
    window.show()
    opts = dict(binary=binary_path, model_path=model_path,
                human_player=human_player, human_deck=human_deck,
                model_deck=model_deck, bo3=bo3,
                analysis=({} if analysis else None),
                record_shards=record_shards,
                human_clock_s=human_clock_s, hard_timeout=hard_timeout)
    if not window.manager.new_play_session(opts):
        return 1
    # Capture the recorder before the smoke path tears the pane down.
    recorder = getattr(window.manager.pane, "recorder", None)
    app.exec()
    window.manager.shutdown()                # idempotent safety net
    if (os.environ.get("ROBOMAGE_ANALYSIS_SMOKE")
            and not window.manager.analysis_smoke_ok):
        print("ANALYSIS SMOKE FAILED: no analysis stats arrived",
              file=sys.stderr)
        return 1
    if record_shards and _smoke_active():
        import glob as _glob
        n_files = (len(_glob.glob(os.path.join(recorder.out_dir,
                                               "shard_*.npz")))
                   if recorder is not None else 0)
        if recorder is None or not recorder.rows_recorded or not n_files:
            print("RECORD SMOKE FAILED: no shard rows were recorded",
                  file=sys.stderr)
            return 1
        print(f"RECORD SMOKE OK: {recorder.rows_recorded} rows in "
              f"{n_files} shard(s) -> {recorder.out_dir}", flush=True)
    return 0


def run_launcher(binary_path=None):
    """gui.sh / no-arguments entry: MainWindow on the welcome pane with the
    menus live — no dialog is auto-opened; start via File ▸ New Session
    (Ctrl+N play, Ctrl+Shift+N analysis). Returns 0 on a clean exit.

    The headless shell smokes hijack this entry: set
    ROBOMAGE_GUI_SESSION_SMOKE / ROBOMAGE_GUI_TRACE_SMOKE /
    ROBOMAGE_BROWSER_SMOKE and run `python train/gui_main.py`."""
    from cli_spec import INTERACTIVE_BINARY
    binary_path = binary_path or INTERACTIVE_BINARY
    app = _ensure_app()
    window = MainWindow(binary_path)
    window.show()
    _start_shell_smoke(window)
    code = app.exec()
    window.manager.shutdown()
    return code


# ── Headless shell smokes ─────────────────────────────────────────────────────
# Each is a QTimer-driven state machine over the REAL SessionManager/menu
# flows (no test doubles), mirroring PlayPane's ROBOMAGE_GUI_SMOKE pattern.
# They app.exit(0/1) themselves; run_launcher returns that code.

class _ShellSmoke(QObject):
    """Base: a polling driver with a hard deadline."""

    def __init__(self, window, timeout_s=240):
        super().__init__(window)
        self.window = window
        self.manager = window.manager
        self._deadline = time.monotonic() + timeout_s
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _finish(self, code, msg):
        self._timer.stop()
        print(msg, flush=True)
        self.manager.shutdown()
        QApplication.instance().exit(code)

    def _poll(self):
        if time.monotonic() > self._deadline:
            self._finish(1, f"{type(self).__name__} FAILED: timeout")
            return
        try:
            self.step()
        except Exception:                                  # noqa: BLE001
            traceback.print_exc()
            self._finish(1, f"{type(self).__name__} FAILED: exception")

    def step(self):
        raise NotImplementedError


class _SessionSmoke(_ShellSmoke):
    """Play a few scripted decisions → Save .rmplay → Open (Continue) → verify
    the saved prefix replayed and the session is live again."""

    DECISIONS = 6

    def __init__(self, window):
        super().__init__(window)
        self._phase = "start"
        self._path = os.path.join(tempfile.mkdtemp(), "session" +
                                  gui_session_io.PLAY_EXT)
        self._saved = None

    def step(self):
        mgr = self.manager
        if self._phase == "start":
            opts = dict(binary=mgr._binary, model_path=None, human_player="A",
                        human_deck="league/ur_delver",
                        model_deck="league/gw_maverick",
                        bo3=False, analysis=None)
            if not mgr.new_play_session(opts):
                self._finish(1, "SESSION SMOKE FAILED: could not build")
                return
            self._phase = "play"
            return
        pane = mgr.pane
        if self._phase == "play":
            if pane is None:
                return
            if len(pane.driver.action_log) >= self.DECISIONS:
                if not mgr.save(self._path) or not os.path.exists(self._path):
                    self._finish(1, "SESSION SMOKE FAILED: save")
                    return
                self._saved = gui_session_io.load_replay(self._path)
                mgr._ask_open_play_choice = lambda doc: "continue"
                if not mgr.open_path(self._path):
                    self._finish(1, "SESSION SMOKE FAILED: open")
                    return
                self._phase = "verify"
                return
            # Drive the game: play the first legal action whenever it's the
            # human's turn (scripted opponent acts on its own).
            if getattr(pane, "_awaiting", False) and pane._actions:
                pane._submit(0)
            return
        if self._phase == "verify":
            if pane is None or mgr.mode != "play":
                return
            n = len(self._saved["actions"])
            log = list(pane.driver.action_log)
            if len(log) < n:
                return                       # replay still fast-forwarding
            if log[:n] != self._saved["actions"]:
                self._finish(1, "SESSION SMOKE FAILED: replayed prefix differs")
                return
            self._finish(0, f"SESSION SMOKE OK: replayed {n} actions, "
                            f"log now {len(log)}")


class _TraceSmoke(_ShellSmoke):
    """Open a synthetic .rmtrace (torch-free) and verify the browser lands in
    browse-only mode with the right game count."""

    GAMES = 2

    def __init__(self, window):
        super().__init__(window, timeout_s=120)
        import numpy as np
        from env import OBS_SIZE
        games = []
        for i in range(self.GAMES):
            games.append({
                "observations": [np.zeros(OBS_SIZE, dtype=np.float32)],
                "values": [0.25 * (1 - 2 * i)], "interp_features": [],
                "actions": [0], "num_choices": [1], "action_probs": None,
                "opp_actions": [], "engine_seed": None, "full_actions": None,
                "prefix_len": [0], "result": 1.0 - 2.0 * i, "model_is_a": True,
                "clock_remaining": [None], "clock_bank": None,
                "opp_clock_bank": None})
        self._path = os.path.join(tempfile.mkdtemp(), "traces" +
                                  gui_session_io.TRACE_EXT)
        gui_session_io.save_traces(self._path, games,
                                   provenance={"model": "smoke"})
        self._opened = False

    def step(self):
        mgr = self.manager
        if not self._opened:
            self._opened = True
            if not mgr.open_path(self._path):
                self._finish(1, "TRACE SMOKE FAILED: open")
            return
        pane = mgr.pane
        if pane is None or mgr.mode != "analysis":
            return
        n = len(pane.traces())
        if n == self.GAMES and not pane.is_busy():
            self._finish(0, f"TRACE SMOKE OK: {n} games")
        elif n > self.GAMES:
            self._finish(1, f"TRACE SMOKE FAILED: {n} games")


class _BrowserSmoke(_ShellSmoke):
    """Shard-mode browser session (torch-free, --no-net) driving the pane's
    own ROBOMAGE_BROWSER_SMOKE hook. Self-skips when no shards exist."""

    def __init__(self, window, shards=None):
        super().__init__(window)
        self._shards = shards or os.environ.get(
            "ROBOMAGE_BROWSER_SMOKE_SHARDS",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "az_data", "gen"))
        self._started = False

    def step(self):
        mgr = self.manager
        if not self._started:
            self._started = True
            import glob as _glob
            if not _glob.glob(os.path.join(self._shards, "shard_*.npz")):
                self._finish(0, f"BROWSER SMOKE SKIP: no shards in "
                                f"{self._shards}")
                return
            opts = {"shards": self._shards, "seat": "A", "no_net": True,
                    "n_games": 3, "bo3": False, "think_time": None,
                    "match_clock": None, "deck_a": None, "deck_b": None,
                    "binary": mgr._binary}
            if not mgr.new_analysis_session(opts):
                self._finish(1, "BROWSER SMOKE FAILED: could not build")
                return
            mgr.pane.smoke_done.connect(
                lambda n: self._finish(0 if n > 0 else 1,
                                       f"BROWSER SMOKE DONE: {n} games"))


def _start_shell_smoke(window):
    """Instantiate the shell smoke selected by env vars (None = interactive)."""
    if os.environ.get("ROBOMAGE_GUI_SESSION_SMOKE") == "1":
        return _SessionSmoke(window)
    if os.environ.get("ROBOMAGE_GUI_TRACE_SMOKE") == "1":
        return _TraceSmoke(window)
    if os.environ.get("ROBOMAGE_BROWSER_SMOKE") == "1":
        return _BrowserSmoke(window)
    return None


if __name__ == "__main__":
    sys.exit(run_launcher())
