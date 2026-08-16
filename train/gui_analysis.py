"""The analysis window: live MCTS evaluation of the current decision (Qt).

A separate top-level window beside the GameWindow board. On demand (or
automatically at each new decision, when the auto box is ticked) it runs an
``mcts.IncrementalSearch`` over the human's current decision on a DETACHED
analysis engine (see train/analysis_session.py) and renders a per-action table
— prior, visits, visit share, and Q as a win% — refreshed after every chunk,
chess-engine style, until the sims cap, the Stop button, or the human acting.

"Opponent's last decision" (F6, reveal toggle on) reviews a move already made:
the analysis engine rewinds to the position just before the opponent's most
recent real decision and searches it from their perspective, with the move they
played marked ▶. The live game keeps running throughout — the side engine
catches back up by forward replay at the next analysis.

Threading: one persistent worker thread (`AnalysisWorker`) owns the
AnalysisSession (and therefore the analysis engine + evaluator); the UI talks
to it through a command queue + a stop event, and results come back through
queued Qt signals (`AnalysisBridge`), mirroring the DriverBridge pattern.

Headless smoke: ``ROBOMAGE_ANALYSIS_SMOKE=1`` (with the GUI smoke driver, see
gui_game.py) forces a uniform evaluator, auto-analyzes the first analyzable
human decision, and records success once a chunk of stats arrives — gui_game's
smoke exit path checks it.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QLabel,
                               QMainWindow, QPushButton, QSlider, QSpinBox,
                               QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

import decode
from analysis_session import (AnalysisError, AnalysisRequest, AnalysisSession,
                              analyze_refusal)
from env import STATE_SIZE, _SELF_IS_A_IDX
from game_driver import decode_human_frame, menu_label

# How deep a principal variation is read out / walked, and how many branches
# the chart draws (top actions by visits).
PV_MAX_LEN = 16
MAX_BRANCHES = 4

# Branch line colors (chart + table-row emphasis). Distinct hues on the dark
# board palette; index parallels the sorted top-branch order.
_BRANCH_COLORS = ("#d8a12a", "#33d17a", "#3a9ad9", "#e05555")

# Controller wording for an OPPONENT-perspective menu. decode renders an obs in
# its own mover's frame, so on the opponent's decision THEIR permanents come out
# tagged "own" and yours "opp" — which reads backwards next to a board that is
# always drawn in your frame ("Target Thespian's Stage (own)" is THEIR Stage).
# Spell the possessives out so the row is unambiguous either way; the phrase
# pairs do the same for decode's metadata-only player-target wording. Both are
# defined in decode.py alongside the other controller-label vocabularies.
_OPP_CTRL_LABELS = decode.OPP_CTRL_LABELS
_OPP_PHRASES = decode.OPP_PHRASES


def opp_menu_labels(obs, num_choices) -> dict:
    """Action-index -> label for an OPPONENT-perspective decision, worded so a
    human reading it cannot mistake whose permanent each choice names."""
    menu = decode.decode_actions_from_obs(obs, num_choices,
                                          labels=_OPP_CTRL_LABELS)
    labels = {}
    for a in menu:
        desc = a["description"]
        for old, new in _OPP_PHRASES:
            desc = desc.replace(old, new)
        labels[a["index"]] = desc
    return labels


def _win_pct(v: float) -> str:
    """A [-1,1] mover-perspective value as a win percentage."""
    return f"{(v + 1.0) * 50.0:.1f}%"


def _req_step(req) -> str:
    """The game step an AnalysisRequest sits at, for status text."""
    try:
        return decode.decode_game_state(req.obs[:STATE_SIZE])["step"]
    except Exception:  # noqa: BLE001 — status text, never worth crashing for
        return "?"


@dataclass
class BranchSeries:
    """Expected-value trajectory of one root action, per determinized world.

    per_world[w] is the [0,1] win-probability (root-mover perspective) at each
    PV depth of world w's tree (variable length — a world's PV ends where its
    tree thins out); mean[d] averages the worlds that reach depth d.
    pv_actions[w] is the env-index action list for walking world w's line."""
    action: int
    per_world: list
    mean: list
    pv_actions: dict


def _branch_series(session: AnalysisSession, action: int,
                   worlds: int) -> BranchSeries:
    """Read one root action's per-world PV value series from the parked
    search's trees (tree-only — no engine interaction)."""
    per_world, pv_actions = [], {}
    for w in range(worlds):
        pv = session.pv(action, w, max_len=PV_MAX_LEN)
        per_world.append([(s.q + 1.0) / 2.0 for s in pv])
        pv_actions[w] = [s.action for s in pv]
    depth = max((len(s) for s in per_world), default=0)
    mean = []
    for d in range(depth):
        vals = [s[d] for s in per_world if len(s) > d]
        mean.append(sum(vals) / len(vals))
    return BranchSeries(action=action, per_world=per_world, mean=mean,
                        pv_actions=pv_actions)


def _walk_step_labels(root_labels, nodes, pv_actions):
    """Human-readable label for each walked step. Step 0's action comes from
    the REAL decision menu (root_labels); step i>0's menu is decoded from the
    previous walked node's obs (synthesized descriptions — the narrative
    side-channel doesn't exist inside a simulation)."""
    labels = []
    for i in range(len(nodes)):
        a = pv_actions[i]
        if i == 0:
            labels.append(root_labels.get(a, f"action {a}"))
            continue
        prev = nodes[i - 1]
        try:
            menu = decode.decode_actions_from_obs(prev.obs, prev.num_choices)
            labels.append(menu[a]["description"] if a < len(menu)
                          else f"action {a}")
        except Exception:  # noqa: BLE001 — a label, never worth crashing for
            labels.append(f"action {a}")
    return labels


# ── Worker → UI bridge ────────────────────────────────────────────────────────

class AnalysisBridge(QObject):
    """Queued-signal bridge from the analysis worker thread to the window."""

    stats_update = Signal(object)   # (tag, LiveStats) after every chunk
    status = Signal(object)         # (state_key, text) for the status line
    evaluator_note = Signal(str)    # evaluator caveat after a run ("" = none)
    run_done = Signal(object)       # (tag, LiveStats|None) when a run ends
    branches_ready = Signal(object)  # (tag, [BranchSeries]) after a run
    walk_ready = Signal(object)     # (tag, action, world, [WalkNode], labels)
    opp_result = Signal(object)     # (obs, num, SearchResult, chosen) from the
    #                                 opponent SearchController's on_result hook


class AnalysisWorker(threading.Thread):
    """The one thread that touches the AnalysisSession (engine + evaluator).

    Commands (via the queue): ("analyze", tag, AnalysisRequest, stop_event),
    ("respawn",), ("shutdown",). Each analyze run gets its OWN stop event —
    `cancel()` sets the latest run's event, so a cancel that lands before the
    worker even dequeues that run still sticks (a shared cleared-at-start
    event would swallow it)."""

    def __init__(self, primary_env, cfg, bridge: AnalysisBridge):
        super().__init__(name="gui-analysis", daemon=True)
        self._bridge = bridge
        self._session = AnalysisSession(primary_env, cfg)
        self._commands: queue.Queue = queue.Queue()
        self._current_stop: threading.Event | None = None

    # ----- UI-thread API -----

    def submit_analyze(self, tag: int, req: AnalysisRequest) -> None:
        self.cancel()
        stop = threading.Event()
        self._current_stop = stop
        self._commands.put(("analyze", tag, req, stop))

    def submit_respawn(self) -> None:
        self.cancel()
        self._commands.put(("respawn",))

    def submit_branches(self, tag: int, actions: list, worlds: int) -> None:
        """Read PV value series for the given root actions from the parked
        search (queued behind any running command)."""
        self._commands.put(("branches", tag, actions, worlds))

    def submit_walk(self, tag: int, action: int, world: int, pv_actions: list,
                    root_labels: dict) -> None:
        """Walk one branch's PV on the analysis engine and return the states."""
        self._commands.put(("walk", tag, action, world, pv_actions,
                            root_labels))

    def cancel(self) -> None:
        stop = self._current_stop
        if stop is not None:
            stop.set()

    def shutdown(self) -> None:
        self.cancel()
        self._commands.put(("shutdown",))

    # ----- worker thread -----

    def run(self) -> None:
        while True:
            cmd = self._commands.get()
            if cmd[0] == "shutdown":
                break
            if cmd[0] == "respawn":
                self._session.respawn()
                self._bridge.status.emit(("idle", "analysis engine respawned"))
                continue
            if cmd[0] == "branches":
                self._run_branches(*cmd[1:])
                continue
            if cmd[0] == "walk":
                self._run_walk(*cmd[1:])
                continue
            _, tag, req, stop = cmd
            self._run_analyze(tag, req, stop)
        self._session.close()

    def _run_branches(self, tag: int, actions: list, worlds: int) -> None:
        try:
            series = [_branch_series(self._session, a, worlds) for a in actions]
            self._bridge.branches_ready.emit((tag, series))
        except AnalysisError:
            pass    # search already closed by a newer sync — stale request

    def _run_walk(self, tag: int, action: int, world: int, pv_actions: list,
                  root_labels: dict) -> None:
        try:
            nodes = self._session.walk(world, pv_actions)
            labels = _walk_step_labels(root_labels, nodes, pv_actions)
            self._bridge.walk_ready.emit((tag, action, world, nodes, labels))
        except AnalysisError as e:
            self._bridge.status.emit(("error", f"walk failed: {e}"))

    def _run_analyze(self, tag: int, req: AnalysisRequest,
                     stop: threading.Event) -> None:
        bridge = self._bridge
        if self._session._evaluator is None:
            bridge.status.emit(("busy", "loading evaluator…"))
            try:
                self._session._ensure_evaluator()
            except Exception as e:  # noqa: BLE001 — bad spec/missing checkpoint
                bridge.status.emit(("error", f"evaluator failed: {e}"))
                bridge.run_done.emit((tag, None))
                return
        if self._session._env is None:
            bridge.status.emit(("busy", "preparing analysis engine…"))
        try:
            stats = self._session.analyze(
                req, stop=stop,
                on_update=lambda s: bridge.stats_update.emit((tag, s)))
            bridge.evaluator_note.emit(self._session.evaluator_note() or "")
            state = "stopped" if stop.is_set() else "done"
            bridge.status.emit((state,
                                f"{'stopped' if state == 'stopped' else 'done'}"
                                f" — {stats.sims_run} sims"))
            bridge.run_done.emit((tag, stats))
        except AnalysisError as e:
            bridge.status.emit(("error", str(e)))
            bridge.run_done.emit((tag, None))


# ── Branch value chart ────────────────────────────────────────────────────────

class BranchValueChart(QWidget):
    """Expected win% along each top branch's principal variation.

    One bold mean line per branch plus thin translucent per-world lines (the
    determinization spread) in the same hue. x = PV depth (decisions into the
    hypothetical future), y = win% for the root mover. Clicking selects a depth
    (drives the scrubber); the selected branch paints thicker."""

    depth_selected = Signal(int)

    _MARGIN_L, _MARGIN_R, _MARGIN_T, _MARGIN_B = 34, 8, 6, 16

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series: list = []       # [BranchSeries] in draw order
        self._selected = None         # selected branch action, or None
        self._cursor = None           # selected depth, or None
        self.setMinimumHeight(140)

    def set_series(self, series, selected_action=None):
        self._series = list(series or [])
        self._selected = selected_action
        self._cursor = None
        self.update()

    def set_selected(self, action):
        self._selected = action
        self.update()

    def set_cursor(self, depth):
        self._cursor = depth
        self.update()

    def _max_depth(self):
        return max((len(s.mean) for s in self._series), default=0)

    def _x(self, depth, max_depth):
        plot_w = self.width() - self._MARGIN_L - self._MARGIN_R
        step = plot_w / max(1, max_depth - 1)
        return self._MARGIN_L + depth * step

    def _y(self, win):
        plot_h = self.height() - self._MARGIN_T - self._MARGIN_B
        return self._MARGIN_T + (1.0 - win) * plot_h

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#0c0c10"))
        max_depth = self._max_depth()

        # Frame + the 50% line + y labels.
        p.setPen(QPen(QColor("#2a2a33"), 1))
        p.drawRect(self.rect().adjusted(0, 0, -1, -1))
        for win, label in ((1.0, "100%"), (0.5, "50%"), (0.0, "0%")):
            y = self._y(win)
            p.setPen(QPen(QColor("#2a2a33"), 1,
                          Qt.DashLine if win == 0.5 else Qt.SolidLine))
            p.drawLine(int(self._MARGIN_L), int(y),
                       int(self.width() - self._MARGIN_R), int(y))
            p.setPen(QColor("#6a6a72"))
            p.drawText(2, int(y) + 4, label)
        if not self._series or max_depth == 0:
            p.setPen(QColor("#6a6a72"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "run an analysis to see branch lines")
            return

        # Cursor depth marker under the lines.
        if self._cursor is not None:
            x = self._x(self._cursor, max_depth)
            p.setPen(QPen(QColor("#4a4a55"), 1))
            p.drawLine(int(x), self._MARGIN_T, int(x),
                       self.height() - self._MARGIN_B)

        for i, s in enumerate(self._series):
            color = QColor(_BRANCH_COLORS[i % len(_BRANCH_COLORS)])
            world_color = QColor(color)
            world_color.setAlpha(60)
            p.setPen(QPen(world_color, 1))
            for series_w in s.per_world:
                self._draw_line(p, series_w, max_depth)
            is_sel = (s.action == self._selected)
            p.setPen(QPen(color, 3 if is_sel else 2))
            self._draw_line(p, s.mean, max_depth)
        p.end()

    def _draw_line(self, p, values, max_depth):
        if len(values) == 1:
            x, y = self._x(0, max_depth), self._y(values[0])
            p.drawLine(int(x) - 2, int(y), int(x) + 2, int(y))
            return
        for d in range(len(values) - 1):
            p.drawLine(int(self._x(d, max_depth)), int(self._y(values[d])),
                       int(self._x(d + 1, max_depth)),
                       int(self._y(values[d + 1])))

    def mousePressEvent(self, event):
        max_depth = self._max_depth()
        if max_depth <= 0:
            return
        plot_w = self.width() - self._MARGIN_L - self._MARGIN_R
        step = plot_w / max(1, max_depth - 1)
        depth = round((event.position().x() - self._MARGIN_L) / max(1e-6, step))
        depth = max(0, min(max_depth - 1, int(depth)))
        self.depth_selected.emit(depth)


# ── Hypothetical-position mini board ─────────────────────────────────────────

class MiniBoard(QWidget):
    """Read-only compact board for a walked (hypothetical) state: a header
    line (life totals + step) over opponent-battlefield / self-battlefield /
    hand card rows, reusing the main board's CardWidget painter and Scryfall
    art. Hidden-hand rule matches the live board: when the walked node is the
    opponent's perspective, its private hand is not rendered."""

    def __init__(self, opp_is_a, parent=None):
        super().__init__(parent)
        # gui_game is fully imported before this window exists (GameWindow
        # constructs AnalysisWindow), so importing its widgets here is safe.
        from gui_game import CardRow, ImageProvider

        self._opp_is_a = opp_is_a
        self._reveal = False                  # show the opponent's hidden hand
        self._provider = ImageProvider()
        self._provider.image_ready.connect(self._refresh_images)
        self._widgets = {}                    # name -> [CardWidget]
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self._header = QLabel("")
        v.addWidget(self._header)
        self._opp_row = CardRow(height=96, min_height=60)
        self._self_row = CardRow(height=96, min_height=60)
        self._hand_row = CardRow(height=96, min_height=60)
        for row in (self._opp_row, self._self_row, self._hand_row):
            v.addWidget(row)

    def set_reveal(self, reveal):
        self._reveal = bool(reveal)

    def clear(self, text=""):
        self._header.setText(text)
        self._widgets = {}
        for row in (self._opp_row, self._self_row, self._hand_row):
            row.set_cards([])

    def show_terminal(self, terminal):
        won = (terminal == "A") != self._opp_is_a
        text = ("Game ends: draw" if terminal == "DRAW"
                else f"Game ends: you {'win' if won else 'lose'}")
        self.clear(text)

    def show_node(self, obs):
        """Render one walked node's obs (a private copy, any perspective)."""
        node_is_a = bool(obs[_SELF_IS_A_IDX] > 0.5)
        u = SimpleNamespace(obs=obs,
                            opp_perspective=(node_is_a == self._opp_is_a),
                            perm_counters=None, perm_token_names=None)
        gs, mirrored = decode_human_frame(u)
        self._header.setText(
            f"YOU ♥ {gs['self']['life']}   OPP ♥ {gs['opponent']['life']}   "
            f"{gs['step']}")
        self._widgets = {}
        self._opp_row.set_cards(
            [self._mk_card(p, "opp") for p in gs["opp_battlefield"]])
        self._self_row.set_cards(
            [self._mk_card(p, "self") for p in gs["self_battlefield"]])
        # A mirrored node's "self_hand" is the OPPONENT's private hand —
        # rendered only in reveal mode (the walk itself is already full-info
        # for that seat), hidden otherwise like the live board.
        if mirrored and not self._reveal:
            self._hand_row.set_cards([QLabel("(opponent to act — hand hidden)")])
        else:
            owner = "opp" if mirrored else "self"
            self._hand_row.set_cards(
                [self._mk_card(c, owner, zone="hand") for c in gs["self_hand"]])

    def _mk_card(self, p, controller, zone="battlefield"):
        from gui_game import CardWidget

        perm = p if zone == "battlefield" else None
        w = CardWidget(p["name"], p["card_idx"], controller, zone, perm=perm)
        w.set_card_height(84)
        self._widgets.setdefault(p["name"], []).append(w)
        pm = self._provider.request(p["name"])
        if pm is not None:
            w.set_pixmap(pm)
        return w

    def _refresh_images(self, name):
        for w in self._widgets.get(name, []):
            pm = self._provider.request(name)
            if pm is not None:
                w.set_pixmap(pm)


# ── The window ────────────────────────────────────────────────────────────────

class AnalysisWindow(QMainWindow):
    """Per-action MCTS analysis of the current decision, in its own window."""

    _COLS = ("#", "Action", "Prior", "Visits", "Visit %", "Q (win %)")

    def __init__(self, primary_env, cfg, opp_is_a, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("RoboMage — Analysis")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self._cfg = cfg
        self._opp_is_a = opp_is_a
        self._primary = primary_env  # live env (read-only: action history)
        self._update = None          # latest analyzable StateUpdate (or None)
        self._labels = {}            # action index -> menu label for _update
        self._tag = 0                # run id; stale signals are dropped
        self._running = False
        self._last_stats_time = None
        self._last_sims = 0
        self.smoke_ok = False        # ROBOMAGE_ANALYSIS_SMOKE: first stats seen
        self.smoke_review_ok = False  # …and the opponent-review rewind ran
        self._smoke = bool(os.environ.get("ROBOMAGE_ANALYSIS_SMOKE"))
        self._smoke_reviewed = False
        # Branch lines / PV walk state (phase 3).
        self._branches = []          # [BranchSeries] for the last finished run
        self._walk = None            # (action, world, [WalkNode], labels)
        # Opponent-analysis state (phase 4). _run_is_opp marks the current
        # run/display as an OPPONENT decision (values are theirs; the branch
        # chart flips them to the human's perspective). _last_req_len keeps
        # AUTOMATIC analysis requests monotonic in history length, so the live
        # stream never makes the engine rewind; an explicit review (F6) may.
        self._run_is_opp = False
        self._last_req_len = 0
        # The opponent's most recent analyzable decision, kept so it can be
        # reviewed AFTER they act (F6 rewinds the analysis engine to it), plus
        # the action they played there — marked ▶ in the table.
        self._opp_last = None        # AnalysisRequest or None
        self._chosen_action = None   # played action for the displayed run
        self._run_is_review = False  # the current run is such a review

        self._bridge = AnalysisBridge()
        self._bridge.stats_update.connect(self._on_stats)
        self._bridge.status.connect(self._on_status)
        self._bridge.evaluator_note.connect(self._on_evaluator_note)
        self._bridge.run_done.connect(self._on_run_done)
        self._bridge.branches_ready.connect(self._on_branches)
        self._bridge.walk_ready.connect(self._on_walk)
        self._bridge.opp_result.connect(self._on_opp_result)
        self._worker = AnalysisWorker(primary_env, cfg, self._bridge)
        self._worker.start()

        self._build_ui()
        if self._smoke:
            self._reveal.setChecked(True)   # exercise the opponent path too
        QShortcut(QKeySequence("F5"), self, activated=self.request_analyze)
        QShortcut(QKeySequence("Shift+F5"), self, activated=self.request_stop)
        QShortcut(QKeySequence("F6"), self, activated=self.request_opp_last)

    # ----- layout -----

    def _build_ui(self):
        central = QWidget()
        v = QVBoxLayout(central)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        label = self._cfg.evaluator_spec
        self._header = QLabel(f"evaluator {label}  ·  {self._cfg.worlds} worlds")
        self._header.setObjectName("analysisHeader")
        v.addWidget(self._header)

        self._values = QLabel("net —  ·  search —  ·  0 sims")
        v.addWidget(self._values)

        # Evaluator caveat (e.g. an AZ value column that was never trained for
        # this matchup). Hidden until a run reports one — a silent constant
        # value would otherwise read as a confident 50%.
        self._eval_note = QLabel("")
        self._eval_note.setObjectName("analysisWarning")
        self._eval_note.setWordWrap(True)
        self._eval_note.setStyleSheet("color: #d8a12a;")
        self._eval_note.hide()
        v.addWidget(self._eval_note)

        controls = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze (F5)")
        self._analyze_btn.clicked.connect(self.request_analyze)
        self._stop_btn = QPushButton("Stop (Shift+F5)")
        self._stop_btn.clicked.connect(self.request_stop)
        self._stop_btn.setEnabled(False)
        self._auto = QCheckBox("Auto-analyze each decision")
        self._auto.setChecked(bool(getattr(self._cfg, "auto_analyze", False)))
        self._cap = QSpinBox()
        self._cap.setRange(0, 1_000_000)
        self._cap.setValue(int(self._cfg.max_sims))
        self._cap.setSpecialValueText("∞")
        self._cap.setToolTip("Simulation cap per run (∞ = run until stopped).")
        self._cap.valueChanged.connect(self._on_cap_changed)
        controls.addWidget(self._analyze_btn)
        controls.addWidget(self._stop_btn)
        controls.addWidget(self._auto)
        controls.addStretch(1)
        controls.addWidget(QLabel("sims cap"))
        controls.addWidget(self._cap)
        v.addLayout(controls)

        controls2 = QHBoxLayout()
        self._reveal = QCheckBox(
            "Show opponent analysis (reveals hidden information)")
        self._reveal.setChecked(False)
        self._reveal.setToolTip(
            "Analyze the opponent's decisions too, from THEIR perspective — "
            "including their hidden hand. A search opponent's own search is "
            "shown for free; other opponents are analyzed on the side engine.")
        self._reveal.toggled.connect(self._sync_opp_last_enabled)
        self._opp_last_btn = QPushButton("Opponent's last decision (F6)")
        self._opp_last_btn.setEnabled(False)
        self._opp_last_btn.setToolTip(
            "Rewind the side analysis engine to just BEFORE the opponent's most "
            "recent real decision and search it from their perspective — what "
            "each of their options was worth, with the move they actually "
            "played marked ▶. Needs the reveal toggle (it shows their hand).")
        self._opp_last_btn.clicked.connect(self.request_opp_last)
        controls2.addWidget(self._reveal)
        controls2.addWidget(self._opp_last_btn)
        controls2.addStretch(1)
        v.addLayout(controls2)

        self._status = QLabel("waiting for a decision…")
        self._status.setWordWrap(True)
        v.addWidget(self._status)

        self._table = QTableWidget(0, len(self._COLS))
        self._table.setHorizontalHeaderLabels(self._COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.itemSelectionChanged.connect(self._on_branch_selected)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        for c in (0, 2, 3, 4, 5):
            header.setSectionResizeMode(c, QHeaderView.ResizeToContents)

        # Lines pane: branch value chart + PV scrubber + hypothetical board.
        lines = QWidget()
        lv = QVBoxLayout(lines)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(4)
        self._chart = BranchValueChart()
        self._chart.depth_selected.connect(self._on_chart_depth)
        lv.addWidget(self._chart)
        scrub = QHBoxLayout()
        self._walk_btn = QPushButton("Walk line")
        self._walk_btn.setToolTip(
            "Replay the selected branch's principal variation in the chosen "
            "world and scrub through the future positions.")
        self._walk_btn.clicked.connect(self._on_walk_clicked)
        self._walk_btn.setEnabled(False)
        self._world_spin = QSpinBox()
        self._world_spin.setRange(0, max(0, self._cfg.worlds - 1))
        self._world_spin.setToolTip("Which determinized world's line to walk.")
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_scrub)
        scrub.addWidget(self._walk_btn)
        scrub.addWidget(QLabel("world"))
        scrub.addWidget(self._world_spin)
        scrub.addWidget(self._slider, 1)
        lv.addLayout(scrub)
        self._step_label = QLabel("")
        self._step_label.setWordWrap(True)
        lv.addWidget(self._step_label)
        self._mini = MiniBoard(self._opp_is_a)
        self._mini.clear()
        lv.addWidget(self._mini, 1)
        self._reveal.toggled.connect(self._mini.set_reveal)

        split = QSplitter(Qt.Vertical)
        split.addWidget(self._table)
        split.addWidget(lines)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        v.addWidget(split, 1)

        self.setCentralWidget(central)
        self.resize(760, 860)

    # ----- GameWindow-facing API (UI thread) -----

    @property
    def pending_analyzable(self) -> bool:
        """True while the current decision is one the window can analyze."""
        return self._update is not None

    def on_decision(self, u) -> None:
        """A new decision landed on the board.

        Human decisions cancel any running analysis (the engine should be free
        for the new position) and arm/auto-start a new one when analyzable.
        Opponent decisions are remembered (so the last one can be reviewed
        after the fact via F6) and, with the reveal toggle on, start a
        THEIR-perspective run on the side engine without cancelling anything —
        the analysis engine simply lags the live game and catches up at the
        next sync. Autopass stream updates (human seat, not acting) touch
        nothing."""
        if u.human_turn:
            self._worker.cancel()
            reason = analyze_refusal(bool(u.search_safe), u.num_choices)
            if reason is not None:
                self._update = None
                self._labels = {}
                self._analyze_btn.setEnabled(False)
                self._set_status("idle", reason)
                return
            self._update = u
            self._labels = {a["index"]: menu_label(a, self._opp_is_a)
                            for a in u.actions}
            self._analyze_btn.setEnabled(True)
            self._set_status("idle",
                             "ready — Analyze (F5) to search this decision")
            if self._auto.isChecked() and (self.isVisible() or self._smoke):
                self.request_analyze()
            return
        if not u.opp_perspective or analyze_refusal(bool(u.search_safe),
                                                    u.num_choices) is not None:
            return
        # A real opponent decision: keep it as the F6 review root (its obs is
        # already a private copy) before they act on it.
        self._opp_last = AnalysisRequest.from_update(u)
        self._sync_opp_last_enabled()
        if (self._reveal.isChecked() and self._auto.isChecked()
                and not self._running
                and u.history_len >= self._last_req_len
                and (self.isVisible() or self._smoke)):
            self._start_opp_analysis(u)

    def on_human_acted(self) -> None:
        """The human committed an action: the analyzed position is now stale."""
        self._worker.cancel()
        self._analyze_btn.setEnabled(False)

    def opp_search_sink(self):
        """A callable for SearchController.on_result (fires on the driver
        thread) — marshals the opponent's own SearchResult to the UI thread."""
        bridge = self._bridge
        return lambda obs, num, result, chosen: bridge.opp_result.emit(
            (obs, num, result, chosen))

    def _on_opp_result(self, payload) -> None:
        """Display a search opponent's OWN search for the move it just played
        (free — no extra compute). Gated by the reveal toggle; skipped while a
        human-decision run is in flight (that display wins)."""
        if not self._reveal.isChecked() or self._running:
            return
        obs, num, result, chosen = payload
        self._tag += 1              # invalidate stale branch/walk signals
        self._run_is_opp = True
        self._run_is_review = False
        self._chosen_action = chosen
        self._labels = opp_menu_labels(obs, num)
        self._clear_lines()
        q = result.q if result.q is not None else np.zeros(num)
        self._values.setText(
            f"opponent's search — their win% {_win_pct(result.root_value)}  ·  "
            f"{result.sims_run} sims")
        self._fill_table(num, result.visits, result.priors, q)
        self._set_status("idle",
                         "opponent's own search for the move they played "
                         "(▶ = chosen)")

    def on_game_over(self) -> None:
        self._worker.cancel()
        self._update = None
        self._analyze_btn.setEnabled(False)
        self._set_status("idle", "game over")

    def shutdown(self) -> None:
        self._worker.shutdown()
        self._worker.join(timeout=2.0)

    # ----- controls -----

    def request_analyze(self) -> None:
        if self._update is None:
            return
        # Rebuild the row labels from the decision we are about to analyze:
        # an opponent search result (_on_opp_result) may have overwritten
        # self._labels with the opponent's menu since this decision was armed,
        # and this method is reachable via the F5 shortcut even after the
        # button was disabled. Without this the table would label the human
        # decision's rows with the opponent's action text.
        self._labels = {a["index"]: menu_label(a, self._opp_is_a)
                        for a in self._update.actions}
        self._tag += 1
        self._running = True
        self._run_is_opp = False
        self._run_is_review = False
        self._chosen_action = None
        self._last_stats_time = None
        self._last_sims = 0
        self._last_req_len = self._update.history_len
        self._stop_btn.setEnabled(True)
        self._clear_lines()
        self._set_status("busy", "analyzing…")
        self._worker.submit_analyze(self._tag,
                                    AnalysisRequest.from_update(self._update))

    def _start_opp_analysis(self, u) -> None:
        """Analyze an OPPONENT decision from their perspective (reveal mode).
        The opponent will usually have acted long before this finishes — the
        result is a retrospective 'what their options were worth'."""
        self._tag += 1
        self._running = True
        self._run_is_opp = True
        self._run_is_review = False
        self._chosen_action = None      # they haven't answered it yet
        self._last_stats_time = None
        self._last_sims = 0
        self._last_req_len = u.history_len
        self._labels = opp_menu_labels(u.obs, u.num_choices)
        self._stop_btn.setEnabled(True)
        self._clear_lines()
        self._set_status("busy",
                         "analyzing OPPONENT decision (values are theirs)…")
        self._worker.submit_analyze(self._tag, AnalysisRequest.from_update(u))

    def request_opp_last(self) -> None:
        """Review the opponent's most recent real decision (F6).

        Rewinds the side analysis engine to the position just BEFORE that
        decision — a respawn at its history prefix, see AnalysisSession.sync —
        and searches it from THEIR perspective, so the table reads as "what
        their options were worth", with the move they went on to play marked ▶.
        The live game is untouched and keeps running; the engine catches back up
        by forward replay at the next analysis."""
        req = self._opp_last
        if req is None:
            return
        if not self._reveal.isChecked():
            self._set_status(
                "idle", "tick 'Show opponent analysis' first — reviewing their "
                        "decision reveals their hidden information")
            return
        self._tag += 1
        self._running = True
        self._run_is_opp = True
        self._run_is_review = True
        self._chosen_action = self._played_action(req)
        self._last_stats_time = None
        self._last_sims = 0
        self._labels = opp_menu_labels(req.obs, req.num_choices)
        self._stop_btn.setEnabled(True)
        self._clear_lines()
        played = ("" if self._chosen_action is None
                  else f" — they played: "
                       f"{self._labels.get(self._chosen_action, '?')}")
        self._set_status(
            "busy",
            f"rewinding to the opponent's last decision ({_req_step(req)})"
            f"{played}")
        self._worker.submit_analyze(self._tag, req)

    def _played_action(self, req: AnalysisRequest):
        """The action the opponent actually took at `req`'s decision: the
        history entry recorded when they answered it, or None if they haven't
        yet. A bounded read of the primary env's append-only list (GIL-safe,
        same discipline as AnalysisSession's delta replay)."""
        hist = getattr(self._primary, "_action_history", None)
        if not hist or req.history_len >= len(hist):
            return None
        return int(hist[req.history_len])

    def _smoke_review(self) -> None:
        """Headless smoke: drive the F6 opponent review once, if a decision of
        theirs has been seen (an all-autopass smoke may never see one)."""
        if self._smoke_reviewed or self._opp_last is None:
            return
        self._smoke_reviewed = True
        self.request_opp_last()

    def _sync_opp_last_enabled(self) -> None:
        self._opp_last_btn.setEnabled(self._opp_last is not None
                                      and self._reveal.isChecked())

    def request_stop(self) -> None:
        self._worker.cancel()

    def _on_cap_changed(self, value: int) -> None:
        # Read by the worker's chunk loop between chunks (single int write —
        # cross-thread safe); takes effect on the running search too.
        self._cfg.max_sims = int(value)

    # ----- worker signals (UI thread) -----

    def _on_stats(self, payload) -> None:
        tag, stats = payload
        if tag != self._tag:
            return
        now = time.monotonic()
        rate = ""
        if self._last_stats_time is not None and now > self._last_stats_time:
            sps = (stats.sims_run - self._last_sims) / (now - self._last_stats_time)
            rate = f"  ·  {sps:.0f} sims/s"
        self._last_stats_time = now
        self._last_sims = stats.sims_run
        whose = "their " if self._run_is_opp else ""
        self._values.setText(
            f"net {whose}{_win_pct(stats.net_value)}  ·  "
            f"search {whose}{_win_pct(stats.root_value)}  ·  "
            f"{stats.sims_run} sims{rate}")
        self._fill_table(stats.num_choices, stats.visits, stats.priors, stats.q)
        if stats.sims_run > 0:
            self.smoke_ok = True
            if self._smoke and self._run_is_review and not self.smoke_review_ok:
                self.smoke_review_ok = True
                print("ANALYSIS SMOKE: reviewed the opponent's last decision "
                      f"({stats.sims_run} sims)", flush=True)

    def _fill_table(self, num_choices, visits, priors, q) -> None:
        """Render root-action stats, sorted by visits. When the run's root is a
        decision that was already answered (a reviewed opponent decision, or an
        opponent hook result), the action actually played is marked ▶."""
        chosen = self._chosen_action
        order = np.argsort(-np.asarray(visits), kind="stable")
        total = float(np.sum(visits)) or 1.0
        self._table.setRowCount(num_choices)
        for row, idx in enumerate(int(i) for i in order):
            visited = visits[idx] > 0
            label = self._labels.get(idx, f"action {idx}")
            if idx == chosen:
                label = "▶ " + label
            cells = (
                str(idx),
                label,
                f"{priors[idx] * 100.0:.1f}%",
                str(int(visits[idx])),
                f"{visits[idx] / total * 100.0:.1f}%",
                _win_pct(float(q[idx])) if visited else "—",
            )
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col != 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 0:
                    item.setData(Qt.UserRole, idx)  # row -> action mapping
                self._table.setItem(row, col, item)

    def _on_status(self, payload) -> None:
        state, text = payload
        self._set_status(state, text)

    def _on_evaluator_note(self, note: str) -> None:
        self._eval_note.setText(f"⚠ {note}" if note else "")
        self._eval_note.setVisible(bool(note))

    def _on_run_done(self, payload) -> None:
        tag, stats = payload
        if tag != self._tag:
            return
        self._running = False
        self._stop_btn.setEnabled(False)
        if self._run_is_review and stats is not None:
            # Restore the review's framing over the worker's generic "done"
            # (queued ahead of this signal) — values are the OPPONENT's.
            self._set_status(
                "done",
                f"reviewed the opponent's last decision — {stats.sims_run} sims, "
                "values in THEIR perspective; ▶ = the move they actually played "
                "(their own agent's pick, not this search's)")
        # Read the top branches' PV series off the parked search for the
        # chart/scrubber (tree-only; cheap).
        if stats is not None and stats.sims_run > 0:
            order = np.argsort(-stats.visits, kind="stable")
            top = [int(a) for a in order[:MAX_BRANCHES] if stats.visits[a] > 0]
            if top:
                self._worker.submit_branches(tag, top, self._cfg.worlds)

    # ----- branch lines / PV walk (UI thread) -----

    def _clear_lines(self) -> None:
        self._branches = []
        self._walk = None
        self._chart.set_series([])
        self._walk_btn.setEnabled(False)
        self._slider.setEnabled(False)
        self._slider.setRange(0, 0)
        self._step_label.setText("")
        self._mini.clear()

    def _selected_branch(self):
        """The BranchSeries for the table's selected row (or the top branch)."""
        row = self._table.currentRow()
        if row >= 0:
            item = self._table.item(row, 0)
            if item is not None:
                action = item.data(Qt.UserRole)
                for s in self._branches:
                    if s.action == action:
                        return s
        return self._branches[0] if self._branches else None

    def _on_branches(self, payload) -> None:
        tag, series = payload
        if tag != self._tag:
            return
        if self._run_is_opp:
            # Chart stays in the HUMAN's perspective: an opponent root's
            # values are theirs, so flip (zero-sum, draws ~0).
            for s in series:
                s.per_world = [[1.0 - v for v in w] for w in s.per_world]
                s.mean = [1.0 - v for v in s.mean]
        self._branches = series
        sel = self._selected_branch()
        self._chart.set_series(series, sel.action if sel else None)
        self._walk_btn.setEnabled(sel is not None)
        if self._smoke and sel is not None:
            self._on_walk_clicked()     # exercise the PV walk headlessly too

    def _on_branch_selected(self) -> None:
        sel = self._selected_branch()
        if sel is not None:
            self._chart.set_selected(sel.action)
            self._walk_btn.setEnabled(True)

    def _on_walk_clicked(self) -> None:
        sel = self._selected_branch()
        if sel is None:
            return
        world = self._world_spin.value()
        pv = sel.pv_actions.get(world, [])
        if not pv:
            self._set_status("idle",
                             f"world {world} never explored that action — "
                             "pick another world")
            return
        self._set_status("busy", "walking the line…")
        self._worker.submit_walk(self._tag, sel.action, world, pv,
                                 dict(self._labels))

    def _on_walk(self, payload) -> None:
        tag, action, world, nodes, labels = payload
        if tag != self._tag or not nodes:
            return
        if self._smoke:
            print(f"ANALYSIS SMOKE: walked {len(nodes)} steps "
                  f"(action {action}, world {world})", flush=True)
            # Exercise the opponent-review rewind (F6) too, once, after the
            # walk — best effort: the smoke may auto-play on before it lands.
            QTimer.singleShot(0, self._smoke_review)
        self._walk = (action, world, nodes, labels)
        self._slider.setEnabled(True)
        self._slider.setRange(0, len(nodes) - 1)
        self._set_status("idle",
                         f"line walked ({len(nodes)} steps) — scrub below")
        if self._slider.value() == 0:
            self._show_walk_depth(0)
        else:
            self._slider.setValue(0)

    def _on_scrub(self, depth: int) -> None:
        self._show_walk_depth(depth)

    def _on_chart_depth(self, depth: int) -> None:
        if self._walk is not None:
            self._slider.setValue(min(depth, self._slider.maximum()))
        else:
            self._chart.set_cursor(depth)

    def _show_walk_depth(self, depth: int) -> None:
        if self._walk is None:
            return
        action, world, nodes, labels = self._walk
        depth = max(0, min(depth, len(nodes) - 1))
        self._chart.set_cursor(depth)
        node = nodes[depth]
        text = f"step {depth + 1}/{len(nodes)}: {labels[depth]}"
        for s in self._branches:
            if s.action == action and len(s.per_world[world]) > depth:
                text += (f"   ·   value {s.per_world[world][depth] * 100:.1f}%"
                         f" (world {world})")
                break
        self._step_label.setText(text)
        if node.terminal is not None:
            self._mini.show_terminal(node.terminal)
        else:
            self._mini.show_node(node.obs)

    def _set_status(self, state: str, text: str) -> None:
        self._status.setText(text)
        self._status.setProperty("state", state)
