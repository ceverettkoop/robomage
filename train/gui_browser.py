"""Qt analysis-browser pane — the PySide6 port of tui_analysis's browser.

`BrowserPane` is a per-session QWidget the shell (gui_main.MainWindow /
SessionManager) embeds behind a small contract: construct with the analysis
dialog's opts dict (cheap — nothing loads), `start()` to submit the initial
load+collect job, `shutdown()` to stop it, `traces()`/`provenance()` to save,
`load_traces()` to open a saved .rmtrace (browse-only when no live env).

Everything front-end-independent lives in browse_session.py (shared with the
Textual TUI): the games store + cursor (`BrowseStore`), the analyses registry,
the presentation-data helpers (game labels, decision rows, phase strip, clock
line), the V(s) histogram geometry (`HistogramModel`), and the engine job
bodies (`EngineCore`). This module supplies only the Qt half:

  * `BrowserBridge` — queued-signal marshalling from the worker threads to the
    UI thread (the Qt analog of Textual's post_message).
  * `EngineWorker` — the ONE thread that touches model/env/opp_model, running
    one queued job at a time (load / collect / whatif / shutdown), reproducing
    Textual's `group="engine"` serialization.
  * The widgets: `TraceBoard` (full-parity board render of a recorded obs,
    composing gui_game's CardWidget/CardRow/PhaseStrip/StackItemWidget),
    `DecisionPanel` (policy table + opponent-actions + clock/shard footers),
    and `ValueHistogramWidget` (clickable per-decision V(s) bars).

Live streaming: EngineCore streams GameStarted/StepAppended/... events while a
game is still being played; the pane shows it as a LIVE games-list row that
grows per decision, with an optional follow mode that keeps the board on the
newest decision (stepping back disengages; End re-engages).

Threading: all widget access happens on the UI thread (bridge signals are
queued). Analyses run on one-shot threads over an immutable snapshot of the
finished games (`browse_session.analysis_pool`), never touching the env.
"""

import argparse
import os
import queue
import threading
import traceback

import numpy as np

from PySide6.QtCore import Qt, QObject, QSize, QTimer, Signal
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QKeySequence,
                           QPainter, QPen, QShortcut)
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QPlainTextEdit, QPushButton,
                               QSplitter, QTabWidget, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

import browse_session as bs
import decode
import shard_probes
from cli_spec import BINARY
from env import STATE_SIZE
from game_driver import stack_target_refs
from gui_game import (CardRow, CardWidget, HAND_CARD_H, HAND_CARD_W,
                      HAND_ROW_H, ImageProvider, OraclePopup, PhaseStrip,
                      STACK_EMPTY_H, STACK_ROW_H, StackItemWidget)
from tui_game import hand_type_icon

# ── Palette (mirrors gui_game._QSS / the TUI histogram colors) ────────────────
_BG = QColor("#0c0c10")
_POS = QColor("#33d17a")          # V > 0 — model favored (green)
_NEG = QColor("#e05555")          # V < 0 — opponent favored (red)
_CURSOR = QColor("#d8a12a")       # bar(s) at the selected decision step (amber)
_CURSOR_BG = QColor("#2a2a33")    # full-height column marker under the cursor
_AXIS = QColor("#6a6a72")
_LIVE_FG = QColor("#d8a12a")      # LIVE games-list rows
_CHOSEN_FG = QColor("#d8a12a")    # chosen decision-table row

_SHARD_CAVEAT = ("⛁ shard replay: chosen = argmax(visits) — sampled, "
                 "not exact, inside each game's temperature window; "
                 "unsearched (fallback) decisions are not recorded")

# Gate messages (verbatim parity with tui_analysis._run_menu_entry).
_MSG_SHARD = ("Not available in shard replay — branching/simulating needs a "
              "live env.")
_MSG_BROWSE_ONLY = ("Not available — browse-only session (no live env to "
                    "branch/simulate on).")
_MSG_NO_ENV = "Live env not ready."
_MSG_BUSY = ("Engine is busy (simulating or branching) — try again when it "
             "finishes.")
_MSG_NO_SEL = "Select a game and step first."

# Namespace dests per cli_spec.ANALYSIS_TUI_TOOL (the schema _load_model_and_env
# consumes): opts-dict key -> default.
_ARG_DEFAULTS = {
    "model": "gen", "opponent": "scripted", "deck_a": None, "deck_b": None,
    "binary": BINARY, "bo3": False, "think_time": None, "match_clock": None,
    "n_games": 20, "shards": None, "seat": "A", "no_net": False,
}


def _make_args(opts):
    """An argparse-style namespace for analysis._load_model_and_env, built from
    the analysis-session dialog's opts dict. The pane keeps this namespace as
    EngineCore's OWN copy (the loader mutates it: _apply_search_budget_flags
    self-clears, deck_a/deck_b are written back)."""
    return argparse.Namespace(
        **{k: opts.get(k, d) for k, d in _ARG_DEFAULTS.items()})


def _mono_font():
    return QFontDatabase.systemFont(QFontDatabase.FixedFont)


def _token_pt(p):
    """Scryfall token lookup key for a permanent (same rule as the play board):
    (p, t) for a P/T token, (None, None) for a non-creature token, else None."""
    if p.get("card_idx") != decode._TOKEN_IDX:
        return None
    if "power" in p:
        return (p["power"], p["toughness"])
    return (None, None)


# ── Worker → UI bridge ────────────────────────────────────────────────────────

class BrowserBridge(QObject):
    """Queued-signal bridge from the worker threads to the pane. Signals are
    emitted on worker threads; Qt queues delivery onto the UI thread."""

    engine_event = Signal(object)     # a browse_session event dataclass
    analysis_done = Signal(object)    # (title, text) from an analysis runner


class EngineWorker(threading.Thread):
    """The one thread that touches EngineCore (model/env/opp_model).

    Commands (via the queue): ("load", n, stop) / ("collect", n, stop) /
    ("whatif", gn, step, k, game) / ("shutdown",). Stop events are created at
    submit time, one per collect run (the AnalysisWorker discipline), so a
    stop that lands before the worker even dequeues the run still sticks. The
    UI rejects submissions while the store is engine_busy, keeping the queue
    depth ≤ 1 job (+ shutdown)."""

    def __init__(self, core):
        super().__init__(name="gui-browser-engine", daemon=True)
        self._core = core
        self._q = queue.Queue()

    def submit(self, cmd):
        self._q.put(cmd)

    def run(self):
        while True:
            cmd = self._q.get()
            kind = cmd[0]
            if kind == "shutdown":
                break
            try:
                if kind == "load":
                    self._core.load_and_collect(cmd[1], cmd[2])
                elif kind == "collect":
                    self._core.collect(cmd[1], cmd[2])
                elif kind == "whatif":
                    self._core.whatif(cmd[1], cmd[2], cmd[3], cmd[4])
            except Exception:   # noqa: BLE001 — the worker loop must survive
                # EngineCore jobs guard themselves; this is the last-resort
                # net so a bug can't kill the queue loop silently.
                self._core.emit(bs.AnalysisDone("engine worker error",
                                                traceback.format_exc()))
                self._core.emit(bs.EngineIdle())
        # Release the env on THIS thread (its owner), as the last job.
        self._core.close()


# ── V(s) histogram ────────────────────────────────────────────────────────────

class ValueHistogramWidget(QWidget):
    """Bar-per-decision V(s) chart for one game; click a column to seek.

    Wraps browse_session.HistogramModel (the same geometry the TUI renders
    with eighth-blocks): symmetric ±vmax scale, whole-step columns widened
    when the game fits, max-|V| bucketing when it doesn't. Painted at a fixed
    COL_PX column pitch; the model is relaid out to however many columns fit
    the current width."""

    COL_PX = 6        # column pitch: 4 px bar + 2 px gap
    BAR_PX = 4
    GUTTER = 34       # left scale labels: +vmax / 0 / -vmax
    AXIS_H = 14       # bottom step-tick strip

    step_picked = Signal(int)
    step_hovered = Signal(object, object)   # (step | None, value | None)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model = bs.HistogramModel()
        self.setMouseTracking(True)
        self.setMinimumHeight(100)

    def sizeHint(self):
        return QSize(400, 140)

    # ----- data API -----

    def set_data(self, values, cursor=0):
        self._model.set_data(values, cursor=cursor)
        self._model.layout(self._n_cols())
        self.update()

    def append(self, v):
        self._model.append(v)
        self.update()

    def set_cursor(self, step):
        self._model.set_cursor(step)
        self.update()

    def value_at(self, step):
        vals = self._model.values
        return vals[step] if step is not None and 0 <= step < len(vals) else None

    # ----- geometry -----

    def _n_cols(self):
        return max(0, (self.width() - self.GUTTER) // self.COL_PX)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._model.layout(self._n_cols())

    def _col_at(self, x):
        return int((x - self.GUTTER) // self.COL_PX)

    # ----- painting -----

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), _BG)
        m = self._model
        if not m.values or not m.cols:
            p.setPen(_AXIS)
            p.drawText(self.rect(), Qt.AlignCenter, "(no game selected)")
            p.end()
            return
        plot_h = self.height() - self.AXIS_H
        mid = plot_h / 2.0
        half = max(1.0, mid - 1.0)
        vmax = m.vmax

        # Gutter scale labels.
        p.setPen(_AXIS)
        top_lab = "+1" if vmax == 1.0 else f"+{vmax:.1f}"
        bot_lab = "-1" if vmax == 1.0 else f"-{vmax:.1f}"
        fm = p.fontMetrics()
        gw = self.GUTTER - 4
        p.drawText(0, 2, gw, fm.height(), Qt.AlignRight, top_lab)
        p.drawText(0, int(mid - fm.height() / 2), gw, fm.height(),
                   Qt.AlignRight, "0")
        p.drawText(0, plot_h - fm.height() - 2, gw, fm.height(),
                   Qt.AlignRight, bot_lab)

        # Cursor column background (full plot height), under the bars.
        ccols = m.cursor_cols()
        for c in ccols:
            x = self.GUTTER + c * self.COL_PX
            p.fillRect(x, 0, self.COL_PX, plot_h, _CURSOR_BG)

        # Dashed zero line across the plot.
        p.setPen(QPen(_AXIS, 1, Qt.DashLine))
        p.drawLine(self.GUTTER, int(mid), self.width() - 2, int(mid))

        # Bars: green up / red down, amber-tinted in the cursor column(s).
        for c, (v, _step) in enumerate(m.cols):
            if v is None or v == 0.0:
                continue
            x = self.GUTTER + c * self.COL_PX
            h = abs(v) / vmax * half
            color = _CURSOR if c in ccols else (_POS if v > 0 else _NEG)
            if v > 0:
                p.fillRect(x, int(mid - h), self.BAR_PX, max(1, int(h)), color)
            else:
                p.fillRect(x, int(mid) + 1, self.BAR_PX, max(1, int(h)), color)

        # Bottom axis: first step, bold ^cursor, last step.
        n = len(m.values)
        used = len(m.cols) * self.COL_PX
        ay = plot_h + self.AXIS_H - 3
        p.setPen(_AXIS)
        p.drawText(self.GUTTER, ay, "0")
        last = str(n - 1)
        p.drawText(self.GUTTER + used - fm.horizontalAdvance(last), ay, last)
        cur_col = min(ccols or {0})
        f = QFont(p.font())
        f.setBold(True)
        p.setFont(f)
        p.setPen(_CURSOR)
        cx = self.GUTTER + cur_col * self.COL_PX
        cx = max(self.GUTTER, min(cx, self.GUTTER + used - 8))
        p.drawText(cx, ay, f"^{m.cursor}")
        p.end()

    # ----- mouse -----

    def mousePressEvent(self, event):
        step = self._model.step_at(self._col_at(event.position().x()))
        if step is not None:
            self.step_picked.emit(step)

    def mouseMoveEvent(self, event):
        step = self._model.step_at(self._col_at(event.position().x()))
        self.step_hovered.emit(step, self.value_at(step))

    def leaveEvent(self, event):
        self.step_hovered.emit(None, None)


# ── Trace board ───────────────────────────────────────────────────────────────

class TraceBoard(QWidget):
    """Full-parity board render of one recorded decision's obs.

    A NEW widget (not MiniBoard, which lacks the stack row, land split, GY and
    info lines and carries live-game mirroring traces never need): traces are
    always model-perspective, so the obs decodes with SELF_OPP_LABELS, no
    mirroring, hand visible. Composes the play board's widgets — PhaseStrip,
    CardRow (land rows tinted), CardWidget with perm= (tapped rotation and
    counter chips for free), StackItemWidget — over decode.decode_game_state.
    No perm side-channels exist in traces (perm_counters=None): same fidelity
    as the TUI browser."""

    def __init__(self, provider, popup, parent=None):
        super().__init__(parent)
        self._provider = provider
        self._popup = popup
        self._provider.image_ready.connect(self._on_image_ready)
        self._registry = {}          # name -> [(widget, token_pt)]
        self._hovered = None
        # Click-parity with the TUI's notify(): the popup auto-hides after a
        # few seconds (a right-button hold still hides on release).
        self._popup_timer = QTimer(self)
        self._popup_timer.setSingleShot(True)
        self._popup_timer.setInterval(8000)
        self._popup_timer.timeout.connect(self._popup.hide)

        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        self._phase = PhaseStrip()
        v.addWidget(self._phase)
        self._opp_info = QLabel("")
        self._opp_info.setObjectName("oppInfo")
        v.addWidget(self._opp_info)
        self._split = QSplitter(Qt.Vertical)
        self._opp_lands = CardRow(land=True)
        self._opp_perms = CardRow()
        self._stack_row = CardRow(height=STACK_ROW_H, min_height=STACK_EMPTY_H)
        self._self_perms = CardRow()
        self._self_lands = CardRow(land=True)
        self._hand_row = CardRow(height=HAND_ROW_H)
        for row in (self._opp_lands, self._opp_perms, self._stack_row,
                    self._self_perms, self._self_lands, self._hand_row):
            self._split.addWidget(row)
        self._split.setSizes([r.preferred for r in
                              (self._opp_lands, self._opp_perms,
                               self._stack_row, self._self_perms,
                               self._self_lands, self._hand_row)])
        v.addWidget(self._split, 1)
        self._self_info = QLabel("")
        self._self_info.setObjectName("selfInfo")
        v.addWidget(self._self_info)
        self._zones = QLabel("")
        self._zones.setObjectName("graveyards")
        self._zones.setWordWrap(True)
        v.addWidget(self._zones)

    # ----- rendering -----

    def clear(self, text=""):
        self._registry = {}
        self._phase.update_strip(-1, text)
        self._opp_info.setText("")
        self._self_info.setText("")
        self._zones.setText("")
        for row in (self._opp_lands, self._opp_perms, self._stack_row,
                    self._self_perms, self._self_lands, self._hand_row):
            row.set_cards([])

    def show_step(self, game, gn, step):
        """Render decision `step` of `game` (traces are model-perspective —
        'self' is always the model; the model's hand is its own, visible)."""
        if not game["observations"]:
            self.clear("(no decisions recorded yet)")
            return
        self._registry = {}
        obs = np.asarray(game["observations"][step], dtype=np.float32)
        gs = decode.decode_game_state(obs[:STATE_SIZE],
                                      labels=decode.SELF_OPP_LABELS)
        pd = bs.phase_data(game, gn, step, obs, gs)
        self._phase.update_strip(pd.cur_step_idx,
                                 pd.header + pd.match + "   " + pd.context)
        self._opp_info.setText(
            bs.info_line("OPPONENT", gs["opponent"], gs["opp_library"]))
        self._self_info.setText(
            bs.info_line("MODEL", gs["self"], gs["self_library"]))
        self._zones.setText(self._zones_text(gs))
        self._rebuild_stack(gs["stack"])
        self._fill_bf(self._opp_perms, self._opp_lands,
                      gs["opp_battlefield"], "opp")
        self._fill_bf(self._self_perms, self._self_lands,
                      gs["self_battlefield"], "self")
        self._hand_row.set_cards([self._mk_hand(c) for c in gs["self_hand"]])

    @staticmethod
    def _zones_text(gs):
        lines = [f"Model GY: {', '.join(gs['self_graveyard']) or '—'}",
                 f"Opp GY:   {', '.join(gs['opp_graveyard']) or '—'}"]
        # Exile shown only when non-empty (the TUI's compact-text convention).
        if gs.get("self_exile") or gs.get("opp_exile"):
            lines.append(f"Model exile: {', '.join(gs['self_exile']) or '—'}"
                         f"   Opp exile: {', '.join(gs['opp_exile']) or '—'}")
        return "\n".join(lines)

    def _fill_bf(self, perms_row, lands_row, perms, controller):
        perms_row.set_cards([self._mk_perm(p, controller)
                             for p in perms if not p.get("is_land")])
        lands_row.set_cards([self._mk_perm(p, controller)
                             for p in perms if p.get("is_land")])

    def _rebuild_stack(self, stack):
        if not stack:
            label = QLabel("Stack: (empty)")
            label.setStyleSheet("color: #6a6a72;")
            self._stack_row.set_cards([label])
            return
        widgets = []
        for e in stack:
            w = StackItemWidget(e, stack_target_refs(e, mirrored=False))
            self._register_image(w._thumb, e["name"], None)
            widgets.append(w)
        self._stack_row.set_cards(widgets)

    # ----- card builders -----

    def _mk_perm(self, p, controller):
        token_pt = _token_pt(p)
        w = CardWidget(p["name"], p["card_idx"], controller, "battlefield",
                       perm=p, token_pt=token_pt)
        self._wire_card(w)
        self._register_image(w, p["name"], token_pt)
        return w

    def _mk_hand(self, c):
        w = CardWidget(c["name"], c["card_idx"], "self", "hand",
                       icon=hand_type_icon(c["card_idx"]),
                       card_size=(HAND_CARD_W, HAND_CARD_H))
        self._wire_card(w)
        self._register_image(w, c["name"], None)
        return w

    def _wire_card(self, w):
        w.clicked.connect(self._on_card_clicked)
        w.hovered.connect(self._on_card_hover)
        w.oracle_show.connect(self._show_oracle)
        w.oracle_hide.connect(self._hide_oracle)

    def _register_image(self, widget, name, token_pt):
        self._registry.setdefault(name, []).append((widget, token_pt))
        pm = self._provider.request(name, token_pt)
        if pm is not None:
            widget.set_pixmap(pm)

    def _on_image_ready(self, name):
        for widget, token_pt in self._registry.get(name, []):
            pm = self._provider.request(name, token_pt)
            if pm is not None:
                widget.set_pixmap(pm)

    # ----- oracle popup -----

    def _on_card_hover(self, widget):
        self._hovered = widget

    def _on_card_clicked(self, widget):
        """Click = inspect (parity with the TUI's click-notify): show the
        oracle popup for a few seconds."""
        self._show_oracle(widget)
        self._popup_timer.start()

    def _show_oracle(self, widget):
        if widget is None:
            return
        self._popup_timer.stop()
        self._popup.show_card(widget._card_idx, widget._name,
                              widget._token_pt)

    def _hide_oracle(self):
        self._popup_timer.stop()
        self._popup.hide()


# ── Decision panel ────────────────────────────────────────────────────────────

class DecisionPanel(QWidget):
    """The model's decision at the current step: every legal action with its
    recorded policy probability (sorted most-probable first, chosen row marked),
    the opponent's actions since the previous decision, the match-clock line,
    and the shard-replay caveat."""

    _COLS = ("#", "P(a)", "", "Action")

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 2, 4, 2)
        v.setSpacing(2)
        self._head = QLabel("Decision — legal actions with policy P(a)")
        self._head.setStyleSheet("color: #b8b8c0; font-weight: bold;")
        v.addWidget(self._head)
        self._clock = QLabel("")
        self._clock.setStyleSheet("color: #9a9aa2;")
        self._clock.hide()
        v.addWidget(self._clock)
        self._table = QTableWidget(0, len(self._COLS))
        self._table.setHorizontalHeaderLabels(self._COLS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QTableWidget.SingleSelection)
        self._table.setShowGrid(False)
        self._table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self._table, 1)
        self._opp = QLabel("")
        self._opp.setStyleSheet("color: #9a9aa2; font-style: italic;")
        self._opp.setWordWrap(True)
        self._opp.hide()
        v.addWidget(self._opp)
        self._caveat = QLabel(_SHARD_CAVEAT)
        self._caveat.setStyleSheet("color: #6a6a72; font-style: italic;")
        self._caveat.setWordWrap(True)
        self._caveat.hide()
        v.addWidget(self._caveat)

    def clear(self):
        self._head.setText("Decision — legal actions with policy P(a)")
        self._clock.hide()
        self._table.setRowCount(0)
        self._opp.hide()
        self._caveat.hide()

    def show_decision(self, game, step):
        dd = bs.decision_data(game, step)
        self._head.setText(f"Decision — {dd.num_choices} legal")
        self._clock.setText(dd.clock)
        self._clock.setVisible(bool(dd.clock))

        has_probs = any(r.prob is not None for r in dd.rows)
        self._table.setColumnHidden(1, not has_probs)
        self._table.setColumnHidden(2, not has_probs)
        self._table.setRowCount(len(dd.rows))
        bold = QFont(self._table.font())
        bold.setBold(True)
        for row, r in enumerate(dd.rows):
            if r.prob is not None:
                bar = "▮" * max(1 if r.prob > 0.005 else 0, round(r.prob * 10))
                cells = (str(r.k), f"{r.prob * 100:5.1f}%", bar, r.desc)
            else:
                cells = (str(r.k), "", "", r.desc)
            if r.is_chosen:
                cells = cells[:3] + (cells[3] + "  ◀ chosen",)
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col in (0, 1):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if col == 0:
                    item.setData(Qt.UserRole, r.k)     # row -> action index
                if r.is_chosen:
                    item.setFont(bold)
                    item.setForeground(_CHOSEN_FG)
                elif col == 2:
                    item.setForeground(_POS)
                self._table.setItem(row, col, item)
        self._table.resizeColumnToContents(0)
        if has_probs:
            self._table.resizeColumnToContents(1)
            self._table.resizeColumnToContents(2)
        self._table.scrollToTop()

        if dd.opp_lines:
            self._opp.setText(f"Opponent since decision {step - 1}:\n"
                              + "\n".join(f"  opp → {ln}" for ln in dd.opp_lines))
            self._opp.show()
        else:
            self._opp.hide()
        self._caveat.setVisible(dd.shard_caveat)


# ── The pane ──────────────────────────────────────────────────────────────────

class BrowserPane(QWidget):
    """The analysis-browser mode of the main window (the SessionManager
    contract): games sidebar + board/decision pager + V(s) histogram +
    analyses menu, over a live simulating env (or shard records, or opened
    traces with no env at all)."""

    dirty_changed = Signal(bool)      # unsaved finished traces exist
    status = Signal(str)              # status-line text for the host
    smoke_done = Signal(int)          # ROBOMAGE_BROWSER_SMOKE completion

    def __init__(self, opts, parent=None):
        super().__init__(parent)
        self._opts = dict(opts or {})
        # Traces-only mode: gui_main constructs with no engine keys and calls
        # load_traces (an opened .rmtrace) — never submit engine jobs then.
        self._has_engine = any(k in self._opts
                               for k in ("model", "opponent", "shards"))
        self._args = _make_args(self._opts)
        self._shards = bool(getattr(self._args, "shards", None))
        self._store = bs.BrowseStore()
        if not self._has_engine:
            self._store.engine_busy = False

        self._bridge = BrowserBridge()
        self._bridge.engine_event.connect(self._on_engine_event)
        self._bridge.analysis_done.connect(self._on_analysis_done)
        self._worker = None
        if self._has_engine:
            core = bs.EngineCore(self._args, self._bridge.engine_event.emit)
            self._worker = EngineWorker(core)
        self._collect_stop = None      # stop event of the running collect job
        self._busy_kind = None         # None | "sim" | "whatif"
        self._analysis_thread = None
        self._probe_net = None         # lazy az_inspect probe net (+ label)
        self._probe_net_label = ""
        self._started = False
        self._closed = False
        self._active = True
        self._load_failed = False
        self._subtitle = None          # from EnvReady
        self._provenance_loaded = None
        self._loading_traces = False
        self._dirty = False
        self._syncing_selection = False

        # Coalesced refresh state (flushed by a single-shot ~50 ms timer).
        self._dirty_rows = set()
        self._dirty_rebuild = False
        self._dirty_summary = False
        self._dirty_selected = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(50)
        self._refresh_timer.timeout.connect(self._flush_refresh)

        self._smoke = os.environ.get("ROBOMAGE_BROWSER_SMOKE") == "1"
        self._smoke_pending = self._smoke
        self._smoke_wait_summary = False

        self._build_ui()
        self._build_shortcuts()

    # ----- UI construction -----

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        self._hsplit = QSplitter(Qt.Horizontal)
        root.addWidget(self._hsplit)

        # Sidebar.
        side = QWidget()
        sv = QVBoxLayout(side)
        sv.setContentsMargins(4, 4, 4, 4)
        sv.setSpacing(4)
        self._summary = QLabel("Loading model…" if self._has_engine
                               else "No traces loaded")
        self._summary.setWordWrap(True)
        sv.addWidget(self._summary)
        games_head = QLabel("Games")
        games_head.setStyleSheet("color: #b8b8c0; font-weight: bold;")
        sv.addWidget(games_head)
        self._games_list = QListWidget()
        self._games_list.setFont(_mono_font())
        self._games_list.currentRowChanged.connect(self._on_game_row)
        sv.addWidget(self._games_list, 1)
        self._follow = QCheckBox("Follow live game")
        self._follow.setChecked(True)
        if self._shards or not self._has_engine:
            self._follow.hide()      # nothing streams in shard/browse-only mode
        sv.addWidget(self._follow)
        ana_head = QLabel("Analyses")
        ana_head.setStyleSheet("color: #b8b8c0; font-weight: bold;")
        sv.addWidget(ana_head)
        self._menu_list = QListWidget()
        for key, label, _fn in bs.ANALYSES:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            self._menu_list.addItem(item)
        engine_ok = self._has_engine and not self._shards
        for key, label in bs.ENGINE_MENU:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            if not engine_ok:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip(_MSG_SHARD if self._shards
                                else "No live env — browse-only session.")
            self._menu_list.addItem(item)
        # Net probes (az_inspect over the browsed records): per-decision
        # block-importance / card-swap / sweeps / recorded-π-vs-net, plus the
        # pooled KL and calibration views. Work in every session mode; the
        # probe net (AZ checkpoint, else PPO warm-start) loads on first use.
        for key, label in shard_probes.PROBE_MENU:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, key)
            self._menu_list.addItem(item)
        self._menu_list.itemClicked.connect(
            lambda item: self._run_menu_entry(item.data(Qt.UserRole)))
        sv.addWidget(self._menu_list, 1)
        self._stop_btn = QPushButton("Stop simulating")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_collect)
        if not engine_ok:
            self._stop_btn.hide()
        sv.addWidget(self._stop_btn)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("color: #9a9aa2;")
        sv.addWidget(self._status_label)
        side.setMinimumWidth(240)
        self._hsplit.addWidget(side)

        # Right: tabs over the histogram.
        right = QSplitter(Qt.Vertical)
        self._tabs = QTabWidget()
        self._provider = ImageProvider()
        self._popup = OraclePopup(self._provider, self)
        board_split = QSplitter(Qt.Vertical)
        self._board = TraceBoard(self._provider, self._popup)
        self._decision = DecisionPanel()
        board_split.addWidget(self._board)
        board_split.addWidget(self._decision)
        board_split.setStretchFactor(0, 3)
        board_split.setStretchFactor(1, 1)
        self._tabs.addTab(board_split, "Board")
        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(_mono_font())
        self._tabs.addTab(self._output, "Analysis output")
        right.addWidget(self._tabs)
        hist_box = QWidget()
        hv = QVBoxLayout(hist_box)
        hv.setContentsMargins(4, 0, 4, 2)
        hv.setSpacing(1)
        self._hist_caption = QLabel(
            "V(s) over game — click a bar to jump to that decision")
        self._hist_caption.setStyleSheet("color: #9a9aa2;")
        hv.addWidget(self._hist_caption)
        self._hist = ValueHistogramWidget()
        self._hist.step_picked.connect(self._set_step)
        self._hist.step_hovered.connect(self._on_hist_hover)
        hv.addWidget(self._hist, 1)
        right.addWidget(hist_box)
        right.setStretchFactor(0, 5)
        right.setStretchFactor(1, 1)
        right.setSizes([600, 150])
        self._hsplit.addWidget(right)
        self._hsplit.setStretchFactor(0, 0)
        self._hsplit.setStretchFactor(1, 1)
        self._hsplit.setSizes([280, 900])
        self._board.clear("Waiting for the first simulated game…"
                          if self._has_engine else "(no traces loaded)")

    def _build_shortcuts(self):
        """Pane-scoped shortcuts (dead while the pane is hidden, so they never
        fight the play mode's app-wide arrow filter)."""
        binds = [("Left", lambda: self._step_by(-1)),
                 ("Right", lambda: self._step_by(1)),
                 ("Shift+Left", lambda: self._step_by(-10)),
                 ("Shift+Right", lambda: self._step_by(10)),
                 ("Home", lambda: self._set_step(0)),
                 ("End", self._step_end),
                 ("W", lambda: self._run_menu_entry("whatif"))]
        for seq, fn in binds:
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(fn)

    # ----- SessionManager contract -----

    def start(self):
        """Submit the initial load+collect job (no-op in traces-only mode).
        Construction is cheap by design — nothing loads until here."""
        if self._started or self._worker is None:
            return
        self._started = True
        self._worker.start()
        self._submit_collect("load", getattr(self._args, "n_games", 20))

    def shutdown(self):
        """Bounded-blocking teardown: stop the running collect, queue the
        shutdown sentinel (the worker closes the env on its own thread), join
        with a timeout, abandon on overrun (daemon thread; the engine
        subprocess dies with the process)."""
        self._closed = True
        self._refresh_timer.stop()
        if self._worker is not None:
            if self._collect_stop is not None:
                self._collect_stop.set()
            self._worker.submit(("shutdown",))
            if self._worker.is_alive():
                self._worker.join(5.0)
        t = self._analysis_thread
        if t is not None and t.is_alive():
            t.join(2.0)

    def is_busy(self):
        return bool(self._store.engine_busy)

    def title(self):
        if self._subtitle:
            return self._subtitle
        if self._shards:
            return f"shard replay: {self._args.shards}"
        if self._has_engine:
            return (f"{self._args.model}  vs  {self._args.opponent}"
                    + ("  · bo3" if getattr(self._args, "bo3", False) else ""))
        prov = self._provenance_loaded or {}
        if prov.get("model"):
            return f"{prov['model']} traces (opened)"
        return "analysis traces"

    def traces(self):
        """Finished games only (the live placeholder is excluded; whatif
        branches and shard records are included — they are complete traces)."""
        return [g for g in self._store.games if not g.get("live")]

    def provenance(self):
        if not self._has_engine and self._provenance_loaded is not None:
            return dict(self._provenance_loaded)
        return {k: getattr(self._args, k, d) for k, d in _ARG_DEFAULTS.items()}

    def load_traces(self, traces, provenance):
        """Append opened traces on the UI thread (no worker). In a session
        with no env this is the whole content (browse-only)."""
        self._provenance_loaded = dict(provenance or {})
        self._loading_traces = True
        try:
            for g in traces:
                self._on_engine_event(bs.GameAdded(g))
        finally:
            self._loading_traces = False
        if not self._has_engine:
            self._summary.setText(bs.summary_line(self._store.games))
        self._flush_refresh()

    def set_active(self, active):
        self._active = bool(active)

    # ----- engine job submission -----

    def _submit_collect(self, kind, n):
        stop = threading.Event()
        self._collect_stop = stop
        self._store.engine_busy = True
        self._busy_kind = "sim"
        self._stop_btn.setEnabled(True)
        self._mark(summary=True)
        self._worker.submit((kind, n, stop))

    def _stop_collect(self):
        if self._collect_stop is not None:
            self._collect_stop.set()
            self._stop_btn.setEnabled(False)
            self._say("Stopping after the current game…")

    # ----- engine events (UI thread; queued from the worker) -----

    def _on_engine_event(self, ev):
        if self._closed:
            return
        applied = self._store.apply(ev)

        if isinstance(ev, bs.EnvReady):
            self._subtitle = ev.subtitle
            if ev.startup_text.strip():
                self._log_output("startup", ev.startup_text)
            self._say(ev.subtitle or "env ready")
            self._mark(summary=True)
        elif isinstance(ev, bs.LoadFailed):
            self._load_failed = True
            self._summary.setText(ev.text.splitlines()[0] if ev.text else
                                  "model/env load failed")
            self._summary.setStyleSheet("color: #e05555;")
            self._log_output("error", ev.text)
            self._tabs.setCurrentWidget(self._output)
        elif isinstance(ev, bs.EngineNote):
            self._log_output("note", ev.text)
        elif isinstance(ev, bs.AnalysisDone):
            # Engine-side results (whatif comparison table, sim errors).
            self._busy_kind = None
            self._log_output(ev.title, ev.text)
            self._tabs.setCurrentWidget(self._output)
        elif isinstance(ev, bs.EngineIdle):
            self._busy_kind = None
            self._collect_stop = None
            self._stop_btn.setEnabled(False)
            self._mark(summary=True)
            self._say("engine idle")
            if self._smoke_pending:
                self._smoke_pending = False
                self._run_smoke()
        elif isinstance(ev, bs.GameStarted):
            self._mark(rows={applied.game_idx}, summary=True)
            if self._follow.isChecked() or self._store.cur_game is None:
                self._store.select_game(self._store.live_idx)
                self._mark(selected=True)
        elif isinstance(ev, bs.StepAppended):
            if applied.game_idx is not None:
                self._mark(rows={applied.game_idx})
            if applied.selected_grew:
                self._follow_advance()
                self._mark(selected=True)
        elif isinstance(ev, bs.OppActionAppended):
            if applied.selected_grew:
                self._mark(selected=True)
        elif isinstance(ev, bs.GameFinished):
            self._mark(rows={applied.game_idx}, summary=True)
            if applied.game_idx == self._store.cur_game:
                self._mark(selected=True)
            if not self._loading_traces:
                self._set_dirty(True)
        elif isinstance(ev, bs.GameAborted):
            if applied.removed:
                self._mark(rebuild=True, summary=True)
                if self._store.cur_game is None:
                    self._mark(selected=True)     # clears the panes
        elif isinstance(ev, bs.GameAdded):
            self._mark(rows={applied.game_idx}, summary=True)
            if self._store.cur_game is None:
                self._store.select_game(applied.game_idx)
                self._mark(selected=True)
            if not self._loading_traces:
                self._set_dirty(True)

    def _follow_advance(self):
        """Live follow mode: when the selected game is the live game and the
        cursor sat on the previous last step, ride the new step (stepping back
        disengages naturally — the cursor is no longer at the end; End
        re-engages)."""
        st = self._store
        if not self._follow.isChecked() or st.cur_game != st.live_idx:
            return
        n = len(st.games[st.cur_game]["observations"])
        if n == 1 or st.cur_step == n - 2:
            st.cur_step = n - 1

    # ----- coalesced refresh -----

    def _mark(self, rows=None, rebuild=False, summary=False, selected=False):
        if rows:
            self._dirty_rows |= rows
        self._dirty_rebuild = self._dirty_rebuild or rebuild
        self._dirty_summary = self._dirty_summary or summary
        self._dirty_selected = self._dirty_selected or selected
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _flush_refresh(self):
        if self._closed:
            return
        self._refresh_timer.stop()
        rows, rebuild = self._dirty_rows, self._dirty_rebuild
        summary, selected = self._dirty_summary, self._dirty_selected
        self._dirty_rows = set()
        self._dirty_rebuild = self._dirty_summary = self._dirty_selected = False

        self._syncing_selection = True
        try:
            if rebuild:
                self._games_list.clear()
                for i, g in enumerate(self._store.games):
                    self._add_row(i, g)
            else:
                for i in sorted(rows):
                    if i >= len(self._store.games):
                        continue     # row vanished (abort) before the flush
                    while self._games_list.count() <= i:
                        self._add_row(self._games_list.count(),
                                      self._store.games[self._games_list.count()])
                    self._style_row(self._games_list.item(i), i,
                                    self._store.games[i])
            if (self._store.cur_game is not None
                    and self._games_list.currentRow() != self._store.cur_game):
                self._games_list.setCurrentRow(self._store.cur_game)
        finally:
            self._syncing_selection = False

        if summary and not self._load_failed:
            self._refresh_summary()
        if selected:
            self._refresh_selected()

    def _add_row(self, i, g):
        item = QListWidgetItem()
        self._games_list.addItem(item)
        self._style_row(item, i, g)

    def _style_row(self, item, i, g):
        item.setText(bs.game_label(i, g))
        item.setData(Qt.UserRole, i)
        item.setForeground(_LIVE_FG if g.get("live")
                           else QColor("#d8d8d8"))

    def _refresh_summary(self):
        text = bs.summary_line(
            self._store.games,
            loading=(self._store.engine_busy and self._busy_kind == "sim"))
        if self._store.engine_busy and self._busy_kind == "whatif":
            text += "  (branching…)"
        self._summary.setText(text)

    def _refresh_selected(self):
        g = self._store.selected()
        if g is None:
            self._board.clear("(no game selected)")
            self._decision.clear()
            self._hist.set_data([])
            self._hist_caption.setText(
                "V(s) over game — click a bar to jump to that decision")
            return
        step = self._store.clamp_step(self._store.cur_step)
        self._store.cur_step = step
        self._hist.set_data(g["values"], cursor=step)
        self._board.show_step(g, self._store.cur_game, step)
        if g["observations"]:
            self._decision.show_decision(g, step)
        else:
            self._decision.clear()
        self._refresh_caption()

    def _refresh_caption(self):
        g = self._store.selected()
        if g is None:
            return
        step = self._store.cur_step
        v = g["values"][step] if step < len(g["values"]) else None
        vs = f" · V={v:+.3f}" if v is not None else ""
        self._hist_caption.setText(
            f"game {self._store.cur_game} [{bs.result_str(g)}] · "
            f"step {step}/{max(len(g['values']) - 1, 0)}{vs}")

    # ----- selection / stepping -----

    def _on_game_row(self, row):
        if self._syncing_selection or row < 0:
            return
        item = self._games_list.item(row)
        idx = item.data(Qt.UserRole) if item is not None else row
        if idx == self._store.cur_game:
            return
        if self._store.select_game(idx):
            self._refresh_selected()
            self._tabs.setCurrentIndex(0)

    def _set_step(self, step):
        st = self._store
        if st.cur_game is None or not self._active:
            return
        step = st.clamp_step(step)
        if step == st.cur_step:
            return
        st.cur_step = step
        g = st.selected()
        self._hist.set_cursor(step)
        self._board.show_step(g, st.cur_game, step)
        self._decision.show_decision(g, step)
        self._refresh_caption()
        self._tabs.setCurrentIndex(0)

    def _step_by(self, delta):
        if self._store.cur_game is not None:
            self._set_step(self._store.cur_step + delta)

    def _step_end(self):
        g = self._store.selected()
        if g is not None:
            self._set_step(len(g["observations"]) - 1)

    def _on_hist_hover(self, step, value):
        if step is None:
            self._refresh_caption()
        else:
            vs = f" · V={value:+.3f}" if value is not None else ""
            self._hist_caption.setText(f"step {step}{vs}")

    # ----- analyses / engine menu -----

    def _run_menu_entry(self, key):
        if not self._active:
            return
        if key in ("whatif", "run5", "run20"):
            self._run_engine_entry(key)
            return
        if key in shard_probes.PROBE_KEYS:
            self._run_probe_entry(key)
            return
        pool = bs.analysis_pool(self._store.games)
        if not pool:
            self._say("No games simulated yet.")
            return
        if self._store.analysis_busy:
            self._say("An analysis is already running.")
            return
        entry = next((e for e in bs.ANALYSES if e[0] == key), None)
        if entry is None:
            return
        _key, _label, fn = entry
        self._store.analysis_busy = True
        self._say(f"Running {key} on {len(pool)} games…")
        bridge = self._bridge

        def runner():
            try:
                text = bs.capture(fn, pool)
            except Exception:  # noqa: BLE001 — report, never crash the UI
                text = traceback.format_exc()
            bridge.analysis_done.emit((key, text))

        self._analysis_thread = threading.Thread(
            target=runner, name="gui-browser-analysis", daemon=True)
        self._analysis_thread.start()

    def _run_probe_entry(self, key):
        """az_inspect net probes over the browsed records (shard_probes).

        Snapshot on the UI thread (cheap list copies; step ndarrays are
        append-only), stack + torch on the one-shot analysis thread. The probe
        net loads on first use and is cached; the analysis_busy gate keeps a
        single worker touching it."""
        if self._store.analysis_busy:
            self._say("An analysis is already running.")
            return
        if (key in shard_probes.DECISION_PROBES
                and self._store.cur_game is None):
            self._say(_MSG_NO_SEL)
            return
        snap = shard_probes.snapshot(self._store.games, self._store.cur_game,
                                     self._store.cur_step)
        if not any(c["observations"] for c in snap["games"]):
            self._say("No browsable decisions yet.")
            return
        model_spec = (getattr(self._args, "model", None)
                      or (self._provenance_loaded or {}).get("model") or "gen")
        self._store.analysis_busy = True
        self._say(f"Running {key}…")
        bridge = self._bridge

        def runner():
            try:
                if self._probe_net is None:
                    self._probe_net, self._probe_net_label = \
                        shard_probes.load_probe_net(model_spec)
                lines = shard_probes.run_probe(key, self._probe_net, snap)
                text = "\n".join(
                    [f"probe net: {self._probe_net_label}", ""] + lines)
            except Exception:  # noqa: BLE001 — report, never crash the UI
                text = traceback.format_exc()
            bridge.analysis_done.emit((key, text))

        self._analysis_thread = threading.Thread(
            target=runner, name="gui-browser-probe", daemon=True)
        self._analysis_thread.start()

    def _run_engine_entry(self, key):
        """Gate chain (verbatim parity with the TUI): live env → not busy →
        (whatif) a selected game. A live/unreplayable game passes through —
        _run_whatif prints its own refusal into the output tab."""
        core = self._worker._core if self._worker is not None else None
        if core is None or self._shards or not core.has_env:
            self._say(_MSG_SHARD if self._shards
                      else (_MSG_BROWSE_ONLY if core is None else _MSG_NO_ENV))
            return
        if self._store.engine_busy:
            self._say(_MSG_BUSY)
            return
        if key == "whatif":
            if self._store.cur_game is None:
                self._say(_MSG_NO_SEL)
                return
            gn, step = self._store.cur_game, self._store.cur_step
            self._store.engine_busy = True
            self._busy_kind = "whatif"
            self._mark(summary=True)
            self._say(f"Branching whatif at game {gn}, step {step}…")
            self._worker.submit(("whatif", gn, step, 3,
                                 self._store.games[gn]))
        else:
            n = 5 if key == "run5" else 20
            self._say(f"Simulating {n} more games…")
            self._submit_collect("collect", n)

    def _on_analysis_done(self, payload):
        if self._closed:
            return
        title, text = payload
        self._store.analysis_busy = False
        self._log_output(title, text)
        self._tabs.setCurrentWidget(self._output)
        if self._smoke_wait_summary and title == "summary":
            self._smoke_wait_summary = False
            self._smoke_report()

    # ----- misc -----

    def _log_output(self, title, text):
        head = f"── {title} " + "─" * max(0, 60 - len(title))
        self._output.appendPlainText(head)
        self._output.appendPlainText((text or "").rstrip("\n") or "(no output)")
        self._output.appendPlainText("")

    def _say(self, text):
        self._status_label.setText(text)
        self.status.emit(text)

    def _set_dirty(self, dirty):
        if dirty != self._dirty:
            self._dirty = dirty
            self.dirty_changed.emit(dirty)

    # ----- smoke hook (ROBOMAGE_BROWSER_SMOKE=1) -----

    def _run_smoke(self):
        """After the first EngineIdle: exercise selection, stepping, and one
        analysis, then report. The ci wiring lives elsewhere — this is just
        the in-pane auto-drive."""
        self._flush_refresh()
        if self._store.games:
            self._syncing_selection = True
            self._games_list.setCurrentRow(0)
            self._syncing_selection = False
            self._store.select_game(0)
            self._refresh_selected()
            for _ in range(3):
                self._step_by(1)
            self._step_end()
        if bs.analysis_pool(self._store.games):
            self._smoke_wait_summary = True
            self._run_menu_entry("summary")
        else:
            self._smoke_report()

    def _smoke_report(self):
        n = len(self.traces())
        print(f"BROWSER SMOKE OK: {n} games", flush=True)
        self.smoke_done.emit(n)
