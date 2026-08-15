"""Model-analysis browser as a full-screen Textual TUI.

Migrates the analysis.py interactive REPL into a terminal UI (a sibling of
tui_game.py). The app loads a checkpoint, simulates games against the chosen
opponent in a background thread (the same per-decision traces analysis.py
collects: observations, V(s), policy probs, actions), and then lets you:

  * pick a game from the sidebar and PAGE THROUGH its board states — the
    board rendered in tui_game.py's style (bordered card widgets with
    color-identity edges, battlefield/land rows, stack, the model's hand,
    life/mana info lines), one decision step at a time, with a decision
    panel showing the model's full policy distribution at that step (every
    legal action with its probability, chosen action marked) and the
    opponent's interleaved actions;
  * seek by CLICKING the V(s) histogram docked at the bottom — one bar per
    decision step (bucketed when the game is wider than the terminal),
    positive V above the zero line, negative below, cursor column highlighted;
  * run every REPL analysis view (summary, cardvalue, targeting, swings,
    boundaries, matchcal, regret, entropy, consistency, calibration, turning,
    clusters, sideboard, sbvalue, shap) from the sidebar menu — output lands
    in the "Analysis output" tab;
  * branch a counterfactual `whatif` at the current game/step (w key), and
    simulate more games, both on the live env. Each whatif ALTERNATIVE is
    grafted onto the source game's prefix and added to the games list as a
    full trace (marked ↳g<src>@<step>), so the counterfactual line can be
    selected, stepped through, and even re-branched — while staying excluded
    from the analysis/summary statistics pools (it is not an independent
    sample).

The matplotlib `chart *` commands and the HTML `report` battery stay in
analysis.py — this front end covers the text/interactive tools.

Run from the repo root (same simulation args as `analysis.py interactive`):
    train/.venv/bin/python train/tui_analysis.py <model.zip|deck> \
        --opponent scripted [--deck-b mav] [--n-games 20] [--bo3]

Shard replay — browse recorded AZ self-play instead of simulating (see
shard_replay.py; the model spec becomes the V(s) net, --no-net keeps the
recorded outcome z, and whatif/run stay disabled without a live env):
    train/.venv/bin/python train/tui_analysis.py gen \
        --shards train/az_data/gen [--seat A|B] [--no-net] [--n-games 20]
"""

import argparse
import io
import traceback
from contextlib import redirect_stdout

import numpy as np
from rich.text import Text

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import (Horizontal, HorizontalScroll, Vertical,
                                VerticalScroll)
from textual.message import Message
from textual.widgets import (Footer, Header, OptionList, RichLog, Static,
                             TabbedContent, TabPane)
from textual.widgets.option_list import Option

import analysis as an
import decode
from cli_spec import ANALYSIS_TUI_TOOL, apply_to_parser
from env import STATE_SIZE, _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE
# Board building blocks shared with the play board: the bordered card widget
# (color-identity edges) and the step-strip abbreviations.
from tui_game import CardButton, CardClicked, _edge_colors, _STEP_ABBR

# ── V(s) histogram colors ─────────────────────────────────────────────────────
# Diverging encoding: sign is carried by POSITION (above/below the zero line),
# so the two hues are redundant reinforcement, and the midline stays a neutral
# dim gray. Green-above / red-below matches the win/loss convention of the
# analysis.py matplotlib charts and tui_game.
_POS_COLOR = "green3"        # V > 0 — model favored
_NEG_COLOR = "red3"          # V < 0 — opponent favored
_CURSOR_COLOR = "yellow1"    # bar at the selected decision step
_CURSOR_BG = "grey27"        # full-height column marker under the cursor
_AXIS_STYLE = "dim"

# Eighth-block characters indexed by how many eighths of the cell are filled
# (from the bottom). Downward bars reuse them via reverse-video (see _cell).
_BLOCKS = " ▁▂▃▄▅▆▇█"

# The front-end-independent pieces live in browse_session (shared with the Qt
# browser, gui_browser.py): the ONE process-global capture lock, the analyses
# registry, and the label/summary helpers. Aliased to the old private names so
# the rest of this file is unchanged.
from browse_session import (CAPTURE_LOCK as _CAPTURE_LOCK, capture as _capture,
                            result_str as _result_str,
                            has_probs as _has_probs, probs_guard as _probs_guard,
                            run_shap as _run_shap, ANALYSES as _ANALYSES,
                            ENGINE_MENU as _ENGINE_MENU)


# ── Clickable V(s) histogram ──────────────────────────────────────────────────

class ValueHistogram(Static):
    """Bar-per-decision V(s) chart for one game; click a column to seek.

    Vertical scale is symmetric ±vmax where vmax = max(1, max|V|) — usually
    ±1, stretched only when a bo3 trace's V exceeds it (game rewards stack with
    the match terminal), so bars never clip. When the game has more decisions
    than plot columns, steps are bucketed and each column shows the bucket's
    max-|V| step (preserving swings that a mean would smooth away); clicking
    such a column seeks to that extreme step.
    """

    GUTTER = 5  # left scale labels: "+1.0 ", "   0 ", "-1.0 "

    class StepPicked(Message):
        def __init__(self, step: int):
            self.step = step
            super().__init__()

    class StepHovered(Message):
        def __init__(self, step, value):
            self.step = step          # int, or None when the mouse left the plot
            self.value = value
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._values = []
        self._cursor = 0
        self._vmax = 1.0      # symmetric plot scale: max(1, max|V|)
        self._cols = []       # per plot column: (value, representative step) | (None, step)
        self._bar_w = 1
        self._bucketed = False

    # ----- data -----

    def set_data(self, values, cursor=0) -> None:
        self._values = [float(v) for v in values]
        self._cursor = max(0, min(cursor, len(self._values) - 1)) if self._values else 0
        self._vmax = max(1.0, max((abs(v) for v in self._values), default=1.0))
        self._layout_cols()
        self.refresh()

    def set_cursor(self, step: int) -> None:
        if not self._values:
            return
        self._cursor = max(0, min(step, len(self._values) - 1))
        self.refresh()

    def on_resize(self, event: events.Resize) -> None:
        self._layout_cols()
        self.refresh()

    # ----- geometry -----

    def _plot_width(self) -> int:
        return max(0, self.content_size.width - self.GUTTER)

    def _layout_cols(self) -> None:
        """Rebuild the column -> (value, step) mapping for the current width."""
        w = self._plot_width()
        n = len(self._values)
        self._cols = []
        self._bucketed = False
        self._bar_w = 1
        if not n or w <= 0:
            return
        if n <= w:
            # Whole steps fit: widen bars up to 3 cells, with a 1-cell spacer
            # between bars when there's room (the spacer still maps to its step
            # so clicks in the gap don't dead-zone).
            per = max(1, min(4, w // n))
            self._bar_w = per - 1 if per >= 2 else 1
            for s, v in enumerate(self._values):
                self._cols.extend([(v, s)] * self._bar_w)
                if per >= 2:
                    self._cols.append((None, s))
        else:
            self._bucketed = True
            for c in range(w):
                lo = c * n // w
                hi = max(lo + 1, (c + 1) * n // w)
                rep = max(range(lo, hi), key=lambda s: abs(self._values[s]))
                self._cols.append((self._values[rep], rep))

    def _cursor_cols(self):
        """Set of plot-column indices highlighted for the cursor step."""
        n = len(self._values)
        if not n or not self._cols:
            return set()
        if self._bucketed:
            return {min(self._cursor * len(self._cols) // n, len(self._cols) - 1)}
        per = self._bar_w + (1 if len(self._cols) > n * self._bar_w else 0)
        start = self._cursor * per
        return set(range(start, min(start + self._bar_w, len(self._cols))))

    def _step_at(self, x: int):
        """Map a widget-relative x coordinate to a decision step (or None)."""
        dx = self.content_region.x - self.region.x   # border/padding offset
        col = x - dx - self.GUTTER
        if 0 <= col < len(self._cols):
            return self._cols[col][1]
        return None

    # ----- rendering -----

    @staticmethod
    def _cell(v, row, half, mid):
        """(char, style) for one plot cell; `v` is already normalized to
        [-1, 1]. Rows count from the top; `mid` is the zero-line row. Upward
        bars use bottom-eighth blocks directly; downward (hanging) bars render
        the COMPLEMENT block in reverse video so the filled fraction hangs
        from the zero line."""
        if row == mid:
            return "─", _AXIS_STYLE
        if v is None:
            return " ", ""
        if row < mid:
            if v <= 0:
                return " ", ""
            k = (mid - 1) - row                     # cells above the zero line
            fill = min(max(min(v, 1.0) * half - k, 0.0), 1.0)
            e = round(fill * 8)
            return (" ", "") if e <= 0 else (_BLOCKS[e], _POS_COLOR)
        if v >= 0:
            return " ", ""
        k = row - (mid + 1)                         # cells below the zero line
        fill = min(max(min(-v, 1.0) * half - k, 0.0), 1.0)
        e = round(fill * 8)
        if e <= 0:
            return " ", ""
        if e >= 8:
            return "█", _NEG_COLOR
        return _BLOCKS[8 - e], f"{_NEG_COLOR} reverse"

    def render(self):
        width, height = self.content_size.width, self.content_size.height
        n = len(self._values)
        if not n or width <= self.GUTTER + 1 or height < 5:
            return Text("(no game selected)", style="dim")
        half = (height - 2) // 2                    # rows above/below the zero line
        mid = half
        plot_h = 2 * half + 1
        cursor_cols = self._cursor_cols()
        vmax = self._vmax
        top = f"+{vmax:.1f}" if vmax != 1.0 else "+1"
        bot = f"-{vmax:.1f}" if vmax != 1.0 else "-1"
        gut_labels = {0: top, mid: "0", plot_h - 1: bot}

        out = Text()
        for row in range(plot_h):
            out.append(f"{gut_labels.get(row, ''):>{self.GUTTER - 1}} ",
                       style=_AXIS_STYLE)
            for c, (v, _step) in enumerate(self._cols):
                ch, style = self._cell(None if v is None else v / vmax,
                                       row, half, mid)
                if c in cursor_cols:
                    if style.startswith((_POS_COLOR, _NEG_COLOR)):
                        style = style.replace(_POS_COLOR, _CURSOR_COLOR) \
                                     .replace(_NEG_COLOR, _CURSOR_COLOR)
                    style = f"{style} on {_CURSOR_BG}".strip()
                out.append(ch, style or None)
            out.append("\n")
        out.append(self._axis_row(len(self._cols)))
        return out

    def _axis_row(self, used: int) -> Text:
        """Step-number ticks: first step, cursor step (bold), last step."""
        n = len(self._values)
        row = [" "] * (self.GUTTER + used)
        spans = []   # (start, text, style)

        def place(col, text, style):
            if len(text) > used:             # tick can't fit in the plot at all
                return
            start = max(self.GUTTER, min(self.GUTTER + used - len(text), col))
            for s, t, _ in spans:            # skip ticks that would collide
                if start < s + len(t) and s < start + len(text):
                    return
            spans.append((start, text, style))
            for i, ch in enumerate(text):
                row[start + i] = ch

        cur_col = min(self._cursor_cols() or {0})
        place(self.GUTTER + cur_col - 1, f"^{self._cursor}", f"bold {_CURSOR_COLOR}")
        place(self.GUTTER, "0", _AXIS_STYLE)
        place(self.GUTTER + used - len(str(n - 1)), str(n - 1), _AXIS_STYLE)

        out = Text("".join(row), style=_AXIS_STYLE)
        for start, text, style in spans:
            out.stylize(style, start, start + len(text))
        return out

    # ----- mouse -----

    def on_click(self, event: events.Click) -> None:
        if event.button != 1:
            return
        step = self._step_at(event.x)
        if step is not None:
            self.post_message(self.StepPicked(step))

    def on_mouse_move(self, event: events.MouseMove) -> None:
        step = self._step_at(event.x)
        value = self._values[step] if step is not None else None
        self.post_message(self.StepHovered(step, value))

    def on_leave(self, event: events.Leave) -> None:
        self.post_message(self.StepHovered(None, None))


# ── Worker → UI messages ──────────────────────────────────────────────────────

class EnvReady(Message):
    def __init__(self, startup_text):
        self.startup_text = startup_text
        super().__init__()


class GameAdded(Message):
    def __init__(self, game):
        self.game = game
        super().__init__()


class EngineIdle(Message):
    """The engine worker finished its current job (collect/whatif)."""


class AnalysisDone(Message):
    def __init__(self, title, text):
        self.title = title
        self.text = text
        super().__init__()


class LoadFailed(Message):
    def __init__(self, text):
        self.text = text
        super().__init__()


# ── The app ───────────────────────────────────────────────────────────────────

class AnalysisApp(App):
    CSS = """
    #sidebar    { width: 36; border-right: solid $accent; }
    #summary    { height: 2; padding: 0 1; color: $text-muted; }
    .head       { height: 1; padding: 0 1; color: $accent; text-style: bold; }
    #games      { height: 1fr; border: round $primary; }
    #analyses   { height: auto; max-height: 45%; border: round $surface; }
    #output     { height: 1fr; }
    #vhist      { height: 15; border: round $primary; }
    /* Board tab: the played board rendered like tui_game (opponent's rows
       flipped so the two battlefields sit adjacent across the stack). */
    #board-col  { height: 1fr; }
    #phase      { height: 1; background: $panel; color: $text; }
    #opp-info   { height: 1; color: red; }
    #self-info  { height: 1; color: green; }
    #opp-bf, #self-bf { height: 7; }
    .bf-row     { height: 1fr; layout: horizontal; }
    .bf-row.lands { background: $panel; }
    #stack      { height: 3; border: round $primary; }
    .stack-item { width: auto; height: 100%; margin: 0 1; padding: 0 1;
                  border-left: thick $secondary; }
    .stack-empty { width: auto; height: 100%; color: $text-muted; }
    /* Height 5 = 2 border rows + 3 inner rows, so a bordered card keeps a
       text row (at 4 the cards render as empty rectangles). */
    #self-hand  { height: 5; border: round green; layout: horizontal; }
    #graveyards { height: 3; color: $text-muted; }
    CardButton  { width: auto; height: 100%; margin: 0 1; padding: 0 1;
                  border: round $surface; }
    CardButton:hover { background: $boost; }
    /* Decision panel: full-width strip between the board and the histogram
       (a right-hand column clipped off narrow terminals). */
    #decision   { height: 12; border: round $accent; }
    #decision-scroll { height: 1fr; }
    #decision-body { padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("left", "step(-1)", "◀ step", priority=True),
        Binding("right", "step(1)", "step ▶", priority=True),
        Binding("shift+left", "step(-10)", "◀ x10", priority=True),
        Binding("shift+right", "step(10)", "x10 ▶", priority=True),
        Binding("home", "step_home", "first", priority=True),
        Binding("end", "step_end", "last", priority=True),
        Binding("w", "whatif", "Whatif @ step"),
    ]

    # ── Responsive vertical budget ────────────────────────────────────────────
    # The board must stay fully visible above the decision panel at any terminal
    # height. _relayout distributes the available rows across the adjustable
    # regions, giving each its comfortable height when there is room and
    # shrinking toward its minimum when there isn't. Tuples are (min, comfortable);
    # the CSS defaults must equal the comfortable values (they are the pre-measure
    # starting state). _relayout gives up rows in priority order — histogram
    # first, then the decision box, then the board's own panels last — so the
    # board stays whole for as long as possible.
    # Panel minima keep their cards renderable (a bordered card needs 3 outer
    # rows for one text row; the split battlefield panels hold two such rows, and
    # #self-hand is fixed at 5 per its CSS comment — below it the hand cards
    # collapse to empty rectangles).
    _VHIST_H = (7, 15)
    _DECISION_H = (6, 12)
    _PANEL_H = {"#opp-bf": (6, 7), "#self-bf": (6, 7), "#self-hand": (5, 5)}
    # Non-resizable board rows: phase(1) + opp-info(1) + stack(3) + self-info(1)
    # + graveyards(3). The board never shrinks below these plus the panel minima.
    _BOARD_FIXED_H = 9

    def __init__(self, args):
        super().__init__()
        self._args = args
        self._games = []
        self._model = None
        self._env = None
        self._opp_model = None
        self._cur_game = None       # index into self._games
        self._cur_step = 0
        self._engine_busy = True    # startup load+collect owns the env first
        self._analysis_busy = False
        # Rows consumed by the chrome (header + footer + tab bar) — everything
        # that is neither the histogram nor the tab's own content. Measured once
        # from the live layout so _relayout never hardcodes Textual's tab-bar
        # height; see _chrome_overhead / _relayout.
        self._overhead = None

    # ----- layout -----

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("Loading model…", id="summary")
                yield Static("Games", classes="head")
                yield OptionList(id="games")
                yield Static("Analyses", classes="head")
                yield OptionList(id="analyses")
            with TabbedContent(id="tabs"):
                with TabPane("Board", id="tab-board"):
                    with VerticalScroll(id="board-col"):
                        yield Static(id="phase")
                        yield Static(id="opp-info")
                        with Vertical(id="opp-bf"):
                            yield VerticalScroll(id="opp-bf-lands",
                                                 classes="bf-row lands")
                            yield VerticalScroll(id="opp-bf-perms",
                                                 classes="bf-row")
                        yield HorizontalScroll(id="stack")
                        with Vertical(id="self-bf"):
                            yield VerticalScroll(id="self-bf-perms",
                                                 classes="bf-row")
                            yield VerticalScroll(id="self-bf-lands",
                                                 classes="bf-row lands")
                        yield VerticalScroll(id="self-hand")
                        yield Static(id="self-info")
                        yield Static(id="graveyards")
                    with Vertical(id="decision"):
                        with VerticalScroll(id="decision-scroll"):
                            yield Static(id="decision-body")
                with TabPane("Analysis output", id="tab-output"):
                    yield RichLog(id="output", wrap=True, highlight=False, markup=False)
        yield ValueHistogram(id="vhist")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "RoboMage · analysis"
        self.sub_title = (f"{self._args.model}  vs  {self._args.opponent}"
                          + ("  (bo3)" if getattr(self._args, "bo3", False) else ""))
        menu = self.query_one("#analyses", OptionList)
        for key, label, _fn in _ANALYSES:
            menu.add_option(Option(label, id=key))
        for key, label in _ENGINE_MENU:
            menu.add_option(Option(label, id=key))
        hist = self.query_one("#vhist", ValueHistogram)
        hist.border_title = "V(s) over game — click a bar to jump to that decision"
        self.query_one("#decision", Vertical).border_title = \
            "Decision — legal actions with policy P(a)"
        self.query_one("#phase", Static).update(
            "[dim]Waiting for the first simulated game…[/dim]")
        # Fit the board to the initial terminal size once the first layout pass
        # has assigned real widget heights (on_mount runs pre-layout).
        self.call_after_refresh(self._relayout)
        self._load_and_collect(self._args.n_games)

    # ----- responsive layout -----

    def on_resize(self, event: events.Resize) -> None:
        # self.size still reports the pre-resize height inside this handler, so
        # feed the new height straight from the event.
        self._relayout(event.size.height)

    def _chrome_overhead(self):
        """Rows taken by the header, footer, and tab bar (everything that is not
        the histogram or the tab's content), measured once from the live layout.

        With #board-col at 1fr and #decision fixed, board-col + decision always
        fills the tab's content region, so screen_height - (board-col + decision
        + vhist) isolates the chrome — a constant independent of how we later
        resize those regions. Uses outer_size (border-inclusive) so the measure
        is in the same units as the CSS/styles heights _relayout assigns.
        Returns None until the widgets report real sizes."""
        if self._overhead is not None:
            return self._overhead
        try:
            board = self.query_one("#board-col").outer_size.height
            dec = self.query_one("#decision").outer_size.height
            vh = self.query_one("#vhist", ValueHistogram).outer_size.height
        except Exception:
            return None
        if board <= 0 or vh <= 0:
            return None
        ov = self.size.height - (board + dec + vh)
        if ov < 0:
            return None
        self._overhead = ov
        return ov

    def _relayout(self, height=None) -> None:
        """Size the histogram, decision box, and board panels so the whole board
        fits above the decision box at the current terminal height, shrinking the
        least board-critical regions first (see the _VHIST_H note above). `height`
        is the live terminal height (passed from on_resize, where self.size lags);
        it defaults to self.size.height for the post-mount initial pass."""
        overhead = self._chrome_overhead()
        if overhead is None:
            return
        h = self.size.height if height is None else height
        budget = h - overhead
        if budget <= 0:
            return
        vhist = self._VHIST_H[1]
        decision = self._DECISION_H[1]
        panels = {sel: hi for sel, (_lo, hi) in self._PANEL_H.items()}
        panel_order = ["#self-hand", "#opp-bf", "#self-bf"]

        def total():
            return vhist + decision + self._BOARD_FIXED_H + sum(panels.values())

        pi = 0
        while total() > budget:
            if vhist > self._VHIST_H[0]:
                vhist -= 1
            elif decision > self._DECISION_H[0]:
                decision -= 1
            elif any(panels[s] > self._PANEL_H[s][0] for s in panel_order):
                # Round-robin the panels so they shrink evenly rather than
                # collapsing one before touching the next.
                for _ in range(len(panel_order)):
                    s = panel_order[pi % len(panel_order)]
                    pi += 1
                    if panels[s] > self._PANEL_H[s][0]:
                        panels[s] -= 1
                        break
            else:
                break   # all regions at their minimum: board-col scrolls (tiny term)

        self.query_one("#vhist", ValueHistogram).styles.height = vhist
        self.query_one("#decision", Vertical).styles.height = decision
        for sel, ph in panels.items():
            self.query_one(sel).styles.height = ph

    # ----- engine worker (owns the env; one job at a time) -----

    @work(thread=True, group="engine")
    def _load_and_collect(self, n_games: int) -> None:
        if getattr(self._args, "shards", None):
            self._load_shards(n_games)
            return
        try:
            buf = io.StringIO()
            with _CAPTURE_LOCK, redirect_stdout(buf):
                self._model, self._env, self._opp_model = an._load_model_and_env(self._args)
            self.post_message(EnvReady(buf.getvalue()))
        except BaseException as exc:   # _load_model_and_env may sys.exit()
            self.post_message(LoadFailed(
                f"model/env load failed: {exc!r}\n{traceback.format_exc()}"))
            return
        self._collect_games(n_games)

    def _load_shards(self, n_games: int) -> None:
        """Shard-replay startup: reconstruct match records from recorded AZ
        self-play instead of simulating. No env is created (self._env stays
        None, so the whatif/run entries stay gated); the model spec is loaded
        only as the V(s) net, unless --no-net keeps values at the recorded z.
        Runs inside the engine worker thread."""
        import shard_replay
        try:
            buf = io.StringIO()
            with _CAPTURE_LOCK, redirect_stdout(buf):
                model = None
                if not getattr(self._args, "no_net", False):
                    model = shard_replay.load_value_model(self._args.model)
                records = shard_replay.load_records(
                    self._args.shards,
                    viewpoint_is_a=getattr(self._args, "seat", "A") != "B",
                    limit=n_games or None,
                    interp_fn=an._extract_interpretable)
                if model is not None:
                    shard_replay.apply_net_values(model, records)
                print(f"{len(records)} match record(s) from {self._args.shards} "
                      f"(seat {getattr(self._args, 'seat', 'A')}, "
                      + ("net V(s))" if model is not None else "z values)"))
            self.post_message(EnvReady(buf.getvalue()))
            for g in records:
                self.post_message(GameAdded(g))
        except BaseException as exc:
            self.post_message(LoadFailed(
                f"shard load failed: {exc!r}\n{traceback.format_exc()}"))
        finally:
            self.post_message(EngineIdle())

    def _collect_games(self, n: int) -> None:
        """Simulate n games one at a time (posting each so the UI fills live).
        Runs inside an engine worker thread."""
        try:
            for _ in range(n):
                games = an._collect_game_traces(self._model, self._env,
                                                self._opp_model, 1, verbose=False)
                self.post_message(GameAdded(games[0]))
        except Exception:
            self.post_message(AnalysisDone("simulation error",
                                           traceback.format_exc()))
        finally:
            self.post_message(EngineIdle())

    @work(thread=True, group="engine")
    def _collect_worker(self, n: int) -> None:
        self._collect_games(n)

    @work(thread=True, group="engine")
    def _whatif_worker(self, gn: int, step: int, k: int) -> None:
        try:
            buf = io.StringIO()
            with _CAPTURE_LOCK, redirect_stdout(buf):
                branches = an._run_whatif(self._model, self._env, self._opp_model,
                                          self._games[gn], gn, step, k,
                                          make_traces=True)
            text = buf.getvalue()
            # Each alternative branch arrives as a full game trace — add it to
            # the games list so the counterfactual line can be selected and
            # stepped through (the chosen branch's line IS the source game).
            added = 0
            for b in branches or []:
                if b.get("trace_game") is not None:
                    self.post_message(GameAdded(b["trace_game"]))
                    added += 1
            if added:
                text += (f"\n  Added {added} branch trace(s) to the games list "
                         f"(marked ↳g{gn}@{step}) — select one to step through it.")
            self.post_message(AnalysisDone(f"whatif — game {gn}, step {step}", text))
        except Exception:
            self.post_message(AnalysisDone("whatif error", traceback.format_exc()))
        finally:
            self.post_message(EngineIdle())

    # ----- analysis worker (trace crunching only; no env) -----

    @work(thread=True, group="analysis")
    def _analysis_worker(self, title: str, fn) -> None:
        try:
            # Whatif branch traces are counterfactual lines sharing their
            # source game's prefix, not independent samples — pooling them
            # would bias every statistic, so analyses run on real games only.
            games = [g for g in self._games if not g.get("whatif")]
            text = _capture(fn, games)
            self.post_message(AnalysisDone(title, text))
        except Exception:
            self.post_message(AnalysisDone(f"{title} error", traceback.format_exc()))
        finally:
            self._analysis_busy = False

    # ----- message handlers -----

    def on_env_ready(self, message: EnvReady) -> None:
        if getattr(self._args, "shards", None):
            net = ("z values" if getattr(self._args, "no_net", False)
                   else f"V(s): {self._args.model}")
            self.sub_title = (f"shard replay: {self._args.shards} · "
                              f"seat {getattr(self._args, 'seat', 'A')} · {net}")
        else:
            deck_a = getattr(self._args, "deck_a", None) or "?"
            deck_b = getattr(self._args, "deck_b", None) or "?"
            self.sub_title = (f"{deck_a} (model)  vs  {deck_b} ({self._args.opponent})"
                              + ("  · bo3" if getattr(self._args, "bo3", False) else ""))
        if message.startup_text.strip():
            self._log_output("startup", message.startup_text)
        self._refresh_summary(loading=True)

    def on_load_failed(self, message: LoadFailed) -> None:
        self._engine_busy = False
        self.query_one("#summary", Static).update(Text(message.text, style="red"))
        self._log_output("error", message.text)

    async def on_game_added(self, message: GameAdded) -> None:
        self._games.append(message.game)
        g = message.game
        i = len(self._games) - 1
        w = g.get("whatif")
        if w:
            # A whatif branch trace: mark its origin instead of the bo3 score.
            label = (f"{i:>3}  {_result_str(g):<4} {len(g['values']):>4}d  "
                     f"↳g{w['src_game']}@{w['step']}")
        else:
            sc = an._match_score(g)
            sc_str = f"{sc[0]}-{sc[1]}" if sc is not None else " — "
            side = "A" if g["model_is_a"] else "B"
            label = (f"{i:>3}  {_result_str(g):<4} {sc_str:<4} "
                     f"{len(g['values']):>4}d  {side}")
            if g.get("shard"):
                label += "  ⛁"
        self.query_one("#games", OptionList).add_option(Option(label, id=str(i)))
        self._refresh_summary(loading=self._engine_busy)
        if self._cur_game is None:
            await self._select_game(0)

    def on_engine_idle(self, message: EngineIdle) -> None:
        self._engine_busy = False
        self._refresh_summary()

    def on_analysis_done(self, message: AnalysisDone) -> None:
        self._log_output(message.title, message.text)
        self.query_one("#tabs", TabbedContent).active = "tab-output"

    # ----- selection / stepping -----

    async def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "games":
            await self._select_game(int(event.option.id))
        elif event.option_list.id == "analyses":
            self._run_menu_entry(event.option.id)

    async def on_value_histogram_step_picked(self, message: ValueHistogram.StepPicked) -> None:
        await self._set_step(message.step)

    def on_value_histogram_step_hovered(self, message: ValueHistogram.StepHovered) -> None:
        hist = self.query_one("#vhist", ValueHistogram)
        if message.step is None:
            self._refresh_hist_subtitle()
        else:
            hist.border_subtitle = f"step {message.step} · V={message.value:+.3f}"

    async def action_step(self, delta: int) -> None:
        if self._cur_game is not None:
            await self._set_step(self._cur_step + delta)

    async def action_step_home(self) -> None:
        await self._set_step(0)

    async def action_step_end(self) -> None:
        if self._cur_game is not None:
            await self._set_step(len(self._games[self._cur_game]["values"]) - 1)

    def action_whatif(self) -> None:
        self._run_menu_entry("whatif")

    async def _select_game(self, gn: int) -> None:
        if not (0 <= gn < len(self._games)):
            return
        self._cur_game = gn
        self._cur_step = 0
        g = self._games[gn]
        hist = self.query_one("#vhist", ValueHistogram)
        hist.set_data(g["values"], cursor=0)
        await self._show_step()
        self.query_one("#tabs", TabbedContent).active = "tab-board"

    async def _set_step(self, step: int) -> None:
        if self._cur_game is None:
            return
        n = len(self._games[self._cur_game]["observations"])
        step = max(0, min(step, n - 1))
        if step == self._cur_step:
            return
        self._cur_step = step
        self.query_one("#vhist", ValueHistogram).set_cursor(step)
        await self._show_step()
        self.query_one("#tabs", TabbedContent).active = "tab-board"

    # ----- board pane (tui_game-style card objects) -----

    async def _show_step(self) -> None:
        """Render the current game/step: the board as card widgets plus the
        decision panel. The recorded obs is always from the MODEL's perspective
        (traces cover model decisions only), so 'self' is the model — no
        mirroring is ever needed."""
        gn, step = self._cur_game, self._cur_step
        g = self._games[gn]
        obs = g["observations"][step]
        gs = decode.decode_game_state(obs[:STATE_SIZE],
                                      labels=decode.SELF_OPP_LABELS)

        self.query_one("#phase", Static).update(self._phase_strip(g, gn, step, obs, gs))
        self.query_one("#opp-info", Static).update(
            self._info_line("OPPONENT", gs["opponent"], gs["opp_library"]))
        self.query_one("#self-info", Static).update(
            self._info_line("MODEL   ", gs["self"], gs["self_library"]))
        self.query_one("#graveyards", Static).update(
            f"Model GY: {', '.join(gs['self_graveyard']) or '—'}\n"
            f"Opp GY:   {', '.join(gs['opp_graveyard']) or '—'}")

        await self._rebuild_stack(gs["stack"])
        await self._rebuild_bf("#opp-bf-perms", "#opp-bf-lands",
                               gs["opp_battlefield"], "opp")
        await self._rebuild_bf("#self-bf-perms", "#self-bf-lands",
                               gs["self_battlefield"], "self")
        await self._rebuild_hand(gs["self_hand"])
        self.query_one("#decision-body", Static).update(
            self._decision_text(g, step, obs))
        num_ch = g["num_choices"][step] if step < len(g["num_choices"]) else 0
        self.query_one("#decision", Vertical).border_subtitle = \
            f"{num_ch} legal · scroll for more"
        self.query_one("#decision-scroll", VerticalScroll).scroll_home(animate=False)
        self._refresh_hist_subtitle()

    def _phase_strip(self, g, gn, step, obs, gs) -> str:
        """One-line header: game/decision/V context plus the step strip."""
        cur = int(np.argmax(
            obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]))
        cells = " ".join(f"[reverse b]{a}[/reverse b]" if i == cur else f"[dim]{a}[/dim]"
                         for i, a in enumerate(_STEP_ABBR))
        val = g["values"][step] if step < len(g["values"]) else None
        vstr = f" · V={val:+.3f}" if val is not None else ""
        m = gs.get("match") or {}
        mstr = ""
        if any(m.get(k) for k in ("self_wins", "opp_wins", "is_sideboard")):
            mstr = f" · match {m['self_wins']}–{m['opp_wins']}"
            if m.get("is_sideboard"):
                mstr += " [b yellow]SIDEBOARD[/b yellow]"
        active = "A" if gs["active_is_a"] else "B"
        return (f"[b]G{gn} ({_result_str(g)}) · decision {step}/"
                f"{len(g['observations']) - 1}{vstr}[/b]{mstr}   "
                f"Turn {gs['turn']} · Active {active} · "
                f"Priority {gs['priority_player']}   " + cells)

    @staticmethod
    def _info_line(label, p, library) -> str:
        # Mirrors tui_game's info line: poison (☠) / energy (⚡) only when set.
        counters = ""
        if p.get("poison", 0) > 0:
            counters += f"  ☠ {p['poison']}"
        if p.get("energy", 0) > 0:
            counters += f"  ⚡ {p['energy']}"
        return (f"{label}  ♥ {p['life']}{counters}  "
                f"mana [{decode.fmt_mana(p['mana'])}]  "
                f"hand {p['hand_count']}  lib {library}")

    async def _rebuild_bf(self, perms_sel: str, lands_sel: str, perms,
                          controller: str) -> None:
        """One player's battlefield, split into non-land and land rows."""
        await self._fill_row(perms_sel,
                             [p for p in perms if not p.get("is_land")], controller)
        await self._fill_row(lands_sel,
                             [p for p in perms if p.get("is_land")], controller)

    async def _fill_row(self, selector: str, perms, controller: str) -> None:
        box = self.query_one(selector, VerticalScroll)
        await box.remove_children()
        widgets = [self._mk_card(decode.fmt_perm(p), p["card_idx"], controller,
                                 "battlefield")
                   for p in perms]
        if widgets:
            await box.mount(*widgets)

    async def _rebuild_hand(self, hand) -> None:
        box = self.query_one("#self-hand", VerticalScroll)
        await box.remove_children()
        widgets = [self._mk_card(c["name"], c["card_idx"], "self", "hand")
                   for c in hand]
        if widgets:
            await box.mount(*widgets)

    async def _rebuild_stack(self, stack) -> None:
        box = self.query_one("#stack", HorizontalScroll)
        await box.remove_children()
        if not stack:
            await box.mount(Static("Stack: (empty)", classes="stack-empty"))
            return
        widgets = []
        for e in stack:
            kind = "spell" if e["is_spell"] else "ability"
            label = f"{e['name']} ({kind}, {e['controller']})"
            if e.get("targets"):
                label += " → " + "; ".join(e["targets"])
            widgets.append(Static(label, classes="stack-item"))
        await box.mount(*widgets)

    @staticmethod
    def _mk_card(label: str, card_idx: int, controller: str,
                 zone: str) -> CardButton:
        """CardButton whose border edges encode the card's color identity
        (same builder as the play board)."""
        edges = _edge_colors(decode.card_border_colors(card_idx))
        return CardButton(label, card_idx, controller, edges, zone)

    def on_card_clicked(self, message: CardClicked) -> None:
        """Clicking a card in the replay shows its oracle text."""
        name = decode.card_index_to_name(message.card_idx)
        oracle = decode.card_oracle_text(message.card_idx)
        self.notify(oracle or "(no oracle text)", title=name or "card", timeout=8)

    # ----- decision panel -----

    def _decision_text(self, g, step, obs) -> Text:
        """The model's decision at this step: every legal action with its
        recorded policy probability (sorted most-likely first, chosen action
        marked), then the opponent's actions since the previous decision.
        Clocked seats (a clock= spec knob) get a match-clock line on top."""
        out = Text()
        clock_line = self._clock_line(g, step)
        if clock_line:
            out.append(clock_line + "\n", style="dim")
        num_ch = g["num_choices"][step] if step < len(g["num_choices"]) else 0
        chosen = g["actions"][step] if step < len(g.get("actions", [])) else None
        probs = None
        if g.get("action_probs") and step < len(g["action_probs"]):
            probs = g["action_probs"][step]

        order = range(num_ch)
        if probs is not None:
            order = sorted(order, key=lambda k: -probs[k])
        for k in order:
            p = float(probs[k]) if probs is not None else None
            desc = an._action_desc(obs, k)
            is_chosen = (k == chosen)
            style = f"bold {_CURSOR_COLOR}" if is_chosen else ""
            if p is not None:
                bar = "▮" * max(1 if p > 0.005 else 0, round(p * 10))
                out.append(f"{p * 100:5.1f}% ", style=style or "dim")
                out.append(f"{bar:<10}", style=style or _POS_COLOR)
            out.append(f"[{k}] {desc}", style=style)
            if is_chosen:
                out.append("  ◀ chosen", style=style)
            out.append("\n")

        opp_lines = self._opp_actions_before(g, step)
        if opp_lines:
            out.append(f"\nOpponent since decision {step - 1}:\n", style="italic")
            for ln in opp_lines:
                out.append(f"  opp → {ln}\n", style="dim")
        if g.get("shard"):
            out.append("\n⛁ shard replay: chosen = argmax(visits) — sampled, "
                       "not exact, inside each game's temperature window; "
                       "unsearched (fallback) decisions are not recorded\n",
                       style="dim italic")
        return out

    @staticmethod
    def _clock_line(g, step) -> str:
        """Match-clock strip for the decision panel: each clocked seat's bank
        entering this model decision, as 'remaining / bank' seconds. Empty when
        neither seat played under a match clock (no clock= knob in its spec).
        The model's reading is recorded at each of its decisions; the
        opponent's is the last reading its actions left at or before this step
        (its full bank before it has acted)."""
        def fmt(remaining, bank):
            rem = f"{remaining:.1f}s" if remaining is not None else "?"
            return f"{rem} / {bank:g}s"
        parts = []
        if g.get("clock_bank") is not None:
            rems = g.get("clock_remaining") or []
            rem = rems[step] if step < len(rems) else None
            parts.append("model " + fmt(rem, g["clock_bank"]))
        if g.get("opp_clock_bank") is not None:
            opp_rem = g["opp_clock_bank"]
            for oa in g.get("opp_actions", []):
                if (oa["before_model_step"] <= step
                        and oa.get("clock") is not None):
                    opp_rem = oa["clock"]
            parts.append("opp " + fmt(opp_rem, g["opp_clock_bank"]))
        return "⏱ clock: " + " · ".join(parts) if parts else ""

    def _refresh_hist_subtitle(self) -> None:
        hist = self.query_one("#vhist", ValueHistogram)
        if self._cur_game is None:
            hist.border_subtitle = ""
            return
        g = self._games[self._cur_game]
        v = g["values"][self._cur_step] if self._cur_step < len(g["values"]) else None
        vs = f" · V={v:+.3f}" if v is not None else ""
        hist.border_subtitle = (f"game {self._cur_game} [{_result_str(g)}] · "
                                f"step {self._cur_step}/{len(g['values']) - 1}{vs}")

    @staticmethod
    def _opp_actions_before(g, step):
        """Opponent actions between model decisions step-1 and step, with runs
        of identical consecutive actions collapsed ('PASS (x19)')."""
        descs = [oa["desc"] for oa in g.get("opp_actions", [])
                 if oa["before_model_step"] == step]
        lines = []
        run_desc, run_len = None, 0

        def flush():
            if run_len == 1:
                lines.append(run_desc)
            elif run_len > 1:
                lines.append(f"{run_desc} (x{run_len})")
        for desc in descs:
            if desc == run_desc:
                run_len += 1
            else:
                flush()
                run_desc, run_len = desc, 1
        flush()
        return lines

    # ----- analyses menu -----

    def _run_menu_entry(self, key: str) -> None:
        if key in ("whatif", "run5", "run20"):
            if self._env is None or self._model is None:
                msg = ("Not available in shard replay — branching/simulating "
                       "needs a live env."
                       if getattr(self._args, "shards", None)
                       else "Live env not ready.")
                self.notify(msg, severity="warning")
                return
            if self._engine_busy:
                self.notify("Engine is busy (simulating or branching) — try again "
                            "when it finishes.", severity="warning")
                return
            if key == "whatif":
                if self._cur_game is None:
                    self.notify("Select a game and step first.", severity="warning")
                    return
                self._engine_busy = True
                self._refresh_summary()
                self.notify(f"Branching whatif at game {self._cur_game}, "
                            f"step {self._cur_step}…")
                self._whatif_worker(self._cur_game, self._cur_step, 3)
            else:
                n = 5 if key == "run5" else 20
                self._engine_busy = True
                self._refresh_summary(loading=True)
                self._collect_worker(n)
            return

        if not self._games:
            self.notify("No games simulated yet.", severity="warning")
            return
        if self._analysis_busy:
            self.notify("An analysis is already running.", severity="warning")
            return
        entry = next((e for e in _ANALYSES if e[0] == key), None)
        if entry is None:
            return
        _key, label, fn = entry
        self._analysis_busy = True
        self.notify(f"Running {key} on {len(self._games)} games…")
        self._analysis_worker(key, fn)

    # ----- misc -----

    def _refresh_summary(self, loading=False) -> None:
        real = [g for g in self._games if not g.get("whatif")]
        w = sum(1 for g in real if g["result"] > 0)
        l = sum(1 for g in real if g["result"] < 0)
        d = len(real) - w - l
        line = f"{len(real)} games · {w}W/{l}L/{d}D"
        n_wf = len(self._games) - len(real)
        if n_wf:
            line += f" · +{n_wf} whatif"
        if loading:
            line += "  (simulating…)"
        self.query_one("#summary", Static).update(line)

    def _log_output(self, title: str, text: str) -> None:
        log = self.query_one("#output", RichLog)
        log.write(Text(f"── {title} " + "─" * max(0, 60 - len(title)), style="bold cyan"))
        log.write(Text(text.rstrip("\n") or "(no output)"))
        log.write(Text(""))

    def on_unmount(self) -> None:
        if self._env is not None:
            self._env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Full-screen TUI for analyzing a trained RoboMage model "
                    "(board-state pager + clickable V(s) histogram + analysis views)")
    # Flags come from cli_spec.ANALYSIS_TUI_TOOL (single source shared with tui.py).
    apply_to_parser(parser, ANALYSIS_TUI_TOOL.subs[0])
    args = parser.parse_args()
    AnalysisApp(args).run()


if __name__ == "__main__":
    main()
