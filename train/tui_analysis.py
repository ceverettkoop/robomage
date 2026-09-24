"""Model-analysis browser as a full-screen Textual TUI.

The terminal board of `analysis.py browse` (the PySide6 board is
gui_browser.py; both sit on browse_session.py's store, engine core and
engine-worker thread, so they offer the same capabilities). The app simulates
games against the chosen opponent (streaming each decision as it is played),
or loads a shard directory / saved .rmtrace session, and then lets you:

  * pick a game from the sidebar and PAGE THROUGH its board states — the
    board rendered in tui_game.py's style (bordered card widgets with
    color-identity edges, battlefield/land rows, stack, the model's hand,
    life/mana info lines, graveyards/exile), one decision step at a time,
    with a decision panel showing the model's full policy distribution at
    that step (every legal action with its probability, chosen action
    marked; a recorded search's N/Q/P columns) and the opponent's
    interleaved actions. A game still being simulated shows as a LIVE row
    that grows per decision; follow mode (f) keeps the board on its newest
    decision, x stops simulating after the current game;
  * seek by CLICKING the V(s) histogram docked at the bottom — one bar per
    decision step (bucketed when the game is wider than the terminal),
    positive V above the zero line, negative below, cursor column highlighted;
  * run every REPL analysis view (summary, cardvalue, targeting, swings,
    boundaries, matchcal, regret, entropy, consistency, calibration, turning,
    clusters, sideboard, sbvalue, shap) and the net probes (shard_probes:
    search π vs net, block importance, card swap, sweeps, pooled KL and
    calibration) from the sidebar menu — output lands in the "Analysis
    output" tab;
  * branch a counterfactual `whatif` at the current game/step (w key), and
    simulate more games, both on the live env. Each whatif ALTERNATIVE is
    grafted onto the source game's prefix and added to the games list as a
    full trace (marked ↳g<src>@<step>), so the counterfactual line can be
    selected, stepped through, and even re-branched — while staying excluded
    from the analysis/summary statistics pools (it is not an independent
    sample);
  * replay a recorded game to the current step and search it (F6), and, on
    a recording, rebuild the recorded search tree of a searched decision
    (F7) and walk it in the Tree tab — per-world trees with N/Q/P per node,
    the principal variation, and each walked node's hypothetical board as
    text;
  * save the finished games as a .rmtrace session (ctrl+s), which
    `analysis.py browse --source FILE` (either board) reopens.

The matplotlib `chart *` commands and the HTML `report` battery stay in
analysis.py — this front end covers the text/interactive tools.

Launched as `analysis.py browse` (the default --board tui; flags:
cli_spec.ANALYSIS_BROWSE_SUB), from the repo root:
    train/.venv/bin/python train/analysis.py browse --player-a <model.zip|gen> \
        --player-b scripted --deck-a delver [--deck-b mav] [--games 20] \
        [--format bo1]

--player-a is the inspected model and --player-b its opponent.

--source picks what is browsed instead of simulating. A shard directory
replays recorded AZ self-play or a GUI recording (see shard_replay.py; the
--player-a spec becomes the V(s) net, --no-net keeps the recorded outcome z,
and whatif/run stay disabled without a live env):
    train/.venv/bin/python train/analysis.py browse --player-a gen \
        --source train/az_data/gen [--seat A|B] [--no-net] [--games 20]
A .rmtrace file opens a saved analysis session (no env; --player-a is the
replay-search and probe net):
    train/.venv/bin/python train/analysis.py browse --source session.rmtrace
"""

import os
import sys
import threading
import time
import traceback

import numpy as np
from rich.text import Text

from textual import events, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import (Horizontal, HorizontalScroll, Vertical,
                                VerticalScroll)
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (Checkbox, Footer, Header, Input, OptionList,
                             RichLog, Select, Static, TabbedContent, TabPane,
                             Tree)
from textual.widgets.option_list import Option

# The front-end-independent pieces live in browse_session (shared with the Qt
# browser, gui_browser.py): the games store, the engine core + its worker
# thread, the analyses registry, the ONE process-global capture lock, and the
# label/summary/tree helpers.
import browse_session as bs
import decode
import shard_probes
from cli_spec import BROWSE_KIND_SHARDS, BROWSE_KIND_SIMULATE, browse_source_kind, is_bo3
from env import STATE_SIZE
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

# Gate messages (the same refusals the GUI pane shows).
_MSG_BUSY = ("Engine is busy (simulating, branching, searching or walking a "
             "tree) — try again when it finishes.")
_MSG_NO_SEL = "Select a game and step first."
_MSG_LIVE = "The live game has no finished record yet."

# Seconds between coalesced UI refreshes while events stream in.
_REFRESH_S = 0.1


def _diag_cells(row) -> str:
    """The recorded-search columns of one decision row: visits N, Q and the
    net prior P (blank where the diag carries none, e.g. a followed row)."""
    n = str(row.visits) if row.visits is not None else ""
    q = f"{row.q:+.3f}" if row.q is not None else ""
    p = f"{row.prior * 100:5.1f}%" if row.prior is not None else ""
    return f"{n:>5} {q:>6} {p:>6}"


def _default_save_path():
    """A fresh .rmtrace name in the working directory."""
    import gui_session_io
    return os.path.join(os.getcwd(), time.strftime("analysis_%Y%m%d_%H%M%S")
                        + gui_session_io.TRACE_EXT)


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

class EngineEvent(Message):
    """A browse_session event from the engine worker thread (post_message is
    thread-safe, so the worker's emit posts these directly)."""

    def __init__(self, event):
        self.event = event
        super().__init__()


class AnalysisResult(Message):
    """An analysis / probe run finished on the analysis worker."""

    def __init__(self, title, text):
        self.title = title
        self.text = text
        super().__init__()


# ── Save prompt ───────────────────────────────────────────────────────────────

class SavePrompt(ModalScreen):
    """Ask for the .rmtrace path to save the session to; dismisses with the
    path, or None on escape / an empty entry."""

    CSS = """
    SavePrompt { align: center middle; }
    #save-box  { width: 90; height: auto; border: round $accent;
                 padding: 0 1; background: $surface; }
    """
    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, default_path):
        super().__init__()
        self._default = default_path

    def compose(self) -> ComposeResult:
        with Vertical(id="save-box"):
            yield Static("Save analysis session (.rmtrace) — enter to save, "
                         "escape to cancel")
            yield Input(value=self._default, id="save-path")

    def on_mount(self) -> None:
        self.query_one("#save-path", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip() or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


# ── Tree tab ──────────────────────────────────────────────────────────────────

_TREE_IDLE = ("no tree open — F7 rebuilds the recorded search tree of a "
              "searched decision (recordings only)")
_TREE_COLS = f"{'N':>7} {'Q':>7} {'P':>6}  action"


def _tree_label(cells) -> Text:
    """One tree row: N / Q / P columns, then the action."""
    action, n, q, p = cells
    return Text(f"{n:>7} {q:>7} {p:>6}  {action}")


class TreePane(Vertical):
    """The Tree tab: the rebuilt search tree of one recorded decision (the
    Textual twin of gui_browser.TreePanel).

    Header (TreeSession summary + verified badge), a world picker ("merged
    root" = the summed root statistics, not expandable; one entry per
    determinized world whose private tree is browsable), a lazily expanded
    Tree (N / Q / P per node; a node's children are fetched from the engine
    worker the first time it is expanded or selected, then cached per
    (world, path)), the principal variation of the selected root action, and
    the text board of the hypothetical position the selected node's walk
    reached (the opponent's hand hidden unless revealed). Engine access goes
    through ExpandRequested — the app submits the job and feeds show_nodes
    back."""

    class ExpandRequested(Message):
        def __init__(self, world, path):
            self.world = world
            self.path = path
            super().__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ready = None            # the TreeReady on display
        self._node_cache = {}              # (world, path) -> TreeNodes
        self._items = {}              # (world, path) -> TreeNode widget node
        self._requested = set()       # (world, path) already submitted
        self._world = None            # world whose root is shown
        self._opp_is_a = True         # viewpoint: the browsed seat's opponent
        self._shown = None            # TreeNodes on the board
        self._follow_target = None    # (world, path) to select on arrival

    def compose(self) -> ComposeResult:
        yield Static(_TREE_IDLE, id="tree-head")
        with Horizontal(id="tree-bar"):
            yield Select([], prompt="world", id="tree-world")
            yield Checkbox("reveal hidden hand", id="tree-reveal")
        yield Static(_TREE_COLS, id="tree-cols")
        tree = Tree("root", id="tree-view")
        tree.show_root = False
        yield tree
        yield Static("", id="tree-pv")
        yield Static("", id="tree-board")

    # ----- viewpoint / board -----

    def set_viewpoint(self, opp_is_a):
        self._opp_is_a = bool(opp_is_a)

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        if self._shown is not None:
            self._render_board(self._shown)

    def _render_board(self, ev):
        self._shown = ev
        board = self.query_one("#tree-board", Static)
        if ev.terminal is not None:
            board.update(bs.walk_terminal_text(ev.terminal, self._opp_is_a))
        elif ev.walk_nodes and ev.walk_nodes[-1].obs is not None:
            reveal = self.query_one("#tree-reveal", Checkbox).value
            board.update("\n".join(bs.walk_board_lines(
                ev.walk_nodes[-1].obs, self._opp_is_a, reveal)))
        else:
            board.update("(tree root — the recorded decision; see the Board "
                         "tab)")

    # ----- tree content -----

    def clear(self, text=_TREE_IDLE):
        self._ready = None
        self._node_cache = {}
        self._items = {}
        self._requested = set()
        self._world = None
        self._shown = None
        self._follow_target = None
        self.query_one("#tree-head", Static).update(text)
        self.query_one("#tree-world", Select).set_options([])
        self.query_one("#tree-view", Tree).clear()
        self.query_one("#tree-pv", Static).update("")
        self.query_one("#tree-board", Static).update("")

    def show_tree(self, ev):
        """Install a TreeReady: root rows per world, PVs, and (a followed
        row) the path to pre-expand and select."""
        self.clear()
        self._ready = ev
        self.query_one("#tree-head", Static).update(bs.tree_header(ev))
        start = (ev.follow_worlds[0] if ev.follow_path and ev.follow_worlds
                 else 0)
        sel = self.query_one("#tree-world", Select)
        sel.set_options([("merged root", bs.MERGED_WORLD)]
                        + [(f"world {w}", w) for w in range(ev.worlds)])
        sel.value = start
        self._rebuild_root(start)
        if ev.follow_path:
            path = tuple(int(a) for a in ev.follow_path)
            self._follow_target = (start, path)
            for i in range(len(path) + 1):
                self._request(start, path[:i])

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        if self._ready is None or event.value is Select.BLANK:
            return
        if int(event.value) != self._world:
            self._rebuild_root(int(event.value))

    def _rebuild_root(self, world):
        tree = self.query_one("#tree-view", Tree)
        tree.clear()
        self._items = {}
        self._world = world
        self.query_one("#tree-pv", Static).update("")
        ev = self._ready
        if ev is None:
            return
        rows = ev.merged_rows if world == bs.MERGED_WORLD else ev.root_rows[world]
        for a, n, cells in bs.tree_node_rows(rows, ev.root_labels):
            self._add_item(tree.root, (world, (a,)), cells,
                           expandable=(world != bs.MERGED_WORLD and n > 0))

    def _add_item(self, parent, key, cells, expandable):
        node = parent.add(_tree_label(cells), data=key,
                          allow_expand=expandable)
        self._items[key] = node
        return node

    def _request(self, world, path):
        key = (int(world), tuple(int(a) for a in path))
        cached = self._node_cache.get(key)
        if cached is not None:
            self.show_nodes(cached)       # a world re-pick rebuilt the items
            return
        if key in self._requested:
            return
        self._requested.add(key)
        self.post_message(self.ExpandRequested(key[0], list(key[1])))

    def request_expand(self, world, path):
        """Public entry: fetch a node like selecting it would."""
        self._request(world, path)

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        event.stop()
        key = event.node.data
        if key is not None and not event.node.children:
            self._request(*key)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        event.stop()
        key = event.node.data
        if key is None or self._ready is None:
            return
        world, path = key
        pv = self.query_one("#tree-pv", Static)
        if world == bs.MERGED_WORLD:
            pv.update("(merged root — pick a world to walk its tree)")
            return
        if len(path) == 1:
            pv.update("PV: " + (self._ready.pv_lines[world].get(path[0])
                                or "—"))
        cached = self._node_cache.get(key)
        if cached is not None:
            self._render_board(cached)
        else:
            self._request(world, path)

    def show_nodes(self, ev):
        """Install a TreeNodes: populate the node's children and, when it is
        the selected/followed node, render its walked position."""
        if self._ready is None:
            return
        key = (int(ev.world), tuple(int(a) for a in ev.path))
        self._node_cache[key] = ev
        node = self._items.get(key)
        if node is not None and not node.children:
            if ev.rows:
                for a, n, cells in bs.tree_node_rows(ev.rows, ev.labels):
                    self._add_item(node, (key[0], key[1] + (a,)), cells,
                                   expandable=(n > 0))
            elif ev.terminal is None:
                node.add_leaf(Text("(unexpanded)", style="dim"))
        tree = self.query_one("#tree-view", Tree)
        cur = tree.cursor_node
        is_current = cur is not None and cur.data == key
        if key == self._follow_target:
            self._follow_target = None
            if node is not None:
                parent = node.parent
                while parent is not None:
                    parent.expand()
                    parent = parent.parent
                tree.move_cursor(node)
            is_current = True
        if is_current or (key[1] == () and self._shown is None):
            self._render_board(ev)


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
    /* Tree tab. */
    #tree-head  { height: auto; padding: 0 1; color: $accent; text-style: bold; }
    #tree-bar   { height: auto; }
    #tree-world { width: 28; }
    #tree-cols  { height: 1; padding: 0 1; color: $text-muted; }
    #tree-view  { height: 1fr; border: round $primary; }
    #tree-pv    { height: auto; padding: 0 1; color: $text-muted; }
    #tree-board { height: auto; max-height: 12; padding: 0 1;
                  border: round $surface; }
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
        Binding("f6", "search", "Search @ step"),
        Binding("f7", "tree", "Tree @ step"),
        Binding("f", "toggle_follow", "Follow live"),
        Binding("x", "stop_sim", "Stop sim"),
        Binding("ctrl+s", "save", "Save .rmtrace"),
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
        self._kind = browse_source_kind(getattr(args, "source", None))
        self._shards = self._kind == BROWSE_KIND_SHARDS
        self._store = bs.BrowseStore()
        # The one engine thread: load/collect/whatif/search/tree jobs, each
        # streaming browse_session events back as EngineEvent messages.
        self._worker = bs.EngineWorker(bs.EngineCore(args, self._emit),
                                       name="tui-browser-engine")
        self._busy_kind = None        # None | "sim" | "whatif" | "search" | "tree"
        self._collect_stop = None     # stop event of the running collect job
        self._tree_open = False       # a TreeSession is open on the worker
        self._tree_jobs = 0           # tree/tree_expand jobs not yet EngineIdle'd
        self._follow = True           # ride the live game's newest decision
        self._probe_net = None        # lazy shard_probes net (+ label)
        self._probe_net_label = ""
        self._loaded_provenance = None  # an opened .rmtrace's own provenance
        self._save_path = None
        self._shutting_down = False
        # Coalesced refresh state (flushed by one pending timer).
        self._dirty_rows = set()
        self._dirty_rebuild = False
        self._dirty_summary = False
        self._dirty_selected = False
        self._flush_pending = False
        # Rows consumed by the chrome (header + footer + tab bar) — everything
        # that is neither the histogram nor the tab's own content. Measured once
        # from the live layout so _relayout never hardcodes Textual's tab-bar
        # height; see _chrome_overhead / _relayout.
        self._overhead = None

    def _emit(self, ev):
        """EngineCore's emit (worker thread): hand the event to the UI."""
        if not self._shutting_down:
            self.post_message(EngineEvent(ev))

    # ----- layout -----

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("Loading…", id="summary")
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
                with TabPane("Tree", id="tab-tree"):
                    yield TreePane(id="tree-pane")
        yield ValueHistogram(id="vhist")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "RoboMage · analysis"
        self.sub_title = (f"{self._args.player_a}  vs  {self._args.player_b}"
                          + ("  (bo3)" if is_bo3(self._args) else ""))
        menu = self.query_one("#analyses", OptionList)
        for key, label, _fn in bs.ANALYSES:
            menu.add_option(Option(label, id=key))
        # Engine / replay / tree entries, then the net probes (every mode; the
        # probe net loads on first use).
        for key, label in (bs.ENGINE_MENU + bs.REPLAY_MENU + bs.TREE_MENU
                           + shard_probes.PROBE_MENU):
            menu.add_option(Option(label, id=key))
        hist = self.query_one("#vhist", ValueHistogram)
        hist.border_title = "V(s) over game — click a bar to jump to that decision"
        self.query_one("#decision", Vertical).border_title = \
            "Decision — legal actions with policy P(a)"
        self.query_one("#phase", Static).update(
            "[dim]Waiting for the first game…[/dim]")
        # Fit the board to the initial terminal size once the first layout pass
        # has assigned real widget heights (on_mount runs pre-layout).
        self.call_after_refresh(self._relayout)
        self._worker.start()
        self._submit_collect("load", self._args.games)

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

    # ----- engine job submission -----

    def _submit_collect(self, kind, n):
        stop = threading.Event()
        self._collect_stop = stop
        self._store.engine_busy = True
        self._busy_kind = "sim"
        self._mark(summary=True)
        self._worker.submit((kind, n, stop))

    def _submit_job(self, busy_kind, cmd):
        self._store.engine_busy = True
        self._busy_kind = busy_kind
        self._mark(summary=True)
        self._worker.submit(cmd)

    # ----- engine events (UI thread) -----

    async def on_engine_event(self, message: EngineEvent) -> None:
        if self._shutting_down:
            return
        ev = message.event
        applied = self._store.apply(ev)
        if isinstance(ev, bs.EnvReady):
            if ev.subtitle:
                self.sub_title = ev.subtitle
            if ev.provenance is not None:
                self._loaded_provenance = dict(ev.provenance)
            if ev.startup_text.strip():
                self._log_output("startup", ev.startup_text)
            self._mark(summary=True)
        elif isinstance(ev, bs.LoadFailed):
            self.query_one("#summary", Static).update(
                Text(ev.text.splitlines()[0] if ev.text else "load failed",
                     style="red"))
            self._log_output("error", ev.text)
            self.query_one("#tabs", TabbedContent).active = "tab-output"
        elif isinstance(ev, bs.EngineNote):
            self._log_output("note", ev.text)
        elif isinstance(ev, bs.AnalysisDone):
            # Engine-side results (whatif table, search, tree refusals, errors).
            self._busy_kind = None
            self._log_output(ev.title, ev.text)
            self.query_one("#tabs", TabbedContent).active = "tab-output"
        elif isinstance(ev, bs.EngineIdle):
            if self._tree_jobs > 0:
                self._tree_jobs -= 1
                if self._tree_jobs > 0:
                    self._store.engine_busy = True   # expansions still queued
                    return
            self._busy_kind = None
            self._collect_stop = None
            self._mark(summary=True)
        elif isinstance(ev, bs.TreeReady):
            self._tree_open = True
            pane = self.query_one("#tree-pane", TreePane)
            if ev.gn < len(self._store.games):
                pane.set_viewpoint(
                    not bool(self._store.games[ev.gn].get("model_is_a")))
            pane.show_tree(ev)
            self.query_one("#tabs", TabbedContent).active = "tab-tree"
            self.notify(bs.tree_ready_status(ev))
        elif isinstance(ev, bs.TreeNodes):
            self.query_one("#tree-pane", TreePane).show_nodes(ev)
        elif isinstance(ev, bs.TreeClosed):
            self._tree_open = False
            if ev.reason != "replaced":
                self.query_one("#tree-pane", TreePane).clear()
        elif isinstance(ev, bs.GameStarted):
            self._mark(rows={applied.game_idx}, summary=True)
            if self._follow or self._store.cur_game is None:
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
        elif isinstance(ev, bs.GameAborted):
            if applied.removed:
                self._mark(rebuild=True, summary=True, selected=True)
        elif isinstance(ev, bs.GameAdded):
            self._mark(rows={applied.game_idx}, summary=True)
            if self._store.cur_game is None:
                self._store.select_game(applied.game_idx)
                self._mark(selected=True)

    def _follow_advance(self) -> None:
        """Live follow mode: when the selected game is the live game and the
        cursor sat on the previous last step, ride the new step (stepping back
        disengages naturally — the cursor is no longer at the end; End
        re-engages)."""
        st = self._store
        if not self._follow or st.cur_game != st.live_idx:
            return
        n = len(st.games[st.cur_game]["observations"])
        if n == 1 or st.cur_step == n - 2:
            st.cur_step = n - 1

    def on_analysis_result(self, message: AnalysisResult) -> None:
        self._store.analysis_busy = False
        self._log_output(message.title, message.text)
        self.query_one("#tabs", TabbedContent).active = "tab-output"

    # ----- coalesced refresh -----

    def _mark(self, rows=None, rebuild=False, summary=False, selected=False):
        if rows:
            self._dirty_rows |= {r for r in rows if r is not None}
        self._dirty_rebuild = self._dirty_rebuild or rebuild
        self._dirty_summary = self._dirty_summary or summary
        self._dirty_selected = self._dirty_selected or selected
        if not self._flush_pending:
            self._flush_pending = True
            self.set_timer(_REFRESH_S, self._flush_refresh)

    async def _flush_refresh(self) -> None:
        self._flush_pending = False
        if self._shutting_down:
            return
        rows, rebuild = self._dirty_rows, self._dirty_rebuild
        summary, selected = self._dirty_summary, self._dirty_selected
        self._dirty_rows = set()
        self._dirty_rebuild = self._dirty_summary = self._dirty_selected = False

        games_list = self.query_one("#games", OptionList)
        games = self._store.games
        if rebuild:
            games_list.clear_options()
            games_list.add_options([Option(bs.game_label(i, g), id=str(i))
                                    for i, g in enumerate(games)])
        else:
            for i in sorted(rows):
                if i >= len(games):
                    continue     # row vanished (abort) before the flush
                while games_list.option_count <= i:
                    j = games_list.option_count
                    games_list.add_option(Option(bs.game_label(j, games[j]),
                                                 id=str(j)))
                games_list.replace_option_prompt_at_index(
                    i, bs.game_label(i, games[i]))
        if summary:
            self._refresh_summary()
        if selected:
            await self._show_selected()

    def _refresh_summary(self) -> None:
        self.query_one("#summary", Static).update(bs.busy_summary_line(
            self._store.games, self._store.engine_busy, self._busy_kind))

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
            g = self._store.selected()
            ss = bs.search_caption(g, message.step) if g is not None else ""
            hist.border_subtitle = (f"step {message.step} · "
                                    f"V={message.value:+.3f}{ss}")

    async def action_step(self, delta: int) -> None:
        if self._store.cur_game is not None:
            await self._set_step(self._store.cur_step + delta)

    async def action_step_home(self) -> None:
        await self._set_step(0)

    async def action_step_end(self) -> None:
        g = self._store.selected()
        if g is not None:
            await self._set_step(len(g["observations"]) - 1)

    def action_whatif(self) -> None:
        self._run_menu_entry("whatif")

    def action_search(self) -> None:
        self._run_menu_entry("search")

    def action_tree(self) -> None:
        self._run_menu_entry("tree")

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        self.notify(f"Follow live game: {'on' if self._follow else 'off'}")

    def action_stop_sim(self) -> None:
        if self._collect_stop is None or self._busy_kind != "sim":
            self.notify("Not simulating.", severity="warning")
            return
        self._collect_stop.set()
        self.notify("Stopping after the current game…")

    async def _select_game(self, gn: int) -> None:
        if gn == self._store.cur_game or not self._store.select_game(gn):
            return
        self._close_tree()
        await self._show_selected()
        self.query_one("#tabs", TabbedContent).active = "tab-board"

    async def _set_step(self, step: int) -> None:
        st = self._store
        if st.cur_game is None:
            return
        step = st.clamp_step(step)
        if step == st.cur_step:
            return
        st.cur_step = step
        self._close_tree()
        self.query_one("#vhist", ValueHistogram).set_cursor(step)
        await self._show_step()
        self.query_one("#tabs", TabbedContent).active = "tab-board"

    async def _show_selected(self) -> None:
        """Render the selected game at its (clamped) cursor, or clear the
        panes when nothing is selected / the live game has no decision yet."""
        g = self._store.selected()
        hist = self.query_one("#vhist", ValueHistogram)
        if g is None or not g["observations"]:
            hist.set_data(g["values"] if g is not None else [])
            self.query_one("#phase", Static).update(
                "[dim](no game selected)[/dim]" if g is None
                else "[dim](no decisions recorded yet)[/dim]")
            self.query_one("#decision-body", Static).update("")
            self._refresh_hist_subtitle()
            return
        self._store.cur_step = self._store.clamp_step(self._store.cur_step)
        hist.set_data(g["values"], cursor=self._store.cur_step)
        await self._show_step()

    def _close_tree(self) -> None:
        """Release the open tree (the selected decision changed)."""
        if self._tree_open:
            self._tree_open = False
            self._worker.submit(("tree_close",))
        self.query_one("#tree-pane", TreePane).clear()

    # ----- board pane (tui_game-style card objects) -----

    async def _show_step(self) -> None:
        """Render the current game/step: the board as card widgets plus the
        decision panel. The recorded obs is always from the MODEL's perspective
        (traces cover model decisions only), so 'self' is the model — no
        mirroring is ever needed."""
        gn, step = self._store.cur_game, self._store.cur_step
        g = self._store.games[gn]
        obs = np.asarray(g["observations"][step], dtype=np.float32)
        gs = decode.decode_game_state(obs[:STATE_SIZE],
                                      labels=decode.SELF_OPP_LABELS)

        self.query_one("#phase", Static).update(self._phase_strip(g, gn, step, obs, gs))
        self.query_one("#opp-info", Static).update(
            bs.info_line("OPPONENT", gs["opponent"], gs["opp_library"]))
        self.query_one("#self-info", Static).update(
            bs.info_line("MODEL   ", gs["self"], gs["self_library"]))
        self.query_one("#graveyards", Static).update(bs.zones_text(gs))

        await self._rebuild_stack(gs["stack"])
        await self._rebuild_bf("#opp-bf-perms", "#opp-bf-lands",
                               gs["opp_battlefield"], "opp")
        await self._rebuild_bf("#self-bf-perms", "#self-bf-lands",
                               gs["self_battlefield"], "self")
        await self._rebuild_hand(gs["self_hand"])
        dd = bs.decision_data(g, step)
        self.query_one("#decision-body", Static).update(
            self._decision_text(g, step, dd))
        self.query_one("#decision", Vertical).border_subtitle = \
            f"{dd.num_choices} legal · scroll for more"
        self.query_one("#decision-scroll", VerticalScroll).scroll_home(animate=False)
        self._refresh_hist_subtitle()

    @staticmethod
    def _phase_strip(g, gn, step, obs, gs) -> str:
        """One-line header: game/decision/V context plus the step strip."""
        pd = bs.phase_data(g, gn, step, obs, gs)
        cells = " ".join(f"[reverse b]{a}[/reverse b]" if i == pd.cur_step_idx
                         else f"[dim]{a}[/dim]"
                         for i, a in enumerate(_STEP_ABBR))
        match = pd.match.replace("SIDEBOARD", "[b yellow]SIDEBOARD[/b yellow]")
        return f"[b]{pd.header}[/b]{match}   {pd.context}   " + cells

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

    @staticmethod
    def _decision_text(g, step, dd) -> Text:
        """The model's decision at this step: every legal action with its
        recorded policy probability (sorted most-likely first, chosen action
        marked), then the opponent's actions since the previous decision.
        Clocked seats (a clock= spec knob) get a match-clock line on top."""
        out = Text()
        if dd.clock:
            out.append(dd.clock + "\n", style="dim")
        if dd.search_line:
            out.append(dd.search_line + "\n", style="dim")
        has_diag = any(r.visits is not None for r in dd.rows)
        if has_diag:
            out.append(f"{'π':>6} {'N':>5} {'Q':>6} {'P':>6}\n", style="dim")
        for r in dd.rows:
            style = f"bold {_CURSOR_COLOR}" if r.is_chosen else ""
            if r.prob is not None:
                bar = "▮" * max(1 if r.prob > 0.005 else 0, round(r.prob * 10))
                out.append(f"{r.prob * 100:5.1f}% ", style=style or "dim")
                if has_diag:
                    out.append(_diag_cells(r) + " ", style=style or "dim")
                out.append(f"{bar:<10}", style=style or _POS_COLOR)
            out.append(f"[{r.k}] {r.desc}", style=style)
            if r.is_chosen:
                out.append("  ◀ chosen", style=style)
            out.append("\n")

        if dd.opp_lines:
            out.append(f"\nOpponent since decision {step - 1}:\n", style="italic")
            for ln in dd.opp_lines:
                out.append(f"  opp → {ln}\n", style="dim")
        if dd.shard_caveat:
            out.append("\n⛁ shard replay: chosen = argmax(recorded π) — "
                       "sampled decisions (a game's temperature window) are "
                       "reconstructed, not exact. Training shards omit "
                       "unsearched decisions; play recordings keep them as "
                       "one-hot rows (a flat 100% = no search ran there)\n",
                       style="dim italic")
        return out

    def _refresh_hist_subtitle(self) -> None:
        hist = self.query_one("#vhist", ValueHistogram)
        g = self._store.selected()
        if g is None:
            hist.border_subtitle = ""
            return
        step = self._store.cur_step
        v = g["values"][step] if step < len(g["values"]) else None
        vs = f" · V={v:+.3f}" if v is not None else ""
        hist.border_subtitle = (f"game {self._store.cur_game} [{bs.result_str(g)}] · "
                                f"step {step}/{max(len(g['values']) - 1, 0)}{vs}"
                                f"{bs.search_caption(g, step)}")

    # ----- analyses / engine menu -----

    def _run_menu_entry(self, key: str) -> None:
        if key in ("whatif", "run5", "run20"):
            self._run_engine_entry(key)
        elif key == "search":
            self._run_search_entry()
        elif key == "tree":
            self._run_tree_entry()
        elif key in shard_probes.PROBE_KEYS:
            self._run_probe_entry(key)
        else:
            self._run_analysis_entry(key)

    def _selected_finished_game(self):
        """(gn, step, game) for the replay/tree jobs, or None after notifying
        why not (no selection / engine busy / the live game)."""
        st = self._store
        if st.cur_game is None:
            self.notify(_MSG_NO_SEL, severity="warning")
            return None
        if st.engine_busy:
            self.notify(_MSG_BUSY, severity="warning")
            return None
        game = st.games[st.cur_game]
        if game.get("live"):
            self.notify(_MSG_LIVE, severity="warning")
            return None
        return st.cur_game, st.cur_step, game

    def _run_search_entry(self) -> None:
        """Replay-to-step MCTS at the current game/step: needs only the game's
        recorded seed/action log (not the live env), so it works in shard and
        saved-session browsing too; an unreplayable game gets the job's
        printed refusal in the output tab."""
        sel = self._selected_finished_game()
        if sel is None:
            return
        gn, step, game = sel
        self.notify(f"Replaying game {gn} to step {step} and searching…")
        self._submit_job("search", ("search", gn, step, game))

    def _run_tree_entry(self) -> None:
        """Exact rebuild of the recorded search tree at the current game/step
        (F7): a recording's searched (kind 1) or tree-followed (kind 2) row
        only. The job replaces any tree already open."""
        if not self._shards:
            self.notify(bs.MSG_TREE_NOT_SHARDS, severity="warning")
            return
        sel = self._selected_finished_game()
        if sel is None:
            return
        gn, step, game = sel
        if not bs.step_has_tree(game, step):
            self.notify(bs.MSG_NO_DIAG, severity="warning")
            return
        self._tree_jobs += 1
        self.notify("Rebuilding search tree…")
        self._submit_job("tree", ("tree", gn, step, game))

    def on_tree_pane_expand_requested(self, message: TreePane.ExpandRequested) -> None:
        """Fetch one node of the open tree. Expansions queue behind each other
        (a followed row pre-expands its whole path), so engine_busy stays set
        until the last one's EngineIdle."""
        if self._shutting_down or not self._tree_open:
            return
        self._tree_jobs += 1
        self._submit_job("tree", ("tree_expand", int(message.world),
                                  list(message.path)))

    def _run_engine_entry(self, key: str) -> None:
        """whatif / run +N: need the live env of a simulate session."""
        if not self._worker.core.has_env:
            self.notify("Not available when browsing recorded shards or a "
                        "saved session — branching/simulating needs a live env."
                        if self._kind != BROWSE_KIND_SIMULATE
                        else "Live env not ready.", severity="warning")
            return
        if self._store.engine_busy:
            self.notify(_MSG_BUSY, severity="warning")
            return
        if key == "whatif":
            st = self._store
            if st.cur_game is None:
                self.notify(_MSG_NO_SEL, severity="warning")
                return
            gn, step = st.cur_game, st.cur_step
            self.notify(f"Branching whatif at game {gn}, step {step}…")
            self._submit_job("whatif", ("whatif", gn, step, 3, st.games[gn]))
        else:
            n = 5 if key == "run5" else 20
            self.notify(f"Simulating {n} more games…")
            self._submit_collect("collect", n)

    def _run_analysis_entry(self, key: str) -> None:
        pool = bs.analysis_pool(self._store.games)
        if not pool:
            self.notify("No finished games yet.", severity="warning")
            return
        if self._store.analysis_busy:
            self.notify("An analysis is already running.", severity="warning")
            return
        entry = next((e for e in bs.ANALYSES if e[0] == key), None)
        if entry is None:
            return
        self._store.analysis_busy = True
        self.notify(f"Running {key} on {len(pool)} games…")
        self._analysis_worker(key, entry[2], pool)

    def _run_probe_entry(self, key: str) -> None:
        """Net probes over the browsed records (shard_probes): snapshot on the
        UI thread, stack + torch on the analysis worker. The probe net loads
        on first use and is cached; analysis_busy keeps one worker on it."""
        if self._store.analysis_busy:
            self.notify("An analysis is already running.", severity="warning")
            return
        if key in shard_probes.DECISION_PROBES and self._store.cur_game is None:
            self.notify(_MSG_NO_SEL, severity="warning")
            return
        snap = shard_probes.snapshot(self._store.games, self._store.cur_game,
                                     self._store.cur_step)
        if not any(c["observations"] for c in snap["games"]):
            self.notify("No browsable decisions yet.", severity="warning")
            return
        self._store.analysis_busy = True
        self.notify(f"Running {key}…")
        self._probe_worker(key, snap, getattr(self._args, "player_a", None)
                           or "gen")

    @work(thread=True, group="analysis")
    def _analysis_worker(self, title: str, fn, pool) -> None:
        # Whatif branch traces and the live game are excluded from the pool
        # (bs.analysis_pool): not independent finished samples.
        try:
            text = bs.capture(fn, pool)
        except Exception:
            text = traceback.format_exc()
        self.post_message(AnalysisResult(title, text))

    @work(thread=True, group="analysis")
    def _probe_worker(self, key: str, snap, model_spec: str) -> None:
        try:
            if self._probe_net is None:
                self._probe_net, self._probe_net_label = \
                    shard_probes.load_probe_net(model_spec)
            lines = shard_probes.run_probe(key, self._probe_net, snap)
            text = "\n".join([f"probe net: {self._probe_net_label}", ""] + lines)
        except Exception:
            text = traceback.format_exc()
        self.post_message(AnalysisResult(key, text))

    # ----- save -----

    def action_save(self) -> None:
        """ctrl+s: save the finished games as a .rmtrace (prompted path)."""
        if not bs.saveable_games(self._store.games):
            self.notify("No finished games to save yet.", severity="warning")
            return
        self.push_screen(SavePrompt(self._save_path or _default_save_path()),
                         self._save_to)

    def _save_to(self, path) -> None:
        """Write the session to ``path`` (None = cancelled)."""
        if not path:
            return
        try:
            path, n = bs.save_session(
                path, self._store.games,
                bs.session_provenance(self._args, self._loaded_provenance))
        except Exception as exc:  # noqa: BLE001 — report, never crash the UI
            self.notify(f"Save failed: {exc}", severity="error", timeout=10)
            return
        self._save_path = path
        self.notify(f"Saved {n} game(s) to {path}")

    # ----- misc -----

    def _log_output(self, title: str, text: str) -> None:
        log = self.query_one("#output", RichLog)
        log.write(Text(f"── {title} " + "─" * max(0, 60 - len(title)), style="bold cyan"))
        log.write(Text(text.rstrip("\n") or "(no output)"))
        log.write(Text(""))

    def on_unmount(self) -> None:
        """Stop the running collect, release the tree, and let the worker
        close the env on its own thread (bounded join; daemon thread)."""
        self._shutting_down = True
        if self._collect_stop is not None:
            self._collect_stop.set()
        if self._tree_open:
            self._worker.submit(("tree_close",))
        self._worker.submit(("shutdown",))
        if self._worker.is_alive():
            self._worker.join(5.0)


if __name__ == "__main__":
    sys.exit("tui_analysis.py was removed as an entry point; use "
             "`analysis.py browse` (e.g. --source train/az_data/gen)")
