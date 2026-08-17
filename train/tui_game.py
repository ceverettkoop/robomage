"""Interactive RoboMage game board as a full-screen Textual TUI.

Reimplements the human-vs-opponent play loop (see play.py's CLI mode) as
a terminal UI: rendered battlefield, hand, stack, graveyards, life/mana, phase
strip, and a numbered action list. The human acts either by clicking a card/zone
or by choosing a numbered option.

Driving the engine is delegated entirely to RoboMageEnv (via NarrativeEnv): the
subprocess, the BQUERY binary protocol, and the attacker/blocker confirm-slot
remapping all live there. This module is a front-end over that loop.

The opponent is either a trained model (MaskablePPO checkpoint) or the rule-based
scripted agent when `model_path` is None or "scripted" (any
opponents.make_controller spec works — checkpoint shorthand or scripted tier).

Invoked via `play.py --tui` (and the tui.py launcher's Play entry).
"""

import time

import numpy as np
from rich.text import Text

from textual import events, work
from textual.app import App, ComposeResult
from textual.content import Content
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Label, OptionList, RichLog, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

from env import _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE
import decode
from game_driver import (GameDriver, build_session, decode_human_frame,
                         actions_for_card, action_zone, stack_target_refs,
                         menu_label, prompt_text, hand_type_icon, _edge_colors,
                         _STEP_ABBR)

# The card-inspect ("hold Q") banner auto-hides this many seconds after the last
# 'q'. A terminal has no key-up event, so "hold" is emulated: OS key auto-repeat
# fires the inspect action repeatedly while Q is down, each firing re-arming this
# timer, so the banner stays up until the key is released (repeats stop) and the
# timer lapses. Must comfortably exceed the auto-repeat interval; tune if a fast
# release flickers or a slow keyboard's first-repeat gap lets it blink.
ORACLE_HIDE_DELAY = 0.8


# ── Clickable card widget ─────────────────────────────────────────────────────

class CardClicked(Message):
    """A battlefield permanent or hand card was clicked."""

    def __init__(self, card_idx: int, controller: str, zone: str):
        self.card_idx = card_idx
        self.controller = controller          # "self" | "opp"
        self.zone = zone                      # "battlefield" | "hand"
        super().__init__()


# Attacking creatures get a dashed border in this red — deliberately brighter and
# more saturated than the muted color-identity red (#d64b3b) so an attacker reads
# distinctly, not like a merely red-costed card.
_ATTACK_BORDER = "#ff2b2b"


class CardButton(Static):
    """A single clickable card (permanent or hand card).

    `edge_colors` is a (top, right, bottom, left) tuple of rich colors encoding
    the card's color identity; each edge is painted its color at mount time so
    multicolor cards show a split border (see `_edge_colors`). `zone`
    ("battlefield"/"hand") distinguishes a card from a same-named copy in the
    other zone so cross-highlighting doesn't spill between them. `attacking`
    replaces the color-identity border with a dashed red one so an attacking
    creature stands out during combat."""

    def __init__(self, label: str, card_idx: int, controller: str,
                 edge_colors=None, zone: str = "battlefield",
                 attacking: bool = False):
        super().__init__(label)
        self._card_idx = card_idx
        self._controller = controller
        self._edge_colors = edge_colors
        self._zone = zone
        self._attacking = attacking

    def on_mount(self) -> None:
        if self._attacking:
            self.styles.border = ("dashed", _ATTACK_BORDER)
            return
        if not self._edge_colors:
            return
        top, right, bottom, left = self._edge_colors
        if top:
            self.styles.border_top = ("round", top)
        if right:
            self.styles.border_right = ("round", right)
        if bottom:
            self.styles.border_bottom = ("round", bottom)
        if left:
            self.styles.border_left = ("round", left)

    def set_action_highlight(self, on: bool) -> None:
        """Toggle the "an action in the menu targets this card" highlight — the
        cross-highlight partner of ActionList's option hover (see GameApp)."""
        self.set_class(on, "action-linked")

    def on_click(self, event: events.Click) -> None:
        # Left button only — right button is reserved for the oracle-text hold
        # (see GameApp.on_mouse_down / on_mouse_up).
        if event.button != 1:
            return
        self.post_message(CardClicked(self._card_idx, self._controller, self._zone))


class StackItem(Static):
    """One delineated object on the stack. Carries its announced targets so the
    app can highlight them on the board when the item is hovered (see GameApp).

    `target_refs` are already in the human's frame: each is
    {"is_player", "is_self", "card_idx"} where is_self means the human (YOU)."""

    def __init__(self, label: str, target_refs):
        super().__init__(label)
        self._target_refs = target_refs


class ActionList(OptionList):
    """OptionList that reports which option the mouse is hovering over.

    The base tracks the hovered row in the private `_mouse_hovering_over`
    reactive (set by its `_on_mouse_move`, cleared by `_on_leave`); we watch it
    and surface the change as an `ActionHovered` message so the app can
    cross-highlight the battlefield permanent that option refers to."""

    class ActionHovered(Message):
        def __init__(self, index):
            self.index = index          # hovered option index, or None
            super().__init__()

    def watch__mouse_hovering_over(self, _old, new) -> None:
        self.post_message(self.ActionHovered(new))


# ── Worker → UI messages ──────────────────────────────────────────────────────

class StateUpdate(Message):
    """Thin Textual wrapper carrying a game_driver.StateUpdate (`u`) to the UI
    thread; the renderer reads the decoded fields off `u`."""

    def __init__(self, u):
        self.u = u
        super().__init__()


class LogLines(Message):
    def __init__(self, lines):
        self.lines = list(lines)
        super().__init__()


class GameOver(Message):
    def __init__(self, text):
        self.text = text
        super().__init__()


class OppThinking(Message):
    """Toggle the "opponent is thinking" indicator around a MODEL opponent's
    move. Posted from the driver thread on either side of the (possibly slow —
    a search controller with a wall-clock budget) `_opp_act` call; the UI thread
    starts/stops an elapsed-seconds ticker in response. Only used for model/search
    opponents — a scripted opponent answers in microseconds and would just flicker."""

    def __init__(self, active):
        self.active = active
        super().__init__()


class ConfirmScreen(ModalScreen):
    """A modal "are you sure?" with N labelled choices plus Cancel.

    Dismisses with the chosen value, or None for Cancel / escape — so any exit
    other than an explicit click on a choice is a cancel, which is the right
    default for the irreversible actions it guards (conceding). Each choice is
    a (label, value) pair; the caller reads the value in its push_screen
    callback."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    DEFAULT_CSS = """
    ConfirmScreen { align: center middle; }
    #confirm-box { width: auto; max-width: 64; height: auto; padding: 1 2;
                   border: round $error; background: $panel; }
    #confirm-buttons { height: auto; padding-top: 1; }
    #confirm-buttons Button { margin: 0 1 0 0; }
    """

    def __init__(self, question, choices):
        super().__init__()
        self._question = question
        self._choices = list(choices)

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-box"):
            yield Label(self._question)
            with Horizontal(id="confirm-buttons"):
                for i, (label, _value) in enumerate(self._choices):
                    yield Button(label, variant="error", id=f"choice-{i}")
                yield Button("Cancel", variant="primary", id="cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if not bid.startswith("choice-"):
            self.dismiss(None)
            return
        self.dismiss(self._choices[int(bid.split("-", 1)[1])][1])

    def action_cancel(self) -> None:
        self.dismiss(None)


class _AppSink:
    """Adapts GameDriver's sink callbacks (fired on the worker thread) to Textual
    messages posted onto the app. Textual's post_message is thread-safe from a
    worker thread — the direct analog of the old _drive posting into the app."""

    def __init__(self, app):
        self._app = app

    def on_state(self, u):
        self._app.post_message(StateUpdate(u))

    def on_log(self, lines):
        self._app.post_message(LogLines(lines))

    def on_opp_thinking(self, active):
        self._app.post_message(OppThinking(active))

    def on_game_over(self, text):
        self._app.post_message(GameOver(text))


# ── The app ───────────────────────────────────────────────────────────────────

class GameApp(App):
    CSS = """
    #phase      { height: 1; background: $panel; color: $text; }
    #prompt     { height: 1; background: $boost; text-style: bold; }
    #opp-info   { height: 1; color: red; }
    #self-info  { height: 1; color: green; }
    #graveyards { height: 2; color: $text-muted; }
    #stack      { height: 3; border: round $primary; }
    /* Each stack object is delineated by a left bar + margin (a full box border
       would not fit the 1-row inner height). */
    StackItem   { width: auto; height: 100%; margin: 0 1; padding: 0 1;
                  border-left: thick $secondary; }
    StackItem:hover { background: $boost; }
    .stack-empty { width: auto; height: 100%; color: $text-muted; }
    /* Hovering a stack item paints its targets red: permanents on the board and
       the YOU/OPPONENT info line for a player target. */
    CardButton.stack-target { background: $error 40%; border: round $error; }
    #self-info.stack-target-player, #opp-info.stack-target-player {
        background: $error; color: $text; text-style: bold; }
    #opp-bf, #self-bf { height: 8; }
    .bf-row     { height: 1fr; layout: horizontal; }
    .bf-row.lands { background: $panel; }
    #self-hand  { height: 5; border: round green; layout: horizontal; }
    CardButton  { width: auto; height: 100%; margin: 0 1; padding: 0 1;
                  border: round $surface; }
    CardButton:hover { background: $boost; }
    /* A menu action targets this card (cross-highlight from action hover). */
    CardButton.action-linked { background: $warning 40%; border: round $warning; }
    #bottom     { height: 1fr; min-height: 8; }
    #actions    { width: 35%; min-width: 24; border: round $primary; }
    #log        { width: 1fr; border: round $surface; }
    /* Card-inspect popup: a floating banner over the board (see action_inspect
       / on_mouse_down). Hidden until 'q' or right-click is held over a card.
       #oracle-layer is a full-screen, out-of-flow overlay whose only job is
       centering #oracle within the screen. */
    Screen { layers: base overlay; }
    #oracle-layer {
        layer: overlay; display: none; dock: top;
        width: 100%; height: 100%;
        align: center middle;
    }
    #oracle {
        width: auto; max-width: 70%; height: auto; max-height: 12;
        padding: 0 1;
        border: round $accent; background: $panel; color: $text;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
        ("q", "inspect", "Oracle (hold)"),
        ("space", "pass_zero", "Pass"),
        ("p", "autopass", "Autopass"),
        # Conceding is irreversible, so it takes a modifier key AND a confirm
        # modal — never a bare letter that could be hit while picking actions.
        ("ctrl+r", "concede", "Concede"),
        ("greater_than_sign", "resize_log(1)", "Bigger log"),
        ("less_than_sign", "resize_log(-1)", "Smaller log"),
        ("0", "pick('0')", "Pick"),
        ("1", "pick('1')", ""), ("2", "pick('2')", ""), ("3", "pick('3')", ""),
        ("4", "pick('4')", ""), ("5", "pick('5')", ""), ("6", "pick('6')", ""),
        ("7", "pick('7')", ""), ("8", "pick('8')", ""), ("9", "pick('9')", ""),
    ]

    def __init__(self, session):
        super().__init__()
        self._env = session.env
        self._opp_is_a = session.opp_is_a
        self._human_deck = session.human_deck
        self._opp_deck = session.opp_deck
        self._bo3 = session.bo3
        self._opp_label = session.opp_label
        # The front-end-agnostic driver owns the engine loop, the human-choice
        # queue, the autopass/quit state, the running reward + bo3 match context,
        # and the winner/game-break banners; the app just marshals its sink
        # callbacks onto the UI thread and delivers the human's picks back.
        # NAME IT `_game_driver`, never `_driver`: textual.app.App keeps its own
        # terminal Driver in `self._driver` and REASSIGNS it while starting up,
        # so an attribute of that name is silently replaced the moment the app
        # runs (on_mount then called HeadlessDriver.run and the board died).
        self._game_driver = GameDriver(
            env=session.env, opp_act=session.opp_act, opp_is_a=session.opp_is_a,
            is_model=session.is_model, opp_label=session.opp_label,
            bo3=session.bo3, sink=_AppSink(self),
            clock_fn=session.clock_fn, pace_idle=session.pace_idle,
            controller=session.controller,
            human_clock_s=session.human_clock_s,
            hard_timeout=session.hard_timeout)
        self._actions = []
        self._awaiting = False
        # Set by on_game_over — after it, conceding is meaningless (and the
        # driver loop is gone, so the queued sentinel would never be read).
        self._game_over = False
        # Elapsed-seconds ticker for the "opponent is thinking" indicator (a
        # model/search opponent only). Started/stopped on the UI thread by
        # on_opp_thinking; None while idle. See OppThinking.
        self._think_timer = None
        self._think_start = 0.0
        # The CardButton the mouse is currently over (Enter/Leave tracked in
        # on_enter/on_leave). Drives the action<->permanent cross-highlighting,
        # and is the anchor the card-inspect popup reads.
        self._hovered_button = None
        # One-shot auto-hide timer for the inspect banner (see action_inspect).
        self._oracle_timer = None
        # Live heights of the resizable board panels (must match the CSS
        # defaults above). Shrinking these frees rows that flow into the 1fr
        # #bottom region, growing the log/command area; see action_resize_log.
        self._panel_h = {"#opp-bf": 8, "#self-bf": 8, "#self-hand": 5}

    # ----- layout -----

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id="phase")
        yield Static(id="opp-info")
        # The opponent's rows are flipped (lands on top, battlefield below) so the
        # two players' battlefields sit adjacent across the stack — a mirror-
        # symmetric board with each side's lands on the outer edge.
        with Vertical(id="opp-bf"):
            yield VerticalScroll(id="opp-bf-lands", classes="bf-row lands")
            yield VerticalScroll(id="opp-bf-perms", classes="bf-row")
        yield HorizontalScroll(id="stack")
        with Vertical(id="self-bf"):
            yield VerticalScroll(id="self-bf-perms", classes="bf-row")
            yield VerticalScroll(id="self-bf-lands", classes="bf-row lands")
        yield VerticalScroll(id="self-hand")
        yield Static(id="self-info")
        yield Static(id="graveyards")
        yield Static(id="prompt")
        with Horizontal(id="bottom"):
            yield ActionList(id="actions")
            yield RichLog(id="log", wrap=True, highlight=False, markup=True)
        with Vertical(id="oracle-layer"):
            yield Static(id="oracle")
        yield Footer()

    def on_mount(self) -> None:
        human_seat = "B" if self._opp_is_a else "A"
        opp_seat = "A" if self._opp_is_a else "B"
        fmt = "Best of 3" if self._bo3 else "Single game"
        self.title = f"RoboMage · {fmt}"
        self.sub_title = (f"You (Player {human_seat}, {self._human_deck})  vs  "
                          f"{self._opp_label} (Player {opp_seat}, {self._opp_deck})")
        self._log("[b]Game starting…[/b]  Click a card or pick a numbered action. "
                  "Keys: digits = pick, space = pass, p = autopass, "
                  "hold q or right-click a card = show oracle text, "
                  "ctrl+r = concede, ctrl+q = quit.")
        self._drive()

    # ----- the driver (background thread) -----

    @work(thread=True, exclusive=True)
    def _drive(self) -> None:
        self._game_driver.run()

    # ----- message handlers (UI thread) -----

    async def on_state_update(self, message: StateUpdate) -> None:
        u = message.u
        obs = u.obs
        # decode_human_frame decodes the priority-player-perspective obs and, when
        # the opponent holds priority (opp_perspective), mirrors it back to the
        # human's view so the board never flips while the opponent acts/decides.
        gs, mirrored = decode_human_frame(u)

        self.query_one("#phase", Static).update(self._phase_strip(obs, gs))
        self.query_one("#opp-info", Static).update(self._info_line("OPPONENT", gs["opponent"], gs["opp_library"]))
        self.query_one("#self-info", Static).update(self._info_line("YOU", gs["self"], gs["self_library"]))
        self.query_one("#graveyards", Static).update(
            f"Your GY: {', '.join(gs['self_graveyard']) or '—'}\n"
            f"Opp GY:  {', '.join(gs['opp_graveyard']) or '—'}")

        await self._rebuild_stack(gs["stack"], mirrored)
        await self._rebuild_bf("#opp-bf-perms", "#opp-bf-lands", gs["opp_battlefield"], "opp")
        await self._rebuild_bf("#self-bf-perms", "#self-bf-lands", gs["self_battlefield"], "self")
        # A mirrored obs's "self_hand" is the OPPONENT's hand (private) — never
        # render it; the human's hand keeps its last own-perspective contents.
        if not mirrored:
            await self._rebuild_hand(gs["self_hand"])

        # A fresh menu/board: drop any stale hover so cross-highlights don't
        # linger onto the newly-rebuilt permanents/options, and close any
        # inspect banner (its card may no longer be on the board).
        self._hovered_button = None
        self._hide_oracle()
        self._clear_stack_target_highlights()
        self._actions = u.actions
        self._awaiting = u.human_turn
        opt = self.query_one("#actions", OptionList)
        opt.clear_options()
        if u.human_turn:
            for a in u.actions:
                opt.add_option(Option(self._option_prompt(a, False),
                                      id=str(a["index"])))
            if u.actions:
                opt.highlighted = 0
            self.query_one("#prompt", Static).update(prompt_text(obs, u.num_choices, gs))
        elif self._game_driver._autopass:
            self.query_one("#prompt", Static).update("Autopass — passing to next upkeep…")
        else:
            self.query_one("#prompt", Static).update(f"{self._opp_label} is thinking…")

    def on_log_lines(self, message: LogLines) -> None:
        for line in message.lines:
            if line.strip():
                self._write_event(line)

    def on_opp_thinking(self, message: OppThinking) -> None:
        """Show/hide the elapsed-seconds "opponent is thinking" indicator around
        a model/search opponent's move (runs on the UI thread, so the ticker is
        safe to start/stop here)."""
        if message.active:
            self._think_start = time.monotonic()
            self._tick_think()
            if self._think_timer is None:
                self._think_timer = self.set_interval(1.0, self._tick_think)
        elif self._think_timer is not None:
            self._think_timer.stop()
            self._think_timer = None

    def _tick_think(self) -> None:
        elapsed = int(time.monotonic() - self._think_start)
        bank = ""
        remaining = self._game_driver.clock_remaining()
        if remaining is not None:
            r = int(remaining)
            bank = f"  ·  bank {r // 60}:{r % 60:02d}"
        self.query_one("#prompt", Static).update(
            f"⏳ {self._opp_label} is thinking…  ({elapsed}s){bank}")

    def on_game_over(self, message: GameOver) -> None:
        self._awaiting = False
        self._game_over = True
        self.query_one("#actions", OptionList).clear_options()
        self.query_one("#prompt", Static).update(message.text + "  (press q to quit)")
        self._log(f"[b]{message.text}[/b]")

    # ----- input -----

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id is not None:
            self._submit(int(event.option.id))

    def on_card_clicked(self, message: CardClicked) -> None:
        if not self._awaiting:
            return
        matches = actions_for_card(self._actions, message.card_idx,
                                   message.controller, message.zone)
        if not matches:
            self._log("[yellow]No legal action for that card — use the numbered list.[/yellow]")
        elif len(matches) == 1:
            # Exactly one legal action → clicking commits it immediately.
            self._submit(matches[0]["index"])
        else:
            # Ambiguous → don't guess. Highlight every option this card feeds and
            # move the cursor to the first; the human disambiguates from the menu.
            idxs = [a["index"] for a in matches]
            self._set_action_highlights(idxs)
            self.query_one("#actions", OptionList).highlighted = idxs[0]
            self._log(f"[yellow]That card has {len(idxs)} options: "
                      f"{', '.join(str(i) for i in idxs)} — pick a number.[/yellow]")

    # ----- action <-> permanent cross-highlighting -----

    def on_enter(self, event: events.Enter) -> None:
        """Mouse moved onto a widget. A card → remember it (shared hover anchor)
        and light up the menu actions it can drive; a stack item → paint its
        announced targets red on the board."""
        node = event.node
        if isinstance(node, CardButton):
            self._hovered_button = node
            idxs = [a["index"] for a in actions_for_card(
                self._actions, node._card_idx, node._controller, node._zone)]
            self._set_action_highlights(idxs)
        elif isinstance(node, StackItem):
            self._highlight_stack_targets(node._target_refs)

    def on_leave(self, event: events.Leave) -> None:
        """Mouse left a card/stack item — clear its highlight (for a card only if
        it's still the tracked one; Enter of the next may have superseded it)."""
        node = event.node
        if isinstance(node, CardButton) and node is self._hovered_button:
            self._hovered_button = None
            self._set_action_highlights([])
        elif isinstance(node, StackItem):
            self._clear_stack_target_highlights()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Right mouse button pressed over a card -> show its oracle-text
        banner, the mouse analog of holding 'q' (see action_inspect). Unlike Q
        (no real key-up; emulated via OS auto-repeat, see ORACLE_HIDE_DELAY),
        MouseDown/MouseUp are real press/release events, so the banner shows
        and hides exactly on press/release with no timer needed."""
        if event.button != 3:
            return
        btn = self._hovered_button
        if btn is not None:
            self._show_oracle(btn._card_idx)

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if event.button != 3:
            return
        self._hide_oracle()

    def on_action_list_action_hovered(self, message: "ActionList.ActionHovered") -> None:
        """Mouse moved over (or off) a menu option — highlight the battlefield
        permanent(s) that option refers to."""
        self._highlight_perms_for_action(message.index)

    def _highlight_perms_for_action(self, index) -> None:
        """Highlight the card(s) the given menu-option index refers to; a None
        index (mouse left the list) clears all highlights. Matches on card id,
        controller, and — when the action carries a zone_ref — the exact board
        zone, so a hand card never lights up its same-named battlefield twin."""
        buttons = list(self.query(CardButton))
        for btn in buttons:
            btn.set_action_highlight(False)
        if index is None or not (0 <= index < len(self._actions)):
            return
        a = self._actions[index]
        if a["card_idx"] < 0:
            return
        want_ctrl = a["controller"]
        want_zone = action_zone(a)
        for btn in buttons:
            if btn._card_idx != a["card_idx"]:
                continue
            if want_ctrl is not None:
                btn_want = "own" if btn._controller == "self" else "opp"
                if btn_want != want_ctrl:
                    continue
            if want_zone is not None and btn._zone != want_zone:
                continue
            btn.set_action_highlight(True)

    def _option_prompt(self, a, highlight: bool):
        """Build a menu option's prompt. Uses Content (not a str) so the label is
        rendered literally — a str is parsed as Textual markup, which would
        swallow bracketed text like the [SELF]/[OPPONENT] tags. `highlight`
        applies the cross-highlight style used when a matching card is hovered."""
        text = f"{a['index']:>2}: {menu_label(a, self._opp_is_a)}"
        if highlight:
            return Content.styled(text, "bold black on yellow")
        return Content(text)

    def _set_action_highlights(self, indices) -> None:
        """Restyle the menu so the options in `indices` show the cross-highlight
        (OptionList highlights only one row natively; this marks a whole set)."""
        if not self._awaiting:
            return
        hl = set(indices)
        opt = self.query_one("#actions", OptionList)
        for a in self._actions:
            try:
                opt.replace_option_prompt(str(a["index"]),
                                          self._option_prompt(a, a["index"] in hl))
            except OptionDoesNotExist:
                pass

    # ----- stack item -> target highlighting -----

    def _highlight_stack_targets(self, refs) -> None:
        """Paint a hovered stack object's announced targets red: matching
        battlefield permanent(s), and the YOU/OPPONENT info line for a player
        target. `refs` are already in the human frame (is_self == YOU)."""
        self._clear_stack_target_highlights()
        for ref in refs:
            if ref["is_player"]:
                sel = "#self-info" if ref["is_self"] else "#opp-info"
                self.query_one(sel, Static).add_class("stack-target-player")
            elif ref["card_idx"] >= 0:
                want = "self" if ref["is_self"] else "opp"
                for btn in self.query(CardButton):
                    if btn._card_idx == ref["card_idx"] and btn._controller == want:
                        btn.add_class("stack-target")

    def _clear_stack_target_highlights(self) -> None:
        for btn in self.query(CardButton):
            btn.remove_class("stack-target")
        for sel in ("#self-info", "#opp-info"):
            self.query_one(sel, Static).remove_class("stack-target-player")

    def action_pick(self, digit: str) -> None:
        idx = int(digit)
        if self._awaiting and 0 <= idx < len(self._actions):
            self._submit(idx)

    def action_autopass(self) -> None:
        """Engage autopass: pass priority every optional window until the next
        UPKEEP step, stopping early for any mandatory decision (declare
        attackers/blockers, discard, targeting, mulligan — anything with no
        'pass priority' option). The driver auto-drives once engaged."""
        if not self._awaiting:
            return
        # The driver queues the pass and takes over; if there was nothing to pass
        # it returns False and we leave the human's menu untouched. On engage,
        # disable input directly (not via _submit) so no stray keypress is queued
        # while the driver auto-drives — input stays disabled until autopass stops.
        if not self._game_driver.engage_autopass():
            return
        self._write_event("[You] Autopass to next upkeep")
        self._awaiting = False
        self.query_one("#actions", OptionList).clear_options()
        self.query_one("#prompt", Static).update("Autopass — passing to next upkeep…")

    def action_concede(self) -> None:
        """Ctrl+R = concede (CR 104.3a), behind a confirmation modal.

        In a bo3 the modal offers both scopes — conceding the GAME is an
        ordinary game loss and the match plays on (sideboard, next game), while
        conceding the MATCH ends it now. A single game has only the one
        meaningful choice: conceding it hands the whole thing to the opponent."""
        if self._game_over:
            return
        if self._bo3:
            question = ("Concede?\n\n"
                        "· Concede game — you lose this game; the match "
                        "continues (sideboard, next game).\n"
                        "· Concede match — the match ends now, "
                        f"{self._opp_label} wins it.")
            choices = [("Concede game", "game"), ("Concede match", "match")]
        else:
            # Single game: conceding the game IS conceding the match, so only
            # the game scope is offered (CONCEDE_GAME already ends everything).
            question = ("Concede the game?\n\nThis is a single game, so "
                        f"conceding it loses the match to {self._opp_label}.")
            choices = [("Concede game", "game")]
        self.push_screen(ConfirmScreen(question, choices), self._on_concede)

    def _on_concede(self, choice) -> None:
        """ConfirmScreen callback: None = cancelled, else "game"/"match"."""
        if choice not in ("game", "match"):
            return
        match = choice == "match"
        self._write_event("[You] Concede the match" if match
                          else "[You] Concede the game")
        # The driver queues the sentinel on the same channel a click uses, so it
        # is taken at the human's next decision; disable input meanwhile.
        self._awaiting = False
        self.query_one("#actions", OptionList).clear_options()
        self.query_one("#prompt", Static).update("Conceding…")
        self._game_driver.concede(match=match)

    def action_pass_zero(self) -> None:
        """Spacebar = pass, but only when action 0 is the pass option.

        If index 0 is 'Pass priority' (category 0), submitting it is equivalent
        to entering 0; otherwise (index 0 is some other action) spacebar is a
        no-op, so it can never fire a non-pass action."""
        if not self._awaiting:
            return
        if self._actions and self._actions[0]["category"] == 0:
            self._submit(0)

    def action_inspect(self) -> None:
        """'q' held over a card → show its oracle-text banner.

        Fires once per keypress; OS auto-repeat while Q is held re-arms the
        auto-hide timer, so the banner stays up until the key is released. With
        no card under the mouse it's a no-op (the banner just times out)."""
        btn = self._hovered_button
        if btn is not None:
            self._show_oracle(btn._card_idx)
        if self._oracle_timer is not None:
            self._oracle_timer.stop()
        self._oracle_timer = self.set_timer(ORACLE_HIDE_DELAY, self._hide_oracle)

    def _show_oracle(self, card_idx: int) -> None:
        name = decode.card_index_to_name(card_idx)
        cost = decode.fmt_mana_cost(decode.card_mana_cost(card_idx))
        oracle = decode.card_oracle_text(card_idx)
        body = Text()
        body.append(name, style="bold")
        if cost:
            body.append("   ")
            body.append(cost, style="bold yellow")
        body.append("\n")
        body.append(oracle or "(no oracle text)",
                    style="" if oracle else "italic dim")
        banner = self.query_one("#oracle", Static)
        banner.update(body)
        self.query_one("#oracle-layer", Vertical).display = True

    def _hide_oracle(self) -> None:
        if self._oracle_timer is not None:
            self._oracle_timer.stop()
            self._oracle_timer = None
        self.query_one("#oracle-layer", Vertical).display = False

    # Min/max row heights for each resizable board panel.
    _PANEL_LIMITS = {"#opp-bf": (6, 16), "#self-bf": (6, 16), "#self-hand": (3, 10)}

    def action_resize_log(self, delta: int) -> None:
        """Grow (delta>0) or shrink (delta<0) the bottom log/command area.

        The board panels are fixed-height and #bottom is 1fr, so freeing rows
        from the panels flows straight into the log/actions region."""
        step = -delta                      # grow log → shrink each panel
        changed = False
        for sel, (lo, hi) in self._PANEL_LIMITS.items():
            new = max(lo, min(hi, self._panel_h[sel] + step))
            if new != self._panel_h[sel]:
                self._panel_h[sel] = new
                self.query_one(sel).styles.height = new
                changed = True
        if not changed:
            self.bell()

    def action_quit(self) -> None:
        # Flag the shutdown via the driver first: if it is blocked in a long
        # opponent search, on_unmount's env.close() kills the engine and the pipe
        # read errors out in the driver, which then returns silently on that flag.
        self._game_driver.request_quit()
        self.exit()

    def on_unmount(self) -> None:
        self._env.close()

    # ----- helpers -----

    def _submit(self, idx: int) -> None:
        if not self._awaiting:
            return
        if 0 <= idx < len(self._actions) and self._actions[idx]["category"] != 0:
            self._write_event(f"[You] {self._actions[idx]['description']}")
        self._awaiting = False
        self.query_one("#actions", OptionList).clear_options()
        self.query_one("#prompt", Static).update("…")
        self._game_driver.submit(idx)

    def _log(self, text: str) -> None:
        """Write a styled (markup) line — used for app messages, not engine text."""
        log = self.query_one("#log", RichLog)
        try:
            log.write(text)
        except Exception:
            from rich.markup import escape
            log.write(escape(text))

    def _write_event(self, line: str) -> None:
        """Write a literal line (no markup parsing) — for action/narrative logs."""
        self.query_one("#log", RichLog).write(Text(line))

    async def _rebuild_bf(self, perms_sel: str, lands_sel: str, perms,
                          controller: str) -> None:
        """Rebuild one player's battlefield, split into a non-land row (above)
        and a land row (below)."""
        await self._fill_row(perms_sel,
                             [p for p in perms if not p.get("is_land")], controller)
        await self._fill_row(lands_sel,
                             [p for p in perms if p.get("is_land")], controller)

    async def _fill_row(self, selector: str, perms, controller: str) -> None:
        box = self.query_one(selector, VerticalScroll)
        await box.remove_children()
        widgets = [self._mk_card(decode.fmt_perm(p), p["card_idx"], controller,
                                 "battlefield", p.get("attacking", False))
                   for p in perms]
        if widgets:
            await box.mount(*widgets)

    async def _rebuild_hand(self, hand) -> None:
        box = self.query_one("#self-hand", VerticalScroll)
        await box.remove_children()
        widgets = [self._mk_card(self._hand_label(c["card_idx"], c["name"]),
                                 c["card_idx"], "self", "hand")
                   for c in hand]
        if widgets:
            await box.mount(*widgets)

    @staticmethod
    def _hand_label(card_idx: int, name: str) -> str:
        icon = hand_type_icon(card_idx)
        return f"{name} {icon}" if icon else name

    @staticmethod
    def _mk_card(label: str, card_idx: int, controller: str,
                 zone: str, attacking: bool = False) -> "CardButton":
        """Build a CardButton whose border edges encode the card's color identity
        (or a dashed red attacking border when `attacking`)."""
        edges = _edge_colors(decode.card_border_colors(card_idx))
        return CardButton(label, card_idx, controller, edges, zone, attacking)

    def _match_strip(self, match) -> str:
        """Compact bo3 score prefix ("Game 2 · You 1–0 · ") for the phase line, or
        "" outside a best-of-three match.

        The human "Game N" is derived from the games played so far
        (self_wins + opp_wins + 1), NOT the engine's serialized game_number: that
        field is 0-based and, at 0.0, is indistinguishable between bo3 game 1 and
        a single game (see src/classes/gamestate.h). Gating on the app's own
        _bo3 flag is the reliable signal that a match is in progress."""
        if not self._bo3 or not match:
            return ""
        you, opp = match["self_wins"], match["opp_wins"]
        game_n = you + opp + 1
        prefix = f"Game {game_n} · You {you}–{opp}"
        if match.get("is_sideboard"):
            prefix += " · [b yellow]SIDEBOARD[/b yellow]"
        return prefix + "   "

    def _phase_strip(self, obs, gs) -> str:
        cur = int(np.argmax(obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]))
        cells = []
        for i, abbr in enumerate(_STEP_ABBR):
            cells.append(f"[reverse b]{abbr}[/reverse b]" if i == cur else f"[dim]{abbr}[/dim]")
        active = "A" if gs["active_is_a"] else "B"
        prio = gs["priority_player"]
        return (self._match_strip(gs.get("match"))
                + f"Turn {gs['turn']} · Active {active} · Priority {prio}   "
                + " ".join(cells))

    @staticmethod
    def _info_line(label, p, library) -> str:
        # Poison (☠) and energy (⚡) shown only when the player actually has some.
        counters = ""
        if p.get("poison", 0) > 0:
            counters += f"  ☠ {p['poison']}"
        if p.get("energy", 0) > 0:
            counters += f"  ⚡ {p['energy']}"
        return (f"{label}  ♥ {p['life']}{counters}  "
                f"mana [{decode.fmt_mana(p['mana'])}]  "
                f"hand {p['hand_count']}  lib {library}")

    async def _rebuild_stack(self, stack, mirrored: bool) -> None:
        """Rebuild the stack as one hoverable, delineated StackItem per object
        (top of stack first). Each carries its human-frame targets so hovering
        it can highlight them on the board."""
        box = self.query_one("#stack", HorizontalScroll)
        await box.remove_children()
        if not stack:
            await box.mount(Static("Stack: (empty)", classes="stack-empty"))
            return
        widgets = [StackItem(self._stack_item_label(e),
                             stack_target_refs(e, mirrored))
                   for e in stack]
        await box.mount(*widgets)

    @staticmethod
    def _stack_item_label(e) -> str:
        kind = "spell" if e["is_spell"] else "ability"
        label = f"{e['name']} ({kind}, {e['controller']})"
        if e.get("targets"):
            label += " → " + "; ".join(e["targets"])
        return label


# ── Entry point (called by play.py --tui) ─────────────────────────────────────

def run(binary_path, model_path, human_player=None,
        human_deck="delver", model_deck="delver", bo3=True,
        human_clock_s=None, hard_timeout=False):
    """Launch the TUI. `model_path` of None/"scripted" ⇒ rule-based opponent.

    Any agent spec ``opponents.make_controller`` accepts works here — a
    checkpoint path / deck shorthand, or a scripted tier ("scripted:hard",
    "explore", ...) — so the TUI opponent shares the one agent grammar.

    `bo3` (default True) plays a best-of-three match — with sideboarding between
    games — in a single engine process; pass ``bo3=False`` for a single game.

    `human_clock_s` arms the human's own chess-clock bank (play.py
    ``--human-clock``); with `hard_timeout` (``--hard-timeout``) an exhausted
    bank concedes the match for whichever seat ran out. Both default off, so an
    untimed session behaves exactly as before.
    """
    GameApp(build_session(binary_path, model_path, human_player=human_player,
                          human_deck=human_deck, model_deck=model_deck,
                          bo3=bo3, human_clock_s=human_clock_s,
                          hard_timeout=hard_timeout)).run()
    return 0
