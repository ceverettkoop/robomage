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

import random
import re
import time

import numpy as np
from rich.text import Text

from textual import events, work
from textual.app import App, ComposeResult
from textual.content import Content
from textual.containers import Horizontal, HorizontalScroll, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, OptionList, RichLog, Static
from textual.widgets.option_list import Option, OptionDoesNotExist

from env import (NarrativeEnv, STATE_SIZE, _SELF_IS_A_IDX,
                 _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE)
import decode

# Abbreviations for the step phase strip (index aligns with the step one-hot).
_STEP_ABBR = ["UNT", "UPK", "DRW", "M1", "BGC", "ATK", "BLK",
              "FSD", "DMG", "EOC", "M2", "END", "CLN"]

# Index of the UPKEEP step WITHIN the step one-hot (argmax result, not an absolute
# obs offset); autopass stops once it reaches this step (its "next turn" mark).
_STEP_UPKEEP_IDX = _STEP_ABBR.index("UPK")

# While the human is only auto-passing priority (spectating the opponent's turn),
# pause this many seconds after the opponent takes a non-pass action so the user
# can briefly observe it before the game moves on. Tweak freely.
OPP_ACTION_OBSERVE_DELAY = 0.5

# The card-inspect ("hold Q") banner auto-hides this many seconds after the last
# 'q'. A terminal has no key-up event, so "hold" is emulated: OS key auto-repeat
# fires the inspect action repeatedly while Q is down, each firing re-arming this
# timer, so the banner stays up until the key is released (repeats stop) and the
# timer lapses. Must comfortably exceed the auto-repeat interval; tune if a fast
# release flickers or a slow keyboard's first-repeat gap lets it blink.
ORACLE_HIDE_DELAY = 0.8

# Controller-word substitution used when decoding an OPPONENT-perspective obs
# (the opponent holds priority, so the state vector's "self" is them). Swapping
# the labels keeps stack entries / announced targets worded from the human's
# point of view; the structural self/opp fields are swapped separately.
_MIRROR_LABELS = {"own": "opp", "opp": "own", "self": "opponent", "opponent": "self"}

# ActionRefZone (src/classes/gamestate.h) -> board zone a CardButton lives in.
# Only battlefield/hand have widgets; other zones (stack/gy/exile/player) map to
# None. Lets cross-highlighting tell a hand card from a same-named battlefield
# permanent, which card_idx + controller alone cannot.
_ZONE_REF_TO_ZONE = {1: "battlefield", 2: "battlefield", 3: "hand"}

# Opponent choices that reveal a hidden card identity (the card never becomes
# public knowledge). When the opponent makes one of these, the log shows a
# generic "a card" message instead of the actual name — e.g. Ponder/Brainstorm
# put a card on top, not "Put Lightning Bolt on top". Keyed by action category
# (see ActionCategory in CLAUDE.md). Choices with no chosen card (card_idx < 0,
# e.g. "Fail to find", "Take nothing") fall through to their normal description.
_OPP_PRIVATE_DESC = {
    12: "Put a card on the bottom of their library",  # BOTTOM_DECK_CARD
    19: "Search their library for a card",            # SEARCH_LIBRARY
    20: "Put a card on top of their library",          # TOP_LIBRARY
    23: "Take a card",                                 # DIG_CHOICE
}


def _opp_event_text(action, opp_label):
    """Log text for an opponent's chosen action, redacting private card identities.

    A choice whose card was publicly revealed (card_is_public, e.g. Personal Tutor)
    keeps its real description even in an otherwise-private category."""
    cat = action["category"]
    if (action["card_idx"] >= 0 and cat in _OPP_PRIVATE_DESC
            and not action.get("card_is_public")):
        desc = _OPP_PRIVATE_DESC[cat]
    else:
        desc = action["description"]
    return f"[{opp_label}] {desc}"


# ── Engine wrapper ────────────────────────────────────────────────────────────

class TuiEnvMixin:
    """Suppresses the library-search option dumps (the TUI shows those choices
    via the action list instead). A mixin so the same filtering layers over
    either env base: NarrativeEnv normally, SearchNarrativeEnv when the
    opponent is a search (MCTS) controller."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._in_search_block = False

    def _print_narrative_line(self, line):
        if "Searching" in line and "library:" in line:
            self._in_search_block = True
            return
        if self._in_search_block:
            if line.startswith("  ") or not line.strip():
                return
            self._in_search_block = False
        self.lines.append(line)


class TuiEnv(TuiEnvMixin, NarrativeEnv):
    pass


# ── Color-identity border helpers ─────────────────────────────────────────────

def _edge_colors(colors):
    """Map a card's color-identity border colors to (top, right, bottom, left).

    A monocolor/land/colorless card paints all four edges its one color; a
    two-color card splits diagonally (top+left vs right+bottom); three or more
    colors distribute round-robin across the edges so multicolor cards read as
    visibly split rather than a single hue."""
    if not colors:
        return (None, None, None, None)
    if len(colors) == 1:
        c = colors[0]
        return (c, c, c, c)
    if len(colors) == 2:
        a, b = colors
        return (a, b, b, a)
    return tuple(colors[i % len(colors)] for i in range(4))


# Trailing icons appended to a hand card's label so its kind is obvious at a
# glance (a hand card shows only its name otherwise). Lands and creatures are the
# two kinds worth flagging; a card that is both reads as a land (that is how it is
# played from hand).
_LAND_ICON = "🏔"
_CREATURE_ICON = "🐾"


def _hand_type_icon(card_idx):
    """Land/creature icon for a hand card's label, or '' for anything else."""
    types = decode.card_types(card_idx).split()
    if "Land" in types:
        return _LAND_ICON
    if "Creature" in types:
        return _CREATURE_ICON
    return ""


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
    def __init__(self, obs, num_choices, actions, human_turn, opp_perspective=False,
                 perm_counters=None, perm_token_names=None):
        self.obs = obs
        self.num_choices = num_choices
        self.actions = actions            # decoded action dicts (human turn) or []
        self.human_turn = human_turn
        # True when this obs was serialized from the OPPONENT's perspective
        # (they hold priority): the renderer must mirror it back to the human's
        # view and must not display the "self" hand (it's the opponent's).
        self.opp_perspective = opp_perspective
        # (self[48], opp[48]) typed-counter summaries (env._perm_counters
        # side-channel), same perspective as obs — decode swaps them with it.
        self.perm_counters = perm_counters
        # (self[48], opp[48]) token names (env._perm_token_names side-channel),
        # same perspective as obs — decode swaps them with it.
        self.perm_token_names = perm_token_names
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
        ("greater_than_sign", "resize_log(1)", "Bigger log"),
        ("less_than_sign", "resize_log(-1)", "Smaller log"),
        ("0", "pick('0')", "Pick"),
        ("1", "pick('1')", ""), ("2", "pick('2')", ""), ("3", "pick('3')", ""),
        ("4", "pick('4')", ""), ("5", "pick('5')", ""), ("6", "pick('6')", ""),
        ("7", "pick('7')", ""), ("8", "pick('8')", ""), ("9", "pick('9')", ""),
    ]

    def __init__(self, env, opp_act, opp_is_a, human_deck, opp_deck, is_model,
                 bo3=True, clock_fn=None):
        super().__init__()
        self._env = env
        self._opp_act = opp_act              # callable(obs, num_choices) -> int
        self._opp_is_a = opp_is_a
        # Optional callable -> the opponent's remaining match-clock seconds
        # (None when no chess clock is armed). Read on the UI thread by the
        # thinking ticker; a single dict-float read, so cross-thread safe.
        self._clock_fn = clock_fn
        self._human_deck = human_deck
        self._opp_deck = opp_deck
        self._bo3 = bo3
        # Latest decoded bo3 match context ({game_number, self_wins, opp_wins,
        # is_sideboard}, human-frame) — drives the score line and the winner
        # text. Populated on every StateUpdate; None until the first one.
        self._match = None
        self._is_model = is_model
        self._opp_label = "Model" if is_model else "Scripted"
        self._human_q = self._make_queue()
        self._actions = []
        self._awaiting = False
        self._reward = 0.0
        # Set the instant ctrl+q is pressed so the driver thread — which may be
        # blocked deep inside a long opponent search when the killed engine pipe
        # raises — exits silently instead of posting into a dying app.
        self._quitting = False
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
        # Autopass (the 'p' key): once engaged, the driver passes priority through
        # every optional window until the next UPKEEP step, stopping early for any
        # mandatory decision.
        self._autopass = False
        # Live heights of the resizable board panels (must match the CSS
        # defaults above). Shrinking these frees rows that flow into the 1fr
        # #bottom region, growing the log/command area; see action_resize_log.
        self._panel_h = {"#opp-bf": 8, "#self-bf": 8, "#self-hand": 5}

    @staticmethod
    def _make_queue():
        import queue
        return queue.Queue(maxsize=1)

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
                  "hold q or right-click a card = show oracle text, ctrl+q = quit.")
        self._drive()

    # ----- the driver (background thread) -----

    @work(thread=True, exclusive=True)
    def _drive(self) -> None:
        env = self._env
        try:
            obs, _ = env.reset()
            self.post_message(LogLines(env.flush_lines()))
            done = False
            while not done:
                num = env._num_choices
                a_has_priority = obs[_SELF_IS_A_IDX] > 0.5
                opp_turn = (a_has_priority == self._opp_is_a)
                actions = decode.decode_actions_from_obs(
                    obs, num, env._action_public,
                    descriptions=getattr(env, "_action_descriptions", None))

                # Resolve who drives this decision BEFORE posting the state, so
                # the UI shows the action menu exactly when the human must act.
                # Autopass keeps passing until it reaches the next turn's upkeep;
                # a mandatory decision (no pass option) always stops it.
                pass_idx = None if opp_turn else self._pass_index(actions)
                autopass_now = (not opp_turn and self._autopass
                                and pass_idx is not None
                                and not self._autopass_should_stop(obs))
                if not opp_turn and not autopass_now:
                    self._autopass = False            # stop autopass; human acts
                human_must_act = not opp_turn and not autopass_now

                self.post_message(StateUpdate(obs, num,
                                              actions if human_must_act else [],
                                              human_turn=human_must_act,
                                              opp_perspective=opp_turn,
                                              perm_counters=getattr(env, "_perm_counters", None),
                                              perm_token_names=getattr(env, "_perm_token_names", None)))

                opp_acted = False
                autopass_acted = False
                if opp_turn:
                    # A model/search opponent can take seconds to answer; show a
                    # thinking indicator around the call (the driver runs on a
                    # worker thread, so the UI stays live meanwhile). The finally
                    # clears it even if _opp_act raises (quit/error).
                    if self._is_model:
                        self.post_message(OppThinking(True))
                    try:
                        action = int(self._opp_act(obs, num))
                    finally:
                        if self._is_model:
                            self.post_message(OppThinking(False))
                    if 0 <= action < len(actions) and actions[action]["category"] != 0:
                        self.post_message(LogLines(
                            [_opp_event_text(actions[action], self._opp_label)]))
                        opp_acted = True
                elif autopass_now:
                    action = pass_idx
                    autopass_acted = True
                else:
                    action = self._human_q.get()       # blocks until UI delivers
                    if action is None:                 # quit signalled
                        return

                obs, reward, terminated, truncated, info = env.step(action)
                if reward:
                    self._reward = reward
                done = terminated or truncated
                flushed = env.flush_lines()
                self.post_message(LogLines(flushed))

                # Announce each bo3 game result as it lands (the running score is
                # only in the next obs, so report from this obs's match context).
                if self._bo3 and info.get("game_result") and not done:
                    self.post_message(LogLines([self._game_break_text(obs)]))

                # Give the human a beat to observe game changes they didn't drive:
                # the opponent acting, or a stack resolution an auto-pass triggered.
                observed = opp_acted or (autopass_acted
                                         and any(ln.strip() for ln in flushed))
                if observed and not done:
                    time.sleep(OPP_ACTION_OBSERVE_DELAY)

            self.post_message(GameOver(self._winner_text()))
        except EOFError:
            if self._quitting:
                return
            self.post_message(GameOver(self._winner_text()))
        except Exception as exc:  # surface, don't swallow
            # A ctrl+q mid-search kills the engine, which unblocks the pipe read
            # with an error that lands here — but the app is already tearing down,
            # so return silently rather than posting into a dying app.
            if self._quitting:
                return
            self.post_message(LogLines([f"[red]driver error: {exc!r}[/red]"]))
            self.post_message(GameOver(self._winner_text()))

    def _game_break_text(self, obs) -> str:
        """A '=== Game N over — You lead 2–1 ===' banner between bo3 games.

        Reads the running match score from the *next* game's obs (its match
        context already reflects the just-finished game), mirroring self/opp when
        that obs is serialized from the opponent's perspective."""
        mctx = decode._decode_match_context(obs[:STATE_SIZE])
        opp_perspective = (obs[_SELF_IS_A_IDX] > 0.5) == self._opp_is_a
        you, opp = mctx["self_wins"], mctx["opp_wins"]
        if opp_perspective:
            you, opp = opp, you
        if you > opp:
            lead = f"You lead {you}–{opp}"
        elif opp > you:
            lead = f"{self._opp_label} leads {opp}–{you}"
        else:
            lead = f"Tied {you}–{opp}"
        return f"=== Game over — {lead} ==="

    def _winner_text(self) -> str:
        human_is_a = not self._opp_is_a
        human_wins = (self._reward > 0 and human_is_a) or (self._reward < 0 and not human_is_a)
        if self._reward == 0:
            return "Game over — no winner detected (draw?)."
        # In bo3 the terminal reward is the MATCH result; report it with the
        # final game score. The match ends the instant the deciding game does, so
        # no fresh obs follows: the stored match context reflects the score
        # *entering* that last game — the winner takes it, so add 1 to their tally.
        if self._bo3:
            score = ""
            if self._match:
                you = self._match["self_wins"] + (1 if human_wins else 0)
                opp = self._match["opp_wins"] + (0 if human_wins else 1)
                score = f" ({you}–{opp})"
            return (f"=== You win the match!{score} ===" if human_wins
                    else f"=== {self._opp_label} wins the match{score}. ===")
        return "=== You win! ===" if human_wins else f"=== {self._opp_label} wins. ==="

    # ----- message handlers (UI thread) -----

    async def on_state_update(self, message: StateUpdate) -> None:
        obs = message.obs
        # The state vector is serialized from the PRIORITY player's perspective.
        # When that's the opponent, mirror the decode back to the human's view
        # (swapped controller labels + swapped self/opp fields) so the board
        # never flips while a model/scripted opponent is acting or deciding.
        mirrored = message.opp_perspective
        gs = decode.decode_game_state(
            obs[:STATE_SIZE],
            labels=_MIRROR_LABELS if mirrored else decode.SELF_OPP_LABELS,
            perm_counters=message.perm_counters,
            perm_token_names=message.perm_token_names)
        if mirrored:
            for a, b in (("self", "opponent"),
                         ("self_library", "opp_library"),
                         ("self_battlefield", "opp_battlefield"),
                         ("self_graveyard", "opp_graveyard")):
                gs[a], gs[b] = gs[b], gs[a]
            # Match wins are viewer-relative too — swap so self_wins == YOU.
            m = gs["match"]
            m["self_wins"], m["opp_wins"] = m["opp_wins"], m["self_wins"]

        # Remember the human-frame match context for the score line + winner text.
        self._match = gs["match"]

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
        self._actions = message.actions
        self._awaiting = message.human_turn
        opt = self.query_one("#actions", OptionList)
        opt.clear_options()
        if message.human_turn:
            for a in message.actions:
                opt.add_option(Option(self._option_prompt(a, False),
                                      id=str(a["index"])))
            if message.actions:
                opt.highlighted = 0
            self.query_one("#prompt", Static).update(self._prompt_text(obs, message.num_choices, gs))
        elif self._autopass:
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
        remaining = self._clock_fn() if self._clock_fn is not None else None
        if remaining is not None:
            r = int(remaining)
            bank = f"  ·  bank {r // 60}:{r % 60:02d}"
        self.query_one("#prompt", Static).update(
            f"⏳ {self._opp_label} is thinking…  ({elapsed}s){bank}")

    def on_game_over(self, message: GameOver) -> None:
        self._awaiting = False
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
        matches = self._actions_for_card(message.card_idx, message.controller,
                                         message.zone)
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
            idxs = [a["index"] for a in self._actions_for_card(
                node._card_idx, node._controller, node._zone)]
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

    @staticmethod
    def _action_zone(a):
        """The board zone the action's referenced card lives in ("battlefield" /
        "hand"), or None when it names no board card (or carries no zone info)."""
        return _ZONE_REF_TO_ZONE.get(a.get("zone_ref", 0))

    def _actions_for_card(self, card_idx: int, controller: str, zone: str):
        """Legal actions this card feeds. Prefer a controller-exact match; fall
        back to card-id alone (some actions carry no controller flag). An action
        whose zone_ref names a DIFFERENT board zone is excluded, so a hand card
        and a same-named battlefield permanent don't cross-match. Mirrors the
        click resolver so click and hover always agree on the same set."""
        if card_idx < 0:
            return []

        def zone_ok(a):
            az = self._action_zone(a)
            return az is None or az == zone   # no zone info → don't exclude
        want = "own" if controller == "self" else "opp"
        strict = [a for a in self._actions
                  if a["card_idx"] == card_idx and a["controller"] == want and zone_ok(a)]
        return strict or [a for a in self._actions
                          if a["card_idx"] == card_idx and zone_ok(a)]

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
        want_zone = self._action_zone(a)
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
        text = f"{a['index']:>2}: {self._menu_label(a)}"
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

    def _menu_label(self, a) -> str:
        """Action description for the menu, tagging player choices SELF/OPPONENT.

        Any action whose description names a player seat ("Player A"/"Player B" —
        target-a-player, choose-a-player mid-resolution, etc.) gets a trailing
        SELF/OPPONENT marker — relative to you — so it's unambiguous which seat
        it refers to. The per-action controller flag can't be used here: the
        engine only emits it for actions with a zone_ref, and player choices
        have none. The menu is only shown on the human's own decisions, so the
        human's seat is fixed by _opp_is_a."""
        desc = a["description"]
        m = re.search(r"\bPlayer ([AB])\b", desc)
        if m:
            human_seat = "B" if self._opp_is_a else "A"
            desc += "  [SELF]" if m.group(1) == human_seat else "  [OPPONENT]"
        return desc

    @staticmethod
    def _pass_index(actions):
        """Index of the 'pass priority' action (category 0) in a menu, else None."""
        for a in actions:
            if a["category"] == 0:
                return a["index"]
        return None

    def _autopass_should_stop(self, obs) -> bool:
        """True once autopass reaches an UPKEEP step — its stop mark.

        Autopass otherwise only halts for mandatory decisions, which the driver
        detects separately by the absence of a pass option."""
        return int(np.argmax(
            obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE])) == _STEP_UPKEEP_IDX

    def action_autopass(self) -> None:
        """Engage autopass: pass priority every optional window until the next
        UPKEEP step, stopping early for any mandatory decision (declare
        attackers/blockers, discard, targeting, mulligan — anything with no
        'pass priority' option). The driver auto-drives once engaged."""
        if not self._awaiting:
            return
        idx = self._pass_index(self._actions)
        if idx is None:                    # nothing to pass right now
            return
        self._autopass = True
        self._write_event("[You] Autopass to next upkeep")
        # Unblock the driver's current get with a pass; it takes over from there.
        # Bypass _submit (which re-enables input) so no stray keypress is queued
        # while the driver auto-drives — input stays disabled until autopass stops.
        self._awaiting = False
        self.query_one("#actions", OptionList).clear_options()
        self.query_one("#prompt", Static).update("Autopass — passing to next upkeep…")
        try:
            self._human_q.put_nowait(idx)
        except Exception:
            self._human_q.put(idx)

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
        # Flag the shutdown first: if the driver is blocked in a long opponent
        # search, on_unmount's env.close() kills the engine and the pipe read
        # errors out in _drive, which then returns silently on this flag.
        self._quitting = True
        try:
            self._human_q.put_nowait(None)
        except Exception:
            pass
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
        try:
            self._human_q.put_nowait(idx)
        except Exception:
            self._human_q.put(idx)

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
        icon = _hand_type_icon(card_idx)
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
                             self._stack_target_refs(e, mirrored))
                   for e in stack]
        await box.mount(*widgets)

    @staticmethod
    def _stack_item_label(e) -> str:
        kind = "spell" if e["is_spell"] else "ability"
        label = f"{e['name']} ({kind}, {e['controller']})"
        if e.get("targets"):
            label += " → " + "; ".join(e["targets"])
        return label

    @staticmethod
    def _stack_target_refs(e, mirrored: bool):
        """Stack-entry targets converted to the human frame. The decoded is_self
        is viewer-relative; when the obs is the opponent's perspective (mirrored)
        it must be flipped so is_self means the human (YOU)."""
        refs = []
        for t in e.get("target_refs", []):
            refs.append({"is_player": t["is_player"],
                         "is_self": (t["is_self"] != mirrored),
                         "card_idx": t["card_idx"]})
        return refs

    @staticmethod
    def _prompt_text(obs, num, gs) -> str:
        cats = decode.action_categories(obs, num)
        match = gs.get("match")
        if match and match.get("is_sideboard"):
            # Sideboard between games of a bo3: swap cards in/out, then finish.
            return ("Sideboarding — add/remove cards for the next game, "
                    "then choose 'Done'.")
        if decode.is_mulligan(cats):
            return "Mulligan decision — keep or mulligan?"
        if decode.is_bottom(cats):
            return "Choose a card to put on the bottom of your library."
        if decode.is_search(cats):
            return "Search your library (pick a card, or 'fail to find')."
        cset = set(int(c) for c in cats)
        if 8 in cset:
            # Name the spell/ability asking for the target (its source may not be
            # on the stack yet — targets are announced first), so the prompt says
            # WHAT you're targeting for, not just "Choose a target."
            pend = gs.get("pending_decision")
            if pend and pend.get("name"):
                return f"Choose a target for {pend['name']}."
            return "Choose a target."
        if cset & {2, 3}:
            return "Declare attackers — pick creatures, then Confirm attackers."
        if cset & {4, 5}:
            return "Declare blockers — pick creatures, then Confirm blockers."
        active = "A" if gs["active_is_a"] else "B"
        return f"Player {active}'s turn — {gs['step']}: choose an action."


# ── Entry point (called by play.py --tui) ─────────────────────────────────────

def run(binary_path, model_path, human_player=None,
        human_deck="delver", model_deck="delver", bo3=True):
    """Launch the TUI. `model_path` of None/"scripted" ⇒ rule-based opponent.

    Any agent spec ``opponents.make_controller`` accepts works here — a
    checkpoint path / deck shorthand, or a scripted tier ("scripted:hard",
    "explore", ...) — so the TUI opponent shares the one agent grammar.

    `bo3` (default True) plays a best-of-three match — with sideboarding between
    games — in a single engine process; pass ``bo3=False`` for a single game.
    """
    from opponents import make_controller, is_scripted_spec

    spec = "scripted" if model_path is None else model_path
    ctrl = make_controller(spec, deterministic=True)
    is_model = isinstance(spec, str) and not is_scripted_spec(spec)

    # Seat assignment mirrors play.py: opponent ("model") seat is A or B.
    if human_player in ("A", "B"):
        opp_is_a = (human_player == "B")
    else:
        opp_is_a = random.random() < 0.5

    deck_a = model_deck if opp_is_a else human_deck
    deck_b = human_deck if opp_is_a else model_deck

    # Pin the engine's private-narrative perspective to the human's seat so the
    # game log only reveals what the human would actually know (their own draws,
    # tutored/top-of-library cards) — not the opponent's hidden information.
    human_seat = "B" if opp_is_a else "A"
    # A search (MCTS) opponent needs the engine's --search-server protocol and
    # a handle on the live env — mirror runner.run_games' duck-typed env swap.
    # Without this a SearchController would run but silently never search
    # (last_search_safe missing -> permanent raw-policy fallback).
    env_cls = TuiEnv
    if getattr(ctrl, "wants_search_env", False):
        from search_env import SearchNarrativeEnv

        class TuiSearchEnv(TuiEnvMixin, SearchNarrativeEnv):
            pass

        env_cls = TuiSearchEnv
    env = env_cls(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                  log_viewer=human_seat, bo3=bo3)
    bind_env = getattr(ctrl, "bind_env", None)
    if bind_env is not None:
        bind_env(env)

    def opp_act(obs, num):
        return int(ctrl.choose(obs, num, action_masks=env.action_masks()))

    # Surface a match-clock bank (mcts:/az: ?clock=) in the thinking indicator.
    clock_fn = None
    if isinstance(getattr(ctrl, "stats", None), dict) and ctrl.stats.get("clock_bank"):
        clock_fn = lambda: ctrl.stats.get("clock_remaining")  # noqa: E731

    GameApp(env, opp_act, opp_is_a, human_deck, model_deck, is_model, bo3=bo3,
            clock_fn=clock_fn).run()
    return 0
