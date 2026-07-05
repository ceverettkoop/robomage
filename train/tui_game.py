"""Interactive RoboMage game board as a full-screen Textual TUI.

Reimplements the human-vs-opponent play loop (see play.py's CLI / raylib GUI) as
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

from textual import work
from textual.app import App, ComposeResult
from textual.content import Content
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from env import NarrativeEnv, STATE_SIZE
import decode

# Abbreviations for the 13-step phase strip (index aligns with state[18:31]).
_STEP_ABBR = ["UNT", "UPK", "DRW", "M1", "BGC", "ATK", "BLK",
              "FSD", "DMG", "EOC", "M2", "END", "CLN"]

# Index of the UPKEEP step in the 13-step one-hot (state[18:31]); autopass stops
# once it reaches this step (its "next turn" mark).
_STEP_UPKEEP_IDX = _STEP_ABBR.index("UPK")

# While the human is only auto-passing priority (spectating the opponent's turn),
# pause this many seconds after the opponent takes a non-pass action so the user
# can briefly observe it before the game moves on. Tweak freely.
OPP_ACTION_OBSERVE_DELAY = 0.5

# Controller-word substitution used when decoding an OPPONENT-perspective obs
# (the opponent holds priority, so the state vector's "self" is them). Swapping
# the labels keeps stack entries / announced targets worded from the human's
# point of view; the structural self/opp fields are swapped separately.
_MIRROR_LABELS = {"own": "opp", "opp": "own", "self": "opponent", "opponent": "self"}

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

class TuiEnv(NarrativeEnv):
    """NarrativeEnv that also suppresses the library-search option dumps (the TUI
    shows those choices via the action list instead)."""

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


# ── Clickable card widget ─────────────────────────────────────────────────────

class CardClicked(Message):
    """A battlefield permanent or hand card was clicked."""

    def __init__(self, card_idx: int, controller: str):
        self.card_idx = card_idx
        self.controller = controller          # "self" | "opp"
        super().__init__()


class CardButton(Static):
    """A single clickable card (permanent or hand card).

    `edge_colors` is a (top, right, bottom, left) tuple of rich colors encoding
    the card's color identity; each edge is painted its color at mount time so
    multicolor cards show a split border (see `_edge_colors`)."""

    def __init__(self, label: str, card_idx: int, controller: str,
                 edge_colors=None):
        super().__init__(label)
        self._card_idx = card_idx
        self._controller = controller
        self._edge_colors = edge_colors

    def on_mount(self) -> None:
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

    def on_click(self) -> None:
        self.post_message(CardClicked(self._card_idx, self._controller))


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


# ── The app ───────────────────────────────────────────────────────────────────

class GameApp(App):
    CSS = """
    #phase      { height: 1; background: $panel; color: $text; }
    #prompt     { height: 1; background: $boost; text-style: bold; }
    #opp-info   { height: 1; color: red; }
    #self-info  { height: 1; color: green; }
    #graveyards { height: 2; color: $text-muted; }
    #stack      { height: 3; border: round $primary; }
    #opp-bf, #self-bf { height: 8; }
    .bf-row     { height: 1fr; layout: horizontal; }
    .bf-row.lands { background: $panel; }
    #self-hand  { height: 5; border: round green; layout: horizontal; }
    CardButton  { width: auto; height: 100%; margin: 0 1; padding: 0 1;
                  border: round $surface; }
    CardButton:hover { background: $boost; }
    #bottom     { height: 1fr; min-height: 8; }
    #actions    { width: 35%; min-width: 24; border: round $primary; }
    #log        { width: 1fr; border: round $surface; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("space", "pass_zero", "Pass"),
        ("p", "autopass", "Autopass"),
        ("plus", "resize_log(1)", "Bigger log"),
        ("minus", "resize_log(-1)", "Smaller log"),
        ("0", "pick('0')", "Pick"),
        ("1", "pick('1')", ""), ("2", "pick('2')", ""), ("3", "pick('3')", ""),
        ("4", "pick('4')", ""), ("5", "pick('5')", ""), ("6", "pick('6')", ""),
        ("7", "pick('7')", ""), ("8", "pick('8')", ""), ("9", "pick('9')", ""),
    ]

    def __init__(self, env, opp_act, opp_is_a, human_deck, opp_deck, is_model):
        super().__init__()
        self._env = env
        self._opp_act = opp_act              # callable(obs, num_choices) -> int
        self._opp_is_a = opp_is_a
        self._human_deck = human_deck
        self._opp_deck = opp_deck
        self._opp_label = "Model" if is_model else "Scripted"
        self._human_q = self._make_queue()
        self._actions = []
        self._awaiting = False
        self._reward = 0.0
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
        with Vertical(id="opp-bf"):
            yield VerticalScroll(id="opp-bf-perms", classes="bf-row")
            yield VerticalScroll(id="opp-bf-lands", classes="bf-row lands")
        yield Static(id="stack")
        with Vertical(id="self-bf"):
            yield VerticalScroll(id="self-bf-perms", classes="bf-row")
            yield VerticalScroll(id="self-bf-lands", classes="bf-row lands")
        yield VerticalScroll(id="self-hand")
        yield Static(id="self-info")
        yield Static(id="graveyards")
        yield Static(id="prompt")
        with Horizontal(id="bottom"):
            yield OptionList(id="actions")
            yield RichLog(id="log", wrap=True, highlight=False, markup=True)
        yield Footer()

    def on_mount(self) -> None:
        human_seat = "B" if self._opp_is_a else "A"
        opp_seat = "A" if self._opp_is_a else "B"
        self.title = "RoboMage"
        self.sub_title = (f"You (Player {human_seat}, {self._human_deck})  vs  "
                          f"{self._opp_label} (Player {opp_seat}, {self._opp_deck})")
        self._log("[b]Game starting…[/b]  Click a card or pick a numbered action. "
                  "Keys: digits = pick, space = pass, p = autopass, q = quit.")
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
                a_has_priority = obs[32] > 0.5
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
                    action = int(self._opp_act(obs, num))
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

                obs, reward, terminated, truncated, _ = env.step(action)
                if reward:
                    self._reward = reward
                done = terminated or truncated
                flushed = env.flush_lines()
                self.post_message(LogLines(flushed))

                # Give the human a beat to observe game changes they didn't drive:
                # the opponent acting, or a stack resolution an auto-pass triggered.
                observed = opp_acted or (autopass_acted
                                         and any(ln.strip() for ln in flushed))
                if observed and not done:
                    time.sleep(OPP_ACTION_OBSERVE_DELAY)

            self.post_message(GameOver(self._winner_text()))
        except EOFError:
            self.post_message(GameOver(self._winner_text()))
        except Exception as exc:  # surface, don't swallow
            self.post_message(LogLines([f"[red]driver error: {exc!r}[/red]"]))
            self.post_message(GameOver(self._winner_text()))

    def _winner_text(self) -> str:
        human_is_a = not self._opp_is_a
        human_wins = (self._reward > 0 and human_is_a) or (self._reward < 0 and not human_is_a)
        if self._reward == 0:
            return "Game over — no winner detected (draw?)."
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

        self.query_one("#phase", Static).update(self._phase_strip(obs, gs))
        self.query_one("#opp-info", Static).update(self._info_line("OPPONENT", gs["opponent"], gs["opp_library"]))
        self.query_one("#self-info", Static).update(self._info_line("YOU", gs["self"], gs["self_library"]))
        self.query_one("#stack", Static).update(self._stack_text(gs["stack"]))
        self.query_one("#graveyards", Static).update(
            f"Your GY: {', '.join(gs['self_graveyard']) or '—'}\n"
            f"Opp GY:  {', '.join(gs['opp_graveyard']) or '—'}")

        await self._rebuild_bf("#opp-bf-perms", "#opp-bf-lands", gs["opp_battlefield"], "opp")
        await self._rebuild_bf("#self-bf-perms", "#self-bf-lands", gs["self_battlefield"], "self")
        # A mirrored obs's "self_hand" is the OPPONENT's hand (private) — never
        # render it; the human's hand keeps its last own-perspective contents.
        if not mirrored:
            await self._rebuild_hand(gs["self_hand"])

        self._actions = message.actions
        self._awaiting = message.human_turn
        opt = self.query_one("#actions", OptionList)
        opt.clear_options()
        if message.human_turn:
            for a in message.actions:
                # Wrap in plain Content: a str prompt is parsed as Textual markup
                # at render time, which swallows bracketed text like the [SELF]/
                # [OPPONENT] tags (and any card text that parses as a style tag).
                opt.add_option(Option(Content(f"{a['index']:>2}: {self._menu_label(a)}"),
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
        want = "own" if message.controller == "self" else "opp"
        strict = [a for a in self._actions
                  if a["card_idx"] == message.card_idx and a["controller"] == want]
        matches = strict or [a for a in self._actions if a["card_idx"] == message.card_idx]
        if not matches:
            self._log("[yellow]No legal action for that card — use the numbered list.[/yellow]")
        elif len(matches) == 1:
            self._submit(matches[0]["index"])
        else:
            idxs = [a["index"] for a in matches]
            self.query_one("#actions", OptionList).highlighted = idxs[0]
            self._log(f"[yellow]That card has {len(idxs)} options: "
                      f"{', '.join(str(i) for i in idxs)} — pick a number.[/yellow]")

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
        return int(np.argmax(obs[18:31])) == _STEP_UPKEEP_IDX

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
        widgets = [self._mk_card(decode.fmt_perm(p), p["card_idx"], controller)
                   for p in perms]
        if widgets:
            await box.mount(*widgets)

    async def _rebuild_hand(self, hand) -> None:
        box = self.query_one("#self-hand", VerticalScroll)
        await box.remove_children()
        widgets = [self._mk_card(c["name"], c["card_idx"], "self") for c in hand]
        if widgets:
            await box.mount(*widgets)

    @staticmethod
    def _mk_card(label: str, card_idx: int, controller: str) -> "CardButton":
        """Build a CardButton whose border edges encode the card's color identity."""
        edges = _edge_colors(decode.card_border_colors(card_idx))
        return CardButton(label, card_idx, controller, edges)

    @staticmethod
    def _phase_strip(obs, gs) -> str:
        cur = int(np.argmax(obs[18:31]))
        cells = []
        for i, abbr in enumerate(_STEP_ABBR):
            cells.append(f"[reverse b]{abbr}[/reverse b]" if i == cur else f"[dim]{abbr}[/dim]")
        active = "A" if gs["active_is_a"] else "B"
        prio = gs["priority_player"]
        return (f"Turn {gs['turn']} · Active {active} · Priority {prio}   "
                + " ".join(cells))

    @staticmethod
    def _info_line(label, p, library) -> str:
        return (f"{label}  ♥ {p['life']}  ☠ {p['poison']}  "
                f"mana [{decode.fmt_mana(p['mana'])}]  "
                f"hand {p['hand_count']}  lib {library}")

    @staticmethod
    def _stack_text(stack) -> str:
        if not stack:
            return "Stack: (empty)"
        parts = []
        for e in stack:
            kind = "spell" if e["is_spell"] else "ability"
            parts.append(f"{e['name']} ({kind}, {e['controller']})")
        return "Stack: " + "  ◄  ".join(parts)

    @staticmethod
    def _prompt_text(obs, num, gs) -> str:
        cats = decode.action_categories(obs, num)
        if decode.is_mulligan(cats):
            return "Mulligan decision — keep or mulligan?"
        if decode.is_bottom(cats):
            return "Choose a card to put on the bottom of your library."
        if decode.is_search(cats):
            return "Search your library (pick a card, or 'fail to find')."
        cset = set(int(c) for c in cats)
        if 8 in cset:
            return "Choose a target."
        if cset & {2, 3}:
            return "Declare attackers — pick creatures, then Confirm attackers."
        if cset & {4, 5}:
            return "Declare blockers — pick creatures, then Confirm blockers."
        active = "A" if gs["active_is_a"] else "B"
        return f"Player {active}'s turn — {gs['step']}: choose an action."


# ── Entry point (called by play.py --tui) ─────────────────────────────────────

def run(binary_path, model_path, human_player=None,
        human_deck="delver", model_deck="delver"):
    """Launch the TUI. `model_path` of None/"scripted" ⇒ rule-based opponent.

    Any agent spec ``opponents.make_controller`` accepts works here — a
    checkpoint path / deck shorthand, or a scripted tier ("scripted:hard",
    "explore", ...) — so the TUI opponent shares the one agent grammar.
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
    env = TuiEnv(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                 log_viewer=human_seat)

    def opp_act(obs, num):
        return int(ctrl.choose(obs, num, action_masks=env.action_masks()))

    GameApp(env, opp_act, opp_is_a, human_deck, model_deck, is_model).run()
    return 0
