"""Interactive RoboMage game board as a full-screen Textual TUI.

Reimplements the human-vs-opponent play loop (see play.py's CLI / raylib GUI) as
a terminal UI: rendered battlefield, hand, stack, graveyards, life/mana, phase
strip, and a numbered action list. The human acts either by clicking a card/zone
or by choosing a numbered option.

Driving the engine is delegated entirely to RoboMageEnv (via NarrativeEnv): the
subprocess, the BQUERY binary protocol, and the attacker/blocker confirm-slot
remapping all live there. This module is a front-end over that loop.

The opponent is either a trained model (MaskablePPO checkpoint) or the rule-based
scripted agent (env.scripted_action) when `model_path` is None or "scripted".

Invoked via `play.py --tui` (and the tui.py launcher's Play entry).
"""

import random

import numpy as np
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Footer, Header, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from env import NarrativeEnv, scripted_action, STATE_SIZE
import decode

# Abbreviations for the 13-step phase strip (index aligns with state[18:31]).
_STEP_ABBR = ["UNT", "UPK", "DRW", "M1", "BGC", "ATK", "BLK",
              "FSD", "DMG", "EOC", "M2", "END", "CLN"]

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


# ── Clickable card widget ─────────────────────────────────────────────────────

class CardClicked(Message):
    """A battlefield permanent or hand card was clicked."""

    def __init__(self, card_idx: int, controller: str):
        self.card_idx = card_idx
        self.controller = controller          # "self" | "opp"
        super().__init__()


class CardButton(Static):
    """A single clickable card (permanent or hand card)."""

    def __init__(self, label: str, card_idx: int, controller: str):
        super().__init__(label)
        self._card_idx = card_idx
        self._controller = controller

    def on_click(self) -> None:
        self.post_message(CardClicked(self._card_idx, self._controller))


# ── Worker → UI messages ──────────────────────────────────────────────────────

class StateUpdate(Message):
    def __init__(self, obs, num_choices, actions, human_turn):
        self.obs = obs
        self.num_choices = num_choices
        self.actions = actions            # decoded action dicts (human turn) or []
        self.human_turn = human_turn
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
    #opp-bf, #self-bf { height: 7; border: round $surface; }
    #self-hand  { height: 5; border: round green; }
    #opp-bf, #self-bf, #self-hand { layout: horizontal; }
    CardButton  { width: auto; height: 100%; margin: 0 1; padding: 0 1;
                  border: round $surface; }
    CardButton:hover { border: round $accent; }
    #bottom     { height: 1fr; }
    #actions    { width: 45%; border: round $primary; }
    #log        { width: 1fr; border: round $surface; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "pass_priority", "Pass"),
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

    @staticmethod
    def _make_queue():
        import queue
        return queue.Queue(maxsize=1)

    # ----- layout -----

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id="phase")
        yield Static(id="opp-info")
        yield VerticalScroll(id="opp-bf")
        yield Static(id="stack")
        yield VerticalScroll(id="self-bf")
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
                  "Keys: digits = pick, p = pass, q = quit.")
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
                actions = decode.decode_actions_from_obs(obs, num, env._action_public)

                self.post_message(StateUpdate(obs, num, [] if opp_turn else actions,
                                              human_turn=not opp_turn))

                if opp_turn:
                    action = int(self._opp_act(obs, num))
                    if 0 <= action < len(actions) and actions[action]["category"] != 0:
                        self.post_message(LogLines(
                            [_opp_event_text(actions[action], self._opp_label)]))
                else:
                    action = self._human_q.get()       # blocks until UI delivers
                    if action is None:                 # quit signalled
                        return

                obs, reward, terminated, truncated, _ = env.step(action)
                if reward:
                    self._reward = reward
                done = terminated or truncated
                self.post_message(LogLines(env.flush_lines()))

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
        gs = decode.decode_game_state(obs[:STATE_SIZE])

        self.query_one("#phase", Static).update(self._phase_strip(obs, gs))
        self.query_one("#opp-info", Static).update(self._info_line("OPPONENT", gs["opponent"], gs["opp_library"]))
        self.query_one("#self-info", Static).update(self._info_line("YOU", gs["self"], gs["self_library"]))
        self.query_one("#stack", Static).update(self._stack_text(gs["stack"]))
        self.query_one("#graveyards", Static).update(
            f"Your GY: {', '.join(gs['self_graveyard']) or '—'}\n"
            f"Opp GY:  {', '.join(gs['opp_graveyard']) or '—'}")

        await self._rebuild("#opp-bf", gs["opp_battlefield"], "opp")
        await self._rebuild("#self-bf", gs["self_battlefield"], "self")
        await self._rebuild_hand(gs["self_hand"])

        self._actions = message.actions
        self._awaiting = message.human_turn
        opt = self.query_one("#actions", OptionList)
        opt.clear_options()
        if message.human_turn:
            for a in message.actions:
                opt.add_option(Option(f"{a['index']:>2}: {a['description']}", id=str(a["index"])))
            if message.actions:
                opt.highlighted = 0
            self.query_one("#prompt", Static).update(self._prompt_text(obs, message.num_choices, gs))
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

    def action_pass_priority(self) -> None:
        if not self._awaiting:
            return
        for a in self._actions:
            if a["category"] == 0:        # PASS
                self._submit(a["index"])
                return

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

    async def _rebuild(self, selector: str, perms, controller: str) -> None:
        box = self.query_one(selector, VerticalScroll)
        await box.remove_children()
        widgets = [CardButton(decode.fmt_perm(p), p["card_idx"], controller) for p in perms]
        if widgets:
            await box.mount(*widgets)

    async def _rebuild_hand(self, hand) -> None:
        box = self.query_one("#self-hand", VerticalScroll)
        await box.remove_children()
        widgets = [CardButton(c["name"], c["card_idx"], "self") for c in hand]
        if widgets:
            await box.mount(*widgets)

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
    """Launch the TUI. `model_path` of None/"scripted" ⇒ rule-based opponent."""
    is_model = model_path not in (None, "scripted")

    if is_model:
        try:
            from sb3_contrib import MaskablePPO as _Algo
            _maskable = True
        except ImportError:
            from stable_baselines3 import PPO as _Algo
            _maskable = False
        model = _Algo.load(model_path)
    else:
        model = None

    # Seat assignment mirrors play.py: opponent ("model") seat is A or B.
    if human_player in ("A", "B"):
        opp_is_a = (human_player == "B")
    else:
        opp_is_a = random.random() < 0.5

    deck_a = model_deck if opp_is_a else human_deck
    deck_b = human_deck if opp_is_a else model_deck

    env = TuiEnv(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b)

    def opp_act(obs, num):
        if model is not None:
            mask = env.action_masks() if _maskable else None
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)
        return int(scripted_action(obs, num))

    GameApp(env, opp_act, opp_is_a, human_deck, model_deck, is_model).run()
    return 0
