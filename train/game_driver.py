"""Front-end-agnostic driver for the RoboMage human-vs-opponent play loop.

Extracted from tui_game.py so the Textual TUI board and a future Qt GUI share
one engine-driving loop and one set of presentation helpers. The driver knows
nothing about the front end: it runs on a worker thread (`GameDriver.run`),
pulls decisions from RoboMageEnv (via NarrativeEnv), and reports every event to
a `DriverSink` the front end supplies. The front end marshals those callbacks
onto its own UI thread (Textual `post_message`, Qt queued signals) and delivers
the human's choices back via `GameDriver.submit`.

`build_session` assembles the env + opponent controller + seat/clock/pace
plumbing (the setup that used to live in tui_game.run) into a `Session` either
front end consumes. The module-level helpers (`decode_human_frame`,
`actions_for_card`, `menu_label`, `prompt_text`, …) are the pure, Rich-free
presentation logic both boards reuse; a front end adds any markup itself.
"""

import random
import re
import time
from dataclasses import dataclass

import numpy as np

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

# Dwell per game step when rendering passive BSTATE frames (a step_pacing
# session): each time the step one-hot advances, hold the frame this long so the
# human can see the game move through its steps instead of fast-forwarding to
# the next decision. Tweak freely.
STEP_OBSERVE_DELAY = 0.2

# Controller-word substitution used when decoding an OPPONENT-perspective obs
# (the opponent holds priority, so the state vector's "self" is them). Swapping
# the labels keeps stack entries / announced targets worded from the human's
# point of view; the structural self/opp fields are swapped separately.
_MIRROR_LABELS = {"own": "opp", "opp": "own", "self": "opponent", "opponent": "self"}

# ActionRefZone (src/classes/gamestate.h) -> board zone a card widget lives in.
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

# Trailing icons appended to a hand card's label so its kind is obvious at a
# glance (a hand card shows only its name otherwise). Lands and creatures are the
# two kinds worth flagging; a card that is both reads as a land (that is how it is
# played from hand).
_LAND_ICON = "🏔"
_CREATURE_ICON = "🐾"


# ── Engine wrapper ────────────────────────────────────────────────────────────

class TuiEnvMixin:
    """Suppresses the library-search option dumps (the board shows those choices
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


# ── Shared presentation helpers (front-end-agnostic, Rich-free) ───────────────

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


def hand_type_icon(card_idx):
    """Land/creature icon for a hand card's label, or '' for anything else."""
    types = decode.card_types(card_idx).split()
    if "Land" in types:
        return _LAND_ICON
    if "Creature" in types:
        return _CREATURE_ICON
    return ""


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


def action_zone(a):
    """The board zone the action's referenced card lives in ("battlefield" /
    "hand"), or None when it names no board card (or carries no zone info)."""
    return _ZONE_REF_TO_ZONE.get(a.get("zone_ref", 0))


def actions_for_card(actions, card_idx, controller, zone, slot=None):
    """Legal actions this card feeds, from the decoded action menu `actions`.
    Prefer a controller-exact match; fall back to card-id alone (some actions
    carry no controller flag). An action whose zone_ref names a DIFFERENT board
    zone is excluded, so a hand card and a same-named battlefield permanent don't
    cross-match. When `slot` is given (the permanent's unified entity-ref slot,
    decode's perm "slot"), an action carrying a slot_ref matches only if it refs
    THIS slot — so with multiple same-named permanents each one feeds exactly
    its own actions; ref-less actions still match by card/controller/zone.
    Mirrors the click resolver so click and hover always agree on the same set."""
    if card_idx < 0:
        return []

    def zone_ok(a):
        az = action_zone(a)
        return az is None or az == zone   # no zone info → don't exclude

    def slot_ok(a):
        ref = a.get("slot_ref", -1)
        return slot is None or ref < 0 or ref == slot   # no ref → don't exclude
    want = "own" if controller == "self" else "opp"
    strict = [a for a in actions
              if a["card_idx"] == card_idx and a["controller"] == want
              and zone_ok(a) and slot_ok(a)]
    return strict or [a for a in actions
                      if a["card_idx"] == card_idx and zone_ok(a) and slot_ok(a)]


def stack_target_refs(e, mirrored):
    """Stack-entry targets converted to the human frame. The decoded is_self
    is viewer-relative; when the obs is the opponent's perspective (mirrored)
    it must be flipped so is_self means the human (YOU)."""
    refs = []
    for t in e.get("target_refs", []):
        refs.append({"is_player": t["is_player"],
                     "is_self": (t["is_self"] != mirrored),
                     "card_idx": t["card_idx"],
                     "slot": t.get("slot", -1)})
    return refs


def menu_label(a, opp_is_a):
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
        human_seat = "B" if opp_is_a else "A"
        desc += "  [SELF]" if m.group(1) == human_seat else "  [OPPONENT]"
    return desc


def prompt_text(obs, num, gs):
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


def decode_human_frame(u):
    """Decode a StateUpdate's obs into the human's viewing frame; returns
    (game_state_dict, mirrored).

    The state vector is serialized from the PRIORITY player's perspective. When
    that's the opponent (u.opp_perspective), mirror the decode back to the
    human's view (swapped controller labels + swapped self/opp fields) so the
    board never flips while a model/scripted opponent is acting or deciding."""
    mirrored = u.opp_perspective
    gs = decode.decode_game_state(
        u.obs[:STATE_SIZE],
        labels=_MIRROR_LABELS if mirrored else decode.SELF_OPP_LABELS,
        perm_counters=u.perm_counters,
        perm_token_names=u.perm_token_names)
    if mirrored:
        for a, b in (("self", "opponent"),
                     ("self_library", "opp_library"),
                     ("self_battlefield", "opp_battlefield"),
                     ("self_graveyard", "opp_graveyard")):
            gs[a], gs[b] = gs[b], gs[a]
        # Match wins are viewer-relative too — swap so self_wins == YOU.
        m = gs["match"]
        m["self_wins"], m["opp_wins"] = m["opp_wins"], m["self_wins"]
    return gs, mirrored


# ── Autopass helpers ──────────────────────────────────────────────────────────

def _pass_index(actions):
    """Index of the 'pass priority' action (category 0) in a menu, else None."""
    for a in actions:
        if a["category"] == 0:
            return a["index"]
    return None


def _frame_step_idx(obs):
    """Index of the current game step within the step one-hot of `obs`."""
    return int(np.argmax(
        obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]))


def _autopass_should_stop(obs):
    """True once autopass reaches an UPKEEP step — its stop mark.

    Autopass otherwise only halts for mandatory decisions, which the driver
    detects separately by the absence of a pass option."""
    return int(np.argmax(
        obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE])) == _STEP_UPKEEP_IDX


# ── Worker → front-end payloads and sink ──────────────────────────────────────

@dataclass
class StateUpdate:
    """A single decision point handed to the front end.

    `opp_perspective` is True when this obs was serialized from the OPPONENT's
    perspective (they hold priority): the renderer must mirror it back to the
    human's view and must not display the "self" hand (it's the opponent's).
    `perm_counters` / `perm_token_names` are (self[48], opp[48]) side-channels
    (env._perm_counters / env._perm_token_names) in the same perspective as obs
    — decode swaps them with it.

    `obs` is a private COPY (the env reuses its observation buffer, and a
    search opponent or the analysis window refills it mid-decision).
    `search_safe` / `history_len` feed the analysis window: whether this
    decision is snapshot-restorable (None when the env has no search
    protocol), and the env action-history length at this decision (the replay
    prefix that reconstructs it on a detached analysis engine)."""
    obs: object
    num_choices: int
    actions: list                # decoded action dicts (human turn) or []
    human_turn: bool
    opp_perspective: bool = False
    perm_counters: object = None
    perm_token_names: object = None
    search_safe: object = None
    history_len: int = 0
    # A passive --broadcast-steps BSTATE frame: board display only — not a real
    # decision, so front ends must not feed it to the analysis window or treat
    # it as a menu/prompt change beyond clearing stale actions.
    passive: bool = False


@dataclass
class GameOver:
    text: str


class DriverSink:
    """Duck-typed callback interface a front end implements to receive GameDriver
    events. All four methods are invoked ON THE WORKER THREAD; the front end must
    marshal them onto its own UI thread (Textual post_message, Qt queued signals).
    The methods are documented here for reference, but need not be inherited —
    any object providing them works."""

    def on_state(self, u):
        """A new decision point: `u` is a StateUpdate to render."""

    def on_log(self, lines):
        """Append narrative/event log `lines` (a list[str])."""

    def on_opp_thinking(self, active):
        """Toggle the "opponent is thinking" indicator (model/search opponent)."""

    def on_game_over(self, text):
        """The game/match ended; `text` is the banner to show."""


# ── The driver ────────────────────────────────────────────────────────────────

class GameDriver:
    """Runs the engine env on a worker thread and reports to a DriverSink.

    Owns everything the old GameApp._drive did: the human-choice queue, the
    autopass state machine, the quit flag, the running reward/match context, and
    the bo3 match banners. A front end constructs one with its sink, starts
    `run()` on a background thread, feeds the human's picks back via `submit`,
    engages autopass via `engage_autopass`, and shuts it down via `request_quit`
    (the front end still owns closing the env)."""

    def __init__(self, env, opp_act, opp_is_a, is_model, opp_label, bo3, sink,
                 clock_fn=None, pace_idle=None, reset_options=None,
                 replay_actions=None):
        self._env = env
        self._opp_act = opp_act              # callable(obs, num_choices) -> int
        self._opp_is_a = opp_is_a
        self._is_model = is_model
        self._opp_label = opp_label
        self._bo3 = bo3
        self._sink = sink
        # Optional callable -> the opponent's remaining match-clock seconds
        # (None when no chess clock is armed). Exposed via clock_remaining() for
        # a front end's thinking ticker; a single dict-float read, so cross-thread
        # safe.
        self._clock_fn = clock_fn
        # Optional callable -> seconds to hold (usually 0.0) when the game
        # never queried the opponent between two human decisions. A paced
        # SearchController occasionally returns a 0.2-0.5s beat here so "no
        # decision was offered" is not leaked by an instant hand-back.
        self._pace_idle = pace_idle
        import queue
        self._human_q = queue.Queue(maxsize=1)
        # Current human-turn action menu (set each loop iteration before the loop
        # blocks for input), read by engage_autopass to find the pass option; []
        # outside a human decision.
        self._actions = []
        self._reward = 0.0
        # Latest decoded bo3 match context ({self_wins, opp_wins, ...}, human
        # frame) — drives the winner text. Updated every loop iteration.
        self._match = None
        # Set the instant a quit is requested so the loop — which may be blocked
        # deep inside a long opponent search when the killed engine pipe raises —
        # exits silently instead of reporting into a dying front end.
        self._quitting = False
        # Autopass (the 'p' key): once engaged, the loop passes priority through
        # every optional window until the next UPKEEP step, stopping early for any
        # mandatory decision.
        self._autopass = False
        # Step one-hot index of the last frame posted to the sink (real or
        # passive) — the passive-frame pacer dwells only when it advances.
        self._last_step_idx = None
        # Options dict for the env reset (e.g. {"engine_seed": N} to restore a
        # saved session's seed); None = the env picks its own seed.
        self._reset_options = reset_options
        # Saved-session action prefix to fast-forward through before handing
        # control to the interactive loop (see _replay_prefix).
        self._replay_actions = list(replay_actions) if replay_actions else None
        # Every action index fed to env.step this session, replayed prefix
        # included, appended just before the step call. Append-only, so a
        # UI-thread list() snapshot is always a consistent replayable prefix
        # regardless of env class (plain TuiEnv has no _action_history).
        self.action_log = []
        # Optional per-decision observer (e.g. shard_record.ShardRecorder
        # .observe_step), called on the worker thread after each MAIN-LOOP
        # env.step with (pre_step_obs, num_choices, action, reward, info,
        # done). Not called during a saved-session replay prefix (those
        # decisions belong to the session that recorded them). Exceptions are
        # swallowed — an observer must never break play.
        self.step_observer = None

    # ----- the loop (worker thread) -----

    def run(self) -> None:
        env = self._env
        try:
            obs, _ = env.reset(options=self._reset_options)
            self._sink.on_log(env.flush_lines())
            self._emit_passive_frames()
            done = False
            if self._replay_actions:
                obs, done = self._replay_prefix(obs)
                if obs is None:            # replay diverged; already reported
                    return
            opp_queried = True   # nothing to mask before the first decision
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
                pass_idx = None if opp_turn else _pass_index(actions)
                autopass_now = (not opp_turn and self._autopass
                                and pass_idx is not None
                                and not _autopass_should_stop(obs))
                if not opp_turn and not autopass_now:
                    self._autopass = False            # stop autopass; human acts
                human_must_act = not opp_turn and not autopass_now
                self._actions = actions if human_must_act else []
                self._match = self._human_match_context(obs)

                # The human is about to get their menu without the opponent
                # having been queried since the human's last decision (it had
                # no legal action, so no query was ever emitted). A paced
                # opponent occasionally asks us to hold a short beat here,
                # presented exactly like a real think, so the instant
                # hand-back doesn't reveal it had nothing to decide.
                if (human_must_act and not opp_queried
                        and self._pace_idle is not None and not self._quitting):
                    hold = float(self._pace_idle())
                    if hold > 0.0:
                        if self._is_model:
                            self._sink.on_opp_thinking(True)
                        try:
                            time.sleep(hold)
                        finally:
                            if self._is_model:
                                self._sink.on_opp_thinking(False)
                opp_queried = opp_turn

                self._last_step_idx = _frame_step_idx(obs)
                self._sink.on_state(StateUpdate(
                    obs.copy(), num,
                    actions if human_must_act else [],
                    human_turn=human_must_act,
                    opp_perspective=opp_turn,
                    perm_counters=getattr(env, "_perm_counters", None),
                    perm_token_names=getattr(env, "_perm_token_names", None),
                    search_safe=getattr(env, "last_search_safe", None),
                    history_len=len(getattr(env, "_action_history", ()))))

                opp_acted = False
                autopass_acted = False
                if opp_turn:
                    # A model/search opponent can take seconds to answer; show a
                    # thinking indicator around the call (the driver runs on a
                    # worker thread, so the UI stays live meanwhile). The finally
                    # clears it even if _opp_act raises (quit/error).
                    if self._is_model:
                        self._sink.on_opp_thinking(True)
                    try:
                        action = int(self._opp_act(obs, num))
                    finally:
                        if self._is_model:
                            self._sink.on_opp_thinking(False)
                    if 0 <= action < len(actions) and actions[action]["category"] != 0:
                        self._sink.on_log(
                            [_opp_event_text(actions[action], self._opp_label)])
                        opp_acted = True
                elif autopass_now:
                    action = pass_idx
                    autopass_acted = True
                else:
                    action = self._human_q.get()       # blocks until UI delivers
                    if action is None:                 # quit signalled
                        return

                self.action_log.append(int(action))
                # Copy the pre-step obs before stepping — the env may reuse
                # its obs buffer, so the reference alone could alias the
                # post-step state by the time the observer runs.
                pre_obs = (obs.copy() if self.step_observer is not None
                           else None)
                obs, reward, terminated, truncated, info = env.step(action)
                if reward:
                    self._reward = reward
                done = terminated or truncated
                if self.step_observer is not None:
                    try:
                        self.step_observer(pre_obs, num, int(action), reward,
                                           info, done)
                    except Exception:  # noqa: BLE001 — observer must not break play
                        pass
                # A finished game disengages autopass: the human drives the
                # between-games sideboard and the next game's mulligan
                # themselves instead of autopass blowing through them (its
                # UPKEEP stop mark never fires during those decisions).
                if self._autopass and info.get("game_result"):
                    self._autopass = False
                flushed = env.flush_lines()
                self._sink.on_log(flushed)

                # Announce each bo3 game result as it lands (the running score is
                # only in the next obs, so report from this obs's match context).
                if self._bo3 and info.get("game_result") and not done:
                    self._sink.on_log([self._game_break_text(obs)])

                # Give the human a beat to observe game changes they didn't drive:
                # the opponent acting, or a stack resolution an auto-pass triggered.
                observed = opp_acted or (autopass_acted
                                         and any(ln.strip() for ln in flushed))
                if observed and not done:
                    time.sleep(OPP_ACTION_OBSERVE_DELAY)

                # Render any forced-pass BSTATE frames the engine emitted while
                # fast-forwarding to the next decision (broadcast_steps env only),
                # dwelling briefly per step so the human sees the progression.
                self._emit_passive_frames()

            self._sink.on_game_over(self._winner_text())
        except EOFError:
            if self._quitting:
                return
            self._sink.on_game_over(self._winner_text())
        except Exception as exc:  # surface, don't swallow
            # A quit mid-search kills the engine, which unblocks the pipe read
            # with an error that lands here — but the front end is already tearing
            # down, so return silently rather than reporting into a dying UI.
            if self._quitting:
                return
            self._sink.on_log([f"[red]driver error: {exc!r}[/red]"])
            self._sink.on_game_over(self._winner_text())

    def _emit_passive_frames(self) -> None:
        """Render any queued passive BSTATE frames (only a broadcast_steps env
        produces them), dwelling STEP_OBSERVE_DELAY whenever the step one-hot
        advances so the human watches the game move through its steps instead
        of it fast-forwarding to the next decision."""
        drain = getattr(self._env, "drain_passive_frames", None)
        if drain is None:
            return
        for frame_obs, ctrs, toks in drain():
            if self._quitting:
                return
            step_idx = _frame_step_idx(frame_obs)
            opp_persp = (frame_obs[_SELF_IS_A_IDX] > 0.5) == self._opp_is_a
            self._sink.on_state(StateUpdate(
                frame_obs, 1, [], human_turn=False, opp_perspective=opp_persp,
                perm_counters=ctrs, perm_token_names=toks, passive=True))
            if step_idx != self._last_step_idx:
                self._last_step_idx = step_idx
                time.sleep(STEP_OBSERVE_DELAY)

    def _replay_prefix(self, obs):
        """Fast-forward through a saved session's action prefix: step each
        recorded action with no observe delays and no per-step state frames,
        batching the narrative into one on_log. Returns (obs, done); on a
        divergence (an action index outside the live menu, or the game ending
        before the prefix is exhausted) reports the failure and returns
        (None, True) so run() exits without entering the interactive loop."""
        env = self._env
        lines = []
        drain = getattr(env, "drain_passive_frames", None)
        done = False
        for i, action in enumerate(self._replay_actions):
            if self._quitting:
                return None, True
            if done or not (0 <= int(action) < env._num_choices):
                self._sink.on_log(lines + [
                    f"replay diverged at action {i}/{len(self._replay_actions)}"
                    " — was this session saved with a different engine build?"])
                self._sink.on_game_over("Replay failed — session not restored.")
                return None, True
            self.action_log.append(int(action))
            obs, reward, terminated, truncated, _info = env.step(int(action))
            if reward:
                self._reward = reward
            done = terminated or truncated
            lines.extend(env.flush_lines())
            if drain is not None:
                for _ in drain():          # discard queued frames; no dwell
                    pass
        lines.append(f"=== Replayed {len(self._replay_actions)} saved actions ===")
        self._sink.on_log(lines)
        self._match = self._human_match_context(obs)
        if done:
            # A completed saved game: render its final board once so the
            # winner banner (posted by run()) lands on a visible position.
            self._sink.on_state(StateUpdate(
                obs.copy(), env._num_choices, [], human_turn=False,
                opp_perspective=(obs[_SELF_IS_A_IDX] > 0.5) == self._opp_is_a,
                perm_counters=getattr(env, "_perm_counters", None),
                perm_token_names=getattr(env, "_perm_token_names", None)))
        return obs, done

    # ----- input / control (called from the front end's UI thread) -----

    def submit(self, idx) -> None:
        """Deliver the human's chosen action index to the blocked loop.

        `submit(None)` signals quit (the loop returns on a None get). Queue-put
        logic moved from GameApp._submit."""
        try:
            self._human_q.put_nowait(idx)
        except Exception:
            self._human_q.put(idx)

    def engage_autopass(self) -> bool:
        """Engage autopass from the current human decision: queue a pass now and
        set the flag so the loop keeps passing every optional window until the
        next UPKEEP step (a mandatory decision — no pass option — stops it early).

        Returns True if a pass was available and engaged, False if there is
        nothing to pass right now (the caller then does no UI bookkeeping)."""
        idx = _pass_index(self._actions)
        if idx is None:                    # nothing to pass right now
            return False
        self._autopass = True
        self.submit(idx)
        return True

    def request_quit(self) -> None:
        """Signal shutdown and unblock the loop.

        Sets _quitting first so a loop blocked deep in a long opponent search —
        when the front end then closes the env and the killed engine pipe raises
        — exits silently on the flag (see run()'s except clauses). Closing the
        env stays the front end's job."""
        self._quitting = True
        try:
            self._human_q.put_nowait(None)
        except Exception:
            pass

    def clock_remaining(self):
        """The opponent's remaining match-clock seconds, or None when no chess
        clock is armed. A passthrough to the session's clock_fn for a front-end
        thinking ticker."""
        return self._clock_fn() if self._clock_fn is not None else None

    # ----- bo3 match banners -----

    def _human_match_context(self, obs):
        """The bo3 match context ({self_wins, opp_wins, ...}) in the human frame,
        mirroring self/opp when this obs is serialized from the opponent's
        perspective (they hold priority)."""
        mctx = decode._decode_match_context(obs[:STATE_SIZE])
        opp_perspective = (obs[_SELF_IS_A_IDX] > 0.5) == self._opp_is_a
        if opp_perspective:
            mctx["self_wins"], mctx["opp_wins"] = mctx["opp_wins"], mctx["self_wins"]
        return mctx

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
        # In bo3 the terminal reward is the DECIDING GAME's result (±1.0), whose
        # sign is also the match winner's; report it with the
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


# ── Session assembly ──────────────────────────────────────────────────────────

@dataclass
class Session:
    """Everything a front end needs to build its board and driver for one
    human-vs-opponent session. Produced by build_session, consumed by either the
    Textual TUI or the Qt GUI."""
    env: object
    opp_act: object
    opp_is_a: bool
    is_model: bool
    opp_label: str
    human_deck: str
    opp_deck: str
    bo3: bool
    clock_fn: object = None
    pace_idle: object = None
    controller: object = None    # the opponent Controller (analysis hooks)
    analysis_cfg: object = None  # AnalysisConfig when the front end enables it
    engine_seed: object = None   # force the engine --seed (saved-session restore)
    record_dir: object = None    # shard-recording directory (front end sets it;
    #                              the pane builds a shard_record.ShardRecorder)


def build_session(binary_path, model_path, human_player=None,
                  human_deck="delver", model_deck="delver", bo3=True,
                  analysis=False, step_pacing=False, engine_seed=None):
    """Assemble the engine env, opponent controller, and seat/clock/pace plumbing
    for one session (front-end-agnostic). Returns a Session.

    `model_path` of None/"scripted" ⇒ rule-based opponent. Any agent spec
    ``opponents.make_controller`` accepts works — a checkpoint path / deck
    shorthand, or a scripted tier ("scripted:hard", "explore", ...) — so both
    front ends share the one agent grammar. `bo3` (default True) plays a
    best-of-three match — with sideboarding between games — in a single engine
    process; pass ``bo3=False`` for a single game. `analysis=True` forces the
    search-capable env even for a non-search opponent, so the analysis window
    can snapshot/replay the session (harmless when unused). `step_pacing=True`
    runs the engine with --broadcast-steps so the driver renders every game
    step (passive BSTATE frames, ~0.2s dwell per step) instead of
    fast-forwarding between decisions — the GUI turns this on. `engine_seed`
    forces the engine --seed on reset (saved-session restore); None lets the
    env pick its own.
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
    if getattr(ctrl, "wants_search_env", False) or analysis:
        from search_env import SearchNarrativeEnv

        class TuiSearchEnv(TuiEnvMixin, SearchNarrativeEnv):
            pass

        env_cls = TuiSearchEnv
    env = env_cls(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                  log_viewer=human_seat, bo3=bo3, broadcast_steps=step_pacing)
    bind_env = getattr(ctrl, "bind_env", None)
    if bind_env is not None:
        bind_env(env)

    def opp_act(obs, num):
        return int(ctrl.choose(obs, num, action_masks=env.action_masks()))

    # Surface a match-clock bank (mcts:/az: ?clock=) in the thinking indicator.
    clock_fn = None
    if isinstance(getattr(ctrl, "stats", None), dict) and ctrl.stats.get("clock_bank"):
        clock_fn = lambda: ctrl.stats.get("clock_remaining")  # noqa: E731

    opp_label = "Model" if is_model else "Scripted"
    return Session(env=env, opp_act=opp_act, opp_is_a=opp_is_a, is_model=is_model,
                   opp_label=opp_label, human_deck=human_deck, opp_deck=model_deck,
                   bo3=bo3, clock_fn=clock_fn,
                   pace_idle=getattr(ctrl, "pace_idle", None),
                   controller=ctrl, engine_seed=engine_seed)
