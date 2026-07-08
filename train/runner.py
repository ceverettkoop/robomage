"""The single observation/decision loop shared by every game-running tool.

There is exactly one game-driving loop in the project, here: :func:`drive_game`.
Everything that plays games against the engine outside the torch training loop
sits on top of it:

- :func:`run_match` — the high-level front door for scripting games. Give each
  seat a deck and an agent spec ("scripted", "explore", "human", "play:...",
  a checkpoint path / deck shorthand, or a prebuilt Controller), pick bo1/bo3
  (bo3 is the default) and an output mode, get a :class:`MatchResult` back.
- :func:`run_games` — the mid-level orchestrator used by the test harness,
  ``train.py observe``, ``fuzz_campaign`` and ``ci_check``: builds a
  NarrativeEnv per game, renders the unified transcript, tallies results.
- :func:`drive_game` — the core loop itself, for callers that manage their own
  env and need per-decision hooks (``analysis.py`` traces, ``bench_engine``).

Callers supply :class:`opponents.Controller` objects (scripted / model /
action-list / interactive / human / auto-pass) or spec strings; the seat with
priority (``obs[_SELF_IS_A_IDX]``) picks which controller acts each decision.

Dependency-light on purpose (numpy + env + decode + the generated enum tables);
torch is only pulled in if a caller passes a model ``Controller`` / checkpoint.
"""

import datetime
import random
import sys

import numpy as np

import decode
from env import (NarrativeEnv, STATE_SIZE, ACTION_CATEGORY_MAX, MAX_ACTIONS,
                 N_CARD_TYPES, BINARY, _IS_ACTIVE_IDX, _SELF_IS_A_IDX,
                 _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE)
from _enums import _CAT_NAMES, _STEP_NAMES


class Decision:
    """Per-decision context handed to drive_game hooks.

    ``menu()`` lazily decodes the legal-action menu (cached), so hooks and
    controllers share a single decode per decision.
    """

    __slots__ = ("env", "obs", "num_choices", "priority_is_a", "controller",
                 "index", "_menu")

    def __init__(self, env, obs, num_choices, priority_is_a, controller, index):
        self.env = env
        self.obs = obs
        self.num_choices = num_choices
        self.priority_is_a = priority_is_a
        self.controller = controller
        self.index = index
        self._menu = None

    def menu(self):
        if self._menu is None:
            self._menu = decode.decode_actions_from_obs(
                self.obs, self.num_choices,
                getattr(self.env, "_action_public", None),
                descriptions=getattr(self.env, "_action_descriptions", None))
        return self._menu


class GameRecord:
    """Result of one drive_game run (one game, or one bo3 match)."""

    __slots__ = ("reward", "decisions", "capped", "actions", "engine_seed")

    def __init__(self, reward, decisions, capped, actions, engine_seed=None):
        self.reward = reward          # cumulative reward, Player A perspective
        self.decisions = decisions
        self.capped = capped          # True if stopped by max_decisions
        self.actions = actions        # every action index fed to env.step
        self.engine_seed = engine_seed

    @property
    def winner(self):
        """'A' / 'B' from the reward sign, None for a draw or a capped game."""
        if self.capped:
            return None
        return "A" if self.reward > 0 else ("B" if self.reward < 0 else None)


def drive_game(env, obs, controller_a, controller_b, *,
               on_query=None, on_action=None, on_narrative=None,
               max_decisions=None, coverage=None):
    """Drive one game (or bo3 match) on an already-reset env to completion.

    The seat holding priority (``obs[_SELF_IS_A_IDX]``) picks the controller each decision;
    pass the same object twice for a single global decision-maker.

    Hooks (all optional):
      - ``on_query(d)`` — before the controller chooses (show the menu, record
        the observation). ``d`` is a :class:`Decision`.
      - ``on_action(d, action)`` — after the choice, before ``env.step``.
      - ``on_narrative(lines)`` — buffered engine narrative, flushed before
        each decision and once at game end (needs a NarrativeEnv-style env).

    ``coverage`` is an optional accumulator with ``record(cats, ids, action)``.
    Returns a :class:`GameRecord`. The caller owns ``env`` (reset and close).
    """
    flush = getattr(env, "flush_lines", None)

    def _narrative():
        if flush is not None:
            lines = flush()
            if on_narrative is not None and lines:
                on_narrative(lines)

    total_reward = 0.0
    decisions = 0
    capped = False
    actions_log = []
    done = False

    while not done:
        _narrative()

        if max_decisions is not None and decisions >= max_decisions:
            capped = True
            break

        priority_is_a = obs[_SELF_IS_A_IDX] > 0.5
        controller = controller_a if priority_is_a else controller_b
        d = Decision(env, obs, env._num_choices, priority_is_a, controller,
                     decisions)
        if on_query is not None:
            on_query(d)

        decoded = d.menu() if getattr(controller, "wants_decoded", False) else None
        action = controller.choose(obs, d.num_choices,
                                   action_masks=env.action_masks(),
                                   decoded_actions=decoded)

        if coverage is not None:
            cats = np.round(obs[STATE_SIZE:STATE_SIZE + d.num_choices]
                            * ACTION_CATEGORY_MAX).astype(int)
            ids = np.rint(obs[STATE_SIZE + MAX_ACTIONS:
                              STATE_SIZE + MAX_ACTIONS + d.num_choices]
                          * N_CARD_TYPES).astype(int)
            coverage.record(cats.tolist(), ids.tolist(), int(action))

        if on_action is not None:
            on_action(d, action)

        actions_log.append(int(action))
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        decisions += 1

    _narrative()
    return GameRecord(total_reward, decisions, capped, actions_log,
                      getattr(env, "last_engine_seed", None))


def _compact_line(decision, player, label, step_name, cats, action):
    """One-line RL-debug summary of a decision (the non-verbose default)."""
    chosen_cat = _CAT_NAMES.get(int(cats[action]), str(cats[action]))
    all_cats = [_CAT_NAMES.get(int(c), str(c)) for c in cats]
    return (f"[{decision:4d}] P{player} ({label}/{player})  {step_name:<14}"
            f"  choices={len(cats)}  available={all_cats}  -> {action} ({chosen_cat})")


def run_games(controller_a, controller_b, *,
              label_a="A", label_b="B",
              binary_path=BINARY, deck_a=None, deck_b=None,
              n_games=1, bo3=False, seed=None, verbose=False,
              transcript=None, out=None,
              battlefield_a=None, battlefield_b=None,
              graveyard_a=None, graveyard_b=None,
              exile_a=None, exile_b=None,
              sideboard_a=None, sideboard_b=None, no_shuffle=False,
              life_a=None, life_b=None,
              max_decisions=None, log_decisions=False, coverage=None,
              on_query=None, on_action=None, on_game_end=None):
    """Run ``n_games`` between two controllers and render the transcript.

    ``controller_a``/``controller_b`` are :class:`opponents.Controller` objects;
    pass the *same* object for both sides for a single global decision-maker
    (the harness's action-list / interactive / auto-pass modes).  ``label_a`` /
    ``label_b`` are the display labels for each side ("Scripted", "Model", ...).

    ``transcript`` selects the output mode (default: ``"verbose"`` when
    ``verbose=True``, else ``"compact"`` — the historical behaviour):
      - ``"verbose"`` — narrative + full board state + legal menu per decision
      - ``"compact"`` — narrative + one summary line per decision
      - ``"narrative"`` — engine narrative and results only (human play)
      - ``"quiet"`` — nothing (programmatic use; draws still dump their log)
    ``out`` is the stream transcripts are written to (default stdout).

    ``coverage`` is an optional :class:`coverage_report.CoverageAccumulator`
    (or anything with the same ``record(cats, ids, action)``); when given, every
    decision's menu categories/card ids and the chosen index are tallied into
    it. Read-only observation — it never alters play or RNG consumption — and
    costs nothing when None.

    ``on_query`` / ``on_action`` are forwarded to :func:`drive_game` (called
    after the runner's own transcript hooks); ``on_game_end(record)`` fires
    after each game with its :class:`GameRecord`.

    Returns ``(wins, losses, draws)`` from Player A's perspective. A game that
    ends with no winner (e.g. the engine's step cap — a stall) counts as a draw
    and its full log is saved to ``draw_<timestamp>.txt``; a game stopped early by
    ``max_decisions`` is reported as incomplete and not counted.
    """
    if transcript is None:
        transcript = "verbose" if verbose else "compact"
    if transcript not in ("verbose", "compact", "narrative", "quiet"):
        raise ValueError(f"unknown transcript mode {transcript!r}")
    is_verbose = transcript == "verbose"
    show_decisions = transcript in ("verbose", "compact")
    stream = out if out is not None else sys.stdout

    wins = losses = draws = incomplete = 0
    unit = "match" if bo3 else "game"

    # Push per-seat deck names to scripted controllers so their doomsday/tron
    # identification uses the shared name rule (duck-typed; decks are fixed
    # across the n_games loop, so once up front is enough).
    for ctrl in (controller_a, controller_b):
        set_names = getattr(ctrl, "set_deck_names", None)
        if set_names is not None:
            set_names(deck_a, deck_b)

    for i in range(n_games):
        env = NarrativeEnv(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                           bo3=bo3, battlefield_a=battlefield_a,
                           battlefield_b=battlefield_b,
                           graveyard_a=graveyard_a, graveyard_b=graveyard_b,
                           exile_a=exile_a, exile_b=exile_b,
                           sideboard_a=sideboard_a, sideboard_b=sideboard_b,
                           life_a=life_a, life_b=life_b,
                           no_shuffle=no_shuffle, log_decisions=log_decisions)
        obs, _ = env.reset(seed=(seed + i) if seed is not None else None)
        # Seed Python's global RNG with the SAME per-game seed passed to the engine.
        # The scripted agent breaks ties on ambiguous OTHER_CHOICE prompts with
        # random.choice (e.g. the "pay {1} or be countered" / Sylvan Library
        # pay-or-return prompts); without this seeding those picks varied between
        # processes, making otherwise-deterministic scripted-vs-scripted games
        # non-reproducible across builds. Tying it to the engine seed makes the full
        # game deterministic.
        if seed is not None:
            random.seed(seed + i)
        # Controllers are reused across the n_games loop; give stateful ones (the
        # scripted EXPLORE tier's per-game novelty set) a fresh-game reset. Duck-typed
        # so index/model/interactive controllers need no stub; calling it twice when
        # one controller drives both seats is harmless.
        for ctrl in (controller_a, controller_b):
            new_game = getattr(ctrl, "new_game", None)
            if new_game is not None:
                new_game()

        # Per-game transcript state. Turn headers mirror the engine's sequential
        # 1-based narrative headers (A=1, B=2, A=3, ...) by decoding Game::turn
        # from the state vector — a local flip counter drifts when a turn yields
        # no decision query (or the same player takes consecutive turns).
        # Mulligan/bottoming decisions happen before turn 1; mark them PREGAME
        # like the engine does instead of showing a turn number.
        turn = 0                      # last turn header shown (1-based)
        pregame_shown = False
        known_hand = {"A": [], "B": []}
        log_lines = []                # buffered so a draw can be dumped to file

        def emit(line):
            log_lines.append(line)
            if transcript != "quiet":
                print(line, file=stream, flush=True)

        def emit_narrative(lines):
            lines = [ln for ln in lines if ln.strip()]
            if not lines:
                return
            block = decode.format_narrative_block(lines) if is_verbose else lines
            for ln in block:
                emit(ln)

        def transcript_on_query(d):
            nonlocal turn, pregame_shown
            if show_decisions:
                obs_ = d.obs
                player = "A" if d.priority_is_a else "B"
                active_is_a = (obs_[_IS_ACTIVE_IDX] > 0.5) == d.priority_is_a
                cats = np.round(obs_[STATE_SIZE:STATE_SIZE + d.num_choices]
                                * ACTION_CATEGORY_MAX).astype(int)
                is_pregame = decode.is_mulligan(cats) or decode.is_bottom(cats)

                known_hand[player] = decode.decode_hand(obs_)

                if is_pregame:
                    if not pregame_shown:
                        emit("--- Pregame ---")
                        pregame_shown = True
                elif (cur_turn := decode.decode_turn(obs_)) != turn:
                    turn = cur_turn
                    emit(f"--- Turn {turn} (Player {'A' if active_is_a else 'B'}) ---")
                    emit(f"  PA: {', '.join(known_hand['A']) or '(empty)'}")
                    emit(f"  PB: {', '.join(known_hand['B']) or '(empty)'}")
            if on_query is not None:
                on_query(d)

        def transcript_on_action(d, action):
            if show_decisions:
                obs_ = d.obs
                player = "A" if d.priority_is_a else "B"
                label = label_a if d.priority_is_a else label_b
                if is_verbose:
                    gs = decode.decode_game_state(
                        obs_[:STATE_SIZE],
                        perm_counters=getattr(d.env, "_perm_counters", None),
                        perm_token_names=getattr(d.env, "_perm_token_names", None))
                    for ln in decode.format_decision_block(d.index + 1, gs, d.menu()):
                        emit(ln)
                    emit(decode.format_chosen_action(f"{label}/{player}", action,
                                                     d.menu()))
                else:
                    cats = np.round(obs_[STATE_SIZE:STATE_SIZE + d.num_choices]
                                    * ACTION_CATEGORY_MAX).astype(int)
                    step_name = _STEP_NAMES[int(np.argmax(
                        obs_[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]))]
                    emit(_compact_line(d.index, player, label, step_name, cats,
                                       action))
            if on_action is not None:
                on_action(d, action)

        if n_games > 1:
            emit(f"--- {unit.capitalize()} {i + 1}/{n_games} ---")

        try:
            record = drive_game(env, obs, controller_a, controller_b,
                                on_query=transcript_on_query,
                                on_action=transcript_on_action,
                                on_narrative=emit_narrative,
                                max_decisions=max_decisions,
                                coverage=coverage)
        finally:
            env.close()

        if on_game_end is not None:
            on_game_end(record)

        if record.capped:
            incomplete += 1
            if transcript != "quiet":
                print(f"\n=== stopped after {record.decisions} decisions "
                      f"(max-decisions cap; no winner) ===", file=stream, flush=True)
        elif record.reward > 0:
            wins += 1
            result = f"{label_a}/A wins"
        elif record.reward < 0:
            losses += 1
            result = f"{label_b}/B wins"
        else:
            draws += 1
            result = "DRAW (should not occur)"
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_path = f"draw_{stamp}.txt"
            with open(log_path, "w") as f:
                f.write("\n".join(log_lines) + "\n")
            # A draw is a finding — always announce it, even in quiet mode.
            print(f"\n=== DRAW (should not occur) — full log saved to {log_path} ===",
                  file=stream, flush=True)

        if not record.capped and transcript != "quiet":
            if n_games > 1:
                print(f"  {unit} {i + 1:2d}/{n_games}: {result}  "
                      f"(W:{wins} L:{losses} D:{draws})", file=stream, flush=True)
            else:
                print(f"\n=== {result} ===", file=stream, flush=True)

    if n_games > 1 and transcript != "quiet":
        total = wins + losses + draws
        win_pct = 100 * wins / total if total else 0
        extra = f", {incomplete} incomplete" if incomplete else ""
        print(f"\n{wins}W / {losses}L / {draws}D over {n_games} {unit}"
              f"{'es' if bo3 else 's'} ({win_pct:.1f}% Player A win rate){extra}",
              file=stream, flush=True)

    return wins, losses, draws


class MatchResult:
    """Aggregate result of a :func:`run_match` call (Player A's perspective)."""

    def __init__(self, wins, losses, draws, records, label_a, label_b):
        self.wins = wins
        self.losses = losses
        self.draws = draws
        self.records = records        # list[GameRecord], one per game/match
        self.label_a = label_a
        self.label_b = label_b

    @property
    def games(self):
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self):
        return self.wins / self.games if self.games else 0.0

    def __repr__(self):
        return (f"MatchResult({self.label_a}/A {self.wins}W-{self.losses}L-"
                f"{self.draws}D vs {self.label_b}/B, "
                f"win_rate={self.win_rate:.2f})")


def run_match(agent_a="scripted", agent_b="scripted", *,
              deck_a=None, deck_b=None,
              games=1, bo3=True, seed=1, transcript="compact", out=None,
              binary_path=BINARY, deterministic_models=True,
              checkpoint_resolver=None, max_decisions=None,
              **run_games_kwargs):
    """Run a matchup between two agents — the front door for scripting games.

    ``agent_a`` / ``agent_b`` are agent specs (see :func:`opponents.make_controller`):
    scripted tiers ("scripted"/"hard", "easy"/"greedy", "random", "explore",
    "explore:patient"), "human" (interactive CLI seat), "auto",
    "play:<specs>", "actions:<indices>", a checkpoint path or deck shorthand
    ("league/bug" → its newest checkpoint), or a prebuilt Controller.

    Defaults are chosen for scripting: **bo3 matches**, a fixed ``seed`` (game
    ``i`` uses ``seed + i``; pass ``seed=None`` for random), and a compact
    transcript (pass ``transcript="quiet"`` for silent programmatic runs,
    "verbose" for full per-decision state, "narrative" for human play).

    Extra keyword arguments (``battlefield_a``, ``no_shuffle``, ``life_b``,
    ``log_decisions``, ``coverage``, hooks, ...) pass through to
    :func:`run_games`. Returns a :class:`MatchResult`.
    """
    from opponents import make_controller

    ctrl_a = make_controller(agent_a, checkpoint_resolver=checkpoint_resolver,
                             deterministic=deterministic_models)
    ctrl_b = make_controller(agent_b, checkpoint_resolver=checkpoint_resolver,
                             deterministic=deterministic_models)
    label_a = getattr(ctrl_a, "label", None) or "A"
    label_b = getattr(ctrl_b, "label", None) or "B"

    records = []
    user_on_game_end = run_games_kwargs.pop("on_game_end", None)

    def collect(record):
        records.append(record)
        if user_on_game_end is not None:
            user_on_game_end(record)

    wins, losses, draws = run_games(
        ctrl_a, ctrl_b, label_a=label_a, label_b=label_b,
        binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
        n_games=games, bo3=bo3, seed=seed, transcript=transcript, out=out,
        max_decisions=max_decisions, on_game_end=collect,
        **run_games_kwargs)
    return MatchResult(wins, losses, draws, records, label_a, label_b)
