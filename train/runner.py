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
from typing import NamedTuple

import numpy as np

import decode
from env import (NarrativeEnv, STATE_SIZE, ACTION_CATEGORY_MAX, MAX_ACTIONS,
                 N_CARD_TYPES, BINARY, _IS_ACTIVE_IDX, _SELF_IS_A_IDX,
                 _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE, _EXTRAS_PLAYS_FIRST)
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


class GameOutcome(NamedTuple):
    """One game's result inside a bo3 match."""
    winner: str | None      # "A" / "B", None if undecided
    a_on_play: bool         # Player A was the starting player of THIS game


class GameRecord:
    """Result of one drive_game run (one game, or one bo3 match)."""

    __slots__ = ("reward", "decisions", "capped", "actions", "engine_seed",
                 "game_results")

    def __init__(self, reward, decisions, capped, actions, engine_seed=None,
                 game_results=None):
        self.reward = reward          # cumulative reward, Player A perspective
        self.decisions = decisions
        self.capped = capped          # True if stopped by max_decisions
        self.actions = actions        # every action index fed to env.step
        self.engine_seed = engine_seed
        # Per-GAME outcomes inside this record, in play order, as GameOutcome
        # (winner, a_on_play). Index 0 is the pre-board game; everything after it
        # was played on a sideboarded deck, which is what makes sideboarding
        # measurable. `a_on_play` is recorded because a bo3's pre/post comparison
        # is otherwise CONFOUNDED: the loser of each game chooses to play first, so
        # whoever wins game 1 is usually on the draw in game 2. Populated in bo3
        # only — the engine emits the per-game marker this is built from under
        # --bo3, so a bo1 run leaves it EMPTY (its single result is already
        # `reward`/`winner`).
        self.game_results = game_results if game_results is not None else []

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
    game_results = []
    # Whether Player A is the starting player of the game currently being played,
    # refreshed every decision from the observation's self_plays_first flag. It is
    # perspective-relative, so un-flip it by the viewer's seat. During a bo3
    # sideboard phase the engine reports the UPCOMING game's starting player,
    # which is the game this value will be filed under — so the same read is
    # correct in-game and between games.
    a_on_play = True
    done = False

    while not done:
        _narrative()

        if max_decisions is not None and decisions >= max_decisions:
            capped = True
            break

        priority_is_a = obs[_SELF_IS_A_IDX] > 0.5
        self_plays_first = obs[_EXTRAS_PLAYS_FIRST] > 0.5
        a_on_play = self_plays_first if priority_is_a else not self_plays_first
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
            # Obs action-metadata block is five MAX_ACTIONS-wide sub-blocks
            # (cats|ids|ctrl|zone|refs — entity-slot refs appended last, see
            # decode.action_slot_refs); coverage reads only the first two,
            # whose offsets are unchanged.
            cats = np.round(obs[STATE_SIZE:STATE_SIZE + d.num_choices]
                            * ACTION_CATEGORY_MAX).astype(int)
            ids = np.rint(obs[STATE_SIZE + MAX_ACTIONS:
                              STATE_SIZE + MAX_ACTIONS + d.num_choices]
                          * N_CARD_TYPES).astype(int)
            coverage.record(cats.tolist(), ids.tolist(), int(action))

        if on_action is not None:
            on_action(d, action)

        actions_log.append(int(action))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        # Per-game winner (bo3): the engine flags the step a game ended on, and
        # this env family reports the raw Player-A-perspective reward, so the sign
        # names the winner. On the last game that reward carries the ±0.3 game
        # result AND the ±1.0 match result, but they always share a sign (the
        # match winner won the final game), so it stays correct.
        # NOTE: valid because run_games drives a NarrativeEnv/SearchNarrativeEnv
        # (RoboMageEnv subclasses, one engine step per env step). The TRAINING
        # wrappers aggregate several inner steps into one info and would break
        # this — they never reach drive_game.
        if info.get("game_result"):
            winner = "A" if reward > 0 else ("B" if reward < 0 else None)
            game_results.append(GameOutcome(winner, a_on_play))
        done = terminated or truncated
        decisions += 1

    _narrative()
    return GameRecord(total_reward, decisions, capped, actions_log,
                      getattr(env, "last_engine_seed", None), game_results)


def tally_per_game(records, flip=False):
    """Tally per-game-index results across GameRecords.

    Returns ``{game index: {"wins", "played", "play_wins", "play_n",
    "draw_wins", "draw_n"}}``, where ``play_*`` / ``draw_*`` restrict to the games
    the subject was on the play / on the draw. Index 0 is the pre-board game; 1+
    were played after sideboarding. Empty for bo1 records, which carry no per-game
    breakdown.

    Tallies are Player A's unless ``flip`` is set, which scores them for Player B
    instead (used when the subject alternates seats between matches).
    """
    per_game = {}
    for rec in records:
        for gi, out in enumerate(rec.game_results):
            cell = per_game.setdefault(gi, {"wins": 0, "played": 0, "play_wins": 0,
                                            "play_n": 0, "draw_wins": 0, "draw_n": 0})
            subject = "B" if flip else "A"
            won = out.winner == subject
            on_play = out.a_on_play if not flip else not out.a_on_play
            cell["played"] += 1
            cell["wins"] += int(won)
            key = "play" if on_play else "draw"
            cell[f"{key}_n"] += 1
            cell[f"{key}_wins"] += int(won)
    return per_game


def merge_per_game(dest, src):
    """Fold one tally_per_game result into another, in place."""
    for gi, cell in src.items():
        into = dest.setdefault(gi, {"wins": 0, "played": 0, "play_wins": 0,
                                    "play_n": 0, "draw_wins": 0, "draw_n": 0})
        for k, v in cell.items():
            into[k] += v
    return dest


def _pct(w, n):
    return f"{w}/{n} ({100 * w / n:.1f}%)" if n else f"{w}/{n} (n/a)"


def format_per_game_split(per_game, subject="Player A"):
    """Per-game-index and play/draw-controlled summary lines, or [] if no split.

    Read this carefully: a raw pre-board vs post-board comparison is CONFOUNDED by
    the bo3 play/draw rule. Game 1's starting player is fixed, but from game 2 the
    LOSER of the previous game chooses to play first — so whoever wins game 1 is
    usually on the draw afterwards. Measured on mirror matchups with agents that
    never sideboard, that alone moves game 1 (~63-75%) to post-board (~46%). Any
    honest read of a sideboarding change therefore has to compare like with like,
    which is what the second line does: post-board win rate split by whether the
    subject was on the play. Returns [] for bo1, or when no match reached game 2.
    """
    if not per_game or max(per_game) == 0:
        return []
    per_idx = "  ".join(
        f"g{gi + 1} {_pct(c['wins'], c['played'])}"
        for gi, c in sorted(per_game.items()))
    post = [c for gi, c in per_game.items() if gi > 0]
    pw, pn = sum(c["play_wins"] for c in post), sum(c["play_n"] for c in post)
    dw, dn = sum(c["draw_wins"] for c in post), sum(c["draw_n"] for c in post)
    return [
        f"per-game ({subject}): {per_idx}",
        f"post-board by seat ({subject}): on the play {_pct(pw, pn)}"
        f" | on the draw {_pct(dw, dn)}"
        "   [g1 vs g2-3 alone is confounded by the play/draw rule]",
    ]


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
    # Every record, so the bo3 per-game-index split can be tallied at the end. The
    # aggregate match W/L cannot distinguish the pre-board game from the games
    # played after sideboarding, which is exactly the comparison that matters.
    all_records = []

    # Push per-seat deck names to scripted controllers so their doomsday/tron
    # identification uses the shared name rule (duck-typed; decks are fixed
    # across the n_games loop, so once up front is enough).
    for ctrl in (controller_a, controller_b):
        set_names = getattr(ctrl, "set_deck_names", None)
        if set_names is not None:
            set_names(deck_a, deck_b)

    # A search controller (MCTS) needs the engine's --search-server protocol
    # and a handle on the env to issue snapshot/restore/determinize commands.
    # Advertised by attribute (like wants_decoded) so plain controllers need no
    # stub; the env class swap is transparent to everything else here.
    needs_search_env = any(getattr(c, "wants_search_env", False)
                           for c in (controller_a, controller_b))
    if needs_search_env:
        from search_env import SearchNarrativeEnv
        env_cls = SearchNarrativeEnv
    else:
        env_cls = NarrativeEnv

    for i in range(n_games):
        env = env_cls(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                      bo3=bo3, battlefield_a=battlefield_a,
                      battlefield_b=battlefield_b,
                      graveyard_a=graveyard_a, graveyard_b=graveyard_b,
                      exile_a=exile_a, exile_b=exile_b,
                      sideboard_a=sideboard_a, sideboard_b=sideboard_b,
                      life_a=life_a, life_b=life_b,
                      no_shuffle=no_shuffle, log_decisions=log_decisions)
        # Hand search controllers the live env (per game — a fresh env/process
        # is created for each one). Duck-typed like new_game below.
        for ctrl in (controller_a, controller_b):
            bind_env = getattr(ctrl, "bind_env", None)
            if bind_env is not None:
                bind_env(env)
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

        all_records.append(record)

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
        for line in format_per_game_split(tally_per_game(all_records)):
            print(f"  {line}", file=stream, flush=True)

    # Lightweight per-side search-effort summary for MCTS/AZ seats — lets the
    # user see the effort actually spent (mean sims/decision), which is the
    # payoff of a wall-clock --think-time budget. Skipped for non-search seats
    # (no .stats["searched"]) and deduped when one object drives both seats.
    if transcript != "quiet":
        seen = set()
        for lbl, ctrl in ((label_a, controller_a), (label_b, controller_b)):
            if id(ctrl) in seen:
                continue
            seen.add(id(ctrl))
            st = getattr(ctrl, "stats", None)
            if not isinstance(st, dict) or not st.get("searched"):
                continue
            searched = st["searched"]
            mean_sims = st.get("sims", 0) / max(1, searched)
            clock_note = ""
            if st.get("clock_bank"):
                clock_note = (f", clock {st.get('clock_remaining', 0.0):.0f}/"
                              f"{st['clock_bank']:.0f}s left, "
                              f"{st.get('early_stops', 0)} early stops")
            print(f"  [{lbl}] search effort: {searched} roots "
                  f"({st.get('sb_searched', 0)} sideboard), {st.get('sims', 0)} "
                  f"sims total, {mean_sims:.0f} mean sims/decision, "
                  f"{st.get('followed', 0)} followed, {st.get('trivial', 0)} "
                  f"trivial, {st.get('fallback', 0)} fallback{clock_note}",
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
