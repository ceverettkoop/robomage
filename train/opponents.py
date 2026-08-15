"""Opponent controllers and the factory/pool that builds them.

This unifies the three historical opponent code paths (ModelVsScriptedEnv's
scripted opponent, SelfPlayEnv's scripted fallback, and observe's per-side
controller) behind a single ``Controller`` interface, so scripted agents and PPO
checkpoints are interchangeable and can be mixed per episode.

sb3/torch are imported lazily (only when a model controller is actually built) so
importing this module — and the env/scripted_agent chain — never pulls torch when
only scripted opponents are used.
"""

import glob as _glob
import math
import os
import re
import time
from typing import Callable, Optional, Protocol, Sequence, Union

import numpy as np

import archetypes
from env import (MAX_ACTIONS, STATE_SIZE, _SELF_IS_A_IDX, _IS_SIDEBOARD_IDX,
                 _CUR_TURN_IDX, _MATCH_CTX_START)
from _enums import (CAT_PASS_PRIORITY, CAT_SELECT_ATTACKER,
                    CAT_CONFIRM_ATTACKERS, CAT_SELECT_BLOCKER,
                    CAT_CONFIRM_BLOCKERS)
from cli_spec import (DEFAULT_SB_SIMS, DEFAULT_SB_WORLDS, DEFAULT_SB_MAX_DEPTH,
                      DEFAULT_SB_ROLLOUT_TURNS, DEFAULT_SB_PERSIST)
from scripted_agent import ScriptedAgent, make_agent

# Bare suffixes (and the "scripted" prefix) that denote a scripted controller.
# Keep in sync with scripted_agent._PRESETS.
_SCRIPTED_SUFFIXES = frozenset({"scripted", "random", "greedy", "easy", "hard",
                                "heuristic", "explore", "fuzz",
                                "explore:patient", "patient"})

# The single generalist checkpoint stem. There is now ONE PPO model that pilots
# ANY deck; the deck it plays is always a separate explicit parameter (never
# inferred from the filename). Its checkpoints are 'gen__final.zip' plus periodic
# 'gen__v{steps}.zip' snapshots. 'gen' is reserved — a roster/league deck may not
# be named 'gen' (see ``assert_not_reserved_deck``).
GEN_STEM = "gen"

# Determinized worlds per search when a spec names no worlds= knob (the mcts:/az:
# grammar default). Also the cap an interactive front end's auto procs count is
# clamped to — see default_search_procs.
DEFAULT_SEARCH_WORLDS = 4

# Checkpoint stem prefix of an ARCHETYPE EXPLOITER: a dedicated run whose learner
# pilots one archetype's decks against the frozen generalist, saved under its own
# stem ('exp_burn__v{steps}.zip' / 'exp_burn__final.zip') so it never touches the
# generalist's files, and injected back into the league pool as an opponent (see
# LeaguePool.refresh tier 3 and train.py's `exploiter` subcommand). Like 'gen',
# the prefix is reserved — a deck may not be named 'exp_*'.
EXPLOITER_PREFIX = "exp_"

# Checkpoint format version. v3 is the single-generalist naming ('gen__final.zip' /
# 'gen__v{steps}.zip'); the double underscore is the format marker that separates
# these from the legacy '{a}_{b}_*.zip' matchup files. Bumping the observation
# layout / net capacity already invalidates old nets, so this is a clean break
# with no back-compat shim — files that don't parse as gen snapshots are simply
# skipped (see ``gen_snapshots``).
CHECKPOINT_FORMAT_VERSION = 3

_SNAPSHOT_RE = re.compile(r"^(?P<deck>.+)__v(?P<steps>\d+)\.zip$")


def exploiter_stem(archetype: str) -> str:
    """Checkpoint stem of an archetype's exploiter ('burn' -> 'exp_burn')."""
    return f"{EXPLOITER_PREFIX}{archetype}"


def is_exploiter_path(path: Optional[str]) -> bool:
    """True when a checkpoint path/label belongs to an exploiter stem."""
    return bool(path) and os.path.basename(path).startswith(EXPLOITER_PREFIX)


def assert_not_reserved_deck(deck: Optional[str]) -> None:
    """Refuse a roster/league deck whose name collides with a reserved model stem.

    Reserved are ``gen`` (the one generalist) and anything starting with ``exp_``
    (the archetype exploiters): such a deck's checkpoints ('gen__v*.zip',
    'exp_*__v*.zip') would be indistinguishable from a model's own snapshots.
    Fail loudly rather than silently corrupt the pool."""
    if not deck:
        return
    stem = os.path.basename(deck).strip().lower()
    if stem == GEN_STEM:
        raise ValueError(
            f"deck name {deck!r} collides with the reserved generalist model stem "
            f"'{GEN_STEM}'. Rename the deck — 'gen' is reserved for the one "
            f"generalist checkpoint (gen__final.zip / gen__v*.zip).")
    if stem.startswith(EXPLOITER_PREFIX):
        raise ValueError(
            f"deck name {deck!r} collides with the reserved exploiter stem prefix "
            f"'{EXPLOITER_PREFIX}'. Rename the deck — '{EXPLOITER_PREFIX}*' is "
            f"reserved for archetype exploiter checkpoints "
            f"({EXPLOITER_PREFIX}<archetype>__final.zip / __v*.zip).")


def stem_final_path(checkpoint_dir: str, stem: str) -> str:
    """Path to ``<stem>__final.zip`` (whether or not it exists yet)."""
    return os.path.join(checkpoint_dir, f"{stem}__final.zip")


def stem_snapshots(checkpoint_dir: Optional[str], stem: str) -> list[str]:
    """All frozen snapshots of one checkpoint stem ('<stem>__v{steps}.zip' + '__final').

    Returns absolute paths sorted by training step (the ``<stem>__final`` snapshot,
    if present, last). Files that don't parse as snapshots of exactly this stem are
    skipped. Empty when the dir is unknown or nothing matches. The shared scan
    behind both the generalist (``gen_snapshots``) and the archetype exploiters
    (``exploiter_snapshots``).
    """
    if not checkpoint_dir:
        return []
    versioned: list[tuple[int, str]] = []
    for path in _glob.glob(os.path.join(checkpoint_dir, f"{stem}__v*.zip")):
        m = _SNAPSHOT_RE.match(os.path.basename(path))
        if m and m.group("deck") == stem:
            versioned.append((int(m.group("steps")), path))
    versioned.sort()
    out = [p for _, p in versioned]
    final = stem_final_path(checkpoint_dir, stem)
    if os.path.exists(final):
        out.append(final)
    return out


def gen_snapshots(checkpoint_dir: Optional[str]) -> list[str]:
    """All frozen snapshots of the one generalist ('gen__v{steps}.zip' + 'gen__final.zip')."""
    return stem_snapshots(checkpoint_dir, GEN_STEM)


def exploiter_snapshots(checkpoint_dir: Optional[str]) -> dict[str, list[str]]:
    """``{archetype: [snapshots oldest..newest, '__final' last]}`` per exploiter.

    Only archetypes that actually have exploiter checkpoints appear, so an
    untrained archetype contributes nothing to the pool."""
    out: dict[str, list[str]] = {}
    for arch in archetypes.ARCHETYPES:
        snaps = stem_snapshots(checkpoint_dir, exploiter_stem(arch))
        if snaps:
            out[arch] = snaps
    return out


def latest_gen_snapshot(checkpoint_dir: Optional[str]) -> Optional[str]:
    """Newest generalist snapshot (highest version, or ``gen__final``); None if none."""
    snaps = gen_snapshots(checkpoint_dir)
    return snaps[-1] if snaps else None


def gen_final_path(checkpoint_dir: str) -> str:
    """Path to the generalist's ``gen__final.zip`` (whether or not it exists yet)."""
    return stem_final_path(checkpoint_dir, GEN_STEM)


_DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "checkpoints")


def resolve_checkpoint(path: Optional[str],
                       checkpoint_dir: str = _DEFAULT_CHECKPOINT_DIR) -> Optional[str]:
    """Resolve a model spec to a full checkpoint path (the shared resolver).

    There is ONE generalist model, so this accepts only:
      - ``None`` → ``None``
      - a full path (returned as-is if it exists)
      - the reserved stem ``'gen'`` → the newest ``gen`` snapshot
        (``gen__final.zip``, else the newest ``gen__v*.zip``; the ``gen__final``
        path is returned even if it does not exist yet so callers can test it)
      - an explicit ``<name>.zip`` / ``<name>_final.zip`` sitting in the checkpoint dir

    A bare token that is none of the above used to be a per-deck shorthand
    (``delver`` → ``delver__final.zip``). That naming is gone: the model is the
    single generalist ``'gen'`` and the deck it pilots is a SEPARATE explicit
    parameter. Such a token raises a clear error rather than resolving.
    """
    if path is None:
        return None
    if os.path.exists(path):
        return path
    if path.strip().lower() == GEN_STEM:
        return latest_gen_snapshot(checkpoint_dir) or gen_final_path(checkpoint_dir)
    for candidate in (os.path.join(checkpoint_dir, f"{path}.zip"),
                      os.path.join(checkpoint_dir, f"{path}_final.zip")):
        if os.path.exists(candidate):
            return candidate
    raise ValueError(
        f"cannot resolve model spec {path!r}. Models are now ONE generalist with "
        f"stem '{GEN_STEM}' — pass '{GEN_STEM}' (or an explicit .zip path) as the "
        f"model, and give the deck it pilots as a SEPARATE parameter (baseline's "
        f"--deck, observe's --deck/--opponent, analysis's --deck-a/--deck-b, "
        f"play's --model-deck).")

# Pool token standing for a random generalist snapshot (the opponent deck it
# pilots is chosen independently by the pool/episode, no longer by the model's
# filename). OpponentPool expands it into the generalist snapshots ('gen__v*.zip'
# / 'gen__final.zip'), the same set SelfPlayEnv samples.
MATCHUP_MODEL_TOKEN = "random-model"


def is_scripted_spec(spec: str) -> bool:
    """True if ``spec`` names a scripted agent rather than a checkpoint path."""
    s = (spec or "").strip().lower()
    return s.startswith("scripted") or s in _SCRIPTED_SUFFIXES


def matchup_checkpoints(model_deck: Optional[str], opp_deck: Optional[str],
                        checkpoint_dir: Optional[str]) -> list[str]:
    """Generalist snapshots usable as an opponent for the current matchup.

    The one generalist pilots any deck, so ANY of its snapshots is a valid
    opponent — the deck it pilots (``opp_deck``) is set by the pool/env, not by
    the checkpoint. ``model_deck``/``opp_deck`` are unused (kept for signature
    compatibility). Returns absolute paths sorted by training step; empty when the
    dir is unknown or nothing matches.
    """
    return gen_snapshots(checkpoint_dir)


class Controller(Protocol):
    """Uniform decision interface for any opponent.

    ``decoded_actions`` is the decision's decoded action menu
    (``decode.decode_actions_from_obs`` output); the runner supplies it only
    when a controller advertises ``wants_decoded = True`` (e.g. PlayController),
    so index/model/scripted controllers ignore it.
    """

    def choose(self, obs: np.ndarray, num_choices: int,
               action_masks: Optional[np.ndarray] = None,
               decoded_actions: Optional[list] = None) -> int: ...


class ScriptedController:
    """Wraps a ScriptedAgent."""

    def __init__(self, agent: ScriptedAgent, label: str = "scripted"):
        self._agent = agent
        self.label = label

    def new_game(self) -> None:
        """Reset the agent's per-game state (EXPLORE novelty set) at game start."""
        self._agent.new_game()

    def set_deck_names(self, deck_a, deck_b) -> None:
        """Push per-seat deck names to the agent (unified deck identification)."""
        self._agent.set_deck_names(deck_a, deck_b)

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        return self._agent.act(obs, num_choices)


class ModelController:
    """Wraps a loaded MaskablePPO checkpoint."""

    def __init__(self, model, label: str = "model", deterministic: bool = False):
        self._model = model
        self.label = label
        self._deterministic = deterministic
        self._mask = np.zeros(MAX_ACTIONS, dtype=bool)
        # This controller only ever runs inference. Torch's per-construction
        # parameter validation can spuriously reject a marginal fp32 softmax
        # (MaskableCategorical re-validates its cached pre-mask probs against
        # Simplex, tolerance ~1e-6) and crash a game mid-decision — disable it.
        import torch
        torch.distributions.Distribution.set_default_validate_args(False)

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        if action_masks is None:
            self._mask[:] = False
            self._mask[:num_choices] = True
            action_masks = self._mask
        action, _ = self._model.predict(obs, action_masks=action_masks,
                                        deterministic=self._deterministic)
        return int(action)


class MatchClock:
    """Chess-clock time manager for a search controller: one wall-clock bank
    spans a whole match (bo3 or bo1), and each searched decision draws a
    variable ``(min_s, max_s)`` deadline pair from it.

    The allocation is ``remaining / C``, scaled by a difficulty multiplier
    derived from the root priors (entropy / top-2 gap / menu width), and
    clamped to ``[t_min, min(t_max, remaining/4)]``. C is the estimated
    decisions left on this clock. The estimate starts from a fixed prior — a
    full bo3 match is assumed to run ``_MATCH_DEC_EST`` (300) decisions for
    one seat, ~``_GAME_DEC_EST`` (120) per game over ~2.5 expected games —
    and is REVISED by the match's observed pace, fed one ``note_decision()``
    per decision faced: completed games revise the per-game estimate (a match
    that reached game 3 after only 90 decisions now expects short games ahead,
    so the tail estimate shrinks and per-decision spend grows), and the
    current game's observed decisions-per-turn rate revises the remainder of
    this game (250 decisions deep in game 1 projects a longer-than-prior
    match, so per-decision spend shrinks). ``t_min`` behaves like a Fischer
    increment: it is granted even on an empty bank, so an exhausted clock
    degrades to fast play rather than stalling. The min deadline arms the
    in-search stability early-stop (see ``mcts.run_search``), which is what
    actually lets obvious decisions finish cheap and contested ones run long.

    Takes only decoded scalars, no obs vector — unit-testable without an engine.
    """

    _TURN_END_EST = 24     # 0-based player-turn by which a game typically ends
    _MIN_TURNS_LEFT = 4    # horizon floor once past _TURN_END_EST
    _MATCH_DEC_EST = 300.0                       # prior: decisions in a full bo3 (one seat)
    _EXP_GAMES = 2.5                             # expected games in a bo3
    _GAME_DEC_EST = _MATCH_DEC_EST / _EXP_GAMES  # prior per-game decisions (120)
    _SB_DEC_EST = 6        # sideboard decisions ahead of each future game
    _PRIOR_GAME_W = 1.0    # completed-game revision: prior worth this many observed games
    _RATE_PRIOR_W = 8.0    # in-game rate revision: prior worth this many observed turns
    _HORIZON_LO, _HORIZON_HI = 8, 400
    _M_LO, _M_HI = 0.35, 1.75  # difficulty-multiplier clamp

    def __init__(self, bank_s: float, *, t_min: float = 0.5,
                 t_max: float = 60.0, sb_t_max: float = 15.0):
        self.bank = float(bank_s)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.sb_t_max = float(sb_t_max)
        self.remaining = self.bank
        self.decisions = 0                   # decisions faced this match (all kinds)
        self._cur_game: Optional[int] = None
        self._game_start_dec: list[int] = []  # decision index at each game's first sight

    def reset(self) -> None:
        self.remaining = self.bank
        self.decisions = 0
        self._cur_game = None
        self._game_start_dec = []

    def debit(self, elapsed_s: float) -> None:
        self.remaining = max(0.0, self.remaining - max(0.0, float(elapsed_s)))

    def note_decision(self, game_number: int) -> None:
        """Record one decision faced on this clock (searched, fallback or
        trivial — they all advance the match) and mark the game boundary when
        ``game_number`` changes. These counters drive the horizon revision."""
        if game_number != self._cur_game:
            self._cur_game = game_number
            self._game_start_dec.append(self.decisions)
        self.decisions += 1

    def _horizon(self, turn: int, game_number: int, self_wins: int,
                 opp_wins: int) -> float:
        """Estimated decisions left on this clock (one seat): the fixed match
        prior revised by the observed per-game length and in-game pace."""
        cur_start = self._game_start_dec[-1] if self._game_start_dec else 0
        in_cur = max(0, self.decisions - cur_start)
        # Completed games revise the per-game estimate: a match that reached
        # game N cheaply expects its remaining games short too (and vice versa).
        completed = max(0, game_number - 1)
        per_game = self._GAME_DEC_EST
        if completed > 0:
            observed = cur_start / completed
            per_game = ((self._PRIOR_GAME_W * self._GAME_DEC_EST
                         + completed * observed)
                        / (self._PRIOR_GAME_W + completed))
        # Rest of the CURRENT game: blend the prior decisions-per-turn rate
        # with the rate actually observed so far this game, over the turns
        # left. A game running hot (many decisions early) projects long and
        # shrinks per-decision spend instead of letting it drain the bank.
        turn = max(0, turn)
        turns_left = max(self._TURN_END_EST - turn, self._MIN_TURNS_LEFT)
        prior_rate = per_game / self._TURN_END_EST
        rate = ((self._RATE_PRIOR_W * prior_rate + in_cur)
                / (self._RATE_PRIOR_W + turn))
        in_game = rate * turns_left
        # Expected FUTURE games: bo1 (game_number 0) and a 1-1 score have none;
        # 0-0 means game 2 is certain and game 3 ~50%; a 1-0 lead ~50% of one.
        if game_number <= 0 or (self_wins >= 1 and opp_wins >= 1):
            extra_games = 0.0
        elif self_wins == 0 and opp_wins == 0:
            extra_games = 1.5
        else:
            extra_games = 0.5
        horizon = in_game + extra_games * (per_game + self._SB_DEC_EST)
        return min(max(horizon, self._HORIZON_LO), self._HORIZON_HI)

    def _difficulty(self, priors, num_choices: int) -> float:
        """Multiplier from the root policy: contested roots (flat priors, wide
        menus) earn more time than obvious ones (one dominant prior)."""
        if priors is None or num_choices < 2:
            return 1.0
        p = np.asarray(priors, dtype=np.float64)[:num_choices]
        total = p.sum()
        if not np.isfinite(total) or total <= 0.0:
            return 1.0
        p = p / total
        top2 = np.sort(p)[-2:]
        gap = float(top2[1] - top2[0])
        with np.errstate(divide="ignore", invalid="ignore"):
            h = -np.where(p > 0.0, p * np.log(p), 0.0).sum()
        h_norm = float(h / math.log(num_choices))
        width = min(num_choices - 2, 14) / 14.0
        m = 0.5 + 1.25 * h_norm - 1.0 * gap + 0.25 * width
        return min(max(m, self._M_LO), self._M_HI)

    def allocate(self, *, turn: int, game_number: int, self_wins: int,
                 opp_wins: int, is_sideboard: bool, priors,
                 num_choices: int) -> tuple[float, float]:
        """Return the ``(min_s, max_s)`` deadline pair for one decision."""
        c = self._horizon(turn, game_number, self_wins, opp_wins)
        base = self.remaining / c
        m = self._difficulty(priors, num_choices)
        t_cap = self.sb_t_max if is_sideboard else self.t_max
        t_hi = min(base * m, t_cap, self.remaining / 4.0)
        t_hi = max(t_hi, self.t_min)  # Fischer floor: granted even when empty
        t_lo = min(max(0.25 * t_hi, self.t_min), t_hi)
        return t_lo, t_hi


class SearchController:
    """MCTS-at-inference controller: PUCT search over the engine's snapshot
    protocol, with priors/values from a pluggable evaluator (a PPO checkpoint's
    policy/value heads, or the torch-free uniform evaluator).

    Requires a search-capable env: advertises ``wants_search_env`` so
    ``runner.run_games`` constructs a :class:`search_env.SearchNarrativeEnv`
    and hands it over via ``bind_env`` (duck-typed, like ``set_deck_names``).
    Decisions where the engine reports ``safe=0`` (mid-resolution prompts,
    mulligans — no snapshot possible) fall back to the evaluator's raw policy;
    the searched/fallback split is tallied in ``stats``. Menus whose legal
    actions are all interchangeable (``decode.menu_is_interchangeable`` —
    duplicate copies of one choice, including single-choice menus) skip both
    search and evaluation and answer with the first action instantly, tallied
    as ``trivial``; under a match clock this spends none of the bank. After a
    search, the per-world trees are kept: while the real game continues along
    a line the sims explored, decisions are answered from the stored visit
    counts (``_try_follow_tree``, tallied as ``followed``) instead of a new
    search — until the observed game diverges from every world's tree, any
    hidden information is revealed (``decode.hidden_info_fingerprint``), or
    the opponent takes an action outside the public-menu-safe categories.

    Bo3 SIDEBOARD prompts between games are also loop-safe search roots, but each
    rollout there is heavier (restore re-crosses init_ecs + deck load + shuffle)
    and has a game-long horizon; the in-game ``max_depth`` (mcts.run_search's
    default of 60) can be swallowed entirely by the remaining sideboard/mulligan
    prompts, landing the leaf value on a masked between-game observation. So at a
    sideboard root (``obs[_IS_SIDEBOARD_IDX] > 0.5``) the controller switches to
    the SIDEBOARD budget (``sb_sims``/``sb_worlds``/``sb_max_depth``), mirroring
    az_selfplay. The sideboard/in-game searched split is tallied separately.

    ``clock`` (opt-in) arms a match-level chess clock (:class:`MatchClock`):
    one wall-clock bank for the whole match, allocated per decision (harder
    roots earn more time, obvious ones early-stop via the search's stability
    check) and reset in ``bind_env``. ``paced`` (opt-in, human-facing play)
    pads every decision — searched, fallback, or single-choice — to a small
    jittered ~0.02-0.05s floor, and additionally offers :meth:`pace_idle`
    beats (occasional 0.2-0.5s fake thinks when the game never queried this
    controller at all), so response timing leaks nothing about whether there
    was anything to think about. Both default off and leave every existing
    code path (training, eval gates, analysis) byte-identical.
    """

    wants_search_env = True

    # Paced-mode timing. Every paced response is padded to at least a small
    # jittered floor (_PACE_FLOOR_S..+_PACE_JITTER_S — fast, just enough to
    # blur "forced" vs "instantly decided"). Separately, pace_idle()
    # occasionally (_PACE_IDLE_P) asks the driver to hold a LONGER
    # _PACE_IDLE_MIN_S.._PACE_IDLE_MAX_S beat when the game never queried
    # this controller at all, so to the human it looks like the model was
    # offered (and took) a decision it never actually had.
    _PACE_FLOOR_S = 0.02
    _PACE_JITTER_S = 0.03
    _PACE_IDLE_P = 0.25
    _PACE_IDLE_MIN_S = 0.2
    _PACE_IDLE_MAX_S = 0.5

    def __init__(self, evaluator, *, sims: int = 128, worlds: int = 4,
                 c_puct: float = 1.5, temperature: float = 0.0,
                 label: str = "mcts", rng_seed: int = 0,
                 sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
                 sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
                 sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
                 sb_persist: bool = bool(DEFAULT_SB_PERSIST),
                 time_budget: float | None = None,
                 sims_cap: int = 0, sb_sims_cap: int = 0, procs: int = 1,
                 clock: float | None = None, clock_t_min: float = 0.5,
                 clock_t_max: float = 60.0, clock_sb_t_max: float = 15.0,
                 paced: bool = False, merge_dupes: bool = True):
        from mcts import run_search  # noqa: F401 — fail fast if unavailable
        if procs < 1:
            raise ValueError(f"procs must be >= 1, got {procs}")
        if clock is not None and clock <= 0:
            raise ValueError(f"clock must be > 0 seconds, got {clock}")
        if clock_t_min <= 0:
            raise ValueError(f"tmin must be > 0 seconds, got {clock_t_min}")
        if clock_t_max < clock_t_min:
            raise ValueError(
                f"tmax ({clock_t_max}) must be >= tmin ({clock_t_min})")
        self._evaluator = evaluator
        self._sims = sims
        self._worlds = worlds
        self._c_puct = c_puct
        self._temperature = temperature
        self._sb_sims = sb_sims
        self._sb_worlds = sb_worlds
        self._sb_max_depth = sb_max_depth
        self._sb_rollout_turns = sb_rollout_turns
        self._sb_persist = bool(sb_persist)
        # Wall-clock per-decision budget (seconds). When set, the deadline
        # terminates each search; sims_cap/sb_sims_cap are the OPTIONAL hard caps
        # (0 => clock is the sole terminator, i.e. run as many sims as fit).
        self._time_budget = time_budget
        self._sims_cap = sims_cap
        self._sb_sims_cap = sb_sims_cap
        # Duplicate-edge merging (decode.menu_merge_reps). Config-constant for
        # the controller's lifetime — must not flip across a sideboard
        # boundary's persisted trees.
        self._merge_dupes = bool(merge_dupes)
        # Mirror-pool engine count for world-parallel interactive search. procs==1
        # (the default) keeps the single-env run_search call site byte-identical
        # (parity depends on it); procs>1 fans the worlds across procs-1 extra
        # lockstep engine processes (interactive consumers only).
        self._procs = procs
        # Match-level chess clock (opt-in): a wall-clock bank spanning one
        # whole match, allocated per decision by MatchClock. Paced mode pads
        # every choose() to a jittered floor to mask timing tells (play-only).
        self._clock = (MatchClock(clock, t_min=clock_t_min, t_max=clock_t_max,
                                  sb_t_max=clock_sb_t_max)
                       if clock is not None else None)
        self._paced = bool(paced)
        self.label = label
        self._rng = np.random.default_rng(rng_seed)
        self._env = None
        # Tree reuse: the previous search's per-world root nodes, the env
        # action-history length at search time, and the revealed-info
        # fingerprint captured then. While the real game walks a line those
        # trees simulated — and reveals nothing new — decisions are answered
        # from the stored visit counts instead of a fresh search
        # (see _try_follow_tree).
        self._followed_trees = None
        self._followed_hist_len = 0
        self._followed_fp = None
        # Sideboard-boundary persistence (sb_persist): one set of per-world
        # trees + one rollout memo shared across a boundary's consecutive
        # picks. key = mcts.sb_root_key identity; seeds = the pinned per-world
        # determinize seeds (latched from the boundary's first search); roots =
        # the previous search's SearchResult.roots (walked down the history
        # delta at the next pick); picks = descriptors of the real picks since
        # the boundary root (the memo key base). Works at any procs: the
        # parallel search forwards the same seeds/roots/memo per world (world
        # order == env order), and the latched per-world state is
        # engine-agnostic — so if the mirror pool dies mid-boundary (fail-open)
        # the next pick simply re-partitions the same seeds/roots across the
        # remaining env count, or runs serially, with nothing lost.
        self._sbp_key = None
        self._sbp_seeds = None
        self._sbp_roots = None
        self._sbp_hist_len = 0
        self._sbp_memo: dict = {}
        self._sbp_picks: list = []
        # Optional observer hook: called after every SEARCHED decision with
        # (obs_copy, num_choices, SearchResult, chosen_action). Fires on the
        # caller's (driver) thread; the GUI analysis window uses it to display
        # the opponent's own search. Fallback (unsearched) decisions produce no
        # result and never fire it. Hook errors are swallowed — a display hook
        # must never break play.
        self.on_result = None
        self.stats = {"searched": 0, "fallback": 0, "trivial": 0, "followed": 0,
                      "sims": 0, "sim_steps": 0, "sb_searched": 0,
                      "sb_reused_visits": 0, "sb_memo_hits": 0,
                      "pool_procs": procs, "early_stops": 0}
        if self._clock is not None:
            self.stats["clock_bank"] = self._clock.bank
            self.stats["clock_remaining"] = self._clock.remaining

    def bind_env(self, env) -> None:
        self._env = env
        self._followed_trees = None
        self._followed_hist_len = 0
        self._followed_fp = None
        self._drop_boundary()
        # bind_env fires once per match (runner.run_games / tui_game.run), so
        # this is the one-bank-per-match reset point.
        if self._clock is not None:
            self._clock.reset()
            self.stats["clock_remaining"] = self._clock.remaining

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        t0 = time.monotonic()
        if self._clock is not None:
            # Every decision faced (searched, followed, trivial, fallback)
            # advances the match pace the clock's horizon is revised from.
            self._clock.note_decision(int(round(float(obs[_MATCH_CTX_START]) * 3)))
        try:
            return self._choose_impl(obs, num_choices, action_masks,
                                     decoded_actions)
        finally:
            if self._clock is not None:
                # Debit BEFORE the pacing pad: the pad is presentation-only
                # (hundreds of trivial decisions would silently drain the bank).
                self._clock.debit(time.monotonic() - t0)
                self.stats["clock_remaining"] = self._clock.remaining
            if self._paced:
                pad = (self._PACE_FLOOR_S
                       + float(self._rng.uniform(0.0, self._PACE_JITTER_S))) \
                    - (time.monotonic() - t0)
                if pad > 0.0:
                    time.sleep(pad)

    def pace_idle(self) -> float:
        """Seconds a human-facing driver should hold when the game passed BY
        this controller without offering it a decision (no legal action, so no
        query was ever emitted). Usually 0.0; occasionally, in paced mode, a
        0.2-0.5s beat — shown to the human exactly like a real think, so an
        instant hand-back no longer proves the model had nothing to decide.
        The controller does not sleep here; the caller owns the pause (it may
        want to show its thinking indicator around it)."""
        if not self._paced or float(self._rng.random()) >= self._PACE_IDLE_P:
            return 0.0
        return float(self._rng.uniform(self._PACE_IDLE_MIN_S,
                                       self._PACE_IDLE_MAX_S))

    _FOLLOW_MIN_VISITS = 16  # combined visits a followed node needs to be trusted

    # Opponent action categories whose menus are built from PUBLIC state only
    # (priority pass, combat declarations and confirms) — the one case where a
    # raw action index means the same thing in every determinized world's menu.
    # Any other opponent action either reveals a hidden card (the fingerprint
    # gate re-searches anyway) or sits in a hand-dependent menu where the same
    # index can mean different actions across worlds — both force a re-search.
    _FOLLOW_OPP_SAFE_CATS = frozenset({
        CAT_PASS_PRIORITY, CAT_SELECT_ATTACKER, CAT_CONFIRM_ATTACKERS,
        CAT_SELECT_BLOCKER, CAT_CONFIRM_BLOCKERS})

    def _drop_trees(self) -> None:
        self._followed_trees = None
        self._followed_fp = None

    def _drop_boundary(self) -> None:
        self._sbp_key = None
        self._sbp_seeds = None
        self._sbp_roots = None
        self._sbp_hist_len = 0
        self._sbp_memo = {}
        self._sbp_picks = []

    def _sbp_latch(self, obs, chosen: int) -> None:
        """Record the chosen action's pick descriptor while a boundary is
        armed (mirrors the C++ actor latching in finalize + its fallback path;
        the walk itself consumes the env action history, so only the memo's
        descriptors need choose-time capture — non-pick actions contribute
        none)."""
        if self._sbp_roots is None:
            return
        from mcts import sb_pick_descriptor
        d = sb_pick_descriptor(obs, int(chosen))
        if d is not None:
            self._sbp_picks.append(d)

    def _try_follow_tree(self, obs, num_choices):
        """Answer from the previous search's trees while the real game stays on
        a simulated line, instead of searching again.

        The env action history recorded since that search is walked down every
        world's tree (tree children are keyed by the same pre-remap env action
        indices the history records). Two gates run before the walk:

        * every intervening OPPONENT action must be public-menu safe
          (``_FOLLOW_OPP_SAFE_CATS``, from the env's per-step ``_action_meta``)
          — own actions were picked from the real menu and are always exact;
        * no hidden information may have been revealed since the search: the
          current ``decode.hidden_info_fingerprint`` must not exceed the one
          captured at search time (a draw, a play from a hidden hand, a mill
          or a tutor/scry reveal invalidates every world's presuppositions,
          even a world that happened to guess it right).

        Worlds that never simulated the real line drop out; the survivors'
        visit counts at the reached node are the answer. Divergence — a gate
        firing, an unexplored real action, a menu-size or seat mismatch at the
        reached node, or fewer than ``_FOLLOW_MIN_VISITS`` combined visits —
        drops the trees and returns None so a fresh search runs. On success
        the reached nodes become the new follow roots and the fingerprint
        ratchets to the current one, so a whole expected sequence can be
        played from one search. Works for ``safe=0`` prompts too (the sims
        played through those decisions), where a fresh search is impossible
        and the fallback would otherwise be the raw policy."""
        from decode import hidden_info_fingerprint

        trees = self._followed_trees
        if not trees:
            return None
        hist = getattr(self._env, "_action_history", None)
        meta = getattr(self._env, "_action_meta", None)
        if hist is None or meta is None or len(meta) != len(hist):
            self._drop_trees()
            return None
        delta = hist[self._followed_hist_len:]
        if not delta:
            self._drop_trees()
            return None
        self_is_a = bool(obs[_SELF_IS_A_IDX] > 0.5)
        # Gate 1: only public-menu-safe opponent actions may be followed
        # through (index semantics are world-independent for those alone).
        for actor_is_a, cat in meta[self._followed_hist_len:]:
            if actor_is_a != self_is_a and cat not in self._FOLLOW_OPP_SAFE_CATS:
                self._drop_trees()
                return None
        # Gate 2: any newly visible card identity since the search means the
        # worlds' presuppositions are (partially) falsified — re-search.
        fp = self._followed_fp
        cur = hidden_info_fingerprint(obs[:STATE_SIZE])
        if fp is None or np.any(cur[0] > fp[0]) or np.any(cur[1] > fp[1]):
            self._drop_trees()
            return None
        nodes = []
        for root in trees:
            node = root
            for a in delta:
                a = int(a)
                # Real actions may be merged-away duplicates; the tree only
                # keeps children at representative indices.
                if node.rep is not None and 0 <= a < node.num_choices:
                    a = int(node.rep[a])
                node = node.children.get(a)
                if node is None:
                    break
            if (node is not None and node.num_choices == num_choices
                    and node.self_is_a == self_is_a):
                nodes.append(node)
        if not nodes:
            self._drop_trees()
            return None
        visits = np.zeros(num_choices, dtype=np.float64)
        for n in nodes:
            visits += n.N
        if visits.sum() < self._FOLLOW_MIN_VISITS:
            self._drop_trees()
            return None
        self._followed_trees = nodes
        self._followed_hist_len = len(hist)
        self._followed_fp = cur  # ratchet: a card that hides re-triggers on re-reveal
        if self._temperature <= 1e-6:
            return int(np.argmax(visits))
        pi = visits ** (1.0 / self._temperature)
        return int(self._rng.choice(num_choices, p=pi / pi.sum()))

    def _choose_impl(self, obs, num_choices, action_masks=None,
                     decoded_actions=None) -> int:
        from mcts import run_search, run_search_parallel
        from decode import menu_is_interchangeable

        # Sideboard-boundary identity of this decision (None off the sideboard
        # phase or with persistence disabled).
        sbp_key = None
        if self._sb_persist:
            from mcts import sb_root_key
            sbp_key = sb_root_key(obs)

        # An interchangeable menu (four "Play Mountain" from a hand of
        # duplicates, N identical untapped basics to tap) offers the same
        # choice N times — search or evaluation would only spend wall clock
        # (and the match-clock bank) telling copies apart. Answer with the
        # first instantly; paced mode's floor pad still masks the timing.
        if menu_is_interchangeable(obs, num_choices):
            self.stats["trivial"] += 1
            self._sbp_latch(obs, 0)
            return 0

        # While the game keeps matching the previous search's expected lines,
        # play on from those trees rather than recalculating (spends none of
        # the match clock); the first divergence falls through to a search.
        # At a PERSISTED sideboard root the follow-only answer is skipped —
        # the boundary path below re-roots those same trees and tops them up
        # with fresh sims instead of answering from stale visits.
        if sbp_key is None:
            followed = self._try_follow_tree(obs, num_choices)
            if followed is not None:
                self.stats["followed"] += 1
                return followed

        env = self._env
        searchable = (
            env is not None
            and getattr(env, "last_search_safe", None)
            and num_choices > 1
        )
        if not searchable:
            self.stats["fallback"] += 1
            priors, _ = self._evaluator.evaluate(obs, num_choices)
            chosen = int(np.argmax(priors))
            self._sbp_latch(obs, chosen)
            return chosen
        # Mirror pool: with procs>1 the worlds fan out across procs-1 extra engine
        # processes (interactive only; duck-typed so a plain env is tolerated).
        # envs is None => procs==1 (or no mirror support) => the original
        # single-env run_search call, byte-identical. Either way the same kw dict
        # (sideboard-boundary seeds/roots/memo included) is forwarded unchanged,
        # so boundary persistence behaves identically on both paths.
        envs = None
        if self._procs > 1:
            ensure = getattr(env, "ensure_mirrors", None)
            if ensure is not None:
                ensure(self._procs - 1)
                envs = env.search_envs()

        def _search(**kw):
            if envs is not None:
                return run_search_parallel(envs, self._evaluator, **kw)
            return run_search(env, self._evaluator, **kw)

        # A bo3 sideboard root gets the deeper/fewer-sim sideboard budget (its
        # horizon is the whole next game). Key ONLY off is_sideboard_phase —
        # is_post_board / game_number still reflect the just-ended game at a
        # g1->g2 root. In-game roots keep run_search's default max_depth (60).
        tb = self._time_budget
        tmin_s = None
        is_sb = obs[_IS_SIDEBOARD_IDX] > 0.5
        if self._clock is not None:
            # Difficulty signal: one extra deterministic root eval (run_search
            # re-evaluates internally). Cheap next to a multi-second search,
            # draws no rng, and keeps the parity-sensitive search code untouched.
            priors, _ = self._evaluator.evaluate(obs, num_choices)
            turn = int(round(float(obs[_CUR_TURN_IDX]) * 50))
            g = int(round(float(obs[_MATCH_CTX_START]) * 3))
            ws = int(round(float(obs[_MATCH_CTX_START + 1]) * 2))
            wo = int(round(float(obs[_MATCH_CTX_START + 2]) * 2))
            tmin_s, alloc = self._clock.allocate(
                turn=turn, game_number=g, self_wins=ws, opp_wins=wo,
                is_sideboard=is_sb, priors=priors, num_choices=num_choices)
            # An explicit time= budget acts as a per-decision ceiling.
            tb = alloc if tb is None else min(alloc, tb)
            tmin_s = min(tmin_s, tb)
        timed = tb is not None
        if is_sb:
            kw = dict(
                sims=(self._sb_sims_cap if timed else self._sb_sims),
                worlds=self._sb_worlds, c_puct=self._c_puct,
                max_depth=self._sb_max_depth,
                rollout_turns=self._sb_rollout_turns,
                rng=self._rng, time_budget_s=tb,
                time_budget_min_s=tmin_s,
                merge_dupes=self._merge_dupes)
            persist_here = sbp_key is not None
            if persist_here:
                # Boundary continue: same identity AND every action since the
                # previous search is our own (the engine runs one seat's picks
                # contiguously, so any foreign action means the boundary is
                # over). On continue the previous trees are walked down the
                # history delta and topped up under the pinned seeds; else a
                # fresh boundary starts (seeds latched from this search).
                from mcts import walk_reuse_root
                cont = False
                hist = getattr(env, "_action_history", None)
                meta = getattr(env, "_action_meta", None)
                self_is_a = bool(obs[_SELF_IS_A_IDX] > 0.5)
                if (self._sbp_roots is not None and self._sbp_key == sbp_key
                        and self._sbp_seeds is not None
                        and hist is not None and meta is not None
                        and len(meta) == len(hist)
                        and all(a_is_a == self_is_a for a_is_a, _c in
                                meta[self._sbp_hist_len:])):
                    delta = hist[self._sbp_hist_len:]
                    kw.update(
                        world_seeds=self._sbp_seeds,
                        reuse_roots=[walk_reuse_root(r, delta, num_choices,
                                                     self_is_a)
                                     for r in self._sbp_roots])
                    cont = True
                if not cont:
                    self._drop_boundary()
                kw.update(rollout_memo=self._sbp_memo,
                          memo_picks=tuple(self._sbp_picks))
            result = _search(**kw)
            self.stats["sb_searched"] += 1
            if persist_here:
                self._sbp_key = sbp_key
                self._sbp_seeds = result.seeds
                self._sbp_roots = result.roots
                self._sbp_hist_len = len(getattr(env, "_action_history", []))
                self.stats["sb_reused_visits"] += result.reused_visits
                self.stats["sb_memo_hits"] += result.memo_hits
        else:
            self._drop_boundary()
            result = _search(
                sims=(self._sims_cap if timed else self._sims),
                worlds=self._worlds, c_puct=self._c_puct,
                rng=self._rng, time_budget_s=tb, time_budget_min_s=tmin_s,
                merge_dupes=self._merge_dupes)
        self.stats["searched"] += 1
        self.stats["sims"] += result.sims_run
        self.stats["sim_steps"] += result.sim_steps
        # Arm tree-following from this search's per-world trees: the next
        # decisions are answered from them for as long as the real game walks
        # a line the sims explored and reveals nothing new (the fingerprint
        # captured here is the reveal baseline). Persisted sideboard roots
        # skip arming — the boundary machinery owns those trees, and a
        # follow-only answer would bypass the next pick's top-up search.
        hist = getattr(env, "_action_history", None)
        if result.roots and hist is not None and sbp_key is None:
            from decode import hidden_info_fingerprint
            self._followed_trees = result.roots
            self._followed_hist_len = len(hist)
            self._followed_fp = hidden_info_fingerprint(obs[:STATE_SIZE])
        else:
            self._drop_trees()
        if result.stopped_early:
            self.stats["early_stops"] += 1
        if self._temperature <= 1e-6:
            chosen = result.best_action()
        else:
            pi = result.policy_target(self._temperature)
            chosen = int(self._rng.choice(len(pi), p=pi))
        self._sbp_latch(obs, chosen)
        if self.on_result is not None:
            try:
                self.on_result(np.array(obs, copy=True), int(num_choices),
                               result, int(chosen))
            except Exception:  # noqa: BLE001 — observer must never break play
                pass
        return chosen


class ActionListController:
    """Plays a fixed sequence of action indices (test harness ``--actions``).

    Consumes one index per decision regardless of which side has priority (the
    sequence is global, matching the harness convention); once exhausted it
    falls back to action 0 (pass / first choice).
    """

    def __init__(self, actions: Sequence[int], label: str = "Actions"):
        self._actions = [int(a) for a in actions]
        self._i = 0
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        if self._i < len(self._actions):
            a = self._actions[self._i]
            self._i += 1
            return a
        return 0


class PlayController:
    """Plays a fixed sequence of semantic action specs (``--play``).

    Each spec (``cast:Lightning Bolt``, ``target:Grizzly Bears@opp``, ``pass``,
    ``#7`` …) is resolved against *this* decision's decoded menu via
    :mod:`action_spec`, so the sequence is robust to dynamic index reordering.
    The sequence is global (one spec consumed per decision); once exhausted it
    falls back to action ``0`` (pass / first choice — always legal) so the game
    keeps advancing to its conclusion or the decision cap.

    **Seat keys.** When the SAME controller drives both seats (the test harness's
    dual-seat ``--play`` mode), sequencing the priority hand-offs between the two
    players by hand is error-prone. So a spec may be pinned to a seat with a
    leading ``A:`` / ``B:`` key (see :mod:`action_spec`). Before applying the next
    spec this controller checks the seat that currently holds priority
    (``obs[_SELF_IS_A_IDX]`` — true = Player A): if the next spec is keyed to the *other*
    seat, the current priority holder passes (the spec is **not** consumed) and
    play advances until the keyed seat is on the clock. Unkeyed specs are applied
    to whoever has priority (the legacy behaviour), so existing scripts are
    unaffected.

    A keyed spec also **passes the keyed seat forward through its own priority
    windows** until the action becomes legal: e.g. ``A:attack:Voice`` given while A
    still holds priority in its main phase auto-passes (leaving the spec unconsumed)
    until A reaches the declare-attackers step where ``attack:`` is offered, instead
    of erroring because no attacker menu exists yet. This fires only on a genuine "not
    offered at this decision" (``no_match``) and only while a pass is itself legal (a
    priority window); an ambiguous/typo'd spec, or one that reaches a mandatory choice
    with no pass, still raises :class:`action_spec.PlayResolveError` with the menu,
    bounded by ``_MAX_WAIT`` so a never-legal spec can't silently pass the game away.

    ``wants_decoded = True`` tells the runner to hand ``choose`` the decoded menu.
    A spec that resolves to zero or multiple legal actions raises
    :class:`action_spec.PlayResolveError` — a loud failure with the legal menu,
    instead of silently playing the wrong index. ``self.resolved`` accumulates
    the integer indices chosen, so a run can be replayed as a plain ``--actions``
    list.
    """

    wants_decoded = True

    # Cap on consecutive auto-passes spent waiting for a seat-keyed spec to become
    # legal, so a spec that is never offered (a typo, or an action the seat can never
    # take) fails loudly instead of silently passing the rest of the game away.
    _MAX_WAIT = 200

    def __init__(self, specs, label: str = "Play"):
        import action_spec
        self._action_spec = action_spec
        self._specs = action_spec.parse_spec_list(specs)
        self._i = 0
        self._wait = 0
        self.label = label
        self.resolved: list[int] = []

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        if decoded_actions is None:
            raise RuntimeError("PlayController requires decoded_actions; drive it "
                               "through runner.run_games (which supplies the menu).")
        # Specs exhausted: auto-advance with action 0 (always legal), like
        # AutoPassController, so the game runs to its end / the decision cap.
        if self._i >= len(self._specs):
            self.resolved.append(0)
            return 0
        spec = self._specs[self._i]
        # Seat-keyed spec for the seat that is NOT on the clock: the current
        # priority holder passes (spec left for later) so we advance to the keyed
        # seat's decision instead of mis-applying its action to this player.
        seat = self._action_spec.spec_seat(spec)
        if seat is not None and seat != ("A" if obs[_SELF_IS_A_IDX] > 0.5 else "B"):
            idx = self._action_spec.resolve_to_index("pass", decoded_actions)
            self.resolved.append(idx)
            return idx

        r = self._action_spec.resolve(spec, decoded_actions)
        if r.ok:
            self._i += 1
            self._wait = 0
            self.resolved.append(r.index)
            return r.index

        # The keyed seat IS on the clock but its action isn't legal at THIS decision
        # (e.g. `A:attack:Voice` while A still holds priority in its main phase — the
        # attacker menu only appears at the declare-attackers step). Pass priority to
        # advance the keyed seat toward the step where the action becomes legal, leaving
        # the spec unconsumed, instead of failing. Bounded by _MAX_WAIT, and only on a
        # genuine "not offered here" (`no_match`): an ambiguous/typo'd spec, or one whose
        # decision offers no pass (a mandatory choice), still fails loudly with the menu.
        if (seat is not None and r.kind == "no_match" and self._wait < self._MAX_WAIT):
            pass_r = self._action_spec.resolve("pass", decoded_actions)
            if pass_r.ok:
                self._wait += 1
                self.resolved.append(pass_r.index)
                return pass_r.index

        raise self._action_spec.PlayResolveError(r, decoded_actions)


class InteractiveController:
    """Prompts stdin for an action index each decision (test harness ``--interactive``)."""

    def __init__(self, label: str = "Human"):
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        while True:
            try:
                raw = input("  >> Enter action index: ").strip()
                c = int(raw)
                if 0 <= c < num_choices:
                    return c
                print(f"     Invalid: must be 0-{num_choices - 1}")
            except (ValueError, EOFError):
                print("     Enter a valid integer")


class AutoPassController:
    """Always picks action 0 (pass priority / first choice) — the harness default."""

    def __init__(self, label: str = "Auto"):
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        return 0


class HumanController:
    """Interactive CLI seat: renders the board + legal menu, accepts an index
    OR a semantic action spec (``cast:bolt``, ``target:bears@opp``, ``pass`` —
    the same grammar as ``--play``), 'quit' to exit.

    This is the "human plays at a terminal" agent for runner-driven games
    (``play.py``'s text mode, ``run_match(..., 'human')``). It renders its own
    decision view, so drive it with the runner's transcript set to
    ``"narrative"`` (or quiet) — the verbose transcript would print the state
    twice.
    """

    wants_decoded = True

    def __init__(self, label: str = "You", show_state: bool = True):
        self.label = label
        self._show_state = show_state

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        import action_spec
        import decode
        from env import STATE_SIZE

        if decoded_actions is None:
            decoded_actions = decode.decode_actions_from_obs(obs, num_choices)
        if self._show_state:
            gs = decode.decode_game_state(obs[:STATE_SIZE])
            for ln in decode.format_state_lines(gs):
                print(ln)
        print(f"  Available actions ({num_choices}):")
        for ln in decode.format_action_lines(decoded_actions):
            print(ln)
        while True:
            try:
                raw = input("Choose> ").strip()
            except EOFError:
                raise SystemExit(0)
            if not raw:
                continue
            if raw.lower() in ("quit", "q", "exit"):
                raise SystemExit(0)
            if raw.lstrip("#").isdigit():
                c = int(raw.lstrip("#"))
                if 0 <= c < num_choices:
                    return c
                print(f"  Enter a number between 0 and {num_choices - 1}.")
                continue
            # Semantic spec — same resolver as --play.
            r = action_spec.resolve(raw, decoded_actions)
            if r.ok:
                return r.index
            print(f"  {r.reason}")


def _load_model(path: str):
    """Load a checkpoint with MaskablePPO (falling back to PPO). Lazy sb3 import."""
    try:
        from sb3_contrib import MaskablePPO as _PPO
    except ImportError:
        from stable_baselines3 import PPO as _PPO
    return _PPO.load(path, device="cpu")


def make_controller(spec, *,
                    checkpoint_resolver: Optional[Callable[[str], str]] = None,
                    deterministic: bool = False) -> Controller:
    """Resolve an agent spec into a Controller (the one agent grammar).

    Accepts:
      - a ``Controller`` instance — returned as-is (lets callers mix
        pre-built controllers with spec strings);
      - scripted specs ("scripted", "scripted:hard", "easy", "greedy",
        "random", "explore", "explore:patient", ...) → ScriptedController;
      - "auto" / "autopass" → AutoPassController (always action 0);
      - "human" → HumanController (interactive CLI seat, index or semantic
        input);
      - "play:<spec,spec,...>" → PlayController (semantic action script,
        same grammar as the test harness ``--play``);
      - "actions:<i,i,...>" → ActionListController (positional indices);
      - "mcts:<gen-or-path>[?sims=128&worlds=4&c=1.5&temp=0&vscale=1&time=&clock=&paced=]" →
        SearchController running PUCT search with that checkpoint's
        policy/value heads as priors/leaf values ("mcts:uniform" for the
        torch-free uniform evaluator — plumbing tests, weak play; "mcts:gen"
        or an explicit .zip for a real net — the deck it pilots is a separate
        parameter). ``time=<seconds>`` sets a wall-clock per-decision budget:
        the search runs as many sims as fit in that many seconds (more time =
        stronger play), overriding sims as the terminator.
        ``clock=<seconds>`` instead arms a whole-MATCH chess-clock bank
        (variable per-decision allocation between tmin/tmax — harder roots
        earn more time, obvious ones stop early); ``paced=1`` masks timing
        tells in human-facing play (a small jittered response floor plus
        occasional fake-think beats when no decision was even offered);
      - "az:<gen-or-path>[?sims=&worlds=&c=&temp=&seed=&time=&clock=&paced=]" → SearchController
        driven by an AZNet ("az:gen" → the generalist AZ net, else a warm-start
        from the gen PPO net; an explicit .pt/.zip path also works);
      - "azraw:<gen-or-path>" → AZRawController (AZNet policy argmax, no
        search — cheap gating);
      - "gen" → the single generalist checkpoint, or any explicit model .zip
        path, resolved via ``checkpoint_resolver`` (default:
        :func:`resolve_checkpoint`) and loaded as a ModelController. A bare
        former per-deck shorthand (e.g. "delver") is no longer accepted: the
        model is the one generalist "gen" and the deck it pilots travels as a
        separate parameter, so such a spec raises a clear error.
    """
    if not isinstance(spec, str):
        return spec
    s = spec.strip()
    low = s.lower()
    if is_scripted_spec(s):
        return ScriptedController(make_agent(s), label=s)
    if low in ("auto", "autopass"):
        return AutoPassController()
    if low == "human":
        return HumanController()
    if low.startswith("play:"):
        return PlayController(s[len("play:"):])
    if low.startswith("actions:"):
        return ActionListController(
            [int(a) for a in s[len("actions:"):].split(",") if a.strip()])
    if low.startswith("mcts:"):
        return _make_search_controller(s[len("mcts:"):],
                                       checkpoint_resolver=checkpoint_resolver)
    if low.startswith("az:"):
        return _make_az_controller(s[len("az:"):], search=True)
    if low.startswith("azraw:"):
        return _make_az_controller(s[len("azraw:"):], search=False)
    resolver = checkpoint_resolver or resolve_checkpoint
    path = resolver(s)
    return ModelController(_load_model(path), label=s, deterministic=deterministic)


def _parse_spec_query(spec: str) -> tuple:
    """Split a ``<base>[?k=v&k=v...]`` controller spec into (base, params).

    Shared by the mcts:/az: builders so the query grammar lives in one place."""
    base, _, query = spec.partition("?")
    params: dict[str, str] = {}
    if query:
        for part in query.split("&"):
            if not part.strip():
                continue
            k, _, v = part.partition("=")
            params[k.strip().lower()] = v.strip()
    return base.strip(), params


def _spec_knob(params: dict, key: str, default, conv, spec: str):
    """Fetch+convert one query knob, failing LOUDLY on a bad/empty value.

    Without this, a malformed spec like "az:delver?worlds=" reaches int("")
    and dies with a bare ValueError traceback instead of naming the spec."""
    raw = params.get(key)
    if raw is None:
        return default
    try:
        return conv(raw)
    except ValueError:
        raise ValueError(
            f"bad controller spec {spec!r}: knob '{key}' needs a "
            f"{conv.__name__} value, got {raw!r}") from None


def default_search_procs(worlds: int = DEFAULT_SEARCH_WORLDS) -> int:
    """The engine-process count an INTERACTIVE front end should fan a search's
    worlds across when the user did not ask for one.

    Half the visible cores (at least 1), capped at ``worlds`` — a world is the
    unit of parallelism, so more processes than worlds would idle, and leaving
    half the machine free keeps the UI (and the human's own machine) responsive
    while a search runs. NOT the spec-grammar default: a bare ``mcts:``/``az:``
    spec still builds procs=1 so gates, eval and parity runs stay reproducible;
    only play.py / the GUI launcher opt in to this.
    """
    return min(int(worlds), max(1, (os.cpu_count() or 2) // 2))


def default_search_procs_for_spec(spec: str) -> int:
    """:func:`default_search_procs` for a controller spec, reading the spec's own
    ``worlds=`` knob (default :data:`DEFAULT_SEARCH_WORLDS`) as the cap."""
    _base, params = _parse_spec_query(spec or "")
    return default_search_procs(
        _spec_knob(params, "worlds", DEFAULT_SEARCH_WORLDS, int, spec))


def _boolean(raw: str) -> bool:
    """Bool converter for _spec_knob (its error message uses conv.__name__)."""
    v = raw.strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(raw)


_boolean.__name__ = "boolean"  # cleaner knob-error text ("needs a boolean value")


def _parse_clock_knobs(params: dict, spec: str):
    """Parse the match-clock/pacing knobs shared by the mcts:/az: grammars.

    Returns ``(clock, tmin, tmax, sb_tmax, paced)``. ``clock=<seconds>`` arms a
    match-level chess clock (one bank for the whole match; see MatchClock);
    tmin/tmax bound the per-decision allocation, sb_tmax caps sideboard roots.
    ``paced=1`` masks response-timing tells (a small jittered floor on every
    decision plus occasional pace_idle fake-think beats; human-facing play
    only). Range validation lives in SearchController so programmatic
    construction fails identically."""
    clock = _spec_knob(params, "clock", None, float, spec)
    if clock is not None and clock <= 0.0:
        raise ValueError(f"bad controller spec {spec!r}: knob 'clock' needs a "
                         f"positive seconds value, got {clock!r}")
    tmin = _spec_knob(params, "tmin", 0.5, float, spec)
    tmax = _spec_knob(params, "tmax", 60.0, float, spec)
    sb_tmax = _spec_knob(params, "sb_tmax", 15.0, float, spec)
    paced = _spec_knob(params, "paced", False, _boolean, spec)
    return clock, tmin, tmax, sb_tmax, paced


def _parse_time_budget(params: dict, spec: str, sims: int, sb_sims: int):
    """Parse the optional ``time=<seconds>`` wall-clock budget knob.

    Returns ``(time_budget, sims_cap, sb_sims_cap)``. When ``time`` is absent
    the budget is ``None`` and the caps are irrelevant (SearchController uses the
    fixed sims counts). When ``time`` is set the wall clock terminates each
    search; a sims/sb_sims value is treated as a hard cap ONLY when the user
    pinned it explicitly in the spec — otherwise the cap is 0 (uncapped) so the
    clock alone governs and the search runs as many sims as fit in the budget."""
    time_budget = _spec_knob(params, "time", None, float, spec)
    if time_budget is not None and time_budget <= 0.0:
        raise ValueError(f"bad controller spec {spec!r}: knob 'time' needs a "
                         f"positive seconds value, got {time_budget!r}")
    sims_cap = sims if "sims" in params else 0
    sb_sims_cap = sb_sims if "sb_sims" in params else 0
    return time_budget, sims_cap, sb_sims_cap


def _effort_label(sims: int, worlds: int, time_budget: float | None,
                  clock: float | None = None) -> str:
    """Compact effort suffix for a search controller's display label."""
    if clock is not None:
        return f"clock{clock:g}s x{worlds}"
    if time_budget is not None:
        return f"{time_budget:g}s x{worlds}"
    return f"{sims}x{worlds}"


def _make_search_controller(spec: str, *,
                            checkpoint_resolver=None) -> "SearchController":
    """Build a SearchController from the part after "mcts:".

    Grammar: ``<base>[?k=v&k=v...]`` where <base> is "uniform" or a checkpoint
    path / deck shorthand, and the knobs are sims, worlds, c (c_puct),
    temp (root temperature), vscale (PPO value tanh scale), seed (search RNG),
    time (wall-clock seconds/decision — runs as many sims as fit, overriding
    sims as the terminator), procs (world-parallel engine processes for
    interactive search; default 1), the bo3 sideboard-root budget sb_sims /
    sb_worlds / sb_max_depth / sb_rollout_turns (leaf-rollout horizon in
    player turns, 0 = off) / sb_persist (persist trees + rollout memo across
    a boundary's picks, 1/0), and the match-clock knobs clock (whole-match
    wall-clock bank in seconds, allocated per decision) / tmin / tmax /
    sb_tmax / paced (response-timing masking for human play).
    """
    from mcts import PPOEvaluator, UniformEvaluator

    base, params = _parse_spec_query(spec)
    sims = _spec_knob(params, "sims", 128, int, spec)
    worlds = _spec_knob(params, "worlds", DEFAULT_SEARCH_WORLDS, int, spec)
    c_puct = _spec_knob(params, "c", 1.5, float, spec)
    temperature = _spec_knob(params, "temp", 0.0, float, spec)
    v_scale = _spec_knob(params, "vscale", 1.0, float, spec)
    rng_seed = _spec_knob(params, "seed", 0, int, spec)
    procs = _spec_knob(params, "procs", 1, int, spec)
    sb_sims = _spec_knob(params, "sb_sims", DEFAULT_SB_SIMS, int, spec)
    sb_worlds = _spec_knob(params, "sb_worlds", DEFAULT_SB_WORLDS, int, spec)
    sb_max_depth = _spec_knob(params, "sb_max_depth", DEFAULT_SB_MAX_DEPTH, int, spec)
    sb_rollout_turns = _spec_knob(params, "sb_rollout_turns",
                                  DEFAULT_SB_ROLLOUT_TURNS, int, spec)
    sb_persist = _spec_knob(params, "sb_persist", DEFAULT_SB_PERSIST, int, spec)
    time_budget, sims_cap, sb_sims_cap = _parse_time_budget(params, spec, sims, sb_sims)
    clock, tmin, tmax, sb_tmax, paced = _parse_clock_knobs(params, spec)

    if base.lower() == "uniform":
        evaluator = UniformEvaluator()
        label = f"mcts:uniform({_effort_label(sims, worlds, time_budget, clock)})"
    else:
        resolver = checkpoint_resolver or resolve_checkpoint
        model = _load_model(resolver(base))
        evaluator = PPOEvaluator(model, v_scale=v_scale)
        label = f"mcts:{base}({_effort_label(sims, worlds, time_budget, clock)})"
    return SearchController(evaluator, sims=sims, worlds=worlds, c_puct=c_puct,
                            temperature=temperature, label=label, rng_seed=rng_seed,
                            sb_sims=sb_sims, sb_worlds=sb_worlds,
                            sb_max_depth=sb_max_depth,
                            sb_rollout_turns=sb_rollout_turns,
                            sb_persist=bool(sb_persist),
                            time_budget=time_budget,
                            sims_cap=sims_cap, sb_sims_cap=sb_sims_cap, procs=procs,
                            clock=clock, clock_t_min=tmin, clock_t_max=tmax,
                            clock_sb_t_max=sb_tmax, paced=paced)


class AZRawController:
    """Raw AZNet policy seat (argmax over softmaxed masked logits, NO search).

    The cheap gating counterpart to a search controller — used by ``azraw:`` to
    compare policies without paying for MCTS."""

    def __init__(self, evaluator, label: str = "azraw"):
        self._evaluator = evaluator
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        priors, _ = self._evaluator.evaluate(obs, num_choices)
        return int(np.argmax(priors))


def _load_az_evaluator(base: str):
    """Load an AZNet -> AZEvaluator for the generalist contract. Accepts:
      - ``'gen'`` → the generalist AZ checkpoint (``gen__azfinal.pt`` / newest
        ``gen__azv*.pt``), else a warm-start from the ``gen`` PPO checkpoint;
      - an explicit ``.pt`` AZ path (loaded directly), or an explicit ``.zip``
        PPO path (warm-started).
    A bare per-deck shorthand (``az:delver``) is rejected with a clear error via
    ``resolve_checkpoint`` — the model is the ONE generalist and the deck it
    pilots travels as a separate parameter.
    Lazy import so opponents.py stays torch-free until a model seat is built."""
    from az_net import AZEvaluator, load_az, from_ppo, resolve_az_checkpoint
    az = resolve_az_checkpoint(base)
    if az:
        return AZEvaluator(load_az(az)), az
    if base.endswith(".pt"):
        return AZEvaluator(load_az(base)), base   # explicit AZ path; load_az raises if missing
    # Otherwise a PPO spec to warm-start from: 'gen' or an explicit .zip.
    # resolve_checkpoint raises a clear error on a bare (non-'gen') deck shorthand.
    return AZEvaluator(from_ppo(resolve_checkpoint(base))), base


def _make_az_controller(spec: str, *, search: bool):
    """Build an ``az:`` (MCTS+AZNet) or ``azraw:`` (raw policy) controller.

    Grammar: ``<base>[?sims=&worlds=&c=&temp=&seed=&time=&procs=&sb_sims=&sb_worlds=&sb_max_depth=&sb_rollout_turns=&sb_persist=&clock=&tmin=&tmax=&sb_tmax=&paced=]``
    where <base> is an AZ checkpoint path / deck shorthand (falls back to a PPO
    warm-start). ``time=<seconds>`` sets a wall-clock per-decision budget (runs
    as many sims as fit, overriding sims as the terminator). ``procs=<n>`` fans
    the worlds across n engine processes (world-parallel interactive search;
    default 1). sb_* set the bo3 sideboard-root search budget
    (``sb_rollout_turns`` the leaf-rollout horizon in player turns, 0 = off —
    note a rolled sim costs a rollout's worth of wall clock, so ``time=``/
    ``clock=`` deadlines, checked between sims, can overshoot by up to one
    rollout). ``clock=<seconds>``
    arms a whole-match chess-clock bank (per-decision allocation bounded by
    tmin/tmax, sideboard roots by sb_tmax); ``paced=1`` masks response-timing
    tells for human play (small jittered floor + occasional fake-think beats)."""
    base, params = _parse_spec_query(spec)
    evaluator, resolved = _load_az_evaluator(base)
    if not search:
        return AZRawController(evaluator, label=f"azraw:{base}")
    sims = _spec_knob(params, "sims", 128, int, spec)
    worlds = _spec_knob(params, "worlds", DEFAULT_SEARCH_WORLDS, int, spec)
    c_puct = _spec_knob(params, "c", 1.5, float, spec)
    temperature = _spec_knob(params, "temp", 0.0, float, spec)
    rng_seed = _spec_knob(params, "seed", 0, int, spec)
    procs = _spec_knob(params, "procs", 1, int, spec)
    sb_sims = _spec_knob(params, "sb_sims", DEFAULT_SB_SIMS, int, spec)
    sb_worlds = _spec_knob(params, "sb_worlds", DEFAULT_SB_WORLDS, int, spec)
    sb_max_depth = _spec_knob(params, "sb_max_depth", DEFAULT_SB_MAX_DEPTH, int, spec)
    sb_rollout_turns = _spec_knob(params, "sb_rollout_turns",
                                  DEFAULT_SB_ROLLOUT_TURNS, int, spec)
    sb_persist = _spec_knob(params, "sb_persist", DEFAULT_SB_PERSIST, int, spec)
    time_budget, sims_cap, sb_sims_cap = _parse_time_budget(params, spec, sims, sb_sims)
    clock, tmin, tmax, sb_tmax, paced = _parse_clock_knobs(params, spec)
    return SearchController(evaluator, sims=sims, worlds=worlds, c_puct=c_puct,
                            temperature=temperature,
                            label=f"az:{base}({_effort_label(sims, worlds, time_budget, clock)})",
                            rng_seed=rng_seed, sb_sims=sb_sims, sb_worlds=sb_worlds,
                            sb_max_depth=sb_max_depth,
                            sb_rollout_turns=sb_rollout_turns,
                            sb_persist=bool(sb_persist),
                            time_budget=time_budget,
                            sims_cap=sims_cap, sb_sims_cap=sb_sims_cap, procs=procs,
                            clock=clock, clock_t_min=tmin, clock_t_max=tmax,
                            clock_sb_t_max=sb_tmax, paced=paced)


def parse_pool_spec(spec: Union[str, Sequence]) -> list[tuple[str, float]]:
    """Parse a pool specification into a list of (spec, weight) pairs.

    Accepts a comma-separated string ("scripted:easy,scripted:hard,ckpt=2") or an
    iterable of specs / (spec, weight) tuples.  Each comma item may carry an
    optional "=<weight>" suffix (default weight 1.0).
    """
    items: list = []
    if isinstance(spec, str):
        items = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        items = list(spec)

    out: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, (tuple, list)):
            s, w = item[0], float(item[1])
        elif "=" in item:
            s, _, w_str = item.rpartition("=")
            s, w = s.strip(), float(w_str)
        else:
            s, w = item, 1.0
        out.append((s, w))
    return out


class OpponentPool:
    """Per-episode weighted sampler over a mix of scripted agents and checkpoints.

    Memory bound: under SubprocVecEnv each env is its own process, so caches are
    not shared and total resident models ≈ (checkpoints loaded per process) ×
    n_envs.  We cap the number of *active* checkpoints to
    ``max(1, floor(max_checkpoint_ratio * n_envs))`` and shard them across
    processes by ``env_index``, so each process only ever loads ≈ ceil(ratio)
    distinct checkpoints (≤ 1 per process at ratio ≤ 1.0).  Scripted specs are
    stateless and free, so every process keeps the full scripted set.
    """

    _matchup_warned = False

    def __init__(self, spec: Union[str, Sequence], *,
                 checkpoint_resolver: Optional[Callable[[str], str]] = None,
                 rng: Optional[np.random.Generator] = None,
                 n_envs: int = 1, env_index: int = 0,
                 max_checkpoint_ratio: float = 1.0,
                 deterministic: bool = False,
                 model_deck: Optional[str] = None,
                 opp_deck: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None):
        self._resolver = checkpoint_resolver
        self._deterministic = deterministic
        self._rng = rng if rng is not None else np.random.default_rng()
        self._cache: dict[str, Controller] = {}

        entries = parse_pool_spec(spec) or [("scripted", 1.0)]
        entries = self._expand_matchup_tokens(entries, model_deck, opp_deck, checkpoint_dir)
        scripted = [(s, w) for s, w in entries if is_scripted_spec(s)]
        ckpts = [(s, w) for s, w in entries if not is_scripted_spec(s)]

        # Cap and shard the checkpoint set to bound per-process memory.
        max_unique = max(1, int(math.floor(max_checkpoint_ratio * max(1, n_envs))))
        active = ckpts[:max_unique]
        n_envs = max(1, n_envs)
        my_ckpts = active[env_index % n_envs::n_envs] if active else []

        self._entries = scripted + my_ckpts
        if not self._entries:
            self._entries = [("scripted", 1.0)]
        weights = np.array([w for _, w in self._entries], dtype=float)
        self._weights = weights / weights.sum()
        self._specs = [s for s, _ in self._entries]

    @classmethod
    def _expand_matchup_tokens(cls, entries, model_deck, opp_deck, checkpoint_dir):
        """Replace each MATCHUP_MODEL_TOKEN entry with the matchup's compatible
        checkpoints (absolute paths), splitting its weight evenly among them.

        A token that matches no checkpoint is dropped (with a one-time warning);
        all other entries pass through unchanged."""
        out: list[tuple[str, float]] = []
        for spec, w in entries:
            if isinstance(spec, str) and spec.strip().lower() == MATCHUP_MODEL_TOKEN:
                files = matchup_checkpoints(model_deck, opp_deck, checkpoint_dir)
                if not files:
                    if not cls._matchup_warned:
                        print(f"[opponent-pool] WARNING: '{MATCHUP_MODEL_TOKEN}' matched no "
                              f"generalist snapshot ({GEN_STEM}__*.zip) in "
                              f"{checkpoint_dir}; ignoring it.")
                        cls._matchup_warned = True
                    continue
                share = w / len(files)
                out.extend((f, share) for f in files)
            else:
                out.append((spec, w))
        return out

    def _controller_for(self, spec: str) -> Controller:
        ctrl = self._cache.get(spec)
        if ctrl is None:
            ctrl = make_controller(spec, checkpoint_resolver=self._resolver,
                                   deterministic=self._deterministic)
            self._cache[spec] = ctrl
        return ctrl

    def sample(self) -> tuple[str, Controller]:
        """Return (label, controller) for this episode (weighted random)."""
        idx = int(self._rng.choice(len(self._specs), p=self._weights))
        spec = self._specs[idx]
        return spec, self._controller_for(spec)

    @property
    def specs(self) -> list[str]:
        return list(self._specs)


def _interleave(*lists) -> list:
    """Round-robin merge of several lists, preserving each list's own order.

    Used to share one bounded budget fairly between competing newest-first
    candidate lists (older generalist snapshots vs older exploiter snapshots), so
    truncating the merged list never starves the shorter source."""
    out = []
    for i in range(max((len(l) for l in lists), default=0)):
        for l in lists:
            if i < len(l):
                out.append(l[i])
    return out


class LeaguePool:
    """League opponent sampler for PFSP / softmax multi-matchup training.

    Per episode it makes the two coupled choices the league needs — which opponent
    *deck* and which *controller* pilots it — from a pool spanning:

      1. a scripted anchor piloting a sampled roster deck (collapse guard),
      2. frozen generalist snapshots (``gen__v*`` / ``gen__final``), each paired
         with a roster deck to pilot (cross-deck league + mirror self-play). There
         is ONE model now, so a pool entry is an explicit ``(gen_snapshot, deck)``
         pair — the same snapshot represents many opponent decks.
      3. the newest generalist snapshot piloting the learner's own current deck
         (the OpenAI-Five 'play the latest self' slot, chosen with probability
         ``self_play_frac``).
      4. frozen ARCHETYPE EXPLOITER snapshots (``exp_<arch>__v*`` /
         ``exp_<arch>__final``), each paired with a deck of the archetype it
         exploits, so the league inoculates the generalist against the alien
         styles a dedicated run discovered (``exploiter_floor`` guarantees them a
         minimum share even once the learner beats them).

    ``pinned_snapshots`` switches the pool into the fixed-opponent configuration
    an EXPLOITER run trains against: exactly those checkpoint files, each paired
    with every roster deck, with no snapshot rotation, no exploiter tier and no
    self-play branch — the frozen target the exploiter is supposed to beat.

    Memory is bounded exactly like :class:`OpponentPool`: the pool is capped at
    ``max(1, floor(max_checkpoint_ratio * n_envs))`` unique generalist snapshot
    files and sharded across processes by ``env_index``; scripted agents are free so
    every process keeps the full anchor set, and the latest-self snapshot is always
    resident (one model). Because the guaranteed per-deck anchors all reuse the one
    newest snapshot file, keeping every roster deck represented as an opponent is
    essentially free.

    Quality weighting for the historical-snapshot branch is supplied externally by
    the PFSPCallback via :meth:`set_weights` (a global win-rate estimate broadcast to
    every process); each process applies it to whatever sharded subset it holds — an
    accepted approximation, documented in
    docs/plan_pfsp_multimatchup_training.md. The weighting *mode* (pfsp vs softmax)
    lives entirely in the callback; the pool only consumes the resulting weights and
    falls back to uniform until the first broadcast arrives.
    """

    REFRESH_EVERY = 50  # episodes between snapshot rescans (pick up new vN files)

    def __init__(self, learner_deck: str, roster: Sequence[str], checkpoint_dir: str, *,
                 self_play_frac: float = 0.8, scripted_anchor_frac: float = 0.1,
                 rng: Optional[np.random.Generator] = None,
                 n_envs: int = 1, env_index: int = 0,
                 max_checkpoint_ratio: float = 1.0,
                 deterministic: bool = False, scripted_spec: str = "scripted",
                 self_decks: Optional[Sequence[str]] = None,
                 exploiter_floor: float = 0.0,
                 pinned_snapshots: Optional[Sequence[str]] = None):
        self._learner = learner_deck
        self._roster = list(roster) or [learner_deck]
        self._dir = checkpoint_dir
        self._self_play_frac = float(self_play_frac)
        self._anchor_frac = float(scripted_anchor_frac)
        self._exploiter_floor = float(exploiter_floor)
        # Pinned mode is selected by PASSING the argument at all (even empty): an
        # exploiter run with no generalist to freeze must face the scripted anchor
        # only, never a rediscovered snapshot pool.
        self._pinned_mode = pinned_snapshots is not None
        self._pinned = [p for p in (pinned_snapshots or []) if p]
        self._rng = rng if rng is not None else np.random.default_rng()
        self._n_envs = max(1, n_envs)
        self._env_index = env_index
        self._ratio = max_checkpoint_ratio
        self._deterministic = deterministic
        self._scripted_spec = scripted_spec
        # Mixed-self-deck mode: when ``self_decks`` is given the learner's OWN deck
        # also varies per episode, cycled (per-pool shuffled round-robin, reshuffled
        # each full pass so every deck is piloted equally often within a rollout).
        # ``None`` keeps the classic fixed-self-deck behavior (the learner always
        # pilots ``learner_deck``; the driver rotates the deck between chunks).
        self._self_decks = list(self_decks) if self_decks else None
        self._mixed = self._self_decks is not None
        self._self_cycle: list[str] = []
        self._self_cycle_idx = 0
        self._cache: dict[str, Controller] = {}    # path -> ModelController
        # (opp_deck, label) aggregate weights AND, in mixed mode, (self_deck,
        # opp_deck, label) matchup weights layered on top — see _entry_weights.
        self._weights: dict = {}
        self._snap_entries: list[tuple[str, str]] = []  # [(path, deck)] this process holds
        # Indices into _snap_entries whose checkpoint is an archetype exploiter
        # (the subset the exploiter floor samples from).
        self._exp_indices: list[int] = []
        self._latest_self: Optional[str] = None    # learner's newest snapshot
        self._total_snaps = 0                      # total snapshots across roster (auto-ramp)
        self._episode = 0
        self._scripted_controller: Optional[Controller] = None
        self.refresh()

    # ── snapshot discovery / sharding ────────────────────────────────────────
    def refresh(self):
        """Rescan the checkpoint dir for gen snapshots, re-shard, and re-find latest-self.

        There is ONE model now, so pool entries are ``(gen_snapshot, deck)`` pairs:
        a generalist snapshot paired with a roster deck for it to pilot. Bounded to
        ``max_unique`` unique snapshot FILES (per-process memory) and built in two
        tiers for league *fairness*:

          * Tier 1 (guaranteed) — every roster deck paired with the newest
            generalist snapshot (== ``gen__final`` when present). All these entries
            reuse the one newest file, so keeping every roster deck represented as
            an opponent costs a single resident model.
          * Tier 2 (discretionary) — the remaining older ``gen__v*`` snapshots and
            older ``exp_*__v*`` exploiter snapshots, newest-first and interleaved,
            each assigned a deck round-robin so historical snapshots face varied
            decks. Capped so the pool holds at most ``max_unique`` unique snapshot
            files.
          * Tier 3 (guaranteed) — the newest snapshot of every archetype exploiter
            that exists, paired with the decks of the archetype it exploits, so an
            exploiter that has been trained is always in the field.

        ``pinned_snapshots`` short-circuits all of it: the pool is exactly those
        files x the roster, unsharded (a pinned run holds one or two files, so
        every process can face every roster deck) and with no self-play slot.
        """
        if self._pinned_mode:
            self._total_snaps = len(self._pinned)
            self._latest_self = None   # no latest-self slot against a frozen target
            self._snap_entries = [(path, deck) for path in self._pinned
                                  for deck in self._roster]
            self._exp_indices = []
            return

        snaps = gen_snapshots(self._dir)             # oldest..newest, gen__final last
        self._total_snaps = len(snaps)
        # The newest generalist snapshot pilots the learner in the latest-self slot.
        self._latest_self = snaps[-1] if snaps else None
        newest = snaps[-1] if snaps else None
        exp_by_arch = exploiter_snapshots(self._dir)

        # Tier 1: every roster deck paired with the newest snapshot (one shared file).
        guaranteed = [(newest, deck) for deck in self._roster] if newest else []

        # Tier 3: the newest snapshot of each exploiter, piloting its archetype's
        # decks. Never evicted (an exploiter the learner has learned to beat is
        # still the only representative of its style in the pool).
        exp_files: list[str] = []
        for arch in sorted(exp_by_arch):
            exp_newest = exp_by_arch[arch][-1]
            exp_files.append(exp_newest)
            guaranteed += [(exp_newest, deck)
                           for deck in self._exploiter_decks(arch)]

        # Tier 2: older snapshots newest-first, each assigned a deck. Bound the pool
        # to max_unique unique FILES; the guaranteed files (newest gen + one per
        # exploiter) are already resident, so the rest of the budget goes to older
        # files, interleaved so a long gen history can't starve the exploiters.
        max_unique = max(1, int(math.floor(self._ratio * self._n_envs)))
        roster = self._roster
        gen_older = [(snap, roster) for snap in reversed(snaps[:-1])]
        exp_older = [(snap, self._exploiter_decks(arch))
                     for arch in sorted(exp_by_arch)
                     for snap in reversed(exp_by_arch[arch][:-1])]
        budget = max(0, max_unique - len(exp_files) - (1 if newest else 0))
        older = _interleave(gen_older, exp_older)[:budget]
        discretionary = [(snap, decks[i % len(decks)])
                         for i, (snap, decks) in enumerate(older)]

        # Shard the pooled set across processes (each env keeps ~max_unique/n_envs).
        active = guaranteed + discretionary
        self._snap_entries = active[self._env_index % self._n_envs::self._n_envs]
        self._exp_indices = [i for i, (path, _deck) in enumerate(self._snap_entries)
                             if is_exploiter_path(path)]

    def _exploiter_decks(self, arch: str) -> list[str]:
        """Decks an ``arch`` exploiter pilots as a pool opponent.

        Its own archetype's decks, preferring the ones in this pool's roster (so a
        league opponent plays a roster deck); if the archetype has no roster deck,
        its tagged decks are used as-is rather than dropping the exploiter."""
        decks = archetypes.decks_for_archetype(arch)
        in_roster = [d for d in decks if d in self._roster]
        return in_roster or decks

    def _maybe_refresh(self):
        self._episode += 1
        if self._episode % self.REFRESH_EVERY == 0:
            self.refresh()

    def _next_self_deck(self) -> str:
        """Deck the learner pilots this episode.

        Fixed mode: always ``learner_deck``. Mixed mode: the next deck from a
        per-pool shuffled cycle over ``self_decks``, reshuffled at each full pass
        (deterministic given the pool's rng, so a seeded run replays identically)."""
        if not self._mixed:
            return self._learner
        if self._self_cycle_idx >= len(self._self_cycle):
            self._self_cycle = list(self._self_decks)
            self._rng.shuffle(self._self_cycle)
            self._self_cycle_idx = 0
        deck = self._self_cycle[self._self_cycle_idx]
        self._self_cycle_idx += 1
        return deck

    # ── weights pushed from the PFSPCallback ─────────────────────────────────
    def set_weights(self, weights):
        """Replace the historical-snapshot weighting.

        Keys are ``(opp_deck, label)`` aggregate weights and, in mixed mode, also
        ``(self_deck, opp_deck, label)`` matchup weights layered on top (the
        callback broadcasts a matchup key only once it has enough decisive samples).
        ``_entry_weights`` prefers the matchup key and falls back to the aggregate."""
        if weights:
            self._weights = dict(weights)

    # ── controllers ──────────────────────────────────────────────────────────
    def _scripted(self) -> Controller:
        if self._scripted_controller is None:
            self._scripted_controller = ScriptedController(
                make_agent(self._scripted_spec), label=self._scripted_spec)
        return self._scripted_controller

    def _model_for(self, path: str) -> Controller:
        ctrl = self._cache.get(path)
        if ctrl is None:
            ctrl = ModelController(_load_model(path), label=os.path.basename(path),
                                   deterministic=self._deterministic)
            self._cache[path] = ctrl
        return ctrl

    # ── auto-ramp (bootstrap / curriculum) ───────────────────────────────────
    def _effective_self_play_frac(self) -> float:
        """Scale the self-play fraction with how full the pool is.

        No learner snapshot yet -> 0.0 (face only scripted/other-deck snapshots);
        grows linearly toward ``self_play_frac`` and saturates once there is at
        least one snapshot per roster deck. Makes the warm-start-vs-scripted
        curriculum automatic rather than a manual phase switch."""
        if self._latest_self is None:
            return 0.0
        target = max(1, len(self._roster))
        return self._self_play_frac * min(1.0, self._total_snaps / float(target))

    # ── per-episode sampling ─────────────────────────────────────────────────
    def sample_episode(self) -> tuple[str, str, str, Controller]:
        """Return ``(self_deck, opp_deck, label, controller)`` for this episode.

        ``self_deck`` is the deck the LEARNER pilots this episode — the fixed
        ``learner_deck`` in classic mode, or the next deck from the shuffled cycle
        in mixed mode. It is returned as a 4-tuple in BOTH modes so the env wrapper
        has one call shape."""
        self._maybe_refresh()
        self_deck = self._next_self_deck()
        # 1. latest-self slot (fast learning vs the current frontier) — a true
        #    mirror: the newest snapshot piloting THIS episode's self deck.
        if (self._latest_self is not None
                and self._rng.random() < self._effective_self_play_frac()):
            return (self_deck, self_deck, os.path.basename(self._latest_self),
                    self._model_for(self._latest_self))
        # 2. scripted anchor — a fixed floor carved out of the historical share so
        #    the collapse guard never vanishes (also the cold-start fallback when
        #    this process holds no snapshots).
        if (not self._snap_entries) or self._rng.random() < self._anchor_frac:
            deck = self._roster[int(self._rng.integers(len(self._roster)))]
            return self_deck, deck, self._scripted_spec, self._scripted()
        # 2b. exploiter floor — a share reserved for the archetype exploiters, so
        #     the styles a dedicated run discovered keep showing up even once PFSP
        #     stops preferring them (the learner beats them).
        if self._exp_indices and self._rng.random() < self._exploiter_floor:
            return self._sample_entry(self_deck, self._exp_indices)
        # 3. weighted historical snapshot (cross-deck league + mirror self-play,
        #    exploiters included in the normal PFSP weighting).
        return self._sample_entry(self_deck, None)

    def _sample_entry(self, self_deck: str, indices) -> tuple[str, str, str, Controller]:
        """PFSP-sample one snapshot entry (optionally restricted to ``indices``)."""
        pool = self._snap_entries if indices is None else [self._snap_entries[i]
                                                           for i in indices]
        weights = self._entry_weights(self_deck, indices)
        path, deck = pool[int(self._rng.choice(len(pool), p=weights))]
        return self_deck, deck, os.path.basename(path), self._model_for(path)

    def _entry_weights(self, self_deck: str, indices=None) -> np.ndarray:
        """Normalised PFSP/softmax weights for this process's snapshot entries.

        Weighting is matchup-aware: for each ``(path, opp_deck)`` entry we prefer
        the matchup-keyed weight ``(self_deck, opp_deck, label)`` when the callback
        has broadcast one (enough decisive games), else the opponent-aggregate
        weight ``(opp_deck, label)``, else the current max weight so unseen entries
        still get tried (OpenAI-Five 'init new snapshot quality to the max' rule).
        In fixed mode the weights dict has no matchup keys, so this collapses to
        the old aggregate-only behavior.

        ``indices`` restricts the weighting to a subset of the entries (the
        exploiter floor weights the exploiter entries among themselves)."""
        default = max(self._weights.values()) if self._weights else 1.0
        entries = (self._snap_entries if indices is None
                   else [self._snap_entries[i] for i in indices])
        w = []
        for path, deck in entries:
            label = os.path.basename(path)
            val = self._weights.get((self_deck, deck, label))
            if val is None:
                val = self._weights.get((deck, label), default)
            w.append(val)
        w = np.clip(np.array(w, dtype=float), 1e-8, None)
        return w / w.sum()
