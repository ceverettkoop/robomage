"""AlphaZero self-play data generation for ONE deck (mirror match, bo1 or bo3).

Each worker owns a :class:`search_env.SearchRoboMageEnv` (deck vs itself) and one
shared :class:`az_net.AZNet` piloting BOTH seats. At every decision that is
loop-safe (``env.last_search_safe``) with >1 choice it runs a determinized PUCT
search (:func:`mcts.run_search`) with root Dirichlet noise; unsafe / trivial
decisions fall back to the net's raw-policy argmax. For each SEARCHED decision it
stores (obs, visit-distribution pi, legal mask, mover seat, game index).

In bo3 mode a "game" of generation is actually a best-of-three MATCH: each match
yields up to three games of samples, and each sample's outcome z (+1/-1 from that
mover's perspective, 0 on a draw) is the result of the PARTICULAR game the
decision belonged to — not the match result. Samples are written to
``az_data/{deck}/shard_{ts}_{pid}_{n}.npz``.

Run standalone (``az_selfplay.py --deck delver --games 2 --sims 12 --worlds 2``)
or via ``train.py az-selfplay`` (bo1); the ``train.py az`` / ``az-league`` cycles
drive it in bo3.

``generate(scripted_opponent_frac=f)`` makes a fraction ``f`` of matches play the
net+MCTS (focus seat) against the rule-based scripted:hard agent (opponent seat)
instead of against itself; only the net seat's decisions become samples
(:func:`_play_match`'s ``agent``/``net_is_a``). ``f=1.0`` trains entirely
against the scripted agent. That path is Python-backend only (the C++ actor is pure self-play).
``generate(exhaustive=True)`` plays the exact matchup matrix instead of a random
draw, splitting HYBRID across both backends: actor for the pure self-play cells,
Python for the vs-scripted cells (see :func:`_generate_hybrid`).
``generate(exhaustive_selfplay=True)`` narrows that matrix to the pure self-play
cells only (one match per unordered deck pair, no scripted/Python component), and
``exhaustive_repeats=n`` plays every cell of the matrix n times.
``generate(scripted_cells=k, slot=s)`` adds a ROTATING slice of k vs-scripted
cells back onto that self-play matrix — the k ordered (focus, opponent) pairs
starting at offset ``(s * k) % n_ordered``, wrapping — so successive az-league
slots tile every ordered pair while the bulk of each slot stays on the actor.

:func:`generate_expert` additionally writes EXPERT demonstration shards —
scripted:hard piloting both seats, pi = one-hot on the expert's action — in the
same shard format, so the trainer behavior-clones hand-coded lines (e.g. the
Doomsday combo) that neither PPO exploration nor prior-guided search discovers.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import namedtuple
from typing import Optional

import numpy as np

try:
    from env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX, _IS_SIDEBOARD_IDX
    from cli_spec import (BIN_DIR, INTERACTIVE_BUILD_DIR, INTERACTIVE_BINARY,
                          DEFAULT_SB_SIMS, DEFAULT_SB_WORLDS,
                          DEFAULT_SB_MAX_DEPTH, DEFAULT_SB_ROLLOUT_TURNS,
                          DEFAULT_SB_PERSIST, DEFAULT_AZ_GAMES, DEFAULT_AZ_SIMS,
                          DEFAULT_AZ_WORLDS, DEFAULT_AZ_MIRROR_FRAC,
                          DEFAULT_AZ_TEMP_MOVES, DEFAULT_AZ_TD_N,
                          DEFAULT_AZ_EXHAUSTIVE_REPEATS,
                          DEFAULT_AZ_SCRIPTED_CELLS)
    from opponents import GEN_STEM
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX, _IS_SIDEBOARD_IDX
    from train.cli_spec import (BIN_DIR, INTERACTIVE_BUILD_DIR, INTERACTIVE_BINARY,
                                DEFAULT_SB_SIMS, DEFAULT_SB_WORLDS,
                                DEFAULT_SB_MAX_DEPTH, DEFAULT_SB_ROLLOUT_TURNS,
                                DEFAULT_SB_PERSIST, DEFAULT_AZ_GAMES,
                                DEFAULT_AZ_SIMS, DEFAULT_AZ_WORLDS,
                                DEFAULT_AZ_MIRROR_FRAC, DEFAULT_AZ_TEMP_MOVES,
                                DEFAULT_AZ_TD_N,
                                DEFAULT_AZ_EXHAUSTIVE_REPEATS,
                                DEFAULT_AZ_SCRIPTED_CELLS)
    from train.opponents import GEN_STEM

_AZ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "az_data")
# Self-play defaults to the RELEASE engine build for speed (see cli_spec.py's
# INTERACTIVE_BINARY doc comment); ROBOMAGE_BUILD overrides both this and the
# az_actor path below.
_ACTOR_BIN = os.path.join(INTERACTIVE_BUILD_DIR, "az_actor")
_DECKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "bin", "resources", "decks")
_LEAGUE_DECKS_DIR = os.path.join(_DECKS_DIR, "league")

# Defaults (AlphaZero-style)
DEFAULT_ROOT_NOISE_EPS = 0.25
DEFAULT_ROOT_NOISE_ALPHA = 1.0
# sample-from-visits for the first N real decisions, then argmax; value lives
# in cli_spec's AZ-defaults block (one home), local alias kept for callers.
DEFAULT_TEMP_MOVES = DEFAULT_AZ_TEMP_MOVES
# Sideboard-rooted search budget (DEFAULT_SB_SIMS/WORLDS/MAX_DEPTH) lives in
# cli_spec — the single home shared with opponents.SearchController and the CLI
# flag defaults — and is imported above.
# P(opponent deck == focus deck) per self-play game; value in cli_spec too.
DEFAULT_MIRROR_FRAC = DEFAULT_AZ_MIRROR_FRAC
# n-step TD horizon baked into each shard's td_q column; value in cli_spec too.
DEFAULT_TD_N = DEFAULT_AZ_TD_N
FLUSH_SAMPLES = 4096           # write a shard once this many samples accumulate
HEARTBEAT_MOVES = 25           # Python backend: mid-game progress line every N decisions

# The shard schema, in the order arrays are written. BOTH backends (this module
# and src/actor) must write exactly these keys; az_train.load_window fails loudly
# on a shard missing any of them.
SHARD_KEYS = ("obs", "pi", "z", "mask", "q", "explored", "td_q")


def _fmt_secs(s: float) -> str:
    return f"{s / 60:.1f}m" if s >= 90 else f"{s:.0f}s"


# ----------------------------------------------------------------------
# Matchup schedule (mirrors + cross-deck, seeded/reproducible)
# ----------------------------------------------------------------------

def league_roster() -> list:
    """Every deck in decks/league/, referenced 'league/<stem>' (sorted)."""
    if not os.path.isdir(_LEAGUE_DECKS_DIR):
        return []
    return sorted("league/" + os.path.splitext(p)[0]
                  for p in os.listdir(_LEAGUE_DECKS_DIR) if p.endswith(".dk"))


def build_matchup_schedule_ex(focus_decks, opponent_decks, games: int,
                              mirror_frac: float, seed: int) -> list:
    """Deterministic per-game (deck_a, deck_b, focus_is_a) schedule.

    The full form of :func:`build_matchup_schedule`, additionally reporting WHICH
    SEAT the focus deck took — the caller needs it to know which seat the net
    pilots when a scripted opponent takes the other one. The RNG draws are
    identical to (and shared with) the pair-only form, so both build the same
    schedule for a given seed."""
    rng = np.random.default_rng(seed)
    focus = list(focus_decks or [])
    pool = list(opponent_decks or [])
    sched = []
    for _ in range(games):
        # Single-focus: don't consume an RNG draw (preserves old schedules exactly).
        fdeck = focus[0] if len(focus) == 1 else focus[int(rng.integers(len(focus)))]
        if not pool or rng.random() < mirror_frac:
            opp = fdeck
        else:
            opp = pool[int(rng.integers(len(pool)))]
        focus_is_a = bool(rng.random() < 0.5)
        pair = (fdeck, opp) if focus_is_a else (opp, fdeck)
        sched.append((pair[0], pair[1], focus_is_a))
    return sched


def ordered_matchup_pairs(focus_decks, opponent_decks) -> list:
    """The full ORDERED ``(focus, opponent)`` pair list of the exhaustive matrix.

    The vs-SCRIPTED family of :func:`build_exhaustive_schedule_ex` — 100 pairs on
    the 10-deck roster — in a fixed, seed-independent order. Shared with the
    rotating ``scripted_cells`` slice so both name the same cell list."""
    focus = list(focus_decks or [])
    pool = list(opponent_decks or []) or list(focus)
    return [(f, o) for f in focus for o in pool]


def rotating_scripted_pairs(focus_decks, opponent_decks, k: int,
                            slot: int = 0) -> list:
    """``k`` ordered (focus, opponent) pairs for ``slot``, rotating with wraparound.

    Slot ``s`` takes the ``k`` consecutive pairs of
    :func:`ordered_matchup_pairs` starting at offset ``(s * k) % n_ordered``,
    wrapping around the end of the list — so consecutive slots tile the whole
    ordered pair list and coverage accumulates across an az-league run instead of
    re-playing the same cells every slot. ``k <= 0`` (or an empty matrix) yields
    nothing; ``k > n_ordered`` walks the list more than once."""
    pairs = ordered_matchup_pairs(focus_decks, opponent_decks)
    k = int(k)
    if k <= 0 or not pairs:
        return []
    n = len(pairs)
    off = (int(slot) * k) % n
    return [pairs[(off + i) % n] for i in range(k)]


def build_exhaustive_schedule_ex(focus_decks, opponent_decks, seed: int,
                                 include_scripted: bool = True,
                                 repeats: int = 1, scripted_cells: int = 0,
                                 slot: int = 0) -> tuple:
    """The EXHAUSTIVE matchup matrix: every cell exactly ``repeats`` times, no
    random draw.

    Two families of cells, one bo3 match each:

    * every ORDERED ``(focus, opponent)`` pair — a vs-SCRIPTED cell: the net+MCTS
      pilots the focus deck, scripted:hard the opponent deck. Ordered because the
      two seats produce different training data (only the net seat is sampled),
      so "net pilots X vs scripted Y" and "net pilots Y vs scripted X" are both
      played. Mirrors included: ``len(focus) * len(pool)`` cells.
    * every UNORDERED deck pair (mirrors included) — a pure SELF-PLAY cell: the
      one generalist pilots both seats, so (X, Y) and (Y, X) are the same cell.

    On the 10-deck league roster that is 100 + 55 = 155 matches. ``repeats=n``
    plays every cell ``n`` times (each repeat is an independent match — its own
    seat draw and, downstream, its own game seed), so the matrix stays exact
    while the per-cell sample count scales. ``include_scripted=False`` drops the
    vs-scripted family entirely, leaving ONLY the pure self-play cells (55 per
    repeat on the 10-deck roster) — a matrix with no Python/scripted component,
    so the whole schedule runs on the C++ actor.

    ``scripted_cells=k`` (self-play-only matrices only — it is inert when
    ``include_scripted`` already plays every ordered pair) adds back a ROTATING
    slice of the vs-scripted family: the ``k`` ordered pairs
    :func:`rotating_scripted_pairs` gives this ``slot``, so a run's slots tile the
    whole ordered list over time. They are added ONCE per call (an absolute
    per-slot count, NOT multiplied by ``repeats``) and marked exactly like the
    full matrix's scripted cells, so the hybrid backend routes them to Python
    while the actor keeps the self-play cells.

    The seeded RNG randomizes only each cell's seat assignment and the
    interleaving of the two families (scripted cells cost less wall-clock than
    searched-both-seats cells, so shuffling balances the contiguous worker
    slices); the CELL SET is exact and deterministic regardless of seed.

    Returns ``(sched_ex, scripted_seats)`` — the same shapes ``generate`` builds
    from :func:`build_matchup_schedule_ex` + its scripted-fraction draw: a list of
    ``(deck_a, deck_b, focus_is_a)`` and a parallel list of per-match ``net_is_a``
    (None = pure self-play)."""
    repeats = max(1, int(repeats))
    rng = np.random.default_rng(seed)
    focus = list(focus_decks or [])
    pool = list(opponent_decks or []) or list(focus)
    cells = []                       # (fdeck, opp, net_is_a_or_None)
    if include_scripted:
        for f, o in ordered_matchup_pairs(focus, pool):   # every ordered pair
            cells.append((f, o, True))
    seen = set()
    for f in focus:                  # unordered: one pure self-play cell per pair
        for o in pool:
            key = tuple(sorted((f, o)))
            if key not in seen:
                seen.add(key)
                cells.append((f, o, None))
    cells = cells * repeats
    if not include_scripted:
        # Rotating vs-scripted slice: k matches for THIS slot, added after the
        # repeats multiplication so k is an exact per-slot count.
        cells += [(f, o, True)
                  for f, o in rotating_scripted_pairs(focus, pool,
                                                      scripted_cells, slot)]
    rng.shuffle(cells)
    sched_ex, seats = [], []
    for fdeck, opp, scripted in cells:
        focus_is_a = bool(rng.random() < 0.5)
        pair = (fdeck, opp) if focus_is_a else (opp, fdeck)
        sched_ex.append((pair[0], pair[1], focus_is_a))
        seats.append(focus_is_a if scripted else None)
    return sched_ex, seats


def build_matchup_schedule(focus_decks, opponent_decks, games: int,
                           mirror_frac: float, seed: int) -> list:
    """Deterministic per-game (deck_a, deck_b) schedule for one generation run.

    Each game a FOCUS deck is drawn uniformly from ``focus_decks`` (a single-deck
    list reduces to the classic single-focus run); its opponent is that same deck
    (mirror) with probability ``mirror_frac``, else a uniform draw from
    ``opponent_decks``. The focus deck's seat (A vs B) is randomized per game so
    seat-A bias doesn't accumulate. The one generalist net values every state, so
    cross-deck negamax backup in MCTS is sound. Seeded RNG → the schedule replays
    identically for a given seed (and, for a single-deck ``focus_decks``, byte-
    identically to the pre-matrix schedule — no extra RNG draw is consumed).

    The seat projection of :func:`build_matchup_schedule_ex`."""
    return [(a, b) for a, b, _ in
            build_matchup_schedule_ex(focus_decks, opponent_decks, games,
                                      mirror_frac, seed)]


def _schedule_summary(schedule: list) -> str:
    from collections import Counter
    c = Counter(schedule)
    parts = [f"{a}|{b}:{n}" for (a, b), n in sorted(c.items())]
    return ", ".join(parts)


# ----------------------------------------------------------------------
# Checkpoint resolution (parent decides once; workers load their own copy)
# ----------------------------------------------------------------------

def resolve_source(deck: str, checkpoint: Optional[str]) -> dict:
    """Pick the net source for the ONE generalist AZ net: an explicit AZ/PPO
    checkpoint, else the generalist AZ checkpoint (newest ``gen__azv*``
    CANDIDATE snapshot, falling back to ``gen__azfinal``), else a warm-start
    from the generalist PPO checkpoint (``gen__final``), else random init.

    Self-play prefers the CANDIDATE line (AlphaZero-style: the latest net
    generates the data), not the gate-promoted incumbent: with the incumbent as
    generator, every gate rejection froze the data distribution, so a stretch
    of kept-incumbent cycles just re-distilled a fixed policy. ``gen__azfinal``
    remains what serving/eval specs (``az:gen``) and the gate opponent resolve
    to — only the data generator tracks the candidate.

    ``deck`` (the FOCUS deck) is not used for net resolution — the net is the
    single generalist, and the deck it pilots travels through the matchup
    schedule, not the checkpoint name.

    Returns a spec dict {mode: 'az'|'ppo'|'random', path: str|None} — picklable so
    each worker reconstructs its own net (torch objects don't cross processes).

    Deliberately NOT the shared ladder ``opponents.load_az_evaluator``: that one
    serves the SERVING contract (``prefer="final"`` — the gate-promoted
    incumbent — and returns a live AZEvaluator). Self-play needs the CANDIDATE
    line (``prefer="snapshot"``), a random-init fallback the serving ladder has
    no rung for, and a picklable dict rather than a torch object. Keep the two in
    sync in SPIRIT (AZ path -> PPO warm-start -> …), not in code."""
    from az_net import resolve_az_checkpoint
    from opponents import resolve_checkpoint

    if checkpoint:
        if checkpoint.endswith(".pt") or resolve_az_checkpoint(checkpoint):
            az = resolve_az_checkpoint(checkpoint) or checkpoint
            return {"mode": "az", "path": az}
        return {"mode": "ppo", "path": resolve_checkpoint(checkpoint)}
    az = resolve_az_checkpoint(GEN_STEM, prefer="snapshot")
    if az:
        return {"mode": "az", "path": az}
    ppo = resolve_checkpoint(GEN_STEM)
    if ppo and os.path.exists(ppo):
        return {"mode": "ppo", "path": ppo}
    return {"mode": "random", "path": None}


def _build_net(source: dict):
    # The worker-side half of resolve_source's dict; see that docstring for why
    # this stays separate from opponents.load_az_evaluator (candidate line,
    # random-init rung, picklable spec, bare net rather than an AZEvaluator).
    from az_net import AZNet, load_az, from_ppo
    if source["mode"] == "az":
        return load_az(source["path"])
    if source["mode"] == "ppo":
        return from_ppo(source["path"])
    return AZNet().eval()


# ----------------------------------------------------------------------
# Shared sample builders / boundary helpers
# ----------------------------------------------------------------------

def winner_from_reward(reward) -> Optional[str]:
    """The winner ('A'/'B'/None for a draw) implied by a step's reward delta.

    The env's reward is always in PLAYER-A perspective (env.py's bo3 reward is
    ±BO3_GAME_WIN_REWARD per game), so the sign alone names the winner. One
    home for the rule, shared by every self-play/expert/recording loop that
    reads a game boundary off ``(reward, info['game_result'])``."""
    return "A" if reward > 0 else ("B" if reward < 0 else None)


def _drop_unfinished(samples, game_winners):
    """Drop samples belonging to a game that never finished, returning
    ``(kept, dropped)``.

    Used when a match TRUNCATED mid-game: the games that completed have a
    winner (hence a valid z), the in-progress one does not. ``shard_record``
    deliberately does NOT use this — it keeps in-progress rows at z = 0 so a
    recording is browsable mid-game (see shard_record's module docstring)."""
    n_done = len(game_winners)
    kept = [s for s in samples if s["game_idx"] < n_done]
    return kept, len(samples) - len(kept)


def sample_from_search_result(obs, num_choices, result):
    """Build a trainer-schema sample dict from a finished search at ``obs``.

    ``pi`` is the root visit posterior (``result.policy_target(1.0)``) padded
    into a MAX_ACTIONS row, ``mask`` the legal prefix, ``q`` the search's root
    value, ``mover_is_a`` the priority seat read straight off the observation
    (``_SELF_IS_A_IDX`` — identical to the play loops' ``priority_is_a``).
    ``explored`` starts at 0; the CALLER finalizes it once it knows which
    action was actually played, and attaches ``game_idx``.

    Shared by :func:`_play_match` and ``shard_record.ShardRecorder`` so a
    recorded searched row is bit-identical to a self-play one by construction.
    ``np.asarray(obs, float32).copy()`` is exactly ``env._obs.copy()`` for the
    float32 observation the env hands out."""
    num_choices = int(num_choices)
    pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
    pi[:num_choices] = result.policy_target(1.0).astype(np.float32)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    mask[:num_choices] = True
    return {"obs": np.asarray(obs, dtype=np.float32).copy(), "pi": pi,
            "mask": mask, "mover_is_a": bool(obs[_SELF_IS_A_IDX] > 0.5),
            "q": float(result.root_value), "explored": 0}


def one_hot_sample(obs, num_choices, action):
    """Build a BEHAVIOR-CLONING sample dict: pi is a one-hot on ``action``.

    No search ran, so there is no root value to bootstrap from: ``q = NaN`` and
    ``explored = 0`` make :func:`compute_td_targets` leave ``td_q == z`` on the
    row (the demonstration's own outcome is the whole signal). The caller
    attaches ``game_idx``. Shared by :func:`_play_match_expert` and
    ``shard_record.ShardRecorder``."""
    num_choices = int(num_choices)
    pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
    pi[int(action)] = 1.0
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    mask[:num_choices] = True
    return {"obs": np.asarray(obs, dtype=np.float32).copy(), "pi": pi,
            "mask": mask, "mover_is_a": bool(obs[_SELF_IS_A_IDX] > 0.5),
            "q": float("nan"), "explored": 0}


# Per-match search knobs, built once per match and passed to
# :func:`_search_and_sample` (which must stay a pure re-spelling of the
# in-loop block it was extracted from — see its BIT CONTRACT note).
_MatchKnobs = namedtuple("_MatchKnobs", (
    "sims", "worlds", "temp_moves", "root_noise_eps", "root_noise_alpha",
    "sb_sims", "sb_worlds", "sb_max_depth", "sb_rollout_turns", "sb_persist",
    "merge_dupes"))


def _search_and_sample(env, evaluator, rng, knobs, *, num_choices,
                       priority_is_a, game_idx, game_move, sb_boundary,
                       sb_stats):
    """Run the search at the current decision and turn it into (action, sample,
    sb_boundary).

    BIT CONTRACT: this is the verbatim searched-decision block of
    :func:`_play_match` (statement order around ``rng`` draws unchanged — the
    single draw is the temperature ``rng.choice``), so extracting it leaves the
    sampled play stream byte-identical. ``sb_stats`` is mutated in place with
    the boundary-persistence counters; the returned ``sb_boundary`` replaces the
    caller's (None outside a sideboard root)."""
    from mcts import run_search, sb_root_key, walk_reuse_root

    # A bo3 sideboard root is more expensive per sim (restore re-crosses
    # init_ecs + deck load + shuffle) and has a game-long horizon, so it
    # gets its own budget. Key ONLY off is_sideboard_phase — is_post_board
    # / game_number still reflect the just-ended game at a g1->g2 root.
    if bool(env._obs[_IS_SIDEBOARD_IDX] > 0.5):
        kw = dict(sims=knobs.sb_sims, worlds=knobs.sb_worlds,
                  max_depth=knobs.sb_max_depth,
                  rollout_turns=knobs.sb_rollout_turns,
                  root_noise_eps=knobs.root_noise_eps,
                  root_noise_alpha=knobs.root_noise_alpha, rng=rng,
                  merge_dupes=knobs.merge_dupes)
        if knobs.sb_persist:
            key = sb_root_key(env._obs)
            b = sb_boundary
            if b is not None and b["key"] == key:
                walked = [walk_reuse_root(r, b["played"], num_choices,
                                          priority_is_a)
                          for r in b["roots"]]
                kw.update(world_seeds=b["seeds"], reuse_roots=walked)
            else:
                b = {"key": key, "seeds": None, "roots": None,
                     "played": [], "picks": [], "memo": {}}
                sb_boundary = b
            kw.update(rollout_memo=b["memo"],
                      memo_picks=tuple(b["picks"]))
            result = run_search(env, evaluator, **kw)
            b["seeds"] = result.seeds
            b["roots"] = result.roots
            b["played"] = []
            sb_stats["sb_reused_visits"] += result.reused_visits
            sb_stats["sb_memo_hits"] += result.memo_hits
        else:
            result = run_search(env, evaluator, **kw)
    else:
        sb_boundary = None
        result = run_search(env, evaluator, sims=knobs.sims,
                            worlds=knobs.worlds,
                            root_noise_eps=knobs.root_noise_eps,
                            root_noise_alpha=knobs.root_noise_alpha, rng=rng,
                            merge_dupes=knobs.merge_dupes)
    visits = result.policy_target(1.0)              # normalized visit counts
    sample = sample_from_search_result(env._obs, num_choices, result)
    sample["game_idx"] = game_idx
    # Temperature schedule is per-game: the first temp_moves decisions of
    # EACH game sample from the visit counts, then switch to argmax.
    if game_move < knobs.temp_moves:
        action = int(rng.choice(num_choices, p=visits))
    else:
        action = result.best_action()
    # An action other than the visit argmax is an EXPLORATORY move: it
    # truncates every earlier sample's n-step bootstrap window (see
    # compute_td_targets), because what follows is no longer the line the
    # search endorsed.
    sample["explored"] = int(action != result.best_action())
    return action, sample, sb_boundary


def _sb_latch_played(sb_boundary, obs, action):
    """Latch a stepped action into the live sideboard boundary (no-op when
    there is none). The walk plays the latched actions through the stored trees
    at the next pick; only sideboard picks contribute memo descriptors. Takes
    the PRE-step obs, matching the actor. A scripted seat's actions latch too —
    the walk must replay the TRUE action sequence, whoever played it."""
    if sb_boundary is None:
        return
    from mcts import sb_pick_descriptor
    sb_boundary["played"].append(int(action))
    d = sb_pick_descriptor(obs, int(action))
    if d is not None:
        sb_boundary["picks"].append(d)


# ----------------------------------------------------------------------
# One game of self-play
# ----------------------------------------------------------------------

def _play_match(env, evaluator, rng, *, sims, worlds, temp_moves,
                root_noise_eps, root_noise_alpha, seed, on_progress=None,
                sb_sims=DEFAULT_SB_SIMS, sb_worlds=DEFAULT_SB_WORLDS,
                sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                sb_persist=bool(DEFAULT_SB_PERSIST),
                merge_dupes=True, agent=None, net_is_a=None):
    """Play one match (bo1: a single game; bo3: a best-of-three) and return
    (samples, game_winners, searched, fallback, dropped, sb_stats) — sb_stats
    reports the boundary persistence's reused visits + rollout-memo hits.

    ``samples`` is a list of dicts {obs, pi, mask, mover_is_a, game_idx}; each is
    tagged with the 0-based index of the game it was played in. ``game_winners``
    is the ordered list of each COMPLETED game's winner ('A'/'B'/None for a draw),
    so ``game_winners[sample['game_idx']]`` prices that sample. ``dropped`` counts
    samples discarded because the match TRUNCATED mid-game (the in-progress game
    has no result, so its samples carry no valid z).

    Default (``net_is_a=None``) is PURE SELF-PLAY: the net+MCTS pilots both
    seats. Passing ``net_is_a`` (with ``agent`` = a rule-based scripted:hard
    agent) puts the net+MCTS on that ONE seat and the agent on the other. Only
    the NET seat's decisions are then searched and recorded as samples — the
    scripted seat is environment, not a policy to imitate. That is sound for the
    value target as-is: :func:`_backfill_and_pack` prices every sample per-mover
    (z = +1 iff its game's winner is that sample's mover), so a one-seat sample
    stream needs no special handling. The ``agent`` branches consume no ``rng``
    and ``net_to_move`` is constant True when ``net_is_a is None``, so pure
    self-play draws exactly as it did before the two loops were merged.

    Game boundaries are detected from ``info['game_result']`` (the engine emitted a
    GAME_RESULT line on that step's read); the game's winner is the sign of that
    step's reward delta (+ -> A won, - -> B won), matching env.py's bo3 reward
    (±BO3_GAME_WIN_REWARD per game).

    Bo3 sideboard prompts BETWEEN games are now loop-safe MCTS roots (Stage 1-3):
    when searchable they enter ``samples`` like any other decision. They are keyed
    off the ``is_sideboard_phase`` state flag (``_IS_SIDEBOARD_IDX``) and searched
    with the SIDEBOARD budget (``sb_sims``/``sb_worlds``/``sb_max_depth``/
    ``sb_rollout_turns``) — heavier per step (each restore re-crosses init_ecs),
    deeper (the horizon is the whole next game), and with leaf rollouts on by
    default (each sim plays the raw policy to end-of-turn-``sb_rollout_turns``
    of the sampled next game before evaluating). Because ``game_move`` and ``game_idx`` are advanced at the preceding
    game's GAME_RESULT boundary, a sideboard sample naturally carries
    ``game_idx == k+1`` (the UPCOMING game) — so ``_backfill_and_pack`` prices it by
    that game's winner — and re-enters the per-game temperature schedule at
    ``game_move == 0``. (Do NOT key off ``is_post_board``/``game_number``: at a
    game-1->2 sideboard root those still reflect the ENDED game.)

    ``on_progress(move, searched, fallback)``, when given, fires every
    HEARTBEAT_MOVES decisions. Observation-only: it must not (and cannot)
    perturb the game or ``rng``, so play stays byte-identical with or without it.

    Pinned by ``test_sideboard_selfplay.py`` (tier ``sbselfplay``, in the default
    ``make check``; semantic, not bit-identity) and by ``test_actor_trains.py``
    (opt-in tier ``actor``; statistical — it replays this exact function and
    compares the resulting shards' schema/trainability against the C++ actor's).
    The hard constraint is therefore rng-DRAW-ORDER preservation: any edit must
    keep the statement order around ``rng`` draws intact so the sampled play
    stream is unchanged. Adding branches that consume no rng is fine."""
    obs, _ = env.reset(seed=seed)
    if agent is not None:
        agent.new_game()
    samples = []
    game_winners = []   # winner of each completed game, in order
    game_idx = 0        # index of the game currently in progress
    game_move = 0       # decisions made in the current game (temperature schedule)
    move = 0            # decisions made in the whole match (heartbeat/progress)
    done = False
    dropped = 0
    searched = 0
    fallback = 0
    # Sideboard-boundary persistence (mirrors the actor's --sb-persist): one
    # set of per-world trees + one rollout memo per boundary, seeds pinned at
    # the boundary's first search, every stepped action latched for the walk.
    sb_boundary = None
    sb_stats = {"sb_reused_visits": 0, "sb_memo_hits": 0}
    knobs = _MatchKnobs(sims=sims, worlds=worlds, temp_moves=temp_moves,
                        root_noise_eps=root_noise_eps,
                        root_noise_alpha=root_noise_alpha, sb_sims=sb_sims,
                        sb_worlds=sb_worlds, sb_max_depth=sb_max_depth,
                        sb_rollout_turns=sb_rollout_turns,
                        sb_persist=sb_persist, merge_dupes=merge_dupes)

    while not done:
        num_choices = env._num_choices
        priority_is_a = bool(env._obs[_SELF_IS_A_IDX] > 0.5)
        net_to_move = (True if net_is_a is None
                       else (priority_is_a == bool(net_is_a)))
        searchable = (net_to_move and bool(env.last_search_safe)
                      and num_choices > 1)

        if searchable:
            action, sample, sb_boundary = _search_and_sample(
                env, evaluator, rng, knobs, num_choices=num_choices,
                priority_is_a=priority_is_a, game_idx=game_idx,
                game_move=game_move, sb_boundary=sb_boundary,
                sb_stats=sb_stats)
            samples.append(sample)
            searched += 1
        elif net_to_move:
            priors, _ = evaluator.evaluate(env._obs, num_choices)
            action = int(np.argmax(priors))
            fallback += 1
        else:
            # Scripted seat: no search, no sample, no counters. The hard-tier
            # agent answers every prompt kind, including the bo3 sideboard
            # prompts (auto_sideboard=False), as in generate_expert.
            action = int(agent.act(env._obs, num_choices)) if num_choices > 1 else 0

        _sb_latch_played(sb_boundary, env._obs, action)

        obs, reward, terminated, truncated, info = env.step(action)
        move += 1
        game_move += 1
        # A GAME_RESULT landed on this step -> the game the just-stepped action
        # belonged to has finished. Record its winner and advance to the next game.
        boundary = bool(info.get("game_result"))
        if boundary:
            game_winners.append(winner_from_reward(reward))
            game_idx += 1
            game_move = 0
            sb_boundary = None
            if agent is not None:
                agent.new_game()
        if on_progress is not None and move % HEARTBEAT_MOVES == 0:
            on_progress(move, searched, fallback)
        if terminated or truncated:
            done = True
            if terminated and not boundary:
                # bo1 mode emits no GAME_RESULT line — the single game ends with a
                # plain "Player X wins" + terminated. Price the in-progress game
                # from the terminal reward sign (bo3's final game already recorded
                # via the boundary branch above, so guard on `not boundary`).
                game_winners.append(winner_from_reward(reward))
            if truncated:
                # The match hit MAX_STEPS_BO3 mid-game: keep samples from the
                # games that finished (they have a z), drop the in-progress game's
                # samples (no result yet -> no valid target).
                samples, dropped = _drop_unfinished(samples, game_winners)
    return samples, game_winners, searched, fallback, dropped, sb_stats


def compute_td_targets(z, q, explored, game_idx, is_sideboard, mover_is_a, td_n):
    """The n-step TD value target ``td_q``, one float per sample.

    Every argument but ``td_n`` is a parallel per-sample sequence in PLAY ORDER.
    A bootstrap window may only run inside one CHAIN — a contiguous run sharing
    ``(game_idx, is_sideboard)`` — so it never crosses a game boundary (the next
    game's values are unrelated) nor the bo3 SIDEBOARD boundary. Sideboard-root
    samples carry the UPCOMING game's ``game_idx`` (they gate it) and sit as a
    contiguous run in FRONT of that game's in-game samples, so they form their own
    chain: they never appear inside an in-game sample's window and their own
    targets are always ``z`` (a sideboard pick's value is only about the game it
    is choosing, and the picks within a boundary are one seat's bookkeeping, not a
    line of play).

    For sample ``t`` in a chain ending at ``L`` (the chain's last sample), with
    ``E`` the first sample after ``t`` in the same chain whose action was
    EXPLORATORY (``explored``; infinity when there is none) and
    ``j_max = min(E - 1, L)``:

      * ``t + td_n <= j_max``  -> bootstrap: ``td_q[t] = s * q[t + td_n]``
      * ``j_max == L``         -> the window runs clean to the end of the game, so
                                  the true outcome is in reach: ``td_q[t] = z[t]``
      * otherwise              -> SHORTENED horizon: bootstrap off the last sample
                                  before the exploratory move,
                                  ``td_q[t] = s * q[j_max]`` (``z[t]`` when the
                                  block is immediate, i.e. ``j_max == t``)

    ``s`` is the mandatory PERSPECTIVE FLIP: ``q`` and ``z`` are both stored from
    their own sample's MOVER's point of view and the seat alternates freely
    between samples, so ``s = +1`` when the bootstrap sample's mover is the same
    seat as ``t``'s and ``-1`` otherwise. A non-finite ``q`` at the bootstrap
    sample (expert shards record ``q = NaN``) falls back to ``z[t]``, so the
    returned array is always finite."""
    m = len(z)
    z = np.asarray(z, dtype=np.float32)
    q = np.asarray(q, dtype=np.float32)
    explored = np.asarray(explored)
    mover_is_a = np.asarray(mover_is_a, dtype=bool)
    td = z.astype(np.float32, copy=True)
    n = max(1, int(td_n))
    key = list(zip(game_idx, is_sideboard))
    lo = 0
    while lo < m:
        hi = lo
        while hi + 1 < m and key[hi + 1] == key[lo]:
            hi += 1
        if not key[lo][1]:
            _fill_chain_td(td, z, q, explored, mover_is_a, lo, hi, n)
        lo = hi + 1
    return td


def _fill_chain_td(td, z, q, explored, mover_is_a, lo, hi, n):
    """Apply the n-step rule inside one chain ``[lo, hi]`` (see
    :func:`compute_td_targets`). ``td`` is pre-seeded with ``z``, so every branch
    that falls back to the outcome simply leaves its entry alone."""
    # next_explored[i] = smallest j > i in [lo, hi] with explored[j], else hi + 1
    # (an out-of-chain sentinel, which makes j_max == hi == L on its own).
    nxt = hi + 1
    next_explored = [hi + 1] * (hi - lo + 1)
    for i in range(hi, lo - 1, -1):
        next_explored[i - lo] = nxt
        if explored[i]:
            nxt = i
    for t in range(lo, hi + 1):
        j_max = min(next_explored[t - lo] - 1, hi)
        if t + n <= j_max:
            j = t + n
        elif j_max >= hi:
            continue            # terminal inside the horizon -> keep z[t]
        elif j_max > t:
            j = j_max           # shortened horizon at the exploratory move
        else:
            continue            # immediate block -> keep z[t]
        qj = q[j]
        if not np.isfinite(qj):
            continue            # no root value recorded (expert shard) -> z[t]
        td[t] = qj if mover_is_a[j] == mover_is_a[t] else -qj


def _backfill_and_pack(samples, game_winners, td_n: int = DEFAULT_TD_N):
    """Fill z per sample from its mover's perspective vs the winner of the GAME the
    sample belongs to (``game_winners[game_idx]``), derive the n-step TD target
    ``td_q`` from the whole match's samples (:func:`compute_td_targets`), then pack
    to the shard arrays. A drawn game (winner None) -> z=0.

    Returns a dict of the :data:`SHARD_KEYS` arrays. Samples recorded WITHOUT a
    search (``generate_expert``'s behavior-cloning rows) carry no root value: they
    get ``q = NaN``, ``explored = 0`` and therefore ``td_q = z``."""
    n = len(samples)
    obs = np.zeros((n, OBS_SIZE), dtype=np.float32)
    pi = np.zeros((n, MAX_ACTIONS), dtype=np.float32)
    z = np.zeros((n,), dtype=np.float32)
    mask = np.zeros((n, MAX_ACTIONS), dtype=bool)
    q = np.zeros((n,), dtype=np.float32)
    explored = np.zeros((n,), dtype=np.uint8)
    mover_is_a = np.zeros((n,), dtype=bool)
    for i, s in enumerate(samples):
        obs[i] = s["obs"]
        pi[i] = s["pi"]
        mask[i] = s["mask"]
        q[i] = s.get("q", np.nan)
        explored[i] = np.uint8(s.get("explored", 0))
        mover_is_a[i] = s["mover_is_a"]
        winner = game_winners[s["game_idx"]]
        if winner is None:
            z[i] = 0.0
        else:
            mover_won = (winner == "A") == s["mover_is_a"]
            z[i] = 1.0 if mover_won else -1.0
    td_q = compute_td_targets(
        z, q, explored, [s["game_idx"] for s in samples],
        [bool(s["obs"][_IS_SIDEBOARD_IDX] > 0.5) for s in samples],
        mover_is_a, td_n)
    return {"obs": obs, "pi": pi, "z": z, "mask": mask, "q": q,
            "explored": explored, "td_q": td_q}


def _write_shard(out_dir, arrays, n_idx):
    ts = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    path = os.path.join(out_dir, f"shard_{ts}_{pid}_{n_idx}.npz")
    np.savez_compressed(path, **{k: arrays[k] for k in SHARD_KEYS})
    return path


# ----------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------

def _match_winner(game_winners) -> str:
    """The match result ('A'/'B'/'DRAW') from a game-winner list — whoever won
    more games (bo1: the single game; bo3: first to two)."""
    a = game_winners.count("A")
    b = game_winners.count("B")
    return "A" if a > b else ("B" if b > a else "DRAW")


def _worker(matchups, source, sims, worlds, temp_moves, root_noise_eps,
            root_noise_alpha, out_dir, base_seed, worker_idx, result_q, bo3,
            sb_sims, sb_worlds, sb_max_depth, sb_rollout_turns, sb_persist,
            scripted_seats=None, td_n=DEFAULT_TD_N, merge_dupes=True):
    """Play this worker's slice of the matchup schedule. ``matchups`` is a list of
    per-MATCH (deck_a, deck_b) pairs (mirror or cross-deck); the env's decks are
    swapped per match before its reset respawns the engine. With ``bo3`` each
    matchup is a best-of-three yielding up to three games of samples (decks stay
    fixed across the games of one match).

    ``scripted_seats`` (when given) is a parallel list of per-match
    Optional[bool]: ``None`` = pure self-play (the net pilots both seats), else
    the value is ``net_is_a`` for a match whose OTHER seat is piloted by
    scripted:hard (:func:`_play_match`'s ``agent``/``net_is_a`` arguments)."""
    import torch
    torch.set_num_threads(1)   # avoid oversubscription across worker processes
    from search_env import SearchRoboMageEnv
    from az_net import AZEvaluator

    net = _build_net(source)
    evaluator = AZEvaluator(net)
    rng = np.random.default_rng(base_seed + 100003 * (worker_idx + 1))
    agent = None
    if scripted_seats is not None and any(s is not None for s in scripted_seats):
        from scripted_agent import make_agent
        agent = make_agent("scripted:hard")

    n_matches = len(matchups)
    da0, db0 = matchups[0] if matchups else (None, None)
    env = SearchRoboMageEnv(deck_a=da0, deck_b=db0, bo3=bo3, auto_sideboard=False,
                           binary_path=INTERACTIVE_BINARY)
    total_samples = 0
    shards = []
    buf = []
    shard_n = 0
    stats = {"searched": 0, "fallback": 0, "wins_a": 0, "wins_b": 0, "draws": 0,
             "games": 0, "dropped": 0, "sb_reused_visits": 0, "sb_memo_hits": 0,
             "scripted_matches": 0, "net_wins_vs_scripted": 0}
    try:
        for m in range(n_matches):
            seed = base_seed + worker_idx * 100000 + m
            # Swap decks for this match; reset() (inside _play_match) respawns the
            # engine reading the current _deck_a/_deck_b.
            env._deck_a, env._deck_b = matchups[m]
            net_is_a = None if scripted_seats is None else scripted_seats[m]

            def beat(move, searched_ct, fallback_ct, _m=m):
                result_q.put({"kind": "beat", "worker": worker_idx,
                              "match": _m + 1, "n_matches": n_matches, "move": move,
                              "searched": searched_ct, "fallback": fallback_ct})

            t0 = time.time()
            if net_is_a is not None:
                agent.set_deck_names(*matchups[m])
            samples, game_winners, searched, fallback, dropped, sb_st = _play_match(
                env, evaluator, rng, sims=sims, worlds=worlds,
                temp_moves=temp_moves, root_noise_eps=root_noise_eps,
                root_noise_alpha=root_noise_alpha, seed=seed,
                on_progress=beat, sb_sims=sb_sims, sb_worlds=sb_worlds,
                sb_max_depth=sb_max_depth, sb_rollout_turns=sb_rollout_turns,
                sb_persist=sb_persist, merge_dupes=merge_dupes,
                agent=(None if net_is_a is None else agent),
                net_is_a=(None if net_is_a is None else bool(net_is_a)))
            buf.append(_backfill_and_pack(samples, game_winners, td_n=td_n))
            total_samples += len(samples)
            stats["searched"] += searched
            stats["fallback"] += fallback
            stats["dropped"] += dropped
            stats["games"] += len(game_winners)
            stats["sb_reused_visits"] += sb_st["sb_reused_visits"]
            stats["sb_memo_hits"] += sb_st["sb_memo_hits"]
            mwinner = _match_winner(game_winners)
            stats["wins_a"] += int(mwinner == "A")
            stats["wins_b"] += int(mwinner == "B")
            stats["draws"] += int(mwinner == "DRAW")
            if net_is_a is not None:
                stats["scripted_matches"] += 1
                stats["net_wins_vs_scripted"] += int(
                    mwinner == ("A" if net_is_a else "B"))
            result_q.put({"kind": "match", "worker": worker_idx, "match": m + 1,
                          "n_matches": n_matches, "winner": mwinner,
                          "game_score": "-".join(str(game_winners.count(x))
                                                  for x in ("A", "B")),
                          "games": len(game_winners), "samples": len(samples),
                          "dropped": dropped, "searched": searched,
                          "fallback": fallback, "secs": time.time() - t0,
                          "scripted": (None if net_is_a is None
                                       else ("B" if net_is_a else "A"))})
            if _buffered(buf) >= FLUSH_SAMPLES:
                shards.append(_write_shard(out_dir, _concat(buf), shard_n))
                result_q.put({"kind": "shard", "worker": worker_idx,
                              "path": shards[-1]})
                shard_n += 1
                buf = []
        if buf:
            shards.append(_write_shard(out_dir, _concat(buf), shard_n))
            result_q.put({"kind": "shard", "worker": worker_idx,
                          "path": shards[-1]})
    finally:
        env.close()
    result_q.put({"kind": "done", "worker": worker_idx, "samples": total_samples,
                  "shards": shards, "stats": stats})


def _concat(buf):
    """Concatenate a list of per-match :func:`_backfill_and_pack` dicts."""
    return {k: np.concatenate([b[k] for b in buf], axis=0) for k in SHARD_KEYS}


def _buffered(buf) -> int:
    """Samples currently held in a shard buffer (list of packed dicts)."""
    return sum(len(b["z"]) for b in buf)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def _discard_pre_bo3_shards(out_dir: str) -> None:
    """One-time cleanup on the FIRST bo3 self-play run into ``out_dir``: delete the
    legacy bo1 ``shard_*.npz`` (bo1 and bo3 shards share a schema and would
    otherwise be mixed by the trainer's recency window). Guarded by a sentinel file
    so subsequent bo3 runs KEEP their accumulated bo3 shards. Never touches the PPO
    ``checkpoints/gen__*.zip`` — only this pooled az_data dir."""
    import glob
    sentinel = os.path.join(out_dir, ".bo3_migrated")
    if os.path.exists(sentinel):
        return
    stale = glob.glob(os.path.join(out_dir, "shard_*.npz"))
    for p in stale:
        try:
            os.remove(p)
        except OSError as exc:
            print(f"[az-selfplay] WARNING: could not remove stale shard {p}: {exc}")
    if stale:
        print(f"[az-selfplay] discarded {len(stale)} pre-bo3 (bo1) shard(s) from "
              f"{out_dir} before the first bo3 run")
    with open(sentinel, "w") as fh:
        fh.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


def generate(deck: str, *, games: int = DEFAULT_AZ_GAMES,
             sims: int = DEFAULT_AZ_SIMS, worlds: int = DEFAULT_AZ_WORLDS,
             workers: Optional[int] = None, checkpoint: Optional[str] = None,
             temp_moves: int = DEFAULT_TEMP_MOVES,
             root_noise_eps: float = DEFAULT_ROOT_NOISE_EPS,
             root_noise_alpha: float = DEFAULT_ROOT_NOISE_ALPHA,
             out_dir: Optional[str] = None, seed: int = 1,
             use_actor: Optional[bool] = None,
             roster: Optional[list] = None,
             focus_decks: Optional[list] = None,
             mirror_frac: float = DEFAULT_MIRROR_FRAC,
             sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
             sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
             sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
             sb_persist: bool = bool(DEFAULT_SB_PERSIST),
             merge_dupes: bool = True,
             scripted_opponent_frac: float = 0.0,
             bo3: bool = False, exhaustive: bool = False,
             exhaustive_selfplay: bool = False,
             exhaustive_repeats: int = DEFAULT_AZ_EXHAUSTIVE_REPEATS,
             scripted_cells: int = DEFAULT_AZ_SCRIPTED_CELLS,
             slot: int = 0,
             td_n: int = DEFAULT_TD_N) -> dict:
    """Generate ``games`` self-play MATCHES over a FOCUS pool and write shards.

    ``games`` is a count of MATCHES (bo1: one game each; bo3: up to three games
    each). ``focus_decks`` is the pool of decks the generalist pilots (default
    ``[deck]``, i.e. single-focus); each match one is drawn uniformly. Its opponent
    is that same deck (mirror) with probability ``mirror_frac``, else a uniform draw
    from ``roster`` (default: every ``decks/league/*.dk``); the seeded schedule
    alternates the focus deck's seat. Passing a multi-deck ``focus_decks`` (with a
    multi-deck ``roster``) makes one run span a full deck×opponent matrix.
    Shards pool into ``out_dir`` (default ``az_data/gen/`` — filenames are globally
    unique, so cross-deck runs share one pool feeding the single generalist net).

    ``bo3`` runs best-of-three matches with a PER-GAME value target (each sample's
    z is the result of the game it belonged to). BOTH backends support bo3 (the
    C++ actor mirrors the Python match loop, including the searched sideboard
    roots), so ``bo3`` no longer constrains the backend choice. In bo3 the
    between-game sideboard prompts are searched MCTS roots with their own (heavier,
    deeper) budget ``sb_sims``/``sb_worlds``/``sb_max_depth``/``sb_rollout_turns``
    — see :func:`_play_match` for the Python path and the actor's ``--sb-sims``/
    ``--sb-worlds``/``--sb-max-depth``/``--sb-rollout-turns`` flags for the C++
    path. These knobs are consumed only in bo3 (they are inert for bo1 on either
    backend).

    ``scripted_opponent_frac`` (0..1, default 0 = pure self-play) is the fraction
    of matches whose OPPONENT seat is piloted by the rule-based scripted:hard
    agent while the net+MCTS pilots the FOCUS seat; only the net seat's decisions
    become samples (see :func:`_play_match`'s ``agent``/``net_is_a``). Which
    matches get a scripted opponent is drawn from a dedicated seeded RNG stream
    indexed per match, so it is reproducible and independent of the worker
    count. The C++ actor has no scripted-opponent support, so a nonzero
    fraction FORCES the
    Python backend (and is a loud error alongside an explicit ``use_actor=True``).

    ``exhaustive`` replaces the random schedule with the exact matchup MATRIX of
    :func:`build_exhaustive_schedule_ex`: one vs-scripted match per ORDERED
    (focus x roster) pair plus one pure self-play match per UNORDERED pair —
    155 matches on the 10-deck league roster, each cell exactly once. ``games``,
    ``mirror_frac`` and ``scripted_opponent_frac`` are ignored (loudly). The
    backend goes HYBRID (:func:`_generate_hybrid`): the C++ actor, when built,
    plays the pure self-play cells and the Python backend the vs-scripted cells
    (the actor has no scripted seat); ``use_actor=False`` keeps everything on
    Python, ``use_actor=True`` demands the actor binary but remains hybrid.

    ``exhaustive_selfplay`` (implies ``exhaustive``) narrows that matrix to the
    pure SELF-PLAY family only — one match per UNORDERED deck pair, mirrors
    included (55 on the 10-deck roster), with NO vs-scripted cells. Since the
    scripted cells are what forced the Python backend, this schedule runs
    entirely on the C++ actor when it is built. ``exhaustive_repeats=n`` plays
    every cell of whichever matrix is selected ``n`` times.

    ``scripted_cells=k`` adds a ROTATING slice of the vs-scripted family back
    onto a ``exhaustive_selfplay`` schedule: ``k`` matches per call, taken from
    the full ordered (focus, opponent) pair list starting at offset
    ``(slot * k) % n_ordered`` and wrapping, so an az-league run's successive
    ``slot``s tile every ordered pair over time (a standalone cycle is slot 0).
    They are marked exactly like the full matrix's scripted cells, so the backend
    goes HYBRID for them — actor self-play cells, Python scripted cells — with no
    special-casing. ``k=0`` disables; it is IGNORED (with a printed note) under
    full ``exhaustive``, which already plays every ordered pair.

    ``td_n`` is the n-step TD horizon baked into every written sample's ``td_q``
    column (see :func:`compute_td_targets`) — a GENERATION-side knob, honoured
    identically by both backends (the actor takes it as ``--td-n``).

    ``use_actor`` picks the generation backend:
      * ``None`` (AUTO, default) — use the C++ ``bin/az_actor`` iff it is built,
        else the pure-Python multiprocess path;
      * ``True`` — force the actor (loud error if ``bin/az_actor`` is missing);
      * ``False`` — force the Python path.
    Both backends write the SAME trainer-compatible ``shard_*.npz`` files into
    ``out_dir`` and return the same summary dict shape."""
    scripted_opponent_frac = float(scripted_opponent_frac or 0.0)
    if not 0.0 <= scripted_opponent_frac <= 1.0:
        raise ValueError(
            f"scripted_opponent_frac must be in [0, 1] "
            f"(got {scripted_opponent_frac})")
    exhaustive = bool(exhaustive) or bool(exhaustive_selfplay)
    exhaustive_repeats = max(1, int(exhaustive_repeats))
    scripted_cells = max(0, int(scripted_cells))
    if exhaustive:
        # The matrix defines the match count; the random-draw knobs are inert.
        ignored = [name for name, on in
                   (("games", True), ("mirror_frac", mirror_frac != DEFAULT_MIRROR_FRAC),
                    ("scripted_opponent_frac", scripted_opponent_frac > 0.0)) if on]
        times = ("exactly once" if exhaustive_repeats == 1
                 else f"exactly {exhaustive_repeats} times")
        print(f"[az-selfplay] exhaustive matrix"
              f"{' (self-play cells only)' if exhaustive_selfplay else ''}: "
              f"ignoring {', '.join(ignored)} (every cell is played {times})")
        scripted_opponent_frac = 0.0
    if exhaustive and not exhaustive_selfplay and scripted_cells:
        print(f"[az-selfplay] ignoring scripted_cells={scripted_cells}: the full "
              f"exhaustive matrix already plays every ordered vs-scripted cell")
        scripted_cells = 0
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    out_dir = out_dir or os.path.join(_AZ_DATA_DIR, GEN_STEM)
    os.makedirs(out_dir, exist_ok=True)
    if roster is None:
        roster = league_roster()
    focus = list(focus_decks) if focus_decks else [deck]

    # Build the schedule BEFORE choosing a backend: whether this run needs the
    # hybrid split is a property of the schedule (does it contain vs-scripted
    # cells?), not of the exhaustive flags — a self-play matrix carrying a
    # rotating scripted_cells slice needs the split just as the full matrix does.
    if exhaustive:
        sched_ex, scripted_seats = build_exhaustive_schedule_ex(
            focus, roster, seed, include_scripted=not exhaustive_selfplay,
            repeats=exhaustive_repeats, scripted_cells=scripted_cells, slot=slot)
        n_scr = sum(1 for s in scripted_seats if s is not None)
        per = ("one" if exhaustive_repeats == 1 else f"{exhaustive_repeats}")
        scr_txt = (
            f"{n_scr} vs scripted:hard, rotating slice of the ordered pair list "
            f"at slot {slot}; " if exhaustive_selfplay and n_scr else
            "" if exhaustive_selfplay else
            f"{n_scr} vs scripted:hard, {per} per ordered focus x opponent pair; ")
        print(f"[az-selfplay] exhaustive matrix"
              f"{' (SELF-PLAY ONLY)' if exhaustive_selfplay else ''}: "
              f"{len(sched_ex)} matches ({scr_txt}"
              f"{len(sched_ex) - n_scr} pure self-play, {per} per unordered pair)")
        games = len(sched_ex)
    else:
        sched_ex = build_matchup_schedule_ex(focus, roster, games, mirror_frac,
                                             seed)
        scripted_seats = None
        n_scr = 0
        if scripted_opponent_frac > 0.0:
            # Dedicated RNG stream (never touches the schedule's draws) indexed
            # per MATCH, so the scripted assignment is reproducible and
            # independent of how the schedule is sliced across workers. Entry =
            # net_is_a; None = pure self-play for that match.
            srng = np.random.default_rng([seed, 0x5C819])
            scripted_seats = [(focus_is_a
                               if srng.random() < scripted_opponent_frac
                               else None)
                              for (_a, _b, focus_is_a) in sched_ex]
            n_scr = sum(1 for s in scripted_seats if s is not None)
    schedule = [(a, b) for a, b, _ in sched_ex]

    have_actor = os.path.exists(_ACTOR_BIN)
    if bo3:
        # One-time discard of legacy bo1 shards before the first bo3 run. This is
        # backend-agnostic (the .bo3_migrated sentinel gates it either way) and
        # runs regardless of which backend generates the bo3 shards below.
        _discard_pre_bo3_shards(out_dir)
    requested = use_actor        # None = AUTO, True = --actor, False = --no-actor
    if use_actor is None:
        use_actor = have_actor
        chosen = "AUTO"
    else:
        chosen = "forced"
    hybrid = False
    if exhaustive and n_scr and n_scr < len(schedule):
        # HYBRID backend: the C++ actor (when built) plays the pure self-play
        # cells while the Python backend plays the vs-scripted cells (the actor
        # has no scripted-opponent seat). --no-actor keeps everything on the
        # Python path; --actor demands the binary but is still hybrid — the
        # scripted cells can never run on the actor.
        if requested is False:
            chosen = "forced PYTHON"
        elif not have_actor:
            if requested is True:
                raise FileNotFoundError(
                    f"--actor requested but the actor binary is not built at "
                    f"{_ACTOR_BIN} (build it with `make actor`, or pass "
                    f"--no-actor)")
            chosen = "AUTO->PYTHON (actor absent)"
        else:
            hybrid = True
            chosen = "HYBRID (forced)" if requested is True else "AUTO->HYBRID"
        use_actor = False
    elif exhaustive and n_scr:
        # Degenerate matrix: every cell is a vs-scripted cell (e.g. a one-deck
        # roster whose only self-play cell was dropped) — pure Python.
        use_actor = False
        chosen += "->PYTHON (all cells vs-scripted)"
    elif use_actor and scripted_opponent_frac > 0.0:
        if chosen == "forced":
            raise ValueError(
                "--actor is incompatible with --scripted-opponent-frac > 0: the "
                "C++ az_actor plays pure self-play (no scripted-opponent seat). "
                "Drop --actor (or pass --no-actor) to use the Python backend.")
        use_actor = False
        chosen = "AUTO->PYTHON (scripted opponent)"
    if use_actor and not have_actor:
        raise FileNotFoundError(
            f"--actor requested but the actor binary is not built at {_ACTOR_BIN} "
            f"(build it with `make actor`, or pass --no-actor)")

    workers = max(1, min(workers, games))
    source = resolve_source(deck, checkpoint)
    focus_lbl = focus[0] if len(focus) == 1 else f"{len(focus)} decks [{','.join(focus)}]"
    unit = "matches" if bo3 else "games"
    print(f"[az-selfplay] focus={focus_lbl} {unit}={games} bo3={bo3} sims={sims} "
          f"worlds={worlds} workers={workers} mirror_frac={mirror_frac}")
    if bo3:
        print(f"[az-selfplay] sideboard-root budget: sb_sims={sb_sims} "
              f"sb_worlds={sb_worlds} sb_max_depth={sb_max_depth} "
              f"sb_rollout_turns={sb_rollout_turns} sb_persist={int(sb_persist)}")
    print(f"[az-selfplay] net source: mode={source['mode']} path={source['path']}")
    print(f"[az-selfplay] out_dir={out_dir}")
    print(f"[az-selfplay] matchups: {_schedule_summary(schedule)}")
    if scripted_seats is not None and any(s is not None for s in scripted_seats):
        n_scripted = sum(1 for s in scripted_seats if s is not None)
        how = ("exhaustive matrix"
               if exhaustive else f"frac={scripted_opponent_frac}")
        print(f"[az-selfplay] scripted-opponent games: {n_scripted}/{len(schedule)} "
              f"({how}; scripted:hard on the opponent seat, "
              f"net samples only)")
    backend_lbl = ("HYBRID" if hybrid
                   else "ACTOR" if use_actor else "PYTHON")
    print(f"[az-selfplay] backend={backend_lbl} ({chosen}); "
          f"az_actor {'present' if have_actor else 'absent'}")

    common = dict(source=source, schedule=schedule, sims=sims, worlds=worlds,
                  workers=workers, temp_moves=temp_moves,
                  root_noise_eps=root_noise_eps, root_noise_alpha=root_noise_alpha,
                  out_dir=out_dir, seed=seed, td_n=td_n,
                  merge_dupes=merge_dupes)
    if hybrid:
        return _generate_hybrid(deck, scripted_seats=scripted_seats,
                                actor_bin=_ACTOR_BIN, bo3=bo3, sb_sims=sb_sims,
                                sb_worlds=sb_worlds, sb_max_depth=sb_max_depth,
                                sb_rollout_turns=sb_rollout_turns,
                                sb_persist=sb_persist, **common)
    if use_actor:
        # The C++ actor mirrors the Python match loop for bo3 (Stage 6), including
        # the searched sideboard roots — pass bo3 + the sb budget through so the
        # actor argv carries --bo3 and --sb-sims/--sb-worlds/--sb-max-depth. The
        # sb_* knobs are inert for bo1.
        return _generate_actor(deck, actor_bin=_ACTOR_BIN, bo3=bo3, sb_sims=sb_sims,
                               sb_worlds=sb_worlds, sb_max_depth=sb_max_depth,
                               sb_rollout_turns=sb_rollout_turns,
                               sb_persist=sb_persist,
                               **common)
    return _generate_python(deck, bo3=bo3, sb_sims=sb_sims, sb_worlds=sb_worlds,
                            sb_max_depth=sb_max_depth,
                            sb_rollout_turns=sb_rollout_turns,
                            sb_persist=sb_persist, scripted_seats=scripted_seats,
                            **common)


# ----------------------------------------------------------------------
# Hybrid backend (exhaustive matrix: actor self-play + Python vs-scripted)
# ----------------------------------------------------------------------

# Seed offset for the hybrid's actor pass, so its per-group seed ranges never
# collide with the Python pass's game seeds run in the same invocation.
_HYBRID_ACTOR_SEED_OFFSET = 10_000_019


def _merge_summaries(a: dict, b: dict) -> dict:
    """Fold two backend summary dicts (same shape both backends return) into
    one: samples/shards concatenate, stats sum key-wise (a key missing from one
    side counts 0)."""
    stats = dict(a["stats"])
    for k, v in b["stats"].items():
        stats[k] = stats.get(k, 0) + v
    return {"samples": a["samples"] + b["samples"],
            "shards": list(a["shards"]) + list(b["shards"]),
            "stats": stats, "out_dir": a["out_dir"], "source": a["source"]}


def _generate_hybrid(deck, *, source, schedule, scripted_seats, sims, worlds,
                     workers, temp_moves, root_noise_eps, root_noise_alpha,
                     out_dir, seed, actor_bin, bo3=False,
                     sb_sims=DEFAULT_SB_SIMS, sb_worlds=DEFAULT_SB_WORLDS,
                     sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                     sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                     sb_persist=bool(DEFAULT_SB_PERSIST),
                     merge_dupes=True,
                     td_n=DEFAULT_TD_N) -> dict:
    """Split an exhaustive-matrix schedule across BOTH backends: the C++ actor
    plays the pure self-play cells (it is much faster per match), the Python
    multiprocess backend the vs-scripted cells (the actor has no scripted seat).
    The passes run sequentially, each with the full ``workers`` budget, and both
    write trainer-compatible ``shard_*.npz`` files into the same ``out_dir``
    (filenames are globally unique), so the merged summary's shard list is
    exactly what an auto (``window=0``) training window should count.

    The actor pass runs on a derived seed stream
    (``seed + _HYBRID_ACTOR_SEED_OFFSET``) so its game seeds never collide with
    the Python pass's; per-pass reproducibility is unchanged."""
    self_sched = [m for m, s in zip(schedule, scripted_seats) if s is None]
    scr_sched = [m for m, s in zip(schedule, scripted_seats) if s is not None]
    scr_seats = [s for s in scripted_seats if s is not None]
    print(f"[az-selfplay] hybrid split: {len(self_sched)} pure self-play "
          f"matches -> ACTOR, {len(scr_sched)} vs-scripted matches -> PYTHON "
          f"(sequential, {workers} workers each)")
    kw = dict(source=source, sims=sims, worlds=worlds, workers=workers,
              temp_moves=temp_moves, root_noise_eps=root_noise_eps,
              root_noise_alpha=root_noise_alpha, out_dir=out_dir, bo3=bo3,
              sb_sims=sb_sims, sb_worlds=sb_worlds, sb_max_depth=sb_max_depth,
              sb_rollout_turns=sb_rollout_turns, sb_persist=sb_persist,
              merge_dupes=merge_dupes, td_n=td_n)
    summaries = []
    if self_sched:
        print(f"[az-selfplay] hybrid pass 1/2: ACTOR ({len(self_sched)} matches)")
        summaries.append(_generate_actor(
            deck, schedule=self_sched, seed=seed + _HYBRID_ACTOR_SEED_OFFSET,
            actor_bin=actor_bin, **kw))
    if scr_sched:
        print(f"[az-selfplay] hybrid pass 2/2: PYTHON ({len(scr_sched)} "
              f"vs-scripted matches)")
        summaries.append(_generate_python(
            deck, schedule=scr_sched, seed=seed, scripted_seats=scr_seats,
            **kw))
    merged = summaries[0]
    for s in summaries[1:]:
        merged = _merge_summaries(merged, s)
    print(f"[az-selfplay] hybrid done: {merged['samples']} samples, "
          f"{len(merged['shards'])} shards from {len(schedule)} matches "
          f"(ACTOR {len(self_sched)} + PYTHON {len(scr_sched)})")
    return merged


# ----------------------------------------------------------------------
# Python multiprocess backend
# ----------------------------------------------------------------------

def _generate_python(deck, *, source, schedule, sims, worlds, workers, temp_moves,
                     root_noise_eps, root_noise_alpha, out_dir, seed,
                     bo3=False, sb_sims=DEFAULT_SB_SIMS, sb_worlds=DEFAULT_SB_WORLDS,
                     sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                     sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                     sb_persist=bool(DEFAULT_SB_PERSIST),
                     merge_dupes=True,
                     scripted_seats=None, td_n=DEFAULT_TD_N) -> dict:
    import multiprocessing as mp

    matches = len(schedule)
    # Split the matchup schedule across workers (contiguous slices).
    per = [matches // workers] * workers
    for i in range(matches % workers):
        per[i] += 1
    slices = []
    seat_slices = []
    off = 0
    for n in per:
        slices.append(schedule[off:off + n])
        seat_slices.append(None if scripted_seats is None
                           else scripted_seats[off:off + n])
        off += n

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for wi in range(workers):
        if per[wi] == 0:
            continue
        p = ctx.Process(target=_worker,
                        args=(slices[wi], source, sims, worlds, temp_moves,
                              root_noise_eps, root_noise_alpha, out_dir, seed,
                              wi, result_q, bo3, sb_sims, sb_worlds, sb_max_depth,
                              sb_rollout_turns, sb_persist, seat_slices[wi],
                              td_n, merge_dupes))
        p.start()
        procs.append(p)

    # Live progress: workers stream beat/match/shard events onto the queue and
    # finish with a 'done' record each. Consume until every worker reported.
    import queue as _queue
    t_start = time.time()
    results = []
    matches_done = 0
    games_done = 0
    samples_so_far = 0
    dropped_so_far = 0
    tally = {"A": 0, "B": 0, "DRAW": 0}
    while len(results) < len(procs):
        try:
            msg = result_q.get(timeout=60)
        except _queue.Empty:
            dead = [p.exitcode for p in procs if not p.is_alive()]
            if len(dead) > len(results):
                raise RuntimeError(
                    f"az-selfplay: {len(dead) - len(results)} worker(s) died "
                    f"without reporting (exit codes {dead}) — see stderr above")
            continue
        kind = msg.get("kind")
        if kind == "beat":
            print(f"[az-selfplay] w{msg['worker']} m{msg['match']}/{msg['n_matches']}: "
                  f"move {msg['move']}, searched {msg['searched']}, "
                  f"fallback {msg['fallback']}", flush=True)
        elif kind == "match":
            matches_done += 1
            games_done += msg["games"]
            samples_so_far += msg["samples"]
            dropped_so_far += msg.get("dropped", 0)
            tally[msg["winner"]] += 1
            elapsed = time.time() - t_start
            eta = elapsed / matches_done * (matches - matches_done)
            drop_note = (f" dropped={msg['dropped']}" if msg.get("dropped") else "")
            scr_note = (f" [scripted:hard={msg['scripted']}]"
                        if msg.get("scripted") else "")
            print(f"[az-selfplay] w{msg['worker']} m{msg['match']}/{msg['n_matches']}: "
                  f"match={msg['winner']}{scr_note} (games A-B {msg['game_score']}) "
                  f"samples={msg['samples']}{drop_note} "
                  f"searched={msg['searched']} fallback={msg['fallback']} "
                  f"in {_fmt_secs(msg['secs'])} | total {matches_done}/{matches} "
                  f"matches ({games_done} games), {samples_so_far} samples, "
                  f"match A {tally['A']} B {tally['B']} D {tally['DRAW']}, "
                  f"elapsed {_fmt_secs(elapsed)}, eta {_fmt_secs(eta)}", flush=True)
        elif kind == "shard":
            print(f"[az-selfplay] w{msg['worker']} wrote {msg['path']}", flush=True)
        else:  # 'done' (also tolerates legacy kind-less records)
            results.append(msg)
    for p in procs:
        p.join()

    total_samples = sum(r["samples"] for r in results)
    all_shards = [s for r in results for s in r["shards"]]
    agg = {"searched": 0, "fallback": 0, "wins_a": 0, "wins_b": 0, "draws": 0,
           "games": 0, "dropped": 0, "scripted_matches": 0,
           "net_wins_vs_scripted": 0}
    for r in results:
        for k in agg:
            agg[k] += r["stats"].get(k, 0)
    print(f"[az-selfplay] done: {total_samples} samples, {len(all_shards)} shards "
          f"from {matches} matches ({agg['games']} games) (PYTHON)")
    if agg["dropped"]:
        print(f"[az-selfplay] dropped {agg['dropped']} sample(s) from "
              f"truncated in-progress games (no game result)")
    print(f"[az-selfplay] decisions searched={agg['searched']} "
          f"fallback={agg['fallback']}; match results A={agg['wins_a']} "
          f"B={agg['wins_b']} draws={agg['draws']}")
    if agg["scripted_matches"]:
        w = agg["net_wins_vs_scripted"]
        n = agg["scripted_matches"]
        print(f"[az-selfplay] net vs scripted:hard: {w}W/{n - w}L over {n} "
              f"matches ({w / n:.3f})")
    return {"samples": total_samples, "shards": all_shards, "stats": agg,
            "out_dir": out_dir, "source": source}


# ----------------------------------------------------------------------
# C++ actor backend (bin/az_actor --selfplay)
# ----------------------------------------------------------------------

def _ensure_actor_torchscript(source: dict):
    """Return (ts_path, tmpdir) — a TorchScript ``.ts.pt`` the actor can load.

    * mode 'az'  : export/refresh the ``.ts.pt`` sibling of the state_dict ``.pt``
      (re-export when missing or older than the ``.pt`` — mtime check). No tmpdir.
    * mode 'ppo' : materialize the AZNet warm-started from the PPO ``.zip`` and
      serialize it to a throwaway temp ``.ts.pt`` (tmpdir returned for cleanup).
    * mode 'random' : same, but from a fresh random AZNet."""
    from az_net import (load_az, from_ppo, AZNet, save_torchscript,
                        torchscript_export_path)
    mode, path = source["mode"], source["path"]
    if mode == "az":
        ts = torchscript_export_path(path)
        stale = (not os.path.exists(ts)) or \
            (os.path.getmtime(ts) < os.path.getmtime(path))
        if stale:
            print(f"[az-selfplay] exporting TorchScript sibling {ts}")
            save_torchscript(load_az(path), ts)
        else:
            print(f"[az-selfplay] using existing TorchScript {ts}")
        return ts, None
    import tempfile
    net = from_ppo(path) if mode == "ppo" else AZNet().eval()
    tmpdir = tempfile.mkdtemp(prefix="az_actor_ts_")
    ts = os.path.join(tmpdir, "model.ts.pt")
    save_torchscript(net, ts)
    print(f"[az-selfplay] exported temp TorchScript {ts} (mode={mode})")
    return ts, tmpdir


def _parse_actor_output(stdout: str, bo3: bool = False):
    """Aggregate one actor worker's stdout: (total_samples, wins_a, wins_b, draws).

    Sample counts always come from the terminal ``SELFPLAY: total_samples=`` line.
    Win tallies come from the appropriate result unit: bo1 counts per-GAME
    ``SELFPLAY: game … winner=`` lines; bo3 counts per-MATCH ``MATCH_RESULT:``
    lines (a bo3 match always ends 2-x, so there is no match draw)."""
    samples = wins_a = wins_b = draws = 0
    for line in stdout.splitlines():
        if line.startswith("SELFPLAY: total_samples="):
            for tok in line.split():
                if tok.startswith("total_samples="):
                    samples = int(tok.split("=", 1)[1])
        elif bo3 and line.startswith("MATCH_RESULT:"):
            if line.startswith("MATCH_RESULT: Player A wins"):
                wins_a += 1
            elif line.startswith("MATCH_RESULT: Player B wins"):
                wins_b += 1
        elif not bo3 and line.startswith("SELFPLAY: game "):
            if line.endswith("winner=A"):
                wins_a += 1
            elif line.endswith("winner=B"):
                wins_b += 1
            elif line.endswith("winner=DRAW"):
                draws += 1
    return samples, wins_a, wins_b, draws


def actor_selfplay_cmd(actor_bin, *, deck, seed, games, sims, worlds, model,
                       out_dir, deck_b=None, noise_eps=DEFAULT_ROOT_NOISE_EPS,
                       noise_alpha=DEFAULT_ROOT_NOISE_ALPHA,
                       temp_moves=DEFAULT_TEMP_MOVES, rng_seed=None,
                       bo3=False, sb_sims=DEFAULT_SB_SIMS,
                       sb_worlds=DEFAULT_SB_WORLDS,
                       sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                       sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                       sb_persist=bool(DEFAULT_SB_PERSIST),
                       merge_dupes=True,
                       td_n=DEFAULT_TD_N) -> list:
    """Build a ``bin/az_actor --selfplay`` argv.

    The single source of the actor CLI contract on the Python side — used by
    _generate_actor and bench_actor so the production and benchmark invocations
    can never drift apart (and both always pin the noise/temperature knobs
    instead of leaning on the actor's compiled-in defaults).

    ``deck`` is Player A's deck; ``deck_b`` is Player B's (None -> mirror = deck),
    so one actor process runs one (deck_a, deck_b) matchup batch.

    With ``bo3`` each of ``games`` units is a best-of-three MATCH (the actor spaces
    match seeds by 3 internally), and the ``sb_*`` knobs set the searched
    sideboard-root budget (``--sb-sims``/``--sb-worlds``/``--sb-max-depth``/
    ``--sb-rollout-turns``); they are inert for bo1."""
    cmd = [actor_bin, "--selfplay", "--deck", deck,
           "--seed", str(seed), "--games", str(games),
           "--sims", str(sims), "--worlds", str(worlds),
           "--model", model, "--out-dir", out_dir,
           "--noise-eps", str(noise_eps),
           "--noise-alpha", str(noise_alpha),
           "--temp-moves", str(temp_moves),
           "--td-n", str(td_n),
           "--merge-dupes", str(int(merge_dupes))]
    if bo3:
        cmd += ["--bo3",
                "--sb-sims", str(sb_sims),
                "--sb-worlds", str(sb_worlds),
                "--sb-max-depth", str(sb_max_depth),
                "--sb-rollout-turns", str(sb_rollout_turns),
                "--sb-persist", str(int(sb_persist))]
    if deck_b is not None and deck_b != deck:
        cmd += ["--deck-b", deck_b]
    if rng_seed is not None:
        cmd += ["--rng-seed", str(rng_seed)]
    return cmd


def _generate_actor(deck, *, source, schedule, sims, worlds, workers, temp_moves,
                    root_noise_eps, root_noise_alpha, out_dir, seed,
                    actor_bin, bo3=False, sb_sims=DEFAULT_SB_SIMS,
                    sb_worlds=DEFAULT_SB_WORLDS,
                    sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                    sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                    sb_persist=bool(DEFAULT_SB_PERSIST),
                    merge_dupes=True,
                    td_n=DEFAULT_TD_N) -> dict:
    import glob
    import shlex
    import shutil
    import subprocess
    import threading
    from collections import Counter

    ts_path, tmpdir = _ensure_actor_torchscript(source)
    out_dir = os.path.abspath(out_dir)
    # `games` is the number of scheduled MATCHES; with --bo3 the actor treats each
    # of a group's `games=n` units as a best-of-three match (same "games = matches"
    # meaning as the Python backend), so no unit conversion is needed here.
    games = len(schedule)
    unit = "matches" if bo3 else "games"

    # Group the schedule into per-(deck_a, deck_b) actor invocations so each actor
    # process runs one matchup batch (mirror or cross-deck) with a DISJOINT seed
    # range. Groups are run with bounded concurrency (<= workers at a time).
    groups = [((da, db), n) for (da, db), n in sorted(Counter(schedule).items())]
    total_groups = len(groups)

    pre = set(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    total_samples = 0
    agg = {"searched": 0, "fallback": 0, "wins_a": 0, "wins_b": 0, "draws": 0}

    # Live progress: a reader thread per group echoes each actor SELFPLAY: game
    # line as it lands (with a running cross-group total + ETA) while accumulating
    # the full stdout/stderr for the final parse.
    t_start = time.time()
    prog_lock = threading.Lock()
    prog = {"done": 0}

    # Progress unit == the scheduling unit: count per-MATCH MATCH_RESULT lines in
    # bo3 (each match may span up to 3 games) and per-GAME SELFPLAY lines in bo1, so
    # `done/games` and the ETA denominator stay consistent (games == #matches).
    def _is_progress_line(line):
        return (line.startswith("MATCH_RESULT:") if bo3
                else line.startswith("SELFPLAY: game "))

    def _pump(gi, stream, sink, is_stdout):
        for line in stream:
            sink.append(line)
            if is_stdout and _is_progress_line(line):
                with prog_lock:
                    prog["done"] += 1
                    done = prog["done"]
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (games - done)
                print(f"[az-selfplay] g{gi} {line.strip()} | total {done}/{games} "
                      f"{unit}, elapsed {_fmt_secs(elapsed)}, eta {_fmt_secs(eta)}",
                      flush=True)
        stream.close()

    def _launch(gi):
        (da, db), n = groups[gi]
        base = seed + gi * 100000
        cmd = actor_selfplay_cmd(
            actor_bin, deck=da, deck_b=db, seed=base, games=n,
            sims=sims, worlds=worlds, model=ts_path, out_dir=out_dir,
            noise_eps=root_noise_eps, noise_alpha=root_noise_alpha,
            temp_moves=temp_moves, rng_seed=seed + 100003 * (gi + 1),
            bo3=bo3, sb_sims=sb_sims, sb_worlds=sb_worlds,
            sb_max_depth=sb_max_depth, sb_rollout_turns=sb_rollout_turns,
            sb_persist=sb_persist, merge_dupes=merge_dupes, td_n=td_n)
        # Run from bin/ so the engine's getcwd-based RESOURCE_DIR resolves.
        p = subprocess.Popen(cmd, cwd=BIN_DIR, text=True, bufsize=1,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out_lines, err_lines = [], []
        threads = [
            threading.Thread(target=_pump, args=(gi, p.stdout, out_lines, True),
                             daemon=True),
            threading.Thread(target=_pump, args=(gi, p.stderr, err_lines, False),
                             daemon=True),
        ]
        for t in threads:
            t.start()
        return {"gi": gi, "da": da, "db": db, "n": n, "p": p, "cmd": cmd,
                "threads": threads, "out": out_lines, "err": err_lines}

    failed = []

    def _reap(rec):
        nonlocal total_samples
        rec["p"].wait()
        for t in rec["threads"]:
            t.join()
        if rec["p"].returncode != 0:
            failed.append((rec["gi"], rec["p"].returncode, rec["da"], rec["db"],
                           rec["cmd"], "".join(rec["err"])))
            return
        s, wa, wb, dr = _parse_actor_output("".join(rec["out"]), bo3=bo3)
        total_samples += s
        agg["wins_a"] += wa
        agg["wins_b"] += wb
        agg["draws"] += dr
        print(f"[az-selfplay] matchup {rec['da']}|{rec['db']}: {unit}={rec['n']} "
              f"samples={s} A={wa} B={wb} draws={dr}")

    cap = max(1, workers)
    active = []
    try:
        # Sliding pool: keep up to `cap` actor processes in flight and launch
        # the next matchup group the moment any one exits. Group durations vary
        # widely (bo3 matches, mixed matchups), so fixed launch batches would
        # idle the whole pool on each batch's slowest match. Per-group seeds are
        # assigned by group index at launch, so scheduling order never affects
        # results.
        next_gi = 0
        while next_gi < total_groups or active:
            while next_gi < total_groups and len(active) < cap:
                active.append(_launch(next_gi))
                next_gi += 1
            done = [rec for rec in active if rec["p"].poll() is not None]
            if not done:
                time.sleep(0.2)
                continue
            for rec in done:
                _reap(rec)
                active.remove(rec)
        if failed:
            for gi, rc, da, db, cmd, err in failed:
                # Surface enough to REPRODUCE the failing group in isolation: the
                # matchup, the exit code, the exact actor argv (copy-pasteable —
                # run it from BIN_DIR), and a generous stderr tail so a rich actor
                # diagnostic (e.g. az_mcts's DIVERGENCE dump, which spans several
                # lines) is not truncated away above the fatal line.
                tail = "\n".join(err.strip().splitlines()[-40:])
                repro = " ".join(shlex.quote(str(c)) for c in cmd)
                print(f"[az-selfplay] matchup group {gi} ({da} vs {db}) FAILED "
                      f"(exit {rc})\n  reproduce (run from {BIN_DIR}):\n    {repro}\n"
                      f"  stderr tail:\n{tail}")
            raise RuntimeError(
                f"az_actor self-play: {len(failed)} of {total_groups} matchup "
                f"group(s) failed (first: group {failed[0][0]}, "
                f"{failed[0][2]} vs {failed[0][3]}, exit {failed[0][1]}); "
                f"see the per-group FAILED block(s) above for the repro command "
                f"and stderr tail")
    finally:
        # Abnormal exit (a failed group raising above, KeyboardInterrupt):
        # don't leave live actor processes running detached.
        for rec in active:
            if rec["p"].poll() is None:
                rec["p"].terminate()
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    post = set(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    all_shards = sorted(post - pre)
    print(f"[az-selfplay] done: {total_samples} samples, {len(all_shards)} shards (ACTOR)")
    # The actor does search internally but does not emit searched/fallback tallies,
    # so those stay 0 for the actor backend (informational only; the trainer reads
    # shards from disk, not this dict).
    print(f"[az-selfplay] results A={agg['wins_a']} B={agg['wins_b']} "
          f"draws={agg['draws']}")
    return {"samples": total_samples, "shards": all_shards, "stats": agg,
            "out_dir": out_dir, "source": source}


# ----------------------------------------------------------------------
# Expert demonstrations (scripted:hard BC shards)
# ----------------------------------------------------------------------

def _play_match_expert(env, agent, seed, opp_agent=None, focus_is_a=None):
    """Play one match with scripted:hard piloting BOTH seats, recording every
    multi-choice decision as a behavior-cloning sample: pi is a ONE-HOT on the
    expert's chosen action (vs self-play's visit distribution), z is the usual
    per-game outcome from the mover's perspective. Unlike :func:`_play_match`,
    non-search-safe prompts are recorded too — a one-hot target is valid
    anywhere, and those prompts are exactly the ones the net answers by raw
    policy at self-play time, so BC coverage there raises fallback quality.

    ``opp_agent`` (with ``focus_is_a`` naming the expert's seat) hands the
    OTHER seat to a different agent — the weak-opponent expert mode: the expert
    keeps the focus seat, and ONLY the focus seat's decisions are recorded
    (the opponent's play is not demonstration-quality, so cloning it would
    poison the policy targets). ``_backfill_and_pack`` prices every sample by
    its own mover's outcome, so a one-seat sample list needs no other change.

    Returns (samples, game_winners, dropped) with the same boundary/truncation
    semantics as :func:`_play_match` (whose shared helpers — ``one_hot_sample``,
    ``winner_from_reward``, ``_drop_unfinished`` — this loop uses), but kept a
    separate function: :func:`_play_match` is pinned by
    ``test_sideboard_selfplay.py`` (semantic, default ``make check``) and
    ``test_actor_trains.py`` (statistical, opt-in ``actor`` tier), whose hard
    constraint is that its rng draw order stay intact)."""
    env.reset(seed=seed)
    agent.new_game()
    if opp_agent is not None:
        opp_agent.new_game()
    samples = []
    game_winners = []
    game_idx = 0
    done = False
    dropped = 0
    while not done:
        num_choices = env._num_choices
        if num_choices > 1:
            mover_is_focus = (opp_agent is None or
                              bool(env._obs[_SELF_IS_A_IDX] > 0.5) == focus_is_a)
            seat_agent = agent if mover_is_focus else opp_agent
            action = int(seat_agent.act(env._obs, num_choices))
            if mover_is_focus:
                sample = one_hot_sample(env._obs, num_choices, action)
                sample["game_idx"] = game_idx
                samples.append(sample)
        else:
            action = 0
        _, reward, terminated, truncated, info = env.step(action)
        boundary = bool(info.get("game_result"))
        if boundary:
            game_winners.append(winner_from_reward(reward))
            game_idx += 1
            agent.new_game()
            if opp_agent is not None:
                opp_agent.new_game()
        if terminated or truncated:
            done = True
            if terminated and not boundary:
                game_winners.append(winner_from_reward(reward))
            if truncated:
                samples, dropped = _drop_unfinished(samples, game_winners)
    return samples, game_winners, dropped


def generate_expert(decks, *, games: int = 16, roster: Optional[list] = None,
                    mirror_frac: float = DEFAULT_MIRROR_FRAC, bo3: bool = True,
                    seed: int = 1, out_dir: Optional[str] = None,
                    opponent: Optional[str] = None) -> dict:
    """Write EXPERT demonstration shards: scripted:hard vs scripted:hard matches
    over the same seeded focus-vs-roster schedule self-play uses, packed into the
    trainer-compatible ``shard_*.npz`` format (pi = one-hot expert action).

    Rationale (the doomsday fix): a deck whose win condition is a long exact
    combo line is invisible to both PPO exploration and MCTS — the warm-started
    value net scores every mid-combo state as lost (the policy never wins with
    the deck), so search prunes the line before sampling it. The scripted agent
    has those combo lines hand-coded; recording its games as ordinary replay
    shards puts prior mass on the line AND prices mid-combo states by games the
    combo actually wins, breaking the chicken-and-egg so search can extend the
    line from there. ``decks`` (str or list) is the focus pool — the deck(s)
    needing demonstrations; opponents follow ``mirror_frac``/``roster`` like
    self-play. Torch-free and fast (no search), so it runs serially.

    ``opponent`` (a scripted-agent spec, e.g. ``"scripted:random"``) hands the
    opponent seat to a WEAKER tier while scripted:hard keeps the focus seat,
    and records ONLY the focus seat's decisions. This is the fix for the case
    the default mode gets wrong: when the expert deck LOSES its scripted:hard
    matchups (doomsday does, everywhere but the mirror), hard-vs-hard shards
    stamp z=-1 on every step of the combo line and reinforce the very value
    collapse they were meant to break. Against a weak opponent the combo
    actually wins, so the line's states are priced z=+1 — an optimistic
    curriculum bias that later self-play recalibrates. Mirror games keep the
    weak tier on the non-focus seat. Default None: hard both seats, both
    recorded (the original behavior, schedule-identical for a given seed).

    Expert rows carry no search root value, so their shard columns are
    ``q = NaN``, ``explored = 0`` and ``td_q = z`` — the n-step TD target
    degenerates to the plain per-game outcome for behavior-cloning data."""
    from search_env import SearchRoboMageEnv
    from scripted_agent import make_agent

    focus = [decks] if isinstance(decks, str) else list(decks)
    roster = list(roster) if roster else league_roster()
    out_dir = out_dir or os.path.join(_AZ_DATA_DIR, GEN_STEM)
    os.makedirs(out_dir, exist_ok=True)
    if bo3:
        _discard_pre_bo3_shards(out_dir)

    # The _ex form draws identically to build_matchup_schedule, so the pairs
    # (and default-mode shards) are unchanged; focus_is_a routes the seats.
    schedule = build_matchup_schedule_ex(focus, roster, games, mirror_frac, seed)
    seats_txt = ("scripted:hard both seats" if opponent is None else
                 f"scripted:hard focus seat vs {opponent}, focus rows only")
    print(f"[az-expert] focus={','.join(focus)} matches={games} "
          f"bo3={bo3} mirror_frac={mirror_frac} ({seats_txt})")
    print(f"[az-expert] matchups: "
          f"{_schedule_summary([(da, db) for da, db, _ in schedule])}")
    print(f"[az-expert] out_dir={out_dir}")

    da0, db0 = (schedule[0][0], schedule[0][1]) if schedule else (None, None)
    env = SearchRoboMageEnv(deck_a=da0, deck_b=db0, bo3=bo3, auto_sideboard=False,
                           binary_path=INTERACTIVE_BINARY)
    # ScriptedAgent is seat-aware (keys its state and deck name off the obs's
    # self-is-A flag), so ONE shared hard-tier agent pilots both seats.
    agent = make_agent("scripted:hard")
    opp_agent = make_agent(opponent) if opponent else None
    total_samples = 0
    shards = []
    buf = []
    shard_n = 0
    stats = {"wins_a": 0, "wins_b": 0, "draws": 0, "games": 0, "dropped": 0,
             "focus_match_wins": 0}
    try:
        for m, (da, db, focus_is_a) in enumerate(schedule):
            env._deck_a, env._deck_b = da, db
            agent.set_deck_names(da, db)
            if opp_agent is not None:
                opp_agent.set_deck_names(da, db)
            samples, game_winners, dropped = _play_match_expert(
                env, agent, seed=seed + m,
                opp_agent=opp_agent, focus_is_a=focus_is_a)
            buf.append(_backfill_and_pack(samples, game_winners))
            total_samples += len(samples)
            stats["games"] += len(game_winners)
            stats["dropped"] += dropped
            mwinner = _match_winner(game_winners)
            stats["wins_a"] += int(mwinner == "A")
            stats["wins_b"] += int(mwinner == "B")
            stats["draws"] += int(mwinner == "DRAW")
            stats["focus_match_wins"] += int(
                mwinner == ("A" if focus_is_a else "B"))
            if _buffered(buf) >= FLUSH_SAMPLES:
                shards.append(_write_shard(out_dir, _concat(buf), shard_n))
                shard_n += 1
                buf = []
        if buf:
            shards.append(_write_shard(out_dir, _concat(buf), shard_n))
    finally:
        env.close()
    print(f"[az-expert] done: {total_samples} samples, {len(shards)} shard(s)  "
          f"A={stats['wins_a']} B={stats['wins_b']} draws={stats['draws']}  "
          f"focus won {stats['focus_match_wins']}/{len(schedule)} matches"
          + (f"  dropped={stats['dropped']}" if stats["dropped"] else ""))
    return {"samples": total_samples, "shards": shards, "stats": stats,
            "out_dir": out_dir}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _resolve_use_actor(args) -> Optional[bool]:
    """--actor -> True, --no-actor -> False, neither -> None (AUTO)."""
    if getattr(args, "actor", False):
        return True
    if getattr(args, "no_actor", False):
        return False
    return None


def run(args) -> None:
    """train.py dispatch entry."""
    if getattr(args, "expert", False):
        # Expert shards are always bo3: the pooled az_data/gen window is bo3
        # (see _discard_pre_bo3_shards) and bo1 shards would mix silently.
        generate_expert(args.deck, games=args.games,
                        mirror_frac=getattr(args, "mirror_frac", DEFAULT_MIRROR_FRAC),
                        bo3=True, out_dir=args.out,
                        seed=args.seed if args.seed is not None else 1,
                        opponent=getattr(args, "expert_opponent", None))
        return
    generate(args.deck, games=args.games, sims=args.sims, worlds=args.worlds,
             workers=args.workers, checkpoint=args.checkpoint,
             temp_moves=args.temp_moves, seed=args.seed if args.seed is not None else 1,
             out_dir=args.out, use_actor=_resolve_use_actor(args),
             mirror_frac=getattr(args, "mirror_frac", DEFAULT_MIRROR_FRAC),
             sb_sims=getattr(args, "sb_sims", DEFAULT_SB_SIMS),
             sb_worlds=getattr(args, "sb_worlds", DEFAULT_SB_WORLDS),
             sb_max_depth=getattr(args, "sb_max_depth", DEFAULT_SB_MAX_DEPTH),
             sb_rollout_turns=getattr(args, "sb_rollout_turns",
                                      DEFAULT_SB_ROLLOUT_TURNS),
             sb_persist=bool(getattr(args, "sb_persist", DEFAULT_SB_PERSIST)),
             merge_dupes=bool(getattr(args, "merge_dupes", 1)),
             td_n=int(getattr(args, "td_n", DEFAULT_TD_N)))


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AlphaZero self-play data generation")
    ap.add_argument("--deck", default="delver",
                    help="Focus deck (.dk stem) — its opponent is a mirror with "
                         "P=--mirror-frac, else a uniform league-roster draw")
    ap.add_argument("--games", type=int, default=DEFAULT_AZ_GAMES)
    ap.add_argument("--sims", type=int, default=DEFAULT_AZ_SIMS,
                    help="PUCT sims per decision, TOTAL across --worlds")
    ap.add_argument("--worlds", type=int, default=DEFAULT_AZ_WORLDS)
    ap.add_argument("--workers", type=int, default=None,
                    help="Worker processes (default max(1, cpu-2))")
    ap.add_argument("--checkpoint", default=None,
                    help="AZ (.pt) / PPO (.zip) checkpoint or 'gen' "
                         "(default: generalist AZ ckpt, else gen PPO warm-start, "
                         "else random)")
    ap.add_argument("--temp-moves", type=int, default=DEFAULT_TEMP_MOVES)
    ap.add_argument("--merge-dupes", type=int, default=1,
                    help="Merge interchangeable duplicate menu actions into one "
                         "search edge (decode.menu_merge_reps; default 1, pass "
                         "0 for the legacy per-copy edges)")
    ap.add_argument("--td-n", type=int, default=DEFAULT_TD_N,
                    help="n-step TD horizon baked into each sample's td_q "
                         "(default %d); the chain is shortened at the next "
                         "exploratory move and falls back to the game outcome "
                         "when the window reaches the end of the game"
                         % DEFAULT_TD_N)
    ap.add_argument("--sb-sims", type=int, default=DEFAULT_SB_SIMS,
                    help="PUCT sims at a bo3 sideboard root (bo3 only; default %d)"
                         % DEFAULT_SB_SIMS)
    ap.add_argument("--sb-worlds", type=int, default=DEFAULT_SB_WORLDS,
                    help="Determinized worlds at a bo3 sideboard root (default %d)"
                         % DEFAULT_SB_WORLDS)
    ap.add_argument("--sb-max-depth", type=int, default=DEFAULT_SB_MAX_DEPTH,
                    help="Descent depth cap at a bo3 sideboard root (default %d)"
                         % DEFAULT_SB_MAX_DEPTH)
    ap.add_argument("--sb-rollout-turns", type=int,
                    default=DEFAULT_SB_ROLLOUT_TURNS,
                    help="Leaf-rollout horizon at a bo3 sideboard root, in "
                         "player turns (0 = off; default %d)"
                         % DEFAULT_SB_ROLLOUT_TURNS)
    ap.add_argument("--sb-persist", type=int, default=DEFAULT_SB_PERSIST,
                    help="Persist trees + rollout memo across a bo3 sideboard "
                         "boundary (1/0; default %d)" % DEFAULT_SB_PERSIST)
    ap.add_argument("--mirror-frac", type=float, default=DEFAULT_MIRROR_FRAC,
                    help="P(opponent deck == focus deck) per game (default %.2f); "
                         "else a uniform league-roster draw" % DEFAULT_MIRROR_FRAC)
    ap.add_argument("--out", default=None, help="Output dir (default az_data/gen)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--expert", action="store_true",
                    help="Write EXPERT demonstration shards instead of self-play: "
                         "scripted:hard both seats, pi = one-hot expert action "
                         "(always bo3; sims/worlds/checkpoint ignored)")
    ap.add_argument("--expert-opponent", default=None,
                    help="Expert mode only: scripted-agent spec for the OPPONENT "
                         "seat (e.g. scripted:random / scripted:easy); "
                         "scripted:hard keeps the focus seat and ONLY its "
                         "decisions are recorded — so a combo deck's expert "
                         "shards come from games the combo actually wins")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--actor", action="store_true",
                   help="Force the C++ az_actor self-play backend (error if not built)")
    g.add_argument("--no-actor", action="store_true",
                   help="Force the pure-Python backend (skip the actor even if built)")
    return ap


if __name__ == "__main__":
    run(_build_arg_parser().parse_args())
