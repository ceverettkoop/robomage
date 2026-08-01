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
(:func:`_play_match_vs_scripted`). ``f=1.0`` trains entirely against the scripted
agent. That path is Python-backend only (the C++ actor is pure self-play).

:func:`generate_expert` additionally writes EXPERT demonstration shards —
scripted:hard piloting both seats, pi = one-hot on the expert's action — in the
same shard format, so the trainer behavior-clones hand-coded lines (e.g. the
Doomsday combo) that neither PPO exploration nor prior-guided search discovers.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np

try:
    from env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX, _IS_SIDEBOARD_IDX
    from cli_spec import (BIN_DIR, DEFAULT_SB_SIMS, DEFAULT_SB_WORLDS,
                          DEFAULT_SB_MAX_DEPTH, DEFAULT_SB_ROLLOUT_TURNS,
                          DEFAULT_SB_PERSIST)
    from opponents import GEN_STEM
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX, _IS_SIDEBOARD_IDX
    from train.cli_spec import (BIN_DIR, DEFAULT_SB_SIMS, DEFAULT_SB_WORLDS,
                                DEFAULT_SB_MAX_DEPTH, DEFAULT_SB_ROLLOUT_TURNS,
                                DEFAULT_SB_PERSIST)
    from train.opponents import GEN_STEM

_AZ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "az_data")
_ACTOR_BIN = os.path.join(BIN_DIR, "az_actor")
_DECKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "bin", "resources", "decks")
_LEAGUE_DECKS_DIR = os.path.join(_DECKS_DIR, "league")

# Defaults (AlphaZero-style)
DEFAULT_ROOT_NOISE_EPS = 0.25
DEFAULT_ROOT_NOISE_ALPHA = 1.0
DEFAULT_TEMP_MOVES = 20        # sample-from-visits for the first N real decisions, then argmax
# Sideboard-rooted search budget (DEFAULT_SB_SIMS/WORLDS/MAX_DEPTH) lives in
# cli_spec — the single home shared with opponents.SearchController and the CLI
# flag defaults — and is imported above.
DEFAULT_MIRROR_FRAC = 0.25     # P(opponent deck == focus deck) per self-play game
FLUSH_SAMPLES = 4096           # write a shard once this many samples accumulate
HEARTBEAT_MOVES = 25           # Python backend: mid-game progress line every N decisions


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
    each worker reconstructs its own net (torch objects don't cross processes)."""
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
    from az_net import AZNet, load_az, from_ppo
    if source["mode"] == "az":
        return load_az(source["path"])
    if source["mode"] == "ppo":
        return from_ppo(source["path"])
    return AZNet().eval()


# ----------------------------------------------------------------------
# One game of self-play
# ----------------------------------------------------------------------

def _play_match(env, evaluator, rng, *, sims, worlds, temp_moves,
                root_noise_eps, root_noise_alpha, seed, on_progress=None,
                sb_sims=DEFAULT_SB_SIMS, sb_worlds=DEFAULT_SB_WORLDS,
                sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                sb_persist=bool(DEFAULT_SB_PERSIST)):
    """Play one match (bo1: a single game; bo3: a best-of-three) and return
    (samples, game_winners, searched, fallback, dropped, sb_stats) — sb_stats
    reports the boundary persistence's reused visits + rollout-memo hits.

    ``samples`` is a list of dicts {obs, pi, mask, mover_is_a, game_idx}; each is
    tagged with the 0-based index of the game it was played in. ``game_winners``
    is the ordered list of each COMPLETED game's winner ('A'/'B'/None for a draw),
    so ``game_winners[sample['game_idx']]`` prices that sample. ``dropped`` counts
    samples discarded because the match TRUNCATED mid-game (the in-progress game
    has no result, so its samples carry no valid z).

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
    perturb the game or ``rng``, so play stays byte-identical with or without
    it (the actor trains-identically gate replays this exact function)."""
    from mcts import run_search, sb_root_key, sb_pick_descriptor, walk_reuse_root

    obs, _ = env.reset(seed=seed)
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

    while not done:
        num_choices = env._num_choices
        priority_is_a = bool(env._obs[_SELF_IS_A_IDX] > 0.5)
        searchable = bool(env.last_search_safe) and num_choices > 1

        if searchable:
            # A bo3 sideboard root is more expensive per sim (restore re-crosses
            # init_ecs + deck load + shuffle) and has a game-long horizon, so it
            # gets its own budget. Key ONLY off is_sideboard_phase — is_post_board
            # / game_number still reflect the just-ended game at a g1->g2 root.
            if bool(env._obs[_IS_SIDEBOARD_IDX] > 0.5):
                kw = dict(sims=sb_sims, worlds=sb_worlds,
                          max_depth=sb_max_depth,
                          rollout_turns=sb_rollout_turns,
                          root_noise_eps=root_noise_eps,
                          root_noise_alpha=root_noise_alpha, rng=rng)
                if sb_persist:
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
                result = run_search(env, evaluator, sims=sims, worlds=worlds,
                                    root_noise_eps=root_noise_eps,
                                    root_noise_alpha=root_noise_alpha, rng=rng)
            visits = result.policy_target(1.0)          # normalized visit counts
            pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
            pi[:num_choices] = visits.astype(np.float32)
            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[:num_choices] = True
            samples.append({"obs": env._obs.copy(), "pi": pi, "mask": mask,
                            "mover_is_a": priority_is_a, "game_idx": game_idx})
            searched += 1
            # Temperature schedule is per-game: the first temp_moves decisions of
            # EACH game sample from the visit counts, then switch to argmax.
            if game_move < temp_moves:
                action = int(rng.choice(num_choices, p=visits))
            else:
                action = result.best_action()
        else:
            priors, _ = evaluator.evaluate(env._obs, num_choices)
            action = int(np.argmax(priors))
            fallback += 1

        # Latch every stepped action while a boundary is live (the walk plays
        # them through the stored trees at the next pick; only sideboard picks
        # contribute memo descriptors). Pre-step obs, matching the actor.
        if sb_boundary is not None:
            sb_boundary["played"].append(int(action))
            d = sb_pick_descriptor(env._obs, int(action))
            if d is not None:
                sb_boundary["picks"].append(d)

        obs, reward, terminated, truncated, info = env.step(action)
        move += 1
        game_move += 1
        # A GAME_RESULT landed on this step -> the game the just-stepped action
        # belonged to has finished. Record its winner and advance to the next game.
        boundary = bool(info.get("game_result"))
        if boundary:
            game_winners.append("A" if reward > 0 else ("B" if reward < 0 else None))
            game_idx += 1
            game_move = 0
            sb_boundary = None
        if on_progress is not None and move % HEARTBEAT_MOVES == 0:
            on_progress(move, searched, fallback)
        if terminated or truncated:
            done = True
            if terminated and not boundary:
                # bo1 mode emits no GAME_RESULT line — the single game ends with a
                # plain "Player X wins" + terminated. Price the in-progress game
                # from the terminal reward sign (bo3's final game already recorded
                # via the boundary branch above, so guard on `not boundary`).
                game_winners.append(
                    "A" if reward > 0 else ("B" if reward < 0 else None))
            if truncated:
                # The match hit MAX_STEPS_BO3 mid-game: keep samples from the
                # games that finished (they have a z), drop the in-progress game's
                # samples (no result yet -> no valid target).
                n_done = len(game_winners)
                kept = [s for s in samples if s["game_idx"] < n_done]
                dropped = len(samples) - len(kept)
                samples = kept
    return samples, game_winners, searched, fallback, dropped, sb_stats


def _play_match_vs_scripted(env, evaluator, agent, rng, *, net_is_a, sims, worlds,
                            temp_moves, root_noise_eps, root_noise_alpha, seed,
                            on_progress=None,
                            sb_sims=DEFAULT_SB_SIMS, sb_worlds=DEFAULT_SB_WORLDS,
                            sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                            sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                            sb_persist=bool(DEFAULT_SB_PERSIST)):
    """Play one match with the net+MCTS on ONE seat (``net_is_a``) and the
    rule-based scripted:hard ``agent`` on the other, returning the same tuple as
    :func:`_play_match`.

    Only the NET seat's decisions are searched and recorded as samples — the
    scripted seat is environment, not a policy to imitate. That is sound for the
    value target as-is: :func:`_backfill_and_pack` prices every sample per-mover
    (z = +1 iff its game's winner is that sample's mover), so a one-seat sample
    stream needs no special handling.

    Kept separate from :func:`_play_match` for the same reason as
    :func:`_play_match_expert`: that function is pinned by the actor
    trains-identically gate (and test_sideboard_selfplay.py) and must not grow
    branches."""
    from mcts import run_search, sb_root_key, sb_pick_descriptor, walk_reuse_root

    obs, _ = env.reset(seed=seed)
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
    sb_boundary = None
    sb_stats = {"sb_reused_visits": 0, "sb_memo_hits": 0}

    while not done:
        num_choices = env._num_choices
        priority_is_a = bool(env._obs[_SELF_IS_A_IDX] > 0.5)
        net_to_move = (priority_is_a == net_is_a)
        searchable = (net_to_move and bool(env.last_search_safe)
                      and num_choices > 1)

        if searchable:
            # A bo3 sideboard root is more expensive per sim (restore re-crosses
            # init_ecs + deck load + shuffle) and has a game-long horizon, so it
            # gets its own budget. Key ONLY off is_sideboard_phase — is_post_board
            # / game_number still reflect the just-ended game at a g1->g2 root.
            if bool(env._obs[_IS_SIDEBOARD_IDX] > 0.5):
                kw = dict(sims=sb_sims, worlds=sb_worlds,
                          max_depth=sb_max_depth,
                          rollout_turns=sb_rollout_turns,
                          root_noise_eps=root_noise_eps,
                          root_noise_alpha=root_noise_alpha, rng=rng)
                if sb_persist:
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
                result = run_search(env, evaluator, sims=sims, worlds=worlds,
                                    root_noise_eps=root_noise_eps,
                                    root_noise_alpha=root_noise_alpha, rng=rng)
            visits = result.policy_target(1.0)          # normalized visit counts
            pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
            pi[:num_choices] = visits.astype(np.float32)
            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[:num_choices] = True
            samples.append({"obs": env._obs.copy(), "pi": pi, "mask": mask,
                            "mover_is_a": priority_is_a, "game_idx": game_idx})
            searched += 1
            # Temperature schedule is per-game: the first temp_moves decisions of
            # EACH game sample from the visit counts, then switch to argmax.
            if game_move < temp_moves:
                action = int(rng.choice(num_choices, p=visits))
            else:
                action = result.best_action()
        elif net_to_move:
            priors, _ = evaluator.evaluate(env._obs, num_choices)
            action = int(np.argmax(priors))
            fallback += 1
        else:
            # Scripted seat: no search, no sample, no counters. The hard-tier
            # agent answers every prompt kind, including the bo3 sideboard
            # prompts (auto_sideboard=False), as in generate_expert.
            action = int(agent.act(env._obs, num_choices)) if num_choices > 1 else 0

        # Latch every stepped action while a boundary is live (the walk plays
        # them through the stored trees at the next pick; only sideboard picks
        # contribute memo descriptors). Pre-step obs, matching the actor. The
        # scripted seat's actions latch too — the walk must replay the TRUE
        # action sequence, whoever played it.
        if sb_boundary is not None:
            sb_boundary["played"].append(int(action))
            d = sb_pick_descriptor(env._obs, int(action))
            if d is not None:
                sb_boundary["picks"].append(d)

        obs, reward, terminated, truncated, info = env.step(action)
        move += 1
        game_move += 1
        # A GAME_RESULT landed on this step -> the game the just-stepped action
        # belonged to has finished. Record its winner and advance to the next game.
        boundary = bool(info.get("game_result"))
        if boundary:
            game_winners.append("A" if reward > 0 else ("B" if reward < 0 else None))
            game_idx += 1
            game_move = 0
            sb_boundary = None
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
                game_winners.append(
                    "A" if reward > 0 else ("B" if reward < 0 else None))
            if truncated:
                # The match hit MAX_STEPS_BO3 mid-game: keep samples from the
                # games that finished (they have a z), drop the in-progress game's
                # samples (no result yet -> no valid target).
                n_done = len(game_winners)
                kept = [s for s in samples if s["game_idx"] < n_done]
                dropped = len(samples) - len(kept)
                samples = kept
    return samples, game_winners, searched, fallback, dropped, sb_stats


def _backfill_and_pack(samples, game_winners):
    """Fill z per sample from its mover's perspective vs the winner of the GAME the
    sample belongs to (``game_winners[game_idx]``), then pack to arrays. A drawn
    game (winner None) -> z=0."""
    n = len(samples)
    obs = np.zeros((n, OBS_SIZE), dtype=np.float32)
    pi = np.zeros((n, MAX_ACTIONS), dtype=np.float32)
    z = np.zeros((n,), dtype=np.float32)
    mask = np.zeros((n, MAX_ACTIONS), dtype=bool)
    for i, s in enumerate(samples):
        obs[i] = s["obs"]
        pi[i] = s["pi"]
        mask[i] = s["mask"]
        winner = game_winners[s["game_idx"]]
        if winner is None:
            z[i] = 0.0
        else:
            mover_won = (winner == "A") == s["mover_is_a"]
            z[i] = 1.0 if mover_won else -1.0
    return obs, pi, z, mask


def _write_shard(out_dir, arrays, n_idx):
    ts = time.strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()
    path = os.path.join(out_dir, f"shard_{ts}_{pid}_{n_idx}.npz")
    obs, pi, z, mask = arrays
    np.savez_compressed(path, obs=obs, pi=pi, z=z, mask=mask)
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
            scripted_seats=None):
    """Play this worker's slice of the matchup schedule. ``matchups`` is a list of
    per-MATCH (deck_a, deck_b) pairs (mirror or cross-deck); the env's decks are
    swapped per match before its reset respawns the engine. With ``bo3`` each
    matchup is a best-of-three yielding up to three games of samples (decks stay
    fixed across the games of one match).

    ``scripted_seats`` (when given) is a parallel list of per-match
    Optional[bool]: ``None`` = pure self-play (the net pilots both seats), else
    the value is ``net_is_a`` for a match whose OTHER seat is piloted by
    scripted:hard (see :func:`_play_match_vs_scripted`)."""
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
    env = SearchRoboMageEnv(deck_a=da0, deck_b=db0, bo3=bo3, auto_sideboard=False)
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
            if net_is_a is None:
                samples, game_winners, searched, fallback, dropped, sb_st = _play_match(
                    env, evaluator, rng, sims=sims, worlds=worlds,
                    temp_moves=temp_moves, root_noise_eps=root_noise_eps,
                    root_noise_alpha=root_noise_alpha, seed=seed,
                    on_progress=beat, sb_sims=sb_sims, sb_worlds=sb_worlds,
                    sb_max_depth=sb_max_depth, sb_rollout_turns=sb_rollout_turns,
                    sb_persist=sb_persist)
            else:
                agent.set_deck_names(*matchups[m])
                samples, game_winners, searched, fallback, dropped, sb_st = \
                    _play_match_vs_scripted(
                        env, evaluator, agent, rng, net_is_a=bool(net_is_a),
                        sims=sims, worlds=worlds,
                        temp_moves=temp_moves, root_noise_eps=root_noise_eps,
                        root_noise_alpha=root_noise_alpha, seed=seed,
                        on_progress=beat, sb_sims=sb_sims, sb_worlds=sb_worlds,
                        sb_max_depth=sb_max_depth,
                        sb_rollout_turns=sb_rollout_turns, sb_persist=sb_persist)
            obs, pi, z, mask = _backfill_and_pack(samples, game_winners)
            buf.append((obs, pi, z, mask))
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
            if sum(len(b[2]) for b in buf) >= FLUSH_SAMPLES:
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
    obs = np.concatenate([b[0] for b in buf], axis=0)
    pi = np.concatenate([b[1] for b in buf], axis=0)
    z = np.concatenate([b[2] for b in buf], axis=0)
    mask = np.concatenate([b[3] for b in buf], axis=0)
    return obs, pi, z, mask


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


def generate(deck: str, *, games: int = 10, sims: int = 256, worlds: int = 4,
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
             scripted_opponent_frac: float = 0.0,
             bo3: bool = False) -> dict:
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
    become samples (see :func:`_play_match_vs_scripted`). Which matches get a
    scripted opponent is drawn from a dedicated seeded RNG stream indexed per
    match, so it is reproducible and independent of the worker count. The C++
    actor has no scripted-opponent support, so a nonzero fraction FORCES the
    Python backend (and is a loud error alongside an explicit ``use_actor=True``).

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
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    workers = max(1, min(workers, games))
    out_dir = out_dir or os.path.join(_AZ_DATA_DIR, GEN_STEM)
    os.makedirs(out_dir, exist_ok=True)
    if roster is None:
        roster = league_roster()
    focus = list(focus_decks) if focus_decks else [deck]

    have_actor = os.path.exists(_ACTOR_BIN)
    if bo3:
        # One-time discard of legacy bo1 shards before the first bo3 run. This is
        # backend-agnostic (the .bo3_migrated sentinel gates it either way) and
        # runs regardless of which backend generates the bo3 shards below.
        _discard_pre_bo3_shards(out_dir)
    if use_actor is None:
        use_actor = have_actor
        chosen = "AUTO"
    else:
        chosen = "forced"
    if use_actor and scripted_opponent_frac > 0.0:
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

    sched_ex = build_matchup_schedule_ex(focus, roster, games, mirror_frac, seed)
    schedule = [(a, b) for a, b, _ in sched_ex]
    scripted_seats = None
    if scripted_opponent_frac > 0.0:
        # Dedicated RNG stream (never touches the schedule's draws) indexed per
        # MATCH, so the scripted assignment is reproducible and independent of
        # how the schedule is sliced across workers. Entry = net_is_a; None =
        # pure self-play for that match.
        srng = np.random.default_rng([seed, 0x5C819])
        scripted_seats = [(focus_is_a if srng.random() < scripted_opponent_frac
                           else None)
                          for (_a, _b, focus_is_a) in sched_ex]
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
    if scripted_seats is not None:
        n_scripted = sum(1 for s in scripted_seats if s is not None)
        print(f"[az-selfplay] scripted-opponent games: {n_scripted}/{len(schedule)} "
              f"(frac={scripted_opponent_frac}; scripted:hard on the opponent seat, "
              f"net samples only)")
    print(f"[az-selfplay] backend={'ACTOR' if use_actor else 'PYTHON'} ({chosen}); "
          f"az_actor {'present' if have_actor else 'absent'}")

    common = dict(source=source, schedule=schedule, sims=sims, worlds=worlds,
                  workers=workers, temp_moves=temp_moves,
                  root_noise_eps=root_noise_eps, root_noise_alpha=root_noise_alpha,
                  out_dir=out_dir, seed=seed)
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
# Python multiprocess backend
# ----------------------------------------------------------------------

def _generate_python(deck, *, source, schedule, sims, worlds, workers, temp_moves,
                     root_noise_eps, root_noise_alpha, out_dir, seed,
                     bo3=False, sb_sims=DEFAULT_SB_SIMS, sb_worlds=DEFAULT_SB_WORLDS,
                     sb_max_depth=DEFAULT_SB_MAX_DEPTH,
                     sb_rollout_turns=DEFAULT_SB_ROLLOUT_TURNS,
                     sb_persist=bool(DEFAULT_SB_PERSIST),
                     scripted_seats=None) -> dict:
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
                              sb_rollout_turns, sb_persist, seat_slices[wi]))
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
                       sb_persist=bool(DEFAULT_SB_PERSIST)) -> list:
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
           "--temp-moves", str(temp_moves)]
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
                    sb_persist=bool(DEFAULT_SB_PERSIST)) -> dict:
    import glob
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
            sb_persist=sb_persist)
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
        return {"gi": gi, "da": da, "db": db, "n": n, "p": p,
                "threads": threads, "out": out_lines, "err": err_lines}

    failed = []

    def _reap(rec):
        nonlocal total_samples
        rec["p"].wait()
        for t in rec["threads"]:
            t.join()
        if rec["p"].returncode != 0:
            failed.append((rec["gi"], rec["p"].returncode, "".join(rec["err"])))
            return
        s, wa, wb, dr = _parse_actor_output("".join(rec["out"]), bo3=bo3)
        total_samples += s
        agg["wins_a"] += wa
        agg["wins_b"] += wb
        agg["draws"] += dr
        print(f"[az-selfplay] matchup {rec['da']}|{rec['db']}: {unit}={rec['n']} "
              f"samples={s} A={wa} B={wb} draws={dr}")

    cap = max(1, workers)
    try:
        for start in range(0, total_groups, cap):
            batch = [_launch(gi) for gi in range(start, min(start + cap, total_groups))]
            for rec in batch:
                _reap(rec)
        if failed:
            for gi, rc, err in failed:
                tail = "\n".join(err.strip().splitlines()[-10:])
                print(f"[az-selfplay] matchup group {gi} FAILED (exit {rc}):\n{tail}")
            raise RuntimeError(
                f"az_actor self-play: {len(failed)} of {total_groups} matchup "
                f"group(s) failed")
    finally:
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

def _play_match_expert(env, agent, seed):
    """Play one match with scripted:hard piloting BOTH seats, recording every
    multi-choice decision as a behavior-cloning sample: pi is a ONE-HOT on the
    expert's chosen action (vs self-play's visit distribution), z is the usual
    per-game outcome from the mover's perspective. Unlike :func:`_play_match`,
    non-search-safe prompts are recorded too — a one-hot target is valid
    anywhere, and those prompts are exactly the ones the net answers by raw
    policy at self-play time, so BC coverage there raises fallback quality.
    Returns (samples, game_winners, dropped) with the same boundary/truncation
    semantics as :func:`_play_match` (kept separate: that function is pinned by
    the actor trains-identically gate and must not grow branches)."""
    env.reset(seed=seed)
    agent.new_game()
    samples = []
    game_winners = []
    game_idx = 0
    done = False
    dropped = 0
    while not done:
        num_choices = env._num_choices
        if num_choices > 1:
            priority_is_a = bool(env._obs[_SELF_IS_A_IDX] > 0.5)
            action = int(agent.act(env._obs, num_choices))
            pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
            pi[action] = 1.0
            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[:num_choices] = True
            samples.append({"obs": env._obs.copy(), "pi": pi, "mask": mask,
                            "mover_is_a": priority_is_a, "game_idx": game_idx})
        else:
            action = 0
        _, reward, terminated, truncated, info = env.step(action)
        boundary = bool(info.get("game_result"))
        if boundary:
            game_winners.append("A" if reward > 0 else ("B" if reward < 0 else None))
            game_idx += 1
            agent.new_game()
        if terminated or truncated:
            done = True
            if terminated and not boundary:
                game_winners.append(
                    "A" if reward > 0 else ("B" if reward < 0 else None))
            if truncated:
                n_done = len(game_winners)
                kept = [s for s in samples if s["game_idx"] < n_done]
                dropped = len(samples) - len(kept)
                samples = kept
    return samples, game_winners, dropped


def generate_expert(decks, *, games: int = 16, roster: Optional[list] = None,
                    mirror_frac: float = DEFAULT_MIRROR_FRAC, bo3: bool = True,
                    seed: int = 1, out_dir: Optional[str] = None) -> dict:
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
    self-play. Torch-free and fast (no search), so it runs serially."""
    from search_env import SearchRoboMageEnv
    from scripted_agent import make_agent

    focus = [decks] if isinstance(decks, str) else list(decks)
    roster = list(roster) if roster else league_roster()
    out_dir = out_dir or os.path.join(_AZ_DATA_DIR, GEN_STEM)
    os.makedirs(out_dir, exist_ok=True)
    if bo3:
        _discard_pre_bo3_shards(out_dir)

    schedule = build_matchup_schedule(focus, roster, games, mirror_frac, seed)
    print(f"[az-expert] focus={','.join(focus)} matches={games} "
          f"bo3={bo3} mirror_frac={mirror_frac} (scripted:hard both seats)")
    print(f"[az-expert] matchups: {_schedule_summary(schedule)}")
    print(f"[az-expert] out_dir={out_dir}")

    da0, db0 = schedule[0] if schedule else (None, None)
    env = SearchRoboMageEnv(deck_a=da0, deck_b=db0, bo3=bo3, auto_sideboard=False)
    # ScriptedAgent is seat-aware (keys its state and deck name off the obs's
    # self-is-A flag), so ONE shared hard-tier agent pilots both seats.
    agent = make_agent("scripted:hard")
    total_samples = 0
    shards = []
    buf = []
    shard_n = 0
    stats = {"wins_a": 0, "wins_b": 0, "draws": 0, "games": 0, "dropped": 0}
    try:
        for m, (da, db) in enumerate(schedule):
            env._deck_a, env._deck_b = da, db
            agent.set_deck_names(da, db)
            samples, game_winners, dropped = _play_match_expert(
                env, agent, seed=seed + m)
            buf.append(_backfill_and_pack(samples, game_winners))
            total_samples += len(samples)
            stats["games"] += len(game_winners)
            stats["dropped"] += dropped
            mwinner = _match_winner(game_winners)
            stats["wins_a"] += int(mwinner == "A")
            stats["wins_b"] += int(mwinner == "B")
            stats["draws"] += int(mwinner == "DRAW")
            if sum(len(b[2]) for b in buf) >= FLUSH_SAMPLES:
                shards.append(_write_shard(out_dir, _concat(buf), shard_n))
                shard_n += 1
                buf = []
        if buf:
            shards.append(_write_shard(out_dir, _concat(buf), shard_n))
    finally:
        env.close()
    print(f"[az-expert] done: {total_samples} samples, {len(shards)} shard(s)  "
          f"A={stats['wins_a']} B={stats['wins_b']} draws={stats['draws']}"
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
                        seed=args.seed if args.seed is not None else 1)
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
             sb_persist=bool(getattr(args, "sb_persist", DEFAULT_SB_PERSIST)))


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AlphaZero self-play data generation")
    ap.add_argument("--deck", default="delver",
                    help="Focus deck (.dk stem) — its opponent is a mirror with "
                         "P=--mirror-frac, else a uniform league-roster draw")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=256,
                    help="PUCT sims per decision, TOTAL across --worlds")
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None,
                    help="Worker processes (default max(1, cpu-2))")
    ap.add_argument("--checkpoint", default=None,
                    help="AZ (.pt) / PPO (.zip) checkpoint or 'gen' "
                         "(default: generalist AZ ckpt, else gen PPO warm-start, "
                         "else random)")
    ap.add_argument("--temp-moves", type=int, default=DEFAULT_TEMP_MOVES)
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
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--actor", action="store_true",
                   help="Force the C++ az_actor self-play backend (error if not built)")
    g.add_argument("--no-actor", action="store_true",
                   help="Force the pure-Python backend (skip the actor even if built)")
    return ap


if __name__ == "__main__":
    run(_build_arg_parser().parse_args())
