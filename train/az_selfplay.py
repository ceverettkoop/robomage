"""AlphaZero self-play data generation for ONE deck (mirror match, bo1).

Each worker owns a :class:`search_env.SearchRoboMageEnv` (deck vs itself) and one
shared :class:`az_net.AZNet` piloting BOTH seats. At every decision that is
loop-safe (``env.last_search_safe``) with >1 choice it runs a determinized PUCT
search (:func:`mcts.run_search`) with root Dirichlet noise; unsafe / trivial
decisions fall back to the net's raw-policy argmax. For each SEARCHED decision it
stores (obs, visit-distribution pi, legal mask, mover seat); at game end each
sample's outcome z (+1/-1 from that mover's perspective, 0 on a draw) is filled
in. Samples are written to ``az_data/{deck}/shard_{ts}_{pid}_{n}.npz``.

Run standalone (``az_selfplay.py --deck delver --games 2 --sims 12 --worlds 2``)
or via ``train.py az-selfplay``.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np

try:
    from env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX

_AZ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "az_data")

# Defaults (AlphaZero-style)
DEFAULT_ROOT_NOISE_EPS = 0.25
DEFAULT_ROOT_NOISE_ALPHA = 1.0
DEFAULT_TEMP_MOVES = 20        # sample-from-visits for the first N real decisions, then argmax
FLUSH_SAMPLES = 4096           # write a shard once this many samples accumulate


# ----------------------------------------------------------------------
# Checkpoint resolution (parent decides once; workers load their own copy)
# ----------------------------------------------------------------------

def resolve_source(deck: str, checkpoint: Optional[str]) -> dict:
    """Pick the net source: explicit AZ/PPO path, else the deck's AZ checkpoint,
    else a PPO warm-start of the deck's PPO checkpoint, else random init.

    Returns a spec dict {mode: 'az'|'ppo'|'random', path: str|None} — picklable so
    each worker reconstructs its own net (torch objects don't cross processes)."""
    from az_net import resolve_az_checkpoint
    from opponents import resolve_checkpoint

    if checkpoint:
        if checkpoint.endswith(".pt") or resolve_az_checkpoint(checkpoint):
            az = resolve_az_checkpoint(checkpoint) or checkpoint
            return {"mode": "az", "path": az}
        return {"mode": "ppo", "path": resolve_checkpoint(checkpoint)}
    az = resolve_az_checkpoint(deck)
    if az:
        return {"mode": "az", "path": az}
    ppo = resolve_checkpoint(deck)
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

def _play_game(env, evaluator, rng, *, sims, worlds, temp_moves,
               root_noise_eps, root_noise_alpha, seed):
    """Play one mirror game; return (samples, winner) where samples is a list of
    dicts {obs, pi, mask, mover_is_a} and winner is 'A'/'B'/None (draw)."""
    from mcts import run_search

    obs, _ = env.reset(seed=seed)
    samples = []
    winner = None
    move = 0
    done = False
    searched = 0
    fallback = 0

    while not done:
        num_choices = env._num_choices
        priority_is_a = bool(env._obs[_SELF_IS_A_IDX] > 0.5)
        searchable = bool(env.last_search_safe) and num_choices > 1

        if searchable:
            result = run_search(env, evaluator, sims=sims, worlds=worlds,
                                root_noise_eps=root_noise_eps,
                                root_noise_alpha=root_noise_alpha, rng=rng)
            visits = result.policy_target(1.0)          # normalized visit counts
            pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
            pi[:num_choices] = visits.astype(np.float32)
            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[:num_choices] = True
            samples.append({"obs": env._obs.copy(), "pi": pi, "mask": mask,
                            "mover_is_a": priority_is_a})
            searched += 1
            if move < temp_moves:
                action = int(rng.choice(num_choices, p=visits))
            else:
                action = result.best_action()
        else:
            priors, _ = evaluator.evaluate(env._obs, num_choices)
            action = int(np.argmax(priors))
            fallback += 1

        obs, reward, terminated, truncated, _ = env.step(action)
        move += 1
        if terminated or truncated:
            done = True
            if terminated:
                winner = "A" if reward > 0 else ("B" if reward < 0 else None)
    return samples, winner, searched, fallback


def _backfill_and_pack(samples, winner):
    """Fill z per sample from its mover's perspective vs the winner, then pack to
    arrays. Draw (winner None) -> z=0."""
    n = len(samples)
    obs = np.zeros((n, OBS_SIZE), dtype=np.float32)
    pi = np.zeros((n, MAX_ACTIONS), dtype=np.float32)
    z = np.zeros((n,), dtype=np.float32)
    mask = np.zeros((n, MAX_ACTIONS), dtype=bool)
    for i, s in enumerate(samples):
        obs[i] = s["obs"]
        pi[i] = s["pi"]
        mask[i] = s["mask"]
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

def _worker(deck, source, n_games, sims, worlds, temp_moves, root_noise_eps,
            root_noise_alpha, out_dir, base_seed, worker_idx, result_q):
    import torch
    torch.set_num_threads(1)   # avoid oversubscription across worker processes
    from search_env import SearchRoboMageEnv
    from az_net import AZEvaluator

    net = _build_net(source)
    evaluator = AZEvaluator(net)
    rng = np.random.default_rng(base_seed + 100003 * (worker_idx + 1))

    env = SearchRoboMageEnv(deck_a=deck, deck_b=deck)
    total_samples = 0
    shards = []
    buf = []
    shard_n = 0
    stats = {"searched": 0, "fallback": 0, "wins_a": 0, "wins_b": 0, "draws": 0}
    try:
        for g in range(n_games):
            seed = base_seed + worker_idx * 100000 + g
            samples, winner, searched, fallback = _play_game(
                env, evaluator, rng, sims=sims, worlds=worlds,
                temp_moves=temp_moves, root_noise_eps=root_noise_eps,
                root_noise_alpha=root_noise_alpha, seed=seed)
            obs, pi, z, mask = _backfill_and_pack(samples, winner)
            buf.append((obs, pi, z, mask))
            total_samples += len(samples)
            stats["searched"] += searched
            stats["fallback"] += fallback
            stats["wins_a"] += int(winner == "A")
            stats["wins_b"] += int(winner == "B")
            stats["draws"] += int(winner is None)
            if sum(len(b[2]) for b in buf) >= FLUSH_SAMPLES:
                shards.append(_write_shard(out_dir, _concat(buf), shard_n))
                shard_n += 1
                buf = []
        if buf:
            shards.append(_write_shard(out_dir, _concat(buf), shard_n))
    finally:
        env.close()
    result_q.put({"worker": worker_idx, "samples": total_samples,
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

def generate(deck: str, *, games: int = 10, sims: int = 128, worlds: int = 4,
             workers: Optional[int] = None, checkpoint: Optional[str] = None,
             temp_moves: int = DEFAULT_TEMP_MOVES,
             root_noise_eps: float = DEFAULT_ROOT_NOISE_EPS,
             root_noise_alpha: float = DEFAULT_ROOT_NOISE_ALPHA,
             out_dir: Optional[str] = None, seed: int = 1) -> dict:
    """Generate ``games`` self-play games of ``deck`` (mirror) across worker
    processes and write shards. Returns a summary dict."""
    import multiprocessing as mp

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    workers = max(1, min(workers, games))
    out_dir = out_dir or os.path.join(_AZ_DATA_DIR, deck)
    os.makedirs(out_dir, exist_ok=True)

    source = resolve_source(deck, checkpoint)
    print(f"[az-selfplay] deck={deck} games={games} sims={sims} worlds={worlds} "
          f"workers={workers}")
    print(f"[az-selfplay] net source: mode={source['mode']} path={source['path']}")
    print(f"[az-selfplay] out_dir={out_dir}")

    # Split games across workers.
    per = [games // workers] * workers
    for i in range(games % workers):
        per[i] += 1

    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for wi in range(workers):
        if per[wi] == 0:
            continue
        p = ctx.Process(target=_worker,
                        args=(deck, source, per[wi], sims, worlds, temp_moves,
                              root_noise_eps, root_noise_alpha, out_dir, seed,
                              wi, result_q))
        p.start()
        procs.append(p)

    results = [result_q.get() for _ in procs]
    for p in procs:
        p.join()

    total_samples = sum(r["samples"] for r in results)
    all_shards = [s for r in results for s in r["shards"]]
    agg = {"searched": 0, "fallback": 0, "wins_a": 0, "wins_b": 0, "draws": 0}
    for r in results:
        for k in agg:
            agg[k] += r["stats"][k]
    print(f"[az-selfplay] done: {total_samples} samples, {len(all_shards)} shards")
    print(f"[az-selfplay] decisions searched={agg['searched']} "
          f"fallback={agg['fallback']}; results A={agg['wins_a']} "
          f"B={agg['wins_b']} draws={agg['draws']}")
    return {"samples": total_samples, "shards": all_shards, "stats": agg,
            "out_dir": out_dir, "source": source}


def run(args) -> None:
    """train.py dispatch entry."""
    generate(args.deck, games=args.games, sims=args.sims, worlds=args.worlds,
             workers=args.workers, checkpoint=args.checkpoint,
             temp_moves=args.temp_moves, seed=args.seed if args.seed is not None else 1,
             out_dir=args.out)


def _build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="AlphaZero self-play data generation")
    ap.add_argument("--deck", default="delver", help="Deck (.dk stem) — mirror match")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=128)
    ap.add_argument("--worlds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=None,
                    help="Worker processes (default max(1, cpu-2))")
    ap.add_argument("--checkpoint", default=None,
                    help="AZ (.pt) or PPO (.zip) checkpoint / deck shorthand "
                         "(default: deck's AZ ckpt, else PPO warm-start, else random)")
    ap.add_argument("--temp-moves", type=int, default=DEFAULT_TEMP_MOVES)
    ap.add_argument("--out", default=None, help="Output dir (default az_data/{deck})")
    ap.add_argument("--seed", type=int, default=1)
    return ap


if __name__ == "__main__":
    run(_build_arg_parser().parse_args())
