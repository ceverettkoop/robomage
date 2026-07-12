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
    from cli_spec import BIN_DIR
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX
    from train.cli_spec import BIN_DIR

_AZ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "az_data")
_ACTOR_BIN = os.path.join(BIN_DIR, "az_actor")

# Defaults (AlphaZero-style)
DEFAULT_ROOT_NOISE_EPS = 0.25
DEFAULT_ROOT_NOISE_ALPHA = 1.0
DEFAULT_TEMP_MOVES = 20        # sample-from-visits for the first N real decisions, then argmax
FLUSH_SAMPLES = 4096           # write a shard once this many samples accumulate
HEARTBEAT_MOVES = 25           # Python backend: mid-game progress line every N decisions


def _fmt_secs(s: float) -> str:
    return f"{s / 60:.1f}m" if s >= 90 else f"{s:.0f}s"


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
               root_noise_eps, root_noise_alpha, seed, on_progress=None):
    """Play one mirror game; return (samples, winner) where samples is a list of
    dicts {obs, pi, mask, mover_is_a} and winner is 'A'/'B'/None (draw).

    ``on_progress(move, searched, fallback)``, when given, fires every
    HEARTBEAT_MOVES decisions. Observation-only: it must not (and cannot)
    perturb the game or ``rng``, so play stays byte-identical with or without
    it (the actor trains-identically gate replays this exact function)."""
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
        if on_progress is not None and move % HEARTBEAT_MOVES == 0:
            on_progress(move, searched, fallback)
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

            def beat(move, searched_ct, fallback_ct, _g=g):
                result_q.put({"kind": "beat", "worker": worker_idx,
                              "game": _g + 1, "n_games": n_games, "move": move,
                              "searched": searched_ct, "fallback": fallback_ct})

            t0 = time.time()
            samples, winner, searched, fallback = _play_game(
                env, evaluator, rng, sims=sims, worlds=worlds,
                temp_moves=temp_moves, root_noise_eps=root_noise_eps,
                root_noise_alpha=root_noise_alpha, seed=seed,
                on_progress=beat)
            obs, pi, z, mask = _backfill_and_pack(samples, winner)
            buf.append((obs, pi, z, mask))
            total_samples += len(samples)
            stats["searched"] += searched
            stats["fallback"] += fallback
            stats["wins_a"] += int(winner == "A")
            stats["wins_b"] += int(winner == "B")
            stats["draws"] += int(winner is None)
            result_q.put({"kind": "game", "worker": worker_idx, "game": g + 1,
                          "n_games": n_games, "winner": winner or "DRAW",
                          "samples": len(samples), "searched": searched,
                          "fallback": fallback, "secs": time.time() - t0})
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

def generate(deck: str, *, games: int = 10, sims: int = 128, worlds: int = 4,
             workers: Optional[int] = None, checkpoint: Optional[str] = None,
             temp_moves: int = DEFAULT_TEMP_MOVES,
             root_noise_eps: float = DEFAULT_ROOT_NOISE_EPS,
             root_noise_alpha: float = DEFAULT_ROOT_NOISE_ALPHA,
             out_dir: Optional[str] = None, seed: int = 1,
             use_actor: Optional[bool] = None) -> dict:
    """Generate ``games`` self-play games of ``deck`` (mirror) and write shards.

    ``use_actor`` picks the generation backend:
      * ``None`` (AUTO, default) — use the C++ ``bin/az_actor`` iff it is built,
        else the pure-Python multiprocess path;
      * ``True`` — force the actor (loud error if ``bin/az_actor`` is missing);
      * ``False`` — force the Python path.
    Both backends write the SAME trainer-compatible ``shard_*.npz`` files into
    ``out_dir`` and return the same summary dict shape."""
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    workers = max(1, min(workers, games))
    out_dir = out_dir or os.path.join(_AZ_DATA_DIR, deck)
    os.makedirs(out_dir, exist_ok=True)

    have_actor = os.path.exists(_ACTOR_BIN)
    if use_actor is None:
        use_actor = have_actor
        chosen = "AUTO"
    else:
        chosen = "forced"
    if use_actor and not have_actor:
        raise FileNotFoundError(
            f"--actor requested but the actor binary is not built at {_ACTOR_BIN} "
            f"(build it with `make actor`, or pass --no-actor)")

    source = resolve_source(deck, checkpoint)
    print(f"[az-selfplay] deck={deck} games={games} sims={sims} worlds={worlds} "
          f"workers={workers}")
    print(f"[az-selfplay] net source: mode={source['mode']} path={source['path']}")
    print(f"[az-selfplay] out_dir={out_dir}")
    print(f"[az-selfplay] backend={'ACTOR' if use_actor else 'PYTHON'} ({chosen}); "
          f"az_actor {'present' if have_actor else 'absent'}")

    common = dict(source=source, games=games, sims=sims, worlds=worlds,
                  workers=workers, temp_moves=temp_moves,
                  root_noise_eps=root_noise_eps, root_noise_alpha=root_noise_alpha,
                  out_dir=out_dir, seed=seed)
    if use_actor:
        return _generate_actor(deck, actor_bin=_ACTOR_BIN, **common)
    return _generate_python(deck, **common)


# ----------------------------------------------------------------------
# Python multiprocess backend
# ----------------------------------------------------------------------

def _generate_python(deck, *, source, games, sims, worlds, workers, temp_moves,
                     root_noise_eps, root_noise_alpha, out_dir, seed) -> dict:
    import multiprocessing as mp

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

    # Live progress: workers stream beat/game/shard events onto the queue and
    # finish with a 'done' record each. Consume until every worker reported.
    import queue as _queue
    t_start = time.time()
    results = []
    games_done = 0
    samples_so_far = 0
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
            print(f"[az-selfplay] w{msg['worker']} g{msg['game']}/{msg['n_games']}: "
                  f"move {msg['move']}, searched {msg['searched']}, "
                  f"fallback {msg['fallback']}", flush=True)
        elif kind == "game":
            games_done += 1
            samples_so_far += msg["samples"]
            tally[msg["winner"]] += 1
            elapsed = time.time() - t_start
            eta = elapsed / games_done * (games - games_done)
            print(f"[az-selfplay] w{msg['worker']} g{msg['game']}/{msg['n_games']}: "
                  f"winner={msg['winner']} samples={msg['samples']} "
                  f"searched={msg['searched']} fallback={msg['fallback']} "
                  f"in {_fmt_secs(msg['secs'])} | total {games_done}/{games} games, "
                  f"{samples_so_far} samples, A {tally['A']} B {tally['B']} "
                  f"D {tally['DRAW']}, elapsed {_fmt_secs(elapsed)}, "
                  f"eta {_fmt_secs(eta)}", flush=True)
        elif kind == "shard":
            print(f"[az-selfplay] w{msg['worker']} wrote {msg['path']}", flush=True)
        else:  # 'done' (also tolerates legacy kind-less records)
            results.append(msg)
    for p in procs:
        p.join()

    total_samples = sum(r["samples"] for r in results)
    all_shards = [s for r in results for s in r["shards"]]
    agg = {"searched": 0, "fallback": 0, "wins_a": 0, "wins_b": 0, "draws": 0}
    for r in results:
        for k in agg:
            agg[k] += r["stats"][k]
    print(f"[az-selfplay] done: {total_samples} samples, {len(all_shards)} shards (PYTHON)")
    print(f"[az-selfplay] decisions searched={agg['searched']} "
          f"fallback={agg['fallback']}; results A={agg['wins_a']} "
          f"B={agg['wins_b']} draws={agg['draws']}")
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


def _parse_actor_output(stdout: str):
    """Aggregate one actor worker's stdout: (total_samples, wins_a, wins_b, draws)."""
    samples = wins_a = wins_b = draws = 0
    for line in stdout.splitlines():
        if line.startswith("SELFPLAY: total_samples="):
            for tok in line.split():
                if tok.startswith("total_samples="):
                    samples = int(tok.split("=", 1)[1])
        elif line.startswith("SELFPLAY: game "):
            if line.endswith("winner=A"):
                wins_a += 1
            elif line.endswith("winner=B"):
                wins_b += 1
            elif line.endswith("winner=DRAW"):
                draws += 1
    return samples, wins_a, wins_b, draws


def actor_selfplay_cmd(actor_bin, *, deck, seed, games, sims, worlds, model,
                       out_dir, noise_eps=DEFAULT_ROOT_NOISE_EPS,
                       noise_alpha=DEFAULT_ROOT_NOISE_ALPHA,
                       temp_moves=DEFAULT_TEMP_MOVES, rng_seed=None) -> list:
    """Build a ``bin/az_actor --selfplay`` argv.

    The single source of the actor CLI contract on the Python side — used by
    _generate_actor and bench_actor so the production and benchmark invocations
    can never drift apart (and both always pin the noise/temperature knobs
    instead of leaning on the actor's compiled-in defaults)."""
    cmd = [actor_bin, "--selfplay", "--deck", deck,
           "--seed", str(seed), "--games", str(games),
           "--sims", str(sims), "--worlds", str(worlds),
           "--model", model, "--out-dir", out_dir,
           "--noise-eps", str(noise_eps),
           "--noise-alpha", str(noise_alpha),
           "--temp-moves", str(temp_moves)]
    if rng_seed is not None:
        cmd += ["--rng-seed", str(rng_seed)]
    return cmd


def _generate_actor(deck, *, source, games, sims, worlds, workers, temp_moves,
                    root_noise_eps, root_noise_alpha, out_dir, seed,
                    actor_bin) -> dict:
    import glob
    import shutil
    import subprocess
    import threading

    ts_path, tmpdir = _ensure_actor_torchscript(source)
    out_dir = os.path.abspath(out_dir)

    # Split games across workers with DISJOINT seed ranges: worker i runs games
    # seeded (seed + i*100000 + g), mirroring the Python path's per-worker
    # schedule so the two backends explore the same seed space.
    per = [games // workers] * workers
    for i in range(games % workers):
        per[i] += 1

    pre = set(glob.glob(os.path.join(out_dir, "shard_*.npz")))
    total_samples = 0
    agg = {"searched": 0, "fallback": 0, "wins_a": 0, "wins_b": 0, "draws": 0}
    procs = []

    # Live progress: a reader thread per worker echoes each actor SELFPLAY:
    # game line as it lands (with a running cross-worker total + ETA) while
    # accumulating the full stdout/stderr for the final parse. Without this,
    # nothing prints until every worker exits.
    t_start = time.time()
    prog_lock = threading.Lock()
    prog = {"done": 0}

    def _pump(wi, stream, sink, is_stdout):
        for line in stream:
            sink.append(line)
            if is_stdout and line.startswith("SELFPLAY: game "):
                with prog_lock:
                    prog["done"] += 1
                    done = prog["done"]
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (games - done)
                print(f"[az-selfplay] w{wi} {line.strip()} | total {done}/{games} "
                      f"games, elapsed {_fmt_secs(elapsed)}, eta {_fmt_secs(eta)}",
                      flush=True)
        stream.close()

    try:
        for wi in range(workers):
            if per[wi] == 0:
                continue
            base = seed + wi * 100000
            cmd = actor_selfplay_cmd(
                actor_bin, deck=deck, seed=base, games=per[wi],
                sims=sims, worlds=worlds, model=ts_path, out_dir=out_dir,
                noise_eps=root_noise_eps, noise_alpha=root_noise_alpha,
                temp_moves=temp_moves, rng_seed=seed + 100003 * (wi + 1))
            # Run from bin/ so the engine's getcwd-based RESOURCE_DIR resolves.
            p = subprocess.Popen(cmd, cwd=BIN_DIR, text=True, bufsize=1,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out_lines, err_lines = [], []
            threads = [
                threading.Thread(target=_pump, args=(wi, p.stdout, out_lines, True),
                                 daemon=True),
                threading.Thread(target=_pump, args=(wi, p.stderr, err_lines, False),
                                 daemon=True),
            ]
            for t in threads:
                t.start()
            procs.append((wi, per[wi], p, threads, out_lines, err_lines))

        failed = []
        for wi, ng, p, threads, out_lines, err_lines in procs:
            p.wait()
            for t in threads:
                t.join()
            if p.returncode != 0:
                failed.append((wi, p.returncode, "".join(err_lines)))
                continue
            s, wa, wb, dr = _parse_actor_output("".join(out_lines))
            total_samples += s
            agg["wins_a"] += wa
            agg["wins_b"] += wb
            agg["draws"] += dr
            print(f"[az-selfplay] worker {wi}: games={ng} samples={s} "
                  f"A={wa} B={wb} draws={dr}")
        if failed:
            for wi, rc, err in failed:
                tail = "\n".join(err.strip().splitlines()[-10:])
                print(f"[az-selfplay] worker {wi} FAILED (exit {rc}):\n{tail}")
            raise RuntimeError(
                f"az_actor self-play: {len(failed)} of {len(procs)} worker(s) failed")
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
    generate(args.deck, games=args.games, sims=args.sims, worlds=args.worlds,
             workers=args.workers, checkpoint=args.checkpoint,
             temp_moves=args.temp_moves, seed=args.seed if args.seed is not None else 1,
             out_dir=args.out, use_actor=_resolve_use_actor(args))


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
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--actor", action="store_true",
                   help="Force the C++ az_actor self-play backend (error if not built)")
    g.add_argument("--no-actor", action="store_true",
                   help="Force the pure-Python backend (skip the actor even if built)")
    return ap


if __name__ == "__main__":
    run(_build_arg_parser().parse_args())
