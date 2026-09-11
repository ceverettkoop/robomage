#!/usr/bin/env python3
"""Baseline sweep: the AZ generalist under full search vs scripted:hard.

The engine behind ``train.py baseline`` (:func:`run`). By default ``az:gen``
(``gen__azfinal.pt``) at the league search budget (``DEFAULT_AZ_SIMS`` sims x
``DEFAULT_AZ_WORLDS`` worlds, ``DEFAULT_AZ_C_PUCT``) pilots every league deck
against scripted:hard piloting every league deck — the full N x N grid, mirrors
included — for ``DEFAULT_BASELINE_GAMES`` bo3 matches per matchup, seats
alternating within each matchup so neither side gets a systematic on-the-play
edge. The report (per matchup, per piloted deck, per opponent deck, per value
bucket, per game index) is appended to ``checkpoints/baseline_report.log`` and
printed. ``--deck`` narrows the grid to one piloted deck (a mirror unless
``--opponent`` names the scripted deck).

Two backends, chosen by the model spec and the --actor/--no-actor pair:

* ACTOR — the default whenever ``bin/az_actor`` is built and the model is an
  AZ net (an ``az:`` spec, ``gen``'s AZ checkpoint, or a ``.pt`` path). Each
  matchup becomes two ``bin/az_actor --search`` legs — the actor's EVAL mode
  (no root Dirichlet noise, argmax(visits), exactly what the promotion gate
  plays) — with scripted:hard on the other seat through train/scripted_oracle.py
  (one oracle process serves the whole fleet) and leaf evaluation on the central
  GPU eval server when one starts (AUTO, like self-play). Up to ``--workers``
  legs run at once in a sliding pool; per-leg seeds derive only from the matchup
  index, so results never depend on worker count or completion order. The
  net seat's searched decisions are RECORDED as trainer-schema shards (unless
  ``--no-record``) into a fresh ``az_data/baseline/baseline_<stamp>/`` — one
  flat directory for az_inspect / shard_replay / the shard browsers, and
  deliberately outside the ``az_data/gen`` training pool.
* PYTHON — PPO ``.zip`` models, ``mcts:`` specs, or ``--no-actor``: the
  runner-based path in train.py (``baseline`` / ``baseline_all``), one Python
  driver plus one engine per worker. Several times slower at the full budget.

Run from the repo root:
    train/.venv/bin/python train/train.py baseline
    train/.venv/bin/python train/train.py baseline --deck league/ur_delver --games 20
    train/.venv/bin/python train/train.py baseline --sims 256 --worlds 4 --workers 16
"""
from __future__ import annotations

import datetime
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

import archetypes
from cli_spec import (BIN_DIR, DEFAULT_BASELINE_MODEL, DEFAULT_AZ_TD_N,
                      append_spec_knob)

from az_selfplay import _ACTOR_BIN   # the build-tier actor (bin/<config>/az_actor)

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_PATH = os.path.join(_HERE, "checkpoints", "baseline_report.log")
# Recorded baseline shards live here, NOT under az_data/gen: the trainer's
# window globs only its own data dir, so a baseline can never leak into training.
RECORD_ROOT = os.path.join(_HERE, "az_data", "baseline")


# ----------------------------------------------------------------------
# Matchup grid + model classification
# ----------------------------------------------------------------------

def resolve_matchups(deck: Optional[str], opponent: Optional[str],
                     all_flag: bool, roster: list,
                     mirrors: bool = False) -> list:
    """The ``(piloted deck, scripted deck)`` cells to play.

    No ``deck`` (or ``--all``) is the full roster grid, every ordered pair
    including mirrors. ``mirrors`` is the grid's diagonal only — every roster
    deck vs itself. A ``deck`` alone is its mirror; ``deck`` + ``opponent``
    one cross cell; ``opponent`` alone is every roster deck vs that one
    scripted deck."""
    if mirrors:
        return [(d, d) for d in roster]
    if all_flag or (not deck and not opponent):
        return [(d, o) for d in roster for o in roster]
    if deck:
        return [(deck, opponent or deck)]
    return [(d, opponent) for d in roster]


def classify_model(spec: str) -> tuple:
    """Split a baseline model spec into ``(kind, az_ckpt, base, params)``.

    ``kind`` is ``"az"`` when the spec names an AZ net the actor can load —
    ``az:<base>[?knobs]`` (``az:gen`` resolving to the incumbent
    ``gen__azfinal.pt``) or a bare ``.pt`` path — else ``"python"`` (a bare
    ``gen`` stays the PPO generalist, as everywhere else in make_controller;
    ``.zip`` paths, ``mcts:``/``azraw:`` specs). ``az_ckpt`` is the resolved
    ``.pt`` for ``"az"`` kinds; ``params`` are the spec's ``?k=v`` knobs
    (sims/worlds/c/sb_* override the command-line budget)."""
    from opponents import _parse_spec_query
    from az_net import resolve_az_checkpoint
    prefix, _, rest = spec.partition(":")
    if not ((prefix == "az" and rest) or spec.endswith(".pt")):
        return "python", None, spec, {}
    base, params = _parse_spec_query(rest if prefix == "az" else spec)
    ckpt = resolve_az_checkpoint(base)
    if ckpt is None:
        raise ValueError(
            f"baseline: {spec!r} names no AZ checkpoint (no gen__azfinal.pt / "
            f"gen__azv*.pt under train/checkpoints/az, and not an existing .pt "
            f"path)")
    return "az", ckpt, base, params


def search_budget(params: dict, spec: str, *, sims: int, worlds: int,
                  c_puct: float, sb_branches: int, sb_worlds: int,
                  sb_rollout_turns: int) -> dict:
    """The effective search budget: the command-line values, overridden by any
    ``?sims=&worlds=&c=&sb_*=`` knob carried in the spec itself (so a spec
    copied from observe/analysis means the same thing here)."""
    from opponents import _spec_knob
    return dict(
        sims=_spec_knob(params, "sims", sims, int, spec),
        worlds=_spec_knob(params, "worlds", worlds, int, spec),
        c_puct=_spec_knob(params, "c", c_puct, float, spec),
        sb_branches=_spec_knob(params, "sb_branches", sb_branches, int, spec),
        sb_worlds=_spec_knob(params, "sb_worlds", sb_worlds, int, spec),
        sb_rollout_turns=_spec_knob(params, "sb_rollout_turns",
                                    sb_rollout_turns, int, spec))


def python_spec_with_budget(spec: str, kind: str, budget: dict) -> str:
    """The controller spec the PYTHON backend should build for ``spec``: an AZ
    kind becomes an explicit ``az:`` search spec carrying the whole budget as
    knobs (appended last, so they win); other specs pass through unchanged."""
    if kind != "az":
        return spec
    base = spec if spec.startswith("az:") else f"az:{spec}"
    for key, val in (("sims", budget["sims"]), ("worlds", budget["worlds"]),
                     ("c", budget["c_puct"]),
                     ("sb_branches", budget["sb_branches"]),
                     ("sb_worlds", budget["sb_worlds"]),
                     ("sb_rollout_turns", budget["sb_rollout_turns"])):
        base = append_spec_knob(base, key, val)
    return base


# ----------------------------------------------------------------------
# Actor backend
# ----------------------------------------------------------------------

def actor_leg_cmd(actor_bin: str, *, deck_a: str, deck_b: str, games: int,
                  seed: int, scripted_seat: str, oracle_sock: str,
                  ts_path: str, server_sock: Optional[str], budget: dict,
                  bo3: bool, device: str, record_dir: Optional[str] = None,
                  td_n: int = DEFAULT_AZ_TD_N,
                  provenance_json: Optional[str] = None) -> list:
    """One ``bin/az_actor --search`` leg: the net pilots the non-scripted seat
    in eval mode (no root noise, argmax visits — the gate's mode), scripted:hard
    answers ``scripted_seat``'s real decisions over the oracle socket.
    ``record_dir`` adds ``--record --out-dir``: the net seat's searched
    decisions are written there as trainer-schema shards (the scripted seat
    plays through the oracle and records nothing); played actions unchanged.
    ``provenance_json`` (a serialized search-provenance object) additionally
    asks for one shard per match with the ``.diag`` / ``.rmplay`` sidecars
    the shard browsers need to rebuild each searched decision's tree."""
    cmd = [actor_bin, "--search", "--deck", deck_a, "--deck-b", deck_b,
           "--seed", str(seed), "--games", str(games),
           "--sims", str(budget["sims"]), "--worlds", str(budget["worlds"]),
           "--c", str(budget["c_puct"]), "--merge-dupes", "1", "--cross-world",
           "--scripted-seat", scripted_seat, "--scripted-oracle", oracle_sock]
    cmd += (["--eval-server", server_sock] if server_sock else ["--model", ts_path])
    if device != "cpu" and not server_sock:
        cmd += ["--device", device]
    if bo3:
        cmd += ["--bo3",
                "--sb-branches", str(budget["sb_branches"]),
                "--sb-worlds", str(budget["sb_worlds"]),
                "--sb-rollout-turns", str(budget["sb_rollout_turns"])]
    if record_dir:
        cmd += ["--record", "--out-dir", record_dir, "--td-n", str(td_n)]
        if provenance_json is not None:
            cmd += ["--replay-sidecars", "--provenance-json", provenance_json]
    return cmd


def search_provenance(spec: str, ckpt: str, budget: dict, device: str) -> dict:
    """The ``search_provenance`` object an actor recording's ``.rmplay``
    carries: the same keys ``opponents.SearchController.search_provenance``
    writes for a GUI recording (so ``tree_rebuild.evaluator_spec_for`` and the
    rebuild knobs read both alike), with the actor's fixed eval-mode facts."""
    from opponents import _file_sha256
    out = {
        "spec": spec, "checkpoint": ckpt, "backend": "az_actor",
        "sims": int(budget["sims"]), "worlds": int(budget["worlds"]),
        "c_puct": float(budget["c_puct"]), "temperature": 0.0,
        "merge_dupes": True, "cross_world": True, "procs": 1,
        "time_budget": None, "sims_cap": None,
        "clock": None, "tmin": None, "tmax": None,
        "sb_branches": int(budget["sb_branches"]),
        "sb_worlds": int(budget["sb_worlds"]),
        "sb_rollout_turns": int(budget["sb_rollout_turns"]),
        "device": device, "torch_threads": None,
    }
    if os.path.isfile(ckpt):
        out["checkpoint_sha256"] = _file_sha256(ckpt)
        out["checkpoint_size"] = int(os.path.getsize(ckpt))
    return out


def leg_record_dir(record_dir: str, deck: str, opp: str, net_is_a: bool) -> str:
    """Per-leg shard directory under a run's ``record_dir``: the leaf name
    encodes the net's seat and the pairing (``net_A__<deck>__<opp>``; ``/`` in
    a deck stem becomes ``-``), so a row's matchup and seat stay recoverable
    from its filename after :func:`flatten_leg_shards`."""
    tag = (f"net_{'A' if net_is_a else 'B'}__{deck.replace('/', '-')}__"
           f"{opp.replace('/', '-')}")
    return os.path.join(record_dir, tag)


def flatten_leg_shards(record_dir: str) -> int:
    """Move a finished run's shards from their per-leg subdirectories up into
    ``record_dir`` itself, renamed ``shard_<legtag>_<original>`` — one flat
    directory of ``shard_*.npz``, which is what az_inspect, shard_replay and
    the shard browsers glob. A shard's same-stem sidecars (``.diag`` /
    ``.rmplay``) move with it under the same renamed stem, so they stay
    paired. Empty leg dirs are removed. Returns the shard count."""
    moved = 0
    if not os.path.isdir(record_dir):
        return 0
    for leg in sorted(os.listdir(record_dir)):
        leg_dir = os.path.join(record_dir, leg)
        if not os.path.isdir(leg_dir):
            continue
        for f in sorted(os.listdir(leg_dir)):
            if not f.startswith("shard_"):
                continue
            stem, ext = os.path.splitext(f)
            if ext == ".npz":
                moved += 1
            shutil.move(os.path.join(leg_dir, f),
                        os.path.join(record_dir,
                                     f"shard_{leg}_{stem[len('shard_'):]}{ext}"))
        if not os.listdir(leg_dir):
            os.rmdir(leg_dir)
    return moved


def parse_leg_output(stdout: str, bo3: bool, net_is_a: bool) -> tuple:
    """Tally one leg's stdout from the NET's view: ``(wins, losses, draws,
    per_game)`` — per-MATCH ``MATCH_RESULT:`` lines in bo3 (a bo3 match has no
    draw), per-GAME ``GAME_RESULT:`` lines in bo1. ``per_game`` maps the
    0-based game index within a match (bo3 only; index 0 = the pre-board game)
    to ``[net wins, games played]``."""
    net = "Player A wins" if net_is_a else "Player B wins"
    opp = "Player B wins" if net_is_a else "Player A wins"
    w = l = d = 0
    per_game: dict = {}
    for line in stdout.splitlines():
        if bo3:
            if line.startswith("MATCH_RESULT: "):
                if net in line:
                    w += 1
                elif opp in line:
                    l += 1
            elif line.startswith("GAME_RESULT: "):
                toks = line.split()
                if len(toks) >= 2 and toks[1].isdigit():
                    cell = per_game.setdefault(int(toks[1]) - 1, [0, 0])
                    cell[1] += 1
                    cell[0] += int(net in line)
        elif line.startswith("GAME_RESULT:"):
            if net in line:
                w += 1
            elif opp in line:
                l += 1
            elif "draw" in line:
                d += 1
    return w, l, d, per_game


def _fmt_secs(s: float) -> str:
    if s >= 5400:
        return f"{s / 3600:.1f}h"
    return f"{s / 60:.1f}m" if s >= 90 else f"{s:.0f}s"


def run_actor_sweep(matchups: list, *, ckpt: str, n_games: int, seed: int,
                    budget: dict, bo3: bool, workers: int,
                    actor_device: str = "cpu", eval_server=None,
                    record_dir: Optional[str] = None,
                    td_n: int = DEFAULT_AZ_TD_N,
                    actor_bin: str = _ACTOR_BIN, tag: str = "baseline",
                    provenance: Optional[dict] = None) -> tuple:
    """Play every ``(deck, opp)`` matchup for ``n_games`` on the C++ actor.

    Returns ``(results, per_game, n_shards)``: ``results[(deck, opp)] =
    (w, l, d)`` from the net's view, ``per_game`` the pooled bo3
    per-game-index tally (``{gi: [wins, played]}``), and ``n_shards`` the
    shard files recorded under ``record_dir`` (0 when not recording).

    ``record_dir`` records every leg's net-seat searched decisions as
    trainer-schema shards (one subdir per leg, see :func:`leg_record_dir`,
    flattened into ``record_dir`` when the sweep finishes so the inspectors'
    flat ``shard_*.npz`` glob sees them). A leg that is terminated early has
    its partial recording deleted, like a gate leg. ``provenance`` (the
    :func:`search_provenance` dict) makes each leg record one shard per match
    with the ``.diag`` / ``.rmplay`` sidecars (tree-rebuildable in the shard
    browsers); None records plain threshold-flushed shards.

    Each matchup is two legs so seats alternate exactly like the Python path:
    net-in-seat-A for ``n_games - n_games//2`` matches (seed
    ``seed + mi*100003``), then net-in-seat-B for ``n_games//2`` (that seed
    ``+ n_games``). Fleet setup happens once: the TorchScript export, the
    scripted oracle, and the eval server (``eval_server`` tri-state: None AUTO
    — cuda server iff it starts, else local forwards on ``actor_device`` with a
    notice; True forced; False off). A failed leg aborts the sweep with its
    repro command and stderr tail, the same way self-play reports one."""
    from az_selfplay import (_ensure_actor_torchscript, _spawn_oracle,
                             actor_gpu_env, start_eval_server,
                             stop_eval_server)

    if not os.path.exists(actor_bin):
        raise RuntimeError(f"{tag}: actor binary not built at {actor_bin} "
                           f"(run `make actor`)")
    half = n_games // 2
    legs = []   # (mi, net_is_a, deck_a, deck_b, games, seed)
    for mi, (dx, dy) in enumerate(matchups):
        mseed = seed + mi * 100003
        if n_games - half:
            legs.append((mi, True, dx, dy, n_games - half, mseed))
        if half:
            legs.append((mi, False, dy, dx, half, mseed + n_games))
    workers = max(1, min(workers, len(legs)))
    unit = "matches" if bo3 else "games"
    total_units = n_games * len(matchups)
    print(f"[{tag}] {len(matchups)} matchup(s) x {n_games} {unit} = "
          f"{total_units} {unit} in {len(legs)} actor leg(s), {workers} in "
          f"flight; sims={budget['sims']} worlds={budget['worlds']} "
          f"c={budget['c_puct']}"
          + (f" sb={budget['sb_branches']}/{budget['sb_worlds']}/"
             f"{budget['sb_rollout_turns']}" if bo3 else ""), flush=True)

    ts_path, _ = _ensure_actor_torchscript({"mode": "az", "path": ckpt}, tag=tag)
    oracle_proc = oracle_sock = oracle_dir = None
    server_proc = server_sock = server_dir = None
    results: dict = {}
    per_game_all: dict = {}
    tallies = {mi: [0, 0, 0] for mi in range(len(matchups))}
    legs_left = {mi: 0 for mi in range(len(matchups))}
    for leg in legs:
        legs_left[leg[0]] += 1
    prog = {"done": 0, "matchups": 0}
    prog_lock = threading.Lock()
    t_start = time.time()
    failed = []
    active = []

    def _is_progress_line(line):
        return line.startswith("MATCH_RESULT:" if bo3 else "GAME_RESULT:")

    def _pump(stream, sink, is_stdout):
        for line in stream:
            sink.append(line)
            if is_stdout and _is_progress_line(line):
                with prog_lock:
                    prog["done"] += 1
                    done = prog["done"]
                    elapsed = time.time() - t_start
                    eta = elapsed / done * (total_units - done)
                if done % 10 == 0 or done == total_units:
                    print(f"[{tag}] {done}/{total_units} {unit}, elapsed "
                          f"{_fmt_secs(elapsed)}, eta {_fmt_secs(eta)}",
                          flush=True)
        stream.close()

    def _launch(leg):
        mi, net_is_a, da, db, games, lseed = leg
        dx, dy = matchups[mi]
        rd = leg_record_dir(record_dir, dx, dy, net_is_a) if record_dir else None
        cmd = actor_leg_cmd(actor_bin, deck_a=da, deck_b=db, games=games,
                            seed=lseed, scripted_seat=("B" if net_is_a else "A"),
                            oracle_sock=oracle_sock, ts_path=ts_path,
                            server_sock=server_sock, budget=budget, bo3=bo3,
                            device=actor_device, record_dir=rd, td_n=td_n,
                            provenance_json=prov_json)
        # Run from bin/ so the engine's getcwd-based RESOURCE_DIR resolves.
        p = subprocess.Popen(cmd, cwd=BIN_DIR, text=True, bufsize=1,
                             env=actor_env, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = [], []
        threads = [threading.Thread(target=_pump, args=(p.stdout, out, True),
                                    daemon=True),
                   threading.Thread(target=_pump, args=(p.stderr, err, False),
                                    daemon=True)]
        for t in threads:
            t.start()
        return {"leg": leg, "p": p, "cmd": cmd, "threads": threads,
                "out": out, "err": err, "record_dir": rd}

    def _reap(rec):
        rec["p"].wait()
        for t in rec["threads"]:
            t.join()
        mi, net_is_a = rec["leg"][0], rec["leg"][1]
        if rec["p"].returncode != 0:
            failed.append((mi, rec["p"].returncode, rec["cmd"], "".join(rec["err"])))
            return
        w, l, d, pg = parse_leg_output("".join(rec["out"]), bo3, net_is_a)
        t = tallies[mi]
        t[0] += w; t[1] += l; t[2] += d
        for gi, (gw, gn) in pg.items():
            cell = per_game_all.setdefault(gi, [0, 0])
            cell[0] += gw; cell[1] += gn
        legs_left[mi] -= 1
        if legs_left[mi] == 0:
            dx, dy = matchups[mi]
            results[(dx, dy)] = tuple(t)
            prog["matchups"] += 1
            print(f"[{tag}]   [{prog['matchups']}/{len(matchups)}] piloting "
                  f"{dx} vs scripted:hard {dy}: {_wld_line(*t)}", flush=True)

    n_shards = 0
    try:
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
            print(f"[{tag}] recording shards under {record_dir} (one subdir per "
                  f"leg while running, flattened at the end)", flush=True)
        oracle_proc, oracle_sock, oracle_dir = _spawn_oracle()
        if eval_server is not False:
            server_proc, server_sock, server_dir = start_eval_server(
                ts_path, device=actor_device if actor_device != "cpu" else "cuda",
                forced=bool(eval_server), tag=tag)
            if server_proc is None:
                print(f"[{tag}] eval-server AUTO: no usable GPU (server failed "
                      f"to start) — actors run local forwards on {actor_device}",
                      flush=True)
        print(f"[{tag}] actor eval: "
              f"{'central server on gpu' if server_sock else 'local ' + actor_device}"
              f", cross-world=on, scripted oracle at {oracle_sock}", flush=True)
        actor_env = (actor_gpu_env()
                     if (actor_device != "cpu" and server_sock is None) else None)
        prov_json = None
        if provenance is not None and record_dir:
            prov = dict(provenance)
            # The device the net forwards actually ran on this sweep.
            prov["device"] = ("cuda" if server_sock else actor_device)
            prov_json = json.dumps(prov)

        # Sliding pool (same shape as az_selfplay._generate_actor): keep up to
        # `workers` legs in flight, launching the next as any one exits.
        next_leg = 0
        while next_leg < len(legs) or active:
            while next_leg < len(legs) and len(active) < workers:
                active.append(_launch(legs[next_leg]))
                next_leg += 1
            done = [rec for rec in active if rec["p"].poll() is not None]
            if not done:
                time.sleep(0.2)
                continue
            for rec in done:
                active.remove(rec)
                _reap(rec)
        if failed:
            for mi, rc, cmd, err in failed:
                tail = "\n".join(err.strip().splitlines()[-40:])
                repro = " ".join(shlex.quote(str(c)) for c in cmd)
                dx, dy = matchups[mi]
                print(f"[{tag}] leg {dx} vs {dy} FAILED (exit {rc})\n"
                      f"  reproduce (run from {BIN_DIR}; needs a live oracle + "
                      f"server at the socket paths shown):\n    {repro}\n"
                      f"  stderr tail:\n{tail}", flush=True)
            raise RuntimeError(
                f"{tag}: {len(failed)} of {len(legs)} actor leg(s) failed; "
                f"see the FAILED block(s) above")
        if record_dir:
            n_shards = flatten_leg_shards(record_dir)
            print(f"[{tag}] {n_shards} shard file(s) recorded under {record_dir}",
                  flush=True)
    finally:
        for rec in active:
            # Abnormal exit: stop the live legs and drop their half-played
            # recordings (a partial match must not look like a finished one).
            if rec["p"].poll() is None:
                rec["p"].terminate()
                try:
                    rec["p"].wait(timeout=30)
                except subprocess.TimeoutExpired:
                    rec["p"].kill()
                    rec["p"].wait()
            if rec["record_dir"]:
                shutil.rmtree(rec["record_dir"], ignore_errors=True)
        if oracle_proc is not None:
            oracle_proc.terminate()
            oracle_proc.wait()
        if oracle_dir:
            shutil.rmtree(oracle_dir, ignore_errors=True)
        stop_eval_server(server_proc, server_dir)
    print(f"[{tag}] done in {_fmt_secs(time.time() - t_start)}", flush=True)
    return results, per_game_all, n_shards


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------

def _wld_line(w: int, l: int, d: int) -> str:
    total = w + l + d
    pct = 100 * w / total if total else 0.0
    return f"{w}W/{l}L/{d}D  {pct:.1f}% win rate"


def format_per_game_index(per_game: dict, subject: str = "model") -> list:
    """The bo3 per-game-index line from an actor-side ``{gi: [wins, played]}``
    tally (game 1 is pre-board; 2-3 post-board). Unlike
    ``runner.format_per_game_split`` there is no play/draw control — the actor
    reports results, not starting players — so read g1 vs g2-3 with the
    play/draw confound in mind. ``[]`` when no match reached game 2."""
    if not per_game or max(per_game) == 0:
        return []
    cells = "  ".join(
        f"g{gi + 1} {w}/{n} ({100 * w / n:.1f}%)" if n else f"g{gi + 1} 0/0 (n/a)"
        for gi, (w, n) in sorted(per_game.items()))
    return [f"per-game ({subject}): {cells}"
            "   [g1 vs g2-3 alone is confounded by the play/draw rule]"]


def build_report(display: str, matchups: list, results: dict, *,
                 n_games: int, bo3: bool, seed, split_lines=()) -> str:
    """The baseline report text: a header, one line per matchup (win rates
    from the model's view), the per-game lines, and — for a multi-cell grid —
    per-piloted-deck rows, per-opponent-deck rows, and the value-bucket /
    self-archetype pooling the multi-head critic is split along."""
    piloted, opps = [], []
    for d, o in matchups:
        if d not in piloted:
            piloted.append(d)
        if o not in opps:
            opps.append(o)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    unit = "matches" if bo3 else "games"
    grid = (f"{len(piloted)}x{len(opps)} matchups" if len(matchups) > 1
            else f"{matchups[0][0]} vs {matchups[0][1]}")
    lines = [f"=== {display} baseline vs scripted:hard ({grid}) — {stamp} — "
             f"{n_games} {unit}/matchup, seed={seed} ==="]
    per_model = {d: [0, 0, 0] for d in piloted}
    per_opp = {o: [0, 0, 0] for o in opps}
    per_bucket: dict = {}
    for deck in piloted:
        row = []
        for opp in opps:
            if (deck, opp) not in results:
                continue
            w, l, d = results[(deck, opp)]
            total = w + l + d
            pct = 100 * w / total if total else 0
            row.append(f"{opp}={w}W/{l}L/{d}D({pct:.0f}%)")
            lines.append(f"{display} piloting {deck:<22} vs scripted:hard "
                         f"{opp:<22} " + _wld_line(w, l, d))
            bucket = archetypes.bucket_index(deck, opp)
            for tally in (per_model[deck], per_opp[opp],
                          per_bucket.setdefault(bucket, [0, 0, 0])):
                tally[0] += w; tally[1] += l; tally[2] += d
        if len(matchups) > 1:
            lines.append(f"  [{display} {deck}] " + "  ".join(row))
    lines.append("")
    if split_lines:
        lines.extend(split_lines)
        lines.append("")
    if len(matchups) > 1:
        lines.append(f"per model deck ({display} piloting it vs the whole "
                     f"scripted field):")
        for deck, (w, l, d) in per_model.items():
            lines.append(f"  {deck:<22} " + _wld_line(w, l, d))
        lines.append("per opponent deck (scripted:hard piloting it vs every "
                     "model deck; win rate is still the model's):")
        for deck, (w, l, d) in per_opp.items():
            lines.append(f"  {deck:<22} " + _wld_line(w, l, d))
        lines.append("per value bucket (self archetype vs opponent archetype — "
                     "the multi-head critic's split):")
        per_self_arch: dict = {}
        for bucket in sorted(per_bucket):
            w, l, d = per_bucket[bucket]
            lines.append(f"  {archetypes.bucket_name(bucket):<34} "
                         + _wld_line(w, l, d))
            tally = per_self_arch.setdefault(
                archetypes.bucket_archetypes(bucket)[0], [0, 0, 0])
            tally[0] += w; tally[1] += l; tally[2] += d
        lines.append("per self archetype (pooled over every opponent archetype):")
        for arch in sorted(per_self_arch):
            w, l, d = per_self_arch[arch]
            lines.append(f"  {archetypes.arch_name_at(arch):<34} "
                         + _wld_line(w, l, d))
    tw = sum(v[0] for v in results.values())
    tl = sum(v[1] for v in results.values())
    td = sum(v[2] for v in results.values())
    lines.append(f"overall ({display} vs scripted:hard): " + _wld_line(tw, tl, td))
    return "\n".join(lines)


def append_report(text: str, log_path: str) -> None:
    """Append one report to the baseline log (created on first use) and echo
    it with the path, so every baseline run — grid or single cell, actor or
    Python — leaves the same record."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(text + "\n\n")
    print(f"\n{text}\n\nreport appended to {log_path}", flush=True)


# ----------------------------------------------------------------------
# train.py entry
# ----------------------------------------------------------------------

def run(args, *, python_sweep: Callable, resolve_model: Callable) -> None:
    """``train.py baseline`` dispatch. ``python_sweep(binary, spec, matchups,
    n_games, seed, bo3, workers) -> (results, per_game)`` is the runner-based
    matchup sweep (train.baseline_sweep) the Python backend uses; its
    ``per_game`` is the ``runner.tally_per_game`` shape."""
    from az_selfplay import league_roster, resolve_seed, _resolve_use_actor
    from az_selfplay import resolve_eval_server
    from opponents import make_controller

    roster = league_roster()
    if not roster:
        print("No league decks found under bin/resources/decks/league")
        return
    matchups = resolve_matchups(args.deck, args.opponent, args.all, roster,
                                mirrors=getattr(args, "mirrors", False))
    n_games = args.games
    bo3 = not args.bo1
    log_path = args.log or DEFAULT_LOG_PATH
    spec = args.model or DEFAULT_BASELINE_MODEL
    try:
        kind, ckpt, base, params = classify_model(spec)
    except ValueError as exc:
        raise SystemExit(str(exc))
    budget = search_budget(params, spec, sims=args.sims, worlds=args.worlds,
                           c_puct=args.c_puct, sb_branches=args.sb_branches,
                           sb_worlds=args.sb_worlds,
                           sb_rollout_turns=args.sb_rollout_turns)
    use_actor = _resolve_use_actor(args)
    actor_built = os.path.exists(_ACTOR_BIN)
    if use_actor is True and kind != "az":
        raise SystemExit(f"baseline: --actor needs an AZ net (az:gen, gen's AZ "
                         f"checkpoint, or a .pt path); {spec!r} is not one")
    if use_actor is True and not actor_built:
        raise SystemExit(f"baseline: --actor requested but {_ACTOR_BIN} is not "
                         f"built (run `make actor`)")
    backend = "actor" if (kind == "az" and use_actor is not False and actor_built) else "python"
    if kind == "az" and backend == "python" and use_actor is None:
        print(f"[baseline] actor AUTO: {_ACTOR_BIN} not built — Python backend "
              f"(several times slower at this budget)", flush=True)
    seed = resolve_seed(args, label="baseline")

    extra: list = []
    if backend == "actor":
        display = (f"{spec} ({os.path.basename(ckpt)}) @ sims={budget['sims']} "
                   f"worlds={budget['worlds']} c={budget['c_puct']} [actor]")
        record_dir = None
        if not args.no_record:
            # Absolute: the actor legs run from bin/ (engine RESOURCE_DIR), so
            # a relative --record-dir would land under bin/ instead.
            record_dir = os.path.abspath(args.record_dir or os.path.join(
                RECORD_ROOT,
                "baseline_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        results, per_game, n_shards = run_actor_sweep(
            matchups, ckpt=ckpt, n_games=n_games, seed=seed, budget=budget,
            bo3=bo3, workers=args.workers, actor_device=args.actor_device,
            eval_server=resolve_eval_server(args), record_dir=record_dir,
            td_n=args.td_n,
            provenance=(search_provenance(spec, ckpt, budget, args.actor_device)
                        if record_dir else None))
        split = format_per_game_index(per_game) if bo3 else []
        if record_dir:
            extra.append(f"shards: {n_shards} file(s) under {record_dir}")
    else:
        if not args.no_record:
            print("[baseline] shard recording is actor-only; the Python backend "
                  "records nothing", flush=True)
        py_spec = python_spec_with_budget(spec, kind, budget)
        try:
            # Build once here so a bad spec dies as a usage error, not mid-run.
            make_controller(py_spec, checkpoint_resolver=resolve_model,
                            deterministic=True)
        except ValueError as exc:
            raise SystemExit(f"baseline: {exc}")
        display = py_spec + " [python]"
        results, pooled = python_sweep(args.binary, py_spec, matchups, n_games,
                                       seed, bo3, args.workers)
        import runner
        split = runner.format_per_game_split(pooled, subject="model") if bo3 else []
    append_report(build_report(display, matchups, results, n_games=n_games,
                               bo3=bo3, seed=seed, split_lines=split + extra),
                  log_path)
