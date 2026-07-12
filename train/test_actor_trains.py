#!/usr/bin/env python3
"""Phase D / M8 gate: actor-generated shards are interchangeable with
Python-generated shards for the AZ trainer.

Proves the two self-play producers — the in-process C++ actor
(``bin/az_actor --selfplay``) and the Python reference (``train/az_selfplay.py``)
— emit datasets the Phase-C trainer ingests identically and learns from
equivalently. We deliberately do NOT assert bitwise-equal losses: the two
producers draw from independent RNG streams, so their datasets are
*statistically similar*, not identical.

  1. Export a deterministic AZNet (torch.manual_seed(0)) to a state_dict ckpt +
     TorchScript module (same net the M5/M6/M7 parity tests use). Both producers
     consume the same weights — the C++ actor loads the .ts.pt, the Python path
     loads the .pt via load_az.
  2. Dataset A: run ``bin/az_actor --selfplay`` (2 games, sims 16, worlds 2) into
     tmp dir A.
  3. Dataset B: run the Python self-play loop in-process, single-worker (reusing
     az_selfplay's own _play_game / _backfill_and_pack / _write_shard helpers,
     unmodified), same net/deck/sims/worlds/seed, into tmp dir B.
  4. Assert both datasets share the exact trainer schema (keys/dtypes/shapes via
     az_train.load_window) and that their value-range stats agree within loose
     tolerances (z in {-1,0,1}, pi rows sum to 1 within the mask, obs finite).
  5. Train two FRESH AZNets (same torch.manual_seed before each, so identical
     init) with the trainer's exact CE+MSE loss — one on each dataset — for a few
     dozen batches. Assert both trajectories are finite and DECREASING (first vs
     last), and print them side by side.

Run: train/.venv/bin/python train/test_actor_trains.py
"""

import glob
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from az_net import (AZNet, obs_space_from_const, load_az, AZEvaluator,
                    save_torchscript, torchscript_export_path)
from env import OBS_SIZE, MAX_ACTIONS
from cli_spec import BIN_DIR
import az_selfplay
import az_train

DECK = "league/ur_delver"
SEED = 1
GAMES = 2
SIMS = 16
WORLDS = 2
TRAIN_BATCHES = 120
TRAIN_BS = 64
TRAIN_LR = 1e-3
TRAIN_SEED = 0
ACTOR_BIN = os.path.join(BIN_DIR, "az_actor")

_TALLY = re.compile(r"^SELFPLAY: game (\d+) samples=(\d+) winner=(A|B|DRAW)$")


# ----------------------------------------------------------------------
# Dataset A — the C++ actor
# ----------------------------------------------------------------------

def _cpp_selfplay(ts_path, out_dir):
    cmd = [ACTOR_BIN, "--selfplay", "--deck", DECK, "--seed", str(SEED),
           "--games", str(GAMES), "--sims", str(SIMS), "--worlds", str(WORLDS),
           "--model", ts_path, "--out-dir", out_dir]
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("FAIL: az_actor exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    tallies = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        m = _TALLY.match(line.strip())
        if m:
            tallies.append((int(m.group(1)), int(m.group(2)), m.group(3)))
    return tallies


# ----------------------------------------------------------------------
# Dataset B — the Python reference, in-process single-worker
# ----------------------------------------------------------------------

def _python_selfplay(ckpt, out_dir):
    """Run GAMES self-play games in this process (no multiprocessing), reusing
    az_selfplay's play/pack/write helpers unchanged. Mirrors az_selfplay._worker
    with worker_idx=0 so the per-game seeds line up with the standard driver."""
    net = load_az(ckpt)
    evaluator = AZEvaluator(net)
    # az_selfplay._worker: rng = default_rng(base_seed + 100003*(worker_idx+1)).
    rng = np.random.default_rng(SEED + 100003 * 1)
    from search_env import SearchRoboMageEnv
    env = SearchRoboMageEnv(deck_a=DECK, deck_b=DECK)
    buf = []
    tallies = []
    try:
        for g in range(GAMES):
            seed = SEED + 0 * 100000 + g          # worker_idx 0 seed schedule
            samples, winner, _searched, _fallback = az_selfplay._play_game(
                env, evaluator, rng, sims=SIMS, worlds=WORLDS,
                temp_moves=az_selfplay.DEFAULT_TEMP_MOVES,
                root_noise_eps=az_selfplay.DEFAULT_ROOT_NOISE_EPS,
                root_noise_alpha=az_selfplay.DEFAULT_ROOT_NOISE_ALPHA, seed=seed)
            buf.append(az_selfplay._backfill_and_pack(samples, winner))
            tallies.append((g + 1, len(samples),
                            winner if winner else "DRAW"))
        if buf:
            az_selfplay._write_shard(out_dir, az_selfplay._concat(buf), 0)
    finally:
        env.close()
    return tallies


# ----------------------------------------------------------------------
# Schema + value-range checks (loose — datasets are statistically similar)
# ----------------------------------------------------------------------

def _load(deck, data_dir):
    obs, pi, z, mask, n_shards = az_train.load_window(deck, window=1000,
                                                      data_dir=data_dir)
    return obs, pi, z, mask, n_shards


def _schema(obs, pi, z, mask):
    return (obs.dtype, pi.dtype, z.dtype, mask.dtype,
            obs.shape[1], pi.shape[1], mask.shape[1])


def _value_stats_ok(tag, obs, pi, z, mask):
    ok = True
    if not np.all(np.isfinite(obs)):
        print(f"FAIL[{tag}]: obs has non-finite values", file=sys.stderr); ok = False
    if not np.all(np.isin(z, (-1.0, 0.0, 1.0))):
        print(f"FAIL[{tag}]: z outside {{-1,0,1}}", file=sys.stderr); ok = False
    in_mask = pi.copy(); in_mask[~mask] = 0.0
    beyond = pi.copy(); beyond[mask] = 0.0
    if not np.allclose(in_mask.sum(axis=1), 1.0, atol=1e-5):
        print(f"FAIL[{tag}]: pi rows don't sum to 1 within mask", file=sys.stderr); ok = False
    if np.any(beyond != 0.0):
        print(f"FAIL[{tag}]: pi nonzero beyond mask", file=sys.stderr); ok = False
    widths = mask.sum(axis=1)
    if not np.all((widths >= 2) & (widths <= MAX_ACTIONS)):
        print(f"FAIL[{tag}]: mask widths outside [2,{MAX_ACTIONS}]", file=sys.stderr); ok = False
    return ok


# ----------------------------------------------------------------------
# Training (the trainer's exact loss, small local loop for the trajectory)
# ----------------------------------------------------------------------

def _train_trajectory(obs, pi, z, mask):
    """Train a FRESH AZNet on (obs,pi,z,mask) with the same CE+MSE loss as
    az_train.train_az. torch.manual_seed(TRAIN_SEED) is set *before* net
    construction so two calls start from identical weights — the only difference
    between the two runs is the data."""
    import torch.nn.functional as F

    torch.manual_seed(TRAIN_SEED)
    rng = np.random.default_rng(TRAIN_SEED)
    net = AZNet(obs_space_from_const())
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=TRAIN_LR, weight_decay=1e-4)

    obs_t = torch.as_tensor(obs)
    pi_t = torch.as_tensor(pi)
    z_t = torch.as_tensor(z)
    mask_t = torch.as_tensor(mask)
    n = obs.shape[0]
    bs = min(TRAIN_BS, n)

    losses = []
    for _ in range(TRAIN_BATCHES):
        idx = torch.as_tensor(rng.integers(0, n, size=bs))
        ob = obs_t[idx]; tp = pi_t[idx]; tz = z_t[idx]; mk = mask_t[idx]
        logits, value = net(ob, mk)
        logp = F.log_softmax(logits, dim=-1)
        logp = torch.where(mk, logp, torch.zeros_like(logp))   # kill -inf*0 nan
        loss_pi = -(tp * logp).sum(dim=1).mean()
        loss_v = F.mse_loss(value, tz)
        loss = loss_pi + loss_v
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    return losses


def _trajectory_ok(tag, losses):
    if not all(np.isfinite(losses)):
        print(f"FAIL[{tag}]: non-finite loss in trajectory", file=sys.stderr)
        return False
    first = float(np.mean(losses[:3]))
    last = float(np.mean(losses[-3:]))
    if not (last < first):
        print(f"FAIL[{tag}]: loss did not decrease (first~{first:.4f} "
              f"last~{last:.4f})", file=sys.stderr)
        return False
    return True


def main():
    if not os.path.exists(ACTOR_BIN):
        print(f"FAIL: {ACTOR_BIN} not found — build it with `make actor`",
              file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        # 1) Deterministic AZNet -> state_dict ckpt + TorchScript (shared weights).
        torch.manual_seed(0)
        net = AZNet(obs_space_from_const()).eval()
        ckpt = os.path.join(td, "m8__azfinal.pt")
        net.save(ckpt)
        ts_path = torchscript_export_path(ckpt)
        save_torchscript(net, ts_path)

        dir_a = os.path.join(td, "cpp"); os.makedirs(dir_a)
        dir_b = os.path.join(td, "py"); os.makedirs(dir_b)

        # 2/3) Generate the two datasets.
        tallies_a = _cpp_selfplay(ts_path, dir_a)
        if tallies_a is None:
            return 1
        tallies_b = _python_selfplay(ckpt, dir_b)
        print(f"[data] C++    per-game tallies: {tallies_a}")
        print(f"[data] Python per-game tallies: {tallies_b}")

        if not glob.glob(os.path.join(dir_a, "shard_*.npz")):
            print("FAIL: C++ actor wrote no shards", file=sys.stderr); return 1
        if not glob.glob(os.path.join(dir_b, "shard_*.npz")):
            print("FAIL: Python selfplay wrote no shards", file=sys.stderr); return 1

        # 4) Schema identical + value ranges within loose tolerances.
        obs_a, pi_a, z_a, mask_a, ns_a = _load(DECK, dir_a)
        obs_b, pi_b, z_b, mask_b, ns_b = _load(DECK, dir_b)
        sch_a, sch_b = _schema(obs_a, pi_a, z_a, mask_a), _schema(obs_b, pi_b, z_b, mask_b)
        if sch_a != sch_b:
            print(f"FAIL: dataset schemas differ:\n  C++   ={sch_a}\n  Python={sch_b}",
                  file=sys.stderr)
            return 1
        if sch_a[4] != OBS_SIZE or sch_a[5] != MAX_ACTIONS:
            print(f"FAIL: schema widths obs={sch_a[4]} pi={sch_a[5]} != "
                  f"{OBS_SIZE}/{MAX_ACTIONS}", file=sys.stderr)
            return 1
        print(f"[schema] identical: dtypes obs/pi/z/mask={sch_a[0]}/{sch_a[1]}/"
              f"{sch_a[2]}/{sch_a[3]} widths obs/pi/mask={sch_a[4]}/{sch_a[5]}/"
              f"{sch_a[6]} (C++ n={obs_a.shape[0]} from {ns_a} shard(s); "
              f"Python n={obs_b.shape[0]} from {ns_b} shard(s))")
        if not (_value_stats_ok("C++", obs_a, pi_a, z_a, mask_a)
                and _value_stats_ok("Python", obs_b, pi_b, z_b, mask_b)):
            return 1

        def zdist(z):
            return {int(v): int((z == v).sum()) for v in (-1.0, 0.0, 1.0)}
        print(f"[stats] z dist  C++={zdist(z_a)}  Python={zdist(z_b)}")
        print(f"[stats] obs range C++=[{obs_a.min():.3f},{obs_a.max():.3f}] "
              f"Python=[{obs_b.min():.3f},{obs_b.max():.3f}]")

        # 5) Train a fresh net on each; assert finite + decreasing trajectories.
        print(f"[train] {TRAIN_BATCHES} batches, bs={TRAIN_BS}, lr={TRAIN_LR}, "
              f"seed={TRAIN_SEED} (identical init per run)")
        traj_a = _train_trajectory(obs_a, pi_a, z_a, mask_a)
        traj_b = _train_trajectory(obs_b, pi_b, z_b, mask_b)

    if not (_trajectory_ok("C++", traj_a) and _trajectory_ok("Python", traj_b)):
        return 1

    def fmt(t):
        pts = [t[0]] + [t[len(t) // 4], t[len(t) // 2], t[3 * len(t) // 4]] + [t[-1]]
        return " -> ".join(f"{v:.4f}" for v in pts)
    print("\n  loss trajectory (start .. end):")
    print(f"    C++-shards    : {fmt(traj_a)}")
    print(f"    Python-shards : {fmt(traj_b)}")
    print(f"\nPASS: actor & Python shards are trainer-interchangeable — identical "
          f"schema, both trajectories finite & decreasing "
          f"[deck={DECK} seed={SEED} sims={SIMS} worlds={WORLDS}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
