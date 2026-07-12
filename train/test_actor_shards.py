#!/usr/bin/env python3
"""Phase D / M7 gate: bin/az_actor --selfplay writes .npz shards the Phase C
trainer ingests unchanged.

  1. Export a deterministic AZNet (torch.manual_seed(0)) to TorchScript — the same
     net the M5/M6 parity tests use — and run `bin/az_actor --selfplay` for 2
     short games (small sims/worlds) into a temp out-dir.
  2. Parse the printed per-game `SELFPLAY: game N samples=K winner=...` tallies.
  3. Load the produced shard(s) with numpy and assert the exact schema the trainer
     expects: keys obs/pi/z/mask, dtypes f4/f4/f4/b1, shapes (n,6700)/(n,64)/(n,)/
     (n,64); every pi row sums to ~1 within its mask and 0 beyond; z in {-1,0,1};
     mask row widths in [2,64]; total n == sum of the per-game tallies.
  4. Assert az_train.load_window ingests the temp dir and returns concatenated
     arrays of the same length.

Run: train/.venv/bin/python train/test_actor_shards.py
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

from az_net import AZNet, obs_space_from_const, save_torchscript, torchscript_export_path
from env import OBS_SIZE, MAX_ACTIONS
from cli_spec import BIN_DIR
import az_train

DECK = "league/ur_delver"
SEED = 1
GAMES = 2
SIMS = 16
WORLDS = 2
ACTOR_BIN = os.path.join(BIN_DIR, "az_actor")

_TALLY = re.compile(r"^SELFPLAY: game (\d+) samples=(\d+) winner=(A|B|DRAW)$")


def _run_selfplay(ts_path, out_dir):
    cmd = [ACTOR_BIN, "--selfplay", "--deck", DECK, "--seed", str(SEED),
           "--games", str(GAMES), "--sims", str(SIMS), "--worlds", str(WORLDS),
           "--model", ts_path, "--out-dir", out_dir]
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("FAIL: az_actor exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    return proc.stdout.decode("utf-8", "replace")


def main():
    if not os.path.exists(ACTOR_BIN):
        print(f"FAIL: {ACTOR_BIN} not found — build it with `make actor`",
              file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        # 1) Deterministic AZNet -> TorchScript export.
        torch.manual_seed(0)
        net = AZNet(obs_space_from_const()).eval()
        ckpt = os.path.join(td, "parity__azfinal.pt")
        net.save(ckpt)
        ts_path = torchscript_export_path(ckpt)
        save_torchscript(net, ts_path)

        out_dir = os.path.join(td, "shards")
        stdout = _run_selfplay(ts_path, out_dir)
        if stdout is None:
            return 1

        # 2) Parse per-game tallies.
        tallies = []
        for line in stdout.splitlines():
            m = _TALLY.match(line.strip())
            if m:
                tallies.append((int(m.group(1)), int(m.group(2)), m.group(3)))
        if len(tallies) != GAMES:
            print(f"FAIL: expected {GAMES} SELFPLAY tallies, got {len(tallies)}\n"
                  f"stdout:\n{stdout}", file=sys.stderr)
            return 1
        tally_total = sum(k for _, k, _ in tallies)
        print(f"[shards] per-game tallies: {tallies} (sum={tally_total})")

        # 3) Load shards and assert schema exactly.
        shard_paths = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")))
        if not shard_paths:
            print(f"FAIL: no shard_*.npz written to {out_dir}", file=sys.stderr)
            return 1

        n_total = 0
        for sp in shard_paths:
            d = np.load(sp)
            keys = set(d.files)
            if keys != {"obs", "pi", "z", "mask"}:
                print(f"FAIL: shard {sp} keys {keys} != obs/pi/z/mask",
                      file=sys.stderr)
                return 1
            obs, pi, z, mask = d["obs"], d["pi"], d["z"], d["mask"]
            n = obs.shape[0]
            n_total += n
            checks = [
                (obs.dtype == np.float32, f"obs dtype {obs.dtype} != float32"),
                (pi.dtype == np.float32, f"pi dtype {pi.dtype} != float32"),
                (z.dtype == np.float32, f"z dtype {z.dtype} != float32"),
                (mask.dtype == np.bool_, f"mask dtype {mask.dtype} != bool"),
                (obs.shape == (n, OBS_SIZE), f"obs shape {obs.shape}"),
                (pi.shape == (n, MAX_ACTIONS), f"pi shape {pi.shape}"),
                (z.shape == (n,), f"z shape {z.shape}"),
                (mask.shape == (n, MAX_ACTIONS), f"mask shape {mask.shape}"),
            ]
            for ok, msg in checks:
                if not ok:
                    print(f"FAIL: shard {sp}: {msg}", file=sys.stderr)
                    return 1

            # z in {-1, 0, 1}.
            if not np.all(np.isin(z, (-1.0, 0.0, 1.0))):
                bad = z[~np.isin(z, (-1.0, 0.0, 1.0))]
                print(f"FAIL: shard {sp}: z has values outside {{-1,0,1}}: "
                      f"{bad[:8]}", file=sys.stderr)
                return 1

            # Per-row: mask width in [2,64]; pi sums to ~1 within mask and 0
            # beyond; pi is nonzero only where mask is True.
            widths = mask.sum(axis=1)
            if not np.all((widths >= 2) & (widths <= MAX_ACTIONS)):
                print(f"FAIL: shard {sp}: mask widths outside [2,64]: "
                      f"{widths[(widths < 2) | (widths > MAX_ACTIONS)][:8]}",
                      file=sys.stderr)
                return 1
            in_mask = pi.copy(); in_mask[~mask] = 0.0
            beyond = pi.copy(); beyond[mask] = 0.0
            if not np.allclose(in_mask.sum(axis=1), 1.0, atol=1e-5):
                bad = np.flatnonzero(~np.isclose(in_mask.sum(axis=1), 1.0, atol=1e-5))
                print(f"FAIL: shard {sp}: {bad.size} pi rows don't sum to 1 "
                      f"within mask (e.g. row {bad[0]} sum="
                      f"{in_mask[bad[0]].sum()})", file=sys.stderr)
                return 1
            if np.any(beyond != 0.0):
                print(f"FAIL: shard {sp}: pi nonzero beyond the mask",
                      file=sys.stderr)
                return 1

        if n_total != tally_total:
            print(f"FAIL: shard rows {n_total} != summed tallies {tally_total}",
                  file=sys.stderr)
            return 1

        # 4) The trainer's loader ingests the temp dir.
        lobs, lpi, lz, lmask, n_shards = az_train.load_window(
            DECK, window=1000, data_dir=out_dir)
        if lobs.shape[0] != n_total or lpi.shape[0] != n_total or \
                lz.shape[0] != n_total or lmask.shape[0] != n_total:
            print(f"FAIL: load_window row count mismatch: obs={lobs.shape[0]} "
                  f"pi={lpi.shape[0]} z={lz.shape[0]} mask={lmask.shape[0]} "
                  f"expected {n_total}", file=sys.stderr)
            return 1
        if lobs.shape[1] != OBS_SIZE or lpi.shape[1] != MAX_ACTIONS:
            print(f"FAIL: load_window widths obs={lobs.shape[1]} pi={lpi.shape[1]}",
                  file=sys.stderr)
            return 1

    print(f"PASS: {len(shard_paths)} shard(s), {n_total} samples "
          f"(== summed tallies), schema exact; load_window ingested "
          f"{n_shards} shard(s) [deck={DECK} seed={SEED} sims={SIMS} "
          f"worlds={WORLDS}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
