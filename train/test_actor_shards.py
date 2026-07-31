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
from env import OBS_SIZE, MAX_ACTIONS, _SELF_IS_A_IDX, _MATCH_CTX_START
from cli_spec import BIN_DIR
import az_train

DECK = "league/ur_delver"
SEED = 1
GAMES = 2
SIMS = 16
WORLDS = 2
# bo3 mini-run: a small in-game + sideboard budget just to exercise the actor's
# bo3 selfplay path (searched sideboard roots + next-game sample flush).
BO3_SIMS = 8
BO3_WORLDS = 2
BO3_SB_SIMS = 8
BO3_SB_WORLDS = 2
BO3_SB_MAX_DEPTH = 200
BO3_SB_ROLLOUT_TURNS = 2  # small pin: exercises the actor's leaf-rollout path cheaply
# is_sideboard_phase flag in the obs (env's _MATCH_CTX layout: game_number,
# self_wins, opp_wins, sideboard_phase).
_IS_SIDEBOARD_IDX = _MATCH_CTX_START + 3
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


def _run_selfplay_bo3(ts_path, out_dir):
    """Run the actor in bo3 selfplay with a small in-game + sideboard budget."""
    cmd = [ACTOR_BIN, "--selfplay", "--bo3", "--deck", DECK, "--seed", str(SEED),
           "--games", str(GAMES), "--sims", str(BO3_SIMS), "--worlds",
           str(BO3_WORLDS), "--sb-sims", str(BO3_SB_SIMS), "--sb-worlds",
           str(BO3_SB_WORLDS), "--sb-max-depth", str(BO3_SB_MAX_DEPTH),
           "--sb-rollout-turns", str(BO3_SB_ROLLOUT_TURNS),
           "--model", ts_path, "--out-dir", out_dir]
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("FAIL: az_actor --bo3 exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    return proc.stdout.decode("utf-8", "replace")


def _shard_sort_key(path):
    """Order pooled shards by write order: (mtime, numeric suffix n) from the
    shard_{ts}_{pid}_{n}.npz name (a lexicographic sort misorders n>=10)."""
    base = os.path.basename(path)
    try:
        n = int(base.rsplit("_", 1)[1].split(".")[0])
    except (IndexError, ValueError):
        n = 0
    return (os.path.getmtime(path), n)


def _load_pooled(out_dir):
    """Load all shards in WRITE order and concatenate. The actor adds samples in
    game-backfill order, so the concatenation partitions cleanly by the per-game
    SELFPLAY tallies. Returns (obs, pi, z, mask) or None if no shards."""
    shard_paths = sorted(glob.glob(os.path.join(out_dir, "shard_*.npz")),
                         key=_shard_sort_key)
    if not shard_paths:
        return None
    obs, pi, z, mask = [], [], [], []
    for sp in shard_paths:
        d = np.load(sp)
        obs.append(d["obs"]); pi.append(d["pi"]); z.append(d["z"]); mask.append(d["mask"])
    return (np.concatenate(obs), np.concatenate(pi),
            np.concatenate(z), np.concatenate(mask))


def _verify_bo3(stdout, out_dir):
    """Verify the actor's bo3 selfplay next-game sample flush.

    The actor buffers a bo3 game's samples and prices+flushes them at the game
    boundary, and (Stage 5) between-game SIDEBOARD-root samples stay buffered and
    flush with the NEXT game — so a sideboard sample's z is the upcoming game's
    result from that mover's perspective. Samples are added to the accumulator in
    game-backfill order, so concatenating the pooled shards in write order and
    segmenting by the per-game `SELFPLAY: game N samples=K winner=W` tallies
    recovers each game's rows. Within a game segment the sideboard rows (obs
    is_sideboard flag set) sit at the front; we assert each such row's z equals
    that game's winner priced from the row's own mover seat.

    Returns 0 on success (with a printed REPORT of what was checked), 1 on failure.
    """
    tallies = []  # (samples_k, winner_k) in game order
    for line in stdout.splitlines():
        m = _TALLY.match(line.strip())
        if m:
            tallies.append((int(m.group(2)), m.group(3)))
    if not tallies:
        print(f"FAIL [bo3]: no SELFPLAY game tallies\nstdout:\n{stdout}",
              file=sys.stderr)
        return 1

    pooled = _load_pooled(out_dir)
    if pooled is None:
        print(f"FAIL [bo3]: no shard_*.npz written to {out_dir}", file=sys.stderr)
        return 1
    obs, _pi, z, _mask = pooled
    n_total = obs.shape[0]
    tally_total = sum(k for k, _ in tallies)
    if n_total != tally_total:
        print(f"FAIL [bo3]: pooled rows {n_total} != summed game tallies "
              f"{tally_total}", file=sys.stderr)
        return 1

    # z sanity across every row (draws are unacceptable in this engine, but the
    # schema still permits 0; per-game verification below is the real check).
    if not np.all(np.isin(z, (-1.0, 0.0, 1.0))):
        print(f"FAIL [bo3]: z outside {{-1,0,1}}", file=sys.stderr)
        return 1

    sb_total = 0
    off = 0
    for gi, (k, winner) in enumerate(tallies):
        seg_obs = obs[off:off + k]
        seg_z = z[off:off + k]
        off += k
        if winner == "DRAW":
            # A draw shouldn't happen; if it did, its samples carry z=0 — skip the
            # winner-perspective check for this segment (it has no signed winner).
            continue
        winner_is_a = (winner == "A")
        sb_rows = np.flatnonzero(seg_obs[:, _IS_SIDEBOARD_IDX] > 0.5)
        for ri in sb_rows:
            mover_is_a = seg_obs[ri, _SELF_IS_A_IDX] > 0.5
            expect = 1.0 if (mover_is_a == winner_is_a) else -1.0
            if float(seg_z[ri]) != expect:
                print(f"FAIL [bo3]: game {gi} sideboard sample row {ri}: z="
                      f"{float(seg_z[ri])} but next-game winner={winner} with "
                      f"mover_is_a={mover_is_a} expects {expect}", file=sys.stderr)
                return 1
        sb_total += len(sb_rows)

    if sb_total == 0:
        print("FAIL [bo3]: no sideboard-phase samples found (obs is_sideboard "
              "flag never set) — the bo3 sideboard roots weren't searched/stored",
              file=sys.stderr)
        return 1

    print(f"PASS [bo3]: {sb_total} sideboard sample(s) across {len(tallies)} game "
          f"segment(s); each is flushed in its NEXT game's segment and its z equals "
          f"that game's winner from the mover's seat (z in {{-1,+1}}) "
          f"[deck={DECK} sims={BO3_SIMS} sb_sims={BO3_SB_SIMS}]")
    return 0


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

        # 5) bo3 mini-run: exercise the actor's bo3 selfplay path (searched
        # sideboard roots + next-game sample flush, Stage 5). Fresh out-dir so its
        # shards don't mix with the bo1 pool above.
        bo3_dir = os.path.join(td, "shards_bo3")
        os.makedirs(bo3_dir, exist_ok=True)
        bo3_stdout = _run_selfplay_bo3(ts_path, bo3_dir)
        if bo3_stdout is None:
            return 1
        rc = _verify_bo3(bo3_stdout, bo3_dir)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
