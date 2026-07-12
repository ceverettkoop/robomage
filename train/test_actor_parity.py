#!/usr/bin/env python3
"""Phase D / M5 gate: obs bit-parity between bin/az_actor and the Python pipeline.

The in-process C++ actor (bin/az_actor) reconstructs the 6700-float RL observation
engine-side (src/actor/obs_builder.cpp) instead of round-tripping the stdio BQUERY
protocol. This test proves that reconstruction is BIT-EXACT against the Python
env's obs for the identical game:

  1. Build a deterministic AZNet (torch.manual_seed(0)), save its state_dict and
     export the TorchScript module both the C++ actor and this test consume.
  2. Run the C++ actor for one game (--deck league/ur_delver --seed 1) with
     --dump-obs, capturing every decision's observation to a binary file.
  3. Drive the SAME game through the Python env (RoboMageEnv + runner.drive_game),
     both seats piloted by an "azraw" controller loading the same TorchScript
     module and picking argmax over the masked logits — the same greedy rule the
     C++ actor uses, so the trajectories stay in lockstep.
  4. Assert identical decision count, identical num_choices per decision, and
     every observation row bit-exact (np.array_equal).

Run: train/.venv/bin/python train/test_actor_parity.py
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from az_net import AZNet, obs_space_from_const, save_torchscript, torchscript_export_path
from env import RoboMageEnv, OBS_SIZE, MAX_ACTIONS
from cli_spec import REPO_ROOT, BIN_DIR
import runner

DECK = "league/ur_delver"
SEED = 1
ACTOR_BIN = os.path.join(BIN_DIR, "az_actor")


class AZRawController:
    """Greedy TorchScript controller: argmax over the masked logits, recording
    each decision's observation so the Python obs stream can be compared to the
    C++ actor's dump. One instance drives both seats (a single global policy)."""

    def __init__(self, ts_path):
        self.module = torch.jit.load(ts_path)
        self.module.eval()
        self.records = []  # list[(num_choices, obs_copy)]

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None):
        self.records.append((int(num_choices), np.array(obs, dtype=np.float32, copy=True)))
        mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool)
        mask[0, :num_choices] = True
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _value = self.module(obs_t, mask)
        return int(torch.argmax(logits[0, :num_choices]))


def _read_dump(path):
    """Read the actor's --dump-obs binary: repeated (int32 num_choices,
    float32[OBS_SIZE]). Returns list[(num_choices, np.ndarray)]."""
    out = []
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    rec = 4 + OBS_SIZE * 4
    while off + rec <= len(data):
        (nc,) = struct.unpack_from("<i", data, off)
        vals = np.frombuffer(data, dtype="<f4", count=OBS_SIZE, offset=off + 4)
        out.append((nc, np.array(vals, dtype=np.float32, copy=True)))
        off += rec
    if off != len(data):
        raise RuntimeError(f"dump file {path} has {len(data) - off} trailing bytes "
                           "— frame size mismatch")
    return out


def main():
    if not os.path.exists(ACTOR_BIN):
        print(f"FAIL: {ACTOR_BIN} not found — build it with `make actor`", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        # 1) Deterministic AZNet -> state_dict ckpt + TorchScript export.
        torch.manual_seed(0)
        net = AZNet(obs_space_from_const()).eval()
        ckpt = os.path.join(td, "parity__azfinal.pt")
        net.save(ckpt)
        ts_path = torchscript_export_path(ckpt)     # parity__azfinal.ts.pt
        save_torchscript(net, ts_path)

        # 2) Run the C++ actor for one game, dumping every decision's obs.
        dump_path = os.path.join(td, "actor_obs.bin")
        cmd = [ACTOR_BIN, "--deck", DECK, "--seed", str(SEED),
               "--model", ts_path, "--dump-obs", dump_path, "--games", "1"]
        proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print("FAIL: az_actor exited nonzero:\n"
                  + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
            return 1
        actor_obs = _read_dump(dump_path)

        # 3) Drive the SAME game through the Python env, both seats azraw.
        ctrl = AZRawController(ts_path)
        env = RoboMageEnv(deck_a=DECK, deck_b=DECK, bo3=False)
        try:
            obs, _ = env.reset(options={"engine_seed": SEED})
            runner.drive_game(env, obs, ctrl, ctrl)
        finally:
            env.close()
        py_obs = ctrl.records

    # 4) Compare.
    if len(actor_obs) != len(py_obs):
        print(f"FAIL: decision count differs — actor={len(actor_obs)} "
              f"python={len(py_obs)}", file=sys.stderr)
        n = min(len(actor_obs), len(py_obs))
        for i in range(n):
            if actor_obs[i][0] != py_obs[i][0] or not np.array_equal(
                    actor_obs[i][1], py_obs[i][1]):
                print(f"  first divergence at decision {i} "
                      f"(nc actor={actor_obs[i][0]} python={py_obs[i][0]})",
                      file=sys.stderr)
                break
        return 1

    for i, ((a_nc, a_obs), (p_nc, p_obs)) in enumerate(zip(actor_obs, py_obs)):
        if a_nc != p_nc:
            print(f"FAIL: num_choices differ at decision {i}: "
                  f"actor={a_nc} python={p_nc}", file=sys.stderr)
            return 1
        if not np.array_equal(a_obs, p_obs):
            diff = np.flatnonzero(a_obs != p_obs)
            print(f"FAIL: obs differ at decision {i} "
                  f"({diff.size} floats differ)", file=sys.stderr)
            for j in diff[:10]:
                print(f"  obs[{j}]: actor={a_obs[j]!r} python={p_obs[j]!r}",
                      file=sys.stderr)
            return 1

    print(f"PASS: obs bit-exact over {len(actor_obs)} decisions "
          f"(deck={DECK} seed={SEED}, OBS_SIZE={OBS_SIZE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
