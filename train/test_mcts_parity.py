#!/usr/bin/env python3
"""Phase D / M6 gate: whole-game visit-count parity between the in-process C++
MCTS (bin/az_actor --search) and the Python reference (train/mcts.py::run_search)
at batch=1.

Both drive the SAME deterministic game (deck league/ur_delver, seed 1, mirror
match, Player A on the play) and run a determinized PUCT search at every
loop-safe root with >1 choice. The two implementations share:

  * the same TorchScript AZNet (torch.jit.load of one exported .ts.pt),
    single-threaded, so the evaluator arithmetic is bit-identical;
  * the same per-world determinize seeds — injected on both sides by the shared
    formula ``world_seed(root r, world w) = (base + 100003*r + w) & 0xffffffff``
    (C++ --world-seeds BASE; Python passes world_seeds=... into run_search);
  * temperature-0 play (argmax over summed root visit counts) with Dirichlet
    root noise disabled, so the two trajectories stay in lockstep.

The C++ side writes every searched root's (root_index, num_choices, int64
visits) to --dump-visits; the Python side records the same triples. We assert
identical root count, identical num_choices sequence, and every root's visit
vector bit-exact — which also proves the two games stepped in lockstep.

Then a K=16 batched-leaf sanity check reruns the C++ side with --batch 16 (same
seeds) and REPORTS (does not assert) the argmax-agreement fraction against the
batch=1 visits — expected high but not 100% (virtual loss perturbs collection).

Run: train/.venv/bin/python train/test_mcts_parity.py
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
from env import MAX_ACTIONS
from cli_spec import BIN_DIR
import runner
from search_env import SearchRoboMageEnv

DECK = "league/ur_delver"
SEED = 1
SIMS = 16
WORLDS = 2
C_PUCT = 1.5
SEED_BASE = 42
ACTOR_BIN = os.path.join(BIN_DIR, "az_actor")


def world_seeds_for(root_index: int) -> list:
    """The shared per-world seed formula (mirrors az_mcts.cpp::begin_world)."""
    return [(SEED_BASE + 100003 * root_index + w) & 0xFFFFFFFF
            for w in range(WORLDS)]


class TSEvaluator:
    """mcts.Evaluator backed by the exported TorchScript module — the exact same
    computation as az_net.py::AZEvaluator.evaluate (float32 softmax over the
    first num_choices logits, widened to float64 and renormalized; tanh value),
    but loaded via torch.jit.load so the C++ actor and this driver run one and
    the same scripted graph."""

    def __init__(self, ts_path):
        torch.set_num_threads(1)
        self.module = torch.jit.load(ts_path)
        self.module.eval()

    def evaluate(self, obs, num_choices):
        mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool)
        mask[0, :num_choices] = True
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.module(obs_t, mask)
            probs = torch.softmax(logits[0, :num_choices], dim=-1)
            priors = probs.detach().cpu().numpy().astype(np.float64)
            v = float(value.item())
        total = priors.sum()
        if not np.isfinite(total) or total <= 0.0:
            priors = np.full(num_choices, 1.0 / num_choices)
            total = 1.0
        return priors / total, v


class ParitySearchController:
    """Drives one seat exactly like the C++ actor: at a loop-safe root with >1
    choice, run_search (temp 0 → argmax visits) with the shared per-world seeds;
    otherwise fall back to the evaluator's raw-policy argmax. Records each
    searched root's (index, num_choices, int64 visits)."""

    wants_search_env = True

    def __init__(self, evaluator):
        from mcts import run_search  # noqa: F401 — fail fast if unavailable
        self.ev = evaluator
        self.env = None
        self.root_counter = 0
        self.records = []  # list[(root_index, num_choices, np.int64[num_choices])]

    def bind_env(self, env):
        self.env = env

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None):
        from mcts import run_search
        env = self.env
        searchable = (env is not None and getattr(env, "last_search_safe", None)
                      and num_choices > 1)
        if not searchable:
            priors, _ = self.ev.evaluate(obs, num_choices)
            return int(np.argmax(priors))
        r = self.root_counter
        self.root_counter += 1
        result = run_search(env, self.ev, sims=SIMS, worlds=WORLDS,
                            c_puct=C_PUCT, world_seeds=world_seeds_for(r))
        self.records.append((r, int(num_choices),
                             result.visits.astype(np.int64)))
        return result.best_action()


def _read_visits_dump(path):
    """Read --dump-visits: repeated (int32 root_index, int32 num_choices,
    int64[num_choices]). Returns list[(root_index, num_choices, np.int64[...])]."""
    with open(path, "rb") as f:
        data = f.read()
    out = []
    off = 0
    while off < len(data):
        ri, nc = struct.unpack_from("<ii", data, off)
        off += 8
        vals = np.frombuffer(data, dtype="<i8", count=nc, offset=off)
        off += 8 * nc
        out.append((ri, nc, np.array(vals, dtype=np.int64, copy=True)))
    if off != len(data):
        raise RuntimeError(f"visits dump {path}: {len(data) - off} trailing bytes")
    return out


def _run_actor(ts_path, dump_path, batch):
    cmd = [ACTOR_BIN, "--search", "--sims", str(SIMS), "--worlds", str(WORLDS),
           "--c", str(C_PUCT), "--batch", str(batch), "--world-seeds",
           str(SEED_BASE), "--deck", DECK, "--seed", str(SEED),
           "--model", ts_path, "--dump-visits", dump_path, "--games", "1"]
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("FAIL: az_actor exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    return _read_visits_dump(dump_path)


def main():
    if not os.path.exists(ACTOR_BIN):
        print(f"FAIL: {ACTOR_BIN} not found — build it with `make actor`",
              file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as td:
        # 1) Deterministic AZNet -> TorchScript export (shared by both sides).
        torch.manual_seed(0)
        net = AZNet(obs_space_from_const()).eval()
        ckpt = os.path.join(td, "parity__azfinal.pt")
        net.save(ckpt)
        ts_path = torchscript_export_path(ckpt)
        save_torchscript(net, ts_path)

        # 2) C++ actor, batch=1: search every root, dump visits.
        dump1 = os.path.join(td, "visits_b1.bin")
        actor1 = _run_actor(ts_path, dump1, batch=1)
        if actor1 is None:
            return 1

        # 3) Python reference: drive the SAME game, search each root.
        ev = TSEvaluator(ts_path)
        ctrl = ParitySearchController(ev)
        env = SearchRoboMageEnv(deck_a=DECK, deck_b=DECK, bo3=False)
        ctrl.bind_env(env)
        try:
            obs, _ = env.reset(options={"engine_seed": SEED})
            runner.drive_game(env, obs, ctrl, ctrl)
        finally:
            env.close()
        py = ctrl.records

        # 4) Compare batch=1 visit vectors exactly.
        if len(actor1) != len(py):
            print(f"FAIL: searched-root count differs — actor={len(actor1)} "
                  f"python={len(py)}", file=sys.stderr)
            n = min(len(actor1), len(py))
            for i in range(n):
                if (actor1[i][1] != py[i][1]
                        or not np.array_equal(actor1[i][2], py[i][2])):
                    print(f"  first divergence at searched root {i} "
                          f"(nc actor={actor1[i][1]} python={py[i][1]})",
                          file=sys.stderr)
                    break
            return 1

        total_sims = 0
        for i, ((a_ri, a_nc, a_v), (p_ri, p_nc, p_v)) in enumerate(zip(actor1, py)):
            if a_ri != p_ri or a_nc != p_nc:
                print(f"FAIL: root {i} header differs: actor=({a_ri},{a_nc}) "
                      f"python=({p_ri},{p_nc})", file=sys.stderr)
                return 1
            if not np.array_equal(a_v, p_v):
                print(f"FAIL: visits differ at searched root {i} "
                      f"(index={a_ri}, num_choices={a_nc})", file=sys.stderr)
                print(f"  actor : {a_v.tolist()}", file=sys.stderr)
                print(f"  python: {p_v.tolist()}", file=sys.stderr)
                return 1
            total_sims += int(a_v.sum())

        print(f"PASS: MCTS visit-count parity exact over {len(actor1)} searched "
              f"roots ({total_sims} total root visits) "
              f"[deck={DECK} seed={SEED} sims={SIMS} worlds={WORLDS} "
              f"c={C_PUCT} batch=1]")

        # 5) K=16 batched-leaf sanity check (report only). The actor plays
        # argmax(visits) at each root, so once a batch=16 root's argmax differs
        # from batch=1 the real move — and every state after — diverges and the
        # remaining roots are searched over DIFFERENT positions (not comparable).
        # Report argmax agreement over the aligned prefix of identical positions
        # (same root_index + num_choices, which holds exactly while the two games
        # have made identical moves), and where the shared prefix ends.
        dump16 = os.path.join(td, "visits_b16.bin")
        actor16 = _run_actor(ts_path, dump16, batch=16)
        if actor16 is None:
            print("WARN: --batch 16 run failed; skipping agreement report",
                  file=sys.stderr)
            return 0
        prefix = 0
        for (r1, nc1, _), (r16, nc16, _) in zip(actor1, actor16):
            if r1 != r16 or nc1 != nc16:
                break
            prefix += 1
        agree = sum(1 for i in range(prefix)
                    if int(np.argmax(actor1[i][2])) == int(np.argmax(actor16[i][2])))
        frac = agree / prefix if prefix else 1.0
        diverged = "" if prefix == len(actor1) else (
            f"; games diverge after root {prefix} "
            f"(batch=1 {len(actor1)} roots, batch=16 {len(actor16)} roots)")
        print(f"REPORT: batch=16 vs batch=1 argmax agreement over {prefix} "
              f"comparable roots = {agree}/{prefix} = {frac:.3f}{diverged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
