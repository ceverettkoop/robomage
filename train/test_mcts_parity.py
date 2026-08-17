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
from env import MAX_ACTIONS, _MATCH_CTX_START, _SELF_IS_A_IDX
from cli_spec import BIN_DIR, BUILD_DIR
import runner
from search_env import SearchRoboMageEnv

DECK = "league/ur_delver"
SEED = 1
SIMS = 16
WORLDS = 2
C_PUCT = 1.5
SEED_BASE = 42
# Explicit sideboard-root budget for the sb-budget parity case (the in-game
# sims/worlds/max_depth stay SIMS/WORLDS/default). Mirrors az_selfplay's separate
# sb_sims/sb_worlds/sb_max_depth knobs; Stage 5 threads them into MCTSConfig.
SB_SIMS = 8
SB_WORLDS = 2
SB_MAX_DEPTH = 200
# Leaf-rollout horizon for the sb-budget case. 3 (not the shipped default 12)
# keeps the gate fast while still exercising every rollout mechanism path:
# the turn anchor, the sideboard-prefix gating, rollout terminals, and the
# cross-language argmax/eval lockstep. The bo1 and inherited-budget bo3 cases
# stay rollout-free as the unrolled baseline.
SB_ROLLOUT_TURNS = 3
# is_sideboard_phase flag in the state vector (env's _MATCH_CTX layout:
# game_number, self_wins, opp_wins, sideboard_phase).
_IS_SIDEBOARD_IDX = _MATCH_CTX_START + 3
ACTOR_BIN = os.path.join(BUILD_DIR, "az_actor")


def _seeds_for(root_index: int) -> list:
    """Shared per-world seed formula via mcts.world_seeds_for (the ONE Python
    home of az_mcts.cpp::begin_world's formula). Yields max(WORLDS, SB_WORLDS)
    seeds so both the in-game and the (possibly wider) sideboard budget can
    draw their first `worlds` entries from the same list."""
    from mcts import world_seeds_for
    return world_seeds_for(SEED_BASE, root_index, max(WORLDS, SB_WORLDS))


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

    def __init__(self, evaluator, sb_budget=False, sb_persist=False,
                 merge_dupes=True):
        from mcts import run_search  # noqa: F401 — fail fast if unavailable
        self.ev = evaluator
        self.env = None
        self.root_counter = 0
        self.records = []  # list[(root_index, num_choices, np.int64[num_choices])]
        self.is_sb = []    # parallel to records: was each searched root a sideboard root
        # Duplicate-edge merging (must match the actor's --merge-dupes).
        # rep_records holds each searched root's menu_merge_reps partition so
        # main() can assert the feature was actually exercised (>= 1
        # non-identity root) and that visit mass never lands on a
        # merged-away duplicate.
        self.merge_dupes = merge_dupes
        self.rep_records = []
        # When set, mirror the actor's sideboard budget (SB_SIMS/SB_WORLDS/
        # SB_MAX_DEPTH/SB_ROLLOUT_TURNS) at is_sideboard_phase roots; in-game
        # roots stay on the SIMS/WORLDS/default budget. Off => every root uses
        # the in-game budget (the inherited-budget bo1/bo3 cases).
        self.sb_budget = sb_budget
        # When ALSO set, mirror the actor's --sb-persist boundary machinery:
        # trees + memo persist across a boundary's picks (seeds pinned to the
        # boundary's first searched root, re-root + top-up per pick, cumulative
        # visits). The boundary dict holds key/root_r/roots/played/picks/memo.
        self.sb_persist = sb_persist
        self._boundary = None

    def bind_env(self, env):
        self.env = env

    def _latch(self, obs, action):
        """Record an actually-played action while a boundary is live (mirrors
        the actor latching in finalize + the fallback path)."""
        from mcts import sb_pick_descriptor
        b = self._boundary
        if b is None:
            return
        b["played"].append(int(action))
        d = sb_pick_descriptor(obs, int(action))
        if d is not None:
            b["picks"].append(d)

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None):
        from mcts import run_search, sb_root_key, walk_reuse_root
        env = self.env
        searchable = (env is not None and getattr(env, "last_search_safe", None)
                      and num_choices > 1)
        if not searchable:
            priors, _ = self.ev.evaluate(obs, num_choices)
            chosen = int(np.argmax(priors))
            self._latch(obs, chosen)
            return chosen
        r = self.root_counter
        self.root_counter += 1
        is_sb = bool(obs[_IS_SIDEBOARD_IDX] > 0.5)
        if self.sb_budget and is_sb:
            kw = dict(sims=SB_SIMS, worlds=SB_WORLDS, max_depth=SB_MAX_DEPTH,
                      rollout_turns=SB_ROLLOUT_TURNS, c_puct=C_PUCT,
                      merge_dupes=self.merge_dupes)
            if self.sb_persist:
                key = sb_root_key(obs)
                b = self._boundary
                if b is not None and b["key"] == key:
                    seat = bool(obs[_SELF_IS_A_IDX] > 0.5)
                    walked = [walk_reuse_root(root, b["played"], num_choices,
                                              seat)
                              for root in b["roots"]]
                    result = run_search(env, self.ev,
                                        world_seeds=_seeds_for(b["root_r"]),
                                        reuse_roots=walked,
                                        rollout_memo=b["memo"],
                                        memo_picks=tuple(b["picks"]), **kw)
                else:
                    b = {"key": key, "root_r": r, "memo": {}, "picks": [],
                         "played": [], "roots": None}
                    self._boundary = b
                    result = run_search(env, self.ev,
                                        world_seeds=_seeds_for(r),
                                        rollout_memo=b["memo"],
                                        memo_picks=(), **kw)
                b["roots"] = result.roots
                b["played"] = []
            else:
                result = run_search(env, self.ev, world_seeds=_seeds_for(r),
                                    **kw)
        else:
            self._boundary = None
            result = run_search(env, self.ev, sims=SIMS, worlds=WORLDS,
                                c_puct=C_PUCT, world_seeds=_seeds_for(r),
                                merge_dupes=self.merge_dupes)
        self.records.append((r, int(num_choices),
                             result.visits.astype(np.int64)))
        self.is_sb.append(is_sb)
        from decode import menu_merge_reps
        self.rep_records.append(menu_merge_reps(obs, num_choices))
        chosen = result.best_action()
        self._latch(obs, chosen)
        return chosen


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


def _run_actor(ts_path, dump_path, batch, bo3=False, sb=None, sb_persist=False,
               merge_dupes=True):
    cmd = [ACTOR_BIN, "--search", "--sims", str(SIMS), "--worlds", str(WORLDS),
           "--c", str(C_PUCT), "--batch", str(batch), "--world-seeds",
           str(SEED_BASE), "--deck", DECK, "--seed", str(SEED),
           "--model", ts_path, "--dump-visits", dump_path, "--games", "1",
           "--merge-dupes", str(int(merge_dupes))]
    if bo3:
        cmd.append("--bo3")
    if sb is not None:
        cmd += ["--sb-sims", str(sb[0]), "--sb-worlds", str(sb[1]),
                "--sb-max-depth", str(sb[2]), "--sb-rollout-turns", str(sb[3]),
                "--sb-persist", str(int(sb_persist))]
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("FAIL: az_actor exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    return _read_visits_dump(dump_path)


def _python_reference(ts_path, bo3, sb_budget=False, sb_persist=False,
                      merge_dupes=True):
    """Drive the SAME game (bo1) or MATCH (bo3) through the Python reference,
    searching each loop-safe root. When ``sb_budget`` mirrors the actor's separate
    sideboard budget at is_sideboard_phase roots; ``sb_persist`` additionally
    mirrors --sb-persist boundary trees + the rollout memo.
    Returns (records, is_sb, rep_records)."""
    ev = TSEvaluator(ts_path)
    ctrl = ParitySearchController(ev, sb_budget=sb_budget, sb_persist=sb_persist,
                                  merge_dupes=merge_dupes)
    env = SearchRoboMageEnv(deck_a=DECK, deck_b=DECK, bo3=bo3)
    # The C++ actor plays to the engine's natural end (no decision cap); disable
    # RoboMageEnv's training-only step truncation so both sides run the SAME full
    # game/match and compare an equal number of searched roots.
    env.MAX_STEPS = env.MAX_STEPS_BO3 = 1 << 30
    ctrl.bind_env(env)
    try:
        obs, _ = env.reset(options={"engine_seed": SEED})
        runner.drive_game(env, obs, ctrl, ctrl)
    finally:
        env.close()
    return ctrl.records, ctrl.is_sb, ctrl.rep_records


def _compare_visits(tag, actor1, py):
    """Assert exact visit-count parity between the actor (batch=1) and the Python
    reference over every searched root. Returns (rc, total_sims)."""
    if len(actor1) != len(py):
        print(f"FAIL [{tag}]: searched-root count differs — actor={len(actor1)} "
              f"python={len(py)}", file=sys.stderr)
        n = min(len(actor1), len(py))
        for i in range(n):
            if (actor1[i][1] != py[i][1]
                    or not np.array_equal(actor1[i][2], py[i][2])):
                print(f"  first divergence at searched root {i} "
                      f"(nc actor={actor1[i][1]} python={py[i][1]})",
                      file=sys.stderr)
                break
        return 1, 0

    total_sims = 0
    for i, ((a_ri, a_nc, a_v), (p_ri, p_nc, p_v)) in enumerate(zip(actor1, py)):
        if a_ri != p_ri or a_nc != p_nc:
            print(f"FAIL [{tag}]: root {i} header differs: actor=({a_ri},{a_nc}) "
                  f"python=({p_ri},{p_nc})", file=sys.stderr)
            return 1, 0
        if not np.array_equal(a_v, p_v):
            print(f"FAIL [{tag}]: visits differ at searched root {i} "
                  f"(index={a_ri}, num_choices={a_nc})", file=sys.stderr)
            print(f"  actor : {a_v.tolist()}", file=sys.stderr)
            print(f"  python: {p_v.tolist()}", file=sys.stderr)
            return 1, 0
        total_sims += int(a_v.sum())
    return 0, total_sims


def _sb_root_summary(records, is_sb):
    """Summarize the sideboard-phase searched roots: count + each root's total
    visit count. Without persistence a root's total visits == its sims budget
    (SIMS inherited, SB_SIMS under the explicit sb budget); WITH persistence
    the totals are CUMULATIVE (inherited + topped-up), so within a boundary
    the numbers read out how much mass each re-rooted pick carried forward."""
    sums = [int(v.sum()) for (_ri, _nc, v), sb in zip(records, is_sb) if sb]
    return f"{len(sums)} sideboard root(s), visit totals {sums}"


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

        # bo1 game AND a full bo3 match. The bo3 case exercises the between-games
        # sideboard-phase decisions, which are now SEARCHED MCTS roots on BOTH
        # sides (Stage 1-3) at the INHERITED in-game budget — so the two games
        # stay in lockstep across the match and every searched root's visit vector
        # (including the sideboard roots) must match bit-for-bit.
        actor1 = None            # bo1 batch=1 visits, reused by the K=16 report below
        sb_default_summary = None  # bo3 inherited-budget sideboard-root summary (gate #4)
        merged_roots = 0         # searched roots whose menu had duplicate groups
        for bo3 in (False, True):
            tag = "bo3" if bo3 else "bo1"
            dump1 = os.path.join(td, f"visits_b1_{tag}.bin")
            visits = _run_actor(ts_path, dump1, batch=1, bo3=bo3)
            if visits is None:
                return 1
            py, is_sb, py_reps = _python_reference(ts_path, bo3)
            rc, total_sims = _compare_visits(tag, visits, py)
            if rc:
                return rc
            # Merge invariants: visit mass never lands on a merged-away
            # duplicate, and (asserted after both games) the run actually
            # contained duplicate-bearing menus — otherwise the merge parity
            # would be passing vacuously.
            for i, ((_ri, nc, v), rep) in enumerate(zip(py, py_reps)):
                nonrep = np.arange(nc) != rep[:nc]
                merged_roots += int(nonrep.any())
                if np.any(v[:nc][nonrep] != 0):
                    print(f"FAIL [{tag}]: root {i} has visits on merged-away "
                          f"duplicates (rep={rep.tolist()}, "
                          f"visits={v.tolist()})", file=sys.stderr)
                    return 1
            print(f"PASS [{tag}]: MCTS visit-count parity exact over {len(visits)} "
                  f"searched roots ({total_sims} total root visits) "
                  f"[deck={DECK} seed={SEED} sims={SIMS} worlds={WORLDS} "
                  f"c={C_PUCT} batch=1]")
            if bo3:
                sb_default_summary = _sb_root_summary(visits, is_sb)
            else:
                actor1 = visits
        if merged_roots == 0:
            print("FAIL: no searched root contained duplicate menu actions — "
                  "the merge feature was not exercised (deck/seed change?)",
                  file=sys.stderr)
            return 1
        print(f"PASS [merge]: {merged_roots} searched root(s) had duplicate "
              f"groups; zero visit mass on merged-away duplicates everywhere")

        # Kill-switch parity (bo1): --merge-dupes 0 on both sides must
        # reproduce the legacy per-copy-edge search bit-for-bit.
        dump_nm = os.path.join(td, "visits_b1_nomerge.bin")
        actor_nm = _run_actor(ts_path, dump_nm, batch=1, merge_dupes=False)
        if actor_nm is None:
            return 1
        py_nm, _, _ = _python_reference(ts_path, bo3=False, merge_dupes=False)
        rc, total_sims = _compare_visits("bo1-nomerge", actor_nm, py_nm)
        if rc:
            return rc
        print(f"PASS [bo1-nomerge]: kill-switch (--merge-dupes 0) parity exact "
              f"over {len(actor_nm)} searched roots ({total_sims} total root "
              f"visits)")

        # Explicit sb-budget cases (bo3): run the actor with a SEPARATE sideboard
        # budget (--sb-sims/--sb-worlds/--sb-max-depth/--sb-rollout-turns) while
        # the in-game budget stays SIMS/WORLDS/default, and mirror the same split
        # in the Python reference (keyed off the is_sideboard_phase state flag).
        # Run it BOTH ways on the boundary-persistence toggle: persist ON (trees
        # + rollout memo survive across each boundary's picks — the production
        # default) and persist OFF (per-pick fresh searches, the pre-persistence
        # baseline). Every root must be bit-exact in both modes.
        sb_budget_summary = {}
        for persist in (True, False):
            tag = "bo3-sb-persist" if persist else "bo3-sb"
            dump_sb = os.path.join(td, f"visits_b1_{tag}.bin")
            actor_sb = _run_actor(ts_path, dump_sb, batch=1, bo3=True,
                                  sb=(SB_SIMS, SB_WORLDS, SB_MAX_DEPTH,
                                      SB_ROLLOUT_TURNS), sb_persist=persist)
            if actor_sb is None:
                return 1
            py_sb, is_sb_sb, _ = _python_reference(ts_path, bo3=True,
                                                   sb_budget=True,
                                                   sb_persist=persist)
            rc, total_sims = _compare_visits(tag, actor_sb, py_sb)
            if rc:
                return rc
            sb_budget_summary[tag] = _sb_root_summary(actor_sb, is_sb_sb)
            print(f"PASS [{tag}]: MCTS visit-count parity exact over "
                  f"{len(actor_sb)} searched roots ({total_sims} total root "
                  f"visits) [in-game sims={SIMS} worlds={WORLDS}; sideboard "
                  f"sims={SB_SIMS} worlds={SB_WORLDS} max_depth={SB_MAX_DEPTH} "
                  f"rollout_turns={SB_ROLLOUT_TURNS} persist={int(persist)}]")
        # Prove the budget split (and, under persist, the carried-forward
        # cumulative visits) took effect at the sideboard roots.
        print(f"REPORT: sideboard-root budget split — default bo3 (inherited): "
              f"{sb_default_summary}; sb-budget bo3: {sb_budget_summary['bo3-sb']}; "
              f"sb-persist bo3: {sb_budget_summary['bo3-sb-persist']}")

        # 5) K=16 batched-leaf sanity check (report only, bo1). The actor plays
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
