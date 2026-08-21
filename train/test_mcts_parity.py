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

Cross-world batching (--cross-world; Stage 0 of
docs/gpu_selfplay_inference_plan.md) gets two legs: EXACT visit gates under
--uniform (bo1 + bo3-sb-persist — with no net, the scheduler must reproduce the
sequential visits bit-for-bit), and a real-net argmax-agreement REPORT (only the
batched forward's last-ulp logits can differ).

Two-model gate matches (az_actor --model-b — the az_eval promotion gate's actor
backend) get their own leg family (see _gate_legs): same-net identity gates
(bit-identical to single-model), a wiring check (a distinct second net must
change the game), EXACT parity vs a two-controller Python gate reference
(per-seat evaluators, shared global root counter), and a cross-world agreement
report on the two-model game.

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
# Opponent deck of the vs-scripted parity legs (cross-deck, so the scripted
# agent's deck-aware heuristics see real deck names on both sides).
SCRIPTED_DECK_B = "league/gw_maverick"
SEED = 1
SIMS = 16
WORLDS = 2
C_PUCT = 1.5
SEED_BASE = 42
# Explicit sideboard PLAN-search budget for the bo3 parity cases (the in-game
# sims/worlds/max_depth stay SIMS/WORLDS/default). Mirrors az_selfplay's
# sb_branches/sb_worlds/sb_rollout_turns knobs.
SB_BRANCHES = 2
SB_WORLDS = 2
# Plan-pricing rollout horizon for the sb cases. 3 (not the shipped default 6)
# keeps the gate fast while still exercising every plan mechanism path: the
# greedy completion + second-best deviation, the turn anchor, replay across
# worlds, plan terminals, the boundary memo, and the cross-language
# argmax/eval lockstep. The bo1 and inherited-budget bo3 cases stay
# rollout-free as the unrolled baseline.
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

    def __init__(self, evaluator, merge_dupes=True, share_with=None):
        from mcts import run_search  # noqa: F401 — fail fast if unavailable
        self.ev = evaluator
        self.env = None
        # Searched-root counter, boxed so a two-seat GATE reference (one
        # controller per seat, each with its OWN evaluator) can share one
        # global counter — mirroring the actor, whose root_counter (and hence
        # the per-root world-seed formula) runs across BOTH seats' searches.
        self.counter = [0]
        # Per searched root: (root_index, num_choices, payload) where payload
        # is the int64 visit vector for a TREE root, or ("plan", q, pi) float64
        # arrays for a sideboard PLAN root — mirroring the actor's dump.
        self.records = []
        self.is_sb = []    # parallel to records: was each searched root a sideboard root
        # Duplicate-edge merging (must match the actor's --merge-dupes).
        # rep_records holds each searched root's menu_merge_reps partition so
        # main() can assert the feature was actually exercised (>= 1
        # non-identity root) and that visit mass never lands on a
        # merged-away duplicate.
        self.merge_dupes = merge_dupes
        self.rep_records = []
        # Sideboard boundary (mirrors the actor's plan-search boundary): the
        # plan-value memo + latched pick descriptors carry across a boundary's
        # picks; world seeds stay pinned to the boundary's first searched root.
        self._boundary = None
        # Gate reference: share the counter and the record streams with the
        # other seat's controller so both seats append into ONE global
        # search-order stream, exactly like the actor's single Impl.
        if share_with is not None:
            self.counter = share_with.counter
            self.records = share_with.records
            self.is_sb = share_with.is_sb
            self.rep_records = share_with.rep_records

    def bind_env(self, env):
        self.env = env

    def _latch(self, obs, action):
        """Record an actually-played pick's descriptor while a boundary is live
        (mirrors the actor latching in finalize_plan_search + the fallback
        path — the memo-key base)."""
        from mcts import sb_pick_descriptor
        b = self._boundary
        if b is None:
            return
        d = sb_pick_descriptor(obs, int(action))
        if d is not None:
            b["picks"].append(d)

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None):
        from mcts import run_search, run_plan_search, sb_root_key
        env = self.env
        searchable = (env is not None and getattr(env, "last_search_safe", None)
                      and num_choices > 1)
        if not searchable:
            priors, _ = self.ev.evaluate(obs, num_choices)
            chosen = int(np.argmax(priors))
            self._latch(obs, chosen)
            return chosen
        r = self.counter[0]
        self.counter[0] += 1
        is_sb = bool(obs[_IS_SIDEBOARD_IDX] > 0.5)
        if is_sb:
            # Sideboard plan search, mirroring the actor: boundary identity by
            # sb_root_key; the memo + latched picks carry across the boundary's
            # picks; the world seeds pin to the boundary's FIRST searched root.
            key = sb_root_key(obs)
            b = self._boundary
            if b is None or b["key"] != key:
                b = {"key": key, "root_r": r, "memo": {}, "picks": []}
                self._boundary = b
            result = run_plan_search(
                env, self.ev, worlds=SB_WORLDS, branches=SB_BRANCHES,
                rollout_turns=SB_ROLLOUT_TURNS,
                world_seeds=_seeds_for(b["root_r"])[:SB_WORLDS],
                plan_memo=b["memo"], memo_picks=tuple(b["picks"]))
            self.records.append((r, int(num_choices),
                                 ("plan", np.array(result.q, copy=True),
                                  result.policy_target(1.0))))
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
    record). A TREE root's record is int64[num_choices] visits; a PLAN
    (sideboard) root writes num_choices NEGATED as a type tag followed by
    float64 q[num_choices] + pi[num_choices]. Returns
    list[(root_index, num_choices, payload)] with payload matching
    ParitySearchController.records."""
    with open(path, "rb") as f:
        data = f.read()
    out = []
    off = 0
    while off < len(data):
        ri, nc = struct.unpack_from("<ii", data, off)
        off += 8
        if nc < 0:
            nc = -nc
            q = np.frombuffer(data, dtype="<f8", count=nc, offset=off)
            off += 8 * nc
            pi = np.frombuffer(data, dtype="<f8", count=nc, offset=off)
            off += 8 * nc
            out.append((ri, nc, ("plan", np.array(q, copy=True),
                                 np.array(pi, copy=True))))
        else:
            vals = np.frombuffer(data, dtype="<i8", count=nc, offset=off)
            off += 8 * nc
            out.append((ri, nc, np.array(vals, dtype=np.int64, copy=True)))
    if off != len(data):
        raise RuntimeError(f"visits dump {path}: {len(data) - off} trailing bytes")
    return out


def _run_actor(ts_path, dump_path, batch, bo3=False,
               merge_dupes=True, cross_world=False, uniform=False,
               deck_b=None, scripted_sock=None, eval_server=None, model_b=None):
    cmd = [ACTOR_BIN, "--search", "--sims", str(SIMS), "--worlds", str(WORLDS),
           "--c", str(C_PUCT), "--batch", str(batch), "--world-seeds",
           str(SEED_BASE), "--deck", DECK, "--seed", str(SEED),
           "--dump-visits", dump_path, "--games", "1",
           "--merge-dupes", str(int(merge_dupes))]
    if eval_server is not None:
        cmd += ["--eval-server", eval_server]
    else:
        cmd += ["--uniform"] if uniform else ["--model", ts_path]
    if model_b is not None:
        cmd += ["--model-b", model_b]
    if cross_world:
        cmd.append("--cross-world")
    if deck_b is not None:
        cmd += ["--deck-b", deck_b]
    if scripted_sock is not None:
        cmd += ["--scripted-seat", "B", "--scripted-oracle", scripted_sock]
    if bo3:
        # A bo3 always crosses sideboard roots, which the actor plan-searches
        # unconditionally — pin the test's small plan budget so both sides use
        # SB_BRANCHES/SB_WORLDS/SB_ROLLOUT_TURNS (there is no inherit mode).
        cmd += ["--bo3", "--sb-branches", str(SB_BRANCHES),
                "--sb-worlds", str(SB_WORLDS),
                "--sb-rollout-turns", str(SB_ROLLOUT_TURNS)]
    proc = subprocess.run(cmd, cwd=BIN_DIR, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("FAIL: az_actor exited nonzero:\n"
              + proc.stderr.decode("utf-8", "replace"), file=sys.stderr)
        return None
    return _read_visits_dump(dump_path)


def _start_eval_server(ts_path, sock, device):
    """Start train/az_eval_server.py and wait for its READY line. Returns the
    Popen, or None if the server died before READY (e.g. --device cuda on a
    box with no usable GPU — the caller treats that as a skip, not a failure).
    The ROCm RDNA2 env overrides are applied for a cuda server, mirroring
    az_selfplay._generate_actor."""
    server_py = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "az_eval_server.py")
    env = dict(os.environ)
    if device != "cpu":
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
        env.setdefault("HIP_VISIBLE_DEVICES", "0")
    proc = subprocess.Popen([sys.executable, server_py, "--model", ts_path,
                             "--socket", sock, "--device", device],
                            env=env, text=True, bufsize=1,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in proc.stdout:
        if line.startswith("READY "):
            # Drain further output so the pipe never fills.
            import threading
            threading.Thread(target=lambda: [None for _ in proc.stdout],
                             daemon=True).start()
            return proc
    proc.wait()
    return None


def _stop_eval_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _python_reference(ts_path, bo3, merge_dupes=True):
    """Drive the SAME game (bo1) or MATCH (bo3) through the Python reference,
    searching each loop-safe root — run_search at in-game roots,
    run_plan_search (the sideboard plan search, SB_BRANCHES/SB_WORLDS/
    SB_ROLLOUT_TURNS, boundary memo + pinned seeds) at sideboard roots.
    Returns (records, is_sb, rep_records)."""
    ev = TSEvaluator(ts_path)
    ctrl = ParitySearchController(ev, merge_dupes=merge_dupes)
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


def _python_reference_scripted(ts_path, deck_b, uniform=False):
    """The vs-scripted twin of :func:`_python_reference` (bo1): net+MCTS on
    seat A (ParitySearchController, same seed formula), scripted:hard on seat B
    — the EXACT `_play_match` agent/net_is_a semantics: the scripted seat's
    decisions come from ``agent.act`` on the same obs the actor ships to the
    oracle, with no search, no record, and (bo1) one ``new_game`` at start.
    Both sides run the identical scripted_agent code on bit-identical
    observations, so whole-game visit parity is EXACT, not statistical."""
    from mcts import UniformEvaluator
    from scripted_agent import make_agent
    ev = UniformEvaluator() if uniform else TSEvaluator(ts_path)
    ctrl = ParitySearchController(ev)
    agent = make_agent("scripted:hard")
    agent.set_deck_names(DECK, deck_b)
    env = SearchRoboMageEnv(deck_a=DECK, deck_b=deck_b)
    env.MAX_STEPS = env.MAX_STEPS_BO3 = 1 << 30
    ctrl.bind_env(env)
    try:
        env.reset(options={"engine_seed": SEED})
        agent.new_game()
        done = False
        while not done:
            nc = env._num_choices
            o = env._obs
            if o[_SELF_IS_A_IDX] > 0.5:
                a = ctrl.choose(o, nc)
            else:
                a = int(agent.act(o, nc)) if nc > 1 else 0
            _obs, _r, term, trunc, _info = env.step(a)
            done = term or trunc
    finally:
        env.close()
    return ctrl.records


def _python_reference_gate(ts_a, ts_b, bo3):
    """Two-model gate reference: seat A searches with ``ts_a``'s net, seat B
    with ``ts_b``'s — one ParitySearchController per seat, each owning its
    evaluator, SHARING one root counter and one record stream (the actor's
    single root_counter runs across both seats). This is exactly the shape of
    an az_eval gate match on the Python backend, so exact visit parity here is
    the gate-correctness proof for ``az_actor --model-b``."""
    ctrl_a = ParitySearchController(TSEvaluator(ts_a))
    ctrl_b = ParitySearchController(TSEvaluator(ts_b), share_with=ctrl_a)
    env = SearchRoboMageEnv(deck_a=DECK, deck_b=DECK, bo3=bo3)
    env.MAX_STEPS = env.MAX_STEPS_BO3 = 1 << 30
    ctrl_a.bind_env(env)
    ctrl_b.bind_env(env)
    try:
        obs, _ = env.reset(options={"engine_seed": SEED})
        runner.drive_game(env, obs, ctrl_a, ctrl_b)
    finally:
        env.close()
    return ctrl_a.records


def _is_plan_payload(payload):
    return isinstance(payload, tuple) and len(payload) == 3 and payload[0] == "plan"


def _payloads_match(a, p):
    """Exactness rule per root kind: TREE visit vectors are int64 and must be
    bit-exact; a PLAN root's Q must be bit-exact (pure shared float64
    arithmetic), while its pi tolerates last-ulp differences (softmax exp —
    numpy's SIMD exp and libm's std::exp may differ in the final ulp; argmax
    pi == argmax Q, so play lockstep is unaffected)."""
    if _is_plan_payload(a) != _is_plan_payload(p):
        return False
    if _is_plan_payload(a):
        return (np.array_equal(a[1], p[1])
                and np.allclose(a[2], p[2], rtol=0.0, atol=1e-12))
    return np.array_equal(a, p)


def _compare_visits(tag, actor1, py):
    """Assert exact search parity between the actor (batch=1) and the Python
    reference over every searched root (see _payloads_match for the per-kind
    exactness rule). Returns (rc, total_sims) — plan roots contribute no
    sims to the total."""
    if len(actor1) != len(py):
        print(f"FAIL [{tag}]: searched-root count differs — actor={len(actor1)} "
              f"python={len(py)}", file=sys.stderr)
        n = min(len(actor1), len(py))
        for i in range(n):
            if (actor1[i][1] != py[i][1]
                    or not _payloads_match(actor1[i][2], py[i][2])):
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
        if not _payloads_match(a_v, p_v):
            print(f"FAIL [{tag}]: search results differ at searched root {i} "
                  f"(index={a_ri}, num_choices={a_nc})", file=sys.stderr)
            if _is_plan_payload(a_v) and _is_plan_payload(p_v):
                print(f"  actor q : {a_v[1].tolist()}", file=sys.stderr)
                print(f"  python q: {p_v[1].tolist()}", file=sys.stderr)
                print(f"  actor pi : {a_v[2].tolist()}", file=sys.stderr)
                print(f"  python pi: {p_v[2].tolist()}", file=sys.stderr)
            else:
                print(f"  actor : {a_v if _is_plan_payload(a_v) else a_v.tolist()}",
                      file=sys.stderr)
                print(f"  python: {p_v if _is_plan_payload(p_v) else p_v.tolist()}",
                      file=sys.stderr)
            return 1, 0
        if not _is_plan_payload(a_v):
            total_sims += int(a_v.sum())
    return 0, total_sims


def _sb_root_summary(records, is_sb):
    """Summarize the sideboard-phase searched roots: count + each plan root's
    best-Q first pick and its Q (the quantity the plan search ranks by)."""
    tops = []
    for (_ri, _nc, payload), sb in zip(records, is_sb):
        if not sb or not _is_plan_payload(payload):
            continue
        q = payload[1]
        tops.append(f"{int(np.argmax(q))}@{float(np.max(q)):+.3f}")
    return f"{len(tops)} sideboard plan root(s), best picks [{', '.join(tops)}]"


def _server_legs(td, ts_path, actor1):
    """Stage C legs (docs/gpu_selfplay_inference_plan.md), compared against
    ``actor1`` (the local --model bo1 visit records).

    (a) EXACT gate, --device cpu: a CPU server forwards the SAME TorchScript
    module single-threaded, and the client keeps the prior math local — so the
    wire protocol must be bit-transparent: server-backed visits equal the local
    --model visits exactly. Runs in CI with no GPU.

    (b) REPORT, --device cuda, ONE bo1 game: the GPU's GEMMs may differ from
    CPU in last-ulp logits, so visits may drift — quantified as argmax
    agreement + visit L1 over the aligned prefix (not asserted; ulp flips on
    near-ties are legitimate). Skips cleanly where no usable GPU exists (the
    server dies before READY)."""
    sock = os.path.join(td, "evs")
    server = _start_eval_server(ts_path, sock, "cpu")
    if server is None:
        print("FAIL [server-cpu]: az_eval_server.py exited before READY",
              file=sys.stderr)
        return 1
    try:
        dump_srv = os.path.join(td, "visits_server_cpu.bin")
        actor_srv = _run_actor(ts_path, dump_srv, batch=1, eval_server=sock)
    finally:
        _stop_eval_server(server)
    if actor_srv is None:
        return 1
    rc, total_sims = _compare_visits("server-cpu", actor_srv, actor1)
    if rc:
        print("FAIL [server-cpu]: server-backed visits diverged from the "
              "local --model search (the 'actor'/'python' rows above are "
              "server/local) — the wire protocol or the client's local "
              "prior math is not bit-transparent", file=sys.stderr)
        return rc
    print(f"PASS [server-cpu]: eval-server visits bit-exact vs local "
          f"forward over {len(actor_srv)} searched roots ({total_sims} "
          f"total root visits)")

    server = _start_eval_server(ts_path, sock, "cuda")
    if server is None:
        print("REPORT: server-gpu drift skipped (no usable GPU — "
              "az_eval_server --device cuda died before READY)")
        return 0
    try:
        dump_gpu = os.path.join(td, "visits_server_gpu.bin")
        actor_gpu = _run_actor(ts_path, dump_gpu, batch=1, eval_server=sock)
    finally:
        _stop_eval_server(server)
    if actor_gpu is None:
        print("WARN: server-gpu run failed; skipping drift report",
              file=sys.stderr)
        return 0
    prefix = 0
    for (r1, nc1, _), (rg, ncg, _) in zip(actor1, actor_gpu):
        if r1 != rg or nc1 != ncg:
            break
        prefix += 1
    agree = sum(1 for i in range(prefix)
                if int(np.argmax(actor1[i][2])) == int(np.argmax(actor_gpu[i][2])))
    frac = agree / prefix if prefix else 1.0
    drift = [int(np.abs(actor_gpu[i][2] - actor1[i][2]).sum())
             for i in range(prefix)]
    diverged = "" if prefix == len(actor1) else (
        f"; games diverge after root {prefix} "
        f"(cpu {len(actor1)} roots, gpu {len(actor_gpu)} roots)")
    print(f"REPORT: server-gpu vs local-cpu (one bo1 game) argmax agreement "
          f"over {prefix} comparable roots = {agree}/{prefix} = {frac:.3f}; "
          f"visit-count L1 drift max={max(drift) if drift else 0} "
          f"mean={sum(drift)/len(drift):.2f}{diverged}"
          if drift else
          f"REPORT: server-gpu vs local-cpu: no comparable roots{diverged}")
    return 0


def _gate_legs(td, ts_path, actor1):
    """Two-model gate legs (az_actor --model-b, the az_eval gate's backend).

    (a) gate-identity EXACT gates (bo1 + bo3-sb-persist): ``--model A
    --model-b A`` must be bit-identical to the single-model ``--model A`` run
    — the per-seat evaluator selection is a pure no-op when both seats hold
    the same net, so ANY difference is a selection/plumbing bug.

    (b) gate-wiring: a ``--model A --model-b B`` run (B = a second,
    differently-seeded net) must DIFFER from the single-model A run — if the
    two streams were identical, --model-b was silently ignored (two
    independently random nets agreeing on every searched root of a full game
    is not a real possibility).

    (c) gate-parity EXACT gate (bo1 + bo3-sb-persist): the (A, B) actor run vs
    the two-controller Python reference (seat A's controller owns net A, seat
    B's net B, shared global root counter — exactly an az_eval gate match on
    the Python backend). Proves the actor's per-seat routing is precisely
    "the root mover's net evaluates that decision's whole search".

    (d) REPORT: (A, B) cross-world vs sequential argmax agreement — the gate
    driver defaults to --cross-world, whose visits differ from sequential only
    by the batched forward's last-ulp logits."""
    # Second deterministic net, differently seeded: the "incumbent".
    torch.manual_seed(1)
    net_b = AZNet(obs_space_from_const()).eval()
    ckpt_b = os.path.join(td, "parity_b__azfinal.pt")
    net_b.save(ckpt_b)
    ts_b = torchscript_export_path(ckpt_b)
    save_torchscript(net_b, ts_b)

    sb_kw = dict(bo3=True)  # bo3 pins the test's sb plan budget in _run_actor

    # (a) identity: same net on both seats == single-model, bit-exact.
    for tag, kw, ref in (("gate-id-bo1", {}, actor1),
                         ("gate-id-bo3-sb", sb_kw, None)):
        if ref is None:
            d_ref = os.path.join(td, f"visits_{tag}_single.bin")
            ref = _run_actor(ts_path, d_ref, batch=1, **kw)
            if ref is None:
                return 1
        d_id = os.path.join(td, f"visits_{tag}_twomodel.bin")
        actor_id = _run_actor(ts_path, d_id, batch=1, model_b=ts_path, **kw)
        if actor_id is None:
            return 1
        rc, total_sims = _compare_visits(tag, actor_id, ref)
        if rc:
            print(f"FAIL [{tag}]: --model-b <same net> diverged from the "
                  f"single-model run (the 'actor'/'python' rows above are "
                  f"two-model/single-model) — the per-seat evaluator "
                  f"selection is not a no-op for identical nets",
                  file=sys.stderr)
            return rc
        print(f"PASS [{tag}]: --model-b (same net) bit-identical to the "
              f"single-model run over {len(actor_id)} searched roots "
              f"({total_sims} total root visits)")

    # (b)+(c): the real two-model game, actor vs Python gate reference.
    d_ab = os.path.join(td, "visits_gate_ab_bo1.bin")
    actor_ab = _run_actor(ts_path, d_ab, batch=1, model_b=ts_b)
    if actor_ab is None:
        return 1
    same = (len(actor_ab) == len(actor1)
            and all(a[0] == b[0] and a[1] == b[1] and np.array_equal(a[2], b[2])
                    for a, b in zip(actor_ab, actor1)))
    if same:
        print("FAIL [gate-wiring]: the (A, B) two-model run is bit-identical "
              "to the single-model A run — --model-b appears to be ignored",
              file=sys.stderr)
        return 1
    print(f"PASS [gate-wiring]: --model-b (distinct net) changed the game "
          f"({len(actor_ab)} vs {len(actor1)} searched roots)")

    for tag, akw in (
            ("gate-parity-bo1", {}),
            ("gate-parity-bo3-sb", sb_kw)):
        d = os.path.join(td, f"visits_{tag}.bin")
        actor_two = _run_actor(ts_path, d, batch=1, model_b=ts_b, **akw)
        if actor_two is None:
            return 1
        py_two = _python_reference_gate(ts_path, ts_b, bo3=bool(akw.get("bo3")))
        rc, total_sims = _compare_visits(tag, actor_two, py_two)
        if rc:
            print(f"FAIL [{tag}]: two-model actor diverged from the "
                  f"two-controller Python gate reference", file=sys.stderr)
            return rc
        print(f"PASS [{tag}]: two-model (A vs B) visit parity exact over "
              f"{len(actor_two)} searched roots ({total_sims} total root "
              f"visits)")

    # (d) cross-world on the two-model game (the gate driver's default).
    d_xw = os.path.join(td, "visits_gate_ab_xw.bin")
    actor_xw = _run_actor(ts_path, d_xw, batch=1, model_b=ts_b,
                          cross_world=True)
    if actor_xw is None:
        print("WARN: two-model --cross-world run failed; skipping report",
              file=sys.stderr)
        return 1
    prefix = 0
    for (r1, nc1, _), (rx, ncx, _) in zip(actor_ab, actor_xw):
        if r1 != rx or nc1 != ncx:
            break
        prefix += 1
    agree = sum(1 for i in range(prefix)
                if int(np.argmax(actor_ab[i][2])) == int(np.argmax(actor_xw[i][2])))
    frac = agree / prefix if prefix else 1.0
    diverged = "" if prefix == len(actor_ab) else (
        f"; games diverge after root {prefix} "
        f"(sequential {len(actor_ab)} roots, cross-world {len(actor_xw)} roots)")
    print(f"REPORT: two-model cross-world vs sequential argmax agreement over "
          f"{prefix} comparable roots = {agree}/{prefix} = {frac:.3f}{diverged}")
    return 0


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="MCTS visit-parity gates (C++ actor vs Python reference)")
    ap.add_argument("--legs", choices=["full", "server", "gate"], default="full",
                    help="'full' (default) runs every gate incl. the slow "
                         "Python-reference legs; 'server' runs only the Stage C "
                         "eval-server legs against a fresh local bo1 reference "
                         "(~minutes — for iterating on server/protocol work); "
                         "'gate' runs only the two-model (--model-b) gate legs "
                         "the az_eval actor backend rides on")
    args = ap.parse_args()

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

        if args.legs in ("server", "gate"):
            # Fast path: one local bo1 game as the reference, then only the
            # requested leg family — no full Python-reference sweep.
            dump1 = os.path.join(td, "visits_b1_bo1.bin")
            actor1 = _run_actor(ts_path, dump1, batch=1)
            if actor1 is None:
                return 1
            print(f"[legs={args.legs}] local bo1 reference: {len(actor1)} "
                  f"searched roots")
            if args.legs == "server":
                return _server_legs(td, ts_path, actor1)
            return _gate_legs(td, ts_path, actor1)

        # bo1 game AND a full bo3 match. The bo3 case exercises the between-games
        # sideboard prompts, which are PLAN-searched on both sides
        # (mcts.run_plan_search / the actor's PLAN_* phases, at the test's
        # SB_BRANCHES/SB_WORLDS/SB_ROLLOUT_TURNS budget with the boundary memo
        # live) — so the two games stay in lockstep across the match and every
        # searched root must match: tree roots by bit-exact visit vectors, plan
        # roots by bit-exact Q (pi to 1e-12; see _payloads_match).
        actor1 = None            # bo1 batch=1 visits, reused by the K=16 report below
        sb_plan_summary = None   # bo3 sideboard plan-root summary
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
            # Merge invariants (tree roots only — a plan root has no search
            # edges to merge; the delta menu is per-distinct-card already):
            # visit mass never lands on a merged-away duplicate, and (asserted
            # after both games) the run actually contained duplicate-bearing
            # menus — otherwise the merge parity would be passing vacuously.
            for i, ((_ri, nc, v), rep) in enumerate(zip(py, py_reps)):
                if _is_plan_payload(v):
                    continue
                nonrep = np.arange(nc) != rep[:nc]
                merged_roots += int(nonrep.any())
                if np.any(v[:nc][nonrep] != 0):
                    print(f"FAIL [{tag}]: root {i} has visits on merged-away "
                          f"duplicates (rep={rep.tolist()}, "
                          f"visits={v.tolist()})", file=sys.stderr)
                    return 1
            print(f"PASS [{tag}]: search parity exact over {len(visits)} "
                  f"searched roots ({total_sims} total root visits) "
                  f"[deck={DECK} seed={SEED} sims={SIMS} worlds={WORLDS} "
                  f"c={C_PUCT} batch=1"
                  + (f"; sideboard plan budget branches={SB_BRANCHES} "
                     f"worlds={SB_WORLDS} rollout_turns={SB_ROLLOUT_TURNS}"
                     if bo3 else "") + "]")
            if bo3:
                sb_plan_summary = _sb_root_summary(visits, is_sb)
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

        # Prove the plan search actually engaged at the bo3 sideboard roots.
        print(f"REPORT: bo3 sideboard plan search — {sb_plan_summary}")

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

        # 6) Cross-world batching (Stage 0 of docs/gpu_selfplay_inference_plan.md).
        # (a) EXACT gates, --uniform: the uniform evaluator returns identical
        # priors however leaves are grouped, so the cross-world scheduler must
        # reproduce the sequential (batch=1) visits BIT-EXACT — any difference
        # is a scheduling bug, isolated from batched-GEMM numerics. bo1 covers
        # the plain in-game case; the bo3-sb case covers the mode split
        # (sideboard roots run the plan search, untouched by --cross-world,
        # while in-game roots run cross) with the boundary memo live.
        for tag, kw in (("xw-uniform-bo1", {}),
                        ("xw-uniform-bo3-sb", dict(bo3=True))):
            d_seq = os.path.join(td, f"visits_{tag}_seq.bin")
            d_xw = os.path.join(td, f"visits_{tag}_xw.bin")
            a_seq = _run_actor(None, d_seq, batch=1, uniform=True, **kw)
            a_xw = _run_actor(None, d_xw, batch=1, uniform=True,
                              cross_world=True, **kw)
            if a_seq is None or a_xw is None:
                return 1
            rc, total_sims = _compare_visits(tag, a_seq, a_xw)
            if rc:
                print(f"FAIL [{tag}]: cross-world scheduler diverged from the "
                      f"sequential search under a uniform evaluator (the "
                      f"'actor'/'python' rows above are sequential/cross-world)",
                      file=sys.stderr)
                return rc
            print(f"PASS [{tag}]: cross-world visits bit-exact vs sequential "
                  f"over {len(a_seq)} searched roots ({total_sims} total root "
                  f"visits, uniform evaluator)")

        # (b) REPORT, real net: cross-world vs batch=1 differ only by the
        # batched forward's last-ulp logits, so argmax agreement over the
        # aligned prefix should be ~1.0 (same shape as the batch=16 report;
        # not asserted — ulp flips on near-ties are legitimate).
        dumpxw = os.path.join(td, "visits_xw.bin")
        actorxw = _run_actor(ts_path, dumpxw, batch=1, cross_world=True)
        if actorxw is None:
            print("WARN: --cross-world run failed; skipping agreement report",
                  file=sys.stderr)
            return 1
        prefix = 0
        for (r1, nc1, _), (rx, ncx, _) in zip(actor1, actorxw):
            if r1 != rx or nc1 != ncx:
                break
            prefix += 1
        agree = sum(1 for i in range(prefix)
                    if int(np.argmax(actor1[i][2])) == int(np.argmax(actorxw[i][2])))
        frac = agree / prefix if prefix else 1.0
        diverged = "" if prefix == len(actor1) else (
            f"; games diverge after root {prefix} "
            f"(batch=1 {len(actor1)} roots, cross-world {len(actorxw)} roots)")
        print(f"REPORT: cross-world vs batch=1 (net) argmax agreement over "
              f"{prefix} comparable roots = {agree}/{prefix} = {frac:.3f}{diverged}")

        # 7) Stage C central inference server.
        rc = _server_legs(td, ts_path, actor1)
        if rc:
            return rc

        # 8) Two-model gate legs (az_actor --model-b — the az_eval gate backend).
        rc = _gate_legs(td, ts_path, actor1)
        if rc:
            return rc

        # ── Vs-scripted parity (actor + scripted oracle vs the Python
        # reference). Both sides run the IDENTICAL scripted_agent code on
        # bit-identical observations (obs builder parity), so these are EXACT
        # gates like bo1 — under the real net too, since only the net seat
        # searches and its evaluator is the same TorchScript graph on both
        # sides. Cross-deck matchup so the agent's deck-aware heuristics see
        # real deck names. The composition leg then pins cross-world + scripted
        # (actor vs actor, uniform, no Python needed).
        import az_selfplay
        oracle_proc, oracle_sock, oracle_dir = az_selfplay._spawn_oracle()
        try:
            for tag, uni in (("scripted-uniform", True), ("scripted-net", False)):
                d_scr = os.path.join(td, f"visits_{tag}.bin")
                a_scr = _run_actor(None if uni else ts_path, d_scr, batch=1,
                                   uniform=uni, deck_b=SCRIPTED_DECK_B,
                                   scripted_sock=oracle_sock)
                if a_scr is None:
                    return 1
                py_scr = _python_reference_scripted(ts_path, SCRIPTED_DECK_B,
                                                    uniform=uni)
                rc, total_sims = _compare_visits(tag, a_scr, py_scr)
                if rc:
                    return rc
                print(f"PASS [{tag}]: vs-scripted visit parity exact over "
                      f"{len(a_scr)} searched roots ({total_sims} total root "
                      f"visits) [net seat A, scripted:hard seat B via oracle, "
                      f"{DECK} vs {SCRIPTED_DECK_B}]")
            d_sx = os.path.join(td, "visits_scripted_xw.bin")
            a_sx = _run_actor(None, d_sx, batch=1, uniform=True,
                              deck_b=SCRIPTED_DECK_B, scripted_sock=oracle_sock,
                              cross_world=True)
            if a_sx is None:
                return 1
            d_ss = os.path.join(td, "visits_scripted_seq.bin")
            a_ss = _run_actor(None, d_ss, batch=1, uniform=True,
                              deck_b=SCRIPTED_DECK_B, scripted_sock=oracle_sock)
            if a_ss is None:
                return 1
            rc, total_sims = _compare_visits("scripted-xw-uniform", a_ss, a_sx)
            if rc:
                print("FAIL [scripted-xw-uniform]: cross-world + scripted-seat "
                      "composition diverged from the sequential scheduler "
                      "(the 'actor'/'python' rows above are sequential/"
                      "cross-world)", file=sys.stderr)
                return rc
            print(f"PASS [scripted-xw-uniform]: cross-world + scripted seat "
                  f"bit-exact vs sequential over {len(a_ss)} searched roots "
                  f"({total_sims} total root visits, uniform evaluator)")
        finally:
            import shutil
            oracle_proc.terminate()
            oracle_proc.wait()
            shutil.rmtree(oracle_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
