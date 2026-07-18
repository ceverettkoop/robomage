"""Determinized PUCT tree search over the engine's snapshot protocol.

AlphaZero-style MCTS for an imperfect-information game: at a (loop-safe) real
decision, the root state is snapshotted once, then K "worlds" are sampled by
DETERMINIZE (reshuffling the hidden zones with a world-local seed). Each world
gets its own tree — action indices at deeper nodes are only meaningful within
one world's deterministic future — and only the ROOT visit counts are summed
across worlds (root-menu identity across worlds is guaranteed: determinize
never touches the root player's own hand/board, and the engine re-derives the
same menu). Each simulation is: RESTORE root -> DETERMINIZE(world seed) ->
replay the node path with sim_step -> expand/evaluate the leaf -> back up.

Values are backed up in each node's OWN mover perspective (the engine already
serializes every observation from the priority player's view, so the evaluator
returns "good for whoever is to move"); selection therefore always maximizes,
and a value crossing a seat boundary flips sign. Terminal SIM_RESULTs back up
exact +/-1.

The evaluator is pluggable so the same search serves Phase B (PPO policy/value
heads), Phase C (AZNet), and torch-free testing (UniformEvaluator).
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from env import _SELF_IS_A_IDX
from search_env import SearchRoboMageEnv, SimQuery


class Evaluator(Protocol):
    """Maps one observation to (priors over the legal menu, value).

    priors: float array of length num_choices summing to 1.
    value: in [-1, 1], from the perspective of the player to move in `obs`
    (the engine serializes obs from the priority player's view).
    """

    def evaluate(self, obs: np.ndarray, num_choices: int) -> tuple[np.ndarray, float]:
        ...


class UniformEvaluator:
    """Torch-free evaluator: uniform priors, neutral value. Search degrades to
    visit-driven exploration with terminal-only signal — used by tests and as
    the fallback when no model is available."""

    def evaluate(self, obs: np.ndarray, num_choices: int) -> tuple[np.ndarray, float]:
        return np.full(num_choices, 1.0 / num_choices, dtype=np.float64), 0.0


class PPOEvaluator:
    """Priors/values from a MaskablePPO checkpoint's policy and value heads.

    The PPO value head was trained on shaped, learner-side returns — it is a
    heuristic here, squashed into [-1, 1] with tanh(v / v_scale) rather than
    trusted as a calibrated game-outcome estimate.
    """

    def __init__(self, model, v_scale: float = 1.0):
        import torch  # local: keep mcts.py importable without torch

        self._torch = torch
        self._model = model
        self._v_scale = float(v_scale)
        from _enums import MAX_ACTIONS

        self._max_actions = MAX_ACTIONS
        # Inference-only path: torch validates every distribution construction,
        # and MaskableCategorical.apply_masking re-validates the cached pre-mask
        # probs, whose fp32 softmax sum occasionally lands just outside the
        # Simplex tolerance (~1e-6). Search builds thousands of distributions
        # per decision, making that spurious ValueError near-certain over a
        # match — disable the validation here.
        torch.distributions.Distribution.set_default_validate_args(False)

    def evaluate(self, obs: np.ndarray, num_choices: int) -> tuple[np.ndarray, float]:
        torch = self._torch
        policy = self._model.policy
        mask = np.zeros(self._max_actions, dtype=bool)
        mask[:num_choices] = True
        with torch.no_grad():
            obs_t, _ = policy.obs_to_tensor(obs)
            try:
                dist = policy.get_distribution(obs_t, action_masks=mask)
                probs = dist.distribution.probs.detach().cpu().numpy()[0][:num_choices]
            except ValueError:
                # Belt-and-braces for the same Simplex flake (e.g. a caller
                # re-enabled validation): one uniform prior among thousands of
                # sims is harmless.
                probs = np.full(num_choices, 1.0 / num_choices)
            value = float(policy.predict_values(obs_t).item())
        total = probs.sum()
        if not np.isfinite(total) or total <= 0.0:
            probs = np.full(num_choices, 1.0 / num_choices)
            total = 1.0
        return probs.astype(np.float64) / total, float(np.tanh(value / self._v_scale))


@dataclass
class SearchResult:
    visits: np.ndarray          # summed root visit counts across worlds
    priors: np.ndarray          # root priors (pre-noise), for diagnostics
    root_value: float           # visit-weighted root Q (root mover perspective)
    num_choices: int
    sims_run: int
    sim_steps: int              # total engine sim_step calls (cost metric)
    stopped_early: bool = False  # timed search ended on the stability check
    # Diagnostics for analysis consumers (populated by run_search /
    # run_search_parallel; default None keeps old constructors/readers valid).
    q: Optional[np.ndarray] = None            # per-root-action Q = ΣW/ΣN across
    #                                           worlds, root-mover perspective
    #                                           (0 where unvisited)
    w_sum: Optional[np.ndarray] = None        # summed root W across worlds (the
    #                                           numerator of q; lets a merge
    #                                           recompute q exactly)
    world_values: Optional[np.ndarray] = None  # (worlds,) per-world root value
    #                                            ΣW/ΣN (0 for an unvisited world)

    def best_action(self) -> int:
        return int(np.argmax(self.visits))

    def policy_target(self, temperature: float = 1.0) -> np.ndarray:
        """Visit distribution over the root menu (the AZ training target)."""
        v = self.visits.astype(np.float64)
        if temperature <= 1e-6:
            out = np.zeros_like(v)
            out[int(np.argmax(v))] = 1.0
            return out
        v = v ** (1.0 / temperature)
        return v / v.sum()


class _Node:
    __slots__ = ("num_choices", "P", "N", "W", "children", "self_is_a")

    def __init__(self, num_choices: int, priors: np.ndarray, self_is_a: bool):
        self.num_choices = num_choices
        self.P = priors
        self.N = np.zeros(num_choices, dtype=np.int64)
        self.W = np.zeros(num_choices, dtype=np.float64)
        self.children: dict[int, _Node] = {}
        self.self_is_a = self_is_a  # which seat moves at this node

    def select(self, c_puct: float) -> int:
        # Q is stored in this node's mover perspective, so plain argmax is
        # correct at both seats. Unvisited actions take Q=0 (neutral FPU).
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), 0.0)
        u = c_puct * self.P * (np.sqrt(1.0 + self.N.sum()) / (1.0 + self.N))
        return int(np.argmax(q + u))


# Stability early-stop tuning (timed searches with a min deadline only).
_STAB_CHECK_EVERY = 32   # sims between stability checks
_STAB_SHARE = 0.6        # global top-action visit share required to stop
_STAB_CLOSE = 0.8        # top2/top1 visit ratio above which the root is contested
_STAB_Q_MIN_FRAC = 0.25  # actions with >= this fraction of top visits are Q candidates


def _stability_stop(roots: "list[_Node]", prev_top: int) -> tuple[bool, int]:
    """Decide whether a timed search past its min deadline has converged.

    Sums the live per-world root visit/value arrays into the GLOBAL root
    statistics and stops only when the top action dominates (visit share above
    ``_STAB_SHARE``), the runner-up is not close (``_STAB_CLOSE``), the
    best-by-Q among reasonably-visited candidates agrees with best-by-visits,
    and the argmax was the same on the previous check (``prev_top``). Returns
    ``(stop, top)`` so the caller threads ``top`` into the next check. Pure
    numpy, draws no rng."""
    N = roots[0].N.copy()
    W = roots[0].W.copy()
    for r in roots[1:]:
        N += r.N
        W += r.W
    total = int(N.sum())
    if total <= 0 or N.size < 2:
        return False, -1
    order = np.argsort(N)
    top = int(order[-1])
    if N[top] / total <= _STAB_SHARE:
        return False, top
    if N[order[-2]] >= _STAB_CLOSE * N[top]:
        return False, top
    # Low-visit Q estimates are noise; only well-visited actions may veto.
    cand = N >= max(1, int(_STAB_Q_MIN_FRAC * N[top]))
    q = np.where(cand, W / np.maximum(N, 1), -np.inf)
    if int(np.argmax(q)) != top:
        return False, top
    return top == prev_top, top


def run_search(
    env: SearchRoboMageEnv,
    evaluator: Evaluator,
    *,
    sims: int = 128,
    worlds: int = 4,
    c_puct: float = 1.5,
    max_depth: int = 60,
    root_noise_eps: float = 0.0,
    root_noise_alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    snapshot_slot: int = 0,
    world_seeds: Optional[Sequence[int]] = None,
    time_budget_s: Optional[float] = None,
    time_budget_min_s: Optional[float] = None,
) -> SearchResult:
    """Search the env's current decision. The env must be parked at a real,
    loop-safe decision (env.last_search_safe). On return the env is back at
    that same decision with all snapshots released, ready for the chosen real
    step().

    ``world_seeds`` (optional): when given, the per-world determinize seeds are
    consumed from it in order instead of drawn from ``rng`` — used for
    reproducible cross-implementation parity (the C++ actor derives the same
    seeds by a shared formula). Default ``None`` keeps the original rng draw.

    ``time_budget_s`` (optional): a WALL-CLOCK budget in seconds. When ``None``
    (the default) the search runs the fixed ``sims`` count via the original
    per-world sequential loop — this path is byte-for-byte unchanged and is what
    the parity corpus / self-play depend on, so do NOT alter its iteration order.
    When set, the deadline is the terminator: worlds' root nodes are built up
    front (identical priors / world-seed derivation), then simulations run
    ROUND-ROBIN across worlds until the budget elapses, so a timed cutoff never
    biases visits toward whichever world ran first. A floor of one simulation per
    world always runs (even under a tiny budget, so every determinized deal is
    looked at). ``sims`` then acts as an OPTIONAL hard cap: when ``sims > 0`` the
    round-robin also stops at ``sims`` total simulations (``min(cap, deadline)``);
    pass ``sims<=0`` to let the clock alone terminate.

    ``time_budget_min_s`` (optional, timed searches only): a MIN deadline that
    arms a stability early-stop. Once it has elapsed, every
    ``_STAB_CHECK_EVERY`` sims the global root visits are checked
    (:func:`_stability_stop`); a dominant, stable, value-consistent top action
    ends the search before ``time_budget_s`` and marks the result
    ``stopped_early``. ``None`` (the default) keeps the timed loop's existing
    semantics; the fixed-budget path ignores it entirely."""
    rng = rng if rng is not None else np.random.default_rng()
    if world_seeds is not None and len(world_seeds) < worlds:
        raise ValueError(
            f"world_seeds needs >= {worlds} entries, got {len(world_seeds)}")

    root_obs = env._obs.copy()
    root_n = env._num_choices
    root_is_a = bool(root_obs[_SELF_IS_A_IDX] > 0.5)
    root_priors, _ = evaluator.evaluate(root_obs, root_n)

    env.snapshot(snapshot_slot)

    visit_totals = np.zeros(root_n, dtype=np.float64)
    value_acc = 0.0
    sims_run = 0
    sim_steps = 0
    sims_per_world = max(1, sims // max(1, worlds))

    # Derive per-world seeds and root nodes up front. The derivation order
    # (world_seed then optional dirichlet noise, for w=0..worlds-1) is identical
    # to the historical inline loop, so the ``rng`` consumption sequence is
    # unchanged — the fixed-budget path below stays bit-exact.
    seeds: list[int] = []
    roots: list[_Node] = []
    for w in range(worlds):
        world_seed = (int(world_seeds[w]) if world_seeds is not None
                      else int(rng.integers(1, 2**31 - 1)))
        priors = root_priors
        if root_noise_eps > 0.0:
            noise = rng.dirichlet([root_noise_alpha] * root_n)
            priors = (1.0 - root_noise_eps) * root_priors + root_noise_eps * noise
        seeds.append(world_seed)
        roots.append(_Node(root_n, priors, root_is_a))

    def _one_sim(w: int) -> None:
        nonlocal sims_run, sim_steps
        env.restore(snapshot_slot)
        env.determinize(seeds[w])
        v = _simulate(env, evaluator, roots[w], c_puct, max_depth)
        sims_run += 1
        sim_steps += v[1]

    stopped_early = False
    if time_budget_s is None:
        # UNCHANGED sequential per-world loop (parity corpus / self-play depend
        # on this exact order and count).
        for w in range(worlds):
            for _ in range(sims_per_world):
                _one_sim(w)
    else:
        deadline = time.monotonic() + float(time_budget_s)
        min_deadline = None
        if time_budget_min_s is not None:
            min_deadline = time.monotonic() + min(
                float(time_budget_min_s), float(time_budget_s))
        cap = sims if sims and sims > 0 else None
        # Floor: one sim per world, unconditionally.
        for w in range(worlds):
            _one_sim(w)
        i = 0
        since_check = 0
        prev_top = -1
        while time.monotonic() < deadline:
            if cap is not None and sims_run >= cap:
                break
            if (min_deadline is not None and since_check >= _STAB_CHECK_EVERY
                    and time.monotonic() >= min_deadline):
                since_check = 0
                stop, prev_top = _stability_stop(roots, prev_top)
                if stop:
                    stopped_early = True
                    break
            _one_sim(i % worlds)
            i += 1
            since_check += 1

    w_totals = np.zeros(root_n, dtype=np.float64)
    world_values = np.zeros(len(roots), dtype=np.float64)
    for w, root in enumerate(roots):
        visit_totals += root.N
        w_totals += root.W
        value_acc += float(root.W.sum())
        n_w = int(root.N.sum())
        world_values[w] = float(root.W.sum()) / n_w if n_w > 0 else 0.0

    # Back to the true root state; drop snapshots BEFORE the caller's real
    # step — a real game-end with a live snapshot parks the engine in the
    # SIM_RESULT intercept, which the plain step() reader cannot leave.
    env.restore(snapshot_slot)
    env.release()

    total_n = visit_totals.sum()
    root_value = value_acc / total_n if total_n > 0 else 0.0
    return SearchResult(
        visits=visit_totals,
        priors=root_priors,
        root_value=root_value,
        num_choices=root_n,
        sims_run=sims_run,
        sim_steps=sim_steps,
        stopped_early=stopped_early,
        q=np.where(visit_totals > 0, w_totals / np.maximum(visit_totals, 1), 0.0),
        w_sum=w_totals,
        world_values=world_values,
    )


class _LockedEvaluator:
    """Serializes an evaluator's calls under a shared lock so the parallel search
    can share one (possibly non-thread-safe, e.g. torch-backed) evaluator across
    worker threads. Engine stepping (~6ms/sim) dominates the cheap net eval, so a
    lock on ``evaluate`` is not a bottleneck."""

    def __init__(self, evaluator: Evaluator, lock: threading.Lock):
        self._evaluator = evaluator
        self._lock = lock

    def evaluate(self, obs: np.ndarray, num_choices: int) -> tuple[np.ndarray, float]:
        with self._lock:
            return self._evaluator.evaluate(obs, num_choices)


def run_search_parallel(
    envs: Sequence[SearchRoboMageEnv],
    evaluator: Evaluator,
    *,
    sims: int = 128,
    worlds: int = 4,
    c_puct: float = 1.5,
    max_depth: int = 60,
    root_noise_eps: float = 0.0,
    root_noise_alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    time_budget_s: Optional[float] = None,
    time_budget_min_s: Optional[float] = None,
) -> SearchResult:
    """World-parallel :func:`run_search` for INTERACTIVE search only.

    The ``worlds`` determinized worlds of a search are fully independent (own root
    node, own seed, own per-sim restore+determinize), so they split cleanly across
    the ``envs`` list — the primary plus its lockstep mirrors (see
    :meth:`SearchRoboMageEnv.search_envs`). Each env runs its slice of worlds via
    an ordinary ``run_search`` on its own snapshot slot 0, in a thread (each thread
    mostly blocks on its engine pipe, so the GIL isn't the bottleneck). The
    per-world sim counts and world seeds match a single-process run exactly, and
    the root visit counts are summed across envs — so with one env this is
    bit-identical to :func:`run_search`, and with N it just fans the same worlds
    out concurrently.

    ``run_search`` itself is untouched (self-play / parity corpus depend on it).
    This is an INFERENCE-only path: root dirichlet noise is unsupported
    (``root_noise_eps`` must be 0) because injecting it per world would consume the
    rng in a thread-order-dependent way. All ``worlds`` world seeds are pre-drawn
    up front with the exact derivation ``run_search`` uses, so a 1-env call
    consumes ``rng`` identically to a plain ``run_search``.

    ``time_budget_min_s`` is forwarded per env, so each worker's stability
    early-stop (see :func:`run_search`) sees only its OWN world slice — a
    noisier per-slice approximation of the global signal (the slices are i.i.d.
    determinizations, and the timed path is already wall-clock
    nondeterministic). Each worker's min deadline starts at its own thread
    start; the workers launch near-simultaneously, so the skew just shifts the
    window."""
    envs = list(envs)
    if len(envs) <= 1:
        # Single engine: delegate straight through, identical behavior (the seed
        # draw happens inside run_search from the same rng).
        return run_search(
            envs[0], evaluator, sims=sims, worlds=worlds, c_puct=c_puct,
            max_depth=max_depth, root_noise_eps=root_noise_eps,
            root_noise_alpha=root_noise_alpha, rng=rng, time_budget_s=time_budget_s,
            time_budget_min_s=time_budget_min_s)

    assert root_noise_eps == 0.0, (
        "run_search_parallel is inference-only: root dirichlet noise would consume "
        "the rng per world in a thread-order-dependent way")

    rng = rng if rng is not None else np.random.default_rng()
    # Pre-draw every world seed with the SAME derivation run_search uses, so a
    # 1-env call would consume the rng identically to plain run_search.
    world_seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(worlds)]

    n_envs = min(len(envs), worlds)
    usable = envs[:n_envs]
    sims_per_world = max(1, sims // max(1, worlds))
    cap = sims if sims and sims > 0 else None

    # Contiguous world split across the usable envs (spread the remainder over the
    # first few envs). Each env owns world_seeds[lo:hi] on its own slot 0.
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n_envs):
        k_i = worlds // n_envs + (1 if i < worlds % n_envs else 0)
        bounds.append((start, start + k_i))
        start += k_i

    lock = threading.Lock()
    locked = _LockedEvaluator(evaluator, lock)

    def _one(i: int) -> SearchResult:
        lo, hi = bounds[i]
        k_i = hi - lo
        seeds_i = world_seeds[lo:hi]
        if time_budget_s is None:
            # Fixed budget: sims_per_world per world -> run_search re-derives the
            # same per-world count from sims_i // k_i.
            sims_i = sims_per_world * k_i
        elif cap is not None:
            # Timed with a hard cap: split the cap proportionally (>= 1/world).
            sims_i = max(cap * k_i // worlds, k_i)
        else:
            sims_i = 0  # clock alone terminates
        return run_search(
            usable[i], locked, sims=sims_i, worlds=k_i, c_puct=c_puct,
            max_depth=max_depth, root_noise_eps=0.0, rng=None,
            world_seeds=seeds_i, time_budget_s=time_budget_s,
            time_budget_min_s=time_budget_min_s)

    results: list[Optional[SearchResult]] = [None] * n_envs
    errors: list[tuple[int, Exception]] = []
    with ThreadPoolExecutor(max_workers=n_envs) as ex:
        futures = {ex.submit(_one, i): i for i in range(n_envs)}
        # as_completed lets a failing worker be noticed, but we still wait for all
        # (the with-block join) before re-raising — never leave an engine
        # mid-search.
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # noqa: BLE001 — surfaced below after join
                errors.append((i, e))
    if errors:
        i, e = errors[0]
        raise RuntimeError(f"run_search_parallel: env {i} search failed: {e}") from e

    # Deterministic merge in env order: visits sum, root_value visit-weighted.
    # Env order == world order (each env owns the contiguous world_seeds[lo:hi]
    # slice), so concatenating world_values in env order preserves world index.
    root_n = results[0].num_choices
    visits = np.zeros(root_n, dtype=np.float64)
    w_sum = np.zeros(root_n, dtype=np.float64)
    weighted_value = 0.0
    total_vis = 0.0
    sims_run = 0
    sim_steps = 0
    for r in results:
        visits += r.visits
        w_sum += r.w_sum
        vs = float(r.visits.sum())
        weighted_value += r.root_value * vs
        total_vis += vs
        sims_run += r.sims_run
        sim_steps += r.sim_steps
    root_value = weighted_value / total_vis if total_vis > 0 else 0.0
    return SearchResult(
        visits=visits,
        priors=results[0].priors,
        root_value=root_value,
        num_choices=root_n,
        sims_run=sims_run,
        sim_steps=sim_steps,
        stopped_early=any(r.stopped_early for r in results),
        q=np.where(visits > 0, w_sum / np.maximum(visits, 1), 0.0),
        w_sum=w_sum,
        world_values=np.concatenate([r.world_values for r in results]),
    )


def _simulate(
    env: SearchRoboMageEnv,
    evaluator: Evaluator,
    root: _Node,
    c_puct: float,
    max_depth: int,
) -> tuple[float, int]:
    """One PUCT descent from the (already restored+determinized) root.
    Returns (leaf value in root-node mover perspective, engine steps used)."""
    node = root
    path: list[tuple[_Node, int]] = []
    steps = 0
    leaf_value = 0.0       # in the perspective of `leaf_seat_is_a`
    leaf_seat_is_a = root.self_is_a

    while True:
        action = node.select(c_puct)
        query: SimQuery = env.sim_step(action)
        steps += 1
        path.append((node, action))

        if query.terminal is not None:
            if query.terminal == "DRAW":
                leaf_value, leaf_seat_is_a = 0.0, root.self_is_a
            else:
                leaf_seat_is_a = root.self_is_a
                won = (query.terminal == "A") == root.self_is_a
                leaf_value = 1.0 if won else -1.0
            break

        child = node.children.get(action)
        if child is None:
            child_is_a = bool(query.obs[_SELF_IS_A_IDX] > 0.5)
            priors, value = evaluator.evaluate(query.obs, query.num_choices)
            node.children[action] = _Node(query.num_choices, priors, child_is_a)
            leaf_value, leaf_seat_is_a = value, child_is_a
            break

        # Same world + same path must re-derive the same menu; a mismatch means
        # the engine's determinism (or our world bookkeeping) broke.
        if child.num_choices != query.num_choices:
            raise RuntimeError(
                f"world-consistency violation: node expected {child.num_choices} "
                f"choices, engine gave {query.num_choices}")
        node = child

        if len(path) >= max_depth:
            _, value = evaluator.evaluate(query.obs, query.num_choices)
            leaf_value, leaf_seat_is_a = value, node.self_is_a
            break

    for parent, action in path:
        parent.N[action] += 1
        parent.W[action] += leaf_value if parent.self_is_a == leaf_seat_is_a else -leaf_value
    return leaf_value, steps


# ── Incremental (chunked) search for interactive analysis ─────────────────────

@dataclass
class LiveStats:
    """A point-in-time view of an IncrementalSearch's root statistics. All
    arrays are fresh copies (safe to hand across threads); values are in the
    ROOT mover's perspective, in [-1, 1]."""
    num_choices: int
    sims_run: int
    sim_steps: int
    visits: np.ndarray          # (n,) summed root visit counts across worlds
    priors: np.ndarray          # (n,) root priors (no noise)
    q: np.ndarray               # (n,) per-action Q = ΣW/ΣN (0 where unvisited)
    root_value: float           # visit-weighted root Q
    net_value: float            # evaluator's raw value at the root ("depth 0")
    world_values: np.ndarray    # (worlds,) per-world root value
    world_visits: np.ndarray    # (worlds, n) per-world root visit counts


@dataclass
class PVStep:
    """One step of a principal variation read out of a world's tree."""
    action: int                 # env-index action taken from this node
    visits: int                 # N of that action at this node
    q: float                    # the action's Q converted to ROOT-mover perspective
    seat_is_a: bool             # which seat moves at the node the action leaves


@dataclass
class WalkNode:
    """One engine state along a walked line (for rendering hypothetical
    positions). obs is a private copy; terminal is None for a decision, else
    "A" / "B" / "DRAW" (with obs None)."""
    obs: Optional[np.ndarray]
    num_choices: int
    pending_confirm: bool
    terminal: Optional[str]


class IncrementalSearch:
    """A run_search split into caller-paced chunks, for live analysis display.

    Same tree machinery (_Node/_simulate), same per-world seed derivation, and
    the same per-sim restore+determinize discipline as :func:`run_search` —
    worlds are independent trees, so running their sims round-robin in chunks
    yields bit-identical trees to run_search's sequential per-world loop once
    each world has received the same sim count (pin ``world_seeds`` and give
    both the same totals to compare). No root dirichlet noise (inference only).

    Unlike run_search, the root SNAPSHOT is held OPEN across chunks so the
    search can resume, and :meth:`pv`/:meth:`walk` can browse the tree and
    replay hypothetical lines after the last chunk. The owner MUST call
    :meth:`close` (restore + release) before the env takes any real step —
    the same driver discipline as every snapshot consumer.
    """

    def __init__(self, env: SearchRoboMageEnv, evaluator: Evaluator, *,
                 worlds: int = 4, c_puct: float = 1.5, max_depth: int = 60,
                 rng: Optional[np.random.Generator] = None,
                 snapshot_slot: int = 0,
                 world_seeds: Optional[Sequence[int]] = None):
        if world_seeds is not None and len(world_seeds) < worlds:
            raise ValueError(
                f"world_seeds needs >= {worlds} entries, got {len(world_seeds)}")
        rng = rng if rng is not None else np.random.default_rng()
        self._env = env
        self._evaluator = evaluator
        self._c_puct = c_puct
        self._max_depth = max_depth
        self._slot = snapshot_slot
        self._worlds = worlds
        self.root_obs = env._obs.copy()
        self.num_choices = env._num_choices
        self.root_is_a = bool(self.root_obs[_SELF_IS_A_IDX] > 0.5)
        priors, net_value = evaluator.evaluate(self.root_obs, self.num_choices)
        self.priors = priors
        self.net_value = float(net_value)
        env.snapshot(snapshot_slot)
        self.seeds: list[int] = []
        self.roots: list[_Node] = []
        for w in range(worlds):
            seed = (int(world_seeds[w]) if world_seeds is not None
                    else int(rng.integers(1, 2**31 - 1)))
            self.seeds.append(seed)
            self.roots.append(_Node(self.num_choices, priors, self.root_is_a))
        self._next_world = 0
        self.sims_run = 0
        self.sim_steps = 0
        self._closed = False

    def run_chunk(self, n_sims: int) -> LiveStats:
        """Run ``n_sims`` more simulations (round-robin across worlds, resuming
        where the previous chunk stopped) and return the updated stats."""
        if self._closed:
            raise RuntimeError("IncrementalSearch already closed")
        env = self._env
        for _ in range(max(0, int(n_sims))):
            w = self._next_world
            self._next_world = (w + 1) % self._worlds
            env.restore(self._slot)
            env.determinize(self.seeds[w])
            _, steps = _simulate(env, self._evaluator, self.roots[w],
                                 self._c_puct, self._max_depth)
            self.sims_run += 1
            self.sim_steps += steps
        return self.stats()

    def stats(self) -> LiveStats:
        n = self.num_choices
        visits = np.zeros(n, dtype=np.float64)
        w_sum = np.zeros(n, dtype=np.float64)
        world_values = np.zeros(self._worlds, dtype=np.float64)
        world_visits = np.zeros((self._worlds, n), dtype=np.int64)
        for w, root in enumerate(self.roots):
            visits += root.N
            w_sum += root.W
            world_visits[w] = root.N
            n_w = int(root.N.sum())
            world_values[w] = float(root.W.sum()) / n_w if n_w > 0 else 0.0
        total = visits.sum()
        return LiveStats(
            num_choices=n,
            sims_run=self.sims_run,
            sim_steps=self.sim_steps,
            visits=visits,
            priors=np.array(self.priors, dtype=np.float64, copy=True),
            q=np.where(visits > 0, w_sum / np.maximum(visits, 1), 0.0),
            root_value=float(w_sum.sum()) / total if total > 0 else 0.0,
            net_value=self.net_value,
            world_values=world_values,
            world_visits=world_visits,
        )

    def result(self) -> SearchResult:
        """The current stats as a plain SearchResult (run_search-compatible)."""
        s = self.stats()
        return SearchResult(
            visits=s.visits, priors=s.priors, root_value=s.root_value,
            num_choices=s.num_choices, sims_run=s.sims_run,
            sim_steps=s.sim_steps, q=s.q,
            w_sum=np.where(s.visits > 0, s.q * s.visits, 0.0),
            world_values=s.world_values)

    def pv(self, action: int, world: int, max_len: int = 24) -> list:
        """The principal variation for taking ``action`` at the root of
        ``world``'s tree: the root action, then argmax-visits descent. Stops at
        an unexpanded/unvisited node. Empty if the root action was never
        visited in that world. q values are in the ROOT mover's perspective."""
        root = self.roots[world]
        out: list[PVStep] = []
        node = root
        a = int(action)
        for _ in range(max(0, int(max_len))):
            n_vis = int(node.N[a])
            if n_vis <= 0:
                break
            q_own = float(node.W[a]) / n_vis   # in `node`'s mover perspective
            q_root = q_own if node.self_is_a == root.self_is_a else -q_own
            out.append(PVStep(action=a, visits=n_vis, q=q_root,
                              seat_is_a=node.self_is_a))
            child = node.children.get(a)
            if child is None or int(child.N.sum()) <= 0:
                break
            node = child
            a = int(np.argmax(node.N))
        return out

    def walk(self, world: int, actions: Sequence[int]) -> list:
        """Replay ``actions`` from the root inside ``world``'s determinization,
        capturing each resulting engine state (obs copies) — for rendering the
        hypothetical positions along a PV. Ends by restoring the root snapshot,
        so the search remains resumable afterwards. Stops early at a simulated
        game end (the final WalkNode carries the terminal result)."""
        if self._closed:
            raise RuntimeError("IncrementalSearch already closed")
        env = self._env
        env.restore(self._slot)
        env.determinize(self.seeds[world])
        out: list[WalkNode] = []
        for a in actions:
            query: SimQuery = env.sim_step(int(a))
            if query.terminal is not None:
                out.append(WalkNode(obs=None, num_choices=0,
                                    pending_confirm=False,
                                    terminal=query.terminal))
                break
            out.append(WalkNode(obs=query.obs.copy(),
                                num_choices=query.num_choices,
                                pending_confirm=query.pending_confirm,
                                terminal=None))
        env.restore(self._slot)
        return out

    def close(self) -> None:
        """Restore the real root state and drop the snapshot. Idempotent. Must
        run before the env's next real step()."""
        if self._closed:
            return
        self._closed = True
        self._env.restore(self._slot)
        self._env.release()
