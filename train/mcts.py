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

from _enums import CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT
from decode import menu_merge_reps
from env import (_CUR_TURN_IDX, _IS_SIDEBOARD_IDX, _MATCH_CTX_START,
                 _SELF_IS_A_IDX, _obs_action_category, _obs_action_card_id)
from search_env import SearchRoboMageEnv, SimQuery

# Leaf-rollout step cap, per horizon turn: a rollout may take at most
# ROLLOUT_STEPS_PER_TURN * rollout_turns engine decisions before it is cut off
# and evaluated where it stands (measured: reaching turn 12 takes 46-92
# decisions across league matchups, so 40/turn is a comfortable bound).
# MIRRORED as kRolloutStepsPerTurn in src/actor/az_mcts.cpp — the C++ actor
# must cut off at the same step or visit parity breaks.
ROLLOUT_STEPS_PER_TURN = 40


class Evaluator(Protocol):
    """Maps one observation to (priors over the legal menu, value).

    priors: float array of length num_choices summing to 1.
    value: in [-1, 1], from the perspective of the player to move in `obs`
    (the engine serializes obs from the priority player's view).

    An evaluator MAY additionally provide ``evaluate_batch(obs_batch,
    num_choices_list) -> list[(priors, value)]`` — one forward over k rows,
    each row's priors/value computed exactly as ``evaluate`` would. The
    cross-world batched search uses it when present (``az_net.AZEvaluator``
    implements it; this is where a GPU device pays — see
    docs/gpu_selfplay_inference_plan.md); evaluators without it are called
    row-by-row, which keeps torch-free/uniform evaluators BIT-IDENTICAL to
    the sequential search under cross-world scheduling.
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
    roots: Optional[list] = None              # per-world root _Node trees, world
    #                                           order. Kept alive for tree reuse
    #                                           (SearchController follows the
    #                                           real game down these trees while
    #                                           it matches the searched line).
    seeds: Optional[list] = None              # per-world determinize seeds the
    #                                           search actually used, world order
    #                                           (drawn or passed through) — the
    #                                           handle a sideboard boundary
    #                                           latches to pin its worlds.
    reused_visits: int = 0                    # Σ over worlds of root N.sum() at
    #                                           search ENTRY (reuse_roots); the
    #                                           reported visits are cumulative,
    #                                           sims_run counts new sims only.
    memo_hits: int = 0                        # rollout-memo hits this search

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


def sample_visits(visits: np.ndarray, temperature: float,
                  rng: np.random.Generator) -> int:
    """Pick a root action from a visit vector under ``temperature``.

    ``temperature <= 1e-6`` is the greedy branch (argmax, consumes NO rng);
    otherwise the visits are raised to ``1/temperature``, normalized, and
    sampled with EXACTLY ONE ``rng.choice`` draw. The arithmetic is the same
    power law :meth:`SearchResult.policy_target` builds — callers that need the
    distribution itself keep using that; callers that only need the action use
    this so the rng is consumed identically everywhere."""
    v = np.asarray(visits, dtype=np.float64)
    if temperature <= 1e-6:
        return int(np.argmax(v))
    pi = v ** (1.0 / temperature)
    return int(rng.choice(len(pi), p=pi / pi.sum()))


class _Node:
    __slots__ = ("num_choices", "P", "N", "W", "children", "self_is_a",
                 "pick_meta", "rep", "sel_mask")

    def __init__(self, num_choices: int, priors: np.ndarray, self_is_a: bool,
                 pick_meta: Optional[tuple] = None,
                 rep: Optional[np.ndarray] = None):
        self.num_choices = num_choices
        self.P = priors
        self.N = np.zeros(num_choices, dtype=np.int64)
        self.W = np.zeros(num_choices, dtype=np.float64)
        self.children: dict[int, _Node] = {}
        self.self_is_a = self_is_a  # which seat moves at this node
        # Per-action sideboard-pick descriptors (sb_pick_descriptor or None),
        # captured from this node's obs at CREATION — the descent has no obs in
        # hand when re-selecting from interior nodes (mirrors the C++ actor,
        # whose AWAITING_ROOT phase never builds one). Populated only when the
        # search runs with a rollout memo; None otherwise.
        self.pick_meta = pick_meta
        # Duplicate-edge merge partition (decode.menu_merge_reps): rep[i] is
        # the representative index of action i's duplicate group. None =
        # merging disabled. Set at creation (and re-set on boundary root
        # adoption) via set_rep, which also folds P — visits/Q/children only
        # ever live at representative indices.
        self.rep = None
        self.sel_mask = None
        if rep is not None:
            self.set_rep(rep)

    def set_rep(self, rep: np.ndarray) -> None:
        """Install a merge partition and fold P onto the representatives.

        MUST be called exactly once after every assignment of raw priors to
        ``P`` (creation, and boundary-root adoption where P is overwritten
        with fresh noise-mixed priors): P is copied (the caller may share one
        raw priors array across worlds / with SearchResult.priors) and each
        non-representative's mass is added to its representative (ascending
        index — the C++ actor folds in the same order) then zeroed.
        ``sel_mask`` (-inf at non-reps) makes select() skip merged-away
        actions; without it an unvisited P=0 non-rep would win the argmax at
        val=0 whenever every representative has Q + U < 0."""
        self.rep = rep
        self.P = np.array(self.P, dtype=np.float64, copy=True)
        mask = np.zeros(self.num_choices, dtype=np.float64)
        for i in range(1, self.num_choices):
            r = int(rep[i])
            if r != i:
                self.P[r] += self.P[i]
                self.P[i] = 0.0
                mask[i] = -np.inf
        self.sel_mask = mask

    def select(self, c_puct: float) -> int:
        # Q is stored in this node's mover perspective, so plain argmax is
        # correct at both seats. Unvisited actions take Q=0 (neutral FPU).
        q = np.where(self.N > 0, self.W / np.maximum(self.N, 1), 0.0)
        u = c_puct * self.P * (np.sqrt(1.0 + self.N.sum()) / (1.0 + self.N))
        if self.sel_mask is None:
            return int(np.argmax(q + u))
        return int(np.argmax(q + u + self.sel_mask))


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


def _init_worlds(worlds: int, world_seeds: Optional[Sequence[int]],
                 rng: np.random.Generator, root_n: int,
                 root_priors: np.ndarray, root_is_a: bool, *,
                 root_noise_eps: float = 0.0, root_noise_alpha: float = 1.0,
                 reuse_roots: Optional[Sequence] = None,
                 root_rep: Optional[np.ndarray] = None,
                 root_pm: Optional[tuple] = None,
                 sims_per_world: int = 0):
    """Derive the per-world determinize seeds and root nodes up front.

    BIT CONTRACT — this is the historical inline loop of :func:`run_search`,
    moved verbatim; every caller's rng consumption sequence must stay exactly
    what it was. Per world w = 0..worlds-1, in this order: exactly ONE
    ``rng.integers(1, 2**31 - 1)`` draw unless ``world_seeds`` is given (then
    NO draw at all), followed by exactly ONE ``rng.dirichlet`` draw iff
    ``root_noise_eps > 0.0``. Do not reorder, hoist, or vectorize these draws —
    the fixed-budget search path is pinned bit-for-bit against the C++ actor
    (train/test_mcts_parity.py) and against IncrementalSearch (the chunked and
    drawn-seed parity tests in train/test_analysis_session.py).

    A reused root (boundary persistence) is adopted in place of a fresh _Node:
    its P is overwritten with the fresh CLEAN root priors (copied — the
    fresh-root path shares one priors array across worlds) and noise is
    re-mixed in the same per-world rng position a fresh root would use; its
    N/W/children are kept, and its budget is topped up to ``sims_per_world``
    total visits.

    Returns ``(seeds, roots, budgets, reused_visits)``.
    """
    seeds: list[int] = []
    roots: list[_Node] = []
    budgets: list[int] = []
    reused_visits = 0
    for w in range(worlds):
        world_seed = (int(world_seeds[w]) if world_seeds is not None
                      else int(rng.integers(1, 2**31 - 1)))
        priors = root_priors
        if root_noise_eps > 0.0:
            noise = rng.dirichlet([root_noise_alpha] * root_n)
            priors = (1.0 - root_noise_eps) * root_priors + root_noise_eps * noise
        seeds.append(world_seed)
        reused = reuse_roots[w] if reuse_roots is not None else None
        if reused is not None:
            reused.P = np.array(priors, copy=True)
            # Adoption overwrites P with fresh raw priors, so the merge fold
            # must be re-applied (exactly once) — and the partition is refilled
            # from THIS search's root obs (same menu by world consistency).
            if root_rep is not None:
                reused.set_rep(root_rep)
            else:
                reused.rep = None
                reused.sel_mask = None
            if root_pm is not None and reused.pick_meta is None:
                reused.pick_meta = root_pm
            n_have = int(reused.N.sum())
            reused_visits += n_have
            roots.append(reused)
            budgets.append(max(0, sims_per_world - n_have))
        else:
            roots.append(_Node(root_n, priors, root_is_a, pick_meta=root_pm,
                               rep=root_rep))
            budgets.append(sims_per_world)
    return seeds, roots, budgets, reused_visits


def _aggregate_roots(roots: "list[_Node]", root_n: int, *,
                     want_world_visits: bool = False):
    """Sum the per-world root trees into the global root statistics.

    BIT CONTRACT — ``value_acc`` accumulates ``float(root.W.sum())`` ONE WORLD
    AT A TIME, in world order; that summation order deliberately mirrors the
    C++ actor (src/actor/az_mcts.cpp ~482 / ~953). NEVER replace it with
    ``w_totals.sum()``: the two differ in the last float ulp and
    train/test_mcts_parity.py pins the Python result against the actor's.
    Callers keep their OWN root_value formula (:func:`run_search` uses
    ``value_acc / total``; :meth:`IncrementalSearch.stats` uses
    ``w_sum.sum() / total``) — this helper shares the LOOP only, never the
    formula. See also :func:`_stability_stop`, which keeps its own int64
    mid-search fast path rather than calling this.

    Returns ``(visits, w_totals, world_values, value_acc)``, plus a
    ``(worlds, root_n)`` int64 ``world_visits`` matrix when requested.
    """
    visits = np.zeros(root_n, dtype=np.float64)
    w_totals = np.zeros(root_n, dtype=np.float64)
    world_values = np.zeros(len(roots), dtype=np.float64)
    world_visits = (np.zeros((len(roots), root_n), dtype=np.int64)
                    if want_world_visits else None)
    value_acc = 0.0
    for w, root in enumerate(roots):
        visits += root.N
        w_totals += root.W
        value_acc += float(root.W.sum())
        if world_visits is not None:
            world_visits[w] = root.N
        n_w = int(root.N.sum())
        world_values[w] = float(root.W.sum()) / n_w if n_w > 0 else 0.0
    if want_world_visits:
        return visits, w_totals, world_values, value_acc, world_visits
    return visits, w_totals, world_values, value_acc


def _merge_root_stats(parts: Sequence):
    """Merge several already-aggregated root views (``SearchResult``s from
    :func:`run_search_parallel`'s workers, or ``LiveStats`` from the analysis
    session's engine fleet) into one.

    Duck-typed on ``visits``/``w_sum``/``sims_run``/``sim_steps``; the parts are
    summed in the caller's order (env/engine order == world order at both
    sites), which is what makes the merged floats bit-identical to the
    single-engine run. Returns ``(visits, w_sum, sims_run, sim_steps, q)`` with
    the character-identical q expression both sites used. root_value is
    deliberately NOT computed here — the two sites use different (both pinned)
    formulas and must keep them at the site.
    """
    root_n = parts[0].num_choices
    visits = np.zeros(root_n, dtype=np.float64)
    w_sum = np.zeros(root_n, dtype=np.float64)
    sims_run = 0
    sim_steps = 0
    for p in parts:
        visits += p.visits
        w_sum += p.w_sum
        sims_run += p.sims_run
        sim_steps += p.sim_steps
    q = np.where(visits > 0, w_sum / np.maximum(visits, 1), 0.0)
    return visits, w_sum, sims_run, sim_steps, q


def run_search(
    env: SearchRoboMageEnv,
    evaluator: Evaluator,
    *,
    sims: int = 128,
    worlds: int = 4,
    c_puct: float = 1.5,
    max_depth: int = 60,
    rollout_turns: int = 0,
    root_noise_eps: float = 0.0,
    root_noise_alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    snapshot_slot: int = 0,
    world_seeds: Optional[Sequence[int]] = None,
    reuse_roots: Optional[Sequence] = None,
    rollout_memo: Optional[dict] = None,
    memo_picks: tuple = (),
    time_budget_s: Optional[float] = None,
    time_budget_min_s: Optional[float] = None,
    merge_dupes: bool = True,
    cross_world: bool = False,
) -> SearchResult:
    """Search the env's current decision. The env must be parked at a real,
    loop-safe decision (env.last_search_safe). On return the env is back at
    that same decision with all snapshots released, ready for the chosen real
    step().

    ``cross_world`` (default False — the sequential reference path stays
    byte-identical): run the per-world sims ROUND-ROBIN and defer each freshly
    expanded (or depth-capped) leaf's net evaluation into a shared pending
    batch, flushed in one ``evaluate_batch`` forward before any world's own
    next descent starts — the Python twin of the C++ actor's Stage 0
    scheduler (see the "Cross-world batched leaf evaluation" block below).
    Per-world trees are arithmetically identical to the sequential search;
    only a batched GEMM's last-ulp logits can differ, and an evaluator
    without ``evaluate_batch`` is bit-identical. Ignored (sequential path)
    when the budget has ``rollout_turns > 0`` — a deferred leaf cannot drive
    a playout, the same restriction the actor enforces.

    ``reuse_roots`` (optional, boundary persistence): a per-world list of
    previously-searched ``_Node`` trees (or ``None`` entries) to adopt as this
    search's roots instead of building fresh ones — the caller obtains them by
    walking the previous ``SearchResult.roots`` down the actually-played
    actions (:func:`walk_reuse_root`) and MUST pass the same pinned
    ``world_seeds`` the trees were built under (at a sideboard root the seed IS
    the sampled next-game deal). On the FIXED-budget path (``time_budget_s is
    None``) a reused world only runs ``max(0, sims_per_world -
    existing_visits)`` new sims (top-up); the TIMED path ignores those per-world
    budgets by design — a wall-clock budget keeps growing every world's tree,
    reused or fresh, until the deadline (or the ``sims`` cap / stability stop)
    ends it. Either way the reported ``visits`` are CUMULATIVE (``sims_run``
    counts new sims only; ``reused_visits`` reports the inherited mass).
    ``None`` (the default) keeps every path byte-identical to the pre-reuse
    search.

    ``rollout_memo`` (optional, sideboard boundaries only): the boundary's
    rollout-value memo dict, shared across the boundary's searches;
    ``memo_picks`` are the descriptors (:func:`sb_pick_descriptor`) of the
    REAL picks played since the boundary root. Rollouts from leaves still in
    the sideboard phase are stored/reused keyed on the pinned world seed and
    the order-insensitive pick multiset (paths holding one card in BOTH
    directions excluded — see :func:`rollout_memo_eligible`).
    A hit backs up the stored result with 0 engine steps; ``memo_hits``
    reports the count. ``None`` disables (byte-identical default).

    ``rollout_turns`` (default 0 = off): leaf rollouts. Each freshly expanded
    leaf is played forward argmax-greedily by the raw policy (both seats) to
    the end of player-turn ``anchor + rollout_turns`` — anchor is the ROOT's
    turn for an in-game root, or 0 (of the sampled NEXT game) at a bo3
    sideboard root — and the reached state's net value (or true terminal ±1)
    is backed up instead of the leaf's. Callers default this ON at sideboard
    roots (DEFAULT_SB_ROLLOUT_TURNS). Rollouts draw no rng and add no tree
    nodes, so ``rollout_turns=0`` remains byte-for-byte identical to the
    pre-rollout search — the parity guarantees below are unaffected.

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
    _check_world_seeds(world_seeds, worlds)

    root_obs = env._obs.copy()
    root_n = env._num_choices
    root_is_a = bool(root_obs[_SELF_IS_A_IDX] > 0.5)
    root_priors, _ = evaluator.evaluate(root_obs, root_n)
    rollout_anchor = _rollout_anchor(root_obs)
    root_rep = menu_merge_reps(root_obs, root_n) if merge_dupes else None

    env.snapshot(snapshot_slot)

    sims_run = 0
    sim_steps = 0
    sims_per_world = max(1, sims // max(1, worlds))

    # Derive per-world seeds and root nodes up front. The derivation order
    # (world_seed then optional dirichlet noise, for w=0..worlds-1) is identical
    # to the historical inline loop — see _init_worlds' BIT CONTRACT — so the
    # ``rng`` consumption sequence is unchanged and the fixed-budget path below
    # stays bit-exact.
    memo_ctx = None
    root_pm = None
    if rollout_memo is not None:
        root_pm = pick_meta_for(root_obs, root_n)
        memo_ctx = {"table": rollout_memo, "base_picks": tuple(memo_picks),
                    "world_seed": 0, "hits": 0}

    seeds, roots, budgets, reused_visits = _init_worlds(
        worlds, world_seeds, rng, root_n, root_priors, root_is_a,
        root_noise_eps=root_noise_eps, root_noise_alpha=root_noise_alpha,
        reuse_roots=reuse_roots, root_rep=root_rep, root_pm=root_pm,
        sims_per_world=sims_per_world)

    def _one_sim(w: int) -> None:
        nonlocal sims_run, sim_steps
        env.restore(snapshot_slot)
        q = env.determinize(seeds[w])
        _check_root_query(q, root_n, w)
        if memo_ctx is not None:
            memo_ctx["world_seed"] = seeds[w]
        v = _simulate(env, evaluator, roots[w], c_puct, max_depth,
                      rollout_turns, rollout_anchor, memo_ctx, merge_dupes)
        sims_run += 1
        sim_steps += v[1]

    # Cross-world deferral applies only when the budget in force has rollouts
    # OFF (the actor's restriction, latched per search); with it on, the
    # sequential path below runs regardless of the flag.
    cross = bool(cross_world) and rollout_turns == 0
    pending: list = []
    world_pending = [False] * worlds

    def _one_sim_cross(w: int) -> None:
        # The deferring twin of _one_sim: flush the shared batch before this
        # world's OWN next descent (its previous leaf must be backed up and
        # expanded first — that is what keeps the per-world trees identical to
        # the sequential search), then descend and park at the leaf.
        nonlocal sims_run, sim_steps
        if world_pending[w]:
            _flush_pending(evaluator, pending, world_pending, merge_dupes,
                           memo_ctx is not None)
        env.restore(snapshot_slot)
        q = env.determinize(seeds[w])
        _check_root_query(q, root_n, w)
        if memo_ctx is not None:
            memo_ctx["world_seed"] = seeds[w]
        pl, steps = _simulate_defer(env, roots[w], c_puct, max_depth)
        if pl is not None:
            pending.append(pl)
            world_pending[w] = True
        sims_run += 1
        sim_steps += steps

    step_fn = _one_sim_cross if cross else _one_sim

    stopped_early = False
    if time_budget_s is None:
        if cross:
            # Round-robin over the worlds (skipping exhausted budgets) so the
            # pending batch collects one leaf per world (K = worlds) between
            # flushes. Interleaving order does not affect any world's own tree
            # — each is private — so visits equal the sequential loop's.
            remaining = list(budgets)
            left = sum(remaining)
            w = 0
            while left > 0:
                if remaining[w] > 0:
                    step_fn(w)
                    remaining[w] -= 1
                    left -= 1
                w = (w + 1) % worlds
        else:
            # UNCHANGED sequential per-world loop (parity corpus / self-play
            # depend on this exact order and count; budgets[w] ==
            # sims_per_world for every world when reuse_roots is None, so the
            # iteration is byte-identical).
            for w in range(worlds):
                for _ in range(budgets[w]):
                    _one_sim(w)
    else:
        deadline = time.monotonic() + float(time_budget_s)
        min_deadline = None
        if time_budget_min_s is not None:
            min_deadline = time.monotonic() + min(
                float(time_budget_min_s), float(time_budget_s))
        cap = sims if sims and sims > 0 else None
        # Floor: one sim per world, unconditionally. (step_fn defers under
        # cross-world — the timed loop below is already a round-robin, so the
        # deferral discipline slots straight in; the stability check reads the
        # roots' backed-up stats, at most one unflushed leaf per world behind,
        # which is well inside its approximation.)
        for w in range(worlds):
            step_fn(w)
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
            step_fn(i % worlds)
            i += 1
            since_check += 1

    # Cross-world tail: the last round's deferred leaves flush before the
    # aggregate (the actor's finalize does the same).
    if cross:
        _flush_pending(evaluator, pending, world_pending, merge_dupes,
                       memo_ctx is not None)

    visit_totals, w_totals, world_values, value_acc = _aggregate_roots(
        roots, root_n)

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
        roots=roots,
        seeds=seeds,
        reused_visits=reused_visits,
        memo_hits=memo_ctx["hits"] if memo_ctx is not None else 0,
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

    def evaluate_batch(self, obs_batch, num_choices):
        """Locked pass-through so a cross-world search behind the lock keeps
        its batched forward (row-by-row under the lock when the inner
        evaluator has no batch entry point — same results either way)."""
        with self._lock:
            inner = getattr(self._evaluator, "evaluate_batch", None)
            if inner is not None:
                return inner(obs_batch, num_choices)
            return [self._evaluator.evaluate(o, n)
                    for o, n in zip(obs_batch, num_choices)]


def run_search_parallel(
    envs: Sequence[SearchRoboMageEnv],
    evaluator: Evaluator,
    *,
    sims: int = 128,
    worlds: int = 4,
    c_puct: float = 1.5,
    max_depth: int = 60,
    rollout_turns: int = 0,
    root_noise_eps: float = 0.0,
    root_noise_alpha: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    world_seeds: Optional[Sequence[int]] = None,
    reuse_roots: Optional[Sequence] = None,
    rollout_memo: Optional[dict] = None,
    memo_picks: tuple = (),
    time_budget_s: Optional[float] = None,
    time_budget_min_s: Optional[float] = None,
    merge_dupes: bool = True,
    cross_world: bool = False,
) -> SearchResult:
    """World-parallel :func:`run_search` for INTERACTIVE search only.

    ``cross_world`` is forwarded per worker, so each env batches the leaves of
    its OWN world slice (K = slice size) — the visits stay equal to the
    1-env cross-world run by the same argument that makes the split exact.

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

    ``world_seeds`` / ``reuse_roots`` / ``rollout_memo`` / ``memo_picks`` are the
    bo3 sideboard-boundary persistence handles (contracts in :func:`run_search`),
    forwarded per env: pinned ``world_seeds`` replace the pre-draw entirely (no
    rng is consumed for seeds then) and ``reuse_roots`` is sliced by the same
    contiguous world split, so each worker tops up exactly the trees belonging to
    its own worlds. Persistence therefore works identically at any ``procs`` —
    the per-world state is engine-agnostic, so a boundary latched under N envs
    may be continued under a different env count (or serially) without loss.

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
            max_depth=max_depth, rollout_turns=rollout_turns,
            root_noise_eps=root_noise_eps,
            root_noise_alpha=root_noise_alpha, rng=rng,
            world_seeds=world_seeds, reuse_roots=reuse_roots,
            rollout_memo=rollout_memo, memo_picks=memo_picks,
            time_budget_s=time_budget_s,
            time_budget_min_s=time_budget_min_s, merge_dupes=merge_dupes,
            cross_world=cross_world)

    assert root_noise_eps == 0.0, (
        "run_search_parallel is inference-only: root dirichlet noise would consume "
        "the rng per world in a thread-order-dependent way")

    rng = rng if rng is not None else np.random.default_rng()
    if world_seeds is None:
        # Pre-draw every world seed with the SAME derivation run_search uses, so a
        # 1-env call would consume the rng identically to plain run_search.
        world_seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(worlds)]
    else:
        # Pinned seeds (boundary persistence): consume no rng at all, exactly like
        # run_search's world_seeds path.
        _check_world_seeds(world_seeds, worlds)
        world_seeds = [int(s) for s in world_seeds[:worlds]]
    if reuse_roots is not None and len(reuse_roots) < worlds:
        raise ValueError(
            f"reuse_roots needs >= {worlds} entries, got {len(reuse_roots)}")

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
        # The rollout memo dict is deliberately SHARED (unlocked) across the
        # worker threads: every key embeds the world seed (rollout_memo_key) and
        # the workers own disjoint seed slices, so two workers can never write
        # the same key, and CPython dict get/setitem are GIL-atomic — no lock is
        # needed (one would only serialize the boundary's hot path).
        return run_search(
            usable[i], locked, sims=sims_i, worlds=k_i, c_puct=c_puct,
            max_depth=max_depth, rollout_turns=rollout_turns,
            root_noise_eps=0.0, rng=None,
            world_seeds=seeds_i,
            reuse_roots=(list(reuse_roots[lo:hi])
                         if reuse_roots is not None else None),
            rollout_memo=rollout_memo, memo_picks=memo_picks,
            time_budget_s=time_budget_s,
            time_budget_min_s=time_budget_min_s, merge_dupes=merge_dupes,
            cross_world=cross_world)

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
    visits, w_sum, sims_run, sim_steps, q = _merge_root_stats(results)
    # root_value stays VERBATIM at this site (visit-weighted average of the
    # workers' own root_values) — deliberately NOT unified with the other
    # merge site's w_sum.sum()/total; see _merge_root_stats.
    weighted_value = 0.0
    total_vis = 0.0
    for r in results:
        vs = float(r.visits.sum())
        weighted_value += r.root_value * vs
        total_vis += vs
    root_value = weighted_value / total_vis if total_vis > 0 else 0.0
    # Roots/seeds alignment (boundary persistence depends on it): the roots are
    # concatenated in env order and each env owns the CONTIGUOUS world slice
    # world_seeds[lo:hi], so env order == world order and roots[w] is the tree
    # built under seeds[w] — index-for-index, exactly as run_search returns them.
    # (_Node carries no seed of its own, so this cannot be asserted cheaply; the
    # parallel-persistence parity test in train/test_mirror_search.py guards it.)
    return SearchResult(
        visits=visits,
        priors=results[0].priors,
        root_value=root_value,
        num_choices=root_n,
        sims_run=sims_run,
        sim_steps=sim_steps,
        stopped_early=any(r.stopped_early for r in results),
        q=q,
        w_sum=w_sum,
        world_values=np.concatenate([r.world_values for r in results]),
        roots=[root for r in results for root in (r.roots or [])],
        seeds=list(world_seeds),
        reused_visits=sum(r.reused_visits for r in results),
        memo_hits=sum(r.memo_hits for r in results),
    )


def _check_root_query(q: SimQuery, root_n: int, world: int) -> None:
    """Restore+determinize must re-derive the exact decision the search rooted
    on. A different menu size means the engine's snapshot round-trip diverged
    (or determinization touched information the root menu depends on) — selecting
    from the stale root arrays would then send the engine an out-of-menu index,
    so fail loudly here instead."""
    if q.num_choices != root_n:
        derived = ""
        try:  # decode the re-derived menu for the report; never mask the error
            import decode
            menu = decode.decode_actions_from_obs(q.obs, q.num_choices)
            derived = " — re-derived menu: " + " | ".join(str(m) for m in menu)
        except Exception:
            pass
        raise RuntimeError(
            f"world-consistency violation at root (world {world}): search root "
            f"has {root_n} choices but restore+determinize re-derived "
            f"{q.num_choices}{derived}")


def _rollout_anchor(root_obs: np.ndarray) -> int:
    """The turn the rollout horizon counts from: 0 at a bo3 sideboard root
    (the horizon is "through player-turn N of the sampled NEXT game"), else
    the root's own turn (an in-game rollout extends N turns past the root).
    Mirrored by the C++ actor's latch in begin_or_fallback (az_mcts.cpp)."""
    if root_obs[_IS_SIDEBOARD_IDX] > 0.5:
        return 0
    return int(round(float(root_obs[_CUR_TURN_IDX]) * 50))


def _rollout_stop(obs: np.ndarray, anchor: int, rollout_turns: int) -> bool:
    """True when a rollout state has reached the end-of-turn horizon. The
    is_sideboard gate exempts a sideboard-root sim's sideboard/mulligan prefix,
    where the obs turn field still holds the ENDED game's turn; once the next
    game starts the turn counter restarts at 0 and the comparison is live.
    (Inert for in-game roots — a bo3 in-game sim ends at SIM_RESULT before it
    can re-enter a sideboard phase.)"""
    if obs[_IS_SIDEBOARD_IDX] > 0.5:
        return False
    turn = int(round(float(obs[_CUR_TURN_IDX]) * 50))
    return turn >= anchor + rollout_turns


# ── Sideboard-boundary helpers (persistent trees + rollout memo) ──────────────
# A bo3 sideboard BOUNDARY is one seat's contiguous run of pick decisions
# between two games. These rules are shared by every boundary consumer
# (SearchController, az_selfplay, test_mcts_parity) and MIRRORED in
# src/actor/az_mcts.cpp — the two sides must derive identical seeds, boundary
# identity, pick descriptors, and walk outcomes or visit parity breaks.

def world_seeds_for(base: int, root_index: int, n: int) -> list[int]:
    """The C++ actor's per-world determinize seed formula (az_mcts.cpp
    begin_world). Under boundary persistence `root_index` is the BOUNDARY's
    first searched root, frozen for every top-up search of that boundary."""
    return [(base + 100003 * root_index + w) & 0xFFFFFFFF for w in range(n)]


def sb_root_key(obs: np.ndarray) -> Optional[tuple[bool, int]]:
    """Boundary identity of a sideboard root: (seat, upcoming game number), or
    None when the decision is not a sideboard prompt. Seat is required because
    both seats' pick runs share one sideboard phase back-to-back; the game
    number distinguishes the g1->g2 boundary from g2->g3. C++ mirror reads the
    same engine globals the obs serializes (sideboard_phase, viewer seat,
    match_game_number + 1)."""
    if obs[_IS_SIDEBOARD_IDX] <= 0.5:
        return None
    return (bool(obs[_SELF_IS_A_IDX] > 0.5),
            int(round(float(obs[_MATCH_CTX_START]) * 3)))


def sb_pick_descriptor(obs: np.ndarray, action: int) -> Optional[tuple[int, int, int]]:
    """Order-insensitive description of a sideboard pick action: (category,
    card vocab id, acting seat as 0/1), or None for non-pick actions (Done,
    anything in-game). Menu INDICES are mutation-history-dependent, so keys and
    memo entries are built from these descriptors, never from indices. ord and
    ctrl metadata are deliberately excluded (live copy count / null sentinel)."""
    cat = _obs_action_category(obs, action)
    if cat not in (CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT):
        return None
    seat = 1 if obs[_SELF_IS_A_IDX] > 0.5 else 0
    return (cat, _obs_action_card_id(obs, action), seat)


def pick_meta_for(obs: np.ndarray, num_choices: int) -> tuple:
    """The per-action sb_pick_descriptor tuple stored on a _Node at creation
    (memo-enabled searches only)."""
    return tuple(sb_pick_descriptor(obs, a) for a in range(num_choices))


def rollout_memo_key(world_seed: int, leaf_seat_is_a: bool, picks) -> tuple:
    """Memo key for a rollout from a leaf still inside the sideboard phase:
    the pinned world seed (the sampled next-game deal), the leaf's mover seat,
    and the SORTED multiset of pick descriptors accumulated from the boundary
    root to the leaf — order-insensitive, so permuted pick orders reaching the
    same net configuration collide (deliberately: an accepted approximation,
    see docs/alphazero_status.md)."""
    return (int(world_seed), bool(leaf_seat_is_a), tuple(sorted(picks)))


def rollout_memo_eligible(picks) -> bool:
    """False when any (card, seat) appears in BOTH pick directions. Under the
    IN-FIRST engine menu that arises only via the forced OUT of a stranded IN,
    which may cut a name that was just sided in: the deck ends unchanged but the
    one-shot direction locks do not, so the greedy completion (and therefore the
    rollout) genuinely diverges between orderings. Such keys are never
    memoized."""
    ins: set = set()
    outs: set = set()
    for cat, cid, seat in picks:
        (ins if cat == CAT_SIDEBOARD_IN else outs).add((cid, seat))
    return not (ins & outs)


def walk_reuse_root(root: "_Node", actions, num_choices: int,
                    seat_is_a: bool) -> Optional["_Node"]:
    """Walk a stored world tree down the actually-played actions since the
    previous search. Returns the reached node when it exists and matches the
    new root's menu size and seat, else None (that world re-roots fresh; a
    failed walk never invalidates the whole boundary). Along the real
    deterministic line indices are stable, so child-by-index is exact."""
    node = root
    for a in actions:
        a = int(a)
        # A real played action may be a merged-away duplicate (the tree only
        # keeps children at representative indices) — canonicalize through the
        # node's merge partition before the child lookup.
        if node.rep is not None and 0 <= a < node.num_choices:
            a = int(node.rep[a])
        node = node.children.get(a)
        if node is None:
            return None
    if node.num_choices != num_choices or node.self_is_a != seat_is_a:
        return None
    return node


def _check_world_seeds(world_seeds: Optional[Sequence[int]],
                       worlds: int) -> None:
    """Validate a caller-pinned per-world seed list (no-op when ``None``).

    Shared by :func:`run_search`, :func:`run_search_parallel` and
    :class:`IncrementalSearch` so all three reject a short list with the same
    message, before any snapshot/evaluate work happens."""
    if world_seeds is not None and len(world_seeds) < worlds:
        raise ValueError(
            f"world_seeds needs >= {worlds} entries, got {len(world_seeds)}")


def _terminal_value(tag: str, root_is_a: bool) -> float:
    """Root-relative value of a terminal game tag ("A" / "B" / "DRAW").

    A draw is worth exactly 0.0 to both seats; any other tag is ±1.0 by whether
    the winning seat is the root mover. These draw semantics are mirrored in
    the C++ actor (``src/actor/az_mcts.cpp``) — do not drift."""
    if tag == "DRAW":
        return 0.0
    return 1.0 if (tag == "A") == root_is_a else -1.0


def _rollout(
    env: SearchRoboMageEnv,
    evaluator: Evaluator,
    query: SimQuery,
    priors: np.ndarray,
    value: float,
    seat_is_a: bool,
    anchor: int,
    rollout_turns: int,
    root_is_a: bool,
) -> tuple[float, bool, int]:
    """Argmax-greedy playout from a freshly expanded leaf to the end-of-turn
    horizon (or terminal / step cap), on the current determinized world.

    `query`/`priors`/`value`/`seat_is_a` describe the LEAF state (whose
    evaluation the caller already paid for); each further state costs exactly
    one evaluate(), whose priors pick the next action (first-max argmax, no
    rng — parity with the C++ actor's strict-> tie-break) and whose value is
    what gets backed up if the state is where the rollout stops. Rolled-out
    states are never added to the tree. Returns (leaf_value, leaf_seat_is_a,
    engine steps taken, terminal) for the unchanged backup loop; ``terminal``
    is "A"/"B"/"DRAW" when the playout hit game end (its value is already
    root-relative), else None (net value at the horizon, mover perspective) —
    the tag the rollout memo stores."""
    cap = ROLLOUT_STEPS_PER_TURN * rollout_turns
    steps = 0
    while True:
        if steps >= cap or _rollout_stop(query.obs, anchor, rollout_turns):
            return value, seat_is_a, steps, None
        query = env.sim_step(int(np.argmax(priors)))
        steps += 1
        if query.terminal is not None:
            return (_terminal_value(query.terminal, root_is_a), root_is_a,
                    steps, query.terminal)
        seat_is_a = bool(query.obs[_SELF_IS_A_IDX] > 0.5)
        priors, value = evaluator.evaluate(query.obs, query.num_choices)


def _simulate(
    env: SearchRoboMageEnv,
    evaluator: Evaluator,
    root: _Node,
    c_puct: float,
    max_depth: int,
    rollout_turns: int = 0,
    rollout_anchor: int = 0,
    memo_ctx: Optional[dict] = None,
    merge_dupes: bool = True,
) -> tuple[float, int]:
    """One PUCT descent from the (already restored+determinized) root.
    Returns (leaf value in root-node mover perspective, engine steps used).

    With ``rollout_turns > 0`` a freshly expanded leaf is not evaluated in
    place: the raw policy plays the world forward to the end-of-turn horizon
    (``rollout_anchor + rollout_turns``) and THAT state's net value (or a true
    terminal ±1) is backed up instead. Only the expansion leaf rolls out — the
    max_depth cutoff below keeps its in-place evaluation (it bounds tree
    descent, not the horizon).

    ``memo_ctx`` (rollout memo, sideboard boundaries only): dict with keys
    ``table`` (the boundary's memo), ``base_picks`` (descriptors of the REAL
    picks since the boundary root), ``world_seed`` (this sim's pinned seed,
    set per sim by the caller) and ``hits``. Sim-path picks are accumulated
    from each node's creation-time ``pick_meta``; an eligible leaf still in
    the sideboard phase reuses a stored rollout result instead of playing
    out (adding 0 engine steps)."""
    node = root
    path: list[tuple[_Node, int]] = []
    steps = 0
    leaf_value = 0.0       # in the perspective of `leaf_seat_is_a`
    leaf_seat_is_a = root.self_is_a
    picks = list(memo_ctx["base_picks"]) if memo_ctx is not None else None

    while True:
        action = node.select(c_puct)
        if picks is not None and node.pick_meta is not None:
            d = node.pick_meta[action]
            if d is not None:
                picks.append(d)
        query: SimQuery = env.sim_step(action)
        steps += 1
        path.append((node, action))

        if query.terminal is not None:
            leaf_seat_is_a = root.self_is_a
            leaf_value = _terminal_value(query.terminal, root.self_is_a)
            break

        child = node.children.get(action)
        if child is None:
            child_is_a = bool(query.obs[_SELF_IS_A_IDX] > 0.5)
            priors, value = evaluator.evaluate(query.obs, query.num_choices)
            pm = (pick_meta_for(query.obs, query.num_choices)
                  if picks is not None else None)
            rep = (menu_merge_reps(query.obs, query.num_choices)
                   if merge_dupes else None)
            node.children[action] = _Node(query.num_choices, priors, child_is_a,
                                          pick_meta=pm, rep=rep)
            if rollout_turns > 0:
                key = None
                if (picks is not None and query.obs[_IS_SIDEBOARD_IDX] > 0.5
                        and rollout_memo_eligible(picks)):
                    key = rollout_memo_key(memo_ctx["world_seed"], child_is_a,
                                           picks)
                    hit = memo_ctx["table"].get(key)
                    if hit is not None:
                        memo_ctx["hits"] += 1
                        if hit[0] == "t":
                            leaf_seat_is_a = root.self_is_a
                            leaf_value = _terminal_value(hit[1], root.self_is_a)
                        else:
                            leaf_value, leaf_seat_is_a = hit[1], hit[2]
                        break
                lv, ls, extra, term = _rollout(
                    env, evaluator, query, priors, value, child_is_a,
                    rollout_anchor, rollout_turns, root.self_is_a)
                steps += extra
                if key is not None:
                    memo_ctx["table"][key] = (("t", term) if term is not None
                                              else ("v", lv, ls))
                leaf_value, leaf_seat_is_a = lv, ls
            else:
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


# ── Cross-world batched leaf evaluation ──────────────────────────────────────
# The Python twin of the C++ actor's cross-world scheduler (Stage 0 of
# docs/gpu_selfplay_inference_plan.md, src/actor/az_mcts.cpp): worlds run
# round-robin, each freshly expanded (or depth-capped) leaf DEFERS its net
# evaluation into a shared pending batch, and the batch is flushed in one
# forward before any world's OWN next descent starts — so no virtual loss is
# needed and every per-world tree is arithmetically identical to the
# sequential search. Only a batched GEMM's last-ulp logits can differ; an
# evaluator without ``evaluate_batch`` (uniform, PPO) is called row-by-row and
# stays BIT-IDENTICAL. Searches whose budget has leaf rollouts on keep the
# sequential path (a deferred leaf cannot drive a playout), mirroring the
# actor's restriction.


class _PendingLeaf:
    """A descent parked at its evaluation point, awaiting the batched forward.

    ``parent is None`` marks a DEPTH-CAP leaf (the reached child already
    exists: value-only backup, no node creation) — the same distinction the
    actor's flush_pending draws via its children.find guard."""

    __slots__ = ("obs", "num_choices", "path", "parent", "action", "child_is_a")

    def __init__(self, obs, num_choices, path, parent, action, child_is_a):
        self.obs = obs
        self.num_choices = num_choices
        self.path = path
        self.parent = parent
        self.action = action
        self.child_is_a = child_is_a


def _backup(path, leaf_value: float, leaf_seat_is_a: bool) -> None:
    """The one backup rule (verbatim from :func:`_simulate`'s tail, which keeps
    its own inline copy — its statement stream is bit-pinned)."""
    for parent, action in path:
        parent.N[action] += 1
        parent.W[action] += (leaf_value if parent.self_is_a == leaf_seat_is_a
                             else -leaf_value)


def _simulate_defer(
    env: SearchRoboMageEnv,
    root: _Node,
    c_puct: float,
    max_depth: int,
) -> tuple[Optional[_PendingLeaf], int]:
    """One PUCT descent that PARKS at its evaluation point instead of
    evaluating: returns ``(pending_leaf, steps)`` for a leaf awaiting the
    batched forward, or ``(None, steps)`` when the sim ended at a TERMINAL
    (whose exact ±1/0 value needs no net and was backed up in place — same as
    the sequential path). The descent arithmetic (select, world-consistency
    check, depth-cap ordering) mirrors :func:`_simulate` exactly; rollouts and
    the sideboard rollout memo are deliberately absent — cross-world
    scheduling only runs when the budget in force has rollouts OFF, where both
    are inert. ``obs`` is COPIED into the pending leaf (the env reuses its
    query buffer on the next sim_step)."""
    node = root
    path: list[tuple[_Node, int]] = []
    steps = 0
    while True:
        action = node.select(c_puct)
        query: SimQuery = env.sim_step(action)
        steps += 1
        path.append((node, action))

        if query.terminal is not None:
            _backup(path, _terminal_value(query.terminal, root.self_is_a),
                    root.self_is_a)
            return None, steps

        child = node.children.get(action)
        if child is None:
            return _PendingLeaf(
                obs=query.obs.copy(), num_choices=query.num_choices,
                path=path, parent=node, action=action,
                child_is_a=bool(query.obs[_SELF_IS_A_IDX] > 0.5)), steps

        if child.num_choices != query.num_choices:
            raise RuntimeError(
                f"world-consistency violation: node expected {child.num_choices} "
                f"choices, engine gave {query.num_choices}")
        node = child

        if len(path) >= max_depth:
            # Depth cap: the child exists — value-only backup at flush.
            return _PendingLeaf(
                obs=query.obs.copy(), num_choices=query.num_choices,
                path=path, parent=None, action=action,
                child_is_a=node.self_is_a), steps


def _flush_pending(evaluator: Evaluator, pending: list, world_pending,
                   merge_dupes: bool, memo_enabled: bool = False) -> None:
    """Evaluate every pending leaf (ONE ``evaluate_batch`` forward when the
    evaluator has one and k > 1, else row-by-row — bit-identical to
    sequential), then create/back up each in submission order. Clears
    ``pending`` and every ``world_pending`` flag."""
    if not pending:
        return
    batch_fn = getattr(evaluator, "evaluate_batch", None)
    if batch_fn is not None and len(pending) > 1:
        results = batch_fn([pl.obs for pl in pending],
                           [pl.num_choices for pl in pending])
    else:
        results = [evaluator.evaluate(pl.obs, pl.num_choices)
                   for pl in pending]
    for pl, (priors, value) in zip(pending, results):
        if pl.parent is not None and pl.action not in pl.parent.children:
            pm = (pick_meta_for(pl.obs, pl.num_choices)
                  if memo_enabled else None)
            rep = (menu_merge_reps(pl.obs, pl.num_choices)
                   if merge_dupes else None)
            pl.parent.children[pl.action] = _Node(
                pl.num_choices, priors, pl.child_is_a, pick_meta=pm, rep=rep)
        _backup(pl.path, float(value), pl.child_is_a)
    pending.clear()
    if world_pending is not None:
        for i in range(len(world_pending)):
            world_pending[i] = False


# ── Sideboard plan search (flat search over complete configurations) ──────────
# A bo3 sideboard root is NOT searched with PUCT: the searchable object there is
# the final deck CONFIGURATION, not a pick ordering (the rollout memo already
# keyed values on the order-insensitive pick multiset), and at the old budgets a
# single world's tree could not even visit every first pick once. Instead the
# root is searched as a flat set of PLANS — complete pick sequences through the
# mover's Done. The engine menu is IN-FIRST (Done + the sideboard's INs while
# balanced; the maindeck's OUTs while a swap is open), so a plan is Done, or an
# alternating in/out/in/out/.../Done chain, and coverage costs one plan per
# distinct sideboard name plus one:
#
#   * COVERAGE: one plan per legal root action (the action, then an
#     argmax-greedy raw-policy completion of the mover's remaining picks).
#   * EXTRAS (`branches`): deterministic alternate completions of the best-Q
#     first picks — variant v takes the SECOND-best prior at the v-th
#     completion decision (no rng, so the C++ twin needs no cross-language rng
#     parity).
#
# Every plan is evaluated on EVERY determinized world: replay its picks, then
# leaf-rollout (the existing _rollout: opponent's picks, mulligans, the next
# game to end of player-turn `rollout_turns`). Plan value = mean over worlds —
# a plan must be chosen before knowing which world is real, so per-plan
# cross-world averaging is the quantity that ranks plans. The training target
# is π = softmax(Q_a / SB_PI_TAU) over first picks (Q_a = best plan value
# starting with a); a flat coverage pass gives every first pick the same
# evaluation count, so visit counts carry no ranking signal and the target
# comes from VALUES.
#
# Values are memoized per (world seed, pick multiset) in a boundary-shared
# dict, which both dedups coverage plans that converge to one configuration
# and makes later picks of the same boundary nearly free (every surviving
# plan's world values carry over) — this REPLACES tree-based boundary
# persistence (reuse_roots/walk_reuse_root are not used at sideboard roots).
#
# Batching note: within one process the engine is a snapshot/restore-sliced
# singleton, so plan evaluation is sequential and net evals are row-by-row —
# the same constraint that kept rollouts out of cross-world batching. Fleet
# batching happens across PROCESSES via the eval server; the analysis window
# splits WORLDS across its engine fleet (plans regenerate identically per
# engine — completions depend only on the mover's obs, which determinize does
# not touch).
#
# MIRRORED in src/actor/az_mcts.cpp — generation order, deviation rule, memo
# keys, and the float64 softmax must stay bit-identical or sb parity breaks.

# Temperature of the π = softmax(Q/τ) sideboard training target, in tanh value
# units: ΔQ of SB_PI_TAU is one e-fold of probability. Mirrored as kSbPiTau in
# src/actor/az_mcts.cpp.
SB_PI_TAU = 0.25

# Safety bound on one plan's own pick sequence, and an EXACT one: the engine's
# menu is IN-FIRST, so every decision is either the IN half of a swap, the OUT
# half that closes it, or Done. SIDEBOARD_SWAP_CAP (15) completed swaps therefore
# cost at most 15 * 2 + 1 (Done) = 31 decisions, and the stranded path is no
# longer: 14 swaps + the stranding IN + its forced OUT + the Done-only menu is 31
# too. The check below is a strict `>` applied BEFORE the pick is appended, so a
# legal line (whose longest prefix at check time is 30) never trips it and 31 is
# not off by one. Anything past it is a broken menu loop, not a long deck.
# Mirrored as kPlanPickCap in src/actor/az_mcts.cpp.
_PLAN_PICK_CAP = 31


@dataclass
class SbPlan:
    """One complete sideboard configuration considered by run_plan_search."""
    first_action: int           # root-menu action the plan starts with
    pick_actions: list          # full env-index pick sequence (incl. first)
    picks: tuple                # sb_pick_descriptor tuple (Done contributes none)
    multiset: tuple             # sorted(base_picks + picks) — the memo identity
    values: np.ndarray          # (worlds,) value per world, ROOT perspective
    variant: int = 0            # 0 = coverage / greedy; v>0 = extras deviation

    @property
    def value(self) -> float:
        return float(self.values.mean())


def _second_best(priors: np.ndarray) -> int:
    """First-max argmax excluding the first-max argmax (parity: strict-> in
    C++). Only called with num_choices >= 2."""
    best = int(np.argmax(priors))
    p = priors.copy()
    p[best] = -np.inf
    return int(np.argmax(p))


def _plan_softmax(q: np.ndarray, tau: float = SB_PI_TAU) -> np.ndarray:
    """Float64 softmax with max-subtraction — the π target arithmetic, mirrored
    bit-for-bit in the C++ actor."""
    z = (q - q.max()) / tau
    e = np.exp(z)
    return e / e.sum()


def _generate_plan(env: SearchRoboMageEnv, evaluator: Evaluator, *,
                   root_obs: np.ndarray, first_action: int, variant: int,
                   root_is_a: bool, counters: dict) -> tuple:
    """Build one plan's pick sequence on the CURRENT (already restored +
    determinized) world by stepping the env through `first_action` and then a
    greedy completion of the mover's remaining picks (variant v swaps in the
    second-best prior at the v-th completion decision). Returns
    ``(pick_actions, picks, handoff_query)`` where handoff_query is the state
    where the mover's run ended (opponent's picks / mulligans / next game —
    the rollout takes it from there). Raises on world-consistency breaks."""
    picks: list = []
    pick_actions: list = [int(first_action)]
    d = sb_pick_descriptor(root_obs, int(first_action))
    if d is not None:
        picks.append(d)
    query: SimQuery = env.sim_step(int(first_action))
    counters["steps"] += 1
    dev = 0
    while (query.terminal is None and query.obs[_IS_SIDEBOARD_IDX] > 0.5
           and bool(query.obs[_SELF_IS_A_IDX] > 0.5) == root_is_a):
        if len(pick_actions) > _PLAN_PICK_CAP:
            raise RuntimeError(
                "plan-search pick sequence exceeded _PLAN_PICK_CAP — "
                "sideboard menu loop did not terminate")
        priors, _ = evaluator.evaluate(query.obs, query.num_choices)
        counters["evals"] += 1
        dev += 1
        if variant > 0 and dev == variant and query.num_choices >= 2:
            a = _second_best(priors)
        else:
            a = int(np.argmax(priors))
        d = sb_pick_descriptor(query.obs, a)
        if d is not None:
            picks.append(d)
        pick_actions.append(a)
        query = env.sim_step(a)
        counters["steps"] += 1
    return pick_actions, tuple(picks), query


def _eval_plan_world(env: SearchRoboMageEnv, evaluator: Evaluator, *,
                     query: SimQuery, root_is_a: bool, rollout_turns: int,
                     counters: dict) -> float:
    """Rollout from a plan's handoff state on the current world; returns the
    value in ROOT perspective."""
    if query.terminal is not None:
        return _terminal_value(query.terminal, root_is_a)
    priors, value = evaluator.evaluate(query.obs, query.num_choices)
    counters["evals"] += 1
    seat = bool(query.obs[_SELF_IS_A_IDX] > 0.5)
    lv, ls, extra, _term = _rollout(env, evaluator, query, priors, value, seat,
                                    0, rollout_turns, root_is_a)
    counters["steps"] += extra
    counters["evals"] += extra  # one evaluate() per rollout step (see _rollout)
    return lv if ls == root_is_a else -lv


def _replay_plan(env: SearchRoboMageEnv, root_n: int, world: int,
                 seed: int, snapshot_slot: int, pick_actions,
                 counters: dict) -> SimQuery:
    """Restore + determinize `world` and force-replay a plan's picks. The pick
    menus derive from the mover's own zones (untouched by determinize), so the
    recorded indices must stay legal — a violation fails loudly."""
    env.restore(snapshot_slot)
    q = env.determinize(seed)
    _check_root_query(q, root_n, world)
    for a in pick_actions:
        a = int(a)
        if q.terminal is not None or a >= q.num_choices:
            raise RuntimeError(
                f"plan-search world-consistency violation (world {world}): "
                f"pick {a} not replayable ({q.num_choices} choices, "
                f"terminal={q.terminal})")
        q = env.sim_step(a)
        counters["steps"] += 1
    return q


def run_plan_search(
    env: SearchRoboMageEnv,
    evaluator: Evaluator,
    *,
    worlds: int = 4,
    branches: int = 8,
    rollout_turns: int = 6,
    rng: Optional[np.random.Generator] = None,
    snapshot_slot: int = 0,
    world_seeds: Optional[Sequence[int]] = None,
    plan_memo: Optional[dict] = None,
    memo_picks: tuple = (),
    time_budget_s: Optional[float] = None,
    time_budget_min_s: Optional[float] = None,
    plans_out: Optional[list] = None,
) -> SearchResult:
    """Flat plan search of a bo3 sideboard root (see the section comment).

    The env must be parked at a sideboard prompt. ``world_seeds`` pins the
    boundary's worlds (a sideboard world seed IS the sampled next-game deal);
    when None, one ``rng.integers(1, 2**31 - 1)`` draw per world, like
    run_search. ``plan_memo`` is the boundary-shared value cache keyed
    ``(world_seed, sorted pick multiset)`` with ``memo_picks`` the descriptors
    of the picks already played this boundary; pass the same dict across the
    boundary's roots and consistent plans re-price for free. ``time_budget_s``
    bounds the EXTRAS only — the coverage pass always completes (it is the
    correctness floor: every first pick evaluated). ``time_budget_min_s`` is
    accepted for interface parity and unused (there is no stability stop —
    plans are a fixed population, not a converging tree). ``plans_out``, when
    given, receives the evaluated :class:`SbPlan` list (analysis consumers).

    Returns a normal :class:`SearchResult`: ``visits = π * evaluations``
    (float64, so ``best_action``/``policy_target``/``sample_visits`` keep
    their contracts), ``q[a] = Q_a``, ``root_value`` = π-weighted mean of Q,
    ``roots`` empty (there are no trees to follow or persist)."""
    rng = rng if rng is not None else np.random.default_rng()
    _check_world_seeds(world_seeds, worlds)
    worlds = max(1, int(worlds))

    root_obs = env._obs.copy()
    root_n = env._num_choices
    if not root_obs[_IS_SIDEBOARD_IDX] > 0.5:
        raise ValueError("run_plan_search: env is not at a sideboard prompt")
    root_is_a = bool(root_obs[_SELF_IS_A_IDX] > 0.5)
    root_priors, _ = evaluator.evaluate(root_obs, root_n)

    seeds = [int(world_seeds[w]) if world_seeds is not None
             else int(rng.integers(1, 2**31 - 1)) for w in range(worlds)]
    memo = plan_memo if plan_memo is not None else {}
    base_picks = tuple(memo_picks)
    counters = {"steps": 0, "evals": 0, "memo_hits": 0}

    env.snapshot(snapshot_slot)
    deadline = (time.monotonic() + float(time_budget_s)
                if time_budget_s is not None else None)

    plans: list[SbPlan] = []
    evaluated: set = set()   # multisets already priced (novelty test for extras)

    def _run_plan(first_action: int, variant: int) -> tuple[SbPlan, bool]:
        # Generate on world 0 (pick menus are world-independent), then price
        # every world — memo first, rollout on miss. Returns (plan, novel):
        # novel is False when the configuration was already fully priced (an
        # extras variant that collapsed onto a known multiset).
        env.restore(snapshot_slot)
        q0 = env.determinize(seeds[0])
        _check_root_query(q0, root_n, 0)
        pick_actions, picks, handoff = _generate_plan(
            env, evaluator, root_obs=root_obs, first_action=first_action,
            variant=variant, root_is_a=root_is_a, counters=counters)
        multiset = tuple(sorted(base_picks + picks))
        memoable = rollout_memo_eligible(multiset)
        novel = not (memoable and multiset in evaluated)
        values = np.zeros(worlds, dtype=np.float64)
        for w in range(worlds):
            key = (seeds[w], multiset)
            if memoable:
                hit = memo.get(key)
                if hit is not None:
                    counters["memo_hits"] += 1
                    values[w] = hit
                    continue
            q = handoff if w == 0 else _replay_plan(
                env, root_n, w, seeds[w], snapshot_slot, pick_actions,
                counters)
            values[w] = _eval_plan_world(
                env, evaluator, query=q, root_is_a=root_is_a,
                rollout_turns=rollout_turns, counters=counters)
            if memoable:
                memo[key] = float(values[w])
        plan = SbPlan(first_action=int(first_action),
                      pick_actions=pick_actions, picks=picks,
                      multiset=multiset, values=values, variant=variant)
        plans.append(plan)
        if memoable:
            evaluated.add(multiset)
        return plan, novel

    # Coverage pass: every legal root action, ascending — always completes
    # (converging configurations price for free through the memo).
    for a in range(root_n):
        _run_plan(a, 0)

    # Extras deviate the best branches: candidate i re-completes branch
    # ranked[i % root_n] with variant 1 + i // root_n. Candidates that collapse
    # onto an already-priced configuration don't consume the extras budget
    # (their cost is generation only — the memo prices them); a bounded
    # attempt count keeps a saturated plan space from spinning.
    cov_q = np.zeros(root_n, dtype=np.float64)
    for p in plans:
        if p.variant == 0:
            cov_q[p.first_action] = p.value
    ranked = np.argsort(-cov_q, kind="stable")

    novel_extras = 0
    attempt = 0
    max_attempts = max(0, int(branches)) * 4
    while novel_extras < int(branches) and attempt < max_attempts:
        if deadline is not None and time.monotonic() >= deadline:
            break
        branch = int(ranked[attempt % root_n])
        variant = 1 + attempt // root_n
        attempt += 1
        _plan, novel = _run_plan(branch, variant)
        if novel:
            novel_extras += 1

    # Q per first pick = best plan value starting with it (coverage guarantees
    # every first pick has at least one plan).
    q_arr = np.full(root_n, -np.inf, dtype=np.float64)
    for p in plans:
        q_arr[p.first_action] = max(q_arr[p.first_action], p.value)

    pi = _plan_softmax(q_arr)
    n_evals = len(plans) * worlds
    visits = pi * float(n_evals)
    w_sum = q_arr * visits
    world_values = np.array(
        [float(np.mean([p.values[w] for p in plans])) for w in range(worlds)],
        dtype=np.float64)

    env.restore(snapshot_slot)
    env.release()

    if plans_out is not None:
        plans_out.extend(plans)
    return SearchResult(
        visits=visits,
        priors=root_priors,
        root_value=float(pi @ q_arr),
        num_choices=root_n,
        sims_run=n_evals,
        sim_steps=counters["steps"],
        q=q_arr,
        w_sum=w_sum,
        world_values=world_values,
        roots=[],
        seeds=seeds,
        reused_visits=0,
        memo_hits=counters["memo_hits"],
    )


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
    w_sum: np.ndarray           # (n,) summed root W (value mass) across worlds
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

    ``world_seeds`` mirrors :func:`run_search`'s contract exactly: when given
    (at least ``worlds`` entries) the per-world determinize seeds are consumed
    from it in order and NO rng draw happens; when absent they are drawn as
    ``rng.integers(1, 2**31 - 1)`` per world, w=0..worlds-1 — the derivation
    the chunked-parity claim above depends on. Pinned seeds are what lets one
    search's worlds be split across several engines (the analysis window's
    world-parallel mode) and still merge to the single-engine result.

    Unlike run_search, the root SNAPSHOT is held OPEN across chunks so the
    search can resume, and :meth:`pv`/:meth:`walk` can browse the tree and
    replay hypothetical lines after the last chunk. The owner MUST call
    :meth:`close` (restore + release) before the env takes any real step —
    the same driver discipline as every snapshot consumer.

    ``cross_world`` (default False) applies the same deferred-batch discipline
    as :func:`run_search`'s cross-world mode inside each chunk: the chunk's
    round-robin defers every fresh/depth-capped leaf, flushes before a
    world's own next descent, and flushes the tail at the CHUNK boundary — so
    ``stats()``/``pv()``/``walk()`` between chunks always see fully backed-up
    trees. Per-world trees stay arithmetically identical to the sequential
    chunked search (bit-identical for evaluators without ``evaluate_batch``);
    ignored when ``rollout_turns > 0``, like the fixed-budget path.
    """

    def __init__(self, env: SearchRoboMageEnv, evaluator: Evaluator, *,
                 worlds: int = 4, c_puct: float = 1.5, max_depth: int = 60,
                 rollout_turns: int = 0,
                 rng: Optional[np.random.Generator] = None,
                 snapshot_slot: int = 0,
                 world_seeds: Optional[Sequence[int]] = None,
                 merge_dupes: bool = True,
                 cross_world: bool = False):
        _check_world_seeds(world_seeds, worlds)
        rng = rng if rng is not None else np.random.default_rng()
        self._env = env
        self._evaluator = evaluator
        self._c_puct = c_puct
        self._max_depth = max_depth
        self._rollout_turns = rollout_turns
        self._slot = snapshot_slot
        self._worlds = worlds
        # Cross-world deferral state (see the class docstring): inert unless
        # requested AND the budget has rollouts off, like run_search.
        self._cross = bool(cross_world) and rollout_turns == 0
        self._pending: list = []
        self._world_pending = [False] * worlds
        self.root_obs = env._obs.copy()
        self.num_choices = env._num_choices
        self.root_is_a = bool(self.root_obs[_SELF_IS_A_IDX] > 0.5)
        priors, net_value = evaluator.evaluate(self.root_obs, self.num_choices)
        self.priors = priors
        self.net_value = float(net_value)
        self._rollout_anchor = _rollout_anchor(self.root_obs)
        self._merge_dupes = bool(merge_dupes)
        root_rep = (menu_merge_reps(self.root_obs, self.num_choices)
                    if merge_dupes else None)
        env.snapshot(snapshot_slot)
        # Same derivation as run_search (no noise, no reuse), so with the same
        # rng state and no pinned world_seeds this draws the IDENTICAL seeds —
        # pinned by test_analysis_session.test_drawn_seed_parity.
        self.seeds, self.roots, _, _ = _init_worlds(
            worlds, world_seeds, rng, self.num_choices, priors, self.root_is_a,
            root_rep=root_rep)
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
            if self._cross and self._world_pending[w]:
                # This world's previous leaf must be expanded + backed up
                # before its next descent (the cross-world identity rule).
                _flush_pending(self._evaluator, self._pending,
                               self._world_pending, self._merge_dupes)
            env.restore(self._slot)
            q = env.determinize(self.seeds[w])
            _check_root_query(q, self.num_choices, w)
            if self._cross:
                pl, steps = _simulate_defer(env, self.roots[w], self._c_puct,
                                            self._max_depth)
                if pl is not None:
                    self._pending.append(pl)
                    self._world_pending[w] = True
            else:
                _, steps = _simulate(env, self._evaluator, self.roots[w],
                                     self._c_puct, self._max_depth,
                                     self._rollout_turns, self._rollout_anchor,
                                     merge_dupes=self._merge_dupes)
            self.sims_run += 1
            self.sim_steps += steps
        if self._cross:
            # Chunk boundary: flush the tail so stats()/pv()/walk() (and a
            # possible close()) always see fully backed-up trees.
            _flush_pending(self._evaluator, self._pending, self._world_pending,
                           self._merge_dupes)
        return self.stats()

    def stats(self) -> LiveStats:
        n = self.num_choices
        visits, w_sum, world_values, _, world_visits = _aggregate_roots(
            self.roots, n, want_world_visits=True)
        total = visits.sum()
        return LiveStats(
            num_choices=n,
            sims_run=self.sims_run,
            sim_steps=self.sim_steps,
            visits=visits,
            priors=np.array(self.priors, dtype=np.float64, copy=True),
            q=np.where(visits > 0, w_sum / np.maximum(visits, 1), 0.0),
            w_sum=w_sum,
            # VERBATIM formula, deliberately NOT _aggregate_roots' value_acc:
            # pinned against analysis_session._merged_stats by an exact `!=`
            # comparison (test_analysis_session.test_parallel_analysis_parity).
            root_value=float(w_sum.sum()) / total if total > 0 else 0.0,
            net_value=self.net_value,
            world_values=world_values,
            world_visits=world_visits,
        )

    def result(self) -> SearchResult:
        """The current stats as a plain SearchResult (run_search-compatible).

        ``w_sum`` is the stats' EXACT per-root W sum (not the lossy ``q *
        visits`` reconstruction), so a merged/derived Q recomputed from it is
        bit-identical to this search's own ``q``."""
        # roots/seeds/reused_visits/memo_hits/stopped_early are deliberately
        # left at their SearchResult defaults: this view is for consumers of the
        # aggregate stats, not for boundary persistence or parity replay.
        s = self.stats()
        return SearchResult(
            visits=s.visits, priors=s.priors, root_value=s.root_value,
            num_choices=s.num_choices, sims_run=s.sims_run,
            sim_steps=s.sim_steps, q=s.q,
            w_sum=s.w_sum,
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
            # A caller may name a merged-away duplicate (e.g. clicking its
            # zero-visit row in the analysis table) — show its group's line.
            if node.rep is not None and 0 <= a < node.num_choices:
                a = int(node.rep[a])
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


class IncrementalPlanSearch:
    """The sideboard plan search split into caller-paced chunks, for the
    analysis window — the plan-search twin of :class:`IncrementalSearch`.

    Same candidate order as :func:`run_plan_search` (coverage ascending, then
    the deterministic extras schedule), same value arithmetic, so a finished
    incremental search prices the same plans as the one-shot function under
    the same ``world_seeds``. Each :meth:`run_chunk` call processes WHOLE
    plans (a plan is generated and priced on every world atomically) until
    roughly ``n_sims`` more evaluator calls have been spent; ``done`` reports
    when the candidate schedule is exhausted. The root snapshot is held open
    across chunks so :meth:`pv`/:meth:`walk` can browse afterwards; the owner
    MUST call :meth:`close` before the env's next real step.

    ``merge_dupes``/``cross_world``/``max_depth``/``c_puct`` are accepted for
    construction-site symmetry with :class:`IncrementalSearch` and are inert —
    a plan search has no tree to merge, descend, or defer."""

    def __init__(self, env: SearchRoboMageEnv, evaluator: Evaluator, *,
                 worlds: int = 4, branches: int = 8, rollout_turns: int = 6,
                 rng: Optional[np.random.Generator] = None,
                 snapshot_slot: int = 0,
                 world_seeds: Optional[Sequence[int]] = None,
                 c_puct: float = 1.5, max_depth: int = 0,
                 merge_dupes: bool = True, cross_world: bool = False):
        _check_world_seeds(world_seeds, worlds)
        rng = rng if rng is not None else np.random.default_rng()
        self._env = env
        self._evaluator = evaluator
        self._slot = snapshot_slot
        self._worlds = max(1, int(worlds))
        self._branches = max(0, int(branches))
        self._rollout_turns = int(rollout_turns)
        self.root_obs = env._obs.copy()
        self.num_choices = env._num_choices
        if not self.root_obs[_IS_SIDEBOARD_IDX] > 0.5:
            raise ValueError(
                "IncrementalPlanSearch: env is not at a sideboard prompt")
        self.root_is_a = bool(self.root_obs[_SELF_IS_A_IDX] > 0.5)
        priors, net_value = evaluator.evaluate(self.root_obs, self.num_choices)
        self.priors = priors
        self.net_value = float(net_value)
        self.seeds = [int(world_seeds[w]) if world_seeds is not None
                      else int(rng.integers(1, 2**31 - 1))
                      for w in range(self._worlds)]
        env.snapshot(snapshot_slot)
        self.plans: list[SbPlan] = []
        self._memo: dict = {}
        self._evaluated: set = set()
        self._counters = {"steps": 0, "evals": 0, "memo_hits": 0}
        self._cursor = 0            # next coverage action while < num_choices
        self._ranked: Optional[np.ndarray] = None
        self._extra_attempt = 0
        self._novel_extras = 0
        self.sims_run = 0           # evaluator calls (the pacing unit)
        self.sim_steps = 0
        self._closed = False

    # -- schedule ------------------------------------------------------------

    @property
    def done(self) -> bool:
        return (self._cursor >= self.num_choices
                and (self._novel_extras >= self._branches
                     or self._extra_attempt >= self._branches * 4))

    def _run_one_plan(self, first_action: int, variant: int) -> bool:
        env = self._env
        counters = self._counters
        env.restore(self._slot)
        q0 = env.determinize(self.seeds[0])
        _check_root_query(q0, self.num_choices, 0)
        pick_actions, picks, handoff = _generate_plan(
            env, self._evaluator, root_obs=self.root_obs,
            first_action=first_action, variant=variant,
            root_is_a=self.root_is_a, counters=counters)
        multiset = tuple(sorted(picks))
        memoable = rollout_memo_eligible(multiset)
        novel = not (memoable and multiset in self._evaluated)
        values = np.zeros(self._worlds, dtype=np.float64)
        for w in range(self._worlds):
            key = (self.seeds[w], multiset)
            if memoable:
                hit = self._memo.get(key)
                if hit is not None:
                    counters["memo_hits"] += 1
                    values[w] = hit
                    continue
            q = handoff if w == 0 else _replay_plan(
                env, self.num_choices, w, self.seeds[w], self._slot,
                pick_actions, counters)
            values[w] = _eval_plan_world(
                env, self._evaluator, query=q, root_is_a=self.root_is_a,
                rollout_turns=self._rollout_turns, counters=counters)
            if memoable:
                self._memo[key] = float(values[w])
        self.plans.append(SbPlan(
            first_action=int(first_action), pick_actions=pick_actions,
            picks=picks, multiset=multiset, values=values, variant=variant))
        if memoable:
            self._evaluated.add(multiset)
        return novel

    def run_chunk(self, n_sims: int) -> LiveStats:
        """Process whole plans until ~``n_sims`` more evaluator calls have
        been spent (or the schedule is exhausted); returns updated stats."""
        if self._closed:
            raise RuntimeError("IncrementalPlanSearch already closed")
        target = self._counters["evals"] + max(0, int(n_sims))
        while not self.done and self._counters["evals"] < target:
            if self._cursor < self.num_choices:
                self._run_one_plan(self._cursor, 0)
                self._cursor += 1
                continue
            if self._ranked is None:
                cov_q = np.zeros(self.num_choices, dtype=np.float64)
                for p in self.plans:
                    if p.variant == 0:
                        cov_q[p.first_action] = p.value
                self._ranked = np.argsort(-cov_q, kind="stable")
            branch = int(self._ranked[self._extra_attempt % self.num_choices])
            variant = 1 + self._extra_attempt // self.num_choices
            self._extra_attempt += 1
            if self._run_one_plan(branch, variant):
                self._novel_extras += 1
        self.sims_run = self._counters["evals"]
        self.sim_steps = self._counters["steps"]
        return self.stats()

    # -- views ---------------------------------------------------------------

    def stats(self) -> LiveStats:
        n = self.num_choices
        best = np.full(n, -np.inf, dtype=np.float64)
        for p in self.plans:
            best[p.first_action] = max(best[p.first_action], p.value)
        covered = np.isfinite(best)
        q = np.where(covered, best, 0.0)
        visits = np.zeros(n, dtype=np.float64)
        if covered.any():
            # softmax over the covered subset only (mid-coverage chunks);
            # equals _plan_softmax once coverage completes.
            z = (q[covered] - q[covered].max()) / SB_PI_TAU
            e = np.exp(z)
            n_evals = len(self.plans) * self._worlds
            visits[covered] = (e / e.sum()) * float(n_evals)
        pi = visits / visits.sum() if visits.sum() > 0 else visits
        world_visits = np.zeros((self._worlds, n), dtype=np.int64)
        world_values = np.zeros(self._worlds, dtype=np.float64)
        for w in range(self._worlds):
            if self.plans:
                world_values[w] = float(
                    np.mean([p.values[w] for p in self.plans]))
            for p in self.plans:
                world_visits[w, p.first_action] += 1
        return LiveStats(
            num_choices=n,
            sims_run=self.sims_run,
            sim_steps=self.sim_steps,
            visits=visits,
            priors=np.array(self.priors, dtype=np.float64, copy=True),
            q=q,
            w_sum=q * visits,
            root_value=float(pi @ q) if visits.sum() > 0 else 0.0,
            net_value=self.net_value,
            world_values=world_values,
            world_visits=world_visits,
        )

    def result(self) -> SearchResult:
        s = self.stats()
        return SearchResult(
            visits=s.visits, priors=s.priors, root_value=s.root_value,
            num_choices=s.num_choices, sims_run=s.sims_run,
            sim_steps=s.sim_steps, q=s.q, w_sum=s.w_sum,
            world_values=s.world_values, seeds=list(self.seeds))

    def best_plan_for(self, action: int) -> Optional[SbPlan]:
        """The highest-value plan whose first pick is ``action`` (None until
        that action's coverage plan has run)."""
        cand = [p for p in self.plans if p.first_action == int(action)]
        if not cand:
            return None
        return max(cand, key=lambda p: p.value)

    def pv(self, action: int, world: int, max_len: int = 24) -> list:
        """The 'principal variation' of a first pick: its best plan's pick
        sequence. Each step carries the PLAN's value on ``world`` (a plan is
        priced as a whole — there are no per-step statistics)."""
        plan = self.best_plan_for(action)
        if plan is None:
            return []
        v = float(plan.values[int(world)])
        return [PVStep(action=int(a), visits=self._worlds, q=v,
                       seat_is_a=self.root_is_a)
                for a in plan.pick_actions[:max(0, int(max_len))]]

    def walk(self, world: int, actions: Sequence[int]) -> list:
        """Identical contract to :meth:`IncrementalSearch.walk`."""
        if self._closed:
            raise RuntimeError("IncrementalPlanSearch already closed")
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
        if self._closed:
            return
        self._closed = True
        self._env.restore(self._slot)
        self._env.release()
