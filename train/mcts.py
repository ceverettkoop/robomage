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

    def evaluate(self, obs: np.ndarray, num_choices: int) -> tuple[np.ndarray, float]:
        torch = self._torch
        policy = self._model.policy
        mask = np.zeros(self._max_actions, dtype=bool)
        mask[:num_choices] = True
        with torch.no_grad():
            obs_t, _ = policy.obs_to_tensor(obs)
            dist = policy.get_distribution(obs_t, action_masks=mask)
            probs = dist.distribution.probs.detach().cpu().numpy()[0][:num_choices]
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
) -> SearchResult:
    """Search the env's current decision. The env must be parked at a real,
    loop-safe decision (env.last_search_safe). On return the env is back at
    that same decision with all snapshots released, ready for the chosen real
    step().

    ``world_seeds`` (optional): when given, the per-world determinize seeds are
    consumed from it in order instead of drawn from ``rng`` — used for
    reproducible cross-implementation parity (the C++ actor derives the same
    seeds by a shared formula). Default ``None`` keeps the original rng draw."""
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

    for w in range(worlds):
        world_seed = (int(world_seeds[w]) if world_seeds is not None
                      else int(rng.integers(1, 2**31 - 1)))
        priors = root_priors
        if root_noise_eps > 0.0:
            noise = rng.dirichlet([root_noise_alpha] * root_n)
            priors = (1.0 - root_noise_eps) * root_priors + root_noise_eps * noise
        root = _Node(root_n, priors, root_is_a)

        for _ in range(sims_per_world):
            env.restore(snapshot_slot)
            env.determinize(world_seed)
            v = _simulate(env, evaluator, root, c_puct, max_depth)
            sims_run += 1
            sim_steps += v[1]

        visit_totals += root.N
        value_acc += float(root.W.sum())

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
