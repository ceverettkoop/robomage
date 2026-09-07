"""Rebuild, verify, cache and browse the MCTS tree of a recorded decision.

A recorded session (``shard_record.ShardRecorder``) keeps, per searched
opponent decision, the root diagnostics the trainer schema omits — per-world
determinize seeds, the sim count, the visit vector — plus the controller's
static ``search_provenance`` in the ``.rmplay`` sidecar. In-game
``SearchController`` searches never reuse roots, add no root noise and pick at
temperature 0, and both ``mcts.run_search``'s loops (fixed-budget or timed)
and ``mcts.IncrementalSearch.run_chunk`` grow each world's PRIVATE tree by the
same per-world sim counts under a round-robin from world 0. So replaying the
recording to the decision and running
``IncrementalSearch(...).run_chunk(sims_run)`` with the recorded seeds and the
same evaluator reproduces the played tree bit-for-bit — which :class:`TreeSession`
checks by comparing the rebuilt root visits with the recorded ones exactly.

The rebuilt tree is cached beside the recording (``tree_cache``) so a second
look installs the saved roots instead of searching again; the cache keeps its
own ``verified`` flag (the tree is the played tree, independent of whatever
checkpoint is on disk later). ``IncrementalSearch`` holds the root snapshot
open, so the session can read principal variations, per-node statistics and
walk hypothetical lines on the engine. Qt-free; torch and the engine are
imported lazily (a cache hit never needs torch).
"""
from __future__ import annotations

import datetime
import os

import numpy as np

from tree_cache import load_tree, save_tree, tree_node_count

DIAG_KIND_SEARCH = 1
DIAG_KIND_FOLLOWED = 2
DIAG_KIND_PLAN = 3

_OBS_ATOL = 1e-4


class RebuildError(Exception):
    """A decision whose tree cannot be rebuilt (reason in ``str(e)``)."""


def evaluator_spec_for(prov: dict) -> str:
    """The ``analysis_session.load_analysis_evaluator`` spec that reproduces a
    recording's evaluator: the resolved checkpoint by extension (``.pt`` ->
    ``az:``, ``.zip`` -> ``mcts:``), ``uniform`` for the torch-free evaluator,
    else the controller spec's base with its knob query stripped."""
    prov = prov or {}
    ckpt = prov.get("checkpoint")
    if isinstance(ckpt, str) and ckpt:
        ext = os.path.splitext(ckpt)[1].lower()
        if ext == ".zip":
            return f"mcts:{ckpt}"
        return f"az:{ckpt}"
    spec = str(prov.get("spec") or "uniform")
    spec = spec.partition("?")[0].strip()
    prefix, sep, base = spec.partition(":")
    if not sep:
        prefix, base = "", prefix
    if base.strip().lower() == "uniform":
        return "uniform"
    return f"{prefix}:{base}" if prefix else base


def cache_path_for(shard_stem: str, row: int) -> str:
    """``<dir>/trees/<shard basename>_row<row>.npz`` beside the shard."""
    d, base = os.path.split(str(shard_stem))
    return os.path.join(d, "trees", f"{base}_row{int(row)}.npz")


def _game_is_replayable(game) -> bool:
    return (game.get("engine_seed") is not None
            and game.get("full_actions") is not None
            and game.get("prefix_len") is not None)


def replay_to_step(game, step, *, binary, deck_a, deck_b, bo3, strict=True):
    """Replay ``game`` on a fresh ``SearchRoboMageEnv`` to model decision
    ``step`` (the recorded engine seed + the action prefix). Returns the env
    parked at that decision; the caller owns it. ``RebuildError`` when the
    game carries no replay data, the replay ends inside the prefix, or (with
    ``strict``) the reached obs differs from the recorded one."""
    from search_env import SearchRoboMageEnv

    if not _game_is_replayable(game):
        raise RebuildError("game has no recorded seed/action log")
    prefix = game["prefix_len"][step]
    if prefix is None:
        raise RebuildError(f"step {step} has no recorded replay position")
    env = SearchRoboMageEnv(binary_path=binary, deck_a=deck_a, deck_b=deck_b,
                            bo3=bool(bo3))
    try:
        obs, _ = env.reset(options={"engine_seed": game["engine_seed"]})
        for a in game["full_actions"][:int(prefix)]:
            obs, _r, term, trunc, _ = env.step(int(a))
            if term or trunc:
                raise RebuildError(
                    f"replay ended inside the action prefix before step {step}")
        expected = np.asarray(game["observations"][step], dtype=np.float32)
        if strict and not np.allclose(obs, expected, atol=_OBS_ATOL):
            n_diff = int(np.sum(~np.isclose(obs, expected, atol=_OBS_ATOL)))
            raise RebuildError(
                f"replay diverged from the recorded state at step {step} "
                f"({n_diff} obs floats differ)")
    except BaseException:
        env.close()
        raise
    return env


class _RecordedRootEvaluator:
    """Root-only evaluator for a cache hit: hands ``IncrementalSearch`` the
    RECORDED root priors (the tree's own P is loaded from the cache) so no
    checkpoint is needed. The value is NaN — the recording keeps no raw net
    value, and nothing on the cached path ever runs a simulation."""

    def __init__(self, priors):
        self._priors = np.asarray(priors, dtype=np.float64)

    def evaluate(self, obs, num_choices):
        return self._priors[:int(num_choices)].copy(), float("nan")


def _first_visit_mismatch(recorded, rebuilt) -> str | None:
    rec = np.asarray(recorded, dtype=np.int64).reshape(-1)
    reb = np.asarray(rebuilt, dtype=np.int64).reshape(-1)
    if rec.shape != reb.shape:
        return f"menu length differs: recorded {len(rec)}, rebuilt {len(reb)}"
    diff = np.flatnonzero(rec != reb)
    if len(diff) == 0:
        return None
    i = int(diff[0])
    return (f"action {i}: recorded {int(rec[i])} visits, rebuilt {int(reb[i])}"
            f" ({len(diff)} of {len(rec)} actions differ)")


def _rep_index(node, action: int) -> int:
    """Canonicalize ``action`` to its duplicate-group representative."""
    a = int(action)
    if node.rep is not None and 0 <= a < node.num_choices:
        return int(node.rep[a])
    return a


def _descend(root, path):
    """The node reached from ``root`` by ``path`` (rep-canonicalized), or None
    where the tree was never expanded along it."""
    node = root
    for a in path:
        node = node.children.get(_rep_index(node, a))
        if node is None:
            return None
    return node


def _action_labels(obs, num_choices) -> list[str]:
    import decode
    acts = decode.decode_actions_from_obs(obs, int(num_choices))
    return [str(a.get("description", f"#{i}")) for i, a in enumerate(acts)]


def _set_torch_threads(n) -> None:
    """Pin torch's intra-op threads to the recording's count (batched GEMM
    reductions can differ by thread count); a no-op without torch."""
    if n is None:
        return
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(n))


class TreeSession:
    """The rebuilt (or cached) search tree of one recorded decision, with the
    engine parked at its root for browsing.

    ``game`` is a ``shard_replay.load_records`` record (``diag``,
    ``origin_step``, ``row_index``, ``diag_prov``, ``shard_stem`` plus the
    replay fields); ``step`` indexes its decisions. A tree-FOLLOWED step
    (kind 2) resolves to its ORIGIN search's tree, with ``follow_path`` /
    ``follow_worlds`` naming where inside it the followed decision sits.
    ``evaluator`` overrides the evaluator loaded from the provenance;
    ``prov`` overrides the record's ``diag_prov``; ``cache_dir`` overrides
    the ``trees/`` dir beside the shard. Call :meth:`open`, browse, then
    :meth:`close` (releases the snapshot and the engine)."""

    def __init__(self, game, step, *, binary, deck_a, deck_b, bo3,
                 evaluator=None, prov=None, cache_dir=None, use_cache=True):
        self._game = game
        self._step = int(step)
        self._binary = binary
        self._deck_a = deck_a
        self._deck_b = deck_b
        self._bo3 = bool(bo3)
        self._evaluator = evaluator
        self._prov = dict(prov) if prov is not None else None
        self._cache_dir = cache_dir
        self._use_cache = bool(use_cache)
        self._env = None
        self._search = None
        self.verified = False
        self.from_cache = False
        self.diag = None
        self.root_step = self._step
        self.follow_path: list[int] = []
        self.follow_worlds: list[int] = []
        self.seeds: list[int] = []
        self.worlds = 0
        self.sims_run = 0
        self.mismatch: str | None = None
        self.evaluator_spec: str | None = None
        self.cache_path: str | None = None

    # ----- open / close -----

    def _resolve_root(self) -> dict:
        """Pick the searched (kind 1) step whose tree this session rebuilds
        and fill the follow bookkeeping. Returns that step's diag."""
        game = self._game
        diags = game.get("diag") or []
        step = self._step
        if not (0 <= step < len(diags)) or diags[step] is None:
            raise RebuildError(f"step {step} recorded no search")
        d = diags[step]
        kind = int(d.get("kind", 0))
        if kind == DIAG_KIND_PLAN:
            raise RebuildError("plan search has no tree")
        if kind == DIAG_KIND_FOLLOWED:
            origins = game.get("origin_step") or []
            origin = origins[step] if step < len(origins) else None
            if origin is None or diags[origin] is None \
                    or int(diags[origin].get("kind", 0)) != DIAG_KIND_SEARCH:
                raise RebuildError(
                    f"followed step {step} has no searched origin in this record")
            self.root_step = int(origin)
            self.follow_path = [int(a) for a in d.get("follow_path") or []]
            self.follow_worlds = [int(w) for w in d.get("follow_worlds") or []]
            return diags[origin]
        if kind != DIAG_KIND_SEARCH:
            raise RebuildError(f"step {step} has diag kind {kind}, not a search")
        self.root_step = step
        self.follow_path = []
        self.follow_worlds = list(range(int(d.get("n_worlds") or
                                            len(d.get("world_seeds") or []))))
        return d

    def _cache_file(self) -> str | None:
        rows = self._game.get("row_index") or []
        row = rows[self.root_step] if self.root_step < len(rows) else None
        if row is None:
            return None
        if self._cache_dir:
            stem = self._game.get("shard_stem") or "shard"
            return os.path.join(self._cache_dir,
                                os.path.basename(cache_path_for(stem, row)))
        stem = self._game.get("shard_stem")
        return cache_path_for(stem, row) if stem else None

    def open(self) -> "TreeSession":
        from mcts import IncrementalSearch

        diag = self._resolve_root()
        self.diag = diag
        prov = self._prov if self._prov is not None \
            else dict(self._game.get("diag_prov") or {})
        timed = diag.get("time_budget_s") is not None
        if int(prov.get("procs") or 1) > 1 and timed:
            raise RebuildError(
                "recorded with a timed budget across procs>1 — the per-world "
                "sim split is not reproducible")
        self.seeds = [int(s) for s in diag.get("world_seeds") or []]
        self.worlds = int(diag.get("n_worlds") or len(self.seeds))
        if self.worlds <= 0 or len(self.seeds) < self.worlds:
            raise RebuildError("recording carries no per-world seeds")
        self.sims_run = int(diag.get("sims_run") or 0)
        c_puct = float(prov.get("c_puct", 1.5))
        merge_dupes = bool(prov.get("merge_dupes", True))
        cross_world = bool(prov.get("cross_world", True))
        self.evaluator_spec = evaluator_spec_for(prov)
        recorded = np.asarray(diag["visits"], dtype=np.int64)

        self.cache_path = self._cache_file() if self._use_cache else None
        cached = None
        if self.cache_path and os.path.exists(self.cache_path):
            try:
                cached = load_tree(self.cache_path)
            except ValueError:
                cached = None

        env = replay_to_step(self._game, self.root_step, binary=self._binary,
                             deck_a=self._deck_a, deck_b=self._deck_b,
                             bo3=self._bo3)
        self._env = env
        try:
            if not getattr(env, "last_search_safe", False):
                raise RebuildError("decision is not a legal search root")
            n = int(env._num_choices)
            if n != int(diag.get("num_choices", n)):
                raise RebuildError(
                    f"menu length {n} differs from the recorded "
                    f"{diag.get('num_choices')}")
            if cached is not None:
                evaluator = _RecordedRootEvaluator(diag["priors"])
            elif self._evaluator is not None:
                evaluator = self._evaluator
            else:
                from analysis_session import load_analysis_evaluator
                evaluator, _label = load_analysis_evaluator(
                    self.evaluator_spec, device=prov.get("device"))
                _set_torch_threads(prov.get("torch_threads"))
            search = IncrementalSearch(
                env, evaluator, worlds=self.worlds, c_puct=c_puct,
                world_seeds=self.seeds[:self.worlds],
                merge_dupes=merge_dupes, cross_world=cross_world)
            self._search = search
            if cached is not None:
                roots, meta = cached
                if len(roots) != self.worlds:
                    raise RebuildError(
                        f"cached tree has {len(roots)} worlds, recording "
                        f"has {self.worlds}")
                search.roots = roots
                search.sims_run = int(meta.get("sims_run", self.sims_run))
                search.sim_steps = int(diag.get("sim_steps") or 0)
                self.from_cache = True
                self.verified = bool(meta.get("verified", False))
                self.mismatch = meta.get("mismatch")
            else:
                stats = search.run_chunk(self.sims_run)
                rebuilt = stats.visits.astype(np.int64)
                self.mismatch = _first_visit_mismatch(recorded, rebuilt)
                self.verified = self.mismatch is None
                if self.cache_path:
                    self._save_cache(prov, recorded, rebuilt, c_puct,
                                     merge_dupes, cross_world)
        except BaseException:
            self.close()
            raise
        return self

    def _save_cache(self, prov, recorded, rebuilt, c_puct, merge_dupes,
                    cross_world) -> None:
        meta = {
            "verified": bool(self.verified),
            "mismatch": self.mismatch,
            "recorded_visits": [int(v) for v in recorded],
            "rebuilt_visits": [int(v) for v in rebuilt],
            "seeds": list(self.seeds[:self.worlds]),
            "sims_run": int(self.sims_run),
            "worlds": int(self.worlds),
            "c_puct": float(c_puct),
            "merge_dupes": bool(merge_dupes),
            "cross_world": bool(cross_world),
            "evaluator_spec": self.evaluator_spec,
            "checkpoint_sha256": prov.get("checkpoint_sha256"),
            "device": prov.get("device"),
            "torch_threads": prov.get("torch_threads"),
            "rebuilt_at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        save_tree(self.cache_path, self._search.roots, meta)

    def close(self) -> None:
        """Release the root snapshot and the engine. Idempotent."""
        search, self._search = self._search, None
        if search is not None:
            try:
                search.close()
            except Exception:  # noqa: BLE001 — the engine is going away anyway
                pass
        env, self._env = self._env, None
        if env is not None:
            env.close()

    # ----- browsing -----

    def _require_open(self):
        if self._search is None:
            raise RebuildError("TreeSession is not open")
        return self._search

    def root_stats(self):
        """``mcts.LiveStats`` of the tree's root (summed across worlds)."""
        return self._require_open().stats()

    def node_stats(self, world, path) -> list[tuple[int, int, float, float]]:
        """``(action, N, Q, P)`` per representative action at the node
        ``path`` reaches in ``world``'s tree; Q in that node's mover
        perspective. ``[]`` where the tree never expanded that far."""
        search = self._require_open()
        node = _descend(search.roots[int(world)], path)
        if node is None:
            return []
        out = []
        for i in range(int(node.num_choices)):
            if node.rep is not None and int(node.rep[i]) != i:
                continue
            n = int(node.N[i])
            q = float(node.W[i]) / n if n > 0 else 0.0
            out.append((i, n, q, float(node.P[i])))
        return out

    def walk(self, world, path) -> tuple[list, list[str]]:
        """Play ``path`` from the root inside ``world``'s determinization
        (``IncrementalSearch.walk``): the ``WalkNode`` per step and the
        action labels of the reached decision's menu (``[]`` at a terminal
        or an unreachable position)."""
        search = self._require_open()
        nodes = search.walk(int(world), [int(a) for a in path])
        if nodes:
            last = nodes[-1]
            if last.terminal is not None or last.obs is None:
                return nodes, []
            return nodes, _action_labels(last.obs, last.num_choices)
        if path:
            return nodes, []
        return nodes, self.root_labels()

    def pv(self, action, world, max_len=24) -> list:
        """``mcts.PVStep`` list for ``action`` at ``world``'s root."""
        return self._require_open().pv(int(action), int(world), int(max_len))

    def root_labels(self) -> list[str]:
        search = self._require_open()
        return _action_labels(search.root_obs, search.num_choices)

    def summary(self) -> str:
        search = self._require_open()
        state = "verified" if self.verified else f"MISMATCH ({self.mismatch})"
        source = "cache hit" if self.from_cache else "rebuilt"
        follow = (f", followed {len(self.follow_path)} deep from step "
                  f"{self.root_step}" if self._step != self.root_step else "")
        return (f"{state}: {self.sims_run} sims x {self.worlds} worlds, "
                f"{source}, {tree_node_count(search.roots)} nodes, "
                f"evaluator {self.evaluator_spec}{follow}")

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False
