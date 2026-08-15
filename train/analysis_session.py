"""Engine-side core of the GUI analysis window (Qt-free).

An AnalysisSession owns one or more DETACHED analysis engines — byte-identical
copies of the live session's engine process (see
SearchRoboMageEnv.spawn_detached_mirror) that are never registered in the
primary's mirror pool, so the live game stays fully responsive while MCTS
grinds here on a worker thread. The analysis engines are kept in lockstep
lazily: each real decision's StateUpdate carries the primary's action-history
length, and sync() replays just the delta into every engine before a new
analysis, verifying each replayed obs byte-equals the decision being analyzed.
A request for an EARLIER decision than the engines currently hold (reviewing a
move already played — the window's "opponent's last decision") is served by
respawning them at that history prefix, since replay only runs forward.

analyze() runs an mcts.IncrementalSearch in caller-paced chunks (live UI
updates via on_update, cancellation via a threading.Event) and leaves the
search OPEN — root snapshot held — so pv()/walk() can browse principal
variations and replay hypothetical lines until the next sync()/close().

WORLD-PARALLEL analysis (AnalysisConfig.procs): a search's determinized worlds
are fully independent (own root, own seed, own per-sim restore+determinize), so
they split contiguously across the engines exactly the way run_search_parallel
fans them across the mirror pool — one IncrementalSearch per engine over its
world slice, all sharing one lock-serialized evaluator, run one chunk at a time
under a per-chunk barrier so on_update always sees a coherent merged snapshot.
All ``worlds`` seeds are pre-drawn from the run's rng in the canonical order, so
the merged stats equal a single-engine run with the same cfg.seed world for
world (visits/world_values bit-identically; the value sums up to float addition
association, which is exact whenever the per-world sums are).

Threading contract: every method here must be called from ONE thread (the
analysis worker). The only cross-thread touch is reading a prefix of the
primary env's append-only _action_history list, which is safe under the GIL.

Torch-free unless a model evaluator is requested: "uniform" runs without torch
(tests), "az:gen" / "mcts:gen" lazily import the checkpoint machinery.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from cli_spec import DEFAULT_SB_MAX_DEPTH, DEFAULT_SB_ROLLOUT_TURNS
from env import _IS_SIDEBOARD_IDX, _SELF_IS_A_IDX
from mcts import IncrementalSearch, LiveStats, UniformEvaluator, _LockedEvaluator


class AnalysisError(RuntimeError):
    """A refused or failed analysis request (reason in str(e))."""


def load_analysis_evaluator(spec: str):
    """Build the evaluator behind an analysis-window spec. Returns
    (evaluator, label).

      - "uniform"          -> UniformEvaluator (torch-free; tests/fallback)
      - "az:<base>" / bare -> AZ net via opponents._load_az_evaluator: the AZ
                              checkpoint when one exists, else an AZNet
                              warm-started from the PPO checkpoint. Default
                              base is "gen" (the one generalist).
      - "mcts:<base>"      -> PPOEvaluator over the MaskablePPO checkpoint.
    """
    spec = (spec or "az:gen").strip()
    if spec.lower() == "uniform":
        return UniformEvaluator(), "uniform"
    if spec.startswith("mcts:"):
        from mcts import PPOEvaluator
        from opponents import _load_model, resolve_checkpoint

        base = spec.split(":", 1)[1] or "gen"
        return PPOEvaluator(_load_model(resolve_checkpoint(base))), f"mcts:{base}"
    from opponents import _load_az_evaluator

    base = spec.split(":", 1)[1] if spec.startswith("az:") else spec
    evaluator, resolved = _load_az_evaluator(base or "gen")
    return evaluator, f"az:{resolved}"


@dataclass
class AnalysisConfig:
    """Knobs for the analysis window's search (launcher-configurable)."""
    evaluator_spec: str = "az:gen"
    worlds: int = 4
    chunk_sims: int = 16     # sims per UI update (~100ms/chunk at ~6ms/sim)
    max_sims: int = 800      # total per run; 0 = run until stopped
    c_puct: float = 2.5
    max_depth: int = 60
    sb_max_depth: int = DEFAULT_SB_MAX_DEPTH  # sideboard roots: descent depth cap
    sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS  # sideboard roots: leaf-rollout horizon (player turns; 0 = off)
    seed: int = 0            # search rng seed; 0 = fresh entropy per run
    auto_analyze: bool = True  # UI: start a run at every new analyzable decision
    merge_dupes: bool = True   # merge interchangeable duplicate menu actions
    # Detached analysis ENGINES the worlds are fanned across: 0 = auto
    # (opponents.default_search_procs — half the cores, capped at `worlds`),
    # 1 = one engine (the original serial behavior), N = N engines. More
    # engines than worlds would idle, so the count is clamped to `worlds`.
    procs: int = 0

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Reject knob values that would misbehave silently. Re-checked at
        analyze() time because the window edits some fields live."""
        if int(self.worlds) < 1:
            raise ValueError(
                f"AnalysisConfig.worlds must be >= 1 (got {self.worlds})")
        if int(self.chunk_sims) < 1:
            raise ValueError(
                f"AnalysisConfig.chunk_sims must be >= 1 (got "
                f"{self.chunk_sims}): a non-positive chunk runs no sims, so an "
                f"uncapped analyze() would spin forever")
        if int(self.procs) < 0:
            raise ValueError(
                f"AnalysisConfig.procs must be >= 0 (got {self.procs}; "
                f"0 = auto)")


def world_bounds(worlds: int, n_engines: int) -> List[Tuple[int, int]]:
    """Contiguous [lo, hi) world slice per engine, remainder spread over the
    first engines — the same split run_search_parallel uses, so engine order ==
    world order and merged per-world arrays keep their global world index."""
    n = max(1, min(int(n_engines), int(worlds)))
    out: List[Tuple[int, int]] = []
    start = 0
    for i in range(n):
        k_i = worlds // n + (1 if i < worlds % n else 0)
        out.append((start, start + k_i))
        start += k_i
    return out


def _slice_target(total: int, lo: int, hi: int, worlds: int) -> int:
    """How many of the first ``total`` sims of a GLOBAL round-robin over
    ``worlds`` worlds land in the world slice [lo, hi).

    IncrementalSearch round-robins its own worlds in order, so handing each
    engine ``target(T+n) - target(T)`` sims per chunk gives every world exactly
    the sim count the single-engine round-robin would have given it — which is
    what makes the merged result equal a 1-engine run."""
    full, rem = divmod(int(total), int(worlds))
    return (hi - lo) * full + max(0, min(hi, rem) - lo)


@dataclass
class AnalysisRequest:
    """One decision to analyze, snapshotted from a StateUpdate (whose obs is
    already a private copy)."""
    obs: np.ndarray
    num_choices: int
    history_len: int
    search_safe: bool
    seat_is_a: bool          # obs perspective seat (True = Player A to move)
    is_sideboard: bool

    @classmethod
    def from_update(cls, u) -> "AnalysisRequest":
        return cls(
            obs=u.obs,
            num_choices=int(u.num_choices),
            history_len=int(u.history_len),
            search_safe=bool(u.search_safe),
            seat_is_a=bool(u.obs[_SELF_IS_A_IDX] > 0.5),
            is_sideboard=bool(u.obs[_IS_SIDEBOARD_IDX] > 0.5),
        )


class AnalysisSession:
    """Owns the detached analysis engine(s) + the open IncrementalSearch(es)."""

    def __init__(self, primary_env, cfg: AnalysisConfig,
                 evaluator=None, evaluator_label: str = ""):
        self._primary = primary_env
        self.cfg = cfg
        self._evaluator = evaluator          # lazily loaded from cfg when None
        self.evaluator_label = evaluator_label
        self._envs: List = []                # the detached analysis engines
        self._synced_len = 0                 # actions replayed into them so far
        self._searches: List[IncrementalSearch] = []   # one per engine
        self._bounds: List[Tuple[int, int]] = []       # engine -> world slice
        self._dead: Optional[str] = None     # error once the engines are unusable

    @property
    def _env(self):
        """The FIRST analysis engine (None before the first spawn) — the
        single-engine handle callers used before world-parallel analysis."""
        return self._envs[0] if self._envs else None

    @property
    def _search(self) -> Optional[IncrementalSearch]:
        """The first open search, or None (single-engine back-compat handle)."""
        return self._searches[0] if self._searches else None

    def engine_count(self) -> int:
        """Engines this session will fan the worlds across: cfg.procs, or the
        interactive auto default, clamped to [1, cfg.worlds]."""
        cfg = self.cfg
        procs = int(cfg.procs or 0)
        if procs <= 0:
            from opponents import default_search_procs   # lazy: keeps import light
            procs = int(default_search_procs(int(cfg.worlds)))
        return max(1, min(procs, int(cfg.worlds)))

    # ----- readiness -----

    def can_analyze(self, req: AnalysisRequest) -> Optional[str]:
        """None when the request is analyzable, else the human-readable
        refusal reason."""
        if self._dead:
            return f"analysis engine unavailable: {self._dead}"
        if not hasattr(self._primary, "_action_history"):
            return "session env is not search-capable (analysis disabled)"
        if not req.search_safe:
            return "not snapshot-safe (mid-resolution prompt)"
        if req.num_choices <= 1:
            return "only one legal action"
        return None

    # ----- engine lifecycle / lockstep -----

    def ensure_engine(self, req: AnalysisRequest) -> Optional[str]:
        """Lazily spawn the detached analysis engine(s) at this request's state.
        Returns None on success, else the error (session marked dead).

        With cfg.procs > 1 (or auto) this spawns the whole fleet at the same
        history prefix; a changed engine count (the window's knobs moved
        between runs) drops the old fleet and respawns."""
        if self._dead:
            return f"analysis engine unavailable: {self._dead}"
        need = self.engine_count()
        if self._envs and len(self._envs) == need:
            return None
        if self._envs:
            self._drop_engine()              # count changed: rebuild the fleet
        spawned: List = []
        try:
            for _ in range(need):
                spawned.append(self._primary.spawn_detached_mirror(
                    req.history_len, expect_obs=req.obs))
            self._envs = spawned
            self._synced_len = req.history_len
            return None
        except Exception as e:  # noqa: BLE001 — surfaced as a status, not a crash
            for env in spawned:              # never leak a half-spawned fleet
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
            self._dead = str(e)
            self._envs = []
            return f"analysis engine unavailable: {self._dead}"

    def sync(self, req: AnalysisRequest) -> Optional[str]:
        """Close any open search, bring the analysis engine to this request's
        position, and verify it matches the request's obs. Returns None on
        success, else the error (session marked dead).

        Forward is the common case: replay the real-action delta into EVERY
        engine. A request BEHIND the engines — reviewing a decision the game has
        already moved past — drops the fleet and respawns it at that history
        prefix, since an engine process only replays forward."""
        if self._dead:
            return f"analysis engine unavailable: {self._dead}"
        if self._searches:
            try:
                for s in self._searches:
                    s.close()            # restore + release BEFORE real steps
            except Exception as e:  # noqa: BLE001
                self._dead = f"search close failed: {e}"
                self._searches = []
                return f"analysis engine unavailable: {self._dead}"
            self._searches = []
        if self._envs and req.history_len < self._synced_len:
            self._drop_engine()          # rewind: respawn at the older prefix
        err = self.ensure_engine(req)
        if err:
            return err
        try:
            delta = self._primary._action_history[self._synced_len:req.history_len]
            for env in self._envs:
                for a in delta:
                    env.step(int(a))
                if not np.array_equal(env._obs, req.obs):
                    raise RuntimeError("analysis engine obs diverged from decision")
            self._synced_len = req.history_len
            return None
        except Exception as e:  # noqa: BLE001
            self._dead = str(e)
            return f"analysis engine unavailable: {self._dead}"

    def _drop_engine(self) -> None:
        """Close every analysis engine (any open search must already be
        closed). The next ensure_engine() spawns a fresh fleet at the requested
        prefix."""
        for env in self._envs:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        self._envs = []
        self._bounds = []
        self._synced_len = 0

    def respawn(self) -> None:
        """Recovery action: drop the (possibly dead) engines + searches; the
        next analyze() spawns a fresh fleet by full history replay."""
        self._searches = []                 # their envs are going away with them
        self._drop_engine()
        self._dead = None

    # ----- the search -----

    def evaluator_note(self) -> Optional[str]:
        """A caveat about the last analysis's evaluator, or None when it has
        none. Today that is the AZ net's multi-head critic reporting that THIS
        matchup's value column was never trained (or was decayed away), so its
        values are a constant that would otherwise read as a confident 50%."""
        bucket = getattr(self._evaluator, "untrained_bucket", None)
        if bucket is None:
            return None
        try:
            from archetypes import bucket_name
            what = bucket_name(int(bucket))
        except Exception:  # noqa: BLE001 — the caveat matters, the name doesn't
            what = f"bucket {bucket}"
        return (f"value head untrained for this matchup ({what}) — the net's "
                "win% is a constant here; only the priors and terminals inform "
                "this search")

    def _ensure_evaluator(self):
        if self._evaluator is None:
            self._evaluator, self.evaluator_label = load_analysis_evaluator(
                self.cfg.evaluator_spec)
        return self._evaluator

    def analyze(self, req: AnalysisRequest, *,
                stop: Optional[threading.Event] = None,
                on_update: Optional[Callable[[LiveStats], None]] = None
                ) -> LiveStats:
        """Search the request's decision in chunks until the stop event, the
        cfg.max_sims cap (0 = unlimited, counting sims across ALL engines), or
        an engine error. on_update fires after every chunk (and once up front
        with the zero-visit priors frame) with the MERGED stats of the whole
        fleet — the per-chunk barrier keeps that snapshot coherent. The
        search(es) are left OPEN for pv()/walk() until the next sync()/close().
        Raises AnalysisError on refusal or engine failure."""
        cfg = self.cfg
        try:
            cfg.validate()       # the window edits knobs live; re-check them
        except ValueError as e:
            raise AnalysisError(str(e)) from e
        err = self.can_analyze(req) or self.sync(req)
        if err:
            raise AnalysisError(err)
        evaluator = self._ensure_evaluator()
        worlds = int(cfg.worlds)
        # Pre-draw EVERY world seed in the canonical order (one draw per world,
        # w = 0..worlds-1) — the derivation a single-engine IncrementalSearch
        # would use — so any engine split searches the same worlds as a 1-engine
        # run with the same cfg.seed.
        rng = np.random.default_rng(cfg.seed if cfg.seed else None)
        seeds = [int(rng.integers(1, 2**31 - 1)) for _ in range(worlds)]
        self._bounds = world_bounds(worlds, len(self._envs))
        n_eng = len(self._bounds)
        # One shared lock around a possibly non-thread-safe (torch) evaluator;
        # a single engine keeps the unwrapped evaluator (unchanged fast path).
        ev = (evaluator if n_eng == 1
              else _LockedEvaluator(evaluator, threading.Lock()))
        pool: Optional[ThreadPoolExecutor] = None
        try:
            # Appended one at a time so a failure part-way leaves the already
            # opened searches reachable for the error path to close.
            self._searches = []
            for i, (lo, hi) in enumerate(self._bounds):
                self._searches.append(IncrementalSearch(
                    self._envs[i], ev,
                    worlds=hi - lo, c_puct=cfg.c_puct,
                    max_depth=(cfg.sb_max_depth if req.is_sideboard
                               else cfg.max_depth),
                    rollout_turns=(cfg.sb_rollout_turns if req.is_sideboard
                                   else 0),
                    world_seeds=seeds[lo:hi], merge_dupes=cfg.merge_dupes))
            if n_eng > 1:
                pool = ThreadPoolExecutor(max_workers=n_eng,
                                          thread_name_prefix="analysis")
            stats = self._merged_stats()
            if on_update is not None:
                on_update(stats)
            while not (stop is not None and stop.is_set()):
                done = sum(s.sims_run for s in self._searches)
                n = int(cfg.chunk_sims)
                if cfg.max_sims:
                    remaining = int(cfg.max_sims) - done
                    if remaining <= 0:
                        break
                    n = min(n, remaining)
                self._run_chunk(n, done, worlds, pool)
                stats = self._merged_stats()
                if on_update is not None:
                    on_update(stats)
            return stats
        except AnalysisError:
            raise
        except Exception as e:  # noqa: BLE001 — engine/pipe failure mid-search
            self._dead = str(e)
            for s in self._searches:
                try:
                    s.close()
                except Exception:  # noqa: BLE001 — the fleet is dead anyway
                    pass
            self._searches = []
            raise AnalysisError(f"analysis engine failed mid-search: {e}") from e
        finally:
            if pool is not None:
                pool.shutdown(wait=True)

    def _run_chunk(self, n: int, done: int, worlds: int,
                   pool: Optional[ThreadPoolExecutor]) -> None:
        """Advance the fleet by ``n`` total sims: each engine gets exactly the
        share the single-engine global round-robin would have given its world
        slice, run concurrently and joined (a barrier) before returning."""
        searches = self._searches
        if len(searches) == 1:
            searches[0].run_chunk(n)
            return
        counts = [_slice_target(done + n, lo, hi, worlds)
                  - _slice_target(done, lo, hi, worlds)
                  for lo, hi in self._bounds]
        futures = [pool.submit(s.run_chunk, k)
                   for s, k in zip(searches, counts)]
        errors = []
        for i, fut in enumerate(futures):     # join ALL before reporting
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001 — surfaced after the barrier
                errors.append((i, e))
        if errors:
            i, e = errors[0]
            raise RuntimeError(f"analysis engine {i} failed mid-chunk: {e}")

    # ----- browsing the finished/parked search -----

    def _open_searches(self) -> List[IncrementalSearch]:
        if not self._searches:
            raise AnalysisError("no open search (analyze first)")
        return self._searches

    def _search_for_world(self, world: int) -> Tuple[IncrementalSearch, int]:
        """The search owning global world index ``world``, plus that world's
        index INSIDE it (the contiguous split makes it world - lo)."""
        searches = self._open_searches()
        w = int(world)
        for s, (lo, hi) in zip(searches, self._bounds):
            if lo <= w < hi:
                return s, w - lo
        total = self._bounds[-1][1] if self._bounds else 0
        raise AnalysisError(f"world {world} out of range (0..{total - 1})")

    def _merged_stats(self) -> LiveStats:
        """The fleet's LiveStats as one root: visits/w_sum/sims summed, priors
        and net_value from engine 0 (every engine evaluated the same root obs),
        root_value visit-weighted, and the per-world arrays concatenated in
        ENGINE order — which is world order, so world indices stay global."""
        parts = [s.stats() for s in self._open_searches()]
        if len(parts) == 1:
            return parts[0]
        n = parts[0].num_choices
        visits = np.zeros(n, dtype=np.float64)
        w_sum = np.zeros(n, dtype=np.float64)
        sims_run = 0
        sim_steps = 0
        for p in parts:
            visits += p.visits
            w_sum += p.w_sum
            sims_run += p.sims_run
            sim_steps += p.sim_steps
        total = visits.sum()
        return LiveStats(
            num_choices=n,
            sims_run=sims_run,
            sim_steps=sim_steps,
            visits=visits,
            priors=parts[0].priors,
            q=np.where(visits > 0, w_sum / np.maximum(visits, 1), 0.0),
            w_sum=w_sum,
            root_value=float(w_sum.sum()) / total if total > 0 else 0.0,
            net_value=parts[0].net_value,
            world_values=np.concatenate([p.world_values for p in parts]),
            world_visits=np.concatenate([p.world_visits for p in parts], axis=0),
        )

    def stats(self) -> LiveStats:
        return self._merged_stats()

    def pv(self, action: int, world: int, max_len: int = 24) -> list:
        search, local = self._search_for_world(world)
        return search.pv(action, local, max_len=max_len)

    def walk(self, world: int, actions) -> list:
        try:
            search, local = self._search_for_world(world)
            return search.walk(local, actions)
        except AnalysisError:
            raise
        except Exception as e:  # noqa: BLE001
            self._dead = str(e)
            self._searches = []
            raise AnalysisError(f"analysis engine failed during walk: {e}") from e

    # ----- teardown -----

    def close(self) -> None:
        """Close every open search and analysis engine. Idempotent."""
        for s in self._searches:
            try:
                s.close()
            except Exception:  # noqa: BLE001 — tearing down anyway
                pass
        self._searches = []
        for env in self._envs:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass
        self._envs = []
        self._bounds = []
