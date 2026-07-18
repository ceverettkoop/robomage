"""Engine-side core of the GUI analysis window (Qt-free).

An AnalysisSession owns one DETACHED analysis engine — a byte-identical copy of
the live session's engine process (see SearchRoboMageEnv.spawn_detached_mirror)
that is never registered in the primary's mirror pool, so the live game stays
fully responsive while MCTS grinds here on a worker thread. The analysis engine
is kept in lockstep lazily: each real decision's StateUpdate carries the
primary's action-history length, and sync() replays just the delta before a new
analysis, verifying the replayed obs byte-equals the decision being analyzed.

analyze() runs an mcts.IncrementalSearch in caller-paced chunks (live UI
updates via on_update, cancellation via a threading.Event) and leaves the
search OPEN — root snapshot held — so pv()/walk() can browse principal
variations and replay hypothetical lines until the next sync()/close().

Threading contract: every method here must be called from ONE thread (the
analysis worker). The only cross-thread touch is reading a prefix of the
primary env's append-only _action_history list, which is safe under the GIL.

Torch-free unless a model evaluator is requested: "uniform" runs without torch
(tests), "az:gen" / "mcts:gen" lazily import the checkpoint machinery.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from cli_spec import DEFAULT_SB_MAX_DEPTH
from env import _IS_SIDEBOARD_IDX, _SELF_IS_A_IDX
from mcts import IncrementalSearch, LiveStats, UniformEvaluator


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
    c_puct: float = 1.5
    max_depth: int = 60
    sb_max_depth: int = DEFAULT_SB_MAX_DEPTH  # sideboard roots: game-long horizon
    seed: int = 0            # search rng seed; 0 = fresh entropy per run
    auto_analyze: bool = True  # UI: start a run at every new analyzable decision


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
    """Owns the detached analysis engine + the open IncrementalSearch."""

    def __init__(self, primary_env, cfg: AnalysisConfig,
                 evaluator=None, evaluator_label: str = ""):
        self._primary = primary_env
        self.cfg = cfg
        self._evaluator = evaluator          # lazily loaded from cfg when None
        self.evaluator_label = evaluator_label
        self._env = None                     # the detached analysis engine
        self._synced_len = 0                 # actions replayed into it so far
        self._search: Optional[IncrementalSearch] = None
        self._dead: Optional[str] = None     # error once the engine is unusable

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
        """Lazily spawn the detached analysis engine at this request's state.
        Returns None on success, else the error (session marked dead)."""
        if self._dead:
            return f"analysis engine unavailable: {self._dead}"
        if self._env is not None:
            return None
        try:
            self._env = self._primary.spawn_detached_mirror(
                req.history_len, expect_obs=req.obs)
            self._synced_len = req.history_len
            return None
        except Exception as e:  # noqa: BLE001 — surfaced as a status, not a crash
            self._dead = str(e)
            self._env = None
            return f"analysis engine unavailable: {self._dead}"

    def sync(self, req: AnalysisRequest) -> Optional[str]:
        """Close any open search, replay the real-action delta into the
        analysis engine, and verify it matches the request's obs. Returns None
        on success, else the error (session marked dead)."""
        err = self.ensure_engine(req)
        if err:
            return err
        if self._search is not None:
            try:
                self._search.close()     # restore + release BEFORE real steps
            except Exception as e:  # noqa: BLE001
                self._dead = f"search close failed: {e}"
                return f"analysis engine unavailable: {self._dead}"
            self._search = None
        if req.history_len < self._synced_len:
            self._dead = (f"history went backwards ({self._synced_len} -> "
                          f"{req.history_len}); primary was reset?")
            return f"analysis engine unavailable: {self._dead}"
        try:
            delta = self._primary._action_history[self._synced_len:req.history_len]
            for a in delta:
                self._env.step(int(a))
            self._synced_len = req.history_len
            if not np.array_equal(self._env._obs, req.obs):
                raise RuntimeError("analysis engine obs diverged from decision")
            return None
        except Exception as e:  # noqa: BLE001
            self._dead = str(e)
            return f"analysis engine unavailable: {self._dead}"

    def respawn(self) -> None:
        """Recovery action: drop the (possibly dead) engine + search; the next
        analyze() spawns a fresh engine by full history replay."""
        self._search = None                 # its env is going away with it
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
        self._env = None
        self._synced_len = 0
        self._dead = None

    # ----- the search -----

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
        cfg.max_sims cap (0 = unlimited), or an engine error. on_update fires
        after every chunk (and once up front with the zero-visit priors frame).
        The search is left OPEN for pv()/walk() until the next sync()/close().
        Raises AnalysisError on refusal or engine failure."""
        err = self.can_analyze(req) or self.sync(req)
        if err:
            raise AnalysisError(err)
        evaluator = self._ensure_evaluator()
        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed if cfg.seed else None)
        try:
            self._search = IncrementalSearch(
                self._env, evaluator,
                worlds=cfg.worlds, c_puct=cfg.c_puct,
                max_depth=cfg.sb_max_depth if req.is_sideboard else cfg.max_depth,
                rng=rng)
            stats = self._search.stats()
            if on_update is not None:
                on_update(stats)
            while not (stop is not None and stop.is_set()):
                n = cfg.chunk_sims
                if cfg.max_sims:
                    remaining = cfg.max_sims - self._search.sims_run
                    if remaining <= 0:
                        break
                    n = min(n, remaining)
                stats = self._search.run_chunk(n)
                if on_update is not None:
                    on_update(stats)
            return stats
        except AnalysisError:
            raise
        except Exception as e:  # noqa: BLE001 — engine/pipe failure mid-search
            self._dead = str(e)
            self._search = None
            raise AnalysisError(f"analysis engine failed mid-search: {e}") from e

    # ----- browsing the finished/parked search -----

    def _open_search(self) -> IncrementalSearch:
        if self._search is None:
            raise AnalysisError("no open search (analyze first)")
        return self._search

    def stats(self) -> LiveStats:
        return self._open_search().stats()

    def pv(self, action: int, world: int, max_len: int = 24) -> list:
        return self._open_search().pv(action, world, max_len=max_len)

    def walk(self, world: int, actions) -> list:
        try:
            return self._open_search().walk(world, actions)
        except AnalysisError:
            raise
        except Exception as e:  # noqa: BLE001
            self._dead = str(e)
            self._search = None
            raise AnalysisError(f"analysis engine failed during walk: {e}") from e

    # ----- teardown -----

    def close(self) -> None:
        """Close the open search and the analysis engine. Idempotent."""
        if self._search is not None:
            try:
                self._search.close()
            except Exception:  # noqa: BLE001 — tearing down anyway
                pass
            self._search = None
        if self._env is not None:
            try:
                self._env.close()
            except Exception:  # noqa: BLE001
                pass
            self._env = None
