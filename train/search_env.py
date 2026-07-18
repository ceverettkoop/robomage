"""Search-server environment: RoboMageEnv plus the MCTS snapshot protocol.

``SearchRoboMageEnv`` drives the engine with ``--search-server``, which extends
machine mode with four text commands alongside the usual integer action index:

    SNAPSHOT <slot>      deep-copy the game state into a slot   -> SNAPSHOT_OK
    RESTORE <slot>       roll back to a slot                    -> (restored query)
    DETERMINIZE <seed>   reshuffle hidden zones, world-local RNG -> DETERMINIZE_OK
    RELEASE              drop all snapshots                     -> RELEASE_OK

Every engine query is preceded by a ``SEARCHINFO safe=<0|1>`` line; SNAPSHOT
and DETERMINIZE are legal only at safe (loop-restorable) decisions — the
priority decision, the attacker/blocker SELECT prompts, and the cleanup
discard. If a game ends while a snapshot is live the engine emits
``SIM_RESULT: <A|B|DRAW>`` and blocks for RESTORE/RELEASE instead of exiting.

Real moves go through the inherited gym ``step()``; simulated moves go through
``sim_step()``, which skips step counting, reward parsing, and the terminal
bookkeeping (a simulated "Player A wins" narrative line means nothing to the
real episode). Driver discipline: snapshots may only be live *during* a
search — ``release()`` must be called before the chosen real action is
stepped, otherwise a real game-end would park the engine in the SIM_RESULT
intercept while ``step()``'s reader waits for a query that never comes.

Mirror pool (world-parallel interactive search)
-----------------------------------------------
For interactive consumers only (play/TUI/observe/analysis), ``ensure_mirrors(k)``
lazily spins up ``k`` extra engine processes kept in lockstep with this primary:
each mirror ``reset()``s to the primary's ``last_engine_seed`` and replays the
recorded real-action history, so it holds a byte-identical game state (same
binary + ctor state flags + seed + action stream => identical engine state).
``step()`` forwards every real action to the live mirrors and re-checks obs
equality; ``search_envs()`` then exposes ``[self] + mirrors`` so
``mcts.run_search_parallel`` can split the determinized worlds across the
processes. Same driver discipline as snapshots: mirrors receive real ``step()``s
only when no snapshots are live (a search restores+releases every env before the
real move). Any mirror drift/failure disables the whole pool (the primary plays
on alone) — interactive robustness beats strictness. Self-play and the parity
corpus never touch this path (procs defaults to 1).
"""

from __future__ import annotations

import sys
from typing import NamedTuple, Optional

import numpy as np

from env import NarrativeEnv, RoboMageEnv


class SimQuery(NamedTuple):
    """One engine response during search: either a decision or a sim game-end.

    obs is the env's REUSED observation buffer (copy before retaining);
    terminal is None for a decision, else "A" / "B" / "DRAW".
    """
    obs: Optional[np.ndarray]
    num_choices: int
    pending_confirm: bool
    terminal: Optional[str]


class ProtocolError(RuntimeError):
    pass


class SearchRoboMageEnv(RoboMageEnv):
    """RoboMageEnv wired for the --search-server snapshot protocol."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mirror pool (interactive world-parallel search). The action history is
        # recorded unconditionally (tiny) so a mirror created mid-game can be
        # fast-forwarded to the current decision by replay.
        self._action_history: list[int] = []
        self._mirrors: list["SearchRoboMageEnv"] = []
        self._mirror_pool_disabled = False

    def _extra_engine_flags(self) -> list:
        return ["--search-server"]

    # ------------------------------------------------------------------
    # gym overrides (record history + keep mirrors in lockstep)
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # A new episode invalidates any mirrors (stale seed/state) and re-arms
        # the pool for this env.
        self._action_history = []
        self._close_mirrors()
        self._mirror_pool_disabled = False
        return obs, info

    def step(self, action: int):
        result = super().step(action)
        # Record the PRE-remap env-index action (the same value step() consumed,
        # before its own confirm-slot remap): replaying it into a mirror
        # re-derives the identical remap, so the mirror stays byte-identical.
        self._action_history.append(int(action))
        if self._mirrors and not self._mirror_pool_disabled:
            self._forward_to_mirrors(int(action))
        return result

    def close(self):
        self._close_mirrors()
        super().close()

    # ------------------------------------------------------------------
    # Mirror pool
    # ------------------------------------------------------------------

    def _mirror_ctor_kwargs(self) -> dict:
        """State-relevant ctor kwargs for a mirror. Only params that affect the
        engine's game state are copied; output-only flags (narrative, log_viewer,
        log_decisions, render_mode) are EXCLUDED — a mirror is a plain
        SearchRoboMageEnv, never Narrative, and must not clobber the primary's
        RMLOG file via log_decisions."""
        return dict(
            binary_path=self.binary_path,
            deck_a=self._deck_a, deck_b=self._deck_b,
            bo3=self._bo3, auto_sideboard=self._auto_sideboard,
            no_shuffle=self._no_shuffle,
            battlefield_a=self._battlefield_a, battlefield_b=self._battlefield_b,
            graveyard_a=self._graveyard_a, graveyard_b=self._graveyard_b,
            exile_a=self._exile_a, exile_b=self._exile_b,
            sideboard_a=self._sideboard_a, sideboard_b=self._sideboard_b,
            life_a=self._life_a, life_b=self._life_b,
        )

    def ensure_mirrors(self, k: int) -> None:
        """Lazily bring the live mirror count up to ``k``. Each new mirror resets
        to this env's seed and replays the recorded action history, then its obs
        is asserted byte-equal to the primary's. Any failure disables the pool for
        this env (the primary plays on alone) — never raises to the caller."""
        if k <= 0 or self._mirror_pool_disabled:
            return
        while len(self._mirrors) < k:
            m = None
            try:
                m = SearchRoboMageEnv(**self._mirror_ctor_kwargs())
                m.reset(options={"engine_seed": self.last_engine_seed})
                for a in self._action_history:
                    m.step(int(a))
                if not np.array_equal(m._obs, self._obs):
                    raise RuntimeError("mirror obs != primary after history replay")
                self._mirrors.append(m)
            except Exception as e:  # noqa: BLE001 — robustness over strictness
                if m is not None:
                    try:
                        m.close()
                    except Exception:
                        pass
                self._disable_pool(f"mirror spawn/replay failed: {e}")
                return

    def search_envs(self) -> list:
        """The engines available to a parallel search: this primary plus every
        live mirror (empty mirror list => just [self], a plain single-env search)."""
        return [self] + list(self._mirrors)

    def spawn_detached_mirror(self, history_len: int,
                              expect_obs: Optional[np.ndarray] = None
                              ) -> "SearchRoboMageEnv":
        """A byte-identical engine copy the CALLER owns — never registered in
        this env's mirror pool, so it receives no forwarded real steps (the
        caller catches it up by replaying further history itself). Used by the
        GUI analysis window, whose engine is driven from a different thread
        than the game loop. Replays ``self._action_history[:history_len]``
        (a prefix read of the append-only list, so safe to call with the
        driver thread stepping the primary concurrently); when ``expect_obs``
        is given, the replayed obs must match it byte-for-byte. Raises on any
        spawn/replay/verify failure (the partial mirror is closed first)."""
        m = SearchRoboMageEnv(**self._mirror_ctor_kwargs())
        try:
            m.reset(options={"engine_seed": self.last_engine_seed})
            for a in self._action_history[:history_len]:
                m.step(int(a))
            if expect_obs is not None and not np.array_equal(m._obs, expect_obs):
                raise RuntimeError(
                    "detached mirror obs != expected after history replay")
            return m
        except Exception:
            try:
                m.close()
            except Exception:
                pass
            raise

    def _forward_to_mirrors(self, action: int) -> None:
        for m in self._mirrors:
            try:
                m.step(action)
                if not np.array_equal(m._obs, self._obs):
                    raise RuntimeError("mirror obs diverged from primary after step")
            except Exception as e:  # noqa: BLE001 — robustness over strictness
                self._disable_pool(f"mirror lockstep step failed: {e}")
                return

    def _disable_pool(self, reason: str) -> None:
        if self._mirror_pool_disabled:
            return
        self._mirror_pool_disabled = True
        print(f"[search-mirror] pool disabled: {reason}; primary continues alone",
              file=sys.stderr, flush=True)
        self._close_mirrors()

    def _close_mirrors(self) -> None:
        for m in self._mirrors:
            try:
                m.close()
            except Exception:
                pass
        self._mirrors = []

    # ------------------------------------------------------------------
    # Protocol commands (each re-emits the current/restored decision query)
    # ------------------------------------------------------------------

    def snapshot(self, slot: int = 0) -> SimQuery:
        self._send_text(f"SNAPSHOT {slot}")
        return self._read_search_query(expect_ack=b"SNAPSHOT_OK")

    def restore(self, slot: int = 0) -> SimQuery:
        # No ack line: the engine unwinds internally and re-emits the restored
        # decision's query.
        self._send_text(f"RESTORE {slot}")
        return self._read_search_query()

    def determinize(self, world_seed: int) -> SimQuery:
        self._send_text(f"DETERMINIZE {world_seed}")
        return self._read_search_query(expect_ack=b"DETERMINIZE_OK")

    def release(self) -> SimQuery:
        """Drop all snapshots. Only legal at a live decision (not at SIM_RESULT
        — there RELEASE ends the game and no query follows)."""
        self._send_text("RELEASE")
        return self._read_search_query(expect_ack=b"RELEASE_OK")

    def sim_step(self, action: int) -> SimQuery:
        """Play one action inside a simulation. Applies the same confirm-slot
        remap as step() so the search tree operates in env-index space."""
        game_action = action
        if self._pending_confirm and action == self._num_choices - 1:
            game_action = -1
        self._send(game_action)
        return self._read_search_query()

    def current_query(self) -> SimQuery:
        """The decision the env is currently parked at (from the last read)."""
        return SimQuery(self._obs, self._num_choices, self._pending_confirm, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _send_text(self, text: str) -> None:
        self._proc.stdin.write((text + "\n").encode())
        self._proc.stdin.flush()

    def _read_search_query(self, expect_ack: bytes | None = None) -> SimQuery:
        """Read engine output until the next BQUERY or SIM_RESULT.

        Narrative lines — including the win-message printf a simulated game
        end produces before SIM_RESULT — are ignored; only SIM_RESULT is
        authoritative for a simulated game's outcome.
        """
        acked = expect_ack is None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                rc = self._proc.wait()
                raise ProtocolError(
                    f"engine exited (code {rc}) during search — a snapshot "
                    "command at an illegal point is fatal engine-side; see stderr")
            line = line.rstrip(b"\n")
            if line.startswith(b"SIM_RESULT:"):
                return SimQuery(None, 0, False, line.split(b":", 1)[1].strip().decode())
            if line.startswith(b"SEARCHINFO"):
                self.last_search_safe = line.endswith(b"safe=1")
                continue
            if expect_ack is not None and line.startswith(expect_ack):
                acked = True
                continue
            if line.startswith(b"BQUERY: "):
                if not acked:
                    raise ProtocolError(
                        f"engine re-emitted a query without acking {expect_ack!r}")
                self._parse_bquery_payload(line)
                return SimQuery(self._obs, self._num_choices,
                                self._pending_confirm, None)
            # anything else is sim narrative — drop it


class SearchNarrativeEnv(NarrativeEnv, SearchRoboMageEnv):
    """NarrativeEnv (buffered real-game narrative for transcripts) with the
    search protocol. MRO gives NarrativeEnv's line buffering for REAL steps
    while _read_search_query drops simulation narrative, so transcripts never
    interleave simulated lines. Used by runner.run_games when a controller
    advertises wants_search_env."""

