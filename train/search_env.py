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
"""

from __future__ import annotations

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

    def _extra_engine_flags(self) -> list:
        return ["--search-server"]

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

