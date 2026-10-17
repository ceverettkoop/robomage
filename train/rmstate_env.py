"""The engine on the RMSTATE protocol, seen through the pass-1 parity adapter.

``RMStateNarrativeEnv`` is a ``env.NarrativeEnv`` whose engine runs with
``--machine-format rmstate`` (plus ``--rmstate-card-scripts``): stdout carries only RMSTATE
records, everything else the engine prints (narrative, ``GAME_RESULT`` / ``MATCH_RESULT``) goes
to stderr. Each seat's frames feed that seat's ``rmstate_adapter.SeatAdapter``; at every decision
the deciding seat's frame becomes the observation and side-channels a BQUERY env would give, so
``runner.drive_game`` and the controllers that read only what the adapter rebuilds (the scripted
agent in every tier, ``auto``, ``actions:``, ``play:``) run on it unchanged. The BQUERY vector is
not available in this mode: the engine never writes it.

The engine prints a decision's narrative before its frame (it flushes stdout, which points at
stderr, first) and then blocks on stdin, so once a decision frame is read, a non-blocking drain of
stderr has every line that belongs before it. Engine ``WARNING:`` / ``ERROR:`` / ``FATAL:`` lines
are not narrative (a BQUERY engine writes them to its own stderr): they go to this process's
stderr. ``obs_partial`` tells the runner the observation holds only the adapter's subset, so its
verbose transcript leaves the board block out (``runner.run_games``).
"""
from __future__ import annotations

import os
import select
import sys

import numpy as np

from env import (DRAW_REWARD, GAME_LOSS_REWARD, GAME_WIN_REWARD, MATCH_LOSS_REWARD,
                 MATCH_WIN_REWARD, OBS_SIZE, NarrativeEnv)
import rmstate
from rmstate_adapter import ADAPTER_TABLES, SeatAdapter

# Engine flags that select the RMSTATE protocol and put the raw scripts into CARDS.
RMSTATE_ENGINE_FLAGS = ["--machine-format", "rmstate", "--rmstate-card-scripts"]
# Engine stderr lines that are diagnostics, not narrative.
_DIAGNOSTIC_PREFIXES = (b"WARNING:", b"ERROR:", b"FATAL:")
_READ_CHUNK = 1 << 16


class RMStateNarrativeEnv(NarrativeEnv):
    """A NarrativeEnv over ``--machine-format rmstate`` (module docstring)."""

    obs_partial = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        import subprocess
        self.engine_stderr = subprocess.PIPE
        self._adapters = {}
        self._decoder = None
        self.frame = None              # the decision frame (rmstate.Frame) of the current decision
        self.seat = None               # its seat, "A" / "B"
        self.decisions = 0
        self.diagnostics = []          # engine WARNING/ERROR/FATAL lines

    def _extra_engine_flags(self) -> list:
        return list(RMSTATE_ENGINE_FLAGS)

    # The match's card knowledge and deck names, as the scripted agent reads them.
    @property
    def card_knowledge(self):
        ad = self._adapters.get(self.seat)
        return ad.cards if ad is not None else None

    @property
    def deck_names(self):
        """{"A": name, "B": name} from the deciding seat's DECKLISTS frame (None when withheld)."""
        ad = self._adapters.get(self.seat)
        names = ad.deck_names if ad is not None else {}
        return {"A": names.get(1), "B": names.get(2)}

    def reset(self, *, seed=None, options=None):
        self._adapters = {"A": SeatAdapter("A"), "B": SeatAdapter("B")}
        self._decoder = rmstate.StreamDecoder()
        self._out = bytearray()
        self._err = bytearray()
        self._out_eof = self._err_eof = False
        self.frame = self.seat = None
        self.decisions = 0
        return super().reset(seed=seed, options=options)

    # ── engine I/O ─────────────────────────────────────────────────────────────────────────
    def _read_until_query(self):
        info = {"reward": 0.0, "done": False, "shaping_a": 0.0, "shaping_b": 0.0, "game_result": False}
        out_fd = self._proc.stdout.fileno()
        err_fd = self._proc.stderr.fileno()
        while True:
            payload = self._next_record()
            if payload is not None:
                frame = self._decoder.feed(payload, ADAPTER_TABLES)
                seat = frame.viewer
                ad = self._adapters.get(seat)
                if ad is None:
                    continue
                ad.feed(frame, self._decoder.schema)
                if frame.kind != "STATE" or frame.tables["header"][0]["terminal"] or "menu" not in frame.tables:
                    continue
                self._drain_stderr(err_fd, info, block=False)
                obs, side = ad.observation(frame)
                self.frame, self.seat = frame, seat
                self.decisions += 1
                self._set_side_channels(obs, side)
                return self._obs, info
            if self._out_eof:
                self._drain_stderr(err_fd, info, block=True)
                rc = self._proc.wait()
                if rc != 0:
                    self._kill_proc()
                    raise RuntimeError(
                        f"game process exited with code {rc} before the game finished — see the "
                        "engine's FATAL/ERROR output: " + " | ".join(self.diagnostics[-3:]))
                info["done"] = True
                self._kill_proc()
                return np.zeros(OBS_SIZE, dtype=np.float32), info
            fds = [out_fd] + ([] if self._err_eof else [err_fd])
            ready, _, _ = select.select(fds, [], [])
            if err_fd in ready:
                self._read_stderr_chunk(err_fd, info)
            if out_fd in ready:
                chunk = os.read(out_fd, _READ_CHUNK)
                if chunk:
                    self._out += chunk
                else:
                    self._out_eof = True

    def _next_record(self):
        """The next complete record's payload from the stdout buffer, or None."""
        nl = self._out.find(b"\n")
        if nl < 0:
            return None
        head = bytes(self._out[:nl])
        if not head.startswith(b"RMSTATE ") or not head[8:].isdigit():
            raise rmstate.RMStateError(f"bad record header {head[:40]!r}")
        n = int(head[8:])
        if len(self._out) < nl + 1 + n:
            return None
        payload = bytes(self._out[nl + 1:nl + 1 + n])
        del self._out[:nl + 1 + n]
        return payload

    def _read_stderr_chunk(self, fd, info):
        chunk = os.read(fd, _READ_CHUNK)
        if not chunk:
            self._err_eof = True
            if self._err:
                self._err += b"\n"
        self._err += chunk
        self._take_lines(info)

    def _drain_stderr(self, fd, info, block):
        """Everything the engine has written to stderr so far (``block``: until its end)."""
        if self._err_eof:
            return
        os.set_blocking(fd, block)
        try:
            while not self._err_eof:
                try:
                    chunk = os.read(fd, _READ_CHUNK)
                except BlockingIOError:
                    break
                if not chunk:
                    self._err_eof = True
                    if self._err:
                        self._err += b"\n"
                    break
                self._err += chunk
        finally:
            if not self._err_eof:
                os.set_blocking(fd, True)
        self._take_lines(info)

    def _take_lines(self, info):
        while True:
            nl = self._err.find(b"\n")
            if nl < 0:
                return
            line = bytes(self._err[:nl])
            del self._err[:nl + 1]
            self._engine_line(line, info)

    def _engine_line(self, line, info):
        """One stderr line, read as RoboMageEnv reads a stdout line in BQUERY mode."""
        if line.startswith(b"GAME_RESULT:"):
            info["game_result"] = True
            if b"Player A wins" in line:
                info["reward"] += GAME_WIN_REWARD
            elif b"Player B wins" in line:
                info["reward"] += GAME_LOSS_REWARD
            else:
                info["reward"] += DRAW_REWARD
        elif self._bo3 and line.startswith(b"MATCH_RESULT:"):
            info["reward"] += MATCH_WIN_REWARD if b"Player A wins" in line else MATCH_LOSS_REWARD
        if line.startswith((b"MANA_WASTED: ", b"MULLIGAN_PENALTY: ")):
            return
        if line.startswith(_DIAGNOSTIC_PREFIXES):
            text = line.decode("utf-8", errors="replace")
            self.diagnostics.append(text)
            print(text, file=sys.stderr)
            return
        if self.render_mode == "human":
            self._print_narrative_line(line.decode("utf-8", errors="replace"))

    def _set_side_channels(self, obs, side):
        self._obs[:] = obs
        self._num_choices = side["num_choices"]
        self._action_cats = side["cats"]
        self._action_ordinals = side["ords"]
        self._action_public = side["public"]
        self._action_descriptions = side["descriptions"] if self._narrative else None
        self._perm_counters = None
        self._perm_token_names = None
        self._pending_confirm = side["pending_confirm"]

    def _kill_proc(self):
        if self._proc is not None and self._proc.stderr is not None:
            try:
                self._proc.stderr.close()
            except OSError:
                pass
        super()._kill_proc()
