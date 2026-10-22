#!/usr/bin/env python3
"""The teacher: the scripted agent as a labeller on the RMSTATE machine protocol (training plan T4).

The engine's ``--teacher-a / --teacher-b <command>`` (``src/agents/teacher.h``) starts a command
and streams one seat's frames to its stdin, exactly the ``--machine`` protocol
(``docs/state_format.md`` §9: ``RMSTATE <nbytes>\\n`` + payload, from the match's first frame,
sideboarding frames and the terminal frames included), reading one integer line back per decision
frame. This script is that command for the scripted tiers:

    python3 python/teacher.py [--tier hard] [--seed N] [--record FILE]

``--tier`` is a ``scripted_core.preset_config`` tier: ``hard`` (the default; Done at every
sideboarding phase), ``hard+sb`` (hard, sideboarding between games per the league guides,
``sideboard_guides.py``), ``easy`` / ``greedy``, ``random``, ``explore``, ...

It answers every non-terminal ``STATE`` frame that holds a menu with
``scripted_core.ScriptedCore``'s row (``rm_agent.decide``, exactly as ``rm_driver.py`` plays a
scripted seat), as a plain menu row (never ``-1``; it never concedes), one line per decision,
flushed. Terminal frames and the other frame kinds are read and followed without a reply. It reads
until the end of its input and then exits with status 0; any decoding or agent error ends it with
a traceback and status 1 (the engine then stops with a fatal error naming the seat).

**State.** Everything it keeps is a function of the stream it has read: the decoder's schema, each
viewer's ``SeatFrames``, the core's per-game state (mulligan counts, the fruitless-activation stall
guard) and Python's global ``random`` (the hard tier breaks ties of uncategorised choices with
``random.choice``). It never assumes its advice was followed: the frames say what happened.

**Matches.** A match starts at the stream's first frame and at the first ``SCHEMA`` frame after a
terminal frame (the engine sends each seat its ``SCHEMA`` / ``CARDS`` / ``DECKLISTS`` /
``METAGAME`` frames the first time it decides in a match, and a terminal frame to each seat when
the match is over). At each match start the teacher forgets every seat's frames, calls the core's
``new_game()`` and seeds ``random`` with ``--seed`` (default 0), as ``rm_driver.play_game`` does
before each match with the match's seed: given that seed, the labels at a seat are the decisions
``rm_driver.py`` would have taken there with ``scripted:<tier>`` playing the seat (and an opponent
that draws nothing from Python's ``random``). One process can therefore label any number of
successive matches on one stream.

The stream may hold both seats' frames (it answers whichever seat a decision frame is for, each
seat followed by its own ``SeatFrames``), so it can also play an engine ``--machine`` stream
directly (``serve``).

``--record FILE`` writes every record it reads to FILE, byte for byte (the gate compares it with
the ``--machine`` stream of the same game).
"""
from __future__ import annotations

import argparse
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import rmstate  # noqa: E402
from rm_agent import FRAME_TABLES, SeatFrames, decide  # noqa: E402
from scripted_core import ScriptedCore, preset_config  # noqa: E402


def read_record(inp):
    """The next record's (header line, payload) from binary file ``inp``; None at the end of input.
    Raises ``rmstate.RMStateError`` on bad framing."""
    head = inp.readline()
    if not head:
        return None
    if not head.endswith(b"\n") or not head.startswith(b"RMSTATE ") or not head[8:-1].isdigit():
        raise rmstate.RMStateError(f"bad record header {head[:40]!r}")
    n = int(head[8:-1])
    payload = inp.read(n)
    if len(payload) != n:
        raise rmstate.RMStateError("truncated record")
    return head, payload


class Teacher:
    """The labeller's state over one stream: ``frame(payload)`` follows a record and returns the
    reply a decision frame takes (None for any other frame)."""

    def __init__(self, tier: str = "hard", seed: int = 0):
        self.core = ScriptedCore(preset_config(tier))
        self.seed = seed
        self.decoder = rmstate.StreamDecoder()
        self.seats = {}
        self._match_open = False

    def _new_match(self):
        random.seed(self.seed)
        self.core.new_game()
        self.seats = {}
        self._match_open = True

    def frame(self, payload: bytes):
        kind = self.decoder.schema.kind_of(payload) if self.decoder.schema is not None else "SCHEMA"
        if not self._match_open and kind == "SCHEMA":
            self._new_match()
        frame = self.decoder.feed(payload, FRAME_TABLES)
        seat = self.seats.get(frame.viewer)
        if seat is None:
            seat = self.seats[frame.viewer] = SeatFrames(frame.viewer)
        seat.feed(frame, self.decoder.schema)
        if frame.kind != "STATE":
            return None
        if frame.tables["header"][0]["terminal"]:
            self._match_open = False
            return None
        if "menu" not in frame.tables:
            return None
        return int(decide(self.core, seat, frame))


def serve(inp, out, *, tier: str = "hard", seed: int = 0, record=None) -> int:
    """Labels the stream on binary file ``inp``, one reply line per decision frame on binary file
    ``out`` (flushed); ``record`` (a binary file) gets every record read. Returns the decisions
    answered."""
    teacher = Teacher(tier, seed)
    answered = 0
    while True:
        rec = read_record(inp)
        if rec is None:
            return answered
        head, payload = rec
        if record is not None:
            record.write(head)
            record.write(payload)
            record.flush()
        reply = teacher.frame(payload)
        if reply is None:
            continue
        out.write(b"%d\n" % reply)
        out.flush()
        answered += 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="teacher.py", description=__doc__.split("\n")[0])
    p.add_argument("--tier", default="hard",
                   help="the scripted tier labelling (scripted_core.preset_config: hard, hard+sb (hard, "
                        "sideboarding per the league guides), easy, greedy, random, explore, ...; default hard)")
    p.add_argument("--seed", type=int, default=0,
                   help="Python's random is seeded with it at each match start (default 0)")
    p.add_argument("--record", metavar="FILE", help="write every record read to FILE")
    args = p.parse_args(argv)
    preset_config(args.tier)
    record = open(args.record, "wb") if args.record else None
    try:
        serve(sys.stdin.buffer, sys.stdout.buffer, tier=args.tier, seed=args.seed, record=record)
    finally:
        if record is not None:
            record.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
