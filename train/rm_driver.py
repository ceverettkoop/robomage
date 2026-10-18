#!/usr/bin/env python3
"""The agent-game driver on the RMSTATE protocol: engine subprocess, frame read, reply write.

Plan step 2, "Driver" (``docs/engine_fork_plan.md``): the minimal, stdlib-only driver that plays
agent games. It launches the engine with ``--machine`` (``docs/state_format.md`` §9: stdout carries
only ``RMSTATE`` records; narrative, ``GAME_RESULT`` and ``MATCH_RESULT`` go to stderr), decodes
each seat's frames with ``rmstate.py``, follows them in that seat's ``rm_agent.SeatFrames``, asks
the deciding seat's agent for one integer and writes it back. It needs only ``rmstate.py``,
``rm_agent.py`` and the agent modules (``scripted_core.py``, ``explore_agent.py``).

**Agents** (``make_agent``; one spec per seat):

- the scripted tiers (``scripted`` = hard, ``scripted:easy`` / ``greedy``, ``scripted:random``):
  ``scripted_core.ScriptedCore`` on the seat's decoded frame (``rm_agent.decide``);
- ``explore`` / ``explore:patient``: the coverage fuzzer (``explore_agent.py``) the same way;
- ``first`` / ``auto``: always action 0 (pass priority, the first choice);
- ``actions:<i,i,...>``: the listed replies for this seat's decisions in order (a concession
  sentinel ``-2`` / ``-3`` is sent as is), then action 0.

The ``play:`` scripts run in the engine (``--play``, ``src/agents/play_spec.h``).

**A game** (``play_game``): before it, ``random.seed(seed)`` (the scripted agent breaks ties and the
explore tier draws from Python's global ``random``) and each agent's ``new_game()``; the engine gets
``--seed engine_seed(seed)`` (``src/agents/game_seed.h`` derives the same seed), so a seed names
the same game here as on the engine's in-process agents (``--game-seed``). At each decision frame
(the deciding seat's non-terminal ``STATE`` frame, its menu unbounded) the reply is the agent's
index, ``-1`` when it picked the last row of an attacker/blocker declaration (the confirm row), or
a concession sentinel. Terminal frames are fed to the seats and end the stream. A
``GAME_RESULT`` line scores the game (+1 Player A, -1 Player B, 0 a draw); the match is won by the
sign of the sum. Engine diagnostics (``WARNING:``, ``ERROR:``, ``FATAL``, the self-check
lines, …) go to this process's stderr, not the narrative; an engine that exits nonzero raises.

**Transcripts** (``run_games``): ``verbose`` (the smoke tier's transcripts): each decision's
engine narrative in a ``--- Narrative ---`` block, ``--- Pregame ---`` / ``--- Turn N (Player X)
---`` markers with the ``PA:`` / ``PB:`` hand lines (each seat's hand as of its last decision, at
most 10 cards, in (token, name, handle) order, empty while sideboarding), ``--- Decision N ---``
with the menu as offered and ``>> label/seat: i (description)``, then the result. ``compact``
writes the narrative and one line per decision (the engine's category and step names); ``quiet``
nothing (a draw is still announced and its log saved to ``draw_<stamp>.txt``).

CLI: ``python3 train/rm_driver.py --player-a scripted --player-b explore --deck-a league/bug ...``
(``--help``). Its flags are ``CLI_FLAGS`` below; the parser is built from them with ``argparse``.
"""
from __future__ import annotations

import argparse
import datetime
import os
import random
import select
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import rmstate  # noqa: E402
from rm_agent import FRAME_TABLES, SeatFrames, decide, fold_name  # noqa: E402

REPO_ROOT = os.path.dirname(_HERE)
BIN_DIR = os.path.join(REPO_ROOT, "bin")          # the engine's working directory (resources/)
# The build the driver runs by default: $ROBOMAGE_BUILD, else debug (like observe and the gate).
DEFAULT_BINARY = os.path.join(BIN_DIR, os.environ.get("ROBOMAGE_BUILD", "debug"), "robomage")
# Extra engine flags every driven engine gets (the gate's tiers set it).
ENGINE_FLAGS_ENV = "ROBOMAGE_ENGINE_FLAGS"
TRANSCRIPTS = ("verbose", "compact", "quiet")
FORMATS = ("bo1", "bo3")
ON_THE_PLAY = ("a", "b", "random")
CONCEDE_GAME, CONCEDE_MATCH = -2, -3
HAND_LINE_CAP = 10                                # cards on a PA: / PB: line

# Engine stderr lines that are diagnostics, not narrative.
_DIAGNOSTIC_PREFIXES = (b"WARNING:", b"ERROR:", b"FATAL", b"error:", b"CRASH:", b"Backtrace:",
                        b"SELFCHECK-STATE:", b"SELFCHECK-KNOWLEDGE:", b"parse_card_script:")
# Menu categories of an attacker/blocker declaration, whose last row is the confirm row.
_DECLARATION_CATS = frozenset({"SELECT_ATTACKER", "CONFIRM_ATTACKERS", "SELECT_BLOCKER",
                               "CONFIRM_BLOCKERS"})
_PREGAME_CATS = (frozenset({"KEEP_HAND", "MULLIGAN"}), frozenset({"BOTTOM_DECK_CARD"}))
_READ_CHUNK = 1 << 16
_NONE16 = 0xFFFF
_SEAT_A = 1                                       # enum:seat value of Player A
_SELF_REL = 1                                     # player-rel value of the viewer


# ── CLI flags ───────────────────────────────────────────────────────────────────────────────

CLI_FLAGS = (
    {"name": "--player-a", "default": "scripted", "suggest": "agent",
     "help": "Player A's agent: scripted (hard), scripted:easy|greedy|random, explore, "
             "explore:patient, first (= auto: action 0) or actions:<i,i,...> (default: scripted)"},
    {"name": "--player-b", "default": "scripted", "suggest": "agent",
     "help": "Player B's agent (see --player-a; default: scripted)"},
    {"name": "--deck-a", "default": "delver", "suggest": "deck",
     "help": "Player A's deck (.dk stem under bin/resources/decks/, default: delver)"},
    {"name": "--deck-b", "default": None, "suggest": "deck",
     "help": "Player B's deck (default: Player A's)"},
    {"name": "--format", "kind": "choice", "choices": FORMATS, "default": "bo3",
     "help": "Match format: bo3 = best-of-three matches with sideboarding, bo1 = single games "
             "(default bo3)"},
    {"name": "--games", "kind": "int", "default": 1,
     "help": "Matches to play (single games under --format bo1; default 1); game i uses seed "
             "--seed + i"},
    {"name": "--seed", "kind": "int", "default": 1, "help": "Base seed (default 1)"},
    {"name": "--on-the-play", "kind": "choice", "choices": ON_THE_PLAY, "default": "a",
     "help": "Whose side starts game 1: a, b (the two sides' agents and decks swap engine seats) "
             "or random (a seeded coin flip) (default a)"},
    {"name": "--verbose", "kind": "flag",
     "help": "The full transcript: narrative blocks, hand lines, every menu and choice"},
    {"name": "--quiet", "kind": "flag",
     "help": "No transcript on stdout, only a one-line W/L/D summary (with --out the file still "
             "gets it)"},
    {"name": "--out", "metavar": "FILE",
     "help": "Write the transcript to FILE and a one-line W/L/D summary to stdout"},
    {"name": "--max-decisions", "kind": "int", "metavar": "N",
     "help": "Stop each game / match after N decisions (reported as incomplete)"},
    {"name": "--timing", "kind": "flag",
     "help": "Print the throughput after the run (with --quiet the engine also runs without "
             "narrative)"},
    {"name": "--closed-decklist", "kind": "flag",
     "help": "Engine --closed-decklist: each seat's opponent list withheld"},
    {"name": "--selfcheck-state", "kind": "flag",
     "help": "Engine --selfcheck-state: the state format's self-checks at every decision"},
    {"name": "--selfcheck-knowledge", "kind": "flag",
     "help": "Engine --selfcheck-knowledge: the knowledge sidecar against the truth"},
    {"name": "--rmstate-dump", "metavar": "FILE",
     "help": "Engine --rmstate-dump FILE (with --games > 1, FILE.<i> per match)"},
    {"name": "--knowledge-a", "kind": "choice", "choices": ("off", "expert", "expert+belief"),
     "help": "Engine --knowledge-a: seat A's knowledge level (default: the engine's, expert)"},
    {"name": "--knowledge-b", "kind": "choice", "choices": ("off", "expert", "expert+belief"),
     "help": "Engine --knowledge-b (see --knowledge-a)"},
    {"name": "--engine-flags", "metavar": "FLAGS",
     "help": "Further engine flags, one quoted string (also read from $ROBOMAGE_ENGINE_FLAGS)"},
    {"name": "--binary", "default": DEFAULT_BINARY,
     "help": "Engine binary (default: bin/$ROBOMAGE_BUILD/robomage, debug when unset)"},
)


def build_parser() -> argparse.ArgumentParser:
    """The driver's parser, from ``CLI_FLAGS`` (argparse only)."""
    p = argparse.ArgumentParser(prog="rm_driver.py", description=__doc__.split("\n")[0])
    for f in CLI_FLAGS:
        kind = f.get("kind", "str")
        kw = {"help": f.get("help", "")}
        if kind == "flag":
            p.add_argument(f["name"], action="store_true", **kw)
            continue
        if f.get("metavar"):
            kw["metavar"] = f["metavar"]
        if kind == "int":
            kw["type"] = int
        elif kind == "choice":
            kw["choices"] = list(f["choices"])
        p.add_argument(f["name"], default=f.get("default"), **kw)
    return p


# ── Seeds ───────────────────────────────────────────────────────────────────────────────────

_M32, _M64, _M128 = (1 << 32) - 1, (1 << 64) - 1, (1 << 128) - 1
_PCG_MULT = 0x2360ED051FC65DA44385DF649FCCF645


def _seed_sequence_words(seed: int) -> list:
    """numpy ``SeedSequence(seed).generate_state(4, uint64)`` (pool size 4, no spawn key)."""
    entropy = []
    while True:
        entropy.append(seed & _M32)
        seed >>= 32
        if not seed:
            break
    hash_const = 0x43B0D7E5

    def hashmix(value):
        nonlocal hash_const
        value = (value ^ hash_const) & _M32
        hash_const = (hash_const * 0x931E8875) & _M32
        value = (value * hash_const) & _M32
        return value ^ (value >> 16)

    def mix(x, y):
        r = (0xCA01F9DD * x - 0x4973F715 * y) & _M32
        return r ^ (r >> 16)

    pool = [hashmix(entropy[i] if i < len(entropy) else 0) for i in range(4)]
    for i in range(4):
        for j in range(4):
            if i != j:
                pool[j] = mix(pool[j], hashmix(pool[i]))
    for i in range(4, len(entropy)):
        for j in range(4):
            pool[j] = mix(pool[j], hashmix(entropy[i]))
    out, hash_b = [], 0x8B51F9DD
    for i in range(8):
        v = (pool[i % 4] ^ hash_b) & _M32
        hash_b = (hash_b * 0x58F38DED) & _M32
        v = (v * hash_b) & _M32
        out.append(v ^ (v >> 16))
    return [out[2 * k] | (out[2 * k + 1] << 32) for k in range(4)]


def engine_seed(seed: int) -> int:
    """The engine ``--seed`` of game seed ``seed``:
    ``np.random.Generator(PCG64(SeedSequence(seed))).integers(0, 2**31 - 1)``, computed without
    numpy (PCG64 XSL-RR, numpy's buffered 32-bit Lemire bounded draw)."""
    s = _seed_sequence_words(seed)
    inc = ((((s[2] << 64) | s[3]) << 1) | 1) & _M128
    state = (inc + ((s[0] << 64) | s[1])) & _M128
    state = (state * _PCG_MULT + inc) & _M128
    spare = []

    def next32():
        nonlocal state
        if spare:
            return spare.pop()
        state = (state * _PCG_MULT + inc) & _M128
        v = ((state >> 64) ^ state) & _M64
        rot = state >> 122
        v = ((v >> rot) | (v << ((64 - rot) & 63))) & _M64
        spare.append(v >> 32)
        return v & _M32

    span = 2 ** 31 - 1                       # values 0 .. 2**31 - 2
    m = next32() * span
    if (m & _M32) < span:
        threshold = (_M32 - (span - 1)) % span
        while (m & _M32) < threshold:
            m = next32() * span
    return m >> 32


def resolve_on_the_play(choice, seed=None) -> str:
    """The side ("A" / "B") ``--on-the-play`` ``choice`` puts on the play ("random": a coin flip
    seeded from ``seed``)."""
    key = str(choice or "a").lower()
    if key not in ON_THE_PLAY:
        raise ValueError(f"--on-the-play must be one of {'/'.join(ON_THE_PLAY)} (got {choice!r})")
    if key != "random":
        return key.upper()
    rng = random.SystemRandom() if seed is None else random.Random(f"on-the-play:{seed}")
    return rng.choice("AB")


# ── Agents ──────────────────────────────────────────────────────────────────────────────────

class ScriptedAgent:
    """A scripted tier (explore included): ``scripted_core.ScriptedCore`` on the seat's frame."""

    def __init__(self, spec: str):
        from scripted_core import ScriptedCore, preset_config
        self.core = ScriptedCore(preset_config(spec))
        self.label = "Scripted"

    def new_game(self):
        self.core.new_game()

    def decide(self, seat: SeatFrames, frame) -> int:
        return decide(self.core, seat, frame)


class FirstAgent:
    """Always action 0 (pass priority / the first choice)."""

    label = "Auto"

    def new_game(self):
        pass

    def decide(self, seat, frame) -> int:
        return 0


class ActionsAgent:
    """Fixed replies for this seat's decisions in order, then action 0."""

    label = "Actions"

    def __init__(self, actions):
        self.actions = [int(a) for a in actions]
        self._i = 0

    def new_game(self):
        pass

    def decide(self, seat, frame) -> int:
        if self._i < len(self.actions):
            self._i += 1
            return self.actions[self._i - 1]
        return 0


def is_scripted_spec(spec: str) -> bool:
    """Whether ``spec`` names a scripted tier (explore included)."""
    from scripted_core import preset_config
    try:
        preset_config(spec)
    except ValueError:
        return False
    return True


def make_agent(spec):
    """The agent a spec names (module docstring); an agent object is returned as is."""
    if not isinstance(spec, str):
        return spec
    s = spec.strip()
    low = s.lower()
    if low in ("first", "auto"):
        return FirstAgent()
    if low.startswith("actions:"):
        return ActionsAgent(a for a in s[len("actions:"):].split(",") if a.strip())
    if is_scripted_spec(s):
        return ScriptedAgent(s)
    raise ValueError(f"unknown agent spec {spec!r} for the RMSTATE driver: scripted tiers, explore, "
                     "explore:patient, first / auto or actions:<i,...> (play: specs run in the "
                     "engine: --play)")


# ── The engine ──────────────────────────────────────────────────────────────────────────────

def engine_command(binary, *, seed, deck_a=None, deck_b=None, bo3=False, narrative=True,
                   engine_flags=()) -> list:
    """The engine command line of one game (or bo3 match) with engine seed ``seed``, plus
    ``engine_flags`` and $ROBOMAGE_ENGINE_FLAGS."""
    cmd = [os.path.realpath(binary), "--machine", "--seed", str(seed)]
    if narrative:
        cmd.append("--narrative")
    if bo3:
        cmd.append("--bo3")
    if deck_a:
        cmd += ["--deck-a", deck_a]
    if deck_b:
        cmd += ["--deck-b", deck_b]
    return cmd + list(engine_flags) + os.environ.get(ENGINE_FLAGS_ENV, "").split()


class Engine:
    """One engine process on the RMSTATE protocol: ``next_decision`` reads records (stdout) and
    lines (stderr) until the next decision frame, ``reply`` answers it."""

    def __init__(self, cmd, on_line):
        self.cmd = cmd
        self._on_line = on_line                # (bytes line) for every non-diagnostic stderr line
        self.diagnostics = []
        self.decoder = rmstate.StreamDecoder()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, cwd=BIN_DIR)
        self._out = bytearray()
        self._err = bytearray()
        self._out_eof = self._err_eof = False

    def next_decision(self, seats):
        """The next decision frame (each frame read before it fed to its seat's ``SeatFrames``
        in ``seats``), or None once the engine has exited cleanly."""
        out_fd, err_fd = self.proc.stdout.fileno(), self.proc.stderr.fileno()
        while True:
            payload = self._next_record()
            if payload is not None:
                frame = self.decoder.feed(payload, FRAME_TABLES)
                seat = seats.get(frame.viewer)
                if seat is None:
                    continue
                seat.feed(frame, self.decoder.schema)
                if frame.kind != "STATE" or frame.tables["header"][0]["terminal"] or "menu" not in frame.tables:
                    continue
                self._drain_stderr(err_fd, block=False)
                return frame
            if self._out_eof:
                self._drain_stderr(err_fd, block=True)
                rc = self.proc.wait()
                if rc != 0:
                    raise RuntimeError(f"engine exited with code {rc} before the game finished: "
                                       + " | ".join(self.diagnostics[-3:]))
                return None
            fds = [out_fd] + ([] if self._err_eof else [err_fd])
            ready, _, _ = select.select(fds, [], [])
            if err_fd in ready:
                self._read_stderr_chunk(err_fd)
            if out_fd in ready:
                chunk = os.read(out_fd, _READ_CHUNK)
                if chunk:
                    self._out += chunk
                else:
                    self._out_eof = True

    def reply(self, value: int):
        self.proc.stdin.write(f"{value}\n".encode())
        self.proc.stdin.flush()

    def close(self):
        if self.proc is None:
            return
        try:
            self.proc.kill()
            for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                f.close()
            self.proc.wait()
        except OSError:
            pass
        self.proc = None

    def _next_record(self):
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

    def _read_stderr_chunk(self, fd):
        chunk = os.read(fd, _READ_CHUNK)
        if not chunk:
            self._err_eof = True
            if self._err:
                self._err += b"\n"
        self._err += chunk
        self._take_lines()

    def _drain_stderr(self, fd, block):
        """Everything the engine has written to stderr so far (``block``: until its end). The
        engine flushes a decision's narrative before writing its frame, so once the frame is read
        a non-blocking drain has every line that belongs before it."""
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
        self._take_lines()

    def _take_lines(self):
        while True:
            nl = self._err.find(b"\n")
            if nl < 0:
                return
            line = bytes(self._err[:nl])
            del self._err[:nl + 1]
            if line.startswith(_DIAGNOSTIC_PREFIXES):
                self.diagnostics.append(line.decode("utf-8", errors="replace"))
                _write_diagnostic(line)
                continue
            self._on_line(line)


def _write_diagnostic(line):
    """One engine diagnostic line to this process's stderr in a single write, so lines from engines
    run in parallel into one log never splice together."""
    sys.stderr.flush()
    try:
        fd = sys.stderr.fileno()
    except (AttributeError, OSError):  # a captured stderr with no descriptor (io.UnsupportedOperation)
        sys.stderr.write(line.decode("utf-8", errors="replace") + "\n")
        return
    os.write(fd, line + b"\n")


# ── One decision as the transcript reads it ─────────────────────────────────────────────────

class _CardNames:
    """A CARDS frame's card names by card id: the display name (the front face's, a split
    card's faces joined by "/", accents folded: as the card vocabulary names it) and the front
    face's own name (the hand order's key)."""

    def __init__(self, schema, cards_frame):
        split = schema.enums["layout"].index("SPLIT") if "SPLIT" in schema.enums["layout"] else -1
        faces, layouts = {}, {}
        for r in cards_frame.tables.get("card", []):
            faces.setdefault(r["card"], []).append((r["face"], r["name"]))
            if r["face"] == 0:
                layouts[r["card"]] = r["layout"]
        self.front, self.display = {}, {}
        for cid, fs in faces.items():
            fs.sort()
            self.front[cid] = fs[0][1]
            joined = layouts.get(cid) == split
            self.display[cid] = fold_name("/".join(n for _f, n in fs) if joined else fs[0][1])


class Decision:
    """A decision frame as the driver and its transcript read it."""

    def __init__(self, frame, seat: SeatFrames, names: _CardNames, index: int):
        self.frame = frame
        self.seat = frame.viewer                  # "A" / "B"
        self.index = index                        # 0-based decision of the game / match
        en = seat.en
        t = frame.tables
        self.header = t["header"][0]
        cats = en["action_category"]
        self.menu = t["menu"]
        self.cats = [cats[r["category"]] for r in self.menu]
        self.turn = t["turn"][0] if t.get("turn") else None
        match = t["match"][0] if t.get("match") else None
        self.a_on_play = match is not None and match["plays_first"] == _SEAT_A
        self.step_name = en["step"][self.turn["step"]] if self.turn else ""
        self._seat = seat
        self._names = names

    @property
    def sideboarding(self) -> bool:
        return bool(self.header["sideboarding"])

    def is_pregame(self) -> bool:
        cats = set(self.cats)
        return bool(cats) and any(cats <= c for c in _PREGAME_CATS)

    def active_is_a(self) -> bool:
        """The active seat; while sideboarding, the other seat (the transcript's reading)."""
        if self.turn is None or self.sideboarding:
            return self.seat != "A"
        return self.turn["active"] == _SEAT_A

    def display_turn(self) -> int:
        """The engine's 1-based turn number (its narrative's TURN N)."""
        return (self.turn["turn"] if self.turn else 0) + 1

    def hand(self) -> list:
        """The seat's hand card names (none while sideboarding), at most HAND_LINE_CAP, in
        (token, owner, front-face name, handle) order."""
        if self.sideboarding:
            return []
        hand = self._seat.en["zone"].index("HAND")
        objs = [o for o in self.frame.tables.get("object", []) if o["zone"] == hand and o["owner_rel"] == _SELF_REL]
        front = self._names.front

        def key(o):
            return (o["token"], o["owner"], front.get(o["card"], ""), o["handle"])
        out = []
        for o in sorted(objs, key=key)[:HAND_LINE_CAP]:
            if o["card"] != _NONE16:
                out.append(self._names.display.get(o["card"], f"?({o['card']})"))
        return out

    def description(self, i) -> str:
        r = self.menu[i]
        return r["description"] or self.cats[i]

    def action_lines(self) -> list:
        out = []
        for i, r in enumerate(self.menu):
            ordv = r["ordinal"]
            out.append(f"  {i:>2}: {self.description(i)}" + (f"  [#{ordv}]" if ordv >= 0 else ""))
        return out

    def wire_reply(self, action: int) -> int:
        """The integer sent for menu row ``action``: -1 for an attacker/blocker declaration's
        last row (its confirm row), a concession sentinel as is."""
        if action >= 0 and action == len(self.menu) - 1 and _DECLARATION_CATS.intersection(self.cats):
            return -1
        return action


# ── Games ───────────────────────────────────────────────────────────────────────────────────

class GameRecord:
    """One driven game (bo1) or match (bo3)."""

    __slots__ = ("reward", "decisions", "capped", "actions", "engine_seed", "game_results",
                 "diagnostics")

    def __init__(self):
        self.reward = 0.0            # sum of the games' scores, Player A's perspective
        self.decisions = 0
        self.capped = False          # stopped by max_decisions
        self.actions = []            # every reply chosen (menu indices / sentinels)
        self.engine_seed = None
        self.game_results = []       # per game: (winner "A" / "B" / None for a draw, A on the play)
        self.diagnostics = []        # the engine's diagnostic lines

    @property
    def winner(self):
        """"A" / "B" by the score's sign; None for a draw or a capped run."""
        if self.capped:
            return None
        return "A" if self.reward > 0 else ("B" if self.reward < 0 else None)

    @property
    def drawn_games(self) -> int:
        return sum(1 for g in self.game_results if g[0] is None)


def play_game(agent_a, agent_b, *, binary=DEFAULT_BINARY, deck_a=None, deck_b=None, bo3=False,
              seed=None, max_decisions=None, narrative=True, engine_flags=(), on_narrative=None,
              on_decision=None, on_action=None) -> GameRecord:
    """Plays one game (bo1) or match (bo3) between two agents (module docstring).

    Hooks: ``on_narrative(lines)`` gets the engine narrative read before each decision and at
    the end; ``on_decision(d)`` each ``Decision`` before its agent decides, ``on_action(d, action)``
    after it."""
    rec = GameRecord()
    if seed is not None:
        random.seed(seed)
    for agent in (agent_a, agent_b) if agent_b is not agent_a else (agent_a,):
        agent.new_game()
    lines = []
    rec.engine_seed = (engine_seed(seed) if seed is not None
                       else random.SystemRandom().randrange(2 ** 31 - 1))
    cmd = engine_command(binary, seed=rec.engine_seed, deck_a=deck_a, deck_b=deck_b, bo3=bo3,
                         narrative=narrative, engine_flags=engine_flags)
    a_on_play = True

    def on_line(line):
        if line.startswith(b"GAME_RESULT:"):
            winner = "A" if b"Player A wins" in line else ("B" if b"Player B wins" in line else None)
            rec.reward += 1.0 if winner == "A" else (-1.0 if winner == "B" else 0.0)
            rec.game_results.append((winner, a_on_play))
        lines.append(line.decode("utf-8", errors="replace"))

    def flush():
        if lines and on_narrative is not None:
            on_narrative(list(lines))
        lines.clear()

    seats = {"A": SeatFrames("A"), "B": SeatFrames("B")}
    names = {}
    engine = Engine(cmd, on_line)
    try:
        while True:
            frame = engine.next_decision(seats)
            flush()
            if frame is None:
                break
            if max_decisions is not None and rec.decisions >= max_decisions:
                rec.capped = True
                break
            seat = seats[frame.viewer]
            if names.get(frame.viewer, (None,))[0] is not seat.cards:
                names[frame.viewer] = (seat.cards, _CardNames(seat.schema, seat.cards_frame))
            d = Decision(frame, seat, names[frame.viewer][1], rec.decisions)
            a_on_play = d.a_on_play
            if on_decision is not None:
                on_decision(d)
            agent = agent_a if d.seat == "A" else agent_b
            action = int(agent.decide(seat, frame))
            if on_action is not None:
                on_action(d, action)
            rec.actions.append(action)
            engine.reply(d.wire_reply(action))
            rec.decisions += 1
        flush()
    finally:
        rec.diagnostics = list(engine.diagnostics)
        engine.close()
    return rec


# ── Transcripts and results ─────────────────────────────────────────────────────────────────

def _pct(w, n):
    return f"{w}/{n} ({100 * w / n:.1f}%)" if n else f"{w}/{n} (n/a)"


def per_game_split_lines(records, subject="Player A"):
    """The bo3 per-game-index and post-board play/draw lines."""
    per_game = {}
    for rec in records:
        for gi, (winner, a_on_play) in enumerate(rec.game_results):
            c = per_game.setdefault(gi, {"wins": 0, "played": 0, "play_wins": 0, "play_n": 0,
                                         "draw_wins": 0, "draw_n": 0})
            won = winner == "A"
            c["played"] += 1
            c["wins"] += int(won)
            key = "play" if a_on_play else "draw"
            c[f"{key}_n"] += 1
            c[f"{key}_wins"] += int(won)
    if not per_game or max(per_game) == 0:
        return []
    per_idx = "  ".join(f"g{gi + 1} {_pct(c['wins'], c['played'])}" for gi, c in sorted(per_game.items()))
    post = [c for gi, c in per_game.items() if gi > 0]
    pw, pn = sum(c["play_wins"] for c in post), sum(c["play_n"] for c in post)
    dw, dn = sum(c["draw_wins"] for c in post), sum(c["draw_n"] for c in post)
    return [f"per-game ({subject}): {per_idx}",
            f"post-board by seat ({subject}): on the play {_pct(pw, pn)} | on the draw {_pct(dw, dn)}"
            "   [g1 vs g2-3 alone is confounded by the play/draw rule]"]


def format_timing(records, wall, bo3=False) -> str:
    """One-line throughput summary of a run."""
    decisions = sum(r.decisions for r in records)
    completed = sum(1 for r in records if not r.capped)
    if bo3:
        head = f"matches={len(records)} games={sum(len(r.game_results) for r in records)}"
        games = sum(len(r.game_results) for r in records)
    else:
        games = len(records)
        head = f"games={games}"
    wall = max(wall, 1e-9)
    return (f"{head} completed={completed} decisions={decisions} wall={wall:.3f}s "
            f"games/s={games / wall:.2f} decisions/s={decisions / wall:.0f} "
            f"ms/decision={1000 * wall / max(1, decisions):.3f}")


def run_games(agent_a="scripted", agent_b="scripted", *, label_a="A", label_b="B",
              binary=DEFAULT_BINARY, deck_a=None, deck_b=None, n_games=1, bo3=False, seed=None,
              transcript="compact", out=None, max_decisions=None, narrative=True, engine_flags=(),
              on_game_end=None, dump=None):
    """Plays ``n_games`` games (bo1) or matches (bo3) and writes their transcript (module
    docstring) to ``out`` (default stdout). Game i uses seed ``seed + i``. ``agent_a`` /
    ``agent_b`` are specs or agents (one instance per seat each match unless the same object is
    passed twice). ``dump`` names an ``--rmstate-dump`` file (``FILE.<i>`` per match when
    ``n_games > 1``). ``on_game_end(record)`` gets each ``GameRecord``.

    Returns (wins, losses, draws), Player A's: a record with no winner counts as a draw (a
    drawn game, CR 104.4a, or a game with no result), and every draw, and every drawn game inside
    a decided match, is announced and its log saved to ``draw_<timestamp>.txt``. A run stopped by
    ``max_decisions`` is reported as incomplete and not counted."""
    if transcript not in TRANSCRIPTS:
        raise ValueError(f"unknown transcript mode {transcript!r}")
    verbose = transcript == "verbose"
    stream = out if out is not None else sys.stdout
    agent_a, agent_b = make_agent(agent_a), make_agent(agent_b)
    wins = losses = draws = incomplete = 0
    unit = "match" if bo3 else "game"
    records = []
    for i in range(n_games):
        game_seed = seed + i if seed is not None else None
        flags = list(engine_flags)
        if dump:
            flags += ["--rmstate-dump", dump if n_games == 1 else f"{dump}.{i}"]
        log = []
        state = {"turn": 0, "pregame": False, "hand": {"A": [], "B": []}}

        def emit(line):
            log.append(line)
            if transcript != "quiet":
                print(line, file=stream, flush=True)

        def on_narrative(lines):
            lines = [ln for ln in lines if ln.strip()]
            if lines:
                for ln in (["--- Narrative ---"] + [f"  {ln}" for ln in lines]) if verbose else lines:
                    emit(ln)

        def on_decision(d):
            if transcript == "quiet":
                return
            state["hand"][d.seat] = d.hand()
            if d.is_pregame():
                if not state["pregame"]:
                    emit("--- Pregame ---")
                    state["pregame"] = True
            elif d.display_turn() != state["turn"]:
                state["turn"] = d.display_turn()
                emit(f"--- Turn {state['turn']} (Player {'A' if d.active_is_a() else 'B'}) ---")
                emit(f"  PA: {', '.join(state['hand']['A']) or '(empty)'}")
                emit(f"  PB: {', '.join(state['hand']['B']) or '(empty)'}")

        def on_action(d, action):
            if transcript == "quiet":
                return
            label = label_a if d.seat == "A" else label_b
            if verbose:
                emit("")
                emit(f"--- Decision {d.index + 1} ---")
                emit("  Actions:")
                for ln in d.action_lines():
                    emit(ln)
                desc = d.description(action) if 0 <= action < len(d.menu) else "?"
                emit(f"  >> {label}/{d.seat}: {action} ({desc})")
            else:
                chosen = d.cats[action] if 0 <= action < len(d.cats) else str(action)
                emit(f"[{d.index:4d}] P{d.seat} ({label}/{d.seat})  {d.step_name:<14}  "
                     f"choices={len(d.cats)}  available={d.cats}  -> {action} ({chosen})")

        if n_games > 1:
            emit(f"--- {unit.capitalize()} {i + 1}/{n_games} ---")
        rec = play_game(agent_a, agent_b, binary=binary, deck_a=deck_a, deck_b=deck_b, bo3=bo3,
                        seed=game_seed, max_decisions=max_decisions, narrative=narrative,
                        engine_flags=flags, on_narrative=on_narrative, on_decision=on_decision,
                        on_action=on_action)
        if on_game_end is not None:
            on_game_end(rec)
        records.append(rec)

        draw_note = None
        result = None
        if rec.capped:
            incomplete += 1
            if transcript != "quiet":
                print(f"\n=== stopped after {rec.decisions} decisions (max-decisions cap; no winner) ===",
                      file=stream, flush=True)
        elif rec.reward > 0:
            wins += 1
            result = f"{label_a}/A wins"
        elif rec.reward < 0:
            losses += 1
            result = f"{label_b}/B wins"
        else:
            draws += 1
            result = "DRAW"
            draw_note = ("DRAW (both players lost at once, CR 104.4a)" if rec.drawn_games
                         else "DRAW (no result — should not occur)")
        if not rec.capped and draw_note is None and rec.drawn_games:
            draw_note = (f"{rec.drawn_games} drawn game(s) in this match "
                         "(both players lost at once, CR 104.4a)")
        if draw_note is not None:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_path = f"draw_{stamp}.txt"
            with open(log_path, "w") as f:
                f.write("\n".join(log) + "\n")
            print(f"\n=== {draw_note} — review; full log saved to {log_path} ===", file=stream, flush=True)
        if not rec.capped and transcript != "quiet":
            if n_games > 1:
                print(f"  {unit} {i + 1:2d}/{n_games}: {result}  (W:{wins} L:{losses} D:{draws})",
                      file=stream, flush=True)
            else:
                print(f"\n=== {result} ===", file=stream, flush=True)

    if n_games > 1 and transcript != "quiet":
        total = wins + losses + draws
        extra = f", {incomplete} incomplete" if incomplete else ""
        print(f"\n{wins}W / {losses}L / {draws}D over {n_games} {unit}{'es' if bo3 else 's'} "
              f"({100 * wins / total if total else 0:.1f}% Player A win rate){extra}",
              file=stream, flush=True)
        for line in per_game_split_lines(records):
            print(f"  {line}", file=stream, flush=True)
    return wins, losses, draws


# ── CLI ─────────────────────────────────────────────────────────────────────────────────────

def _engine_flags(args) -> list:
    flags = []
    for dest, flag in (("closed_decklist", "--closed-decklist"), ("selfcheck_state", "--selfcheck-state"),
                       ("selfcheck_knowledge", "--selfcheck-knowledge")):
        if getattr(args, dest):
            flags.append(flag)
    for dest, flag in (("knowledge_a", "--knowledge-a"), ("knowledge_b", "--knowledge-b")):
        if getattr(args, dest):
            flags += [flag, getattr(args, dest)]
    if args.engine_flags:
        flags += args.engine_flags.split()
    return flags


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    bo3 = args.format == "bo3"
    deck_a, deck_b = args.deck_a, args.deck_b or args.deck_a
    spec_a, spec_b = args.player_a, args.player_b
    try:
        side = resolve_on_the_play(args.on_the_play, args.seed)
        if side == "B":
            spec_a, spec_b, deck_a, deck_b = spec_b, spec_a, deck_b, deck_a
        agent_a, agent_b = make_agent(spec_a), make_agent(spec_b)
    except ValueError as e:
        parser.error(str(e))
    quiet = args.quiet and not args.out
    transcript = "quiet" if quiet else ("verbose" if args.verbose else "compact")
    unit = f"{'match' if bo3 else 'game'}{'es' if bo3 else 's'}"
    records = []

    def run(stream):
        if not quiet:
            note = " (player B's side on the play: the sides swap engine seats)" if side == "B" else ""
            print(f"=== {agent_a.label}/A [{spec_a}] ({deck_a} deck) vs {agent_b.label}/B [{spec_b}] "
                  f"({deck_b} deck) — {args.games} {unit}{note} ===\n", file=stream, flush=True)
        return run_games(agent_a, agent_b, label_a=agent_a.label, label_b=agent_b.label,
                         binary=args.binary, deck_a=deck_a, deck_b=deck_b, n_games=args.games, bo3=bo3,
                         seed=args.seed, transcript=transcript, out=stream,
                         max_decisions=args.max_decisions, narrative=not (args.timing and quiet),
                         engine_flags=_engine_flags(args), on_game_end=records.append,
                         dump=args.rmstate_dump)

    t0 = time.perf_counter()
    if args.out:
        with open(args.out, "w") as fh:
            wins, losses, draws = run(fh)
    else:
        wins, losses, draws = run(sys.stdout)
    wall = time.perf_counter() - t0
    if args.out or args.quiet:
        print(f"{deck_a} vs {deck_b} [{spec_a} vs {spec_b}] {args.games} {unit}: {wins}W / {losses}L / "
              f"{draws}D (completed {wins + losses + draws}){f' -> {args.out}' if args.out else ''}",
              flush=True)
    if args.timing:
        print(format_timing(records, wall, bo3=bo3), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
