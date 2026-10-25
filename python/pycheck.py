#!/usr/bin/env python3
"""The kept Python's own checks: the second step of ``robomage_check``'s ``smoke`` tier.

The smoke games test the scripted agent and the RMSTATE driver by playing; these checks test what
playing does not:

  decode    ``python/rmstate.py``'s decode of three ``--rmstate-dump`` files renders exactly the
            JSON lines the engine renders for them (``--rmstate-read DUMP --state-dump json``),
            byte for byte, and both equal the ``DUMP.jsonl`` the engine wrote live with
            ``--state-dump json``. The dumps: a bo1 game and a bo3 match of the engine's
            in-process explore agents, and a bo3 ``--machine-format rmstate`` session under
            ``--closed-decklist`` played by ``rm_driver.py``'s Python explore agents.
  refusals  each mutation of a dump's first STATE frame (another format version, another schema
            hash, an unknown frame kind, a known table at another row width) is refused by both
            decoders; an unknown table block is skipped by both, with the same JSON (listed under
            ``"skipped"``).
  selfkill  the scripted agent's self-lethal guard (``scripted_core`` docstring) in sculpted
            positions: at 2 life it does not cast a Price of Progress that would deal it 2 (also
            when it would kill the opponent too, while the opponent's board has it dead next
            turn), nor a Dismember whose Phyrexian pip only life can pay; at 3 life it casts the
            same Price of Progress (the position does tempt it).
  guides    ``sideboard_guides.json`` against ``docs/sideboard_guides.md``'s tables (the same draw lands
            and rows) and the league deck files (each row balanced, its In cards and Draw +1 card in
            the deck's sideboard and its Out cards and draw land in its main deck with enough
            copies, no card both in and out, each deck held out once as each side); and the
            ``scripted:hard+sb`` tier's phases over a simulated engine sideboard menu (in-first,
            one-shot locks, Done only while balanced): every row's target on the play and on the
            draw from the registered list, game 3 after a play/draw flip (one swap: the land swap
            undone or applied), a deck with no draw land the same on both, a held-out pair and an
            unknown opponent list Done at once.
  sideboard ``SIDEBOARD_MATCHES``: bo3 matches of ``scripted:hard+sb`` on both seats through
            ``rm_driver.py``; at each seat's first decision of games 2 and 3 its main deck (the
            frame's ``own_deck``) is its guide's target (the registered main where held out) for
            the play or draw the game results give (the loser plays first; after a draw the same
            player again), which the frame's ``match.plays_first`` also says; and the narrative's
            swap count of each phase is the number of cards the main deck changed by.
  rmtb      ``python/rmtb.py`` reads the RMTB files the C++ writer wrote (``src/featurize/rmtb.cpp``;
            ``robomage_check``'s smoke tier writes two samples, an array of every dtype and a text
            table, and their meta and values, ``--rmtb-samples FILE``) to exactly those values,
            floats bit for bit, the f16 halves equal to ``struct``'s rounding of their sources; and
            the committed text table, when there is one, has zero rows outside its rows_meta.
  featurize ``python/rmtb.py`` reads the featurizer's files ``robomage_check``'s featurize tier wrote
            and lists (``--featurize-expected FILE``: the exported card, ability and part tables, a
            tensors file and a targets file, each array's FNV-1a 64 of the values the C++ side
            built): every array's values repacked hash the same, every file carries the spec's
            hash (FNV-1a 64 of ``spec.json``'s bytes) and vocab hash, and the shapes and indices
            agree with the spec (the tables' rows, widths and column names, the key list, F, C, K
            and Q, every relation, action, card and ability index in range).
  traj      ``python/rmtraj.py`` reads the trajectory episodes ``robomage_check``'s traj tier lists
            (``--traj-expected FILE``: per episode its path, each step's seat, decision key, reply,
            μ, policy version, teacher label and evaluations as the C++ reader read them, and
            tensors files holding the C++ reader's expansion of every step's tensors, and of the
            omniscient tensors and the targets of an omniscient episode) to exactly those values.
  open_vocab  (not in the default set: the ``featurize`` tier runs it) the S5 acceptance's inputs: the
            gate's sources of the open-vocabulary fixtures (``tests/open_vocab/inputs.json``'s
            ``gate_sources``, decision logs under ``tests/open_vocab/sources/``) replayed by the engine
            (``--machine --rmstate-dump``) and featurized by ``rmfeat`` beside it (seats A and B, the
            omniscient viewer, over its own ``--cards-frame`` tables): every fixture decision of them
            still listed in ``inputs.json`` has the digest recorded there (sha256 of its seat and its
            omniscient decision's arrays as written), the inputs on which the migrated checkpoints were
            shown to reproduce S0's outputs (``learn/open_vocab_acceptance.py``).
  driver    ``rm_driver.py`` derives the engine seeds it always has (pinned values), and a game it
            plays (scripted against explore) on an interpreter without site-packages (``-S``)
            imports no module from outside the standard library and this directory: the kept
            Python is stdlib only.

Stdlib only. Prints one ``  OK   …`` line per passing check and one ``FAIL: …`` line per finding;
exit 1 on any finding::

    python3 python/pycheck.py [--binary bin/debug/robomage] [--out-dir DIR] [--part decode,selfkill,guides]
                              [--rmtb-samples FILE] [--featurize-expected FILE] [--traj-expected FILE]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import rm_driver  # noqa: E402
import rmstate  # noqa: E402
import sideboard_guides  # noqa: E402
import rmtb  # noqa: E402
import rmtraj  # noqa: E402

PARTS = ("decode", "refusals", "selfkill", "driver", "guides", "sideboard", "rmtb", "featurize", "traj")
OPTIONAL_PARTS = ("open_vocab",)   # run only when named
# The open-vocabulary fixtures (docs/open_vocab_plan.md S0 / S5).
OPEN_VOCAB = os.path.join(os.path.dirname(_HERE), "tests", "open_vocab")
# The decision arrays of a tensors file a digest covers, in order, with the count naming their rows.
OPEN_VOCAB_TENSORS = (("tok_type", "tok_type"), ("tok_num", "tok_type"), ("tok_cat", "tok_type"),
                      ("tok_card", "tok_type"), ("tok_ability", "tok_type"), ("tok_mask", "tok_type"),
                      ("rel", "rel"), ("act_tok", "act_tok"), ("act_mask", "act_tok"))
# The committed oracle-text table (docs/featurizer.md §5).
TEXT_TABLE = os.path.join(os.path.dirname(_HERE), "bin", "resources", "embeddings", "oracle_text.rmtb")
_TIMEOUT_S = 900
# The in-process agents' dumps: (deck_a, deck_b, bo3, game seed).
ENGINE_DUMPS = (("league/ur_delver", "league/gw_maverick", False, 41),
                ("league/reanimator", "league/uw_control", True, 42))
# The driver's closed-decklist session: (deck_a, deck_b, bo3, seed).
STREAM_DUMP = ("league/wrb_energy", "league/bw_dnt", True, 43)
# rm_driver.engine_seed's values (numpy's SeedSequence → PCG64 derivation, computed without numpy).
PINNED_ENGINE_SEEDS = ((0, 1826701614), (1, 1016164991), (2, 1798679647), (7, 2029167940),
                       (1001, 1940026651), (2147483647, 1751933748), (4294967295, 409763535),
                       (4294967296, 1121323793), (1099511627781, 30406389), (13532794167, 596834746),
                       (1641108517, 1187322509))
# Run in the probe after a driven game: every imported module with a file outside the standard
# library and the kept Python's directory (argv[2]).
_FOREIGN_MODULES = ("import os, sysconfig; "
                    "roots = {os.path.realpath(sysconfig.get_paths()[k]) for k in ('stdlib', 'platstdlib')}; "
                    "roots.add(os.path.realpath(sys.argv[2])); "
                    "files = {n: getattr(m, '__file__', None) for n, m in list(sys.modules.items())}; "
                    "print(' '.join(sorted(n for n, f in files.items() if f and not any("
                    "os.path.realpath(f).startswith(r + os.sep) for r in roots))))")


def _engine_json(binary, path):
    """(exit code, stdout, stderr) of the engine's --rmstate-read PATH --state-dump json."""
    r = subprocess.run([binary, "--rmstate-read", path, "--state-dump", "json"], cwd=rm_driver.BIN_DIR,
                       capture_output=True, timeout=_TIMEOUT_S)
    return r.returncode, r.stdout, r.stderr.decode(errors="replace")


def _py_json(path):
    with open(path, "rb") as fp:
        return b"".join(line.encode("utf-8", "surrogateescape") + b"\n" for line in rmstate.json_lines(fp))


def _first_diff_line(a, b):
    la, lb = a.split(b"\n"), b.split(b"\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return i + 1
    return min(len(la), len(lb)) + 1


def check_dump(binary, path, label):
    """The decode check on one dump; returns failure messages."""
    rc, cpp, err = _engine_json(binary, path)
    summary = [ln for ln in err.splitlines() if ln.startswith("RMSTATE_READ:")]
    fails = []
    if rc != 0 or "ERROR:" in err:
        errs = [ln for ln in err.splitlines() if "ERROR:" in ln]
        fails.append(f"{label}: --rmstate-read exit {rc}: {errs[:3]} {summary}")
    try:
        py = _py_json(path)
    except rmstate.RMStateError as exc:
        fails.append(f"{label}: the Python decoder refused the dump: {exc}")
        py = None
    if py is not None and py != cpp:
        fails.append(f"{label}: the Python JSON differs from the engine's at line {_first_diff_line(py, cpp)}")
    live = path + ".jsonl"
    if not os.path.exists(live):
        fails.append(f"{label}: no live --state-dump json file {live}")
    else:
        with open(live, "rb") as fp:
            if fp.read() != cpp:
                fails.append(f"{label}: the live --state-dump json file differs from --rmstate-read's JSON")
    if not fails:
        print(f"  OK   decode {label}: {summary[0] if summary else ''}", flush=True)
    return fails


def _engine_dump(binary, work, deck_a, deck_b, bo3, seed):
    """Plays one in-process explore game (match) with a dump; returns (dump path, failures)."""
    path = os.path.join(work, f"engine_{seed}.rms")
    cmd = [binary, "--agent-a", "explore", "--agent-b", "explore", "--game-seed", str(seed), "--deck-a", deck_a,
           "--deck-b", deck_b, "--rmstate-dump", path, "--state-dump", "json"]
    if bo3:
        cmd.append("--bo3")
    r = subprocess.run(cmd, cwd=rm_driver.BIN_DIR, capture_output=True, timeout=_TIMEOUT_S)
    if r.returncode != 0 or not os.path.exists(path):
        return None, [f"engine {deck_a} vs {deck_b} seed {seed}: exit {r.returncode}: "
                      f"{r.stderr.decode(errors='replace')[-300:]}"]
    return path, []


def _stream_dump(binary, work, deck_a, deck_b, bo3, seed):
    """Plays the driver's closed-decklist session with a dump; returns (dump path, failures)."""
    path = os.path.join(work, f"stream_{seed}.rms")
    cwd = os.getcwd()
    os.chdir(work)  # a draw's saved log stays in the scratch directory
    try:
        wld = rm_driver.run_games("explore", "explore", binary=binary, deck_a=deck_a, deck_b=deck_b, bo3=bo3,
                                  seed=seed, transcript="quiet", dump=path,
                                  engine_flags=["--closed-decklist", "--state-dump", "json"])
    finally:
        os.chdir(cwd)
    if not os.path.exists(path):
        return None, [f"driver {deck_a} vs {deck_b} seed {seed}: no dump written ({wld})"]
    return path, []


def check_decode(binary, work):
    """The decode part; returns (failures, the first dump for the refusals part)."""
    fails = []
    dumps = []
    for deck_a, deck_b, bo3, seed in ENGINE_DUMPS:
        path, f = _engine_dump(binary, work, deck_a, deck_b, bo3, seed)
        fails += f
        if path:
            dumps.append((path, f"engine {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} "
                                f"{'bo3' if bo3 else 'bo1'} seed {seed}"))
    deck_a, deck_b, bo3, seed = STREAM_DUMP
    path, f = _stream_dump(binary, work, deck_a, deck_b, bo3, seed)
    fails += f
    if path:
        dumps.append((path, f"driver {os.path.basename(deck_a)} vs {os.path.basename(deck_b)} "
                            f"{'bo3' if bo3 else 'bo1'} --closed-decklist seed {seed}"))
    for path, label in dumps:
        fails += check_dump(binary, path, label)
    return fails, dumps[0][0] if dumps else None


# ── refusals: frames both decoders must refuse or skip ────────────────────────────────────────

def _rebuild(header, blocks):
    """A frame payload from (version, kind, viewer, hash) and [(id, width, rows, body)]."""
    version, kind, viewer, h = header
    out = bytearray(struct.pack("<4sHBBQII", b"RMSF", version, kind, viewer, h, len(blocks), 0))
    for tid, width, rows, body in blocks:
        out += struct.pack("<HHIII", tid, 0, width, rows, 0) + body
        out += b"\0" * ((-len(out)) % 8)
    return bytes(out)


def _mutants(state_payload):
    """(label, payload, what the engine reports, whether the decoders refuse it) per mutation of
    one STATE frame."""
    header, blocks = rmstate._blocks(state_payload)
    version, kind, viewer, h = header
    widen = next(i for i, b in enumerate(blocks) if b[0] != 1)
    tid, width, rows, body = blocks[widen]
    wide = b"".join(body[r * width:(r + 1) * width] + b"\0" for r in range(rows))
    wider = blocks[:widen] + [(tid, width + 1, rows, wide)] + blocks[widen + 1:]
    return [
        ("version", _rebuild((version + 1, kind, viewer, h), blocks), "format version", True),
        ("hash", _rebuild((version, kind, viewer, h ^ 1), blocks), "schema hash differs", True),
        ("kind", _rebuild((version, 200, viewer, h), blocks), "unknown frame kind", True),
        ("row width", _rebuild(header, wider), "has row width", True),
        ("unknown table", _rebuild(header, blocks + [(0xFFFE, 8, 1, b"\1" * 8)]),
         "blocks the decoding schema does not know", False),
    ]


def check_refusals(binary, dump_path, work):
    """Each mutation of the dump's first STATE frame against both decoders; returns failures."""
    with open(dump_path, "rb") as fp:
        payloads = list(rmstate.records(fp))
    dec = rmstate.StreamDecoder()
    head = []
    state = None
    for p in payloads:
        if dec.feed(p).kind == "STATE":
            state = p
            break
        head.append(p)
    if state is None:
        return [f"refusals: {os.path.basename(dump_path)} has no STATE frame to mutate"]
    fails = []
    mutants = _mutants(state)
    for label, mutant, needle, refused in mutants:
        path = os.path.join(work, f"mutant_{label.replace(' ', '_')}.rms")
        with open(path, "wb") as fp:
            for p in head + [mutant]:
                fp.write(b"RMSTATE %d\n" % len(p) + p)
        rc, out, err = _engine_json(binary, path)
        if rc != 1 or needle not in err:
            fails.append(f"mutant {label}: the engine should report {needle!r} (exit {rc}): {err[-300:]}")
        try:
            py = _py_json(path)
            if refused:
                fails.append(f"mutant {label}: the Python decoder accepted it")
            elif py != out or b'"skipped":[65534]' not in py:
                fails.append(f"mutant {label}: the Python and engine JSON differ, or the block is not listed as skipped")
        except rmstate.RMStateError:
            if not refused:
                fails.append(f"mutant {label}: the Python decoder refused it")
    if not fails:
        print(f"  OK   refusals ({len(mutants)} mutations)", flush=True)
    return fails


# ── selfkill: the scripted agent's self-lethal guard ─────────────────────────────────────────

# A burn hand whose only other spells have no target: Price of Progress deals Player A 2 (its
# Fiery Islet) and Player B 4 (two Underground Seas, at 4 life).
_POP_POSITION = {"--deck-a": "league/burn", "--deck-b": "league/reanimator",
                 "--hand-a": "Price of Progress,Mountain,Mountain,Searing Blood,Searing Blood,"
                             "Smash to Smithereens,Smash to Smithereens",
                 "--library-a": "Mountain", "--battlefield-a": "Mountain,Mountain,Fiery Islet",
                 "--battlefield-b": "Underground Sea,Underground Sea", "--life-b": "4",
                 "--hand-b": "Island,Island,Island,Brainstorm,Ponder,Ponder,Brainstorm", "--library-b": "Island"}
# Dismember ({1}{B/P}{B/P}) with one black source: one pip is 2 life.
_DISMEMBER_POSITION = {"--deck-a": "league/bug", "--deck-b": "league/burn",
                       "--hand-a": "Dismember,Uro Titan of Natures Wrath,Island,Island,Ponder,Ponder,Ponder", "--library-a": "Island",
                       "--battlefield-a": "Island,Swamp", "--battlefield-b": "Goblin Guide,Mountain",
                       "--hand-b": "Mountain,Mountain,Mountain,Lightning Bolt,Lightning Bolt,Lightning Bolt,"
                                   "Lightning Bolt", "--library-b": "Mountain"}
# (label, position, overrides, the narrative line Player A's first turn must (not) show, shown).
SELFKILL_CASES = (
    ("Price of Progress at 2 life", _POP_POSITION, {"--life-a": "2"}, "Player A casts Price of Progress", False),
    ("Price of Progress at 2 life, killing both, dead on board", _POP_POSITION,
     {"--life-a": "2", "--battlefield-b": "Underground Sea,Underground Sea,Griselbrand"},
     "Player A casts Price of Progress", False),
    ("Dismember at 2 life, one black source", _DISMEMBER_POSITION, {"--life-a": "2"}, "Player A casts Dismember",
     False),
    ("Price of Progress at 3 life", _POP_POSITION, {"--life-a": "3"}, "Player A casts Price of Progress", True),
)
_SELFKILL_DECISIONS = 6     # the keeps and Player A's first-turn main phase


def check_selfkill(binary, work):
    """The selfkill part: each SELFKILL_CASES position, Player A's scripted agent deciding;
    returns failures."""
    fails = []
    cwd = os.getcwd()
    os.chdir(work)  # a draw's saved log stays in the scratch directory
    try:
        for label, position, overrides, line, shown in SELFKILL_CASES:
            opts = dict(position, **overrides)
            deck_a, deck_b = opts.pop("--deck-a"), opts.pop("--deck-b")
            flags = [x for kv in opts.items() for x in kv]
            narrative = []
            rec = rm_driver.play_game(rm_driver.make_agent("scripted"), rm_driver.make_agent("scripted"),
                                      binary=binary, deck_a=deck_a, deck_b=deck_b, seed=1,
                                      max_decisions=_SELFKILL_DECISIONS, engine_flags=flags,
                                      on_narrative=narrative.extend)
            seen = any(ln.strip().startswith(line) for ln in narrative)
            if seen != shown:
                fails.append(f"selfkill {label}: the narrative {'lacks' if shown else 'shows'} {line!r} "
                             f"(results {rec.game_results})")
    finally:
        os.chdir(cwd)
    if not fails:
        print(f"  OK   selfkill: {len(SELFKILL_CASES)} positions", flush=True)
    return fails


# ── the driver ────────────────────────────────────────────────────────────────────────────────

def check_driver(binary, work):
    """The driver part; returns failures."""
    fails = []
    for seed, want in PINNED_ENGINE_SEEDS:
        if rm_driver.engine_seed(seed) != want:
            fails.append(f"driver: engine_seed({seed}) = {rm_driver.engine_seed(seed)}, pinned {want}")
    probe = ("import sys; import rm_driver; rm_driver.run_games('scripted', 'explore', binary=sys.argv[1], "
             "deck_a='league/burn', seed=3, transcript='quiet'); " + _FOREIGN_MODULES)
    r = subprocess.run([sys.executable, "-S", "-c", probe, binary, _HERE], capture_output=True, text=True,
                       cwd=work, env=dict(os.environ, PYTHONPATH=_HERE), timeout=_TIMEOUT_S)
    if r.returncode != 0 or r.stdout.strip():
        fails.append(f"driver: a game imported {r.stdout.strip() or '?'} (exit {r.returncode}): {r.stderr[-300:]}")
    if not fails:
        print(f"  OK   driver: {len(PINNED_ENGINE_SEEDS)} engine seeds pinned, a game imports only the "
              "standard library and the kept Python", flush=True)
    return fails


# ── sideboard guides: the hard+sb tier's data and its sideboarding ──────────────────────────────

GUIDES_DOC = os.path.join(os.path.dirname(_HERE), "docs", "sideboard_guides.md")
_LIST_SPLIT = re.compile(r",\s*(?=\d+ )")


def _doc_cards(cell):
    """A guide table cell ("1 Hydroblast, 2 Endurance") as name -> copies."""
    out = {}
    for item in _LIST_SPLIT.split(cell.strip()):
        n, _, name = item.strip().partition(" ")
        out[name.strip()] = out.get(name.strip(), 0) + int(n)
    return out


def parse_guides_doc(text):
    """``docs/sideboard_guides.md``'s tables as (draw land per deck (None for none), per (deck,
    opponent) (in, out, draw card or None)), names as the doc spells them."""
    draw_land, rows = {}, {}
    section = None
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            continue
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if section == "Draw land per deck" and len(cells) == 2 and cells[0] != "Deck":
            draw_land[cells[0]] = None if cells[1] == "none" else cells[1].partition(" ")[2]
        elif len(cells) == 4 and cells[0] != "vs":
            rows[(section, cells[0])] = (_doc_cards(cells[1]), _doc_cards(cells[2]),
                                         None if cells[3] == "-" else cells[3])
    return draw_land, rows


def _guides_data_fails(g, doc_text):
    """The guides' JSON against the doc's tables and the league deck files; returns failures."""
    key = sideboard_guides.card_key
    fails = []
    doc_land, doc_rows = parse_guides_doc(doc_text)

    def keyed(cards):
        return {key(n): c for n, c in cards.items()}
    if set(g.draw_land) != set(g.decks) or set(doc_land) != set(g.decks):
        fails.append(f"guides: draw lands for {sorted(g.draw_land)} (doc {sorted(doc_land)}), "
                     f"league decks {sorted(g.decks)}")
    for deck, land in g.draw_land.items():
        if (land and key(land)) != (doc_land.get(deck) and key(doc_land[deck])):
            fails.append(f"guides: {deck}'s draw land is {land!r} in the JSON, {doc_land.get(deck)!r} in the doc")
        if land is not None and g.decks.get(deck, ({}, {}))[0].get(land, 0) < 1:
            fails.append(f"guides: {deck}'s draw land {land!r} is not in its main deck")
    json_rows = {(d, o): r for d, opps in g.guides.items() for o, r in opps.items()}
    if set(json_rows) != set(doc_rows):
        fails.append(f"guides: rows only in the JSON {sorted(set(json_rows) - set(doc_rows))}, only in the doc "
                     f"{sorted(set(doc_rows) - set(json_rows))}")
    for (deck, opp), r in sorted(json_rows.items()):
        where = f"guides: {deck} vs {opp}"
        if deck not in g.decks or opp not in g.decks or deck == opp:
            fails.append(f"{where}: not a pair of league decks")
            continue
        doc = doc_rows.get((deck, opp))
        if doc is not None and (keyed(r["in"]), keyed(r["out"]), r["draw"] and key(r["draw"])) != \
                (keyed(doc[0]), keyed(doc[1]), doc[2] and key(doc[2])):
            fails.append(f"{where}: the JSON row differs from the doc's")
        if sum(r["in"].values()) != sum(r["out"].values()):
            fails.append(f"{where}: {sum(r['in'].values())} in, {sum(r['out'].values())} out")
        main, side = g.decks[deck]
        land = g.draw_land[deck]
        if (r["draw"] is None) != (land is None):
            fails.append(f"{where}: Draw +1 {r['draw']!r} with draw land {land!r}")
        need_in = dict(r["in"])
        if r["draw"] is not None:
            need_in[r["draw"]] = need_in.get(r["draw"], 0) + 1
        for name, n in need_in.items():
            if side.get(name, 0) < n:
                fails.append(f"{where}: {n} {name} in (with the draw), {side.get(name, 0)} in its sideboard")
        need_out = dict(r["out"])
        if land is not None:
            need_out[land] = need_out.get(land, 0) + 1
        for name, n in need_out.items():
            if main.get(name, 0) < n:
                fails.append(f"{where}: {n} {name} out (with the draw land), {main.get(name, 0)} in its main deck")
        if set(need_in) & set(need_out):
            fails.append(f"{where}: {sorted(set(need_in) & set(need_out))} both in and out")
    for deck, opp in sorted(g.holdout):
        if (deck, opp) not in json_rows:
            fails.append(f"guides: held-out {deck} vs {opp} has no guide row")
    held = sorted(g.holdout)
    if len(held) != len(g.decks) or {d for d, _ in held} != set(g.decks) or {o for _, o in held} != set(g.decks):
        fails.append(f"guides: the holdout {held} does not hold each deck out once as each side")
    return fails


class _FakeCards:
    """The view's ``cards`` for the simulated phases: names are the deck files'."""

    @staticmethod
    def canonical(name):
        return name


class _PhaseView:
    """A sideboarding decision of ``_simulate_phase`` as the tier's view."""

    cards = _FakeCards()

    def __init__(self, rows, main, own, opp, plays_first):
        self.cats = [c for c, _ in rows]
        self._cards = [n for _, n in rows]
        self.n = len(rows)
        self.own_current_main = {k: v for k, v in main.items() if v}
        self._lists = {True: own, False: opp}
        self.plays_first = plays_first

    def card(self, i):
        return self._cards[i]

    def registered_list(self, mine):
        return self._lists[mine]


def _simulate_phase(main, side, own, opp, plays_first, choose):
    """One sideboarding phase as the engine runs it (``run_sideboard_phase``: in-first, the one-shot
    locks, Done only while balanced, the swap cap) with ``choose(view)`` deciding; returns (main,
    side, swaps)."""
    main, side = dict(main), dict(side)
    locked_in, locked_out = set(), set()
    swaps, outstanding = 0, False
    for _ in range(200):
        if not outstanding:
            if swaps >= 15:
                break
            rows = [("SIDEBOARD_DONE", None)] + [("SIDEBOARD_IN", n) for n, c in side.items()
                                                 if c and n not in locked_out]
        else:
            rows = [("SIDEBOARD_OUT", n) for n, c in main.items() if c and n not in locked_in]
        pick = choose(_PhaseView(rows, main, own, opp, plays_first)) if len(rows) > 1 else 0
        cat, name = rows[pick]
        if cat == "SIDEBOARD_DONE":
            return main, side, swaps
        src, dst = (side, main) if cat == "SIDEBOARD_IN" else (main, side)
        src[name] -= 1
        dst[name] = dst.get(name, 0) + 1
        (locked_in if cat == "SIDEBOARD_IN" else locked_out).add(name)
        swaps += outstanding
        outstanding = not outstanding
    return main, side, swaps


def _nonzero(deck):
    return {k: v for k, v in deck.items() if v}


def _guides_unit_fails(g):
    """The tier's target and its phases over a simulated engine menu; returns failures."""
    fails = []
    choose = sideboard_guides.choose

    def phase(own, opp, plays_first, start=None, opp_lists=None):
        reg = g.decks[own]
        main, side = start if start is not None else reg
        return _simulate_phase(main, side, reg, opp_lists or g.decks[opp], plays_first, choose)

    def expect(label, got, want):
        if _nonzero(got) != _nonzero(want):
            diff = {k: (got.get(k, 0), want.get(k, 0)) for k in set(got) | set(want) if got.get(k, 0) != want.get(k, 0)}
            fails.append(f"guides unit: {label}: main differs from the target (got, want): {diff}")

    play = g.target("burn", "bw_dnt", True)
    draw = g.target("burn", "bw_dnt", False)
    reg = g.decks["burn"][0]
    if play is None or play.get("Smash to Smithereens") != 2 or play.get("Eidolon of the Great Revel", 0) != 0 \
            or play.get("Mountain") != reg["Mountain"]:
        fails.append(f"guides unit: burn vs bw_dnt on the play: target {play}")
    if draw is None or draw.get("Smash to Smithereens") != 3 or draw.get("Mountain") != reg["Mountain"] - 1:
        fails.append(f"guides unit: burn vs bw_dnt on the draw: target {draw}")
    m, s, n = phase("burn", "bw_dnt", True)
    expect("burn vs bw_dnt on the play", m, play)
    if n != 7:
        fails.append(f"guides unit: burn vs bw_dnt on the play: {n} swaps, want 7")
    m2, s2, n = phase("burn", "bw_dnt", False)
    expect("burn vs bw_dnt on the draw", m2, draw)
    # Game 3 after a flip from the draw to the play: the land swap is undone, nothing else moves.
    m3, _s3, n = phase("burn", "bw_dnt", True, start=(m2, s2))
    expect("burn vs bw_dnt, game 3 from the draw to the play", m3, play)
    if n != 1:
        fails.append(f"guides unit: burn vs bw_dnt, game 3 from the draw to the play: {n} swaps, want 1")
    # And back: the play configuration onto the draw.
    m4, _s4, n = phase("burn", "bw_dnt", False, start=(m, s))
    expect("burn vs bw_dnt, game 3 from the play to the draw", m4, draw)
    if n != 1:
        fails.append(f"guides unit: burn vs bw_dnt, game 3 from the play to the draw: {n} swaps, want 1")
    # A deck with no draw land boards the same on the play and on the draw.
    mp, _, _ = phase("gw_maverick", "bug", True)
    md, _, _ = phase("gw_maverick", "bug", False)
    expect("gw_maverick vs bug on the play", mp, g.target("gw_maverick", "bug", True))
    if _nonzero(mp) != _nonzero(md) or g.target("gw_maverick", "bug", True) != g.target("gw_maverick", "bug", False):
        fails.append("guides unit: gw_maverick vs bug boards differently on the play and on the draw")
    # A held-out pair, and an opponent that is no league deck: Done at once.
    if g.target("bug", "reanimator", False) is not None or g.target("bug", "reanimator", False, holdout=False) is None:
        fails.append("guides unit: bug vs reanimator is not held out (or has no row)")
    for label, opp_lists in (("bug vs reanimator (held out)", None),
                             ("bug vs an unknown list", ({"Island": 60}, {}))):
        mh, _, n = phase("bug", "reanimator", False, opp_lists=opp_lists)
        if n != 0 or _nonzero(mh) != _nonzero(g.decks["bug"][0]):
            fails.append(f"guides unit: {label}: {n} swaps, want Done at once")
    # Every row: a phase from the registered list on the play and on the draw reaches its target.
    for own, opps in g.guides.items():
        for opp in opps:
            for first in (True, False):
                want = g.target(own, opp, first)
                got, _, _ = phase(own, opp, first)
                expect(f"{own} vs {opp} {'on the play' if first else 'on the draw'}", got,
                       want if want is not None else g.decks[own][0])
    return fails


def check_guides():
    """The guides part: the JSON against the doc and the deck files, and the tier's unit checks;
    returns failures."""
    g = sideboard_guides.guides()
    with open(GUIDES_DOC, encoding="utf-8") as f:
        fails = _guides_data_fails(g, f.read())
    fails += _guides_unit_fails(g)
    if not fails:
        print(f"  OK   guides: {sum(len(o) for o in g.guides.values())} guide rows agree with the doc and the "
              f"deck files, {len(g.holdout)} held out; the tier's phases reach their targets", flush=True)
    return fails


# The sideboard part's bo3 matches, scripted:hard+sb on both seats: (deck_a, deck_b, seed). Burn has
# a draw land, gw_maverick none; bug vs reanimator is held out for bug (not for reanimator). The seeds
# are matches that go to three games (each seat's play/draw flips between games 2 and 3); the part
# fails when none of them does any more (repin the seeds).
SIDEBOARD_MATCHES = (("league/burn", "league/gw_maverick", 4),
                     ("league/bug", "league/reanimator", 3),
                     ("league/uw_control", "league/wubg_doomsday", 9))
_SB_HEADER = re.compile(r"^Player ([AB]) sideboarding \(\d+ in sideboard, (\d+) swap\(s\) made\):")
_GAME_RESULT = re.compile(r"^GAME_RESULT: (\d+) (?:Player ([AB]) wins|draw)")


class _DeckRecorder:
    """A scripted agent that also records, at its seat's first decision of each game, the maindeck
    and who plays first (the frames' ``own_deck`` and ``match``)."""

    def __init__(self, spec, seen):
        self.inner = rm_driver.make_agent(spec)
        self.seen = seen

    def new_game(self):
        self.inner.new_game()

    def decide(self, seat, frame):
        header = frame.tables["header"][0]
        if not header["sideboarding"] and (seat.seat, header["game"]) not in self.seen:
            v = seat.view(frame)
            self.seen[(seat.seat, header["game"])] = (v.own_current_main, v.plays_first, seat.cards.canonical)
        return self.inner.decide(seat, frame)


def check_sideboard(binary, work):
    """The sideboard part: SIDEBOARD_MATCHES played by rm_driver.py; each seat's maindeck at games 2
    and 3 is its guide's target for the matchup and play/draw (the registered list where held out),
    the play/draw the loser-goes-first rule's and each phase's swap count the difference to the
    previous game's deck; returns failures."""
    g = sideboard_guides.guides()
    fails, phases, game3 = [], 0, 0
    cwd = os.getcwd()
    os.chdir(work)
    try:
        for deck_a, deck_b, seed in SIDEBOARD_MATCHES:
            label = f"sideboard: {deck_a} vs {deck_b} seed {seed}"
            seen, narrative = {}, []
            rec = rm_driver.play_game(_DeckRecorder("scripted:hard+sb", seen), _DeckRecorder("scripted:hard+sb", seen),
                                      binary=binary, deck_a=deck_a, deck_b=deck_b, bo3=True, seed=seed,
                                      on_narrative=narrative.extend)
            winners, swaps = [], [{}]
            for ln in narrative:
                m = _GAME_RESULT.match(ln.strip())
                if m:
                    winners.append(m.group(2))
                    swaps.append({})
                m = _SB_HEADER.match(ln.strip())
                if m:
                    swaps[-1][m.group(1)] = int(m.group(2))
            if len(winners) < 2:
                fails.append(f"{label}: {len(winners)} game(s) in a bo3 match")
            game3 += len(winners) >= 3
            decks = {"A": (deck_a.split("/")[-1], deck_b.split("/")[-1]),
                     "B": (deck_b.split("/")[-1], deck_a.split("/")[-1])}
            a_first = True
            for game in range(1, len(winners)):
                if winners[game - 1] is not None:
                    a_first = winners[game - 1] == "B"
                for seat in "AB":
                    got = seen.get((seat, game))
                    if got is None:
                        fails.append(f"{label}: no decision of Player {seat} in game {game + 1}")
                        continue
                    main, plays_first, canonical = got
                    own, opp = decks[seat]
                    on_play = a_first == (seat == "A")
                    if plays_first != on_play:
                        fails.append(f"{label}: game {game + 1}: Player {seat}'s frames say "
                                     f"{'play' if plays_first else 'draw'}, the results "
                                     f"{'play' if on_play else 'draw'}")
                    want = g.target(own, opp, on_play)
                    want = {canonical(n): c for n, c in (want if want is not None else g.decks[own][0]).items()}
                    if _nonzero(main) != _nonzero(want):
                        diff = {k: (main.get(k, 0), want.get(k, 0)) for k in set(main) | set(want)
                                if main.get(k, 0) != want.get(k, 0)}
                        fails.append(f"{label}: game {game + 1}: Player {seat}'s main differs from the target "
                                     f"(got, want): {diff}")
                    prev = seen.get((seat, game - 1), ({},))[0]
                    moved = sum(max(0, c - prev.get(k, 0)) for k, c in main.items())
                    if swaps[game].get(seat, -1) != moved:
                        fails.append(f"{label}: game {game + 1}: Player {seat} made {swaps[game].get(seat)} swap(s), "
                                     f"its main moved {moved} card(s)")
                    phases += 1
            if rec.drawn_games:
                fails.append(f"{label}: {rec.drawn_games} drawn game(s)")
    finally:
        os.chdir(cwd)
    if not game3:
        fails.append("sideboard: no match went to three games (repin SIDEBOARD_MATCHES' seeds)")
    if not fails:
        print(f"  OK   sideboard: {len(SIDEBOARD_MATCHES)} bo3 matches of scripted:hard+sb ({game3} to three games), "
              f"{phases} sideboarding phases each at the guide's target", flush=True)
    return fails


# ── RMTB ──────────────────────────────────────────────────────────────────────────────────────

def _bits(dtype, x):
    """A float's bytes at its stored width, so -0.0 and the rounding are compared exactly; a value
    past the half range is its infinity (IEEE rounding; ``struct`` raises instead)."""
    try:
        return struct.pack("<e" if dtype == "f16" else "<f", float(x))
    except OverflowError:
        return struct.pack("<e" if dtype == "f16" else "<f", math.copysign(math.inf, float(x)))


def _sample_fails(t):
    """One sample file of robomage_check's against what python/rmtb.py reads from it."""
    name = os.path.basename(t["path"])
    try:
        meta, arrays = rmtb.read(t["path"])
    except (OSError, ValueError) as e:
        return [f"rmtb: {name}: {e}"]
    fails = []
    if meta != t["meta"]:
        fails.append(f"rmtb: {name}: meta {meta!r}, written {t['meta']!r}")
    for aname, want in t["arrays"].items():
        a = arrays.get(aname)
        if a is None:
            fails.append(f"rmtb: {name}: no array {aname}")
            continue
        fl = a.dtype in ("f32", "f16")
        same = len(a.values) == len(want["values"]) and all(
            (_bits(a.dtype, x) == _bits(a.dtype, y)) if fl else x == y for x, y in zip(a.values, want["values"]))
        if not same:
            fails.append(f"rmtb: {name}: {aname} reads {a.values!r}, written {want['values']!r}")
        # struct's 'e' rounds to nearest even like the C++ writer: the stored halves are its packing
        # of the floats they were written from.
        source = [struct.unpack("<f", _bits("f32", x))[0] for x in want.get("source", ())]  # the exact floats
        if "source" in want and [_bits("f16", x) for x in source] != [_bits("f16", x) for x in a.values]:
            fails.append(f"rmtb: {name}: {aname}'s halves differ from struct's rounding of {want['source']!r}")
    return fails


def check_rmtb(samples):
    """The rmtb part; returns failures."""
    fails = []
    checked = []
    if samples:
        with open(samples, encoding="utf-8") as f:
            expected = json.load(f)["files"]
        for t in expected:
            fails += _sample_fails(t)
            checked.append(os.path.basename(t["path"]))
    if os.path.isfile(TEXT_TABLE):
        try:
            meta, arrays = rmtb.read(TEXT_TABLE)
            present = {r["id"] for r in meta["rows_meta"]}
            rows = arrays["vectors"].rows()
            if meta["kind"] != "text" or any(any(rows[i]) for i in range(len(rows)) if i not in present):
                fails.append("rmtb: the text table is not a text table with zero rows outside rows_meta")
            checked.append(os.path.basename(TEXT_TABLE))
        except (OSError, ValueError, KeyError) as e:
            fails.append(f"rmtb: {TEXT_TABLE}: {e}")
    if not fails:
        print(f"  OK   rmtb: {', '.join(checked) or 'nothing to read (no --rmtb-samples, no text table)'}",
              flush=True)
    return fails


# ── The featurizer's files ────────────────────────────────────────────────────────────────────

_PACK = {"f32": "f", "f16": "e", "i64": "q", "i32": "i", "i16": "h", "u16": "H", "u8": "B"}


def _fnv1a64(data):
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def _table_fails(kind, arrays, want, spec):
    """An exported table's shapes against the spec's card_tables entry."""
    feats = arrays["features"]
    fails = []
    if feats.shape != [want["rows"], want["width"]] or feats.columns != want["columns"]:
        fails.append(f"featurize: the {kind} table's features are {feats.shape}, the spec says "
                     f"[{want['rows']}, {want['width']}] with its columns")
    if kind == "card" and arrays["back_face"].shape != [want["rows"]]:
        fails.append("featurize: the card table's back_face is not one entry per card row")
    if kind == "ability":
        keys = arrays["key_rows"]
        if keys.columns != spec["ab_keys"] or keys.values != spec["ab_key_rows"] or \
                any(not want["key_base"] <= r < want["rows"] for r in keys.values):
            fails.append("featurize: the ability table's key_rows are not the spec's keys within its rows")
    if kind == "part":
        rows = spec["card_tables"]["ability"]["rows"]
        if any(not 0 <= r < rows for r in arrays["ability"].values):
            fails.append("featurize: a part row names no ability row")
    return fails


def _tensor_fails(arrays, spec):
    """A tensors file's per-decision shapes and indices against the spec."""
    widths = {"tok_num": spec["F"], "tok_cat": spec["C"], "tok_card": spec["K"], "tok_ability": spec["Q"], "rel": 3}
    fails = []
    for name, w in widths.items():
        if arrays[name].shape[1:] != [w]:
            fails.append(f"featurize: {name} has shape {arrays[name].shape}, width {w} in the spec")
    if fails:
        return fails
    type_ids = {t["id"] for t in spec["token_types"]}
    rel_ids = {r["id"] for r in spec["relation_types"]}
    menu = next(t["id"] for t in spec["token_types"] if t["name"] == spec["menu"])
    cards, abilities = spec["card_tables"]["card"]["rows"], spec["card_tables"]["ability"]["rows"]
    tok_off, rel_off, act_off = (arrays[n + "_offsets"].values for n in ("tok_type", "rel", "act_tok"))
    for name in ("tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask"):
        if arrays[name + "_offsets"].values != tok_off:
            fails.append(f"featurize: {name}_offsets differ from tok_type_offsets")
    tt, rel, act = arrays["tok_type"].values, arrays["rel"].values, arrays["act_tok"].values
    if any(t not in type_ids for t in tt):
        fails.append("featurize: a tok_type is no token type")
    if any(not -1 <= c < cards for c in arrays["tok_card"].values):
        fails.append("featurize: a tok_card is no card row")
    if any(not -1 <= a < abilities for a in arrays["tok_ability"].values):
        fails.append("featurize: a tok_ability is no ability row")
    if any(not 0 <= c < spec["categorical_size"] for c in arrays["tok_cat"].values):
        fails.append("featurize: a tok_cat is outside the categorical id space")
    for d in range(len(tok_off) - 1):
        n = tok_off[d + 1] - tok_off[d]
        for r in range(rel_off[d], rel_off[d + 1]):
            src, dst, ty = rel[3 * r:3 * r + 3]
            if not (0 <= src < n and 0 <= dst < n and ty in rel_ids):
                fails.append(f"featurize: decision {d}: a relation outside its tokens")
                break
        for a in act[act_off[d]:act_off[d + 1]]:
            if not 0 <= a < n or tt[tok_off[d] + a] != menu:
                fails.append(f"featurize: decision {d}: an act_tok that is no menu token")
                break
    return fails[:10]


def check_featurize(expected_path):
    """The featurize part; returns failures."""
    if not expected_path:
        print("  OK   featurize: nothing to read (no --featurize-expected)", flush=True)
        return []
    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)
    with open(expected["spec"], "rb") as f:
        spec_bytes = f.read().rstrip(b"\n")
    spec = json.loads(spec_bytes.decode("utf-8"))
    spec_hash = "%016x" % _fnv1a64(spec_bytes)
    fails = []
    for want in expected["files"]:
        name = os.path.basename(want["path"])
        try:
            meta, arrays = rmtb.read(want["path"])
        except (OSError, ValueError) as e:
            fails.append(f"featurize: {name}: {e}")
            continue
        if meta.get("kind") != want["kind"] or meta.get("spec_hash") != spec_hash or \
                meta.get("vocab_hash") != spec["vocab_hash"]:
            fails.append(f"featurize: {name}: kind, spec_hash or vocab_hash is not {want['kind']}, {spec_hash}, "
                         f"{spec['vocab_hash']}")
        for aname, a_want in want["arrays"].items():
            a = arrays.get(aname)
            if a is None:
                fails.append(f"featurize: {name}: no array {aname}")
                continue
            packed = struct.pack("<%d%s" % (len(a.values), _PACK[a.dtype]), *a.values)
            if a.shape != a_want["shape"] or "%016x" % _fnv1a64(packed) != a_want["fnv"]:
                fails.append(f"featurize: {name}: {aname} does not read as the values written")
        kind = want["kind"]
        if kind in ("card", "ability", "part"):
            fails += _table_fails(kind, arrays, spec["card_tables"][kind], spec)
        elif kind == "tensors":
            fails += _tensor_fails(arrays, spec)
        elif kind == "targets":
            depth = spec["library_depth"]
            for lib in ("own_library", "opp_library"):
                off = arrays[lib + "_offsets"].values
                if any(off[d + 1] - off[d] > depth for d in range(len(off) - 1)):
                    fails.append(f"featurize: {name}: {lib} deeper than {depth}")
    if not fails:
        print(f"  OK   featurize: {len(expected['files'])} files read as written, spec {spec_hash}: shapes and "
              "indices agree", flush=True)
    return fails


# ── Trajectory episodes ───────────────────────────────────────────────────────────────────────

def _f32(x):
    """``x`` rounded to the nearest float32 (the stored μ, values and log π)."""
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def _decision_slices(arrays, names, d):
    """Decision ``d``'s values of each ragged array of a tensors or targets file."""
    out = {}
    for name in names:
        a, off = arrays[name], arrays[name + "_offsets"].values
        width = a.shape[1] if len(a.shape) == 2 else 1
        out[name] = a.values[off[d] * width:off[d + 1] * width]
    return out


def _episode_fails(want):
    name = os.path.basename(want["path"])
    try:
        ep = rmtraj.read(want["path"])
    except (OSError, ValueError, KeyError) as e:
        return [f"traj: {name}: {e}"]
    fails = []
    if len(ep.steps) != len(want["steps"]):
        return [f"traj: {name}: {len(ep.steps)} steps, the C++ reader {len(want['steps'])}"]
    _, tensors = rmtb.read(want["tensors"])
    omni = rmtb.read(want["omniscient"])[1] if want.get("omniscient") else None
    targets = rmtb.read(want["targets"])[1] if want.get("targets") else None
    if (omni is None) != (ep.steps and ep.steps[0].omniscient is None):
        fails.append(f"traj: {name}: the omniscient part is {'present' if omni is None else 'absent'}")
    for d, (s, w) in enumerate(zip(ep.steps, want["steps"])):
        mu = None if w["mu"] is None else [_f32(x) for x in w["mu"]]
        policies = [(p[0], p[1], _f32(p[2]), [_f32(x) for x in p[3]]) for p in w["policies"]]
        got = (s.seat, s.game, s.decision, s.sideboarding, s.chosen, s.policy_version, s.teacher, s.mu,
               [(p[0], p[1], p[2], list(p[3])) for p in s.policies])
        exp = (w["seat"], w["game"], w["decision"], w["sideboarding"], w["chosen"], w["policy_version"],
               w["teacher"], mu, policies)
        if got != exp:
            fails.append(f"traj: {name}: step {d}: {got[:7]} differs from the C++ reader's {exp[:7]} (or its mu, "
                         "evaluations)")
        if _decision_slices(tensors, rmtraj.TENSORS, d) != {k: s.tensors[k] for k in rmtraj.TENSORS}:
            fails.append(f"traj: {name}: step {d}: the tensors differ from the C++ reader's")
        if omni is not None:
            if _decision_slices(omni, rmtraj.TENSORS, d) != {k: s.omniscient[k] for k in rmtraj.TENSORS}:
                fails.append(f"traj: {name}: step {d}: the omniscient tensors differ from the C++ reader's")
            tg = _decision_slices(targets, ("opp_hand", "own_library", "opp_library"), d)
            if tg != s.targets:
                fails.append(f"traj: {name}: step {d}: the targets differ from the C++ reader's")
        if len(fails) > 10:
            break
    return fails


def check_traj(expected_path):
    """The traj part; returns failures."""
    if not expected_path:
        print("  OK   traj: nothing to read (no --traj-expected)", flush=True)
        return []
    with open(expected_path, encoding="utf-8") as f:
        expected = json.load(f)["episodes"]
    fails = []
    for want in expected:
        fails += _episode_fails(want)
    if not fails:
        steps = sum(len(w["steps"]) for w in expected)
        print(f"  OK   traj: {len(expected)} episodes, {steps} steps read as the C++ reader reads them", flush=True)
    return fails


# ── The open-vocabulary acceptance's inputs ─────────────────────────────────────────────────

def _rmtb_raw(path):
    """(meta, {name: (dtype, shape, the array's bytes)}) of an RMTB file, nothing decoded."""
    with open(path, "rb") as fp:
        data = fp.read()
    _, _, meta_len = rmtb._HEADER.unpack_from(data, 0)
    meta = json.loads(data[rmtb._HEADER.size:rmtb._HEADER.size + meta_len].decode("utf-8"))
    body = memoryview(data)[rmtb._HEADER.size + meta_len:]
    out = {}
    for a in meta.get("arrays", ()):
        n = 1
        for d in a["shape"]:
            n *= d
        width = rmtb.DTYPES[a["dtype"]][1]
        out[a["name"]] = (a["dtype"], a["shape"], body[a["offset"]:a["offset"] + n * width])
    return meta, out


def _ints(raw):
    dtype, shape, b = raw
    code, width = rmtb.DTYPES[dtype]
    return list(struct.unpack("<%d%s" % (len(b) // width, code), b))


def _decision_digests(path):
    """{(viewer, game, sideboarding, decision): sha256 hex[:16]} of a tensors file's decisions."""
    _, arr = _rmtb_raw(path)
    offs = {k: _ints(arr[k + "_offsets"]) for k in ("tok_type", "rel", "act_tok")}
    fields = {k: _ints(arr[k]) for k in ("viewer", "game", "sideboarding", "decision")}
    out = {}
    for d in range(len(fields["game"])):
        h = hashlib.sha256()
        for name, count in OPEN_VOCAB_TENSORS:
            dtype, shape, b = arr[name]
            row = rmtb.DTYPES[dtype][1]
            for x in shape[1:]:
                row *= x
            h.update(b[offs[count][d] * row:offs[count][d + 1] * row])
        out[(fields["viewer"][d], fields["game"][d], fields["sideboarding"][d], fields["decision"][d])] = \
            h.hexdigest()[:16]
    return out


def check_open_vocab(binary, work):
    """The open_vocab part; returns failures."""
    path = os.path.join(OPEN_VOCAB, "inputs.json")
    if not os.path.exists(path):
        return [f"open_vocab: no {path}"]
    with open(path, encoding="utf-8") as f:
        inputs = json.load(f)
    with open(os.path.join(OPEN_VOCAB, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    _, index = _rmtb_raw(os.path.join(OPEN_VOCAB, "baseline", "decisions.rmtb"))
    idx = {k: _ints(index[k]) for k in ("source", "viewer", "game", "sideboarding", "decision")}
    bin_dir = os.path.join(os.path.dirname(_HERE), "bin")
    rmfeat = os.path.join(os.path.dirname(binary), "rmfeat")
    cards, tables = os.path.join(work, "cards.rmstate"), os.path.join(work, "tables")
    os.makedirs(tables, exist_ok=True)
    fails = []

    def run_(argv):
        r = subprocess.run(argv, cwd=bin_dir, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                           timeout=_TIMEOUT_S)
        if r.returncode != 0:
            fails.append(f"open_vocab: {' '.join(argv)} exited {r.returncode}: {r.stderr[-300:]}")
        return r.returncode == 0

    if not (run_([binary, "--cards-frame", cards]) and run_([rmfeat, "--export-tables", cards, tables])):
        return fails
    sources = {s["name"]: s for s in man["sources"]}
    checked = moved = 0
    for name in inputs["gate_sources"]:
        src = sources[name]
        dump = os.path.join(work, name + ".rms")
        if not run_([binary, "--replay", os.path.join(OPEN_VOCAB, src["log"]), "--machine", "--rmstate-dump", dump]):
            continue
        dig = {}
        for viewer in ("A", "B", "omniscient"):
            out = os.path.join(work, f"{name}.{viewer}.rmtb")
            if not run_([rmfeat, "--featurize", dump, "--viewer", viewer, "--out", out, "--cards", cards,
                         "--tables", tables]):
                break
            dig[viewer] = _decision_digests(out)
            os.unlink(out)
        os.unlink(dump)
        if len(dig) != 3:
            continue
        seat = {**dig["A"], **dig["B"]}
        omni = {k[1:]: v for k, v in dig["omniscient"].items()}
        for r in range(len(idx["source"])):
            want = inputs["decisions"].get(str(r))
            if idx["source"][r] != src["index"] or want is None:
                continue
            key = (idx["viewer"][r], idx["game"][r], idx["sideboarding"][r], idx["decision"][r])
            got = [seat.get(key), omni.get(key[1:])]
            checked += 1
            if got != want:
                moved += 1
                if moved <= 5:
                    fails.append(f"open_vocab: {name}: fixture decision {r} {key}: inputs {got}, recorded {want}")
    if moved > 5:
        fails.append(f"open_vocab: {moved - 5} more fixture decisions' inputs moved")
    if moved:
        fails.append("open_vocab: the featurization of the open-vocabulary fixtures moved: the migration's acceptance "
                     "holds on the recorded inputs only (docs/open_vocab_plan.md S5); if intended, drop the moved "
                     "decisions: learn/open_vocab_acceptance.py inputs --drop-moved")
    if not fails:
        print(f"  OK   open_vocab: {checked} fixture decisions of {len(inputs['gate_sources'])} sources featurize to "
              "their recorded inputs", flush=True)
    return fails


def run(binary, out_dir=None, parts=PARTS, rmtb_samples=None, featurize_expected=None, traj_expected=None):
    """The requested parts; returns the failure messages (empty = pass)."""
    binary = os.path.abspath(binary)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    work = tempfile.mkdtemp(prefix="pycheck_", dir=out_dir)
    fails = []
    try:
        first_dump = None
        if "decode" in parts or "refusals" in parts:
            f, first_dump = check_decode(binary, work) if "decode" in parts else ([], None)
            fails += f
            if "refusals" in parts:
                if first_dump is None:
                    path, f = _engine_dump(binary, work, *ENGINE_DUMPS[0])
                    fails += f
                    first_dump = path
                if first_dump:
                    fails += check_refusals(binary, first_dump, work)
        if "selfkill" in parts:
            fails += check_selfkill(binary, work)
        if "driver" in parts:
            fails += check_driver(binary, work)
        if "guides" in parts:
            fails += check_guides()
        if "sideboard" in parts:
            fails += check_sideboard(binary, work)
        if "rmtb" in parts:
            fails += check_rmtb(rmtb_samples)
        if "featurize" in parts:
            fails += check_featurize(featurize_expected)
        if "traj" in parts:
            fails += check_traj(traj_expected)
        if "open_vocab" in parts:
            fails += check_open_vocab(binary, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return fails


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="pycheck", description="The kept Python's checks (robomage_check's "
                                 "smoke tier runs them).")
    ap.add_argument("--binary", default=rm_driver.DEFAULT_BINARY, help="the engine (default %(default)s)")
    ap.add_argument("--out-dir", default=None, help="the scratch directory's parent (default the system's)")
    ap.add_argument("--part", default=",".join(PARTS), help=f"comma list of {', '.join(PARTS + OPTIONAL_PARTS)} "
                                                           f"(default all but {', '.join(OPTIONAL_PARTS)})")
    ap.add_argument("--rmtb-samples", default=None, help="the rmtb part's samples (robomage_check's smoke tier "
                    "writes them)")
    ap.add_argument("--featurize-expected", default=None, help="the featurize part's file list (robomage_check's "
                    "featurize tier writes it)")
    ap.add_argument("--traj-expected", default=None, help="the traj part's episode list (robomage_check's "
                    "traj tier writes it)")
    args = ap.parse_args(argv)
    parts = [p for p in args.part.split(",") if p]
    unknown = [p for p in parts if p not in PARTS + OPTIONAL_PARTS]
    if unknown:
        ap.error(f"unknown part(s) {', '.join(unknown)}")
    fails = run(args.binary, args.out_dir, parts, args.rmtb_samples, args.featurize_expected, args.traj_expected)
    for f in fails:
        print(f"FAIL: {f}", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
