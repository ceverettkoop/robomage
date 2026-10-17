#!/usr/bin/env python3
"""The ``agentparity`` gate tier: the scripted agent's vector view on RMSTATE plays its BQUERY games.

Plan step 2, pass 1 (``docs/engine_fork_plan.md``): the scripted agent reads the old observation
rebuilt from RMSTATE frames by the parity adapter (``train/rmstate_adapter.py``, driven by
``train/rmstate_env.py`` under ``runner.run_games(machine_format="rmstate")``), with its card
knowledge and deck names from the ``CARDS`` / ``DECKLISTS`` frames and its two memory reads from the
knowledge sidecar's ``k_*`` tables. The exit criterion is byte-identical games. A game's transcript
is compared through ``project``: the RMSTATE run's verbose transcript has no board block (the
adapter rebuilds only what the agent reads), so the block between a ``--- Decision N ---`` header
and its ``Actions:`` line is left out of both, and the ``PA:`` / ``PB:`` hand lines are compared as
sorted lists (hand slots follow entity order, which the format does not carry). Everything else
— the engine narrative, every menu as offered, every choice — must match byte for byte.

Parts:

  1. cards   the card knowledge: ``rmstate_adapter.FrameCardKnowledge`` built from a
             ``--closed-decklist`` CARDS frame (every vocab card) answers every card read of
             ``scripted_agent.VocabCardKnowledge`` (the generated tables and scripts) identically:
             the cast-cost row, the land flag, all property columns, the face's script lines and
             type line.
  2. corpus  replay-corpus scenarios (``regression/replay_diff.py``: scripted hard on both seats,
             its frozen seed; ``CORPUS_GAMES``; ``--full``: all of them) on both protocols: a game
             of the light set is played on BQUERY and must equal its frozen transcript
             (``regression/agent_parity/bquery_corpus/``: the replay corpus itself is the agent's
             frame-view games, so the vector view keeps these pinned until plan step 3b), and its
             RMSTATE projection must equal the BQUERY game's.
  3. smoke   scripted-hard league games (``SMOKE_GAMES``: smoke-tier games and two bo3 matches, and
             ``DIVERGENT_GAMES``; ``--full``: every smoke-tier game, a bo3 match per mirror and every
             listed divergence) played on both protocols; the projections must be equal.

A game whose old vector leaked or had a quirk that changed what the agent saw cannot be identical.
Such a game is listed in ``regression/agent_parity_divergences.json`` (``game``, the first decision
that differs, the seat, which read differed, the classification: a signature of
``regression/knowledge_divergences.json``, and why) and re-recorded: its RMSTATE projection is
committed under ``regression/agent_parity/`` and the check compares against that, and also checks
the BQUERY game still first differs at the listed decision. ``--record`` rewrites those baselines;
``--lockstep GAME…`` is the diagnostic that finds and attributes a divergence: the BQUERY game is
played with ``--rmstate-dump`` and ``--selfcheck-knowledge``, every decision's adapter observation is
compared with the real vector block by block, a second agent decides on the adapter observation,
and the first differing choice is reported with the differing blocks and the ``SHADOW:`` lines
(classified) of the deciding seat at that decision.

  train/.venv/bin/python train/test_agent_parity.py                 # the tier (light)
  train/.venv/bin/python train/test_agent_parity.py --full          # every corpus and smoke game
  train/.venv/bin/python train/test_agent_parity.py --lockstep smoke/burn__bw_dnt/1001
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import io
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "regression"))

from cli_spec import BINARY  # noqa: E402

PARTS = ("cards", "corpus", "smoke")
DIVERGENCES_PATH = os.path.join(_HERE, "regression", "agent_parity_divergences.json")
BASELINE_DIR = os.path.join(_HERE, "regression", "agent_parity")
CORPUS_DIR = os.path.join(_HERE, "regression", "corpus")
# The BQUERY transcripts of the light corpus set, frozen: the replay corpus itself is the scripted
# agent's games on its RMSTATE frame view (replay_diff.py), so the agent's vector view keeps its
# own pinned games here until the BQUERY path goes (plan step 3b).
BQUERY_CORPUS_DIR = os.path.join(_HERE, "regression", "agent_parity", "bquery_corpus")
LEAGUE = ("bug", "burn", "bw_dnt", "gw_maverick", "lands", "reanimator", "ur_delver", "uw_control",
          "wrb_energy", "wubg_doomsday")
# The light corpus set: a curated scenario per legacy deck, and league games covering tokens and
# haste, an impending permanent and a modal double-faced land, Doomsday piles, Cabal Therapy,
# reanimation, Sylvan Library and the lands deck's Mox Diamond.
CORPUS_GAMES = ("delver_vs_doomsday", "mav_vs_delver", "lg_bug__burn", "lg_bw_dnt__bw_dnt",
                "lg_gw_maverick__ur_delver", "lg_lands__reanimator", "lg_uw_control__wubg_doomsday",
                "lg_wrb_energy__wubg_doomsday")
# The light smoke set: smoke-tier games (seed 1 + 1000 k + i as the smoke tier draws them) and two
# bo3 matches (sideboarding frames).
SMOKE_GAMES = ("smoke/burn__bw_dnt/11001", "smoke/reanimator__ur_delver/15001",
               "smoke/wrb_energy__wubg_doomsday/18001", "match/bug__bug/1", "match/lands__reanimator/4007")
# Listed divergences the light set re-checks against their baselines: an old-vector leak and a quirk.
DIVERGENT_GAMES = ("match/wrb_energy__uw_control/710", "match/bw_dnt__wrb_energy/298")


# ── games ─────────────────────────────────────────────────────────────────────────────────────

def smoke_tier_games():
    """Every game of the smoke tier (ci_check: mirrors + ring, 2 games per matchup, game i of
    matchup k seeded 1 + 1000 k + i) as game keys."""
    pairs = [(d, d) for d in LEAGUE] + [(LEAGUE[i], LEAGUE[(i + 1) % len(LEAGUE)]) for i in range(len(LEAGUE))]
    return [f"smoke/{a}__{b}/{1 + 1000 * k + i}" for k, (a, b) in enumerate(pairs) for i in range(2)]


def full_smoke_games():
    return smoke_tier_games() + [f"match/{d}__{d}/{1 + 1000 * k}" for k, d in enumerate(LEAGUE)]


def corpus_scenarios():
    import replay_diff
    return {name: (a, b) for name, a, b in replay_diff.SCENARIOS}


def game_setup(key):
    """(deck_a, deck_b, seed, bo3) of a game key: ``corpus/<scenario>``, ``smoke/<a>__<b>/<seed>``
    (league decks, bo1) or ``match/<a>__<b>/<seed>`` (bo3)."""
    kind, _, rest = key.partition("/")
    if kind == "corpus":
        import replay_diff
        a, b = corpus_scenarios()[rest]
        return a, b, replay_diff.SEED, False
    pair, _, seed = rest.partition("/")
    a, _, b = pair.partition("__")
    return f"league/{a}", f"league/{b}", int(seed), kind == "match"


# The scripted agent on each protocol: on RMSTATE through the pass-1 parity adapter (the vector
# view of its rebuilt observation), not the frame view the plain spec reads there.
AGENT_SPEC = {"bquery": "scripted", "rmstate": "scripted?view=vector"}


def play(key, machine_format, binary=BINARY):
    """The verbose transcript of one scripted-hard game (as replay_diff records the corpus), the
    agent on the vector view (AGENT_SPEC)."""
    import runner
    from opponents import make_controller
    deck_a, deck_b, seed, bo3 = game_setup(key)
    spec = AGENT_SPEC[machine_format]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runner.run_games(make_controller(spec), make_controller(spec), label_a="A", label_b="B",
                         binary_path=binary, deck_a=deck_a, deck_b=deck_b, n_games=1, seed=seed, bo3=bo3,
                         verbose=True, machine_format=machine_format)
    return buf.getvalue()


def project(text):
    """The transcript lines both protocols must agree on (module docstring)."""
    out = []
    in_board = False
    for line in text.splitlines():
        if line.startswith("--- Decision "):
            out.append(line)
            in_board = True
            continue
        if in_board:
            if line == "  Actions:":
                in_board = False
                out.append(line)
            continue
        if line.startswith(("  PA: ", "  PB: ")) and line[6:] != "(empty)":
            line = line[:6] + ", ".join(sorted(line[6:].split(", ")))
        out.append(line)
    return out


def first_difference(expected, actual):
    """(line number, expected line, actual line, decision number) of the first difference of two
    projections, or None when they are equal."""
    decision = 0
    for i in range(max(len(expected), len(actual))):
        e = expected[i] if i < len(expected) else "<end>"
        a = actual[i] if i < len(actual) else "<end>"
        if e.startswith("--- Decision ") and e == a:
            decision = int(e.split()[2])
        if e != a:
            return i + 1, e, a, decision
    return None


def baseline_path(key):
    return os.path.join(BASELINE_DIR, key.replace("/", "__") + ".txt")


def load_divergences(path=DIVERGENCES_PATH):
    """The listed divergences, keyed by game (checked: every classification names a signature of
    knowledge_divergences.json)."""
    import test_knowledge
    names = {sig["name"]: sig["classification"] for sig in test_knowledge.load_divergences()}
    with open(path) as f:
        entries = json.load(f)
    out = {}
    for e in entries:
        missing = {"game", "decision", "seat", "read", "classification", "why"} - set(e)
        if missing:
            raise ValueError(f"{path}: entry {e.get('game')} lacks {sorted(missing)}")
        if e["classification"] not in names:
            raise ValueError(f"{path}: {e['game']}: classification {e['classification']!r} is no signature "
                             "of knowledge_divergences.json")
        out[e["game"]] = e
    return out


def _compare_task(task):
    """One game's check (a process-pool task): (key, failure or None, note). A game with a
    frozen BQUERY transcript (``reference``) is also played on BQUERY and must equal it."""
    key, binary, reference, listed = task
    rm = project(play(key, "rmstate", binary))
    bq = play(key, "bquery", binary)
    if reference is not None and bq != reference:
        line, e, a, decision = first_difference(reference.splitlines(), bq.splitlines())
        return key, (f"{key}: the BQUERY game differs from its frozen transcript at line {line} "
                     f"(decision {decision}): {e!r} vs {a!r} (re-record with --record when intended)"), None
    ref = project(bq)
    diff = first_difference(ref, rm)
    if listed is None:
        if diff is None:
            return key, None, f"{len(rm)} lines identical" + (", BQUERY frozen" if reference is not None else "")
        line, e, a, decision = diff
        return key, (f"{key}: first differs at line {line} (decision {decision}): bquery {e!r}, rmstate {a!r} "
                     f"(attribute it: test_agent_parity.py --lockstep {key})"), None
    try:
        with open(baseline_path(key)) as f:
            base = f.read().splitlines()
    except OSError:
        return key, f"{key}: listed divergence without its baseline {baseline_path(key)} (--record)", None
    fails = []
    bdiff = first_difference(base, rm)
    if bdiff is not None:
        fails.append(f"{key}: differs from its re-recorded baseline at line {bdiff[0]}: {bdiff[1]!r} vs {bdiff[2]!r}")
    if diff is None:
        fails.append(f"{key}: listed as divergent ({listed['classification']}) but identical on both protocols")
    elif diff[3] != listed["decision"]:
        fails.append(f"{key}: first differs at decision {diff[3]}, listed at {listed['decision']}")
    return key, "; ".join(fails) or None, f"re-recorded divergence ({listed['classification']}, decision {listed['decision']})"


def bquery_corpus_path(name):
    return os.path.join(BQUERY_CORPUS_DIR, name + ".txt")


def check_games(keys, binary, references=None, workers=None):
    """Failure messages for the games `keys` (a corpus game with a frozen BQUERY transcript
    compares with it)."""
    listed = load_divergences()
    tasks = []
    for key in keys:
        ref = None
        if key.startswith("corpus/") and os.path.exists(bquery_corpus_path(key[len("corpus/"):])):
            with open(bquery_corpus_path(key[len("corpus/"):])) as f:
                ref = f.read()
        tasks.append((key, binary, ref, listed.get(key)))
    fails = []
    with ProcessPoolExecutor(max_workers=workers or min(len(tasks), os.cpu_count() or 4)) as ex:
        for key, fail, note in ex.map(_compare_task, tasks):
            print(f"  {'FAIL' if fail else 'ok  '} {key}" + (f": {note}" if note else ""), flush=True)
            if fail:
                fails.append(fail)
    return fails


def record(binary):
    """Rewrites the baseline of every listed divergence and the frozen BQUERY transcripts of the
    light corpus set."""
    os.makedirs(BQUERY_CORPUS_DIR, exist_ok=True)
    for name in CORPUS_GAMES:
        with open(bquery_corpus_path(name), "w") as f:
            f.write(play(f"corpus/{name}", "bquery", binary))
        print(f"  recorded {bquery_corpus_path(name)}", flush=True)
    os.makedirs(BASELINE_DIR, exist_ok=True)
    for key in load_divergences():
        with open(baseline_path(key), "w") as f:
            f.write("\n".join(project(play(key, "rmstate", binary))) + "\n")
        print(f"  recorded {baseline_path(key)}", flush=True)


# ── card knowledge ────────────────────────────────────────────────────────────────────────────

def check_cards(binary):
    """FrameCardKnowledge from a closed-decklist CARDS frame against VocabCardKnowledge."""
    import numpy as np
    import rmstate
    import scripted_agent
    from card_props import _PROP_NAMES
    from rmstate_adapter import FrameCardKnowledge
    from rmstate_env import RMSTATE_ENGINE_FLAGS
    proc = subprocess.Popen([os.path.realpath(binary), *RMSTATE_ENGINE_FLAGS, "--closed-decklist", "--seed", "1",
                             "--deck-a", "league/bug", "--deck-b", "league/burn"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            cwd=os.path.join(_REPO_ROOT, "bin"))
    dec = rmstate.StreamDecoder()
    cards = decklists = None
    try:
        for payload in rmstate.records(proc.stdout):
            frame = dec.feed(payload)
            if frame.kind == "CARDS":
                cards = frame
            elif frame.kind == "DECKLISTS":
                decklists = frame
                break
    finally:
        proc.kill()
        proc.wait()
    if cards is None or decklists is None:
        return ["cards: the engine wrote no CARDS / DECKLISTS frame"]
    frame_k = FrameCardKnowledge(cards, decklists)
    if not frame_k.has_scripts():
        return ["cards: the CARDS frame carries no scripts (--rmstate-card-scripts)"]
    vocab_k = scripted_agent.VocabCardKnowledge()
    zero = np.zeros(7, dtype=np.float32)
    fails = []
    for cid in sorted(frame_k.names):
        name = frame_k.names[cid]
        fr, vr = frame_k.cost_row(cid), vocab_k.cost_row(cid)
        if not np.array_equal(fr if fr is not None else zero, vr if vr is not None else zero):
            fails.append(f"cards: {cid} {name}: cost row {fr} vs {vr}")
        for col, prop in enumerate(_PROP_NAMES):
            if frame_k.prop(cid, col) != vocab_k.prop(cid, col):
                fails.append(f"cards: {cid} {name}: {prop} {frame_k.prop(cid, col)} vs {vocab_k.prop(cid, col)}")
        for what in ("is_land", "script_lines", "types_line"):
            if getattr(frame_k, what)(cid) != getattr(vocab_k, what)(cid):
                fails.append(f"cards: {cid} {name}: {what} differs")
    print(f"  {len(frame_k.names)} cards compared", flush=True)
    return fails


# ── lockstep diagnostic ───────────────────────────────────────────────────────────────────────

def _blocks(obs):
    """An observation's comparable parts: each block as the agent can tell it apart (permanent
    slots as a sorted list, their references as the referenced card; the slot order follows
    entity ids, which the format does not carry)."""
    import numpy as np
    import env as E

    def did(v):
        return int(round(float(v) * E.N_CARD_TYPES))

    def dref(v):
        return int(round(float(v) * E.N_ENTITY_REF_SLOTS)) - 1

    slot_card = {}
    for side, start in ((0, E._SELF_PERM_START), (1, E._OPP_PERM_START)):
        for k in range(E._PERM_SLOTS):
            slot_card[side * E._PERM_SLOTS + k] = did(obs[start + k * E._PERM_SLOT_SIZE + E._PERM_CARD_OFF])
    for k in range(E._STACK_SLOTS):
        slot_card[2 * E._PERM_SLOTS + k] = ("stack", did(obs[E._STACK_START + k * E._STACK_SLOT_SIZE + 1]))
    ref_offs = (E._OFF_ATTACHED_TO, E._OFF_ATTACHED_BY, E._OFF_ATTACK_TGT, E._OFF_BLOCKING_TGT)
    perms = []
    for side, start in ((0, E._SELF_PERM_START), (1, E._OPP_PERM_START)):
        for k in range(E._PERM_SLOTS):
            v = obs[start + k * E._PERM_SLOT_SIZE:start + (k + 1) * E._PERM_SLOT_SIZE].copy()
            if did(v[E._PERM_CARD_OFF]) < 0:
                continue
            refs = tuple(slot_card.get(dref(v[r])) if dref(v[r]) >= 0 else None for r in ref_offs)
            v[list(ref_offs)] = 0
            perms.append(repr((side, np.round(v, 5).tolist(), refs)))
    stack = []
    for k in range(E._STACK_SLOTS):
        v = obs[E._STACK_START + k * E._STACK_SLOT_SIZE:E._STACK_START + (k + 1) * E._STACK_SLOT_SIZE].copy()
        refs = []
        for t in range(E._STACK_TGT_SLOTS):
            i = E._STACK_TGT_START + t * E._STACK_TGT_FIELDS + 3
            refs.append(slot_card.get(dref(v[i])) if dref(v[i]) >= 0 else None)
            v[i] = 0
        stack.append(repr((np.round(v, 5).tolist(), refs)))
    acts = []
    for i in range(E.MAX_ACTIONS):
        r = dref(obs[E.ACT_REFS_START + i])
        acts.append(repr((round(float(obs[E.ACT_CATS_START + i]), 5), did(obs[E.ACT_IDS_START + i]),
                          round(float(obs[E.ACT_CTRL_START + i]), 5), round(float(obs[E.ACT_ZONE_START + i]), 5),
                          round(float(obs[E.ACT_ORDS_START + i]), 5), slot_card.get(r) if r >= 0 else None)))
    flat = lambda lo, hi: repr(np.round(obs[lo:hi], 5).tolist())  # noqa: E731
    return {
        "header": flat(0, E._SELF_PERM_START),
        "battlefield": sorted(perms),
        "stack": stack,
        "graveyard_exile": flat(E._GY_START, E._HAND_START),
        "hand": sorted(did(obs[E._HAND_START + k]) for k in range(E.MAX_HAND_SLOTS)),
        "context": flat(E._MATCH_CTX_START, E._KNOWN_TOP_LIB_START),
        "own_top": flat(E._KNOWN_TOP_LIB_START, E._KNOWN_TOP_LIB_END),
        "opp_top": flat(E._OPP_KNOWN_TOP_LIB_START, E._OPP_KNOWN_TOP_LIB_END),
        "opp_known_hand": sorted(did(obs[E._OPP_KNOWN_HAND_START + k]) for k in range(E.MAX_HAND_SLOTS)),
        "pending": flat(E._PENDING_DECISION_START, E._PENDING_DECISION_END),
        "extras": flat(E._EXTRAS_START, E._EXTRAS_END),
        "opp_decklist": repr([round(float(obs[i]), 5) for i in range(E._OPP_DECK_MAIN_START, E._OPP_DECK_SIDE_END)
                              if (i - E._OPP_DECK_MAIN_START) % E._OPP_DECKLIST_SLOT_SIZE
                              != E._OPP_DECKLIST_REVEALED_OFF]),
        "per_turn": flat(E._PER_TURN_START, E._PER_TURN_END),
        "actions": acts,
    }


# The SHADOW comparison's kind and owner for each memory read of the old vector.
_SHADOW_OF_BLOCK = {"own_top": ("top", "self"), "opp_top": ("top", "opp"), "opp_known_hand": ("hand", "opp")}


def _shadow_at(lines, seat, header, blocks):
    """The latest SHADOW line of the deciding seat, in the frame's game and at or before its
    decision, for each memory block that differs."""
    out = []
    for block in blocks:
        if block not in _SHADOW_OF_BLOCK:
            continue
        kind, owner = _SHADOW_OF_BLOCK[block]
        latest = None
        for line in lines:
            if not line.startswith(f"SHADOW: {kind} ") or f" viewer={seat} owner={owner} " not in line:
                continue
            if f" game={header['game']} " not in line:
                continue
            decision = int(line.split(" decision=")[1].split()[0])
            sideboarding = " sideboarding=1 " in line
            if sideboarding == bool(header["sideboarding"]) and decision <= header["decision"]:
                latest = line
        if latest is not None:
            out.append(latest)
    return out


def lockstep(key, binary=BINARY, show=3):
    """Plays the BQUERY game with a dump and the knowledge self-check; compares every decision's
    adapter observation and agent choice with the real ones. Returns a stats Counter."""
    import rmstate
    import runner
    import scripted_agent
    import test_knowledge
    from opponents import make_controller
    from rmstate_adapter import SeatAdapter
    from rmstate_env import RMSTATE_ENGINE_FLAGS
    deck_a, deck_b, seed, bo3 = game_setup(key)
    work = tempfile.mkdtemp(prefix="agentparity_")
    dump = os.path.join(work, "game.rms")
    err_path = os.path.join(work, "engine.err")
    records = []
    saved = os.environ.get("ROBOMAGE_ENGINE_FLAGS")
    os.environ["ROBOMAGE_ENGINE_FLAGS"] = f"--rmstate-dump {dump} {RMSTATE_ENGINE_FLAGS[2]} --selfcheck-knowledge"
    err_fd = os.open(err_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    saved_err = os.dup(2)
    os.dup2(err_fd, 2)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run_games(make_controller(AGENT_SPEC["bquery"]), make_controller(AGENT_SPEC["bquery"]),
                             deck_a=deck_a, deck_b=deck_b,
                             n_games=1, seed=seed, bo3=bo3, transcript="quiet", binary_path=binary,
                             on_query=lambda d: records.append((d.obs.copy(), d.num_choices, d.priority_is_a,
                                                                list(d.env._action_descriptions or []))))
    finally:
        os.dup2(saved_err, 2)
        os.close(saved_err)
        os.close(err_fd)
        if saved is None:
            os.environ.pop("ROBOMAGE_ENGINE_FLAGS", None)
        else:
            os.environ["ROBOMAGE_ENGINE_FLAGS"] = saved
    adapters = {"A": SeatAdapter("A"), "B": SeatAdapter("B")}
    dec = rmstate.StreamDecoder()
    frames = []
    with open(dump, "rb") as fp:
        for payload in rmstate.records(fp):
            frame = dec.feed(payload)
            if frame.viewer not in adapters:
                continue
            adapters[frame.viewer].feed(frame, dec.schema)
            if frame.kind == "STATE" and "menu" in frame.tables and not frame.tables["header"][0]["terminal"]:
                frames.append((frame.viewer, *adapters[frame.viewer].observation(frame), frame))
    sigs = test_knowledge.load_divergences()
    shadow = [ln for ln in open(err_path, errors="replace") if ln.startswith("SHADOW:")]
    for path in (dump, dump + ".jsonl", err_path):
        if os.path.exists(path):
            os.remove(path)
    os.rmdir(work)
    print(f"{key}: {len(records)} bquery decisions, {len(frames)} rmstate decision frames", flush=True)
    real = {True: scripted_agent.make_agent("scripted"), False: scripted_agent.make_agent("scripted")}
    shadow_agents = {True: scripted_agent.make_agent("scripted"), False: scripted_agent.make_agent("scripted")}
    for agent in list(real.values()) + list(shadow_agents.values()):
        agent.set_deck_names(deck_a, deck_b)
    random.seed(seed)
    stats = collections.Counter()
    shown = 0
    for k, ((obs, n, is_a, descs), (seat, adapted, side, frame)) in enumerate(zip(records, frames)):
        stats["decisions"] += 1
        bq, rm = _blocks(obs), _blocks(adapted)
        diffs = [b for b in bq if bq[b] != rm[b]]
        for b in diffs:
            stats["block " + b] += 1
        state = random.getstate()
        scripted_agent.use_card_knowledge(None)
        a_bq = real[is_a].act(obs, n)
        after = random.getstate()
        random.setstate(state)
        scripted_agent.use_card_knowledge(adapters[seat].cards)
        a_rm = shadow_agents[is_a].act(adapted, n)
        scripted_agent.use_card_knowledge(None)
        random.setstate(after)
        if a_bq != a_rm:
            stats["choice differs"] += 1
        if a_bq != a_rm or (diffs and shown < show):
            shown += a_bq == a_rm
            header = frame.tables["header"][0]
            print(f"  decision {k + 1} (game {header['game']}, frame decision {header['decision']}) seat {seat}: "
                  f"choice bquery {a_bq} rmstate {a_rm}; blocks differ: {', '.join(diffs) or '-'}", flush=True)
            if a_bq != a_rm:
                print(f"    menu: {descs[:n]}", flush=True)
                for line in _shadow_at(shadow, seat, header, diffs):
                    print(f"    {test_knowledge.classify(line, sigs) or 'UNCLASSIFIED'}: {line.strip()[:400]}",
                          flush=True)
    print(f"  {dict(stats)}", flush=True)
    return stats


# ── the tier ──────────────────────────────────────────────────────────────────────────────────

def run(binary=BINARY, parts=None, full=False):
    """The tier; returns failure messages (empty = pass)."""
    parts = list(parts or PARTS)
    fails = []
    times = []
    for part in PARTS:
        if part not in parts:
            continue
        t0 = time.monotonic()
        print(f"agentparity: {part}", flush=True)
        if part == "cards":
            fails += check_cards(binary)
        elif part == "corpus":
            names = sorted(corpus_scenarios()) if full else CORPUS_GAMES
            fails += check_games([f"corpus/{n}" for n in names], binary)
        elif part == "smoke":
            games = full_smoke_games() + sorted(load_divergences()) if full else list(SMOKE_GAMES + DIVERGENT_GAMES)
            fails += check_games(games, binary)
        times.append((part, time.monotonic() - t0))
    print("agentparity: " + ", ".join(f"{p} {dt:.0f}s" for p, dt in times), flush=True)
    return fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--binary", default=BINARY, help="engine binary (default: the debug tier)")
    ap.add_argument("--part", action="append", choices=PARTS, help="run only this part (repeatable)")
    ap.add_argument("--full", action="store_true",
                    help="every replay-corpus game, every smoke-tier game and a bo3 match per league mirror")
    ap.add_argument("--games", nargs="+", metavar="GAME",
                    help="check only these games (corpus/<scenario>, smoke/<a>__<b>/<seed>, match/<a>__<b>/<seed>)")
    ap.add_argument("--record", action="store_true",
                    help="re-record the RMSTATE baseline of every listed divergence and the frozen "
                         "BQUERY transcripts of the light corpus set")
    ap.add_argument("--lockstep", nargs="+", metavar="GAME",
                    help="the diagnostic: per-decision adapter observation and choice against the BQUERY game")
    ap.add_argument("--show", type=int, default=3, help="--lockstep: decisions with differing blocks to print")
    args = ap.parse_args(argv)
    if args.lockstep:
        for key in args.lockstep:
            lockstep(key, args.binary, args.show)
        return 0
    if args.record:
        record(args.binary)
        return 0
    fails = check_games(args.games, args.binary) if args.games else run(args.binary, args.part, args.full)
    for msg in fails:
        print(f"FAIL {msg}", flush=True)
    print("agentparity: " + ("FAIL" if fails else "PASS"), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
