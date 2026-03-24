"""
Analyze .rmrec recording files produced by train.py --record.

Usage:
    python analysis.py summary <file.rmrec>
    python analysis.py winrate <file.rmrec>
    python analysis.py actions <file.rmrec>
    python analysis.py cards <file.rmrec>
    python analysis.py replay <file.rmrec> --game <N>
    python analysis.py compare <file1.rmrec> <file2.rmrec>
    python analysis.py wl-split <file.rmrec>
    python analysis.py cast-timing <file.rmrec>
    python analysis.py choice-rates <file.rmrec>
    python analysis.py shap <model.zip> --opponent scripted [--n-games 50 --n-samples 200 --n-background 50]
    python analysis.py value-swings <model.zip> --opponent scripted [--n-games 50 --top 10]
    python analysis.py regret <model.zip> --opponent scripted [--n-games 50 --top 20]
    python analysis.py entropy <model.zip> --opponent scripted [--n-games 50]
    python analysis.py consistency <model.zip> --opponent scripted [--n-games 50 --top 20]
    python analysis.py interactive <model.zip> --opponent scripted [--n-games 20]

Interactive session commands (available after shap, value-swings, or via 'interactive'):
    list                  list all games
    replay <N>            board-state trace for game N
    boardstate <N> [step] full board + decision detail; enters GDB-style stepping mode
    summary               win/loss/draw stats
    swings [N]            top N value-function swings
    shap                  run SHAP analysis on collected data
    regret [N]            policy regret analysis (top N high-regret decisions)
    entropy               policy entropy by game phase and board state
    consistency [N]       decision consistency for similar states (top N pairs)
    calibration           V(s) at game start vs actual win rate
    turning               find the permanent zero-crossing ('point of no return')
    clusters              classify games by V(s) curve shape (archetypes)
    chart <N>             value curve plot for game N
    chart swings [N]      value curve plots for top N swing games
    chart shap            SHAP summary plot
    chart calibration     calibration curve plot
    chart turning         turning point distribution plot
    chart clusters        overlay V(s) curves by archetype
    run <N>               simulate N more games (interactive command only)
    quit                  exit
"""

import argparse
import struct
import sys
import os
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Import format constants from train.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (
    REC_MAGIC, REC_VERSION, REC_GAME_START, REC_DECISION, REC_GAME_END,
    _SESSION_HDR_FMT, _GAME_START_FMT, _DECISION_FMT, _GAME_END_FMT,
    _write_length_prefixed, _read_length_prefixed,
    _CAT_NAMES, _STEP_NAMES,
)
from card_costs import _VOCAB_NAMES, N_CARD_TYPES
from env import (ACTION_CATEGORY_MAX, RoboMageEnv, scripted_action,
                 OBS_SIZE, STATE_SIZE, MAX_ACTIONS, BINARY,
                 _HAND_START)


# ── Data classes for parsed records ──────────────────────────────────────────

@dataclass
class SessionHeader:
    version: int
    timestamp: int
    n_envs: int
    model_path: str
    model_deck: str
    self_play: bool


@dataclass
class GameStart:
    env_id: int
    game_id: int
    model_is_a: bool
    opp_deck: str
    opp_type: str


@dataclass
class Decision:
    env_id: int
    game_id: int
    decision_idx: int
    step_idx: int
    priority_is_a: bool
    active_is_a: bool
    num_choices: int
    action_chosen: int
    self_life: int
    opp_life: int
    self_mana: bytes
    categories: bytes
    card_ids: bytes
    ctrl_flags: bytes
    self_creatures: int
    self_lands: int
    opp_creatures: int
    opp_lands: int


@dataclass
class GameEnd:
    env_id: int
    game_id: int
    result: int  # +1 win, -1 loss, 0 draw
    n_decisions: int
    total_reward: float
    timestep: int


# ── Record reader ────────────────────────────────────────────────────────────

class RecordReader:
    """Iterator over a .rmrec file, yielding typed records."""

    def __init__(self, path: str):
        self.path = path
        self._f = open(path, "rb")
        self.header = self._read_header()

    def _read_header(self) -> SessionHeader:
        hdr_size = struct.calcsize(_SESSION_HDR_FMT)
        data = self._f.read(hdr_size)
        if len(data) < hdr_size:
            raise ValueError("File too short for session header")
        magic, version, timestamp, n_envs = struct.unpack(_SESSION_HDR_FMT, data)
        if magic != REC_MAGIC:
            raise ValueError(f"Bad magic: {magic!r}, expected {REC_MAGIC!r}")
        model_path = _read_length_prefixed(self._f)
        model_deck = _read_length_prefixed(self._f)
        (self_play_byte,) = struct.unpack("B", self._f.read(1))
        return SessionHeader(version, timestamp, n_envs, model_path, model_deck,
                             bool(self_play_byte))

    def __iter__(self):
        return self

    def __next__(self):
        tag_byte = self._f.read(1)
        if not tag_byte:
            raise StopIteration
        tag = tag_byte[0]

        if tag == REC_GAME_START:
            # We already consumed the type byte; read the rest
            rest_size = struct.calcsize(_GAME_START_FMT) - 1
            data = tag_byte + self._f.read(rest_size)
            _, env_id, game_id, model_is_a = struct.unpack(_GAME_START_FMT, data)
            opp_deck = _read_length_prefixed(self._f)
            opp_type = _read_length_prefixed(self._f)
            return GameStart(env_id, game_id, bool(model_is_a), opp_deck, opp_type)

        elif tag == REC_DECISION:
            rest_size = struct.calcsize(_DECISION_FMT) - 1
            data = tag_byte + self._f.read(rest_size)
            (_, env_id, game_id, decision_idx, step_idx, priority_is_a, active_is_a,
             num_choices, action_chosen, self_life, opp_life, self_mana,
             categories, card_ids, ctrl_flags,
             self_creatures, self_lands, opp_creatures, opp_lands) = struct.unpack(
                _DECISION_FMT, data)
            return Decision(env_id, game_id, decision_idx, step_idx,
                            bool(priority_is_a), bool(active_is_a),
                            num_choices, action_chosen, self_life, opp_life,
                            self_mana, categories, card_ids, ctrl_flags,
                            self_creatures, self_lands, opp_creatures, opp_lands)

        elif tag == REC_GAME_END:
            rest_size = struct.calcsize(_GAME_END_FMT) - 1
            data = tag_byte + self._f.read(rest_size)
            _, env_id, game_id, result, n_decisions, total_reward, timestep = struct.unpack(
                _GAME_END_FMT, data)
            return GameEnd(env_id, game_id, result, n_decisions, total_reward, timestep)

        else:
            raise ValueError(f"Unknown record tag: 0x{tag:02x} at offset {self._f.tell()}")

    def close(self):
        self._f.close()


# ── Session data loader ─────────────────────────────────────────────────────

@dataclass
class SessionData:
    header: SessionHeader
    game_starts: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    game_ends: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str) -> "SessionData":
        reader = RecordReader(path)
        sd = cls(header=reader.header)
        for rec in reader:
            if isinstance(rec, GameStart):
                sd.game_starts.append(rec)
            elif isinstance(rec, Decision):
                sd.decisions.append(rec)
            elif isinstance(rec, GameEnd):
                sd.game_ends.append(rec)
        reader.close()
        return sd


# ── CLI subcommands ──────────────────────────────────────────────────────────

def cmd_summary(args):
    sd = SessionData.load(args.file)
    h = sd.header
    ts = datetime.fromtimestamp(h.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    print(f"Session: {os.path.basename(args.file)}")
    print(f"  Recorded:   {ts}")
    print(f"  Model:      {h.model_path}")
    print(f"  Deck:       {h.model_deck}")
    print(f"  Self-play:  {h.self_play}")
    print(f"  Envs:       {h.n_envs}")
    print(f"  Games:      {len(sd.game_ends)}")
    print(f"  Decisions:  {len(sd.decisions)}")

    if not sd.game_ends:
        return

    # Win rate
    wins = sum(1 for g in sd.game_ends if g.result > 0)
    losses = sum(1 for g in sd.game_ends if g.result < 0)
    draws = sum(1 for g in sd.game_ends if g.result == 0)
    total = len(sd.game_ends)
    print(f"\n  Win rate:   {wins}W / {losses}L / {draws}D ({100 * wins / total:.1f}%)")

    # By opponent deck
    deck_map = {g.game_id: g.opp_deck for g in sd.game_starts}
    side_map = {g.game_id: g.model_is_a for g in sd.game_starts}
    by_deck = {}
    by_side = {"A": [0, 0], "B": [0, 0]}
    for g in sd.game_ends:
        deck = deck_map.get(g.game_id, "unknown")
        if deck not in by_deck:
            by_deck[deck] = [0, 0, 0]
        if g.result > 0:
            by_deck[deck][0] += 1
        elif g.result < 0:
            by_deck[deck][1] += 1
        else:
            by_deck[deck][2] += 1
        side = "A" if side_map.get(g.game_id, True) else "B"
        if g.result > 0:
            by_side[side][0] += 1
        else:
            by_side[side][1] += 1

    if len(by_deck) > 1:
        print("\n  By opponent deck:")
        for deck in sorted(by_deck):
            w, l, d = by_deck[deck]
            t = w + l + d
            print(f"    vs {deck}: {w}W {l}L {d}D ({100 * w / t:.1f}%)")

    for side in ("A", "B"):
        w, l = by_side[side]
        t = w + l
        if t > 0:
            print(f"  As Player {side}: {w}W {l}L ({100 * w / t:.1f}%)")

    # Game length stats
    lengths = [g.n_decisions for g in sd.game_ends if g.n_decisions > 0]
    if lengths:
        arr = np.array(lengths)
        print(f"\n  Game length: mean={arr.mean():.1f}  median={np.median(arr):.0f}"
              f"  min={arr.min()}  max={arr.max()}")

    # Top 10 most-cast cards
    cast_counts = {}
    for d in sd.decisions:
        cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        if cat == 7:  # CAST
            cid = d.card_ids[d.action_chosen] if d.action_chosen < len(d.card_ids) else 255
            if cid < len(_VOCAB_NAMES) and cid != 255:
                name = _VOCAB_NAMES[cid]
                cast_counts[name] = cast_counts.get(name, 0) + 1
    if cast_counts:
        print("\n  Top cast cards:")
        for name, count in sorted(cast_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {count:5d}  {name}")

    # Action category distribution
    cat_counts = {}
    for d in sd.decisions:
        cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        if cat != 255:
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    if cat_counts:
        total_d = sum(cat_counts.values())
        print("\n  Action distribution:")
        for cat in sorted(cat_counts):
            name = _CAT_NAMES.get(cat, str(cat))
            pct = 100 * cat_counts[cat] / total_d
            print(f"    {name:<14} {cat_counts[cat]:6d} ({pct:5.1f}%)")


def cmd_winrate(args):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    sd = SessionData.load(args.file)
    if not sd.game_ends:
        print("No completed games.")
        return

    deck_map = {g.game_id: g.opp_deck for g in sd.game_starts}
    # Group results by deck, ordered by game sequence
    by_deck = {}
    for g in sd.game_ends:
        deck = deck_map.get(g.game_id, "unknown")
        by_deck.setdefault(deck, []).append(1 if g.result > 0 else 0)

    window = min(100, max(10, len(sd.game_ends) // 20))
    fig, ax = plt.subplots(figsize=(10, 5))
    for deck, results in sorted(by_deck.items()):
        arr = np.array(results, dtype=float)
        if len(arr) < window:
            continue
        rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(arr)), rolling * 100, label=f"vs {deck}")

    ax.set_xlabel("Game #")
    ax.set_ylabel(f"Win rate (%) (rolling {window})")
    ax.set_title(f"Win Rate — {os.path.basename(args.file)}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def cmd_actions(args):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    sd = SessionData.load(args.file)
    if not sd.decisions:
        print("No decisions recorded.")
        return

    n_steps = len(_STEP_NAMES)
    n_cats = max(_CAT_NAMES.keys()) + 1
    heatmap = np.zeros((n_cats, n_steps), dtype=float)

    for d in sd.decisions:
        cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        if cat < n_cats and d.step_idx < n_steps:
            heatmap[cat, d.step_idx] += 1

    # Normalize columns (per step)
    col_sums = heatmap.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    heatmap_pct = heatmap / col_sums * 100

    fig, ax = plt.subplots(figsize=(12, 6))
    cat_labels = [_CAT_NAMES.get(i, str(i)) for i in range(n_cats)]
    im = ax.imshow(heatmap_pct, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(n_steps))
    ax.set_xticklabels(_STEP_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(cat_labels, fontsize=8)
    ax.set_xlabel("Game Step")
    ax.set_ylabel("Action Category")
    ax.set_title(f"Action Distribution by Step — {os.path.basename(args.file)}")
    plt.colorbar(im, ax=ax, label="% of decisions at step")
    plt.tight_layout()
    plt.show()


def cmd_cards(args):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    sd = SessionData.load(args.file)

    cast_counts = {}
    land_counts = {}
    for d in sd.decisions:
        cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        cid = d.card_ids[d.action_chosen] if d.action_chosen < len(d.card_ids) else 255
        if cid == 255 or cid >= len(_VOCAB_NAMES):
            continue
        name = _VOCAB_NAMES[cid]
        if cat == 7:  # CAST
            cast_counts[name] = cast_counts.get(name, 0) + 1
        elif cat == 9:  # LAND
            land_counts[name] = land_counts.get(name, 0) + 1

    all_names = sorted(set(list(cast_counts.keys()) + list(land_counts.keys())))
    if not all_names:
        print("No cast/land actions found.")
        return

    y = np.arange(len(all_names))
    cast_vals = [cast_counts.get(n, 0) for n in all_names]
    land_vals = [land_counts.get(n, 0) for n in all_names]

    fig, ax = plt.subplots(figsize=(10, max(4, len(all_names) * 0.35)))
    ax.barh(y - 0.2, cast_vals, 0.4, label="Cast", color="steelblue")
    ax.barh(y + 0.2, land_vals, 0.4, label="Play Land", color="forestgreen")
    ax.set_yticks(y)
    ax.set_yticklabels(all_names, fontsize=8)
    ax.set_xlabel("Count")
    ax.set_title(f"Cards Played — {os.path.basename(args.file)}")
    ax.legend()
    plt.tight_layout()
    plt.show()


def cmd_replay(args):
    sd = SessionData.load(args.file)
    target_gid = args.game

    # Find matching game
    start = None
    for gs in sd.game_starts:
        if gs.game_id == target_gid:
            start = gs
            break
    if start is None:
        print(f"Game {target_gid} not found. Available: "
              f"{min(g.game_id for g in sd.game_starts)}-{max(g.game_id for g in sd.game_starts)}")
        return

    end = None
    for ge in sd.game_ends:
        if ge.game_id == target_gid:
            end = ge
            break

    model_side = "A" if start.model_is_a else "B"
    print(f"Game {target_gid}: Model ({model_side}) vs {start.opp_type} [{start.opp_deck}]")
    if end:
        result_str = {1: "Model wins", -1: "Model loses", 0: "Draw"}.get(end.result, "?")
        print(f"Result: {result_str}  ({end.n_decisions} decisions, timestep {end.timestep})")
    print()

    decs = [d for d in sd.decisions if d.game_id == target_gid]
    decs.sort(key=lambda d: d.decision_idx)

    for d in decs:
        step_name = _STEP_NAMES[d.step_idx] if d.step_idx < len(_STEP_NAMES) else f"?{d.step_idx}"
        cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        cat_name = _CAT_NAMES.get(cat, str(cat)) if cat != 255 else "?"
        cid = d.card_ids[d.action_chosen] if d.action_chosen < len(d.card_ids) else 255
        card_name = ""
        if cid < len(_VOCAB_NAMES) and cid != 255:
            card_name = f" ({_VOCAB_NAMES[cid]})"

        mana_str = "/".join(str(b) for b in d.self_mana)
        print(f"[{d.decision_idx:3d}] {step_name:<14} "
              f"Life {d.self_life}/{d.opp_life}  "
              f"Board {d.self_creatures}c+{d.self_lands}l / {d.opp_creatures}c+{d.opp_lands}l  "
              f"Mana {mana_str}  "
              f"choices={d.num_choices}  "
              f"-> {d.action_chosen} {cat_name}{card_name}")


def cmd_compare(args):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.", file=sys.stderr)
        return

    sds = []
    for path in [args.file, args.file2]:
        sds.append(SessionData.load(path))

    window = 100
    fig, ax = plt.subplots(figsize=(10, 5))
    for sd, path in zip(sds, [args.file, args.file2]):
        results = [1 if g.result > 0 else 0 for g in sd.game_ends]
        if len(results) < window:
            continue
        arr = np.array(results, dtype=float)
        rolling = np.convolve(arr, np.ones(window) / window, mode="valid")
        label = f"{os.path.basename(path)} ({sd.header.model_path})"
        ax.plot(range(window - 1, len(arr)), rolling * 100, label=label)

    ax.set_xlabel("Game #")
    ax.set_ylabel(f"Win rate (%) (rolling {window})")
    ax.set_title("Win Rate Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _game_results(sd):
    """Return dict mapping game_id -> result (+1 win, -1 loss, 0 draw)."""
    return {g.game_id: g.result for g in sd.game_ends}


def _available_cats(d):
    """Return set of action categories available in a decision."""
    cats = set()
    for i in range(d.num_choices):
        if i < len(d.categories):
            cats.add(d.categories[i])
    return cats


def cmd_wl_split(args):
    """Action category distribution split by game outcome (win vs loss)."""
    sd = SessionData.load(args.file)
    results = _game_results(sd)
    if not results:
        print("No completed games.")
        return

    n_cats = max(_CAT_NAMES.keys()) + 1
    win_counts = np.zeros(n_cats, dtype=float)
    loss_counts = np.zeros(n_cats, dtype=float)

    for d in sd.decisions:
        r = results.get(d.game_id)
        if r is None or r == 0:
            continue
        cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        if cat >= n_cats:
            continue
        if r > 0:
            win_counts[cat] += 1
        else:
            loss_counts[cat] += 1

    # Normalize to percentages
    win_total = win_counts.sum()
    loss_total = loss_counts.sum()
    if win_total == 0 or loss_total == 0:
        print("Need both wins and losses to compare.")
        return
    win_pct = win_counts / win_total * 100
    loss_pct = loss_counts / loss_total * 100

    print(f"Action distribution: wins vs losses — {os.path.basename(args.file)}")
    print(f"  ({int(win_total)} win decisions, {int(loss_total)} loss decisions)\n")
    print(f"  {'Category':<14} {'Win%':>6}  {'Loss%':>6}  {'Delta':>6}")
    print(f"  {'-' * 14} {'-' * 6}  {'-' * 6}  {'-' * 6}")
    for i in range(n_cats):
        if win_counts[i] == 0 and loss_counts[i] == 0:
            continue
        name = _CAT_NAMES.get(i, str(i))
        delta = win_pct[i] - loss_pct[i]
        marker = " *" if abs(delta) > 2.0 else ""
        print(f"  {name:<14} {win_pct[i]:5.1f}%  {loss_pct[i]:5.1f}%  {delta:+5.1f}%{marker}")

    # Per-step breakdown for categories with large deltas
    n_steps = len(_STEP_NAMES)
    interesting_cats = [i for i in range(n_cats) if abs(win_pct[i] - loss_pct[i]) > 2.0]
    if interesting_cats:
        print(f"\n  Per-step breakdown for categories with >2% delta:\n")
        for cat_idx in interesting_cats:
            cat_name = _CAT_NAMES.get(cat_idx, str(cat_idx))
            win_by_step = np.zeros(n_steps)
            loss_by_step = np.zeros(n_steps)
            for d in sd.decisions:
                r = results.get(d.game_id)
                if r is None or r == 0:
                    continue
                c = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
                if c != cat_idx or d.step_idx >= n_steps:
                    continue
                if r > 0:
                    win_by_step[d.step_idx] += 1
                else:
                    loss_by_step[d.step_idx] += 1
            w_tot = win_by_step.sum()
            l_tot = loss_by_step.sum()
            if w_tot == 0 or l_tot == 0:
                continue
            print(f"  {cat_name}:")
            for s in range(n_steps):
                if win_by_step[s] == 0 and loss_by_step[s] == 0:
                    continue
                wp = win_by_step[s] / w_tot * 100
                lp = loss_by_step[s] / l_tot * 100
                print(f"    {_STEP_NAMES[s]:<14} win {wp:5.1f}%  loss {lp:5.1f}%")
            print()


def cmd_cast_timing(args):
    """Per-card cast timing and board state, split by game outcome."""
    sd = SessionData.load(args.file)
    results = _game_results(sd)
    if not results:
        print("No completed games.")
        return

    # Collect cast decisions per card
    # Each entry: (game_result, step_idx, decision_idx_in_game, life_diff, creature_diff)
    card_casts = {}
    # Track per-game decision count for relative timing
    game_decision_counts = {}
    for d in sd.decisions:
        game_decision_counts[d.game_id] = max(
            game_decision_counts.get(d.game_id, 0), d.decision_idx + 1)

    for d in sd.decisions:
        r = results.get(d.game_id)
        if r is None or r == 0:
            continue
        cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        if cat != 7:  # CAST only
            continue
        cid = d.card_ids[d.action_chosen] if d.action_chosen < len(d.card_ids) else 255
        if cid >= len(_VOCAB_NAMES) or cid == 255:
            continue
        name = _VOCAB_NAMES[cid]
        life_diff = d.self_life - d.opp_life
        creature_diff = d.self_creatures - d.opp_creatures
        card_casts.setdefault(name, []).append((
            r, d.step_idx, d.decision_idx, life_diff, creature_diff,
            game_decision_counts.get(d.game_id, 1)))

    if not card_casts:
        print("No cast actions found.")
        return

    print(f"Cast timing by card — {os.path.basename(args.file)}\n")

    for name in sorted(card_casts, key=lambda n: -len(card_casts[n])):
        entries = card_casts[name]
        if len(entries) < 5:
            continue
        wins = [e for e in entries if e[0] > 0]
        losses = [e for e in entries if e[0] < 0]

        print(f"  {name} ({len(wins)}W / {len(losses)}L casts)")

        for label, subset in [("win", wins), ("loss", losses)]:
            if len(subset) < 2:
                continue
            steps = np.array([e[1] for e in subset])
            # Relative timing: decision_idx / total_decisions_in_game
            rel_timing = np.array([e[2] / e[5] for e in subset])
            life_diffs = np.array([e[3] for e in subset])
            creature_diffs = np.array([e[4] for e in subset])

            # Most common step
            step_counts = np.bincount(steps, minlength=len(_STEP_NAMES))
            top_step = _STEP_NAMES[np.argmax(step_counts)]

            print(f"    {label}: timing {rel_timing.mean():.0%} thru game"
                  f"  life_diff {life_diffs.mean():+.1f}"
                  f"  creature_diff {creature_diffs.mean():+.1f}"
                  f"  step: {top_step}")
        print()


def cmd_choice_rates(args):
    """P(chose X | X was legal) conditioned on board state."""
    sd = SessionData.load(args.file)
    results = _game_results(sd)

    # For each action category, track: times available, times chosen,
    # bucketed by board state
    def _board_bucket(d):
        """Categorize board state as a simple label."""
        life_diff = d.self_life - d.opp_life
        creature_diff = d.self_creatures - d.opp_creatures
        if life_diff <= -5:
            life_label = "life--"
        elif life_diff < 0:
            life_label = "life-"
        elif life_diff == 0:
            life_label = "life="
        elif life_diff < 5:
            life_label = "life+"
        else:
            life_label = "life++"

        if creature_diff < -1:
            board_label = "behind"
        elif creature_diff > 1:
            board_label = "ahead"
        else:
            board_label = "even"
        return life_label, board_label

    # cats we care about for choice-rate analysis
    interesting = {0, 2, 6, 7, 9}  # PASS, SEL_ATK, ACTIVATE, CAST, LAND
    n_cats = max(_CAT_NAMES.keys()) + 1

    # Nested: cat -> (life_bucket, board_bucket) -> [available, chosen]
    rates = {}
    for cat in interesting:
        rates[cat] = {}

    for d in sd.decisions:
        available = _available_cats(d)
        chosen_cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
        lb, bb = _board_bucket(d)
        bucket = (lb, bb)
        for cat in interesting:
            if cat not in available:
                continue
            if bucket not in rates[cat]:
                rates[cat][bucket] = [0, 0]
            rates[cat][bucket][0] += 1
            if chosen_cat == cat:
                rates[cat][bucket][1] += 1

    print(f"Choice rates (P(chose | legal)) by board state — {os.path.basename(args.file)}\n")

    for cat in sorted(interesting):
        cat_name = _CAT_NAMES.get(cat, str(cat))
        buckets = rates[cat]
        if not buckets:
            continue
        total_avail = sum(v[0] for v in buckets.values())
        total_chosen = sum(v[1] for v in buckets.values())
        if total_avail == 0:
            continue
        overall = total_chosen / total_avail * 100
        print(f"  {cat_name} (overall {overall:.1f}% when legal):")

        # Sort buckets by life order then board order
        life_order = {"life--": 0, "life-": 1, "life=": 2, "life+": 3, "life++": 4}
        board_order = {"behind": 0, "even": 1, "ahead": 2}
        sorted_buckets = sorted(buckets.keys(),
                                key=lambda b: (life_order.get(b[0], 5), board_order.get(b[1], 5)))
        for bucket in sorted_buckets:
            avail, chosen = buckets[bucket]
            if avail < 10:
                continue
            rate = chosen / avail * 100
            lb, bb = bucket
            print(f"    {lb:<7} {bb:<7}  {rate:5.1f}%  (n={avail})")
        print()

    # Win/loss split for choice rates
    if not results:
        return

    print(f"  Choice rates split by game outcome:\n")
    for cat in sorted(interesting):
        cat_name = _CAT_NAMES.get(cat, str(cat))
        win_avail = 0
        win_chosen = 0
        loss_avail = 0
        loss_chosen = 0
        for d in sd.decisions:
            r = results.get(d.game_id)
            if r is None or r == 0:
                continue
            available = _available_cats(d)
            if cat not in available:
                continue
            chosen_cat = d.categories[d.action_chosen] if d.action_chosen < len(d.categories) else 255
            if r > 0:
                win_avail += 1
                if chosen_cat == cat:
                    win_chosen += 1
            else:
                loss_avail += 1
                if chosen_cat == cat:
                    loss_chosen += 1
        if win_avail < 10 or loss_avail < 10:
            continue
        win_rate = win_chosen / win_avail * 100
        loss_rate = loss_chosen / loss_avail * 100
        delta = win_rate - loss_rate
        marker = " *" if abs(delta) > 2.0 else ""
        print(f"  {cat_name:<14} win {win_rate:5.1f}%  loss {loss_rate:5.1f}%  delta {delta:+5.1f}%{marker}")


# ── SHAP / value-function analysis ────────────────────────────────────────────

# Human-readable feature names extracted from the raw observation vector.
# These are the features SHAP will explain.
_INTERP_FEATURE_NAMES = [
    "self_life", "opp_life", "life_diff",
    "self_mana_W", "self_mana_U", "self_mana_B",
    "self_mana_R", "self_mana_G", "self_mana_C", "self_total_mana",
    "opp_mana_W", "opp_mana_U", "opp_mana_B",
    "opp_mana_R", "opp_mana_G", "opp_mana_C", "opp_total_mana",
    "self_hand_size",
    "self_creatures", "self_lands", "self_other_perms",
    "opp_creatures", "opp_lands", "opp_other_perms",
    "creature_diff", "land_diff",
    "self_tapped_lands", "opp_tapped_lands",
    "self_attacking", "self_blocking",
    "opp_attacking", "opp_blocking",
    "self_total_power", "opp_total_power", "power_diff",
    "self_total_toughness", "opp_total_toughness",
    "self_gy_size", "opp_gy_size",
    "stack_size",
    "is_active_player",
    "step_untap", "step_upkeep", "step_draw",
    "step_first_main", "step_begin_combat",
    "step_declare_atk", "step_declare_blk",
    "step_first_strike", "step_combat_dmg",
    "step_end_combat", "step_second_main",
    "step_end", "step_cleanup",
]

# Permanent slot layout (from extractor.py / machine_io.h)
_PERM_START   = 34
_PERM_SLOTS   = 96
_PERM_SLOT_SZ = 138
_SELF_PERM_SLOTS = 48  # slots 0-47 = self, 48-95 = opponent
# Per-slot offsets: power(0), toughness(1), tapped(2), attacking(3), blocking(4),
#                   sickness(5), damage(6), controller_is_self(7), is_creature(8), is_land(9)
_GY_START_OBS    = 14842
_GY_SLOTS        = 128
_GY_SLOT_SZ      = 128
_GY_SELF_SLOTS   = 64   # slots 0-63 = self GY, 64-127 = opp GY

# Stack layout: [13282-14841] 12 slots x 130 floats
# Per slot: controller_is_self(1), card_id one-hot(128), is_spell(1)
_STACK_START   = 13282
_STACK_SLOTS   = 12
_STACK_SLOT_SZ = 130

# Hand layout: [31226-32505] 10 slots x 128 floats (imported _HAND_START from env.py)
_HAND_SLOTS = 10

# Mana color labels
_MANA_COLORS = ["W", "U", "B", "R", "G", "C"]


def _extract_interpretable(obs):
    """Extract human-readable features from a raw observation vector."""
    f = np.zeros(len(_INTERP_FEATURE_NAMES), dtype=np.float32)
    i = 0

    # Player stats (denormalize: life * 20, mana * 10)
    self_life = obs[0] * 20.0
    opp_life  = obs[9] * 20.0
    f[i] = self_life;     i += 1  # self_life
    f[i] = opp_life;      i += 1  # opp_life
    f[i] = self_life - opp_life; i += 1  # life_diff

    # Self mana (obs[2:8])
    for j in range(6):
        f[i] = obs[2 + j] * 10.0; i += 1
    f[i] = sum(obs[2 + j] * 10.0 for j in range(6)); i += 1  # total mana

    # Opp mana (obs[11:17])
    for j in range(6):
        f[i] = obs[11 + j] * 10.0; i += 1
    f[i] = sum(obs[11 + j] * 10.0 for j in range(6)); i += 1  # total mana

    # Hand size (count non-empty hand slots)
    hand_count = 0
    for slot in range(10):
        base = _HAND_START + slot * N_CARD_TYPES
        if np.max(obs[base:base + N_CARD_TYPES]) > 0.5:
            hand_count += 1
    f[i] = hand_count; i += 1

    # Permanent counts and stats
    self_creatures = self_lands = self_other = 0
    opp_creatures = opp_lands = opp_other = 0
    self_tapped_lands = opp_tapped_lands = 0
    self_attacking = self_blocking = 0
    opp_attacking = opp_blocking = 0
    self_power = self_toughness = 0.0
    opp_power = opp_toughness = 0.0

    for slot in range(_PERM_SLOTS):
        base = _PERM_START + slot * _PERM_SLOT_SZ
        power     = obs[base + 0] * 10.0
        toughness = obs[base + 1] * 10.0
        tapped    = obs[base + 2] > 0.5
        attacking = obs[base + 3] > 0.5
        blocking  = obs[base + 4] > 0.5
        is_creat  = obs[base + 8] > 0.5
        is_land   = obs[base + 9] > 0.5

        # Check if slot is occupied (any card one-hot active)
        card_vec = obs[base + 10 : base + 10 + N_CARD_TYPES]
        if np.max(card_vec) < 0.5:
            continue

        is_self = slot < _SELF_PERM_SLOTS
        if is_self:
            if is_creat:
                self_creatures += 1
                self_power += power
                self_toughness += toughness
                if attacking: self_attacking += 1
                if blocking:  self_blocking += 1
            elif is_land:
                self_lands += 1
                if tapped: self_tapped_lands += 1
            else:
                self_other += 1
        else:
            if is_creat:
                opp_creatures += 1
                opp_power += power
                opp_toughness += toughness
                if attacking: opp_attacking += 1
                if blocking:  opp_blocking += 1
            elif is_land:
                opp_lands += 1
                if tapped: opp_tapped_lands += 1
            else:
                opp_other += 1

    f[i] = self_creatures; i += 1
    f[i] = self_lands;     i += 1
    f[i] = self_other;     i += 1
    f[i] = opp_creatures;  i += 1
    f[i] = opp_lands;      i += 1
    f[i] = opp_other;      i += 1
    f[i] = self_creatures - opp_creatures; i += 1
    f[i] = self_lands - opp_lands;         i += 1
    f[i] = self_tapped_lands;  i += 1
    f[i] = opp_tapped_lands;   i += 1
    f[i] = self_attacking;     i += 1
    f[i] = self_blocking;      i += 1
    f[i] = opp_attacking;      i += 1
    f[i] = opp_blocking;       i += 1
    f[i] = self_power;        i += 1
    f[i] = opp_power;         i += 1
    f[i] = self_power - opp_power; i += 1
    f[i] = self_toughness;    i += 1
    f[i] = opp_toughness;     i += 1

    # Graveyard sizes
    self_gy = opp_gy = 0
    for slot in range(_GY_SLOTS):
        base = _GY_START_OBS + slot * _GY_SLOT_SZ
        if np.max(obs[base:base + _GY_SLOT_SZ]) > 0.5:
            if slot < 64:
                self_gy += 1
            else:
                opp_gy += 1
    f[i] = self_gy; i += 1
    f[i] = opp_gy;  i += 1

    # Stack size
    f[i] = obs[33] * 10.0; i += 1

    # Is active player
    f[i] = 1.0 if obs[31] > 0.5 else 0.0; i += 1

    # Step one-hot (obs[18:31])
    for j in range(13):
        f[i] = obs[18 + j]; i += 1

    return f


def _infer_deck(model_path):
    """Try to extract deck name from model filename like 'delver_delver_final.zip'."""
    basename = os.path.splitext(os.path.basename(model_path))[0]
    # Pattern: {model_deck}_{opp_deck}_{suffix}
    parts = basename.split("_")
    if len(parts) >= 2:
        return parts[0]
    return None


_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def _resolve_model_path(path):
    """Return path as-is if it exists, else try train/checkpoints/<path>."""
    if os.path.exists(path):
        return path
    candidate = os.path.join(_CHECKPOINTS_DIR, path)
    if os.path.exists(candidate):
        return candidate
    return path  # let the loader raise a meaningful error


def _load_model_and_env(args):
    """Load model, set up env with the right decks and opponent. Returns (model, env, opp_model_or_none)."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO

    binary = getattr(args, "binary", BINARY)

    model_path = _resolve_model_path(args.model)

    # Deck inference
    deck_a = getattr(args, "deck_a", None)
    deck_b = getattr(args, "deck_b", None)
    if not deck_a:
        inferred = _infer_deck(model_path)
        if inferred:
            deck_a = inferred
            print(f"Inferred model deck from filename: {deck_a}")
        else:
            print("Could not infer model deck from filename; use --deck-a", file=sys.stderr)
            sys.exit(1)
    if not deck_b:
        if args.opponent == "scripted":
            deck_b = deck_a  # mirror match by default
        else:
            opp_path = _resolve_model_path(args.opponent)
            inferred = _infer_deck(opp_path)
            if inferred:
                deck_b = inferred
                print(f"Inferred opponent deck from filename: {deck_b}")
            else:
                print("Could not infer opponent deck from filename; use --deck-b", file=sys.stderr)
                sys.exit(1)

    model = MaskablePPO.load(model_path)
    opp_model = None
    if args.opponent != "scripted":
        opp_model = MaskablePPO.load(_resolve_model_path(args.opponent))

    env = RoboMageEnv(binary_path=binary, deck_a=deck_a, deck_b=deck_b)
    return model, env, opp_model


def _get_policy_probs(model, obs, num_choices):
    """Get masked action probability distribution from the policy.

    Returns numpy array of shape (num_choices,) with probabilities for each
    legal action.
    """
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        try:
            features = model.policy.extract_features(obs_t, model.policy.features_extractor)
        except TypeError:
            features = model.policy.extract_features(obs_t)
        latent_pi, _ = model.policy.mlp_extractor(features)
        logits = model.policy.action_net(latent_pi)[0].clone()
        logits[num_choices:] = float('-inf')
        probs = torch.softmax(logits, dim=0).cpu().numpy()
    return probs[:num_choices].astype(np.float64)


def _collect_game_traces(model, env, opp_model, n_games, verbose=True):
    """Play n_games and collect per-step (obs, value, action) traces.

    Returns a list of game dicts:
      { "observations": [obs, ...],
        "values": [float, ...],
        "interp_features": [array, ...],
        "actions": [int, ...],       # model's chosen action index at each step
        "num_choices": [int, ...],   # number of legal actions at each step
        "result": float,  # +1 win, -1 loss from model perspective
        "model_is_a": bool }
    """
    import torch

    games = []
    for g in range(n_games):
        obs, _ = env.reset()
        model_is_a = bool(np.random.random() < 0.5)
        # Swap decks so model always plays its deck
        if not model_is_a:
            old_a, old_b = env._deck_a, env._deck_b
            env._deck_a, env._deck_b = old_b, old_a
            obs, _ = env.reset()
            env._deck_a, env._deck_b = old_a, old_b

        trace_obs = []
        trace_vals = []
        trace_interp = []
        trace_actions = []
        trace_num_choices = []
        trace_probs = []
        done = False
        total_reward = 0.0

        while not done:
            a_has_priority = obs[32] > 0.5
            model_has_priority = a_has_priority if model_is_a else not a_has_priority
            num_choices = env._num_choices

            if model_has_priority:
                # Record observation and value for model's decisions
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    value = model.policy.predict_values(obs_t).item()
                probs = _get_policy_probs(model, obs, num_choices)
                trace_obs.append(obs.copy())
                trace_vals.append(value)
                trace_interp.append(_extract_interpretable(obs))
                trace_num_choices.append(num_choices)
                trace_probs.append(probs)

                mask = np.zeros(MAX_ACTIONS, dtype=bool)
                mask[:num_choices] = True
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
                trace_actions.append(action)
            else:
                if opp_model is not None:
                    mask = np.zeros(MAX_ACTIONS, dtype=bool)
                    mask[:num_choices] = True
                    action, _ = opp_model.predict(obs, action_masks=mask, deterministic=True)
                    action = int(action)
                else:
                    action = scripted_action(obs, num_choices)

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        model_reward = total_reward if model_is_a else -total_reward
        games.append({
            "observations": trace_obs,
            "values": trace_vals,
            "interp_features": trace_interp,
            "actions": trace_actions,
            "num_choices": trace_num_choices,
            "action_probs": trace_probs,
            "result": model_reward,
            "model_is_a": model_is_a,
        })
        if verbose:
            result_str = "W" if model_reward > 0 else ("L" if model_reward < 0 else "D")
            print(f"  game {g + 1}/{n_games}: {len(trace_obs)} decisions, {result_str}",
                  flush=True)
    return games


def _decode_legal_actions(obs, num_choices, chosen_action):
    """Return a list of strings describing each legal action, marking the chosen one."""
    lines = []
    for i in range(num_choices):
        cat_raw = obs[STATE_SIZE + i]
        cat = int(round(cat_raw * ACTION_CATEGORY_MAX))
        cat_name = _CAT_NAMES.get(cat, str(cat))

        card_raw = obs[STATE_SIZE + MAX_ACTIONS + i]
        if card_raw < 0:
            card_str = ""
        else:
            cid = int(round(card_raw * N_CARD_TYPES))
            if 0 <= cid < len(_VOCAB_NAMES):
                card_str = f" ({_VOCAB_NAMES[cid]})"
            else:
                card_str = f" (card#{cid})"

        marker = " <-- chosen" if i == chosen_action else ""
        lines.append(f"    [{i}] {cat_name}{card_str}{marker}")
    return lines


def _add_sim_args(parser):
    """Add common simulation arguments to a subparser."""
    parser.add_argument("model", help="Path to model .zip")
    parser.add_argument("--opponent", required=True,
                        help="Opponent model .zip path, or 'scripted' for rule-based agent")
    parser.add_argument("--deck-a", default=None,
                        help="Model's deck (.dk stem). Inferred from model filename if omitted.")
    parser.add_argument("--deck-b", default=None,
                        help="Opponent's deck (.dk stem). Inferred from opponent filename if omitted; "
                             "defaults to --deck-a for scripted opponent.")
    parser.add_argument("--binary", default=BINARY, help="Path to robomage binary")


def cmd_shap(args):
    """SHAP analysis of the value function over simulated games."""
    import shap
    import torch

    n_games = args.n_games
    n_samples = args.n_samples
    n_background = args.n_background

    model, env, opp_model = _load_model_and_env(args)

    print(f"\nCollecting {n_games} game traces...")
    games = _collect_game_traces(model, env, opp_model, n_games)
    env.close()

    # Pool all model decision points
    all_interp = np.array([f for g in games for f in g["interp_features"]])
    all_obs    = np.array([o for g in games for o in g["observations"]])
    all_vals   = np.array([v for g in games for v in g["values"]])
    print(f"Collected {len(all_interp)} decision points across {n_games} games.")
    print(f"Value stats: mean={all_vals.mean():.3f}  std={all_vals.std():.3f}  "
          f"min={all_vals.min():.3f}  max={all_vals.max():.3f}")

    if len(all_interp) < n_background + 10:
        print("Not enough data points for SHAP analysis.", file=sys.stderr)
        return

    # Build a wrapper that maps interpretable features -> value prediction.
    # We need the raw observations paired with the interpretable features so we
    # can look up (or interpolate) the value. Since SHAP perturbs the
    # interpretable features, we use a nearest-neighbor approach: for each
    # perturbed sample, find the closest real observation in interpretable space
    # and return its value. This is valid because KernelExplainer works by
    # masking/replacing features with background values.
    #
    # But a cleaner approach: SHAP KernelExplainer on the actual value-function
    # with the interpretable features requires a function f(interp) -> value.
    # We fit a lightweight surrogate (gradient-boosted trees) on
    # (interp_features -> raw_value) to make SHAP tractable.

    from sklearn.ensemble import GradientBoostingRegressor

    print("\nFitting surrogate model on interpretable features...")
    surrogate = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8,
    )
    surrogate.fit(all_interp, all_vals)
    r2 = surrogate.score(all_interp, all_vals)
    print(f"Surrogate R^2 on training data: {r2:.4f}")
    if r2 < 0.5:
        print("Warning: surrogate fit is poor — SHAP results may be unreliable.", file=sys.stderr)

    # SHAP on the surrogate
    background_idx = np.random.choice(len(all_interp), size=min(n_background, len(all_interp)),
                                      replace=False)
    background = all_interp[background_idx]

    sample_idx = np.random.choice(len(all_interp), size=min(n_samples, len(all_interp)),
                                  replace=False)
    samples = all_interp[sample_idx]

    print(f"\nRunning SHAP KernelExplainer ({n_background} background, {len(samples)} samples)...")
    explainer = shap.KernelExplainer(surrogate.predict, background)
    shap_values = explainer.shap_values(samples)

    # Print feature importance (mean |SHAP|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(-mean_abs_shap)

    print(f"\n{'Feature':<25} {'Mean |SHAP|':>12} {'Std SHAP':>12} {'Surrogate Importance':>20}")
    print("-" * 72)
    feat_importance = surrogate.feature_importances_
    for rank, idx in enumerate(sorted_idx):
        name = _INTERP_FEATURE_NAMES[idx]
        print(f"  {name:<23} {mean_abs_shap[idx]:12.4f} {shap_values[:, idx].std():12.4f}"
              f" {feat_importance[idx]:20.4f}")

    # Try to plot
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        shap.summary_plot(shap_values, samples, feature_names=_INTERP_FEATURE_NAMES,
                          show=False)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nCould not display SHAP plot: {e}", file=sys.stderr)
        print("SHAP values printed above.", file=sys.stderr)

    _interactive_session({
        "games": games, "swing_data": None,
        "shap_values": shap_values, "shap_samples": samples,
        "model": None, "env": None, "opp_model": None, "args": args,
    })


_INTERP_STEP_NAMES = [
    "UNTAP", "UPKEEP", "DRAW", "FIRST_MAIN", "BEGIN_COMBAT",
    "DECLARE_ATK", "DECLARE_BLK", "FIRST_STRIKE", "COMBAT_DMG",
    "END_COMBAT", "SECOND_MAIN", "END", "CLEANUP",
]
_INTERP_STEP_OFFSET = 41  # index of step_untap in _INTERP_FEATURE_NAMES


def _step_name_from_feat(feat):
    """Decode step one-hot from interp feature vector."""
    best = -1
    best_val = -1.0
    for i, name in enumerate(_INTERP_STEP_NAMES):
        v = feat[_INTERP_STEP_OFFSET + i]
        if v > best_val:
            best_val = v
            best = i
    if best < 0:
        return "?"
    return _INTERP_STEP_NAMES[best]


def _replay_sim_game(game, game_idx):
    """Print a human-readable trace for a simulation game."""
    result_str = "WIN" if game["result"] > 0 else ("LOSS" if game["result"] < 0 else "DRAW")
    side = "A" if game["model_is_a"] else "B"
    print(f"\nGame {game_idx} — Model={side}, {result_str}, {len(game['values'])} model decisions")
    print()

    feats = game["interp_features"]
    vals = game["values"]
    for i, (feat, val) in enumerate(zip(feats, vals)):
        step = _step_name_from_feat(feat)
        mana_total = feat[9]
        print(f"  [{i:3d}] {step:<14}  "
              f"Life {feat[0]:.0f}/{feat[1]:.0f}  "
              f"Board {feat[18]:.0f}c+{feat[19]:.0f}l / {feat[21]:.0f}c+{feat[22]:.0f}l  "
              f"Mana {mana_total:.0f}  "
              f"GY {feat[37]:.0f}/{feat[38]:.0f}  "
              f"V={val:+.3f}")
    print()


def _sim_summary(games):
    """Print win/loss/draw summary for a list of sim games."""
    wins   = sum(1 for g in games if g["result"] > 0)
    losses = sum(1 for g in games if g["result"] < 0)
    draws  = sum(1 for g in games if g["result"] == 0)
    total  = len(games)
    if total == 0:
        print("  No games.")
        return
    lengths = [len(g["values"]) for g in games]
    arr = np.array(lengths)
    print(f"  {total} games: {wins}W / {losses}L / {draws}D  ({100 * wins / total:.1f}% win rate)")
    print(f"  Decisions/game: mean={arr.mean():.1f}  min={arr.min()}  max={arr.max()}")


def _compute_swings(games):
    """Return swing_data list sorted by magnitude."""
    swing_data = []
    for g_idx, game in enumerate(games):
        vals = game["values"]
        if len(vals) < 2:
            continue
        deltas = [abs(vals[i + 1] - vals[i]) for i in range(len(vals) - 1)]
        max_delta_idx = int(np.argmax(deltas))
        max_delta = deltas[max_delta_idx]
        swing_data.append({
            "game_idx": g_idx,
            "swing_step": max_delta_idx,
            "swing_magnitude": max_delta,
            "swing_from": vals[max_delta_idx],
            "swing_to": vals[max_delta_idx + 1],
            "result": game["result"],
            "n_decisions": len(vals),
            "model_is_a": game["model_is_a"],
        })
    swing_data.sort(key=lambda x: -x["swing_magnitude"])
    return swing_data


def _print_swing_table(top_swings):
    print(f"{'Game':<6} {'Step':<6} {'From':>8} {'To':>8} {'Delta':>8} {'Result':<8} {'Decisions':<10}")
    print("-" * 60)
    for s in top_swings:
        result_str = "WIN" if s["result"] > 0 else ("LOSS" if s["result"] < 0 else "DRAW")
        side = "A" if s["model_is_a"] else "B"
        print(f"  {s['game_idx']:<4}   {s['swing_step']:<4}   {s['swing_from']:>+7.3f} {s['swing_to']:>+7.3f}"
              f" {s['swing_to'] - s['swing_from']:>+7.3f}   {result_str:<6}({side}) {s['n_decisions']}")


def _decode_board_state(obs, value=None):
    """Print a detailed board state decoded from a raw observation vector."""
    # obs[32] = 1.0 if "self" (priority player) is Player A
    priority_is_a    = obs[32] > 0.5
    priority_is_active = obs[31] > 0.5
    self_label = "A" if priority_is_a else "B"
    opp_label  = "B" if priority_is_a else "A"

    self_life    = obs[0] * 20.0
    opp_life     = obs[9] * 20.0
    self_hand_ct = obs[1] * 10.0
    opp_hand_ct  = obs[10] * 10.0
    self_mana    = [obs[2 + j] * 10.0 for j in range(6)]
    opp_mana     = [obs[11 + j] * 10.0 for j in range(6)]
    stack_size   = int(round(obs[33] * 10.0))

    step_idx  = int(np.argmax(obs[18:31]))
    step_name = _INTERP_STEP_NAMES[step_idx] if step_idx < len(_INTERP_STEP_NAMES) else f"?{step_idx}"
    val_str   = f"  V={value:+.3f}" if value is not None else ""

    def mana_str(mana):
        parts = [f"{_MANA_COLORS[j]}:{mana[j]:.0f}" for j in range(6) if mana[j] > 0.4]
        return " ".join(parts) if parts else "—"

    def card_at(base):
        """Decode card name from one-hot starting at obs[base]."""
        vec = obs[base:base + N_CARD_TYPES]
        if np.max(vec) < 0.5:
            return None
        cid = int(np.argmax(vec))
        return _VOCAB_NAMES[cid] if cid < len(_VOCAB_NAMES) else f"card#{cid}"

    def perm_lines(slot_range):
        lines = []
        for slot in slot_range:
            base = _PERM_START + slot * _PERM_SLOT_SZ
            name = card_at(base + 10)
            if name is None:
                continue
            power     = obs[base + 0] * 10.0
            toughness = obs[base + 1] * 10.0
            tapped    = obs[base + 2] > 0.5
            attacking = obs[base + 3] > 0.5
            blocking  = obs[base + 4] > 0.5
            sickness  = obs[base + 5] > 0.5
            damage    = obs[base + 6] * 10.0
            is_creat  = obs[base + 8] > 0.5
            flags = []
            if tapped:       flags.append("tapped")
            if attacking:    flags.append("atk")
            if blocking:     flags.append("blk")
            if sickness:     flags.append("sick")
            if damage > 0.4: flags.append(f"dmg={damage:.0f}")
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            if is_creat:
                lines.append(f"    {name}  {power:.0f}/{toughness:.0f}{flag_str}")
            else:
                lines.append(f"    {name}{flag_str}")
        return lines

    print(f"Step: {step_name}  ({'active' if priority_is_active else 'non-active'} player has priority){val_str}")
    print(f"Stack: {stack_size} item(s)")

    if stack_size > 0:
        print("  Stack (top first):")
        for slot in range(_STACK_SLOTS):
            base = _STACK_START + slot * _STACK_SLOT_SZ
            name = card_at(base + 1)
            if name is None:
                continue
            ctrl_is_self = obs[base] > 0.5
            is_spell     = obs[base + _STACK_SLOT_SZ - 1] > 0.5
            ctrl_str     = self_label if ctrl_is_self else opp_label
            type_str     = "spell" if is_spell else "ability"
            print(f"    [{ctrl_str}] {name} ({type_str})")

    print()
    print(f"  [{self_label}] Priority player  "
          f"Life={self_life:.0f}  Hand={self_hand_ct:.0f}  Mana=[{mana_str(self_mana)}]")

    sp = perm_lines(range(_SELF_PERM_SLOTS))
    if sp:
        print(f"  Battlefield ({len(sp)}):")
        for ln in sp: print(ln)
    else:
        print("  Battlefield: empty")

    hand_cards = [card_at(_HAND_START + s * N_CARD_TYPES)
                  for s in range(_HAND_SLOTS)]
    hand_cards = [n for n in hand_cards if n is not None]
    if hand_cards:
        print(f"  Hand ({len(hand_cards)}): {', '.join(hand_cards)}")
    else:
        print("  Hand: empty")

    self_gy = [card_at(_GY_START_OBS + s * _GY_SLOT_SZ)
               for s in range(_GY_SELF_SLOTS)]
    self_gy = [n for n in self_gy if n is not None]
    if self_gy:
        print(f"  Graveyard ({len(self_gy)}): {', '.join(self_gy)}")
    else:
        print("  Graveyard: empty")

    print()
    print(f"  [{opp_label}] Opponent          "
          f"Life={opp_life:.0f}  Hand={opp_hand_ct:.0f}  Mana=[{mana_str(opp_mana)}]")

    op = perm_lines(range(_SELF_PERM_SLOTS, _PERM_SLOTS))
    if op:
        print(f"  Battlefield ({len(op)}):")
        for ln in op: print(ln)
    else:
        print("  Battlefield: empty")

    opp_gy = [card_at(_GY_START_OBS + s * _GY_SLOT_SZ)
              for s in range(_GY_SELF_SLOTS, _GY_SLOTS)]
    opp_gy = [n for n in opp_gy if n is not None]
    if opp_gy:
        print(f"  Graveyard ({len(opp_gy)}): {', '.join(opp_gy)}")
    else:
        print("  Graveyard: empty")
    print()


def _interactive_session(ctx):
    """Interactive REPL for inspecting simulation results.

    ctx keys: games, swing_data, shap_values, shap_samples,
              model, env, opp_model, args
    """
    try:
        import readline
        readline.set_history_length(500)
    except ImportError:
        pass

    games = ctx["games"]
    args  = ctx.get("args")

    def _banner():
        can_run = ctx.get("env") is not None
        print("\n" + "=" * 60)
        print(f"Interactive session — {len(games)} games in memory.")
        cmds = ["list", "replay <N>", "boardstate <N> <step>", "summary",
                "swings [N]", "shap", "regret [N]", "entropy", "consistency [N]",
                "calibration", "turning", "clusters",
                "chart <N>", "chart swings [N]", "chart shap",
                "chart calibration", "chart turning", "chart clusters"]
        if can_run:
            cmds.append("run <N>")
        cmds += ["help", "quit"]
        print("Commands: " + ", ".join(cmds))
        print("=" * 60)

    _banner()

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd in ("help", "?", "h"):
            can_run = ctx.get("env") is not None
            print("  list / games              — list all games with result/decisions/side")
            print("  replay <N>                — print board-state trace for game N")
            print("  boardstate <N> [step]     — full board + decision at step in game N (default 0)")
            print("  bs <N> [step]             — alias; enters GDB-style stepping mode")
            print("  summary                   — win/loss/draw stats for all simulated games")
            print("  swings [N]                — show top N value-function swings (default 10)")
            print("  shap [n_bg N] [n_smp N]   — run SHAP analysis on collected game data")
            print("  regret [N]                — policy regret analysis (top N high-regret decisions)")
            print("  entropy                   — policy entropy by phase and board state")
            print("  consistency [N]           — find similar states with different actions (top N pairs)")
            print("  calibration               — V(s) at game start vs actual win rate (is model biased?)")
            print("  turning                   — find the 'point of no return' in each game")
            print("  clusters                  — classify games by V(s) curve shape (archetypes)")
            print("  chart <N>                 — value curve plot for game N")
            print("  chart swings [N]          — value curve plots for top N swing games")
            print("  chart shap                — SHAP summary plot (requires shap run first)")
            print("  chart calibration         — calibration curve plot")
            print("  chart turning             — turning point distribution plot")
            print("  chart clusters            — overlay V(s) curves by archetype")
            if can_run:
                print("  run <N>                   — simulate N more games and add to pool")
            print("  quit / exit               — leave interactive session")

        elif cmd in ("list", "games", "ls"):
            print(f"  {'Game':<6} {'Result':<8} {'Decisions':<12} {'Side'}")
            print(f"  {'-'*6} {'-'*8} {'-'*12} {'-'*4}")
            for i, g in enumerate(games):
                r = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
                side = "A" if g["model_is_a"] else "B"
                print(f"  {i:<6} {r:<8} {len(g['values']):<12} {side}")

        elif cmd == "summary":
            _sim_summary(games)

        elif cmd == "replay":
            if len(parts) < 2:
                print("  Usage: replay <game_index>")
                continue
            try:
                n = int(parts[1])
            except ValueError:
                print("  Expected an integer game index.")
                continue
            if n < 0 or n >= len(games):
                print(f"  Game index out of range. Valid range: 0–{len(games) - 1}")
                continue
            _replay_sim_game(games[n], n)

        elif cmd in ("boardstate", "bs"):
            if len(parts) < 2:
                print("  Usage: boardstate <game_index> [decision_step]")
                continue
            try:
                gn = int(parts[1])
                step = int(parts[2]) if len(parts) >= 3 else 0
            except ValueError:
                print("  Expected integer game_index and optional decision_step.")
                continue
            if gn < 0 or gn >= len(games):
                print(f"  Game index out of range. Valid range: 0–{len(games) - 1}")
                continue

            def _show_step(g, gn, step):
                n_obs = len(g["observations"])
                obs = g["observations"][step]
                val = g["values"][step] if step < len(g["values"]) else None
                result_str = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
                print(f"\nGame {gn} [{result_str}]  —  decision {step}/{n_obs - 1}")

                # Model's decision at this step
                has_action = "actions" in g and step < len(g["actions"])
                if has_action:
                    action_idx  = g["actions"][step]
                    num_ch      = g["num_choices"][step]
                    action_lines = _decode_legal_actions(obs, num_ch, action_idx)
                    print(f"  Legal actions ({num_ch}):")
                    for ln in action_lines:
                        print(ln)
                print()
                _decode_board_state(obs, value=val)

            g = games[gn]
            n_obs = len(g["observations"])
            if step < 0 or step >= n_obs:
                print(f"  Step out of range for game {gn}. Valid range: 0–{n_obs - 1}")
                continue
            _show_step(g, gn, step)

            # GDB-style stepping sub-loop
            last_step_cmd = "n"
            print("  Stepping mode: n/Enter=next  p=prev  g <N>=go to step  q=quit stepping")
            while True:
                try:
                    raw = input(f"(g{gn}:{step}) ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                sc = raw.lower() if raw else last_step_cmd
                sp2 = sc.split()
                scmd = sp2[0] if sp2 else last_step_cmd

                if scmd in ("n", "next", ""):
                    last_step_cmd = "n"
                    if step < n_obs - 1:
                        step += 1
                        _show_step(g, gn, step)
                    else:
                        print("  End of game.")
                elif scmd in ("p", "prev", "previous", "b", "back"):
                    last_step_cmd = "p"
                    if step > 0:
                        step -= 1
                        _show_step(g, gn, step)
                    else:
                        print("  Beginning of game.")
                elif scmd == "g":
                    if len(sp2) < 2:
                        print("  Usage: g <step>")
                        continue
                    try:
                        target = int(sp2[1])
                    except ValueError:
                        print("  Expected an integer step.")
                        continue
                    if target < 0 or target >= n_obs:
                        print(f"  Step out of range. Valid range: 0–{n_obs - 1}")
                        continue
                    step = target
                    _show_step(g, gn, step)
                elif scmd in ("q", "quit", "exit"):
                    break
                else:
                    print("  n/Enter=next  p=prev  g <N>=go to step  q=quit stepping")

        elif cmd == "swings":
            top_n = 10
            if len(parts) >= 2:
                try:
                    top_n = int(parts[1])
                except ValueError:
                    pass
            if ctx["swing_data"] is None:
                print("  Computing value swings...")
                ctx["swing_data"] = _compute_swings(games)
            top = ctx["swing_data"][:top_n]
            print(f"\nTop {min(top_n, len(top))} value swings:")
            _print_swing_table(top)

        elif cmd == "shap":
            n_background = 50
            n_samples    = 200
            for j in range(1, len(parts) - 1, 2):
                if parts[j] == "n_bg":
                    try: n_background = int(parts[j + 1])
                    except ValueError: pass
                elif parts[j] == "n_smp":
                    try: n_samples = int(parts[j + 1])
                    except ValueError: pass
            if args is not None:
                n_background = getattr(args, "n_background", n_background)
                n_samples    = getattr(args, "n_samples", n_samples)
            try:
                import shap
                from sklearn.ensemble import GradientBoostingRegressor
                all_interp = np.array([f for g in games for f in g["interp_features"]])
                all_vals   = np.array([v for g in games for v in g["values"]])
                print(f"\nFitting surrogate on {len(all_interp)} points...")
                surrogate = GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8)
                surrogate.fit(all_interp, all_vals)
                r2 = surrogate.score(all_interp, all_vals)
                print(f"Surrogate R^2: {r2:.4f}")
                bg_idx  = np.random.choice(len(all_interp),
                                           size=min(n_background, len(all_interp)), replace=False)
                smp_idx = np.random.choice(len(all_interp),
                                           size=min(n_samples, len(all_interp)), replace=False)
                print(f"Running SHAP ({n_background} background, {len(smp_idx)} samples)...")
                explainer  = shap.KernelExplainer(surrogate.predict, all_interp[bg_idx])
                shap_vals  = explainer.shap_values(all_interp[smp_idx])
                ctx["shap_values"]  = shap_vals
                ctx["shap_samples"] = all_interp[smp_idx]
                mean_abs = np.abs(shap_vals).mean(axis=0)
                sidx = np.argsort(-mean_abs)
                print(f"\n{'Feature':<25} {'Mean |SHAP|':>12}")
                print("-" * 40)
                for idx in sidx:
                    print(f"  {_INTERP_FEATURE_NAMES[idx]:<23} {mean_abs[idx]:12.4f}")
            except ImportError as e:
                print(f"  Missing dependency: {e}")
            except Exception as e:
                print(f"  Error running SHAP: {e}")

        elif cmd == "chart":
            sub = parts[1].lower() if len(parts) >= 2 else ""
            try:
                import matplotlib
                matplotlib.use("TkAgg")
                import matplotlib.pyplot as plt
            except Exception as e:
                print(f"  matplotlib unavailable: {e}")
                continue

            if sub == "shap":
                if ctx["shap_values"] is None or ctx["shap_samples"] is None:
                    print("  Run 'shap' first to generate SHAP values.")
                    continue
                try:
                    import shap
                    shap.summary_plot(ctx["shap_values"], ctx["shap_samples"],
                                      feature_names=_INTERP_FEATURE_NAMES, show=False)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"  SHAP plot error: {e}")

            elif sub == "swings":
                top_n = 5
                if len(parts) >= 3:
                    try: top_n = int(parts[2])
                    except ValueError: pass
                if ctx["swing_data"] is None:
                    ctx["swing_data"] = _compute_swings(games)
                top = ctx["swing_data"][:top_n]
                if not top:
                    print("  No swing data.")
                    continue
                fig, axes = plt.subplots(len(top), 1, figsize=(10, 3 * len(top)), squeeze=False)
                for i, s in enumerate(top):
                    ax = games[s["game_idx"]]
                    vals = ax["values"]
                    result_str = "WIN" if ax["result"] > 0 else ("LOSS" if ax["result"] < 0 else "DRAW")
                    a = axes[i, 0]
                    a.plot(vals, color="steelblue", linewidth=1.2)
                    a.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                    a.axvline(s["swing_step"], color="red", linewidth=1, linestyle="--",
                              label=f"swing ({s['swing_to'] - s['swing_from']:+.2f})")
                    a.set_ylabel("V(s)")
                    a.set_title(f"Game {s['game_idx']} ({result_str})")
                    a.legend(loc="upper right", fontsize=8)
                    a.grid(True, alpha=0.3)
                axes[-1, 0].set_xlabel("Decision step")
                plt.tight_layout()
                plt.show()

            elif sub == "calibration":
                cal = ctx.get("calibration_data")
                if cal is None:
                    print("  Computing calibration...")
                    cal = _analyze_calibration(games, verbose=False)
                    ctx["calibration_data"] = cal
                if not cal:
                    print("  No calibration data.")
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                mean_vs = [b["mean_v"] for b in cal]
                win_rates = [b["win_rate"] for b in cal]
                counts = [b["n"] for b in cal]
                labels = [b["label"] for b in cal]
                # Scatter with size proportional to count
                sizes = [max(40, min(300, c * 5)) for c in counts]
                ax.scatter(mean_vs, win_rates, s=sizes, c="steelblue",
                           alpha=0.7, edgecolors="navy", zorder=3)
                for i, lab in enumerate(labels):
                    ax.annotate(f"{lab}\n(n={counts[i]})",
                                (mean_vs[i], win_rates[i]),
                                textcoords="offset points", xytext=(8, 8),
                                fontsize=7)
                # Perfect calibration line: V(s) maps to win rate as (V+1)/2
                xs = np.linspace(-1, 1, 50)
                ax.plot(xs, (xs + 1) / 2, color="gray", linestyle="--",
                        linewidth=1, label="Perfect calibration", alpha=0.6)
                ax.set_xlabel("Mean V(s) at game start")
                ax.set_ylabel("Actual win rate")
                ax.set_title("Value Function Calibration")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-0.05, 1.05)
                plt.tight_layout()
                plt.show()

            elif sub == "turning":
                tp_data = ctx.get("turning_data")
                if tp_data is None:
                    print("  Computing turning points...")
                    tp_data = _analyze_turning_points(games, verbose=False)
                    ctx["turning_data"] = tp_data
                if not tp_data:
                    print("  No turning points found.")
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Histogram of turning point timing
                ax = axes[0]
                win_fracs = [t["frac"] for t in tp_data if t["result"] > 0]
                loss_fracs = [t["frac"] for t in tp_data if t["result"] < 0]
                bins_hist = np.linspace(0, 1, 15)
                if win_fracs:
                    ax.hist(win_fracs, bins=bins_hist, alpha=0.6,
                            color="green", label=f"Wins ({len(win_fracs)})")
                if loss_fracs:
                    ax.hist(loss_fracs, bins=bins_hist, alpha=0.6,
                            color="red", label=f"Losses ({len(loss_fracs)})")
                ax.set_xlabel("Fraction of game elapsed")
                ax.set_ylabel("Count")
                ax.set_title("When Turning Points Occur")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # V(s) curves for a few games with turning points marked
                ax = axes[1]
                n_show = min(8, len(tp_data))
                colors_win = plt.cm.Greens(np.linspace(0.4, 0.9, n_show))
                colors_loss = plt.cm.Reds(np.linspace(0.4, 0.9, n_show))
                ci_w = 0
                ci_l = 0
                for t in tp_data[:n_show]:
                    g = games[t["game_idx"]]
                    vals = g["values"]
                    won = t["result"] > 0
                    if won:
                        c = colors_win[ci_w % len(colors_win)]
                        ci_w += 1
                    else:
                        c = colors_loss[ci_l % len(colors_loss)]
                        ci_l += 1
                    xs = np.linspace(0, 1, len(vals))
                    ax.plot(xs, vals, color=c, alpha=0.5, linewidth=1)
                    ax.axvline(t["frac"], color=c, linestyle=":",
                               linewidth=0.8, alpha=0.6)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.set_xlabel("Fraction of game elapsed")
                ax.set_ylabel("V(s)")
                ax.set_title(f"V(s) Curves with Turning Points ({n_show} games)")
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            elif sub == "clusters":
                clust = ctx.get("cluster_data")
                if clust is None:
                    print("  Computing clusters...")
                    clust = _analyze_clusters(games, verbose=False)
                    ctx["cluster_data"] = clust
                archetype_colors = {
                    "early_lead_held": "green",
                    "slow_grind": "steelblue",
                    "comeback": "orange",
                    "lead_blown": "red",
                    "volatile": "purple",
                }
                nonempty = {k: v for k, v in clust.items() if v}
                if not nonempty:
                    print("  No cluster data.")
                    continue
                n_types = len(nonempty)
                fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4),
                                         squeeze=False)
                for col, (label, indices) in enumerate(nonempty.items()):
                    ax = axes[0, col]
                    n_plot = min(15, len(indices))
                    for i in indices[:n_plot]:
                        g = games[i]
                        vals = g["values"]
                        xs = np.linspace(0, 1, len(vals))
                        won = g["result"] > 0
                        ax.plot(xs, vals, color=archetype_colors.get(label, "gray"),
                                alpha=0.3, linewidth=1)
                    # Plot mean curve
                    if indices:
                        max_len = max(len(games[i]["values"]) for i in indices)
                        interp_vals = []
                        for i in indices:
                            v = games[i]["values"]
                            xs_orig = np.linspace(0, 1, len(v))
                            xs_new = np.linspace(0, 1, max_len)
                            interp_vals.append(np.interp(xs_new, xs_orig, v))
                        mean_curve = np.mean(interp_vals, axis=0)
                        xs_mean = np.linspace(0, 1, max_len)
                        ax.plot(xs_mean, mean_curve,
                                color=archetype_colors.get(label, "gray"),
                                linewidth=2.5, label="mean")
                    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                    w = sum(1 for i in indices if games[i]["result"] > 0)
                    ax.set_title(f"{label}\n({len(indices)} games, "
                                 f"{w}/{len(indices)} wins)")
                    ax.set_xlabel("Game progress")
                    ax.set_ylabel("V(s)")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(-1.1, 1.1)
                plt.tight_layout()
                plt.show()

            else:
                # chart <N> — value curve for a single game
                try:
                    gn = int(sub)
                except ValueError:
                    print("  Usage: chart <game_index> | chart swings [N] | chart shap "
                          "| chart calibration | chart turning | chart clusters")
                    continue
                if gn < 0 or gn >= len(games):
                    print(f"  Game index out of range. Valid range: 0–{len(games) - 1}")
                    continue
                g = games[gn]
                vals = g["values"]
                result_str = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
                side = "A" if g["model_is_a"] else "B"
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(vals, color="steelblue", linewidth=1.2)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.set_xlabel("Decision step")
                ax.set_ylabel("V(s)")
                ax.set_title(f"Game {gn} — Model={side}, {result_str}")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

        elif cmd == "regret":
            top_n = 20
            if len(parts) >= 2:
                try: top_n = int(parts[1])
                except ValueError: pass
            has_probs = any(g.get("action_probs") for g in games)
            if not has_probs:
                print("  No action probability data. Re-collect games with a model to enable regret analysis.")
            else:
                _analyze_regret(games, top_n=top_n)

        elif cmd == "entropy":
            has_probs = any(g.get("action_probs") for g in games)
            if not has_probs:
                print("  No action probability data. Re-collect games with a model to enable entropy analysis.")
            else:
                _analyze_entropy(games)

        elif cmd == "consistency":
            top_n = 20
            if len(parts) >= 2:
                try: top_n = int(parts[1])
                except ValueError: pass
            _analyze_consistency(games, top_n=top_n)

        elif cmd == "calibration":
            ctx["calibration_data"] = _analyze_calibration(games)

        elif cmd == "turning":
            ctx["turning_data"] = _analyze_turning_points(games)

        elif cmd == "clusters":
            ctx["cluster_data"] = _analyze_clusters(games)

        elif cmd == "run":
            if ctx.get("env") is None or ctx.get("model") is None:
                print("  No live env available. Use the 'interactive' command to enable 'run'.")
                continue
            try:
                n = int(parts[1]) if len(parts) >= 2 else 10
            except ValueError:
                print("  Usage: run <N>")
                continue
            print(f"  Simulating {n} more games...")
            new_games = _collect_game_traces(ctx["model"], ctx["env"],
                                             ctx.get("opp_model"), n)
            games.extend(new_games)
            ctx["swing_data"] = None  # invalidate cached data
            ctx["calibration_data"] = None
            ctx["turning_data"] = None
            ctx["cluster_data"] = None
            print(f"  Pool now has {len(games)} games.")

        else:
            print(f"  Unknown command: {cmd!r}. Type 'help' for available commands.")


def cmd_value_swings(args):
    """Find games where the value function swung most dramatically."""
    import torch

    n_games = args.n_games
    top_n = args.top

    model, env, opp_model = _load_model_and_env(args)

    print(f"\nCollecting {n_games} game traces...")
    games = _collect_game_traces(model, env, opp_model, n_games)
    env.close()

    swing_data = _compute_swings(games)
    top_swings = swing_data[:top_n]

    print(f"\nTop {min(top_n, len(top_swings))} value function swings:\n")
    _print_swing_table(top_swings)

    # Detailed breakdowns of the top swings
    print(f"\n{'='*70}")
    print("Detailed breakdown of top swings:\n")

    for rank, s in enumerate(top_swings[:min(5, len(top_swings))]):
        g = games[s["game_idx"]]
        step = s["swing_step"]
        result_str = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
        side = "A" if g["model_is_a"] else "B"

        print(f"--- Game {s['game_idx']} (Model={side}, {result_str}, "
              f"{s['n_decisions']} decisions) ---")
        print(f"    Swing at step {step}: {s['swing_from']:+.3f} -> {s['swing_to']:+.3f} "
              f"(delta {s['swing_to'] - s['swing_from']:+.3f})")

        # Show board state before and after the swing
        for label, idx in [("Before", step), ("After", step + 1)]:
            if idx >= len(g["interp_features"]):
                continue
            feat = g["interp_features"][idx]
            print(f"    {label}: "
                  f"Life {feat[0]:.0f}/{feat[1]:.0f}  "
                  f"Creatures {feat[18]:.0f}v{feat[21]:.0f}  "
                  f"Lands {feat[19]:.0f}v{feat[22]:.0f}  "
                  f"Hand {feat[17]:.0f}  "
                  f"Power {feat[32]:.0f}v{feat[33]:.0f}  "
                  f"GY {feat[37]:.0f}/{feat[38]:.0f}")

        # Show value trajectory for this game
        vals = g["values"]
        n = len(vals)
        # Show a compressed trajectory: every Nth step + the swing point
        stride = max(1, n // 20)
        keypoints = set(range(0, n, stride))
        keypoints.add(step)
        if step + 1 < n:
            keypoints.add(step + 1)
        keypoints.add(n - 1)

        trajectory = []
        for i in sorted(keypoints):
            marker = " <-- SWING" if i == step else ""
            trajectory.append(f"      [{i:3d}] V={vals[i]:+.3f}{marker}")
        print("    Value trajectory:")
        for line in trajectory:
            print(line)
        print()

    # Aggregate statistics
    if swing_data:
        all_swings = [s["swing_magnitude"] for s in swing_data]
        win_swings = [s["swing_magnitude"] for s in swing_data if s["result"] > 0]
        loss_swings = [s["swing_magnitude"] for s in swing_data if s["result"] < 0]
        print(f"\nSwing statistics across {len(swing_data)} games:")
        print(f"  Overall: mean={np.mean(all_swings):.3f}  "
              f"median={np.median(all_swings):.3f}  max={np.max(all_swings):.3f}")
        if win_swings:
            print(f"  Wins:    mean={np.mean(win_swings):.3f}  "
                  f"median={np.median(win_swings):.3f}")
        if loss_swings:
            print(f"  Losses:  mean={np.mean(loss_swings):.3f}  "
                  f"median={np.median(loss_swings):.3f}")

    # Try to plot value curves for top swing games
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        n_plot = min(5, len(top_swings))
        fig, axes = plt.subplots(n_plot, 1, figsize=(10, 3 * n_plot), squeeze=False)
        for i, s in enumerate(top_swings[:n_plot]):
            ax = axes[i, 0]
            g = games[s["game_idx"]]
            vals = g["values"]
            result_str = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
            ax.plot(vals, color="steelblue", linewidth=1.2)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.axvline(s["swing_step"], color="red", linewidth=1, linestyle="--",
                       label=f"swing ({s['swing_to'] - s['swing_from']:+.2f})")
            ax.set_ylabel("V(s)")
            ax.set_title(f"Game {s['game_idx']} ({result_str})")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1, 0].set_xlabel("Decision step")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"\nCould not display plot: {e}", file=sys.stderr)

    _interactive_session({
        "games": games, "swing_data": swing_data,
        "shap_values": None, "shap_samples": None,
        "calibration_data": None, "turning_data": None, "cluster_data": None,
        "model": None, "env": None, "opp_model": None, "args": args,
    })


def _board_bucket_from_feat(feat):
    """Categorize board state from interpretable features into (life_bucket, board_bucket, timing_bucket)."""
    life_diff = feat[2]  # life_diff
    creature_diff = feat[24]  # creature_diff
    # Timing: use hand size + land count as a rough game-phase proxy
    self_lands = feat[19]
    if self_lands <= 2:
        timing = "early"
    elif self_lands <= 4:
        timing = "mid"
    else:
        timing = "late"

    if life_diff <= -5:
        life = "behind"
    elif life_diff >= 5:
        life = "ahead"
    else:
        life = "even"

    if creature_diff <= -1:
        board = "behind"
    elif creature_diff >= 1:
        board = "ahead"
    else:
        board = "even"

    return life, board, timing


def _analyze_regret(games, top_n=20, verbose=True):
    """Compute policy-based regret proxy from collected game traces.

    Regret proxy = 1 - P(chosen action). High values mean the model spread
    probability across alternatives — it was uncertain about its choice.
    Also computes margin = P(chosen) - P(second best).

    Returns list of regret entries sorted by descending regret.
    """
    entries = []
    for g_idx, game in enumerate(games):
        probs_list = game.get("action_probs", [])
        if not probs_list:
            continue
        for step, (probs, action, nc, obs, feat) in enumerate(zip(
                probs_list, game["actions"], game["num_choices"],
                game["observations"], game["interp_features"])):
            chosen_prob = probs[action]
            sorted_probs = np.sort(probs)[::-1]
            second_best = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            regret = 1.0 - chosen_prob
            margin = chosen_prob - second_best
            best_alt_idx = -1
            if nc > 1:
                alt_probs = probs.copy()
                alt_probs[action] = -1.0
                best_alt_idx = int(np.argmax(alt_probs))
            entries.append({
                "game_idx": g_idx, "step": step,
                "regret": regret, "margin": margin,
                "chosen_prob": chosen_prob, "chosen_action": action,
                "best_alt_idx": best_alt_idx, "best_alt_prob": second_best,
                "num_choices": nc, "obs": obs, "feat": feat,
                "result": game["result"],
            })

    entries.sort(key=lambda e: -e["regret"])

    if not entries:
        print("No decisions with action probabilities found.")
        return entries

    if not verbose:
        return entries

    n_decisions = len(entries)
    mean_regret = np.mean([e["regret"] for e in entries])
    mean_margin = np.mean([e["margin"] for e in entries])

    print(f"\nPolicy regret analysis — {len(games)} games, {n_decisions} model decisions")
    print(f"  NOTE: Regret proxy = 1 - P(chosen). High = model spread probability")
    print(f"        across alternatives. Margin = P(chosen) - P(2nd best).")
    print(f"\n  Overall: mean regret={mean_regret:.3f}  mean margin={mean_margin:.3f}")

    # By game outcome
    win_regrets = [e["regret"] for e in entries if e["result"] > 0]
    loss_regrets = [e["regret"] for e in entries if e["result"] < 0]
    if win_regrets and loss_regrets:
        print(f"\n  By outcome:")
        print(f"    Wins:   mean regret={np.mean(win_regrets):.3f}  "
              f"mean margin={np.mean([e['margin'] for e in entries if e['result'] > 0]):.3f}")
        print(f"    Losses: mean regret={np.mean(loss_regrets):.3f}  "
              f"mean margin={np.mean([e['margin'] for e in entries if e['result'] < 0]):.3f}")

    # By game phase
    phase_regrets = {}
    for e in entries:
        step_name = _step_name_from_feat(e["feat"])
        phase_regrets.setdefault(step_name, []).append(e["regret"])

    print(f"\n  By game phase:")
    print(f"    {'Phase':<14} {'Mean Regret':>12} {'Mean Margin':>12} {'N':>6}")
    phase_margins = {}
    for e in entries:
        step_name = _step_name_from_feat(e["feat"])
        phase_margins.setdefault(step_name, []).append(e["margin"])
    for phase in _INTERP_STEP_NAMES:
        if phase not in phase_regrets:
            continue
        regs = phase_regrets[phase]
        mars = phase_margins[phase]
        print(f"    {phase:<14} {np.mean(regs):12.3f} {np.mean(mars):12.3f} {len(regs):6d}")

    # By board state
    bucket_regrets = {}
    for e in entries:
        life, board, timing = _board_bucket_from_feat(e["feat"])
        key = f"{life}/{board}"
        bucket_regrets.setdefault(key, []).append(e["regret"])

    print(f"\n  By board state (life/creatures):")
    for key in sorted(bucket_regrets):
        regs = bucket_regrets[key]
        if len(regs) < 5:
            continue
        print(f"    {key:<16} regret={np.mean(regs):.3f}  (n={len(regs)})")

    # Top regret decisions
    top = entries[:min(top_n, len(entries))]
    print(f"\n  Top {len(top)} highest-regret decisions:")
    print(f"    {'Game':<6} {'Step':<6} {'Phase':<14} {'Regret':>7} {'Margin':>7} "
          f"{'Chosen':>7} {'2nd Best':>9} {'#Acts':>5} {'Result':<6}")
    print(f"    {'-'*6} {'-'*6} {'-'*14} {'-'*7} {'-'*7} {'-'*7} {'-'*9} {'-'*5} {'-'*6}")
    for e in top:
        phase = _step_name_from_feat(e["feat"])
        result_str = "W" if e["result"] > 0 else ("L" if e["result"] < 0 else "D")
        print(f"    {e['game_idx']:<6} {e['step']:<6} {phase:<14} "
              f"{e['regret']:7.3f} {e['margin']:7.3f} "
              f"{e['chosen_prob']:7.3f} {e['best_alt_prob']:9.3f} "
              f"{e['num_choices']:5d} {result_str:<6}")

    # Detailed board states for top 5
    print(f"\n  Detailed board states for top {min(5, len(top))} regret decisions:\n")
    for rank, e in enumerate(top[:5]):
        phase = _step_name_from_feat(e["feat"])
        result_str = "WIN" if e["result"] > 0 else ("LOSS" if e["result"] < 0 else "DRAW")
        print(f"  --- #{rank + 1}: Game {e['game_idx']} step {e['step']} "
              f"({result_str}, regret={e['regret']:.3f}) ---")
        action_lines = _decode_legal_actions(e["obs"], e["num_choices"], e["chosen_action"])
        # Annotate with probabilities
        probs = games[e["game_idx"]]["action_probs"][e["step"]]
        for i, line in enumerate(action_lines):
            prob = probs[i] if i < len(probs) else 0.0
            print(f"  {line}  P={prob:.3f}")
        print()
        _decode_board_state(e["obs"], value=games[e["game_idx"]]["values"][e["step"]])
        print()

    return entries


def _analyze_entropy(games, verbose=True):
    """Compute policy entropy at each decision point.

    H = -sum(p * ln(p)) for legal actions. Normalized entropy H_norm = H / ln(N)
    ranges from 0 (certain) to 1 (uniform).

    Returns list of (entropy, normalized_entropy, feat, step_name, result, game_idx, step) tuples.
    """
    records = []
    for g_idx, game in enumerate(games):
        probs_list = game.get("action_probs", [])
        if not probs_list:
            continue
        for step, (probs, nc, feat) in enumerate(zip(
                probs_list, game["num_choices"], game["interp_features"])):
            # Entropy: -sum(p * ln(p)), skip zero-probability actions
            p = probs[:nc]
            p_safe = p[p > 1e-10]
            entropy = -np.sum(p_safe * np.log(p_safe))
            max_entropy = np.log(nc) if nc > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            records.append({
                "entropy": entropy, "norm_entropy": norm_entropy,
                "feat": feat, "num_choices": nc,
                "result": game["result"], "game_idx": g_idx, "step": step,
            })

    if not records:
        print("No decisions with action probabilities found.")
        return records

    if not verbose:
        return records

    all_h = np.array([r["entropy"] for r in records])
    all_hn = np.array([r["norm_entropy"] for r in records])

    print(f"\nPolicy entropy analysis — {len(games)} games, {len(records)} decisions")
    print(f"  Raw entropy:  mean={all_h.mean():.3f}  std={all_h.std():.3f}  "
          f"min={all_h.min():.3f}  max={all_h.max():.3f}")
    print(f"  Norm entropy: mean={all_hn.mean():.3f}  std={all_hn.std():.3f}  "
          f"(0=certain, 1=uniform)")

    # By game outcome
    win_h = [r["norm_entropy"] for r in records if r["result"] > 0]
    loss_h = [r["norm_entropy"] for r in records if r["result"] < 0]
    if win_h and loss_h:
        print(f"\n  By outcome:")
        print(f"    Wins:   norm_H={np.mean(win_h):.3f} +/- {np.std(win_h):.3f}")
        print(f"    Losses: norm_H={np.mean(loss_h):.3f} +/- {np.std(loss_h):.3f}")
        if np.mean(loss_h) > np.mean(win_h) + 0.05:
            print(f"    ** Model is more uncertain in losing games — "
                  f"suggests confusion rather than deliberate unpredictability.")

    # By game phase
    print(f"\n  By game phase:")
    print(f"    {'Phase':<14} {'Mean H':>8} {'Norm H':>8} {'Std':>8} {'N':>6}")
    phase_data = {}
    for r in records:
        step_name = _step_name_from_feat(r["feat"])
        phase_data.setdefault(step_name, []).append(r)
    for phase in _INTERP_STEP_NAMES:
        if phase not in phase_data:
            continue
        recs = phase_data[phase]
        h_vals = [r["entropy"] for r in recs]
        hn_vals = [r["norm_entropy"] for r in recs]
        print(f"    {phase:<14} {np.mean(h_vals):8.3f} {np.mean(hn_vals):8.3f} "
              f"{np.std(hn_vals):8.3f} {len(recs):6d}")

    # By board state bucket
    print(f"\n  By board state (life / creatures / timing):")
    print(f"    {'Bucket':<28} {'Norm H':>8} {'Std':>8} {'N':>6}")
    bucket_data = {}
    for r in records:
        life, board, timing = _board_bucket_from_feat(r["feat"])
        key = f"{life:<7} {board:<7} {timing}"
        bucket_data.setdefault(key, []).append(r["norm_entropy"])
    for key in sorted(bucket_data):
        vals = bucket_data[key]
        if len(vals) < 5:
            continue
        flag = " **" if np.mean(vals) > all_hn.mean() + all_hn.std() else ""
        print(f"    {key:<28} {np.mean(vals):8.3f} {np.std(vals):8.3f} {len(vals):6d}{flag}")

    # Low entropy check (potential overfit)
    low_entropy_frac = np.mean(all_hn < 0.1)
    if low_entropy_frac > 0.8:
        print(f"\n  WARNING: {low_entropy_frac:.0%} of decisions have norm_H < 0.1 — "
              f"potential overfit to a narrow strategy.")

    # Phase × outcome breakdown for phases with interesting patterns
    print(f"\n  Phase × Outcome breakdown:")
    print(f"    {'Phase':<14} {'Win H':>8} {'Loss H':>8} {'Delta':>8}")
    for phase in _INTERP_STEP_NAMES:
        if phase not in phase_data:
            continue
        recs = phase_data[phase]
        w_h = [r["norm_entropy"] for r in recs if r["result"] > 0]
        l_h = [r["norm_entropy"] for r in recs if r["result"] < 0]
        if len(w_h) < 3 or len(l_h) < 3:
            continue
        delta = np.mean(l_h) - np.mean(w_h)
        marker = " *" if abs(delta) > 0.05 else ""
        print(f"    {phase:<14} {np.mean(w_h):8.3f} {np.mean(l_h):8.3f} {delta:+8.3f}{marker}")

    return records


def _analyze_consistency(games, top_n=20, verbose=True):
    """Find similar observations where the model chose different actions.

    Uses cosine similarity on _INTERP_FEATURE_NAMES vectors. Reports
    inconsistency rates for simple game states.

    Returns list of inconsistent pairs sorted by descending similarity.
    """
    # Collect all (interp_feat, action_category, game_idx, step) tuples
    points = []
    for g_idx, game in enumerate(games):
        for step, (feat, action, nc, obs) in enumerate(zip(
                game["interp_features"], game["actions"],
                game["num_choices"], game["observations"])):
            # Decode chosen action category
            cat_raw = obs[STATE_SIZE + action]
            cat = int(round(cat_raw * ACTION_CATEGORY_MAX))
            points.append({
                "feat": feat, "action": action, "cat": cat,
                "game_idx": g_idx, "step": step,
                "num_choices": nc, "obs": obs,
                "result": game["result"],
            })

    if len(points) < 10:
        print("Not enough decision points for consistency analysis.")
        return []

    # Build feature matrix and normalize for cosine similarity
    feat_mat = np.array([p["feat"] for p in points], dtype=np.float64)
    norms = np.linalg.norm(feat_mat, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    feat_norm = feat_mat / norms

    # For efficiency, sample if too many points
    max_compare = 5000
    if len(points) > max_compare:
        idx = np.random.choice(len(points), size=max_compare, replace=False)
        idx.sort()
        points_sub = [points[i] for i in idx]
        feat_sub = feat_norm[idx]
    else:
        points_sub = points
        feat_sub = feat_norm

    # Compute pairwise cosine similarity (dot product of normalized vectors)
    sim_matrix = feat_sub @ feat_sub.T

    # Find pairs with high similarity but different action categories
    pairs = []
    n = len(points_sub)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] < 0.95:
                continue
            if points_sub[i]["cat"] == points_sub[j]["cat"]:
                continue
            pairs.append({
                "similarity": sim_matrix[i, j],
                "i": points_sub[i], "j": points_sub[j],
            })

    pairs.sort(key=lambda p: -p["similarity"])

    if not verbose:
        return pairs

    n_high_sim = np.sum(sim_matrix > 0.95) // 2  # upper triangle
    n_same_action = 0
    n_diff_action = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] < 0.95:
                continue
            if points_sub[i]["cat"] == points_sub[j]["cat"]:
                n_same_action += 1
            else:
                n_diff_action += 1

    print(f"\nDecision consistency analysis — {len(games)} games, {len(points)} decisions")
    if len(points) > max_compare:
        print(f"  (sampled {max_compare} decisions for pairwise comparison)")
    print(f"\n  Pairs with cosine similarity > 0.95: {n_high_sim}")
    if n_high_sim > 0:
        consistency_rate = n_same_action / (n_same_action + n_diff_action) * 100
        print(f"    Same action category: {n_same_action} ({consistency_rate:.1f}%)")
        print(f"    Different action:     {n_diff_action} ({100 - consistency_rate:.1f}%)")
    else:
        print("    No highly similar state pairs found.")
        return pairs

    # Simple state inconsistency: few creatures, plenty of mana
    simple_mask = []
    for p in points_sub:
        f = p["feat"]
        total_creatures = f[18] + f[21]  # self + opp creatures
        total_mana = f[9]  # self total mana
        simple_mask.append(total_creatures <= 2 and total_mana >= 3)

    simple_idx = [i for i, m in enumerate(simple_mask) if m]
    if len(simple_idx) >= 10:
        simple_feat = feat_sub[simple_idx]
        simple_sim = simple_feat @ simple_feat.T
        simple_same = simple_diff = 0
        for i in range(len(simple_idx)):
            for j in range(i + 1, len(simple_idx)):
                if simple_sim[i, j] < 0.95:
                    continue
                if points_sub[simple_idx[i]]["cat"] == points_sub[simple_idx[j]]["cat"]:
                    simple_same += 1
                else:
                    simple_diff += 1
        simple_total = simple_same + simple_diff
        if simple_total > 0:
            simple_inconsistent = simple_diff / simple_total * 100
            flag = " ** RED FLAG" if simple_inconsistent > 20 else ""
            print(f"\n  Simple states (<=2 creatures, >=3 mana): "
                  f"{len(simple_idx)} decisions")
            print(f"    High-similarity pairs: {simple_total}")
            print(f"    Inconsistency rate: {simple_inconsistent:.1f}%{flag}")

    # Top inconsistent pairs
    top = pairs[:min(top_n, len(pairs))]
    if top:
        print(f"\n  Top {len(top)} most inconsistent pairs (high sim, different action):")
        print(f"    {'#':<4} {'Sim':>6} {'Game A':>7} {'Step A':>7} "
              f"{'Game B':>7} {'Step B':>7} {'Action A':<14} {'Action B':<14}")
        print(f"    {'-'*4} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*14} {'-'*14}")
        for rank, pair in enumerate(top):
            pi, pj = pair["i"], pair["j"]
            cat_a = _CAT_NAMES.get(pi["cat"], str(pi["cat"]))
            cat_b = _CAT_NAMES.get(pj["cat"], str(pj["cat"]))
            print(f"    {rank + 1:<4} {pair['similarity']:6.3f} "
                  f"{pi['game_idx']:>7} {pi['step']:>7} "
                  f"{pj['game_idx']:>7} {pj['step']:>7} "
                  f"{cat_a:<14} {cat_b:<14}")

        # Detailed comparison for top 3
        print(f"\n  Detailed comparison for top {min(3, len(top))} pairs:\n")
        for rank, pair in enumerate(top[:3]):
            pi, pj = pair["i"], pair["j"]
            print(f"  --- Pair #{rank + 1} (similarity={pair['similarity']:.4f}) ---")
            for label, p in [("A", pi), ("B", pj)]:
                cat_name = _CAT_NAMES.get(p["cat"], str(p["cat"]))
                f = p["feat"]
                result_str = "W" if p["result"] > 0 else ("L" if p["result"] < 0 else "D")
                print(f"    [{label}] Game {p['game_idx']} step {p['step']} ({result_str})  "
                      f"Action: {cat_name}")
                print(f"        Life {f[0]:.0f}/{f[1]:.0f}  "
                      f"Creatures {f[18]:.0f}v{f[21]:.0f}  "
                      f"Lands {f[19]:.0f}v{f[22]:.0f}  "
                      f"Hand {f[17]:.0f}  Mana {f[9]:.0f}  "
                      f"Phase: {_step_name_from_feat(f)}")
            # Show what features differ most
            diff = np.abs(pi["feat"] - pj["feat"])
            top_diff_idx = np.argsort(-diff)[:5]
            diffs_str = ", ".join(
                f"{_INTERP_FEATURE_NAMES[k]}({pi['feat'][k]:.1f}→{pj['feat'][k]:.1f})"
                for k in top_diff_idx if diff[k] > 0.01)
            if diffs_str:
                print(f"    Largest feature diffs: {diffs_str}")
            print()

    return pairs


def _analyze_calibration(games, verbose=True):
    """Check whether V(s) at game start predicts actual win rate.

    Bins games by their initial value estimate and compares against the
    actual win rate within each bin.  A well-calibrated value function has
    win-rate track V(s); systematic deviation indicates bias.

    Returns list of dicts: {lo, hi, mean_v, win_rate, n, wins, losses, draws}.
    """
    points = []
    for g in games:
        if not g["values"]:
            continue
        points.append((g["values"][0], g["result"]))

    if not points:
        if verbose:
            print("  No games with value data.")
        return []

    vs = np.array([p[0] for p in points])
    results = np.array([p[1] for p in points])

    edges = [-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, np.inf]
    labels = ["< -0.50", "-0.50…-0.25", "-0.25…0.00",
              "0.00…0.25", "0.25…0.50", "> 0.50"]
    bins = []
    for i in range(len(edges) - 1):
        mask = (vs > edges[i]) & (vs <= edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        w = int((results[mask] > 0).sum())
        l = int((results[mask] < 0).sum())
        d = int((results[mask] == 0).sum())
        mean_v = float(vs[mask].mean())
        wr = w / n
        bins.append({
            "lo": edges[i], "hi": edges[i + 1],
            "label": labels[i], "mean_v": mean_v,
            "win_rate": wr, "n": n, "wins": w, "losses": l, "draws": d,
        })

    if not verbose:
        return bins

    print(f"\nValue function calibration — {len(points)} games")
    print(f"  Initial V(s) ranges vs actual win rate:\n")
    print(f"    {'Bin':<16} {'Mean V':>8} {'Win Rate':>10} {'N':>5}  {'W/L/D'}")
    print(f"    {'-'*16} {'-'*8} {'-'*10} {'-'*5}  {'-'*10}")
    for b in bins:
        print(f"    {b['label']:<16} {b['mean_v']:>+8.3f} {b['win_rate']:>9.1%} "
              f"{b['n']:>5}  {b['wins']}/{b['losses']}/{b['draws']}")

    # Bias summary
    # Expected: V(s) ≈ (win_rate - 0.5) * 2 roughly, since V in [-1, 1]
    # Compare mean_v to (win_rate * 2 - 1)
    if len(bins) >= 2:
        total_bias = 0.0
        total_n = 0
        for b in bins:
            implied_v = b["win_rate"] * 2.0 - 1.0
            total_bias += (b["mean_v"] - implied_v) * b["n"]
            total_n += b["n"]
        avg_bias = total_bias / total_n if total_n > 0 else 0.0
        if avg_bias < -0.1:
            print(f"\n    Model is systematically PESSIMISTIC (avg bias {avg_bias:+.3f})")
            print(f"    May be playing too defensively.")
        elif avg_bias > 0.1:
            print(f"\n    Model is systematically OPTIMISTIC (avg bias {avg_bias:+.3f})")
            print(f"    May be overcommitting / underestimating risk.")
        else:
            print(f"\n    Model calibration looks reasonable (avg bias {avg_bias:+.3f}).")

    return bins


def _analyze_turning_points(games, verbose=True):
    """Find the 'point of no return' in each game.

    For each game, find the last decision step where V(s) permanently crossed
    from negative to positive (for wins) or positive to negative (for losses).
    This is the turning point — more meaningful than the max swing because
    swings can recover.

    Returns list of dicts with turning point info, one per game that has one.
    """
    turning_points = []
    for g_idx, game in enumerate(games):
        vals = game["values"]
        result = game["result"]
        if len(vals) < 3 or result == 0:
            continue

        # For wins, find last crossing from negative to positive that held
        # For losses, find last crossing from positive to negative that held
        won = result > 0
        target_sign = 1 if won else -1

        # Find the last step where value crossed to the target sign and stayed
        crossing_step = None
        for i in range(len(vals) - 1):
            before_sign = 1 if vals[i] >= 0 else -1
            after_sign = 1 if vals[i + 1] >= 0 else -1
            if before_sign != target_sign and after_sign == target_sign:
                # Check if it stays on the target side for the rest of the game
                stayed = all(
                    (v >= 0) == (target_sign == 1)
                    for v in vals[i + 1:]
                )
                if stayed:
                    crossing_step = i

        if crossing_step is None:
            continue

        feat = game["interp_features"][crossing_step] if crossing_step < len(game["interp_features"]) else None
        turning_points.append({
            "game_idx": g_idx,
            "step": crossing_step,
            "total_steps": len(vals),
            "frac": crossing_step / len(vals),
            "v_before": vals[crossing_step],
            "v_after": vals[crossing_step + 1],
            "result": result,
            "model_is_a": game["model_is_a"],
            "feat": feat,
        })

    if not verbose:
        return turning_points

    if not turning_points:
        print("\n  No turning points found (games may start and stay on one side).")
        return turning_points

    win_tps = [t for t in turning_points if t["result"] > 0]
    loss_tps = [t for t in turning_points if t["result"] < 0]
    all_fracs = [t["frac"] for t in turning_points]

    print(f"\nTurning point analysis — {len(games)} games, "
          f"{len(turning_points)} with identifiable turning points")
    print(f"  (A turning point is the last permanent zero-crossing of V(s))\n")

    print(f"  Games with turning point: {len(turning_points)}/{len(games)} "
          f"({100 * len(turning_points) / len(games):.0f}%)")
    print(f"    Wins:   {len(win_tps)}")
    print(f"    Losses: {len(loss_tps)}")

    print(f"\n  Timing (fraction of game elapsed at turning point):")
    print(f"    Overall: mean={np.mean(all_fracs):.2f}  "
          f"median={np.median(all_fracs):.2f}  "
          f"std={np.std(all_fracs):.2f}")
    if win_tps:
        wf = [t["frac"] for t in win_tps]
        print(f"    Wins:    mean={np.mean(wf):.2f}  median={np.median(wf):.2f}")
    if loss_tps:
        lf = [t["frac"] for t in loss_tps]
        print(f"    Losses:  mean={np.mean(lf):.2f}  median={np.median(lf):.2f}")

    # Board state at turning points
    if any(t["feat"] is not None for t in turning_points):
        print(f"\n  Board state at turning points:")
        phase_counts = {}
        board_counts = {}
        for t in turning_points:
            if t["feat"] is None:
                continue
            phase = _step_name_from_feat(t["feat"])
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            life, board, timing = _board_bucket_from_feat(t["feat"])
            key = f"{timing}/{life}/{board}"
            board_counts[key] = board_counts.get(key, 0) + 1

        print(f"    By phase:")
        for phase in _INTERP_STEP_NAMES:
            if phase in phase_counts:
                pct = 100 * phase_counts[phase] / len(turning_points)
                print(f"      {phase:<14} {phase_counts[phase]:>4} ({pct:5.1f}%)")

        print(f"    By board state (timing/life/creatures):")
        for key in sorted(board_counts, key=lambda k: -board_counts[k]):
            pct = 100 * board_counts[key] / len(turning_points)
            print(f"      {key:<24} {board_counts[key]:>4} ({pct:5.1f}%)")

    # Show a few example turning points
    print(f"\n  Example turning points (first {min(8, len(turning_points))}):")
    print(f"    {'Game':<6} {'Step':<6} {'Of':<6} {'Frac':>6} "
          f"{'V before':>9} {'V after':>9} {'Result':<6}")
    print(f"    {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*6}")
    for t in turning_points[:8]:
        r = "WIN" if t["result"] > 0 else "LOSS"
        print(f"    {t['game_idx']:<6} {t['step']:<6} {t['total_steps']:<6} "
              f"{t['frac']:>5.0%} {t['v_before']:>+9.3f} {t['v_after']:>+9.3f} {r:<6}")

    return turning_points


def _analyze_clusters(games, verbose=True):
    """Cluster games by V(s) curve shape using simple shape descriptors.

    Classifies games into archetypes:
      - "early_lead_held":  V(s) starts positive and stays mostly positive
      - "slow_grind":       V(s) starts near zero, gradually moves toward outcome
      - "comeback":         V(s) spends significant time negative then finishes positive
      - "lead_blown":       V(s) spends significant time positive then finishes negative
      - "volatile":         V(s) crosses zero 3+ times

    Returns dict: {archetype_name: [game_indices]}.
    """
    archetypes = {
        "early_lead_held": [],
        "slow_grind": [],
        "comeback": [],
        "lead_blown": [],
        "volatile": [],
    }
    game_labels = []  # (game_idx, label) for all classified games

    for g_idx, game in enumerate(games):
        vals = game["values"]
        result = game["result"]
        if len(vals) < 3:
            game_labels.append((g_idx, "too_short"))
            continue

        arr = np.array(vals)
        won = result > 0
        lost = result < 0

        # Count zero crossings
        signs = np.sign(arr)
        signs[signs == 0] = 1  # treat zero as positive
        crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

        # Fraction of game spent positive/negative
        frac_positive = np.mean(arr > 0)
        frac_negative = np.mean(arr < 0)

        # Early game tendency (first quarter)
        q1 = max(1, len(arr) // 4)
        early_mean = arr[:q1].mean()

        if crossings >= 3:
            label = "volatile"
        elif won and early_mean < -0.05 and frac_negative > 0.3:
            label = "comeback"
        elif lost and early_mean > 0.05 and frac_positive > 0.3:
            label = "lead_blown"
        elif won and early_mean > 0.05 and frac_positive > 0.6:
            label = "early_lead_held"
        elif lost and early_mean < -0.05 and frac_negative > 0.6:
            label = "early_lead_held"  # opponent held lead from start
        elif abs(early_mean) <= 0.15:
            label = "slow_grind"
        else:
            label = "slow_grind"  # fallback

        archetypes[label].append(g_idx)
        game_labels.append((g_idx, label))

    if not verbose:
        return archetypes

    print(f"\nValue trajectory clustering — {len(games)} games\n")
    print(f"  {'Archetype':<20} {'Count':>6} {'Wins':>6} {'Losses':>6} "
          f"{'Win%':>6} {'Avg Length':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")

    for label in ["early_lead_held", "slow_grind", "comeback", "lead_blown", "volatile"]:
        indices = archetypes[label]
        if not indices:
            continue
        n = len(indices)
        w = sum(1 for i in indices if games[i]["result"] > 0)
        l = sum(1 for i in indices if games[i]["result"] < 0)
        avg_len = np.mean([len(games[i]["values"]) for i in indices])
        wr = w / n * 100 if n > 0 else 0
        print(f"  {label:<20} {n:>6} {w:>6} {l:>6} {wr:>5.1f}% {avg_len:>10.1f}")

    # Per-archetype description and notable features
    descs = {
        "early_lead_held": "Model had an early advantage and maintained it",
        "slow_grind": "Close game that gradually resolved toward the outcome",
        "comeback": "Model was behind but recovered to win",
        "lead_blown": "Model had an early lead but lost it",
        "volatile": "Highly uncertain game with 3+ momentum shifts",
    }
    print()
    for label, desc in descs.items():
        indices = archetypes[label]
        if not indices:
            continue
        print(f"  {label}: {desc}")

        # Value stats
        all_vals = [np.array(games[i]["values"]) for i in indices]
        mean_start = np.mean([v[0] for v in all_vals])
        mean_end = np.mean([v[-1] for v in all_vals])
        mean_crossings = np.mean([
            int(np.sum(np.abs(np.diff(np.sign(np.where(v == 0, 1, v)))) > 0))
            for v in all_vals
        ])
        print(f"    Avg start V: {mean_start:+.3f}  Avg end V: {mean_end:+.3f}  "
              f"Avg zero-crossings: {mean_crossings:.1f}")

        # Board state at midpoint
        mid_feats = []
        for i in indices:
            g = games[i]
            mid = len(g["interp_features"]) // 2
            if mid < len(g["interp_features"]):
                mid_feats.append(g["interp_features"][mid])
        if mid_feats:
            mf = np.mean(mid_feats, axis=0)
            print(f"    Avg midgame: Life {mf[0]:.0f}/{mf[1]:.0f}  "
                  f"Creatures {mf[18]:.1f}v{mf[21]:.1f}  "
                  f"Lands {mf[19]:.1f}v{mf[22]:.1f}  "
                  f"Hand {mf[17]:.1f}")
        print()

    return archetypes


def cmd_regret(args):
    """Action regret / counterfactual analysis using policy distribution."""
    model, env, opp_model = _load_model_and_env(args)

    print(f"\nCollecting {args.n_games} game traces...")
    games = _collect_game_traces(model, env, opp_model, args.n_games)
    env.close()

    _analyze_regret(games, top_n=args.top)

    _interactive_session({
        "games": games, "swing_data": None,
        "shap_values": None, "shap_samples": None,
        "model": None, "env": None, "opp_model": None, "args": args,
    })


def cmd_entropy(args):
    """Policy entropy analysis over game phases and board states."""
    model, env, opp_model = _load_model_and_env(args)

    print(f"\nCollecting {args.n_games} game traces...")
    games = _collect_game_traces(model, env, opp_model, args.n_games)
    env.close()

    records = _analyze_entropy(games)

    # Try to plot
    try:
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        # Entropy by phase
        phase_data = {}
        for r in records:
            step_name = _step_name_from_feat(r["feat"])
            phase_data.setdefault(step_name, []).append(r["norm_entropy"])

        phases_present = [p for p in _INTERP_STEP_NAMES if p in phase_data]
        if phases_present:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Box plot by phase
            ax = axes[0]
            bp_data = [phase_data[p] for p in phases_present]
            ax.boxplot(bp_data, labels=phases_present, vert=True)
            ax.set_xticklabels(phases_present, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Normalized Entropy")
            ax.set_title("Policy Entropy by Game Phase")
            ax.grid(True, alpha=0.3)

            # Entropy by outcome over time (game progress)
            ax = axes[1]
            win_records = [r for r in records if r["result"] > 0]
            loss_records = [r for r in records if r["result"] < 0]
            for label, recs, color in [("Wins", win_records, "steelblue"),
                                       ("Losses", loss_records, "firebrick")]:
                if not recs:
                    continue
                # Bin by step index within game
                max_step = max(r["step"] for r in recs)
                if max_step < 5:
                    continue
                n_bins = min(20, max_step)
                bin_edges = np.linspace(0, max_step + 1, n_bins + 1)
                bin_means = []
                bin_centers = []
                for b in range(n_bins):
                    in_bin = [r["norm_entropy"] for r in recs
                              if bin_edges[b] <= r["step"] < bin_edges[b + 1]]
                    if in_bin:
                        bin_means.append(np.mean(in_bin))
                        bin_centers.append((bin_edges[b] + bin_edges[b + 1]) / 2)
                ax.plot(bin_centers, bin_means, color=color, label=label, linewidth=1.5)
            ax.set_xlabel("Decision Step in Game")
            ax.set_ylabel("Mean Normalized Entropy")
            ax.set_title("Entropy Over Game Progress")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"\nCould not display plot: {e}", file=sys.stderr)

    _interactive_session({
        "games": games, "swing_data": None,
        "shap_values": None, "shap_samples": None,
        "model": None, "env": None, "opp_model": None, "args": args,
    })


def cmd_consistency(args):
    """Decision consistency analysis for similar game states."""
    model, env, opp_model = _load_model_and_env(args)

    print(f"\nCollecting {args.n_games} game traces...")
    games = _collect_game_traces(model, env, opp_model, args.n_games)
    env.close()

    _analyze_consistency(games, top_n=args.top)

    _interactive_session({
        "games": games, "swing_data": None,
        "shap_values": None, "shap_samples": None,
        "model": None, "env": None, "opp_model": None, "args": args,
    })


def cmd_interactive(args):
    """Load model, simulate games, then enter the interactive session."""
    model, env, opp_model = _load_model_and_env(args)

    games = []
    if args.n_games > 0:
        print(f"\nSimulating {args.n_games} games...")
        games = _collect_game_traces(model, env, opp_model, args.n_games)

    ctx = {
        "games": games,
        "swing_data": None,
        "shap_values": None,
        "shap_samples": None,
        "calibration_data": None,
        "turning_data": None,
        "cluster_data": None,
        "model": model,
        "env": env,
        "opp_model": opp_model,
        "args": args,
    }

    try:
        _interactive_session(ctx)
    finally:
        env.close()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze .rmrec recording files")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("summary", help="Session summary and statistics")
    p.add_argument("file")

    p = sub.add_parser("winrate", help="Win rate plot over time")
    p.add_argument("file")

    p = sub.add_parser("actions", help="Action category heatmap by game step")
    p.add_argument("file")

    p = sub.add_parser("cards", help="Card usage bar chart")
    p.add_argument("file")

    p = sub.add_parser("replay", help="Replay a single game")
    p.add_argument("file")
    p.add_argument("--game", type=int, required=True, help="Game ID to replay")

    p = sub.add_parser("compare", help="Compare win rates from two sessions")
    p.add_argument("file")
    p.add_argument("file2")

    p = sub.add_parser("wl-split", help="Action distribution split by win/loss")
    p.add_argument("file")

    p = sub.add_parser("cast-timing", help="Per-card cast timing and state by outcome")
    p.add_argument("file")

    p = sub.add_parser("choice-rates", help="P(chose X | X legal) by board state")
    p.add_argument("file")

    p = sub.add_parser("shap", help="SHAP analysis of value function over simulated games")
    _add_sim_args(p)
    p.add_argument("--n-games", type=int, default=50, help="Number of games to simulate (default: 50)")
    p.add_argument("--n-samples", type=int, default=200, help="SHAP sample count (default: 200)")
    p.add_argument("--n-background", type=int, default=50, help="SHAP background size (default: 50)")

    p = sub.add_parser("value-swings", help="Find games with largest value function swings")
    _add_sim_args(p)
    p.add_argument("--n-games", type=int, default=50, help="Number of games to simulate (default: 50)")
    p.add_argument("--top", type=int, default=10, help="Show top N swings (default: 10)")

    p = sub.add_parser("regret",
                       help="Action regret analysis using policy distribution")
    _add_sim_args(p)
    p.add_argument("--n-games", type=int, default=50, help="Number of games to simulate (default: 50)")
    p.add_argument("--top", type=int, default=20, help="Show top N high-regret decisions (default: 20)")

    p = sub.add_parser("entropy",
                       help="Policy entropy by game phase and board state")
    _add_sim_args(p)
    p.add_argument("--n-games", type=int, default=50, help="Number of games to simulate (default: 50)")

    p = sub.add_parser("consistency",
                       help="Decision consistency for similar game states")
    _add_sim_args(p)
    p.add_argument("--n-games", type=int, default=50, help="Number of games to simulate (default: 50)")
    p.add_argument("--top", type=int, default=20, help="Show top N inconsistent pairs (default: 20)")

    p = sub.add_parser("interactive",
                       help="Interactive session: simulate games then inspect replays, "
                            "board states, value charts, SHAP, and more")
    _add_sim_args(p)
    p.add_argument("--n-games", type=int, default=20,
                   help="Games to pre-simulate before entering session (default: 20; 0 = skip)")
    p.add_argument("--n-samples", type=int, default=200, help="SHAP sample count (default: 200)")
    p.add_argument("--n-background", type=int, default=50, help="SHAP background size (default: 50)")

    args = parser.parse_args()
    {
        "summary": cmd_summary,
        "winrate": cmd_winrate,
        "actions": cmd_actions,
        "cards": cmd_cards,
        "replay": cmd_replay,
        "compare": cmd_compare,
        "wl-split": cmd_wl_split,
        "cast-timing": cmd_cast_timing,
        "choice-rates": cmd_choice_rates,
        "shap": cmd_shap,
        "value-swings": cmd_value_swings,
        "regret": cmd_regret,
        "entropy": cmd_entropy,
        "consistency": cmd_consistency,
        "interactive": cmd_interactive,
    }[args.command](args)


if __name__ == "__main__":
    main()
