"""
Analyze .rmrec recording files produced by train.py --record.

Usage:
    python analysis.py summary <file.rmrec>
    python analysis.py winrate <file.rmrec>
    python analysis.py actions <file.rmrec>
    python analysis.py cards <file.rmrec>
    python analysis.py replay <file.rmrec> --game <N>
    python analysis.py compare <file1.rmrec> <file2.rmrec>
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
from env import ACTION_CATEGORY_MAX


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

    args = parser.parse_args()
    {
        "summary": cmd_summary,
        "winrate": cmd_winrate,
        "actions": cmd_actions,
        "cards": cmd_cards,
        "replay": cmd_replay,
        "compare": cmd_compare,
    }[args.command](args)


if __name__ == "__main__":
    main()
