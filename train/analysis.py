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
    }[args.command](args)


if __name__ == "__main__":
    main()
