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
    python analysis.py shap <model.zip> [--n-games 50 --n-samples 200 --n-background 50]
    python analysis.py value-swings <model.zip> [--n-games 50 --top 10]
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
_GY_START_OBS = 14842
_GY_SLOTS     = 128
_GY_SLOT_SZ   = 128


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


def _load_model_and_env(args):
    """Load model, set up env with the right decks and opponent. Returns (model, env, opp_model_or_none)."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO

    binary = getattr(args, "binary", BINARY)

    # Deck inference
    deck_a = getattr(args, "deck_a", None)
    deck_b = getattr(args, "deck_b", None)
    if not deck_a:
        inferred = _infer_deck(args.model)
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
            inferred = _infer_deck(args.opponent)
            if inferred:
                deck_b = inferred
                print(f"Inferred opponent deck from filename: {deck_b}")
            else:
                print("Could not infer opponent deck from filename; use --deck-b", file=sys.stderr)
                sys.exit(1)

    model = MaskablePPO.load(args.model)
    opp_model = None
    if args.opponent != "scripted":
        opp_model = MaskablePPO.load(args.opponent)

    env = RoboMageEnv(binary_path=binary, deck_a=deck_a, deck_b=deck_b)
    return model, env, opp_model


def _collect_game_traces(model, env, opp_model, n_games, verbose=True):
    """Play n_games and collect per-step (obs, value) traces.

    Returns a list of game dicts:
      { "observations": [obs, ...],
        "values": [float, ...],
        "interp_features": [array, ...],
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
                trace_obs.append(obs.copy())
                trace_vals.append(value)
                trace_interp.append(_extract_interpretable(obs))

                mask = np.zeros(MAX_ACTIONS, dtype=bool)
                mask[:num_choices] = True
                action, _ = model.predict(obs, action_masks=mask, deterministic=True)
                action = int(action)
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
            "result": model_reward,
            "model_is_a": model_is_a,
        })
        if verbose:
            result_str = "W" if model_reward > 0 else ("L" if model_reward < 0 else "D")
            print(f"  game {g + 1}/{n_games}: {len(trace_obs)} decisions, {result_str}",
                  flush=True)
    return games


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


def cmd_value_swings(args):
    """Find games where the value function swung most dramatically."""
    import torch

    n_games = args.n_games
    top_n = args.top

    model, env, opp_model = _load_model_and_env(args)

    print(f"\nCollecting {n_games} game traces...")
    games = _collect_game_traces(model, env, opp_model, n_games)
    env.close()

    # For each game, find the max single-step value swing
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
    top_swings = swing_data[:top_n]

    print(f"\nTop {min(top_n, len(top_swings))} value function swings:\n")
    print(f"{'Game':<6} {'Step':<6} {'From':>8} {'To':>8} {'Delta':>8} {'Result':<8} {'Decisions':<10}")
    print("-" * 60)

    for s in top_swings:
        result_str = "WIN" if s["result"] > 0 else ("LOSS" if s["result"] < 0 else "DRAW")
        side = "A" if s["model_is_a"] else "B"
        print(f"  {s['game_idx']:<4}   {s['swing_step']:<4}   {s['swing_from']:>+7.3f} {s['swing_to']:>+7.3f}"
              f" {s['swing_to'] - s['swing_from']:>+7.3f}   {result_str:<6}({side}) {s['n_decisions']}")

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
    }[args.command](args)


if __name__ == "__main__":
    main()
