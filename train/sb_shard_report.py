#!/usr/bin/env python
"""Sideboard-decision report over recorded AZ self-play shards.

Reads trainer-schema shards (az_selfplay.SHARD_KEYS .npz files), finds every
between-games sideboarding SESSION (one player's boarding stage before game 2/3
of a bo3), and reports the average net cards brought IN and cut OUT per session,
grouped by matchup (the sideboarding player's deck vs the opponent's deck).

A session's swaps are recovered from the observation itself, not from the
recorded policy: the viewer's LIVE maindeck block (env._SELF_DECK_MAIN block)
tracks each completed swap mid-phase, so diffing the deck configuration at the
session's first row against its last balanced row yields exactly the completed
swaps. Rows are grouped into sessions by the is_sideboard_phase flag, with a
boundary whenever the viewer seat flips (state[_SELF_IS_A_IDX]), the
swaps-completed counter resets, or a non-sideboard row intervenes.

Decks are identified by matching the (boarding-invariant) main+side 75 multiset
against the league decklists; unmatched decks fall back to the matchup tail's
archetype one-hot. All fetchlands are treated as fungible (collapsed to one
"Fetchland" line) since which fetch a deck boards says nothing strategic.

Usage:
  sb_shard_report.py [shard.npz ...] [--dir train/az_data/gen] [--last N]
                     [--mtime-after 'YYYY-mm-dd HH:MM'] [--mtime-before ...]
                     [--min-sessions N] [--json out.json]
"""

import argparse
import collections
import datetime
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import env  # noqa: E402  (offset authority; asserts its chain vs machine_io.h)
from card_costs import N_CARD_TYPES, _VOCAB_NAMES  # noqa: E402
from archetypes import ARCHETYPES, UNKNOWN_NAME  # noqa: E402

DECKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "bin", "resources", "decks")

# Fetchlands present in the vocab — fungible for this report.
FETCHLANDS = {
    "Scalding Tarn", "Flooded Strand", "Polluted Delta", "Wooded Foothills",
    "Misty Rainforest", "Windswept Heath", "Bloodstained Mire",
    "Verdant Catacombs", "Arid Mesa", "Marsh Flats", "Prismatic Vista",
}
FETCH_LABEL = "Fetchland (any)"


def _norm_name(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


_NORM_TO_IDX = {_norm_name(n): i for i, n in enumerate(_VOCAB_NAMES)}
_FETCH_IDS = {i for i, n in enumerate(_VOCAB_NAMES) if n in FETCHLANDS}


def card_label(idx):
    if idx in _FETCH_IDS:
        return FETCH_LABEL
    if 0 <= idx < len(_VOCAB_NAMES):
        return _VOCAB_NAMES[idx]
    return f"card#{idx}"


def decode_slots(vec):
    """(card_id, count) slot block -> Counter{vocab_idx: count}."""
    out = collections.Counter()
    for i in range(0, len(vec), 2):
        idx = int(round(float(vec[i]) * N_CARD_TYPES))
        if idx < 0:
            continue
        ct = int(round(float(vec[i + 1]) * 4.0))
        if ct > 0:
            out[idx] += ct
    return out


def load_league_decks():
    """{deck_stem: Counter(vocab_idx -> count over the full 75)}."""
    decks = {}
    for path in sorted(glob.glob(os.path.join(DECKS_DIR, "league", "*.dk"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        counts = collections.Counter()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.upper().startswith("SIDEBOARD"):
                    continue
                m = re.match(r"^(\d+)\s+(.+)$", line)
                if not m:
                    continue
                idx = _NORM_TO_IDX.get(_norm_name(m.group(2)))
                if idx is not None:
                    counts[idx] += int(m.group(1))
        decks[stem] = counts
    return decks


def identify_deck(seventy_five, league, arch_onehot):
    """Best league-deck stem for a 75 multiset; archetype fallback."""
    best, best_ov = None, -1
    for stem, counts in league.items():
        ov = sum((seventy_five & counts).values())
        if ov > best_ov:
            best, best_ov = stem, ov
    total = sum(seventy_five.values())
    if best is not None and total and best_ov >= 0.8 * total:
        return best
    a = int(np.argmax(arch_onehot)) if arch_onehot.max() > 0.5 else len(ARCHETYPES)
    return ARCHETYPES[a] if a < len(ARCHETYPES) else UNKNOWN_NAME


def iter_sessions(obs):
    """Yield (first_row_idx, [row indices]) for each sideboarding session."""
    cur, cur_viewer, prev_swaps = [], None, -1.0
    for i in range(obs.shape[0]):
        row = obs[i]
        if row[env._IS_SIDEBOARD_IDX] <= 0.5:
            if cur:
                yield cur
            cur, cur_viewer, prev_swaps = [], None, -1.0
            continue
        viewer = row[env._SELF_IS_A_IDX] > 0.5
        swaps = float(row[env._EXTRAS_SB_SWAPS])
        if cur and (viewer != cur_viewer or swaps < prev_swaps - 1e-6):
            yield cur
            cur = []
        cur.append(i)
        cur_viewer, prev_swaps = viewer, swaps
    if cur:
        yield cur


def analyze_session(obs, rows):
    """One session -> (self_75, opp_75, self_arch_onehot, opp_arch_onehot,
    in_counts, out_counts) with fetchlands collapsed; None if unusable."""
    first = obs[rows[0]]
    # last row where the config is balanced (drift float == 0.5 midpoint)
    last = None
    for i in reversed(rows):
        if abs(float(obs[i][env._EXTRAS_SB_DELTA]) - 0.5) < 0.1:
            last = obs[i]
            break
    if last is None:
        return None

    def main_cfg(row):
        return decode_slots(row[env._SELF_DECK_MAIN_START:env._SELF_DECK_MAIN_END])

    def collapse(counter):
        out = collections.Counter()
        for idx, ct in counter.items():
            out[card_label(idx)] += ct
        return out

    before, after = collapse(main_cfg(first)), collapse(main_cfg(last))
    ins = collections.Counter({c: n for c, n in (after - before).items()})
    outs = collections.Counter({c: n for c, n in (before - after).items()})

    self75 = (decode_slots(first[env._SELF_DECK_MAIN_START:env._SELF_DECK_MAIN_END])
              + decode_slots(first[env._SELF_DECK_SIDE_START:env._SELF_DECK_SIDE_END]))
    opp75 = (decode_slots(first[env._OPP_DECK_MAIN_START:env._OPP_DECK_MAIN_END])
             + decode_slots(first[env._OPP_DECK_SIDE_START:env._OPP_DECK_SIDE_END]))
    n_arch = len(ARCHETYPES) + 1
    self_oh = first[env.ARCH_ONEHOT_START:env.ARCH_ONEHOT_START + n_arch]
    opp_oh = first[env.ARCH_ONEHOT_START + n_arch:env.ARCH_ONEHOT_END]
    return self75, opp75, self_oh, opp_oh, ins, outs


def collect(paths):
    """Aggregate sessions across shards -> {matchup: stats dict}."""
    league = load_league_decks()
    groups = {}
    n_shards_with_sb = 0
    for path in paths:
        try:
            obs = np.load(path)["obs"]
        except Exception as e:  # unreadable / truncated shard
            print(f"[warn] skipping {path}: {e}", file=sys.stderr)
            continue
        found = False
        for rows in iter_sessions(obs):
            res = analyze_session(obs, rows)
            if res is None:
                continue
            self75, opp75, self_oh, opp_oh, ins, outs = res
            found = True
            key = (identify_deck(self75, league, self_oh),
                   identify_deck(opp75, league, opp_oh))
            g = groups.setdefault(key, {
                "sessions": 0, "in": collections.Counter(),
                "out": collections.Counter(), "swap_counts": [],
                "unbalanced": 0,
            })
            g["sessions"] += 1
            g["in"] += ins
            g["out"] += outs
            g["swap_counts"].append(sum(ins.values()))
            if sum(ins.values()) != sum(outs.values()):
                g["unbalanced"] += 1
        n_shards_with_sb += found
    return groups, n_shards_with_sb


def render(groups, n_paths, n_shards_with_sb, label=""):
    lines = []
    title = f"Sideboard report{' — ' + label if label else ''}"
    lines.append(title)
    lines.append("=" * len(title))
    total_sessions = sum(g["sessions"] for g in groups.values())
    lines.append(f"{n_paths} shard(s) scanned, {n_shards_with_sb} with sideboard "
                 f"sessions, {total_sessions} session(s) total. Fetchlands fungible.")
    for (me, opp), g in sorted(groups.items(),
                               key=lambda kv: -kv[1]["sessions"]):
        n = g["sessions"]
        avg_swaps = np.mean(g["swap_counts"]) if g["swap_counts"] else 0.0
        lines.append("")
        lines.append(f"{me} vs {opp}  —  {n} sessions, avg {avg_swaps:.2f} swaps/session"
                     + (f", {g['unbalanced']} unbalanced" if g["unbalanced"] else ""))
        rows = sorted(set(g["in"]) | set(g["out"]),
                      key=lambda c: -(g["in"][c] + g["out"][c]))
        for c in rows:
            i, o = g["in"][c] / n, g["out"][c] / n
            marks = []
            if i:
                marks.append(f"in {i:+.2f}")
            if o:
                marks.append(f"out {-o:+.2f}")
            lines.append(f"    {c:<32} {'  '.join(marks)}")
    return "\n".join(lines)


def to_json(groups):
    out = {}
    for (me, opp), g in groups.items():
        n = g["sessions"]
        out[f"{me} vs {opp}"] = {
            "self_deck": me, "opp_deck": opp, "sessions": n,
            "avg_swaps": float(np.mean(g["swap_counts"])) if g["swap_counts"] else 0.0,
            "unbalanced": g["unbalanced"],
            "avg_in": {c: ct / n for c, ct in sorted(g["in"].items())},
            "avg_out": {c: ct / n for c, ct in sorted(g["out"].items())},
        }
    return out


def parse_when(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"unrecognized time: {s!r}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("shards", nargs="*", help="explicit shard .npz paths")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "az_data", "gen"),
        help="shard directory to scan when no explicit paths given")
    ap.add_argument("--last", type=int, default=None,
                    help="only the N most recently modified shards")
    ap.add_argument("--mtime-after", type=parse_when, default=None)
    ap.add_argument("--mtime-before", type=parse_when, default=None)
    ap.add_argument("--min-sessions", type=int, default=1,
                    help="hide matchups with fewer sessions than this")
    ap.add_argument("--label", default="", help="report title suffix")
    ap.add_argument("--json", default=None, help="also write JSON to this path")
    args = ap.parse_args()

    paths = args.shards or sorted(glob.glob(os.path.join(args.dir, "*.npz")),
                                  key=os.path.getmtime)
    if args.mtime_after is not None:
        paths = [p for p in paths if os.path.getmtime(p) >= args.mtime_after]
    if args.mtime_before is not None:
        paths = [p for p in paths if os.path.getmtime(p) < args.mtime_before]
    if args.last is not None:
        paths = sorted(paths, key=os.path.getmtime)[-args.last:]
    if not paths:
        print("no shards matched", file=sys.stderr)
        return 1

    groups, n_with_sb = collect(paths)
    shown = {k: g for k, g in groups.items() if g["sessions"] >= args.min_sessions}
    print(render(shown, len(paths), n_with_sb, args.label))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"label": args.label, "n_shards": len(paths),
                       "matchups": to_json(groups)}, f, indent=2)
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
