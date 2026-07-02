#!/usr/bin/env python3
"""Per-run coverage accounting for fuzz campaigns.

A campaign runs many single-game `test_harness.py` processes; during the
441-game league campaign the coverage gaps (cards never castable, action
categories never exercised) were only found by hand-grepping transcripts.
This module automates that bookkeeping:

  * ``CoverageAccumulator`` — the collection object. ``test_harness.py
    --coverage-json <path>`` builds one and ``runner.run_games`` calls its
    ``record()`` once per decision (menu categories + card ids + the chosen
    index), then the harness writes the counters as JSON to <path>. Collection
    is read-only observation: it never touches the RNG or the chosen action,
    so a run with and without the flag plays the identical game.

  * CLI ``merge`` / ``summarize`` — stdlib-only aggregation of the per-run
    JSONs a campaign produces (one harness process per game):

        train/.venv/bin/python train/coverage_report.py merge out.json in1.json in2.json ...
        train/.venv/bin/python train/coverage_report.py summarize out.json

JSON schema (one run; ``merge`` outputs the same shape with summed counters):

    {
      "decks": {"a": "league/bw_dnt", "b": "league/tron"},   # list of pairs after merge
      "seeds": [3],
      "n_games": 1,
      "runs": 1,
      "timestamp": 1751400000.0,
      "categories": {"CAST": {"id": 7, "offered": 41, "taken": 6}, ...},
      "cards": {"Aether Vial": {"id": 132, "offered": 9, "cast": 1,
                                 "activated": 2, "played_land": 0,
                                 "other_chosen": 3}, ...},
      "never_offered": ["Emrakul the Aeons Torn", ...],
      "unresolved": []
    }

``categories`` is keyed by the short display name from ``_enums._CAT_NAMES``;
"offered" counts menu-slot appearances, "taken" counts chosen slots. ``cards``
is keyed by vocab card name (sorted by vocab id) and covers every card that
belongs to either deck in play plus any other card seen in a menu — not the
whole 1024-row vocab. ``never_offered`` is the headline signal: deck cards
(mainboard + sideboard, resolved through the vocab) that never appeared as any
menu option. ``unresolved`` lists deck entries with no vocab identity (should
stay empty — such cards are unimplemented).
"""

import argparse
import json
import sys
import time

_CARD_COUNT_KEYS = ("offered", "cast", "activated", "played_land", "other_chosen")


def read_deck_cards(path):
    """Card names (with multiplicity) from a .dk file, mainboard + sideboard."""
    cards = []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.upper().startswith("SIDEBOARD"):
                continue
            qty, _, card = line.partition(" ")
            if qty.isdigit() and card.strip():
                cards.append(card.strip())
    return cards


def _rev_cat(cat_names, short_name):
    """ActionCategory int for a short display name (single source: _enums)."""
    for cat, name in cat_names.items():
        if name == short_name:
            return cat
    raise KeyError(short_name)


class CoverageAccumulator:
    """Offered/taken counters over the decisions of one harness run.

    Viewer-agnostic (both seats combined). ``record()`` is the only per-decision
    call — plain dict increments, cheap enough for fuzzing; when the harness flag
    is off no accumulator exists and the runner skips it with a None check.
    """

    def __init__(self, decks, seeds, n_games, deck_cards_a=(), deck_cards_b=()):
        # Lazy project imports so the merge/summarize CLI stays stdlib-only.
        from _enums import _CAT_NAMES
        from card_costs import _VOCAB_NAMES
        from missing_cards import name_to_uid, vocab_identity

        self._cat_names = _CAT_NAMES
        self._vocab_names = _VOCAB_NAMES
        self._cast = _rev_cat(_CAT_NAMES, "CAST")
        self._activate = _rev_cat(_CAT_NAMES, "ACTIVATE")
        self._land = _rev_cat(_CAT_NAMES, "LAND")
        self.decks = decks
        self.seeds = list(seeds)
        self.n_games = n_games
        self.cats = {}    # cat int -> [offered, taken]
        self.cards = {}   # vocab id -> [offered, cast, activated, played_land, other]
        # Resolve each seat's deck list (mainboard + sideboard + zone presets)
        # to vocab ids the same way the engine does — name_to_uid folding with
        # the DFC front-face fallback — so never_offered is deck-grounded.
        vocab = {name_to_uid(n): i for i, n in enumerate(_VOCAB_NAMES) if n}
        self.deck_ids = {}    # vocab id -> canonical vocab name
        self.unresolved = []  # deck entries with no vocab identity
        for name in list(deck_cards_a) + list(deck_cards_b):
            vid = vocab_identity(name_to_uid(name), vocab)
            if vid is None:
                if name not in self.unresolved:
                    self.unresolved.append(name)
            else:
                self.deck_ids.setdefault(vid, _VOCAB_NAMES[vid] or name)

    def record(self, cats, ids, action):
        """Tally one decision: the menu's categories/card ids + the chosen slot."""
        n_vocab = len(self._vocab_names)
        for j, cat in enumerate(cats):
            c = self.cats.get(cat)
            if c is None:
                c = self.cats[cat] = [0, 0]
            c[0] += 1
            cid = ids[j]
            if 0 <= cid < n_vocab:
                row = self.cards.get(cid)
                if row is None:
                    row = self.cards[cid] = [0, 0, 0, 0, 0]
                row[0] += 1
        if 0 <= action < len(cats):
            cat = cats[action]
            self.cats[cat][1] += 1
            cid = ids[action]
            if 0 <= cid < n_vocab:
                row = self.cards[cid]
                if cat == self._cast:
                    row[1] += 1
                elif cat == self._activate:
                    row[2] += 1
                elif cat == self._land:
                    row[3] += 1
                else:
                    row[4] += 1

    def _card_name(self, vid):
        # Slot N_CARD_TYPES - 1 is the TOKEN_SENTINEL (src/card_vocab.h): every
        # token permanent shares it, so its counters aggregate all tokens.
        if vid == len(self._vocab_names) - 1:
            return "(token)"
        name = self.deck_ids.get(vid)
        if name is None and vid < len(self._vocab_names):
            name = self._vocab_names[vid]
        return name or f"vocab_{vid}"

    def to_dict(self):
        categories = {}
        for cat in sorted(self.cats):
            offered, taken = self.cats[cat]
            categories[self._cat_names.get(cat, f"?{cat}")] = {
                "id": cat, "offered": offered, "taken": taken}
        cards = {}
        for vid in sorted(set(self.cards) | set(self.deck_ids)):
            row = self.cards.get(vid, [0, 0, 0, 0, 0])
            cards[self._card_name(vid)] = dict(
                {"id": vid}, **dict(zip(_CARD_COUNT_KEYS, row)))
        never = [self.deck_ids[vid] for vid in sorted(self.deck_ids)
                 if vid not in self.cards]
        return {"decks": self.decks, "seeds": self.seeds, "n_games": self.n_games,
                "runs": 1, "timestamp": time.time(), "categories": categories,
                "cards": cards, "never_offered": never,
                "unresolved": self.unresolved}

    def write(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.write("\n")


# ── merge / summarize (stdlib-only, operate on the JSON dicts) ────────────────

def _decks_list(report):
    """Normalize the decks field to a list of pairs (single run stores one)."""
    decks = report.get("decks")
    return decks if isinstance(decks, list) else ([decks] if decks else [])


def merge_reports(reports):
    """Sum a list of per-run report dicts into one report dict.

    never_offered is recomputed, not intersected: a card is kept only if the
    MERGED counters never saw it offered — so a card unreachable in one run but
    offered in another drops out.
    """
    out = {"decks": [], "seeds": [], "n_games": 0, "runs": 0,
           "timestamp": time.time(), "categories": {}, "cards": {},
           "never_offered": [], "unresolved": []}
    never_candidates = []
    for r in reports:
        for pair in _decks_list(r):
            if pair not in out["decks"]:
                out["decks"].append(pair)
        out["seeds"] += r.get("seeds", [])
        out["n_games"] += r.get("n_games", 0)
        out["runs"] += r.get("runs", 1)
        for name, e in r.get("categories", {}).items():
            m = out["categories"].setdefault(
                name, {"id": e["id"], "offered": 0, "taken": 0})
            m["offered"] += e.get("offered", 0)
            m["taken"] += e.get("taken", 0)
        for name, e in r.get("cards", {}).items():
            m = out["cards"].setdefault(
                name, dict({"id": e["id"]}, **{k: 0 for k in _CARD_COUNT_KEYS}))
            for k in _CARD_COUNT_KEYS:
                m[k] += e.get(k, 0)
        for name in r.get("never_offered", []):
            if name not in never_candidates:
                never_candidates.append(name)
        for name in r.get("unresolved", []):
            if name not in out["unresolved"]:
                out["unresolved"].append(name)
    out["categories"] = dict(sorted(out["categories"].items(),
                                    key=lambda kv: kv[1]["id"]))
    out["cards"] = dict(sorted(out["cards"].items(), key=lambda kv: kv[1]["id"]))
    out["never_offered"] = [
        n for n, e in out["cards"].items()
        if n in never_candidates and e["offered"] == 0]
    return out


def summarize(report):
    """Print the human summary: never-offered cards, dead categories, tables."""
    decks = _decks_list(report)
    pairs = ", ".join(f"{d.get('a', '?')} vs {d.get('b', '?')}" for d in decks)
    seeds = report.get("seeds", [])
    seed_str = (f"{min(seeds)}..{max(seeds)}" if seeds else "?")
    print(f"Coverage: {pairs}")
    print(f"  runs={report.get('runs', '?')}  games={report.get('n_games', '?')}"
          f"  seeds={seed_str} ({len(seeds)} listed)")

    never = report.get("never_offered", [])
    print(f"\nNever offered ({len(never)} deck card(s) with zero menu appearances):")
    cards = report.get("cards", {})
    for name in never:
        vid = cards.get(name, {}).get("id", "?")
        print(f"  [{vid:>4}] {name}")
    if not never:
        print("  (none — every deck card was offered at least once)")

    unresolved = report.get("unresolved", [])
    if unresolved:
        print(f"\nUnresolved deck entries (no vocab identity): {', '.join(unresolved)}")

    categories = report.get("categories", {})
    zero_taken = [n for n, e in categories.items()
                  if e["offered"] > 0 and e["taken"] == 0]
    print(f"\nCategories offered but never taken ({len(zero_taken)}): "
          f"{', '.join(zero_taken) or '(none)'}")

    print("\nCategory counts:")
    print(f"  {'id':>3}  {'name':<12} {'offered':>8} {'taken':>7}")
    for name, e in categories.items():
        print(f"  {e['id']:>3}  {name:<12} {e['offered']:>8} {e['taken']:>7}")

    print("\nCard counts (by vocab id):")
    print(f"  {'id':>4}  {'name':<34} {'offered':>8} {'cast':>5} "
          f"{'activ':>5} {'land':>5} {'other':>5}")
    for name, e in cards.items():
        print(f"  {e['id']:>4}  {name:<34} {e['offered']:>8} {e['cast']:>5} "
              f"{e['activated']:>5} {e['played_land']:>5} {e['other_chosen']:>5}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    mp = sub.add_parser("merge", help="sum per-run coverage JSONs into one")
    mp.add_argument("out", help="output JSON path")
    mp.add_argument("inputs", nargs="+", help="per-run coverage JSON files")
    sp = sub.add_parser("summarize", help="print a human summary of a coverage JSON")
    sp.add_argument("report", help="coverage JSON file (per-run or merged)")
    args = ap.parse_args(argv)

    if args.cmd == "merge":
        reports = []
        for p in args.inputs:
            with open(p) as f:
                reports.append(json.load(f))
        merged = merge_reports(reports)
        with open(args.out, "w") as f:
            json.dump(merged, f, indent=2)
            f.write("\n")
        print(f"merged {len(reports)} report(s) "
              f"({merged['n_games']} game(s)) -> {args.out}")
    else:
        with open(args.report) as f:
            report = json.load(f)
        summarize(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
