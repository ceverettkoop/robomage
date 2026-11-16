#!/usr/bin/env python3
"""Ablations of a network's card inputs over a recorded window (rmlearn/ablation.py): what the policy
and the value lose without a card's identity, or without a group of card properties.

    cd learn
    HSA_OVERRIDE_GFX_VERSION=10.3.0 flock /tmp/rm-gpu.lock .venv/bin/python ablate.py identity \\
        --run ../runs/league26-24h --checkpoint m9 --since ../runs/hist-off/final-model.pt --plot
    HSA_OVERRIDE_GFX_VERSION=10.3.0 flock /tmp/rm-gpu.lock .venv/bin/python ablate.py properties \\
        --run ../runs/league26-24h --checkpoint m9 --plot
    python3 ablate.py report ../runs/league26-24h/ablation/identity-m9 --by key,in_menu --min-n 50
    python3 ablate.py plot ../runs/league26-24h/ablation/identity-m9

``identity`` and ``properties`` evaluate one network (``--checkpoint``: ``m<k>``, ``final`` or a file;
default the run's last) on a seeded sample of the run's recorded decisions (``--episodes`` episode
files, ``--per-episode`` decisions of each; the same ``--seed`` and filters give the same decisions for
every network) and write ``result.npz`` + ``meta.json`` and their tables as CSV into ``--out``
(default ``<run>/ablation/<kind>-<checkpoint>``). They need the learner's venv (torch); on the GPU this
must be the only process on it (no training run). ``report`` and ``plot`` read a result directory with
numpy alone (``plot`` needs matplotlib: any Python that has it, e.g. the system's).

Narrowing: ``--deck league/elves`` (the deciding seat's deck; repeatable), ``--cards "Trinisphere;Natural
Order"`` (';'-separated names, unique substrings or ids: only decisions holding one of them, and only
those cards ablated), ``--sideboarding only|skip``.

``properties`` groups: ``--groups cost,text,id`` picks among the defaults (``--list-groups`` prints
them; ``none`` drops them), ``--group NAME=TABLE:REGEX[+TABLE:REGEX]`` adds one from column names
(tables ``card``, ``text``, ``ability``, ``part``), ``--split card:^card_keyword`` adds one group per
matching column that varies. ``--mode mean`` replaces a group by its mean over the cards the network has
seen, ``resample`` by another seen card's values.

Columns: ``tv`` total variation between the unablated and the ablated policy, ``kl``, ``flip`` the share of
decisions whose argmax changed, ``dlogp`` the change in log π of the unablated argmax, ``adv`` / ``dv``
the absolute / signed value change, ``dse`` the change in the value's squared error against the game's
result, each ``mean ± se`` (clustered by episode).
"""

import argparse
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rmlearn import ablation as AB  # noqa: E402


def resolve_checkpoint(run, name):
    """A checkpoint file from `name` (``m<k>``, ``final``, a path) or, without one, the run's last."""
    fixed = os.path.join(run or "", "ckpt", "fixed")
    final = [os.path.join(run or "", "ckpt", "rnad-final.pt"), os.path.join(run or "", "final-model.pt")]
    if name and os.path.isfile(name):
        return name
    if not run:
        raise SystemExit("ablate: --checkpoint FILE (or --run DIR)")
    if name and re.fullmatch(r"m\d+", name):
        cands = [os.path.join(fixed, name + ".pt")]
    elif name in (None, "final"):
        ms = sorted(glob.glob(os.path.join(fixed, "m*.pt")), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p))))
        cands = final + ([] if name else ms[::-1])
    else:
        cands = []
    for c in cands:
        if os.path.isfile(c):
            return c
    raise SystemExit("ablate: no checkpoint %r in %s" % (name, run))


def add_compute_args(p):
    p.add_argument("--run", help="a run directory: its tables/, episodes/selfplay/ and checkpoints")
    p.add_argument("--checkpoint", help="m<k>, final, or a checkpoint file (default: the run's last)")
    p.add_argument("--tables", help="the tables directory (default <run>/tables)")
    p.add_argument("--episodes-dir", help="the episodes directory (default <run>/episodes/selfplay)")
    p.add_argument("--episodes", type=int, default=2000, help="episode files sampled (0: all)")
    p.add_argument("--per-episode", type=int, default=8, help="decisions per episode (0: all)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--deck", action="append", default=[], help="only the decisions of seats playing this deck")
    p.add_argument("--cards", help="';'-separated card names or ids: only these cards, where they appear")
    p.add_argument("--sideboarding", choices=("all", "only", "skip"), default="all")
    p.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    p.add_argument("--batch", type=int, default=128, help="decisions per batch")
    p.add_argument("--forward-batch", type=int, default=256, help="decisions per forward")
    p.add_argument("--out", help="the result directory (default <run>/ablation/<kind>-<checkpoint>)")
    p.add_argument("--plot", action="store_true", help="also write the charts (needs matplotlib)")
    p.add_argument("--top", type=int, default=30, help="rows printed and charted")
    p.add_argument("--min-n", type=int, default=20, help="a card's fewest rows to be listed")
    p.add_argument("--quiet", action="store_true")


def setup(args, kind):
    import torch
    from rmlearn import spec as S
    tables_dir = args.tables or os.path.join(args.run or "", "tables")
    episodes_dir = args.episodes_dir or os.path.join(args.run or "", "episodes", "selfplay")
    ckpt = resolve_checkpoint(args.run, args.checkpoint)
    tables = S.load_tables(tables_dir)
    net = AB.load_network(ckpt, tables)
    names = {int(i): n for i, n in tables.hashes.get("vocab", [])}
    cards = AB.resolve_cards(names, args.cards.split(";")) if args.cards else None
    decks = [d for arg in args.deck for d in arg.split(",") if d]
    sb = {"all": None, "only": True, "skip": False}[args.sideboarding]
    sample = AB.Sample(episodes_dir, args.episodes, args.per_episode, args.seed, decks or None, cards, sb,
                       args.batch)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    stem = os.path.splitext(os.path.basename(ckpt))[0]
    out = args.out or (os.path.join(args.run, "ablation", "%s-%s" % (kind, stem)) if args.run else None)
    if not out:
        raise SystemExit("ablate: --out DIR (or --run DIR)")
    if not args.quiet:
        print("%s: %s on %s, %d episode files in %s → %s" % (kind, ckpt, device, len(sample.files), episodes_dir,
                                                               out), flush=True)
    return tables, net, sample, cards, device, ckpt, out


def finish(res, args, ckpt, out):
    res.meta["checkpoint"] = os.path.abspath(ckpt)
    res.meta["argv"] = sys.argv[1:]
    res.save(out)
    report(res, out, args.top, args.min_n, write=True)
    if args.plot:
        plot(res, out, args.top, args.min_n)


def report(res, out, top, min_n, write=False):
    """Prints the result's standard tables; with `write` also as CSV files in `out`."""
    b = res.base_summary()
    print("\n%s ablation, %s\n%d decisions of %d episodes; unablated value: MSE %.3f, correlation with the result "
          "%.3f; mean entropy %.3f over %.1f menu rows" % (res.kind, res.meta.get("checkpoint", ""), b["decisions"],
                                                         b["episodes"], b["value_mse"], b["value_corr"], b["entropy"],
                                                         b["menu"]))
    tables = []
    if res.kind == "identity":
        tables.append(("every id at once", ("key",), res.table(("key",), where={"key": AB.ALL}), None, None))
        tables.append(("every id at once, by deck", ("deck",), res.table(("deck",), where={"key": AB.ALL}, sort="tv"),
                       "decks.csv", None))
        cards = [r for r in res.table(("key",), min_n=min_n, sort="tv") if "card_id" in r]
        extra = ("n", "episodes", "drift") + (("new",) if "new" in res.meta["card"] else ()) + AB.DEFAULT_COLUMNS[2:]
        tables.append(("one card's id (cards with ≥ %d rows, by policy change)" % min_n, ("key",), cards, "cards.csv",
                       extra))
        if "new" in res.meta["card"]:
            col = res.samples["key"]
            new = [int(k) for k in res.meta["card"]["new"]]
            for label, ids in (("cards first seen after --since", new),
                               ("cards seen before", sorted(set(col[col >= 0].tolist()) - set(new)))):
                rows = res.table(("in_menu",), where={"key": ids}) if ids else []
                tables.append(("one card's id: %s, by whether the card is on a menu row" % label, ("in_menu",), rows,
                               None, None))
    else:
        tables.append(("property groups (%s)" % res.meta.get("mode"), ("key",), res.table(("key",), sort="tv"),
                       "groups.csv", ("n", "touched") + AB.DEFAULT_COLUMNS[2:]))
        if write:
            AB.write_csv(os.path.join(out, "groups_by_deck.csv"), res.table(("key", "deck")), ("key", "deck"))
    for title, by, rows, name, columns in tables:
        print("\n%s%s" % (title, "" if not top or len(rows) <= top else " (top %d of %d)" % (top, len(rows))))
        print(AB.format_rows(rows, by, columns or AB.DEFAULT_COLUMNS, top) if rows else "  (no rows)")
        if write and name:
            AB.write_csv(os.path.join(out, name), rows, by)
    if write:
        print("\nresult: %s (result.npz, meta.json, *.csv)" % out)


def plot(res, out, top, min_n):
    try:
        from rmlearn import ablation_plot as AP
        files = AP.plot(res, out, top=top, min_n=min_n)
    except ImportError as e:
        print("no charts: %s (this Python lacks matplotlib; any that has it will do: python3 %s plot %s)"
              % (e, os.path.relpath(__file__), out))
        return
    for f in files:
        print("chart: %s" % f)


def cmd_identity(args):
    tables, net, sample, cards, device, ckpt, out = setup(args, "identity")
    res = AB.identity_ablation(net, tables, sample, per_decision=args.per_decision, cards=cards,
                               balance=not args.no_balance, device=device, forward_batch=args.forward_batch,
                               since=args.since, quiet=args.quiet)
    finish(res, args, ckpt, out)


def cmd_properties(args):
    if args.list_groups and not (args.run or args.tables):
        raise SystemExit("ablate: --list-groups needs --run or --tables")
    tables, net, sample, cards, device, ckpt, out = setup(args, "properties")
    defaults = AB.default_groups(tables)
    if args.groups == "none":
        groups = []
    elif args.groups:
        unknown = [g for g in args.groups.split(",") if g not in defaults]
        if unknown:
            raise SystemExit("ablate: no default group %s (one of %s)" % (", ".join(unknown), ", ".join(defaults)))
        groups = [defaults[g] for g in args.groups.split(",")]
    else:
        groups = list(defaults.values())
    for spec in args.group:
        name, _, body = spec.partition("=")
        groups.append(AB.custom_group(tables, name, body))
    for spec in args.split:
        groups.extend(AB.split_groups(tables, spec, net))
    if args.list_groups:
        for g in groups:
            d = g.describe()
            print("%-18s %s%s" % (d["name"], ", ".join("%s: %d columns" % kv for kv in d["columns"].items()),
                                  (", " if d["columns"] else "") + "card ids" if d["ids"] else ""))
        return
    if not groups:
        raise SystemExit("ablate: no groups")
    res = AB.property_ablation(net, tables, sample, groups=groups, mode=args.mode, cards=cards, device=device,
                               forward_batch=args.forward_batch, quiet=args.quiet)
    finish(res, args, ckpt, out)


def cmd_report(args):
    res = AB.Result.load(args.dir)
    if not args.by and not args.where:
        report(res, args.dir, args.top, args.min_n)
        return
    by = tuple(args.by.split(",")) if args.by else ("key",)
    where = {}
    for item in filter(None, (args.where or "").split(",")):
        col, _, val = item.partition("=")
        where.setdefault(col, []).append(val)
    rows = res.table(by, where=where, min_n=args.min_n, sort=args.sort)
    columns = tuple(args.columns.split(",")) if args.columns else AB.DEFAULT_COLUMNS
    print(AB.format_rows(rows, by, columns, args.top))
    if args.csv:
        AB.write_csv(args.csv, rows, by)


def cmd_plot(args):
    res = AB.Result.load(args.dir)
    plot(res, args.out or args.dir, args.top, args.min_n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("identity", help="a card's id row replaced by the unknown-card vector")
    add_compute_args(p)
    p.add_argument("--per-decision", type=int, default=8, help="single cards ablated per decision (0: all present)")
    p.add_argument("--no-balance", action="store_true", help="choose a decision's cards at random, not rarest first")
    p.add_argument("--since", help="an earlier checkpoint: marks the cards first seen after it")
    p.set_defaults(fn=cmd_identity)
    p = sub.add_parser("properties", help="groups of card-table columns replaced")
    add_compute_args(p)
    p.add_argument("--mode", choices=("mean", "resample"), default="mean")
    p.add_argument("--groups", help="default groups to run, comma-separated (none: only --group / --split)")
    p.add_argument("--group", action="append", default=[], metavar="NAME=TABLE:REGEX[+TABLE:REGEX]")
    p.add_argument("--split", action="append", default=[], metavar="TABLE:REGEX")
    p.add_argument("--list-groups", action="store_true", help="print the groups and exit")
    p.set_defaults(fn=cmd_properties)
    p = sub.add_parser("report", help="tables of a result directory")
    p.add_argument("dir")
    p.add_argument("--by", help="columns to group by, comma-separated: key, deck, opp_deck, in_menu, sideboarding, "
                                "seat, game, slots, menu")
    p.add_argument("--where", help="column=value filters, comma-separated (labels or integers; repeat a column "
                                   "for several values)")
    p.add_argument("--sort", default="tv")
    p.add_argument("--columns", help="columns shown (default %s)" % ",".join(AB.DEFAULT_COLUMNS))
    p.add_argument("--top", type=int, default=40)
    p.add_argument("--min-n", type=int, default=20)
    p.add_argument("--csv", help="also write the rows to this file")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("plot", help="charts of a result directory (matplotlib)")
    p.add_argument("dir")
    p.add_argument("--out", help="where the charts go (default the result directory)")
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--min-n", type=int, default=20)
    p.set_defaults(fn=cmd_plot)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
