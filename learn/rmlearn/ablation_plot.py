"""Charts of an ablation Result (rmlearn/ablation.py) as PNG files: numpy + matplotlib, no torch.

    plot(Result.load("out/dir"), "out/dir")     # → the files written

Identity: ``identity_cards.png`` (the cards whose identity moves the policy most, with the value
change beside it), ``identity_drift.png`` (per card, the policy effect against its id row's distance
from the unknown-card vector) and ``identity_decks.png`` (every id ablated at once, per deck).
Properties: ``property_groups.png`` (each group's effect on the policy, the value and the value's
squared error) and ``property_by_deck.png`` (the policy effect per group and deck).
Bars carry ±1 standard error (clustered by episode); every chart's numbers are the CSV files beside it.
"""

import os

import numpy as np

SURFACE, TEXT, TEXT_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, NEUTRAL = "#2a78d6", "#eb6834", "#a9a8a1"
RAMP = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b")     # sequential: one hue, light → dark
LABEL = {"tv": "policy change (total variation)", "kl": "policy change (KL)", "flip": "argmax changed (share)",
         "adv": "|value change|", "dv": "value change", "dse": "value squared-error change",
         "dlogp": "change in log π of the preferred action"}
COMPOSITE = ("card.all", "card+text", "ability.all", "all")


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT,
        "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
    })
    return plt


def _short(deck):
    return deck.split("/")[-1]


def _bars(ax, rows, metric, colors=BLUE, labels=None):
    """Horizontal bars of `rows`' metric (first row on top) with ±1 SE whiskers."""
    y = np.arange(len(rows))[::-1]
    val = np.array([r[metric] for r in rows])
    se = np.array([r.get(metric + "_se", 0.0) for r in rows])
    ax.barh(y, val, height=0.62, color=colors, linewidth=0)
    ax.errorbar(val, y, xerr=se, fmt="none", ecolor=TEXT_2, elinewidth=0.8, capsize=0)
    if labels is not None:
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel(LABEL[metric])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    if val.min() >= 0:
        ax.set_xlim(0, max(1e-9, float((val + se).max()) * 1.05))


def _finish(plt, fig, path, title, note):
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _sample_note(res, bars=True):
    s = res.base_summary()
    return "%d decisions of %d episodes · %s%s" % (
        s["decisions"], s["episodes"], os.path.basename(str(res.meta.get("checkpoint", ""))),
        " · bars ±1 SE (clustered by episode)" if bars else "")


def plot_identity(res, out_dir, top=30, metric="tv", min_n=20):
    plt = _pyplot()
    files = []
    card = res.meta["card"]
    rows = [r for r in res.table(("key",), min_n=min_n, sort=metric) if r.get("card_id") is not None]
    mark = "new" in card
    name = lambda r: r["key"] + (" *" if r.get("new") else "")  # noqa: E731
    starred = " · * first seen after %s" % os.path.basename(card["since"]) if mark else ""
    if rows:
        shown = rows[:top]
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.24 * len(shown) + 1.6), sharey=True, layout="constrained")
        _bars(axes[0], shown, metric, labels=[name(r) for r in shown])
        _bars(axes[1], shown, "adv")
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.95))
        files.append(_finish(plt, fig, os.path.join(out_dir, "identity_cards.png"),
                             "Card identity ablated one card at a time: the %d cards that move the policy most"
                             % len(shown), _sample_note(res) + starred + " · cards with ≥ %d rows" % min_n))
        # Policy effect against the id row's drift, one dot per card.
        pts = [r for r in rows if r.get("drift") is not None]
        if pts:
            fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")
            ax.grid(True, axis="both")
            series = [("seen before", BLUE, [r for r in pts if not r.get("new")])]
            if mark:
                series.append(("first seen after %s" % os.path.basename(card["since"]), ORANGE,
                               [r for r in pts if r.get("new")]))
            for label, color, rs in series:
                ax.scatter([r["drift"] for r in rs], [r[metric] for r in rs], s=22, color=color, edgecolor=SURFACE,
                           linewidth=0.8, label=label, zorder=3)
            far = sorted(pts, key=lambda r: -r[metric])[:12]
            for r in {r["card_id"]: r for r in far}.values():
                ax.annotate(r["key"], (r["drift"], r[metric]), xytext=(4, 3), textcoords="offset points", fontsize=7,
                            color=TEXT_2)
            ax.set_xlabel("id row's distance from the unknown-card vector")
            ax.set_ylabel(LABEL[metric])
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            if mark:
                ax.legend(loc="upper left")
            fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.94))
            files.append(_finish(plt, fig, os.path.join(out_dir, "identity_drift.png"),
                                 "What a card's id row holds against what removing it changes",
                                 _sample_note(res, bars=False) + " · one dot per card with ≥ %d rows" % min_n))
    decks = res.table(("key", "deck"), where={"key": -1}, sort=metric)
    if decks:
        fig, axes = plt.subplots(1, 2, figsize=(10, 0.24 * len(decks) + 1.6), sharey=True, layout="constrained")
        _bars(axes[0], decks, metric, labels=[_short(r["deck"]) for r in decks])
        _bars(axes[1], decks, "adv")
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.95))
        files.append(_finish(plt, fig, os.path.join(out_dir, "identity_decks.png"),
                             "Every card id ablated at once, by the deciding seat's deck", _sample_note(res)))
    return files


def plot_properties(res, out_dir, metric="tv", top=40):
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch
    files = []
    rows = res.table(("key",), sort=metric)[:top]
    if rows:
        colors = [NEUTRAL if r["key"] in COMPOSITE else BLUE for r in rows]
        fig, axes = plt.subplots(1, 3, figsize=(12, 0.26 * len(rows) + 1.9), sharey=True, layout="constrained")
        _bars(axes[0], rows, metric, colors, labels=[r["key"] for r in rows])
        _bars(axes[1], rows, "adv", colors)
        _bars(axes[2], rows, "dse", colors)
        if NEUTRAL in colors:
            axes[0].legend(handles=[Patch(color=BLUE, label="one group"), Patch(color=NEUTRAL, label="composite")],
                           loc="lower right")
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.95))
        files.append(_finish(plt, fig, os.path.join(out_dir, "property_groups.png"),
                             "Card property groups ablated (%s substitution)" % res.meta.get("mode", ""),
                             _sample_note(res)))
    cells = res.table(("key", "deck"))
    groups = [r["key"] for r in rows]
    decks = sorted({r["deck"] for r in cells})
    if len(groups) > 1 and len(decks) > 1:
        grid = np.full((len(groups), len(decks)), np.nan)
        for r in cells:
            if r["key"] in groups:
                grid[groups.index(r["key"]), decks.index(r["deck"])] = r[metric]
        fig, ax = plt.subplots(figsize=(0.42 * len(decks) + 2.6, 0.27 * len(groups) + 2.6), layout="constrained")
        ax.grid(False)
        im = ax.imshow(grid, aspect="auto", cmap=LinearSegmentedColormap.from_list("seq", RAMP), vmin=0)
        ax.set_xticks(np.arange(len(decks)))
        ax.set_xticklabels([_short(d) for d in decks], rotation=60, ha="right", color=TEXT)
        ax.set_yticks(np.arange(len(groups)))
        ax.set_yticklabels(groups)
        ax.set_xticks(np.arange(len(decks) + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(groups) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.5)
        ax.tick_params(which="minor", length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        bar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        bar.set_label(LABEL[metric], color=TEXT_2)
        bar.outline.set_visible(False)
        fig.get_layout_engine().set(rect=(0, 0.03, 1, 0.95))
        files.append(_finish(plt, fig, os.path.join(out_dir, "property_by_deck.png"),
                             "Property groups ablated, by the deciding seat's deck",
                             _sample_note(res, bars=False) + " · %s substitution" % res.meta.get("mode", "")))
    return files


def plot(res, out_dir, **kw):
    """Writes the Result's charts into `out_dir`; the files' paths. Needs matplotlib."""
    os.makedirs(out_dir, exist_ok=True)
    if res.kind == "identity":
        return plot_identity(res, out_dir, **{k: v for k, v in kw.items() if k in ("top", "metric", "min_n")})
    return plot_properties(res, out_dir, **{k: v for k, v in kw.items() if k in ("top", "metric")})
