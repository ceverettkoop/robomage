#!/usr/bin/env python
"""Visualize the AZ card-identity embedding as a 2D chart — trained rows only.

Builds on az_inspect's data functions (embedding tables, exposure, labels) and
viz.py's headless-safe matplotlib helpers. The key difference from
``az_inspect project`` (the terminal scatter): rows whose embedding never
received gradient are EXCLUDED, so the picture shows only what training
actually wrote into the space, not the warm-start residue of absent cards.

"Never trained" is measured cumulatively: the exposure baseline defaults to
the PPO checkpoint AZ warm-started from (a row untouched since warm-start is
bit-identical — az_train exempts the embedding from weight decay), falling
back to the oldest local ``gen__azv*`` snapshot when no PPO gen exists.

Examples:
    train/.venv/bin/python train/az_embed_viz.py                       # PCA, colored by type
    train/.venv/bin/python train/az_embed_viz.py --method tsne --mark color
    train/.venv/bin/python train/az_embed_viz.py --label-top 40 --show
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import az_inspect
import viz
from az_inspect import (CARD_SPACES, LABEL_KINDS, _snapshot_steps,
                        card_exposure, card_labels, card_matrix, card_name,
                        load_net, named_card_ids, pca2, resolve_spec)


def cumulative_baseline(path):
    """The baseline against which "ever trained" is measured: the PPO gen
    checkpoint AZ warm-started from, else the OLDEST az snapshot strictly older
    than ``path`` (az_inspect.resolve_baseline prefers the NEWEST, which only
    measures the last training slice)."""
    from opponents import resolve_checkpoint
    ppo = resolve_checkpoint("gen")
    if ppo and os.path.isfile(ppo):
        return ppo
    import glob
    ckpt_dir = os.path.dirname(os.path.abspath(path))
    steps = _snapshot_steps(path)
    snaps = [(s, p) for p in glob.glob(os.path.join(ckpt_dir, "gen__azv*.pt"))
             for s in (_snapshot_steps(p),)
             if s >= 0 and (steps < 0 or s < steps)]
    return min(snaps)[1] if snaps else None


def trained_ids(path, baseline=None):
    """``(vocab ids whose embedding row ever moved, per-id movement, exposure)``
    over the named vocab."""
    exp = card_exposure(path, baseline=baseline)
    ids = named_card_ids()
    delta = exp["delta"]
    keep = ids[np.array([delta[i] > 0.0 for i in ids])]
    return keep, np.array([delta[i] for i in keep]), exp


def movement_deltas(path, baseline, ids):
    """Per-card embedding movement VECTORS (current row - baseline row) for the
    given vocab ids — the directions training pushed each card, not just how
    far (card_exposure keeps only the norms)."""
    emb_a, _ = az_inspect._card_emb_and_value(baseline)
    emb_b, _ = az_inspect._card_emb_and_value(path)
    # Row i+1 is vocab card i (row 0 is the padding row).
    delta = (emb_b - emb_a).numpy().astype(np.float64)[1:]
    return delta[np.asarray(ids)]


def movement_pcs(delta):
    """SVD of the movement matrix, UNCENTERED so the decomposition is exact:
    per card, the squared projections across all PCs sum to its squared total
    movement. Returns ``(proj (n, d), energy_frac (d,))`` — proj[i, k] is card
    i's movement along movement-PC k."""
    u, s, _ = np.linalg.svd(delta, full_matrices=False)
    proj = u * s
    energy = s ** 2
    return proj, energy / energy.sum() if energy.sum() > 0 else energy


def render_movement_map(proj, energy, ids, movement, args):
    """Heatmap of |movement| per card per movement-PC (rows sorted by total
    movement), with the per-PC energy scree above it."""
    plt = viz.pyplot(show=viz.want_show(args))
    if plt is None:
        print("matplotlib is unavailable — no chart")
        return None
    order = np.argsort(-movement)
    if args.map_top > 0:
        order = order[:args.map_top]
    mag = np.abs(proj[order])
    names = [card_name(ids[j]) for j in order]
    d = proj.shape[1]

    # Adaptive row height/font so the map stays legible from a few dozen rows
    # up to the whole trained vocab.
    row_in = 0.22 if len(order) <= 60 else 0.16
    font = 6 if len(order) <= 60 else 5
    fig, (ax_scree, ax_map) = plt.subplots(
        2, 1, figsize=(16, 3.0 + row_in * len(order)),
        gridspec_kw={"height_ratios": [1, max(3, (row_in / 2.5) * len(order))]},
        sharex=True)
    ax_scree.bar(np.arange(d), energy * 100.0, color="#4878a8")
    ax_scree.plot(np.arange(d), np.cumsum(energy) * 100.0, "k.-", ms=3, lw=0.8)
    ax_scree.set_ylabel("% of movement\nenergy")
    scope = (f"all {len(order)} trained cards" if args.map_top <= 0
             else f"top {len(order)} movers")
    ax_scree.set_title(f"card movement across movement-PCs — {scope}; "
                       "scree = per-PC share (line: cumulative)")

    im = ax_map.imshow(mag, aspect="auto", cmap="magma",
                       interpolation="nearest")
    ax_map.set_yticks(np.arange(len(order)))
    ax_map.set_yticklabels([f"{n}  ({movement[j]:.3f})"
                            for n, j in zip(names, order)], fontsize=font)
    ax_map.set_xticks(np.arange(0, d, 2))
    ax_map.set_xlabel("movement PC (SVD of the delta matrix, uncentered — "
                      "row energies sum to total movement²)")
    fig.colorbar(im, ax=ax_map, label="|Δ along PC|", pad=0.01)
    fig.tight_layout()
    return viz.save_or_show(plt, fig, "az_embed_movement_map", args)


def render_movement_text(proj, energy, ids, top_cards=6, top_pcs=8):
    """Terminal companion to the heatmap: for each leading movement-PC, the
    cards moving furthest along it (signed — opposite signs moved opposite
    ways along the same axis)."""
    lines = [f"movement-PC loadings — top {top_pcs} PCs carry "
             f"{energy[:top_pcs].sum() * 100:.1f}% of movement energy"]
    for k in range(min(top_pcs, proj.shape[1])):
        lead = np.argsort(-np.abs(proj[:, k]))[:top_cards]
        cards = ", ".join(f"{card_name(ids[j])}({proj[j, k]:+.3f})"
                          for j in lead)
        lines.append(f"  PC{k:<2} {energy[k] * 100:5.1f}%  {cards}")
    return lines


def project(mat, method="pca", seed=0, perplexity=30.0):
    """2D coordinates for the rows of ``mat`` plus an axis-title suffix."""
    if method == "pca":
        coords, frac = pca2(mat)
        return coords, (f"PC1 {frac[0] * 100:.1f}%", f"PC2 {frac[1] * 100:.1f}%")
    if method == "tsne":
        from sklearn.manifold import TSNE
        # Cosine metric to match the neighbour/cluster views, which all
        # compare directions, not magnitudes.
        coords = TSNE(n_components=2, metric="cosine", init="pca",
                      perplexity=min(perplexity, (len(mat) - 1) / 3.0),
                      random_state=seed).fit_transform(mat)
        return np.asarray(coords, dtype=np.float64), ("t-SNE 1", "t-SNE 2")
    raise ValueError(f"unknown method {method!r} (pca | tsne)")


def render_chart(coords, ids, movement, labels, axes, args):
    """Scatter the projection: color = label class, size = row movement,
    annotate the most-moved cards. Returns the saved PNG path (or None)."""
    plt = viz.pyplot(show=viz.want_show(args))
    if plt is None:
        print("matplotlib is unavailable — no chart; "
              "use `az_inspect project` for the terminal scatter")
        return None
    fig, ax = plt.subplots(figsize=(13, 9))

    uniq = sorted(set(str(l) for l in labels))
    cmap = plt.get_cmap("tab20" if len(uniq) > 10 else "tab10")
    color_of = {lab: cmap(i % cmap.N) for i, lab in enumerate(uniq)}

    # Area 20..320 pt^2 across the movement range, so the most-trained cards
    # dominate the eye the way they dominated the gradient.
    mv = movement / movement.max() if movement.max() > 0 else movement
    sizes = 20.0 + 300.0 * mv

    for lab in uniq:
        m = np.array([str(l) == lab for l in labels])
        ax.scatter(coords[m, 0], coords[m, 1], s=sizes[m], c=[color_of[lab]],
                   alpha=0.75, edgecolors="none", label=f"{lab} ({m.sum()})")

    for j in np.argsort(-movement)[:max(0, args.label_top)]:
        ax.annotate(card_name(ids[j]), (coords[j, 0], coords[j, 1]),
                    fontsize=7, alpha=0.85,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(axes[0])
    ax.set_ylabel(axes[1])
    ax.set_title(f"AZ card-identity embedding ({args.space} space, "
                 f"{args.method}) — {len(ids)} trained cards; "
                 f"size = row movement since baseline")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    name = f"az_embed_{args.space}_{args.method}_{args.mark}"
    return viz.save_or_show(plt, fig, name, args)


def build_parser():
    ap = argparse.ArgumentParser(
        prog="az_embed_viz",
        description="Chart the AZ card embedding in 2D, excluding cards whose "
                    "row never received gradient.")
    ap.add_argument("--model", default="gen",
                    help="AZ checkpoint spec: 'gen' or a .pt path")
    ap.add_argument("--baseline", default=None,
                    help="checkpoint to measure 'ever trained' against "
                         "(default: the PPO gen warm-start, else the oldest "
                         "local gen__azv* snapshot)")
    ap.add_argument("--view", default="scatter", choices=("scatter", "movement"),
                    help="'scatter': 2D projection of the embedding; "
                         "'movement': per-card heatmap of |Δ| across the "
                         "movement matrix's own PCs (total across all PCs = "
                         "the card's full movement)")
    ap.add_argument("--map-top", type=int, default=0,
                    help="movement view: cards (most-moved first) to map "
                         "(default: 0 = every trained card)")
    ap.add_argument("--space", default="identity", choices=CARD_SPACES,
                    help="card space to project (default: the trainable "
                         "identity table)")
    ap.add_argument("--method", default="pca", choices=("pca", "tsne"))
    ap.add_argument("--mark", default="type", choices=LABEL_KINDS,
                    help="label used for point colors (default: type)")
    ap.add_argument("--label-top", type=int, default=30,
                    help="annotate the N most-moved cards (default: 30)")
    ap.add_argument("--perplexity", type=float, default=30.0,
                    help="t-SNE perplexity (auto-capped for small sets)")
    ap.add_argument("--seed", type=int, default=0, help="t-SNE seed")
    ap.add_argument("--out", default=None,
                    help="output directory (default: train/analysis_out/)")
    ap.add_argument("--show", action="store_true",
                    help="also open an interactive window")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    path = resolve_spec(args.model)
    if path is None:
        raise SystemExit(f"could not resolve AZ checkpoint {args.model!r}")
    baseline = args.baseline or cumulative_baseline(path)
    if baseline is None:
        raise SystemExit(
            "no baseline checkpoint to measure exposure against — need the "
            "PPO gen warm-start or an earlier gen__azv* snapshot "
            "(or pass --baseline)")

    ids, movement, exp = trained_ids(path, baseline=baseline)
    n_named = len(named_card_ids())
    print(f"checkpoint : {path}")
    print(f"baseline   : {exp['baseline']} ({exp['kind']})")
    print(f"trained    : {len(ids)}/{n_named} named cards "
          f"({n_named - len(ids)} never received gradient — excluded)")
    if len(ids) < 3:
        raise SystemExit("fewer than 3 trained cards — nothing to project")

    if args.view == "movement":
        delta = movement_deltas(path, exp["baseline"], ids)
        proj, energy = movement_pcs(delta)
        print("\n".join(render_movement_text(proj, energy, ids)))
        saved = render_movement_map(proj, energy, ids, movement, args)
        return 0 if saved or args.show else 1

    net, _ = load_net(path)
    mat = card_matrix(net, args.space)[ids]
    coords, axes = project(mat, args.method, seed=args.seed,
                           perplexity=args.perplexity)
    saved = render_chart(coords, ids, movement, card_labels(args.mark, ids),
                         axes, args)
    return 0 if saved or args.show else 1


if __name__ == "__main__":
    sys.exit(main())
