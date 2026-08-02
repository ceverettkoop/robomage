"""Static inspection of an AlphaZero checkpoint — weights and recorded self-play,
never a live game.

Every view here answers "what has this net actually learned to value?" from two
engine-free sources:

  * the checkpoint's WEIGHTS (``checkpoints/az/gen__az*.pt`` + its ``.meta.json``),
    loaded through :func:`az_net.load_az` so the obs-layout handshake is checked;
  * the recorded self-play SHARDS (``az_data/gen/shard_*.npz``: ``obs, pi, z,
    mask``), which are real observed states with the search's visit posterior and
    the eventual game result attached.

Neither needs the C++ engine, a subprocess, or a played game — the point is to be
able to say something about the model that is not "it won N% of its games".

The card-identity embedding is the interpretable table: ``trunk.card_emb`` is an
``nn.Embedding(N_CARD_TYPES + 1, card_embed_dim, padding_idx=0)`` whose row
``i + 1`` is vocab card ``i`` (row 0 is the empty-slot padding), so its rows line
up 1:1 with ``src/card_vocab.h`` and can be named, typed, and costed via
``decode``/``card_costs``.

Views (all also exposed as CLI subcommands, ``--help`` on each):

  overview     checkpoint meta, value-bucket coverage, shard availability
  neighbors    cosine nearest neighbours of a card in embedding space
  structure    kNN label purity — how much real card structure the embedding recovers
  clusters     k-means over the embedding, members named
  project      PCA-to-2D terminal scatter, marked by color / type
  occur        how many recorded states contain each card (trained-on counts)
  catemb       full cosine matrix of the small action-category embedding
  buckets      per-matchup critic column map (norm, constant, dead)
  calib        per-bucket value calibration against the shards' outcomes
  divergence   raw-net priors vs the search posterior, by action category
  diff         two checkpoints: per-tensor, per-card and per-bucket movement

Every view returns ``list[str]`` display lines from a ``render_*`` function on top
of a data-only ``compute``-style function, so the TUI (``tui_az_inspect.py``)
renders exactly what the CLI prints.

Run from the repo root:
    train/.venv/bin/python train/az_inspect.py overview
    train/.venv/bin/python train/az_inspect.py neighbors "Lightning Bolt" -k 12
    train/.venv/bin/python train/az_inspect.py buckets
    train/.venv/bin/python train/az_inspect.py calib --max-rows 4000
"""

import argparse
import functools
import glob
import json
import os
import sys

import numpy as np

import archetypes
import decode
from card_costs import (N_CARD_TYPES, _VOCAB_NAMES as VOCAB_NAMES,
                        _CARD_COST_MATRIX as CARD_COST_MATRIX,
                        _LAND_VOCAB_IDS as LAND_VOCAB_IDS)
from _enums import _CAT_NAMES

# Paths. Same layout constants az_net/az_train pin (duplicated deliberately: this
# module must stay importable without dragging in the trainer's torch/sb3 chain).
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
AZ_CKPT_DIR = os.path.join(_TRAIN_DIR, "checkpoints", "az")
AZ_DATA_DIR = os.path.join(_TRAIN_DIR, "az_data", "gen")

# The vocab's sentinel row for "a token permanent" — a real embedding row, but not
# a card, so it stays out of the card-space views.
TOKEN_IDX = decode._TOKEN_IDX

# Supertypes to skip when reducing a type line to its primary card type.
_SUPERTYPES = ("Legendary", "Basic", "Snow", "World", "Ongoing", "Elite", "Host")
_PRIMARY_TYPES = ("Land", "Creature", "Instant", "Sorcery", "Artifact",
                  "Enchantment", "Planeswalker", "Battle", "Kindred", "Tribal")

_COLOR_LETTERS = "WUBRG"


# ----------------------------------------------------------------------
# Vocab-side labels (names, colors, types, costs)
# ----------------------------------------------------------------------

def named_card_ids():
    """Vocab indices that name a real card (non-empty name, excluding the token
    sentinel) — the only rows whose embedding is meaningful to compare."""
    return np.array([i for i, n in enumerate(VOCAB_NAMES)
                     if n and i != TOKEN_IDX], dtype=np.int64)


def card_name(idx):
    return decode.card_index_to_name(int(idx)) or f"#{int(idx)}"


def resolve_card(query):
    """Vocab index for a card query: exact name wins (case/punctuation-insensitive),
    else a unique substring. Raises with the candidate list when ambiguous."""
    def norm(s):
        return "".join(c for c in str(s).lower() if c.isalnum())

    q = norm(query)
    if not q:
        raise ValueError("empty card query")
    exact = [i for i, n in enumerate(VOCAB_NAMES) if n and norm(n) == q]
    if exact:
        return exact[0]
    subs = [i for i, n in enumerate(VOCAB_NAMES) if n and q in norm(n)]
    if len(subs) == 1:
        return subs[0]
    if not subs:
        raise ValueError(f"no vocab card matches {query!r} "
                         "(cards absent from src/card_vocab.h are unimplemented)")
    names = ", ".join(VOCAB_NAMES[i] for i in subs[:12])
    more = "" if len(subs) <= 12 else f" (+{len(subs) - 12} more)"
    raise ValueError(f"{query!r} is ambiguous: {names}{more}")


def card_cmc(idx):
    """Mana value from the generated cost matrix (pip counts are stored /10)."""
    return float(round(CARD_COST_MATRIX[int(idx)].sum() * 10))


def card_cost_colors(idx):
    """Colors appearing in the card's MANA COST as a 'WUR'-style string ('C' when
    costless/colorless). Cost colors, not full color identity — the cost matrix is
    what the engine actually feeds the net."""
    row = CARD_COST_MATRIX[int(idx)]
    letters = "".join(_COLOR_LETTERS[c] for c in range(5) if row[c] > 0)
    return letters or "C"


@functools.lru_cache(maxsize=None)
def card_primary_type(idx):
    """Primary card type from the card's Forge script ('Creature', 'Land', ...).

    Cached: decode.card_types re-reads the script file on every call."""
    if int(idx) in LAND_VOCAB_IDS:
        # Trust the generated land set even when a script lookup fails.
        fallback = "Land"
    else:
        fallback = "?"
    line = decode.card_types(int(idx)) or ""
    for tok in line.replace("—", " ").split():
        if tok in _SUPERTYPES:
            continue
        if tok in _PRIMARY_TYPES:
            return tok
    return fallback


def card_labels(kind, ids=None):
    """Label array over ``ids`` (default: every named card) for a label ``kind``:
    ``color`` (cost colors), ``type`` (primary type), ``cmc`` (mana-value bucket),
    ``land`` (land / nonland). These are the ground-truth structure the embedding
    either recovered or did not."""
    ids = named_card_ids() if ids is None else np.asarray(ids)
    if kind == "color":
        return np.array([card_cost_colors(i) for i in ids])
    if kind == "type":
        return np.array([card_primary_type(i) for i in ids])
    if kind == "cmc":
        def bucket(i):
            v = card_cmc(i)
            return "0-1" if v <= 1 else "2-3" if v <= 3 else "4-5" if v <= 5 else "6+"
        return np.array([bucket(i) for i in ids])
    if kind == "land":
        return np.array(["land" if int(i) in LAND_VOCAB_IDS else "nonland"
                         for i in ids])
    raise ValueError(f"unknown label kind {kind!r} "
                     "(color | type | cmc | land)")


LABEL_KINDS = ("color", "type", "cmc", "land")


# ----------------------------------------------------------------------
# Checkpoint loading + embedding tables
# ----------------------------------------------------------------------

def resolve_spec(spec="gen", checkpoint_dir=AZ_CKPT_DIR, prefer="final"):
    """Path for an AZ checkpoint spec, or None.

    az_net.resolve_az_checkpoint accepts 'gen' and existing ``.pt`` paths; the
    snapshot views here are routinely pointed at ONE specific snapshot (diffing
    two of them is the whole point of the diff view), so a bare snapshot name or
    stem inside the checkpoint dir resolves too: ``gen__azv128000``."""
    from az_net import resolve_az_checkpoint
    path = resolve_az_checkpoint(spec, checkpoint_dir=checkpoint_dir,
                                 prefer=prefer)
    if path is not None:
        return path
    for cand in (spec, spec + ".pt"):
        p = os.path.join(checkpoint_dir, os.path.basename(str(cand)))
        if p.endswith(".pt") and os.path.isfile(p):
            return p
    return None


def load_net(spec="gen", checkpoint_dir=AZ_CKPT_DIR, prefer="final"):
    """Resolve an AZ checkpoint spec and load it. Returns ``(net, path)``.

    torch is imported lazily so the vocab-side helpers above stay importable in a
    torch-free environment."""
    from az_net import load_az
    path = resolve_spec(spec, checkpoint_dir=checkpoint_dir, prefer=prefer)
    if path is None:
        raise FileNotFoundError(
            f"no AZ checkpoint for {spec!r} in {checkpoint_dir} — run "
            "'train.py az' (which warm-starts from the PPO gen checkpoint) first")
    return load_az(path), path


def checkpoint_meta(path):
    """The ``.meta.json`` handshake dict written beside a checkpoint ({} if absent)."""
    base, _ = os.path.splitext(path)
    mp = base + ".meta.json"
    if not os.path.isfile(mp):
        return {}
    with open(mp) as f:
        return json.load(f)


def card_embedding(net):
    """The card-identity embedding as ``(N_CARD_TYPES, D)``, row ``i`` = vocab card
    ``i``. The stored table has ``N_CARD_TYPES + 1`` rows — row 0 is the padding
    row for empty slots — so this drops it and re-bases the index."""
    w = net.trunk.card_emb.weight.detach().cpu().numpy()
    return np.array(w[1:], dtype=np.float64)


def category_embedding(net):
    """The action-category embedding as ``(ACTION_CATEGORY_MAX + 1, D)``."""
    return np.array(net.trunk.action_cat_emb.weight.detach().cpu().numpy(),
                    dtype=np.float64)


def _unit(mat):
    """Row-normalized copy; all-zero rows stay zero (cosine with them is 0)."""
    n = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(n > 0, n, 1.0)


def nearest_cards(mat, idx, k=15, candidates=None):
    """Top-``k`` cosine neighbours of vocab card ``idx``. Returns
    ``[(vocab_idx, cosine), ...]``, excluding the query itself."""
    cand = named_card_ids() if candidates is None else np.asarray(candidates)
    cand = cand[cand != int(idx)]
    u = _unit(mat)
    sims = u[cand] @ u[int(idx)]
    order = np.argsort(-sims)[:k]
    return [(int(cand[j]), float(sims[j])) for j in order]


# ----------------------------------------------------------------------
# Embedding structure: kNN purity, k-means, PCA
# ----------------------------------------------------------------------

def knn_purity(mat, labels, ids, k=10):
    """Fraction of each card's ``k`` nearest neighbours sharing its label, against
    the chance baseline (the probability two random cards share a label).

    This is the quantitative form of "did the embedding recover real card
    structure": purity at/below baseline means the axis is not encoded at all."""
    ids = np.asarray(ids)
    labels = np.asarray(labels)
    u = _unit(mat)[ids]
    sims = u @ u.T
    np.fill_diagonal(sims, -np.inf)
    k = int(min(k, len(ids) - 1))
    nn = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    share = (labels[nn] == labels[:, None]).mean(axis=1)
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    baseline = float((p ** 2).sum())
    per_label = {}
    for lab in np.unique(labels):
        m = labels == lab
        per_label[str(lab)] = (int(m.sum()), float(share[m].mean()))
    return {"k": k, "n": int(len(ids)), "purity": float(share.mean()),
            "baseline": baseline, "per_label": per_label,
            "lift": float(share.mean() - baseline)}


def kmeans(mat, k, seed=0, iters=60):
    """Plain k-means++ over ``mat`` (numpy only — no sklearn dependency).
    Returns ``(assignments, centers)``."""
    rng = np.random.default_rng(seed)
    x = _unit(mat)                      # cosine space: cluster on directions
    n = x.shape[0]
    k = int(min(k, n))
    centers = np.empty((k, x.shape[1]))
    centers[0] = x[rng.integers(n)]
    d2 = ((x - centers[0]) ** 2).sum(axis=1)
    for c in range(1, k):
        probs = d2 / d2.sum() if d2.sum() > 0 else np.full(n, 1.0 / n)
        centers[c] = x[rng.choice(n, p=probs)]
        d2 = np.minimum(d2, ((x - centers[c]) ** 2).sum(axis=1))
    assign = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = dist.argmin(axis=1)
        if np.array_equal(new, assign):
            break
        assign = new
        for c in range(k):
            m = assign == c
            if m.any():
                centers[c] = x[m].mean(axis=0)
    return assign, centers


def pca2(mat):
    """Project to 2D via SVD. Returns ``(coords (n,2), explained_fraction (2,))``."""
    x = mat - mat.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    coords = u[:, :2] * s[:2]
    var = (s ** 2)
    frac = var[:2] / var.sum() if var.sum() > 0 else np.zeros(2)
    return coords, frac


# ----------------------------------------------------------------------
# Self-play shards
# ----------------------------------------------------------------------

def shard_paths(data_dir=AZ_DATA_DIR):
    """Recorded self-play shards, oldest first (mtime order, as az_train windows)."""
    return sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")),
                  key=os.path.getmtime)


def load_shard_sample(data_dir=AZ_DATA_DIR, max_rows=4000, window=None, seed=0):
    """Load a bounded RANDOM SAMPLE of recorded decisions.

    The full pool is gigabytes of float32 observations, so this walks the shards
    newest-first and takes an even per-shard quota, keeping the sample spread over
    the window rather than concentrated in whichever shard happened to be biggest.
    Returns a dict with ``obs, pi, z, mask, n_shards, n_rows_total``.
    """
    from env import OBS_SIZE, MAX_ACTIONS
    paths = shard_paths(data_dir)
    if not paths:
        raise FileNotFoundError(
            f"no self-play shards in {data_dir} — run 'train.py az' (self-play "
            "writes shard_*.npz there) before the shard-backed views")
    paths = list(reversed(paths))                 # newest first
    if window:
        paths = paths[:int(window)]
    rng = np.random.default_rng(seed)
    quota = max(1, int(max_rows) // len(paths))
    obs, pi, z, mask = [], [], [], []
    total = 0
    used = 0
    for p in paths:
        d = np.load(p)
        o = d["obs"]
        if o.shape[1] != OBS_SIZE or d["mask"].shape[1] != MAX_ACTIONS:
            raise RuntimeError(
                f"{os.path.basename(p)} was recorded against a different "
                f"observation layout (obs width {o.shape[1]}, mask "
                f"{d['mask'].shape[1]}) but this build has OBS_SIZE={OBS_SIZE}, "
                f"MAX_ACTIONS={MAX_ACTIONS} — regenerate the shards")
        total += o.shape[0]
        take = min(quota, o.shape[0])
        sel = np.sort(rng.choice(o.shape[0], size=take, replace=False))
        obs.append(o[sel]); pi.append(d["pi"][sel])
        z.append(d["z"][sel]); mask.append(d["mask"][sel])
        used += 1
        if sum(a.shape[0] for a in obs) >= max_rows:
            break
    out = {"obs": np.concatenate(obs), "pi": np.concatenate(pi),
           "z": np.concatenate(z), "mask": np.concatenate(mask),
           "n_shards": used, "n_rows_total": total}
    if out["obs"].shape[0] > max_rows:
        keep = np.sort(rng.choice(out["obs"].shape[0], size=max_rows,
                                  replace=False))
        for key in ("obs", "pi", "z", "mask"):
            out[key] = out[key][keep]
    return out


def _names_in_state(gs):
    """Every card name mentioned by a decoded game state (both boards, both
    graveyards/exiles, hand, stack, known top-of-library, known opponent hand)."""
    names = set()
    for key in ("self_battlefield", "opp_battlefield", "self_hand", "stack",
                "known_top_library", "opp_known_hand"):
        for item in gs.get(key) or ():
            n = item.get("name") if isinstance(item, dict) else item
            if n:
                names.add(n)
    for key in ("self_graveyard", "opp_graveyard", "self_exile", "opp_exile",
                "opp_revealed"):
        for n in gs.get(key) or ():
            if isinstance(n, str) and n:
                names.add(n)
    return names


def card_occurrences(obs, limit=1500, seed=0):
    """Per-card count of sampled states that MENTION the card, via the shared
    board decoder (layout-proof, unlike hand-counting id offsets).

    This is what keeps the embedding views honest: a card with no occurrences got
    no training signal, so its embedding row is warm-start noise and its
    "neighbours" mean nothing. Returns ``(counts (N_CARD_TYPES,), n_states)``.
    """
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    idx = np.arange(n) if n <= limit else np.sort(
        rng.choice(n, size=int(limit), replace=False))
    name_to_id = {n_: i for i, n_ in enumerate(VOCAB_NAMES) if n_}
    counts = np.zeros(N_CARD_TYPES, dtype=np.int64)
    for r in idx:
        gs = decode.decode_game_state(obs[r])
        for nm in _names_in_state(gs):
            i = name_to_id.get(nm)
            if i is not None:
                counts[i] += 1
    return counts, int(len(idx))


# ----------------------------------------------------------------------
# Critic: bucket map, calibration; policy divergence
# ----------------------------------------------------------------------

def bucket_table(net):
    """One row per (self archetype x opp archetype) value bucket: the critic
    column's weight norm, its bias, the constant it degenerates to when dead, and
    the dead flag from AZNet.dead_value_buckets."""
    w = net.value_head.weight.detach().cpu().numpy()
    b = net.value_head.bias.detach().cpu().numpy()
    dead = net.dead_value_buckets()
    rows = []
    for i in range(w.shape[0]):
        rows.append({
            "bucket": i,
            "name": archetypes.bucket_name(i),
            "norm": float(np.linalg.norm(w[i])),
            "bias": float(b[i]),
            "const": float(np.tanh(b[i])),
            "dead": bool(dead[i]),
        })
    return rows


def predict(net, obs, mask, batch=256):
    """Batched ``(values, priors)`` for recorded rows. ``priors`` is the softmax
    over each row's LEGAL actions (illegal slots left at 0)."""
    import torch
    vals = np.zeros(obs.shape[0], dtype=np.float64)
    priors = np.zeros(mask.shape, dtype=np.float64)
    net.eval()
    with torch.no_grad():
        for s in range(0, obs.shape[0], batch):
            e = min(s + batch, obs.shape[0])
            ob = torch.as_tensor(np.ascontiguousarray(obs[s:e]))
            mk = torch.as_tensor(np.ascontiguousarray(mask[s:e]))
            logits, value = net(ob, mk)
            p = torch.softmax(logits, dim=-1)
            p = torch.where(mk, p, torch.zeros_like(p))
            vals[s:e] = value.cpu().numpy()
            priors[s:e] = p.cpu().numpy()
    return vals, priors


def obs_buckets(net, obs):
    """The value bucket each observation selects (same round+clamp forward does)."""
    return np.array([net.obs_value_bucket(o) for o in obs], dtype=np.int64)


def bucket_calibration(net, sample, bins=6):
    """Predicted V vs the shards' realized outcome z, per value bucket.

    Answers "is this matchup's critic honest?" with no games played: the shards
    already carry the eventual result of the game each state came from."""
    obs, z, mask = sample["obs"], sample["z"], sample["mask"]
    vals, _ = predict(net, obs, mask)
    buckets = obs_buckets(net, obs)
    dead = net.dead_value_buckets()
    rows = []
    for b in np.unique(buckets):
        m = buckets == b
        v, zz = vals[m], z[m]
        edges = np.linspace(-1, 1, bins + 1)
        rel = []
        for i in range(bins):
            # Half-open bins, except the last which must include the +1 endpoint.
            upper = (v <= edges[i + 1]) if i == bins - 1 else (v < edges[i + 1])
            sel = (v >= edges[i]) & upper
            if sel.any():
                rel.append((float(edges[i]), float(edges[i + 1]), int(sel.sum()),
                            float(v[sel].mean()), float(zz[sel].mean())))
        rows.append({
            "bucket": int(b), "name": archetypes.bucket_name(int(b)),
            "n": int(m.sum()), "mse": float(((v - zz) ** 2).mean()),
            "mean_pred": float(v.mean()), "mean_z": float(zz.mean()),
            "sign_agree": float((np.sign(v) == np.sign(zz))[zz != 0].mean())
                          if (zz != 0).any() else float("nan"),
            "dead": bool(dead[int(b)]), "reliability": rel,
        })
    rows.sort(key=lambda r: -r["n"])
    return {"rows": rows, "n": int(obs.shape[0]),
            "overall_mse": float(((vals - z) ** 2).mean())}


def policy_divergence(net, sample, top_n=12):
    """KL(search posterior || raw-net priors) grouped by the action CATEGORY the
    search preferred — where the net has and has not internalized search.

    A category with high KL is one the raw net cannot reproduce without search;
    that is the concrete answer to "what does it still not understand".
    """
    obs, pi, mask = sample["obs"], sample["pi"], sample["mask"]
    _, priors = predict(net, obs, mask)
    from env import ACT_CATS_START, MAX_ACTIONS
    from _enums import ACTION_CATEGORY_MAX
    cats = np.round(obs[:, ACT_CATS_START:ACT_CATS_START + MAX_ACTIONS]
                    * ACTION_CATEGORY_MAX).astype(int)

    eps = 1e-9
    tot = pi.sum(axis=1)
    ok = tot > 0
    p = np.where(mask, pi, 0.0)
    p = p / np.where(tot[:, None] > 0, tot[:, None], 1.0)
    q = np.clip(priors, eps, None)
    kl = np.where(p > 0, p * (np.log(np.clip(p, eps, None)) - np.log(q)), 0.0)
    kl = kl.sum(axis=1)
    best_pi = p.argmax(axis=1)
    best_q = np.where(mask, priors, -1.0).argmax(axis=1)
    agree = best_pi == best_q
    n_legal = mask.sum(axis=1)

    groups = {}
    for r in np.nonzero(ok)[0]:
        c = int(cats[r, best_pi[r]])
        g = groups.setdefault(c, {"kl": [], "agree": [], "legal": []})
        g["kl"].append(kl[r]); g["agree"].append(agree[r])
        g["legal"].append(n_legal[r])
    rows = []
    for c, g in groups.items():
        rows.append({"category": c, "name": _CAT_NAMES.get(c, f"?({c})"),
                     "n": len(g["kl"]), "kl": float(np.mean(g["kl"])),
                     "top1": float(np.mean(g["agree"])),
                     "legal": float(np.mean(g["legal"]))})
    rows.sort(key=lambda r: -r["kl"])
    return {"rows": rows[:top_n], "all_rows": rows,
            "n": int(ok.sum()),
            "kl": float(kl[ok].mean()), "top1": float(agree[ok].mean())}


# ----------------------------------------------------------------------
# Probes — what a single evaluation actually rests on
# ----------------------------------------------------------------------

def obs_blocks():
    """Partition of the observation vector into named blocks:
    ``[(name, start, end), ...]``, contiguous and covering ``[0, OBS_SIZE)``.

    Derived entirely from env.py's offset chain (never a literal), and asserted
    contiguous so a layout change fails here rather than silently mislabelling
    every attribution below."""
    import env as e
    blocks = [
        ("self player", e._SELF_BLOCK_START, e._OPP_BLOCK_START),
        ("opp player", e._OPP_BLOCK_START, e._STEP_ONEHOT_START),
        ("step one-hot", e._STEP_ONEHOT_START, e._IS_ACTIVE_IDX),
        ("header flags", e._IS_ACTIVE_IDX, e._GLOBAL_SIZE),
        ("self battlefield", e._SELF_PERM_START, e._OPP_PERM_START),
        ("opp battlefield", e._OPP_PERM_START, e._STACK_START),
        ("stack", e._STACK_START, e._GY_START),
        ("graveyards", e._GY_START, e._EXILE_START),
        ("exiles", e._EXILE_START, e._HAND_START),
        ("hand", e._HAND_START, e._HIST_START),
        ("action history", e._HIST_START, e._HIST_END),
        ("match context", e._MATCH_CTX_START, e._LIBRARY_CTX_START),
        ("library context", e._LIBRARY_CTX_START, e._CUR_TURN_IDX),
        ("turn", e._CUR_TURN_IDX, e._KNOWN_TOP_LIB_START),
        ("known top library", e._KNOWN_TOP_LIB_START, e._REVEALED_START),
        ("opp revealed", e._REVEALED_START, e._OPP_KNOWN_HAND_START),
        ("opp known hand", e._OPP_KNOWN_HAND_START, e._PENDING_DECISION_START),
        ("pending decision", e._PENDING_DECISION_START, e._EXTRAS_START),
        ("global extras", e._EXTRAS_START, e.STATE_SIZE),
        ("action categories", e.ACT_CATS_START, e.ACT_CATS_START + e.MAX_ACTIONS),
        ("action card ids", e.ACT_IDS_START, e.ACT_IDS_START + e.MAX_ACTIONS),
        ("action controllers", e.ACT_CTRL_START, e.ACT_CTRL_START + e.MAX_ACTIONS),
        ("action zones", e.ACT_ZONE_START, e.ACT_ZONE_START + e.MAX_ACTIONS),
        ("action refs", e.ACT_REFS_START, e.ACT_REFS_START + e.MAX_ACTIONS),
        ("action ordinals", e.ACT_ORDS_START, e.ACT_ORDS_START + e.MAX_ACTIONS),
        ("hand cost feats", e.ACT_BLOCKS_END, e._BF_COST_START),
        ("battlefield cost feats", e._BF_COST_START, e.MATCHUP_TAIL_START),
        ("matchup tail", e.MATCHUP_TAIL_START, e.OBS_SIZE),
    ]
    blocks = [(n, int(s), int(t)) for n, s, t in blocks]
    blocks.sort(key=lambda b: b[1])
    pos = 0
    for name, s, t in blocks:
        if s != pos or t <= s:
            raise RuntimeError(
                f"observation block table is not contiguous at {name!r} "
                f"({s}..{t}, expected to start at {pos}) — env.py's offset chain "
                "changed shape; update obs_blocks()")
        pos = t
    if pos != int(e.OBS_SIZE):
        raise RuntimeError(f"observation block table covers {pos} floats but "
                           f"OBS_SIZE is {e.OBS_SIZE}")
    return blocks


def state_value(net, obs_row, mask_row):
    """V for a single observation, in the current mover's perspective."""
    v, _ = predict(net, np.asarray(obs_row)[None], np.asarray(mask_row)[None])
    return float(v[0])


def block_importance(net, sample, n_rows=150, donors=3, seed=0, blocks=None):
    """Permutation importance of each observation block on V.

    For each block, the block's floats are replaced by another sampled state's
    (a valid value for that block, unlike zeroing, which invents states the net
    never saw — a 0 life total is not "no information", it is "dead"), and the
    mean |ΔV| is recorded. High = the evaluation leans on that block.
    """
    blocks = blocks or obs_blocks()
    rng = np.random.default_rng(seed)
    obs, mask = sample["obs"], sample["mask"]
    n = min(int(n_rows), obs.shape[0])
    rows = np.sort(rng.choice(obs.shape[0], size=n, replace=False))
    base, _ = predict(net, obs[rows], mask[rows])
    out = []
    for name, s, t in blocks:
        acc = np.zeros(n)
        for _ in range(int(donors)):
            donor = rng.choice(obs.shape[0], size=n)
            pert = obs[rows].copy()
            pert[:, s:t] = obs[donor][:, s:t]
            v, _ = predict(net, pert, mask[rows])
            acc += np.abs(v - base)
        out.append({"name": name, "start": s, "width": t - s,
                    "delta": float((acc / donors).mean())})
    out.sort(key=lambda r: -r["delta"])
    return {"rows": out, "n": n, "donors": int(donors),
            "base_spread": float(base.std())}


def state_block_importance(net, sample, row, donors=16, seed=0, blocks=None):
    """Permutation importance for ONE recorded state — which blocks this
    particular evaluation rests on."""
    blocks = blocks or obs_blocks()
    rng = np.random.default_rng(seed)
    obs, mask = sample["obs"], sample["mask"]
    base = state_value(net, obs[row], mask[row])
    batch = np.repeat(obs[row][None], donors, axis=0)
    mk = np.repeat(mask[row][None], donors, axis=0)
    out = []
    for name, s, t in blocks:
        donor = rng.choice(obs.shape[0], size=donors)
        pert = batch.copy()
        pert[:, s:t] = obs[donor][:, s:t]
        v, _ = predict(net, pert, mk)
        out.append({"name": name, "start": s, "width": t - s,
                    "delta": float(np.abs(v - base).mean()),
                    "signed": float((v - base).mean())})
    out.sort(key=lambda r: -r["delta"])
    return {"rows": out, "base": base, "donors": int(donors)}


def card_id_sites(obs_row):
    """Every swappable card-identity float in a state:
    ``[(label, offset, card_idx), ...]`` over both battlefields and the hand."""
    import env as e
    sites = []
    for tag, start in (("self bf", e._SELF_PERM_START),
                       ("opp bf", e._OPP_PERM_START)):
        for s in range(e._PERM_SLOTS):
            off = start + s * e._PERM_SLOT_SIZE + e._PERM_CARD_OFF
            idx = int(round(float(obs_row[off]) * N_CARD_TYPES))
            if idx >= 0:
                sites.append((f"{tag} slot {s}: {card_name(idx)}", off, idx))
    for s in range(e._HAND_SLOTS_TOTAL):
        off = e._HAND_START + s * e._HAND_SLOT_SIZE
        idx = int(round(float(obs_row[off]) * N_CARD_TYPES))
        if idx >= 0:
            sites.append((f"hand {s}: {card_name(idx)}", off, idx))
    return sites


def card_swap_probe(net, obs_row, mask_row, offset, candidates=None, batch=256):
    """Replace the card identity at ``offset`` with each candidate card and
    measure ΔV — a grounded per-position card valuation.

    Only the IDENTITY float moves: the slot's status floats (power/toughness,
    tapped, counters, refs) stay as they are, so this asks "what if the net
    believed this object were card X", not "what if card X were played here".
    """
    cand = named_card_ids() if candidates is None else np.asarray(candidates)
    base = state_value(net, obs_row, mask_row)
    rows = np.repeat(np.asarray(obs_row, dtype=np.float32)[None], len(cand),
                     axis=0)
    rows[:, offset] = cand / float(N_CARD_TYPES)
    mk = np.repeat(np.asarray(mask_row)[None], len(cand), axis=0)
    vals, _ = predict(net, rows, mk, batch=batch)
    order = np.argsort(-vals)
    return {"base": base,
            "rows": [(int(cand[i]), float(vals[i]), float(vals[i] - base))
                     for i in order]}


def sweep_fields():
    """Sweepable scalar fields: ``name -> (obs index, scale, values)``.

    ``scale`` is the engine's normalizer, so ``obs[idx] = value / scale``."""
    import env as e
    return {
        "self_life": (e._SELF_BLOCK_START + e._PB_LIFE, 20.0,
                      list(range(0, 21))),
        "opp_life": (e._OPP_BLOCK_START + e._PB_LIFE, 20.0,
                     list(range(0, 21))),
        "self_hand": (e._SELF_BLOCK_START + e._PB_HAND_CT, 10.0,
                      list(range(0, 11))),
        "opp_hand": (e._OPP_BLOCK_START + e._PB_HAND_CT, 10.0,
                     list(range(0, 11))),
        "turn": (e._CUR_TURN_IDX, 50.0, list(range(0, 26))),
    }


def sweep(net, obs_row, mask_row, field):
    """V as one scalar field is swept over its range — the monotonicity sanity
    check ("does its own life total falling make it less happy?")."""
    idx, scale, values = sweep_fields()[field]
    rows = np.repeat(np.asarray(obs_row, dtype=np.float32)[None], len(values),
                     axis=0)
    rows[:, idx] = np.array(values, dtype=np.float32) / scale
    mk = np.repeat(np.asarray(mask_row)[None], len(values), axis=0)
    vals, _ = predict(net, rows, mk)
    return {"field": field, "values": list(values),
            "v": [float(x) for x in vals],
            "current": float(round(float(obs_row[idx]) * scale))}


# Fields whose value SHOULD move V in a known direction, and that direction.
# Not laws of the game — a state can be won regardless of a life total — but a
# net that trends the wrong way across the whole sweep is worth knowing about.
_SWEEP_EXPECTED = {"self_life": +1, "opp_life": -1}


def sweep_trend(res):
    """Spearman-free trend summary: net rise across the sweep, and whether it
    agrees with the expected direction (None when there is no expectation)."""
    v = np.asarray(res["v"])
    rise = float(v[-1] - v[0])
    want = _SWEEP_EXPECTED.get(res["field"])
    ok = None if want is None else (rise * want >= 0)
    return {"rise": rise, "expected": want, "agrees": ok,
            "span": float(v.max() - v.min())}


# ----------------------------------------------------------------------
# Checkpoint diff
# ----------------------------------------------------------------------

def checkpoint_diff(path_a, path_b, top_n=15):
    """Per-tensor, per-card and per-bucket movement between two checkpoints —
    "what did the last training rotation actually change"."""
    import torch
    sa = torch.load(path_a, map_location="cpu")
    sb = torch.load(path_b, map_location="cpu")
    shared = [k for k in sa if k in sb and sa[k].shape == sb[k].shape]
    only_a = sorted(k for k in sa if k not in sb)
    only_b = sorted(k for k in sb if k not in sa)
    shape_diff = sorted(k for k in sa if k in sb and sa[k].shape != sb[k].shape)

    tensors = []
    for k in shared:
        d = (sb[k] - sa[k]).float()
        base = sa[k].float().norm().item()
        tensors.append({"name": k, "delta": float(d.norm()),
                        "base": base,
                        "rel": float(d.norm() / base) if base > 0 else float("nan")})
    tensors.sort(key=lambda t: -t["rel"] if np.isfinite(t["rel"]) else 0.0)

    cards = []
    ck = "trunk.card_emb.weight"
    if ck in shared:
        d = (sb[ck] - sa[ck]).float().norm(dim=1).numpy()[1:]   # drop padding row
        for i in np.argsort(-d)[:top_n]:
            if VOCAB_NAMES[i]:
                cards.append((int(i), card_name(i), float(d[i])))

    buckets = []
    vk = "value_head.weight"
    if vk in shared:
        d = (sb[vk] - sa[vk]).float().norm(dim=1).numpy()
        for i in np.argsort(-d)[:top_n]:
            if d[i] > 0:
                buckets.append((int(i), archetypes.bucket_name(int(i)), float(d[i])))

    return {"a": path_a, "b": path_b, "tensors": tensors, "cards": cards,
            "buckets": buckets, "only_a": only_a, "only_b": only_b,
            "shape_diff": shape_diff}


# ----------------------------------------------------------------------
# Renderers — every view's display lines, shared by the CLI and the TUI
# ----------------------------------------------------------------------

def _bar(frac, width=12, ch="█"):
    frac = 0.0 if not np.isfinite(frac) else max(0.0, min(1.0, float(frac)))
    n = int(round(frac * width))
    return ch * n + "·" * (width - n)


def render_overview(net, path, sample=None):
    meta = checkpoint_meta(path)
    rows = bucket_table(net)
    live = [r for r in rows if not r["dead"]]
    lines = [f"checkpoint : {path}",
             f"steps      : {meta.get('steps', '?')}   embed_dim="
             f"{meta.get('embed_dim', '?')}  obs_size={meta.get('obs_size', '?')}",
             f"vocab      : {len(named_card_ids())} named cards of "
             f"{N_CARD_TYPES} slots",
             f"critic     : {len(live)}/{len(rows)} value buckets alive "
             f"({len(rows) - len(live)} dead — constant value in that matchup)"]
    top = sorted(live, key=lambda r: -r["norm"])[:8]
    if top:
        lines.append("  strongest columns: "
                     + ", ".join(f"{r['name']}({r['norm']:.2f})" for r in top))
    paths = shard_paths()
    lines.append(f"shards     : {len(paths)} in {AZ_DATA_DIR}"
                 + (f" (newest {os.path.basename(paths[-1])})" if paths else ""))
    if sample is not None:
        lines.append(f"sample     : {sample['obs'].shape[0]} decisions from "
                     f"{sample['n_shards']} shard(s)")
    return lines


NEIGHBOR_HEADER = f"  {'cos':>6}  {'card':<34} {'cost':<6} {'type':<13} seen"


def neighbor_row(idx, cos, counts=None):
    """One formatted neighbour line. Shared so the TUI's clickable list and the
    text view show identical columns."""
    seen = "" if counts is None else str(int(counts[int(idx)]))
    return (f"  {cos:6.3f}  {card_name(idx)[:34]:<34} "
            f"{card_cost_colors(idx):<6} {card_primary_type(idx):<13} {seen}")


def render_neighbors(mat, idx, k=15, counts=None, candidates=None, rows=True):
    """Header + neighbour rows for a card. ``rows=False`` returns the header
    alone, for callers (the TUI) that render the neighbours as a widget."""
    lines = [f"{card_name(idx)}  [{card_cost_colors(idx)} "
             f"mv{card_cmc(idx):.0f} {card_primary_type(idx)}]",
             f"  embedding row norm {np.linalg.norm(mat[idx]):.3f}"]
    if counts is not None:
        seen = int(counts[idx])
        lines.append(f"  seen in {seen} sampled states"
                     + ("  ⚠ NEVER SEEN — its row is warm-start noise, so its "
                        "neighbours are meaningless" if seen == 0 else ""))
    if not rows:
        return lines
    lines.append("")
    lines.append(NEIGHBOR_HEADER)
    for j, cos in nearest_cards(mat, idx, k=k, candidates=candidates):
        lines.append(neighbor_row(j, cos, counts))
    return lines


def render_structure(mat, k=10, ids=None, counts=None, min_seen=0):
    ids = named_card_ids() if ids is None else np.asarray(ids)
    note = ""
    if counts is not None and min_seen > 0:
        keep = np.array([counts[i] >= min_seen for i in ids])
        note = (f" (restricted to {int(keep.sum())} cards seen >= {min_seen} "
                f"times of {len(ids)})")
        ids = ids[keep]
    lines = [f"kNN label purity over {len(ids)} card embeddings, k={k}{note}",
             "  purity = share of a card's k nearest neighbours with its label;",
             "  baseline = chance purity from the label distribution alone.", ""]
    if len(ids) < k + 2:
        lines.append("  too few cards to score — widen the filter")
        return lines
    lines.append(f"  {'label':<8} {'purity':>7} {'chance':>7} {'lift':>7}  ")
    for kind in LABEL_KINDS:
        res = knn_purity(mat, card_labels(kind, ids), ids, k=k)
        lines.append(f"  {kind:<8} {res['purity']:7.3f} {res['baseline']:7.3f} "
                     f"{res['lift']:+7.3f}  {_bar(res['purity'])}")
    lines.append("")
    res = knn_purity(mat, card_labels("type", ids), ids, k=k)
    lines.append("  by primary type:")
    for lab, (n, pur) in sorted(res["per_label"].items(), key=lambda t: -t[1][0]):
        lines.append(f"    {lab:<14} n={n:<4} purity={pur:5.3f}  {_bar(pur)}")
    return lines


def render_clusters(mat, k=8, seed=0, ids=None, per_cluster=12, counts=None,
                    min_seen=0):
    ids = named_card_ids() if ids is None else np.asarray(ids)
    if counts is not None and min_seen > 0:
        ids = ids[np.array([counts[i] >= min_seen for i in ids])]
    if len(ids) < k:
        return [f"only {len(ids)} cards available — fewer than k={k}"]
    assign, _ = kmeans(mat[ids], k, seed=seed)
    lines = [f"k-means (k={k}, seed={seed}) over {len(ids)} card embeddings", ""]
    for c in range(k):
        members = ids[assign == c]
        if not len(members):
            continue
        colors = [card_cost_colors(i) for i in members]
        types = [card_primary_type(i) for i in members]
        top_color = max(set(colors), key=colors.count)
        top_type = max(set(types), key=types.count)
        lines.append(f"cluster {c}  n={len(members)}  mostly {top_color}/"
                     f"{top_type} ({types.count(top_type)}/{len(members)})")
        shown = [card_name(i) for i in members[:per_cluster]]
        more = "" if len(members) <= per_cluster else f" … +{len(members) - per_cluster}"
        lines.append("   " + ", ".join(shown) + more)
    return lines


def render_projection(mat, ids=None, width=78, height=24, mark="color",
                      counts=None, min_seen=0):
    """PCA-to-2D terminal scatter. Each point is a single character: the first
    letter of its color (or type) label, '*' where points collide."""
    ids = named_card_ids() if ids is None else np.asarray(ids)
    if counts is not None and min_seen > 0:
        ids = ids[np.array([counts[i] >= min_seen for i in ids])]
    if len(ids) < 3:
        return ["not enough cards to project"]
    coords, frac = pca2(mat[ids])
    labels = card_labels(mark, ids)
    grid = [[" "] * width for _ in range(height)]
    xs, ys = coords[:, 0], coords[:, 1]
    sx = (xs - xs.min()) / (np.ptp(xs) or 1.0) * (width - 1)
    sy = (ys - ys.min()) / (np.ptp(ys) or 1.0) * (height - 1)
    for i in range(len(ids)):
        col, row = int(round(sx[i])), height - 1 - int(round(sy[i]))
        ch = str(labels[i])[0]
        cur = grid[row][col]
        grid[row][col] = ch if cur == " " else ("*" if cur != ch else ch)
    lines = [f"PCA of {len(ids)} card embeddings — PC1 {frac[0]*100:.1f}%, "
             f"PC2 {frac[1]*100:.1f}% of variance; marker = {mark}[0], "
             "'*' = mixed"]
    lines += ["".join(r) for r in grid]
    # Several labels can share a first letter (W, WU, WUBG all plot as 'W'), so
    # the legend groups by the character actually drawn.
    by_char = {}
    for lab in sorted(set(str(l) for l in labels)):
        by_char.setdefault(lab[0], []).append(lab)
    lines.append("legend: " + "  ".join(f"{ch}={'/'.join(v)}"
                                        for ch, v in sorted(by_char.items())))
    return lines


def render_occurrences(counts, n_states, top_n=25):
    ids = named_card_ids()
    order = sorted(ids, key=lambda i: -counts[i])
    zero = [i for i in ids if counts[i] == 0]
    lines = [f"card occurrences over {n_states} sampled decision states",
             f"  {len(ids) - len(zero)}/{len(ids)} named cards appear; "
             f"{len(zero)} never seen (their embedding rows are untrained)", ""]
    for i in order[:top_n]:
        frac = counts[i] / max(1, n_states)
        lines.append(f"  {counts[i]:6d}  {frac*100:5.1f}%  "
                     f"{card_name(i)[:38]:<38} {_bar(frac)}")
    if zero:
        lines.append("")
        lines.append(f"never seen ({len(zero)}): "
                     + ", ".join(card_name(i) for i in zero[:40])
                     + (" …" if len(zero) > 40 else ""))
    return lines


def render_category_embedding(net, top_k=4):
    mat = category_embedding(net)
    u = _unit(mat)
    sims = u @ u.T
    lines = ["action-category embedding — nearest categories by cosine",
             "  (does the net treat these decision kinds alike?)", ""]
    live = [c for c in range(mat.shape[0]) if np.linalg.norm(mat[c]) > 1e-6]
    for c in live:
        order = [j for j in np.argsort(-sims[c]) if j != c][:top_k]
        near = ", ".join(f"{_CAT_NAMES.get(int(j), int(j))}({sims[c][j]:.2f})"
                         for j in order)
        lines.append(f"  {_CAT_NAMES.get(c, c):<22} → {near}")
    return lines


def render_buckets(net, sample_buckets=None):
    rows = bucket_table(net)
    n_arch = archetypes.N_ARCH
    lines = ["value-head columns — one per (self archetype x opp archetype)",
             "  cell = column weight norm; '·' = DEAD (constant value there)",
             ""]
    head = "  " + " " * 15 + "".join(f"{archetypes.arch_name_at(j)[:7]:>8}"
                                     for j in range(n_arch))
    lines.append(head + "   (opponent)")
    for i in range(n_arch):
        cells = []
        for j in range(n_arch):
            r = rows[i * n_arch + j]
            cells.append("       ·" if r["dead"] else f"{r['norm']:8.2f}")
        lines.append(f"  {archetypes.arch_name_at(i)[:14]:<15}" + "".join(cells))
    lines.append("  (self)")
    dead = [r for r in rows if r["dead"]]
    lines.append("")
    lines.append(f"{len(rows) - len(dead)}/{len(rows)} columns alive.")
    if dead:
        lines.append("dead columns return a constant "
                     f"(e.g. {dead[0]['name']} → {dead[0]['const']:+.3f}), which "
                     "reads downstream as a confident 50%.")
    if sample_buckets is not None:
        seen, cnt = np.unique(sample_buckets, return_counts=True)
        lines.append("")
        lines.append(f"{len(seen)}/{len(rows)} buckets appear in the sampled "
                     "self-play — a column with no data is untrained whether or "
                     "not it reads as alive above:")
        for b, c in sorted(zip(seen, cnt), key=lambda t: -t[1]):
            r = rows[int(b)]
            flag = "  ⚠ DEAD but being trained on" if r["dead"] else ""
            lines.append(f"  {archetypes.bucket_name(int(b)):<34} n={c}{flag}")
    return lines


def render_calibration(cal, bins=True):
    lines = [f"value calibration over {cal['n']} recorded decisions "
             f"(predicted V vs the game's realized outcome z)",
             f"overall MSE {cal['overall_mse']:.3f}", "",
             f"  {'bucket':<32} {'n':>5} {'MSE':>6} {'V̄':>7} {'z̄':>7} {'sign':>6}"]
    for r in cal["rows"]:
        flag = " DEAD" if r["dead"] else ""
        lines.append(f"  {r['name'][:32]:<32} {r['n']:5d} {r['mse']:6.3f} "
                     f"{r['mean_pred']:+7.3f} {r['mean_z']:+7.3f} "
                     f"{r['sign_agree']:6.2f}{flag}")
    if bins and cal["rows"]:
        top = cal["rows"][0]
        lines.append("")
        lines.append(f"reliability, {top['name']} (n={top['n']}):")
        lines.append(f"  {'predicted V':<16} {'n':>5} {'mean V':>8} {'mean z':>8}")
        for lo, hi, n, mv, mz in top["reliability"]:
            lines.append(f"  [{lo:+.2f},{hi:+.2f}]{'':<3} {n:5d} {mv:+8.3f} "
                         f"{mz:+8.3f}  {_bar(abs(mv - mz))}")
    return lines


def render_divergence(div):
    lines = [f"raw-net priors vs search posterior over {div['n']} decisions",
             f"  mean KL(search‖net) {div['kl']:.3f}   top-1 agreement "
             f"{div['top1']*100:.1f}%",
             "  high KL = the net cannot reproduce this decision kind without "
             "search", "",
             f"  {'action category':<26} {'n':>6} {'KL':>7} {'top-1':>7} "
             f"{'legal':>6}"]
    for r in div["rows"]:
        lines.append(f"  {r['name'][:26]:<26} {r['n']:6d} {r['kl']:7.3f} "
                     f"{r['top1']*100:6.1f}% {r['legal']:6.1f}  "
                     f"{_bar(min(1.0, r['kl']))}")
    return lines


_SPARK = "▁▂▃▄▅▆▇█"


def _spark(values, min_span=0.0):
    """Terminal sparkline over a float series.

    ``min_span`` renders a flat line when the series barely moves — a sparkline
    autoscaled to a 0.001-wide range otherwise draws a dramatic ramp out of
    nothing."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return ""
    lo, hi = float(v.min()), float(v.max())
    span = hi - lo
    if span <= 0 or span < min_span:
        return "─" * len(v)
    idx = np.clip(((v - lo) / span * (len(_SPARK) - 1)).round().astype(int),
                  0, len(_SPARK) - 1)
    return "".join(_SPARK[i] for i in idx)


def render_state(sample, row, net=None, top_n=12):
    """One recorded decision: the board, the search's posterior next to the raw
    net's priors, and the game's eventual result."""
    from env import MAX_ACTIONS
    obs = sample["obs"][row]
    pi, mask, z = sample["pi"][row], sample["mask"][row], float(sample["z"][row])
    n_legal = int(mask.sum())
    lines = [f"recorded decision {row} of {sample['obs'].shape[0]}   "
             f"outcome z={z:+.0f}   legal actions={n_legal}"]
    if net is not None:
        lines[0] += f"   net V={state_value(net, obs, mask):+.3f}"
    lines.append("")
    lines += decode.format_state_lines(decode.decode_game_state(obs))
    lines.append("")

    priors = None
    if net is not None:
        _, p = predict(net, obs[None], mask[None])
        priors = p[0]
    acts = decode.decode_actions(decode.action_categories(obs, MAX_ACTIONS),
                                 decode.action_card_ids(obs),
                                 decode.action_ctrls(obs), n_legal,
                                 zone_refs=decode.action_zone_refs(obs,
                                                                   MAX_ACTIONS))
    order = np.argsort(-pi[:n_legal])[:top_n]
    lines.append(f"  {'search':>7} {'net':>7}  action")
    for i in order:
        p_net = "" if priors is None else f"{priors[i]*100:6.1f}%"
        lines.append(f"  {pi[i]*100:6.1f}% {p_net:>7}  "
                     f"{acts[i]['description'][:64]}")
    return lines


def render_block_importance(imp, top_n=20, single=False):
    head = ("what THIS evaluation rests on" if single
            else f"what the value head rests on, over {imp['n']} states")
    lines = [head,
             "  each block's floats are replaced by another real state's; the "
             "number is the mean |ΔV| that causes.", ""]
    if single:
        lines.insert(1, f"  base V={imp['base']:+.3f}  "
                        f"({imp['donors']} donor states per block)")
    top = imp["rows"][0]["delta"] or 1.0
    lines.append(f"  {'block':<24} {'width':>6} {'mean |ΔV|':>10}"
                 + ("  signed" if single else ""))
    for r in imp["rows"][:top_n]:
        sign = f"  {r['signed']:+.3f}" if single else ""
        lines.append(f"  {r['name']:<24} {r['width']:6d} {r['delta']:10.4f}"
                     f"{sign}  {_bar(r['delta'] / top)}")
    return lines


def render_card_swap(probe, site_label, top_n=12, counts=None):
    rows = probe["rows"]
    lines = [f"card-swap probe — {site_label}",
             f"  base V={probe['base']:+.3f}; each row is V with that card's "
             "identity in this slot",
             "  (only the identity float changes — the slot keeps its power, "
             "toughness and status)", "",
             f"  {'ΔV':>7} {'V':>7}  {'card':<34} seen"]
    for idx, v, dv in rows[:top_n]:
        seen = "" if counts is None else str(int(counts[idx]))
        lines.append(f"  {dv:+7.3f} {v:+7.3f}  {card_name(idx)[:34]:<34} {seen}")
    lines.append(f"  {'…':>7}")
    for idx, v, dv in rows[-top_n:]:
        seen = "" if counts is None else str(int(counts[idx]))
        lines.append(f"  {dv:+7.3f} {v:+7.3f}  {card_name(idx)[:34]:<34} {seen}")
    return lines


def render_sweeps(net, obs_row, mask_row, fields=None):
    """V across every sweepable scalar, with the monotonicity check."""
    fields = fields or list(sweep_fields())
    lines = ["value response to single scalars (everything else held fixed)",
             "  a net whose V does not fall as its OWN life falls is broken; "
             "the arrow flags disagreement.", ""]
    for f in fields:
        res = sweep(net, obs_row, mask_row, f)
        tr = sweep_trend(res)
        flag = ("" if tr["agrees"] is None
                else ("  ok" if tr["agrees"] else "  ⚠ WRONG DIRECTION"))
        lines.append(f"  {f:<10} now={res['current']:<5.0f} "
                     f"V {res['v'][0]:+.2f} → {res['v'][-1]:+.2f} "
                     f"(span {tr['span']:.2f})  "
                     f"{_spark(res['v'], min_span=0.01)}{flag}")
    return lines


def render_diff(d, top_n=15):
    lines = [f"A: {d['a']}", f"B: {d['b']}", ""]
    if d["only_a"] or d["only_b"] or d["shape_diff"]:
        lines.append(f"structural: {len(d['only_a'])} only in A, "
                     f"{len(d['only_b'])} only in B, "
                     f"{len(d['shape_diff'])} shape mismatches")
        lines.append("")
    lines.append(f"  {'tensor':<44} {'rel Δ':>8} {'|Δ|':>10}")
    for t in d["tensors"][:top_n]:
        lines.append(f"  {t['name'][:44]:<44} {t['rel']:8.4f} {t['delta']:10.4f}")
    if d["cards"]:
        lines.append("")
        lines.append("cards whose embedding moved most:")
        for _, name, dist in d["cards"]:
            lines.append(f"  {dist:8.4f}  {name}")
    if d["buckets"]:
        lines.append("")
        lines.append("value-head columns that moved most:")
        for _, name, dist in d["buckets"]:
            lines.append(f"  {dist:8.4f}  {name}")
    return lines


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _add_shard_args(p, default_rows=4000):
    p.add_argument("--shards", default=AZ_DATA_DIR,
                   help=f"directory of shard_*.npz self-play records "
                        f"(default: {AZ_DATA_DIR})")
    p.add_argument("--max-rows", type=int, default=default_rows,
                   help="max recorded decisions to sample")
    p.add_argument("--window", type=int, default=None,
                   help="use only the newest N shards")
    p.add_argument("--seed", type=int, default=0, help="sampling seed")


def _sample(args):
    return load_shard_sample(args.shards, max_rows=args.max_rows,
                             window=args.window, seed=args.seed)


def _maybe_counts(args):
    """Occurrence counts when the caller asked for them, else None."""
    if not getattr(args, "with_counts", False):
        return None
    s = load_shard_sample(args.shards, max_rows=args.count_rows,
                          window=args.window, seed=args.seed)
    counts, _ = card_occurrences(s["obs"], limit=args.count_rows, seed=args.seed)
    return counts


def _add_count_args(p):
    p.add_argument("--with-counts", action="store_true",
                   help="annotate/filter by how often each card appears in "
                        "recorded self-play (loads shards)")
    p.add_argument("--count-rows", type=int, default=800,
                   help="states to decode for occurrence counts (default: 800)")
    p.add_argument("--min-seen", type=int, default=0,
                   help="with --with-counts: drop cards seen fewer than N times")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="az_inspect",
        description="Inspect an AlphaZero checkpoint's weights and recorded "
                    "self-play — no games played.")
    ap.add_argument("--model", default="gen",
                    help="AZ checkpoint spec: 'gen' (the generalist) or a path")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("overview", help="checkpoint meta + critic + shard status")
    _add_shard_args(p)
    p.add_argument("--no-shards", action="store_true",
                   help="skip the shard sample (weights only)")

    p = sub.add_parser("neighbors", help="nearest cards in embedding space")
    p.add_argument("card", help="card name (exact or unique substring)")
    p.add_argument("-k", type=int, default=15, help="neighbours to show")
    _add_shard_args(p); _add_count_args(p)

    p = sub.add_parser("structure", help="kNN label purity of the embedding")
    p.add_argument("-k", type=int, default=10, help="neighbours per card")
    _add_shard_args(p); _add_count_args(p)

    p = sub.add_parser("clusters", help="k-means over the card embedding")
    p.add_argument("-k", type=int, default=8, help="clusters")
    p.add_argument("--cluster-seed", type=int, default=0)
    _add_shard_args(p); _add_count_args(p)

    p = sub.add_parser("project", help="PCA-to-2D terminal scatter")
    p.add_argument("--mark", default="color", choices=LABEL_KINDS)
    p.add_argument("--width", type=int, default=78)
    p.add_argument("--height", type=int, default=24)
    _add_shard_args(p); _add_count_args(p)

    p = sub.add_parser("occur", help="how often each card appears in self-play")
    _add_shard_args(p)
    p.add_argument("--count-rows", type=int, default=1500,
                   help="states to decode (default: 1500)")
    p.add_argument("--top", type=int, default=25)

    sub.add_parser("catemb", help="action-category embedding neighbours")

    p = sub.add_parser("buckets", help="per-matchup critic column map")
    _add_shard_args(p)
    p.add_argument("--no-shards", action="store_true",
                   help="skip the sampled-bucket census (weights only)")

    p = sub.add_parser("calib", help="per-bucket value calibration vs outcomes")
    _add_shard_args(p)

    p = sub.add_parser("divergence", help="net priors vs search posterior")
    _add_shard_args(p)
    p.add_argument("--top", type=int, default=12)

    p = sub.add_parser("diff", help="compare two checkpoints")
    p.add_argument("other", help="second checkpoint spec/path (B); --model is A")
    p.add_argument("--top", type=int, default=15)

    p = sub.add_parser("state", help="browse one recorded decision")
    _add_shard_args(p)
    p.add_argument("--row", type=int, default=0, help="which sampled decision")
    p.add_argument("--top", type=int, default=12)

    p = sub.add_parser("blocks", help="permutation importance of obs blocks")
    _add_shard_args(p)
    p.add_argument("--row", type=int, default=None,
                   help="attribute ONE recorded decision instead of the mean "
                        "over many")
    p.add_argument("--rows", type=int, default=150,
                   help="states averaged when --row is not given")
    p.add_argument("--donors", type=int, default=3,
                   help="donor states per block")
    p.add_argument("--top", type=int, default=20)

    p = sub.add_parser("swap", help="per-slot card valuation by identity swap")
    _add_shard_args(p)
    p.add_argument("--row", type=int, default=0, help="which sampled decision")
    p.add_argument("--site", type=int, default=None,
                   help="index into the state's card-identity sites "
                        "(omit to list them)")
    p.add_argument("--top", type=int, default=12)

    p = sub.add_parser("sweep", help="V across single scalars (life, hand, turn)")
    _add_shard_args(p)
    p.add_argument("--row", type=int, default=0, help="which sampled decision")
    p.add_argument("--field", default=None,
                   help="one field (default: every sweepable field)")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cmd = args.cmd

    if cmd == "catemb":
        net, _ = load_net(args.model)
        print("\n".join(render_category_embedding(net)))
        return 0

    if cmd == "diff":
        a = resolve_spec(args.model)
        b = resolve_spec(args.other)
        if a is None or b is None:
            raise SystemExit(f"could not resolve both checkpoints "
                             f"({args.model!r} -> {a}, {args.other!r} -> {b})")
        print("\n".join(render_diff(checkpoint_diff(a, b, top_n=args.top),
                                    top_n=args.top)))
        return 0

    if cmd == "occur":
        s = load_shard_sample(args.shards, max_rows=max(args.max_rows,
                                                        args.count_rows),
                              window=args.window, seed=args.seed)
        counts, n = card_occurrences(s["obs"], limit=args.count_rows,
                                     seed=args.seed)
        print("\n".join(render_occurrences(counts, n, top_n=args.top)))
        return 0

    net, path = load_net(args.model)

    if cmd == "overview":
        sample = None if args.no_shards else _sample(args)
        print("\n".join(render_overview(net, path, sample)))
        return 0

    if cmd == "buckets":
        sb = None
        if not args.no_shards:
            sb = obs_buckets(net, _sample(args)["obs"])
        print("\n".join(render_buckets(net, sb)))
        return 0

    if cmd == "calib":
        print("\n".join(render_calibration(bucket_calibration(net, _sample(args)))))
        return 0

    if cmd == "divergence":
        print("\n".join(render_divergence(
            policy_divergence(net, _sample(args), top_n=args.top))))
        return 0

    if cmd in ("state", "blocks", "swap", "sweep"):
        s = _sample(args)
        row = getattr(args, "row", 0)
        if row is not None and not 0 <= row < s["obs"].shape[0]:
            raise SystemExit(f"--row {row} out of range "
                             f"(0..{s['obs'].shape[0] - 1} in this sample)")
        if cmd == "state":
            print("\n".join(render_state(s, row, net=net, top_n=args.top)))
        elif cmd == "blocks":
            if args.row is None:
                imp = block_importance(net, s, n_rows=args.rows,
                                       donors=args.donors, seed=args.seed)
                print("\n".join(render_block_importance(imp, top_n=args.top)))
            else:
                imp = state_block_importance(net, s, row, seed=args.seed)
                print("\n".join(render_block_importance(imp, top_n=args.top,
                                                        single=True)))
        elif cmd == "swap":
            sites = card_id_sites(s["obs"][row])
            if not sites:
                raise SystemExit(f"decision {row} has no card on the "
                                 "battlefield or in hand to swap")
            if args.site is None:
                print(f"card-identity sites in decision {row}:")
                for i, (label, _, _) in enumerate(sites):
                    print(f"  {i:3d}  {label}")
                print("\nre-run with --site N")
                return 0
            if not 0 <= args.site < len(sites):
                raise SystemExit(f"--site {args.site} out of range "
                                 f"(0..{len(sites) - 1})")
            label, off, _ = sites[args.site]
            probe = card_swap_probe(net, s["obs"][row], s["mask"][row], off)
            print("\n".join(render_card_swap(probe, label, top_n=args.top)))
        else:
            fields = [args.field] if args.field else None
            print("\n".join(render_sweeps(net, s["obs"][row], s["mask"][row],
                                          fields)))
        return 0

    mat = card_embedding(net)
    counts = _maybe_counts(args)

    if cmd == "neighbors":
        idx = resolve_card(args.card)
        cand = None
        if counts is not None and args.min_seen > 0:
            cand = np.array([i for i in named_card_ids()
                             if counts[i] >= args.min_seen])
        print("\n".join(render_neighbors(mat, idx, k=args.k, counts=counts,
                                         candidates=cand)))
    elif cmd == "structure":
        print("\n".join(render_structure(mat, k=args.k, counts=counts,
                                         min_seen=args.min_seen)))
    elif cmd == "clusters":
        print("\n".join(render_clusters(mat, k=args.k, seed=args.cluster_seed,
                                        counts=counts, min_seen=args.min_seen)))
    elif cmd == "project":
        print("\n".join(render_projection(mat, width=args.width,
                                          height=args.height, mark=args.mark,
                                          counts=counts,
                                          min_seen=args.min_seen)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
