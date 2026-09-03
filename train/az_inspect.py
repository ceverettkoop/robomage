"""Static inspection of an AlphaZero checkpoint — weights and recorded self-play,
never a live game.

Every view here answers "what has this net actually learned to value?" from two
engine-free sources:

  * the checkpoint's WEIGHTS (``checkpoints/az/gen__az*.pt`` + its ``.meta.json``),
    loaded through :func:`az_net.load_az` so the obs-layout handshake is checked;
  * the recorded self-play SHARDS (``az_data/gen/shard_*.npz``: ``obs, pi, z,
    mask``, plus the optional n-step TD columns ``q, explored, td_q``), which are
    real observed states with the search's visit posterior and the eventual game
    result attached.

Neither needs the C++ engine, a subprocess, or a played game — the point is to be
able to say something about the model that is not "it won N% of its games".

The card-identity embedding is the interpretable table: ``trunk.card_emb`` is an
``nn.Embedding(N_CARD_TYPES + 1, card_embed_dim, padding_idx=0)`` whose row
``i + 1`` is vocab card ``i`` (row 0 is the empty-slot padding), so its rows line
up 1:1 with ``src/card_vocab.h`` and can be named, typed, and costed via
``decode``/``card_costs``. Since the card-representation split it is the
TRAINABLE RESIDUAL half of the card vector — the network consumes
``[card_emb | trunk.card_props]``, where ``card_props`` is the frozen
printed-property buffer (card_props codegen). The embedding views default to
the residual table (that is what training moves); pass ``--space props`` /
``--space full`` to measure the frozen block or the concatenation instead.

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
  firstlayer   which input columns each encoder's first layer actually reads
  bodylayer    what the policy/value bodies read from the pooled features —
               incl. the archetype one-hots (named) and decklist aggregates
  unit         one hidden unit's signed input-variable recipe (exact, per-row)
  pathto       weights-only input connectivity of one value bucket's column
  spectra      singular-value spectrum / effective rank of every weight matrix
  popart       PPO PopArt per-bucket value-normalizer statistics (PPO .zip only)
  valuegeom    value-head row geometry — which matchups share a value direction
  zoneemb      full cosine matrix of the small zone-ref embedding
  cardsel      entity-encoder units ranked by card selectivity

The weight-space views (firstlayer, bodylayer, unit, pathto, spectra, popart,
valuegeom, zoneemb, cardsel, and diff) read EITHER checkpoint family: an AZ ``.pt``, or a PPO ``.zip`` whose
``policy.pth`` state dict is renamed tensor-for-tensor into the AZNet key space
(``features_extractor.`` → ``trunk.``, ``value_net.`` → ``value_head.``, …) —
the two families share the CardGameExtractor trunk, so every trunk view means
the same thing on both. Their ``--model gen`` falls back to the PPO generalist
(``checkpoints/gen__final.zip``) when no AZ checkpoint exists yet.

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
import io
import json
import os
import sys
import zipfile

import numpy as np

import archetypes
import decode
from card_costs import (N_CARD_TYPES, _VOCAB_NAMES as VOCAB_NAMES,
                        _CARD_COST_MATRIX as CARD_COST_MATRIX,
                        _LAND_VOCAB_IDS as LAND_VOCAB_IDS)
from card_props import N_CARD_PROPS, _PROP_NAMES
from _enums import (_CAT_NAMES, _OBS_KEYWORDS, _REF_NAMES, N_OBS_KEYWORDS,
                    N_REF_ZONES, STACK_QUAL_FIELDS, MAX_STACK_MODES,
                    MAX_STACK_TGTS)

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


def _as_state_dict(net_or_sd):
    """The table accessors below take either a loaded AZNet or an AZNet-keyed
    state dict (load_weight_state's output — which is how a PPO .zip reaches
    them), so the embedding views work on both checkpoint families."""
    return net_or_sd if isinstance(net_or_sd, dict) else net_or_sd.state_dict()


def card_embedding(net):
    """The TRAINABLE card-identity embedding as ``(N_CARD_TYPES, D)``, row ``i`` =
    vocab card ``i``. The stored table has ``N_CARD_TYPES + 1`` rows — row 0 is
    the padding row for empty slots — so this drops it and re-bases the index.
    This is only the residual half of the card vector; see card_matrix for the
    frozen property block and the full concatenation."""
    w = _as_state_dict(net)["trunk.card_emb.weight"].detach().cpu().numpy()
    return np.array(w[1:], dtype=np.float64)


def card_property_block(net):
    """The FROZEN printed-property block as ``(N_CARD_TYPES, P)``, row-aligned
    with card_embedding (padding row dropped). Constant by construction — a
    registered buffer, not a parameter — so purity measured on it is the
    ceiling the printed facts alone provide."""
    w = _as_state_dict(net)["trunk.card_props"].detach().cpu().numpy()
    return np.array(w[1:], dtype=np.float64)


CARD_SPACES = ("identity", "props", "full")


def card_matrix(net, space="identity"):
    """The per-card matrix for one of the three card spaces: the trainable
    ``identity`` table (default — what training moves), the frozen ``props``
    block, or the ``full`` concatenation the network actually consumes."""
    if space == "identity":
        return card_embedding(net)
    if space == "props":
        return card_property_block(net)
    if space == "full":
        return np.concatenate([card_embedding(net), card_property_block(net)],
                              axis=1)
    raise ValueError(f"unknown card space {space!r} (want one of {CARD_SPACES})")


def category_embedding(net):
    """The action-category embedding as ``(ACTION_CATEGORY_MAX + 1, D)``."""
    w = _as_state_dict(net)["trunk.action_cat_emb.weight"]
    return np.array(w.detach().cpu().numpy(), dtype=np.float64)


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


# Shard columns this reader gathers: the four the analysis views need, plus the
# n-step TD trio, which is optional here (see load_shard_sample).
_SHARD_CORE_COLS = ("obs", "pi", "z", "mask")
_SHARD_TD_COLS = ("q", "explored", "td_q")


def load_shard_sample(data_dir=AZ_DATA_DIR, max_rows=4000, window=None, seed=0):
    """Load a bounded RANDOM SAMPLE of recorded decisions.

    The full pool is gigabytes of float32 observations, so this walks the shards
    newest-first and takes an even per-shard quota, keeping the sample spread over
    the window rather than concentrated in whichever shard happened to be biggest.
    Returns a dict with ``obs, pi, z, mask, n_shards, n_rows_total``, plus the
    n-step TD columns ``q, explored, td_q`` when the shards carry them. Those are
    OPTIONAL here on purpose: this is a diagnostic reader that gets pointed at
    arbitrary (including hand-built or pre-schema) shard directories, unlike the
    trainer's :func:`az_train.load_window`, which requires them.
    """
    from env import OBS_SIZE, MAX_ACTIONS
    data_dir = data_dir or AZ_DATA_DIR
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
    # The n-step TD columns are taken only when EVERY sampled shard carries them,
    # so a mixed directory can never yield ragged parallel arrays.
    cols = {k: [] for k in _SHARD_CORE_COLS + _SHARD_TD_COLS}
    have_td = True
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
        for k in _SHARD_CORE_COLS:
            cols[k].append(d[k][sel])
        if all(k in d.files for k in _SHARD_TD_COLS):
            for k in _SHARD_TD_COLS:
                cols[k].append(d[k][sel])
        else:
            have_td = False
        used += 1
        if sum(a.shape[0] for a in cols["obs"]) >= max_rows:
            break
    keys = _SHARD_CORE_COLS + (_SHARD_TD_COLS if have_td else ())
    out = {k: np.concatenate(cols[k]) for k in keys}
    out["n_shards"] = used
    out["n_rows_total"] = total
    if out["obs"].shape[0] > max_rows:
        keep = np.sort(rng.choice(out["obs"].shape[0], size=max_rows,
                                  replace=False))
        for key in keys:
            out[key] = out[key][keep]
    return out


# Decoded-state zones whose card ids are looked up in ``trunk.card_emb`` (the
# perm / stack / entity encoders all embed the slot's id). Dict-valued zones
# carry {"name": ...}; the rest are plain name lists.
_EMB_DICT_ZONES = ("self_battlefield", "opp_battlefield", "self_hand", "stack",
                   "known_top_library", "opp_known_hand")
_EMB_STR_ZONES = ("self_graveyard", "opp_graveyard", "self_exile", "opp_exile")


def _decklist_id_slots():
    """``[(start, slots)]`` for the five ``(card_id, count)`` decklist blocks.

    These are NOT part of the decoded board dump, but their ids do go through
    ``card_emb`` (via ``decklist_encoder``), so a card sitting in a decklist is
    embedded at every decision even though it is nowhere on the board."""
    import env as e
    return [(e._SELF_LIVE_LIB_START, e.DECKLIST_MAIN_SLOTS),
            (e._SELF_DECK_MAIN_START, e.DECKLIST_MAIN_SLOTS),
            (e._SELF_DECK_SIDE_START, e.DECKLIST_SIDE_SLOTS),
            (e._OPP_DECK_MAIN_START, e.DECKLIST_MAIN_SLOTS),
            (e._OPP_DECK_SIDE_START, e.DECKLIST_SIDE_SLOTS)]


def _state_card_ids(obs_row, name_to_id, deck_slots, slot_size):
    """``(embedded_ids, revealed_ids)`` for one observation.

    The split matters: the opponent-revealed block is a dense N_CARD_TYPES
    multi-hot fed to ``revealed_encoder``, NOT a card-id lookup, so a mention
    there gives the card's EMBEDDING ROW no gradient at all. It is also sticky
    for the whole match (revealed once = set in every later state), so folding
    it into one number both overstates exposure and misattributes it."""
    gs = decode.decode_game_state(obs_row)
    emb, revealed = set(), set()
    for key in _EMB_DICT_ZONES:
        for item in gs.get(key) or ():
            nm = item.get("name") if isinstance(item, dict) else item
            if nm and nm in name_to_id:
                emb.add(name_to_id[nm])
    for key in _EMB_STR_ZONES:
        for nm in gs.get(key) or ():
            if isinstance(nm, str) and nm in name_to_id:
                emb.add(name_to_id[nm])
    for nm in gs.get("opp_revealed") or ():
        if isinstance(nm, str) and nm in name_to_id:
            revealed.add(name_to_id[nm])
    for start, slots in deck_slots:
        for k in range(slots):
            idx = int(round(float(obs_row[start + k * slot_size])
                            * N_CARD_TYPES))
            if idx >= 0:
                emb.add(idx)
    return emb, revealed


def card_occurrence_split(obs, limit=1500, seed=0):
    """Per-card state counts, split by how the card reaches the network:

      ``emb``      — states where the card's id is EMBEDDED (both boards, hand,
                     stack, graveyards, exiles, known top-library, known
                     opponent hand, and the five decklist blocks)
      ``revealed`` — states where it appears ONLY in the dense opponent-revealed
                     multi-hot, which never touches the card embedding

    Counting is per state (a hand of four Brainstorms counts once). Returns
    ``(emb, revealed, n_states)``."""
    rng = np.random.default_rng(seed)
    n = obs.shape[0]
    idx = np.arange(n) if n <= limit else np.sort(
        rng.choice(n, size=int(limit), replace=False))
    name_to_id = {n_: i for i, n_ in enumerate(VOCAB_NAMES) if n_}
    deck_slots = _decklist_id_slots()
    import env as e
    slot_size = e._DECKLIST_SLOT_SIZE
    emb = np.zeros(N_CARD_TYPES, dtype=np.int64)
    rev = np.zeros(N_CARD_TYPES, dtype=np.int64)
    for r in idx:
        e_ids, r_ids = _state_card_ids(obs[r], name_to_id, deck_slots, slot_size)
        for i in e_ids:
            emb[i] += 1
        for i in r_ids - e_ids:
            rev[i] += 1
    return emb, rev, int(len(idx))


def card_occurrences(obs, limit=1500, seed=0):
    """Per-card count of sampled states that EMBED the card — the occurrence
    number the embedding views should filter on. See card_occurrence_split for
    the revealed-only column. Returns ``(counts, n_states)``.

    Note this measures *visibility in the sampled data*; :func:`card_exposure`
    answers the stronger question (did this row actually receive gradient) from
    the weights, with no shards at all."""
    emb, _, n = card_occurrence_split(obs, limit=limit, seed=seed)
    return emb, n


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
        # NB: this span also covers the deck-identity tail blocks, which have never
        # had their own row here; it stops at the mana-development block below.
        ("global extras", e._EXTRAS_START, e._MANA_DEV_START),
        ("mana development", e._MANA_DEV_START, e._LOG_VITALS_START),
        ("log vitals", e._LOG_VITALS_START, e.STATE_SIZE),
        ("action categories", e.ACT_CATS_START, e.ACT_CATS_START + e.MAX_ACTIONS),
        ("action card ids", e.ACT_IDS_START, e.ACT_IDS_START + e.MAX_ACTIONS),
        ("action controllers", e.ACT_CTRL_START, e.ACT_CTRL_START + e.MAX_ACTIONS),
        ("action zones", e.ACT_ZONE_START, e.ACT_ZONE_START + e.MAX_ACTIONS),
        ("action refs", e.ACT_REFS_START, e.ACT_REFS_START + e.MAX_ACTIONS),
        ("action ordinals", e.ACT_ORDS_START, e.ACT_ORDS_START + e.MAX_ACTIONS),
        # The tail's two halves are separated on purpose: permuting the bucket
        # index re-routes the SAME latent through a different critic read-out
        # (the index is stripped before the trunk), while permuting the one-hots
        # perturbs the trunk's matchup conditioning — one combined row conflated
        # read-out switching with representation change.
        ("matchup bucket idx", e.BUCKET_IDX, e.ARCH_ONEHOT_START),
        ("matchup arch one-hots", e.ARCH_ONEHOT_START, e.OBS_SIZE),
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


def value_latents(net, obs, batch=256):
    """Batched value-body latents (the vector every critic column reads out)."""
    import torch
    out = []
    net.eval()
    with torch.no_grad():
        for s in range(0, obs.shape[0], batch):
            ob = torch.as_tensor(np.ascontiguousarray(obs[s:s + batch]))
            out.append(net.value_body(net.trunk(ob)).cpu().numpy())
    return np.concatenate(out)


def _readout_decomposition(net, base_lat, pert_lat, sel):
    """Split a perturbation's value-side effect into latent movement and what
    the read-outs pass: per row, (||dlatent||, |dz| through the BASE state's
    selected column, mean |dz| across live columns). Pre-tanh, so saturation
    can't hide a shift. sel << all means the state's own matchup read-out
    discards what the perturbed block moved."""
    w = net.value_head.weight.detach().cpu().numpy()
    live = ~net.dead_value_buckets()
    if not live.any():
        live = np.ones(w.shape[0], dtype=bool)
    dlat = pert_lat - base_lat
    dz = dlat @ w.T
    return (np.linalg.norm(dlat, axis=1),
            np.abs(dz[np.arange(len(sel)), sel]),
            np.abs(dz[:, live]).mean(axis=1))


def resolve_block(name, blocks=None):
    """Resolve a block spec (exact or unique substring, case-insensitive)
    to its ``(name, start, end)`` row in the obs-block table."""
    blocks = blocks or obs_blocks()
    want = name.strip().lower()
    exact = [b for b in blocks if b[0].lower() == want]
    if exact:
        return exact[0]
    hits = [b for b in blocks if want in b[0].lower()]
    if len(hits) == 1:
        return hits[0]
    names = ", ".join(b[0] for b in (hits or blocks))
    kind = "ambiguous" if hits else "unknown"
    raise SystemExit(f"{kind} block {name!r} (want one of: {names})")


def block_active_rows(obs, start, end):
    """Row indices where the block deviates from its per-column modal value —
    i.e. where it holds something beyond its empty/baseline pattern. Modal
    (not zero) so card-id slots' -1 sentinel counts as empty."""
    active = np.zeros(obs.shape[0], dtype=bool)
    for c in range(start, end):
        col = obs[:, c]
        vals, counts = np.unique(col, return_counts=True)
        active |= col != vals[np.argmax(counts)]
    return np.flatnonzero(active)


def policy_shift(p, q, mask):
    """Per-row total variation distance between two legal-action policies."""
    return 0.5 * np.abs(np.where(mask, p - q, 0.0)).sum(axis=1)


def block_importance(net, sample, n_rows=150, donors=3, seed=0, blocks=None,
                     only_active=None):
    """Permutation importance of each observation block on V and on the policy.

    For each block, the block's floats are replaced by another sampled state's
    (a valid value for that block, unlike zeroing, which invents states the net
    never saw — a 0 life total is not "no information", it is "dead"), and the
    mean |ΔV| plus the mean total-variation policy shift are recorded.
    High = that head leans on the block.

    ``only_active`` names one block: evaluated rows AND donor draws are then
    restricted to states where that block is non-empty, so a sparse block
    (stack, known top-of-library) is scored on the states where it exists
    instead of being averaged away by its empty majority.
    """
    blocks = blocks or obs_blocks()
    rng = np.random.default_rng(seed)
    obs, mask = sample["obs"], sample["mask"]
    pool = np.arange(obs.shape[0])
    active_n = None
    if only_active:
        name, s, t = resolve_block(only_active, blocks)
        only_active = name
        pool = block_active_rows(obs, s, t)
        active_n = pool.size
        if pool.size < 2:
            raise SystemExit(f"only {pool.size} sampled state(s) have "
                             f"{name!r} active — raise --max-rows/--window")
    n = min(int(n_rows), pool.size)
    rows = np.sort(rng.choice(pool, size=n, replace=False))
    base_v, base_p = predict(net, obs[rows], mask[rows])
    base_lat = value_latents(net, obs[rows])
    sel = np.array([net.obs_value_bucket(o) for o in obs[rows]])
    out = []
    for name, s, t in blocks:
        acc_v = np.zeros(n)
        acc_p = np.zeros(n)
        acc_ro = np.zeros((3, n))
        for _ in range(int(donors)):
            donor = rng.choice(pool, size=n)
            pert = obs[rows].copy()
            pert[:, s:t] = obs[donor][:, s:t]
            v, p = predict(net, pert, mask[rows])
            acc_v += np.abs(v - base_v)
            acc_p += policy_shift(base_p, p, mask[rows])
            acc_ro += _readout_decomposition(
                net, base_lat, value_latents(net, pert), sel)
        out.append({"name": name, "start": s, "width": t - s,
                    "delta": float((acc_v / donors).mean()),
                    "dpi": float((acc_p / donors).mean()),
                    "dlat": float((acc_ro[0] / donors).mean()),
                    "dz_sel": float((acc_ro[1] / donors).mean()),
                    "dz_all": float((acc_ro[2] / donors).mean())})
    out.sort(key=lambda r: -r["delta"])
    return {"rows": out, "n": n, "donors": int(donors),
            "only_active": only_active, "active_n": active_n,
            "base_spread": float(base_v.std())}


def state_block_importance(net, sample, row, donors=16, seed=0, blocks=None):
    """Permutation importance for ONE recorded state — which blocks this
    particular evaluation rests on."""
    blocks = blocks or obs_blocks()
    rng = np.random.default_rng(seed)
    obs, mask = sample["obs"], sample["mask"]
    base_v, base_p = predict(net, obs[row][None], mask[row][None])
    base = float(base_v[0])
    base_lat = np.repeat(value_latents(net, obs[row][None]), donors, axis=0)
    sel = np.full(donors, net.obs_value_bucket(obs[row]))
    batch = np.repeat(obs[row][None], donors, axis=0)
    mk = np.repeat(mask[row][None], donors, axis=0)
    out = []
    for name, s, t in blocks:
        donor = rng.choice(obs.shape[0], size=donors)
        pert = batch.copy()
        pert[:, s:t] = obs[donor][:, s:t]
        v, p = predict(net, pert, mk)
        dlat, dz_sel, dz_all = _readout_decomposition(
            net, base_lat, value_latents(net, pert), sel)
        out.append({"name": name, "start": s, "width": t - s,
                    "delta": float(np.abs(v - base).mean()),
                    "dpi": float(policy_shift(
                        np.repeat(base_p, donors, axis=0), p, mk).mean()),
                    "dlat": float(dlat.mean()),
                    "dz_sel": float(dz_sel.mean()),
                    "dz_all": float(dz_all.mean()),
                    "signed": float((v - base).mean())})
    out.sort(key=lambda r: -r["delta"])
    return {"rows": out, "base": base, "donors": int(donors)}


def value_readouts(net, obs_row):
    """Every critic column's V for ONE state — the same shared value latent
    read out by all N_VALUE_BUCKETS matchup heads.

    Splits each column's pre-tanh logit into its state-dependent part
    (``w·latent``, what the board contributes) and its bias (the column's
    outcome prior — exactly what a memorized column degenerates to). A state
    whose selected column has ``|w·latent|`` far below the bias magnitude is
    being valued by the matchup prior, not the position."""
    import torch
    net.eval()
    with torch.no_grad():
        ob = torch.as_tensor(np.ascontiguousarray(
            np.asarray(obs_row, dtype=np.float32)[None]))
        latent = net.value_body(net.trunk(ob))
        z = net.value_head(latent)[0].cpu().numpy()
        bias = net.value_head.bias.detach().cpu().numpy()
    sel = net.obs_value_bucket(obs_row)
    dead = net.dead_value_buckets()
    rows = [{"bucket": i, "name": archetypes.bucket_name(i),
             "v": float(np.tanh(z[i])), "prior": float(np.tanh(bias[i])),
             "state_part": float(z[i] - bias[i]),
             "selected": i == sel, "dead": bool(dead[i])}
            for i in range(z.shape[0])]
    live_v = np.tanh(z[~dead]) if (~dead).any() else np.tanh(z)
    return {"rows": rows, "selected": sel,
            "value": float(np.tanh(z[sel])),
            "readout_spread": float(live_v.std())}


def render_value_readouts(ro, top_n=None):
    lines = ["V for THIS state under every matchup read-out (one shared "
             "latent, 64 critic columns)",
             f"  selected bucket ▶ {archetypes.bucket_name(ro['selected'])}"
             f"  V={ro['value']:+.3f}   spread across live read-outs "
             f"σ={ro['readout_spread']:.3f}",
             "  state = the board's contribution (w·latent, pre-tanh); "
             "prior = tanh(bias), the column's",
             "  memorized outcome constant. |state| << the V spread means "
             "the matchup prior is doing the valuing.", "",
             f"    {'V':>7} {'prior':>7} {'state':>7}  bucket"]
    rows = sorted(ro["rows"], key=lambda r: -r["v"])
    for r in rows[:top_n]:
        mark = "▶" if r["selected"] else ("×" if r["dead"] else " ")
        lines.append(f"  {mark} {r['v']:+7.3f} {r['prior']:+7.3f} "
                     f"{r['state_part']:+7.3f}  {r['name']}")
    if ro["rows"] and any(r["dead"] for r in ro["rows"]):
        lines.append("  (× = dead column: constant output, no state info)")
    return lines


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

def _snapshot_steps(path):
    """Step count encoded in a ``gen__azv{steps}.pt`` filename, else -1."""
    try:
        return int(os.path.basename(path).split("__azv")[1].split(".pt")[0])
    except (IndexError, ValueError):
        return -1


def resolve_baseline(path, checkpoint_dir=None):
    """The checkpoint to measure training AGAINST, for exposure.

    Prefers the newest ``gen__azv*`` snapshot strictly older than ``path``
    (exposure over that training interval); falls back to the PPO checkpoint AZ
    warm-started from (exposure since warm-start), since AZ has no from-scratch
    path. Returns ``(spec, kind)`` with kind ``'snapshot' | 'ppo' | None``.

    Siblings are searched in ``path``'s OWN directory by default — for the
    normal generalist that is the AZ checkpoint dir, and for a checkpoint kept
    elsewhere the neighbours next to it are the meaningful baselines."""
    checkpoint_dir = checkpoint_dir or os.path.dirname(os.path.abspath(path))
    steps = _snapshot_steps(path)
    snaps = [(s, p) for p in glob.glob(os.path.join(checkpoint_dir,
                                                    "gen__azv*.pt"))
             for s in (_snapshot_steps(p),)
             if s >= 0 and (steps < 0 or s < steps)]
    if snaps:
        return max(snaps)[1], "snapshot"
    from opponents import resolve_checkpoint
    # resolve_checkpoint returns the gen__final path even when it does not exist
    # yet (so callers can test it), so existence is checked here.
    ppo = resolve_checkpoint("gen")
    return (ppo, "ppo") if ppo and os.path.isfile(ppo) else (None, None)


def _card_emb_and_value(spec):
    """``(card_emb weight, value_head weight)`` for a checkpoint spec — either an
    AZ ``.pt`` state dict or a PPO ``.zip`` mapped through the warm-start."""
    import torch
    if str(spec).endswith(".zip"):
        from az_net import from_ppo
        sd = from_ppo(spec).state_dict()
    else:
        sd = torch.load(spec, map_location="cpu")
    return sd["trunk.card_emb.weight"].float(), sd["value_head.weight"].float()


def card_exposure(path, baseline=None, checkpoint_dir=None):
    """Which embedding rows and critic columns actually TRAINED, from weights.

    A row that received no gradient is bit-identical between two checkpoints —
    az_train exempts the embedding tables and the critic's bucket columns from
    weight decay precisely so an untouched row does not drift (az_train.py's
    decay_exempt_param_groups), which makes "moved at all" an exact binary.

    This is the shard-free (and stricter) form of :func:`card_occurrences`: it
    measures gradient rather than visibility, so it also catches cards embedded
    only through a decklist block."""
    if baseline is None:
        baseline, kind = resolve_baseline(path, checkpoint_dir)
    else:
        baseline = resolve_spec(
            baseline, checkpoint_dir
            or os.path.dirname(os.path.abspath(path))) or baseline
        kind = "ppo" if str(baseline).endswith(".zip") else "snapshot"
    if baseline is None:
        raise FileNotFoundError(
            "no baseline checkpoint to measure exposure against — exposure "
            "needs an earlier gen__azv* snapshot or the PPO gen checkpoint AZ "
            "warm-started from")
    emb_a, val_a = _card_emb_and_value(baseline)
    emb_b, val_b = _card_emb_and_value(path)
    # Row i+1 is vocab card i (row 0 is the padding row).
    delta = (emb_b - emb_a).norm(dim=1).numpy()[1:]
    bucket_delta = (val_b - val_a).norm(dim=1).numpy()
    return {"path": path, "baseline": baseline, "kind": kind,
            "delta": delta, "moved": delta > 0.0,
            "bucket_delta": bucket_delta, "bucket_moved": bucket_delta > 0.0}


def exposure_counts(exp):
    """The exposure vector in the shape the embedding views take for filtering
    (per-card, 0 = never trained), so --min-seen works without shards."""
    return exp["delta"]


def filter_vector(counts=None, exposure=None):
    """The per-card vector ``--min-seen`` is compared against, shared by both
    front ends so they filter identically.

    With a shard sample that is the occurrence count. Without one it is a BINARY
    trained/not-trained vector derived from the exposure movement — the movement
    is a float distance (0.39, 0.03, …), so comparing it to a count threshold
    would silently discard almost every card. Under it, ``--min-seen 1`` reads as
    "this row actually received gradient"."""
    if counts is not None:
        return counts
    if exposure is not None:
        return (np.asarray(exposure) > 0).astype(np.int64)
    return None


def try_exposure(path):
    """``card_exposure`` if a baseline exists, else None — the embedding views
    annotate with it when no shard sample is loaded."""
    try:
        return card_exposure(path)
    except (FileNotFoundError, KeyError, RuntimeError):
        return None


def exposure_counts_or_none(path):
    """The per-card movement vector for ``path``, or None when there is no
    baseline checkpoint to measure against."""
    exp = try_exposure(path)
    return None if exp is None else exposure_counts(exp)


def resolve_origin(path, checkpoint_dir=None):
    """The checkpoint closest to this net's INITIAL state, for drift.

    The from-scratch random init is never saved, so "initial" means the PPO
    checkpoint AZ warm-started from when it exists (the embedding's state at
    the start of AZ training), else the OLDEST ``gen__azv*`` snapshot strictly
    older than ``path``. Returns ``(spec, kind)`` like
    :func:`resolve_baseline` (which prefers the NEWEST older snapshot — the
    last training interval rather than the whole run)."""
    from opponents import resolve_checkpoint
    ppo = resolve_checkpoint("gen")
    if ppo and os.path.isfile(ppo):
        return ppo, "ppo"
    checkpoint_dir = checkpoint_dir or os.path.dirname(os.path.abspath(path))
    steps = _snapshot_steps(path)
    snaps = [(s, p) for p in glob.glob(os.path.join(checkpoint_dir,
                                                    "gen__azv*.pt"))
             for s in (_snapshot_steps(p),)
             if s >= 0 and (steps < 0 or s < steps)]
    if snaps:
        return min(snaps)[1], "snapshot"
    return None, None


def _fresh_init_card_emb():
    """The card-embedding table of a freshly constructed, untrained AZNet,
    seeded for reproducibility."""
    import torch
    from az_net import AZNet
    torch.manual_seed(0)
    return AZNet().state_dict()["trunk.card_emb.weight"].float()


def embedding_drift(path, baseline=None, checkpoint_dir=None):
    """Signed per-dimension movement of every card's identity vector.

    Where :func:`card_exposure` reduces movement to one distance per row (and
    measures the last training interval), this keeps the full signed
    (card x dim) delta and measures against the net's ORIGIN
    (:func:`resolve_origin` — the PPO warm-start by default), answering "how
    far, and in which directions, has training pushed each card's learned
    vector since the start". ``baseline='init'`` compares against a freshly
    constructed seed-0 net instead — note per-dimension SIGN against a random
    init is mostly noise; only the magnitudes stay meaningful there."""
    if baseline == "init":
        emb_a, kind = _fresh_init_card_emb(), "init"
        base_label = "fresh untrained net (seed 0)"
    else:
        if baseline is None:
            baseline, kind = resolve_origin(path, checkpoint_dir)
        else:
            baseline = resolve_spec(
                baseline, checkpoint_dir
                or os.path.dirname(os.path.abspath(path))) or baseline
            kind = "ppo" if str(baseline).endswith(".zip") else "snapshot"
        if baseline is None:
            raise FileNotFoundError(
                "no origin checkpoint to measure drift against — drift needs "
                "the PPO gen warm-start, an older gen__azv* snapshot, or "
                "--baseline init")
        emb_a, _ = _card_emb_and_value(baseline)
        base_label = os.path.basename(str(baseline))
    emb_b, _ = _card_emb_and_value(path)
    if emb_a.shape != emb_b.shape:
        raise RuntimeError(
            f"embedding shapes differ ({tuple(emb_a.shape)} vs "
            f"{tuple(emb_b.shape)}) — the baseline predates a layout change")
    # Row i+1 is vocab card i (row 0 is the padding row).
    delta = (emb_b - emb_a).numpy()[1:]
    return {"path": path, "baseline": base_label, "kind": kind, "delta": delta}


def checkpoint_diff(path_a, path_b, top_n=15):
    """Per-tensor, per-card and per-bucket movement between two checkpoints —
    "what did the last training rotation actually change".

    Either side may be an AZ ``.pt`` or a PPO ``.zip`` (keys normalized into the
    AZNet space), so it also answers "what did MCTS training change relative to
    the PPO warm-start"."""
    sa, _ = normalize_state_dict(_load_raw_state(path_a))
    sb, _ = normalize_state_dict(_load_raw_state(path_b))
    shared = [k for k in sa if k in sb and sa[k].shape == sb[k].shape]
    only_a = sorted(k for k in sa if k not in sb)
    only_b = sorted(k for k in sb if k not in sa)
    shape_diff = sorted(k for k in sa if k in sb and sa[k].shape != sb[k].shape)

    tensors = []
    for k in shared:
        va, vb = sa[k].float().flatten(), sb[k].float().flatten()
        d = vb - va
        base = float(va.norm())
        den = float(va.norm() * vb.norm())
        tensors.append({"name": k, "delta": float(d.norm()),
                        "base": base,
                        "cos": float(va @ vb) / den if den > 0 else float("nan"),
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
# Weight-space views — PPO .zip or AZ .pt, one normalized key space
# ----------------------------------------------------------------------

# PPO policy keys → the AZNet naming the rest of this module speaks. The
# pi_/vf_ feature-extractor aliases in an SB3 state dict are the SAME shared
# extractor serialized again, so they are dropped rather than renamed.
_PPO_DROP = ("pi_features_extractor.", "vf_features_extractor.")
_PPO_RENAME = (
    ("features_extractor.", "trunk."),
    ("mlp_extractor.policy_net.", "policy_body."),
    ("mlp_extractor.value_net.", "value_body."),
    ("value_net.", "value_head."),
)


def normalize_state_dict(sd):
    """``(state dict in AZNet key space, family kind 'ppo' | 'az')``.

    A PPO PerActionMaskablePolicy state dict is renamed tensor-for-tensor; the
    PopArt buffers and the action heads keep their names. An AZ dict passes
    through unchanged — the two families share the CardGameExtractor trunk, so
    after this every weight-space view means the same thing on both."""
    if not any(k.startswith("features_extractor.") for k in sd):
        return dict(sd), "az"
    out = {}
    for k, v in sd.items():
        if k.startswith(_PPO_DROP):
            continue
        for src, dst in _PPO_RENAME:
            if k.startswith(src):
                k = dst + k[len(src):]
                break
        out[k] = v
    return out, "ppo"


def _load_raw_state(path):
    """The raw state dict at ``path`` — ``policy.pth`` out of a PPO ``.zip``
    (no sb3 import, just the tensors), or the flat AZ ``.pt`` dict."""
    import torch
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return torch.load(io.BytesIO(z.read("policy.pth")),
                              map_location="cpu", weights_only=True)
    return torch.load(path, map_location="cpu")


def resolve_weights_spec(spec="gen", checkpoint_dir=AZ_CKPT_DIR):
    """Checkpoint path for a weight-space view: an explicit ``.zip``/``.pt``
    path passes through; otherwise the AZ resolution is tried first and — since
    the trunk is shared and the AZ net is always warm-started from the PPO gen —
    the spec falls back to the PPO resolution (``gen`` → ``gen__final.zip``)
    when no AZ checkpoint exists yet."""
    s = str(spec)
    if s.endswith((".zip", ".pt")):
        return s if os.path.isfile(s) else None
    path = resolve_spec(s, checkpoint_dir)
    if path is not None:
        return path
    from opponents import resolve_checkpoint
    ppo = resolve_checkpoint(s)
    return ppo if ppo and os.path.isfile(ppo) else None


def load_weight_state(spec="gen", checkpoint_dir=AZ_CKPT_DIR):
    """Resolve, load, and normalize a checkpoint of either family.
    Returns ``(state_dict, path, kind)``."""
    path = resolve_weights_spec(spec, checkpoint_dir)
    if path is None:
        raise FileNotFoundError(
            f"no checkpoint for {spec!r} — want an AZ .pt, a PPO .zip, or "
            "'gen' with either checkpoints/az/gen__az*.pt or "
            "checkpoints/gen__final.zip present")
    sd, kind = normalize_state_dict(_load_raw_state(path))
    return sd, path, kind


def _t2np(t):
    """Tensor → float64 numpy (works on detached checkpoint tensors without
    importing torch at the call site)."""
    return np.asarray(t.detach().cpu(), dtype=np.float64)


# The per-permanent scalar input columns, in serialized order (mirrors the
# perm-slot field offsets in env.py / machine_io.h), then the keyword multi-hot.
_PERM_SCALAR_NAMES = [
    "power", "toughness", "tapped", "attacking", "blocking", "sickness",
    "damage", "ctrl_is_self", "is_creature", "is_land", "loyalty", "p1p1_net",
    "other_counters", "ref attached_to", "ref attached_by", "ref attack_tgt",
    "ref blocking_tgt", "is_blocked", "is_phased_out",
] + ["kw " + k for k in _OBS_KEYWORDS]
assert len(_PERM_SCALAR_NAMES) == 19 + N_OBS_KEYWORDS

# Cast-qualifier flags in STACK_QUAL_FIELDS order (machine_io.h).
_STACK_QUAL_NAMES = ["is_copy", "kicked", "flashback", "evoke", "escape",
                     "offspring", "impending"]
assert len(_STACK_QUAL_NAMES) == STACK_QUAL_FIELDS


def _card_feat_cols(prefix, card_dim):
    """Column groups for one embedded card-identity input: the trainable
    identity half (dims unnamed) and the frozen printed-property half, whose
    columns carry the card_props codegen names."""
    return [(f"{prefix} identity", card_dim, None),
            (f"{prefix} props", N_CARD_PROPS,
             [f"{prefix} {p}" for p in _PROP_NAMES])]


def _encoder_specs(sd):
    """``[(encoder name, [(group name, width, column names | None), ...]), ...]``
    for every first-layer Linear in the trunk, mirroring the exact ``torch.cat``
    order in ``CardGameExtractor.forward``. first_layer_attribution asserts each
    spec's widths sum to the layer's in-dim, so an extractor input reorder fails
    loudly here instead of silently mislabeling columns."""
    card_dim = int(sd["trunk.card_emb.weight"].shape[1])
    specs = []

    perm = [("status/refs/keywords", len(_PERM_SCALAR_NAMES),
             _PERM_SCALAR_NAMES)]
    perm += _card_feat_cols("chosen-name", card_dim)
    perm += _card_feat_cols("returnable", card_dim)
    perm += _card_feat_cols("card", card_dim)
    specs.append(("perm_encoder", perm))

    stack_scalars = (["ctrl_is_self", "is_spell", "x_or_amount"]
                     + _STACK_QUAL_NAMES
                     + [f"mode_{i}" for i in range(MAX_STACK_MODES)]
                     + [f"tgt{t} {f}" for t in range(MAX_STACK_TGTS)
                        for f in ("present", "is_player", "ctrl", "slot_ref")]
                     + ["stack_pos"])
    stack = [("scalars", len(stack_scalars), stack_scalars)]
    stack += _card_feat_cols("object", card_dim)
    stack += _card_feat_cols("tgts-mean", card_dim)
    specs.append(("stack_encoder", stack))

    specs.append(("entity_encoder",
                  _card_feat_cols("card", card_dim)
                  + [("draw distance", 1, ["draw_dist"])]))
    specs.append(("decklist_encoder",
                  _card_feat_cols("card", card_dim)
                  + [("copies count", 1, ["count"])]))

    # The second-pass reference combiners: every group is an ENCODED entity
    # embedding (no per-column names — the columns are latent dims).
    if "trunk.ref_combiner.0.weight" in sd:
        e_dim = int(sd["trunk.ref_combiner.0.weight"].shape[0])
        specs.append(("ref_combiner",
                      [("own encoding", e_dim, None),
                       ("attached_to enc", e_dim, None),
                       ("attached_by enc", e_dim, None),
                       ("attack_tgt enc", e_dim, None),
                       ("block_tgt enc", e_dim, None)]))
    if "trunk.stk_combiner.0.weight" in sd:
        e_dim = int(sd["trunk.ref_combiner.0.weight"].shape[0])
        half = int(sd["trunk.stk_combiner.0.weight"].shape[1]) - e_dim
        specs.append(("stk_combiner",
                      [("own encoding", half, None),
                       ("targets-mean enc", e_dim, None)]))
    # The revealed multi-hot's columns ARE vocab cards: its per-column weight
    # norm names which opponent reveals the net reacts to.
    specs.append(("revealed_encoder",
                  [("revealed multi-hot", N_CARD_TYPES,
                    [card_name(i) for i in range(N_CARD_TYPES)])]))

    if "trunk.action_encoder.0.weight" in sd:
        in_dim = int(sd["trunk.action_encoder.0.weight"].shape[1])
        cat_dim = int(sd["trunk.action_cat_emb.weight"].shape[1])
        zone_dim = int(sd["trunk.zone_emb.weight"].shape[1])
        card_feat = card_dim + N_CARD_PROPS
        ref_dim = in_dim - (cat_dim + card_feat + 1 + zone_dim + 1)
        act = [("category emb", cat_dim, None)]
        act += _card_feat_cols("tgt-card", card_dim)
        act += [("ctrl flag", 1, ["ctrl_is_self"]),
                ("zone emb", zone_dim, None),
                ("referenced-entity enc", ref_dim, None),
                ("option ordinal", 1, ["option_ordinal"])]
        specs.append(("action_encoder", act))
    return specs


def first_layer_attribution(sd, encoders=None):
    """How hard each encoder's FIRST Linear reads each of its input columns.

    The first layer is the only place the input columns are still separable, and
    every column has a NAME (perm status flags, the 96 card_props codegen
    columns, the revealed multi-hot's vocab cards, …) — so its per-column weight
    norms literally answer "does the net read printed flying, or the learned
    identity residual, or the tapped bit?". Per group: ``share`` = fraction of
    the layer's input weight energy (Σ column-norm²), ``rms`` = RMS per-column
    norm, comparable across groups of different width."""
    out = []
    for enc, spec in _encoder_specs(sd):
        key = f"trunk.{enc}.0.weight"
        if key not in sd or (encoders is not None and enc not in encoders):
            continue
        w = _t2np(sd[key])
        widths = sum(width for _, width, _ in spec)
        assert widths == w.shape[1], (
            f"{enc}: column spec covers {widths} of {w.shape[1]} inputs — the "
            "extractor's forward() input order changed; update _encoder_specs")
        col = np.linalg.norm(w, axis=0)
        energy = col ** 2
        total = float(energy.sum()) or 1.0
        groups, named = [], []
        off = 0
        for name, width, colnames in spec:
            seg = energy[off:off + width]
            groups.append({"name": name, "width": int(width),
                           "share": float(seg.sum() / total),
                           "rms": float(np.sqrt(seg.mean()))})
            if colnames is not None:
                named += [(colnames[j], float(col[off + j]))
                          for j in range(width)]
            off += width
        named.sort(key=lambda t: -t[1])
        out.append({"encoder": enc, "in_dim": int(w.shape[1]),
                    "out_dim": int(w.shape[0]), "groups": groups,
                    "named": named})
    return out


def _body_segments(sd):
    """``(segments, arch_offset)`` for the pooled feature vector the policy and
    value MLP bodies consume — ``segments`` is ``[(name, width), ...]`` in the
    exact ``torch.cat`` order of ``CardGameExtractor.forward``'s ``base``, with
    every width taken from env.py's offset chain or the checkpoint's own tensor
    shapes (never a re-spelled literal). ``arch_offset`` is the start of the
    archetype one-hot block within the vector."""
    import env
    from extractor import _HIST_RECENT_K
    E = int(sd["trunk.perm_encoder.0.weight"].shape[0])
    card_feat = (int(sd["trunk.card_emb.weight"].shape[1])
                 + int(sd["trunk.card_props"].shape[1]))
    cat_dim = int(sd["trunk.action_cat_emb.weight"].shape[1])
    from extractor import _BOARD_COUNT_FEATS
    segments = [
        ("global ctx",         env._GLOBAL_SIZE),
        ("history recent-K",   _HIST_RECENT_K * (cat_dim + card_feat + 2)),
        ("match/lib/turn ctx", env._KNOWN_TOP_LIB_START - env._MATCH_CTX_START),
        ("board counts",       _BOARD_COUNT_FEATS),
        ("revealed agg",       E),
        ("pending decision",   card_feat + 1),
        ("global extras",      env._EXTRAS_END - env._EXTRAS_START),
        ("mana development",   env._MANA_DEV_END - env._MANA_DEV_START),
        ("log vitals",         env._LOG_VITALS_END - env._LOG_VITALS_START),
        # (the raw action-metadata passthrough exists only on the stock
        # per_action_head=False path, which AZNet never uses)
        ("arch one-hots",      env.ARCH_ONEHOT_END - env.ARCH_ONEHOT_START),
        ("perm agg",           2 * E),
        ("stack agg",          2 * E),
        ("top-of-stack",       E),
        ("graveyard agg",      2 * E),
        ("exile agg",          2 * E),
        ("hand+top-lib agg",   2 * E),
        ("next-draw embed",    card_feat),
        ("opp-hand agg",       2 * E),
        ("self live-lib agg",  2 * E),
        ("self main agg",      2 * E),
        ("self side agg",      2 * E),
        ("opp main agg",       2 * E),
        ("opp side agg",       2 * E),
    ]
    arch_offset = 0
    for n, w in segments:
        if n == "arch one-hots":
            break
        arch_offset += w
    return segments, arch_offset


def body_layer_attribution(sd):
    """How hard the policy and value bodies' FIRST layers read each segment of
    the extractor's pooled feature vector — the downstream companion to
    first_layer_attribution. This is where the matchup conditioning becomes
    visible: the archetype one-hot columns (named per self/opp archetype) and
    the five decklist aggregates are individual segments, so "does the model
    read the matchup inputs, and how hard relative to the board?" is answered
    directly from the weights. Same share/rms metrics as firstlayer."""
    segments, arch_off = _body_segments(sd)
    base = sum(w for _, w in segments)
    pa_dim = (int(sd["trunk.action_encoder.2.weight"].shape[0])
              if "trunk.action_encoder.2.weight" in sd else 0)
    n_arch = archetypes.N_ARCH
    arch_names = ([f"self:{archetypes.arch_name_at(i)}" for i in range(n_arch)]
                  + [f"opp:{archetypes.arch_name_at(i)}" for i in range(n_arch)])
    out = []
    for body in ("policy_body", "value_body"):
        key = f"{body}.0.weight"
        if key not in sd:
            continue
        w = _t2np(sd[key])
        pa = w.shape[1] - base
        assert pa >= 0 and (pa_dim == 0 or pa % pa_dim == 0), (
            f"{body}: base segments cover {base} of {w.shape[1]} inputs and "
            f"the {pa}-column remainder is not a per-action block — the "
            "extractor's base concat changed; update _body_segments")
        rows = segments + [("per-action features", pa)]
        col = np.linalg.norm(w, axis=0)
        energy = col ** 2
        total = float(energy.sum()) or 1.0
        groups, off = [], 0
        for name, width in rows:
            seg = energy[off:off + width]
            groups.append({"name": name, "width": int(width),
                           "share": float(seg.sum() / total),
                           "rms": float(np.sqrt(seg.mean()))})
            off += width
        arch = sorted(zip(arch_names,
                          col[arch_off:arch_off + 2 * n_arch].tolist()),
                      key=lambda t: -t[1])
        out.append({"body": body, "in_dim": int(w.shape[1]),
                    "out_dim": int(w.shape[0]), "groups": groups,
                    "arch": [(n, float(v)) for n, v in arch]})
    return out


# The match/library/turn context scalars, in serialized order (machine_io.h:
# match context ×4, library counts & post-board ×3, current turn).
_META_CTX_NAMES = ("game_number", "self_match_wins", "opp_match_wins",
                   "is_sideboard_phase", "self_library_ct", "opp_library_ct",
                   "is_post_board", "turn")


def layer_column_names(sd, layer):
    """``(column labels, state-dict key prefix)`` for a first layer whose input
    columns are nameable: a trunk encoder (perm_encoder, stack_encoder,
    entity_encoder, decklist_encoder, revealed_encoder, action_encoder) or a
    body (policy_body, value_body). Columns without an individual name (learned
    embedding dims, pooled-aggregate dims) get ``group[j]`` labels so every
    label still says which input the column came from."""
    base = str(layer).split(".")[0]
    enc = dict(_encoder_specs(sd))
    if base in enc:
        names = []
        for gname, width, colnames in enc[base]:
            names += (list(colnames) if colnames is not None
                      else [f"{gname}[{j}]" for j in range(width)])
        return names, f"trunk.{base}.0"
    if base in ("policy_body", "value_body"):
        segments, _ = _body_segments(sd)
        n_arch = archetypes.N_ARCH
        names = []
        for gname, width in segments:
            if gname == "arch one-hots":
                names += [f"self:{archetypes.arch_name_at(i)}"
                          for i in range(n_arch)]
                names += [f"opp:{archetypes.arch_name_at(i)}"
                          for i in range(n_arch)]
            elif gname == "match/lib/turn ctx":
                assert width == len(_META_CTX_NAMES), (width, _META_CTX_NAMES)
                names += list(_META_CTX_NAMES)
            else:
                names += [f"{gname}[{j}]" for j in range(width)]
        pa = int(sd[f"{base}.0.weight"].shape[1]) - len(names)
        names += [f"per-action[{j}]" for j in range(pa)]
        return names, f"{base}.0"
    known = ", ".join(sorted(list(enc) + ["policy_body", "value_body"]))
    raise ValueError(f"unknown layer {layer!r} (want one of: {known})")


def unit_profile(sd, layer, unit, top=12):
    """One hidden unit's signed input recipe: the strongest positive and
    negative input-variable weights of row ``unit`` in ``layer``'s first
    Linear. Exact — the row IS the unit's per-variable weighting; the sign
    structure shows what the unit contrasts against what."""
    names, prefix = layer_column_names(sd, layer)
    w = _t2np(sd[prefix + ".weight"])
    b = _t2np(sd[prefix + ".bias"])
    if not 0 <= int(unit) < w.shape[0]:
        raise ValueError(f"{prefix} has units 0..{w.shape[0] - 1}, not {unit}")
    row = w[int(unit)]
    assert len(names) == w.shape[1], (prefix, len(names), w.shape)
    order = np.argsort(row)
    pos = [(names[j], float(row[j])) for j in order[::-1] if row[j] > 0][:top]
    neg = [(names[j], float(row[j])) for j in order if row[j] < 0][:top]
    return {"layer": prefix, "unit": int(unit), "n_units": int(w.shape[0]),
            "in_dim": int(w.shape[1]), "bias": float(b[int(unit)]),
            "pos": pos, "neg": neg}


def resolve_bucket(query):
    """Value-bucket index for a query: an integer, or a bucket-name substring
    (``doomsday_vs_burn``). Raises with candidates when ambiguous."""
    q = str(query).strip().lower()
    n = archetypes.N_VALUE_BUCKETS
    if q.lstrip("-").isdigit():
        i = int(q)
        if not 0 <= i < n:
            raise ValueError(f"bucket index {i} out of range 0..{n - 1}")
        return i
    names = [archetypes.bucket_name(i) for i in range(n)]
    exact = [i for i, nm in enumerate(names) if nm.lower() == q]
    if exact:
        return exact[0]
    subs = [i for i, nm in enumerate(names) if q in nm.lower()]
    if len(subs) == 1:
        return subs[0]
    if not subs:
        raise ValueError(f"no value bucket matches {query!r}")
    cands = ", ".join(names[i] for i in subs[:8])
    more = "" if len(subs) <= 8 else f" (+{len(subs) - 8} more)"
    raise ValueError(f"{query!r} is ambiguous: {cands}{more}")


def bucket_input_connectivity(sd, bucket, top=15):
    """Weights-only relevance chain from ONE value bucket's head column back to
    the input variables: ``|w_bucket| · |W_L| · … · |W_0|`` composed through the
    value body, aggregated by input segment and named per column.

    This measures potential connectivity — which input pathways exist and how
    wide they are — not state-conditional influence (the ReLU/Tanh gates are
    ignored, which is exactly what makes it computable from weights alone; the
    exact per-state answer is gradient saliency and needs recorded states)."""
    body_keys = sorted(
        (k for k in sd if k.startswith("value_body.") and k.endswith(".weight")),
        key=lambda k: int(k.split(".")[1]))
    if not body_keys:
        raise ValueError("this checkpoint has no value_body (stock-policy "
                         "PPO?) — pathto needs the multi-head critic stack")
    rel = np.abs(_t2np(sd["value_head.weight"])[int(bucket)])
    for k in reversed(body_keys):
        rel = rel @ np.abs(_t2np(sd[k]))
    names, _ = layer_column_names(sd, "value_body")
    assert rel.shape[0] == len(names), (rel.shape, len(names))
    segments, _ = _body_segments(sd)
    rows = segments + [("per-action features", rel.shape[0]
                        - sum(w for _, w in segments))]
    total = float(rel.sum()) or 1.0
    groups, off = [], 0
    for gname, width in rows:
        seg = rel[off:off + width]
        groups.append({"name": gname, "width": int(width),
                       "share": float(seg.sum() / total),
                       "mean": float(seg.mean())})
        off += width
    order = np.argsort(-rel)[:top]
    return {"bucket": int(bucket), "name": archetypes.bucket_name(int(bucket)),
            "groups": groups,
            "top": [(names[j], float(rel[j])) for j in order]}


def weight_spectra(sd):
    """Singular-value health of every 2-D weight tensor: top σ, effective rank
    (exp of the σ² spectrum's entropy), and the rank capturing 90% of the
    energy. Rank collapse shows up as eff ≪ full; compared across snapshots it
    shows which layers are still learning. The frozen card_props buffer is
    skipped (constant by construction)."""
    rows = []
    for k, t in sd.items():
        if getattr(t, "ndim", 0) != 2 or k.endswith("card_props"):
            continue
        s = np.linalg.svd(_t2np(t), compute_uv=False)
        e = s ** 2
        total = float(e.sum())
        if total <= 0:
            rows.append({"name": k, "shape": tuple(t.shape), "top": 0.0,
                         "eff": 0.0, "r90": 0, "rank": int(len(s)), "sv": s})
            continue
        p = e / total
        ent = float(-(p * np.where(p > 0, np.log(np.where(p > 0, p, 1.0)),
                                   0.0)).sum())
        rows.append({"name": k, "shape": tuple(t.shape), "top": float(s[0]),
                     "eff": float(np.exp(ent)),
                     "r90": int(np.searchsorted(np.cumsum(p), 0.90) + 1),
                     "rank": int(len(s)), "sv": s})
    return rows


def popart_table(sd):
    """The PPO policy's PopArt buffers, per value bucket: the running return
    mean/std the head's normalized predictions are denormalized with, and the
    sample count that fitted them. Direct per-matchup training statistics no
    behavioral eval can see — a bucket with count 0 was never trained on.
    PPO-only: AZ checkpoints have no PopArt counterpart."""
    if "popart_mu" not in sd:
        raise ValueError("no PopArt buffers in this checkpoint — the popart "
                         "view reads a PPO .zip (AZ has no PopArt)")
    mu = _t2np(sd["popart_mu"])
    sigma = _t2np(sd["popart_sigma"])
    counts = _t2np(sd["popart_count"]) if "popart_count" in sd else None
    rows = []
    for i in range(len(mu)):
        n = float(counts[i]) if counts is not None else None
        trained = (n > 0) if n is not None else (mu[i] != 0.0
                                                 or sigma[i] != 1.0)
        rows.append({"bucket": i, "name": archetypes.bucket_name(i),
                     "mu": float(mu[i]), "sigma": float(sigma[i]),
                     "count": n, "trained": bool(trained)})
    return {"rows": rows, "n_trained": sum(r["trained"] for r in rows)}


def value_head_geometry(sd, tol=1e-8):
    """Pairwise cosine between the multi-head critic's bucket rows — which
    matchups the net values along the SAME latent direction. The buckets view
    maps which columns are alive; this measures how the alive ones relate: a
    near-1 pair is one value function reused for two matchups, a negative pair
    is two matchups the critic reads through opposed features."""
    w = _t2np(sd["value_head.weight"])
    b = _t2np(sd["value_head.bias"])
    norms = np.linalg.norm(w, axis=1)
    live = np.where(norms > tol)[0]
    sims = _unit(w) @ _unit(w).T
    pairs = [(int(live[x]), int(live[y]), float(sims[live[x], live[y]]))
             for x in range(len(live)) for y in range(x + 1, len(live))]
    pairs.sort(key=lambda t: -t[2])
    rows = [{"bucket": i, "name": archetypes.bucket_name(i),
             "norm": float(norms[i]), "bias": float(b[i]),
             "dead": bool(norms[i] <= tol)} for i in range(w.shape[0])]
    return {"rows": rows, "pairs": pairs, "n_live": int(len(live))}


def zone_embedding(sd):
    """The per-action zone-ref embedding ``(N_REF_ZONES, D)`` (per-action-head
    trunks only)."""
    key = "trunk.zone_emb.weight"
    if key not in sd:
        raise ValueError("this trunk has no zone embedding — the checkpoint "
                         "was built without the per-action head")
    return _t2np(sd[key])


def entity_unit_activations(sd):
    """Every vocab card pushed through the entity encoder alone: the
    ``(N_CARD_TYPES, embed_dim)`` unit-activation matrix. A pure numpy mirror
    of ``trunk.entity_encoder`` on ``[card_emb | card_props | draw_dist=0]``
    rows (row i = vocab card i; the padding row is dropped; distance 0 = the
    "in hand" reading, the value every zone except the known top-of-library
    feeds) — no game state involved, since that encoder consumes card identity
    plus that one scalar and nothing else."""
    ident = _t2np(sd["trunk.card_emb.weight"])[1:]
    x = np.concatenate([ident, _t2np(sd["trunk.card_props"])[1:],
                        np.zeros((ident.shape[0], 1), dtype=ident.dtype)], axis=1)
    h = np.maximum(x @ _t2np(sd["trunk.entity_encoder.0.weight"]).T
                   + _t2np(sd["trunk.entity_encoder.0.bias"]), 0.0)
    return np.maximum(h @ _t2np(sd["trunk.entity_encoder.2.weight"]).T
                      + _t2np(sd["trunk.entity_encoder.2.bias"]), 0.0)


def card_selectivity(sd, top_cards=6):
    """Entity-encoder units ranked by how card-SELECTIVE they are (peak
    activation over mean, across the named vocab), with each unit's
    top-activating cards — "this unit fires for counterspells". Dead units
    (never active for any card) are counted separately; they answer the
    activation-level question the static exposure view cannot."""
    acts = entity_unit_activations(sd)
    ids = named_card_ids()
    a = acts[ids]
    mx, mean, active = a.max(axis=0), a.mean(axis=0), (a > 0).mean(axis=0)
    dead = mx <= 0
    sel = np.where(dead, 0.0, mx / np.where(mean > 0, mean, 1.0))
    units = []
    for u in np.argsort(-sel):
        if dead[u]:
            continue
        top = np.argsort(-a[:, u])[:top_cards]
        units.append({"unit": int(u), "max": float(mx[u]),
                      "mean": float(mean[u]), "active_frac": float(active[u]),
                      "sel": float(sel[u]),
                      "top": [(int(ids[j]), float(a[j, u])) for j in top]})
    return {"n_units": int(acts.shape[1]), "n_dead": int(dead.sum()),
            "units": units}


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
             f"card rep   : {net.trunk.card_emb.weight.shape[1]} trainable "
             f"identity + {net.trunk.card_props.shape[1]} frozen printed props",
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


def neighbor_header(counts=None, exposure=None):
    """Column header matching what neighbor_row will annotate with."""
    tail = "seen" if counts is not None else ("Δemb" if exposure is not None
                                              else "")
    return f"  {'cos':>6}  {'card':<34} {'cost':<6} {'type':<13} {tail}"


# Back-compat alias for the plain (shard-annotated) header.
NEIGHBOR_HEADER = neighbor_header(counts=True)


def neighbor_row(idx, cos, counts=None, exposure=None):
    """One formatted neighbour line. Shared so the TUI's clickable list and the
    text view show identical columns. Annotated with occurrence counts when a
    shard sample is loaded, else with the weights-only embedding movement."""
    if counts is not None:
        tail = str(int(counts[int(idx)]))
    elif exposure is not None:
        d = float(exposure[int(idx)])
        tail = f"{d:.3f}" if d > 0 else "—"
    else:
        tail = ""
    return (f"  {cos:6.3f}  {card_name(idx)[:34]:<34} "
            f"{card_cost_colors(idx):<6} {card_primary_type(idx):<13} {tail}")


def render_neighbors(mat, idx, k=15, counts=None, candidates=None, rows=True,
                     exposure=None):
    """Header + neighbour rows for a card. ``rows=False`` returns the header
    alone, for callers (the TUI) that render the neighbours as a widget."""
    lines = [f"{card_name(idx)}  [{card_cost_colors(idx)} "
             f"mv{card_cmc(idx):.0f} {card_primary_type(idx)}]",
             f"  embedding row norm {np.linalg.norm(mat[idx]):.3f}"]
    untrained = ("  ⚠ its row never trained, so it still holds warm-start "
                 "values and these neighbours are meaningless")
    if counts is not None:
        seen = int(counts[idx])
        lines.append(f"  embedded in {seen} sampled states"
                     + (untrained if seen == 0 else ""))
    if exposure is not None:
        d = float(exposure[idx])
        lines.append(f"  embedding moved {d:.4f} since the baseline"
                     + (untrained if d == 0 else ""))
    if not rows:
        return lines
    lines.append("")
    lines.append(neighbor_header(counts, exposure))
    for j, cos in nearest_cards(mat, idx, k=k, candidates=candidates):
        lines.append(neighbor_row(j, cos, counts, exposure))
    return lines


def render_structure(mat, k=10, ids=None, counts=None, min_seen=0):
    ids = named_card_ids() if ids is None else np.asarray(ids)
    note = ""
    if counts is not None and min_seen > 0:
        keep = np.array([counts[i] >= min_seen for i in ids])
        # Neutral wording: the vector is occurrence counts with a shard sample
        # loaded, and a binary trained/untrained flag without one.
        note = (f" (restricted to the {int(keep.sum())} of {len(ids)} cards "
                f"passing --min-seen {min_seen})")
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


def render_occurrences(counts, n_states, top_n=25, revealed=None):
    ids = named_card_ids()
    order = sorted(ids, key=lambda i: -counts[i])
    zero = [i for i in ids if counts[i] == 0]
    lines = [f"card occurrences over {n_states} sampled decision states",
             f"  {len(ids) - len(zero)}/{len(ids)} named cards are EMBEDDED "
             f"(board zones, hand, graveyards/exiles, and the decklist blocks); "
             f"{len(zero)} never appear",
             "  'revealed' counts states where the card shows up ONLY in the "
             "dense opponent-revealed",
             "  multi-hot — that block never touches the card embedding, and it "
             "is sticky for the match.", "",
             f"  {'embedded':>8} {'':>6} {'revealed':>8}  card"]
    for i in order[:top_n]:
        frac = counts[i] / max(1, n_states)
        rev = "" if revealed is None else f"{int(revealed[i]):8d}"
        lines.append(f"  {counts[i]:8d} {frac*100:5.1f}% {rev:>8}  "
                     f"{card_name(i)[:34]:<34} {_bar(frac)}")
    if zero:
        lines.append("")
        lines.append(f"never embedded ({len(zero)}): "
                     + ", ".join(card_name(i) for i in zero[:40])
                     + (" …" if len(zero) > 40 else ""))
        lines.append("  (this is visibility in the SAMPLE; the 'exposure' view "
                     "answers it exactly from the weights)")
    return lines


def render_exposure(exp, top_n=20):
    """Which embedding rows / critic columns actually received gradient."""
    ids = named_card_ids()
    delta = exp["delta"]
    d = np.array([delta[i] for i in ids])
    moved = d > 0
    src = ("the previous AZ snapshot" if exp["kind"] == "snapshot"
           else "the PPO checkpoint AZ warm-started from")
    lines = [f"training exposure — {os.path.basename(exp['path'])} vs "
             f"{os.path.basename(exp['baseline'])}",
             f"  ({src}; a row that got no gradient is bit-identical, so "
             "'moved' is exact — az_train",
             "   exempts these tables from weight decay precisely so an "
             "untouched row cannot drift)", "",
             f"card embedding rows trained: {int(moved.sum())}/{len(ids)}"]
    top = np.argsort(-d)[:top_n]
    hi = d[top[0]] if len(top) and d[top[0]] > 0 else 1.0
    for j in top:
        if d[j] <= 0:
            break
        lines.append(f"  {d[j]:8.4f}  {card_name(ids[j])[:38]:<38} "
                     f"{_bar(d[j] / hi)}")
    frozen = [card_name(ids[j]) for j in np.nonzero(~moved)[0]]
    if frozen:
        lines.append("")
        lines.append(f"never trained ({len(frozen)}): " + ", ".join(frozen[:40])
                     + (" …" if len(frozen) > 40 else ""))
        lines.append("  their rows still hold warm-start values — exclude them "
                     "with --min-seen 1")
    bd = exp["bucket_delta"]
    bmoved = exp["bucket_moved"]
    lines.append("")
    lines.append(f"value-head columns trained: {int(bmoved.sum())}/{len(bd)}")
    cold = [archetypes.bucket_name(i) for i in np.nonzero(~bmoved)[0]]
    if cold:
        lines.append(f"  untrained matchups ({len(cold)}): "
                     + ", ".join(cold[:12]) + (" …" if len(cold) > 12 else ""))
    return lines


def render_drift(drift, top_n=0):
    """Per-dimension embedding movement + every card ranked by total |shift|."""
    ids = named_card_ids()
    d = drift["delta"][ids]                                    # (cards, dim)
    src = {"ppo": "the PPO checkpoint AZ warm-started from",
           "snapshot": "the oldest AZ snapshot on disk",
           "init": "a fresh untrained net — per-dim SIGNS vs a random init "
                   "are mostly noise; trust the magnitudes only"}[drift["kind"]]
    lines = [f"embedding drift — {os.path.basename(drift['path'])} vs "
             f"{drift['baseline']}",
             f"  ({src})", ""]
    mag = np.abs(d).mean(axis=0)
    net_shift = d.mean(axis=0)
    hi = mag.max() if mag.max() > 0 else 1.0
    lines.append(f"per-dimension movement over {len(ids)} cards, largest "
                 "first (mean |shift| / mean signed shift / % pushed +):")
    for k in np.argsort(-mag):
        pos = float((d[:, k] > 0).mean() * 100)
        lines.append(f"  dim {k:2d}  {mag[k]:8.4f}  {net_shift[k]:+8.4f}  "
                     f"{pos:5.1f}%+  {_bar(mag[k] / hi)}")
    l1 = np.abs(d).sum(axis=1)
    order = np.argsort(-l1)
    if top_n:
        order = order[:top_n]
    lines += ["", f"cards by total movement (L1 over all {d.shape[1]} dims; "
                  "strip = per-dim direction,",
              "  '·' = |shift| under 25% of that card's largest dim):"]
    for j in order:
        row = d[j]
        t = np.abs(row).max() * 0.25
        strip = "".join("+" if v > t else "-" if v < -t else "·" for v in row)
        lines.append(f"  {l1[j]:8.4f}  {card_name(ids[j])[:32]:<32} {strip}")
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


def render_block_importance(imp, top_n=20, single=False, sort="v"):
    what = "policy" if sort == "pi" else "evaluation"
    head = (f"what THIS {what} rests on" if single
            else f"what the heads rest on, over {imp['n']} states"
                 + (" (ranked by policy shift)" if sort == "pi" else ""))
    lines = [head,
             "  each block's floats are replaced by another real state's; "
             "|ΔV| is the value shift, Δπ the total-variation policy shift "
             "that causes.", ""]
    if single:
        lines.insert(1, f"  base V={imp['base']:+.3f}  "
                        f"({imp['donors']} donor states per block)")
    elif imp.get("only_active"):
        lines.insert(1, f"  states and donors restricted to the "
                        f"{imp['active_n']} sampled states where "
                        f"{imp['only_active']!r} is non-empty")
    lines.append("  Δh = value-latent movement; |Δz| = pre-tanh shift through "
                 "the state's own matchup")
    lines.append("  read-out (sel) vs the live-column mean (all) — sel ≪ all "
                 "means this matchup's read-out")
    lines.append("  discards what the block moved; Δh ≈ 0 with big |ΔV| is "
                 "pure read-out re-routing.")
    rows = imp["rows"]
    if sort == "pi":
        rows = sorted(rows, key=lambda r: -r["dpi"])
    top = rows[0]["dpi" if sort == "pi" else "delta"] or 1.0
    lines.append(f"  {'block':<24} {'width':>6} {'mean |ΔV|':>10} {'Δπ':>7}"
                 f" {'Δh':>7} {'|Δz|sel':>8} {'|Δz|all':>8}"
                 + ("  signed" if single else ""))
    for r in rows[:top_n]:
        sign = f"  {r['signed']:+.3f}" if single else ""
        bar = _bar((r["dpi"] if sort == "pi" else r["delta"]) / top)
        lines.append(f"  {r['name']:<24} {r['width']:6d} {r['delta']:10.4f}"
                     f" {r['dpi']:7.4f} {r['dlat']:7.3f} {r['dz_sel']:8.3f}"
                     f" {r['dz_all']:8.3f}{sign}  {bar}")
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
    lines.append(f"  {'tensor':<44} {'rel Δ':>8} {'|Δ|':>10} {'cos':>7}")
    for t in d["tensors"][:top_n]:
        cos = f"{t['cos']:7.4f}" if np.isfinite(t.get("cos", float("nan"))) \
            else "      -"
        lines.append(f"  {t['name'][:44]:<44} {t['rel']:8.4f} "
                     f"{t['delta']:10.4f} {cos}")
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


def render_first_layer(attr, top_cols=10):
    lines = ["first-layer input attribution — what each encoder actually reads",
             "  share = fraction of the layer's input weight energy on the "
             "group",
             "  rms   = RMS per-column weight norm (width-independent)"]
    for a in attr:
        lines.append("")
        lines.append(f"{a['encoder']}  ({a['in_dim']} → {a['out_dim']})")
        for g in sorted(a["groups"], key=lambda g: -g["share"]):
            lines.append(f"  {g['name']:<24} w={g['width']:<5d} "
                         f"{_bar(g['share'])} {g['share'] * 100:5.1f}%   "
                         f"rms {g['rms']:.3f}")
        if a["named"]:
            lines.append("  top named columns: "
                         + ", ".join(f"{n}({v:.2f})"
                                     for n, v in a["named"][:top_cols]))
    return lines


def render_body_layer(attr, top_arch=8):
    lines = ["policy/value body first-layer attribution — what the heads read "
             "from the pooled features",
             "  the matchup conditioning (arch one-hots, decklist aggregates) "
             "lives here",
             "  share = fraction of input weight energy; rms = per-column "
             "norm (width-independent)"]
    for a in attr:
        lines.append("")
        lines.append(f"{a['body']}.0  ({a['in_dim']} → {a['out_dim']})")
        for g in sorted(a["groups"], key=lambda g: -g["rms"]):
            lines.append(f"  {g['name']:<20} w={g['width']:<5d} "
                         f"{_bar(g['share'])} {g['share'] * 100:5.1f}%   "
                         f"rms {g['rms']:.3f}")
        lines.append("  arch columns: "
                     + ", ".join(f"{n}({v:.2f})"
                                 for n, v in a["arch"][:top_arch])
                     + (f", … {len(a['arch']) - top_arch} more"
                        if len(a["arch"]) > top_arch else ""))
    return lines


def render_unit(prof):
    lines = [f"{prof['layer']}  unit {prof['unit']} of {prof['n_units']}   "
             f"bias {prof['bias']:+.3f}",
             "  the unit's signed input recipe — reads the + list, is "
             "suppressed by the − list", ""]
    lines.append("  strongest positive inputs:")
    for n, v in prof["pos"]:
        lines.append(f"    {v:+7.3f}  {n}")
    lines.append("  strongest negative inputs:")
    for n, v in prof["neg"]:
        lines.append(f"    {v:+7.3f}  {n}")
    return lines


def render_pathto(conn):
    lines = [f"input connectivity of value bucket {conn['bucket']} "
             f"({conn['name']})",
             "  weights-only |w|-chain through the value body: potential "
             "pathway width per input,",
             "  NOT state-conditional influence (activation gating ignored)",
             ""]
    for g in sorted(conn["groups"], key=lambda g: -g["mean"]):
        lines.append(f"  {g['name']:<20} w={g['width']:<5d} "
                     f"{_bar(g['share'])} {g['share'] * 100:5.1f}%   "
                     f"mean {g['mean']:.3f}")
    lines.append("")
    lines.append("  widest individual input columns:")
    for n, v in conn["top"]:
        lines.append(f"    {v:8.3f}  {n}")
    return lines


def render_spectra(rows):
    lines = ["singular-value spectra — effective rank per weight matrix",
             "  eff ≪ full rank = the layer collapsed onto few directions",
             "",
             f"  {'tensor':<40} {'shape':>12} {'top σ':>8} {'eff':>7} "
             f"{'r90':>5} {'full':>5}"]
    for r in rows:
        shape = "x".join(str(s) for s in r["shape"])
        lines.append(f"  {r['name'][:40]:<40} {shape:>12} {r['top']:8.2f} "
                     f"{r['eff']:7.1f} {r['r90']:5d} {r['rank']:5d}  "
                     f"{_spark(r['sv'][:24])}")
    return lines


def render_popart(pt):
    rows = pt["rows"]
    lines = ["PopArt per-bucket value normalizer (PPO) — the running return "
             "mean/std",
             "  the value head predicts NORMALIZED returns; count = samples "
             "that fitted the bucket",
             f"  {pt['n_trained']}/{len(rows)} buckets ever trained", "",
             f"  {'bucket':<34} {'mu':>8} {'sigma':>8} {'count':>10}"]
    have_counts = bool(rows) and rows[0]["count"] is not None
    key = (lambda r: -r["count"]) if have_counts else (lambda r: r["bucket"])
    for r in sorted(rows, key=key):
        if not r["trained"]:
            continue
        cnt = f"{r['count']:10.0f}" if r["count"] is not None else "         ?"
        lines.append(f"  {r['name'][:34]:<34} {r['mu']:+8.3f} "
                     f"{r['sigma']:8.3f} {cnt}")
    untrained = [r for r in rows if not r["trained"]]
    if untrained:
        lines.append(f"  … {len(untrained)} untrained buckets omitted "
                     "(mu=0, sigma=1, count=0)")
    return lines


def render_value_geometry(geo, top_n=12):
    lines = ["value-head row geometry — cosine between matchup columns",
             "  ~1 = one value function reused for both matchups; negative = "
             "valued through opposed features",
             f"  {geo['n_live']}/{len(geo['rows'])} columns alive", ""]
    name = {r["bucket"]: r["name"] for r in geo["rows"]}
    if geo["pairs"]:
        lines.append("most similar live pairs:")
        for i, j, c in geo["pairs"][:top_n]:
            lines.append(f"  {c:+.3f}  {name[i]}  ~  {name[j]}")
        lines.append("")
        lines.append("most opposed live pairs:")
        for i, j, c in geo["pairs"][-min(top_n // 2, len(geo["pairs"])):]:
            lines.append(f"  {c:+.3f}  {name[i]}  ~  {name[j]}")
    return lines


def render_zone_embedding(mat, top_k=4):
    u = _unit(mat)
    sims = u @ u.T
    lines = ["zone-ref embedding — nearest zones by cosine",
             "  (does the net treat actions referencing these zones alike?)",
             ""]
    for z in range(mat.shape[0]):
        if np.linalg.norm(mat[z]) <= 1e-6:
            continue
        order = [j for j in np.argsort(-sims[z]) if j != z][:top_k]
        near = ", ".join(f"{_REF_NAMES.get(int(j), int(j))}({sims[z][j]:.2f})"
                         for j in order)
        lines.append(f"  {_REF_NAMES.get(z, z):<10} → {near}")
    return lines


def render_card_selectivity(sel, top_units=12, counts=None):
    lines = ["entity-encoder unit → card selectivity",
             "  sel = peak/mean activation over the named vocab; a selective "
             "unit is a card-class detector",
             f"  {sel['n_units'] - sel['n_dead']}/{sel['n_units']} units "
             f"alive ({sel['n_dead']} never fire for any card)", ""]
    for uinfo in sel["units"][:top_units]:
        top = ", ".join(f"{card_name(i)}({v:.2f})" for i, v in uinfo["top"])
        lines.append(f"  unit {uinfo['unit']:>3}  sel {uinfo['sel']:6.1f}  "
                     f"active {uinfo['active_frac'] * 100:4.1f}%  → {top}")
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
                    help="AZ checkpoint spec: 'gen' (the generalist) or a "
                         "path. The weight-space views (firstlayer, bodylayer, "
                         "spectra, popart, valuegeom, zoneemb, cardsel, diff) "
                         "also take a PPO .zip, and their 'gen' falls back to "
                         "the PPO generalist when no AZ checkpoint exists")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("overview", help="checkpoint meta + critic + shard status")
    _add_shard_args(p)
    p.add_argument("--no-shards", action="store_true",
                   help="skip the shard sample (weights only)")

    def _add_space_arg(p):
        p.add_argument("--space", default="identity", choices=CARD_SPACES,
                       help="card space to measure: the trainable identity "
                            "table (default), the frozen printed-property "
                            "block, or the full concatenation")

    p = sub.add_parser("neighbors", help="nearest cards in embedding space")
    p.add_argument("card", help="card name (exact or unique substring)")
    p.add_argument("-k", type=int, default=15, help="neighbours to show")
    _add_shard_args(p); _add_count_args(p); _add_space_arg(p)

    p = sub.add_parser("structure", help="kNN label purity of the embedding")
    p.add_argument("-k", type=int, default=10, help="neighbours per card")
    _add_shard_args(p); _add_count_args(p); _add_space_arg(p)

    p = sub.add_parser("clusters", help="k-means over the card embedding")
    p.add_argument("-k", type=int, default=8, help="clusters")
    p.add_argument("--cluster-seed", type=int, default=0)
    _add_shard_args(p); _add_count_args(p); _add_space_arg(p)

    p = sub.add_parser("project", help="PCA-to-2D terminal scatter")
    p.add_argument("--mark", default="color", choices=LABEL_KINDS)
    p.add_argument("--width", type=int, default=78)
    p.add_argument("--height", type=int, default=24)
    _add_shard_args(p); _add_count_args(p); _add_space_arg(p)

    p = sub.add_parser("occur", help="how often each card appears in self-play")
    _add_shard_args(p)
    p.add_argument("--count-rows", type=int, default=1500,
                   help="states to decode (default: 1500)")
    p.add_argument("--top", type=int, default=25)

    p = sub.add_parser("exposure",
                       help="which embedding rows / critic columns actually "
                            "trained (weights only, no shards)")
    p.add_argument("--baseline", default=None,
                   help="checkpoint to measure against (default: the previous "
                        "gen__azv* snapshot, else the PPO gen warm-start)")
    p.add_argument("--top", type=int, default=20)

    p = sub.add_parser("drift",
                       help="signed per-dimension embedding movement since "
                            "the origin; all cards ranked by total |shift|")
    p.add_argument("--baseline", default=None,
                   help="checkpoint to measure against (default: the PPO gen "
                        "warm-start, else the oldest gen__azv* snapshot; "
                        "'init' = a fresh untrained seed-0 net)")
    p.add_argument("--top", type=int, default=0,
                   help="cards listed (default: 0 = all)")

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
    p.add_argument("--only-active", default=None, metavar="BLOCK",
                   help="restrict states and donors to those where BLOCK "
                        "(name or unique substring) is non-empty — scores a "
                        "sparse block on the states where it exists")
    p.add_argument("--sort", choices=("v", "pi"), default="v",
                   help="rank by value shift (v) or policy shift (pi)")
    p.add_argument("--top", type=int, default=20)

    p = sub.add_parser("readout", help="one state's V under every matchup "
                                       "critic column (read-out vs prior)")
    _add_shard_args(p)
    p.add_argument("--row", type=int, default=0, help="which sampled decision")
    p.add_argument("--top", type=int, default=None,
                   help="show only the top-N columns by V (default: all)")

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

    p = sub.add_parser("firstlayer",
                       help="first-layer input-column attribution per encoder")
    p.add_argument("--encoder", default=None,
                   help="one encoder (e.g. perm_encoder; default: all)")
    p.add_argument("--top", type=int, default=10,
                   help="named columns to show per encoder")

    p = sub.add_parser("bodylayer",
                       help="policy/value body first-layer attribution "
                            "(arch one-hots + decklist aggregates)")
    p.add_argument("--top-arch", type=int, default=16,
                   help="archetype columns to name (default: all 16)")

    p = sub.add_parser("unit",
                       help="one hidden unit's signed input-variable profile")
    p.add_argument("layer",
                   help="layer name: perm_encoder, stack_encoder, "
                        "entity_encoder, decklist_encoder, revealed_encoder, "
                        "action_encoder, policy_body, value_body")
    p.add_argument("unit", type=int, help="unit (row) index in that layer")
    p.add_argument("--top", type=int, default=12,
                   help="inputs to show per sign")

    p = sub.add_parser("pathto",
                       help="weights-only input connectivity of one value "
                            "bucket's head column")
    p.add_argument("bucket",
                   help="bucket index or name substring "
                        "(e.g. doomsday_vs_burn)")
    p.add_argument("--top", type=int, default=15,
                   help="widest individual input columns to show")

    sub.add_parser("spectra",
                   help="singular-value spectrum / effective rank per "
                        "weight matrix")

    sub.add_parser("popart",
                   help="PPO PopArt per-bucket value statistics (PPO .zip)")

    p = sub.add_parser("valuegeom",
                       help="value-head row geometry (which matchups share "
                            "a value direction)")
    p.add_argument("--top", type=int, default=12)

    sub.add_parser("zoneemb", help="zone-ref embedding neighbours")

    p = sub.add_parser("cardsel",
                       help="entity-encoder units ranked by card selectivity")
    p.add_argument("--units", type=int, default=12, help="units to show")
    p.add_argument("--cards", type=int, default=6,
                   help="top cards listed per unit")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    cmd = args.cmd

    if cmd == "catemb":
        net, _ = load_net(args.model)
        print("\n".join(render_category_embedding(net)))
        return 0

    if cmd == "diff":
        a = resolve_weights_spec(args.model)
        b = resolve_weights_spec(args.other)
        if a is None or b is None:
            raise SystemExit(f"could not resolve both checkpoints "
                             f"({args.model!r} -> {a}, {args.other!r} -> {b})")
        print("\n".join(render_diff(checkpoint_diff(a, b, top_n=args.top),
                                    top_n=args.top)))
        return 0

    if cmd in ("firstlayer", "bodylayer", "unit", "pathto", "spectra",
               "popart", "valuegeom", "zoneemb", "cardsel"):
        sd, path, kind = load_weight_state(args.model)
        print(f"[{kind}] {path}")
        if cmd == "bodylayer":
            print("\n".join(render_body_layer(body_layer_attribution(sd),
                                              top_arch=args.top_arch)))
        elif cmd == "unit":
            try:
                prof = unit_profile(sd, args.layer, args.unit, top=args.top)
            except ValueError as e:
                raise SystemExit(str(e))
            print("\n".join(render_unit(prof)))
        elif cmd == "pathto":
            try:
                conn = bucket_input_connectivity(sd, resolve_bucket(args.bucket),
                                                 top=args.top)
            except ValueError as e:
                raise SystemExit(str(e))
            print("\n".join(render_pathto(conn)))
        elif cmd == "firstlayer":
            enc = [args.encoder] if args.encoder else None
            attr = first_layer_attribution(sd, enc)
            if not attr:
                names = ", ".join(e for e, _ in _encoder_specs(sd))
                raise SystemExit(f"unknown encoder {args.encoder!r} "
                                 f"(want one of: {names})")
            print("\n".join(render_first_layer(attr, top_cols=args.top)))
        elif cmd == "spectra":
            print("\n".join(render_spectra(weight_spectra(sd))))
        elif cmd == "popart":
            try:
                pt = popart_table(sd)
            except ValueError as e:
                raise SystemExit(str(e))
            print("\n".join(render_popart(pt)))
        elif cmd == "valuegeom":
            print("\n".join(render_value_geometry(value_head_geometry(sd),
                                                  top_n=args.top)))
        elif cmd == "zoneemb":
            try:
                zmat = zone_embedding(sd)
            except ValueError as e:
                raise SystemExit(str(e))
            print("\n".join(render_zone_embedding(zmat)))
        else:
            sel = card_selectivity(sd, top_cards=args.cards)
            print("\n".join(render_card_selectivity(sel,
                                                    top_units=args.units)))
        return 0

    if cmd == "occur":
        s = load_shard_sample(args.shards, max_rows=max(args.max_rows,
                                                        args.count_rows),
                              window=args.window, seed=args.seed)
        emb, rev, n = card_occurrence_split(s["obs"], limit=args.count_rows,
                                            seed=args.seed)
        print("\n".join(render_occurrences(emb, n, top_n=args.top,
                                           revealed=rev)))
        return 0

    if cmd == "exposure":
        path = resolve_spec(args.model)
        if path is None:
            raise SystemExit(f"could not resolve checkpoint {args.model!r}")
        print("\n".join(render_exposure(card_exposure(path, args.baseline),
                                        top_n=args.top)))
        return 0

    if cmd == "drift":
        path = resolve_spec(args.model)
        if path is None:
            raise SystemExit(f"could not resolve checkpoint {args.model!r}")
        print("\n".join(render_drift(embedding_drift(path, args.baseline),
                                     top_n=args.top)))
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

    if cmd in ("state", "blocks", "swap", "sweep", "readout"):
        s = _sample(args)
        row = getattr(args, "row", 0)
        if row is not None and not 0 <= row < s["obs"].shape[0]:
            raise SystemExit(f"--row {row} out of range "
                             f"(0..{s['obs'].shape[0] - 1} in this sample)")
        if cmd == "state":
            print("\n".join(render_state(s, row, net=net, top_n=args.top)))
        elif cmd == "readout":
            print("\n".join(render_value_readouts(
                value_readouts(net, s["obs"][row]), top_n=args.top)))
        elif cmd == "blocks":
            if args.row is None:
                imp = block_importance(net, s, n_rows=args.rows,
                                       donors=args.donors, seed=args.seed,
                                       only_active=args.only_active)
                print("\n".join(render_block_importance(imp, top_n=args.top,
                                                        sort=args.sort)))
            else:
                if args.only_active:
                    raise SystemExit("--only-active applies to the aggregate "
                                     "view; --row already names one state")
                imp = state_block_importance(net, s, row, seed=args.seed)
                print("\n".join(render_block_importance(imp, top_n=args.top,
                                                        single=True,
                                                        sort=args.sort)))
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

    mat = card_matrix(net, getattr(args, "space", "identity"))
    counts = _maybe_counts(args)
    # Without --with-counts, annotate/filter with the weights-only exposure
    # instead of loading shards (same signal the TUI defaults to).
    exposure = None if counts is not None else exposure_counts_or_none(path)
    filt = filter_vector(counts, exposure)

    if cmd == "neighbors":
        idx = resolve_card(args.card)
        cand = None
        if filt is not None and args.min_seen > 0:
            cand = np.array([i for i in named_card_ids()
                             if filt[i] >= args.min_seen])
        print("\n".join(render_neighbors(mat, idx, k=args.k, counts=counts,
                                         candidates=cand, exposure=exposure)))
    elif cmd == "structure":
        print("\n".join(render_structure(mat, k=args.k, counts=filt,
                                         min_seen=args.min_seen)))
    elif cmd == "clusters":
        print("\n".join(render_clusters(mat, k=args.k, seed=args.cluster_seed,
                                        counts=filt, min_seen=args.min_seen)))
    elif cmd == "project":
        print("\n".join(render_projection(mat, width=args.width,
                                          height=args.height, mark=args.mark,
                                          counts=filt,
                                          min_seen=args.min_seen)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
