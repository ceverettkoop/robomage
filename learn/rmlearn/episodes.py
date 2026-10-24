"""Trajectory episodes (docs/trajectory_format.md) in numpy: the reader and the expansion of steps
into the network contract's padded tensors (docs/rnad_training_plan.md §4 L1).

    ep = read_episode("runs/r1/episodes/a3-m17.g2.rmtraj")   # Episode: the file's compact arrays
    ep.attrs["result_a"], ep.steps, ep.omniscient
    x = expand([(ep, 0), (ep, 5), (other, 2)])              # {name: ndarray} padded [B, ...] inputs
    x = expand(items, viewer="omni")                        # the omniscient featurization's
    t = targets(items, n_cards, depth)                      # the auxiliary heads' targets

**In memory an episode is its file**: the bytes are read once and every array is a read-only view
of them (``rmtb.read_bytes``), so a window of episodes costs what the files cost on disk (about
5 KB per decision for the seat's tensors, ~15 KB with the omniscient part). The reader adds a few
``[D + 1]`` offset arrays per featurization (where each step's categorical, card and ability
values start: the per-type widths' running sums at each step's first token) and nothing per token.

**Expansion** (``expand``) is §3.3 of the format, vectorised over a whole batch of steps that may
come from many episodes: one slice per array and step, then one scatter per tensor. The output is
exactly ``batch.pad`` of the same decisions (each dimension padded to the batch's largest N, R, A
rounded up to a multiple of 8, at least 8; pads 0, ``tok_card`` / ``tok_ability`` -1, masks 0),
with ``tok_num`` the stored f16 values widened to f32. ``act_tok`` is rebuilt from the episode's
``menu_type``: the menu-type tokens in order.

Nothing here names a token type: the widths (``F``, ``C``, ``K``, ``Q``), the per-type slot widths
(``type_widths``) and the menu's type come from each file, so the reader follows the spec the
episodes were written with. Steps of one batch must share a spec (``Episode.spec_hash``).
"""

import json
import os
import struct

import numpy as np

from . import rmtb

KIND = "episode"
VERSION = 3
PAD_MULTIPLE = 8
INPUTS = ["tok_type", "tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask", "rel", "rel_mask", "act_tok",
          "act_mask"]
STEP_FIELDS = ("seat", "game", "decision", "sideboarding", "chosen", "has_mu", "policy_version", "teacher")


def padded(size, multiple=PAD_MULTIPLE):
    return max(multiple, (size + multiple - 1) // multiple * multiple)


def _starts(sizes):
    s = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=s[1:])
    return s


def _need(arrays, name):
    a = arrays.get(name)
    if a is None:
        raise ValueError("episode: no array %s" % name)
    return a


def _offsets(arrays, name, steps, total):
    off = _need(arrays, name)
    if off.shape != (steps + 1,) or (steps >= 0 and (off[0] != 0 or off[-1] != total)) or \
            (steps and (np.diff(off) < 0).any()):
        raise ValueError("episode: %s does not tile its array" % name)
    return off


class Featurization:
    """One viewer's compact token arrays of an episode (``prefix`` "" the seat's, "omni_" the
    omniscient), with per step where its tokens, numeric values, relations and categorical / card /
    ability values start."""

    def __init__(self, arrays, prefix, steps, widths, dims):
        g = lambda n: _need(arrays, prefix + n)  # noqa: E731
        self.tok_type = g("tok_type")
        self.num_index, self.num_value = g("tok_num_index"), g("tok_num_value")
        self.cat, self.card, self.ability = g("tok_cat"), g("tok_card"), g("tok_ability")
        self.rel = g("rel")
        if self.rel.ndim != 2 or self.rel.shape[1] != 3:
            raise ValueError("episode: %srel is not [R, 3]" % prefix)
        self.tok_off = _offsets(arrays, prefix + "tok_offsets", steps, len(self.tok_type))
        self.num_off = _offsets(arrays, prefix + "tok_num_offsets", steps, len(self.num_index))
        self.rel_off = _offsets(arrays, prefix + "rel_offsets", steps, len(self.rel))
        if len(self.tok_type) and int(self.tok_type.max()) >= len(widths):
            raise ValueError("episode: a token type out of range")
        # Each step's first categorical / card / ability value: the per-type widths' running sums.
        w = widths[self.tok_type.astype(np.int64)]                       # [ΣN, 4]
        run = np.zeros((len(self.tok_type) + 1, 3), np.int64)
        np.cumsum(w[:, 1:], axis=0, out=run[1:])
        at = run[self.tok_off]                                           # [D + 1, 3]
        self.cat_off, self.card_off, self.ab_off = at[:, 0].copy(), at[:, 1].copy(), at[:, 2].copy()
        for name, arr, total in (("tok_cat", self.cat, run[-1, 0]), ("tok_card", self.card, run[-1, 1]),
                                 ("tok_ability", self.ability, run[-1, 2])):
            if len(arr) != total:
                raise ValueError("episode: %s%s holds %d values, its tokens' widths %d" % (prefix, name, len(arr),
                                                                                           total))
        F = dims[0]
        if len(self.num_index):
            n_of = np.repeat(np.diff(self.tok_off), np.diff(self.num_off))
            if (self.num_index.astype(np.int64) >= n_of * F).any():
                raise ValueError("episode: a numeric position past its decision's tokens")
        if len(self.rel):
            n_of = np.repeat(np.diff(self.tok_off), np.diff(self.rel_off))
            if (self.rel[:, :2].astype(np.int64) >= n_of[:, None]).any():
                raise ValueError("episode: a relation names no token of its decision")

    def tokens(self, d):
        return int(self.tok_off[d + 1] - self.tok_off[d])

    def card_ids(self):
        """The distinct card ids of the episode's card slots (ascending, i64), computed once."""
        ids = getattr(self, "_card_ids", None)
        if ids is None:
            card = np.asarray(self.card, np.int64)
            ids = self._card_ids = np.unique(card[card >= 0])
        return ids


class Episode:
    """One episode file in memory: ``meta``, ``attrs``, the per-step arrays (``seat``, ``game``,
    ``decision``, ``sideboarding``, ``chosen``, ``has_mu``, ``policy_version``, ``teacher``,
    ``menu_off``, ``mu``), ``evals`` (None or ``eval_policy`` / ``eval_version`` / ``eval_value``
    [D, P], ``eval_log_pi`` [ΣA, P]), ``seat_feat`` and ``omni_feat`` (``Featurization``, the latter
    None unless omniscient), the targets (ragged ``opp_hand``, ``own_library``, ``opp_library`` as
    (values, offsets); None unless omniscient)."""

    def __init__(self, data, path=None):
        meta, arrays = rmtb.read_bytes(data)
        attrs = meta.get("attrs", {})
        if meta.get("kind") != KIND:
            raise ValueError("not an episode file (kind %r)" % meta.get("kind"))
        if attrs.get("episode_version") != VERSION:
            raise ValueError("episode version %r, expected %d" % (attrs.get("episode_version"), VERSION))
        self.path = path
        self.name = os.path.basename(path) if path else None
        self.nbytes = len(data)
        self.meta, self.attrs = meta, attrs
        self.spec_hash = meta.get("spec_hash")
        self.dims = tuple(int(attrs[k]) for k in ("F", "C", "K", "Q"))
        self.menu_type = int(attrs["menu_type"])
        self.omniscient = bool(attrs["omniscient"])
        self.widths = np.ascontiguousarray(_need(arrays, "type_widths"), dtype=np.int64)
        if self.widths.ndim != 2 or self.widths.shape[1] != 4:
            raise ValueError("episode: type_widths is not [T, 4]")
        if (self.widths > np.array(self.dims)).any():
            raise ValueError("episode: a type's slot widths pass F, C, K, Q")
        self.seat = _need(arrays, "seat")
        self.steps = D = len(self.seat)
        for k in STEP_FIELDS[1:]:
            v = _need(arrays, k)
            if len(v) != D:
                raise ValueError("episode: the per-decision arrays differ in length")
            setattr(self, k, v)
        self.mu = _need(arrays, "mu")
        self.menu_off = _offsets(arrays, "menu_offsets", D, len(self.mu))
        self.seat_feat = Featurization(arrays, "", D, self.widths, self.dims)
        step_of = np.repeat(np.arange(D), np.diff(self.seat_feat.tok_off))
        menu = np.bincount(step_of[self.seat_feat.tok_type == self.menu_type], minlength=D)
        if not np.array_equal(menu, np.diff(self.menu_off)):
            raise ValueError("episode: a decision's menu tokens are not its menu_offsets rows")
        self.evals = None
        if "eval_policy" in arrays:
            self.evals = {k: arrays[k] for k in ("eval_policy", "eval_version", "eval_value", "eval_log_pi")}
        self.omni_feat = None
        self.targets = None
        if self.omniscient:
            self.omni_feat = Featurization(arrays, "omni_", D, self.widths, self.dims)
            self.targets = {}
            for n in ("opp_hand", "own_library", "opp_library"):
                vals = _need(arrays, n)
                self.targets[n] = (vals, _offsets(arrays, n + "_offsets", D, len(vals)))
        self.result = (int(attrs.get("result_a", 0)), int(attrs.get("result_b", 0)))

    def __len__(self):
        return self.steps

    def menu_size(self, d):
        return int(self.menu_off[d + 1] - self.menu_off[d])

    def outcome(self):
        """[D] f32: each step's deciding seat's result (+1 / -1 / 0)."""
        r = np.array(self.result, np.float32)
        return r[self.seat.astype(np.int64)]

    def game_key(self):
        """The game's identity across runs: (the match's engine seed, the game index, the log's name)."""
        return "%s:%s:%s" % (self.attrs.get("engine_seed"), self.attrs.get("game"),
                             os.path.basename(str(self.attrs.get("rmlog", ""))))

    def match_key(self):
        """The match's identity across runs: (its engine seed, the log's name)."""
        return "%s:%s" % (self.attrs.get("engine_seed"), os.path.basename(str(self.attrs.get("rmlog", ""))))


def episode_attrs(path):
    """An episode file's attrs (header and footer) from its meta alone, without its arrays."""
    with open(path, "rb") as fp:
        head = fp.read(12)
        magic, version, meta_len = struct.unpack("<4sII", head)
        if magic != rmtb.MAGIC:
            raise ValueError("%s: not an RMTB file" % path)
        meta = json.loads(fp.read(meta_len).decode("utf-8"))
    if meta.get("kind") != KIND:
        raise ValueError("%s: not an episode file" % path)
    return meta.get("attrs", {})


def read_episode(path):
    with open(path, "rb") as fp:
        return Episode(fp.read(), path)


# ── expansion ────────────────────────────────────────────────────────────────────────────────

def _feat(ep, viewer):
    f = ep.seat_feat if viewer == "seat" else ep.omni_feat
    if f is None:
        raise ValueError("episode %s has no %s featurization" % (ep.name, viewer))
    return f


def _check_same_spec(items):
    e0 = items[0][0]
    for ep, _ in items:
        if ep is not e0 and (ep.dims != e0.dims or ep.menu_type != e0.menu_type or
                             not np.array_equal(ep.widths, e0.widths)):
            raise ValueError("expand: the steps' episodes are of different specs")


def expand(items, viewer="seat", min_dims=None):
    """The contract's padded inputs ({name: ndarray}, as ``batch.pad``) of the steps ``items``
    ([(Episode, step)]); ``viewer`` "seat" or "omni". Also ``sizes`` ({"n", "r", "a"} → [B])."""
    if not items:
        raise ValueError("expand: no steps")
    _check_same_spec(items)
    e0 = items[0][0]
    F, C, K, Q = e0.dims
    widths, menu_type = e0.widths, e0.menu_type
    b = len(items)
    types, nums_i, nums_v, cats, cards, abs_, rels = [], [], [], [], [], [], []
    n = np.empty(b, np.int64)
    r = np.empty(b, np.int64)
    k_num = np.empty(b, np.int64)
    for j, (ep, d) in enumerate(items):
        f = _feat(ep, viewer)
        t0, t1 = f.tok_off[d], f.tok_off[d + 1]
        types.append(f.tok_type[t0:t1])
        n[j] = t1 - t0
        a0, a1 = f.num_off[d], f.num_off[d + 1]
        nums_i.append(f.num_index[a0:a1])
        nums_v.append(f.num_value[a0:a1])
        k_num[j] = a1 - a0
        cats.append(f.cat[f.cat_off[d]:f.cat_off[d + 1]])
        cards.append(f.card[f.card_off[d]:f.card_off[d + 1]])
        abs_.append(f.ability[f.ab_off[d]:f.ab_off[d + 1]])
        r0, r1 = f.rel_off[d], f.rel_off[d + 1]
        rels.append(f.rel[r0:r1])
        r[j] = r1 - r0
    md = min_dims or {}
    tt = np.concatenate(types).astype(np.int64)
    is_menu = tt == menu_type
    tok_b = np.repeat(np.arange(b), n)
    tok_pos = np.arange(len(tt)) - np.repeat(_starts(n)[:-1], n)
    a = np.bincount(tok_b[is_menu], minlength=b).astype(np.int64)
    N = max(padded(int(n.max())), md.get("N", 0))
    R = max(padded(int(r.max())), md.get("R", 0))
    A = max(padded(int(a.max())), md.get("A", 0))
    x = {
        "tok_type": np.zeros((b, N), np.int64), "tok_num": np.zeros((b, N, F), np.float32),
        "tok_cat": np.zeros((b, N, C), np.int64), "tok_card": np.full((b, N, K), -1, np.int64),
        "tok_ability": np.full((b, N, Q), -1, np.int64), "tok_mask": np.zeros((b, N), np.bool_),
        "rel": np.zeros((b, R, 3), np.int64), "rel_mask": np.zeros((b, R), np.bool_),
        "act_tok": np.zeros((b, A), np.int64), "act_mask": np.zeros((b, A), np.bool_),
    }
    x["tok_type"][tok_b, tok_pos] = tt
    x["tok_mask"][tok_b, tok_pos] = True
    # Numerics: each value's flat position within its step's [N, F].
    idx = np.concatenate(nums_i).astype(np.int64)
    vb = np.repeat(np.arange(b), k_num)
    x["tok_num"][vb, idx // F, idx % F] = np.concatenate(nums_v).astype(np.float32)
    # Per-type-width slots: token i of type t takes the next widths[t, k] values.
    for k, (name, vals) in enumerate((("tok_cat", cats), ("tok_card", cards), ("tok_ability", abs_)), start=1):
        w = widths[tt, k]
        total = int(w.sum())
        if not total:
            continue
        tok = np.repeat(np.arange(len(tt)), w)
        col = np.arange(total) - np.repeat(_starts(w)[:-1], w)
        x[name][tok_b[tok], tok_pos[tok], col] = np.concatenate(vals).astype(np.int64)
    rel = np.concatenate(rels).astype(np.int64) if len(rels) else np.zeros((0, 3), np.int64)
    rb = np.repeat(np.arange(b), r)
    rp = np.arange(len(rel)) - np.repeat(_starts(r)[:-1], r)
    x["rel"][rb, rp] = rel
    x["rel_mask"][rb, rp] = True
    mb = tok_b[is_menu]
    mp = np.arange(int(a.sum())) - np.repeat(_starts(a)[:-1], a)
    x["act_tok"][mb, mp] = tok_pos[is_menu]
    x["act_mask"][mb, mp] = True
    x["sizes"] = {"n": n, "r": r, "a": a}
    return x


def menu_rows(items, A, field="mu", fill=0.0, dtype=np.float32):
    """[B, A]: a per-menu-row array (``mu``) of the steps ``items``, padded with ``fill``."""
    out = np.full((len(items), A), fill, dtype)
    for j, (ep, d) in enumerate(items):
        m0, m1 = ep.menu_off[d], ep.menu_off[d + 1]
        out[j, :m1 - m0] = getattr(ep, field)[m0:m1]
    return out


def step_field(items, name, dtype=np.int64):
    """[B]: a per-step array (``seat``, ``chosen``, ``teacher``, …) of the steps ``items``."""
    return np.fromiter((getattr(ep, name)[d] for ep, d in items), dtype, len(items))


def evaluations(items, policy_id, A):
    """The recorded evaluations of policy ``policy_id`` at the steps ``items``: {``present`` [B]
    bool, ``version`` [B] i64 (-1 absent), ``value`` [B] f32, ``log_pi`` [B, A] f32 (0 absent / pad)}."""
    b = len(items)
    present = np.zeros(b, np.bool_)
    version = np.full(b, -1, np.int64)
    value = np.zeros(b, np.float32)
    log_pi = np.zeros((b, A), np.float32)
    for j, (ep, d) in enumerate(items):
        ev = ep.evals
        if ev is None:
            continue
        cols = np.nonzero(ev["eval_policy"][d] == policy_id)[0]
        if not len(cols):
            continue
        c = int(cols[0])
        present[j] = True
        version[j] = ev["eval_version"][d, c]
        value[j] = ev["eval_value"][d, c]
        m0, m1 = ep.menu_off[d], ep.menu_off[d + 1]
        log_pi[j, :m1 - m0] = ev["eval_log_pi"][m0:m1, c]
    return {"present": present, "version": version, "value": value, "log_pi": log_pi}


def targets(items, n_cards, depth):
    """The auxiliary heads' targets of the steps ``items`` (omniscient episodes; others get none:
    -1 / empty), as numpy: ``opp_hand`` [B, n_cards] counts, ``own_library``, ``opp_library``
    [B, depth] (-1 none); and ``present`` [B] bool."""
    b = len(items)
    out = {"opp_hand": np.zeros((b, n_cards), np.float32), "own_library": np.full((b, depth), -1, np.int64),
           "opp_library": np.full((b, depth), -1, np.int64), "present": np.zeros(b, np.bool_)}
    for j, (ep, d) in enumerate(items):
        t = ep.targets
        if t is None:
            continue
        out["present"][j] = True
        vals, off = t["opp_hand"]
        np.add.at(out["opp_hand"][j], vals[off[d]:off[d + 1]].astype(np.int64), 1.0)
        for name in ("own_library", "opp_library"):
            vals, off = t[name]
            v = vals[off[d]:off[d + 1]][:depth]
            out[name][j, :len(v)] = v
    return out
