"""Featurized decisions, their padding into the network contract's inputs, and learner targets.

A ``Decisions`` holds decisions unpadded and concatenated, as an RMTB ``tensors`` file stores them
(docs/featurizer.md §3) and an RMIP request carries them (docs/inference_protocol.md §4): per
tensor its values over every decision, plus per decision its token, relation and menu-row counts.

    decs = Decisions.from_file("sample_A.rmtb")
    x = pad(decs.select([0, 3, 5]), spec)          # {name: ndarray} padded [B, ...] inputs
    inputs = to_torch(x, device)                    # the ten tensors in contract order

``pad`` is ``featurize::pad_batch`` (``src/featurize/batch.cpp``): each dimension padded to the
batch's largest N, R and A rounded up to a multiple of 8 and at least 8; pads 0 (``tok_card`` and
``tok_ability`` -1) and every mask 0; ``rel_mask`` marks the real relations. ``min_dims`` pads
further (the padding-invariance check).

``forward_groups`` cuts decisions sorted by token count into the forwards of a batch too large for
one: attention's scores are rows × N² per head and layer (``attention_cells``), so a few long
decisions bound how many rows a forward can take.

``Targets`` reads an RMTB ``targets`` file (``rmfeat --targets``, ``src/featurize/targets.h``) and
gives per decision the auxiliary heads' targets.
"""

import numpy as np
import torch

from . import rmtb

INPUTS = ["tok_type", "tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask", "rel", "rel_mask", "act_tok",
          "act_mask"]
PAD_MULTIPLE = 8
# The per-decision tensors of a tensors file / request: name, the size counting its rows.
TENSORS = [("tok_type", "n"), ("tok_num", "n"), ("tok_cat", "n"), ("tok_card", "n"), ("tok_ability", "n"),
           ("tok_mask", "n"), ("rel", "r"), ("act_tok", "a"), ("act_mask", "a")]
# The per-decision scalars of a tensors file.
DECISION_FIELDS = ["viewer", "game", "decision", "sideboarding"]


def padded(size, multiple=PAD_MULTIPLE):
    return max(multiple, (size + multiple - 1) // multiple * multiple)


def _starts(sizes):
    s = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=s[1:])
    return s


def attention_cells(rows, n):
    """The attention scores of a forward of `rows` decisions padded to `n` tokens, per head and layer."""
    return rows * padded(n) ** 2


def forward_groups(n, order, max_rows=0, max_tokens=0, max_attention=0):
    """`order` (decision indices by ascending token count `n[i]`) cut into consecutive groups, one
    forward each, each padded to its last decision's count N: a group grows while it stays within
    `max_rows` rows, `max_tokens` padded tokens (rows × N) and `max_attention` attention cells
    (rows × N², ``attention_cells``), each 0 for no limit; a decision alone is always a group."""
    groups, cur = [], []
    for i in order:
        k, m = len(cur) + 1, padded(int(n[i]))
        if cur and ((max_rows and k > max_rows) or (max_tokens and k * m > max_tokens) or
                    (max_attention and attention_cells(k, m) > max_attention)):
            groups.append(np.array(cur, np.int64))
            cur = []
        cur.append(i)
    if cur:
        groups.append(np.array(cur, np.int64))
    return groups


class Decisions:
    """Unpadded decisions: `arrays` (name → the values concatenated over decisions), `sizes`
    ({"n", "r", "a"} → [D] i64) and `fields` (name → [D] per-decision scalars, may be empty)."""

    def __init__(self, arrays, sizes, fields=None):
        self.arrays = arrays
        self.sizes = sizes
        self.fields = fields or {}

    def __len__(self):
        return len(self.sizes["n"])

    @classmethod
    def from_file(cls, path):
        meta, arr = rmtb.read(path)
        if meta.get("kind") != "tensors":
            raise ValueError("%s is a %r file, not tensors" % (path, meta.get("kind")))
        sizes = {"n": np.diff(arr["tok_type_offsets"]), "r": np.diff(arr["rel_offsets"]),
                 "a": np.diff(arr["act_tok_offsets"])}
        arrays = {name: arr[name] for name, _ in TENSORS}
        fields = {k: arr[k] for k in DECISION_FIELDS if k in arr}
        d = cls(arrays, sizes, fields)
        d.meta = meta
        return d

    def select(self, indices):
        """The decisions at `indices` (in that order) as a new Decisions."""
        indices = np.asarray(indices, np.int64)
        starts = {k: _starts(v) for k, v in self.sizes.items()}
        arrays = {}
        for name, key in TENSORS:
            st = starts[key]
            arrays[name] = np.concatenate([self.arrays[name][st[i]:st[i + 1]] for i in indices]) \
                if len(indices) else self.arrays[name][:0]
        sizes = {k: v[indices] for k, v in self.sizes.items()}
        fields = {k: v[indices] for k, v in self.fields.items()}
        return Decisions(arrays, sizes, fields)

    def decision(self, i):
        """Decision `i` as {name: unpadded array}."""
        starts = {k: _starts(v) for k, v in self.sizes.items()}
        return {name: self.arrays[name][starts[key][i]:starts[key][i + 1]] for name, key in TENSORS}


def pad(decs, spec_dims, min_dims=None):
    """The contract's padded inputs (numpy, {name: array}) of `decs` (Decisions), as
    src/featurize/batch.cpp pads them. `spec_dims` has F, C, K, Q (a Spec or a dict); `min_dims`
    ({"N", "R", "A"}) pads at least that far."""
    dims = spec_dims if isinstance(spec_dims, dict) else spec_dims.dims()
    F, C, K, Q = dims["F"], dims["C"], dims["K"], dims["Q"]
    n, r, a = decs.sizes["n"], decs.sizes["r"], decs.sizes["a"]
    b = len(n)
    md = min_dims or {}
    N = max(padded(int(n.max())), md.get("N", 0))
    R = max(padded(int(r.max()) if b else 0), md.get("R", 0))
    A = max(padded(int(a.max())), md.get("A", 0))
    x = {
        "tok_type": np.zeros((b, N), np.int64), "tok_num": np.zeros((b, N, F), np.float32),
        "tok_cat": np.zeros((b, N, C), np.int64), "tok_card": np.full((b, N, K), -1, np.int64),
        "tok_ability": np.full((b, N, Q), -1, np.int64), "tok_mask": np.zeros((b, N), np.bool_),
        "rel": np.zeros((b, R, 3), np.int64), "rel_mask": np.zeros((b, R), np.bool_),
        "act_tok": np.zeros((b, A), np.int64), "act_mask": np.zeros((b, A), np.bool_),
    }
    for key, size in (("n", n), ("r", r), ("a", a)):
        row = np.repeat(np.arange(b), size)
        col = np.arange(int(size.sum())) - np.repeat(_starts(size)[:-1], size)
        for name, k in TENSORS:
            if k != key:
                continue
            v = decs.arrays[name]
            x[name][row, col] = (v != 0) if name in ("tok_mask", "act_mask") else v
        if key == "r":
            x["rel_mask"][row, col] = True
    return x


def to_torch(x, device="cpu", non_blocking=False):
    """The ten contract inputs of a padded batch as torch tensors on `device`, in contract order."""
    out = []
    for k in INPUTS:
        t = torch.from_numpy(x[k])
        out.append(t.to(device, non_blocking=non_blocking) if str(device) != "cpu" else t)
    return out


class Targets:
    """An RMTB targets file: per decision `seat`, `game`, `decision`, `sideboarding`, and the ragged
    `opp_hand`, `own_library`, `opp_library` (card ids)."""

    def __init__(self, path):
        meta, arr = rmtb.read(path)
        if meta.get("kind") != "targets":
            raise ValueError("%s is a %r file, not targets" % (path, meta.get("kind")))
        self.meta = meta
        self.arr = arr
        self.key_index = {(int(g), int(d)): i for i, (g, d) in enumerate(zip(arr["game"], arr["decision"]))}

    def __len__(self):
        return len(self.arr["seat"])

    def _ragged(self, name, i):
        off = self.arr[name + "_offsets"]
        return self.arr[name][off[i]:off[i + 1]]

    def batch(self, indices, n_cards, depth):
        """The targets of decisions `indices` as torch tensors: `opp_hand` [B, n_cards] counts,
        `own_library`, `opp_library` [B, depth] (-1 none)."""
        b = len(indices)
        hand = np.zeros((b, n_cards), np.float32)
        own = np.full((b, depth), -1, np.int64)
        opp = np.full((b, depth), -1, np.int64)
        for j, i in enumerate(indices):
            np.add.at(hand[j], self._ragged("opp_hand", i), 1.0)
            ol = self._ragged("own_library", i)[:depth]
            pl = self._ragged("opp_library", i)[:depth]
            own[j, :len(ol)] = ol
            opp[j, :len(pl)] = pl
        return {"opp_hand": torch.from_numpy(hand), "own_library": torch.from_numpy(own),
                "opp_library": torch.from_numpy(opp)}
