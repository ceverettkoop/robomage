"""On-disk cache for a determinized MCTS search tree.

Serializes a list of per-world ``mcts._Node`` roots (one tree per world) to a
single compressed ``.npz`` so a rebuilt tree can sit next to a recording and be
reloaded without re-searching. numpy only — no Qt, no torch; ``mcts`` is
imported lazily inside ``load_tree`` for the ``_Node`` class.

Layout (all nodes of all worlds flattened in pre-order DFS, world by world):

  per-node arrays, length n_nodes
    world        int16   world index of the tree the node belongs to
    parent       int32   flat index of the parent node, -1 for a root
    edge_action  int16   env action index leading from parent to this node, -1 for a root
    num_choices  int16   node.num_choices
    self_is_a    uint8   node.self_is_a
    offset       int64   start of this node's rows in the per-action arrays
    has_rep      uint8   1 if node.rep is not None

  per-action arrays, length sum(num_choices)
    P            float64 node.P — ALREADY folded onto representatives
    N            int64   node.N
    W            float64 node.W
    rep          int16   node.rep, or -1 rows for nodes without a partition
                         (one row per action for every node, so it shares
                         ``offset``; has_rep decides whether the row is used)

  meta_json      0-d str  json.dumps(meta | {"format_version": TREE_FORMAT_VERSION})

Because ``P`` is stored post-fold, loading constructs each node with
``rep=None`` (so ``set_rep`` does not fold a second time) and then installs
``rep``/``sel_mask`` directly. ``pick_meta`` is not serialized (None for
in-game searches); a tree carrying it is refused with ``ValueError``.
"""
from __future__ import annotations

import json
import os

import numpy as np

TREE_FORMAT_VERSION = 1

_NODE_KEYS = ("world", "parent", "edge_action", "num_choices", "self_is_a",
              "offset", "has_rep")
_ACTION_KEYS = ("P", "N", "W", "rep")
_REQUIRED_KEYS = _NODE_KEYS + _ACTION_KEYS + ("meta_json",)


def tree_node_count(roots) -> int:
    """Total number of nodes across every world's tree."""
    total = 0
    for root in roots:
        stack = [root]
        while stack:
            node = stack.pop()
            total += 1
            stack.extend(node.children.values())
    return total


def _sel_mask_from_rep(rep, num_choices: int):
    if rep is None:
        return None
    mask = np.zeros(num_choices, dtype=np.float64)
    for i in range(num_choices):
        if int(rep[i]) != i:
            mask[i] = -np.inf
    return mask


def _flatten(roots) -> dict:
    world, parent, edge_action, num_choices, self_is_a, offset, has_rep = (
        [], [], [], [], [], [], [])
    P, N, W, rep = [], [], [], []
    cursor = 0
    for w, root in enumerate(roots):
        # (node, parent flat index, edge action); pre-order DFS, children in
        # ascending action order so the layout is deterministic.
        stack = [(root, -1, -1)]
        while stack:
            node, par, act = stack.pop()
            if node.pick_meta is not None:
                raise ValueError("tree_cache: pick_meta is not serializable")
            idx = len(world)
            nc = int(node.num_choices)
            world.append(w)
            parent.append(par)
            edge_action.append(act)
            num_choices.append(nc)
            self_is_a.append(1 if node.self_is_a else 0)
            offset.append(cursor)
            has_rep.append(0 if node.rep is None else 1)
            P.append(np.asarray(node.P, dtype=np.float64).reshape(nc))
            N.append(np.asarray(node.N, dtype=np.int64).reshape(nc))
            W.append(np.asarray(node.W, dtype=np.float64).reshape(nc))
            if node.rep is None:
                rep.append(np.full(nc, -1, dtype=np.int16))
            else:
                rep.append(np.asarray(node.rep, dtype=np.int16).reshape(nc))
            cursor += nc
            for a in sorted(node.children, reverse=True):
                stack.append((node.children[a], idx, int(a)))

    def cat(parts, dtype):
        if not parts:
            return np.zeros(0, dtype=dtype)
        return np.concatenate(parts).astype(dtype, copy=False)

    return {
        "world": np.asarray(world, dtype=np.int16),
        "parent": np.asarray(parent, dtype=np.int32),
        "edge_action": np.asarray(edge_action, dtype=np.int16),
        "num_choices": np.asarray(num_choices, dtype=np.int16),
        "self_is_a": np.asarray(self_is_a, dtype=np.uint8),
        "offset": np.asarray(offset, dtype=np.int64),
        "has_rep": np.asarray(has_rep, dtype=np.uint8),
        "P": cat(P, np.float64),
        "N": cat(N, np.int64),
        "W": cat(W, np.float64),
        "rep": cat(rep, np.int16),
    }


def save_tree(path, roots, meta: dict) -> None:
    """Write ``roots`` (one ``_Node`` per world) plus ``meta`` to ``path``
    atomically: the npz is written to ``path + ".tmp"`` then ``os.replace``d."""
    arrays = _flatten(roots)
    meta_out = dict(meta or {})
    meta_out["format_version"] = TREE_FORMAT_VERSION
    arrays["meta_json"] = np.array(json.dumps(meta_out))
    tmp = str(path) + ".tmp"
    try:
        # np.savez_compressed appends ".npz" unless the name already ends
        # with it, so hand it an open file object to keep the exact name.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def load_tree(path) -> tuple[list, dict]:
    """Reload a tree saved by ``save_tree``. Returns ``(roots, meta)`` with
    roots in world order. Raises ``ValueError`` on a malformed file or a
    ``format_version`` other than ``TREE_FORMAT_VERSION``."""
    from mcts import _Node

    try:
        with np.load(str(path), allow_pickle=False) as data:
            missing = [k for k in _REQUIRED_KEYS if k not in data.files]
            if missing:
                raise ValueError(f"tree_cache: {path} missing keys {missing}")
            arrays = {k: data[k] for k in _REQUIRED_KEYS}
    except (OSError, ValueError) as exc:
        raise ValueError(f"tree_cache: cannot read {path}: {exc}") from None

    try:
        meta = json.loads(str(arrays["meta_json"]))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"tree_cache: bad meta_json in {path}: {exc}") from None
    if not isinstance(meta, dict):
        raise ValueError(f"tree_cache: meta_json in {path} is not an object")
    version = meta.get("format_version")
    if version != TREE_FORMAT_VERSION:
        raise ValueError(
            f"tree_cache: {path} has format_version {version!r}, "
            f"expected {TREE_FORMAT_VERSION}")

    world = arrays["world"]
    parent = arrays["parent"]
    edge_action = arrays["edge_action"]
    num_choices = arrays["num_choices"]
    self_is_a = arrays["self_is_a"]
    offset = arrays["offset"]
    has_rep = arrays["has_rep"]
    P, N, W, rep = arrays["P"], arrays["N"], arrays["W"], arrays["rep"]
    n_nodes = len(world)
    for arr in (parent, edge_action, num_choices, self_is_a, offset, has_rep):
        if len(arr) != n_nodes:
            raise ValueError(f"tree_cache: per-node array length mismatch in {path}")
    n_actions = int(num_choices.astype(np.int64).sum())
    for arr in (P, N, W, rep):
        if len(arr) != n_actions:
            raise ValueError(f"tree_cache: per-action array length mismatch in {path}")

    nodes: list = []
    roots: dict[int, _Node] = {}
    for i in range(n_nodes):
        nc = int(num_choices[i])
        lo = int(offset[i])
        hi = lo + nc
        node = _Node(nc, np.array(P[lo:hi], dtype=np.float64),
                     bool(self_is_a[i]), pick_meta=None, rep=None)
        node.N = np.array(N[lo:hi], dtype=np.int64)
        node.W = np.array(W[lo:hi], dtype=np.float64)
        if has_rep[i]:
            node.rep = np.array(rep[lo:hi], dtype=np.int16)
            node.sel_mask = _sel_mask_from_rep(node.rep, nc)
        nodes.append(node)
        par = int(parent[i])
        if par < 0:
            w = int(world[i])
            if w in roots:
                raise ValueError(f"tree_cache: world {w} has two roots in {path}")
            roots[w] = node
        else:
            if par >= i:
                raise ValueError(f"tree_cache: parent {par} not before node {i} in {path}")
            nodes[par].children[int(edge_action[i])] = node

    n_worlds = len(roots)
    if sorted(roots) != list(range(n_worlds)):
        raise ValueError(f"tree_cache: non-contiguous world indices in {path}")
    return [roots[w] for w in range(n_worlds)], meta
