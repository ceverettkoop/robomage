#!/usr/bin/env python3
"""Regression for train/tree_cache.py: synthetic per-world MCTS trees survive a
save_tree/load_tree round trip node-for-node, including merge partitions
(rep/sel_mask) and the argmax-visit PV descent. Engine- and torch-free."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcts import _Node  # noqa: E402
from tree_cache import (TREE_FORMAT_VERSION, load_tree, save_tree,  # noqa: E402
                        tree_node_count)


def _random_rep(rng, nc: int):
    """A duplicate-group partition: rep[i] <= i, rep[rep[i]] == rep[i]."""
    rep = np.arange(nc, dtype=np.int16)
    for i in range(1, nc):
        if rng.random() < 0.4:
            rep[i] = rep[int(rng.integers(0, i))]
    return rep


def _build_node(rng, depth: int, self_is_a: bool) -> _Node:
    nc = int(rng.integers(1, 7))
    priors = rng.random(nc)
    priors /= priors.sum()
    rep = _random_rep(rng, nc) if (nc > 1 and rng.random() < 0.5) else None
    node = _Node(nc, priors, self_is_a, rep=rep)
    # Visits/values only live at representative indices (as in a real search).
    reps = set(range(nc)) if rep is None else {int(r) for r in rep}
    for a in reps:
        if rng.random() < 0.8:
            node.N[a] = int(rng.integers(0, 50))
            node.W[a] = float(rng.normal()) * node.N[a]
    if depth > 0:
        for a in reps:
            if node.N[a] > 0 and rng.random() < 0.7:
                node.children[a] = _build_node(rng, depth - 1, not self_is_a)
    return node


def _build_forest(seed: int) -> list:
    rng = np.random.default_rng(seed)
    n_worlds = int(rng.integers(2, 5))
    return [_build_node(rng, int(rng.integers(1, 5)), bool(rng.integers(0, 2)))
            for _ in range(n_worlds)]


def _assert_node_equal(a: _Node, b: _Node, where: str) -> None:
    assert a.num_choices == b.num_choices, where
    assert a.self_is_a == b.self_is_a, where
    assert a.pick_meta is None and b.pick_meta is None, where
    assert np.array_equal(a.P, b.P), where
    assert np.array_equal(a.N, b.N), where
    assert np.array_equal(a.W, b.W), where
    assert a.P.dtype == b.P.dtype == np.float64, where
    assert a.N.dtype == b.N.dtype == np.int64, where
    assert a.W.dtype == b.W.dtype == np.float64, where
    assert (a.rep is None) == (b.rep is None), where
    assert (a.sel_mask is None) == (b.sel_mask is None), where
    if a.rep is not None:
        assert np.array_equal(a.rep, b.rep), where
        # array_equal treats -inf == -inf as True, but be explicit about it.
        assert np.array_equal(np.isneginf(a.sel_mask), np.isneginf(b.sel_mask)), where
        assert np.array_equal(a.sel_mask[np.isfinite(a.sel_mask)],
                              b.sel_mask[np.isfinite(b.sel_mask)]), where
    assert sorted(a.children) == sorted(b.children), where
    for k in a.children:
        _assert_node_equal(a.children[k], b.children[k], f"{where}/{k}")


def _pv(root: _Node, action: int, max_len: int = 24) -> list:
    """Argmax-visit descent mirroring IncrementalSearch.pv (canonicalized via rep)."""
    out = []
    node = root
    a = int(action)
    for _ in range(max_len):
        if node.rep is not None and 0 <= a < node.num_choices:
            a = int(node.rep[a])
        n_vis = int(node.N[a])
        if n_vis <= 0:
            break
        out.append((a, n_vis, float(node.W[a]) / n_vis, node.self_is_a))
        child = node.children.get(a)
        if child is None or int(child.N.sum()) <= 0:
            break
        node = child
        a = int(np.argmax(node.N))
    return out


def _check_round_trip(seed: int, tmp: str) -> None:
    roots = _build_forest(seed)
    meta = {"seed": seed, "note": "synthetic", "worlds": len(roots),
            "nested": {"x": [1, 2.5, "s"]}}
    path = os.path.join(tmp, f"tree_{seed}.npz")
    save_tree(path, roots, meta)
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp"), "tmp file left behind"

    loaded, meta_back = load_tree(path)
    assert len(loaded) == len(roots)
    assert tree_node_count(loaded) == tree_node_count(roots)
    for k, v in meta.items():
        assert meta_back[k] == v, (k, meta_back[k], v)
    assert meta_back["format_version"] == TREE_FORMAT_VERSION
    for w, (a, b) in enumerate(zip(roots, loaded)):
        _assert_node_equal(a, b, f"seed{seed}/world{w}")
        for act in range(a.num_choices):
            assert _pv(a, act) == _pv(b, act), (seed, w, act)


def _check_tamper(tmp: str) -> None:
    roots = _build_forest(99)
    path = os.path.join(tmp, "tamper.npz")
    save_tree(path, roots, {})
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    import json
    meta = json.loads(str(arrays["meta_json"]))
    meta["format_version"] = TREE_FORMAT_VERSION + 1
    arrays["meta_json"] = np.array(json.dumps(meta))
    with open(path, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    try:
        load_tree(path)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered format_version did not raise ValueError")

    # A garbage file is a ValueError too, never a bare OSError/KeyError.
    bad = os.path.join(tmp, "garbage.npz")
    with open(bad, "wb") as fh:
        fh.write(b"not an npz")
    try:
        load_tree(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("garbage file did not raise ValueError")


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="tree_cache_test_")
    try:
        total_nodes = 0
        for seed in range(1, 9):
            _check_round_trip(seed, tmp)
            total_nodes += tree_node_count(_build_forest(seed))
        _check_tamper(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_tree_cache: OK ({total_nodes} nodes round-tripped over 8 forests)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
