#!/usr/bin/env python3
"""Assert the generated C++ cost tables match train/card_costs.py bit-exactly.

Parses src/gen/card_costs_gen.h and the numpy matrices in train/card_costs.py and
checks np.float32 elementwise equality for both CARD_COST_MATRIX and
CARD_ABILITY_COST_MATRIX. A future C++ actor reads these tables to reproduce the
RL obs cost blocks, so the two sources MUST hold identical float32 values.

Stdlib + numpy only. Standalone: train/.venv/bin/python train/test_cost_tables.py
"""
import os
import re
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_HEADER = os.path.join(_REPO_ROOT, "src/gen/card_costs_gen.h")

# Header matrix name -> card_costs.py matrix attribute.
_TABLES = {
    "CARD_COST_MATRIX": "_CARD_COST_MATRIX",
    "CARD_ABILITY_COST_MATRIX": "_CARD_ABILITY_COST_MATRIX",
}


def parse_header_matrix(text, name):
    """Return an (N, 7) np.float32 array for the named table in the header text."""
    m = re.search(name + r"\[(\d+)\]\[(\d+)\]\s*=\s*\{(.*?)\n\};", text, re.DOTALL)
    if not m:
        raise RuntimeError(f"could not find matrix {name} in {_HEADER}")
    n_rows, n_cols = int(m.group(1)), int(m.group(2))
    rows = re.findall(r"\{([^{}]*)\}", m.group(3))
    if len(rows) != n_rows:
        raise RuntimeError(
            f"{name}: parsed {len(rows)} rows, declared {n_rows}")
    out = np.empty((n_rows, n_cols), dtype=np.float32)
    for i, row in enumerate(rows):
        vals = [np.float32(tok.strip().rstrip("f"))
                for tok in row.split(",") if tok.strip()]
        if len(vals) != n_cols:
            raise RuntimeError(
                f"{name} row {i}: {len(vals)} values, declared {n_cols}")
        out[i] = vals
    return out


def main():
    if _THIS_DIR not in sys.path:
        sys.path.insert(0, _THIS_DIR)
    import card_costs  # noqa: E402  (path set above)

    with open(_HEADER) as f:
        header_text = f.read()

    total_cells = 0
    for hdr_name, py_attr in _TABLES.items():
        py_mat = np.asarray(getattr(card_costs, py_attr), dtype=np.float32)
        hdr_mat = parse_header_matrix(header_text, hdr_name)
        if hdr_mat.shape != py_mat.shape:
            print(f"FAIL: {hdr_name} shape {hdr_mat.shape} != "
                  f"card_costs {py_attr} {py_mat.shape}")
            return 1
        # np.float32 exact equality (not allclose) — identity by construction.
        if not np.array_equal(hdr_mat, py_mat):
            bad = np.argwhere(hdr_mat != py_mat)
            print(f"FAIL: {hdr_name} differs from {py_attr} at "
                  f"{len(bad)} cell(s); first {tuple(bad[0])}: "
                  f"header={hdr_mat[tuple(bad[0])]} py={py_mat[tuple(bad[0])]}")
            return 1
        total_cells += py_mat.size
        print(f"  {hdr_name}: {py_mat.shape[0]} rows x {py_mat.shape[1]} "
              f"cols == {py_attr}")

    print(f"PASS: 2 tables, {total_cells} cells match np.float32-exactly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
