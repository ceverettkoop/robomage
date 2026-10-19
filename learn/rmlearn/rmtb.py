"""RMTB files (docs/featurizer.md §4.4) read and written with numpy.

The numpy counterpart of ``python/rmtb.py`` (which stays stdlib-only): the magic ``RMTB``, a
little-endian u32 version (1), a u32 meta length, the meta (UTF-8 JSON padded with spaces to a
multiple of 8 bytes; file-level keys, then ``arrays``) and the data section, each array's
row-major little-endian values at its 8-aligned ``offset``::

    meta, arrays = rmtb.read("out/card.rmtb")
    arrays["features"]             # np.ndarray [1025, Fc] float32 (a read-only view of the file)
    rmtb.columns(meta, "features") # the last dimension's names, or None

``write`` lays out arrays in the same canonical form the C++ writer produces (meta keys in the
documented order, the arrays packed at 8-byte alignment). Malformed input raises ``ValueError``.
"""

import json
import struct

import numpy as np

MAGIC = b"RMTB"
VERSION = 1
_HEADER = struct.Struct("<4sII")
DTYPES = {"f32": np.dtype("<f4"), "f16": np.dtype("<f2"), "i64": np.dtype("<i8"), "i32": np.dtype("<i4"),
          "u16": np.dtype("<u2"), "u8": np.dtype("u1")}
# File-level meta keys in the order the format fixes (featurizer.md §4.4); `arrays` comes last.
META_ORDER = ["kind", "vocab_hash", "spec_hash", "model", "pooling", "norm_version", "rows_meta"]


def _align8(x):
    return (x + 7) & ~7


def read_bytes(data):
    """(meta dict, {name: np.ndarray}) of an RMTB file's bytes; the arrays are views of `data`."""
    if len(data) < _HEADER.size:
        raise ValueError("not an RMTB file (too short)")
    magic, version, meta_len = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not an RMTB file (no RMTB magic)")
    if version != VERSION:
        raise ValueError("RMTB version %d, expected %d" % (version, VERSION))
    if meta_len % 8 or _HEADER.size + meta_len > len(data):
        raise ValueError("meta length %d is not a multiple of 8 within the file" % meta_len)
    meta = json.loads(bytes(data[_HEADER.size:_HEADER.size + meta_len]).decode("utf-8"))
    base = _HEADER.size + meta_len
    body_len = len(data) - base
    arrays = {}
    end = 0
    for a in meta.get("arrays", ()):
        name, dtype, shape, offset = a["name"], a["dtype"], a["shape"], a["offset"]
        if dtype not in DTYPES:
            raise ValueError("array %s: dtype %r is not one of %s" % (name, dtype, ", ".join(DTYPES)))
        if name in arrays:
            raise ValueError("two arrays are named %s" % name)
        dt = DTYPES[dtype]
        count = int(np.prod(shape)) if shape else 1
        if offset != _align8(end) or offset + count * dt.itemsize > body_len:
            raise ValueError("array %s: offset %d is not the 8-aligned end of the previous array within the data"
                             % (name, offset))
        arrays[name] = np.frombuffer(data, dt, count, base + offset).reshape(shape)
        end = offset + count * dt.itemsize
    if end != body_len:
        raise ValueError("the data section is %d bytes, the arrays end at %d" % (body_len, end))
    return meta, arrays


def read(path):
    """(meta dict, {name: np.ndarray}) of the RMTB file at `path`."""
    with open(path, "rb") as fp:
        return read_bytes(fp.read())


def columns(meta, name):
    """The `columns` list of array `name` in `meta`, or None."""
    for a in meta.get("arrays", ()):
        if a["name"] == name:
            return a.get("columns")
    raise KeyError(name)


def _dtype_name(arr):
    for k, v in DTYPES.items():
        if arr.dtype == v or (arr.dtype.kind == v.kind and arr.dtype.itemsize == v.itemsize):
            return k
    if arr.dtype == np.bool_:
        return "u8"
    raise ValueError("dtype %s has no RMTB name" % arr.dtype)


def write_bytes(meta, arrays, columns_of=None):
    """An RMTB file's bytes: `meta` (file-level keys) and `arrays` ({name: ndarray}, in order);
    `columns_of` maps an array name to its last dimension's names."""
    columns_of = columns_of or {}
    descs = []
    blobs = []
    off = 0
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        dname = _dtype_name(arr)
        raw = arr.astype(DTYPES[dname], copy=False).tobytes()
        off = _align8(off)
        d = {"name": name, "dtype": dname, "shape": list(arr.shape), "offset": off}
        if name in columns_of:
            d["columns"] = list(columns_of[name])
        descs.append(d)
        blobs.append((off, raw))
        off += len(raw)
    m = {k: meta[k] for k in META_ORDER if k in meta}
    m.update({k: v for k, v in meta.items() if k not in m and k != "arrays"})
    m["arrays"] = descs
    text = json.dumps(m, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    text += b" " * (_align8(len(text)) - len(text))
    data = bytearray(off)
    for o, raw in blobs:
        data[o:o + len(raw)] = raw
    return _HEADER.pack(MAGIC, VERSION, len(text)) + text + bytes(data)


def write(path, meta, arrays, columns_of=None):
    """Writes an RMTB file (see write_bytes)."""
    with open(path, "wb") as fp:
        fp.write(write_bytes(meta, arrays, columns_of))
