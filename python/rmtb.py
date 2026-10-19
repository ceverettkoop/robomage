"""RMTB files (docs/featurizer.md §4.4) read with the standard library only.

An RMTB file holds one or more named arrays: the magic ``RMTB``, a little-endian u32 version (1),
a little-endian u32 meta length, the meta (UTF-8 JSON padded with spaces to a multiple of 8 bytes;
file-level keys plus ``arrays``: ``{"name","dtype","shape","offset"[,"columns"]}`` in file order)
and the data section (from byte 12 + meta length), each array's row-major little-endian values at
its 8-aligned ``offset``. Dtypes: f32, f16 (IEEE binary16), i64, i32, i16, u16, u8. The optional
``attrs`` meta key is an object of named strings and integers (a file kind's header fields: the
trajectory episode's, ``python/rmtraj.py``)::

    meta, arrays = rmtb.read("bin/resources/embeddings/oracle_text.rmtb")
    vec = arrays["vectors"]          # Array: .name .dtype .shape .values (flat list)
    vec.rows()[17][:4]               # a rank-2 array as a list of rows

``read_bytes`` decodes bytes already in memory. Malformed input raises ``ValueError``. The C++
reader and writer are ``src/featurize/rmtb.{h,cpp}``.
"""
from __future__ import annotations

import json
import struct

MAGIC = b"RMTB"
VERSION = 1
_HEADER = struct.Struct("<4sII")
# dtype → (struct format character, width)
DTYPES = {"f32": ("f", 4), "f16": ("e", 2), "i64": ("q", 8), "i32": ("i", 4), "i16": ("h", 2), "u16": ("H", 2), "u8": ("B", 1)}


class Array:
    """One named array: its dtype, shape and values (flat, row-major)."""

    def __init__(self, name, dtype, shape, values, columns=None):
        self.name, self.dtype, self.shape, self.values, self.columns = name, dtype, shape, values, columns

    def rows(self):
        """A rank-2 array as a list of rows."""
        if len(self.shape) != 2:
            raise ValueError("%s has shape %r, not rank 2" % (self.name, self.shape))
        cols = self.shape[1]
        return [self.values[r * cols:(r + 1) * cols] for r in range(self.shape[0])]


def _count(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def read_bytes(data: bytes):
    """(meta dict, {name: Array}) of an RMTB file's bytes."""
    if len(data) < _HEADER.size:
        raise ValueError("not an RMTB file (too short)")
    magic, version, meta_len = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not an RMTB file (no RMTB magic)")
    if version != VERSION:
        raise ValueError("RMTB version %d, expected %d" % (version, VERSION))
    if meta_len % 8 or _HEADER.size + meta_len > len(data):
        raise ValueError("meta length %d is not a multiple of 8 within the file" % meta_len)
    meta = json.loads(data[_HEADER.size:_HEADER.size + meta_len].decode("utf-8"))
    body = memoryview(data)[_HEADER.size + meta_len:]
    arrays = {}
    end = 0
    for a in meta.get("arrays", ()):
        name, dtype, shape, offset = a["name"], a["dtype"], a["shape"], a["offset"]
        if dtype not in DTYPES:
            raise ValueError("array %s: dtype %r is not one of %s" % (name, dtype, ", ".join(DTYPES)))
        if name in arrays:
            raise ValueError("two arrays are named %s" % name)
        code, width = DTYPES[dtype]
        n = _count(shape)
        if offset != (end + 7) // 8 * 8 or offset + n * width > len(body):
            raise ValueError("array %s: offset %d is not the 8-aligned end of the previous array within the data"
                             % (name, offset))
        values = list(struct.unpack_from("<%d%s" % (n, code), body, offset))
        arrays[name] = Array(name, dtype, list(shape), values, a.get("columns"))
        end = offset + n * width
    if end != len(body):
        raise ValueError("the data section is %d bytes, the arrays end at %d" % (len(body), end))
    return meta, arrays


def read(path: str):
    """(meta dict, {name: Array}) of the RMTB file at ``path``."""
    with open(path, "rb") as fp:
        return read_bytes(fp.read())
