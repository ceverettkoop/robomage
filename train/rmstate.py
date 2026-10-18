"""RMSTATE decoder for Python, built from the SCHEMA frame (docs/state_format.md §9, App. A).

The engine's state format is a stream of records, ``RMSTATE <nbytes>\\n`` + payload, from
``--rmstate-dump FILE`` or from stdout under ``--machine-format rmstate``. Each payload is a frame:
a 24-byte header and fixed-width little-endian table blocks. Everything this module knows about
the tables (ids, names, column layouts, semantic tags, enum names) comes from the stream's SCHEMA
frame; nothing about them is copied here.

**Bootstrap.** The one fixed layout is what it takes to read a SCHEMA frame: the frame header and
table-block header, the 8-byte row alignment, format version 1, and the columns of the SCHEMA
frame's own tables 1-4 (``strings``, ``schema_table``, ``schema_column``, ``schema_enum``; ids,
names and column names/storages in ``_BOOT_TABLES``). Once read, the frame must describe those four
tables exactly that way, and its hash must match its table blocks; the storage names it lists map to
``struct`` codes by name (``_STORAGE_CODES``).

API (stdlib only):

- ``records(fp)``: the payloads of a binary file object (a file or a pipe);
- ``Schema.from_frame(payload)``: the schema a SCHEMA frame describes; ``schema.decode(payload)``
  → ``Frame`` (``frame.tables``: table name → list of row dicts, text resolved to ``str``; ``only=``
  limits the rows decoded to some tables);
- ``StreamDecoder``: follows a stream's SCHEMA frames, ``feed(payload)`` → ``Frame``;
- ``frame_json(frame, schema)``: the same JSON line the engine's ``--state-dump json`` writes
  (``src/state_format/json.h``), byte for byte once encoded as UTF-8 with ``surrogateescape``.

CLI: ``python3 train/rmstate.py dump FILE`` prints FILE's frames as JSON lines (``-`` = stdin).
"""
from __future__ import annotations

import math
import struct
import sys
from dataclasses import dataclass, field

# ── Bootstrap: the layout needed to read a SCHEMA frame ──────────────────────────────────────────

_FRAME_HEADER = struct.Struct("<4sHBBQII")  # magic, version, kind, viewer, hash, n_tables, reserved
_TABLE_HEADER = struct.Struct("<HHIII")     # id, reserved, row_width, row_count, reserved
_MAGIC = b"RMSF"
_ROW_ALIGN = 8
_BOOTSTRAP_VERSION = 1
_BOOT_TABLES = {
    1: ("strings", [("byte", "U8")]),
    2: ("schema_table", [("id", "U16"), ("name", "TEXT"), ("frame_kind", "U8"), ("row_width", "U32")]),
    3: ("schema_column", [("table", "U16"), ("col", "U16"), ("name", "TEXT"), ("storage", "U8"),
                          ("offset", "U16"), ("tag", "TEXT"), ("has_range", "U8"), ("min", "I64"),
                          ("max", "I64")]),
    4: ("schema_enum", [("enum", "TEXT"), ("value", "U16"), ("name", "TEXT")]),
}
# struct code per storage name; TEXT is a u32 offset into the frame's string pool.
_STORAGE_CODES = {"U8": "B", "U16": "H", "U32": "I", "U64": "Q", "I8": "b", "I16": "h",
                  "I32": "i", "I64": "q", "F32": "f", "TEXT": "I"}
_SIGNED = {"I8", "I16", "I32", "I64"}


class RMStateError(Exception):
    """A frame or stream the decoder refuses (malformed, or another version or schema)."""


@dataclass
class Column:
    name: str
    storage: str  # storage name: U8 … F32, TEXT
    offset: int
    tag: str = ""
    has_range: bool = False
    min: int = 0
    max: int = 0

    @property
    def width(self) -> int:
        return struct.calcsize("<" + _STORAGE_CODES[self.storage])


@dataclass
class Table:
    id: int
    name: str
    frame_kind: str
    row_width: int
    columns: list = field(default_factory=list)

    def __post_init__(self):
        self.struct = None

    def finish(self):
        """Checks the columns pack to the row width and builds the row struct."""
        off = 0
        for c in self.columns:
            if c.storage not in _STORAGE_CODES:
                raise RMStateError(f"column {c.name} of table {self.name}: unknown storage {c.storage}")
            if c.offset != off:
                raise RMStateError(f"column {c.name} of table {self.name} is not packed")
            off += c.width
        if off != self.row_width:
            raise RMStateError(f"columns of table {self.name} do not add up to its row width")
        self.struct = struct.Struct("<" + "".join(_STORAGE_CODES[c.storage] for c in self.columns))
        return self


@dataclass
class Frame:
    kind: str          # frame_kind name
    viewer: str        # viewer name
    kind_value: int
    viewer_value: int
    version: int
    schema_hash: int
    strings: bytes
    tables: dict       # table name → list of row dicts, ascending table id
    skipped: list      # ids of blocks the schema does not know


def _blocks(payload: bytes):
    """(header fields, [(id, width, rows, body)]) of a frame, checking the framing."""
    if len(payload) < _FRAME_HEADER.size:
        raise RMStateError("frame shorter than its header")
    magic, version, kind, viewer, h, n_tables, reserved = _FRAME_HEADER.unpack_from(payload, 0)
    if magic != _MAGIC:
        raise RMStateError("not an RMSF frame")
    if reserved:
        raise RMStateError("nonzero reserved header bytes")
    off = _FRAME_HEADER.size
    last = -1
    out = []
    for _ in range(n_tables):
        if len(payload) - off < _TABLE_HEADER.size:
            raise RMStateError("truncated table header")
        tid, r1, width, rows, r2 = _TABLE_HEADER.unpack_from(payload, off)
        if r1 or r2:
            raise RMStateError(f"nonzero reserved bytes in table {tid}")
        if tid <= last:
            raise RMStateError("table blocks out of id order")
        last = tid
        if rows == 0:
            raise RMStateError(f"empty table block {tid}")
        off += _TABLE_HEADER.size
        n = width * rows
        if n > len(payload) - off:
            raise RMStateError(f"truncated table {tid}")
        body = payload[off:off + n]
        off += n
        pad = (-off) % _ROW_ALIGN
        if off + pad > len(payload):
            raise RMStateError("truncated table padding")
        if payload[off:off + pad].strip(b"\0"):
            raise RMStateError("nonzero table padding")
        off += pad
        out.append((tid, width, rows, body))
    if off != len(payload):
        raise RMStateError("bytes after the last table block")
    return (version, kind, viewer, h), out


def _fnv1a64(data: bytes) -> int:
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


class _Pool:
    """A frame's string pool: offset → str (UTF-8, undecodable bytes kept via surrogateescape)."""

    def __init__(self, pool: bytes):
        if not pool or pool[0] != 0 or pool[-1] != 0:
            raise RMStateError("missing or malformed string pool")
        self.pool = pool
        self.cache = {0: ""}

    def text(self, off: int) -> str:
        s = self.cache.get(off)
        if s is not None:
            return s
        if off >= len(self.pool):
            raise RMStateError("text offset past the string pool")
        if self.pool[off - 1] != 0:
            raise RMStateError("text offset is not the start of a pooled string")
        end = self.pool.index(b"\0", off)
        s = self.pool[off:end].decode("utf-8", "surrogateescape")
        self.cache[off] = s
        return s


class Schema:
    """A schema as a SCHEMA frame describes it: tables by id and name, enum string tables."""

    def __init__(self, version: int, hash_: int, tables: dict, enums: dict):
        self.version = version
        self.hash = hash_
        self.tables = dict(sorted(tables.items()))  # id → Table
        self.by_name = {t.name: t for t in self.tables.values()}
        self.enums = enums                          # enum name → [value names]
        self.row_renderers = {}                     # table name → JSON row renderer (frame_json)

    # ── building from a SCHEMA frame ──────────────────────────────────────────────────────────

    @classmethod
    def _bootstrap(cls, hash_: int) -> "Schema":
        tables = {}
        for tid, (name, cols) in _BOOT_TABLES.items():
            t = Table(tid, name, "", 0)
            off = 0
            for cname, storage in cols:
                c = Column(cname, storage, off)
                t.columns.append(c)
                off += c.width
            t.row_width = off
            tables[tid] = t.finish()
        return cls(_BOOTSTRAP_VERSION, hash_, tables, {})

    @classmethod
    def from_frame(cls, payload: bytes) -> "Schema":
        """The schema a SCHEMA frame payload describes (the frame checked against its own hash
        and the bootstrap)."""
        (version, kind, viewer, h), _ = _blocks(payload)
        if version != _BOOTSTRAP_VERSION:
            raise RMStateError(f"format version {version}, expected {_BOOTSTRAP_VERSION}")
        if _fnv1a64(payload[_FRAME_HEADER.size:]) != h:
            raise RMStateError("SCHEMA frame hash does not match its table blocks")
        boot = cls._bootstrap(h)
        rows = boot._rows(payload, check_kind=False)
        if set(rows) - {"schema_table", "schema_column", "schema_enum"}:
            raise RMStateError("SCHEMA frame with tables other than 1-4")
        enums = {}
        last_enum = None
        for r in rows.get("schema_enum", []):
            if r["enum"] != last_enum:
                if r["enum"] in enums:
                    raise RMStateError(f"enum {r['enum']} is split")
                enums[r["enum"]] = []
                last_enum = r["enum"]
            values = enums[r["enum"]]
            if r["value"] != len(values):
                raise RMStateError(f"enum {r['enum']} values out of order")
            values.append(r["name"])
        storages = enums.get("storage", [])
        frame_kinds = enums.get("frame_kind", [])
        if kind >= len(frame_kinds) or frame_kinds[kind] != "SCHEMA":
            raise RMStateError("not a SCHEMA frame")
        tables = {}
        last = -1
        for r in rows.get("schema_table", []):
            if r["id"] <= last:
                raise RMStateError("schema_table rows out of id order")
            last = r["id"]
            if r["frame_kind"] >= len(frame_kinds):
                raise RMStateError(f"table {r['name']}: bad kind")
            tables[r["id"]] = Table(r["id"], r["name"], frame_kinds[r["frame_kind"]], r["row_width"])
        for r in rows.get("schema_column", []):
            t = tables.get(r["table"])
            if t is None:
                raise RMStateError("schema_column names an unlisted table")
            if r["col"] != len(t.columns):
                raise RMStateError(f"schema_column rows of table {t.name} out of order")
            if r["storage"] >= len(storages):
                raise RMStateError(f"column {r['name']}: bad storage")
            t.columns.append(Column(r["name"], storages[r["storage"]], r["offset"], r["tag"],
                                    bool(r["has_range"]), r["min"], r["max"]))
        for t in tables.values():
            t.finish()
        for tid, (name, cols) in _BOOT_TABLES.items():
            t = tables.get(tid)
            if (t is None or t.name != name
                    or [(c.name, c.storage) for c in t.columns] != cols):
                raise RMStateError(f"SCHEMA frame describes table {name} differently")
        return cls(version, h, tables, enums)

    # ── decoding ──────────────────────────────────────────────────────────────────────────────

    def _checked_blocks(self, payload: bytes, check_kind: bool = True):
        (version, kind, viewer, h), blocks = _blocks(payload)
        if version != self.version:
            raise RMStateError(f"format version {version}, expected {self.version}")
        if check_kind:
            if kind >= len(self.enums.get("frame_kind", [])):
                raise RMStateError(f"unknown frame kind {kind}")
            if viewer >= len(self.enums.get("viewer", [])):
                raise RMStateError(f"unknown viewer {viewer}")
        if h != self.hash:
            raise RMStateError("schema hash differs")
        pool = None
        known = []
        skipped = []
        frame_kind = self.enums["frame_kind"][kind] if check_kind else None
        for tid, width, rows, body in blocks:
            t = self.tables.get(tid)
            if t is None:
                skipped.append(tid)
                continue
            if width != t.row_width:
                raise RMStateError(f"table {t.name} has row width {width}, schema says {t.row_width}")
            if t.name == "strings" and tid == 1:
                pool = body
                continue
            if check_kind and t.frame_kind != frame_kind:
                raise RMStateError(f"table {t.name} in a frame of another kind")
            known.append((t, rows, body))
        if pool is None:
            raise RMStateError("missing or malformed string pool")
        return (version, kind, viewer, h), _Pool(pool), known, skipped

    def _rows(self, payload: bytes, check_kind: bool = True) -> dict:
        _, pool, known, _ = self._checked_blocks(payload, check_kind)
        return {t.name: self._table_rows(t, body, pool) for t, _rows, body in known}

    @staticmethod
    def _table_rows(t: Table, body: bytes, pool: _Pool) -> list:
        names = [c.name for c in t.columns]
        text_cols = [i for i, c in enumerate(t.columns) if c.storage == "TEXT"]
        out = []
        for vals in t.struct.iter_unpack(body):
            if text_cols:
                vals = list(vals)
                for i in text_cols:
                    vals[i] = pool.text(vals[i])
            out.append(dict(zip(names, vals)))
        return out

    def decode(self, payload: bytes, only=None) -> Frame:
        """A frame payload (any kind, SCHEMA included) decoded against this schema; with `only` (a
        set of table names) the other tables are checked but not decoded into rows."""
        (version, kind, viewer, h), pool, known, skipped = self._checked_blocks(payload)
        tables = {t.name: self._table_rows(t, body, pool) for t, _rows, body in known
                  if only is None or t.name in only}
        return Frame(self.enums["frame_kind"][kind], self.enums["viewer"][viewer], kind, viewer,
                     version, h, pool.pool, tables, skipped)

    def kind_of(self, payload: bytes) -> str:
        """The frame kind name a payload's header gives (no other check)."""
        if len(payload) < _FRAME_HEADER.size:
            raise RMStateError("frame shorter than its header")
        kind = payload[6]
        kinds = self.enums.get("frame_kind", [])
        return kinds[kind] if kind < len(kinds) else str(kind)


class StreamDecoder:
    """Decodes a stream's frames in order, building the schema from its SCHEMA frames (the first
    record must be one; each later one replaces the schema in use)."""

    def __init__(self):
        self.schema = None

    def feed(self, payload: bytes, only=None) -> Frame:
        """The payload decoded (`only`: as Schema.decode)."""
        if self.schema is None or self.schema.kind_of(payload) == "SCHEMA":
            self.schema = Schema.from_frame(payload)
            only = None
        return self.schema.decode(payload, only)


def records(fp):
    """Yields each record's payload from a binary file object (a file or a pipe)."""
    while True:
        head = fp.readline()
        if not head:
            return
        if not head.endswith(b"\n") or not head.startswith(b"RMSTATE ") or not head[8:-1].isdigit():
            raise RMStateError(f"bad record header {head[:40]!r}")
        n = int(head[8:-1])
        payload = fp.read(n)
        if len(payload) != n:
            raise RMStateError("truncated record")
        yield payload


# ── JSON (the engine's --state-dump json, src/state_format/json.h) ─────────────────────────────

_ESCAPES = {'"': '\\"', "\\": "\\\\", "\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def _json_string(s: str) -> str:
    out = ['"']
    for ch in s:
        e = _ESCAPES.get(ch)
        if e is not None:
            out.append(e)
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _json_renderer(schema: Schema, col: Column):
    """A function rendering one value of `col` (the rules of src/state_format/json.h)."""
    if col.storage == "TEXT":
        return _json_string
    if col.storage == "F32":
        return lambda v: "%.9g" % v if math.isfinite(v) else "null"
    mask = (1 << (8 * col.width)) - 1
    tag = col.tag
    if tag.startswith("enum:"):
        names = [_json_string(n) for n in schema.enums.get(tag[5:], [])]

        def enum_value(v):
            bits = v & mask
            if bits == mask:
                return "null"
            return names[bits] if bits < len(names) else str(bits)
        return enum_value
    if tag.startswith("mask:"):
        names = [_json_string(n) for n in schema.enums.get(tag[5:], [])]

        def mask_value(v):
            bits = v & mask
            return "[" + ",".join((names[b] if b < len(names) else str(b))
                                  for b in range(bits.bit_length()) if (bits >> b) & 1) + "]"
        return mask_value
    if tag == "bool":
        return lambda v: ("true" if v & mask else "false") if (v & mask) <= 1 else str(v)
    if tag in ("card", "index") or tag.startswith("row:"):
        return lambda v: "null" if (v & mask) == mask else str(v)
    return str


def _row_renderer(schema: Schema, table: Table):
    """A function rendering one row dict of `table` as a JSON object."""
    parts = [(_json_string(c.name) + ":", c.name, _json_renderer(schema, c)) for c in table.columns]

    def row_json(row):
        return "{" + ",".join(key + render(row[name]) for key, name, render in parts) + "}"
    return row_json


def frame_json(frame: Frame, schema: Schema) -> str:
    """The frame as the engine's --state-dump json renders it (one line, no newline)."""
    renderers = schema.row_renderers
    out = ['{"frame":', _json_string(frame.kind), ',"viewer":', _json_string(frame.viewer), ',"version":',
           str(frame.version), ',"schema_hash":"%016x"' % frame.schema_hash, ',"tables":{']
    tables = []
    for name, rows in frame.tables.items():
        render = renderers.get(name)
        if render is None:
            render = renderers[name] = _row_renderer(schema, schema.by_name[name])
        tables.append(_json_string(name) + ":[" + ",".join(render(row) for row in rows) + "]")
    out.append(",".join(tables))
    out.append("}")
    if frame.skipped:
        out.append(',"skipped":[' + ",".join(str(i) for i in frame.skipped) + "]")
    out.append("}")
    return "".join(out)


def json_lines(fp):
    """Yields the JSON line (str, no newline) of every frame of a stream."""
    dec = StreamDecoder()
    for payload in records(fp):
        frame = dec.feed(payload)
        yield frame_json(frame, dec.schema)


# ── CLI ───────────────────────────────────────────────────────────────────────────────────────

def build_parser():
    """The rmstate.py parser."""
    import argparse
    ap = argparse.ArgumentParser(prog="rmstate", description="Decode RMSTATE streams (the engine's "
                                 "--rmstate-dump files or --machine-format rmstate output).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    dump = sub.add_parser("dump", help="Print an RMSTATE stream (an engine --rmstate-dump file, or a "
                          "--machine-format rmstate capture) as JSON lines, as the engine's "
                          "--state-dump json renders them")
    dump.add_argument("file", metavar="FILE", help="RMSTATE stream to decode (- = stdin)")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "dump":
        fp = sys.stdin.buffer if args.file == "-" else open(args.file, "rb")
        try:
            out = sys.stdout.buffer
            for line in json_lines(fp):
                out.write(line.encode("utf-8", "surrogateescape") + b"\n")
        except RMStateError as exc:
            print(f"rmstate: {exc}", file=sys.stderr)
            return 1
        finally:
            if fp is not sys.stdin.buffer:
                fp.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
