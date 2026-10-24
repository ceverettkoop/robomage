"""The FEATURE_SPEC document and the card tables (docs/featurizer.md §4, §6) as the learner loads them.

    spec = load_spec("out/spec.json")         # Spec: .doc (the parsed JSON), .hash, dims, sizes
    tables = load_tables("out", text_table)   # Tables: card / text / ability / part arrays + hashes

``load_tables`` reads ``rmfeat --export-tables DIR`` (``card.rmtb``, ``ability.rmtb``,
``part.rmtb``, ``spec.json``, ``manifest.json``, ``registry.json``; ``lineage.json`` when ``rmtrain
retable`` wrote one) and the text table (``oracle_text.rmtb``, by default the one in the same
directory, else the committed ``bin/resources/embeddings/oracle_text.rmtb``) and checks they belong
together: every table carries the spec's hash, and every table the same vocab hash.

**Compatibility** (docs/featurizer.md §6.2; ``src/featurize/compat.h`` is the same rule in C++).
``Tables.hashes`` is the tables' *model manifest*, which a model trained on them carries (checkpoint
``hashes``, export metadata): ``spec_hash`` and ``vocab_hash`` (exact identity), ``text_model``,
``layout_hash``, ``tables_digest`` (every card's card, ability and part rows, src/featurize/drift.h),
``vocab`` (the vocab manifest: ``[id, name]`` per card-table row, the CARDS frame's names) and ``registry`` (the layout registry's text, the manifest of positions). ``check_compatible(
model, now, exact)`` is None or a ``(code, message)`` refusal: the same ``layout_hash`` and
``text_model``, every ``[id, name]`` of the model's vocab in the current one unchanged (new ids pass;
a removed or renamed id is refused naming it), no registry entry of the model's moved (when both
carry one); ``exact`` (an export, whose card rows are baked in) also the same ``spec_hash``, ``vocab_hash`` and
``tables_digest`` and no current id the model lacks. A manifest without ``layout_hash`` (a model made
before manifests) passes only on its own tables' ``spec_hash``, ``vocab_hash`` and ``text_model``.
"""

import json
import os

import numpy as np

from . import rmtb

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
COMMITTED_TEXT_TABLE = os.path.join(REPO_ROOT, "bin", "resources", "embeddings", "oracle_text.rmtb")


def fnv1a64(data):
    """FNV-1a 64 of `data` (bytes), as the featurizer's hashes are computed."""
    h = 0xcbf29ce484222325
    for b in data:
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


class Spec:
    """The FEATURE_SPEC document: its JSON, its hash and the sizes a model is built from.

    Every size a model's parameters have is a capacity of the layout registry
    (``src/featurize/registry.json``, docs/featurizer.md §6.1), so cards, enum values, slots and types
    appended within it change no shape: ``n_types`` (token type ids), ``n_rel`` (relation type ids),
    ``categorical_size`` (categorical ids), ``G`` (every type's numeric block side by side) and the
    tables' widths. ``token_types`` lists the types the spec uses, each with its registry ``id``."""

    def __init__(self, doc_bytes):
        doc_bytes = doc_bytes.rstrip(b"\n")
        self.doc = json.loads(doc_bytes)
        self.hash = "%016x" % fnv1a64(doc_bytes)
        d = self.doc
        self.vocab_hash = d["vocab_hash"]
        self.layout_hash = d["layout_hash"]
        self.F, self.C, self.K, self.Q, self.G = d["F"], d["C"], d["K"], d["Q"], d["G"]
        self.categorical_size = d["categorical_size"]
        self.token_types = d["token_types"]
        self.n_types = d["type_capacity"]
        self.n_rel = d["relation_capacity"]
        self.type_ids = {t["name"]: t["id"] for t in self.token_types}
        self.type_by_id = {t["id"]: t for t in self.token_types}
        self.menu_type = self.type_ids[d["menu"]]
        self.library_depth = d["library_depth"]
        self.card_rows = d["card_tables"]["card"]["rows"]
        # The recent-event history (docs/featurizer.md §3.7; absent unless the spec was built with it):
        # the event tokens' type and the relation tying a card token to its event's token.
        self.history = d.get("history")
        self.hist_event_type = self.hist_card_rel = None
        if self.history:
            self.hist_event_type = self.type_ids[self.history["event_type"]]
            for r in d["relation_types"]:
                if r["table"] == self.history["card_type"] and r["column"] == "event":
                    self.hist_card_rel = r["id"]

    def dims(self):
        """The handshake's `dims` (docs/inference_protocol.md §3)."""
        return {"F": self.F, "C": self.C, "K": self.K, "Q": self.Q}

    def enum_values(self, name):
        """The registry's values of enum `name` (a categorical slot's value i is the i-th), or None."""
        return self.doc.get("enums", {}).get(name)

    def numeric_layout(self):
        """Per token type id (``n_types`` of them) its numeric block's first column in G and its
        capacity (a token's slots past it are always 0; an unused id has capacity 0)."""
        offsets, counts = [0] * self.n_types, [0] * self.n_types
        for t in self.token_types:
            offsets[t["id"]], counts[t["id"]] = t["num_at"], t["num_capacity"]
        return offsets, counts


def load_spec(path):
    with open(path, "rb") as fp:
        return Spec(fp.read())


class Tables:
    """The whole-vocab tables (float32 numpy arrays) and the manifest a model carries."""

    def __init__(self, spec, card, text, ability, part, part_ability, hashes, lineage=()):
        self.spec = spec
        self.card = card                  # [N_CARD_TYPES + 1, Fc]
        self.text = text                  # [N_CARD_TYPES + 1, 384]
        self.ability = ability            # [Nab, Fa]
        self.part = part                  # [Npart, Fp]
        self.part_ability = part_ability  # [Npart] i64, the ability row each part belongs to
        self.hashes = hashes              # the model manifest (module docstring)
        self.lineage = list(lineage)      # the run's earlier tables these extend (lineage.json), oldest first

    def accepted_spec_hashes(self):
        """The spec hashes whose episodes a model on these tables trains on: these tables' and their
        lineage's (``rmtrain retable`` checked each earlier manifest is a subset of this one; the layout
        is the same, and the stored card and ability ids are stable)."""
        return frozenset([self.spec.hash] + [x["spec_hash"] for x in self.lineage])


def load_tables(tables_dir, text_table=None, legacy=False):
    """The exported tables of `tables_dir` and the text table; ValueError when they disagree. A
    directory exported before model manifests (no manifest.json) is refused unless `legacy`, which
    gives it a manifest of its hashes alone (what such a directory's checkpoints carry: the S5
    migration reads old runs' tables this way)."""
    spec = load_spec(os.path.join(tables_dir, "spec.json"))
    if text_table is None:
        local = os.path.join(tables_dir, "oracle_text.rmtb")
        text_table = local if os.path.exists(local) else COMMITTED_TEXT_TABLE
    card_meta, card = rmtb.read(os.path.join(tables_dir, "card.rmtb"))
    ab_meta, ability = rmtb.read(os.path.join(tables_dir, "ability.rmtb"))
    part_meta, part = rmtb.read(os.path.join(tables_dir, "part.rmtb"))
    text_meta, text = rmtb.read(text_table)
    for name, m in (("card", card_meta), ("ability", ab_meta), ("part", part_meta)):
        if m.get("spec_hash") != spec.hash:
            raise ValueError("%s/%s.rmtb carries spec hash %s, spec.json hashes to %s"
                             % (tables_dir, name, m.get("spec_hash"), spec.hash))
    for name, m in (("card", card_meta), ("ability", ab_meta), ("part", part_meta), ("text", text_meta)):
        if m.get("vocab_hash") != spec.vocab_hash:
            raise ValueError("the %s table is of vocab %s, the spec of %s" % (name, m.get("vocab_hash"),
                                                                              spec.vocab_hash))
    if legacy and not os.path.exists(os.path.join(tables_dir, "manifest.json")):
        hashes = {"spec_hash": spec.hash, "vocab_hash": spec.vocab_hash, "text_model": text_meta["model"]}
    else:
        hashes = read_manifest(tables_dir)
        if hashes.get("spec_hash") != spec.hash or hashes.get("vocab_hash") != spec.vocab_hash or \
                hashes.get("layout_hash") != spec.layout_hash:
            raise ValueError("%s/manifest.json is not the manifest of spec.json" % tables_dir)
    if hashes.get("text_model") != text_meta["model"]:
        raise ValueError("%s/manifest.json names text model %s, the text table %s is %s"
                         % (tables_dir, hashes.get("text_model"), text_table, text_meta["model"]))
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    lineage = read_lineage(tables_dir)
    return Tables(spec, f32(card["features"]), f32(text["vectors"]), f32(ability["features"]),
                  f32(part["features"]), np.ascontiguousarray(part["ability"], dtype=np.int64), hashes, lineage)


# ── the model manifest and the compatibility check ──────────────────────────────────────────────

MANIFEST_KEYS = ("spec_hash", "vocab_hash", "text_model", "layout_hash", "tables_digest", "vocab", "registry")
SHOWN = 4
REBUILD = ("; an export is built for one vocab and tables: rebuild it from its checkpoint on the current tables "
           "(docs/featurizer.md §6.2)")


def read_manifest(tables_dir):
    """A tables directory's model manifest (manifest.json with registry.json's text); ValueError for a
    directory exported before manifests."""
    path = os.path.join(tables_dir, "manifest.json")
    try:
        with open(path) as fp:
            m = json.load(fp)
        with open(os.path.join(tables_dir, "registry.json")) as fp:
            m["registry"] = fp.read()
    except FileNotFoundError:
        raise ValueError("%s has no manifest.json / registry.json: tables exported before model manifests "
                         "(export them again: rmfeat --export-tables, or rmtrain retable for a run)" % tables_dir)
    m["vocab"] = [[int(i), str(n)] for i, n in m.get("vocab", [])]
    return m


def read_lineage(tables_dir):
    """lineage.json: the earlier tables (spec_hash, vocab_hash, layout_hash, dir) these tables extend."""
    try:
        with open(os.path.join(tables_dir, "lineage.json")) as fp:
            return json.load(fp)
    except FileNotFoundError:
        return []


def manifest_of(obj):
    """The model manifest a checkpoint's ``hashes``, an export's metadata or an InferencePolicy carries:
    a dict of MANIFEST_KEYS (``vocab`` as [id, name] lists; it may be stored as its JSON text)."""
    get = obj.get if isinstance(obj, dict) else (lambda k, d=None: getattr(obj, k, d))
    m = {k: get(k) for k in MANIFEST_KEYS if get(k) is not None}
    v = m.get("vocab")
    if isinstance(v, str):
        v = json.loads(v)
    if v is not None:
        m["vocab"] = [[int(i), str(n)] for i, n in v]
    return m


def vocab_text(vocab):
    """The vocab manifest as the JSON text exports carry (compat.h's vocab_json)."""
    return json.dumps([[int(i), str(n)] for i, n in vocab], ensure_ascii=False, separators=(",", ":"))


def registry_entries(text):
    """The layout registry's entries, each a hashable tuple: a registry extends another (no entry
    moved, registry.h's registry_moves) when its entries include the other's."""
    doc = json.loads(text)
    out = {("layout_version", doc.get("layout_version")), ("dims", json.dumps(doc.get("dims"), sort_keys=True))}
    for e in doc.get("enums", []):
        out.add(("enum", e["name"], e["capacity"], bool(e.get("closed", False))))
        out.update(("enum_value", e["name"], i, v) for i, v in enumerate(e["values"]))
    for sec in ("token_types", "numeric", "categorical", "cards", "abilities", "relations", "tables", "columns",
                "ab_keys"):
        out.update((sec, json.dumps(item, sort_keys=True)) for item in doc.get(sec, []))
    return out


def _listed(items):
    return ", ".join(items[:SHOWN]) + (" and %d more" % (len(items) - SHOWN) if len(items) > SHOWN else "")


def check_compatible(model, now, exact=False):
    """None when a model carrying manifest `model` may run on tables of manifest `now` (both
    ``manifest_of`` dicts, or anything it takes), else ``(code, message)``; see the module docstring."""
    model, now = manifest_of(model), manifest_of(now)

    def refuse(code, msg):
        return code, msg + (REBUILD if exact else "")

    if not model.get("layout_hash"):
        for k in ("spec_hash", "vocab_hash", "text_model"):
            if model.get(k) != now.get(k):
                return refuse(k, "made before model manifests (no layout_hash), its %s %s is not the tables' %s"
                              % (k, model.get(k), now.get(k)))
        return None
    if model["layout_hash"] != now.get("layout_hash"):
        return refuse("layout_hash", "its layout_hash %s is not the tables' %s (a layout bump: migrate the checkpoint)"
                      % (model["layout_hash"], now.get("layout_hash")))
    if model.get("text_model") != now.get("text_model"):
        return refuse("text_model", "its text_model %s is not the text table's %s (a re-pinned text model)"
                      % (model.get("text_model"), now.get("text_model")))
    cur = {i: n for i, n in now.get("vocab", [])}
    known = {i for i, _ in model.get("vocab", [])}
    gone = ["card %d '%s'" % (i, n) for i, n in model.get("vocab", []) if i not in cur]
    if gone:
        return refuse("vocab_hash", "the current vocab lacks %s (an id removed)" % _listed(gone))
    renamed = ["card %d is '%s' in the model's vocab, '%s' now" % (i, n, cur[i])
               for i, n in model.get("vocab", []) if cur[i] != n]
    if renamed:
        return refuse("vocab_hash", "%s (an id renamed or reused)" % _listed(renamed))
    if exact:
        added = ["card %d '%s'" % (i, n) for i, n in now.get("vocab", []) if i not in known]
        if added:
            return refuse("vocab_hash", "the current vocab has %s, which the model's lacks" % _listed(added))
    if model.get("registry") and now.get("registry"):
        moved = registry_entries(model["registry"]) - registry_entries(now["registry"])
        if moved:
            return refuse("layout_hash", "the layout registry moved what the model was trained on: %s"
                          % _listed(sorted(" ".join(str(x) for x in e) for e in moved)))
    if exact:
        for k in ("spec_hash", "vocab_hash"):
            if model.get(k) != now.get(k):
                return refuse(k, "its %s %s is not the tables' %s" % (k, model.get(k), now.get(k)))
        a, b = model.get("tables_digest"), now.get("tables_digest")
        if a and b and a != b:
            return refuse("tables_digest", "its tables' digest %s is not the current tables' %s (a known card's rows "
                          "changed: the table drift, rmfeat --table-drift)" % (a, b))
    return None


def require_compatible(model, tables, what, exact=False):
    """Raises ValueError naming `what` when `model`'s manifest does not run on `tables`."""
    r = check_compatible(model, tables.hashes, exact)
    if r is not None:
        raise ValueError("%s: %s" % (what, r[1]))
