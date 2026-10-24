"""The FEATURE_SPEC document and the card tables (docs/featurizer.md §4, §6) as the learner loads them.

    spec = load_spec("out/spec.json")         # Spec: .doc (the parsed JSON), .hash, dims, sizes
    tables = load_tables("out", text_table)   # Tables: card / text / ability / part arrays + hashes

``load_tables`` reads ``rmfeat --export-tables DIR`` (``card.rmtb``, ``ability.rmtb``,
``part.rmtb``, ``spec.json``) and the text table (``oracle_text.rmtb``, by default the one in
the same directory, else the committed ``bin/resources/embeddings/oracle_text.rmtb``) and checks
they belong together: every table carries the spec's hash, and every table the same vocab hash.
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
    """The whole-vocab tables (float32 numpy arrays) and the hashes a model carries."""

    def __init__(self, spec, card, text, ability, part, part_ability, hashes):
        self.spec = spec
        self.card = card                  # [N_CARD_TYPES + 1, Fc]
        self.text = text                  # [N_CARD_TYPES + 1, 384]
        self.ability = ability            # [Nab, Fa]
        self.part = part                  # [Npart, Fp]
        self.part_ability = part_ability  # [Npart] i64, the ability row each part belongs to
        self.hashes = hashes              # spec_hash, vocab_hash, text_model


def load_tables(tables_dir, text_table=None):
    """The exported tables of `tables_dir` and the text table; ValueError when they disagree."""
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
    hashes = {"spec_hash": spec.hash, "vocab_hash": spec.vocab_hash, "text_model": text_meta["model"]}
    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    return Tables(spec, f32(card["features"]), f32(text["vectors"]), f32(ability["features"]),
                  f32(part["features"]), np.ascontiguousarray(part["ability"], dtype=np.int64), hashes)
