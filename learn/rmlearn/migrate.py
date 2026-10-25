"""Checkpoints of an earlier layout onto the current one, by name (docs/open_vocab_plan.md S5).

    python -m rmlearn.migrate OLD.pt --tables NEW_TABLES --seen IDS --out NEW.pt [--old-tables DIR]
    python -m rmlearn.migrate --run runs/league24h-sb --seen IDS [--tables NEW_TABLES]

**What moves.** A network's parameters are indexed by the layout (docs/featurizer.md §6.1): the
token type rows of ``type_emb``, the categorical ids of ``cat_emb``, the numeric columns of
``num_proj``, the relation rows of each layer's ``rel_bias``, the card / ability / part table columns
of ``card_proj`` / ``ability_proj`` / ``part_proj`` and the card / ability slot rows. Each is copied
from its old position to its new one **by name**: a type by its name, a numeric slot by (type, slot
name), a categorical id by (type, slot name) and the value's name (an enum value through the old
format's SCHEMA enum strings, read from the old run's cards frame; a player-relative or hashed id by
its index), a relation by (table, column, target), a table column by its name, a card / ability slot
by (type, slot name). Every position the old layout lacks reads zero (reserved capacity, and the
none / pad rows' places). The omniscient critic's own encoder is remapped the same way. Everything
else must keep its name and shape.

**What is dropped:** only what S1 removed (``REMOVED_TYPES``, ``REMOVED_VALUES``,
``REMOVED_PARAMS``: the ``lists`` / ``b_lists`` token types and their slots, the ``EXPERT_BELIEF``
knowledge level, the ``aux_list`` / ``aux_archetype`` heads). Any other old name with no place in
the current layout refuses the migration, naming it: a migration that would silently drop an input
the network reads is not one.

**Seen cards.** A network made before the seen-card mask (S4) gets ``card_seen`` from ``--seen``
(``CardRows.set_seen``: the mask exactly those ids, every other ``card_id`` row zeroed): the card ids
the run's training inputs held, a JSON list or an object with ``all`` (``runs/open_vocab_s4/
card_ids_<run>.json``). Its ``card_id`` rows do not tell (S4's Done).

**Kinds of file.** A ``model.save_checkpoint`` file (``config``, ``hashes``, ``state_dict``; its
``extra`` kept, ``extra["migrated"]`` added); beside a BC checkpoint its optimizer state
``<path>.opt`` (``BCTrainer.save``; Adam's moments remapped like their parameters); an R-NaD learner
state (``{"learner", "window"}``): ``model``, ``target`` and each regularisation policy remapped,
Adam's ``m`` / ``v`` like their parameters (``card_id``'s unseen rows zeroed with the rows), every
counter, generator and the configs kept, and the replay window emptied: the old episodes are of the
old layout and are not migrated (a resumed run refills the window by self-play). Every migrated file
carries the current tables' model manifest (``hashes``), not the old one, and is loaded back through
the ordinary loaders before it is written.

**A run** (``--run DIR``): every ``DIR/ckpt/**/*.pt`` but TorchScript exports (and BC ``.opt`` files) migrated into
``DIR/ckpt.migrating/``, then the originals moved aside to ``DIR/ckpt.v1/`` and the migrated set to
``DIR/ckpt/`` (with ``migration.json``: per file its source, both sha256s, the old spec hash, the seen
ids), and ``DIR/exports/`` (exports of the old tables) moved to ``DIR/exports.v1/``; nothing is
deleted. The old tables are the run's (``run.tables`` of its ``config.json``, else ``DIR/tables``)
and its cards frame ``DIR/cards.rmstate``. A run's checkpoints then sit where ``state.json`` names
them: ``learn/rmtrain retable DIR`` (docs/rmtrain.md §1.1) exports the current tables into
``DIR/tables/`` (keeping the old ones), checks every migrated checkpoint and rebuilds the exports.
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

from . import model as M
from . import spec as S

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))
import rmstate  # noqa: E402

FORMAT = 1
# What S1 removed (docs/open_vocab_plan.md S1's Done): dropped, never refused.
REMOVED_TYPES = ("lists", "b_lists")
REMOVED_VALUES = {("knowledge_level", "EXPERT_BELIEF")}
REMOVED_PARAMS = ("aux_list.", "aux_archetype.")
OLD_FORMAT_VERSIONS = (1, 2)    # RMSTATE format versions whose SCHEMA frame names the old enums
EMPTY_WINDOW = {"files": [], "fresh": [], "complete": True}


class MigrationError(Exception):
    pass


# ── the layouts ───────────────────────────────────────────────────────────────────────────────

def old_enums(cards_path):
    """{enum name: [value names]} of the SCHEMA frame of the cards file `cards_path`."""
    with open(cards_path, "rb") as fp:
        for payload in rmstate.records(fp):
            return rmstate.Schema.from_frame(payload, versions=OLD_FORMAT_VERSIONS).enums
    raise MigrationError("%s holds no SCHEMA frame" % cards_path)


def read_spec_doc(tables_dir):
    with open(os.path.join(tables_dir, "spec.json"), "rb") as fp:
        raw = fp.read()
    return json.loads(raw), "%016x" % S.fnv1a64(raw.rstrip(b"\n"))


class Layout:
    """One layout's positions, by name: from a FEATURE_SPEC document (any version: a running offset
    layout before the registry, the registry's positions after it) and its enums' value names."""

    def __init__(self, doc, enums):
        self.doc = doc
        self.registry = "layout_hash" in doc
        self.types = {t["name"]: t for t in doc["token_types"]}
        if not self.registry and [t["id"] for t in doc["token_types"]] != list(range(len(doc["token_types"]))):
            raise MigrationError("a pre-registry spec whose token types are not in id order")
        self.n_types = doc["type_capacity"] if self.registry else len(doc["token_types"])
        self.n_rel = doc["relation_capacity"] if self.registry else len(doc["relation_types"])
        self.num_at = {}
        at = 0
        for t in doc["token_types"]:
            self.num_at[t["name"]] = t["num_at"] if self.registry else at
            at += len(t["numeric"])
        self.G = doc["G"] if self.registry else at
        self.enums = enums
        self.relations = {(r["table"], r["column"], r["target"]): r["id"] for r in doc["relation_types"]}
        self.columns = {k: doc["card_tables"][k]["columns"] for k in ("card", "ability", "part")}

    def numeric(self, t):
        """{slot name: column in G} of type `t`."""
        base = self.num_at[t["name"]]
        return {s["name"]: base + (s["at"] if self.registry else j) for j, s in enumerate(t["numeric"])}

    def slot_columns(self, t, key):
        """{slot name: tok_card / tok_ability column} of type `t`'s card or ability slots."""
        return {s["name"]: (s["col"] if self.registry else j) for j, s in enumerate(t[key])}

    def values(self, slot):
        """A categorical slot's value names by local id (local 0, the none id, is None), or None for a
        slot whose ids carry no names (player-relative, hashed)."""
        src = slot["source"]
        if not src.startswith("enum:"):
            return None
        name = src[len("enum:"):]
        vals = self.enums.get(name)
        if vals is None:
            raise MigrationError("enum %s of a categorical slot has no value names" % name)
        if len(vals) + 1 > slot["size"]:
            raise MigrationError("enum %s has %d values, its slot %d ids" % (name, len(vals), slot["size"]))
        return [None] + list(vals)


class LayoutMap:
    """The old → new position maps of every layout-indexed parameter, by name; ``dropped`` lists what
    the old layout had and the new lacks (all of it S1's removals, else MigrationError)."""

    def __init__(self, old, new):
        self.old, self.new = old, new
        self.dropped = []
        bad = []
        self.type_rows, self.num_cols, self.cat_rows = [], [], [(0, 0)]
        self.card_slots, self.ability_slots = {}, {}
        for name, t in old.types.items():
            nt = new.types.get(name)
            if nt is None:
                (self.dropped if name in REMOVED_TYPES else bad).append("token type %s" % name)
                continue
            if name in REMOVED_TYPES:
                bad.append("token type %s, which S1 removed, is in the new layout" % name)
            self.type_rows.append((t["id"], nt["id"]))
            onum, nnum = old.numeric(t), new.numeric(nt)
            for s, c in onum.items():
                if s not in nnum:
                    bad.append("numeric slot %s.%s" % (name, s))
                    continue
                self.num_cols.append((c, nnum[s]))
            ncat = {s["name"]: s for s in nt["categorical"]}
            for s in t["categorical"]:
                ns = ncat.get(s["name"])
                if ns is None:
                    bad.append("categorical slot %s.%s" % (name, s["name"]))
                    continue
                bad += self._cat_slot(name, s, ns)
            for key, out in (("cards", self.card_slots), ("abilities", self.ability_slots)):
                ocol, ncol = old.slot_columns(t, key), new.slot_columns(nt, key)
                for s, c in ocol.items():
                    if s not in ncol:
                        bad.append("%s slot %s.%s" % (key, name, s))
                    elif out.setdefault(c, ncol[s]) != ncol[s]:
                        bad.append("%s slot column %d moves to two columns" % (key, c))
        self.rel_rows = []
        for key, r in old.relations.items():
            nr = new.relations.get(key)
            if nr is None:
                bad.append("relation %s.%s -> %s" % key)
                continue
            self.rel_rows += [(r, nr), (old.n_rel + r, new.n_rel + nr)]
        self.table_cols = {}
        for k in ("card", "ability", "part"):
            ncol = {c: j for j, c in enumerate(new.columns[k])}
            pairs = []
            for j, c in enumerate(old.columns[k]):
                if c not in ncol:
                    bad.append("%s table column %s" % (k, c))
                else:
                    pairs.append((j, ncol[c]))
            self.table_cols[k] = pairs
        if bad:
            raise MigrationError("the old layout has %d name(s) the current one lacks (not S1's removals): %s"
                                 % (len(bad), "; ".join(bad[:8]) + (" ..." if len(bad) > 8 else "")))

    def _cat_slot(self, type_name, s, ns):
        """Appends the slot's id pairs; the problems found."""
        bad = []
        ov, nv = self.old.values(s), self.new.values(ns)
        if (ov is None) != (nv is None):
            return ["categorical slot %s.%s changed kind" % (type_name, s["name"])]
        if ov is None:
            if s["size"] > ns["size"]:
                return ["categorical slot %s.%s shrank" % (type_name, s["name"])]
            self.cat_rows += [(s["offset"] + i, ns["offset"] + i) for i in range(s["size"])]
            return bad
        enum = s["source"][len("enum:"):]
        at = {v: i for i, v in enumerate(nv)}
        for i, v in enumerate(ov):
            if v not in at:
                if (enum, v) in REMOVED_VALUES:
                    self.dropped.append("value %s of %s.%s" % (v, type_name, s["name"]))
                else:
                    bad.append("value %s of %s.%s (enum %s)" % (v, type_name, s["name"], enum))
                continue
            self.cat_rows.append((s["offset"] + i, ns["offset"] + at[v]))
        return bad

    # ── tensors ──

    def pairs(self, name):
        """("rows" | "cols", [(old, new)] positions, the old layout's size) of a layout-indexed
        parameter `name` (a critic's own too), or None for any other."""
        base = name[len("critic."):] if name.startswith("critic.") else name
        if base == "encoder.embed.type_emb.weight":
            return "rows", self.type_rows, self.old.n_types
        if base == "encoder.embed.cat_emb.weight":
            return "rows", self.cat_rows, self.old.doc["categorical_size"]
        if base.startswith("encoder.layers.") and base.endswith(".rel_bias.weight"):
            return "rows", self.rel_rows, 2 * max(self.old.n_rel, 1)
        if base == "cards.card_slot":
            return "rows", sorted(self.card_slots.items()), max(self.old.doc["K"], 1)
        if base == "cards.ability_slot":
            return "rows", sorted(self.ability_slots.items()), max(self.old.doc["Q"], 1)
        if base == "encoder.embed.num_proj.weight":
            return "cols", self.num_cols, self.old.G
        if base in ("cards.card_proj.weight", "cards.ability_proj.weight", "cards.part_proj.weight"):
            table = base.split(".")[1].split("_")[0]
            return "cols", self.table_cols[table], len(self.old.columns[table])
        return None

    def remap(self, name, old, new_shape):
        """The parameter (or one of its optimizer moments) `name` of the old layout at its new
        positions (zeros elsewhere); None when `name` is not layout-indexed."""
        p = self.pairs(name)
        if p is None:
            return None
        kind, pairs, n_old = p
        size = old.shape[0] if kind == "rows" else old.shape[1]
        if size != n_old:
            raise MigrationError("%s has %d %s, the old layout %d" % (name, size, kind, n_old))
        out = torch.zeros(new_shape, dtype=old.dtype)
        ij = np.asarray(pairs, np.int64).reshape(-1, 2)
        src, dst = torch.from_numpy(ij[:, 0]), torch.from_numpy(ij[:, 1])
        if kind == "rows":
            out[dst] = old[src]
        else:
            out[:, dst] = old[:, src]
        return out


def layout_map(old_tables, old_cards, tables_dir):
    """(LayoutMap, old spec hash) from the old tables directory and cards frame to the tables at
    `tables_dir`."""
    old_doc, old_hash = read_spec_doc(old_tables)
    new_doc, _ = read_spec_doc(tables_dir)
    return LayoutMap(Layout(old_doc, old_enums(old_cards)), Layout(new_doc, new_doc.get("enums", {}))), old_hash


# ── networks ──────────────────────────────────────────────────────────────────────────────────

def read_seen(path_or_ids):
    """The seen card ids from a JSON file (a list, or an object with ``all``) or an iterable."""
    if isinstance(path_or_ids, str):
        with open(path_or_ids) as fp:
            doc = json.load(fp)
        ids = doc["all"] if isinstance(doc, dict) else doc
    else:
        ids = path_or_ids
    return sorted({int(i) for i in ids})


def migrate_state(lm, old_sd, model):
    """`old_sd` (an old network's state_dict) in `model`'s layout (a fresh RMNet over the current
    tables): every new key from its old one, remapped or copied; ``card_seen`` left to the caller."""
    tmpl = model.state_dict()
    out = {}
    for k, t in tmpl.items():
        if k == "cards.card_seen":
            out[k] = t.clone()
            continue
        if k not in old_sd:
            raise MigrationError("the current model's %s has no old counterpart" % k)
        o = old_sd[k]
        r = lm.remap(k, o, t.shape)
        if r is None:
            if tuple(o.shape) != tuple(t.shape):
                raise MigrationError("%s: shape %s, the current model's %s" % (k, tuple(o.shape), tuple(t.shape)))
            r = o.clone()
        out[k] = r.to(t.dtype)
    gone = [k for k in old_sd if k not in tmpl]
    bad = [k for k in gone if not k.startswith(REMOVED_PARAMS) and k != "cards.card_seen"]
    if bad:
        raise MigrationError("old parameters with no place in the current model: %s" % ", ".join(bad[:6]))
    return out


def build_network(lm, cfg, old_sd, tables, seen):
    """The migrated RMNet (eval mode, CPU) of one old network."""
    model = M.RMNet(M.ModelConfig.from_dict(cfg), tables)
    model.load_state_dict(migrate_state(lm, old_sd, model))
    model.cards.set_seen(seen)
    model.eval()
    return model


def migrate_moment(lm, name, old, new_param, seen_mask):
    """One optimizer moment of parameter `name` in the new layout (``card_id``'s unseen rows zeroed,
    as ``set_seen`` zeroes the rows)."""
    r = lm.remap(name, old, new_param.shape)
    if r is None:
        if tuple(old.shape) != tuple(new_param.shape):
            raise MigrationError("the moment of %s has shape %s, the parameter %s" % (name, tuple(old.shape),
                                                                                    tuple(new_param.shape)))
        r = old.clone()
    if name == "cards.card_id.weight":
        r = r * seen_mask.unsqueeze(1).to(r.dtype)
    return r


def old_param_names(old_sd, n):
    """The old network's parameter names in ``named_parameters`` order: its state_dict's keys (a
    network before the seen-card mask had no persistent buffer), checked to number `n`."""
    names = [k for k in old_sd if k != "cards.card_seen"]
    if len(names) != n:
        raise MigrationError("the optimizer state has %d moments, the old network %d parameters" % (n, len(names)))
    return names


def migrate_adam(lm, adam, old_sd, model):
    """An R-NaD ``_Adam.state_dict()`` (``m`` / ``v`` per parameter in order) in `model`'s order."""
    names = old_param_names(old_sd, len(adam["m"]))
    for i, k in enumerate(names):
        if tuple(adam["m"][i].shape) != tuple(old_sd[k].shape):
            raise MigrationError("Adam's moment %d does not have the shape of %s" % (i, k))
    at = {k: i for i, k in enumerate(names)}
    mask = model.cards.card_seen.cpu()
    out = {"t": adam["t"], "m": [], "v": []}
    for k, p in model.named_parameters():
        if k not in at:
            raise MigrationError("no Adam moment for %s" % k)
        out["m"].append(migrate_moment(lm, k, adam["m"][at[k]], p, mask))
        out["v"].append(migrate_moment(lm, k, adam["v"][at[k]], p, mask))
    return out


def migrate_bc_opt(lm, opt_state, old_sd, model):
    """A BCTrainer optimizer file (torch.optim Adam / AdamW over the trainable parameters in order)
    in `model`'s order."""
    o = opt_state["opt"]
    if len(o["param_groups"]) != 1:
        raise MigrationError("a BC optimizer state with %d parameter groups" % len(o["param_groups"]))
    names = old_param_names(old_sd, len(o["param_groups"][0]["params"]))
    at = {k: i for i, k in enumerate(names)}
    mask = model.cards.card_seen.cpu()
    state = {}
    params = [(k, p) for k, p in model.named_parameters() if p.requires_grad]
    for j, (k, p) in enumerate(params):
        st = o["state"].get(o["param_groups"][0]["params"][at[k]])
        if st is None:
            continue
        state[j] = {sk: (migrate_moment(lm, k, sv, p, mask) if sk in ("exp_avg", "exp_avg_sq") else sv)
                    for sk, sv in st.items()}
    group = dict(o["param_groups"][0], params=list(range(len(params))))
    return dict(opt_state, opt={"state": state, "param_groups": [group]})


# ── files ─────────────────────────────────────────────────────────────────────────────────────

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(data, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(data, tmp)
    os.replace(tmp, path)


def _sd(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def migrate_file(src, out, tables, lm, seen, info):
    """Migrates one checkpoint (and a BC ``.opt`` beside it) to `out`; the record of it."""
    data = torch.load(src, map_location="cpu", weights_only=False)
    rec = dict(info, source=os.path.abspath(src), source_sha256=sha256_file(src), out=os.path.abspath(out))
    if "learner" in data:
        L = data["learner"]
        if "cards.card_seen" in L["model"]:
            raise MigrationError("%s already carries the seen-card mask" % src)
        cfg = L["model_config"]
        model = build_network(lm, cfg, L["model"], tables, seen)
        target = build_network(lm, cfg, L["target"], tables, seen)
        regs = []
        for r in L["regs"]:
            if r.get("same_as") is not None:
                regs.append({"same_as": r["same_as"]})
                continue
            reg = build_network(lm, cfg, r["state"], tables, seen)
            regs.append(dict(r, state=_sd(reg)))
        new = dict(L, hashes=dict(tables.hashes), model=_sd(model), target=_sd(target), regs=regs,
                   adam=migrate_adam(lm, L["adam"], L["model"], model))
        rec.update(kind="learner_state", networks=2 + sum(1 for r in regs if "state" in r),
                   window_dropped={"files": len((data.get("window") or {}).get("files", [])),
                                   "fresh": len((data.get("window") or {}).get("fresh", []))})
        doc = {"learner": new, "window": dict(EMPTY_WINDOW), "migrated": rec}
        _save(doc, out)
        from .rnad_learner import RNaDLearner
        back = RNaDLearner.from_state_dict(torch.load(out, map_location="cpu", weights_only=False)["learner"], tables)
        if back.model.cards.seen_ids() != list(seen):
            raise MigrationError("%s: the reloaded learner's seen set is not the one set" % out)
    else:
        if "cards.card_seen" in data["state_dict"]:
            raise MigrationError("%s already carries the seen-card mask" % src)
        model = build_network(lm, data["config"], data["state_dict"], tables, seen)
        extra = dict(data.get("extra") or {})
        rec.update(kind="save_checkpoint")
        extra["migrated"] = rec
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        M.save_checkpoint(model, out + ".part", extra)
        os.replace(out + ".part", out)
        if M.load_checkpoint(out, tables).cards.seen_ids() != list(seen):
            raise MigrationError("%s: the reloaded model's seen set is not the one set" % out)
        if os.path.exists(src + ".opt"):
            st = torch.load(src + ".opt", map_location="cpu", weights_only=False)
            _save(migrate_bc_opt(lm, st, data["state_dict"], model), out + ".opt")
            rec["opt"] = {"source_sha256": sha256_file(src + ".opt"), "sha256": sha256_file(out + ".opt")}
    rec["sha256"] = sha256_file(out)
    return rec


def run_paths(run_dir):
    """(old tables, old cards frame) of a run directory: its config's run.tables or its tables/, and
    its cards.rmstate (or the tables directory's, as rmtrain retable keeps one)."""
    cfg = {}
    try:
        with open(os.path.join(run_dir, "config.json")) as fp:
            cfg = json.load(fp)
    except FileNotFoundError:
        pass
    tables = (cfg.get("run") or {}).get("tables") or os.path.join(run_dir, "tables")
    for cards in (os.path.join(run_dir, "cards.rmstate"), os.path.join(tables, "cards.rmstate"),
                  os.path.join(os.path.dirname(os.path.abspath(tables)), "cards.rmstate")):
        if os.path.exists(cards):
            return tables, cards
    raise MigrationError("%s: no cards.rmstate beside the run or its tables %s" % (run_dir, tables))


def run_of(checkpoint):
    """The run directory of a checkpoint (the directory holding the ckpt/ it is in)."""
    d = os.path.dirname(os.path.abspath(checkpoint))
    while os.path.basename(d) not in ("ckpt", "ckpt.v1") and d != os.path.dirname(d):
        d = os.path.dirname(d)
    return os.path.dirname(d)


def migrate_run(run_dir, tables_dir, seen, log=print, old_tables=None, old_cards=None):
    """Migrates every checkpoint of `run_dir`'s ckpt/ and swaps the directories (module docstring);
    the migration record."""
    ckpt = os.path.join(run_dir, "ckpt")
    keep, stage = os.path.join(run_dir, "ckpt.v1"), os.path.join(run_dir, "ckpt.migrating")
    if os.path.exists(keep):
        raise MigrationError("%s exists: the run was migrated already" % keep)
    files = sorted(p for p in glob.glob(os.path.join(ckpt, "**", "*.pt"), recursive=True)
                   if not p.endswith(".ts.pt"))                     # a TorchScript export beside them stays behind
    if not files:
        raise MigrationError("%s holds no checkpoint" % ckpt)
    if not (old_tables and old_cards):
        t, c = run_paths(run_dir)
        old_tables, old_cards = old_tables or t, old_cards or c
    tables = S.load_tables(tables_dir)
    lm, old_hash = layout_map(old_tables, old_cards, tables_dir)
    shutil.rmtree(stage, ignore_errors=True)
    info = {"tool": "rmlearn.migrate", "format": FORMAT, "old_tables": os.path.abspath(old_tables),
            "old_spec_hash": old_hash, "spec_hash": tables.hashes["spec_hash"],
            "layout_hash": tables.hashes["layout_hash"], "seen": list(seen), "dropped": lm.dropped}
    records = []
    for src in files:
        out = os.path.join(stage, os.path.relpath(src, ckpt))
        t0 = time.time()
        records.append(migrate_file(src, out, tables, lm, seen, dict(info, at=t0)))
        log("migrated %s (%s, %.1f s)" % (os.path.relpath(src, run_dir), records[-1]["kind"], time.time() - t0))
    for p in glob.glob(os.path.join(ckpt, "**", "*"), recursive=True):
        if os.path.isfile(p) and not p.endswith((".pt", ".pt.opt", ".pt2")):
            shutil.copy2(p, os.path.join(stage, os.path.relpath(p, ckpt)))     # e.g. a keep marker
    doc = {"format": FORMAT, "at": time.time(), "run": os.path.abspath(run_dir), "originals": "ckpt.v1",
           "exports_kept": None, "old_tables": os.path.abspath(old_tables), "old_spec_hash": old_hash,
           "tables": os.path.abspath(tables_dir), "manifest": {k: tables.hashes[k] for k in
                                                              ("spec_hash", "vocab_hash", "layout_hash",
                                                               "tables_digest", "text_model")},
           "seen": list(seen), "dropped": lm.dropped,
           "files": [{k: r[k] for k in ("source", "source_sha256", "out", "sha256", "kind")} for r in records]}
    os.rename(ckpt, keep)
    os.rename(stage, ckpt)
    exports = os.path.join(run_dir, "exports")
    if os.path.isdir(exports):
        os.rename(exports, os.path.join(run_dir, "exports.v1"))
        os.makedirs(exports)
        doc["exports_kept"] = "exports.v1"
    for r in doc["files"]:
        r["out"] = os.path.join(os.path.abspath(ckpt), os.path.relpath(r["out"], os.path.abspath(stage)))
        r["source"] = os.path.join(os.path.abspath(keep), os.path.relpath(r["source"], os.path.abspath(ckpt)))
    with open(os.path.join(ckpt, "migration.json"), "w") as fp:
        json.dump(doc, fp, indent=1)
        fp.write("\n")
    log("%s: %d checkpoints migrated into ckpt/ (originals in ckpt.v1/%s)"
        % (run_dir, len(records), ", old exports in exports.v1/" if doc["exports_kept"] else ""))
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoint", nargs="?", help="one checkpoint to migrate (with --out)")
    ap.add_argument("--run", help="a run directory: every ckpt/ checkpoint, then the swap")
    ap.add_argument("--tables", required=True, help="the current tables (rmfeat --export-tables of the engine's "
                                                    "--cards-frame)")
    ap.add_argument("--old-tables", help="the checkpoint's tables (default: those of the run whose ckpt/ holds it)")
    ap.add_argument("--old-cards", help="the old tables' cards frame (default: beside the run or tables)")
    ap.add_argument("--seen", required=True, help="the card ids the training inputs held (JSON list or {all})")
    ap.add_argument("--out", help="the migrated checkpoint (one checkpoint)")
    a = ap.parse_args(argv)
    if bool(a.checkpoint) == bool(a.run):
        ap.error("give one checkpoint or --run")
    seen = read_seen(a.seen)
    try:
        if a.run:
            migrate_run(a.run, a.tables, seen, old_tables=a.old_tables, old_cards=a.old_cards)
            return 0
        if not a.out:
            ap.error("--out")
        if a.old_tables and a.old_cards:
            old_tables, old_cards = a.old_tables, a.old_cards
        else:
            old_tables, old_cards = run_paths(run_of(a.checkpoint))
            old_tables, old_cards = a.old_tables or old_tables, a.old_cards or old_cards
        tables = S.load_tables(a.tables)
        lm, old_hash = layout_map(old_tables, old_cards, a.tables)
        rec = migrate_file(a.checkpoint, a.out, tables, lm, seen,
                           {"tool": "rmlearn.migrate", "format": FORMAT, "old_tables": os.path.abspath(old_tables),
                            "old_spec_hash": old_hash, "spec_hash": tables.hashes["spec_hash"],
                            "layout_hash": tables.hashes["layout_hash"], "seen": seen, "dropped": lm.dropped,
                            "at": time.time()})
        print("migrated %s -> %s (%s)" % (a.checkpoint, a.out, rec["kind"]))
        return 0
    except MigrationError as err:
        print("migrate: %s" % err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
