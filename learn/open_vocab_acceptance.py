#!/usr/bin/env python3
"""The checkpoint migration's acceptance test (docs/open_vocab_plan.md S5) against S0's fixtures.

    learn/.venv/bin/python learn/open_vocab_acceptance.py full [--work runs/open_vocab_s5/acceptance]
    learn/.venv/bin/python learn/open_vocab_acceptance.py gate --out-dir DIR [--engine E --rmfeat R]
    learn/.venv/bin/python learn/open_vocab_acceptance.py inputs     # rewrites tests/open_vocab/inputs.json

**(a) The fixtures.** Every runnable checkpoint of S0's manifest (``tests/open_vocab/manifest.json``)
is migrated in memory from its original (``rmlearn.migrate``; the original is the manifest's file, by
its sha256, at its path or in the run's ``ckpt.v1/`` after the cut-over) and each of its networks
evaluated, fp32 on the CPU, on the fixture decisions as today's engine featurizes them (the committed
sources replayed, ``open_vocab_baseline``'s helpers; S2 showed they equal S0's ``*_nolists`` decisions
mapped by name): logits and values against S0's ``logits_nolists`` / ``value_nolists``, a critic's
values on the omniscient frames against ``critic_value_nolists``. Twice: with the seen mask widened to
the fixtures' card ids (a test-only copy: the old function, within ``TOL`` on every decision), and with
the run's true seen set (within ``TOL`` on the decisions holding no card outside it; on the others the
mask's effect, an unseen card reading ``unknown_id``, is reported, not bounded). A checkpoint already
migrated by the cut-over (the file at the manifest's path, with ``migrated`` in it) must equal the
in-memory migration with its true mask parameter for parameter.

**(b) An expansion.** The migrated networks over a copy of today's tables with one card appended
(``expanded_tables``: the next free card id, ``appended_id``, with card 262's rows, its abilities at their stable rows, its
parts; the manifest's vocab grown) load (``spec.check_compatible``) and give identical outputs on
every decision (none mentions the new card); their exports' folded rows differ in the new card's row
only. ``full --expanded DIR`` also loads every migrated checkpoint on another expanded tables
directory (S3's scratch tree's, ``runs/open_vocab_s3/expansion_work/appended/tables``) and compares
outputs on the decisions mentioning neither its new nor its drifted card (``--skip-cards``).

**Modes.** ``full``: every source, every network (a few minutes on 8 processes; the cut-over's log).
``gate``: the gate's subset (``GATE_SOURCES``, ``GATE_CHECKPOINTS``), run by the ``actor`` tier
(``$ROBOMAGE_TORCH_PYTHON``) when the original checkpoints are on disk, else ``skipped:``. ``inputs``:
``tests/open_vocab/inputs.json``, each fixture decision's input digest (``decision_digest``: sha256 of
its seat and omniscient decisions' arrays as rmfeat writes them) under today's featurization, which
the ``featurize`` tier checks without torch (``python/pycheck.py --part open_vocab``): run it only
after showing today's featurization equals S0's mapped one; a decision whose inputs later move
intentionally is dropped from the file (``inputs --drop-moved``), and (a) runs on the rest.

The last line is ``OPEN-VOCAB-ACCEPTANCE: ok …`` or ``… N failure(s)`` (exit 1), or ``skipped: …``
(exit 0).
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import concurrent.futures as CF  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as MP  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import open_vocab_baseline as OVB  # noqa: E402
from rmlearn import batch as B  # noqa: E402
from rmlearn import migrate as MG  # noqa: E402
from rmlearn import model as M  # noqa: E402
from rmlearn import rmtb  # noqa: E402
from rmlearn import spec as S  # noqa: E402

# The fixtures are fp32, and fp32 rounding alone (the migrated layout sums its zero-padded columns in
# another order) moves logits by up to ~5e-5: the fixtures themselves are ~2e-5 from the float64
# function, and in float64 the migrated networks equal the originals exactly (runs/open_vocab_s5/
# fp64_check.py; docs/open_vocab_plan.md S5's Done). A mapping error moves outputs by far more.
TOL = 1e-4
FIXTURE = os.path.join(REPO_ROOT, "tests", "open_vocab")
INPUTS = os.path.join(FIXTURE, "inputs.json")
# The card ids each run's training inputs held (S4's episode scan); the critic fixture is m11's.
SEEN = {"runs/league24h-sb": "runs/open_vocab_s4/card_ids_league24h-sb.json",
        "runs/pilotA": "runs/open_vocab_s4/card_ids_pilotA.json",
        "runs/open_vocab_s0": "runs/open_vocab_s4/card_ids_league24h-sb.json"}
# The gate's subset: corpus games of league and non-league decks (unseen card ids), one self-play match.
GATE_SOURCES = ["delver_vs_doomsday", "mav_vs_delver", "lg_bug__burn", "lg_burn__reanimator", "lg_lands__ur_delver",
                "lg_wrb_energy__wubg_doomsday", "lg_bw_dnt__uw_control", "a2-m17912"]
GATE_CHECKPOINTS = ["runs/league24h-sb/ckpt/fixed/m11.pt", "runs/open_vocab_s0/ckpt/m11-critic.pt",
                    "runs/league24h-sb/ckpt/rnad-final.pt"]
GATE_ROLES = {"runs/league24h-sb/ckpt/rnad-final.pt": ("model", "target")}
APPENDED_FROM = 262     # whose rows the synthetic expansion's new card copies (5 abilities)
TOKEN_BAND = 900        # the first token id (src/card_vocab.h): the new card is the next free id below it
BATCH_ROWS = 32


def log(msg):
    print(msg, flush=True)


# ── decisions ─────────────────────────────────────────────────────────────────────────────────

def fixture_index():
    """(manifest, index arrays of decisions.rmtb, the sources by name)."""
    with open(os.path.join(FIXTURE, "manifest.json")) as fp:
        man = json.load(fp)
    _, idx = rmtb.read(os.path.join(FIXTURE, "baseline", "decisions.rmtb"))
    return man, {k: np.asarray(v) for k, v in idx.items()}, {s["name"]: s for s in man["sources"]}


def featurize(work, engine, rmfeat, names, jobs):
    """(seat Decisions, omniscient Decisions, fixture rows [D]) of the fixture decisions of the sources
    `names` (all when None), in fixture order, as today's engine featurizes them; the tables used."""
    man, idx, by_name = fixture_index()
    args = argparse.Namespace(out=FIXTURE, work=work, bin_dir=os.path.join(REPO_ROOT, "bin"), engine=engine,
                              rmfeat=rmfeat)
    p = OVB.Paths(args)
    os.makedirs(p.work, exist_ok=True)
    OVB.export_tables(p)
    srcs = [s for s in OVB.sources(FIXTURE) if names is None or s[1] in names]
    if names is not None and len(srcs) != len(names):
        raise SystemExit("open_vocab_acceptance: sources %s not all found" % names)
    feats = OVB.featurize_all(p, srcs, jobs)
    seat_parts, omni_parts, rows = [], [], []
    for (kind, name, _), files in zip(srcs, feats):
        si = by_name[name]["index"]
        seat, omni = OVB.seat_decisions(files)
        key = {(int(seat.fields["viewer"][i]), int(seat.fields["game"][i]), int(seat.fields["sideboarding"][i]),
                int(seat.fields["decision"][i])): i for i in range(len(seat))}
        want = np.nonzero(idx["source"] == si)[0]
        pick = []
        for r in want:
            k = (int(idx["viewer"][r]), int(idx["game"][r]), int(idx["sideboarding"][r]), int(idx["decision"][r]))
            if k not in key:
                raise SystemExit("open_vocab_acceptance: %s: fixture decision %s not replayed" % (name, k))
            pick.append(key[k])
        seat_parts.append(seat.select(pick))
        omni_parts.append(omni.select(pick))
        rows.append(want)
    seat, omni = OVB.concat(seat_parts), OVB.concat(omni_parts)
    omni.fields["viewer"] = seat.fields["viewer"].copy()
    rows = np.concatenate(rows)
    order = np.argsort(rows, kind="stable")
    seat, omni, rows = seat.select(order), omni.select(order), rows[order]
    menu = idx["menu_offsets"]
    if not np.array_equal(seat.sizes["a"], menu[rows + 1] - menu[rows]):
        raise SystemExit("open_vocab_acceptance: the replayed menus are not the fixture's")
    return seat, omni, rows, p.tables


def recorded_only(seat, omni, rows):
    """The decisions ``tests/open_vocab/inputs.json`` records (a decision whose inputs moved intentionally is
    dropped from it, ``inputs --drop-moved``); (seat, omniscient, rows, how many were left out)."""
    if not os.path.exists(INPUTS):
        return seat, omni, rows, 0
    with open(INPUTS) as fp:
        kept = json.load(fp)["decisions"]
    pick = np.array([i for i, r in enumerate(rows.tolist()) if str(r) in kept], np.int64)
    return seat.select(pick), omni.select(pick), rows[pick], len(rows) - len(pick)


def decision_ids(decs):
    """[D] sets of the card ids each decision's tokens hold."""
    st = np.concatenate([[0], np.cumsum(decs.sizes["n"])])
    c = decs.arrays["tok_card"]
    return [set(int(x) for x in np.unique(c[st[i]:st[i + 1]]) if x >= 0) for i in range(len(decs))]


def decision_digest(decs, i, starts):
    h = hashlib.sha256()
    for name, key in B.TENSORS:
        st = starts[key]
        h.update(np.ascontiguousarray(decs.arrays[name][st[i]:st[i + 1]]).tobytes())
    return h.hexdigest()[:16]


def digests(decs):
    starts = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in decs.sizes.items()}
    return [decision_digest(decs, i, starts) for i in range(len(decs))]


# ── networks ──────────────────────────────────────────────────────────────────────────────────

def run_of_path(path):
    """runs/<x> (or the nested directory holding ckpt/) of a manifest path."""
    return MG.run_of(os.path.join(REPO_ROOT, path))


def original_of(entry):
    """The original file of a manifest checkpoint: at its path while unmigrated, else in ckpt.v1/; checked
    by the manifest's sha256. None when it is in neither place."""
    p = os.path.join(REPO_ROOT, entry["path"])
    run = run_of_path(entry["path"])
    rel = os.path.relpath(p, os.path.join(run, "ckpt"))
    for c in (p, os.path.join(run, "ckpt.v1", rel)):
        if os.path.exists(c) and MG.sha256_file(c) == entry["sha256"]:
            return c
    return None


def true_seen(entry):
    """The run's seen set: the cut-over file's record, else S4's scan."""
    p = os.path.join(REPO_ROOT, entry["path"])
    if os.path.exists(p):
        data = torch.load(p, map_location="cpu", weights_only=False)
        rec = data.get("migrated") or (data.get("extra") or {}).get("migrated")
        if rec:
            return sorted(rec["seen"]), p
    run = os.path.relpath(run_of_path(entry["path"]), REPO_ROOT)
    return MG.read_seen(os.path.join(REPO_ROOT, SEEN[run])), None


def networks_of(data):
    """{role: (model config, state_dict)} of a loaded checkpoint (S0's roles)."""
    if "learner" in data:
        L = data["learner"]
        out = {"model": (L["model_config"], L["model"]), "target": (L["model_config"], L["target"])}
        for i, r in enumerate(L["regs"]):
            if r.get("same_as") is None:
                out["reg%d" % i] = (L["model_config"], r["state"])
        return out
    return {"model": (data["config"], data["state_dict"])}


_W = {}


def _init(work, tables_dir, threads):
    torch.set_num_threads(threads)
    _W["tables"] = S.load_tables(tables_dir)
    _W["decs"] = {n: OVB.load_decisions(os.path.join(work, n + ".rmtb")) for n in ("seat", "omni")}
    with open(os.path.join(work, "jobs.json")) as fp:
        _W["ctx"] = json.load(fp)


def forward(model, decs, critic=False):
    """[D] values and (unless `critic`) the real menu rows' logits concatenated, in fixed batches."""
    spec = model.spec
    order = np.argsort(decs.sizes["n"], kind="stable")
    D = len(decs)
    value = np.zeros(D, np.float32)
    logits = [None] * D
    with torch.no_grad():
        for g in B.forward_groups(decs.sizes["n"], order, max_rows=BATCH_ROWS):
            sub = decs.select(g)
            inputs = B.to_torch(B.pad(sub, spec))
            if critic:
                seat = torch.from_numpy(np.asarray(sub.fields["viewer"], np.int64))
                v = model.critic(model.cards, 0.0, *inputs[:8], seat)
            else:
                out = model(*inputs, aux=False)
                v = out["value"]
                a = sub.sizes["a"]
                for j, i in enumerate(g):
                    logits[i] = out["logits"][j, :a[j]].numpy().copy()
            value[g] = v.numpy()
    return value, (None if critic else np.concatenate(logits).astype(np.float32))


def compare(got_v, got_l, want_v, want_l, rows_mask, menu_mask):
    """(max |Δ value|, max |Δ logit|) over the decisions `rows_mask` (and their menu rows)."""
    dv = float(np.abs(got_v - want_v)[rows_mask].max()) if rows_mask.any() else 0.0
    dl = 0.0
    if got_l is not None and menu_mask.any():
        dl = float(np.abs(got_l - want_l)[menu_mask].max())
    return dv, dl


def argmax_moves(got_l, want_l, a_sizes, rows_mask):
    st = np.concatenate([[0], np.cumsum(a_sizes)])
    n = 0
    for i in np.nonzero(rows_mask)[0]:
        if np.argmax(got_l[st[i]:st[i + 1]]) != np.argmax(want_l[st[i]:st[i + 1]]):
            n += 1
    return n


def evaluate(job):
    """One network (a manifest checkpoint's role) migrated twice and compared; its result dict."""
    ctx = _W["ctx"]
    tables = _W["tables"]
    if job["kind"] == "params":
        lm = MG.layout_map(job["old_tables"], job["old_cards"], ctx["tables_dir"])[0]
        cfg, sd = networks_of(torch.load(job["original"], map_location="cpu", weights_only=False))[job["role"]]
        same = cutover_equal(job["cutover"], job["role"], MG.build_network(lm, cfg, sd, tables, job["seen"]))
        return {"kind": "params", "checkpoint": job["path"], "role": job["role"], "network": job["net_id"],
                "cutover_equal": same,
                "fails": [] if same else ["%s %s: the cut-over file differs from the migration" % (job["path"],
                                                                                                  job["role"])]}
    seat, omni = _W["decs"]["seat"], _W["decs"]["omni"]
    rows = np.asarray(ctx["rows"], np.int64)
    menu_off = np.asarray(ctx["menu_offsets"], np.int64)
    sel_menu = np.concatenate([np.arange(menu_off[r], menu_off[r + 1]) for r in rows])
    fx = rmtb.read(os.path.join(FIXTURE, "baseline", job["net_id"] + ".rmtb"))[1]
    want_v = np.asarray(fx["value_nolists"])[rows]
    want_l = np.asarray(fx["logits_nolists"])[sel_menu]
    lm = MG.layout_map(job["old_tables"], job["old_cards"], ctx["tables_dir"])[0]
    data = torch.load(job["original"], map_location="cpu", weights_only=False)
    cfg, sd = networks_of(data)[job["role"]]
    res = {"kind": "evaluate", "checkpoint": job["path"], "role": job["role"], "network": job["net_id"], "fails": []}
    seat_ids = [set(x) for x in ctx["seat_ids"]]
    omni_ids = [set(x) for x in ctx["omni_ids"]]
    a = seat.sizes["a"]
    menu_rows = np.repeat(np.arange(len(a)), a)
    for mode in ("widened", "true"):
        seen = sorted(set(job["seen"]) | set(ctx["fixture_ids"])) if mode == "widened" else job["seen"]
        model = MG.build_network(lm, cfg, sd, tables, seen)
        v, lg = forward(model, seat)
        clean = np.ones(len(seat), bool) if mode == "widened" else \
            np.array([s <= set(seen) for s in seat_ids])
        dv, dl = compare(v, lg, want_v, want_l, clean, clean[menu_rows])
        r = {"decisions": int(clean.sum()), "max_dvalue": dv, "max_dlogit": dl}
        if mode == "true":
            other = ~clean
            r["unseen_decisions"] = int(other.sum())
            if other.any():
                ev, el = compare(v, lg, want_v, want_l, other, other[menu_rows])
                r["unseen_max_dvalue"], r["unseen_max_dlogit"] = ev, el
                r["unseen_argmax_moved"] = argmax_moves(lg, want_l, a, other)
            if job.get("cutover"):
                r["cutover_equal"] = cutover_equal(job["cutover"], job["role"], model)
                if not r["cutover_equal"]:
                    res["fails"].append("%s %s: the cut-over file differs from the migration" % (job["path"], job["role"]))
        if max(dv, dl) > TOL:
            res["fails"].append("%s %s %s: max |Δ value| %.2e, |Δ logit| %.2e over %d decisions"
                                % (job["path"], job["role"], mode, dv, dl, int(clean.sum())))
        if model.critic is not None and "critic_value_nolists" in fx:
            want_c = np.asarray(fx["critic_value_nolists"])[rows]
            cv, _ = forward(model, omni, critic=True)
            cclean = np.ones(len(omni), bool) if mode == "widened" else \
                np.array([s <= set(seen) and t <= set(seen) for s, t in zip(seat_ids, omni_ids)])
            r["critic_decisions"] = int(cclean.sum())
            r["critic_max_dvalue"] = float(np.abs(cv - want_c)[cclean].max())
            if r["critic_max_dvalue"] > TOL:
                res["fails"].append("%s critic %s: max |Δ value| %.2e" % (job["path"], mode, r["critic_max_dvalue"]))
        res[mode] = r
        if mode == "true" and job.get("expansion"):
            res["expansion"] = expansion_check(model, seat, v, lg, job, res["fails"])
    return res


def cutover_equal(path, role, model):
    data = torch.load(path, map_location="cpu", weights_only=False)
    got = networks_of(data)[role][1]
    want = model.state_dict()
    return set(got) == set(want) and all(torch.equal(got[k].cpu(), want[k].cpu()) for k in want)


# ── (b) the expansion ─────────────────────────────────────────────────────────────────────────

def appended_id(tables):
    """The synthetic expansion's new card id: the next free one, past the last card id the vocab
    uses below the token band."""
    used = [i for i, _ in tables.hashes["vocab"] if i < TOKEN_BAND]
    return max(used) + 1


def expanded_tables(tables):
    """A copy of `tables` with card ``appended_id`` appended (APPENDED_FROM's card, text, ability and
    part rows at its own stable addresses) and the manifest's vocab grown by it."""
    spec = tables.spec
    per = spec.doc["card_tables"]["ability"]["abilities_per_card"]
    card, text, ability = tables.card.copy(), tables.text.copy(), tables.ability.copy()
    new_id = appended_id(tables)
    if card[new_id].any() or text[new_id].any():
        raise SystemExit("open_vocab_acceptance: card %d is not free" % new_id)
    card[new_id], text[new_id] = card[APPENDED_FROM], text[APPENDED_FROM]
    src, dst = APPENDED_FROM * per, new_id * per
    ability[dst:dst + per] = ability[src:src + per]
    pa = tables.part_ability
    take = np.nonzero((pa >= src) & (pa < src + per))[0]
    part = np.concatenate([tables.part, tables.part[take]])
    part_ability = np.concatenate([pa, pa[take] - src + dst])
    hashes = dict(tables.hashes)
    hashes["vocab"] = list(hashes["vocab"]) + [[new_id, "Synthetic Appended Card"]]
    hashes["tables_digest"] = "synthetic-%s" % hashes.get("tables_digest")
    return S.Tables(spec, card, text, ability, part, part_ability, hashes, tables.lineage)


def folded_changes(a, b):
    """What differs between two CardRows' export folds (``model._fold_rows``): the card rows, the
    ability addresses (each read through its ``ability_index``) and whether the slot vectors."""
    fa, ia = M._fold_rows(a)
    fb, ib = M._fold_rows(b)
    n = a.n_cards
    cards = torch.nonzero((fa[:n] != fb[:n]).any(1)).squeeze(1).tolist()
    abil = torch.nonzero((fa[ia] != fb[ib]).any(1)).squeeze(1).tolist()
    k = a.card_slot.shape[0] + a.ability_slot.shape[0] + 1
    return {"cards": cards, "abilities": abil, "slots": not torch.equal(fa[-k:], fb[-k:])}


def expansion_check(model, seat, v, lg, job, fails):
    """The migrated network saved, loaded over the expanded tables: identical outputs; the export's
    folded rows differ in the new card's row only."""
    tables = _W["tables"]
    ext = _W.setdefault("expanded", expanded_tables(tables))
    path = os.path.join(_W["ctx"]["work"], "expansion_%s_%s.pt" % (job["net_id"], job["role"]))
    M.save_checkpoint(model, path)
    big = M.load_checkpoint(path, ext)
    big.eval()
    v2, lg2 = forward(big, seat)
    dv, dl = float(np.abs(v2 - v).max()), float(np.abs(lg2 - lg).max())
    moved = folded_changes(model.cards, big.cards)
    out = {"max_dvalue": dv, "max_dlogit": dl, "folded_rows_changed": moved}
    if dv or dl:
        fails.append("%s %s: outputs moved over the expanded tables (%.2e, %.2e)" % (job["path"], job["role"], dv, dl))
    per = tables.spec.doc["card_tables"]["ability"]["abilities_per_card"]
    new_id = appended_id(tables)
    src = APPENDED_FROM * per
    used = [i for i in range(per) if tables.ability[src + i].any() or (tables.part_ability == src + i).any()]
    want = {"cards": [new_id], "abilities": [new_id * per + i for i in used], "slots": False}
    if moved != want:
        fails.append("%s %s: the expanded export's folded rows changed otherwise than card %d's: %s"
                     % (job["path"], job["role"], new_id, moved))
    os.unlink(path)
    return out


# ── the modes ─────────────────────────────────────────────────────────────────────────────────

def old_tables_of(run, spec_hash):
    """(old tables, old cards frame) of a run whose checkpoints S0 ran: its tables while unmigrated,
    the tables directory rmtrain retable kept (``tables.<spec_hash>/``, with its cards frame) after."""
    kept = os.path.join(run, "tables.%s" % spec_hash)
    if os.path.exists(os.path.join(kept, "cards.rmstate")):
        return kept, os.path.join(kept, "cards.rmstate")
    tables, cards = MG.run_paths(run)
    if MG.read_spec_doc(tables)[1] != spec_hash:
        raise SystemExit("open_vocab_acceptance: %s: no tables of spec %s" % (run, spec_hash))
    return tables, cards


def plan_jobs(man, checkpoints, roles):
    """The jobs: per distinct network one that evaluates it (the first checkpoint and role holding it),
    and per other (checkpoint, role) already migrated by the cut-over one that only compares the file
    with its in-memory migration (``params``); the fixture checkpoints with no original on disk."""
    jobs, seen_net, missing = [], set(), []
    for entry in man["checkpoints"]:
        if not entry.get("runnable") or (checkpoints is not None and entry["path"] not in checkpoints):
            continue
        orig = original_of(entry)
        if orig is None:
            missing.append(entry["path"])
            continue
        seen, cutover = true_seen(entry)
        old_tables, old_cards = old_tables_of(run_of_path(entry["path"]), man["tables"]["spec_hash"])
        for role, nid in sorted(entry["networks"].items()):
            if roles and entry["path"] in roles and role not in roles[entry["path"]]:
                continue
            kind = "params" if nid in seen_net else "evaluate"
            if kind == "params" and not cutover:
                continue
            seen_net.add(nid)
            jobs.append({"kind": kind, "path": entry["path"], "role": role, "net_id": nid, "original": orig,
                         "seen": seen, "cutover": cutover, "old_tables": old_tables, "old_cards": old_cards,
                         "expansion": role == "model"})
    return jobs, missing


def acceptance(args, names, checkpoints, roles, expansion_every):
    t0 = time.time()
    man, idx, _ = fixture_index()
    work = os.path.abspath(args.work)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(work)
    jobs, missing = plan_jobs(man, checkpoints, roles)
    if not jobs:
        log("skipped: no original of S0's fixture checkpoints on disk (%s)" % ", ".join(missing[:3]))
        return 0
    seat, omni, rows, tables_dir = featurize(os.path.join(work, "feat"), args.engine, args.rmfeat, names, args.jobs)
    seat, omni, rows, moved = recorded_only(seat, omni, rows)
    OVB.save_decisions(os.path.join(work, "seat.rmtb"), seat)
    OVB.save_decisions(os.path.join(work, "omni.rmtb"), omni)
    seat_ids, omni_ids = decision_ids(seat), decision_ids(omni)
    fixture_ids = sorted(set().union(*seat_ids, *omni_ids))
    ev = [j for j in jobs if j["kind"] == "evaluate"]
    for i, j in enumerate(ev):
        j["expansion"] = j["expansion"] and expansion_every > 0 and i % expansion_every == 0
    ctx = {"rows": rows.tolist(), "menu_offsets": idx["menu_offsets"].tolist(), "tables_dir": tables_dir,
           "seat_ids": [sorted(s) for s in seat_ids], "omni_ids": [sorted(s) for s in omni_ids],
           "fixture_ids": fixture_ids, "work": work}
    with open(os.path.join(work, "jobs.json"), "w") as fp:
        json.dump(ctx, fp)
    log("open_vocab_acceptance: %d decisions (%d sources; %d whose inputs moved left out), %d networks (%d more "
        "cut-over files), %d card ids in the fixtures; %d fixture checkpoints without an original on disk%s"
        % (len(seat), len(set(idx["source"][rows].tolist())), moved, len(ev), len(jobs) - len(ev),
                                                      len(fixture_ids), len(missing),
                                                      (" (%s)" % ", ".join(missing)) if missing else ""))
    fails, results = [], []
    ctxm = MP.get_context("spawn")
    with CF.ProcessPoolExecutor(min(args.workers, len(jobs)), mp_context=ctxm, initializer=_init,
                                initargs=(work, tables_dir, args.threads)) as ex:
        for r in ex.map(evaluate, jobs):
            results.append(r)
            fails += r["fails"]
            if r["kind"] == "params":
                log("  %s %s (%s): cut-over file %s" % (r["checkpoint"], r["role"], r["network"],
                                                       "equal to its migration" if r["cutover_equal"] else "DIFFERS"))
                continue
            w, t = r["widened"], r["true"]
            line = ("  %s %s (%s): widened %d dec |Δv| %.1e |Δl| %.1e; true %d dec |Δv| %.1e |Δl| %.1e"
                    % (r["checkpoint"], r["role"], r["network"], w["decisions"], w["max_dvalue"], w["max_dlogit"],
                       t["decisions"], t["max_dvalue"], t["max_dlogit"]))
            if t.get("unseen_decisions"):
                line += "; %d with unseen ids: |Δv| %.2f |Δl| %.2f, argmax moved on %d" % (
                    t["unseen_decisions"], t["unseen_max_dvalue"], t["unseen_max_dlogit"], t["unseen_argmax_moved"])
            if "critic_max_dvalue" in w:
                line += "; critic widened %.1e true %.1e" % (w["critic_max_dvalue"], t["critic_max_dvalue"])
            if "cutover_equal" in t:
                line += "; cut-over file %s" % ("equal" if t["cutover_equal"] else "DIFFERS")
            if "expansion" in r:
                e = r["expansion"]
                line += "; expansion |Δ| %.1e/%.1e, folded rows changed %s" % (e["max_dvalue"], e["max_dlogit"],
                                                                               e["folded_rows_changed"])
            log(line)
    with open(os.path.join(work, "results.json"), "w") as fp:
        json.dump({"results": results, "fails": fails, "seconds": time.time() - t0}, fp, indent=1)
    if args.expanded:
        fails += expanded_dir_check(args, jobs, seat, seat_ids)
    for f in fails:
        log("FAIL " + f)
    tail = "%d networks, %d decisions, %.0f s" % (len(ev), len(seat), time.time() - t0)
    log("OPEN-VOCAB-ACCEPTANCE: " + ("ok, " + tail if not fails else "%d failure(s), %s" % (len(fails), tail)))
    return 1 if fails else 0


def expanded_dir_check(args, jobs, seat, seat_ids):
    """Every migrated network over the tables of `args.expanded` (another vocab's): loads; on a sample of
    networks, outputs identical to today's tables' on the decisions holding none of `args.skip_cards`."""
    fails = []
    torch.set_num_threads(args.threads * 4)
    tables = S.load_tables(os.path.join(args.work, "feat", "tables"))
    ext = S.load_tables(args.expanded)
    skip = {int(x) for x in args.skip_cards.split(",") if x}
    clean = np.array([not (s & skip) for s in seat_ids])
    sub = seat.select(np.nonzero(clean)[0])
    _W["tables"] = tables
    for i, j in enumerate(j for j in jobs if j["kind"] == "evaluate"):
        lm = MG.layout_map(j["old_tables"], j["old_cards"], os.path.join(args.work, "feat", "tables"))[0]
        cfg, sd = networks_of(torch.load(j["original"], map_location="cpu", weights_only=False))[j["role"]]
        model = MG.build_network(lm, cfg, sd, tables, j["seen"])
        path = os.path.join(args.work, "expanded.pt")
        M.save_checkpoint(model, path)
        try:
            big = M.load_checkpoint(path, ext)
        except ValueError as err:
            fails.append("%s %s does not load over %s: %s" % (j["path"], j["role"], args.expanded, err))
            continue
        if i % args.expanded_every == 0:
            big.eval()
            v, lg = forward(model, sub)
            v2, lg2 = forward(big, sub)
            dv, dl = float(np.abs(v - v2).max()), float(np.abs(lg - lg2).max())
            log("  expanded %s %s: loads; %d decisions without cards %s: |Δv| %.1e |Δl| %.1e"
                % (j["path"], j["role"], len(sub), sorted(skip), dv, dl))
            if dv or dl:
                fails.append("%s %s: outputs moved over %s" % (j["path"], j["role"], args.expanded))
    files = sorted({j["cutover"] for j in jobs if j["cutover"]} | set(args.also_load))
    for f in files:
        try:
            data = torch.load(f, map_location="cpu", weights_only=False)
            if "learner" in data:
                from rmlearn.rnad_learner import RNaDLearner
                RNaDLearner.from_state_dict(data["learner"], ext)
            else:
                M.load_checkpoint(f, ext)
        except ValueError as err:
            fails.append("%s does not load over %s: %s" % (f, args.expanded, err))
    log("expanded tables %s: %d cut-over checkpoint files load" % (args.expanded, len(files)))
    log("expanded tables %s: every network loads" % args.expanded)
    return fails


def write_inputs(args):
    """tests/open_vocab/inputs.json from today's featurization of every fixture decision."""
    work = os.path.abspath(args.work)
    seat, omni, rows, _ = featurize(os.path.join(work, "feat"), args.engine, args.rmfeat, None, args.jobs)
    man, idx, _ = fixture_index()
    ds, do = digests(seat), digests(omni)
    old = None
    if args.drop_moved and os.path.exists(INPUTS):
        with open(INPUTS) as fp:
            old = json.load(fp)
    keep = {}
    dropped = []
    for r, a, b in zip(rows.tolist(), ds, do):
        if old is not None:
            prev = old["decisions"].get(str(r))
            if prev is None or prev != [a, b]:
                if prev is not None:
                    dropped.append(r)
                continue
        keep[str(r)] = [a, b]
    doc = {"format": 1, "plan": "docs/open_vocab_plan.md S5",
           "what": "per fixture decision (its row of baseline/decisions.rmtb) the sha256 (16 hex) of its seat and "
                   "omniscient decisions' arrays as rmfeat --featurize writes them (tok_type, tok_num, tok_cat, "
                   "tok_card, tok_ability, tok_mask, rel, act_tok, act_mask), today's featurization equal to S0's "
                   "*_nolists decisions mapped by name; a decision whose inputs moved since is absent",
           "generator": "learn/open_vocab_acceptance.py inputs", "gate_sources": GATE_SOURCES,
           "sources": [s["name"] for s in man["sources"]], "decisions": keep}
    head = {k: v for k, v in doc.items() if k != "decisions"}
    with open(INPUTS, "w") as fp:          # one decision a line, so a drop's diff names the decisions
        fp.write(json.dumps(head, indent=1)[:-2] + ',\n "decisions": {\n')
        fp.write(",\n".join("  %s: %s" % (json.dumps(k), json.dumps(v)) for k, v in keep.items()))
        fp.write("\n }\n}\n")
    log("open_vocab_acceptance: %s: %d decisions%s" % (INPUTS, len(keep),
                                                      ", %d dropped (moved)" % len(dropped) if old else ""))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["full", "gate", "inputs"])
    ap.add_argument("--work", default=os.path.join(REPO_ROOT, "runs", "open_vocab_s5", "acceptance"))
    ap.add_argument("--out-dir", help="gate: the work directory (the gate's <out>/actor/open_vocab)")
    ap.add_argument("--engine", default=os.path.join(REPO_ROOT, "bin", "debug", "robomage"))
    ap.add_argument("--rmfeat", default=os.path.join(REPO_ROOT, "bin", "debug", "rmfeat"))
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--expanded", help="full: another expanded tables directory every migrated network loads on")
    ap.add_argument("--expanded-every", type=int, default=8, help="compare outputs over --expanded for every k-th")
    ap.add_argument("--skip-cards", default="", help="card ids the --expanded tables add or change")
    ap.add_argument("--also-load", nargs="*", default=[], help="full: more migrated checkpoint files to load over "
                                                             "--expanded (the runs not in S0's fixtures)")
    ap.add_argument("--drop-moved", action="store_true", help="inputs: keep only decisions whose digest is unchanged")
    a = ap.parse_args(argv)
    if a.mode == "inputs":
        return write_inputs(a)
    if a.mode == "gate":
        if a.out_dir:
            a.work = a.out_dir
        a.workers = min(a.workers, 4)
        return acceptance(a, GATE_SOURCES, GATE_CHECKPOINTS, GATE_ROLES, 1)
    return acceptance(a, None, None, None, 6)


if __name__ == "__main__":
    sys.exit(main())
