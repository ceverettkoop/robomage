"""The run's tables after a vocab expansion: ``rmtrain retable <run>`` and the start-time check
(docs/rmtrain.md §1.1; docs/open_vocab_plan.md S3).

The vocab is compiled into the engine, so cards appended to it (or a change to how the engine parses
a known card) make the engine's whole-vocab cards frame differ from the run's ``cards.rmstate``,
and its tables from the run's ``tables/``. The actors featurize with the engine's own tables and an
export bakes the card rows in, so a run must not resume on stale tables: ``check_start`` (the
orchestrator, at every start) stops it with the command to run, and ``retable`` (a stopped run):

1. writes the engine's cards frame and exports its tables (``rmfeat --export-tables``) into
   ``<run>/tables.new/``; when the frame is the run's and the tables' spec hash too, nothing changed;
2. checks the new tables against the old ones (the old tables' manifest compatible with the new:
   the same layout and text model, every old card id present under its name) and every checkpoint of
   the run (``ckpt/**/*.pt``) against the new tables (``spec.check_compatible``); any refusal changes
   nothing and is printed;
3. reports the table drift (``rmfeat --table-drift``): the known cards whose card, ability or part
   rows changed, the new and the removed ones;
4. keeps the old tables as ``tables.<old spec_hash>/`` (with the old cards frame), moves the new ones
   to ``tables/`` and the new frame to ``cards.rmstate``, and writes ``tables/lineage.json`` (the
   earlier tables, oldest first: the replay window keeps their episodes, spec.Tables.accepted_spec_hashes);
5. rebuilds the exports (every ``state.json`` export, latest per name, from its checkpoint into
   ``exports/<name>``, in the run's ``checkpoint.export_formats``; the evaluations' and the live
   exports are removed, the orchestrator makes them again);
6. records the change in ``state.json`` (``tables``, ``retables``) and ``metrics.jsonl`` (a
   ``retable`` line).

At start, when the run's tables differ from the ones ``state.json`` records (``same_tables``: the
tables were replaced by hand, or a retable's exports did not finish), the orchestrator re-exports
every export (``latest_exports``). Stdlib only but for the checkpoint check (torch, the learner's venv).
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

from .actors import cpu_env
from .rundir import REPO_ROOT, JsonLines, read_json, write_json

LEARN_DIR = os.path.join(REPO_ROOT, "learn")
SUMMARY_KEYS = ("spec_hash", "vocab_hash", "layout_hash", "tables_digest", "text_model")


class RetableError(Exception):
    pass


def tables_summary(tables_dir):
    """The tables' identity (spec, vocab, layout hashes, the tables' digest, text model, card count)
    from manifest.json; None for tables exported before model manifests."""
    m = read_json(os.path.join(tables_dir, "manifest.json"))
    if m is None:
        return None
    out = {k: m.get(k) for k in SUMMARY_KEYS}
    out["cards"] = len(m.get("vocab", []))
    return out


def engine_paths(cfg):
    """(bin_dir, robomage_nn, rmfeat) of the run's config."""
    bin_dir = cfg["run"]["bin_dir"]
    bin_dir = bin_dir if os.path.isabs(bin_dir) else os.path.join(REPO_ROOT, bin_dir)
    build = os.path.join(bin_dir, cfg["run"]["build"])
    return bin_dir, os.path.join(build, "robomage_nn"), os.path.join(build, "rmfeat")


def write_cards_frame(engine, bin_dir, path):
    subprocess.run([engine, "--cards-frame", path], cwd=bin_dir, env=cpu_env(), check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _read_bytes(path):
    try:
        with open(path, "rb") as fp:
            return fp.read()
    except FileNotFoundError:
        return None


def engine_cards_differ(engine, bin_dir, cards_path):
    """Whether the engine's whole-vocab cards frame is not the bytes of `cards_path`."""
    fd, tmp = tempfile.mkstemp(prefix="rmtrain-cards-", suffix=".rmstate")
    os.close(fd)
    try:
        write_cards_frame(engine, bin_dir, tmp)
        return _read_bytes(tmp) != _read_bytes(cards_path)
    finally:
        os.unlink(tmp)


def check_start(run, cfg, engine, bin_dir):
    """At an orchestrator start: "" when the run's tables are the engine's, else why not and what to run."""
    if cfg["run"]["tables"]:
        return ""                    # tables the user manages (run.tables)
    hint = "stop the run and `learn/rmtrain retable %s` (docs/rmtrain.md §1.1)" % run.path
    if tables_summary(run.tables) is None:
        return "the run's tables %s were exported before model manifests: %s" % (run.tables, hint)
    if engine_cards_differ(engine, bin_dir, run.cards):
        return ("the engine's cards frame is not the run's %s (cards appended to the vocab, or a known card parsed "
                "otherwise): %s" % (run.cards, hint))
    return ""


def same_tables(a, b):
    """Whether two tables summaries name the same tables (a vocab grown without `make text-table`
    keeps the spec and vocab hashes: the card count and the digest tell)."""
    return all(a.get(k) == b.get(k) for k in SUMMARY_KEYS + ("cards",))


def latest_exports(state):
    """The run's exports (state.json's), the latest per name, whose checkpoint still exists."""
    by = {}
    for e in state.get("exports", []):
        by[e["name"]] = e
    return [e for e in by.values() if os.path.exists(e["checkpoint"])]


def stale_exports(run):
    """The evaluations' and the live exports (and their model sources): made again on demand."""
    out = []
    for d in (os.path.join(run.exports, "eval"), os.path.join(run.exports, "live")):
        out += [p for p in glob.glob(os.path.join(d, "*")) if os.path.isfile(p)]
    return out


def run_checkpoints(run):
    """Every checkpoint of the run (ckpt/ and its subdirectories: periodic, phase ends, fixed points)."""
    return sorted(glob.glob(os.path.join(run.ckpt, "**", "*.pt"), recursive=True))


def checkpoint_manifest(path):
    """A checkpoint's manifest (save_checkpoint's ``hashes``, an R-NaD learner state's)."""
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "learner" in data:
        return data["learner"]["hashes"]
    if "hashes" in data:
        return data["hashes"]
    raise RetableError("%s is neither a model checkpoint nor an R-NaD learner state" % path)


def table_drift(rmfeat, bin_dir, old_dir, new_dir):
    """rmfeat --table-drift: {"summary", "changed": [lines], "new": [lines], "removed": [lines]}."""
    r = subprocess.run([rmfeat, "--table-drift", old_dir, new_dir], cwd=bin_dir, env=cpu_env(), capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise RetableError("rmfeat --table-drift failed (%d): %s" % (r.returncode, (r.stderr or r.stdout)[-600:]))
    out = {"summary": "", "changed": [], "new": [], "removed": []}
    for line in r.stdout.splitlines():
        if line.startswith("TABLE-DRIFT: "):
            out["summary"] = line[len("TABLE-DRIFT: "):]
        for kind in ("changed", "new", "removed"):
            if line.startswith("  %s " % kind):
                out[kind].append(line[len("  %s " % kind):])
    return out


def export_checkpoint(tables_dir, checkpoint, prefix, version, formats, log_fp):
    argv = [sys.executable, "-m", "rmlearn.orchestrator.exporter", "--tables", tables_dir, "--checkpoint", checkpoint,
            "--prefix", prefix, "--version", str(version), "--formats", ",".join(formats)]
    log_fp.write("=== %s %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(argv)))
    log_fp.flush()
    env = cpu_env()
    env["PYTHONPATH"] = LEARN_DIR
    r = subprocess.run(argv, cwd=LEARN_DIR, env=env, stdout=log_fp, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RetableError("the export of %s from %s failed (exit %d; see %s)" % (prefix, checkpoint, r.returncode,
                                                                                 log_fp.name))


def retable(run, cfg, log=print):
    """The procedure of the module docstring; the record written to state.json (None: nothing changed).
    RetableError when a check refuses (nothing changed then)."""
    from .. import spec as S
    if cfg["run"]["tables"]:
        raise RetableError("the run reads its tables from run.tables %s: retable manages %s only"
                           % (cfg["run"]["tables"], run.tables))
    bin_dir, engine, rmfeat = engine_paths(cfg)
    for p in (engine, rmfeat):
        if not os.path.exists(p):
            raise RetableError("no %s (make BUILD=%s actor)" % (p, cfg["run"]["build"].upper()))
    new_dir = run.tables + ".new"
    shutil.rmtree(new_dir, ignore_errors=True)
    os.makedirs(new_dir)
    new_cards = os.path.join(new_dir, "cards.rmstate")
    try:
        write_cards_frame(engine, bin_dir, new_cards)
        subprocess.run([rmfeat, "--export-tables", new_cards, new_dir], cwd=bin_dir, env=cpu_env(), check=True,
                       stdout=subprocess.DEVNULL)
        old, new = tables_summary(run.tables), tables_summary(new_dir)
        if old is not None and old["spec_hash"] == new["spec_hash"] and _read_bytes(new_cards) == _read_bytes(run.cards):
            log("the run's tables are the engine's (spec %s, %d cards): nothing to do" % (new["spec_hash"],
                                                                                          new["cards"]))
            shutil.rmtree(new_dir)
            return None
        tables = S.load_tables(new_dir)
        if old is not None:
            r = S.check_compatible(S.read_manifest(run.tables), tables.hashes)
            if r is not None:
                raise RetableError("the new tables do not extend the run's (%s): %s" % (r[0], r[1]))
        refused = []
        ckpts = run_checkpoints(run)
        for ck in ckpts:
            r = S.check_compatible(checkpoint_manifest(ck), tables.hashes)
            if r is not None:
                refused.append("%s: %s" % (os.path.relpath(ck, run.path), r[1]))
        if refused:
            raise RetableError("%d of %d checkpoints do not load on the new tables:\n  %s"
                               % (len(refused), len(ckpts), "\n  ".join(refused)))
        log("%d checkpoints load on the new tables (spec %s, %d cards)" % (len(ckpts), new["spec_hash"], new["cards"]))
        drift = table_drift(rmfeat, bin_dir, run.tables, new_dir) if old is not None else \
            {"summary": "no earlier manifest", "changed": [], "new": [], "removed": []}
        log("table drift: %s" % drift["summary"])
        for kind in ("changed", "new", "removed"):
            for line in drift[kind]:
                log("  %s %s" % (kind, line))
    except Exception as err:  # nothing changed: the new tables are dropped, the reason reported
        shutil.rmtree(new_dir, ignore_errors=True)
        if isinstance(err, RetableError):
            raise
        raise RetableError("%s: %s" % (type(err).__name__, err))
    # The swap: the old tables (and their cards frame) kept, the new ones in place.
    old_name = "tables.%s" % (old["spec_hash"] if old else "pre-manifest")
    keep = os.path.join(run.path, old_name)
    if os.path.exists(keep):
        keep = "%s.%d" % (keep, int(time.time()))
    lineage = S.read_lineage(run.tables)
    earlier = old
    if earlier is None and os.path.exists(os.path.join(run.tables, "spec.json")):    # tables before manifests
        sp = S.load_spec(os.path.join(run.tables, "spec.json"))
        earlier = {"spec_hash": sp.hash, "vocab_hash": sp.vocab_hash, "layout_hash": sp.layout_hash,
                   "tables_digest": None, "text_model": None}
    if earlier is not None:
        lineage.append(dict({k: earlier[k] for k in SUMMARY_KEYS}, dir=os.path.basename(keep), until=time.time()))
    os.rename(run.tables, keep)
    if os.path.exists(run.cards):
        os.rename(run.cards, os.path.join(keep, "cards.rmstate"))
    os.rename(new_cards, run.cards)
    with open(os.path.join(new_dir, "lineage.json"), "w") as fp:
        json.dump(lineage, fp, indent=1)
        fp.write("\n")
    os.rename(new_dir, run.tables)
    log("tables: %s kept as %s; %s is the engine's (lineage: %d earlier)" % (run.tables, keep, run.tables,
                                                                            len(lineage)))
    state = read_json(run.state) or {}
    removed = stale_exports(run)
    for p in removed:
        os.unlink(p)
    rebuilt = []
    with open(run.log("export"), "a", buffering=1) as log_fp:
        for e in latest_exports(state):
            export_checkpoint(run.tables, e["checkpoint"], os.path.join(run.exports, e["name"]), e["version"],
                              cfg["checkpoint"]["export_formats"], log_fp)
            rebuilt.append(e["name"])
            log("re-exported %s from %s" % (e["name"], os.path.relpath(e["checkpoint"], run.path)))
    record = {"at": time.time(), "from": old, "to": new, "kept": os.path.basename(keep), "checkpoints": len(ckpts),
              "drift": drift, "exports": rebuilt, "removed_exports": len(removed)}
    state["tables"] = new
    state.setdefault("retables", []).append(record)
    write_json(run.state, state)
    metrics = JsonLines(run.metrics)
    metrics.write({"time": time.time(), "phase": state.get("phase"), "stage": state.get("stage"), "event": "retable",
                   "from": (old or {}).get("spec_hash"), "to": new["spec_hash"], "cards": new["cards"],
                   "drift": drift["summary"], "changed": len(drift["changed"]), "exports": len(rebuilt)})
    metrics.close()
    return record
