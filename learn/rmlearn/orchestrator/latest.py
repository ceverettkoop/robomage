"""``runs/latest``: the most recent policy under one fixed name (docs/rmtrain.md §4.1).

    dest = latest_dir(run.path)                      # <the runs root>/latest, beside the run directories
    write(dest, run.path, "m7", {"ts": ".../exports/m7.ts.pt"}, tables_dir, model_src=".../ckpt/fixed/m7.pt",
          learner_steps=4200, version=4200, trained_at=t)
    rebuild_exports(dest, run.path, run.exports, run.tables, ["ts", "pt2"])   # after a retable of that run

The directory holds copies, never links into a run, so it is copied or archived as it stands and
outlives the run it came from:

- ``model.pt``: the network as a model checkpoint (``model.save_checkpoint``: what ``run.start_from``
  loads), copied from a fixed point or written from an R-NaD learner state (``final``);
- ``policy.ts.pt`` / ``policy.pt2`` / ``policy.onnx``: the run's exports of that checkpoint, in the
  formats of its ``checkpoint.export_formats``;
- ``latest.json``: the run (``run``, ``run_dir``), the policy's ``name`` (``m<k>``, ``final``),
  ``learner_steps``, ``version``, ``trained_at`` and ``written_at`` (Unix seconds), ``engine_commit`` (the
  repository's HEAD when written; null outside a git checkout), ``manifest`` (the tables' model manifest
  the exports were made over: spec, vocab, layout hashes, tables digest, text model, card count), and
  ``files``: each file's ``sha256`` and ``bytes``.

Every file is written through a temporary file in the directory renamed into place, ``latest.json``
last, so a reader that finds a ``latest.json`` whose sha256s match the files has a whole policy; a
policy file of a format the new policy lacks is removed after ``latest.json`` is in place. The last
policy written wins (two runs exporting at once: no locking beyond the renames). Stdlib only: the
caller writes ``model.pt`` from a learner state (``save_model``).
"""

import hashlib
import os
import re
import shutil
import subprocess
import threading
import time

from .rundir import REPO_ROOT, read_json, write_json

NAME = "latest"                 # the reserved run name (config.run_name_refusal)
MODEL = "model.pt"
INFO = "latest.json"
FILES = {"ts": "policy.ts.pt", "pt2": "policy.pt2", "onnx": "policy.onnx"}       # format -> file in latest/
SUFFIX = {"ts": ".ts.pt", "pt2": ".pt2", "onnx": ".onnx"}                        # format -> the run's export suffix
EXPORTED_WITHOUT_LATEST = 3     # the exporter's exit status: the exports made, latest/ not written


def latest_dir(run_path):
    """The ``latest`` directory beside the run directory `run_path` (its parent's ``latest``)."""
    return os.path.join(os.path.dirname(os.path.abspath(run_path).rstrip("/")), NAME)


def is_release(name):
    """Whether an export name is one ``latest`` takes: an R-NaD fixed point ``m<k>`` or ``final``."""
    return name == "final" or re.fullmatch(r"m\d+", name) is not None


def run_exports(prefix, formats):
    """{format: path} of the exports PREFIX.<suffix> in `formats` that exist."""
    out = {}
    for f in formats:
        p = prefix + SUFFIX[f]
        if os.path.exists(p):
            out[f] = p
    return out


def read(dest):
    """``latest.json`` of `dest` (None when there is none)."""
    return read_json(os.path.join(dest, INFO))


def names_run(info, run_path):
    """Whether a ``latest.json`` names the run directory `run_path`."""
    return bool(info) and info.get("run_dir") == os.path.realpath(run_path)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def engine_commit():
    """The repository's HEAD (None outside a git checkout)."""
    try:
        r = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def write(dest, run_path, name, exports, tables_dir, model_src=None, save_model=None, learner_steps=None,
          version=None, trained_at=None):
    """`run_path`'s policy `name` written into `dest`: ``model.pt`` copied from `model_src` (a model
    checkpoint) or written by ``save_model(path)``, the `exports` ({format: path}) copied, then
    ``latest.json``; its document."""
    if (model_src is None) == (save_model is None):
        raise ValueError("latest: one of model_src and save_model")
    os.makedirs(dest, exist_ok=True)
    files = {MODEL: _put(dest, MODEL, model_src, save_model)}
    for f in sorted(exports):
        files[FILES[f]] = _put(dest, FILES[f], exports[f])
    info = {"run": os.path.basename(os.path.realpath(run_path)), "run_dir": os.path.realpath(run_path), "name": name,
            "learner_steps": learner_steps, "version": version, "trained_at": trained_at, "written_at": time.time(),
            "engine_commit": engine_commit(), "manifest": _manifest(tables_dir), "files": files}
    _finish(dest, info)
    return info


def rebuild_exports(dest, run_path, exports_dir, tables_dir, formats, rebuilt=None):
    """After the run `run_path`'s exports were made again (a retable; new tables at a start): when
    ``latest.json`` of `dest` names that run (and its policy is among the names `rebuilt`, when given),
    its exports are copied again from ``exports_dir/<name>`` and ``latest.json`` rewritten with their
    sha256s and the tables' manifest; ``model.pt`` stays. The new document, or None when latest is
    another run's (or none)."""
    info = read(dest)
    if not names_run(info, run_path) or (rebuilt is not None and info.get("name") not in rebuilt):
        return None
    files = {MODEL: info["files"][MODEL]} if MODEL in info.get("files", {}) else {}
    exports = run_exports(os.path.join(exports_dir, info["name"]), formats)
    for f in sorted(exports):
        files[FILES[f]] = _put(dest, FILES[f], exports[f])
    info = dict(info, files=files, manifest=_manifest(tables_dir), written_at=time.time())
    _finish(dest, info)
    return info


def _manifest(tables_dir):
    from .retable import tables_summary
    return tables_summary(tables_dir)


def _put(dest, fname, src=None, save=None):
    """`src` copied (or ``save(path)`` written) to dest/`fname` through a temporary file renamed into
    place; {"sha256", "bytes"} of the file."""
    out = os.path.join(dest, fname)
    tmp = "%s.tmp%d.%d" % (out, os.getpid(), threading.get_ident() % 100000)
    try:
        if save is not None:
            save(tmp)
        else:
            shutil.copyfile(src, tmp)
        with open(tmp, "rb+") as fp:
            os.fsync(fp.fileno())
        rec = {"sha256": sha256_of(tmp), "bytes": os.path.getsize(tmp)}
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return rec


def _finish(dest, info):
    """``latest.json`` in place, then the policy files it does not list removed."""
    write_json(os.path.join(dest, INFO), info)
    for fname in FILES.values():
        if fname not in info["files"] and os.path.exists(os.path.join(dest, fname)):
            os.unlink(os.path.join(dest, fname))
