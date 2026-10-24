"""Exports a checkpoint as the actor's formats, in a CPU-only process (the orchestrator's export jobs;
a ``.pt2`` export in a process that sees the GPU would initialise it).

    python -m rmlearn.orchestrator.exporter --tables DIR --checkpoint FILE --prefix PREFIX --version N [--formats ts,pt2]
    python -m rmlearn.orchestrator.exporter --tables DIR --serve

FILE is a model checkpoint (``model.save_checkpoint``: BC checkpoints, R-NaD fixed points, the GPU
process's ``save_model``) or an R-NaD learner checkpoint (its current network). Writes
PREFIX.ts.pt / PREFIX.pt2 / PREFIX.onnx and prints one JSON line with the paths.

``--serve`` (the live exporter, ``orchestrator/live.py``): the tables loaded once, then one JSON
request per stdin line, ``{"id": n, "checkpoint": FILE, "out": PATH, "format": "ts"|"pt2",
"version": N}``, each answered by one JSON line on stdout, ``{"re": n, "ok": true, "out": PATH,
"version": N, "seconds": s}`` (or ``"ok": false, "error": ...``). PATH is replaced atomically (a
temporary file renamed over it), so an actor's ``nn:PATH`` reloads a whole export. Ends at end of input.
"""

import os

os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from .. import export as EX  # noqa: E402
from .. import model as M  # noqa: E402
from .. import spec as S  # noqa: E402


def load_model(path, tables):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "learner" in data:
        sd = data["learner"]
        S.require_compatible(sd["hashes"], tables, path)
        model = M.RMNet(M.ModelConfig.from_dict(sd["model_config"]), tables)
        model.load_state_dict(sd["model"])
        return model
    return M.load_checkpoint(path, tables)


def export_one(model, out, fmt, version):
    """`model` exported as `fmt` at `out` (replaced atomically)."""
    policy = model.inference_module(int(version)).cpu().eval()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with torch.no_grad():
        if fmt == "ts":
            return EX.export_torchscript(policy, out)
        if fmt == "pt2":
            return EX.export_pt2(policy, out, EX.example_inputs(policy))
    raise ValueError("unknown export format %r" % fmt)


def serve(tables_dir):
    proto = os.fdopen(os.dup(1), "w", buffering=1)     # replies; fd 1 (a library's prints) goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    tables = S.load_tables(tables_dir)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        t = time.time()
        req = {}
        try:
            req = json.loads(line)
            model = load_model(req["checkpoint"], tables).eval()
            out = export_one(model, req["out"], req.get("format", "ts"), req.get("version", 0))
            reply = {"re": req.get("id"), "ok": True, "out": out, "version": req.get("version", 0),
                     "seconds": round(time.time() - t, 3)}
        except Exception as err:  # a failed export is a reply; the orchestrator decides
            reply = {"re": req.get("id"), "ok": False, "error": "%s: %s" % (type(err).__name__, err),
                     "seconds": round(time.time() - t, 3)}
        proto.write(json.dumps(reply) + "\n")
        proto.flush()
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--prefix")
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--formats", default="ts,pt2")
    ap.add_argument("--serve", action="store_true", help="export the requests read from stdin (the live exporter)")
    a = ap.parse_args()
    torch.set_num_threads(2)
    if a.serve:
        return serve(a.tables)
    if not a.checkpoint or not a.prefix:
        ap.error("--checkpoint and --prefix (or --serve)")
    t = time.time()
    tables = S.load_tables(a.tables)
    model = load_model(a.checkpoint, tables).eval()
    paths = EX.export_policy(model, a.prefix, a.version, tuple(f for f in a.formats.split(",") if f))
    print(json.dumps({"paths": paths, "seconds": round(time.time() - t, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
