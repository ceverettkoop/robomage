"""Exports a checkpoint as the actor's formats, in a CPU-only process (the orchestrator's export jobs;
a ``.pt2`` export in a process that sees the GPU would initialise it).

    python -m rmlearn.orchestrator.exporter --tables DIR --checkpoint FILE --prefix PREFIX --version N [--formats ts,pt2]

FILE is a model checkpoint (``model.save_checkpoint``: BC checkpoints, R-NaD fixed points) or an
R-NaD learner checkpoint (its current network). Writes PREFIX.ts.pt / PREFIX.pt2 / PREFIX.onnx and
prints one JSON line with the paths.
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
        model = M.RMNet(M.ModelConfig.from_dict(sd["model_config"]), tables)
        model.load_state_dict(sd["model"])
        return model
    return M.load_checkpoint(path, tables)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tables", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--version", type=int, default=0)
    ap.add_argument("--formats", default="ts,pt2")
    a = ap.parse_args()
    torch.set_num_threads(2)
    t = time.time()
    tables = S.load_tables(a.tables)
    model = load_model(a.checkpoint, tables).eval()
    paths = EX.export_policy(model, a.prefix, a.version, tuple(f for f in a.formats.split(",") if f))
    print(json.dumps({"paths": paths, "seconds": round(time.time() - t, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
