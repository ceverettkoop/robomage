"""Exports of an InferencePolicy in the actor's formats (docs/featurizer.md §6, "Model files").

    paths = export_policy(model, "run/ckpt/p17", version=17)            # {"ts": ..., "pt2": ...}
    paths = export_policy(model, prefix, version, formats=("ts", "pt2", "onnx"))

Each export carries the four metadata strings the actor's evaluator (``src/actor/evaluator.cpp``)
checks: ``spec_hash``, ``vocab_hash``, ``text_model`` and ``policy_version``:

- ``PREFIX.ts.pt``: TorchScript (``torch.jit.script``), the strings as module attributes;
- ``PREFIX.pt2``: an AOTInductor package (``torch.export`` + ``aoti_compile_and_package``, CPU),
  the strings in the package metadata (inductor config ``aot_inductor.metadata``); machine code for
  this CPU's ISA;
- ``PREFIX.onnx`` (optional; needs onnx + onnxscript): the dynamo exporter, the strings in
  ``metadata_props``.

``read_metadata(path)`` returns the four strings of a ``.ts.pt`` or ``.pt2`` file; ``load(path)`` a
callable over the contract's ten inputs.
"""

import importlib
import os
import zipfile

import torch

from .batch import INPUTS
from .model import InferencePolicy

META_KEYS = ("spec_hash", "vocab_hash", "text_model", "policy_version")


def example_inputs(policy, b=2, n=16, r=24, a=8, seed=0):
    """Random inputs of the contract's shapes (for export tracing): every dimension ≥ 2 and not 8,
    so nothing specialises on a size."""
    g = torch.Generator().manual_seed(seed)
    enc = policy.encoder.embed
    T = enc.type_emb.num_embeddings
    F, C, K, Q = policy.F, policy.C, policy.K, policy.Q
    cat_size = enc.cat_emb.num_embeddings
    n_rel = policy.encoder.n_rel
    tok_card = torch.randint(-1, policy.none_card, (b, n, K), generator=g)
    tok_ability = torch.randint(-1, policy.none_ability, (b, n, Q), generator=g)
    tok_mask = torch.ones(b, n, dtype=torch.bool)
    tok_mask[:, n - 3:] = False
    rel = torch.stack([torch.randint(0, n - 3, (b, r), generator=g), torch.randint(0, n - 3, (b, r), generator=g),
                       torch.randint(0, max(n_rel, 1), (b, r), generator=g)], -1)
    rel_mask = torch.ones(b, r, dtype=torch.bool)
    rel_mask[:, r - 4:] = False
    act_mask = torch.ones(b, a, dtype=torch.bool)
    act_mask[:, a - 2:] = False
    return (torch.randint(0, T, (b, n), generator=g), torch.randn(b, n, F, generator=g),
            torch.randint(0, cat_size, (b, n, C), generator=g), tok_card, tok_ability, tok_mask, rel, rel_mask,
            torch.randint(0, n - 3, (b, a), generator=g), act_mask)


def dynamic_shapes():
    from torch.export import Dim
    B, N, R, A = Dim("B", min=1, max=4096), Dim("N", min=2, max=65536), Dim("R", min=2, max=1 << 20), \
        Dim("A", min=2, max=65536)
    tok = {0: B, 1: N}
    return {"tok_type": tok, "tok_num": tok, "tok_cat": tok, "tok_card": tok, "tok_ability": tok, "tok_mask": tok,
            "rel": {0: B, 1: R}, "rel_mask": {0: B, 1: R}, "act_tok": {0: B, 1: A}, "act_mask": {0: B, 1: A}}


def metadata(policy):
    return {k: str(getattr(policy, k)) for k in META_KEYS}


def export_torchscript(policy, path):
    scripted = torch.jit.script(policy)
    tmp = path + ".tmp"
    scripted.save(tmp)
    os.replace(tmp, path)
    return path


def export_pt2(policy, path, example=None):
    importlib.import_module("torch._inductor")
    ex = example if example is not None else example_inputs(policy)
    ep = torch.export.export(policy, ex, dynamic_shapes=dynamic_shapes(), strict=False)
    tmp = path + ".tmp.pt2"
    torch._inductor.aoti_compile_and_package(ep, package_path=tmp,
                                             inductor_configs={"aot_inductor.metadata": metadata(policy)})
    os.replace(tmp, path)
    return path


def export_onnx(policy, path, example=None):
    import onnx
    ex = example if example is not None else example_inputs(policy)
    prog = torch.onnx.export(policy, ex, dynamo=True, dynamic_shapes=dynamic_shapes(), input_names=INPUTS,
                             output_names=["logits", "value"], external_data=False)
    prog.save(path)
    m = onnx.load(path)
    for k, v in metadata(policy).items():
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(m, path)
    return path


def export_policy(model_or_policy, prefix, version=0, formats=("ts", "pt2")):
    """Exports `model_or_policy` (an RMNet, folded here at `version`, or an InferencePolicy) as
    PREFIX.ts.pt / PREFIX.pt2 / PREFIX.onnx; {format: path}."""
    policy = model_or_policy if isinstance(model_or_policy, InferencePolicy) \
        else model_or_policy.inference_module(version)
    policy = policy.cpu().eval()
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    ex = example_inputs(policy)
    out = {}
    with torch.no_grad():
        for f in formats:
            if f == "ts":
                out[f] = export_torchscript(policy, prefix + ".ts.pt")
            elif f == "pt2":
                out[f] = export_pt2(policy, prefix + ".pt2", ex)
            elif f == "onnx":
                out[f] = export_onnx(policy, prefix + ".onnx", ex)
            else:
                raise ValueError("unknown export format %r" % f)
    return out


def _format(path):
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
    if any(n.endswith("archive_format") for n in names):
        return "pt2"
    if any(n.endswith("constants.pkl") for n in names):
        return "ts"
    raise ValueError("%s is neither an AOTInductor package nor a TorchScript module" % path)


def read_metadata(path):
    """The four metadata strings of a .ts.pt or .pt2 export."""
    if path.endswith(".onnx"):
        import onnx
        return {p.key: p.value for p in onnx.load(path).metadata_props if p.key in META_KEYS}
    if _format(path) == "ts":
        m = torch.jit.load(path, map_location="cpu")
        return {k: getattr(m, k) for k in META_KEYS if hasattr(m, k)}
    importlib.import_module("torch._inductor.codecache")
    pkg = torch._inductor.aoti_load_package(path)
    return {k: v for k, v in pkg.get_metadata().items() if k in META_KEYS}


def load(path):
    """A callable (ten inputs) -> (logits, value) of an export."""
    if _format(path) == "ts":
        return torch.jit.load(path, map_location="cpu").eval()
    importlib.import_module("torch._inductor.codecache")
    return torch._inductor.aoti_load_package(path)
