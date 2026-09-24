#!/usr/bin/env python3
"""Regression for the model-spec resolver (opponents.parse_model_spec and its
loaders): the resolver table (kind / prefix / base / search / canonical
evaluator spec for every spec family, knob stripping, case folding), the
recorded-evaluator mapping tree_rebuild.evaluator_spec_for builds on it, and —
with every checkpoint loader stubbed — that the search evaluator, the V(s)
model and the probe net of one spec all read the SAME checkpoint, through
every consumer (analysis, shard_replay, shard_probes, az_inspect).
Torch-free, engine-free, instant."""
from __future__ import annotations

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import opponents  # noqa: E402
from opponents import (MODEL_KIND_AGENT, MODEL_KIND_AZ,  # noqa: E402
                       MODEL_KIND_PPO, MODEL_KIND_UNIFORM, parse_model_spec,
                       strip_spec_knobs)

_FAILS: list[str] = []


def check(cond, what):
    print(("  ok   " if cond else "  FAIL ") + what)
    if not cond:
        _FAILS.append(what)


# spec -> (kind, prefix, base, search, evaluator_spec)
TABLE = [
    ("gen", MODEL_KIND_PPO, "", "gen", False, "mcts:gen"),
    ("", MODEL_KIND_PPO, "", "gen", False, "mcts:gen"),
    (None, MODEL_KIND_PPO, "", "gen", False, "mcts:gen"),
    ("/x/gen__final.zip", MODEL_KIND_PPO, "", "/x/gen__final.zip", False,
     "mcts:/x/gen__final.zip"),
    ("/x/gen__azfinal.pt", MODEL_KIND_AZ, "", "/x/gen__azfinal.pt", False,
     "az:/x/gen__azfinal.pt"),
    ("mcts:gen", MODEL_KIND_PPO, "mcts", "gen", True, "mcts:gen"),
    ("mcts:gen?sims=64&worlds=8", MODEL_KIND_PPO, "mcts", "gen", True,
     "mcts:gen"),
    ("MCTS:gen", MODEL_KIND_PPO, "mcts", "gen", True, "mcts:gen"),
    ("mcts:", MODEL_KIND_PPO, "mcts", "gen", True, "mcts:gen"),
    ("mcts:uniform?sims=8", MODEL_KIND_UNIFORM, "mcts", "uniform", True,
     "uniform"),
    ("uniform", MODEL_KIND_UNIFORM, "", "uniform", False, "uniform"),
    ("az:gen", MODEL_KIND_AZ, "az", "gen", True, "az:gen"),
    ("az:gen?sims=8&time=2", MODEL_KIND_AZ, "az", "gen", True, "az:gen"),
    ("AZ:gen", MODEL_KIND_AZ, "az", "gen", True, "az:gen"),
    ("az:/x/gen__final.zip", MODEL_KIND_AZ, "az", "/x/gen__final.zip", True,
     "az:/x/gen__final.zip"),
    ("azraw:gen", MODEL_KIND_AZ, "azraw", "gen", False, "az:gen"),
    ("azraw:/x/a.pt", MODEL_KIND_AZ, "azraw", "/x/a.pt", False, "az:/x/a.pt"),
    ("scripted", MODEL_KIND_AGENT, "", "scripted", False, "scripted"),
    ("scripted:hard", MODEL_KIND_AGENT, "", "scripted:hard", False,
     "scripted:hard"),
    ("human", MODEL_KIND_AGENT, "", "human", False, "human"),
    ("play:cast:bolt,pass", MODEL_KIND_AGENT, "", "play:cast:bolt,pass",
     False, "play:cast:bolt,pass"),
    ("actions:0,1", MODEL_KIND_AGENT, "", "actions:0,1", False,
     "actions:0,1"),
]


def test_table():
    print("resolver table")
    for spec, kind, prefix, base, search, ev in TABLE:
        ms = parse_model_spec(spec)
        got = (ms.kind, ms.prefix, ms.base, ms.search, ms.evaluator_spec)
        check(got == (kind, prefix, base, search, ev),
              f"{spec!r} -> {got}")
    check(strip_spec_knobs("az:gen?sims=8&c=1.5") == "az:gen",
          "strip_spec_knobs drops the query")
    check(strip_spec_knobs("  gen  ") == "gen", "strip_spec_knobs trims")
    check(strip_spec_knobs(None) == "", "strip_spec_knobs(None)")
    check(parse_model_spec("az:").base == "gen"
          and parse_model_spec("", default="other").base == "other",
          "empty base -> default")


def test_evaluator_spec_for():
    print("tree_rebuild.evaluator_spec_for")
    from tree_rebuild import evaluator_spec_for
    cases = [
        ({"spec": "mcts:uniform?sims=64", "checkpoint": None}, "uniform"),
        ({"checkpoint": "/x/gen__azfinal.pt"}, "az:/x/gen__azfinal.pt"),
        ({"checkpoint": "/x/gen__final.zip"}, "mcts:/x/gen__final.zip"),
        ({"spec": "az:gen?sims=8"}, "az:gen"),
        ({"spec": "mcts:gen?sims=8", "checkpoint": "/x/gen__v9.zip"},
         "mcts:/x/gen__v9.zip"),
        # An az: seat that warm-started records its PPO base as the
        # checkpoint; the rebuild must stay on the AZ ladder.
        ({"spec": "az:gen", "checkpoint": "gen"}, "az:gen"),
        ({"spec": "az:/x/gen__final.zip",
          "checkpoint": "/x/gen__final.zip"}, "az:/x/gen__final.zip"),
        ({}, "uniform"),
    ]
    for prov, want in cases:
        got = evaluator_spec_for(prov)
        check(got == want, f"{prov} -> {got!r} (want {want!r})")


class _Stubs:
    """Replace every checkpoint loader with a recorder: each returns a
    (family, path) token so the test can compare what each view loaded."""

    def __init__(self, tmp, az_present):
        self.tmp = tmp
        self.ppo = os.path.join(tmp, "gen__final.zip")
        self.az = os.path.join(tmp, "gen__azfinal.pt")
        for p in (self.ppo, self.az):
            open(p, "w").close()
        self.az_present = az_present
        self._saved = []

    def _set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name, None)))
        setattr(obj, name, value)

    def __enter__(self):
        import mcts
        stub = self

        def resolve_checkpoint(path, checkpoint_dir=None):
            if path is None:
                return None
            if path.endswith(".zip"):
                return path
            if path == "gen":
                return stub.ppo
            raise ValueError(f"cannot resolve {path!r}")

        def resolve_az_checkpoint(spec, checkpoint_dir=None, prefer="final"):
            if spec == "gen" and stub.az_present:
                return stub.az
            return None

        class _Eval:
            def __init__(self, net):
                self._net = net

        def load_az_evaluator(base, *, ppo_resolver=None, on_warm_start=None,
                              device=None):
            az = resolve_az_checkpoint(base)
            if az:
                return _Eval(("az", az)), az
            if base.endswith(".pt"):
                return _Eval(("az", base)), base
            ppo = (ppo_resolver or opponents.resolve_checkpoint)(base)
            if on_warm_start:
                on_warm_start(base, ppo)
            return _Eval(("az<-ppo", ppo)), base

        fake_az = types.ModuleType("az_net")
        fake_az.resolve_az_checkpoint = resolve_az_checkpoint
        fake_az.from_ppo = lambda p, map_location="cpu": ("az<-ppo", p)
        fake_az.load_az = lambda p, map_location="cpu": ("az", p)
        self._saved.append((sys.modules, "az_net", sys.modules.get("az_net")))
        sys.modules["az_net"] = fake_az
        self._set(opponents, "resolve_checkpoint", resolve_checkpoint)
        self._set(opponents, "load_az_evaluator", load_az_evaluator)
        self._set(opponents, "_load_model", lambda p: ("ppo", p))
        self._set(mcts, "PPOEvaluator",
                  lambda model, v_scale=1.0: _Eval(model))
        self._set(mcts, "UniformEvaluator", lambda: _Eval(None))
        import analysis
        self._set(analysis, "_AZModelAdapter", lambda net: net)
        return self

    def __exit__(self, *exc):
        for obj, name, old in reversed(self._saved):
            if obj is sys.modules:
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
            else:
                setattr(obj, name, old)


def _checkpoint_of(token):
    """The path a stub token names; a PPO net and its from_ppo transcription
    are the same checkpoint."""
    return token[1]


def test_one_net_per_spec():
    print("one net per spec (stubbed loaders)")
    import analysis
    import az_inspect
    import shard_probes
    import shard_replay
    specs = ["gen", "mcts:gen?sims=8", "az:gen?sims=8", "azraw:gen",
             "mcts:{ppo}", "{az}"]
    for az_present in (True, False):
        with tempfile.TemporaryDirectory() as tmp, \
                _Stubs(tmp, az_present) as st:
            for tmpl in specs:
                spec = tmpl.format(ppo=st.ppo, az=st.az)
                ms = parse_model_spec(spec)
                ev, _label = opponents.load_spec_evaluator(spec)
                net, _ = opponents.load_spec_net(spec)
                val, kind = opponents.load_spec_value_model(spec)
                views = {
                    "evaluator": ev._net,
                    "probe net": shard_probes.load_probe_net(spec)[0],
                    "value model": val,
                    "analysis inspection": analysis.load_inspection_model(spec),
                    "shard_replay V(s)": shard_replay.load_value_model(spec),
                    "spec net": net,
                }
                paths = {k: _checkpoint_of(v) for k, v in views.items()}
                check(len(set(paths.values())) == 1,
                      f"[az ckpt {'present' if az_present else 'absent'}] "
                      f"{spec!r}: every view reads {sorted(set(paths.values()))}")
                if ms.kind == MODEL_KIND_PPO:
                    want = st.ppo if "gen" == ms.base else ms.base
                    check(paths["evaluator"] == want and kind == MODEL_KIND_PPO,
                          f"  {spec!r} is the PPO checkpoint")
                if ms.kind == MODEL_KIND_AZ and ms.base == "gen":
                    want = st.az if az_present else st.ppo
                    check(paths["evaluator"] == want,
                          f"  {spec!r} follows the AZ ladder -> {want}")
            # az_inspect.load_net: a bare name is an az: spec on the ladder.
            _net, path = az_inspect.load_net("gen", checkpoint_dir=tmp)
            want = st.az if az_present else st.ppo
            check(path == want,
                  f"az_inspect.load_net('gen') -> {os.path.basename(path)} "
                  f"(az ckpt {'present' if az_present else 'absent'})")
            _net, path = az_inspect.load_net(f"mcts:{st.ppo}",
                                              checkpoint_dir=tmp)
            check(path == st.ppo, "az_inspect.load_net(mcts:<zip>) is the PPO")
    with tempfile.TemporaryDirectory() as tmp, _Stubs(tmp, False):
        for bad in ("uniform", "scripted"):
            try:
                opponents.load_spec_net(bad)
                ok = False
            except ValueError:
                ok = True
            check(ok, f"load_spec_net({bad!r}) refuses (no net)")
        ev, label = opponents.load_spec_evaluator("mcts:uniform?sims=4")
        check(label == "uniform", "uniform evaluator label")


def main():
    print("model-spec resolver regression")
    test_table()
    test_evaluator_spec_for()
    test_one_net_per_spec()
    if _FAILS:
        print(f"\n{len(_FAILS)} failure(s)")
        return 1
    print("\nall ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
