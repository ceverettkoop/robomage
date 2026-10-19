# learn/ — the learner side (PyTorch)

The learner of `docs/rnad_training_plan.md` §4. It needs torch + numpy (`learn/requirements.txt`,
its own venv `learn/.venv`) and is never imported by `python/`, the engine or the gate.

## Layout

| Path | What |
|---|---|
| `rmlearn/rmtb.py` | RMTB files (`docs/featurizer.md` §4.4) in numpy: `read`, `read_bytes`, `write`, `write_bytes` |
| `rmlearn/spec.py` | `load_spec` (FEATURE_SPEC + its hash), `load_tables` (card / text / ability / part tables, hashes checked) |
| `rmlearn/batch.py` | `Decisions` (a tensors file or an RMIP request: unpadded, concatenated), `pad` (= `featurize::pad_batch`), `to_torch`, `Targets` (an `rmfeat --targets` file) |
| `rmlearn/model.py` | `ModelConfig`, `RMNet` (policy, value, auxiliary heads, optional omniscient critic), `InferencePolicy` (the exported contract), `aux_losses`, `masked_log_softmax`, `smoothed_log_policy`, `save_checkpoint` / `load_checkpoint` |
| `rmlearn/export.py` | `export_policy` (TorchScript `.ts.pt`, AOTInductor `.pt2`, optional ONNX, with the four metadata strings), `read_metadata`, `load` |
| `rmlearn/server.py` | `InferenceServer`: RMIP v1 (`docs/inference_protocol.md`) inside the host process; `python -m rmlearn.server` runs it standalone |
| `rmlearn/client.py` | a Python RMIP client (checks, benches) |
| `check.py` | the CPU-only correctness checks |
| `integration.py` | checks and benches against the C++ actor (`robomage_nn`): `actor`, `server`, `bench` |

## Using it

```python
from rmlearn import spec, model, export, server
tables = spec.load_tables("out")                       # rmfeat --export-tables out (+ the text table)
net = model.RMNet(model.ModelConfig(d=192, layers=4), tables)
out = net(*inputs)                                     # logits, value, summary, aux_* (+ critic_value)
export.export_policy(net, "ckpt/p12", version=12)      # p12.ts.pt, p12.pt2 for nn:<model>
srv = server.InferenceServer(sock, device="cuda", max_batch=512, deadline_us=2000)
srv.set_policy(0, net, 12); srv.start()                # actors: nn:remote:<sock>[:policies=0,1,2]
srv.set_policy(1, reg_net, 11)                         # swapped between batches
srv.stats(); srv.stop()
```

## Checks

```sh
learn/.venv/bin/python learn/check.py --tables DIR --tensors A.rmtb [--omniscient O.rmtb --targets T.rmtb]
learn/.venv/bin/python learn/check.py --engine-dir bin/debug --bin-dir bin    # makes the inputs first
```

`check.py` hides the GPU from itself and needs no engine when given its inputs; it ends with
`RMLEARN-CHECK: ok` (exit 0). `integration.py` needs a `robomage_nn` (`make actor`); its `server`
and `bench` commands use the GPU (`HSA_OVERRIDE_GFX_VERSION=10.3.0` on the RX 6700 XT) and must be
the only process on it — the engine processes it starts run with the GPU hidden.

Exporting a `.pt2` in a process that sees the GPU initialises it (an inductor pass); in a
CPU-only process hide the GPU first (`HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=`).
