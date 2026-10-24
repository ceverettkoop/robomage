# learn/ — the learner side (PyTorch)

The learner of `docs/rnad_training_plan.md` §4. It needs torch + numpy (`learn/requirements.txt`,
its own venv `learn/.venv`) and is never imported by `python/`, the engine or the gate.

## Layout

| Path | What |
|---|---|
| `rmlearn/rmtb.py` | RMTB files (`docs/featurizer.md` §4.4) in numpy: `read`, `read_bytes`, `write`, `write_bytes` |
| `rmlearn/spec.py` | `load_spec` (FEATURE_SPEC + its hash), `load_tables` (card / text / ability / part tables, hashes checked; the model manifest `hashes`, the lineage), `check_compatible` (a model's manifest against the tables': `docs/featurizer.md` §6.2, the C++ `featurize::check_compatible`'s rule) |
| `rmlearn/batch.py` | `Decisions` (a tensors file or an RMIP request: unpadded, concatenated), `pad` (= `featurize::pad_batch`), `to_torch`, `Targets` (an `rmfeat --targets` file) |
| `rmlearn/model.py` | `ModelConfig`, `RMNet` (policy, value, auxiliary heads, optional omniscient critic), `InferencePolicy` (the exported contract), `aux_losses`, `masked_log_softmax`, `smoothed_log_policy`, `save_checkpoint` / `load_checkpoint` |
| `rmlearn/export.py` | `export_policy` (TorchScript `.ts.pt`, AOTInductor `.pt2`, optional ONNX, with the tables' model manifest and the policy version as metadata strings), `read_metadata`, `load`, `export_refusal` (an export is of one vocab and tables: rebuild it otherwise) |
| `rmlearn/episodes.py` | trajectory episodes (`docs/trajectory_format.md`) in numpy: `read_episode` (the file's bytes kept, arrays as views), `expand` (steps of many episodes → the contract's padded inputs, = `batch.pad`), `targets`, `evaluations`, `episode_attrs` |
| `rmlearn/replay.py` | `ReplayWindow` (RAM window over trajectory directories: polling ingestion, eviction by decisions / bytes, optional deletion of evicted `.rmtraj`, never `.rmlog`; labels; held-out split by game), `DecisionBatch` (BC), `TrajectoryBatch` (R-NaD: full games time-major, chunked), `Prefetcher` (background batch threads) |
| `rmlearn/bc.py` | behaviour cloning: `BCConfig` (incl. `sideboard_weight`, the policy loss's weight of sideboarding decisions), `BCTrainer` (`train(n)`, `evaluate()`, `save`, `restore`, `export` → metric dicts), `python -m rmlearn.bc` |
| `rmlearn/corpus.py` | the teacher corpus (plan B0): scripted:hard games through engine actors (or `rm_driver.py` + `--replay`) into episodes; `make_schedule`, `run_actor_corpus`; `refeaturize` (`--refeaturize RMLOG_DIR`: the kept logs replayed into episodes after a spec change) |
| `rmlearn/evaluate.py` | an export as `nn:<model>` against `scripted` / `explore` / `random` through engine actors, scored per game (game 1 apart from games 2–3): `run_eval`, `summarize` |
| `rmlearn/matches.py` | evaluation matches through engine actors: `eval_schedule` (a policy vs an opponent over deck pairs, `swap: sides`), `ActorPool` (N actors, their results lines as games finish, SIGTERM stop), `game_record`, per-game `tally` / `report` (Wilson interval, match-clustered SE, game 1 apart from games 2–3, per matchup / deck / seat / play-draw, draws and caps as findings) |
| `rmlearn/sprt.py` | the SPRT match runner (plan T8): `SPRTConfig`, `run_sprt(policy, opponent, cfg, out_dir)` → result dict (`sprt.json`, `summary.txt`), `python -m rmlearn.sprt` |
| `rmlearn/benchmark.py` | the end-of-run benchmark (`rmtrain bench`, docs/rmtrain.md §7.1): a run's policies (default `bc-final`, every `m<k>`, `final` at argmax and projected) against scripted:hard over every league pair with common random numbers; `summary.md`: overall, per deck playing and facing, the difference from the baseline, the deck matrices |
| `rmlearn/ladder.py` | past fixed points: `Ladder(dir)` (`ladder.json`), `add_fixed_point`, `evaluate_against_ladder`, `ratings` (Bradley–Terry, Elo scale), `python -m rmlearn.ladder` |
| `rmlearn/exploiter.py` | a PPO best response against a frozen checkpoint (exploiter and victim served by one `InferenceServer`): `ExploiterConfig`, `run_exploiter(victim, init, cfg, out_dir, tables)`, `ExploiterSession` (driven by a host's loop: rmtrain's GPU process), `python -m rmlearn.exploiter` |
| `rmlearn/server.py` | `InferenceServer`: RMIP v2 (`docs/inference_protocol.md`) inside the host process; `python -m rmlearn.server` runs it standalone |
| `rmlearn/client.py` | a Python RMIP client (checks, benches) |
| `rmlearn/rnad.py` | R-NaD's math, a port of OpenSpiel's `rnad.py` (8156646): `v_trace`, `get_loss_v`, `get_loss_nerd`, `EntropySchedule`, `FineTuning` (= `src/agents/policy_projection.cpp`) |
| `rmlearn/rnad_learner.py` | `RNaDConfig`, `trainable_episodes`, `RNaDLearner` (on `replay.TrajectoryBatch`; (`step`, the iteration boundary, `state_dict` / `from_state_dict`, `from_bc_checkpoint`, `configure`), `ServerLink` (pushes to `InferenceServer`) |
| `rmtrain`, `rmlearn/rmtrain.py`, `rmlearn/orchestrator/` | the training program (`docs/rmtrain.md`): config, run directory, control socket, actor pool, the GPU process (learner + server), evaluation registry, exporter, the phases B0–B5, `retable` (the run's tables after a vocab expansion) |
| `check.py` | the CPU-only correctness checks |
| `check_rnad.py` | the R-NaD learner's CPU-only checks (a numpy oracle of the reference, the engine's projection, synthetic games, resume) |
| `check_rmtrain.py` | rmtrain's CPU checks, and `--smoke`: a tiny end-to-end run (GPU, engine) |
| `bench_rnad.py` | the learner step's throughput and memory |
| `integration.py` | checks and benches against the C++ actor (`robomage_nn`): `actor`, `server`, `bench` |
| `open_vocab_baseline.py` | the open-vocabulary refactor's baseline fixtures (`docs/open_vocab_plan.md` S0): every runnable `runs/**/ckpt/` checkpoint's logits, values and critic values on a fixed decision set (`tests/open_vocab/`: sources, `baseline/`, `manifest.json`), also with the `lists` / `b_lists` tokens removed; `generate` (CPU, reproduces bit for bit), `select` (the decision sources) |

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

from rmlearn import replay, bc
win = replay.ReplayWindow(replay.WindowConfig(dirs=["runs/r1/episodes"], heldout_fraction=0.05))
win.poll()                                             # or win.start(): a polling thread
trainer = bc.BCTrainer(net, win, bc.BCConfig(batch_size=64), device="cuda")
metrics = trainer.train(250); metrics.update(trainer.evaluate())
tb = replay.TrajectoryBatch(win.take_fresh(64), chunk_len=64, policies=(1, 2))   # R-NaD's batches
```

```sh
cd learn
.venv/bin/python -m rmlearn.corpus --out ../runs/bc1/corpus --actors 16 --matches 2200     # B0
HSA_OVERRIDE_GFX_VERSION=10.3.0 flock /tmp/rm-gpu.lock .venv/bin/python -m rmlearn.bc --tables T \
    --episodes ../runs/bc1/corpus/episodes --out ../runs/bc1/bc --minutes 25                   # B1
.venv/bin/python -m rmlearn.evaluate --model ../runs/bc1/bc/ckpt/p1.pt2 --out ../runs/bc1/eval
.venv/bin/python -m rmlearn.sprt --policy ../runs/bc1/bc/ckpt/p1.pt2 --opponent scripted --out ../runs/bc1/sprt  # T8
.venv/bin/python -m rmlearn.ladder --dir ../runs/r1/ladder add --id bc --model ../runs/bc1/bc/ckpt/p1.pt2
.venv/bin/python -m rmlearn.ladder --dir ../runs/r1/ladder eval --id it4 --model P.pt2 --games 100 --add
HSA_OVERRIDE_GFX_VERSION=10.3.0 flock /tmp/rm-gpu.lock .venv/bin/python -m rmlearn.exploiter \
    --victim ../runs/bc1/bc/ckpt/bc.pt --tables ../runs/bc1/tables --out ../runs/x1 --minutes 20
```

## Checks

```sh
learn/.venv/bin/python learn/check.py --tables DIR --tensors A.rmtb [--omniscient O.rmtb --targets T.rmtb]
learn/.venv/bin/python learn/check.py --engine-dir bin/debug --bin-dir bin    # makes the inputs first
```

With `--engine-dir` the check also plays two short bo3 matches (explore vs random, the scripted
teacher on seat A, omniscient episodes) for the episode reader (against `python/rmtraj.py`), the
replay window, both batch kinds and a few BC steps (`--episodes DIR` gives them a directory instead).

```sh
learn/.venv/bin/python learn/check_rnad.py [--tables DIR --episodes DIR [--omni-episodes DIR]]   # RNAD-CHECK: ok
HSA_OVERRIDE_GFX_VERSION=10.3.0 flock /tmp/rm-gpu.lock learn/.venv/bin/python learn/bench_rnad.py \
    --tables DIR --episodes DIR --games 32 --chunk 8 --precision fp32,fp16 --recorded-regs
```

`check.py` hides the GPU from itself and needs no engine when given its inputs; it ends with
`RMLEARN-CHECK: ok` (exit 0). `integration.py` needs a `robomage_nn` (`make actor`); its `server`
and `bench` commands use the GPU (`HSA_OVERRIDE_GFX_VERSION=10.3.0` on the RX 6700 XT) and must be
the only process on it — the engine processes it starts run with the GPU hidden.

Exporting a `.pt2` in a process that sees the GPU initialises it (an inductor pass); in a
CPU-only process hide the GPU first (`HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES=`).
