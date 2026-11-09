# Plan: league48h, 48 hours of R-NaD over the whole league, without history

Goal: one 48-hour R-NaD run over the 26 league decks, carrying on from the history-off paired run
(`runs/hist-off`, 5,002 learner steps from `m11`), with the recent-event history off. The four held-out
decks stay out of training and measure generalisation.

History is off: the paired run found no gain from it (difference in differences +0.002 ± 0.008) at ~20%
more wall time per step. New decks enter through self-play from the current policy, with no teacher, no BC
and no deck-specific logic.

Config: `runs/league48h.toml` (validated with `rmlearn.orchestrator.config.validate`). Launching it needs the
user's go-ahead.

## 1. Starting point

`rmtrain start` can't resume `runs/hist-off` itself: a finished run (phase `done`) is refused, and
`run.feature_history` can't change on resume anyway. So the new run starts from hist-off's final network
through `run.start_from`, as hist-off started from `m11`: B3 seeds R-NaD from it (π_0,reg is that network
smoothed by `rnad.reg_eps`), then B4.

`run.start_from` takes a model checkpoint (`model.save_checkpoint`). `ckpt/rnad-final.pt` is a learner
state (`{"learner", "window"}`) and the fixed points stop at `m5` (step 4,500), so save the final network
first. This was checked in a scratch directory: all 86 tensors round-trip through `load_checkpoint`.

```sh
cd learn && .venv/bin/python - <<'EOF'
from rmlearn import model as M, spec as S
from rmlearn.orchestrator.exporter import load_model
t = S.load_tables("../runs/hist-off/tables")
M.save_checkpoint(load_model("../runs/hist-off/ckpt/rnad-final.pt", t), "../runs/hist-off/final-model.pt",
                  {"version": 5002, "learner_steps": 5002, "source": "runs/hist-off/ckpt/rnad-final.pt"})
EOF
```

The file goes beside `ckpt/`, not inside it, so a later `retable` of hist-off doesn't count it as one of
that run's checkpoints.

What carries over and what doesn't:

- **Carried over:** the network weights (online network at step 5,002, the policy benchmarked as `off-final`)
  and its seen-card mask.
- **Not carried over:** Adam's moments, the target network (it starts equal to the network), the
  regularisation policies (they restart from the network, smoothed), the entropy schedule's position
  (step 0), and the replay window (it starts empty and fills from fresh self-play). R-NaD learns from its
  current policy, so the empty window costs only the first few minutes.

The engine hasn't changed since hist-off ran (last commit `28bcc3e3`, 2026-11-07), so the run's tables are
hist-off's. The start check (`check_start`) confirms it.

## 2. The league

All 26 decks of `decks/league/` start at once, 325 pairs:

- **The original ten** (hist-off's league): `bug`, `burn`, `reanimator`, `uw_control`, `wrb_energy`,
  `wubg_doomsday`, `ur_tempo`, `lands_top8`, `death_and_taxes`, `maverick`.
- **The other 16:** `artifacts_blue`, `artifacts_prison`, `boros_aggro`, `cloudpost_tron_ramp`,
  `cradle_control`, `dimir_tempo`, `doomsday`, `eldrazi_aggro`, `elves`, `hogaak`, `mystic_forge`, `painter`,
  `reanimator_top8`, `stoneblade`, `uwx_control`, `welder_cam`.
- **Held out** (never trained on; the config refuses them): `bant_control`, `dragon_stompy`, `show_and_tell`,
  `storm`.

Every pair weighs 1, except combo against combo, which weighs 0.5 (15 pairs), as in pilotA. The combo
decks are `reanimator`, `reanimator_top8`, `wubg_doomsday`, `doomsday`, `painter` and `elves`. That list is
a judgement call and `league.pairs` is live, so it can be revised mid-run. Each deck appears in 25 of the
325 pairs. The pairs among the original ten are 45 of 325 (14% of matches), so most of the training goes to
the 16 decks hist-off never played.

Of the 16 new decks, `eldrazi_aggro` and the other fifteen were in the hist-pair bench but not in
training. Their cards are on the unknown-card path only until the first batch that holds them, a few steps
into the run.

## 3. Settings

The same as hist-off unless the table says otherwise. hist-off's settings gave +0.020 ± 0.006 over `m11`
in 5k steps.

| key | value | why |
|---|---|---|
| `run.seed` | 5 | new seed (hist-off used 3) |
| `run.feature_history` | 0 | history off (decided) |
| `rnad.learning_rate` | 2e-4 | as hist-off and league24h-sb |
| `rnad.entropy_schedule_sizes` / `repeats` | [500, 1500, 2000] / [1, 1, 1] | fixed points at 500, 2,000, then every 2,000: `m11` at 20,000. The first boundary comes early, because a fixed point reads cards it hadn't seen when frozen on the unknown-card path until the next boundary. The later iterations are longer than hist-off's 1,500 because 325 pairs need more games per iteration |
| `rnad.reg_eps`, `reg_eps_sideboarding` | 0.02, 0.0 | as hist-off |
| `b4.max_learner_steps`, `b5.max_learner_steps` | 20,000 (re-set at hour 3) | the run ends in B4 with no fine-tuning, as hist-off did |
| `sb_explore` | 0.25 → 0.05, hold 0.33 | as hist-off. The hold and the slope scale with `b4.max_learner_steps`, so moving the cap moves them too |
| `model.id_dropout` | 0.1 | unchanged. 0.25 is a later experiment, judged on the held-out decks, not this run |
| `eval.pairs` | the default four, plus `elves`–`painter`, `doomsday`–`stoneblade`, `hogaak`–`cradle_control`, `mystic_forge`–`boros_aggro` | the run's own evaluations also follow new decks |
| `eval.every_minutes` | 90 (hist-off: 60) | `per_iteration` already adds an evaluation every 2,000 steps |
| actors | 26, `local_count` 16, local TorchScript | as hist-off |

**Budget.** hist-off took 7.2 s per learner step, evaluations included, at 134 decisions a game. In the
30-deck hist-pair bench, games averaged 172 decisions, about 28% longer. If step time scales with that, a
step takes about 9 s and 47 hours give about 19k steps. The config caps the run at 20,000 steps. At hour 3,
read the step rate from `rmtrain status` and set the cap so the run ends at about hour 47:

```sh
learn/rmtrain set league48h b4.max_learner_steps=N b5.max_learner_steps=N    # N = steps/h × 47, rounded to 2,000
```

Keep N a multiple of 2,000 so the run ends on a fixed point. Before hour 44, don't stop the run with
`rmtrain stop`: stopping mid-B4 leaves no `final` export. Change the cap instead.

## 4. Timeline

| hour | what |
|---|---|
| −2 to 0 | the cleanup in §6. `make BUILD=RELEASE && make BUILD=RELEASE actor PYTHON=learn/.venv/bin/python` (rmtrain runs `bin/release/robomage_nn`). The start checkpoint from §1. No gate run is needed: the engine hasn't changed since M13's green gate. If it is rebuilt from changed sources, run `make check` first |
| 0 | `learn/rmtrain start league48h --config runs/league48h.toml`. One GPU process at a time: nothing else on the GPU |
| 0–3 | watch `rmtrain status` / `tail`. Check that the seed was written (`rnad-seed.pt`), that 26 actors are healthy, that every deck appears in the results within the first hour, that the window fills and steps start, and the caps and draws (findings). Note s/step and decisions per game |
| 3 | set the step cap from the measured rate (§3) |
| 12 | review 1 (§5) |
| 24 | review 2, with a mid-run bench: `rmtrain pause`, bench the latest fixed point (§5, about 16 min), `rmtrain resume` |
| 36 | review 3 |
| ~47 | the run reaches its cap. The phase goes to `done` and `final` is exported |
| 47–49 | the end-of-run bench (§5) |

## 5. Measuring it

**The paired baseline already exists.** The `hist-pair` bench (`runs/hist-on/bench/hist-pair/`: 30 decks,
4,800 matches, seed 11, 16 actors, against `scripted:hard`) played `off-final`, which is this run's
starting network, and `m11`. Each of the run's benches uses the same `--decks`, `--matches`, `--seed` and
`--actors`, and starts with copies of those two policies' result directories, so every policy is paired
with the start and with `m11` by common random numbers, without playing them again. A policy takes about
16 minutes.

```sh
B=runs/league48h/bench/end; mkdir -p $B
cp -r runs/hist-on/bench/hist-pair/off-final runs/hist-on/bench/hist-pair/m11 $B/
learn/rmtrain bench league48h --wait --label end --decks league,heldout --matches 4800 --actors 16 --seed 11 \
    --policies off-final=runs/hist-off/exports/final.ts.pt,m11=runs/league24h-sb/exports/m11.ts.pt,m6,final
```

The first policy listed is the summary's baseline (`benchmark.summarize`), so `off-final` goes first. The
mid-run bench (`--label mid`, hour 24) is the same command with the latest fixed point in place of `m6,final`.

**What decides it**, all paired against `off-final` (and against `m11` for continuity with the earlier runs):

- `plays:league` and `within:league`: the trained league as a whole.
- `plays:league:original`: no regression on the decks hist-off trained.
- **The 16 new decks' per-deck rows**, which should gain most. Before the run they range from 0.09
  (`artifacts_prison`, `cloudpost_tron_ramp`) to 0.85 (`boros_aggro`) against `scripted:hard`.
- `plays:heldout` / `faces:heldout`: generalisation. hist-off moved `plays:heldout` by +0.004 ± 0.018 and
  `faces:heldout` by +0.035 ± 0.016.
- `value_mse_gap` and `value_mse_first_sight` from `metrics.jsonl`: memorisation. Watch for a gap that
  grows over the run.

**Reviews** (hours 12, 24, 36):

- `evals.jsonl`: the fixed evaluator's eight pairs, SPRT against `scripted:hard`, and the ladder against
  the previous fixed points and the start network.
- `value_mse_gap`.
- The findings: every draw must be read; CLAUDE.md doesn't accept draws. A draw is either the rules'
  outcome (hist-off had Price of Progress and Archon of Cruelty decking) or an engine bug to report. Also
  the caps (`mystic_forge` and `welder_cam` caused most bench caps).
- Actor restarts, and the sb_explore coverage line.

**Stop early** on a `plays:league:original` regression beyond two SEs with no gain on the new or
held-out decks at the mid-run bench, on a crash loop, or on an engine finding that corrupts games (a
FATAL, not a reviewed draw). Don't stop on noise from one evaluation.

## 6. Training files to clean up

`du` on 2026-11-09, with 933 GB free on `/home`. The run itself needs at most ~40 GB of replay window
(`replay.max_gb`) plus its checkpoints, so none of this deletion is required for the run.

*Removed (2026-11-09, 93 GB):* the two history runs' `episodes`, `open_vocab_s7`'s fuzz and acceptance
scratch, and the S0–S5 and `scythecat_fix` scratch, as listed below. For `open_vocab_s1`–`s3`, the top-level
scripts and logs the docs cite were kept. The rest of the table is still on disk.

**Safe to delete: nothing reads them.**

| path | size | what it is |
|---|---|---|
| `train/` (untracked) | 129 GB | the archived RL stack's data: `az_data` (103 GB of AlphaZero shards), `checkpoints` (7.5 GB), `logs`, `ci_out`, `analysis_out`, generated `card_costs.py` / `card_props.py` / `sb_rules.py`. The obs format is long dead; the code is on `archive/rl-stack` |
| `src/gen/` (untracked) | 156 KB | `card_costs_gen.h`, `sb_rules_gen.h`: the old stack's codegen output. Nothing includes them |
| `runs/hist-off/episodes`, `runs/hist-on/episodes` | 17 + 21 GB | the finished paired runs' replay windows. Neither run can be resumed, and league48h starts with an empty window. hist-on's are of the history spec |
| `runs/open_vocab_s7/onboard/fuzz`, `fuzz_prefix` | 12 GB | M-stage onboarding fuzz transcripts. Keep `onboard/*.log`, `campaign.sh`, `scripted*/` (the onboarding record) |
| `runs/open_vocab_s7/subtypes/{inputs_work,inputs_work2,acceptance}` | 9.6 GB | acceptance scratch. Keep `agents_{before,after}` (cited) |
| `runs/open_vocab_s5/{acceptance,acceptance_pre,inputs_work,rehearsal,try,gate_try}` | 12.8 GB | S5 acceptance work directories. `acceptance` is `open_vocab_acceptance.py full`'s default `--work` and is rebuilt when run |
| `runs/open_vocab_s0/{decisions,feat,tensors,dumps}` | 6.5 GB | S0 fixture generation scratch. The fixtures are committed in `tests/open_vocab/`. **Keep `open_vocab_s0/ckpt*`** (`m11-critic.pt`, read by the gate) |
| `runs/open_vocab_s1`, `runs/open_vocab_s2` | 2.2 + 6.9 GB | S1/S2 fixture checks and scratch |
| `runs/open_vocab_s3` except `expansion_work/appended` | ~3 GB | S3 scratch. `expansion_work/appended/tables` is named in `open_vocab_acceptance.py`'s docstring |
| `runs/open_vocab_s4/{smoke,check_work,check_work2}` | 1.2 GB | S4 scratch. **Keep `card_ids_*.json`** (`open_vocab_acceptance.py` `SEEN`) |
| `runs/scythecat_fix` | 2.9 GB | acceptance inputs for one fix |
| `ci_out/` | 5.1 GB | gate output, rewritten by every `make check` |
| `*/exports.v1`, `*/exports.layout1` (league12h, league24h-sb, val1, pilotA, pilot-p1/p2, ov-verify) | 4.0 GB | exports in old layouts. The current engine can't load them, and `retable` rebuilt current ones |
| `runs/t8`, `runs/pilot-chain.*`, `runs/agent_logs`, `runs/vocab_fix_*` | ~180 MB | old exploiter experiments, pilot chain script, empty dir, build/gate logs |
| `runs/pilotB-prep` | 12 KB | its input (B0 episodes) was deleted on 2026-11-06 |

Together: about **235 GB**, 129 GB of it `train/`.

**Probably safe, but ask first.** `*/ckpt.layout1` (5.3 GB): the intermediate copies from the layout-1 →
layout-2 migration. The current `ckpt/` files are migrated, and the acceptance reads the originals from
`ckpt.v1/`. Deleting these loses only the option of re-migrating from layout 1. `migration.json`'s
`previous` names them.

**Keep.**

- `runs/league24h-sb` (`m11`, the gate's checkpoints).
- Every `ckpt/` and `ckpt.v1/` of league12h, league24h-sb, val1, pilotA, pilot-p1/p2, bc1 and open_vocab_s0:
  the S0 fixture manifest's originals for `learn/open_vocab_acceptance.py full`, not yet run over the
  cards added since M7.
- `runs/hist-off` except `episodes`: the start checkpoint, its exports and `final-model.pt`.
- `runs/hist-on/bench/hist-pair`: the paired baseline above.
- `runs/hist-on`'s `ckpt` and `exports`, for any later history comparison.
- Every run's `rmlog/` and `bench/`.
- `learn/.venv` (13 GB).
