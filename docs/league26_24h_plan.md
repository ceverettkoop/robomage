# Plan: league26-24h, 24 hours of R-NaD over the whole league, without history

Goal: one 24-hour R-NaD run over the 26 league decks, carrying on from the history-off paired run's latest
checkpoint (`runs/hist-off/ckpt/rnad-final.pt`, learner step 5,002), with the recent-event history off. The
four held-out decks stay out of training and measure generalisation.

This is `docs/league48h_plan.md`'s run at half the budget, with the changes of §3. History off and the
starting checkpoint are decided (2026-11-14). New decks enter through self-play from the current policy,
with no teacher, no BC and no deck-specific logic.

Config: `runs/league26-24h.toml` (validated with `rmlearn.orchestrator.config.validate`). Launched on
2026-11-14 at 10:25 PST (`runs/league26-24h`), on the user's go-ahead once the smoke of §7 had passed: hour 2
is 12:25, hour 12 is 22:25, and the cap is due at about 09:50 on 2026-11-15.

## 1. Starting point

The run starts from hist-off's final network through `run.start_from`: B3 seeds R-NaD from it (π_0,reg is
that network smoothed by `rnad.reg_eps`), then B4. A finished run can't be resumed, and `rnad-final.pt` is a
learner state, not the model checkpoint `run.start_from` takes, so the network is saved on its own as
`runs/hist-off/final-model.pt` (written 2026-11-14 with league48h's §1 recipe; its 86 tensors equal the
learner state's network, 241 cards seen).

- **Carried over:** the network weights (the policy benchmarked as `off-final`) and its seen-card mask.
- **Not carried over:** Adam's moments, the target network, the regularisation policies (they restart from
  the network, smoothed), the entropy schedule's position (step 0) and the replay window (it fills from
  fresh self-play in the first minutes).

The engine plays as it did for hist-off: the sources since then differ in comments and the version string
(`robomage 0.3`). `bin/release` was rebuilt from HEAD on 2026-11-14 (§7); the start check (`check_start`)
passed against it.

## 2. The league

As league48h's §2: all 26 decks of `decks/league/` from the start, 325 pairs, every pair weighing 1 except
combo against combo at 0.5 (15 pairs; the combo decks are `reanimator`, `reanimator_top8`, `wubg_doomsday`,
`doomsday`, `painter`, `elves`). `league.pairs` is live.

- **Trained by hist-off (ten):** `bug`, `burn`, `reanimator`, `uw_control`, `wrb_energy`, `wubg_doomsday`,
  `ur_tempo`, `lands_top8`, `death_and_taxes`, `maverick`.
- **New to training (16):** `artifacts_blue`, `artifacts_prison`, `boros_aggro`, `cloudpost_tron_ramp`,
  `cradle_control`, `dimir_tempo`, `doomsday`, `eldrazi_aggro`, `elves`, `hogaak`, `mystic_forge`, `painter`,
  `reanimator_top8`, `stoneblade`, `uwx_control`, `welder_cam`.
- **Held out** (the config refuses them): `bant_control`, `dragon_stompy`, `show_and_tell`, `storm`.

## 3. Settings

league48h's (so hist-off's, which gave +0.020 ± 0.006 over `m11` in 5k steps) except:

| key | league48h | here | why |
|---|---|---|---|
| `league.max_game_decisions` | 5000 | 2000 | A capped game is dropped by the learner, so its actor time is lost. In the 30-deck bench the league's capped games were 0.8% of games (69 of 8,388, mostly `mystic_forge` and `welder_cam`) but 22% of decisions and 26% of actor seconds, and the longest decisive game had 1,275 decisions (hist-off's self-play: 921 over 201,834 games). The smoke's self-play had no capped game in 4,086 (§7), so this bounds the cost if they appear, and shortens the in-run evaluations against `scripted:hard`, which play under the same cap. Not a live key. `rmtrain bench` keeps its own 5,000, so the paired baseline of §6 is unaffected |
| `rnad.entropy_schedule_sizes` / `repeats` | [500, 1500, 2000] / [1, 1, 1] | [500, 1000, 1500] / [2, 2, 1] | hist-off's schedule, the one its result was measured with: fixed points at 500, 1,000, 2,000, 3,000, 4,500, 6,000, 7,500, 9,000, 10,500. A 2,000-step iteration is untested and, at about 4.5 hours each, would leave a 24-hour run five fixed points. The first boundary still comes at 500, where the fixed point takes the new cards into its seen mask |
| `b4.max_learner_steps`, `b5.max_learner_steps` | 20,000 | 10,500 (checked at hour 2) | the budget below |
| `eval.every_minutes` | 90 | 120 | `per_iteration` already evaluates eight or nine fixed points in 24 hours |

Unchanged: `run.seed` 5, `run.feature_history` 0, `rnad.learning_rate` 2e-4, `rnad.reg_eps` 0.02 / `reg_eps_sideboarding`
0.0, `sb_explore` 0.25 → 0.05 with hold 0.33 (the hold and the slope scale with `b4.max_learner_steps`),
`model.id_dropout` 0.1, the eight `eval.pairs`, 26 actors with `local_count` 16 on local TorchScript.

**Budget.** hist-off's learner was the bottleneck (it waited for games 1.5% of the time): 6.1 s per step of
32 games until the first boundary, where the learner recomputes one regularisation policy's log π, and 7.15 s
after it, where it recomputes two (7.27 s of wall time with evaluations and boundaries), on games of 134.5
decisions. The smoke of §7 measured the 26-deck league: games of 144.5 decisions and 8.0 s per step past the
first boundary, 8.1 s of wall time, the learner waiting 1% of the time. (league48h's 9 s per step counted the
bench's capped games, which the learner never sees.) At about 6.8 s for the first 500 steps and 8.1 s after,
10,500 steps end at hour 23.4, so the config caps the run there, at the fixed point `m9`.

At hour 2 (step 500 comes at about hour 1), check that estimate against the second hour's rate and move the
cap if the run would end after hour 23.4:

```sh
# N = the steps at hour 2 + steps/h over the second hour × 21.4, rounded down to 500; 10,500 when within 300 of it
learn/rmtrain set league26-24h b4.max_learner_steps=N b5.max_learner_steps=N
```

N need not be a fixed point's step: hist-off ended at 5,000, 500 steps into an iteration, and its `final` is
the policy every later comparison uses. The bench plays the last fixed point and `final`. Change the cap
rather than stopping: `rmtrain stop` mid-B4 leaves no `final` export.

**What 24 hours buys.** 10,500 steps are 336k games, about 26k for each deck. hist-off gave each of its ten
decks 32k and gained +0.035 ± 0.010 playing them (and +0.015 ± 0.008 on the 16 decks it never played, by
transfer alone). So on the ten trained decks a gain inside the bench's noise is the likely outcome; the
question this run answers is whether the 16 new decks move clearly more than transfer moved them.

## 4. Timeline

| hour | what |
|---|---|
| before | done on 2026-11-14 (§7): the release build, the start checkpoint, the config validated, a 120-step smoke. If the sources change before the launch, rebuild (`make BUILD=RELEASE && make BUILD=RELEASE actor PYTHON=learn/.venv/bin/python`) after a green `make check` |
| 0 | `learn/rmtrain start league26-24h --config runs/league26-24h.toml`. One GPU process at a time: nothing else on the GPU |
| 0–2 | the checks below |
| 2 | check the step cap against the measured rate (§3) |
| 6 | review 1 (§6) |
| 12 | review 2, with the mid-run bench: `rmtrain pause`, bench the latest fixed point (about 16 min), `rmtrain resume` |
| 18 | review 3 |
| ~23.4 | the run reaches its cap. The phase goes to `done` and `final` is exported |
| to 24 | the end-of-run bench (§6), about 35 min for two policies |

**Hours 0–2**, from `rmtrain status` / `tail` and the `rnad_step` lines of `metrics.jsonl`:

| check | expect | if not |
|---|---|---|
| `ckpt/rnad-seed.pt` written, 16 actors running, the live export in place | within a minute | read `logs/orchestrator.log` |
| every deck in the actors' results | within the first half hour | check `schedule.json` |
| `wait_fraction` (the learner waiting for games) | under 0.05 once the window has filled | above 0.10: `rmtrain set league26-24h actors.local_count=20` (live) |
| capped games | none in the smoke's 4,086 games | more than 1% of games: a finding to read, as a draw is; the decks are in the results lines |
| `skipped_oversize` against `ingested` in the step line's `window` | hist-off: 5 of 201,788. The new decks go wide (`elves`, `hogaak`, `cradle_control`), and an episode with one decision over `replay.max_decision_tokens` (640) is left out whole | above 0.5%: raise `replay.max_decision_tokens` (live) to 800 and watch the GPU's peak memory |
| GPU memory | the smoke's peak: 6.3 GB of 12 | above 11 GB: lower `rnad.forward_tokens` |
| s per step, decisions per game | 8.0 s past step 500, 144.5 | they move the cap at hour 2 |
| draws | each one read (CLAUDE.md accepts none unreviewed) | the rules' outcome (hist-off: Price of Progress, Archon of Cruelty decking), or an engine bug to report |

## 5. Risks

- **The budget is half of league48h's** for the same 26 decks: see "What 24 hours buys". A null result on the
  trained ten is not a regression signal by itself.
- **Forgetting on the ten trained decks.** Their own pairs are 45 of 325 (14% of matches). The mid-run bench
  checks it.
- **Caps in self-play.** The bench's 0.8% is argmax against `scripted:hard`; the smoke's self-play had none.
  A policy that learns a loop could bring them; the hour 0–2 check and the reviews read the count.
- **Fixed points and new cards.** A fixed point reads a card it hadn't seen when frozen on the unknown-card
  path until the next boundary. π_0,reg has seen none of the new decks' cards, so the first 500 steps
  regularise the new decks towards a policy reading them as unknown.

## 6. Measuring it

**The paired baseline exists.** The `hist-pair` bench (`runs/hist-on/bench/hist-pair/`: 30 decks, 4,800
matches, seed 11, 16 actors, against `scripted:hard`) played `off-final`, this run's starting network, and
`m11`. Each bench here uses the same `--decks`, `--matches`, `--seed` and `--actors` and starts with copies
of those two policies' result directories, so every policy is paired with the start and with `m11` by common
random numbers without playing them again. A policy takes about 16 minutes.

```sh
B=runs/league26-24h/bench/end; mkdir -p $B
cp -r runs/hist-on/bench/hist-pair/off-final runs/hist-on/bench/hist-pair/m11 $B/
learn/rmtrain bench league26-24h --wait --label end --decks league,heldout --matches 4800 --actors 16 --seed 11 \
    --policies off-final=runs/hist-off/exports/final.ts.pt,m11=runs/league24h-sb/exports/m11.ts.pt,m9,final
```

The first policy listed is the summary's baseline, so `off-final` goes first. `m9` is the fixed point at
10,500 steps; name the run's last one if the cap moved (`m8` at 9,000). The mid-run bench (`--label mid`,
hour 12, `bench/mid`) is the same command with the latest fixed point alone in place of `m9,final`.

**The groups that decide it.** The summary's `league:original` is `config.py`'s eleven decks, which hold
`eldrazi_aggro`, a deck hist-off never played, and it has no group for the new decks. Both groups, paired
against `off-final`, from the bench's result directories:

```sh
cd learn && .venv/bin/python - <<'EOF'
from rmlearn import benchmark as B
D, POLICY = "../runs/league26-24h/bench/end", "final"
TEN = ["bug", "burn", "reanimator", "uw_control", "wrb_energy", "wubg_doomsday", "ur_tempo", "lands_top8",
       "death_and_taxes", "maverick"]
NEW = ["artifacts_blue", "artifacts_prison", "boros_aggro", "cloudpost_tron_ramp", "cradle_control", "dimir_tempo",
       "doomsday", "eldrazi_aggro", "elves", "hogaak", "mystic_forge", "painter", "reanimator_top8", "stoneblade",
       "uwx_control", "welder_cam"]
base, other = B.match_records(D + "/off-final"), B.match_records(D + "/" + POLICY)
for name, group in (("ten", TEN), ("new", NEW)):
    for kind in ("plays", "faces", "within"):
        print(name, kind, B._delta(B.paired_delta(base, other, keep=B._group_keep(kind, group))))
EOF
```

Run over `m11` → `off-final` it gives hist-off's own figures: the ten +0.035 ± 0.010 playing, +0.023 ± 0.011
facing; the new 16 +0.015 ± 0.008 playing, +0.014 ± 0.008 facing.

**What decides it**, all paired against `off-final`:

- **`new plays`**: the run's purpose. It should clear the +0.015 that transfer alone gave. The per-deck rows
  say which decks carry it; before the run they range from 0.09 (`artifacts_prison`, `cloudpost_tron_ramp`)
  to 0.85 (`boros_aggro`) against `scripted:hard`, at about ±0.05 a deck.
- **`ten plays`**: no regression beyond two SEs.
- `plays:league`, `within:league`: the league as a whole.
- `plays:heldout` / `faces:heldout`: generalisation. hist-off moved them by +0.004 ± 0.018 and +0.035 ± 0.016.
- `value_mse_gap` and `value_mse_first_sight` from `metrics.jsonl`: memorisation. hist-off's gap stayed
  between 0.03 and 0.06; watch for one that grows over the run.

**Reviews** (hours 6, 12, 18):

- `evals.jsonl`: the fixed evaluator's eight pairs, SPRT against `scripted:hard`, the ladder against the
  previous fixed points. An evaluation queued at the run's last step is cancelled by the stop; the end bench
  covers it.
- `value_mse_gap`, `wait_fraction`, `skipped_oversize`, the policy lag (hist-off: p95 6 steps).
- The findings: every draw, and the caps.
- Actor restarts, and the sb_explore coverage line.

**Stop early** on a `ten plays` regression beyond two SEs with no gain on the new or held-out decks at the
mid-run bench, on a crash loop, or on an engine finding that corrupts games (a FATAL, not a reviewed draw).
Don't stop on noise from one evaluation.

## 7. Checked before the launch (2026-11-14)

- **The release build.** `make BUILD=RELEASE && make BUILD=RELEASE actor PYTHON=learn/.venv/bin/python` from
  HEAD (`0d874c75`): `bin/release/robomage`, `robomage_nn`, `rmfeat`, `rmknow`, `rmserve`, `robomage_check`,
  all `robomage 0.3`. `run.build` is `release`, so the run's actors, evaluations and `rmtrain bench` play
  `bin/release/robomage_nn`.
- **The start checkpoint.** `runs/hist-off/final-model.pt`, read back equal to the learner state's network.
- **The config.** `runs/league26-24h.toml` validates: 26 decks, 325 pairs (weight 317.5).
- **A smoke run** of the config with the schedule shrunk (`rnad.entropy_schedule_sizes=[20,40,60]`, cap 120;
  `runs/league26-24h-smoke`, deleted after): the start check passed against the rebuilt engine, B0 → B3 → B4
  from the checkpoint, 16 actors with no restart, 120 learner steps in 17 minutes, the fixed points `m1`–`m4`
  written and exported (`m4` at the cap, as `m9` will be), the evaluations `it1` and `it1-sprt` played on the
  eight pairs (SPRT H1 against `scripted:hard`), B5 → `done`, `final` exported as TorchScript and `.pt2`, exit
  0, no error in the orchestrator, GPU or export logs. Its self-play: 4,086 games, every deck in at least 260,
  144.5 decisions a game (p99 434, the longest 825), no cap, no draw, no episode over
  `replay.max_decision_tokens`; 8.0 s per step past the first boundary, `wait_fraction` 0.01, policy lag p95
  7 steps, GPU peak 6.3 GB.

Disk: 985 GB free on `/home`; the run needs at most ~40 GB of replay window (`replay.max_gb`; hist-off's
full window was 17 GB) plus its checkpoints and about 1 GB of decision logs. league48h's §6 lists what can
be deleted; none of it is required.
