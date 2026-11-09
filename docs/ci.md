# Continuous integration & the test standard

The standard for changes to this repo: **`make check` must pass.** CI runs exactly it; run it
locally before pushing.

```bash
make check    # `make` (debug build), then bin/debug/robomage_check (every tier)
```

It needs the Forge card scripts provisioned first (the SessionStart hook does this in cloud
sessions only; on a fresh local clone run it yourself; stdlib only):

```bash
python3 tools/forge_fetch/provision_decks.py
```

It fetches every card in `src/card_vocab.h`, every card of the top-level, `league/`, `heldout/` and
`candidates/` decks, and the tokens they create, at the Forge commit pinned in
`tools/forge_fetch/FORGE_PIN`.

## The gate: `robomage_check`

`tools/robomage_check/` builds `bin/<config>/robomage_check` (part of `make`). It runs the engine
as subprocesses in parallel, with `bin/` as their working directory, and links no engine
translation unit (only the std-only decoder, the agents' frame reader and `src/featurize/`). It is
built `-fno-exceptions`; a scenario regex is compiled in a forked child first, so a malformed one
is reported, not an abort.

Every requested tier runs even if an earlier one fails. The run ends with each tier's PASS/FAIL, a
wall-time table, every warning and error, and a `Reproduce:` line (tiers, seed, mode, matchups).
Exit 1 on any **error** (warnings alone pass); exit 2 on bad usage, an unknown tier, or a missing
engine binary or card scripts.

```
robomage_check [--tier a,b,...] [-j N] [--seed N] [--binary PATH] [--out-dir DIR]
               [--fuzz-games N] [--manual-mana-games N] [--smoke-games N] [--matchups PRESET]
               [--mode MODE] [--scenario GLOB ...] [--show output|narrative|dump] [--record]
```

| Flag | Meaning |
|---|---|
| `--tier a,b` | run these tiers, in the order listed (default: all, in the table's order) |
| `-j N` / `--jobs N` | parallel engine processes (default: the cores) |
| `--seed N` | base seed of the `fuzz` and `smoke` games (default 1) |
| `--binary PATH` | the engine (default `robomage` beside the gate, so `bin/release/robomage_check` tests the release build); `rmknow`, `rmfeat` and `robomage_nn` are taken from the same directory |
| `--out-dir DIR` | transcripts, engine stderr and draw logs (default `ci_out/`) |
| `--fuzz-games N` / `--smoke-games N` | games per matchup (default 8 / 2) |
| `--manual-mana-games N` | `fuzz` tier: `--no-auto-mana` games per ring matchup (default 2; 0 skips them) |
| `--matchups PRESET` | over the playable decks (`league/` then `heldout/`): `mirrors`, `ring` (each deck vs the next), `mirrors+ring`, `all` (every ordered pair); or `"a:b,c:d"` (deck specs relative to `decks/`; `a:*` is `a` against every playable deck); default `fuzz` `mirrors`, `smoke` `mirrors+ring` |
| `--mode MODE` | the fuzz agent (bo1 and `--no-auto-mana` games): `explore` (default) or `explore:patient` |
| `--scenario GLOB` | `scenarios` tier: only the scenarios whose name matches (repeatable) |
| `--show output\|narrative\|dump` | `scenarios` tier: print each run's engine output, the narrative lines the assertions see, or its knowledge lines |
| `--record` | `replay` / `cli` / `vocab` tier: rewrite the corpus baselines / CLI goldens / parse-report baseline instead of checking them |

### The tiers

Times are wall clock for the debug build on 4 cores (`make check` total about 6.3 minutes).

| Tier | Checks | Time |
|---|---|---|
| `vocab` | `--check-decks`: every top-level, `league/` and `heldout/` deck card loads and is in the vocab; the WordPiece tokenizer against `tests/embed/tokens_golden.jsonl` (skipped with a notice without the pinned model's `vocab.txt` in `.cache/embed/`); the layout registry `src/featurize/registry.json` canonical, compiled in, needing no `make registry`, and append-only against HEAD unless `layout_version` was raised (`docs/featurizer.md` §6.1); the card digest `src/featurize/card_digest.json` current (§6.2); `--parse-report @vocab` against `tests/parse_report/vocab.json` (a new or vanished finding is an error; `--record` rewrites it); `tools/onboard/onboard.py check` over `decks/candidates/` (a tool failure is an error, a stale status file a warning, and a vocab card from id 367 up that no scenario names in a zone key or `deck_a/b` deck an error) | ~1 s |
| `concede` | the `-2` / `-3` sentinels (CR 104.3a) over RMSTATE sessions the gate answers itself: bo1, bo3 into sideboarding and game 2, between games, at parked queries; same seed and replies give the same frames; results, narrative and `--replay` of every decision log | ~2 s |
| `scenarios` | every `tests/scenarios/*.json` on the engine (presets as flags, `--play`, `--narrative --trace-decisions`), every assertion key of `tools/robomage_check/scenarios.h` | ~4 s |
| `cli` | each `tests/cli/<name>.json` (an engine command line and stdin lines): exit 0, empty stderr, stdout (the `Decision log:` path reduced to `<name>.rmlog`) byte-identical to `<name>.out` (a drift's output goes to `ci_out/cli/<name>.out`); `--replay` of the session's log clean | <1 s |
| `replay` | each `tests/corpus/<name>.rmlog` replayed, stdout byte-identical to `<name>.replay` (below) | ~17 s |
| `state` | the corpus replayed under `--selfcheck-knowledge` (same output, clean summaries); explore fuzz games with self-checks and dumps; every scenario at `--selfcheck-state-sample 1`; `--rmstate-read` on the dumps; decoder refusals of mutated frames; stream sessions against their dumps; name-a-card menus with open and closed decklists; the `CARDS` frame's ability parameters; include isolation of `src/knowledge/` and `src/featurize/` | ~2.2 min |
| `featurize` | the card tables and FEATURE_SPEC from `--cards-frame`, exported, read back and equal to `rmfeat`'s; the corpus and explore games' dumps under `--selfcheck-featurize`; every decision featurized against the spec (shapes, indices), deterministic and row-order invariant; the actor entry point refusing the omniscient frame; learner targets; `rmfeat --featurize` / `--targets` equal to in-process; the recent-event history option (`docs/featurizer.md` §3.7); the registry's append-only property (a card and a keyword appended, an enum reordered: same layout and tensors; capacities refused); `python/pycheck.py --part featurize` and `--part open_vocab` (the fixture sources' decisions hashing to `tests/open_vocab/inputs.json`; drop an intentionally moved one with `learn/open_vocab_acceptance.py inputs --drop-moved`) | ~1.5 min |
| `traj` | the trajectory writer (`docs/trajectory_format.md`) under `ci_out/traj/`: in-process runs (bo1 with the omniscient part, bo3, concessions, a capped seat-A-only run), each `--replay` writing the same episodes; footers, replies and tensors against the decision log and dump; μ a distribution with mass at the reply; `python/rmtraj.py` vs the C++ reader; an `rm_driver.py --rmlog-dir` match re-featurized; the teacher channel (`python/teacher.py`) at mix 1 (scripted:hard's decisions, its stream the `--machine` stream byte for byte), mix 0 (decision log unchanged) and mix 0.5 (the coin, the μ mixture, the replay); broken teachers fatal | ~27 s |
| `actorloop` | the continuous actor (`docs/actor.md`) under `ci_out/actorloop/`: status and results files against the episodes' footers; a two-entry schedule's matches each replayed alone to the same episodes, a rerun with the same seed and id identical; the per-game cap; a taught entry's labels a fresh `teacher.py`'s on each replay; a schedule rewritten mid-run, then SIGTERM (exit 0, `QUIT` in the log, no partial episode); `sb_explore` (μ = `q` from replayed frames, swap counts covering 0–10, card weights moving frequencies); malformed schedules refused | ~10 s |
| `actor` | the policy projection's unit checks always; the rest opt-in (below): the reference policy built (`tools/nn/make_reference_policy.py`, under `ci_out/actor/`, rebuilt only when the spec or script changes); `learn/open_vocab_acceptance.py gate` (skipped without the original checkpoints under `runs/`); `nn:` vs explore bo3 matches (TorchScript and `.pt2` argmax identical, a `temp=0.02:thresh=0.03:disc=16` match's μ checked, a `--replay` re-featurized exactly, the three exports agreeing); `nn:remote:` through `rmserve` and `tools/nn/serve_check.py` (same games, evaluations within 1e-5, handshake refusals, a killed server failing the actor); hot reload under `--actor`; a continuous `--actor` smoke run; `sb_explore` with `nn:` self-play; the recent-event history case (`--feature-history 32`) | ~3 min (+~50 s when the reference policy is rebuilt) |
| `knowledge` | `rmknow --selftest`, `--exactness 40`; explore games under `--selfcheck-knowledge`; stream sessions against their dumps (two under `--closed-decklist`); `rmknow --isolation` and `--truth` on every dump | ~60 s |
| `fuzz` | C++ explore vs explore in process over the matchups: `--fuzz-games` bo1 games each (matchup `k` game `i`: seed `seed + 1000k + i`); one bo3 `explore:patient` match per ring matchup (`seed + 500000 + k`); `--manual-mana-games` bo1 games per ring matchup under `--no-auto-mana` (`seed + 700000 + 1000k + i`; with an explicit `--matchups` list the ring is that list); the fixture decks (below). Transcripts `ci_out/fuzz_<a>__<b>.txt`, `fuzz_bo3_…`, `fuzz_manual_…`, `fuzz_mech_…` | ~50 s |
| `bench` | per-decision wall clock of a fixed workload (explore vs explore, bo1, three league pairings × three games, one engine at a time, process start left out) by stage: engine alone (`--replay`), + frames (knowledge `off`), + knowledge sidecar (`expert`), + featurizer; the stages must play the same games. With the `actor` opt-in, `nn:` in process and `nn:remote:` through `rmserve` (compute vs round trip). An optimised build (`bin/release/`, `bin/pgo/`) whose actor loop is below 1,500 decisions/s is a warning, never an error | ~1.7 min, ~3.5 min with evaluation (debug) |
| `smoke` | scripted hard vs hard through `python/rm_driver.py` (transcripts `ci_out/smoke_<a>__<b>.txt`) and `python/pycheck.py`: the Python decoder against the engine's `--state-dump json`, decoder refusals, the self-lethal guard, the driver's seeds and imports (stdlib only), the sideboard guides against `docs/sideboard_guides.md` and the decks, `scripted:hard+sb` bo3 matches at the guides' targets, `python/rmtb.py` on C++-written RMTB files. Skipped with a notice without `python3` | ~35 s |

**The `actor` tier is opt-in.** It needs `bin/<config>/robomage_nn` and `bin/<config>/rmserve`
beside the engine (`make actor LIBTORCH_DIR=…`) and `$ROBOMAGE_TORCH_PYTHON` naming a Python with
torch, onnx, onnxscript and onnxruntime; without either it prints `skipped: …` (not a warning) and
passes, so plain `make check` needs no torch. The `bench` tier's evaluation stages have the same
opt-in. The learner's checks (`learn/check_rmtrain.py`, its `--smoke` runs) are not part of the gate
and are run by hand (`docs/rmtrain.md` §9).

```bash
ROBOMAGE_TORCH_PYTHON=/path/to/venv/bin/python bin/debug/robomage_check --tier actor,bench
bin/debug/robomage_check --tier scenarios --scenario 'cr702_21a*' --show narrative
bin/debug/robomage_check --tier replay                 # one tier
bin/debug/robomage_check --tier cli --record           # rewrite the CLI goldens after an intended change
bin/debug/robomage_check --tier fuzz --matchups all --mode explore:patient --fuzz-games 20 --seed 7
bin/release/robomage_check --tier bench                # throughput of the release build
```

### Fixture decks (the `fuzz` tier)

A fuzz game passing proves nothing about a path it never reached, so a primitive gets a fixture: a
dense deck that reaches it in most games, and a list of what must have been reached.

- `bin/resources/decks/tests/mech_<primitive>.dk`: vocab cards only. It is outside `league/` and
  `heldout/`, so no matchup preset, training run or onboarding step plays it.
- `mech_<primitive>.expect` beside it: one coverage key per line, `<kind>:<name>` (`cost:SACRIFICE`,
  `cast:flashback`; the kinds and names are `docs/game_running.md`'s `--coverage` table), `#` starting
  a comment.

Every `fuzz` run plays each fixture against itself, `--fuzz-games` bo1 `<mode>` games and at least 16,
under `--coverage` (fixture `f` game `i`: seed `seed + 800000 + 1000 slot(f) + i`, the slot a hash of
the fixture's name), and sums the counts into `ci_out/coverage/mech_<primitive>.txt` in
`--coverage`'s format, so a new `.expect` is written by reading that file. An **error** names each
`.expect` key with no hits; so does a deck `--check-decks` refuses, a missing or empty `.expect`, an
`.expect` with no deck, and a game that wrote no counts. Crashes, error lines and draws report as
for any fuzz game. Only the mirror's counts count.

An explicit `--matchups` list plays only the fixtures it names: `--matchups tests/mech_<primitive>`
for one fixture, `--matchups 'tests/mech_<primitive>:*'` for it plus the deck against every
playable deck in every leg.

```bash
bin/debug/robomage_check --tier fuzz --matchups tests/mech_cost_paths --fuzz-games 100
```

### Throughput reference (bench)

The `bench` tier on the 7950X box, idle, one process at a time: wall clock per game (9 bo1 games,
process start left out; per decision in brackets).

| Stage | release | `make pgo` |
|---|---|---|
| engine alone (`--replay`) | 14.5 ms (138 us) | 11.7 ms (112 us) |
| + frames, knowledge off | +19.0 ms (+182 us) | +16.1 ms (+155 us) |
| + knowledge sidecar (expert) | +13.8 ms (+132 us) | +12.3 ms (+118 us) |
| + featurizer | +2.7 ms (+26 us) | +2.2 ms (+21 us) |
| the actor loop | 49.9 ms, 2,094 decisions/s | 42.4 ms, 2,463 decisions/s |

As one `--actor` process (40 bo3 matches, explore vs explore, five league pairings, knowledge
expert, `--featurize-seats`): 2,116 decisions/s release, 2,355 with `make pgo`.

### Error and warning classification

`tools/robomage_check/scan.h`:

- **Error** (fails the gate): an engine crash (nonzero exit / EOF mid-game), fewer games completed
  than requested, a failed check, any output line starting `ERROR:` / `FATAL:` or holding a Python
  traceback, `Segmentation fault`, a failed assertion, `Aborted` or `core dumped`, or a
  `SELFCHECK-*:` summary without `errors=0`.
- **Warning** (flagged for review, passes): a game with no winner — a draw (CR 104.4a) or a
  **stall** — or an engine `WARNING:` line. A drawn game's transcript is saved as
  `ci_out/<tier>_<a>__<b>_g<i>_draw_<stamp>.txt`.

`WARNING: Unrecognized ability param` lines stay in the transcripts but are not surfaced (they fire
in bulk for cosmetic / AI-hint params). Params the parser drops without warning are `ignored_keys`
in `src/parse.cpp`, and those of `param_form_checked`, which `--parse-report` reports outside their
safe forms.

## The replay corpus (`tests/corpus/`)

Each corpus game is two files:

- `<name>.rmlog` — its decision log (RMLOG v3), written while `python/record_corpus.py` plays the
  scripted agent (hard, both seats) through `python/rm_driver.py`: one bo1 game per pairing at a
  frozen seed — the curated `delver` / `doomsday` / `mav` decks in every seating (9 games) and the
  `lg_<a>__<b>` league pairings and mirrors (45 games) — plus two explore vs explore games under
  `--no-auto-mana` (`manual_doomsday_vs_delver`, `manual_mav_mirror`), which cover interactive
  mana payment;
- `<name>.replay` — its baseline: the stdout of `robomage --replay <name>.rmlog --machine
  --narrative --trace-decisions`, the banner's log path reduced to the file name and the trace's
  board lines left out (`tools/robomage_check/corpus.h`).

Checking needs no Python; re-recording needs the agent. The corpus is identical from debug and
release builds and across platforms: all gameplay randomness goes through `src/stable_rng.h`. Never
call `std::shuffle`, `std::uniform_int_distribution` or `rand()` for a gameplay decision: their
output differs between libstdc++ and libc++.

## Workflows

- **`.github/workflows/ci.yml`** — push / PR to `main`, and `workflow_dispatch`: `make check` with
  the fixed seed, so a red run is the diff's fault, not RNG. `ci_out/` is uploaded as
  `ci-transcripts-<run_id>` even on success.
- **`.github/workflows/nightly-fuzz.yml`** — daily 08:00 UTC, plus manual (`games`, `seed`
  inputs). Two matrix jobs (`explore`, `explore:patient`) run the `fuzz` tier with `--matchups all`,
  40 games per matchup and per manual-mana ring matchup by default, with a seed rotating by run
  number unless given. The job summary prints the local reproduce command; artifact
  `nightly-fuzz-explore[-patient]-<run>`.

Both use `.github/actions/setup-robomage`: Python 3.12, card-script provisioning, ccache and the
debug build. The card-script cache is keyed on the `FORGE_PIN` hash and saved right after
provisioning; the fetch tool is add-only, so a stale cache within a pin re-fetches only the delta.

## Reproducing a CI failure locally

1. **PR gate:** run `make check` (same command, same seed). The failing tier names its transcript or
   engine stderr under `ci_out/` (a replay drift writes the actual output to
   `ci_out/replay/<name>.replay` and names the first differing line); the `ci-transcripts-*`
   artifact has CI's copies. Re-run with the run's last line, its `Reproduce:` command.
2. **One scenario:** `bin/debug/robomage_check --tier scenarios --scenario <name> --show output`
   prints the engine's whole output; the engine command is the scenario's presets as flags
   (`scenarios.h`).
3. **One fuzz or corpus game by hand:** a fuzz game is `cd bin && ./debug/robomage --agent-a explore
   --agent-b explore --game-seed <seed> --deck-a league/<a> --deck-b league/<b> --narrative
   --trace-decisions` (bo3 leg: `explore:patient` and `--bo3`; manual-mana leg: `--no-auto-mana`;
   the seed is in the tier's report line); a corpus game is `cd bin && ./debug/robomage --replay
   ../tests/corpus/<name>.rmlog --machine --narrative --trace-decisions`.
4. **Nightly fuzz:** run the `robomage_check --tier fuzz …` line from the job summary; the
   `nightly-fuzz-*` artifact has the transcripts.

**Engine self-checks on any game.** `--selfcheck-state` (`docs/state_format.md` §10),
`--selfcheck-featurize` and `--selfcheck-knowledge` (`docs/knowledge_store.md` §9.5) change no other
output; failures are `ERROR:` lines on stderr, and each prints a `SELFCHECK-…: … errors=N` summary
at match end. `--selfcheck-state-sample N` (default 8) sets how often the sampled permutation and
snapshot checks run. `--rmstate-dump FILE` (`%p` = the process id) writes every viewer's frames;
`robomage --rmstate-read FILE` re-encodes them byte for byte, and `bin/debug/rmknow --isolation
--viewer A|B --in FILE` recomputes each seat frame's `k_*` tables and compares them with the
engine's. The driver passes engine flags through `--engine-flags "…"` or `$ROBOMAGE_ENGINE_FLAGS`.

## Intentionally changing behavior

The `replay` tier pins current behavior, so an **intended** change re-records the corpus in the
same commit:

```bash
make regen    # provision the card set, `make`, then python3 python/record_corpus.py
```

`record_corpus.py` replays the corpus games with the fresh debug binary, rewriting each
`<name>.rmlog`, then runs `robomage_check --tier replay --record` for the baselines. Review and
commit the changed files with the reason in the commit message. A scenario whose assertion an
intended change breaks is edited by hand.

`make refetch` deletes the local card/token scripts, re-provisions them at the pinned Forge commit
and runs `make check` **without** re-recording, to detect script drift from the committed corpus;
if the drift is intended, follow with `make regen`.
