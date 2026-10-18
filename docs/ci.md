# Continuous integration & the test standard

The standard for changes to this repo: **`make check` must pass.** CI runs exactly it; run it
locally before pushing.

```bash
make check    # `make` (debug build), then bin/debug/robomage_check (every tier)
```

It needs the Forge card scripts provisioned first (the SessionStart hook does this in cloud
sessions only; run it yourself on a fresh local clone; stdlib only):

```bash
python3 tools/forge_fetch/provision_decks.py
```

`provision_decks.py` fetches the script for **every card in `src/card_vocab.h`** plus every card
named by the top-level, `meta/` and `league/` decks, and the tokens they create, at the Forge
commit pinned in `tools/forge_fetch/FORGE_PIN`. `robomage_check` exits 2 without running anything
when the engine binary or the card scripts are missing, or when `--tier` names an unknown tier.

## The gate: `robomage_check`

`tools/robomage_check/` builds `bin/<config>/robomage_check` (part of `make`). It runs the engine
as subprocesses, in parallel (`-j N`, default the number of cores), with `bin/` as their working
directory, and links no engine translation unit: only the std-only RMSTATE decoder
(`src/state_format/decode.*`, `schema.*`) and the agents' frame reader and Python-compatible
generator (`src/agents/seat_frames.*`, `py_random.*`). It is built like the engine
(`-fno-exceptions`; a scenario regex is compiled in a forked child first, so a malformed one is
reported, not an abort).

Every requested tier runs even if an earlier one fails, so one invocation reports every finding.
The run ends with each tier's PASS/FAIL, a wall-time table, every warning and error, and a
`Reproduce:` line naming its tiers, seed, mode and matchups; it exits 1 on any **error**
(warnings alone pass), 2 on bad usage or a missing binary.

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
| `--binary PATH` | the engine (default `robomage` beside the gate, so `bin/release/robomage_check` tests the release build); `rmknow` is taken from the same directory |
| `--out-dir DIR` | transcripts, engine stderr and draw logs (default `ci_out/`) |
| `--fuzz-games N` / `--smoke-games N` | games per matchup (default 8 / 2) |
| `--manual-mana-games N` | `fuzz` tier: `--no-auto-mana` games per ring matchup (default 2; 0 skips them) |
| `--matchups PRESET` | `mirrors`, `ring` (each league deck vs the next), `mirrors+ring`, `all` (every ordered pair), or `"a:b,c:d"` (deck specs relative to `decks/`); default `fuzz` `mirrors`, `smoke` `mirrors+ring` |
| `--mode MODE` | the fuzz agent (bo1 and `--no-auto-mana` games): `explore` (default) or `explore:patient` |
| `--scenario GLOB` | `scenarios` tier: only the scenarios whose name matches (repeatable) |
| `--show output\|narrative\|dump` | `scenarios` tier: print each run's engine output, the narrative lines the assertions see, or its knowledge lines |
| `--record` | `replay` / `cli` tier: rewrite the baselines / goldens instead of checking them |

### The tiers

Times are wall clock for the debug build on 4 cores (`make check` total about 5.3 minutes).

| Tier | Checks | Time |
|---|---|---|
| `vocab` | `robomage --check-decks` over the top-level and `league/` decks: every card loads and is in the vocab (13 decks, 215 cards) | <1 s |
| `concede` | the CR 104.3a sentinels (140 checks) over RMSTATE sessions the gate answers itself: bo1 `-2`; bo3 `-2` into sideboarding and game 2; `-3`; the sentinels between games (`-2` ignored, `-3` ends the match); the same seed and replies give the same frames and the logged concession replays; and concessions at parked queries (Brainstorm's put-back mid-resolution, Cavern of Souls' ETB creature type), each also piped unread into a second `--machine` session, which must write the same frames. Results, narrative and `--replay` of every decision log | ~2 s |
| `scenarios` | every rules-regression scenario (`tests/scenarios/*.json`, 98) on the engine alone: presets as engine flags, the `--play` resolver, `--narrative --trace-decisions`; every assertion key (`tools/robomage_check/scenarios.h`), the knowledge lines read from the binary dump | ~4 s |
| `cli` | the human CLI's golden transcripts (`tests/cli/`, 6 sessions): each `<name>.json` is an engine command line and the lines a person types; the engine runs it (`--decision-log` under the out dir) with those lines on stdin, then end of input, must exit 0 with nothing on stderr, and its stdout, the `Decision log:` path reduced to `<name>.rmlog`, must match `<name>.out` byte for byte (a drift's actual output goes to `ci_out/cli/<name>.out`); `--replay` of the session's decision log must exit 0 with no error, stopping at the session's quit (`QUIT: ...`) when it ended by `quit` or end of input. Human vs explore, hotseat, `--no-auto-mana` payment, auto-pass, a bo3 match conceded, a split card | <1 s |
| `replay` | each corpus decision log replayed (`robomage --replay <name>.rmlog --machine --narrative --trace-decisions`), its stdout against `<name>.replay` byte for byte (66 games) | ~17 s |
| `state` | the corpus replayed under `--selfcheck-knowledge` (same output, clean `SELFCHECK-STATE` / `SELFCHECK-KNOWLEDGE` summaries); 16 explore fuzz games with the self-checks and dumps; every scenario at `--selfcheck-state-sample 1`; `--rmstate-read` on the dumps; the decoder's refusals of mutated frames; two RMSTATE stream sessions against their dumps (with and without one written); the name-a-card menus with open and closed decklists; the include isolation of the std-only sources | ~2.2 min |
| `knowledge` | `rmknow --selftest`, `--exactness 40`, `--belief-exactness 10`; six explore games under `--selfcheck-knowledge`; three stream sessions and two `expert+belief` sessions against their dumps (`k_*` / `b_*` included); `rmknow --isolation` on every dump and saved stream, `--truth` on every dump; four archetype-inference sessions against `archetypes.json`; the belief calibration sessions and their report | ~75 s |
| `fuzz` | C++ explore vs explore in-process (`--agent-a/-b <mode>`) over the matchups (default the ten league mirrors), `--fuzz-games` bo1 games each (matchup `k` game `i`: game seed `seed + 1000k + i`), then one bo3 `explore:patient` match per ring matchup (seed `seed + 500000 + k`), then `--manual-mana-games` bo1 games per ring matchup under `--no-auto-mana`, so manual mana payment is covered (every mana ability offered at priority, each cost paid in the payment loop; ring matchup `k` game `i`: seed `seed + 700000 + 1000k + i`); transcripts `ci_out/fuzz_<a>__<b>.txt`, `fuzz_bo3_<a>__<b>.txt`, `fuzz_manual_<a>__<b>.txt` | ~50 s |
| `bench` | decisions per second of explore vs explore, bo1, three league pairings × three games, one engine at a time (reported, no threshold) | ~8 s |
| `smoke` | scripted hard vs hard through `python/rm_driver.py` (one process per matchup, default the mirrors and the ring, `--smoke-games` each; transcripts `ci_out/smoke_<a>__<b>.txt`), and beside them `python/pycheck.py`: `python/rmstate.py`'s decode of three dumps (an in-process bo1 game and bo3 match, a driver bo3 session under `--closed-decklist`) renders the engine's `--state-dump json` lines byte for byte, both decoders refuse (or skip) the same mutated frames, the scripted agent's self-lethal guard in sculpted positions, the driver's pinned engine seeds, and a driven game imports nothing outside the standard library and `python/`. Skipped with a notice when no `python3` is on `PATH` | ~35 s |

```bash
bin/debug/robomage_check --tier scenarios --scenario 'cr702_21a*' --show narrative
bin/debug/robomage_check --tier replay                 # one tier
bin/debug/robomage_check --tier cli --record           # rewrite the CLI goldens after an intended change
bin/debug/robomage_check --tier fuzz --matchups all --mode explore:patient --fuzz-games 20 --seed 7
bin/release/robomage_check --tier bench                # throughput of the release build
make check                                             # build, then every tier
```

### Error and warning classification

Draws are not acceptable, but the causes differ in severity (`tools/robomage_check/scan.h`):

- **Error** (fails the gate): an engine crash (nonzero exit / EOF mid-game), fewer games
  completed than requested, a failed check, or any output line starting `ERROR:` / `FATAL:` or
  holding a Python traceback, `Segmentation fault`, a failed assertion, `Aborted` or
  `core dumped` (a non-fatal engine `ERROR:` is not acceptable per `CLAUDE.md`), or a
  `SELFCHECK-STATE:` / `SELFCHECK-KNOWLEDGE:` summary without `errors=0`.
- **Warning** (flagged for review, passes): a game that ends with no winner — a drawn game (both
  players lost at once, CR 104.4a) or a **stall** — or an engine `WARNING:` line. A drawn game's
  transcript is also saved as `ci_out/<tier>_<a>__<b>_g<i>_draw_<stamp>.txt`.

`WARNING: Unrecognized ability param` lines stay in the transcripts (greppable) but are not
surfaced: they fire in bulk for cosmetic / AI-hint params. Params the parser drops without
warning are the `ignored_keys` set in `src/parse.cpp`; their review is tracked in `todo.md`
("Audit: ability-param keys the parser silently ignores").

## The replay corpus (`tests/corpus/`)

Each corpus game is two files:

- `<name>.rmlog` — its decision log (RMLOG v2: `SEED`, `FLAGS`, both decklists, every decision),
  written by the engine's `--decision-log` while `python/record_corpus.py` plays the scripted
  agent (hard, both seats) through `python/rm_driver.py`: one bo1 game per scenario at a frozen
  seed — the curated `delver` / `doomsday` / `mav` decks in every seating (9 games) and every
  league pairing plus each mirror (55 games, `lg_<a>__<b>`) — plus two explore vs explore games
  under `--no-auto-mana` (`manual_doomsday_vs_delver`, `manual_mav_mirror`: every mana ability
  offered and each cost paid in the payment loop, its cancel row included; the `FLAGS` line
  carries `no-auto-mana`), since the scripted agent pays with the first row offered;
- `<name>.replay` — its baseline: the stdout of `robomage --replay <name>.rmlog --machine
  --narrative --trace-decisions`, with the replay banner's log path reduced to the file name and
  the trace's board lines (between `| decision` and `| menu:`) left out, so it holds the
  narrative and each decision's header, menu and choice (`tools/robomage_check/corpus.h`).

Checking needs no Python; re-recording needs the agent. The corpus is identical from the debug
and release builds and **platform-portable**: all gameplay randomness goes through
`src/stable_rng.h` (hand-rolled Fisher–Yates + rejection sampling over the raw mt19937 stream), so
Mac and Linux produce byte-identical transcripts from a seed. Never call `std::shuffle`,
`std::uniform_int_distribution` or `rand()` with `cur_game.rng.engine` (or for any gameplay
decision): their output is implementation-defined and differs between libstdc++ (CI) and libc++
(macOS), which would lock the corpus, RMLOG replays and bug-repro seeds to one platform.

## Workflows

- **`.github/workflows/ci.yml`** — push / PR to `main`, and `workflow_dispatch`. Runs `make check`
  (every tier) with the fixed seed, so a red run is the diff's fault, not RNG. `ci_out/` is
  uploaded as `ci-transcripts-<run_id>` even on success, so stall and draw logs are reviewable.
- **`.github/workflows/nightly-fuzz.yml`** — daily 08:00 UTC, plus manual (`games`, `seed`
  inputs). Two matrix jobs run the gate's `fuzz` tier over **all** league matchups (100 ordered
  pairs plus the bo3 ring matches and the `--no-auto-mana` ring games), one with `explore` and one
  with `explore:patient`, 40 games per matchup (and per ring matchup without auto-mana) by default, with a **rotating** seed (run number × 100000 unless given). The job summary
  prints the exact local reproduce command; artifact `nightly-fuzz-explore[-patient]-<run>`.

Both share `.github/actions/setup-robomage`: Python 3.12 (stdlib only: card-script provisioning
and the `smoke` tier), card-script provisioning, ccache and the debug build. The card-script cache
is keyed on the `FORGE_PIN` hash (a pin bump restores nothing and re-fetches everything) and saved
right after provisioning, so even a failing first run leaves a warm cache; the fetch tool is
add-only, so a stale cache within a pin only re-fetches the delta.

## Reproducing a CI failure locally

1. **PR gate:** run `make check` (same command, same fixed seed). The failing tier names its
   transcript or engine stderr under `ci_out/` (a replay drift writes the actual output to
   `ci_out/replay/<name>.replay` and names the first differing line); the run's
   `ci-transcripts-*` artifact has CI's copies. Re-run one tier with
   `bin/debug/robomage_check --tier <name>` (the run's last line is its `Reproduce:` command).
2. **One scenario:** `bin/debug/robomage_check --tier scenarios --scenario <name> --show output`
   prints the engine's whole output; the engine command it ran is the scenario's presets as flags
   (see `scenarios.h`).
3. **One fuzz or corpus game by hand:** a fuzz game is exactly `cd bin && ./debug/robomage
   --agent-a explore --agent-b explore --game-seed <seed> --deck-a league/<a> --deck-b league/<b>
   --narrative --trace-decisions` (the bo3 leg: `explore:patient` and `--bo3`; the manual-mana leg:
   `--no-auto-mana`; the seed is in the tier's report line); a corpus game is `cd bin && ./debug/robomage --replay
   ../tests/corpus/<name>.rmlog --machine --narrative --trace-decisions`.
4. **Nightly fuzz:** run the `bin/debug/robomage_check --tier fuzz --matchups all --mode …
   --fuzz-games … --seed …` line from the job summary; the `nightly-fuzz-*` artifact has the
   transcripts.

**Engine self-checks on any game.** `--selfcheck-state` runs the state format's self-checks
(`src/state_format/selfcheck.h`, `docs/state_format.md` §10) and `--selfcheck-knowledge` adds the
knowledge sidecar's soundness check (`docs/knowledge_store.md` §9.5); neither changes any other
output (the replay corpus stays byte-identical under them), failures are `ERROR:` lines on the
engine's stderr, and each engine prints a `SELFCHECK-STATE: … errors=N` /
`SELFCHECK-KNOWLEDGE: … errors=N` summary at match end. `--selfcheck-state-sample N` (default 8)
sets how often the sampled permutation and snapshot checks run. `--rmstate-dump FILE` (`%p` = the
process id) writes every viewer's frames beside an unchanged run; `robomage --rmstate-read FILE`
re-encodes them byte for byte, and `bin/debug/rmknow --isolation --viewer A|B --in FILE`
recomputes every seat frame's `k_*` tables from the frames and compares them with the engine's.
The driver passes any engine flag through `--engine-flags "…"` or `$ROBOMAGE_ENGINE_FLAGS`.

## Intentionally changing behavior

The `replay` tier pins current behavior, so an **intended** change re-records the corpus in the
same commit (the diff is part of the review):

```bash
make regen    # provision the card set, `make`, then python3 python/record_corpus.py
```

`record_corpus.py` replays the corpus games with the fresh debug binary, rewriting each
`<name>.rmlog`, then runs `robomage_check --tier replay --record` for the baselines; an unchanged
engine + agent + card data reproduces every byte. Review and commit the changed files with the
reason in the commit message. A scenario whose assertion an intended change breaks is edited by
hand.

`make refetch` deletes the local card/token scripts, re-provisions them at the pinned Forge
commit and runs `make check` **without** re-recording, to detect whether fresh scripts drift from
the committed corpus; if the drift is intended, follow with `make regen`.

The gating fuzz is `robomage_check --tier fuzz`; a larger campaign is the same tier with
`--matchups all --fuzz-games N` (and `--mode explore:patient`), its transcripts under `ci_out/`.
