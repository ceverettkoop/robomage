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
(`src/state_format/decode.*`, `schema.*`), the agents' frame reader, Python-compatible
generator and policy projection (`src/agents/seat_frames.*`, `py_random.*`,
`policy_projection.*`) and `src/featurize/` (std only: RMTB, the
tokenizer, the featurizer the `featurize` tier runs in-process). It is built like the engine
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
| `--binary PATH` | the engine (default `robomage` beside the gate, so `bin/release/robomage_check` tests the release build); `rmknow`, `rmfeat` and `robomage_nn` are taken from the same directory |
| `--out-dir DIR` | transcripts, engine stderr and draw logs (default `ci_out/`) |
| `--fuzz-games N` / `--smoke-games N` | games per matchup (default 8 / 2) |
| `--manual-mana-games N` | `fuzz` tier: `--no-auto-mana` games per ring matchup (default 2; 0 skips them) |
| `--matchups PRESET` | `mirrors`, `ring` (each league deck vs the next), `mirrors+ring`, `all` (every ordered pair), or `"a:b,c:d"` (deck specs relative to `decks/`); default `fuzz` `mirrors`, `smoke` `mirrors+ring` |
| `--mode MODE` | the fuzz agent (bo1 and `--no-auto-mana` games): `explore` (default) or `explore:patient` |
| `--scenario GLOB` | `scenarios` tier: only the scenarios whose name matches (repeatable) |
| `--show output\|narrative\|dump` | `scenarios` tier: print each run's engine output, the narrative lines the assertions see, or its knowledge lines |
| `--record` | `replay` / `cli` tier: rewrite the baselines / goldens instead of checking them |

### The tiers

Times are wall clock for the debug build on 4 cores (`make check` total about 6.3 minutes).

| Tier | Checks | Time |
|---|---|---|
| `vocab` | `robomage --check-decks` over the top-level and `league/` decks: every card loads and is in the vocab (13 decks, 215 cards); `src/featurize/wordpiece.cpp` against `tests/embed/tokens_golden.jsonl` id for id, and every current text entry tokenises (skipped with a notice when the pinned model's `vocab.txt` is not in `.cache/embed/`) | ~1 s |
| `concede` | the CR 104.3a sentinels (140 checks) over RMSTATE sessions the gate answers itself: bo1 `-2`; bo3 `-2` into sideboarding and game 2; `-3`; the sentinels between games (`-2` ignored, `-3` ends the match); the same seed and replies give the same frames and the logged concession replays; and concessions at parked queries (Brainstorm's put-back mid-resolution, Cavern of Souls' ETB creature type), each also piped unread into a second `--machine` session, which must write the same frames. Results, narrative and `--replay` of every decision log | ~2 s |
| `scenarios` | every rules-regression scenario (`tests/scenarios/*.json`, 98) on the engine alone: presets as engine flags, the `--play` resolver, `--narrative --trace-decisions`; every assertion key (`tools/robomage_check/scenarios.h`), the knowledge lines read from the binary dump | ~4 s |
| `cli` | the human CLI's golden transcripts (`tests/cli/`, 6 sessions): each `<name>.json` is an engine command line and the lines a person types; the engine runs it (`--decision-log` under the out dir) with those lines on stdin, then end of input, must exit 0 with nothing on stderr, and its stdout, the `Decision log:` path reduced to `<name>.rmlog`, must match `<name>.out` byte for byte (a drift's actual output goes to `ci_out/cli/<name>.out`); `--replay` of the session's decision log must exit 0 with no error, stopping at the session's quit (`QUIT: ...`) when it ended by `quit` or end of input. Human vs explore, hotseat, `--no-auto-mana` payment, auto-pass, a bo3 match conceded, a split card | <1 s |
| `replay` | each corpus decision log replayed (`robomage --replay <name>.rmlog --machine --narrative --trace-decisions`), its stdout against `<name>.replay` byte for byte (66 games) | ~17 s |
| `state` | the corpus replayed under `--selfcheck-knowledge` (same output, clean `SELFCHECK-STATE` / `SELFCHECK-KNOWLEDGE` summaries); 16 explore fuzz games with the self-checks and dumps; every scenario at `--selfcheck-state-sample 1`; `--rmstate-read` on the dumps; the decoder's refusals of mutated frames; two RMSTATE stream sessions against their dumps (with and without one written); the name-a-card menus with open and closed decklists; the `CARDS` frame's ability parameters (register 41: Lightning Bolt, Brainstorm's sub-ability, Kozilek's Command's modes, every ability's parts in order); the include isolation of the std-only sources (`src/knowledge/`, `src/featurize/`) | ~2.2 min |
| `featurize` | the featurizer (`docs/featurizer.md` §7): `robomage --cards-frame`, then the card tables and FEATURE_SPEC built from it in-process, exported, read back (the same tables and spec bytes) and equal to `rmfeat --spec` / `--export-tables`'s; the corpus replayed with dumps and four explore games (bo1, bo3, `--closed-decklist`, seat A at `expert+belief`) under `--selfcheck-featurize` (clean `SELFCHECK-STATE` / `SELFCHECK-FEATURIZE` summaries: each seat's tensors from the actor entry point equal under the scramble and permutation checks' frames, the omniscient frame refused) with dumps; every decision with a menu of every dump featurized (seat frames by the actor entry point, which must refuse the omniscient frame): widths and every categorical, card, ability, relation and action index against the spec and the tables, `act_tok` one per menu row; every third decision featurized twice to the same tensors and, with every token table but `menu` reshuffled (and its `row:` references renumbered), to the same tensors up to the token permutation; each seat decision's targets against the seat's frame (hand and library sizes) and the opponent's (its `k_template` registered row); `rmfeat --featurize` / `--targets` over one dump equal to the in-process files; `python/pycheck.py --part featurize` reads the tables, a tensors and a targets file to the values written, with the spec's hashes, shapes and indices | ~1 min |
| `traj` | the trajectory writer (`docs/trajectory_format.md`): four in-process runs with `--trajectory-out` under `ci_out/traj/` (random against explore: a bo1 game with `--trajectory-omniscient` and a dump, a bo3 match with a dump, a bo3 match whose game 1 seat A concedes and whose match it concedes in sideboarding, a bo3 match of seat A's decisions only stopped at `--max-decisions 40`), each with no error, then a `--replay` of its decision log with `--trajectory-out`: the same episode files, every array byte for byte but the behaviour policy's (`has_mu`, `mu`, `policy_version`), the same attrs but the agents'; no temporary file left; per episode the game and seats, sideboarding decisions first and only before games 2+, every reply a menu row or sentinel, the random agent's steps carrying a uniform μ (sum 1 within 1e-5, mass at the reply) and explore's none, no teacher label, the omniscient arrays present exactly when asked; one episode per game, each footer's result the `GAME_RESULT` line's, the end reasons (win, concession twice, cap), the replies in order the decision log's and the `game_decisions` its decision count; every decision's (seat, game, sideboarding, decision) and seat tensors those `featurize_seat` gives on the dump's frame (tok_num rounded to f16), the omniscient tensors and targets those of the dump's omniscient frame; the bytes per decision by array printed; `python/pycheck.py --part traj`: `python/rmtraj.py` reads two episodes (the omniscient one, a bo3 game 2 with sideboarding) to the C++ reader's values; and a scripted vs explore bo3 match through `python/rm_driver.py --rmlog-dir` re-featurized from its kept log (μ-absent episodes named after the log, replies and footers as above); then the teacher channel (`--teacher-a/-b`, `python/teacher.py` at the hard tier, under `ci_out/traj/teacher/`; bug vs burn): **mix1**, seat A taught at `--teacher-mix 1` (explore deciding, `first` on B) in a bo3 match: its decision log's decisions are those of `rm_driver.py --player-a scripted --player-b first` with the same seed, the `--machine` run of the same game (seat A answered with those replies) has the same decisions and results and its seat A records are the teacher's recorded stream byte for byte (match frames, sideboarding and terminal frames included), every A step's label is its reply with μ one-hot there (an A sideboarding step included), B's steps carry neither; **mix0**, both seats taught at mix 0 (explore against random, bo3): the decision log byte for byte the run's without teachers, every step's reply and μ unchanged, every step labelled, and the labels those `teacher.py` gives on each recorded stream and on both streams read as two successive matches of one stream; **mix0.5**, seat A (random) taught at 0.5: the reply is the label wherever the coin (`random.Random(seed + 2**32)`) falls under 0.5 (both outcomes occur), μ = 0.5·onehot(label) + 0.5/n within 1e-6, the labels `teacher.py`'s on its stream, and a `--replay` of the decision log (no teacher) writes the same episodes (all but μ and the labels) and results; **failures**: a teacher that exits at once, replies garbage, hangs past `--teacher-timeout 1`, or exits with status 3 at the end, and `--teacher-mix` on a stream seat, each exit 1 with a `FATAL:` naming the seat | ~27 s |
| `actorloop` | the continuous actor (`--actor`, `docs/actor.md`) under `ci_out/actorloop/`, four parts at once, each actor run ending with exit 0, no error line, a status file whose state, matches, games, decisions, caps, draws, concessions, errors and policy versions agree with its results lines (decisions the seats' sum), and every results line agreeing with its episode's footer (game, first player, results, end reason, decisions = `game_decisions`, match score, decks, agents, seeds, decision log, the recorded seats' policy versions; one line per episode): **main**, six matches of a two-entry schedule (a bo1 entry, random against explore, swapping sides; a bo3 entry, explore against random with seat B's knowledge `off`, swapping decks), both entries played, seat A holding the deck and agent the swap coin says, every match's decision log replayed alone (`--replay` with the line's knowledge levels and `--trajectory-out`) to the same episodes (`compare_replayed`) and game ends, and a second run with the same run seed and actor id writing the same decision logs byte for byte and the same results lines but the paths and timings; **cap**, a bo3 entry with `max_game_decisions` 25 over three matches: each match one game, capped (end reason `cap`, 25 decisions, results 0, no `MATCH_RESULT`), the actor going on, the replays capping at the same decision; **teacher** (skipped without `python3`), a bo1 entry whose side a is taught by `python/teacher.py --seed 11` at mix 0 and swaps sides, three matches: every step of the taught seat labelled and none of the other, and each match's labels equal to those of a fresh teacher on the match's `--replay`; **change_stop**, an actor run until stopped with an eval schedule (bo1, no episodes) rewritten after its first game to a train bo3 schedule: the rewritten schedule's matches come after every old one (status: the new version, two loads), the eval games have no episodes and no eval episode directory exists, and SIGTERM sent after the new schedule's first game 1 stops it within that match: exit 0, state `stopped`, the match's decision log ending with `QUIT` (its `--replay` stops there), only its game 1's episode, no temporary file | ~10 s |
| `actor` | opt-in (below; otherwise one `skipped:` line): the reference policy built (`tools/nn/make_reference_policy.py build` under `ci_out/actor/`, rebuilt only when the spec or the script changes, ~50 s); with or without the opt-in, the policy projection (`src/agents/policy_projection.h`, linked in) on known answers and 30,000 random menus: thresholding (rows below p 0, the rest renormalised, untouched when every row is below p), discretisation (multiples of 1/n whose counts sum to n, exactly 1 for a power of two, the most probable row kept, a row of 0 kept at 0, a fixed point, deterministic) and a draw always landing on a row with mass; `bin/<config>/robomage_nn` plays four bo3 matches against explore: `nn:` on the TorchScript export (sampled, `dump=`), `nn:` sampled under `temp=0.02:thresh=0.03:disc=16` (the low temperature puts rows of the reference policy's nearly flat logits below the threshold; `evals=`: every decision's `mu` is the projection of `softmax(logits / 0.02)`, multiples of 1/16 summing to 1, rows below 0.03 at 0, the chosen row with mass; the rows dropped and distributions moved are counted) and `argmax` on the TorchScript and on the `.pt2` export, each with a result, no drawn game and no error or warning line; the two argmax matches' decision logs are identical; `rmfeat --featurize` over a plain `robomage --replay` of the sampled match's decision log (`--rmstate-dump`) equals the tensors `nn:` featurized live, byte for byte; `make_reference_policy.py check` on those decisions: the TorchScript, `.pt2` and ONNX exports' logits and values agree, alone and in padded batches (1e-4 absolute plus relative; the largest differences are printed); the plain engine refuses `nn:` naming `make actor`, and `robomage_nn` an unknown key, a remote-only key without `remote:`, `thresh=1`, `disc=-16` and `nn:remote:` (with `thresh=` and `disc=`) with no server. Then remote evaluation (`ci_out/actor/remote/`): four argmax bo1 games (`nn:` on both seats, `evals=` files) in process, and the same four played at once by four actor processes (eight clients) through one `bin/<config>/rmserve` (`--max-batch 64 --deadline-us 3000`; seat A asks for policy 0, the TorchScript export, and policy 1, the `.pt2` export served as version 7): every game's decision log equals the in-process one, every evaluation's logits, `log π` and value agree within 1e-5 (absolute plus relative; largest differences printed) and carry the expected versions, rmserve's `RMSERVE-STATS:` shows a batch of more than one decision (throughput, batches and server compute printed); rmserve refuses raw handshakes with another spec hash, protocol 2 and an unserved policy (ERROR `spec_hash`, `protocol`, `policy`), and an actor asking for an unserved policy exits naming the endpoint; an rmserve stopped by `--pause-after 40` and killed leaves the actor (both seats remote) exiting 1 with a `FATAL:` naming the endpoint and exactly 40 logged decisions; `tools/nn/serve_check.py` (the Python server) plays the first game decision for decision; each server stops cleanly on SIGTERM. Trajectories (`--trajectory-out`): the sampled match's episodes hold seat A's live `dump=` tensors (tok_num rounded to f16; the writer reuses the agent's) and replay identically from its decision log; the projected match's steps carry its `evals=` μ, replies, versions, values and `log π`; remote game 0's steps carry seat A's two policies' and seat B's one. Hot reload (`ci_out/actor/reload/`): `robomage_nn --actor` with an `nn:<model>:stats` vs explore entry over three bo1 matches, the model a copy of the TorchScript export saved with policy_version 3 and, after match 0's game, renamed over by one with policy_version 5: match 0 records version 3, match 2 version 5, match 1 one of them, each episode's `policy_versions_a` its results line's, the status counts both, `NN-STATS:` shows one reload | ~1.5 min (+~50 s when the reference policy is rebuilt) |
| `knowledge` | `rmknow --selftest`, `--exactness 40`, `--belief-exactness 10`; six explore games under `--selfcheck-knowledge`; three stream sessions and two `expert+belief` sessions against their dumps (`k_*` / `b_*` included); `rmknow --isolation` on every dump and saved stream, `--truth` on every dump; four archetype-inference sessions against `archetypes.json`; the belief calibration sessions and their report | ~75 s |
| `fuzz` | C++ explore vs explore in-process (`--agent-a/-b <mode>`) over the matchups (default the ten league mirrors), `--fuzz-games` bo1 games each (matchup `k` game `i`: game seed `seed + 1000k + i`), then one bo3 `explore:patient` match per ring matchup (seed `seed + 500000 + k`), then `--manual-mana-games` bo1 games per ring matchup under `--no-auto-mana`, so manual mana payment is covered (every mana ability offered at priority, each cost paid in the payment loop; ring matchup `k` game `i`: seed `seed + 700000 + 1000k + i`); transcripts `ci_out/fuzz_<a>__<b>.txt`, `fuzz_bo3_<a>__<b>.txt`, `fuzz_manual_<a>__<b>.txt` | ~50 s |
| `bench` | per decision (wall clock) of a fixed workload — explore vs explore, bo1, three league pairings × three games, one engine at a time — by stage, each the previous plus one part of the actor loop: the engine alone (`--replay` of the games' logs), + frames with knowledge `off`, + the knowledge sidecar (`expert`; its decisions/s, process start included, is the headline figure), + the featurizer (`--featurize-seats`, counted by its in-process time: building its inputs at start parses every vocab card, which the other stages parse during play, so its own wall clock without the start understates it); each game the faster of two runs, the process start (the same command under `--max-decisions 0`) left out and shown beside it; the stages must play the same games. With the `actor` tier's opt-in, the evaluation stages: the same pairings and seeds with `nn:` on both seats in `robomage_nn` (other games), wall clock and the agents' featurize and evaluate time per decision; then the same games with `nn:remote:` through an `rmserve` (`--deadline-us 0`, one actor at a time, so no batching wait): wall clock, and per decision the client's evaluate time split into the server's compute and the round trip (protocol, copies, socket), and whether it played the same games. Reported, no threshold | ~1.7 min, ~3.5 min with the evaluation stages (debug build) |
| `smoke` | scripted hard vs hard through `python/rm_driver.py` (one process per matchup, default the mirrors and the ring, `--smoke-games` each; transcripts `ci_out/smoke_<a>__<b>.txt`), and beside them `python/pycheck.py`: `python/rmstate.py`'s decode of three dumps (an in-process bo1 game and bo3 match, a driver bo3 session under `--closed-decklist`) renders the engine's `--state-dump json` lines byte for byte, both decoders refuse (or skip) the same mutated frames, the scripted agent's self-lethal guard in sculpted positions, the driver's pinned engine seeds, a driven game imports nothing outside the standard library and `python/`, and `python/rmtb.py` reads the RMTB samples the gate wrote with `src/featurize/rmtb.cpp` (every dtype, a ragged pair, a text table; f16 against `struct`'s rounding) and the committed text table to the written values. Skipped with a notice when no `python3` is on `PATH` | ~35 s |

**The `actor` tier is opt-in.** It needs `bin/<config>/robomage_nn` and `bin/<config>/rmserve`
beside the engine (`make actor LIBTORCH_DIR=…`, `CLAUDE.md`) and `$ROBOMAGE_TORCH_PYTHON` naming a Python with torch, onnx,
onnxscript and onnxruntime (which builds and checks the reference policy); without either it prints
`skipped: …` (not a warning) and passes, so plain `make check` needs no torch. The `bench` tier's
evaluation stage has the same opt-in.

```bash
ROBOMAGE_TORCH_PYTHON=/path/to/venv/bin/python bin/debug/robomage_check --tier actor,bench
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

- `<name>.rmlog` — its decision log (RMLOG v3: `SEED`, `FLAGS`, both decklists, every decision),
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
(`src/state_format/selfcheck.h`, `docs/state_format.md` §10), `--selfcheck-featurize` adds the
featurizer's (`src/state_format/selfcheck_featurize.h`; `SELFCHECK-FEATURIZE: … errors=N`) and
`--selfcheck-knowledge` adds the knowledge sidecar's soundness check (`docs/knowledge_store.md`
§9.5); none changes any other
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
