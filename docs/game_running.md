# Running games: modes, agents, I/O

How a RoboMage game gets run outside the torch training loop — what the engine
provides, where agents live, and the one Python API to script games with.

## The engine has no agents

`bin/robomage` knows nothing about scripted tiers, models, or humans. In
`--machine` mode it emits a `BQUERY` binary frame on stdout at every decision
point — for **both** seats, interleaved in priority order — and reads a single
integer choice back on stdin. Who picks that integer is entirely the driver's
concern. The only engine-side seat distinction is `--player A|B` (diverts one
seat to the CLI's interactive stdin prompt, a blocking `getchar`/`scanf` read;
used by direct interactive play, not by any Python-driven mode) and
`--log-viewer A|B` (redacts the private narrative to one seat's view without
rerouting input; used by the TUI).

Engine-level knobs (see `src/main.cpp` argv parsing for the full list):
decks (`--deck-a/-b`), `--seed`, `--no-shuffle`, zone presets
(`--battlefield/graveyard/exile/sideboard-a/-b`, `--life-a/-b`), `--narrative`
(full game log + per-action description/counter side-channels), `--bo3`
(match loop: per-game seed = base+game, loser goes first, sideboarding between
games, `GAME_RESULT:`/`MATCH_RESULT:` protocol lines), `--replay <rmlog>`
(self-contained deterministic replay), `--log-decisions` (write the replay
log in machine mode).

## The Python stack (train/)

One layer per concern — everything that runs games sits on this stack:

| Layer | Module | Role |
|---|---|---|
| Engine wrapper | `env.py` (`RoboMageEnv` / `NarrativeEnv`) | subprocess launch, BQUERY parse, obs assembly, result/reward parse, seeds |
| Decoding | `decode.py` | state vector / action menu → human-readable structures and transcript blocks |
| Agents | `opponents.py` (`Controller`) | scripted tiers, model checkpoints, `--play` scripts, action lists, human CLI, autopass |
| Loop | `runner.py` (`drive_game`) | THE decision loop: route priority seat → controller, step, hooks |
| Orchestration | `runner.py` (`run_games`, `run_match`) | env per game, transcripts, tallies, records |

### Agent specs (one grammar everywhere)

`opponents.make_controller(spec)` — used by `run_match`, `observe`, the test
harness, the TUI, and `play.py` — accepts:

- `"scripted"` / `"hard"` — the heuristic HARD tier (the default anywhere a
  bare `scripted` appears)
- `"easy"` / `"greedy"`, `"random"`, `"explore"`, `"explore:patient"` — other
  scripted tiers (fuzzing profiles included)
- `"human"` — interactive CLI seat: renders board + menu, accepts an index or
  a semantic spec (`cast:bolt`, `target:bears@opp`, `pass`), `quit` to exit
- `"play:<spec,spec,...>"` — pre-baked semantic action script (the harness
  `--play` grammar, `action_spec.py`)
- `"actions:<i,i,...>"` — positional action-index list
- `"auto"` — always action 0 (pass / first choice)
- the generalist model spec `"gen"` (→ `checkpoints/gen__final.zip`, else newest
  `gen__v*`) or an explicit checkpoint path, resolved by
  `opponents.resolve_checkpoint` — the single resolver shared by train.py,
  play.py and the TUI (a bare deck shorthand is rejected: the deck travels
  separately as an explicit parameter)
- `"az:<gen-or-path>"` / `"mcts:<ckpt>"` (MCTS search) or `"azraw:<gen-or-path>"`
  (raw AZ policy) — `SearchController`/`AZRawController`. The search specs take a
  `?k=v&…` query: `sims`/`worlds`/`c`/`temp`/`seed` (in-game search) plus
  `sb_branches`/`sb_worlds`/`sb_rollout_turns` — the **bo3 sideboard
  plan-search** budget (defaults `8`/`4`/`6`, the `DEFAULT_SB_*` constants in
  `cli_spec.py`). A sideboard prompt is not searched with PUCT: the controller
  runs `mcts.run_plan_search`, a flat search over complete sideboard
  configurations — one argmax-greedy completion per legal first pick (the
  coverage pass, Done included) plus `sb_branches` deterministic alternate
  completions of the best branches — each priced by a raw-policy rollout on
  every `sb_worlds` world to end of player-turn `sb_rollout_turns` of the
  sampled next game (`sb_rollout_turns=0` prices the completed decklist with
  the net's static read). Plan value = the cross-world mean, Q per first pick
  = its best plan, and the played pick / recorded `pi` come from
  `softmax(Q / mcts.SB_PI_TAU)`. Plan values are memoized per (world seed,
  pick multiset) in a boundary-shared table, so a boundary's later picks
  mostly re-price from cache (the world seeds stay pinned to the boundary's
  first searched root); see `test_mirror_search.py`'s
  `parallel_sb_persistence` and `test_plan_search.py`. e.g.
  `az:gen?sims=64&worlds=4&sb_branches=4&sb_rollout_turns=8`.
  - `time=<seconds>` sets a **wall-clock per-decision budget** instead of a fixed
    sim count: the search interleaves its `worlds` round-robin and runs as many
    simulations as fit in that many seconds, then stops (more time = stronger
    play). The one budget applies to both in-game and sideboard roots (at a
    sideboard root the plan search's coverage pass is the floor and the clock
    truncates the extras; note the deadline is checked between plan
    evaluations, so it can overshoot by up to one rollout). It overrides
    `sims` as the terminator — `sims`, when explicitly pinned alongside
    `time=`, acts only as a hard cap. A floor of one sim per world always
    runs. e.g. `az:gen?time=5&worlds=4`.
    `play.py --think-time <seconds>` is the CLI front door that appends this knob.
    When `time=` is absent the fixed-`sims` path is byte-for-byte unchanged (the
    actor visit-parity corpus depends on it).
  - `procs=<n>` (default `1`) runs a **world-parallel mirror pool** for
    **interactive** search: the engine is single-threaded, but a search's `worlds`
    are independent, so `n-1` extra engine processes are kept in lockstep with the
    primary game and the worlds fan out across all `n` processes concurrently
    (~near-linear more sims/decision for `procs ≤ worlds`, whether the terminator
    is `sims` or `time=`). `procs=1` is byte-identical to the plain single-engine
    search — self-play and the parity corpus never use the pool. `play.py
    --search-procs <n>` is the CLI front door. e.g. `az:gen?time=2&procs=4`.
    The **spec-grammar default stays 1** (gates / eval / parity reproducibility),
    but the INTERACTIVE front doors default it to AUTO when neither the spec nor
    the flag names one: every tool offering `--search-procs` — `play.py`
    (hence `./tui.sh`'s play entry and the GUI play launcher, whose field IS
    that flag) and the analysis browser — appends
    `procs=min(worlds, max(1, cpu_count//2))` (`cli_spec.search_knob_pairs` via
    `opponents.default_search_procs`); an explicit `procs=` / `--search-procs`
    / a set launcher field always wins.
- a prebuilt `Controller` instance (passed through)

The human-on-a-board seat is the exception: the TUI and GUI boards
(`tui_game.py` / `gui_game.py`, launched via `play.py --board tui|gui`,
`./tui.sh`, `./gui.sh`) host their own UI-coupled loop (`game_driver.GameDriver`)
and queue human clicks; the *opponent* seat uses the same spec grammar above.

## Scripting games: `runner.run_match`

```python
import runner

# bo3 match, scripted HARD mirror, deterministic, compact transcript
r = runner.run_match("scripted", "scripted", deck_a="league/bug", deck_b="league/bug")

# model vs scripted, 10 bo1 games, no output, per-game records
r = runner.run_match("league/bug", "scripted",
                     deck_a="league/bug", deck_b="league/bug",
                     games=10, bo3=False, transcript="quiet")
print(r.win_rate, r.records[0].engine_seed, r.records[0].actions)

# drive one seat through a fixed line, sculpted state (harness-style kwargs)
r = runner.run_match("play:A:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp",
                     "auto", bo3=False,
                     battlefield_a="Mountain", battlefield_b="Grizzly Bears",
                     max_decisions=40)
```

Defaults: **bo3 on**, `seed=1` (game *i* uses `seed+i`; `seed=None` for
random), compact transcript. `transcript=` one of `"verbose"` (full board +
menu per decision), `"compact"` (one line per decision), `"narrative"`
(engine narrative + results only — human play), `"quiet"` (nothing; draws
still dump `draw_<n>.txt`). `out=` redirects the transcript to any stream.
Returns a `MatchResult` (W/L/D from seat A, `win_rate`, per-game
`GameRecord`s with reward, decision count, engine seed, and the full action
log — enough to replay).

Zone presets, `no_shuffle`, `life_a/b`, `log_decisions`, and `coverage` pass
through to `run_games` (same kwargs the test harness uses).

For custom instrumentation, drop one layer to `runner.drive_game(env, obs,
ctrl_a, ctrl_b, on_query=..., on_action=..., max_decisions=...)` — the hooks
receive a `Decision` context (obs, num_choices, priority seat, lazily-decoded
menu). This is how `analysis.py` records traces and `bench_engine.py`
benchmarks; there is no other decision loop in the tree.

## The tools and where they sit

Every Python tool that plays games takes the same match-format flag,
`--format bo1|bo3`, defaulting to **bo3** (the old `--bo1`/`--bo3` switches
error with a pointer to it). A game/match count is always `--games` (whole
matches under bo3; each tool's help says what it counts), the PUCT constant is
always `--c-puct`, and `--seed` defaults to **1** on every test, eval and
inspection tool (observe, baseline, az-eval, analysis, the harness, fuzz,
benches, az_inspect) but to a **random, printed** seed on long training runs
(az-selfplay, az-train, az, az-league).

- **`test_harness.py`** — state sculpting (hands/zones/scenarios); a global
  both-seat `--play`/`--actions` script makes decisions first, then each
  seat's `--player-a/-b` agent spec (default `auto`); `run_games` under the
  hood. Pass `--format bo1` for a single game (sculpted scenarios usually
  want one).
- **`train.py observe`** — per-seat agent specs (`--player-a/-b`, including
  `play:<specs>` scripts) and decks (`--deck-a/-b`), any matchup. bo3 by
  default; `--format bo1` for single games.
- **`train.py baseline`** — the AZ generalist (`az:gen` = `gen__azfinal.pt`)
  under the full league search budget (1028 sims × 8 worlds) vs scripted
  **HARD** over the whole league grid (every deck piloted vs every deck, mirrors
  included), 10 bo3 matches per matchup, seats alternating. Runs on the C++
  actor (`bin/az_actor --search` eval mode + the scripted oracle + the GPU eval
  server, 48 legs in flight) — NOT on `drive_game`; PPO `.zip` models,
  `mcts:` specs and `--no-actor` fall back to the runner-based
  `train.baseline_sweep`. `--player-a` is the model under test (player A
  is the side under test, not a fixed physical seat), `--deck-a` narrows to
  one piloted deck (mirror unless `--deck-b`), `--games`/`--sims`/
  `--worlds`/`--workers` scale it, `--seed` reproduces. Every run appends its report to `checkpoints/baseline_report.log`
  (override with `--log`), and the actor path records the net's searched
  decisions as shards under `az_data/baseline/baseline_<stamp>/` (analyzable
  with `az_inspect`/the shard browsers; never pooled into training;
  `--no-record` to skip). Implementation: `train/az_baseline.py`.
- **`play.py`** — human vs model: `--player-a`/`--player-b` are agent specs,
  exactly one of them `human` (default: human on A vs `az:gen` on B),
  with `--deck-a`/`--deck-b`. `--board text` = `run_games` with a
  `HumanController` seat (semantic input) vs `make_controller(opponent)`;
  `--board tui|gui` (default gui) = the GameDriver boards.
- **`fuzz_campaign.py`** — explore-tier fuzz sweeps for one matchup;
  `run_games` verbose transcripts to a file.
- **`ci_check.py`** — the `make check` gate; league smoke + fuzz tiers run
  through `run_games`, replay corpus through the engine's `--replay`.
- **`analysis.py`** — trace collection / counterfactual rollouts on
  `drive_game` hooks; replays via recorded engine seed + action log.
- **`bench_engine.py`** — throughput benchmark on `drive_game`.

The torch training loop (`train.py train/league/sweep`, the vectorized env
wrappers `ModelVsScriptedEnv`/`SelfPlayEnv`/`FixedModelEnv`) is a separate
path by design and is not routed through the runner.
