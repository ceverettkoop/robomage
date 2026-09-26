# Running games: modes, agents, I/O

How a RoboMage game is run outside the torch training loop: what the engine provides, the agent
spec grammar, and the one Python API to script games with.

## The engine has no agents

`bin/<config>/robomage` knows nothing about scripted tiers, models or humans. In `--machine` mode
it emits a `BQUERY` frame on stdout at every decision, for **both** seats in priority order, and
reads one integer back on stdin. Who picks that integer is the driver's concern. The only
engine-side seat options are `--player A|B` (routes one seat to the engine's own interactive stdin
prompt; no Python tool uses it) and `--log-viewer A|B` (redacts private narrative to one seat's
view without rerouting input; set by the TUI/GUI play boards).

Other engine flags (`src/main.cpp` argv loop): `--deck-a/-b` (or `--deck` for both), `--seed`,
`--no-shuffle`, zone presets `--battlefield/graveyard/exile/sideboard-a/-b`, `--life-a/-b`,
`--narrative` (full game log + per-action description side-channels), `--bo3` (match loop:
per-game seed = base + game, loser goes first, sideboarding, `GAME_RESULT:`/`MATCH_RESULT:`
lines), `--replay <rmlog>` (deterministic replay), `--log-decisions` (write the replay log in
machine mode), `--search-server` (snapshot/restore/determinize protocol for MCTS; implies
`--machine`), `--broadcast-steps` (passive `BSTATE` frames at auto-passed steps, for the GUI's
step pacing).

## The Python stack (`train/`)

| Layer | Module | Role |
|---|---|---|
| Engine wrapper | `env.py` (`RoboMageEnv`, `NarrativeEnv`); `search_env.py` (`SearchNarrativeEnv`) | subprocess launch, BQUERY parse, obs, result/reward, seeds; the search variant is swapped in when a controller sets `wants_search_env` |
| Decoding | `decode.py` | state vector / action menu → readable structures and transcript blocks |
| Agents | `opponents.py` (`Controller`, `make_controller`) | every agent kind below |
| Loop | `runner.py` (`drive_game`) | THE decision loop: priority seat → controller, step, hooks |
| Orchestration | `runner.py` (`run_games`, `run_match`) | env per game, transcripts, tallies, records |

## Agent specs (one grammar everywhere)

`opponents.make_controller(spec)` backs every `--player-a/-b` flag (harness, observe, baseline,
play.py, analysis, the TUI/GUI) and `run_match`. Specs are case-insensitive:

| Spec | Agent |
|---|---|
| `scripted`, `hard`, `heuristic`, `scripted:hard` | scripted HARD tier (what a bare `scripted` means everywhere) |
| `easy`, `greedy` (or `scripted:easy`…) | greedy tier |
| `random` | uniform random |
| `explore`, `fuzz` | coverage fuzzer (vary `--seed`) |
| `explore:patient`, `patient` | fuzzer's big-mana profile |
| `auto`, `autopass` | always action 0 (pass / first choice) |
| `human` | terminal seat: prints board + menu, takes an index or a semantic spec (`cast:bolt`, `target:bears@opp`, `pass`), `quit`/`q`/`exit` to leave |
| `play:<spec,spec,…>` | semantic action script (the harness `--play` grammar, `action_spec.py`); once exhausted, action 0 |
| `actions:<i,i,…>` | positional action indices |
| `gen`, or a checkpoint path / name | PPO model (`ModelController`) via `opponents.resolve_checkpoint`: `gen` → `checkpoints/gen__final.zip`, else newest `gen__v*.zip`; a name is looked up in `checkpoints/` as-is, `.zip`, `_final.zip`. A bare deck name (`delver`) is rejected: the deck is always a separate parameter |
| `mcts:<gen\|path\|uniform>[?knobs]` | PUCT search with a PPO net's policy/value (`uniform` = torch-free uniform evaluator, for plumbing tests) |
| `az:<gen\|path>[?knobs]` | PUCT search with an AZNet: `az:gen` → `gen__azfinal.pt`, else newest `gen__azv*.pt`, else a warm start from the `gen` PPO net; a `.pt` path loads directly, a `.zip` warm-starts |
| `azraw:<gen\|path>` | the AZNet's argmax policy, no search (knobs are not parsed) |
| a `Controller` instance | passed through |

Which checkpoint a spec names for analysis purposes (V(s), probes, replay search) is resolved by
`opponents.parse_model_spec` and its `load_spec_*` loaders; do not strip prefixes by hand.

### Search knobs (`mcts:` / `az:`)

Query string `?k=v&k=v` (later keys win); a malformed value fails naming the spec.

| Knob | Default | Meaning |
|---|---|---|
| `sims` | 128 | in-game simulations per decision, split across worlds |
| `worlds` | 4 | determinized worlds |
| `c` | `DEFAULT_AZ_C_PUCT` (2.5) | PUCT constant |
| `temp` | 0 | root temperature |
| `seed` | 0 | search RNG seed |
| `time` | — | wall-clock seconds per decision (see below) |
| `procs` | 1 | engine processes for world-parallel search (see below) |
| `clock`, `tmin`, `tmax`, `sb_tmax` | —, 0.5, 60, 15 | whole-match chess-clock bank in seconds; each decision draws a variable budget in [`tmin`, `tmax`] (sideboard roots capped at `sb_tmax`) |
| `paced` | 0 | human-facing play: small jittered response floor plus occasional fake-think pauses, masking timing tells |
| `xw` | 1 | cross-world batched leaf evaluation (identical visits, pure speed); `xw=0` disables |
| `device` | `ROBOMAGE_EVAL_DEVICE`, else cpu | `az:` only: torch device for the net |
| `vscale` | 1 | `mcts:` only: PPO value tanh scale |
| `sb_branches`, `sb_worlds`, `sb_rollout_turns` | 8, 4, 6 (`cli_spec.DEFAULT_SB_*`) | bo3 sideboard plan-search budget |

- **`time=<s>`** replaces `sims` as the terminator: worlds run round-robin until the deadline, with
  a floor of one simulation per world; `sims`, if explicitly given, becomes a hard cap. Without
  `time=` the fixed-sims path is byte-for-byte unchanged (the actor parity corpus depends on it).
  `play.py --think-time` is the CLI front door. e.g. `az:gen?time=5&worlds=4`.
- **`procs=<n>`** keeps `n-1` mirror engine processes in lockstep with the game and fans the
  worlds across all `n` (about linear speedup for `procs ≤ worlds`); `procs=1` is identical to
  the single-engine search, and self-play / parity never use the pool. The spec default stays 1
  for reproducible gates and eval, but the interactive front doors (`play.py` and so the
  `./tui.sh` play entry and GUI Play dialog, and the analysis browser) add
  `procs=min(worlds, max(1, cpu_count // 2))` when neither the spec nor `--search-procs` sets it
  (`cli_spec.search_knob_pairs` via `opponents.default_search_procs`).

The human seat on a board is the exception to `runner`: the TUI and GUI play boards
(`tui_game.py` / `gui_game.py`, via `play.py --board tui|gui`, `./tui.sh`, `./gui.sh`) run their
own UI-coupled loop (`game_driver.GameDriver`) and queue human clicks; the opponent seat uses the
spec grammar above.

## Scripting games: `runner.run_match`

```python
import runner

# bo3 match, scripted HARD mirror, deterministic, compact transcript
r = runner.run_match("scripted", "scripted", deck_a="league/bug", deck_b="league/bug")

# PPO generalist vs scripted, 10 bo1 games, no output
r = runner.run_match("gen", "scripted", deck_a="league/bug", deck_b="league/bug",
                     games=10, bo3=False, transcript="quiet")
print(r.win_rate, r.records[0].engine_seed, r.records[0].actions)

# one seat through a fixed line, sculpted state (harness-style kwargs)
r = runner.run_match("play:keep,cast:Lightning Bolt,target:Grizzly Bears@opp", "auto",
                     bo3=False, battlefield_a="Mountain", battlefield_b="Grizzly Bears",
                     max_decisions=40)
```

```python
run_match(agent_a="scripted", agent_b="scripted", *, deck_a=None, deck_b=None,
          games=1, bo3=True, seed=1, transcript="compact", out=None,
          binary_path=BINARY, deterministic_models=True, checkpoint_resolver=None,
          max_decisions=None, **run_games_kwargs) -> MatchResult
```

- `seed`: game *i* uses `seed + i`; `seed=None` is random. Python's `random` is seeded the same,
  so scripted tie-breaks replay too.
- `transcript`: `"verbose"` (narrative + board + menu per decision), `"compact"` (narrative + one
  line per decision), `"narrative"` (engine narrative and results only, for a `human` seat),
  `"quiet"` (nothing). `out=` redirects to any stream.
- Every draw (no winner, e.g. the engine's step cap) saves its full log to
  `draw_<timestamp>.txt` in the cwd, even when quiet. A game stopped by `max_decisions` is
  reported incomplete and not counted.
- Extra kwargs pass to `run_games`: zone presets (`battlefield_a`, `graveyard_b`, `exile_a`,
  `sideboard_b`, …), `no_shuffle`, `life_a/b`, `log_decisions`, `coverage` (a
  `coverage_report.CoverageAccumulator`), hooks `on_query` / `on_action` / `on_game_end(record)`,
  and `narrative=False` (engine without its game log; the benchmark path, only sensible quiet).
- Returns `MatchResult`: `wins` / `losses` / `draws` from seat A, `games`, `win_rate`,
  `label_a/b`, and `records` — one `GameRecord` per game (bo1) or match (bo3) with `reward`,
  `decisions`, `capped`, `actions` (every index sent; with `engine_seed`, enough to replay),
  `engine_seed`, `winner`, and under bo3 `game_results` (per-game `(winner, a_on_play)`).

`run_games(controller_a, controller_b, *, …)` is the layer below: it takes `Controller` objects
(pass one object twice for a single global decision-maker, as the harness does), `n_games`, and
defaults to `bo3=False`, `seed=None`; it returns `(wins, losses, draws)`.

For custom instrumentation drop to `runner.drive_game(env, obs, controller_a, controller_b, *,
on_query=None, on_action=None, on_narrative=None, max_decisions=None, coverage=None)` on an env
you have reset (and will close). `on_query(d)` / `on_action(d, action)` receive a `Decision`
(`obs`, `num_choices`, `priority_is_a`, `controller`, `index`, lazily decoded `menu()`). It
returns a `GameRecord`. There is no other decision loop outside the training wrappers and the
play boards.

## The tools and where they sit

Every tool that plays games takes `--format bo1|bo3` (default **bo3**). A count is always
`--games` (whole matches under bo3), the PUCT constant is always `--c-puct`, and `--seed`
defaults to **1** on test, eval and inspection tools (observe, baseline, az-eval, analysis, the
harness, `bench-actor` / `bench-workers`, az_inspect) but to a **random, printed** seed on long
training runs (az-selfplay, az-train, az, az-league); `play.py` is random unless given.

- **`test_harness.py`** (`run_games`) — state sculpting (hands / zones / scenarios). One global
  `--play`/`--actions` script drives both seats first (seat keys `A:`/`B:`), then each seat's
  `--player-a/-b` agent (default `auto`). Use `--format bo1` for a single sculpted game.
- **`train.py observe`** (`run_games`) — any matchup: `--player-a/-b` (default `scripted`),
  `--deck-a` (default `delver`) / `--deck-b` (default: A's deck); game *i* uses `--seed + i`.
  Transcript `--verbose` / compact / `--quiet` (one-line W/L/D only); `--out FILE` writes the
  transcript to FILE with the summary on stdout; `--max-decisions N` caps each game. Two forms:
  - **fuzz campaign** — `--player-a explore --player-b explore` (or `explore:patient`)
    `--verbose --out FILE`; any draw is a finding (also saved to `draw_<stamp>.txt`).
  - **throughput benchmark** — `--timing` prints games, decisions, wall, games/s, decisions/s,
    ms/decision (`matches=` under bo3); with `--quiet` the engine runs without narrative:
    `--format bo1 --games 40 --max-decisions 4000 --quiet --timing`.
- **`train.py az-eval`** — the promotion gate; its Python fallback plays through `run_match`.
- **`play.py`** — one seat `human` (default: human on A vs `az:gen` on B), `--deck-a/-b`.
  `--board text` = `run_games` with a human seat (index or semantic spec; `concede` /
  `concede:match` resign); `--board tui|gui` (default gui) = the GameDriver boards.
- **`ci_check.py`** — the `make check` gate; `smoke`/`fuzz` tiers and the `replay` corpus
  (`regression/replay_diff.py`) run through `run_games`. See [`ci.md`](ci.md).
- **`analysis.py`** — trace collection and counterfactual rollouts on `drive_game` hooks, replayed
  from a recorded engine seed + action log; `gui_session_io.py` replays saved sessions the same
  way.

The torch training loop (`train.py train/league/sweep`, the vectorized wrappers
`ModelVsScriptedEnv`/`SelfPlayEnv`/`FixedModelEnv`) is a separate path by design and does not use
the runner (only its periodic rollout transcripts do, via `run_games`).
