# Running games: agents, drivers, scripts and logs

Who decides each seat, the in-process agents, the human CLI, the Python driver, `--play` /
`--actions` scripts, the machine protocol and decision logs. Engine commands run from `bin/` (the
engine finds `resources/` relative to its working directory); `./debug/robomage --help` lists every
flag.

## Who decides a seat

Each decision belongs to one seat (the priority holder, or the sideboarding player between games).
In order of precedence it goes to:

1. **`--replay LOG`** — every decision comes from the log; nothing else is consulted.
2. **A global script** — `--play <specs>` or `--actions <i,i,…>`, for both seats until it runs out.
3. **The seat's in-process agent** — `--agent-a/-b <spec>`; with a script, a seat without one takes
   `first`.
4. **A human seat** — the C++ CLI: the `--player A|B` seat, and without `--machine` every seat with
   no agent (so a plain run is hotseat). It takes over a script's seat once the script runs out.
5. **The RMSTATE stream** — under `--machine`, every other seat: frames on stdout, one integer reply
   per decision on stdin (how an external agent such as the Python driver plays).

`--agent-a/-b`, `--play` and `--actions` imply `--machine` and cannot be combined with `--replay` or
with an agent on the `--player` seat. A human seat reads stdin, so with one no seat may be left to
the stream (`--player A --machine` with no agent for B is refused). `--machine` changes only I/O:
which decisions are asked and what each menu offers are the same for every consumer.

**Auto-mana** is on by default: ordinary mana abilities are not offered at priority (instant-speed
sources and sources targeted by removal on the stack are, so mana can be floated), costs are paid by
the greedy payer and hybrid pips resolve automatically. `--no-auto-mana` (whole match, recorded in
the decision log) offers every mana ability at priority and pays a cost the pool doesn't cover by
choosing sources one at a time, with a last row that cancels the payment and reverses the spell or
ability.

Every game prints `GAME_RESULT: N Player A|B wins` (or `GAME_RESULT: N draw`, CR 104.4a); a bo3
match ends with `MATCH_RESULT: Player A|B wins X-Y`. Under the stream these, like all other output,
go to stderr.

## Playing as a human: the C++ CLI

```bash
cd bin
./debug/robomage --player A --agent-b explore --deck-a league/burn --deck-b league/bug   # vs an agent
./debug/robomage --deck-a league/burn --deck-b league/bug --bo3                          # hotseat
```

A human seat (`src/cli/human.h`) is an in-process agent: it renders only its seat's decoded RMSTATE
frames, knowledge tables included (`src/cli/board.h`), so it shows exactly what that seat may see.
No perfect-information narrative is printed unless `--narrative` is given (with `--player`, its
one-sided lines limited to that seat's view). A human game always writes its decision log
(`Decision log: <path>` printed at the start). At each decision the seat sees what happened since
its previous decision (events, life changes), the board when it changed (players, battlefields with
P/T, statuses, counters, attachments and combat, the stack top first with targets, its hand,
graveyards, exile, open reveals, effects, a "You know" knowledge summary), the pending decision
(with what a cost still owes, by pip, during payment) and the numbered menu. Objects are labelled
`#<handle>` (the seat frame's handle). Hotseat prints `==== Player X to decide ====` before each
decision. Commands:

| Command | |
|---|---|
| `<n>` | menu row *n*; `-1` is the last row (an attacker / blocker declaration's confirm, marked `(or -1)`) |
| `concede` / `concede match` | concede the game (`-2`) / the match (`-3`) |
| `board` | the board and menu again |
| `zone <name> [opp]` | one zone, yours or the opponent's: `hand`, `library`, `graveyard`, `exile`, `battlefield`, `stack`, `sideboard`; a hidden zone shows its size, the cards the frame shows in it and what the knowledge tables say |
| `card <#n\|name>` | a card's details from the `CARDS` frame (cost, type line, P/T, oracle text): a board object by handle, or a card of the match by name |
| `know` | the full knowledge panel: pools with each card's least and most copies, known library positions, copies seen; off under `--knowledge-a/-b off` |
| `log [n\|all]` | this seat's events this match, the last *n* lines (default 30) |
| `z [mine]` (`pass turn [mine]`) | auto-pass: answer "Pass priority" at this seat's priority decisions until the next turn begins (`mine`: this seat's own next turn), printing what happens and one line per pass |
| `help`, `quit` | |

Auto-pass reads only the seat's frames. It starts only at a plain priority decision (a pass row, no
pending decision or cost) and stops, prompting as usual, at the first decision of the turn it waits
for, at any other decision (mandatory choice, attackers or blockers, target, payment, mulligan, …)
and when the opponent puts a new object on the stack.

Anything else re-prompts. End of input ends the run (exit 0), as `quit` does; the decision log then
ends with a `QUIT` line, and `--replay` of it stops there with `QUIT: the recorded session quit at
decision N (no result)`, exit 0.

## The engine's in-process agents (`--agent-a/-b`)

The agents (`src/agents/`) run on the engine's decision hook and read only the deciding seat's
decoded RMSTATE frames (the stream's bytes, knowledge tables included, handed over in memory by
`src/state_format/agent_feed.h`):

| Spec | Agent |
|---|---|
| `first` (`auto`) | always row 0 (pass priority, the first choice) |
| `random` | a uniform row |
| `explore` | the coverage fuzzer: novelty-weighted choices (`src/agents/explore.h`, a port of `python/explore_agent.py`) |
| `explore:patient` | its big-mana profile (holds cheap activations, favours expensive casts) |
| `actions:<i,i,…>` | this seat's replies in order (a `-2` / `-3` sentinel sent as is), then row 0 |
| `play:<specs>` | the `--play` resolver on this seat's decisions only, then row 0 |

All agents of the process draw from one generator, a bit-exact port of Python's `random.Random`
(`src/agents/py_random.h`), seeded once before the match. `--game-seed N` is the game the Python
driver and the scenario files call seed N: the engine seed derived from it (`src/agents/game_seed.h`,
as `rm_driver.engine_seed`) and the generator seeded with N, so C++ explore plays the same games as
Python explore. `--seed N` sets the raw engine seed instead (the generator then gets N).

```bash
cd bin
./debug/robomage --agent-a explore --agent-b explore --game-seed 5 \
  --deck-a league/bug --deck-b league/burn                          # one game, results only
./debug/robomage --agent-a explore:patient --agent-b explore:patient --game-seed 5 --bo3 \
  --deck-a league/bug --deck-b league/burn --narrative --trace-decisions   # a bo3 match, every decision
```

Output flags: `--narrative` prints the perfect-information game log (silent under `--machine`
without it); `--log-viewer A|B` restricts its one-sided lines to one seat's view;
`--trace-decisions` prints each decision's board, pending decision, menu and the choice made, as the
CLI renders the deciding seat's frame, on `| ` lines; `--max-decisions N` stops before decision N+1
(`MAX_DECISIONS:` line, exit 0, no result for that game).

## Scripts: `--play` and `--actions`

`--play` is a comma-separated list of semantic specs (`keep`, `cast:Lightning Bolt`,
`target:Grizzly Bears@opp`, `divide:Grizzly Bears@opp`, `A:attack:Delver`, `sb-out:Daze`,
`concede`, …), each resolved against the live menu of its decision; an unmatched or ambiguous spec
stops the run with `FATAL: --play: …` and the legal menu. Seat keys `A:` / `B:` let each player's
line be written without sequencing the other seat's passes. Grammar: `src/agents/play_spec.h`
(tabulated with the seat-key rules in `CLAUDE.md`, "Testing card behavior"). `--actions` is a list
of raw menu indices (fragile: breaks when a menu changes).

At exit a `--play` run prints `resolved --actions: <i,i,…>` (every reply given, replayable as
`--actions` with the same seed) and, when specs were left unapplied, `SCRIPT INCOMPLETE: …`.

Test-setup presets sculpt positions: `--hand-a/-b` + `--library-a/-b` (a stacked deck, hand first;
implies `--no-shuffle`), `--battlefield-a/-b`, `--face-down-a/-b` (on the battlefield face down:
2/2s with no text, CR 708.2a), `--graveyard-a/-b`, `--exile-a/-b`, `--sideboard-a/-b`,
`--life-a/-b`; `--offer-cancel` adds a "Cancel" row to every cast and activation prompt (CR 733.1).

```bash
cd bin
./debug/robomage --narrative --trace-decisions --game-seed 1 --max-decisions 10 \
  --hand-a "Lightning Bolt" --battlefield-a "Mountain" --battlefield-b "Grizzly Bears" \
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp"
```

The rules-regression scenarios (`tests/scenarios/*.json`) are such runs with assertions
(`bin/debug/robomage_check --tier scenarios --scenario <glob>`, `docs/ci.md`).

## The scripted agent: `python/rm_driver.py`

The rule-based scripted agent (`python/scripted_core.py`, over one seat's decoded frames via
`python/rm_agent.py`) is played by `python/rm_driver.py` (stdlib only): it launches the engine with
`--machine`, decodes each seat's frames with `python/rmstate.py` and writes back the deciding seat's
agent's integer.

| Spec | Agent |
|---|---|
| `scripted` (`hard`, `heuristic`, `scripted:hard`) | the hard tier: smart mulligan, combat simulation, combo lines; Done at every sideboarding phase |
| `scripted:hard+sb` (`hard+sb`) | the hard tier, sideboarding between bo3 games per the league guides (`docs/sideboard_guides.md`, `python/sideboard_guides.py`): both decks identified from the registered lists, the matchup's target configuration for the coming game on the play or draw; Done for an unknown deck or a held-out matchup |
| `scripted:easy` / `greedy` (`easy`) | the greedy tier |
| `scripted:random` (`random`) | uniform random |
| `explore` (`fuzz`), `explore:patient` (`patient`) | the coverage fuzzer (`python/explore_agent.py`; the same games as the C++ port for the same seed) |
| `first` (`auto`) | action 0 |
| `actions:<i,i,…>` | this seat's replies in order (`-2` / `-3` concede), then action 0 |

`play:` scripts run in the engine (`--play`, `--agent-a play:…`), not the driver, so they cannot be
combined with the driver's agents.

```bash
python3 python/rm_driver.py --player-a scripted --player-b explore \
  --deck-a league/bug --deck-b league/burn --format bo1 --games 5 --verbose
python3 python/rm_driver.py --deck-a league/ur_tempo --deck-b league/eldrazi_aggro \
  --games 20 --quiet --timing                                        # 20 bo3 matches, summary only
```

Flags (`rm_driver.CLI_FLAGS`; `--help`): `--player-a/-b` (default `scripted`), `--deck-a` (default
`delver`) / `--deck-b` (default: A's deck), `--format bo1|bo3` (default **bo3**), `--games N`
(matches under bo3), `--seed N` (default 1; game *i* uses `--seed + i`), `--on-the-play a|b|random`
(b swaps the two sides' agents and decks between engine seats, since seat A starts game 1),
`--verbose` / `--quiet` / `--out FILE`, `--max-decisions N`, `--timing` (throughput; with `--quiet`
the engine runs without narrative), the engine's `--closed-decklist`, `--no-auto-mana`,
`--selfcheck-state`, `--selfcheck-knowledge`, `--knowledge-a/-b`, `--rmstate-dump FILE` (`FILE.<i>`
per match when `--games > 1`), `--rmlog-dir DIR` (each game's, under bo3 each match's, decision log
as `DIR/<deck a>-<deck b>.<bo1|bo3>.seed<seed>.rmlog`), `--engine-flags "…"` for any other engine
flag (`$ROBOMAGE_ENGINE_FLAGS` is added too), and `--binary` (default `bin/$ROBOMAGE_BUILD/robomage`,
debug when unset).

**Seeds.** Before game *i* the driver seeds Python's `random` with `seed + i` (scripted tie-breaks,
explore's draws) and passes the engine `--seed engine_seed(seed + i)`, so a seed names the same game
as the engine's `--game-seed`.

**Transcripts.** `verbose` (the smoke tier's): each decision's narrative in a `--- Narrative ---`
block, `--- Pregame ---` / `--- Turn N (Player X) ---` markers with `PA:` / `PB:` hand lines,
`--- Decision N ---` with the menu as offered and `>> label/seat: i (description)`, then the result.
The default (`compact`): the narrative and one line per decision; `--quiet`: a W/L/D summary line
only. Every draw (a drawn game, or a game with no result) is announced and its log saved to
`draw_<stamp>.txt`, even when quiet; a run stopped by `--max-decisions` is reported incomplete and
not counted. Engine diagnostics (`WARNING:`, `ERROR:`, `FATAL`, self-check summaries) go to the
driver's stderr; an engine that exits nonzero raises.

**From Python:**

```python
import sys; sys.path.insert(0, "python")
import rm_driver

wins, losses, draws = rm_driver.run_games(
    "scripted", "explore", deck_a="league/bug", deck_b="league/burn",
    n_games=10, bo3=False, seed=1, transcript="quiet", on_game_end=print)
```

`run_games(agent_a, agent_b, *, label_a, label_b, binary, deck_a, deck_b, n_games=1, bo3=False,
seed=None, transcript="compact|verbose|quiet", out=None, max_decisions=None, narrative=True,
engine_flags=(), on_game_end=None, dump=None)` returns Player A's `(wins, losses, draws)` and hands
each `GameRecord` (`reward`, `decisions`, `capped`, `actions` — every reply sent, with
`engine_seed` enough to replay — `engine_seed`, `game_results` as `(winner, a_on_play)`,
`diagnostics`; properties `winner`, `drawn_games`) to `on_game_end`. `play_game(agent_a, agent_b, *,
binary, deck_a, deck_b, bo3, seed, max_decisions, narrative, engine_flags, on_narrative,
on_decision, on_action)` plays one game or match with hooks (`on_decision(d)` gets a `Decision`:
the seat, its decoded frame, the menu and its categories). An agent is any object with `new_game()`
and `decide(seat_frames, frame) -> int` (`make_agent` builds the spec'd ones).

## An external agent on the machine protocol

Under `--machine` every seat without an in-process agent (and not the `--player` seat) decides over
the RMSTATE stream (`docs/state_format.md` §9): stdout carries only `RMSTATE <nbytes>\n` + payload
records; the deciding seat gets its `SCHEMA`, `CARDS` and `DECKLISTS` frames the first time it
decides, then a `STATE` frame per decision, answered by one integer line (a menu row, `-1` = the
last row, `-2` / `-3` = concede game / match); at match end each seat gets a terminal `STATE` frame
(`header.terminal = 1`, no reply) and the engine exits. Everything else goes to stderr. A minimal
random agent:

```python
import random, subprocess, sys
sys.path.insert(0, "python")
import rmstate

eng = subprocess.Popen(["./debug/robomage", "--machine", "--deck-a", "league/bug",
                        "--deck-b", "league/burn", "--seed", "7"], cwd="bin",
                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
dec = rmstate.StreamDecoder()
for payload in rmstate.records(eng.stdout):
    frame = dec.feed(payload)                       # tables: name -> list of row dicts
    if frame.kind != "STATE" or frame.tables["header"][0]["terminal"]:
        continue                                    # once-per-match frames, or the match is over
    eng.stdin.write(b"%d\n" % random.randrange(len(frame.tables["menu"])))
    eng.stdin.flush()
eng.wait()
```

`frame.viewer` is the frame's seat; its knowledge is in the `k_*` tables
(`docs/knowledge_store.md`). The C++ decoder is `src/state_format/decode.h` (std-only).

## Decision logs and replay

`--decision-log FILE` writes the RMLOG v3 log: `SEED`, `FLAGS` (the setup flags that change the
game), both decklists, then every asked decision, concession sentinels included. A one-row menu is
taken by the engine unasked and not logged (`docs/state_format.md` §9); `--replay` refuses an RMLOG
v2 log (re-record it). `--log-decisions` writes to `resources/logs/game_<seed>.log` (where
interactive games always log).

`--replay FILE` replays deterministically: the header alone fixes the setup (decks, seed, `--bo3`,
`--no-shuffle`, `--no-auto-mana`, the test-setup presets, `--max-decisions`); setup flags given
alongside are ignored with a warning; output flags (`--machine`, `--narrative`, `--trace-decisions`,
`--rmstate-dump`, `--knowledge-a/-b`, the self-checks, …) apply. A plain `--replay` prints the
narrative and each replayed input (no board); under `--machine` only what the output flags ask for.
`--player` / `--log-viewer` limit the narrative's one-sided lines to that seat's view. A logged
index outside the live menu stops with `replay diverged at decision N`; a log ending before the game
does is fatal (`Replay file ended unexpectedly`) unless it ends with a human seat's `QUIT` line
(clean stop, `QUIT: ...`, exit 0).

```bash
cd bin
./debug/robomage --agent-a explore --agent-b explore --game-seed 5 \
  --deck-a league/bug --deck-b league/burn --decision-log /tmp/g5.rmlog
./debug/robomage --replay /tmp/g5.rmlog --machine --narrative --trace-decisions
```

The replay corpus (`tests/corpus/`) is such logs from scripted-agent games (and two explore games
under `--no-auto-mana`), with their replay output as byte-for-byte baselines (`docs/ci.md`).

## Trajectories

`--trajectory-out DIR` writes one episode per game (each recorded seat's featurized decisions, the
behaviour distribution, the reply, the result) and keeps the decision log; `--replay LOG
--trajectory-out DIR` re-featurizes a logged game (e.g. a `rm_driver.py --rmlog-dir` corpus).
Format and flags: [`trajectory_format.md`](trajectory_format.md).

`--teacher-a/-b <command>` streams a seat's `--machine` frames to a teacher process whose reply at
each decision is recorded as the trajectory's `teacher` label; `--teacher-mix p` plays the label
with probability p. The scripted agent is the teacher: `python3 ../python/teacher.py [--tier hard]
[--seed N]`.

```bash
cd bin
# explore decides seat A; scripted:hard labels every A decision; half of them its label is played
./release/robomage --agent-a explore --agent-b explore --deck-a league/bug --deck-b league/burn \
  --bo3 --game-seed 7 --teacher-a "python3 ../python/teacher.py --seed 7" --teacher-mix 0.5 \
  --trajectory-out /tmp/episodes
```

With `--teacher-mix 1`, `--teacher-a "python3 ../python/teacher.py --seed N"` and `--game-seed N`
the game is the one `rm_driver.py --player-a scripted --seed N` plays (as long as the other seat
draws nothing from Python's `random`). Semantics, μ under mixing and failures:
[`trajectory_format.md`](trajectory_format.md) §1.1, `src/agents/teacher.h`.

## The continuous actor: `--actor`

One engine process plays match after match: `--actor` draws each match (decks, format, agents,
teachers, seats, purpose) from a schedule file re-read between matches, derives its seeds from (run
seed, actor id, match index), writes each match's decision log (and, for training, its episodes) and
reports through a status file and a per-game results file; SIGTERM stops it cleanly.
`--max-game-decisions N` (recorded in the decision log) ends a runaway game with end reason `cap`.

```bash
cd bin
./release/robomage --actor --actor-schedule RUN/schedule.json --actor-out RUN/actors --actor-id 3 --actor-seed 42
```

Details: [`actor.md`](actor.md).

## State dumps

`--rmstate-dump FILE` writes every viewer's frames (A, B, omniscient) at each decision beside any
run (`%p` in FILE = the process id), and a terminal frame each at match end; `--state-dump json`
adds `FILE.jsonl`, one JSON line per frame. `--rmstate-read FILE` decodes, re-encodes and compares a
stream (with `--state-dump json`, its JSON lines on stdout); `python3 python/rmstate.py dump FILE`
prints the same lines from Python; `bin/<config>/rmknow --viewer A --in FILE --text` runs the
knowledge sidecar over a seat's frames.

## Coverage: `--coverage`

`--coverage FILE` counts which engine paths a run reaches and writes them at process exit,
atomically, as sorted `<kind> <name> <count>` lines (`src/coverage.h`). It works beside any run,
`--replay` included, and changes nothing else (output, decision log, frames, RNG); a game and its
replay give the same counts. A run ended by `FATAL:` still writes; one killed by a signal does not.
Counts span the whole process (a bo3 match, an `--actor` run), and work a snapshot restore repeats
is counted again.

| Kind | One hit when | Name |
|---|---|---|
| `effect` | an ability finishes its dispatch through `effects::handler_for` (a suspended handler counts once, when its re-entry finishes) | the category (`ChangeZone`), `None` for an ability with none; a mana ability activated off the stack does not resolve through it |
| `trigger` | a triggered ability goes on the stack, or resolves at once if it uses no stack (`Static$ True`, `TapsForMana`) | the `T:` line's `Mode$` (`ChangesZone`); else `Delayed`, its keyword or engine key (`keyword:Prowess`, `triggered:SuspendTick`), or `Reflexive` |
| `replacement` | `replacement::dispatch` applies one to a real event (not a mana simulation), or a dredge replaces a draw | `SELF_TAPPED`, `PENDING_TAPPED`, `ETB_COUNTERS`, `EXILE_INSTEAD`, `EXILE_INSTEAD_OF_ETB`, `SKIP_UNTAP`, `PREVENT_ETB`, `PRODUCE_MANA`, `DISCARD_ELSE_GRAVEYARD`, `DREDGE`; draw additions and can't-be-countered do not pass through the dispatcher and are not counted |
| `category` | a decision is asked and answered (the sum is the decision log's length less its concessions) | the taken row's `ActionCategory`, by its SCHEMA name (`SACRIFICE_PERMANENT`) |
| `cost` | a cast or activation pays this part at its pay step | the part's kind (`CostPart::Kind`): `MANA`, `TAP`, `LOYALTY`, `LIFE`, `ENERGY`, `SACRIFICE`, `RETURN`, `DISCARD`, `EXILE`, `EXILE_FROM_HAND`, `EXILE_FROM_GRAVEYARD`; with a form, `LIFE.x`, `LIFE.phyrexian`, `SACRIFICE.self`, `DISCARD.self`, `DISCARD.hand`, `EXILE_FROM_GRAVEYARD.delve`, `MANA.tap_to_pay` (a pip paid by tapping a permanent, once per tap); `DISCARD.trigger` and `MANA.trigger` (a triggered ability's offered cost), `EXILE.self`, `EXILE_FROM_HAND.self`, `EXILE_FROM_GRAVEYARD.self` (the source exiling itself), `LIFE.additional`, `DISCARD.additional` (an additional cost announced), `MANA.hybrid` (hybrid pips announced); a mana ability the payer taps by itself, unless costs and costs of special actions are not counted |
| `row` | a filter or SVar table row is read that no other kind tells from its neighbours | `leastPower`, `without`, `TargetedPlayerOwn`, `TargetedPlayer`, `DivideEvenlyDown`, `sunburst` (a Sunburst entry that counted a color), `mana_rider_keyword` (a spell gained a rider's keyword) |
| `cast` | a spell becomes cast, once per way its `CastOption` departs from a cast of the mana cost | `flashback`, `escape`, `alt_cost`, `offspring`, `impulse`, `without_mana_cost`, `back_face`, `split_half`, `fused`, and its overlay's name (`overload`, `adventure`, `bestow`, `prepared`, `face_down`); with none, `hand` or `graveyard_permission`. Special actions too: `suspend`, `companion`, `land_back_face`, `turn_face_up` |
| `cast_offer` | a row of the cast table (`src/cast_options.cpp`) is offered at priority | the row's name (`hand`, `evoke`, `graveyard_free`, `exile_land`) |
| `cast_row` | a spell becomes cast by that row, or a land is played from another zone | the row's name, `graveyard_land`, `exile_land`; an exile grant's cast also `grant_free`, `grant_energy` or `grant_life` |

The `fuzz` tier uses the counts to check that each fixture deck's games reached their paths
(`docs/ci.md`).

## Checking decks and card scripts (no game)

`--check-decks` checks that every card of the playable decks (top-level, `league/`, `heldout/`) has
a script and is in the card vocab, one `check-decks: MISSING …` line per card that is not;
`--check-decks-dir DIR` checks `resources/decks/DIR` instead (repeatable; a path ending in `.dk` is
that one deck).

`--parse-report <card | @vocab | @FILE>` (repeatable) parses each card's script afresh and prints
one JSON line per card with what the parser does not model: ability categories with no handler (they
resolve as no-ops), params no branch reads, trigger and static modes never applied, replacement
lines that yield nothing, keywords no rule reads, cost tokens, filter qualifiers and SVar
expressions with no reading, SVars and token scripts the card names but lacks. `@vocab` is every
vocab card, `@FILE` a `.dk` deck's cards or one card name per line. A clean card is not proven to
play correctly; a finding names something the engine does not do. Kinds and JSON shape:
`src/parse_report.h`.

```bash
cd bin
./debug/robomage --check-decks --check-decks-dir heldout/storm.dk
./debug/robomage --parse-report "City of Traitors" --parse-report @../bin/resources/decks/heldout/storm.dk
./debug/robomage --parse-report @vocab | python3 -c "import json,sys; [print(r['card'], f['kind'], f['name']) for r in map(json.loads, sys.stdin) for f in r.get('findings', [])]"
```

The vocab's known findings are committed in `tests/parse_report/vocab.json` (the `vocab` tier fails
on a new or vanished one); the onboarding pipeline (`tools/onboard/README.md`) runs both modes over
scraped decks.
