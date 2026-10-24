# Running games: agents, drivers, scripts and logs

How a RoboMage game is run: who decides each seat, the engine's in-process agents, playing as a
human at the terminal, the Python driver for the scripted agent, `--play` / `--actions` scripts,
the machine protocol for an external agent, and decision logs. Every engine command runs from `bin/` (the engine finds
`resources/` relative to its working directory); `./debug/robomage --help` lists every flag.

## Who decides a seat

Each decision belongs to one seat (the priority holder, or the sideboarding player between
games). In order of precedence it goes to:

1. **`--replay LOG`** — every decision comes from the log (below); nothing else is consulted.
2. **A global script** — `--play <specs>` or `--actions <i,i,…>` decides for both seats until it
   runs out.
3. **The seat's in-process agent** — `--agent-a <spec>` / `--agent-b <spec>`; when a script is
   given, a seat without one takes `first`.
4. **A human seat** — the C++ CLI at the terminal (below): the `--player A|B` seat, and without
   `--machine` every seat that has no agent (so a plain run is hotseat). A human seat takes over a
   script's seat once the script runs out.
5. **The RMSTATE stream** — under `--machine`, every other seat: its frames on stdout, one integer
   reply per decision on stdin. This is how an external agent (such as the Python driver) plays.

`--agent-a/-b`, `--play` and `--actions` imply `--machine` and cannot be combined with `--replay`
or with an agent on the `--player` seat. A human seat reads stdin, so with one no seat may be left
to the stream (`--player A --machine` with no agent for B is refused). `--machine` changes only
how decisions are read and written: the decisions themselves (which are asked, what each menu
offers) are the same for every consumer — human seat, stream, in-process agent or replay.

**Auto-mana** is on by default for all of them: ordinary mana abilities are not offered at
priority (instant-speed sources and sources targeted by removal on the stack are, so mana can be
floated), costs are paid by the greedy payer and hybrid pips resolve automatically.
`--no-auto-mana` turns it off for the whole match: every mana ability is offered at priority and a
cost the pool doesn't cover is paid by choosing sources one at a time, with a last row that
cancels the payment and reverses the spell or ability. The flag is recorded in the decision log.

Whatever decides, every game prints `GAME_RESULT: N Player A|B wins` (or `GAME_RESULT: N draw`,
CR 104.4a) and a bo3 match ends with `MATCH_RESULT: Player A|B wins X-Y`. Under the stream those
lines, like all other output, go to stderr.

## Playing as a human: the C++ CLI

```bash
cd bin
./debug/robomage --player A --agent-b explore --deck-a league/burn --deck-b league/bug   # vs an agent
./debug/robomage --deck-a league/burn --deck-b league/bug --bo3                          # hotseat
```

A human seat (`src/cli/human.h`) is an in-process agent like the others: it is handed its seat's
RMSTATE frames (knowledge tables included) and renders only those, decoded (`src/cli/board.h`),
so it shows exactly what that seat may see. No perfect-information narrative is printed unless
`--narrative` is given (with `--player`, its one-sided lines are limited to that seat's view); a
human game always writes its decision log (`Decision log: <path>` is printed at the start; replay
it with `--replay`). At each decision the seat sees what happened since its previous decision
(its frames' events, life changes), the board when it changed (players, battlefields with P/T,
statuses, counters, attachments and combat, the stack top first with targets, its hand,
graveyards, exile, open reveals, effects, a "You know" knowledge summary), the pending decision
(with what a cost still owes, by pip, during payment) and the numbered menu. Objects are labelled
`#<handle>`, the handle of the seat's frame. Hotseat prints a `==== Player X to decide ====`
banner before each decision. Commands at the prompt:

| Command | |
|---|---|
| `<n>` | menu row *n*; `-1` is the last row (an attacker / blocker declaration's confirm, marked `(or -1)`) |
| `concede` / `concede match` | concede the game (`-2`) / the match (`-3`) |
| `board` | the board and menu again |
| `zone <name> [opp]` | one zone, yours or the opponent's: `hand`, `library`, `graveyard`, `exile`, `battlefield`, `stack`, `sideboard`; a hidden zone shows its size, the cards the frame shows in it and what the knowledge tables say |
| `card <#n\|name>` | a card's details from the `CARDS` frame (cost, type line, P/T, oracle text): an object of the board by handle, or a card of the match by name |
| `know` | the knowledge panel in full: pools with each card's least and most copies, known library positions, copies seen; off under `--knowledge-a/-b off` |
| `log [n\|all]` | what this seat saw happen this match (its events), the last *n* lines (default 30) |
| `z [mine]` (`pass turn [mine]`) | auto-pass: answer "Pass priority" at this seat's priority decisions until the next turn begins (`mine`: until this seat's own next turn begins), printing what happens and one line per pass |
| `help`, `quit` | |

Auto-pass reads only the seat's frames. It starts only at a plain priority decision (a pass row,
no pending decision or cost) and stops, prompting as usual, at the first decision of the turn it
waits for, at any other decision (a mandatory choice, attackers or blockers, a target, a payment, a
mulligan, ...) and when the opponent puts a new object on the stack, so the seat can respond.

Anything else re-prompts; end of input ends the run (exit status 0), as `quit` does. The decision
log then ends with a `QUIT` line, so `--replay` of it stops at that decision with
`QUIT: the recorded session quit at decision N (no result)` and exit status 0. `--trace-decisions` prints
the same board, pending decision and menu for every decision as `| ` lines.

## The engine's in-process agents (`--agent-a/-b`)

The agents (`src/agents/`) run inside the engine on its decision hook and read only the deciding
seat's decoded RMSTATE frames (the bytes the stream would send it, knowledge tables included,
handed over in memory by `src/state_format/agent_feed.h`):

| Spec | Agent |
|---|---|
| `first` (`auto`) | always row 0 (pass priority, the first choice) |
| `random` | a uniform row |
| `explore` | the coverage fuzzer: novelty-weighted choices (`python/explore_agent.py` ported, `src/agents/explore.h`) |
| `explore:patient` | its big-mana profile (holds cheap activations, favours expensive casts) |
| `actions:<i,i,…>` | this seat's replies in order (a `-2` / `-3` sentinel is sent as is), then row 0 |
| `play:<specs>` | the `--play` resolver on this seat's decisions only, then row 0 |

Every agent of the process draws from one generator, a bit-exact port of Python's
`random.Random` (`src/agents/py_random.h`), seeded once before the match with the game seed.
`--game-seed N` is the game the Python driver and the scenario files call seed N: the engine seed
derived from it (`src/agents/game_seed.h`, the same derivation as `rm_driver.engine_seed`) and the
agents' generator seeded with N, so C++ explore plays exactly the games Python explore plays for
the same seed. `--seed N` instead sets the raw engine seed (the agents' generator then gets N).

```bash
cd bin
./debug/robomage --agent-a explore --agent-b explore --game-seed 5 \
  --deck-a league/bug --deck-b league/burn                          # one game, results only
./debug/robomage --agent-a explore:patient --agent-b explore:patient --game-seed 5 --bo3 \
  --deck-a league/bug --deck-b league/burn --narrative --trace-decisions   # a bo3 match, every decision
```

`--narrative` prints the perfect-information game log (under `--machine` it is silent without
it); `--log-viewer A|B` restricts the one-sided lines to one seat's view. `--trace-decisions`
prints each decision's board, pending decision and menu as the CLI renders the deciding seat's
frame (`src/cli/board.h`), and the choice made, on lines starting `| `. `--max-decisions N` stops the run before decision N+1
(`MAX_DECISIONS:` line, exit status 0, no result for that game).

## Scripts: `--play` and `--actions`

`--play` is a comma-separated list of semantic specs (`keep`, `cast:Lightning Bolt`,
`target:Grizzly Bears@opp`, `A:attack:Delver`, `sb-out:Daze`, `concede`, …), each resolved
against the live menu of the decision it is applied to; an unmatched or ambiguous spec stops the
run with `FATAL: --play: …` and the legal menu. Seat keys `A:` / `B:` let each player's line be
written without sequencing the other seat's passes. The grammar is in
`src/agents/play_spec.h`, and `CLAUDE.md` ("Testing card behavior") tabulates it with the
seat-key rules. `--actions` is the same as a list of raw menu indices (fragile: it breaks when a
menu changes).

At exit a `--play` run prints `resolved --actions: <i,i,…>` (every reply it gave, so the line can
be replayed as `--actions` with the same seed) and, when some specs were never applied,
`SCRIPT INCOMPLETE: …`.

Scripts combine with the test-setup presets for sculpted positions: `--hand-a/-b` +
`--library-a/-b` (a stacked deck, hand first; implies `--no-shuffle`), `--battlefield-a/-b`,
`--graveyard-a/-b`, `--exile-a/-b`, `--sideboard-a/-b`, `--life-a/-b`; `--offer-cancel` adds a
"Cancel" row to every cast and activation prompt (CR 733.1).

```bash
cd bin
./debug/robomage --narrative --trace-decisions --game-seed 1 --max-decisions 10 \
  --hand-a "Lightning Bolt" --battlefield-a "Mountain" --battlefield-b "Grizzly Bears" \
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Grizzly Bears@opp"
```

The rules-regression scenarios (`tests/scenarios/*.json`) are such runs with assertions; the gate
plays them (`bin/debug/robomage_check --tier scenarios --scenario <glob>`, `docs/ci.md`).

## The scripted agent: `python/rm_driver.py`

The rule-based scripted agent is Python (`python/scripted_core.py`, read through one seat's
decoded frames by `python/rm_agent.py`). `python/rm_driver.py` plays it, with the standard
library only: it launches the engine with `--machine`, decodes each seat's frames with
`python/rmstate.py`, asks the deciding seat's agent for one integer and writes it back.

| Spec | Agent |
|---|---|
| `scripted` (`hard`, `heuristic`, `scripted:hard`) | the hard tier: smart mulligan, combat simulation, combo lines; Done at every sideboarding phase |
| `scripted:hard+sb` (`hard+sb`) | the hard tier, sideboarding between bo3 games per the league guides (`docs/sideboard_guides.md`, `python/sideboard_guides.py`): both league decks identified from the registered lists, the matchup's target configuration for the coming game on the play or on the draw; Done for an unknown deck or a held-out matchup |
| `scripted:easy` / `greedy` (`easy`) | the greedy tier |
| `scripted:random` (`random`) | uniform random |
| `explore` (`fuzz`), `explore:patient` (`patient`) | the coverage fuzzer (`python/explore_agent.py`; the same games as the engine's C++ port for the same seed) |
| `first` (`auto`) | action 0 |
| `actions:<i,i,…>` | this seat's replies in order (`-2` / `-3` concede), then action 0 |

`play:` scripts are not driver agents: they run in the engine (`--play`, `--agent-a play:…`),
which then decides those seats itself, so they cannot be combined with the driver's agents.

```bash
python3 python/rm_driver.py --player-a scripted --player-b explore \
  --deck-a league/bug --deck-b league/burn --format bo1 --games 5 --verbose
python3 python/rm_driver.py --deck-a league/ur_delver --deck-b league/gw_maverick \
  --games 20 --quiet --timing                                        # 20 bo3 matches, summary only
```

Flags (`rm_driver.CLI_FLAGS`; `--help`): `--player-a/-b` (default `scripted`), `--deck-a`
(default `delver`) / `--deck-b` (default: A's deck), `--format bo1|bo3` (default **bo3**),
`--games N` (matches under bo3), `--seed N` (default 1; game *i* uses `--seed + i`),
`--on-the-play a|b|random` (side b swaps the two sides' agents and decks between the engine seats,
since engine seat A starts game 1), `--verbose` / `--quiet` / `--out FILE`, `--max-decisions N`,
`--timing` (throughput; with `--quiet` the engine runs without narrative), the engine's
`--closed-decklist`, `--selfcheck-state`, `--selfcheck-knowledge`, `--rmstate-dump FILE`
(`FILE.<i>` per match when `--games > 1`), `--rmlog-dir DIR` (each game's, under bo3 each match's,
decision log kept as `DIR/<deck a>-<deck b>.<bo1|bo3>.seed<seed>.rmlog`), `--knowledge-a/-b`,
`--engine-flags "…"` for any other
engine flag (`$ROBOMAGE_ENGINE_FLAGS` is added too), and `--binary` (default
`bin/$ROBOMAGE_BUILD/robomage`, debug when unset).

**Seeds.** Before game *i* the driver seeds Python's `random` with `seed + i` (the scripted agent
breaks ties and the explore tier draws from it) and the engine gets `--seed engine_seed(seed +
i)`, so a seed names the same game here as on the engine's `--game-seed`.

**Transcripts.** `verbose` (the smoke tier's): each decision's engine narrative in a
`--- Narrative ---` block, `--- Pregame ---` / `--- Turn N (Player X) ---` markers with `PA:` /
`PB:` hand lines, `--- Decision N ---` with the menu as offered and `>> label/seat: i
(description)`, then the result. The default (`compact`) writes the narrative and one line per
decision; `--quiet` only a W/L/D summary line. Every draw (a drawn game, or a game with no result)
is announced and its log saved to `draw_<stamp>.txt`, even when quiet; a run stopped by
`--max-decisions` is reported incomplete and not counted. Engine diagnostic lines (`WARNING:`,
`ERROR:`, `FATAL`, the self-check summaries) go to the driver's stderr; an engine that exits
nonzero raises.

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
each `GameRecord` (`reward`, `decisions`, `capped`, `actions` — every reply sent, enough with
`engine_seed` to replay — `engine_seed`, `game_results` as `(winner, a_on_play)`, `diagnostics`,
and the properties `winner` and `drawn_games`) to `on_game_end`. `play_game(agent_a, agent_b, *, binary, deck_a, deck_b, bo3, seed,
max_decisions, narrative, engine_flags, on_narrative, on_decision, on_action)` is one game or match
with hooks (`on_decision(d)` gets a `Decision`: the seat, its decoded frame, the menu and its
categories). An agent is any object with `new_game()` and `decide(seat_frames, frame) -> int`
(`make_agent` builds the spec'd ones).

## An external agent on the machine protocol

`--machine` makes every seat without an in-process agent (and not the `--player` seat) decide
over the RMSTATE stream (`docs/state_format.md` §9): stdout carries only `RMSTATE <nbytes>\n` +
payload records; the deciding seat gets its `SCHEMA`, `CARDS` and `DECKLISTS` frames
the first time it decides, then a `STATE` frame per decision, answered by one integer line (a menu
row, `-1` = the last row, `-2` / `-3` = concede game / match); at match end each seat gets a
terminal `STATE` frame (`header.terminal = 1`, no reply) and the engine exits. Everything else the
engine prints goes to stderr. A minimal random agent with the stdlib decoder:

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

`frame.viewer` says which seat a frame is for; the seat's knowledge (what it remembers and can
deduce) is in the frame's `k_*` tables (`docs/knowledge_store.md`). In C++ the same decoding is
`src/state_format/decode.h` (std-only), as the in-process agents and the gate use it.

## Decision logs and replay

Every decision is logged as an integer. `--decision-log FILE` writes the RMLOG v3 log (`SEED`,
`FLAGS` — the setup flags that change the game — both decklists, then every decision, concession
sentinels included). A decision whose menu has one row is taken by the engine and never asked, so
it is not logged (`docs/state_format.md` §9); an RMLOG v2 log, written before that rule, is refused
by `--replay` (`Replay log … is RMLOG v2`: re-record it); `--log-decisions` writes it to `resources/logs/game_<seed>.log` (interactive
games always log there). `--replay FILE` replays it deterministically: the header alone fixes the
setup (decks, seed, `--bo3`, `--no-shuffle`, `--no-auto-mana`, the test-setup presets,
`--max-decisions`), setup flags given alongside are ignored with a warning, and output flags
(`--machine`, `--narrative`, `--trace-decisions`, `--rmstate-dump`, `--knowledge-a/-b`, the
self-checks, …) still apply. A plain `--replay` prints the narrative and each replayed input (no
board); under `--machine` it prints only what the output flags ask for. `--player` / `--log-viewer`
limit the narrative's one-sided lines to that seat's view (a replay has no human seat). A replay
whose logged index is outside the live menu stops with `replay diverged at decision N`; a log that
ends before the game does is fatal (`Replay file ended unexpectedly`) unless it ends with the
`QUIT` line of a human seat's quit, where the replay stops cleanly (`QUIT: ...`, exit status 0).

```bash
cd bin
./debug/robomage --agent-a explore --agent-b explore --game-seed 5 \
  --deck-a league/bug --deck-b league/burn --decision-log /tmp/g5.rmlog
./debug/robomage --replay /tmp/g5.rmlog --machine --narrative --trace-decisions
```

The replay corpus (`tests/corpus/`) is such logs from the scripted agent's games (and two explore
games under `--no-auto-mana`), with their replay output as byte-for-byte baselines (`docs/ci.md`).

## Trajectories

`--trajectory-out DIR` writes one episode per game (each recorded seat's decisions featurized, the
behaviour distribution, the reply, the result) and keeps the decision log; `--replay LOG
--trajectory-out DIR` re-featurizes a logged game, such as a `rm_driver.py --rmlog-dir` corpus.
Format and flags: [`trajectory_format.md`](trajectory_format.md).

`--teacher-a / --teacher-b <command>` streams a seat's frames (the `--machine` protocol's bytes) to
a teacher process whose reply at each decision is recorded as the trajectory's `teacher` label;
`--teacher-mix p` plays the label with probability p (training plan T4, DAgger). The scripted agent
is the teacher: `python3 ../python/teacher.py [--tier hard] [--seed N]`.

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
[`trajectory_format.md`](trajectory_format.md) §1.1 and `src/agents/teacher.h`.

## The continuous actor: `--actor`

For training and evaluation runs one engine process plays match after match: `--actor` draws each
match (decks, format, agents, teachers, seats, purpose) from a schedule file the orchestrator
writes and re-reads between matches, derives the match's seeds from (run seed, actor id, match
index), writes each match's decision log (and, for training, its episodes) and reports through a
status file and a per-game results file; SIGTERM stops it cleanly. A per-game decision cap
(`--max-game-decisions N`, recorded in the decision log) ends a runaway game with end reason `cap`.

```bash
cd bin
./release/robomage --actor --actor-schedule RUN/schedule.json --actor-out RUN/actors --actor-id 3 --actor-seed 42
```

Everything about it: [`actor.md`](actor.md).

## State dumps

`--rmstate-dump FILE` writes every viewer's frames (A, B and omniscient) at each decision beside
any run (`%p` in FILE = the process id), and a terminal frame each at match end;
`--state-dump json` adds `FILE.jsonl`, one JSON line per frame. `--rmstate-read FILE` decodes,
re-encodes and compares a stream (`--state-dump json`: its JSON lines on stdout);
`python3 python/rmstate.py dump FILE` prints the same JSON lines from Python;
`bin/<config>/rmknow --viewer A --in FILE --text` runs the knowledge sidecar over a seat's frames.
