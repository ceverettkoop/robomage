# Running games: agents, drivers, scripts and logs

How a RoboMage game is run: who decides each seat, the engine's in-process agents, the Python
driver for the scripted agent, `--play` / `--actions` scripts, the machine protocol for an
external agent, and decision logs. Every engine command runs from `bin/` (the engine finds
`resources/` relative to its working directory); `./debug/robomage --help` lists every flag.

## Who decides a seat

Each decision belongs to one seat (the priority holder, or the sideboarding player between
games). In order of precedence it goes to:

1. **`--replay LOG`** — every decision comes from the log (below); nothing else is consulted.
2. **A global script** — `--play <specs>` or `--actions <i,i,…>` decides for both seats until it
   runs out.
3. **The seat's in-process agent** — `--agent-a <spec>` / `--agent-b <spec>`; when a script is
   given, a seat without one takes `first`.
4. **The terminal** — the CLI prompt: without `--machine` every decision is prompted there
   (`--player A|B` shows hidden information from that seat's view only); a `--player` seat under
   `--machine` is prompted there too, but without its menu printed (a human seat against an agent
   is plan step 4b's C++ CLI).
5. **The RMSTATE stream** — under `--machine`, every other seat: its frames on stdout, one integer
   reply per decision on stdin. This is how an external agent (such as the Python driver) plays.

`--agent-a/-b`, `--play` and `--actions` imply `--machine` (the machine decision schedule) and
cannot be combined with `--replay` or with an agent on the `--player` seat.

Whatever decides, every game prints `GAME_RESULT: N Player A|B wins` (or `GAME_RESULT: N draw`,
CR 104.4a) and a bo3 match ends with `MATCH_RESULT: Player A|B wins X-Y`. Under the stream those
lines, like all other output, go to stderr.

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
prints each decision's board and menu as the deciding seat's frame shows them, and the choice
made, on lines starting `| `. `--max-decisions N` stops the run before decision N+1
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
| `scripted` (`hard`, `heuristic`, `scripted:hard`) | the hard tier: smart mulligan, combat simulation, combo lines |
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
(`FILE.<i>` per match when `--games > 1`), `--knowledge-a/-b`, `--engine-flags "…"` for any other
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
payload records; the deciding seat gets its `SCHEMA`, `CARDS`, `DECKLISTS` and `METAGAME` frames
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

Every decision is logged as an integer. `--decision-log FILE` writes the RMLOG v2 log (`SEED`,
`FLAGS` — the setup flags that change the game — both decklists, then every decision, concession
sentinels included); `--log-decisions` writes it to `resources/logs/game_<seed>.log` (interactive
games always log there). `--replay FILE` replays it deterministically: the header alone fixes the
setup (decks, seed, `--bo3`, `--no-shuffle`, the test-setup presets, `--max-decisions`), setup
flags given alongside are ignored with a warning, and output flags (`--machine`, `--narrative`,
`--trace-decisions`, `--rmstate-dump`, `--knowledge-a/-b`, the self-checks, …) still apply. A
replay whose logged index is outside the live menu stops with `replay diverged at decision N`.

```bash
cd bin
./debug/robomage --agent-a explore --agent-b explore --game-seed 5 \
  --deck-a league/bug --deck-b league/burn --decision-log /tmp/g5.rmlog
./debug/robomage --replay /tmp/g5.rmlog --machine --narrative --trace-decisions
```

The replay corpus (`tests/corpus/`) is such logs from the scripted agent's games, with their
replay output as byte-for-byte baselines (`docs/ci.md`).

## State dumps

`--rmstate-dump FILE` writes every viewer's frames (A, B and omniscient) at each decision beside
any run (`%p` in FILE = the process id), and a terminal frame each at match end;
`--state-dump json` adds `FILE.jsonl`, one JSON line per frame. `--rmstate-read FILE` decodes,
re-encodes and compares a stream (`--state-dump json`: its JSON lines on stdout);
`python3 python/rmstate.py dump FILE` prints the same JSON lines from Python;
`bin/<config>/rmknow --viewer A --in FILE --text` runs the knowledge sidecar over a seat's frames.
