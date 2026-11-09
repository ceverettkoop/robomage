# The actor: continuous play from a schedule (`--actor`)

An actor is one engine process that plays match after match, each drawn from a **schedule** file
(written by the orchestrator, `docs/rmtrain.md`), writing per match its decision log and, for
training matches, its trajectory episodes (`docs/trajectory_format.md`). It reports through a
**status** file (rewritten every couple of seconds) and a **results** file (one line per finished
game). Code: `src/agents/actor.{h,cpp}`, `src/agents/actor_schedule.{h,cpp}`. Tested by the gate's
`actorloop` tier, and the `actor` tier (hot reload of an in-process model) (`docs/ci.md`).

```bash
cd bin
./release/robomage --actor --actor-schedule RUN/schedule.json --actor-out RUN/actors \
    --actor-id 3 --actor-seed 42                 # plays until SIGTERM
./release/robomage_nn --actor ...                # the same, with nn: agents (make actor)
```

## 1. Options

| Flag | Meaning |
|---|---|
| `--actor` | play matches until stopped (or `--actor-matches`); every seat is decided in process |
| `--actor-schedule FILE` | the schedule (§2); re-read between matches when the file changes (its modification time, size, inode) or on `SIGHUP` |
| `--actor-out DIR` | the actor's files: `DIR/a<k>.status.json` (§3), `DIR/a<k>.results.jsonl` (§4), and the default output directories `DIR/<purpose>/rmlog` and `DIR/train/traj` |
| `--actor-id K` | the actor's id (default 0): its file names and its seeds |
| `--actor-seed S` | the run seed (default 0) |
| `--actor-start I` | the first match index (default 0). Restart with the status file's `next_match` to continue the sequence: the unfinished match is replayed from its start, rewriting its log and its finished games' episodes (the same games when the policies are the same); `next_match + 1` writes it off |
| `--actor-matches N` | stop after N matches (default: until stopped) |
| `--max-game-decisions N` | the per-game decision cap, under every entry that sets none (default 5000; §5) |

The command line's `--deck-a/-b`, `--bo3`, `--agent-a/-b`, `--teacher-a/-b`, `--teacher-mix`,
`--teacher-timeout`, `--knowledge-a/-b`, `--max-game-decisions`, `--trajectory-seats`,
`--trajectory-omniscient`, `--no-auto-mana` and `--closed-decklist` are defaults under every entry;
an entry or the schedule's `defaults` can override all but the last two, which apply to every match.
Refused under `--actor`: what fixes one game (`--seed`, `--game-seed`, `--replay`, `--play`,
`--actions`, `--player`, the zone presets, `--life-a/-b`, `--hand-a/-b`, `--library-a/-b`), the
outputs the schedule decides (`--decision-log`, `--log-decisions`, `--trajectory-out`),
`--max-decisions` (it ends the process; use `--max-game-decisions`) and other frame writers
(`--rmstate-dump`, `--selfcheck-*`). `--narrative` and `--trace-decisions` work (output only).

## 2. The schedule

A JSON object, written whole (a temporary file renamed over it). Read at start (an error is fatal)
and before a match when it changed or after `SIGHUP`; a reload that fails keeps the last good
schedule and reports the error in the status file's `schedule.error` and `last_error`.

```json
{
  "version": "it12",
  "outputs": {
    "train": {"rmlog_dir": "/runs/r1/rmlog/train", "trajectory_dir": "/runs/r1/episodes"},
    "eval":  {"rmlog_dir": "/runs/r1/rmlog/eval"}
  },
  "defaults": {"format": "bo3", "knowledge_a": "expert", "knowledge_b": "expert", "max_game_decisions": 5000},
  "entries": [
    {"id": "self-bug-burn", "weight": 4, "purpose": "train", "deck_a": "league/bug", "deck_b": "league/burn",
     "swap": "decks", "agent_a": "nn:remote:/runs/r1/rmip.sock:policies=0,1,2",
     "agent_b": "nn:remote:/runs/r1/rmip.sock:policies=0,1,2"},
    {"id": "vs-past-bug-burn", "weight": 1, "purpose": "eval", "deck_a": "league/bug", "deck_b": "league/burn",
     "swap": "sides", "agent_a": "nn:remote:/runs/r1/rmip.sock:argmax", "agent_b": "nn:/runs/r1/ckpt/p8.pt2:argmax"},
    {"id": "vs-scripted-bug-burn", "weight": 1, "purpose": "eval", "deck_a": "league/bug", "deck_b": "league/burn",
     "swap": "sides", "agent_a": "nn:remote:/runs/r1/rmip.sock:argmax", "agent_b": "first",
     "teacher_b": "python3 ../python/teacher.py", "teacher_mix": 1}
  ]
}
```

| Key | Meaning |
|---|---|
| `version` | optional, any value, echoed (as text) in the status and results files; absent, the FNV-1a 64 of the file's bytes (16 hex digits) |
| `outputs` | optional, per purpose (`train`, `eval`): `rmlog_dir`, `trajectory_dir` (null: no episodes). Defaults: `<actor out>/<purpose>/rmlog`; episodes `<actor out>/train/traj` for `train`, none for `eval` |
| `defaults` | optional, an entry's keys (not `id`) applied under every entry |
| `entries` | at least one, with a positive total weight |

An entry's keys; those ending in `_a` / `_b` name its two **sides**, which sit in seats A and B
unless `swap` moves them:

| Key | Meaning |
|---|---|
| `id` | required, unique: names the entry in the status and results files |
| `weight` | ≥ 0 (default 1): the entry's share of the matches |
| `purpose` | `train` (default) or `eval`: which outputs the match writes |
| `deck_a`, `deck_b` | deck stems relative to `resources/decks/` (`league/bug`); the files must exist |
| `format` | `bo3` (default) or `bo1` |
| `swap` | `none` (default); `decks`: the two decks change seats when the match's coin says so (the agents stay); `sides`: everything of side a — deck, agent, teacher, knowledge level, its letter in `trajectory_seats` — changes seats with side b's |
| `agent_a`, `agent_b` | agent specs (`docs/game_running.md`): `explore`, `random`, `nn:<model>[:…]`, `nn:remote:<socket>[:policies=…]`, … |
| `teacher_a`, `teacher_b` | a teacher command (`--teacher-a/-b`, `docs/trajectory_format.md` §1.1) or null |
| `teacher_mix` | in [0, 1] (default 0); needs a teacher |
| `knowledge_a`, `knowledge_b` | `off`, `expert` |
| `max_game_decisions` | ≥ 0: the per-game decision cap (§5) |
| `trajectory_seats` | `A`, `B` or `AB` (default): the recorded sides |
| `trajectory_omniscient` | bool: the episodes' omniscient part |
| `rmlog_dir`, `trajectory_dir` | override the purpose's outputs |
| `sb_explore` | structured sideboard exploration (below); the whole object replaces the defaults' |

Errors naming the entry: an unknown key, a wrong type or value, a missing deck file, an entry
without both decks and both agents (after the defaults and the command line), a mix without a
teacher, an `eval` entry whose `sb_explore` (own or from `defaults`) has p > 0. Relative paths are
relative to `bin/`. Agent specs and teacher commands are checked when a match uses them (a bad one
is fatal then).

**Structured sideboard exploration** (`src/agents/sb_explore.{h,cpp}`, played by the driver's
decision hook, `src/agents/driver.cpp`):

```json
"sb_explore": {
  "p": 0.25,
  "count_weights": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  "weights": {"league/bug": {"in": {"Pernicious Deed": 0.5}, "out": {"Fatal Push": 2.0}}},
  "default_weight": 1.0
}
```

| Key | Meaning |
|---|---|
| `p` | required, in [0, 1]: per seat and sideboarding phase, the probability that the phase explores |
| `count_weights` | 11 numbers ≥ 0, not all 0: the weights of k = 0 … 10 swaps (default uniform) |
| `weights` | per deck stem as `deck_a` / `deck_b` name it: `in` and `out`, card name → weight > 0. Names are the frames' card names (the `CARDS` frame's front face name: `Dead` for Dead/Gone), compared on their ASCII letters and digits, lower-cased; a name that matches no card is never used |
| `default_weight` | > 0 (default 1): the weight of a card its deck's map does not name |
| `agents` | `nn` (default): only seats whose agent is the learned policy (`nn:`, `nn:remote:`); `all`: every seat an in-process agent decides (the gate's torch-free check) |

It applies to each qualifying seat; weights are looked up by the deck the seat plays (under `swap`
the deck, not the side, picks the map). At a seat's first decision of a sideboarding phase (before
games 2 and 3) one coin `random() < p` is drawn from the agents' shared generator (seeded per match,
§6); it holds for the seat's whole phase. In an exploring phase the agent decides as always (an
`nn:` agent's evaluation is made and recorded), then the row played is drawn from the proposal `q`
on the same generator; the step's μ is `q` over the whole menu, with the agent's policy version and
evaluations kept (so π/μ is exact). With `s` the phase's completed swaps (the `MATCH` table's
`sb_swaps`: an IN paired with its balancing OUT) and `P` the normalised count weights:

- a balanced deck (Done offered): `q(Done) = P(k = s) / P(k ≥ s)` (1 when `P(k ≥ s)` is 0 or no
  IN row is offered), the rest over the offered IN rows in proportion to their card's in-weight;
- a swap opened (an IN made, Done withheld): the offered OUT rows in proportion to their card's
  out-weight;
- any other row 0 (a menu with none of the three kinds: uniform).

So the swap count is k drawn from `P` (fewer only when the deck runs out of IN rows), every offered
IN and OUT row has mass (the engine's locks and 15-swap cap apply as for any agent), and `q` is a
function of the seat's frame. Other decisions, and phases whose coin did not land, are the policy's.
A non-qualifying seat, `p` 0 or no `sb_explore` draws nothing: the matches are byte for byte those
without the key.

**Opponents** are entries: self-play (`nn:remote:` on both sides); a past checkpoint (`nn:<past
export>` on one side, or a second server policy id); `scripted:hard` (side b `first` with `teacher_b`
the scripted teacher at `teacher_mix: 1`, so the teacher plays it: μ one-hot, version 0); `explore`
/ `random`. `swap: sides` with one network on both sides gives it both decks and both seats.

## 3. The status file: `<actor out>/a<k>.status.json`

Written whole (a temporary file renamed over it) at start, at least every 2 s while a match plays
(checked at decisions and game ends), after every match and at exit, including a fatal error.

| Key | Meaning |
|---|---|
| `actor`, `pid`, `run_seed` | |
| `state` | `running`; `done` (`--actor-matches` reached); `stopped` (a signal); `failed` (a fatal error: `exit_reason` is its message, the exit status 1) |
| `started_at`, `updated_at` | Unix time, seconds |
| `uptime_s` | |
| `first_match`, `next_match` | `--actor-start`; the index of the match in progress, or of the next one (restart with `--actor-start next_match`) |
| `matches`, `games`, `decisions` | finished matches and games; decisions read (the unfinished match's included) |
| `decisions_per_s`, `recent_decisions_per_s`, `matches_per_s` | over the uptime; decisions over the last window of at least 10 s (the uptime's rate before the first) |
| `caps`, `draws`, `concessions` | games so ended (findings to inspect); `errors`: non-fatal engine errors (`last_error` the latest) |
| `current` | the match in progress (`match`, `entry`, `purpose`, `game_seed`, `game`, `decisions`, `rmlog`), or null |
| `schedule` | `path`, `version`, `loaded_at`, `loads` (successful reads), `entries`, `error` (the last reload's, "" when it parsed) |
| `policy_versions` | the distinct versions of the policies that sampled (`count`, `min`, `max`, `last`): every step with a behaviour distribution |
| `last_error`, `exit_reason` | |

## 4. The results file: `<actor out>/a<k>.results.jsonl`

Appended (and flushed) once per finished game, in order; a match conceded during sideboarding
before a game writes no line for that game. One JSON object per line:

| Key | Meaning |
|---|---|
| `actor`, `match`, `game` | the game is 1-based (the `GAME_RESULT` number, the episode's `g<n>`); score game 1 apart from games 2–3 |
| `entry`, `purpose`, `schedule` | the entry's id and purpose, the schedule's version |
| `format`, `swapped`, `side_a_seat` | `bo1` / `bo3`; whether the coin swapped (the entry's `swap`); the seat side a's agent sat in |
| `deck_a`, `deck_b`, `agent_a`, `agent_b`, `teacher_a`, `teacher_b`, `teacher_mix`, `knowledge_a`, `knowledge_b` | per **seat** (after the swap); a `--replay` of the log must be given the same knowledge levels to write the same episodes |
| `game_seed`, `engine_seed` | the match's (§6) |
| `first` | `A` / `B`: who played first |
| `wins_a`, `wins_b` | the match score at the game's start |
| `result_a`, `result_b` | +1 / −1 / 0 |
| `end_reason` | `win`, `concession`, `draw`, `cap` |
| `decisions`, `decisions_a`, `decisions_b` | the game's decisions (its episode's: the sideboarding before it included; = the footer's `game_decisions`), per seat |
| `policy_versions_a`, `policy_versions_b` | the distinct versions of the seat's behaviour policy in the game (arrays; the random agent's 0, none for `explore`) |
| `rmlog` | the match's decision log |
| `sb` | games 2 and 3 only: the sideboarding phase before the game, per seat (A / B, engine seats), for every agent kind: `{"A": {"explored": bool, "swaps": n, "in": [card, …], "out": [card, …]}, "B": {…}}`, `explored` whether the phase's sb_explore coin landed (false for a seat that does not explore), `swaps` the completed swaps (the cuts), `in` / `out` the cards moved in order, named as the frames name them (including a one-row menu the engine took) |
| `episode` | the game's episode file, or null (an eval match, no recorded seat) |
| `seconds` | the game's wall time |

## 5. Caps, draws, signals and failures

- **The per-game cap** (`--max-game-decisions N`, `max_game_decisions`; default 5000 under
  `--actor`; recorded in the decision log, so a replay caps at the same decision): a game reaching
  its decision N + 1 (its own decisions, pregame included, not the sideboarding before it) stops
  before it is asked, with no result: `GAME_CAP: <n> stopped after <N> decisions` instead of
  `GAME_RESULT`, episode end reason `cap` (results 0), a results line with end reason `cap`. The
  match ends with it (a bo3 prints no `MATCH_RESULT`) and the next match starts. Caps and draws are
  counted in the status file; under self-play they are findings.
- **SIGTERM / SIGINT**: nothing new starts. Between matches the actor exits at once; within a match
  it stops before the next decision is read: the decision log gets a `QUIT` line (a `--replay` stops
  there cleanly), the unfinished game writes no episode (episodes are written whole through a
  temporary file, so no partial `.rmtraj` is visible), the status says `stopped`, exit status 0. A
  second signal kills the process. Teachers get end of input and are reaped.
- **SIGHUP**: re-read the schedule before the next match (a changed file is re-read anyway).
- **Fatal errors** (an evaluator that fails or goes away, a teacher that dies, a model reload that
  fails, a bad agent spec, a decision log that cannot be written): exit status 1, `FATAL:` on
  stderr, and the status file's `state` `failed` with the message in `exit_reason`. The match in
  progress is written off (its log ends where it stopped).

## 6. Seeding and reproduction

Match *i* of actor *k* under run seed *S*:

- **game seed** `g = actor_game_seed(S, k, i)`: three rounds of SplitMix64, `h = mix(S)`,
  `h = mix(h ^ k)`, `h = mix(h ^ i)`, then `g = h >> 2` (62 bits, so matches of a large run do not
  collide), with `mix(x)` SplitMix64's output for state x: `z = x + 0x9E3779B97F4A7C15`,
  `z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9`, `z = (z ^ (z >> 27)) * 0x94D049BB133111EB`,
  `z ^ (z >> 31)`;
- **engine seed** `engine_seed_for_game(g)` (the drivers' derivation, so `--game-seed g` names the
  same game); the agents' generator is seeded with `g`, the teacher channel's coin with `g + 2**32`;
- **the entry**: `u = (mix(g ^ 0x6A09E667F3BCC909) >> 11) / 2**53`, the first entry (file order)
  whose cumulative weight exceeds `u × total`; **the swap coin**: `mix(g ^ 0xBB67AE8584CAA73B) >> 63`.

The match's decision log `<rmlog dir>/a<k>-m<i>.rmlog` replays it alone; its episodes
`a<k>-m<i>.g<n>.rmtraj` are what `--replay … --trajectory-out` writes (μ absent). Rerunning with the
same *S*, *k*, `--actor-start` and schedule plays the same matches byte for byte when the policies
are the same (an `nn:remote:` server's versions can differ between runs; every step records the
version it sampled from).

**Between matches** the process resets the match scope: the per-game reset (`init_ecs`, a fresh
`Game`), the match state (`reset_match_state`: score, concession and cap latches, decklist stores),
the agents' generator, each seat's decoded stream, the in-process feed's once-per-match frames (each
seat gets `SCHEMA`, `CARDS`, `DECKLISTS` again), the knowledge instances, the trajectory writer and
the teacher channel's coin. Agents are reused when a seat's spec repeats (`begin_match` clears what
they kept, e.g. explore's novelty set; an in-process `nn:<model>` reloads a changed file, §7). Each
match gets a new decision log.

**Teachers across matches.** A seat's teacher process keeps running while matches name the same
command for that seat; otherwise it gets end of input, must exit 0, and the new one starts.
`python/teacher.py` resets at each match of its stream (a `SCHEMA` frame after a terminal frame) and
reseeds `random` with its constant `--seed`, so its labels in a match equal a fresh `teacher.py
--seed <same>` on that match's stream (e.g. a `--replay` of the match's log with the teacher
attached). The match's seed is not passed to the teacher (it would leak the shuffles).

## 7. Hot reload

- `nn:remote:` policies are versioned by the server, response by response; the actor records the
  versions (the episodes' `policy_version`, the results lines, the status).
- An in-process `nn:<model>` is reloaded before a match when its file changed (modification time,
  size, inode or device); write the new export to a temporary file and rename it over the path. A
  file that does not load or does not match the featurizer is fatal. The match's decisions carry the
  new model's `policy_version` (`NN-STATS: … reloads=<n>` under `:stats`).
