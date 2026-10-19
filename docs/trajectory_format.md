# Trajectory episodes (`--trajectory-out`, training plan T1)

The actor-side record of what was played, for the learner (`docs/rnad_training_plan.md` T1, L1):
one **episode** file per game, holding every decision of the recorded seats with the deciding
seat's featurized tensors (exactly as the actor's featurizer produced them), the behaviour
distribution μ it was sampled from, the reply, the policies' evaluations and the game's result.
The engine side is `src/agents/trajectory.{h,cpp}`; the format, its writer and its C++ reader are
`src/featurize/episode.{h,cpp}` (standard library only); `python/rmtraj.py` is the stdlib Python
reader. The gate's `traj` tier (`docs/ci.md`) and the `actor` tier's trajectory checks test it.

## 1. Producing episodes

```bash
cd bin
# live: in-process agents (or a stream seat), every game's episode into DIR
./release/robomage_nn --agent-a nn:remote:/tmp/s.sock:policies=0,1,2 --agent-b nn:remote:/tmp/s.sock \
    --deck-a league/burn --deck-b league/bug --bo3 --game-seed 7 --trajectory-out DIR
# the same games re-featurized from their decision log (μ absent)
./release/robomage --replay DIR/s<seed>.rmlog --trajectory-out DIR2
# the behaviour-cloning corpus (plan B0): scripted games through the Python driver, logs kept
python3 python/rm_driver.py --player-a scripted --player-b scripted --deck-a league/bug \
    --deck-b league/burn --format bo3 --games 100 --quiet --rmlog-dir LOGS
for f in LOGS/*.rmlog; do ./release/robomage --replay "$f" --trajectory-out EPISODES; done
```

| Flag | Meaning |
|---|---|
| `--trajectory-out DIR` | write every game's episode into DIR (created). Play is unchanged |
| `--trajectory-seats A\|B\|AB` | the seats whose decisions are recorded (default `AB`) |
| `--trajectory-omniscient` | each decision also carries the omniscient frame's featurization and the learner targets (§3.4) |

The options are output only: not recorded in the decision log, honored under `--replay`.

- **The decision log is always kept.** Outside `--replay`, `--trajectory-out` turns decision
  logging on: `--decision-log FILE` when given, else `DIR/s<engine seed>.rmlog`. Under bo3 one log
  covers the match; each episode records which game it is. Under `--replay` the replayed log is the
  episodes' log.
- **Names.** `DIR/<stem>.g<n>.rmtraj` for game `n` (1-based, the `GAME_RESULT` number) of the
  match, `<stem>` the decision log's file name without its extension. A live run and a `--replay`
  of its log write the same names; a corpus of logs replayed into one directory keeps one name per
  log (`rm_driver.py --rmlog-dir` names each `<deck a>-<deck b>.<bo1|bo3>.seed<seed>.rmlog`).
- **Atomic.** Each file is written whole when its game ends (or the run stops at
  `--max-decisions`, or a human seat quits), to `<name>.tmp` and renamed into place, so a reader
  polling DIR for `*.rmtraj` never sees part of one.
- **Re-featurization.** The frames are a function of the decision log, so `--replay LOG
  --trajectory-out` writes byte-identical tensors (and every other array but the behaviour
  policy's: `has_mu`, `mu`, `policy_version`, `eval_*`). This is how old games are re-featurized
  after a feature-spec change, and how the scripted agent's corpus becomes tensors. The knowledge
  level (`--knowledge-a/-b`) shapes the `k_*` tokens and is output only, so a replay must be given
  the same levels as the live run (each episode records them).
- **The decision log carries no μ.** It holds one integer per decision and replays byte for byte
  (the replay corpus depends on it), so μ, the policy versions and the evaluations exist only in
  the live run's episodes; a replay's episodes have μ absent, which is what behaviour cloning on
  the scripted corpus needs (the reply is the label).

**Which decisions.** Every decision the engine asks of a recorded seat, in order: a decision the
engine takes itself (a lone pass) is never asked and never appears, so the seat sequence is
irregular and every record carries its seat. Concession unwinding (CR 104.3a) asks nothing.

**What the recorded seat's decider gives.**

| Decider | μ | policy version, evaluations |
|---|---|---|
| `nn:<model>` | the distribution sampled from (after temperature, `thresh=`, `disc=`; one-hot under `argmax`) | the model's version; its one policy (id 0): value, `log π` |
| `nn:remote:<socket>:policies=a,b,…` | the same, from the first policy | the first policy's version; every requested policy's value, `log π`, version |
| `random` | uniform, `1/n` | version 0, none |
| `explore`, `first`, `actions:`, `play:`, a human seat, a stream seat (`--machine`, e.g. the Python scripted agent), every decision of `--replay` | absent | -1, none |
| `--play` / `--actions` script | absent on the script's own replies; the seat agent's once the script hands over | as the seat agent |

## 2. Episodes

An episode is **one game with the sideboarding phase before it.** A sideboarding frame's `game` is
already the upcoming game, so in a bo3 match the sideboarding decisions after game 1 open game 2's
episode, are credited with game 2's result, and come before game 2's own decisions. Game 1's
episode has none. A bo3 match writes one episode per game played.

How an episode ends (`end_reason`, results per seat +1 / −1 / 0):

| End | `end_reason` | Results |
|---|---|---|
| a player won (CR 104.2) | `win` | the `GAME_RESULT` line's |
| a player conceded the game, or the match during the game (CR 104.3a) | `concession` | the `GAME_RESULT` line's (a concession is a loss) |
| a drawn game (CR 104.4a) | `draw` | 0 / 0 |
| a match conceded (`-3`) in the sideboarding phase before game `n` | `concession` | the conceder −1; game `n` was never played: no `GAME_RESULT`, `first` empty, `game_engine_seed` −1 |
| the run stopped at `--max-decisions` (the cap, recorded in the decision log's setup flags, applies to a replay too) | `cap` | 0 / 0 |
| a human seat quit, or `--replay` reached a log's `QUIT` | `quit` | 0 / 0 |

A `-2` read in the sideboarding phase is ignored by the engine (nothing to concede) and is
recorded as the reply it was.

## 3. The file

An RMTB file (`docs/featurizer.md` §4.4) of kind `episode`. Meta: `kind` `"episode"`,
`vocab_hash`, `spec_hash` (the featurizer's: FEATURE_SPEC's hash), `model` (the text table's
encoder, `<repo>@<commit>`: the `text_model` a policy is checked against), `attrs`, `arrays`.

### 3.1 Attrs (header and footer)

In this order; integers unless marked as strings.

| Attr | Meaning |
|---|---|
| `episode_version` | 1 |
| `F`, `C`, `K`, `Q` | the spec's widths of `tok_num`, `tok_cat`, `tok_card`, `tok_ability` |
| `menu_type` | the spec's token type of `menu` rows (`act_tok` is rebuilt from it, §3.3) |
| `omniscient` | 1 when the omniscient arrays are present |
| `format` (string) | `bo1` or `bo3` |
| `game` | the game's index in the match, 0-based (the state format's `header.game`; every step's `game`) |
| `wins_a`, `wins_b` | the match score at the game's start |
| `first` (string) | `A` or `B`: who plays first; empty when the game never started |
| `deck_a`, `deck_b` (string) | the deck stems (`league/burn`; `stacked_a` for a sculpted deck); the decklists themselves are in the decision log's header |
| `engine_seed` | the match's engine seed (the decision log's `SEED`) |
| `game_engine_seed` | this game's engine seed (bo3: `engine_seed` + game index); −1 when it never started |
| `game_seed` | `--game-seed` (the driver's seed: the agents' generator), −1 when not given (always under `--replay`) |
| `agent_a`, `agent_b` (string) | each seat's decider: its agent spec, `play+<agent>` / `actions+<agent>` under a script, `human`, `stream` (the RMSTATE stream: an external agent), `replay` |
| `knowledge_a`, `knowledge_b` (string) | each seat's knowledge level (`off`, `expert`, `expert+belief`) |
| `seats` (string) | the recorded seats: `A`, `B` or `AB` |
| `policy_versions_a`, `policy_versions_b` (string) | the distinct policy versions of the seat's steps with μ, ascending, comma-separated |
| `rmlog` (string) | the decision log, as the engine was given it (relative to its working directory, `bin/`, unless absolute) |
| `result_a`, `result_b` | +1 / −1 / 0 (§2) |
| `end_reason` (string) | `win`, `concession`, `draw`, `cap`, `quit` |
| `decisions` | the steps in this file |
| `game_decisions` | every decision read in the episode (both seats, recorded or not) |

### 3.2 Arrays

`D` = steps, `ΣA` = menu rows over all steps, `ΣN` / `ΣR` = tokens / relations over all steps,
`T` = token types. All offsets arrays are `i64` with `D + 1` entries from 0 (step `d` is
`values[off[d]:off[d+1]]`).

| Array | dtype | Shape | Contents |
|---|---|---|---|
| `type_widths` | i32 | `[T, 4]` | per token type its numeric, categorical, card and ability slots (columns `num`, `cat`, `card`, `ability`) |
| `seat` | u8 | `[D]` | the deciding seat: 0 = A, 1 = B (the state format's viewer) |
| `game` | i32 | `[D]` | `header.game` (0-based; a sideboarding step's is the upcoming game) |
| `decision` | i32 | `[D]` | `header.decision`: within the game from 1, or within the sideboarding phase from 1 |
| `sideboarding` | u8 | `[D]` | `header.sideboarding`; (game, sideboarding, decision) is the state format's decision key |
| `chosen` | i32 | `[D]` | the reply as the decision log holds it: a menu row, or −2 / −3 (a concession) |
| `has_mu` | u8 | `[D]` | 1 when the decider gave a behaviour distribution |
| `policy_version` | i64 | `[D]` | the version of the policy μ came from; −1 without μ (0 for an unversioned model and the random agent) |
| `teacher` | i32 | `[D]` | the teacher's label (a menu row, training plan T4); −1 none |
| `menu_offsets` | i64 | `[D + 1]` | menu rows per step (`A` = the step's `act_tok` length) |
| `tok_offsets`, `rel_offsets` | i64 | `[D + 1]` | tokens and relations per step |
| `tok_type` | u8 | `[ΣN]` | each token's type |
| `tok_num_offsets` | i64 | `[D + 1]` | nonzero numeric values per step |
| `tok_num_index` | u16 or i32 | `[nnz]` | each nonzero value's position in its step's row-major `[N, F]` `tok_num` |
| `tok_num_value` | f16 | `[nnz]` | the value (the featurizer's f32 rounded to nearest-even binary16; every f16 other than +0 is stored, −0 included) |
| `tok_cat` | u16 or i32 | `[Σ cat widths]` | each token's first `type_widths[t].cat` categorical ids, tokens in order |
| `tok_card` | i16 or i32 | `[Σ card widths]` | likewise its card slots (vocab ids, −1 none) |
| `tok_ability` | i16 or i32 | `[Σ ability widths]` | likewise its ability slots (ability-table rows, −1 none) |
| `rel` | u16 or i32 | `[ΣR, 3]` | `(src token, dst token, relation type)`, token indices within the step |
| `mu` | f32 | `[ΣA]` | μ over each step's menu rows (0s when `has_mu` is 0) |
| `eval_policy` | i64 | `[D, P]` | per step each evaluated policy's id (−1 pad); present when some step has evaluations |
| `eval_version` | i64 | `[D, P]` | its version |
| `eval_value` | f32 | `[D, P]` | its value (the deciding seat's view) |
| `eval_log_pi` | f32 | `[ΣA, P]` | its `log π` over the menu rows (log-softmax of its logits, before temperature) |
| `omni_*` | | | the omniscient frame's featurization of the same decision, the same eight arrays as the seat's (`omni_tok_offsets` … `omni_rel`), present with `omniscient` |
| `opp_list`, `opp_archetype` | i32 | `[D]` | the learner targets (`docs/featurizer.md` §3.5): `METAGAME` row indices, −1 none |
| `opp_hand`, `own_library`, `opp_library` | i16 or i32 | ragged | card ids, each with `<name>_offsets` |

The integer widths follow the spec and the file: categorical ids u16 unless the spec's
categorical size passes 65536 (then i32); card ids i16 unless the card table passes 32768 rows;
ability rows i16 unless the ability table passes 32768 rows; `rel` and `tok_num_index` u16 unless a
value in the file passes 65535. A reader takes the dtype from the meta.

### 3.3 Expanding a step to the featurizer's tensors

The tensors of `docs/featurizer.md` §3 for step `d`, exactly (the gate checks equality with
`featurize_seat` on the decision's frame, `tok_num` rounded to f16):

1. `tok_type = tok_type[tok_offsets[d]:tok_offsets[d+1]]`, `N` its length.
2. `tok_num = zeros(N * F)`; for each `k` in `tok_num_offsets[d] … tok_num_offsets[d+1]`:
   `tok_num[tok_num_index[k]] = tok_num_value[k]`; reshape `[N, F]`.
3. `tok_cat = zeros(N, C)`, `tok_card = full((N, K), -1)`, `tok_ability = full((N, Q), -1)`; walk the
   episode's tokens in order with one cursor per array: token `i` of type `t` takes the next
   `type_widths[t].cat` values into `tok_cat[i, :w]`, likewise card and ability. (Vectorised: the
   per-token widths `type_widths[tok_type]` and their cumulative sums give every value's row and
   column.)
4. `rel = rel[rel_offsets[d]:rel_offsets[d+1]]`.
5. `act_tok` = the indices `i` with `tok_type[i] == menu_type`, in order (menu row `a` is the `a`-th);
   its length is the step's menu size. `tok_mask` and `act_mask` are all ones.

A numpy reader: `np.frombuffer(buf, dtype, count, offset=12 + meta_length + array_offset)` per
array (`rmtb.py`'s docstring), then the steps above on whole episodes at once.

### 3.4 The omniscient part

With `--trajectory-omniscient`, each step also carries the omniscient viewer's frame of the same
decision featurized with the same spec (`featurize_viewer`; every object identified, no `k_*`
tokens, seat-relative columns as the omniscient frame defines them: self = A) and the seat's
learner targets (`decision_targets`). Learner-side data: the frame is built by the trajectory writer
itself (encoded and decoded once per recorded decision) and no agent receives it (the isolation
rule of `docs/featurizer.md` §3.5).

## 4. Readers

- C++: `featurize::read_episode` (`src/featurize/episode.h`) expands every step into
  `featurize::Tensors` (and the omniscient tensors and `Targets`).
- Python (stdlib): `python/rmtraj.py`, `rmtraj.read(path)` → `Episode` (`attrs`, `steps`: seat,
  key, `chosen`, `mu` or None, `policies`, `tensors` dicts of flat lists, `omniscient`, `targets`).
  `python/pycheck.py --part traj` checks it against the C++ reader.

## 5. Size and cost

Measured on explore/random games (league decks, debug-built episodes; the sizes do not depend on
the build):

| Part | Bytes per decision |
|---|---|
| seat tensors | ~5.3 KB: `tok_num` index + value ~2.5 KB (about 600 nonzero of ~4,300 per-type slots), `tok_cat` ~0.9 KB, `rel` ~0.75 KB, `tok_card` ~0.7 KB, `tok_type` ~0.3 KB, the rest < 0.1 KB |
| omniscient tensors and targets | ~8.8 KB more |

For comparison, the featurizer's own tensors file (`rmfeat --featurize`: f32 and i64, every slot
of `[N, F]`) is ~100 KB per decision.

Cost (release build, explore vs explore bo3, one process per match, both seats already served by the
in-process feed): ~470 µs per decision without trajectories, ~520 µs with (`featurize_seat` on the
decoded frame, the compaction and the file write; the agent's tensors are reused when it is `nn:`),
~810 µs with `--trajectory-omniscient` (the omniscient frame is encoded, decoded and featurized per
decision); loading the featurizer's inputs adds ~13 ms per process. A recorded seat that the RMSTATE
stream decides (not otherwise served in process) also gets its frames built and decoded for the
writer.
