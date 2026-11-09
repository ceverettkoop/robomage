# The featurizer and the card tables

What a learned policy reads, how it is built from one seat's RMSTATE frames, and the card tables
every ML consumer shares. Code: `src/featurize/` (std + the state-format decoder only), the tool
`bin/<config>/rmfeat`, the text-table generator `bin/<config>/rmembed` (libtorch) and the actor in
`src/actor/`.

## 1. Principles

- **One seat's frames in, tensors out.** A pure function of the seat's decoded `SCHEMA`, `CARDS`
  and current `STATE` frame (`k_*` tables included) and the card tables (§4); what a model can know
  is exactly what the seat can know (`docs/state_format.md` §1).
- **Driven by the schema's tags** (`state_format.md` Appendix A.1): a new table or column reaches
  the model with no featurizer change. The exceptions are §3.4, listed in the spec.
- **No card vector per decision.** Cards travel as vocab ids, row indices into the card tables.
- **Framework-neutral.** Tensors, spec and tables are plain arrays and JSON; the network contract
  (§6) is one consumer.
- **Deterministic.** The same frames give the same bytes; row order follows the frame, canonical
  where the format says order is meaningless (§7). So featurizing a `--replay` of a game's decision
  log reproduces what its `nn:` seats featurized live, byte for byte.
- **Decoded structs in** (`rmstate::DecodedFrame`, what the in-process agent feed holds), never
  frame bytes; nothing is re-encoded.

## 2. Inputs the featurizer reads

| Frame | Used for |
|---|---|
| `SCHEMA` | every table, column, tag and enum string table; the spec is built from it |
| `CARDS` | not read per decision: card features and the back-face map (`back_face`, §4.1) come from the whole-vocab tables, so a decision needs only its `STATE` frame |
| `STATE` | snapshot tables (ids 20-48) and knowledge tables (90-95); event tables (80-82) only under the history option (§3.7) — otherwise memory is the sidecar's job (`knowledge_store.md`) |

The **actor entry point** (`featurize::featurize_seat`, what `nn:` calls) takes the deciding seat's
frames only and refuses a frame whose viewer is not a seat, or that is not a `STATE` frame of the
spec's schema. The **learner entry points** (§3.5) also take the omniscient viewer's frames; only
`rmfeat` and learner-facing writers (the trajectory writer) call them.

## 3. Per-decision tensors

A decision is a set of **tokens**, one per row of every featurized `STATE` table, plus relations
and the menu. The featurizer emits one decision unpadded; a batcher pads and stacks to the network
contract's batch dimension `B` (§6). On disk: an RMTB file of kind `tensors`
(`featurize::tensors_file`): per decision `viewer`, `game`, `decision`, `sideboarding` (i64), then
per tensor its values over every decision concatenated along the first dimension plus
`<tensor>_offsets` (i64, decisions + 1 entries, in that tensor's rows); masks stored as `u8`.

| Tensor | Shape | dtype | Contents |
|---|---|---|---|
| `tok_type` | `[N]` | i64 | token type: its layout-registry id (§6.1) |
| `tok_num` | `[N, F]` | f32 | numeric features, each slot at its registry column within its type (§3.2); unused/reserved 0 |
| `tok_cat` | `[N, C]` | i64 | categorical ids in one global id space (§3.2), each slot at its registry column; unused 0 (pad) |
| `tok_card` | `[N, K]` | i64 | vocab ids of the row's card columns (card-table rows); `-1` none |
| `tok_ability` | `[N, Q]` | i64 | ability-table rows of the row's ability references (§3.3, §4.3); `-1` none |
| `tok_mask` | `[N]` | bool | real token |
| `rel` | `[R, 3]` | i64 | relations `(src token, dst token, relation type)` (§3.3) |
| `act_tok` | `[A]` | i64 | the token of menu row `a` (= reply `a`) |
| `act_mask` | `[A]` | bool | real menu row |

`F`, `C`, `K`, `Q` are the registry's per-token capacities (§6.1), fixed by the layout. `N`, `R`,
`A` vary per decision; padding is a batcher option, never a format cap.

### 3.1 Token types

Every `STATE` table with id 20-49 or 90-95 is a token type, except `strings` and the tables a fold
takes whole (§3.6: `own_deck`, `k_seen`, `k_known_hand`); plus the synthetic `ledger` and, with the
history option, `hist_event` / `hist_card` (§3.7). A type's `tok_type` is its registry position
(§6.1), matched by name; tokens come in id order. Current ids: the table types in ascending table id
(0-31), `ledger` 32, `hist_event` 33, `hist_card` 34, `pool_mana` 35; a new type takes the next free
id. A table's rows are its tokens in frame order, except where a fold filters them (`k_pool_card`)
or keeps only an object's differences from its card (`object_subtype`, `object_keyword`).
Single-row tables (`header`, `turn`, `pending`, `pending_cost`, `match`, `k_meta`) are one token
each. Child tables (`object_keyword`, `object_counter`, `stack_target`, `k_pool_card`, …) are tokens
related to their parent (§3.3).

### 3.2 Columns by tag

| Tag | Becomes |
|---|---|
| `bool` | one numeric slot, 0/1 |
| `count` | one numeric slot, `log1p(x)` |
| `signed` | one numeric slot, `sign(x)·log1p(|x|)` |
| `index` (`pos`, `order`, a libpos's `_pos`) | one numeric slot, `log1p(x)`; `-1` → 0 and the none flag |
| `mask:<e>` | a block of numeric slots of `e`'s capacity (§6.1): value `v` at block start + `v`'s registry index |
| `enum:<e>` | one categorical slot over `e`'s capacity range (`c` values take `c + 1` ids; the first is none: all ones or out of range), value `v` at its registry index |
| `player-rel` | one categorical slot (none, self, opponent); the absolute `enum:seat` beside it is dropped (seat symmetry) |
| `card` | one `tok_card` slot |
| `handle`, `row:<t>` | a relation (§3.3), no feature |
| `text` | dropped, except §3.4's hashed columns |

A `card` or `index` column also gets a numeric none flag (1 at all ones); an enum's none is its
slot's first id; a `row:` column's none is the missing relation. The global categorical id space
holds every (token type, categorical column) range at its registry offset, id 0 the pad, ids past
the last range reserved; one embedding table covers it. Enum values match by name, so the engine's
string table may be reordered or grown without moving an id. Composites are recognised by column
name: a seat pair `<n>` / `<n>_rel`; a library position `<n>_anchor` / `<n>_pos` / `<n>_span` (each
part by its tag); an ability reference `ab_card`, `ab_index`, `ab_key` (§3.3).

### 3.3 Relations and ability references

- **Identity.** A table whose first column is a `handle` named `handle` (`object`, `stack`,
  `reveal`) owns that handle; the lowest table id wins (`object` over `stack`).
- **Handle columns** (`obj`, `source`, `target`, `attached_to`, `attacker`, `blocker`, `for`, …)
  relate the row's token to the token owning that handle, typed by `(table, column)` (registry id,
  §6.1); so does an owning table's own `handle` column when another table wins it (`stack` →
  `object`). A handle no token owns gives no relation.
- **`row:<t>` columns** relate to row `v` of table `t`'s tokens (`k_pool_card.pool`,
  `stack_target.lki`, `delayed_subject.trigger`, …), typed the same way; dropped when `t` has no
  token type.
- **Ability references** (`ab_card`, `ab_index`, `ab_key` on `object_ability`, `stack`, `menu`,
  `delayed_trigger`) become one `tok_ability` slot: the ability-table row of `(ab_card, ab_index)`,
  else of `ab_key` (§4.3), else `-1`.
- **Menu.** Each `menu` row is a token; `act_tok[a]` is menu row `a`'s token. Its `source` /
  `target` handles are ordinary relations, so a pointer head scores menu tokens after attention.

### 3.4 Columns treated otherwise (listed in the spec)

| Column | Treatment |
|---|---|
| `object.card` | also a second `tok_card` slot: the current face (the back face's vocab id when `transformed`), since card tables are per face |
| `object_counter.kind`, `player_counter.kind`, `object_keyword.text`, `player_effect.filter`, an abref's `ab_key` | categorical: 1 + FNV-1a 64 of the text mod 256 buckets (empty text = none) |
| `menu.description` | dropped |

### 3.5 Learner-side: the omniscient featurization and targets

- **Omniscient featurization.** The same spec over the omniscient frame (as `--rmstate-dump`
  writes it): every object identified, hidden zones are tokens with their cards, no `k_*` rows, the
  ledger holds both players' `own_deck`. Seat-relative columns read self = A, so a learner pairs it
  with the deciding seat explicitly. For a critic that sees the whole game.
- **Targets** (`rmfeat --targets`, `src/featurize/targets.h`), per seat decision, from the same
  decision's omniscient frame: the opponent's hand as a multiset of card ids (ascending); the seat's
  own library top first (`object.pos`) up to the spec's `library_depth` (10); the opponent's library
  likewise. RMTB kind `targets`: per decision `seat`, `game`, `decision`, `sideboarding`, and
  `opp_hand`, `own_library`, `opp_library` each as values plus `_offsets`.
- **Isolation.** Neither reaches a decision: the in-process feed carries seat frames only and the
  actor entry point refuses other viewers. `--selfcheck-featurize`
  (`src/state_format/selfcheck_featurize.h`) featurizes each seat's frame at every decision through
  the actor entry point and requires the same tensors from the scramble check's frame (hidden
  identities rotated) and the permutation check's (hidden zones permuted), and the omniscient frame
  refused; the gate also checks an `nn:` seat's tensors equal `rmfeat`'s over that seat's stream.

### 3.6 Folds (listed in the spec)

Two lossless folds make fewer tokens of the same facts (`src/featurize/folds.{h,cpp}`): every number
of a folded row lands in a numeric slot of a merged token, and its identity (player, card, object)
is that token's categorical, card slot or relation. RMSTATE and the sidecar are unchanged.

**The card ledger** (type `ledger`): one token per (player, card) named by the frame's `own_deck`,
`k_seen`, `k_known_hand`, `k_pool_card` and exact `k_position` rows, in ascending (`player_rel`,
card id) order — the player's universe as the viewer has it (own current 75; for the opponent the
registered list under open decklists, or the known cards under `--closed-decklist`). Slots:
`player_rel` (categorical), the card (`tok_card`) and twelve numeric slots, `log1p` of a count
except `deck`:

| Slot | Value |
|---|---|
| `deck` | 1 when the card has an `own_deck` row (the viewer's own list only) |
| `main`, `side` | `own_deck.main`, `own_deck.side` |
| `seen_game`, `seen_match` | `k_seen.game`, `k_seen.match` |
| `known_hand` | `k_known_hand.copies` (the opponent's) |
| `hand_min`, `hand_max` | the `HAND` pool's `k_pool_card` min / max |
| `library_min`, `library_max` | sum over the player's library **runs** of the run's `k_pool_card` min / max, plus the player's `k_position` rows naming the card exactly |
| `sideboard_min`, `sideboard_max` | the `SIDEBOARD` pool's `k_pool_card` min / max |

A `LIBRARY_RANGE` pool is a run when no other library pool of its owner, and no exact `k_position`
of its owner, lies strictly inside its range; the others are blocks of runs (`knowledge_store.md`
§6). Exact positions are runs with a determined card. So library bounds are always sound, and exact
when the library has one undetermined run.

`k_pool_card` rows:

- **fold** (bounds become ledger slots): every row of a `HAND` or `SIDEBOARD` pool; every row of a
  player's library pool when it is the player's only `LIBRARY_RANGE` pool;
- **stay** tokens (type `k_pool_card`, kind `filtered`): every library pool's rows when a player has
  several (runs after a put-back or scry, and blocks, which carry joint facts), and `FACE_DOWN` and
  `SET_ASIDE` pools.

Every `k_pool` row stays a token, so `k_position.pool` keeps its relation. A folded row's `pool`
relation is replaced by its slot; a kept row's points at its pool's token (row references into a
filtered type are renumbered to its tokens).

**Object deltas** (types `object_subtype`, `object_keyword`, kind `delta`): the card table holds
each face's printed subtypes and keywords (§4.1), so an object's rows are emitted only where they
differ from its **current face's printed rows** (`object.card`, the back face when `transformed`:
the `card:face` slot), compared as multisets of the columns other than `obj` (`subtype`; `kw` and
`text`), read from the `CARDS` frame's `card_subtype` / `card_keyword` rows of that face:

- a row the object has and the printed rows lack: a token, `removed` = 0;
- a printed row the object lacks: a token with its values, `removed` = 1;
- an object whose face the viewer cannot see (`card` none) or whose face has no card row keeps
  every row with `removed` 0; so do rows whose handle no object row has.

Tokens come per object row in frame order, added rows sorted by value, then removed ones. The
printed rows are built from the whole-vocab `CARDS` frame (`build_printed_rows`, `card_tables.h`;
with tables read from files, from the cards file given beside them).

**Checks** (`featurize` tier, every decision of the corpus's and explore games' dumps, seat and
omniscient): each object's subtype and keyword rows rebuilt from the tensors equal the frame's;
every `own_deck`, `k_seen`, `k_known_hand` and folded `k_pool_card` row has its ledger token with
its values, kept `k_pool_card` tokens are exactly the unfolded rows, no other ledger token. The
row-order check (§7) shuffles folded tables too and finds every token by origin
(`featurize_traced`). Deliberately broken folds fail it.

### 3.7 Recent-event history (an option, off by default)

By default event tables are left out (the network has no recurrence; memory is the sidecar's
facts). The option (`FeatureOptions::history_events` N > 0; `rmfeat --history N`; the engine's
`--feature-history N` for the trajectory writer, `nn:`, `--featurize-seats` and
`--selfcheck-featurize`; `python -m rmlearn.corpus --refeaturize … --feature-history N`; a training
run's `run.feature_history`, `docs/rmtrain.md` §3) adds the seat's recent observation events as
tokens.

- **Off**, every output (spec document, hash, tensors, episodes) is byte-identical to no option.
  **On**, the spec gains two token types and a `history` key, so `spec_hash` differs; `layout_hash`
  does not. Hence (§6.2): a **checkpoint** loads either way (a model trained without history starts a
  history run with those inputs untrained); an **export** is exact, so one built without history is
  refused (`spec_hash`) by an evaluator with it and vice versa — rebuild it from its checkpoint. A
  run's replay window keeps only episodes of its tables' spec (`spec.Tables.accepted_spec_hashes`).
- **Event history** (`featurize::EventHistory`, `src/featurize/history.h`): kept per seat and match
  by whoever featurizes (the trajectory writer per recorded seat; `nn:` through the agent hook
  `observe`, which sees every frame of its seat before anyone decides; `rmfeat` per seat over a
  dump), fed every frame of the seat in order, menu or not. Holds the events since the last
  `game_start` whose turn ≥ the latest frame's turn − 1, each with its `event_card` rows
  (`event_player` rows not kept). Independent of how events were batched into frames, so a dump,
  the `--machine` stream and the in-process feed give the same tokens.
- **Selection** per decision: events of the frame's turn (`turn.turn`) and the previous one, newest
  first, at most N; each with its first `cards_per_event` (8) cards in frame order (a search is a
  `look` at the whole library).
- **Token types**, after the folds' synthetic ones: `hist_event` (kind `history`): the `event`
  table's columns by tag (§3.2) plus numeric `turn_offset` (0 this turn, 1 previous), `age`
  (`log1p` of the newest-first rank), `more_cards` (`log1p` of cards past the cap); `hist_card`: the
  `event_card` columns, its `event` column a relation to its event's token.
- **Dropped** (spec `history.dropped`): `event.seq`, `event.turn`, `event.game`, `event.batch`,
  `event_card.draw_seq`, `event_card.draw_idx`. Handles are never values (§3.3).
- **Relations.** `hist_event.source`, `hist_card.handle_before` / `handle_after` relate to the
  object holding the handle now, followed through the kept events' old → new links (handles are
  never reissued, so links form chains); a retired handle or one whose object left the frame relates
  to nothing.
- **Learner side.** `featurize_seat` refuses a history spec without the seat's event history;
  `featurize_viewer` without one emits no history tokens. BC's `event_dropout` p
  (`learn/rmlearn/bc.py`) masks each event token (and its card tokens) with probability p.

## 4. The card tables

Whole-vocab tables, the same for every match, indexed by vocab id (`src/card_vocab.h`; ids
`0 … N_CARD_TYPES-1`, the token band from 900, the token sentinel 1023), built from a **whole-vocab
`CARDS` frame** (the engine's `--cards-frame FILE`: `SCHEMA` and `CARDS` records, the
`--closed-decklist` card set) by the same tag rules (`src/featurize/card_tables.h`). The featurizer
reads `N_CARD_TYPES + 1` from the text table's row count, never an engine header. `rmfeat
--export-tables CARDS DIR` writes `card.rmtb`, `ability.rmtb`, `part.rmtb`, `spec.json` (§6), the
model manifest `manifest.json` and `registry.json` (§6.2).

### 4.1 Card table `[N_CARD_TYPES + 1, Fc]` f32

One row per **face vocab id** (the `card` row whose `face_card` is that id). Columns:

- the `card` row's numeric and mask columns by §3.2 (mana value, pips, colours, types, supertypes,
  P/T, loyalty, token), its enums one-hot (`face`, `layout`);
- per `(card, face)` child table (`card_subtype`, `card_keyword`, `card_pip`), the face's row count
  per value of each enum column (multi-hot over `subtype` and `keyword`, per-`pip_kind` counts);
- `n_abilities`: `log1p` of the face's `card_ability` rows;
- `present` (0 for an id with no card).

Names: `<col>`, `<col>:<VALUE>` (mask bit), `<col>=<VALUE>` (one-hot), `<table>.<col>=<VALUE>`
(child count). Every column sits at its registry position (§6.1): an enum's one-hot, mask or counts
are a block of the enum's capacity; the width is the registry capacity; reserved columns read 0,
named `<entry>#<i>` (past a block's registered values) or `#<position>` (past every entry). Row
`N_CARD_TYPES` is the none row (zeros), read by a `-1` card slot. The file also holds `back_face`
(i64, per card id: its back face's vocab id; itself when it has none; -1 for a back face with no id
of its own), which gives `object.card`'s current-face slot (§3.4).

### 4.2 Text table `[N_CARD_TYPES + 1, 384]` f32

Oracle-text vectors, one row per face vocab id with oracle text, zeros elsewhere and at the none
row. Committed (§5).

### 4.3 Ability table `[Nab, Fa]` f32 and part table `[Npart, Fp]` f32

- **Ability rows** are stable addresses: card `c`'s ability `i` is row `c × A + i` (`A` = registry
  `abilities_per_card`, 16; more abilities on a card is refused, a layout bump); unused rows are
  zeros; the **key band** from `key_base = N_CARD_TYPES × A` (16384) holds one row per `ab_key` of
  the fixed key list (abilities with no card row: granted or engine-built) at its registry row,
  capacity 256 (rows 16384-16639). Appending a card fills only its own rows, so a stored
  `tok_ability` keeps naming the same ability. The table is data, no parameters. Columns by the tag
  rules: `face`, `ability_kind`, `category`, `timing` one-hot; `tap`, `limit`; the structured
  parameters (amount and its kind, power/toughness modifiers, target classes, target bounds);
  `n_parts`, `charm_num`; `keyed` (1 for a key row).
- **The fixed key list**, in key-row order: every distinct non-empty `card_ability.key` of the frame
  (keyword-built abilities: `Equip`, `Cycling`, `Reconfigure:unattach`), plus the engine-built key
  forms (`state_format.md` Appendix A.1) the `SCHEMA` frame enumerates: `keyword:<k>` per `keyword`
  value, `subtype_mana:<c>` per `color` value, and `LandType`. A key row that is some card row's key
  has the features of the first such row in `(card, index)` order (a card appended later never
  becomes it); the others only `keyed`. Keys built during play from text no frame enumerates
  (`granted_trigger:<script>`, `enchant:<filter>`, delayed-trigger keys such as
  `return_from_exile:<n>`) resolve to `-1`; the hashed `ab_key` categorical (§3.4) still tells them
  apart.
- **Part rows**: one per `card_ability_part` row (sub-abilities and modes) in `(card, ability,
  part)` order, the same parameter columns plus part kind, depth and parent, and `ability` (the
  owning ability row; an i64 array, not a feature). Their order may shift on a card append: they are
  pooled into their ability rows and never referenced by a token.
- **The ability index** (in the ability file): `ab_card`, `ab_index` (i64 per row; -1 for key or
  unused rows), `key_rows` (i64 per key; its `columns` are the keys), attrs `abilities_per_card` and
  `key_base`. `tok_ability` resolves through it (§3.3).

### 4.4 Files: RMTB

Every exported table is an RMTB file of named arrays, readable with any language's standard library
(also used by trajectory episodes, `docs/trajectory_format.md`):

| Part | Layout |
|---|---|
| magic | 4 bytes `RMTB` |
| version | u32 LE (1) |
| meta length | u32 LE, bytes of the JSON that follows (padding included) |
| meta | UTF-8 JSON, keys in a fixed order, space-padded to a multiple of 8 bytes |
| data | values, LE, row-major, each array at an 8-byte-aligned offset from data start (zero padding between; the file ends with the last array) |

Meta keys, in order, each omitted when not applicable: `kind` (`card`, `text`, `ability`, `part`,
`tensors`, …); `vocab_hash` (FNV-1a 64 over the vocab's `(id, name)` pairs in ascending id, each as
the id's u32 LE bytes, the name's UTF-8 bytes and a 0 byte; 16 hex digits); `spec_hash` (§6); for
the text table `model` (`<repo>@<commit>`; episodes carry it too), `pooling` (`cls`, `mean`),
`norm_version` (§5) and `rows_meta` (per present row ascending: `{"id": <vocab id>, "hash": <FNV-1a
64 of its normalised text, 16 hex>}`); `attrs` (named strings and integers in writer order: a file
kind's own header fields, e.g. an episode's, `docs/trajectory_format.md` §3.1, or the ability
table's `abilities_per_card` and `key_base`); then `arrays`: one `{"name", "dtype", "shape": [...],
"offset"[, "columns": [...]]}` per array in file order (`shape` any rank, `[]` a scalar; `offset`
from data start, the previous array's end rounded up to 8; `columns` optionally names the last
dimension's entries). The same content always encodes to the same bytes.

Dtypes: `f32`, `f16` (IEEE binary16, round-to-nearest-even), `i64`, `i32`, `i16`, `u16`, `u8`.
Ragged data is two arrays: values and `i64` offsets (`n + 1`; row `r` is
`values[offsets[r]:offsets[r+1]]`), no padding on disk. The text table is one array, `vectors`
`[N_CARD_TYPES + 1, hidden]` f32, with `rows_meta` in the meta.

Readers: `src/featurize/rmtb.{h,cpp}` (C++, std only; also the writer and f16 conversions),
`python/rmtb.py` (`struct`, f16 as `e`, i16 as `h`); numpy: `np.frombuffer(buf, dtype, count,
offset=12 + meta_length + array_offset)`.

## 5. The text table and `make`

- **Entries.** One per vocab id (card entries and the registered token band) whose face has oracle
  text: a front face under its own id, a DFC's back face under the back face's entry, a split card
  (both halves under one id) as both halves, a token from its token script; read from the parse
  cache, no entity created. Engine side `src/text_table.{h,cpp}`; std-only rules
  `src/featurize/oracle_text.{h,cpp}`.
- **Text.** `"<type line>. <oracle>"`, split halves joined with `" // "`. The type line is the
  face's `Types:` line placed as printed: supertypes, `Token` for a token, card types, ` — ` (em
  dash), subtypes. In the oracle, every whole-word occurrence of the face's name, the other face's
  name and each one's part before its first `, ` (the legendary short name) becomes `CARDNAME`,
  longest first; the type line keeps its words. Then newlines and tabs become spaces, space runs
  collapse, the text is trimmed and the pin's `prefix` prepended. This is normalisation version 1
  (`featurize::NORM_VERSION`); any change bumps it, making every row stale. Row hash: FNV-1a 64 of
  the text's UTF-8 bytes (`featurize::text_hash`).
- **The pin**, `tools/embed/MODEL_PIN`: `key value` lines `repo`, `commit` (full SHA), `pooling`
  (`cls` or `mean`) and `prefix` (a JSON string). The table's `model`, `pooling` and `norm_version`
  must equal the pin's, else every row is stale (a pin change regenerates the whole table). A
  different `vocab_hash` (a vocab append) makes stale only entries without a row of their hash; the
  other rows are kept byte for byte and the header rewritten.
- **Committed** at `bin/resources/embeddings/oracle_text.rmtb` (kind `text`, `vectors` `[1025, 384]`
  f32).
- **Check**: `robomage --text-table-check FILE --model-pin PIN` (std only, no network) exits 0 when
  every entry has a `rows_meta` row with its current hash, no other row is listed and the header
  matches; else prints `TEXT-TABLE: stale <id> <name>` per missing or stale entry (`extra <id>` per
  listed non-entry) and exits 1. A missing table is all stale.
- **`make text-table`** (a dependency of `all`, so every `make` runs the check; a current table
  needs no network or libtorch; a stale table fails CI) builds `robomage` and runs the check; on
  failure:
  1. `robomage --text-table-inputs .cache/embed/text_table_inputs.jsonl --text-table TABLE` writes
     the stale entries as JSON lines after a header (row count, vocab hash, model, pooling,
     normalisation, every present entry; `featurize::text_inputs_jsonl`);
  2. `python3 tools/embed/fetch_model.py` downloads the pinned model's `config.json`, `vocab.txt`,
     `tokenizer_config.json` and `model.safetensors` into `.cache/embed/<repo, / as __>/<commit>/`
     (weights checked against the repository's SHA-256);
  3. `make rmembed` (libtorch found as for `make actor`; C++20) and `bin/<config>/rmembed --model DIR
     --pin PIN --inputs FILE --table TABLE` encodes those entries and rewrites the table, keeping
     every other present row's bytes and zeroing rows no longer present; it prints each encoded
     row's three nearest rows for review;
  4. the check again.
  Without libtorch or network it fails, naming the stale entries, the step and the command; the
  table is replaced by a rename, never half written.
- **Tokenizer** (`src/featurize/wordpiece.{h,cpp}`, std only): BERT basic tokenization as the
  reference fast tokenizer does it (special tokens matched in raw text; cleaning; CJK split;
  lowercasing and accent stripping for an uncased model; whitespace and punctuation split), then
  greedy longest-match WordPiece with `##` continuations and `[UNK]`; `[CLS] … [SEP]`, never
  truncated (`rmembed` fails on a text past the model's positions). Unicode tables cover ASCII,
  Latin-1, Latin Extended-A, combining marks, General Punctuation, Mathematical Operators and the
  unified CJK blocks; a text with any other character is refused.
- **Encoder** (`tools/rmembed/`): BERT in the libtorch C++ API over safetensors weights read
  directly (F32, F16, BF16; names with or without `bert.`): embeddings, LayerNorm, per layer masked
  self-attention, dense + residual + LayerNorm, exact-GELU intermediate, dense + residual +
  LayerNorm; `cls` or `mean` pooling; L2 normalisation. Each text is encoded alone on one thread, so
  a row depends on its text only.
- **Goldens** (`tools/embed/make_goldens.py`, torch + transformers, never run by `make`): `tokens`
  records `tests/embed/tokens_golden.jsonl` (every entry's ids from the pinned `tokenizer.json`,
  cross-checked against `vocab.txt` + `tokenizer_config.json`, plus hard and coverage strings); the
  `vocab` tier checks the C++ tokenizer against it id for id when the model's vocab is cached (else
  a skip). `encoder` compares `rmembed --embed-test` (batch 1 and padded batches) with `BertModel` in
  eval, both poolings, on a seeded small random BERT (2 layers, hidden 64; F32, or F16 with `bert.`
  names) or the real model (`--model DIR`); `table` compares a generated table with reference
  vectors. Each requires cosine ≥ 0.99999.

## 6. The spec and the network contract

**`FEATURE_SPEC`** (version 4, `src/featurize/feature_spec.h`): one-line JSON, keys in a fixed
order, written by `rmfeat --spec` from the whole-vocab cards file's `SCHEMA` frame, its card tables
and the layout registry (the actor builds it in memory from the same inputs). Contents: layout
version and `layout_hash`; schema and vocab hashes; hashed-text bucket count; targets' library
depth; the registry dims `F`, `C`, `K`, `Q`, `G`, `categorical_size`, `type_capacity`,
`relation_capacity`; the menu's token type; per token type (by registry `id`) its numeric block in
`G` (`num_at`, `num_capacity`), numeric slots (name, source column, transform, `at`), categorical
slots (name, source, `col`, id offset, size), card slots (`col`), ability slots (`col`) and relation
types; relation types `(table, column, target)` by registry id; `enums` (registry values of every
enum a categorical or mask slot reads; value `i` is a slot's id `offset + 1 + i`); the §3.4
overrides; the §3.6 folds (per type its `kind`: `table`, `filtered`, `delta` with its `printed`
table, `ledger`, `history`; a synthetic type's `table` is null); the card, text, ability and part
tables' rows, widths and column names (every position, reserved included; the ability table with
`abilities_per_card`, `key_base`); `ab_keys` and `ab_key_rows`; with history (§3.7), a final
`history` key (caps, the two types, dropped columns). `spec_hash` (FNV-1a 64 of the document bytes)
is the exact identity, in every exported file and model; `layout_hash` names the layout alone
(§6.1).

### 6.1 The layout registry

**The layout is the contract, the vocab is data.** Every position a network's parameters are
indexed by is fixed by one committed, append-only file, `src/featurize/registry.json`
(`src/featurize/registry.h`; compiled into every featurizing binary via `.incbin` in `registry.cpp`,
its object depending on the file). Sections, every entry matched by name:

| Section | Gives | Capacity |
|---|---|---|
| `dims` | `F` 96 (numeric columns per token), `C` 12, `K` 6, `Q` 2 (categorical, card, ability columns per token), `G` 1408 (every type's numeric block side by side), `types` 64, `categorical` 6144 ids, `relations` 64, `ab_keys` 256, `abilities_per_card` 16, `hash_buckets` 256 | fixed |
| `enums` | per enum a slot or column reads: its values (index = position) | closed enums (`face`, `seat`, `step`, `color`, `layer`, `day_night`, `lib_anchor`) their size; open ones max(8, the power of two ≥ 2n), e.g. `keyword` 128, `effect_kind` 256, `action_category` 128, `subtype` 1024, `card_type` 32, `supertype` 8, `layout` 16 |
| `token_types` | name → id, the type's numeric block (`num_at`, `num_capacity` ≤ F) | `types` |
| `numeric` | per (type, slot) its column in the block; a mask an enum-capacity block | the type's `num_capacity`: slots used rounded up to 8 with a quarter more room (≥ 8) |
| `categorical` | per (type, slot) its `tok_cat` column and id range (`offset`, `size`: enum capacity + 1, player-rel values + 1, hash buckets + 1) | `C` columns, `categorical` ids |
| `cards`, `abilities` | per (type, slot) its `tok_card` / `tok_ability` column | `K`, `Q` |
| `relations` | per (table, column, target) its relation type id | `relations` |
| `tables`, `columns` | per card / ability / part table its width; per column its position (an enum's one-hot, mask or counts an enum-capacity block) | card 1536, ability 448, part 448 |
| `ab_keys` | per fixed ability key its key-band row | `ab_keys` |

`build_spec` and `build_card_tables` place everything at its registry position, leave reserved
positions 0 (-1 for card and ability slots) and **fail**, naming the entry and `make registry`, on
a value, column, slot, type, relation or key the registry lacks; past a capacity (an enum full, a
type's block full, a card past `abilities_per_card`, …) they fail naming it: a layout bump. An entry
the schema no longer makes is a hole that reads 0; an engine enum may be reordered.

- **`make registry`** (`rmfeat --registry-update FILE --cards CARDS`) appends what the engine's
  schema and whole-vocab tables need at the next free positions (an enum value after its last, a
  slot at its type's end, a categorical range at the id space's end, a new type's block after the
  last, capacity from its use), never moving an entry, then rebuilds; commit the file.
- **The check** (`vocab` tier, `rmfeat --registry-check FILE --cards CARDS --base <HEAD's file>`):
  the file parses (no overlap, positions within capacities) and is canonical, is the registry the
  binaries were built with, needs no `make registry`, and moves no entry of
  `git show HEAD:src/featurize/registry.json` unless `layout_version` was raised.
- **`layout_hash`** (FNV-1a 64 over `layout_version`, the dims, the tables' capacities, the card
  rows and the targets' library depth; in the spec and every episode's attrs) is the same for every
  spec of one layout whatever was appended. A change that is not an append (a capacity, a moved
  entry) raises `layout_version`; checkpoints are migrated by name (`learn/rmlearn/migrate.py`,
  `docs/rmtrain.md` §1.2).
- **A capacity bump** (`rmfeat --registry-raise FILE ENUM=CAPACITY[,…]`,
  `Registry::raise_capacities`): each named enum's capacity raised and the enum made open; every
  block of it (numeric masks, table columns, categorical ranges) widens, later positions in the same
  space shift by the growth before them, and a space whose entries no longer fit (a type's block,
  G, a table, the categorical ids) grows to them plus its previous spare room (rounded up to 8, 128,
  64, 512); names, order and value indices are kept, `layout_version` goes up by one. Then `make
  registry` appends the new values and `python -m rmlearn.migrate --run DIR --keep layout<N>` moves
  every checkpoint.

The `featurize` tier tests the property (`tools/robomage_check/featurize_layout.h`): a card appended
at the next free id and a fake keyword appended keep the `layout_hash`, every position and every
existing card's, ability's and part's rows, and featurize every corpus decision to the same
tensors; the keyword enum reordered gives the same positions, card table and tensors; a full enum,
a card past `abilities_per_card` and a missing value are refused.

**Network contract** (any framework):

```
forward(tok_type [B,N] i64, tok_num [B,N,F] f32, tok_cat [B,N,C] i64, tok_card [B,N,K] i64,
        tok_ability [B,N,Q] i64, tok_mask [B,N] bool, rel [B,R,3] i64, rel_mask [B,R] bool,
        act_tok [B,A] i64, act_mask [B,A] bool) -> (logits [B,A] f32, value [B] f32)
```

The card, text, ability and part tables are the model's frozen buffers, loaded from the RMTB files
when it is built; an export folds them in and carries the tables' model manifest (§6.2), and the
actor refuses an export not of its own tables exactly. Logits at padded menu rows are ignored.

**Padding** (`featurize::pad_batch`, `src/featurize/batch.h`, std only): to the batch's largest `N`,
`R`, `A`, each rounded up to a multiple of 8 and at least 8 (no dimension 0 or 1, which
`torch.export` would specialise); pads 0 (`tok_card`, `tok_ability` -1), masks 0; `rel_mask` marks
real relations. Buffers are reused across batches.

**Model metadata.** A model carries seven strings in its export's own metadata, never a sidecar
file: `spec_hash`, `vocab_hash`, `layout_hash`, `tables_digest` (16 hex digits each), `text_model`
(the text table's `model`), `vocab` (the vocab manifest, JSON `[id, name]` array, §6.2) and
`policy_version` (unsigned, `"0"` when absent). Stored as a TorchScript module's string attributes,
an AOTInductor package's metadata (inductor config `aot_inductor.metadata`, read with
`AOTIModelPackageLoader::get_metadata`) or an ONNX model's `metadata_props`. The evaluator tells the
formats apart by the zip's records (AOTInductor: `archive_format`; TorchScript: `constants.pkl`),
not the file name. An AOTInductor package is machine code for the exporting CPU's ISA
(`AOTI_CPU_ISA` in its metadata); TorchScript is portable.

**Export-friendly.** Inputs are padded tensors and masks only; a conforming model has no
data-dependent control flow (masks select real rows, gathers and scatters apply relations, no loops
over real rows or shape-dependent branches). So one model exports as TorchScript, a `torch.export`
/ AOTInductor `.pt2` and ONNX; the actor loads the first two, ONNX is for other runtimes.

**Batched evaluation** (`actor::Evaluator::evaluate(batch, &out, &err)`, `src/actor/evaluator.h`):
a batch of featurized decisions (one game or many) → per decision an `Evaluation`: the value and,
per policy served (policy 0 first, the one sampled from), the logits over its menu rows, `log π`
(log-softmax over menu rows), its value and the policy version. A failure is `false` with a
message, never a default action. Implementations: in-process (`load_model_evaluator`: CPU, one
thread, TorchScript via `torch::jit::load` or AOTInductor via `AOTIModelPackageLoader`; one policy)
and remote (`connect_remote_evaluator`, `src/actor/remote_evaluator.h`).

**Remote evaluation** (`nn:remote:<endpoint>`) speaks RMIP v2 (`docs/inference_protocol.md`; wire
code `src/inference/protocol.h`, std + POSIX): a Unix stream socket carries a handshake (the model
manifest without the registry, §6.2, the widths, the policy ids) and small `EVAL` / `DONE` messages;
a POSIX shared-memory region per connection (created and named by the client, unlinked once the
server has it) holds the request (the decisions' unpadded tensors laid out as an RMTB `tensors`
file's arrays with per-decision sizes for offsets, the spec hash in its header) and the response
(per requested policy its version, values, and logits and `log π` over each decision's menu rows).
The server may batch across clients up to a size or deadline, refuses a handshake whose tables are
not its models' exactly (§6.2) or a request whose spec hash differs, and answers with each policy's
current version, so weights can be swapped between decisions. The client fills one `Evaluation`
entry per requested policy (`policies=`, the first sampled from). A server that refuses, errs, goes
away or times out ends the actor with a fatal error naming the endpoint.
- `rmserve` (`tools/rmserve/`, C++, libtorch, CPU, `make actor`): the reference server; several
  policies (`--policy <id>=<model>`, `--version <id>=<n>`) through the in-process evaluator, many
  clients on one poll loop, batches up to `--max-batch` decisions, or `--deadline-us` after the
  oldest queued request, or as soon as every connected client has a request queued;
  `RMSERVE-STATS:` on SIGTERM.
- `tools/nn/serve_check.py`: a minimal Python server (`socket`, `struct`,
  `multiprocessing.shared_memory`, numpy, torch). The learner's server (`learn/rmlearn/server.py`)
  implements the same protocol.

**The `nn:` agent** (`src/actor/nn_agent.h`): `nn:<model>[:<key>=<value>…]`; an unknown key fails
loudly.

| Key | Meaning |
|---|---|
| `temp=<t>` | sampling temperature (default 1), drawn with the agents' shared generator (replays from the seed) |
| `argmax` | the largest logit; rows within 1e-5 (absolute plus relative) tie and the first is taken, so float drift between exports cannot reorder rows equal in exact arithmetic |
| `thresh=<p>` | DeepNash projection (`src/agents/policy_projection.h`, OpenSpiel `rnad.py` `_threshold`): rows below p after temperature dropped, the rest renormalised (none dropped when every row is below p); default 0, off |
| `disc=<n>` | then (`_discretize`) probabilities rounded to multiples of 1/n summing to 1, rows taking `ceil(p·n)` in descending order until n is spent; default 0, off |
| `dump=<file>` | every decision it featurized, written at exit as one `tensors` file (the bytes `rmfeat --featurize` writes for that seat) |
| `evals=<file>` | every evaluation received, at exit, RMTB kind `evaluations`: per decision and policy the version and value, per menu row and policy logits and `log π`, per menu row the behaviour `mu`, per decision `chosen` |
| `stats` | an `NN-STATS:` line at exit (remote: also requests, server compute time attributed to them, mean server batch) |

`nn:remote:<endpoint>` takes the same keys plus `policies=<id>,…` (default `0`), `timeout=<s>`
(default 60) and `shm=<MiB>` (request area, default 32), and needs no model file. Sideboarding
decisions take the same path. For its last decision the agent keeps (`Agent::last_behaviour`,
`agents/agent.h`) the exact distribution it sampled from over every menu row (after temperature and
projection; one-hot under `argmax`), the policy version, every requested policy's evaluation and
the tensors, which the trajectory writer (`--trajectory-out`, `docs/trajectory_format.md`) reuses.
The featurizer's inputs are built once before the match, in process (`load_engine_featurization`,
`src/state_format/cards_file.h`: the whole-vocab records `--cards-frame` writes and the committed
text table), so they equal what `rmfeat` builds from that file; per decision the agent reads only
the seat's decoded `STATE` frame. It runs in `bin/<config>/robomage_nn` (`make actor`), whose `main`
registers the `nn:` kind (`agents::register_agent_kind`) then calls `engine_main`; the plain engine
refuses `nn:`, naming `make actor`.

**Reference policy** (`tools/nn/make_reference_policy.py`, torch, never run by `make`), the
recommended model family: per token type a linear projection of `tok_num`, plus `tok_cat`
embeddings, plus projections of the gathered card, text and ability rows, summed into a width-`D`
token; an entity transformer whose attention adds a learned bias per relation type for `(src, dst)`
pairs; a pointer head (MLP over each `act_tok` token's output) for logits; a value head over the
mean of real tokens. `build` reads the exported tables and the text table, initialises from a seed
(width 64, 4 heads, 2 layers; relation biases at `(src, dst)` and `(dst, src)`, scattered into
`[B, H, N, N]` with `index_add`; ability rows average their part rows in; table projections folded
into frozen buffers, so a forward gathers rows rather than projecting the whole vocab) and exports
`<prefix>.ts.pt`, `.pt2` and `.onnx`; `check` evaluates the three on a `tensors` file's decisions
alone and in padded batches, requiring agreement within 1e-4 (absolute plus relative).

### 6.2 Compatibility: what a model was trained with (`src/featurize/compat.h`)

A checkpoint is the portable artefact; an export is built for one vocab and tables. Every model
carries the **model manifest** of the tables it was trained on (written by `rmfeat --export-tables`
as `manifest.json` + `registry.json`; a checkpoint's `hashes`, an export's metadata, the inference
handshake): `spec_hash`, `vocab_hash` (exact identity), `text_model`, `layout_hash` (§6.1),
`tables_digest` (FNV-1a 64 of the tables' card digests, below), `vocab` (the **vocab manifest**:
`[id, name]` per card-table row, the name the whole-vocab CARDS frame gives each face vocab id) and,
in a checkpoint, `registry` (the layout registry's text). **One check**, `check_compatible(model,
now, exact)`, in C++ (`featurize::check_compatible`) and Python (`learn/rmlearn/spec.py`) with the
same verdicts (`learn/check.py` runs both on the same cases; `rmfeat --compat MODEL NOW
[--exact]`):

- **every model:** the same `layout_hash` and `text_model`; every `[id, name]` of the model's vocab
  present unchanged now (a new id passes; a removed or renamed id is refused, named); when both
  carry a registry, no entry of the model's moved (`registry_moves`, §6.1). Used by
  `load_checkpoint`, the R-NaD learner's state and `_load_with_config`, the exploiter's and the
  exporter's loads; a loaded model then carries the current tables' manifest;
- **exact** (an export, whose rows are folded in): also the same `spec_hash`, `vocab_hash` and
  `tables_digest` (drift in a known card's rows makes folded rows stale) and no current id the model
  lacks. The evaluator (`nn:`), `rmserve` (its policies among themselves and against each client's
  handshake), `rmlearn/server.py`, `tools/nn/serve_check.py`, `rmtrain` and the benchmark's
  `export_missing` refuse or rebuild such an export; the refusal says to rebuild from its
  checkpoint.

Refusal codes are the protocol's (`layout_hash`, `text_model`, `vocab_hash`, `spec_hash`,
`tables_digest`). The vocab hash is the text table's, so a card appended without `make text-table`
leaves it and the spec hash unchanged; only the vocab manifest shows the new id. A manifest without
`layout_hash` passes only on its own tables' `spec_hash`, `vocab_hash` and `text_model`.

**Table drift** (`src/featurize/drift.h`): a parse change to a known card passes the check but
changes its inputs. A card's rows are its card-table row, its ability rows (`id · A + i`) and their
part rows, as nonzero values by column name. The committed **card digest**
`src/featurize/card_digest.json` holds per card a hash of its rows (values printed to 6 significant
digits, so a libm's last bit does not move it); the `vocab` tier (`rmfeat --digest-check`) fails
when a known card's rows hash otherwise or a card is new or gone; `make registry` rewrites it
(`--digest-update`), so the commit's diff names the cards whose inputs changed. `rmfeat
--table-drift OLD NEW` compares two exported tables directories exactly, naming per changed card the
columns and old/new values (`rmtrain retable`'s report, `docs/rmtrain.md` §1.1).

## 7. Tests

The `featurize` tier (`docs/ci.md`) runs the model-free checks over the replayed corpus's and
explore games' dumps; the text table's are the `vocab` tier's; the `nn:`, re-featurization, export
and remote ones are the `actor` tier's, run when the actor binary and a Python with torch are
available.

- deterministic: the same frames give identical tensors;
- the folds lose nothing (§3.6);
- shapes and slot meanings agree with the spec; every `rel` and `act_tok` index is a real token;
  every `tok_card` is `-1` or a card-table row, every `tok_ability` `-1` or an ability row; every
  column no slot of the token's type uses is 0 (-1);
- the registry's append-only property (§6.1);
- row-order invariance where the format says order is meaningless: the permutation self-check
  gives the same frame bytes and, under `--selfcheck-featurize`, the same tensors; every token table
  but `menu` (folded ones included) with rows shuffled and `row:` references renumbered gives the
  same tensors up to the matching token permutation (each token found by its origin: table row or
  place in a fold's canonical order);
- the exported tables equal a standard-library Python reader's view of the files;
- the text table check passes on a clean checkout; regenerating a deleted row leaves every other
  row's bytes unchanged; the tokenizer matches the golden ids and the encoder the golden vectors;
- exact re-featurization: `rmfeat` over a `--replay` of an `nn:` game equals what `nn:` featurized;
- history (§3.7): off, no `history` key or type; on, the off types first and another hash,
  `rmfeat --spec --history` the same document; a corpus game's seat decisions featurized twice with
  history give the same tensors, which less their history tokens are the off tensors, history
  tokens within the caps, each card token related to one event; episodes written with
  `--feature-history` equal `rmfeat --featurize --history` over the dump;
- isolation (§3.5): an `nn:` seat's tensors equal `rmfeat`'s for that seat's stream, unchanged by the
  scramble self-check; the actor entry point refuses an omniscient or other-seat frame;
- the reference policy's three exports give the same logits and values; in-process and `rmserve`
  evaluation agree within 1e-5 (with batches across clients); `nn:remote:` through `rmserve` and the
  Python server plays the in-process argmax games;
- the play-time projection: known answers and random menus (sums, dropped rows, multiples of 1/n,
  the most probable row kept, a fixed point), and a sampled `temp=0.02:thresh=0.03:disc=16` bo3
  match whose every behaviour distribution is the projection of its logits, the row taken one with
  mass.
