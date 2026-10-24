# The featurizer and the card tables (plan step 5a)

Design of plan step 5 (`docs/engine_fork_plan.md`): what a learned policy reads, how it is built
from one seat's RMSTATE frames, and the card tables every ML consumer shares. The implementation
is `src/featurize/` (std + the state-format decoder only), the tool `bin/<config>/rmfeat`, the
text-table generator `bin/<config>/rmembed` (libtorch) and the actor in `src/actor/`.

## 1. Principles

- **One seat's frames in, tensors out.** The featurizer is a pure function of the seat's decoded
  `SCHEMA`, `CARDS` and current `STATE` frame (`k_*` tables included) and the card
  tables (§4). It sees nothing else of the engine, so what a model can know is exactly what the
  seat can know (`docs/state_format.md` §1).
- **Driven by the schema's tags.** Every table and column is handled by its semantic tag
  (Appendix A.1 of `state_format.md`), so a new table or column reaches the model with no
  featurizer change. The few columns treated otherwise are listed in §3.4 and in the spec.
- **No card vector per decision.** Cards travel as vocab ids, which are row indices into the card
  tables; a model gathers the rows it needs.
- **Framework-neutral.** The tensors, the spec and the card tables are plain arrays and JSON,
  defined here without reference to any ML library. The network contract (§6) is one consumer.
- **Deterministic.** The same frames give the same bytes; row order follows the frame, which is
  canonical where the format says order is meaningless (tested: §7). Since the frames are a
  function of the RMLOG, featurizing a `--replay` of a game's decision log reproduces what its
  `nn:` seats featurized live, byte for byte; `docs/rnad_training_plan.md` re-featurizes old games
  on that.
- **Decoded structs in.** The input is the decoded tables (`rmstate::DecodedFrame`, what the
  in-process agent feed already holds), never frame bytes, and nothing is re-encoded. A later
  path that builds those tables straight from the engine needs no featurizer change.

## 2. Inputs the featurizer reads

| Frame | Used for |
|---|---|
| `SCHEMA` | every table, column, tag and enum string table; the spec is built from it |
| `CARDS` | not read per decision: card features, and the map from a physical card to its back face's vocab id (the card table's `back_face`, §4.1), come from the whole-vocab tables, so a decision needs only its `STATE` frame |
| `STATE` | the snapshot tables (ids 20-48) and knowledge tables (90-95); event tables (80-82) are not featurized (memory is the sidecar's job, `knowledge_store.md`), unless the recent-event history option is on (§3.7) |

The **actor entry point** (`featurize::featurize_seat`, what `nn:` calls) takes the deciding seat's
frames only and refuses a frame whose header viewer is not a seat (and any frame that is not a
`STATE` frame of the spec's schema). The **learner entry points** (§3.5) also take the omniscient
viewer's frames; only `rmfeat` and learner-facing writers (the training plan's trajectory writer)
call them.

## 3. Per-decision tensors

A decision is a set of **tokens**, one per row of every featurized `STATE` table, plus relations
between them and the menu. The featurizer emits one decision unpadded (the shapes below); a
batcher pads and stacks decisions to the network contract's leading batch dimension `B` (§6). On
disk a decision (or a list of them) is an RMTB file of kind `tensors` (`featurize::tensors_file`):
per decision its `viewer`, `game`, `decision` and `sideboarding` (i64), then per tensor its values
over every decision concatenated along the first dimension and `<tensor>_offsets` (i64, decisions
+ 1 entries, in that tensor's rows); the masks are stored as `u8`.

| Tensor | Shape | dtype | Contents |
|---|---|---|---|
| `tok_type` | `[N]` | i64 | token type: its id in the layout registry (§6.1) |
| `tok_num` | `[N, F]` | f32 | numeric features, each slot at its registry column within its type (§3.2); unused and reserved columns 0 |
| `tok_cat` | `[N, C]` | i64 | categorical ids in one global id space (§3.2), each slot at its registry column; unused columns 0 (the pad id) |
| `tok_card` | `[N, K]` | i64 | vocab ids of the row's card columns, row indices into the card tables; `-1` none |
| `tok_ability` | `[N, Q]` | i64 | ability-table rows of the row's ability references (§3.3, stable rows §4.3); `-1` none |
| `tok_mask` | `[N]` | bool | real token (false = padding) |
| `rel` | `[R, 3]` | i64 | relations `(src token, dst token, relation type)` (§3.3) |
| `act_tok` | `[A]` | i64 | the token of menu row `a` (menu row `a` = reply `a`) |
| `act_mask` | `[A]` | bool | real menu row |

`F`, `C`, `K`, `Q` are the layout registry's per-token capacities (§6.1), fixed by the layout, not
by what today's schema uses. `N`, `R` and `A` vary per decision; padding is a runtime option of the
batcher, never a format cap.

### 3.1 Token types

Every `STATE` table with id 20-48 or 90-95 is a token type, except `strings` and the tables a fold
takes whole (§3.6: `own_deck`, `k_seen`, `k_known_hand`); so are the fold's synthetic type,
`ledger`, and with the history option its two types (§3.7). A type's id (its `tok_type`) is its
permanent position in the layout registry (§6.1), matched by name; tokens come in id order. Today's
ids are the table types in ascending table id (0-31), `ledger` (32), `hist_event` and `hist_card`
(33, 34); a type added later takes the next free id, whatever its table id. A table's
rows are its tokens in frame order, except where a fold filters them (`k_pool_card`) or keeps only an
object's differences from its card (`object_subtype`, `object_keyword`). Single-row tables (`header`, `turn`, `pending`, `pending_cost`, `match`,
`k_meta`) are one token each, so global state is a few tokens rather than a special vector. Child
tables (`object_keyword`, `object_counter`, `stack_target`, `k_pool_card`, …) are tokens of their
own, related to their parent (§3.3).

### 3.2 Columns by tag

| Tag | Becomes |
|---|---|
| `bool` | one numeric slot, 0/1 |
| `count` | one numeric slot, `log1p(x)` |
| `signed` | one numeric slot, `sign(x)·log1p(|x|)` |
| `index` (a position: `pos`, `order`, `_pos` of a libpos) | one numeric slot, `log1p(x)`, `-1` → slot 0 and the none flag below |
| `mask:<e>` | a block of numeric slots of `e`'s capacity (§6.1): value `v` at the block's start plus `v`'s registry index |
| `enum:<e>` | one categorical slot: the id range of `e`'s capacity (`c` values take `c + 1` ids; the first is the cell's none: all ones or out of range), value `v` at its registry index |
| `player-rel` | one categorical slot (none, self, opponent); the absolute `enum:seat` beside it is dropped, so a model is seat-symmetric |
| `card` | one `tok_card` slot |
| `handle`, `row:<t>` | a relation (§3.3), no feature |
| `text` | dropped, except the columns §3.4 hashes |

A `card` or `index` column also gets a numeric none flag (1 at all ones); an enum's none is its
slot's first id, and a `row:` column's is the missing relation. The global categorical id space
holds every (token type, categorical column) range at its registry offset, with id 0 the pad and the
ids past the last range reserved; one embedding table covers it. Enum values are matched by name,
so the engine's string table may be reordered or grown without moving an id. A composite is recognised by its column names: a seat pair `<n>` / `<n>_rel`, a
library position `<n>_anchor` / `<n>_pos` / `<n>_span` (each part by its tag) and an ability
reference `ab_card`, `ab_index`, `ab_key` (§3.3).

### 3.3 Relations and ability references

- **Identity.** A table whose first column is a `handle` named `handle` (`object`, `stack`,
  `reveal`) owns that handle; the lowest table id wins, so `object` wins over `stack` (a stack
  object is an `object` row too).
- **Handle columns** (`obj`, `source`, `target`, `attached_to`, `attacker`, `blocker`, `for`, …)
  give a relation from the row's token to the token owning that handle, typed by
  `(table, column)` (a relation type's id is its registry position, §6.1); so does an owning table's own `handle` column when another table wins the
  handle (a `stack` row relates to its `object` row). A handle no token owns gives no relation (it
  names an object the viewer cannot identify).
- **`row:<t>` columns** give a relation to row `v` of table `t`'s tokens (`k_pool_card.pool`,
  `stack_target.lki`, `delayed_subject.trigger`, …), typed the same way; one into a table with no
  token type is dropped.
- **Ability references** (the `abref` triple `ab_card`, `ab_index`, `ab_key`: `object_ability`,
  `stack`, `menu`, `delayed_trigger`) become one `tok_ability` slot: the ability table's row for
  `(ab_card, ab_index)`, else its row for `ab_key` (§4.3), else `-1`.
- **Menu.** Each `menu` row is a token; `act_tok[a]` is the token of menu row `a`. Its `source` /
  `target` handles are ordinary relations, so a pointer head scores menu tokens after attention.

### 3.4 Columns treated otherwise (listed in the spec)

| Column | Treatment | Why |
|---|---|---|
| `object.card` | also a second `tok_card` slot: the face as it is now (the back face's vocab id when `transformed`) | the card tables are per face |
| `object_counter.kind`, `player_counter.kind`, `object_keyword.text`, `player_effect.filter`, the `ab_key` of an abref | categorical: 1 + FNV-1a 64 of the text modulo 256 buckets (the empty text is the slot's none) | counter kinds and keyword parameters have no enum |
| `menu.description` | dropped | free text, covered by the structured columns |

### 3.5 Learner-side: the omniscient featurization and targets

- **Omniscient featurization.** The same spec featurizes the omniscient viewer's frame (as
  `--rmstate-dump` writes it at every decision): every object is identified, hidden zones are
  tokens with their cards, and the `k_*` token types have no rows (the
  omniscient viewer has no knowledge tables; its ledger holds both players' `own_deck`). Seat-relative columns read as the omniscient frame defines them (self = A),
  so a learner pairs it with the deciding seat explicitly. For a critic that sees the whole game.
- **Targets** (`rmfeat --targets`, `src/featurize/targets.h`), per decision of a seat, from the
  omniscient frame of the same decision: the opponent's hand as a multiset of card ids (ascending);
  the seat's own library, top first (`object.pos`), as card ids up to the spec's `library_depth`
  (10); the opponent's library likewise. An RMTB `targets` file: per decision `seat`, `game`,
  `decision`, `sideboarding`, and `opp_hand`, `own_library`, `opp_library` each as values plus
  `_offsets`, for auxiliary heads.
- **Isolation.** Neither reaches a decision: the in-process feed carries seat frames only and the
  actor entry point refuses other viewers. The engine's `--selfcheck-featurize`
  (`src/state_format/selfcheck_featurize.h`) featurizes each seat's frame at every decision with
  the actor entry point and requires the same tensors from the scramble check's frame (hidden
  identities rotated) and the permutation check's (hidden zones permuted), and the omniscient frame
  refused; the gate (5d) adds that an `nn:` seat's tensors equal `rmfeat`'s featurization of that
  seat's own stream.

### 3.6 Folds (listed in the spec)

Two folds make fewer tokens of the same facts (`src/featurize/folds.{h,cpp}`). Each is lossless:
every number of a folded row lands in a numeric slot of a merged token, and a folded row's identity
(its player, card or object) is that token's categorical, card slot or relation. The gate's `featurize` tier reads every decision's folds back (below). RMSTATE and the
knowledge sidecar are unchanged.

**The card ledger** (type `ledger`): one token per (player, card) of the cards the frame's
`own_deck`, `k_seen`, `k_known_hand`, `k_pool_card` and exact `k_position` rows name, in ascending
(`player_rel`, card id) order. That is the player's universe as the viewer has it: their own current
75 (`own_deck`; for the omniscient frame both players'), and for the opponent the registered list
under open decklists (every card of it is in some hidden pool with a maximum above 0, or seen) or
the cards known under `--closed-decklist`. Slots: `player_rel` (categorical), the card (a `tok_card`
slot) and twelve numeric slots, `log1p` of a count except the first:

| Slot | Value |
|---|---|
| `deck` | 1 when the card has an `own_deck` row (only the viewer's own list: the opponent's registered counts and post-sideboarding split are not in a seat's `STATE` frame, as before) |
| `main`, `side` | `own_deck.main`, `own_deck.side` |
| `seen_game`, `seen_match` | `k_seen.game`, `k_seen.match` |
| `known_hand` | `k_known_hand.copies` (the opponent's) |
| `hand_min`, `hand_max` | the `HAND` pool's `k_pool_card` min / max (the opponent's hand) |
| `library_min`, `library_max` | the sum over the player's library **runs** of the run's `k_pool_card` min / max, plus the player's `k_position` rows naming the card exactly |
| `sideboard_min`, `sideboard_max` | the `SIDEBOARD` pool's `k_pool_card` min / max (the opponent's sideboard) |

A library pool (kind `LIBRARY_RANGE`) is a run when no other library pool of its owner lies strictly
inside its range and no exact `k_position` of its owner inside it; the others are blocks of runs
(`knowledge_store.md` §6). Exact positions are runs whose card is determined, which have no pool. So
the library bounds are sound always (each run's bound holds, the positions are known) and exact when
the library has one undetermined run.

Which `k_pool_card` rows fold (their pool's bounds are then the ledger's slots) and which stay
tokens (type `k_pool_card`, kind `filtered`):

- fold: every row of a `HAND` or `SIDEBOARD` pool; every row of a player's library pool when it is
  the player's only `LIBRARY_RANGE` pool (one undetermined run: its bounds are the ledger's library
  slots less the exact positions, exactly);
- stay: the rows of every library pool when a player has several (the runs after a put-back or a
  scry, and the blocks, which carry joint facts the per-card sums cannot), and of `FACE_DOWN` and
  `SET_ASIDE` pools.

Every `k_pool` row stays a token (its owner, kind, size, range and handle), so a `k_position.pool`
reference keeps its relation. A folded `k_pool_card` row's relation to its pool is replaced by the
slot it lands in (its pool is the `HAND` / `SIDEBOARD` / only library pool of the token's player); a
kept row's `pool` relation points at its pool's token as before (row references into a filtered
type are renumbered to its tokens).

**Object deltas** (types `object_subtype`, `object_keyword`, kind `delta`): the card table already
holds each face's printed subtypes and keywords (§4.1: `card_subtype.subtype=…` multi-hot,
`card_keyword.kw=…` counts), so an object's rows are emitted only where they differ from its
**current face's printed rows** (`object.card`, its back face when `transformed`: the token's
`card:face` slot), compared as multisets of the rows' columns other than `obj` (`subtype`; `kw` and
`text`, the keyword's parameters included), read from the `CARDS` frame's `card_subtype` /
`card_keyword` rows of that face:

- a row the object has and the printed rows lack is a token, `removed` = 0 (a granted keyword, an
  added creature type, a different keyword parameter);
- a printed row the object lacks is a token with that row's values and `removed` = 1 (a keyword or
  subtype lost);
- an object whose face the viewer cannot see (face down, `card` none), whose face has no card row
  (a back face with no vocab id of its own, a token with no card row), keeps every row with
  `removed` 0; so do rows whose handle no object row has.

So the object's rows are its printed rows plus the `removed` = 0 tokens less the `removed` = 1
ones. Tokens come per object row in frame order, the added rows sorted by value, then the removed
ones. The printed rows are not an exported table: the featurizer builds them from the whole-vocab
`CARDS` frame it builds the card tables from (`build_printed_rows`, `card_tables.h`; with tables
read from files, from the cards file given beside them).

**Checks** (the `featurize` tier, every decision of the corpus's and the explore games' dumps, seat
and omniscient frames): from the tensors alone each object's subtype and keyword rows rebuilt (its
face's printed rows plus and less the delta tokens) equal the frame's; every `own_deck`, `k_seen`,
`k_known_hand` row and every folded `k_pool_card` row has its ledger token with its values (the
library slots with the exact positions added), the kept `k_pool_card` tokens are the unfolded rows
and there is no other ledger token. The row-order check (§7) shuffles the folded tables too and finds every token by its origin
(`featurize_traced`), so the canonical orders are tested. Deliberately broken folds (half the
removed rows dropped; a sideboard maximum off by one) fail it.

**Sizes** (explore games, release build; seat decisions with a menu): bo3 burn vs uw_control
(209 decisions): 263.5 → 126.0 tokens per decision (max 432 → 257); five league matchups
(813 decisions): 306.8 → 157.4 (max 611 → 353); their omniscient frames 400.5 → 247.6.

### 3.7 Recent-event history (an option, off by default)

The network has no recurrence, and a `STATE` frame's event tables hold only the events since the
viewer's previous frame, so by default the featurizer leaves them out: what a seat remembers is the
knowledge sidecar's facts. The history option (`FeatureOptions::history_events` N > 0; `rmfeat
--history N`, the engine's `--feature-history N` for the trajectory writer, `nn:` and
`--featurize-seats`, `python -m rmlearn.corpus --refeaturize … --feature-history N`) adds the seat's
recent observation events as tokens. Off, every output is what it is without the option: the spec
document, its hash, the tensors and the episodes, byte for byte. On, the spec gains two token types
and a `history` key, so its hash differs and a model trained one way is refused by the other.

- **The event history** (`featurize::EventHistory`, `src/featurize/history.h`): kept per seat and
  match by whoever featurizes (the trajectory writer per recorded seat, the `nn:` agent through the
  agent hook `observe`, which sees every frame of its seat before any script or agent decides it,
  `rmfeat` per seat over a dump), fed every frame of the seat in order, those without a menu too. It
  holds the events since the last `game_start` whose turn is at least the latest frame's turn less
  one, each with its `event_card` rows (`event_player` rows are not kept). Its contents depend on the
  events received, not on how they were batched into frames, so a dump (every viewer's frame at
  every decision), the `--machine` stream and the in-process feed give the same tokens.
- **Selection**, per decision: the events of the frame's turn (`turn.turn`) and the previous one,
  newest first, at most N; each with its first `cards_per_event` (8) cards in frame order. A search
  is also a `look` at the whole library (some 50 cards), hence the per-event cap.
- **Token types**, after the folds' synthetic ones: `hist_event` (kind `history`), the `event`
  table's columns by their tags (§3.2), plus three numeric slots: `turn_offset` (0 this turn, 1 the
  previous one), `age` (`log1p` of the newest-first rank) and `more_cards` (`log1p` of the cards past
  the cap); `hist_card`, the `event_card` table's columns, its `event` column a relation to its
  event's token.
- **Left out** (the spec's `history.dropped`): the columns that identify the game rather than
  describe the event: `event.seq`, the absolute `event.turn`, `event.game`, `event.batch`,
  `event_card.draw_seq` and `event_card.draw_idx`. Handles are never values (§3.3).
- **Relations.** `hist_event.source`, `hist_card.handle_before` and `handle_after` relate the token
  to the object holding the handle now: each handle is followed through the old → new links of the
  kept events' moves (a handle is never issued again, so the links form chains); a handle that was
  retired (the card became hidden) or whose object left the frame relates to nothing.
- **Learner side.** `featurize_seat` refuses a history spec without the seat's event history;
  `featurize_viewer` without one emits no history tokens (the omniscient featurization of an
  episode has none). BC's `event_dropout` p (`learn/rmlearn/bc.py`) masks each event token out in
  training with probability p, its card tokens with it.
- **Cost** (val1's B0 games, scripted:hard on ur_delver / burn / uw_control / wrb_energy, N 32):
  about 30 more tokens per decision (some 15 events and 15 cards; 121 → 152 tokens in one bo3
  match), relations 21 → 53; an episode with the omniscient part grows ~9%.

## 4. The card tables

Three whole-vocab tables, the same for every match, indexed by vocab id (`src/card_vocab.h`; ids
`0 … N_CARD_TYPES-1`, the token band from 900, the token sentinel 1023). They are built from a
**whole-vocab `CARDS` frame** (the engine's `--cards-frame FILE`: `SCHEMA` and `CARDS` records, the
`--closed-decklist` card set) by `rmfeat --export-tables` (`src/featurize/card_tables.h`),
by the same tag rules. The featurizer reads `N_CARD_TYPES + 1` from the text table's row count,
never from an engine header. `rmfeat --export-tables CARDS DIR` writes `card.rmtb`, `ability.rmtb`,
`part.rmtb`, `spec.json` (§6), and the tables' model manifest `manifest.json` and layout registry
`registry.json` (§6.2).

### 4.1 Card table `[N_CARD_TYPES + 1, Fc]` f32

One row per **face vocab id**: the `card` row whose `face_card` is that id (a front face's
`face_card` is its own id; a back face's is the back face's id). Columns:

- the `card` row's numeric and mask columns by §3.2 (mana value, pips, colours, types,
  supertypes, P/T, loyalty, token) and its enums one-hot (`face`, `layout`);
- per `(card, face)` child table of `CARDS` (`card_subtype`, `card_keyword`, `card_pip`), the count
  of the face's rows per value of each enum column: multi-hot over `subtype` and `keyword`, per
  `pip_kind` counts;
- `n_abilities`: `log1p` of the face's `card_ability` rows;
- a `present` slot (0 for an id with no card).

Columns are named `<col>`, `<col>:<VALUE>` (a mask bit), `<col>=<VALUE>` (one-hot) and
`<table>.<col>=<VALUE>` (a child count). Every column sits at its registry position (§6.1): an enum's
one-hot, mask or child counts are a block of the enum's capacity (value `v` at the block's start plus
`v`'s registry index), and the table's width is its registry capacity; a reserved column reads 0 and
is named `<entry>#<i>` (a block's position past its registered values) or `#<position>` (past every
entry). Row `N_CARD_TYPES` is the none row (all zeros): a `-1` card slot reads it. The file also holds `back_face` (i64, a row per card id): a
physical card's back face's vocab id (itself when it has none, -1 for a back face with no id of its
own), which gives `object.card`'s current-face slot (§3.4).

### 4.2 Text table `[N_CARD_TYPES + 1, 384]` f32

The oracle-text vectors (plan 5c), one row per face vocab id with oracle text, zero rows
elsewhere and at the none row. Committed (§5).

### 4.3 Ability table `[Nab, Fa]` f32 and part table `[Npart, Fp]` f32

- **Ability rows** are stable addresses, never vocab positions: card id `c`'s ability of index `i`
  is row `c × A + i` (`A` = the registry's `abilities_per_card`, 16; a card with more abilities is
  refused, a layout bump), a row no ability has is all zeros, and the **key band** from
  `key_base = N_CARD_TYPES × A` (16384) holds one row per `ab_key` of the spec's fixed key list
  (abilities with no card row: granted or engine-built) at the key's registry row, capacity 256
  (rows 16384-16639). Appending a card fills its own rows only, so a stored `tok_ability` keeps
  naming the same ability whatever is appended to the vocab. The table is data (no parameters). Columns: the row's by the tag rules (`face`, `ability_kind`, `category`, `timing`
  one-hot; `tap`, `limit`; the structured parameters, `state_format.md` register 41: amount and
  its kind, power/toughness modifiers, target classes, target bounds; `n_parts`, `charm_num`) and
  `keyed` (1 for a key row).
- **The fixed key list**: every distinct non-empty `card_ability.key` of the frame (keyword-built
  abilities: `Equip`, `Cycling`, `Reconfigure:unattach`), plus the key forms Appendix A.1 gives
  engine-built definitions that the `SCHEMA` frame enumerates: `keyword:<k>` for each `keyword`
  value (keyword-derived triggers), `subtype_mana:<c>` for each `color` value (a land type's mana
  ability) and `LandType`; in key-row order. A key row that is some card row's key has the features
  of the first such row in `(card, index)` order (a card appended at the next free id never becomes
  it); the others only `keyed`. Keys an abref can carry outside the list (a granted
  trigger's `granted_trigger:<script>`, `enchant:<filter>`, the engine's delayed-trigger keys such
  as `return_from_exile:<n>`) are built during play from text no frame enumerates; they resolve to
  `-1`, and the hashed `ab_key` categorical (§3.4) still tells them apart.
- **Part rows**: one per `card_ability_part` row (sub-abilities and modes), in `(card, ability,
  part)` order, with the same parameter columns, the part kind, depth and parent, and `ability`:
  the ability row it belongs to (an i64 array of the file, not a feature). Their order may shift
  when a card is appended: they are pooled into their ability rows inside the tables and never
  referenced by a token.
- **The ability index** is in the ability file: `ab_card` and `ab_index` (i64, per row its card
  and index; -1 for a key row or an unused row), `key_rows` (i64, per key its row; the array's
  `columns` are the keys) and the attrs `abilities_per_card` and `key_base`. The featurizer resolves `tok_ability` with it: the row of `(ab_card, ab_index)`, else of
  `ab_key`, else `-1`.

### 4.4 Files: RMTB

Every exported table is an RMTB file of one or more **named arrays**, readable with any language's
standard library (the trajectory episodes, `docs/trajectory_format.md`, reuse it):

| Part | Layout |
|---|---|
| magic | 4 bytes `RMTB` |
| version | u32 little-endian (1) |
| meta length | u32 little-endian, bytes of the JSON that follows (padding included) |
| meta | UTF-8 JSON, keys in a fixed order, padded with spaces to a multiple of 8 bytes |
| data | the arrays' values, little-endian, row-major, each array at an 8-byte-aligned offset from the start of the data (zero padding between; the file ends with the last array) |

Meta keys, in this order, each left out when it does not apply: `kind` (`card`, `text`,
`ability`, `part`, `tensors`, …), `vocab_hash` (FNV-1a 64 over the vocab's `(id, name)` pairs in
ascending id, each pair as the id's u32 little-endian bytes, the name's UTF-8 bytes and a 0 byte;
16 hex digits), `spec_hash` (§6), and for the text table `model` (the encoder's repository and
pinned commit, `<repo>@<commit>`; an episode carries it too), `pooling` (`cls`, `mean`),
`norm_version` (the text normalisation, §5) and `rows_meta` (per present row, ascending: `{"id":
<vocab id>, "hash": <FNV-1a 64 of its normalised text, 16 hex digits>}`); `attrs`, an object of
named strings and integers in the writer's order (a file kind's own header fields: an episode's,
`docs/trajectory_format.md` §3.1; the ability table's `abilities_per_card` and `key_base`); then
`arrays`: one
`{"name", "dtype", "shape": [...], "offset"[, "columns": [...]]}` per array in file order (`shape`
of any rank, `[]` a scalar; `offset` from the start of the data, the previous array's end rounded
up to a multiple of 8; `columns`, optional, names the last dimension's entries). The same content
always encodes to the same bytes.

Dtypes: `f32`, `f16` (IEEE binary16, written with round-to-nearest-even), `i64`, `i32`, `i16`,
`u16`, `u8`. Ragged data is two arrays, the values and an `i64` offsets array (`n + 1` entries; row `r`
is `values[offsets[r]:offsets[r+1]]`), with no padding on disk. The text table is one array,
`vectors` `[N_CARD_TYPES + 1, hidden]` f32, with `rows_meta` in the meta.

Readers: `src/featurize/rmtb.{h,cpp}` (C++, std only; also the writer and the f16 conversions),
`python/rmtb.py` (Python's `struct`, f16 as format `e`, i16 as `h`); numpy reads an array with
`np.frombuffer(buf, dtype, count, offset=12 + meta_length + array_offset)`.

## 5. The text table and `make`

- **Entries.** One per vocab id (`src/card_vocab.h`, card entries and the registered token band)
  whose face has oracle text: a front face under its own id, a double-faced card's back face under
  the back face's entry, a split card (both halves under one id) as both halves, a token from its
  token script; read from the parse cache, no entity created. Engine side: `src/text_table.{h,cpp}`;
  the std-only rules: `src/featurize/oracle_text.{h,cpp}`.
- **Text.** `"<type line>. <oracle>"`, a split card's halves joined with `" // "`. The type line is
  the face's `Types:` line in script order, placed as printed: supertypes, `Token` for a token,
  card types, then ` — ` (an em dash) and the subtypes. In the oracle, every whole-word occurrence
  of the face's name, the other face's name and each one's part before its first `, ` (the short
  name oracle text uses for a legendary card: "exile Ajani") becomes `CARDNAME`, longest first;
  the type line keeps its words (a planeswalker's subtype is its short name). Then newlines and
  tabs become spaces, runs of spaces collapse, the text is trimmed and the pin's `prefix` put
  before it (none for the bge and MiniLM families). This is normalisation version 1
  (`featurize::NORM_VERSION`); any change bumps it, which makes every row stale. The row's hash is
  FNV-1a 64 of the text's UTF-8 bytes (`featurize::text_hash`).
- **The pin**, `tools/embed/MODEL_PIN`: `key value` lines `repo`, `commit` (full SHA), `pooling`
  (`cls` or `mean`, as the model was trained) and `prefix` (a JSON string). The table's `model`
  (`<repo>@<commit>`), `pooling`, `norm_version` and `vocab_hash` must equal the pin's and the
  vocab's, else every row is stale: a pin change regenerates the whole table.
- **Committed** at `bin/resources/embeddings/oracle_text.rmtb` (§4.4: kind `text`, array
  `vectors` `[1025, 384]` f32, ~1.5 MB dense).
- **Check**: `robomage --text-table-check FILE --model-pin PIN` (std only, no network) exits 0 when
  every entry has a `rows_meta` row with its current hash, no other row is listed and the header
  matches; otherwise it prints `TEXT-TABLE: stale <id> <name>` per missing or stale entry
  (`extra <id>` per listed row that is no entry) and exits 1. A missing table is all stale.
- **`make text-table`** builds `robomage` and runs the check. On success nothing else runs.
  Otherwise:
  1. `robomage --text-table-inputs .cache/embed/text_table_inputs.jsonl --text-table TABLE`
     writes the stale entries as JSON lines, after a header with the row count, vocab hash, model,
     pooling, normalisation and every present entry (`featurize::text_inputs_jsonl`);
  2. `python3 tools/embed/fetch_model.py` (stdlib) downloads the pinned model's `config.json`,
     `vocab.txt`, `tokenizer_config.json` and `model.safetensors` into
     `.cache/embed/<repo, / as __>/<commit>/` (the weights checked against the repository's
     SHA-256);
  3. `make rmembed` (libtorch, found as `make actor` finds it; C++20, which libtorch's headers
     require) and `bin/<config>/rmembed --model DIR --pin PIN --inputs FILE --table TABLE`
     tokenises, encodes, pools and normalises those entries and rewrites the table, keeping every
     other present row's bytes, zeroing rows no longer present; it then prints each encoded row's
     three nearest rows for review;
  4. the check again.
  Without libtorch or network it fails, naming the stale entries, the step and the command; it
  never leaves the table half written (the file is replaced by a rename). `all` depends on it,
  so every `make` runs the check; a clean checkout with a current table never downloads the model
  or needs libtorch, and a stale table fails CI.
- **Tokenizer** (`src/featurize/wordpiece.{h,cpp}`, std only): BERT basic tokenization as the
  reference fast tokenizer does it (special tokens matched in the raw text; cleaning; CJK split;
  lowercasing and accent stripping for an uncased model; whitespace and punctuation split), then
  greedy longest-match WordPiece with `##` continuations and `[UNK]`; `[CLS] … [SEP]`, never
  truncated (`rmembed` fails on a text past the model's positions; the longest card text is 153
  tokens). Its Unicode tables cover ASCII, Latin-1, Latin Extended-A, the combining marks, General
  Punctuation, Mathematical Operators and the unified CJK blocks; a text with any other character
  is refused, never tokenised on a guess.
- **Encoder** (`tools/rmembed/`): the BERT encoder in the libtorch C++ API over the safetensors
  weights read directly (F32, F16, BF16; names with or without `bert.`; no TorchScript, no
  Python): embeddings, LayerNorm, per layer masked self-attention, dense + residual + LayerNorm,
  exact-GELU intermediate, dense + residual + LayerNorm; `cls` or `mean` pooling; L2
  normalisation. Each text is encoded alone on one thread, so a row depends on its text only.
- **Goldens** (`tools/embed/make_goldens.py`, needs torch + transformers, never run by `make`):
  `tokens` records `tests/embed/tokens_golden.jsonl` (committed): every entry's ids from the
  pinned `tokenizer.json`, cross-checked against `vocab.txt` + `tokenizer_config.json`, plus hard
  and coverage strings (every character of the tokenizer's tables); the gate's `vocab` tier checks
  the C++ tokenizer against it id for id when the model's vocab is cached (else it prints a skip).
  `encoder` builds a seeded random BERT from the pinned config at a small size (2 layers, hidden
  64), saved as F32 or as F16 with `bert.` names, or takes the real model (`--model DIR`), and
  compares `rmembed --embed-test` (batch 1 and padded batches) with `BertModel` in eval, both
  poolings; `table` compares a generated table with the reference vectors of its texts. Each
  requires cosine ≥ 0.99999.

## 6. The spec and the network contract

**`FEATURE_SPEC`** (version 4) is a JSON document `rmfeat --spec` writes
(`src/featurize/feature_spec.h`) from the whole-vocab cards file's `SCHEMA` frame, the card tables
its `CARDS` frame gives and the layout registry (§6.1; the actor builds it in memory from the same
inputs): the layout's version and `layout_hash`; the schema and vocab hashes; the hashed-text
bucket count; the targets' library depth; the registry's dims `F`, `C`, `K`, `Q`, `G`, the
categorical id space's size (`categorical_size`), `type_capacity` and `relation_capacity`; the menu's
token type; the token-type list (the types the schema makes, by registry `id`), per type its
numeric block in `G` (`num_at`, `num_capacity`) and its numeric slots (name, source column,
transform, column `at`), categorical slots (name, source, `col`, id offset, size), card slots
(`col`), ability slots (`col`) and relation types; the relation types `(table, column, target)` by
registry id; `enums`, the registry's values of every enum a categorical or mask slot reads (value
`i` is a slot's id `offset + 1 + i`); the §3.4 overrides; the §3.6 folds (per type its `kind`:
`table`, `filtered`, `delta` with the `printed` table it compares with, `ledger`, `history`; a
synthetic type's `table` is null); the card, text, ability and part tables' rows, widths and column
names (every position, reserved ones included; the ability table with `abilities_per_card` and
`key_base`); the fixed key list (`ab_keys`) and the keys' rows (`ab_key_rows`); with the history
option (§3.7), a final `history` key (the caps, the two types and the dropped columns). One line,
its keys in a fixed order. Its `spec_hash` (FNV-1a 64 of the document bytes: the exact identity) is
in every exported file and in a model; `layout_hash` names the layout alone (§6.1). A JSON document
rather than an RMSTATE frame, so consumers outside C++ read it with any JSON parser.

### 6.1 The layout registry

**The layout is the contract, the vocab is data** (`docs/open_vocab_plan.md` S2). Every position a
network's parameters are indexed by is fixed by one committed, append-only file,
`src/featurize/registry.json` (`src/featurize/registry.h`; compiled into every binary that
featurizes: the assembler's `.incbin` in `registry.cpp`, the Makefile making its object depend on
the file), rather than by a running offset over today's schema and vocab. Its
sections, every entry matched by name:

| Section | Gives | Capacity |
|---|---|---|
| `dims` | `F` 96 (numeric columns per token), `C` 12, `K` 6, `Q` 2 (categorical, card, ability columns per token), `G` 1408 (every type's numeric block side by side), `types` 64, `categorical` 6144 ids, `relations` 64, `ab_keys` 256, `abilities_per_card` 16, `hash_buckets` 256 | fixed |
| `enums` | per enum a slot or column reads: its values (index = position) | closed enums (`face`, `seat`, `step`, `color`, `layer`, `day_night`, `lib_anchor`, `subtype` 518, `card_type`, `supertype`) their size; open ones max(8, the power of two ≥ 2n): `keyword` 128, `effect_kind` 256, `action_category` 128, … |
| `token_types` | name → id, the type's numeric block (`num_at`, `num_capacity` ≤ F) | `types` |
| `numeric` | per (type, slot) its column in the block; a mask an enum-capacity block | the type's `num_capacity`: the slots used rounded up to 8 with room for a quarter more (at least 8) |
| `categorical` | per (type, slot) its `tok_cat` column and its id range (`offset`, `size`: an enum's capacity + 1, a player-rel's values + 1, a hashed column's buckets + 1) | `C` columns, `categorical` ids |
| `cards`, `abilities` | per (type, slot) its `tok_card` / `tok_ability` column | `K`, `Q` |
| `relations` | per (table, column, target) its relation type id | `relations` |
| `tables`, `columns` | per card / ability / part table its width; per column its position (an enum's one-hot, mask or counts an enum-capacity block) | card 960, ability 448, part 448 |
| `ab_keys` | per fixed ability key its row in the key band | `ab_keys` |

Today's file was generated from the S1 layout in its order (so every slot keeps its order within its
type, every relation type and key row its index, every enum value its index; only offsets and widths
moved) and uses 856 of G's 1408 columns, 3902 of 6144 categorical ids, 38 of 64 relation types, 35
of 64 type ids, 73 of 256 key rows, and the card, ability and part tables' 720, 340 and 320 columns.
`build_spec` and `build_card_tables` place every slot, column and row at its registry position,
leave reserved positions 0 (-1 for card and ability slots) and **fail**, naming the entry and `make
registry`, on a value, column, slot, type, relation or key the registry lacks; past a capacity (an
enum full, a type's block full, more than `abilities_per_card` abilities on a card, …) they fail
naming the capacity: that is a layout bump. An entry the schema no longer makes is a hole that reads
0; an engine enum may be reordered.

- **`make registry`** (`rmfeat --registry-update FILE --cards CARDS`) appends what the engine's schema
  and whole-vocab tables need at the next free positions (an enum value after its enum's last, a slot
  at its type's end, a categorical range at the id space's end, a new type's block after the last,
  with its capacity from its use), never moving an entry, then rebuilds; commit the file.
- **The check** (the gate's `vocab` tier, `rmfeat --registry-check FILE --cards CARDS --base HEAD's`):
  the file parses (no overlap, every position within its capacity) and is in its canonical form, is
  the registry the binaries were built with, needs no `make registry`, and moves no entry of the
  committed file (`git show HEAD:src/featurize/registry.json`) unless `layout_version` was raised.
- **`layout_hash`** (FNV-1a 64 over `layout_version`, the dims, the tables' capacities, the card rows
  and the targets' library depth; in the spec and every episode's attrs) is the same for every spec of
  one layout whatever was appended to it; `spec_hash` stays the exact identity. A deliberate change
  that is not an append (a capacity, a moved entry) raises `layout_version` by hand and is migrated
  by name like the S5 cut-over (`docs/open_vocab_plan.md`).

The gate's `featurize` tier tests the property (`tools/robomage_check/featurize_layout.h`): a card
appended at the next free id and a fake keyword appended to the registry keep the `layout_hash`,
every position and every existing card's, ability's and part's rows, and featurize every corpus
decision to the same tensors; the keyword enum reordered (the fake keyword first, the frames' keyword
columns shifted) gives the same positions, card table and tensors; a full enum, a card past
`abilities_per_card` and a missing value are refused.

**Network contract** (the step-5 actor; a policy in any framework follows the same tensors):

```
forward(tok_type [B,N] i64, tok_num [B,N,F] f32, tok_cat [B,N,C] i64, tok_card [B,N,K] i64,
        tok_ability [B,N,Q] i64, tok_mask [B,N] bool, rel [B,R,3] i64, rel_mask [B,R] bool,
        act_tok [B,A] i64, act_mask [B,A] bool) -> (logits [B,A] f32, value [B] f32)
```

The card, text, ability and part tables are the model's frozen buffers, loaded from the RMTB
files when the model is built; an export folds them in and carries the tables' model manifest
(§6.2), and the actor refuses an export that is not of its own tables exactly.
Logits at padded menu rows are ignored; the actor takes the argmax (or samples) over real rows.

**Padding** (`featurize::pad_batch`, `src/featurize/batch.h`, std only): a batch is padded to its
largest `N`, `R` and `A`, each rounded up to a multiple of 8 and at least 8 (so no dimension is 0 or
1, which `torch.export` would specialise); pads are 0 (`tok_card`, `tok_ability` -1) and every mask
0; `rel_mask` marks the real relations. The buffers are reused from batch to batch.

**Model files and their metadata.** A model carries seven strings: `spec_hash`, `vocab_hash`,
`layout_hash`, `tables_digest` (16 hex digits each), `text_model` (the text table's `model`), `vocab` (the vocab
manifest, a JSON array of `[id, name]`, §6.2) and `policy_version` (an unsigned integer, `"0"` when
absent), each in its export's own metadata, never in a sidecar file: a TorchScript
module's string attributes, an AOTInductor package's metadata (written by the export's inductor
config `aot_inductor.metadata`, read with `AOTIModelPackageLoader::get_metadata`) and an ONNX
model's `metadata_props`. The evaluator tells the two it loads apart by the zip archive's records
(an AOTInductor package has `archive_format`, a TorchScript module `constants.pkl`), not by the
file name. An AOTInductor package holds machine code compiled for the exporting CPU's ISA
(`AOTI_CPU_ISA` in its metadata); TorchScript is portable.

**Export-friendly.** Inputs are padded tensors and masks only, and a conforming model has no
control flow that depends on the data: real rows are selected by masks, relations applied by
gathers and scatters, never by loops over real rows or shape-dependent branches. So one model
exports as TorchScript, as a `torch.export` / AOTInductor `.pt2` package and as ONNX; the
actor's evaluator loads the first two, ONNX is for other runtimes. TorchScript is in maintenance
mode upstream and is not the contract.

**Batched evaluation.** The actor evaluates through `actor::Evaluator::evaluate(batch, &out,
&err)` (`src/actor/evaluator.h`): a batch of featurized decisions (from one game or many),
returning per decision an `Evaluation`: the value and, per policy the evaluator serves (policy 0
first, the one the actor samples from), the logits over its menu rows, their `log π` (log-softmax
over the menu rows), its value and the policy's version. A failure is `false` with a message, never
a default action. Implementations: in-process (`load_model_evaluator`: the actor's model, CPU, one
thread, TorchScript through `torch::jit::load` or AOTInductor through `AOTIModelPackageLoader`; one
policy) and remote (`connect_remote_evaluator`, `src/actor/remote_evaluator.h`).

**Remote evaluation** (`nn:remote:<endpoint>`) speaks the inference protocol,
`docs/inference_protocol.md` (RMIP v2; the wire code is `src/inference/protocol.h`, standard
library and POSIX): a Unix stream socket at the endpoint carries a handshake (the featurizer's
model manifest without the registry, §6.2, and widths, the policy ids to evaluate) and small
`EVAL` / `DONE` messages; each connection has a POSIX shared-memory region the client creates,
names in the handshake and unlinks once the server has it, holding the request (the decisions'
unpadded tensors, laid out as an RMTB `tensors` file's arrays with per-decision sizes for offsets,
the spec hash in its header) and the response (per requested policy its version, the values, and
the logits and `log π` over each decision's menu rows). The server may batch requests across
clients up to a size or a deadline, refuses a handshake whose tables are not its models' exactly
(§6.2: an export is rebuilt per vocab) or a request whose spec hash differs from the handshake's
models', and answers each request with each policy's current version, so weights can be swapped
between decisions. The client fills one `Evaluation` entry per requested policy (`policies=`, the
first the one sampled from). A server that refuses, errs, goes away or does not answer within the
timeout ends the actor with a fatal error naming the endpoint, never a default action. `rmserve`
(`tools/rmserve/`, C++, libtorch, CPU, built by `make actor`) is the reference server: several
policies (`--policy <id>=<model>`, `--version <id>=<n>`) each through the in-process evaluator,
many clients on one poll loop, batches up to `--max-batch` decisions, or `--deadline-us` after the
oldest queued request, or as soon as every connected client has a request queued; an
`RMSERVE-STATS:` line on SIGTERM. `tools/nn/serve_check.py` is a minimal server in Python
(`socket`, `struct`, `multiprocessing.shared_memory`, numpy, torch) that proves the protocol needs
no C++; the training plan's GPU server (L5) implements the same protocol.

**The `nn:` agent** (`src/actor/nn_agent.h`). `nn:<model>[:<key>=<value>…]`, keys `temp=<t>`
(sampling temperature, default 1; drawn with the agents' shared generator, so a game replays from
its seed) and `argmax` (the largest logit; rows within 1e-5, absolute plus relative, of it tie and
the first is taken, so float drift between two exports of one model cannot reorder rows equal in
exact arithmetic, such as two identical cards in hand), DeepNash's play-time projection
(`src/agents/policy_projection.h`, OpenSpiel's `rnad.py` `_threshold` / `_discretize`):
`thresh=<p>` (rows below p after temperature dropped and the rest renormalised, none dropped when
every row is below p; default 0, off) and `disc=<n>` (then the probabilities rounded to multiples
of 1/n summing to 1, the rows taking `ceil(p·n)` in descending order until n is spent; default 0,
off), applied in that order before the draw; plus three for tests and the bench:
`dump=<file>` (every decision it featurized, written at exit as one `tensors` file, the bytes
`rmfeat --featurize` writes for that seat), `evals=<file>` (every evaluation it received, at exit,
as an RMTB file of kind `evaluations`: per decision and policy the version and value, per menu row
and policy the logits and `log π`, per menu row the behaviour distribution `mu` and per decision
the row taken `chosen`) and `stats` (an `NN-STATS:` line at exit; remote: also the requests,
the server compute time attributed to them and the mean server batch); an unknown key fails
loudly. `nn:remote:<endpoint>` takes the same keys plus `policies=<id>,…` (default `0`),
`timeout=<s>` (default 60) and `shm=<MiB>` (the request area, default 32), and needs no model file;
the featurizer's inputs are built as for `nn:<model>`. A sideboarding decision goes through the
same path as any other. It keeps the exact distribution it sampled from over every menu row
(after temperature and any projection; one-hot under `argmax`), the policy version, every
requested policy's evaluation and the tensors it featurized, for its last decision
(`Agent::last_behaviour`, `agents/agent.h`), for the trajectory writer (`--trajectory-out`,
`docs/trajectory_format.md`), which reuses those tensors rather than featurizing the frame again. The featurizer's inputs are built once, before the match, in
process from the engine (`load_engine_featurization`, `src/state_format/cards_file.h`: the
whole-vocab records `--cards-frame` writes and the committed text table), so they are what
`rmfeat` builds from that file; at each decision the agent reads only the seat's decoded `STATE`
frame. It runs in `bin/<config>/robomage_nn` (`make actor`): the engine's objects and entry
(`src/engine_main.h`) with the actor's TUs, whose `main` registers the `nn:` kind
(`agents::register_agent_kind`) first; the plain engine refuses `nn:`, naming `make actor`.

**Recommended model family** (the reference policy, `tools/nn/make_reference_policy.py`): per
token type a linear projection of `tok_num`, plus embeddings of `tok_cat`, plus projections of
the gathered card, text and ability rows, summed into a width-`D` token; an entity transformer
whose attention adds a learned bias per relation type for `(src, dst)` pairs; a pointer head (an
MLP over each `act_tok` token's output) for logits; a value head over the mean of real tokens. The
reference policy's `build` (torch; never run by `make`) reads the exported tables
(`rmfeat --export-tables`) and the text table, initialises the model from a seed (width 64, 4
heads, 2 layers; a bias per relation type at `(src, dst)` and another at `(dst, src)`, scattered
into `[B, H, N, N]` with `index_add`; the ability rows average their part rows in; the tables'
projections folded into frozen buffers, constants of the weights, so a forward gathers rows rather
than projecting the whole vocab) and exports it as
`<prefix>.ts.pt`, `.pt2` and `.onnx`; `check` evaluates the three on a `tensors` file's decisions
alone and in padded batches and requires them to agree within 1e-4 (absolute plus relative).

### 6.2 Compatibility: what a model was trained with (`src/featurize/compat.h`)

A checkpoint is the portable artefact; an export is built for one vocab and tables. Every model
carries the **model manifest** of the tables it was trained on (`rmfeat --export-tables` writes it
beside them: `manifest.json`, `registry.json`; a checkpoint's `hashes`, an export's metadata, the
inference protocol's handshake): `spec_hash` and `vocab_hash` (exact identity), `text_model`,
`layout_hash` (§6.1), `tables_digest` (FNV-1a 64 of the tables' card digests, below: every card's
card, ability and part rows), `vocab` — the **vocab manifest**, `[id, name]` for every card-table row (the
name the whole-vocab CARDS frame gives each face vocab id) — and, in a checkpoint, `registry` (the
layout registry's text: the manifest of positions). **One check**, `check_compatible(model, now,
exact)`, implemented once in C++ (`featurize::check_compatible`) and once in Python
(`learn/rmlearn/spec.py`), with the same verdicts (`learn/check.py` runs both on the same cases,
`rmfeat --compat MODEL NOW [--exact]`):

- **every model:** the same `layout_hash` and `text_model`; every `[id, name]` of the model's vocab
  present unchanged in the current vocab — a new id passes, a removed or renamed id is refused naming
  it; and, when both carry a registry, no entry of the model's moved (`registry_moves`, §6.1).
  `load_checkpoint`, the R-NaD learner's state and `_load_with_config`, the exploiter's and the
  exporter's loads use it, and a loaded model carries the current tables' manifest from then on;
- **exact** (an export, whose card and ability rows are folded in): also the same `spec_hash`,
  `vocab_hash` and `tables_digest` (a known card whose rows changed — drift — makes the folded rows
  stale) and no current id the model lacks. The evaluator (`nn:`), `rmserve` (its policies
  among themselves and against each client's handshake), the learner's server (`rmlearn/server.py`),
  `tools/nn/serve_check.py`, `rmtrain` and the benchmark's `export_missing` refuse or rebuild an
  export that is not of the current tables; the refusal says to rebuild it from its checkpoint.

The codes are the protocol's (`layout_hash`, `text_model`, `vocab_hash`, `spec_hash`,
`tables_digest`). The vocab manifest names what the tables hold, which the vocab hash alone does not:
the vocab hash is the text table's (`oracle_text.rmtb`), so a card appended without `make
text-table` leaves it and the spec hash as they were, and only the manifest shows the new id. A manifest
without `layout_hash` (a model made before manifests) passes only on its own tables' `spec_hash`,
`vocab_hash` and `text_model`.

**Table drift** (`src/featurize/drift.h`). A change to how the engine parses a known card passes
the check but changes that card's inputs. A card's rows are its card-table row, its ability rows
(`id · A + i`) and their part rows, as nonzero values by column name. The committed **card digest**
`src/featurize/card_digest.json` holds per card a hash of its rows (printed with 6 significant
digits, so a libm's last bit does not move it); the gate's `vocab` tier (`rmfeat --digest-check`)
fails when a known card's rows hash otherwise, or a card is new or gone, and `make registry`
rewrites it (`--digest-update`), so the commit's diff names the cards whose inputs changed.
`rmfeat --table-drift OLD NEW` compares two exported tables directories exactly, naming per changed
card the columns and their old and new values (`rmtrain retable`'s report, `docs/rmtrain.md` §1.1).

## 7. Tests

The gate's `featurize` tier (`docs/ci.md`) runs those of this list that need no model, over the
replayed corpus's dumps and explore games' (the text table's are the `vocab` tier's; the `nn:`,
export and `rmserve` ones come with 5d):

- deterministic: the same frames give identical tensors;
- the folds lose nothing (§3.6): the deltas and the ledger read back to the frame's rows;
- shapes and slot meanings agree with the spec; every `rel` and `act_tok` index is a real token;
  every `tok_card` is `-1` or a card-table row, every `tok_ability` `-1` or an ability row; every
  column no slot of the token's type uses is 0 (-1);
- the layout registry's append-only property (§6.1): a card and a keyword appended, the keyword enum
  reordered, the capacities refused;
- invariant to row order where the format says order is meaningless: the engine's permutation
  self-check (a permuted hidden zone) gives the same frame bytes and so, under
  `--selfcheck-featurize`, the same tensors; and every token table but `menu` (whose row is the
  reply), the folded ones included, with its rows shuffled and every `row:` reference into it
  renumbered gives the same tensors up to the matching token permutation (each token found by its
  origin: its table row, or its place in a fold's canonical order);
- the exported tables equal a standard-library Python reader's view of the same files;
- the text table check passes on a clean checkout; regenerating a deleted row leaves every other
  row's bytes unchanged; the tokenizer matches the golden ids and the encoder the golden vectors
  (cosine ≥ 0.9999);
- exact re-featurization: `rmfeat` over a `--replay` of an `nn:` game equals what `nn:` featurized;
- the history option (§3.7): off, no `history` key or type; on, the off types first and another
  hash, `rmfeat --spec --history` the same document; a corpus game's seat decisions featurized
  twice with history give the same tensors, which less their history tokens are the off tensors, the
  history tokens within the caps with each card token related to one event; the game's episodes
  written with `--feature-history` and `rmfeat --featurize --history` over its dump the same;
- isolation (§3.5): an `nn:` seat's tensors equal `rmfeat`'s for that seat's stream, unchanged by
  the scramble self-check; the actor entry point refuses an omniscient or other-seat frame;
- the reference policy's TorchScript, `.pt2` and ONNX exports give the same logits and values;
  in-process and `rmserve` evaluation agree (within 1e-5, with batches across clients), and
  `nn:remote:` through `rmserve` and through the Python server plays the in-process argmax games;
- the play-time projection: known answers and random menus (sums, dropped rows, multiples of 1/n,
  the most probable row kept, a fixed point), and a sampled `temp=0.02:thresh=0.03:disc=16` bo3 match whose
  every behaviour distribution is the projection of its logits, the row taken one with mass.

The gate's `actor` tier (`docs/ci.md`) runs the `nn:`, re-featurization, export and remote
evaluation checks when the actor binary and a Python with torch are available.
