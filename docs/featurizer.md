# The featurizer and the card tables (plan step 5a)

Design of plan step 5 (`docs/engine_fork_plan.md`): what a learned policy reads, how it is built
from one seat's RMSTATE frames, and the card tables every ML consumer shares. The implementation
is `src/featurize/` (std + the state-format decoder only), the tool `bin/<config>/rmfeat`, the
text-table generator `bin/<config>/rmembed` (libtorch) and the actor in `src/actor/`.

## 1. Principles

- **One seat's frames in, tensors out.** The featurizer is a pure function of the seat's decoded
  `SCHEMA`, `CARDS`, `METAGAME` and current `STATE` frame (`k_*` tables included) and the card
  tables (§4). It sees nothing else of the engine, so what a model can know is exactly what the
  seat can know (`docs/state_format.md` §1).
- **Driven by the schema's tags.** Every table and column is handled by its semantic tag
  (Appendix A.1 of `state_format.md`), so a new table or column reaches the model with no
  featurizer change. The few columns treated otherwise are listed in §3.4 and in the spec.
- **No card vector per decision.** Cards travel as vocab ids, which are row indices into the card
  tables; a model gathers the rows it needs.
- **Framework-neutral.** The tensors, the spec and the card tables are plain arrays and JSON,
  defined here without reference to any ML library. The TorchScript contract (§6) is one consumer.
- **Deterministic.** The same frames give the same bytes; row order follows the frame, which is
  canonical where the format says order is meaningless (tested: §7).

## 2. Inputs the featurizer reads

| Frame | Used for |
|---|---|
| `SCHEMA` | every table, column, tag and enum string table; the spec is built from it |
| `CARDS` | the match's cards (only to map a physical card and face to the face's vocab id; card features come from the whole-vocab tables) |
| `METAGAME` | the number of reference lists and archetype labels (categorical sizes in the spec) |
| `STATE` | the snapshot tables (ids 20-48) and knowledge tables (90-99); event tables (80-82) are not featurized (memory is the sidecar's job, `knowledge_store.md`) |

## 3. Per-decision tensors

A decision is a set of **tokens**, one per row of every featurized `STATE` table, plus relations
between them and the menu. All tensors carry a leading batch dimension `B` in the network
contract (§6); the featurizer emits `B = 1` and a batcher pads and stacks.

| Tensor | Shape | dtype | Contents |
|---|---|---|---|
| `tok_type` | `[N]` | i64 | token type: the table's index in the spec's token-type list |
| `tok_num` | `[N, F]` | f32 | numeric features, laid out per token type (§3.2); unused slots 0 |
| `tok_cat` | `[N, C]` | i64 | categorical ids in one global id space (§3.2); unused slots 0 (the pad id) |
| `tok_card` | `[N, K]` | i64 | vocab ids of the row's card columns, row indices into the card tables; `-1` none |
| `tok_ability` | `[N, Q]` | i64 | ability-table rows of the row's ability references (§3.3); `-1` none |
| `tok_mask` | `[N]` | bool | real token (false = padding) |
| `rel` | `[R, 3]` | i64 | relations `(src token, dst token, relation type)` (§3.3) |
| `act_tok` | `[A]` | i64 | the token of menu row `a` (menu row `a` = reply `a`) |
| `act_mask` | `[A]` | bool | real menu row |

`F`, `C`, `K`, `Q` are the maxima over token types, fixed by the spec. `N`, `R` and `A` vary per
decision; padding is a runtime option of the batcher, never a format cap.

### 3.1 Token types

Every `STATE` table with id 20-48 or 90-99 is a token type, in ascending table id, except
`strings`. Single-row tables (`header`, `turn`, `pending`, `pending_cost`, `match`, `k_meta`) are
one token each, so global state is a few tokens rather than a special vector. Child tables
(`object_keyword`, `object_counter`, `stack_target`, `k_pool_card`, …) are tokens of their own,
related to their parent (§3.3).

### 3.2 Columns by tag

| Tag | Becomes |
|---|---|
| `bool` | one numeric slot, 0/1 |
| `count` | one numeric slot, `log1p(x)` |
| `signed` | one numeric slot, `sign(x)·log1p(|x|)` |
| `prob` | one numeric slot, `x / 10^9` |
| `index` (a position: `pos`, `order`, `_pos` of a libpos) | one numeric slot, `log1p(x)`, `-1` → slot 0 and the none flag below |
| `mask:<e>` | one numeric slot per value of `e` (its string table's length) |
| `enum:<e>` | one categorical slot: the id range of `e` |
| `player-rel` | one categorical slot (none, self, opponent); the absolute `enum:seat` beside it is dropped, so a model is seat-symmetric |
| `card` | one `tok_card` slot |
| `handle`, `row:<t>` | a relation (§3.3), no feature |
| `text` | dropped, except the columns §3.4 hashes |

Every column whose cell can hold its none value (all ones: `card`, `index`, `row:`, nullable
`enum`) also gets a numeric none flag. The global categorical id space is the concatenation of
every (token type, categorical column) range, with id 0 the pad; one embedding table covers it.

### 3.3 Relations and ability references

- **Identity.** A table whose first column is a `handle` named `handle` (`object`, `stack`) owns
  that handle; `object` wins over `stack` (a stack object is an `object` row too).
- **Handle columns** (`obj`, `source`, `target`, `attached_to`, `attacker`, `blocker`, `for`, …)
  give a relation from the row's token to the token owning that handle, typed by
  `(table, column)`. A handle no token owns gives no relation (it names an object the viewer
  cannot identify).
- **`row:<t>` columns** give a relation to row `v` of table `t`'s tokens (`k_pool_card.pool`,
  `stack_target.lki`, `delayed_subject.trigger`, …), typed the same way. A `row:` into a frame the
  featurizer does not tokenize (`row:meta_list`, `row:meta_archetype`, into `METAGAME`) is instead
  a categorical slot sized by that frame's row count.
- **Ability references** (the `abref` triple `ab_card`, `ab_index`, `ab_key`: `object_ability`,
  `stack`, `menu`, `delayed_trigger`) become one `tok_ability` slot: the ability table's row for
  `(ab_card, ab_index)`, else its row for `ab_key` (§4.3), else `-1`.
- **Menu.** Each `menu` row is a token; `act_tok[a]` is the token of menu row `a`. Its `source` /
  `target` handles are ordinary relations, so a pointer head scores menu tokens after attention.

### 3.4 Columns treated otherwise (listed in the spec)

| Column | Treatment | Why |
|---|---|---|
| `object.card` | also a second `tok_card` slot: the face as it is now (the back face's vocab id when `transformed`) | the card tables are per face |
| `object_counter.kind`, `player_counter.kind`, `object_keyword.text`, `player_effect.filter`, the `ab_key` of an abref | categorical: FNV-1a 64 of the text modulo 256 buckets | counter kinds and keyword parameters have no enum |
| `menu.description` | dropped | free text, covered by the structured columns |

## 4. The card tables

Three whole-vocab tables, the same for every match, indexed by vocab id (`src/card_vocab.h`; ids
`0 … N_CARD_TYPES-1`, the token band from 900, the token sentinel 1023). They are built from a
**whole-vocab `CARDS` frame** (the engine's `--cards-frame FILE`: `SCHEMA` + `CARDS` records, the
`--closed-decklist` card set) by `rmfeat --export-tables`, by the same tag rules.

### 4.1 Card table `[N_CARD_TYPES + 1, Fc]` f32

One row per **face vocab id**: the `card` row whose `face_card` is that id (a front face's
`face_card` is its own id; a back face's is the back face's id). Columns:

- the `card` row's numeric and mask columns by §3.2 (mana value, pips, colours, types,
  supertypes, P/T, loyalty, token) and its enums one-hot (`layout`);
- multi-hot over `subtype` from `card_subtype`, over `keyword` from `card_keyword`, and per
  `pip_kind` counts from `card_pip`, each for that face;
- `n_abilities`: `log1p` of the face's `card_ability` rows;
- a `present` slot (0 for an id with no card).

Row `N_CARD_TYPES` is the none row (all zeros): a `-1` card slot reads it.

### 4.2 Text table `[N_CARD_TYPES + 1, 384]` f32

The oracle-text vectors (plan 5c), one row per face vocab id with oracle text, zero rows
elsewhere and at the none row. Committed (§5).

### 4.3 Ability table `[Nab, Fa]` f32 and part table `[Npart, Fp]` f32

- **Ability rows**: one per `card_ability` row of every vocab card, in `(card, index)` order,
  then one per `ab_key` of the spec's fixed key list (abilities with no card row: granted or
  engine-built). Columns: `ability_kind`, `category`, `timing` one-hot; `tap`, `limit`; the
  structured parameters (`state_format.md` register 41: amount and its kind, power/toughness
  modifiers, target classes, target bounds); `n_parts`.
- **Part rows**: one per `card_ability_part` row (sub-abilities and modes), with the same
  parameter columns, the part kind and depth, and `ability`: the ability row it belongs to (an i64
  column of the file, not a feature).
- The ability index `ability_index.json` maps `"<card>:<index>"` and `"key:<ab_key>"` to rows;
  the featurizer resolves `tok_ability` with it.

### 4.4 Files: RMTB

Every exported array is one RMTB file, readable with any language's standard library:

| Part | Layout |
|---|---|
| magic | 4 bytes `RMTB` |
| version | u32 little-endian (1) |
| meta length | u32 little-endian, bytes of the JSON that follows |
| meta | UTF-8 JSON, keys in a fixed order, padded with spaces to a multiple of 8 bytes |
| data | `rows × cols` little-endian values of `dtype`, row-major |

Meta keys: `kind` (`card`, `text`, `ability`, `part`, `tensors`), `dtype` (`f32`, `i64`),
`rows`, `cols`, `columns` (one name per column), `vocab_hash` (FNV-1a 64 over the vocab's
`(id, name)` pairs, hex), `spec_hash` (§6), and for the text table `model` (the encoder's
repository and pinned commit), `pooling`, `norm_version` (the text normalisation, §5) and `rows_meta`
(per present row: vocab id and the FNV-1a 64 of its normalised text). numpy reads the data with
`np.frombuffer(buf, dtype, offset=12 + meta_length)`; Python's `struct`/`array` with no install.

## 5. The text table and `make`

- **Text.** Per face vocab id: `"<type line>. <oracle>"` with the face's name (and, for a
  double-faced card, the other face's name) replaced by `CARDNAME`, `\n` by a space, whitespace
  collapsed, plus the model's required prefix (none for the bge and MiniLM families). This is
  normalisation version 1; any change bumps it, which makes every row stale.
- **Committed** at `bin/resources/embeddings/oracle_text.rmtb` (~1.5 MB dense).
- **`make` keeps it complete.** After building `robomage`, `make` runs
  `robomage --check-text-table` (std only, no network): every vocab face with oracle text has a
  row whose hash matches its current normalised text, and the header's `model` and
  `norm_version` match `tools/embed/MODEL_PIN`. On success nothing else runs. Otherwise `make`
  runs the generator:
  1. `robomage --text-table-inputs FILE` writes the stale entries (vocab id, normalised text,
     hash);
  2. `python3 tools/embed/fetch_model.py` (stdlib) downloads the pinned model's `config.json`,
     `vocab.txt`, `tokenizer_config.json` and `model.safetensors` into `.cache/embed/<commit>/`;
  3. `bin/<config>/rmembed` (libtorch, built on demand like `make actor`) tokenises, encodes,
     pools and normalises those entries and rewrites the table, keeping every other row's bytes.
  Without libtorch or network the generator fails, naming the stale entries and the command; the
  build never continues with an incomplete table.
- **Encoder.** `rmembed` implements the BERT encoder in the libtorch C++ API, reading the
  safetensors weights directly (no TorchScript export and no Python), and a WordPiece tokenizer
  over `vocab.txt`. It is checked against golden token ids and golden vectors produced once by the
  reference implementation (`tools/embed/make_goldens.py`, needs torch + transformers; the goldens
  are committed, the script is not run by `make`).

## 6. The spec and the network contract

**`FEATURE_SPEC`** is a JSON document `rmfeat --spec` writes from a `SCHEMA`, `CARDS` and
`METAGAME` frame (and that the actor builds in memory): the token-type list; per type its numeric
slots (name, source column, transform) and categorical slots (name, id offset, size); `F`, `C`,
`K`, `Q`; the relation types; the global categorical size; the hashed-text bucket count; the §3.4
overrides; the card tables' widths and column names. Its `spec_hash` (FNV-1a 64 of the document
bytes) is in every exported file and in a model. A JSON document rather than an RMSTATE frame, so
consumers outside C++ read it with any JSON parser.

**TorchScript contract** (the step-5 actor; a policy in any framework follows the same tensors):

```
forward(tok_type [B,N] i64, tok_num [B,N,F] f32, tok_cat [B,N,C] i64, tok_card [B,N,K] i64,
        tok_ability [B,N,Q] i64, tok_mask [B,N] bool, rel [B,R,3] i64, rel_mask [B,R] bool,
        act_tok [B,A] i64, act_mask [B,A] bool) -> (logits [B,A] f32, value [B] f32)
```

The card, text, ability and part tables are the model's frozen buffers, loaded from the RMTB
files when the model is built; the model exports `spec_hash`, `vocab_hash` and `text_model` as
attributes, and the actor refuses a model whose values differ from its own spec and tables.
Logits at padded menu rows are ignored; the actor takes the argmax (or samples) over real rows.

**Recommended model family** (the reference policy, `tools/nn/make_reference_policy.py`): per
token type a linear projection of `tok_num`, plus embeddings of `tok_cat`, plus projections of
the gathered card, text and ability rows, summed into a width-`D` token; an entity transformer
whose attention adds a learned bias per relation type for `(src, dst)` pairs; a pointer head (an
MLP over each `act_tok` token's output) for logits; a value head over the mean of real tokens.

## 7. Tests

- deterministic: the same frames give identical tensors;
- shapes and slot meanings agree with the spec; every `rel` and `act_tok` index is a real token;
  every `tok_card` is `-1` or a card-table row, every `tok_ability` `-1` or an ability row;
- invariant to row order where the format says order is meaningless (a permuted hidden zone);
- the exported tables equal a standard-library Python reader's view of the same files;
- the text table check passes on a clean checkout; regenerating a deleted row leaves every other
  row's bytes unchanged; the tokenizer matches the golden ids and the encoder the golden vectors
  (cosine ≥ 0.9999).
