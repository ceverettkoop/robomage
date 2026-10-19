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
| `METAGAME` | the number of reference lists and archetype labels (categorical sizes in the spec) |
| `STATE` | the snapshot tables (ids 20-48) and knowledge tables (90-99); event tables (80-82) are not featurized (memory is the sidecar's job, `knowledge_store.md`) |

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
`strings`; a table's rows are its tokens in frame order. Single-row tables (`header`, `turn`,
`pending`, `pending_cost`, `match`, `k_meta`) are one token each, so global state is a few tokens
rather than a special vector. Child tables
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
| `enum:<e>` | one categorical slot: the id range of `e` (its `n` values take `n + 1` ids; the first is the cell's none: all ones or out of range) |
| `player-rel` | one categorical slot (none, self, opponent); the absolute `enum:seat` beside it is dropped, so a model is seat-symmetric |
| `card` | one `tok_card` slot |
| `handle`, `row:<t>` | a relation (§3.3), no feature |
| `text` | dropped, except the columns §3.4 hashes |

A `card` or `index` column also gets a numeric none flag (1 at all ones); an enum's none is its
slot's first id, and a `row:` column's is the missing relation. The global categorical id space is
the concatenation of every (token type, categorical column) range, with id 0 the pad; one embedding
table covers it. A composite is recognised by its column names: a seat pair `<n>` / `<n>_rel`, a
library position `<n>_anchor` / `<n>_pos` / `<n>_span` (each part by its tag) and an ability
reference `ab_card`, `ab_index`, `ab_key` (§3.3).

### 3.3 Relations and ability references

- **Identity.** A table whose first column is a `handle` named `handle` (`object`, `stack`,
  `reveal`) owns that handle; the lowest table id wins, so `object` wins over `stack` (a stack
  object is an `object` row too).
- **Handle columns** (`obj`, `source`, `target`, `attached_to`, `attacker`, `blocker`, `for`, …)
  give a relation from the row's token to the token owning that handle, typed by
  `(table, column)`; so does an owning table's own `handle` column when another table wins the
  handle (a `stack` row relates to its `object` row). A handle no token owns gives no relation (it
  names an object the viewer cannot identify).
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
| `object_counter.kind`, `player_counter.kind`, `object_keyword.text`, `player_effect.filter`, the `ab_key` of an abref | categorical: 1 + FNV-1a 64 of the text modulo 256 buckets (the empty text is the slot's none) | counter kinds and keyword parameters have no enum |
| `menu.description` | dropped | free text, covered by the structured columns |
| `k_template.list`, `b_list.list`, `k_archetype.archetype`, `b_archetype.archetype` | categorical over the `METAGAME` frame's `meta_list` / `meta_archetype` rows, as a `row:` into `METAGAME` | tagged `index`, but each is a row of a `METAGAME` table, not a position |

### 3.5 Learner-side: the omniscient featurization and targets

- **Omniscient featurization.** The same spec featurizes the omniscient viewer's frame (as
  `--rmstate-dump` writes it at every decision): every object is identified, hidden zones are
  tokens with their cards, and the `k_*` token types have no rows (the omniscient viewer has no
  knowledge tables). Seat-relative columns read as the omniscient frame defines them (self = A),
  so a learner pairs it with the deciding seat explicitly. For a critic that sees the whole game.
- **Targets** (`rmfeat --targets`, `src/featurize/targets.h`), per decision of a seat, from the
  omniscient frame of the same decision: the opponent's hand as a multiset of card ids (ascending);
  the opponent's registered list and archetype (`METAGAME` row indices, -1 none): the
  classification of `knowledge_store.md` §10.4 of the omniscient `DECKLISTS` frame's list (the
  reference list of the same deck name, else the one with the most cards in common), the row the
  opponent's own frames mark `registered` in `k_template` (the gate checks they agree); the seat's
  own library, top first (`object.pos`), as card ids up to the spec's `library_depth` (10); the
  opponent's library likewise. An RMTB `targets` file: per decision `seat`, `game`, `decision`,
  `sideboarding`, `opp_list`, `opp_archetype`, and `opp_hand`, `own_library`, `opp_library` each
  as values plus `_offsets`, for auxiliary heads.
- **Isolation.** Neither reaches a decision: the in-process feed carries seat frames only and the
  actor entry point refuses other viewers. The engine's `--selfcheck-featurize`
  (`src/state_format/selfcheck_featurize.h`) featurizes each seat's frame at every decision with
  the actor entry point and requires the same tensors from the scramble check's frame (hidden
  identities rotated) and the permutation check's (hidden zones permuted), and the omniscient frame
  refused; the gate (5d) adds that an `nn:` seat's tensors equal `rmfeat`'s featurization of that
  seat's own stream.

## 4. The card tables

Three whole-vocab tables, the same for every match, indexed by vocab id (`src/card_vocab.h`; ids
`0 … N_CARD_TYPES-1`, the token band from 900, the token sentinel 1023). They are built from a
**whole-vocab `CARDS` frame** (the engine's `--cards-frame FILE`: `SCHEMA`, `CARDS` and `METAGAME`
records, the `--closed-decklist` card set) by `rmfeat --export-tables` (`src/featurize/card_tables.h`),
by the same tag rules. The featurizer reads `N_CARD_TYPES + 1` from the text table's row count,
never from an engine header.

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
`<table>.<col>=<VALUE>` (a child count), in that order. Row `N_CARD_TYPES` is the none row (all
zeros): a `-1` card slot reads it. The file also holds `back_face` (i64, a row per card id): a
physical card's back face's vocab id (itself when it has none, -1 for a back face with no id of its
own), which gives `object.card`'s current-face slot (§3.4).

### 4.2 Text table `[N_CARD_TYPES + 1, 384]` f32

The oracle-text vectors (plan 5c), one row per face vocab id with oracle text, zero rows
elsewhere and at the none row. Committed (§5).

### 4.3 Ability table `[Nab, Fa]` f32 and part table `[Npart, Fp]` f32

- **Ability rows**: one per `card_ability` row of every vocab card, in `(card, index)` order,
  then one per `ab_key` of the spec's fixed key list (abilities with no card row: granted or
  engine-built). Columns: the row's by the tag rules (`face`, `ability_kind`, `category`, `timing`
  one-hot; `tap`, `limit`; the structured parameters, `state_format.md` register 41: amount and
  its kind, power/toughness modifiers, target classes, target bounds; `n_parts`, `charm_num`) and
  `keyed` (1 for a key row).
- **The fixed key list**: every distinct non-empty `card_ability.key` of the frame (keyword-built
  abilities: `Equip`, `Cycling`, `Reconfigure:unattach`), plus the key forms Appendix A.1 gives
  engine-built definitions that the `SCHEMA` frame enumerates: `keyword:<k>` for each `keyword`
  value (keyword-derived triggers), `subtype_mana:<c>` for each `color` value (a land type's mana
  ability) and `LandType`; sorted. A key row that is some card row's key has the features of the
  first such row; the others only `keyed`. Keys an abref can carry outside the list (a granted
  trigger's `granted_trigger:<script>`, `enchant:<filter>`, the engine's delayed-trigger keys such
  as `return_from_exile:<n>`) are built during play from text no frame enumerates; they resolve to
  `-1`, and the hashed `ab_key` categorical (§3.4) still tells them apart.
- **Part rows**: one per `card_ability_part` row (sub-abilities and modes), in `(card, ability,
  part)` order, with the same parameter columns, the part kind, depth and parent, and `ability`:
  the ability row it belongs to (an i64 array of the file, not a feature).
- **The ability index** is in the ability file: `ab_card` and `ab_index` (i64, per row its card
  and index; -1 for a key row) and `key_rows` (i64, per key its row; the array's `columns` are the
  keys). The featurizer resolves `tok_ability` with it: the row of `(ab_card, ab_index)`, else of
  `ab_key`, else `-1`.

### 4.4 Files: RMTB

Every exported table is an RMTB file of one or more **named arrays**, readable with any language's
standard library (the trajectory files of `docs/rnad_training_plan.md` reuse it):

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
pinned commit, `<repo>@<commit>`), `pooling` (`cls`, `mean`), `norm_version` (the text
normalisation, §5) and `rows_meta` (per present row, ascending: `{"id": <vocab id>, "hash": <FNV-1a
64 of its normalised text, 16 hex digits>}`); then `arrays`: one
`{"name", "dtype", "shape": [...], "offset"[, "columns": [...]]}` per array in file order (`shape`
of any rank, `[]` a scalar; `offset` from the start of the data, the previous array's end rounded
up to a multiple of 8; `columns`, optional, names the last dimension's entries). The same content
always encodes to the same bytes.

Dtypes: `f32`, `f16` (IEEE binary16, written with round-to-nearest-even), `i64`, `i32`, `u16`,
`u8`. Ragged data is two arrays, the values and an `i64` offsets array (`n + 1` entries; row `r`
is `values[offsets[r]:offsets[r+1]]`), with no padding on disk. The text table is one array,
`vectors` `[N_CARD_TYPES + 1, hidden]` f32, with `rows_meta` in the meta.

Readers: `src/featurize/rmtb.{h,cpp}` (C++, std only; also the writer and the f16 conversions),
`python/rmtb.py` (Python's `struct`, f16 as format `e`); numpy reads an array with
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

**`FEATURE_SPEC`** is a JSON document `rmfeat --spec` writes (`src/featurize/feature_spec.h`) from
the whole-vocab cards file's `SCHEMA` and `METAGAME` frames and the card tables its `CARDS` frame
gives (and that the actor builds in memory from the same inputs): the schema and vocab hashes; the
hashed-text bucket count; the targets' library depth; the `METAGAME` row counts it sizes slots by;
`F`, `C`, `K`, `Q`; the global categorical size; the menu's token type; the token-type list, per
type its numeric slots (name, source column, transform), categorical slots (name, source, id
offset, size), card slots, ability slots and relation types; the relation types `(table, column,
target)`; the §3.4 overrides; the card, text, ability and part tables' rows, widths and column
names; the fixed key list. One line, its keys in a fixed order. Its `spec_hash` (FNV-1a 64 of the
document bytes) is in every exported file and in a model. A JSON document rather than an RMSTATE
frame, so consumers outside C++ read it with any JSON parser.

**Network contract** (the step-5 actor; a policy in any framework follows the same tensors):

```
forward(tok_type [B,N] i64, tok_num [B,N,F] f32, tok_cat [B,N,C] i64, tok_card [B,N,K] i64,
        tok_ability [B,N,Q] i64, tok_mask [B,N] bool, rel [B,R,3] i64, rel_mask [B,R] bool,
        act_tok [B,A] i64, act_mask [B,A] bool) -> (logits [B,A] f32, value [B] f32)
```

The card, text, ability and part tables are the model's frozen buffers, loaded from the RMTB
files when the model is built; the model exports `spec_hash`, `vocab_hash` and `text_model` as
attributes, and the actor refuses a model whose values differ from its own spec and tables.
Logits at padded menu rows are ignored; the actor takes the argmax (or samples) over real rows.

**Padding** (`featurize::pad_batch`, `src/featurize/batch.h`, std only): a batch is padded to its
largest `N`, `R` and `A`, each rounded up to a multiple of 8 and at least 8 (so no dimension is 0 or
1, which `torch.export` would specialise); pads are 0 (`tok_card`, `tok_ability` -1) and every mask
0; `rel_mask` marks the real relations. The buffers are reused from batch to batch.

**Model files and their metadata.** A model carries four strings: `spec_hash`, `vocab_hash` (16 hex
digits each), `text_model` (the text table's `model`) and `policy_version` (an unsigned integer,
`"0"` when absent), each in its export's own metadata, never in a sidecar file: a TorchScript
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
over the menu rows) and the policy's version. A failure is `false` with a message, never a default
action. Implementations: in-process (`load_model_evaluator`: the actor's model, CPU, one thread,
TorchScript through `torch::jit::load` or AOTInductor through `AOTIModelPackageLoader`; one
policy) and remote (`nn:remote:<endpoint>`, 5d part 2), over the protocol of
`docs/inference_protocol.md`: decisions in shared memory, a Unix socket for signalling, the spec
hash on every request; a request may name several policies (the training plan's current and two
regularisation policies) and gets `log π` per menu row for each, with each policy's version, as
more entries of the same `Evaluation`. A server that goes away fails the actor loudly. `rmserve`
(C++, libtorch, CPU) is the reference server; the training plan's GPU server speaks the same
protocol.

**The `nn:` agent** (`src/actor/nn_agent.h`). `nn:<model>[:<key>=<value>…]`, keys `temp=<t>`
(sampling temperature, default 1; drawn with the agents' shared generator, so a game replays from
its seed) and `argmax` (the largest logit; rows within 1e-5, absolute plus relative, of it tie and
the first is taken, so float drift between two exports of one model cannot reorder rows equal in
exact arithmetic, such as two identical cards in hand), plus two for tests and the bench:
`dump=<file>` (every decision it featurized, written at exit as one `tensors` file, the bytes
`rmfeat --featurize` writes for that seat) and `stats` (an `NN-STATS:` line at exit); an unknown key
fails loudly. The training plan adds `thresh=` and `disc=`. It keeps the exact distribution it
sampled from over every menu row (after temperature and any projection; one-hot under `argmax`)
and the policy version for its last decision (`Agent::last_behaviour`, `agents/agent.h`), for the
training plan's trajectory writer. The featurizer's inputs are built once, before the match, in
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

## 7. Tests

The gate's `featurize` tier (`docs/ci.md`) runs those of this list that need no model, over the
replayed corpus's dumps and explore games' (the text table's are the `vocab` tier's; the `nn:`,
export and `rmserve` ones come with 5d):

- deterministic: the same frames give identical tensors;
- shapes and slot meanings agree with the spec; every `rel` and `act_tok` index is a real token;
  every `tok_card` is `-1` or a card-table row, every `tok_ability` `-1` or an ability row;
- invariant to row order where the format says order is meaningless: the engine's permutation
  self-check (a permuted hidden zone) gives the same frame bytes and so, under
  `--selfcheck-featurize`, the same tensors; and every token table but `menu` (whose row is the
  reply) with its rows shuffled and every `row:` reference into it renumbered gives the same
  tensors up to the matching token permutation;
- the exported tables equal a standard-library Python reader's view of the same files;
- the text table check passes on a clean checkout; regenerating a deleted row leaves every other
  row's bytes unchanged; the tokenizer matches the golden ids and the encoder the golden vectors
  (cosine ≥ 0.9999);
- exact re-featurization: `rmfeat` over a `--replay` of an `nn:` game equals what `nn:` featurized;
- isolation (§3.5): an `nn:` seat's tensors equal `rmfeat`'s for that seat's stream, unchanged by
  the scramble self-check; the actor entry point refuses an omniscient or other-seat frame;
- the reference policy's TorchScript, `.pt2` and ONNX exports give the same logits and values;
  in-process and `rmserve` evaluation agree.

The gate's `actor` tier (`docs/ci.md`) runs the `nn:`, re-featurization and export checks when the
actor binary and a Python with torch are available; `rmserve` is 5d part 2.
