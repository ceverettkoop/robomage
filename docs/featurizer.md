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
