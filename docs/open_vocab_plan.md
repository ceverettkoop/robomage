# Plan: an open vocabulary, no archetypes (vocab-safe network, deck onboarding)

**Goal.** A network trained with one card vocab keeps working — same checkpoints, same outputs on
the cards it knows — when cards are appended to the vocab later; it judges an opponent by the
composition of their decklist, with no developer-chosen archetype categories anywhere; and it can
play and face decks it has never seen, through the cards' shared characteristics. New meta decks
reach the vocab through one repeatable onboarding pipeline.

**Fixed decisions** (do not revisit while executing):

- **Open decklist** stays (`docs/state_format.md` decision 11): each seat sees the opponent's
  registered list. `--closed-decklist` remains a legal engine mode but is not a training target.
- **`bin/resources/decks/archetypes.json` is deleted**, and with it every archetype label and the
  reference metagame built from it (the `METAGAME` frame, `k_template` / `k_archetype`, the belief
  level `b_list` / `b_archetype`). The network sees the opponent's list card by card (§1.3) and
  forms its own notion of "what kind of deck"; nothing in the repo names archetypes.
- **The scripted agent changes minimally.** It is used only for the initial warm start (B0–B2) and
  as the benchmark opponent; after the initial phase ~90% of training is self-play. New decks get
  no deck-specific scripted logic and no behaviour cloning: they enter training through self-play
  from the current checkpoint.
- **Card indices never change** and real cards stay below the token band (ids 0–899; tokens
  900–1022; 1023 the token sentinel; `N_CARD_TYPES` = 1024, `src/card_vocab.h`).

## 0. Executing this plan (read first)

This plan is written to be executed in a session with no context beyond the repo. Before starting:

1. **Starting state.** Written at commit `6545ca7c` on branch `claude/rnad-training`, with two
   intentional uncommitted changes: this file, and `learn/rmlearn/orchestrator/gpu.py`
   (`rnad_init` passes the run's `model.omniscient_critic` / `critic_layers` to
   `RNaDLearner.from_bc_checkpoint`, so a `run.start_from` run honours them). Commit both before
   branching if they are still uncommitted: `git log`, `git status`.
2. **No training run may be active while engine binaries, the vocab or the state format change.**
   A running `rmtrain` run's actors load `bin/release/robomage_nn` and the run's `tables/` on every
   restart. Check: `systemctl --user list-units 'rmtrain*' 'rmpilot*' 'rmbench*' --all` and
   `learn/rmtrain status <run>` for each directory under `runs/`. Wait for, or ask the user before
   stopping, any active unit. Build and test only with `make` (debug) until no run is active.
3. **Experiments that precede this plan** (results under `runs/`, read them, do not rerun):
   `league24h-sb` (the 24 h R-NaD run; its fixed points `ckpt/fixed/m1.pt` … `m11.pt`, `bc-final.pt`,
   `rnad-final.pt`; benchmark `bench/end/summary.md`), `pilotA` (warm start from `m11`, lower
   `reg_eps`, fair-pair weights) and `pilotB` (pilotA + the omniscient critic, seeded from
   `runs/pilotB-prep/m11-critic.pt`); their benchmarks are `runs/pilot{A,B}/bench/end/summary.md`,
   their chain log `runs/pilot-chain.log`. Every checkpoint under `runs/*/ckpt/` (fixed points, `*-final.pt`,
   `bc-*.pt`) plus `runs/pilotB-prep/m11-critic.pt` is migrated in S5 — none is to be lost.
4. **Workflow** (CLAUDE.md): a branch `refactor/open-vocab` off the starting branch; one stage at a
   time, each by one subagent told not to spawn agents; every stage ends with `make check` green
   and the stage's own acceptance checks; one commit per stage (or per coherent sub-step). Long
   commands run in the background with their output path stated. `DO NOT MODIFY CARD SCRIPTS`; no
   retagging of script parameters.
5. **Vocabulary of this plan.** *Layout*: the position of every input slot, table column and
   embedding row the network's parameters are indexed by. *Vocab*: the (id → card name) entries of
   `src/card_vocab.h`. *Tables*: the run's exported card / text / ability / part tables
   (`rmfeat --export-tables`, `docs/featurizer.md` §4). *Export*: a TorchScript / `.pt2` file the
   actors and `rmserve` load. *Checkpoint*: a `model.save_checkpoint` file or an R-NaD learner
   state.

## 1. Background (measured at `6545ca7c`)

### 1.1 What already survives a vocab expansion

- **Card tables are data.** `CardRows` (`learn/rmlearn/model.py`) loads `card_table`, `text_table`,
  `ability_table`, `part_table` as `persistent=False` buffers; they are not in a `state_dict`.
- **The id embedding is sized by the index space.** `cards.card_id` is
  `nn.Embedding(N_CARD_TYPES + 1, d)` (1025 rows); appending a card changes no shape.
- **No feature is normalised over the vocab.** Card, ability and part columns use fixed per-value
  transforms (`src/featurize/columns.cpp`: bool, `log1p`, signed `log1p`, one-hot, counts), so a
  new card never rescales a known card's row.
- **The aux hand / library heads are size-agnostic** (dot products against `card_rows()`).
- **Unseen cards have a trained fallback:** the oracle text is encoded by the pinned frozen model
  (`tools/embed/MODEL_PIN`) into `text_table`, `text_proj` is shared, and `id_dropout` (0.1) trains
  the `unknown_id` path (a card's features plus `unknown_id` in place of its id row).
- **`card_subtype` is closed:** its 518 columns are the schema's whole subtype enum.

### 1.2 What breaks

| # | Where | Why it breaks on growth | Severity |
|---|---|---|---|
| B1 | Hash equality: `load_checkpoint` / `rnad_learner._load_with_config` (`spec_hash`, `vocab_hash`, `text_model`), `learn/rmlearn/export.py` `META_KEYS`, `src/actor/evaluator.cpp`, `tools/rmserve/server.cpp` (policy load and client handshake), `learn/rmlearn/server.py` | `vocab_hash` (over every (id, name), `src/featurize/oracle_text.h`) changes with any card; `spec_hash` covers the tables' row counts, widths, column names and the fixed key list (`docs/featurizer.md` §6). Every consumer refuses. | refusal (safe, total) |
| B2 | Exports bake the card rows: `InferencePolicy` folds `cards.token_table()` into a buffer | An export knows only the vocab it was built with; a new id reads its old placeholder row. | silent if B1 is relaxed alone |
| B3 | Categorical layout: `add_categorical` (`src/featurize/feature_spec.cpp`) gives each slot `offset = categorical_size` and adds its size — one running offset over every slot (3468 ids) | An added enum value anywhere shifts every later slot's ids: `cat_emb` rows are reinterpreted. | **silent corruption** |
| B4 | Numeric layout: `TokenEmbed` concatenates every type's numeric slots (`num_proj` input `G` = 407 columns); a MASK-tagged enum column has one slot per value; token types sit in schema order (`type_emb`, 35 rows) | An added enum value or schema table shifts later columns / rows and changes `num_proj`'s shape. | shape change or silent shift |
| B5 | Table columns: `card_keyword.kw=<k>` over `parser_keyword_names()` (50), ability / part `category` over the effect kinds (73 + none = 74), `ability_kind`, `timing`, `pip_kind`, `layout`, … | A new keyword or effect category (the usual cost of a new mechanic) adds a column; `card_proj` / `ability_proj` / `part_proj` change shape. | shape change |
| B6 | Metagame-sized parts: the `lists` pivot (`k_template` + `k_archetype`, 55 slots), `b_lists` (47), `row:meta_list` / `row:meta_archetype` categoricals, `aux_list [13]`, `aux_archetype [8]`, targets `opp_list` / `opp_archetype` | A new reference deck or label changes `G` at once and shifts later types. | shape change — removed by S1 |
| B7 | Never-trained id rows | `card_id` rows are `N(0, 0.1)` at init and get no gradient until their card is seen; a new card reads a random vector (~1.4 in norm), not `unknown_id`. | silent quality loss |
| B8 | **Ability row ids are vocab positions.** A token's ability slot holds a row of the ability table (`CardTables::ability_row(card, index)`, `src/featurize/card_tables.h`); rows are ordered `(card id, index)` then the fixed key rows. Episodes store `tok_ability` as those raw rows (`learn/rmlearn/episodes.py` `INPUTS`). | A card appended at id < 900 shifts every token card's rows and every key row. Already-recorded episodes (a replay window, a corpus) and exports then point at the wrong abilities, silently. | **silent corruption** of stored data |

### 1.3 The opponent's composition is already an input

Under open decklist the featurizer's **card ledger** (type `ledger`, `docs/featurizer.md` §3.6) has
one token per (player, card) of each player's universe — for the opponent, every card of their
registered list — with the card as a `tok_card` slot and numeric slots for seen counts, known
hand, and hand / library / sideboard min-max bounds. The network therefore already sees the
opponent's 75 card by card, featurized through the shared card rows (types, costs, keywords,
oracle-text vector). The archetype inputs (`lists`, `b_lists`) only summarise it with
developer-chosen labels. **No new contract input is needed** to "account for the composition of the
opponent's decklist"; removing the labels (S1) and making the vocab safe (S2–S5) is the work. An
optional in-model deck summary is S6.

## 2. Stages

Each stage lists what to change, then **acceptance** (all must hold before its commit).

### S0. Baseline fixtures (½ day)

- Record the reference outputs the later stages must reproduce: pick a fixed decision set (the
  replay corpus's dumps, `tests/corpus/*.rmlog` replayed with `--rmstate-dump`, plus 200 decisions
  from recent self-play episodes of the best run) and, for each checkpoint listed in §0.3, its logits
  and values on that set under the current tables (fp32, CPU). Store them under
  `tests/open_vocab/baseline/` with a manifest (checkpoint path, decision source, tables'
  `spec_hash`).
- Note the current best policy from the benchmark summaries (`runs/*/bench/end/summary.md`): it is
  the policy S5's verification continues from.
- **Acceptance:** the fixture files reproduce bit-for-bit when regenerated.

### S1. Remove archetypes and the reference metagame (3–4 days)

Delete the concept; nothing replaces it but the ledger (§1.3).

- **Data:** delete `bin/resources/decks/archetypes.json`; delete `src/state_format/metagame.{h,cpp}`;
  the `METAGAME` frame and its tables (`meta_list`, `meta_list_card`, `meta_archetype`) leave the
  schema (`src/state_format/schema.cpp`, `encode.cpp`, `stream.cpp`, `dump.cpp`, `cards_file.cpp`:
  `--cards-frame` writes SCHEMA + CARDS only; CARDS no longer adds "every metagame card", it carries
  the vocab as before). Bump the RMSTATE format version.
- **Knowledge sidecar:** delete the archetype and reference-list layers — `src/knowledge/templates.*`,
  `archetype.cpp`, `belief.*`, the `k_template` / `k_archetype` / `b_list` / `b_archetype` rows
  (`k_rows.*`, `output.cpp`, `knowledge.cpp`, `frame_view.*`, `truth.*`), the knowledge level
  `expert+belief` (`--knowledge-a/-b` accepts `off|expert`; `expert+belief` becomes a removed-flag
  error in `src/cli_args.cpp`). `tools/rmknow`: drop `--belief`, `--hide-lists`,
  `--belief-exactness`, `generative.*` and the template / belief parts of `selftest`, `exactness`.
  The sound `k_*` facts about pools, positions, seen cards stay unchanged.
- **Featurizer:** drop the `lists` / `b_lists` pivots and the `row:meta_*` overrides
  (`src/featurize/feature_spec.cpp`, `folds.*`); `spec.json` loses `metagame`. Targets: drop
  `opp_list` / `opp_archetype` (`targets.*`, `episode.*`; episode version bump).
- **Learner:** drop `aux_list` / `aux_archetype` (`model.py` `RMNet`, `aux_losses`), their BC
  metrics (`bc.py`), `batch.py` / `episodes.py` / `python/rmtraj.py` fields; `tools/nn/make_reference_policy.py`.
- **Scripted agent (minimal):** `python/rm_agent.py` stops reading `METAGAME` (`meta_lists`,
  `_opp_possible_lists`); `opp_deck` / `opp_runs_card` use the registered list (always present under
  open decklist) and return an empty deck under `--closed-decklist`. `python/teacher.py` and
  `src/agents/teacher.h` docstrings drop METAGAME. Nothing else: the scripted agent keys
  deck-specific behaviour on its **own deck's stem** (`_deck_named`, `python/scripted_core.py`) and
  card heuristics, never on labels; the sideboard guides key on deck stems
  (`python/sideboard_guides.py`; an unknown deck has no guide target, so `hard+sb` keeps the
  maindeck).
- **CLI:** the knowledge panel loses the archetype line (`src/cli/board.cpp`).
- **Tests and gate:** delete `tests/scenarios/knowledge_archetype_open_registered.json`; the
  `knowledge` tier loses archetype inference and belief calibration
  (`tools/robomage_check/knowledge_meta.*`, `tier_knowledge.cpp`, `knowledge_lines.*`); re-record
  the `cli` goldens (`robomage_check --tier cli --record`; the diffs must be only the removed
  archetype lines); `python/pycheck.py` drops its `meta_list` checks. The replay corpus's decision
  logs are unaffected (no decision changes); re-record `.replay` baselines only if their text
  changes, and then explain the diff in the commit.
- **Docs:** `docs/state_format.md` (§4, A.4, A.6), `docs/knowledge_store.md` (§10 removed; keep a
  one-paragraph history note), `docs/featurizer.md`, `docs/trajectory_format.md`, `CLAUDE.md`
  (the "no codegen … archetypes.json" paragraph, the metagame / belief key-file entries, the
  `knowledge` tier row), `tools/meta_scraper/README.md` (it writes deck files only; "archetype"
  there is MTGTop8's deck-name grouping, which names the output file).
- **Acceptance:** `grep -rniE "archetype|metagame|meta_list|b_list|belief" src python tools learn
  tests docs CLAUDE.md` finds only the meta-scraper's MTGTop8 terminology and history notes;
  `make check` green; the knowledge tier's soundness checks (`--selfcheck-knowledge`) still pass on
  the fuzz games; agent games (`python3 python/rm_driver.py --player-a scripted --player-b explore
  --games 20`) and a `scripted:hard+sb` bo3 smoke behave as before (same results on the same
  seeds: the scripted agent's decisions under open decklist must be byte-identical — compare
  `--rmlog-dir` logs before and after).

S1 changes the network layout (the pivots go); existing checkpoints are carried across by S5's
migration, which runs **once, after S2–S4**, so there is one cut-over, not two.

### S2. A stable, append-only layout (4–5 days)

Principle: **the layout is the contract, the vocab is data.** A committed file,
`src/featurize/registry.json` (append-only, machine-checked), gives every layout-defining value a
permanent position with reserved capacity:

- **Enums** read by any MASK / ENUM / child-count column (keywords, effect categories, card types,
  colours, pip kinds, action categories, zones, …): value name → permanent index, plus a capacity
  (open enums: about 2× today's size, e.g. keywords 128, effect categories 256, action categories
  128; closed enums such as seats or faces: their size). Matched by name, so an engine enum may be
  reordered without moving a column; a removed value leaves a hole that reads 0.
- **Token types:** name → permanent index (capacity 64; 33 after S1).
- **Numeric slots per type:** name → permanent column within the type, per-type capacity (today's
  count rounded up to a multiple of 8, plus headroom); `G` becomes the sum of capacities.
- **Categorical slots:** name → fixed offset and capacity; no running offset. Hashed slots
  (`hash_buckets`) keep their bucket count.
- **Card / ability / part table columns:** column name → permanent column with the enums'
  capacities (`card_subtype` stays 518, closed).
- **Ability rows (fixes B8):** a stable row per (card id, ability index): row =
  `card_id × A_MAX + index`, then the fixed key rows in a band at the end (key name → permanent row,
  capacity ~2× today's 73). Measure the vocab's most abilities on one face, set `A_MAX` to at least
  twice that; a card exceeding it is a deliberate layout bump. The table is data (no parameters), so
  its ~1025 × A_MAX rows cost memory, not weights; `token_table()` folds it in exports as today.
  Part rows keep their order (they are pooled into ability rows inside the tables, never referenced
  by tokens).

`build_spec` (`src/featurize/feature_spec.cpp`) and `build_card_tables` (`card_tables.cpp`) place
every slot, column and row at its registry position, zero-fill reserved positions, and **fail** on a
value with no entry. `make registry` appends new entries (never moves one); the gate's `vocab` tier
checks the file is current and that no entry moved since the last commit (diff against
`git show HEAD:src/featurize/registry.json`). The spec gains `layout_hash` (FNV-1a over positions
and capacities only) beside `spec_hash` (exact identity, kept for logs and episodes).

- **Acceptance:** `make check` green; `rmfeat --spec` shows the registry layout; a synthetic
  test (a copy of the vocab with one card appended at the next free id and one fake keyword
  appended to the registry) produces the same `layout_hash`, the same token tensors for every
  decision of the corpus dumps that does not mention the new card or keyword, and the same ability
  row numbers for every existing (card, index).

### S3. Compatibility instead of identity; exports per vocab (2 days)

- **Model metadata** (checkpoint `hashes`, export strings, `InferencePolicy` attributes, the
  inference protocol handshake, `docs/inference_protocol.md`) carries `layout_hash`, `text_model` and
  a **vocab manifest**: the (id, name) pairs of the tables the model was trained with.
- **One check**, implemented once in Python (`learn/rmlearn/spec.py`) and once in C++
  (`src/featurize/`), used by `load_checkpoint`, `_load_with_config`, `export.py`,
  `src/actor/evaluator.cpp`, `tools/rmserve/server.cpp`, `learn/rmlearn/server.py`: equal
  `layout_hash` and `text_model`; every manifest pair present unchanged in the current vocab. New
  ids pass; a renamed or removed id is refused naming it.
- **Exports are rebuilt per vocab** (B2): an export's manifest must equal the current vocab
  exactly (the evaluator and `rmserve` refuse otherwise); checkpoints are the portable artefact.
  `rmtrain` re-exports the live policy, the ladder's fixed points and the regularisation policies
  from their checkpoints when it starts on tables whose manifest differs from the run's last.
  `learn/rmlearn/benchmark.py`'s `export_missing` gains the same rule (rebuild an export whose
  manifest is not current).
- **`rmtrain retable <run>`** (new): for a stopped run after a vocab expansion — export the new
  tables into `<run>/tables/` (keeping the old ones as `tables.<spec_hash>/`), check every
  checkpoint of the run against them, rebuild the exports, and record the change in `state.json`
  and `metrics.jsonl`. The replay window survives (same layout; stored episodes' card and ability
  ids are stable by S2); `ReplayWindow` accepts episodes whose layout matches and whose manifest is
  a subset of the current one.
- **Table drift report:** `retable` (and the gate's `vocab` tier) also diffs the old and new tables'
  rows for **known** ids: an engine change that alters a known card's parsed features passes the
  compatibility check but changes that card's inputs — the report lists such cards so the change is
  intended, not accidental.
- **Acceptance:** the synthetic expansion of S2 loads every migrated checkpoint (after S5) and the
  pre-expansion exports are refused by the evaluator and `rmserve` with a "rebuild the export"
  message; `rmtrain retable` on a copy of a small finished run succeeds and the run resumes.

### S4. Unseen cards read `unknown_id` (½ day)

- `CardRows` gains a persistent buffer `card_seen [N_CARD_TYPES + 1]` (bool), set for every id in a
  training batch's `tok_card` (an `index_fill_` per batch), saved in the checkpoint.
- `token_table()` uses `feat + (card_seen ? card_id : unknown_id)` per row; exports fold the same
  rule, so an unseen card plays exactly the path id-dropout trains. New id rows start at zero (a
  residual on `unknown_id` once seen).
- Training lever (config, not contract): raise `model.id_dropout` from 0.1 to 0.2–0.25 for runs
  meant to generalise; decided by the held-out benchmark (S8), not assumed.
- **Acceptance:** a unit check that a never-seen id's export row equals its features plus
  `unknown_id`, and that marking it seen changes only that row.

### S5. Migration, acceptance test and the one-time cut-over (2–3 days)

- **`python -m rmlearn.migrate OLD.pt --tables NEW_TABLES --out NEW.pt`** for `save_checkpoint` files
  and R-NaD learner states (`model`, `target`, both `regs`, Adam `m` / `v`): build the new-layout
  model, copy each old parameter slot to its new position **by name** (`cat_emb` rows by slot and
  value name, `num_proj` columns by type and slot name, `type_emb` rows by type name, table
  projection columns by column name), drop slots S1 removed (`lists`, `b_lists` columns, the two aux
  heads), zero reserved positions, set `card_seen` from the ids whose `card_id` row differs from its
  initialisation (or from the run's episodes when available), zero never-seen rows. The migration
  needs the old tables (each run's `tables/`) and the old spec to read names from.
- **Acceptance test** (a new check in the gate's `featurize` tier; `actor` when opt-in): (a) S0's
  fixtures: each migrated checkpoint on the new layout gives the fixture outputs, except for the
  effect of S1's removed inputs — so the comparison is against the old checkpoint run with the
  `lists` / `b_lists` slots zeroed (generate that reference in S0 too); logits and values within
  1e-5 (fp32, CPU); (b) the synthetic one-card expansion: identical outputs on every decision not
  mentioning the new card.
- **Cut-over:** migrate every checkpoint listed in §0.3 into `runs/<run>/ckpt-v2/` (originals kept);
  regenerate exports; then a verification run: `rmtrain start` from the best migrated checkpoint
  with its original config (`run.start_from`), 2–3 h of R-NaD, benchmarked (`rmtrain bench`) paired
  against the pre-cut-over export of the same checkpoint: the overall Δ and every deck's Δ must be
  within two SEs of 0 at step 0 (bench the migrated export itself) and not negative beyond two SEs
  after the continuation.
- Old episodes (the replay windows, B0/DAgger corpora) are of the old layout and are not migrated;
  a resumed or new run starts with an empty window (self-play refills it in minutes). Keep the
  old corpora on disk until no experiment needs them.

### S6. Optional: an in-model deck summary (1–2 days, measured)

The ledger already carries the composition (§1.3); attention can pool it. If the held-out benchmark
(S8) shows the network does not use the opponent's list well, add per player one summary token
computed **inside the model** from that player's ledger tokens (a learned attention pool over the
ledger tokens' input embeddings, keyed by `player_rel`), prepended to the sequence before the
encoder. No contract input changes, no protocol change; the exports include it (plain torch). Its
output projection starts at zero, so a migrated model's outputs are unchanged until it trains; it
gets two `type_emb` rows from the registry's reserved capacity. Ship only if a paired benchmark on
held-out decks improves.

### S7. The deck onboarding pipeline (3–4 days for the tooling; engine work per mechanic)

One command, `python3 tools/onboard/onboard.py <stage|all> [--deck STEM]`, stdlib only, idempotent,
with a status file per deck `bin/resources/decks/candidates/<stem>.status.json` (stage → result,
timestamp, input hashes, so a stage reruns only when its inputs changed) and `onboard.py report`
rendering every status file as one table (deck, meta share, missing cards, missing mechanics,
stage reached). Deck directories replace any roster file: `decks/candidates/` (scraped, not yet
playable), `decks/league/` (training decks; `rmtrain`'s `league.decks` names stems here),
`decks/heldout/` (playable, never trained on: the generalisation test). A deck moves between them
by `git mv` in stage 10.

| # | Stage | What | Tool | Where |
|---|---|---|---|---|
| 1 | scrape | top-N Legacy decks → `.dk` in `decks/candidates/` | `tools/meta_scraper/scrape_meta.py --out bin/resources/decks/candidates` (network, polite delay) | manual |
| 2 | diff | each card against `src/card_vocab.h` (accent-insensitive, DFC-aware): in vocab / script present, not in vocab / no script | `robomage --check-decks --check-decks-dir candidates` | CI-able |
| 3 | provision | fetch missing scripts and the tokens they make (engine-synthesised kinds: add stems to `tools/forge_fetch/provision_decks.py` `keyword_tokens`) at the pinned Forge commit | `tools/forge_fetch/fetch_script.py` | manual (network) |
| 4 | parse report | load each new script with parser findings captured: unknown ability categories (would resolve as no-ops), unmodelled params, unknown keywords, SVars | new engine mode `--parse-report CARD…` (JSON per card; plays no game) | CI-able |
| 5 | capability | per card clean / needs mechanic X; per deck missing cards and mechanics; mechanics shared across candidates ranked by how many decks they unlock | `onboard.py capability` | CI-able |
| 6 | implement | engine work per missing mechanic, honouring the scripts' tags (no script edits, no retagging), a rules scenario each, `docs/rules_deviations.md` for gaps | developer, one mechanic per commit | manual |
| 7 | vocab append | append the deck's cards at the next free ids (< 900), `make registry` for new enum values (new keywords / effect categories land in reserved capacity) | `onboard.py vocab` edits only `card_vocab_entries`; reviewed diff | manual |
| 8 | text rows | encode the new cards' oracle text: `make text-table` (fetches the pinned model with `tools/embed/fetch_model.py`, regenerates only stale rows with `rmembed`; needs network and libtorch, `make rmembed`) | existing | manual |
| 9 | checks + fuzz | `make check`; `robomage_check --tier fuzz --matchups <new>:* --fuzz-games 100` bo1, bo3 `explore:patient` and `--no-auto-mana` against every league deck: any draw, cap, `FATAL`, `ERROR` or non-fatal error line is a finding that blocks the deck | the gate | manual campaign; 10-game smoke per onboarded deck in CI |
| 10 | designate | `git mv` the deck to `decks/league/` (train) or `decks/heldout/`; at least one fair and, when available, one combo deck stay held out permanently | `onboard.py designate` | manual decision |
| 11 | baseline | how well `scripted:hard` pilots the deck (does it play it at all) and the current policy's zero-shot benchmark on it (S8) | `rm_driver.py`, `rmtrain bench --decks` | manual |

Rules: a stage never runs past a failed earlier one; `vocab` needs zero parse findings and zero
missing mechanics; `designate` needs a clean fuzz campaign; status files are committed (the audit
trail). `--check-decks` and the gate's `vocab` tier cover `decks/league/` and `decks/heldout/`.
Stages 1–5 need nothing from S1–S5 and may run early, read-only, to choose which decks and mechanics
come first; stage 7 onwards waits for S5.

### S8. First expansion and the generalisation measurement (1 day tooling + runs)

- **Benchmark on held-out decks:** `rmtrain bench` gains `--decks league,heldout` (the schedule over
  pairs of the listed decks; today it uses the league pairs only, `learn/rmlearn/matches.py`). Report
  per held-out deck: the policy **playing** it and **facing** it, paired between checkpoints.
  `scripted:hard` pilots unseen decks with generic heuristics only, so compare policies with each
  other on the same matches (paired deltas, the ladder), not absolute win rates against the
  league decks' numbers.
- **Expansion procedure for a training run:** stop the run (`rmtrain stop`) → `make BUILD=RELEASE`
  (the vocab is compiled into the engine) → `rmtrain retable <run>` → add the new stems to the
  run's `league.decks` (`rmtrain set` if live, else `config.json`) → `rmtrain start <run>`. The new
  decks enter through self-play from the current policy; no teacher, no behaviour cloning.
- **First measurement:** zero-shot (held-out decks, policy never trained on them) versus the same
  decks after N self-play learner steps with them added to `league.decks` — the gap is what
  generalisation has to close; S4's `id_dropout` and S6 are judged by how they move it.
- **Acceptance:** a held-out deck benchmarked against two checkpoints with paired deltas; one
  training deck added to a running run with the procedure above and its first 500 learner steps
  without errors or refusals.

## 3. Order, cost and what each stage invalidates

| Stage | Depends on | Effort | Risk | Invalidates |
|---|---|---|---|---|
| S0 fixtures | — | ½ day | low | nothing |
| S1 archetypes out | S0 | 3–4 days | medium: wide but mostly deletion; the scripted agent's open-decklist play must stay byte-identical | the RMSTATE format, episodes, the layout (migrated in S5) |
| S2 registry layout, stable ability rows | S1 | 4–5 days | medium: every featurized byte moves once | layout, episodes (migrated / refilled in S5) |
| S3 compatibility, exports per vocab, retable | S2 | 2 days | low | nothing beyond S2 |
| S4 seen mask | S2 | ½ day | low | nothing beyond S2 |
| S5 migration, acceptance, cut-over | S1–S4 | 2–3 days + a 2–3 h verification run | medium; the acceptance test is the safety net — write it first | none: it preserves every checkpoint |
| S6 deck summary (optional) | S5, S8's measurement | 1–2 days | low | none (zero-initialised; registry capacity) |
| S7 onboarding pipeline | stages 1–5: none; 7+: S5 | 3–4 days tooling; stage 6 per mechanic | tooling low; engine risk per mechanic | nothing |
| S8 expansion + measurement | S5, S7 | 1 day + runs | low | nothing (adds cards; S2–S5 keep models valid) |

Order: S0 → S1 → S2 → S3 → S4 → S5 (one cut-over) → S7 stages 7–11 → S8 → S6 if measured useful.
S7 stages 1–5 can run in parallel with S1–S5 (read-only). No training run may be active during
S1–S5's builds; after S5, a run only stops for the minutes of an expansion (S8's procedure).

## 4. Challenges and deviations

- **Removing `archetypes.json` removes the reference metagame, not just the labels.** The
  reference lists came from that file, so `k_template`, the belief level (`expert+belief`,
  `b_list`), its gate checks and `rmknow`'s belief modes go too. Under open decklist these were
  redundant (the opponent's list is known); under `--closed-decklist` the sidecar now knows only
  what it has seen. Restoring closed-decklist research later means re-adding a reference set — by
  design no longer developer-labelled.
- **The deck composition needs no new input.** The ledger already holds the opponent's list card
  by card (§1.3). The earlier draft's two new contract inputs (a `DECKLISTS`-fed deck token) and the
  inference-protocol bump are dropped; an in-model summary (S6) is optional and measured.
- **Stored data is the hidden risk (B8).** Ability rows numbered by vocab position would silently
  corrupt replay windows, corpora and exports after an expansion; S2 makes them stable. Without it
  "checkpoint compatibility" would hold while the data fed to the checkpoint was wrong.
- **Compatible is not the same as uninterrupted.** The vocab is compiled into the engine, so adding
  cards needs a rebuild and a stopped run (minutes, S8's procedure); exports are per vocab and are
  rebuilt; new cards' text rows need the pinned text model and `rmembed` (network, libtorch).
- **A known card's inputs can still change** when engine parsing improves; the compatibility check
  passes and the outputs move. S3's drift report makes such changes visible and deliberate.
- **Generalisation is enabled, not guaranteed.** An unseen card is understood only through its
  structured features and oracle-text vector; quality is measured on held-out decks (S8). Decks whose
  value lies in precise multi-step lines (combo) will likely need real self-play time once added —
  league24h-sb's doomsday / reanimator gains came on top of a behaviour-cloned start — and the plan
  deliberately gives new decks no scripted teacher.
- **The benchmark opponent is weak on new decks.** `scripted:hard` pilots unseen decks with generic
  heuristics; absolute win rates against it on new decks are not comparable with league decks. Use
  paired deltas between checkpoints and the ladder.
- **Two layout changes, one migration.** S1 and S2 both change the layout; migrating once after both
  (S5) means no checkpoint is trained in between — S1–S4 must not be interleaved with experiments
  that need new checkpoints.
- **Capacities are guesses.** Registry capacities and `A_MAX` cover a few hundred more cards and a
  few dozen mechanics; exceeding one is a deliberate layout bump through S5's migration.

## 5. Open issues

- `card_seen` marks a card seen however rarely it appeared; a minimum count before its id row is
  trusted may be worth it — decide from the held-out measurement.
- The text model pin (`tools/embed/MODEL_PIN`) is part of compatibility; re-pinning changes every
  text row and is a retraining decision outside this plan.
- Engine-built ability keys outside the fixed key list (`granted_trigger:<script>`,
  `enchant:<filter>`, delayed-trigger keys) resolve to `-1` plus a hashed `ab_key`; new cards add
  more of them, which is fine by construction.
- Whether the `explore` fuzz agents exercise a new deck's mechanics enough is per deck; stage 9's
  campaign size may need raising for decks with rare lines.
