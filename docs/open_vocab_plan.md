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
   `rnad-final.pt`; benchmark `bench/end/summary.md`) is the source of the best checkpoints
   (`m10`, `m11`, `final`, +0.04 paired over `bc-final`). **The pilots were stopped before a
   result** (2026-10-24): `pilotA` (warm start from `m11`, `reg_eps` 0.02, fair-pair weights,
   `runs/pilotA.toml`) ran to learner step 562 (`ckpt/rnad-p562.pt`, fixed point `m1`) and has no
   benchmark; `pilotB` (pilotA + the omniscient critic, `runs/pilotB.toml`, critic pretraining
   script `runs/pilotB-prep/pretrain_critic.py`) never ran, so no checkpoint has a critic. The A
   vs B experiment is deferred until this plan is complete, if it is run at all; it would then
   start from the migrated `m11`. Every checkpoint under `runs/**/ckpt/` (fixed points, `*-final.pt`,
   `bc-*.pt`, `rnad-*.pt`) is migrated in S5 — none is to be lost. The critic paths of S0 / S5 are
   still required (the code supports a critic) and are tested on a checkpoint built with one
   (S0's `runs/open_vocab_s0/ckpt/m11-critic.pt`).
   **What the pilots' results carry over:** the refactor keeps a migrated model's behaviour on the
   league decks (S5 verifies it), so the pilots' *training-recipe* conclusions apply to training
   after it: the warm start, `rnad.reg_eps` / `reg_eps_sideboarding`, pair weighting, and whether
   the omniscient critic earns its ~2× learner cost. The best pilot's migrated checkpoint is the
   starting point of the first post-refactor run. What does not carry over: pair weights are per
   deck and must be re-chosen when decks are added (S8); pilotB's critic pretraining read the old
   B0 / DAgger episodes, which are not migrated, so after the cut-over a critic is carried in
   migrated checkpoints or retrained from new omniscient episodes.
4. **Workflow** (CLAUDE.md): a branch `refactor/open-vocab` off the starting branch; one stage at a
   time, each by one subagent told not to spawn agents; every stage ends with the stage's own
   acceptance checks and `make check` green locally before its commit (CLAUDE.md; GitHub CI is
   only the fallback for a "commit and push" with no local run); one commit per stage
   (or per coherent sub-step). Long
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
- **`card_subtype` is closed:** its 518 columns are the schema's whole subtype enum (opened, 1024,
  by layout version 2: S7's Done).

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
  and values on that set under the current tables (fp32, CPU) — and, for a checkpoint with an
  omniscient critic, its critic values on the dumps' omniscient frames. Store them under
  `tests/open_vocab/baseline/` with a manifest (checkpoint path, decision source, tables'
  `spec_hash`).
- Note the current best policy from the benchmark summaries (`runs/*/bench/end/summary.md`): it is
  the policy S5's verification continues from.
- **Acceptance:** the fixture files reproduce bit-for-bit when regenerated.

**Done** (`learn/open_vocab_baseline.py`, its docstring has the details; `generate` ~18 min on 8 × 4
CPU threads; it regenerates the fixtures on the pre-S1 tree only, see S1's Done). Decision sources are committed copies, `tests/open_vocab/sources/`: the 66 corpus logs
and five complete self-play bo3 matches from the end of `league24h-sb` (`select` checked that their
replay featurizes to the run's episodes step for step); 8831 seat decisions (every corpus seat
decision, 200 of the self-play ones), 30987 menu rows. A decision is (source, viewer, game,
sideboarding, decision) — a sideboarding phase numbers its decisions from 1 again. Outputs:
`tests/open_vocab/baseline/` (`decisions.rmtb` + one RMTB per distinct network, ~12 MB) and
`manifest.json` (tables' hashes, torch 2.10.0+rocm7.0, numpy 2.4.2, threads 4, every source's and
checkpoint's sha256). Intermediates (dumps, tensors, tables) are under `runs/open_vocab_s0/`.

- **Removed, not zeroed.** The reference for S5 removes the `lists` / `b_lists` **tokens** (arrays
  `logits_nolists`, `value_nolists`, `critic_value_nolists`): S1 stops emitting them, while zeroing
  their numeric slots would keep the tokens with their type and `player_rel` embeddings, which a
  migrated model never sees. `b_lists` appears only at knowledge `expert+belief`, which no fixture
  decision (and no run) uses; omniscient frames carry no `k_*` tables, so the critic values do not
  change. **The removal is not small:** on the fixture decisions it changes m11's argmax on 2.9% of
  decisions (mean total variation 0.014, p99 0.22) and its value by 0.13 on average (bc-final
  0.23): the trained networks read the archetype tokens.
- **Runnable: 30 of 80 checkpoints, 35 distinct networks** (an R-NaD learner state is `model`,
  `target` and its distinct regularisation policies; a network stored several times — `m11` is
  pilotA's seed and the regularisation policy of league24h-sb's last learners — is evaluated once):
  every `league24h-sb` checkpoint (8 `bc-*`, `m1`–`m11`, `rnad-final`, three `rnad-p*`, `rnad-seed`),
  every `pilotA` one (`m1`, `rnad-p319/p481/p562`, `rnad-seed`) and the critic fixture.
  **Not runnable** (spec hash `9dd29020dd612c00`, not today's `9430d9f42d4559c0`): every checkpoint
  of `league12h`, `val1`, `pilot-p1`, `pilot-p2` and `bc1/bc`; `bc1/v1` is older still
  (`c557af8b9dea864a`). The `9dd2…` tables predate Lorien Revealed's (id 273) two ability rows and
  the `TypeCycling` key row (`ab_keys` 72 → 73): card row 273 differs, ability rows from 310 on
  are shifted (B8 in practice); parameter shapes are the same, so S5 can migrate them by name like
  the others, but no fixture checks their outputs. `runs/t8/*.pt` and `runs/*/exports/live/src-*.pt`
  are not in a `ckpt/` directory and are not covered.
- **Critic fixture** (no run's checkpoint has a critic; `runs/pilotB-prep/m11-critic.pt` was never
  produced): `runs/open_vocab_s0/ckpt/m11-critic.pt`, league24h-sb's `m11` through
  `rnad_learner._load_with_config` with `omniscient_critic` on (critic initialised under a fixed
  seed, its layers' zero-initialised relation biases drawn from a second seeded generator); its
  policy outputs equal `m11`'s bit for bit. `generate` rebuilds it identically.
- **Best policy:** `league24h-sb` `m11` (with `m10` and `final`, tied within their SEs at +0.04
  paired over `bc-final` in both `bench/end` and `bench/end-vs-hard_sb`; the pilots have no
  benchmark).
- **For S5:** featurize exactly as `generate` does (replay each source with `--machine
  --rmstate-dump`, `rmfeat --featurize`, f32 `tok_num` — episodes store f16) and select by the
  index's key; a fixture's batching (sorted by token count, 32 a forward) only moves results at the
  1e-7 level (against `exports/m11.ts.pt`: logits 1.9e-6, values 1.2e-7).

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

**Done** (one commit on `refactor/open-vocab`). Everything in the list above, plus what it implied:

- **Formats.** RMSTATE format version 1 → 2 (`schema.h` `FORMAT_VERSION`, `python/rmstate.py`):
  `FrameKind::METAGAME` and table ids 17-19 / 96-99 are gone (no other id moved); schema hash
  `96ba032381f717e7` → `c4411fe5d918f05c`. CARDS now holds the registered lists' cards and their
  tokens only (no metagame cards; the whole-vocab `--cards-frame` set is unchanged), and
  `--cards-frame` writes SCHEMA + CARDS. FEATURE_SPEC version 2 → 3 (the `metagame` key and the
  `pivot` type kind are gone; the `prob` tag and transform went with `b_*`, their only user);
  `spec_hash` `9430d9f42d4559c0` → `94149a6fc75cbc8d`. Episode version 1 → 2 (no `opp_list` /
  `opp_archetype`; `src/featurize/episode.h`, `learn/rmlearn/episodes.py`, `python/rmtraj.py`):
  version-1 episodes are refused. The sidecar's serialised state `RMKN` 4 → 5.
- **Knowledge.** `expert+belief` is a removed-value error in the engine (`--knowledge-a/-b`),
  refused by `rmknow --level`, the actor schedule, `rm_driver.py` and `rmtrain`'s
  `league.knowledge`. `RootPart` stays in the network (still written by the roots; it was only
  read by the hypotheses). Under `--closed-decklist` the scripted agent's `opp_deck` is `{}` and
  `opp_runs_card` is `None` (unknown), so its closed-decklist play changes; open decklist is
  unchanged (below).
- **Gate.** The `knowledge` tier's two belief sessions became plain expert sessions (seed 51
  stays `--closed-decklist`); the `featurize` tier's `expert+belief` explore game (seed 304) runs
  seat A at knowledge `off` instead; the pivot fold check and the `opp_list` target check are gone.
  The `cli` goldens lost only the archetype / reference-list lines and the `You know:` headers
  that had no other fact under them (22 lines in 3 files). No `.replay` baseline changed.
- **Evidence.** `make check` green with every tier, the opt-in `actor` tier included
  (`ROBOMAGE_TORCH_PYTHON=learn/.venv/bin/python`, `make actor` rebuilt), the fuzz games'
  `SELFCHECK-KNOWLEDGE` clean (`runs/open_vocab_s1/make_check.log`); `learn/check.py`,
  `check_rnad.py`, `check_rmtrain.py` ok. Agent games before / after on the same seeds (pre-change
  binary kept in `runs/open_vocab_s1/before_bin/`, `runs/open_vocab_s1/agent_runs.sh`): 158
  decision logs (scripted vs explore bo3 ×20 on five league pairs and bo1 ×20, scripted vs scripted
  bo3 ×10 on two pairs, `scripted:hard+sb` bo3 ×6 on three pairs) byte-identical, the same
  results. `runs/open_vocab_s1/fixture_check.py`: the S1 engine's featurization of all 8831 S0
  fixture decisions (seat and omniscient) equals S0's `seat_nolists` / `omni_nolists` decisions
  (`runs/open_vocab_s0/decisions/`) exactly, every array, once `tok_cat` is remapped slot by slot
  by offset — so S1 removed the `lists` / `b_lists` inputs and nothing else.
- **For S5 (the categorical vocabularies S1 changes).** `knowledge_level` (the `header.knowledge`
  and `k_meta.level` slots) was `[none, OFF, EXPERT, EXPERT_BELIEF]` (size 4) and is `[none, OFF,
  EXPERT]` (size 3); `frame_kind` lost `METAGAME` (not featurized). No other enum changed. The
  removed value is the last, so every kept value keeps its index within its slot, and only offsets
  move: `header.knowledge` stays at 1, every slot after it shifts −1 (`turn.step` 5 → 4) up to
  `k_meta.level` (3432 → 3431), every slot after that −2 (`k_pool.owner_rel` 3436 → 3434 …
  `ledger.player_rel` 3456 → 3454); the `lists.player_rel` (3460, 4 ids) and `b_lists.player_rel`
  (3464, 4) slots are gone; `categorical_size` 3468 → 3458. Types 33 (`lists`) and 34 (`b_lists`)
  are gone (35 → 33, no other type moves: drop `type_emb` rows 33-34); their numeric slots (55 and
  47) leave `G` 407 → 305 (`F` stays 74); no relation type changes; the card / ability / part
  tables are unchanged. `tools/nn/make_reference_policy.py` needed no change (it reads no list
  size). `RMNet` has no `aux_list` / `aux_archetype` heads (their parameters and
  Adam state are dropped). Removed-value ids (`EXPERT_BELIEF`) never occur in the fixtures.
- **For S5 (the S0 tooling).** `learn/open_vocab_baseline.py generate` reproduces the fixtures
  only on the pre-S1 tree (`13ffb640`, e.g. a worktree): on today's tables every old checkpoint is
  "not runnable" (spec hash), and `RMNet` would refuse their `aux_list` / `aux_archetype` keys. Its
  helpers (`featurize_all`, `decision_set`, `load_decisions`, `forward_all`) work on the new layout
  as they are; `fixture_check.py` shows the use: featurize the committed sources with the current
  engine and select by the fixture's keys. The committed fixtures (`tests/open_vocab/`) are the
  reference and stay as they are.

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
- **Relation types:** (table, column, target) → permanent id (capacity 64; 34 after S1, 38 with the
  history types): `rel_bias` rows are indexed by them, and a running id would shift on any new
  handle or `row:` column.
- **Per-token slot columns:** per (type, slot) its `tok_cat` / `tok_card` / `tok_ability` column, and
  `C`, `K`, `Q` fixed (`card_slot` / `ability_slot` are indexed by the card / ability column).
- **Numeric slots per type:** name → permanent column within the type, per-type capacity (today's
  count rounded up to a multiple of 8, plus headroom); `G` becomes the sum of capacities.
- **Categorical slots:** name → fixed offset and capacity; no running offset. Hashed slots
  (`hash_buckets`) keep their bucket count.
- **Card / ability / part table columns:** column name → permanent column with the enums'
  capacities (`card_subtype` stays 518, closed; layout version 2 opened it at 1024: S7's Done).
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
`git show HEAD:src/featurize/registry.json`). The spec gains `layout_hash` (FNV-1a over the layout's
version and capacities — not over the entries, which an append adds to; the append-only check keeps
every existing entry in place, and a deliberate move raises the version) beside `spec_hash` (exact
identity, kept for logs and episodes).

- **Acceptance:** `make check` green; `rmfeat --spec` shows the registry layout; a synthetic
  test (a copy of the vocab with one card appended at the next free id and one fake keyword
  appended to the registry) produces the same `layout_hash`, the same token tensors for every
  decision of the corpus dumps that does not mention the new card or keyword, and the same ability
  row numbers for every existing (card, index).

**Done** (one commit on `refactor/open-vocab`). The registry, `src/featurize/registry.{h,cpp,json}`
(`docs/featurizer.md` §6.1 has the design; the file is compiled in with `#embed`, which S3 replaced by an assembler `.incbin`: GitHub CI's gcc 13
has no `#embed`), and everything
above, plus what it implied:

- **The registry.** Sections `dims`, `enums`, `token_types`, `numeric`, `categorical`, `cards`,
  `abilities`, `relations`, `tables`, `columns`, `ab_keys`; every entry by name with an explicit
  position; validated at parse (no overlap, within capacity), written canonically one entry per line.
  Capacities: open enums max(8, the power of two ≥ 2n) (`keyword` 50 → 128, `effect_kind` 74 → 256,
  `action_category` 48 → 128, `cast_variant` 12 → 32, `zone` 7 → 16, …); closed enums their size
  (`face`, `seat`, `step`, `color`, `layer`, `day_night`, `lib_anchor`, `subtype` 518 and, deviating
  from the "card types … open" reading, `card_type` 27 and `supertype` 6: they are Forge's whole
  lists, like `subtype`; layout version 2 opened all three, S7's Done). A type's numeric capacity:
  its slots (enum blocks at the enum's capacity) rounded up to 8 with a quarter more, at least 8 (object 74 → 96, stack 53 → 72, player 46 → 64, menu
  35 → 48, lki 39 → 56, the rest 8-24); `F` = 96 (the largest), `G` = 1408 (856 used, 1.5× rounded to
  128); categorical ids 6144 (3902 used); `C` 12 (`hist_event` uses 8, `turn` 6), `K` 6 (object 4),
  `Q` 2 (1); 64 type ids (35 used), 64 relation ids (38); tables card 960 (720 used), ability 448
  (340), part 448 (320); `abilities_per_card` A = 16 (the vocab's most on one card is 5: ids 100,
  262, 288, 298), key band 256 (73 used; 128 keyword keys alone could fill 2× 73).
  `layout_version` 1, `layout_hash` `733819ee9e505abc`; FEATURE_SPEC version 3 → 4, spec hash
  `94149a6fc75cbc8d` → `a56ad74e27244a25`. The bootstrap appended in S1's build order, so every name
  keeps its order: today's file is S1's layout re-addressed.
- **Tooling.** `make registry` (`rmfeat --registry-update`), `rmfeat --registry-check [--base]`
  (the gate's `vocab` tier, `tools/robomage_check/registry_check.cpp`, against `git show HEAD:`),
  `rmfeat --registry FILE` (any mode over another registry). The `featurize` tier's
  `featurize_layout.cpp` is the synthetic acceptance test, kept in the gate: card 353 appended (a copy
  of card 262's rows, 5 abilities) and `FakeKeyword` appended to the registry give the same
  `layout_hash` and dims, every slot / range / relation / key / column in place (only the reserved
  `card_keyword.kw#50` column renamed `card_keyword.kw=FakeKeyword`), every existing card, ability
  and part row unchanged, card 353's abilities at rows 5648-5652, and every corpus decision (17262,
  seat and omniscient) the same tensors; the keyword enum reordered (`FakeKeyword` inserted first, the
  frames' keyword columns shifted) the same positions, card table and tensors (every third
  decision); a full enum, a card past A and a value the registry lacks refused with messages naming
  the cause. A deliberately moved keyword makes it fail (tried, reverted).
- **Formats beside the spec.** Ability rows `card × 16 + index`, the key band at 16384-16639: the
  ability table is `[16640, 448]` (29.8 MB f32; 533 × 114 before), its file carries attrs
  `abilities_per_card` and `key_base`; part rows keep their order and point at the new rows. Episode
  version 2 → 3 (attr `layout_hash` after `episode_version`; version-2 episodes are of the old
  layout and refused; `src/featurize/episode.h`, `learn/rmlearn/episodes.py`, `python/rmtraj.py`).
  The spec gains `layout_version`, `layout_hash`, `G`, `type_capacity`, `relation_capacity`, `enums`
  (the registry's values per enum a slot reads), per type `num_at` / `num_capacity`, per slot `at` /
  `col`, `ab_key_rows`, and the ability table's `abilities_per_card` / `key_base`; `token_types` lists
  the types the spec uses by id (an unused id is ABSENT, `TypeKind::ABSENT`).
- **Learner.** `rmlearn/spec.py` (`n_types` = `type_capacity`, `n_rel` = `relation_capacity`, `G`,
  `numeric_layout()` per type id, `type_by_id`, `enum_values`, `layout_hash`), `model.py`'s
  `TokenEmbed` scatters each type's slots into its registry block; `bc.Categories` reads the slot's
  column and the names from the spec's `enums` (registry order) — `bc.category_names` and `bc.py
  --cards` are gone; `tools/nn/make_reference_policy.py` and `learn/check_rnad.py`'s fabricated spec
  follow; `python/pycheck.py` checks type and relation ids as sets and the key rows. `learn/check.py`,
  `check_rnad.py`, `check_rmtrain.py` ok (`runs/open_vocab_s2/learn_check*.log`).
- **Deviation:** the export's folded `token_table` is `(K × 1025 + Q × 16640 + 1) × d` f32 (≈ 30 MB at
  d 192, 3.6 MB before) because each ability slot copies every ability row; a later change could keep
  the slot vectors as separate bag rows. The GPU's ability projection is 16640 × 448 per step.
- **Evidence.** `make check` green with every tier, the opt-in `actor` tier included (`make actor
  PYTHON=learn/.venv/bin/python`, `ROBOMAGE_TORCH_PYTHON` the venv's python by an **absolute** path:
  a relative one fails the tier's Python-server part, which execs it from another directory;
  `runs/open_vocab_s2/make_check.log`). No game decision changed: no `.replay` baseline or CLI golden
  changed, and agent games before / after on the same seeds (`runs/open_vocab_s1/agent_runs.sh`, the
  pre-change binary in `runs/open_vocab_s2/before_bin/`): 158 decision logs byte-identical, the same
  results. **Fixtures:** `runs/open_vocab_s2/fixture_check.py` featurizes all 8831 S0 fixture
  decisions (seat and omniscient) with the S2 engine and finds them equal, every array, to S0's
  `*_nolists` decisions mapped by name onto the registry (the map below, composed with S1's) — so S2
  re-addressed the inputs and changed nothing else. **A real expansion:**
  `runs/open_vocab_s2/vocab_append_check.py` against an engine built from a scratch copy of the tree
  with Prodigal Sorcerer appended at id 353 (its script fetched into the copy only;
  `runs/open_vocab_s2/vocab_append/`): the same `layout_hash` and spec document, the registry current
  (no entry needed), only card row 353 and ability row 5648 new, every corpus game's dump byte-identical
  and its 17262 decisions featurized to identical tensors. Past a capacity: `rmfeat --registry` with
  `abilities_per_card` 4 refuses card 100, with `keyword` capacity 49 the registry is refused at parse.
- **For S5: the S1 → S2 map** (all by name; both specs name every slot, S1's from a run's
  `tables/spec.json`, S2's from `rmfeat --spec`; `fixture_check.py`'s `Mapping` is the template).
  Token type ids unchanged (0-32; `hist_event` / `hist_card` 33 / 34). Relation type ids unchanged
  (0-33, history 34-37), but `relation_capacity` is 64, so a layer's `rel_bias` `[2 n_rel, H]` maps
  forward row r → r and reverse row `n_rel + r` → `64 + r` (`n_rel` 34, 38 with history). `tok_num`:
  each type's slots keep their S1 index as their column except `stack` (from `copy`, S1 slot 19, +20:
  `variant` is a 32-wide block, was 12) and `menu` (from `card_public`, slot 14, +20, the same); in
  `G` (305 → 1408) a type's block starts at its `num_at` (header 0, turn 16, player 32, player_counter
  96, object 112, …, ledger 784; per type in the spec), so `num_proj` column `S1 offset(t) + j` →
  `num_at(t) + at(t, slot j)`. `tok_cat`: every slot keeps its S1 column; its id range moves to the
  registry's offset (`header.knowledge` 1 → 1, `turn.step` 4 → 10, …, `ledger.player_rel` 3454 →
  3797) and grows to its enum's capacity + 1, every value keeping its index (`cat_emb` row
  `old_offset + 1 + i` → `new_offset + 1 + i`, the none row likewise, the pad 0 → 0; 3458 → 6144
  rows). `type_emb` 33 → 64 rows by id. `tok_card` K 4 → 6 and `tok_ability` Q 1 → 2: columns
  unchanged (`card_slot` / `ability_slot` gain zero rows). Card / ability / part table columns by
  name (634 → 960, 114 → 448, 102 → 448: `card_proj` / `ability_proj` / `part_proj` input columns by
  name, the rest zero). Ability rows: S1 row of `(c, i)` → `16 c + i`, S1 key row `533 - 73 + k` (k in
  sorted key order, unchanged) → `16384 + k`; episodes' `tok_ability` (not migrated) likewise. Part
  rows unchanged. For S0-layout checkpoints compose with S1's map (S1's Done); the fixture check does
  exactly that, and found it exact.
- **For S3.** Check `layout_hash` (layout_version + capacities) **and** that the model's registry
  entries are where the current registry has them: S2's `registry_moves(base, now)`
  (`src/featurize/registry.h`) is that check in C++; a model or export can carry the registry's text
  (45 KB) or the spec document as its manifest of positions, beside the vocab manifest. Episodes
  already carry `layout_hash`. The table drift report can diff `card.rmtb` rows by id directly (rows
  are stable), ability rows by address. A new keyword or effect category is an append (`make
  registry`), not a bump; a card with more than 16 abilities, an enum past its capacity or a new
  type past `F` is a bump (`layout_version` + 1, migration by name).
- **For S4.** `card_seen` indexes card ids, which S2 did not touch; new id rows of `card_id` stay
  1025. The export's `token_table` is the place to fold `card_seen` (and see the deviation above).

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
  a subset of the current one (episodes carry no manifest: the window accepts the spec hashes of the
  run's tables lineage, each checked a subset by `retable`; S3's Done).
- **Table drift report:** `retable` (and the gate's `vocab` tier) also diffs the old and new tables'
  rows for **known** ids: an engine change that alters a known card's parsed features passes the
  compatibility check but changes that card's inputs — the report lists such cards so the change is
  intended, not accidental.
- **Acceptance:** the synthetic expansion of S2 loads every migrated checkpoint (after S5) and the
  pre-expansion exports are refused by the evaluator and `rmserve` with a "rebuild the export"
  message; `rmtrain retable` on a copy of a small finished run succeeds and the run resumes.

**Done** (three commits on `refactor/open-vocab`: the two prep items of S2's review, then S3).

- **Prep 1, `#embed` → `.incbin`** (`cf6c5499`). GitHub CI's `ubuntu-latest` g++ 13 has no `#embed`.
  `src/featurize/registry.cpp` now carries a top-level `asm` block that `.incbin`s
  `src/featurize/registry.json` and a NUL into `.rodata` (Mach-O `.const`) under a global symbol;
  the Makefile passes the file's absolute path (`-DRM_REGISTRY_JSON`) and makes `registry.o`
  depend on it (dependency files do not list an `.incbin`; ccache refuses to cache a TU using it, so
  a cached object is never stale). No codegen. Built in an `ubuntu:24.04` container with g++ 13.3.0
  (`runs/open_vocab_s3/gcc13_build.log`): every binary of plain `make`, the release `rmfeat` /
  `robomage_check` with LTO, the compiled-in text byte-identical to the file in both; the final tree
  again at the end (below). clangd's complaints about `featurize_layout.{h,cpp}` were clangd's (no
  compile database): clang++ and g++ 13 compile both cleanly.
- **Prep 2, the export's size** (`fec48388`). `InferencePolicy` folds each card row once, each
  distinct ability row once (every address with no ability, no features and no parts projects alike
  and shares one row) and the slot vectors (`folded_rows`, 1568 × 192 at today's tables, plus
  `ability_index [16640]` i64), and `gather_folded` adds a slot's row and its slot vector in one bag:
  the token buffer 30.3 MB → 1.33 MB, a d 192 TorchScript export 44.2 MB → 15.3 MB
  (`runs/open_vocab_s3/export_size_{before,after}.log`). The training model's `token_table` is
  unchanged. Folded vs training 1.04e-7, `.ts.pt` 0 and `.pt2` 1.2e-7 from eager (`learn/check.py`).
- **The model manifest** (`src/featurize/compat.h`, `docs/featurizer.md` §6.2): `spec_hash`,
  `vocab_hash`, `text_model`, `layout_hash`, `tables_digest` (FNV-1a 64 of the card digests, below),
  `vocab` — `[id, name]` per card-table row, **named by the whole-vocab CARDS frame** (each face id's
  `card` row), not by `src/card_vocab.h`'s entries (an alias such as Dead / Gone / Dead/Gone is one
  row) — and, in a checkpoint, `registry` (the layout registry's text, the manifest of positions).
  `rmfeat --export-tables` writes it as `manifest.json` (without the registry) and `registry.json`
  beside the tables; `spec.load_tables` requires them (`legacy=True` reads a directory exported
  before S3 with a hashes-only manifest). Checkpoints carry it in `hashes` (a loaded model carries
  the current tables' from then on), exports as metadata strings (`InferencePolicy` attributes,
  TorchScript attributes, AOTInductor metadata, ONNX props, the reference policy too), the inference
  protocol's `HELLO` as `layout_hash`, `tables_digest`, `vocab` (**RMIP v2**: those keys are
  required; requests and responses are v1's but for the number).
- **One check** (`featurize::check_compatible`, `spec.check_compatible`): the same `layout_hash` and
  `text_model`, every `[id, name]` of the model's vocab in the current one unchanged (new ids pass; a
  removed or renamed id refused naming it), no registry entry of the model's moved when both carry a
  registry (C++ `registry_moves`, Python the registry's entries as a set); **exact** (exports) also the
  same `spec_hash`, `vocab_hash` and `tables_digest` and no current id the model lacks, the refusal
  saying to rebuild the export from its checkpoint. Codes are the protocol's (`layout_hash`,
  `text_model`, `vocab_hash`, `spec_hash`, `tables_digest`). A manifest without `layout_hash` (a model
  made before S3) passes only on its own spec / vocab / text hashes. Used by `load_checkpoint`,
  `_load_with_config`, `RNaDLearner.from_state_dict`, the exploiter's and the exporter's loads,
  `export.export_refusal`, the evaluator (`check_model`), `rmserve` (its policies among themselves and
  every handshake), `rmlearn/server.py` and `tools/nn/serve_check.py`. `learn/check.py` runs 15 cases
  through both implementations (`rmfeat --compat`) with the same verdicts.
- **Beyond the plan's list:** `tables_digest` in the exact check. The plan's export rule (manifest
  equal) misses drift: a known card parsed otherwise leaves the vocab, the manifest and the spec
  hash as they were while an export's folded rows go stale; the digest catches it. And **the vocab
  hash alone does not see an appended card** — it is the text table's (`oracle_text.rmtb`), so a card
  appended without `make text-table` leaves `vocab_hash` and `spec_hash` unchanged (the synthetic
  expansion below: both identical before and after); only the manifest shows the new id.
- **Table drift** (`src/featurize/drift.h`): a card's rows (its card row, ability rows `16 id + i`,
  their parts) as nonzero values by column name. The committed `src/featurize/card_digest.json` (330
  cards, 21 KB) holds per card a hash of them printed with 6 significant digits (a libm's last bit does
  not move it); the gate's `vocab` tier (`rmfeat --digest-check`) fails on a known card whose rows
  changed, a new or a gone card, and `make registry` rewrites it (`--digest-update`), so the commit's
  diff names the cards. `rmfeat --table-drift OLD NEW` compares two tables directories exactly
  (columns, old and new values): `retable`'s report.
- **rmtrain** (`learn/rmlearn/orchestrator/retable.py`, `docs/rmtrain.md` §1.1). Every start checks the
  engine's cards frame against the run's `cards.rmstate` (byte for byte) and that the tables carry a
  manifest, else stops naming `rmtrain retable` (the actors featurize with the engine's tables, so a
  run never resumes on stale ones); tables that differ from `state.json`'s `tables` (replaced by hand,
  or a retable's exports interrupted) have every export queued again and the evaluations' and live
  exports removed. **`rmtrain retable <run>`** (a stopped run): the engine's frame and tables into
  `tables.new/`, nothing to do when frame and spec are the run's; the old manifest and **every
  checkpoint** (`ckpt/**/*.pt`) checked against the new tables (any refusal changes nothing); the drift
  report; `tables/` → `tables.<old spec_hash>/` (with its frame), `tables/lineage.json`; every
  `state.json` export (latest per name) rebuilt from its checkpoint into `exports/<name>`, the
  evaluations' and live exports removed; `state.json` (`tables`, `retables`) and a `retable` metrics
  line. The regularisation policies are served by the GPU process from checkpoints, not exports, so
  they need nothing. **The window:** episodes carry no manifest, so the plan's "manifest is a subset"
  is the lineage: `spec.Tables.accepted_spec_hashes` (the tables' and their lineage's, each checked by
  retable) is what the GPU process's window, R-NaD's `trainable_episodes` and BC accept.
  `benchmark.export_missing` rebuilds an export that is missing or not of the run's tables
  (`export_stale`).
- **Gate.** `vocab`: the card digest. `actor`: the reference policy carries the manifest; two
  TorchScript variants built over tables whose manifest lacks the vocab's last card (`stale`) or has
  another digest (`drifted`) are refused by `robomage_nn nn:` (the rebuild message, the digest
  message), `rmserve` serving `stale` refuses today's `nn:remote:` actor (ERROR `vocab_hash`, the
  rebuild message), and the raw-HELLO refusals cover `spec_hash`, `layout_hash`, `tables_digest`,
  `vocab_hash`, the next protocol and an unserved policy. Learner checks: `learn/check.py`
  (compatibility cases, a checkpoint across an appended card and refused across a removed / renamed
  card, another layout, text model or a moved registry entry, an export refused on the grown vocab,
  the C++ parity, the handshake refusals), `check_rmtrain.py` (the start check, the exports a retable
  rebuilds and removes, the window keeping the lineage, the benchmark's rebuild rule), `check_rnad.py`.
- **Acceptance** (S3's part; "every migrated checkpoint" is S5's). The scratch tree
  `runs/open_vocab_s3/vocab_append/` (today's code, Prodigal Sorcerer appended at id 353 with its script
  in the copy only, and Grizzly Bears parsed with power 3 in the copy's `parse.cpp`: a known card's
  inputs changed; the tables' `vocab_hash` and `spec_hash` stay `be1266fd27b3ee13` /
  `a56ad74e27244a25`, the text table not regenerated). `runs/open_vocab_s3/expansion_check.py` (`expansion_check.log`): a checkpoint of
  today's tables loads on the expanded ones (parameters equal, the new manifest); today's `.ts.pt` and
  `.pt2` exports are refused by the expanded `robomage_nn nn:` and its `rmserve` refuses an expanded
  `nn:remote:` client ("the current vocab has card 353 'Prodigal Sorcerer', which the model's lacks;
  … rebuild it from its checkpoint"); the export rebuilt from the checkpoint on the expanded tables
  plays (TorchScript, `.pt2`, through `rmserve`); the drift report lists exactly Grizzly Bears (`power
  1.09861 -> 1.38629`) and the new card, today's tables against themselves nothing, and the digest
  check flags the same two on the expanded engine and passes on today's.
  `runs/open_vocab_s3/retable_check.py` (`retable_check.log`): a copy of the finished smoke run
  (`check_rmtrain.py --smoke --keep`, `runs/open_vocab_s3/smoke/`) on the expanded engine: retable
  checks its 22 checkpoints, reports the drift, keeps the old tables, rebuilds its 5 exports (the
  latest per name) with the new manifest, records it; a second retable has nothing to do. A run
  started from the smoke's `bc-final.pt` and stopped in B4 at learner step 98
  (`retable_check_resume.log`): its copy on the expanded engine refuses to start (the start check
  names retable), retable succeeds, and the run resumes (the learner restored at step 98, the window
  restored with its 273 episodes, stepping on past 148).
- **Evidence.** `make check` green with every tier, the opt-in `actor` tier included
  (`runs/open_vocab_s3/make_check5.log`); `learn/check.py`, `check_rnad.py`, `check_rmtrain.py` ok
  (`runs/open_vocab_s3/learn_check*4.log`, `learn_check_rmtrain5.log`); `check_rmtrain.py --smoke` ok
  (`smoke2.log`; one earlier smoke, `smoke_race.log`, failed only its `phase next (B5 -> done)` step:
  B4 had run past B5's `max_learner_steps` while an evaluation held it, so B5 ended on its own as
  `phase next` arrived and the orchestrator exited unanswered — a race in the smoke's script, not in
  S3). The final tree built with g++ 13.3.0 in `ubuntu:24.04` (glibc 2.39): the card digest checks
  there and the tables' digest is the same (`8a70d95cb6314af9`), so the 6-digit printing holds across
  libms (`runs/open_vocab_s3/gcc13_final.log`). **No decision changed:** no corpus `.replay`, CLI
  golden or scenario changed; agent games before / after (`runs/open_vocab_s1/agent_runs.sh`; before
  = S2's logs `runs/open_vocab_s2/agents_after/`, after `runs/open_vocab_s3/agents_after/`): 158
  decision logs byte-identical.
- **For S4.** `card_seen` folds in two places: the training `CardRows.token_table` and the export's
  `model._fold_rows` (card rows now folded once there, the slot vectors separate rows).
- **For S5.** (1) A migrated checkpoint must carry the S3 manifest: write `hashes` from the new
  tables (`load_tables` of current tables), never copy the old `hashes` (a manifest without
  `layout_hash` is the legacy rule: it loads only on its own exact tables). (2) Old runs' `tables/` have
  no `manifest.json`: read them with `load_tables(dir, legacy=True)` (or the spec and RMTB files
  directly). (3) The orchestrator refuses to start a run whose tables have no manifest: after
  migrating a run's checkpoints, `rmtrain retable` it (it exports the current tables, checks the
  migrated checkpoints, rebuilds the exports; it scans `ckpt/**/*.pt`, so migrated checkpoints must
  be in `ckpt/`, the originals elsewhere — the plan's `ckpt-v2/` is outside `ckpt/`: decide which side
  `ckpt/` holds). (4) The S3 acceptance item left to S5: rerun `expansion_check.py`'s pattern with every
  migrated checkpoint on the scratch tree's expanded tables. (5) Copying a run directory keeps
  `state.json`'s absolute paths (checkpoints, ladder): `retable_check.py`'s `copy_run` rebases them.

### S4. Unseen cards read `unknown_id` (½ day)

- `CardRows` gains a persistent buffer `card_seen [N_CARD_TYPES + 1]` (bool), set for every id in a
  training batch's `tok_card` (an `index_fill_` per batch), saved in the checkpoint.
- `token_table()` uses `feat + (card_seen ? card_id : unknown_id)` per row; exports fold the same
  rule, so an unseen card plays exactly the path id-dropout trains. New id rows start at zero (never
  read while unseen); a card's first marking copies `unknown_id` into its row, so it goes on from the
  path it played unseen (no jump when a new deck's cards enter training).
- Training lever (config, not contract): raise `model.id_dropout` from 0.1 to 0.2–0.25 for runs
  meant to generalise; decided by the held-out benchmark (S8), not assumed.
- **Acceptance:** a unit check that a never-seen id's export row equals its features plus
  `unknown_id`, and that marking it seen changes only that row.

**Done** (one commit on `refactor/open-vocab`; learner only, no engine or format change).

- **The mask** (`learn/rmlearn/model.py` `CardRows`). `card_seen [1025]` bool, a persistent buffer
  (in every `state_dict`: checkpoints, learner states' model / target / regs, the server's pushed
  states). `card_id_rows()` = `card_id · seen + unknown_id · (1 − seen)` (× `card_real`), so a seen
  row is bit for bit what it was before S4 and an unseen one is `unknown_id`; `card_rows()`, the
  training `token_table()` and the export's `_fold_rows` all read it (the id-dropout copy stays
  `feat + unknown_id`). `card_id` initialises to zero (no RNG draw: a fresh model's later
  initialisations draw differently than before; nothing pinned them).
- **Marking.** `CardRows.mark_seen(ids)` (no host sync; negative entries and the none row skipped)
  is called by every training loop on each batch before its forward: `BCTrainer.train` (the batch's
  `tok_card`), the exploiter's PPO minibatch, and `RNaDLearner.accumulate_gradients` (the batch's ids,
  seat and, with the critic, omniscient: `TrajectoryBatch.card_ids`, cached per episode by
  `Featurization.card_ids`) on the current network **and** the target network. The regularisation
  policies keep the mask they were frozen with (a fixed point is a frozen function). Marking is
  explicit, not inside `forward`: R-NaD's chunked, checkpointed forwards would otherwise mark
  between a forward and its recomputation.
- **Deviation: a newly seen card's id row is set to `unknown_id`** (each network its own) at its
  first marking, rather than starting from zero. The row it reads is then unchanged at the marking
  and trains on from the unseen path — so when S8 adds a deck to a running run, the learner, the
  target and the actors' μ (recorded with the unseen path) agree at first sight, and v-trace sees no
  jump. The plan's "a residual on `unknown_id`" read literally (`feat + unknown_id + card_id`) would
  have made every seen slot train `unknown_id` and needed every migrated row reparametrised.
- **Deviation: the aux heads read `unknown_id` detached.** They score every id (a softmax over all
  1025), so with the mask the hundreds of unseen and empty ids would each train the shared unknown
  vector toward "unlikely"; detached, only the tokens' path (id-dropout) shapes it, and an unseen
  card's own row gets no aux gradient either (it stays zero).
- **Old checkpoints** (no `card_seen`): refused by every load (`CardRows._load_from_state_dict`
  appends the error, so `strict=False` loads — `_load_with_config` — refuse too): "card_seen is
  missing: … migrate it (S5) or state the ids its training saw". The explicit hooks:
  `CardRows.set_seen(ids)` (the mask exactly `ids`, every other `card_id` row zeroed) and
  `load_checkpoint(path, tables, seen=ids)` (only for a file without the mask; refused for one with
  it). Neither default is safe: "all seen" gives never-trained ids their stale rows, "none seen"
  moves every known card onto `unknown_id`.
- **What an old network's rows say: nothing.** `runs/open_vocab_s4/seen_analysis*.py` rebuilds
  league24h-sb's initial `card_id` (seed 2, the pre-S1 construction order) and compares: in `bc-b1`,
  `bc-final`, `m1`, `m11` and `rnad-final` (model, target, both regs) **every** row but the none row
  differs from its initialisation, and none is a scaled copy of it — the aux heads' softmax over all
  ids moved even the 547 ids with no card. So S5's "the ids whose `card_id` row differs from its
  initialisation" was wrong (fixed above). **The sets from the data:**
  `runs/open_vocab_s4/episode_card_ids.py` (the union of `tok_card` / `omni_tok_card` over a run's
  episode files, any episode version) gives, in `runs/open_vocab_s4/card_ids_<run>.json`:
  league24h-sb 216 ids (203 cards, 13 tokens: 900-902, 906, 908, 909, 911-916, 1023), the same in
  each of its b0 / dagger / selfplay directories (every decision's ledger holds both registered lists,
  so the first batches already hold every deck card); league12h, pilotA, pilot-p1, pilot-p2 the same
  216; val1 103, a subset. (`bc1` has no `ckpt/` and was not scanned.)
- **The lever.** `model.id_dropout` (default 0.1, unchanged) is documented as the generalisation
  lever (`docs/rmtrain.md`, the config's help, `ModelConfig`). It did not reach R-NaD runs seeded
  from a checkpoint (`rnad_init` took the checkpoint's value) nor resumed models: the run's value now
  applies to a `run.start_from` / BC-final seed, a resumed R-NaD learner and a resumed BC model
  (every existing run has 0.1 in both, so no run's behaviour changes).
- **Acceptance** (`learn/check.py` "seen cards", run by the learner checks; `check_rnad.py` "seen
  cards", `check.py` BC): on a model with every card seen but one card `u` of the batch, `u`'s
  folded export row equals `card_feature_rows()[u] + unknown_id` exactly, every other folded row its
  features plus its id row, the training token table's and id-dropout copy's rows likewise; marking
  `u` seen (`set_seen`) changes folded row `u` alone (and the batch's outputs); training's marking
  (`mark_seen`) marks exactly the batch's cards, sets `u`'s id row to `unknown_id`, moves no other
  row and no output (0.0); the aux heads give `unknown_id` and `u`'s row no gradient; the mask through
  `save_checkpoint` / `load_checkpoint`; a checkpoint with the buffer removed refused with the message,
  loaded with `seen=`, `seen=` refused on a file with the mask; the TorchScript and `.pt2` exports of
  the network with `u` unseen against eager 0 and 1.2e-7, the TorchScript buffer's row `u` the eager
  one. R-NaD: an omniscient-critic learner's accumulate step marks the batch's 11 seat and omniscient
  ids in the network and the target (each id row its own `unknown_id`), no output of either moves,
  gradient reaches those id rows only and `unknown_id` none, the boundary's new regularisation policy
  carries the target's mask and the older one its own, and the masks survive the learner state. BC:
  30 steps mark the episode's 67 cards, the checkpoint round trip includes the mask.
- **For S5.** (1) Set each migrated network's mask with `CardRows.set_seen(ids)` (it zeroes the
  other `card_id` rows) to its run's set above — the run whose training produced it, plus, for a run
  warm-started from another run's checkpoint (pilotA from `m11`), the seed's set (all 216 anyway);
  the critic fixture `m11-critic.pt` takes league24h-sb's. Every network of a run takes the run's set
  (per-network differences would only be tokens first created after an early checkpoint;
  checkpoints are all past thousands of BC steps). For a run whose episodes are gone, its league
  decks' cards and the tokens they make (the episode scan's way is exact; prefer it). (2) Migrate the
  target and both regularisation policies of a learner state the same way. (3) `card_seen` is not in
  the old state dicts: build the new model, copy by name, `set_seen`, save — never `load_state_dict`
  an old state into a new model (refused). (4) **The mask moves S5 acceptance (a).** A seen card's row is the
  old one bit for bit, but the fixture decisions hold 17 card ids outside league24h-sb's 216 (8, 16,
  17, 38, 49, 59, 69, 70, 75, 78, 79, 83, 85, 86, 87, 88, 89: corpus games of non-league decks), in
  2047 of the 8831 decisions; with the true mask those cards read `unknown_id` where S0's outputs
  read their (aux-trained, never input-trained) rows. So run (a) twice: with the mask widened to the
  fixtures' ids (a test-only copy: the old function, 1e-5 everywhere), and with the true mask (equal
  on the 6784 decisions without such a card; the 2047 report the mask's effect, the intended
  change). The cut-over's step-0 benchmark plays league decks only, all seen. (5) Adam state (`m`, `v`) of `card_id` carries over unchanged; unseen rows' `m` / `v`
  may be zeroed with the rows (they are never read until a first marking resets the row).
- **Evidence.** `learn/check.py` (`runs/open_vocab_s4/learn_check2.log`), `check_rnad.py`
  (`learn_check_rnad.log`), `check_rmtrain.py` (`learn_check_rmtrain.log`) ok; `check_rmtrain.py
  --smoke` ok (`smoke.log`, 197 s, the `phase next (B5 -> done)` race of S3 not hit; its BC
  checkpoints carry 88 seen ids, its last fixed point 89: self-play met one more token);
  `make check` green with every tier, the opt-in `actor` tier included (`make_check.log`; `make actor`
  rebuilt, `ROBOMAGE_TORCH_PYTHON` absolute). No engine, format, decision or export layout changed.

### S5. Migration, acceptance test and the one-time cut-over (2–3 days)

- **`python -m rmlearn.migrate OLD.pt --tables NEW_TABLES --out NEW.pt`** for `save_checkpoint` files
  and R-NaD learner states (`model`, `target`, both `regs`, Adam `m` / `v`): build the new-layout
  model, copy each old parameter slot to its new position **by name** (`cat_emb` rows by slot and
  value name, `num_proj` columns by type and slot name, `type_emb` rows by type name, table
  projection columns by column name), drop slots S1 removed (`lists`, `b_lists` columns, the two aux
  heads, the `EXPERT_BELIEF` value of `knowledge_level`: S1's Done lists every id that moved), zero
  reserved positions, and set `card_seen` with `CardRows.set_seen(ids)` (it also zeroes every other
  `card_id` row) to the card ids the run's training inputs held — **not** from the `card_id` rows,
  which do not tell (S4's Done: every row of an old network moved); S4's Done gives the sets per
  run. The migration needs the old tables (each run's `tables/`) and the old spec to read names from. **The omniscient
  critic** (`model.critic`; in no run's checkpoint, in S0's fixture `runs/open_vocab_s0/ckpt/m11-critic.pt`)
  has its own `Encoder` and so its own `num_proj` / `type_emb` / `cat_emb`: remap those the same
  way, and include critic values (on the decisions' omniscient frames) in S0's fixtures and the
  acceptance test.
- **Acceptance test** (a new check in the gate's `featurize` tier; `actor` when opt-in): (a) S0's
  fixtures: each migrated checkpoint on the new layout gives the fixture outputs, except for the
  effect of S1's removed inputs — so the comparison is against the old checkpoint run with the
  `lists` / `b_lists` tokens removed (S0's `*_nolists` arrays); logits and values within
  1e-4 (fp32, CPU; 1e-5 is below the fixtures' own fp32 rounding, S5's Done), with the seen mask widened to the fixtures' card ids (with the true mask only
  the decisions holding no card outside the run's set match: S4's Done, For S5 (4)); (b) the synthetic one-card expansion: identical outputs on every decision not
  mentioning the new card.
- **Cut-over:** migrate every checkpoint listed in §0.3 (`python -m rmlearn.migrate --run`: the migrated
  set in `runs/<run>/ckpt/`, where `state.json` and `rmtrain retable` look, the originals moved to
  `runs/<run>/ckpt.v1/` and the old exports to `exports.v1/`; nothing deleted); regenerate exports
  (`rmtrain retable` per run); then a verification run: `rmtrain start` from the best migrated checkpoint
  with its original config (`run.start_from`), 2–3 h of R-NaD, benchmarked (`rmtrain bench`) paired
  against the pre-cut-over export of the same checkpoint: the overall Δ and every deck's Δ must be
  within two SEs of 0 at step 0 (bench the migrated export itself) and not negative beyond two SEs
  after the continuation. The step-0 Δ includes S1's removal of the archetype inputs, which S0
  measured as not small (m11's argmax moves on 2.9% of the fixture decisions): a step-0 Δ beyond
  two SEs is then S1's cost, not a migration error, when the acceptance test (a) passes; the
  continuation must recover it.
- Old episodes (the replay windows, B0/DAgger corpora) are of the old layout and are not migrated;
  a resumed or new run starts with an empty window (self-play refills it in minutes). Keep the
  old corpora on disk until no experiment needs them.

**Done** (two commits on `refactor/open-vocab`: the tooling and gate checks, then this block with the
cut-over's results).

- **`python -m rmlearn.migrate`** (`learn/rmlearn/migrate.py`, `docs/rmtrain.md` §1.2). A `LayoutMap`
  from the old spec document (any version: running offsets before the registry, registry positions
  after it) to the current one, all by name: token types, numeric slots (type, slot), categorical
  ids (type, slot, and an enum value by its name — the old enums' names come from the old run's
  cards frame, whose SCHEMA frame `python/rmstate.py` now reads at an older format version,
  `Schema.from_frame(versions=)`; player-relative and hashed ids by index), relations (table, column,
  target; reverse rows at `n_rel + r` → `64 + r'`), card / ability table columns, card / ability slot
  rows. Every position the old layout lacks is zero. Only S1's removals are dropped (`lists` /
  `b_lists`, `EXPERT_BELIEF`, `aux_list` / `aux_archetype`); any other old name the current layout
  lacks refuses, naming it. The policy encoder and the critic's own (`critic.encoder.*`) take the same
  map. `card_seen` from `--seen` (`set_seen`). Kinds: `save_checkpoint` files (`extra["migrated"]`
  records source, sha256s, seen set, dropped names), BC `.opt` optimizer files (Adam / AdamW moments by
  parameter name), R-NaD learner states (model, target, each distinct reg; Adam `m` / `v` by
  parameter name, `card_id`'s unseen rows zeroed; counters, RNG and configs kept; the window emptied,
  the dropped window's size recorded). Every output carries the current tables' manifest and is
  reloaded through `load_checkpoint` / `RNaDLearner.from_state_dict` before it counts. ~1 s a file.
- **Where migrated checkpoints live** (the plan's `ckpt-v2/` replaced, text above fixed): `--run DIR`
  migrates `DIR/ckpt/**/*.pt` (not TorchScript exports) into `ckpt.migrating/`, then moves the
  originals to `DIR/ckpt.v1/`, the migrated set to `DIR/ckpt/` (`ckpt/migration.json`: per file
  source, both sha256s, kind; the seen set, the old spec hash, the dropped names) and `DIR/exports/`
  to `DIR/exports.v1/`. So `state.json`'s paths, `rmtrain retable` (which scans `ckpt/**/*.pt`) and a
  resume see only migrated files; nothing was deleted. `rmtrain retable` needed three fixes for
  pre-manifest tables: it read the old `spec.json` through `Spec` (a pre-registry spec has no
  `layout_hash`: KeyError), kept them as `tables.pre-manifest/` (now `tables.<spec_hash>/`), and put
  them in `tables/lineage.json` — the window would then have accepted episodes of the old layout's
  spec hash; tables of another layout now join no lineage. A refusal of an unmigrated checkpoint names
  `rmlearn.migrate`. `benchmark.run_benchmark` reuses a policy's `report.json` without its export
  (the step-0 comparison's pre-cut-over policy, whose export the new engine refuses).
- **The cut-over** (`runs/open_vocab_s5/cutover.sh`, `cutover.log`; current tables
  `a56ad74e27244a25`, layout `733819ee9e505abc`, exported by the release engine, byte-identical to the
  debug engine's): 79 of 80 checkpoints migrated, then `rmtrain retable` of league24h-sb, pilotA,
  league12h and val1 (38 exports rebuilt: 17 / 1 / 12 / 8; drift none; lineage empty).

  | run | old spec | files (save_checkpoint + learner state) | seen ids (source) | retable |
  |---|---|---|---|---|
  | league24h-sb | 9430d9f42d4559c0 | 24 (19 + 5) | 216 (S4 scan) | yes, 17 exports |
  | pilotA | 9430d9f42d4559c0 | 5 (1 + 4) | 216 (S4; ⊇ m11's) | yes, 1 export |
  | open_vocab_s0 (critic fixture) | 9430d9f42d4559c0 | 1 (1 + 0) | 216 (league24h-sb's) | not a run |
  | league12h | 9dd29020dd612c00 | 20 (15 + 5) | 216 (S4) | yes, 12 exports |
  | val1 | 9dd29020dd612c00 | 12 (7 + 5) | 103 (S4) | yes, 8 exports |
  | pilot-p1, pilot-p2 | 9dd29020dd612c00 (league12h's tables) | 8 (3 + 5) each | 216 (S4) | no: `run.tables` is league12h's (now the current tables); exports rebuilt on demand (`benchmark.export_missing`) |
  | bc1/bc | 9dd29020dd612c00 | 1 (1 + 0, with its `.opt`) | 216 (`runs/open_vocab_s5/card_ids_bc1.json`, the corpus's episodes) | not a run |
  | bc1/v1/bc | c557af8b9dea864a | **not migrated** | — | — |

  bc1/v1 (FEATURE_SPEC version 1) is refused by name: its token types `own_deck`, `k_seen`,
  `k_known_hand`, `k_template`, `k_archetype`, `b_list`, `b_archetype` have no place in today's layout
  (removed before or in S1; migrating would silently drop inputs it reads); it stays in its `ckpt/`.
  `runs/t8/*.pt` and `exports/live/src-*.pt` are not in a `ckpt/` and were not migrated (as S0).
- **Acceptance (a)** (`learn/open_vocab_acceptance.py full`, `runs/open_vocab_s5/acceptance_full.log`,
  `acceptance/results.json`; 28 min on 8 × 4 threads): all 35 distinct networks of S0's 30 runnable
  checkpoints, migrated in memory from their originals in `ckpt.v1/`, on all 8831 fixture decisions
  as today's engine featurizes them (`runs/open_vocab_s2/fixture_check.py` rerun on this tree: equal
  to S0's mapped decisions, `fixture_check_s2_rerun.log`). Widened mask (the 229 fixture ids):
  max |Δ value| 1.07e-5, |Δ logit| 4.7e-5 on every decision. True mask: the same on the 6784 decisions
  holding only seen cards; on the 2047 others (cards of non-league decks reading `unknown_id`) |Δ value|
  up to 0.10–1.41 and the argmax moved on 28–47 decisions per network (m11: 0.14, 35) — the mask's
  intended effect. The critic fixture's critic values: 6.0e-8 both ways. Every network of the 30 cut-over files
  (53 file-network pairs: the 35 evaluated, 18 more holding the same networks) equals its in-memory
  migration parameter for parameter. **Tolerance 1e-4, not 1e-5** (plan text fixed): the fixtures are
  fp32 and the migrated layout sums its zero-padded columns in another order; the fixtures themselves
  are up to 1.8e-5 (logits) from the float64 function (m1, bc-final), and in **float64 every migrated
  network equals its original exactly** (`runs/open_vocab_s5/fp64_check.py`, `fp64_all.log`: the
  pre-S1 learner code, 13ffb640's `learn/rmlearn` extracted under `old_learn/`, on S0's `seat_nolists`
  decisions over the old tables vs the migration on today's featurization: |Δ| = 0.0 for all 35
  networks, 1104 decisions each). **The pre-Lorien-Revealed checkpoints** (no fixture):
  `runs/open_vocab_s5/check_9dd2.py` (`check_9dd2_new.log`): 60 distinct networks of the 49 `9dd2`
  checkpoints, the original under old code over its own tables on S0's decisions renumbered to the
  9dd2 ability rows (6002 decisions without card 273 or the TypeCycling key; every 4th, 1501) vs the
  migration (widened mask): max |Δ value| 1.0e-5, |Δ logit| 2.2e-5, within 1e-4.
- **Acceptance (b)**: on a copy of today's tables with card 353 appended (`expanded_tables`: card 262's
  rows at 353's addresses, the manifest's vocab grown), 4 networks of the full run (and the gate's 3
  policy networks) load, give identical outputs (0.0) on all 8831 decisions, and their export's folded
  rows differ in card row 353 and ability rows 5648–5652 only. S3's open item: all 79 migrated files
  load over S3's scratch tables (`runs/open_vocab_s3/expansion_work/appended/tables`: Prodigal
  Sorcerer appended, Grizzly Bears drifted), 5 of the distinct networks compared there: 0.0 on the
  decisions holding neither card (all 8831).
- **In the gate.** `actor` tier (opt-in): `open_vocab_acceptance.py gate`, 4 networks (league24h-sb
  `m11`, `rnad-final` model and target, the critic fixture) on the 620 fixture decisions of 8 sources
  (league and non-league decks, one self-play match), both masks, the cut-over files, the synthetic
  expansion: ~25 s, `skipped:` on a clone without `runs/`. `featurize` tier (no torch): `python/pycheck.py
  --part open_vocab` replays and featurizes the same 8 sources and checks each of their fixture
  decisions against `tests/open_vocab/inputs.json` (per fixture decision the sha256 of its seat and
  omniscient arrays, recorded on this tree after the S2 fixture check): ~30 s. `learn/check.py`
  "migration": a synthetic old layout (every order permuted, S1's removals added) migrates back
  exactly, Adam's moments follow their parameters, an unknown slot is refused by name.
- **Verification run `runs/ov-verify`** (league24h-sb's `config.json`, `run.start_from` the migrated
  `m11`, `b4.max_learner_steps` 1200 and `b5` 1400 — set live from 900 / 1100 when the step rate was
  known; `runs/open_vocab_s5/ov-verify.json`): B3 → B4 at once, 1400 learner steps (B4 to 1200, B5 to 1400)
  in 2 h 51 min (17:35–20:27, about 0.15 steps/s with 16 local actors), 55,799 self-play games, 7.59 M
  decisions, fixed points `m1` (step 500) and `m2` (1000), `rnad-final`; no fatal, no actor restart,
  no cap. The window started empty and refilled by self-play (first learner step within a minute).
  Its evaluations: `m1` sprt vs scripted 0.59 over 83 games, `m2` 0.546 over 174. One drawn game
  (a6 m593 game 3, burn vs bw_dnt): burn's own Eidolon of the Great Revel and Price of Progress take
  both players below 0 at once — CR 104.4a, the engine right, the policy's line a self-kill.
- **Benchmarks** (`rmtrain bench`, 480 bo3 matches, 12 actors, seed 11, every league pair both ways).
  The pre-cut-over `m11` was not replayed on a pre-S1 engine: league24h-sb's own `bench/end/m11` and
  `bench/end-vs-hard_sb/m11` were played with exactly the benchmark's defaults (480, 12, seed 11, the
  same opponents) on the engine of `6545ca7c`, decision-identical to 13ffb640's, and S1–S4 kept every
  scripted decision byte-identical, so the same (actor, match) keys are the same deals, decks and
  seats; their directories were copied in as `pre-m11` (the report-reuse rule above), and the
  migrated `m11` (`runs/league24h-sb/exports/m11.ts.pt`, retable's) played. Half the matches came out
  move for move identical (`same` 49% / 51%), which only identical setups allow. Step 0
  (`runs/ov-verify/bench/step0/`, `step0-vs-hard_sb/`): vs scripted:hard pre 0.499, migrated 0.499,
  **Δ −0.000 ± 0.014**; vs hard+sb pre 0.505, migrated 0.507, **Δ +0.002 ± 0.014**. Per deck (vs hard;
  vs hard+sb): bug +0.002 ± 0.067; −0.006 ± 0.040 · burn +0.014 ± 0.020; +0.011 ± 0.047 · bw_dnt
  −0.005 ± 0.044; +0.000 ± 0.017 · gw_maverick +0.019 ± 0.045; +0.005 ± 0.047 · lands +0.013 ± 0.045;
  −0.008 ± 0.027 · reanimator −0.016 ± 0.043; −0.009 ± 0.022 · ur_delver −0.012 ± 0.023; +0.019 ± 0.045
  · uw_control −0.012 ± 0.035; +0.003 ± 0.053 · wrb_energy −0.005 ± 0.049; +0.040 ± 0.061 ·
  wubg_doomsday −0.009 ± 0.076; −0.041 ± 0.073. Every Δ within two SEs of 0: S1's removal of the
  archetype inputs (2.9% of the fixture decisions' argmax) costs nothing measurable against the
  scripted opponents, and acceptance (a) passed, so there is no migration error to separate out.
  After the continuation (`runs/ov-verify/bench/end/`, `end-vs-hard_sb/`; Δ vs `pre-m11`, paired):

  | policy | vs scripted:hard | vs scripted:hard+sb |
  |---|---|---|
  | m11 migrated (step 0) | 0.499, −0.000 ± 0.014 | 0.507, +0.002 ± 0.014 |
  | ov-verify m1 (step 500) | 0.499, −0.000 ± 0.014 | 0.512, +0.007 ± 0.015 |
  | ov-verify m2 (step 1000) | 0.500, +0.001 ± 0.015 | 0.516, +0.010 ± 0.016 |
  | ov-verify final (step 1400) | 0.505, +0.006 ± 0.018 | 0.512, +0.006 ± 0.018 |
  | final@thresh=0.03:disc=16 | 0.521, +0.022 ± 0.027 | 0.536, +0.031 ± 0.027 |

  Per deck `final` vs `pre-m11` (hard; hard+sb): bug +0.007 ± 0.059; −0.039 ± 0.058 · burn +0.006 ±
  0.035; +0.029 ± 0.038 · bw_dnt −0.037 ± 0.057; −0.019 ± 0.034 · gw_maverick −0.004 ± 0.065; +0.019 ±
  0.068 · lands +0.020 ± 0.051; −0.025 ± 0.051 · reanimator −0.023 ± 0.061; +0.005 ± 0.024 · ur_delver
  +0.019 ± 0.051; +0.035 ± 0.059 · uw_control +0.053 ± 0.077; +0.043 ± 0.081 · wrb_energy +0.042 ±
  0.052; +0.023 ± 0.057 · wubg_doomsday −0.028 ± 0.077; −0.007 ± 0.070 (the other policies' rows are
  in the summaries; none is negative beyond two SEs). Nothing is negative beyond two SEs, overall or
  for any deck or policy: the cut-over keeps the policy's strength and training continues from it.
  (The projected `final` plays another option set than the argmax baseline; its +0.03 is the
  fine-tuning projection's, as in league24h-sb, not a gain of the continuation.)
- **Evidence.** `make check` green with every tier, the opt-in `actor` tier included (before the
  cut-over, the acceptance then on the originals in place: `runs/open_vocab_s5/make_check.log`), and the
  `featurize` and `actor` tiers again after it (originals in `ckpt.v1/`, cut-over files compared:
  `gate_after_cutover.log`); `learn/check.py`, `check_rnad.py`, `check_rmtrain.py` and `check_rmtrain.py
  --smoke` ok (`learn_check*.log`, `smoke.log`). A rehearsal on a copy of pilotA
  (`runs/open_vocab_s5/rehearsal/`): retable refuses the unmigrated run naming the migration; migrate,
  then retable, succeed. No engine, format or decision change in S5.
- **Deviations.** Tolerance 1e-4 (above). Migrated checkpoints replace `ckpt/` (originals in
  `ckpt.v1/`), not `ckpt-v2/`. bc1/v1 not migrated (reason above). BC `.opt` files migrated too (the
  plan named only checkpoints and learner states). The step-0 baseline reused league24h-sb's own
  benchmark rather than an old-engine replay (the instructions' second option; conditions above).
- **For S7 / S8.** (1) A layout change past the registry's capacities (a `layout_version` bump; done
  once since, layout version 2: S7's Done) is a migration by the same tool: give the old run's tables and cards frame (`--old-tables`,
  `--old-cards`); for a registry-layout old spec the positions come from its `num_at` / `at` / `col`
  / offsets, already supported. (2) The S0 fixtures' acceptance now holds on recorded inputs
  (`tests/open_vocab/inputs.json`): an engine change that moves a fixture decision's inputs fails the
  `featurize` tier; if intended, `learn/open_vocab_acceptance.py inputs --drop-moved` drops those
  decisions (the gate's acceptance then checks the rest) — expect this when S7 changes parsing of a
  known card. (3) Seen sets: every migrated network's mask is its run's 216 ids (val1 103); a new deck's
  cards play `unknown_id` until a training batch holds them. (4) league24h-sb's migrated `m11` (and
  `ov-verify`'s networks) are the starting points; old episodes are not in any window (league24h-sb's
  `episodes/` stay on disk, refused by episode version, and are outside every lineage). (5) Systemd
  user lingering is off on this box: an `rmtrain` unit dies at logout.

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
| 2 | diff | each card against `src/card_vocab.h` (accent-insensitive, DFC-aware): in vocab / script present, not in vocab / no script | `robomage --check-decks --check-decks-dir candidates/<stem>.dk` (one deck) | CI-able |
| 3 | provision | fetch missing scripts and the tokens they make (engine-synthesised kinds: add stems to `tools/forge_fetch/provision_decks.py` `keyword_tokens`) at the pinned Forge commit; a card Forge lacks at the pin becomes the capability item `script:<card>` (a `FORGE_PIN` bump) | `tools/forge_fetch/fetch_script.py` | manual (network) |
| 4 | parse report | load each new script with parser findings captured: unknown ability categories (would resolve as no-ops), unmodelled params, unknown keywords, SVars | new engine mode `--parse-report CARD…` (JSON per card; plays no game) | CI-able |
| 5 | capability | per card clean / needs mechanic X; per deck missing cards and mechanics; mechanics shared across candidates ranked by how many decks they unlock | `onboard.py capability` | CI-able |
| 6 | implement | engine work per missing mechanic, honouring the scripts' tags (no script edits, no retagging), a rules scenario each, `docs/rules_deviations.md` for gaps | developer, one mechanic per commit | manual |
| 7 | vocab append | append the deck's cards at the next free ids (< 900), `make registry` for new enum values (new keywords / effect categories land in reserved capacity) | `onboard.py vocab` edits only `src/card_vocab.h` (`card_vocab_entries`, and `token_vocab_entries` for the tokens the new cards make); reviewed diff | manual |
| 8 | text rows | encode the new cards' oracle text: `make text-table` (fetches the pinned model with `tools/embed/fetch_model.py`, regenerates only stale rows with `rmembed`; needs network and libtorch, `make rmembed`) | existing | manual |
| 9 | checks + fuzz | `make check`; `robomage_check --tier fuzz --matchups <dir>/<new>:* --fuzz-games 100` bo1, bo3 `explore:patient` and `--no-auto-mana` against every playable (league and held-out) deck: any draw, cap, `FATAL`, `ERROR` or non-fatal error line is a finding that blocks the deck | the gate | manual campaign; in CI an onboarded deck joins the `fuzz` / `smoke` presets' mirrors and ring |
| 10 | designate | `git mv` the deck to `decks/league/` (train) or `decks/heldout/`; at least one fair and, when available, one combo deck stay held out permanently | `onboard.py designate` | manual decision |
| 11 | baseline | how well `scripted:hard` pilots the deck (does it play it at all) and the current policy's zero-shot benchmark on it (S8) | `rm_driver.py`, `rmtrain bench --decks` | manual |

Rules: a stage never runs past a failed earlier one; `vocab` needs zero parse findings and zero
missing mechanics; `designate` needs a clean fuzz campaign; status files are committed (the audit
trail). `--check-decks` and the gate's `vocab` tier cover `decks/league/` and `decks/heldout/`.
Stages 1–5 need nothing from S1–S5 and may run early, read-only, to choose which decks and mechanics
come first; stage 7 onwards waits for S5.

**Done (tooling)** (two commits on `refactor/open-vocab`: the tooling, then stages 1–5 run for real).
Stages 6–11 were not run for any deck.

- **`tools/onboard/onboard.py`** (its README has the details) — the eleven stages of the table, `all`
  (stages 2–6, the automatic ones: read-only but for provision's fetches), `report` and `check`.
  Status files `decks/candidates/<stem>.status.json`: the deck's source (MTGTop8 name, rank, share,
  section, ids, scrape date and window) and per stage result (`ok` / `fail` / `pending`), time, input
  hashes (deck, `src/card_vocab.h`, the scripts the deck's cards resolve to, the engine binary, the
  token scripts, `mechanic_sizes.json`, the text table), summary and detail. A stage is skipped while
  its inputs are unchanged; stages 2–6 rerun themselves when a later stage needs them, 7–10 need a
  recorded `ok`; `vocab` needs stage 6 `ok` (no finding, no mechanic), `designate` stage 9. The
  status file stays in `candidates/` after stage 10's `git mv`. Stage 9 is `make check` plus
  `robomage_check --tier fuzz --matchups <dir>/<stem>:* --fuzz-games 100 --manual-mana-games 10`;
  stage 11 runs `rm_driver.py` `scripted:hard` with the deck against every league deck, then `rmtrain
  bench <run> --decks`, and fails naming S8 until that flag exists (left to S8, as planned).
- **`robomage --parse-report <card | @vocab | @FILE>`** (`src/parse_report.{h,cpp}`, plays no game):
  each script parsed afresh (`parse_card_findings`, `src/parse.h`) with the parser's findings
  collected — games record nothing — plus a script scan: kinds `category` (no handler: a no-op),
  `ability_param` (`<category>.<key>`), `amount_svar`, `svar_missing` (SubAbility$ / Execute$ /
  Choices$ / … naming no SVar), `trigger_mode` / `trigger_form` (no event binding: the trigger is
  dropped), `trigger_param`, `trigger_phase`, `trigger_zone` (zones the binding reads as "any"),
  `static_mode` (a mode the engine applies nowhere), `static_param`, `static_form`, `replacement`
  (an R: line yielding nothing), `keyword` (a plain keyword no rule reads: the list is the
  keywords the engine's rules read by name — Defender, though in the state format's keyword enum,
  is not one), `cost`, `layout` (AlternateMode Adventure / Meld / …), `filter` (a filter token the
  shared matcher has no reading for: `filter_unknown_qualifiers`, `queries/filters.h`, probing
  `eval_qualifier` without an object; PascalCase tokens that are no type name included, which today
  silently match nothing), `svar_expr` (a count expression `evaluate_svar` would report
  unrecognized: `svar_expr_recognized`, `svar_eval.h`), `token_script`. Tokens named in filters or
  expressions that the engine reads where the filter is used (EquippedBy, YouDontCtrl, sameName,
  ExiledWithSource, …) and the parse-time SVar forms (SVar$ sums, Count$Compare, …) are listed in
  `parse_report.cpp` as read elsewhere. A clean report is not proof (a modelled param may be read
  only in its first cards' forms); a finding names something the engine does not do.
- **Over the current vocab** (308 scripts): three cards, four findings, all real gaps, now
  `docs/rules_deviations.md` §2.8 and the gate's baseline `tests/parse_report/vocab.json`:
  Mystical Dispute's conditional {2} reduction is never applied (`ReduceCost` with `EffectZone$ All`
  and `ValidTarget$ Spell.Blue`: cost statics apply from the battlefield only; checked in the harness
  — one Island cannot cast it against a blue spell), Carpet of Flowers' `ResolvedLimit$ 1` is unread
  (it can add mana in both main phases), Moonshadow's `NoResolvingCheck$ True` is unread (its
  intervening-if is checked at resolution too). Calibration removed the report's false positives
  without changing the parser's behaviour: the static params whose value is what the engine assumes
  (`Type$ Spell` on cost statics, `ValidSA$ Activated[.!ManaAbility]` on CantBeActivated, MustAttack's
  `ValidCreature$ Card.Self`, DisableTriggers' `ValidMode$` / `Destination$`, Damping Sphere's
  `Relative$`), trigger zone `Command` (an Effect's floating trigger), `LockTokenScript$` not read as a
  token, and two AI / prose params (`AIManaPref$`, `DiscardValidDesc$`) added to the parser's
  ignored keys.
- **Scrape** (2026-10-24, MTGTop8 Legacy, window "Last 2 Months" = `meta=39`): N = 25, every
  archetype with at least 1% share — dimir_tempo 10%, uwx_control 8, boros_aggro 6, eldrazi_aggro 5,
  cloudpost_tron_ramp 5, lands_top8 5, show_and_tell 5, ur_tempo 4, doomsday 4, aluren 4,
  death_and_taxes 3, storm 3, reanimator_top8 3, artifacts_blue 2, artifacts_prison 2, bant_control 2,
  welder_cam 2, hogaak 2, mystic_forge 2, elves 2, maverick 1, dragon_stompy 1, cradle_control 1,
  stoneblade 1, painter 1 (MTGTop8's sections: 7 aggro, 7 control, 11 combo). The scraper had three
  bugs: its `.dk` lines were tab-separated, which `classes/deck.cpp` splits wrongly (every card read
  as missing); it took the page's first N links, but the page lists its AGGRO, CONTROL and COMBO
  sections one after another, so `--top 25` was every aggro deck first (now merged by share); the
  window was the site's default two weeks. It gained `--meta` and `--manifest`. Stems that name a
  league deck got `_top8` (lands, reanimator).
- **Diff and provision:** 349 distinct cards, 122 not in the vocab; 116 scripts fetched at the pin (plus
  10 token scripts), 6 cards are newer than `FORGE_PIN` (2026-07-19): Archive Arbiter, Dark Matter
  Manipulator, Giant's Boulder, The Echoverse Fulcrum, Theorist's Proxy (on Forge's master) and
  Carnivorous Cultivator (not under its own name there). They are the capability item
  `script:<card>` (grouped as one mechanic, the pin bump), not a stage failure, so their decks'
  other needs are still measured. `fetch_script.py`'s DFC discovery listed Forge directories with
  the contents API, which stops at 1000 entries — four double-faced cards (Brazen Borrower, Shatterskull
  Smashing, Sink into Stupor, Sundering Eruption) were reported "not on Forge"; it now lists with the
  git trees API (`<pin>:<path>`), the contents API as the fallback.
- **Capability** (`bin/resources/decks/candidates/REPORT.md`, generated): of the 122 new cards 51 parse
  clean, 65 need a mechanic, 6 need the pin bump. `tools/onboard/mechanic_sizes.json` groups the
  findings into mechanics with a size each (61 of them needed by the candidates) (S ≤ ½ day … XL a week+; my estimates from the
  scripts). Decks needing nothing: death_and_taxes and uwx_control (every card already in the vocab),
  bant_control (5 new cards, all clean); then lands_top8 (2 S filter qualifiers for Drop of Honey),
  boros_aggro (Lazotep Quarry's copy-with-exceptions M, Sand Scout S), eldrazi_aggro (LandPlayed
  trigger M, repeated modes M), reanimator_top8 (8 points), show_and_tell (9) … painter (54). Mechanics
  by the decks that need them: the pin bump 8; `without<Keyword>` filter qualifier (S) 7; can't-block
  restrictions (M) 5; LandPlayed trigger (City of Traitors, M) 5; repeated modes (the Confluences, M)
  5; discard-a-card costs (M) 4; Sunburst (M) 4; Adventure (L) 4; exile-the-card-itself costs (S) 3;
  DB$ Repeat (M) 3; the rest 1–2 decks each. No mechanic is any deck's sole blocker.
- **Recommended first onboarding set:** held out, fair — **bant_control** (no engine work; five unseen
  cards: Celestial Purge, Hedge Maze, Snow-Covered Plains, Terminus, Up the Beanstalk; a control
  shell the league's uw_control resembles only partly); held out, combo — **show_and_tell** (an
  archetype the league lacks; three M mechanics: free-cast statics for Omniscience, MinCharmNum for
  Avengers Disassembled, the LandPlayed trigger for City of Traitors, ≈ 3 days); added to training —
  **eldrazi_aggro** (5% share, an archetype the league lacks; the LandPlayed trigger it shares with
  Show and Tell plus repeated modes, which four more candidates' Confluences need) or, if no engine work
  may precede S8, **lands_top8** (two S filter qualifiers; three new cards). death_and_taxes and
  uwx_control add no card and test only new lists of known cards.
- **Gate.** `vocab`: `--check-decks` covers the top-level, `league/` and `heldout/` decks
  (`heldout/` may be empty; `--check-decks-dir` also takes one `.dk`); the parse report over the vocab
  against `tests/parse_report/vocab.json` (a new or vanished finding fails; `--record` rewrites it);
  `onboard.py check` (stages 2, 4, 5 of every candidate in memory, no network: a tool failure
  fails, a stale status file warns), ~1 s together. `fuzz` / `smoke`: the presets run over the
  playable decks, `league/` then `heldout/` (`playable_decks`), so a deck designated into either
  directory gets 8 bo1 mirror games, a bo3 and two `--no-auto-mana` ring games and four scripted
  smoke games per gate run — the "10-game smoke per onboarded deck" with no further wiring;
  `--matchups a:*` is `a` against every playable deck, and an explicit `--matchups` list also drives
  the bo3 and manual-mana legs (a deck's whole campaign). `tools/forge_fetch/provision_decks.py`
  provisions the `heldout/` and `candidates/` decks' cards (`meta/` is gone), so CI has the
  candidates' scripts.
- **Evidence.** `make check` green with every tier, the opt-in `actor` tier included (`make actor
  PYTHON=learn/.venv/bin/python`, `ROBOMAGE_TORCH_PYTHON` absolute; `runs/open_vocab_s7/make_check.log`;
  its two `STALE` warnings were status files regenerated while it ran — the `vocab` tier rerun on the
  committed files is clean). No game decision changed (no corpus `.replay`, CLI golden or scenario
  changed): the parser's findings are recorded only under `--parse-report`, and the filter probe and
  the SVar recogniser read no game state. `onboard.py vocab --dry-run` on bant_control and
  `append_vocab` on a copy of `src/card_vocab.h` (a card, a transforming DFC, a split card, a token;
  idempotent) checked by hand; stages 8–11 were not exercised.
- **Deviations.** Stage 7 also appends the new cards' token scripts to `token_vocab_entries` (the
  table said only `card_vocab_entries`; a token outside it reads the sentinel). A card missing at the
  Forge pin is a capability item rather than a provision failure (above). `all` runs stages 2–6, not
  every stage: 1 and 3's fetches need the network, 7–11 change the tree or need a decision.

**Done (layout version 2: the type enums' capacities)** (one commit on `refactor/open-vocab`, before
the engine's subtype list; `runs/open_vocab_s7/subtypes/`). Onboarding met token types the engine's
subtype list lacks (Eldrazi Scion, Sand); the list was Forge's, not CR 205.3's, and the registry had
`subtype` closed at its 518 values, so no subtype could be appended. One bump, so that the next one is
far off:

- **Capacities** (`rmfeat --registry-raise`, `Registry::raise_capacities`: the raised enums' blocks
  widen in place, every later position in their space moves by the growth, a space that no longer
  fits grows keeping its spare room): `subtype` 518 → **1024**, open (CR 205.3 lists 535 today; new
  sets add a few creature types a year: decades of room; the 2n rule's 2048 would double three
  blocks for nothing); `card_type` 27 → **32** and `supertype` 6 → **8**, open (their masks' widths,
  `u32` / `u8`, so no larger capacity is possible without a format change; CR 205.2a has a card type
  the engine lists as a subtype, Dungeon, and 205.4a five supertypes); `layout` 8 → **16** (4 used, and
  the candidates already need Adventure and Meld; Flip, Omen and others follow). The other closed
  enums stay closed: `face`, `seat`, `step`, `color`, `layer` (CR 613's layers and sublayers),
  `day_night`, `lib_anchor` are fixed by the rules; every open enum has at least twice its use.
  Result: the card table 960 → **1536** columns (1241 used), the two `enum:subtype` categorical ranges
  519 → 1025 ids (4914 of the 6144 ids used, the capacity unchanged), `object` / `lki` numeric slots
  moved within their blocks (G, F and every type's capacity unchanged). `layout_version` 2,
  `layout_hash` `733819ee9e505abc` → **`e3a65b1f5306272f`**, spec `a56ad74e27244a25` →
  `17a6f15fabca2aaa`; the cards frame, the vocab and the tables' digest (`8a70d95cb6314af9`, by
  column name) unchanged, no decision changed (the registry is featurizer-only).
- **Migration** (`bump.sh`, `bump.log`): `rmlearn.migrate --run … --keep layout1` (new: `--keep`; a
  network carrying the seen mask keeps it, its `card_id` rows and their Adam moments exactly;
  registry-layout old specs read their own `enums`; old reserved columns are not carried; a file of
  another layout than the old tables' refused) over league24h-sb (24 files), pilotA (5),
  open_vocab_s0 (1), pilot-p1 (8), pilot-p2 (8), league12h (20), val1 (12), bc1/bc (1) and ov-verify
  (7): 86 files, the layout-1 originals in each run's **`ckpt.layout1/`** and their exports in
  `exports.layout1/` (S5's `ckpt.v1/` untouched). `rmtrain retable` of league24h-sb, pilotA,
  league12h, val1 and ov-verify (`retable.log`; retable now accepts a layout bump when every
  checkpoint is migrated, and drops the old lineage): 41 exports rebuilt (17 / 1 / 12 / 8 / 3), drift
  none, lineage empty; the layout-1 tables kept as `tables.a56ad74e27244a25/`.
- **Outputs unchanged.** `bump_check.py --fp64` (`bump_check_fp64.log`): every network of the 86 files
  (105 distinct, 179 file-network pairs; the critic fixture's critic too), the layout-1 original over
  the layout-1 tables and featurization (the pre-bump `rmfeat`) against the migrated file over today's,
  on 1018 seat decisions of 10 corpus games, in float64: **0.0** everywhere (fp32: up to 1e-5, the
  summation order). Acceptance (a) (`learn/open_vocab_acceptance.py full`, `acceptance_full.log`): 35
  networks from S0's originals on all 8831 fixture decisions, widened mask max |Δ value| 1.2e-5,
  |Δ logit| 3.1e-5, the critic 6.3e-8, the true mask's 2047 unseen-card decisions as in S5, all 53
  cut-over file-network pairs equal to their in-memory migration; (b) the synthetic expansion 0.0.
  `runs/open_vocab_s2/fixture_check.py` on the new featurization: equal to S0's decisions mapped by
  name (`fixture_check_s2_rerun.log`), so `tests/open_vocab/inputs.json` was rewritten (every digest
  moves with the positions; `inputs.layout1.json` keeps the old one).

**Done (the rules' subtype list)** (one commit after layout version 2). `all_subtypes`
(`src/constants.cpp`) was Forge's list: 74 of CR 205.3g–q's 535 subtypes were missing (among them
Blood, Book, Germ, Gold, Incubator, Infinity, Junk, Map, Omen, Planet, Role, Room, Sand, Saproling,
Scion, Servo, Shard, Stone, Town, Undercity; creature types such as Bison, Hamster, Inkling, Llama,
Survivor, Symbiote; planeswalker types Dellian, Wrenn, Xenagos, Zariel …; planes Avishkar, Zendikar,
Zhalfir …), so token scripts with them (Eldrazi Scion, Sand Warrior) printed `ERROR: UNRECOGNIZED
TYPE TOKEN … registering as subtype`. They are added (592 values); the 57 entries no longer in the
CR (Un-set and retired types: `Baddest,`, `and/or`, `The`, `Elemental?`, …) are kept and listed in a
comment, since the registry's positions are by name. Type lines now match multi-word subtypes
longest first (Time Lord, the planar types; `subtype_entry`, `subtype_max_words`), and an apostrophe
straight or curly alike (the entries use Forge's straight one but in `Serra’s Realm`; the CR spells
C’tan, Urza’s … curly). No local script (466 type lines) parses differently but for the tokens that
were unknown. `make registry` appended the 74 values in reserved capacity: `layout_hash`
`e3a65b1f5306272f` unchanged, spec `17a6f15fabca2aaa` → `cd12b7e39b500e60`, the card digest unchanged, every
fixture decision's input digest unchanged (`tests/open_vocab/inputs.json` as committed), no corpus
`.replay`, CLI golden or scenario changed, and the agent games of `runs/open_vocab_s1/agent_runs.sh`
before and after (`runs/open_vocab_s7/subtypes/agents_{before,after}/`) byte-identical. The runs were
retabled again (`retable2.log`; the SCHEMA frame's subtype table changed): an append, 41 exports rebuilt, the layout-2 tables in each lineage, so the migrated checkpoints load
as they are.

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
