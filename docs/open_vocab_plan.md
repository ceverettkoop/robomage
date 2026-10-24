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
  lists, like `subtype`). A type's numeric capacity: its slots (enum blocks at the enum's capacity)
  rounded up to 8 with a quarter more, at least 8 (object 74 → 96, stack 53 → 72, player 46 → 64, menu
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
  heads, the `EXPERT_BELIEF` value of `knowledge_level`: S1's Done lists every id that moved), zero reserved positions, set `card_seen` from the ids whose `card_id` row differs from its
  initialisation (or from the run's episodes when available), zero never-seen rows. The migration
  needs the old tables (each run's `tables/`) and the old spec to read names from. **The omniscient
  critic** (`model.critic`; in no run's checkpoint, in S0's fixture `runs/open_vocab_s0/ckpt/m11-critic.pt`)
  has its own `Encoder` and so its own `num_proj` / `type_emb` / `cat_emb`: remap those the same
  way, and include critic values (on the decisions' omniscient frames) in S0's fixtures and the
  acceptance test.
- **Acceptance test** (a new check in the gate's `featurize` tier; `actor` when opt-in): (a) S0's
  fixtures: each migrated checkpoint on the new layout gives the fixture outputs, except for the
  effect of S1's removed inputs — so the comparison is against the old checkpoint run with the
  `lists` / `b_lists` tokens removed (S0's `*_nolists` arrays); logits and values within
  1e-5 (fp32, CPU); (b) the synthetic one-card expansion: identical outputs on every decision not
  mentioning the new card.
- **Cut-over:** migrate every checkpoint listed in §0.3 into `runs/<run>/ckpt-v2/` (originals kept);
  regenerate exports; then a verification run: `rmtrain start` from the best migrated checkpoint
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
