# Dead Code & Duplicated Logic Audit

Audit of `src/` (~17.5k lines, 123 files). Every dead-code claim was verified by
repo-wide `grep` confirming the definition is the only reference (accounting for
dispatch tables, virtual overrides, and ECS template instantiation). Duplication
findings quote the concrete repeated sites. The generic Austin-Morlan ECS layer
(`src/ecs/`) is intentional library API and is excluded.

Findings are ordered by payoff. Each has a concrete fix.

---

## Tier 1 — Highest payoff (cross-cutting duplication)

These patterns recur across many files; fixing them removes the most code and the
most drift risk. Several agents independently flagged the same underlying idioms.

### 1. "Scan battlefield permanents (optionally controlled by P)" loop — 15+ copies
The same loop guard — iterate entities, require `Permanent` + `Zone`,
`location == BATTLEFIELD`, often `!is_phased_out`, optional `controller` check —
is open-coded throughout. Sites include:
- `state_manager_actions.cpp:57-67`, `:76-84`, `:158-176`, `:402-499`
- `state_manager.cpp:156-167`, `:171-184`, `:198-206` (SBA loops)
- `replacement_effects.cpp:113-136`, `:147-171`
- `effect_sacrifice.cpp:38-50`, `effect_amass.cpp:45-52`, `effect_pump.cpp:29-37`,
  `effect_destroy_all.cpp:33-52`, `effect_choose_card.cpp:31-39`

`game_queries.h` already factors the *subtype-filtered* case
(`controlled_permanents_matching`) but not the general walk. Drift is already
present: some sites check `is_phased_out`, some don't; `effect_pump.cpp:29` and
`effect_choose_card.cpp:31` use the slower full `GetMaxIssuedEntity()` scan rather
than `mEntities` (against the CLAUDE.md guideline).

**Fix:** add to `game_queries.h`:
```cpp
inline bool active_battlefield_permanent(Entity e, Zone::Ownership ctrl = Zone::UNKNOWN);
// std::vector<Entity> battlefield_permanents(orderer, Zone::Ownership ctrl = UNKNOWN);
```
Use the predicate as the loop guard everywhere and the vector form in effect
handlers. Collapses 3–5 lines per site to one and removes the phased-out drift.

### 2. Entity → display name (`"Player A/B"` else card name) — 7+ copies
The literal ternary "if entity has `Player`, name it Player A/B by comparing to
`cur_game.player_a_entity`, else `entity_name(e)`" recurs at:
- `action_processor.cpp:175-182`, `:310-317`, `:866`, `:1046-1057`
- `attack_target_name` helper at `action_processor.cpp:485-489` (already the full body)
- `machine_io.cpp:226`, `state_manager_combat.cpp:99`, `:160`

**Fix:** promote `attack_target_name`'s body to
`std::string target_display_name(Entity)` in `game_queries.h`; call it everywhere.

### 3. Entity → card-vocab-index resolution — 4 drifting copies
The Permanent(token)→CardData→Ability.source chain to `card_name_to_index` is
reimplemented with **different subsets** at:
- `machine_io.cpp:33-41` (`get_card_vocab_idx`)
- `machine_io.cpp:43-54` (`get_stack_card_vocab_idx`)
- `machine_io.cpp:344-357` (inline in `populate_query`)
- `input_logger.cpp:133-141` (inline in `record_chosen_action`)

These drift: `record_chosen_action` omits the Permanent/token case that
`populate_query` handles, so a **logged action's vocab id can differ from the
queried one** — a latent correctness bug for replay/ML, not just cleanup.

**Fix:** one `int entity_card_vocab_idx(Entity)` covering the full chain, used by
all four.

### 4. "Permanent name else `<unknown>`" idiom — ~15 copies (effects/)
`entity_has_component<Permanent>(e) ? GetComponent<Permanent>(e).name : "<unknown>"`
(and the CardData-name variant) at `effect_destroy.cpp:26-28,37-39,55-56`,
`effect_destroy_all.cpp:54-55`, `effect_pump.cpp:69-70`,
`effect_change_zone.cpp:65-67,81-82`, `effect_counter.cpp:45-72` (4×).

**Fix:** `std::string permanent_name(Entity)` in `cli_output.h`/`game_queries.h`.

### 5. "Controller of an ability's source" idiom — 6 copies (effects/)
`Permanent.controller else Zone.owner` of `ab.source` at
`effect_deal_damage.cpp:27-29`, `effect_amass.cpp:39-41`, `effect_token.cpp:28`,
`effect_surveil.cpp:21`, `effect_delayed_trigger.cpp:28`, `effect_gain_life.cpp:27-30`.

**Fix:** `Zone::Ownership source_controller(Entity source)` in `game_queries.h`.

### 6. Creature keyword scan — 1 canonical + 2 verbatim re-implementations
`creature_has_keyword(const Creature&, const char*)` exists at `game_queries.h:104`
but is copy-pasted at `damage.cpp:10` (`source_has_keyword`) and
`action_processor.cpp:675` (`creature_has_kw`, **also dead — see Tier 2**).

**Fix:** delete `creature_has_kw`; make `source_has_keyword` delegate
(`entity_has_component<Creature>(e) && creature_has_keyword(...)`).

---

## Tier 2 — Confirmed dead code (safe deletions)

Each verified: definition is the only reference in `src/`.

| Symbol | Location | Note |
|---|---|---|
| `StackManager::get_stack_contents()` | `stack_manager.cpp:123-134` + `.h:18` | Zero callers; duplicates `Orderer::get_stack()` minus sort. Delete. |
| `creature_has_kw()` | `action_processor.cpp:675-678` | Zero callers; verbatim dup of `creature_has_keyword`. Delete. |
| `cli_print_gui_exit()` | `cli_output.cpp:286-288` + `.h:16` | Zero callers; message emitted inline at `input_logger.cpp:41`. Delete. |
| `pending_actions` + dead `useful_mana_abilities` | `state_manager_actions.cpp:185, 350, 523-528` | Vector built every `determine_legal_actions` call but read only inside the `/* */` block at 524-528 (`useful_mana_abilities` doesn't exist). Remove the decl, the `push_back` at :350, and the comment block. |
| `apply_lifelink_if_any(Game&)` unused param | `state_manager_combat.cpp:48-63` | Body has `(void)game;`; param threaded through 5 call sites for nothing. Drop it. |

### Dead struct fields (write-only or never-touched)
| Field | Location | Note |
|---|---|---|
| `Ability::is_ultimate` | `ability.h:65`, set `parse.cpp:814` | Parsed from `Ultimate$`, never read (comment admits "informational"). Remove field + parse branch; route `Ultimate$` to ignored keys. |
| `CounterParams::count_from_delve` | `ability_params.h:50`, read `effect_put_counter.cpp:24` | Never assigned `true` → the `if` branch at effect_put_counter.cpp:24-27 is unreachable. Real delve path uses `StaticAbility::counter_count_from_delve`. Delete field + branch. |
| `DiscardParams::mode` | `ability_params.h:56`, set `effect_discard.cpp:94` | `Mode$` parsed, never read (comment confirms). Drop field or implement `RevealYouChoose`. |
| `ActionChoice::slot_idx` | `gamestate.h:72`, set `machine_io.cpp:339` | Always `-1`, never read (sibling `zone_ref` *is* read). Remove field + write. |
| `Player::otp` | `player.h:11`, set `game.cpp:34` | Always `false` (comment says "set properly for A in caller" — never happens); read only by `error.cpp:130` dumper. Wire up on-the-play, or remove. |
| `Player::energy_counters` | `player.h:14`, set `game.cpp:37` | Never read by logic/serialization (unlike `poison_counters`); no energy cards. Remove or mark placeholder. |
| `Effect::Replacement::applied` | `effect.h:22` | Never read/written — consumption tracked via a separate local set in `replacement_effects.cpp`. Delete. |

### `Effect` component — registered but never instantiated
`RegisterComponent<Effect>()` runs (`main.cpp:106`) and `error.cpp:159` dumps it,
but no path ever `AddComponent<Effect>`. Its non-nested fields are all dead:
`replacements`, `affected_zones`, `affected_types`, `category`, `amount`
(`effect.h:26-32`). **Keep** the nested `struct Replacement` (used via
`CardData::replacement_effects`) — consider hoisting it out of `Effect`. Drop the
component registration and dead fields, or document as a deliberate placeholder.

### Minor dead / redundant
- `ActionCategory::MANA_ABILITY` (`action.h:20`) — self-labeled `// legacy, unused`;
  zero references. Either delete or leave as an explicitly-reserved hole.
- Duplicate `#include` lines: `parse.cpp:22-23` (`error.h` twice);
  `ability.cpp:19` & `:22` (`action_processor.h` twice). Header-guarded but dead lines.

---

## Tier 3 — Localized duplication (lower priority)

- **K: keyword comma-split loop** copied 3× in `parse.cpp:480-490, 502-510, 547-558`
  → extract `static void split_keywords(const std::string&, std::vector<std::string>&)`.
- **Pipe-param key/value split+trim** repeated 6× in `parse.cpp` (`:316-323, 887-894,
  1097-1110, 1323-1332, 1520-1529, 1685-1692`) → `static bool next_param(line, pos&,
  key&, value&)`. Largest copy-paste surface in the file.
- **Trigger-collection loop** identical at `parse.cpp:563-567` (token) and
  `:1493-1497` → have `parse_token_script` call `parse_triggered_abilities`.
- **`library_size()`** (`replacement_effects.cpp:41-50`) hand-rolls a scan that
  `Orderer::get_library_contents(owner).size()` already provides → replace.
- **London mulligan flow** duplicated verbatim per seat at `orderer.cpp:467-511`
  (A) and `:513-556` (B), ~44 lines → extract a `decide_for(owner, kept&,
  mulligans&, priority_is_a)` lambda.
- **`remove_all_abilities` remover collection** identical at
  `state_manager_statics.cpp:717-719` and `:731-733` → shared
  `collect_ability_removers()`.
- **Color-identity-from-mana-cost** block identical at `orderer.cpp:226-236` and
  `:573-581` → `static ColorIdentity color_identity_from(const CardData&)`.
- **Mana-source activation** logic duplicated between auto-payer
  (`mana_system.cpp:520-549`) and interactive payer (`:719-765`), plus the
  activation-counter bump at `action_processor.cpp:294-303, 324-331` and
  `mana_system.cpp:539-548, 755-764` → extract `activate_mana_source(...)` +
  `increment_activation_count(Permanent&, const Ability&)`.
- **`permanent_has_type`** is a file-local static in `effect_amass.cpp:24-28` while
  the same loop is inlined at `effect_sacrifice.cpp:45-47` and
  `effect_destroy_all.cpp:40-43` → promote next to `card_has_type` in `game_queries.h`.
- **Log-write boilerplate** (`if (log_file.is_open()) {...flush();} record_chosen_action`)
  6× in `input_logger.cpp:188-252` → private `commit_choice(actions, choice)`.
- **`mana_symbol(int)`** in `cli_output.cpp:57-67` duplicates the color table in
  `colors.cpp` → call `mana_symbol_str((Colors)idx)` at the lone site
  `cli_output.cpp:361`. Lowest priority.

---

## Notes / non-findings

- **Effect dispatch table is fully live** — every `EffectKind` in
  `effect_table.cpp` is registered and reachable; no dead handlers. But the
  "legacy if/else chain" migration comments (`effect_table.cpp:3-7`, `effects.h:24-26,
  98-99`) are stale: migration is *complete* (`ability.cpp:821-826` is the sole
  dispatch). Update the comments.
- **Layer 1/2/3/5 appliers** in `state_manager_layers.cpp:24-47` are empty but
  *reachable* documented extension points for CR 613 — keep, not dead.
- The two `compare_svar` definitions (`svar_eval.cpp:28` strict vs.
  `ability.cpp:662` permissive) differ intentionally and already share
  `apply_svar_op` — not a true duplicate.
- **Doc drift (not a code defect):** `machine_io.h:40` defines `STATE_SIZE = 2919`
  and `train/env.py` agrees, but `CLAUDE.md`/`MEMORY.md` cite `33794`/`OBS_SIZE
  34392`. Code is internally consistent; the docs are stale.
