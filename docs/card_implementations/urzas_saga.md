# Urza's Saga

**Vocab index:** 296

## Oracle
```
Urza's Saga
Enchantment Land — Urza's Saga
(As this Saga enters and after your draw step, add a lore counter. Sacrifice after III.)
I  — Urza's Saga gains "{T}: Add {C}."
II — Urza's Saga gains "{2}, {T}: Create a 0/0 colorless Construct artifact creature token
     with 'This creature gets +1/+1 for each artifact you control.'"
III — Search your library for an artifact card with mana cost {0} or {1}, put it onto the
      battlefield, then shuffle.
```

## Forge source
Pre-existing local script `bin/resources/cardsfolder/u/urzas_saga.txt`. Token script
`bin/resources/tokenscripts/c_0_0_a_construct_total_artifacts.txt` was already present. Key tags:
- `K:Chapter:3:Animate1,Animate2,Tutor` — three chapter abilities (final chapter number 3).
- `SVar:Animate1: DB$ Animate | Defined$ Self | Abilities$ ABMana | Duration$ Permanent` →
  `ABMana: AB$ Mana | Cost$ T | Produced$ C` (chapter I grants the mana ability).
- `SVar:Animate2: DB$ Animate | Defined$ Self | Abilities$ ABToken | Duration$ Permanent` →
  `ABToken: AB$ Token | Cost$ 2 T | TokenScript$ c_0_0_a_construct_total_artifacts` (chapter II
  grants the token-maker ability).
- `SVar:Tutor: DB$ ChangeZone | Origin$ Library | Destination$ Battlefield |
  ChangeType$ Artifact.ManaCost0,Artifact.ManaCost1 | ChangeNum$ 1` (chapter III tutor).
- Construct token: `S:Mode$ Continuous | Affected$ Card.Self | AddPower$ X | AddToughness$ X`,
  `SVar:X:Count$Valid Artifact.YouCtrl`.

## Engine work

### Shared Saga engine (CR 714) — `src/saga.{h,cpp}`
Used by **both** this card and Summon: Bahamut. Pieces:
- **Chapter parse** (`src/parse.cpp`, K-loop): `K:Chapter:<final>:<svar1>,…,<svarN>` parses each
  named SVar's DB$ body into `CardData::saga_chapters[i]` (1-indexed chapter abilities;
  `saga_chapters.size()` is the final chapter number, CR 714.2d). Multiple chapters may name the
  same SVar.
- **Lore counters** (`saga_add_lore_counters`, `saga_put_precombat_lore_counters` in `saga.cpp`):
  a Saga enters with one `LORE` counter (CR 714.3a — placed in the newly-entered block of
  `apply_permanent_components`, `src/systems/state_manager_statics.cpp`); one more is added as its
  controller's precombat main phase begins (CR 714.3c turn-based action — called from the
  draw→first-main transition in `Game::advance_step`, `src/classes/game.cpp`).
- **Chapter-trigger dispatch** (CR 714.3): each lore counter that newly reaches a chapter emits a
  `SAGA_CHAPTER` event (new id in `src/ecs/events.h`). `StateManager::check_triggered_abilities`
  (`src/systems/state_manager_triggers.cpp`) turns that into the chapter's triggered ability
  (`is_saga_chapter`, source = Saga, controller = its controller) and queues it for normal APNAP
  placement + target selection.
- **Sacrifice SBA** (CR 714.4): a state-based action in `StateManager::state_based_effects`
  (`src/systems/state_manager.cpp`) sacrifices a Saga whose lore counters ≥ its final chapter
  number once no chapter ability of it is still on the stack. The "still on the stack" gate is
  `Permanent::saga_chapters_in_flight`, incremented when a chapter fires (saga.cpp) and
  decremented when the chapter ability leaves the stack (`StackManager::resolve_top`). This holds
  the Saga off until its final chapter resolves.

**714.4 reconciliation.** The checked-in CR 714.4 has no creature-Saga exception, and both this
card and Summon: Bahamut print "Sacrifice after III/IV", so the SBA sacrifices a Saga regardless
of whether it is also a creature — no special-casing needed.

### Granted activated abilities via Animate (chapters I & II)
`DB$ Animate | Abilities$ <svar>` now grants activated abilities. `Ability::animate_granted_abilities`
holds the parsed AB$ ability(ies) (parsed in `parse_svar_ability`, `src/parse.cpp`); for a
`Duration$ Permanent` animate, `effects::animate` (`src/effects/effect_animate.cpp`) pushes them
onto the permanent's activated-ability list (CR 613.1f lasting grant), deduped. Chapter I grants
`{T}: Add {C}`; chapter II grants `{2}, {T}: Create a Construct`.

### Construct token dynamic P/T
Tokens now carry continuous static abilities. `Token::static_abilities` is parsed from the token
script's `S:` lines (`parse_token_script`, `src/parse.cpp`) and copied onto the Permanent in
`bootstrap_token_components` (`src/components/token.cpp`), so `gather_active_statics` applies the
self-buff. The static's `AddPower$ X / AddToughness$ X` with `X = Count$Valid Artifact.YouCtrl`
required a battlefield `Count$Valid <filter>` handler in `evaluate_sa_svar`
(`src/svar_eval.cpp`) — the static-buff path's count routes through the shared
`permanent_matches_filter`.

### Chapter III tutor filter
`Artifact.ManaCost0` / `Artifact.ManaCost1` needed a `ManaCostN` filter qualifier ("mana value
equals exactly N") added to `eval_qualifier` (`src/game_queries.cpp`). The library search then
offers only MV-0/1 artifacts, puts the choice onto the battlefield, and shuffles.

## Behavioral decisions
- Lore counter type is `LORE` (mirrors `TIME`/`Stun`/`LOYALTY` counter naming).
- The precombat-main lore counter is a turn-based action at the draw→first-main step transition,
  matching CR 714.3c (functionally "after your draw step" per the card's reminder text).

## Tests → results
- **Full arc** (`--battlefield-a "Urzas Saga"` / play it, auto-advance): lore 1→2→3, chapter I/II/III
  fire on the right turns, Saga **sacrificed** after III. ✓
- **Chapter I mana**: with the Saga as the only mana source, cast Aether Spellbomb ({1}) paid by
  the granted `{T}: Add {C}`. ✓
- **Chapter II construct**: activated `{2},{T}`; the 0/0 Construct entered and was buffed to **3/3**
  with two other artifacts out (counts itself, scales with artifact count). ✓
- **Chapter III tutor**: search menu offered only the MV-0/1 artifacts (Aether Spellbomb, Mox Opal,
  Pithing Needle) and **excluded** Paradox Engine (MV 5); the chosen artifact entered the
  battlefield and the library was shuffled. ✓
- **Regression**: scripted games (Urza deck vs Bahamut deck), seeds 1–5 — decisive results, no
  draws, no non-fatal errors; "Urza's Saga is sacrificed (final chapter completed)" observed in
  real games.

## Result
Implemented and verified — the full I/II/III arc, the construct's dynamic P/T, the MV-0/1 tutor,
and the post-III sacrifice all behave per the Comprehensive Rules. Built on the shared CR 714 Saga
engine (`src/saga.{h,cpp}`), also used by Summon: Bahamut.
