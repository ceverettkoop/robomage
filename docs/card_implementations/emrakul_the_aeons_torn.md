# Emrakul, the Aeons Torn — vocab index 292

## Oracle
```
Emrakul, the Aeons Torn   {15}
Legendary Creature — Eldrazi
15/15
This spell can't be countered.
When you cast this spell, take an extra turn after this one.
Flying, protection from colored spells, annihilator 6
When Emrakul, the Aeons Torn is put into a graveyard from anywhere, its owner
shuffles their graveyard into their library.
```

## Forge source / key script tags
Pre-existing local script: `bin/resources/cardsfolder/e/emrakul_the_aeons_torn.txt`.

| Script line | Mechanic |
|---|---|
| `R:Event$ Counter \| ValidCard$ Card.Self \| ValidSA$ Spell \| Layer$ CantHappen` | can't be countered (self) |
| `K:Flying` | flying (already supported) |
| `K:Protection:Spell.nonColorless:colored spells` | protection from colored spells (NEW) |
| `K:Annihilator:6` | annihilator 6 (NEW) |
| `T:Mode$ ChangesZone \| Origin$ Any \| Destination$ Graveyard \| ValidCard$ Card.Self \| Execute$ TrigShuffle` | death-from-anywhere shuffle trigger |
| `SVar:TrigShuffle:DB$ ChangeZoneAll \| Defined$ TriggeredCardOwner \| ChangeType$ Card \| Origin$ Graveyard \| Destination$ Library \| Shuffle$ True` | shuffle graveyard into library |
| `T:Mode$ SpellCast \| ValidCard$ Card.Self \| Execute$ TrigAddTurn` | self-cast trigger |
| `SVar:TrigAddTurn:DB$ AddTurn \| Defined$ You \| NumTurns$ 1` | take an extra turn (NEW) |

## Engine work

### 1. Annihilator N (CR 702.85)
- Parsed `K:Annihilator:<N>` in `src/parse.cpp` (K-keyword loop): synthesizes a TRIGGERED
  `Sacrifice` ability — `trigger_on = CREATURE_ATTACKED`, `trigger_only_self`, `mandatory`,
  `defined_each_opponent = true`, `sac_valid = "Permanent"`, `sac_count = N`. Fires once per
  declared attack (CR 702.85a) and, being a normal triggered ability, resolves before blockers.
- Generalized the sacrifice effect (`effects::sacrifice`, `src/effects/effect_sacrifice.cpp`):
  the edict / controller branches now loop `Ability::sac_count` (new field, default 1), each
  iteration a fresh `sacrifice_one` choice. The defending player (the controller's opponent, CR
  702.85b) chooses and sacrifices, one at a time; with fewer than N permanents they sacrifice all
  they have (the loop breaks when `sacrifice_one` returns 0). Reuses the existing
  `SACRIFICE_PERMANENT` choice path.

### 2. Extra turn / AddTurn (CR 500.7 / 720)
- New `EffectKind::AddTurn` + handler `effects::add_turn` (`src/effects/effect_add_turn.cpp`),
  registered in `effect_kind.h/.cpp`, `effect_table.cpp`, `effects.h`. `Defined$ You` ⇒ the
  ability's controller; pushes `NumTurns$` (parsed into `Ability::amount`) entries onto a new
  `Game::extra_turns` LIFO queue (`src/classes/game.h`).
- Turn hand-off integration in `Game::advance_step` (`src/classes/game.cpp`, cleanup → next turn):
  before the normal active-player flip, if `extra_turns` is non-empty the player on top takes the
  next turn (the most-recently-added extra turn first, CR 500.7) instead of passing to the opponent.

### 3. Protection from colored spells (CR 702.16)
- Parsed structured `K:Protection:<quality>:<desc>` in `src/parse.cpp`: `Spell.nonColorless` ⇒
  the creature keyword `"Protection from colored spells"`; a structured single colour quality
  normalizes to the literal `"Protection from <color>"` form the color-protection path already
  uses.
- `has_protection_from_colored_spells(const Creature&)` (`src/components/creature.{h,cpp}`) reports
  the keyword; `Ability::is_legal_target` (`src/components/ability.cpp`) rejects a candidate when
  the targeting object **is a SPELL** (`ability_type == Ability::SPELL`) **and** the source has at
  least one colour (`!effective_colors(source).empty()`). The spell-vs-ability distinction is read
  from the Ability's type (known at target-selection time, before the source has a `Spell`
  component), not from the source.

### 4. Death-from-anywhere shuffle (CR 603.6e / 603.10)
- `ChangeZoneAll` shuffle: new `Ability::shuffle_after` flag (`Shuffle$ True`, parsed in
  `apply_param_to_ability`); `effects::change_zone_all` (`src/effects/effect_change_zone_all.cpp`)
  shuffles the destination library after the move (clearing known-top-of-library tracking).
- `Defined$ TriggeredCardOwner`: `change_zone_all` resolves the operated-on player from the source
  card's `Zone.owner` (CR 400.3 ownership is fixed), so "its **owner** shuffles **their**
  graveyard" is correct even if a different player controlled Emrakul.
- Trigger scan fix (`src/systems/state_manager_triggers.cpp`): the self zone-change look-back scan
  (previously battlefield-origin only — "leaves the battlefield") now fires for a `Card.Self`
  `CARD_CHANGED_ZONE` trigger from **any** origin (so a discard from hand / mill from library to
  the graveyard fires it). The per-ability `Origin$` filter still gates an `Origin$ Battlefield`
  "dies" trigger to battlefield departures; ETB self-triggers (destination battlefield) are
  excluded here since the battlefield scan already handles them (no double-fire).

### 5. Can't be countered (CR 614 / 701.5f)
Already handled by the existing self `CANT_BE_COUNTERED` replacement (`Event$ Counter |
ValidCard$ Card.Self | Layer$ CantHappen`, parsed in `src/parse.cpp`; shared with Hexing
Squelcher). No new work — verified.

## Behavioral decisions / scope
- **Protection from colored spells** is implemented for its **targeting** consequence (CR
  702.16b/e): a one-or-more-colors spell can't target Emrakul; a colorless spell, or any
  ability (colored or not), still can. The other protection consequences (CR 702.16c–f — Auras,
  Equipment, damage prevention, blocking) have **no in-vocab interaction** for a "colored spells"
  quality (no colored Aura/Equipment targets a creature in vocab, and damage from a colored spell
  to a 15/15 with this protection has no relevant case), so they are not wired. The targeting
  path is the load-bearing one.
- **Extra turns** are modeled as a per-player LIFO queue consulted at turn hand-off (two-player
  scope per CLAUDE.md). Emrakul grants exactly one extra turn on cast and does not re-trigger, so
  no infinite loop. `NumTurns$ N` queues N.
- **Annihilator**: any permanent the defender controls is a valid sacrifice (`SacValid$
  Permanent`); the defender chooses (CR 702.85b); fewer-than-N controlled ⇒ sacrifice all.

## Tests (train/test_harness.py, JSON scenarios; Emrakul referenced comma-free where args split on commas)
| Scenario | Result |
|---|---|
| Annihilator 6: preset Emrakul + 2 Mountains on A, B has 8 permanents; A attacks | B sacrifices 6 (its choice); on a later attack only 2 remain → B sacrifices both (fewer-than-N case); A wins, no draw |
| Extra turn on cast: 15 Mountains preset, Emrakul cast from hand | cast auto-pays 15, AddTurn trigger logs "Player A takes 1 extra turn"; turn order TURN 0 (A) → TURN 1 (A) → TURN 2 (B) — A takes two turns in a row |
| Can't be countered: A casts Emrakul, B casts Counterspell @ Emrakul | "Emrakul, the Aeons Torn can't be countered"; Emrakul resolves and enters |
| Protection: A controls Emrakul + Grizzly Bears, B casts Lightning Bolt | target menu offers Player A / Grizzly Bears / Player B but **not** Emrakul (red spell can't target it); a colorless source would still be offered (by construction) |
| Death shuffle: A discards Emrakul (cleanup) with Wasteland + Ponder in GY | "Emrakul triggered" → all 3 GY cards moved to library → "Player A shuffles their library" (GY emptied, top-of-library tracking cleared); repeats when re-drawn and re-discarded |
| Scripted full games: temp Emrakul deck vs temp opp deck, seeds 1–4 | all decisive, no draws, no fatal/non-fatal errors |
| Regression: delver vs mav (seeds 1–3), doomsday vs delver (seeds 1–2) | unaffected — no draws/crashes; ChangeZoneAll (Doomsday) and other zone-change triggers still work |

## Result
**Implemented and verified.** Annihilator N, extra-turn/AddTurn, protection-from-colored-spells,
and the death-from-anywhere shuffle are general, reusable handlers; can't-be-countered and flying
reuse existing mechanics. Build clean, no draws, no non-fatal errors.
