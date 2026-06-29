# The One Ring  (vocab index 285)

## Oracle text
Indestructible
When The One Ring enters, if you cast it, you gain protection from everything until your next turn.
(You can't be targeted, dealt damage, enchanted, or equipped by anything.)
At the beginning of your upkeep, you lose 1 life for each burden counter on The One Ring.
{T}: Put a burden counter on The One Ring, then draw a card for each burden counter on The One Ring.

(Legendary Artifact, mana cost {4}.)

## Forge script
- Source: fetched script — `bin/resources/cardsfolder/t/the_one_ring.txt` (not edited).
- Key tags:
  - `K:Indestructible` — the static keyword.
  - `T:Mode$ ChangesZone | ValidCard$ Card.wasCastByYou+Self | Destination$ Battlefield |
    Execute$ TrigPump` — the ETB trigger, gated by the `wasCastByYou` cast-condition ("if you cast
    it").
  - `SVar:TrigPump:DB$ Pump | Defined$ You | Duration$ UntilYourNextTurn |
    KW$ Protection from everything` — grant the controller protection from everything until their
    next turn.
  - `T:Mode$ Phase | Phase$ Upkeep | ValidPlayer$ You | Execute$ TrigLoseLife` with
    `SVar:TrigLoseLife:DB$ LoseLife | LifeAmount$ X` — the upkeep drain.
  - `A:AB$ PutCounter | Cost$ T | Defined$ Self | CounterType$ BURDEN | CounterNum$ 1 |
    SubAbility$ DBDraw` with `SVar:DBDraw:DB$ Draw | Defined$ You | NumCards$ X` — the tap ability.
  - `SVar:X:Count$CardCounters.BURDEN` — the dynamic count shared by the drain and the draw.
- Tags parsed as written; no category was retagged.

## Engine work (all general, keyed on each tag's intended meaning)

### 1. Indestructible (CR 702.12)
- Already supported. `is_indestructible` (`src/game_queries.h`) reads `K:Indestructible` from a
  non-creature permanent's `CardData::keywords`, and `effects::destroy` / `destroy_all` honor it.
  Verified, not rebuilt.

### 2. "Enters, if you cast it" cast-condition (CR 614.12; `wasCastByYou`)
- New `Permanent::entered_by_cast` flag (`src/components/permanent.h`), set one-shot from the
  existing `cur_game.cast_to_battlefield` marker when the `Permanent` is created
  (`src/systems/state_manager_statics.cpp`). True only for a permanent that resolved onto the
  battlefield as a cast spell; false for tokens / reanimation / ChangeZone-to-battlefield / etc.
- Parse: a `ValidCard$ ...wasCastByYou...` qualifier sets a new
  `Ability::trigger_requires_entered_by_cast`; the `+Self`/`.Self` qualifier (here a trailing
  token, not the head) now also sets `valid_card_self` (`src/parse.cpp`).
- Trigger match: a self ETB trigger with `trigger_requires_entered_by_cast` only fires when its
  source permanent's `entered_by_cast` is true (`src/systems/state_manager_triggers.cpp`, next to
  the existing `evoked` / `entered_with_offspring` self-ETB gates). General over any "enters, if
  you cast it" trigger.

### 3. Player "protection from everything" with an UntilYourNextTurn duration (CR 702.16)
- New player-scoped state `Game::PlayerProtectionFromEverything { player; until_your_next_turn }`
  +  `cur_game.player_protection_from_everything` (`src/classes/game.h`) — mirrors Veil of Summer's
  player-scoped `hexproof_from_colors_this_turn`, but covers **everything**, not just named colors.
- Parse: `KW$ Protection from everything` on a `DB$ Pump` is pulled out of the keyword list into a
  new `PumpParams::grant_protection_from_everything` flag (so it is a player grant, not a per-
  creature keyword) — `src/effects/effect_pump.cpp parse_pump`. A generic
  `Duration$ UntilYourNextTurn` on a non-Animate effect now sets `Ability::duration_until_your_next_turn`
  (`src/parse.cpp`).
- Resolve: `effects::pump` (`src/effects/effect_pump.cpp`), when the flag is set, registers the
  grant for `ab.controller` (Defined$ You) with the parsed duration and returns — like the existing
  hexproof-from-colors short-circuit.
- **Targeting protection** (CR 702.16e): `player_has_protection_from_everything` (static in
  `src/components/ability.cpp`), checked up front in `Ability::is_legal_target` next to the Veil
  hexproof check — the protected player can't be the target of an **opponent's** spell/ability.
- **Damage protection** (CR 702.16d): shared predicate `player_protected_from_source(player, source)`
  (`src/game_queries.{h,cpp}`) — true when the source is controlled by an opponent of the protected
  player. Enforced at:
  - `deal_damage_to_player` (`src/components/damage.cpp`) — the effect-damage chokepoint (covers
    direct-damage spells/abilities).
  - the combat-damage path (`src/systems/state_manager_combat.cpp`) — both the unblocked and the
    trample-over-to-player branches, which subtract life directly rather than through
    `deal_damage_to_player`. Prevented combat damage gains no lifelink.
- **Duration / revert.** `until_your_next_turn` grants are reverted at the protected player's untap
  step (`src/classes/game.cpp` UNTAP, beside Karn's `revert_until_turn_animates`); a non-
  `until_your_next_turn` (end-of-turn) grant lapses at cleanup. General: any future "protection from
  everything (until your next turn / until end of turn)" player grant reuses this.

#### Scope decision — protection from damage
Implemented in full for the two damage paths a two-player game produces against a player: effect
damage (`deal_damage_to_player`) and combat damage (the two direct-life-subtraction sites). The
"can't be enchanted / equipped" clause of CR 702.16 is not separately modeled because the engine
has no Aura/Equipment that an opponent can attach to a player, and targeted attachment is already
blocked by the targeting protection above. No partial/second-step deferral was needed.

### 4. `Count$CardCounters.<TYPE>` dynamic amount (CR 122)
- New branch in `evaluate_dynamic_amount` (`src/components/ability.cpp`): counts the `<TYPE>`
  counters on the ability's **source** permanent. The function gained an optional `Entity source`
  parameter (defaulted, so existing callers are unchanged); `effect_lose_life` and `effect_draw`
  pass `ab.source`. `effect_draw` now evaluates `dynamic_amount_expr` (it previously read only the
  static `amount`).
- Parse: `Count$CardCounters` is added to the SVar→`dynamic_amount_expr` recognition lists in both
  the sub-ability (`parse_svar_ability`) and top-level (`parse_abilities`) paths — `src/parse.cpp`.
- Drives both the upkeep `LoseLife` (lose N = burden counters) and the tap `Draw` (draw N = burden
  counters, counted **after** the `PutCounter` sub-ability resolves).

## Tests (test harness, scripted + `--play`)
- **Tap ability scales.** Activate `{T}` → "Put 1 BURDEN counter (now 1)", draw 1; next turn →
  "(now 2)", draw 2.
- **Upkeep drain = burden count.** With 1 counter, each of A's upkeeps loses exactly 1 life
  (19 → 18 → 17). In a full scripted game the drain scales 1 → 2 → 3 as counters accrue.
- **ETB protection (cast condition).** Casting The One Ring logs "Player A gains protection from
  everything until their next turn"; a preset (non-cast) One Ring does **not** (the `wasCastByYou`
  gate). While protected, an opponent's Lightning Bolt offers only Player B as a target (Player A
  excluded), and Grizzly Bears' combat damage is prevented ("2 combat damage prevented", A stays at
  20). The protection **expires** at A's next turn: in the scripted game, turn-2 combat is prevented
  but turn-4 combat ("Grizzly Bears deals 2 damage to Player A") lands.
- **Indestructible.** Abrade's "Destroy target artifact" mode targeting The One Ring resolves with
  "The One Ring is indestructible — not destroyed"; it stays on the battlefield.
- No game-result draws, no non-fatal errors.

## Reusable mechanics added
- `Permanent::entered_by_cast` + `Ability::trigger_requires_entered_by_cast` — general "enters, if
  you cast it" (`wasCastByYou`) ETB cast-condition.
- `cur_game.player_protection_from_everything` + `player_protected_from_source` +
  `player_has_protection_from_everything` — a player-scoped protection-from-everything state with an
  UntilYourNextTurn (or EOT) duration, consulted by targeting and all player-damage paths.
- `Ability::duration_until_your_next_turn` — generic UntilYourNextTurn duration on a non-Animate
  effect.
- `evaluate_dynamic_amount` `Count$CardCounters.<TYPE>` (counters on the source) + an optional
  `source` argument; `effect_draw` now honors `dynamic_amount_expr`.
