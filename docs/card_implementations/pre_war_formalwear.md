# Pre-War Formalwear

```
Name:Pre-War Formalwear
ManaCost:2 W
Types:Artifact Equipment
T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigChange | TriggerDescription$ When CARDNAME enters, return target creature card with mana value 3 or less from your graveyard to the battlefield and attach CARDNAME to it.
SVar:TrigChange:DB$ ChangeZone | Origin$ Graveyard | Destination$ Battlefield | TgtPrompt$ Choose target creature card with mana value 3 or less in your graveyard | ValidTgts$ Creature.cmcLE3+YouOwn | RememberChanged$ True | SubAbility$ DBAttach
SVar:DBAttach:DB$ Attach | Defined$ Remembered | SubAbility$ DBCleanup
SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True
S:Mode$ Continuous | Affected$ Creature.EquippedBy | AddPower$ 2 | AddToughness$ 2 | AddKeyword$ Vigilance | Description$ Equipped creature gets +2/+2 and has vigilance.
K:Equip:3
Oracle:When Pre-War Formalwear enters, return target creature card with mana value 3 or less from your graveyard to the battlefield and attach Pre-War Formalwear to it.\nEquipped creature gets +2/+2 and has vigilance.\nEquip {3}
```

**Oracle:** When Pre-War Formalwear enters, return target creature card with mana value 3 or
less from your graveyard to the battlefield and attach Pre-War Formalwear to it. Equipped
creature gets +2/+2 and has vigilance. Equip {3}.

Vocab index: **230** (`src/card_vocab.h`).

Forge script source: pre-existing local script
`bin/resources/cardsfolder/p/pre_war_formalwear.txt` (unchanged).

## Engine work

The card's individual clauses are covered by existing handlers:

- **ETB trigger** — `T: Mode$ ChangesZone … Destination$ Battlefield … Execute$ TrigChange`,
  the standard self-ETB trigger (`src/systems/state_manager_triggers.cpp`).
- **Targeted reanimation** — `DB$ ChangeZone | Origin$ Graveyard | Destination$ Battlefield |
  ValidTgts$ Creature.cmcLE3+YouOwn | RememberChanged$ True`, handled by
  `effects::change_zone` (`src/effects/effect_change_zone.cpp`); the `cmcLE3+YouOwn` target
  filter restricts to a mana-value-≤3 creature in your own graveyard, and `RememberChanged$`
  stashes the returned creature.
- **Attach** — `DB$ Attach | Defined$ Remembered` resolves through `effects::attach`
  (`src/effects/effect_attach.cpp`), reading the remembered creature.
- **EquippedBy +2/+2 & Vigilance static** — `S: Affected$ Creature.EquippedBy | AddPower$ 2 |
  AddToughness$ 2 | AddKeyword$ Vigilance`, applied by the existing EquippedBy static-layer
  appliers (`src/systems/state_manager_statics.cpp` / layers).
- **Equip {3}** — the `K:Equip:3` keyword activated ability.

### One ordering fix (general, not a new mechanic)

The reanimate-then-attach *combination in a single resolution* exposed an ordering gap. When
`DB$ ChangeZone` puts the creature onto the battlefield, the creature's **`Permanent` component
is not created synchronously** — it is added on the next state-based pass by
`StateManager::apply_permanent_components`. The chained `DB$ Attach` resolves *immediately
after* the move (still inside the same ability resolution), so the target creature was on the
battlefield (`Zone`) but had **no `Permanent` yet**, and the attach silently no-opped (so the
+2/+2 and vigilance never applied).

Fix (mirrors the existing `pending_enters_tapped` / `pending_evoked` one-shot pattern):

- `Game::pending_attach` — a `std::map<Entity,Entity>` ({creature → equipment}) added in
  `src/classes/game.h`.
- `effects::attach` (`src/effects/effect_attach.cpp`): if the target creature is on the
  battlefield but has no `Permanent` yet, record `pending_attach[creature] = equipment` and
  defer (instead of dropping the attach).
- `StateManager::apply_permanent_components` (`src/systems/state_manager_statics.cpp`): when it
  creates the creature's `Permanent`, it consumes `pending_attach` and finalizes the
  `equipped_to`/`equipped_by` link, after which the EquippedBy static applies normally.

This is plumbing that makes the **existing** Attach handler robust for "attach to a creature
that is entering in the same resolution"; it is not a new effect category. The normal
(both-permanents-already-exist) attach path and the token-attach path (Cori-Steel Cutter) are
unchanged.

## Behavioral decisions

- The returned creature enters under the controller's control (the player reanimating from their
  own graveyard), per the existing ChangeZone reanimation rules (CR 608.2).
- The attach happens whether the equipment entered by being cast or by any other means; the
  +2/+2 and vigilance come solely from the EquippedBy static, so they end the moment the
  equipment leaves or is moved.

## Tests (`train/test_harness.py`)

- **Reanimate + attach + buff (ETB):** sacrifice Birds of Paradise to Goblin Bombardment
  (→ graveyard), then cast Pre-War Formalwear targeting Birds. Result: "Birds of Paradise is
  moved to the battlefield" → "Equipment attached." → "Birds of Paradise gains 2/2 Vigilance" →
  Birds shows **[2/3]** with vigilance.
- **Manual Equip {3}:** with Formalwear and Birds already in play, activating Equip and choosing
  Birds attaches it and applies +2/+2 + vigilance (Birds → [2/3]). Confirms the Equip keyword and
  static work independent of the ETB chain.
- **Real-game regression:** `temp/form_a` (mav list with 3 Pre-War Formalwear) vs `temp/form_b`
  (delver), scripted seeds 1/2/3 → B/A/B winners, no errors, no draws. Formalwear casts in real
  games and the full chain fires (e.g. seed 5 reanimates Noble Hierarch, seed 6 reanimates Dryad
  Arbor, each with "Equipment attached.").
- **Cross-regression (engine change safety):** delver/mav/doomsday round-robin (seeds 1/2) all
  clean — the shared Attach handler and EquippedBy statics still behave for existing equipment
  (Cori-Steel Cutter, Stoneforge package).

## Result

Implemented. Clauses covered by existing handlers; one general ordering fix (`pending_attach`)
lets the existing Attach handler work when the target enters in the same resolution. Behavior
verified end-to-end.
