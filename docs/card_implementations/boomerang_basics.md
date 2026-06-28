# Boomerang Basics  (vocab index 252)

## Oracle text
Return target nonland permanent to its owner's hand. If you controlled that permanent, draw a card.

(Sorcery — Lesson, mana cost {U}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/b/boomerang_basics.txt`
- Key tags:
  - `A:SP$ ChangeZone | Origin$ Battlefield | Destination$ Hand | Defined$ Targeted |
    ValidTgts$ Permanent.nonLand | RememberLKI$ True | SubAbility$ DBDraw` — bounce target
    nonland permanent, remembering its last-known information.
  - `SVar:DBDraw:DB$ Draw | ConditionDefined$ RememberedLKI | ConditionPresent$ Card.YouCtrl |
    NumCards$ 1 | SubAbility$ DBCleanup` — draw 1 **only if** the remembered permanent was under
    your control as it left the battlefield.
  - `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True`.

## Engine work
- Targeted bounce (`SP$ ChangeZone` Battlefield→Hand) and the condition-gated `DB$ Draw` sub-ability
  chain were already covered (`src/effects/effect_change_zone.cpp`, `effect_draw.cpp`,
  `evaluate_present_condition` in `src/systems/state_manager_actions.cpp`).
- **New, general:** `RememberLKI$` on a ChangeZone, and a `ConditionDefined$ RememberedLKI` gate that
  resolves a `YouCtrl`/`OppCtrl` qualifier from **last-known information**:
  - `Ability::remember_lki` flag (`src/components/ability.h`), parsed from `RememberLKI$ True`
    (`src/parse.cpp`); the targeted ChangeZone pushes the moved entity into
    `cur_game.remembered_entities` when `remember_lki` is set (`src/effects/effect_change_zone.cpp`).
  - `ConditionDefined$ RememberedLKI` now sets `condition_on_remembered` (`src/parse.cpp`), reusing
    the remembered-set condition path.
  - In `evaluate_present_condition` (`src/systems/state_manager_actions.cpp`), a remembered card now
    in a **non-battlefield** zone has its `YouCtrl`/`OppCtrl` qualifier resolved from
    `cur_game.last_known_info[e].controller` (CR 608.2g — the controller it had as it left play),
    instead of the previous behaviour where `card_matches_filter` treats an off-battlefield
    controller qualifier as a no-op (which would have made the draw fire unconditionally).
  - `cur_game.last_known_info` is already captured for **every** permanent leaving the battlefield
    (`src/systems/orderer.cpp`), so no new capture site was needed.

## Behavioral decisions
- The draw is gated on **control** (`ConditionPresent$ Card.YouCtrl`), matching the Oracle "If you
  controlled that permanent". It is *not* gated on the permanent being a creature.
- LKI controller is read at resolution of the draw, after the bounce already moved the card to its
  owner's hand — the correct reading per CR 608.2g.

## Tests (test_harness)
- **Bounce own creature:** A bounces its own Grizzly Bears → "Grizzly Bears is moved to hand",
  then "Resolving ability (category: Draw)" → "Player A draws Mountain". Draw fired. PASS.
- **Bounce opponent's creature:** A bounces B's Grizzly Bears → moved to B's hand; the Draw ability
  resolves but **no card is drawn** (LKI controller = B ≠ caster). PASS.
- Regression (`--scripted`, seeds 1-3, deck containing Boomerang Basics): all decisive, no draws,
  no non-fatal errors.

## Result
implemented
