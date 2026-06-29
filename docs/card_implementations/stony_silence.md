# Stony Silence  (vocab index 249)

## Oracle text
Activated abilities of artifacts can't be activated.

(Enchantment, mana cost {1}{W}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/s/stony_silence.txt`
- Key tags:
  - `S:Mode$ CantBeActivated | AffectedZone$ Battlefield | ValidCard$ Artifact | ValidSA$ Activated`
    — symmetric static: no artifact's activated abilities (including mana abilities) may be activated.

## Engine work
- none — fully covered by existing handlers. This static line is byte-for-byte identical to
  **Null Rod** (`bin/resources/cardsfolder/n/null_rod.txt:4`, vocab idx 71, pre-existing) and
  Collector Ouphe (idx 31):
  - Parsed at `src/parse.cpp:2400-2406` → `cant_activate_card_filter = "Artifact"`.
  - Non-mana activated abilities suppressed via `rules_mod::activation_prohibited`
    (`src/systems/rules_modifying.cpp:40-55`), queried at `src/systems/state_manager_actions.cpp:731`.
  - Mana abilities of artifacts also suppressed via `rules_mod::mana_activation_prohibited`
    (`src/mana_system.cpp:251`) — a plain type-filter static (not a NamedCard one) blocks mana
    abilities too, matching Forge's `ValidSA$ Activated` (no `!ManaAbility` exclusion).
  - `ValidSA$ Activated` is cosmetic here (behavior fully implied by the `ValidCard$` type filter).

## Behavioral decisions
- Symmetric (affects both players' artifacts). The `CantBeActivated` static is not controller-scoped,
  matching the Oracle (CR 605/602: an "activated abilities … can't be activated" prohibition applies
  to all such abilities regardless of controller).

## Tests
- Isolation (test_harness): A controls Expedition Map (an artifact with a {2}{T}{Sac} activated
  ability); B controls Stony Silence. A's First Main menu offered only Pass/Play Island — "Activate
  Expedition Map" was **absent**. Control without Stony Silence: the same board offered "Activate
  Expedition Map (ChangeZone)" at index 7. PASS.
- Regression (test_harness --scripted, full games): WX Stony Silence deck vs an artifact deck
  (Expedition Map, Candelabra of Tawnos), seeds 1-2 — both decisive (2 B wins), no draws, no
  non-fatal errors.

## Result
implemented
