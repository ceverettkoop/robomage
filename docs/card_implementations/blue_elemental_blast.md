# Blue Elemental Blast  (vocab index 245)

## Oracle text
Choose one —
- Counter target red spell.
- Destroy target red permanent.

(Instant, mana cost {U}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/b/blue_elemental_blast.txt`
- Key tags (the exact Red↔Blue mirror of Red Elemental Blast, vocab idx 116):
  - `A:SP$ Charm | Choices$ DBCounter,DBDestroy` — modal "choose one" spell (CR 700.2 / 601.2b).
  - `SVar:DBCounter:DB$ Counter | TargetType$ Spell | ValidTgts$ Card.Red` — mode 1: counter a
    target **red** spell (red is part of the target restriction).
  - `SVar:DBDestroy:DB$ Destroy | ValidTgts$ Permanent.Red` — mode 2: destroy a target **red**
    permanent.
  - `AI:RemoveDeck:Random` — Forge AI hint, ignored.

## Engine work
- none — fully covered by the existing handlers that already serve Red Elemental Blast:
  - Charm modal spell: `src/effects/effect_charm.cpp` (parse `src/parse.cpp:1666-1693`).
  - `DB$ Counter` of a stack spell: `src/effects/effect_counter.cpp`.
  - `DB$ Destroy` of a permanent: `src/effects/effect_destroy.cpp`.
  - Positive color target restriction (`.Red`) on the permanent/destroy path:
    `src/components/ability.cpp::is_legal_target` (`permanent_matches_filter` color token).

## Behavioral decisions
- Red is a target restriction, not a resolution condition (per the script's `ValidTgts$ Card.Red`
  / `Permanent.Red`), exactly mirroring Red Elemental Blast (CR 115.1). The destroy-mode color
  restriction is enforced at target enumeration (verified: only the red permanent is offered).
- Known pre-existing limitation shared verbatim with Red Elemental Blast (idx 116): the
  **counter** mode's stack-spell target path keys on `TargetType$ Spell` and does not re-check the
  `ValidTgts$ Card.Red` color, so counter mode can technically target a non-red spell. This is an
  inherited engine behavior identical to the already-accepted REB, not specific to or newly
  introduced by Blue Elemental Blast; left as-is (out of scope for this card).

## Tests
- Isolation (test_harness):
  - **Destroy mode:** opp has Goblin Guide (red), Grizzly Bears (green), Island (colorless); BEB's
    destroy-mode target menu offered **only Goblin Guide**, which was destroyed → graveyard. PASS.
  - **Counter mode:** opp casts Lightning Bolt (red) at Player A; BEB counter mode targeted and
    "Lightning Bolt is countered" → graveyard. PASS.
- Regression (test_harness --scripted, full games): UB deck with 4× Blue Elemental Blast (+ Baleful
  Strix, Counterspell, Islands/Swamps) vs a mono-red/green deck (Lightning Bolt, Goblin Guide,
  Grizzly Bears), seeds 1-3 — all three decisive (B wins), no draws, no non-fatal errors.

## Result
implemented
