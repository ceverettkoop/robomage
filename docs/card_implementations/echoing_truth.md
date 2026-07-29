# Echoing Truth

## Oracle text
Return target nonland permanent and all other permanents with the same name as that permanent to their owners' hands. ({1}{U} instant)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/e/echoing_truth.txt`).

Key tags:
- `A:SP$ Pump | ValidTgts$ Permanent.nonLand | SubAbility$ DBChangeZoneAll` — a no-op
  target-acquiring shell (no `NumAtt$`/`NumDef$` amounts) that just chooses the target nonland
  permanent at cast (CR 601.2c).
- `SVar:DBChangeZoneAll:DB$ ChangeZoneAll | Origin$ Battlefield | Destination$ Hand | ChangeType$ TargetedCard.Self,Permanent.NotDefinedTargeted+sharesNameWith Targeted`
  — the actual bounce.

## Engine work
- **Mechanics added (general): `samename-mass-bounce-target`** — a battlefield -> hand mover
  that returns the chosen TARGET permanent plus every other battlefield permanent sharing its
  name, each to its OWNER's hand.
- `src/effects/effect_change_zone_all.cpp:20` — `change_zone_all` gains a branch keyed on
  `change_type.find("sharesNameWith") && origin == BATTLEFIELD`, dispatching to the new static
  `change_zone_shares_name_battlefield` (same file). It scans the whole battlefield (both
  players) for permanents whose name matches the target's and moves each to its destination.
  Because `Zone::owner` is fixed (CR 108.3 / 400.3), a `Destination$ Hand` routes each permanent
  to its own owner's hand, correctly handling copies split across both players.
- The target itself is included with no special case: it is a battlefield permanent whose name
  trivially matches (covers `TargetedCard.Self`).
- A small `permanent_name(Entity)` helper reads the name off `CardData` (real cards) or `Token`
  (token copies, which have no `CardData`), so token copies still match.
- The `SP$ Pump` shell with no amounts degrades to a harmless no-op: `effect_pump.cpp`'s
  `apply_pump_to_creature` early-returns when `pump_att == 0 && pump_def == 0` and no keywords
  are granted, and the pre-chosen target skips the target-selection block. No change needed.

CR: 400.3 (a permanent's owner is fixed), 108.3 (ownership), 601.2c (target chosen at cast).

## Behavioral decisions
- Existing `change_zone_same_name` (Surgical Extraction family) keys on `Remembered.sameName` /
  `Targeted.sameName` over hidden zones (graveyard/library/hand) for ONE searched player and
  moves to a single destination; it does not fit a battlefield-wide, owner-routed bounce that
  includes the target. Adding a dedicated battlefield branch keeps that path untouched and is
  reusable for any "return X and all same-named permanents" card.

## Tests (isolation)
Harness, `--battlefield-a "Island,Island,Grizzly Bears"`, `--battlefield-b "Grizzly Bears,Runeclaw Bear"`:
- Target one Grizzly Bears -> BOTH Grizzly Bears return to their owners' hands (A's to A, B's
  to B), Runeclaw Bear stays. Target itself included. PASS.
- Target the unique Runeclaw Bear -> only Runeclaw Bear returns. PASS.
- CI gate: `make check` tiers pygen/vocab/smoke.

## Result
Implemented.
