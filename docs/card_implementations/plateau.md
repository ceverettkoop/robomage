# Plateau  (vocab index 114)

## Oracle text
({T}: Add {R} or {W}.)

(Plateau is an original dual land — it has no rules text of its own; its mana ability
is derived entirely from its basic land types.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/p/plateau.txt`
- Contents:
  - `Name:Plateau`
  - `ManaCost:no cost`
  - `Types:Land Mountain Plains`
  - `Oracle:({T}: Add {R} or {W}.)`
- The script carries **no** `A:` / `T:` / `S:` ability lines. The mana ability is implicit
  in the basic land subtypes `Mountain` and `Plains`, exactly like the other original dual
  lands already in the vocab (Savannah `Forest Plains`, Volcanic Island `Island Mountain`,
  Underground Sea `Island Swamp`).

## Engine work
- None — fully covered by existing handlers. Plateau is mechanically identical to the
  already-implemented original dual lands and relies on the shared basic-land-subtype mana
  injection.
  - `StateManager::apply_land_abilities` (`src/systems/state_manager_statics.cpp`) scans a
    land permanent's subtypes, and for each basic land subtype injects a `{T}: Add <color>`
    activated `AddMana` ability via `mana_color_for_subtype`. That mapping already covers
    `Mountain → RED` and `Plains → WHITE`, so a `Land Mountain Plains` permanent automatically
    gains both a "{T}: Add {R}" and a "{T}: Add {W}" mana ability (each tap-cost, amount 1,
    `subtype_derived = true`).
  - These are independent abilities sharing one tap cost (CR 605/606 mana abilities); the
    player chooses which color when tapping. No on-stack interaction — mana abilities resolve
    immediately at activation.
- Mechanics added (general, not card-specific): none.

## Behavioral decisions (made in lieu of asking a human)
- Plateau enters the battlefield **untapped** — it has no "enters tapped" replacement (it is
  an original ABU dual land), matching the other duals. Handled by the shared ETB →
  Battlefield path; verified untapped in isolation.
- The two color abilities are derived from the subtypes, not printed in the script. This is
  the intended design (the parser honors `Types:Land Mountain Plains`; no tag was retagged or
  shortcut). The subtype-derived injection is the same code path that powers every basic land
  and every dual/triple land in the vocab.
- No card-specific ambiguity exists — behavior is unambiguous and identical to existing duals.

## Tests
- Isolation (test_harness):
  - Plateau pre-placed on the battlefield enters **untapped** and is available the same turn.
  - With one Plateau in play, casting Lightning Bolt: `Player A activated Plateau for 1(R)` →
    Bolt cast, Player B 20→17. Confirms the {R} ability. PASS.
  - With two Plateaus in play, casting Containment Priest (cost `1 W`):
    `Player A activated Plateau for 1(W)` + `Player A activated Plateau for 1(R)` → Priest cast.
    Confirms the {W} ability (one Plateau tapped for white to satisfy the colored pip). PASS.
- Regression (test_harness `--scripted`, 6 games, seeds 1–6): deck `temp/plateau_a`
  (4 Plateau + Lightning Bolt + Containment Priest + Mountain/Plains/Island) vs `temp/plateau_b`
  (4 Plateau + Lightning Bolt + Mountain/Forest). All 6 games finished with a decisive winner
  (A wins all six seeds in this matchup), no draws. Plateau was played and tapped for both
  colors by the scripted agent. No non-fatal errors / asserts / tracebacks.

## Result
implemented
