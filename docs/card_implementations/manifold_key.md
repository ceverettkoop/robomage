# Manifold Key  (vocab index 251)

## Oracle text
{1}, {T}: Untap another target artifact.
{3}, {T}: Target creature can't be blocked this turn.

(Artifact, mana cost {1}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/m/manifold_key.txt`
- Key tags:
  - `A:AB$ Untap | Cost$ 1 T | ValidTgts$ Artifact.Other` — untap ANOTHER target artifact.
  - `A:AB$ Effect | Cost$ 3 T | ValidTgts$ Creature | RememberObjects$ Targeted |
    StaticAbilities$ Unblockable` with `SVar:Unblockable:Mode$ CantBlockBy | ValidAttacker$
    Card.IsRemembered` — grant the target creature unblockable until end of turn.

## Engine work
- none — fully covered by existing handlers:
  - Targeted `AB$ Untap`: `src/effects/effect_untap.cpp` (proven by Voltaic Key / Candelabra).
  - `.Other` target restriction (can't target itself): enforced in `Ability::is_legal_target`
    (`src/components/ability.cpp`), proven by Solitude / Flickerwisp.
  - `AB$ Effect` + `StaticAbilities$ Unblockable` / `Mode$ CantBlockBy`: `effects::grant_cast`
    (`src/effects/effect_grant_cast.cpp`) sets `Creature::cant_be_blocked_this_turn` on the
    targeted creature (`RememberObjects$ Targeted` → `ab.target`), read by the blocker-legality
    check (`src/action_processor.cpp`) and cleared at cleanup. Proven by Kappa Cannoneer (idx 136).
- The `ExileOnMoved$ Battlefield` param is unrecognized (cosmetic warning) — it governs when the
  temporary Effect entity is torn down; the engine instead keys the grant to the
  `cant_be_blocked_this_turn` until-end-of-turn flag, which is the correct duration for this card.

## Behavioral decisions
- The unblockable grant lasts until end of turn (CR 509/702: "can't be blocked this turn"),
  matching the `cant_be_blocked_this_turn` cleanup. Unambiguous.

## Tests
- Isolation (test_harness):
  - **Untap another artifact:** A controls Manifold Key + Grim Monolith; cast Grafdigger's Cage
    tapping Grim Monolith, then Manifold Key's untap ability — the target menu offered **only Grim
    Monolith** (Manifold Key itself excluded by `.Other`); "Grim Monolith untaps". PASS.
  - **Unblockable:** A's Grizzly Bears + Manifold Key vs B's Grizzly Bears. Activated the {3}{T}
    Effect ability targeting A's Grizzly Bears → "Grizzly Bears can't be blocked this turn." A
    attacked; "No creatures eligible to block"; "Grizzly Bears deals 2 damage to Player B" — B's
    blocker could not block the unblockable attacker. PASS.
- Regression (test_harness --scripted, full games): GR Manifold Key deck vs a creature deck, seeds
  1-2 — both decisive (2 A wins), no draws, no non-fatal errors (only the cosmetic
  `ExileOnMoved$ Battlefield` unrecognized-param line).

## Result
implemented
