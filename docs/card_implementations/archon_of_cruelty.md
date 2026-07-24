# Archon of Cruelty

## Oracle text
Flying

Whenever Archon of Cruelty enters or attacks, target opponent sacrifices a creature or
planeswalker, discards a card, and loses 3 life. You draw a card and gain 3 life.

6/6 — {6}{B}{B} Creature — Archon

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/a/archon_of_cruelty.txt`).
- **Key tags:**
  - `T: Mode$ ChangesZone ... Destination$ Battlefield` (ETB) and `T: Mode$ Attacks` both `Execute$ TrigSac`.
  - `TrigSac: DB$ Sacrifice | ValidTgts$ Opponent | SacValid$ Creature,Planeswalker`
    → `DBDiscard: DB$ Discard | Defined$ Targeted | Mode$ TgtChoose`
    → `DBLoseLife: DB$ LoseLife | Defined$ Targeted | LifeAmount$ 3`
    → `DBDraw: DB$ Draw` → `DBGainLife: DB$ GainLife | LifeAmount$ 3`.

## Engine work
The card exercised a targeted-player edict plus a sub-ability chain that mixes
target-relative and controller-relative players. Four small, GENERAL fixes:

1. **Targeted-player sacrifice** (`src/effects/effect_sacrifice.cpp`, CR 701.16a). The edict path
   was keyed only on `Defined$ Opponent` (`defined_each_opponent`). A `ValidTgts$ <Player>`
   targeted sacrifice fell through to the final branch and made the CONTROLLER sacrifice. Added a
   branch: when `ab.target` is a Player entity, that targeted player is the sacrificer (they
   choose which of their permanents to sacrifice), mandatory like an edict. `ab.target` stays set
   so the downstream `Defined$ Targeted` discard/lose-life still resolve against that same player.
   *Mechanics added (general): targeted-player DB$ Sacrifice.*

2. **Comma-OR in valid filters** (`src/game_queries.cpp` `match_filter_core`, CR 701.16). Forge
   writes `SacValid$ Creature,Planeswalker` with a comma, but the shared filter split alternatives
   only on `;`, so a comma-joined list matched nothing (nothing was sacrificed). `;` and `,` are
   now both top-level OR separators (additive — no qualifier token contains a comma).
   *Mechanics added (general): comma as an OR separator in Valid/Sac filters.*

3. **Sub-ability Draw defaults to controller** (`src/effects/effect_draw.cpp`, CR 608.2c). Sub-ability
   chaining copies the parent's target into `ab.target`; the Draw handler drew for any player
   target, so `DBDraw` ("You draw a card") drew for the sacrifice target instead of the caster.
   Now the target is used only when the Draw itself declared it (`ValidTgts$`, or `Defined$
   Targeted/ParentTarget/Parent`) — mirroring the identical guard already in `effect_lose_life.cpp`.
   *Mechanics added (general): no-`Defined$` DB$ Draw draws for the controller.*

4. **`Mode$ TgtChoose` discard** (`src/effects/effect_discard.cpp`, CR 701.8). Previously
   unrecognized, so it defaulted to revealing the target's hand and letting the CONTROLLER choose.
   `TgtChoose` now has the TARGET player choose one of their own cards to discard, with no hand
   reveal (the DiscardValid$ filter and pick loop are shared with the default branch).
   *Mechanics added (general): `Mode$ TgtChoose` (affected player chooses their own discard).*

## Behavioral decisions
- Two-player scope: "target opponent" is the single opponent (CR 102.1). The sacrifice is
  mandatory for the targeted player when they control a matching permanent.
- The five sub-abilities resolve as one chain: opponent sacrifices → opponent discards (own
  choice) → opponent loses 3 → controller draws → controller gains 3.

## Tests
- Isolation (test_harness `--play` with seat keys): preset Archon on A, Grizzly Bears + Birds of
  Paradise on B, a 2-card hand for B; attacked with Archon.
  - Result: B chose and sacrificed Grizzly Bears (menu seated on B, both creatures offered); B
    chose and discarded a card (no hand reveal to A); B lost 3 (20→17); **A** drew a card; **A**
    gained 3 (20→23). Confirmed the opponent sacrifices/discards/loses, and the controller
    draws/gains.
- CI gate: `ci_check.py --tier pygen,vocab,smoke` (run once after all three cards).

## Result: implemented.
