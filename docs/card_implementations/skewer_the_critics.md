# Skewer the Critics

## Oracle text
Spectacle {R} (You may cast this spell for its spectacle cost rather than its mana cost if an
opponent lost life this turn.)
Skewer the Critics deals 3 damage to any target.

(`{2}{R}` sorcery)

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/s/skewer_the_critics.txt`).
- **Key tags:**
  - `K:Spectacle:R`
  - `A:SP$ DealDamage | ValidTgts$ Any | NumDmg$ 3`

## Engine work
Mechanic added (general): **spectacle** (a conditional alternative casting cost) plus its
supporting per-player **life-lost-this-turn** tracker. The 3-damage effect (`SP$ DealDamage`)
was already fully supported.

1. `src/components/player.h:16` — added `int32_t life_lost_this_turn` sibling to
   `life_gained_this_turn` (CR 702.107a needs "an opponent lost life this turn").
2. `src/game_queries.h:719` — added `player_lose_life(entity, amount)`, the mirror of
   `player_gain_life`: lowers `life_total` and accumulates `life_lost_this_turn`. Single
   source so the per-turn counter cannot drift from `life_total`.
3. Routed the life-LOSS sites through the helper (CR 120.3 — damage to a player is a loss of
   that much life):
   - `src/components/damage.cpp:69` — noncombat `deal_damage_to_player`.
   - `src/systems/state_manager_combat.cpp:109,190` — unblocked combat damage and trampled-over
     combat damage to a player.
   - `src/effects/effect_lose_life.cpp:42` — the `LoseLife` effect (also added the
     `game_queries.h` include).
4. `src/classes/game.cpp:449` — reset `life_lost_this_turn` for both players each cleanup,
   next to the existing `life_gained_this_turn` reset.
5. `src/components/carddata.h:37` — added `bool is_spectacle` to `AltCost`.
6. `src/parse.cpp:673` — parse `K:Spectacle:<cost>` into `card.alt_cost` (mana portion via the
   shared `parse_alt_cost_tokens`, `is_spectacle = true`), mirroring the Impending/Evoke keyword
   alt-cost parses.
7. `src/systems/state_manager_actions.cpp:82` — in `can_afford_alt`, gate an `is_spectacle`
   alt cost on the caster's opponent having `life_lost_this_turn > 0` (two-player game, sole
   opponent). The existing alt-cost mana affordability + offering (`use_alt_cost` LegalAction)
   and payment (`pay_alternate_cost`) are reused unchanged; the alt-cast description now reads
   "(spectacle)".

CR 702.107 (Spectacle), CR 702.107a (offered only if an opponent lost life this turn),
CR 118.9 / 601.2f (alternative cost substituted for the mana cost).

Mechanics added (general): **spectacle** — any card with `K:Spectacle:<cost>` may be cast for
that alternative cost when an opponent lost life this turn. Reusable Spectacle infra for later
cards: `AltCost::is_spectacle`, the `K:Spectacle:<cost>` parse, the `can_afford_alt` gate, and
the shared `Player::life_lost_this_turn` / `player_lose_life()` tracker.

## Behavioral decisions
- **Life-lost tracking (mirror of life_gained):** `player_lose_life` is fed by damage to a
  player (combat + noncombat, CR 120.3) and by explicit "lose life" effects. Life PAID as a
  cost (Phyrexian mana, life-cost activations, alt-cost life) is deliberately NOT routed
  through it — Spectacle tracks life LOSS from damage/effects, matching the `life_gained`
  mirror which likewise tracks only explicit gains. (A burn deck — the natural Spectacle
  home — enables Spectacle via damage, which is covered.)
- **Two-player scope:** the opponent is the single other seat, per the engine's 1-v-1 scope.
- **Offering:** both the normal `{2}{R}` cast and the `(spectacle)` `{R}` cast are offered as
  separate legal actions whenever both are affordable; Spectacle is simply omitted from the
  menu when no opponent has lost life this turn.

## Tests (isolation)
- (a) No opponent life lost this turn — A casts Skewer with 3 Mountains in play → only
  **"Cast Skewer the Critics"** offered; NO "(spectacle)" variant. PASS.
- (b) A bolts Player B first (B 20 → 17, so B lost life), then casts Skewer → both
  **"Cast Skewer the Critics"** and **"Cast Skewer the Critics (spectacle)"** offered; chose
  spectacle; only **one Mountain tapped for {R}** (not {2}{R}); Skewer dealt 3 (B 17 → 14).
  PASS.
- CI gate: `ci_check.py --tier pygen,vocab,smoke` — see batch report.

## Result
Implemented.
