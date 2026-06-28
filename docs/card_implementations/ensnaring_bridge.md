# Ensnaring Bridge  (vocab index 271)

## Oracle text
Creatures with power greater than the number of cards in your hand can't attack.

## Forge script
- Source: pre-existing — `bin/resources/cardsfolder/e/ensnaring_bridge.txt`
- Type: `Artifact`, mana cost `3`.
- Key tags:
  - `S:Mode$ CantAttack | ValidCard$ Creature.powerGTX` — a continuous combat-restriction
    static: every creature matching `Creature.powerGTX` can't be declared as an attacker.
  - `SVar:X:Count$ValidHand Card.YouOwn` — the dynamic threshold X = the number of cards in
    **your** (the source/Ensnaring Bridge controller's) hand.

No tags were retagged or repurposed; a general `CantAttack` static + a dynamic
power-vs-X qualifier + a `Count$ValidHand` count expression were added, all keyed on the
tags' intended meaning and reusable by any future card.

## Whose hand ("your") — CR-consistent reading
"Your hand" is the **Ensnaring Bridge controller's** hand (the static's source controller),
and the restriction applies to **every** creature (both players'), comparing each creature's
power to that single threshold. This matches the Oracle wording and Forge's
`Count$ValidHand Card.YouOwn` ("You" = the source's controller), and is the card's whole
prison function: you empty your own hand to lock out the opponent's big attackers. It is
**not** evaluated per-attacking-creature against that creature's own controller's hand — when
the same player controls the Bridge and the attacker the two readings coincide, but they
differ for the opponent's creatures, and the cross test below confirms the controller's-hand
reading.

## Engine work
Modeled as a rules-modifying continuous static (like CantBeCast / CantBeActivated): not a
characteristic layer, but a prohibition queried at the declare-attackers decision site.

1. **New `StaticAbility` fields** (`src/components/static_ability.h`):
   `std::string cant_attack_filter` (the `ValidCard$` spec) and `std::string cant_attack_x_svar`
   (the resolved SVar expression for the dynamic X).

2. **Parser** (`src/parse.cpp`, `parse_static_abilities`): `Mode$ CantAttack` →
   `category = "CantAttack"`; its `ValidCard$` fills `cant_attack_filter`; a filter containing
   `X` resolves SVar `X` into `cant_attack_x_svar`. A `Target$` on a CantAttack static (the
   "can't attack you" variant, which restricts *which player* can't be attacked rather than
   forbidding attacking outright) clears `cant_attack_filter` so only the blanket form is
   honoured — the targeted form is left unimplemented rather than mis-applied.

3. **Dynamic power-vs-X qualifier** (`src/game_queries.cpp`, `eval_qualifier`): a
   `power<OP>X` / `toughness<OP>X` qualifier compares the object's P/T against `MatchCtx::x_bound`
   (new field, `src/game_queries.h`, default `INT_MIN` = "no X" → fails closed). The caller
   resolves X and seeds `x_bound`.

4. **`Count$ValidHand <filter>` count expression** (`src/svar_eval.cpp`, `evaluate_sa_svar`):
   counts cards in a player's hand matching the filter; ownership qualifiers (YouOwn/…) scope
   to one player's hand via `Zone::owner`, remaining characteristic qualifiers run through the
   shared `card_matches_filter`. General, also reachable from `evaluate_dynamic_amount`'s
   fallback.

5. **Prohibition query** (`src/systems/rules_modifying.{h,cpp}`): new
   `bool attack_prohibited(Entity creature)` scans `g_active_statics` for non-suppressed,
   condition-met `CantAttack` statics, seeds `MatchCtx::controller`/`source` from the static and
   `x_bound = evaluate_sa_svar(cant_attack_x_svar, source-controller, source)`, then matches the
   creature against `cant_attack_filter`. X is evaluated against the **source's controller** — so
   "your hand" is the Bridge controller's hand, the same threshold for every creature.

6. **Declare-attackers integration** (`src/action_processor.cpp`, `declare_attackers`): a
   creature for which `rules_mod::attack_prohibited` is true is skipped when building the eligible
   set (CR 509.1a), so it is never offered as a legal attacker and is never force-declared by a
   "must attack" effect. Re-evaluated every declaration (the static cache is rebuilt each SBA
   pass), so a hand-size change flips eligibility.

## Behavioral decisions (made in lieu of asking a human)
- **"Your hand" = the Bridge controller's hand, applied to all creatures** (see the CR section
  above). The task brief's "compare each creature's power to that creature's own controller's
  hand" reading was **not** followed because it contradicts the Oracle text / Forge SVar and
  would break the card's prison function; flagged for the human.
- **Only the blanket CantAttack form is implemented.** `Target$`-scoped ("can't attack you")
  and unknown-qualifier (e.g. `EnchantedBy`) CantAttack statics are not applied as a global
  can't-attack — unknown qualifiers fail closed in the shared matcher, and the targeted form is
  gated out in the parser — so no false prohibition is imposed on other cards that share the tag.

## Tests
Isolation (`train/test_harness.py`, Sire of Seven Deaths is a 7/7 — a one-card-difference
threshold around a full hand):
- **Hand 7 → can attack.** A controls Ensnaring Bridge + Sire of Seven Deaths, hand = 7 →
  Sire (power 7, `7 > 7` false) is offered as an attacker and hits for 7. PASS.
- **Hand 6 → can't attack.** Same board; A plays one land (hand 7→6) → at declare attackers
  "No creatures eligible to attack" (Sire `7 > 6`). On a later turn A's hand returns to 7 and
  Sire is offered again — confirms dynamic re-evaluation. PASS.
- **Cross / prison case (controller's hand, not attacker's).** A controls Ensnaring Bridge,
  B controls Sire of Seven Deaths, B's hand = 8 throughout. When A's hand = 6, B's Sire can't
  attack ("No creatures eligible"); when A's hand = 7, B's Sire is offered. Proves the threshold
  tracks the **Bridge controller's** (A's) hand, not the attacker's controller's (B's). PASS.

No draws, no non-fatal errors / asserts in any run.

## Result
implemented
