# Pick Your Poison  (vocab index 257)

## Oracle text
Choose one —
• Each opponent sacrifices an artifact.
• Each opponent sacrifices an enchantment.
• Each opponent sacrifices a creature with flying.

(Sorcery, mana cost {G}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/p/pick_your_poison.txt`
- Key tags:
  - `A:SP$ Charm | Choices$ SacArtifact,SacEnchantment,SacFlier`.
  - `SVar:SacArtifact:DB$ Sacrifice | Defined$ Opponent | SacValid$ Artifact | Amount$ 1`.
  - `SVar:SacEnchantment:DB$ Sacrifice | Defined$ Opponent | SacValid$ Enchantment | Amount$ 1`.
  - `SVar:SacFlier:DB$ Sacrifice | Defined$ Opponent | SacValid$ Creature.withFlying | Amount$ 1`.

## Engine work
- `SP$ Charm` modal + `DB$ Sacrifice | Defined$ Opponent` edict were already covered
  (`src/effects/effect_charm.cpp`, `src/effects/effect_sacrifice.cpp`).
- **New, general:** the `with<Keyword>` filter qualifier. `SacValid$ Creature.withFlying` previously
  failed closed (the lowercase `withFlying` token was unrecognized and matched nothing), so the
  flier mode sacrificed nothing. Added a `with<Keyword>` branch to `eval_qualifier`
  (`src/game_queries.cpp`): it strips the `with` prefix, re-inserts a space before each interior
  capital ("FirstStrike" → "First Strike"), and tests `permanent_has_keyword(entity, kw)` (the
  effective keyword list for a battlefield permanent, printed keywords for an off-battlefield card
  view). General — any `Creature.withFlying`/`.withTrample`/etc. filter now works.
- `SacMessage$` is a cosmetic prose param, added to the parser's ignored set (`src/parse.cpp`).

## Behavioral decisions
- Each mode is an edict: the **opponent** (controller of the permanent) chooses which of their
  matching permanents to sacrifice (CR 701.16a), already the edict path in effect_sacrifice.

## Tests (test_harness)
- **Flying mode:** B has Baleful Strix (flyer) + Grizzly Bears (non-flyer). A casts Pick Your Poison,
  chooses the flying mode → only "Sacrifice Baleful Strix" offered; Baleful Strix ended in B's
  graveyard, Grizzly Bears survived. PASS.
- **Artifact mode:** B has Voltaic Key + Grizzly Bears → only "Sacrifice Voltaic Key" offered. PASS.
- **Enchantment mode:** B has Stony Silence + Grizzly Bears → only "Sacrifice Stony Silence"
  offered. PASS.
- Regression (`--scripted`, seeds 1-3): all decisive, no draws, no non-fatal errors.

## Result
implemented
