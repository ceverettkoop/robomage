# Aether Spellbomb (vocab index 154)

## Oracle text
{U}, Sacrifice Aether Spellbomb: Return target creature to its owner's hand.
{1}, Sacrifice Aether Spellbomb: Draw a card.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/a/aether_spellbomb.txt`)

Key tags:
- `Types:Artifact`, `ManaCost:1`
- `A:AB$ ChangeZone | Cost$ U Sac<1/CARDNAME> | ValidTgts$ Creature | Origin$ Battlefield | Destination$ Hand` — bounce a creature.
- `A:AB$ Draw | Cost$ 1 Sac<1/CARDNAME>` — draw a card.

## Engine work
None — covered:
- `Sac<1/CARDNAME>` activation cost (sacrifice self) parsed in `src/parse.cpp`.
- `AB$ ChangeZone` Battlefield→Hand (bounce) handled by effect_change_zone, used by
  other bounce cards (e.g. Brazen Borrower / Petty Theft).
- `AB$ Draw` handled by effect_draw.

## Behavioral decisions
None — both abilities are standard activated abilities with a mana + sacrifice-self cost.

## Tests
Both modes verified via test_harness (Aether Spellbomb pre-set on battlefield with Islands):
- Draw mode: `activate:Aether Spellbomb (Draw)` → "Player A activated Island for 1" →
  "Player A sacrifices Aether Spellbomb" → "Resolving ability (category: Draw)" →
  "Player A draws Island"; Spellbomb to graveyard. PASS.
- Bounce mode: `activate:Aether Spellbomb (ChangeZone)` targeting opponent's Birds of
  Paradise → "Player A sacrifices Aether Spellbomb" → "Resolving ability (category:
  ChangeZone)" → "Birds of Paradise is moved to hand" (opp hand 7→8, Birds back in B's
  hand); Spellbomb to graveyard. PASS.

## Result
Done — registered in vocab, clean build, both activated modes verified.
