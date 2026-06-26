# Preordain (vocab index 152)

## Oracle text
Scry 2, then draw a card. (To scry 2, look at the top two cards of your library, then put any number of them on the bottom of your library and the rest on top in any order.)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/p/preordain.txt`)

Key tags:
- `ManaCost:U`, `Types:Sorcery`
- `A:SP$ Scry | ScryNum$ 2 | SubAbility$ DBDraw`
- `SVar:DBDraw:DB$ Draw`

## Engine work
None — covered. `EffectKind::Scry` handler (`src/effects/effect_scry.cpp`) implements
CR 701.18 (look at top N, keep on top or put on bottom per card). `SubAbility$ DBDraw`
chains a `DB$ Draw` after the scry resolves. This is the first vocab card to use
`SP$ Scry` directly, hence the test below.

## Behavioral decisions
None. Scry handler omits the optional reorder-among-kept cards (documented simplification
in effect_scry.cpp); not relevant for Preordain's typical use.

## Tests
- Cast Preordain (test_harness, inline hand/library): "Player A casts Preordain" →
  "Resolving ability (category: Scry, amount: 2)" → "Player A scries 2." → per-card
  top/bottom choice offered for both top cards → after scry, "Player A draws Island".
  Confirms scry 2 then draw 1, with Preordain moving to graveyard. PASS.

## Result
Done — registered in vocab, clean build, scry+draw verified.
