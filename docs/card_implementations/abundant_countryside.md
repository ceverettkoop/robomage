# Abundant Countryside  (vocab index 120)

## Oracle text
{T}: Add {C}.
{T}: Add one mana of any color. Spend this mana only to cast a creature spell.
{6}, {T}: Create a 1/1 colorless Shapeshifter creature token with changeling. (It's every creature type.)

## Forge script
- Source: fetched (Forge@master) → `bin/resources/cardsfolder/a/abundant_countryside.txt`
- `Types:Land` — a nonbasic colorless land with three activated abilities (no land subtypes,
  so its mana abilities are scripted, not injected from subtypes).
- Key tags:
  - `A:AB$ Mana | Cost$ T | Produced$ C` — {T}: Add {C}.
  - `A:AB$ Mana | Cost$ T | Produced$ Any | RestrictValid$ Spell.Creature` — {T}: add one mana
    of any color, usable only to cast a creature spell.
  - `A:AB$ Token | Cost$ 6 T | TokenScript$ c_1_1_shapeshifter_changeling | TokenOwner$ You` —
    {6}, {T}: create the Shapeshifter token.
  - `DeckHas:Ability$Mana.Colorless|Token` — cosmetic deck-hint metadata (ignored).

## Engine work
- **`RestrictValid$ Spell.Creature` (any creature spell, no subtype)** — *new general handler.*
  The parser previously only recognized the Cavern-of-Souls form
  (`RestrictValid$ Spell.Creature+ChosenType`, flag `restrict_to_chosen_type_creature`). Added a
  sibling flag `restrict_to_creature` on `Ability` (`src/components/ability.h`, equality in
  `src/components/ability.cpp`), set in `src/parse.cpp` when the value mentions `Creature` but
  not `ChosenType`. Mana from such a source is now filtered to creature-spell payments via a new
  helper `creature_restricted_mana_matches(paid_for)` in `src/mana_system.cpp`, wired into the
  same three filter sites as the Cavern check (action listing, affordability pruning, payment
  source selection). When not paying for a spell (`paid_for == 0`) the source is hidden, exactly
  like the existing restricted-mana behavior, so the restricted mana is never floated freely.
- **Token script resource** `bin/resources/tokenscripts/c_1_1_shapeshifter_changeling.txt` —
  added (canonical Forge content) so `parse_token_script` can build the 1/1 Shapeshifter with
  `K:Changeling`. This is an engine resource referenced by the script, not a Forge card script.
- The `{C}` mana ability and the `{6}, {T}` Token ability use existing handlers
  (`parse_activation_cost` for the `6 T` cost, `effects::token` for resolution).
- Mechanics added (general, not card-specific): the `restrict_to_creature` mana-spending
  restriction (any "spend only on a creature spell" mana, not only Abundant Countryside).

## Behavioral decisions (made in lieu of asking a human)
- **Creature-only mana** (CR 106.7): mana with a spending restriction can only pay for spells the
  restriction allows; here, only creature spells. Verified both directions — the any-color
  ability produced {G} to cast Grizzly Bears (a creature), and with Abundant Countryside the
  *only* source, Lightning Bolt (noncreature, {R}) was correctly **not** offered as a legal cast.
- **`Produced$ Any`**: handled by the existing color-choice mechanism (Birds of Paradise path) —
  the source expands to one selectable entry per color {W/U/B/R/G}; the `restrict_to_creature`
  flag is preserved on each color copy, so every color is creature-restricted.
- **Changeling** (CR 702.73): the token is every creature type. Stored as a `K:Changeling`
  keyword on the token; the engine has no behavioral consumer of changeling yet (it matters only
  for tribal interactions not present in scope), so it is parsed and stored without effect. No
  triggered ability is generated (`keyword_triggered_ability` returns a no-op), so it is inert
  and error-free.
- **`TokenOwner$ You`**: the token's controller defaults to the activating player, which the
  engine already does; the tag emits the pre-existing cosmetic
  `WARNING: Unrecognized ability param: TokenOwner$ You` (shared with Cori-Steel Cutter /
  Ocelot Pride) and is ignorable.

## Tests
- Isolation (test_harness, inline hands / preset battlefield):
  - **Token ability**: 6 Mountains + Abundant Countryside in play → "Activate Abundant
    Countryside (Token)" paid {6} and tapped the land → "Token created: 1/1 Shapeshifter Token";
    the token entered, was summoning-sick, then attacked next turn. PASS.
  - **Creature-restricted mana (positive)**: Abundant Countryside + Mountain in play, cast
    Grizzly Bears (1G) → "activated Abundant Countryside for 1(G)" paid the {G}, Mountain paid
    the generic, Grizzly Bears entered. PASS.
  - **{C} + creature-{G} together**: two Abundant Countrysides as sole sources cast Grizzly
    Bears — one tapped for 1(C) (generic), the other for 1(G) (creature-restricted). PASS.
  - **Creature-restricted mana (negative)**: Abundant Countryside as the only source, Lightning
    Bolt ({R}) in hand — Lightning Bolt was **not** a legal cast (only Pass / Play Mountain),
    proving the any-color mana cannot pay for a noncreature spell. PASS.
- Regression (test_harness `--scripted`, 6 seeds, deck with 6 Abundant Countryside + Forest /
  Mountain / Grizzly Bears / Lightning Bolt, mirror match): all 6 games decisive, no draws, no
  fatal or non-fatal errors (only the allowed cosmetic `TokenOwner$ You` warning). Mana
  activations and the Shapeshifter token were observed firing in real games. PASS.

## Result
implemented
