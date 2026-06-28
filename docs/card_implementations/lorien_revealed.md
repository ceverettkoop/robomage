# Lórien Revealed  (vocab index 273)

## Oracle text
Draw three cards.

Islandcycling {1} ({1}, Discard this card: Search your library for an Island card, reveal it,
put it into your hand, then shuffle.)

(Sorcery, mana cost {3}{U}{U} per the local Forge script.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/l/lorien_revealed.txt`
- Key tags:
  - `A:SP$ Draw | NumCards$ 3` — main spell, draw three (already covered by `effect_draw`).
  - `K:TypeCycling:Island:1` — typecycling keyword (CR 702.29f). The NEW mechanic.

## Name / accent note
The display name "Lórien Revealed" contains a non-ASCII "ó". The engine's `name_to_uid`
strips non-ASCII → `lrien_revealed`, but Forge's script file is `lorien_revealed.txt`. So the
card can only be loaded by its ASCII spelling ("Lorien Revealed" → uid `lorien_revealed`), and
`load_card` then keys it under the script's accented `Name`. The vocab entry uses the exact
accented name (`{"Lórien Revealed", 273}`) so it matches the `CardData::name` the engine reads.
Decks/tests must reference the card by its ASCII spelling.

## Engine work
- **General `TypeCycling:<Subtype>:<cost>` parser** — `src/parse.cpp` (the new keyword branch,
  just before the `Flashback:` branch). Parses the subtype and cost portions and builds a
  hand-activated ability mirroring Cycling but with a subtype-filtered Library→Hand search
  instead of a draw:
  - `ability_type = ACTIVATED`, `activation_zone = Zone::HAND`
  - `category = "ChangeZone"`, `origin = LIBRARY`, `destination = HAND`, `change_type = <Subtype>`
  - `mandatory = false` (searches may fail to find, CR 701.19c)
  - mana portion parsed via the shared `parse_activation_cost` token grammar.
  - General over the subtype: `Islandcycling`/`Swampcycling`/`Plainscycling`/... all supported;
    the named subtype is honored, never hardcoded.
- Reuses existing infrastructure (no new resolution code):
  - Hand-activation path: `process_activate_ability` in `src/action_processor.cpp:222-268` —
    pays the mana, auto-consumes the source card to the graveyard (the "Discard this card"
    cost, lines 246-251), then pushes the `ChangeZone` ability on the stack.
  - Legal-action offering of hand abilities (any category): `src/systems/state_manager_actions.cpp:917-945`.
  - Subtype-filtered library search + reveal + shuffle: `effects::change_zone` /
    `search_zone` (`src/effects/effect_change_zone.cpp:377-427`, `src/components/ability.cpp:101`).
    `search_zone` matches the `change_type` against the card's subtypes, so "Island" offers only
    Island cards; the search auto-shuffles the library (origin LIBRARY) and `search_reveals_card`
    reveals the chosen card (library + specific type).
- **`gen_card_costs.py`** — added a general NFKD-transliteration fallback in `find_card_file`
  so accented vocab names (the C++ `name_to_uid` strips the accent to `lrien_revealed`, while the
  file is `lorien_revealed.txt`) resolve to their on-disk ASCII stem. Reusable for any accented
  card name; without it the cast-cost row for this card was all zeros.

## Behavioral decisions
- none — behavior unambiguous. Typecycling = cycling that tutors a card of the named subtype to
  hand instead of drawing (CR 702.29f). The discard-this-card cost is the standard hand-ability
  auto-consume; the search may fail to find.

## Tests (test_harness)
- **Main spell**: cast Lórien Revealed with {3}{U}{U} available → "Resolving ability
  (category: Draw, amount: 3)", drew three cards, Lórien Revealed to graveyard. PASS.
- **Islandcycling**: with {1} available (one Island in play) and Lórien Revealed in hand,
  activated its typecycling ability. The {1} cost was paid, Lórien Revealed went to the
  graveyard (discard cost), then the library search menu offered only `Fail to find` and the
  single `Island` — the **Mountains in library were correctly NOT offered** (subtype filter).
  Chose Island → "Player A shuffles their library", "Player A reveals Island and puts it to
  hand"; Island in hand, Lórien Revealed in graveyard. PASS.

## Result
implemented
