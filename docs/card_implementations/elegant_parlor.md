# Elegant Parlor  (vocab index 118)

## Oracle text
({T}: Add {R} or {W}.)
Elegant Parlor enters tapped.
When Elegant Parlor enters, surveil 1. (Look at the top card of your library. You may put it into your graveyard.)

## Forge script
- Source: fetched (Forge@master) → `bin/resources/cardsfolder/e/elegant_parlor.txt`
- `Types:Land Mountain Plains` — a nonbasic dual land with the Mountain and Plains land
  subtypes; the {R}/{W} mana abilities are injected from the subtypes, not scripted.
- Key tags:
  - `R:Event$ Moved | ValidCard$ Card.Self | Destination$ Battlefield | ReplacementResult$ Updated | ReplaceWith$ ETBTapped`
    with `SVar:ETBTapped:DB$ Tap | Defined$ Self | ETB$ True` — enters-tapped replacement.
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigSurveil`
    with `SVar:TrigSurveil:DB$ Surveil | Amount$ 1` — ETB "surveil 1" trigger.
- This is the Murders at Karlov Manor "surveil dual land" cycle. It is structurally identical
  to the already-implemented **Undercity Sewers** (index 63, Island Swamp → {U}/{B}) and
  **Thundering Falls** (index 25, Island Mountain → {U}/{R}); Elegant Parlor differs only in
  its land subtypes (Mountain Plains → {R}/{W}).

## Engine work
- None — fully covered by existing handlers:
  - Basic-land-subtype mana abilities are injected by `StateManager::apply_land_abilities`
    from the `Mountain` and `Plains` subtypes (`{R}` and `{W}`), exactly as for the other
    cycle members and for dual lands like Plateau (Mountain Plains).
  - The enters-tapped replacement effect (`R:Event$ Moved ... ReplaceWith$ ETBTapped`) is
    handled by the replacement-effects subsystem (`src/systems/replacement_effects.cpp`),
    which applies the `DB$ Tap | ETB$ True` sub-ability as part of the enter-battlefield event.
  - The ETB `DB$ Surveil | Amount$ 1` trigger resolves through the existing Surveil effect
    (`src/effects/effect_surveil.cpp`), prompting keep-on-top / put-in-graveyard.
- Mechanics added (general, not card-specific): none.

## Behavioral decisions (made in lieu of asking a human)
- **Enters tapped** (CR 603.6d): "enters tapped" is a static enter-the-battlefield ability that
  occurs as part of the event putting the permanent onto the battlefield. Forge models it as a
  `Moved`→`Battlefield` replacement that taps the card, and the engine applies it during the
  ETB event so the land is tapped the moment it arrives (verified: it never has a window untapped
  on the turn it is played). Because it is not a triggered ability, it does not use the stack.
- **ETB surveil trigger** (CR 603.6a / 701.25): "When Elegant Parlor enters, surveil 1" is a
  separate triggered ability that goes on the stack. Surveil 1 = look at the top card of the
  library and choose to keep it on top or put it into the graveyard (CR 701.25a). Both branches
  are exposed and were exercised in isolation.
- **Two mana colors**: the {T}: Add {R} or {W} ability is the standard land-subtype mana
  ability, so the engine offers an {R} source and a {W} source from the single permanent. This
  was confirmed indirectly but conclusively: with two untapped Elegant Parlors and an empty mana
  pool as the only mana sources, the engine reported **both** "Cast Lightning Bolt" (needs {R})
  and "Cast Swords to Plowshares" (needs {W}) as legal casts.

## Tests
- Isolation (test_harness, inline hands):
  - **Enters tapped + surveil (keep-on-top)**: played Elegant Parlor from hand →
    "Elegant Parlor enters tapped." then "Resolving ability (category: Surveil, amount: 1)",
    "Top card of Player A's library: …", with the surveil menu offering
    {Keep on top, Put in graveyard}. PASS.
  - **Surveil (put-in-graveyard branch)**: chose "Put in graveyard" → "Player A puts <card>
    into the graveyard." and the top card moved to the graveyard. PASS.
  - **Both mana colors**: two untapped Elegant Parlors as the sole mana sources, empty pool —
    engine listed both "Cast Lightning Bolt" ({R}) and "Cast Swords to Plowshares" ({W}) as
    legal, proving the land taps for each of its two colors. PASS.
- Regression (test_harness `--scripted`, 6 seeds, R/W deck with 4 Elegant Parlors vs a
  Forest/Mountain creature deck): all 6 games decisive (Player B won all 6 — a deck/agent
  matchup artifact, not a correctness issue), no draws, no fatal or non-fatal errors. In a
  sampled game Elegant Parlor was observed entering tapped and resolving Surveil 1 three times
  with zero warnings. PASS.

## Result
implemented
