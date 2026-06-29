# Expedition Map  (vocab index 246)

## Oracle text
{2}, {T}, Sacrifice Expedition Map: Search your library for a land card, reveal it, put it
into your hand, then shuffle.

(Artifact, mana cost {1}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/e/expedition_map.txt`
- Key tags:
  - `A:AB$ ChangeZone | Cost$ 2 T Sac<1/CARDNAME> | Origin$ Library | Destination$ Hand |
    ChangeType$ Land | ChangeNum$ 1` — activated library tutor for ANY land, to hand.

## Engine work
- none — fully covered by existing handlers:
  - `ChangeType$ Land` library search (any land, not a specific subtype): `search_zone` in
    `src/components/ability.cpp:95-155` (matches `t.name == "Land"`).
  - `Sac<1/CARDNAME>` self-sacrifice cost: parsed at `src/parse.cpp:236-256` (sets
    `ability.sac_self`), paid at activation `src/action_processor.cpp:172-176`.
  - `{2}` + `{T}` cost and reveal-to-hand + shuffle handled by the generic ChangeZone resolution.

## Behavioral decisions
- none — behavior unambiguous. Proving cards for the pattern: Elvish Reclaimer / Knight of the
  Reliquary (`ChangeType$ Land` search), Scalding Tarn (`Sac<1/CARDNAME>` cost).

## Tests
- Isolation (test_harness): cast Expedition Map ({1}); next turn activated it ({2} from two
  Islands, tap, sacrifice). The library search menu correctly offered every land remaining in
  library (Wastes/Mountains/etc.); chose **Wastes** (a typeless basic) → "Player A reveals Wastes
  and puts it to hand", library shuffled, Wastes in hand, Expedition Map in graveyard. PASS. Also
  confirmed: when all non-Mountain lands were already in hand, the menu correctly listed only the
  Mountains still in library (the `Land` filter matches any land, gated by library contents).
- Regression (test_harness --scripted, full games): RG deck with 4× Expedition Map vs a green/red
  creature deck, seeds 1-2 — both decisive (1 A win, 1 B win), Expedition Map activated and
  fetched in real games ("reveals Mountain and puts it to hand"), no draws, no non-fatal errors.

## Result
implemented
