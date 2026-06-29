# Hexing Squelcher  (vocab index 278)

## Oracle text
This spell can't be countered.
Ward—Pay 2 life.
Spells you control can't be countered.
Other creatures you control have "Ward—Pay 2 life."

(2/2 Goblin Sorcerer, mana cost `1 R`.)

## Forge script
- Source: `bin/resources/cardsfolder/h/hexing_squelcher.txt`
- Type: `Creature Goblin Sorcerer`, `PT:2/2`.
- Key tags:
  - `R:Event$ Counter | ValidCard$ Card.Self | ValidSA$ Spell | Layer$ CantHappen` — "This spell
    can't be countered" (the SELF form). **Already supported** before this card (Long Goodbye /
    Cavern of Souls pattern): the cast path stamps `Spell::cant_be_countered`.
  - `R:Event$ Counter | ValidSA$ Spell.YouCtrl | Layer$ CantHappen | ActiveZones$ Battlefield` —
    **"Spells you control can't be countered"** — the NEW mechanic: a continuous, battlefield-active
    can't-be-countered replacement (CR 614.13 / "CantHappen") scoped by a `ValidSA$` filter rather
    than the source spell itself.
  - `K:Ward:PayLife<2>` — Ward (already supported).
  - `S:Mode$ Continuous | Affected$ Creature.Other+YouCtrl | AddKeyword$ Ward:PayLife<2>` — grants
    Ward to other creatures you control (existing keyword-granting static).

No tags were retagged or repurposed.

## Engine work

### General battlefield "can't be countered" replacement (the new mechanic)
The self form ("This spell can't be countered") was already a cast-time stamp; the battlefield
continuous form ("Spells you control can't be countered") is consulted at counter-resolution time.

- **`Effect::Replacement` fields** (`src/components/effect.h`): added `from_battlefield` (the
  `ActiveZones$ Battlefield` continuous form) and `valid_sa_filter` (the `ValidSA$` spec, e.g.
  `Spell.YouCtrl`) to the existing `CANT_BE_COUNTERED` replacement kind.
- **Parser** (`src/parse.cpp`, replacement-effect block): captures `ValidSA$` into
  `valid_sa_filter`, and when an `R:Event$ Counter | Layer$ CantHappen | ActiveZones$ Battlefield`
  line has a `ValidSA$` filter (and is not the `Card.Self` self form), builds a `CANT_BE_COUNTERED`
  replacement with `from_battlefield = true`. The self form is unchanged.
- **Cast-time stamp gated** (`src/action_processor.cpp`): the cast-time loop that sets
  `Spell::cant_be_countered` from a card's own replacements now only fires for the SELF form
  (`!from_battlefield`), so the battlefield static never wrongly stamps the source's own spell.
- **Reusable query** `spell_uncounterable_by_static(spell, entities)`
  (`src/game_queries.h`): scans live battlefield permanents for a `from_battlefield`
  `CANT_BE_COUNTERED` replacement and tests the spell against its `ValidSA$` filter. The controller
  scope (`YouCtrl`/`OppCtrl`) is read from the spell's caster (`Spell::caster`) relative to the
  replacement source's controller — necessary because a stack spell has no `Permanent` controller,
  so the generic filter matcher's `YouCtrl` token is a no-op off the battlefield. Any remaining
  type/characteristic qualifiers are checked via `card_matches_filter`. Reusable by any future
  can't-be-countered permanent.
- **Counter-resolution hook** (`src/effects/effect_counter.cpp`): the existing "can't be countered"
  guard now also consults `spell_uncounterable_by_static(...)`. When it matches, the countering
  ability still resolves but does nothing to the spell (CR 701.5g) — the spell stays on the stack
  and resolves normally; the counterspell goes to its graveyard.

## Behavioral decisions (made in lieu of asking a human)
- **Check at counter-resolution, not cast time.** "Spells you control can't be countered" is a
  continuous effect, so a spell's counterability is evaluated when the counter would apply — this
  correctly handles Hexing Squelcher entering/leaving after the protected spell is cast.
- **The counterspell still resolves.** Per CR 701.5g a counter ability that can't counter its
  target simply does nothing; the counterspell is not "fizzled" and moves to the graveyard as
  normal. Mirrors the existing Cavern/Pyroblast handling in `effect_counter.cpp`.
- **Controller scope from the spell's caster.** `Spell.YouCtrl` protects only spells controlled by
  Hexing Squelcher's controller; an opponent's spell remains counterable while you control it.

## Tests
Isolation (`train/test_harness.py`), seed 1:
- **Protected (the card works).** Hexing Squelcher + 2 Forest on A's battlefield; A casts Grizzly
  Bears, B casts Counterspell targeting it → `Resolving ability (category: Counter)` →
  `Grizzly Bears can't be countered` → `Grizzly Bears enters the battlefield`; Counterspell to B's
  graveyard. PASS.
- **Control (no Hexing Squelcher).** Same line without Hexing Squelcher → `Grizzly Bears is
  countered` → Grizzly Bears to A's graveyard. PASS — proves the static, not a constant.

Counter source used: **Counterspell** (vocab index 22), the only hard counter in the vocab.

No draws, no non-fatal errors / asserts in either run.

## Result
implemented
