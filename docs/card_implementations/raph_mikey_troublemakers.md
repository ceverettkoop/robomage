# Raph & Mikey, Troublemakers

**Vocab index:** 352

## Oracle text

```
Trample, haste
Whenever Raph & Mikey attack, reveal cards from the top of your library until you
reveal a creature card. Put that card onto the battlefield tapped and attacking and
the rest on the bottom of your library in a random order.
```

`{5}{R/G}{R/G}` — Legendary Creature — Mutant Ninja Turtle, 7/7.

## Forge script

Source: pre-existing local **official Forge** script, relocated (not authored/modified) from
`bin/resources/cardsfolder/upcoming/raph_mikey_troublemakers.txt` to
`bin/resources/cardsfolder/r/raph_mikey_troublemakers.txt` so the on-demand loader
(`load_card`, `cardsfolder/<first-letter>/<uid>.txt`) can find it. The `upcoming/` copy is left
in place.

Key tags:

- `K:Trample`, `K:Haste` — already supported keywords.
- `T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigDigUntil` — the attack trigger (supported,
  as on Goblin Guide).
- `SVar:TrigDigUntil:DB$ DigUntil | Valid$ Creature | FoundDestination$ Battlefield | Tapped$ True |
  Attacking$ True | RevealedDestination$ Library | RevealedLibraryPosition$ -1 | RevealRandomOrder$ True`

## Engine work

### `name_to_uid` underscore-collapse fix (general)

The engine's `name_to_uid` (`src/parse.cpp`) turned `"Raph & Mikey, Troublemakers"` into
`raph__mikey_troublemakers` (the `"& "` becomes `space->_`, `&` excised, `space->_`, leaving a
doubled `__`), but Forge names the file with a **single** underscore
(`raph_mikey_troublemakers.txt`). Fixed by collapsing consecutive underscores to one at the end of
`name_to_uid`, aligning the engine with Forge's filename convention. Verified safe: no shipped
Forge script has a `__` in its filename (`find bin/resources/cardsfolder -name "*__*"` is empty),
so collapsing cannot shadow another card's load. Mirrored the same collapse in the Python
`tools/forge_fetch/fetch_script.py` `name_to_uid` (script fetching) and in
`train/gen_card_costs.py` `find_card_file` (cast-cost codegen must resolve the same script, else
the cost row is all-zero).

### DigUntil onto the battlefield, tapped & attacking, random-bottom (general)

`src/effects/effect_dig_until.cpp` previously only handled the Amped Raptor case (found + revealed
both to Exile, impulse-style). Extended it to be general over destinations and the entering flags:

- `Tapped$ True` — reuses the existing `ab.enters_tapped` flag (claimed by `parse_change_zone`).
- `Attacking$ True` — new `ab.dig_until_attacking` flag. When the found creature is put onto the
  battlefield (`FoundDestination$ Battlefield`), it enters tapped and attacking the same defender
  the source (Raph) is attacking. **Reuses the ninjutsu enters-attacking machinery** — the
  `pending_enters_tapped` / `pending_enters_attacking` one-shots that `apply_permanent_components`
  consumes once the card's `Creature` component exists (the same path used by K:Ninjutsu and by
  Mobilize's tapped-and-attacking tokens). CR 508.4: the creature is *put onto the battlefield
  attacking*, not declared, so no new "attacks" triggers fire.
- `RevealedDestination$ Library` + `RevealedLibraryPosition$ -1` + `RevealRandomOrder$ True` — the
  revealed non-matching cards go to the **bottom** of the library (`-1`, reuses
  `ab.dig_library_position`) in **random order** (reuses `ab.rest_random_order` + the seeded,
  platform-stable `stable_shuffle` from `stable_rng.h`, as in `effect_change_zone_all.cpp`).

The reveal now snapshots the library top-first (`get_library_top`) and walks it (CR 701.16 reveal,
CR 401 library ordering), holding the passed-over cards until placement. The Amped Raptor path
(found + revealed to Exile, one-at-a-time order) is preserved: revealed cards are placed first,
in top order, then the found card, so exile ordering is unchanged.

### Shared log genericized

`state_manager_statics.cpp`'s enters-attacking log ("X is attacking (ninjutsu)") is now shared by
Raph, so the "(ninjutsu)" qualifier was dropped ("X is attacking.") — cosmetic, narrative-only.

## Behavioral decisions

- The found creature enters under the **digging player's** control (CR 608.2 — from their own
  library).
- If the library empties before a creature is revealed, the dig stops gracefully (nothing enters);
  the revealed cards still go to the bottom.

## Mechanics added (general)

- `diguntil-onto-battlefield-attacking`
- `name_to_uid` underscore-collapse

## Tests

Isolation (test harness, `--deck-a temp/raph_test_a --no-shuffle --battlefield-a "Raph Mikey
Troublemakers"`, library stacked Island, Island, Grizzly Bears): Raph attacks → trigger reveals
Island, Island (bottomed), finds Grizzly Bears → Grizzly Bears enters **tapped and attacking**,
deals 2 combat damage to the defender that same combat (alongside Raph's 7). Next turn's attack
confirms the two Islands are now at the bottom of the library. Card loads (name resolves via the
collapsed uid) and the temp deck referencing `1 Raph & Mikey, Troublemakers` loads with no fatal
error. Trample/Haste confirmed (attacked turn 1).

CI gate: `ci_check.py --tier pygen,vocab,smoke` — the `vocab` tier loads every league deck, so the
`name_to_uid` change is exercised across the full card set.

## Result

Implemented.
