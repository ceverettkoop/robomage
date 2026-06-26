# White Orchid Phantom (vocab index 147)

## Oracle text
Flying, first strike
When White Orchid Phantom enters, destroy up to one target nonbasic land. Its controller may
search their library for a basic land card, put it onto the battlefield tapped, then shuffle.

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/w/white_orchid_phantom.txt`).
Key tags:
- `K:Flying`, `K:First Strike`
- `T:Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDestroy`
  — ETB trigger.
- `SVar:TrigDestroy:DB$ Destroy | ValidTgts$ Land.nonBasic | TargetMin$ 0 | TargetMax$ 1 | SubAbility$ DBSearch`
- `SVar:DBSearch:DB$ ChangeZone | Origin$ Library | Destination$ Battlefield | DefinedPlayer$ TargetedController | ChangeType$ Land.Basic | Tapped$ True`

This is the Ghost Quarter (vocab index 134) template — destroy target land, its controller
fetches a basic — with two differences: it is an ETB trigger (not an activated ability), and
the fetched basic enters **tapped** (`Tapped$ True`).

## Engine work
None new — covered.
- `Flying` / `First Strike` keywords: parsed and applied as evasion/combat keywords.
- ETB `T:Mode$ ChangesZone | Destination$ Battlefield | ValidCard$ Card.Self`: standard
  enters-the-battlefield trigger registration (`src/parse.cpp`, state-manager triggers).
- `DB$ Destroy | ValidTgts$ Land.nonBasic | TargetMin$ 0 | TargetMax$ 1`: optional
  (up-to-one) targeting + `EffectKind::Destroy` (`src/effects/effect_destroy.cpp`).
- `DB$ ChangeZone | DefinedPlayer$ TargetedController` (search for the *targeted land's*
  controller): handled in `src/effects/effect_change_zone.cpp` (proven by Erode / Ghost
  Quarter).
- `Tapped$ True`: parsed at `effect_change_zone.cpp:294` into `Ability::enters_tapped`,
  consumed via `cur_game.pending_enters_tapped` (proven by Edge of Autumn).

## Behavioral decisions
None new. Follows Ghost Quarter semantics; the only addition is the tapped-entry of the
fetched basic, which uses the existing `Tapped$` path.
- `ChangeTypeDesc$ basic land` emits a cosmetic "Unrecognized ability param" warning (display
  text only; ignored).

## Tests
- Scenario: Player A casts White Orchid Phantom; ETB trigger targets Player B's nonbasic
  land (Undercity Sewers). Observed: "Undercity Sewers is destroyed" (→ opponent graveyard),
  then "Searching Player B's library" — Player B (the targeted land's controller) chose
  Swamp, "Player B shuffles their library", "Player B puts Swamp to the battlefield",
  **"Swamp enters tapped."** and board shows "Opp BF: Swamp (T)". Result: pass — the fetched
  basic correctly enters tapped, and the search is performed by the targeted land's
  controller.

## Result
Implemented (registration only; mechanic pre-existing via the Ghost Quarter template plus the
`Tapped$` enters-tapped path).
