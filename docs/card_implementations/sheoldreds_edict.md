# Sheoldred's Edict (vocab index 163)

## Oracle text
Choose one —
- Each opponent sacrifices a nontoken creature.
- Each opponent sacrifices a creature token.
- Each opponent sacrifices a planeswalker.

(`1B` Instant.)

## Forge script (Source: pre-existing local `bin/resources/cardsfolder/s/sheoldreds_edict.txt`)
Key tags:
- `A:SP$ Charm | Choices$ SacNontoken,SacToken,SacPW` — modal "choose one" (the Charm
  framework already handled, Kozilek's Command precedent).
- `SVar:SacNontoken:DB$ Sacrifice | Defined$ Opponent | SacValid$ Creature.!token`
- `SVar:SacToken:DB$ Sacrifice | Defined$ Opponent | SacValid$ Creature.token`
- `SVar:SacPW:DB$ Sacrifice | Defined$ Opponent | SacValid$ Planeswalker`

Cosmetic `SacMessage$` tags are ignored (harmless "Unrecognized ability param" warning, same
class as `StackDescription$`).

## Engine work
General, reusable `Defined$ Opponent` (edict) support added to the **Sacrifice effect**; no
retagging — the parser already maps `Defined$ Opponent` to `Ability::defined_each_opponent`
(`src/parse.cpp:900`, shared with `deal_damage`'s "each opponent" branch).

- `src/effects/effect_sacrifice.cpp`
  - Refactored the per-player sacrifice into a static helper `sacrifice_one(ab, sacrificer,
    optional, orderer)`: builds the SACRIFICE_PERMANENT choice menu over the *sacrificer's*
    matching battlefield permanents and resolves the choice. The sacrificer is both the chooser
    and the controller of the sacrificed permanent (this is what makes it an edict).
  - `sacrifice()` now branches on `ab.defined_each_opponent`: if set, each opponent of
    `ab.controller` runs `sacrifice_one(..., optional=false, ...)` (the edict is mandatory for an
    opponent who controls a matching permanent). Otherwise the controller sacrifices as before
    (`Optional$` honored). Two-player game ⇒ "each opponent" is the single other player
    (CR 109.5 / 102.1), mirroring `effect_deal_damage.cpp`.
  - The `SacValid$` filter now routes through the shared `permanent_matches_cards_filter`
    (declared in `effects.h`, defined in `effect_put_counter_all.cpp`) instead of the old
    "strip everything after the first dot + `permanent_has_type`" logic. This honors token /
    nontoken / color / `YouCtrl` qualifiers generally, and the filter's controller argument is
    the sacrificer so `YouCtrl`-style qualifiers resolve relative to the player sacrificing.
- `src/effects/effect_put_counter_all.cpp`
  - `permanent_matches_cards_filter`: added Forge's `!token` negation spelling as an alias of
    `nonToken` (general; the script writes `Creature.!token`).

Charm modal handling and the SACRIFICE_PERMANENT choice machinery were reused unchanged.

## Behavioral decisions (CR cites)
- **Edict = the controller of the permanent chooses.** "Each opponent sacrifices a …" — the
  *opponent* (controller of the permanent) chooses which of their own permanents to sacrifice,
  not the spell's caster (CR 701.16 / 701.16a: a player sacrifices a permanent *they* control;
  the choosing player is the one whose permanent it is). The implementation presents the
  SACRIFICE_PERMANENT menu to the opponent and restricts candidates to permanents they control.
- **Mandatory.** An opponent who controls a matching permanent must sacrifice one (the edict is
  not optional). An opponent who controls no matching permanent sacrifices nothing (no choice,
  no crash) — `sacrifice_one` returns 0 when the candidate set is empty.
- **"Each opponent"** in a two-player game = the caster's single opponent (CR 109.5 / 102.1).
- The caster's own permanents are never candidates.

## Tests (`train/test_harness.py`, inline hands/battlefields + semantic `--play`)
- **Nontoken-creature mode**, opponent controls two nontoken creatures (Noble Hierarch, Birds of
  Paradise) and the caster controls a Noble Hierarch: the *opponent* is offered both of THEIR
  nontoken creatures (caster's Hierarch absent), picks Birds → "Player B sacrifices Birds of
  Paradise"; caster's Hierarch untouched.
- **Token vs nontoken discrimination**, opponent controls one nontoken creature (Noble Hierarch)
  AND a token (Eldrazi Spawn from Kozilek's Command):
  - token mode → only `Sacrifice Eldrazi Spawn Token` offered (nontoken excluded); B sacks the token.
  - nontoken mode → only `Sacrifice Noble Hierarch` offered (token excluded); B sacks the Hierarch.
- **Planeswalker mode, no matching permanent**: opponent controls only a creature → sacrifices
  nothing, no crash, game continues.
- **Regression** (scripted full games, `--deck-a`/`--deck-b` temp decks, seeds 1/2/3): each game
  terminates with a winner (no draws), no non-fatal/fatal errors (only the cosmetic
  `SacMessage$` warning). Sheoldred's Edict is cast and resolves in scripted play. Temp decks
  cleaned up.

## Result
`make HEADLESS=TRUE` builds clean. The `Defined$ Opponent` edict sacrifice is general and reused
by any future `DB$ Sacrifice | Defined$ Opponent` card; the `!token` filter alias and the
`permanent_matches_cards_filter` routing generalize the `SacValid$` matcher.
