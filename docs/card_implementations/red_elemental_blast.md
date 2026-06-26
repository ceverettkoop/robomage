# Red Elemental Blast  (vocab index 116)

## Oracle text
Choose one —
- Counter target blue spell.
- Destroy target blue permanent.

(Instant, mana cost {R}.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/r/red_elemental_blast.txt`
- Key tags:
  - `A:SP$ Charm | Choices$ DBCounter,DBDestroy` — a modal "choose one" spell (CR 700.2 / 601.2b),
    resolved by the existing `effects::charm` handler.
  - `SVar:DBCounter:DB$ Counter | TargetType$ Spell | ValidTgts$ Card.Blue | …` — mode 1: counter
    a target **blue** spell. Blue is part of the *target restriction* (`ValidTgts$ Card.Blue`),
    not a resolution condition.
  - `SVar:DBDestroy:DB$ Destroy | ValidTgts$ Permanent.Blue | …` — mode 2: destroy a target **blue**
    permanent. Again blue is a target restriction (`ValidTgts$ Permanent.Blue`).
  - `AI:RemoveDeck:Random` — Forge AI hint, ignored.

## Relationship to Pyroblast / Hydroblast (near-identical siblings)
Pyroblast (index 77) and Hydroblast (76) are the functional twins, but their Oracle wording —
and therefore their Forge scripts — differ in an important way from Red Elemental Blast:

- **Pyroblast/Hydroblast** read "Counter target spell **if it's blue**" / "Destroy target permanent
  **if it's blue**." Blue is a *resolution condition*, not a targeting restriction. Their scripts use
  `ValidTgts$ Card` (any spell) / `ValidTgts$ Permanent` (any permanent) plus
  `ConditionPresent$ Spell.Blue` / `Card.Blue`. They may legally *target* anything; the counter/destroy
  effect simply does nothing if the chosen target is not blue (enforced in
  `effects::counter` / `effects::destroy` via `target_color_condition_met`).
- **Red Elemental Blast / Blue Elemental Blast** (the original 1993 wording) read "Counter target
  **blue** spell" / "Destroy target **blue** permanent." Blue is baked into the *target restriction*
  (`ValidTgts$ Card.Blue` / `Permanent.Blue`), so a non-blue spell/permanent is not even a legal
  target (CR 115.1, 601.2c: a target must meet the spell's targeting restrictions when chosen).

The gameplay outcome is usually the same, but the distinction is real: REB cannot be *cast targeting*
a non-blue object at all (and a mode with no legal blue target is unavailable), whereas Pyroblast can
be cast at a non-blue object and then fizzle on resolution.

## Engine work
One small, general change keyed on tag intent — no card-specific code.

- **Positive color target restriction (`src/components/ability.cpp::is_legal_target`).** The target
  legality predicate already enforced `non<Color>` exclusions (`passes_noncolor_restriction`, for
  Snuff Out's `Creature.nonBlack`) but had no handling for a *positive* `.<Color>` qualifier in
  `ValidTgts$`. Added `passes_color_restriction(vt, cd)`: if `valid_tgts` contains a dotted color
  token (`.White`/`.Blue`/`.Black`/`.Red`/`.Green`) the candidate must be that color
  (`card_is_color`, which reads `explicit_colors` else mana-cost colors). The check is applied in
  two places:
  - the **Spell** target branch (so `ValidTgts$ Card.Blue` only offers/accepts blue spells on the
    stack), and
  - the **battlefield-permanent** branch, right after the permanent is confirmed on the battlefield
    (so `ValidTgts$ Permanent.Blue` only offers/accepts blue permanents).

  The token is matched in its dotted form (`.Blue`) specifically so it does not collide with the
  `non<Color>` tokens, which embed the bare color name (`nonBlue` contains `Blue`). Because the same
  predicate (`is_legal_target`) drives both target *enumeration* (`build_valid_targets`) and
  resolution-time *re-verification* (`is_target_valid`), the restriction is enforced consistently at
  announcement and on resolution (CR 608.2b) without any drift.

  Pyroblast/Hydroblast are unaffected: their `valid_tgts` is `Card` / `Permanent` (no dotted color
  token), so `passes_color_restriction` is a no-op for them and their condition-based color logic
  continues to run unchanged.

## Behavioral decisions (made in lieu of asking a human)
- **Blue is a targeting restriction, not a resolution condition (CR 115.1).** Implemented per the
  card's actual script tags (`ValidTgts$ Card.Blue` / `Permanent.Blue`) — a non-blue object is never
  a legal target, and a mode whose only would-be target is non-blue is simply not offered (the charm
  handler already filters out modes with no legal target; if neither mode has a legal blue target the
  whole spell fizzles per CR 608.2b). This is deliberately *not* shortcut into Pyroblast's
  condition-based form; the two scripts are honored as written.
- **No retag.** The fix is a general positive-color filter applied to whatever `ValidTgts$` color
  qualifier the script carries, so it serves Blue Elemental Blast and any future card with a
  same-color target restriction, not just REB.

## Tests
- Isolation (test_harness):
  - **Mode 1 counters a blue spell:** B casts Brainstorm; A casts Red Elemental Blast, chooses
    "Counter target blue spell.", targets Brainstorm — "Brainstorm is countered" and it goes to the
    graveyard. No fizzle / non-fatal error.
  - **Mode 2 destroys a blue permanent, non-blue not offered:** B has Flying Men (blue), Grizzly
    Bears (green), and Island (colorless) in play; A casts Red Elemental Blast, chooses "Destroy
    target blue permanent." — the target menu offers **only Flying Men**; Grizzly Bears and Island
    are not legal targets. Flying Men is destroyed → graveyard.
  - **Negative (counter mode rejects a red spell):** with only a red Lightning Bolt on the stack and
    no blue permanent in play, REB finds no legal mode and "charm fizzles" — neither mode accepts the
    non-blue spell/permanents; the Bolt resolves normally.
- Regression (test_harness --scripted, full games): mono-red deck with 4× Red Elemental Blast
  (+ Lightning Bolt, Grizzly Bears, Mountains) vs a mono-blue deck (Brainstorm, Ponder, Flying Men,
  Islands), seeds 1–6 — all six decisive (1 A win, 5 B wins), no draws, no max-decision caps, no
  non-fatal errors (only the pre-existing cosmetic `Unrecognized ability param: Reorder$ True` for
  Brainstorm). REB was cast and resolved in real games (multiple copies seen in graveyards).

## Result
implemented
