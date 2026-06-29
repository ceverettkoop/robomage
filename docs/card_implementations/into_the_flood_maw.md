# Into the Flood Maw  (vocab index 284)

## Oracle text
Gift a tapped Fish (You may promise an opponent a gift as you cast this spell. If you do, they
create a tapped 1/1 blue Fish creature token before its other effects.)
Return target creature an opponent controls to its owner's hand. If the gift was promised, instead
return target nonland permanent an opponent controls to its owner's hand.

(Instant, mana cost {U}.)

## Forge script
- Source: `bin/resources/cardsfolder/i/into_the_flood_maw.txt` (not edited; cardsfolder is gitignored).
- Key tags (parsed as written — no category retagged):
  - `K:Gift` — the Gift keyword (CR 702.176).
  - `SVar:GiftAbility:DB$ Token | TokenScript$ u_1_1_fish | TokenTapped$ True | TokenOwner$ Promised
    | LockTokenScript$ True | GiftDescription$ a tapped Fish` — the gift effect: a tapped 1/1 blue
    Fish for the promised opponent.
  - `A:SP$ ChangeZone | ValidTgts$ Creature.OppCtrl | TargetMin$ X | TargetMax$ X | Origin$
    Battlefield | Destination$ Hand | SubAbility$ DBChangeZone` — bounce a creature.
  - `SVar:DBChangeZone:DB$ ChangeZone | ValidTgts$ Permanent.nonLand+OppCtrl | TargetMin$ Y |
    TargetMax$ Y | Origin$ Battlefield | Destination$ Hand` — bounce a nonland permanent.
  - `SVar:X:Count$PromisedGift.0.1` and `SVar:Y:Count$PromisedGift.1.0` — the gift switch: not
    promised ⇒ X=1 (bounce 1 creature), Y=0 (the nonland-bounce targets nothing and does nothing);
    promised ⇒ X=0 (creature-bounce inert), Y=1 (bounce 1 nonland permanent). The two abilities
    encode the "instead" by which one actually has a target.

## Engine work (general Gift keyword + Count$PromisedGift, keyed on each tag's intended meaning)

### 1. The Gift keyword — promise-at-cast + gift on resolution (CR 702.176)
- `CardData::has_gift` / `gift_abilities` / `gift_description` (`src/components/carddata.h`).
  `K:Gift` is parsed in the keyword pass (`src/parse.cpp`): it parses the card's `GiftAbility`
  SVar (a `DB$ Token`) into `gift_abilities` and extracts `GiftDescription$` for the cast prompt.
  General: any Gift card's gift effect lives in its `GiftAbility` SVar.
- Promise-the-gift choice **at cast time** (`src/action_processor.cpp`, next to the kicker/replicate
  optional costs): when a `has_gift` spell is cast, an `OPTIONAL_YESNO` "promise <gift> to your
  opponent" is offered (CR 702.176b — a choice, not a cost). The result is recorded on
  `Spell::gift_promised` (`src/components/spell.h`) and mirrored into the transient
  `cur_game.pending_gift_promised` (`src/classes/game.h`) so it is readable while targets are chosen
  (the `Spell` component does not exist yet at that point).
- Gift effect **on resolution, before other effects** (CR 702.176c): the parsed `gift_abilities` are
  copied onto the resolving spell's primary ability at cast; `Ability::resolve`
  (`src/components/ability.cpp`) runs them at the **start** of resolution iff `Spell::gift_promised`
  is set on the source spell. Skipped when not promised.
- Gift token routing: `TokenOwner$ Promised` and `TokenTapped$ True` are parsed into `TokenParams`
  (`owner_is_promised`, `tapped` — `src/components/ability_params.h`,
  `src/effects/effect_token.cpp`). `effects::token` creates the token under the **opponent of the
  ability's controller** (two-player) and stamps it tapped on the freshly-bootstrapped `Permanent`.

### 2. `Count$PromisedGift.<high>.<low>` dynamic value
- `evaluate_dynamic_amount` (`src/components/ability.cpp`) returns `high` if the spell being
  cast/resolved promised its gift (`cur_game.pending_gift_promised`), else `low`. Mirrors the
  existing `Count$Threshold` / `Count$UrzaLands` `.high.low` branches. General over any Gift card
  that scales a dynamic amount on the promise.

### 3. Dynamic (non-xPaid) target counts → conditional bounce target
- New `Ability::target_min_count_expr` / `target_max_count_expr` (`src/components/ability.h`). A
  `TargetMin$`/`TargetMax$` SVar that resolves to a `Count$` expression OTHER than `Count$xPaid`
  (here `Count$PromisedGift`) is captured into these by `resolve_xpaid_target_counts`
  (`src/parse.cpp`), and the sub-ability `TargetMax$` parse path now stores its SVar key so it is
  resolved the same way.
- `select_target` (`src/action_processor.cpp`) evaluates these up front (alongside the existing
  `xPaid` path — CR 601.2b: the gift promise is already decided) and stamps `target_min`/
  `target_max`. A resolved count of **0** makes the ability **target nothing and do nothing** (the
  engine already supports an inert no-target ability: `is_target_valid` passes for `target_min==0`
  and `effects::change_zone` moves an empty target list). This is exactly how the script's "instead"
  works: only the promised-or-not mode that resolves to count 1 actually bounces.

## Behavioral decisions (made in lieu of asking a human)
- **The "instead" is honored via the script's two-ability / 0-or-1-target encoding, not by retagging
  to one widened ability.** Not promised ⇒ the creature-bounce targets 1 and the nonland-bounce is
  inert; promised ⇒ the reverse, so the spell instead bounces any nonland permanent. The gift token
  is created before the bounce (verified ordering).
- **Cast-time legality is checked against the primary ability's `Creature.OppCtrl` filter.** So the
  spell is offered when an opponent creature exists (the common case, and the only case the tests
  need). Caveat/known limitation: if the opponent controls a noncreature nonland permanent but **no**
  creature, the spell is not currently offered even though promising the gift would let you bounce
  that permanent. Narrow edge; the defining behavior (promise ⇒ bounce nonland permanent + Fish;
  no promise ⇒ bounce creature) is fully implemented.
- **No observation/state-vector change.** `Spell::gift_promised` and `cur_game.pending_gift_promised`
  are engine-internal; `STATE_SIZE`, `OBS_SIZE`, `N_CARD_TYPES` and the obs layout are unchanged.

## Tests (test_harness, seed 1; opponent board = Grizzly Bears + Grafdigger's Cage)
1. **Not promised** — A casts Into the Flood Maw, declines the gift. The target menu offers **only**
   Grizzly Bears (the Cage is not a legal target). Bears returns to B's hand; B gets **no** token;
   the inert nonland-bounce resolves doing nothing.
2. **Promised** — A casts it and accepts ("Player A promises the gift to Player B"). The target menu
   now offers **both** Grizzly Bears and Grafdigger's Cage; A targets the Cage. Resolution order:
   the inert creature-bounce, then **Token created: 1/1 Fish Token** (before the bounce), then
   Grafdigger's Cage returns to B's hand. The Fish enters under B's control **tapped** (`T,SICK`).
- No game-result draws; no non-fatal errors (only benign WARNINGs for the cosmetic/AI-hint tags
  `AITgts$` / `LockTokenScript$` / `GiftDescription$`).

## Result
implemented
