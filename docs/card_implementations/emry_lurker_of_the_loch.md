# Emry, Lurker of the Loch  (vocab index 131)

## Oracle text
Affinity for artifacts (This spell costs {1} less to cast for each artifact you control.)

When Emry, Lurker of the Loch enters, mill four cards.

{T}: Choose target artifact card in your graveyard. You may cast that card this turn. (You
still pay its costs. Timing rules still apply.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/e/emry_lurker_of_the_loch.txt`
- Type: `Legendary Creature Merfolk Wizard`, mana cost `2 U`, P/T 1/2.
- Key tags:
  - `K:Affinity:Artifact` — the spell costs {1} less to cast per artifact its controller
    controls (CR 702.41).
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self |
    Execute$ TrigMill` + `SVar:TrigMill:DB$ Mill | NumCards$ 4 | Defined$ You` — the ETB
    "mill four cards" trigger (already-supported Mill effect).
  - `A:AB$ Effect | Cost$ T | TgtZone$ Graveyard | ValidTgts$ Artifact.YouOwn |
    RememberObjects$ Targeted | StaticAbilities$ STPlay | ExileOnMoved$ Graveyard` — the tap
    ability: it creates a transient continuous Effect that grants permission to cast the
    targeted artifact card from the graveyard this turn.
  - `SVar:STPlay:Mode$ Continuous | MayPlay$ True | Affected$ Card.IsRemembered+nonLand |
    AffectedZone$ Graveyard` — the "you may cast that card this turn" grant.

No tags were retagged or repurposed; every mechanic below is keyed on the tag's intended
meaning. The cosmetic `StaticAbilities$`, `ExileOnMoved$`, `RememberObjects$` and prompt tags
are not load-bearing for a single artifact-card grant and produce only the pre-existing
`WARNING: Unrecognized ability param` line; the grant's behavior is inferred from the
`AB$ Effect` + `MayPlay$ True` (AffectedZone$ Graveyard) intent.

## Engine work
Three pieces, each a general handler keyed on the tag's meaning (CR 702.41 Affinity / 601.2f
cost reductions, CR 118 mill, CR 601.3e cast permissions):

1. **Affinity for artifacts** (`K:Affinity:Artifact`).
   - `src/components/carddata.h`: new `bool affinity_artifact` flag.
   - `src/parse.cpp`: a `K:Affinity` branch sets the flag (only the `Artifact` variant is
     supported) alongside the existing Delve branch.
   - `src/systems/state_manager_statics.cpp`: `effective_base_cost` — the single
     effective-base-cost builder shared by legality (`determine_legal_actions`) and payment
     (`action_processor`) — gained an optional `caster` parameter. When the card has
     `affinity_artifact` and `caster` is known, it removes one generic ({1}) pip per artifact
     the caster controls (`artifacts_controlled_by`, an `is_battlefield_permanent` +
     `permanent_has_type("Artifact")` scan). Reductions are applied after the RaiseCost
     additions (CR 601.2f); only generic pips are removed (a colored pip is never reduced) and
     the generic total never drops below zero.
   - The two callers (`state_manager_actions.cpp` legality, `action_processor.cpp` payment) now
     pass the caster so affordability and the actual mana paid agree on the reduced cost.

2. **ETB mill 4** — no new work; the existing `Mill` effect (`src/effects/effect_mill.cpp`)
   already handles `DB$ Mill | NumCards$ 4 | Defined$ You`.

3. **"You may cast that card this turn" grant** (`AB$ Effect`).
   - `src/components/ability.cpp` (`is_legal_target`): the graveyard-target path now also
     recognizes `YouOwn`/`OppOwn` (not only `YouCtrl`/`OppCtrl`) so `ValidTgts$ Artifact.YouOwn`
     restricts to artifact cards in the caster's own graveyard. (A card in a graveyard is owned
     and controlled by the same player, so the two qualifiers are equivalent there.)
   - `src/effects/effect_grant_cast.cpp` (new): the `Effect` category resolves to a per-turn
     cast permission rather than a stack/continuous Effect object — it inserts the targeted
     graveyard card into `cur_game.may_cast_this_turn`.
   - `src/effects/effect_kind.{h,cpp}`, `effect_table.cpp`, `effects.h`: register the new
     `GrantCast` kind, mapped from the `"Effect"` category string and dispatched to `grant_cast`.
   - `src/classes/game.h`: new `std::set<Entity> may_cast_this_turn` (CR 601.3e cast permission
     set). `src/classes/game.cpp`: cleared each cleanup so the grant expires at end of turn.
   - `src/systems/state_manager_actions.cpp`: after the flashback block,
     `determine_legal_actions` offers a `Cast <card> (from graveyard)` action for each card in
     `may_cast_this_turn` that is still in the priority player's graveyard — at the timing its
     type allows (Flash/Instant any time, else sorcery-speed), with a legal-targets check, a
     `cast_prohibited(…, Zone::GRAVEYARD)` check (Grafdigger's Cage etc.), and an affordability
     check on its normal (Affinity-reduced) cost. The existing zone-agnostic `CAST_SPELL`
     execution path moves the card from the graveyard to the stack and resolves it.

## Behavioral decisions (made in lieu of asking a human)
- **Affinity reduces only generic mana** (CR 702.41 / 601.2f): {2}{U} with 2 artifacts becomes
  {U}; the {U} pip is never reduced. Verified Emry castable off a single blue source with 2
  artifacts in play, and *not* castable off a single source with 0 artifacts.
- **The grant is a per-turn permission, not a persistent Effect object.** Forge models it as a
  transient continuous Effect whose static grants MayPlay; the engine records the specific
  graveyard card and offers it this turn, then clears the permission at cleanup (CR 601.3e). The
  card is cast for its normal cost from the graveyard via the standard cast path, so it enters
  the battlefield (or resolves) normally and leaves the graveyard.
- **Only the granting player may cast it** — the permission set is filtered by graveyard owner,
  so an opponent never sees the offer.
- **`YouOwn` == `YouCtrl` for a graveyard target.** A card in a graveyard is owned and
  controlled by the same player, so the target filter restricts to the caster's own graveyard.
- **Cosmetic tags ignored, not repurposed.** `ExileOnMoved$ Graveyard` / `StaticAbilities$
  STPlay` / `RememberObjects$ Targeted` describe Forge's Effect-object plumbing; the single-card
  behavior (grant a cast for the chosen artifact this turn) is fully captured without them.

## Tests
Isolation (`train/test_harness.py`):
- **Affinity, positive.** Emry in hand; 2 `Mishra's Bauble` + 1 `Island` on A's battlefield. In
  the main phase `Cast Emry, Lurker of the Loch` is offered even though only one blue source is
  available — {2}{U} was reduced to {U}. PASS.
- **Affinity, negative control.** Same line with 0 artifacts and 1 `Island`: `Cast Emry` is NOT
  offered (3 mana unaffordable off one source). PASS — proves the reduction is what enabled the
  positive case.
- **ETB mill 4.** Cast Emry (affinity-reduced); on entry the engine logs four `Player A mills …`
  lines and the four cards appear in A's graveyard. PASS.
- **Tap ability grants cast-from-graveyard.** Emry pre-set on A's battlefield (no summoning
  sickness), `Lotus Petal` in A's graveyard. Activate Emry targeting Lotus Petal → log `Lotus
  Petal may be cast from the graveyard this turn` and the menu gains `Cast Lotus Petal (from
  graveyard)`. Casting it logs `Player A casts Lotus Petal` / `Lotus Petal enters the
  battlefield`, and Lotus Petal leaves the graveyard. PASS.

Regression (`train/test_harness.py --scripted`, 6 games, seeds 1–6): deck `temp/emry_test`
(4 Emry, 4 Mishra's Bauble, 4 Lotus Petal, 8 Grizzly Bears, 4 Murktide Regent, 4 Lightning Bolt,
Island/Volcanic Island/Mountain) vs `mav`. Decisive in 4/6 games (the other 2 are scripted-agent
stalemate draws between two slow boards, not an engine fault); no fatal/non-fatal errors, no
asserts or tracebacks, engine stable. Emry was drawn, cast (Affinity-reduced) and its ETB mill
resolved in real games. Only the pre-existing cosmetic `WARNING: Unrecognized ability param`
lines (Emry's own STPlay/ExileOnMoved, and unrelated cards like Green Sun's Zenith) appeared.
Temp decks cleaned up.

## Result
implemented
