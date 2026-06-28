# Witch Enchanter // Witch-Blessed Meadow (vocab indices 263 / 264)

A **modal double-faced card** (MDFC): the player chooses, when playing it from hand, to cast
the front (a creature) or play the back (a land).

## Oracle text
**Witch Enchanter** — `{3}{W}` Creature — Human Warlock, 2/2
When Witch Enchanter enters, destroy target artifact or enchantment an opponent controls.

**Witch-Blessed Meadow** — Land (back face)
As Witch-Blessed Meadow enters, you may pay 3 life. If you don't, it enters tapped.
{T}: Add {W}.

## Forge script
Source: pre-existing combined local script
`bin/resources/cardsfolder/w/witch_enchanter_witch_blessed_meadow.txt` (one file, both faces,
`AlternateMode:Modal`). Per the DFC rule, no separate front-name file was created.

Front:
```
T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDestroy
SVar:TrigDestroy:DB$ Destroy | ValidTgts$ Artifact.OppCtrl,Enchantment.OppCtrl
```
Back:
```
R:Event$ Moved | ValidCard$ Card.Self | Destination$ Battlefield | ReplaceWith$ DBTap
SVar:DBTap:DB$ Tap | ETB$ True | Defined$ Self | UnlessCost$ PayLife<3> | UnlessPayer$ You
A:AB$ Mana | Cost$ T | Produced$ W
```

## Engine work
- **Front face: none required.** The ETB-destroy uses the existing `ChangesZone→Battlefield`
  trigger + `DB$ Destroy` with a `ValidTgts$ Artifact.OppCtrl,Enchantment.OppCtrl` comma-OR
  target spec — all already supported. Both faces are parsed (`parse_card_face` on the
  `ALTERNATE`-split back into `card.backside`), so both faces load.
- Both faces registered in `src/card_vocab.h` (front 263, back 264), mirroring the DFC
  convention used for Delver of Secrets / Insectile Aberration and Ajani Nacatl Pariah /
  Avenger. The "no card file found for 'Witch-Blessed Meadow'" line during `gen_card_costs.py`
  is the expected DFC back-face note (same as Insectile Aberration / Ajani Avenger) — the back
  has no mana cost and defaults to zero, which is correct for a land.

## Modal-DFC play-from-hand (general mechanic — IMPLEMENTED)
The engine previously only supported **transform** DFCs (a back face reached by flipping the
front in play: Delver, Ajani). Witch Enchanter is a **modal** DFC (MDFC, CR 712.x): both faces
are playable *from hand*, and the chosen face is what enters — it is not a transforming
permanent. Two reusable pieces were added; a future MDFC opts in by carrying `AlternateMode:Modal`
on its front script (and, if the back is a land, no further work).

1. **Parse `AlternateMode:Modal` → `CardData::is_modal_dfc`** (`src/parse.cpp`, in
   `parse_card_face`; field on `src/components/carddata.h`). Distinct from a transform DFC; only
   the front face carries the line.
2. **Offer the back face from hand.** `StateManager::determine_legal_actions`
   (`src/systems/state_manager_actions.cpp`, land-from-hand loop) now also offers a modal DFC
   whose **back face is a land** as a `PLAY_LAND` action ("Play Witch-Blessed Meadow"), gated by
   the same one-land-per-turn drop. The new `LegalAction::play_back_face` flag
   (`src/classes/action.h`) marks it. The front face is still offered as a `CAST_SPELL` in the
   spell loop (it is not a land, so it is never skipped). The branch is keyed on the back face's
   card type, so a future MDFC whose back is a nonland spell would instead route to a cast.
3. **Enter as the back face** (`src/action_processor.cpp`, `SPECIAL_ACTION`/play-land). When
   `play_back_face` is set, the card is marked `cur_game.pending_enters_transformed` before it is
   put onto the battlefield — reusing the existing transform machinery: `apply_permanent_components`
   calls `set_permanent_face(entity, true)` at entry so the permanent enters showing the back
   face (its name, Land types, and `{T}: Add {W}` mana ability), and the front-face ETB trigger is
   suppressed (`perm.transformed` gate in `state_manager_triggers.cpp`). As a modal card it never
   flips again (nothing grants it a transform ability).

## Modal-DFC NONLAND back face — cast the back (general mechanic — IMPLEMENTED)
The original support above only handled an MDFC whose **back face is a land** (a `PLAY_LAND`).
This was generalized so an MDFC whose **back face is a NONLAND spell** (creature / artifact /
enchantment / instant / sorcery / planeswalker) can be **cast as its back face** from hand,
paying the back face's mana cost and using only the back face's characteristics (CR 712.8 — a
modal DFC's spell/permanent has only the characteristics of the face being cast; CR 601 casting,
CR 608 resolution). The front face is still offered as a normal cast in the spell loop; the two
faces are independent choices from the one hand card.

Mechanism — a single new flag `LegalAction::cast_back_face` (`src/classes/action.h`), parallel
to `play_back_face`:

1. **Offer the back-face cast** — `offer_modal_back_face_casts()`
   (`src/systems/state_manager_actions.cpp`, called from `determine_legal_actions` right after the
   hand-cast loop). For each hand card with `is_modal_dfc && backside` whose back is **not** a
   land, it evaluates castability **against the back face** (`*front.backside`): timing
   (instant/Flash anytime, else sorcery-speed), spell-target legality + `ConditionPresent$` gate,
   `CantBeCast` statics, the payment-fail mask, and affordability of the back's
   `effective_base_cost`. It pushes a `CAST_SPELL` with `cast_back_face = true` and description
   "Cast <back name>". Kept in its own hand pass (not folded into the front-face loop) so a
   prohibition/`continue` on the front face cannot suppress the back-face option.
2. **Cast paying the back's cost + abilities** (`src/action_processor.cpp`, `CAST_SPELL`). The
   entity's `CardData` component is the front face; when `cast_back_face` is set, the whole cast
   path rebinds its local `card_data` to `*front_data.backside` (a `const CardData&`), so every
   read — mana cost paid (`effective_base_cost`/`prompt_mana_payment`), `SP$` spell ability put on
   the stack, target selection, type-driven cast events (NONCREATURE_SPELL_CAST / creature-spell
   tracking), and the cast log — comes from the back face. (Artifacts/enchantments like Sword of
   the Realms carry no `SP$` ability, so they simply resolve to the battlefield as a permanent;
   an instant/sorcery back puts its `SP$` effect on the stack.)
3. **Enter as the back face if it is a permanent.** When the back is a permanent
   (`is_permanent_card`), the cast marks `cur_game.pending_enters_transformed` on the spell entity
   — reusing the exact transform machinery the land case uses: on resolution
   `apply_permanent_components` calls `set_permanent_face(entity, true)`, so the permanent enters
   showing the back face (its name, types, statics, triggered/activated abilities) and the
   front-face ETBs are suppressed. Per the user's decision this deliberately reuses
   `pending_enters_transformed`/`set_permanent_face` rather than introducing a separate modal-face
   state, and adds no anti-transform guard. An instant/sorcery back is **not** marked (it leaves
   the stack on resolution, so no flip is needed and a stale marker can't mis-fire on a later ETB).
4. **Action card-id** (`src/machine_io.cpp`, `action_card_vocab_idx(const LegalAction&)`). The
   shared overload now resolves the **back** face's name for both `play_back_face` and
   `cast_back_face`, so the cast-back action encodes the back face's vocab id (mirroring the land
   case). Same single overload feeds BQUERY emission and the input-logger action log.

### Test card (not registered in vocab — test-only)
Verified with **Halvar, God of Battle // Sword of the Realms** (`AlternateMode:Modal`; front
`{2}{W}{W}` Legendary Creature God 4/4, back `{1}{W}` Legendary Artifact Equipment). Both faces
are offered from hand; casting the back pays `{1}{W}` (two Plains tapped) and resolves to the
battlefield **as Sword of the Realms** — the engine logs "casts Sword of the Realms" then
"transforms into Sword of the Realms", and the permanent is an Equipment (no creature is eligible
to attack), i.e. NOT the Halvar creature front. Casting the front still enters as the Halvar
creature (no transform; eligible to attack on a later turn). The Witch Enchanter land-back +
front-creature regression both still pass, and a scripted delver-vs-mav game finishes decisively
with no errors. (Tergrid, God of Fright // Tergrid's Lantern was tried first but its back face's
`UnlessCost$ Sac<...>` GenericChoice ability hits a *pre-existing* `std::stoi` parse crash at
`parse.cpp:1016` — unrelated to MDFC casting — so Halvar was used instead.) The test card was
**not** added to `src/card_vocab.h`; it encodes as vocab id -1, which is fine.

## "Enters tapped unless you pay N life" replacement (general mechanic — IMPLEMENTED)
The back's `R:Event$ Moved ... ReplaceWith$ DBTap` names an SVar
`DB$ Tap | ETB$ True | UnlessCost$ PayLife<3> | UnlessPayer$ You` — the shock-land pattern.
- **Parse** (`src/parse.cpp`, `parse_replacement_effects`): the conditional-ETB-tapped detector
  now also reads `UnlessCost$ PayLife<N>` off the named SVar body into the new
  `Effect::Replacement::tapped_unless_life` field (`src/components/effect.h`). It coexists with
  the existing `ConditionPresent$` board gate (Ba Sing Se).
- **Apply** (`src/systems/replacement_effects.cpp`): when the `SELF_TAPPED` candidate carries
  `tapped_unless_life > 0`, the controller is asked (via `request_optional_yesno`) whether to pay
  the life. If they can afford it (CR 119.4 — life ≥ N) and accept, they pay and the permanent
  enters **untapped**; declining or being unable to pay leaves it entering **tapped** (CR 614.1d
  self-replacement + 118.8 pay life).
- **Back-face replacement effects are read from the active face.** `collect()` for
  `ENTERS_BATTLEFIELD` now redirects the `ENTERS_TAPPED` scan to `cd.backside->replacement_effects`
  when the permanent is entering as its back face (live `Permanent::transformed`, or the one-shot
  `pending_enters_transformed` marker before the Permanent exists) — otherwise the back's
  replacement would never be seen. Single-faced lands are unaffected (no `backside`).

## Behavioral decisions (CR cites)
- Front ETB (CR 603.6a): "destroy target artifact or enchantment an opponent controls" — a
  targeted `Destroy`; the comma-OR `Artifact.OppCtrl,Enchantment.OppCtrl` spec restricts targets
  to opponent-controlled artifacts/enchantments.

## Behavioral note (back-face land action card id)
The back-face `PLAY_LAND` action's source entity is the combined card (whose `CardData` is the
front face), so the machine-mode card-id one-hot it emits resolves to the **front** vocab index
(263), not the back (264). The human-readable description is correct ("Play Witch-Blessed
Meadow"), so CLI/observers show the right name; only the RL card-id side-channel is approximate.
Left as-is (out of scope for the play-from-hand mechanic); revisit if MDFC back-face identity
matters to the policy.

## Tests (`train/test_harness.py`)
- **Both faces offered from hand** (battlefield_a 4 Plains so the cast is affordable): the menu
  shows BOTH "Play Witch-Blessed Meadow" (a `PLAY_LAND`) AND "Cast Witch Enchanter" (a
  `CAST_SPELL`) for the single hand card.
- **Back, decline 3 life → enters tapped**: `play:Witch Enchanter` plays the back; the pay-life
  prompt is declined → "Witch Enchanter enters tapped." → "transforms into Witch-Blessed Meadow";
  life stays 20; board shows `Witch-Blessed Meadow (T)`. Used the land-for-turn drop.
- **Back, pay 3 life → enters untapped + taps for {W}**: accepting the prompt logs "pays 3 life"
  (life 20→17), the meadow enters untapped, and its `{T}: Add {W}` is then activated to cast
  Guide of Souls ("activated Witch-Blessed Meadow for 1(W)").
- **Front creature (regression)**: cast Witch Enchanter `{3}{W}` off 4 Plains, ETB targets the
  opponent's Expedition Map → "Expedition Map is destroyed" (to the opponent's graveyard). The
  front enters as a creature (untransformed), confirming the cast path is unaffected.
- **Scripted regression**: full scripted games, `temp/witch_mdfc_a` (Witch Enchanter + Guide of
  Souls + Ocelot Pride + Plains/Island) vs a Mountain/Bolt/Bears/Map deck, seeds 1–4 — the
  scripted agent plays the meadow (enters tapped, untaps, taps for {W}), games end decisively (A
  wins all four), no draws, no non-fatal errors.
- **Existing-path regression**: Mystic Sanctuary still enters tapped (the back-face `reps`
  redirect only fires for cards with a `backside`; single-faced tapped lands are untouched).

## Result
Front implemented and verified (ETB destroy). The back-face land **Witch-Blessed Meadow is now
playable**: general modal-DFC play-from-hand support (`AlternateMode:Modal` → offer both faces;
back enters via the transform machinery) plus the "enters tapped unless you pay N life"
replacement were added. Build clean; all scenarios pass; scripted regression decisive with no
errors or draws.
