# Teferi, Time Raveler  (vocab index 344)

## Oracle text
Each opponent can cast spells only any time they could cast a sorcery.

[+1]: Until your next turn, you may cast sorcery spells as though they had flash.

[-3]: Return up to one target artifact, creature, or enchantment to its owner's hand. Draw a card.

(Legendary Planeswalker — Teferi, loyalty 4, mana cost {1}{W}{U}.)

## Forge script
- Source: pre-existing local — `bin/resources/cardsfolder/t/teferi_time_raveler.txt`
- Key tags:
  - `Loyalty:4` — starting loyalty (existing planeswalker framework).
  - `S:Mode$ CantBeCast | ValidCard$ Card | Caster$ Opponent | OnlySorcerySpeed$ True` — the
    static: each opponent may cast spells only at sorcery speed.
  - `A:AB$ Effect | Cost$ AddCounter<1/LOYALTY> | Planeswalker$ True | StaticAbilities$ STPlay |
    Duration$ UntilYourNextTurn` with `SVar:STPlay:Mode$ CastWithFlash | ValidCard$ Sorcery |
    ValidSA$ Spell | Caster$ You` — the +1 cast-with-flash grant.
  - `A:AB$ ChangeZone | Cost$ SubCounter<3/LOYALTY> | Planeswalker$ True | Origin$ Battlefield |
    Destination$ Hand | TargetMin$ 0 | TargetMax$ 1 | ValidTgts$ Artifact,Creature,Enchantment |
    SubAbility$ DBDraw` with `SVar:DBDraw:DB$ Draw` — the -3 bounce + draw.

## Engine work
All changes are general (keyed on each tag's intended meaning), not card-specific.

**Mechanics added (general): opponent-sorcery-speed-lock, cast-with-flash.**

- **opponent-sorcery-speed-lock (new casting-timing mechanic).** CR 601.3a / 307 / 117.1a.
  - `src/components/static_ability.h`: added `bool only_sorcery_speed` to the CantBeCast fields.
  - `src/parse.cpp` (`parse_one_static_ability`): parse `OnlySorcerySpeed$ True` on a CantBeCast
    static into `only_sorcery_speed`. (`ValidCard$ Card` + `Caster$ Opponent` already set
    `cant_cast_filter`/`cant_cast_by_opponent`.)
  - `src/systems/rules_modifying.cpp` (`cast_prohibited`): in the existing `cant_cast_by_opponent`
    branch, a static that also sets `only_sorcery_speed` is **skipped** — it is a timing
    restriction, not a "can't cast at all" prohibition, so it must not blanket-block the opponent
    (that would be Voice of Victory's behavior).
  - `src/systems/rules_modifying.{h,cpp}` (`opponent_sorcery_speed_locked(caster)`): new query —
    true when an active `OnlySorcerySpeed$` + `Caster$ Opponent` CantBeCast static's controller is
    the caster's opponent. Reuses the same `Caster$ Opponent` CantBeCast path a future opponent
    cmc/land-count lock (Lavinia) would key on.
  - `src/systems/state_manager_actions.cpp` (`determine_legal_actions`, cast-legality path): the
    **enforcement point**. In the hand-cast timing gate (and, defensively, the flashback, escape,
    graveyard-permission, and impulse/exile cast paths), when the priority player is
    `opponent_sorcery_speed_locked`, the "instant-speed" flag is forced off, so **every** spell —
    even an instant or a flash spell — requires the sorcery-speed window (the caster's own main
    phase, empty stack) to be legal.

- **cast-with-flash (new casting-timing mechanic).** CR 702.8 / 116 "as though" timing permission.
  - `src/components/ability.h`: added `bool effect_cast_with_flash` + `std::string
    effect_cast_with_flash_filter` to the DB$ Effect fields.
  - `src/parse.cpp`: helper `param_value()` (fetch one `key$` value from a Forge line/SVar body);
    both the top-level and sub-ability `AB$ Effect | StaticAbilities$ <SVar>` resolution now
    detect a `Mode$ CastWithFlash` SVar body and capture its `ValidCard$` filter (e.g. "Sorcery").
  - `src/classes/game.h`: added `struct CastWithFlashPermission { controller; filter;
    until_your_next_turn; }` and `std::vector<...> cast_with_flash_permissions` — a sourceless,
    duration-scoped cast-timing permission (models the transient Forge Effect object without a
    stack/effect entity, mirroring the existing `PlayerProtectionFromEverything` pattern).
  - `src/effects/effect_grant_cast.cpp` (`grant_cast`): on `effect_cast_with_flash`, records a
    `CastWithFlashPermission` bound to the effect's controller + filter for the effect's Duration
    (`duration_until_your_next_turn`).
  - `src/classes/game.cpp`: lapse the permission at the controller's untap step for the
    `until_your_next_turn` form, and at cleanup for the end-of-turn form (identical to the
    protection-from-everything lapse points).
  - `src/systems/rules_modifying.{h,cpp}` (`cast_with_flash_active(caster, card)`): new query —
    true when an active permission owned by `caster` covers the spell (its `ValidCard$` filter
    matches via `card_matches_filter`).
  - `src/systems/state_manager_actions.cpp` (hand-cast timing gate): a matching spell is treated
    as instant-speed. Order in the gate is `flash-permission → then lock veto`, so a player under
    an opponent lock cannot use their own flash-granting (the lock wins — correct per CR).

- **-3 bounce + draw (existing mechanics).** `AB$ ChangeZone` up-to-1 target A/C/E → owner's hand
  plus a `SubAbility$ DB$ Draw` — both already covered by the planeswalker loyalty-cost framework,
  the ChangeZone-to-hand mover, and the sub-ability draw. Verified working (no new code).

## Behavioral decisions (made in lieu of asking a human)
- **"only any time they could cast a sorcery" = sorcery-speed timing, not a prohibition**
  (CR 601.3a): the affected caster may cast a spell only in their own main phase with an empty
  stack, regardless of the spell's instant/flash type. So it is enforced as a timing gate over the
  normal instant-speed check, NOT as a `cast_prohibited` block — an opponent can still cast at
  sorcery speed on their own turn.
- **The lock overrides self-granted flash.** If a player is simultaneously under an opponent
  sorcery-speed lock and holds a cast-with-flash permission (two Teferis in play), the lock still
  forces sorcery timing — the gate applies the flash permission first, then the lock veto. This
  matches CR: "as though it had flash" lifts only the spell's inherent sorcery-speed timing, not a
  separate rules-modifying restriction on the player.
- **"cast sorcery spells as though they had flash" is a permission scoped by the `ValidCard$`
  filter** ("Sorcery"), granted to the effect's controller, lasting until that player's next turn
  (removed at their untap step, CR 611.2). Sourceless (the Forge Effect belongs to no permanent),
  like Veil of Summer / Roiling Vortex turn-long grants.
- **The lock is enforced at every cast window** — the hand-cast path is primary; the flashback,
  escape, graveyard-permission, and impulse-from-exile cast paths are gated too, so an opponent
  cannot flash back / escape / impulse-cast an instant on your turn either.

## Tests
- Isolation (test_harness, `--play` + seat keys):
  - **Static lock (negative):** Teferi on A, B holding Lightning Bolt with an untapped Mountain.
    A casts Bolt on A's turn; B receives **no** response window offering the instant (its only
    legal action is Pass, so the window is auto-skipped). A no-Teferi control run confirms B
    *does* get a "Cast Lightning Bolt" response window on A's turn without the lock — so the lock
    is what removes it.
  - **Static lock (positive):** on B's own First Main with an empty stack, B's menu offers
    "Cast Lightning Bolt" (a legal sorcery-speed window).
  - **+1 cast-with-flash:** A activates +1 (loyalty 4→5), log shows "may cast Sorcery spells as
    though they had flash until their next turn." On B's turn (First Main), in response to B's
    Lightning Bolt on the stack, A's non-active priority window offers "Cast Chain Lightning" (a
    sorcery) and A casts it at instant speed, dealing 3 to B.
  - **-3 bounce + draw:** A activates -3 (loyalty 4→1) targeting B's Grizzly Bears → Bears moved
    to B's hand (shows in "Known opp hand"), A draws a card (hand 7→8, library 8→7).
- CI gate: `train/ci_check.py --tier pygen,vocab,smoke` (codegen-sync, vocab coverage,
  deterministic league smoke).

## Result
implemented
