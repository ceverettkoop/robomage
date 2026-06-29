# Tamiyo, Inquisitive Student // Tamiyo, Seasoned Scholar

uid: `tamiyo_inquisitive_student_tamiyo_seasoned_scholar` (one combined transforming-DFC script)
vocab: 298 (front) / 299 (back)

## Oracle

**Front — Tamiyo, Inquisitive Student** — `{U}` Legendary Creature — Moonfolk Wizard, 0/3
- Flying
- Whenever Tamiyo, Inquisitive Student attacks, investigate. (Create a Clue token. It's an
  artifact with "{2}, Sacrifice this artifact: Draw a card.")
- Whenever you draw your third card in a turn, exile Tamiyo, then return her to the battlefield
  transformed under her owner's control.

**Back — Tamiyo, Seasoned Scholar** — Legendary Planeswalker — Tamiyo, loyalty 2, colors green/blue
- [+2]: Until your next turn, whenever a creature an opponent controls attacks you or a
  planeswalker you control, it gets -1/-0 until end of turn.
- [-3]: Return target instant or sorcery card from your graveyard to your hand. If it's a green
  card, add one mana of any color.
- [-7]: Draw cards equal to half the number of cards in your library, rounded up. You get an
  emblem with "You have no maximum hand size."

## Forge source

Pre-existing local combined script
`bin/resources/cardsfolder/t/tamiyo_inquisitive_student_tamiyo_seasoned_scholar.txt` (DFC, both
faces under one file). Clue token script pre-existing at
`bin/resources/tokenscripts/c_a_clue_draw.txt`. Key tags:

- Front: `T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigInvestigate` → `DB$ Investigate`;
  `T:Mode$ Drawn | ValidCard$ Card.YouCtrl | Number$ 3 | Execute$ TrigTransform` → the
  exile-and-return-transformed chain (`ChangeZone Battlefield→Exile RememberChanged$`, then
  `ChangeZone Defined$ Remembered Exile→Battlefield Transformed$ True`).
- Back: `AB$ Effect | Cost$ AddCounter<2/LOYALTY> | Triggers$ TrigAttack | Duration$ UntilYourNextTurn`;
  `TrigAttack:Mode$ Attacks | ValidCard$ Creature.OppCtrl | Attacked$ You,Planeswalker.YouCtrl |
  TriggerZones$ Command | Execute$ TamiyoPump`; `TamiyoPump:DB$ Pump | Defined$
  TriggeredAttackerLKICopy | NumAtt$ -1`. `AB$ ChangeZone | Cost$ SubCounter<3/LOYALTY> | Origin$
  Graveyard | Destination$ Hand | ValidTgts$ Instant.YouCtrl,Sorcery.YouCtrl | RememberTargets$
  True` + `DBAddMana:DB$ Mana | Produced$ Any | ConditionDefined$ Remembered | ConditionPresent$
  Card.Green`. `AB$ Draw | Cost$ SubCounter<7/LOYALTY> | NumCards$ X` (X =
  `Count$ValidLibrary Card.YouOwn/HalfUp`) + `DBEmblem:DB$ Effect | StaticAbilities$ UnlimitedHand
  | Duration$ Permanent`, `UnlimitedHand:Mode$ Continuous | Affected$ You | SetMaxHandSize$
  Unlimited`.

## Engine work

The DFC transform machinery (`src/transform.*`, the exile-and-return-transformed ChangeZone path),
planeswalker loyalty abilities (`+2`/`-3`/`-7`, once-per-turn gating, sorcery-speed), the
emblem/Effect subsystem (`cur_game.emblems`, `effect_emblem_statics`, gathered into
`g_active_statics`), GY→hand targeted return, conditional `AddMana`, `Draw`, and `Pump -1/-0` were
all already present (Ajani / Kaito / Karn / The One Ring / Forth Eorlingas!). New, **general**
mechanics added:

1. **Investigate + Clue token (CR 701.x / 122).** New `EffectKind::Investigate`
   (`effect_kind.{h,cpp}`, `effect_table.cpp`) and `effects::investigate` (`effect_token.cpp`):
   creates `amount` (default 1) Clue tokens by delegating to the shared `token()` machinery with the
   pre-existing `c_a_clue_draw` token script (which carries the `{2}, Sacrifice: Draw a card`
   ability). General over any Investigate count.

2. **"Drawn your Nth card this turn" trigger (CR 603.2).** `Ability::trigger_draw_number_eq`
   (`ability.h`), parsed from `Number$ N` on a `Mode$ Drawn` trigger (`parse.cpp`). The drawer's
   per-turn running draw ordinal is stamped onto each `PLAYER_DREW_CARD` event as `Params::AMOUNT`
   (`orderer.cpp`, off the existing `cards_drawn_this_turn` vector) so the gate fires on **exactly**
   the Nth draw even when several cards are drawn in one batch
   (`state_manager_triggers.cpp`). `ValidCard$ Card.YouCtrl` reuses the existing
   `trigger_valid_player_is_controller` (drawer == controller) gate.

3. **`SetMaxHandSize` continuous static (CR 402.2).** `StaticAbility::set_max_hand_size`
   (`static_ability.h`; `-1` = no maximum/Unlimited, `>0` = a numeric cap), parsed in
   `parse_one_static_ability` (`parse.cpp`). The cleanup-step discard check (`state_manager.cpp`)
   scans `g_active_statics` for a `set_max_hand_size` static controlled by the active player and
   skips the discard when it is Unlimited (or raises the cap), instead of the hard-coded 7.

4. **Duration-bounded TRIGGERED Effect (CR 611 / 603.7e).** The emblem/Effect subsystem previously
   hosted only STATIC abilities (Kaito) and until-end-of-turn floating triggers (Forth Eorlingas!).
   Extended so a **top-level** `AB$ Effect | Triggers$ <SVar>` resolves its trigger into
   `effect_floating_triggers` (`parse.cpp` parse_abilities post-loop — the existing handler only
   covered `DB$` sub-Effects), and `Duration$ UntilYourNextTurn` makes the registered floating
   trigger survive cleanup and lapse at the controller's next untap (`effect_grant_cast.cpp`,
   `game.cpp` UNTAP/CLEANUP). The floating-trigger scan gained a `CREATURE_ATTACKED` branch
   (`state_manager_triggers.cpp`): for each attacker controlled by an opponent of the effect's
   controller (`ValidCard$ Creature.OppCtrl`; `Attacked$ You,Planeswalker.YouCtrl` is auto-satisfied
   in the two-player engine), it fires the pump binding the attacker as the target via
   `Defined$ TriggeredAttacker(LKICopy)` (`ability.h`/`parse.cpp`). `effect_pump.cpp` short-circuits
   to the pre-bound target (no target menu) for that Defined form.

5. **`/HalfUp` library-count dynamic.** `evaluate_dynamic_amount` (`ability.cpp`) now applies a
   `/HalfUp` suffix to `Count$ValidLibrary Card.YouOwn` → `ceil(library/2)` (the -7 draw count),
   matching the existing `/HalfUp` on `Count$YourLifeTotal`.

Also fixed a **general** crash: `effects::draw` (`effect_draw.cpp`) read the drawing player off the
ability's source's `Zone`, which crashes when the source was sacrificed as part of the activation
cost before the Draw resolves (a Clue). It now falls back to the ability's stable `controller`
(CR 608.2g) when the source is gone.

## Behavioral decisions

- **Two-player simplification (CLAUDE.md scope):** `Attacked$ You,Planeswalker.YouCtrl` on the +2
  trigger is treated as auto-satisfied, since the only defender an opponent's attacker can have is
  the effect's controller (or their planeswalker). The gate is purely `ValidCard$ Creature.OppCtrl`.
- The front exile-and-return-transformed chain reuses the engine's existing in-place transform path
  (identical to Ajani) — no duplicate card is created.
- `ForgetOtherRemembered$`, `Stackable$` (emblem de-dup) and the cosmetic `Triggers$` warning were
  added to the parser's ignored-keys set (the first two are no-ops in the two-player/no-dedup model;
  `Triggers$` is genuinely handled in the post-loop).

## Tests (test_harness, sculpted temp decks) → results

- **Front Investigate + Clue:** Tamiyo attacks → "investigate" creates a Clue token; activated
  `{2}, Sacrifice: Draw a card` → token sacrificed, drew a card (no crash). ✓
- **Draw-3rd-card transform:** with Tamiyo on the battlefield, the 3rd draw of the turn (natural
  draw + Brainstorm) transformed her into Tamiyo, Seasoned Scholar (planeswalker, loyalty 2); the
  1st/2nd draws did not. ✓
- **Back +2:** activated (loyalty 2→4), created an "until your next turn" floating trigger; an
  opponent Grizzly Bears attacking got -1/-0 (2/2 → 1/2, dealt 1 instead of 2). A turn later (after
  the controller's untap) the effect had lapsed and the attacker dealt full 2. ✓
- **Back -3:** loyalty 4→1, returned a target instant/sorcery from the graveyard to hand; a GREEN
  card (Edge of Autumn) added one mana of any color (chose W), a non-green card (Brainstorm) added
  none. ✓
- **Back -7 ultimate:** loyalty 8→1, drew ceil(library/2) = 12 cards, created the emblem; the
  controller held 17 cards through its cleanup step with no forced discard. ✓
- **Regression:** scripted full games (Delver/Tamiyo deck vs green creature deck) across seeds
  1/2/3 — all produced a winner, no draws, no crashes, no non-fatal errors.

## Result

Implemented — both faces fully functional (Investigate/Clue, draw-Nth-card transform, +2
duration-bounded attack-pump trigger, -3 conditional mana, -7 HalfUp draw + no-max-hand-size emblem).
