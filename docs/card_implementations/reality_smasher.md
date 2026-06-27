# Reality Smasher  (vocab index 181)

## Oracle text
({C} represents colorless mana.)

Trample, haste

Whenever Reality Smasher becomes the target of a spell an opponent controls, counter that spell
unless its controller discards a card.

## Forge script
- Source: present in repo — `bin/resources/cardsfolder/r/reality_smasher.txt`
- Type: `Creature Eldrazi`, mana cost `4 C` (four generic + one colorless {C}), P/T 5/5.
- Key tags:
  - `K:Trample`, `K:Haste` — existing keyword abilities (CR 702.19 / 702.10).
  - `T:Mode$ BecomesTarget | ValidSource$ Spell.OppCtrl | ValidTarget$ Card.Self |
    TriggerZones$ Battlefield | Execute$ TrigCounter` — "whenever this permanent becomes the
    target of a spell an opponent controls, do TrigCounter" (CR 603.2c becomes-a-target trigger).
  - `SVar:TrigCounter:DB$ Counter | Defined$ TriggeredSourceSA | UnlessCost$ Discard<1/Card> |
    UnlessPayer$ TriggeredSourceSAController` — counter the triggering spell unless its controller
    discards a card (CR 701.5 counter / CR 701.8 discard).

No tags were retagged or repurposed; every mechanic below is keyed on the tag's intended meaning.

## Engine work
Trample and haste are already implemented keywords, so the cast cost `{C}` (colorless pip) already
works. The new, **general** engine work is the becomes-target trigger path and the discard
unless-cost. The user directed that the existing hardcoded **Ward** path
(`trigger_ward_for_targets` in `src/action_processor.cpp`) be left **completely untouched** and that
`Mode$ BecomesTarget` be built as a separate general trigger path alongside it — that is what was
done; the Ward code is unchanged.

1. **`BECAME_TARGET` event** (`src/ecs/events.h`): new `Events::BECAME_TARGET = 16` plus
   `Params::TARGET = 7`. Carries `ENTITY` = the targeting spell/ability, `PLAYER` = its controller,
   `TARGET` = the permanent that became a target.

2. **General becomes-target hook** (`src/action_processor.cpp`,
   `fire_became_target_events`): fired at the **same two points** the Ward hook fires — right after a
   spell, or an activated ability, with chosen targets is placed on the stack. For each target that
   is a battlefield permanent it emits one `BECAME_TARGET` event per (targeting object, target) pair.
   Because the targeting object is already on the stack, the trigger placed by the next SBA pass
   lands **above** it and resolves first (CR 603.3 / 603.3b APNAP), so it can counter the spell
   before the spell resolves.

3. **Trigger parse** (`src/parse.cpp`, `parse_one_trigger`): `Mode$ BecomesTarget` maps to
   `trigger_on = BECAME_TARGET`. `ValidSource$ Spell.OppCtrl` sets two new `Ability` flags
   (`trigger_source_must_be_spell`, `trigger_source_opp_ctrl`); `ValidTarget$ Card.Self` reuses the
   existing `trigger_only_self`. All three are carried across the `Execute$` SVar copy.

4. **Trigger scan** (`src/systems/state_manager_triggers.cpp`): a dedicated `BECAME_TARGET` block
   matches `ValidTarget$ Card.Self` against the event's `TARGET` (not `ENTITY`, which here is the
   targeting object — the generic `trigger_only_self`-vs-`ENTITY` check is explicitly exempted for
   this event), requires the targeting object to be a `Spell` (`ValidSource$ Spell`), and requires
   its controller to be an opponent of the source's controller (`OppCtrl`). On a match it binds
   `Defined$ TriggeredSourceSA` (the targeting spell) as the trigger's `target`, and binds
   `UnlessPayer$ TriggeredSourceSAController` to that spell's controller in `Ability::unless_payer`.

5. **`Defined$ TriggeredSourceSA` / `TriggeredSourceSAController`** (`src/parse.cpp`,
   `src/components/ability.h`): new flags `defined_triggered_source_sa` and
   `unless_payer_is_triggered_source_sa_ctrl`, plus a resolved `unless_payer` field. The reused
   `DB$ Counter` effect (`src/effects/effect_counter.cpp`) counters the bound spell — it already
   verifies the target is still on the stack and honors "can't be countered".

6. **Discard unless-cost** (`src/parse.cpp`, `src/effects/effects.h`,
   `src/components/ability.{h,cpp}`, `src/effects/effect_counter.cpp`): `UnlessCost$ Discard<N/Card>`
   parses to `unless_generic_cost = N` + `unless_cost_is_discard`. `run_unless_loop` now takes an
   `UnlessPayKind` enum (`MANA` / `LIFE` / `DISCARD`) instead of a bare `pay_as_life` bool; the new
   `DISCARD` branch (`run_discard_unless`) offers the payer "discard N card(s) (not countered)" vs
   "don't discard (countered)", suppressing the discard option when the payer holds fewer than N
   cards (CR 701.8 — must discard the full count or none). The discarded cards go to the graveyard
   (a public zone) and are recorded in the belief state. `effect_counter.cpp` selects the payer from
   `unless_payer` (falling back to the countered spell's controller, as Ward does) and the pay kind.

These are all general: any future "becomes the target of …" trigger, "counter unless discard", or
"unless its controller discards/pays" effect reuses this path, not a Reality-Smasher special case.

## Behavioral decisions (made in lieu of asking a human)
- **Opponent's spells only** (`ValidSource$ Spell.OppCtrl`): the controller's own spell targeting
  Reality Smasher does **not** trigger it. Verified (test c).
- **Trigger resolves before the targeting spell** (CR 603.3): the counter/discard choice happens
  before the spell would resolve, so a saved spell (payer discarded) then resolves normally.
- **"Unless its controller discards" payer is the opponent** (`UnlessPayer$
  TriggeredSourceSAController`), i.e. the controller of the targeting spell — not Reality Smasher's
  controller. In the two-player case this coincides with the countered spell's controller; the
  explicit `unless_payer` binding makes the rule correct in general.
- **Can't pay ⇒ countered**: an empty/too-small hand means the discard option isn't offered and the
  spell is countered (CR 701.8). Verified with a genuinely empty hand (test b).
- **Spells only, not abilities**: `ValidSource$ Spell` gates out targeted activated/triggered
  abilities; the generic hook fires for abilities too, but Reality Smasher's filter ignores them.
- **Ward left untouched** (user decision): the hardcoded `trigger_ward_for_targets` path is
  unchanged; `BecomesTarget` is a parallel general path.

## Tests
Isolation (`train/test_harness.py`, semantic `--play`, seed 1), Reality Smasher on Player A's
battlefield, Player B with Lightning Bolt:
- **(a) Discard to save the spell.** B casts Lightning Bolt targeting Reality Smasher →
  `Reality Smasher triggered`, `Resolving ability (category: Counter, amount: 0)`,
  `Player B may discard 1 card to save Lightning Bolt`; B discards Island →
  `spell is not countered`, then `Dealt 3 damage to creature` (the 5/5 survives). PASS.
- **(b) Decline ⇒ countered.** Same line, B chooses "Don't discard" → `Lightning Bolt is
  countered`, no damage. PASS.
- **(b′) Empty hand ⇒ countered.** B mulliganed to a 1-card hand (the Bolt) so the hand is empty
  when the trigger resolves → only "Don't discard (spell is countered)" offered →
  `Lightning Bolt is countered`. PASS — proves the can't-pay path.
- **(c) Own spell does not trigger.** A casts Lightning Bolt targeting A's own Reality Smasher →
  no trigger, no discard prompt; Bolt resolves and deals 3 damage. PASS — `Spell.OppCtrl`.
- **(d) Trample + Haste.** Reality Smasher cast for `4 C` (paid with Wastes) attacks the turn it
  enters (haste, despite `(SICK)`), is blocked by a 0/1 Birds of Paradise →
  `Reality Smasher deals 1 damage to Birds of Paradise`, `Reality Smasher tramples 4 damage to
  Player B` (20→16), `Birds of Paradise is destroyed`. PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–3): deck `temp/rs_reg_a`
(4 Reality Smasher + 20 Wastes) vs `temp/rs_reg_b` (8 Lightning Bolt + 16 Mountain). All three
games finished decisively (Player A wins each), no draws, no non-fatal errors, no asserts/
tracebacks. Reality Smasher was drawn and cast (paying `{C}`) in real games with the engine stable;
the scripted opponent prefers to bolt face rather than its opponent's creature, so the
becomes-target trigger's full firing is proven by the isolation tests rather than the scripted
regression. Temp decks cleaned up.

## needs_review
- **BecomesTarget trigger timing** (flagged per the task): the trigger is fired from the cast/
  activation path right after targets are chosen and the object is on the stack (same seam as Ward),
  and lands above the targeting object via the normal APNAP placement. This is correct for the
  single-target spell case tested; multi-target and multi-permanent interactions (one spell, several
  becomes-target triggers) reuse the once-per-(object,target) firing rule but were not exhaustively
  tested.

## Result
implemented
