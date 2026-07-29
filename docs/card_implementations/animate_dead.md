# Animate Dead

```
Name:Animate Dead
ManaCost:1 B
Types:Enchantment Aura
K:Enchant:Creature.inZoneGraveyard:creature card in a graveyard
T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | IsPresent$ Card.StrictlySelf | Execute$ TrigReanimate | ...
SVar:TrigReanimate:DB$ ChangeZone | Origin$ Graveyard | Destination$ Battlefield | Defined$ Enchanted | RememberChanged$ True | GainControl$ True | SubAbility$ DBAnimate
SVar:DBAnimate:DB$ Animate | Defined$ Self | Keywords$ Enchant:Creature.IsRemembered:... | RemoveKeywords$ Enchant:Creature.inZoneGraveyard:... | Duration$ Permanent | SubAbility$ DBAttach
SVar:DBAttach:DB$ Attach | Defined$ Remembered | SubAbility$ DBDelay
SVar:DBDelay:DB$ DelayedTrigger | Mode$ ChangesZone | ValidCard$ Card.Self | Origin$ Battlefield | Execute$ TrigSacrifice | RememberObjects$ RememberedLKI | ...
SVar:TrigSacrifice:DB$ SacrificeAll | Defined$ DelayTriggerRememberedLKI
T:Mode$ ChangesZone | Origin$ Battlefield | Destination$ Any | ValidCard$ Card.Self | Execute$ DBCleanup | Static$ True
S:Mode$ Continuous | Affected$ Creature.EnchantedBy | AddPower$ -1 | Description$ Enchanted creature gets -1/-0.
Oracle:Enchant creature card in a graveyard\nWhen Animate Dead enters, if it's on the battlefield, it loses "enchant creature card in a graveyard" and gains "enchant creature put onto the battlefield with Animate Dead." Return enchanted creature card to the battlefield under your control and attach Animate Dead to it. When Animate Dead leaves the battlefield, that creature's controller sacrifices it.\nEnchanted creature gets -1/-0.
```

**Oracle:** Enchant creature card in a graveyard. When Animate Dead enters, if it's on the
battlefield, it loses "enchant creature card in a graveyard" and gains "enchant creature put onto
the battlefield with Animate Dead." Return enchanted creature card to the battlefield under your
control and attach Animate Dead to it. When Animate Dead leaves the battlefield, that creature's
controller sacrifices it. Enchanted creature gets -1/-0.

Vocab index: **348** (`src/card_vocab.h`).

Forge script source: pre-existing local script `bin/resources/cardsfolder/a/animate_dead.txt`
(unchanged — card scripts are never edited).

## Mechanics added (general): `aura-enchant-graveyard-card`

The one genuinely new mechanic is **casting an Aura whose Enchant restriction names a creature
CARD in a graveyard** (`K:Enchant:Creature.inZoneGraveyard`, CR 303.4). Everything else — the
reanimate → attach → -1/0 static → leaves-the-battlefield sacrifice spine — reuses the handlers
built for **Pre-War Formalwear** (`docs/card_implementations/pre_war_formalwear.md`).

### 1. Aura targets a creature card in a graveyard

An Aura's cast-time target search normally scans battlefield permanents. Both the cast-legality
check (`state_manager_actions.cpp`) and the actual target pick (`action_processor.cpp`, the
`AURA_TARGET` cast step) build a transient targeting `Ability` from `CardData::enchant_filter`.
When that filter contains `inZoneGraveyard`, the transient ability now sets
`Ability::target_in_graveyard = true` (shared helper `enchant_targets_graveyard()` in
`src/game_queries.h`), which routes both `build_valid_targets` and `is_legal_target`
(`action_processor.cpp` / `components/ability.cpp`) through their existing graveyard branches —
so the aura offers, and can only be cast on, a creature card sitting in **either** graveyard.

### 2. `Defined$ Enchanted` threads the chosen graveyard card to the ETB reanimation

The enchant target chosen at cast is recorded in `Game::pending_aura_target[aura]`. For a normal
aura, `apply_permanent_components` attaches the aura to that object once the aura's `Permanent` is
created. Here the object is a graveyard card (no `Permanent`), so:

- `state_manager_statics.cpp` (finalize) now **keeps** the `pending_aura_target` entry when the
  enchanted object is not a `Permanent` (rather than erasing it): the aura enters **unattached**,
  and the retained entry both marks the reanimation-pending window and lets `Defined$ Enchanted`
  find the card.
- `state_manager.cpp` — the "Aura attached to an illegal object" state-based action (CR 704.5n)
  **skips** any aura still in `pending_aura_target`, so the unattached Animate Dead is not binned
  before its ETB trigger resolves.
- `effects::change_zone` (`effect_change_zone.cpp`) gets a new **`Defined$ Enchanted`** branch: it
  resolves the enchanted object from `pending_aura_target[source]` (fallback to a live attach
  link), moves it to the battlefield, honors `GainControl$ True` (enters under the aura
  controller's control — the whole point of reanimating from any graveyard) and `RememberChanged$
  True`, then erases the pending marker. The chained `DB$ Attach | Defined$ Remembered` re-attaches
  via the existing `pending_attach` deferred-attach path (the same one Pre-War Formalwear uses when
  the reanimated creature's `Permanent` is created on the following state-based pass).
- **`GainControl$ True`** was previously unparsed (an "unless GainControl$ True" comment existed but
  no field). Added `Ability::gain_control` and its parse (`GainControl$`), honored by the new
  branch. General to any ChangeZone-to-battlefield that gains control.

### 3. Leaves-the-battlefield sacrifice (general fix to the ChangesZone delayed trigger)

`DBDelay` registers a `Mode$ ChangesZone | ValidCard$ Card.Self` delayed trigger that fires when
**Animate Dead itself** leaves the battlefield and then sacrifices the **remembered** creature.
The existing `Mode$ ChangesZone` delayed-trigger handler (`effect_delayed_trigger.cpp`) only
supported Searing Blood's shape — watch the *remembered target* leave — so it watched the wrong
object and never carried the remembered objects for the fire ability. Generalized:

- `DelayedTriggerParams::valid_card` now stores `ValidCard$`. `Card.Self` watches the trigger's own
  `source`; any other value keeps the Searing-Blood behavior (watch the remembered target).
- When `RememberObjects$ RememberedLKI` is present, the fire ability's
  `restore_remembered_exiled_with` (and `DelayedTrigger::remembered_objects`) are populated so
  `Defined$ DelayTriggerRememberedLKI` restores exactly the animated creature at fire time.
- `effects::sacrifice_all` (`effect_sacrifice_all.cpp`) gained a `Defined$ Remembered` path:
  sacrifice the remembered object(s) still on the battlefield, instead of a `ValidCards$` filter
  scan. (A bare `SacrificeAll` with an empty filter previously matched nothing.)

### 4. `DB$ Animate` keyword swap — intentional no-op

`DBAnimate` "swaps" the aura's Enchant keyword text from "creature card in a graveyard" to "the
creature put onto the battlefield with this." This is cosmetic to the engine: the aura is already
anchored to the returned creature through `equipped_to` (set by the Attach), and the CR 704.5n
state-based action reads that attachment link, **not** the Enchant filter string. The `DB$ Animate`
handler (`effect_animate.cpp`) already ignores `Keywords$` / `RemoveKeywords$` (no `Types$`/`PT$`,
so it is a no-op); those two params were added to `parse.cpp`'s `ignored_keys` so no spurious
"unrecognized param" warning is emitted. This falls under the CLAUDE.md "ignore an irrelevant tag"
allowance — the tag is not repurposed, just harmlessly skipped.

## Files touched (engine)

- `src/game_queries.h` — `enchant_targets_graveyard()` helper.
- `src/systems/state_manager_actions.cpp`, `src/action_processor.cpp` — set `target_in_graveyard`
  on the transient enchant-target ability.
- `src/components/ability.h` / `src/parse.cpp` — `gain_control` field + `GainControl$` parse.
- `src/effects/effect_change_zone.cpp` — `Defined$ Enchanted` branch.
- `src/systems/state_manager_statics.cpp` — retain `pending_aura_target` for a graveyard-card
  enchant target.
- `src/systems/state_manager.cpp` — skip the unattached-aura SBA during reanimation.
- `src/components/ability_params.h` / `src/effects/effect_delayed_trigger.cpp` — `valid_card` +
  watch-Self / restore-remembered on the ChangesZone delayed trigger.
- `src/effects/effect_sacrifice_all.cpp` — `Defined$ Remembered` sacrifice path.
- `src/parse.cpp` — ignore cosmetic `Keywords$` / `RemoveKeywords$` on `DB$ Animate`.
- `src/card_vocab.h` (+ regenerated `train/card_costs.py`, `src/gen/card_costs_gen.h`).

Relevant rules: CR 303 (Auras), 303.4 (enchant a card in a graveyard), 110.2a (control on entry),
603.6/603.7 (triggered / delayed triggered abilities), 704.5n (aura on illegal object).

## Behavioral decisions

- Animate Dead can enchant a creature card in **either** graveyard and reanimates it under **your**
  control (`GainControl$ True`), verified from the opponent's graveyard.
- The enchanted creature gets **-1/0** from the `EnchantedBy` continuous static while attached.
- When Animate Dead leaves the battlefield, the animated creature's controller sacrifices it.

## Tests (`train/test_harness.py`)

- **Reanimate from own graveyard:** Animate Dead in hand, two Swamps in play, Grizzly Bears in A's
  graveyard. Cast offers Grizzly Bears as the enchant target → "casts Animate Dead enchanting
  Grizzly Bears" → "Grizzly Bears is moved to the battlefield" → "Equipment attached." → Grizzly
  Bears shows **[1/2]** (2/2 with -1/0) on A's battlefield, `eq:Animate Dead`, and A attacks with
  it. Delayed sacrifice trigger registered.
- **Reanimate from opponent's graveyard (GainControl):** same, Grizzly Bears in **B's** graveyard,
  target `Grizzly Bears@opp`. The creature enters under **A's** control (shown on A's battlefield),
  1/2, attached.
- **Leaves-the-battlefield sacrifice:** after reanimating, B casts Abrupt Decay destroying Animate
  Dead. "Animate Dead is destroyed" → "Delayed trigger fires." → "Player A sacrifices Grizzly
  Bears." Both end in A's graveyard; the -1/0 is gone (aura left).
- **CI gate:** `ci_check.py --tier pygen,vocab,smoke` clean (no errors, no draws).

## Result

Implemented. New general mechanic `aura-enchant-graveyard-card` (Aura targeting a creature card in
a graveyard + `Defined$ Enchanted` reanimation); the reanimate/attach/-1-0/leaves-sacrifice spine
reuses Pre-War Formalwear's handlers, plus general fixes to `GainControl$`, the ChangesZone delayed
trigger (watch-Self + remembered restore), and `SacrificeAll Defined$ Remembered`. Verified
end-to-end.
