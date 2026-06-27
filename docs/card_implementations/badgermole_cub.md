# Badgermole Cub  (vocab index 175)

## Oracle text
When this creature enters, earthbend 1. (Target land you control becomes a 0/0 creature with
haste that's still a land. Put a +1/+1 counter on it. When it dies or is exiled, return it to
the battlefield tapped.)

Whenever you tap a creature for mana, add an additional {G}.

(2/2 Badger Mole, mana cost {1}{G}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/b/badgermole_cub.txt` (not edited).
- Key tags:
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self |
    Execute$ TrigEarthbend` → `TrigEarthbend:DB$ Earthbend | Num$ 1`.
  - `T:Mode$ TapsForMana | ValidCard$ Creature | Activator$ You | Execute$ TrigMana |
    TriggerZones$ Battlefield | Static$ True` → `TrigMana:DB$ Mana | Produced$ G | Amount$ 1`.
- Tags parsed as written; no category was retagged.

## Engine work (all general, keyed on each tag's intended meaning)

### Earthbend keyword action (`DB$/AB$ Earthbend | Num$ N`) — shared with Ba Sing Se
- New `EffectKind::Earthbend` (`src/effects/effect_kind.{h,cpp}`, registered in
  `src/effects/effect_table.cpp`) with handler `effects::earthbend`
  (`src/effects/effect_earthbend.cpp`). Resolution (CR-style earthbend N):
  1. The target land you control "becomes a 0/0 creature with haste that's still a land" — baked
     onto its `Permanent` via the **Animate extension points** (`animate_make_creature`,
     `animate_set_pt` + 0/0 base, `animate_added_keywords += "Haste"`, Duration Permanent) so the
     layer system reapplies it for the rest of the game.
  2. The Creature/Damage components are bootstrapped immediately
     (`effects::apply_animate_creature_bootstrap`, `src/effects/effect_animate.cpp`), and
     `Num$ N` +1/+1 counters are added with `add_counters` — so the land is N/N (not the transient
     0/0) before any state-based action runs.
  3. A "when it leaves the battlefield, return it tapped under its owner's control" delayed
     trigger is registered (see below).
- Parse: `Num$` was added to the amount-key set in `apply_param_to_ability`
  (`src/parse.cpp`); the parser defaults `ValidTgts$ Land.YouCtrl` for an Earthbend ability (the
  Forge scripts carry no `ValidTgts$`, the keyword action inherently targets a land you control).
- Target legality (`Ability::is_legal_target`, `src/components/ability.cpp`) now honours the
  `YouCtrl`/`OppCtrl` controller restriction for **battlefield** permanent targets (previously
  only graveyard/player targets were filtered by controller) — so `Land.YouCtrl` only offers your
  own lands.

### Animate land-animation wiring (completed the stubbed extension points)
- `effects::apply_animate_creature_bootstrap(Entity)` (`src/effects/effect_animate.cpp`,
  declared in `effects.h`) turns a noncreature permanent into a creature when
  `animate_make_creature` is set: adds Creature (base P/T from `animate_power`/`animate_toughness`)
  + Damage, merges `animate_added_keywords` (Haste), and re-syncs P/T from counters. Idempotent.
- `effect_animate.cpp::animate` now routes through the bootstrap when `animate_make_creature` is
  set (the type-add path is unchanged for Guide of Souls).
- `StateManager::apply_permanent_components` (`src/systems/state_manager_statics.cpp`)
  re-bootstraps the Creature on an animated land each SBA pass if it was lost (an animated land is
  a creature even though `is_creature_card` is false for a land).
- The keyword merge in `gather_active_statics` already reapplies `animate_added_keywords` each
  pass, and `recompute_pt` reads `base_power`/`base_toughness` (0/0) + the counter bonus — so the
  animated land stays N/N for the rest of the game.

### Leaves-the-battlefield "return tapped" delayed trigger (general; reused by future cards)
- `DelayedTrigger` (`src/classes/game.h`) gained `watch_entity` + `fire_on_leave_battlefield`:
  a delayed trigger that fires once when a **specific** permanent changes zone from the
  battlefield (CR 603.6e), instead of on a phase.
- `StateManager::check_triggered_abilities` (`src/systems/state_manager_triggers.cpp`) matches it
  against `CARD_CHANGED_ZONE` with origin BATTLEFIELD and `ENTITY == watch_entity`.
- The fire ability is a generic `ChangeZone | Defined$ Self | Destination$ Battlefield |
  enters_tapped`. The `defined_self` ChangeZone path (`src/effects/effect_change_zone.cpp`) now
  honours `enters_tapped` (inserts `pending_enters_tapped` before the move so the ENTERS_TAPPED
  replacement sees it). The returned card is a NEW object — a plain land (the animate fields lived
  on the old, now-destroyed Permanent), entering tapped under its owner's control.

### `TapsForMana` mana-additional trigger (`Mode$ TapsForMana | Static$ True`)
- New `Events::TAPPED_FOR_MANA` (`src/ecs/events.h`).
- Parse (`src/parse.cpp`): `Mode$ TapsForMana` + `Activator$ You` (`ValidPlayer$ You`-equivalent)
  + `Static$ True` set `trigger_taps_for_mana_static`, `trigger_valid_card_is_creature`,
  `trigger_valid_player_is_controller` on the trigger; the `Execute$ TrigMana` AddMana
  (Produced$ G, Amount$ 1) is folded onto it as the produced color/amount. The flag is preserved
  through the Execute$ metadata-restore.
- `Static$ True` means it is a mana-additional effect that does NOT use the stack (CR 605.1a /
  605.3): the stack-trigger scan skips `trigger_taps_for_mana_static` triggers, and the mana
  system resolves them immediately. `fire_taps_for_mana_triggers` (`src/mana_system.cpp`), called
  from `activate_mana_source` whenever a permanent taps for mana, scans the tapping player's
  battlefield for matching TapsForMana triggers and adds their extra mana straight to the working
  pool. It fires in BOTH commit and simulate modes so affordability (`can_pay_mana`) and the real
  payment agree on the available mana; the narrative line is emitted only on the real activation.
  `ValidCard$ Creature` is gated on the tapped source currently being a creature — an earthbended
  land that became a creature counts.

## CR references
- 605.1a / 605.3 — mana abilities and "additional mana" effects resolve as part of the
  mana-producing activation (no stack).
- 613.4 (layer 7), 613.1 — the land-animation (set base P/T, grant Haste, become a creature) is a
  continuous effect reapplied each SBA pass.
- 603.6e — a delayed triggered ability set up while another ability resolves (the return-tapped
  trigger).
- 614.1d — "enters tapped" replacement (the returned land enters tapped).
- 122.1 — +1/+1 counters.

## Behavioral decisions
- The +1/+1 counters are applied in the same resolution as the animation, so the land is N/N
  before the toughness-0 state-based action (704.5f) can fire — a 0/0 with N counters never dies.
- The returned land is a fresh object (no longer animated): it comes back as its printed self, a
  plain land, tapped, under its owner's control.
- An earthbended land that is also a creature counts as a creature for Badgermole's TapsForMana
  trigger (tapping it for mana adds the extra {G}); a plain noncreature land does not.

## Tests (test_harness.py, scripted/semantic `--play`)
- ETB earthbend 1: cast Badgermole, target a Forest → "Forest becomes a 0/0 creature with haste
  ... put 1 +1/+1 counter"; board shows `Forest [1/1]`, still a land, and it attacked the same
  turn (haste). ✓
- Dies → returns tapped: Player B Lightning Bolts the 1/1 animated Forest → "Forest is destroyed
  (lethal damage)" → "Delayed trigger fires" → "Forest is moved to the battlefield" → "Forest
  enters tapped"; board shows the returned Forest as a plain land (no P/T), tapped. ✓
- TapsForMana: with Badgermole in play, tapping Ignoble Hierarch (a creature) for {G} prints
  "Badgermole Cub adds an additional 1(G)" and the single tap (GG) pays a {1}{G} spell. ✓
- TapsForMana negative: tapping plain Forests for mana prints NO "additional" line (noncreature
  lands don't trigger). ✓
- Regression: scripted full games (4 Badgermole Cub / 3 Ba Sing Se / dorks / Forests vs a
  Mountain creature/burn deck), seeds 1–6 — decisive results both ways, no draws, no non-fatal
  errors; earthbend + TapsForMana fired in real games. ✓
