# Roiling Vortex

Vocab index **341**.

## Oracle text

> Enchantment — {1}{R}
>
> At the beginning of each player's upkeep, Roiling Vortex deals 1 damage to them.
>
> Whenever a player casts a spell, if no mana was spent to cast that spell, Roiling Vortex
> deals 5 damage to that player.
>
> {R}: Your opponents can't gain life this turn.

## Forge script

Source: pre-existing local (`bin/resources/cardsfolder/r/roiling_vortex.txt`). Key tags:

- `T:Mode$ Phase | Phase$ Upkeep | ValidPlayer$ Player | Execute$ Trig1Damage`
  → `SVar:Trig1Damage:DB$ DealDamage | Defined$ TriggeredPlayer | NumDmg$ 1`
  (unrestricted `ValidPlayer$ Player` ⇒ fires on **both** players' upkeeps; damages the
  upkeep player).
- `T:Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ Player | ValidSA$ Spell.ManaSpent EQ0 | Execute$ Trig5Damage`
  → `SVar:Trig5Damage:DB$ DealDamage | Defined$ TriggeredActivator | NumDmg$ 5`
  (fires only when the cast spell paid **no mana**; damages the caster).
- `A:AB$ Effect | Cost$ R | StaticAbilities$ STCantGain`
  → `SVar:STCantGain:Mode$ CantGainLife | ValidPlayer$ Player.Opponent`
  (turn-long life-gain prohibition on the activating player's opponents).

## Engine work

**Mechanics added (general): triggered-player, mana-spent, cant-gain-life.**

### 1. `Defined$ TriggeredPlayer` binding (the upkeep 1-damage)

The player whose event fired the trigger — a sibling of the existing
`TriggeredActivator` / `TriggeredDefendingPlayer` bindings (CR 603.x). The each-player upkeep
`Mode$ Phase | ValidPlayer$ Player` trigger fires on **both** players' upkeeps because
`ValidPlayer$ Player` (not `You`) leaves `trigger_valid_player_is_controller` false
(`src/parse.cpp:2982-2985`), and the `UPKEEP_BEGAN` event carries `PLAYER` = the active
player (`src/classes/game.cpp:194`).

- `src/components/ability.h` — `defined_triggered_player` flag + `triggered_player` (bound
  Ownership), mirroring the activator/defender fields.
- `src/parse.cpp` (`Defined$` branch, ~1540) — parse `Defined$ TriggeredPlayer`.
- `src/systems/state_manager_triggers.cpp` — `bind_triggered_player()` (recurses into
  subabilities/charm_choices like `bind_triggered_activator`), called at both trigger-fire
  sites from the event's `PLAYER` param.
- `src/game_queries.cpp` `resolve_defined_player()` — returns `ab.triggered_player`.
- `src/effects/effect_deal_damage.cpp` — routes `defined_triggered_player` through the
  Defined-player damage path (CR 119.3 / 603).

### 2. Per-cast mana-spent tracking (`ValidSA$ Spell.ManaSpent EQ<n>`, the free-spell 5-damage)

Records how much mana was actually spent to cast each spell (CR 106 / 601.2g). **This is the
field/API Lavinia, Azorius Renegade reuses.**

- **Field:** `Spell::mana_spent` (int, `src/components/spell.h`) — total mana pips paid.
  Default 0; **0 ⇒ cast for free / via an alternative cost that paid no mana** (Force of Will's
  pitch+life, Daze, a 0-cost spell like Mishra's Bauble).
- **Set at cast time** (`src/action_processor.cpp`):
  - `Game::PendingCast::mana_spent` accumulates it during the cast flow.
  - Deferred (regular / flashback / escape / impulse-normal) paths: at the `MANA_PAY` step,
    `pc.mana_spent = pc.deferred_mana_cost.size()` (the `std::multiset<Colors>` pip count,
    after any delve/improvise reduction already removed pips).
  - Alternative-cost path (`pay_alternate_cost`): `game.pending_cast.mana_spent =
    alt_mana.size()` (the floored alt mana; 0 for a pitch/life-only cost).
  - Copied onto `Spell::mana_spent` at the `FINISH` step, before the `SPELL_CAST` event fires.
- **Filter parse** (`src/parse.cpp`, trigger `ValidSA` branch): `ValidSA$ Spell.ManaSpent
  <op><n>` → `Ability::trigger_mana_spent_op` (`EQ`/`NE`/`LE`/`GE`/`LT`/`GT`) +
  `trigger_mana_spent_val`. **Also whitelisted in the single-DB Execute promotion copy**
  (`ability = effect`, ~line 3273) — trigger-line filter fields are copied field-by-field
  there, so a new one must be added or it is silently dropped (this was the one bug found in
  testing; see below).
- **Filter match** (`src/systems/state_manager_triggers.cpp`, SPELL_CAST block): compares the
  cast spell's `Spell::mana_spent` to the trigger's bound; a mismatch `continue`s (no trigger).

### 3. `Mode$ CantGainLife` life-gain prohibition (the {R} ability)

A turn-long "can't gain life" prohibition (CR 119.x), modeled on the sibling turn-long
grants (`cant_counter_spells_of`, Veil of Summer).

- **State:** `Game::cant_gain_life_this_turn` (`std::set<Zone::Ownership>`,
  `src/classes/game.h`) — players who can't gain life this turn. Cleared at cleanup
  (`src/classes/game.cpp`, CR 514.2).
- **Central check:** `player_gain_life()` (`src/game_queries.h`) consults the new out-of-line
  helper `player_cant_gain_life()` (`src/game_queries.cpp`) and no-ops the gain (no life added,
  no `life_gained_this_turn` accrual) when the player is prohibited. Every life-gain site
  routes through `player_gain_life`, so all gains obey the prohibition.
- **Granting path** (`AB$ Effect | StaticAbilities$ <SVar>`): parse resolves the named static
  SVar; a `CantGainLife` mode sets `Ability::effect_cant_gain_life`
  (`CantGainLifeScope::{OPPONENTS,YOU,ALL}` by `ValidPlayer$`). The `GrantCast` handler
  (`src/effects/effect_grant_cast.cpp`) registers the affected player(s) at resolution.
  Roiling Vortex uses `ValidPlayer$ Player.Opponent` ⇒ OPPONENTS (only the activator's
  opponent is blocked).

## Behavioral decisions

- **Upkeep trigger fires on both players' upkeeps** and damages the upkeep player, per the
  unrestricted `ValidPlayer$ Player`.
- **`ManaSpent EQ0` semantics** follow CR/Forge: a spell whose alternative cost pays no mana
  (Force of Will's pitch + 1 life, a 0-mana spell) has `mana_spent == 0` and **does** trigger
  the 5 damage; any spell that paid ≥1 mana does not. A Phyrexian pip paid with life still
  leaves its pip in the deferred cost, so such a spell counts that pip as mana spent — an
  acceptable edge case, not exercised by any vocab card.
- **CantGainLife is scoped to opponents** relative to the activating player, so the activator
  can still gain life; only opponents are blocked.

## Tests (isolation via `train/test_harness.py --play`)

1. **Upkeep 1 damage (TriggeredPlayer).** Roiling Vortex preset on A's battlefield, advance
   several turns. Result: A takes 1 on A's upkeep (20→19), B takes 1 on B's upkeep (20→19),
   and so on — **each player is damaged only on their own upkeep.** PASS.
2. **Free-spell 5 damage (ManaSpent).** A has Roiling Vortex + lands + Mishra's Bauble (0 mana)
   + Lightning Bolt (R). Result: casting Mishra's Bauble ⇒ Roiling Vortex deals 5 to A
   (19→14); casting Lightning Bolt (mana_spent=1) ⇒ **no** 5-damage trigger (only its own 3 to
   the creature). PASS. Confirms the trigger both fires on a genuinely-free spell and does
   **not** fire on a normally-paid one.
   - Verified `Spell::mana_spent` = 0 for Mishra's Bauble and 1 for Lightning Bolt (debug
     instrumentation, since removed).
3. **Cant-gain-life ({R} ability).** A and B each have a Soul Warden; A also has Roiling
   Vortex + lands. A activates `{R}` ("Player B can't gain life this turn."), then A casts
   Goblin Guide. Both Soul Wardens trigger on the entering creature. Result: **A gains 1
   (19→20, allowed); B's gain is prevented (stays at 20).** PASS. (The `GainLife` effect still
   logs its intent line for B, but B's life total is unchanged — the gain is a no-op.)

**Bug found & fixed during testing:** the trigger-line `ValidSA$ Spell.ManaSpent` filter was
initially dropped because the single-DB `Execute$` promotion (`ability = effect` in
`parse_one_trigger`) copies trigger-level fields one-by-one and did not include the new fields;
adding `trigger_mana_spent_op`/`trigger_mana_spent_val` to that copy list fixed it.

CI gate: `train/.venv/bin/python train/ci_check.py --tier pygen,vocab,smoke --smoke-games 1`.

## Result

Implemented.
