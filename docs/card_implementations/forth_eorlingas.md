# Forth Eorlingas!

```
Name:Forth Eorlingas!
ManaCost:X R W
Types:Sorcery
A:SP$ Token | TokenAmount$ X | TokenScript$ r_2_2_human_knight_trample_haste | TokenOwner$ You | SubAbility$ DBEffect
SVar:DBEffect:DB$ Effect | Triggers$ TrigDamage
SVar:TrigDamage:Mode$ DamageAll | ValidSource$ Creature.YouCtrl | ValidTarget$ Player | CombatDamage$ True | Execute$ TrigMonarch
SVar:TrigMonarch:DB$ BecomeMonarch
SVar:X:Count$xPaid
```

**Oracle:** Create X 2/2 red Human Knight creature tokens with trample and haste.
Whenever one or more creatures you control deal combat damage to one or more players
this turn, you become the monarch.

Vocab index: **207** (`src/card_vocab.h`).

## Behavior

`{X}{R}{W}` sorcery. On resolution:

1. Create X 2/2 red Human Knight tokens with trample and haste (token script
   `bin/resources/tokenscripts/r_2_2_human_knight_trample_haste.txt`, already present).
2. Create an until-end-of-turn floating triggered ability:
   "Whenever one or more creatures you control deal combat damage to one or more players
   this turn, you become the monarch." (CR 603.7e-style "this turn" delayed/floating trigger.)

Becoming the monarch follows **CR 725**: the monarch draws an extra card at the beginning of
their end step, and whenever a creature deals combat damage to the monarch, that creature's
controller becomes the monarch.

## Two reusable pieces implemented

### 1. General Monarch subsystem (CR 725)

Monarch state is internal game state — **deliberately not added to the obs/state vector**
(no `STATE_SIZE`/`OBS_SIZE`/layout change).

- `Game::monarch_entity` (`src/classes/game.h`) — the current monarch's player entity.
  Sentinel `MAX_ENTITIES` = no monarch (none until an effect makes a player the monarch).
- `Game::set_monarch(Entity)` (`src/classes/game.cpp`) — makes a player the monarch; the
  previous monarch ceases to be the monarch (725.3); no-op if already the monarch.
- `DB$ BecomeMonarch` → `EffectKind::BecomeMonarch` → `effects::become_monarch`
  (`src/effects/effect_become_monarch.cpp`): sets the monarch to the ability's controller.
- The monarch's two **sourceless inherent triggered abilities** (725.2) are produced directly
  from drained events in `StateManager::check_triggered_abilities`
  (`src/systems/state_manager_triggers.cpp`), since they have no source object and are not card
  abilities:
  - On `END_STEP_BEGAN` for the monarch → a `Draw 1` ability (controller = monarch, `target` =
    the monarch player so `effect_draw` draws for them). The monarch draws an **extra** card.
  - On `COMBAT_DAMAGE_TO_PLAYER` where the damaged player is the monarch → a `BecomeMonarch`
    ability controlled by the damaging creature's controller, so the title is stolen.

These hooks are general: any future monarch card works by setting the monarch via
`DB$ BecomeMonarch` (or `Game::set_monarch`).

### 2. General floating triggered ability — `DB$ Effect | Triggers$ <SVar>`

A transient until-end-of-turn floating triggered ability.

- Parsing: `parse_svar_ability` (`src/parse.cpp`) handles `Triggers$` on a `DB$ Effect` by
  parsing each named trigger SVar through `parse_one_trigger` (the same parser used for a card's
  `T:` line) and storing the result on `Ability::effect_floating_triggers`
  (`src/components/ability.h`).
- Registration: the `DB$ Effect` handler `effects::grant_cast`
  (`src/effects/effect_grant_cast.cpp`) copies each parsed trigger, binds its controller, and
  pushes it into `Game::floating_triggers`.
- Firing: `check_triggered_abilities` scans `Game::floating_triggers` against drained events
  (alongside delayed triggers) and queues matches through the normal APNAP trigger placement.
- Expiry: `Game::floating_triggers` is cleared at the cleanup step (`src/classes/game.cpp`), so a
  floating "this turn" trigger lasts only its turn of creation.

The one combat-damage floating trigger needed here is parsed from
`Mode$ DamageAll | ValidSource$ Creature.YouCtrl | ValidTarget$ Player | CombatDamage$ True`:
a new branch in `parse_one_trigger` maps it to `COMBAT_DAMAGE_TO_PLAYER` with the flag
`Ability::trigger_damage_source_youctrl`, matched at fire time by checking the damaging
creature (the event's `ENTITY`) is controlled by the trigger's controller. It fires once per
combat-damage batch ("one or more creatures … one or more players").

### Side fix — top-level `TokenAmount$ X` (`X = Count$xPaid`)

`parse_abilities` did not route a top-level ability's `Count$xPaid` amount SVar into
`dynamic_amount_expr` (only `parse_svar_ability` did, for sub-abilities), so `SP$ Token |
TokenAmount$ X` fell back to the single-token default. Added `xPaid` to the runtime-expression
branch in `parse_abilities` (`src/parse.cpp`) so the spell creates exactly X tokens.

## Test evidence (`train/test_harness.py`, `--play`)

- **X tokens, haste, trample:** casting with X=2 creates two 2/2 Human Knight tokens; they attack
  the turn they enter (haste — shown attacking while `SICK`) and deal combat damage; trample
  keyword present (`K:Trample` on the token script).
- **Become monarch on combat damage:** after the tokens deal combat damage to the opponent, the
  floating trigger fires and "Player A becomes the monarch."
- **End-step extra draw:** "The monarch draws a card at the beginning of their end step" — the
  monarch draws an additional card.
- **Steal on being hit:** while Player A is the monarch, an opponent creature (Street Wraith)
  dealing combat damage to A makes "Player B becomes the monarch," and B then draws the extra card
  at B's end step.
- **Floating trigger expires EOT:** the floating trigger is created and fires only on the turn it
  was created (turn 1); later monarch changes come solely from the inherent CR 725.2 steal
  trigger, never from the floating trigger re-firing.

Scripted full-game regression (`delver`/`mav`/`doomsday`, several seeds) runs clean — no
non-fatal errors, real winners.
