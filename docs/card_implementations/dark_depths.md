# Dark Depths

**Vocab index:** 334

## Oracle text

> Dark Depths enters the battlefield with ten ice counters on it.
> {3}: Remove an ice counter from Dark Depths.
> When Dark Depths has no ice counters on it, sacrifice it. If you do, create Marit Lage, a
> legendary 20/20 black Avatar creature token with flying and indestructible.

## Forge script

Source: pre-existing local (`bin/resources/cardsfolder/d/dark_depths.txt`). Key tags:

- `K:etbCounter:ICE:10` — enters with ten ICE counters (already handled by the engine).
- `A:AB$ RemoveCounter | Cost$ 3 | CounterType$ ICE | CounterNum$ 1` — the {3} counter-removal
  activated ability (already handled).
- `T:Mode$ Always | TriggerZones$ Battlefield | IsPresent$ Card.Self+counters_EQ0_ICE |
  Execute$ TrigToken` — the state-triggered ability (the new mechanic).
- `SVar:TrigToken:AB$ Token | TokenScript$ marit_lage | TokenOwner$ You |
  Cost$ Mandatory Sac<1/CARDNAME>` — sacrifice the source (mandatory) and create the token.

Token script `marit_lage` (`bin/resources/tokenscripts/marit_lage.txt`) already present: a
legendary 20/20 black Avatar with Flying and Indestructible.

## Engine work

**Mechanics added (general): `state-trigger-always` (Mode$ Always, CR 603.8), plus
`Cost$ Mandatory Sac` and a generalized counter present-condition comparator.**

- `src/components/ability.h` — new `bool trigger_state_condition` flag: a state-triggered ability
  fires off a game STATE (its IsPresent$), not an event.
- `src/parse.cpp`
  - trigger-mode parser: `Mode$ Always` → sets `trigger_state_condition` (and `trigger_only_self`
    so the source is the permanent whose state is examined). `parse_triggered_abilities` now keeps
    an ability with `trigger_state_condition` even though `trigger_on == 0`, and the Execute-SVar
    copy block carries `trigger_state_condition` onto the resolved effect ability.
  - `parse_activation_cost`: recognizes the `Mandatory` cost token (`Cost$ Mandatory Sac<...>`) →
    sets `ability.mandatory`, and prevents the word from being mis-parsed as a mana symbol.
- `src/systems/state_manager_triggers.cpp` — new state-trigger scan in
  `check_triggered_abilities`, run every SBA pass (outside the event-batch guard). For each
  battlefield permanent's `trigger_state_condition` ability it evaluates the IsPresent$ condition
  and fires it once when the condition becomes true, latching it in `Permanent::state_triggers_armed`
  so it does not re-fire until the condition goes false and true again (CR 603.8). The fired trigger
  goes into the same APNAP `pending` queue as event triggers.
- `src/components/permanent.h` — `std::set<std::string> state_triggers_armed` per-permanent latch
  (keyed by a stable signature, snapshot-safe; resets naturally when the permanent leaves).
- `src/systems/state_manager_actions.cpp` — generalized the `Card.Self+counters_<OP><N>_<TYPE>`
  present condition to any comparator (was `GE`-only), so `counters_EQ0_ICE` ("has no ice
  counters") evaluates via `compare_svar`.
- `src/components/ability.cpp`
  - the `TRIGGERED && sac_self` resolution branch now honors `mandatory`: a `Cost$ Mandatory Sac`
    sacrifices the source unconditionally (no optional decline prompt), while the non-mandatory
    form keeps The Fantasticar's "you may sacrifice" prompt.
  - the resolution-time intervening-if re-check is skipped for a `trigger_state_condition` ability:
    the IsPresent$ is the trigger condition (checked when firing), not an intervening-if — so
    Marit Lage is still created even though the just-performed sacrifice makes "no ice counters"
    read false (the source has left the battlefield).

## Behavioral decisions

- The sacrifice is modeled as a mandatory part of the triggered ability's resolution (the Forge
  `Cost$ Mandatory Sac<1/CARDNAME>`), consistent with the Oracle "sacrifice it. If you do, ...".
  The "if you do" clause is honored: if the source is already gone when the ability resolves, no
  sacrifice and no token.
- The state trigger is latched per-permanent so a Dark Depths sitting at 0 ice counters with its
  sacrifice trigger already on the stack does not queue a second copy on every SBA pass.

## Tests

Isolation (test_harness `--play` with seat keys), Dark Depths preset on A's battlefield with
enough Islands to pay the ten `{3}` activations:

- Preset Dark Depths enters with `[ice:10]` (ETB counter confirmed).
- `{3}` RemoveCounter reduces the count (10 → 9 per activation).
- Removing all ten counters drives ice to 0 → the Mode$ Always state trigger fires → Dark Depths
  is sacrificed (mandatory, no decline) → **Marit Lage [20/20] [Flying, Indestructible]** token is
  created under A's control. It then attacks for 20 and reduces Player B to lethal.

This end-to-end line is also exercised, more compactly, by the Thespian's Stage combo (a copy of
Dark Depths enters with zero ice counters and triggers immediately) — see
`thespians_stage.md`.

CI gate: `ci_check.py --tier pygen,vocab,smoke` after all three cards.

**Result: implemented.**
