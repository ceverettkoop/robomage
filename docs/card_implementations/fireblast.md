# Fireblast

## Oracle text

Instant — {4}{R}{R} (script `ManaCost:4 R R`)

> You may sacrifice two Mountains rather than pay this spell's mana cost.
> Fireblast deals 4 damage to any target.

## Forge script

Source: pre-existing local (`bin/resources/cardsfolder/f/fireblast.txt`). Key tags:

- `S:Mode$ AlternativeCost | ValidSA$ Spell.Self | EffectZone$ All | Cost$ Sac<2/Mountain>` —
  the alternative casting cost: sacrifice two Mountains instead of paying mana.
- `A:SP$ DealDamage | ValidTgts$ Any | NumDmg$ 4` — the 4-damage effect (already supported).

## Engine work

**Mechanics added (general): sacrifice-alt-cost** — an `AlternativeCost` (`S:Mode$
AlternativeCost`) whose payment is sacrificing N permanents matching a filter (CR 118.9 —
alternative costs; CR 601.2f/g — paying costs while casting). Previously the alt-cost parser
handled PayLife / PayEnergy / ExileFromHand / Return / ExileFromGrave / mana, but not
`Sac<N/Type>`, and `pay_alternate_cost` paid only mana + life.

- `src/components/carddata.h` — `AltCost::sac_cost_count` added (the N of `Sac<N/Type>`;
  the pre-existing `sac_cost_spec` holds the Type filter).
- `src/parse.cpp` `parse_alt_cost_tokens` — parses `Sac<N/Type>` (count before `/`, filter
  after, trailing `;`/`/label` stripped), mirroring the Sac grammar already used by
  `parse_activation_cost`.
- `src/systems/state_manager_actions.cpp` `can_afford_alt` — the alt cost is offered only when
  the caster controls ≥ N permanents matching the filter (via the shared
  `controlled_permanents_matching`).
- `src/classes/game.h` — new `PendingCast::Step::ALT_SAC` and counter `alt_sac_done`.
- `src/action_processor.cpp` `run_cast_flow` — new suspendable `ALT_SAC` step (inserted between
  `ALT_RETURN` and `GIFT`): one `SACRIFICE_PERMANENT` pick per required permanent, the menu of
  matching controlled permanents re-derived each pass. Modeled on the existing `ALT_RETURN` /
  deferred flashback-sacrifice (`DEF_SAC`) machinery.

The 4-damage effect uses the existing `DealDamage` ability handler.

## Behavioral decisions

- Like the other alt-cost components (pitch/return), the sacrifice is paid **before** targets
  are chosen — the pre-existing alt-cost ordering kept for byte-compatible replays. Rules-wise
  costs are locked after targets (CR 601.2c before 601.2f/g), but the outcome is identical here.
- `sac_cost_count` is a new field distinct from the flashback path, which always sacrifices
  exactly one and does not read it — so no existing card's behavior changes.

## Tests

Isolation (`train/test_harness.py`, seed 1):

- 2 Mountains on A's battlefield + Fireblast in hand, insufficient mana for {4}{R}{R}. Only
  "Cast Fireblast (alternate cost)" offered → sacrificed both Mountains (→ graveyard) → 4 damage
  to Grizzly Bears → Bears destroyed; Fireblast to A's graveyard. PASS.
- 1 Mountain only: "Cast Fireblast" is NOT offered (can_afford_alt false, normal cost also
  unaffordable). PASS.

CI gate: `ci_check.py --tier pygen,vocab,smoke` (run once after both cards).

## Result

implemented
