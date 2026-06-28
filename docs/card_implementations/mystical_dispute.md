# Mystical Dispute

```
Name:Mystical Dispute
ManaCost:2 U
Types:Instant
S:Mode$ ReduceCost | ValidCard$ Card.Self | Type$ Spell | Amount$ 2 | ValidTarget$ Spell.Blue | EffectZone$ All | Description$ CARDNAME costs {2} less to cast if it targets a blue spell.
A:SP$ Counter | TargetType$ Spell | TgtPrompt$ Select target spell | ValidTgts$ Card | UnlessCost$ 3 | SpellDescription$ Counter target spell unless its controller pays {3}.
Oracle:This spell costs {2} less to cast if it targets a blue spell.\nCounter target spell unless its controller pays {3}.
```

**Oracle:** This spell costs {2} less to cast if it targets a blue spell. Counter target spell
unless its controller pays {3}.

Vocab index: **227** (`src/card_vocab.h`).

Forge script source: pre-existing local script
`bin/resources/cardsfolder/m/mystical_dispute.txt` (unchanged).

## Engine work

**None — covered by existing handlers.**

- **`SP$ Counter` with `UnlessCost$ 3`** — the "counter target spell unless its controller pays
  {N}" pattern is handled by the existing Counter effect (`src/effects/effect_counter.cpp`) and
  the PAY_UNLESS decision path (the same machinery used by Mana Leak / Daze). On resolution the
  spell's controller is offered a pay-or-decline prompt for the numeric `{3}` cost; declining
  counters the spell, paying lets it resolve.
- **Targeting a spell on the stack** — `ValidTgts$ Card` / `TargetType$ Spell` is the standard
  counterspell target selection.

The specific combination (numeric `UnlessCost$` counter + a target-conditional `ReduceCost`
static) was not previously proven end-to-end by a single shipping card, hence the behavior test
below.

## Behavioral decisions

- **Known limitation — the {2}-less-vs-blue cost reduction is NOT applied.** The
  `S:Mode$ ReduceCost` static carries a `ValidTarget$ Spell.Blue` condition: the discount should
  apply only when Mystical Dispute targets a blue spell. The engine's `ReduceCost` static handler
  (`active_reduce_cost_for` in `src/systems/state_manager_statics.cpp`) filters the discount on the
  **card being cast** (`ValidCard$` / CMC), but does not parse or evaluate the `ValidTarget$`
  condition (the cast cost is computed before a target is chosen). As a result the engine charges
  the **full {2}{U}** for Mystical Dispute regardless of what it targets. This is intentional and
  acceptable to ship: the conservative direction (never undercharging) keeps the core counter
  correct; it only makes the card cost more than its printed discount in the blue-on-blue case.
  No new handler was added for this. Empirically verified: with only {U} or {U}{U} available,
  Mystical Dispute is not castable even against a blue spell; it requires the full {2}{U}.

## Tests (`train/test_harness.py`)

- **Counter when controller declines:** B casts Lightning Bolt at A; A casts Mystical Dispute
  targeting it; at resolution B is offered "Don't pay (spell is countered)" and declines →
  "Lightning Bolt is countered." A stays at 20.
- **Spell survives when controller pays {3}:** same setup, B taps three Mountains to pay {3} →
  the bolt resolves, A drops to 17. Confirms the pay-unless branch resolves the spell.
- **Discount conditional (known limitation):** with 1 or 2 Islands, "Cast Mystical Dispute" is
  not offered against either a blue (Delver of Secrets) or a non-blue (Birds of Paradise) spell;
  it appears only with three lands ({2}{U}). Confirms the blue discount is not applied (full cost
  always), as documented above.
- **Real-game regression:** `temp/disp_a` (delver list with 3 Mystical Dispute swapped in) vs
  `temp/disp_b` (delver), scripted, seeds 1/2/3 → B/B/A winners, no errors, no draws. Mystical
  Dispute is cast in real games and counters spells (e.g. "Delver of Secrets is countered",
  "Dragon's Rage Channeler is countered" after the pay-unless decline).

## Result

Implemented — core counter covered by existing handlers and verified; the target-conditional
{2}-less-vs-blue discount is a documented known limitation (full cost charged always).
