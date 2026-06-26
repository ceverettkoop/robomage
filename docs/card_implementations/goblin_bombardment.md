# Goblin Bombardment  (vocab index 110)

## Oracle text
Sacrifice a creature: Goblin Bombardment deals 1 damage to any target.

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/g/goblin_bombardment.txt`
- Type: `Enchantment`, mana cost `1 R`.
- Key tags:
  - `A:AB$ DealDamage | Cost$ Sac<1/Creature> | ValidTgts$ Any | NumDmg$ 1` — a repeatable
    activated ability whose cost is sacrificing a creature you control and whose effect is 1
    damage to any target (creature, player, or planeswalker).
  - `AI:RemoveDeck:All` — AI-hint, no engine behavior. Ignored (cosmetic).
  - `SVar:NonStackingEffect:True` — irrelevant to this card's one-shot damage effect (it does
    not create a continuous effect). Ignored (cosmetic); does not surface a warning because it
    is an `SVar`, not an ability param.

## Engine work
**None.** Every tag this card uses is already a fully-general, pre-existing handler — no new
code was added to the engine. Verified the path end to end:

1. **`Cost$ Sac<1/Creature>`** — the `Sac<.../...>` cost grammar is parsed by
   `parse_activation_cost` (`src/parse.cpp`), which stores the type spec (`Creature`) in
   `Ability::sac_cost_spec` (the `1/` quantity and the optional human label after a second `/`
   are stripped, leaving the type filter). This is the same field used by Cycling "Sac a land",
   Knight of the Reliquary, etc.
   - **Legality** (`src/systems/state_manager_actions.cpp`): an activated ability with a
     non-empty `sac_cost_spec` is only offered when
     `controlled_permanents_matching(controller, spec, ...)` is non-empty — i.e. the ability is
     unavailable with no creature to sacrifice.
   - **Payment** (`src/action_processor.cpp`): on activation, after targets are chosen, the
     player is prompted (`ActionCategory::SACRIFICE_PERMANENT`) to choose one matching creature,
     which is moved to its owner's graveyard.

2. **`ValidTgts$ Any | NumDmg$ 1`** — the `DealDamage` effect (`src/effects/effect_deal_damage.cpp`)
   already handles `Any` targets (creature / player / planeswalker) and a fixed `NumDmg`. Lethal
   damage to a creature is resolved by the normal state-based-action check.

## Behavioral decisions (made in lieu of asking a human)
- **Targets chosen before cost paid (CR 601.2 / 602.2b).** The engine selects the `DealDamage`
  target *before* prompting for the sacrifice (per the documented rule that activated abilities
  with `valid_tgts != "N_A"` pick targets, then pay costs). This is correct: an activated
  ability's targets are chosen as it is put on the stack (602.2b → 601.2c), and the sacrifice is
  a cost paid afterward (601.2h). Target legality is re-verified at resolution.
- **You may sacrifice the same creature you are pointing the damage at.** `ValidTgts$ Any`
  includes your own creatures, so a creature can be both the sacrificed cost and the damage
  target; this is legal and handled (the target check at resolution sees it already in the
  graveyard and the damage simply has no surviving creature to land on — standard "target gone"
  handling). No special-casing was needed.
- **`AI:RemoveDeck:All` and `SVar:NonStackingEffect:True` ignored.** Both are Forge AI / engine
  hints with no rules meaning for this one-shot damage ability; ignoring an irrelevant tag is
  sanctioned (CLAUDE.md), and neither changes behavior.

## Tests
Isolation (`train/test_harness.py`, pre-set battlefields, seed 1; Goblin Bombardment cast for
`1 R`, with a Dragon's Rage Channeler on the battlefield as sacrifice fodder):
- **Sacrifice a creature → 1 damage to a player.** Activate Goblin Bombardment, choose target
  Player B, then pay the cost by sacrificing Dragon's Rage Channeler. Narrative: "Player A
  sacrifices Dragon's Rage Channeler" → "Dealt 1 damage to player (now at 19 life)". DRC moves to
  the graveyard; Player B drops 20 → 19. PASS.
- **Sacrifice a creature → 1 damage kills a 1-toughness creature.** With an opposing
  Dragon's Rage Channeler (1/1) on the battlefield, activate targeting it, then sacrifice the
  controller's own DRC. Narrative: "Dealt 1 damage to creature" → "Dragon's Rage Channeler is
  destroyed (lethal damage)". Both DRCs end in their owners' graveyards. PASS.

Regression (`train/test_harness.py --scripted`, 6 games, seeds 1–6): deck `temp/gb_test` (the
`delver` UR deck with 3 Goblin Bombardment swapped in for the 3 maindeck Unholy Heat) mirror
match (both seats). All 6 games finished decisively (Player B 4, Player A 2), **no draws**, and
no non-fatal errors / asserts / tracebacks were introduced — only the pre-existing cosmetic
`WARNING: Unrecognized ability param` lines on unrelated cards (Delver, Brainstorm, Mishra's
Bauble, Cori-Steel Cutter). Goblin Bombardment was cast and resolved on the battlefield in the
transcripts. (`train.py observe` could not be used — `torch` is not installed in this
environment — so the scripted regression was run directly through the harness.)

## Result
implemented
