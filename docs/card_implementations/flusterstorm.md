# Flusterstorm  (vocab index 291)

## Oracle text
Counter target instant or sorcery spell unless its controller pays {1}.

Storm (When you cast this spell, copy it for each spell cast before it this turn. You may choose
new targets for the copies.)

(Instant, mana cost {U}.)

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/f/flusterstorm.txt` (not edited).
- Key tags:
  - `A:SP$ Counter | TargetType$ Spell | ValidTgts$ Instant,Sorcery | UnlessCost$ 1`
    → counter target instant/sorcery spell unless its controller pays {1} (the
    counter-unless-pay base spell, already supported — `run_unless_loop` / `effects::counter`,
    ActionCategory `PAY_UNLESS`).
  - `K:Storm` → the Storm keyword (CR 702.40), the new general mechanic.
- Tags parsed as written; no category was retagged. `TgtPrompt$`/`SpellDescription$` are cosmetic.

## Engine work (general, keyed on the K:Storm keyword)

Storm (CR 702.40a) is a triggered ability that functions on the stack: "When you cast this spell,
copy it for each other spell that was cast before it this turn. If the spell has any targets, you
may choose new targets for any of the copies." It is implemented as a **general** keyword handler
reusable by any future Storm card — no Flusterstorm-specific behavior.

### Keyword → synthesized self-cast trigger (`src/parse.cpp`)
- `K:Storm` is parsed in the keyword loop (alongside Offspring/Replicate) into a synthesized
  `Ability` with `ability_type = TRIGGERED`, `category = "Storm"`, `trigger_on =
  Events::SPELL_CAST`, `trigger_only_self = true` (ValidCard$ Card.Self — fires for the cast spell
  itself while it sits on the stack), `valid_tgts = "N_A"` (the trigger takes no target; each copy
  chooses its own), `mandatory = true`. Mirrors the Offspring keyword's trigger synthesis.

### Storm count locked in at trigger-fire time (`src/systems/state_manager_triggers.cpp`)
- The existing self-cast SPELL_CAST trigger loop (`StateManager::check_triggered_abilities`) picks
  up the synthesized trigger. New static helper `storm_count_this_turn(game)` computes the storm
  count = (sum of **both** players' `Player::spells_cast_this_turn`) − 1. The cast is recorded
  (per-player counters bumped, `action_processor.cpp` ~line 1881) *before* `SPELL_CAST` fires, so
  the both-player total minus the storm spell itself is the number of spells cast before it this
  turn by either player (CR 702.40a). The count is snapshotted into the trigger instance's
  `amount` at fire time — spells cast in **response** to the storm trigger come after this spell
  and must not inflate the count. Copies are put on the stack (not cast), so they never increment
  the counters and never inflate later storm counts.

### Storm resolution → N copies on the stack (`src/effects/effect_storm.cpp`, new)
- `EffectKind::Storm` registered in `effect_kind.{h,cpp}`, `effect_table.cpp`, declared in
  `effects.h`. Handler `effects::storm` puts `ab.amount` copies of the source spell on the stack
  via the **shared** `copy_spell_on_stack` (`src/effects/effect_copy_spell.cpp`) — the same
  primitive Replicate uses. Each copy is not cast (pays no costs, fires no cast triggers — so a
  copy storms nothing further, CR 707.10), is a spell on the stack, and its controller may choose
  new targets (CR 702.40a / 707.10c) through the normal `select_target` path. A copy with no legal
  target is simply not created (handled inside `copy_spell_on_stack`). The copies sit above the
  original storm spell and resolve first, each independently (for Flusterstorm: each copy counters
  its chosen target unless that target's controller pays {1}).

## Behavioral decisions (made in lieu of asking a human)
- **Storm counts BOTH players' spells** cast before the storm spell this turn (CR 702.40a), not
  just the caster's — verified by the cross-player test below.
- **Copies are put on the stack, not cast** (CR 707.10): they pay no costs, do not re-trigger
  Storm, and cease to exist when they leave the stack (`Spell::is_copy`).
- **An original/copy whose target is gone fizzles** (CR 608.2b): when an earlier copy counters the
  shared target, the later copy/original finds no legal target on resolution and fizzles — observed
  and correct.
- **"Storm count zero" is unreachable for Flusterstorm with a legal target.** Flusterstorm must
  target an instant/sorcery *spell on the stack*; any such spell was itself cast this turn before
  Flusterstorm, so the storm count is ≥ 1 whenever Flusterstorm has a legal target. The
  `amount == 0` branch (make no copies) is still guarded in `effects::storm`; it is simply not
  reachable by this card in a single turn. No simplification of the rule was made — the count is
  computed correctly for the 0 case, there is just no board on which Flusterstorm casts with it.

## Tests (test_harness, seed 1)
- **(a) Storm count 1:** A casts Lightning Bolt, then (in response) Flusterstorm targeting that
  Bolt → "Resolving ability (category: Storm, amount: 1)"; exactly **1** copy ("Player A copies
  Flusterstorm"); the copy counters Lightning Bolt (controller declined to pay {1} → "Lightning
  Bolt is countered"), the copy "ceases to exist", and the original Flusterstorm fizzles on its
  now-illegal target. PASS.
- **(b) Storm count 2:** A casts Brainstorm, Lightning Bolt, then Flusterstorm → "Storm, amount:
  2"; exactly **2** copies created. PASS.
- **(c) Cross-player count (CR 702.40a, both players):** A casts Lightning Bolt, B responds with
  Lightning Bolt, then A casts Flusterstorm → "Storm, amount: 2" (A's Bolt + B's Bolt), even though
  A cast only 2 of the 3 spells. Confirms both players' casts are counted. PASS.
- **Regression (--scripted full games, seeds 1, 2, 3, 7, 11):** mirror blue/red deck (4× Flusterstorm
  + Brainstorm / Ponder / Lightning Bolt / Counterspell + Grizzly Bears / Air Elemental) — all games
  decisive (no draws), zero non-fatal errors. Seed 1 exercised Storm in real scripted play
  ("Storm, amount: 1" → "Player B copies Flusterstorm").

## Result
implemented
