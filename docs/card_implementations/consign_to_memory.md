# Consign to Memory

```
Name:Consign to Memory
ManaCost:U
Types:Instant
K:Replicate:1
A:SP$ Counter | TargetType$ Spell.Colorless,Triggered | TgtPrompt$ Select target triggered ability or colorless spell | ValidTgts$ Card,Emblem | SpellDescription$ Counter target triggered ability or colorless spell.
Oracle:Replicate {1} (When you cast this spell, copy it for each time you paid its replicate cost. You may choose new targets for the copies.)\nCounter target triggered ability or colorless spell.
```

**Vocab index:** 206 (`src/card_vocab.h`).

Consign to Memory is a one-mana instant that counters a target **triggered ability or
colorless spell**, with **Replicate {1}** — an optional additional cost you may pay any number
of times as you cast it, copying the spell once per payment (the copies may choose new targets).

Two engine subsystems were missing and built here as **general, reusable** mechanics:

1. A general **Replicate** keyword (CR 702.x) — a *repeatable* optional additional cost that
   drives a real on-cast copy.
2. A general **copy-a-spell-on-the-stack** routine (CR 707.10 / 707.12) reusable by any
   copy-spell effect (storm, fork, …), with per-copy new-target selection.

The `SP$ Counter` half (countering a triggered ability or a colorless spell on the stack)
reuses the existing counter/targeting infrastructure; the `TargetType$` matcher was generalized
to handle the comma-separated OR list and the `.Colorless` qualifier.

---

## What was added

### 1. `TargetType$` OR-list + Colorless matching (`src/components/ability.cpp`)

`TargetType$ Spell.Colorless,Triggered` is an **OR list** of stack-object alternatives. The old
code matched `target_type == "Spell"` exactly (so the full `Spell.Colorless,Triggered` string
matched neither the spell branch nor honored Colorless) and only `find("Triggered")` for the
ability branch. `is_legal_target` now routes any `target_type` naming `Spell`/`Activated`/
`Triggered` through `target_type_matches_stack_object`, which:

- splits `target_type` on commas and accepts the candidate if **any** alternative matches;
- `stack_spell_alt_matches` matches a spell on the stack with the alternative's own qualifiers:
  type negations (`nonCreature` / Instant|Sorcery-only), a positive color restriction
  (`.Blue`, Red Elemental Blast), and **`.Colorless`** (the target spell must have no color —
  `effective_colors(cand).empty()`, CR 105.2c);
- `stack_ability_alt_matches` matches a standalone `Activated`/`Triggered` ability on the stack
  (spells excluded), as Stifle already did.

This preserves all prior counterspell/Stifle behavior (now flowing through one matcher) and
adds the colorless-spell case. A **colored** spell is therefore not a legal target, so Consign
is not even castable when the only stack object is a colored spell (verified).

### 2. Replicate keyword storage (`src/parse.cpp`, `src/components/carddata.h`)

`K:Replicate:<cost>` parses into `CardData::replicate_cost` (the per-instance mana) and
`CardData::has_replicate`. One per-instance cost is stored; the *count* paid is decided at cast.

### 3. Repeatable optional payment at cast (`src/action_processor.cpp`)

In the regular-cost branch of `CAST_SPELL`, after the kicker loop, a `while` loop offers a
`request_optional_yesno` "pay replicate cost again?" each iteration — folding another
`replicate_cost` worth of mana into `cost_to_pay` and incrementing a local `replicate_count`,
stopping when the next payment is unaffordable (`can_pay_mana`) or declined. This reuses the
exact optional-additional-cost infrastructure introduced for Kicker (Wastescape Battlemage).
The count is copied onto the spell as `Spell::replicate_count`.

### 4. `Spell` state (`src/components/spell.h`)

- `replicate_count` — how many times the replicate cost was paid (drives the copy effect). Does
  **not** touch the observation/state vector (purely internal cast bookkeeping).
- `is_copy` — marks a copy of a spell so it ceases to exist when it leaves the stack.

### 5. General copy-spell-on-stack routine (`src/effects/effect_copy_spell.cpp`)

`copy_spell_on_stack(Entity original, int count, Zone::Ownership controller, orderer)` (declared
in `src/action_processor.h`) creates `count` independent copies of a spell on the stack:

- copies the original's copiable characteristics (CardData + ColorIdentity), adds a `Zone(STACK)`
  and a `Spell{is_copy=true}` (carrying the original's `x_paid`/`cant_be_countered`), and copies
  the resolving `Ability`;
- a copy is **not cast** — it pays no costs, fires no cast triggers, and replicates nothing
  (its own `replicate_count` is 0);
- each copy **chooses new targets** (CR 707.12) by re-running the shared `select_target` path; a
  copy with no legal target is simply not created (`has_legal_targets` guard);
- copies are pushed on **top** of the stack, so they resolve before the original.

### 6. Wiring Replicate to the copy routine (`src/action_processor.cpp`)

Right after the cast spell is placed on the stack and its `SPELL_CAST` event fires (the moment
the Replicate reflexive trigger would resolve), if `Spell::replicate_count > 0` the cast handler
calls `copy_spell_on_stack` for that many copies.

### 7. Copies cease to exist (`src/systems/stack_manager.cpp`, `src/effects/effect_counter.cpp`)

- When an instant/sorcery **copy** resolves, the stack manager destroys it instead of moving it
  to the graveyard (CR 707.10c).
- When a copy is **countered**, `effects::counter` destroys it (joining the standalone-ability
  path) rather than sending it to a graveyard.

---

## Tests (test harness, `--play` isolation)

- **Counter a colorless spell:** opponent casts Lotus Petal (colorless artifact); Consign to
  Memory targets and counters it → both go to graveyard. ✅
- **Colored spell is not a legal target:** with only Lightning Bolt (red) on the stack, Consign
  is not a castable action (no legal target) — verified the cast is never offered. ✅
- **Counter a triggered ability:** White Orchid Phantom's ETB trigger goes on the stack; Consign
  targets and counters the triggered ability (its destroy-land effect does not happen). ✅
- **Replicate is optional and repeatable:**
  - decline → only the original (1 instance);
  - pay once → original + 1 copy (stack shows `Consign -> Consign -> target`);
  - pay twice → original + 2 copies (`Consign -> Consign -> Consign -> target`).
  Each instance runs its own target selection; copies cease to exist on resolve and never leak
  into the graveyard (only the original Consign is found there). ✅
- **Scripted regression:** full scripted games across seeds 1/3/7 with a curated Consign deck —
  no draws, no non-fatal errors, no asserts. A copy whose target was already countered fizzles
  (correct CR 608.2b behavior) and still ceases to exist.

## Rules references
- CR 702.x — Replicate (optional additional cost, copy on cast, new targets for copies).
- CR 707.10 / 707.12 — copying a spell; copies are not cast, may choose new targets, cease to
  exist when they leave the stack.
- CR 105.2c / 115.1 — colorless objects; target restrictions checked against the candidate.
- CR 608.2b — a spell/ability with all targets illegal at resolution does not resolve.
