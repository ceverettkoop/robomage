TODO:

## Open engine correctness issues

- **Counters can go negative.** `add_counters` (`src/queries/counters.cpp`) adds `delta` with no
  floor at 0, so removing more counters than an object has stores a negative count. Seen as
  planeswalker loyalty after excess damage: `damage_planeswalker` (`src/queries/damage.h`) removes
  `amount` LOYALTY counters ("loyalty now -10"). The 0-loyalty SBA still destroys it, but the
  stored count is wrong (CR 306.8 / 609.3: only as many as exist can be removed; an effect that
  reads the count, or one that adds counters back first, sees the negative value). Repro (from
  `bin/`): `./debug/robomage --narrative --game-seed 1 --max-decisions 20 --hand-a "Lightning
  Bolt,Lightning Bolt" --battlefield-a "Mountain,Mountain" --battlefield-b "Teferi Time Raveler"
  --play "A:keep,B:keep,A:cast:Lightning Bolt,A:target:Teferi,A:cast:Lightning
  Bolt,A:target:Teferi"` logs "loyalty now -2".

- `rules_mod::may_play_lands_from_graveyard` (`src/systems/rules_modifying.cpp`) ignores the
  static's `Affected$` filter beyond its `Land` head: `parse.cpp` (~4207) sets
  `may_play_from_graveyard` for any `MayPlay$` graveyard static whose filter is headed `Land`,
  and the permission check (`graveyard_land_static`, `src/queries/permissions.cpp`) reads only
  that flag, so every land card qualifies. Latent — the vocab's statics (Icetill Explorer, Mole
  Man, …) are all `Affected$ Land.YouOwn`; wrong once a narrower filter (`Land.Basic+YouOwn`, …)
  is added.

## Observation gaps

- **A cast's actual cost is not in the observation** (noted 2026-11-22; not planned). A `menu` row
  (`src/state_format/schema.cpp`, `MENU`) carries its category, source, ability, `variant`
  (`cast_variant`) and a text description, but no cost: what the cast would really take after
  increases and reductions (Thalia's tax, a cost reducer, delve, affinity, convoke) or for an
  alternative cost is not in the frame at the decision where the cast is chosen. The network sees only
  that the row is offered (so affordable: `can_pay`) and the permanents causing the change; the
  workbench shows the printed cost (from CARDS and the oracle text, `priority_costs`). The adjusted
  cost appears only afterwards, in `pending_cost`, when payment is a decision of its own (rarely,
  under the default auto-mana). Fix if wanted: the cost `can_pay` computes, as menu columns (the
  `rem_*` pips, life, energy, discard; the `menu` token type has 13 free numeric slots of 48) or a
  `menu_cost` child table. Either is a registry append: checkpoints keep loading, exports need
  `rmtrain retable`, the corpus `make regen`. Before resuming training on it, zero the new slots'
  `num_proj` columns (and the critic's) on load: reserved columns still hold their random
  initialisation (std 1/√8; a zero input gets no gradient and `weight_decay` is 0), so a new nonzero
  input would add a fixed random vector to every cast row. That load rule (beside `INHERIT_PARAMS`,
  `learn/rmlearn/model.py`) applies to any column appended to an existing token type.

## Deferred — bigger, needs its own session

### Aura-on-PLAYER state-based action (CR 704.5m) — DEFERRED, latent
The Aura SBA (`aura_attached_illegally`, `src/systems/state_manager.cpp`) puts an Aura into the
graveyard whenever `Permanent::equipped_to` names no battlefield permanent, and an Aura can only be
attached to a permanent, so an Aura that enchants a PLAYER (a Curse, "Enchant player") resolves
unattached and is immediately put into the graveyard. No such Aura is in the vocab. The engine
deliberately has no player-attachment model (see The One Ring). When one is added: record the
enchanted PLAYER as the attach target and make the SBA treat a legally player-attached Aura as
attached. Until then, do NOT add an "Enchant player" Aura to the vocab.

### Unbounded decisionless loops (CR 104.4b)
Nothing detects a mandatory loop that makes no decisions (a trigger / SBA / replacement cycle in
which every menu has one row, so the engine auto-takes it): the per-game cap
(`--max-game-decisions`, `cap_current_game`) counts only asked decisions, so such a game hangs.
CR 104.4b makes it a draw. A loop that creates entities stops at `EntityManager::CreateEntity`'s
`fatal_error("Too many entities in existence.")` (`MAX_ENTITIES` 1000) instead of drawing. Fix:
detect the repetition (or a per-turn cap on auto-taken decisions / resolutions) and end the game
as a draw with a logged diagnostic.
