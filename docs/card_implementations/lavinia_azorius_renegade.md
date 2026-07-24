# Lavinia, Azorius Renegade

Vocab index: 345

## Oracle text

Legendary Creature — Human Soldier (2/2), {W}{U}

- Each opponent can't cast noncreature spells with mana value greater than the number of lands that player controls.
- Whenever an opponent casts a spell, if no mana was spent to cast it, counter that spell.

## Forge script (Source: pre-existing local, `bin/resources/cardsfolder/l/lavinia_azorius_renegade.txt`)

Key tags:

- `S:Mode$ CantBeCast | ValidCard$ Card.nonCreature+nonLand | Caster$ Opponent | cmcGT$ Land`
- `T:Mode$ SpellCast | ValidCard$ Card | ValidActivatingPlayer$ Opponent | TriggerZones$ Battlefield | Execute$ TrigCounter | ValidSA$ Spell.ManaSpent EQ0`
- `SVar:TrigCounter:DB$ Counter | Defined$ TriggeredSpellAbility`

## Engine work

Mechanics added (general): **cant-cast-cmc-gt-landcount** (+ reused mana-spent counter trigger).

### 1. Static: can't cast noncreature spell with MV > caster's land count (NEW)

Relevant CR: 601.3e (a rule that modifies what a player may cast is applied when checking
cast legality); the bound is dynamic (the caster's land count at cast-legality time).

- `src/components/static_ability.h` — new field `cant_cast_cmc_gt_land` on the CantBeCast
  static block (dynamic MV bound, distinct from the static numeric `extract_static_cmc_bound`
  path used by Gaddock Teeg).
- `src/parse.cpp` (~L3608, static S: key loop) — parse `cmcGT$ Land` on a `CantBeCast`
  static into `sa.cant_cast_cmc_gt_land`.
- `src/systems/rules_modifying.cpp` — `cast_prohibited`, opponent branch
  (`cant_cast_by_opponent`): when `cant_cast_cmc_gt_land` is set, instead of the blanket
  opponent lock (Voice of Victory), (a) require the spell to match the `ValidCard$` filter
  (`Card.nonCreature+nonLand`) via `card_matches_filter` with `ctx.controller = caster`, then
  (b) count the caster's battlefield lands via the shared accessor
  `count_battlefield_matching("Land.YouCtrl", caster, 0)` and prohibit iff
  `card_mana_value(card) > land_count`. Reuses the existing `Caster$ Opponent` machinery
  (extended for Teferi in a prior session).

### 2. Trigger: counter opponent's free/0-mana cast (MOSTLY REUSED)

Relevant CR: 603 (triggered abilities); 106/601.2g (mana spent recorded per cast).

- Mana-spent tracking (`Spell::mana_spent`, set from `Game::PendingCast::mana_spent`) and the
  `ValidSA$ Spell.ManaSpent EQ0` parse/compare were already built (Roiling Vortex). The
  `DB$ Counter | Defined$ TriggeredSpellAbility` effect was already built (Chalice of the
  Void: `effects::counter` + `defined_triggered_spell`).
- NEW: `ValidActivatingPlayer$ Opponent` support so the trigger fires only on OPPONENT casts:
  - `src/components/ability.h` — new flag `trigger_valid_player_is_opponent`.
  - `src/parse.cpp` (`ValidPlayer`/`ValidActivatingPlayer` key) — set the flag on
    `value == "Opponent"`; carried through the `Execute$` SVar promotion.
  - `src/systems/state_manager_triggers.cpp` — in the SPELL_CAST match block, mirror the
    `trigger_valid_player_is_controller` gate: skip when the event's `PLAYER` (the caster)
    equals the source's controller (two-player game: opponent = not controller).

## Behavioral decisions

- The static is a prohibition queried at cast-legality time (like every `rules_mod` check),
  not a layer effect; the land-count bound is recomputed each check so it tracks the caster's
  board live.
- Land count uses `Land.YouCtrl` so phased-out lands are excluded (shared battlefield
  accessor rule) and the caster's own lands are counted, per the card text ("that player").
- Creatures and lands are unaffected (the `ValidCard$ Card.nonCreature+nonLand` filter);
  the controller of Lavinia is unaffected (`Caster$ Opponent`).
- Two-player scope: "opponent" = the single non-controller player.

## Tests (isolation via `train/test_harness.py --play`; seed 1)

Static (MV2 noncreature `Sylvan Library` = 1G; MV2 creature `Scryb Ranger` = 1G; extra mana
via `Lotus Petal`):

- Control (no Lavinia), B controls 1 land + Petal: `Cast Sylvan Library` offered (mana suffices). PASS
- Lavinia on A, B controls 1 land + Petal: `Cast Sylvan Library` NOT offered (MV2 > 1). PASS
- Lavinia on A, B controls 2 lands: `Cast Sylvan Library` offered (MV2 ≤ 2). PASS (boundary)
- Lavinia on A, B controls 1 land + Petal: `Cast Scryb Ranger` (MV2 creature) offered — creatures unaffected. PASS
- Lavinia on A, A controls 1 land + Petal: `Cast Sylvan Library` offered — controller unaffected. PASS

Trigger (counter free casts):

- Lavinia on A, B casts `Mishra's Bauble` (0 mana): "Mishra's Bauble is countered". PASS
- Lavinia on A, B casts `Ponder` (1 mana): NOT countered — Ponder resolves (rearrange/shuffle prompt). PASS

CI gate: `train/.venv/bin/python train/ci_check.py --tier pygen,vocab,smoke --smoke-games 1`.

## Result

Implemented.
