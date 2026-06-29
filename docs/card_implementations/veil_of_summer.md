# Veil of Summer  (vocab index 283)

## Oracle text
Draw a card if an opponent has cast a blue or black spell this turn. Spells you control can't be
countered this turn. You and permanents you control gain hexproof from blue and from black until
end of turn. (You and they can't be the targets of blue or black spells or abilities your opponents
control.)

(Instant, mana cost {G}.)

## Forge script
- Source: fetched script — `bin/resources/cardsfolder/v/veil_of_summer.txt` (not edited).
- Key tags:
  - `A:SP$ Draw | Defined$ You | ConditionCheckSVar$ X | SubAbility$ DBEffect` — the conditional
    draw, gated by SVar `X`.
  - `SVar:X:Count$ThisTurnCast_Card.OppCtrl+Blue,Card.OppCtrl+Black` — the gate: has an opponent
    cast a blue or black spell this turn? (no `ConditionSVarCompare$` ⇒ Forge default `GE1`).
  - `SVar:DBEffect:DB$ Effect | ReplacementEffects$ AntiMagic | SubAbility$ DBPump` — the turn-long
    "can't be countered" grant, chaining to the hexproof grant.
  - `SVar:AntiMagic:Event$ Counter | ValidSA$ Spell.YouCtrl | Layer$ CantHappen` — "spells you
    control can't be countered this turn" (CR 614.13 / CantHappen).
  - `SVar:DBPump:DB$ Pump | Defined$ You & Valid Permanent.YouCtrl |
    KW$ Hexproof:Card.Black:black & Hexproof:Card.Blue:blue` — "you and permanents you control gain
    hexproof from blue and from black until end of turn".
- Tags parsed as written; no category was retagged.

## Engine work (all general, keyed on each tag's intended meaning)

### 1. Conditional draw — "an opponent has cast a blue or black spell this turn" (CR 603, 608.2)
- New per-player tracker `Player::spell_colors_cast_this_turn` (`std::set<Colors>`) —
  `src/components/player.h`. Recorded at cast time from the spell entity's `ColorIdentity` —
  `src/action_processor.cpp` (next to the existing `spells_cast_this_turn++`). Reset for **both**
  players each cleanup (so "this turn" is accurate for instants cast on the opponent's turn) —
  `src/classes/game.cpp`.
- `evaluate_dynamic_amount` (`src/components/ability.cpp`) now handles
  `Count$ThisTurnCast_Card.<Ctrl>+<Color>[,...]`: picks the player relative to the source's
  controller (`OppCtrl` ⇒ opponent, else self) and returns 1 if that player cast a spell of any
  named color this turn, else 0 — sufficient for the `GE1` conditions that consume it. General over
  any "an opponent/you cast a `<color>` spell this turn" check.
- Top-level `ConditionCheckSVar$` is now parsed (`apply_param_to_ability`) and resolved against the
  card's SVars in `parse_abilities`' post-pass, defaulting the comparator to `GE1` when no
  `ConditionSVarCompare$` is given (Forge default) — `src/parse.cpp`. Previously a top-level
  `ConditionCheckSVar$` was silently ignored (body always ran). The existing `resolve()` gate
  (condition fails ⇒ skip the body, **still** chain subabilities) is exactly what Veil needs: the
  draw is conditional, but the can't-be-countered + hexproof riders always happen. Sub-ability
  `ConditionCheckSVar$` parsing/behavior is unchanged. (Only Veil of Summer uses a top-level
  `ConditionCheckSVar$` among the vocab, so the new gate touches no other tracked card.)

### 2. "Spells you control can't be countered this turn" (turn-long, sourceless — CR 614.13)
- New turn-long set `cur_game.cant_counter_spells_of` (`std::set<Zone::Ownership>`) — `game.h`;
  cleared at cleanup — `game.cpp`. A player here ⇒ every spell they control can't be countered this
  turn. Unlike Hexing Squelcher's battlefield static (`spell_uncounterable_by_static`), this form
  belongs to no permanent (the instant resolves to the graveyard), so it is recorded in game state.
- `effects::counter` (`src/effects/effect_counter.cpp`) consults it in the existing can't-be-
  countered check (alongside the self-stamp and the battlefield static), keyed on the countered
  spell's caster.
- Parse: a `DB$ Effect | ReplacementEffects$ <SVar>` whose SVar is a `CantHappen` `Counter` on
  `Spell.YouCtrl` sets `Ability::effect_spells_uncounterable_this_turn` (`parse_svar_ability`,
  `src/parse.cpp` + flag in `src/components/ability.h`). The `GrantCast` handler
  (`src/effects/effect_grant_cast.cpp` — the existing dispatcher for transient `DB$ Effect`
  continuous effects) records the controller in `cant_counter_spells_of`.

### 3. "Hexproof from blue and from black" for the player + their permanents (CR 702.11e)
- New turn-long, player-scoped grant `cur_game.hexproof_from_colors_this_turn`
  (`{Zone::Ownership player; std::set<Colors> colors}` list) — `game.h`; cleared at cleanup. Each
  entry protects the player **and** every permanent that player controls from being targeted by a
  spell/ability an opponent controls whose **source** is one of `colors`.
- `Ability::is_legal_target` (`src/components/ability.cpp`) consults it up front (before the
  type-specific branches, so it covers both player and permanent candidates) via the new
  `target_has_color_hexproof(cand, source, caster)` helper, which reads the targeting object's
  `ColorIdentity` (a spell's = the spell's color; an ability's = its source permanent's color) and
  rejects a protected candidate. Enumeration (`build_valid_targets`) and re-verification
  (`is_target_valid`) share this one predicate, so a hexproof-from-color target is excluded at
  selection and a spell already targeting it fizzles at resolution (CR 608.2b).
- Parse: the `KW$ Hexproof:Card.<Color>:<desc>` token is split out of the keyword list into
  `PumpParams::grant_hexproof_from_colors` (`src/effects/effect_pump.cpp` +
  `src/components/ability_params.h`). The Pump handler turns it into the player-scoped grant for the
  ability's controller and returns early (no creature target selection) — general over any
  "hexproof from `<color>`" Pump.

## Behavioral decisions (made in lieu of asking a human)
- **Hexproof-from-color is modeled player-scoped (the player + permanents they control) rather than
  as a per-permanent keyword grant.** This is what lets it protect the *player* object (Veil's most
  important use — answering Thoughtseize/targeted discard) and every permanent type (lands,
  artifacts), which a creature-only `Pump` keyword bucket cannot. Minor deviation: a permanent that
  enters *after* Veil resolves is also covered for the rest of the turn, whereas CR locks the set at
  resolution; this is harmless for a one-turn effect and far simpler/more robust.
- **`Count$ThisTurnCast_...+<Color>` is presence-tracked (0/1 per color), not an exact count.** The
  only consumer is a `GE1` condition, so presence is sufficient and avoids double-counting a
  multicolored spell.
- **No observation/state-vector change.** All three pieces live in engine-internal game/player state
  (read only at resolution / targeting); `STATE_SIZE`, `OBS_SIZE`, `N_CARD_TYPES` and the obs layout
  are unchanged.

## Tests (test_harness, seed 1)
1. **Conditional draw fires** — B casts Fatal Push (black) at A's Grizzly Bears; A responds with
   Veil: "Resolving ability (category: Draw)" → "Player A draws Mountain". (Bonus: the in-flight
   Fatal Push then fizzles — "Destroy fizzles" — its target became hexproof from black mid-stack.)
2. **No draw without an opponent U/B spell** — A casts Veil on a clean turn: the Draw body resolves
   with no card drawn; the can't-be-countered + hexproof riders still apply.
3. **Can't be countered** — after Veil resolves, A casts Lightning Bolt; B's Counterspell targeting
   it ⇒ "Lightning Bolt can't be countered" → Bolt resolves ("DealDamage, amount: 3").
4. **Hexproof from blue/black** — after Veil resolves: a black Fatal Push can't choose A's Grizzly
   Bears as a target (it is no longer offered; the target spec fails as illegal), while a **red**
   Lightning Bolt still targets and destroys it.
- No draws (game-result), no non-fatal errors. Scripted regression games (mixed U/B/R/G hands, and a
  Veil-in-hand scripted game) ran clean to a decisive result.

## Result
implemented
