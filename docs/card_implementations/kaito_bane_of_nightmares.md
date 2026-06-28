# Kaito, Bane of Nightmares

**Vocab index:** 288
**Types:** Legendary Planeswalker Kaito — mana cost {2}{U}{B}, starting loyalty 4
**Script:** `bin/resources/cardsfolder/k/kaito_bane_of_nightmares.txt`

## Oracle
- **Ninjutsu {1}{U}{B}** ({1}{U}{B}, Return an unblocked attacker you control to hand: Put this
  card onto the battlefield from your hand tapped and attacking.)
- During your turn, as long as Kaito has one or more loyalty counters on him, he's a 3/4 Ninja
  creature and has hexproof.
- **[+1]:** You get an emblem with "Ninjas you control get +1/+1."
- **[0]:** Surveil 2. Then draw a card for each opponent who lost life this turn.
- **[-2]:** Tap target creature. Put two stun counters on it.

## Rules references
- **CR 702.49 — Ninjutsu.** A keyword on a card in hand granting an activated ability usable only
  during the declare-blockers step, after blockers are declared, while the controller has an
  **unblocked** attacker (CR 702.49e). The cost is the ninjutsu mana cost **and** returning one
  unblocked attacker you control to its owner's hand; the effect puts the ninja onto the
  battlefield from hand **tapped and attacking** the player or planeswalker the returned attacker
  was attacking.
- **CR 122.1d — Stun counters.** "If a permanent with a stun counter would become untapped,
  instead remove a stun counter from it." Modeled in the untap step.
- **CR 606 — Loyalty abilities.** Sorcery-speed, once per turn per planeswalker; the cost
  adds/removes loyalty counters.

## Implementation

### Mechanic 1 — Ninjutsu (general, reusable)
- **Parse — `src/parse.cpp`:** `K:Ninjutsu:<cost>` creates an `ACTIVATED` ability with
  `activation_zone = Zone::HAND`, `category = "Ninjutsu"`, and the new flag `is_ninjutsu = true`
  (`src/components/ability.h`). Only the mana portion of the cost is parsed
  (`parse_activation_cost`); the "return an unblocked attacker" cost is intrinsic.
- **Offer gate — `src/systems/state_manager_actions.cpp`:** in the hand-activated-ability loop,
  an `is_ninjutsu` ability is offered only when `game.cur_step == DECLARE_BLOCKERS` and the
  activator controls an unblocked attacker (`unblocked_attackers(...)`), in addition to the normal
  mana-affordability check. Labeled "Ninjutsu <card>".
- **Shared accessor — `src/game_queries.h`:** `unblocked_attackers(entities, ctrl)` returns the
  battlefield creatures controlled by `ctrl` that are attacking and not blocked (CR 509.1h). Used
  both to gate the offer and to pay the cost.
- **Execute — `src/action_processor.cpp` `process_ninjutsu()`:** pays the ninjutsu mana cost
  (cancellable), returns one chosen unblocked attacker to hand (capturing its `attack_target`),
  then puts the ninja onto the battlefield from hand. It marks `pending_enters_tapped` and a new
  one-shot `pending_enters_attacking[ninja] = attack_target` (`src/classes/game.h`).
  `process_activate_ability` dispatches to it when `ability.is_ninjutsu`.
- **Enters tapped + attacking — `src/systems/state_manager_statics.cpp`:** in
  `apply_permanent_components`, once the ninja's components exist, the `pending_enters_attacking`
  mark sets `Creature::is_attacking / attack_target / is_blocked = false` so a **creature** ninja
  is a real attacker (deals combat damage, can be blocked... but it entered after blocks, so it is
  unblocked). The existing `pending_enters_tapped` path taps it.

### Mechanic 2 — Stun counters (general, reusable)
- **Untap step — `src/classes/game.cpp`:** before untapping each of the active player's
  permanents, if a permanent is tapped and has one or more `Stun` counters, it sheds one counter
  and stays tapped instead of untapping (CR 122.1d). Counters are added by the existing
  `PutCounter` effect (`CounterType$ Stun`); no new counter infrastructure was needed. The counter
  key is the verbatim script string `"Stun"`.

### Loyalty abilities (reuse existing frameworks)
- **[-2] Tap + 2 stun:** `AB$ Tap` (existing tap effect) with a chained `DB$ PutCounter |
  CounterType$ Stun | CounterNum$ 2` (existing counter effect) — fully functional with the new
  untap-step stun rule.
- **[0] Surveil 2 + draw:** `AB$ Surveil` (existing) chained to `DB$ Draw` whose count is
  "each opponent who lost life this turn." In the two-player engine this resolves to 0/1; it
  resolves cleanly (drew 0 when the lone opponent had not lost life).
- **[+1] Emblem:** `AB$ Effect` resolves via the existing Effect handler and pays the loyalty cost,
  but the engine has no persistent **emblem** entity, so the "Ninjas you control get +1/+1"
  continuous buff is **not** applied (see Simplifications).

## Simplifications / follow-ups (documented)
- **Conditional creature-static deferred.** The `S:` line that makes Kaito "a 3/4 Ninja creature
  with hexproof during your turn while he has a loyalty counter" is **not** applied. It needs three
  static-layer features the engine lacks: `AddType$` for creatures (only lands today), a
  `counters_GE1_LOYALTY` qualifier in static `Affected$` filters, and `RemoveCardTypes$`. Kaito
  therefore remains a planeswalker (never becomes a creature) in the engine. Consequence below.
- **Attacking planeswalker is a no-op for combat.** Because Kaito is not a creature in the engine,
  the ninjutsu "enters attacking" half does nothing for Kaito specifically — he enters **tapped**
  (verified) but, lacking a `Creature` component, cannot be a combatant (the
  `pending_enters_attacking` mark is dropped). The ninjutsu path *does* fully set up attacking for a
  **creature** ninja (the general case). This matches CR only loosely for Kaito (a planeswalker
  that is also a creature would attack), but is consistent given the deferred creature-static.
- **[+1] emblem buff not applied** — no emblem subsystem (the ability still resolves and pays
  loyalty).
- **Ninjutsu resolves immediately** at activation rather than using the stack. In a two-player
  engine this only loses the narrow opponent-response window during the declare-blockers step.

## Tests (test harness)
- **Ninjutsu:** attack with an unblocked Grizzly Bears; in the declare-blockers step the menu
  offers "Ninjutsu Kaito"; activating it returned Grizzly Bears to hand, paid {1}{U}{B} (3 lands
  tapped), and put Kaito onto the battlefield **tapped** at loyalty 4. The returned attacker dealt
  no combat damage (opponent stayed at 20).
- **Stun counters:** Kaito's [-2] tapped a Grizzly Bears and put 2 stun counters on it (Kaito
  4→2 loyalty). At the opponent's next two untap steps it stayed tapped, shedding one counter each
  ("…stun counter removed instead of untapping"); on the third untap (counters gone) it untapped
  normally and attacked.
- **[0] Surveil:** resolved Surveil 2 then Draw 0 (no opponent had lost life), loyalty unchanged.
- **[+1] Effect:** resolved, loyalty 4→5, no error (emblem buff not applied — documented).
