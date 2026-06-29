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
- **[+1] Emblem:** `AB$ Effect | StaticAbilities$ STNinjaBoost | Duration$ Permanent` creates a
  player-owned **emblem** carrying the continuous static "Ninjas you control get +1/+1." (now a
  general subsystem, below).

### Mechanic 3 — Conditional creature-static (general, reusable)
The `S:Mode$ Continuous | Affected$ Permanent.Self+counters_GE1_LOYALTY | Condition$ PlayerTurn |
AddType$ Creature & Ninja | RemoveCardTypes$ True | SetPower$ 3 | SetToughness$ 4 | AddKeyword$
Hexproof` line is honoured exactly as written, gated and re-evaluated every state-based-effects
pass through the existing CR 613 layer engine:

- **Parse (`src/parse.cpp`, `parse_one_static_ability`).** New tags: `RemoveCardTypes$ True` →
  `StaticAbility::remove_card_types`; the `counters_<CMP><N>_<TYPE>` qualifier on the `Affected$`
  filter (`counters_GE1_LOYALTY`) is split into `self_counter_compare` ("GE1") / `self_counter_type`
  ("LOYALTY"). `SetPower$ 3` / `SetToughness$ 4` / `AddKeyword$ Hexproof` reuse the existing
  `set_power_svar` / `set_toughness_svar` / `add_keyword` fields.
- **Condition (`gather_active_statics`, `src/systems/state_manager_statics.cpp`).** `Condition$
  PlayerTurn` (already present for Voice of Victory) ANDs with a new per-source counter gate: the
  static is active only while the source has ≥1 LOYALTY counter. Re-evaluated each pass, so Kaito
  stops being a creature the instant his loyalty hits 0 or the turn passes to the opponent.
- **Layer 4 — type add + creature bootstrap (`apply_self_animate_statics`).** When the gate holds,
  the `AddType$` CARD types (`Creature` + the `Ninja` subtype) are added to the source's live type
  set **additively** and, because `Creature` is among them, a `Creature`/`Damage` component is
  bootstrapped (base 0/0; summoning sickness = "entered this turn"). `RemoveCardTypes$ True` drops
  the source's printed CARD types **other than Planeswalker** — the Oracle's "that's still a
  planeswalker" (CR 306): a planeswalker that becomes a creature stays a planeswalker, so it can
  still be attacked as a planeswalker and keeps its loyalty abilities. When the gate lapses the
  added types and the bootstrapped `Creature`/`Damage` are stripped (unless the source is a creature
  by a permanent means).
- **Layer 7b — set P/T.** The existing non-CDA "set P/T" setter (Humility's machinery) matches the
  `Affected$ ...Self` form and sets Kaito to 3/4 while the gate holds.
- **Layer 6 — keyword grant.** The existing keyword-grant applier grants `Hexproof` to the (now
  creature) source. Hexproof is already enforced in `Ability::is_legal_target`
  (`permanent_has_keyword`), so an opponent's targeted removal/burn cannot target Kaito during your
  turn.
- **Emblem composition.** Because Kaito is a *Ninja creature* during your turn, the [+1] emblem's
  "Ninjas you control get +1/+1." (layer 7c) stacks on top of the 3/4 — Kaito is 4/5 with one
  emblem, 5/6 with two, etc.

### Mechanic 4 — Emblem subsystem (general, reusable)
- **Data model (`src/classes/game.h`).** `struct Emblem { Zone::Ownership controller;
  std::vector<StaticAbility> statics; };` stored in `Game::emblems` — a zoneless, unremovable
  continuous-effect source owned by a player (CR 114). Persists for the rest of the game; a fresh
  `Game` (next game of a match) starts with none.
- **Parse (`src/parse.cpp`).** An `AB$ Effect` with `StaticAbilities$ <SVar>` **and** `Duration$
  Permanent` resolves the named continuous-static SVar (`STNinjaBoost`) into a `StaticAbility` via
  `parse_one_static_ability` and stores it on `Ability::effect_emblem_statics`. The permanent
  duration distinguishes an emblem from the transient `StaticAbilities$` Effects (Unblockable,
  Ugin's MayPlay), which keep their own EOT handling.
- **Create (`effects::grant_cast`, `src/effects/effect_grant_cast.cpp`).** At resolution the handler
  pushes an `Emblem{controller, statics}` into `cur_game.emblems`.
- **Apply (`gather_active_statics`).** Each emblem's statics are gathered into `g_active_statics`
  every SBA pass with entity 0 (no Permanent) and the owner as their controller, so the layer
  appliers fan them out through the same path as a battlefield anthem — no emblem is ever a
  targetable / counted / destructible permanent. All `g_active_statics` consumers were audited to
  guard `as.entity` deref behind the relevant category, so the entity-0 sentinel is safe.

### Mechanic 5 — Attacking planeswalker (resolved)
With the creature-static done, Kaito is a real creature during your turn, so combat just works:
- He is **eligible to attack** (the declare-attackers scan keys off the live `Creature` component);
  verified declared as an attacker and dealing 3 combat damage.
- **Ninjutsu enters-attacking now applies to Kaito.** `apply_permanent_components` runs before the
  layer-4 bootstrap, so a planeswalker entering via ninjutsu has no `Creature` component yet; the
  `pending_enters_attacking` mark is now **left pending** there (instead of dropped) and consumed by
  the layer-4 self-animate bootstrap once the `Creature` exists — so a ninjutsu'd Kaito enters
  tapped **and attacking** as a 3/4 Ninja creature.

## Simplifications / follow-ups (documented)
- **`RemoveCardTypes$ True` preserves Planeswalker.** The script tag literally says "remove the
  source's card types", but the Oracle ("that's still a planeswalker") requires Kaito to remain a
  planeswalker. The self-animate applier therefore removes the printed CARD types **except
  Planeswalker**. This is the only correct reading for this card; no other vocab card uses
  `RemoveCardTypes$` on a self-animate static.
- **Summoning sickness is approximate.** A permanent that becomes a creature is treated as sick iff
  it entered this turn (`entered_on_turn == turn`), rather than tracking continuous control since
  the controller's most recent turn began. Correct for every normal case (cast/ninjutsu'd Kaito is
  sick the turn he enters, able to attack thereafter); only a control-change corner case would
  differ, which is out of two-player scope here.
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
- **Conditional creature-static:** with Kaito preset (loyalty 4), on **your** turn the board shows
  `Kaito, Bane of Nightmares [3/4] [loy 4]` — a 3/4 creature that is still a planeswalker (loyalty
  abilities still offered) — and he is declared as an attacker `(ATK)`. On the **opponent's** turn
  the same permanent shows `[loy 4]` with no P/T (not a creature). An opponent's Lightning Bolt cast
  during your turn offers only the two players as targets, never Kaito (hexproof).
- **[+1] emblem:** activating [+1] creates an emblem; Kaito (a Ninja creature on your turn) grows
  3/4 → 4/5 → 5/6 → 6/7 as successive emblems stack "Ninjas you control get +1/+1." (effect fires,
  not a no-op).
- **Ninjutsu + creature:** attacking with an unblocked Grizzly Bears then activating "Ninjutsu
  Kaito" returned the Bears, auto-paid {1}{U}{B}, and put Kaito onto the battlefield
  `[3/4] [loy 4] (T,SICK)` **tapped and attacking** — he dealt 3 combat damage to the opponent.
- **Regression:** scripted delver-vs-mav, seeds 1–3, each finished decisively (Player A wins; no
  draws, no Kaito parse warnings or non-fatal errors) — the layer/static hot-path changes are clean.
