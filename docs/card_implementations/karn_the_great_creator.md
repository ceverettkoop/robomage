# Karn, the Great Creator  (vocab index 280)

## Oracle text
Legendary Planeswalker — Karn. Loyalty 5. Mana cost {4}.

- Static: Activated abilities of artifacts your opponents control can't be activated.
- [+1]: Until your next turn, up to one target noncreature artifact becomes an artifact creature
  with power and toughness each equal to its mana value.
- [−2]: You may reveal an artifact card you own from outside the game or choose a face-up
  artifact card you own in exile. Put that card into your hand.

## Forge script
- Source: in-repo — `bin/resources/cardsfolder/k/karn_the_great_creator.txt` (parsed as written,
  no retag).
- Type `Legendary Planeswalker Karn`, mana cost `4`, `Loyalty:5`.
- Key tags:
  - `S:Mode$ CantBeActivated | AffectedZone$ Battlefield | ValidCard$ Artifact.OppCtrl |
    ValidSA$ Activated` — the static lock.
  - `A:AB$ Animate | Cost$ AddCounter<1/LOYALTY> | TargetMin$ 0 | TargetMax$ 1 |
    Planeswalker$ True | ValidTgts$ Artifact.nonCreature | Power$ X | Toughness$ X |
    Types$ Artifact,Creature | Duration$ UntilYourNextTurn` with `SVar:X:Targeted$CardManaCost` —
    the +1.
  - `A:AB$ ChangeZone | Cost$ SubCounter<2/LOYALTY> | Planeswalker$ True |
    Origin$ Sideboard,Exile | Destination$ Hand | ChangeType$ Artifact.YouOwn | ChangeNum$ 1 |
    Hidden$ True | Reveal$ True` — the −2.

## Rules (CR)
- **606** loyalty abilities (one per turn, sorcery speed); the loyalty cost is paid by adding/
  removing loyalty counters. Already supported by the planeswalker framework (Ajani/Ugin).
- **613** continuous effects / **613.7** "becomes" layer interactions; **613.4** layer 7b sets
  the animated permanent's base P/T.
- **514 / 613** effect durations: "until your next turn" is a longer-than-EOT duration that ends
  as the controller's next turn begins (not at this turn's cleanup).
- **202.3 / 107.14** mana value (the +1's P/T source).
- **CantBeActivated** lock is a continuous ability that prevents activating matching activated
  abilities (here, artifacts the static's controller's opponent controls).

## Engine work
Three general mechanisms (no card-specific retagging):

### 1. Animate with a dynamic P/T equal to the target's mana value (the key deliverable)
The until-EOT/permanent `DB$/AB$ Animate` effect (`src/effects/effect_animate.cpp`) already
existed (The Fantasticar / Earthbend extension points on `Permanent`). Extended it to set a base
P/T from `Power$`/`Toughness$`:
- New `Ability` fields (`src/components/ability.h`): `animate_has_pt`, `animate_base_power/
  toughness`, `animate_power_token/toughness_token` (raw, pre-SVar), `animate_power_expr/
  toughness_expr` (resolved runtime expr).
- Parse: `apply_param_to_ability` (`src/parse.cpp`) claims `Power$`/`Toughness$` when
  `category == "Animate"` (numeric → base int; SVar token → stored raw, resolved post-parse into
  the dynamic expr alongside the existing `amount_svar`/pump resolution).
- New dynamic-amount branch `Targeted$CardManaCost` in `evaluate_dynamic_amount`
  (`src/components/ability.cpp`): the target's mana value (`CardData::mana_cost.size()`).
- At resolution `effects::animate` evaluates the expr against the animate target and sets
  `Permanent::animate_set_pt` + `animate_power/toughness`; the existing
  `apply_animate_creature_bootstrap` builds the `Creature`/`Damage` components.
- **General:** any `Power$ X | X$ <dynamic expr>` Animate reuses this; the bootstrap is unchanged.

### 2. `Duration$ UntilYourNextTurn` — a longer continuous-effect duration
The existing Animate had two durations: `Permanent` (rest-of-game `animate_*` fields) and the
Forge default until-EOT (the `*_eot` fields cleared at this turn's CLEANUP). Added a third:
- Parse: `Duration$ UntilYourNextTurn` → `Ability::animate_duration_until_your_next_turn`.
- `effects::animate` routes this duration to the rest-of-game `animate_*` fields (so the grant
  **survives this cleanup**, reapplied each SBA pass) PLUS new `Permanent` markers
  (`animate_until_my_turn`, `animate_added_types_until_turn`, `animate_until_turn_controller`)
  recording who animated it and exactly which types were added.
- Revert: `effects::revert_until_turn_animates(active_player)`
  (`src/effects/effect_animate.cpp`), called at the **UNTAP step** in
  `Game::advance_step` (`src/classes/game.cpp`) — at the start of the animating player's next
  turn it erases the granted types, drops the base P/T, and strips the bootstrapped
  `Creature`/`Damage` unless the permanent is a creature by a permanent means. General over any
  UntilYourNextTurn Animate.

### 3. `ChangeZone` searching Exile / Sideboard ("outside the game")
The search/tutor machinery (`search_zone` / `search_multi_zone`, `src/components/ability.cpp`)
handled Library/Hand/Graveyard. Added `Zone::EXILE` and `Zone::SIDEBOARD` collection (enumerate
entities by `Zone` owner, like the graveyard), and added `Sideboard` to the `Origin$` parse
lambda (`src/effects/effect_change_zone.cpp`). The −2 then reuses the existing
search → reveal → move-to-Hand path unchanged (`ChangeType$ Artifact.YouOwn`,
`Destination$ Hand`, `Reveal$ True`).

### 4. CantBeActivated honoring the `.OppCtrl` controller qualifier (necessary fix)
The static was **not** correctly supported as the ledger assumed: the existing
`CantBeActivated` matcher (`src/systems/rules_modifying.cpp`) compared the filter string to a
permanent's type *name only*, so `Artifact.OppCtrl` matched nothing (and ignored the controller
restriction). Rewrote `mana_activation_prohibited` / `activation_prohibited` to route the filter
through the shared `permanent_matches_any` with `MatchCtx.controller = ` the static's own
controller. Bare-type filters (Null Rod `Artifact`, Clarion Conqueror
`Artifact,Creature,Planeswalker`) behave identically; `Artifact.OppCtrl` now correctly suppresses
only the opponent's artifacts, never the controller's own. (Comma-OR and type qualifiers are
honored for free.)

### 5. Parser fix exposed by Karn (pre-existing latent bug)
`value_from_script` (`src/parse.cpp`) did a naive substring `find`, so the top-level `PT` field
lookup matched the `PT` inside Karn's `AILogic$ PTByCMC`, returning garbage that `parse_power`'s
`stoi` crashed on. Fixed it to match a key only as a **line-start field header** (`Key:` at the
start of the script or after a `\n`). Strictly more correct for line-based field extraction;
verified existing decks still parse and play.

## Behavioral decisions (made in lieu of asking a human)
- **P/T snapshot vs. CDA.** Karn's "power and toughness each equal to its mana value" is set as a
  **snapshot of the target's mana value taken at resolution** (stored as plain ints in
  `Permanent::animate_power/toughness`), not a live characteristic-defining value. Mana value is a
  fixed printed characteristic of a noncreature artifact, so the snapshot equals the CDA reading
  in practice, and this matches how the existing animate effect stores base P/T. Documented and
  general (`animate_power_expr` is evaluated once at resolution).
- **−2 from the sideboard ("outside the game").** Exile and Sideboard origins are both wired into
  the search. In a single game the engine does not instantiate sideboard cards as entities (only
  the bo3 sideboard phase manipulates the `Deck` struct), and exile cannot be pre-populated by
  the test harness, so a *positive* fetch is not demonstrable here — see Tests. The engine path
  is correct and reuses the shared search machinery.

## Tests
Isolation (`train/test_harness.py`, scenario JSON, seed 1; Karn cast from hand because the
comma in its name blocks `--battlefield`/`--hand` preset):

- **(+1 dynamic P/T, MV 2).** Karn cast; +1 targets a preset Grim Monolith (mana value 2):
  "Grim Monolith becomes an Artifact… becomes a Creature…", decoded **[2/2]**, loyalty 5 → 6.
  **PASS.**
- **(+1 dynamic P/T, MV 1).** Same line vs a preset Expedition Map (mana value 1): decoded
  **[1/1]** — proving the P/T tracks the target's mana value, not a constant. The animated Map
  attacked turn 1 (it is a creature, and was controlled continuously). **PASS.**
- **(Duration "until your next turn").** The Map stays **[1/1]** through Turn 1's cleanup AND all
  of Turn 2 (the opponent's whole turn) — i.e. it does **not** revert at end-of-turn like the
  Fantasticar EOT path — then "Expedition Map is no longer a creature" fires exactly at the start
  of **Turn 3 (the animating player's next turn)**. **PASS.**
- **(−2 ChangeZone).** −2 activated: "activates a loyalty ability (−2, loyalty now 3)",
  "Resolving ability (category: ChangeZone, amount: 1)", the Exile/Sideboard→Hand search runs and
  (with empty exile/sideboard) offers "Fail to find" and resolves cleanly. The activation, the −2
  loyalty cost, and the search path are verified; a positive fetch needs an exile/sideboard entity
  the single-game harness can't provide (see Behavioral decisions). **PASS (path).**
- **(Static).** With Karn in play, an opponent's Voltaic Key's activated ability is **not offered**
  on any of the opponent's subsequent turns (turns 2/4/6), while the controller's own Grim
  Monolith remains activatable throughout — confirming the lock is controller-scoped to the
  opponent. **PASS.**

Regression (`train/test_harness.py --scripted`, delver vs mav, seeds 1–3): all three games
finished decisively (A wins each), no draws, no non-fatal errors / asserts / tracebacks — the
`value_from_script` and CantBeActivated changes do not break existing card parsing/play.
(`train.py observe` not used — `torch` absent — so the regression ran directly through the
harness, per CLAUDE.md.)

## Result
implemented
