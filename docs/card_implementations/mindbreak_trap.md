# Mindbreak Trap  (vocab index 112)

## Oracle text
If an opponent cast three or more spells this turn, you may pay {0} rather than pay this
spell's mana cost.

Exile any number of target spells.

(Instant — Trap, mana cost {2}{U}{U}.)

## Forge script
- Source: fetched (Forge@master) — `bin/resources/cardsfolder/m/mindbreak_trap.txt`
- Key tags:
  - `S:Mode$ AlternativeCost | ValidSA$ Spell.Self | Cost$ 0 | CheckSVar$ OppCastThisTurn`
    — the **Trap** alternative cost (CR 702.59): you may pay {0} (`Cost$ 0`, free) instead of the
    mana cost, but only while the condition `OppCastThisTurn` holds.
  - `SVar:OppCastThisTurn:PlayerCountOpponents$ConditionGE3 SpellsCastThisTurn` — the condition:
    count opponents whose per-turn spells-cast total is `>= 3`. Forge's implicit `CheckSVar` test is
    "> 0", i.e. *at least one* opponent has cast three or more spells this turn (CR 702.59c — Trap's
    alternative-cost condition).
  - `A:SP$ ChangeZone | TargetType$ Spell | ValidTgts$ Card | TgtZone$ Stack | Origin$ Stack |
    Destination$ Exile | TargetMin$ 0 | TargetMax$ MaxTgts` — the spell ability: exile any number
    of target spells. This is a counter-by-exile expressed as a `ChangeZone` (Stack → Exile), not a
    `Counter` tag, and is honored as such (no retag).
  - `SVar:MaxTgts:Count$ValidStack Card` — the target cap is "every spell on the stack", i.e. "any
    number of target spells".

## Engine work
All changes are general (keyed on the tag's intended meaning), not card-specific.

- **`TargetMax$` with a count-SVar value (`src/parse.cpp`).** `TargetMax` previously parsed its
  value with `std::stoi`, which is undefined behavior on a non-numeric value under
  `-fno-exceptions` (the script supplies `TargetMax$ MaxTgts`). The parser now uses the numeric
  value directly when it is numeric, and otherwise treats the cap as "any number" by setting
  `target_max = MAX_ENTITIES`. The existing multi-target selection loop
  (`action_processor.cpp::select_target`) already stops on its own once no further legal targets
  remain, so an effectively-unbounded cap correctly limits to the spells actually on the stack.

- **`ChangeZone` removing a spell/ability from the stack (`src/effects/effect_change_zone.cpp`).**
  The targeted `ChangeZone` path moved the target entity to the destination but did not clean up
  stack-object components. When the target is a spell (or standalone ability) on the stack, its
  `Spell`/`Ability` components are now stripped before the move — mirroring `effects::counter` —
  so the stack no longer treats the exiled object as a live spell to resolve (CR 701.5a / 702.59c:
  a spell removed from the stack does not resolve). A standalone ability entity (no `CardData`) is
  destroyed after it leaves the stack, exactly as the counter handler does. Cards/permanents moved
  from other zones (Swords to Plowshares, Faerie Macabre, Life from the Loam) are unaffected — they
  have no `Spell` component and are not on the stack, so the new branch is inert for them.

- **Trap alternative-cost condition (`src/systems/state_manager_actions.cpp::can_afford_alt`).**
  Added handling for the `PlayerCountOpponents$Condition<OP><N> SpellsCastThisTurn` condition SVar.
  The op/threshold (e.g. `GE3`) is embedded in the SVar's `Condition<OP><N>` token (there is no
  separate `SVarCompare$`), so it is parsed out of the SVar string and compared against the
  opponent's `Player::spells_cast_this_turn`. The per-turn spell-cast counter already exists
  (incremented in `action_processor.cpp` at cast time, reset each turn in `game.cpp`), so no new
  tracking subsystem was needed — only the new condition reader. The existing
  `Count$YouCastThisGame` branch (Once Upon a Time) is preserved unchanged.

## Behavioral decisions (made in lieu of asking a human)
- **The Trap alt-cost is implemented, not ignored.** The condition is core to the card and the
  per-turn opponent spell-count infrastructure already existed, so deferring/ignoring the Trap
  clause was unnecessary. The card is *also* hard-castable for `{2}{U}{U}` at all times — both the
  regular cast and the alternate cast are presented independently whenever affordable.
- **"Any number of target spells" includes zero (`TargetMin$ 0`).** Optional targeting is honored;
  the multi-target loop offers "Done selecting targets" once the minimum is met. Exiling zero
  spells does nothing (the resolver no-ops on an empty target list).
- **Exile, not counter, per the script.** Although the gameplay result resembles a counter, the
  script uses `ChangeZone … Destination$ Exile`. It is implemented faithfully as a zone move that
  removes the spell from the stack, rather than retagged to `Counter` — preserving the rule that a
  spell removed from the stack this way is exiled (not put in the graveyard) and is not "countered"
  for cards that care about that distinction.
- **Ignored cosmetic tags (documented):** `EffectZone$ All` and `ValidSA$ Spell.Self` describe the
  alt-cost's scope/self-application (already the engine default for a card's own alt cost);
  `StackDescription$`/`SpellDescription$`/`TgtPrompt$`/`Description$` are reminder/UI text;
  `AI:RemoveDeck:All` is a Forge AI hint. None change behavior; they are silently ignored.

## Tests
- Isolation (test_harness):
  - **Hard-cast exile of one spell:** A (4 Islands in play, Mindbreak Trap in hand) responds to
    B's Lightning Bolt (targeting A) by hard-casting Mindbreak Trap and targeting the Bolt — "Lightning
    Bolt is moved to exile"; A stays at 20 life (Bolt never resolved); no fizzle/non-fatal error.
  - **Trap free-cast exiling 3 spells:** B (3 Mountains) casts three Lightning Bolts at A; A has no
    lands (cannot hard-cast) but is offered "Cast Mindbreak Trap (alternate cost)", casts it "for
    free (alternate cost)", and exiles all three Bolts ("moved to exile" ×3). A stays at 20 life.
  - **Negative condition (2 spells):** with only two opponent spells cast this turn and no lands, A
    is offered *no* Mindbreak Trap action at all (no hard-cast mana, alt cost disabled) — confirming
    the `>= 3` threshold.
- Regression (test_harness --scripted, full games): deck = delver shell with 2× Mindbreak Trap (in
  place of 2× Daze) vs mav, seeds 1-6 — all six decisive (4 A wins, 2 B wins), no draws, no
  max-decisions caps, no non-fatal errors. Mindbreak Trap was exercised in 5 of the 6 games. The
  only warnings are the pre-existing cosmetic `Unrecognized ability param` lines for other cards.

## Result
implemented
