# Cabal Therapy (vocab index 159)

## Oracle text
Choose a nonland card name. Target player reveals their hand and discards all cards with
that name.

Flashback—Sacrifice a creature. *(You may cast this card from your graveyard for its
flashback cost. Then exile it.)*

(Sorcery — `{B}`.)

## Forge script (Source: pre-existing local `bin/resources/cardsfolder/c/cabal_therapy.txt`)
Key tags:
- `A:SP$ NameCard | Defined$ You | ValidCards$ Card.nonLand | ValidDescription$ nonland | SubAbility$ DBDiscard`
  — the primary spell effect names a nonland card; the named card is chosen at resolution
  (CR 201.4) by the spell's controller.
- `SVar:DBDiscard:DB$ Discard | ValidTgts$ Player | Mode$ RevealDiscardAll | DiscardValid$ Card.NamedCard`
  — chained sub-ability: the **target player** reveals their hand and discards **every** card
  whose name is the named card.
- `K:Flashback:Sac<1/Creature>` — alternate cast-from-graveyard cost of sacrificing a creature.

## Engine work
Two new general mechanics, both keyed on the script's real tags (no retagging):

1. **`NameCard` spell/ability effect** (`src/effects/effect_name_card.cpp`,
   `effects::name_card`). New `EffectKind::NameCard`
   (`src/effects/effect_kind.{h,cpp}`, dispatched in `src/effects/effect_table.cpp`). The
   ability's controller chooses a card name (`ActionCategory::NAME_CARD`, value 34 — the same
   decision category Disruptor Flute's ETB name-a-card already uses). The candidate set is the
   distinct **nonland** vocab card names owned by the spell's target player (the player who
   will discard), ordered by copy count then name for deterministic replay — mirroring the
   Disruptor Flute candidate-building in `state_manager_statics.cpp`. The `ValidCards$
   Card.nonLand` filter is honored by excluding lands. The chosen name is stored in
   `cur_game.named_card` (new field in `src/classes/game.h`) so a chained `Card.NamedCard`
   sub-ability can reference it; it is cleared at the end of the top-level `NameCard`
   resolve (`src/components/ability.cpp`).

2. **`RevealDiscardAll` discard mode** (`src/effects/effect_discard.cpp`). `DiscardParams`
   gains a `mode` field (`src/components/ability_params.h`); the discard parse hook claims
   `Mode$ RevealYouChoose` / `Mode$ RevealDiscardAll` (only those two values, so other effects'
   `Mode$` — e.g. SetState `Mode$ Transform` — are untouched). The filter test was factored into
   `discard_filter_matches`, which now also understands the `NamedCard` constraint (card name ==
   `cur_game.named_card`). In `RevealDiscardAll` mode the target player reveals their hand and
   **every** matching card is moved to the graveyard (no choice), versus the existing
   `RevealYouChoose` single-pick path (Thoughtseize/Duress), which is unchanged.

3. **Sub-ability targets chosen at cast.** Cabal Therapy's primary `SP$ NameCard` does not
   itself target, but its `DB$ Discard` sub-ability has `ValidTgts$ Player`. The cast path
   (`src/action_processor.cpp`) now also runs `select_target` for any sub-ability with its own
   `ValidTgts$`, storing the target on the sub-ability template (CR 601.2c — targets are chosen
   as the spell is cast). `Ability::resolve` (`src/components/ability.cpp`) preserves a
   sub-ability's own target when it targets independently (only sub-abilities with
   `valid_tgts == "N_A"` inherit the parent's target, as before).

4. **Flashback Sacrifice cost (the triage's "already handled" was wrong).** The flashback
   parser dropped the `Sac<1/Creature>` portion of the flashback cost. Added
   `AltCost::sac_cost_spec` (`src/components/carddata.h`); the Flashback keyword parser
   (`src/parse.cpp`) now carries the sac filter into `flashback_alt_cost.sac_cost_spec`. The
   flashback cast path (`src/action_processor.cpp`) pays it by prompting a
   `SACRIFICE_PERMANENT` choice among matching controlled permanents (reusing
   `controlled_permanents_matching` / `prompt_permanent_choice`), and the flashback cast
   legality check (`src/systems/state_manager_actions.cpp`) only offers the cast when a matching
   permanent exists to sacrifice (CR 601.2f/601.3a). The flashback exile-after-resolution was
   already handled.

## Behavioral decisions (CR cites)
- **Naming a card** happens at resolution and may be any card name (CR 201.4); the engine
  restricts the offered set to nameable (nonland, per the script) vocab cards owned by the
  target player — the only decidable candidate set, matching the existing name-a-card decision.
- **Target chosen at cast** (CR 601.2c): the discarding player is a target of the spell, locked
  in as it is cast; only the named card is chosen on resolution.
- **Reveal + discard all** (the card's effect): the target player reveals their whole hand and
  discards every card matching the named name; naming a card the player does not hold discards
  nothing.
- **Flashback** (CR 702.34): cast from the graveyard for its flashback cost (here, sacrifice a
  creature; CR 601.2f cost payment), then the card is exiled as it leaves the stack.

## Tests (`train/test_harness.py`, seed 1)
- **(a) Name a held duplicate, discard all copies.** A casts Cabal Therapy (from hand,
  paying `{B}`), names Lightning Bolt; opponent B reveals a hand of 2 Lightning Bolt + 5 Island
  and discards **both** Bolts and nothing else (B hand 7→5, GY = 2 Lightning Bolt). PASS.
- **(b) Name a card not held.** B's hand is 7 Islands (Bolts are deeper in the deck, so still a
  nameable candidate); A names Lightning Bolt; B reveals hand and "No matching cards to discard"
  — hand stays 7. PASS.
- **(c) Flashback.** Cabal Therapy in A's graveyard, Birds of Paradise on A's battlefield;
  A casts via flashback → prompted "Sacrifice Birds of Paradise" → sacrifices it, names
  Lightning Bolt, B reveals hand and discards both Bolts, then "Cabal Therapy is exiled
  (flashback)" (A's GY contains only the sacrificed Birds, not Cabal Therapy). PASS.
- **Regression:** scripted-vs-scripted full games (black deck with 4 Cabal Therapy + 4 Duress
  vs a Lightning Bolt deck) over seeds 1/2/3 — each game produced a decisive winner (no draws),
  no non-fatal errors or unrecognized-param warnings, and Cabal Therapy was cast and resolved.

## Result
Implemented. Cabal Therapy casts from hand and via flashback (sacrificing a creature), names a
nonland card, and makes the target player reveal their hand and discard every copy of the named
card. General `NameCard` effect, `RevealDiscardAll` discard mode, cast-time sub-ability
targeting, and the flashback Sacrifice cost are reusable by future cards.
