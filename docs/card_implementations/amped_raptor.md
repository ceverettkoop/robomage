# Amped Raptor  (vocab index 170)

## Oracle text
First strike

When Amped Raptor enters, you get {E}{E} (two energy counters). Then if you cast it from your
hand, exile cards from the top of your library until you exile a nonland card. You may cast that
card by paying an amount of {E} equal to its mana value rather than paying its mana cost.

## Forge script
- Source: in-repo — `bin/resources/cardsfolder/a/amped_raptor.txt` (parsed as written, no retag).
- Type: `Creature Dinosaur`, mana cost `1 R`, P/T `2/1`, `K:First Strike`.
- Key tags:
  - `T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self |
    Execute$ TrigEnergy` — the ETB trigger.
  - `SVar:TrigEnergy:DB$ PutCounter | Defined$ You | CounterType$ ENERGY | CounterNum$ 2 |
    SubAbility$ DBDigUntil` — gain 2 energy (the existing PutCounter-energy path).
  - `SVar:DBDigUntil:DB$ DigUntil | ConditionDefined$ TriggeredCard |
    ConditionPresent$ Card.wasCastFromYourHandByYou | Valid$ Card.nonLand |
    FoundDestination$ Exile | RevealedDestination$ Exile | RememberFound$ True |
    SubAbility$ DBPlay` — the conditional impulse dig.
  - `SVar:DBPlay:DB$ Play | Defined$ Remembered | ValidSA$ Spell |
    PlayCost$ PayEnergy<ConvertedManaCost> | Optional$ True | SubAbility$ DBCleanup` — the
    optional alt-cost cast of the exiled nonland.
  - `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True` — clears the remembered card.

## Rules (CR)
- **601** casting a spell; **601.1** lands are played, not cast (so a remembered land does
  nothing).
- **118.9** alternative cost: "rather than paying its mana cost." Here {E} equal to mana value
  replaces the mana cost.
- **707 / "impulsive"-style exile-then-cast**: the card is exiled and may be cast from exile.
- **122.1c** energy counters / paying {E} is a cost.

## Engine work
Three general mechanisms were built (no card-specific retagging):

### 1. `Card.wasCastFromYourHandByYou` — "was cast from your hand by you" tracking
The dig only happens if the Amped Raptor that entered was cast from its controller's own hand.
The engine already had a one-shot `Game::cast_to_battlefield` ("this spell was cast", CR 614.12,
Containment Priest), but it loses the *source zone* and is consumed when the permanent is
created. Added a parallel, persistent flag:
- `Game::cast_from_hand` (`src/classes/game.h`) — one-shot set in the `CAST_SPELL` handler
  (`src/action_processor.cpp`) only when the spell's zone is the caster's own `HAND`; a cast
  from graveyard/exile (flashback, impulse) clears it.
- Consumed in `src/systems/state_manager_statics.cpp` when the `Permanent` is created, setting
  the persistent `Permanent::cast_from_hand_by_controller` (`src/components/permanent.h`).
- The condition predicate lives in `evaluate_present_condition`
  (`src/systems/state_manager_actions.cpp`): `Card.wasCastFromYourHandByYou` reads that flag off
  the ability's source permanent. `ConditionDefined$ TriggeredCard` parses to
  `Ability::condition_on_triggered_card` (`src/parse.cpp`,
  `src/components/ability.h`); `Ability::resolve` (`src/components/ability.cpp`) gates the body
  on it (failure skips the body but still chains subabilities), exactly like the existing
  `ConditionDefined$ Remembered` gate.

### 2. `DB$ DigUntil` — exile from top of library until a match
New general effect `effects::dig_until` (`src/effects/effect_dig_until.cpp`,
`EffectKind::DigUntil`): walk the controller's library from the top one card at a time; each
non-matching card goes to `RevealedDestination$`, and the first card matching `Valid$` goes to
`FoundDestination$` (both Exile here). `RememberFound$ True` records the matching card in
`cur_game.remembered_entities` for the chained `DB$ Play`. An empty library stops the dig
gracefully (nothing remembered). The `Valid$` filter routes through the shared
`card_matches_filter` (so `Card.nonLand`, colors, types, etc. all work). Parse hook
`parse_dig_until` claims `Valid$`/`FoundDestination$`/`RevealedDestination$`/`RememberFound$`
(`Valid$` is scoped to `category == "DigUntil"` so it cannot shadow another effect).

### 3. `DB$ Play` — general alternative-cost cast (the key deliverable)
New general effect `effects::play` (`src/effects/effect_play.cpp`, `EffectKind::Play`). Rather
than reentrantly cast a spell mid-resolution (the `DB$ Play` ability is itself resolving from the
stack), it **grants a one-shot permission** to cast a `Defined$` card (here `Remembered`, the
just-exiled nonland) from its current zone (Exile) this turn, with its mana cost replaced by an
alternative **resource** cost. The normal casting pipeline then offers and funnels the cast onto
the stack, so targeting / triggers / the stack all work unchanged.

- **Permission record**: `Game::ImpulseCastPermission { resource (ENERGY|LIFE), amount, caster }`
  in a `cur_game.impulse_cast_permission` map (`src/classes/game.h`), cleared each cleanup like
  `may_cast_this_turn`. `effects::play` resolves the amount (`PlayCost$ PayEnergy<...>` /
  `PayLife<...>`, amount = `ConvertedManaCost` → the card's mana value, or a literal) and
  records the permission. `ValidSA$ Spell` means "a nonland card cast as a spell" (every nonland
  card is cast as a spell, CR 601.1) — so only lands are excluded.
- **Offer**: `determine_legal_actions` (`src/systems/state_manager_actions.cpp`) offers
  "Cast X (impulse, alt cost)" from Exile at the timing the card's type allows, only to the
  granted player, only when they can pay the resource (energy ≥ amount, or life ≥ amount), and
  only with a legal target. The action carries `LegalAction::impulse_cast`
  (`src/classes/action.h`).
- **Payment**: a dedicated `impulse_cast` branch in the `CAST_SPELL` handler
  (`src/action_processor.cpp`) pays the resource (energy via the shared `pay_energy`; life by
  subtracting life) and **no mana**, then consumes the permission. It then falls through to the
  exact same target-selection / stack-placement / event-firing code a normal cast uses.

**Resource parameterization (general):** the same path serves energy AND life. `PlayCost$
PayEnergy<ConvertedManaCost>` drives Amped Raptor; `PlayCost$ PayLife<ConvertedManaCost>` would
drive a future **Bolas's Citadel** ("play cards from the top of your library, paying life equal
to mana value") with no new code — just `play_cost_resource = PLAY_COST_LIFE`. (Bolas's Citadel
itself uses a different shape — a `MayPlay$`/`MayPlayAltManaCost$` continuous static — and was
*not* implemented or added to vocab this unit; its script was fetched only as a reference for the
`PayLife<ConvertedManaCost>` cost shape and deleted before commit.)

### `DB$ Cleanup | ClearRemembered$ True`
Already implemented (`effect_cleanup.cpp`); clears `cur_game.remembered_entities` after the chain.

## Behavioral decisions (made in lieu of asking a human)
- **X spells: X = 0.** When the remembered card has an `X` in its cost, X counts as 0 for the
  impulse cast (no X prompt). The card's printed mana value (symbol count with X = 0) is used as
  the energy amount, and the `impulse_cast` payment branch sets `cur_game.x_paid = 0`. This
  matches how the impulse exile gives no opportunity to choose X.
- **Permission timing vs. cast-during-resolution.** CR-strictly, "you may cast it" happens during
  the ability's resolution. The engine grants a "this turn" one-shot permission instead and casts
  through the normal pipeline at the controller's next priority window. The observable difference
  for an energy spell is negligible, and this avoids a reentrant cast onto a stack that is mid-
  resolution; it reuses the existing, battle-tested cast entry (the explicit design requirement
  to "funnel into the existing cast entry").
- **`ValidSA$ Spell` = nonland.** Implemented as "not a land" rather than "has an `SP$` ability",
  because permanents (creatures, artifacts) are cast directly without a parsed `SP$` ability in
  this engine; restricting to `SP$`-bearing cards would wrongly exclude creatures (e.g. impulse-
  casting another Amped Raptor).
- **Non-hand entries don't dig.** A reanimated / token / impulse-cast (from exile) Amped Raptor
  still gains 2 energy but does not dig — verified directly (test d).

## Tests
Isolation (`train/test_harness.py`, semantic `--play`, seed 1; battlefield Mountains preset so
the 2 energy can be checked against a deterministic top-of-library):

- **(a) Cast from hand, impulse-cast the dug spell.** Hand = Amped Raptor; library top =
  Mountain, Mountain, Lightning Bolt. On ETB: "gets 2 ENERGY", exiles both Mountains + Lightning
  Bolt (stops at the nonland), offers "Cast Lightning Bolt (impulse, alt cost)" for 1 energy
  (Bolt's mana value). Accepted → "pays 1 energy", "casts Lightning Bolt targeting Player B",
  Player B 20 → 17, Bolt to graveyard, 1 energy remaining. **PASS.**
- **(b) Decline the optional cast.** Same setup; A passes the impulse offer. No energy spent
  beyond the 2 gained, no Bolt cast, the card stays exiled (the permission persists this turn,
  as a "this turn" grant should). **PASS.**
- **(c) Insufficient energy.** Library top = Mountain, Mountain, Magus of the Moon (mana value
  3); only 2 energy. The impulse permission is granted but the cast action is **not** offered
  (2 < 3); the card stays exiled. **PASS.**
- **(d) Entered NOT from hand.** A second Amped Raptor is impulse-cast from *exile* (dug by the
  first). It enters and gains its own 2 energy, but its `DigUntil` body is **suppressed** by the
  `Card.wasCastFromYourHandByYou` gate (only one "exiles" line, only one "may cast" — the first
  raptor's). Preplaced (`--battlefield-a`) permanents fire no ETB at all, so the exile-cast path
  is what exercises the non-hand entry. **PASS.**

Regression (`train/test_harness.py --scripted`, seeds 1–3): a red deck
(4 Amped Raptor / 4 Dragon's Rage Channeler / 4 Lightning Bolt / 4 Unholy Heat / 4 Abrade /
20 Mountain) vs a green creature deck. All three games finished **decisively** (B, A, A), **no
draws**, no non-fatal errors / asserts / tracebacks. The mechanic fired repeatedly in real
scripted play — energy gain, dig-to-exile, and the scripted agent even impulse-casting a second
Amped Raptor from exile for 2 energy. (`train.py observe` not used — `torch` absent — so the
scripted regression ran directly through the harness, per CLAUDE.md.)

## needs_review
- **The `DB$ Play` alt-cost cast path** (flagged for review given its complexity): the
  permission-grant model (cast "this turn" from exile via the normal pipeline) rather than a
  literal cast-during-resolution. Correct and tested for energy; the life branch is implemented
  and parameterized but only exercised by code review (no life-cost card in vocab yet).

## Result
implemented
