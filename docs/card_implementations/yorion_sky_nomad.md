# Yorion, Sky Nomad  (vocab index 289)

## Oracle text (per the in-repo Forge script)
Legendary Creature — Bird Serpent, mana cost `{3}{W/U}{W/U}` (mana value 5; hybrid mana is now
supported — see `_hybrid_mana.md`), 4/5, Flying.

Companion — Your starting deck contains at least twenty cards more than the minimum deck size. (If
this card is your chosen companion, you may put it into your hand from outside the game for `{3}`
any time you could cast a sorcery.)

When Yorion enters, exile any number of other nonland permanents you own and control. Return those
cards to the battlefield at the beginning of the next end step.

## Forge script
- Source: in-repo — `bin/resources/cardsfolder/y/yorion_sky_nomad.txt` (parsed as written, no retag).
- Key tags:
  - `K:Companion:Special:DeckSizePlus20:...` — the Companion keyword (the new mechanic). The third
    colon field `DeckSizePlus20` is the deckbuilding-restriction token.
  - `K:Flying` (already supported).
  - `T:Mode$ ChangesZone | Destination$ Battlefield | Execute$ TrigExile` — the ETB trigger.
  - `SVar:TrigExile:DB$ ChangeZone | Origin$ Battlefield | Destination$ Exile |
    ChangeType$ Permanent.nonLand+Other+YouOwn+YouCtrl | SelectPrompt$ ... | Hidden$ True |
    ChangeNum$ X | RememberChanged$ True | SubAbility$ DelTrig` — exile any number of the chosen
    nonland permanents (player-driven multi-select, not targeting; `X` = `Count$` of the matching
    set, i.e. "any number up to all").
  - `SVar:DelTrig:DB$ DelayedTrigger | Mode$ Phase | Phase$ End of Turn | Execute$ TrigReturn |
    RememberObjects$ RememberedLKI` — schedule the return at the next end step.
  - `SVar:TrigReturn:DB$ ChangeZone | Origin$ Exile | Destination$ Battlefield |
    Defined$ DelayTriggerRememberedLKI` — return those exact cards.

## Rules (CR)
- **702.139 Companion.** A keyword ability of a card outside the game (its sideboard) with a
  deckbuilding restriction. If the starting deck meets the restriction, the card is the player's
  *chosen companion*; once per game, any time the player could cast a sorcery, they may pay `{3}`
  as a special action to put it from outside the game into their hand (702.139c–e).
- **603.7 / 603.7b** delayed triggered ability ("return … at the beginning of the next end step").
- **702.9** Flying (pre-existing).
- The returned permanents are **new objects** under their owner's control (their ETBs re-trigger).

## Engine work
Both mechanics are built generally; a future card reuses the same framework.

### 1. Companion (CR 702.139) — deck-restriction gate + once-per-game pay-{3}-from-sideboard
- **Parse** (`src/parse.cpp`, the card `K:` loop): a `Companion:` arm sets
  `CardData::is_companion` and stores the restriction token (`DeckSizePlus20`) in
  `CardData::companion_restriction` (`src/components/carddata.h`). `SelectPrompt$` was added to the
  parser's cosmetic `ignored_keys` set (it is prompt prose, like `TgtPrompt$`).
- **Deckbuilding gate + game-start setup** (new `src/companion.{h,cpp}`):
  - `deck_meets_companion_restriction(restriction, deck)` evaluates the restriction against a `Deck`.
    `DeckSizePlus<N>` ⇒ the main deck's total card count must be `>= MINIMUM_DECK_SIZE (60) + N`
    (Yorion: `>= 80`). Generalized over `N`; an unrecognized token fails closed.
  - `setup_companions(deck_a, deck_b, orderer)` — called once per game in `play_single_game`
    (`src/main.cpp`) after libraries/preset sideboard entities are in place. For each player it
    finds a Companion-keyworded card (first an existing `Zone::SIDEBOARD` entity — the test-harness
    `--sideboard` preset / a bo3 sideboard — otherwise a name in `Deck::sideboard`, which it
    instantiates), checks the restriction, and on success records it in `Player::chosen_companion`
    (`src/components/player.h`).
- **Special action** (`src/systems/state_manager_actions.cpp`, `determine_legal_actions`): mirroring
  the play-land sorcery-speed gate (main phase, own turn, empty stack), when the player has a
  `chosen_companion` still in the sideboard, has not used it this game
  (`Player::companion_brought_to_hand`), and can pay `{3}` (`can_pay_mana`), a `SPECIAL_ACTION` is
  offered (category `PLAY_FREE`, flagged `LegalAction::companion_to_hand`, `src/classes/action.h`).
- **Dispatch** (`src/action_processor.cpp`, `SPECIAL_ACTION` case): the `companion_to_hand` branch
  pays `{3}` via the shared `prompt_mana_payment` (rewinding on cancel), moves the entity
  Sideboard → Hand (`Orderer::add_to_zone`), and sets `companion_brought_to_hand` so it can be used
  only once per game.

### 2. ETB blink — non-targeted "any number of permanents" exile + delayed end-step return
- **Exile selection** (`src/effects/effect_change_zone.cpp`, new branch in `change_zone`): a
  non-targeted `Origin$ Battlefield` ChangeZone with a non-empty `ChangeType$` filter. It enumerates
  the battlefield via `battlefield_permanents(orderer->mEntities)`, filters with
  `permanent_matches_any(e, ab.change_type, ctx)` where `ctx.source = ab.source` (drives `+Other`)
  and `ctx.controller = owner` (drives `+YouOwn/+YouCtrl`), and presents a player-driven
  multi-select that re-derives candidates after each pick and offers a "done" (so `ChangeNum$ X` /
  "any number" needs no count math — the cap is "all matching"). Each chosen permanent is moved with
  the existing `change_zone_move` helper and pushed to `cur_game.remembered_entities` when
  `RememberChanged$`. Distinct from the empty-filter self-move branch above it and from
  `ChangeZoneAll` (which moves the whole set with no choice).
- **Delayed return** — entirely reused, no new code: `DB$ DelayedTrigger` (`effect_delayed_trigger`)
  schedules a fire at `END_STEP_BEGAN`, capturing the remembered set via `RememberObjects$
  RememberedLKI`; at the next end step the `Defined$ DelayTriggerRememberedLKI` ChangeZone
  (`effect_change_zone`'s blanket `defined_remembered` branch) returns those exact cards to the
  battlefield under their owner's control, as fresh objects.

## Tests
Isolation (`train/test_harness.py`, JSON scenarios + semantic `--play`, seed 1):
- **(a) Companion — restriction met.** Main deck `80 Plains` + sideboard `1 Yorion, Sky Nomad`,
  3 Plains preset. At sorcery speed the menu offers "Companion: pay {3}, put Yorion, Sky Nomad into
  your hand"; taking it pays `{3}`, moves Yorion sideboard → hand, and the action then **disappears**
  (once per game). **PASS.**
- **(b) Companion — restriction NOT met.** Main deck `60 Plains` (< 80) + same sideboard: the
  Companion action is **never offered**. **PASS.**
- **(c) ETB blink.** Cast Yorion with a Grizzly Bears (own nonland permanent) in play; the ETB
  offers "Exile Grizzly Bears" / "Done"; exiling it then returns it to the battlefield at the next
  end step as a **new object** (re-enters with summoning sickness). Lands/tokens are not offered
  (nonland filter). **PASS.**
- **(d) Flying.** Yorion attacks; the opponent's ground Grizzly Bears is **not** a legal blocker —
  the declare-blockers step is skipped and Yorion deals 4 to the defending player. **PASS.**

No non-fatal errors / asserts / tracebacks across the scenarios (the `SelectPrompt$` warning is gone).
A `--scripted` regression (real `delver` deck) ran without crashes/asserts. (`train.py observe` not
used — `torch` absent — so scripted regression ran directly through the harness, per CLAUDE.md.)

## Caveats / follow-ups
- **Hybrid mana cost — CLOSED.** Forge writes the hybrid symbol `{W/U}` as `WU`, so Yorion's script
  cost is `3 WU WU` = `{3}{W/U}{W/U}`. The engine now parses hybrid pips generally (see
  `docs/card_implementations/_hybrid_mana.md`): Yorion's **mana value is 5** (each `{W/U}` counts 1,
  CR 202.3f), each hybrid pip is payable with **either** white or blue mana, and Yorion's color
  identity is **both white and blue** (verified: Red Elemental Blast's "destroy target blue
  permanent" legally targets and destroys Yorion). Verified castable off `{3}` + 2 W, off `{3}` + 2 U,
  and off `{3}` + W + U (5 mana total, not 7), and correctly *not* castable off 5 red sources (no
  W/U) or 4 lands (mana value 5).
- **Companion game-start reveal (CR 702.139c) simplified.** This 2-player engine has no formal
  "reveal your companion from outside the game" pre-game step; the load-bearing, playable behavior —
  the deck-restriction gate plus the once-per-game pay-{3}-to-hand special action — is implemented
  fully. The reveal is implicit (the chosen companion is determined at game start by
  `setup_companions`).

## Result
implemented
