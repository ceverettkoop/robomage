# Show and Tell

## Oracle text
Each player may put an artifact, creature, enchantment, or land card from their hand onto the battlefield. ({2}{U} sorcery)

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/s/show_and_tell.txt`).

Key tags:
- `A:SP$ ChangeZone | Origin$ Hand | Destination$ Battlefield | ChangeType$ Creature,Artifact,Enchantment,Land | DefinedPlayer$ Player | ChangeNum$ 1`
  — the bare `DefinedPlayer$ Player` means EACH player.

## Engine work
- **Mechanics added (general): `each-player-put-from-hand`** — a `ChangeZone` variant where EACH
  player, in APNAP order, MAY put one matching card from THEIR OWN hand onto the battlefield.
- `src/effects/effect_change_zone.cpp` — `change_zone` gains a branch (guarded on
  `ab.defined == "Player" && origin == HAND && destination == BATTLEFIELD && valid_tgts == "N_A"`)
  dispatching to the new static `each_player_put_from_hand`, placed BEFORE the generic
  caster-scoped search path (which would otherwise let only the caster put a card).
- `each_player_put_from_hand` iterates both players in APNAP order (active player first, CR 101.4
  / 405.6 via `cur_game.player_a_active`). For each player it builds the candidate menu from THAT
  player's live hand filtered by the ChangeType (`card_matches_any`, printed characteristics —
  the card is not a permanent yet), adds a "Put nothing" decline option (the effect is a "may"),
  and asks that player (`fctx.ask` seated on them). A chosen card is moved to the battlefield and
  its controller set to that player (CR 110.2a — enters under its owner's control; no mana is
  paid, so a big creature simply enters, it is not cast). ETBs fire via the normal SBA pass.
- Suspend-safe: a new `EachPlayerPutRt` (`src/resolution_frame.h`, added to the `EffectRuntime`
  variant) persists only the player index; each player's menu is a live-menu loop re-derived from
  their hand every pass.

CR: 101.4 / 405.6 (APNAP order for "each player" effects), 110.2a (enters under owner's control).

## Behavioral decisions
- Modeled per-player as SEPARATE decisions (each player chooses privately from their own hand),
  which matches the intent though MTG treats the puts as one simultaneous action — observationally
  identical for a sorcery since nothing happens between the two choices.
- Reused the existing single-actor Hand->Battlefield put machinery only conceptually; the
  per-player loop is its own compact live-menu handler (like the Yorion battlefield multi-select),
  avoiding entangling the big caster-scoped search-loop rt state.

## Tests (isolation)
Harness — A hand: Show and Tell + Grizzly Bears (only 3 Islands for mana); B hand: Goblin Guide:
- A casts Show and Tell -> A (active) is asked FIRST, puts Grizzly Bears; then B is asked, puts
  Goblin Guide. Grizzly Bears enters A's battlefield [2/2] (SICK), Goblin Guide enters B's [2/2].
  Grizzly Bears (1G) entered with no green mana available -> confirmed it was PUT, not cast.
  APNAP + own-hand confirmed (A's menu had A's cards, B's menu had B's). PASS.
- B chooses "Put nothing" -> Goblin Guide stays in B's hand (the "may" declines). PASS.
- CI gate: `make check` tiers pygen/vocab/smoke.

## Result
Implemented.
