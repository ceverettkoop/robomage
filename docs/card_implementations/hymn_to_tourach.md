# Hymn to Tourach (vocab index 165)

## Oracle text
Target player discards two cards at random.

## Forge script
Source: pre-existing local script (`bin/resources/cardsfolder/h/hymn_to_tourach.txt`).

```
Name:Hymn to Tourach
ManaCost:B B
Types:Sorcery
A:SP$ Discard | ValidTgts$ Player | NumCards$ 2 | Mode$ Random | SpellDescription$ Target player discards two cards at random.
```

Key tags: `SP$ Discard`, `ValidTgts$ Player`, `NumCards$ 2`, `Mode$ Random`.
`NumCards$ 2` is parsed into `Ability::amount` (the general count handler in `parse.cpp`).

## Engine work
General, reusable additions in `src/effects/effect_discard.cpp` (`effects::discard`)
and its parser (`effects::parse_discard`):

- `parse_discard` now also accepts `Mode$ Random` (alongside the existing
  `RevealYouChoose` / `RevealDiscardAll`), storing it in `DiscardParams::mode`.
- `discard()` gained a general `Mode == "Random"` path that runs **before** the
  hand is revealed (a random discard does not reveal the hand). It reads the
  discard count from `Ability::amount` (the `NumCards$ N` value — **not** hardcoded),
  clamps it to the target's hand size, shuffles a copy of the target player's hand
  with the game's seeded RNG (`cur_game.gen`, the same generator
  `Orderer::shuffle_library` uses), and discards the first `count` cards. No player
  makes a choice; each discarded card is recorded in the belief state via
  `mark_card_revealed` as it enters the graveyard (a public zone).

Using `cur_game.gen` keeps replays deterministic, matching how library shuffling
draws randomness in `src/systems/orderer.cpp`.

## Behavioral decisions
- "Discards at random" (CR 701.8e/701.8f): the player discards cards chosen at
  random, without anyone choosing them. Implemented as a uniform random selection
  from the hand, no `DISCARD` action prompt is presented to any player.
- If the hand holds fewer cards than `NumCards$`, the count is clamped to the hand
  size and the player discards all remaining cards (CR 701.8: discard as many as
  possible).
- Determinism: random selection uses the seeded `cur_game.gen`, so the same seed
  reproduces the same discards across replays.

## Tests
Verified with `train/test_harness.py` (semantic `--play`, fixed seed):
- (a) Opponent with a 7-card hand, A casts Hymn targeting Player B → "Player B
  discards <X> at random" twice, hand drops by 2, and **no** discard-choice menu is
  presented (random, not chosen).
- (b) Clamp: drained Player B to a 1-card hand, then a further Hymn discarded just
  that 1 card (NumCards 2 clamped to 1), no crash.
- (c) Determinism: seed 11 reproduced the same two cards (Forest, Swamp) on two runs;
  seed 99 produced a different pair (Swamp, Lightning Bolt) — confirming the pick is
  random and seed-driven.
- Regression: scripted full games, mono-black deck (incl. 4x Hymn to Tourach) vs
  `mav`, seeds 1/2/3 — each produced a decisive winner (A wins, B wins, B wins),
  no draws, no engine ERROR/Traceback. (Pre-existing cosmetic parser WARNINGs for
  unrelated cards are unchanged.)

## Result
Implemented. General `Mode$ Random` + `NumCards$ N` random discard via the seeded
RNG; Hymn to Tourach discards two cards at random correctly, with clamping and
determinism.
