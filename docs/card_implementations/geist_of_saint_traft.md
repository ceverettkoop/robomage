# Geist of Saint Traft (vocab index 330)

## Oracle text
Hexproof (This creature can't be the target of spells or abilities your opponents control.)
Whenever Geist of Saint Traft attacks, create a 4/4 white Angel creature token with flying
that's tapped and attacking. Exile that token at end of combat.

## Forge script (Source: pre-existing local; key tags)
`bin/resources/cardsfolder/g/geist_of_saint_traft.txt`
- `Types:Legendary Creature Spirit Cleric`, `PT:2/2`, `ManaCost:1 W U`
- `K:Hexproof` — already supported by the engine (legal-target predicate).
- `T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigToken` — the attack trigger.
- `SVar:TrigToken:DB$ Token | TokenScript$ w_4_4_angel_flying | TokenTapped$ True |
  TokenAttacking$ True | AtEOT$ ExileCombat` — create the Angel tapped and attacking, exile at
  end of combat.

Token script `bin/resources/tokenscripts/w_4_4_angel_flying.txt` was already present.

## Engine work
Mechanics added (general): **token-tapped-attacking + eot-exile** — two new `DB$ Token` params.

- `src/effects/effect_token.cpp`
  - `parse_token`: `TokenAttacking$ True` (→ `TokenParams::attacking`) and `AtEOT$ <mode>`
    (→ `TokenParams::at_eot`).
  - `token` handler: when `attacking`, each created token is put onto the battlefield already
    attacking the same defender the source creature is attacking — read from the source's
    `Creature::attack_target` (CR 508.4a: put onto the battlefield attacking, not declared, so no
    summoning-sickness / attack-trigger checks). Mirrors `effect_mobilize.cpp`. `TokenTapped$` was
    already supported. When `at_eot == "ExileCombat"`, a delayed trigger firing at
    `END_OF_COMBAT_BEGAN` is registered carrying exactly the created tokens.
  - New `exile_tokens` handler: exiles each token in `ab.targets` still on the battlefield (the
    graveyard-bound sibling of Mobilize's `sacrifice_tokens`).
- `src/ecs/events.h` — new `Events::END_OF_COMBAT_BEGAN` (22).
- `src/classes/game.cpp` — the COMBAT_DAMAGE → END_OF_COMBAT step transition now sends the
  END_OF_COMBAT_BEGAN event (there was no end-of-combat event before), so end-of-combat delayed
  triggers fire before the combat state is cleared.
- `src/effects/effect_kind.{h,cpp}`, `src/effects/effect_table.cpp`, `src/effects/effects.h` —
  register the `ExileTokens` effect category / handler.
- `src/components/ability_params.h` — `TokenParams::attacking` / `TokenParams::at_eot`.
- `src/card_vocab.h` — `Geist of Saint Traft` (330) and the `w_4_4_angel_flying` → "Angel" token
  identity (band index 916).

The script's real tags are honoured (no retag): the `T:Mode$ Attacks` trigger and its `DB$ Token`
Execute keep their categories; the new params are read alongside the existing TokenScript$/
TokenTapped$ handling.

## Behavioral decisions (CR cites)
- CR 508.4a — a token put onto the battlefield "tapped and attacking" is not a declared attacker,
  so it neither triggers "whenever a creature attacks" abilities nor needs to be able to attack; it
  attacks the same player/planeswalker the source is attacking.
- CR 512 — "at end of combat" abilities trigger as the end-of-combat step begins; the exile happens
  after combat damage (the Angel deals its damage this combat, then is exiled).
- CR 111.x — the token ceases to exist once it leaves the battlefield (exile).
- CR 702.11 — Hexproof (already implemented) keeps the Geist off opponents' target menus.

## Tests
Built `make` clean (only the pre-existing cosmetic parse.cpp warnings). Verified with
`train/test_harness.py` (preset battlefield, semantic `--play` with seat keys):
- Geist preset on A's battlefield attacks Player B: the attack trigger creates a **4/4 Angel**
  logged as "tapped and attacking". In combat, both Geist (2) **and** the Angel (4) deal damage to
  B (20 → 14), confirming the token is attacking. At end of combat the log shows "Angel is exiled
  at end of combat" and the token ceases to exist — gone from the battlefield the following turn.
- Hexproof: with Geist on A's battlefield, Player B's Lightning Bolt offers only Player A / Player B
  as targets — the opponent's Geist is not a legal target.
- CI gate below.

## Result
Done. General `TokenAttacking$` (enter tapped and attacking, reusing the Mobilize combat-entry
path) and `AtEOT$ ExileCombat` (end-of-combat delayed exile, via a new END_OF_COMBAT_BEGAN step
event and an `ExileTokens` handler) implemented; Geist of Saint Traft makes a 4/4 flying Angel that
attacks alongside it and is exiled at end of combat, and it has Hexproof.
