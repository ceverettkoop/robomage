# Witherbloom Command (vocab index 260)

## Oracle text
Choose two —
- Target player mills three cards, then you return a land card from your graveyard to your hand.
- Destroy target noncreature, nonland permanent with mana value 2 or less.
- Target creature gets -3/-1 until end of turn.
- Target opponent loses 2 life and you gain 2 life.

(Sorcery — `{B}{G}`)

## Forge script
Source: pre-existing local script at `bin/resources/cardsfolder/w/witherbloom_command.txt`.

```
A:SP$ Charm | CharmNum$ 2 | Choices$ DBMill,DBDestroy,DBPump,DBLoseLife
SVar:DBMill:DB$ Mill | NumCards$ 3 | ValidTgts$ Player | SubAbility$ DBReturn | ...
SVar:DBReturn:DB$ ChangeZone | Origin$ Graveyard | Destination$ Hand | Hidden$ True | Mandatory$ True | ChangeType$ Land.YouOwn
SVar:DBDestroy:DB$ Destroy | ValidTgts$ Permanent.nonCreature+nonLand+cmcLE2 | ...
SVar:DBPump:DB$ Pump | ValidTgts$ Creature | NumAtt$ -3 | NumDef$ -1 | ...
SVar:DBLoseLife:DB$ LoseLife | ValidTgts$ Player.Opponent | LifeAmount$ 2 | SubAbility$ DBGainLife | ...
SVar:DBGainLife:DB$ GainLife | LifeAmount$ 2
```

Key tags: `SP$ Charm` + `CharmNum$ 2` (choose-two modal); the four `Choices$` SVars are
ordinary chained sub-abilities (`Mill`, `ChangeZone`, `Pump`, `LoseLife`+`GainLife`).

## Engine work
The modal scaffolding (`Charm` with `CharmNum$ 2`) and every individual effect already
exist. Two pre-existing **general** bugs in player-targeting effects surfaced while
testing and were fixed (they affect any targeted Mill / "target player loses life", not
just this card):

- `src/effects/effect_mill.cpp` — `mill()` always milled `ab.controller`, ignoring the
  chosen target. Added the standard player-target resolution (mirrors `effect_draw.cpp`):
  if `ab.target` is a `Player` entity, mill that player; otherwise the controller mills
  (self-mill / `Defined$ You`).
- `src/effects/effect_lose_life.cpp` — `lose_life()` always drained `ab.controller`, so
  "Target opponent loses 2 life" would have hit the caster. Added a `loser` redirect: when
  `ab.target` is a `Player` entity, that player loses the life. The dynamic-amount
  reference (`evaluate_dynamic_amount`) is deliberately left on the controller's "you" so
  count-based self-drains are unchanged.

No retagging; no card script edits. `DBReturn` (return a land from your own graveyard),
`DBDestroy` (cmcLE2 filter), and `DBPump` (-3/-1) route through their existing handlers.

## Behavioral decisions (CR cites)
- Choose two different modes (CR 700.2 / 601.2b): the engine offers the remaining modes
  again after each pick and re-prompts targets per mode.
- "Target player mills three" can target either player (CR 109.5); the milled player is
  the target, while the follow-up land return is always from the caster's graveyard
  (`ChangeType$ Land.YouOwn`).
- "Target opponent loses 2 life and you gain 2 life": the opponent loses, the caster gains
  (separate `LoseLife` target + `GainLife` self sub).

## Tests (`train/test_harness.py`)
Single-cast scenario, `{B}{G}` off a Swamp + Forest, opponent with Expedition Map
(artifact, mv 1) on the battlefield and an Island library:
- `Destroy` mode → Expedition Map destroyed.
- `Mill` mode targeting Player B → "Player B mills Island" x3 (target honored after fix).
- `LoseLife` mode targeting Player B → "Player B loses 2 life (now at 18)", "Player A
  gains 2 life (now at 22)".

Regression: scripted full games, `temp/wbc_a` (4 Witherbloom Command + Grizzly Bears +
Swamp/Forest) vs `temp/wbc_b` (Expedition Map + Island/Mountain), seeds 1 and 7 — both
end with a decisive winner (Player A), no draws, no non-fatal errors.

## Result
Implemented. Modal choose-two resolves all four modes; fixed two general player-target
bugs (Mill, LoseLife). Build clean; regression clean across two seeds.
