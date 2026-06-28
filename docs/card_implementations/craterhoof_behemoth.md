# Craterhoof Behemoth

```
Name:Craterhoof Behemoth
ManaCost:5 G G G
Types:Creature Beast
PT:5/5
K:Haste
T:Mode$ ChangesZone | ValidCard$ Card.Self | Origin$ Any | Destination$ Battlefield | Execute$ BehemothPump | TriggerDescription$ When CARDNAME enters, creatures you control gain trample and get +X/+X until end of turn, where X is the number of creatures you control.
SVar:BehemothPump:DB$ PumpAll | ValidCards$ Creature.YouCtrl | KW$ Trample | NumAtt$ +X | NumDef$ +X
SVar:X:Count$Valid Creature.YouCtrl
SVar:PlayMain1:TRUE
Oracle:Haste\nWhen Craterhoof Behemoth enters, creatures you control gain trample and get +X/+X until end of turn, where X is the number of creatures you control.
```

**Oracle:** Haste. When Craterhoof Behemoth enters, creatures you control gain trample and
get +X/+X until end of turn, where X is the number of creatures you control.

Vocab index: **218** (`src/card_vocab.h`).

Forge script source: pre-existing local script
`bin/resources/cardsfolder/c/craterhoof_behemoth.txt` (unchanged).

## Engine work

**None — covered by existing handlers.** This card combines three already-shipping pieces:

- **Haste** — the `K:Haste` keyword, parsed and applied as for any other creature.
- **ETB trigger** — `T: Mode$ ChangesZone … Destination$ Battlefield … Execute$ BehemothPump`
  is the standard self-ETB ChangesZone trigger handled by the trigger system
  (`src/systems/state_manager_triggers.cpp`).
- **`DB$ PumpAll`** — `effects::pump_all` (`src/effects/effect_pump_all.cpp`) applies a
  power/toughness bonus and grants a keyword (`KW$ Trample`) to every permanent matching
  `ValidCards$ Creature.YouCtrl`.
- **Dynamic X** — `NumAtt$ +X`/`NumDef$ +X` with `SVar:X:Count$Valid Creature.YouCtrl` is
  resolved by the existing `Count$Valid …` SVar evaluator (`src/svar_eval.cpp`), counting the
  creatures the controller has on the battlefield at resolution.

The specific combination (ETB PumpAll granting a keyword with a `Count$Valid`-driven dynamic
`+X/+X`) was not previously proven end-to-end by a single shipping card, hence the
behavior test below.

## Behavioral decisions

- X is evaluated at resolution and includes Craterhoof itself (it is on the battlefield by the
  time its ETB trigger resolves), matching CR — "the number of creatures you control" counts
  Craterhoof. Test confirms X=3 with two other creatures present (2 Birds + Craterhoof).

## Tests (`train/test_harness.py`)

- **Pump-all + trample + dynamic X:** battlefield = 2 Birds of Paradise (0/1) + lands; cast
  Craterhoof. Result: trigger fires, X = 3 (Craterhoof + 2 Birds). Craterhoof 5/5 → 8/8 and gains
  Trample; each Birds 0/1 → 3/4 and gains Trample. Each creature shows `(T)` trample status.
- **Real-game regression:** `temp/crater_a` (mav list with 3 Craterhoof swapped in) vs
  `temp/mav_b`, scripted, seeds 1/2/3. Results: A wins, A wins, B wins. No non-fatal errors,
  no draws.

## Result

Implemented — covered by existing handlers, behavior verified.
