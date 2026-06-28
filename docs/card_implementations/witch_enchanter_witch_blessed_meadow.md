# Witch Enchanter // Witch-Blessed Meadow (vocab indices 263 / 264)

A **modal double-faced card** (MDFC): the player chooses, when playing it from hand, to cast
the front (a creature) or play the back (a land).

## Oracle text
**Witch Enchanter** — `{3}{W}` Creature — Human Warlock, 2/2
When Witch Enchanter enters, destroy target artifact or enchantment an opponent controls.

**Witch-Blessed Meadow** — Land (back face)
As Witch-Blessed Meadow enters, you may pay 3 life. If you don't, it enters tapped.
{T}: Add {W}.

## Forge script
Source: pre-existing combined local script
`bin/resources/cardsfolder/w/witch_enchanter_witch_blessed_meadow.txt` (one file, both faces,
`AlternateMode:Modal`). Per the DFC rule, no separate front-name file was created.

Front:
```
T:Mode$ ChangesZone | Origin$ Any | Destination$ Battlefield | ValidCard$ Card.Self | Execute$ TrigDestroy
SVar:TrigDestroy:DB$ Destroy | ValidTgts$ Artifact.OppCtrl,Enchantment.OppCtrl
```
Back:
```
R:Event$ Moved | ValidCard$ Card.Self | Destination$ Battlefield | ReplaceWith$ DBTap
SVar:DBTap:DB$ Tap | ETB$ True | Defined$ Self | UnlessCost$ PayLife<3> | UnlessPayer$ You
A:AB$ Mana | Cost$ T | Produced$ W
```

## Engine work
- **Front face: none required.** The ETB-destroy uses the existing `ChangesZone→Battlefield`
  trigger + `DB$ Destroy` with a `ValidTgts$ Artifact.OppCtrl,Enchantment.OppCtrl` comma-OR
  target spec — all already supported. Both faces are parsed (`parse_card_face` on the
  `ALTERNATE`-split back into `card.backside`), so both faces load.
- Both faces registered in `src/card_vocab.h` (front 263, back 264), mirroring the DFC
  convention used for Delver of Secrets / Insectile Aberration and Ajani Nacatl Pariah /
  Avenger. The "no card file found for 'Witch-Blessed Meadow'" line during `gen_card_costs.py`
  is the expected DFC back-face note (same as Insectile Aberration / Ajani Avenger) — the back
  has no mana cost and defaults to zero, which is correct for a land.

## Deferred: the back-face land (modal-DFC play-from-hand)
The engine's DFC support is **transform-only** — a back face is reached by a transform
ability on the front (Delver, Ajani). Witch Enchanter is a **modal** DFC with no transform; its
back is an *alternate face you choose to play from hand*. The engine has no "play either face of
a modal DFC from hand" path, so **Witch-Blessed Meadow is not currently playable**. Making it
playable needs two pieces, both deferred as new mechanics:
1. **Modal-DFC play-from-hand**: `determine_legal_actions` must offer the back face (here, a land
   play) as a distinct hand action and enter the permanent as its backside characteristics.
2. **"Enters tapped unless you pay N life" replacement** (the shock-land pattern): the back's
   `R:Event$ Moved ... ReplaceWith$ DBTap` carries `UnlessCost$ PayLife<3>`. The current
   `ENTERS_TAPPED` self-replacement (`src/systems/replacement_effects.cpp`) supports only an
   unconditional / board-conditional tap (Ba Sing Se); it does not yet offer the controller a
   pay-N-life-to-enter-untapped choice. (The `DBTap` body is detected as a conditional
   ENTERS_TAPPED with an empty filter today, so it would just enter tapped.)

These were left unimplemented rather than faked: the back face is registered for card identity
but cannot be played until modal-DFC support lands.

## Behavioral decisions (CR cites)
- Front ETB (CR 603.6a): "destroy target artifact or enchantment an opponent controls" — a
  targeted `Destroy`; the comma-OR `Artifact.OppCtrl,Enchantment.OppCtrl` spec restricts targets
  to opponent-controlled artifacts/enchantments.

## Tests (`train/test_harness.py`)
- **Both faces parse / load**: the combined card loads; casting the front works and the back's
  characteristics are present on `card.backside`.
- **Front ETB**: cast Witch Enchanter `{3}{W}`, target opponent's Expedition Map → "Expedition
  Map is destroyed" (into the opponent's graveyard).
- Regression: scripted full games, `temp/witch_a` (Witch Enchanter + Grizzly Bears +
  Plains/Forest) vs `temp/urza_a` (artifact deck), seeds 1 and 3 — Witch Enchanter is cast, its
  ETB destroys the opponent's Grim Monolith, games end decisively (A wins), no draws, no
  non-fatal errors.

## Result
Front implemented and verified (ETB destroy). Both faces registered and parse. The back-face
land (Witch-Blessed Meadow) is **deferred**: it needs modal-DFC play-from-hand support plus the
pay-life "enters tapped unless" replacement, neither of which exists yet. Build clean;
regression decisive with no errors.
