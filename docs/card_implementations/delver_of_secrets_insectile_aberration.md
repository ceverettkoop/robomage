# Delver of Secrets // Insectile Aberration  (vocab index 128)

## Oracle text
**Delver of Secrets** ({U}, 1/1, Creature — Human Wizard) — front face:
> At the beginning of your upkeep, look at the top card of your library. You may reveal that
> card. If an instant or sorcery card is revealed this way, transform Delver of Secrets.

**Insectile Aberration** (3/2, Creature — Human Insect, blue) — back face:
> Flying

## Forge script
- Source: pre-existing local script — `bin/resources/cardsfolder/d/delver_of_secrets_insectile_aberration.txt`
- Transforming double-faced card (`AlternateMode:DoubleFaced`), front + `ALTERNATE` back face.
- Key tags (front face):
  - `T:Mode$ Phase | Phase$ Upkeep | ValidPlayer$ You | TriggerZones$ Battlefield |
    Execute$ TrigPeek` — beginning-of-your-upkeep trigger (CR 603.2 / 508-style phase trigger).
  - `SVar:TrigPeek:DB$ PeekAndReveal | PeekAmount$ 1 | RevealOptional$ True |
    RememberRevealed$ True | AILogic$ InstantOrSorcery | SubAbility$ DBTransform` — look at the
    top card, optionally reveal it (CR 701.18 "look at"; the reveal is the player's choice).
  - `SVar:DBTransform:DB$ SetState | Defined$ Self | Mode$ Transform |
    ConditionDefined$ Remembered | ConditionPresent$ Card.Instant,Card.Sorcery |
    ConditionCompare$ EQ1` — transform iff the revealed card is an instant or sorcery
    (CR 701.28 transform; CR 711 double-faced cards).
  - `SVar:DBCleanup:DB$ Cleanup | ClearRemembered$ True` — bookkeeping.
- Key tags (back face): `PT:3/2`, `K:Flying` (CR 702.9 flying), `Colors:blue`.

No tags were retagged or repurposed. The PeekAndReveal effect honors the `PeekAndReveal`
category's intended meaning (look-at-top + optional reveal + conditional transform on the
revealed card's type); the cosmetic `RevealOptional$`/`Mode$ Transform`/`RememberRevealed$`/
`AILogic$` params are emitted as the harmless pre-existing `WARNING: Unrecognized ability param`
lines because their behavior is already fully realized by the effect.

## Engine work
**No new engine code was required** — the full mechanic was already implemented by prior DFC and
peek work, and is exercised by the stock `delver` deck. The card needed only ML-vocabulary
registration so its combined DFC name has an embedding index.

The pre-existing infrastructure that realizes the card, for the record:
- **DFC parse split** (`src/parse.cpp`): the `\nALTERNATE` marker splits the script into front
  and back faces; each face is parsed by `parse_card_face` into a complete `CardData`, with the
  back stored as `CardData::backside`. So the transformed permanent has its own name, types,
  3/2 P/T, and Flying — not just a P/T swap (CR 711.2).
- **Combined-name loading** (`src/card_db.cpp`): the card is loaded by the combined uid
  `delver_of_secrets_insectile_aberration`; aliases for the front (`Delver of Secrets`) and back
  (`Insectile Aberration`) names are registered so either resolves to the same entity. Deck files
  reference the combined name (`4 Delver of Secrets Insectile Aberration`).
- **Peek / optional-reveal / conditional-transform** (`src/effects/effect_peek_and_reveal.cpp`):
  the Delver branch finds the top card of the controller's library, logs it privately, offers a
  binary "Reveal" / "Don't reveal" choice, and — only if the player reveals and the card is an
  Instant or Sorcery — calls `set_permanent_face(source, /*show_back=*/true)`. If the card is
  already transformed it does nothing (one-way flip).
- **Transform subsystem** (`src/transform.{h,cpp}`): `set_permanent_face` rebuilds the
  permanent's name, types, Creature/Damage components (base 3/2), and the active face's keywords
  (Flying) from the back face. The same path serves Delver's creature→creature flip and Ajani's
  creature→planeswalker flip (CR 701.28).

## Behavioral decisions (made in lieu of asking a human)
- **The reveal is optional and the player controls it** (`RevealOptional$ True`). Declining to
  reveal never transforms, even when the top card is an instant/sorcery (CR follows the printed
  "You may reveal").
- **Transform condition is type = Instant OR Sorcery**, checked on the revealed card's printed
  card types; any other type (creature, land, etc.) leaves Delver a 1/1 front face.
- **One-way flip**: once transformed, the upkeep trigger no longer flips it back (the back face
  has no upkeep trigger; the effect also guards on `!perm.transformed`).
- **Looking at the top card is private information** to Delver's controller (logged via
  `game_log_private`); the reveal makes it public.

## Tests
Isolation (`train/test_harness.py`), Delver pre-placed on the battlefield, library top controlled
via `--library-a` with `--no-shuffle`:
- **Instant on top + reveal → transforms.** Lightning Bolt on top, choose "Reveal" →
  `Delver of Secrets transforms into Insectile Aberration!`; board shows
  `Insectile Aberration [3/2]`. PASS.
- **Creature on top + reveal → no transform.** Grizzly Bears on top, choose "Reveal" → no
  transform; board stays `Delver of Secrets [1/1]`. PASS (negative case).
- **Instant on top + decline reveal → no transform.** Lightning Bolt on top, choose
  "Don't reveal" → no transform; board stays `Delver of Secrets [1/1]`. PASS.
- **Flying after transform.** Transformed Insectile Aberration [3/2] attacks into a ground
  Grizzly Bears → `No creatures eligible to block` (flying evasion working). PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–6): stock `delver` vs `delver` (the deck
runs 4 copies of the card). All 6 games finished decisively (no draws), the engine stayed stable,
and the upkeep transform fired in real games (`Delver of Secrets transforms into Insectile
Aberration!` in seeds 1, 4, 6). The only error lines were the pre-existing, Delver-unrelated
`Could not open token script: w_1_1_monk_prowess.txt` from Cori-Steel Cutter, which also appears
in a stock delver-vs-delver run on a clean tree at HEAD (verified by stash). No temp decks were
created.

## Result
implemented
