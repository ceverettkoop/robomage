# Elvish Reclaimer  (vocab index 220)

## Oracle text
Elvish Reclaimer gets +2/+2 as long as there are three or more land cards in your graveyard.

{2}, {T}, Sacrifice a land: Search your library for a land card, put it onto the battlefield
tapped, then shuffle.

## Forge script
- Source: present in tree — `bin/resources/cardsfolder/e/elvish_reclaimer.txt`
- Type: `Creature Elf Warrior`, P/T `1/2`, mana cost `G`.
- Key tags:
  - `S:Mode$ Continuous | Affected$ Card.Self | AddPower$ 2 | AddToughness$ 2 |
    IsPresent$ Land.YouOwn | PresentZone$ Graveyard | PresentCompare$ GE3` — the continuous
    +2/+2 static, gated on a COUNT of land cards in the controller's graveyard.
  - `A:AB$ ChangeZone | Cost$ 2 T Sac<1/Land> | Origin$ Library | Destination$ Battlefield |
    ChangeType$ Land | ChangeNum$ 1 | Tapped$ True` — the activated fetch.

No tags were retagged or repurposed; every mechanic below is keyed on the tag's intended meaning.

## Engine work
The activated fetch needed no new code — `{2} {T} Sac<1/Land>` cost, library search for a
`Land`, enters `Tapped$ True`, then shuffle, are all already-supported `ChangeZone` tags
(verified working: pays {2}, taps Reclaimer, sacrifices a land, searches, the fetched land
enters tapped, library is shuffled).

The gap was the **static +2/+2 gated on a COUNT of cards in a zone**. `parse_static_abilities`
understood `Condition$` (Delirium / PlayerTurn) and `CheckSVar$`/`SVarCompare$` SVar
conditions, but it did not parse `IsPresent$ / PresentZone$ / PresentCompare$` for *static*
abilities (those keys were parsed only for triggers / sub-abilities), and `StaticAbility` had
no present-count condition fields. A general present-count gate was added (CR 604.3 — a
continuous static functions only while its conditions hold; CR 611 continuous effects):

1. **`StaticAbility` fields** (`src/components/static_ability.h`): `present_filter`,
   `present_zone`, `present_compare`. Empty `present_filter` means "no present gate".

2. **Parser** (`src/parse.cpp`, `parse_static_abilities`): parse `IsPresent$` →
   `present_filter`, `PresentZone$` → `present_zone`, `PresentCompare$` → `present_compare`.

3. **Evaluation** (`src/systems/state_manager_statics.cpp`):
   - New static helper `static_present_condition_met(filter, zone, compare, controller,
     entities)` — counts cards/permanents matching `filter` in `zone` and compares against
     `compare` (default zone `Battlefield` per Forge convention when `PresentZone$` is omitted;
     default compare `GE1`). It is general over **any zone** (Battlefield/Graveyard/Hand/
     Exile/Library/Stack), **any compare op** (GE/LE/EQ/GT/LT/NE via the shared `compare_svar`),
     and **any filter**. Ownership qualifiers (`YouOwn`/`YouCtrl`/`OppOwn`/`OppCtrl`) are split
     off and checked against the containing `Zone::owner` (a card off the battlefield has no
     controller, so `.YouOwn` is scoped via `Zone::owner` here and stripped before the
     type/characteristic match — `card_view`/`card_matches_filter` has no `.YouOwn` token and
     would otherwise fail-closed), mirroring `evaluate_present_condition`'s owner handling. The
     remaining filter is matched through `permanent_matches_filter` (battlefield) or
     `card_matches_filter` (other zones).
   - `gather_active_statics` ANDs the present gate into `ActiveStatic::condition_met` after the
     existing `Condition$`/`CheckSVar$` chain, so the gate composes with any other condition and
     is re-evaluated every SBA pass (turns on/off live as the graveyard count changes). A static
     that ONLY has a present gate (Elvish Reclaimer) is provisionally `true` before the AND so the
     trailing "unrecognised condition → unmet" branch does not zero it.

The +2/+2 itself flows through the existing layer-7c additive P/T applier (`Affected$ Card.Self`
is the single-target form; `add_power`/`add_toughness` are applied when `condition_met`).

## Behavioral decisions (made in lieu of asking a human)
- **Sorcery-speed restriction not enforced** — the script as checked in does NOT carry a
  `SorcerySpeed$ True` tag (only the Oracle reminder text says "Activate only as a sorcery").
  Per the project rule to honor the script's actual tags and never hand-author / retag a script,
  the ability is left at the timing the script declares (instant-speed `ChangeZone`). This is a
  script-content omission, not an engine gap; the general present-count gate is the implemented
  mechanic. (CR 605 / "as a sorcery" timing would be enabled simply by the script gaining the
  tag, which the engine already supports via `sorcery_speed_only`.)
- **`.YouOwn` is scoped to the graveyard's owner**, not a controller check, because graveyard
  cards have no controller. Lands in the OPPONENT's graveyard never count toward the gate.
- **Continuous re-evaluation** — the gate is recomputed every state-based-effects pass, so the
  buff turns on the instant the count reaches 3 and back off the instant it drops below 3.

## Tests
Isolation (`train/test_harness.py`, `--play`, seed 1). The graveyard was staged in one turn by
cracking three fetchlands (each `Wooded Foothills` sacrifices itself — a land — to the graveyard
and fetches a basic), so the graveyard land count climbs 0→1→2→3 deterministically:
- **Below threshold stays 1/2.** With 0, 1, and 2 land cards in graveyard, Elvish Reclaimer is
  `[1/2]`. PASS.
- **At threshold becomes 3/4.** When the third cracked fetchland lands in the graveyard the log
  shows `Elvish Reclaimer gains 2/2 (always)` and the board shows `Elvish Reclaimer [3/4]`; it
  attacked as a 3/4. PASS.
- **Turns back off below threshold.** Casting `Life from the Loam` to return a land card from the
  graveyard to hand (graveyard drops to 2 land cards) logs `Elvish Reclaimer loses static bonus`
  and the board reverts to `Elvish Reclaimer [1/2]`. PASS — proves live on/off re-evaluation.
- **Activated fetch.** `{2} {T} Sacrifice a land` → the ability goes on the stack, a library land
  search is offered, the chosen land `enters tapped`, and the library is shuffled
  (`Player A shuffles their library`). PASS.

Regression (`train/test_harness.py --scripted`, seeds 1–3): deck `temp/recl_mav` (the green
maverick list with 4 Elvish Reclaimer swapped in for Scythecat Cub) vs `delver`. All three games
finished decisively (B/A/B), no draws, no fatal/non-fatal errors, no asserts/tracebacks. Existing
conditional statics still evaluate: `delver` vs `mav` seeds 1–2 finished decisively with Knight of
the Reliquary's SVar static still applying (`Knight of the Reliquary gains (always)`). (Only the
pre-existing cosmetic `WARNING: Unrecognized ability param` lines from unrelated cards appeared.)
Temp decks cleaned up.

## Result
implemented
