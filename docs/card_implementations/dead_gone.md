# Dead // Gone

## Oracle text

Split card (CR 709) — two instant halves on one card, each castable from hand:

- **Dead** — Instant, {R}: "Dead deals 2 damage to target creature."
- **Gone** — Instant, {2}{R}: "Return target creature you don't control to its owner's hand."

## Forge script

Source: pre-existing local (`bin/resources/cardsfolder/d/dead_gone.txt`). Key tags:

- Front half `Name:Dead`, `A:SP$ DealDamage | ValidTgts$ Creature | NumDmg$ 2`.
- `AlternateMode:Split` — marks this as a split card.
- `ALTERNATE` block, back half `Name:Gone`, `A:SP$ ChangeZone | ValidTgts$ Creature.YouDontCtrl
  | Origin$ Battlefield | Destination$ Hand`.

The parser already splits the script at `\nALTERNATE` and stores the back half in
`CardData::backside` (like an MDFC). Both `DealDamage` and `ChangeZone` (Battlefield→Hand) are
existing effect handlers.

## Engine work

**Mechanics added (general): split-card** — a split card (CR 709): both halves are cast from
hand at their own cost; neither half is a permanent, and the whole card goes to the graveyard on
resolution.

- `src/components/carddata.h` — `CardData::is_split` flag. Kept distinct from `is_modal_dfc` so
  split cards never pick up MDFC permanent-entering logic (there is none to apply — both halves
  are spells).
- `src/parse.cpp` `parse_card_face` — `is_split = (AlternateMode == "Split")` (mirrors the
  `is_modal_dfc = (AlternateMode == "Modal")` line; `AlternateMode:Split` previously set neither).
- `src/systems/state_manager_actions.cpp` `offer_modal_back_face_casts` — the back-half offer
  gate now fires for `is_modal_dfc` OR `is_split`. Both faces reuse the existing
  `cast_back_face` path (front half offered by the normal hand-cast loop; back half as a
  `CAST_SPELL` with `cast_back_face = true`). No other `is_modal_dfc` site was touched, so the
  MDFC-only permanent/land-face logic is not shared with split cards.

**Card-loading generalization (CR 709 combined names)** — `src/parse.cpp` `name_to_uid` now
treats `/` as a separator (→`_`), like space and dash. A combined "Front/Back" deck reference
(`1 Dead/Gone`) therefore normalizes to Forge's underscore-joined filename `dead_gone` and
resolves to `cardsfolder/d/dead_gone.txt`. No existing card name / deck reference contains `/`,
so the change is a no-op for every current input (replay corpus unaffected).

## Behavioral decisions

### Vocab name (the interesting part)

Dead // Gone is **one** physical card → **one** vocab index, **328**. But two different name
normalizations must both land on 328, and they disagree:

- **Deck-identity serialization** (`src/classes/deck_state.cpp`) matches by `name_to_uid`
  (lowercases, separators→`_`). A `1 Dead/Gone` decklist → uid `dead_gone`.
- **In-game observation** (`src/machine_io.cpp` → `card_name_to_index`) matches by
  `ascii_fold_card_name` (case/punctuation-preserving). The loaded front name is `Dead`; a
  back-half cast action reports `Gone`.

No single string satisfies both schemes for both the decklist token and the in-game faces, so
index 328 is **aliased under three names**, all → 328:

```
{"Dead", 328}, {"Gone", 328}, {"Dead/Gone", 328}
```

- `Dead/Gone` → `name_to_uid` `dead_gone` → the decklist token serializes, and (with the `/`
  separator change) the *file* `dead_gone.txt` loads.
- `Dead` / `Gone` → `ascii_fold` `Dead` / `Gone` → the in-game front/back faces each encode 328.

`Dead/Gone` is listed **last** so the cost-matrix codegen (last-write-wins per index) leaves an
honest **zero** row for 328: `gen_card_costs.find_card_file` prefix-matches `Dead`→
`dead_before_sunrise.txt` and `Gone`→`gone_missing.txt` (unrelated cards, wrong cost), whereas
`Dead/Gone` finds no file and yields the standard zero/unresolvable sentinel row (same as DFC
backs). It also makes `card_index_to_name(328)` = `"Dead/Gone"`, the accurate split-card display
name.

Consequence (minor): because the harness/`action_spec` decodes both half-actions' card as
`Dead/Gone`, a `cast:Gone` spec is ambiguous *by card name* and must be disambiguated with
`desc:Cast Gone`. This is a test-harness spec-matching artifact only — the engine distinguishes
the two halves by action index/description, and the ML sees both as card_id 328 (correct: same
card).

### Split vs MDFC

Only the back-face **offer** gate treats `is_split` like `is_modal_dfc`. Everywhere else
`is_modal_dfc` is unchanged, so split cards get none of the MDFC permanent-/land-face machinery
(they never enter as permanents; the spell resolves and the card goes to the graveyard).

## Tests

Isolation (`train/test_harness.py`, seed 1):

- Dead//Gone in hand with ≥3 Mountains: **both** "Cast Dead [#0]" and "Cast Gone [#3]" offered.
- Cast Dead → 2 damage to Grizzly Bears → 2/2 destroyed; card to A's graveyard. PASS.
- Cast Gone → Grizzly Bears returned to owner's hand (opp hand 7→8, revealed); card to A's
  graveyard. PASS.
- Deck reference `1 Dead/Gone` (temp deck): loads with **no fatal** (previously fatal on both
  the vocab-index lookup and the missing-file open); both halves offered and cast correctly from
  the deck-loaded card. PASS.

CI gate: `ci_check.py --tier pygen,vocab,smoke` (run once after both cards).

## Result

implemented
