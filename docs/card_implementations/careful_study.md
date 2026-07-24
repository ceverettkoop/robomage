# Careful Study

## Oracle text
Draw two cards, then discard two cards.

(`{U}` sorcery)

## Forge script
- **Source:** pre-existing local (`bin/resources/cardsfolder/c/careful_study.txt`).
- **Key tags:**
  - `A:SP$ Draw | NumCards$ 2 | SubAbility$ DBDiscard`
  - `SVar:DBDiscard:DB$ Discard | Defined$ You | NumCards$ 2 | Mode$ TgtChoose`

## Engine work
Fix key (general): **discard-choose-n**.

The player-choice discard path in `src/effects/effect_discard.cpp` picked exactly ONE card and
ignored `NumCards$` — only the `Random` (Hymn to Tourach) and `RevealDiscardAll` (Cabal Therapy)
modes honored a count. Extended the `RevealYouChoose` / `TgtChoose` choice path into a loop that
runs `NumCards$` times, each pick a separate query (so it round-trips in machine mode):

- Loop count `= ab.amount > 0 ? ab.amount : 1`, so `NumCards$ 0`/unset still means one card (the
  Thoughtseize/Archon single-discard case) — unchanged for every existing card.
- The `DiscardValid$` pool is rebuilt from the **live** hand each pick (a picked card left the
  hand), so a card can't be re-picked; loop progress persists in a new `DiscardRt` runtime
  (`src/resolution_frame.h`) so a machine-mode suspension resumes at the next unmade pick.
- Guards `N > hand size`: the loop breaks when no matching card remains (discard as many as
  possible).

CR 701.8b (discard). The multi-pick loop mirrors the existing `run_discard_unless` machinery
(Reality Smasher's "unless they discard a card").

Mechanics added (general): **choose-N-to-discard loop** — any player-choice discard now honors
`NumCards$`, not just the single-card case.

## Behavioral decisions
- Reuses the existing single-choice discard machinery (menu build, filter, `ctx.ask`) inside the
  loop; no new discard primitive.
- `DiscardRt` holds only a counter (the per-pick menu is a live hand scan), so the determinization
  pin visitor needs no entry for it.
- The Draw sub-ability runs first (drawing two), then the Discard sub-ability — the natural
  sub-ability chain order, matching "draw two, then discard two".

## Tests (isolation)
- `--hand-a "Careful Study,..."`, cast it: Player A **draws 2** (two Islands), then makes **two
  separate DISCARD choices** (Mountain, then Forest). Graveyard gains exactly 2 cards (plus the
  spell itself); hand net unchanged (5 → cast → +2 draw → −2 discard). PASS.
- Regression — Thoughtseize (RevealYouChoose, count 1): discards **exactly 1** card (Grizzly
  Bears), unchanged. PASS.
- CI gate: `ci_check.py --tier pygen,vocab,smoke` — see batch report.

## Result
Implemented.
