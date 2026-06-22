---
name: implement-missing-cards
description: Implement Magic cards that decks reference but are missing from the engine, one at a time — diff decks vs card_vocab, fetch/verify the Forge script, register the card, rebuild, and test each with the test harness. Use when adding cards from scraped metagame decks (bin/resources/decks/meta) or whenever a deck names a card not in src/card_vocab.h. Stops to ask the user on any ambiguity.
---

# Implement missing cards

Adds cards the engine doesn't yet know about, **one card at a time**, verifying each
before moving on. The guiding rule: **when anything is less than totally clear about a
card's expected behavior or the test scope, STOP and ask the user** rather than guess.

## Background

- Cards are registered in `src/card_vocab.h` as `{"Card Name", N}` (next free index `N`).
  The embedding refactor raised the cap to `N_CARD_TYPES = 1024`, so indices up to 1022 are
  free (1023 is the token sentinel). A card absent from this file is "unimplemented".
- Card behavior comes from a Forge-format script in
  `bin/resources/cardsfolder/<letter>/<name>.txt`. Many missing cards already have a local
  script; others can be downloaded from the upstream Forge repo.
- **Never edit an existing card script** (project rule: DO NOT MODIFY CARD SCRIPTS). The
  downloader only *adds* new files. Hand-author a script only when none exists, after
  confirming behavior with the user.

## Tools

- `train/.venv/bin/python train/missing_cards.py [--json]` — the worklist: cards referenced
  by decks (default `bin/resources/decks/meta/`) but missing from the vocab, sorted by
  cross-deck frequency, each tagged `has_local_script` and a `suggested_index`.
- `python tools/forge_fetch/fetch_script.py "Card Name"` — fetch a card's Forge script into
  `cardsfolder/` (add-only; exit code = cards not found, so non-zero ⇒ hand-author needed).
- `train/.venv/bin/python train/gen_card_costs.py` — regenerate `train/card_costs.py` after
  editing the vocab (required).
- `train/test_harness.py` — exercise a card's behavior (see `CLAUDE.md` for full usage).

## Procedure

1. **Build the worklist.** Run `train/missing_cards.py --json`. Work the cards in the order
   given (most-played first). Do them **one at a time** — never batch.

2. For each card:

   a. **Get a script.** If `has_local_script` is false, run
      `tools/forge_fetch/fetch_script.py "<name>"`.
      - On success, read the fetched script.
      - On NOT FOUND (e.g. double-faced/adventure cards whose Forge filename differs, or
        a card Forge doesn't have), **STOP and ask the user** to confirm the card's intended
        behavior before hand-authoring a script. Do not invent behavior.

   b. **Check feasibility against the engine.** Read the script's `A:` / `T:` / `S:` / `K:` /
      `R:` lines and their `AB$` / `SP$` / `DB$` categories. Confirm each is supported by the
      parser (`src/parse.cpp`) and the effect handlers (`src/effects/effect_kind.cpp`,
      `src/effects/effect_table.cpp`). If the card needs a mechanic the engine does not
      implement (an unsupported category, keyword, or replacement effect), **STOP and ask the
      user** how to proceed: implement the mechanic, simplify the card, or skip it. The
      `WARNING: Unrecognized ability param` lines the engine prints are acceptable for
      cosmetic sub-params but NOT when they change what the card does — when unsure, ask.

   c. **Register the card.** Append `{"<Name>", N}` to `src/card_vocab.h` using the next free
      index. Indices must stay `< 1023` (the token sentinel). If a card name has an apostrophe,
      keep it (e.g. `Mishra's Bauble`) — the engine normalizes it away when resolving the
      script. If adding the card would exceed the embedding vocab, **STOP and ask the user**
      (growing `N_CARD_TYPES` changes the observation size).

   d. **Regenerate costs.** Run `train/gen_card_costs.py`.

   e. **Build.** Run plain `make`. A clean build is required; non-fatal errors are not
      acceptable (per `CLAUDE.md`). Fix any new build error before continuing.

   f. **Test with the harness.** Construct a focused scenario that exercises *this* card's
      behavior — use inline `--hand-a/--library-a/--battlefield-a` (and the opponent side) or
      a stacked `temp/` deck, driven with `--scripted` or an explicit `--actions` sequence.
      Verify the narrative and decoded state match the expected outcome (e.g. the spell
      resolves, deals the right damage, moves the right zones, leaves the right board).
      - **If the expected behavior or the full test scope is not totally clear** — modal
        spells, complex/conditional triggers, unusual timing, abilities with several
        reasonable interpretations, or you are unsure what a complete test should cover —
        **STOP and ask the user** before declaring pass/fail.

   g. **Record the result** (card name, index, pass/fail, notes) and move to the next card.

3. **When a deck's cards are all implemented**, sanity-check it end-to-end:
   `train/.venv/bin/python train/train.py diag --deck <meta-deck> --opponent <meta-deck>` —
   expect no draws and no non-fatal errors (per `CLAUDE.md`).

## Stop-and-ask gates (summary)

Default to asking the user whenever you hit any of these:

- The card has **no Forge script** and must be hand-authored.
- The script needs a **mechanic the engine doesn't support**.
- The card's **expected behavior is ambiguous**.
- The **test scope is unclear** — you can't confidently say what a complete test covers.
- Adding the card would **overflow the vocab** (`N_CARD_TYPES`).

Total clarity or ask. Never guess a card's behavior.
