---
name: implement-missing-cards
description: Implement Magic cards that decks reference but are missing from the engine, one at a time — diff decks vs card_vocab, fetch/verify the Forge script, register the card, rebuild, and test each with the test harness. Use when adding cards from scraped metagame decks (bin/resources/decks/meta) or whenever a deck names a card not in src/card_vocab.h. Stops to ask the user on any ambiguity.
---

# Implement missing cards

Adds cards the engine doesn't yet know about, **one card at a time**, verifying each
before moving on.

**The intent is always to implement the card with its full rules functionality.** Every
mechanic the card has should work correctly per the Comprehensive Rules — we do **not**
want to skip, stub, or simplify a card's mechanics. Skipping the implementation of a
mechanic is a **very rare exception**, taken only when the user explicitly approves it.

The guiding rule for stopping: **when it is unclear *how* a rule or mechanic should
behave** (ambiguous timing, modal choices, several reasonable interpretations) **— or
when the test scope is unclear — STOP and ask the user** rather than guess. Stopping is
about resolving *how the rule works*, not about getting permission to leave a mechanic
out. Default is full implementation; ask only to clarify behavior, never to find a
shortcut.

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
- The authoritative rules for how a mechanic *should* behave are checked in at
  `docs/mtg_comprehensive_rules.txt` (navigation guide: `docs/mtg_comprehensive_rules.md`).
  **When a card's behavior, timing, or keyword semantics are in question, look up the relevant
  numbered rule there** (e.g. rule 702 for keyword abilities, 704 for state-based actions, 5xx
  for combat) rather than relying on memory — grep by rule number, don't read the whole file.

## Tools

- `train/.venv/bin/python train/missing_cards.py [--json]` — the worklist: cards referenced
  by decks (default `bin/resources/decks/meta/`) but missing from the vocab, sorted by
  cross-deck frequency, each tagged `has_local_script` and a `suggested_index`.
- `python tools/forge_fetch/fetch_script.py "Card Name"` — fetch a card's Forge script into
  `cardsfolder/` (add-only; exit code = cards not found, so non-zero ⇒ hand-author needed). It is
  DFC-aware: double-faced cards live under ONE combined `<front>_<back>.txt` script, so the tool
  skips a card already present in that form and, on a front-name miss, fetches the combined file
  Forge serves under its combined name — **never hand-create a front-name `<front>.txt` for a DFC,
  that double-adds the card** (the front-name file shadows the combined one; see CLAUDE.md "Card
  Loading System").
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
      - On NOT FOUND (a card Forge doesn't have, or an accented name the tool can't normalize —
        it mirrors the engine's ASCII-stripping `name_to_uid`, so e.g. "Lórien Revealed" must be
        named by its `lorien_revealed` stem by hand), **STOP and ask the user** to confirm the
        card's intended behavior before hand-authoring a script. Do not invent behavior. (DFCs no
        longer hit this — the tool resolves their combined `<front>_<back>.txt` filename for you.)

   b. **Check feasibility against the engine.** Read the script's `A:` / `T:` / `S:` / `K:` /
      `R:` lines and their `AB$` / `SP$` / `DB$` categories. Confirm each is supported by the
      parser (`src/parse.cpp`) and the effect handlers (`src/effects/effect_kind.cpp`,
      `src/effects/effect_table.cpp`). If the card needs a mechanic the engine does not
      implement (an unsupported category, keyword, or replacement effect), **the default is
      to implement that mechanic** so the card works in full — that is the whole point of the
      task. Build a real, general handler for the missing mechanic (see the retag warning
      below). Only **STOP and ask the user** when it is genuinely unclear *how* the mechanic
      should behave, or when the mechanic is large enough that you need a decision on scope —
      and in that case present implementing it as the expected path. **Simplifying or skipping
      a mechanic is a very rare exception that requires the user's explicit approval**; never
      choose it on your own to shortcut the work. The `WARNING: Unrecognized ability param`
      lines the engine prints are acceptable for cosmetic sub-params but NOT when they change
      what the card does — when unsure, ask.

      **When you do implement a mechanic, parse the script tags as intended — never retag
      them.** Honor the script's actual `SP$`/`AB$`/`DB$` category and its
      `Origin$`/`Destination$`/`ChangeType$`/`DefinedPlayer$`/etc. by adding a real, general
      handler keyed on the tag's meaning. Do NOT rewrite one category into another or force a
      different Origin/Destination to shortcut a single card — a retag that satisfies one card
      silently corrupts every other (often unimplemented) card sharing that tag. Follow-up: it
      is acceptable to *ignore* an irrelevant tag when the card's full behavior is already
      inferable from the others (a cosmetic `StackDescription$`, or a `ChangeNum$` count-SVar
      when the effect already moves all matching cards). Ignoring a tag is fine; repurposing
      one to mean something else is not.

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
      a stacked `temp/` deck, driven with `--scripted` or, to script a precise line, the
      semantic `--play` flag (e.g.
      `--play "keep,keep,play:Mountain,pass,cast:Lightning Bolt,target:Grizzly Bears@opp"`;
      grammar in `CLAUDE.md` / `train/action_spec.py`). `--play` resolves each spec against
      the live menu by intent, so it's robust to index shifts and fails loudly with the legal
      menu when a spec doesn't match — read that menu and fix the spec. Do **not** use
      `--interactive` (it waits on terminal input you can't provide). The opening decisions
      are mulligans, so sculpted-hand lines start with `keep,keep,…`.
      Verify the narrative and decoded state match the expected outcome (e.g. the spell
      resolves, deals the right damage, moves the right zones, leaves the right board).
      - **If the expected behavior or the full test scope is not totally clear** — modal
        spells, complex/conditional triggers, unusual timing, abilities with several
        reasonable interpretations, or you are unsure what a complete test should cover —
        **STOP and ask the user** before declaring pass/fail.

   g. **Regression-test the card inside a real game.** Beyond the focused harness scenario,
      run a few scripted games with the new card actually in a deck, to confirm it doesn't
      break normal play (no non-fatal errors, no draws). Find a deck in
      `bin/resources/decks/` that already runs (its cards are all implemented) and contains —
      or can cast — the new card:
      - If such a deck already lists the card, use it directly:
        `train/.venv/bin/python train/train.py observe --deck <that-deck> --opponent <impl-deck> --games 6`.
      - Otherwise, copy a fully-implemented deck that has the right colors/mana to cast the
        new card, drop in **4 copies** of it (trimming 4 other cards to keep the count), and
        write that as a **temporary** deck under `bin/resources/decks/temp/` (or the decks
        folder). Run the scripted regression with that temp deck as one side. The point is to
        see the card drawn and played across real games, not just the sculpted scenario.
      Expect **no non-fatal errors and no draws** (per `CLAUDE.md`); only pre-existing cosmetic
      `WARNING: Unrecognized ability param` lines are acceptable. Clean up the temp deck after.
      **Treat any draw as a bug — every draw is an engine failure, not a pass.** A drawn game
      means a game that can't resolve (a loop, a stuck stack, an unkillable board) and is a real
      defect to root-cause and fix, exactly like a non-fatal error; never wave it through. The
      **only** exception is a draw the **user has explicitly identified as acceptable** — if a
      regression draws and you're unsure why, STOP and surface it to the user rather than
      passing the card.

   h. **Record the result** (card name, index, pass/fail, notes) and move to the next card.

3. **When a deck's cards are all implemented**, sanity-check it end-to-end:
   `train/.venv/bin/python train/train.py diag --deck <meta-deck> --opponent <meta-deck>` —
   expect no draws and no non-fatal errors (per `CLAUDE.md`).

## Stop-and-ask gates (summary)

Default to **implementing the card in full**. Stop and ask the user only to resolve
*how* something should work — never to get permission to leave a mechanic out:

- The card has **no Forge script** and must be hand-authored.
- The card's **expected behavior is ambiguous** — unclear timing, modal choices, or
  several reasonable interpretations of how a rule should resolve.
- The **test scope is unclear** — you can't confidently say what a complete test covers.
- A needed **mechanic is large enough that scope needs a decision** — present implementing
  it as the expected path; simplifying or skipping it requires the user's explicit approval.
- Adding the card would **overflow the vocab** (`N_CARD_TYPES`).

A mechanic the engine doesn't yet support is **not** a reason to skip — the default is to
build a real handler for it. Total clarity or ask, but ask to clarify behavior, not to
shortcut implementation. Never guess a card's behavior, and never silently drop a mechanic.
