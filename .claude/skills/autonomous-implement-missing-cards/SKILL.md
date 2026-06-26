---
name: autonomous-implement-missing-cards
description: Autonomously implement many missing Magic cards in sequence inside a sandbox cloud container — one card at a time, each fully implemented, tested in isolation and in a real game, documented in a committed design doc, and committed on its own. No user input is available, so behavior is resolved from the Comprehensive Rules and Oracle text (and documented) or the card is deferred with a written rationale; the working tree is never left broken or half-implemented. Context is cleared after every commit by dispatching a fresh subagent per card. Use for unattended batch card implementation (e.g. draining the bin/resources/decks/meta worklist) when no human is in the loop.
---

# Autonomous implement missing cards

Drains the missing-card worklist **unattended**, **one card at a time**, in a sandbox
cloud container. Each card is fully implemented, tested in isolation **and** in a real
game, documented in a committed design doc, and **committed on its own**. After each commit
the per-card working context is discarded so the next card starts fresh.

This is the autonomous sibling of `implement-missing-cards`. The engineering procedure for a
single card is the same; the differences are operational:

- **No user is available.** There is no one to ask. Every gate that the interactive skill
  resolves by "STOP and ask the user" is resolved here by the **Autonomous decision policy**
  below — either decide-and-document, or defer-and-document. Never block waiting for input.
- **Each card gets its own commit** plus a committed design doc under
  `docs/card_implementations/`.
- **Context is cleared after each commit** by running each card in a **fresh subagent**
  (see Orchestration). The orchestrator holds only a tiny ledger between cards.
- **The tree is never left broken.** A card is committed only when it builds clean and all
  its tests pass. Anything else → revert to a clean tree and defer.

**The intent is always to implement the card with its full rules functionality** per the
Comprehensive Rules. We do **not** stub, skip, or simplify a card's mechanics to save effort.
The only acceptable non-implementation outcome is a **clean deferral** (card untouched in the
tree, reason documented) — used when the card genuinely cannot be implemented and verified
autonomously with confidence.

## Background

- Cards are registered in `src/card_vocab.h` as `{"Card Name", N}` (next free index `N`).
  `N_CARD_TYPES = 1024`, so indices up to 1022 are free (1023 is the token sentinel). A card
  absent from this file is "unimplemented".
- Card behavior comes from a Forge-format script in
  `bin/resources/cardsfolder/<letter>/<name>.txt`. Many missing cards already have a local
  script; others download from the upstream Forge repo.
- **Never edit an existing card script** (project rule: DO NOT MODIFY CARD SCRIPTS). The
  downloader only *adds* new files.
- **NEVER hand-author a card script.** A card's script must come either from a pre-existing
  local file or from the upstream Forge repo via the downloader. If `fetch_script.py` cannot
  find the card (non-zero exit) and no local script exists, **abandon the card immediately and
  move on** — defer it (clean tree, reason "no Forge script available"). Do not write, invent,
  or reconstruct a script by hand under any circumstances.
- The authoritative rules for how a mechanic *should* behave are checked in at
  `docs/mtg_comprehensive_rules.txt` (navigation guide: `docs/mtg_comprehensive_rules.md`).
  **When behavior, timing, or keyword semantics are in question, grep the relevant numbered
  rule** (702 keyword abilities, 704 state-based actions, 5xx combat, 613 layers) rather than
  relying on memory. This is the ground truth that replaces asking the user.

## Tools

- `train/.venv/bin/python train/missing_cards.py [--json]` — the worklist: cards referenced
  by decks (default `bin/resources/decks/meta/`) but missing from the vocab, sorted by
  cross-deck frequency, each tagged `has_local_script` and a `suggested_index`.
- `python tools/forge_fetch/fetch_script.py "Card Name"` — fetch a card's Forge script into
  `cardsfolder/` (add-only; non-zero exit ⇒ not found ⇒ **defer the card and move on; never
  hand-author**).
- `train/.venv/bin/python train/gen_card_costs.py` — regenerate `train/card_costs.py` after
  editing the vocab (required).
- `train/test_harness.py` — exercise a card's behavior (full usage in `CLAUDE.md`).
- `train/.venv/bin/python train/train.py observe --deck <a> --opponent <b> --games N` —
  scripted in-game regression (replaces the old `diag`/`watch`).

## Orchestration (how context is cleared per card)

The skill runs as a small **orchestrator loop**. The orchestrator does almost no engineering
itself; it dispatches each card to a **fresh subagent** so that card's heavy context (script
reads, build output, multi-screen test transcripts) lives and dies inside the subagent and is
**discarded on return** — this is the mechanism that "clears context after each commit".

1. **Preflight (once).** Confirm a clean tree (`git status --short` empty; if not, stash or
   commit pre-existing work first so per-card commits stay isolated). Confirm the build is
   green with a plain `make`. Confirm the branch is the designated development branch. Create
   `docs/card_implementations/` if absent.

2. **Build the worklist (once).** Run `train/missing_cards.py --json`. Persist the ordered
   list (most-played first) plus a status column to a ledger the orchestrator keeps **outside**
   per-card context — write it to the scratchpad dir (e.g.
   `…/scratchpad/card_worklist.json`) so it survives the orchestrator's own context
   summarization. Status ∈ `pending | done | deferred`.

3. **Loop.** For each `pending` card, in worklist order:
   a. **Dispatch one subagent** (Agent tool, `general-purpose`) with the **Per-card subagent
      prompt** below, filled in with the card name and its `suggested_index`. The subagent does
      the entire single-card procedure end-to-end **including the commit**, then returns a
      one-paragraph structured summary (outcome, index used, files touched, commit hash, or
      defer-reason).
   b. **Record** the returned outcome in the ledger (`done` or `deferred`, with the reason).
      Do **not** carry the subagent's transcript forward — only its summary line.
   c. **Move to the next card.** The subagent's context is gone; the next card starts fresh.

4. **Stop conditions.** Continue until the worklist is exhausted, or a caller-supplied card
   budget/time budget is reached. At the end, emit a final tally (implemented / deferred, with
   deferral reasons) so a human can review deferrals later.

The orchestrator must **never** itself open card scripts, build logs, or transcripts beyond
what it needs to dispatch and tally — that defeats the context-clearing design. If for some
reason subagents are unavailable, fall back to doing cards linearly but **explicitly compact /
clear context after each commit**; the per-card procedure and the autonomous policy are
identical.

## Per-card subagent prompt (template the orchestrator dispatches)

Give the subagent this task verbatim, substituting `<NAME>` and `<INDEX>`. The subagent has no
memory of other cards and must finish with a clean tree no matter what.

> Implement the single Magic card **`<NAME>`** into the Robomage engine, fully and correctly,
> then commit it on its own. You are running autonomously — **there is no user to ask.** Follow
> this exactly and finish with a **clean git tree** (committed on success, reverted on defer).
>
> 1. **Get a script.** If no local `cardsfolder/` script exists, run
>    `python tools/forge_fetch/fetch_script.py "<NAME>"`. If NOT FOUND (non-zero exit),
>    **abandon this card immediately and defer it** (see Defer protocol) with reason "no Forge
>    script available". **NEVER hand-author or reconstruct a script by hand** — a script must
>    come only from a pre-existing local file or the Forge repo. Never modify an existing script.
> 2. **Check feasibility.** Read the `A:`/`T:`/`S:`/`K:`/`R:` lines and their
>    `AB$`/`SP$`/`DB$` categories. Confirm each is supported by `src/parse.cpp`,
>    `src/effects/effect_kind.cpp`, and `src/effects/effect_table.cpp`. If a needed mechanic is
>    missing, **the default is to implement it** as a real, general handler keyed on the tag's
>    intended meaning — **never retag** one category/Origin/Destination into another to shortcut
>    this one card (that silently corrupts every other card sharing the tag). Ignoring a purely
>    cosmetic tag is fine; repurposing a tag is not. Look up exact rule behavior in
>    `docs/mtg_comprehensive_rules.txt` by rule number.
> 3. **Register.** Append `{"<NAME>", <INDEX>}` to `src/card_vocab.h` (keep apostrophes; index
>    must stay `< 1023`). Then run `train/.venv/bin/python train/gen_card_costs.py`.
> 4. **Build.** Run plain `make`. The build must be clean — fix any new error. Non-fatal errors
>    are not acceptable.
> 5. **Test in isolation.** With `train/test_harness.py`, construct a focused scenario that
>    exercises *this* card's behavior (inline `--hand-a/--library-a/--battlefield-a/...` or a
>    `temp/` stacked deck, driven by the semantic `--play` specs; never `--interactive`).
>    Verify the narrative and decoded state match the expected outcome (resolves, deals correct
>    damage, moves correct zones, leaves the correct board). Cover the card's real modes /
>    triggers, not just the happy path.
> 6. **Regression in a real game.** Run a few scripted games with the card actually in a deck
>    via `train/train.py observe … --games 6` (use an existing fully-implemented deck that lists
>    or can cast it, else copy one into `bin/resources/decks/temp/` with 4 copies swapped in).
>    Expect **no non-fatal errors and no draws**; only pre-existing cosmetic
>    `WARNING: Unrecognized ability param` lines are acceptable. Clean up any temp deck.
> 7. **Document.** Write `docs/card_implementations/<uid>.md` (uid = lowercased name, spaces→`_`,
>    apostrophes dropped) using the **Design-doc template** below — record the Oracle text, the
>    Forge tags, every engine change and *why*, every behavioral decision you made in lieu of
>    asking a human (with the rule citation that justifies it), the exact test scenarios and
>    their observed results, and the regression result.
> 8. **Commit.** Stage exactly: the new/edited engine source, `src/card_vocab.h`,
>    `train/card_costs.py`, the new `cardsfolder/` script (if freshly fetched), and the new
>    design doc. Commit with message
>    `Implement <NAME>` + a short body summarizing mechanics added and tests run, plus the
>    required Co-Authored-By / Claude-Session trailers. Do **not** push (the orchestrator or a
>    final step handles pushing). Report the commit hash.
> 9. **Defer protocol (if you cannot finish with confidence).** If the card is genuinely
>    ambiguous, needs a mechanic too large to implement safely here, has **no Forge script
>    available** (fetch failed and no local script — never hand-author one), or would overflow
>    the vocab — do **not** commit a partial or guessed implementation. Run `git checkout -- . && git clean -fd` to **fully restore a clean tree**
>    (remove your vocab edit, regenerated costs, any fetched script and partial source), then
>    return a deferral summary stating the card name and the specific reason. A wrong guess is
>    worse than a deferral.
>
> Return a single structured summary: `OUTCOME: done|deferred`, `INDEX`, `COMMIT`, `FILES`,
> `TESTS` (one line each scenario + result), and `NOTES/REASON`.

## Autonomous decision policy (replaces "ask the user")

Because no human is in the loop, each interactive stop-gate maps to a deterministic action.
**Default is always full implementation.** Decide from the Comprehensive Rules + Oracle text
and **document the decision**; only defer when you cannot be confident.

| Interactive gate | Autonomous action |
|---|---|
| Behavior/timing/modal ambiguity | Resolve it from `docs/mtg_comprehensive_rules.txt` (cite the rule number in the doc). If, after consulting the rules, two readings remain genuinely defensible and would produce different game results → **defer**. |
| Needed mechanic is unsupported | **Implement** a real, general handler (default). Defer only if the mechanic is large/cross-cutting enough that a safe, correct implementation can't be verified within this single-card scope. |
| No Forge script | **Defer and move on** — never hand-author a script. A script must come only from a pre-existing local file or the Forge repo. |
| Unclear test scope | Derive the scope from the card's modes/triggers and rules; test each. If you can't state what a complete test covers → **defer**. |
| Would overflow the vocab (`N_CARD_TYPES`) | **Defer** (never grow `N_CARD_TYPES` autonomously — it changes the observation size and breaks trained checkpoints). |
| Build won't go clean, or a test fails and the fix is unclear | Fix if clear; otherwise **revert and defer**. |

**Invariants (non-negotiable):**
- Never commit a card that isn't fully implemented and passing both isolation and regression
  tests.
- Never leave the working tree dirty or broken between cards — defer means `git checkout -- . &&
  git clean -fd` back to a pristine tree.
- Never retag a script category/Origin/Destination to shortcut one card.
- Never edit an existing card script.
- Never hand-author a card script. If no local script exists and the Forge fetch fails, defer
  the card and move on.
- Never guess behavior that the rules don't support; defer instead.
- Every committed card carries a design doc that a human could audit later — that doc **is**
  the record that would otherwise have been a conversation with the user.

## Design-doc template (`docs/card_implementations/<uid>.md`)

```markdown
# <Card Name>  (vocab index <N>)

## Oracle text
<verbatim oracle text>

## Forge script
- Source: fetched (Forge@master) | pre-existing local
- Key tags: <A:/T:/S:/K:/R: lines and the AB$/SP$/DB$ categories that matter>

## Engine work
- <each parser/effect/handler change and the reason; "none — fully covered by existing
  handlers" if nothing new was needed>
- Mechanics added (general, not card-specific): <…>

## Behavioral decisions (made in lieu of asking a human)
- <each ambiguity encountered, the reading chosen, and the CR rule number that justifies it>
- <"none — behavior unambiguous" if so>

## Tests
- Isolation (test_harness): <scenario → observed result> for each mode/trigger
- Regression (observe, N games): <deck used, result: no non-fatal errors / no draws>

## Result
implemented | deferred(<reason>)
```

## Final report

When the loop ends, print a tally: count implemented and count deferred, list each deferred
card with its one-line reason (these are the only items a human needs to revisit), and confirm
the tree is clean. Push the accumulated per-card commits to the designated development branch
once at the end (or as instructed by the caller); each card remains its own commit in history.
