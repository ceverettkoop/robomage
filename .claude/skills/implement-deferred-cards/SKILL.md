---
name: implement-deferred-cards
description: Implement a bounded batch of cards FROM the deferred-cards doc (docs/card_implementations/deferred_cards.md) — the cards prior autonomous runs triaged but left for a human. Unlike the autonomous runs, this skill DEFERS NOTHING: every selected card is implemented. It front-loads all implementation questions to the user in ONE up-front round (card selection + the genuine per-card behavioral/approach decisions), then runs autonomously — one card at a time, implement → build → test → commit each before the next — until a final code-review + push. Use when the user wants to drain the deferred queue (e.g. "implement 10 cards from the deferred list", "use the implement-deferred workflow").
---

# Implement deferred cards

Implements a **bounded** batch of cards pulled **from the deferred-cards doc**
(`docs/card_implementations/deferred_cards.md`) rather than from the fresh missing-cards worklist.
These are the cards earlier autonomous runs deliberately set aside because each needs a missing
engine subsystem or a human decision. This skill is the **user-directed, no-deferral** counterpart
to `autonomous-implement-cards`:

- **Nothing is deferred.** The caller names a count `N` (default ~10). Every selected card is
  implemented to full rules functionality. The only acceptable non-implementation outcome is a
  genuine, surfaced blocker — and even then the tree is left clean and building and the situation
  is reported, not silently dropped.
- **All questions are front-loaded.** Before any code is written, investigate every candidate's
  script and the engine subsystems it needs, then ask the user **every** implementation question in
  ONE up-front round (`AskUserQuestion`, ≤4 questions/call, multiple calls allowed): which cards to
  implement, and the genuine per-card behavioral / approach / architecture decisions. After the
  user answers, **go autonomous** — make no further stops until the end-of-run review.
- **Sequential, one tree, one commit per card (or shared-mechanic batch).** Identical mechanics to
  `autonomous-implement-cards`: work directly on the run branch; each unit runs the full
  implement → build → test → commit cycle to completion before the next begins; shared mechanics are
  written once and reused by later units.
- **The deferred doc is kept honest.** Each implemented card is removed from (or struck through in)
  `deferred_cards.md` so the queue shrinks as the run lands cards.

The engineering procedure for a single card is the same as `implement-missing-cards` /
`autonomous-implement-cards`; read those skills for the per-card mechanics. The differences are
operational, captured below.

## When to use

- The user points at the deferred queue and wants it drained: "implement N cards from the deferred
  list", "use the implement-deferred workflow", "knock out the deferred cards one at a time".
- A human IS available up front (for the question round) but wants the bulk of the run unattended.

If the user instead wants a fresh-from-the-meta-worklist run that *triages and defers*, use
`autonomous-implement-cards`. If they want a single interactive card with a stop on every ambiguity,
use `implement-missing-cards`.

## Background (same invariants as the autonomous run)

- **Build headless:** every `make` means `make HEADLESS=TRUE` (raylib is absent; a plain `make`
  fails at link). The headless build still runs `pygen` (regenerating `train/card_costs.py`).
- Register cards in `src/card_vocab.h` as `{"Card Name", N}` (next free index `< 1023`).
  **Never** grow `N_CARD_TYPES` and **never** change `STATE_SIZE`/`OBS_SIZE` or the
  observation/state-vector layout — trained checkpoints depend on them. New per-player or
  per-permanent state that isn't already in the vector stays internal (not added to the obs).
- **Never edit or hand-author a card script.** Scripts come from a pre-existing local file or the
  upstream Forge repo via `python tools/forge_fetch/fetch_script.py "Card Name"`. Honor the script's
  actual `SP$`/`AB$`/`DB$` categories and `Origin$`/`Destination$`/`ChangeType$`/etc. — **never
  retag** one category into another to shortcut a card.
- Rules ground truth is `docs/mtg_comprehensive_rules.txt` (nav guide `docs/mtg_comprehensive_rules.md`).
  Grep the numbered rule (702 keywords, 704 SBAs, 5xx combat, 611/613 continuous effects & layers)
  rather than relying on memory.
- Testing without torch: `train.py observe` needs torch (usually unavailable in the container). Use
  the torch-free path instead — `train/test_harness.py` with semantic `--play` specs for isolation,
  and `--deck-a/--deck-b ... --scripted --seed N` across a few seeds for regression. **Never**
  `--interactive` (no TTY). No commas inside card names in harness args.

## Phase 0 — Preflight (once)

Confirm a clean tree (`git status --short` empty; commit/stash any stray pre-existing work). Confirm
`make HEADLESS=TRUE` is green. Be on the caller-specified run branch (create it off the development
branch if needed) and **record its starting HEAD** as the "prior-to-this-branch" reference (used
later to decide verification-skip eligibility and to scope the final review diff). Read
`docs/card_implementations/deferred_cards.md`.

## Phase 1 — Candidate selection + deep investigation (read-only, may fan out)

1. Read `N` from the caller (default ~10 if unspecified). Read the deferred doc's candidate list.
2. **Pick the candidate pool** (≈`N`, or a few more to give the user choice): prefer highest
   cross-deck frequency (cross-check `train/missing_cards.py --json`) and **group cards that share
   one new mechanic** (e.g. an energy trio, an earthbend pair) so the mechanic is written once and
   the group is a single batch unit.
3. **Investigate every candidate** before asking anything. Fan out read-only `Explore` agents (they
   make no edits): for each card read its `A:`/`T:`/`S:`/`K:`/`R:` lines and `AB$`/`SP$`/`DB$`
   categories, and check each against `src/parse.cpp`, `src/effects/effect_kind.cpp`,
   `effect_table.cpp`, and the relevant systems — reporting, with `file:line`, exactly what engine
   support EXISTS vs is MISSING and the nearest pattern to extend. The goal is to know precisely
   which subsystems must be built and where the genuine decision points are.

## Phase 2 — Front-load ALL questions (one round, then go quiet)

From the investigation, formulate **every** question whose answer changes what you build, and ask
them up front with `AskUserQuestion`. Good questions (make your recommended option first, labelled
"(Recommended)" only when you genuinely recommend it):

- **Card selection / order** — confirm the `N` cards (offer to swap the riskiest for simpler
  deferred cards). Highest-leverage question; once answered you will not ask again.
- **Scope / generality of the hardest mechanic** — e.g. "build a fully general alt-cost-cast-from-a-zone
  path (reusable for other resources) vs a scoped-but-correct one". Capture any generalization the
  user wants (e.g. "parameterize by resource so energy and life share one path").
- **Architecture forks** — e.g. "refactor the existing hardcoded Ward path into a general
  `Mode$ BecomesTarget` trigger vs add the new trigger separately and leave Ward untouched".
- **State representation** — e.g. "generic player-counter map (and migrate the existing poison field
  into it) vs a dedicated field".

Record the answers in a run ledger (scratchpad) so they survive context summarization, then **stop
asking** — every remaining ambiguity is resolved autonomously from the Comprehensive Rules for the
rest of the run. Update `deferred_cards.md` if the selection changes which cards remain deferred.

## Phase 3 — Sequential implementation (no deferrals)

Order the selected units by priority (a batch takes its highest-frequency member's priority).
Maintain a **run ledger** in the scratchpad listing, after each unit, the **reusable mechanics it
added** (handler/effect names, helper signatures, file:line) — forward this verbatim to later units
so they REUSE rather than re-implement shared mechanics (this is the whole point of one-tree
sequencing).

For each unit, **dispatch one subagent** (no worktree — it edits the real run-branch tree; `await`
it so only one agent mutates the tree at a time). Give it: the user's locked decisions, the relevant
investigation findings (so it doesn't re-investigate), the ledger's "mechanics already built" list,
the card script(s), the exact sub-features to build, and the hard constraints above. The subagent
runs the full cycle:

a. Get the script (local or `fetch_script.py`; never hand-author).
b. Implement the mechanic(s) as **real, general handlers** keyed on the tag's intended meaning
   (never retag), reusing anything already built this run. Look up exact behavior in the CR by rule
   number.
c. Register `{"<NAME>", <INDEX>}` in `src/card_vocab.h`.
d. `make HEADLESS=TRUE` — must be clean (only pre-existing cosmetic `WARNING: Unrecognized ability
   param` lines acceptable).
e. **Test** with `test_harness.py`: isolation (`--play` specs covering every mode/trigger) +
   torch-free scripted regression (`--deck-a/--deck-b --scripted --seed N` for a few N). **No
   non-fatal errors, no draws** — a draw is a defect, never a pass. Clean up temp decks. (A
   `covered` card whose every mechanic is already proven by a card that existed before the run
   branch may skip the test step — register, build clean, document.)
f. **Document** `docs/card_implementations/<uid>.md` (uid = lowercased name, spaces→`_`, apostrophes
   dropped) per the design-doc template (Oracle text; script source + key tags; engine work + CR
   rule numbers; behavioral decisions made from the rules; tests → results; result line).
g. **Remove the card from `deferred_cards.md`** (delete its bullet or mark it ✅ implemented this
   run, with the commit/branch).
h. **Commit** just that unit (engine sources, `src/card_vocab.h`, regenerated
   `train/card_costs.py`, any freshly-fetched `cardsfolder/` script, the design doc, the
   deferred-doc edit). Message `Implement <NAME>` + short body (mechanics added / CR rules), ending
   with the required `Co-Authored-By:` and `Claude-Session:` trailers. Do **not** push. Do **not**
   put any model identifier in the commit. (For a batch, the shared mechanic's edits fold into the
   first card's commit or land as a leading `Implement <KEY> mechanic` commit.)

**No-deferral rule.** Because the user opted into implementing these, push to finish each card. If a
unit hits a genuine blocker, still leave the tree **clean and building** (revert any piece that
won't compile, commit the correct working subset), and **report the blocker in the final summary** —
do not silently drop it and do not leave the tree broken. A wrong guess is worse than an honest
report, but the default and strong expectation is full implementation.

## Phase 4 — End-of-run review + handoff

When all selected cards are committed (or the count is reached), while the session is live:

1. Run the **`code-review`** skill at **medium** over the run branch's accumulated diff (vs the
   recorded starting HEAD).
2. **Fix only low-risk findings** (clear, localized, re-verifiable with a quick build + harness
   check). Leave anything risky/cross-cutting for the human; never destabilize the green branch.
   Each fix is its own commit (or folded into the relevant card) with the usual trailers.
3. **Final report:** tally implemented (each its own commit; note any verification-skips and why),
   any surfaced blockers, and the review findings (fixed vs left-for-human). Confirm the branch
   builds clean and `git status` is clean. **Push the run branch** (`git push -u origin <branch>`,
   retrying network failures with backoff). **Do not open a PR** unless the user asks.

## Invariants (non-negotiable)

- Sequential implementation; each unit's full cycle completes before the next starts (read-only
  triage/investigation may fan out).
- Front-load **all** questions in Phase 2; after that, resolve ambiguity from the CR and do not stop
  until the Phase-4 review.
- No deferrals: implement every selected card; a true blocker is reported, not silently dropped, and
  never leaves the tree broken or dirty.
- Every draw or non-fatal error in a regression is a defect to fix, never a pass.
- Never change `N_CARD_TYPES`/`STATE_SIZE`/`OBS_SIZE` or the obs/state layout; never retag a script
  category/Origin/Destination; never edit or hand-author a card script.
- Keep `deferred_cards.md` honest — remove/strike each card as it lands.
- Every implemented card carries an auditable design doc; the run ledger records every reusable
  mechanic for later units.
