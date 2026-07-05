---
name: implement-missing-cards
description: Implement Magic cards a deck references but that are missing from the engine — triage ALL the user-specified cards together, front-load every implementation question in ONE up-front round, then implement them sequentially (batching cards that share one new mechanic), building/testing/committing each unit before the next, gated throughout by the project's `make check` CI suite. Defers nothing by default, and — unless the user explicitly asks it to run "autonomously" — PAUSES to ask the user whenever, even mid-run, it becomes unclear how to implement a card fully per the Comprehensive Rules using generalizable engine patterns. Use when adding cards from scraped metagame decks (bin/resources/decks/meta) or whenever a deck names a card not in src/card_vocab.h.
---

# Implement missing cards

Adds cards the engine doesn't yet know about. The cards the user names are **triaged together**,
**every implementation question is gathered up front in one round**, and the cards are then
implemented **sequentially — one unit at a time** (a unit is a single card, or a small group that
shares one new mechanic), each fully implemented, built, tested, and committed before the next
begins. Every unit's build and test step is the project's own CI gate (`make check` /
`train/ci_check.py`, see `docs/ci.md`) — the same command CI runs on push — rather than a
bespoke build/regression recipe, so a passing unit is provably CI-clean before it's committed.

By default this skill runs **interactively**:

- **A human is in the loop.** Ambiguity is resolved by **asking the user** — front-loaded in one
  round, and again mid-run for anything that round didn't anticipate (see *Stop-and-ask gates*).
- **Nothing is deferred by default.** Every card the user selects is implemented to full rules
  functionality. There is no "triage → defer to a doc" step; a card that looks hard becomes a
  *question*, not a deferral.
- **The user can opt into autonomous behavior.** If the user explicitly says to run
  "autonomously" (or "don't stop to ask", "just decide"), switch off the mid-run pauses and
  resolve ambiguity per the *Autonomous decision policy* below — from the Comprehensive Rules, or
  defer-and-document — without blocking on input. Absent that explicit instruction, the default is
  to **pause and ask**.

**The intent is always to implement the card with its full rules functionality.** Every mechanic a
card has must work correctly per the Comprehensive Rules — we do **not** stub, skip, or simplify a
card's mechanics. The only acceptable non-implementation outcome is a card the **user** chooses to
set aside (and even then the tree is left clean and building). Skipping a mechanic is a very rare
exception taken only with the user's explicit approval.

The guiding rule for the mid-run pause: **stop to ask only to resolve *how* a rule or mechanic
should behave** (ambiguous timing, modal choices, several reasonable interpretations) **or *how*
to fit it to a generalizable engine pattern, or when the test scope is unclear** — never to get
permission to leave a mechanic out, and never to shortcut the work. Default is full
implementation; ask to clarify behavior, not to find an exit.

## Background

- **Build headless:** every `make` in this skill means `make HEADLESS=TRUE` — the preferred
  development build (the raylib GUI front end is deprecated/unmaintained and often unavailable, so
  a plain `make` can fail at link). The headless build still runs `pygen` (regenerating
  `train/card_costs.py`) and produces the same `bin/robomage` the test harness drives, so it
  satisfies every build/regen step below.
- Cards are registered in `src/card_vocab.h` as `{"Card Name", N}` (next free index `N`). The
  embedding refactor raised the cap to `N_CARD_TYPES = 1024`, so indices up to 1022 are free (1023
  is the token sentinel). A card absent from this file is "unimplemented". **Never grow
  `N_CARD_TYPES`** (or change `STATE_SIZE`/`OBS_SIZE` or the obs/state layout) — trained
  checkpoints depend on them; if a selection would overflow the vocab, that is a stop-and-ask gate.
- Card behavior comes from a Forge-format script in
  `bin/resources/cardsfolder/<letter>/<name>.txt`. Many missing cards already have a local script;
  others can be downloaded from the upstream Forge repo.
- **Never edit an existing card script** (project rule: DO NOT MODIFY CARD SCRIPTS). The downloader
  only *adds* new files. Hand-author a script only when none exists, after confirming behavior with
  the user.
- **Never add a duplicate script for a card that already exists.** A double-faced card is stored
  under ONE combined `<front>_<back>.txt` file; a second front-name `<front>.txt` adds the card to
  the engine twice and shadows the combined script. The downloader is DFC-aware and handles this —
  rely on it rather than creating a `<front>.txt` yourself.
- The authoritative rules for how a mechanic *should* behave are checked in at
  `docs/mtg_comprehensive_rules.txt` (navigation guide: `docs/mtg_comprehensive_rules.md`).
  **When a card's behavior, timing, or keyword semantics are in question, look up the relevant
  numbered rule there** (e.g. rule 702 for keyword abilities, 704 for state-based actions, 5xx for
  combat, 611/613 for continuous effects & layers) rather than relying on memory — grep by rule
  number, don't read the whole file.

## Tools

- `train/.venv/bin/python train/missing_cards.py [--json]` — the worklist: cards referenced by
  decks (default `bin/resources/decks/meta/`) but missing from the vocab, sorted by cross-deck
  frequency, each tagged `has_local_script` and a `suggested_index`. Use this both to resolve a
  "top-N from the worklist" request and to look up `suggested_index`/frequency for any named card.
- `python tools/forge_fetch/fetch_script.py "Card Name"` — fetch a card's Forge script into
  `cardsfolder/` (add-only; exit code = cards not found, so non-zero ⇒ hand-author needed). It is
  DFC-aware: double-faced cards live under ONE combined `<front>_<back>.txt` script, so the tool
  skips a card already present in that form and, on a front-name miss, fetches the combined file
  Forge serves under its combined name — **never hand-create a front-name `<front>.txt` for a DFC,
  that double-adds the card** (the front-name file shadows the combined one; see CLAUDE.md "Card
  Loading System"). Accented names aren't transliterated — fetch e.g. "Lórien Revealed" by its
  ASCII `lorien_revealed` stem by hand.
- `train/.venv/bin/python train/gen_card_costs.py` — regenerate `train/card_costs.py` after editing
  the vocab. Normally unnecessary to call directly: a `make HEADLESS=TRUE` regenerates it via `pygen`.
- `train/test_harness.py` — exercise a card's exact behavior (see `CLAUDE.md` for full usage).
  Drive a precise line with semantic `--play` specs; **never `--interactive`** (no TTY). This is
  the isolation-test tool — it targets one card's modes/triggers, which the generic CI tiers below
  don't check for a brand-new card.
- **`make check`** (equivalently `train/.venv/bin/python train/ci_check.py`) — the project's CI
  gate (see `docs/ci.md`): builds headless, then runs the `pygen` / `vocab` / `replay` / `smoke` /
  `fuzz` tiers. This **is** the build-and-regression step for every unit in this skill — it is the
  exact command CI runs on push, so a unit that passes it locally is provably CI-clean. Useful
  flags for a faster per-unit pass: `--tier pygen,vocab,smoke` (skip the slower `replay`/`fuzz`
  tiers mid-run; run the **full** `make check` at least once before Phase 4 handoff),
  `--smoke-games N` / `--fuzz-games N` to scale depth, `--matchups "a:b"` to scope smoke/fuzz to a
  deck containing the new card. `make check` requires the Forge scripts provisioned first
  (`train/.venv/bin/python tools/forge_fetch/provision_decks.py`; the SessionStart hook does this
  in a fresh cloud session).

## Workflow

The orchestrator owns the user conversation (the up-front question round and every mid-run pause)
and a small run **ledger** in the scratchpad dir (the selected units, their indices, and — after
each unit — the **reusable mechanics it added**: handler/effect names, helper signatures,
`file:line`). The ledger survives context summarization and is forwarded to later units so they
**reuse** rather than re-implement a shared mechanic.

### Phase 0 — Preflight (once)

Confirm a clean tree (`git status --short` empty; commit/stash any stray pre-existing work so
per-card commits stay isolated). Confirm a `make check` is green on the starting HEAD, so any
later red run is attributable to this session's changes. Be on the working branch the user
wants (create one off the development branch if the user asks), and **record its starting HEAD** as
the "prior-to-this-branch" reference — it defines which mechanics already existed before the run
(used to decide verification-skip eligibility and to scope the final review diff). Ensure
`docs/card_implementations/` exists.

### Phase 1 — Resolve the card set and triage it together

1. **Resolve the cards the user specified.** "All cards specified by the user" can be: explicit
   card names; one or more decks to scan against the vocab; or a top-N slice of
   `train/missing_cards.py` (when the user just gives a count). Resolve them all into one ordered
   candidate list (highest cross-deck frequency first); pre-assign each a distinct vocab index
   (`suggested_index`, de-duplicated; all `< 1023`).

2. **Triage every candidate together** — read-only, so this step **may fan out** (`Explore`
   agents make no edits). For each card: fetch the script if absent (`fetch_script.py`); read its
   `A:`/`T:`/`S:`/`K:`/`R:` lines and their `AB$`/`SP$`/`DB$` categories; check each against
   `src/parse.cpp`, `src/effects/effect_kind.cpp`, `effect_table.cpp`, and the relevant systems,
   reporting (with `file:line`) exactly what engine support **exists** vs is **missing** and the
   nearest pattern to extend. Classify each card:
   - `covered` — every tag is already handled; **no engine change** needed. Flag
     `verify_skip: true` only when every tag is **already exercised by a card that existed before
     this branch's starting HEAD** (cite the proving card); such a card may skip the test step.
   - `mechanic:<key>` — needs a new handler; `<key>` is a stable short name for the missing
     mechanic (e.g. `affinity`, `convoke`, the unhandled `AB$`/`SP$`/`DB$` category). Cards
     sharing a `<key>` are **rationally batchable** → one batch unit, mechanic written once.
   - `needs-decision:<question>` — the script has no Forge source, the behavior is genuinely
     ambiguous, the mechanic is large/cross-cutting enough to need a scope decision, or it would
     overflow the vocab. This is **not a deferral** — it becomes a question in Phase 2.

   Triage is a judgement aid, not a contract: a card that turns out to need more (or less) than its
   class suggested is handled when its unit runs (and, interactively, surfaced back as a question).

### Phase 2 — Front-load ALL questions (one round), then implement

From the triage, formulate **every** question whose answer changes what you build, and ask them up
front with `AskUserQuestion` (≤4 questions/call, multiple calls allowed). Make your recommended
option first, labelled "(Recommended)" only when you genuinely recommend it. Good questions:

- **Card selection / order** — confirm the cards to implement (offer to swap a risky one for a
  simpler candidate, or to drop it). Highest-leverage question.
- **Behavior / timing of any ambiguous card** — the `needs-decision` items: present the
  Comprehensive-Rules-consistent reading you recommend and ask the user to confirm or correct it.
  Never ask in order to leave a mechanic out — frame full implementation as the expected path.
- **Scope / generality of the hardest mechanic** — e.g. "build a fully general handler (reusable by
  other cards) vs a scoped-but-correct one", and any generalization the user wants.
- **Architecture forks** — e.g. "refactor the existing hardcoded path into a general trigger vs add
  the new trigger separately and leave the old path untouched".
- **No Forge script** — if a card must be hand-authored, confirm its intended behavior here before
  writing anything.
- **Vocab overflow** — if a selection would exceed the vocab, raise it (never grow `N_CARD_TYPES`
  silently).

Record the answers in the run ledger. After this round, proceed to Phase 3 — but in the default
(interactive) mode you are **not** committed to silence: anything this round didn't anticipate is a
mid-run pause (Phase 3, *Stop-and-ask gates*). If — and only if — the user explicitly asked to run
**autonomously**, skip the mid-run pauses and resolve later ambiguity per the autonomous policy
(Comprehensive Rules, or defer-and-document).

### Phase 3 — Sequential implementation (one unit at a time)

Build the ordered list of **units**: each `covered` card is a singleton; each `mechanic:<key>`
group is one batch unit. Order by worklist priority (a batch takes its highest-frequency member's
priority). Implement units **strictly one at a time, directly on the working branch** — never in
parallel — each running the full **implement → build → test → commit** cycle to completion before
the next begins, so a later unit *sees and reuses* an earlier unit's new mechanic.

For each unit, you may **dispatch one subagent** to hold the heavy per-card context (the
orchestrator `await`s it, so only one agent mutates the tree at a time) or do the work inline. Give
a subagent: the user's locked Phase-2 decisions, the relevant triage findings (so it doesn't
re-investigate), the ledger's "mechanics already built" list, the card script(s), the exact
sub-features to build, and the constraints below. Per card, in order:

a. **Get the script.** Local file, else `fetch_script.py`. NOT FOUND with no local script ⇒
   stop-and-ask (a hand-authored script needs the user's confirmed behavior). Never modify or
   hand-author a script without that confirmation.
b. **Implement the mechanic (batch) or confirm coverage (covered card)** as a **real, general
   handler keyed on the tag's intended meaning** — honor the script's actual `SP$`/`AB$`/`DB$`
   category and `Origin$`/`Destination$`/`ChangeType$`/`DefinedPlayer$`/etc., reusing anything the
   ledger says was already built this run. **NEVER retag** one category into another to shortcut a
   card — a retag silently corrupts every other card sharing that tag. (It is fine to *ignore* an
   irrelevant tag when the rest already fully specify the behavior — a cosmetic
   `StackDescription$`, or a `ChangeNum$` count-SVar when the effect already moves all matching
   cards — but never repurpose a tag.) Look up exact behavior in `docs/mtg_comprehensive_rules.txt`
   by rule number and cover the whole mechanic, not just this card.
c. **Register** `{"<Name>", <Index>}` in `src/card_vocab.h` (keep apostrophes; index already
   `< 1023`).
d. **Build** with `make HEADLESS=TRUE` for the fast dev loop while iterating (this also
   regenerates `train/card_costs.py` via `pygen`). The build MUST be clean; non-fatal errors are
   unacceptable. Fix any new error before continuing.
e. **Test** — UNLESS this card is `verify_skip` (mechanics already proven by a pre-existing
   shipping card), in which case a clean `make check` (below) is sufficient. Otherwise: first
   **isolation test** with `train/test_harness.py` (inline `--hand-a/--library-a/--battlefield-a/...`
   or a `temp/` stacked deck, driven by semantic `--play` specs), covering every mode/trigger this
   card has. Then run **`make check`** (or, mid-run, `train/ci_check.py --tier pygen,vocab,smoke`
   scoped to a deck containing the new card via `--matchups` for a faster loop — but the **full**
   `make check` must pass at least once for the unit before its commit). This is the same command
   CI runs on push, so it doubles as the unit's regression gate: `pygen`/`vocab` catch codegen or
   vocab drift, `smoke`/`fuzz` catch crashes, stalls, and draws in real games with the new card in
   the mix (add it to a `league/`-style deck first if no existing deck plays it). Expect **no
   non-fatal errors and no draws**; only pre-existing cosmetic `WARNING: Unrecognized ability
   param` lines are acceptable. **Treat any draw as a bug** (a loop / stuck stack / unkillable
   board) — root-cause and fix it; the only pass-able draw is one the **user** has explicitly
   accepted. Clean up any temp decks.
f. **Document** `docs/card_implementations/<uid>.md` (uid = lowercased name, spaces→`_`, apostrophes
   dropped) per the design-doc template (Oracle text; script source + key tags; engine work + CR
   rule numbers; behavioral decisions and which were confirmed with the user; tests → results;
   result line — note explicitly when verification was skipped and which pre-existing card proves
   each mechanic).
g. **Commit** just that unit: the new/edited engine source, `src/card_vocab.h`, the regenerated
   `train/card_costs.py`, any freshly-fetched `cardsfolder/` script, and the design doc. Message
   `Implement <Name>` + a short body (mechanics added / CR rules, or "verification skipped — proven
   by <card>") with the required `Co-Authored-By:` / `Claude-Session:` trailers. Do **not** push,
   and do **not** put any model identifier in the commit. (For a batch, the shared mechanic's edits
   fold into the first card's commit or land as a leading `Implement <KEY> mechanic` commit.)
h. **Record** the reusable mechanic(s) the unit added in the ledger, then move to the next unit.

After the ledger has each unit's added mechanics, forward that list verbatim to the next unit so it
reuses rather than re-implements — this is the whole point of one-tree sequencing.

**Stop-and-ask gates (interactive default).** During Phase 3, if something the up-front round did
not resolve makes it **unclear how to implement a card fully per the Comprehensive Rules using a
generalizable engine pattern**, **pause and ask the user** rather than guess. When the per-unit
work runs in a subagent, the subagent does **not** guess and does **not** defer: it **returns the
open question to the orchestrator** (with the script context and its recommended CR-consistent
reading), the orchestrator asks the user via `AskUserQuestion`, then **re-dispatches the unit with
the answer**. The gates:

- A card's **expected behavior is ambiguous** — unclear timing, modal choices, or several
  defensible interpretations of how a rule should resolve.
- The **right generalizable pattern is unclear** — more than one reasonable way to fit the mechanic
  into the engine's existing handlers, with a real correctness/architecture tradeoff.
- The **test scope is unclear** — you can't confidently state what a complete test covers.
- A needed **mechanic is large enough that scope needs a decision** — present implementing it in
  full as the expected path; simplifying or skipping requires the user's explicit approval.
- **No Forge script** exists and the card must be hand-authored.
- Adding the card would **overflow the vocab** (`N_CARD_TYPES`).

A mechanic the engine doesn't yet support is **not** itself a reason to stop — the default is to
build a real, general handler for it. Stop to clarify *behavior* or *pattern*, never to find a
shortcut. Never guess a card's behavior, and never silently drop a mechanic.

**Autonomous override / decision policy.** If the user explicitly asked to run autonomously, do
not pause: resolve each gate yourself —
- **Behavior/pattern ambiguity:** resolve from `docs/mtg_comprehensive_rules.txt`, choose the
  reading consistent with the CR, and document the exact rule number in the design doc.
- **Defer-and-document** (skip this card, do not implement it) only when: two readings remain
  genuinely defensible *and* would change game results, no Forge script exists and none can be
  confidently hand-authored, or implementing it would overflow the vocab (`N_CARD_TYPES`).
- The tree is never left dirty or broken: a deferred card reverts its own changes (`git checkout
  -- . && git clean -fd`) and is logged (with the reason) for the user instead of guessed at.

### Phase 4 — End-of-run review + handoff

When all selected units are committed (or the user's count is reached), while the session is live:

1. Run the **`code-review`** skill at **medium** over the working branch's accumulated diff (vs the
   recorded starting HEAD).
2. **Fix only low-risk findings** — clear, localized, re-verifiable with a quick build (and a
   harness/CI check when behavior is touched). Anything risky or cross-cutting is raised with the
   user, not patched silently; never destabilize the green branch. Each fix is its own commit (or
   folded into the relevant card) with the usual trailers.
3. Run the **full `make check`** one final time over the accumulated diff (every tier, default
   seed) — the same gate CI runs on push. It must be green before handoff; if any post-review fix
   broke a tier, fix it and re-run.
4. **Final report:** tally implemented (each its own commit; note any verification-skips and why),
   anything the user chose to set aside, the review findings (fixed vs left-for-user), and the
   final `make check` result. Confirm the branch builds clean and `git status` is clean. Push only
   if the user asks; do **not** open a PR unless asked.

## Design-doc template (`docs/card_implementations/<uid>.md`)

```markdown
# <Card Name>  (vocab index <N>)

## Oracle text
<verbatim oracle text>

## Forge script
- Source: fetched (Forge@master) | pre-existing local | hand-authored (behavior confirmed with user)
- Key tags: <A:/T:/S:/K:/R: lines and the AB$/SP$/DB$ categories that matter>

## Engine work
- <each parser/effect/handler change and the reason; "none — fully covered by existing handlers"
  if nothing new was needed>
- Mechanics added (general, not card-specific): <…, with the shared mechanic key if batched>

## Behavioral decisions
- <each ambiguity, the reading chosen, the CR rule number, and whether the user confirmed it>
- <"none — behavior unambiguous" if so>

## Tests
- Isolation (test_harness): <scenario → observed result> for each mode/trigger
  — OR "skipped: mechanics already proven by <pre-existing card(s)>"
- CI gate (`make check` / `train/ci_check.py`): <tiers run, deck(s) exercising the new card,
  result: no non-fatal errors / no draws> — OR "skipped (verify_skip)"

## Result
implemented | implemented (verification skipped — proven by <card>) | set aside by user(<reason>)
```

## Invariants (non-negotiable)

- **Triage together, then ask once.** All user-specified cards are triaged as one set; every
  question that round can foresee is asked up front in a single round.
- **Sequential implementation.** Cards are never implemented in parallel; each unit's full
  implement → build → test → commit cycle completes before the next starts. (Read-only triage may
  fan out.)
- **Default is full implementation, no silent deferral.** Interactively, an unclear card becomes a
  *question* (front-loaded, or a mid-run pause that a subagent surfaces back to the orchestrator),
  never a guess and never a quiet skip. Deferral only happens when the user opted into autonomous
  mode, or chose to set a card aside.
- **Every draw or non-fatal error is a defect to fix, never a pass** — the only exception is a draw
  the user explicitly accepts.
- A card may skip verification **only** when every mechanic it uses is verifiably already
  implemented by a card that existed before this branch's starting HEAD; the design doc names the
  proving card(s).
- Never change `N_CARD_TYPES`/`STATE_SIZE`/`OBS_SIZE` or the obs/state layout; never retag a script
  category/Origin/Destination; never edit an existing card script; never hand-author one without the
  user's confirmed behavior.
- Never leave the branch dirty or broken between units.
- Every implemented card carries an auditable design doc; the run ledger records every reusable
  mechanic for later units.
