---
name: autonomous-implement-missing-cards
description: Autonomously implement a bounded batch of missing Magic cards inside a sandbox cloud container. The caller gives a maximum number of cards to implement; the system inspects roughly twice that many missing cards, triages them, and writes every triage-deferred card to a deferred-cards doc for user-directed implementation later. Cards are then implemented one at a time IN SEQUENCE — never in parallel — each (or each rationally-batched group sharing one new mechanic) fully implemented, built, tested, and committed on its own before the next begins. A covered card whose mechanics are all verifiably already implemented by cards that existed in the repo before this branch may skip the test step. Behavior is resolved from the Comprehensive Rules and Oracle text (no user is in the loop) or the card is cleanly deferred. The run branch is never left broken. Use for unattended, capped batch card implementation (e.g. draining the bin/resources/decks/meta worklist).
---

# Autonomous implement missing cards

Implements a **bounded** batch of missing cards **unattended**, in a sandbox cloud container.
The caller supplies a **maximum card count `N`**; the run inspects roughly **`2N`** missing cards,
triages them, and implements **up to `N`** of them — **one at a time, in sequence** — each fully
implemented, built, tested, and **committed on its own** (or as a small rationally-batched group)
before the next begins. Every card the triage step **defers** is written to a committed
**deferred-cards doc** so a human can pick it up later.

This is the autonomous sibling of `implement-missing-cards`. The engineering procedure for a
single card is the same; the differences are operational:

- **The caller sets a cap.** `N` = the maximum number of cards to *implement* this run. The run
  triages ~`2N` candidates so that, after deferrals, there are still enough implementable cards to
  reach `N`. The cap is a hard maximum on cards committed, not a quota — landing fewer is fine.
- **No user is available.** Every gate the interactive skill resolves by "STOP and ask the user"
  is resolved here by the **Autonomous decision policy** below — decide-and-document, or
  defer-and-document. Never block waiting for input.
- **Implementation is strictly sequential.** Cards are *not* implemented in parallel. The run
  works directly on one run branch; each card (or batch) runs the full
  implement → build → test → commit cycle to completion before the next card starts. (Read-only
  *triage* may still fan out — it makes no edits.)
- **One commit per card or per batch.** Each card is its own commit, except a group of cards that
  the triage step marks as rationally batchable (they share one new mechanic) — those are
  implemented and tested together and may share the mechanic commit, but each card still lands its
  own card commit where practical.
- **Triage-deferred cards are listed in a doc.** `docs/card_implementations/deferred_cards.md`
  gets a dated section naming every deferred card and why, for user-directed implementation.
- **A covered card may skip verification when its mechanics are already proven.** If every tag a
  card uses is **verifiably already implemented by a card that existed in the repo before this run
  branch** (the handler is not just present in code but exercised by a shipping card), the test
  and regression steps may be skipped for that card — registering it and confirming a clean build
  is enough. Anything that introduces or first-exercises a mechanic is fully tested.
- **The run branch is never left broken.** A card lands only after the build is clean and (unless
  verification was legitimately skipped) its tests pass. A card that cannot finish cleanly is
  reverted and deferred — the tree stays green between every commit.

**The intent is always to implement the card with its full rules functionality** per the
Comprehensive Rules. We do **not** stub, skip, or simplify a card's mechanics to save effort. The
only acceptable non-implementation outcome is a **clean deferral** (card untouched, reason
recorded in the deferred-cards doc).

## Why this shape (the design rationale)

Sequential, single-tree implementation trades parallel throughput for **simplicity and safety**:

1. **No divergent-duplicate mechanics.** The classic batch hazard is two workers independently
   implementing the same mechanic in the central dispatch files (`effect_kind.{h,cpp}`,
   `effect_table.cpp`, `parse.cpp`, `ability.{h,cpp}`, …) — each green in isolation, wrong when
   merged. Working one card at a time on one tree means the second card *sees* the first card's
   mechanic and reuses it. Triage batches cards that share a new mechanic so the mechanic is
   written exactly once.
2. **No merge/cherry-pick/quarantine machinery.** Because every card is committed onto the run
   branch the moment it is green, there is no integration phase, no worktrees, and no quarantine
   namespace. `train/card_vocab.h` and `train/card_costs.py` are edited and rebuilt inline per
   card, so they never conflict.
3. **`train/card_costs.py` stays in sync automatically.** A plain `make` runs `pygen`, which
   regenerates `card_costs.py` whenever `src/card_vocab.h` changes. Each card's `make` rebuild
   therefore refreshes the derived file, and it is committed alongside that card.
4. **The cap bounds blast radius.** Inspecting `2N` and implementing `≤N` keeps each run small
   enough to review as a single branch, while the `2N` triage buffer absorbs deferrals so the run
   still reaches `N` implementable cards when it can.

## Background

- **Build headless.** Every `make` invocation in this skill means `make HEADLESS=TRUE` unless the
  user explicitly asks for the GUI. The GUI (raylib) front end is deprecated/unmaintained and
  raylib is typically unavailable in the sandbox cloud container, so a plain `make` fails at link
  time. `HEADLESS=TRUE` still runs `pygen` (regenerating `train/card_costs.py`) and produces the
  same `bin/robomage` the test harness drives, so it satisfies every build/regen step below.
- Cards are registered in `src/card_vocab.h` as `{"Card Name", N}` (next free index `N`).
  `N_CARD_TYPES = 1024`, so indices up to 1022 are free (1023 is the token sentinel). A card
  absent from this file is "unimplemented".
- Card behavior comes from a Forge-format script in
  `bin/resources/cardsfolder/<letter>/<name>.txt`. Many missing cards already have a local
  script; others download from the upstream Forge repo.
- **Never edit an existing card script** (project rule: DO NOT MODIFY CARD SCRIPTS). The
  downloader only *adds* new files.
- **Never add a duplicate script for a card that already exists.** A double-faced card is stored
  under ONE combined `<front>_<back>.txt` file; a second front-name `<front>.txt` adds the card to
  the engine twice and shadows the combined script. The downloader is DFC-aware and handles this —
  rely on it rather than creating a `<front>.txt` yourself.
- **NEVER hand-author a card script.** A card's script must come from a pre-existing local file
  or the upstream Forge repo via the downloader. If `fetch_script.py` cannot find the card
  (non-zero exit) and no local script exists, **defer the card** (clean tree, reason "no Forge
  script available") and move on. Do not write, invent, or reconstruct a script by hand.
- The authoritative rules for how a mechanic *should* behave are checked in at
  `docs/mtg_comprehensive_rules.txt` (navigation guide: `docs/mtg_comprehensive_rules.md`).
  **When behavior, timing, or keyword semantics are in question, grep the relevant numbered rule**
  (702 keyword abilities, 704 state-based actions, 5xx combat, 613 layers) rather than relying on
  memory. This is the ground truth that replaces asking the user.

## Tools

- `train/.venv/bin/python train/missing_cards.py [--json]` — the worklist: cards referenced by
  decks (default `bin/resources/decks/meta/`) but missing from the vocab, **sorted by cross-deck
  frequency**, each tagged `has_local_script` and a `suggested_index`. Take the **top `2N`** rows
  (it has no `--limit`; slice the JSON yourself).
- `python tools/forge_fetch/fetch_script.py "Card Name"` — fetch a card's Forge script into
  `cardsfolder/` (add-only; non-zero exit ⇒ not found ⇒ **defer the card; never hand-author**).
  DFC-aware: a double-faced card lives under ONE combined `<front>_<back>.txt` script, so the tool
  skips a card already present in that form and resolves the combined filename on a front-name
  miss. **Never write a front-name `<front>.txt` for a DFC — it double-adds the card** (shadows
  the combined script; see CLAUDE.md "Card Loading System").
- `train/.venv/bin/python train/gen_card_costs.py` — regenerate `train/card_costs.py` after the
  vocab changes. Normally unnecessary to call directly: a plain `make` regenerates it via `pygen`.
- `train/test_harness.py` — exercise a card's behavior (full usage in `CLAUDE.md`). Also the
  torch-free way to run a **scripted full-game regression**: `--deck-a <a> --deck-b <b> --scripted
  --seed N` across a few seeds drives the same C++ engine and the same rule-based agent as
  `observe`.
- `train/.venv/bin/python train/train.py observe --deck <a> --opponent <b> --games N` —
  scripted in-game regression. **Requires torch** (`train.py` imports `stable_baselines3`/`sb3-contrib`
  at load), so it is unavailable wherever torch isn't installed — e.g. the headless sandbox
  container, where torch would also cost ~0.5–1 GB to re-install every session. **When `observe`
  is unavailable, fall back to the test-harness scripted games above** — they give equivalent
  engine/rules-correctness coverage (`observe` only adds model-driven play and the gym
  env/extractor pipeline, which are RL-eval concerns, not card-behavior verification).

## Orchestration

The orchestrator keeps a small ledger in the scratchpad dir
(`…/scratchpad/card_worklist.json`, status ∈ `pending | done | deferred | not-reached`) so it
survives the orchestrator's own context summarization, and dispatches one subagent per card/batch
so the heavy per-card context lives and dies inside that subagent. The orchestrator must **never**
open card scripts, build logs, or transcripts beyond what it needs to dispatch and tally.

### Branch model (one deliverable branch)

All work happens on a single **run branch `autonomous-cards/<YYYY-MM-DD>`** (or a caller-supplied
name), branched from the current development branch. Each card or batch is **its own commit** on
this branch, applied in sequence. There are **no worktrees, no cherry-pick integration, and no
quarantine branches** — a card that fails simply never gets committed (its changes are reverted)
and is recorded as deferred. The run branch is the deliverable and is always clean and building.

Record the run branch's **starting HEAD** at preflight: this is the **"prior to this branch"**
reference that defines which cards/mechanics already existed before the run (used to decide
verification-skip eligibility, below).

### Phase 0 — Preflight (once)

Confirm a clean tree (`git status --short` empty; if not, commit/stash pre-existing work so
per-card commits stay isolated). Confirm a plain `make` is green. **Create and check out the run
branch** off the current development branch, and **record its starting HEAD** as the
"prior-to-this-branch" reference. Ensure `docs/card_implementations/` exists.

### Phase 1 — Worklist + triage (inspect ~2× the cap)

1. Read the cap **`N`** from the caller (the maximum number of cards to implement). If the caller
   gave no number, **stop and surface that the cap is required** — the run is defined by it.
2. Run `train/missing_cards.py --json`. Take the **top `2N`** rows by frequency. **Pre-assign each
   a distinct vocab index** (use `suggested_index`, de-duplicated; all must stay `< 1023` — drop
   overflow as `defer(vocab overflow)`). Persist the ordered list to the ledger.
3. **Classify each of the `2N` cards** (read-only — this step may fan out, since it makes no
   edits). For each card a triage subagent fetches the script if needed (fetch fails →
   `defer(no script)`), reads its `A:`/`T:`/`S:`/`K:`/`R:` lines and their `AB$`/`SP$`/`DB$`
   categories, and checks each against `src/parse.cpp`, `src/effects/effect_kind.cpp`,
   `effect_table.cpp`. It returns a **class**:
   - `covered` — every tag is already handled; **no engine change** needed. Additionally flag
     **`verify_skip: true`** when every tag is **already exercised by at least one card that was in
     the vocab before this run branch** (cite an example card) — i.e. the mechanic is proven, not
     merely present in code. A covered card relying on a handler that *no* shipping card uses is
     `verify_skip: false` (test it).
   - `mechanic:<key>` — needs a new handler; `<key>` is a stable short name for the missing
     mechanic (e.g. `affinity`, `convoke`, `cant-attack-unless`, the unhandled `AB$`/`SP$`/`DB$`
     category). Cards sharing a `<key>` are **rationally batchable** → implemented together so the
     mechanic is written once.
   - `defer:<reason>` — no script, vocab overflow, or genuinely irreducible ambiguity.

   Triage is a *judgement aid*, not a contract: a card that turns out to need more (or less) work
   than its class suggests is handled by the implementing subagent (it implements or defers) and
   surfaced in its summary.

### Phase 2 — Record deferrals (the deferred-cards doc)

Write/append a dated section to **`docs/card_implementations/deferred_cards.md`** listing **every
triage-deferred card** with its one-line reason and whether a Forge script is available, framed as
a worklist **for user-directed implementation**. In the same doc, under a separate
**"Not reached (cap)"** heading, list any *implementable* cards from the `2N` pool that the `N`
cap leaves unimplemented this run, so nothing is silently dropped. Commit this doc (its own commit,
`Record deferred cards (<date>)`). Use the **Deferred-cards doc format** below.

### Phase 3 — Sequential implementation (up to `N` cards)

Build the ordered list of **units**: each `covered` card is a singleton unit; each
`mechanic:<key>` group is **one batch unit** (all its cards). Order units by worklist priority
(highest cross-deck frequency first); a batch takes the priority of its highest-frequency member.

Then implement units **strictly one at a time, directly on the run branch**, until the count of
**cards committed reaches `N`**:

- Before starting a unit, if `committed >= N`, **stop** (the remaining implementable cards become
  "Not reached (cap)" entries in the deferred doc). A batch is **atomic** — if started it runs to
  completion, so the final unit may land the count slightly above `N`; that is acceptable.
- For each unit, **dispatch one subagent** (no worktree — it operates on the real run-branch
  working tree; the orchestrator `await`s it so only one mutates the tree at a time) with the
  **Per-unit subagent prompt** below. The subagent runs the full cycle —
  implement → build → test → commit — and returns before the next unit begins.
- A unit that cannot finish cleanly **reverts its own changes** (`git checkout -- . &&
  git clean -fd`) and returns `deferred`; the orchestrator adds those cards to the deferred doc and
  moves on. A batch may ship its mechanic + passing cards and defer only the card(s) that failed.

**Verification-skip:** a `covered` card flagged `verify_skip: true` may **skip the isolation test
and regression** — register it, confirm a clean `make`, document it, and commit. Every other card
(covered-unproven, and every card in a mechanic batch) is fully tested before its commit.

### Phase 3.5 — Post-implementation code review (when the run completes in-session)

If the run reaches its target — the `N`-card cap is hit, or the implementable queue is exhausted —
**while the session is still live** (the user has not terminated it), run a self-review before the
final report:

1. Run the **`code-review`** skill at **medium** effort over the run branch's accumulated diff
   (its changes vs. the run's starting HEAD).
2. **Fix only LOW-RISK findings** — a clear, localized correctness or cleanup fix that is obviously
   safe and that you can re-verify with a quick build (and a harness/regression check when it
   touches card behavior). Anything risky, cross-cutting, or ambiguous is **left for the human**,
   not patched autonomously — never destabilize the green run branch to chase a finding. Each fix
   is its own commit (or folded into the relevant card's follow-up commit) with the usual trailers,
   and the branch must still build clean afterward.
3. **Emit a summary report** of the review: issues found (with severity and location) vs. issues
   fixed (with the commit), and which were deliberately left for the user and why. This report goes
   into the Phase 4 final report (and may be appended to the deferred-cards doc as a "Review
   findings" section if any were deferred for the human).

This step is best-effort and bounded: if the session is terminated before it runs, skip it; the
per-card design docs and deferred-cards doc remain the durable handoff.

### Phase 4 — Final report & handoff

Print a tally: implemented (each its own commit on the run branch; note which skipped verification
and why), deferred (with the path to `deferred_cards.md`), and not-reached-due-to-cap. Confirm the
run branch builds clean and `git status` is clean. **Push the run branch.** **Do not open the PR
automatically** — leave that to the human, who owns the deferred queue. The deferred-cards doc is
the explicit, committed handoff of everything left to do.

## Per-unit subagent prompt (a single card, or a shared-mechanic batch)

Give the subagent this prompt, filling in the unit. It has no memory of other cards, works on the
**real run-branch tree** (not a worktree), and must finish with a **clean tree** — committed on
success, fully reverted on defer.

> Implement the following into the Robomage engine, fully and correctly, **in sequence on the
> current branch** (you are NOT in a worktree; do not create one). You are autonomous — **there is
> no user to ask.** Finish with a **clean tree**: commit on success, fully revert on defer
> (`git checkout -- . && git clean -fd`). Never leave the tree dirty or broken.
>
> **Unit:** either a single card `<NAME>=<INDEX>`, or a batch sharing one new mechanic `<KEY>`:
> `<NAME1>=<INDEX1>`, `<NAME2>=<INDEX2>`, … Each card may carry a `verify_skip` flag.
>
> 1. **Get every script.** For each card, if no local `cardsfolder/` script exists, run
>    `python tools/forge_fetch/fetch_script.py "<NAME>"`. NOT FOUND (non-zero exit) → **defer that
>    card** (reason "no Forge script available"); in a batch the others continue. **NEVER
>    hand-author, reconstruct, or modify a script.**
> 2. **Implement the mechanic (batch) or confirm coverage (covered card).**
>    - *Batch:* implement the shared mechanic **once** as a real, general handler keyed on the
>      tag's *intended meaning* — honor the script's actual `SP$`/`AB$`/`DB$` category and
>      `Origin$`/`Destination$`/`ChangeType$`/etc. **NEVER retag** one category into another to
>      shortcut a card; a retag corrupts every other card sharing that tag. Look up exact behavior
>      in `docs/mtg_comprehensive_rules.txt` by rule number and cover the whole mechanic, not just
>      these cards. If it is too large/cross-cutting to implement and verify safely → **defer the
>      whole batch** (clean tree).
>    - *Covered card:* confirm every `A:`/`T:`/`S:`/`K:`/`R:` tag is already handled by
>      `parse.cpp` / `effect_kind.cpp` / `effect_table.cpp`. If a new mechanic is in fact needed
>      and it is small, implement it as a real general handler (never retag); if it is
>      large/cross-cutting, **defer** and note the misclassification.
> 3. **For each card in the unit, in order:**
>    a. **Register** `{"<NAME>", <INDEX>}` in `src/card_vocab.h` (keep apostrophes; index already
>       `< 1023`).
>    b. **Build** with plain `make` (this also regenerates `train/card_costs.py` via `pygen`). The
>       build MUST be clean; fix any new error.
>    c. **Test** — UNLESS this card is flagged `verify_skip` (its mechanics are already proven by a
>       pre-existing shipping card), in which case the clean build in (b) is sufficient and you may
>       skip to (d). Otherwise: **isolation test** with `train/test_harness.py` (inline
>       `--hand-a/--library-a/--battlefield-a/...` or a `temp/` stacked deck, driven by semantic
>       `--play` specs; **never `--interactive`**), covering every mode/trigger; then **real-game
>       regression** with a deck that casts it (swap ~4 copies into a `temp/` deck if needed):
>       prefer `train/.venv/bin/python train/train.py observe --games 6`, but **if `observe` is
>       unavailable (no torch — e.g. the headless container) fall back to torch-free scripted full
>       games through the harness across a few seeds**: `train/.venv/bin/python
>       train/test_harness.py --deck-a temp/<deck> --deck-b temp/<opp> --scripted --seed N` for
>       N=1,2,3 (same engine + same scripted agent). Expect **no non-fatal errors and no draws**;
>       only pre-existing cosmetic `WARNING: Unrecognized ability param` lines are acceptable. Clean
>       up temp decks.
>    d. **Document** `docs/card_implementations/<uid>.md` (uid = lowercased name, spaces→`_`,
>       apostrophes dropped) using the **Design-doc template** — note explicitly when verification
>       was skipped and which pre-existing card proves each mechanic.
>    e. **Commit** just that card: stage the new/edited engine source, `src/card_vocab.h`, the
>       regenerated `train/card_costs.py`, the new `cardsfolder/` script (if freshly fetched), and
>       the design doc. Message `Implement <NAME>` + a short body (mechanics, tests or
>       "verification skipped — mechanics proven by <card>") and the required Co-Authored-By /
>       Claude-Session trailers. Do **not** push. (For a batch, the shared mechanic's engine edits
>       fold into the **first** card's commit, or land as a leading `Implement <KEY> mechanic`
>       commit before the card commits.)
> 4. **Defer protocol.** If a card cannot be finished with confidence (ambiguous, mechanic too
>    large, no script, vocab overflow, build/test won't go clean and the fix is unclear), do **not**
>    commit a partial/guessed implementation — revert its changes and report it deferred. A wrong
>    guess is worse than a deferral. Never leave the tree dirty between cards.
>
> Return a structured summary: per card `{name, outcome: done|deferred, index, commit, tests
> (one line per scenario + result, or "skipped — proven by <card>"), reason, verify_skipped:
> true|false}`, plus the unit's branch (the run branch) and, for a batch, a one-line `mechanic`
> note (what was added + the CR rules). Flag `needs_review: true` for any card you are **not fully
> confident** is correct (a behavior resolved from the rules where a second reading was plausible,
> a complex interaction not exhaustively tested, a freshly-added mechanic) so it can get a closer
> look.

## Autonomous decision policy (replaces "ask the user")

Because no human is in the loop, each interactive stop-gate maps to a deterministic action.
**Default is always full implementation.** Decide from the Comprehensive Rules + Oracle text and
**document the decision**; only defer when you cannot be confident.

| Interactive gate | Autonomous action |
|---|---|
| Behavior/timing/modal ambiguity | Resolve from `docs/mtg_comprehensive_rules.txt` (cite the rule number in the doc). If two readings remain genuinely defensible and would change game results → **defer**. |
| Needed mechanic is unsupported | **Implement** a real, general handler (default). In a batch, do it once for the whole group. Defer only if a safe, correct implementation can't be verified within scope. |
| No Forge script | **Defer and move on** — never hand-author. |
| Unclear test scope | Derive scope from the card's modes/triggers and rules; test each. If you can't state what a complete test covers → **defer**. (Does not apply to `verify_skip` cards.) |
| Would overflow the vocab (`N_CARD_TYPES`) | **Defer** (never grow `N_CARD_TYPES` autonomously — it changes the observation size and breaks trained checkpoints). |
| Build won't go clean / a test fails and the fix is unclear | Fix if clear; otherwise **revert and defer**. |

**Invariants (non-negotiable):**
- **Implementation is sequential.** Cards are never implemented in parallel; each unit's full
  implement → build → test → commit cycle completes before the next unit starts. (Read-only triage
  may fan out.)
- **The cap is a maximum.** Never commit more than `N` cards (a final atomic batch may land
  slightly over); fewer is fine.
- **Every draw is an engine failure — treat any drawn game as a bug, never as a pass.** A draw in a
  scripted regression or verification game is a real defect (a loop, a stuck stack, an unkillable
  board) and must be investigated, root-caused, and fixed — exactly like a non-fatal error. A card
  whose regression produces a draw does **not** ship: fix the cause or revert/defer the card. Since
  no user is in the loop, the "user explicitly accepted this draw" exception never applies — a draw
  always blocks the card.
- Never commit a card that isn't fully implemented and (unless verification was legitimately
  skipped per the `verify_skip` rule) passing both isolation and regression tests.
- A card may **skip verification only** when every mechanic it uses is verifiably already
  implemented by a card that existed in the repo **before this run branch**; the design doc must
  name the proving card(s). Anything that introduces or first-exercises a mechanic is fully tested.
- Never leave the run branch dirty or broken — defer means `git checkout -- . && git clean -fd`.
- Never retag a script category/Origin/Destination to shortcut a card.
- Never edit an existing card script. Never hand-author a card script.
- Never guess behavior the rules don't support; defer instead.
- Every committed card carries a design doc a human could audit later; every deferred card is named
  in `deferred_cards.md`. Those docs **are** the record that would otherwise have been a
  conversation with the user.

## Deferred-cards doc format (`docs/card_implementations/deferred_cards.md`)

```markdown
# Deferred cards — for user-directed implementation

Cards the autonomous runs triaged but did NOT implement. Each needs a human decision or a larger
change than an unattended run should make. Pick one up by running the interactive
`implement-missing-cards` skill (or directing a fresh autonomous run at it once unblocked).

## Run <YYYY-MM-DD>  (cap N=<N>, inspected <2N> cards)

### Deferred at triage
- **<Card Name>** — <one-line reason> (Forge script: available | not found)
- …

### Not reached (cap)
Implementable this run but left for a future run because the N=<N> cap was reached first
(highest-frequency first):
- **<Card Name>** — <covered | mechanic:<key>>, suggested index <N>
- …
```

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
- Mechanics added (general, not card-specific): <…, with the shared mechanic key if batched>

## Behavioral decisions (made in lieu of asking a human)
- <each ambiguity encountered, the reading chosen, and the CR rule number that justifies it>
- <"none — behavior unambiguous" if so>

## Tests
- Isolation (test_harness): <scenario → observed result> for each mode/trigger
  — OR "skipped: mechanics already proven by <pre-existing card(s)>"
- Regression (observe, or torch-free harness scripted games, N games/seeds): <tool + deck used,
  result: no non-fatal errors / no draws> — OR "skipped (verify_skip)"

## Result
implemented | implemented (verification skipped — proven by <card>) | deferred(<reason>)
```

## Final report

When the loop ends, print a tally: count implemented (each its own commit on the run branch; note
which skipped verification), deferred (with reasons — and confirm they are recorded in
`docs/card_implementations/deferred_cards.md`), and not-reached-due-to-cap. Confirm the run branch
is clean and builds. Push the accumulated per-card commits to the run branch once at the end; each
card remains its own commit in history. Do not open the PR — the deferred-cards doc is the handoff.

## Fallback (no subagents available)

If subagent dispatch is unavailable, the orchestrator does the same procedure inline: triage the
top `2N`, write the deferred doc, then loop over the units doing register → `make` →
test (unless `verify_skip`) → document → commit one unit at a time directly on the run branch,
**explicitly compacting/clearing context after each commit**. The cap, the sequencing, and every
invariant are identical.
