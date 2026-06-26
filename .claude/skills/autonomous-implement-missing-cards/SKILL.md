---
name: autonomous-implement-missing-cards
description: Autonomously implement many missing Magic cards inside a sandbox cloud container — triaged into parallel-safe and mechanic-shared tiers, each card fully implemented, tested in isolation and in a real game, documented in a committed design doc, and committed on its own. No user input is available, so behavior is resolved from the Comprehensive Rules and Oracle text (and documented) or the card is deferred with a written rationale; the integrated tree is never left broken. Cards with no shared engine work run in parallel git worktrees; cards that need the same new mechanic are batched so the mechanic is implemented exactly once. Use for unattended batch card implementation (e.g. draining the bin/resources/decks/meta worklist) when no human is in the loop.
---

# Autonomous implement missing cards

Drains the missing-card worklist **unattended**, in a sandbox cloud container. Each card is
fully implemented, tested in isolation **and** in a real game, documented in a committed design
doc, and **committed on its own**. Work is split into tiers so it parallelizes without the merge
hazards that naive one-worktree-per-card would create.

This is the autonomous sibling of `implement-missing-cards`. The engineering procedure for a
single card is the same; the differences are operational:

- **No user is available.** Every gate the interactive skill resolves by "STOP and ask the
  user" is resolved here by the **Autonomous decision policy** below — decide-and-document, or
  defer-and-document. Never block waiting for input.
- **Each card gets its own commit** plus a committed design doc under
  `docs/card_implementations/`.
- **Heavy context lives and dies in a per-task subagent**, so the orchestrator only ever holds
  a tiny ledger between cards. Parallel-safe cards run in their own **git worktrees**.
- **The integrated trunk is never left broken.** A card's commit is integrated onto the trunk
  only after it builds clean and its tests pass *against the merged tree* — not just in
  isolation. Anything else → the branch is quarantined for human review, trunk stays green.

**The intent is always to implement the card with its full rules functionality** per the
Comprehensive Rules. We do **not** stub, skip, or simplify a card's mechanics to save effort.
The only acceptable non-implementation outcome is a **clean deferral** (card untouched on
trunk, reason documented).

## Why tiers (the design rationale)

The contention in batch card work is not the cards — it's three shared files:

1. **`src/card_vocab.h`** — every card appends `{"Name", N}` at the list tail. Solved by
   **pre-assigning each card a distinct index** during triage; appends then never collide
   semantically.
2. **`train/card_costs.py`** — *fully derived* from the vocab by `gen_card_costs.py`. It must
   **never be committed from a parallel worker**; it is regenerated once, serially, at
   integration. A per-worktree copy conflicts on every single card for no reason.
3. **The central dispatch files** — `src/effects/effect_kind.{h,cpp}`, `effect_table.cpp`,
   `parse.cpp`, `action_processor.cpp`, `ability.{h,cpp}`, `state_manager_triggers.cpp`. Almost
   every *nontrivial* card edits these. Two workers touching them independently risk (a)
   append-only textual conflicts (resolvable) and, worse, (b) **two divergent implementations
   of the same mechanic**, neither aware of the other — each tests green in isolation but the
   merged build is wrong. This is the same failure mode as the retag rule, one level up.

So we **triage first**, then run two tiers:

- **Tier A — covered cards** (no engine change; existing handlers fully cover the script).
  These touch only per-card files (`cardsfolder/<x>.txt`, the design doc) plus their
  pre-assigned vocab line. **Embarrassingly parallel** — each in its own worktree, own commit.
- **Tier B — mechanic cards** (need a new handler). **Batched by shared mechanic**: one
  subagent owns mechanic X end-to-end and implements *every* card in X's group, committing each
  card separately. The mechanic is implemented exactly once, so no divergent-duplicate risk.
  Distinct mechanic groups may run in parallel worktrees (different handler files); their only
  shared touchpoints are the append-only central enum/table, resolved at serial integration.

Integration is **always serial** and **always rebuilds + smoke-tests against the merged tree**.
That rebuild is the safety net the naive design lacks.

## Background

- Cards are registered in `src/card_vocab.h` as `{"Card Name", N}` (next free index `N`).
  `N_CARD_TYPES = 1024`, so indices up to 1022 are free (1023 is the token sentinel). A card
  absent from this file is "unimplemented".
- Card behavior comes from a Forge-format script in
  `bin/resources/cardsfolder/<letter>/<name>.txt`. Many missing cards already have a local
  script; others download from the upstream Forge repo.
- **Never edit an existing card script** (project rule: DO NOT MODIFY CARD SCRIPTS). The
  downloader only *adds* new files.
- **NEVER hand-author a card script.** A card's script must come from a pre-existing local file
  or the upstream Forge repo via the downloader. If `fetch_script.py` cannot find the card
  (non-zero exit) and no local script exists, **defer the card** (clean tree, reason "no Forge
  script available") and move on. Do not write, invent, or reconstruct a script by hand.
- The authoritative rules for how a mechanic *should* behave are checked in at
  `docs/mtg_comprehensive_rules.txt` (navigation guide: `docs/mtg_comprehensive_rules.md`).
  **When behavior, timing, or keyword semantics are in question, grep the relevant numbered
  rule** (702 keyword abilities, 704 state-based actions, 5xx combat, 613 layers) rather than
  relying on memory. This is the ground truth that replaces asking the user.

## Tools

- `train/.venv/bin/python train/missing_cards.py [--json]` — the worklist: cards referenced by
  decks (default `bin/resources/decks/meta/`) but missing from the vocab, sorted by cross-deck
  frequency, each tagged `has_local_script` and a `suggested_index`.
- `python tools/forge_fetch/fetch_script.py "Card Name"` — fetch a card's Forge script into
  `cardsfolder/` (add-only; non-zero exit ⇒ not found ⇒ **defer the card; never hand-author**).
- `train/.venv/bin/python train/gen_card_costs.py` — regenerate `train/card_costs.py` after the
  vocab changes (orchestrator-only, at integration).
- `train/test_harness.py` — exercise a card's behavior (full usage in `CLAUDE.md`).
- `train/.venv/bin/python train/train.py observe --deck <a> --opponent <b> --games N` —
  scripted in-game regression.

## Orchestration

The orchestrator runs six phases. It does almost no engineering itself; it dispatches
subagents and keeps a small ledger in the scratchpad dir
(`…/scratchpad/card_worklist.json`, status ∈ `pending | done | deferred | quarantined`) so the
ledger survives the orchestrator's own context summarization. The orchestrator must **never**
open card scripts, build logs, or transcripts beyond what it needs to dispatch and tally —
that defeats the context-clearing design.

### Branch model (one deliverable branch off `main`; quarantine namespace for the rest)

- **Run branch `autonomous-cards/<YYYY-MM-DD>`** (or a caller-supplied name), off `main`. The
  **deliverable** — every card that passes all gates lands here as its own commit, and the PR is
  opened from this branch. Always builds clean.
- **Implementation branches `card/<uid>` / `mech/<key>`**, each off the run branch's *starting
  point* (self-contained snapshots). Subagents commit here. **Kept alive until Phase 5 ends** —
  this is how quarantined work is preserved without reconstruction.
- **Quarantine branches `quarantine/<YYYY-MM-DD>/<uid>`.** A card that fails at integration *or*
  at verification has its implementation branch **renamed into this namespace** and its commits
  kept **off the run branch**: nothing is lost (each is a checkout-able, self-contained branch a
  human can build and test), nothing broken rides in the PR. This namespace *is* the "needs a
  human" queue. After the run a human can, per card, **fix it** (check out the branch, repair,
  then re-integrate onto the run branch via the Phase 4 gate) or **abandon it** (delete the
  branch) — see Phase 6.

Each implementation branch ends one of two ways: **passed → its commits are on the run branch,
delete the redundant impl branch; quarantined → renamed to `quarantine/<date>/<uid>`, kept aside.**

### Phase 0 — Preflight (once)

Confirm a clean tree (`git status --short` empty; if not, commit/stash pre-existing work so
per-card commits stay isolated). Confirm a plain `make` is green. **Create a dedicated run
branch and check it out** — `autonomous-cards/<YYYY-MM-DD>` (or a caller-supplied name), branched
from the current development branch. **All work in this run happens on that branch**: it is the
**trunk** that integration cherry-picks onto, so the branch you started from is never touched and
the whole batch is reviewable/revertable as one branch. Record its HEAD as the integration base.
Create `docs/card_implementations/` if absent.

### Phase 1 — Worklist + triage

1. Run `train/missing_cards.py --json`. **Pre-assign each card a distinct vocab index** (use
   `suggested_index`, de-duplicated; all must stay `< 1023` — drop overflow cards as
   `deferred(vocab overflow)`). Persist the ordered list to the ledger.
2. **Classify each card in parallel** (read-only — fan out freely). For each card a triage
   subagent reads its script (fetch if needed; if fetch fails → `defer(no script)`), reads its
   `A:`/`T:`/`S:`/`K:`/`R:` lines and their `AB$`/`SP$`/`DB$` categories, and checks each
   against `src/parse.cpp`, `src/effects/effect_kind.cpp`, `effect_table.cpp`. It returns a
   **class**:
   - `covered` — every tag is already handled; **no engine change** needed → Tier A.
   - `mechanic:<key>` — needs a new handler; `<key>` is a stable short name for the missing
     mechanic (e.g. `affinity`, `convoke`, `cant-attack-unless`, the unhandled `AB$`/`SP$`/`DB$`
     category). Cards sharing a `<key>` will be batched → Tier B.
   - `defer:<reason>` — no script, vocab overflow, or genuinely irreducible ambiguity.

   Triage is a *judgement aid*, not a contract: a Tier-A card that turns out to need engine
   work, or a misgrouped mechanic, is handled by the implementing subagent (it implements the
   mechanic or defers) and surfaced in its summary.

### Phase 2 — Tier A: parallel covered cards

For each `covered` card, **dispatch one subagent in its own git worktree** (Agent tool,
`general-purpose`, `isolation: "worktree"`) with the **Per-card subagent prompt**. It commits on
its `card/<uid>` branch (off the run branch's start) and does the full single-card procedure
(steps 1–8). Because covered cards touch no engine code, these run fully concurrently. Each
returns its branch name + commit hash + test summary, or a defer reason. The worktree (and its
heavy context) is discarded on return; the branch persists for integration.

### Phase 3 — Tier B: mechanic-grouped cards

Group the `mechanic:<key>` cards by `<key>`. For each group, **dispatch one subagent in its own
worktree** with the **Per-mechanic-group subagent prompt** (below), committing on its
`mech/<key>` branch (off the run branch's start). It implements the mechanic **once** as a real,
general handler, then implements **each card in the group**, **committing each card separately**
(the mechanic edits fold into the first card's commit, or land as a leading `Implement <mechanic>`
commit). Distinct groups run concurrently. Each returns its branch + an ordered list of
`{card, commit}` + per-card status.

### Phase 4 — Serial integration (the safety net)

Working on **trunk**, integrate branches **one at a time**, in a deterministic order (Tier A
first — least likely to conflict — then Tier B groups by ascending card count). For each
branch, dispatch a single **integrator subagent** (no worktree — it operates on the real trunk
working tree; the `await` serializes them so only one mutates trunk at a time):

1. `git cherry-pick` the branch's per-card commits in order.
2. On conflict, resolve **only** the known append-only files by keeping **both** sides:
   `src/card_vocab.h` (both vocab lines), and the central enum/table tails
   (`effect_kind.h`, `effect_table.cpp`, etc.). **Any other conflict, or any non-trivial
   conflict, is a red flag → abort the cherry-pick (`git cherry-pick --abort`) and quarantine
   the branch** rather than hand-merging blind.
3. Regenerate `train/card_costs.py` (`gen_card_costs.py`) and amend it into the integrated
   commit(s) so the derived file matches the merged vocab.
4. Run a plain `make`. **It must be clean.**
5. Run a smoke test of *this* card against the merged tree — the same isolation scenario the
   implementing subagent used (or a quick `observe … --games 4` with a deck that casts it).
6. **Green → keep** (the card is now on the run branch as its own commit; its impl branch is now
   redundant and is deleted at the end). **Red →** reset the run branch to the pre-cherry-pick
   HEAD (`git reset --hard`) so it stays green, then **quarantine**: rename the card's impl
   branch to `quarantine/<date>/<uid>` (`git branch -m card/<uid> quarantine/<date>/<uid>`) so
   its commits are preserved off the run branch, and mark the card `quarantined` with the failure.

A quarantined card is the integration-time equivalent of a deferral: its self-contained branch
survives under `quarantine/<date>/` for a human to fix or abandon (Phase 6), but it does not ship.

### Phase 5 — Whole-batch verification (capstone)

Integration smoke-tests each card *as it lands*; this phase verifies the **fully integrated run
branch** with every new card present at once. Run it only after Phase 4 finishes, on the run
branch, by invoking the **`demonstrate-session-cards`** skill — that is the verification capstone
for exactly this situation. It enumerates the run's new cards from the committed design docs,
builds temp decks that contain them, runs the `engine-sanity-check` scripted games over real
matches, **proves every new card is actually demonstrated** (cast + its defining clause resolved)
— backfilling any gap with a targeted `test_harness.py` scenario — and reviews the verbose output
for engine bugs. Prioritize cards returned `NEEDS_REVIEW: yes` (ambiguity resolved from rules,
complex interactions, happy-path-only coverage, new-mechanic cards) for the deepest targeted
scrutiny.

`demonstrate-session-cards` *flags* findings rather than fixing them. Because no human is in the
loop here, the orchestrator resolves each finding per the Autonomous decision policy: a genuine
engine bug whose fix is **clear and small** → fix it on the run branch (its own follow-up commit)
and re-verify; otherwise **quarantine that card** — snapshot its commits onto
`quarantine/<date>/<uid>` (`git branch quarantine/<date>/<uid> <run-branch>` *before* removing
the commits), reset/revert it off the run branch so the branch stays green, and record the
failure. A finding that is scripted-agent suboptimality, not an engine bug, is noted but not acted
on. Never leave a known-broken card on the run branch.

### Phase 6 — Final report & handoff

Print a tally: implemented (on the run branch; note which passed Phase 5 clean vs. fixed-on-review),
deferred (one-line reasons), quarantined (failure + `quarantine/<date>/<uid>` branch name).
Confirm the run branch builds clean and `git status` is clean. **Push the run branch** (each card
is its own commit; the whole batch is one reviewable branch off `main`). **Do not open the PR
automatically** — leave that to the human, because the quarantine queue is theirs to resolve
first.

The handoff is explicitly two-pathed per quarantined card, and the report must spell this out so a
human (or a follow-up Claude run they direct) can act on it:

- **Fix before PR:** `git checkout quarantine/<date>/<uid>`, repair the card, then re-integrate it
  onto the run branch through the **Phase 4 gate** (cherry-pick → regen costs → `make` →
  smoke-test); on green, delete the quarantine branch. Repeat per card, then open the PR from the
  run branch.
- **Abandon:** delete `quarantine/<date>/<uid>`. It was never on the run branch, so the PR is
  unaffected.

Delete the impl branches of cards that shipped (their commits are on the run branch); keep every
`quarantine/<date>/*` branch until the human fixes or abandons it.

**Fallback if subagents/worktrees are unavailable:** do cards linearly as the old skill did —
per-card procedure unchanged — but **explicitly compact/clear context after each commit**, and
still keep vocab + costs as the serial step (register, regen, build, test, commit one card at a
time directly on trunk). The autonomous policy and invariants are identical.

## Per-card subagent prompt (Tier A — covered cards)

Give the subagent this verbatim, substituting `<NAME>` and `<INDEX>`. It has no memory of other
cards and must finish with a clean worktree (committed on success, reverted on defer).

> Implement the single Magic card **`<NAME>`** into the Robomage engine, fully and correctly,
> then commit it on its own branch. You are running autonomously in a git worktree — **there is
> no user to ask.** Finish with a **clean tree** (committed on success, reverted on defer).
>
> 1. **Get a script.** If no local `cardsfolder/` script exists, run
>    `python tools/forge_fetch/fetch_script.py "<NAME>"`. If NOT FOUND (non-zero exit),
>    **defer** (see Defer protocol) with reason "no Forge script available". **NEVER
>    hand-author or reconstruct a script.** Never modify an existing script.
> 2. **Confirm covered.** Read the `A:`/`T:`/`S:`/`K:`/`R:` lines and their categories and
>    confirm each is already handled by `src/parse.cpp`, `effect_kind.cpp`, `effect_table.cpp`.
>    **If it turns out a new mechanic is needed after all**, you may implement it as a real,
>    general handler (look up exact behavior in `docs/mtg_comprehensive_rules.txt` by rule
>    number) — but **never retag** a category/Origin/Destination to shortcut this card, and if
>    the mechanic is large/cross-cutting, **defer** instead and note it was misclassified.
> 3. **Register.** Append `{"<NAME>", <INDEX>}` to `src/card_vocab.h` (keep apostrophes; index
>    is `<INDEX>`, already `< 1023`). Do **not** run `gen_card_costs.py` and do **not** stage
>    `train/card_costs.py` — the orchestrator regenerates it at integration.
> 4. **Build.** Run plain `make`. Clean build required; fix any new error. (You may regen costs
>    locally if the build needs it, but do not commit the regenerated file.)
> 5. **Test in isolation.** With `train/test_harness.py`, build a focused scenario exercising
>    *this* card (inline `--hand-a/--library-a/--battlefield-a/...` or a `temp/` stacked deck,
>    driven by semantic `--play` specs; never `--interactive`). Verify narrative + decoded
>    state match the expected outcome across the card's real modes/triggers.
> 6. **Regression in a real game.** Run a few scripted games with the card in a deck via
>    `train/train.py observe … --games 6` (existing deck that casts it, else a `temp/` copy with
>    4 copies swapped in). Expect **no non-fatal errors and no draws**; only pre-existing
>    cosmetic `WARNING: Unrecognized ability param` lines are acceptable. Clean up temp decks.
> 7. **Document.** Write `docs/card_implementations/<uid>.md` (uid = lowercased name, spaces→`_`,
>    apostrophes dropped) using the **Design-doc template** — Oracle text, Forge tags, every
>    engine change and *why* (likely "none — fully covered"), every behavioral decision with its
>    CR rule citation, the test scenarios + observed results, the regression result.
> 8. **Commit.** Stage exactly: any new/edited engine source, `src/card_vocab.h`, the new
>    `cardsfolder/` script (if freshly fetched), and the design doc — **NOT** `card_costs.py`.
>    Commit `Implement <NAME>` + a short body (mechanics, tests) and the required Co-Authored-By
>    / Claude-Session trailers. Do **not** push. Report branch + commit hash.
> 9. **Defer protocol.** If you cannot finish with confidence (ambiguous, mechanic too large,
>    no script, vocab overflow), do **not** commit a partial/guessed implementation. Run
>    `git checkout -- . && git clean -fd` to fully restore a clean tree, then return a deferral
>    summary (card name + specific reason). A wrong guess is worse than a deferral.
>
> Return a single structured summary: `OUTCOME: done|deferred`, `INDEX`, `BRANCH`, `COMMIT`,
> `FILES`, `TESTS` (one line per scenario + result), `NOTES/REASON`, and
> `NEEDS_REVIEW: yes|no` + `REVIEW_REASON` — set `yes` whenever you are **not fully confident**
> the card is correct: a behavior you resolved from the rules where a second reading was
> plausible, a complex/conditional interaction you couldn't exhaustively test, a new mechanic you
> added, or any happy-path-only coverage. These cards get a targeted re-test in Phase 5.

## Per-mechanic-group subagent prompt (Tier B — shared mechanic)

Give the subagent this verbatim, substituting `<KEY>` and the `<NAME>=<INDEX>` list for the
group. It implements the mechanic **once**, then each card, **one commit per card**.

> Several cards share one missing mechanic, **`<KEY>`**: `<NAME1>=<INDEX1>`, `<NAME2>=<INDEX2>`,
> … Implement the mechanic **once** as a real, general handler, then implement **each card**,
> committing **each card separately** on this branch. You are autonomous in a git worktree —
> **no user to ask.** Finish with a clean tree.
>
> 1. **Get every script** (`fetch_script.py` per card; a card whose fetch fails is **deferred**,
>    not blocking the rest). Never hand-author or modify scripts.
> 2. **Implement the mechanic generally**, keyed on the tag's *intended meaning* — honor the
>    script's actual `SP$`/`AB$`/`DB$` category and `Origin$`/`Destination$`/`ChangeType$`/etc.
>    **Never retag** one category into another to shortcut a card; a retag corrupts every other
>    card sharing that tag. Look up exact behavior in `docs/mtg_comprehensive_rules.txt` by rule
>    number and design the handler to cover the whole mechanic, not just these cards. If the
>    mechanic is too large/cross-cutting to implement and verify safely here, **defer the whole
>    group** with that reason (clean tree).
> 3. **For each card, in order:** register `{"<NAME>", <INDEX>}` in `src/card_vocab.h` (do NOT
>    stage `card_costs.py`); plain `make` clean; **isolation test** (`test_harness.py`, semantic
>    `--play`, all modes/triggers); **real-game regression** (`observe … --games 6`, no
>    non-fatal errors, no draws); write `docs/card_implementations/<uid>.md`; then **commit just
>    that card** (`Implement <NAME>`, trailers). The mechanic's engine edits fold into the
>    **first** card's commit (or land as a leading `Implement <KEY> mechanic` commit). A card
>    that fails to verify is **reverted out of the staging** and reported deferred; the rest of
>    the group still ships.
> 4. **Never** commit a card that isn't tested green. Never leave the tree dirty between cards.
>
> Return: `KEY`, `MECHANIC` (one line on what was added + the CR rules), then per card
> `OUTCOME: done|deferred`, `INDEX`, `COMMIT`, `TESTS`, `NOTES/REASON`, `NEEDS_REVIEW: yes|no` +
> `REVIEW_REASON` (set `yes` when not fully confident — see the Tier A prompt; **every card in a
> group that added a new mechanic defaults to `NEEDS_REVIEW: yes`**), plus `BRANCH`.

## Autonomous decision policy (replaces "ask the user")

Because no human is in the loop, each interactive stop-gate maps to a deterministic action.
**Default is always full implementation.** Decide from the Comprehensive Rules + Oracle text and
**document the decision**; only defer when you cannot be confident.

| Interactive gate | Autonomous action |
|---|---|
| Behavior/timing/modal ambiguity | Resolve from `docs/mtg_comprehensive_rules.txt` (cite the rule number in the doc). If two readings remain genuinely defensible and would change game results → **defer**. |
| Needed mechanic is unsupported | **Implement** a real, general handler (default). In Tier B, do it once for the whole group. Defer only if a safe, correct implementation can't be verified within scope. |
| No Forge script | **Defer and move on** — never hand-author. |
| Unclear test scope | Derive scope from the card's modes/triggers and rules; test each. If you can't state what a complete test covers → **defer**. |
| Would overflow the vocab (`N_CARD_TYPES`) | **Defer** (never grow `N_CARD_TYPES` autonomously — it changes the observation size and breaks trained checkpoints). |
| Build won't go clean / a test fails and the fix is unclear | Fix if clear; otherwise **revert and defer**. |
| Merge conflict at integration beyond append-only vocab/table tails | **Quarantine** the branch (trunk stays green); never hand-merge blind. |

**Invariants (non-negotiable):**
- **Every draw is an engine failure — treat any drawn game as a bug, never as a pass.** A draw
  in a scripted regression or verification game is a real defect (a game that can't resolve,
  e.g. a loop, a stuck stack, an unkillable board) and must be investigated, root-caused, and
  fixed — exactly like a non-fatal error. A card whose regression produces a draw does **not**
  ship: fix the cause (its own follow-up commit) and re-verify, or revert/quarantine the card.
  The **only** exception is a draw the **user has explicitly identified as acceptable**; since no
  user is in the loop here, that exception never applies autonomously — so autonomously, a draw
  always blocks the card.
- Never commit a card that isn't fully implemented and passing both isolation and regression
  tests — and at integration, passing against the **merged** tree.
- Never leave trunk dirty or broken — defer means `git checkout -- . && git clean -fd`;
  a failed integration means `git reset --hard` back to the pre-cherry-pick HEAD.
- `train/card_costs.py` is **derived**: regenerated only by the orchestrator at integration,
  never committed from a parallel worker.
- Never retag a script category/Origin/Destination to shortcut a card.
- Never edit an existing card script. Never hand-author a card script.
- Never guess behavior the rules don't support; defer instead.
- Every committed card carries a design doc a human could audit later — that doc **is** the
  record that would otherwise have been a conversation with the user.

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
- Mechanics added (general, not card-specific): <…, with the shared mechanic key if Tier B>

## Behavioral decisions (made in lieu of asking a human)
- <each ambiguity encountered, the reading chosen, and the CR rule number that justifies it>
- <"none — behavior unambiguous" if so>

## Tests
- Isolation (test_harness): <scenario → observed result> for each mode/trigger
- Regression (observe, N games): <deck used, result: no non-fatal errors / no draws>
- Integration smoke (merged tree): <result>

## Result
implemented | deferred(<reason>) | quarantined(<integration failure>)
```

## Final report

When the loop ends, print a tally: count implemented (on trunk), deferred, and quarantined;
list each deferred card with its one-line reason and each quarantined card with its integration
failure + branch name (these are the only items a human needs to revisit), and confirm trunk is
clean and builds. Push the accumulated per-card commits to the designated development branch
once at the end; each card remains its own commit in history.
