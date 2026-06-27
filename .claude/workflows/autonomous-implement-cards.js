export const meta = {
  name: 'autonomous-implement-cards',
  description: 'Bounded autonomous card run: the caller gives a max card count N; inspect ~2N missing cards, triage them, record every deferral in a deferred-cards doc, then implement up to N IN SEQUENCE (one commit per card or shared-mechanic batch) — implement, build, test, commit each unit before the next. Covered cards whose mechanics are already proven by pre-existing cards may skip the test step.',
  phases: [
    { title: 'Setup', detail: 'create the run branch; record its starting HEAD (prior-to-branch reference)' },
    { title: 'Triage', detail: 'inspect ~2N missing cards: classify covered | mechanic:<key> | defer; flag verify_skip' },
    { title: 'Defer doc', detail: 'write docs/card_implementations/deferred_cards.md (deferred + not-reached)' },
    { title: 'Implement', detail: 'sequential, up to N cards: each card/batch implement→build→test→commit before the next' },
    { title: 'Report', detail: 'tally implemented / deferred / not-reached; push run branch (no PR)' },
  ],
}

// ---------------------------------------------------------------------------
// args: { maxCards: number, cards?: [{name, index, hasScript}], runBranch?: string, date?: string }
//   - maxCards (N): REQUIRED. Maximum number of cards to IMPLEMENT this run. The run inspects
//     ~2N missing cards and triages them, so deferrals still leave enough to reach N.
//   - cards: optionally pass the inspected worklist explicitly (already sliced to ~2N and
//     pre-assigned distinct indices < 1023). If omitted, Triage discovers it via missing_cards.py.
//   - date: the run date string (scripts can't call Date.now(); pass it in). Defaults to "run".
//   - runBranch: deliverable branch name. Defaults to `autonomous-cards/<date>`. Implementation
//     work happens directly on this one branch — no worktrees, no quarantine, no cherry-pick.
// ---------------------------------------------------------------------------
const N = (args && Number(args.maxCards)) || 0
if (!N || N < 1) throw new Error('autonomous-implement-cards: args.maxCards (the max number of cards to implement, N>=1) is required')
const INSPECT = 2 * N
const DATE = (args && args.date) || 'run'
const RUN_BRANCH = (args && args.runBranch) || `autonomous-cards/${DATE}`
const DEFER_DOC = 'docs/card_implementations/deferred_cards.md'

const TRIAGE_SCHEMA = {
  type: 'object',
  required: ['name', 'index', 'klass'],
  properties: {
    name: { type: 'string' },
    index: { type: 'integer' },
    klass: { type: 'string', enum: ['covered', 'mechanic', 'defer'] },
    mechanicKey: { type: 'string', description: 'stable short name when klass=mechanic, else empty' },
    verifySkip: { type: 'boolean', description: 'covered card whose every tag is already exercised by a card that existed before this run branch (mechanics proven) — its test step may be skipped' },
    provenBy: { type: 'string', description: 'when verifySkip=true, an example pre-existing card that proves the mechanics' },
    hasScript: { type: 'boolean' },
    deckCount: { type: 'integer', description: 'cross-deck frequency (worklist priority); higher = implement sooner' },
    reason: { type: 'string', description: 'one line; required when klass=defer' },
  },
}

const CARD_RESULT_SCHEMA = {
  type: 'object',
  required: ['name', 'outcome'],
  properties: {
    name: { type: 'string' },
    outcome: { type: 'string', enum: ['done', 'deferred'] },
    index: { type: 'integer' },
    commit: { type: 'string' },
    tests: { type: 'string' },
    verifySkipped: { type: 'boolean' },
    needsReview: { type: 'boolean' },
    reason: { type: 'string' },
  },
}

const UNIT_RESULT_SCHEMA = {
  type: 'object',
  required: ['cards'],
  properties: {
    kind: { type: 'string', enum: ['covered', 'batch'] },
    key: { type: 'string', description: 'mechanic key when kind=batch' },
    mechanic: { type: 'string', description: 'one line: what was added + CR rules (batch only)' },
    cards: { type: 'array', items: CARD_RESULT_SCHEMA },
  },
}

// === Phase 0: Setup — create the run branch, record its starting HEAD ======
phase('Setup')
await agent(
  `Preflight for a bounded autonomous card-implementation run (cap N=${N}; will inspect ~${INSPECT} cards). On the repo at the cwd:\n` +
  `1. Confirm a clean tree (\`git status --short\` empty) and that plain \`make\` is green.\n` +
  `2. Create and check out the run branch \`${RUN_BRANCH}\` off the current development branch. All work (every card commit) lands here in sequence — no worktrees.\n` +
  `3. Record this branch's STARTING HEAD commit hash — it is the "prior to this branch" reference that defines which cards/mechanics already existed before the run (used for verification-skip eligibility).\n` +
  `4. Ensure docs/card_implementations/ exists.\n` +
  `Report the run branch name and its starting HEAD. Make NO other changes.`,
  { label: 'setup', phase: 'Setup' }
)

// === Phase 1: Triage — inspect ~2N cards (read-only, may fan out) ==========
phase('Triage')

let worklist = args && Array.isArray(args.cards) ? args.cards : null
if (!worklist) {
  const disc = await agent(
    `Run \`train/.venv/bin/python train/missing_cards.py --json\` from the repo root. It returns missing cards sorted by ` +
    `cross-deck frequency (most-referenced first), each with name, suggested_index, has_local_script, deck_count.\n` +
    `Take the TOP ${INSPECT} rows (2× the cap N=${N}). Pre-assign each a DISTINCT vocab index from suggested_index, de-duplicated ` +
    `so every index is distinct and < 1023; drop any that would overflow. Return ONLY a JSON array of ` +
    `{name, index, hasScript, deckCount}, preserving the frequency order.`,
    { phase: 'Triage', schema: { type: 'array', items: { type: 'object', required: ['name', 'index'],
      properties: { name: { type: 'string' }, index: { type: 'integer' }, hasScript: { type: 'boolean' }, deckCount: { type: 'integer' } } } } }
  )
  worklist = (disc || []).slice(0, INSPECT)
}
log(`inspecting ${worklist.length} card(s) (cap N=${N})`)

const triaged = (await parallel(worklist.map((c) => () =>
  agent(
    `TRIAGE one Magic card for the Robomage engine — READ ONLY, make NO edits.\n` +
    `Card: "${c.name}" (pre-assigned vocab index ${c.index}, worklist frequency ${c.deckCount != null ? c.deckCount : '?'}).\n` +
    `1. If no local script exists in bin/resources/cardsfolder/, run \`python tools/forge_fetch/fetch_script.py "${c.name}"\`. ` +
    `If it exits non-zero (NOT FOUND), return klass="defer", reason="no Forge script available".\n` +
    `2. Read the script's A:/T:/S:/K:/R: lines and their AB$/SP$/DB$ categories. Check each against ` +
    `src/parse.cpp, src/effects/effect_kind.cpp, src/effects/effect_table.cpp.\n` +
    `3. Classify:\n` +
    `   - "covered": every tag is already handled, NO engine change needed. ALSO set verifySkip=true and provenBy="<example card>" ` +
    `IF every tag is already EXERCISED by at least one card that existed in the vocab before this run branch (the mechanic is proven, ` +
    `not merely present in code). If a handler exists but no shipping card uses it, verifySkip=false.\n` +
    `   - "mechanic": needs a new handler. Set mechanicKey to a stable short name for the missing mechanic ` +
    `(e.g. "affinity", "convoke", or the unhandled AB$/SP$/DB$ category). Cards sharing a key are batched together, so name the ` +
    `*mechanic*, not the card.\n` +
    `   - "defer": no script, or behavior is irreducibly ambiguous even after consulting docs/mtg_comprehensive_rules.txt (give the reason).\n` +
    `Return {name:"${c.name}", index:${c.index}, klass, mechanicKey, verifySkip, provenBy, hasScript, deckCount:${c.deckCount != null ? c.deckCount : 0}, reason}.`,
    { label: `triage:${c.name}`, phase: 'Triage', schema: TRIAGE_SCHEMA }
  )
))).filter(Boolean)

const covered = triaged.filter((t) => t.klass === 'covered')
const deferredAtTriage = triaged.filter((t) => t.klass === 'defer')
const mechCards = triaged.filter((t) => t.klass === 'mechanic')

// group mechanic cards by key → batch units
const groups = {}
for (const t of mechCards) {
  const k = t.mechanicKey || 'misc'
  ;(groups[k] = groups[k] || []).push(t)
}

const prio = (t) => -(t.deckCount || 0) // higher frequency sorts first
// Units: each covered card is a singleton; each mechanic group is one batch.
// Order units by the worklist priority of their highest-frequency member.
const units = [
  ...covered.map((c) => ({ kind: 'covered', cards: [c], prio: prio(c) })),
  ...Object.keys(groups).map((key) => {
    const list = groups[key].slice().sort((a, b) => prio(a) - prio(b))
    return { kind: 'batch', key, cards: list, prio: Math.min(...list.map(prio)) }
  }),
].sort((a, b) => a.prio - b.prio)

log(`triage → ${covered.length} covered (${covered.filter((c) => c.verifySkip).length} verify-skip), ` +
  `${mechCards.length} mechanic in ${Object.keys(groups).length} batch(es), ${deferredAtTriage.length} deferred`)

// === Phase 2: Defer doc — record deferred + (after Implement) not-reached ==
// Written now with the triage deferrals; the not-reached (cap) list is appended in Report,
// once we know which units the cap left out. (Single committed doc; this agent creates/updates it.)
phase('Defer doc')
await agent(
  `Create or update ${DEFER_DOC} on branch ${RUN_BRANCH}. Add a dated section for this run.\n` +
  `Header (only if the file is new): "# Deferred cards — for user-directed implementation" plus a one-line note that these ` +
  `cards were triaged but not implemented and can be picked up via the interactive implement-missing-cards skill.\n` +
  `Append a section:\n` +
  `## Run ${DATE}  (cap N=${N}, inspected ${worklist.length} cards)\n` +
  `### Deferred at triage\n` +
  `one bullet per card below — "**<name>** — <reason> (Forge script: available|not found)":\n` +
  JSON.stringify(deferredAtTriage.map((t) => ({ name: t.name, reason: t.reason || 'triage deferral', hasScript: !!t.hasScript }))) + `\n` +
  `Leave a "### Not reached (cap)" subsection with a placeholder line "(filled in after implementation)"; the final report step ` +
  `will replace it with the implementable cards the N=${N} cap left out.\n` +
  `Commit ONLY this doc: "Record deferred cards (${DATE})" with Co-Authored-By + Claude-Session trailers. Do not push. Make no other changes.`,
  { label: 'defer-doc', phase: 'Defer doc' }
)

// === Phase 3: Implement — strictly sequential, up to N cards ===============
// One unit at a time on the real run-branch tree. The await chain guarantees only one subagent
// mutates the tree at a time; each runs the full implement→build→test→commit cycle to completion
// before the next unit starts. NO parallelism, NO worktrees.
phase('Implement')

const unitResults = []
const notReached = []
let committed = 0
for (const u of units) {
  if (committed >= N) { notReached.push(u); continue } // cap reached — leave the rest for the deferred doc
  const list = u.cards.map((c) => `${c.name}=${c.index}` + (c.verifySkip ? ` [verify_skip, proven by ${c.provenBy || 'existing card'}]` : '')).join(', ')
  const res = await agent(
    perUnitPrompt(u, list),
    { label: u.kind === 'batch' ? `batch:${u.key}` : `card:${u.cards[0].name}`, phase: 'Implement', schema: UNIT_RESULT_SCHEMA }
  )
  if (res) {
    unitResults.push(res)
    committed += (res.cards || []).filter((c) => c.outcome === 'done').length
  }
  log(`implemented ${committed}/${N} card(s)`)
}

// === Phase 4: Report & handoff ============================================
phase('Report')

const doneCards = unitResults.flatMap((u) => (u.cards || []).filter((c) => c.outcome === 'done')
  .map((c) => ({ name: c.name, commit: c.commit, verifySkipped: !!c.verifySkipped, needsReview: !!c.needsReview })))
const deferredDuringImpl = unitResults.flatMap((u) => (u.cards || []).filter((c) => c.outcome === 'deferred')
  .map((c) => ({ name: c.name, reason: c.reason || 'failed during implementation' })))
const notReachedCards = notReached.flatMap((u) => u.cards.map((c) => ({ name: c.name, klass: u.kind === 'batch' ? `mechanic:${u.key}` : 'covered', index: c.index })))

// Backfill the "Not reached (cap)" subsection in the deferred doc and amend its commit.
if (notReachedCards.length || deferredDuringImpl.length) {
  await agent(
    `Update ${DEFER_DOC} on branch ${RUN_BRANCH}: in this run's "## Run ${DATE}" section, replace the "### Not reached (cap)" ` +
    `placeholder with one bullet per card below ("**<name>** — <klass>, suggested index <index>"):\n` +
    JSON.stringify(notReachedCards) + `\n` +
    (deferredDuringImpl.length ? `Also append to "### Deferred at triage" (rename it "### Deferred") these cards that failed DURING implementation: ` + JSON.stringify(deferredDuringImpl) + `\n` : '') +
    `Amend the existing "Record deferred cards (${DATE})" commit (or add a follow-up commit if amending is awkward). Commit ONLY this doc. Do not push.`,
    { label: 'defer-doc:finalize', phase: 'Report' }
  )
}

return {
  cap: N,
  inspected: worklist.length,
  runBranch: RUN_BRANCH,
  triage: { covered: covered.length, verifySkip: covered.filter((c) => c.verifySkip).length,
    mechanicCards: mechCards.length, mechanicBatches: Object.keys(groups).length, deferred: deferredAtTriage.length },
  implemented: doneCards,
  implementedCount: doneCards.length,
  deferred: [
    ...deferredAtTriage.map((t) => ({ name: t.name, reason: t.reason, when: 'triage' })),
    ...deferredDuringImpl.map((c) => ({ ...c, when: 'implement' })),
  ],
  notReached: notReachedCards,
  deferredDoc: DEFER_DOC,
  handoff:
    `Deliverable is branch ${RUN_BRANCH} — ${doneCards.length} card(s) implemented, each its own commit ` +
    `(implementation was strictly sequential; one commit per card or shared-mechanic batch). Push it; do NOT open the PR ` +
    `automatically. Everything left to do is recorded in ${DEFER_DOC} (deferred at triage/impl + cards the N=${N} cap left ` +
    `unreached) for user-directed implementation via the interactive implement-missing-cards skill.`,
}

// ---------------------------------------------------------------------------
// Prompt builder (mirrors the SKILL.md Per-unit subagent prompt)
// ---------------------------------------------------------------------------
function perUnitPrompt(unit, list) {
  const isBatch = unit.kind === 'batch'
  return (
    `Implement the following into the Robomage engine, fully and correctly, IN SEQUENCE on the CURRENT branch (${RUN_BRANCH}). ` +
    `You are NOT in a worktree — do not create one. You are autonomous: NO user to ask. Finish with a CLEAN tree (commit on ` +
    `success, fully revert on defer via \`git checkout -- . && git clean -fd\`). Never leave the tree dirty or broken.\n` +
    (isBatch
      ? `Unit: a BATCH of cards sharing one new mechanic "${unit.key}": ${list}.\n` +
        `1. Fetch each script (\`fetch_script.py\` per card; a card whose fetch fails is deferred, not blocking the rest). Never hand-author or modify scripts.\n` +
        `2. Implement the mechanic "${unit.key}" ONCE as a real, general handler keyed on the tag's INTENDED MEANING — honor the actual ` +
        `SP$/AB$/DB$ category and Origin$/Destination$/ChangeType$/etc. NEVER retag to shortcut a card. Look up exact behavior in ` +
        `docs/mtg_comprehensive_rules.txt by rule number; cover the whole mechanic. If too large/cross-cutting to verify safely → defer the WHOLE batch.\n`
      : `Unit: a single COVERED card: ${list}.\n` +
        `1. Fetch the script if no local one exists (\`fetch_script.py\`; NOT FOUND → defer, reason "no Forge script available"). Never hand-author or modify scripts.\n` +
        `2. Confirm every A:/T:/S:/K:/R: tag is already handled by parse.cpp / effect_kind.cpp / effect_table.cpp. If a new mechanic is in fact ` +
        `needed: small → implement a real general handler (CR lookup; NEVER retag); large/cross-cutting → defer and note the misclassification.\n`) +
    `3. For EACH card in order: (a) append {"<name>", <index>} to src/card_vocab.h; (b) plain \`make\` — MUST be clean (this also ` +
    `regenerates train/card_costs.py via pygen); (c) TEST unless the card is flagged [verify_skip] (its mechanics are already proven by a ` +
    `pre-existing shipping card) — for verify_skip cards the clean build is sufficient, skip to (d); otherwise isolation test with ` +
    `train/test_harness.py (semantic --play specs, never --interactive; all modes/triggers) AND real-game regression ` +
    `(\`train/.venv/bin/python train/train.py observe --games 6\` with a deck that casts it; NO non-fatal errors, NO draws; clean up temp decks); ` +
    `(d) write docs/card_implementations/<uid>.md per the skill template, noting if verification was skipped and which card proves each mechanic; ` +
    `(e) COMMIT just that card — stage engine source + src/card_vocab.h + regenerated train/card_costs.py + new cardsfolder/ script (if fetched) + ` +
    `the design doc — message "Implement <name>" + trailers (Co-Authored-By, Claude-Session). Do NOT push.` +
    (isBatch ? ` The mechanic's engine edits fold into the FIRST card's commit (or a leading "Implement ${unit.key} mechanic" commit).` : '') + `\n` +
    `4. Defer protocol: a card you cannot finish with confidence is reverted and reported deferred (a wrong guess is worse than a deferral); ` +
    `${isBatch ? 'the rest of the batch still ships' : 'return outcome=deferred'}. Never leave the tree dirty between cards.\n` +
    `Return {kind:"${unit.kind}"${isBatch ? `, key:"${unit.key}", mechanic` : ''}, cards:[{name, outcome:done|deferred, index, commit, tests, verifySkipped, needsReview, reason}]}.`
  )
}
