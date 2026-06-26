export const meta = {
  name: 'autonomous-implement-cards',
  description: 'Triage missing cards, implement covered cards in parallel worktrees and shared-mechanic cards in batches, then serially integrate each onto trunk with a rebuild+smoke-test gate (one commit per card)',
  phases: [
    { title: 'Setup', detail: 'create the run branch off main' },
    { title: 'Triage', detail: 'classify each missing card: covered | mechanic:<key> | defer' },
    { title: 'Tier A', detail: 'covered cards — parallel worktrees, one card/<uid> commit each' },
    { title: 'Tier B', detail: 'mechanic-grouped cards — one mech/<key> worktree, mechanic implemented once, one commit per card' },
    { title: 'Integrate', detail: 'serial cherry-pick onto run branch + regen costs + rebuild + smoke-test gate; quarantine failures' },
    { title: 'Verify', detail: 'demonstrate-session-cards capstone over the integrated run branch' },
    { title: 'Report', detail: 'tally implemented / deferred / quarantined; PR handoff (fix or abandon)' },
  ],
}

// ---------------------------------------------------------------------------
// args: { cards?: [{name, index, hasScript}], maxCards?: number, runBranch?: string, date?: string }
//   - cards: pass the worklist explicitly (the orchestrator runs missing_cards.py --json,
//     pre-assigns distinct indices < 1023, and hands them in here). If omitted, Triage runs a
//     discovery agent to build it — but pre-assigning indices OUTSIDE the workflow is preferred
//     so the run is reproducible.
//   - date: the run date string (scripts can't call Date.now(); pass it in). Defaults to "run".
//   - runBranch: deliverable branch name off main. Defaults to `autonomous-cards/<date>`.
//     Quarantined cards land on `quarantine/<date>/<uid>`; impl work on `card/<uid>` / `mech/<key>`.
// ---------------------------------------------------------------------------
const DATE = (args && args.date) || 'run'
const RUN_BRANCH = (args && args.runBranch) || `autonomous-cards/${DATE}`
const QUARANTINE_NS = `quarantine/${DATE}`

const TRIAGE_SCHEMA = {
  type: 'object',
  required: ['name', 'index', 'klass'],
  properties: {
    name: { type: 'string' },
    index: { type: 'integer' },
    klass: { type: 'string', enum: ['covered', 'mechanic', 'defer'] },
    mechanicKey: { type: 'string', description: 'stable short name when klass=mechanic, else empty' },
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
    branch: { type: 'string' },
    commit: { type: 'string' },
    tests: { type: 'string' },
    reason: { type: 'string' },
  },
}

const GROUP_RESULT_SCHEMA = {
  type: 'object',
  required: ['key', 'branch', 'cards'],
  properties: {
    key: { type: 'string' },
    branch: { type: 'string' },
    mechanic: { type: 'string', description: 'one line: what was added + CR rules' },
    cards: { type: 'array', items: CARD_RESULT_SCHEMA },
    groupDeferred: { type: 'boolean', description: 'true if the whole group was deferred (mechanic too large)' },
    reason: { type: 'string' },
  },
}

const INTEGRATE_SCHEMA = {
  type: 'object',
  required: ['branch', 'status'],
  properties: {
    branch: { type: 'string' },
    status: { type: 'string', enum: ['integrated', 'quarantined'] },
    commits: { type: 'array', items: { type: 'string' }, description: 'commit hashes now on trunk' },
    failure: { type: 'string', description: 'one line; required when quarantined' },
  },
}

// === Phase 0: Setup — create the run branch off main ======================
phase('Setup')
await agent(
  `Preflight for an autonomous card-implementation run. On the repo at the cwd:\n` +
  `1. Confirm a clean tree (\`git status --short\` empty) and that plain \`make\` is green.\n` +
  `2. Create and check out the deliverable branch \`${RUN_BRANCH}\` off the current development branch ` +
  `(this is "trunk" for the run; all integration happens here, leaving the base branch untouched).\n` +
  `3. Ensure docs/card_implementations/ exists.\n` +
  `Report the run branch name and its HEAD (the integration base). Make NO other changes.`,
  { label: 'setup', phase: 'Setup' }
)

// === Phase 1: Triage (read-only, fan out freely) ==========================
phase('Triage')

let worklist = args && Array.isArray(args.cards) ? args.cards : null
if (!worklist) {
  const disc = await agent(
    `Run \`train/.venv/bin/python train/missing_cards.py --json\` from the repo root and return the worklist as JSON: ` +
    `an array of {name, index, hasScript}. Use each card's suggested_index, de-duplicated so every index is distinct ` +
    `and < 1023; drop any that would overflow. Return ONLY the array.`,
    { phase: 'Triage', schema: { type: 'array', items: { type: 'object', required: ['name', 'index'],
      properties: { name: { type: 'string' }, index: { type: 'integer' }, hasScript: { type: 'boolean' } } } } }
  )
  worklist = (disc || [])
}
if (args && args.maxCards) worklist = worklist.slice(0, args.maxCards)
log(`worklist: ${worklist.length} card(s)`)

const triaged = (await parallel(worklist.map((c) => () =>
  agent(
    `TRIAGE one Magic card for the Robomage engine — READ ONLY, make NO edits.\n` +
    `Card: "${c.name}" (pre-assigned vocab index ${c.index}).\n` +
    `1. If no local script exists in bin/resources/cardsfolder/, run \`python tools/forge_fetch/fetch_script.py "${c.name}"\`. ` +
    `If it exits non-zero (NOT FOUND), return klass="defer", reason="no Forge script available".\n` +
    `2. Read the script's A:/T:/S:/K:/R: lines and their AB$/SP$/DB$ categories. Check each against ` +
    `src/parse.cpp, src/effects/effect_kind.cpp, src/effects/effect_table.cpp.\n` +
    `3. Classify:\n` +
    `   - "covered": every tag is already handled, NO engine change needed.\n` +
    `   - "mechanic": needs a new handler. Set mechanicKey to a stable short name for the missing mechanic ` +
    `(e.g. "affinity", "convoke", or the unhandled AB$/SP$/DB$ category). Cards sharing a key get batched together, ` +
    `so name the *mechanic*, not the card.\n` +
    `   - "defer": no script, or behavior is irreducibly ambiguous even after consulting docs/mtg_comprehensive_rules.txt ` +
    `(give the reason).\n` +
    `Return {name:"${c.name}", index:${c.index}, klass, mechanicKey, reason}.`,
    { label: `triage:${c.name}`, phase: 'Triage', schema: TRIAGE_SCHEMA }
  )
))).filter(Boolean)

const covered = triaged.filter((t) => t.klass === 'covered')
const deferredAtTriage = triaged.filter((t) => t.klass === 'defer')
const mechCards = triaged.filter((t) => t.klass === 'mechanic')

// group mechanic cards by key
const groups = {}
for (const t of mechCards) {
  const k = t.mechanicKey || 'misc'
  ;(groups[k] = groups[k] || []).push(t)
}
const groupKeys = Object.keys(groups).sort()
log(`triage → ${covered.length} covered, ${mechCards.length} mechanic in ${groupKeys.length} group(s), ${deferredAtTriage.length} deferred`)

// === Phase 2: Tier A — covered cards, parallel worktrees ==================
phase('Tier A')

const tierA = await parallel(covered.map((c) => () =>
  agent(
    perCardPrompt(c.name, c.index),
    { label: `cardA:${c.name}`, phase: 'Tier A', schema: CARD_RESULT_SCHEMA, isolation: 'worktree' }
  )
))

// === Phase 3: Tier B — one worktree per mechanic group ====================
phase('Tier B')

const tierB = await parallel(groupKeys.map((key) => () => {
  const list = groups[key].map((t) => `${t.name}=${t.index}`).join(', ')
  return agent(
    perGroupPrompt(key, list),
    { label: `mech:${key}`, phase: 'Tier B', schema: GROUP_RESULT_SCHEMA, isolation: 'worktree' }
  )
}))

// collect every branch that produced commits, in integration order:
// Tier A first (least likely to conflict), then Tier B groups ascending by card count.
const tierABranches = (tierA || []).filter(Boolean)
  .filter((r) => r.outcome === 'done' && r.branch)
  .map((r) => ({ branch: r.branch, label: r.name, kind: 'A' }))

const tierBBranches = (tierB || []).filter(Boolean)
  .filter((g) => !g.groupDeferred && g.branch && g.cards.some((c) => c.outcome === 'done'))
  .sort((a, b) => a.cards.filter((c) => c.outcome === 'done').length - b.cards.filter((c) => c.outcome === 'done').length)
  .map((g) => ({ branch: g.branch, label: `mech:${g.key}`, kind: 'B' }))

const toIntegrate = [...tierABranches, ...tierBBranches]
log(`integrating ${toIntegrate.length} branch(es)`)

// === Phase 4: Serial integration onto trunk (one branch at a time) ========
// NOT a worktree and NOT parallel — each integrator mutates the real trunk tree;
// the sequential await chain guarantees only one touches trunk at a time, and each
// rebuilds + smoke-tests against the MERGED tree before keeping the commits.
phase('Integrate')

const integrations = []
for (const b of toIntegrate) {
  const res = await agent(
    `Integrate branch "${b.branch}" (${b.label}) onto the run branch "${RUN_BRANCH}". Work on the real working tree ` +
    `(NOT a worktree); make sure "${RUN_BRANCH}" is checked out. Steps:\n` +
    `1. \`git cherry-pick\` the branch's per-card commit(s) onto "${RUN_BRANCH}", in order.\n` +
    `2. On conflict, resolve ONLY append-only files by keeping BOTH sides: src/card_vocab.h (both vocab lines) and the ` +
    `central enum/table tails (src/effects/effect_kind.h, src/effects/effect_table.cpp, etc.). ANY other conflict, or any ` +
    `non-trivial conflict, → \`git cherry-pick --abort\` and go to the quarantine step (6-RED).\n` +
    `3. Regenerate costs: \`train/.venv/bin/python train/gen_card_costs.py\`, and amend train/card_costs.py into the ` +
    `integrated commit(s) so the derived file matches the merged vocab.\n` +
    `4. Run plain \`make\`. It MUST be clean.\n` +
    `5. Smoke-test each integrated card against the merged tree (the isolation scenario from its design doc under ` +
    `docs/card_implementations/, or \`train/.venv/bin/python train/train.py observe --games 4\` with a deck that casts it). ` +
    `Expect no non-fatal errors, no draws.\n` +
    `6. GREEN → leave the commit(s) on "${RUN_BRANCH}", return status="integrated" with the commit hashes. ` +
    `RED (conflict, build, or smoke fails) → \`git reset --hard\` to the pre-pick HEAD so "${RUN_BRANCH}" stays green, then ` +
    `QUARANTINE: preserve the work off the run branch by renaming its impl branch — ` +
    `\`git branch -m ${b.branch} ${QUARANTINE_NS}/<uid>\` (use the card's uid; for a group, one quarantine branch is fine). ` +
    `Return status="quarantined", branch="${QUARANTINE_NS}/<uid>", with the one-line failure.\n` +
    `"${RUN_BRANCH}" must be clean and building when you return either way. Never hand-merge blind.`,
    { label: `integrate:${b.label}`, phase: 'Integrate', schema: INTEGRATE_SCHEMA }
  )
  if (res) integrations.push(res)
}

const integrated = integrations.filter((i) => i.status === 'integrated')
let quarantined = integrations.filter((i) => i.status === 'quarantined')

// === Phase 5: Verify — demonstrate-session-cards capstone =================
// Runs once over the fully-integrated run branch: proves every shipped card is actually
// demonstrated in real games (engine-sanity-check), backfilling targeted test_harness
// scenarios, and reviews for engine bugs. Findings are resolved per the autonomous policy:
// clear+small fix on the run branch, else quarantine that card off it.
phase('Verify')
let verify = null
if (integrated.length) {
  verify = await agent(
    `Run the whole-batch verification capstone over the integrated run branch "${RUN_BRANCH}" (it must be checked out).\n` +
    `Invoke the \`demonstrate-session-cards\` skill: enumerate this run's new cards from the committed design docs under ` +
    `docs/card_implementations/, build temp decks containing them, run the engine-sanity-check scripted games, PROVE every ` +
    `new card is actually demonstrated (cast + its defining clause resolved) — backfilling any gap with a targeted ` +
    `test_harness.py scenario — and review the verbose output for engine bugs. Give the deepest scrutiny to cards flagged ` +
    `NEEDS_REVIEW during implementation.\n` +
    `No user is in the loop, so RESOLVE each finding: a genuine engine bug whose fix is clear and small → fix it on ` +
    `"${RUN_BRANCH}" as its own follow-up commit and re-verify; otherwise QUARANTINE that card — ` +
    `\`git branch ${QUARANTINE_NS}/<uid> ${RUN_BRANCH}\` to snapshot its work, then revert/reset its commit(s) off ` +
    `"${RUN_BRANCH}" so the branch stays green. Scripted-agent suboptimality (not an engine bug) is noted, not acted on. ` +
    `"${RUN_BRANCH}" must build clean and have a clean tree when you return.\n` +
    `Return JSON: {fixed:[{card,commit}], newlyQuarantined:[{branch,failure}], notedNonBugs:[string], summary:string}.`,
    { label: 'verify:capstone', phase: 'Verify', schema: {
      type: 'object', required: ['summary'],
      properties: {
        fixed: { type: 'array', items: { type: 'object', properties: { card: { type: 'string' }, commit: { type: 'string' } } } },
        newlyQuarantined: { type: 'array', items: { type: 'object', properties: { branch: { type: 'string' }, failure: { type: 'string' } } } },
        notedNonBugs: { type: 'array', items: { type: 'string' } },
        summary: { type: 'string' },
      } } }
  )
  if (verify && Array.isArray(verify.newlyQuarantined)) {
    quarantined = [...quarantined, ...verify.newlyQuarantined.map((q) => ({ branch: q.branch, failure: q.failure }))]
  }
}

// === Phase 6: Report & handoff ============================================
phase('Report')

return {
  triage: { covered: covered.length, mechanicGroups: groupKeys.length, mechanicCards: mechCards.length },
  runBranch: RUN_BRANCH,
  deferred: [
    ...deferredAtTriage.map((t) => ({ name: t.name, reason: t.reason })),
    ...(tierA || []).filter(Boolean).filter((r) => r.outcome === 'deferred').map((r) => ({ name: r.name, reason: r.reason })),
    ...(tierB || []).filter(Boolean).flatMap((g) =>
      g.groupDeferred
        ? [{ name: `group:${g.key}`, reason: g.reason }]
        : (g.cards || []).filter((c) => c.outcome === 'deferred').map((c) => ({ name: c.name, reason: c.reason }))),
  ],
  integratedBranches: integrated.map((i) => ({ branch: i.branch, commits: i.commits })),
  verify: verify && { fixed: verify.fixed || [], notedNonBugs: verify.notedNonBugs || [], summary: verify.summary },
  quarantined,
  handoff:
    `Deliverable is branch ${RUN_BRANCH} (push it; do NOT open the PR automatically). ` +
    `Quarantined cards live on ${QUARANTINE_NS}/<uid> — per card a human can FIX (checkout, repair, re-integrate ` +
    `through the Phase 4 gate, delete the quarantine branch, then open the PR) or ABANDON (delete the branch; the PR is ` +
    `unaffected since it was never on the run branch).`,
}

// ---------------------------------------------------------------------------
// Prompt builders (mirror the SKILL.md per-card / per-group prompts)
// ---------------------------------------------------------------------------
function perCardPrompt(name, index) {
  return (
    `Implement the single Magic card "${name}" into the Robomage engine, fully and correctly, then commit it on its own ` +
    `branch. You are autonomous in a git worktree branched from "${RUN_BRANCH}" — NO user to ask. Name your branch ` +
    `card/<uid> (uid = lowercased name, spaces→_, apostrophes dropped). Finish with a clean tree (committed on success, ` +
    `reverted on defer).\n` +
    `1. Script: if no local cardsfolder/ script, run \`python tools/forge_fetch/fetch_script.py "${name}"\`. NOT FOUND ` +
    `(non-zero) → defer, reason "no Forge script available". NEVER hand-author or modify a script.\n` +
    `2. This card was triaged "covered" (no engine change). Confirm every A:/T:/S:/K:/R: tag is handled by parse.cpp / ` +
    `effect_kind.cpp / effect_table.cpp. If a new mechanic IS actually needed and it's large/cross-cutting, defer and note ` +
    `the misclassification; if small, implement a real general handler (CR rule lookup in docs/mtg_comprehensive_rules.txt) ` +
    `— NEVER retag a category/Origin/Destination.\n` +
    `3. Register: append {"${name}", ${index}} to src/card_vocab.h. Do NOT run gen_card_costs.py; do NOT stage ` +
    `train/card_costs.py (the integrator regenerates it).\n` +
    `4. Build: plain \`make\`, clean. Fix any new error.\n` +
    `5. Isolation test with train/test_harness.py (semantic --play specs; never --interactive); cover all modes/triggers.\n` +
    `6. Real-game regression: \`train/.venv/bin/python train/train.py observe --games 6\` with a deck that casts it; expect ` +
    `no non-fatal errors, no draws. Clean up any temp deck.\n` +
    `7. Document docs/card_implementations/<uid>.md (uid = lowercased, spaces→_, apostrophes dropped) per the skill template.\n` +
    `8. Commit "Implement ${name}" (Co-Authored-By + Claude-Session trailers), staging engine source + src/card_vocab.h + ` +
    `the new cardsfolder/ script (if fetched) + the design doc — NOT card_costs.py. Do NOT push.\n` +
    `9. Defer protocol: if not confident, \`git checkout -- . && git clean -fd\` to a clean tree and return outcome=deferred.\n` +
    `Return {name, outcome, index, branch, commit, tests, reason}.`
  )
}

function perGroupPrompt(key, list) {
  return (
    `Several cards share one missing mechanic "${key}": ${list}. Implement the mechanic ONCE as a real, general handler, ` +
    `then implement EACH card, committing EACH card separately on this branch. Autonomous in a git worktree branched from ` +
    `"${RUN_BRANCH}" — NO user. Name your branch mech/${key}. Finish with a clean tree.\n` +
    `1. Fetch every script (\`fetch_script.py\` per card; a card whose fetch fails is deferred, not blocking the rest). ` +
    `Never hand-author or modify scripts.\n` +
    `2. Implement the mechanic generally, keyed on the tag's intended meaning — honor the actual SP$/AB$/DB$ category and ` +
    `Origin$/Destination$/ChangeType$/etc. NEVER retag to shortcut a card. Look up exact behavior in ` +
    `docs/mtg_comprehensive_rules.txt by rule number; cover the whole mechanic. If it's too large/cross-cutting to verify ` +
    `safely here, defer the WHOLE group (groupDeferred=true, clean tree).\n` +
    `3. For each card in order: append {"<name>", <index>} to src/card_vocab.h (do NOT stage card_costs.py); plain \`make\` ` +
    `clean; isolation test (test_harness.py, all modes/triggers); real-game regression (observe --games 6, no non-fatal ` +
    `errors / no draws); write docs/card_implementations/<uid>.md; then commit JUST that card ("Implement <name>", ` +
    `trailers). The mechanic's engine edits fold into the FIRST card's commit (or a leading "Implement ${key} mechanic" ` +
    `commit). A card that fails to verify is reverted out of staging and reported deferred; the rest still ship.\n` +
    `Return {key:"${key}", branch, mechanic, groupDeferred, reason, cards:[{name, outcome, index, commit, tests, reason}]}.`
  )
}
