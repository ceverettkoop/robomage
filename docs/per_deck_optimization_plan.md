# Archetype Pools, Bucketed Value Heads, League Exploiters, and a Curriculum Interface

## Context

The single generalist (`gen`) currently trains only on "fair" midrange-ish decks. To extend it to
strategically alien archetypes (burn, draw-go control, Doomsday combo, big mana) without capacity
collapse, we implement Stage 1 (matchup-aware value function: per-archetype-bucket value heads,
with optional PopArt normalization) and Stage 2 (AlphaStar-style exploiter runs feeding the PFSP
league), plus a **unified curriculum interface**: design, track, run, and resume multi-phase
PPO + AZ training plans from the Textual control panel instead of ad-hoc sessions.

User decisions already made:
- **Archetypes (6 + unknown):** `tempo` (league/ur_delver, league/wrb_energy, league/gw_maverick,
  delver, mav), `slow_midrange` (league/bug, league/bw_dnt), `control` (new decks, user will
  provide lists), `burn` (new decks, user will provide lists), `big_mana` (tron + future),
  `doomsday` (doomsday, car_doomsday). `unknown` is the fallback bucket for unmapped decks.
- New decklists and their **20–30 missing cards** come from user-made lists (separate track,
  via the implement-missing-cards skill).

Training restarts from scratch are acceptable (per standing feedback; OBS layout will change).

---

## Part 0 — Archetype metadata foundation

**New data file** `bin/resources/decks/archetypes.json`: `{ "<deck stem>": "<archetype>" }`
(stems relative to `decks/`, e.g. `"league/ur_delver": "tempo"`). Seed it with the mapping above.

**New module** `train/archetypes.py` (stdlib-only so `cli_spec`/TUI can import it):
- `ARCHETYPES = ["tempo", "slow_midrange", "control", "burn", "big_mana", "doomsday"]` +
  `UNKNOWN = len(ARCHETYPES)`; `N_ARCH = 7`; `N_VALUE_BUCKETS = N_ARCH * N_ARCH` (49).
- `arch_index(deck: str) -> int` — exact stem match against the JSON; league decks that are
  unmapped **error loudly** at league startup (roster validation in `train.py:league()` where the
  roster is listed, ~`train/train.py:1300`); ad-hoc decks fall back to `UNKNOWN`.
- `bucket_index(self_deck, opp_deck) -> int` = `arch_index(self)*N_ARCH + arch_index(opp)`.

**Deck promotions now** (independent of user lists): move `temp/tron.dk` → `league/tron.dk` and
`temp/car_doomsday.dk` → `league/car_doomsday.dk` with their `temp/sb_*.dk` sideboards renamed to
match the league sideboard convention (matches what `docs/distributed_league_training.md` already
anticipates). Update the SessionStart provisioning hook deck list if needed.

---

## Part 1 — Per-archetype-bucket value heads (PPO + AZ)

**AS BUILT (supersedes the counts below).** Part 0 shipped 7 archetypes + `unknown`,
so `N_ARCH = 8` and `N_VALUE_BUCKETS = 64` (not 7/49), and the obs tail is
1 bucket float + one-hot(8) + one-hot(8) = **17 floats** (not 15):
`OBS_SIZE` 6986 → **7003**. The tail is written by the single helper
`env.write_matchup_tail()` from `RoboMageEnv._parse_bquery_payload`, keyed on the
state vector's own `self_is_a` flag so the bucket is perspective-relative like the
rest of the observation. The C++ AZ actor mirrors it in
`src/actor/obs_builder.cpp` off the generated `src/gen/archetypes_gen.h`
(`train/gen_archetypes.py`, wired into `make pygen`), keeping
`train/archetypes.py` + `decks/archetypes.json` the one source of truth.

### How it works (mechanics)

**Bucket delivery — through the observation tail (Python side only, no C++ change).**
The env already knows both decks per episode (`RoboMageEnv.__init__ deck_a/deck_b`,
`ModelVsScriptedEnv.reset` at `train/env.py:1280` samples `(self_deck, opp_deck, …)` from the
`LeaguePool`). Append to the obs, after the existing cost-feature tail:
- 1 float: raw bucket index (consumed by the policy, stripped before the trunk), and
- 14 floats: one-hot(self archetype 7) + one-hot(opp archetype 7) — cheap explicit conditioning
  that *does* feed the trunk alongside the existing deck-identity blocks.

`OBS_SIZE` grows by 15. All obs-assembly paths go through one helper so the search/analysis
paths (`search_env.py` mirrors, AZ self-play) get the bucket for free. Update the
`env.py` `OBS_SIZE` computation, the extractor mirror assert, and note that
`test_obs_invariants.py` tests the raw STATE vector (unaffected — `STATE_SIZE` unchanged).

**Multi-head critic — PPO** (`train/extractor.py`):
- `CardGameExtractor.forward` slices the bucket scalar off (same pattern as the `per_action`
  tail slice at `extractor.py:582`) and stashes it as a side-channel tensor on the extractor;
  the archetype one-hots stay in the base feature cat.
- `PerActionMaskablePolicy._build` (`extractor.py:620`): replace the stock
  `self.value_net = Linear(latent_vf, 1)` with `Linear(latent_vf, N_VALUE_BUCKETS)` (49 columns,
  ~25k params — trivial). In `forward` (:652), `evaluate_actions` (:662), and `predict_values`,
  gather the column by the stashed bucket index. Everything upstream (GAE, value loss) is
  unchanged — SB3 just sees a scalar value per sample, but each (self-arch × opp-arch) matchup
  trains its own final linear head, so a burn race's value statistics stop fighting a control
  grind's in the last layer.
- Bucket is constant within an episode, varies across envs/episodes — the rollout buffer needs
  no changes since the bucket rides inside the obs.

**Multi-head critic — AZ** (`train/az_net.py`):
- `AZNet.value_head` (:389) becomes `Linear(latent, N_VALUE_BUCKETS)`; `forward` (:391) gathers
  by the bucket slice of the obs, then `tanh`. AZ's targets are already bounded [-1,1], so
  normalization matters less there; the head split still separates matchup statistics.
- `AZNet.from_ppo` (:552): map the PPO multi-head `value_net.weight/bias` 1:1 (shapes now match).

**Per-bucket PopArt normalization — follow-on, behind a flag (`--popart`).**
Multi-head alone fixes last-layer interference; PopArt additionally fixes *gradient-scale*
interference in the shared torso. Implementation (Hessel et al. 2019): per-bucket running
(μ_b, σ_b) of returns; the head predicts normalized values; on every stats update, rescale that
bucket's head column (`w ← w·σ_old/σ_new`, `b ← (b·σ_old + μ_old − μ_new)/σ_new`) so outputs are
preserved. Requires a thin `MaskablePPO` subclass (unnormalize values for GAE/advantages in the
rollout collection, normalize returns in the value loss, update stats per batch). This is the one
genuinely invasive piece — ship it as a separate commit after the multi-head lands and is
validated, defaulting OFF until compared on a short league run.

**Diagnostics:** extend `PFSPCallback` (`train/train.py:177`) — it already tracks per-matchup
win rates — to also log per-bucket value loss / explained variance so we can see the heads
specialize; add an archetype-grouped view to the `baseline --all` matrix report.

**Files:** `train/archetypes.py` (new), `train/env.py`, `train/extractor.py`, `train/az_net.py`,
`train/train.py` (validation + logging), `train/cli_spec.py` (`--popart` flag later).

---

## Part 2 — League exploiters

**AS BUILT.** Shipped as described below, with these specifics:
`train.py exploiter --archetype <arch> [--steps N] [--chunk-steps N] [--decks …]
[--fresh] [--resume]`. The learner pilots every deck tagged with `<arch>` in
`archetypes.json` (per-episode cycling, league "mixed" mode); the opponent pool is
**pinned** to the frozen `gen__final.zip` (else the newest `gen__v*`) piloting the whole
league roster — `LeaguePool(pinned_snapshots=…)` replaces snapshot discovery, drops the
latest-self slot, and is deliberately *unsharded* so every env process can face every
roster deck. PFSP still weights across the frozen opponent's decks. Checkpoints are
written **only** under `exp_<arch>` (`_league_chunk` gained a `stem` parameter; the
generalist's files are never touched), warm-started from `gen` with a **fresh step
counter** (so exploiter snapshot versions count exploiter steps and the LR/shaping
schedules restart), and an incompatible-layout donor fails with an actionable message
(`_load_warm_start`). Progress lives in
`checkpoints/_exploiter_<arch>_progress.json` via the shared atomic
`_write_progress_state`. Pool integration: `LeaguePool.refresh` tier 3 guarantees each
archetype's newest exploiter paired with that archetype's roster decks (never evicted),
older `exp_*__v*` interleave with older `gen__v*` in the discretionary budget, and
`--exploiter-floor` (default 0.1) reserves a share of episodes for exploiter entries on
top of their normal PFSP weight. `exp_*` joined `gen` as a reserved deck-name stem.
The optional BC **kickstart** (item 4 below) was NOT implemented — it is a separate
supervised-learning path (obs/action collection through the gym env + a masked
cross-entropy pretrain of the policy head) with its own failure modes, and was left
out rather than half-landed; `exploiter` has no `--kickstart` flag yet.

**Concept:** dedicated runs whose learner pilots one archetype's decks against the *frozen*
current `gen`, saved under their own stem, then injected into the PFSP opponent pool — so the
generalist gets inoculated against burn/combo/control without having to discover those styles.

1. **Checkpoint category.** Stem `exp_<archetype>` (e.g. `exp_burn__v1500000.zip`,
   `exp_burn__final.zip`) in `checkpoints/`. The existing `_SNAPSHOT_RE` (`opponents.py:51`)
   already parses `<stem>__v<steps>`; add `exp_*` to the reserved-stem guard
   (`assert_not_reserved_deck`, `opponents.py:60`).

2. **New `train.py exploiter` subcommand** (new `Sub` in `cli_spec.py` + dispatch branch):
   `exploiter --archetype burn [--steps N] [--fresh]`. A league-like run reusing `_league_chunk`
   (`train/train.py:1091`) with: learner decks = the archetype's decks from `archetypes.json`;
   opponent pool pinned to the frozen `gen__final.zip` piloting the full roster (main-exploiter
   configuration — no snapshot rotation, no self-play branch); warm-start from `gen__final`
   unless `--fresh`; saves only under `exp_<archetype>` (never touches `gen`). Its own progress
   sidecar `checkpoints/_exploiter_<archetype>_progress.json` using the existing atomic-write
   helpers (`_write_league_state` pattern, `train/train.py:757-786`) so it is `--resume`-able.

3. **Pool integration** (`train/opponents.py`):
   - `gen_snapshots()`/`LeaguePool.refresh()` (:1424): third tier — for each archetype with
     exploiter checkpoints, guarantee the newest `exp_<arch>__final` (paired with that
     archetype's decks) as a never-evicted anchor; older `exp_*__v*` join the discretionary
     round-robin.
   - `sample_episode()` (:1524): exploiter entries participate in the normal PFSP weighting
     (`_entry_weights`) — no separate fixed fraction needed; PFSP's `(1-wr)^p` automatically
     concentrates play on the exploiters the learner loses to. Add `--exploiter-floor`
     (default ~0.1, like the scripted anchor) to guarantee minimum exposure even after the
     learner starts beating them.
   - `PFSPCallback` win-rate tracking already keys on `(self_deck, opp_deck, label)` — exploiter
     labels flow through unchanged.

4. **Combo kickstart (optional sub-step, after the above works).** For the `doomsday` exploiter,
   PPO exploration rarely completes the combo line. Add a small BC pretrain utility: run
   scripted-vs-scripted Doomsday games via `runner.run_match`, collect (obs, action) pairs
   (the scripted agent's Doomsday lines in `train/scripted_agent.py` are the teacher), and
   supervised-train the policy head for a few epochs before the RL phase. Ship as
   `exploiter --kickstart scripted`.

**Files:** `train/train.py`, `train/opponents.py`, `train/cli_spec.py`; later a small
`train/bc_kickstart.py`.

---

## Part 3 — Unified curriculum interface (design / track / run / resume, PPO + AZ, Textual)

**AS BUILT.** Shipped as described below. Specifics worth knowing:
`train/curriculum.py` is stdlib-only (imports `cli_spec` + the new
`train/progress_io.py`), so the Textual launcher and the plan/progress
machinery never pull in torch. Part 2's `_write/_read_progress_state` moved
into `progress_io.py` as the single home for the crash-safe sidecar writer
(train.py re-imports them under their old private names; `az_train.py`'s
hand-rolled duplicate now calls it too). A phase's top-level fields are the
small **alias** table `PHASE_FIELDS` (`decks`/`steps`/`archetype`/`rotations`/
`games`/`model`/`deck`) onto that subcommand's arg dests; everything else is an
`overrides` key, validated against `cli_spec` — unknown kind, unknown override,
wrong value type, a set-at-once mutex pair, or the runner-owned `resume` all
fail loudly at load. `decks: "all"` means "the subcommand's own default roster"
(the flag is omitted). Resume relaunches only a phase left in status `running`,
and for a resumable kind emits just the run-SELECTING flags plus `--resume`
(`--archetype` for an exploiter, `--shard` for a sharded league) since those
drivers restore everything else from their own sidecar; `done`/`running` phases
are hash-immutable while `pending`/`failed` ones may be rewritten. SIGINT
(child rc 130/-2) leaves the phase `running` and exits 130; any other nonzero
exit marks it `failed` and stops. Verified by `train/test_curriculum.py` (new
`make check` tier `curriculum`) plus a live interrupted-and-resumed 2-phase toy
plan in a sandboxed checkpoint dir. The TUI screen reuses the launcher's form
machinery through a new `ArgFormMixin` (extracted from `LauncherApp`, no
behavior change) and turns the az-form random-seed convenience OFF for plan
editing.

### Plan + progress files
- **Plan definition** `train/checkpoints/curricula/<name>.plan.json` (versioned):
  ```json
  {
    "version": 1, "name": "q3_archetypes",
    "phases": [
      {"kind": "league",    "decks": ["league/ur_delver", "..."], "steps": 2000000,
       "overrides": {"self_play_frac": 0.2, "promote_margin": 0.05}},
      {"kind": "exploiter", "archetype": "burn", "steps": 500000},
      {"kind": "league",    "decks": "all", "steps": 1000000},
      {"kind": "az-league", "rotations": 1, "overrides": {"games": 400}},
      {"kind": "baseline",  "games": 100}
    ]
  }
  ```
  Phase kinds map 1:1 onto existing `train.py` subcommands (`league`, `exploiter`, `az`,
  `az-league`, `baseline`); `overrides` keys are that subcommand's `cli_spec` arg dests — the
  spec IS the schema, so no second source of truth.
- **Progress sidecar** `<name>.progress.json` (same atomic-write versioned pattern as
  `_league_progress.json`): `{plan_hash, phase_index, phases: [{status: pending|running|done|failed,
  started, finished, exit_code, summary}]}`. Refuses to resume if the plan file changed for
  already-completed phases (hash prefix check), allows edits to future phases.

### Runner — `train.py curriculum` (new module `train/curriculum.py`)
- `curriculum --plan <file> [--resume] [--status] [--dry-run]`.
- Executes phases **as subprocesses** (`sys.executable train/train.py <sub> <argv…>`), composed
  from the plan via the `cli_spec` arg definitions. Subprocess-per-phase gives memory isolation
  between PPO and AZ phases, crash containment, and reuses each phase's own resume machinery.
- Resume semantics: a phase whose status is `running` (crashed mid-way) is relaunched with
  `--resume` appended when the subcommand supports it (`league`, `az-league`, `exploiter`);
  non-resumable kinds (`az`, `baseline`) restart from the phase start (checkpoints accumulate,
  so this is safe). On phase completion the runner records `done` before advancing, so a stale
  per-phase sidecar from an earlier phase is never misread.
- `--dry-run` prints each phase's composed argv without running — this is also the unit-test
  surface.
- Budget note: `league` phases pass their `steps` as `--total-timesteps`; the existing league
  sidecar handles intra-phase rotations exactly as today.

### TUI — new `CurriculumScreen` in `train/tui.py`
The launcher stays a thin front end; the screen composes/edits JSON plans and launches the
runner through the existing terminal-suspend path.
- New `Tool`/`Sub` entry `curriculum` in `cli_spec.py` so it appears in the command tree; unlike
  the flat forms, selecting it pushes a Textual `Screen` (first use of `push_screen` in this app).
- **Plan builder:** left pane = phase list (`ListView`/`DataTable`) with add / remove / reorder
  (reuse the `[`/`]` reorder pattern from the league roster list, `tui.py:423-427`) and a phase-kind
  picker; right pane = the selected phase's form, **generated by the existing `_build_arg`
  machinery from that kind's `cli_spec` Sub items** (the big reuse win — league phases get the
  same deck multipick, AZ phases the same knobs). Save/Load plan files from
  `checkpoints/curricula/`.
- **Status view:** reads `<name>.progress.json` + the per-phase sidecars to show each phase's
  state, steps done, and last gate/baseline results.
- **Run / Resume buttons:** compose `train.py curriculum --plan … [--resume]` and hand it to
  `_run_in_terminal` (`tui.py:586`) exactly like every other command — output streams in the
  suspended terminal and is teed to the rolling command log.

**Files:** `train/curriculum.py` (new), `train/cli_spec.py`, `train/train.py` (dispatch),
`train/tui.py` (screen).

---

## Part 4 — New decks & cards (user-provided; parallel track)

Blocked on the user's decklists for **control**, **burn**, and any extra **big_mana** decks
(20–30 new cards expected). Per list, the workflow is:
1. Drop the `.dk` into `bin/resources/decks/league/` (+ sideboard file), tag it in
   `archetypes.json`.
2. `train/missing_cards.py` to enumerate missing cards; implement them via the
   **implement-missing-cards** skill (vocab entry + `make` regenerates `card_costs.py`),
   `make check` gating each unit.
3. Fetch scripts via `tools/forge_fetch/fetch_script.py`; update the SessionStart hook token map
   if a new engine-synthesized token appears.

---

## Ordering

1. Part 0 (archetype metadata + tron/car_doomsday promotion) — small, unblocks everything.
2. Part 1 multi-head value heads (PPO + AZ + obs tail) — checkpoint-breaking; land while the
   from-scratch retrain is pending anyway. PopArt as a separate follow-on commit, default off.
3. Part 2 exploiter subcommand + pool integration (kickstart last, optional).
4. Part 3 curriculum runner, then the TUI screen.
5. Part 4 whenever the decklists arrive (independent; only `archetypes.json` couples them).

## Verification

- `make check` after every part (obs-invariants tier tests the raw STATE vector — unaffected;
  env/extractor asserts catch OBS_SIZE mismatches).
- Part 1: short league smoke (tiny `--total-timesteps`) on the 7-deck roster; log per-bucket
  value loss to confirm every bucket that appears receives gradients; `baseline gen --deck …`
  still loads and runs; AZ `from_ppo` warm-start smoke.
- Part 2: `exploiter --archetype tempo --steps 20000` smoke — verify it never writes a `gen`
  file, its sidecar resumes, and a subsequent league run's `[pfsp]` logs show `exp_tempo`
  entries being sampled.
- Part 3: `curriculum --dry-run` argv-composition unit test; a live 2-phase toy plan
  (league 20k steps → baseline 4 games), interrupted with Ctrl-C mid-phase-1 and resumed with
  `--resume`, completes with correct progress JSON. TUI screen: build the toy plan in the
  screen, save, reload, run.
