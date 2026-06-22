# Plan: embed_dim bump + PFSP/softmax sampler + multi-matchup league training

Goal: move from **per-matchup specialist** models (`{model}_{opp}_final.zip`, one
network per A-vs-B pairing) to **one model per deck** that pilots that deck well
against the *entire field*, trained with a self-managing league: a shared pool of
frozen snapshots across all decks, sampled by **prioritized fictitious self-play
(PFSP)** / OpenAI-Five-style softmax quality scores, with a permanent scripted
anchor. Also increase the policy network's `embed_dim`.

Incorporates the research (`docs/opponent_selection_research.md`): OpenAI Five
80/20 latest-vs-pool split + softmax quality scores; AlphaStar PFSP `f(x)=(1-x)^p`;
SIMPLE-style promotion-on-margin; keep a scripted anchor to prevent collapse;
skip the full main/exploiter league (compute-impractical at this scale).

## Architectural shift

A checkpoint is now identified by the **single deck it pilots**, not a matchup.
An opponent for learner-deck D is drawn from a pool spanning:

1. **Scripted anchor** piloting a sampled opponent deck (always present — collapse
   guard, per research medium-confidence inference + diversity rationale).
2. **Frozen snapshots of other decks' models** (cross-deck league play).
3. **Frozen snapshots of D's own model** (mirror self-play).
4. **The latest snapshot of D** (the OpenAI Five "play the latest self" slot for
   fast learning).

Two coupled sampling decisions per episode — which opponent **deck**, and which
**controller** for it — both driven by win-rate so the agent faces what it is
currently losing to.

**Co-training strategy: rotating single learner (recommended).** Training all
decks simultaneously (N PPO learners) is heavy and complex. Instead, mirror
AlphaStar's "main agent vs league": **one learner at a time**, trained for a chunk
of steps against the shared frozen pool, dropping new snapshots into the pool,
then **rotate** to the next deck. This reuses existing single-learner infra
(`SubprocVecEnv` + one `MaskablePPO`) and the `alternate` subcommand's spirit,
generalized to N decks and a persistent pool. This is the "self-generated
structure" — the league driver manages snapshots, promotion, and matchup sampling
automatically.

## Checkpoint naming migration

Current convention (≈9 sites, all assume matchup naming):
`{model_deck}_{opp_deck}_*.zip` — save `train.py:547`; resolve `train.py:427-448`;
SelfPlayEnv mirror glob `env.py:1010`; OpponentPool `matchup_checkpoints`
`opponents.py:50` + `_expand_matchup_tokens` `opponents.py:196-210`; plus
`train.py:549,627-628,631,637,710,1087`.

New convention — **pilot deck only**:
- Periodic: `{deck}__v{global_steps}.zip`  (double underscore separates deck from
  version so deck names with underscores still parse)
- Final: `{deck}__final.zip`
- "All snapshots piloting deck E" → glob `{E}__v*.zip` + `{E}__final.zip`.

Because the embed_dim change *already* invalidates old checkpoints, do a **clean
break** — no back-compat shim. Add a one-line `CHECKPOINT_FORMAT_VERSION` constant
and skip files that don't match, with a warning.

## Phased implementation

### Phase 0 — `embed_dim` knob (independent, ship first)
- `extractor.py` already computes `features_dim` from `embed_dim` (default 64,
  `extractor.py:131`); no change needed there.
- `train.py:542` policy_kwargs: add
  `features_extractor_kwargs=dict(embed_dim=args.embed_dim)`.
- `cli_spec.py`: add `--embed-dim` (int, default 128) to the train/league subs.
- **Persist `embed_dim` in checkpoint metadata** (or a sidecar `.json`) so load
  reconstructs the matching network; SB3 stores `policy_kwargs` in the zip, so
  loading via `MaskablePPO.load` recovers it — verify and rely on that.
- Note: `OBS_SIZE` is unchanged by embed_dim; only network shapes change → old
  checkpoints incompatible (expected, clean break).

### Phase 1 — checkpoint naming + resolver
- Replace `model_prefix = f"{model_deck}_{opp_deck}"` (`train.py:547`) with
  `f"{deck}_"` style → SB3 emits `{deck}__v{steps}.zip` / `{deck}__final.zip`.
- Rewrite `_resolve_model` (`train.py:427-448`) to expand `{deck}` → newest
  `{deck}__v*.zip` (or `__final`).
- New helper `deck_snapshots(deck, checkpoint_dir) -> list[path]` in
  `opponents.py`, replacing `matchup_checkpoints` (`opponents.py:38-51`).
- Update SelfPlayEnv glob (`env.py:1010`) to `{opp_deck}__v*.zip`.

### Phase 2 — per-episode opponent-deck sampling (env)
- The deck-swap already swaps per episode (`env.py:626-627`); extend so
  `opp_deck` is **sampled from a roster** each episode rather than fixed.
- `ModelVsScriptedEnv` (and SelfPlayEnv) gain an `opp_deck_sampler` that, per
  reset, picks `(opp_deck, controller)` from the pool. Wire `opp_deck` into the
  deck-file selection that already feeds `--deck-a/--deck-b` (`env.py:204-207`).
- Deck roster comes from the filesystem listing already used by `sweep`
  (`train.py:1066-1067`) and the TUI (`tui.py:33-40`): currently
  `delver, doomsday, mav`.

### Phase 3 — PFSP / softmax sampler (`OpponentPool`)
Extend `OpponentPool` (`opponents.py:142-229`) — its `sample()` (`:220-224`) and
sharding (`:175-179`) stay, the weights become dynamic:

- **Per-opponent quality scores.** Maintain `q_i` per pool entry keyed by
  `(opp_deck, snapshot_id | "scripted")`. Two interchangeable weightings:
  - OpenAI Five softmax: `p_i ∝ exp(q_i)`; on a win vs i, `q_i -= η/(N·p_i)`
    (η=0.01); on a loss, no update. (`docs/opponent_selection_research.md`,
    finding on OpenAI Five Appendix N.)
  - AlphaStar PFSP: `p_i ∝ f(x_i) = (1 - winrate_i)^p`, p≈2.
- **80/20 latest/pool split.** With prob `--self-play-frac` (default 0.8) pick the
  **latest snapshot of the learner's own deck** (fast learning); else sample the
  historical pool by quality. Keep a fixed minimum scripted-anchor probability
  (e.g. 0.1 carved out of the pool share) so the anchor never vanishes.
- **Controller interface is already uniform** — `ScriptedController` /
  `ModelController` both expose `.choose(obs, num_choices, action_masks)`
  (`opponents.py:61-88`), so new entries slot in without touching the env.

### Phase 4 — win-rate feedback loop (callback)
- Generalize `WinTallyCallback` (`train.py:238-274`) into a `PFSPCallback`:
  read each finished episode's `info["episode"]["r"]` and
  `info["game_meta"]` (`opp_deck`, `opp_type`, `model_is_a` — set at
  `env.py:638-642,770`), attribute the result to the exact opponent entry, and
  update its `q_i`/win-rate.
- Push updated weights into the subproc envs with
  `vec_env.set_attr("opponent_weights", new_weights)` — same mechanism
  `ShapingScaleCallback` already uses (`train.py:308`,
  `set_attr("shaping_scale", ...)`). `OpponentPool.sample` reads the latest
  pushed weights. (Cross-process: each subproc keeps its own sharded entries; the
  broadcast weights are the global win-rate estimate, applied to whatever subset
  that env holds — accept this as an approximation, documented.)
- Log a **matchup win-rate matrix** each rollout for visibility (replaces/extends
  `--tally`).

### Phase 5 — snapshot creation + promotion (callback)
- Every `--snapshot-every` steps (default ≈ OpenAI Five's "every 10 iterations"),
  save `{deck}__v{steps}.zip` and register it into the pool with `q` initialized
  to the **current max** quality (OpenAI Five rule), so fresh snapshots get tried.
- Optional **SIMPLE-style promotion gate**: only add a snapshot to the *opponent*
  pool when it beats the current best of its deck by a margin (default 0.2
  win-rate) — keeps the pool of genuinely distinct, stronger snapshots instead of
  near-duplicates. Use the PFSPCallback's tracked win-rates to evaluate the gate.
- **Bound pool memory** with the existing `max_checkpoint_ratio` sharding
  (`opponents.py:175-179`), now sharding across the `(deck × snapshot)` space.
  Log what was dropped (no silent truncation).

### Phase 6 — league driver + CLI
- New subcommand **`league`** in `cli_spec.py` (single source of truth → `tui.py`
  picks it up automatically):
  - `--decks` (roster; default = all `.dk`)
  - `--self-play-frac` (default 0.8)
  - `--pfsp-mode` (`softmax` | `pfsp`), `--pfsp-p` (default 2.0)
  - `--snapshot-every` (steps), `--promote-margin` (default 0.2)
  - `--scripted-anchor-frac` (default 0.1)
  - `--rotate-every` (steps per learner before rotating deck)
  - `--embed-dim`, `--n-envs`, `--bo3`, `--no-shaping`, `--total-timesteps`
- Driver loop: for each rotation, pick the next learner deck, build the
  `OpponentPool` from the shared snapshot dir (all decks) + scripted anchors,
  train `--rotate-every` steps via the existing `train()` core, save snapshots,
  rotate. Continue until `--total-timesteps`.
- Keep `sweep`/`baseline`/`observe` for **evaluation** (a generalist model vs each
  opponent deck is exactly what `baseline`/`sweep` measure).

### Phase 7 — bootstrap / curriculum
- **Cold start:** no snapshots exist → pool is scripted-anchor only (the existing
  fallback at `env.py:1016` and OpponentPool warning at `opponents.py:200-204`).
  This *is* the warm-start-vs-scripted phase the research recommends.
- **Auto-ramp:** scale the effective self-play fraction with the number of
  available snapshots (0 snapshots → 100% scripted; grows toward
  `--self-play-frac` as the pool fills), so the curriculum is automatic rather
  than a manual phase switch.

## Risks & decisions
- **Non-stationarity / co-training.** Rotating single-learner keeps opponents
  frozen during each learner's chunk (stable gradients), at the cost of opponents
  lagging. Accept; this is the AlphaStar main-agent pattern minus exploiters.
- **Approximate per-env PFSP.** Sharding means each subproc sees a subset; global
  win-rates are broadcast as the weighting signal. Documented approximation;
  acceptable at this scale. Alternative (centralized matchup assignment by the
  callback) is more exact but more plumbing — defer unless needed.
- **Single shared policy is retained** — perspective-normalized obs + per-episode
  seat/deck swap already make one network the right call (research: AlphaStar's
  single race-conditioned net). Do **not** split per role.
- **Hyperparameters are tunable, not laws.** 80/20, η=0.01, every-10-iters,
  +0.2 margin all come from large-compute systems; expose them as flags and tune.
- **One retrain.** Sequence the embed_dim bump, naming change, and the
  revealed-cards accumulator (`docs/plan_revealed_cards_accumulator.md`) together
  so there is a single checkpoint-invalidating retrain.

## Suggested order of work
1. Phase 0 (embed_dim) — small, independent, validates the retrain pipeline.
2. Phases 1–2 (naming + per-episode opp-deck) — unlocks multi-matchup envs.
3. Phases 3–5 (PFSP sampler + feedback + snapshots) — the core algorithm.
4. Phase 6 (league driver/CLI) — ties it together.
5. Phase 7 (bootstrap ramp) — polish.
