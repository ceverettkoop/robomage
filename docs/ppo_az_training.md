# PPO ↔ AlphaZero training strategy

How the two training regimes on this branch relate, when to hand off from one
to the other, and the (partly future) paths for moving strength back from the
AZ model into the PPO model. Companion to
[`alphazero_status.md`](alphazero_status.md), which documents the machinery
itself.

## The two regimes

| | MaskablePPO (`train.py train/league`) | AlphaZero (`train.py az`/`az-league`) |
|---|---|---|
| Sample cost | ~0.25 ms/decision rollouts | sims × (restore + replay + net eval) per decision — orders of magnitude more, even with the Phase D C++ actor |
| Learning signal | own sampled actions weighted by advantage (high variance, no lookahead) | search-vetted visit distributions + real outcomes (dense, low variance per decision) |
| Strength ceiling | what a one-shot policy can express | adds everything lookahead can verify |
| Checkpoints | `checkpoints/{deck}__final.zip` | `checkpoints/az/{deck}__azfinal.pt` (gate-promoted incumbent) |

**Intended pattern: PPO to competence, AZ to mastery.** Spending search-priced
samples to teach a random net that lands are good is waste — PPO learns the
basics almost free, and `from_ppo` warm-starts AZNet from the finished PPO
checkpoint so the AZ phase inherits rather than rediscovers them. The PPO phase
trains the per-action head by default (`PerActionMaskablePolicy`; opt out with
`--stock-head` or `ROBOMAGE_PER_ACTION_HEAD=0`): AZNet mirrors it
parameter-for-parameter, so the per-action flavor transfers policy/value heads
1:1 (a legacy stock-MlpPolicy checkpoint transfers the extractor trunk only —
its AZ warm-start starts with random heads, which is exactly the
uniform-policy cold start the default avoids).

## When to stop PPO and switch — the metrics

The decisive measurement is **search lift** — `train/eval_search_gate.py`
(search vs the SAME checkpoint raw, seats alternating):

- **Lift ≥ ~60%** (ur_delver measured 66.7%): the net knows more than its
  one-shot policy executes. That gap is exactly the signal AZ training
  distills into weights. Switch.
- **Lift ~50–53%**: either the checkpoint is too weak to guide search — check
  the `scripted:hard` reference batches; if raw also loses to scripted, keep
  training PPO — or the value head is miscalibrated (sweep `--vscale 1..3`
  before concluding).

Supporting signals, all cheaper:

1. **Baseline plateau** — `train.py baseline <deck> --games 200` across
   successive snapshots; when the win rate vs `scripted:hard` moves < ~2–3
   points across consecutive 1–2M-step chunks, gradient learning has
   saturated.
2. **Value-head health** — SB3 `explained_variance` (tensorboard) stable and
   reasonably high, plus `analysis.py` calibration. A badly calibrated critic
   makes early AZ search worse; it can be worth more PPO (or a higher vf
   coefficient) purely to hand AZ a better leaf evaluator.
3. **League stagnation** — the `__v*` promotion gate passing less often, PFSP
   weights stuck on the same losing matchups.

**Rule of thumb: switch when the baseline curve flattens AND the search gate
shows a healthy lift.**

## Going back: AZ → PPO

Three paths, in increasing order of directness. Only the first exists with
zero new code today.

### 1. AZ opponents in the PPO pool (available now)

`az:<deck>?sims=…` and `azraw:<deck>` are ordinary controller specs, usable
anywhere controllers are accepted — including PPO opponent pools. PPO then
continues training *against* MCTS-strength opposition and absorbs its gains
adversarially. Zero-cost bridge whenever both regimes coexist.

### 2. Distillation from the self-play shards (small future addition)

The AZ shards (`train/az_data/{deck}/shard_*.npz`) are literally `(obs, π,
z, mask)` — a supervised KL/cross-entropy term pulling PPO's policy head
toward the stored visit distributions is a ~50-line trainer addition. This
transfers the *search-improved policy* without PPO ever paying for search.

### 3. Weight round-trip: a `to_ppo` converter (future, deliberately cheap)

Not implemented yet, but designed for: AZNet's layout mirrors
`PerActionMaskablePolicy` 1:1 (`trunk.*` ↔ `features_extractor.*`,
`policy_body.*` ↔ `mlp_extractor.policy_net.*`, `value_body.*` ↔
`mlp_extractor.value_net.*`, `action_scorer.*` ↔ `action_scorer.*`), so
`to_ppo` is `from_ppo`'s rename map run in reverse into a loaded PPO
checkpoint's `policy.state_dict()`. Caveats to build in:

- **Per-action head required** for a full round-trip; a stock-MlpPolicy target
  can only take the trunk back.
- **Value-head semantics differ**: AZ's value is a tanh-squashed game outcome
  in [−1, 1]; PPO's critic predicts discounted *shaped* returns (bo3 game/match
  rewards, shaping terms). On resume the critic is mis-scaled, so either reset
  the final value Linear or run a short low-LR / high-vf-coef warmup while it
  recalibrates — expect briefly noisy advantages either way.
- Fresh optimizer state on resume (SB3 will not have Adam moments for the
  transplanted tensors' history).

### Do you actually want to go back?

Usually no. Once AZ training works, `azraw:` — the AZ net played one-shot —
*is* the stronger "PPO-style" policy; the search has been absorbed into the
weights. The legitimate reason to return to PPO gradients is cheap bulk
adaptation (a new deck, a shifted metagame) where search-priced samples are
not justified — and the Phase D actor's 10–100× self-play throughput narrows
even that case. The clean lifecycle:

> **PPO to competence → AZ to mastery → PPO again only to bootstrap new
> decks**, with AZ opponents in the PPO pool (path 1) as the standing bridge.
