# Opponent selection for asymmetric-game PPO — findings vs. Robomage

> Research synthesis (2026-06-21): how open-source RL projects and the academic
> literature handle opponent selection when training PPO agents on asymmetric,
> complex two-player games, mapped against how Robomage trains today.
>
> Sourcing: deep-research sweep, 22 sources, 24/25 claims passed 3-vote
> adversarial verification. Primary sources cited inline.

## The one-line consensus

Nobody who succeeds at this uses *pure* self-play against the latest policy. The
verified consensus across AlphaStar, OpenAI Five, and the self-play survey
literature is a **structured blend**: a pool of frozen past snapshots, harder
opponents sampled more often (PFSP), a reserved slot against the latest self for
fast learning, and often a scripted anchor for diversity. Robomage already has
most of the *machinery* (`OpponentPool`, `SelfPlayEnv`, scripted agent) but wires
it up in the uniform/un-prioritized way the literature specifically warns against.

## Why pure self-play fails (and where Robomage is exposed)

Confirmed 3-0 from primary sources: vanilla self-play works in **transitive**
games but in **non-transitive** ones (asymmetric, imperfect-information — i.e. an
MTG matchup) it "chases cycles (A beats B, B beats C, A loses to C) indefinitely
without making progress" (AlphaStar, Nature 2019) and suffers **strategy
collapse** where "the agent forgets how to play against a wide variety of
opponents because it only requires a narrow set of strategies to defeat its
immediate past version" (OpenAI Five, https://arxiv.org/pdf/1912.06680).

**Where Robomage sits:** `SelfPlayEnv` (env.py:874) pools all
`{opp}_{model}_*.zip` checkpoints and picks **one uniformly at random**
(`np.random.choice(files)`, env.py:1023), refreshed every 100 episodes
(`RELOAD_EVERY=100`). That's roughly **uniform fictitious self-play** over saved
snapshots — better than "latest-only" (you do replay history), but it has two
gaps the literature flags:

- **No prioritization** — a checkpoint you already crush is sampled as often as
  the one that beats you.
- **No latest-self slot** — you only ever play *frozen* checkpoints, never the
  live improving policy, so learning is slower than it needs to be.

## The principled fixes, ranked by how much they'd cost you

**1. Fictitious Self-Play (FSP / NFSP)** — best-respond to the historical
*average* of past policies, not the latest. Heinrich, Lanctot & Silver (ICML
2015, https://proceedings.mlr.press/v37/heinrich15.html) and NFSP (Heinrich &
Silver, https://arxiv.org/abs/1603.01121) prove convergence toward approximate
Nash in two-player zero-sum / imperfect-info games "whereas common reinforcement
learning methods diverged." *Caveat:* these guarantees hold only for
zero-sum/potential games — which is your regime, so it applies, but don't
over-generalize.

**2. Prioritized FSP (PFSP)** — sample opponent *i* with probability ∝
`f(x)=(1-x)^p` where `x` = your win-rate vs *i*. Hard opponents get upweighted
(AlphaStar). This is the single highest-value upgrade to your
`SelfPlayEnv`/`OpponentPool`.

**3. OpenAI Five recipe (the one to copy)** — the most practical template for a
small project, verified 3-0:

- **80% of games vs. the latest parameters, 20% vs. the past-version pool.**
- The 20% is *not uniform*: each past opponent has a quality score `q_i`, sampled
  via softmax `∝ exp(q_i)`; when the current agent **beats** past opponent *i*,
  `q_i -= η/(N·p_i)` with `η=0.01`; when the past opponent wins, **no update**.
  This auto-upweights still-challenging snapshots — a lightweight PFSP equivalent.
- **Snapshot cadence:** add the current agent to the pool **every 10 iterations**,
  init its score to the current max.

**4. AlphaStar league** (main + main-exploiters + league-exploiters, exploiters
reset to a supervised base, frozen into the pool when they hit ~70% vs. main) —
verified, but the Minimax-Exploiter paper (AAMAS 2024,
https://arxiv.org/abs/2311.17190) explicitly calls full leagues' "large
computational cost… impractical to deploy in highly iterative settings."
**Overkill for Robomage.** Skip it; the exact role *counts* were even refuted
(1-2) as non-canonical.

## Cadence & pool-size guidance (all system-specific, not laws)

- OpenAI Five: snapshot **every 10 iterations**.
- TStarBot-X (https://arxiv.org/pdf/2011.13729): freeze a self-copy **every ~12h**;
  promote an exploiter when it beats the main agent **>70% win-rate**.
- **SIMPLE** (https://github.com/davidADSP/SIMPLE) — the closest concrete
  TCG-adjacent template: promotes a new opponent into the "network bank" only when
  it beats the current best by a **0.2 margin (~30-40 PPO iterations)**.

Your `RELOAD_EVERY=100` episodes governs *opponent refresh*, but your
*snapshot-creation* cadence is whatever `train.py` saves checkpoints at — worth
making that an explicit, tuned knob (SIMPLE's "promote on +0.2 win-margin" gate is
a clean, compute-cheap pattern that fits your existing `--tally` win-rate
tracking).

## Scripted anchor — you already have the right tool, used in the wrong mode

`OpponentPool` with weighted specs (`"scripted:hard=2,random-model"`,
opponents.py:142) **is** the per-episode weighted scripted+self-play mixing, and
SIMPLE ships exactly this (a rules-based baseline as a selectable opponent). The
diversity rationale that prevents collapse is well-verified; that a scripted
anchor *specifically* prevents collapse is an **inference (medium confidence)** —
no source directly measured it.

**Gap:** your `SelfPlayEnv` only falls back to scripted when *no* checkpoint
exists (env.py:1016). Best practice says keep a scripted weight in the mix
*always*. Practically: prefer driving self-play **through `OpponentPool`** with a
permanent scripted slice, rather than the pure-checkpoint `SelfPlayEnv` path.

## Asymmetry — Robomage's design is already aligned

The verified literature didn't directly resolve shared-vs-separate policy for
asymmetric TCGs (flagged as an open gap), but AlphaStar's approach is a **single
network conditioned on race**, and Robomage matches that philosophy well:

- **Single shared policy** + **perspective-normalized observations**
  (priority-player view, env.py:878) + **random seat/deck swap per episode**
  (env.py:624-627) + **reward negation for seat B** (env.py:767).

This is the right call for a small project — one policy, more data, symmetry
broken per-episode. Do *not* split into per-deck policies. One caveat: with
asymmetric *matchups* (deck A vs deck B is not a mirror), the
`{opp}_{model}_*.zip` mirror-matchup checkpoint convention means self-play pools
are matchup-scoped — correct, but it fragments your snapshot pool across
matchups, so each pool stays small. PFSP weighting matters *more* when pools are
small.

## Reward shaping — literature gap; current scheme is reasonable but unvalidated

The research found **no verified guidance** specific to sparse, long-horizon card
games (explicit open question). Robomage's current scheme — ±1 terminal,
hand/board-advantage shaping capped per episode and **annealed by `(1-win_rate)`**
(train.py:277), bo3 ±0.3/game + ±1.0/match — follows the sensible general
principle of decaying shaping as competence rises, and per-game intermediate
rewards in bo3 match the dense-reward intuition. Nothing in the literature
contradicts it; nothing validates the specific magnitudes either. Treat the caps
as empirical knobs.

## Directly relevant repos to mine (beyond AlphaStar/OpenAI Five papers)

- **SIMPLE** (https://github.com/davidADSP/SIMPLE) — single-player wrapper +
  network-bank snapshot self-play + rules-based baseline + 80/20 "mostly_best"
  sampling. Closest architectural analog to what you'd build next.
- **gym-locm** (https://github.com/ronaldosvieira/gym-locm) — *Legends of Code and
  Magic*, an actual TCG, PPO + scripted baselines. Most TCG-specific RL codebase
  surfaced.
- **RL-CCG-project** (https://github.com/sacktock/RL-CCG-project),
  **alphastone** (https://github.com/sirmammingtonham/alphastone, Hearthstone) —
  smaller, illustrative.
- No mature MTG-specific PPO repo surfaced.

## Concrete recommendations for Robomage, in priority order

1. **Add win-rate prioritization to opponent sampling.** Replace the uniform
   `np.random.choice(files)` in `SelfPlayEnv._reload_opponent` (and the equal
   weights in `OpponentPool`'s `random-model` expansion) with OpenAI Five's
   softmax quality score, or PFSP `(1-x)^p`. You already track per-opponent
   win-rate via `--tally`; feed it in. **Highest value, lowest effort.**
2. **Adopt the 80/20 latest/pool split.** Reserve ~80% of self-play games for the
   most recent checkpoint, 20% for the prioritized pool — recovers fast learning
   you currently forgo by only playing frozen snapshots.
3. **Always keep a scripted slice in the mix.** Route self-play through
   `OpponentPool` with a permanent `scripted:hard` weight rather than the
   pure-checkpoint `SelfPlayEnv`, so the anchor never drops out.
4. **Make snapshot-creation an explicit gate.** Adopt SIMPLE's "promote only when
   it beats current best by margin" rule so your pool fills with genuinely
   distinct, stronger snapshots instead of near-duplicates.
5. **Skip full league/exploiter play** — verified as compute-impractical at your
   scale; PFSP-over-a-pool + scripted anchor captures most of the benefit.

**Caveats:** every cadence/ratio above (10 iters, 12h, ±0.2 margin, 80/20) is a
*system-specific design choice from massive-compute projects*, not a universal
constant — tune them. The strongest evidence is from 2019-era systems still
treated as canonical.

## Primary sources

- AlphaStar (Nature 2019): https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf
- AlphaStar blog: https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/
- OpenAI Five: https://arxiv.org/pdf/1912.06680
- Fictitious Self-Play (Heinrich, Lanctot & Silver, ICML 2015): https://proceedings.mlr.press/v37/heinrich15.html
- NFSP (Heinrich & Silver 2016): https://arxiv.org/abs/1603.01121
- Self-play survey (2024): https://arxiv.org/pdf/2408.01072
- TStarBot-X: https://arxiv.org/pdf/2011.13729
- Minimax Exploiter (AAMAS 2024): https://arxiv.org/abs/2311.17190
- SIMPLE (repo): https://github.com/davidADSP/SIMPLE
- gym-locm (repo): https://github.com/ronaldosvieira/gym-locm
