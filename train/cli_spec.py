"""Single source of truth for the train.py / analysis.py / play.py command lines.

This module is intentionally dependency-free (stdlib only) so that both the
heavyweight scripts (which pull in numpy/torch/gymnasium) and the lightweight
``tui.py`` launcher can import it without paying for those imports.

The scripts build their ``argparse`` parsers from the ``Tool`` objects here via
``apply_to_parser``; the TUI builds its input forms from the same objects.  Add
or change a flag in one place and both stay in sync.
"""

import os
from dataclasses import dataclass, field, replace

from archetypes import ARCHETYPES
from gate_sprt import DEFAULT_GATE_ALPHA as _GATE_ALPHA

# ── Canonical CLI constants (single home; imported by env.py / train.py) ──────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN_DIR = os.path.join(REPO_ROOT, "bin")  # game must be run from here for resource lookup
# The Makefile writes each configuration's binaries to bin/<config>/ (see its
# CONFIG variable): bin/debug/robomage from a plain `make`, bin/release/robomage
# from `make BUILD=RELEASE`. Tooling selects one deliberately via ROBOMAGE_BUILD
# (default "debug", matching a plain `make`); every command's --binary flag still
# defaults to BINARY, so an explicit path stays an escape hatch. BIN_DIR stays the
# launch cwd (the engine finds resources/ via getcwd()), independent of BUILD_DIR.
BUILD = os.environ.get("ROBOMAGE_BUILD", "debug")
BUILD_DIR = os.path.join(BIN_DIR, BUILD)  # where this config's binaries live
BINARY = os.path.join(BUILD_DIR, "robomage")

# Interactive / training tools (GUI, standalone analysis browser, PPO and AZ
# training) default to the RELEASE build instead — debug's assertions and
# -O0 make them noticeably slower and there's no need for the extra checking
# once a card's behavior is already verified by `make check`. ROBOMAGE_BUILD
# still overrides this the same as it overrides BUILD above (set it to
# "debug" to force these tools back onto the debug binary, e.g. to reproduce
# an assertion failure only the debug build catches).
INTERACTIVE_BUILD = os.environ.get("ROBOMAGE_BUILD", "release")
INTERACTIVE_BUILD_DIR = os.path.join(BIN_DIR, INTERACTIVE_BUILD)
INTERACTIVE_BINARY = os.path.join(INTERACTIVE_BUILD_DIR, "robomage")

# AlphaZero sideboard-root search budget (single home; imported by az_selfplay,
# opponents.SearchController, and the CLI flag defaults below). A bo3 sideboard
# root is searched by mcts.run_plan_search — a FLAT search over complete
# sideboard configurations ("plans"), not a PUCT tree: a coverage pass builds
# one greedily-completed plan per legal root action (Done stand-pat included),
# then DEFAULT_SB_BRANCHES deterministic alternate completions of the best
# branches; every plan is priced by leaf rollout on every determinized world
# and the training target is pi = softmax(Q / mcts.SB_PI_TAU) over first picks.
# See the plan-search section of train/mcts.py (mirrored in
# src/actor/az_mcts.cpp) and docs/alphazero_status.md.
#
# Why plans, not a tree (2026-08): at the old sb_sims=128 / 4 worlds a single
# world's tree got 32 sims for the then ~33-child delta menu — it could not even
# visit every first pick once, and shard analysis showed sideboarding as the
# least-converged decision domain (KL(search||net) 0.39-0.47 vs ~0.11 in-game).
# The rollout memo already keyed values on the order-insensitive pick multiset,
# i.e. on configurations; the plan search makes that the primary object and
# spends nothing on pick orderings. (The engine menu has since gone IN-FIRST, so
# the root menu is Done + the distinct SIDEBOARD names only; the ~33-child figure
# and the measurements below are history from the old delta menu.) NOTE: the az /
# az-league / az-selfplay CLI args MUST reference these constants, not literals —
# argparse always supplies the CLI default, so a drifted literal silently
# overrides this "one home" (that bug shipped runs at sb_sims=32 while the
# constant said 128).
#
# Extra plans per root beyond the coverage pass. Coverage alone costs ~menu_size
# plans — under the IN-FIRST menu that is (distinct sideboard names + 1), ~16 on
# a 15-card sideboard; each extra adds worlds more rollouts. 0 = coverage only.
DEFAULT_SB_BRANCHES = 8
# Determinized worlds at a sideboard root (each world seed IS a sampled
# next-game deal). Every plan is priced on every world (value = mean), so more
# worlds buys determinization spread at linear cost per plan.
DEFAULT_SB_WORLDS = 4
# Leaf-rollout horizon for pricing a plan, in player turns of the sampled next
# game (6 = 3 full turn cycles). After a plan's picks are replayed, the raw
# policy plays the opponent's picks, the mulligans, and the next game to
# end-of-turn-N; THAT state's net value (or a true terminal +/-1) is the
# plan's value on that world. 12 turns measurably ~tripled az-league match
# wall-clock in the old tree search; 6 still reaches the turns where a
# sideboard swap actually shows up. 0 prices the completed decklist with the
# net's static read (no playout). The per-turn step cap lives in
# mcts.ROLLOUT_STEPS_PER_TURN (mirrored in src/actor/az_mcts.cpp).
DEFAULT_SB_ROLLOUT_TURNS = 6

# ── Sideboard prior-rollout self-play ───────────────────────────────────────
# GENERATION side: training self-play does NOT plan-search sideboard roots by
# default. Picks are sampled from the net prior with exploration and the row is
# recorded as a ONE-HOT behavior sample; the trainer learns it REINFORCE-style
# against the realized next-game z ("the real next game is the rollout").
# 'plan' restores the multi-world plan search at generation time; gates, eval,
# analysis, and play always use the plan search regardless of this mode.
# Dead-card rules (train/sb_dead_rules.json) prune table-dead INs in both
# modes and on both backends.
DEFAULT_SB_SELFPLAY_MODE = "prior"
# Behavior policy at a prior-mode sb root:
#   b = (1-eps) * softmax(log(prior)/temp) + eps * uniform(live actions).
# The eps floor keeps every live pick collecting outcome evidence even after
# the prior sharpens (no self-reinforcing collapse).
DEFAULT_SB_EXPLORE_TEMP = 1.0
DEFAULT_SB_EXPLORE_EPS = 0.10
# TRAINING side (az-train): sb one-hot rows train the policy with an
# advantage-weighted (z - V_detached) log-prob term, weighted DEFAULT_SB_LOSS_COEF,
# and each batch draws DEFAULT_SB_BATCH_FRAC of its rows from the sb-row pool
# (sb rows are ~0.3% of a window; uniform draws would starve the signal).
DEFAULT_SB_BATCH_FRAC = 0.05
DEFAULT_SB_LOSS_COEF = 1.0

# ── AZ pipeline defaults ────────────────────────────────────────────────────
# One home for every tunable the AZ subcommands share, same one-home rule as
# the sb_* knobs above: the az-selfplay / az-train / az-eval / az / az-league
# CLI args AND the az_train.py / az_selfplay.py function defaults MUST
# reference these constants, not literals — argparse always supplies the CLI
# default, so a drifted literal silently overrides the constant.

# Self-play generation.
DEFAULT_AZ_GAMES = 50        # matches per az/az-league slot (and az-selfplay)
DEFAULT_AZ_SIMS = 1028       # in-game PUCT sims, TOTAL across worlds (the 8_20 run budget)
DEFAULT_AZ_WORLDS = 8        # determinized worlds per search (the 8_20 run budget)
# Playout-cap randomization (KataGo-style), the anti-memorization lever: each
# searched in-game root gets the FULL --sims budget with probability
# --full-search-frac (pi is recorded from those rows only), else a fast
# --fast-sims search that picks the move but records NO policy target — the
# shard row keeps its real root value q (and so td_q/z) with an all-zero pi
# the trainer's policy loss skips. At the same engine budget this multiplies
# distinct games per slot by ~ sims / (frac*sims + (1-frac)*fast_sims)
# (~3x at the defaults): more distinct outcomes for the value head instead of
# more sims per position. Sideboard plan-search roots are exempt (their own
# sb_* budget; always full policy rows). frac >= 1 restores the classic
# every-root-full behavior. GENERATION side, honored by both backends; the
# full-vs-fast coin is a deterministic hash (az_selfplay.playout_cap_full),
# never a play-rng draw.
DEFAULT_AZ_FULL_SEARCH_FRAC = 0.25
DEFAULT_AZ_FAST_SIMS = 128   # the fast budget; also the az:gen serving sims
DEFAULT_AZ_MIRROR_FRAC = 0.25  # P(opponent deck == focus deck) per self-play game
# Opponent-pool self-play: that fraction of the schedule's pure-self-play
# matches puts an OLDER checkpoint on the opponent seat — the incumbent
# (gen__azfinal) plus the newest candidate snapshots distinct from both it and
# the generator, up to DEFAULT_AZ_OPP_POOL_SIZE nets — instead of mirroring the
# learner. Only the learner seat's decisions are recorded (the opponent is
# environment, like a vs-scripted cell); the opponent searches noise-free
# argmax under the shared playout-cap coin. External pressure against
# self-play drift at unchanged engine budget; 0 disables (pure mirror
# self-play). GENERATION side, honored by both backends.
DEFAULT_AZ_OPP_POOL_FRAC = 0.25
DEFAULT_AZ_OPP_POOL_SIZE = 3
# Self-play exploration clock (GENERATION side, both backends). Keyed on the
# obs's game-turn float (the engine's turn counter advances once per PLAYER
# turn, so 8 = each player's first 4 turns); a bo3 sideboard root counts as
# turn 0 of the upcoming game. At each learner searched root the exploration
# probability eps(turn) decides — by a deterministic (seed, root_idx) hash coin,
# never a play-rng draw — whether the real action is SAMPLED from the visit
# distribution (an exploratory row) or the visit argmax is played:
#   turn <= full_turns                : eps = 1.0
#   full_turns < turn <= full + decay : eps falls linearly from 1.0 to floor
#   beyond                            : eps = floor
# Sampling uses the raw visit posterior (temperature 1), so a late exploratory
# draw stays near the search's own line. See az_selfplay.explore_prob.
DEFAULT_AZ_EXPLORE_FULL_TURNS = 8
DEFAULT_AZ_EXPLORE_DECAY_TURNS = 8
DEFAULT_AZ_EXPLORE_FLOOR = 0.05
# n-step TD value target (GENERATION side — baked into each shard's td_q column).
# Each recorded sample's value target bootstraps off the search root value q of
# the sample n decisions later in the same game, sign-flipped when that decision's
# mover is the other seat. The chain is SHORTENED at an exploratory move (the
# temperature branch played a non-argmax action), because everything after such a
# move is off-policy noise rather than the line the search endorsed; a window that
# reaches the end of the game uses the true outcome z instead.
DEFAULT_AZ_TD_N = 10

# Optimization (az-train / the train step of az / az-league).
# LR is half the PPO LR_CONST below: az-train refines an already-competent warm
# start against a small, heavily game-correlated shard window, where the old
# 1e-3 rewrote the PPO trunk/bodies (~60-100% weight drift within a few
# thousand minibatches) and let the value head memorize per-game outcomes.
DEFAULT_AZ_LR = 5e-5
DEFAULT_AZ_WEIGHT_DECAY = 1e-4
DEFAULT_AZ_VALUE_DECAY = 1e-4    # weight decay on the dense value_body.* params
                                 # (value_head columns stay decay-exempt). Matches the
                                 # shared decay: playout-cap randomization is the
                                 # anti-memorization lever now, and heavier decay here
                                 # shrinks leaf values (td-calibration gain a stuck
                                 # ~1.5-1.8), starving PUCT of value signal
DEFAULT_AZ_EPOCH_FRAC = 1.0      # auto-batches (batches=0) = this fraction of one epoch
                                 # — only with --rows-per-game 0 (row-uniform sampling)
# Game-uniform training sampler: each cycle draws at most this many rows from
# every GAME in the window (uniformly within the game, without replacement),
# shuffles the pool and trains one pass over it. Row-uniform sampling let a
# 230-row game push its outcome label six times harder than a 40-row game —
# the value head's cheapest fit is then a per-game fingerprint (the 2x
# pre/post-train MSE tripwire). The cap equalizes every game's weight and
# bounds how often any one label is seen; the pool size (~games x cap) sets
# the batch count, so --epoch-frac is inert unless this is 0.
DEFAULT_AZ_ROWS_PER_GAME = 32
                                 # over the window; safe at a full epoch now that
                                 # playout-cap randomization supplies ~3x distinct games
                                 # per slot — the pre-vs-post-train window MSE tripwire
                                 # polices memorization (warns at a 2x gap)
DEFAULT_AZ_C_PUCT = 2.5          # PUCT exploration constant, EVERY search path: az
                                 # training/gate/baseline, the az:/mcts: spec grammar
                                 # (play, observe, analysis), mcts.py defaults and the
                                 # parity tests. Higher weights search Q over the net
                                 # prior (1.5 was prior-dominated). The C++ actor's
                                 # compiled default mirrors this BY HAND (az_mcts.h /
                                 # az_actor_main.cpp) and every launcher passes --c.
DEFAULT_AZ_BATCH_SIZE = 256
DEFAULT_AZ_TRAIN_BATCHES = 1000  # standalone az-train
# Per az / az-league slot: 0 = AUTO, one epoch over the loaded window
# (max(1, n_samples // batch_size) optimizer updates). A fixed 1000 batches x 256
# over a ~137k-sample window was ~1.9 epochs of re-fitting one small, heavily
# game-correlated window — the memorization regime — and the effective epoch
# count drifted silently as the window's data volume changed. Auto pins it at
# exactly one pass regardless of volume. The STANDALONE az-train default above
# stays a fixed batch count (it is a manual, one-off refit, not a cycle step).
DEFAULT_AZ_CYCLE_BATCHES = 0
DEFAULT_AZ_WINDOW = 50           # newest-N-shards training window (0 = AUTO)
DEFAULT_AZ_CV = 1.0              # value-loss weight
# Mixing weight of the shard's n-step TD target against the per-game outcome:
# v_target = (1 - q_mix) * z + q_mix * td_q. 0 = the classic pure-outcome AZ
# target, 1 = pure bootstrap. TRAINING side (the shards always carry td_q).
DEFAULT_AZ_Q_MIX = 0.5       # equal blend of outcome and n-step TD target: td_q
                             # cuts z's per-game variance, but over-weighting it
                             # (0.75) contracted decided-game value targets to
                             # ±~0.5-0.7 — the 2026-08-25 gate failure. Distinct-
                             # game volume is a generation-side concern (playout-
                             # cap randomization), not q_mix's

# Promotion gate (az-eval / the gate step of az / az-league).
#
# The gate is a SEQUENTIAL test (train/gate_sprt.py): the matchup panel is
# scheduled in balanced ROUNDS of DEFAULT_AZ_EVAL_GAMES matches, up to
# DEFAULT_AZ_GATE_MAX_ROUNDS, and an SPRT on the accumulated aggregate is
# re-asked after EVERY completed match — promote, keep, or keep playing. A
# decisive candidate stops as soon as the evidence is in (the actor legs still
# in flight are terminated); a marginal one keeps playing rather than taking a
# verdict from an underpowered sample. Hypotheses are symmetric about 0.5
# (H1 p=DEFAULT_AZ_PROMOTE_THRESHOLD vs H0 p=1-threshold), so a genuinely equal
# candidate is a coin flip rather than a near-certain rejection. The per-deck
# floor veto is checked as results land too, and stops the gate once it is
# beyond rescue under the cap.
DEFAULT_AZ_EVAL_GAMES = 28        # matches per ROUND (>= 2 per panel matchup so
                                  # seats alternate within every round)
# Gate at SERVING strength, which is also the TRAINING budget: the `az:gen`
# serving spec searches 128 sims over 4 worlds, so anything shallower gates a
# different — and much weaker, prior-dominated — player than the one actually
# deployed, and its promote/keep verdict says little about the served net. The
# lever for gate COST is more matches per verdict (--gate-max-rounds), never
# fewer sims per match.
DEFAULT_AZ_EVAL_SIMS = DEFAULT_AZ_SIMS
DEFAULT_AZ_EVAL_WORLDS = DEFAULT_AZ_WORLDS
# `baseline`: the AZ generalist at the full league search budget vs scripted:hard
# over the whole league grid — 10 bo3 matches per matchup (1000 matches on the
# 10-deck roster, ~10-13 h on a 32-core box with the GPU eval server) on the C++
# actor with 48 legs in flight (the engines are CPU-bound; more clients only
# fill GPU batches, they do not add throughput).
DEFAULT_BASELINE_MODEL = "az:gen"
DEFAULT_BASELINE_OPPONENT = "scripted:hard"
DEFAULT_BASELINE_GAMES = 10
DEFAULT_BASELINE_WORKERS = 48
DEFAULT_AZ_PROMOTE_THRESHOLD = 0.55   # SPRT's H1; H0 is its mirror, 0.45
DEFAULT_AZ_GATE_MAX_ROUNDS = 8    # hard cap: 8 x 28 = 224 matches. At the cap
                                  # the incumbent keeps the seat unless the
                                  # score reached the promote bar; a score
                                  # inside the indifference region is
                                  # UNDECIDED (kept, not a failed gate)
DEFAULT_AZ_GATE_ALPHA = _GATE_ALPHA   # symmetric SPRT error rates (alpha=beta)
DEFAULT_AZ_GATE_FLOOR = 0.2
# Minimum candidate matches on a piloted deck before the per-deck floor veto can
# fire. Sized so an equal candidate is not plausibly wiped on a deck by chance:
# at 8 matches a 0-for-8 is a ~0.4% shot per deck, small enough across a ~10-deck
# roster to leave the aggregate test in charge of the verdict, while the veto
# still catches a candidate that genuinely forgot how to pilot a deck. A veto
# overrides the aggregate, so a false positive here is a silent bias toward the
# incumbent.
DEFAULT_AZ_GATE_FLOOR_MIN = 8
DEFAULT_AZ_GATE_CROSS_PAIRS = 2
# Gate every K az-league slots. 2 amortizes the gate's cost (it searches at the
# full eval budget above, over at least one panel round): at DEFAULT_AZ_LR a
# single slot's weight delta rarely carries the sequential test to a verdict, so
# gating every slot mostly buys "kept-incumbent" lines.
DEFAULT_AZ_GATE_EVERY = 2

# Expert (behavior-cloning) shard generation. --expert-decks defaults to the
# ROSTER sentinel: every deck in decks/league/ gets scripted:hard demonstration
# shards each cycle (the hand-coded lines search never finds are not a
# doomsday-only problem). 'none' (case-insensitive) or an empty value disables.
# The sentinel is resolved in ONE place — az_train._resolve_expert_decks, where
# expert_decks is consumed — so the az-league sidecar can persist the RAW user
# value and a --resume re-resolves it against the roster of the day.
EXPERT_DECKS_ROSTER = "roster"
EXPERT_DECKS_NONE = "none"
DEFAULT_AZ_EXPERT_GAMES = 8

# Exhaustive-matrix repeats: how many times each cell of the exact matchup matrix
# is played per az cycle / az-league slot (--exhaustive-repeats).
DEFAULT_AZ_EXHAUSTIVE_REPEATS = 2

# Rotating vs-scripted cells added to an --exhaustive-selfplay slot: K matches
# per slot drawn from the full ORDERED (focus, opponent) pair list, starting at
# offset (slot * K) % n_ordered and wrapping, so coverage of every ordered pair
# accumulates across slots while a slot stays overwhelmingly C++-actor self-play.
# 0 disables. Ignored under full --exhaustive (which plays every scripted cell).
DEFAULT_AZ_SCRIPTED_CELLS = 20

TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 32            # parallel game processes
N_ENVS_SELF_PLAY = 10  # self-play (each loads an opponent model)
EMBED_DIM = 128        # policy feature-extractor embed dim for fresh models
# PPO entropy bonus coefficient. Canonical value for every training path (train,
# league) AND re-asserted on checkpoint resume: MaskablePPO.load restores the
# ent_coef the checkpoint was *saved* with, so without the override a model born
# under an old value keeps it forever. (A stale 0.12 — 10x the intended 0.012 —
# rode along this way in checkpoints created before the original fix; the entropy
# term then dominated the PPO loss ~2.5:1 over the policy-gradient term, forcing
# a permanently noisy policy. That hurts precise-line decks like Doomsday far
# more than aggro decks, and capped league win-rates.)
ENT_COEF = 0.012

# PPO KL early-stop threshold. When the per-epoch approximate KL exceeds
# 1.5 * TARGET_KL, MaskablePPO aborts the rest of that update's epochs — a
# safety brake against destructively large policy steps that becomes more
# valuable as a generalist matures. Canonical value for every training path AND
# re-asserted on checkpoint resume (older checkpoints saved with target_kl=None
# would otherwise never early-stop). See train.py _reassert_hparams.
TARGET_KL = 0.02

# PPO optimization epochs per update and policy clip range. Canonical values for
# every training path AND re-asserted on checkpoint resume (MaskablePPO.load
# restores whatever the checkpoint was saved with — see train.py
# _reassert_hparams), so resumed generalists follow the current values rather
# than the ones they were born under. Both are overridable per session with the
# --n-epochs / --clip-range training flags (the override applies to fresh AND
# resumed models for that session only; it is not persisted to the checkpoint's
# future resumes, which fall back to these canonical values).
N_EPOCHS = 5
CLIP_RANGE = 0.2

# Learning rate: held CONSTANT at LR_CONST across all training, regardless of a
# model's cumulative num_timesteps. (Previously a linear decay from LR_PEAK to
# LR_FLOOR over LR_DECAY_STEPS; now flat so the LR never anneals with step
# count.) Still applied by train.py's LRDecayCallback on every training path,
# which keeps the LR pinned to this value on every rollout — overriding whatever
# learning_rate a resumed checkpoint was saved with. See lr_for_timesteps().
LR_CONST = 1e-4
# Back-compat aliases for callers/imports that referenced the old decay knobs.
LR_PEAK = LR_CONST


def lr_for_timesteps(num_timesteps: int) -> float:
    """Constant learning rate LR_CONST, independent of num_timesteps.

    Kept as a function (rather than a bare constant) so LRDecayCallback and every
    other caller keep the same interface; the num_timesteps argument is now
    ignored since the LR no longer decays."""
    return LR_CONST


# Shaping-reward anneal, keyed to ABSOLUTE cumulative num_timesteps like the LR
# decay above (SB3's progress_remaining resets every learn(), so a per-deck
# generalist resumed across many short sessions would restart the anneal each
# time). Scale falls linearly from 1.0 at step 0 to 0.0 at SHAPING_DECAY_STEPS
# and stays there, so past that point the objective is pure win/loss. The old
# win-rate-keyed anneal (scale = 1 - win_rate) could never reach 0 in hard
# matchups — exactly where the PFSP league concentrates games — leaving shaping
# in the asymptotic objective forever, and made the reward scale depend on the
# opponent pool (unobservable to the value function). Applied by train.py's
# ShapingScaleCallback on every shaped training path.
SHAPING_DECAY_STEPS = 5_000_000


def shaping_scale_for_timesteps(num_timesteps: int) -> float:
    """Linear shaping-scale anneal from 1.0 to 0.0 over [0, SHAPING_DECAY_STEPS]
    cumulative timesteps, clamped at 0.0 beyond that.

    Keyed to the model's absolute num_timesteps so the schedule is continuous
    across checkpoint resumes (same rationale as lr_for_timesteps)."""
    return max(0.0, 1.0 - max(0, num_timesteps) / SHAPING_DECAY_STEPS)

# PPO hyperparameters for newly constructed models — single home so the
# single-opponent train() path and the league path can never drift apart (the
# stale-ent_coef bug survived one fix precisely because these were duplicated
# inline at both sites). On checkpoint RESUME, MaskablePPO.load restores
# whatever the checkpoint was saved with; ent_coef, target_kl, n_epochs,
# clip_range, gamma, and gae_lambda are re-asserted afterwards (train.py
# _reassert_hparams), and the LR is driven every rollout by LRDecayCallback
# regardless of the saved learning_rate. `learning_rate` here is just the
# nominal starting value the callback overrides on the first update. The
# --n-epochs / --clip-range CLI flags mutate this dict for the session
# (train.py _apply_ppo_overrides) so fresh construction and the resume
# re-assertion both see the override.
PPO_KWARGS = dict(
    learning_rate=LR_PEAK,
    n_steps=1548,       # steps per env per update
    batch_size=1024,
    n_epochs=N_EPOCHS,
    gamma=0.9975,     # 1/(1-γ) = 400-step horizon; episodes run 100-400 decisions with mostly terminal reward
    gae_lambda=0.97,  # keep γλ high enough that terminal reward reaches back directly in GAE
    clip_range=CLIP_RANGE,
    ent_coef=ENT_COEF,
    target_kl=TARGET_KL,
)
NET_ARCH = [512, 512]  # policy/value MLP head sizes (after the feature extractor)

# League (PFSP) defaults.
# self-play is a small floor, not the bulk of games: the PFSP-weighted historical
# branch (1-winrate)^p already concentrates training on the learner's worst
# matchups, and the mirror is one entry in that weighted pool. A large fixed
# self-play slot would drown that concentration in ~50% mirror games, so keep it
# low and let PFSP send the majority of episodes to the hardest opponents.
LEAGUE_SELF_PLAY_FRAC      = 0.2     # prob of facing the latest snapshot of the learner's own deck
LEAGUE_SCRIPTED_ANCHOR_FRAC = 0.1    # min share of the historical pool reserved for the scripted anchor
LEAGUE_PFSP_P              = 2.0      # exponent p in (1-winrate)^p
LEAGUE_SOFTMAX_ETA         = 0.01    # softmax quality learning rate
LEAGUE_SNAPSHOT_EVERY      = 250_000 # steps between frozen snapshots
LEAGUE_PROMOTE_MARGIN      = 0.05    # only snapshot when win-rate >= 0.5 + margin (negative gates below 50%; first exempt; 0 disables)
LEAGUE_ROTATE_EVERY        = 500_000 # steps to train one learner deck before rotating
# Adaptive rotation length: a struggling deck's rotation is stretched up to this
# multiplier of --rotate-every, scaled by how far its last league win-rate sits
# below 50% OR how far its cumulative trained steps trail the roster leader
# (whichever need is greater). Every deck still rotates — the boost is bounded,
# so strong decks are never starved, and it self-corrects as win-rates recover.
# 1.0 disables (fixed-length rotations).
LEAGUE_ADAPTIVE_BOOST      = 2.0
# Minimum share of league episodes reserved for the archetype EXPLOITERS
# ('exp_<arch>__*.zip', see train.py's `exploiter` subcommand). Exploiters also take
# part in the normal PFSP weighting; this floor keeps their alien styles in the field
# after the learner starts beating them (PFSP alone would weight them away).
LEAGUE_EXPLOITER_FLOOR     = 0.1

# Exploiter-run defaults: a dedicated learner piloting ONE archetype's decks against
# the frozen generalist, saved under its own 'exp_<archetype>' stem.
EXPLOITER_STEPS            = 500_000 # default step budget for an exploiter run
EXPLOITER_CHUNK            = 100_000 # steps per sidecar/progress chunk


# ── Distributed league sharding (docs/distributed_league_training.md) ─────────
# A league run may train only a slice of the roster ("--shard i/n") while still
# sampling opponents from the full roster — one shard per machine over a
# shared/synced checkpoint dir. These helpers are the single home for the shard
# spec format and the filename tag that namespaces each shard's sidecar,
# heartbeat, and control files; they live here (stdlib-only) so the ops scripts
# (scripts/league_worker.py, scripts/league_dashboard.py) can import them
# without pulling in train.py's torch stack.

def parse_shard(spec):
    """Parse a '--shard i/n' spec into (shard_id, num_shards); None -> None.

    Raises ValueError on malformed specs (not 'i/n', n < 1, or i outside
    0..n-1) so a typo fails at startup rather than silently training the
    wrong roster slice."""
    if spec is None:
        return None
    parts = str(spec).split("/")
    if len(parts) != 2:
        raise ValueError(f"--shard must look like 'i/n' (got {spec!r})")
    try:
        i, n = int(parts[0]), int(parts[1])
    except ValueError:
        raise ValueError(f"--shard must be two integers 'i/n' (got {spec!r})")
    if n < 1 or not (0 <= i < n):
        raise ValueError(f"--shard {spec!r}: need 0 <= i < n and n >= 1")
    return i, n


def shard_tag(spec):
    """Filename tag for a shard's coordination files: '' when unsharded, else
    '.shard{i}of{n}' — e.g. _league_progress.shard0of2.json."""
    parsed = parse_shard(spec)
    if parsed is None:
        return ""
    return f".shard{parsed[0]}of{parsed[1]}"


# ── Match format (--format bo1|bo3) ──────────────────────────────────────────

FORMAT_BO1 = "bo1"
FORMAT_BO3 = "bo3"
FORMAT_CHOICES = (FORMAT_BO3, FORMAT_BO1)
DEFAULT_FORMAT = FORMAT_BO3


def is_bo3(args) -> bool:
    """True if a parsed namespace (or opts dict) selects best-of-three.

    Reads the ``format`` dest every ``--format`` flag writes; a namespace that
    carries none (a hand-built one) gets the default format."""
    fmt = (args.get("format") if isinstance(args, dict)
           else getattr(args, "format", None))
    return (fmt or DEFAULT_FORMAT) == FORMAT_BO3


def format_name(bo3: bool) -> str:
    """The ``--format`` value for a bo3 boolean (the inverse of ``is_bo3``)."""
    return FORMAT_BO3 if bo3 else FORMAT_BO1


# ── Interactive play (play.py and the GUI launcher) ──────────────────────────
#
# play.py's flags and the GUI's New Play Session dialog share these defaults
# (the dialog's fields are the PLAY_TOOL flags, keyed by dest; see
# launcher_config.py). The shipped matchup is the release's recommended one: a
# league deck on both seats against the AZ net searching on a 25-minute bo3
# match clock, with the analysis window open on the GUI board.

BOARD_GUI = "gui"
BOARD_TUI = "tui"
BOARD_TEXT = "text"
BOARD_CHOICES = (BOARD_GUI, BOARD_TUI, BOARD_TEXT)
DEFAULT_BOARD = BOARD_GUI
DEFAULT_PLAY_OPPONENT = "az:gen"
DEFAULT_PLAY_DECK_A = "league/bug"
DEFAULT_PLAY_DECK_B = "league/ur_delver"
DEFAULT_PLAY_WORLDS = 8
DEFAULT_PLAY_MATCH_CLOCK = 1500.0       # 25 min of thinking for the whole bo3
# The analysis window (GUI board only): --analysis/--no-analysis defaults to
# None = on for the GUI board (the other boards have no analysis window).
DEFAULT_ANALYSIS_EVALUATOR = "az:gen"
DEFAULT_ANALYSIS_WORLDS = 4
DEFAULT_ANALYSIS_CAP = 2000
# Evaluator torch devices a search / analysis net may run on (unset = the
# ROBOMAGE_EVAL_DEVICE environment variable, else cpu).
EVAL_DEVICE_CHOICES = ("cpu", "cuda")
# bench-actor's --player-b value for pure self-play (the net on both seats).
BENCH_PLAYER_SELF = "self"
# The analysis browser's (analysis.py browse / the GUI's New Analysis Session)
# default inspected deck.
DEFAULT_BROWSE_DECK_A = "league/ur_delver"


def resolve_board(board):
    """The board to run: ``board``, except gui falls back to tui (with a
    printed notice) when PySide6 is not installed. Shared by play.py and
    analysis.py browse."""
    if board != BOARD_GUI:
        return board
    try:
        import PySide6  # noqa: F401
    except ImportError:
        print("PySide6 not installed — falling back to the TUI board "
              "(pip install -r train/requirements-gui.txt for the GUI).",
              flush=True)
        return BOARD_TUI
    return BOARD_GUI


# ── Analysis browser source (analysis.py browse --source) ────────────────────
#
# The browser's one input selects what it pages through:
#   simulate (the default)   simulate --games games of --player-a (the
#                            inspected model) vs --player-b on --deck-a/-b
#   a directory              recorded shards (shard_*.npz — AZ self-play, a GUI
#                            recording): --player-a is the V(s) net, --seat
#                            the viewpoint, --no-net keeps the recorded z
#   a .rmtrace file          a saved analysis session: --player-a is the net
#                            for the replay search / probes
# Each browse flag applies to the source kinds in BROWSE_SOURCE_DESTS; setting
# one on another kind is an error (browse_inapplicable_dests).

BROWSE_SOURCE_SIMULATE = "simulate"
BROWSE_KIND_SIMULATE = "simulate"
BROWSE_KIND_SHARDS = "shards"
BROWSE_KIND_TRACE = "trace"
BROWSE_BOARD_CHOICES = (BOARD_TUI, BOARD_GUI)
DEFAULT_BROWSE_BOARD = BOARD_TUI
TRACE_EXT = ".rmtrace"               # a saved analysis session (gui_session_io)
SHARD_GLOB = "shard_*.npz"

_SIM_ONLY = (BROWSE_KIND_SIMULATE,)
BROWSE_SOURCE_DESTS = {
    "player_b": _SIM_ONLY, "deck_a": _SIM_ONLY, "deck_b": _SIM_ONLY,
    "seed": _SIM_ONLY, "format": _SIM_ONLY,
    "sims": _SIM_ONLY, "worlds": _SIM_ONLY, "think_time": _SIM_ONLY,
    "search_procs": _SIM_ONLY, "match_clock": _SIM_ONLY,
    "search_device": _SIM_ONLY, "search_xw": _SIM_ONLY,
    "games": (BROWSE_KIND_SIMULATE, BROWSE_KIND_SHARDS),
    "seat": (BROWSE_KIND_SHARDS,), "no_net": (BROWSE_KIND_SHARDS,),
}


def browse_source_kind(source):
    """``simulate`` | ``shards`` | ``trace`` for a browse ``--source`` value
    (None = simulate). Raises ValueError naming what a source may be."""
    import glob
    if source in (None, "", BROWSE_SOURCE_SIMULATE):
        return BROWSE_KIND_SIMULATE
    if source.lower().endswith(TRACE_EXT):
        if not os.path.isfile(source):
            raise ValueError(f"--source {source}: no such {TRACE_EXT} file")
        return BROWSE_KIND_TRACE
    if os.path.isdir(source):
        if not glob.glob(os.path.join(source, SHARD_GLOB)):
            raise ValueError(f"--source {source}: the directory holds no "
                             f"{SHARD_GLOB} files")
        return BROWSE_KIND_SHARDS
    raise ValueError(f"--source {source!r} is not '{BROWSE_SOURCE_SIMULATE}', "
                     f"a shard/recording directory, or a {TRACE_EXT} file "
                     "(the model to inspect is --player-a)")


def browse_inapplicable_dests(kind, explicit):
    """The explicitly-set browse dests that do not apply to source ``kind``,
    in a stable order."""
    return sorted(d for d in explicit
                  if d in BROWSE_SOURCE_DESTS and kind not in BROWSE_SOURCE_DESTS[d])


# ── Removed flags / env vars ──────────────────────────────────────────────────
#
# A flag that was renamed or folded into another must ERROR with a pointer to
# its replacement rather than vanish (argparse's "unrecognized arguments" says
# nothing about where the knob went) or silently alias. Each entry names the old
# flag, the hint appended to "`--old` was removed; ", and the scopes it applies
# to: () = every parser; otherwise tool keys ("train", "play", "harness", …) or
# "tool/sub" ("train/observe") — scope a removal when the same spelling is still
# a live flag somewhere else. Every parser built through ``apply_to_parser``
# registers the entries in its scope as hidden (help-suppressed) options; a
# standalone parser calls ``add_removed_flags(parser, scope)`` itself. The table
# is separate from the Tool specs, so removed flags never render in the TUI.
# When a scoped entry and an unscoped one name the same flag, the scoped entry
# wins in its scope (a different hint where the flag meant something else). An
# entry whose name has no leading dashes is a removed POSITIONAL: the parser
# gets a hidden optional positional that errors if anything fills it.

@dataclass(frozen=True)
class RemovedFlag:
    flag: str
    hint: str
    scopes: tuple = ()

    @property
    def is_positional(self) -> bool:
        return not self.flag.startswith("-")


# Seat vocabulary: every single-seat deck is --deck-a / --deck-b and every seat
# agent (an opponents.make_controller spec) is --player-a / --player-b.
REMOVED_FLAGS = (
    RemovedFlag("--bo1", "use --format bo1"),
    RemovedFlag("--bo3", "use --format bo3 (the default)"),
    RemovedFlag("--deck", "use --deck-a (player A's deck; --deck-b is player B's)"),
    RemovedFlag("--deck", "use --decks (the comma-separated focus deck pool)",
                scopes=("train/az",)),
    RemovedFlag("--deck", "az-train fits the one generalist on the pooled "
                          "az_data/gen shard window and takes no deck",
                scopes=("train/az-train",)),
    RemovedFlag("--opponent", "use --deck-b (player B's deck)"),
    RemovedFlag("--opponent", "use --player-b (the opponent agent spec)",
                scopes=("analysis",)),
    RemovedFlag("--opponent", "use --opponents (the comma-separated opponent "
                              "deck pool)",
                scopes=("train/sweep", "train/az")),
    RemovedFlag("--human-deck", "use --deck-a (with --player-a human)"),
    RemovedFlag("--model-deck", "use --deck-b (the deck --player-b pilots)"),
    RemovedFlag("--model", "use --player-b SPEC (the opponent seat; "
                           "--player-a is human by default)", scopes=("play",)),
    RemovedFlag("--scripted", "use --player-b scripted", scopes=("play",)),
    RemovedFlag("--player", "use --player-a human or --player-b human",
                scopes=("play",)),
    RemovedFlag("--gui", "use --board gui (the default)", scopes=("play",)),
    RemovedFlag("--tui", "use --board tui", scopes=("play",)),
    RemovedFlag("model", "use --player-a SPEC",
                scopes=("analysis", "train/baseline")),
    # The harness's seat agents are --player-a / --player-b; --play / --actions
    # stay as the both-seat script that runs before them.
    RemovedFlag("--scripted", "use --player-a scripted --player-b scripted",
                scopes=("harness",)),
    RemovedFlag("--scripted-spec", "use --player-a / --player-b with the tier "
                                   "spec (e.g. scripted:easy, explore)",
                scopes=("harness",)),
    RemovedFlag("--interactive", "use --player-a human (and/or --player-b human)",
                scopes=("harness",)),
    RemovedFlag("--play-a", "use --player-a \"play:<spec,spec,...>\"",
                scopes=("train/observe",)),
    RemovedFlag("--play-b", "use --player-b \"play:<spec,spec,...>\"",
                scopes=("train/observe",)),
    # Every game/match count is --games; the PUCT constant is --c-puct.
    RemovedFlag("--n-games", "use --games"),
    # The bench scripts' flags, renamed to the az-* / training vocabulary.
    RemovedFlag("--scripted", "use --player-b scripted",
                scopes=("train/bench-actor",)),
    RemovedFlag("--device", "use --actor-device", scopes=("train/bench-actor",)),
    RemovedFlag("--cross", "the cross-world leg runs by default "
                           "(--no-cross-world skips it)",
                scopes=("train/bench-actor",)),
    RemovedFlag("--counts", "use --workers (comma-separated worker counts)",
                scopes=("train/bench-workers",)),
    RemovedFlag("--repeats", "use --exhaustive-repeats",
                scopes=("train/bench-workers",)),
    RemovedFlag("--train-window", "use --window", scopes=("train/bench-workers",)),
    RemovedFlag("--train-batches", "use --batches",
                scopes=("train/bench-workers",)),
    RemovedFlag("--train-deck", "use --deck-a (the az-eval leg's deck)",
                scopes=("train/bench-workers",)),
    RemovedFlag("--envs", "use --n-envs (comma-separated n_envs values)",
                scopes=("train/bench-nenvs",)),
    # az_inspect: one --shards DIR (absent = weights only), one long name per
    # count, and the folded az_embed_viz / sb_shard_report flags.
    RemovedFlag("--with-shards", "use --shards DIR (e.g. train/az_data/gen)",
                scopes=("az-inspect",)),
    RemovedFlag("--no-shards", "omit --shards (absent = weights only)",
                scopes=("az-inspect",)),
    RemovedFlag("--with-counts", "use --shards DIR (occurrence counts come "
                                 "from that shard directory)",
                scopes=("az-inspect",)),
    RemovedFlag("-k", "use --neighbors", scopes=("az-inspect/neighbors",)),
    RemovedFlag("-k", "use --knn", scopes=("az-inspect/structure",)),
    RemovedFlag("-k", "use --clusters", scopes=("az-inspect/clusters",)),
    RemovedFlag("--cluster-seed", "use --seed", scopes=("az-inspect/clusters",)),
    RemovedFlag("--rows", "use --block-rows", scopes=("az-inspect/blocks",)),
    RemovedFlag("--max-rows", "the embedding views sample --count-rows states "
                              "for their occurrence counts",
                scopes=tuple(f"az-inspect/{s}" for s in
                             ("neighbors", "structure", "clusters", "project",
                              "occur"))),
    RemovedFlag("--map-top", "use --top", scopes=("az-inspect/drift",)),
    RemovedFlag("--dir", "use --shards", scopes=("az-inspect/sbreport",)),
    RemovedFlag("--last", "use --window", scopes=("az-inspect/sbreport",)),
    # analysis.py browse: one --source picks simulate / shards / a saved trace.
    RemovedFlag("--shards", "use --source DIR (the shard or recording "
                            "directory to browse)",
                scopes=("analysis/browse",)),
)

# Subcommands that were folded into another: (tool key, name) -> hint appended
# to "`name` was removed; ". ``add_removed_subcommands`` registers each on its
# tool's argparse subparsers as a hidden command that errors with the hint.
REMOVED_SUBCOMMANDS = {
    ("analysis", "interactive"): "use `analysis.py browse` (its analyses menu "
                                 "has every former REPL view, the chart "
                                 "PNGs and the text transcript)",
    ("analysis", "search"): "use `analysis.py report --player-a az:gen "
                            "[--workers N]` (search-vs-net sections of the "
                            "HTML report) or `analysis.py browse` with a "
                            "search --player-a (its `net KL` / `net V vs "
                            "search` probes); --sims/--worlds are flags, "
                            "c-puct and the sideboard budget are spec knobs "
                            "(az:gen?c=1.5&sb_branches=4)",
}

# Environment variables that duplicated a flag: name -> hint appended to
# "environment variable NAME was removed; ".
REMOVED_ENV_VARS = {}


def _scope_matches(entry_scopes, scopes) -> bool:
    return not entry_scopes or any(s in entry_scopes for s in scopes)


def removed_flags_for(*scopes):
    """The ``RemovedFlag`` entries that apply to any of ``scopes`` — a scoped
    entry shadows an unscoped one for the same flag."""
    hits = [r for r in REMOVED_FLAGS if _scope_matches(r.scopes, scopes)]
    scoped = {r.flag for r in hits if r.scopes}
    return [r for r in hits if r.scopes or r.flag not in scoped]


def removed_flag_message(r: RemovedFlag) -> str:
    if r.is_positional:
        return f"the positional {r.flag.upper()} argument was removed; {r.hint}"
    return f"{r.flag} was removed; {r.hint}"


def removed_flag_hint(flag: str, *scopes):
    """The error message for ``flag`` if it is a removed flag in ``scopes``,
    else None (lets non-argparse consumers — curriculum plans — say the same
    thing the CLI does)."""
    for r in removed_flags_for(*scopes):
        if r.flag == flag:
            return removed_flag_message(r)
    return None


def _removed_flag_action(entry: RemovedFlag):
    import argparse

    class _Removed(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            # An absent optional positional is "called" with its default.
            if entry.is_positional and values is None:
                setattr(namespace, self.dest, None)
                return
            parser.error(removed_flag_message(entry))
    return _Removed


def add_removed_flags(parser, *scopes) -> None:
    """Register every removed flag in ``scopes`` (plus the global ones) on
    ``parser`` as a hidden option whose use errors with the replacement hint,
    and fail at startup on any removed environment variable that is set."""
    import argparse
    check_removed_env(parser)
    for r in removed_flags_for(*scopes):
        dest = f"_removed_{r.flag.lstrip('-').replace('-', '_')}"
        if r.is_positional:
            parser.add_argument(dest, nargs="?", default=None,
                                help=argparse.SUPPRESS,
                                action=_removed_flag_action(r))
        else:
            parser.add_argument(r.flag, nargs="?", help=argparse.SUPPRESS,
                                dest=dest, action=_removed_flag_action(r))


def removed_subcommand_message(tool_key, name):
    """The error for removed subcommand ``name`` of ``tool_key``, else None."""
    hint = REMOVED_SUBCOMMANDS.get((tool_key, name))
    return None if hint is None else f"`{name}` was removed; {hint}"


def add_removed_subcommands(subparsers, tool_key) -> None:
    """Register ``tool_key``'s removed subcommands on an argparse subparsers
    action (after the live ones): each is a hidden command that errors with
    its hint whatever follows it. The usage line keeps listing only the live
    subcommands."""
    import argparse
    live = list(subparsers.choices)
    for (key, name), _hint in REMOVED_SUBCOMMANDS.items():
        if key != tool_key:
            continue
        msg = removed_subcommand_message(key, name)

        class _Removed(argparse.Action):
            def __call__(self, parser, namespace, values, option_string=None,
                         _msg=msg):
                parser.error(_msg)

        sp = subparsers.add_parser(name, add_help=False)
        sp.add_argument("_removed_rest", nargs=argparse.REMAINDER,
                        action=_Removed)
    subparsers.metavar = "{" + ",".join(live) + "}"


def check_removed_env(parser=None, environ=None) -> None:
    """Exit with an error naming the replacement if any removed environment
    variable is set. Through ``parser.error`` when a parser is given."""
    environ = os.environ if environ is None else environ
    for name, hint in REMOVED_ENV_VARS.items():
        if name in environ:
            msg = f"environment variable {name} was removed; {hint}"
            if parser is not None:
                parser.error(msg)
            raise SystemExit(f"error: {msg}")


# ── Spec dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Arg:
    """One CLI argument.

    ``name`` with a leading ``--`` is an optional flag; otherwise it is a
    positional.  ``kind`` is one of ``str``, ``int``, ``float``, ``flag``
    (store_true), ``bool`` (a ``--name`` / ``--no-name`` pair; default True,
    False, or None = unset), or ``choice`` (requires ``choices``).
    """
    name: str
    kind: str = "str"
    default: object = None
    choices: tuple = ()
    required: bool = False
    help: str = ""
    metavar: str = None
    suggest: str = None   # autocomplete source: "deck" | "league_deck" | "checkpoint" | None
    multi: bool = False   # TUI: render a suggest-tagged arg as a multi-select (comma-joined)

    @property
    def is_positional(self) -> bool:
        return not self.name.startswith("-")

    @property
    def dest(self) -> str:
        """The argparse attribute name this argument resolves to."""
        return self.name.lstrip("-").replace("-", "_")


@dataclass
class MutexGroup:
    """A mutually-exclusive group of flags (rendered as one Select in the TUI).

    ``label`` is the short TUI row name; ``help`` shows in the field-help pane."""
    args: list
    required: bool = False
    label: str = "opp-mode"
    help: str = "Mutually exclusive — (neither) accepts the default"


@dataclass
class Sub:
    """A subcommand: a list of Arg/MutexGroup items plus an execution mode.

    ``mode`` is ``capture`` (stream output into the TUI log pane) or
    ``interactive`` (the command needs the real terminal — TUI suspends).
    """
    name: str
    help: str = ""
    items: list = field(default_factory=list)
    mode: str = "capture"
    tool: str = None     # owning Tool.key, set by Tool (removed-flag scoping)

    @property
    def scopes(self) -> tuple:
        """Removed-flag scopes this subcommand's parser answers to."""
        if self.tool is None:
            return ()
        return (self.tool, f"{self.tool}/{self.name}")


@dataclass
class Tool:
    """A script exposing one or more subcommands."""
    key: str
    script: str          # path relative to repo root, e.g. "train/train.py"
    subs: list = field(default_factory=list)
    default_sub: str = None  # subcommand assumed when none is given (train)
    flat: bool = False   # True when the script has a flat parser (no subcommand
                         # token in argv, e.g. play.py / test_harness.py)

    def __post_init__(self):
        for s in self.subs:
            s.tool = self.key


# ── Reusable argument groups (mirror the helper functions in the scripts) ─────

def format_arg(help_extra: str = "") -> Arg:
    """--format bo1|bo3: the match format, shared by every tool that plays
    games. Default bo3 everywhere; read it back with ``is_bo3(args)``."""
    return Arg("--format", "choice", choices=FORMAT_CHOICES,
               default=DEFAULT_FORMAT,
               help="Match format: bo3 = best-of-three matches (deck swap + "
                    "sideboarding between games), bo1 = single games "
                    f"(default {DEFAULT_FORMAT})" + help_extra)


def common_args(binary_default=BINARY):
    """Args shared by every train.py subcommand (was train.py _add_common).

    ``binary_default`` lets a subcommand override the engine binary default —
    PPO training subcommands pass ``INTERACTIVE_BINARY`` (release-by-default;
    see its definition above) since debug's assertions and -O0 make training
    noticeably slower; observe (game watching, not training) keeps the plain
    ``BINARY`` (debug-by-default) since it's usually paired with a debug
    engine while iterating on a card/rule."""
    return [
        Arg("--binary", "str", default=binary_default, help="Path to robomage binary"),
        format_arg(),
    ]


def train_opts():
    """Args shared by training subcommands (was train.py _add_train_opts)."""
    return [
        Arg("--total-timesteps", "int", default=TOTAL_TIMESTEPS,
            help="Total training timesteps"),
        Arg("--tally", "flag", help="Print A/B win tally after each rollout"),
        Arg("--fresh", "flag",
            help="Start the generalist from scratch instead of auto-resuming "
                 "the existing gen__final.zip / newest gen__v*.zip (overwrites it)"),
        Arg("--n-envs", "int", default=None,
            help="Number of parallel environments (default: %d, self-play: %d)"
                 % (N_ENVS, N_ENVS_SELF_PLAY)),
        Arg("--no-shaping", "flag",
            help="Disable all shaping rewards (forces shaping_scale=0, skips annealing)"),
        Arg("--auto-sideboard", "flag",
            help="Auto-skip sideboard phase in bo3 (model never sees sideboard decisions)"),
        Arg("--embed-dim", "int", default=EMBED_DIM,
            help="Feature-extractor embed dim for fresh models (default: %d). "
                 "Ignored when resuming a checkpoint (its embed_dim is restored from "
                 "the saved policy_kwargs)." % EMBED_DIM),
        Arg("--stock-head", "flag",
            help="Build fresh models with the legacy stock MlpPolicy positional "
                 "head instead of the default per-action logit head "
                 "(PerActionMaskablePolicy). The flavors are not "
                 "checkpoint-compatible; resuming always keeps the checkpoint's "
                 "own head, so this only affects fresh (--fresh / first-time) "
                 "models."),
        Arg("--popart", "flag",
            help="Per-archetype-bucket PopArt value normalization (default OFF). "
                 "The multi-head critic already isolates each matchup class in the "
                 "last layer; PopArt additionally keeps a running (mu, sigma) of "
                 "each bucket's returns and predicts normalized values, so a "
                 "high-variance matchup can't dominate the SHARED torso's value "
                 "gradients. Output-preserving (the head column is rescaled on "
                 "every stats update), so it is safe to switch on mid-run. "
                 "Incompatible with --stock-head and with clip_range_vf."),
        Arg("--n-epochs", "int", default=N_EPOCHS,
            help="PPO optimization epochs per update (default: %d). Applies to "
                 "fresh models AND overrides whatever a resumed checkpoint was "
                 "saved with, for this session only." % N_EPOCHS),
        Arg("--clip-range", "float", default=CLIP_RANGE,
            help="PPO policy clip range (default: %.2f). Applies to fresh models "
                 "AND overrides whatever a resumed checkpoint was saved with, for "
                 "this session only." % CLIP_RANGE),
    ]


def train_opts_except(*dests):
    """``train_opts()`` minus the named arg dests.

    Lets a training subcommand keep the shared knobs while naming its step budget
    differently (the exploiter's budget is ``--steps``, so it drops
    ``--total-timesteps`` rather than offering two budget flags)."""
    skip = set(dests)
    return [a for a in train_opts() if a.dest not in skip]


def train_opts_only(*dests):
    """The named ``train_opts()`` args — lets bench-nenvs build its throwaway
    model with the same knobs (and defaults) a training run takes."""
    keep = set(dests)
    return [a for a in train_opts() if a.dest in keep]


def parse_int_list(text, flag: str) -> list:
    """A bench sweep's comma-separated positive ints -> list. Exits with an
    error naming ``flag`` when the list is empty or holds a bad value."""
    try:
        values = [int(t) for t in str(text).split(",") if t.strip()]
    except ValueError:
        raise SystemExit(f"error: {flag}: expected comma-separated integers, "
                         f"got {text!r}")
    if not values or any(v < 1 for v in values):
        raise SystemExit(f"error: {flag}: expected one or more integers >= 1, "
                         f"got {text!r}")
    return values


def _opponent_mode():
    """The --self-play | --scripted mutually-exclusive pair (train/sweep)."""
    return MutexGroup([
        Arg("--self-play", "flag",
            help="Train against a frozen snapshot of the generalist piloting the "
                 "opponent deck (gen__v*.zip / gen__final.zip; falls back to the "
                 "scripted agent if none exists yet)"),
        Arg("--scripted", "flag",
            help="Train against the rule-based scripted agent (the default; "
                 "mutually exclusive with --self-play)"),
    ], label="opp-mode",
       help="Opponent mode — mutually exclusive (default: neither = scripted)")


def opponent_pool_opts():
    """Mixed opponent-pool args shared by the train and sweep subcommands."""
    return [
        Arg("--opponent-pool", "str", default=None,
            help="Comma-separated mix of opponent controllers to randomize per "
                 "episode, e.g. 'scripted:easy,scripted:hard=2,mav'. "
                 "The token 'random-model' expands to a random generalist snapshot "
                 "(gen__v*.zip / gen__final.zip) piloting the opponent's deck. Each item may "
                 "carry an optional '=<weight>'. Overrides the plain scripted "
                 "opponent (ignored with --self-play). In a sweep the same pool "
                 "is applied to every matchup, resolving 'random-model' per matchup."),
        Arg("--opponent-ckpt-ratio", "float", default=1.0,
            help="Cap on unique opponent checkpoints kept resident, as a ratio of "
                 "n_envs (default 1.0 -> <=1 checkpoint per env process). Scripted "
                 "agents don't count toward the cap."),
    ]


def _actor_mode():
    """The --actor | --no-actor self-play backend pair (az-selfplay / az).

    Default (neither) is AUTO: use the C++ ``bin/az_actor`` iff it is built, else
    the pure-Python multiprocess backend."""
    return MutexGroup([
        Arg("--actor", "flag",
            help="Force the C++ az_actor self-play backend (error if bin/az_actor "
                 "is not built). Default AUTO: use it iff it is built."),
        Arg("--no-actor", "flag",
            help="Force the pure-Python self-play backend, skipping the actor even "
                 "if bin/az_actor is built."),
    ], label="actor",
       help="Self-play backend — (neither) = AUTO: the C++ az_actor iff it is "
            "built, else the pure-Python backend")


def _actor_device():
    """The --actor-device pass-through (az-selfplay / az / az-league): the C++
    actor's ``--device`` (Stage A of docs/gpu_selfplay_inference_plan.md)."""
    return Arg("--actor-device", "str", default="cpu",
               help="Eval device for the C++ actor's net forwards: cpu (default) "
                    "or cuda (the Radeon under the ROCm torch build; the launcher "
                    "exports HSA_OVERRIDE_GFX_VERSION/HIP_VISIBLE_DEVICES for the "
                    "actor processes). Python-backend games ignore it. With "
                    "--eval-server it selects the SERVER's device instead "
                    "(cpu -> the server still defaults to cuda).")


def _eval_server():
    """The --eval-server | --no-eval-server pair (az-selfplay / az /
    az-league): Stage C central inference (docs/gpu_selfplay_inference_plan.md).
    Default (neither) is AUTO: start a cuda server iff the box has a usable
    GPU, else fall back to local-CPU actors with a printed notice."""
    return MutexGroup([
        Arg("--eval-server", "flag",
            help="Force the central train/az_eval_server.py: one GPU-owning "
                 "server, every C++ actor evaluates leaves over its Unix "
                 "socket (fleet-wide forward batches). Error if it cannot "
                 "start. Actor backend only; fresh server per generation pass."),
        Arg("--no-eval-server", "flag",
            help="Never start the central server — actors load the net and "
                 "forward locally (on --actor-device)."),
    ], label="eval-server",
       help="Central-inference server — (neither) = AUTO: use a cuda server "
            "iff it starts (no usable GPU -> local-CPU actors, with a notice)")


def _c_puct():
    """--c-puct (az-selfplay / az / az-league / az-eval): the PUCT constant."""
    return Arg("--c-puct", "float", default=DEFAULT_AZ_C_PUCT,
               help=f"PUCT exploration constant (default {DEFAULT_AZ_C_PUCT}): "
                    "higher weights search Q over the net prior, so the value "
                    "signal — not the prior — steers the visit distribution. "
                    "Passed to both the Python search and the C++ actor")


def _epoch_frac():
    """--epoch-frac (az-train / az / az-league): auto-batches epoch fraction."""
    return Arg("--epoch-frac", "float", default=DEFAULT_AZ_EPOCH_FRAC,
               help="ROW-UNIFORM sampling only (--rows-per-game 0): with "
                    "batches=0 (AUTO), train this fraction of one epoch over "
                    f"the loaded window (default {DEFAULT_AZ_EPOCH_FRAC}). "
                    "A full epoch (1.0) memorizes the window's small set of "
                    "distinct games; explicit --batches ignores this. Inert "
                    "under the default game-uniform sampler, whose batch count "
                    "comes from --rows-per-game")


def _rows_per_game():
    """--rows-per-game (az-train / az / az-league): game-uniform sampler cap."""
    return Arg("--rows-per-game", "int", default=DEFAULT_AZ_ROWS_PER_GAME,
               help="Game-uniform training sampler: each cycle takes at most "
                    "this many rows from every game in the window (uniform "
                    "within the game, no replacement), shuffles the pool and "
                    "trains one pass over it, so every game's outcome label "
                    "carries the same weight whatever its length (default "
                    f"{DEFAULT_AZ_ROWS_PER_GAME}); the pool size sets the "
                    "AUTO batch count. 0 = the old row-uniform draw governed "
                    "by --epoch-frac")


def _full_search_frac():
    """--full-search-frac (az-selfplay / az / az-league): playout-cap coin."""
    return Arg("--full-search-frac", "float",
               default=DEFAULT_AZ_FULL_SEARCH_FRAC,
               help="Playout-cap randomization: fraction of searched in-game "
                    "roots that get the FULL --sims budget and record a pi "
                    f"policy target (default {DEFAULT_AZ_FULL_SEARCH_FRAC}); "
                    "the rest run a fast --fast-sims search that picks the "
                    "move but records no pi (value targets record from every "
                    "row). Multiplies distinct games per engine budget; 1.0 "
                    "restores every-root-full. Sideboard roots are exempt")


def _fast_sims():
    """--fast-sims (az-selfplay / az / az-league): playout-cap fast budget."""
    return Arg("--fast-sims", "int", default=DEFAULT_AZ_FAST_SIMS,
               help="PUCT sims (TOTAL across --worlds) for the fast searches "
                    "of the playout cap — the in-game roots the "
                    "--full-search-frac coin does not pick "
                    f"(default {DEFAULT_AZ_FAST_SIMS})")


def _opp_pool_frac():
    """--opp-pool-frac (az-selfplay / az / az-league): opponent-pool mixing."""
    return Arg("--opp-pool-frac", "float", default=DEFAULT_AZ_OPP_POOL_FRAC,
               help="Fraction of the pure-self-play matches whose OPPONENT "
                    "seat is piloted by an older checkpoint (the incumbent "
                    "plus the newest snapshots distinct from it and the "
                    f"generator, up to {DEFAULT_AZ_OPP_POOL_SIZE} nets) "
                    "instead of the learner itself; only the learner seat's "
                    "decisions are recorded "
                    f"(default {DEFAULT_AZ_OPP_POOL_FRAC}; 0 = pure mirror "
                    "self-play, also the silent fallback when no distinct "
                    "older checkpoint exists yet)")


def _gate_max_rounds():
    """--gate-max-rounds (az-eval / az / az-league): the sequential gate's cap."""
    return Arg("--gate-max-rounds", "int", default=DEFAULT_AZ_GATE_MAX_ROUNDS,
               help="Hard cap on the sequential gate's panel ROUNDS (default "
                    f"{DEFAULT_AZ_GATE_MAX_ROUNDS}; 1 = a single fixed "
                    "panel). Rounds of --games matches are scheduled over the "
                    "panel and the SPRT is re-asked after every completed "
                    "match, stopping (and cutting the legs still in flight) "
                    "the moment it decides; at the cap the incumbent keeps the "
                    "seat unless the score reached the promote bar, and a "
                    "score inside the indifference region is UNDECIDED (kept, "
                    "not a failed gate)")


def _gate_alpha():
    """--gate-alpha (az-eval / az / az-league): the SPRT's error rates."""
    return Arg("--gate-alpha", "float", default=DEFAULT_AZ_GATE_ALPHA,
               help="Symmetric SPRT error rates (alpha=beta, default "
                    f"{DEFAULT_AZ_GATE_ALPHA}): the chance of promoting a "
                    "candidate that is really at H0 and of keeping one that is "
                    "really at H1. Lower = stricter and slower (roughly doubles "
                    "the matches per verdict per halving)")


def _gate_floor_min():
    """--gate-floor-min (az-eval / az / az-league): floor-veto sample floor."""
    return Arg("--gate-floor-min", "int", default=DEFAULT_AZ_GATE_FLOOR_MIN,
               help="Minimum candidate matches on a piloted deck before the "
                    f"per-deck floor veto may fire (default "
                    f"{DEFAULT_AZ_GATE_FLOOR_MIN}). Rounds accumulate, so a "
                    "long sequential gate powers the veto up as it goes; set 0 "
                    "with --gate-floor 0 to drop the veto entirely and let the "
                    "aggregate test carry the verdict alone")


def _no_gate_shards():
    """--no-gate-shards (az / az-league): kill switch for gate-shard pooling."""
    return Arg("--no-gate-shards", "flag",
               help="Do NOT record the gate's candidate-vs-incumbent matches "
                    "into the training pool. Default ON (actor-backend gates "
                    "record both nets' searched decisions as shards and pool "
                    "them into az_data/gen): cross-net games are the signal "
                    "pure self-play cannot provide — each net's mistakes get "
                    "punished by a DIFFERENT policy")


def _no_cross_world():
    """--no-cross-world (az-selfplay / az / az-league): Stage 0 kill switch."""
    return Arg("--no-cross-world", "flag",
               help="Disable the actor's cross-world batched leaf evaluation "
                    "(default ON — visits are arithmetically identical to the "
                    "sequential search, ~1.7-3.3x per decision; see "
                    "docs/gpu_selfplay_inference_plan.md Stage 0)")


# n-step TD knobs. Two sides of one scheme, so their help lives in one place and
# is reused verbatim by every subcommand that exposes them: --td-n is a
# GENERATION knob (it is baked into the shard's td_q column at pack time), --q-mix
# a TRAINING knob (how much of the value target that column supplies).
_TD_N_HELP = (
    "n-step TD horizon for the recorded value target (default "
    f"{DEFAULT_AZ_TD_N}): a sample's td_q bootstraps off the search ROOT VALUE of "
    "the decision n samples later in the same game, negated when that decision's "
    "mover is the other seat. The chain is SHORTENED at the next exploratory "
    "(non-argmax) move, and a window that reaches the end of the game uses the "
    "true outcome z instead. Generation-side: it is stored in the shard.")
_Q_MIX_HELP = (
    "Weight of the shard's n-step TD target in the value loss (default "
    f"{DEFAULT_AZ_Q_MIX}): v_target = (1 - q_mix) * z + q_mix * td_q. 0 = the "
    "classic pure-outcome AlphaZero target, 1 = pure bootstrap. Training-side: "
    "it re-weights already-recorded shards, so it can be changed without "
    "regenerating self-play.")


def append_spec_knob(spec: str, key, value) -> str:
    """Append ``key=value`` to a controller spec's ``?query`` knobs.

    Appended LAST so it wins over any ``key=`` already present in the spec
    (later keys overwrite in opponents.py's knob parser). Shared by play.py's
    and analysis.py's convenience flags (--think-time/--match-clock/...)."""
    return spec + ("&" if "?" in spec else "?") + f"{key}={value}"


# ── Search knobs (flags folded into an az:/mcts: spec's ?query) ───────────────
#
# One home for the search-seat convenience flags every interactive front end
# offers — play.py, the analysis browser, and the GUI dialogs that mirror them.
# Each flag dest maps to the spec-query key make_controller parses; the fold is
# ``apply_search_knobs``. A flag applies to EVERY seat whose spec is a search
# spec, appended last so it overrides the same key already in the spec (put the
# knobs in the specs themselves for per-seat budgets).

# flag dest -> spec query key, in the order the knobs are appended.
SEARCH_KNOB_KEYS = (("sims", "sims"), ("worlds", "worlds"),
                    ("think_time", "time"), ("search_procs", "procs"),
                    ("match_clock", "clock"), ("search_xw", "xw"),
                    ("search_device", "device"), ("paced", "paced"))
SEARCH_SPEC_PREFIXES = ("az:", "mcts:")


def is_search_spec(spec) -> bool:
    """True for an agent spec that runs a tree search (az:/mcts:), which is
    what the search knobs apply to. azraw: is the raw AZ policy (no search),
    so it — like scripted tiers and PPO models — takes no search knobs."""
    return isinstance(spec, str) and spec.strip().lower().startswith(
        SEARCH_SPEC_PREFIXES)


def search_knob_args(*, worlds=None, match_clock=None, paced=False):
    """The search-knob flags. ``worlds`` / ``match_clock`` are the tool's
    defaults for those two (play's shipped matchup sets both); ``paced`` adds
    --paced/--no-paced (human-facing play only). --search-procs unset means
    AUTO (half the visible cores, capped at the world count) wherever the flag
    is offered."""
    budget = [
        Arg("--think-time", "float", default=None,
            help="Search seats only (an az:/mcts: player spec): wall-clock "
                 "seconds per decision — the search runs as many simulations "
                 "as fit in this budget (more time = stronger play), "
                 "overriding sims as the terminator"),
        Arg("--match-clock", "float", default=match_clock,
            help="Search seats only: whole-match chess-clock bank in seconds "
                 "(1500 = 25 min for a bo3); each decision draws a variable "
                 "budget from it — harder decisions earn more time, obvious "
                 "ones stop early; 0 = no clock. Mutually exclusive with "
                 "--sims"
                 + (f" (default {match_clock:g}; an explicit --sims drops it)"
                    if match_clock is not None else "")),
    ]
    args = [
        Arg("--sims", "int", default=None,
            help="Search seats only: MCTS simulations per decision"),
        Arg("--worlds", "int", default=worlds,
            help="Search seats only: determinized worlds per decision (sims "
                 "are split across them; default "
                 + (f"{worlds}" if worlds is not None
                    else "the spec's own, 4") + ")"),
        budget[0],
        Arg("--search-procs", "int", default=None,
            help="Search seats only: engine processes to fan the determinized "
                 "worlds across (world-parallel; more procs = more sims per "
                 "decision in the same wall-clock). Default AUTO: half the "
                 "visible cores, capped at the world count"),
        budget[1],
        Arg("--search-xw", "bool", default=True,
            help="Search seats only: cross-world batched leaf evaluation — one "
                 "net forward per round over every world's leaf. Visit counts "
                 "are identical to the sequential search (pure speed); "
                 "--no-search-xw only to debug"),
        Arg("--search-device", "choice", choices=EVAL_DEVICE_CHOICES,
            default=None,
            help="Search seats only: torch device for the search net's "
                 "forwards — cpu, or cuda (the Radeon under the ROCm torch "
                 "build). Unset = ROBOMAGE_EVAL_DEVICE, else cpu. The GPU pays "
                 "together with cross-world batching and higher world counts"),
    ]
    if paced:
        args.append(Arg(
            "--paced", "bool", default=None,
            help="Search opponent only: mask response-timing tells — a small "
                 "jittered (~0.02-0.05s) floor on every decision plus "
                 "occasional 0.2-0.5s fake-think pauses when the opponent was "
                 "never even offered a decision. Unset = on whenever the "
                 "search has a variable budget (--match-clock/--think-time); "
                 "--no-paced forces instant obvious decisions"))
    return args


def search_knob_pairs(values, *, auto_procs=True):
    """``[(query_key, value)]`` for a dict of search-knob dests (see
    ``SEARCH_KNOB_KEYS``; absent / None dests are omitted).

    ``auto_procs``: an unset ``search_procs`` becomes AUTO — half the visible
    cores, capped at the world count in effect (the spec grammar's own default
    stays procs=1 so gates/eval stay reproducible). ``search_xw`` appends only
    its off position (the controller batches by default). ``paced`` is folded
    only when the dict carries the key: True/False wins, None turns pacing on
    whenever the search has a variable time budget (think time / clock). A
    ``match_clock`` of 0 is no clock."""
    values = dict(values)
    if not values.get("match_clock"):
        values["match_clock"] = None
    if auto_procs and values.get("search_procs") is None:
        from opponents import default_search_procs, DEFAULT_SEARCH_WORLDS
        worlds = values.get("worlds")
        values["search_procs"] = default_search_procs(
            worlds if worlds is not None else DEFAULT_SEARCH_WORLDS)
    xw = values.get("search_xw")
    values["search_xw"] = 0 if xw is False else None
    if "paced" in values:
        paced = values["paced"]
        if paced is None:
            paced = (values.get("think_time") is not None
                     or values.get("match_clock") is not None)
        values["paced"] = int(bool(paced))
    return [(key, values[dest]) for dest, key in SEARCH_KNOB_KEYS
            if values.get(dest) is not None]


def with_spec_query(spec: str, pairs) -> str:
    """Append ``pairs`` to a controller spec's ``?k=v&…`` query (later keys
    win in make_controller's parser, so appending is always safe)."""
    for key, value in pairs:
        spec = append_spec_knob(spec, key, value)
    return spec


def apply_search_knobs(spec, values, *, auto_procs=True):
    """``spec`` with the search-knob ``values`` folded into its query when it
    is a search spec; any other spec is returned unchanged."""
    if not is_search_spec(spec):
        return spec
    return with_spec_query(spec, search_knob_pairs(values,
                                                   auto_procs=auto_procs))


def spec_query_keys(spec) -> set:
    """The knob keys a controller spec's ``?k=v&…`` query already carries."""
    if not isinstance(spec, str) or "?" not in spec:
        return set()
    query = spec.split("?", 1)[1]
    return {part.split("=", 1)[0].strip().lower()
            for part in query.split("&") if part.strip()}


# ── Deck scan (the "deck" suggestion source) ─────────────────────────────────

DECKS_DIR = os.path.join(REPO_ROOT, "bin", "resources", "decks")
# Deck subfolders hidden from deck pickers: temp/ holds auto-generated test
# decks (see test_harness.py), not_used/ parked development stubs.
DECK_SCAN_EXCLUDE = frozenset({"temp", "not_used"})
# League decks live in their own folder so the league roster is curated
# separately from the top-level training decks. A deck here is referenced as
# 'league/<stem>' (a path relative to decks/), which the engine resolves to
# decks/league/<stem>.dk.
LEAGUE_DECKS_DIR = os.path.join(DECKS_DIR, "league")


def league_decks():
    """The league roster: every decks/league/*.dk as 'league/<stem>', sorted
    (empty when the folder is missing). The one default roster the PPO league,
    exploiter, baseline sweep, az self-play and the TUI deck picker share."""
    if not os.path.isdir(LEAGUE_DECKS_DIR):
        return []
    return sorted("league/" + os.path.splitext(p)[0]
                  for p in os.listdir(LEAGUE_DECKS_DIR) if p.endswith(".dk"))


def scan_decks():
    """All .dk decks under decks/ (recursive), as decks/-relative stems.

    Subfolder decks are offered in the 'league/ur_delver' path-relative form
    that train.py and the engine accept alongside top-level stems like
    'delver'. temp/ and not_used/ are excluded. Sorted top-level first, then
    grouped by subfolder, alphabetical within each group. Shared by the TUI
    form and the GUI dialogs so both offer the same decks."""
    out = []
    for root, dirs, files in os.walk(DECKS_DIR):
        dirs[:] = sorted(d for d in dirs if d not in DECK_SCAN_EXCLUDE)
        rel_dir = os.path.relpath(root, DECKS_DIR).replace(os.sep, "/")
        for fname in files:
            if fname.endswith(".dk"):
                stem = os.path.splitext(fname)[0]
                out.append(stem if rel_dir == "." else f"{rel_dir}/{stem}")
    return sorted(out, key=lambda rel: (rel.count("/"), rel))


def sb_search_args():
    """The bo3 sideboard plan-search budget flags, shared verbatim by every Sub
    that runs searches over bo3 matches (az-selfplay / az / az-league). One home — the defaults are the DEFAULT_SB_*
    constants above; see their comment block for the design."""
    return [
        Arg("--sb-branches", "int", default=DEFAULT_SB_BRANCHES,
            help="Extra sideboard plans per bo3 sideboard root beyond the "
                 "coverage pass (one plan per legal first pick); each plan is "
                 "priced on every --sb-worlds world "
                 f"(default {DEFAULT_SB_BRANCHES})"),
        Arg("--sb-worlds", "int", default=DEFAULT_SB_WORLDS,
            help="Determinized worlds at a bo3 sideboard root "
                 f"(default {DEFAULT_SB_WORLDS})"),
        Arg("--sb-rollout-turns", "int", default=DEFAULT_SB_ROLLOUT_TURNS,
            help="Rollout horizon pricing each sideboard plan, in player turns "
                 "of the next game (0 = static decklist read; default "
                 f"{DEFAULT_SB_ROLLOUT_TURNS})"),
    ]


def sb_selfplay_args():
    """The prior-rollout sideboard self-play flags (GENERATION side), shared
    by az-selfplay / az / az-league. One home — defaults are the
    DEFAULT_SB_SELFPLAY_* / DEFAULT_SB_EXPLORE_* constants above."""
    return [
        Arg("--sb-selfplay-mode", "choice", choices=("prior", "plan"),
            default=DEFAULT_SB_SELFPLAY_MODE,
            help="Sideboard roots during self-play GENERATION: 'prior' = "
                 "sample boarding picks from the net prior with exploration "
                 "and record one-hot behavior rows (trained on next-game z); "
                 "'plan' = the multi-world plan search (always used by "
                 f"gates/eval/play; default {DEFAULT_SB_SELFPLAY_MODE})"),
        Arg("--sb-explore-temp", "float", default=DEFAULT_SB_EXPLORE_TEMP,
            help="prior^(1/temp) sampling temperature at prior-mode sideboard "
                 f"roots (default {DEFAULT_SB_EXPLORE_TEMP})"),
        Arg("--sb-explore-eps", "float", default=DEFAULT_SB_EXPLORE_EPS,
            help="Uniform-mix exploration floor over live actions at "
                 f"prior-mode sideboard roots (default {DEFAULT_SB_EXPLORE_EPS})"),
    ]


def sb_train_args():
    """The sideboard-row trainer flags (az-train / az / az-league). One home —
    defaults are DEFAULT_SB_BATCH_FRAC / DEFAULT_SB_LOSS_COEF above."""
    return [
        Arg("--sb-batch-frac", "float", default=DEFAULT_SB_BATCH_FRAC,
            help="Share of each az-train batch drawn from sideboard-phase "
                 f"rows when any exist (default {DEFAULT_SB_BATCH_FRAC})"),
        Arg("--sb-loss-coef", "float", default=DEFAULT_SB_LOSS_COEF,
            help="Weight of the sideboard REINFORCE policy term "
                 f"(default {DEFAULT_SB_LOSS_COEF})"),
    ]


def sim_args():
    """Common simulation args for analysis.py (was analysis.py _add_sim_args).

    Player A is the INSPECTED model (its value/probs/SHAP fill every view) and
    player B its opponent. The physical seat still alternates per simulated
    game; --deck-a always travels with --player-a."""
    return [
        Arg("--player-a", "str", required=True, suggest="agent",
            help="The model to analyze (player A): 'gen', a .zip path, or "
                 "az:gen/azraw:gen for the generalist AlphaZero net. A SEARCH "
                 "spec (az:/mcts: prefix, e.g. az:gen?sims=128&worlds=4) makes "
                 "the simulated trace games be PLAYED by the real MCTS "
                 "controller, so the browser inspects states arising from "
                 "search-quality play (slow); azraw:gen and a bare PPO spec keep "
                 "raw-policy traces. The inspection net (value/probs/SHAP) is "
                 "the same either way."),
        Arg("--player-b", "str", default="scripted", suggest="agent",
            help="Opponent (player B): 'gen', a model .zip path, az:gen/azraw:gen, "
                 "or 'scripted' for the rule-based agent piloting --deck-b "
                 "(the default)"),
        Arg("--deck-a", "str", default=None, suggest="deck",
            help="Player A's deck (.dk stem) — the deck the inspected model "
                 "pilots. REQUIRED for a model seat: the one generalist encodes "
                 "no deck in its filename."),
        Arg("--deck-b", "str", default=None, suggest="deck",
            help="Player B's deck (.dk stem). REQUIRED for a model opponent (the "
                 "generalist encodes no deck); a scripted opponent defaults to a "
                 "mirror match (--deck-a)."),
        Arg("--binary", "str", default=INTERACTIVE_BINARY, help="Path to robomage binary"),
        format_arg(),
        Arg("--seed", "int", default=1,
            help="Base seed: simulated game N (counted from 0 over the "
                 "session) plays engine --seed seed+N, so a run is "
                 "reproducible (default: 1)"),
        Arg("--out", "str", default=None,
            help="Directory for saved charts/reports (default: train/analysis_out/)"),
        Arg("--show", "flag",
            help="Also open charts in a GUI window (needs a local display)"),
    ]


# ── Tool definitions ──────────────────────────────────────────────────────────

TRAIN_TOOL = Tool("train", "train/train.py", default_sub="train", subs=[
    Sub("train", "Train the one generalist model (default command)", items=[
        Arg("--deck-a", "str", default="delver", suggest="deck",
            help="Deck the generalist (player A) plays this session (.dk stem, "
                 "default: delver). Always saved to the single gen__final.zip; "
                 "sessions on any deck/opponent accumulate onto that one generalist."),
        Arg("--deck-b", "str", required=True, suggest="deck",
            help="Opponent deck (player B) this session trains against (.dk stem). The model "
                 "stays one generalist — training continues the same gen__final.zip "
                 "rather than forging a per-deck or matchup-specific model."),
        Arg("--load", "str", default=None, suggest="checkpoint",
            help="Resume from a specific checkpoint .zip ('gen' or a path), "
                 "overriding the default auto-resume of gen__final.zip"),
        _opponent_mode(),
        *opponent_pool_opts(),
        *train_opts(),
        *common_args(binary_default=INTERACTIVE_BINARY),
    ]),
    Sub("league", "PFSP league: train one generalist model per deck vs the whole field", items=[
        Arg("--resume", "flag",
            help="Resume an interrupted league run from its saved progress "
                 "(checkpoints/_league_progress.json, rewritten on every snapshot). "
                 "Restores the roster, total budget, rotation, and all hyperparameters "
                 "from the sidecar — other flags are ignored when set."),
        Arg("--decks", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated deck roster to train + sample opponents from "
                 "(default: every deck in decks/league/, referenced as "
                 "'league/<stem>'). Roster ORDER is the training rotation order. "
                 "In the TUI, pick multiple with space; reorder the highlighted "
                 "deck in the rotation with [ / ]."),
        Arg("--self-play-frac", "float", default=LEAGUE_SELF_PLAY_FRAC,
            help="Probability of facing the latest snapshot of the learner's own "
                 "deck (OpenAI-Five 'play the latest self' slot; default %.2f). "
                 "Auto-ramped down while few snapshots exist." % LEAGUE_SELF_PLAY_FRAC),
        Arg("--scripted-anchor-frac", "float", default=LEAGUE_SCRIPTED_ANCHOR_FRAC,
            help="Minimum share of the historical-pool branch reserved for the "
                 "scripted anchor so it never vanishes (default %.2f)." % LEAGUE_SCRIPTED_ANCHOR_FRAC),
        Arg("--exploiter-floor", "float", default=LEAGUE_EXPLOITER_FLOOR,
            help="Minimum share of the historical-pool branch reserved for the "
                 "archetype exploiters (exp_<arch>__*.zip from 'train.py exploiter'), "
                 "so their styles stay in the field once the learner starts beating "
                 "them; they also take part in the normal PFSP weighting. 0 disables "
                 "the floor (default %.2f)." % LEAGUE_EXPLOITER_FLOOR),
        Arg("--pfsp-mode", "choice", choices=("pfsp", "softmax"), default="pfsp",
            help="Opponent quality weighting: 'pfsp' = (1-winrate)^p (AlphaStar) or "
                 "'softmax' = exp(q) with OpenAI-Five quality updates (default pfsp)."),
        Arg("--pfsp-p", "float", default=LEAGUE_PFSP_P,
            help="PFSP exponent p in (1-winrate)^p (default %.1f)." % LEAGUE_PFSP_P),
        Arg("--softmax-eta", "float", default=LEAGUE_SOFTMAX_ETA,
            help="Softmax quality learning rate eta (default %.3f)." % LEAGUE_SOFTMAX_ETA),
        Arg("--snapshot-every", "int", default=LEAGUE_SNAPSHOT_EVERY,
            help="Save a frozen gen__v{steps}.zip snapshot every N steps "
                 "(default %d)." % LEAGUE_SNAPSHOT_EVERY),
        Arg("--promote-margin", "float", default=LEAGUE_PROMOTE_MARGIN,
            help="Only keep a snapshot when the learner's recent-window win-rate "
                 ">= 0.5 + margin (negative gates below 0.5, e.g. -0.1 -> 0.40; the "
                 "first snapshot of each deck is exempt so self-play can bootstrap; "
                 "0 disables the gate; default %.2f)." % LEAGUE_PROMOTE_MARGIN),
        Arg("--fixed-self-deck", "flag",
            help="Restore the classic one-deck-per-rotation mode: the learner's OWN "
                 "deck is fixed for a whole rotation (adaptive-boost rotations, "
                 "focus-deck stats). Default (off) is mixed mode, where the learner's "
                 "self deck also cycles per episode across the rotation's deck set, so "
                 "one rollout trains every deck as pilot."),
        Arg("--rotate-every", "int", default=LEAGUE_ROTATE_EVERY,
            help="Steps to train one learner deck before rotating to the next "
                 "in fixed-self-deck mode; in mixed mode, the fixed chunk length "
                 "between snapshot/sidecar boundaries (default %d)." % LEAGUE_ROTATE_EVERY),
        Arg("--adaptive-boost", "float", default=LEAGUE_ADAPTIVE_BOOST,
            # argparse %-expands help at display time, so a literal percent sign
            # must stay doubled ('%%') in the final string — hence the f-string
            # (old-style '%' interpolation would collapse it and argparse then
            # chokes on '% o' in 'or'; Python 3.14 raises at add_argument time).
            help="Max rotation-length multiplier for catch-up decks: a rotation "
                 "stretches toward boost x --rotate-every as the deck's last league "
                 "win-rate falls below 50%% or its trained steps trail the roster "
                 "leader. Rotation order is unchanged (no deck is starved). "
                 f"1 = fixed-length rotations (default {LEAGUE_ADAPTIVE_BOOST:.1f})."),
        Arg("--shard", "str", default=None, metavar="i/n",
            help="Distributed training: train only roster slice i of n (0-indexed, "
                 "strided) while still sampling opponents from the FULL roster. Run "
                 "one shard per machine over a shared/synced checkpoint dir "
                 "(docs/distributed_league_training.md). Each shard keeps its own "
                 "progress sidecar (_league_progress.shard{i}of{n}.json); pass the "
                 "same --shard together with --resume. Omit for single-machine "
                 "training."),
        Arg("--train-decks", "str", default=None, metavar="A,B,...",
            help="Distributed training: explicit comma-separated subset of --decks "
                 "that THIS driver trains (rotates over), overriding the strided "
                 "--shard slice while opponents still span the full --decks roster. "
                 "The web distribution UI (scripts/league_agent.py) sets this "
                 "per machine for arbitrary deck-to-machine assignment; --shard is "
                 "still passed alongside for the sidecar tag. Omit to use the "
                 "strided slice."),
        Arg("--opponent-ckpt-ratio", "float", default=1.0,
            help="Cap on unique opponent checkpoints kept resident, as a ratio of "
                 "n_envs (default 1.0 -> <=1 checkpoint per env process)."),
        *train_opts(),
        *common_args(binary_default=INTERACTIVE_BINARY),
    ]),
    Sub("exploiter",
        "Train a dedicated ARCHETYPE EXPLOITER vs the frozen generalist "
        "(saved as exp_<archetype>__*.zip; never touches gen)", items=[
        Arg("--archetype", "choice", choices=tuple(ARCHETYPES), required=True,
            help="Archetype to exploit WITH: the learner pilots this archetype's "
                 "decks (from decks/archetypes.json) against the frozen generalist "
                 "piloting the whole roster. Saved under the stem "
                 "exp_<archetype> (exp_burn__v{steps}.zip / exp_burn__final.zip)."),
        Arg("--steps", "int", default=EXPLOITER_STEPS,
            help="Step budget for this exploiter run (default %d)." % EXPLOITER_STEPS),
        Arg("--resume", "flag",
            help="Resume this archetype's interrupted exploiter run from its saved "
                 "progress (checkpoints/_exploiter_<archetype>_progress.json, "
                 "rewritten on every snapshot). Restores the budget and all "
                 "hyperparameters from the sidecar — other flags are ignored "
                 "(--archetype is still required: it selects the sidecar)."),
        Arg("--decks", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated roster the FROZEN OPPONENT pilots (default: every "
                 "deck in decks/league/). The learner's own decks always come from "
                 "the archetype tag, never from this list."),
        Arg("--chunk-steps", "int", default=EXPLOITER_CHUNK,
            help="Steps per progress chunk: the run is trained in chunks of this "
                 "size so the sidecar/snapshots advance and --resume re-enters "
                 "mid-run (default %d)." % EXPLOITER_CHUNK),
        Arg("--scripted-anchor-frac", "float", default=LEAGUE_SCRIPTED_ANCHOR_FRAC,
            help="Share of episodes played against the scripted anchor rather than "
                 "the frozen generalist (collapse guard; default %.2f)."
                 % LEAGUE_SCRIPTED_ANCHOR_FRAC),
        Arg("--pfsp-mode", "choice", choices=("pfsp", "softmax"), default="pfsp",
            help="Weighting across the frozen opponent's decks: 'pfsp' = "
                 "(1-winrate)^p (AlphaStar) or 'softmax' = exp(q) (default pfsp). "
                 "Concentrates the exploiter on the roster decks it loses to."),
        Arg("--pfsp-p", "float", default=LEAGUE_PFSP_P,
            help="PFSP exponent p in (1-winrate)^p (default %.1f)." % LEAGUE_PFSP_P),
        Arg("--softmax-eta", "float", default=LEAGUE_SOFTMAX_ETA,
            help="Softmax quality learning rate eta (default %.3f)." % LEAGUE_SOFTMAX_ETA),
        Arg("--snapshot-every", "int", default=LEAGUE_SNAPSHOT_EVERY,
            help="Save a frozen exp_<archetype>__v{steps}.zip snapshot every N "
                 "steps (default %d)." % LEAGUE_SNAPSHOT_EVERY),
        Arg("--promote-margin", "float", default=0.0,
            help="Only keep a snapshot when the exploiter's recent-window win-rate "
                 ">= 0.5 + margin (default 0.0 = keep every snapshot; an exploiter "
                 "is worth pooling even below 50%%, so the gate is off by default)."),
        Arg("--fresh", "flag",
            help="Start the exploiter from RANDOM weights instead of warm-starting "
                 "from the generalist (gen__final.zip / newest gen__v*.zip). An "
                 "existing exp_<archetype> checkpoint always wins over both — this "
                 "flag only affects the FIRST run of an archetype's exploiter."),
        *train_opts_except("total_timesteps", "fresh"),
        *common_args(binary_default=INTERACTIVE_BINARY),
    ]),
    Sub("curriculum",
        "Run / resume a multi-phase training PLAN (league, exploiter, az, "
        "az-league, baseline phases) from one JSON file", items=[
        Arg("--plan", "str", required=True, suggest="curriculum",
            help="Curriculum plan to run: a name under "
                 "train/checkpoints/curricula/ (e.g. 'q3_archetypes' -> "
                 "q3_archetypes.plan.json) or a path to a .plan.json file. Each "
                 "phase is a train.py subcommand with its own arguments; "
                 "progress is tracked in <name>.progress.json next to the plan. "
                 "In the TUI, the 'curriculum' entry opens a plan builder."),
        Arg("--resume", "flag",
            help="Continue an interrupted curriculum from its progress file: "
                 "completed phases are skipped and the phase that was in flight "
                 "is relaunched (with --resume of its own when the subcommand "
                 "supports it). Refuses to run if an already-executed phase was "
                 "edited; phases still ahead may be freely rewritten."),
        Arg("--status", "flag",
            help="Print each phase's state (pending/running/done/failed, steps "
                 "done, last gate result) from the progress file and exit."),
        Arg("--dry-run", "flag",
            help="Print the command each phase would run and exit — the way to "
                 "check a plan's composed argv before spending GPU-days on it."),
    ]),
    Sub("sweep", "PFSP sweep: train the generalist on one deck vs a pool of the other decks", items=[
        Arg("--deck-a", "str", required=True, suggest="deck",
            help="Deck to train on (.dk stem). Always saved to gen__final.zip; this "
                 "session accumulates onto the one generalist, same as 'train'."),
        Arg("--opponents", "str", default=None, suggest="deck", multi=True,
            help="Comma-separated pool of opponent decks to sample from via PFSP "
                 "(default: every other deck in bin/resources/decks/). Like league's "
                 "roster, but this pool is opponents only — --deck-a is never rotated "
                 "into training and never part of the pool."),
        Arg("--self-play-frac", "float", default=LEAGUE_SELF_PLAY_FRAC,
            help="Probability of facing the latest snapshot of --deck-a itself (the "
                 "'play the latest self' slot; default %.2f). Auto-ramped down while "
                 "few snapshots exist." % LEAGUE_SELF_PLAY_FRAC),
        Arg("--scripted-anchor-frac", "float", default=LEAGUE_SCRIPTED_ANCHOR_FRAC,
            help="Minimum share of the historical-pool branch reserved for the "
                 "scripted anchor so it never vanishes (default %.2f)." % LEAGUE_SCRIPTED_ANCHOR_FRAC),
        Arg("--pfsp-mode", "choice", choices=("pfsp", "softmax"), default="pfsp",
            help="Opponent quality weighting: 'pfsp' = (1-winrate)^p (AlphaStar) or "
                 "'softmax' = exp(q) with OpenAI-Five quality updates (default pfsp)."),
        Arg("--pfsp-p", "float", default=LEAGUE_PFSP_P,
            help="PFSP exponent p in (1-winrate)^p (default %.1f)." % LEAGUE_PFSP_P),
        Arg("--softmax-eta", "float", default=LEAGUE_SOFTMAX_ETA,
            help="Softmax quality learning rate eta (default %.3f)." % LEAGUE_SOFTMAX_ETA),
        Arg("--snapshot-every", "int", default=LEAGUE_SNAPSHOT_EVERY,
            help="Save a frozen gen__v{steps}.zip snapshot every N steps "
                 "(default %d)." % LEAGUE_SNAPSHOT_EVERY),
        Arg("--promote-margin", "float", default=LEAGUE_PROMOTE_MARGIN,
            help="Only keep a snapshot when --deck-a's recent-window win-rate "
                 ">= 0.5 + margin (negative gates below 0.5, e.g. -0.1 -> 0.40; the "
                 "first snapshot is exempt so self-play can bootstrap; 0 disables "
                 "the gate; default %.2f)." % LEAGUE_PROMOTE_MARGIN),
        Arg("--opponent-ckpt-ratio", "float", default=1.0,
            help="Cap on unique opponent checkpoints kept resident, as a ratio of "
                 "n_envs (default 1.0 -> <=1 checkpoint per env process)."),
        *train_opts(),
        *common_args(binary_default=INTERACTIVE_BINARY),
    ]),
    Sub("fixed-model", "Train --deck-a vs a fixed (never-reloaded) opponent model", items=[
        Arg("--deck-a", "str", default="delver", suggest="deck",
            help="Deck the trained model (player A) plays (.dk stem)"),
        Arg("--deck-b", "str", required=True, suggest="deck",
            help="Deck the frozen opponent model (player B) plays (.dk stem)"),
        Arg("--load", "str", default=None, suggest="checkpoint",
            help="Resume from checkpoint .zip ('gen' or a path)"),
        *train_opts(),
        *common_args(binary_default=INTERACTIVE_BINARY),
    ]),
    Sub("alternate", "Swap which side is trained every N timesteps", items=[
        Arg("--deck-a", "str", default="delver", suggest="deck",
            help="Player A's deck (.dk stem; trained first)"),
        Arg("--deck-b", "str", required=True, suggest="deck",
            help="Player B's deck (.dk stem)"),
        Arg("--every", "int", required=True, metavar="N",
            help="Swap the trained side every N timesteps"),
        *train_opts(),
        *common_args(binary_default=INTERACTIVE_BINARY),
    ]),
    Sub("observe",
        "Observe game(s) between any pair of {scripted | model} controllers; "
        "also the fuzz campaign (--player-a/-b explore --verbose --out FILE) "
        "and the engine throughput benchmark (--quiet --timing)", items=[
        Arg("--player-a", "str", default="scripted", suggest="agent",
            help="Player A controller: 'scripted' (or 'scripted:*'), the "
                 "'explore' / 'explore:patient' coverage fuzzer, 'gen', a model "
                 ".zip path, az:gen/azraw:gen/mcts:gen, or a semantic action "
                 "script \"play:cast:Lightning Bolt,target:Grizzly Bears@opp,pass\" "
                 "(action_spec.py grammar; passes / first choice once it runs "
                 "out) (default: scripted)"),
        Arg("--player-b", "str", default="scripted", suggest="agent",
            help="Player B controller (see --player-a; default: scripted)"),
        Arg("--deck-a", "str", default="delver", suggest="deck", help="Player A deck (.dk stem, default: delver)"),
        Arg("--deck-b", "str", default=None, suggest="deck", help="Player B deck (.dk stem, default: Player A's deck)"),
        Arg("--games", "int", default=1,
            help="Matches to run — single games under --format bo1 (default: 1). "
                 ">1 prints per-match results and a W/L/D summary"),
        Arg("--seed", "int", default=1,
            help="Base RNG seed (game N uses seed+N; default: 1)"),
        Arg("--verbose", "flag",
            help="Dump full board state (battlefield, hands, mana, stack, graveyards) at each decision"),
        Arg("--quiet", "flag",
            help="Print no transcript, only a one-line W/L/D summary (a draw "
                 "is still announced and its log saved to draw_<stamp>.txt)"),
        Arg("--out", "str", default=None, metavar="FILE",
            help="Write the transcript (per-game results and W/L/D summary "
                 "included) to FILE and print a one-line W/L/D summary to "
                 "stdout — the fuzz-campaign form: --player-a explore "
                 "--player-b explore --verbose --out FILE"),
        Arg("--max-decisions", "int", default=None, metavar="N",
            help="Stop each game/match after N decisions (reported as "
                 "incomplete; default: run to completion)"),
        Arg("--timing", "flag",
            help="Print engine throughput after the run (games, decisions, "
                 "wall time, games/s, decisions/s, ms/decision). With --quiet "
                 "the engine also runs without narrative — the lean "
                 "benchmark path"),
        *common_args(),
    ]),
    Sub("baseline",
        "Evaluate --player-a vs --player-b (default: the AZ generalist under full "
        "search, C++ actor, vs scripted:hard) over the league matchup grid or one "
        "--deck-a/--deck-b cell; report appended to checkpoints/baseline_report.log",
        items=[
        Arg("--player-a", "str", default=DEFAULT_BASELINE_MODEL, suggest="agent",
            help="Agent under test — player A (the two players alternate "
                 f"physical seats). Default {DEFAULT_BASELINE_MODEL} = the "
                 "incumbent gen__azfinal.pt under search. An 'az:' spec or a .pt "
                 "path runs on the C++ actor when --player-b is scripted:hard; "
                 "any other pair (a PPO 'gen'/.zip, an 'mcts:'/'azraw:' spec, "
                 "or a non-scripted --player-b) runs on the Python backend"),
        Arg("--player-b", "str", default=DEFAULT_BASELINE_OPPONENT, suggest="agent",
            help="The reference agent — player B (default "
                 f"{DEFAULT_BASELINE_OPPONENT}). Any agent spec: e.g. --player-a "
                 "mcts:gen --player-b gen measures what search adds over the "
                 "raw policy (the search A/B gate)"),
        Arg("--games", "int", default=DEFAULT_BASELINE_GAMES,
            help=f"Matches per matchup — single games under --format bo1 "
                 f"(default {DEFAULT_BASELINE_GAMES}); seats "
                 "alternate within each matchup (player A in seat A for the "
                 "first half, rounded up)"),
        Arg("--deck-a", "str", default=None, suggest="deck",
            help="Restrict the grid to this deck piloted by --player-a (a "
                 "mirror match unless --deck-b names player B's deck). "
                 "Default: every league deck piloted vs every league deck — the "
                 "full N×N grid, mirrors included"),
        Arg("--deck-b", "str", default=None, suggest="deck",
            help="Restrict player B to this deck (alone: every league deck vs "
                 "it; with --deck-a: that one cell)"),
        Arg("--all", "flag",
            help="Force the full league grid even when --deck-a/--deck-b are "
                 "given (the grid is already the default without them)"),
        Arg("--mirrors", "flag",
            help="Only the grid's diagonal: every league deck piloted by both "
                 "players (one leg per deck, all sharing the run's single eval "
                 "server)"),
        Arg("--sims", "int", default=DEFAULT_AZ_SIMS,
            help=f"PUCT simulations per decision, TOTAL across --worlds (default "
                 f"{DEFAULT_AZ_SIMS}, the league budget). This and --worlds / "
                 "--c-puct / --sb-* apply to every search seat (az:/mcts:/.pt) "
                 "whose spec does not carry that ?knob itself"),
        Arg("--worlds", "int", default=DEFAULT_AZ_WORLDS,
            help=f"Determinized worlds per search (default {DEFAULT_AZ_WORLDS})"),
        _c_puct(),
        *sb_search_args(),
        Arg("--workers", "int", default=DEFAULT_BASELINE_WORKERS,
            help="Actor legs (each one engine + search process) or Python "
                 "workers in flight at once (default "
                 f"{DEFAULT_BASELINE_WORKERS}). The Python backend splits a "
                 "matchup into contiguous game chunks when there are fewer "
                 "matchups than workers"),
        Arg("--log", "str", default=None,
            help="Report file (default: checkpoints/baseline_report.log, appended)"),
        Arg("--record-dir", "str", default=None,
            help="Record the net's searched decisions as trainer-schema shards "
                 "(shard_net_<seat>__<deck>__<opp>_*.npz, one flat directory) for "
                 "az_inspect / the shard browsers. Actor backend only. Default: a "
                 "fresh dir under az_data/baseline/ — OUTSIDE the az_data/gen "
                 "training pool, so a baseline never becomes training data"),
        Arg("--no-record", "flag", help="Do not record shards"),
        Arg("--td-n", "int", default=DEFAULT_AZ_TD_N,
            help="n-step TD horizon stored in the recorded shards"),
        Arg("--seed", "int", default=1,
            help="Base RNG seed (matchup i uses seed + i*100003; default: 1)"),
        format_arg(),
        _actor_mode(),
        _actor_device(),
        _eval_server(),
        Arg("--binary", "str", default=BINARY,
            help="Path to the robomage binary (Python backend only)"),
    ]),
    # ── AlphaZero (Phase C) ───────────────────────────────────────────────────
    Sub("az-selfplay",
        "Generate AlphaZero self-play data (focus deck vs mirror + roster)", items=[
        Arg("--deck-a", "str", default="delver", suggest="deck",
            help="Focus deck the learner pilots (.dk stem); its opponent is a "
                 "mirror with "
                 "P=--mirror-frac, else a uniform league-roster draw"),
        Arg("--games", "int", default=DEFAULT_AZ_GAMES,
            help="Matches to generate — single games under --format bo1 "
                 f"(default {DEFAULT_AZ_GAMES})"),
        Arg("--sims", "int", default=DEFAULT_AZ_SIMS,
            help="PUCT simulations per decision, TOTAL across --worlds"),
        Arg("--worlds", "int", default=DEFAULT_AZ_WORLDS, help="Determinized worlds per search"),
        _full_search_frac(),
        _fast_sims(),
        _opp_pool_frac(),
        _c_puct(),
        Arg("--workers", "int", default=None,
            help="Worker processes (default max(1, cpu-2))"),
        Arg("--checkpoint", "str", default=None, suggest="az_checkpoint",
            help="AZ (.pt) / PPO (.zip) ckpt or 'gen' (default: generalist AZ "
                 "ckpt, else gen PPO warm-start, else random init)"),
        Arg("--explore-full-turns", "int", default=DEFAULT_AZ_EXPLORE_FULL_TURNS,
            help="Exploration clock: through this game turn (player turns; "
                 "sideboard roots are turn 0) every searched root samples its "
                 "action from the visit distribution"),
        Arg("--explore-decay-turns", "int", default=DEFAULT_AZ_EXPLORE_DECAY_TURNS,
            help="Exploration clock: over the next N game turns the per-root "
                 "sampling probability falls linearly to --explore-floor"),
        Arg("--explore-floor", "float", default=DEFAULT_AZ_EXPLORE_FLOOR,
            help="Exploration clock: per-root sampling probability for the rest "
                 "of the game (0 = argmax after the decay)"),
        Arg("--td-n", "int", default=DEFAULT_AZ_TD_N, help=_TD_N_HELP),
        *sb_search_args(),
        *sb_selfplay_args(),
        Arg("--mirror-frac", "float", default=DEFAULT_AZ_MIRROR_FRAC,
            help="P(opponent deck == focus deck) per game "
                 f"(default {DEFAULT_AZ_MIRROR_FRAC}); else a "
                 "uniform league-roster draw"),
        Arg("--out", "str", default=None, help="Output dir (default az_data/gen)"),
        Arg("--seed", "int", default=None,
            help="Base RNG seed (default: randomly drawn at launch and "
                 "printed, so the run stays reproducible after the fact)"),
        format_arg(" — the pooled az_data/gen window is bo3, so write bo1 "
                   "shards to a separate --out"),
        Arg("--expert", "flag",
            help="Write EXPERT demonstration shards instead of self-play: "
                 "scripted:hard pilots both seats and pi is a one-hot on the "
                 "expert's action (always bo3 to match the pooled shard window, so "
                 "--format bo1 is rejected; "
                 "sims/worlds/checkpoint are ignored)"),
        Arg("--expert-opponent", "str", default=None,
            help="Expert mode only: scripted-agent spec for the OPPONENT seat "
                 "(e.g. scripted:random / scripted:easy). scripted:hard keeps "
                 "the focus seat and ONLY its decisions are recorded — so a "
                 "combo deck's demonstrations come from games it actually "
                 "wins. Default: hard both seats, both recorded"),
        Arg("--merge-dupes", "int", default=1,
            help="Merge interchangeable duplicate menu actions into one search "
                 "edge (decode.menu_merge_reps; default 1, 0 = one edge per copy)"),
        _actor_mode(),
        _actor_device(),
        _eval_server(),
        _no_cross_world(),
    ]),
    Sub("az-train", "Train an AZNet on self-play shards", items=[
        Arg("--batches", "int", default=DEFAULT_AZ_TRAIN_BATCHES, help="Optimizer updates"),
        Arg("--batch-size", "int", default=DEFAULT_AZ_BATCH_SIZE),
        Arg("--lr", "float", default=DEFAULT_AZ_LR),
        Arg("--c-v", "float", default=DEFAULT_AZ_CV, help="Value-loss weight"),
        Arg("--q-mix", "float", default=DEFAULT_AZ_Q_MIX, help=_Q_MIX_HELP),
        Arg("--window", "int", default=DEFAULT_AZ_WINDOW,
            help="Number of most-recent shards to train on"),
        _epoch_frac(),
        _rows_per_game(),
        Arg("--from-ppo", "str", default=None, suggest="checkpoint",
            help="Warm-start from a PPO checkpoint instead of resuming AZ"),
        Arg("--fresh", "flag", help="Start from random init"),
        Arg("--snapshot-every", "int", default=0,
            help="Also save an intermediate gen__azv{steps}.pt every N batches (0=off)"),
        Arg("--seed", "int", default=None,
            help="Init / batch-sampling seed (default: randomly drawn at "
                 "launch and printed)"),
        *sb_train_args(),
    ]),
    Sub("az-eval", "Gate a candidate AZNet vs the incumbent (sequential test, "
                   "MCTS at the training sim budget)", items=[
        Arg("--deck-a", "str", default="delver", suggest="deck",
            help="Focus deck (.dk stem) added to the gate's roster-wide panel"),
        Arg("--candidate", "str", required=True, suggest="az_checkpoint",
            help="Candidate AZ .pt ('gen' or a path)"),
        Arg("--incumbent", "str", default=None, suggest="az_checkpoint",
            help="Incumbent AZ .pt (default: gen__azfinal.pt; scripted if none yet)"),
        Arg("--games", "int", default=DEFAULT_AZ_EVAL_GAMES,
            help="Gate matches per ROUND, split over the roster-wide panel (a "
                 "mirror per roster deck + direction-balanced cross pairs). The "
                 "gate plays rounds until the SPRT decides or --gate-max-rounds "
                 f"is reached (default {DEFAULT_AZ_EVAL_GAMES} = 2 matches per "
                 "panel matchup, the smallest seat-balanced round)"),
        _gate_max_rounds(),
        _gate_alpha(),
        Arg("--sims", "int", default=DEFAULT_AZ_EVAL_SIMS),
        Arg("--worlds", "int", default=DEFAULT_AZ_EVAL_WORLDS),
        Arg("--promote-threshold", "float", default=DEFAULT_AZ_PROMOTE_THRESHOLD,
            help="The sequential test's H1 win-rate; H0 is its mirror "
                 "(1-threshold), so the test is symmetric about 50%% and does "
                 "not favor the incumbent"),
        Arg("--promote", "flag", help="Copy candidate to gen__azfinal.pt if it clears the bar"),
        Arg("--gate-floor", "float", default=DEFAULT_AZ_GATE_FLOOR,
            help="Per-piloted-deck gate floor: a deck the candidate piloted in "
                 ">=--gate-floor-min gate matches whose win-rate deficit vs the "
                 "incumbent on LIKE pairings falls below 2*floor-1 vetoes "
                 "promotion even when the aggregate test accepts (0 disables; "
                 "on mirrors alone this equals win-rate < floor)"),
        _gate_floor_min(),
        Arg("--cross-pairs", "int", default=DEFAULT_AZ_GATE_CROSS_PAIRS,
            help="Seeded cross-deck pairings added to the panel on top of the "
                 "per-deck mirrors, each played in BOTH directions (default "
                 f"{DEFAULT_AZ_GATE_CROSS_PAIRS}); raise it for a broader gate"),
        Arg("--record-dir", "str", default=None,
            help="Record every gate leg's searched decisions (both nets) as "
                 "trainer-schema shards under this directory, one subdir per "
                 "leg (cand_<seat>__r<round>__<deck_x>__<deck_y>), for "
                 "az_inspect / the shard browsers. Actor backend only. Default: "
                 "a fresh dir under az_data/gate (the shards are pooled into "
                 "training unless --no-pool-shards)"),
        Arg("--td-n", "int", default=DEFAULT_AZ_TD_N,
            help="n-step TD horizon stored in the recorded shards (--record-dir)"),
        Arg("--no-pool-shards", "flag",
            help="Do NOT move the gate's recorded shards into the az_data/gen "
                 "training pool afterwards. Default ON, and the gate records by "
                 "default (bo3 + actor backend), so the sequential gate's "
                 "compute is never wasted: however many rounds a verdict costs, "
                 "every one of those candidate-vs-incumbent games becomes "
                 "training data — the cross-net signal pure self-play lacks"),
        _c_puct(),
        Arg("--seed", "int", default=1,
            help="Base gate seed (every round's and matchup's seeds derive "
                 "from it; default: 1)"),
        format_arg(" — the gate's win rate is per match in bo3"),
        Arg("--workers", "int", default=None,
            help="Process-pool fan-out over the gate's matchup panel (default "
                 "max(1, cpu-1), capped at the panel size; 1 = serial). "
                 "Result-identical for any worker count — per-matchup seeds "
                 "and fresh controllers make matchups independent"),
        # Gate backend (mirrors the az-selfplay/az knobs): the C++ actor plays
        # candidate-vs-incumbent two-model matches (--model-b) with cross-world
        # batching and the optional Stage C eval-server (one per net). The
        # no-incumbent vs-scripted fallback always stays on Python.
        _actor_mode(),
        _actor_device(),
        _eval_server(),
        _no_cross_world(),
    ]),
    Sub("az",
        "One AlphaZero cycle (self-play -> train -> eval/gate) over a deck x "
        "opponent matrix (default: whole league; pass one deck in --decks to fix "
        "a focus). "
        "bo3 by default (per-game value target); --format bo1 to opt out",
        items=[
        Arg("--decks", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated FOCUS deck pool the generalist pilots "
                 "(default: every deck in decks/league/). Pass a single deck to "
                 "fix one focus (the classic single-deck cycle)."),
        Arg("--opponents", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated opponent-deck pool for self-play + gating "
                 "(default: every deck in decks/league/). Each focus deck plays "
                 "each; per game the opponent is the mirror with P=--mirror-frac, "
                 "else a uniform draw from this pool."),
        Arg("--games", "int", default=DEFAULT_AZ_GAMES,
            help="Self-play matches this cycle — single games under --format "
                 f"bo1 (default {DEFAULT_AZ_GAMES})"),
        Arg("--sims", "int", default=DEFAULT_AZ_SIMS,
            help="Self-play PUCT sims, TOTAL across --worlds "
                 f"({DEFAULT_AZ_SIMS}/{DEFAULT_AZ_WORLDS} = "
                 f"{DEFAULT_AZ_SIMS // DEFAULT_AZ_WORLDS} per determinized world tree)"),
        Arg("--worlds", "int", default=DEFAULT_AZ_WORLDS),
        _full_search_frac(),
        _fast_sims(),
        _opp_pool_frac(),
        _c_puct(),
        Arg("--td-n", "int", default=DEFAULT_AZ_TD_N, help=_TD_N_HELP),
        *sb_search_args(),
        *sb_selfplay_args(),
        *sb_train_args(),
        Arg("--workers", "int", default=None),
        Arg("--batches", "int", default=DEFAULT_AZ_CYCLE_BATCHES,
            help=f"Optimizer updates this cycle (default {DEFAULT_AZ_CYCLE_BATCHES}). "
                 "0 = AUTO: exactly one epoch over the loaded training window, "
                 "max(1, samples // batch_size) updates, so the epoch count "
                 "cannot drift as the window's data volume changes"),
        _epoch_frac(),
        _rows_per_game(),
        Arg("--batch-size", "int", default=DEFAULT_AZ_BATCH_SIZE),
        Arg("--lr", "float", default=DEFAULT_AZ_LR),
        Arg("--q-mix", "float", default=DEFAULT_AZ_Q_MIX, help=_Q_MIX_HELP),
        Arg("--window", "int", default=DEFAULT_AZ_WINDOW,
            help=f"Training window: newest N shards (default {DEFAULT_AZ_WINDOW}). "
                 "0 = AUTO — 2x "
                 "the shards this cycle's generation writes, so every training "
                 "pass covers exactly this pass plus the previous one"),
        Arg("--exhaustive", "flag",
            help="Replace the random self-play draw with the EXACT matchup "
                 "matrix: one bo3 match vs scripted:hard per ORDERED focus x "
                 "opponent pair (net pilots the focus seat) plus one pure "
                 "self-play match per UNORDERED pair, mirrors included — "
                 "10x10 + 55 = 155 matches on the 10-deck roster, every cell "
                 "exactly once. --games/--mirror-frac/--scripted-opponent-frac "
                 "are ignored. With the actor built the WHOLE matrix runs on "
                 "it (vs-scripted cells via the scripted oracle); --no-actor "
                 "keeps everything on Python"),
        Arg("--exhaustive-selfplay", "flag",
            help="Exhaustive matrix restricted to the pure SELF-PLAY cells "
                 "(implies --exhaustive): one bo3 match per UNORDERED deck "
                 "pair, mirrors included (55 on the 10-deck roster), with NO "
                 "vs-scripted cells — so the whole schedule runs on the C++ "
                 "actor backend and nothing falls back to Python"),
        Arg("--exhaustive-repeats", "int", default=DEFAULT_AZ_EXHAUSTIVE_REPEATS,
            help="Play every cell of the exhaustive matrix N times per cycle "
                 f"(default {DEFAULT_AZ_EXHAUSTIVE_REPEATS}). Each repeat is an "
                 "independent match with its own seat draw and game seed."),
        Arg("--scripted-cells", "int", default=DEFAULT_AZ_SCRIPTED_CELLS,
            help="With --exhaustive-selfplay only: ALSO play K vs-scripted:hard "
                 "matches this cycle, taken from the full ORDERED (focus, "
                 "opponent) pair list — the same list plain --exhaustive plays "
                 "in full — starting at offset (slot * K) %% n_ordered and "
                 "wrapping, so successive az-league slots cover every ordered "
                 f"pair in turn (default {DEFAULT_AZ_SCRIPTED_CELLS}; 0 "
                 "disables). A standalone az cycle is slot 0. These cells are "
                 "marked exactly like --exhaustive's, so the C++ actor plays "
                 "them via the scripted oracle alongside the self-play cells. "
                 "Ignored (with a note) under full --exhaustive."),
        Arg("--eval-games", "int", default=DEFAULT_AZ_EVAL_GAMES,
            help="Gate matches per ROUND, split over the roster-wide panel (a "
                 "mirror per roster deck + direction-balanced cross pairs; "
                 f"default {DEFAULT_AZ_EVAL_GAMES} = 2 per matchup on a 10-deck "
                 "roster, the smallest seat-balanced round). The gate plays "
                 "rounds until its SPRT decides or --gate-max-rounds is hit"),
        _gate_max_rounds(),
        _gate_alpha(),
        Arg("--eval-sims", "int", default=DEFAULT_AZ_EVAL_SIMS),
        Arg("--eval-worlds", "int", default=DEFAULT_AZ_EVAL_WORLDS),
        Arg("--promote-threshold", "float", default=DEFAULT_AZ_PROMOTE_THRESHOLD,
            help="The sequential test's H1 win-rate; H0 is its mirror "
                 "(1-threshold), so the test is symmetric about 50%% and does "
                 "not favor the incumbent"),
        Arg("--gate-floor", "float", default=DEFAULT_AZ_GATE_FLOOR,
            help="Per-piloted-deck gate floor: a deck the candidate piloted in "
                 ">=--gate-floor-min gate matches whose win-rate deficit vs the "
                 "incumbent on LIKE pairings falls below 2*floor-1 vetoes "
                 "promotion (0 disables; on mirrors alone this equals win-rate "
                 "< floor)"),
        _gate_floor_min(),
        _no_gate_shards(),
        Arg("--expert-decks", "str", default=EXPERT_DECKS_ROSTER,
            suggest="league_deck", multi=True,
            help="Comma-separated decks to ALSO write scripted:hard EXPERT "
                 "demonstration shards for each cycle (pi = one-hot expert "
                 "action): behavior-cloning targets for hand-coded combo lines "
                 "(e.g. league/wubg_doomsday) that neither PPO exploration nor "
                 f"prior-guided search discovers. Default '{EXPERT_DECKS_ROSTER}' "
                 "= every deck in decks/league/; pass 'none' (or an empty "
                 "value) to write no expert shards"),
        Arg("--expert-games", "int", default=DEFAULT_AZ_EXPERT_GAMES,
            help="Expert matches per expert deck per cycle"),
        Arg("--expert-opponent", "str", default=None,
            help="Scripted-agent spec for the expert games' OPPONENT seat "
                 "(e.g. scripted:random / scripted:easy). scripted:hard keeps "
                 "the focus seat and ONLY its decisions are recorded — so a "
                 "combo deck's demonstrations come from games it actually "
                 "wins. A comma-separated list plays --expert-games per deck "
                 "against EACH listed opponent (an explicit scripted:hard "
                 "there records the focus seat only). Default: hard both "
                 "seats, both recorded"),
        Arg("--selfplay-exclude", "str", default=None,
            suggest="league_deck", multi=True,
            help="Comma-separated decks to leave OUT of self-play (the "
                 "matrix, the scripted cells, the opponent pool) while the "
                 "gate panel and --expert-decks still cover them — for a deck "
                 "the net cannot pilot yet, whose self-play seat only writes "
                 "z=-1 rows"),
        Arg("--seed", "int", default=None,
            help="Base RNG seed (default: randomly drawn at launch and printed)"),
        Arg("--mirror-frac", "float", default=DEFAULT_AZ_MIRROR_FRAC,
            help="P(opponent deck == focus deck) per self-play game "
                 f"(default {DEFAULT_AZ_MIRROR_FRAC}); "
                 "else a uniform league-roster draw"),
        Arg("--scripted-opponent-frac", "float", default=0.0,
            help="Fraction of self-play games (0..1) whose opponent seat is "
                 "piloted by the rule-based scripted:hard agent while the "
                 "net+MCTS pilots the focus seat (only the net seat's decisions "
                 "become training samples). Forces the Python backend; 1.0 = "
                 "every game vs scripted hard. Default 0 = pure self-play."),
        format_arg(" — self-play + gate; the value target is per game either way"),
        _actor_mode(),
        _actor_device(),
        _eval_server(),
        _no_cross_world(),
    ]),
    Sub("az-league",
        "AlphaZero league: rotate az cycles (self-play -> train -> gate) over the "
        "decks/league/ roster (bo3 by default; --format bo1 to opt out)", items=[
        Arg("--resume", "flag",
            help="Resume an interrupted az-league run from its saved progress "
                 "(checkpoints/_az_league_progress.json, rewritten after each deck "
                 "cycle). Restores the roster, budgets, and all knobs from the "
                 "sidecar — other flags are ignored when set."),
        Arg("--decks", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated deck roster to rotate over (default: every deck in "
                 "decks/league/, referenced 'league/<stem>'). Roster ORDER is the "
                 "rotation order. In the TUI, pick multiple with space."),
        Arg("--rotations", "int", default=1,
            help="Full passes over the roster (0 = run indefinitely until "
                 "interrupted; still resumable via --resume)"),
        Arg("--cycles-per-deck", "int", default=1,
            help="az cycles to run per deck per rotation"),
        Arg("--games", "int", default=DEFAULT_AZ_GAMES,
            help="Self-play matches per cycle — single games under --format "
                 f"bo1 (default {DEFAULT_AZ_GAMES})"),
        Arg("--sims", "int", default=DEFAULT_AZ_SIMS,
            help="Self-play PUCT sims, TOTAL across --worlds "
                 f"({DEFAULT_AZ_SIMS}/{DEFAULT_AZ_WORLDS} = "
                 f"{DEFAULT_AZ_SIMS // DEFAULT_AZ_WORLDS} per determinized world tree)"),
        Arg("--worlds", "int", default=DEFAULT_AZ_WORLDS),
        _full_search_frac(),
        _fast_sims(),
        _opp_pool_frac(),
        _c_puct(),
        Arg("--td-n", "int", default=DEFAULT_AZ_TD_N, help=_TD_N_HELP),
        *sb_search_args(),
        *sb_selfplay_args(),
        *sb_train_args(),
        Arg("--workers", "int", default=None,
            help="Self-play worker processes (default max(1, cpu-2))"),
        Arg("--batches", "int", default=DEFAULT_AZ_CYCLE_BATCHES,
            help=f"Optimizer updates per slot (default {DEFAULT_AZ_CYCLE_BATCHES}). "
                 "0 = AUTO: exactly one epoch over the loaded training window, "
                 "max(1, samples // batch_size) updates, so the epoch count "
                 "cannot drift as the window's data volume changes"),
        _epoch_frac(),
        _rows_per_game(),
        Arg("--batch-size", "int", default=DEFAULT_AZ_BATCH_SIZE),
        Arg("--lr", "float", default=DEFAULT_AZ_LR),
        Arg("--q-mix", "float", default=DEFAULT_AZ_Q_MIX, help=_Q_MIX_HELP),
        Arg("--window", "int", default=DEFAULT_AZ_WINDOW,
            help=f"Training window: newest N shards (default {DEFAULT_AZ_WINDOW}). "
                 "0 = AUTO — 2x "
                 "the shards each slot's generation writes, so every training "
                 "pass covers exactly that pass plus the previous one"),
        Arg("--eval-games", "int", default=DEFAULT_AZ_EVAL_GAMES,
            help="Gate matches per ROUND, split over the roster-wide panel (a "
                 "mirror per roster deck + direction-balanced cross pairs; "
                 f"default {DEFAULT_AZ_EVAL_GAMES} = 2 per matchup on a 10-deck "
                 "roster, the smallest seat-balanced round). The gate plays "
                 "rounds until its SPRT decides or --gate-max-rounds is hit"),
        _gate_max_rounds(),
        _gate_alpha(),
        Arg("--eval-sims", "int", default=DEFAULT_AZ_EVAL_SIMS),
        Arg("--eval-worlds", "int", default=DEFAULT_AZ_EVAL_WORLDS),
        Arg("--promote-threshold", "float", default=DEFAULT_AZ_PROMOTE_THRESHOLD,
            help="The sequential test's H1 win-rate; H0 is its mirror "
                 "(1-threshold), so the test is symmetric about 50%% and does "
                 "not favor the incumbent"),
        Arg("--gate-floor", "float", default=DEFAULT_AZ_GATE_FLOOR,
            help="Per-piloted-deck gate floor: a deck the candidate piloted in "
                 ">=--gate-floor-min gate matches whose win-rate deficit vs the "
                 "incumbent on LIKE pairings falls below 2*floor-1 vetoes "
                 "promotion (0 disables; on mirrors alone this equals win-rate "
                 "< floor)"),
        _gate_floor_min(),
        _no_gate_shards(),
        Arg("--gate-every", "int", default=DEFAULT_AZ_GATE_EVERY,
            help="Run the eval/gate every K slots instead of every slot: the "
                 "candidate accumulates K cycles of training (and "
                 "candidate-generated self-play) between promotions, and the "
                 "gate's wall-clock cost is paid 1/K as often. Candidate "
                 "snapshots still save every slot. 0 = NO gating: no slot is "
                 "ever gated and the final candidate is promoted to "
                 "gen__azfinal UNCONDITIONALLY when the run completes (use a "
                 "finite --rotations)."),
        Arg("--matrix", "flag",
            help="Whole-roster focus MATRIX every slot instead of the per-deck "
                 "focus rotation: each cycle's self-play draws its focus deck "
                 "uniformly from the roster per game, keeping the training "
                 "window stationary (no one-deck-at-a-time forgetting sweep). "
                 "A rotation then counts --cycles-per-deck matrix cycles."),
        Arg("--exhaustive", "flag",
            help="Like --matrix but EXACT: every slot's self-play plays the "
                 "full matchup matrix once — one bo3 match vs scripted:hard "
                 "per ORDERED deck pair (net pilots the focus seat) plus one "
                 "pure self-play match per UNORDERED pair, mirrors included "
                 "(10x10 + 55 = 155 matches on the 10-deck roster). "
                 "--games/--mirror-frac/--scripted-opponent-frac are ignored. "
                 "With the actor built the WHOLE matrix runs on it "
                 "(vs-scripted cells via the scripted oracle); --no-actor "
                 "keeps everything on Python. A rotation counts "
                 "--cycles-per-deck matrix cycles, as with --matrix."),
        Arg("--exhaustive-selfplay", "flag",
            help="Exhaustive matrix restricted to the pure SELF-PLAY cells "
                 "(implies --exhaustive): every slot plays one bo3 match per "
                 "UNORDERED deck pair, mirrors included (55 on the 10-deck "
                 "roster), with NO vs-scripted cells — so every slot's "
                 "self-play runs entirely on the C++ actor backend "
                 "(persisted in the resume sidecar)"),
        Arg("--exhaustive-repeats", "int", default=DEFAULT_AZ_EXHAUSTIVE_REPEATS,
            help="Play every cell of the exhaustive matrix N times per slot "
                 f"(default {DEFAULT_AZ_EXHAUSTIVE_REPEATS}) — e.g. 2 makes each "
                 "slot every self-play matchup twice. Each repeat is an "
                 "independent match with its own seat draw and game seed "
                 "(persisted in the resume sidecar)."),
        Arg("--scripted-cells", "int", default=DEFAULT_AZ_SCRIPTED_CELLS,
            help="With --exhaustive-selfplay only: ALSO play K vs-scripted:hard "
                 "matches per slot, taken from the full ORDERED (focus, "
                 "opponent) pair list — the same list plain --exhaustive plays "
                 "in full — starting at offset (slot * K) %% n_ordered and "
                 "wrapping, so successive slots cover every ordered pair in "
                 f"turn (default {DEFAULT_AZ_SCRIPTED_CELLS}; 0 disables). "
                 "These cells are marked exactly like --exhaustive's, so the "
                 "C++ actor plays them via the scripted oracle alongside the "
                 "self-play cells. Ignored (with a note) under full "
                 "--exhaustive (persisted in the resume sidecar)."),
        Arg("--expert-decks", "str", default=EXPERT_DECKS_ROSTER,
            suggest="league_deck", multi=True,
            help="Comma-separated decks to ALSO write scripted:hard EXPERT "
                 "demonstration shards for each slot (pi = one-hot expert "
                 "action): behavior-cloning targets for hand-coded combo lines "
                 "(e.g. league/wubg_doomsday) that neither PPO exploration nor "
                 f"prior-guided search discovers. Default '{EXPERT_DECKS_ROSTER}' "
                 "= every deck in decks/league/; pass 'none' (or an empty "
                 "value) to write no expert shards"),
        Arg("--expert-games", "int", default=DEFAULT_AZ_EXPERT_GAMES,
            help="Expert matches per expert deck per slot"),
        Arg("--expert-opponent", "str", default=None,
            help="Scripted-agent spec for the expert games' OPPONENT seat "
                 "(e.g. scripted:random / scripted:easy). scripted:hard keeps "
                 "the focus seat and ONLY its decisions are recorded — so a "
                 "combo deck's demonstrations come from games it actually "
                 "wins. A comma-separated list plays --expert-games per deck "
                 "against EACH listed opponent (an explicit scripted:hard "
                 "there records the focus seat only). Persisted in the resume "
                 "sidecar. Default: hard both seats, both recorded"),
        Arg("--selfplay-exclude", "str", default=None,
            suggest="league_deck", multi=True,
            help="Comma-separated decks to leave OUT of every slot's self-play "
                 "(the matrix, the scripted cells, the opponent pool) while "
                 "the gate panel and --expert-decks still cover them — for a "
                 "deck the net cannot pilot yet, whose self-play seat only "
                 "writes z=-1 rows. Persisted in the resume sidecar"),
        Arg("--seed", "int", default=None,
            help="Base RNG seed (slot i uses seed+i; default: randomly drawn "
                 "at launch and printed — a --resume run restores the "
                 "sidecar's recorded seed instead)"),
        Arg("--mirror-frac", "float", default=DEFAULT_AZ_MIRROR_FRAC,
            help="P(opponent deck == focus deck) per self-play game "
                 f"(default {DEFAULT_AZ_MIRROR_FRAC}); "
                 "else a uniform league-roster draw"),
        Arg("--scripted-opponent-frac", "float", default=0.0,
            help="Fraction of self-play games (0..1) whose opponent seat is "
                 "piloted by the rule-based scripted:hard agent while the "
                 "net+MCTS pilots the focus seat (only the net seat's decisions "
                 "become training samples). Forces the Python backend; 1.0 = "
                 "every game vs scripted hard. Default 0 = pure self-play "
                 "(persisted in the resume sidecar)."),
        format_arg(" — every slot's self-play + gate; the value target is per "
                   "game either way (persisted in the resume sidecar)"),
        _actor_mode(),
        _actor_device(),
        _eval_server(),
        _no_cross_world(),
    ]),
    # ── Throughput benchmarks ─────────────────────────────────────────────────
    Sub("bench-actor",
        "Benchmark AZ self-play: the C++ az_actor legs (batch sweep, "
        "cross-world, central eval server) vs the in-process Python "
        "az_selfplay leg on the same net and workload", items=[
        Arg("--deck-a", "str", default="league/ur_delver", suggest="deck",
            help="Player A deck (default league/ur_delver)"),
        Arg("--deck-b", "str", default=None, suggest="deck",
            help="Player B deck (default: mirror = --deck-a)"),
        Arg("--player-b", "choice", default=BENCH_PLAYER_SELF,
            choices=(BENCH_PLAYER_SELF, "scripted"),
            help=f"Player B: '{BENCH_PLAYER_SELF}' (default) = pure self-play, "
                 "the net+MCTS on both seats; 'scripted' = scripted:hard on "
                 "seat B (the C++ legs via the scripted oracle, the Python leg "
                 "in-process) with the net+MCTS on seat A — both legs the same "
                 "workload"),
        Arg("--games", "int", default=4,
            help="Matches per leg (per actor process with --fleet; default 4)"),
        Arg("--sims", "int", default=DEFAULT_AZ_FAST_SIMS,
            help="PUCT simulations per decision, TOTAL across --worlds "
                 f"(default {DEFAULT_AZ_FAST_SIMS}, the fast budget — the "
                 "Python leg makes the league budget impractically slow)"),
        Arg("--worlds", "int", default=4,
            help="Determinized worlds per search (default 4)"),
        Arg("--seed", "int", default=1,
            help="Base RNG seed (actor process i uses seed + i*100000; "
                 "default 1)"),
        Arg("--batch", "str", default="1", metavar="K[,K...]",
            help="Comma-separated actor --batch values, one C++ leg each "
                 "(K>1 = virtual-loss batched leaf evaluation; default 1)"),
        _no_cross_world(),
        Arg("--no-python", "flag",
            help="Skip the Python az_selfplay leg (C++ legs only)"),
        _actor_device(),
        _eval_server(),
        Arg("--eval-server-device", "choice", default=None,
            choices=EVAL_DEVICE_CHOICES,
            help="Device of the eval-server leg's az_eval_server (default: "
                 "--actor-device, cpu -> cuda, as the az-* commands start it; "
                 "cpu exercises the Stage C socket path without a GPU)"),
        Arg("--fleet", "int", default=1,
            help="Concurrent actor processes per C++ leg (each plays --games "
                 "matches on a disjoint seed range); with the eval server this "
                 "measures the fleet-wide batching the server exists for "
                 "(default 1)"),
    ]),
    Sub("bench-workers",
        "Benchmark AZ self-play throughput across worker counts (bo3, shards "
        "pooled for the next az-train; optional az-train + az-eval legs)",
        items=[
        Arg("--workers", "str", default="32,48,64,74", metavar="N[,N...]",
            help="Comma-separated worker counts, one self-play leg each "
                 "(default 32,48,64,74)"),
        Arg("--random-draw", "flag",
            help="Use the random --mirror-frac draw schedule of --games "
                 "matches instead of the default --exhaustive-selfplay matrix "
                 "(what the az-league curriculum slots run)"),
        Arg("--exhaustive-repeats", "int", default=DEFAULT_AZ_EXHAUSTIVE_REPEATS,
            help="Exhaustive mode: play every self-play cell N times per leg "
                 f"(default {DEFAULT_AZ_EXHAUSTIVE_REPEATS})"),
        Arg("--scripted-cells", "int", default=DEFAULT_AZ_SCRIPTED_CELLS,
            help="Exhaustive mode: rotating vs-scripted:hard cells per leg "
                 f"(default {DEFAULT_AZ_SCRIPTED_CELLS}); the slot index "
                 "advances per leg so legs tile different cells"),
        Arg("--slot-base", "int", default=0,
            help="Exhaustive mode: slot index of the FIRST leg for the "
                 "rotating scripted-cell slice (leg i uses slot-base+i)"),
        Arg("--games", "int", default=148,
            help="Matches per leg under --random-draw (default 148; keep it >= "
                 "the largest worker count or generate() clamps workers down "
                 "to it). Ignored by the exhaustive matrix, which fixes the "
                 "count itself"),
        Arg("--decks", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated focus-deck pool (default: every deck in "
                 "decks/league/ — the most distinct actor matchup groups, so "
                 "high worker counts can bind)"),
        Arg("--sims", "int", default=DEFAULT_AZ_SIMS,
            help="PUCT simulations per decision, TOTAL across --worlds "
                 f"(default {DEFAULT_AZ_SIMS})"),
        Arg("--worlds", "int", default=DEFAULT_AZ_WORLDS,
            help=f"Determinized worlds per search (default {DEFAULT_AZ_WORLDS})"),
        _c_puct(),
        Arg("--mirror-frac", "float", default=DEFAULT_AZ_MIRROR_FRAC,
            help="--random-draw only: P(opponent deck == focus deck) per match "
                 f"(default {DEFAULT_AZ_MIRROR_FRAC})"),
        Arg("--td-n", "int", default=DEFAULT_AZ_TD_N, help=_TD_N_HELP),
        Arg("--checkpoint", "str", default=None, suggest="az_checkpoint",
            help="AZ (.pt) / PPO (.zip) ckpt or 'gen' (default: generalist AZ "
                 "ckpt, else gen PPO warm-start)"),
        Arg("--seed", "int", default=1,
            help="Base RNG seed; leg i uses seed + i*1000003 so every leg "
                 "plays fresh games (default 1)"),
        Arg("--out", "str", default=None,
            help="Shard output dir (default: the az_data/gen training pool, "
                 "so the next az-train incorporates the shards; point elsewhere "
                 "to keep the bench data OUT of the pool)"),
        _actor_mode(),
        _actor_device(),
        _eval_server(),
        _no_cross_world(),
        Arg("--train", "flag",
            help="After all legs, run `train.py az-train` over the fresh shards "
                 f"(AUTO batches: --epoch-frac {DEFAULT_AZ_EPOCH_FRAC}, "
                 f"--q-mix {DEFAULT_AZ_Q_MIX} — the az-train defaults), then "
                 "`train.py az-eval` gating the candidate at the training "
                 "budget with one panel round"),
        Arg("--deck-a", "str", default="delver", suggest="deck",
            help="The az-eval leg's --deck-a (default delver; the az-train "
                 "shard pool is deck-agnostic)"),
        Arg("--window", "int", default=0,
            help="az-train --window (default 0 = AUTO: this benchmark's shards "
                 "PLUS --pool-extra shards, else 2x the bench shards)"),
        Arg("--pool-extra", "str", default="auto",
            help="Pre-existing pool shards the AUTO --window also covers. "
                 "'auto' (default) counts the most recent az-league run's "
                 "shards (its completed slots from the league progress "
                 "sidecar, plus the pooled gate shards and an interrupted "
                 "slot's partial shards); an integer sets the count; 0 "
                 "disables"),
        Arg("--batches", "int", default=DEFAULT_AZ_CYCLE_BATCHES,
            help=f"az-train --batches (default {DEFAULT_AZ_CYCLE_BATCHES} = "
                 "AUTO, as in the az cycle)"),
        Arg("--eval-games", "int", default=DEFAULT_AZ_EVAL_GAMES,
            help=f"az-eval --games (default {DEFAULT_AZ_EVAL_GAMES})"),
        Arg("--no-eval", "flag", help="With --train: skip the az-eval leg"),
        Arg("--no-promote", "flag",
            help="Do NOT pass --promote to az-eval (default passes it, like the "
                 "az cycle's gate; the sequential test + floor still decide)"),
        _no_gate_shards(),
        Arg("--dry-run", "flag",
            help="Print each leg's plan (schedule size, distinct matchup "
                 "groups, effective concurrency cap) and the train/eval argv, "
                 "then exit without playing anything"),
    ]),
    Sub("bench-nenvs",
        "Benchmark PPO training throughput vs --n-envs to size it for this "
        "machine (steps/s, per-env steps/s, peak RAM; nothing is saved)",
        items=[
        Arg("--mode", "choice", default="self-play",
            choices=("league", "self-play", "scripted"),
            help="Training path to benchmark: league (the PFSP league pool, "
                 "mixed self-deck — what 'train.py league' runs), self-play "
                 "(default; needs a gen checkpoint, else it silently measures "
                 "the scripted fallback) or scripted"),
        Arg("--deck-a", "str", default="delver", suggest="deck",
            help="Deck the learner pilots (ignored by --mode league)"),
        Arg("--deck-b", "str", default=None, suggest="deck",
            help="Opponent deck (default: mirror = --deck-a; ignored by "
                 "--mode league)"),
        Arg("--n-envs", "str", default=None, metavar="N[,N...]",
            help="Comma-separated n_envs values to sweep (default: derived "
                 "from the CPU count)"),
        Arg("--timesteps", "int", default=250_000,
            help="Env steps in the timed phase per n_envs point, after a "
                 "1-rollout warmup (rounded up to whole rollouts; default "
                 "250000)"),
        *train_opts_only("embed_dim", "popart"),
        Arg("--ram-budget-gb", "float", default=None,
            help="Recommend the fastest n_envs whose peak RAM stays under this"),
        *common_args(INTERACTIVE_BINARY),
    ]),
])

# analysis.py — every command loads a trained model and simulates games (the
# .rmrec recording-file commands were removed; the live model-sim path is the
# single source; 'browse' also pages recorded shards and saved traces).
# 'report' is capture-mode (emits a self-contained HTML battery and exits).
#
# analysis.py browse — the full-screen analysis browser: game list,
# board-state pager (one decision step at a time), a clickable V(s) histogram
# for seeking, every analysis view (text analyses, per-game transcripts, saved
# PNG charts, counterfactual whatif), on the Textual board (tui_analysis)
# or the PySide6 app (gui_browser). ONE --source picks what it browses (see
# BROWSE_SOURCE_DESTS). Same sim args as the other analysis commands minus the
# chart-output flags. The GUI's New Analysis Session dialog mirrors these flags
# (see launcher_config.py).
ANALYSIS_BROWSE_SUB = Sub(
    "browse",
    "Full-screen analysis browser: page through board states with a "
    "clickable V(s) histogram, plus every analysis view — over simulated "
    "games, recorded shards, or a saved .rmtrace session", mode="interactive",
    items=[
        Arg("--source", "str", default=BROWSE_SOURCE_SIMULATE,
            help=f"What to browse: '{BROWSE_SOURCE_SIMULATE}' (the default) "
                 "simulates --games games of --player-a vs --player-b; a "
                 f"directory of {SHARD_GLOB} files (recorded AZ self-play, e.g. "
                 "train/az_data/gen, or a GUI recording under "
                 "train/az_data/recorded/) replays those decisions — pi (the "
                 "search's visit posterior) fills the policy column and whatif/"
                 f"run stay disabled without a live env; a {TRACE_EXT} file "
                 "opens a saved analysis session. Flags that do not apply to "
                 "the chosen source are errors"),
        Arg("--board", "choice", choices=BROWSE_BOARD_CHOICES,
            default=DEFAULT_BROWSE_BOARD,
            help="Browser front end: tui (the Textual terminal browser) or gui "
                 "(the PySide6 app's analysis pane — falls back to tui when "
                 f"PySide6 is missing) (default {DEFAULT_BROWSE_BOARD})"),
        # --player-a (the inspected model) doubles as the shard/trace sources'
        # value / replay-search net, so it gets a 'gen' default (simulate
        # resolves that to the one generalist anyway); --deck-a defaults to a
        # league deck so a bare launch simulates straight away.
        *[replace(a, required=False, default="gen",
                  help=a.help + " With a shard or .rmtrace --source: the net "
                                "for V(s), the probes and the replay search "
                                "(default gen)")
          if a.name == "--player-a" else
          replace(a, default=DEFAULT_BROWSE_DECK_A,
                  help=a.help + f" (default {DEFAULT_BROWSE_DECK_A})")
          if a.name == "--deck-a" else a
          for a in sim_args() if a.name not in ("--out", "--show")],
        *search_knob_args(),
        Arg("--games", "int", default=20,
            help="Games to simulate on startup — each a whole match under "
                 "--format bo3 (default: 20). With a shard --source: the "
                 "first N recorded bo3 matches (oldest first) to load, reading "
                 "only the shards they need (0 = every match; refused for a "
                 "directory over 2 GiB of shards, e.g. a training pool)"),
        Arg("--seat", "choice", choices=("A", "B"), default="A",
            help="Shard --source: viewpoint seat — that seat's searched "
                 "decisions are the browsable steps, the other seat's are "
                 "summarized as opponent actions (default: A)"),
        Arg("--no-net", "flag",
            help="Shard --source: skip loading the value net; V(s) falls back "
                 "to each step's recorded game outcome z (torch-free)"),
    ])

ANALYSIS_TOOL = Tool("analysis", "train/analysis.py", subs=[
    ANALYSIS_BROWSE_SUB,
    Sub("report", "Run the standard battery and emit a single HTML report "
        "(a search --player-a adds the search-vs-net sections)", items=[
        *sim_args(),
        *search_knob_args(),
        Arg("--games", "int", default=50,
            help="Games to simulate — each a whole match under --format bo3 "
                 "(default: 50)"),
        Arg("--workers", "int", default=1,
            help="Parallel worker processes simulating the --games (default: "
                 "1 = in-process). Each rebuilds its own model/engine and "
                 "plays a contiguous slice of the seeds, so the engine seeds "
                 "and seats are the same whatever the count (a search seat's "
                 "own RNG stream restarts per worker); an unset "
                 "--search-procs is 1 per worker"),
    ]),
])

# play.py — interactive human-vs-opponent play on one of three boards: the
# PySide6 GUI (gui_main.py), the Textual TUI (tui_game.py), or plain text on
# the shared runner loop. The GUI's New Play Session dialog mirrors these flags
# field for field (same dests, same defaults; see launcher_config.py).
PLAY_TOOL = Tool("play", "train/play.py", flat=True, subs=[
    Sub("play", "Play interactively against a trained model", mode="interactive", items=[
        Arg("--board", "choice", choices=BOARD_CHOICES, default=DEFAULT_BOARD,
            help="Game board: gui (PySide6 desktop board — falls back to the "
                 "TUI with a notice when PySide6 is missing), tui (Textual "
                 "terminal board), or text (plain transcript with typed "
                 "actions). --board gui with no --player-a/-b or --deck-a/-b "
                 f"opens the GUI app on its welcome pane (default {DEFAULT_BOARD})"),
        Arg("--player-a", "str", default=None, suggest="agent",
            help="Player A (on the play in game 1): 'human' for you, or any "
                 "opponents.make_controller spec for the opponent — 'gen', a "
                 "model .zip path, az:gen (MCTS+AZNet), azraw:gen (raw AZ "
                 "policy), mcts:gen, scripted:<tier>. Exactly one seat is "
                 "'human'; an omitted seat is the human when the other names "
                 "the opponent, else the default opponent "
                 f"({DEFAULT_PLAY_OPPONENT}). Default: human"),
        Arg("--player-b", "str", default=None, suggest="agent",
            help="Player B: 'human' or an opponent spec, as for --player-a "
                 f"(default: {DEFAULT_PLAY_OPPONENT}, or the human when "
                 "--player-a names the opponent)"),
        Arg("--deck-a", "str", default=DEFAULT_PLAY_DECK_A, suggest="deck",
            help=f"Player A's deck (.dk stem; default {DEFAULT_PLAY_DECK_A})"),
        Arg("--deck-b", "str", default=DEFAULT_PLAY_DECK_B, suggest="deck",
            help=f"Player B's deck (.dk stem; default {DEFAULT_PLAY_DECK_B})"),
        format_arg(),
        Arg("--human-clock", "float", default=None,
            help="Arm YOUR OWN chess clock: total wall-clock thinking bank in "
                 "seconds for the whole match, debited by the time you spend "
                 "on each of your decisions (the opponent's bank is the "
                 "separate --match-clock). Unset = untimed. On its own the "
                 "bank is only a readout; add --hard-timeout to make it "
                 "decisive (gui/tui boards)"),
        Arg("--hard-timeout", "flag",
            help="Losing on time is real: a seat that reaches its own decision "
                 "with an empty bank concedes the match (CR 104.3a). Applies to "
                 "both your --human-clock and a search opponent's --match-clock, "
                 "whose bank is otherwise SOFT (it just thinks faster) "
                 "(gui/tui boards)"),
        *search_knob_args(worlds=DEFAULT_PLAY_WORLDS,
                          match_clock=DEFAULT_PLAY_MATCH_CLOCK, paced=True),
        Arg("--record-shards", "flag",
            help="Record every decision of the session into trainer-schema "
                 "shard files under train/az_data/recorded/ — a search "
                 "opponent's searched decisions with their full visit "
                 "posterior, everything else as one-hot rows. Browse them "
                 "with `analysis.py browse --source DIR` or az-inspect "
                 "--shards DIR (on the GUI board also live via "
                 "View ▸ Analyze Recording…, F10) (gui/tui boards)"),
        Arg("--analysis", "bool", default=None,
            help="The analysis window: live MCTS evaluation of your "
                 "decisions on a detached engine copy (F9 toggles it). GUI "
                 "board only; unset = on for the GUI board"),
        Arg("--analysis-evaluator", "str", default=DEFAULT_ANALYSIS_EVALUATOR,
            help="Analysis window evaluator: az:gen (AZ net, calibrated win rate), "
                 "mcts:gen (PPO heads), uniform (no model), or a checkpoint "
                 f"path (default {DEFAULT_ANALYSIS_EVALUATOR})"),
        Arg("--analysis-worlds", "int", default=DEFAULT_ANALYSIS_WORLDS,
            help="Analysis window: determinized worlds per analysis run "
                 f"(default {DEFAULT_ANALYSIS_WORLDS})"),
        Arg("--analysis-procs", "int", default=None,
            help="Analysis window: detached engines to fan the worlds across "
                 "(default AUTO: half the visible cores, capped at the world "
                 "count; the merged result is the same, just faster)"),
        Arg("--analysis-cap", "int", default=DEFAULT_ANALYSIS_CAP,
            help="Analysis window: simulation cap per analysis run (0 = run "
                 f"until stopped; default {DEFAULT_ANALYSIS_CAP})"),
        Arg("--analysis-auto", "bool", default=True,
            help="Analysis window: start a run at every new analyzable "
                 "decision (default on; --no-analysis-auto = only on F5)"),
        Arg("--analysis-xw", "bool", default=True,
            help="Analysis window: cross-world batched leaf evaluation "
                 "(identical visits, faster chunks; default on — "
                 "--no-analysis-xw only to debug)"),
        Arg("--analysis-device", "choice", choices=EVAL_DEVICE_CHOICES,
            default=None,
            help="Analysis window: torch device for the evaluator's forwards "
                 "(az:/checkpoint specs; uniform and mcts: stay on cpu). "
                 "Unset = ROBOMAGE_EVAL_DEVICE, else cpu"),
        Arg("--seed", "int", default=None,
            help="Engine RNG seed for a reproducible game (default: random)"),
        Arg("--binary", "str", default=INTERACTIVE_BINARY, help="Path to robomage binary"),
    ]),
])

# play.py's dests that name the session (seats / decks): a --board gui launch
# with none of them opens the GUI app's welcome pane instead of a game.
PLAY_SESSION_DESTS = frozenset({"player_a", "player_b", "deck_a", "deck_b"})
# The analysis-window dests (GUI board only), minus the on/off switch itself.
PLAY_ANALYSIS_DESTS = ("analysis_evaluator", "analysis_worlds", "analysis_procs",
                       "analysis_cap", "analysis_auto", "analysis_xw",
                       "analysis_device")


HUMAN_SPEC = "human"


def is_human_spec(spec) -> bool:
    """True for the agent spec that seats the interactive human."""
    return isinstance(spec, str) and spec.strip().lower() == HUMAN_SPEC


def resolve_play_seats(player_a, player_b):
    """``(human_seat, opponent_spec)`` for play's --player-a / --player-b.

    Exactly one seat is the spec ``human``. An omitted (None) seat fills in:
    the default opponent when the other seat is human, else the human. With
    neither given the human is player A. ``opponent_spec`` None means the
    default opponent (the generalist). Raises ValueError unless exactly one
    seat ends up human."""
    a, b = player_a, player_b
    if a is None and b is None:
        a = HUMAN_SPEC
    elif a is None:
        a = None if is_human_spec(b) else HUMAN_SPEC
    elif b is None:
        b = None if is_human_spec(a) else HUMAN_SPEC
    if is_human_spec(a) and is_human_spec(b):
        raise ValueError("--player-a and --player-b are both 'human'; one "
                         "seat must be the opponent")
    if not (is_human_spec(a) or is_human_spec(b)):
        raise ValueError("exactly one of --player-a / --player-b must be "
                         f"'{HUMAN_SPEC}' (got {a!r} and {b!r})")
    return ("A", b) if is_human_spec(a) else ("B", a)

# The harness seat default: pass priority / take the first choice at every
# decision the --play/--actions script does not make.
HARNESS_DEFAULT_PLAYER = "auto"

# test_harness.py — card-behaviour test harness (flat parser, no subcommand).
# test_harness.main() builds its parser from this Sub; the launcher composes a
# command and runs it in the real terminal (so a 'human' seat's prompts work).
HARNESS_TOOL = Tool("harness", "train/test_harness.py", flat=True, subs=[
    Sub("harness", "Run a card-behaviour scenario through the engine",
        mode="interactive", items=[
        Arg("--scenario", "str",
            help="Path to a JSON scenario file (hand_a/library_a/battlefield_a/"
                 "…, life_a/b, actions or play, seed, max_decisions); a flag "
                 "given on the command line overrides the scenario's value"),
        Arg("--hand-a", "str",
            help="Player A starting hand (comma-separated card names; builds a "
                 "stacked temp deck, implies --no-shuffle)"),
        Arg("--library-a", "str",
            help="Player A library after the hand (comma-separated; padded to a "
                 "15-card deck)"),
        Arg("--hand-b", "str", help="Player B starting hand (see --hand-a)"),
        Arg("--library-b", "str", help="Player B library after the hand (see --library-a)"),
        Arg("--deck-a", "str", suggest="deck",
            help="Existing deck file for Player A (stem relative to decks/, "
                 "not a path; default: delver). Ignored when --hand-a is given"),
        Arg("--deck-b", "str", suggest="deck",
            help="Existing deck file for Player B (see --deck-a)"),
        Arg("--battlefield-a", "str",
            help="Cards starting on Player A's battlefield (comma-separated; "
                 "no summoning sickness)"),
        Arg("--battlefield-b", "str", help="Cards starting on Player B's battlefield"),
        Arg("--graveyard-a", "str", help="Cards starting in Player A's graveyard (comma-separated)"),
        Arg("--graveyard-b", "str", help="Cards starting in Player B's graveyard"),
        Arg("--exile-a", "str", help="Cards starting in Player A's exile (comma-separated)"),
        Arg("--exile-b", "str", help="Cards starting in Player B's exile"),
        Arg("--sideboard-a", "str",
            help="Cards starting in Player A's sideboard / 'outside the game' "
                 "(comma-separated)"),
        Arg("--sideboard-b", "str", help="Cards starting in Player B's sideboard"),
        Arg("--life-a", "int", default=None,
            help="Player A's starting life total (default 20) — exercises "
                 "life-payment costs at a chosen life"),
        Arg("--life-b", "int", default=None, help="Player B's starting life total (default 20)"),
        Arg("--play", "str",
            help="Semantic action script for BOTH seats (one spec per decision, "
                 "comma-separated), resolved against the live menu, e.g. "
                 "\"cast:Lightning Bolt,target:Grizzly Bears@opp,pass\" "
                 "(grammar: action_spec.py). Prefix a spec with A:/B: to pin it "
                 "to a seat — the other seat auto-passes until the keyed seat is "
                 "on the clock. An unmatched/ambiguous spec fails loudly with the "
                 "legal menu. Once the script runs out, --player-a/--player-b "
                 "make the remaining decisions"),
        Arg("--actions", "str",
            help="Positional action-index script for BOTH seats (e.g. 9,0,7,0,8; "
                 "fragile — prefer --play). Once it runs out, "
                 "--player-a/--player-b make the remaining decisions"),
        Arg("--player-a", "str", default=HARNESS_DEFAULT_PLAYER, suggest="agent",
            help="Player A's agent for every decision the --play/--actions "
                 "script does not make: any opponents.make_controller spec — "
                 "'auto' (pass / first choice), 'scripted' (hard tier), "
                 "'scripted:easy', 'scripted:random', 'explore' / "
                 "'explore:patient' (coverage fuzzer; vary --seed), "
                 "'human' (prompt at the terminal — needs a TTY), 'gen', "
                 f"az:gen, … (default: {HARNESS_DEFAULT_PLAYER})"),
        Arg("--player-b", "str", default=HARNESS_DEFAULT_PLAYER, suggest="agent",
            help=f"Player B's agent (see --player-a; default: {HARNESS_DEFAULT_PLAYER})"),
        format_arg(
            ". A bo3 match: loser goes first next game; both players sideboard "
            "between games; the default --max-decisions is 1500 (up to 3 games "
            "+ sideboard decisions) vs 500 for bo1. Sculpted scenarios usually "
            "want --format bo1"),
        Arg("--merge-sideboard", "flag",
            help="Fold each deck's SIDEBOARD: section into its mainboard "
                 "(quantities summed) and run from a merged temp deck with NO "
                 "sideboard — lets single-game fuzzing reach sideboard-only "
                 "cards. Requires --deck-a and --deck-b (not inline "
                 "--hand/--library seats) and --format bo1. Merged decks still "
                 "shuffle"),
        Arg("--no-shuffle", "flag",
            help="Don't shuffle libraries — deck-file order = draw order (first "
                 "7 cards = opening hand). Implied by --hand-a/--hand-b; without "
                 "it libraries shuffle with the seeded RNG"),
        Arg("--coverage-json", "str", metavar="PATH",
            help="Accumulate per-action-category and per-card offered/taken "
                 "counters and write them as JSON to PATH at exit (with a "
                 "never_offered list of deck cards). Read-only observation — "
                 "play and RNG are unchanged. Combine per-game JSONs with "
                 "train/coverage_report.py merge/summarize"),
        Arg("--log-decisions", "flag",
            help="Have the engine write its self-contained RMLOG v2 decision "
                 "log (bin/resources/logs/game_<seed>.log), replayable with "
                 "--replay alone. Off by default in machine mode"),
        Arg("--seed", "int", default=None,
            help="RNG seed (default: 1, or the scenario's seed)"),
        Arg("--max-decisions", "int", default=None,
            help="Stop after N decisions (default: 500 for bo1 / 1500 for bo3, "
                 "or the scenario's max_decisions)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
    ]),
])

# az_inspect.py — static AZ checkpoint inspector: one subcommand per view (each
# prints its lines to the terminal), plus `tui`, the Textual front end
# (tui_az_inspect.InspectApp) over the same views. No games are played.

AZI_LABEL_KINDS = ("color", "type", "cmc", "land")
AZI_CARD_SPACES = ("identity", "props", "full")
# Shared shard-sample sizes (one default each, CLI views and the TUI alike).
DEFAULT_AZI_MAX_ROWS = 4000
DEFAULT_AZI_COUNT_ROWS = 1500
DEFAULT_AZI_BLOCK_ROWS = 150
DEFAULT_AZI_DONORS = 3
DEFAULT_AZI_NEIGHBORS = 20
DEFAULT_AZI_KNN = 10
DEFAULT_AZI_CLUSTERS = 8


def _azi_model():
    return Arg("--model", "str", default="gen", suggest="az_checkpoint",
               help="Checkpoint to inspect (an opponents.parse_model_spec "
                    "spec): 'gen' / az:gen (the AZ generalist, else the AZNet "
                    "warm-started from the PPO gen), a snapshot stem "
                    "(gen__azv384000), an AZ .pt path, or mcts:gen / a PPO "
                    ".zip (default: gen)")


def _azi_shards(optional):
    """--shards DIR. ``optional``: the view runs weights-only without it (and
    adds the shard-backed parts with it); otherwise the view needs recorded
    self-play and an absent --shards reads train/az_data/gen."""
    what = ("Recorded self-play shard directory (shard_*.npz) — or one or more "
            "comma-separated .npz files — ")
    if optional:
        return Arg("--shards", "str", default=None,
                   help=what + "adding the shard-backed views/annotations. "
                               "Absent = weights only")
    return Arg("--shards", "str", default=None,
               help=what + "to sample (default: train/az_data/gen)")


def _azi_window():
    return Arg("--window", "int", default=None,
               help="Use only the newest N shards (default: all)")


def _azi_seed():
    return Arg("--seed", "int", default=1,
               help="Sampling / k-means / t-SNE seed (default: 1)")


def _azi_sample_args(optional=False):
    """The shard-sample args of a view that reads a random row sample."""
    return [_azi_shards(optional),
            Arg("--max-rows", "int", default=DEFAULT_AZI_MAX_ROWS,
                help=f"Recorded decisions to sample (default: "
                     f"{DEFAULT_AZI_MAX_ROWS})"),
            _azi_window(), _azi_seed()]


def _azi_count_rows():
    return Arg("--count-rows", "int", default=DEFAULT_AZI_COUNT_ROWS,
               help="States decoded for per-card occurrence counts (default: "
                    f"{DEFAULT_AZI_COUNT_ROWS})")


def _azi_embedding_args():
    """An embedding view: weights-only, annotated/filtered by occurrence
    counts from --shards when given, else by the weights-only exposure."""
    return [_azi_model(), _azi_shards(True), _azi_window(), _azi_seed(),
            _azi_count_rows(),
            Arg("--min-seen", "int", default=0,
                help="Drop cards seen fewer than N times (occurrence counts "
                     "with --shards, else 1 = the row ever trained)"),
            Arg("--space", "choice", choices=AZI_CARD_SPACES,
                default="identity",
                help="Card space to measure: the trainable identity table "
                     "(default), the frozen printed-property block, or the "
                     "full concatenation")]


def _azi_mark(help_extra=""):
    return Arg("--mark", "choice", choices=AZI_LABEL_KINDS, default="color",
               help="Card label used as the scatter marker / chart color "
                    "(default: color)" + help_extra)


def _azi_top(default, what="Rows shown"):
    return Arg("--top", "int", default=default,
               help=f"{what} (default: {default})")


def _azi_row():
    return Arg("--row", "int", default=0,
               help="Which sampled decision (default: 0)")


def _azi_baseline(help_text):
    return Arg("--baseline", "str", default=None, help=help_text)


def _azi_chart_args(what):
    return [Arg("--chart", "flag",
                help=f"Also save {what} as a PNG chart (matplotlib, "
                     "headless-safe) under --out"),
            Arg("--out", "str", default=None,
                help="Directory for saved charts (default: train/analysis_out/)"),
            Arg("--show", "flag",
                help="With --chart: also open the chart in a GUI window (needs "
                     "a local display)")]


_AZI_ORIGIN_HELP = ("Checkpoint to measure against (default: the PPO gen "
                    "warm-start, else the oldest gen__azv* snapshot; 'init' = "
                    "a fresh untrained seed-0 net)")


def _azi_weights_sub(name, help_text, *items):
    """A weight-space view: reads an AZ .pt or a PPO .zip, never shards."""
    return Sub(name, help_text, items=[_azi_model(), *items])


AZ_INSPECT_TOOL = Tool("az-inspect", "train/az_inspect.py", subs=[
    Sub("tui",
        "Full-screen inspector (Textual): card embedding space with a "
        "clickable drill-down, the per-matchup critic, the weight-space views "
        "and — with --shards — per-decision probes. Opens weights-only in "
        "about a second", mode="interactive", items=[
            _azi_model(), *_azi_sample_args(optional=True), _azi_count_rows(),
            Arg("--min-seen", "int", default=0,
                help="Drop cards seen fewer than N times from the embedding "
                     "views (their rows never trained)"),
            Arg("--neighbors", "int", default=DEFAULT_AZI_NEIGHBORS,
                help=f"Neighbours listed per card (default: "
                     f"{DEFAULT_AZI_NEIGHBORS})"),
            Arg("--knn", "int", default=DEFAULT_AZI_KNN,
                help=f"k for the label-purity view (default: {DEFAULT_AZI_KNN})"),
            Arg("--clusters", "int", default=DEFAULT_AZI_CLUSTERS,
                help=f"k for k-means (default: {DEFAULT_AZI_CLUSTERS})"),
            _azi_mark(),
            _azi_top(25, "Rows in the occurrence / divergence / probe tables"),
            Arg("--block-rows", "int", default=DEFAULT_AZI_BLOCK_ROWS,
                help="States averaged by the mean block-attribution probe "
                     f"(default: {DEFAULT_AZI_BLOCK_ROWS})"),
            Arg("--donors", "int", default=DEFAULT_AZI_DONORS,
                help=f"Donor states per block (default: {DEFAULT_AZI_DONORS})"),
        ]),
    Sub("overview", "Checkpoint meta + critic coverage (+ shard status with "
                    "--shards)",
        items=[_azi_model(), *_azi_sample_args(optional=True)]),
    Sub("neighbors", "Nearest cards in embedding space", items=[
        Arg("card", "str", required=True,
            help="Card name (exact or unique substring)"),
        Arg("--neighbors", "int", default=DEFAULT_AZI_NEIGHBORS,
            help=f"Neighbours to show (default: {DEFAULT_AZI_NEIGHBORS})"),
        *_azi_embedding_args()]),
    Sub("structure", "kNN label purity of the embedding", items=[
        Arg("--knn", "int", default=DEFAULT_AZI_KNN,
            help=f"Neighbours per card (default: {DEFAULT_AZI_KNN})"),
        *_azi_embedding_args()]),
    Sub("clusters", "k-means over the card embedding", items=[
        Arg("--clusters", "int", default=DEFAULT_AZI_CLUSTERS,
            help=f"Number of clusters (default: {DEFAULT_AZI_CLUSTERS})"),
        *_azi_embedding_args()]),
    Sub("project", "PCA-to-2D terminal scatter; --chart saves a PCA/t-SNE "
                   "chart of the TRAINED rows only", items=[
        _azi_mark(),
        Arg("--width", "int", default=78, help="Terminal scatter width (default: 78)"),
        Arg("--height", "int", default=24, help="Terminal scatter height (default: 24)"),
        *_azi_embedding_args(),
        *_azi_chart_args("the 2D projection of every card whose row ever "
                         "trained (size = movement since --baseline)"),
        Arg("--method", "choice", choices=("pca", "tsne"), default="pca",
            help="Chart projection (default: pca)"),
        Arg("--perplexity", "float", default=30.0,
            help="Chart t-SNE perplexity (auto-capped for small sets; "
                 "default: 30)"),
        Arg("--label-top", "int", default=30,
            help="Chart: annotate the N most-moved cards (default: 30)"),
        _azi_baseline("Chart: checkpoint 'ever trained' is measured against "
                      "(default: the PPO gen warm-start, else the oldest "
                      "gen__azv* snapshot)"),
    ]),
    Sub("occur", "How often each card appears in recorded self-play", items=[
        _azi_model(), _azi_shards(False), _azi_window(), _azi_seed(),
        _azi_count_rows(), _azi_top(25)]),
    Sub("exposure", "Which embedding rows / critic columns actually trained "
                    "(weights only)", items=[
        _azi_model(),
        _azi_baseline("Checkpoint to measure against (default: the previous "
                      "gen__azv* snapshot, else the PPO gen warm-start)"),
        _azi_top(20)]),
    Sub("drift", "Signed per-dimension embedding movement since the origin; "
                 "all cards ranked by total |shift|; --chart maps it across "
                 "the movement matrix's own PCs", items=[
        _azi_model(), _azi_baseline(_AZI_ORIGIN_HELP),
        _azi_top(0, "Cards listed / charted (0 = all)"),
        *_azi_chart_args("a per-card heatmap of |movement| along each "
                         "movement-PC (with the per-PC energy scree)")]),
    Sub("catemb", "Action-category embedding neighbours", items=[_azi_model()]),
    Sub("buckets", "Per-matchup critic column map (+ the sampled-bucket census "
                   "with --shards)",
        items=[_azi_model(), *_azi_sample_args(optional=True)]),
    Sub("calib", "Per-bucket value calibration vs recorded outcomes",
        items=[_azi_model(), *_azi_sample_args()]),
    Sub("divergence", "Net priors vs the search posterior, by action category",
        items=[_azi_model(), *_azi_sample_args(), _azi_top(12)]),
    Sub("sbreport", "Between-games sideboarding sessions in recorded shards: "
                    "average cards brought in / cut per session, by matchup "
                    "(fetchlands fungible)", items=[
        _azi_shards(False), _azi_window(),
        Arg("--mtime-after", "str", default=None, metavar="'YYYY-mm-dd HH:MM'",
            help="Only shards modified at or after this time"),
        Arg("--mtime-before", "str", default=None, metavar="'YYYY-mm-dd HH:MM'",
            help="Only shards modified before this time"),
        Arg("--min-sessions", "int", default=1,
            help="Hide matchups with fewer sessions than this (default: 1)"),
        Arg("--label", "str", default="", help="Report title suffix"),
        Arg("--json", "str", default=None, metavar="PATH",
            help="Also write the report as JSON to PATH"),
    ]),
    Sub("diff", "Compare two checkpoints (either family)", items=[
        Arg("other", "str", required=True,
            help="Second checkpoint spec/path (B); --model is A"),
        _azi_model(), _azi_top(15)]),
    Sub("state", "Browse one recorded decision", items=[
        _azi_model(), *_azi_sample_args(), _azi_row(), _azi_top(12)]),
    Sub("blocks", "Permutation importance of obs blocks", items=[
        _azi_model(), *_azi_sample_args(),
        Arg("--row", "int", default=None,
            help="Attribute ONE recorded decision instead of the mean over "
                 "--block-rows states"),
        Arg("--block-rows", "int", default=DEFAULT_AZI_BLOCK_ROWS,
            help="States averaged when --row is not given (default: "
                 f"{DEFAULT_AZI_BLOCK_ROWS})"),
        Arg("--donors", "int", default=DEFAULT_AZI_DONORS,
            help=f"Donor states per block (default: {DEFAULT_AZI_DONORS})"),
        Arg("--only-active", "str", default=None, metavar="BLOCK",
            help="Restrict states and donors to those where BLOCK (name or "
                 "unique substring) is non-empty — scores a sparse block on "
                 "the states where it exists"),
        Arg("--sort", "choice", choices=("v", "pi"), default="v",
            help="Rank by value shift (v) or policy shift (pi)"),
        _azi_top(20)]),
    Sub("readout", "One state's V under every matchup critic column", items=[
        _azi_model(), *_azi_sample_args(), _azi_row(),
        Arg("--top", "int", default=None,
            help="Show only the top-N columns by V (default: all)")]),
    Sub("swap", "Per-slot card valuation by identity swap", items=[
        _azi_model(), *_azi_sample_args(), _azi_row(),
        Arg("--site", "int", default=None,
            help="Index into the state's card-identity sites (omit to list "
                 "them)"),
        _azi_top(12)]),
    Sub("sweep", "V across single scalars (life, hand, turn)", items=[
        _azi_model(), *_azi_sample_args(), _azi_row(),
        Arg("--field", "str", default=None,
            help="One field (default: every sweepable field)")]),
    _azi_weights_sub("firstlayer", "First-layer input-column attribution per "
                                   "encoder",
                     Arg("--encoder", "str", default=None,
                         help="One encoder (e.g. perm_encoder; default: all)"),
                     _azi_top(10, "Named columns shown per encoder")),
    _azi_weights_sub("bodylayer", "Policy/value body first-layer attribution "
                                  "(arch one-hots + decklist aggregates)",
                     Arg("--top-arch", "int", default=16,
                         help="Archetype columns to name (default: 16)")),
    _azi_weights_sub("unit", "One hidden unit's signed input-variable profile",
                     Arg("layer", "str", required=True,
                         help="Layer name: perm_encoder, stack_encoder, "
                              "entity_encoder, decklist_encoder, "
                              "revealed_encoder, action_encoder, policy_body, "
                              "value_body"),
                     Arg("unit", "int", required=True,
                         help="Unit (row) index in that layer"),
                     _azi_top(12, "Inputs shown per sign")),
    _azi_weights_sub("pathto", "Weights-only input connectivity of one value "
                               "bucket's head column",
                     Arg("bucket", "str", required=True,
                         help="Bucket index or name substring "
                              "(e.g. doomsday_vs_burn)"),
                     _azi_top(15, "Widest individual input columns shown")),
    _azi_weights_sub("spectra", "Singular-value spectrum / effective rank per "
                                "weight matrix"),
    _azi_weights_sub("popart", "PPO PopArt per-bucket value statistics "
                               "(PPO .zip)"),
    _azi_weights_sub("valuegeom", "Value-head row geometry (which matchups "
                                  "share a value direction)", _azi_top(12)),
    _azi_weights_sub("zoneemb", "Zone-ref embedding neighbours"),
    _azi_weights_sub("cardsel", "Entity-encoder units ranked by card "
                                "selectivity",
                     Arg("--units", "int", default=12,
                         help="Units to show (default: 12)"),
                     Arg("--cards", "int", default=6,
                         help="Top cards listed per unit (default: 6)")),
])

ALL_TOOLS = [TRAIN_TOOL, ANALYSIS_TOOL, AZ_INSPECT_TOOL,
             PLAY_TOOL, HARNESS_TOOL]


# ── argparse bridge (used by the scripts) ─────────────────────────────────────

def _add_one(target, a: Arg):
    import argparse
    if a.kind == "flag":
        target.add_argument(a.name, action="store_true", help=a.help)
        return
    if a.kind == "bool":
        target.add_argument(a.name, action=argparse.BooleanOptionalAction,
                            default=a.default, help=a.help)
        return
    kwargs = {"help": a.help}
    if a.metavar is not None:
        kwargs["metavar"] = a.metavar
    if a.kind == "int":
        kwargs["type"] = int
    elif a.kind == "float":
        kwargs["type"] = float
    elif a.kind == "choice":
        kwargs["choices"] = list(a.choices)
    if a.is_positional:
        # Current positionals are all required; support optional defensively.
        if not a.required:
            kwargs["nargs"] = "?"
            kwargs["default"] = a.default
        target.add_argument(a.name, **kwargs)
    else:
        kwargs["default"] = a.default
        if a.required:
            kwargs["required"] = True
        target.add_argument(a.name, **kwargs)


def add_args(parser, *args):
    """Add individual cli_spec Args to a standalone argparse parser."""
    for a in args:
        _add_one(parser, a)


def apply_to_parser(parser, sub: Sub):
    """Populate an argparse (sub)parser from a Sub spec, plus the hidden
    removed-flag options in the Sub's scope (see ``REMOVED_FLAGS``)."""
    for item in sub.items:
        if isinstance(item, MutexGroup):
            group = parser.add_mutually_exclusive_group(required=item.required)
            for a in item.args:
                _add_one(group, a)
        else:
            _add_one(parser, item)
    add_removed_flags(parser, *sub.scopes)


def iter_args(sub: Sub):
    """Yield every Arg in a Sub, flattening MutexGroups."""
    for item in sub.items:
        if isinstance(item, MutexGroup):
            yield from item.args
        else:
            yield item


def sub_defaults(sub: Sub) -> dict:
    """``{dest: default}`` for every Arg in a Sub (the values a bare
    invocation parses to — a store_true flag's is False)."""
    return {a.dest: arg_default(a) for a in iter_args(sub)}


def arg_default(a: Arg):
    """The value an Arg parses to when it is not given."""
    return bool(a.default) if a.kind == "flag" else a.default


def explicit_dests(parser, argv=None) -> set:
    """The dests ``argv`` sets on the command line (as opposed to leaving at
    their defaults) — re-parses with every default swapped for a marker. A
    removed optional positional keeps its None default (argparse hands it the
    default when nothing fills it, and anything else is the removal error)."""
    import argparse
    marker = object()
    saved = {}
    for act in parser._actions:
        if act.dest.startswith("_removed_"):
            continue
        if act.dest != argparse.SUPPRESS and act.default is not argparse.SUPPRESS:
            saved[act] = act.default
            act.default = marker
    try:
        ns, _extra = parser.parse_known_args(argv)
    finally:
        for act, default in saved.items():
            act.default = default
    return {k for k, v in vars(ns).items()
            if v is not marker and not k.startswith("_removed_")}
