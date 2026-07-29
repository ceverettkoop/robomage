"""Single source of truth for the train.py / analysis.py / play.py command lines.

This module is intentionally dependency-free (stdlib only) so that both the
heavyweight scripts (which pull in numpy/torch/gymnasium) and the lightweight
``tui.py`` launcher can import it without paying for those imports.

The scripts build their ``argparse`` parsers from the ``Tool`` objects here via
``apply_to_parser``; the TUI builds its input forms from the same objects.  Add
or change a flag in one place and both stay in sync.
"""

import os
from dataclasses import dataclass, field

from archetypes import ARCHETYPES

# ── Canonical CLI constants (single home; imported by env.py / train.py) ──────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO_ROOT, "bin", "robomage")
BIN_DIR = os.path.join(REPO_ROOT, "bin")  # game must be run from here for resource lookup

# AlphaZero sideboard-root search budget (single home; imported by az_selfplay,
# opponents.SearchController, and the CLI flag defaults below). A bo3 sideboard
# prompt IS a valid MCTS root, but each rollout there re-crosses init_ecs() + deck
# load + shuffle on RESTORE and its horizon spans the whole next game, so it gets
# its own deeper budget rather than the per-in-game-decision one (whose max_depth
# is mcts.run_search's default of 60 — too shallow for a game-long horizon).
#
# sims was 32, chosen when a sideboard decision was the old paired IN->OUT menu.
# The balanced delta menu offers every sideboard card AND every maindeck card at
# once — ~33 children on a league deck, up to ~39 — so 32 sims was roughly ONE
# visit per child: the visit distribution the policy trains on was essentially
# noise, and could not rank cards at all. 256 gives ~6-8 visits per child (split
# over DEFAULT_SB_WORLDS trees), enough for the ordering to be meaningful rather
# than borderline. This affects only the AZ / search paths (az_selfplay,
# bin/az_actor, az*/eval, the analysis window); PPO training does no search, so
# its cost is unchanged. NOTE: the az / az-league / az-selfplay CLI args MUST
# reference this constant, not a literal — argparse always supplies the CLI
# default, so a drifted literal silently overrides this "one home" (that bug
# shipped runs at sb_sims=32 while this constant said 128).
DEFAULT_SB_SIMS = 256
DEFAULT_SB_WORLDS = 4
DEFAULT_SB_MAX_DEPTH = 200

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
    n_steps=4096,       # steps per env per update
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


# ── Spec dataclasses ──────────────────────────────────────────────────────────

@dataclass
class Arg:
    """One CLI argument.

    ``name`` with a leading ``--`` is an optional flag; otherwise it is a
    positional.  ``kind`` is one of ``str``, ``int``, ``flag`` (store_true), or
    ``choice`` (requires ``choices``).
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


@dataclass
class Tool:
    """A script exposing one or more subcommands."""
    key: str
    script: str          # path relative to repo root, e.g. "train/train.py"
    subs: list = field(default_factory=list)
    default_sub: str = None  # subcommand assumed when none is given (train)
    flat: bool = False   # True when the script has a flat parser (no subcommand
                         # token in argv, e.g. play.py / test_harness.py)


# ── Reusable argument groups (mirror the helper functions in the scripts) ─────

def common_args():
    """Args shared by every train.py subcommand (was train.py _add_common)."""
    return [
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
        Arg("--bo3", "flag",
            help="Best-of-three match mode (deck swap + sideboarding between games)"),
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


def sim_args():
    """Common simulation args for analysis.py (was analysis.py _add_sim_args)."""
    return [
        Arg("model", "str", required=True, suggest="agent",
            help="Model to analyze: 'gen', a .zip path, or az:gen/azraw:gen "
                 "for the generalist AlphaZero net. A SEARCH spec (az:/mcts: "
                 "prefix, e.g. az:gen?sims=128&worlds=4) makes the simulated "
                 "trace games be PLAYED by the real MCTS controller, so the "
                 "browser inspects states arising from search-quality play "
                 "(slow); azraw:gen and a bare PPO spec keep raw-policy traces. "
                 "The inspection net (value/probs/SHAP) is the same either way."),
        Arg("--opponent", "str", default="scripted", suggest="agent",
            help="Opponent controller: 'gen', a model .zip path, az:gen/azraw:gen, "
                 "or 'scripted' for the rule-based agent piloting the opponent deck "
                 "(the default)"),
        Arg("--deck-a", "str", default=None, suggest="deck",
            help="Model's deck (.dk stem) — the deck it pilots. REQUIRED for a "
                 "model seat: the one generalist encodes no deck in its filename."),
        Arg("--deck-b", "str", default=None, suggest="deck",
            help="Opponent's deck (.dk stem). REQUIRED for a model opponent (the "
                 "generalist encodes no deck); a scripted opponent defaults to a "
                 "mirror match (--deck-a)."),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
        Arg("--bo3", "flag",
            help="Run best-of-three matches (decks must include SIDEBOARD entries)"),
        Arg("--out", "str", default=None,
            help="Directory for saved charts/reports (default: train/analysis_out/)"),
        Arg("--show", "flag",
            help="Also open charts in a GUI window (needs a local display)"),
    ]


# ── Tool definitions ──────────────────────────────────────────────────────────

TRAIN_TOOL = Tool("train", "train/train.py", default_sub="train", subs=[
    Sub("train", "Train the one generalist model (default command)", items=[
        Arg("--deck", "str", default="delver", suggest="deck",
            help="Deck the generalist plays this session (.dk stem, default: "
                 "delver). Always saved to the single gen__final.zip; sessions on "
                 "any deck/opponent accumulate onto that one generalist."),
        Arg("--opponent", "str", required=True, suggest="deck",
            help="Opponent deck this session trains against (.dk stem). The model "
                 "stays one generalist — training continues the same gen__final.zip "
                 "rather than forging a per-deck or matchup-specific model."),
        Arg("--load", "str", default=None, suggest="checkpoint",
            help="Resume from a specific checkpoint .zip ('gen' or a path), "
                 "overriding the default auto-resume of gen__final.zip"),
        _opponent_mode(),
        *opponent_pool_opts(),
        *train_opts(),
        *common_args(),
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
        *common_args(),
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
        *common_args(),
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
        Arg("--deck", "str", required=True, suggest="deck",
            help="Deck to train on (.dk stem). Always saved to gen__final.zip; this "
                 "session accumulates onto the one generalist, same as 'train'."),
        Arg("--opponents", "str", default=None, suggest="deck", multi=True,
            help="Comma-separated pool of opponent decks to sample from via PFSP "
                 "(default: every other deck in bin/resources/decks/). Like league's "
                 "roster, but this pool is opponents only — --deck is never rotated "
                 "into training and never part of the pool."),
        Arg("--self-play-frac", "float", default=LEAGUE_SELF_PLAY_FRAC,
            help="Probability of facing the latest snapshot of --deck itself (the "
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
            help="Only keep a snapshot when --deck's recent-window win-rate "
                 ">= 0.5 + margin (negative gates below 0.5, e.g. -0.1 -> 0.40; the "
                 "first snapshot is exempt so self-play can bootstrap; 0 disables "
                 "the gate; default %.2f)." % LEAGUE_PROMOTE_MARGIN),
        Arg("--opponent-ckpt-ratio", "float", default=1.0,
            help="Cap on unique opponent checkpoints kept resident, as a ratio of "
                 "n_envs (default 1.0 -> <=1 checkpoint per env process)."),
        *train_opts(),
        *common_args(),
    ]),
    Sub("fixed-model", "Train --deck vs a fixed (never-reloaded) opponent model", items=[
        Arg("--deck", "str", default="delver", suggest="deck", help="Deck the model plays (.dk stem)"),
        Arg("--opponent", "str", required=True, suggest="deck", help="Opponent deck (.dk stem)"),
        Arg("--load", "str", default=None, suggest="checkpoint",
            help="Resume from checkpoint .zip ('gen' or a path)"),
        *train_opts(),
        *common_args(),
    ]),
    Sub("alternate", "Swap which side is trained every N timesteps", items=[
        Arg("--deck", "str", default="delver", suggest="deck", help="First deck (.dk stem)"),
        Arg("--opponent", "str", required=True, suggest="deck", help="Second deck (.dk stem)"),
        Arg("--every", "int", required=True, metavar="N",
            help="Swap the trained side every N timesteps"),
        *train_opts(),
        *common_args(),
    ]),
    Sub("observe",
        "Observe game(s) between any pair of {scripted | model} controllers "
        "(replaces the old watch/diag/observe commands)", items=[
        Arg("--player-a", "str", default="scripted", suggest="agent",
            help="Player A controller: 'scripted' (or 'scripted:*'), 'gen', a model "
                 ".zip path, or az:gen/azraw:gen/mcts:gen (default: scripted)"),
        Arg("--player-b", "str", default="scripted", suggest="agent",
            help="Player B controller: 'scripted' (or 'scripted:*'), 'gen', a model "
                 ".zip path, or az:gen/azraw:gen/mcts:gen (default: scripted)"),
        Arg("--play-a", "str", default=None,
            help="Drive Player A by semantic action specs instead of --player-a, e.g. "
                 "\"cast:Lightning Bolt,target:Grizzly Bears@opp,pass\" (see action_spec.py grammar)"),
        Arg("--play-b", "str", default=None,
            help="Drive Player B by semantic action specs instead of --player-b (see --play-a)"),
        Arg("--deck", "str", default="delver", suggest="deck", help="Player A deck (.dk stem, default: delver)"),
        Arg("--opponent", "str", default=None, suggest="deck", help="Player B deck (.dk stem, default: Player A's deck)"),
        Arg("--games", "int", default=1,
            help="Number of games/matches to run (default: 1). >1 prints per-game results and a W/L/D summary"),
        Arg("--seed", "int", default=None,
            help="RNG seed for reproducible games (game N uses seed+N; default: random)"),
        Arg("--verbose", "flag",
            help="Dump full board state (battlefield, hands, mana, stack, graveyards) at each decision"),
        Arg("--bo1", "flag",
            help="Single-game mode. observe defaults to bo3 matches; this opts back "
                 "into one-off games (--bo3 is a redundant no-op here)"),
        *common_args(),
    ]),
    Sub("baseline", "Evaluate model win rate vs the scripted HARD agent (mirror match)", items=[
        Arg("model", "str", required=False, suggest="checkpoint",
            help="Model to evaluate: 'gen', a .zip/.pt path (e.g. an exp_* "
                 "exploiter checkpoint), or a search spec ('az:gen', 'mcts:gen', "
                 "'azraw:gen', with the usual ?sims=... knobs). With --all it "
                 "picks the model round-robined (default: gen)"),
        Arg("--games", "int", default=None,
            help="Games per matchup (default: 100 for a single model, 50 per opponent "
                 "with --all)"),
        Arg("--all", "flag",
            help="Round-robin the model (default: the generalist gen__final.zip) on "
                 "every league deck vs scripted:hard on every league deck (including "
                 "the mirror): an N-deck roster runs N×N matchups of --games each, "
                 "and the per-matchup win rates are appended to the report log"),
        Arg("--log", "str", default=None,
            help="Report file for --all (default: checkpoints/baseline_report.log, appended)"),
        Arg("--deck", "str", default=None, suggest="deck",
            help="Deck the model pilots — REQUIRED (the generalist encodes no deck). "
                 "The scripted opponent mirrors it."),
        Arg("--seed", "int", default=None,
            help="RNG seed for reproducible runs (game N uses seed+N; default: random)"),
        Arg("--bo1", "flag",
            help="Single-game mode. baseline defaults to bo3 matches; this opts back "
                 "into one-off games (--bo3 is a redundant no-op here)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
    ]),
    # ── AlphaZero (Phase C) ───────────────────────────────────────────────────
    Sub("az-selfplay",
        "Generate AlphaZero self-play data (focus deck vs mirror + roster, bo1)", items=[
        Arg("--deck", "str", default="delver", suggest="deck",
            help="Focus deck (.dk stem); its opponent is a mirror with "
                 "P=--mirror-frac, else a uniform league-roster draw"),
        Arg("--games", "int", default=50, help="Games to generate"),
        Arg("--sims", "int", default=256,
            help="PUCT simulations per decision, TOTAL across --worlds"),
        Arg("--worlds", "int", default=4, help="Determinized worlds per search"),
        Arg("--workers", "int", default=None,
            help="Worker processes (default max(1, cpu-2))"),
        Arg("--checkpoint", "str", default=None, suggest="az_checkpoint",
            help="AZ (.pt) / PPO (.zip) ckpt or 'gen' (default: generalist AZ "
                 "ckpt, else gen PPO warm-start, else random init)"),
        Arg("--temp-moves", "int", default=20,
            help="Sample from visit counts for the first N real decisions, then argmax"),
        Arg("--sb-sims", "int", default=DEFAULT_SB_SIMS,
            help="PUCT sims at a bo3 sideboard root (bo3 only; heavier per step "
                 f"than an in-game decision; default {DEFAULT_SB_SIMS})"),
        Arg("--sb-worlds", "int", default=4,
            help="Determinized worlds at a bo3 sideboard root (default 4)"),
        Arg("--sb-max-depth", "int", default=200,
            help="Rollout depth cap at a bo3 sideboard root (game-long horizon; "
                 "default 200)"),
        Arg("--mirror-frac", "float", default=0.25,
            help="P(opponent deck == focus deck) per game (default 0.25); else a "
                 "uniform league-roster draw"),
        Arg("--out", "str", default=None, help="Output dir (default az_data/gen)"),
        Arg("--seed", "int", default=1, help="Base RNG seed"),
        Arg("--expert", "flag",
            help="Write EXPERT demonstration shards instead of self-play: "
                 "scripted:hard pilots both seats and pi is a one-hot on the "
                 "expert's action (always bo3 to match the pooled shard window; "
                 "sims/worlds/checkpoint are ignored)"),
        _actor_mode(),
    ]),
    Sub("az-train", "Train an AZNet on self-play shards", items=[
        Arg("--deck", "str", default="delver", suggest="deck", help="Deck (.dk stem)"),
        Arg("--batches", "int", default=1000, help="Optimizer updates"),
        Arg("--batch-size", "int", default=256),
        Arg("--lr", "float", default=1e-3),
        Arg("--c-v", "float", default=1.0, help="Value-loss weight"),
        Arg("--window", "int", default=50, help="Number of most-recent shards to train on"),
        Arg("--from-ppo", "str", default=None, suggest="checkpoint",
            help="Warm-start from a PPO checkpoint instead of resuming AZ"),
        Arg("--fresh", "flag", help="Start from random init"),
        Arg("--snapshot-every", "int", default=0,
            help="Also save an intermediate gen__azv{steps}.pt every N batches (0=off)"),
        Arg("--seed", "int", default=0),
    ]),
    Sub("az-eval", "Gate a candidate AZNet vs the incumbent (MCTS, low sims)", items=[
        Arg("--deck", "str", default="delver", suggest="deck", help="Deck (.dk stem)"),
        Arg("--candidate", "str", required=True, suggest="az_checkpoint",
            help="Candidate AZ .pt ('gen' or a path)"),
        Arg("--incumbent", "str", default=None, suggest="az_checkpoint",
            help="Incumbent AZ .pt (default: gen__azfinal.pt; scripted if none yet)"),
        Arg("--games", "int", default=56,
            help="Total gate matches, split over the roster-wide panel (a mirror "
                 "per roster deck + direction-balanced cross pairs)"),
        Arg("--sims", "int", default=32),
        Arg("--worlds", "int", default=2),
        Arg("--promote-threshold", "float", default=0.55),
        Arg("--promote", "flag", help="Copy candidate to gen__azfinal.pt if it clears the bar"),
        Arg("--gate-floor", "float", default=0.2,
            help="Per-piloted-deck gate floor: a deck the candidate piloted in "
                 ">=4 gate matches whose win-rate deficit vs the incumbent on "
                 "LIKE pairings falls below 2*floor-1 vetoes promotion even "
                 "when the aggregate clears the bar (0 disables; on mirrors "
                 "alone this equals win-rate < floor)"),
        Arg("--seed", "int", default=1),
        Arg("--bo1", "flag",
            help="Single-game gate. az-eval defaults to bo3 match win-rate; this "
                 "opts back into one-off games"),
    ]),
    Sub("az",
        "One AlphaZero cycle (self-play -> train -> eval/gate) over a deck x "
        "opponent matrix (default: whole league; pass one --deck to fix a focus). "
        "bo3 by default (per-game value target); --bo1 to opt out",
        items=[
        Arg("--deck", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated FOCUS deck pool the generalist pilots "
                 "(default: every deck in decks/league/). Pass a single deck to "
                 "fix one focus (the classic single-deck cycle)."),
        Arg("--opponents", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated opponent-deck pool for self-play + gating "
                 "(default: every deck in decks/league/). Each focus deck plays "
                 "each; per game the opponent is the mirror with P=--mirror-frac, "
                 "else a uniform draw from this pool."),
        Arg("--games", "int", default=50, help="Self-play games this cycle"),
        Arg("--sims", "int", default=256,
            help="Self-play PUCT sims, TOTAL across --worlds (256/4 = 64 per "
                 "determinized world tree)"),
        Arg("--worlds", "int", default=4),
        Arg("--sb-sims", "int", default=DEFAULT_SB_SIMS,
            help=f"PUCT sims at a bo3 sideboard root (bo3 only; default {DEFAULT_SB_SIMS})"),
        Arg("--sb-worlds", "int", default=4,
            help="Determinized worlds at a bo3 sideboard root (default 4)"),
        Arg("--sb-max-depth", "int", default=200,
            help="Rollout depth cap at a bo3 sideboard root (default 200)"),
        Arg("--workers", "int", default=None),
        Arg("--batches", "int", default=500),
        Arg("--batch-size", "int", default=256),
        Arg("--lr", "float", default=1e-3),
        Arg("--window", "int", default=50),
        Arg("--eval-games", "int", default=56,
            help="Total gate matches, split over the roster-wide panel (a mirror "
                 "per roster deck + direction-balanced cross pairs; default 56 "
                 "= 4 per matchup on a 10-deck roster)"),
        Arg("--eval-sims", "int", default=32),
        Arg("--eval-worlds", "int", default=2),
        Arg("--promote-threshold", "float", default=0.55),
        Arg("--gate-floor", "float", default=0.2,
            help="Per-piloted-deck gate floor: a deck the candidate piloted in "
                 ">=4 gate matches whose win-rate deficit vs the incumbent on "
                 "LIKE pairings falls below 2*floor-1 vetoes promotion (0 "
                 "disables; on mirrors alone this equals win-rate < floor)"),
        Arg("--expert-decks", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated decks to ALSO write scripted:hard EXPERT "
                 "demonstration shards for each cycle (pi = one-hot expert "
                 "action): behavior-cloning targets for hand-coded combo lines "
                 "(e.g. league/wubg_doomsday) that neither PPO exploration nor "
                 "prior-guided search discovers"),
        Arg("--expert-games", "int", default=16,
            help="Expert matches per expert deck per cycle"),
        Arg("--seed", "int", default=1),
        Arg("--mirror-frac", "float", default=0.25,
            help="P(opponent deck == focus deck) per self-play game (default 0.25); "
                 "else a uniform league-roster draw"),
        Arg("--bo1", "flag",
            help="Run bo1 self-play + gate. The az cycle defaults to bo3 matches "
                 "with a per-game value target; this opts back into single games"),
        _actor_mode(),
    ]),
    Sub("az-league",
        "AlphaZero league: rotate az cycles (self-play -> train -> gate) over the "
        "decks/league/ roster (bo3 by default; --bo1 to opt out)", items=[
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
        Arg("--games", "int", default=50, help="Self-play games per cycle"),
        Arg("--sims", "int", default=256,
            help="Self-play PUCT sims, TOTAL across --worlds (256/4 = 64 per "
                 "determinized world tree)"),
        Arg("--worlds", "int", default=4),
        Arg("--sb-sims", "int", default=DEFAULT_SB_SIMS,
            help=f"PUCT sims at a bo3 sideboard root (bo3 only; default {DEFAULT_SB_SIMS})"),
        Arg("--sb-worlds", "int", default=4,
            help="Determinized worlds at a bo3 sideboard root (default 4)"),
        Arg("--sb-max-depth", "int", default=200,
            help="Rollout depth cap at a bo3 sideboard root (default 200)"),
        Arg("--workers", "int", default=None,
            help="Self-play worker processes (default max(1, cpu-2))"),
        Arg("--batches", "int", default=500),
        Arg("--batch-size", "int", default=256),
        Arg("--lr", "float", default=1e-3),
        Arg("--window", "int", default=50),
        Arg("--eval-games", "int", default=56,
            help="Total gate matches, split over the roster-wide panel (a mirror "
                 "per roster deck + direction-balanced cross pairs; default 56 "
                 "= 4 per matchup on a 10-deck roster)"),
        Arg("--eval-sims", "int", default=32),
        Arg("--eval-worlds", "int", default=2),
        Arg("--promote-threshold", "float", default=0.55),
        Arg("--gate-floor", "float", default=0.2,
            help="Per-piloted-deck gate floor: a deck the candidate piloted in "
                 ">=4 gate matches whose win-rate deficit vs the incumbent on "
                 "LIKE pairings falls below 2*floor-1 vetoes promotion (0 "
                 "disables; on mirrors alone this equals win-rate < floor)"),
        Arg("--gate-every", "int", default=1,
            help="Run the eval/gate every K slots instead of every slot: the "
                 "candidate accumulates K cycles of training (and "
                 "candidate-generated self-play) between promotions, and the "
                 "gate's wall-clock cost is paid 1/K as often. Candidate "
                 "snapshots still save every slot."),
        Arg("--matrix", "flag",
            help="Whole-roster focus MATRIX every slot instead of the per-deck "
                 "focus rotation: each cycle's self-play draws its focus deck "
                 "uniformly from the roster per game, keeping the training "
                 "window stationary (no one-deck-at-a-time forgetting sweep). "
                 "A rotation then counts --cycles-per-deck matrix cycles."),
        Arg("--expert-decks", "str", default=None, suggest="league_deck", multi=True,
            help="Comma-separated decks to ALSO write scripted:hard EXPERT "
                 "demonstration shards for each slot (pi = one-hot expert "
                 "action): behavior-cloning targets for hand-coded combo lines "
                 "(e.g. league/wubg_doomsday) that neither PPO exploration nor "
                 "prior-guided search discovers"),
        Arg("--expert-games", "int", default=16,
            help="Expert matches per expert deck per slot"),
        Arg("--seed", "int", default=1,
            help="Base RNG seed (slot i uses seed+i)"),
        Arg("--mirror-frac", "float", default=0.25,
            help="P(opponent deck == focus deck) per self-play game (default 0.25); "
                 "else a uniform league-roster draw"),
        Arg("--bo1", "flag",
            help="Run bo1 self-play + gate for every slot. The league defaults to "
                 "bo3 matches with a per-game value target; this opts back into "
                 "single games (persisted in the resume sidecar)"),
        _actor_mode(),
    ]),
])

# analysis.py — every command loads a trained model and simulates games (the
# .rmrec recording-file commands were removed; the live model-sim path is the
# single source). Two commands remain: 'report' is capture-mode (emits a
# self-contained HTML battery and exits), while 'interactive' opens the REPL
# (TUI hands over the terminal) — the REPL supersets every per-analysis view
# (cardvalue, shap, value-swings, regret, entropy, consistency, targeting,
# calibration, turning, clusters, whatif, …) and is the only mode with a live
# env for `run`/`whatif`. The former standalone analysis subcommands were thin
# wrappers over those same REPL views and were dropped.
ANALYSIS_TOOL = Tool("analysis", "train/analysis.py", subs=[
    Sub("report", "Run the standard battery and emit a single HTML report", items=[
        *sim_args(),
        Arg("--n-games", "int", default=50, help="Number of games to simulate (default: 50)"),
    ]),
    Sub("interactive",
        "Interactive session: simulate games then inspect replays, board states, "
        "value charts, SHAP, counterfactual whatif, and more", mode="interactive", items=[
            *sim_args(),
            Arg("--n-games", "int", default=20,
                help="Games to pre-simulate before entering session (default: 20; 0 = skip)"),
            Arg("--n-samples", "int", default=200, help="SHAP sample count (default: 200)"),
            Arg("--n-background", "int", default=50, help="SHAP background size (default: 50)"),
        ]),
    Sub("search",
        "Search-vs-raw comparison: per searched decision, net priors vs MCTS "
        "visit distribution and net value vs search root value (AZ or PPO ckpt)",
        items=[
            *sim_args(),
            Arg("--n-games", "int", default=4,
                help="Games to drive with the MCTS controller (default: 4)"),
            Arg("--sims", "int", default=64, help="PUCT simulations per decision (default: 64)"),
            Arg("--worlds", "int", default=4, help="Determinized worlds per search (default: 4)"),
            Arg("--sb-sims", "int", default=DEFAULT_SB_SIMS,
                help=f"bo3 sideboard-root sims (default: {DEFAULT_SB_SIMS})"),
            Arg("--sb-worlds", "int", default=DEFAULT_SB_WORLDS,
                help=f"bo3 sideboard-root determinized worlds (default: {DEFAULT_SB_WORLDS})"),
            Arg("--sb-max-depth", "int", default=DEFAULT_SB_MAX_DEPTH,
                help=f"bo3 sideboard-root rollout depth (default: {DEFAULT_SB_MAX_DEPTH})"),
            Arg("--c", "float", default=1.5, help="PUCT exploration constant c_puct (default: 1.5)"),
            Arg("--seed", "int", default=1, help="Base RNG/engine seed (game N uses seed+N; default: 1)"),
            Arg("--top", "int", default=8,
                help="Biggest prior-vs-visit disagreement decisions to decode (default: 8)"),
            Arg("--workers", "int", default=1,
                help="Parallel worker processes (default: 1 = sequential). Splits "
                     "--n-games evenly across processes, each with its own "
                     "evaluator/controller; results are merged before reporting."),
        ]),
])

# tui_analysis.py — the analysis REPL as a full-screen Textual app: game list,
# board-state pager (one decision step at a time), a clickable V(s) histogram
# for seeking, and every REPL analysis view. Same sim args as analysis.py minus
# the chart-output flags (charts stay in analysis.py's report/interactive).
ANALYSIS_TUI_TOOL = Tool("analysis-tui", "train/tui_analysis.py", flat=True, subs=[
    Sub("browse",
        "Full-screen analysis browser: page through board states with a "
        "clickable V(s) histogram, plus every analysis view", mode="interactive",
        items=[
            *[a for a in sim_args() if a.name not in ("--out", "--show")],
            Arg("--n-games", "int", default=20,
                help="Games to simulate on startup (default: 20)"),
        ]),
])

# play.py — interactive game; the TUI path delegates to tui_game.py (placeholder).
PLAY_TOOL = Tool("play", "train/play.py", flat=True, subs=[
    Sub("play", "Play interactively against a trained model", mode="interactive", items=[
        Arg("--human-deck", "str", required=True, suggest="deck",
            help="Deck the human plays (stem of .dk file)"),
        Arg("--model-deck", "str", required=True, suggest="deck",
            help="Deck the model plays (stem of .dk file). The default opponent is "
                 "the one generalist (gen__final.zip, else the newest gen__v*.zip) "
                 "piloting this deck."),
        Arg("--model", "str", default=None, suggest="agent",
            help="Override: explicit path to trained model .zip, or any "
                 "opponents.make_controller spec — az:gen (MCTS+AZNet), "
                 "azraw:gen (raw AZ policy), mcts:gen, scripted:<tier> "
                 "(default: the generalist gen__final.zip)"),
        Arg("--sims", "int", default=None,
            help="Search opponent only (az:/mcts: --model): MCTS simulations "
                 "per decision; overrides any sims= already in the spec (TUI only)"),
        Arg("--worlds", "int", default=None,
            help="Search opponent only: determinized worlds per decision "
                 "(sims are split across worlds); overrides the spec's worlds= (TUI only)"),
        Arg("--think-time", "float", default=None,
            help="Search opponent only (az:/mcts: --model): wall-clock seconds "
                 "per decision — the search runs as many simulations as fit in "
                 "this budget (more time = stronger play); overrides sims= as the "
                 "terminator (TUI only)"),
        Arg("--search-procs", "int", default=None,
            help="Search opponent only (az:/mcts: --model): number of engine "
                 "processes to fan the determinized worlds across for a faster "
                 "search (world-parallel; more procs = more sims/decision in the "
                 "same wall-clock). Default 1 (TUI only)"),
        Arg("--match-clock", "float", default=None,
            help="Search opponent only: total wall-clock thinking bank in "
                 "seconds for the WHOLE match (chess clock; 1500 = 25 min for "
                 "a bo3). Each decision draws a variable budget from the bank "
                 "— harder decisions earn more time, obvious ones stop early. "
                 "Appends clock= to the spec (TUI only)"),
        Arg("--paced", "flag",
            help="Mask opponent response-timing tells: a small jittered "
                 "(~0.02-0.05s) floor on every decision, plus occasional "
                 "0.2-0.5s fake-think pauses when the opponent was never even "
                 "offered a decision (default ON for a search opponent with "
                 "--match-clock/--think-time; TUI only)"),
        Arg("--no-paced", "flag",
            help="Disable the paced-response floor (instant obvious decisions)"),
        Arg("--tui", "flag", default=True, help="Launch the TUI game board (train/tui_game.py)"),
        Arg("--gui", "flag",
            help="Launch the PySide6 desktop game board (train/gui_game.py). "
                 "Takes precedence over --tui. Needs PySide6 (pip install -r "
                 "train/requirements-gui.txt); if it is missing, falls back to "
                 "the TUI when --tui is also set, else errors with the install hint."),
        Arg("--analysis", "flag",
            help="GUI only: open the analysis window (live MCTS evaluation of "
                 "your decisions on a detached engine copy; default evaluator "
                 "az:gen). The no-args GUI launcher has its own checkbox for this."),
        Arg("--scripted", "flag",
            help="Use the rule-based scripted agent as the opponent (no checkpoint needed; TUI only)"),
        Arg("--bo1", "flag",
            help="Play a single game instead of the default best-of-three match (TUI only)"),
        Arg("--player", "choice", choices=("A", "B"), default=None,
            help="Which player the human controls, in CLI text mode (default: random)"),
        Arg("--seed", "int", default=None,
            help="Engine RNG seed for a reproducible game (CLI text mode; default: random)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
    ]),
])

# test_harness.py — card-behaviour test harness (flat parser, no subcommand).
# Mirrors the argparse in test_harness.main(); the launcher composes a command
# and runs it in the real terminal (so --interactive's stdin prompts work).
HARNESS_TOOL = Tool("harness", "train/test_harness.py", flat=True, subs=[
    Sub("harness", "Run a card-behaviour scenario through the engine",
        mode="interactive", items=[
        Arg("--scenario", "str", help="Path to a JSON scenario file (supplies hands/library/etc.)"),
        Arg("--hand-a", "str", help="Player A starting hand (comma-separated card names)"),
        Arg("--library-a", "str", help="Player A library after the hand (comma-separated)"),
        Arg("--hand-b", "str", help="Player B starting hand (comma-separated card names)"),
        Arg("--library-b", "str", help="Player B library after the hand (comma-separated)"),
        Arg("--deck-a", "str", suggest="deck", help="Use an existing deck file for Player A (stem, not path)"),
        Arg("--deck-b", "str", suggest="deck", help="Use an existing deck file for Player B (stem, not path)"),
        Arg("--battlefield-a", "str", help="Cards pre-placed on Player A's battlefield (comma-separated)"),
        Arg("--battlefield-b", "str", help="Cards pre-placed on Player B's battlefield (comma-separated)"),
        Arg("--actions", "str", help="Comma-separated action indices to play (e.g. 9,0,7,0,8)"),
        Arg("--interactive", "flag", help="Prompt for an action index at each decision"),
        Arg("--scripted", "flag", help="Drive both sides with the rule-based scripted agent"),
        Arg("--no-shuffle", "flag",
            help="Don't shuffle libraries — deck-file order = draw order (implied by --hand-a/--hand-b)"),
        Arg("--seed", "int", default=None, help="RNG seed (default: 1, or the scenario's seed)"),
        Arg("--max-decisions", "int", default=None, help="Stop after N decisions (default: 500)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
    ]),
])

ALL_TOOLS = [TRAIN_TOOL, ANALYSIS_TOOL, ANALYSIS_TUI_TOOL, PLAY_TOOL, HARNESS_TOOL]


# ── argparse bridge (used by the scripts) ─────────────────────────────────────

def _add_one(target, a: Arg):
    if a.kind == "flag":
        target.add_argument(a.name, action="store_true", help=a.help)
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


def apply_to_parser(parser, sub: Sub):
    """Populate an argparse (sub)parser from a Sub spec."""
    for item in sub.items:
        if isinstance(item, MutexGroup):
            group = parser.add_mutually_exclusive_group(required=item.required)
            for a in item.args:
                _add_one(group, a)
        else:
            _add_one(parser, item)


def iter_args(sub: Sub):
    """Yield every Arg in a Sub, flattening MutexGroups."""
    for item in sub.items:
        if isinstance(item, MutexGroup):
            yield from item.args
        else:
            yield item
