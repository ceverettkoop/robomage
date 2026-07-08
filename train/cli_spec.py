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

# ── Canonical CLI constants (single home; imported by env.py / train.py) ──────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO_ROOT, "bin", "robomage")
BIN_DIR = os.path.join(REPO_ROOT, "bin")  # game must be run from here for resource lookup

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
TARGET_KL = 0.025

# Learning-rate decay schedule, keyed to a model's ABSOLUTE cumulative
# num_timesteps (not SB3's per-learn() progress_remaining, which resets every
# resume). LR falls linearly from LR_PEAK at step 0 to LR_FLOOR at
# LR_DECAY_STEPS, then stays flat at LR_FLOOR. Because it is anchored to the
# checkpoint's own step count, a resumed generalist that already trained N steps
# picks the schedule up at position N — so decay survives the many short resume
# sessions a per-deck generalist accumulates over. Applied by train.py's
# LRDecayCallback on every training path. See lr_for_timesteps().
LR_PEAK = 3e-4
LR_FLOOR = 5e-5
LR_DECAY_STEPS = 10_000_000


def lr_for_timesteps(num_timesteps: int) -> float:
    """Linear LR decay from LR_PEAK to LR_FLOOR over [0, LR_DECAY_STEPS] cumulative
    timesteps, clamped flat at LR_FLOOR beyond LR_DECAY_STEPS.

    Keyed to the model's absolute num_timesteps so the schedule is continuous
    across checkpoint resumes (a generalist resumed at N steps continues the same
    global decay curve rather than restarting it)."""
    if num_timesteps >= LR_DECAY_STEPS:
        return LR_FLOOR
    frac = max(0, num_timesteps) / LR_DECAY_STEPS  # 0.0 at step 0 → 1.0 at floor
    return LR_PEAK + (LR_FLOOR - LR_PEAK) * frac


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
# whatever the checkpoint was saved with; ent_coef and target_kl are re-asserted
# afterwards (train.py _reassert_hparams), and the LR is driven every rollout by
# LRDecayCallback regardless of the saved learning_rate. `learning_rate` here is
# just the nominal starting value the callback overrides on the first update.
PPO_KWARGS = dict(
    learning_rate=LR_PEAK,
    n_steps=4096,       # steps per env per update
    batch_size=1024,
    n_epochs=8,
    gamma=0.9975,     # 1/(1-γ) = 400-step horizon; episodes run 100-400 decisions with mostly terminal reward
    gae_lambda=0.97,  # keep γλ high enough that terminal reward reaches back directly in GAE
    clip_range=0.25,
    ent_coef=ENT_COEF,
    target_kl=TARGET_KL,
)
NET_ARCH = [256, 256]  # policy/value MLP head sizes (after the feature extractor)

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
    """A mutually-exclusive group of flags (rendered as one Select in the TUI)."""
    args: list
    required: bool = False


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
            help="Start the deck's generalist from scratch instead of auto-resuming "
                 "its existing {deck}__final.zip / newest {deck}__v*.zip (overwrites it)"),
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
    ]


def _opponent_mode():
    """The --self-play | --scripted mutually-exclusive pair (train/sweep)."""
    return MutexGroup([
        Arg("--self-play", "flag",
            help="Train against a frozen deck-pilot snapshot of the opponent deck "
                 "({opp}__v*.zip / {opp}__final.zip; falls back to the scripted "
                 "agent if none exists yet)"),
        Arg("--scripted", "flag",
            help="Train against the rule-based scripted agent (the default; "
                 "mutually exclusive with --self-play)"),
    ])


def opponent_pool_opts():
    """Mixed opponent-pool args shared by the train and sweep subcommands."""
    return [
        Arg("--opponent-pool", "str", default=None,
            help="Comma-separated mix of opponent controllers to randomize per "
                 "episode, e.g. 'scripted:easy,scripted:hard=2,mav'. "
                 "The token 'random-model' expands to a random generalist piloting "
                 "the opponent's deck (its deck-pilot snapshots {opp_deck}__v*.zip "
                 "/ {opp_deck}__final.zip). Each item may "
                 "carry an optional '=<weight>'. Overrides the plain scripted "
                 "opponent (ignored with --self-play). In a sweep the same pool "
                 "is applied to every matchup, resolving 'random-model' per matchup."),
        Arg("--opponent-ckpt-ratio", "float", default=1.0,
            help="Cap on unique opponent checkpoints kept resident, as a ratio of "
                 "n_envs (default 1.0 -> <=1 checkpoint per env process). Scripted "
                 "agents don't count toward the cap."),
    ]


def sim_args():
    """Common simulation args for analysis.py (was analysis.py _add_sim_args)."""
    return [
        Arg("model", "str", required=True, help="Path to model .zip", suggest="checkpoint"),
        Arg("--opponent", "str", required=True, suggest="checkpoint",
            help="Opponent model .zip path, or 'scripted' for rule-based agent"),
        Arg("--deck-a", "str", default=None, suggest="deck",
            help="Model's deck (.dk stem) — the deck it pilots. Inferred from the "
                 "model's deck-pilot filename ({deck}__final.zip) if omitted."),
        Arg("--deck-b", "str", default=None, suggest="deck",
            help="Opponent's deck (.dk stem). Per-deck checkpoints don't encode "
                 "their opponent, so supply this explicitly; for a model opponent "
                 "it is inferred from that model's own filename, and a scripted "
                 "opponent defaults to a mirror match (--deck-a)."),
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
    Sub("train", "Train a deck's generalist model (default command)", items=[
        Arg("--deck", "str", default="delver", suggest="deck",
            help="Deck the generalist plays (.dk stem, default: delver). Saved as "
                 "{deck}__final.zip; sessions against any opponent accumulate onto it."),
        Arg("--opponent", "str", required=True, suggest="deck",
            help="Opponent deck this session trains against (.dk stem). The model "
                 "stays a generalist — training vs one opponent continues the same "
                 "{deck}__final.zip rather than forging a matchup-specific model."),
        Arg("--load", "str", default=None, suggest="checkpoint",
            help="Resume from a specific checkpoint .zip (or shorthand), overriding "
                 "the default auto-resume of the deck's own generalist"),
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
        Arg("--pfsp-mode", "choice", choices=("pfsp", "softmax"), default="pfsp",
            help="Opponent quality weighting: 'pfsp' = (1-winrate)^p (AlphaStar) or "
                 "'softmax' = exp(q) with OpenAI-Five quality updates (default pfsp)."),
        Arg("--pfsp-p", "float", default=LEAGUE_PFSP_P,
            help="PFSP exponent p in (1-winrate)^p (default %.1f)." % LEAGUE_PFSP_P),
        Arg("--softmax-eta", "float", default=LEAGUE_SOFTMAX_ETA,
            help="Softmax quality learning rate eta (default %.3f)." % LEAGUE_SOFTMAX_ETA),
        Arg("--snapshot-every", "int", default=LEAGUE_SNAPSHOT_EVERY,
            help="Save a frozen {deck}__v{steps}.zip snapshot every N steps "
                 "(default %d)." % LEAGUE_SNAPSHOT_EVERY),
        Arg("--promote-margin", "float", default=LEAGUE_PROMOTE_MARGIN,
            help="Only keep a snapshot when the learner's recent-window win-rate "
                 ">= 0.5 + margin (negative gates below 0.5, e.g. -0.1 -> 0.40; the "
                 "first snapshot of each deck is exempt so self-play can bootstrap; "
                 "0 disables the gate; default %.2f)." % LEAGUE_PROMOTE_MARGIN),
        Arg("--rotate-every", "int", default=LEAGUE_ROTATE_EVERY,
            help="Steps to train one learner deck before rotating to the next "
                 "(default %d)." % LEAGUE_ROTATE_EVERY),
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
    Sub("sweep", "PFSP sweep: train one deck's generalist vs a pool of the other decks", items=[
        Arg("--deck", "str", required=True, suggest="deck",
            help="Deck to train (.dk stem). Saved as {deck}__final.zip; this session "
                 "accumulates onto it, same as 'train'."),
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
            help="Save a frozen {deck}__v{steps}.zip snapshot every N steps "
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
            help="Resume from checkpoint .zip (or shorthand)"),
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
        Arg("--player-a", "str", default="scripted", suggest="checkpoint",
            help="Player A controller: 'scripted' (or 'scripted:*') or a model .zip path/shorthand (default: scripted)"),
        Arg("--player-b", "str", default="scripted", suggest="checkpoint",
            help="Player B controller: 'scripted' (or 'scripted:*') or a model .zip path/shorthand (default: scripted)"),
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
            help="Model .zip path or shorthand (omit with --all)"),
        Arg("--games", "int", default=None,
            help="Games per matchup (default: 100 for a single model, 50 per opponent "
                 "with --all)"),
        Arg("--all", "flag",
            help="Round-robin every league deck's {deck}__final.zip vs scripted:hard on "
                 "every league deck (including the mirror): an N-deck roster runs N×N "
                 "matchups of --games each, and the per-matchup win rates are appended "
                 "to the report log"),
        Arg("--log", "str", default=None,
            help="Report file for --all (default: checkpoints/baseline_report.log, appended)"),
        Arg("--deck", "str", default=None, suggest="deck",
            help="Deck the model pilots (default: inferred from the checkpoint's "
                 "deck-pilot filename). The scripted opponent mirrors it."),
        Arg("--seed", "int", default=None,
            help="RNG seed for reproducible runs (game N uses seed+N; default: random)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
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
])

# play.py — interactive game; the TUI path delegates to tui_game.py (placeholder).
PLAY_TOOL = Tool("play", "train/play.py", flat=True, subs=[
    Sub("play", "Play interactively against a trained model", mode="interactive", items=[
        Arg("--human-deck", "str", required=True, suggest="deck",
            help="Deck the human plays (stem of .dk file)"),
        Arg("--model-deck", "str", required=True, suggest="deck",
            help="Deck the model plays (stem of .dk file). Per-deck checkpoints "
                 "are keyed on this alone: auto-loads checkpoints/<model-deck>__final.zip "
                 "(else the newest <model-deck>__v*.zip, else a legacy matchup file)."),
        Arg("--model", "str", default=None, suggest="checkpoint",
            help="Override: explicit path to trained model .zip "
                 "(default: checkpoints/<model-deck>__final.zip)"),
        Arg("--tui", "flag", default=True, help="Launch the TUI game board (train/tui_game.py)"),
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

ALL_TOOLS = [TRAIN_TOOL, ANALYSIS_TOOL, PLAY_TOOL, HARNESS_TOOL]


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
