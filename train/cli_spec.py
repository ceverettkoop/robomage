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

# League (PFSP) defaults.
LEAGUE_SELF_PLAY_FRAC      = 0.8      # prob of facing the latest snapshot of the learner's own deck
LEAGUE_SCRIPTED_ANCHOR_FRAC = 0.1    # min share of the historical pool reserved for the scripted anchor
LEAGUE_PFSP_P              = 2.0      # exponent p in (1-winrate)^p
LEAGUE_SOFTMAX_ETA         = 0.01    # softmax quality learning rate
LEAGUE_SNAPSHOT_EVERY      = 250_000 # steps between frozen snapshots
LEAGUE_PROMOTE_MARGIN      = 0.2     # only snapshot when recent win-rate >= 0.5 + margin (first exempt; 0 disables)
LEAGUE_ROTATE_EVERY        = 500_000 # steps to train one learner deck before rotating


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
    suggest: str = None   # autocomplete source: "deck" | "checkpoint" | "recording" | None
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
        Arg("--record", "flag",
            help="Record all game decisions to a .rmrec binary file in recordings/"),
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
            help="Train against a frozen saved checkpoint of the mirror matchup "
                 "(falls back to the scripted agent if none exists yet)"),
        Arg("--scripted", "flag",
            help="Train against the rule-based scripted agent (the default; "
                 "mutually exclusive with --self-play)"),
    ])


def opponent_pool_opts():
    """Mixed opponent-pool args shared by the train and sweep subcommands."""
    return [
        Arg("--opponent-pool", "str", default=None,
            help="Comma-separated mix of opponent controllers to randomize per "
                 "episode, e.g. 'scripted:easy,scripted:hard=2,delver_mav_final'. "
                 "The token 'random-model' expands to a random checkpoint "
                 "compatible with the matchup (a model trained to pilot the "
                 "opponent's deck, i.e. {opp_deck}_{deck}_*.zip). Each item may "
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
            help="Model's deck (.dk stem). Inferred from model filename if omitted."),
        Arg("--deck-b", "str", default=None, suggest="deck",
            help="Opponent's deck (.dk stem). Inferred from opponent filename if omitted; "
                 "defaults to --deck-a for scripted opponent."),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
        Arg("--bo3", "flag",
            help="Run best-of-three matches (decks must include SIDEBOARD entries)"),
    ]


# ── Tool definitions ──────────────────────────────────────────────────────────

TRAIN_TOOL = Tool("train", "train/train.py", default_sub="train", subs=[
    Sub("train", "Train a model (default command)", items=[
        Arg("--deck", "str", default="delver", suggest="deck",
            help="Deck the model plays (.dk stem, default: delver)"),
        Arg("--opponent", "str", required=True, suggest="deck", help="Opponent deck (.dk stem)"),
        Arg("--load", "str", default=None, suggest="checkpoint",
            help="Resume from checkpoint .zip (or shorthand)"),
        _opponent_mode(),
        *opponent_pool_opts(),
        *train_opts(),
        *common_args(),
    ]),
    Sub("league", "PFSP league: train one generalist model per deck vs the whole field", items=[
        Arg("--decks", "str", default=None, suggest="deck", multi=True,
            help="Comma-separated deck roster to train + sample opponents from "
                 "(default: every .dk deck). In the TUI, pick multiple with space."),
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
            help="Only keep a snapshot when the learner's recent win-rate >= "
                 "0.5 + margin (the first snapshot of each deck is exempt so "
                 "self-play can bootstrap; 0 disables the gate; default %.2f)." % LEAGUE_PROMOTE_MARGIN),
        Arg("--rotate-every", "int", default=LEAGUE_ROTATE_EVERY,
            help="Steps to train one learner deck before rotating to the next "
                 "(default %d)." % LEAGUE_ROTATE_EVERY),
        Arg("--opponent-ckpt-ratio", "float", default=1.0,
            help="Cap on unique opponent checkpoints kept resident, as a ratio of "
                 "n_envs (default 1.0 -> <=1 checkpoint per env process)."),
        *train_opts(),
        *common_args(),
    ]),
    Sub("sweep", "Train many deck×deck matchups sequentially", items=[
        Arg("--deck", "str", default=None, suggest="deck",
            help="Only matchups featuring this deck. Omit to train ALL matchups."),
        _opponent_mode(),
        *opponent_pool_opts(),
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
    Sub("diag", "Run quick games (scripted vs scripted) to verify the env", items=[
        Arg("--deck", "str", default="delver", suggest="deck", help="Player A deck (.dk stem)"),
        Arg("--opponent", "str", default=None, suggest="deck", help="Player B deck (.dk stem)"),
        Arg("--games", "int", default=10, help="Number of games (default: 10)"),
        *common_args(),
    ]),
    Sub("watch", "Watch one game: scripted A vs scripted B", items=[
        Arg("--deck", "str", default="delver", suggest="deck", help="Player A deck (.dk stem)"),
        Arg("--opponent", "str", default=None, suggest="deck", help="Player B deck (.dk stem)"),
        *common_args(),
    ]),
    Sub("observe", "Watch one game with chosen controllers and decks for each side", items=[
        Arg("--player-a", "str", default="scripted", suggest="checkpoint",
            help="Player A controller: 'scripted' or a model .zip path/shorthand (default: scripted)"),
        Arg("--player-b", "str", default="scripted", suggest="checkpoint",
            help="Player B controller: 'scripted' or a model .zip path/shorthand (default: scripted)"),
        Arg("--deck", "str", default="delver", suggest="deck", help="Player A deck (.dk stem, default: delver)"),
        Arg("--opponent", "str", default="delver", suggest="deck", help="Player B deck (.dk stem, default: delver)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
    ]),
    Sub("baseline", "Evaluate a model's win rate vs the scripted agent", items=[
        Arg("model", "str", required=True, suggest="checkpoint", help="Model .zip path or shorthand"),
        Arg("--games", "int", default=100, help="Number of games (default: 100)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
    ]),
])

# analysis.py — .rmrec commands are non-interactive (capture); the simulation
# commands enter a REPL afterwards (interactive → TUI hands over the terminal).
ANALYSIS_TOOL = Tool("analysis", "train/analysis.py", subs=[
    Sub("summary", "Session summary and statistics",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("winrate", "Win rate plot over time",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("actions", "Action category heatmap by game step",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("cards", "Card usage bar chart",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("replay", "Replay a single game", items=[
        Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file"),
        Arg("--game", "int", required=True, help="Game ID to replay"),
    ]),
    Sub("compare", "Compare win rates from two sessions", items=[
        Arg("file", "str", required=True, suggest="recording", help="First .rmrec recording file"),
        Arg("file2", "str", required=True, suggest="recording", help="Second .rmrec recording file"),
    ]),
    Sub("wl-split", "Action distribution split by win/loss",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("cast-timing", "Per-card cast timing and state by outcome",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("choice-rates", "P(chose X | X legal) by board state",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("targeting", "Targeting self vs opp, hold vs cast analysis",
        items=[Arg("file", "str", required=True, suggest="recording", help="Path to .rmrec recording file")]),
    Sub("shap", "SHAP analysis of value function over simulated games", mode="interactive", items=[
        *sim_args(),
        Arg("--n-games", "int", default=50, help="Number of games to simulate (default: 50)"),
        Arg("--n-samples", "int", default=200, help="SHAP sample count (default: 200)"),
        Arg("--n-background", "int", default=50, help="SHAP background size (default: 50)"),
    ]),
    Sub("value-swings", "Find games with largest value function swings", mode="interactive", items=[
        *sim_args(),
        Arg("--n-games", "int", default=50, help="Number of games to simulate (default: 50)"),
        Arg("--top", "int", default=10, help="Show top N swings (default: 10)"),
    ]),
    Sub("regret", "Action regret analysis using policy distribution", mode="interactive", items=[
        *sim_args(),
        Arg("--n-games", "int", default=50, help="Number of games to simulate (default: 50)"),
        Arg("--top", "int", default=20, help="Show top N high-regret decisions (default: 20)"),
    ]),
    Sub("entropy", "Policy entropy by game phase and board state", mode="interactive", items=[
        *sim_args(),
        Arg("--n-games", "int", default=50, help="Number of games to simulate (default: 50)"),
    ]),
    Sub("consistency", "Decision consistency for similar game states", mode="interactive", items=[
        *sim_args(),
        Arg("--n-games", "int", default=50, help="Number of games to simulate (default: 50)"),
        Arg("--top", "int", default=20, help="Show top N inconsistent pairs (default: 20)"),
    ]),
    Sub("interactive",
        "Interactive session: simulate games then inspect replays, board states, "
        "value charts, SHAP, and more", mode="interactive", items=[
            *sim_args(),
            Arg("--n-games", "int", default=20,
                help="Games to pre-simulate before entering session (default: 20; 0 = skip)"),
            Arg("--n-samples", "int", default=200, help="SHAP sample count (default: 200)"),
            Arg("--n-background", "int", default=50, help="SHAP background size (default: 50)"),
        ]),
])

# play.py — interactive game; the TUI path delegates to tui_game.py (placeholder).
PLAY_TOOL = Tool("play", "train/play.py", subs=[
    Sub("play", "Play interactively against a trained model", mode="interactive", items=[
        Arg("--human-deck", "str", required=True, suggest="deck",
            help="Deck the human plays (stem of .dk file)"),
        Arg("--model-deck", "str", required=True, suggest="deck",
            help="Deck the model plays (stem of .dk file). "
                 "Automatically loads checkpoints/<model-deck>_<human-deck>_final.zip"),
        Arg("--model", "str", default=None, suggest="checkpoint",
            help="Override: explicit path to trained model .zip "
                 "(default: checkpoints/<model-deck>_<human-deck>_final.zip)"),
        Arg("--gui", "flag", help="Launch raylib GUI window for human input"),
        Arg("--tui", "flag", help="Launch the TUI game board (train/tui_game.py)"),
        Arg("--scripted", "flag",
            help="Use the rule-based scripted agent as the opponent (no checkpoint needed; TUI only)"),
        Arg("--player", "choice", choices=("A", "B"), default=None,
            help="Which player the human controls, in both CLI and GUI modes (default: random)"),
        Arg("--binary", "str", default=BINARY, help="Path to robomage binary"),
    ]),
])

ALL_TOOLS = [TRAIN_TOOL, ANALYSIS_TOOL, PLAY_TOOL]


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
