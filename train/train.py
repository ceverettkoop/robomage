"""
Self-play PPO training for RoboMage.

Dependencies:
    pip install gymnasium stable-baselines3 sb3-contrib

sb3-contrib provides MaskablePPO, which respects the action_masks() method
from the environment so the agent never wastes probability mass on illegal actions.

Usage:
    # From the robomage repo root:
    cd train
    python train.py

    # Override binary path:
    python train.py --binary ../bin/robomage

    # Models are per-deck generalists ({deck}__final.zip / {deck}__v{steps}.zip):
    # a training session auto-resumes and continues the deck's one model, so
    # training vs a single opponent just generalizes it further.
    python train.py --deck delver --opponent mav    # continue delver's generalist
    python train.py --deck delver --opponent mav --fresh    # start it over
"""

import argparse
import datetime
import json
import os
import re
import sys
from collections import deque

from env import (RoboMageEnv, ModelVsScriptedEnv, SelfPlayEnv, FixedModelEnv, NarrativeEnv,
                 scripted_action,
                 OBS_SIZE, STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, BINARY)
from extractor import CardGameExtractor, PerActionMaskablePolicy
from card_costs import N_CARD_TYPES
import decode
# Action-category / step display names are generated from the C++ enums
# (train/gen_enums.py) and shared via decode.py — re-exported here so existing
# `from train import _CAT_NAMES, _STEP_NAMES` consumers (analysis.py) keep working.
from _enums import _CAT_NAMES, _STEP_NAMES
# CLI definitions + training defaults live in cli_spec.py (single source shared with the TUI).
from cli_spec import (TOTAL_TIMESTEPS, N_ENVS, N_ENVS_SELF_PLAY, EMBED_DIM,
                      ENT_COEF, TARGET_KL, lr_for_timesteps, PPO_KWARGS, NET_ARCH,
                      LEAGUE_SELF_PLAY_FRAC, LEAGUE_SCRIPTED_ANCHOR_FRAC,
                      LEAGUE_PFSP_P, LEAGUE_SOFTMAX_ETA, LEAGUE_SNAPSHOT_EVERY,
                      LEAGUE_PROMOTE_MARGIN, LEAGUE_ROTATE_EVERY,
                      LEAGUE_ADAPTIVE_BOOST, parse_shard, shard_tag,
                      TRAIN_TOOL, apply_to_parser)

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    USE_MASKABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    USE_MASKABLE = False
    print("Warning: sb3-contrib not found, using plain PPO (no action masking).")
    print("Install with: pip install sb3-contrib")

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import ConstantSchedule
from stable_baselines3.common.logger import configure as sb3_configure_logger

import numpy as np

# ── Per-action logit head (prototype, opt-in) ────────────────────────────────
# Set ROBOMAGE_PER_ACTION_HEAD=1 to train with PerActionMaskablePolicy, which
# scores each candidate action from its own encoded features (category + target
# card embedding + controller_is_self) instead of a flat positional Linear. This
# changes the network shape, so such runs are NOT checkpoint-compatible with
# stock MlpPolicy models — start them --fresh.
USE_PER_ACTION_HEAD = os.environ.get("ROBOMAGE_PER_ACTION_HEAD", "0").lower() \
    not in ("0", "", "false", "no")


def _policy_config(policy_kwargs):
    """Resolve (policy, policy_kwargs) for MaskablePPO construction.

    Swaps in the per-action-logit head (and flips the extractor into
    per_action_head mode) when ROBOMAGE_PER_ACTION_HEAD is set; otherwise returns
    the stock "MlpPolicy" untouched.
    """
    if not (USE_PER_ACTION_HEAD and USE_MASKABLE):
        return "MlpPolicy", policy_kwargs
    pk = dict(policy_kwargs)
    fe = dict(pk.get("features_extractor_kwargs", {}))
    fe["per_action_head"] = True
    pk["features_extractor_kwargs"] = fe
    return PerActionMaskablePolicy, pk


class WinTallyCallback(BaseCallback):
    """Prints win rate since the last rollout, broken down by opponent deck."""

    def __init__(self):
        super().__init__()
        self._matchups: dict[str, list[int]] = {}  # deck -> [wins, losses]

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            r = info["episode"]["r"]
            if r == 0:
                continue
            deck = info.get("opp_deck", "unknown")
            if deck not in self._matchups:
                self._matchups[deck] = [0, 0]
            if r > 0:
                self._matchups[deck][0] += 1
            else:
                self._matchups[deck][1] += 1
        return True

    def _on_rollout_end(self) -> None:
        if not self._matchups:
            return
        total_w = total_l = 0
        for deck in sorted(self._matchups):
            w, l = self._matchups[deck]
            total = w + l
            pct = 100.0 * w / total
            print(f"[tally] vs {deck}: {w}W {l}L ({pct:.1f}%)")
            total_w += w
            total_l += l
        grand = total_w + total_l
        print(f"[tally] overall: {total_w}W {total_l}L ({100.0 * total_w / grand:.1f}%)")
        self._matchups.clear()


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class PFSPCallback(BaseCallback):
    """Win-rate feedback loop for the PFSP league.

    Generalises WinTallyCallback: attributes each finished episode to the exact
    opponent entry it was played against ((opp_deck, controller_label) from
    ``info['game_meta']``) and, after every rollout, broadcasts updated per-entry
    quality weights to every env so the LeaguePool faces what the learner is
    currently losing to.

    Two weighting modes (the mode lives here, not in the pool):
      * ``pfsp``    — weight ∝ (1 - winrate)^p   (AlphaStar).
      * ``softmax`` — weight ∝ exp(q); on a win vs entry i, q_i -= eta/(N·p_i)
        (OpenAI Five Appendix N); losses leave q unchanged. New entries start at
        the current max q so fresh snapshots get tried.

    Weights are pushed with ``vec_env.env_method`` (not ``set_attr``): env_method
    resolves through the wrapper chain to the inner ModelVsScriptedEnv, whereas
    set_attr would only touch the outer Monitor wrapper.
    """

    def __init__(self, vec_env, mode: str = "pfsp", p: float = 2.0, eta: float = 0.01,
                 recent_window: int = 200):
        super().__init__()
        self._vec_env = vec_env
        self._mode = mode
        self._p = p
        self._eta = eta
        self._stats: dict[tuple, list[int]] = {}  # (opp_deck, label) -> [wins, losses]
        self._q: dict[tuple, float] = {}          # (opp_deck, label) -> quality (softmax)
        # Sliding window of the most recent decisive episode outcomes (1.0 win /
        # 0.0 loss), used by the snapshot promotion gate so it reflects *current*
        # strength rather than the cumulative-since-chunk-start average.
        self._recent: deque = deque(maxlen=max(1, recent_window))

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            r = info["episode"]["r"]
            if r == 0:
                continue
            meta = info.get("game_meta") or {}
            key = (meta.get("opp_deck", "unknown"), meta.get("opp_type", "scripted"))
            wl = self._stats.setdefault(key, [0, 0])
            if key not in self._q:
                # New entry: init quality to the current max so it gets sampled.
                self._q[key] = max(self._q.values()) if self._q else 0.0
            self._recent.append(1.0 if r > 0 else 0.0)
            if r > 0:
                wl[0] += 1
                if self._mode == "softmax":
                    keys = list(self._q)
                    probs = _softmax(np.array([self._q[k] for k in keys], dtype=float))
                    p_i = float(probs[keys.index(key)])
                    self._q[key] -= self._eta / (len(keys) * max(p_i, 1e-8))
            else:
                wl[1] += 1
        return True

    def _weight_for(self, key) -> float:
        if self._mode == "softmax":
            return float(np.exp(self._q.get(key, 0.0)))
        w, l = self._stats.get(key, (0, 0))
        total = w + l
        winrate = w / total if total else 0.0
        return float((1.0 - winrate) ** self._p)

    def overall_winrate(self) -> float:
        """Aggregate learner win-rate across all opponents seen so far (lifetime)."""
        w = sum(v[0] for v in self._stats.values())
        l = sum(v[1] for v in self._stats.values())
        return w / (w + l) if (w + l) else 0.0

    def recent_winrate(self) -> tuple[float, int]:
        """Learner win-rate over the recent-episode window, plus the sample count.

        This is the snapshot promotion gate's input: a sliding window so the gate
        tracks the policy's *current* strength instead of being dragged down by
        weak early-chunk games. Returns ``(winrate, n_samples)``."""
        n = len(self._recent)
        return (sum(self._recent) / n if n else 0.0), n

    def _on_rollout_end(self) -> None:
        if not self._stats:
            return
        weights = {key: self._weight_for(key) for key in self._stats}
        # Broadcast to every env's LeaguePool (no-op for non-league envs).
        try:
            self._vec_env.env_method("update_opponent_weights", weights)
        except Exception as exc:
            print(f"[pfsp] WARNING: failed to broadcast weights ({exc})")
        # Matchup win-rate matrix for visibility (extends --tally).
        by_deck: dict[str, list[int]] = {}
        for (deck, label), (w, l) in sorted(self._stats.items()):
            total = w + l
            pct = 100.0 * w / total if total else 0.0
            print(f"[pfsp] vs {deck:<10} [{label}]: {w}W {l}L ({pct:.1f}%)  "
                  f"weight={self._weight_for((deck, label)):.3f}")
            d = by_deck.setdefault(deck, [0, 0])
            d[0] += w
            d[1] += l
        for deck in sorted(by_deck):
            w, l = by_deck[deck]
            total = w + l
            pct = 100.0 * w / total if total else 0.0
            print(f"[pfsp] deck total vs {deck:<10}: {w}W {l}L ({pct:.1f}%)")
        rwr, rn = self.recent_winrate()
        print(f"[pfsp] overall win-rate: {100.0 * self.overall_winrate():.1f}%  "
              f"recent (n={rn}): {100.0 * rwr:.1f}%")


class SnapshotCallback(BaseCallback):
    """Saves a frozen ``{deck}__v{steps}.zip`` snapshot every ``snapshot_every`` steps.

    Optional SIMPLE-style promotion gate: when ``promote_margin != 0`` a snapshot is
    only kept if the learner's *recent-window* win-rate (from ``pfsp_callback``) is
    at least ``0.5 + promote_margin``, so the pool collects genuinely stronger
    snapshots rather than near-duplicates. The window (not the cumulative average)
    is used so a deck that started weak can still promote once it is *currently*
    strong; the gate is also skipped until the window holds at least
    ``min_gate_samples`` decisive games (bias toward feeding the pool over
    starving it). A negative margin gates below 50% (e.g. -0.1 keeps snapshots
    once win-rate clears 40%), useful for decks that are slow to break even; ``0``
    disables the gate entirely. The first snapshot of each deck is exempt so
    self-play can bootstrap. Drops are logged (no silent truncation).
    """

    def __init__(self, checkpoint_dir: str, deck: str, snapshot_every: int,
                 promote_margin: float = 0.0, pfsp_callback: "PFSPCallback | None" = None,
                 min_gate_samples: int = 30, on_snapshot=None):
        super().__init__()
        self._dir = checkpoint_dir
        self._deck = deck
        self._every = max(1, snapshot_every)
        self._margin = promote_margin
        self._pfsp = pfsp_callback
        self._min_gate_samples = min_gate_samples
        # Optional callable(num_timesteps) run after a snapshot is saved — the league
        # driver uses it to persist resumable progress alongside each checkpoint.
        self._on_snapshot = on_snapshot
        self._next_at = None  # set on training start relative to current num_timesteps

    def _on_training_start(self) -> None:
        # Align to the next multiple of `every` above the current step count so a
        # resumed learner doesn't immediately re-snapshot.
        n = self.num_timesteps
        self._next_at = ((n // self._every) + 1) * self._every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_at:
            return True
        self._next_at += self._every
        from opponents import deck_snapshots
        first = len(deck_snapshots(self._deck, self._dir)) == 0
        if self._margin != 0 and not first and self._pfsp is not None:
            wr, n = self._pfsp.recent_winrate()
            # Too few decisive games in the window to judge current strength: don't
            # block (bias toward keeping the pool fed rather than starving it).
            if n >= self._min_gate_samples and wr < 0.5 + self._margin:
                print(f"[snapshot] gate: {self._deck} recent win-rate {wr:.2f} "
                      f"(n={n}) < {0.5 + self._margin:.2f}; skipping snapshot at "
                      f"{self.num_timesteps} steps")
                return True
        path = os.path.join(self._dir, f"{self._deck}__v{self.num_timesteps}")
        self.model.save(path)
        print(f"[snapshot] saved {self._deck}__v{self.num_timesteps}.zip")
        if self._on_snapshot is not None:
            self._on_snapshot(self.num_timesteps)
        return True


class ShapingScaleCallback(BaseCallback):
    """After each rollout, sets shaping_scale on all envs to (1 - win_rate).

    A 25% win rate → 75% shaping; 100% win rate → 0% shaping.
    Requires at least one completed game before it takes effect; scale
    stays at 1.0 until then.
    """

    def __init__(self, vec_env):
        super().__init__()
        self._vec_env = vec_env
        self._wins = 0
        self._losses = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            r = info["episode"]["r"]
            if r > 0:
                self._wins += 1
            elif r < 0:
                self._losses += 1
        return True

    def _on_rollout_end(self) -> None:
        total = self._wins + self._losses
        if total == 0:
            return
        win_rate = self._wins / total
        scale = 1.0 - win_rate
        # env_method (not set_attr): set_attr only sets the outer Monitor wrapper
        # and never reaches the inner env's shaping_scale. See env.set_shaping_scale.
        self._vec_env.env_method("set_shaping_scale", scale)
        print(f"[shaping] win_rate={win_rate:.2f}  shaping_scale={scale:.2f}")
        self._wins = 0
        self._losses = 0


class ReplayLogCallback(BaseCallback):
    """After each rollout, runs one model-vs-scripted game and saves a transcript."""

    def __init__(self, binary_path: str, replay_dir: str = "replays",
                 model_deck: str | None = None, opp_deck: str | None = None,
                 bo3: bool = False):
        super().__init__()
        self.binary_path = binary_path
        self.replay_dir = replay_dir
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._bo3 = bo3
        self._rollout = 0
        os.makedirs(replay_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import numpy as np
        import runner
        from opponents import ModelController, ScriptedController
        from scripted_agent import make_agent

        self._rollout += 1
        log_path = os.path.join(self.replay_dir, f"rollout_{self._rollout:05d}.txt")

        model_is_a = bool(np.random.random() < 0.5)
        ctrl_model = ModelController(self.model, label="Model", deterministic=True)
        ctrl_scripted = ScriptedController(make_agent("scripted"), label="Scripted")
        if self._model_deck is not None:
            deck_a = self._model_deck if model_is_a else self._opp_deck
            deck_b = self._opp_deck if model_is_a else self._model_deck
        else:
            deck_a = deck_b = None
        ctrl_a, ctrl_b = ((ctrl_model, ctrl_scripted) if model_is_a
                          else (ctrl_scripted, ctrl_model))
        label_a, label_b = (("Model", "Scripted") if model_is_a
                            else ("Scripted", "Model"))

        try:
            with open(log_path, "w") as f:
                model_side = "A" if model_is_a else "B"
                scripted_side = "B" if model_is_a else "A"
                f.write(f"=== Rollout {self._rollout}: Model ({model_side}) "
                        f"vs Scripted ({scripted_side}) ===\n\n")
                wins, losses, _draws = runner.run_games(
                    ctrl_a, ctrl_b, label_a=label_a, label_b=label_b,
                    binary_path=self.binary_path, deck_a=deck_a, deck_b=deck_b,
                    n_games=1, bo3=self._bo3, out=f)
                model_won = wins if model_is_a else losses
                model_lost = losses if model_is_a else wins
                unit = " match" if self._bo3 else ""
                result = (f"Model wins{unit}" if model_won
                          else f"Scripted wins{unit}" if model_lost else "Draw")
                f.write(f"\n=== {result} ===\n")

            print(f"[replay] rollout {self._rollout}: {result} -> {log_path}")
        except Exception as exc:
            print(f"[replay] rollout {self._rollout}: game failed ({exc})")


CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
_CHECKPOINT_ABS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def _reassert_hparams(model, label: str):
    """Re-assert canonical hyperparameters on a resumed checkpoint.

    MaskablePPO.load restores the values the checkpoint was saved with, so a
    model created under an old (or unset) value would otherwise carry it across
    every future resume. Covers:
      * ent_coef — a stale 0.12 rode along in old checkpoints (see cli_spec).
      * target_kl — checkpoints predating the KL early-stop were saved with
        target_kl=None and would never early-stop without this.
    The learning rate is NOT set here: it is driven every rollout by
    LRDecayCallback (keyed to num_timesteps), which overrides whatever LR the
    checkpoint restored. Each override is logged when the stored value differs."""
    if model.ent_coef != ENT_COEF:
        print(f"[hyperparam] {label}: overriding stale ent_coef "
              f"{model.ent_coef} -> {ENT_COEF}")
        model.ent_coef = ENT_COEF
    if model.target_kl != TARGET_KL:
        print(f"[hyperparam] {label}: overriding stale target_kl "
              f"{model.target_kl} -> {TARGET_KL}")
        model.target_kl = TARGET_KL
    return model


class LRDecayCallback(BaseCallback):
    """Drive the learning rate from a schedule keyed to ABSOLUTE num_timesteps.

    SB3's native ``learning_rate`` schedule is a function of ``progress_remaining``,
    which resets to 1.0 at the start of every ``learn()`` call — so it is blind to
    how many steps a resumed checkpoint already trained and would restart the decay
    each short resume session. This callback instead sets ``model.lr_schedule`` to
    a constant computed from ``lr_for_timesteps(model.num_timesteps)`` before each
    optimizer update, giving a single global decay curve that survives resume
    (a generalist resumed at N steps continues from position N).

    ``ConstantSchedule`` (not a lambda) is used so the assigned ``lr_schedule``
    stays picklable — ``model.save()`` serializes it, and a closure capturing the
    model would recursively drag the whole model into the checkpoint. It is
    re-assigned every rollout, so the constant only holds for the one update that
    follows; ``_update_learning_rate`` then applies it to the optimizer and logs
    ``train/learning_rate``.
    """

    def _apply(self) -> float:
        lr = lr_for_timesteps(self.model.num_timesteps)
        self.model.lr_schedule = ConstantSchedule(lr)
        return lr

    def _on_training_start(self) -> None:
        lr = self._apply()
        if self.verbose:
            print(f"[lr] start num_timesteps={self.model.num_timesteps:,} -> lr={lr:.2e}")

    def _on_rollout_end(self) -> None:
        # Fires after each rollout's collection (num_timesteps updated) and before
        # the following train(), so that update uses the step-appropriate LR.
        self._apply()

    def _on_step(self) -> bool:
        return True


def _tb_log_name(deck: str) -> str:
    """Base rollout log name for a training session: '{deck}_{YYYY-MM-DD}'.

    A trailing '_{run_number}' is appended by _configure_run_logger so repeated
    sessions on the same day land in distinct 'delver_2026-07-03_1',
    '..._2', ... folders instead of overwriting one another.
    """
    return f"{deck}_{datetime.date.today().isoformat()}"


def _next_run_id(base: str) -> int:
    """Next free trailing run number for a '{base}_{n}' log folder under LOG_DIR.

    Scans the (possibly nested — e.g. 'league/<deck>_<date>') parent directory
    for existing '{leaf}_{n}' run folders and returns max(n) + 1, so a fresh
    session always claims an unused number. We roll our own instead of SB3's
    get_latest_run_id because that helper mis-parses names containing a path
    separator (league decks like 'league/bug') and always returns 0 for them."""
    parent = os.path.join(LOG_DIR, os.path.dirname(base))
    leaf = os.path.basename(base)
    max_id = 0
    if os.path.isdir(parent):
        prefix = leaf + "_"
        for name in os.listdir(parent):
            if name.startswith(prefix):
                tail = name[len(prefix):]
                if tail.isdigit():
                    max_id = max(max_id, int(tail))
    return max_id + 1


def _configure_run_logger(model, deck: str) -> str:
    """Give this session its own tensorboard run folder, incrementing the trailing
    number so a resumed session (reset_num_timesteps=False) does NOT overwrite the
    previous run's curve.

    SB3 couples the log-dir suffix to reset_num_timesteps: on a resume it reuses
    the last '{name}_{n}' folder ("continue the same curve"), which collides every
    same-date session — the common case for auto-resuming per-deck generalists.
    We need num_timesteps to keep counting (reset stays False) while the log dir
    stays fresh, so we compute the next '{deck}_{date}_{n}' ourselves and install
    the logger directly; set_logger marks it custom, so _setup_learn skips SB3's
    reset-coupled auto-config and honours this folder verbatim. Returns the run
    name for logging."""
    base = _tb_log_name(deck)
    run_name = f"{base}_{_next_run_id(base)}"
    save_path = os.path.join(LOG_DIR, run_name)
    fmt = ["stdout", "tensorboard"] if model.verbose >= 1 else ["tensorboard"]
    model.set_logger(sb3_configure_logger(save_path, fmt))
    return run_name

# League resume: the driver's loop position (which deck is up, how many global steps
# are done) lives outside any single model checkpoint, so we persist it to a small
# JSON sidecar that is rewritten every time a snapshot is saved (and at each rotation
# boundary). A crashed/interrupted `league` run then resumes from `league --resume`.
LEAGUE_STATE_VERSION = 1


def _league_state_path(checkpoint_dir: str, tag: str = "") -> str:
    # `tag` namespaces the sidecar per roster shard ('' for a single-machine
    # run, '.shard{i}of{n}' for distributed shards — see cli_spec.shard_tag),
    # so two drivers sharing a synced checkpoint dir never clobber each other.
    return os.path.join(checkpoint_dir, f"_league_progress{tag}.json")


def _write_league_state(checkpoint_dir: str, state: dict, tag: str = "") -> None:
    """Atomically persist the league driver's progress to its JSON sidecar."""
    path = _league_state_path(checkpoint_dir, tag)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[league] WARNING: could not write progress file {path}: {exc}")


def _read_league_state(checkpoint_dir: str, tag: str = "") -> dict | None:
    """Load the league progress sidecar, or None if it is absent/unreadable."""
    path = _league_state_path(checkpoint_dir, tag)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[league] WARNING: could not read progress file {path}: {exc}")
        return None


def _deck_trained_steps(deck: str, checkpoint_dir: str) -> int:
    """Cumulative trained steps of a deck's generalist, from its snapshot filenames.

    The newest ``{deck}__v{steps}.zip`` version number is the model's absolute
    ``num_timesteps`` at save; ``__final`` carries no count, so this slightly
    undercounts (by at most one snapshot interval). 0 when the deck has no
    versioned snapshots (fresh deck). Used only to seed the adaptive-rotation
    catch-up need for decks the sidecar has no stats for yet."""
    from opponents import deck_snapshots, _SNAPSHOT_RE
    best = 0
    for path in deck_snapshots(deck, checkpoint_dir):
        m = _SNAPSHOT_RE.match(os.path.basename(path))
        if m:
            best = max(best, int(m.group("steps")))
    return best


def _rotation_target(learner: str, roster: list[str], deck_stats: dict,
                     checkpoint_dir: str, rotate_every: int,
                     adaptive_boost: float) -> int:
    """Step budget for this learner's rotation, stretched for catch-up decks.

    Need is the greater of two 0..1 signals: how far the deck's last league
    win-rate sits below 50%, and how far its cumulative trained steps trail the
    roster's most-trained deck. The budget interpolates from ``rotate_every``
    (need 0) to ``adaptive_boost * rotate_every`` (need 1), so rotation order is
    untouched and no deck is ever starved — struggling decks just get a longer
    turn, and the boost decays automatically as their win-rate recovers."""
    if adaptive_boost <= 1.0:
        return rotate_every
    stats = deck_stats.get(learner) or {}
    wr = stats.get("winrate")
    wr_need = 0.0 if wr is None else max(0.0, min(1.0, (0.5 - float(wr)) / 0.5))
    steps = {d: int((deck_stats.get(d) or {}).get("steps")
                    or _deck_trained_steps(d, checkpoint_dir)) for d in roster}
    max_steps = max(steps.values(), default=0)
    steps_need = (1.0 - steps[learner] / max_steps) if max_steps > 0 else 0.0
    need = max(wr_need, max(0.0, min(1.0, steps_need)))
    target = int(rotate_every * (1.0 + (adaptive_boost - 1.0) * need))
    if target != rotate_every:
        print(f"[league] adaptive rotation for {learner}: winrate="
              f"{'n/a' if wr is None else f'{float(wr):.2f}'} (need {wr_need:.2f}), "
              f"steps {steps[learner]:,}/{max_steps:,} (need {steps_need:.2f}) "
              f"-> {target:,} steps this rotation")
    return target


def _resolve_model(path: str) -> str:
    """Resolve a model shorthand to a full checkpoint path.

    Thin alias for :func:`opponents.resolve_checkpoint` (the shared resolver —
    full path, deck-pilot shorthand like 'delver' or 'league/ur_delver', legacy
    matchup name, bare basename with '.zip' appended) pinned to this module's
    checkpoint dir.
    """
    from opponents import resolve_checkpoint
    return resolve_checkpoint(path, _CHECKPOINT_ABS)


def _ensure_deck_ckpt_subdir(checkpoint_dir: str, deck: str) -> None:
    """Create the mirrored checkpoints subfolder for a subfolder deck.

    Decks may be namespaced under a subfolder of decks/ (e.g. 'league/<stem>');
    their checkpoints are saved under the matching checkpoint subdir
    ('checkpoints/league/<stem>__*.zip'), which must exist before the first
    snapshot save."""
    deck_subdir = os.path.dirname(deck)
    if deck_subdir:
        os.makedirs(os.path.join(checkpoint_dir, deck_subdir), exist_ok=True)
# TOTAL_TIMESTEPS / N_ENVS / N_ENVS_SELF_PLAY imported from cli_spec (see top of file).
_DECKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "bin", "resources", "decks")
# League decks live in their own folder so the league roster is curated separately
# from the top-level training decks. A deck here is referenced as 'league/<stem>'
# (a path relative to decks/), which the engine resolves to decks/league/<stem>.dk
# and which namespaces its checkpoints under a matching 'league/' subdir.
_LEAGUE_DECKS_DIR = os.path.join(_DECKS_DIR, "league")


def _limit_worker_threads():
    """Pin a SubprocVecEnv worker process to a small intra-op math-thread count.

    Every parallel env runs its own torch opponent policy on CPU (self-play /
    league / random-model pools). Left at torch's default (≈ physical core
    count), N worker processes each spawn ≈cores intra-op threads, so N envs
    oversubscribe the machine by ≈N× — aggregate throughput then *falls* as envs
    are added instead of rising (measurable with train/bench_nenvs.py). Data
    parallelism here should come from the number of envs, not threads per env, so
    each worker is pinned to a single math thread by default.

    Called at the top of each env factory's ``_init`` so it applies only to the
    workers, never the main-process learner — whose PPO gradient update keeps
    full multithreaded backprop. Override the per-worker count with the
    ``ROBOMAGE_WORKER_THREADS`` env var (0 = leave torch's default / opt out),
    e.g. re-benchmark with ``ROBOMAGE_WORKER_THREADS=2``.
    """
    try:
        n = int(os.environ.get("ROBOMAGE_WORKER_THREADS", "1"))
    except ValueError:
        n = 1
    if n <= 0:
        return
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:  # noqa: BLE001 — torch may be absent (scripted-only worker)
        pass


def make_env(rank: int, model_deck: str = "delver", opp_deck: str = "delver",
             opponent_pool: str | None = None, opp_ckpt_ratio: float = 1.0,
             n_envs: int = 1, **env_kwargs):
    def _init():
        _limit_worker_threads()
        opponent = "scripted"
        if opponent_pool:
            from opponents import OpponentPool
            opponent = OpponentPool(
                opponent_pool, checkpoint_resolver=_resolve_model,
                rng=np.random.default_rng(1000 + rank),
                n_envs=n_envs, env_index=rank, max_checkpoint_ratio=opp_ckpt_ratio,
                model_deck=model_deck, opp_deck=opp_deck, checkpoint_dir=_CHECKPOINT_ABS)
        env = ModelVsScriptedEnv(model_deck=model_deck, opp_deck=opp_deck,
                                 opponent=opponent, **env_kwargs)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def make_fixed_model_env(opp_model_path: str, rank: int,
                         model_deck: str = "delver", opp_deck: str = "delver",
                         **env_kwargs):
    def _init():
        _limit_worker_threads()
        env = FixedModelEnv(opp_model_path=opp_model_path,
                            model_deck=model_deck, opp_deck=opp_deck, **env_kwargs)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def make_self_play_env(checkpoint_dir: str, rank: int,
                       model_deck: str = "delver", opp_deck: str = "delver",
                       **env_kwargs):
    def _init():
        _limit_worker_threads()
        env = SelfPlayEnv(checkpoint_dir=checkpoint_dir,
                          model_deck=model_deck, opp_deck=opp_deck, **env_kwargs)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def make_league_env(rank: int, learner_deck: str, roster: list[str], checkpoint_dir: str,
                    n_envs: int, opp_ckpt_ratio: float, self_play_frac: float,
                    scripted_anchor_frac: float, **env_kwargs):
    def _init():
        _limit_worker_threads()
        from opponents import LeaguePool
        pool = LeaguePool(
            learner_deck, roster, checkpoint_dir,
            self_play_frac=self_play_frac, scripted_anchor_frac=scripted_anchor_frac,
            rng=np.random.default_rng(2000 + rank),
            n_envs=n_envs, env_index=rank, max_checkpoint_ratio=opp_ckpt_ratio)
        # opp_deck is None: the LeaguePool picks the opponent's deck per episode.
        env = ModelVsScriptedEnv(model_deck=learner_deck, opp_deck=None,
                                 opponent=pool, **env_kwargs)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def train(binary_path: str, load_path: str | None = None, total_timesteps: int = TOTAL_TIMESTEPS,
          tally: bool = False, self_play: bool = False,
          model_deck: str = "delver", opp_deck: str = "delver",
          n_envs_override: int | None = None, no_shaping: bool = False,
          opponent_pool: str | None = None, opp_ckpt_ratio: float = 1.0,
          embed_dim: int = EMBED_DIM, fresh: bool = False, **env_kwargs):
    """Train the per-deck generalist model that pilots ``model_deck``.

    Models are **per-deck generalists**, not matchup-specific: one model plays
    ``model_deck`` against any opponent, saved as ``{model_deck}__final.zip`` with
    periodic ``{model_deck}__v{steps}.zip`` snapshots (the deck-pilot naming the
    league and self-play pools sample from). Training against a single opponent
    in a session just continues that one generalist, so unless ``--load`` or
    ``fresh`` is given the deck's existing ``__final`` (or newest snapshot) is
    auto-resumed and this session's steps accumulate onto it.

    Two opponent modes (mutually exclusive):
      * default (``self_play=False``) — every env trains against the rule-based
        scripted agent (``ModelVsScriptedEnv``), piloting ``opp_deck``.
      * ``self_play=True`` — every env trains against a frozen deck-pilot snapshot
        of ``opp_deck`` (``SelfPlayEnv`` samples ``{opp_deck}__v*.zip`` /
        ``{opp_deck}__final.zip``); if none exists yet, that env's opponent falls
        back to the scripted agent.

    Extra keyword arguments (``bo3``, ``auto_sideboard``, etc.) are forwarded
    to the underlying ``RoboMageEnv`` via the env factory helpers.
    """
    env_kwargs.setdefault("binary_path", binary_path)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    _ensure_deck_ckpt_subdir(checkpoint_dir, model_deck)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Parallel environments for faster data collection
    if self_play:
        n_envs = n_envs_override if n_envs_override is not None else N_ENVS_SELF_PLAY
        print(f"Opponent: self-play ({n_envs} envs vs frozen mirror checkpoints)")
        vec_env = SubprocVecEnv(
            [make_self_play_env(checkpoint_dir, i, model_deck, opp_deck, **env_kwargs) for i in range(n_envs)])
    else:
        n_envs = n_envs_override if n_envs_override is not None else N_ENVS
        if opponent_pool:
            print(f"Opponent: pool [{opponent_pool}] ({n_envs} envs, "
                  f"ckpt_ratio={opp_ckpt_ratio})")
        else:
            print(f"Opponent: scripted agent ({n_envs} envs)")
        vec_env = SubprocVecEnv([
            make_env(i, model_deck, opp_deck, opponent_pool=opponent_pool,
                     opp_ckpt_ratio=opp_ckpt_ratio, n_envs=n_envs, **env_kwargs)
            for i in range(n_envs)])

    try:
        policy_kwargs = dict(
            features_extractor_class=CardGameExtractor,
            features_extractor_kwargs=dict(embed_dim=embed_dim),
            net_arch=list(NET_ARCH),
        )

        # Per-deck generalist: auto-resume this deck's own latest checkpoint so a
        # single-opponent session accumulates onto the one model (unless --load
        # gave an explicit path or --fresh forced a scratch start).
        if not load_path and not fresh:
            auto = _resolve_model(model_deck)
            if auto != model_deck and os.path.exists(auto):
                load_path = auto
                print(f"Auto-resuming deck generalist: {auto} "
                      f"(use --fresh to start from scratch)")

        if load_path:
            print(f"Resuming from {load_path}")
            model = _reassert_hparams(MaskablePPO.load(load_path, env=vec_env),
                                      model_deck)
        else:
            policy_cls, policy_kwargs = _policy_config(policy_kwargs)
            model = MaskablePPO(
                policy_cls,
                vec_env,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=LOG_DIR,
                **PPO_KWARGS,
            )

        actual_n_envs = n_envs
        if no_shaping:
            vec_env.env_method("set_shaping_scale", 0.0)
            print("[shaping] disabled for this session (--no-shaping)")
        # Periodic deck-pilot snapshots ('{deck}__v{steps}.zip') feed the shared
        # self-play / league pools; the '{deck}__final.zip' is saved at the end.
        callbacks = [
            LRDecayCallback(),
            SnapshotCallback(checkpoint_dir, model_deck, LEAGUE_SNAPSHOT_EVERY),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))
        if tally:
            callbacks.append(WinTallyCallback())
        callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                           model_deck=model_deck, opp_deck=opp_deck,
                                           bo3=env_kwargs.get("bo3", False)))

        run_name = _configure_run_logger(model, model_deck)
        print(f"Training for {total_timesteps:,} timesteps across {actual_n_envs} envs... "
              f"(logs/{run_name})")
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    reset_num_timesteps=load_path is None)
        model.save(os.path.join(checkpoint_dir, f"{model_deck}__final"))
        print(f"Saved final model as {model_deck}__final.")
    finally:
        vec_env.close()


def _league_chunk(binary_path: str, learner_deck: str, roster: list[str],
                  checkpoint_dir: str, chunk_steps: int, *,
                  n_envs: int, opp_ckpt_ratio: float, self_play_frac: float,
                  scripted_anchor_frac: float, pfsp_mode: str, pfsp_p: float,
                  softmax_eta: float, snapshot_every: int, promote_margin: float,
                  embed_dim: int, no_shaping: bool, fresh: bool = False,
                  on_progress=None, **env_kwargs):
    """Train one learner deck for ``chunk_steps`` against the shared league pool.

    Resumes the learner's own latest checkpoint (``{deck}__final`` or newest
    ``{deck}__v*``) so its cumulative step count — and therefore snapshot version
    numbering — keeps growing across rotations; starts from scratch only the very
    first time a deck is trained, or whenever ``fresh`` is set.

    ``on_progress(steps_this_chunk)`` (optional) is called after every snapshot save
    with the number of new steps trained so far in this chunk, so the driver can
    persist resumable league progress alongside each checkpoint.
    """
    env_kwargs.setdefault("binary_path", binary_path)
    # League decks may be namespaced under a subfolder (e.g. 'league/<stem>'); mirror
    # that subdir under the checkpoint dir so snapshot saves don't fail.
    _ensure_deck_ckpt_subdir(checkpoint_dir, learner_deck)
    vec_env = SubprocVecEnv([
        make_league_env(i, learner_deck, roster, checkpoint_dir, n_envs,
                        opp_ckpt_ratio, self_play_frac, scripted_anchor_frac, **env_kwargs)
        for i in range(n_envs)])
    try:
        policy_kwargs = dict(
            features_extractor_class=CardGameExtractor,
            features_extractor_kwargs=dict(embed_dim=embed_dim),
            net_arch=list(NET_ARCH),
        )
        from opponents import latest_snapshot
        resume = None
        if not fresh:
            resume = os.path.join(checkpoint_dir, f"{learner_deck}__final.zip")
            if not os.path.exists(resume):
                resume = latest_snapshot(learner_deck, checkpoint_dir)
        resuming = bool(resume and os.path.exists(resume))
        if resuming:
            print(f"[league] resuming {learner_deck} from {os.path.basename(resume)}")
            model = _reassert_hparams(MaskablePPO.load(resume, env=vec_env),
                                      learner_deck)
        else:
            print(f"[league] starting {learner_deck} from scratch (embed_dim={embed_dim})")
            policy_cls, policy_kwargs = _policy_config(policy_kwargs)
            model = MaskablePPO(
                policy_cls, vec_env, policy_kwargs=policy_kwargs,
                verbose=1, tensorboard_log=LOG_DIR, **PPO_KWARGS)

        if no_shaping:
            vec_env.env_method("set_shaping_scale", 0.0)
            print("[shaping] disabled for this session (--no-shaping)")

        start_steps = model.num_timesteps
        # Translate a snapshot's absolute step count into "new steps this chunk" so the
        # driver can persist the global league position whenever a checkpoint is saved.
        snap_hook = None
        if on_progress is not None:
            snap_hook = lambda steps: on_progress(steps - start_steps)

        pfsp_cb = PFSPCallback(vec_env, mode=pfsp_mode, p=pfsp_p, eta=softmax_eta)
        callbacks = [
            LRDecayCallback(),
            pfsp_cb,
            SnapshotCallback(checkpoint_dir, learner_deck, snapshot_every,
                             promote_margin=promote_margin, pfsp_callback=pfsp_cb,
                             on_snapshot=snap_hook),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))

        _configure_run_logger(model, learner_deck)
        model.learn(total_timesteps=chunk_steps, callback=callbacks,
                    reset_num_timesteps=not resuming)
        model.save(os.path.join(checkpoint_dir, f"{learner_deck}__final"))
        print(f"[league] saved {learner_deck}__final")
        # PPO collects whole rollouts, so the chunk overshoots chunk_steps; return
        # the actual new steps so the driver's global budget stays accurate, plus
        # the learner's end-of-rotation win-rate and absolute step count for the
        # driver's adaptive-rotation stats (recent window when it has enough
        # games to judge current strength, else the chunk-lifetime average;
        # None when no decisive game finished).
        wr, n = pfsp_cb.recent_winrate()
        if n == 0:
            wr = None  # no decisive game finished this chunk
        elif n < 50:
            wr = pfsp_cb.overall_winrate()
        return model.num_timesteps - start_steps, wr, model.num_timesteps
    finally:
        vec_env.close()


def league(binary_path: str, decks: str | None = None,
           total_timesteps: int = TOTAL_TIMESTEPS,
           rotate_every: int = LEAGUE_ROTATE_EVERY,
           self_play_frac: float = LEAGUE_SELF_PLAY_FRAC,
           scripted_anchor_frac: float = LEAGUE_SCRIPTED_ANCHOR_FRAC,
           pfsp_mode: str = "pfsp", pfsp_p: float = LEAGUE_PFSP_P,
           softmax_eta: float = LEAGUE_SOFTMAX_ETA,
           snapshot_every: int = LEAGUE_SNAPSHOT_EVERY,
           promote_margin: float = LEAGUE_PROMOTE_MARGIN,
           embed_dim: int = EMBED_DIM, n_envs_override: int | None = None,
           opp_ckpt_ratio: float = 1.0, no_shaping: bool = False,
           adaptive_boost: float = LEAGUE_ADAPTIVE_BOOST,
           shard: str | None = None, train_decks_spec: str | None = None,
           tally: bool = False, resume: bool = False, **env_kwargs):
    """PFSP league driver: rotating single learner over a shared snapshot pool.

    One learner deck at a time trains for ``rotate_every`` steps against a frozen
    pool spanning a scripted anchor, every deck's snapshots, and the learner's own
    latest self; then the run rotates to the next deck. Snapshots dropped into the
    shared dir by earlier rotations become opponents for later ones, so the league
    structure is self-managing. Continues until ``total_timesteps`` total.

    Adaptive rotation length (``adaptive_boost`` > 1): a catch-up deck's rotation
    is stretched toward ``adaptive_boost * rotate_every``, scaled by how far its
    last league win-rate sits below 50% or its cumulative trained steps trail the
    roster leader (see ``_rotation_target``). Rotation *order* is unchanged, so
    every deck keeps training; strong decks just take shorter turns.

    Distributed sharding (``shard='i/n'``): this driver trains only the strided
    roster slice ``roster[i::n]`` while still sampling opponents from the FULL
    roster — run one shard per machine over a shared/synced checkpoint dir and
    each machine's pool ingests the other's snapshots automatically (see
    docs/distributed_league_training.md). ``total_timesteps`` is per-driver (the
    compute THIS process spends on its slice); there is deliberately no
    cross-machine step counter. Each shard keeps its own sidecar, so pass the
    same ``--shard`` together with ``--resume``.

    The driver's loop position (roster, total budget, global steps done, current
    rotation, and every hyperparameter) is persisted to a JSON sidecar
    (``checkpoints/_league_progress{.shard{i}of{n}}.json``) every time a snapshot
    checkpoint is saved and at each rotation boundary. ``resume=True`` reloads
    that sidecar and continues where an interrupted run left off — so a crashed
    league restarts with just ``train.py league --resume`` (all other flags are
    restored from the sidecar).
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    shard_split = parse_shard(shard)  # fail fast on a malformed spec
    tag = shard_tag(shard)

    steps_done = 0
    rotation = 0
    if resume:
        state = _read_league_state(checkpoint_dir, tag)
        if state is None:
            raise FileNotFoundError(
                f"--resume: no league progress file at "
                f"{_league_state_path(checkpoint_dir, tag)}. Start a league run "
                f"first (it writes the file as it trains), then resume it"
                + (" — a sharded run must be resumed with the SAME --shard."
                   if shard else "."))
        saved_shard = state.get("shard")
        if saved_shard != shard:
            raise ValueError(
                f"--resume: sidecar {_league_state_path(checkpoint_dir, tag)} was "
                f"written by shard {saved_shard!r} but --shard is {shard!r}.")
        # Restore the full run configuration from the sidecar so the resumed league
        # is identical to the interrupted one; only --binary stays caller-supplied.
        roster = list(state["roster"])
        total_timesteps = int(state["total_timesteps"])
        steps_done = int(state["steps_done"])
        rotation = int(state["rotation"])
        p = state.get("params", {})
        rotate_every = int(p.get("rotate_every", rotate_every))
        self_play_frac = float(p.get("self_play_frac", self_play_frac))
        scripted_anchor_frac = float(p.get("scripted_anchor_frac", scripted_anchor_frac))
        pfsp_mode = p.get("pfsp_mode", pfsp_mode)
        pfsp_p = float(p.get("pfsp_p", pfsp_p))
        softmax_eta = float(p.get("softmax_eta", softmax_eta))
        snapshot_every = int(p.get("snapshot_every", snapshot_every))
        promote_margin = float(p.get("promote_margin", promote_margin))
        embed_dim = int(p.get("embed_dim", embed_dim))
        n_envs_override = p.get("n_envs", n_envs_override)
        opp_ckpt_ratio = float(p.get("opp_ckpt_ratio", opp_ckpt_ratio))
        no_shaping = bool(p.get("no_shaping", no_shaping))
        adaptive_boost = float(p.get("adaptive_boost", adaptive_boost))
        train_decks_spec = p.get("train_decks", train_decks_spec)
        env_kwargs.setdefault("bo3", bool(p.get("bo3", env_kwargs.get("bo3", False))))
        env_kwargs.setdefault("auto_sideboard",
                              bool(p.get("auto_sideboard", env_kwargs.get("auto_sideboard", False))))
        print(f"[league] resuming from {_league_state_path(checkpoint_dir, tag)}: "
              f"{steps_done:,}/{total_timesteps:,} steps, rotation {rotation}")
        if steps_done >= total_timesteps:
            print("[league] saved progress is already complete — nothing to resume.")
            return
    elif decks:
        roster = [d.strip() for d in decks.split(",") if d.strip()]
    else:
        # Default roster: every deck in the dedicated league folder, referenced as
        # 'league/<stem>' so the engine loads decks/league/<stem>.dk.
        roster = sorted(
            "league/" + os.path.splitext(p)[0]
            for p in (os.listdir(_LEAGUE_DECKS_DIR) if os.path.isdir(_LEAGUE_DECKS_DIR) else [])
            if p.endswith(".dk"))
    if not roster:
        raise ValueError(
            f"No decks found for league (looked in {_LEAGUE_DECKS_DIR}). "
            f"Add deck files there, or pass --decks explicitly.")

    # Distributed sharding: this driver TRAINS only its slice of the roster, but
    # the opponent pool (and adaptive-rotation leader comparison) still spans the
    # full roster — peer shards' snapshots arrive via the shared/synced dir. The
    # slice is either an explicit --train-decks subset (arbitrary assignment from
    # the web distribution UI) or the strided --shard slice.
    if train_decks_spec:
        train_decks = [d.strip() for d in train_decks_spec.split(",") if d.strip()]
        unknown = [d for d in train_decks if d not in roster]
        if unknown:
            raise ValueError(
                f"--train-decks: {unknown} not in the --decks roster {roster}.")
        if not train_decks:
            raise ValueError("--train-decks is empty.")
    elif shard_split is not None:
        shard_id, num_shards = shard_split
        train_decks = roster[shard_id::num_shards]
        if not train_decks:
            raise ValueError(
                f"--shard {shard}: slice {shard_id}::{num_shards} of a "
                f"{len(roster)}-deck roster is empty.")
    else:
        train_decks = roster

    # League opponents load checkpoint models per env (like self-play), so default
    # to the lighter self-play env count rather than N_ENVS (sized for scripted opps).
    n_envs = n_envs_override if n_envs_override is not None else N_ENVS_SELF_PLAY
    print(f"League roster: {', '.join(roster)}")
    if train_decks != roster:
        label = f"shard {shard}" if shard else "this driver"
        print(f"  {label}: training only {', '.join(train_decks)}")
    print(f"  total={total_timesteps:,}  rotate_every={rotate_every:,}  "
          f"adaptive_boost={adaptive_boost}  n_envs={n_envs}")
    print(f"  self_play_frac={self_play_frac}  scripted_anchor_frac={scripted_anchor_frac}")
    print(f"  pfsp_mode={pfsp_mode}  p={pfsp_p}  eta={softmax_eta}")
    print(f"  snapshot_every={snapshot_every:,}  promote_margin={promote_margin}  embed_dim={embed_dim}")

    # Everything the sidecar needs to faithfully reconstruct this run on --resume.
    base_state = {
        "version": LEAGUE_STATE_VERSION,
        "roster": roster,
        "shard": shard,
        "total_timesteps": total_timesteps,
        "params": {
            "rotate_every": rotate_every,
            "self_play_frac": self_play_frac,
            "scripted_anchor_frac": scripted_anchor_frac,
            "pfsp_mode": pfsp_mode,
            "pfsp_p": pfsp_p,
            "softmax_eta": softmax_eta,
            "snapshot_every": snapshot_every,
            "promote_margin": promote_margin,
            "embed_dim": embed_dim,
            "n_envs": n_envs,
            "opp_ckpt_ratio": opp_ckpt_ratio,
            "no_shaping": no_shaping,
            "adaptive_boost": adaptive_boost,
            "train_decks": train_decks_spec,
            "bo3": bool(env_kwargs.get("bo3", False)),
            "auto_sideboard": bool(env_kwargs.get("auto_sideboard", False)),
        },
    }

    # Per-deck adaptive-rotation stats ({deck: {"winrate": float|None, "steps": int}}),
    # updated at each rotation end and persisted so --resume keeps the same targets.
    deck_stats: dict = (state.get("deck_stats") or {}) if resume else {}
    # The current rotation's step budget (adaptive; == rotate_every when boost is 1
    # or the deck has no catch-up need). Persisted so a mid-chunk resume finishes
    # the same-sized rotation it started.
    rotation_target = state.get("rotation_target") if resume else None

    def save_progress(done: int, rot: int, cstart: int):
        # `chunk_start` = global steps_done at the start of the current rotation, so a
        # mid-chunk resume can train only the *remainder* of an interrupted rotation
        # rather than a fresh full chunk.
        _write_league_state(checkpoint_dir, {**base_state, "steps_done": int(done),
                                             "rotation": int(rot),
                                             "chunk_start": int(cstart),
                                             "rotation_target": rotation_target,
                                             "deck_stats": deck_stats}, tag)

    # `chunk_start` tracks where the current rotation began. On a fresh run it equals
    # steps_done; on --resume it is restored from the sidecar (see below).
    chunk_start = steps_done if not resume else int(state.get("chunk_start", steps_done))

    # Record the starting position immediately so an interruption before the first
    # snapshot still leaves a resumable (if un-advanced) sidecar.
    save_progress(steps_done, rotation, chunk_start)

    while steps_done < total_timesteps:
        # Rotate over this shard's slice; the pool below still spans `roster`.
        learner = train_decks[rotation % len(train_decks)]
        done_in_rotation = steps_done - chunk_start
        if done_in_rotation == 0 or rotation_target is None:
            # Fresh rotation (or a pre-adaptive sidecar resumed mid-rotation):
            # size this rotation by the learner's catch-up need.
            rotation_target = _rotation_target(
                learner, roster, deck_stats, checkpoint_dir, rotate_every,
                adaptive_boost)
        chunk = min(rotation_target - done_in_rotation, total_timesteps - steps_done)
        if chunk <= 0:
            # Rotation already satisfied (only reachable via an odd resume) — advance.
            rotation += 1
            chunk_start = steps_done
            rotation_target = None
            save_progress(steps_done, rotation, chunk_start)
            continue
        print(f"\n{'='*60}")
        print(f"[league rotation {rotation + 1}] learner={learner}  "
              f"({chunk:,} steps, {steps_done:,}/{total_timesteps:,} done"
              + (f", {done_in_rotation:,} already this rotation" if done_in_rotation else "")
              + ")")
        print(f"{'='*60}")
        # Persist fine-grained progress on every snapshot mid-chunk: the projected
        # global step count, keeping `rotation`/`chunk_start` fixed so a resume
        # re-enters this same learner for the remainder of its rotation.
        rotation_start_done = chunk_start
        ran, learner_wr, learner_steps = _league_chunk(
            binary_path, learner, roster, checkpoint_dir, chunk,
            n_envs=n_envs, opp_ckpt_ratio=opp_ckpt_ratio,
            self_play_frac=self_play_frac, scripted_anchor_frac=scripted_anchor_frac,
            pfsp_mode=pfsp_mode, pfsp_p=pfsp_p, softmax_eta=softmax_eta,
            snapshot_every=snapshot_every, promote_margin=promote_margin,
            embed_dim=embed_dim, no_shaping=no_shaping,
            on_progress=lambda chunk_steps: save_progress(
                steps_done + chunk_steps, rotation, rotation_start_done),
            **env_kwargs)
        deck_stats[learner] = {"winrate": learner_wr, "steps": int(learner_steps)}
        steps_done += ran if ran else chunk
        rotation += 1
        chunk_start = steps_done
        rotation_target = None
        save_progress(steps_done, rotation, chunk_start)

    print(f"\nLeague complete: {total_timesteps:,} total timesteps over {rotation} rotations.")


def train_fixed_model(binary_path: str, model_deck: str, opp_deck: str,
                      load_path: str | None = None,
                      total_timesteps: int = TOTAL_TIMESTEPS,
                      tally: bool = False,
                      n_envs_override: int | None = None,
                      no_shaping: bool = False, **env_kwargs):
    """Train ``model_deck``'s generalist against ``opp_deck``'s fixed generalist.

    Both sides are per-deck generalists: resumes ``{model_deck}__final.zip`` (or
    ``load_path``) as the training model and freezes ``{opp_deck}__final.zip``
    (or its newest snapshot) as the opponent for every game.

    Extra keyword arguments (``bo3``, ``auto_sideboard``, etc.) are forwarded
    to the underlying ``RoboMageEnv`` via the env factory helpers.
    """
    env_kwargs.setdefault("binary_path", binary_path)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    _ensure_deck_ckpt_subdir(checkpoint_dir, model_deck)
    os.makedirs(LOG_DIR, exist_ok=True)

    if not load_path:
        candidate = _resolve_model(model_deck)
        if candidate != model_deck and os.path.exists(candidate):
            load_path = candidate
    if not load_path:
        raise FileNotFoundError(f"No training model found piloting {model_deck} "
                                f"({model_deck}__final.zip or {model_deck}__v*.zip)")

    opp_model_path = _resolve_model(opp_deck)
    if opp_model_path == opp_deck or not os.path.exists(opp_model_path):
        raise FileNotFoundError(f"No opponent model found piloting {opp_deck} "
                                f"({opp_deck}__final.zip or {opp_deck}__v*.zip)")

    print(f"Training {model_deck} generalist against fixed {opp_deck} generalist")
    print(f"  training model: {load_path}")
    print(f"  opponent model: {opp_model_path}")

    n_envs = n_envs_override if n_envs_override is not None else N_ENVS_SELF_PLAY
    vec_env = SubprocVecEnv([
        make_fixed_model_env(opp_model_path, i, model_deck, opp_deck, **env_kwargs)
        for i in range(n_envs)
    ])

    try:
        print(f"Resuming from {load_path}")
        model = _reassert_hparams(MaskablePPO.load(load_path, env=vec_env),
                                  model_deck)

        if no_shaping:
            vec_env.env_method("set_shaping_scale", 0.0)
            print("[shaping] disabled for this session (--no-shaping)")
        callbacks = [
            LRDecayCallback(),
            SnapshotCallback(checkpoint_dir, model_deck, LEAGUE_SNAPSHOT_EVERY),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))
        if tally:
            callbacks.append(WinTallyCallback())
        callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                           model_deck=model_deck, opp_deck=opp_deck,
                                           bo3=env_kwargs.get("bo3", False)))

        run_name = _configure_run_logger(model, model_deck)
        print(f"Training for {total_timesteps:,} timesteps across {n_envs} envs... "
              f"(logs/{run_name})")
        model.learn(total_timesteps=total_timesteps, callback=callbacks,
                    reset_num_timesteps=False)
        model.save(os.path.join(checkpoint_dir, f"{model_deck}__final"))
        print(f"Saved final model as {model_deck}__final.")
    finally:
        vec_env.close()


def train_alternate(binary_path: str, deck_a: str, deck_b: str,
                    alternate_steps: int, total_timesteps: int = TOTAL_TIMESTEPS,
                    tally: bool = False,
                    n_envs_override: int | None = None,
                    no_shaping: bool = False, **env_kwargs):
    """Alternate training between two decks' generalists every ``alternate_steps``.

    Each round trains one deck's generalist against the other's frozen generalist,
    then saves and swaps roles.

    Extra keyword arguments (``bo3``, ``auto_sideboard``, etc.) are forwarded
    to the underlying ``RoboMageEnv`` via ``train_fixed_model``.
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Verify both decks have a generalist checkpoint to alternate between.
    for d in (deck_a, deck_b):
        resolved = _resolve_model(d)
        if resolved == d or not os.path.exists(resolved):
            raise FileNotFoundError(f"Missing model piloting {d} "
                                    f"({d}__final.zip or {d}__v*.zip)")

    steps_done = 0
    round_num = 0
    # Start by training deck_a
    training_deck, opp_deck = deck_a, deck_b

    while steps_done < total_timesteps:
        round_num += 1
        remaining = total_timesteps - steps_done
        round_steps = min(alternate_steps, remaining)

        print(f"\n{'='*60}")
        print(f"[alternate round {round_num}] Training {training_deck} vs fixed {opp_deck}"
              f"  ({round_steps:,} steps, {steps_done:,}/{total_timesteps:,} done)")
        print(f"{'='*60}")

        train_fixed_model(binary_path, training_deck, opp_deck,
                          total_timesteps=round_steps, tally=tally,
                          n_envs_override=n_envs_override,
                          no_shaping=no_shaping, **env_kwargs)

        steps_done += round_steps

        # Swap roles
        training_deck, opp_deck = opp_deck, training_deck

    print(f"\nAlternate training complete: {total_timesteps:,} total timesteps over {round_num} rounds.")


def _deck_from_checkpoint(model_path: str) -> str | None:
    """Infer the deck a checkpoint pilots from its deck-pilot filename.

    Works on the v2 naming ('{deck}__final.zip' / '{deck}__v{steps}.zip'); a
    checkpoint under a checkpoints/ subfolder keeps that subfolder as the deck
    namespace ('checkpoints/league/bug__final.zip' → 'league/bug'). None when
    the filename doesn't parse as deck-pilot naming.
    """
    rel = os.path.relpath(os.path.abspath(model_path), _CHECKPOINT_ABS)
    if rel.startswith(".."):  # outside checkpoints/ — use the bare filename
        rel = os.path.basename(model_path)
    m = re.match(r"^(?P<deck>.+?)__(final|v\d+)\.zip$", rel)
    return m.group("deck") if m else None


def baseline(binary_path: str, model_path: str, n_games: int = 100,
             deck: str | None = None, opp_deck: str | None = None,
             seed: int | None = None, quiet: bool = False):
    """Evaluate a model's win rate vs the scripted HARD agent.

    The model pilots ``deck`` (inferred from the checkpoint's deck-pilot
    filename when not given) and faces scripted:hard piloting ``opp_deck``
    (defaults to ``deck`` — a mirror match). Seats alternate each game (model
    is Player A in even games) so neither side gets a systematic on-the-play
    edge. ``seed`` makes the run reproducible (game ``i`` uses ``seed + i``;
    None = random per game). Returns ``(wins, losses, draws)`` from the model's
    perspective.
    """
    import runner
    from opponents import ModelController, ScriptedController
    from scripted_agent import make_agent

    model = MaskablePPO.load(model_path)
    if deck is None:
        deck = _deck_from_checkpoint(model_path)
    if opp_deck is None:
        opp_deck = deck
    ctrl_model = ModelController(model, label="Model", deterministic=True)
    ctrl_scripted = ScriptedController(make_agent("scripted:hard"), label="Scripted")
    wins = losses = draws = 0

    for i in range(n_games):
        model_is_a = (i % 2 == 0)
        ctrl_a, ctrl_b = ((ctrl_model, ctrl_scripted) if model_is_a
                          else (ctrl_scripted, ctrl_model))
        # The model always pilots `deck`; the scripted opponent always `opp_deck`.
        deck_a, deck_b = ((deck, opp_deck) if model_is_a else (opp_deck, deck))
        w, l, d = runner.run_games(
            ctrl_a, ctrl_b,
            label_a="Model" if model_is_a else "Scripted",
            label_b="Scripted" if model_is_a else "Model",
            binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
            n_games=1, seed=(seed + i) if seed is not None else None,
            transcript="quiet")
        wins += w if model_is_a else l
        losses += l if model_is_a else w
        draws += d
        if not quiet:
            print(f"\rGame {i+1}/{n_games}  W:{wins} L:{losses} D:{draws}",
                  end="", flush=True)

    if not quiet:
        print()
        vs = (f"scripted:hard ({opp_deck})" if opp_deck != deck else "scripted:hard")
        print(f"{os.path.basename(model_path)} ({deck or 'default deck'}) vs "
              f"{vs} over {n_games} games: {wins}W / {losses}L / {draws}D "
              f"({100 * wins / n_games:.1f}% win rate)")
    return wins, losses, draws


def _league_roster() -> list[str]:
    """The league deck roster ('league/<stem>' for every decks/league/*.dk).

    Mirrors :func:`league`'s default roster so the baseline sweep evaluates the
    same set of decks the league trains.
    """
    return sorted(
        "league/" + os.path.splitext(p)[0]
        for p in (os.listdir(_LEAGUE_DECKS_DIR) if os.path.isdir(_LEAGUE_DECKS_DIR) else [])
        if p.endswith(".dk"))


def baseline_all(binary_path: str, n_games: int = 50, seed: int | None = None,
                 log_path: str | None = None):
    """Round-robin every league deck's model vs scripted:hard on every league deck.

    For each league deck (decks/league/*.dk) that has a ``{deck}__final.zip``
    checkpoint, the model plays ``n_games`` (default 50) against scripted:hard
    piloting *each* league deck — including its own (the mirror). So an
    N-deck roster runs N×N matchups of ``n_games`` each (e.g. 7 decks → 49
    matchups → 2450 games at the default 50). A per-matchup win-rate grid plus a
    timestamped summary are appended to ``log_path`` (default
    checkpoints/baseline_report.log) as well as printed to stdout.
    """
    roster = _league_roster()
    if not roster:
        print(f"No league decks found under {_LEAGUE_DECKS_DIR}")
        return

    # Resolve each roster deck to its final checkpoint; skip decks without one.
    learners = []  # (deck, checkpoint_path)
    for deck in roster:
        try:
            ckpt = _resolve_model(deck)
        except Exception:
            ckpt = None
        if ckpt and os.path.exists(ckpt):
            learners.append((deck, ckpt))
        else:
            print(f"[baseline] skipping {deck}: no __final checkpoint found", flush=True)
    if not learners:
        print(f"No league __final checkpoints found under {_CHECKPOINT_ABS}")
        return

    if log_path is None:
        log_path = os.path.join(_CHECKPOINT_ABS, "baseline_report.log")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (f"=== league baseline round-robin vs scripted:hard — {stamp} — "
              f"{n_games} games/matchup, seed={seed} ===")
    lines = [header]
    print(header, flush=True)

    for deck, ckpt in learners:
        row_cells = []
        print(f"\n--- {deck} model vs scripted:hard on each league deck ---", flush=True)
        for opp in roster:
            print(f"\n  {deck} (model) vs {opp} (scripted:hard):", flush=True)
            w, l, d = baseline(binary_path, ckpt, n_games=n_games, deck=deck,
                               opp_deck=opp, seed=seed)
            total = w + l + d
            pct = 100 * w / total if total else 0
            row_cells.append(f"vs {opp}={w}W/{l}L/{d}D({pct:.0f}%)")
            lines.append(f"{deck:<22} vs {opp:<22} "
                         f"{w}W/{l}L/{d}D  {pct:.1f}% win rate")
        lines.append(f"  [{deck}] " + "  ".join(row_cells))

    summary = "\n".join(lines)
    with open(log_path, "a") as f:
        f.write(summary + "\n\n")
    print(f"\n{summary}\n\nreport appended to {log_path}", flush=True)


def observe(binary_path: str,
            player_a: str = "scripted", player_b: str = "scripted",
            deck_a: str | None = None, deck_b: str | None = None,
            n_games: int = 1, bo3: bool = False,
            seed: int | None = None, verbose: bool = False,
            play_a: str | None = None, play_b: str | None = None):
    """Observe one or more games between any pair of {scripted | model} controllers.

    ``player_a``/``player_b`` are either the literal "scripted" (or a
    "scripted:*" variant) or a model checkpoint (.zip path or shorthand).
    ``play_a``/``play_b`` override the corresponding side with a semantic action
    script (see ``action_spec``) — handy for driving one seat through a fixed
    line while watching the other.  ``deck_a``/``deck_b`` set each side's deck.
    Every decision by each agent is logged; ``--verbose`` additionally dumps the
    full board state and the legal action menu at each decision (the same
    transcript format the test harness prints).  With ``n_games > 1`` a per-game
    result line and a final W/L/D summary are printed.

    This is a thin wrapper: the actual game-driving loop lives in
    ``runner.run_games`` (shared with the test harness).
    """
    from opponents import make_controller, is_scripted_spec, PlayController
    import runner

    # Observation is a fixed replay, so use deterministic model predictions.
    # A --play-{a,b} script takes precedence over --player-{a,b} for that seat.
    if play_a is not None:
        ctrl_a, label_a = PlayController(play_a), "Play"
    else:
        ctrl_a = make_controller(player_a or "scripted",
                                 checkpoint_resolver=_resolve_model, deterministic=True)
        label_a = "Scripted" if is_scripted_spec(player_a or "scripted") else "Model"
    if play_b is not None:
        ctrl_b, label_b = PlayController(play_b), "Play"
    else:
        ctrl_b = make_controller(player_b or "scripted",
                                 checkpoint_resolver=_resolve_model, deterministic=True)
        label_b = "Scripted" if is_scripted_spec(player_b or "scripted") else "Model"

    unit = "match" if bo3 else "game"
    print(f"=== {label_a}/A ({deck_a or 'default'} deck) vs "
          f"{label_b}/B ({deck_b or 'default'} deck) — "
          f"{n_games} {unit}{'es' if bo3 else 's'} ===\n", flush=True)

    runner.run_games(ctrl_a, ctrl_b, label_a=label_a, label_b=label_b,
                     binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                     n_games=n_games, bo3=bo3, seed=seed, verbose=verbose)


# _CAT_NAMES / _STEP_NAMES are imported from _enums at the top of this module.
# (Per-decision transcript decoding lives in decode.py/runner.py — the shared
# formatters — since the observe/baseline/rollout paths all run through
# runner.run_games now.)


def _is_debug_build(binary_path: str) -> bool | None:
    """True if the engine binary was built debug, False if release, None if unknown.

    The Makefile's release build (BUILD=RELEASE) compiles -O2 -flto -DNDEBUG with
    no debug info; the default debug build adds -ggdb, which embeds DWARF
    `.debug_info` sections. So the presence of that section name in the binary is
    a reliable debug-vs-release signal (no engine change / extra dependency)."""
    try:
        with open(binary_path, "rb") as f:
            return b".debug_info" in f.read()
    except OSError:
        return None


def _warn_if_debug_build(binary_path: str) -> None:
    """Warn (once) that RL training should use a release engine build for speed."""
    if _is_debug_build(binary_path):
        print(
            "\n\033[33mWARNING: training against a DEBUG engine build "
            f"({binary_path}).\033[0m\n"
            "  The debug build runs the engine with assertions and no optimization, "
            "which makes\n  self-play data collection several times slower. Build the "
            "release engine first:\n      make BUILD=RELEASE\n  and re-run (pass "
            "--binary to point at it if it is not the default bin/robomage).\n",
            flush=True)
        
def _run_sweep(args, parser):
    """Train one deck's generalist against a PFSP pool of the other decks.

    Like ``league``, but with a single fixed learner: ``args.deck`` is the only
    model that trains and it never rotates away — the roster (default: every
    other deck in ``bin/resources/decks/``) supplies opponents only, sampled the
    same way ``league`` samples them (scripted anchor floor, PFSP/softmax-weighted
    historical snapshots, and a latest-self slot). This is a single, un-rotated
    call into the same ``_league_chunk`` the league driver uses per rotation.
    """
    all_decks = sorted(os.path.splitext(p)[0]
                       for p in os.listdir(_DECKS_DIR) if p.endswith(".dk"))
    all_decks += sorted(
        "league/" + os.path.splitext(p)[0]
        for p in (os.listdir(_LEAGUE_DECKS_DIR) if os.path.isdir(_LEAGUE_DECKS_DIR) else [])
        if p.endswith(".dk"))
    if args.deck not in all_decks:
        parser.error(f"Deck '{args.deck}' not found in {_DECKS_DIR}. "
                     f"Available: {', '.join(all_decks)}")
    if args.opponents:
        roster = [d.strip() for d in args.opponents.split(",") if d.strip()]
        unknown = [d for d in roster if d not in all_decks]
        if unknown:
            parser.error(f"--opponents: {unknown} not found in {_DECKS_DIR}. "
                         f"Available: {', '.join(all_decks)}")
    else:
        roster = [d for d in all_decks if d != args.deck]
    if not roster:
        parser.error("No opponent decks available for the pool (need at least "
                     "one other deck in bin/resources/decks/, or pass --opponents).")

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    n_envs = args.n_envs if args.n_envs is not None else N_ENVS_SELF_PLAY
    env_kwargs = dict(bo3=args.bo3, auto_sideboard=args.auto_sideboard)

    print(f"Sweep: training '{args.deck}' vs pool [{', '.join(roster)}]")
    print(f"  total={args.total_timesteps:,}  n_envs={n_envs}")
    print(f"  self_play_frac={args.self_play_frac}  scripted_anchor_frac={args.scripted_anchor_frac}")
    print(f"  pfsp_mode={args.pfsp_mode}  p={args.pfsp_p}  eta={args.softmax_eta}")
    print(f"  snapshot_every={args.snapshot_every:,}  promote_margin={args.promote_margin}  "
          f"embed_dim={args.embed_dim}")

    _league_chunk(args.binary, args.deck, roster, checkpoint_dir, args.total_timesteps,
                 n_envs=n_envs, opp_ckpt_ratio=args.opponent_ckpt_ratio,
                 self_play_frac=args.self_play_frac,
                 scripted_anchor_frac=args.scripted_anchor_frac,
                 pfsp_mode=args.pfsp_mode, pfsp_p=args.pfsp_p, softmax_eta=args.softmax_eta,
                 snapshot_every=args.snapshot_every, promote_margin=args.promote_margin,
                 embed_dim=args.embed_dim, no_shaping=args.no_shaping,
                 fresh=args.fresh, **env_kwargs)
    print(f"\nSweep complete: {args.total_timesteps:,} timesteps for '{args.deck}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RoboMage RL training and evaluation.",
        epilog="Run a subcommand with -h for its options (e.g. 'train.py train -h'). "
               "If no subcommand is given, 'train' is assumed, so legacy one-liners "
               "like 'train.py --opponent mav' still work.")
    sub = parser.add_subparsers(dest="command")

    # All subcommands and their flags come from cli_spec.TRAIN_TOOL (single source
    # shared with the TUI). Dispatch below stays hand-written.
    for s in TRAIN_TOOL.subs:
        sp = sub.add_parser(s.name, help=s.help)
        apply_to_parser(sp, s)

    # Default to the 'train' subcommand when none is given, so legacy one-liners
    # such as 'train.py --opponent mav' continue to work.
    COMMANDS = {s.name for s in TRAIN_TOOL.subs}
    argv = sys.argv[1:]
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["train"] + argv
    args = parser.parse_args(argv)

    if args.command in ("train", "sweep", "fixed-model", "alternate", "league"):
        env_kwargs = dict(bo3=args.bo3, auto_sideboard=args.auto_sideboard)
        # These are the training subcommands — nudge toward a release engine build.
        _warn_if_debug_build(args.binary)

    if args.command == "league":
        league(args.binary, decks=args.decks, total_timesteps=args.total_timesteps,
               rotate_every=args.rotate_every, self_play_frac=args.self_play_frac,
               scripted_anchor_frac=args.scripted_anchor_frac, pfsp_mode=args.pfsp_mode,
               pfsp_p=args.pfsp_p, softmax_eta=args.softmax_eta,
               snapshot_every=args.snapshot_every, promote_margin=args.promote_margin,
               embed_dim=args.embed_dim, n_envs_override=args.n_envs,
               opp_ckpt_ratio=args.opponent_ckpt_ratio, no_shaping=args.no_shaping,
               adaptive_boost=args.adaptive_boost, shard=args.shard,
               train_decks_spec=args.train_decks,
               tally=args.tally, resume=args.resume, **env_kwargs)
    elif args.command == "train":
        train(args.binary, _resolve_model(args.load), args.total_timesteps,
              tally=args.tally, self_play=args.self_play,
              model_deck=args.deck, opp_deck=args.opponent,
              n_envs_override=args.n_envs, no_shaping=args.no_shaping,
              opponent_pool=args.opponent_pool, opp_ckpt_ratio=args.opponent_ckpt_ratio,
              embed_dim=args.embed_dim, fresh=args.fresh, **env_kwargs)
    elif args.command == "sweep":
        _run_sweep(args, parser)
    elif args.command == "fixed-model":
        train_fixed_model(args.binary, args.deck, args.opponent,
                          load_path=_resolve_model(args.load),
                          total_timesteps=args.total_timesteps,
                          tally=args.tally,
                          n_envs_override=args.n_envs,
                          no_shaping=args.no_shaping, **env_kwargs)
    elif args.command == "alternate":
        train_alternate(args.binary, args.deck, args.opponent,
                        alternate_steps=args.every,
                        total_timesteps=args.total_timesteps,
                        tally=args.tally,
                        n_envs_override=args.n_envs,
                        no_shaping=args.no_shaping, **env_kwargs)
    elif args.command == "observe":
        # observe defaults to bo3 matches; --bo1 opts back into single games
        # (--bo3 is accepted as a redundant no-op for backward compatibility).
        observe(args.binary, player_a=args.player_a, player_b=args.player_b,
                deck_a=args.deck, deck_b=args.opponent,
                n_games=args.games, bo3=not args.bo1, seed=args.seed,
                verbose=args.verbose,
                play_a=args.play_a, play_b=args.play_b)
    elif args.command == "baseline":
        if args.all:
            baseline_all(args.binary, n_games=args.games or 50, seed=args.seed,
                         log_path=args.log)
        elif args.model is None:
            parser.error("baseline: give a model checkpoint, or --all to sweep "
                         "every league deck")
        else:
            baseline(args.binary, _resolve_model(args.model), args.games or 100,
                     deck=args.deck, seed=args.seed)
