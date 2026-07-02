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
import json
import os
import sys
from collections import deque

from env import (RoboMageEnv, ModelVsScriptedEnv, SelfPlayEnv, FixedModelEnv, NarrativeEnv,
                 scripted_action,
                 OBS_SIZE, STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, BINARY)
from extractor import CardGameExtractor
from card_costs import N_CARD_TYPES
import decode
# Action-category / step display names are generated from the C++ enums
# (train/gen_enums.py) and shared via decode.py — re-exported here so existing
# `from train import _CAT_NAMES, _STEP_NAMES` consumers (analysis.py) keep working.
from _enums import _CAT_NAMES, _STEP_NAMES
# CLI definitions + training defaults live in cli_spec.py (single source shared with the TUI).
from cli_spec import (TOTAL_TIMESTEPS, N_ENVS, N_ENVS_SELF_PLAY, EMBED_DIM,
                      LEAGUE_SELF_PLAY_FRAC, LEAGUE_SCRIPTED_ANCHOR_FRAC,
                      LEAGUE_PFSP_P, LEAGUE_SOFTMAX_ETA, LEAGUE_SNAPSHOT_EVERY,
                      LEAGUE_PROMOTE_MARGIN, LEAGUE_ROTATE_EVERY,
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

import numpy as np


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
        self._rollout += 1
        log_path = os.path.join(self.replay_dir, f"rollout_{self._rollout:05d}.txt")

        env = NarrativeEnv(binary_path=self.binary_path,
                           deck_a=self._model_deck, deck_b=self._opp_deck,
                           bo3=self._bo3)
        if USE_MASKABLE:
            from sb3_contrib.common.wrappers import ActionMasker as _AM
            masked = _AM(env, lambda e: e.action_masks())
        else:
            masked = env

        try:
            model_is_a = bool(np.random.random() < 0.5)
            if self._model_deck is not None:
                env._deck_a = self._model_deck if model_is_a else self._opp_deck
                env._deck_b = self._opp_deck if model_is_a else self._model_deck
            obs, _ = masked.reset()
            done = False
            total_reward = 0.0
            turn = 0   # last turn header shown (1-based, from the state vector)
            known_hand = {"A": [], "B": []}

            with open(log_path, "w") as f:
                model_side = "A" if model_is_a else "B"
                scripted_side = "B" if model_is_a else "A"
                f.write(f"=== Rollout {self._rollout}: Model ({model_side}) vs Scripted ({scripted_side}) ===\n\n")

                while not done:
                    for line in env.flush_lines():
                        if line.strip():
                            f.write(line + "\n")

                    a_has_priority = obs[32] > 0.5
                    model_has_priority = a_has_priority if model_is_a else not a_has_priority
                    num_choices = env._num_choices
                    cur_side = "A" if a_has_priority else "B"

                    priority_is_a = a_has_priority
                    active_is_a = (obs[31] > 0.5) == priority_is_a
                    cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
                    card_ids = obs[STATE_SIZE + MAX_ACTIONS:STATE_SIZE + 2 * MAX_ACTIONS]
                    is_mulligan = any(c == 11 for c in cats)

                    known_hand[cur_side] = _decode_hand(obs)

                    # Sequential 1-based turn from the state vector (matches the
                    # engine's TURN headers); a flip counter drifts when a turn
                    # yields no decision query.
                    cur_turn = decode.decode_turn(obs)
                    if not is_mulligan and cur_turn != turn:
                        turn = cur_turn
                        active_label = "A" if active_is_a else "B"
                        f.write(f"--- Turn {turn} (Player {active_label}) ---\n")
                        f.write(f"  PA: {', '.join(known_hand['A']) or '(empty)'}\n")
                        f.write(f"  PB: {', '.join(known_hand['B']) or '(empty)'}\n")

                    if model_has_priority:
                        masks = env.action_masks() if USE_MASKABLE else None
                        action, _ = self.model.predict(obs, action_masks=masks, deterministic=True)
                        action = int(action)
                        desc = _describe_action(cats, card_ids, action, num_choices)
                        f.write(f"[Model/{cur_side}] {desc}  ({action + 1} of {num_choices})\n")
                    else:
                        action = scripted_action(obs, num_choices)
                        desc = _describe_action(cats, card_ids, action, num_choices)
                        f.write(f"[Scripted/{cur_side}] {desc}  ({action + 1} of {num_choices})\n")

                    obs, reward, terminated, truncated, _ = masked.step(action)
                    total_reward += reward
                    done = terminated or truncated

                for line in env.flush_lines():
                    if line.strip():
                        f.write(line + "\n")

                model_reward = total_reward if model_is_a else -total_reward
                if self._bo3:
                    result = "Model wins match" if model_reward > 0 else "Scripted wins match" if model_reward < 0 else "Draw"
                else:
                    result = "Model wins" if model_reward > 0 else "Scripted wins" if model_reward < 0 else "Draw"
                f.write(f"\n=== {result} ===\n")

            print(f"[replay] rollout {self._rollout}: {result} -> {log_path}")
        except Exception as exc:
            print(f"[replay] rollout {self._rollout}: game failed ({exc})")
        finally:
            env.close()


CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
_CHECKPOINT_ABS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

# League resume: the driver's loop position (which deck is up, how many global steps
# are done) lives outside any single model checkpoint, so we persist it to a small
# JSON sidecar that is rewritten every time a snapshot is saved (and at each rotation
# boundary). A crashed/interrupted `league` run then resumes from `league --resume`.
LEAGUE_STATE_VERSION = 1


def _league_state_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "_league_progress.json")


def _write_league_state(checkpoint_dir: str, state: dict) -> None:
    """Atomically persist the league driver's progress to its JSON sidecar."""
    path = _league_state_path(checkpoint_dir)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=2)
        os.replace(tmp, path)
    except OSError as exc:
        print(f"[league] WARNING: could not write progress file {path}: {exc}")


def _read_league_state(checkpoint_dir: str) -> dict | None:
    """Load the league progress sidecar, or None if it is absent/unreadable."""
    path = _league_state_path(checkpoint_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"[league] WARNING: could not read progress file {path}: {exc}")
        return None


def _resolve_model(path: str) -> str:
    """Resolve a model shorthand to a full checkpoint path.

    Accepts (in priority order):
      - Full path (returned as-is if it exists)
      - Deck-pilot shorthand like 'delver' → checkpoints/delver__final.zip, else the
        newest checkpoints/delver__v*.zip snapshot (v2 league naming)
      - Legacy matchup name like 'delver_mav' → checkpoints/delver_mav_final.zip
      - Bare basename with '.zip' appended
    """
    if path is None:
        return None
    # Already a real path
    if os.path.exists(path):
        return path
    # v2 deck-pilot shorthand → '{deck}__final.zip' or newest '{deck}__v*.zip'.
    deck_final = os.path.join(_CHECKPOINT_ABS, f"{path}__final.zip")
    if os.path.exists(deck_final):
        return deck_final
    from opponents import latest_snapshot
    snap = latest_snapshot(path, _CHECKPOINT_ABS)
    if snap:
        return snap
    # Legacy matchup shorthand → checkpoints/{name}_final.zip
    candidate = os.path.join(_CHECKPOINT_ABS, f"{path}_final.zip")
    if os.path.exists(candidate):
        return candidate
    # Try with .zip appended (e.g. 'delver_mav_100000_steps')
    candidate2 = os.path.join(_CHECKPOINT_ABS, f"{path}.zip")
    if os.path.exists(candidate2):
        return candidate2
    # Return original — let downstream code report the error
    return path
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
            net_arch=[256, 256],
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
            model = MaskablePPO.load(load_path, env=vec_env)
        else:
            model = MaskablePPO(
                "MlpPolicy",
                vec_env,
                policy_kwargs=policy_kwargs,
                learning_rate=3e-4,
                n_steps=4096,           # steps per env per update
                batch_size=1024,
                n_epochs=8,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.25,
                ent_coef=0.012,
                verbose=1,
                tensorboard_log=LOG_DIR,
            )

        actual_n_envs = n_envs
        if no_shaping:
            vec_env.env_method("set_shaping_scale", 0.0)
            print("[shaping] disabled for this session (--no-shaping)")
        # Periodic deck-pilot snapshots ('{deck}__v{steps}.zip') feed the shared
        # self-play / league pools; the '{deck}__final.zip' is saved at the end.
        callbacks = [
            SnapshotCallback(checkpoint_dir, model_deck, LEAGUE_SNAPSHOT_EVERY),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))
        if tally:
            callbacks.append(WinTallyCallback())
        callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                           model_deck=model_deck, opp_deck=opp_deck,
                                           bo3=env_kwargs.get("bo3", False)))

        print(f"Training for {total_timesteps:,} timesteps across {actual_n_envs} envs...")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=load_path is None)
        model.save(os.path.join(checkpoint_dir, f"{model_deck}__final"))
        print(f"Saved final model as {model_deck}__final.")
    finally:
        vec_env.close()


def _league_chunk(binary_path: str, learner_deck: str, roster: list[str],
                  checkpoint_dir: str, chunk_steps: int, *,
                  n_envs: int, opp_ckpt_ratio: float, self_play_frac: float,
                  scripted_anchor_frac: float, pfsp_mode: str, pfsp_p: float,
                  softmax_eta: float, snapshot_every: int, promote_margin: float,
                  embed_dim: int, no_shaping: bool, on_progress=None, **env_kwargs):
    """Train one learner deck for ``chunk_steps`` against the shared league pool.

    Resumes the learner's own latest checkpoint (``{deck}__final`` or newest
    ``{deck}__v*``) so its cumulative step count — and therefore snapshot version
    numbering — keeps growing across rotations; starts from scratch only the very
    first time a deck is trained.

    ``on_progress(steps_this_chunk)`` (optional) is called after every snapshot save
    with the number of new steps trained so far in this chunk, so the driver can
    persist resumable league progress alongside each checkpoint.
    """
    env_kwargs.setdefault("binary_path", binary_path)
    # League decks may be namespaced under a subfolder (e.g. 'league/<stem>'); mirror
    # that subdir under the checkpoint dir so snapshot saves don't fail.
    deck_subdir = os.path.dirname(learner_deck)
    if deck_subdir:
        os.makedirs(os.path.join(checkpoint_dir, deck_subdir), exist_ok=True)
    vec_env = SubprocVecEnv([
        make_league_env(i, learner_deck, roster, checkpoint_dir, n_envs,
                        opp_ckpt_ratio, self_play_frac, scripted_anchor_frac, **env_kwargs)
        for i in range(n_envs)])
    try:
        policy_kwargs = dict(
            features_extractor_class=CardGameExtractor,
            features_extractor_kwargs=dict(embed_dim=embed_dim),
            net_arch=[256, 256],
        )
        from opponents import latest_snapshot
        resume = os.path.join(checkpoint_dir, f"{learner_deck}__final.zip")
        if not os.path.exists(resume):
            resume = latest_snapshot(learner_deck, checkpoint_dir)
        resuming = bool(resume and os.path.exists(resume))
        if resuming:
            print(f"[league] resuming {learner_deck} from {os.path.basename(resume)}")
            model = MaskablePPO.load(resume, env=vec_env)
        else:
            print(f"[league] starting {learner_deck} from scratch (embed_dim={embed_dim})")
            model = MaskablePPO(
                "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
                learning_rate=3e-4, n_steps=4096, batch_size=1024, n_epochs=8,
                gamma=0.99, gae_lambda=0.95, clip_range=0.25, ent_coef=0.12,
                verbose=1, tensorboard_log=LOG_DIR)

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
            pfsp_cb,
            SnapshotCallback(checkpoint_dir, learner_deck, snapshot_every,
                             promote_margin=promote_margin, pfsp_callback=pfsp_cb,
                             on_snapshot=snap_hook),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))

        model.learn(total_timesteps=chunk_steps, callback=callbacks,
                    reset_num_timesteps=not resuming)
        model.save(os.path.join(checkpoint_dir, f"{learner_deck}__final"))
        print(f"[league] saved {learner_deck}__final")
        # PPO collects whole rollouts, so the chunk overshoots chunk_steps; return
        # the actual new steps so the driver's global budget stays accurate.
        return model.num_timesteps - start_steps
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
           tally: bool = False, resume: bool = False, **env_kwargs):
    """PFSP league driver: rotating single learner over a shared snapshot pool.

    One learner deck at a time trains for ``rotate_every`` steps against a frozen
    pool spanning a scripted anchor, every deck's snapshots, and the learner's own
    latest self; then the run rotates to the next deck. Snapshots dropped into the
    shared dir by earlier rotations become opponents for later ones, so the league
    structure is self-managing. Continues until ``total_timesteps`` total.

    The driver's loop position (roster, total budget, global steps done, current
    rotation, and every hyperparameter) is persisted to a JSON sidecar
    (``checkpoints/_league_progress.json``) every time a snapshot checkpoint is saved
    and at each rotation boundary. ``resume=True`` reloads that sidecar and continues
    where an interrupted run left off — so a crashed league restarts with just
    ``train.py league --resume`` (all other flags are restored from the sidecar).
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    steps_done = 0
    rotation = 0
    if resume:
        state = _read_league_state(checkpoint_dir)
        if state is None:
            raise FileNotFoundError(
                f"--resume: no league progress file at "
                f"{_league_state_path(checkpoint_dir)}. Start a league run first "
                f"(it writes the file as it trains), then resume it.")
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
        env_kwargs.setdefault("bo3", bool(p.get("bo3", env_kwargs.get("bo3", False))))
        env_kwargs.setdefault("auto_sideboard",
                              bool(p.get("auto_sideboard", env_kwargs.get("auto_sideboard", False))))
        print(f"[league] resuming from {_league_state_path(checkpoint_dir)}: "
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

    # League opponents load checkpoint models per env (like self-play), so default
    # to the lighter self-play env count rather than N_ENVS (sized for scripted opps).
    n_envs = n_envs_override if n_envs_override is not None else N_ENVS_SELF_PLAY
    print(f"League roster: {', '.join(roster)}")
    print(f"  total={total_timesteps:,}  rotate_every={rotate_every:,}  n_envs={n_envs}")
    print(f"  self_play_frac={self_play_frac}  scripted_anchor_frac={scripted_anchor_frac}")
    print(f"  pfsp_mode={pfsp_mode}  p={pfsp_p}  eta={softmax_eta}")
    print(f"  snapshot_every={snapshot_every:,}  promote_margin={promote_margin}  embed_dim={embed_dim}")

    # Everything the sidecar needs to faithfully reconstruct this run on --resume.
    base_state = {
        "version": LEAGUE_STATE_VERSION,
        "roster": roster,
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
            "bo3": bool(env_kwargs.get("bo3", False)),
            "auto_sideboard": bool(env_kwargs.get("auto_sideboard", False)),
        },
    }

    def save_progress(done: int, rot: int, cstart: int):
        # `chunk_start` = global steps_done at the start of the current rotation, so a
        # mid-chunk resume can train only the *remainder* of an interrupted rotation
        # rather than a fresh full chunk.
        _write_league_state(checkpoint_dir, {**base_state, "steps_done": int(done),
                                             "rotation": int(rot),
                                             "chunk_start": int(cstart)})

    # `chunk_start` tracks where the current rotation began. On a fresh run it equals
    # steps_done; on --resume it is restored from the sidecar (see below).
    chunk_start = steps_done if not resume else int(state.get("chunk_start", steps_done))

    # Record the starting position immediately so an interruption before the first
    # snapshot still leaves a resumable (if un-advanced) sidecar.
    save_progress(steps_done, rotation, chunk_start)

    while steps_done < total_timesteps:
        learner = roster[rotation % len(roster)]
        done_in_rotation = steps_done - chunk_start
        chunk = min(rotate_every - done_in_rotation, total_timesteps - steps_done)
        if chunk <= 0:
            # Rotation already satisfied (only reachable via an odd resume) — advance.
            rotation += 1
            chunk_start = steps_done
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
        ran = _league_chunk(
            binary_path, learner, roster, checkpoint_dir, chunk,
            n_envs=n_envs, opp_ckpt_ratio=opp_ckpt_ratio,
            self_play_frac=self_play_frac, scripted_anchor_frac=scripted_anchor_frac,
            pfsp_mode=pfsp_mode, pfsp_p=pfsp_p, softmax_eta=softmax_eta,
            snapshot_every=snapshot_every, promote_margin=promote_margin,
            embed_dim=embed_dim, no_shaping=no_shaping,
            on_progress=lambda chunk_steps: save_progress(
                steps_done + chunk_steps, rotation, rotation_start_done),
            **env_kwargs)
        steps_done += ran if ran else chunk
        rotation += 1
        chunk_start = steps_done
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
        policy_kwargs = dict(
            features_extractor_class=CardGameExtractor,
            net_arch=[256, 256],
        )

        print(f"Resuming from {load_path}")
        model = MaskablePPO.load(load_path, env=vec_env)

        if no_shaping:
            vec_env.env_method("set_shaping_scale", 0.0)
            print("[shaping] disabled for this session (--no-shaping)")
        callbacks = [
            SnapshotCallback(checkpoint_dir, model_deck, LEAGUE_SNAPSHOT_EVERY),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))
        if tally:
            callbacks.append(WinTallyCallback())
        callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                           model_deck=model_deck, opp_deck=opp_deck,
                                           bo3=env_kwargs.get("bo3", False)))

        print(f"Training for {total_timesteps:,} timesteps across {n_envs} envs...")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
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


def baseline(binary_path: str, model_path: str, n_games: int = 100):
    """Evaluate win rate of the model against the scripted agent.

    The model is randomly assigned to Player A or B each game (matching training
    conditions).  Reward is from the model's perspective so wins/losses are
    counted directly.
    """
    import numpy as np
    model = MaskablePPO.load(model_path)
    env = ModelVsScriptedEnv(binary_path=binary_path)
    if USE_MASKABLE:
        env = ActionMasker(env, lambda e: e.action_masks())
    wins = losses = draws = 0

    for i in range(n_games):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(int(action))
            total_reward += reward
            done = terminated or truncated
        if total_reward > 0:
            wins += 1
        elif total_reward < 0:
            losses += 1
        else:
            draws += 1
        print(f"\rGame {i+1}/{n_games}  W:{wins} L:{losses} D:{draws}", end="", flush=True)

    env.close()
    print()
    print(f"vs scripted over {n_games} games: {wins}W / {losses}L / {draws}D "
          f"({100 * wins / n_games:.1f}% win rate)")


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
# Card-name and hand decoding route through decode.py (the single source of
# truth) so vocab lookups, the Token sentinel and out-of-range handling stay
# consistent with the rest of the tooling.

def _decode_hand(obs):
    """Return list of card names for the priority player's hand."""
    return decode.decode_hand(obs)


def _describe_action(cats, card_ids, action, num_choices):
    """Return a human-readable string for the chosen action (CAT-token style)."""
    cat = int(cats[action])
    cat_name = _CAT_NAMES.get(cat, str(cat))
    card_name = decode.card_from_id(card_ids[action])
    if card_name:
        return f"{cat_name} {card_name}"
    return cat_name


def _run_sweep(args, parser, decks_filter):
    """Train every deck×deck matchup, optionally filtered to one deck."""
    all_decks = sorted(os.path.splitext(p)[0]
                       for p in os.listdir(_DECKS_DIR) if p.endswith(".dk"))
    if decks_filter:
        if decks_filter not in all_decks:
            parser.error(f"Deck '{decks_filter}' not found in {_DECKS_DIR}. "
                         f"Available: {', '.join(all_decks)}")
        matchups = [(d, o) for d in all_decks for o in all_decks
                    if d == decks_filter or o == decks_filter]
        label = f"featuring '{decks_filter}'"
    else:
        matchups = [(d, o) for d in all_decks for o in all_decks]
        label = "all"
    env_kwargs = dict(bo3=args.bo3, auto_sideboard=args.auto_sideboard)
    print(f"Training {len(matchups)} matchups ({label}) for {args.total_timesteps:,} timesteps each:")
    for d, o in matchups:
        print(f"  {d} vs {o}")
    # Each matchup session continues the trained deck's one generalist ({d}__final),
    # so a deck trained across several opponents in a sweep accumulates onto a
    # single model. train() auto-resumes it; we only force-fresh the very first
    # session of each deck so a re-run keeps building rather than wiping.
    seen_decks = set()
    for i, (d, o) in enumerate(matchups):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(matchups)}] {d} vs {o}")
        print(f"{'='*60}")
        train(args.binary, load_path=None, total_timesteps=args.total_timesteps,
              tally=args.tally, self_play=args.self_play,
              model_deck=d, opp_deck=o, fresh=(args.fresh and d not in seen_decks),
              n_envs_override=args.n_envs, no_shaping=args.no_shaping,
              opponent_pool=args.opponent_pool, opp_ckpt_ratio=args.opponent_ckpt_ratio,
              embed_dim=args.embed_dim, **env_kwargs)
        seen_decks.add(d)
    print(f"\nAll {len(matchups)} matchups complete.")


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

    if args.command == "league":
        league(args.binary, decks=args.decks, total_timesteps=args.total_timesteps,
               rotate_every=args.rotate_every, self_play_frac=args.self_play_frac,
               scripted_anchor_frac=args.scripted_anchor_frac, pfsp_mode=args.pfsp_mode,
               pfsp_p=args.pfsp_p, softmax_eta=args.softmax_eta,
               snapshot_every=args.snapshot_every, promote_margin=args.promote_margin,
               embed_dim=args.embed_dim, n_envs_override=args.n_envs,
               opp_ckpt_ratio=args.opponent_ckpt_ratio, no_shaping=args.no_shaping,
               tally=args.tally, resume=args.resume, **env_kwargs)
    elif args.command == "train":
        train(args.binary, _resolve_model(args.load), args.total_timesteps,
              tally=args.tally, self_play=args.self_play,
              model_deck=args.deck, opp_deck=args.opponent,
              n_envs_override=args.n_envs, no_shaping=args.no_shaping,
              opponent_pool=args.opponent_pool, opp_ckpt_ratio=args.opponent_ckpt_ratio,
              embed_dim=args.embed_dim, fresh=args.fresh, **env_kwargs)
    elif args.command == "sweep":
        _run_sweep(args, parser, args.deck)
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
        observe(args.binary, player_a=args.player_a, player_b=args.player_b,
                deck_a=args.deck, deck_b=args.opponent,
                n_games=args.games, bo3=args.bo3, seed=args.seed, verbose=args.verbose,
                play_a=args.play_a, play_b=args.play_b)
    elif args.command == "baseline":
        baseline(args.binary, _resolve_model(args.model), args.games)
