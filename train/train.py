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

    # Resume from checkpoint:
    python train.py --load checkpoints/robomage_100000_steps.zip
"""

import argparse
import os
import struct
import sys
import time
from collections import deque

from env import (RoboMageEnv, ModelVsScriptedEnv, SelfPlayEnv, FixedModelEnv, NarrativeEnv,
                 scripted_action,
                 OBS_SIZE, STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, BINARY,
                 _HAND_START)
from extractor import CardGameExtractor
from card_costs import _VOCAB_NAMES, N_CARD_TYPES
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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

import numpy as np

# ── Recording format constants (.rmrec) ──────────────────────────────────────
RECORD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
REC_MAGIC = b"RMRC"
REC_VERSION = 2  # v2: card_ids widened from u8 to u16 (vocab > 255)
REC_GAME_START = 0x01
REC_DECISION   = 0x02
REC_GAME_END   = 0x03
REC_CARD_ID_NULL = 0xFFFF  # u16 null sentinel for a per-action card id

# struct formats (all little-endian)
_SESSION_HDR_FMT = "<4sHqH"   # magic(4) + version(u16) + timestamp(i64) + n_envs(u16)
_GAME_START_FMT  = "<BHIb"    # type(u8) + env_id(u16) + game_id(u32) + model_is_a(i8)
_DECISION_FMT    = "<BHIHbbbBBBB6s64s128s64sBBBB"
# type(u8) + env_id(u16) + game_id(u32) + decision_idx(u16) + step_idx(i8)
# + priority_is_a(i8) + active_is_a(i8) + num_choices(u8) + action_chosen(u8)
# + self_life(u8) + opp_life(u8) + self_mana(6B)
# + categories(64B) + card_ids(64×u16 = 128B) + ctrl_flags(64B)
# + self_creatures(u8) + self_lands(u8) + opp_creatures(u8) + opp_lands(u8)
_GAME_END_FMT    = "<BHIbHfI"
# type(u8) + env_id(u16) + game_id(u32) + result(i8) + n_decisions(u16)
# + total_reward(f32) + timestep(u32)


def _write_length_prefixed(f, s: str):
    """Write a u16-length-prefixed UTF-8 string."""
    encoded = s.encode("utf-8")
    f.write(struct.pack("<H", len(encoded)))
    f.write(encoded)


def _read_length_prefixed(f) -> str:
    """Read a u16-length-prefixed UTF-8 string."""
    (length,) = struct.unpack("<H", f.read(2))
    return f.read(length).decode("utf-8")


class RecordCallback(BaseCallback):
    """Records every game decision to a binary .rmrec file during training."""

    def __init__(self, path: str, n_envs: int, model_path: str, model_deck: str,
                 self_play: bool):
        super().__init__()
        self._path = path
        self._n_envs = n_envs
        self._model_path = model_path or "scratch"
        self._model_deck = model_deck
        self._self_play = self_play
        self._file = None
        self._game_id_counter = 0
        self._env_game_ids = {}      # env_id -> current game_id
        self._env_seen_game = set()  # env_ids that have had their Game Start written
        self._env_decisions = {}     # env_id -> count of decisions this game
        self._env_rewards = {}       # env_id -> accumulated reward this game
        self._step_counter = 0

    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._file = open(self._path, "wb")
        # Write session header
        self._file.write(struct.pack(_SESSION_HDR_FMT, REC_MAGIC, REC_VERSION,
                                     int(time.time()), self._n_envs))
        _write_length_prefixed(self._file, self._model_path)
        _write_length_prefixed(self._file, self._model_deck)
        self._file.write(struct.pack("B", 1 if self._self_play else 0))
        self._file.flush()

    def _on_step(self) -> bool:
        self._step_counter += 1
        infos = self.locals["infos"]
        obs = self.locals.get("obs_tensor")
        if obs is not None:
            obs = obs.cpu().numpy() if hasattr(obs, "cpu") else np.asarray(obs)
        actions = self.locals.get("actions")
        if actions is not None:
            actions = np.asarray(actions).flatten()

        for i, info in enumerate(infos):
            meta = info.get("game_meta")
            if meta is None:
                continue

            # New game detection: write Game Start if not seen yet
            if i not in self._env_seen_game:
                self._game_id_counter += 1
                gid = self._game_id_counter
                self._env_game_ids[i] = gid
                self._env_seen_game.add(i)
                self._env_decisions[i] = 0
                self._env_rewards[i] = 0.0
                self._file.write(struct.pack(_GAME_START_FMT, REC_GAME_START,
                                             i, gid, 1 if meta["model_is_a"] else 0))
                _write_length_prefixed(self._file, meta.get("opp_deck", "unknown"))
                _write_length_prefixed(self._file, meta.get("opp_type", "scripted"))

            gid = self._env_game_ids.get(i, 0)
            dec_idx = info.get("decision_idx", self._env_decisions.get(i, 0))
            self._env_decisions[i] = dec_idx

            # Write Decision record
            if obs is not None and i < len(obs):
                o = obs[i]
                step_idx = int(np.argmax(o[18:31]))
                priority_is_a = 1 if o[32] > 0.5 else 0
                active_is_a = 1 if ((o[31] > 0.5) == (priority_is_a == 1)) else 0
                self_life = min(255, max(0, int(round(o[0] * 20))))
                opp_life = min(255, max(0, int(round(o[9] * 20))))
                self_mana = bytes([min(255, max(0, int(round(o[3 + j] * 10)))) for j in range(6)])

                cats_raw = np.round(o[STATE_SIZE:STATE_SIZE + MAX_ACTIONS] * ACTION_CATEGORY_MAX).astype(int)
                ids_raw = o[STATE_SIZE + MAX_ACTIONS:STATE_SIZE + 2 * MAX_ACTIONS]
                ctrl_raw = o[STATE_SIZE + 2 * MAX_ACTIONS:STATE_SIZE + 3 * MAX_ACTIONS]

                # Authoritative choice count from the env (BQUERY header N).
                # The padded action slots are content-indistinguishable from a
                # real PASS (cat 0, id null sentinel), so they cannot be inferred
                # from the obs alone — rely on the env-supplied count, defaulting
                # to MAX_ACTIONS only if it is somehow absent.
                num_choices = min(int(info.get("num_choices", MAX_ACTIONS)), MAX_ACTIONS)

                cats_bytes = bytes([max(0, min(127, int(c))) if j < num_choices else 255
                                    for j, c in enumerate(cats_raw[:MAX_ACTIONS])])
                # Card ids are u16 (vocab can exceed 255); null/out-of-range → 0xFFFF.
                ids_list = []
                for v in ids_raw[:MAX_ACTIONS]:
                    idx = int(round(float(v) * N_CARD_TYPES))
                    ids_list.append(idx if 0 <= idx < N_CARD_TYPES else REC_CARD_ID_NULL)
                ids_bytes = struct.pack(f"<{MAX_ACTIONS}H", *ids_list)
                ctrl_bytes = bytes([1 if v > 0.5 else (0 if v > -0.01 else 255)
                                    for v in ctrl_raw[:MAX_ACTIONS]])

                action_chosen = int(actions[i]) if actions is not None and i < len(actions) else 0

                # Count creatures and lands on each side
                from env import _BF_START, _BF_SLOT_SIZE, _PERM_A_SLOTS, _OFF_IS_CREATURE, _OFF_IS_LAND
                self_creatures = self_lands = opp_creatures = opp_lands = 0
                for s in range(_PERM_A_SLOTS):
                    base = _BF_START + s * _BF_SLOT_SIZE
                    if o[base + _OFF_IS_CREATURE] > 0.5:
                        self_creatures += 1
                    if o[base + _OFF_IS_LAND] > 0.5:
                        self_lands += 1
                    base_opp = _BF_START + (s + _PERM_A_SLOTS) * _BF_SLOT_SIZE
                    if o[base_opp + _OFF_IS_CREATURE] > 0.5:
                        opp_creatures += 1
                    if o[base_opp + _OFF_IS_LAND] > 0.5:
                        opp_lands += 1

                self._file.write(struct.pack(
                    _DECISION_FMT, REC_DECISION, i, gid, dec_idx,
                    step_idx, priority_is_a, active_is_a,
                    min(255, num_choices), min(255, action_chosen),
                    self_life, opp_life, self_mana,
                    cats_bytes, ids_bytes, ctrl_bytes,
                    min(255, self_creatures), min(255, self_lands),
                    min(255, opp_creatures), min(255, opp_lands),
                ))

            # Track reward
            reward = self.locals.get("rewards")
            if reward is not None:
                self._env_rewards[i] = self._env_rewards.get(i, 0.0) + float(reward[i])

            # Game End
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                result = 1 if ep_reward > 0 else (-1 if ep_reward < 0 else 0)
                self._file.write(struct.pack(
                    _GAME_END_FMT, REC_GAME_END, i, gid, result,
                    self._env_decisions.get(i, 0),
                    float(self._env_rewards.get(i, 0.0)),
                    self.num_timesteps,
                ))
                # Reset for next game on this env
                self._env_seen_game.discard(i)

        # Periodic flush
        if self._step_counter % 1000 == 0:
            self._file.flush()

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            print(f"[record] Saved recording to {self._path}")


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
                 min_gate_samples: int = 30):
        super().__init__()
        self._dir = checkpoint_dir
        self._deck = deck
        self._every = max(1, snapshot_every)
        self._margin = promote_margin
        self._pfsp = pfsp_callback
        self._min_gate_samples = min_gate_samples
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
            turn = 0
            prev_active_is_a = None
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

                    if not is_mulligan and active_is_a != prev_active_is_a:
                        turn += 1
                        active_label = "A" if active_is_a else "B"
                        f.write(f"--- Turn {turn} (Player {active_label}) ---\n")
                        f.write(f"  PA: {', '.join(known_hand['A']) or '(empty)'}\n")
                        f.write(f"  PB: {', '.join(known_hand['B']) or '(empty)'}\n")
                        prev_active_is_a = active_is_a

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


def make_env(rank: int, model_deck: str = "delver", opp_deck: str = "delver",
             opponent_pool: str | None = None, opp_ckpt_ratio: float = 1.0,
             n_envs: int = 1, **env_kwargs):
    def _init():
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
          model_deck: str = "delver", opp_deck: str = "delver", record: bool = False,
          n_envs_override: int | None = None, no_shaping: bool = False,
          opponent_pool: str | None = None, opp_ckpt_ratio: float = 1.0,
          embed_dim: int = EMBED_DIM, **env_kwargs):
    """Train the model.

    Two opponent modes (mutually exclusive):
      * default (``self_play=False``) — every env trains against the rule-based
        scripted agent (``ModelVsScriptedEnv``).
      * ``self_play=True`` — every env trains against a frozen saved checkpoint
        of the mirror matchup (``SelfPlayEnv``); if no compatible checkpoint
        exists yet, that env's opponent falls back to the scripted agent.

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

        model_prefix = f"{model_deck}_{opp_deck}"
        if not load_path and self_play:
            candidate = os.path.join(checkpoint_dir, f"{model_prefix}_final.zip")
            if os.path.exists(candidate):
                load_path = candidate
                print(f"Auto-loading self-play checkpoint: {candidate}")

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
                ent_coef=0.12,
                verbose=1,
                tensorboard_log=LOG_DIR,
            )

        actual_n_envs = n_envs
        if no_shaping:
            vec_env.env_method("set_shaping_scale", 0.0)
            print("[shaping] disabled for this session (--no-shaping)")
        callbacks = [
            CheckpointCallback(
                save_freq=250_000 // actual_n_envs,
                save_path=checkpoint_dir,
                name_prefix=model_prefix,
            ),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))
        if tally:
            callbacks.append(WinTallyCallback())
        callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                           model_deck=model_deck, opp_deck=opp_deck,
                                           bo3=env_kwargs.get("bo3", False)))
        if record:
            rec_path = os.path.join(RECORD_DIR, f"{model_prefix}_{int(time.time())}.rmrec")
            callbacks.append(RecordCallback(
                path=rec_path, n_envs=actual_n_envs, model_path=load_path,
                model_deck=model_deck, self_play=self_play,
            ))

        print(f"Training for {total_timesteps:,} timesteps across {actual_n_envs} envs...")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=load_path is None)
        model.save(os.path.join(checkpoint_dir, f"{model_prefix}_final"))
        print(f"Saved final model as {model_prefix}_final.")
    finally:
        vec_env.close()


def _league_chunk(binary_path: str, learner_deck: str, roster: list[str],
                  checkpoint_dir: str, chunk_steps: int, *,
                  n_envs: int, opp_ckpt_ratio: float, self_play_frac: float,
                  scripted_anchor_frac: float, pfsp_mode: str, pfsp_p: float,
                  softmax_eta: float, snapshot_every: int, promote_margin: float,
                  embed_dim: int, no_shaping: bool, record: bool, **env_kwargs):
    """Train one learner deck for ``chunk_steps`` against the shared league pool.

    Resumes the learner's own latest checkpoint (``{deck}__final`` or newest
    ``{deck}__v*``) so its cumulative step count — and therefore snapshot version
    numbering — keeps growing across rotations; starts from scratch only the very
    first time a deck is trained.
    """
    env_kwargs.setdefault("binary_path", binary_path)
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

        pfsp_cb = PFSPCallback(vec_env, mode=pfsp_mode, p=pfsp_p, eta=softmax_eta)
        callbacks = [
            pfsp_cb,
            SnapshotCallback(checkpoint_dir, learner_deck, snapshot_every,
                             promote_margin=promote_margin, pfsp_callback=pfsp_cb),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))
        if record:
            rec_path = os.path.join(RECORD_DIR, f"{learner_deck}__league_{int(time.time())}.rmrec")
            callbacks.append(RecordCallback(
                path=rec_path, n_envs=n_envs, model_path=resume,
                model_deck=learner_deck, self_play=True))

        start_steps = model.num_timesteps
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
           record: bool = False, tally: bool = False, **env_kwargs):
    """PFSP league driver: rotating single learner over a shared snapshot pool.

    One learner deck at a time trains for ``rotate_every`` steps against a frozen
    pool spanning a scripted anchor, every deck's snapshots, and the learner's own
    latest self; then the run rotates to the next deck. Snapshots dropped into the
    shared dir by earlier rotations become opponents for later ones, so the league
    structure is self-managing. Continues until ``total_timesteps`` total.
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if decks:
        roster = [d.strip() for d in decks.split(",") if d.strip()]
    else:
        roster = sorted(os.path.splitext(p)[0]
                        for p in os.listdir(_DECKS_DIR) if p.endswith(".dk"))
    if not roster:
        raise ValueError(f"No decks found for league (looked in {_DECKS_DIR})")

    # League opponents load checkpoint models per env (like self-play), so default
    # to the lighter self-play env count rather than N_ENVS (sized for scripted opps).
    n_envs = n_envs_override if n_envs_override is not None else N_ENVS_SELF_PLAY
    print(f"League roster: {', '.join(roster)}")
    print(f"  total={total_timesteps:,}  rotate_every={rotate_every:,}  n_envs={n_envs}")
    print(f"  self_play_frac={self_play_frac}  scripted_anchor_frac={scripted_anchor_frac}")
    print(f"  pfsp_mode={pfsp_mode}  p={pfsp_p}  eta={softmax_eta}")
    print(f"  snapshot_every={snapshot_every:,}  promote_margin={promote_margin}  embed_dim={embed_dim}")

    steps_done = 0
    rotation = 0
    while steps_done < total_timesteps:
        learner = roster[rotation % len(roster)]
        chunk = min(rotate_every, total_timesteps - steps_done)
        print(f"\n{'='*60}")
        print(f"[league rotation {rotation + 1}] learner={learner}  "
              f"({chunk:,} steps, {steps_done:,}/{total_timesteps:,} done)")
        print(f"{'='*60}")
        ran = _league_chunk(
            binary_path, learner, roster, checkpoint_dir, chunk,
            n_envs=n_envs, opp_ckpt_ratio=opp_ckpt_ratio,
            self_play_frac=self_play_frac, scripted_anchor_frac=scripted_anchor_frac,
            pfsp_mode=pfsp_mode, pfsp_p=pfsp_p, softmax_eta=softmax_eta,
            snapshot_every=snapshot_every, promote_margin=promote_margin,
            embed_dim=embed_dim, no_shaping=no_shaping, record=record, **env_kwargs)
        steps_done += ran if ran else chunk
        rotation += 1

    print(f"\nLeague complete: {total_timesteps:,} total timesteps over {rotation} rotations.")


def train_fixed_model(binary_path: str, model_deck: str, opp_deck: str,
                      load_path: str | None = None,
                      total_timesteps: int = TOTAL_TIMESTEPS,
                      tally: bool = False, record: bool = False,
                      n_envs_override: int | None = None,
                      no_shaping: bool = False, **env_kwargs):
    """Train model_deck against a fixed opponent model for opp_deck.

    Loads ``{model_deck}_{opp_deck}_final.zip`` as the training model (or
    ``load_path`` if given) and ``{opp_deck}_{model_deck}_final.zip`` as the
    fixed opponent for every game.

    Extra keyword arguments (``bo3``, ``auto_sideboard``, etc.) are forwarded
    to the underlying ``RoboMageEnv`` via the env factory helpers.
    """
    env_kwargs.setdefault("binary_path", binary_path)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    model_prefix = f"{model_deck}_{opp_deck}"
    opp_prefix = f"{opp_deck}_{model_deck}"

    if not load_path:
        candidate = os.path.join(checkpoint_dir, f"{model_prefix}_final.zip")
        if os.path.exists(candidate):
            load_path = candidate
    if not load_path:
        raise FileNotFoundError(f"No training model found: {model_prefix}_final.zip")

    opp_model_path = os.path.join(checkpoint_dir, f"{opp_prefix}_final.zip")
    if not os.path.exists(opp_model_path):
        raise FileNotFoundError(f"No opponent model found: {opp_model_path}")

    print(f"Training {model_prefix} against fixed opponent {opp_prefix}")
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
            CheckpointCallback(
                save_freq=250_000 // n_envs,
                save_path=checkpoint_dir,
                name_prefix=model_prefix,
            ),
        ]
        if not no_shaping:
            callbacks.append(ShapingScaleCallback(vec_env))
        if tally:
            callbacks.append(WinTallyCallback())
        callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                           model_deck=model_deck, opp_deck=opp_deck,
                                           bo3=env_kwargs.get("bo3", False)))
        if record:
            rec_path = os.path.join(RECORD_DIR, f"{model_prefix}_fixed_{int(time.time())}.rmrec")
            callbacks.append(RecordCallback(
                path=rec_path, n_envs=n_envs, model_path=load_path,
                model_deck=model_deck, self_play=False,
            ))

        print(f"Training for {total_timesteps:,} timesteps across {n_envs} envs...")
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=False)
        model.save(os.path.join(checkpoint_dir, f"{model_prefix}_final"))
        print(f"Saved final model as {model_prefix}_final.")
    finally:
        vec_env.close()


def train_alternate(binary_path: str, deck_a: str, deck_b: str,
                    alternate_steps: int, total_timesteps: int = TOTAL_TIMESTEPS,
                    tally: bool = False, record: bool = False,
                    n_envs_override: int | None = None,
                    no_shaping: bool = False, **env_kwargs):
    """Alternate training between two decks every ``alternate_steps`` timesteps.

    Each round trains one side against the other's latest final checkpoint,
    then saves and swaps roles.

    Extra keyword arguments (``bo3``, ``auto_sideboard``, etc.) are forwarded
    to the underlying ``RoboMageEnv`` via ``train_fixed_model``.
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Verify both models exist
    for d, o in [(deck_a, deck_b), (deck_b, deck_a)]:
        path = os.path.join(checkpoint_dir, f"{d}_{o}_final.zip")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}")

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
                          total_timesteps=round_steps, tally=tally, record=record,
                          n_envs_override=n_envs_override,
                          no_shaping=no_shaping, **env_kwargs)

        steps_done += round_steps

        # Swap roles
        training_deck, opp_deck = opp_deck, training_deck

    print(f"\nAlternate training complete: {total_timesteps:,} total timesteps over {round_num} rounds.")


def diag(binary_path: str, n_games: int = 10,
         deck_a: str | None = None, deck_b: str | None = None,
         bo3: bool = False):
    """Quick diagnostic: run n_games of scripted vs scripted to verify the env.

    Logs every decision (like watch_scripted).  On a draw the full log is saved
    to diag_draw_<game>.txt and printed to stdout, since draws should not occur.
    """
    import numpy as np

    wins = losses = draws = 0
    label = "matches" if bo3 else "games"
    print(f"Running {n_games} {label} (scripted vs scripted)...")
    for i in range(n_games):
        env = NarrativeEnv(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b, bo3=bo3)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        decision = 0
        turn = 0
        prev_active_is_a = None
        known_hand = {"A": [], "B": []}
        log_lines = [f"=== Game {i+1}: Scripted (A) vs Scripted (B) ===\n"]

        while not done:
            # Flush narrative lines from the game process
            raw_env = env.env if hasattr(env, "env") else env
            for line in raw_env.flush_lines():
                if line.strip():
                    log_lines.append(line)

            num_choices = raw_env._num_choices
            priority_is_a = obs[32] > 0.5
            player = "A" if priority_is_a else "B"
            active_is_a = (obs[31] > 0.5) == priority_is_a
            step_name = _STEP_NAMES[int(np.argmax(obs[18:31]))]

            cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
            is_mulligan = any(c == 11 for c in cats)

            known_hand[player] = _decode_hand(obs)

            if not is_mulligan and active_is_a != prev_active_is_a:
                turn += 1
                active_label = "A" if active_is_a else "B"
                a_hand = ", ".join(known_hand["A"]) or "(empty)"
                b_hand = ", ".join(known_hand["B"]) or "(empty)"
                log_lines.append(f"--- Turn {turn} (Player {active_label}) ---")
                log_lines.append(f"  PA: {a_hand}")
                log_lines.append(f"  PB: {b_hand}")
                prev_active_is_a = active_is_a

            # Scripted agent controls both players
            if priority_is_a:
                action = scripted_action(obs, num_choices)
                agent_label = "Scripted/A"
            else:
                action = scripted_action(obs, num_choices)
                agent_label = "Scripted/B"

            chosen_cat = _CAT_NAMES.get(int(cats[action]), str(cats[action]))
            all_cats = [_CAT_NAMES.get(int(c), str(c)) for c in cats]
            log_lines.append(
                f"[{decision:4d}] P{player}  {step_name:<14}  choices={num_choices}"
                f"  available={all_cats}  -> {action} ({chosen_cat})  [{agent_label}]"
            )

            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            decision += 1

        for line in raw_env.flush_lines():
            if line.strip():
                log_lines.append(line)

        env.close()

        if total_reward > 0:
            wins += 1
            result = "W"
        elif total_reward < 0:
            losses += 1
            result = "L"
        else:
            draws += 1
            result = "D"
            # Draws should not occur — save and print the full log
            log_path = f"diag_draw_{i+1}.txt"
            log_lines.append("\n=== DRAW (should not occur) ===")
            log_text = "\n".join(log_lines)
            with open(log_path, "w") as f:
                f.write(log_text + "\n")
            print(f"\n{'='*60}")
            print(f"DRAW in game {i+1} — saving log to {log_path}")
            print('='*60)
            print(log_text, flush=True)

        print(f"  game {i+1:2d}/{n_games}: {result}  (W:{wins} L:{losses} D:{draws})", flush=True)

    total = wins + losses + draws
    win_pct = 100 * wins / total if total else 0
    print(f"\n{wins}W / {losses}L / {draws}D over {n_games} games ({win_pct:.1f}% win rate)")


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
            deck_a: str | None = None, deck_b: str | None = None):
    """Watch one game with user-chosen controllers and decks for each side.

    ``player_a``/``player_b`` are either the literal "scripted" or a path to a
    model checkpoint (.zip path or shorthand).  ``deck_a``/``deck_b`` set each
    side's deck.  Both players' decisions are printed so the full game flow is
    visible.
    """
    from opponents import make_controller, is_scripted_spec

    def _label(spec, side):
        kind = "Scripted" if is_scripted_spec(spec or "scripted") else "Model"
        return f"{kind}/{side}"

    # observe watches a fixed game, so use deterministic model predictions.
    ctrl_a = make_controller(player_a or "scripted",
                             checkpoint_resolver=_resolve_model, deterministic=True)
    ctrl_b = make_controller(player_b or "scripted",
                             checkpoint_resolver=_resolve_model, deterministic=True)

    env = RoboMageEnv(binary_path=binary_path, render_mode="human",
                      deck_a=deck_a, deck_b=deck_b)
    obs, _ = env.reset()

    print(f"=== {_label(player_a, 'A')} ({deck_a or 'default'} deck) vs "
          f"{_label(player_b, 'B')} ({deck_b or 'default'} deck) ===\n")

    done = False
    total_reward = 0.0
    while not done:
        a_has_priority = obs[32] > 0.5
        cur_side = "A" if a_has_priority else "B"
        num_choices = env._num_choices
        controller = ctrl_a if a_has_priority else ctrl_b
        masks = env.action_masks() if USE_MASKABLE else None
        action = controller.choose(obs, num_choices, action_masks=masks)
        spec = player_a if a_has_priority else player_b
        print(f"  [{_label(spec, cur_side)}] action {action + 1} of {num_choices}")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    print()
    # total_reward is from Player A's perspective.
    if total_reward > 0:
        print(f"=== {_label(player_a, 'A')} wins ===")
    elif total_reward < 0:
        print(f"=== {_label(player_b, 'B')} wins ===")
    else:
        print("=== Draw ===")


_CAT_NAMES = {
    0: "PASS", 1: "MANA", 2: "SEL_ATK", 3: "CONF_ATK",
    4: "SEL_BLK", 5: "CONF_BLK", 6: "ACTIVATE", 7: "CAST",
    8: "TARGET", 9: "LAND", 10: "OTHER", 11: "MULLIGAN", 12: "BOTTOM_CARD",
    13: "MANA_W", 14: "MANA_U", 15: "MANA_B", 16: "MANA_R", 17: "MANA_G",
    18: "MANA_C", 19: "SEARCH", 20: "TOP_LIB", 21: "SHUFFLE", 22: "PAYING",
    23: "DIG", 24: "SB_IN", 25: "SB_OUT", 26: "SB_DONE",
}

_STEP_NAMES = [
    "Untap", "Upkeep", "Draw", "First Main", "Begin Combat",
    "Declare Atk", "Declare Blk", "First Strike Dmg", "Combat Dmg",
    "End Combat", "Second Main", "End Step", "Cleanup",
]


def _decode_hand(obs):
    """Return list of card names for the priority player's hand."""
    cards = []
    for slot in range(10):            # MAX_HAND_SLOTS = 10
        idx = int(round(float(obs[_HAND_START + slot]) * N_CARD_TYPES))
        if idx >= 0:
            cards.append(_VOCAB_NAMES[idx] if idx < len(_VOCAB_NAMES) else f"?{idx}")
    return cards


def _describe_action(cats, card_ids, action, num_choices):
    """Return a human-readable string for the chosen action."""
    cat = int(cats[action])
    cat_name = _CAT_NAMES.get(cat, str(cat))

    # Decode card name from the normalized card ID float
    raw_id = float(card_ids[action])
    card_name = None
    if raw_id >= 0:
        vocab_idx = round(raw_id * N_CARD_TYPES)
        if vocab_idx < len(_VOCAB_NAMES) and _VOCAB_NAMES[vocab_idx]:
            card_name = _VOCAB_NAMES[vocab_idx]

    if card_name:
        return f"{cat_name} {card_name}"
    return cat_name


def watch_scripted(binary_path: str, deck_a: str | None = None, deck_b: str | None = None,
                   bo3: bool = False):
    """Run one game with both players driven by the scripted agent and print every decision."""
    import numpy as np

    env = NarrativeEnv(binary_path=binary_path, deck_a=deck_a, deck_b=deck_b, bo3=bo3)
    obs, _ = env.reset()
    done = False
    decision = 0
    turn = 0
    prev_active_is_a = None       # None until first non-mulligan query
    known_hand = {"A": [], "B": []}

    label_a = deck_a or "default"
    label_b = deck_b or "default"
    print(f"=== Scripted (A: {label_a}) vs Scripted (B: {label_b}) ===\n", flush=True)

    while not done:
        # Print narrative from the previous step; skip blank lines (e.g. leading \n
        # from C++ printf("\n--- Declare Attackers...")) to avoid extra whitespace.
        for line in env.flush_lines():
            if line.strip():
                print(line, flush=True)

        num_choices = env._num_choices
        priority_is_a = obs[32] > 0.5
        player = "A" if priority_is_a else "B"
        # active_is_a: obs[31]=1 means priority player IS the active player
        active_is_a = (obs[31] > 0.5) == priority_is_a
        step_name = _STEP_NAMES[int(np.argmax(obs[18:31]))]

        cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
        is_mulligan = any(c == 11 for c in cats)

        # Cache the priority player's hand on every query
        known_hand[player] = _decode_hand(obs)

        # Print a turn banner when the active player changes (ignore mulligan phase)
        if not is_mulligan and active_is_a != prev_active_is_a:
            turn += 1
            active_label = "A" if active_is_a else "B"
            a_hand = ", ".join(known_hand["A"]) or "(empty)"
            b_hand = ", ".join(known_hand["B"]) or "(empty)"
            print(f"--- Turn {turn} (Player {active_label}) ---", flush=True)
            print(f"  PA: {a_hand}", flush=True)
            print(f"  PB: {b_hand}", flush=True)
            prev_active_is_a = active_is_a

        action = scripted_action(obs, num_choices)
        chosen_cat = _CAT_NAMES.get(int(cats[action]), str(cats[action]))
        all_cats = [_CAT_NAMES.get(int(c), str(c)) for c in cats]

        print(f"[{decision:4d}] P{player}  {step_name:<14}  choices={num_choices}  available={all_cats}  -> {action} ({chosen_cat})",
              flush=True)

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        decision += 1

    for line in env.flush_lines():
        if line.strip():
            print(line, flush=True)

    print(flush=True)
    if reward > 0:
        print("=== Player A wins ===")
    elif reward < 0:
        print("=== Player B wins ===")
    else:
        print("=== No reward recorded ===")
    env.close()


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
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    for i, (d, o) in enumerate(matchups):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(matchups)}] {d} vs {o}")
        print(f"{'='*60}")
        candidate = os.path.join(checkpoint_dir, f"{d}_{o}_final.zip")
        resume_path = candidate if os.path.exists(candidate) else None
        train(args.binary, load_path=resume_path, total_timesteps=args.total_timesteps,
              tally=args.tally, self_play=args.self_play,
              model_deck=d, opp_deck=o, record=args.record,
              n_envs_override=args.n_envs, no_shaping=args.no_shaping,
              opponent_pool=args.opponent_pool, opp_ckpt_ratio=args.opponent_ckpt_ratio,
              embed_dim=args.embed_dim, **env_kwargs)
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
               record=args.record, tally=args.tally, **env_kwargs)
    elif args.command == "train":
        train(args.binary, _resolve_model(args.load), args.total_timesteps,
              tally=args.tally, self_play=args.self_play,
              model_deck=args.deck, opp_deck=args.opponent, record=args.record,
              n_envs_override=args.n_envs, no_shaping=args.no_shaping,
              opponent_pool=args.opponent_pool, opp_ckpt_ratio=args.opponent_ckpt_ratio,
              embed_dim=args.embed_dim, **env_kwargs)
    elif args.command == "sweep":
        _run_sweep(args, parser, args.deck)
    elif args.command == "fixed-model":
        train_fixed_model(args.binary, args.deck, args.opponent,
                          load_path=_resolve_model(args.load),
                          total_timesteps=args.total_timesteps,
                          tally=args.tally, record=args.record,
                          n_envs_override=args.n_envs,
                          no_shaping=args.no_shaping, **env_kwargs)
    elif args.command == "alternate":
        train_alternate(args.binary, args.deck, args.opponent,
                        alternate_steps=args.every,
                        total_timesteps=args.total_timesteps,
                        tally=args.tally, record=args.record,
                        n_envs_override=args.n_envs,
                        no_shaping=args.no_shaping, **env_kwargs)
    elif args.command == "diag":
        diag(args.binary, args.games, deck_a=args.deck, deck_b=args.opponent, bo3=args.bo3)
    elif args.command == "watch":
        watch_scripted(args.binary, deck_a=args.deck, deck_b=args.opponent, bo3=args.bo3)
    elif args.command == "observe":
        observe(args.binary, player_a=args.player_a, player_b=args.player_b,
                deck_a=args.deck, deck_b=args.opponent)
    elif args.command == "baseline":
        baseline(args.binary, _resolve_model(args.model), args.games)
