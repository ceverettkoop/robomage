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
import time

from env import (RoboMageEnv, ModelVsScriptedEnv, SelfPlayEnv, FixedModelEnv, NarrativeEnv,
                 scripted_action,
                 OBS_SIZE, STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, BINARY,
                 _HAND_START)
from extractor import CardGameExtractor
from card_costs import _VOCAB_NAMES, N_CARD_TYPES

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
REC_VERSION = 1
REC_GAME_START = 0x01
REC_DECISION   = 0x02
REC_GAME_END   = 0x03

# struct formats (all little-endian)
_SESSION_HDR_FMT = "<4sHqH"   # magic(4) + version(u16) + timestamp(i64) + n_envs(u16)
_GAME_START_FMT  = "<BHIb"    # type(u8) + env_id(u16) + game_id(u32) + model_is_a(i8)
_DECISION_FMT    = "<BHIHbbbBBBB6s32s32s32sBBBB"
# type(u8) + env_id(u16) + game_id(u32) + decision_idx(u16) + step_idx(i8)
# + priority_is_a(i8) + active_is_a(i8) + num_choices(u8) + action_chosen(u8)
# + self_life(u8) + opp_life(u8) + self_mana(6B)
# + categories(32B) + card_ids(32B) + ctrl_flags(32B)
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

                num_c = int(info.get("decision_idx", 0))  # approximate
                # Determine num_choices from the action mask pattern
                num_choices = int(np.sum(cats_raw[:MAX_ACTIONS] >= 0))  # heuristic
                # Better: count non-padded
                for nc in range(MAX_ACTIONS):
                    if nc > 0 and cats_raw[nc] == 0 and ids_raw[nc] < -0.5:
                        num_choices = nc
                        break
                else:
                    num_choices = MAX_ACTIONS

                cats_bytes = bytes([max(0, min(127, int(c))) if j < num_choices else 255
                                    for j, c in enumerate(cats_raw[:MAX_ACTIONS])])
                ids_bytes = bytes([max(0, min(127, int(round(float(v) * N_CARD_TYPES)))) if v >= 0 else 255
                                   for v in ids_raw[:MAX_ACTIONS]])
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
        self._vec_env.set_attr("shaping_scale", scale)
        print(f"[shaping] win_rate={win_rate:.2f}  shaping_scale={scale:.2f}")
        self._wins = 0
        self._losses = 0


class ReplayLogCallback(BaseCallback):
    """After each rollout, runs one model-vs-scripted game and saves a transcript."""

    def __init__(self, binary_path: str, replay_dir: str = "replays",
                 model_deck: str | None = None, opp_deck: str | None = None):
        super().__init__()
        self.binary_path = binary_path
        self.replay_dir = replay_dir
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._rollout = 0
        os.makedirs(replay_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        import numpy as np
        self._rollout += 1
        log_path = os.path.join(self.replay_dir, f"rollout_{self._rollout:05d}.txt")

        env = NarrativeEnv(binary_path=self.binary_path,
                           deck_a=self._model_deck, deck_b=self._opp_deck)
        if USE_MASKABLE:
            from sb3_contrib.common.wrappers import ActionMasker as _AM
            masked = _AM(env, lambda e: e.action_masks())
        else:
            masked = env

        try:
            obs, _ = masked.reset()
            model_is_a = bool(np.random.random() < 0.5)
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
                        f.write(f"[Model/{cur_side}] {desc}  ({action} of {num_choices})\n")
                    else:
                        action = scripted_action(obs, num_choices)
                        desc = _describe_action(cats, card_ids, action, num_choices)
                        f.write(f"[Scripted/{cur_side}] {desc}  ({action} of {num_choices})\n")

                    obs, reward, terminated, truncated, _ = masked.step(action)
                    total_reward += reward
                    done = terminated or truncated

                for line in env.flush_lines():
                    if line.strip():
                        f.write(line + "\n")

                model_reward = total_reward if model_is_a else -total_reward
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

    Accepts:
      - Full path (returned as-is if it exists)
      - Bare matchup name like 'delver_boomer-mav' → checkpoints/delver_boomer-mav_final.zip
    """
    if path is None:
        return None
    # Already a real path
    if os.path.exists(path):
        return path
    # Try as matchup shorthand → checkpoints/{name}_final.zip
    candidate = os.path.join(_CHECKPOINT_ABS, f"{path}_final.zip")
    if os.path.exists(candidate):
        return candidate
    # Try with .zip appended (e.g. 'delver_boomer-mav_100000_steps')
    candidate2 = os.path.join(_CHECKPOINT_ABS, f"{path}.zip")
    if os.path.exists(candidate2):
        return candidate2
    # Return original — let downstream code report the error
    return path
TOTAL_TIMESTEPS = 2_000_000
N_ENVS = 32           # parallel game processes
N_ENVS_SELF_PLAY = 12 # self-play (each loads an opponent model)
_DECKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "bin", "resources", "decks")


def make_env(binary_path: str, rank: int, model_deck: str = "delver", opp_deck: str = "delver"):
    def _init():
        env = ModelVsScriptedEnv(binary_path=binary_path,
                                 model_deck=model_deck, opp_deck=opp_deck)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def make_fixed_model_env(binary_path: str, opp_model_path: str, rank: int,
                         model_deck: str = "delver", opp_deck: str = "delver"):
    def _init():
        env = FixedModelEnv(opp_model_path=opp_model_path, binary_path=binary_path,
                            model_deck=model_deck, opp_deck=opp_deck)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def make_self_play_env(binary_path: str, checkpoint_dir: str, rank: int,
                       model_deck: str = "delver", opp_deck: str = "delver"):
    def _init():
        env = SelfPlayEnv(checkpoint_dir=checkpoint_dir, binary_path=binary_path,
                          model_deck=model_deck, opp_deck=opp_deck)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def train(binary_path: str, load_path: str | None = None, total_timesteps: int = TOTAL_TIMESTEPS,
          tally: bool = False, self_play: bool = False, scripted_fraction: float = 0.0,
          model_deck: str = "delver", opp_deck: str = "delver", record: bool = False):
    """Train the model.

    ``scripted_fraction`` controls how many of the N_ENVS parallel environments
    use the scripted agent instead of self-play.  E.g. 0.0 = all self-play,
    0.3 = ~2 scripted + 5 self-play (with N_ENVS=7).  Has no effect unless
    ``self_play`` is also True.  Mixing in scripted environments prevents the
    policy drifting to strategies that beat itself but lose to general play.
    """
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), CHECKPOINT_DIR)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Parallel environments for faster data collection
    if self_play:
        n_envs = N_ENVS_SELF_PLAY
        n_scripted = round(n_envs * scripted_fraction)
        n_self_play = n_envs - n_scripted
        env_fns = (
            [make_self_play_env(binary_path, checkpoint_dir, i, model_deck, opp_deck) for i in range(n_self_play)]
            + [make_env(binary_path, n_envs - n_scripted + i, model_deck, opp_deck) for i in range(n_scripted)]
        )
        if n_scripted:
            print(f"Env mix: {n_self_play} self-play + {n_scripted} scripted")
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv([make_env(binary_path, i, model_deck, opp_deck) for i in range(N_ENVS)])

    policy_kwargs = dict(
        features_extractor_class=CardGameExtractor,
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
            n_epochs=4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.25,
            ent_coef=0.08,         
            verbose=1,
            tensorboard_log=LOG_DIR,
        )

    actual_n_envs = n_envs if self_play else N_ENVS
    callbacks = [
        CheckpointCallback(
            save_freq=100_000 // actual_n_envs,
            save_path=checkpoint_dir,
            name_prefix=model_prefix,
        ),
        ShapingScaleCallback(vec_env),
    ]
    if tally:
        callbacks.append(WinTallyCallback())
    callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                       model_deck=model_deck, opp_deck=opp_deck))
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

    vec_env.close()


def train_fixed_model(binary_path: str, model_deck: str, opp_deck: str,
                      load_path: str | None = None,
                      total_timesteps: int = TOTAL_TIMESTEPS,
                      tally: bool = False, record: bool = False):
    """Train model_deck against a fixed opponent model for opp_deck.

    Loads ``{model_deck}_{opp_deck}_final.zip`` as the training model (or
    ``load_path`` if given) and ``{opp_deck}_{model_deck}_final.zip`` as the
    fixed opponent for every game.
    """
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

    n_envs = N_ENVS_SELF_PLAY
    vec_env = SubprocVecEnv([
        make_fixed_model_env(binary_path, opp_model_path, i, model_deck, opp_deck)
        for i in range(n_envs)
    ])

    policy_kwargs = dict(
        features_extractor_class=CardGameExtractor,
        net_arch=[256, 256],
    )

    print(f"Resuming from {load_path}")
    model = MaskablePPO.load(load_path, env=vec_env)

    callbacks = [
        CheckpointCallback(
            save_freq=100_000 // n_envs,
            save_path=checkpoint_dir,
            name_prefix=model_prefix,
        ),
        ShapingScaleCallback(vec_env),
    ]
    if tally:
        callbacks.append(WinTallyCallback())
    callbacks.append(ReplayLogCallback(binary_path=binary_path,
                                       model_deck=model_deck, opp_deck=opp_deck))
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
    vec_env.close()


def train_alternate(binary_path: str, deck_a: str, deck_b: str,
                    alternate_steps: int, total_timesteps: int = TOTAL_TIMESTEPS,
                    tally: bool = False, record: bool = False):
    """Alternate training between two decks every ``alternate_steps`` timesteps.

    Each round trains one side against the other's latest final checkpoint,
    then saves and swaps roles.
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
                          total_timesteps=round_steps, tally=tally, record=record)

        steps_done += round_steps

        # Swap roles
        training_deck, opp_deck = opp_deck, training_deck

    print(f"\nAlternate training complete: {total_timesteps:,} total timesteps over {round_num} rounds.")


def diag(binary_path: str, n_games: int = 10):
    """Quick diagnostic: spin up a fresh random model and run n_games vs scripted.

    Logs every decision (like watch_scripted).  On a draw the full log is saved
    to diag_draw_<game>.txt and printed to stdout, since draws should not occur.
    """
    import numpy as np

    policy_kwargs = dict(features_extractor_class=CardGameExtractor, net_arch=[256, 256])

    # Create a throw-away env just to give MaskablePPO the spaces it needs.
    _tmp_env = NarrativeEnv(binary_path=binary_path)
    if USE_MASKABLE:
        _tmp_env = ActionMasker(_tmp_env, lambda e: e.action_masks())
    model = MaskablePPO("MlpPolicy", _tmp_env, policy_kwargs=policy_kwargs, verbose=0)
    _tmp_env.close()

    wins = losses = draws = 0
    print(f"Running {n_games} games (random model vs scripted)...")
    for i in range(n_games):
        env = NarrativeEnv(binary_path=binary_path)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())

        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        decision = 0
        turn = 0
        prev_active_is_a = None
        known_hand = {"A": [], "B": []}
        log_lines = [f"=== Game {i+1}: Random Model (A) vs Scripted (B) ===\n"]

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

            # Model controls player A, scripted controls player B
            if priority_is_a:
                masks = raw_env.action_masks() if USE_MASKABLE else None
                action, _ = model.predict(obs, action_masks=masks, deterministic=False)
                action = int(action)
                agent_label = "Model/A"
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


def observe(binary_path: str, model_path: str):
    """Watch the model play one game against the scripted agent.

    The model is randomly assigned to Player A or B.  Both players' decisions
    are printed so the full game flow is visible.
    """
    import numpy as np
    model = MaskablePPO.load(model_path)
    env = RoboMageEnv(binary_path=binary_path, render_mode="human")
    obs, _ = env.reset()

    model_is_a = bool(np.random.random() < 0.5)
    model_side = "A" if model_is_a else "B"
    scripted_side = "B" if model_is_a else "A"
    print(f"=== Model (Player {model_side}) vs Scripted (Player {scripted_side}) ===\n")

    done = False
    total_reward = 0.0
    while not done:
        a_has_priority = obs[32] > 0.5
        model_has_priority = a_has_priority if model_is_a else not a_has_priority
        cur_side = "A" if a_has_priority else "B"
        num_choices = env._num_choices

        if model_has_priority:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)
            print(f"  [Model/{cur_side}]    action {action} of {num_choices}")
        else:
            action = scripted_action(obs, num_choices)
            print(f"  [Scripted/{cur_side}] action {action} of {num_choices}")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    print()
    model_reward = total_reward if model_is_a else -total_reward
    if model_reward > 0:
        print(f"=== Model ({model_side}) wins ===")
    elif model_reward < 0:
        print(f"=== Scripted ({scripted_side}) wins ===")
    else:
        print("=== Draw ===")


_CAT_NAMES = {
    0: "PASS", 1: "MANA", 2: "SEL_ATK", 3: "CONF_ATK",
    4: "SEL_BLK", 5: "CONF_BLK", 6: "ACTIVATE", 7: "CAST",
    8: "TARGET", 9: "LAND", 10: "OTHER", 11: "MULLIGAN", 12: "BOTTOM_CARD"
}

_STEP_NAMES = [
    "Untap", "Upkeep", "Draw", "First Main", "Begin Combat",
    "Declare Atk", "Declare Blk", "First Strike Dmg", "Combat Dmg",
    "End Combat", "Second Main", "End Step", "Cleanup",
]


def _decode_hand(obs):
    """Return list of card names for the priority player's hand."""
    import numpy as np
    cards = []
    for slot in range(10):            # MAX_HAND_SLOTS = 10
        base = _HAND_START + slot * N_CARD_TYPES
        vec = obs[base : base + N_CARD_TYPES]
        idx = int(np.argmax(vec))
        if vec[idx] > 0.5:
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


def watch_scripted(binary_path: str):
    """Run one game with both players driven by the scripted agent and print every decision."""
    import numpy as np

    env = NarrativeEnv(binary_path=binary_path)
    obs, _ = env.reset()
    done = False
    decision = 0
    turn = 0
    prev_active_is_a = None       # None until first non-mulligan query
    known_hand = {"A": [], "B": []}

    print("=== Scripted (A) vs Scripted (B) ===\n", flush=True)

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


def evaluate(binary_path: str, model_path: str, n_games: int = 100):
    """Play n_games with a trained model and report win rate."""
    env = RoboMageEnv(binary_path=binary_path)
    if USE_MASKABLE:
        env = ActionMasker(env, lambda e: e.action_masks())

    model = MaskablePPO.load(model_path)

    wins = losses = 0
    for _ in range(n_games):
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
        else:
            losses += 1

    env.close()
    print(f"Results over {n_games} games: {wins} wins / {losses} losses "
          f"({100*wins/n_games:.1f}% win rate from Player A perspective)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--deck", default="delver",
                        help="Deck the model plays (stem of .dk file, default: delver)")
    parser.add_argument("--opponent", required=False, default=None,
                        help="Opponent deck (stem of .dk file). Required for training.")
    parser.add_argument("--load", default=None, help="Resume from checkpoint .zip")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval", default=None, help="Self-play evaluation (note: always ~50%%)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--baseline", default=None, help="Evaluate model .zip vs scripted agent")
    parser.add_argument("--baseline-games", type=int, default=100)
    parser.add_argument("--observe", default=None, help="Watch model .zip play one game vs scripted agent")
    parser.add_argument("--diag", action="store_true", help="Run 10 quick games (random model vs scripted) to verify the env")
    parser.add_argument("--diag-games", type=int, default=10)
    parser.add_argument("--watch-scripted", action="store_true", help="Watch one game: scripted A vs scripted B")
    parser.add_argument("--tally", action="store_true", help="Print A/B win tally after each rollout")
    parser.add_argument("--self-play", action="store_true",
                        help="Train via self-play against frozen previous checkpoints")
    parser.add_argument("--scripted-fraction", type=float, default=0.0,
                        help="Fraction of envs that use the scripted opponent during self-play (default 0.0). "
                             "E.g. 0.3 gives ~2 scripted + 5 self-play with N_ENVS=7.")
    parser.add_argument("--record", action="store_true",
                        help="Record all game decisions to a .rmrec binary file in recordings/")
    parser.add_argument("--fixed-model", action="store_true",
                        help="Train --deck vs a fixed opponent model for --opponent. "
                             "Loads {deck}_{opponent}_final.zip and trains against "
                             "{opponent}_{deck}_final.zip (never reloaded).")
    parser.add_argument("--alternate", type=int, default=None, metavar="N",
                        help="Swap which side is trained every N timesteps. "
                             "Requires --deck and --opponent. Each round saves its "
                             "final checkpoint then trains the other side against it.")
    parser.add_argument("--train-all", action="store_true",
                        help="Train every deck×deck matchup sequentially (ignores --deck/--opponent)")
    parser.add_argument("--train-deck", type=str, default=None,
                        help="Train all matchups that include the given deck (ignores --deck/--opponent)")
    args = parser.parse_args()

    # Resolve model shorthands (e.g. 'delver_boomer-mav' → checkpoints/delver_boomer-mav_final.zip)
    args.load = _resolve_model(args.load)
    args.baseline = _resolve_model(args.baseline)
    args.observe = _resolve_model(args.observe)
    args.eval = _resolve_model(args.eval)

    if args.alternate is not None:
        if not args.opponent:
            parser.error("--alternate requires --opponent")
        train_alternate(args.binary, args.deck, args.opponent,
                        alternate_steps=args.alternate,
                        total_timesteps=args.total_timesteps,
                        tally=args.tally, record=args.record)
    elif args.fixed_model:
        if not args.opponent:
            parser.error("--fixed-model requires --opponent")
        train_fixed_model(args.binary, args.deck, args.opponent,
                          load_path=args.load,
                          total_timesteps=args.total_timesteps,
                          tally=args.tally, record=args.record)
    elif args.train_deck:
        all_decks = sorted(os.path.splitext(p)[0]
                           for p in os.listdir(_DECKS_DIR) if p.endswith(".dk"))
        target = args.train_deck
        if target not in all_decks:
            parser.error(f"Deck '{target}' not found in {_DECKS_DIR}. Available: {', '.join(all_decks)}")
        matchups = [(d, o) for d in all_decks for o in all_decks if d == target or o == target]
        print(f"Training {len(matchups)} matchups featuring '{target}' for {args.total_timesteps:,} timesteps each:")
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
                  scripted_fraction=args.scripted_fraction,
                  model_deck=d, opp_deck=o, record=args.record)
        print(f"\nAll {len(matchups)} matchups for '{target}' complete.")
    elif args.train_all:
        all_decks = sorted(os.path.splitext(p)[0]
                           for p in os.listdir(_DECKS_DIR) if p.endswith(".dk"))
        matchups = [(d, o) for d in all_decks for o in all_decks]
        print(f"Training all {len(matchups)} matchups for {args.total_timesteps:,} timesteps each:")
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
                  scripted_fraction=args.scripted_fraction,
                  model_deck=d, opp_deck=o, record=args.record)
        print(f"\nAll {len(matchups)} matchups complete.")
    elif args.diag:
        diag(args.binary, args.diag_games)
    elif args.watch_scripted:
        watch_scripted(args.binary)
    elif args.observe:
        observe(args.binary, args.observe)
    elif args.baseline:
        baseline(args.binary, args.baseline, args.baseline_games)
    elif args.eval:
        evaluate(args.binary, args.eval, args.eval_games)
    else:
        if args.opponent is None:
            parser.error("--opponent is required for training")
        train(args.binary, args.load, args.total_timesteps, tally=args.tally,
              self_play=args.self_play, scripted_fraction=args.scripted_fraction,
              model_deck=args.deck, opp_deck=args.opponent, record=args.record)
