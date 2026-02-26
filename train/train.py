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
import random as _random

from env import RoboMageEnv, ModelVsScriptedEnv, scripted_action, OBS_SIZE, STATE_SIZE, MAX_ACTIONS, ACTION_CATEGORY_MAX, BINARY
from extractor import CardGameExtractor

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


class WinTallyCallback(BaseCallback):
    """Prints a running tally of Player A vs Player B wins after each rollout."""

    def __init__(self):
        super().__init__()
        self.a_wins = 0
        self.b_wins = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            r = info["episode"]["r"]
            if r > 0:
                self.a_wins += 1
            elif r < 0:
                self.b_wins += 1
        return True

    def _on_rollout_end(self) -> None:
        total = self.a_wins + self.b_wins
        if total == 0:
            return
        pct = 100.0 * self.a_wins / total
        print(f"[tally] A wins: {self.a_wins}  B wins: {self.b_wins}  "
              f"total: {total}  A win rate: {pct:.1f}%")


CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 7  # parallel game processes


def make_env(binary_path: str, rank: int):
    def _init():
        env = ModelVsScriptedEnv(binary_path=binary_path)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def train(binary_path: str, load_path: str | None = None, total_timesteps: int = TOTAL_TIMESTEPS, tally: bool = False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Parallel environments for faster data collection
    vec_env = SubprocVecEnv([make_env(binary_path, i) for i in range(N_ENVS)])

    policy_kwargs = dict(
        features_extractor_class=CardGameExtractor,
        net_arch=[256, 256],
    )

    if load_path:
        print(f"Resuming from {load_path}")
        model = MaskablePPO.load(load_path, env=vec_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=512,           # steps per env per update
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,         # encourage exploration early on
            verbose=1,
            tensorboard_log=LOG_DIR,
        )

    callbacks = [
        CheckpointCallback(
            save_freq=25_000 // N_ENVS,
            save_path=CHECKPOINT_DIR,
            name_prefix="robomage",
        ),
    ]
    if tally:
        callbacks.append(WinTallyCallback())

    print(f"Training for {total_timesteps:,} timesteps across {N_ENVS} envs...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=load_path is None)
    model.save(os.path.join(CHECKPOINT_DIR, "robomage_final"))
    print("Saved final model.")

    vec_env.close()


def _play_model_vs_random(binary_path: str, model, verbose: bool = False) -> float:
    """Play one game: model controls Player A, random agent controls Player B.
    Returns +1.0 if model wins, -1.0 if random wins.
    obs[31] is 1.0 when Player A has priority, 0.0 when Player B does.
    """
    env = RoboMageEnv(binary_path=binary_path, render_mode="human" if verbose else None)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        a_has_priority = obs[31] > 0.5
        num_choices = env._num_choices

        if a_has_priority:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)
            if verbose:
                print(f"  [Model/A]  action {action} of {num_choices}")
        else:
            action = _random.randint(0, num_choices - 1)
            if verbose:
                print(f"  [Random/B] action {action} of {num_choices}")

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    env.close()
    return total_reward


def baseline(binary_path: str, model_path: str, n_games: int = 100):
    """Evaluate win rate of the model against a random opponent."""
    model = MaskablePPO.load(model_path)
    wins = losses = draws = 0

    for i in range(n_games):
        result = _play_model_vs_random(binary_path, model)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1
        print(f"\rGame {i+1}/{n_games}  W:{wins} L:{losses} D:{draws}", end="", flush=True)

    print()
    print(f"vs random over {n_games} games: {wins}W / {losses}L / {draws}D "
          f"({100 * wins / n_games:.1f}% win rate)")


def observe(binary_path: str, model_path: str):
    """Watch the model play one game against a random opponent."""
    model = MaskablePPO.load(model_path)
    print("=== Model (Player A) vs Random (Player B) ===\n")
    result = _play_model_vs_random(binary_path, model, verbose=True)
    print()
    if result > 0:
        print("=== Model (A) wins ===")
    elif result < 0:
        print("=== Random (B) wins ===")
    else:
        print("=== Draw ===")


_CAT_NAMES = {
    0: "PASS", 1: "MANA", 2: "SEL_ATK", 3: "CONF_ATK",
    4: "SEL_BLK", 5: "CONF_BLK", 6: "ACTIVATE", 7: "CAST",
    8: "TARGET", 9: "LAND", 10: "OTHER", 11: "MULLIGAN", 12: "BOTTOM_CARD"
}


def watch_scripted(binary_path: str):
    """Run one game with both players driven by the scripted agent and print every decision."""
    import numpy as np
    import sys

    env = RoboMageEnv(binary_path=binary_path, render_mode="human")
    obs, _ = env.reset()
    done = False
    step = 0

    print("=== Scripted (A) vs Scripted (B) ===\n", flush=True)

    while not done:
        num_choices = env._num_choices
        player = "A" if obs[31] > 0.5 else "B"
        action = scripted_action(obs, num_choices)

        cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
        chosen_cat = _CAT_NAMES.get(int(cats[action]), str(cats[action]))
        all_cats = [_CAT_NAMES.get(int(c), str(c)) for c in cats]

        print(f"[{step:4d}] P{player}  choices={num_choices}  available={all_cats}  -> {action} ({chosen_cat})",
              flush=True)
        sys.stderr.flush()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

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
    parser.add_argument("--load", default=None, help="Resume from checkpoint .zip")
    parser.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS)
    parser.add_argument("--eval", default=None, help="Self-play evaluation (note: always ~50%%)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--baseline", default=None, help="Evaluate model .zip vs random agent")
    parser.add_argument("--baseline-games", type=int, default=100)
    parser.add_argument("--observe", default=None, help="Watch model .zip play one game vs random")
    parser.add_argument("--watch-scripted", action="store_true", help="Watch one game: scripted A vs scripted B")
    parser.add_argument("--tally", action="store_true", help="Print A/B win tally after each rollout")
    args = parser.parse_args()

    if args.watch_scripted:
        watch_scripted(args.binary)
    elif args.observe:
        observe(args.binary, args.observe)
    elif args.baseline:
        baseline(args.binary, args.baseline, args.baseline_games)
    elif args.eval:
        evaluate(args.binary, args.eval, args.eval_games)
    else:
        train(args.binary, args.load, args.total_timesteps, tally=args.tally)
