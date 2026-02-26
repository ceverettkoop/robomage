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

from env import RoboMageEnv, ModelVsRandomEnv, OBS_SIZE, MAX_ACTIONS, BINARY

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
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor


CHECKPOINT_DIR = "checkpoints"
LOG_DIR = "logs"
TOTAL_TIMESTEPS = 1_000_000
N_ENVS = 4  # parallel game processes


def make_env(binary_path: str, rank: int):
    def _init():
        env = ModelVsRandomEnv(binary_path=binary_path)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
        env = Monitor(env)
        return env
    return _init


def train(binary_path: str, load_path: str | None = None):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # Parallel environments for faster data collection
    vec_env = SubprocVecEnv([make_env(binary_path, i) for i in range(N_ENVS)])

    policy_kwargs = dict(
        net_arch=[256, 256],  # two hidden layers, 256 units each
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
            save_freq=25_000,
            save_path=CHECKPOINT_DIR,
            name_prefix="robomage",
        ),
    ]

    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} envs...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, reset_num_timesteps=load_path is None)
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
    parser.add_argument("--eval", default=None, help="Self-play evaluation (note: always ~50%)")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--baseline", default=None, help="Evaluate model .zip vs random agent")
    parser.add_argument("--baseline-games", type=int, default=100)
    parser.add_argument("--observe", default=None, help="Watch model .zip play one game vs random")
    args = parser.parse_args()

    if args.observe:
        observe(args.binary, args.observe)
    elif args.baseline:
        baseline(args.binary, args.baseline, args.baseline_games)
    elif args.eval:
        evaluate(args.binary, args.eval, args.eval_games)
    else:
        train(args.binary, args.load)
