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

from env import RoboMageEnv, STATE_SIZE, MAX_ACTIONS

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
        env = RoboMageEnv(binary_path=binary_path)
        env = Monitor(env)
        if USE_MASKABLE:
            env = ActionMasker(env, lambda e: e.action_masks())
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
    parser.add_argument("--binary", default=os.path.join("..", "bin", "robomage"))
    parser.add_argument("--load", default=None, help="Resume from checkpoint .zip")
    parser.add_argument("--eval", default=None, help="Evaluate a saved model .zip")
    parser.add_argument("--eval-games", type=int, default=100)
    args = parser.parse_args()

    if args.eval:
        evaluate(args.binary, args.eval, args.eval_games)
    else:
        train(args.binary, args.load)
