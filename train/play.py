"""
Play interactively against a trained RoboMage model.

The model plays as Player A. You play as Player B.

Usage:
    train/.venv/bin/python train/play.py --model checkpoints/robomage_final.zip
"""

import argparse
import sys
import numpy as np

from env import RoboMageEnv, STATE_SIZE, ACTION_CATEGORY_MAX, BINARY

try:
    from sb3_contrib import MaskablePPO
    USE_MASKABLE = True
except ImportError:
    from stable_baselines3 import PPO as MaskablePPO
    USE_MASKABLE = False

_CAT_NAMES = {
    0: "pass priority",
    1: "tap for mana",
    2: "declare attacker",
    3: "confirm attackers",
    4: "declare blocker",
    5: "confirm blockers",
    6: "activate ability",
    7: "cast spell",
    8: "select target",
    9: "play land",
    10: "choose",
    11: "mulligan",
    12: "bottom card"
}


def play(binary_path: str, model_path: str):
    model = MaskablePPO.load(model_path)
    # render_mode="human" prints game narrative lines to stderr
    env = RoboMageEnv(binary_path=binary_path, render_mode="human")
    obs, _ = env.reset()
    done = False

    print("=== Model (Player A) vs You (Player B) ===\n", flush=True)

    while not done:
        num_choices = env._num_choices
        a_has_priority = obs[31] > 0.5
        cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)

        if a_has_priority:
            masks = env.action_masks() if USE_MASKABLE else None
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            action = int(action)
            cat_name = _CAT_NAMES.get(int(cats[action]) if action < len(cats) else -1, "?")
            print(f"[Model/A] {cat_name}", flush=True)
        else:
            print(f"\nYour turn — {num_choices} option(s):", flush=True)
            for i, c in enumerate(cats):
                print(f"  {i}: {_CAT_NAMES.get(int(c), str(c))}")
            while True:
                try:
                    raw = input("Choose> ").strip()
                    action = int(raw)
                    if 0 <= action < num_choices:
                        break
                    print(f"Enter a number between 0 and {num_choices - 1}.")
                except (ValueError, EOFError):
                    print("Enter a number.")

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()
    print()
    if reward > 0:
        print("=== Model (A) wins! ===")
    elif reward < 0:
        print("=== You (B) win! ===")
    else:
        print("=== Draw ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default=BINARY)
    parser.add_argument("--model", required=True, help="Path to trained model .zip")
    args = parser.parse_args()
    play(args.binary, args.model)
