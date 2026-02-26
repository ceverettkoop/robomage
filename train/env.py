"""
RoboMage gymnasium environment.

The game runs as a subprocess with --machine mode. On each decision point it
emits a QUERY line to stdout:

    QUERY: <num_choices> <f0> <f1> ... <f1152>

The environment sends back a single integer on stdin.

Action convention
-----------------
For top-level priority actions (main.cpp):
    action index 0..N-1 maps directly to legal_actions[i].

For mandatory-choice loops (declare attackers / blockers):
    The game uses -1 as "confirm/done". In the gym action space we reserve
    action index = num_choices - 1 as the confirm slot, which the environment
    remaps to -1 before sending to the game.

    e.g. num_choices=3 means actions 0,1 are creatures and action 2 = confirm.

Observation space
-----------------
193-float vector. See src/machine_io.h for the full layout.

Reward
------
+1.0 for winning, -1.0 for losing (from Player A's perspective).
"""

import subprocess
import sys
import os
import re
import random as _random
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

STATE_SIZE = 1153
MAX_ACTIONS = 32         # practical upper bound on num_choices per step
ACTION_CATEGORY_MAX = 10 # highest ActionCategory enum value (OTHER_CHOICE)
OBS_SIZE = STATE_SIZE + MAX_ACTIONS  # 1185: state + per-action category features
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(_REPO_ROOT, "bin", "robomage")
BIN_DIR = os.path.join(_REPO_ROOT, "bin")  # game must be run from here for resource lookup


class RoboMageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, binary_path: str = BINARY, render_mode=None):
        super().__init__()
        self.binary_path = os.path.realpath(binary_path)
        self.render_mode = render_mode

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        # Discrete action space sized to the max we'd ever see.
        # Invalid actions are masked at each step via `action_masks()`.
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._proc = None
        self._num_choices = 1
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._pending_confirm = False  # True when last query used the -1 convention

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._kill_proc()
        self._proc = subprocess.Popen(
            [self.binary_path, "--machine"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # suppress game chatter
            text=True,
            bufsize=1,
            cwd=BIN_DIR,  # game uses getcwd() to locate resources/
        )
        obs, info = self._read_until_query()
        return obs, info

    def step(self, action: int):
        assert self._proc is not None, "Call reset() first"

        # Remap: if the last query included a confirm slot (num_choices had +1),
        # and the agent chose the last index, send -1 to the game.
        game_action = action
        if self._pending_confirm and action == self._num_choices - 1:
            game_action = -1

        self._send(game_action)

        obs, info = self._read_until_query()
        reward = info.get("reward", 0.0)
        terminated = info.get("done", False)
        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid actions for the current step (for MaskablePPO)."""
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[: self._num_choices] = True
        return mask

    def close(self):
        self._kill_proc()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send(self, action: int):
        self._proc.stdin.write(f"{action}\n")
        self._proc.stdin.flush()

    def _read_until_query(self):
        """
        Read lines from the game process until a QUERY line appears or the
        process exits (game over).  Returns (obs, info).
        """
        reward = 0.0
        done = False

        while True:
            line = self._proc.stdout.readline()

            if not line:
                # Process ended — game over
                done = True
                break

            line = line.rstrip("\n")

            # Detect win/loss
            if "Player A wins" in line:
                reward = 1.0
                done = True
            elif "Player B wins" in line:
                reward = -1.0
                done = True

            if line.startswith("QUERY: "):
                parts = line[7:].split()
                self._num_choices = min(int(parts[0]), MAX_ACTIONS)
                # The +1 confirm convention: if num_choices from game is N+1
                # and the last slot represents -1, we set pending_confirm.
                # We detect this heuristically: any mandatory-choice query
                # with num_choices > 1 may use -1. For simplicity we always
                # allow confirm as the last action — the game will just ignore
                # invalid inputs from the main loop.
                self._pending_confirm = True

                # Parse state vector (first STATE_SIZE floats after the count)
                state_floats = parts[1 : STATE_SIZE + 1]
                state_arr = np.array(state_floats, dtype=np.float32)
                if len(state_arr) < STATE_SIZE:
                    state_arr = np.pad(state_arr, (0, STATE_SIZE - len(state_arr)))

                # Parse per-action category ints (one per legal action, after state)
                # Normalise by ACTION_CATEGORY_MAX so values are in [0.0, 1.0].
                cat_start = STATE_SIZE + 1
                cat_raw = parts[cat_start : cat_start + self._num_choices]
                cat_arr = np.zeros(MAX_ACTIONS, dtype=np.float32)
                for i, c in enumerate(cat_raw):
                    cat_arr[i] = int(c) / ACTION_CATEGORY_MAX

                self._obs = np.concatenate([state_arr, cat_arr])
                break

            # Non-QUERY output: optionally print for human render mode
            if self.render_mode == "human":
                print(line, file=sys.stderr)

        info = {"reward": reward, "done": done}
        if done:
            # Drain any remaining output so the process can exit cleanly
            self._proc.stdout.read()
            self._kill_proc()
            # Return a zero obs on terminal step — will be replaced by reset()
            return np.zeros(OBS_SIZE, dtype=np.float32), info

        return self._obs.copy(), info

    def _kill_proc(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.stdout.close()
                self._proc.kill()
                self._proc.wait()
            except Exception:
                pass
            self._proc = None


class ModelVsRandomEnv(gym.Env):
    """Wraps RoboMageEnv so the model always plays as Player A.

    All Player B decision points are resolved automatically with uniformly
    random legal actions before the step is returned to the training loop.
    The model therefore only ever sees and acts on Player A's turns.

    obs[31] == 1.0 means Player A has priority (guaranteed on every step
    returned to the caller).  obs[31] == 0.0 would indicate Player B, but
    that never surfaces here.
    """

    def __init__(self, binary_path: str = BINARY, render_mode=None):
        super().__init__()
        self._env = RoboMageEnv(binary_path=binary_path, render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = render_mode

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        obs, _reward, terminated, truncated, info = self._skip_b_turns(
            obs, 0.0, False, False, info
        )
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self._skip_b_turns(
                obs, reward, terminated, truncated, info
            )
        return obs, reward, terminated, truncated, info

    def _skip_b_turns(self, obs, reward, terminated, truncated, info):
        """Resolve consecutive Player B turns with random actions."""
        while not (terminated or truncated) and obs[31] <= 0.5:
            random_action = _random.randint(0, self._env._num_choices - 1)
            obs, reward, terminated, truncated, info = self._env.step(random_action)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()
