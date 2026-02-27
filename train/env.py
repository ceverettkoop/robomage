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
1153-float state vector + 32 action-category floats + 70 hand cost floats
+ 140 battlefield ability cost floats = 1395 total.
See src/machine_io.h for the full state layout.

Reward
------
+1.0 for winning, -1.0 for losing (from Player A's perspective).
"""

import subprocess
import sys
import os
import re
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

try:
    from card_costs import _CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS
except ImportError:
    from train.card_costs import _CARD_COST_MATRIX, _CARD_ABILITY_COST_MATRIX, N_CARD_TYPES, _N_COST_FEATS

STATE_SIZE = 1153
MAX_ACTIONS = 32         # practical upper bound on num_choices per step
ACTION_CATEGORY_MAX = 18 # highest ActionCategory enum value (MANA_C)
MAX_HAND_SLOTS = 10
_HAND_COST_FEATS  = MAX_HAND_SLOTS * _N_COST_FEATS  # 10 * 7 = 70
_BF_ABILITY_FEATS = 20 * _N_COST_FEATS              # 20 * 7 = 140
OBS_SIZE = STATE_SIZE + MAX_ACTIONS + _HAND_COST_FEATS + _BF_ABILITY_FEATS  # 1395
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(_REPO_ROOT, "bin", "robomage")
BIN_DIR = os.path.join(_REPO_ROOT, "bin")  # game must be run from here for resource lookup


class RoboMageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Maximum decision steps per episode before truncation.  Prevents infinite
    # loops (e.g. a model that toggles attackers forever) from hanging training.
    MAX_STEPS = 1000

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
        self._step_count = 0

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
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

        self._step_count += 1
        if self._step_count >= self.MAX_STEPS:
            self._kill_proc()
            return np.zeros(OBS_SIZE, dtype=np.float32), 0.0, False, True, {}

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

                # Parse per-action category ints (one per legal action, after state)
                # Normalise by ACTION_CATEGORY_MAX so values are in [0.0, 1.0].
                cat_start = STATE_SIZE + 1
                cat_raw = parts[cat_start : cat_start + self._num_choices]
                cat_arr = np.zeros(MAX_ACTIONS, dtype=np.float32)
                for i, c in enumerate(cat_raw):
                    cat_arr[i] = int(c) / ACTION_CATEGORY_MAX

                # The -1 confirm convention only applies to mandatory attacker/blocker
                # choice queries.  Categories 2=SEL_ATK, 3=CONF_ATK, 4=SEL_BLK,
                # 5=CONF_BLK indicate a mandatory choice; the last action in those
                # queries must be sent as -1 to the game.
                _MANDATORY = {2, 3, 4, 5}
                self._pending_confirm = any(int(c) in _MANDATORY for c in cat_raw)

                # Parse state vector (first STATE_SIZE floats after the count)
                state_floats = parts[1 : STATE_SIZE + 1]
                state_arr = np.array(state_floats, dtype=np.float32)
                if len(state_arr) < STATE_SIZE:
                    state_arr = np.pad(state_arr, (0, STATE_SIZE - len(state_arr)))

                # Hand cast costs: matrix-multiply one-hots against cost matrix
                _HAND_START = 833
                hand_onehots = state_arr[_HAND_START:_HAND_START + MAX_HAND_SLOTS * N_CARD_TYPES]
                hand_costs = hand_onehots.reshape(MAX_HAND_SLOTS, N_CARD_TYPES) @ _CARD_COST_MATRIX  # (10,7)

                # Battlefield activated ability costs (all zeros until non-mana abilities are added)
                _BF_CARD_OFF = 8  # offset of card one-hot within each 40-float slot
                bf_ability_costs = np.zeros((20, _N_COST_FEATS), dtype=np.float32)
                for slot in range(20):
                    base = 33 + slot * 40 + _BF_CARD_OFF
                    bf_ability_costs[slot] = state_arr[base:base + N_CARD_TYPES] @ _CARD_ABILITY_COST_MATRIX

                self._obs = np.concatenate([state_arr, cat_arr,
                                            hand_costs.flatten(),
                                            bf_ability_costs.flatten()])
                break

            # Non-QUERY output: optionally print for human render mode
            if self.render_mode == "human":
                print(line, file=sys.stderr)

        info = {"reward": reward, "done": done}
        if done:
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


# ── Action category constants (mirror ActionCategory enum in classes/action.h) ─
_CAT_PASS       = 0
_CAT_MANA       = 1   # legacy, no longer emitted by the game
_CAT_SEL_ATK    = 2
_CAT_CONF_ATK   = 3
_CAT_SEL_BLK    = 4
_CAT_CONF_BLK   = 5
_CAT_CAST       = 7
_CAT_TARGET     = 8
_CAT_LAND       = 9
_CAT_MULLIGAN   = 11
_CAT_MANA_W     = 13
_CAT_MANA_U     = 14
_CAT_MANA_B     = 15
_CAT_MANA_R     = 16
_CAT_MANA_G     = 17
_CAT_MANA_C     = 18

# All mana-producing categories
_MANA_CATS = {_CAT_MANA_W, _CAT_MANA_U, _CAT_MANA_B, _CAT_MANA_R, _CAT_MANA_G, _CAT_MANA_C}

# Map from mana pool color index (W=0,U=1,B=2,R=3,G=4,C=5) to action category
_COLOR_TO_MANA_CAT = [_CAT_MANA_W, _CAT_MANA_U, _CAT_MANA_B, _CAT_MANA_R, _CAT_MANA_G, _CAT_MANA_C]

# Colored mana requirements per card vocab index (card_vocab.h).
# Keys are color pool indices: W=0, U=1, B=2, R=3, G=4, C=5.
# Generic mana is omitted — any color satisfies it.
_CARD_COLORED_COSTS = {
    2: {3: 1},  # Lightning Bolt (1R): needs 1 red
    3: {4: 1},  # Grizzly Bears (1G): needs 1 green
}

# ── Battlefield layout (mirror machine_io.h) ────────────────────────────────
_BF_START         = 33
_BF_SLOT_SIZE     = 40   # 8 status floats + 32 card one-hot
_BF_A_SLOTS       = 10   # Player A occupies slots 0-9
# Status offsets within a slot
_OFF_POWER        = 0
_OFF_TOUGHNESS    = 1
_OFF_IS_TAPPED    = 2
_OFF_IS_ATTACKING = 3
_OFF_HAS_SICKNESS = 5


def _all_eligible_creatures_attacking(obs: np.ndarray, slot_start: int) -> bool:
    """Return True if every untapped, non-sick creature in the given 10-slot range is attacking."""
    any_eligible = False
    for slot in range(_BF_A_SLOTS):
        base = _BF_START + (slot_start + slot) * _BF_SLOT_SIZE
        if obs[base + _OFF_POWER] <= 0.0 and obs[base + _OFF_TOUGHNESS] <= 0.0:
            continue  # empty slot
        if obs[base + _OFF_IS_TAPPED] > 0.5 or obs[base + _OFF_HAS_SICKNESS] > 0.5:
            continue  # can't attack
        any_eligible = True
        if obs[base + _OFF_IS_ATTACKING] <= 0.5:
            return False  # eligible but not yet attacking
    return any_eligible


def scripted_action(obs: np.ndarray, num_choices: int) -> int:
    """
    Rule-based Player B:
      - Never blocks (confirms immediately when blockers are requested)
      - Attacks with every eligible creature
      - Plays the first available land
      - Taps mana during main phases, preferring the color the hand needs most
      - Casts every spell (Grizzly Bears, Lightning Bolt aimed at Player A)
      - Passes priority otherwise

    Action categories are stored in obs[STATE_SIZE:] normalised by ACTION_CATEGORY_MAX.
    """
    cats = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)

    # 0. Mulligan: always keep — return the first non-mulligan action (the keep action)
    if any(c == _CAT_MULLIGAN for c in cats):
        for i, c in enumerate(cats):
            if c != _CAT_MULLIGAN:
                return i

    # 1. Confirm blockers immediately — never block
    for i, c in enumerate(cats):
        if c == _CAT_CONF_BLK:
            return i

    # 2. Attacker selection: select until all eligible creatures are attacking, then confirm.
    #    The game re-offers already-attacking creatures as SEL_ATK (for deselection), so we
    #    must check the battlefield state rather than blindly picking SEL_ATK every time.
    if any(c == _CAT_SEL_ATK for c in cats):
        # Player A occupies slots 0-9, Player B occupies slots 10-19
        slot_start = 0 if obs[31] > 0.5 else _BF_A_SLOTS
        if _all_eligible_creatures_attacking(obs, slot_start):
            for i, c in enumerate(cats):
                if c == _CAT_CONF_ATK:
                    return i
        else:
            for i, c in enumerate(cats):
                if c == _CAT_SEL_ATK:
                    return i

    # 3. Confirm attack declaration (fallback, e.g. no eligible attackers)
    for i, c in enumerate(cats):
        if c == _CAT_CONF_ATK:
            return i

    # 4. Select target — action 0 targets Player A (the opponent)
    for i, c in enumerate(cats):
        if c == _CAT_TARGET:
            return i

    # 5. Cast any spell immediately if affordable
    for i, c in enumerate(cats):
        if c == _CAT_CAST:
            return i

    # 6. Play land (may unlock mana for a spell next query)
    for i, c in enumerate(cats):
        if c == _CAT_LAND:
            return i

    # 7. Tap mana during main phases only, choosing the color the hand needs most
    _STEP_ONE_HOT_START = 18
    _STEP_FIRST_MAIN    = _STEP_ONE_HOT_START + 3   # obs[21]
    _STEP_SECOND_MAIN   = _STEP_ONE_HOT_START + 9   # obs[27]
    in_main_phase = obs[_STEP_FIRST_MAIN] > 0.5 or obs[_STEP_SECOND_MAIN] > 0.5
    if in_main_phase and any(c in _MANA_CATS for c in cats):
        # Determine what colored mana is still needed for cards in hand
        pool_start = 3 if obs[31] > 0.5 else 12
        pool = [int(round(obs[pool_start + i] * 10)) for i in range(6)]
        needed = [0] * 6
        for slot in range(10):
            base = 833 + slot * 32
            slot_vec = obs[base:base + 32]
            card_idx = int(np.argmax(slot_vec))
            if slot_vec[card_idx] < 0.5:
                continue  # empty slot
            for color_idx, count in _CARD_COLORED_COSTS.get(card_idx, {}).items():
                needed[color_idx] = max(needed[color_idx], count)
        short = [max(0, needed[i] - pool[i]) for i in range(6)]

        # Prefer the color we're shortest on
        for color_idx in sorted(range(6), key=lambda c: -short[c]):
            if short[color_idx] <= 0:
                break
            target_cat = _COLOR_TO_MANA_CAT[color_idx]
            for i, c in enumerate(cats):
                if c == target_cat:
                    return i

        # Fallback: tap any available mana source
        for i, c in enumerate(cats):
            if c in _MANA_CATS:
                return i

    # Default: pass priority
    return 0


class ModelVsScriptedEnv(gym.Env):
    """Wraps RoboMageEnv so the model always plays as Player A.

    Player B is controlled by the scripted_action rule-based agent instead of
    a random agent, providing a more meaningful training opponent.
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
        """Resolve consecutive Player B turns with the scripted agent."""
        while not (terminated or truncated) and obs[31] <= 0.5:
            action = scripted_action(obs, self._env._num_choices)
            obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()
