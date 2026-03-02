"""
RoboMage gymnasium environment.

The game runs as a subprocess with --machine mode. On each decision point it
emits a QUERY line to stdout:

    QUERY: <num_choices> <f0> ... <f8684> <cat0>...<catN-1> <id0>...<idN-1> <ctrl0>...<ctrlN-1>

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
State is always emitted from the PRIORITY PLAYER'S perspective ("self").

8685-float state vector + 32 action-category floats + 32 action card-ID floats
+ 32 action controller_is_self floats + 70 hand cost floats
+ 336 battlefield ability cost floats = 9187 total.
See src/machine_io.h for the full state layout.

Reward
------
+1.0 for winning, -1.0 for losing (from Player A's perspective).
"""

import glob as _glob
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

STATE_SIZE = 8685
MAX_ACTIONS = 32         # practical upper bound on num_choices per step
ACTION_CATEGORY_MAX = 21 # highest ActionCategory enum value (SHUFFLE)
_ACTION_CARD_ID_NULL = -1.0 / N_CARD_TYPES  # -0.03125 — null sentinel for non-card slots
_ACTION_CTRL_NULL    = -1.0 / N_CARD_TYPES  # -0.03125 — null sentinel for non-entity actions
MAX_HAND_SLOTS = 10
_HAND_COST_FEATS  = MAX_HAND_SLOTS * _N_COST_FEATS  # 10 * 7 = 70
_BF_ABILITY_FEATS = 48 * _N_COST_FEATS              # 48 * 7 = 336
OBS_SIZE = STATE_SIZE + 3 * MAX_ACTIONS + _HAND_COST_FEATS + _BF_ABILITY_FEATS  # 9187

# ── State layout offsets (mirror src/machine_io.h) ───────────────────────────
_LAND_START   = 33 + 48 * 40   # 1953: land slots  (48 × 40)
_STACK_START  = 1953 + 48 * 40  # 3873: stack slots (12 × 33)
_GY_START     = 3873 + 12 * 33  # 4269: graveyard   (128 × 32)
_HAND_START   = 4269 + 128 * 32 # 8365: hand        (10 × 32)
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
        shaping_a = 0.0
        shaping_b = 0.0

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

            # Shaping signal: mana wasted at end of phase (pool non-empty on drain)
            if line.startswith("MANA_WASTED: "):
                side = line[13:].strip()
                if side == "A":
                    shaping_a -= 0.1
                elif side == "B":
                    shaping_b -= 0.1
                continue

            # Shaping signal: excessive mulligan (3rd and beyond = -0.1 each)
            if line.startswith("MULLIGAN_PENALTY: "):
                side = line[18:].strip()
                if side == "A":
                    shaping_a -= 0.1
                elif side == "B":
                    shaping_b -= 0.1
                continue

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

                # Parse per-action card ID floats (after category ints)
                id_start = cat_start + self._num_choices
                id_raw = parts[id_start : id_start + self._num_choices]
                card_id_arr = np.full(MAX_ACTIONS, _ACTION_CARD_ID_NULL, dtype=np.float32)
                for i, v in enumerate(id_raw):
                    card_id_arr[i] = float(v)

                # Parse per-action controller_is_self floats (after card ID floats)
                ctrl_start = id_start + self._num_choices
                ctrl_raw = parts[ctrl_start : ctrl_start + self._num_choices]
                ctrl_arr = np.full(MAX_ACTIONS, _ACTION_CTRL_NULL, dtype=np.float32)
                for i, v in enumerate(ctrl_raw):
                    ctrl_arr[i] = float(v)

                # Parse state vector (first STATE_SIZE floats after the count)
                state_floats = parts[1 : STATE_SIZE + 1]
                state_arr = np.array(state_floats, dtype=np.float32)
                if len(state_arr) < STATE_SIZE:
                    state_arr = np.pad(state_arr, (0, STATE_SIZE - len(state_arr)))

                # Hand cast costs: matrix-multiply one-hots against cost matrix
                hand_onehots = state_arr[_HAND_START:_HAND_START + MAX_HAND_SLOTS * N_CARD_TYPES]
                hand_costs = hand_onehots.reshape(MAX_HAND_SLOTS, N_CARD_TYPES) @ _CARD_COST_MATRIX  # (10,7)

                # Battlefield activated ability costs (all 48 creature slots; 40-float format)
                bf_ability_costs = np.zeros((48, _N_COST_FEATS), dtype=np.float32)
                for slot in range(48):
                    base = 33 + slot * 40 + _BF_CARD_OFF
                    bf_ability_costs[slot] = state_arr[base:base + N_CARD_TYPES] @ _CARD_ABILITY_COST_MATRIX

                self._obs = np.concatenate([state_arr, cat_arr, card_id_arr, ctrl_arr,
                                            hand_costs.flatten(),
                                            bf_ability_costs.flatten()])
                break

            # Non-QUERY output: optionally print for human render mode
            if self.render_mode == "human":
                self._print_narrative_line(line)

        info = {"reward": reward, "done": done, "shaping_a": shaping_a, "shaping_b": shaping_b}
        if done:
            self._kill_proc()
            # Return a zero obs on terminal step — will be replaced by reset()
            return np.zeros(OBS_SIZE, dtype=np.float32), info

        return self._obs.copy(), info

    def _print_narrative_line(self, line: str):
        print(line, file=sys.stderr)

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


class NarrativeEnv(RoboMageEnv):
    """RoboMageEnv that collects non-QUERY game lines into a list instead of printing them."""

    def __init__(self, **kwargs):
        kwargs.setdefault("render_mode", "human")
        super().__init__(**kwargs)
        self.lines: list = []

    def _print_narrative_line(self, line: str):
        self.lines.append(line)

    def flush_lines(self) -> list:
        out, self.lines = self.lines, []
        return out


# ── Action category constants (mirror ActionCategory enum in classes/action.h) ─
_CAT_PASS       = 0
_CAT_MANA       = 1   # legacy, no longer emitted by the game
_CAT_SEL_ATK    = 2
_CAT_CONF_ATK   = 3
_CAT_SEL_BLK    = 4
_CAT_CONF_BLK   = 5
_CAT_ACTIVATE   = 6   # activate a non-mana ability (fetch lands, Wasteland destroy)
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
_CAT_SEARCH     = 19  # search library (action 0 = fail to find, 1+ = actual cards)

# All mana-producing categories
_MANA_CATS = {_CAT_MANA_W, _CAT_MANA_U, _CAT_MANA_B, _CAT_MANA_R, _CAT_MANA_G, _CAT_MANA_C}

# Map from mana pool color index (W=0,U=1,B=2,R=3,G=4,C=5) to action category
_COLOR_TO_MANA_CAT = [_CAT_MANA_W, _CAT_MANA_U, _CAT_MANA_B, _CAT_MANA_R, _CAT_MANA_G, _CAT_MANA_C]

# Colored mana requirements per card vocab index (card_vocab.h).
# Keys are color pool indices: W=0, U=1, B=2, R=3, G=4, C=5.
# Generic mana is omitted — any color satisfies it.
# Deck: test_minimal.dk (blue/red, fetch-heavy)
_CARD_COLORED_COSTS = {
    2: {3: 1}, # bolt
    11: {1: 1},   # Ponder        (U)     — 1 blue
    13: {1: 1},   # Daze          (1U)    — 1 blue
    16: {1: 1},   # Delver of Secrets (U) — 1 blue
    18: {1: 1},   # Flying Men    (U)     — 1 blue
    20: {3: 1},   # Dragon's Rage Channeler (R) — 1 red
    21: {1: 2},   # Air Elemental (3UU)   — 2 blue
    22: {1: 2},   # Counterspell  (UU)    — 2 blue
    23: {3: 1},   # Lightning Strike (1R) — 1 red
    24: {1: 1},   # Brainstorm      (U)   — 1 blue
}

# ── Battlefield layout (mirror machine_io.h) ────────────────────────────────
_BF_START         = 33
_BF_SLOT_SIZE     = 40   # 8 status floats + 32 card one-hot
_BF_A_SLOTS       = 24   # self occupies slots 0-23
_BF_CARD_OFF      = 8    # offset of card one-hot within each 40-float permanent slot
_CTRL_OFF         = 7    # offset of controller_is_self within a permanent slot
_STACK_SLOT_SIZE  = 33   # controller_is_self(1) + card one-hot(32)
_GY_SLOT_SIZE     = N_CARD_TYPES  # 32 — graveyard slots are just card one-hots
_GY_A_SLOTS       = 64   # self occupies GY slots 0-63
# Start of bf_ability_costs block in the full obs vector
_BF_COST_START    = STATE_SIZE + 3 * MAX_ACTIONS + _HAND_COST_FEATS  # 8851
# Status offsets within a slot
_OFF_POWER        = 0
_OFF_TOUGHNESS    = 1
_OFF_IS_TAPPED    = 2
_OFF_IS_ATTACKING = 3
_OFF_HAS_SICKNESS = 5

# Vocab indices used for targeting decisions (mirror src/card_vocab.h)
_WASTELAND_VOCAB_IDX     = 10
_BASIC_LAND_IDS          = frozenset({0, 19})  # Mountain(0), Island(19)
_COUNTER_SPELL_VOCAB_IDS = frozenset({12, 13, 22})  # Force of Will(12), Daze(13), Counterspell(22)
_COUNTERSPELL_VOCAB_IDX  = 22
_BLUE_POOL_IDX           = 4   # obs[3 + 1]; mana pool is at obs[3:9], W/U/B/R/G/C, /10


def _all_eligible_creatures_attacking(obs: np.ndarray) -> bool:
    """Return True if every untapped, non-sick creature in self's slots (0-23) is attacking."""
    any_eligible = False
    for slot in range(_BF_A_SLOTS):
        base = _BF_START + slot * _BF_SLOT_SIZE
        if obs[base + _OFF_POWER] <= 0.0 and obs[base + _OFF_TOUGHNESS] <= 0.0:
            continue  # empty slot
        if obs[base + _OFF_IS_TAPPED] > 0.5 or obs[base + _OFF_HAS_SICKNESS] > 0.5:
            continue  # can't attack
        any_eligible = True
        if obs[base + _OFF_IS_ATTACKING] <= 0.5:
            return False  # eligible but not yet attacking
    return any_eligible


def _opponent_has_nonbasic_land(obs: np.ndarray) -> bool:
    """Return True if opponent has at least one nonbasic land (land slots 24-47)."""
    for slot in range(24, 48):
        base = _LAND_START + slot * _BF_SLOT_SIZE
        card_vec = obs[base + _BF_CARD_OFF : base + _BF_CARD_OFF + N_CARD_TYPES]
        idx = int(np.argmax(card_vec))
        if card_vec[idx] > 0.5 and idx not in _BASIC_LAND_IDS:
            return True
    return False


def _opponent_has_spell_on_stack(obs: np.ndarray) -> bool:
    """Return True if at least one spell/ability on the stack is not controlled by self."""
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        ctrl_is_self = obs[base]
        card_vec = obs[base + 1 : base + 1 + N_CARD_TYPES]
        if np.max(card_vec) > 0.5 and ctrl_is_self < 0.5:
            return True
    return False


def scripted_action(obs: np.ndarray, num_choices: int) -> int:
    """
    Rule-based agent for test_minimal.dk (blue/red fetch-land deck).
    Works correctly for either Player A or Player B because the observation
    is always emitted from the priority player's perspective.

      - Never blocks (confirms immediately)
      - Attacks with every eligible creature each combat
      - Selects target 0 (opponent player or first offered spell/permanent)
      - Searches library: always finds the first offered card (never fails to find)
      - Casts every spell the moment it becomes affordable
      - Plays the first available land
      - Activates non-mana abilities (fetch lands, Wasteland destroy) during main phase
      - Taps mana during main phases, preferring the color the hand needs most
        (blue for Flying Men, Delver, Ponder, Daze, Counterspell, Air Elemental;
         red for Dragon's Rage Channeler, Lightning Strike)
      - Passes priority otherwise

    Action categories are stored in obs[STATE_SIZE:] normalised by ACTION_CATEGORY_MAX.
    """
    cats     = np.round(obs[STATE_SIZE:STATE_SIZE + num_choices] * ACTION_CATEGORY_MAX).astype(int)
    card_ids = obs[STATE_SIZE + MAX_ACTIONS     : STATE_SIZE + 2 * MAX_ACTIONS]
    ctrl_arr = obs[STATE_SIZE + 2 * MAX_ACTIONS : STATE_SIZE + 3 * MAX_ACTIONS]

    _STEP_FIRST_MAIN  = 21   # obs[18 + 3]
    _STEP_SECOND_MAIN = 27   # obs[18 + 9]
    in_main_phase = obs[_STEP_FIRST_MAIN] > 0.5 or obs[_STEP_SECOND_MAIN] > 0.5

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
        if _all_eligible_creatures_attacking(obs):
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

    # 4. Select target — prefer non-self-controlled targets.
    #    ctrl_arr[i] == 1.0 means self-controlled; 0.0 = opponent permanent/spell;
    #    _ACTION_CTRL_NULL (-0.03125) = player target (also non-self).
    #    The C++ game sorts targets opponent-first so action 0 is usually correct,
    #    but guard against accidentally targeting own spells/permanents.
    for i, c in enumerate(cats):
        if c == _CAT_TARGET and ctrl_arr[i] < 0.5:
            return i
    # Fallback: all targets are self-controlled — return first (shouldn't happen in practice).
    for i, c in enumerate(cats):
        if c == _CAT_TARGET:
            return i

    # 5. Search library — action 0 = fail to find; pick action 1 (first actual card).
    #    Used when fetch lands (Scalding Tarn, Misty Rainforest, Polluted Delta,
    #    Wooded Foothills) resolve their ChangeZone ability.
    if any(c == _CAT_SEARCH for c in cats):
        return 1 if num_choices > 1 else 0

    # 6. Cast any spell immediately if affordable (game only offers CAST when legal).
    #    Counter spells (Counterspell, Daze, Force of Will) require an opponent's spell
    #    on the stack; skip them when the stack holds only own spells or is empty.
    opponent_spell_on_stack = _opponent_has_spell_on_stack(obs)

    # Priority: if opponent has a spell on stack and we have UU, cast Counterspell first.
    if opponent_spell_on_stack and int(round(obs[_BLUE_POOL_IDX] * 10)) >= 2:
        for i, c in enumerate(cats):
            if c == _CAT_CAST and round(float(card_ids[i]) * N_CARD_TYPES) == _COUNTERSPELL_VOCAB_IDX:
                return i

    for i, c in enumerate(cats):
        if c == _CAT_CAST:
            cid = round(float(card_ids[i]) * N_CARD_TYPES)
            if cid in _COUNTER_SPELL_VOCAB_IDS and not opponent_spell_on_stack:
                continue
            return i

    # 7. Play land (may unlock mana for a spell next query)
    for i, c in enumerate(cats):
        if c == _CAT_LAND:
            return i

    # 8. Activate non-mana abilities during main phase:
    #    - Fetch lands (Scalding Tarn, Misty Rainforest, Polluted Delta, Wooded Foothills):
    #      tap + pay 1 life + sacrifice → search for land (resolved by step 5 above)
    #    - Wasteland: tap + sacrifice → destroy target nonbasic land (target chosen in step 4).
    #      Guard: only activate if the opponent actually has a nonbasic land to target,
    #      otherwise action 0 for SELECT_TARGET would be our own land.
    if in_main_phase:
        for i, c in enumerate(cats):
            if c == _CAT_ACTIVATE:
                cid = round(float(card_ids[i]) * N_CARD_TYPES)
                if cid == _WASTELAND_VOCAB_IDX and not _opponent_has_nonbasic_land(obs):
                    continue  # no opponent nonbasic land to target — skip
                return i

    # 9. Tap mana during main phases only, choosing the color the hand needs most
    if in_main_phase and any(c in _MANA_CATS for c in cats):
        # Determine what colored mana is still needed for cards in hand
        # Perspective-normalized: self's mana pool is always at [3-8]
        pool_start = 3
        pool = [int(round(obs[pool_start + i] * 10)) for i in range(6)]
        needed = [0] * 6
        for slot in range(10):
            base = _HAND_START + slot * N_CARD_TYPES
            slot_vec = obs[base:base + N_CARD_TYPES]
            card_idx = int(np.argmax(slot_vec))
            if slot_vec[card_idx] < 0.5:
                continue  # empty slot
            for color_idx, count in _CARD_COLORED_COSTS.get(card_idx, {}).items():
                needed[color_idx] = max(needed[color_idx], count)
        short = [max(0, needed[i] - pool[i]) for i in range(6)]

        # Prefer the color we're shortest on
        for color_idx in sorted(range(6), key=lambda ci: -short[ci]):
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
    """Wraps RoboMageEnv so the model plays against a scripted agent.

    Each episode the model is randomly assigned to Player A or B; the scripted
    agent takes the other side.  Reward is negated when the model plays as B so
    it is always from the model's perspective (+1 win, -1 loss).
    """

    def __init__(self, binary_path: str = BINARY, render_mode=None):
        super().__init__()
        self._env = RoboMageEnv(binary_path=binary_path, render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = render_mode
        self._training_is_a = True
        self._pending_shaping = 0.0
        self._opponent_below_10 = False

    def reset(self, *, seed=None, options=None):
        self._training_is_a = bool(np.random.random() < 0.5)
        self._pending_shaping = 0.0
        self._opponent_below_10 = False
        obs, info = self._env.reset(seed=seed, options=options)
        obs, _reward, terminated, truncated, info = self._skip_opponent_turns(
            obs, 0.0, False, False, info
        )
        self._pending_shaping = 0.0  # discard any shaping from setup turns
        self._opponent_below_10 = False  # reset after mulligan/setup
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._accumulate_shaping(info)
        if not (terminated or truncated):
            self._check_opponent_below_10(obs)
            obs, reward, terminated, truncated, info = self._skip_opponent_turns(
                obs, reward, terminated, truncated, info
            )
        if not self._training_is_a:
            reward = -reward
        reward += self._pending_shaping
        self._pending_shaping = 0.0
        return obs, reward, terminated, truncated, info

    def _accumulate_shaping(self, info):
        """Add the model's per-step mana-waste penalty to the running total."""
        if self._training_is_a:
            self._pending_shaping += info.get("shaping_a", 0.0)
        else:
            self._pending_shaping += info.get("shaping_b", 0.0)

    def _check_opponent_below_10(self, obs):
        """Issue +0.2 shaping reward the first time the scripted opponent's life drops below 10."""
        if self._opponent_below_10:
            return
        # obs is always from the priority player's perspective.
        # When the model has priority, obs[9] = scripted life / 20.
        # When the scripted agent has priority, obs[0] = scripted ("self") life / 20.
        a_has_priority = obs[31] > 0.5
        model_has_priority = a_has_priority if self._training_is_a else not a_has_priority
        scripted_life = (obs[9] if model_has_priority else obs[0]) * 20.0
        if scripted_life < 10.0:
            self._opponent_below_10 = True
            self._pending_shaping += 0.2

    def _skip_opponent_turns(self, obs, reward, terminated, truncated, info):
        """Resolve consecutive opponent turns with the scripted agent."""
        while not (terminated or truncated) and (obs[31] > 0.5) != self._training_is_a:
            action = scripted_action(obs, self._env._num_choices)
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._accumulate_shaping(info)
            if not (terminated or truncated):
                self._check_opponent_below_10(obs)
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()


# ── Observation mirroring ─────────────────────────────────────────────────────

def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Flip the observation from Player A's perspective to Player B's (or vice versa).

    NOTE: This function is no longer invoked in normal training flow.
    Perspective normalization is now handled by the game engine (serialize_state
    always emits from the priority player's view).  mirror_obs is retained for
    legacy compatibility and offline analysis only.

    After mirroring:
      - obs[31] = 1.0 always means "I (the calling player) have priority"
      - controller_is_self = 1.0 always marks the calling player's permanents
      - slots 0-23 always contain the calling player's creatures / lands
      - slots 0-63 always contain the calling player's graveyard

    Only the first STATE_SIZE floats and the bf_ability_costs block (derived from
    creature slot ordering) are modified.  Action categories, action card-IDs, and
    hand cast-costs are perspective-independent and left unchanged.
    """
    m = obs.copy()

    # 1. Swap player scalar stats: life, hand size, poison, mana×6 (indices 0-8 ↔ 9-17)
    m[0:9], m[9:18] = obs[9:18].copy(), obs[0:9].copy()

    # 2. Flip turn/priority flags
    m[30] = 1.0 - obs[30]
    m[31] = 1.0 - obs[31]

    # 3. Swap creature slots 0-23 ↔ 24-47 and flip controller_is_self in all 48 slots
    for i in range(_BF_A_SLOTS):
        a = _BF_START + i * _BF_SLOT_SIZE
        b = _BF_START + (i + _BF_A_SLOTS) * _BF_SLOT_SIZE
        m[a:a + _BF_SLOT_SIZE] = obs[b:b + _BF_SLOT_SIZE]
        m[b:b + _BF_SLOT_SIZE] = obs[a:a + _BF_SLOT_SIZE]
        m[a + _CTRL_OFF] = 1.0 - obs[b + _CTRL_OFF]
        m[b + _CTRL_OFF] = 1.0 - obs[a + _CTRL_OFF]

    # 4. Swap land slots 0-23 ↔ 24-47 and flip controller_is_self
    for i in range(_BF_A_SLOTS):
        a = _LAND_START + i * _BF_SLOT_SIZE
        b = _LAND_START + (i + _BF_A_SLOTS) * _BF_SLOT_SIZE
        m[a:a + _BF_SLOT_SIZE] = obs[b:b + _BF_SLOT_SIZE]
        m[b:b + _BF_SLOT_SIZE] = obs[a:a + _BF_SLOT_SIZE]
        m[a + _CTRL_OFF] = 1.0 - obs[b + _CTRL_OFF]
        m[b + _CTRL_OFF] = 1.0 - obs[a + _CTRL_OFF]

    # 5. Flip controller_is_self in stack slots (first float of each 33-float slot)
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        m[base] = 1.0 - obs[base]

    # 6. Swap graveyard slots 0-63 ↔ 64-127 (pure card one-hots, no flag to flip)
    for i in range(_GY_A_SLOTS):
        a = _GY_START + i * _GY_SLOT_SIZE
        b = _GY_START + (i + _GY_A_SLOTS) * _GY_SLOT_SIZE
        m[a:a + _GY_SLOT_SIZE] = obs[b:b + _GY_SLOT_SIZE]
        m[b:b + _GY_SLOT_SIZE] = obs[a:a + _GY_SLOT_SIZE]

    # 7. Hand slots (obs[_HAND_START:STATE_SIZE]): always the priority player's hand —
    #    no change needed; after mirror obs[31]=1.0 still means the acting player's hand
    #    is shown here.

    # 8. Swap bf_ability_costs slots 0-23 ↔ 24-47 (mirrors the creature slot reordering)
    for i in range(_BF_A_SLOTS):
        a = _BF_COST_START + i * _N_COST_FEATS
        b = _BF_COST_START + (i + _BF_A_SLOTS) * _N_COST_FEATS
        m[a:a + _N_COST_FEATS] = obs[b:b + _N_COST_FEATS]
        m[b:b + _N_COST_FEATS] = obs[a:a + _N_COST_FEATS]

    return m


# ── Self-play environment ─────────────────────────────────────────────────────

class SelfPlayEnv(gym.Env):
    """Self-play: the training model plays against a frozen previous checkpoint.

    Each episode one player role (A or B) is randomly assigned to the training
    model; the other is controlled by the frozen opponent.  The game engine emits
    observations from the priority player's perspective, so both sides always
    receive a perspective-normalised view without any mirroring.

    The opponent checkpoint is sampled from ``checkpoint_dir`` and reloaded every
    ``RELOAD_EVERY`` episodes so it gradually tracks the improving policy.  If no
    checkpoints exist yet the opponent acts randomly (bootstrapping phase).
    """

    RELOAD_EVERY = 10  # episodes between opponent checkpoint reloads

    def __init__(self, checkpoint_dir: str, binary_path: str = BINARY, render_mode=None):
        super().__init__()
        self._env = RoboMageEnv(binary_path=binary_path, render_mode=render_mode)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = render_mode
        self._checkpoint_dir = checkpoint_dir
        self._opponent = None    # loaded model, or None → random
        self._episode_count = 0
        self._training_is_a = True
        self._reload_opponent()  # attempt to load an initial checkpoint

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self._episode_count += 1
        if self._episode_count % self.RELOAD_EVERY == 0:
            self._reload_opponent()

        obs, info = self._env.reset(seed=seed, options=options)
        self._training_is_a = bool(np.random.random() < 0.5)

        obs, reward, terminated, truncated, info = self._handle_opponent_turns(
            obs, 0.0, False, False, info
        )
        if terminated or truncated:
            return np.zeros(OBS_SIZE, dtype=np.float32), info
        return self._training_obs(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self._handle_opponent_turns(
                obs, reward, terminated, truncated, info
            )
        # Reward is from Player A's perspective; negate if training model plays as B.
        if not self._training_is_a:
            reward = -reward
        return self._training_obs(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _training_has_priority(self, obs: np.ndarray) -> bool:
        """True when it is the training model's turn to act (raw obs)."""
        a_has_priority = obs[31] > 0.5
        return a_has_priority if self._training_is_a else not a_has_priority

    def _training_obs(self, obs: np.ndarray) -> np.ndarray:
        """Return obs from the training model's perspective.

        Perspective normalization is handled by the game engine, so no mirroring
        is needed — the observation is already from the priority player's view.
        """
        return obs

    def _handle_opponent_turns(self, obs, reward, terminated, truncated, info):
        """Step with the frozen opponent until it is the training model's turn."""
        while not (terminated or truncated) and not self._training_has_priority(obs):
            num_choices = self._env._num_choices
            # Opponent receives the state as emitted (already from its perspective)
            opp_obs = obs
            if self._opponent is not None:
                masks = np.zeros(MAX_ACTIONS, dtype=bool)
                masks[:num_choices] = True
                action, _ = self._opponent.predict(opp_obs, action_masks=masks, deterministic=False)
                action = int(action)
            else:
                action = np.random.randint(0, num_choices)
            obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info

    def _reload_opponent(self):
        """Sample a random checkpoint from the pool as the new frozen opponent."""
        files = _glob.glob(os.path.join(self._checkpoint_dir, "*.zip"))
        if not files:
            self._opponent = None
            return
        path = str(np.random.choice(files))
        try:
            try:
                from sb3_contrib import MaskablePPO as _PPO
            except ImportError:
                from stable_baselines3 import PPO as _PPO
            self._opponent = _PPO.load(path, device="cpu")
        except Exception:
            self._opponent = None  # fall back to random if load fails
