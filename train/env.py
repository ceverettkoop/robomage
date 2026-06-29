"""
RoboMage gymnasium environment.

The game runs as a subprocess with --machine mode. On each decision point it
emits a BQUERY line to stdout:

    BQUERY: <num_choices>\n<float32[STATE_SIZE] binary><int32[MAX_ACTIONS] cats><float32[MAX_ACTIONS] ids><float32[MAX_ACTIONS] ctrl>

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

2813-float state vector. Card identity is a single normalized id float per slot
(idx/N_CARD_TYPES, -1/N_CARD_TYPES = empty), NOT a one-hot — the policy network
maps ids through a learned nn.Embedding. The opponent revealed-cards block is the
only vocab-width block (N_CARD_TYPES multi-hot).
State (2813) + 64 action-category floats + 64 action card-ID floats
+ 64 action controller_is_self floats + 70 hand cost floats
+ 336 battlefield ability cost floats = OBS_SIZE total.
NOTE: ActionChoice.description is NOT part of the observation — it is for
human-readable display only (GUI/CLI) and is never sent to the ML model.
NOTE: Exile zones are tracked in GameState but not serialized to the observation.
See src/machine_io.h for the full state layout.

Reward
------
+1.0 for winning, -1.0 for losing (from Player A's perspective).
"""

import struct
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

# ACTION_CATEGORY_MAX is generated from the C++ ActionCategory enum (single source
# of truth) by train/gen_enums.py — import it so this module never drifts from the
# engine's category normalization (used for both the action block and history).
try:
    from _enums import ACTION_CATEGORY_MAX
except ImportError:
    from train._enums import ACTION_CATEGORY_MAX

STATE_SIZE = 2919  # see src/machine_io.h; card identity is 1 id float/slot, not a one-hot
# NOTE: Exile zones are tracked in GameState but not serialized to the observation.
# NOTE: ActionChoice.description is never emitted in the BQUERY payload — it is for
#       human-readable display only and is not part of the ML observation.
MAX_ACTIONS = 64         # practical upper bound on num_choices per step
# Binary BQUERY payload sizes (bytes): state float32s + MAX_ACTIONS each of
# cats(int32)/ids/ctrl(float32)/pub(float32)
_BQUERY_STATE_BYTES = STATE_SIZE * 4
_BQUERY_CATS_BYTES  = MAX_ACTIONS * 4  # int32
_BQUERY_IDS_BYTES   = MAX_ACTIONS * 4  # float32
_BQUERY_CTRL_BYTES  = MAX_ACTIONS * 4  # float32
_BQUERY_PUB_BYTES   = MAX_ACTIONS * 4  # float32 — card_is_public per action
# Per-action human-readable descriptions, emitted ONLY under --narrative
# (gated on the engine side too). Fixed [MAX_ACTIONS][MAX_CHOICE_DESC] NUL-padded
# char block — must match MAX_CHOICE_DESC in src/classes/gamestate.h.
MAX_CHOICE_DESC     = 128
_BQUERY_DESC_BYTES  = MAX_ACTIONS * MAX_CHOICE_DESC
# ACTION_CATEGORY_MAX imported from _enums above (mirrors src/classes/action.h).

# ── Shaping reward magnitudes ─────────────────────────────────────────────────
SHAPING_MANA_WASTED      = -0.00  # per drain event with mana remaining in pool; commented out because we aren't letting it float anymore
SHAPING_MULLIGAN_PENALTY =  0.00  # per mulligan taken beyond the 2nd (C++: >= 3rd)
SHAPING_OPPONENT_BELOW10 =  0.00  # one-time bonus when opponent life first drops < 10
SHAPING_HAND_ADV_PER_CARD = 0.01  # potential weight per card of hand advantage (potential-based)
SHAPING_POWER_ADV_PER_PT  = 0.005 # potential weight per point of power advantage on board
SHAPING_EPISODE_CAP       = 0.3   # max absolute shaping bonus per episode
SHAPING_EPISODE_CAP_DOOMSDAY = 0.6  # higher cap for doomsday deck

# ── Doomsday deck shaping ────────────────────────────────────────────────────
SHAPING_DD_CAST_DOOMSDAY    = 0.2  # reward for casting Doomsday
SHAPING_DD_RITUAL_SETUP     = 0.1  # reward for casting Dark Ritual when Doomsday in hand + main phase
SHAPING_DD_PICK_ORACLE      = 0.2  # reward for picking Thassa's Oracle during Doomsday pile
SHAPING_DD_CAST_DISCARD     = 0.03  # reward for casting Thoughtseize/Duress targeting opponent
SHAPING_DD_STRIP_COUNTER    = 0.00 # reward for selecting Force of Will or Daze with discard spell
SHAPING_DD_TUTOR_DOOMSDAY   = 0.05 # reward for selecting Doomsday with Personal Tutor
SHAPING_DD_KEEP_DOOMSDAY    = 0.00  # reward for not shuffling after placed Doomsday on top (Ponder)
SHAPING_DD_LED_WITH_DRAW    = 0.03 # reward for cracking LED with a cycling/draw ability on the stack
SHAPING_DD_LED_EMPTY_STACK  = -0.02 # penalty for cracking LED with nothing on the stack

# ── Bo3 match rewards ────────────────────────────────────────────────────────
BO3_GAME_WIN_REWARD   =  0.3   # intermediate reward for winning a game in bo3
BO3_GAME_LOSS_REWARD  = -0.3   # intermediate penalty for losing a game in bo3
BO3_MATCH_WIN_REWARD  =  1.0   # terminal reward for winning the match
BO3_MATCH_LOSS_REWARD = -1.0   # terminal penalty for losing the match
_ACTION_CARD_ID_NULL = -1.0 / N_CARD_TYPES  # null sentinel for non-card slots
_ACTION_CTRL_NULL    = -1.0 / N_CARD_TYPES  # null sentinel for non-entity actions
MAX_HAND_SLOTS = 10
_HAND_COST_FEATS  = MAX_HAND_SLOTS * _N_COST_FEATS  # 10 * 7 = 70
_BF_ABILITY_FEATS = 48 * _N_COST_FEATS              # 48 * 7 = 336
OBS_SIZE = STATE_SIZE + 3 * MAX_ACTIONS + _HAND_COST_FEATS + _BF_ABILITY_FEATS  # 34392

# ── State layout offsets (mirror src/machine_io.h) ───────────────────────────
# Creatures, lands, and other permanents share one unified section (no separate land slots).
# Card identity is a single normalized id float per slot (idx/N_CARD_TYPES, or
# -1/N_CARD_TYPES for empty/unknown). Decode with round(val * N_CARD_TYPES).
_GLOBAL_SIZE            = 34                   # header: player blocks, step one-hot, flags, stack size
_PERM_SLOTS             = 48                   # per-player; 96 total (self + opp)
_PERM_SLOT_SIZE         = 12                   # 11 status (incl. loyalty) + 1 card id
_STACK_SLOTS            = 12
_STACK_SLOT_SIZE        = 3                    # ctrl + card id + is_spell
_GY_SLOTS_TOTAL         = 128                  # 64 self + 64 opponent
_GY_SLOT_SIZE           = 1                    # card id only
_HAND_SLOTS_TOTAL       = 10
_HAND_SLOT_SIZE         = 1
_ACTION_HISTORY_SIZE    = 128                  # entries in the action history ring
_ACTION_HISTORY_ENTRY   = 4                    # cat_norm, card_id, is_self, turn/50
_MATCH_CTX_SIZE         = 4                    # game_number, self_wins, opp_wins, sideboard_phase
_LIBRARY_CTX_SIZE       = 3                    # self_lib/60, opp_lib/60, is_post_board
_CUR_TURN_SIZE          = 1                    # current turn / 50
_KNOWN_TOP_LIB_SLOTS    = 5                    # serialized known top-of-library cards
_KNOWN_TOP_LIB_SLOT_SIZE = 1                   # card id per slot
_REVEALED_SIZE          = N_CARD_TYPES         # opponent revealed-cards multi-hot (only vocab-width block)
_OPP_KNOWN_HAND_SLOTS   = 10                   # known opponent-hand card identities
_OPP_KNOWN_HAND_SLOT_SIZE = 1                  # card id per slot

_SELF_PERM_START     = _GLOBAL_SIZE                                                  # 34
_OPP_PERM_START      = _SELF_PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE              # 610
_STACK_START         = _OPP_PERM_START + _PERM_SLOTS * _PERM_SLOT_SIZE               # 1186
_GY_START            = _STACK_START + _STACK_SLOTS * _STACK_SLOT_SIZE                # 1222
_HAND_START          = _GY_START + _GY_SLOTS_TOTAL * _GY_SLOT_SIZE                   # 1350
_HIST_START          = _HAND_START + _HAND_SLOTS_TOTAL * _HAND_SLOT_SIZE             # 1360
_HIST_END            = _HIST_START + _ACTION_HISTORY_SIZE * _ACTION_HISTORY_ENTRY    # 1872
_MATCH_CTX_START     = _HIST_END                                                     # 1872
_LIBRARY_CTX_START   = _MATCH_CTX_START + _MATCH_CTX_SIZE                            # 1876
_CUR_TURN_IDX        = _LIBRARY_CTX_START + _LIBRARY_CTX_SIZE                        # 1879
_KNOWN_TOP_LIB_START = _CUR_TURN_IDX + _CUR_TURN_SIZE                                # 1880
_KNOWN_TOP_LIB_END   = _KNOWN_TOP_LIB_START + _KNOWN_TOP_LIB_SLOTS * _KNOWN_TOP_LIB_SLOT_SIZE  # 1885
_REVEALED_START      = _KNOWN_TOP_LIB_END                                            # 1885
_REVEALED_END        = _REVEALED_START + _REVEALED_SIZE                              # 2909
_OPP_KNOWN_HAND_START = _REVEALED_END                                                # 2909
_OPP_KNOWN_HAND_END  = _OPP_KNOWN_HAND_START + _OPP_KNOWN_HAND_SLOTS * _OPP_KNOWN_HAND_SLOT_SIZE  # 2919

assert _OPP_KNOWN_HAND_END == STATE_SIZE, (_OPP_KNOWN_HAND_END, STATE_SIZE)

# Offset of the card-id float within a permanent slot (after the 11 status floats).
_PERM_CARD_OFF = 11


def _slot_card_idx(obs, i):
    """Decode the vocab index from a single normalized id float at obs[i] (-1 = empty)."""
    return int(round(float(obs[i]) * N_CARD_TYPES))


_SELF_PERM_POWER_IDX = np.arange(_PERM_SLOTS) * _PERM_SLOT_SIZE + _SELF_PERM_START
_SELF_PERM_CREATURE_IDX = _SELF_PERM_POWER_IDX + 8
_OPP_PERM_POWER_IDX = np.arange(_PERM_SLOTS) * _PERM_SLOT_SIZE + _OPP_PERM_START
_OPP_PERM_CREATURE_IDX = _OPP_PERM_POWER_IDX + 8

def _board_power_advantage(obs):
    """Return self_power - opp_power from the observation vector."""
    self_mask = obs[_SELF_PERM_CREATURE_IDX] > 0.5
    self_power = np.sum(obs[_SELF_PERM_POWER_IDX[self_mask]]) * 10.0
    opp_mask = obs[_OPP_PERM_CREATURE_IDX] > 0.5
    opp_power = np.sum(obs[_OPP_PERM_POWER_IDX[opp_mask]]) * 10.0
    return self_power - opp_power

# CLI constants live in cli_spec.py (the single source shared with the TUI).
from cli_spec import REPO_ROOT as _REPO_ROOT, BINARY, BIN_DIR  # noqa: E402


class RoboMageEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # Maximum decision steps per episode before truncation.  Prevents infinite
    # loops (e.g. a model that toggles attackers forever) from hanging training.
    MAX_STEPS = 1000
    MAX_STEPS_BO3 = 3000  # higher limit for best-of-three matches

    def __init__(self, binary_path: str = BINARY, render_mode=None,
                 deck_a: str | None = None, deck_b: str | None = None,
                 bo3: bool = False, auto_sideboard: bool = False,
                 narrative: bool = False, no_shuffle: bool = False,
                 battlefield_a: str | None = None, battlefield_b: str | None = None,
                 graveyard_a: str | None = None, graveyard_b: str | None = None,
                 exile_a: str | None = None, exile_b: str | None = None,
                 sideboard_a: str | None = None, sideboard_b: str | None = None,
                 log_viewer: str | None = None):
        super().__init__()
        self.binary_path = os.path.realpath(binary_path)
        self.render_mode = render_mode
        self._deck_a = deck_a
        self._deck_b = deck_b
        self._bo3 = bo3
        self._auto_sideboard = auto_sideboard
        # State-seeding / observation flags passed through to the engine. These
        # mirror the test-harness CLI so the shared run loop (runner.run_games)
        # can sculpt a starting state: --narrative emits the full game log,
        # --no-shuffle keeps deck-file order = draw order, and --battlefield-a/-b
        # pre-place permanents (comma-joined deck-name strings).
        self._narrative = narrative
        self._no_shuffle = no_shuffle
        self._battlefield_a = battlefield_a
        self._battlefield_b = battlefield_b
        self._graveyard_a = graveyard_a
        self._graveyard_b = graveyard_b
        # --exile-a/-b and --sideboard-a/-b pre-place cards in those non-battlefield
        # zones so zone-change effects that pull from exile / "outside the game"
        # (e.g. Karn, the Great Creator's -2) can be exercised in isolation.
        self._exile_a = exile_a
        self._exile_b = exile_b
        self._sideboard_a = sideboard_a
        self._sideboard_b = sideboard_b
        # When set ("A"/"B"), the engine redacts game_log_private narrative to
        # that seat's view (hidden draws, tutored/top-of-library cards) without
        # rerouting input — both seats still respond over the machine protocol.
        self._log_viewer = log_viewer

        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(OBS_SIZE,), dtype=np.float32
        )
        # Discrete action space sized to the max we'd ever see.
        # Invalid actions are masked at each step via `action_masks()`.
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self._proc = None
        self._num_choices = 1
        self._obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._action_public = np.zeros(MAX_ACTIONS, dtype=np.float32)  # card_is_public per action
        self._action_descriptions = None  # list[str] per action under --narrative, else None
        self._pending_confirm = False  # True when last query used the -1 convention
        self._step_count = 0

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._kill_proc()
        # Generate a unique seed for each game so time(nullptr) collisions don't
        # produce repeated games when many resets happen within the same second.
        rng_seed = self.np_random.integers(0, 2**31 - 1)
        cmd = [self.binary_path, "--machine", "--seed", str(rng_seed)]
        if self._narrative:
            cmd += ["--narrative"]
        if self._bo3:
            cmd += ["--bo3"]
        if self._deck_a:
            cmd += ["--deck-a", self._deck_a]
        if self._deck_b:
            cmd += ["--deck-b", self._deck_b]
        if self._no_shuffle:
            cmd += ["--no-shuffle"]
        if self._battlefield_a:
            cmd += ["--battlefield-a", self._battlefield_a]
        if self._battlefield_b:
            cmd += ["--battlefield-b", self._battlefield_b]
        if self._graveyard_a:
            cmd += ["--graveyard-a", self._graveyard_a]
        if self._graveyard_b:
            cmd += ["--graveyard-b", self._graveyard_b]
        if self._exile_a:
            cmd += ["--exile-a", self._exile_a]
        if self._exile_b:
            cmd += ["--exile-b", self._exile_b]
        if self._sideboard_a:
            cmd += ["--sideboard-a", self._sideboard_a]
        if self._sideboard_b:
            cmd += ["--sideboard-b", self._sideboard_b]
        if self._log_viewer:
            cmd += ["--log-viewer", self._log_viewer]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit parent stderr so engine errors are visible
            bufsize=-1,  # binary mode, fully buffered
            cwd=BIN_DIR,  # game uses getcwd() to locate resources/
        )
        obs, info = self._read_until_query()
        return obs, info

    def step(self, action: int):
        assert self._proc is not None, "Call reset() first"

        self._step_count += 1
        max_steps = self.MAX_STEPS_BO3 if self._bo3 else self.MAX_STEPS
        if self._step_count >= max_steps:
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
        self._proc.stdin.write(f"{action}\n".encode())
        self._proc.stdin.flush()

    def _read_exactly(self, n: int) -> bytes:
        """Read exactly n bytes from stdout, blocking until available."""
        buf = bytearray()
        while len(buf) < n:
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError("Game process ended while reading binary payload")
            buf.extend(chunk)
        return bytes(buf)

    def _read_until_query(self):
        """
        Read lines from the game process until a BQUERY line appears or the
        process exits (game over).  After the BQUERY header line, reads the
        binary payload: float32[STATE_SIZE], int32[MAX_ACTIONS] cats,
        float32[MAX_ACTIONS] ids, float32[MAX_ACTIONS] ctrl.
        Returns (obs, info).
        """
        reward = 0.0
        done = False
        game_result = False
        shaping_a = 0.0
        shaping_b = 0.0

        while True:
            line = self._proc.stdout.readline()

            if not line:
                # Process ended — game over
                done = True
                break

            line = line.rstrip(b"\n")

            # Detect win/loss
            if self._bo3:
                # In bo3 mode, individual game results are intermediate rewards
                if line.startswith(b"GAME_RESULT:"):
                    game_result = True
                    if b"Player A wins" in line:
                        reward += BO3_GAME_WIN_REWARD
                    elif b"Player B wins" in line:
                        reward += BO3_GAME_LOSS_REWARD
                elif line.startswith(b"MATCH_RESULT:"):
                    if b"Player A wins" in line:
                        reward += BO3_MATCH_WIN_REWARD
                    elif b"Player B wins" in line:
                        reward += BO3_MATCH_LOSS_REWARD
                    done = True
            else:
                if b"Player A wins" in line:
                    reward = 1.0
                    done = True
                elif b"Player B wins" in line:
                    reward = -1.0
                    done = True

            # Shaping signal: mana wasted at end of phase (pool non-empty on drain)
            if line.startswith(b"MANA_WASTED: "):
                if line.endswith(b"A"):
                    shaping_a += SHAPING_MANA_WASTED
                elif line.endswith(b"B"):
                    shaping_b += SHAPING_MANA_WASTED
                continue

            # Shaping signal: excessive mulligan (3rd and beyond)
            if line.startswith(b"MULLIGAN_PENALTY: "):
                if line.endswith(b"A"):
                    shaping_a += SHAPING_MULLIGAN_PENALTY
                elif line.endswith(b"B"):
                    shaping_b += SHAPING_MULLIGAN_PENALTY
                continue

            if line.startswith(b"BQUERY: "):
                n = int(line[8:])
                self._num_choices = min(n, MAX_ACTIONS)

                # Binary reads: state floats, then padded action metadata
                state_arr = np.frombuffer(
                    self._read_exactly(_BQUERY_STATE_BYTES), dtype=np.float32).copy()
                cats_int = np.frombuffer(
                    self._read_exactly(_BQUERY_CATS_BYTES), dtype=np.int32)
                id_arr = np.frombuffer(
                    self._read_exactly(_BQUERY_IDS_BYTES), dtype=np.float32).copy()
                ctrl_arr = np.frombuffer(
                    self._read_exactly(_BQUERY_CTRL_BYTES), dtype=np.float32).copy()
                pub_arr = np.frombuffer(
                    self._read_exactly(_BQUERY_PUB_BYTES), dtype=np.float32).copy()

                # Per-action "card identity is public" flags (revealed tutors). Kept as a
                # side-channel — observers (TUI) read it; not part of the ML observation
                # vector yet, so OBS_SIZE and trained checkpoints are unaffected.
                self._action_public = pub_arr

                # Under --narrative the engine appends a fixed char block of
                # per-action descriptions (the exact CLI/GUI labels). Read and
                # decode it so observers can show "Target Player B", "Pay 4 life",
                # etc. — things the numeric metadata can't express. Off the
                # training path (narrative=False), so OBS is unaffected.
                if self._narrative:
                    desc_bytes = self._read_exactly(_BQUERY_DESC_BYTES)
                    descs = []
                    for k in range(MAX_ACTIONS):
                        chunk = desc_bytes[k * MAX_CHOICE_DESC:(k + 1) * MAX_CHOICE_DESC]
                        end = chunk.find(b"\x00")
                        if end >= 0:
                            chunk = chunk[:end]
                        descs.append(chunk.decode("utf-8", errors="replace"))
                    self._action_descriptions = descs
                else:
                    self._action_descriptions = None

                # The -1 confirm convention applies to mandatory attacker/blocker queries.
                self._pending_confirm = any(
                    c in _MANDATORY_CATS for c in cats_int[:self._num_choices])

                # Hand cast costs: gather cost rows by hand-slot card id
                hand_ids = np.rint(
                    state_arr[_HAND_START:_HAND_START + MAX_HAND_SLOTS] * N_CARD_TYPES).astype(np.intp)
                hand_costs = _gather_costs(_CARD_COST_MATRIX, hand_ids)

                # Battlefield activated ability costs (48 self permanent slots)
                bf_ids = np.rint(state_arr[_BF_ID_IDX] * N_CARD_TYPES).astype(np.intp)
                bf_ability_costs = _gather_costs(_CARD_ABILITY_COST_MATRIX, bf_ids)

                # Write sections into preallocated obs buffer (avoids concatenate allocation)
                o = self._obs
                o[:STATE_SIZE] = state_arr
                _act_end = STATE_SIZE + MAX_ACTIONS
                o[STATE_SIZE:_act_end] = cats_int / ACTION_CATEGORY_MAX
                o[_act_end:_act_end + MAX_ACTIONS] = id_arr
                o[_act_end + MAX_ACTIONS:_act_end + 2 * MAX_ACTIONS] = ctrl_arr
                _hc_start = _act_end + 2 * MAX_ACTIONS
                o[_hc_start:_hc_start + _HAND_COST_FEATS] = hand_costs.ravel()
                _bf_start = _hc_start + _HAND_COST_FEATS
                o[_bf_start:_bf_start + _BF_ABILITY_FEATS] = bf_ability_costs.ravel()

                # Auto-sideboard: if enabled, automatically pick "done" (action 0)
                # for all sideboard queries so the model never sees them.
                if self._auto_sideboard and any(
                    c == _CAT_SB_DONE for c in cats_int[:self._num_choices]
                ):
                    self._send(0)
                    continue

                break

            # Non-BQUERY output: optionally print for human render mode
            if self.render_mode == "human":
                self._print_narrative_line(line.decode("utf-8", errors="replace"))

        info = {"reward": reward, "done": done, "shaping_a": shaping_a, "shaping_b": shaping_b,
                "game_result": game_result}
        if done:
            self._kill_proc()
            # Return a zero obs on terminal step — will be replaced by reset()
            return np.zeros(OBS_SIZE, dtype=np.float32), info

        # np.concatenate already produced a fresh array; no copy needed
        return self._obs, info

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
    """RoboMageEnv that collects non-BQUERY game lines into a list instead of printing them.

    Defaults ``narrative=True`` so the engine emits its full game log (casts,
    damage, zone moves, combat); ``flush_lines`` hands those buffered lines to
    the caller. This is what makes the shared observation loop show the same
    narrative the test harness does.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("render_mode", "human")
        kwargs.setdefault("narrative", True)
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
_MANDATORY_CATS = frozenset({2, 3, 4, 5})  # attacker/blocker confirm categories
_CAT_CONF_ATK   = 3
_CAT_SEL_BLK    = 4
_CAT_CONF_BLK   = 5
_CAT_ACTIVATE   = 6   # activate a non-mana ability (fetch lands, Wasteland destroy)
_CAT_CAST       = 7
_CAT_TARGET     = 8
_CAT_LAND       = 9
_CAT_MULLIGAN   = 11
_CAT_SEARCH     = 19  # search library (action 0 = fail to find, 1+ = actual cards)
_CAT_OTHER      = 10  # generic/unclassified choice (fallback default)
_CAT_DISCARD    = 30  # choose a card to discard (cost, effect, or cleanup)
_CAT_PAYING     = 22  # paying costs (tap lands for mana, delve exile, pitch cards)
_CAT_DIG        = 23  # dig choice (Once Upon a Time: pick creature/land from top N)
_CAT_SB_IN      = 24  # sideboard: choose card from sideboard to add
_CAT_SB_OUT     = 25  # sideboard: choose card from main deck to remove
_CAT_SB_DONE    = 26  # sideboard: finish sideboarding

# Colored mana requirements per card vocab index (card_vocab.h).
# Keys are color pool indices: W=0, U=1, B=2, R=3, G=4, C=5.
# Generic mana is omitted — any color satisfies it.
_CARD_COLORED_COSTS = {
    2:  {3: 1},   # Lightning Bolt          (R)   — 1 red
    11: {1: 1},   # Ponder                  (U)   — 1 blue
    13: {1: 1},   # Daze                    (1U)  — 1 blue
    16: {1: 1},   # Delver of Secrets       (U)   — 1 blue
    18: {1: 1},   # Flying Men              (U)   — 1 blue
    20: {3: 1},   # Dragon's Rage Channeler (R)   — 1 red
    21: {1: 2},   # Air Elemental           (3UU) — 2 blue
    22: {1: 2},   # Counterspell            (UU)  — 2 blue
    23: {3: 1},   # Lightning Strike        (1R)  — 1 red
    24: {1: 1},   # Brainstorm              (U)   — 1 blue
    26: {1: 2},   # Murktide Regent         (5UU, Delve) — 2 blue
    28: {3: 1},   # Cori-Steel Cutter       (1R)  — 1 red
    29: {3: 1},   # Unholy Heat             (R)   — 1 red
}

# ── Battlefield layout (aliases of the unified state offsets above) ─────────
_BF_START         = _SELF_PERM_START           # 34
_BF_SLOT_SIZE     = _PERM_SLOT_SIZE            # 12
_PERM_A_SLOTS     = _PERM_SLOTS                # 48: self occupies perm slots 0-47, opponent slots 48-95
_BF_CARD_OFF      = _PERM_CARD_OFF             # offset of the card-id float within each permanent slot
# Precomputed indices for gathering the 48 self-permanent card-id floats from the state array
_BF_ID_IDX        = _BF_START + np.arange(_PERM_A_SLOTS) * _BF_SLOT_SIZE + _BF_CARD_OFF


def _gather_costs(matrix, ids):
    """Gather per-slot cost rows by vocab id; empty slots (id < 0) get a zero row.

    matrix : (N_CARD_TYPES, _N_COST_FEATS) cost table
    ids    : (S,) int vocab ids (decoded via round(id_float * N_CARD_TYPES))
    returns: (S, _N_COST_FEATS) float32
    """
    safe = np.clip(ids, 0, N_CARD_TYPES - 1)
    rows = matrix[safe]
    return np.where((ids >= 0)[:, None], rows, 0.0).astype(np.float32)
# Start of bf_ability_costs block in the full obs vector
_BF_COST_START    = STATE_SIZE + 3 * MAX_ACTIONS + _HAND_COST_FEATS  # 34056
# Status offsets within a permanent slot
_OFF_POWER        = 0
_OFF_TOUGHNESS    = 1
_OFF_IS_TAPPED    = 2
_OFF_IS_ATTACKING = 3
_OFF_HAS_SICKNESS = 5
_OFF_IS_CREATURE  = 8    # new: 1.0 if this slot is a creature
_OFF_IS_LAND      = 9    # new: 1.0 if this slot is a land

# Vocab indices used for targeting decisions (mirror src/card_vocab.h)
_WASTELAND_VOCAB_IDX     = 10
_BASIC_LAND_IDS          = frozenset({0, 19})  # Mountain(0), Island(19)
_COUNTER_SPELL_VOCAB_IDS = frozenset({12, 13, 22})  # Force of Will(12), Daze(13), Counterspell(22)
_COUNTERSPELL_VOCAB_IDX  = 22
_BLUE_POOL_IDX           = 4   # obs[3 + 1]; mana pool is at obs[3:9], W/U/B/R/G/C, /10

# Doomsday deck card vocab indices (mirror src/card_vocab.h)
_DOOMSDAY_VOCAB_IDX      = 53
_THOUGHTSEIZE_VOCAB_IDX  = 54
_DARK_RITUAL_VOCAB_IDX   = 55
_LOTUS_PETAL_VOCAB_IDX   = 56
_LED_VOCAB_IDX           = 57
_THASSAS_ORACLE_VOCAB_IDX = 58
_PERSONAL_TUTOR_VOCAB_IDX = 59
_STREET_WRAITH_VOCAB_IDX = 60
_EDGE_OF_AUTUMN_VOCAB_IDX = 61
_CONSIDER_VOCAB_IDX      = 68
_DEEP_ANALYSIS_VOCAB_IDX = 70
_CAVERN_OF_SOULS_VOCAB_IDX = 67
_DURESS_VOCAB_IDX        = 69
_PONDER_VOCAB_IDX        = 11
_FORCE_OF_WILL_VOCAB_IDX = 12
_DAZE_VOCAB_IDX          = 13
_BRAINSTORM_VOCAB_IDX    = 24
_ONCE_UPON_A_TIME_VOCAB_IDX = 43

# Maverick (green/white) deck card vocab indices (mirror src/card_vocab.h)
_GREEN_SUNS_ZENITH_VOCAB_IDX = 35
_KEEN_EYED_CURATOR_VOCAB_IDX = 40
# Untapped creatures that tap for mana (count toward X for Green Sun's Zenith).
_MANA_DORK_IDS           = frozenset({30, 38, 42})  # Birds(30), Ignoble(38), Noble Hierarch(42)
# Lands that do NOT reliably produce mana when untapped, so they shouldn't count
# toward an affordable X: fetchlands (sacrifice for a land, no mana) and Gaea's
# Cradle (G per creature — zero with an empty board).
_UNRELIABLE_LAND_IDS     = frozenset({5, 6, 7, 8, 9, 52, 65, 66, 34})
# Cantrips/digging that let us keep a one-land opening hand (they find more lands)
_KEEP_ONE_LANDER_IDS     = frozenset({_PONDER_VOCAB_IDX, _BRAINSTORM_VOCAB_IDX,
                                      _ONCE_UPON_A_TIME_VOCAB_IDX})
_DISCARD_SPELL_IDS       = frozenset({_THOUGHTSEIZE_VOCAB_IDX, _DURESS_VOCAB_IDX})
_COUNTER_STRIP_IDS       = frozenset({_FORCE_OF_WILL_VOCAB_IDX, _DAZE_VOCAB_IDX})
_LED_DRAW_STACK_IDS      = frozenset({_STREET_WRAITH_VOCAB_IDX, _EDGE_OF_AUTUMN_VOCAB_IDX,
                                      _CONSIDER_VOCAB_IDX, _DEEP_ANALYSIS_VOCAB_IDX,
                                      _PONDER_VOCAB_IDX})
_DOOMSDAY_DECK_IDS       = frozenset({53, 54, 55, 56, 57, 58, 59, 60, 61, 67, 68, 69, 70})

_CAT_TOP_LIBRARY = 20  # choose card to put on top of library (Doomsday pile ordering)
_CAT_SHUFFLE     = 21  # shuffle choice (0 = don't shuffle, 1 = shuffle)


def _hand_has_card(obs: np.ndarray, vocab_idx: int) -> bool:
    """Check if the priority player's hand contains a card with the given vocab index."""
    for slot in range(MAX_HAND_SLOTS):
        if _slot_card_idx(obs, _HAND_START + slot) == vocab_idx:
            return True
    return False


def _obs_action_category(obs: np.ndarray, action: int) -> int:
    """Extract the raw action category int for the given action index from a full obs vector."""
    return int(round(obs[STATE_SIZE + action] * ACTION_CATEGORY_MAX))


def _obs_action_card_id(obs: np.ndarray, action: int) -> int:
    """Extract the card vocab index for the given action index from a full obs vector."""
    return int(round(obs[STATE_SIZE + MAX_ACTIONS + action] * N_CARD_TYPES))


def _obs_is_main_phase(obs: np.ndarray) -> bool:
    """Check if the current step is a main phase (FIRST_MAIN or SECOND_MAIN)."""
    # Step one-hot at obs[18:31], FIRST_MAIN=index 3 (obs[21]), SECOND_MAIN=index 10 (obs[28])
    return obs[21] > 0.5 or obs[28] > 0.5


def _self_has_draw_on_stack(obs: np.ndarray) -> bool:
    """Return True if self controls a cycling or draw ability/spell on the stack."""
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        ctrl_is_self = obs[base]
        idx = _slot_card_idx(obs, base + 1)
        if idx >= 0 and ctrl_is_self > 0.5 and idx in _LED_DRAW_STACK_IDS:
            return True
    return False


def _stack_is_empty(obs: np.ndarray) -> bool:
    """Return True if there are no items on the stack."""
    for i in range(12):
        base = _STACK_START + i * _STACK_SLOT_SIZE
        if _slot_card_idx(obs, base + 1) >= 0:
            return False
    return True


class ModelVsScriptedEnv(gym.Env):
    """Wraps RoboMageEnv so the model plays against a scripted agent.

    Each episode the model is randomly assigned to Player A or B; the scripted
    agent takes the other side.  Reward is negated when the model plays as B so
    it is always from the model's perspective (+1 win, -1 loss).
    """

    def __init__(self, model_deck: str | None = None, opp_deck: str | None = None,
                 opponent="scripted", **env_kwargs):
        super().__init__()
        # model_deck/opp_deck override deck_a/deck_b: decks are swapped each episode
        # to match which side the model is assigned to.
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._bo3 = env_kwargs.get("bo3", False)
        self._env = RoboMageEnv(**env_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = self._env.render_mode
        self._training_is_a = True
        self._opponent_below_10 = False
        self._last_obs = None
        # Scale applied to all shaping signals. Set externally (e.g. via SB3 callback)
        # to anneal shaping toward 0 as training matures.
        self.shaping_scale = 1.0
        # Opponent controller / pool. ``opponent`` may be an OpponentPool, an
        # already-built Controller, or a spec string ("scripted" by default,
        # which is byte-identical to the historical scripted agent).
        from opponents import make_controller
        # A pool exposes per-episode sampling (OpponentPool.sample /
        # LeaguePool.sample_episode); a Controller exposes .choose; anything else
        # is a spec string resolved into a Controller.
        if hasattr(opponent, "sample_episode") or hasattr(opponent, "sample"):
            self._opp_pool = opponent
            self._opp_controller = None
        elif hasattr(opponent, "choose"):
            self._opp_pool = None
            self._opp_controller = opponent
        else:
            self._opp_pool = None
            self._opp_controller = make_controller(opponent)
        self._opp_label = getattr(self._opp_controller, "label", "scripted")

    def reset(self, *, seed=None, options=None):
        self._training_is_a = bool(np.random.random() < 0.5)
        # Sample this episode's opponent (and, in league mode, its deck) BEFORE the
        # deck assignment below so the right opp_deck reaches the game process. A
        # plain OpponentPool keeps the fixed opp_deck and only picks a controller;
        # a LeaguePool also chooses which deck the opponent pilots this episode.
        if self._opp_pool is not None:
            if hasattr(self._opp_pool, "sample_episode"):
                self._opp_deck, self._opp_label, self._opp_controller = \
                    self._opp_pool.sample_episode()
            else:
                self._opp_label, self._opp_controller = self._opp_pool.sample()
        if self._model_deck is not None:
            self._env._deck_a = self._model_deck if self._training_is_a else self._opp_deck
            self._env._deck_b = self._opp_deck if self._training_is_a else self._model_deck
        self._opponent_below_10 = False
        self._last_obs = None
        self._decision_idx = 0
        self._episode_shaping = 0.0
        self._is_doomsday = self._model_deck is not None and "doomsday" in self._model_deck
        self._dd_placed_doomsday = False  # set when agent picks Doomsday in a TOP_LIBRARY choice
        self._dd_fired = set()  # tracks which DD shaping rewards have fired this game
        self._game_meta = {
            "model_is_a": self._training_is_a,
            "opp_deck": self._opp_deck or "unknown",
            "opp_type": self._opp_label,
        }
        obs, info = self._env.reset(seed=seed, options=options)
        obs, _reward, terminated, truncated, info, _shaping = self._skip_opponent_turns(
            obs, 0.0, False, False, info
        )
        self._opponent_below_10 = False  # reset after mulligan/setup turns
        self._last_obs = obs.copy()
        return obs, info

    def step(self, action: int):
        self._decision_idx += 1

        # Doomsday deck shaping: inspect the action the model just chose
        # using the pre-step observation (self._last_obs has the action metadata)
        if self._is_doomsday and self._last_obs is not None:
            cat = _obs_action_category(self._last_obs, action)
            card = _obs_action_card_id(self._last_obs, action)
            self._dd_pending_shaping = 0.0
            # Reward casting Doomsday (once per game)
            if cat == _CAT_CAST and card == _DOOMSDAY_VOCAB_IDX and "cast_dd" not in self._dd_fired:
                self._dd_pending_shaping += SHAPING_DD_CAST_DOOMSDAY
                self._dd_fired.add("cast_dd")
            # Reward casting Dark Ritual when Doomsday is in hand and it's a main phase (once per game)
            if (cat == _CAT_CAST and card == _DARK_RITUAL_VOCAB_IDX
                    and _hand_has_card(self._last_obs, _DOOMSDAY_VOCAB_IDX)
                    and _obs_is_main_phase(self._last_obs)
                    and "ritual" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_RITUAL_SETUP
                self._dd_fired.add("ritual")
            # Reward picking Thassa's Oracle during Doomsday pile building (once per game)
            if (cat == _CAT_TOP_LIBRARY and card == _THASSAS_ORACLE_VOCAB_IDX
                    and "pick_oracle" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_PICK_ORACLE
                self._dd_fired.add("pick_oracle")
            # Reward selecting Doomsday with Personal Tutor (once per game)
            if cat == _CAT_TOP_LIBRARY and card == _DOOMSDAY_VOCAB_IDX:
                if "tutor_dd" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_TUTOR_DOOMSDAY
                    self._dd_fired.add("tutor_dd")
                self._dd_placed_doomsday = True
            # Reward not shuffling after placing Doomsday on top (once per game)
            if cat == _CAT_SHUFFLE and action == 0 and self._dd_placed_doomsday:
                if "keep_dd" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_KEEP_DOOMSDAY
                    self._dd_fired.add("keep_dd")
                self._dd_placed_doomsday = False
            elif cat == _CAT_SHUFFLE:
                self._dd_placed_doomsday = False
            # Reward casting Thoughtseize/Duress (once per game)
            if (cat == _CAT_CAST and card in _DISCARD_SPELL_IDS
                    and "cast_discard" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_CAST_DISCARD
                self._dd_fired.add("cast_discard")
            # Reward selecting Force of Will or Daze with the discard choice (once per game)
            if (cat == _CAT_DISCARD and card in _COUNTER_STRIP_IDS
                    and "strip_counter" not in self._dd_fired):
                self._dd_pending_shaping += SHAPING_DD_STRIP_COUNTER
                self._dd_fired.add("strip_counter")
            # Reward cracking LED with a draw/cycling ability on the stack (once per game)
            if cat == _CAT_ACTIVATE and card == _LED_VOCAB_IDX:
                if _self_has_draw_on_stack(self._last_obs) and "led_draw" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_LED_WITH_DRAW
                    self._dd_fired.add("led_draw")
                elif _stack_is_empty(self._last_obs) and "led_empty" not in self._dd_fired:
                    self._dd_pending_shaping += SHAPING_DD_LED_EMPTY_STACK
                    self._dd_fired.add("led_empty")
        else:
            self._dd_pending_shaping = 0.0

        obs, reward, terminated, truncated, info = self._env.step(action)

        # Shaping: mana waste and mulligan penalties for the model player
        shaping_key = "shaping_a" if self._training_is_a else "shaping_b"
        shaping = info.get(shaping_key, 0.0)

        # Doomsday deck shaping (computed above from pre-step obs)
        shaping += self._dd_pending_shaping

        # Shaping: +0.2 the first time the opponent's life drops below 10.
        # obs is always from the priority player's perspective:
        #   model has priority → obs[9] = opponent life / 20
        #   opponent has priority → obs[0] = opponent ("self") life / 20
        if not (terminated or truncated) and not self._opponent_below_10:
            a_has_priority = obs[32] > 0.5
            model_has_priority = a_has_priority if self._training_is_a else not a_has_priority
            scripted_life = (obs[9] if model_has_priority else obs[0]) * 20.0
            if scripted_life < 10.0:
                self._opponent_below_10 = True
                shaping += SHAPING_OPPONENT_BELOW10

        if not (terminated or truncated):
            obs, reward, terminated, truncated, info, opp_shaping = self._skip_opponent_turns(
                obs, reward, terminated, truncated, info
            )
            shaping += opp_shaping

        # Potential-based hand/power advantage (skip for doomsday — not relevant to combo gameplan)
        if not self._is_doomsday and not (terminated or truncated) and self._last_obs is not None:
            phi_prev = max(0.0, self._last_obs[1] - self._last_obs[10]) * 10.0
            phi_curr = max(0.0, obs[1] - obs[10]) * 10.0
            shaping += SHAPING_HAND_ADV_PER_CARD * (phi_curr - phi_prev)

            # Potential-based board power advantage
            power_prev = _board_power_advantage(self._last_obs)
            power_curr = _board_power_advantage(obs)
            shaping += SHAPING_POWER_ADV_PER_PT * (power_curr - power_prev)

        shaping *= self.shaping_scale
        # In bo3, scale per-game shaping to 1/3 so total across 3 games
        # stays proportional to the match reward (1.0)
        if self._bo3:
            shaping /= 3.0
        # Clamp to remaining episode budget
        ep_cap = SHAPING_EPISODE_CAP_DOOMSDAY if self._is_doomsday else SHAPING_EPISODE_CAP
        remaining = ep_cap - self._episode_shaping
        floor = -(ep_cap + self._episode_shaping)
        shaping = max(floor, min(remaining, shaping))
        self._episode_shaping += shaping
        # Reset shaping budget and DD flags per game in bo3
        if self._bo3 and info.get("game_result", False):
            self._episode_shaping = 0.0
            self._dd_fired.clear()
            self._dd_placed_doomsday = False
        self._last_obs = obs.copy() if not (terminated or truncated) else None

        if not self._training_is_a:
            reward = -reward
        reward += shaping
        info["game_meta"] = self._game_meta
        info["decision_idx"] = self._decision_idx
        info["num_choices"] = self._env._num_choices
        if terminated or truncated:
            info["opp_deck"] = self._opp_deck or "unknown"
        return obs, reward, terminated, truncated, info

    def _skip_opponent_turns(self, obs, reward, terminated, truncated, info):
        """Resolve consecutive opponent turns with the scripted agent.

        Returns the updated (obs, reward, terminated, truncated, info) tuple plus
        the shaping reward accumulated across all opponent steps.
        """
        shaping_key = "shaping_a" if self._training_is_a else "shaping_b"
        shaping = 0.0
        while not (terminated or truncated) and (obs[32] > 0.5) != self._training_is_a:
            action = self._opp_controller.choose(obs, self._env._num_choices)
            obs, reward, terminated, truncated, info = self._env.step(action)

            shaping += info.get(shaping_key, 0.0)

            if not (terminated or truncated) and not self._opponent_below_10:
                a_has_priority = obs[32] > 0.5
                model_has_priority = a_has_priority if self._training_is_a else not a_has_priority
                scripted_life = (obs[9] if model_has_priority else obs[0]) * 20.0
                if scripted_life < 10.0:
                    self._opponent_below_10 = True
                    shaping += SHAPING_OPPONENT_BELOW10

        return obs, reward, terminated, truncated, info, shaping

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def update_opponent_weights(self, weights):
        """Push fresh PFSP/softmax weights into the league pool (if any).

        Called via ``vec_env.env_method`` from the PFSPCallback. ``env_method``
        resolves through the wrapper chain (get_wrapper_attr), so it reliably
        reaches this inner env — unlike ``set_attr``, which only sets the outer
        wrapper."""
        if self._opp_pool is not None and hasattr(self._opp_pool, "set_weights"):
            self._opp_pool.set_weights(weights)

    def set_shaping_scale(self, value):
        """Set the shaping-reward scale (read in ``step`` as ``self.shaping_scale``).

        MUST be called via ``vec_env.env_method('set_shaping_scale', v)``, not
        ``vec_env.set_attr('shaping_scale', v)``: under SubprocVecEnv/DummyVecEnv
        ``set_attr`` does a plain ``setattr`` on the OUTER Monitor wrapper, which
        gymnasium never forwards inward, so it would never reach this attribute.
        ``env_method`` resolves through the wrapper chain (get_wrapper_attr) to
        this inner env."""
        self.shaping_scale = float(value)

    def close(self):
        self._env.close()


# ── Self-play environment ─────────────────────────────────────────────────────

class SelfPlayEnv(gym.Env):
    """Self-play: the training model plays against a frozen saved checkpoint.

    Each episode one player role (A or B) is randomly assigned to the training
    model; the other is controlled by the frozen opponent.  The game engine emits
    observations from the priority player's perspective, so both sides always
    receive a perspective-normalised view without any mirroring.

    The training model always pilots ``model_deck`` and the frozen opponent always
    pilots ``opp_deck``.  Because models are per-deck generalists, the opponent's
    checkpoint is sampled from that deck's pilots ``{opp_deck}__v*.zip`` /
    ``{opp_deck}__final.zip`` — a generalist actually trained to play that deck
    (for a mirror match this is simply this run's own past selves).  The
    checkpoint is resampled every ``RELOAD_EVERY``
    episodes so it tracks the improving policy.  If no compatible checkpoint
    exists yet, the opponent falls back to the scripted agent (with a warning).
    """

    RELOAD_EVERY = 100  # episodes between opponent checkpoint reloads
    _model_cache = {}  # class-level: {path: (mtime, model)}

    def __init__(self, checkpoint_dir: str,
                 model_deck: str | None = None, opp_deck: str | None = None,
                 **env_kwargs):
        super().__init__()
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._env = RoboMageEnv(**env_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = self._env.render_mode
        self._checkpoint_dir = checkpoint_dir
        self._opponent = None    # loaded model, or None → scripted fallback
        self._episode_count = 0
        self._training_is_a = True
        self._opponent_loaded = False  # defer loading to first reset()
        self._scripted_fallback_warned = False  # warn once when no checkpoint found
        self._opp_mask = np.zeros(MAX_ACTIONS, dtype=bool)  # reusable mask buffer
        # Controller used when no compatible checkpoint exists (default GREEDY).
        from opponents import ScriptedController
        from scripted_agent import make_agent
        self._fallback_controller = ScriptedController(make_agent("scripted"))

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        self._episode_count += 1
        if not self._opponent_loaded or self._episode_count % self.RELOAD_EVERY == 0:
            self._reload_opponent()
            self._opponent_loaded = True

        self._training_is_a = bool(np.random.random() < 0.5)
        if self._model_deck is not None:
            self._env._deck_a = self._model_deck if self._training_is_a else self._opp_deck
            self._env._deck_b = self._opp_deck if self._training_is_a else self._model_deck
        self._decision_idx = 0
        opp_name = "scripted"
        if self._opponent is not None:
            opp_name = getattr(self, "_opp_checkpoint_path", "checkpoint")
        self._game_meta = {
            "model_is_a": self._training_is_a,
            "opp_deck": self._opp_deck or "unknown",
            "opp_type": opp_name,
        }
        obs, info = self._env.reset(seed=seed, options=options)

        obs, reward, terminated, truncated, info = self._handle_opponent_turns(
            obs, 0.0, False, False, info
        )
        if terminated or truncated:
            return np.zeros(OBS_SIZE, dtype=np.float32), info
        return self._training_obs(obs), info

    def step(self, action: int):
        self._decision_idx += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self._handle_opponent_turns(
                obs, reward, terminated, truncated, info
            )
        # Reward is from Player A's perspective; negate if training model plays as B.
        if not self._training_is_a:
            reward = -reward
        info["game_meta"] = self._game_meta
        info["decision_idx"] = self._decision_idx
        info["num_choices"] = self._env._num_choices
        return self._training_obs(obs), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()

    def set_shaping_scale(self, value):
        """Accept shaping-scale broadcasts uniformly (self-play applies no shaping)."""
        self.shaping_scale = float(value)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _training_has_priority(self, obs: np.ndarray) -> bool:
        """True when it is the training model's turn to act (raw obs)."""
        a_has_priority = obs[32] > 0.5
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
                self._opp_mask[:] = False
                self._opp_mask[:num_choices] = True
                action, _ = self._opponent.predict(opp_obs, action_masks=self._opp_mask, deterministic=False)
                action = int(action)
            else:
                # No compatible checkpoint — pilot the opponent with the scripted agent.
                action = self._fallback_controller.choose(opp_obs, num_choices)
            obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info

    def _reload_opponent(self):
        """Sample a frozen checkpoint trained to pilot the opponent's deck.

        The frozen opponent plays ``opp_deck``, so we look for a model saved to
        pilot it: the v2 deck-pilot snapshots ``{opp_deck}__v*.zip`` /
        ``{opp_deck}__final.zip``.  If no compatible checkpoint exists, fall back
        to the scripted agent and warn (once)."""
        from opponents import deck_snapshots
        deck = self._opp_deck or self._model_deck
        files = deck_snapshots(deck, self._checkpoint_dir)
        if not files:
            if not self._scripted_fallback_warned:
                print(f"[self-play] WARNING: no '{deck}__v*.zip' / '{deck}__final.zip' "
                      f"checkpoint in {self._checkpoint_dir}; opponent falling back to "
                      f"the scripted agent.")
                self._scripted_fallback_warned = True
            self._opponent = None
            return
        path = str(np.random.choice(files))
        self._opp_checkpoint_path = path
        try:
            mtime = os.path.getmtime(path)
            cached = SelfPlayEnv._model_cache.get(path)
            if cached is not None and cached[0] == mtime:
                self._opponent = cached[1]
                return
            try:
                from sb3_contrib import MaskablePPO as _PPO
            except ImportError:
                from stable_baselines3 import PPO as _PPO
            model = _PPO.load(path, device="cpu")
            SelfPlayEnv._model_cache[path] = (mtime, model)
            self._opponent = model
        except Exception:
            print(f"[self-play] WARNING: failed to load opponent checkpoint {path}; "
                  f"falling back to the scripted agent.")
            self._opponent = None  # fall back to scripted if load fails


class FixedModelEnv(gym.Env):
    """Train against a fixed opponent model that never changes.

    Like SelfPlayEnv but the opponent is loaded once from a specific path
    and is never reloaded or sampled from checkpoints.
    """

    def __init__(self, opp_model_path: str,
                 model_deck: str | None = None, opp_deck: str | None = None,
                 **env_kwargs):
        super().__init__()
        self._model_deck = model_deck
        self._opp_deck = opp_deck
        self._env = RoboMageEnv(**env_kwargs)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.render_mode = self._env.render_mode
        self._training_is_a = True
        self._decision_idx = 0

        # Load fixed opponent once
        try:
            from sb3_contrib import MaskablePPO as _PPO
        except ImportError:
            from stable_baselines3 import PPO as _PPO
        self._opponent = _PPO.load(opp_model_path, device="cpu")
        self._opp_model_path = opp_model_path
        self._opp_mask = np.zeros(MAX_ACTIONS, dtype=bool)  # reusable mask buffer

    def reset(self, *, seed=None, options=None):
        self._training_is_a = bool(np.random.random() < 0.5)
        if self._model_deck is not None:
            self._env._deck_a = self._model_deck if self._training_is_a else self._opp_deck
            self._env._deck_b = self._opp_deck if self._training_is_a else self._model_deck
        self._decision_idx = 0
        self._game_meta = {
            "model_is_a": self._training_is_a,
            "opp_deck": self._opp_deck or "unknown",
            "opp_type": self._opp_model_path,
        }
        obs, info = self._env.reset(seed=seed, options=options)

        obs, reward, terminated, truncated, info = self._handle_opponent_turns(
            obs, 0.0, False, False, info
        )
        if terminated or truncated:
            return np.zeros(OBS_SIZE, dtype=np.float32), info
        return obs, info

    def step(self, action: int):
        self._decision_idx += 1
        obs, reward, terminated, truncated, info = self._env.step(action)
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = self._handle_opponent_turns(
                obs, reward, terminated, truncated, info
            )
        if not self._training_is_a:
            reward = -reward
        info["game_meta"] = self._game_meta
        info["decision_idx"] = self._decision_idx
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._env.action_masks()

    def close(self):
        self._env.close()

    def set_shaping_scale(self, value):
        """Accept shaping-scale broadcasts uniformly (fixed-model applies no shaping)."""
        self.shaping_scale = float(value)

    def _training_has_priority(self, obs: np.ndarray) -> bool:
        a_has_priority = obs[32] > 0.5
        return a_has_priority if self._training_is_a else not a_has_priority

    def _handle_opponent_turns(self, obs, reward, terminated, truncated, info):
        while not (terminated or truncated) and not self._training_has_priority(obs):
            num_choices = self._env._num_choices
            self._opp_mask[:] = False
            self._opp_mask[:num_choices] = True
            action, _ = self._opponent.predict(obs, action_masks=self._opp_mask, deterministic=False)
            action = int(action)
            obs, reward, terminated, truncated, info = self._env.step(action)
        return obs, reward, terminated, truncated, info


# ── Back-compat re-export ───────────────────────────────────────────────────
# The scripted agent now lives in scripted_agent.py.  Importers historically do
# `from env import scripted_action`, so re-export it here.  This import runs at
# the bottom of env.py (after every constant/helper scripted_agent needs is
# defined), so when env is imported first the chain resolves cleanly.  If
# scripted_agent is imported first, scripted_action is not yet bound when this
# line runs, so fall back to a lazy wrapper that resolves it on first call.
try:
    from scripted_agent import scripted_action  # noqa: E402,F401
except ImportError:
    def scripted_action(obs, num_choices):  # noqa: E402
        from scripted_agent import scripted_action as _sa
        return _sa(obs, num_choices)
