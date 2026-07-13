"""
Analyze a trained RoboMage model by simulating games and inspecting its play.

Every command loads a checkpoint, plays N games against a chosen opponent
(another checkpoint or the scripted agent) for a specific matchup, and analyzes
the resulting per-decision traces (full observations, value estimates V(s), and
policy probabilities). The older offline .rmrec recording-file commands were
removed; this live model-sim path is now the single source.

There are two commands:
    python analysis.py report <model.zip> --opponent scripted [--n-games 50]
        Run the standard battery once and emit a single self-contained HTML
        report (headless, non-interactive — for CI / sharing). Exits when done.
    python analysis.py interactive <model.zip> --opponent scripted [--n-games 20]
        Simulate games, then open the REPL below. This is the only mode with a
        live env, so 'run' and 'whatif' work only here. The REPL supersets every
        per-analysis view (cardvalue, shap, regret, entropy, consistency, …), so
        the former standalone analysis subcommands were folded into it.

Charts save as PNGs under --out (default train/analysis_out/) so the tool works
headless; pass --show to also open a GUI window. The REPL also prints terminal
sparklines/bars so the common views need no display at all.

Interactive session commands (via 'interactive'):
    list                  list all games
    replay <N> [-v]       per-decision trace for game N; -v adds zones, chosen
                          action, and interleaved opponent actions
    boardstate <N> [step] full board + decision detail; enters GDB-style stepping mode
    summary               win/loss/draw stats
    cardvalue [N]         rank cards by importance (ΔV, priority, win-rate)
    swings [N]            top N in-game value-function swings (bo3 boundaries excluded)
    boundaries            V(s) across bo3 game transitions (result-pricing vs re-anchoring)
    matchcal              per match/game: V at game start vs empirical remaining return by score
    shap                  run SHAP analysis on collected data
    regret [N]            policy regret analysis (top N high-regret decisions)
    entropy               policy entropy by game phase and board state
    consistency [N]       decision consistency for similar states (top N pairs)
    targeting             self vs opp targeting, hold vs cast analysis
    sideboard             sideboard decisions by each agent (bo3)
    sbvalue               sideboard preference: take-rate, confidence, ΔV, ΔWR per card,
                          + net post-board game-WR impact, fetchlands fungible (bo3)
    whatif <N> <step> [k] counterfactual: branch chosen + top-k actions from the
                          same seed, roll each to the end, compare result/V(s)
    calibration           V(s) at game start vs actual win rate
    turning               find the permanent zero-crossing ('point of no return')
    clusters              classify games by V(s) curve shape (archetypes)
    chart <N>             value curve plot for game N
    chart swings [N]      value curve plots for top N swing games
    chart cardvalue [N]   diverging bar chart of per-card ΔV
    chart sbvalue [N]     sideboard swap preference + net ΔWR bars (bo3)
    chart shap            SHAP summary plot
    chart calibration     calibration curve plot
    chart turning         turning point distribution plot
    chart clusters        overlay V(s) curves by archetype
    run <N>               simulate N more games (interactive command only)
    quit                  exit
"""

import argparse
import glob
import re
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Action-category display names come from the generated C++ enum tables
# (train/gen_enums.py), the single source of truth.
from _enums import (
    _CAT_NAMES, _REF_NAMES, REF_ZONE_MAX,
    CAT_PASS_PRIORITY, CAT_ACTIVATE_ABILITY, CAT_CAST_SPELL, CAT_SELECT_TARGET,
    CAT_PLAY_LAND, CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE)
from card_costs import _VOCAB_NAMES, N_CARD_TYPES
import decode
import viz
# CLI definitions come from cli_spec.py (single source shared with the TUI).
from cli_spec import ANALYSIS_TOOL, apply_to_parser
from env import (ACTION_CATEGORY_MAX, RoboMageEnv,
                 STATE_SIZE, MAX_ACTIONS, BINARY, BO3_GAME_WIN_REWARD,
                 _HAND_START, _MATCH_CTX_START, _LIBRARY_CTX_START,
                 _SELF_PERM_START, _PERM_SLOTS as _ENV_PERM_SLOTS, _PERM_SLOT_SIZE,
                 _GY_START, _GY_SLOTS_TOTAL, _GY_SLOT_SIZE,
                 _STACK_START as _ENV_STACK_START, _STACK_SLOTS as _ENV_STACK_SLOTS,
                 _STACK_SLOT_SIZE, _HAND_SLOT_SIZE, _slot_card_idx,
                 _PERM_CARD_OFF, _CUR_TURN_IDX,
                 _OFF_IS_PHASED_OUT, N_ENTITY_REF_SLOTS,
                 _SELF_BLOCK_START, _OPP_BLOCK_START,
                 _PB_LIFE, _PB_HAND_CT, _PB_POISON, _PB_MANA, _PB_ENERGY,
                 _STEP_ONEHOT_START, _STEP_ONEHOT_SIZE,
                 _IS_ACTIVE_IDX, _SELF_IS_A_IDX, _STACK_SIZE_IDX)


def _resolve_card_name(cid):
    """Resolve a card vocab index to a display name. Negative/null → 'Player'
    (a player target carries no card id)."""
    if cid < 0:
        return "Player"
    return decode.card_index_to_name(cid)


# ── SHAP / value-function analysis ────────────────────────────────────────────

# Human-readable feature names extracted from the raw observation vector.
# These are the features SHAP will explain.
_INTERP_FEATURE_NAMES = [
    "self_life", "opp_life", "life_diff",
    "self_mana_W", "self_mana_U", "self_mana_B",
    "self_mana_R", "self_mana_G", "self_mana_C", "self_total_mana",
    "opp_mana_W", "opp_mana_U", "opp_mana_B",
    "opp_mana_R", "opp_mana_G", "opp_mana_C", "opp_total_mana",
    "self_hand_size", "opp_hand_size",
    "self_creatures", "self_lands", "self_other_perms",
    "opp_creatures", "opp_lands", "opp_other_perms",
    "creature_diff", "land_diff",
    "self_tapped_lands", "opp_tapped_lands",
    "self_attacking", "self_blocking",
    "opp_attacking", "opp_blocking",
    "self_total_power", "opp_total_power", "power_diff",
    "self_total_toughness", "opp_total_toughness",
    "self_gy_size", "opp_gy_size",
    "stack_size",
    "self_stack_count", "opp_stack_count",
    "stack_spell_count", "stack_ability_count",
    "is_active_player",
    "step_untap", "step_upkeep", "step_draw",
    "step_first_main", "step_begin_combat",
    "step_declare_atk", "step_declare_blk",
    "step_first_strike", "step_combat_dmg",
    "step_end_combat", "step_second_main",
    "step_end", "step_cleanup",
    "self_library_size", "opp_library_size", "is_post_board",
    "is_sideboard",
    "turn",
]

# Name → index into an interp feature vector. Always index feats through this
# (feat[_FEAT["self_creatures"]]) — hardcoded integers go stale when a feature
# is inserted upstream.
_FEAT = {name: i for i, name in enumerate(_INTERP_FEATURE_NAMES)}

# Permanent slot layout (derived from env.py / machine_io.h). Card identity is a
# single normalized id float per slot; decode via _slot_card_idx (round(val*N)).
_PERM_START   = _SELF_PERM_START          # 36 (self slots first, then opponent)
_PERM_SLOTS   = _ENV_PERM_SLOTS * 2       # 96 = 48 self + 48 opponent
_PERM_SLOT_SZ = _PERM_SLOT_SIZE           # 38 (status + counters + refs + keywords + chosen-name id + returnable-exile id + card id)
_SELF_PERM_SLOTS = _ENV_PERM_SLOTS        # 48: slots 0-47 = self, 48-95 = opponent
# Per-slot offsets: power(0), toughness(1), tapped(2), attacking(3), blocking(4),
#                   sickness(5), damage(6), controller_is_self(7), is_creature(8),
#                   is_land(9), loyalty(10), then the enriched fields — counters
#                   (11-12), attachment/combat refs (13-16), is_blocked(17),
#                   is_phased_out(18 = _OFF_IS_PHASED_OUT), keyword multi-hot
#                   (19-34), chosen-name id(35), returnable-exile id(36), and
#                   card_id(37 = _PERM_CARD_OFF from env.py, LAST)
_PERM_LOYALTY_OFF = 10
_GY_START_OBS    = _GY_START
_GY_SLOTS        = _GY_SLOTS_TOTAL        # 128
_GY_SLOT_SZ      = _GY_SLOT_SIZE          # 1
_GY_SELF_SLOTS   = _GY_SLOTS_TOTAL // 2   # slots 0-63 = self GY, 64-127 = opp GY

# Stack layout: 12 slots x 37 floats. Per slot: controller_is_self(1), card id(1),
# is_spell(1), x_or_amount(1), cast qualifiers(7), chosen-mode multi-hot(6),
# 4 announced-target sub-slots x 5 floats (present, is_player, ctrl, slot_ref, card id).
_STACK_START   = _ENV_STACK_START
_STACK_SLOTS   = _ENV_STACK_SLOTS
_STACK_SLOT_SZ = _STACK_SLOT_SIZE         # 37

# Hand layout: 10 slots x 1 float (imported _HAND_START from env.py)
_HAND_SLOTS = 10

# Mana color labels
_MANA_COLORS = ["W", "U", "B", "R", "G", "C"]


def _extract_interpretable(obs):
    """Extract human-readable features from a raw observation vector."""
    f = np.zeros(len(_INTERP_FEATURE_NAMES), dtype=np.float32)
    i = 0

    # Player stats (denormalize: life * 20, mana * 10)
    self_life = obs[_SELF_BLOCK_START + _PB_LIFE] * 20.0
    opp_life  = obs[_OPP_BLOCK_START + _PB_LIFE] * 20.0
    f[i] = self_life;     i += 1  # self_life
    f[i] = opp_life;      i += 1  # opp_life
    f[i] = self_life - opp_life; i += 1  # life_diff

    # Self mana — player block is life, hand_ct, poison, mana[W,U,B,R,G,C], energy
    _self_mana = _SELF_BLOCK_START + _PB_MANA
    for j in range(6):
        f[i] = obs[_self_mana + j] * 10.0; i += 1
    f[i] = sum(obs[_self_mana + j] * 10.0 for j in range(6)); i += 1  # total mana

    # Opp mana (same block layout, offset by _OPP_BLOCK_START)
    _opp_mana = _OPP_BLOCK_START + _PB_MANA
    for j in range(6):
        f[i] = obs[_opp_mana + j] * 10.0; i += 1
    f[i] = sum(obs[_opp_mana + j] * 10.0 for j in range(6)); i += 1  # total mana

    # Hand size (count non-empty hand slots for self; opp comes from player block)
    hand_count = 0
    for slot in range(10):
        if _slot_card_idx(obs, _HAND_START + slot * _HAND_SLOT_SIZE) >= 0:
            hand_count += 1
    f[i] = hand_count;        i += 1  # self_hand_size
    f[i] = obs[_OPP_BLOCK_START + _PB_HAND_CT] * 10.0;    i += 1  # opp_hand_size

    # Permanent counts and stats
    self_creatures = self_lands = self_other = 0
    opp_creatures = opp_lands = opp_other = 0
    self_tapped_lands = opp_tapped_lands = 0
    self_attacking = self_blocking = 0
    opp_attacking = opp_blocking = 0
    self_power = self_toughness = 0.0
    opp_power = opp_toughness = 0.0

    for slot in range(_PERM_SLOTS):
        base = _PERM_START + slot * _PERM_SLOT_SZ
        power     = obs[base + 0] * 10.0
        toughness = obs[base + 1] * 10.0
        tapped    = obs[base + 2] > 0.5
        attacking = obs[base + 3] > 0.5
        blocking  = obs[base + 4] > 0.5
        is_creat  = obs[base + 8] > 0.5
        is_land   = obs[base + 9] > 0.5

        # Check if slot is occupied (card id present)
        if _slot_card_idx(obs, base + _PERM_CARD_OFF) < 0:
            continue
        # Phased-out permanents are serialized (so the model can anticipate the
        # phase-in) but the rules treat them as nonexistent (CR 702.26e) — keep
        # them out of the board stats so they don't inflate counts/power sums.
        if obs[base + _OFF_IS_PHASED_OUT] > 0.5:
            continue

        is_self = slot < _SELF_PERM_SLOTS
        if is_self:
            if is_creat:
                self_creatures += 1
                self_power += power
                self_toughness += toughness
                if attacking: self_attacking += 1
                if blocking:  self_blocking += 1
            elif is_land:
                self_lands += 1
                if tapped: self_tapped_lands += 1
            else:
                self_other += 1
        else:
            if is_creat:
                opp_creatures += 1
                opp_power += power
                opp_toughness += toughness
                if attacking: opp_attacking += 1
                if blocking:  opp_blocking += 1
            elif is_land:
                opp_lands += 1
                if tapped: opp_tapped_lands += 1
            else:
                opp_other += 1

    f[i] = self_creatures; i += 1
    f[i] = self_lands;     i += 1
    f[i] = self_other;     i += 1
    f[i] = opp_creatures;  i += 1
    f[i] = opp_lands;      i += 1
    f[i] = opp_other;      i += 1
    f[i] = self_creatures - opp_creatures; i += 1
    f[i] = self_lands - opp_lands;         i += 1
    f[i] = self_tapped_lands;  i += 1
    f[i] = opp_tapped_lands;   i += 1
    f[i] = self_attacking;     i += 1
    f[i] = self_blocking;      i += 1
    f[i] = opp_attacking;      i += 1
    f[i] = opp_blocking;       i += 1
    f[i] = self_power;        i += 1
    f[i] = opp_power;         i += 1
    f[i] = self_power - opp_power; i += 1
    f[i] = self_toughness;    i += 1
    f[i] = opp_toughness;     i += 1

    # Graveyard sizes
    self_gy = opp_gy = 0
    for slot in range(_GY_SLOTS):
        base = _GY_START_OBS + slot * _GY_SLOT_SZ
        if _slot_card_idx(obs, base) >= 0:
            if slot < 64:
                self_gy += 1
            else:
                opp_gy += 1
    f[i] = self_gy; i += 1
    f[i] = opp_gy;  i += 1

    # Stack size
    f[i] = obs[_STACK_SIZE_IDX] * 10.0; i += 1

    # Stack contents (12 slots: controller_is_self, card id, is_spell, modes, targets)
    self_stack = opp_stack = 0
    stack_spells = stack_abilities = 0
    for slot in range(_STACK_SLOTS):
        base = _STACK_START + slot * _STACK_SLOT_SZ
        # Slot occupied iff the card id is present (offset 1)
        if _slot_card_idx(obs, base + 1) < 0:
            continue
        if obs[base] > 0.5:
            self_stack += 1
        else:
            opp_stack += 1
        if obs[base + 2] > 0.5:  # is_spell (fixed offset; targets/modes follow)
            stack_spells += 1
        else:
            stack_abilities += 1
    f[i] = self_stack;      i += 1
    f[i] = opp_stack;       i += 1
    f[i] = stack_spells;    i += 1
    f[i] = stack_abilities; i += 1

    # Is active player
    f[i] = 1.0 if obs[_IS_ACTIVE_IDX] > 0.5 else 0.0; i += 1

    # Step one-hot
    for j in range(_STEP_ONEHOT_SIZE):
        f[i] = obs[_STEP_ONEHOT_START + j]; i += 1

    # Library counts and post-board flag (mirror env.py _LIBRARY_CTX_START)
    f[i] = obs[_LIBRARY_CTX_START]     * 60.0; i += 1  # self_library_size
    f[i] = obs[_LIBRARY_CTX_START + 1] * 60.0; i += 1  # opp_library_size
    f[i] = 1.0 if obs[_LIBRARY_CTX_START + 2] > 0.5 else 0.0; i += 1  # is_post_board
    f[i] = 1.0 if obs[_MATCH_CTX_START + 3] > 0.5 else 0.0; i += 1  # is_sideboard

    # Current game turn (obs stores turn / 50, mirror machine_io.h TURN_NORMALIZER)
    f[i] = obs[_CUR_TURN_IDX] * 50.0; i += 1

    return f


_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
_SNAPSHOT_VER_RE = re.compile(r"__v(\d+)\.zip$")


def _resolve_model_path(path):
    """Resolve a model argument to a checkpoint path.

    Accepts an explicit path, the literal ``scripted``, or the generalist stem
    ``gen`` (→ ``checkpoints/gen__final.zip``, else the newest
    ``checkpoints/gen__v{steps}.zip`` snapshot). The deck a model pilots is
    supplied separately via ``--deck-a``/``--deck-b``, never inferred here.
    Mirrors train.py's ``_resolve_model``.
    """
    if path == "scripted" or os.path.exists(path):
        return path
    # 'gen' → 'gen__final.zip' or newest 'gen__v*.zip'.
    final = os.path.join(_CHECKPOINTS_DIR, f"{path}__final.zip")
    if os.path.exists(final):
        return final
    snaps = glob.glob(os.path.join(_CHECKPOINTS_DIR, f"{path}__v*.zip"))
    if snaps:
        def _ver(p):
            m = _SNAPSHOT_VER_RE.search(p)
            return int(m.group(1)) if m else -1
        return max(snaps, key=_ver)
    # Legacy fallbacks.
    for cand in (os.path.join(_CHECKPOINTS_DIR, path),
                 os.path.join(_CHECKPOINTS_DIR, f"{path}.zip"),
                 os.path.join(_CHECKPOINTS_DIR, f"{path}_final.zip")):
        if os.path.exists(cand):
            return cand
    return path  # let the loader raise a meaningful error


# ── AlphaZero (AZNet) checkpoint support ──────────────────────────────────────
#
# analysis.py only ever touches a model through three surfaces: the value head
# (``policy.predict_values``), the masked action distribution
# (``policy.get_distribution``), and ``model.predict`` (via ModelController).
# An AZNet natively provides all three — a tanh outcome value and a masked-softmax
# policy — so the entire existing battery (cardvalue, targeting, calibration,
# turning, clusters, swings, regret, entropy, consistency, SHAP surrogate, whatif,
# run) transfers to an AZ checkpoint by wrapping it in a thin adapter that mimics
# that subset of the sb3 policy API. No analysis fundamentally needs PPO-only
# internals, so none has to be skipped; the one caveat is that the AZ value is a
# bounded game-outcome estimate in [-1, 1] (tanh), not the PPO shaped-return
# critic, so absolute V(s) magnitudes are not directly comparable across the two.


def _is_az_model_spec(spec):
    """True if ``spec`` names an AZNet checkpoint rather than a PPO ``.zip``.

    Recognizes an explicit ``az:``/``azraw:`` prefix, a bare ``.pt`` path, or a
    deck shorthand that (a) does NOT resolve to a PPO checkpoint and (b) DOES
    resolve to an AZ checkpoint under checkpoints/az/. The PPO-first ordering
    keeps normal checkpoints on the unchanged path (and avoids importing az_net
    for them)."""
    if not isinstance(spec, str):
        return False
    s = spec.strip()
    low = s.lower()
    if low.startswith("az:") or low.startswith("azraw:"):
        return True
    if s.endswith(".pt"):
        return True
    if _resolve_model_path(s) != s:  # a PPO checkpoint resolved — not AZ
        return False
    try:
        from az_net import resolve_az_checkpoint
    except Exception:
        return False
    return resolve_az_checkpoint(s) is not None


def _effective_bo3(args) -> bool:
    """Whether to simulate in bo3. AZ/MCTS models are trained and gated in bo3
    (Phase 1a), so analysing one defaults to bo3 even without ``--bo3``; a
    scripted/PPO model keeps ``--bo3`` opt-in. The explicit flag always forces
    bo3 on."""
    if getattr(args, "bo3", False):
        return True
    model = getattr(args, "model", None)
    if isinstance(model, str) and model.strip().lower().startswith("mcts:"):
        return True
    return _is_az_model_spec(model)


def _az_spec_base(spec):
    """Strip an ``az:``/``azraw:`` prefix from a model spec (else return it)."""
    s = spec.strip()
    for pfx in ("az:", "azraw:"):
        if s.lower().startswith(pfx):
            return s[len(pfx):].strip()
    return s


def _resolve_any_path(spec):
    """Resolve a model spec to a checkpoint path, AZ-aware (for deck inference /
    display). AZ specs resolve via resolve_az_checkpoint (falling back to the PPO
    checkpoint used for a warm-start); everything else via _resolve_model_path."""
    if _is_az_model_spec(spec):
        from az_net import resolve_az_checkpoint
        base = _az_spec_base(spec)
        return resolve_az_checkpoint(base) or _resolve_model_path(base)
    return _resolve_model_path(spec)


class _AZDistribution:
    """Stand-in for an sb3 action distribution: exposes ``.probs`` like
    MaskableCategorical so ``_get_policy_probs`` reads it unchanged."""

    def __init__(self, probs):
        self.probs = probs


class _AZDistributionWrap:
    """Mirror of ``get_distribution``'s return: a ``.distribution`` with ``.probs``."""

    def __init__(self, probs):
        self.distribution = _AZDistribution(probs)


class _AZPolicyAdapter:
    """Adapts an AZNet to the sb3 ``policy`` subset analysis.py calls, both taking
    a torch batch tensor: ``predict_values(obs_t)`` and
    ``get_distribution(obs_t, action_masks=)``."""

    def __init__(self, net):
        import torch
        self._torch = torch
        self._net = net.eval()

    def predict_values(self, obs_t):
        torch = self._torch
        b = obs_t.shape[0]
        mask = torch.ones(b, MAX_ACTIONS, dtype=torch.bool)
        with torch.no_grad():
            _, value = self._net(obs_t, mask)
        return value.reshape(-1, 1)  # so .item() works for a batch of 1

    def get_distribution(self, obs_t, action_masks=None):
        torch = self._torch
        b = obs_t.shape[0]
        if action_masks is None:
            mask = torch.ones(b, MAX_ACTIONS, dtype=torch.bool)
        else:
            mask = torch.as_tensor(np.asarray(action_masks, dtype=bool))
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._net(obs_t, mask)
            probs = torch.softmax(logits, dim=-1)
        return _AZDistributionWrap(probs)


class _AZModelAdapter:
    """Drop-in for a MaskablePPO model across analysis.py: a ``.policy`` with
    predict_values/get_distribution and a ``.predict`` for ModelController.

    The value is the AZ tanh outcome estimate in [-1, 1] (a bounded game-result
    prediction), NOT the PPO shaped-return critic."""

    is_az = True

    def __init__(self, net):
        self._net = net
        self.policy = _AZPolicyAdapter(net)

    def predict(self, obs, action_masks=None, deterministic=True):
        import torch
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        if action_masks is None:
            mask = torch.ones(1, MAX_ACTIONS, dtype=torch.bool)
        else:
            mask = torch.as_tensor(np.asarray(action_masks, dtype=bool)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self._net(obs_t, mask)
            action = int(torch.argmax(logits[0]).item())
        return action, None


def _load_az_analysis_model(spec):
    """Load an AZNet for analysis from a model spec. Returns (adapter, path).

    Resolves an AZ checkpoint (az:/azraw: prefix, ``.pt`` path, or deck shorthand)
    via resolve_az_checkpoint; when only a PPO checkpoint exists it warm-starts an
    AZNet from it (``from_ppo``) so an ``az:`` spec still yields an AZNet-shaped
    model."""
    from az_net import load_az, from_ppo, resolve_az_checkpoint
    base = _az_spec_base(spec)
    az = resolve_az_checkpoint(base)
    if az is not None:
        return _AZModelAdapter(load_az(az)), az
    ppo_path = _resolve_model_path(base)
    print(f"No AZ checkpoint for {base!r}; warm-starting an AZNet from PPO {ppo_path}")
    return _AZModelAdapter(from_ppo(ppo_path)), ppo_path


def _load_model_and_env(args):
    """Load model, set up env with the right decks and opponent. Returns (model, env, opp_model_or_none)."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError:
        from stable_baselines3 import PPO as MaskablePPO

    binary = getattr(args, "binary", BINARY)

    model_path = _resolve_any_path(args.model)

    # Deck resolution. A checkpoint no longer encodes a deck — there is one
    # generalist that pilots whatever deck it is told to. So the model's deck
    # (deck_a) is REQUIRED via --deck-a. The opponent deck (deck_b) is required
    # too, except a scripted opponent defaults to a mirror match (deck_b = deck_a).
    deck_a = getattr(args, "deck_a", None)
    deck_b = getattr(args, "deck_b", None)
    if not deck_a:
        print("Model deck is required — a checkpoint no longer encodes a deck; "
              "the generalist pilots whatever you name. Pass --deck-a.",
              file=sys.stderr)
        sys.exit(1)
    if not deck_b:
        if args.opponent == "scripted":
            deck_b = deck_a  # no opponent deck given; mirror by default
            print(f"No --deck-b given for scripted opponent; defaulting to a mirror "
                  f"match (opponent plays {deck_b}). Pass --deck-b for a different matchup.")
        else:
            print("Opponent deck is required for a model opponent — a checkpoint no "
                  "longer encodes a deck. Pass --deck-b.", file=sys.stderr)
            sys.exit(1)

    # Write the resolved decks back so downstream consumers (e.g. the report
    # title) see the actual decks even when they were inferred, not just given.
    args.deck_a, args.deck_b = deck_a, deck_b

    if _is_az_model_spec(args.model):
        model, _ = _load_az_analysis_model(args.model)
    else:
        model = MaskablePPO.load(model_path)
    opp_model = None
    if args.opponent != "scripted":
        if _is_az_model_spec(args.opponent):
            opp_model, _ = _load_az_analysis_model(args.opponent)
        else:
            opp_model = MaskablePPO.load(_resolve_model_path(args.opponent))

    env = RoboMageEnv(binary_path=binary, deck_a=deck_a, deck_b=deck_b,
                      bo3=_effective_bo3(args))
    return model, env, opp_model


def _get_policy_probs(model, obs, num_choices):
    """Get masked action probability distribution from the policy.

    Returns numpy array of shape (num_choices,) with probabilities for each
    legal action.
    """
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    # Go through the policy's own get_distribution rather than reading
    # action_net directly — PerActionMaskablePolicy computes logits with its
    # action scorer and leaves the stock action_net unused/untrained.
    mask = np.zeros((1, MAX_ACTIONS), dtype=bool)
    mask[0, :num_choices] = True
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t, action_masks=mask)
        probs = dist.distribution.probs[0].cpu().numpy()
    return probs[:num_choices].astype(np.float64)


def _reset_for_game(env, model_is_a, engine_seed=None):
    """Reset `env` for a game the model plays as side `model_is_a`.

    Mirrors the deck-swap the collector uses so the model always pilots its own
    deck: when the model is Player B the decks are swapped across the reset (the
    engine process is spawned with the swapped decks, then the env attributes are
    restored). Passing `engine_seed` forces that exact engine --seed so a
    previously-collected game replays byte-identically. Returns
    (obs, engine_seed_used).
    """
    opts = {"engine_seed": engine_seed} if engine_seed is not None else None
    if model_is_a:
        obs, _ = env.reset(options=opts)
    else:
        old_a, old_b = env._deck_a, env._deck_b
        env._deck_a, env._deck_b = old_b, old_a
        try:
            obs, _ = env.reset(options=opts)
        finally:
            env._deck_a, env._deck_b = old_a, old_b
    return obs, env.last_engine_seed


def _controllers_for(model, opp_model, model_is_a):
    """Build (ctrl_a, ctrl_b, ctrl_model) for a model-vs-opponent game.

    The opponent is a deterministic ModelController when ``opp_model`` is
    given, else the scripted HARD agent — the same seats the hand-rolled loops
    used before this moved onto runner.drive_game.
    """
    from opponents import ModelController, ScriptedController
    from scripted_agent import make_agent

    ctrl_model = ModelController(model, label="Model", deterministic=True)
    if opp_model is not None:
        ctrl_opp = ModelController(opp_model, label="Opp", deterministic=True)
    else:
        ctrl_opp = ScriptedController(make_agent("scripted"), label="Scripted")
    ctrl_a, ctrl_b = ((ctrl_model, ctrl_opp) if model_is_a
                      else (ctrl_opp, ctrl_model))
    return ctrl_a, ctrl_b, ctrl_model


def _collect_game_traces(model, env, opp_model, n_games, verbose=True):
    """Play n_games and collect per-step (obs, value, action) traces.

    Returns a list of game dicts:
      { "observations": [obs, ...],
        "values": [float, ...],
        "interp_features": [array, ...],
        "actions": [int, ...],       # model's chosen action index at each step
        "num_choices": [int, ...],   # number of legal actions at each step
        "opp_actions": [{"desc": str, "before_model_step": int}, ...],
        "engine_seed": int,          # engine --seed of the played game (replay key)
        "full_actions": [int, ...],  # every action index fed to env.step, in order
        "prefix_len": [int, ...],    # per model step: # of full_actions before it
        "result": float,  # +1 win, -1 loss from model perspective
        "model_is_a": bool }

    Per-step lists cover MODEL decisions only (the analyses pool them as such).
    Opponent decisions are captured separately in "opp_actions" as pre-decoded
    description strings — "before_model_step": i means the action was taken
    after model decision i-1 and before model decision i (== len(values) for
    opponent actions after the model's last decision).

    `engine_seed` + `full_actions` + `prefix_len` make each game exactly
    replayable (see _replay_to_step / the interactive `whatif` command): reset
    with the recorded seed and feed full_actions[:prefix_len[step]] to land back
    in the state where the model made decision `step`.

    The game loop itself is runner.drive_game (the shared loop); the trace
    recording rides its on_query/on_action hooks.
    """
    import torch
    import runner

    games = []
    for g in range(n_games):
        model_is_a = bool(np.random.random() < 0.5)
        obs, engine_seed = _reset_for_game(env, model_is_a)
        ctrl_a, ctrl_b, ctrl_model = _controllers_for(model, opp_model, model_is_a)

        trace_obs = []
        trace_vals = []
        trace_interp = []
        trace_actions = []
        trace_num_choices = []
        trace_probs = []
        trace_opp_actions = []
        prefix_len = []

        def on_query(d):
            # Record observation, value and policy probs for MODEL decisions
            # (before the controller acts; the obs is unchanged by choosing).
            if d.controller is not ctrl_model:
                return
            obs_t = torch.as_tensor(d.obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                value = model.policy.predict_values(obs_t).item()
            trace_obs.append(d.obs.copy())
            trace_vals.append(value)
            trace_interp.append(_extract_interpretable(d.obs))
            trace_num_choices.append(d.num_choices)
            trace_probs.append(_get_policy_probs(model, d.obs, d.num_choices))
            prefix_len.append(d.index)   # actions fed before this model step

        def on_action(d, action):
            if d.controller is ctrl_model:
                trace_actions.append(int(action))
            else:
                # Decode now (obs is from the opponent's perspective and is not
                # kept) so the replay can interleave opponent actions later.
                trace_opp_actions.append({
                    "desc": _action_desc(d.obs, action),
                    "before_model_step": len(trace_obs),
                })

        rec = runner.drive_game(env, obs, ctrl_a, ctrl_b,
                                on_query=on_query, on_action=on_action)
        full_actions = rec.actions
        model_reward = rec.reward if model_is_a else -rec.reward
        games.append({
            "observations": trace_obs,
            "values": trace_vals,
            "interp_features": trace_interp,
            "actions": trace_actions,
            "num_choices": trace_num_choices,
            "action_probs": trace_probs,
            "opp_actions": trace_opp_actions,
            "engine_seed": engine_seed,
            "full_actions": full_actions,
            "prefix_len": prefix_len,
            "result": model_reward,
            "model_is_a": model_is_a,
        })
        if verbose:
            result_str = "W" if model_reward > 0 else ("L" if model_reward < 0 else "D")
            score = _match_score(games[-1])
            if score is not None:
                result_str += f" {score[0]}-{score[1]}"
            print(f"  game {g}/{n_games - 1}: {len(trace_obs)} decisions, {result_str}",
                  flush=True)
    return games


def _game_is_replayable(game):
    """True if a game trace carries the seed + action log needed for replay."""
    return (game.get("engine_seed") is not None
            and game.get("full_actions") is not None
            and game.get("prefix_len") is not None)


def _replay_to_step(env, game, step):
    """Re-run `game` in `env` up to (not including) model decision `step`.

    Resets with the game's recorded engine seed and deck arrangement, then feeds
    the recorded interleaved action log until the model is on the clock for
    decision `step`. Returns (obs, ok, prefix_reward): `ok` is False (with a
    printed warning) if replay diverged from the stored observation, so callers
    never present a counterfactual built on a desynced state. `prefix_reward` is
    the cumulative Player-A reward accrued during the prefix — nonzero when the
    branch point is in game 2+ of a bo3 match (the ±0.3 intermediates from
    earlier games land here, not after the branch).
    """
    engine_seed = game["engine_seed"]
    model_is_a = game["model_is_a"]
    prefix = game["prefix_len"][step]
    full_actions = game["full_actions"]

    obs, _ = _reset_for_game(env, model_is_a, engine_seed)
    prefix_reward = 0.0
    for a in full_actions[:prefix]:
        obs, r, terminated, truncated, _ = env.step(a)
        prefix_reward += r
        if terminated or truncated:
            print(f"  Replay ended early at prefix action; cannot reach step {step}.")
            return obs, False, prefix_reward

    expected = game["observations"][step]
    if not np.allclose(obs, expected, atol=1e-4):
        n_diff = int(np.sum(~np.isclose(obs, expected, atol=1e-4)))
        print(f"  WARNING: replay diverged from recorded state at step {step} "
              f"({n_diff} obs floats differ). Engine nondeterminism? "
              f"Counterfactual results may be unreliable.")
        return obs, False, prefix_reward
    return obs, True, prefix_reward


def _rollout_from(model, env, opp_model, obs, model_is_a, first_action,
                  record=False):
    """Play a branch to completion: take `first_action`, then let the model and
    opponent finish the game (via runner.drive_game). Returns a dict with the
    branch's model V(s) trajectory, final result (+1/-1/0 from the model's
    perspective), and the number of model decisions after the branch.

    With ``record=True`` the dict also carries ``"trace"`` — the full
    per-decision record of the branch (observations, interp features, legal
    counts, policy probs, model actions, decoded opponent actions, the
    interleaved action log incl. `first_action`, and per-decision prefix
    indexes): everything `_assemble_branch_trace` needs to graft the branch
    onto its source game's prefix as a complete, replayable game trace.
    """
    import torch
    import runner

    branch_vals = []
    trace = ({"observations": [], "interp": [], "num_choices": [], "probs": [],
              "actions": [], "opp_actions": [], "prefix_idx": []}
             if record else None)
    total_reward = 0.0
    obs, reward, terminated, truncated, _ = env.step(first_action)
    total_reward += reward
    rec_actions = []

    if not (terminated or truncated):
        ctrl_a, ctrl_b, ctrl_model = _controllers_for(model, opp_model, model_is_a)

        def on_query(d):
            if d.controller is not ctrl_model:
                return
            obs_t = torch.as_tensor(d.obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                branch_vals.append(model.policy.predict_values(obs_t).item())
            if record:
                trace["observations"].append(d.obs.copy())
                trace["interp"].append(_extract_interpretable(d.obs))
                trace["num_choices"].append(d.num_choices)
                trace["probs"].append(_get_policy_probs(model, d.obs, d.num_choices))
                trace["prefix_idx"].append(d.index)

        on_action = None
        if record:
            def on_action(d, action):
                if d.controller is ctrl_model:
                    trace["actions"].append(int(action))
                else:
                    trace["opp_actions"].append({
                        "desc": _action_desc(d.obs, action),
                        "before_model_step": len(trace["observations"]),
                    })

        rec = runner.drive_game(env, obs, ctrl_a, ctrl_b,
                                on_query=on_query, on_action=on_action)
        total_reward += rec.reward
        rec_actions = rec.actions

    result = total_reward if model_is_a else -total_reward
    out = {
        "values": branch_vals,
        "result": result,
        "n_decisions": len(branch_vals),
    }
    if record:
        trace["full_actions"] = [int(first_action)] + list(rec_actions)
        out["trace"] = trace
    return out


def _assemble_branch_trace(game, game_idx, step, branch, t):
    """Graft a recorded whatif branch onto its source game's prefix, yielding
    a complete game-trace dict in `_collect_game_traces`' schema (including
    the seed/action-log replay keys, so a branch trace can itself be stepped
    through, analyzed, and re-branched with whatif).

    The prefix rows (decisions 0..step) are copied from the source game — the
    replay is byte-identical up to the branch point (verified by
    `_replay_to_step`) — with the branch's action substituted at decision
    `step`; everything after comes from the branch's recorded rollout `t`.
    The ``"whatif"`` key marks the trace as synthetic (a counterfactual line,
    not an independently sampled game), so statistics pools can exclude it.
    """
    n_pre = step + 1                      # prefix decisions incl. the branch point
    prefix = game["prefix_len"][step]     # engine actions fed before decision `step`
    opp_actions = [dict(oa) for oa in game.get("opp_actions", [])
                   if oa["before_model_step"] <= step]
    opp_actions += [{"desc": oa["desc"],
                     "before_model_step": oa["before_model_step"] + n_pre}
                    for oa in t["opp_actions"]]
    return {
        "observations": list(game["observations"][:n_pre]) + t["observations"],
        "values": list(game["values"][:n_pre]) + branch["values"],
        "interp_features": list(game["interp_features"][:n_pre]) + t["interp"],
        "actions": list(game["actions"][:step]) + [branch["action"]] + t["actions"],
        "num_choices": list(game["num_choices"][:n_pre]) + t["num_choices"],
        "action_probs": list(game["action_probs"][:n_pre]) + t["probs"],
        "opp_actions": opp_actions,
        "engine_seed": game["engine_seed"],
        "full_actions": list(game["full_actions"][:prefix]) + t["full_actions"],
        "prefix_len": (list(game["prefix_len"][:n_pre])
                       + [prefix + 1 + ix for ix in t["prefix_idx"]]),
        "result": branch["result"],
        "model_is_a": game["model_is_a"],
        "whatif": {"src_game": game_idx, "step": step,
                   "action": branch["action"], "desc": branch["desc"]},
    }


def _run_whatif(model, env, opp_model, game, game_idx, step, k,
                make_traces=False):
    """Counterfactual rollout: at model decision `step` of `game`, branch on the
    chosen action and the top-`k` alternatives (by recorded policy probability),
    rolling each to completion from the SAME seed. Returns a list of branch dicts
    (chosen first) or None if the game can't be replayed.

    With ``make_traces=True`` each ALTERNATIVE branch dict also carries
    ``"trace_game"`` — the branch grafted onto the source game's prefix as a
    complete game trace (see `_assemble_branch_trace`) that can be browsed and
    re-branched like any simulated game. The chosen branch gets no trace: its
    line IS the source game.
    """
    if not _game_is_replayable(game):
        print("  This game has no recorded seed/action log — it predates the "
              "replay-enabled collector. Re-collect (or 'run <N>') to enable whatif.")
        return None
    if env is None or model is None:
        print("  No live env available. Use the 'interactive' command to enable whatif.")
        return None

    n_steps = len(game["observations"])
    if step < 0 or step >= n_steps:
        print(f"  Step out of range for game {game_idx}. Valid range: 0–{n_steps - 1}")
        return None

    chosen = game["actions"][step]
    num_choices = game["num_choices"][step]
    probs = game["action_probs"][step] if game.get("action_probs") else None

    # Candidate order: chosen action first, then the highest-probability
    # alternatives (falling back to index order when no probs are recorded).
    alts = [i for i in range(num_choices) if i != chosen]
    if probs is not None:
        alts.sort(key=lambda i: -probs[i])
    candidates = [chosen] + alts[:max(0, k)]

    obs0 = game["observations"][step]
    result_str = "WIN" if game["result"] > 0 else ("LOSS" if game["result"] < 0 else "DRAW")
    print(f"\nWhatif — game {game_idx} [{result_str}], decision {step}/{n_steps - 1}, "
          f"branching {len(candidates)} action(s) from the same seed "
          f"(seed={game['engine_seed']}):")

    branches = []
    for ci, act in enumerate(candidates):
        obs, ok, prefix_reward = _replay_to_step(env, game, step)
        if not ok:
            # Divergence already warned; abort the whole whatif — every branch
            # would share the same bad prefix.
            return None
        # Rewards accrued before the branch (earlier bo3 games) belong to every
        # branch's match total, from the model's perspective like roll["result"].
        prefix_result = prefix_reward if game["model_is_a"] else -prefix_reward
        desc = _action_desc(obs0, act)
        p = probs[act] if probs is not None else None
        roll = _rollout_from(model, env, opp_model, obs, game["model_is_a"], act,
                             record=make_traces and act != chosen)
        branch = {
            "action": act,
            "desc": desc,
            "prob": p,
            "is_chosen": act == chosen,
            "v_after": roll["values"][0] if roll["values"] else None,
            "result": roll["result"] + prefix_result,
            "n_decisions": roll["n_decisions"],
            "values": roll["values"],
        }
        if roll.get("trace") is not None:
            branch["trace_game"] = _assemble_branch_trace(
                game, game_idx, step, branch, roll["trace"])
        branches.append(branch)

    # Determinism sanity check on the chosen branch: replaying the recorded
    # action should reproduce the recorded game result.
    chosen_branch = branches[0]
    if abs(chosen_branch["result"] - game["result"]) > 1e-6:
        print(f"  WARNING: replaying the chosen action gave result "
              f"{chosen_branch['result']:+.1f} but the recorded game was "
              f"{game['result']:+.1f} — engine nondeterminism; treat results with care.")

    _print_whatif_table(branches)
    return branches


def _print_whatif_table(branches):
    """Print the whatif branch comparison table."""
    print(f"\n  {'Action':<34} {'P(a)':>7} {'V after':>9} {'Result':>8} {'Len':>5}")
    print(f"  {'-'*34} {'-'*7} {'-'*9} {'-'*8} {'-'*5}")
    for b in branches:
        tag = " *" if b["is_chosen"] else "  "
        desc = (b["desc"][:30] + "…") if len(b["desc"]) > 31 else b["desc"]
        pstr = f"{b['prob']:.3f}" if b["prob"] is not None else "  –  "
        vstr = f"{b['v_after']:+.3f}" if b["v_after"] is not None else "  –  "
        res = b["result"]
        rstr = "WIN" if res > 0 else ("LOSS" if res < 0 else "DRAW")
        print(f"{tag}{desc:<34} {pstr:>7} {vstr:>9} {rstr:>8} {b['n_decisions']:>5}")
    print("  (* = action the model actually took; V after = model V(s) at the "
          "first decision after branching)")


def _action_desc(obs, i):
    """Human-readable description of legal-action slot i ("CAST (Lightning Bolt)").
    Tokens render as "Token"; a target choice with no card entity is a player.
    The action's zone_ref is appended when set ("TARGET (Wasteland @opp bf)"),
    and its entity-slot ref when set ("TARGET (Wasteland @opp bf @slot48)")."""
    cat = int(round(obs[STATE_SIZE + i] * ACTION_CATEGORY_MAX))
    cat_name = _CAT_NAMES.get(cat, str(cat))

    zone = int(round(obs[STATE_SIZE + 3 * MAX_ACTIONS + i] * REF_ZONE_MAX))
    zone_str = f" @{_REF_NAMES.get(zone, zone)}" if zone > 0 else ""
    # Entity-slot ref (5th metadata block; -1 = none) disambiguates same-named cards.
    slot = int(round(obs[STATE_SIZE + 4 * MAX_ACTIONS + i] * N_ENTITY_REF_SLOTS)) - 1
    if slot >= 0:
        zone_str += f" @slot{slot}"

    card_raw = obs[STATE_SIZE + MAX_ACTIONS + i]
    if card_raw < 0:
        if cat_name in ("TARGET", "ATK_TGT"):
            return f"{cat_name} (Player{zone_str})"
        return cat_name
    cid = int(round(card_raw * N_CARD_TYPES))
    return f"{cat_name} ({decode.card_index_to_name(cid) or f'card#{cid}'}{zone_str})"


def _decode_legal_actions(obs, num_choices, chosen_action):
    """Return a list of strings describing each legal action, marking the chosen one."""
    lines = []
    for i in range(num_choices):
        marker = " <-- chosen" if i == chosen_action else ""
        lines.append(f"    [{i}] {_action_desc(obs, i)}{marker}")
    return lines




_INTERP_STEP_NAMES = [
    "UNTAP", "UPKEEP", "DRAW", "FIRST_MAIN", "BEGIN_COMBAT",
    "DECLARE_ATK", "DECLARE_BLK", "FIRST_STRIKE", "COMBAT_DMG",
    "END_COMBAT", "SECOND_MAIN", "END", "CLEANUP",
    "SIDEBOARD",
]
_INTERP_STEP_OFFSET = _INTERP_FEATURE_NAMES.index("step_untap")


_INTERP_SIDEBOARD_IDX = _INTERP_FEATURE_NAMES.index("is_sideboard")


def _step_name_from_feat(feat):
    """Decode step one-hot from interp feature vector."""
    if feat[_INTERP_SIDEBOARD_IDX] > 0.5:
        return "SIDEBOARD"
    best = -1
    best_val = -1.0
    # only iterate the 13 game-step one-hots, not SIDEBOARD
    for i in range(13):
        v = feat[_INTERP_STEP_OFFSET + i]
        if v > best_val:
            best_val = v
            best = i
    if best < 0:
        return "?"
    return _INTERP_STEP_NAMES[best]


def _replay_sim_game(game, game_idx, verbose=False):
    """Print a human-readable trace for a simulation game.

    verbose=True adds, under each decision's one-liner, the decoded zones
    (both battlefields, hand, graveyards, stack) and the model's chosen action,
    and interleaves the opponent's actions between the model's decisions.
    The obs at each recorded step is from the model's perspective ("self" = model).
    """
    result_str = "WIN" if game["result"] > 0 else ("LOSS" if game["result"] < 0 else "DRAW")
    side = "A" if game["model_is_a"] else "B"
    print(f"\nGame {game_idx} — Model={side}, {result_str}, {len(game['values'])} model decisions")
    print()

    feats = game["interp_features"]
    vals = game["values"]
    obs_list = game.get("observations", [])
    actions = game.get("actions", [])
    num_choices = game.get("num_choices", [])

    # Opponent actions grouped by the model decision they precede
    opp_by_step = {}
    if verbose:
        for oa in game.get("opp_actions", []):
            opp_by_step.setdefault(oa["before_model_step"], []).append(oa["desc"])

    def print_opp_actions(step):
        # Collapse runs of identical consecutive actions ("PASS (x19)")
        run_desc, run_len = None, 0
        def flush():
            if run_len == 1:
                print(f"        opp --> {run_desc}")
            elif run_len > 1:
                print(f"        opp --> {run_desc} (x{run_len})")
        for desc in opp_by_step.get(step, []):
            if desc == run_desc:
                run_len += 1
            else:
                flush()
                run_desc, run_len = desc, 1
        flush()

    for i, (feat, val) in enumerate(zip(feats, vals)):
        print_opp_actions(i)
        step = _step_name_from_feat(feat)
        mana_total = feat[_FEAT["self_total_mana"]]
        # Game::turn counts from 0; display 1-based like the engine's turn header
        turn_no = 1 + (int(round(feat[_FEAT["turn"]])) if len(feat) > _FEAT["turn"] else 0)
        whose = "self" if feat[_FEAT["is_active_player"]] > 0.5 else "opp"
        print(f"  [{i:3d}] T{turn_no:<2d} {whose:<4} {step:<14}  "
              f"Life {feat[_FEAT['self_life']]:.0f}/{feat[_FEAT['opp_life']]:.0f}  "
              f"Board {feat[_FEAT['self_creatures']]:.0f}c+{feat[_FEAT['self_lands']]:.0f}l"
              f" / {feat[_FEAT['opp_creatures']]:.0f}c+{feat[_FEAT['opp_lands']]:.0f}l  "
              f"Mana {mana_total:.0f}  "
              f"GY {feat[_FEAT['self_gy_size']]:.0f}/{feat[_FEAT['opp_gy_size']]:.0f}  "
              f"V={val:+.3f}")
        if not verbose or i >= len(obs_list):
            continue
        obs = obs_list[i]
        sp = _perm_summaries(obs, range(_SELF_PERM_SLOTS))
        op = _perm_summaries(obs, range(_SELF_PERM_SLOTS, _PERM_SLOTS))
        print(f"        BF self : {', '.join(sp) if sp else '—'}")
        print(f"        BF opp  : {', '.join(op) if op else '—'}")
        hand = _hand_card_names(obs)
        print(f"        Hand    : {', '.join(hand) if hand else '—'}")
        gy_self = _gy_card_names(obs, self_side=True)
        gy_opp  = _gy_card_names(obs, self_side=False)
        if gy_self:
            print(f"        GY self : {', '.join(gy_self)}")
        if gy_opp:
            print(f"        GY opp  : {', '.join(gy_opp)}")
        stack = _stack_summaries(obs)
        if stack:
            print(f"        Stack   : {'; '.join(stack)}")
        if i < len(actions) and i < len(num_choices):
            print(f"        Action  : {_action_desc(obs, actions[i])}  "
                  f"[choice {actions[i]} of {num_choices[i]}]")
        print()
    print_opp_actions(len(feats))  # opponent actions after the model's last decision
    print()


def _sim_summary(games):
    """Print win/loss/draw summary for a list of sim games."""
    wins   = sum(1 for g in games if g["result"] > 0)
    losses = sum(1 for g in games if g["result"] < 0)
    draws  = sum(1 for g in games if g["result"] == 0)
    total  = len(games)
    if total == 0:
        print("  No games.")
        return
    lengths = [len(g["values"]) for g in games]
    arr = np.array(lengths)
    print(f"  {total} games: {wins}W / {losses}L / {draws}D  ({100 * wins / total:.1f}% win rate)")
    print(f"  Decisions/game: mean={arr.mean():.1f}  min={arr.min()}  max={arr.max()}")

    # Bo3 breakdown: per-match game scores and the individual-game record.
    score_dist = {}
    game_w = game_l = 0
    for g in games:
        sc = _match_score(g)
        if sc is None:
            continue
        r = "W" if g["result"] > 0 else ("L" if g["result"] < 0 else "D")
        score_dist[f"{r} {sc[0]}-{sc[1]}"] = score_dist.get(f"{r} {sc[0]}-{sc[1]}", 0) + 1
        game_w += sc[0]
        game_l += sc[1]
    if score_dist:
        order = {"W 2-0": 0, "W 2-1": 1, "L 1-2": 2, "L 0-2": 3}
        parts = [f"{k}: {n}" for k, n in
                 sorted(score_dist.items(), key=lambda kv: (order.get(kv[0], 9), kv[0]))]
        gt = game_w + game_l
        print(f"  Bo3 match scores: " + "   ".join(parts))
        if gt:
            print(f"  Individual games: {game_w}W / {game_l}L"
                  f"  ({100 * game_w / gt:.1f}% game win rate)")


def _match_meta(obs):
    """Decode (game_number, self_wins, opp_wins, is_sideboard) match context
    from a stored observation. game_number is 0-based; all fields are zero in
    single-game (bo1) mode. During the sideboard phase game_number still holds
    the finished game's number (the convention the `sideboard` report uses)."""
    return (int(round(obs[_MATCH_CTX_START] * 3.0)),
            int(round(obs[_MATCH_CTX_START + 1] * 2.0)),
            int(round(obs[_MATCH_CTX_START + 2] * 2.0)),
            bool(obs[_MATCH_CTX_START + 3] > 0.5))


def _match_score(game):
    """Final bo3 game score (self_wins, opp_wins) from the model's perspective,
    or None for a bo1 trace (match context all-zero at every decision). The
    obs win counters never include the final game — the match ends on it — so
    the match winner (sign of result) gets one added to the highest observed
    counts. Draws (result 0) return the counters as observed."""
    hi_self = hi_opp = 0
    bo3 = False
    for obs in game["observations"]:
        gn, sw, ow, sb = _match_meta(obs)
        if gn or sw or ow or sb:
            bo3 = True
        hi_self = max(hi_self, sw)
        hi_opp = max(hi_opp, ow)
    if not bo3:
        return None
    if game["result"] > 0:
        hi_self += 1
    elif game["result"] < 0:
        hi_opp += 1
    return hi_self, hi_opp


def _compute_swings(games):
    """Return swing_data list sorted by magnitude.

    Only in-game deltas count: consecutive model decisions inside the SAME game
    of a bo3 match, neither of them a sideboard decision. The V(s)
    discontinuities at match boundaries (the board context resetting between
    games) are a separate signal — see the `boundaries` command."""
    swing_data = []
    for g_idx, game in enumerate(games):
        vals = game["values"]
        if len(vals) < 2:
            continue
        metas = [_match_meta(o) for o in game["observations"]]
        max_delta_idx, max_delta = None, -1.0
        for i in range(len(vals) - 1):
            (ga, _, _, sba), (gb, _, _, sbb) = metas[i], metas[i + 1]
            if sba or sbb or ga != gb:
                continue
            d = abs(vals[i + 1] - vals[i])
            if d > max_delta:
                max_delta_idx, max_delta = i, d
        if max_delta_idx is None:
            continue
        swing_data.append({
            "game_idx": g_idx,
            "swing_step": max_delta_idx,
            "match_game": metas[max_delta_idx][0] + 1,
            "swing_magnitude": max_delta,
            "swing_from": vals[max_delta_idx],
            "swing_to": vals[max_delta_idx + 1],
            "result": game["result"],
            "n_decisions": len(vals),
            "model_is_a": game["model_is_a"],
        })
    swing_data.sort(key=lambda x: -x["swing_magnitude"])
    return swing_data


def _print_swing_table(top_swings):
    print(f"{'Game':<6} {'Step':<6} {'MG':<4} {'From':>8} {'To':>8} {'Delta':>8} {'Result':<8} {'Decisions':<10}")
    print("-" * 64)
    for s in top_swings:
        result_str = "WIN" if s["result"] > 0 else ("LOSS" if s["result"] < 0 else "DRAW")
        side = "A" if s["model_is_a"] else "B"
        print(f"  {s['game_idx']:<4}   {s['swing_step']:<4}   {s.get('match_game', 1):<2} "
              f"{s['swing_from']:>+7.3f} {s['swing_to']:>+7.3f}"
              f" {s['swing_to'] - s['swing_from']:>+7.3f}   {result_str:<6}({side}) {s['n_decisions']}")


def _compute_boundaries(games):
    """Per bo3 match boundary: V(s) at the end of the finished game, through
    the sideboard block, and at the next game's first decision. Splits each
    boundary into a result-pricing component (end of game → first sideboard
    decision, where the match-score features update) and a re-anchoring
    component (last sideboard decision → fresh game, where the board context
    resets). Boundaries the model crossed without a sideboard decision of its
    own (game_number changes between consecutive in-game steps) are reported
    with the sideboard columns empty."""
    rows = []
    for g_idx, game in enumerate(games):
        vals = game["values"]
        metas = [_match_meta(o) for o in game["observations"]]
        i = 0
        while i < len(metas) - 1:
            if metas[i][3]:  # sideboard block
                j = i
                while j < len(metas) and metas[j][3]:
                    j += 1
                rows.append({
                    "game_idx": g_idx,
                    "after_game": metas[i][0] + 1,
                    "score": (metas[i][1], metas[i][2]),
                    "v_pre": vals[i - 1] if i > 0 else None,
                    "v_sb_first": vals[i],
                    "v_sb_last": vals[j - 1],
                    "v_post": vals[j] if j < len(vals) else None,
                    "result": game["result"],
                })
                i = j
            elif metas[i][0] != metas[i + 1][0]:  # boundary with no model sb step
                rows.append({
                    "game_idx": g_idx,
                    "after_game": metas[i][0] + 1,
                    "score": (metas[i + 1][1], metas[i + 1][2]),
                    "v_pre": vals[i],
                    "v_sb_first": None,
                    "v_sb_last": None,
                    "v_post": vals[i + 1],
                    "result": game["result"],
                })
                i += 1
            else:
                i += 1
    return rows


def _print_boundaries(rows):
    def fmt(v):
        return f"{v:+7.3f}" if v is not None else "     — "

    if not rows:
        print("  No bo3 match boundaries in the sample (bo1 games, or every "
              "match ended 2-0 inside its trace).")
        return
    print("\nMatch boundaries — V(s) across bo3 game transitions:")
    print("  ΔV result   = end of game → first sideboard decision (match-score features update)")
    print("  ΔV reanchor = last sideboard decision → next game's first decision (board context resets)")
    print(f"\n  {'Game':<5} {'After':<6} {'Score':<6} {'V end':>7} {'V sb1':>7} {'V sbN':>7} "
          f"{'V next':>7} {'ΔV result':>10} {'ΔV reanchor':>12} {'Result':<6}")
    print("  " + "-" * 84)
    for r in rows:
        d_res = (r["v_sb_first"] - r["v_pre"]
                 if r["v_sb_first"] is not None and r["v_pre"] is not None else None)
        d_re = (r["v_post"] - r["v_sb_last"]
                if r["v_post"] is not None and r["v_sb_last"] is not None else None)
        if d_res is None and d_re is None and r["v_pre"] is not None and r["v_post"] is not None:
            d_re = r["v_post"] - r["v_pre"]  # no sb steps: whole delta is the re-anchor
        result_str = "WIN" if r["result"] > 0 else ("LOSS" if r["result"] < 0 else "DRAW")
        score = f"{r['score'][0]}-{r['score'][1]}"
        print(f"  {r['game_idx']:<5} g{r['after_game']:<5} {score:<6} {fmt(r['v_pre'])} "
              f"{fmt(r['v_sb_first'])} {fmt(r['v_sb_last'])} {fmt(r['v_post'])} "
              f"{fmt(d_res):>10} {fmt(d_re):>12} {result_str:<6}")


def _print_match_calibration(games):
    """Per (match, game) score-conditioned value calibration: V(s) at each
    game's first decision vs the empirical mean REMAINING match return for
    that score across the sample (remaining = final result minus the ±0.3
    intermediates already accrued at that score)."""
    # Collect one row per (match, game-within-match).
    rows = []
    for g_idx, game in enumerate(games):
        vals = game["values"]
        metas = [_match_meta(o) for o in game["observations"]]
        seen = set()
        for i, (gno, sw, ow, sb) in enumerate(metas):
            if sb or gno in seen:
                continue
            seen.add(gno)
            accrued = BO3_GAME_WIN_REWARD * (sw - ow)
            rows.append({
                "game_idx": g_idx,
                "match_game": gno + 1,
                "score": (sw, ow),
                "v_start": vals[i],
                "remaining": game["result"] - accrued,
                "result": game["result"],
            })
    if not rows:
        print("  No games.")
        return

    buckets = {}
    for r in rows:
        buckets.setdefault(r["score"], []).append(r)

    print("\nMatch-score calibration — V(s) at each game's first decision vs the")
    print("empirical mean remaining return for that match score (gap >0 = optimistic):")
    print(f"\n  By score entering the game:")
    print(f"  {'Score':<6} {'n':>4} {'mean V start':>13} {'mean remaining':>15} {'gap':>8}")
    print("  " + "-" * 50)
    for score in sorted(buckets):
        rs = buckets[score]
        mv = float(np.mean([r["v_start"] for r in rs]))
        mr = float(np.mean([r["remaining"] for r in rs]))
        print(f"  {score[0]}-{score[1]:<4} {len(rs):>4} {mv:>+13.3f} {mr:>+15.3f} {mv - mr:>+8.3f}")

    print(f"\n  Per match, per game (gap vs that score's empirical mean):")
    print(f"  {'Match':<6} {'Game':<5} {'Score':<6} {'V start':>8} {'empirical':>10} "
          f"{'gap':>8} {'Result':<6}")
    print("  " + "-" * 56)
    for r in rows:
        emp = float(np.mean([b["remaining"] for b in buckets[r["score"]]]))
        result_str = "WIN" if r["result"] > 0 else ("LOSS" if r["result"] < 0 else "DRAW")
        score = f"{r['score'][0]}-{r['score'][1]}"
        print(f"  {r['game_idx']:<6} g{r['match_game']:<4} {score:<6} {r['v_start']:>+8.3f} "
              f"{emp:>+10.3f} {r['v_start'] - emp:>+8.3f} {result_str:<6}")


def _card_name_at(obs, base):
    """Decode card name from the card-id float at obs[base] (None if empty).
    Tokens (TOKEN_SENTINEL id) render as "Token"."""
    cid = _slot_card_idx(obs, base)
    if cid < 0:
        return None
    return decode.card_index_to_name(cid) or f"card#{cid}"


def _perm_summaries(obs, slot_range):
    """One compact string per occupied battlefield slot ("Grizzly Bears 2/2 [tapped]")."""
    out = []
    for slot in slot_range:
        base = _PERM_START + slot * _PERM_SLOT_SZ
        name = _card_name_at(obs, base + _PERM_CARD_OFF)
        if name is None:
            continue
        power     = obs[base + 0] * 10.0
        toughness = obs[base + 1] * 10.0
        tapped    = obs[base + 2] > 0.5
        attacking = obs[base + 3] > 0.5
        blocking  = obs[base + 4] > 0.5
        sickness  = obs[base + 5] > 0.5
        damage    = obs[base + 6] * 10.0
        is_creat  = obs[base + 8] > 0.5
        loyalty   = obs[base + _PERM_LOYALTY_OFF] * 10.0
        phased    = obs[base + _OFF_IS_PHASED_OUT] > 0.5
        flags = []
        if tapped:        flags.append("tapped")
        if attacking:     flags.append("atk")
        if blocking:      flags.append("blk")
        if sickness:      flags.append("sick")
        if phased:        flags.append("phased")
        if damage > 0.4:  flags.append(f"dmg={damage:.0f}")
        if loyalty > 0.4: flags.append(f"loy={loyalty:.0f}")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        if is_creat:
            out.append(f"{name} {power:.0f}/{toughness:.0f}{flag_str}")
        else:
            out.append(f"{name}{flag_str}")
    return out


def _stack_summaries(obs, self_label="self", opp_label="opp"):
    """One compact string per occupied stack slot ("[A] Lightning Bolt (spell)")."""
    out = []
    for slot in range(_STACK_SLOTS):
        base = _STACK_START + slot * _STACK_SLOT_SZ
        name = _card_name_at(obs, base + 1)
        if name is None:
            continue
        ctrl_str = self_label if obs[base] > 0.5 else opp_label
        type_str = "spell" if obs[base + 2] > 0.5 else "ability"  # fixed offset; modes/targets follow
        out.append(f"[{ctrl_str}] {name} ({type_str})")
    return out


def _hand_card_names(obs):
    """Names of the priority player's hand cards (non-empty slots only)."""
    names = [_card_name_at(obs, _HAND_START + s * _HAND_SLOT_SIZE)
             for s in range(_HAND_SLOTS)]
    return [n for n in names if n is not None]


def _gy_card_names(obs, self_side):
    """Names of the cards in one player's graveyard (non-empty slots only)."""
    rng = range(_GY_SELF_SLOTS) if self_side else range(_GY_SELF_SLOTS, _GY_SLOTS)
    names = [_card_name_at(obs, _GY_START_OBS + s * _GY_SLOT_SZ) for s in rng]
    return [n for n in names if n is not None]


def _decode_board_state(obs, value=None):
    """Print a detailed board state decoded from a raw observation vector."""
    # _SELF_IS_A_IDX = 1.0 if "self" (priority player) is Player A
    priority_is_a    = obs[_SELF_IS_A_IDX] > 0.5
    priority_is_active = obs[_IS_ACTIVE_IDX] > 0.5
    self_label = "A" if priority_is_a else "B"
    opp_label  = "B" if priority_is_a else "A"

    self_life    = obs[_SELF_BLOCK_START + _PB_LIFE] * 20.0
    opp_life     = obs[_OPP_BLOCK_START + _PB_LIFE] * 20.0
    self_hand_ct = obs[_SELF_BLOCK_START + _PB_HAND_CT] * 10.0
    opp_hand_ct  = obs[_OPP_BLOCK_START + _PB_HAND_CT] * 10.0
    self_mana    = [obs[_SELF_BLOCK_START + _PB_MANA + j] * 10.0 for j in range(6)]
    opp_mana     = [obs[_OPP_BLOCK_START + _PB_MANA + j] * 10.0 for j in range(6)]
    stack_size   = int(round(obs[_STACK_SIZE_IDX] * 10.0))

    # Match context (_MATCH_CTX_START .. +4) and library/post-board (_LIBRARY_CTX_START .. +3)
    game_number      = int(round(obs[_MATCH_CTX_START]     * 3.0))
    self_match_wins  = int(round(obs[_MATCH_CTX_START + 1] * 2.0))
    opp_match_wins   = int(round(obs[_MATCH_CTX_START + 2] * 2.0))
    is_sideboard     = obs[_MATCH_CTX_START + 3] > 0.5
    self_library_ct  = int(round(obs[_LIBRARY_CTX_START]     * 60.0))
    opp_library_ct   = int(round(obs[_LIBRARY_CTX_START + 1] * 60.0))
    is_post_board    = obs[_LIBRARY_CTX_START + 2] > 0.5

    step_idx  = int(np.argmax(obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]))
    step_name = _INTERP_STEP_NAMES[step_idx] if step_idx < len(_INTERP_STEP_NAMES) else f"?{step_idx}"
    val_str   = f"  V={value:+.3f}" if value is not None else ""

    def mana_str(mana):
        parts = [f"{_MANA_COLORS[j]}:{mana[j]:.0f}" for j in range(6) if mana[j] > 0.4]
        return " ".join(parts) if parts else "—"

    # "self" = the priority player in this decode, so the active-player flag says whose turn it is.
    # Game::turn counts from 0; display 1-based like the engine's turn header.
    turn_no = int(round(obs[_CUR_TURN_IDX] * 50.0)) + 1
    whose = self_label if priority_is_active else opp_label
    print(f"Turn {turn_no} ({whose}'s turn) — Step: {step_name}  (priority: {self_label}){val_str}")
    if game_number > 0 or self_match_wins > 0 or opp_match_wins > 0 or is_sideboard or is_post_board:
        sb_str = "  [sideboarding]" if is_sideboard else ("  [post-board]" if is_post_board else "")
        print(f"Match: game {game_number}  wins {self_label}={self_match_wins} {opp_label}={opp_match_wins}{sb_str}")
    print(f"Stack: {stack_size} item(s)")

    if stack_size > 0:
        print("  Stack (top first):")
        for ln in _stack_summaries(obs, self_label, opp_label):
            print(f"    {ln}")

    # The spell/ability asking for the current mid-resolution choice (target
    # select, dig/search pick, discard, modal, ...). May not be on the stack yet.
    pending = decode._decode_pending_decision(obs)
    if pending:
        who = self_label if pending["is_self"] else opp_label
        print(f"Pending choice from: {pending['name']} ({who})")

    print()
    print(f"  [{self_label}] Priority player  "
          f"Life={self_life:.0f}  Hand={self_hand_ct:.0f}  Lib={self_library_ct}  Mana=[{mana_str(self_mana)}]")

    sp = _perm_summaries(obs, range(_SELF_PERM_SLOTS))
    if sp:
        print(f"  Battlefield ({len(sp)}):")
        for ln in sp: print(f"    {ln}")
    else:
        print("  Battlefield: empty")

    hand_cards = _hand_card_names(obs)
    if hand_cards:
        print(f"  Hand ({len(hand_cards)}): {', '.join(hand_cards)}")
    else:
        print("  Hand: empty")

    self_gy = _gy_card_names(obs, self_side=True)
    if self_gy:
        print(f"  Graveyard ({len(self_gy)}): {', '.join(self_gy)}")
    else:
        print("  Graveyard: empty")

    print()
    print(f"  [{opp_label}] Opponent          "
          f"Life={opp_life:.0f}  Hand={opp_hand_ct:.0f}  Lib={opp_library_ct}  Mana=[{mana_str(opp_mana)}]")

    op = _perm_summaries(obs, range(_SELF_PERM_SLOTS, _PERM_SLOTS))
    if op:
        print(f"  Battlefield ({len(op)}):")
        for ln in op: print(f"    {ln}")
    else:
        print("  Battlefield: empty")

    opp_gy = _gy_card_names(obs, self_side=False)
    if opp_gy:
        print(f"  Graveyard ({len(opp_gy)}): {', '.join(opp_gy)}")
    else:
        print("  Graveyard: empty")
    print()


def _obs_action_cat(obs, i):
    """Extract action category integer for slot i from obs array."""
    return int(round(obs[STATE_SIZE + i] * ACTION_CATEGORY_MAX))


def _obs_card_id(obs, i):
    """Extract card vocab index for slot i from obs array. Returns -1 for null."""
    raw = obs[STATE_SIZE + MAX_ACTIONS + i]
    if raw < 0:
        return -1
    return int(round(raw * N_CARD_TYPES))


def _obs_ctrl_flag(obs, i):
    """Extract controller flag for slot i. 1=self, 0=opp, -1=null."""
    raw = obs[STATE_SIZE + 2 * MAX_ACTIONS + i]
    if raw > 0.5:
        return 1
    elif raw > -0.01:
        return 0
    return -1


def _sim_targeting(games):
    """Targeting analysis over interactive-mode game traces."""
    if not games:
        print("  No games in memory.")
        return

    # cast_targets: card_name -> [(tgt_name, tgt_is_self, result,
    #                              step_idx_placeholder, alt_self, alt_opp)]
    cast_targets = {}
    # hold_stats / cast_stats: card_name -> [(result,)]
    hold_stats = {}
    cast_stats = {}

    for g in games:
        result = g["result"]
        observations = g["observations"]
        actions = g["actions"]
        num_choices_list = g["num_choices"]
        n_steps = len(observations)

        for si in range(n_steps):
            obs = observations[si]
            action = actions[si]
            nc = num_choices_list[si]
            chosen_cat = _obs_action_cat(obs, action)

            # Link CAST -> TARGET
            if chosen_cat == CAT_CAST_SPELL:
                cid = _obs_card_id(obs, action)
                if 0 <= cid < len(_VOCAB_NAMES):
                    cast_name = decode.card_index_to_name(cid)
                    # Look ahead for TARGET
                    for sj in range(si + 1, n_steps):
                        obs2 = observations[sj]
                        act2 = actions[sj]
                        cat2 = _obs_action_cat(obs2, act2)
                        if cat2 == CAT_SELECT_TARGET:
                            tgt_cid = _obs_card_id(obs2, act2)
                            tgt_ctrl = _obs_ctrl_flag(obs2, act2)
                            tgt_name = _resolve_card_name(tgt_cid)
                            tgt_is_self = (tgt_ctrl == 1)
                            # Count available targets by owner
                            nc2 = num_choices_list[sj]
                            alt_self = 0
                            alt_opp = 0
                            for k in range(nc2):
                                if _obs_action_cat(obs2, k) == CAT_SELECT_TARGET:
                                    cf = _obs_ctrl_flag(obs2, k)
                                    if cf == 1:
                                        alt_self += 1
                                    elif cf == 0:
                                        alt_opp += 1
                            cast_targets.setdefault(cast_name, []).append((
                                tgt_name, tgt_is_self, result,
                                alt_self, alt_opp,
                            ))
                            break
                        elif cat2 != CAT_PASS_PRIORITY:  # non-PASS means no target for this cast
                            break

            # Hold analysis: PASS when CAST was available
            castable = set()
            for k in range(nc):
                if _obs_action_cat(obs, k) == CAT_CAST_SPELL:
                    kid = _obs_card_id(obs, k)
                    if 0 <= kid < len(_VOCAB_NAMES):
                        castable.add(decode.card_index_to_name(kid))
            if not castable:
                continue
            if chosen_cat == CAT_PASS_PRIORITY:
                for name in castable:
                    hold_stats.setdefault(name, []).append((result,))
            elif chosen_cat == CAT_CAST_SPELL:
                cid = _obs_card_id(obs, action)
                if 0 <= cid < len(_VOCAB_NAMES):
                    cast_stats.setdefault(decode.card_index_to_name(cid), []).append((result,))

    # ── Print ────────────────────────────────────────────────────────────────
    print(f"\nTargeting analysis — {len(games)} games\n")

    if cast_targets:
        print(f"{'=' * 60}")
        print("TARGETING: self vs opponent (linked CAST -> TARGET)")
        print(f"{'=' * 60}\n")

        for name in sorted(cast_targets, key=lambda n: -len(cast_targets[n])):
            entries = cast_targets[name]
            if len(entries) < 2:
                continue
            self_tgt = [e for e in entries if e[1]]
            opp_tgt = [e for e in entries if not e[1]]
            total = len(entries)
            self_pct = len(self_tgt) / total * 100
            opp_pct = len(opp_tgt) / total * 100

            print(f"  {name} ({total} targeting decisions)")
            print(f"    targets self: {len(self_tgt):4d} ({self_pct:5.1f}%)")
            print(f"    targets opp:  {len(opp_tgt):4d} ({opp_pct:5.1f}%)")

            # Breakdown by target card
            tgt_counts = {}
            for e in entries:
                label = f"{'own' if e[1] else 'opp'} {e[0]}"
                tgt_counts[label] = tgt_counts.get(label, 0) + 1
            if tgt_counts:
                print("    target breakdown:")
                for label, count in sorted(tgt_counts.items(), key=lambda x: -x[1])[:10]:
                    print(f"      {count:4d}  {label}")

            # Win rate when targeting self vs opponent
            for label, subset in [("self", self_tgt), ("opp", opp_tgt)]:
                if len(subset) < 2:
                    continue
                wins = sum(1 for e in subset if e[2] > 0)
                losses = sum(1 for e in subset if e[2] < 0)
                wr = wins / len(subset) * 100
                alt_self_avg = np.mean([e[3] for e in subset])
                alt_opp_avg = np.mean([e[4] for e in subset])
                print(f"    when targeting {label}: {wins}W/{losses}L ({wr:.0f}% WR)"
                      f"  avg avail: {alt_self_avg:.1f} self / {alt_opp_avg:.1f} opp targets")
            print()

    # Hold vs cast
    all_cards = sorted(set(list(hold_stats.keys()) + list(cast_stats.keys())))
    has_hold_data = any(len(hold_stats.get(n, [])) >= 3 for n in all_cards)
    if has_hold_data:
        print(f"{'=' * 60}")
        print("HOLD vs CAST: how often each card is held when castable")
        print(f"{'=' * 60}\n")

        for name in all_cards:
            holds = hold_stats.get(name, [])
            casts = cast_stats.get(name, [])
            total = len(holds) + len(casts)
            if total < 3:
                continue
            hold_pct = len(holds) / total * 100

            print(f"  {name}: held {len(holds)}/{total} ({hold_pct:.0f}%) when castable")
            for label, subset in [("hold", holds), ("cast", casts)]:
                if len(subset) < 2:
                    continue
                wins = sum(1 for e in subset if e[0] > 0)
                losses = sum(1 for e in subset if e[0] < 0)
                wr = wins / len(subset) * 100
                print(f"    {label}: {wins}W/{losses}L ({wr:.0f}% WR)")
            print()


def _sim_sideboard_report(games):
    """Report sideboard decisions made by model and scripted agent across games."""
    _CAT_SB_IN   = CAT_SIDEBOARD_IN
    _CAT_SB_OUT  = CAT_SIDEBOARD_OUT
    _CAT_SB_DONE = CAT_SIDEBOARD_DONE

    # Per-phase records: list of (match_idx, after_game_num, cards_in, cards_out).
    # after_game_num is 1-based: the game that just ended (so a 2-game match has
    # one phase with after_game_num=1; a 3-game match has after_game_num=1 and 2).
    phases = []

    for gi, g in enumerate(games):
        observations = g["observations"]
        actions = g["actions"]
        n_steps = len(observations)
        model_is_a = g["model_is_a"]

        # Track current sideboard phase
        cards_in = []
        cards_out = []
        current_after_game = None  # match_game_number at the phase (0-based)

        for si in range(n_steps):
            obs = observations[si]
            action = actions[si]

            cat_raw = obs[STATE_SIZE + action]
            cat = int(round(cat_raw * ACTION_CATEGORY_MAX))

            is_sb_action = cat in (_CAT_SB_IN, _CAT_SB_OUT, _CAT_SB_DONE)

            # If an in-progress phase has no DONE (engine hit its 15-swap cap
            # and moved on silently), close it out when gameplay resumes so
            # after-game-1 and after-game-2 stay separate.
            if not is_sb_action and current_after_game is not None:
                after_game = current_after_game + 1
                phases.append((gi, after_game, list(cards_in), list(cards_out)))
                cards_in.clear()
                cards_out.clear()
                current_after_game = None

            if is_sb_action and current_after_game is None:
                current_after_game = int(round(obs[_MATCH_CTX_START] * 3.0))

            if cat == _CAT_SB_IN:
                card_raw = obs[STATE_SIZE + MAX_ACTIONS + action]
                if card_raw >= 0:
                    cid = int(round(card_raw * N_CARD_TYPES))
                    name = _VOCAB_NAMES[cid] if 0 <= cid < len(_VOCAB_NAMES) else f"card#{cid}"
                else:
                    name = "?"
                cards_in.append(name)
            elif cat == _CAT_SB_OUT:
                card_raw = obs[STATE_SIZE + MAX_ACTIONS + action]
                if card_raw >= 0:
                    cid = int(round(card_raw * N_CARD_TYPES))
                    name = _VOCAB_NAMES[cid] if 0 <= cid < len(_VOCAB_NAMES) else f"card#{cid}"
                else:
                    name = "?"
                cards_out.append(name)
            elif cat == _CAT_SB_DONE:
                # match_game_number is 0-based; add 1 so "after game 1" is human-readable.
                after_game = (current_after_game + 1) if current_after_game is not None else 0
                phases.append((gi, after_game, list(cards_in), list(cards_out)))
                cards_in.clear()
                cards_out.clear()
                current_after_game = None

        # Flush any in-progress phase (shouldn't happen but safety)
        if cards_in or cards_out:
            after_game = (current_after_game + 1) if current_after_game is not None else 0
            phases.append((gi, after_game, list(cards_in), list(cards_out)))

    if not phases:
        print("  No sideboard phases found. (Are these bo3 games?)")
        return

    # Scripted agent never sideboards (always picks DONE immediately)
    # so we only report model decisions

    # Aggregate: how often each card was boarded in/out
    in_counts = {}
    out_counts = {}
    n_phases_with_changes = 0
    n_phases_no_changes = 0

    for gi, _ag, c_in, c_out in phases:
        if c_in or c_out:
            n_phases_with_changes += 1
        else:
            n_phases_no_changes += 1
        for c in c_in:
            in_counts[c] = in_counts.get(c, 0) + 1
        for c in c_out:
            out_counts[c] = out_counts.get(c, 0) + 1

    total_phases = len(phases)
    print(f"\n  Sideboard Report — {total_phases} sideboard phases across {len(games)} matches")
    print(f"  Phases with changes: {n_phases_with_changes}  |  No changes: {n_phases_no_changes}")
    print(f"  (Scripted agent: never sideboards)")

    if in_counts or out_counts:
        # Cards boarded IN (from sideboard to main)
        print(f"\n  {'Cards boarded IN':<30} {'Count':>6}")
        print(f"  {'-'*30} {'-'*6}")
        for name, cnt in sorted(in_counts.items(), key=lambda x: -x[1]):
            print(f"  {name:<30} {cnt:>6}")

        # Cards boarded OUT (from main to sideboard)
        print(f"\n  {'Cards boarded OUT':<30} {'Count':>6}")
        print(f"  {'-'*30} {'-'*6}")
        for name, cnt in sorted(out_counts.items(), key=lambda x: -x[1]):
            print(f"  {name:<30} {cnt:>6}")

    # Per-game detail — block layout with deduplicated counts (long swap lists
    # wrap badly in a table, so each phase gets its own indented block).
    def _count_str(cards):
        if not cards:
            return "(none)"
        counts = {}
        for c in cards:
            counts[c] = counts.get(c, 0) + 1
        parts = [f"{n}x {name}" if n > 1 else name
                 for name, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
        return ", ".join(parts)

    print(f"\n  Per-match sideboard detail:")
    last_match = None
    for gi, after_game, c_in, c_out in phases:
        if gi != last_match:
            r = games[gi]["result"]
            result = "WIN" if r > 0 else ("LOSS" if r < 0 else "DRAW")
            print(f"    Match {gi} ({result})")
            last_match = gi
        n_in, n_out = len(c_in), len(c_out)
        label = f"after game {after_game}" if after_game else "(unknown game)"
        print(f"      {label}:  +{n_in} / -{n_out}")
        if c_in or c_out:
            print(f"        IN : {_count_str(c_in)}")
            print(f"        OUT: {_count_str(c_out)}")


_CAT_SB_IN, _CAT_SB_OUT, _CAT_SB_DONE = CAT_SIDEBOARD_IN, CAT_SIDEBOARD_OUT, CAT_SIDEBOARD_DONE

# Fungible card classes for the NET sideboard-impact table: swapping one
# fetchland for another is mana-base tuning, not a card-choice signal, so all
# fetches pool into one class (a fetch-for-fetch swap nets to zero). Applies
# only to the net-impact section of sbvalue — the per-swap preference tables
# above it stay per-name.
_SB_FUNGIBLE = {
    name: "Fetchland (any)"
    for name in (
        "Scalding Tarn", "Flooded Strand", "Polluted Delta", "Wooded Foothills",
        "Misty Rainforest", "Windswept Heath", "Bloodstained Mire",
        "Verdant Catacombs", "Arid Mesa", "Marsh Flats", "Prismatic Vista",
    )
}


def _analyze_sbvalue(games, verbose=True):
    """Cardvalue-style report for sideboard decisions: what the model prefers
    to bring in / take out, how confident it is, and whether it pays off.
    Returns {"in": rows, "out": rows, "net": rows} for charting/reporting.

    The `sideboard` report only counts swaps the model actually made; here the
    full legal menu and the recorded policy distribution at every sideboard
    decision are used, so preference over the options the model did NOT take
    is measured too:

      * off   — sideboard phases where the swap was on the menu
      * taken — times the swap was chosen (extra copies count)
      * rate  — phases with ≥1 take / phases offered
      * conf  — mean policy probability on the swap when offered (probability
                mass summed over duplicate copies within one decision)
      * ΔV    — mean change in V(s) across a taken swap
      * ΔWR   — match win rate when taken at least once minus offered-but-
                never-taken (conditions on availability, unlike cardvalue)

    A final NET-impact table works at GAME granularity: every post-board game
    is bucketed, per card class, by the class's cumulative net copies in the
    deck for that game (boarded in minus out over the phases played so far in
    the match; fetchlands pooled as one fungible class per `_SB_FUNGIBLE`),
    and compares the post-board GAME win rate when the card was net-in vs
    net-out vs net-zero. Per-game outcomes are reconstructed from the match
    win counters at each game's first decision (the match winner takes the
    final game).
    """
    # key = (category, card name); phase key = (match idx, phase ordinal)
    prob_sum, prob_n = {}, {}
    offered_phases, taken_phases = {}, {}
    taken_ct, taken_matches, offered_matches = {}, {}, {}
    dv = {}
    # Net-impact accumulators (fungibility-pooled class names).
    offered_cls = {}     # class -> set of match idx where offered in either direction
    net_cls_games = {}   # class -> {"in"/"out"/"zero": [post-board game outcomes]}
    n_phases = 0
    phases_no_swaps = 0
    done_first_probs = []   # P(SB_DONE) at the first decision of each phase

    for gi, g in enumerate(games):
        obs_list = g["observations"]
        actions = g["actions"]
        vals = g.get("values", [])
        probs_list = g.get("action_probs", [])
        ncs = g["num_choices"]
        in_phase = False
        phase_id = -1
        phase_swaps = 0
        swap_events = []   # (after_game_number, class, ±1) for this match
        game_starts = {}   # game_number -> (self_wins, opp_wins) at first decision

        for si in range(len(obs_list)):
            obs = obs_list[si]
            meta = _match_meta(obs)
            if not meta[3]:
                if meta[0] not in game_starts:
                    game_starts[meta[0]] = (meta[1], meta[2])
                if in_phase:
                    n_phases += 1
                    phases_no_swaps += (phase_swaps == 0)
                    in_phase = False
                continue
            if not in_phase:
                in_phase = True
                phase_id += 1
                phase_swaps = 0
            pkey = (gi, phase_id)
            probs = probs_list[si] if si < len(probs_list) else None

            # Menu scan: per-card probability mass on each direction.
            dec_mass = {}
            for k in range(ncs[si]):
                cat = _obs_action_cat(obs, k)
                if cat in (_CAT_SB_IN, _CAT_SB_OUT):
                    cid = _obs_card_id(obs, k)
                    if 0 <= cid < len(_VOCAB_NAMES):
                        name = decode.card_index_to_name(cid)
                        key = (cat, name)
                        p = float(probs[k]) if probs is not None and k < len(probs) else 0.0
                        dec_mass[key] = dec_mass.get(key, 0.0) + p
                        offered_cls.setdefault(_SB_FUNGIBLE.get(name, name), set()).add(gi)
            # P(done) at the phase's first decision = confidence in the current 60.
            if probs is not None and (si == 0 or not _match_meta(obs_list[si - 1])[3]):
                for k in range(ncs[si]):
                    if _obs_action_cat(obs, k) == _CAT_SB_DONE and k < len(probs):
                        done_first_probs.append(float(probs[k]))
                        break
            for key, mass in dec_mass.items():
                prob_sum[key] = prob_sum.get(key, 0.0) + mass
                prob_n[key] = prob_n.get(key, 0) + 1
                offered_phases.setdefault(key, set()).add(pkey)
                offered_matches.setdefault(key, set()).add(gi)

            # Chosen-swap attribution.
            action = actions[si]
            cat = _obs_action_cat(obs, action)
            if cat in (_CAT_SB_IN, _CAT_SB_OUT):
                cid = _obs_card_id(obs, action)
                if 0 <= cid < len(_VOCAB_NAMES):
                    name = decode.card_index_to_name(cid)
                    key = (cat, name)
                    taken_ct[key] = taken_ct.get(key, 0) + 1
                    taken_phases.setdefault(key, set()).add(pkey)
                    taken_matches.setdefault(key, set()).add(gi)
                    phase_swaps += 1
                    if si + 1 < len(vals):
                        dv.setdefault(key, []).append(vals[si + 1] - vals[si])
                    cls = _SB_FUNGIBLE.get(name, name)
                    swap_events.append((meta[0], cls,
                                        1 if cat == _CAT_SB_IN else -1))
        if in_phase:  # trace ended inside a sideboard phase
            n_phases += 1
            phases_no_swaps += (phase_swaps == 0)

        # Per-game outcomes: a game was won iff self_wins ticked up by the next
        # game's start; the final game goes to the match winner. Then bucket
        # each POST-BOARD game by every offered class's cumulative net copies
        # in the deck for that game (swaps after game k apply to games > k).
        gns = sorted(game_starts)
        outcomes = {}
        for idx, gn in enumerate(gns):
            if idx + 1 < len(gns):
                outcomes[gn] = 1 if game_starts[gns[idx + 1]][0] > game_starts[gn][0] else 0
            elif g["result"] != 0:
                outcomes[gn] = 1 if g["result"] > 0 else 0
        match_classes = {cls for cls, ms in offered_cls.items() if gi in ms}
        for gn in gns:
            if gn == 0 or gn not in outcomes:
                continue  # game 1 is pre-board; an unfinished draw has no outcome
            for cls in match_classes:
                net = sum(d for agn, c, d in swap_events if c == cls and agn < gn)
                bucket = "in" if net > 0 else ("out" if net < 0 else "zero")
                net_cls_games.setdefault(
                    cls, {"in": [], "out": [], "zero": []})[bucket].append(outcomes[gn])

    result = {"in": [], "out": [], "net": []}
    if n_phases == 0:
        if verbose:
            print("  No sideboard phases found. (Are these bo3 games?)")
        return result

    if verbose:
        print(f"\nSideboard preference — {n_phases} phases across {len(games)} matches")
        print(f"  Phases with no swaps: {phases_no_swaps}/{n_phases}", end="")
        if done_first_probs:
            print(f"  |  mean P(done) at first sideboard decision: "
                  f"{np.mean(done_first_probs):.0%}")
        else:
            print()
        print("  conf = mean policy prob on the swap when offered.  "
              "ΔWR = taken-at-least-once minus offered-but-never-taken.")

    for direction, dkey, label in ((_CAT_SB_IN, "in", "Boarded IN"),
                                   (_CAT_SB_OUT, "out", "Boarded OUT")):
        keys = [k for k in offered_phases if k[0] == direction]
        if not keys:
            continue
        rows = []
        for key in keys:
            off = len(offered_phases[key])
            t_phases = taken_phases.get(key, set())
            t_matches = taken_matches.get(key, set())
            not_taken = offered_matches[key] - t_matches
            wr_taken = (sum(1 for i in t_matches if games[i]["result"] > 0) / len(t_matches)
                        if t_matches else None)
            wr_not = (sum(1 for i in not_taken if games[i]["result"] > 0) / len(not_taken)
                      if not_taken else None)
            deltas = dv.get(key, [])
            rows.append({
                "card": key[1],
                "off": off,
                "taken": taken_ct.get(key, 0),
                "rate": len(t_phases) / off if off else None,
                "conf": prob_sum[key] / prob_n[key] if prob_n.get(key) else None,
                "dv": float(np.mean(deltas)) if deltas else None,
                "wr_lift": (wr_taken - wr_not
                            if wr_taken is not None and wr_not is not None else None),
            })
        rows.sort(key=lambda r: (-(r["conf"] or 0.0), -r["taken"]))
        result[dkey] = rows

        if not verbose:
            continue
        print(f"\n  {label:<26} {'off':>4} {'taken':>6} {'rate':>6} {'conf':>6} "
              f"{'ΔV':>8} {'ΔWR':>6}")
        print(f"  {'-'*26} {'-'*4} {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*6}")
        for r in rows:
            rate_str = f"{r['rate']*100:5.0f}%" if r["rate"] is not None else "    —"
            conf_str = f"{r['conf']*100:5.0f}%" if r["conf"] is not None else "    —"
            dv_str = f"{r['dv']:+8.3f}" if r["dv"] is not None else "      — "
            lift_str = f"{r['wr_lift']*100:+5.0f}%" if r["wr_lift"] is not None else "    —"
            print(f"  {r['card'][:26]:<26} {r['off']:>4} {r['taken']:>6} "
                  f"{rate_str} {conf_str} {dv_str} {lift_str}")

    # ---- Net impact per post-board GAME ---------------------------------------
    # Each post-board game was bucketed above by the class's cumulative net
    # copies in the deck for that game. Fungible classes: a fetch swapped for
    # another fetch nets to zero under "Fetchland (any)".
    def _gwr(outcome_list):
        return (sum(outcome_list) / len(outcome_list)) if outcome_list else None

    net_rows = []
    for cls, buckets in net_cls_games.items():
        if not (buckets["in"] or buckets["out"]):
            continue  # offered but never net-moved for any post-board game
        wr_in, wr_out, wr_zero = (_gwr(buckets["in"]), _gwr(buckets["out"]),
                                  _gwr(buckets["zero"]))
        net_rows.append({
            "card": cls,
            "n_in": len(buckets["in"]), "n_out": len(buckets["out"]),
            "n_zero": len(buckets["zero"]),
            "wr_in": wr_in, "wr_out": wr_out, "wr_zero": wr_zero,
            "dwr_in": (wr_in - wr_zero
                       if wr_in is not None and wr_zero is not None else None),
            "dwr_out": (wr_out - wr_zero
                        if wr_out is not None and wr_zero is not None else None),
        })
    net_rows.sort(key=lambda r: -(r["n_in"] + r["n_out"]))
    result["net"] = net_rows

    if verbose and net_rows:
        print(f"\n  Net impact — post-board GAME win rate, bucketed by whether the card")
        print(f"  was net-in / net-out of the deck for that game (fetchlands pooled as")
        print(f"  one fungible class — fetch-for-fetch swaps net to zero). Counts are")
        print(f"  post-board games. ΔWR columns are relative to that card's net-zero games.")
        print(f"\n  {'card':<26} {'in':>4} {'out':>4} {'zero':>5} "
              f"{'WR(in)':>7} {'WR(out)':>8} {'WR(0)':>7} {'ΔWR(in)':>8} {'ΔWR(out)':>9}")
        print(f"  {'-'*26} {'-'*4} {'-'*4} {'-'*5} {'-'*7} {'-'*8} {'-'*7} {'-'*8} {'-'*9}")
        for r in net_rows:
            def _pct(v, w):
                return f"{v*100:{w-1}.0f}%" if v is not None else " " * (w - 1) + "—"
            def _spct(v, w):
                return f"{v*100:+{w-1}.0f}%" if v is not None else " " * (w - 1) + "—"
            print(f"  {r['card'][:26]:<26} {r['n_in']:>4} {r['n_out']:>4} {r['n_zero']:>5} "
                  f"{_pct(r['wr_in'], 7)} {_pct(r['wr_out'], 8)} {_pct(r['wr_zero'], 7)} "
                  f"{_spct(r['dwr_in'], 8)} {_spct(r['dwr_out'], 9)}")
    elif verbose:
        print("\n  Net impact: no card ever net-entered or net-left the deck.")
    return result


def _interactive_session(ctx):
    """Interactive REPL for inspecting simulation results.

    ctx keys: games, swing_data, shap_values, shap_samples,
              model, env, opp_model, args
    """
    try:
        import readline
        readline.set_history_length(500)
    except ImportError:
        pass

    games = ctx["games"]
    args  = ctx.get("args")

    def _banner():
        can_run = ctx.get("env") is not None
        print("\n" + "=" * 60)
        print(f"Interactive session — {len(games)} games in memory.")
        cmds = ["list", "replay <N> [-v]", "boardstate <N> <step>", "summary",
                "cardvalue [N]", "targeting", "sideboard", "sbvalue",
                "swings [N]", "boundaries", "matchcal",
                "shap", "regret [N]", "entropy", "consistency [N]",
                "calibration", "turning", "clusters", "whatif <N> <step> [k]",
                "chart <N>", "chart swings [N]", "chart cardvalue [N]",
                "chart sbvalue [N]", "chart shap",
                "chart calibration", "chart turning", "chart clusters", "chart whatif"]
        if can_run:
            cmds.append("run <N>")
        cmds += ["help", "quit"]
        print("Commands: " + ", ".join(cmds))
        print("=" * 60)

    _banner()

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q"):
            break

        elif cmd in ("help", "?", "h"):
            can_run = ctx.get("env") is not None
            print("  list / games              — list all games with result/decisions/side")
            print("  replay <N> [-v]           — one-line-per-decision trace for game N; -v adds")
            print("                              zones (battlefields/hand/GYs/stack), chosen action,")
            print("                              and interleaved opponent actions")
            print("  boardstate <N> [step]     — full board + decision at step in game N (default 0)")
            print("  bs <N> [step]             — alias; enters GDB-style stepping mode")
            print("  summary                   — win/loss/draw stats for all simulated games")
            print("  swings [N]                — show top N in-game value-function swings (default 10;")
            print("                              same bo3 game, sideboard steps excluded)")
            print("  boundaries                — V(s) across bo3 game transitions: result-pricing vs")
            print("                              board re-anchoring components per boundary")
            print("  matchcal                  — per match/game: V at each game's first decision vs the")
            print("                              empirical remaining return for that match score")
            print("  shap [n_bg N] [n_smp N]   — run SHAP analysis on collected game data")
            print("  regret [N]                — policy regret analysis (top N high-regret decisions)")
            print("  entropy                   — policy entropy by phase and board state")
            print("  consistency [N]           — find similar states with different actions (top N pairs)")
            print("  cardvalue [N]             — rank cards by importance (ΔV, priority, win-rate lift)")
            print("  targeting                 — self vs opp targeting, hold vs cast analysis")
            print("  sideboard                 — sideboard decisions by each agent (bo3)")
            print("  sbvalue                   — sideboard preference: take-rate, policy confidence,")
            print("                              ΔV and win-rate lift per boarded-in/out card, plus")
            print("                              NET impact on post-board GAME win rate with")
            print("                              fetchlands pooled as one fungible class (bo3)")
            print("  whatif <N> <step> [k]     — counterfactual: branch the chosen action + top-k")
            print("                              alternatives from the same seed, roll each to the end,")
            print("                              and compare final result / V(s) (k default 3)")
            print("  calibration               — V(s) at game start vs actual win rate (is model biased?)")
            print("  turning                   — find the 'point of no return' in each game")
            print("  clusters                  — classify games by V(s) curve shape (archetypes)")
            print("  chart <N>                 — value curve plot for game N")
            print("  chart swings [N]          — value curve plots for top N swing games")
            print("  chart cardvalue [N]       — diverging bar chart of per-card ΔV")
            print("  chart sbvalue [N]         — sideboard swap preference + net ΔWR bars")
            print("  chart shap                — SHAP summary plot (requires shap run first)")
            print("  chart calibration         — calibration curve plot")
            print("  chart turning             — turning point distribution plot")
            print("  chart clusters            — overlay V(s) curves by archetype")
            print("  chart whatif              — overlay branch V(s) curves from the last whatif")
            if can_run:
                print("  run <N>                   — simulate N more games and add to pool")
            print("  quit / exit               — leave interactive session")

        elif cmd in ("list", "games", "ls"):
            print(f"  {'Game':<6} {'Result':<8} {'Score':<7} {'Decisions':<12} {'Side'}")
            print(f"  {'-'*6} {'-'*8} {'-'*7} {'-'*12} {'-'*4}")
            for i, g in enumerate(games):
                r = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
                sc = _match_score(g)
                sc_str = f"{sc[0]}-{sc[1]}" if sc is not None else "—"
                side = "A" if g["model_is_a"] else "B"
                print(f"  {i:<6} {r:<8} {sc_str:<7} {len(g['values']):<12} {side}")

        elif cmd == "summary":
            _sim_summary(games)

        elif cmd == "replay":
            if len(parts) < 2:
                print("  Usage: replay <game_index> [-v]")
                continue
            flags = [p for p in parts[1:] if p in ("-v", "v", "verbose", "full")]
            args_ = [p for p in parts[1:] if p not in flags]
            if not args_:
                print("  Usage: replay <game_index> [-v]")
                continue
            try:
                n = int(args_[0])
            except ValueError:
                print("  Expected an integer game index.")
                continue
            if n < 0 or n >= len(games):
                print(f"  Game index out of range. Valid range: 0–{len(games) - 1}")
                continue
            _replay_sim_game(games[n], n, verbose=bool(flags))

        elif cmd in ("boardstate", "bs"):
            if len(parts) < 2:
                print("  Usage: boardstate <game_index> [decision_step]")
                continue
            try:
                gn = int(parts[1])
                step = int(parts[2]) if len(parts) >= 3 else 0
            except ValueError:
                print("  Expected integer game_index and optional decision_step.")
                continue
            if gn < 0 or gn >= len(games):
                print(f"  Game index out of range. Valid range: 0–{len(games) - 1}")
                continue

            def _opp_actions_before(g, step):
                # Opponent actions that occurred between model decision step-1 and
                # model decision `step` (before_model_step == step), with runs of
                # identical consecutive actions collapsed ("PASS (x19)").
                descs = [oa["desc"] for oa in g.get("opp_actions", [])
                         if oa["before_model_step"] == step]
                if not descs:
                    return
                print(f"  Opponent actions since decision {step - 1}:")
                run_desc, run_len = None, 0
                def flush():
                    if run_len == 1:
                        print(f"        opp --> {run_desc}")
                    elif run_len > 1:
                        print(f"        opp --> {run_desc} (x{run_len})")
                for desc in descs:
                    if desc == run_desc:
                        run_len += 1
                    else:
                        flush()
                        run_desc, run_len = desc, 1
                flush()

            def _show_step(g, gn, step):
                n_obs = len(g["observations"])
                obs = g["observations"][step]
                val = g["values"][step] if step < len(g["values"]) else None
                result_str = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
                print(f"\nGame {gn} [{result_str}]  —  decision {step}/{n_obs - 1}")
                _opp_actions_before(g, step)

                # Model's decision at this step
                has_action = "actions" in g and step < len(g["actions"])
                if has_action:
                    action_idx  = g["actions"][step]
                    num_ch      = g["num_choices"][step]
                    action_lines = _decode_legal_actions(obs, num_ch, action_idx)
                    print(f"  Legal actions ({num_ch}):")
                    for ln in action_lines:
                        print(ln)
                print()
                _decode_board_state(obs, value=val)

            g = games[gn]
            n_obs = len(g["observations"])
            if step < 0 or step >= n_obs:
                print(f"  Step out of range for game {gn}. Valid range: 0–{n_obs - 1}")
                continue
            _show_step(g, gn, step)

            # GDB-style stepping sub-loop
            last_step_cmd = "n"
            print("  Stepping mode: n/Enter=next  p=prev  g <N>=go to step  q=quit stepping")
            while True:
                try:
                    raw = input(f"(g{gn}:{step}) ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                sc = raw.lower() if raw else last_step_cmd
                sp2 = sc.split()
                scmd = sp2[0] if sp2 else last_step_cmd

                if scmd in ("n", "next", ""):
                    last_step_cmd = "n"
                    if step < n_obs - 1:
                        step += 1
                        _show_step(g, gn, step)
                    else:
                        print("  End of game.")
                elif scmd in ("p", "prev", "previous", "b", "back"):
                    last_step_cmd = "p"
                    if step > 0:
                        step -= 1
                        _show_step(g, gn, step)
                    else:
                        print("  Beginning of game.")
                elif scmd == "g":
                    if len(sp2) < 2:
                        print("  Usage: g <step>")
                        continue
                    try:
                        target = int(sp2[1])
                    except ValueError:
                        print("  Expected an integer step.")
                        continue
                    if target < 0 or target >= n_obs:
                        print(f"  Step out of range. Valid range: 0–{n_obs - 1}")
                        continue
                    step = target
                    _show_step(g, gn, step)
                elif scmd in ("q", "quit", "exit"):
                    break
                else:
                    print("  n/Enter=next  p=prev  g <N>=go to step  q=quit stepping")

        elif cmd == "swings":
            top_n = 10
            if len(parts) >= 2:
                try:
                    top_n = int(parts[1])
                except ValueError:
                    pass
            if ctx["swing_data"] is None:
                print("  Computing value swings...")
                ctx["swing_data"] = _compute_swings(games)
            top = ctx["swing_data"][:top_n]
            print(f"\nTop {min(top_n, len(top))} value swings (in-game only; "
                  f"see 'boundaries' for bo3 game transitions):")
            _print_swing_table(top)

        elif cmd in ("boundaries", "boundary"):
            _print_boundaries(_compute_boundaries(games))

        elif cmd == "matchcal":
            _print_match_calibration(games)

        elif cmd == "shap":
            n_background = 50
            n_samples    = 200
            for j in range(1, len(parts) - 1, 2):
                if parts[j] == "n_bg":
                    try: n_background = int(parts[j + 1])
                    except ValueError: pass
                elif parts[j] == "n_smp":
                    try: n_samples = int(parts[j + 1])
                    except ValueError: pass
            if args is not None:
                n_background = getattr(args, "n_background", n_background)
                n_samples    = getattr(args, "n_samples", n_samples)
            try:
                import shap
                from sklearn.ensemble import GradientBoostingRegressor
                all_interp = np.array([f for g in games for f in g["interp_features"]])
                all_vals   = np.array([v for g in games for v in g["values"]])
                print(f"\nFitting surrogate on {len(all_interp)} points...")
                surrogate = GradientBoostingRegressor(
                    n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8)
                surrogate.fit(all_interp, all_vals)
                r2 = surrogate.score(all_interp, all_vals)
                print(f"Surrogate R^2: {r2:.4f}")
                bg_idx  = np.random.choice(len(all_interp),
                                           size=min(n_background, len(all_interp)), replace=False)
                smp_idx = np.random.choice(len(all_interp),
                                           size=min(n_samples, len(all_interp)), replace=False)
                print(f"Running SHAP ({n_background} background, {len(smp_idx)} samples)...")
                explainer  = shap.KernelExplainer(surrogate.predict, all_interp[bg_idx])
                shap_vals  = explainer.shap_values(all_interp[smp_idx])
                ctx["shap_values"]  = shap_vals
                ctx["shap_samples"] = all_interp[smp_idx]
                mean_abs = np.abs(shap_vals).mean(axis=0)
                sidx = np.argsort(-mean_abs)
                print(f"\n{'Feature':<25} {'Mean |SHAP|':>12}")
                print("-" * 40)
                for idx in sidx:
                    print(f"  {_INTERP_FEATURE_NAMES[idx]:<23} {mean_abs[idx]:12.4f}")
            except ImportError as e:
                print(f"  Missing dependency: {e}")
            except Exception as e:
                print(f"  Error running SHAP: {e}")

        elif cmd == "chart":
            sub = parts[1].lower() if len(parts) >= 2 else ""
            plt = viz.pyplot(show=viz.want_show(args))
            if plt is None:
                print("  matplotlib unavailable.")
                continue

            if sub == "cardvalue":
                top_n = 20
                if len(parts) >= 3:
                    try: top_n = int(parts[2])
                    except ValueError: pass
                rows = ctx.get("cardvalue_rows")
                if rows is None:
                    rows = _analyze_cardvalue(games, verbose=False)
                    ctx["cardvalue_rows"] = rows
                _chart_cardvalue(rows, args=args, top_n=top_n)

            elif sub == "sbvalue":
                top_n = 20
                if len(parts) >= 3:
                    try: top_n = int(parts[2])
                    except ValueError: pass
                sbrows = ctx.get("sbvalue_rows")
                if sbrows is None:
                    sbrows = _analyze_sbvalue(games, verbose=False)
                    ctx["sbvalue_rows"] = sbrows
                _chart_sbvalue(sbrows, args=args, top_n=top_n)

            elif sub == "shap":
                if ctx["shap_values"] is None or ctx["shap_samples"] is None:
                    print("  Run 'shap' first to generate SHAP values.")
                    continue
                try:
                    import shap
                    shap.summary_plot(ctx["shap_values"], ctx["shap_samples"],
                                      feature_names=_INTERP_FEATURE_NAMES, show=False)
                    plt.tight_layout()
                    viz.save_or_show(plt, plt.gcf(), "shap_summary", args)
                except Exception as e:
                    print(f"  SHAP plot error: {e}")

            elif sub == "swings":
                top_n = 5
                if len(parts) >= 3:
                    try: top_n = int(parts[2])
                    except ValueError: pass
                if ctx["swing_data"] is None:
                    ctx["swing_data"] = _compute_swings(games)
                top = ctx["swing_data"][:top_n]
                if not top:
                    print("  No swing data.")
                    continue
                fig, axes = plt.subplots(len(top), 1, figsize=(10, 3 * len(top)), squeeze=False)
                for i, s in enumerate(top):
                    ax = games[s["game_idx"]]
                    vals = ax["values"]
                    result_str = "WIN" if ax["result"] > 0 else ("LOSS" if ax["result"] < 0 else "DRAW")
                    a = axes[i, 0]
                    a.plot(vals, color="steelblue", linewidth=1.2)
                    a.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                    a.axvline(s["swing_step"], color="red", linewidth=1, linestyle="--",
                              label=f"swing ({s['swing_to'] - s['swing_from']:+.2f})")
                    a.set_ylabel("V(s)")
                    a.set_title(f"Game {s['game_idx']} ({result_str})")
                    a.legend(loc="upper right", fontsize=8)
                    a.grid(True, alpha=0.3)
                axes[-1, 0].set_xlabel("Decision step")
                viz.save_or_show(plt, fig, "swings", args)

            elif sub == "calibration":
                cal = ctx.get("calibration_data")
                if cal is None:
                    print("  Computing calibration...")
                    cal = _analyze_calibration(games, verbose=False)
                    ctx["calibration_data"] = cal
                if not cal:
                    print("  No calibration data.")
                    continue
                fig, ax = plt.subplots(figsize=(8, 6))
                mean_vs = [b["mean_v"] for b in cal]
                win_rates = [b["win_rate"] for b in cal]
                counts = [b["n"] for b in cal]
                labels = [b["label"] for b in cal]
                # Scatter with size proportional to count
                sizes = [max(40, min(300, c * 5)) for c in counts]
                ax.scatter(mean_vs, win_rates, s=sizes, c="steelblue",
                           alpha=0.7, edgecolors="navy", zorder=3)
                for i, lab in enumerate(labels):
                    ax.annotate(f"{lab}\n(n={counts[i]})",
                                (mean_vs[i], win_rates[i]),
                                textcoords="offset points", xytext=(8, 8),
                                fontsize=7)
                # Perfect calibration line: V(s) maps to win rate as (V+1)/2
                xs = np.linspace(-1, 1, 50)
                ax.plot(xs, (xs + 1) / 2, color="gray", linestyle="--",
                        linewidth=1, label="Perfect calibration", alpha=0.6)
                ax.set_xlabel("Mean V(s) at game start")
                ax.set_ylabel("Actual win rate")
                ax.set_title("Value Function Calibration")
                ax.legend(loc="upper left", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-0.05, 1.05)
                viz.save_or_show(plt, fig, "calibration", args)

            elif sub == "turning":
                tp_data = ctx.get("turning_data")
                if tp_data is None:
                    print("  Computing turning points...")
                    tp_data = _analyze_turning_points(games, verbose=False)
                    ctx["turning_data"] = tp_data
                if not tp_data:
                    print("  No turning points found.")
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Histogram of turning point timing
                ax = axes[0]
                win_fracs = [t["frac"] for t in tp_data if t["result"] > 0]
                loss_fracs = [t["frac"] for t in tp_data if t["result"] < 0]
                bins_hist = np.linspace(0, 1, 15)
                if win_fracs:
                    ax.hist(win_fracs, bins=bins_hist, alpha=0.6,
                            color="green", label=f"Wins ({len(win_fracs)})")
                if loss_fracs:
                    ax.hist(loss_fracs, bins=bins_hist, alpha=0.6,
                            color="red", label=f"Losses ({len(loss_fracs)})")
                ax.set_xlabel("Fraction of game elapsed")
                ax.set_ylabel("Count")
                ax.set_title("When Turning Points Occur")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

                # V(s) curves for a few games with turning points marked
                ax = axes[1]
                n_show = min(8, len(tp_data))
                colors_win = plt.cm.Greens(np.linspace(0.4, 0.9, n_show))
                colors_loss = plt.cm.Reds(np.linspace(0.4, 0.9, n_show))
                ci_w = 0
                ci_l = 0
                for t in tp_data[:n_show]:
                    g = games[t["game_idx"]]
                    vals = g["values"]
                    won = t["result"] > 0
                    if won:
                        c = colors_win[ci_w % len(colors_win)]
                        ci_w += 1
                    else:
                        c = colors_loss[ci_l % len(colors_loss)]
                        ci_l += 1
                    xs = np.linspace(0, 1, len(vals))
                    ax.plot(xs, vals, color=c, alpha=0.5, linewidth=1)
                    ax.axvline(t["frac"], color=c, linestyle=":",
                               linewidth=0.8, alpha=0.6)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.set_xlabel("Fraction of game elapsed")
                ax.set_ylabel("V(s)")
                ax.set_title(f"V(s) Curves with Turning Points ({n_show} games)")
                ax.grid(True, alpha=0.3)

                viz.save_or_show(plt, fig, "turning", args)

            elif sub == "clusters":
                clust = ctx.get("cluster_data")
                if clust is None:
                    print("  Computing clusters...")
                    clust = _analyze_clusters(games, verbose=False)
                    ctx["cluster_data"] = clust
                archetype_colors = {
                    "early_lead_held": "green",
                    "slow_grind": "steelblue",
                    "comeback": "orange",
                    "lead_blown": "red",
                    "volatile": "purple",
                }
                nonempty = {k: v for k, v in clust.items() if v}
                if not nonempty:
                    print("  No cluster data.")
                    continue
                n_types = len(nonempty)
                fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4),
                                         squeeze=False)
                for col, (label, indices) in enumerate(nonempty.items()):
                    ax = axes[0, col]
                    n_plot = min(15, len(indices))
                    for i in indices[:n_plot]:
                        g = games[i]
                        vals = g["values"]
                        xs = np.linspace(0, 1, len(vals))
                        won = g["result"] > 0
                        ax.plot(xs, vals, color=archetype_colors.get(label, "gray"),
                                alpha=0.3, linewidth=1)
                    # Plot mean curve
                    if indices:
                        max_len = max(len(games[i]["values"]) for i in indices)
                        interp_vals = []
                        for i in indices:
                            v = games[i]["values"]
                            xs_orig = np.linspace(0, 1, len(v))
                            xs_new = np.linspace(0, 1, max_len)
                            interp_vals.append(np.interp(xs_new, xs_orig, v))
                        mean_curve = np.mean(interp_vals, axis=0)
                        xs_mean = np.linspace(0, 1, max_len)
                        ax.plot(xs_mean, mean_curve,
                                color=archetype_colors.get(label, "gray"),
                                linewidth=2.5, label="mean")
                    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                    w = sum(1 for i in indices if games[i]["result"] > 0)
                    ax.set_title(f"{label}\n({len(indices)} games, "
                                 f"{w}/{len(indices)} wins)")
                    ax.set_xlabel("Game progress")
                    ax.set_ylabel("V(s)")
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(-1.1, 1.1)
                viz.save_or_show(plt, fig, "clusters", args)

            elif sub == "whatif":
                wf = ctx.get("whatif_data")
                if not wf:
                    print("  Run 'whatif <game> <step> [k]' first to generate branches.")
                    continue
                gn, step, branches = wf["game_idx"], wf["step"], wf["branches"]
                g = games[gn]
                actual_vals = g["values"]
                fig, ax = plt.subplots(figsize=(11, 5))
                # Actual game V(s) up to and including the branch point.
                xs_actual = list(range(len(actual_vals)))
                ax.plot(xs_actual, actual_vals, color="black", linewidth=1.4,
                        alpha=0.5, label="actual game")
                ax.axvline(step, color="gray", linestyle="--", linewidth=1,
                           label=f"branch @ step {step}")
                for b in branches:
                    # Each branch's V(s) trajectory starts at the decision after
                    # the branch, so offset the x-axis to step+1.
                    bx = list(range(step + 1, step + 1 + len(b["values"])))
                    res = b["result"]
                    rstr = "W" if res > 0 else ("L" if res < 0 else "D")
                    lab = ("[chosen] " if b["is_chosen"] else "") + f"{b['desc']} → {rstr}"
                    lw = 2.0 if b["is_chosen"] else 1.1
                    ax.plot(bx, b["values"], linewidth=lw, alpha=0.85,
                            label=(lab[:40] + "…") if len(lab) > 41 else lab)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.set_xlabel("Model decision step")
                ax.set_ylabel("V(s)")
                ax.set_title(f"Whatif — game {gn}, branch at step {step}")
                ax.legend(loc="best", fontsize=7)
                ax.grid(True, alpha=0.3)
                viz.save_or_show(plt, fig, f"whatif_g{gn}_s{step}", args)

            else:
                # chart <N> — value curve for a single game
                try:
                    gn = int(sub)
                except ValueError:
                    print("  Usage: chart <game_index> | chart swings [N] | chart cardvalue [N] "
                          "| chart sbvalue [N] | chart shap | chart calibration | chart turning "
                          "| chart clusters | chart whatif")
                    continue
                if gn < 0 or gn >= len(games):
                    print(f"  Game index out of range. Valid range: 0–{len(games) - 1}")
                    continue
                g = games[gn]
                vals = g["values"]
                result_str = "WIN" if g["result"] > 0 else ("LOSS" if g["result"] < 0 else "DRAW")
                side = "A" if g["model_is_a"] else "B"
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(vals, color="steelblue", linewidth=1.2)
                ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
                ax.set_xlabel("Decision step")
                ax.set_ylabel("V(s)")
                ax.set_title(f"Game {gn} — Model={side}, {result_str}")
                ax.grid(True, alpha=0.3)
                viz.save_or_show(plt, fig, f"game{gn}", args)

        elif cmd == "regret":
            top_n = 20
            if len(parts) >= 2:
                try: top_n = int(parts[1])
                except ValueError: pass
            has_probs = any(g.get("action_probs") for g in games)
            if not has_probs:
                print("  No action probability data. Re-collect games with a model to enable regret analysis.")
            else:
                _analyze_regret(games, top_n=top_n)

        elif cmd == "entropy":
            has_probs = any(g.get("action_probs") for g in games)
            if not has_probs:
                print("  No action probability data. Re-collect games with a model to enable entropy analysis.")
            else:
                _analyze_entropy(games)

        elif cmd == "consistency":
            top_n = 20
            if len(parts) >= 2:
                try: top_n = int(parts[1])
                except ValueError: pass
            _analyze_consistency(games, top_n=top_n)

        elif cmd == "calibration":
            ctx["calibration_data"] = _analyze_calibration(games)

        elif cmd == "turning":
            ctx["turning_data"] = _analyze_turning_points(games)

        elif cmd == "clusters":
            ctx["cluster_data"] = _analyze_clusters(games)

        elif cmd == "cardvalue":
            top_n = 30
            if len(parts) >= 2:
                try: top_n = int(parts[1])
                except ValueError: pass
            has_probs = any(g.get("action_probs") for g in games)
            if not has_probs:
                print("  Note: no policy probabilities in these traces — "
                      "'prio' column will be blank.")
            ctx["cardvalue_rows"] = _analyze_cardvalue(games, top_n=top_n)

        elif cmd == "targeting":
            _sim_targeting(games)

        elif cmd == "sideboard":
            _sim_sideboard_report(games)

        elif cmd == "sbvalue":
            ctx["sbvalue_rows"] = _analyze_sbvalue(games)

        elif cmd == "whatif":
            if len(parts) < 3:
                print("  Usage: whatif <game_index> <step> [k]   "
                      "(k = # of alternative actions, default 3)")
                continue
            try:
                gn = int(parts[1])
                step = int(parts[2])
                k = int(parts[3]) if len(parts) >= 4 else 3
            except ValueError:
                print("  Expected integer game_index, step, and optional k.")
                continue
            if gn < 0 or gn >= len(games):
                print(f"  Game index out of range. Valid range: 0–{len(games) - 1}")
                continue
            branches = _run_whatif(ctx.get("model"), ctx.get("env"),
                                   ctx.get("opp_model"), games[gn], gn, step, k)
            if branches:
                ctx["whatif_data"] = {"game_idx": gn, "step": step, "branches": branches}

        elif cmd == "run":
            if ctx.get("env") is None or ctx.get("model") is None:
                print("  No live env available. Use the 'interactive' command to enable 'run'.")
                continue
            try:
                n = int(parts[1]) if len(parts) >= 2 else 10
            except ValueError:
                print("  Usage: run <N>")
                continue
            print(f"  Simulating {n} more games...")
            new_games = _collect_game_traces(ctx["model"], ctx["env"],
                                             ctx.get("opp_model"), n)
            games.extend(new_games)
            ctx["swing_data"] = None  # invalidate cached data
            ctx["calibration_data"] = None
            ctx["turning_data"] = None
            ctx["cluster_data"] = None
            ctx["whatif_data"] = None
            print(f"  Pool now has {len(games)} games.")

        else:
            print(f"  Unknown command: {cmd!r}. Type 'help' for available commands.")




def _board_bucket_from_feat(feat):
    """Categorize board state from interpretable features into (life_bucket, board_bucket, timing_bucket)."""
    life_diff = feat[_FEAT["life_diff"]]
    creature_diff = feat[_FEAT["creature_diff"]]
    # Timing: use hand size + land count as a rough game-phase proxy
    self_lands = feat[_FEAT["self_lands"]]
    if self_lands <= 2:
        timing = "early"
    elif self_lands <= 4:
        timing = "mid"
    else:
        timing = "late"

    if life_diff <= -5:
        life = "behind"
    elif life_diff >= 5:
        life = "ahead"
    else:
        life = "even"

    if creature_diff <= -1:
        board = "behind"
    elif creature_diff >= 1:
        board = "ahead"
    else:
        board = "even"

    return life, board, timing


def _analyze_regret(games, top_n=20, verbose=True):
    """Compute policy-based regret/uncertainty proxies from collected game traces.

    Per decision it reports raw regret plus five menu-size-aware metrics (A-E) so
    large menus (e.g. sideboarding, N up to ~40) don't top the ranking purely
    because raw regret floors at 1 - 1/N for a uniform policy:
      Raw = 1 - P(chosen)                menu-size biased
      A   = regret / (1 - 1/N)           uniform-normalized; 1.0 == uniform policy
      B   = H(policy) / ln(N)            normalized entropy, 0 (decisive)..1 (uniform)
      C   = (exp(H) - 1) / (N - 1)       normalized perplexity, 0..1
      D   = P(chosen) - P(2nd best)      top-2 margin (high == decisive)
      E   = -ln P(chosen) / ln(N)        normalized log-loss; 1.0 == uniform policy

    Entries are sorted by descending A (the uniform-normalized regret).
    """
    entries = []
    for g_idx, game in enumerate(games):
        probs_list = game.get("action_probs", [])
        if not probs_list:
            continue
        for step, (probs, action, nc, obs, feat) in enumerate(zip(
                probs_list, game["actions"], game["num_choices"],
                game["observations"], game["interp_features"])):
            chosen_prob = probs[action]
            sorted_probs = np.sort(probs)[::-1]
            second_best = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
            regret = 1.0 - chosen_prob            # raw: 1 - P(chosen)
            margin = chosen_prob - second_best    # D: gap from chosen to runner-up
            # Menu-size-normalized variants (A/B/C/E) so large menus (e.g.
            # sideboarding) don't dominate the ranking purely because raw regret
            # floors at 1 - 1/N for a uniform policy. For nc<=1 there's no real
            # decision, so all normalized metrics are 0.
            if nc > 1:
                ln_n = np.log(nc)
                p = np.asarray(probs[:nc], dtype=float)
                p_safe = p[p > 1e-10]
                entropy = float(-np.sum(p_safe * np.log(p_safe)))    # policy entropy (nats)
                reg_unif  = regret / (1.0 - 1.0 / nc)                # A: 1.0 == uniform policy
                ent_norm  = entropy / ln_n                           # B: H/ln(N), 0..1
                perp_norm = (np.exp(entropy) - 1.0) / (nc - 1.0)     # C: (perplexity-1)/(N-1), 0..1
                ll_norm   = -np.log(max(chosen_prob, 1e-12)) / ln_n  # E: -lnP(chosen)/ln(N), 1.0==uniform
            else:
                reg_unif = ent_norm = perp_norm = ll_norm = 0.0
            best_alt_idx = -1
            if nc > 1:
                alt_probs = probs.copy()
                alt_probs[action] = -1.0
                best_alt_idx = int(np.argmax(alt_probs))
            entries.append({
                "game_idx": g_idx, "step": step,
                "regret": regret, "margin": margin,
                "reg_unif": reg_unif, "ent_norm": ent_norm,
                "perp_norm": perp_norm, "ll_norm": ll_norm,
                "chosen_prob": chosen_prob, "chosen_action": action,
                "best_alt_idx": best_alt_idx, "best_alt_prob": second_best,
                "num_choices": nc, "obs": obs, "feat": feat,
                "result": game["result"],
            })

    entries.sort(key=lambda e: -e["reg_unif"])

    if not entries:
        print("No decisions with action probabilities found.")
        return entries

    if not verbose:
        return entries

    n_decisions = len(entries)
    mean_regret = np.mean([e["regret"] for e in entries])
    mean_margin = np.mean([e["margin"] for e in entries])
    mean_a = np.mean([e["reg_unif"] for e in entries])
    mean_b = np.mean([e["ent_norm"] for e in entries])
    mean_c = np.mean([e["perp_norm"] for e in entries])
    mean_e = np.mean([e["ll_norm"] for e in entries])

    print(f"\nPolicy regret analysis — {len(games)} games, {n_decisions} model decisions")
    print(f"  Metrics per decision (menu size N):")
    print(f"    Raw = 1-P(chosen) [menu-size biased]   A = regret/(1-1/N) [1.0==uniform policy]")
    print(f"    B = H(policy)/ln(N) [0..1]              C = (perplexity-1)/(N-1) [0..1]")
    print(f"    D = P(chosen)-P(2nd) [margin, high=decisive]   E = -lnP(chosen)/ln(N) [1.0==uniform]")
    print(f"  Ranked by A (uniform-normalized regret).")
    print(f"\n  Overall means: Raw={mean_regret:.3f}  A={mean_a:.3f}  B={mean_b:.3f}  "
          f"C={mean_c:.3f}  D(margin)={mean_margin:.3f}  E={mean_e:.3f}")

    # By game outcome
    win_regrets = [e["regret"] for e in entries if e["result"] > 0]
    loss_regrets = [e["regret"] for e in entries if e["result"] < 0]
    if win_regrets and loss_regrets:
        print(f"\n  By outcome:")
        print(f"    Wins:   mean regret={np.mean(win_regrets):.3f}  "
              f"mean margin={np.mean([e['margin'] for e in entries if e['result'] > 0]):.3f}")
        print(f"    Losses: mean regret={np.mean(loss_regrets):.3f}  "
              f"mean margin={np.mean([e['margin'] for e in entries if e['result'] < 0]):.3f}")

    # By game phase
    phase_regrets = {}
    for e in entries:
        step_name = _step_name_from_feat(e["feat"])
        phase_regrets.setdefault(step_name, []).append(e["regret"])

    print(f"\n  By game phase:")
    print(f"    {'Phase':<14} {'Raw':>8} {'A':>8} {'D(marg)':>9} {'N':>6}")
    phase_margins = {}
    phase_a = {}
    for e in entries:
        step_name = _step_name_from_feat(e["feat"])
        phase_margins.setdefault(step_name, []).append(e["margin"])
        phase_a.setdefault(step_name, []).append(e["reg_unif"])
    for phase in _INTERP_STEP_NAMES:
        if phase not in phase_regrets:
            continue
        regs = phase_regrets[phase]
        mars = phase_margins[phase]
        aa = phase_a[phase]
        print(f"    {phase:<14} {np.mean(regs):8.3f} {np.mean(aa):8.3f} "
              f"{np.mean(mars):9.3f} {len(regs):6d}")

    # By board state
    bucket_regrets = {}
    for e in entries:
        life, board, timing = _board_bucket_from_feat(e["feat"])
        key = f"{life}/{board}"
        bucket_regrets.setdefault(key, []).append(e["regret"])

    print(f"\n  By board state (life/creatures):")
    for key in sorted(bucket_regrets):
        regs = bucket_regrets[key]
        if len(regs) < 5:
            continue
        print(f"    {key:<16} regret={np.mean(regs):.3f}  (n={len(regs)})")

    # Top decisions, ranked by A (uniform-normalized regret)
    top = entries[:min(top_n, len(entries))]
    print(f"\n  Top {len(top)} decisions by A (uniform-normalized regret):")
    print(f"    (Raw=1-P(chosen)  A=regret/(1-1/N)  B=H/lnN  C=(perp-1)/(N-1)  "
          f"D=margin  E=-lnP(chosen)/lnN  P(ch)=P(chosen))")
    print(f"    {'Game':<5} {'Step':<5} {'Phase':<13} {'#A':>4} "
          f"{'Raw':>6} {'A':>6} {'B':>6} {'C':>6} {'D':>6} {'E':>6} "
          f"{'P(ch)':>6} {'Res':<3}")
    print(f"    {'-'*5} {'-'*5} {'-'*13} {'-'*4} "
          f"{'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*3}")
    for e in top:
        phase = _step_name_from_feat(e["feat"])
        result_str = "W" if e["result"] > 0 else ("L" if e["result"] < 0 else "D")
        print(f"    {e['game_idx']:<5} {e['step']:<5} {phase:<13} {e['num_choices']:>4} "
              f"{e['regret']:6.3f} {e['reg_unif']:6.3f} {e['ent_norm']:6.3f} "
              f"{e['perp_norm']:6.3f} {e['margin']:6.3f} {e['ll_norm']:6.3f} "
              f"{e['chosen_prob']:6.3f} {result_str:<3}")

    # Detailed board states for top 5
    print(f"\n  Detailed board states for top {min(5, len(top))} regret decisions:\n")
    for rank, e in enumerate(top[:5]):
        phase = _step_name_from_feat(e["feat"])
        result_str = "WIN" if e["result"] > 0 else ("LOSS" if e["result"] < 0 else "DRAW")
        print(f"  --- #{rank + 1}: Game {e['game_idx']} step {e['step']} "
              f"({result_str}, regret={e['regret']:.3f}) ---")
        action_lines = _decode_legal_actions(e["obs"], e["num_choices"], e["chosen_action"])
        # Annotate with probabilities
        probs = games[e["game_idx"]]["action_probs"][e["step"]]
        for i, line in enumerate(action_lines):
            prob = probs[i] if i < len(probs) else 0.0
            print(f"  {line}  P={prob:.3f}")
        print()
        _decode_board_state(e["obs"], value=games[e["game_idx"]]["values"][e["step"]])
        print()

    return entries


def _analyze_entropy(games, verbose=True):
    """Compute policy entropy at each decision point.

    H = -sum(p * ln(p)) for legal actions. Normalized entropy H_norm = H / ln(N)
    ranges from 0 (certain) to 1 (uniform).

    Returns list of (entropy, normalized_entropy, feat, step_name, result, game_idx, step) tuples.
    """
    records = []
    for g_idx, game in enumerate(games):
        probs_list = game.get("action_probs", [])
        if not probs_list:
            continue
        for step, (probs, nc, feat) in enumerate(zip(
                probs_list, game["num_choices"], game["interp_features"])):
            # Entropy: -sum(p * ln(p)), skip zero-probability actions
            p = probs[:nc]
            p_safe = p[p > 1e-10]
            entropy = -np.sum(p_safe * np.log(p_safe))
            max_entropy = np.log(nc) if nc > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            records.append({
                "entropy": entropy, "norm_entropy": norm_entropy,
                "feat": feat, "num_choices": nc,
                "result": game["result"], "game_idx": g_idx, "step": step,
            })

    if not records:
        print("No decisions with action probabilities found.")
        return records

    if not verbose:
        return records

    all_h = np.array([r["entropy"] for r in records])
    all_hn = np.array([r["norm_entropy"] for r in records])

    print(f"\nPolicy entropy analysis — {len(games)} games, {len(records)} decisions")
    print(f"  Raw entropy:  mean={all_h.mean():.3f}  std={all_h.std():.3f}  "
          f"min={all_h.min():.3f}  max={all_h.max():.3f}")
    print(f"  Norm entropy: mean={all_hn.mean():.3f}  std={all_hn.std():.3f}  "
          f"(0=certain, 1=uniform)")

    # By game outcome
    win_h = [r["norm_entropy"] for r in records if r["result"] > 0]
    loss_h = [r["norm_entropy"] for r in records if r["result"] < 0]
    if win_h and loss_h:
        print(f"\n  By outcome:")
        print(f"    Wins:   norm_H={np.mean(win_h):.3f} +/- {np.std(win_h):.3f}")
        print(f"    Losses: norm_H={np.mean(loss_h):.3f} +/- {np.std(loss_h):.3f}")
        if np.mean(loss_h) > np.mean(win_h) + 0.05:
            print(f"    ** Model is more uncertain in losing games — "
                  f"suggests confusion rather than deliberate unpredictability.")

    # By game phase
    print(f"\n  By game phase:")
    print(f"    {'Phase':<14} {'Mean H':>8} {'Norm H':>8} {'Std':>8} {'N':>6}")
    phase_data = {}
    for r in records:
        step_name = _step_name_from_feat(r["feat"])
        phase_data.setdefault(step_name, []).append(r)
    for phase in _INTERP_STEP_NAMES:
        if phase not in phase_data:
            continue
        recs = phase_data[phase]
        h_vals = [r["entropy"] for r in recs]
        hn_vals = [r["norm_entropy"] for r in recs]
        print(f"    {phase:<14} {np.mean(h_vals):8.3f} {np.mean(hn_vals):8.3f} "
              f"{np.std(hn_vals):8.3f} {len(recs):6d}")

    # By board state bucket
    print(f"\n  By board state (life / creatures / timing):")
    print(f"    {'Bucket':<28} {'Norm H':>8} {'Std':>8} {'N':>6}")
    bucket_data = {}
    for r in records:
        life, board, timing = _board_bucket_from_feat(r["feat"])
        key = f"{life:<7} {board:<7} {timing}"
        bucket_data.setdefault(key, []).append(r["norm_entropy"])
    for key in sorted(bucket_data):
        vals = bucket_data[key]
        if len(vals) < 5:
            continue
        flag = " **" if np.mean(vals) > all_hn.mean() + all_hn.std() else ""
        print(f"    {key:<28} {np.mean(vals):8.3f} {np.std(vals):8.3f} {len(vals):6d}{flag}")

    # Low entropy check (potential overfit)
    low_entropy_frac = np.mean(all_hn < 0.1)
    if low_entropy_frac > 0.8:
        print(f"\n  WARNING: {low_entropy_frac:.0%} of decisions have norm_H < 0.1 — "
              f"potential overfit to a narrow strategy.")

    # Phase × outcome breakdown for phases with interesting patterns
    print(f"\n  Phase × Outcome breakdown:")
    print(f"    {'Phase':<14} {'Win H':>8} {'Loss H':>8} {'Delta':>8}")
    for phase in _INTERP_STEP_NAMES:
        if phase not in phase_data:
            continue
        recs = phase_data[phase]
        w_h = [r["norm_entropy"] for r in recs if r["result"] > 0]
        l_h = [r["norm_entropy"] for r in recs if r["result"] < 0]
        if len(w_h) < 3 or len(l_h) < 3:
            continue
        delta = np.mean(l_h) - np.mean(w_h)
        marker = " *" if abs(delta) > 0.05 else ""
        print(f"    {phase:<14} {np.mean(w_h):8.3f} {np.mean(l_h):8.3f} {delta:+8.3f}{marker}")

    return records


def _analyze_consistency(games, top_n=20, verbose=True):
    """Find similar observations where the model chose different actions.

    Uses cosine similarity on _INTERP_FEATURE_NAMES vectors. Reports
    inconsistency rates for simple game states.

    Returns list of inconsistent pairs sorted by descending similarity.
    """
    # Collect all (interp_feat, action_category, game_idx, step) tuples
    points = []
    for g_idx, game in enumerate(games):
        for step, (feat, action, nc, obs) in enumerate(zip(
                game["interp_features"], game["actions"],
                game["num_choices"], game["observations"])):
            # Decode chosen action category
            cat_raw = obs[STATE_SIZE + action]
            cat = int(round(cat_raw * ACTION_CATEGORY_MAX))
            points.append({
                "feat": feat, "action": action, "cat": cat,
                "game_idx": g_idx, "step": step,
                "num_choices": nc, "obs": obs,
                "result": game["result"],
            })

    if len(points) < 10:
        print("Not enough decision points for consistency analysis.")
        return []

    # Build feature matrix and normalize for cosine similarity
    feat_mat = np.array([p["feat"] for p in points], dtype=np.float64)
    norms = np.linalg.norm(feat_mat, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    feat_norm = feat_mat / norms

    # For efficiency, sample if too many points
    max_compare = 5000
    if len(points) > max_compare:
        idx = np.random.choice(len(points), size=max_compare, replace=False)
        idx.sort()
        points_sub = [points[i] for i in idx]
        feat_sub = feat_norm[idx]
    else:
        points_sub = points
        feat_sub = feat_norm

    # Compute pairwise cosine similarity (dot product of normalized vectors)
    sim_matrix = feat_sub @ feat_sub.T

    # Find pairs with high similarity but different action categories
    pairs = []
    n = len(points_sub)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] < 0.95:
                continue
            if points_sub[i]["cat"] == points_sub[j]["cat"]:
                continue
            pairs.append({
                "similarity": sim_matrix[i, j],
                "i": points_sub[i], "j": points_sub[j],
            })

    pairs.sort(key=lambda p: -p["similarity"])

    if not verbose:
        return pairs

    n_high_sim = np.sum(sim_matrix > 0.95) // 2  # upper triangle
    n_same_action = 0
    n_diff_action = 0
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] < 0.95:
                continue
            if points_sub[i]["cat"] == points_sub[j]["cat"]:
                n_same_action += 1
            else:
                n_diff_action += 1

    print(f"\nDecision consistency analysis — {len(games)} games, {len(points)} decisions")
    if len(points) > max_compare:
        print(f"  (sampled {max_compare} decisions for pairwise comparison)")
    print(f"\n  Pairs with cosine similarity > 0.95: {n_high_sim}")
    if n_high_sim > 0:
        consistency_rate = n_same_action / (n_same_action + n_diff_action) * 100
        print(f"    Same action category: {n_same_action} ({consistency_rate:.1f}%)")
        print(f"    Different action:     {n_diff_action} ({100 - consistency_rate:.1f}%)")
    else:
        print("    No highly similar state pairs found.")
        return pairs

    # Simple state inconsistency: few creatures, plenty of mana
    simple_mask = []
    for p in points_sub:
        f = p["feat"]
        total_creatures = f[18] + f[21]  # self + opp creatures
        total_mana = f[9]  # self total mana
        simple_mask.append(total_creatures <= 2 and total_mana >= 3)

    simple_idx = [i for i, m in enumerate(simple_mask) if m]
    if len(simple_idx) >= 10:
        simple_feat = feat_sub[simple_idx]
        simple_sim = simple_feat @ simple_feat.T
        simple_same = simple_diff = 0
        for i in range(len(simple_idx)):
            for j in range(i + 1, len(simple_idx)):
                if simple_sim[i, j] < 0.95:
                    continue
                if points_sub[simple_idx[i]]["cat"] == points_sub[simple_idx[j]]["cat"]:
                    simple_same += 1
                else:
                    simple_diff += 1
        simple_total = simple_same + simple_diff
        if simple_total > 0:
            simple_inconsistent = simple_diff / simple_total * 100
            flag = " ** RED FLAG" if simple_inconsistent > 20 else ""
            print(f"\n  Simple states (<=2 creatures, >=3 mana): "
                  f"{len(simple_idx)} decisions")
            print(f"    High-similarity pairs: {simple_total}")
            print(f"    Inconsistency rate: {simple_inconsistent:.1f}%{flag}")

    # Top inconsistent pairs
    top = pairs[:min(top_n, len(pairs))]
    if top:
        print(f"\n  Top {len(top)} most inconsistent pairs (high sim, different action):")
        print(f"    {'#':<4} {'Sim':>6} {'Game A':>7} {'Step A':>7} "
              f"{'Game B':>7} {'Step B':>7} {'Action A':<14} {'Action B':<14}")
        print(f"    {'-'*4} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*14} {'-'*14}")
        for rank, pair in enumerate(top):
            pi, pj = pair["i"], pair["j"]
            cat_a = _CAT_NAMES.get(pi["cat"], str(pi["cat"]))
            cat_b = _CAT_NAMES.get(pj["cat"], str(pj["cat"]))
            print(f"    {rank + 1:<4} {pair['similarity']:6.3f} "
                  f"{pi['game_idx']:>7} {pi['step']:>7} "
                  f"{pj['game_idx']:>7} {pj['step']:>7} "
                  f"{cat_a:<14} {cat_b:<14}")

        # Detailed comparison for top 3
        print(f"\n  Detailed comparison for top {min(3, len(top))} pairs:\n")
        for rank, pair in enumerate(top[:3]):
            pi, pj = pair["i"], pair["j"]
            print(f"  --- Pair #{rank + 1} (similarity={pair['similarity']:.4f}) ---")
            for label, p in [("A", pi), ("B", pj)]:
                cat_name = _CAT_NAMES.get(p["cat"], str(p["cat"]))
                f = p["feat"]
                result_str = "W" if p["result"] > 0 else ("L" if p["result"] < 0 else "D")
                print(f"    [{label}] Game {p['game_idx']} step {p['step']} ({result_str})  "
                      f"Action: {cat_name}")
                print(f"        Life {f[0]:.0f}/{f[1]:.0f}  "
                      f"Creatures {f[18]:.0f}v{f[21]:.0f}  "
                      f"Lands {f[19]:.0f}v{f[22]:.0f}  "
                      f"Hand {f[17]:.0f}  Mana {f[9]:.0f}  "
                      f"Phase: {_step_name_from_feat(f)}")
            # Show what features differ most
            diff = np.abs(pi["feat"] - pj["feat"])
            top_diff_idx = np.argsort(-diff)[:5]
            diffs_str = ", ".join(
                f"{_INTERP_FEATURE_NAMES[k]}({pi['feat'][k]:.1f}→{pj['feat'][k]:.1f})"
                for k in top_diff_idx if diff[k] > 0.01)
            if diffs_str:
                print(f"    Largest feature diffs: {diffs_str}")
            print()

    return pairs


def _analyze_calibration(games, verbose=True):
    """Check whether V(s) at game start predicts actual win rate.

    Bins games by their initial value estimate and compares against the
    actual win rate within each bin.  A well-calibrated value function has
    win-rate track V(s); systematic deviation indicates bias.

    Returns list of dicts: {lo, hi, mean_v, win_rate, n, wins, losses, draws}.
    """
    points = []
    for g in games:
        if not g["values"]:
            continue
        points.append((g["values"][0], g["result"]))

    if not points:
        if verbose:
            print("  No games with value data.")
        return []

    vs = np.array([p[0] for p in points])
    results = np.array([p[1] for p in points])

    edges = [-np.inf, -0.5, -0.25, 0.0, 0.25, 0.5, np.inf]
    labels = ["< -0.50", "-0.50…-0.25", "-0.25…0.00",
              "0.00…0.25", "0.25…0.50", "> 0.50"]
    bins = []
    for i in range(len(edges) - 1):
        mask = (vs > edges[i]) & (vs <= edges[i + 1])
        n = int(mask.sum())
        if n == 0:
            continue
        w = int((results[mask] > 0).sum())
        l = int((results[mask] < 0).sum())
        d = int((results[mask] == 0).sum())
        mean_v = float(vs[mask].mean())
        wr = w / n
        bins.append({
            "lo": edges[i], "hi": edges[i + 1],
            "label": labels[i], "mean_v": mean_v,
            "win_rate": wr, "n": n, "wins": w, "losses": l, "draws": d,
        })

    if not verbose:
        return bins

    print(f"\nValue function calibration — {len(points)} games")
    print(f"  Initial V(s) ranges vs actual win rate:\n")
    print(f"    {'Bin':<16} {'Mean V':>8} {'Win Rate':>10} {'N':>5}  {'W/L/D'}")
    print(f"    {'-'*16} {'-'*8} {'-'*10} {'-'*5}  {'-'*10}")
    for b in bins:
        print(f"    {b['label']:<16} {b['mean_v']:>+8.3f} {b['win_rate']:>9.1%} "
              f"{b['n']:>5}  {b['wins']}/{b['losses']}/{b['draws']}")

    # Bias summary
    # Expected: V(s) ≈ (win_rate - 0.5) * 2 roughly, since V in [-1, 1]
    # Compare mean_v to (win_rate * 2 - 1)
    if len(bins) >= 2:
        total_bias = 0.0
        total_n = 0
        for b in bins:
            implied_v = b["win_rate"] * 2.0 - 1.0
            total_bias += (b["mean_v"] - implied_v) * b["n"]
            total_n += b["n"]
        avg_bias = total_bias / total_n if total_n > 0 else 0.0
        if avg_bias < -0.1:
            print(f"\n    Model is systematically PESSIMISTIC (avg bias {avg_bias:+.3f})")
            print(f"    May be playing too defensively.")
        elif avg_bias > 0.1:
            print(f"\n    Model is systematically OPTIMISTIC (avg bias {avg_bias:+.3f})")
            print(f"    May be overcommitting / underestimating risk.")
        else:
            print(f"\n    Model calibration looks reasonable (avg bias {avg_bias:+.3f}).")

    return bins


def _analyze_turning_points(games, verbose=True):
    """Find the 'point of no return' in each game.

    For each game, find the last decision step where V(s) permanently crossed
    from negative to positive (for wins) or positive to negative (for losses).
    This is the turning point — more meaningful than the max swing because
    swings can recover.

    Returns list of dicts with turning point info, one per game that has one.
    """
    turning_points = []
    for g_idx, game in enumerate(games):
        vals = game["values"]
        result = game["result"]
        if len(vals) < 3 or result == 0:
            continue

        # For wins, find last crossing from negative to positive that held
        # For losses, find last crossing from positive to negative that held
        won = result > 0
        target_sign = 1 if won else -1

        # Find the last step where value crossed to the target sign and stayed
        crossing_step = None
        for i in range(len(vals) - 1):
            before_sign = 1 if vals[i] >= 0 else -1
            after_sign = 1 if vals[i + 1] >= 0 else -1
            if before_sign != target_sign and after_sign == target_sign:
                # Check if it stays on the target side for the rest of the game
                stayed = all(
                    (v >= 0) == (target_sign == 1)
                    for v in vals[i + 1:]
                )
                if stayed:
                    crossing_step = i

        if crossing_step is None:
            continue

        feat = game["interp_features"][crossing_step] if crossing_step < len(game["interp_features"]) else None
        turning_points.append({
            "game_idx": g_idx,
            "step": crossing_step,
            "total_steps": len(vals),
            "frac": crossing_step / len(vals),
            "v_before": vals[crossing_step],
            "v_after": vals[crossing_step + 1],
            "result": result,
            "model_is_a": game["model_is_a"],
            "feat": feat,
        })

    if not verbose:
        return turning_points

    if not turning_points:
        print("\n  No turning points found (games may start and stay on one side).")
        return turning_points

    win_tps = [t for t in turning_points if t["result"] > 0]
    loss_tps = [t for t in turning_points if t["result"] < 0]
    all_fracs = [t["frac"] for t in turning_points]

    print(f"\nTurning point analysis — {len(games)} games, "
          f"{len(turning_points)} with identifiable turning points")
    print(f"  (A turning point is the last permanent zero-crossing of V(s))\n")

    print(f"  Games with turning point: {len(turning_points)}/{len(games)} "
          f"({100 * len(turning_points) / len(games):.0f}%)")
    print(f"    Wins:   {len(win_tps)}")
    print(f"    Losses: {len(loss_tps)}")

    print(f"\n  Timing (fraction of game elapsed at turning point):")
    print(f"    Overall: mean={np.mean(all_fracs):.2f}  "
          f"median={np.median(all_fracs):.2f}  "
          f"std={np.std(all_fracs):.2f}")
    if win_tps:
        wf = [t["frac"] for t in win_tps]
        print(f"    Wins:    mean={np.mean(wf):.2f}  median={np.median(wf):.2f}")
    if loss_tps:
        lf = [t["frac"] for t in loss_tps]
        print(f"    Losses:  mean={np.mean(lf):.2f}  median={np.median(lf):.2f}")

    # Board state at turning points
    if any(t["feat"] is not None for t in turning_points):
        print(f"\n  Board state at turning points:")
        phase_counts = {}
        board_counts = {}
        for t in turning_points:
            if t["feat"] is None:
                continue
            phase = _step_name_from_feat(t["feat"])
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
            life, board, timing = _board_bucket_from_feat(t["feat"])
            key = f"{timing}/{life}/{board}"
            board_counts[key] = board_counts.get(key, 0) + 1

        print(f"    By phase:")
        for phase in _INTERP_STEP_NAMES:
            if phase in phase_counts:
                pct = 100 * phase_counts[phase] / len(turning_points)
                print(f"      {phase:<14} {phase_counts[phase]:>4} ({pct:5.1f}%)")

        print(f"    By board state (timing/life/creatures):")
        for key in sorted(board_counts, key=lambda k: -board_counts[k]):
            pct = 100 * board_counts[key] / len(turning_points)
            print(f"      {key:<24} {board_counts[key]:>4} ({pct:5.1f}%)")

    # Show a few example turning points
    print(f"\n  Example turning points (first {min(8, len(turning_points))}):")
    print(f"    {'Game':<6} {'Step':<6} {'Of':<6} {'Frac':>6} "
          f"{'V before':>9} {'V after':>9} {'Result':<6}")
    print(f"    {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*6}")
    for t in turning_points[:8]:
        r = "WIN" if t["result"] > 0 else "LOSS"
        print(f"    {t['game_idx']:<6} {t['step']:<6} {t['total_steps']:<6} "
              f"{t['frac']:>5.0%} {t['v_before']:>+9.3f} {t['v_after']:>+9.3f} {r:<6}")

    return turning_points


def _analyze_clusters(games, verbose=True):
    """Cluster games by V(s) curve shape using simple shape descriptors.

    Classifies games into archetypes:
      - "early_lead_held":  V(s) starts positive and stays mostly positive
      - "slow_grind":       V(s) starts near zero, gradually moves toward outcome
      - "comeback":         V(s) spends significant time negative then finishes positive
      - "lead_blown":       V(s) spends significant time positive then finishes negative
      - "volatile":         V(s) crosses zero 3+ times

    Returns dict: {archetype_name: [game_indices]}.
    """
    archetypes = {
        "early_lead_held": [],
        "slow_grind": [],
        "comeback": [],
        "lead_blown": [],
        "volatile": [],
    }
    game_labels = []  # (game_idx, label) for all classified games

    for g_idx, game in enumerate(games):
        vals = game["values"]
        result = game["result"]
        if len(vals) < 3:
            game_labels.append((g_idx, "too_short"))
            continue

        arr = np.array(vals)
        won = result > 0
        lost = result < 0

        # Count zero crossings
        signs = np.sign(arr)
        signs[signs == 0] = 1  # treat zero as positive
        crossings = int(np.sum(np.abs(np.diff(signs)) > 0))

        # Fraction of game spent positive/negative
        frac_positive = np.mean(arr > 0)
        frac_negative = np.mean(arr < 0)

        # Early game tendency (first quarter)
        q1 = max(1, len(arr) // 4)
        early_mean = arr[:q1].mean()

        if crossings >= 3:
            label = "volatile"
        elif won and early_mean < -0.05 and frac_negative > 0.3:
            label = "comeback"
        elif lost and early_mean > 0.05 and frac_positive > 0.3:
            label = "lead_blown"
        elif won and early_mean > 0.05 and frac_positive > 0.6:
            label = "early_lead_held"
        elif lost and early_mean < -0.05 and frac_negative > 0.6:
            label = "early_lead_held"  # opponent held lead from start
        elif abs(early_mean) <= 0.15:
            label = "slow_grind"
        else:
            label = "slow_grind"  # fallback

        archetypes[label].append(g_idx)
        game_labels.append((g_idx, label))

    if not verbose:
        return archetypes

    print(f"\nValue trajectory clustering — {len(games)} games\n")
    print(f"  {'Archetype':<20} {'Count':>6} {'Wins':>6} {'Losses':>6} "
          f"{'Win%':>6} {'Avg Length':>10}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*10}")

    for label in ["early_lead_held", "slow_grind", "comeback", "lead_blown", "volatile"]:
        indices = archetypes[label]
        if not indices:
            continue
        n = len(indices)
        w = sum(1 for i in indices if games[i]["result"] > 0)
        l = sum(1 for i in indices if games[i]["result"] < 0)
        avg_len = np.mean([len(games[i]["values"]) for i in indices])
        wr = w / n * 100 if n > 0 else 0
        print(f"  {label:<20} {n:>6} {w:>6} {l:>6} {wr:>5.1f}% {avg_len:>10.1f}")

    # Per-archetype description and notable features
    descs = {
        "early_lead_held": "Model had an early advantage and maintained it",
        "slow_grind": "Close game that gradually resolved toward the outcome",
        "comeback": "Model was behind but recovered to win",
        "lead_blown": "Model had an early lead but lost it",
        "volatile": "Highly uncertain game with 3+ momentum shifts",
    }
    print()
    for label, desc in descs.items():
        indices = archetypes[label]
        if not indices:
            continue
        print(f"  {label}: {desc}")

        # Value stats
        all_vals = [np.array(games[i]["values"]) for i in indices]
        mean_start = np.mean([v[0] for v in all_vals])
        mean_end = np.mean([v[-1] for v in all_vals])
        mean_crossings = np.mean([
            int(np.sum(np.abs(np.diff(np.sign(np.where(v == 0, 1, v)))) > 0))
            for v in all_vals
        ])
        print(f"    Avg start V: {mean_start:+.3f}  Avg end V: {mean_end:+.3f}  "
              f"Avg zero-crossings: {mean_crossings:.1f}")

        # Board state at midpoint
        mid_feats = []
        for i in indices:
            g = games[i]
            mid = len(g["interp_features"]) // 2
            if mid < len(g["interp_features"]):
                mid_feats.append(g["interp_features"][mid])
        if mid_feats:
            mf = np.mean(mid_feats, axis=0)
            print(f"    Avg midgame: Life {mf[0]:.0f}/{mf[1]:.0f}  "
                  f"Creatures {mf[18]:.1f}v{mf[21]:.1f}  "
                  f"Lands {mf[19]:.1f}v{mf[22]:.1f}  "
                  f"Hand {mf[17]:.1f}")
        print()

    return archetypes


# ── Card importance (value attribution per card) ─────────────────────────────

# Action categories used for per-card attribution.
_CAT_ACTIVATE, _CAT_CAST, _CAT_LAND = CAT_ACTIVATE_ABILITY, CAT_CAST_SPELL, CAT_PLAY_LAND


def _analyze_cardvalue(games, top_n=30, verbose=True):
    """Rank cards by how much the model's play values them in this matchup.

    For every spell cast / ability activated, three independent signals are
    aggregated per card:

      * mean ΔV — change in the value estimate across the move
        (vals[i+1]-vals[i]); how much deploying the card moves the model's own
        win-probability estimate. The most direct "importance" signal.
      * priority — mean policy probability the masked policy puts on casting the
        card whenever it is a legal choice (how much the model "wants" it).
      * win-rate lift — win rate in games where the model deployed the card minus
        games where it did not (ΔWR). Confounded by draw luck; shown for colour.

    Needs model traces (values + action_probs). Returns per-card dict rows sorted
    by mean ΔV; cards below a small sample gate sink to the bottom (ordered by
    usage). The matchup is fixed by the caller's deck-a / deck-b, so the ranking
    is matchup-specific by construction.
    """
    dv = {}          # card -> list of ΔV across casts/activations
    prio_mass = {}   # card -> [sum prob when legal, times legal]
    present = {}     # card -> set of game indices where the model deployed it
    cast_ct = {}     # card -> number of cast/activate actions
    n_games = len(games)

    for gi, g in enumerate(games):
        obs_list = g["observations"]
        actions = g["actions"]
        vals = g.get("values", [])
        probs_list = g.get("action_probs", [])
        ncs = g["num_choices"]
        for si in range(len(obs_list)):
            obs = obs_list[si]
            nc = ncs[si]
            # Priority: probability mass on each castable card this decision.
            if si < len(probs_list) and probs_list[si] is not None:
                probs = probs_list[si]
                for k in range(nc):
                    if _obs_action_cat(obs, k) == _CAT_CAST:
                        cid = _obs_card_id(obs, k)
                        if 0 <= cid < len(_VOCAB_NAMES):
                            name = decode.card_index_to_name(cid)
                            slot = prio_mass.setdefault(name, [0.0, 0])
                            slot[0] += float(probs[k]) if k < len(probs) else 0.0
                            slot[1] += 1
            # Chosen-action attribution.
            action = actions[si]
            cat = _obs_action_cat(obs, action)
            if cat in (_CAT_CAST, _CAT_ACTIVATE):
                cid = _obs_card_id(obs, action)
                if 0 <= cid < len(_VOCAB_NAMES):
                    name = decode.card_index_to_name(cid)
                    cast_ct[name] = cast_ct.get(name, 0) + 1
                    present.setdefault(name, set()).add(gi)
                    if si + 1 < len(vals):
                        dv.setdefault(name, []).append(vals[si + 1] - vals[si])
            elif cat == _CAT_LAND:
                cid = _obs_card_id(obs, action)
                if 0 <= cid < len(_VOCAB_NAMES):
                    # Lands count toward presence (win-rate lift) but carry no
                    # meaningful ΔV, so they are not added to dv/cast_ct.
                    present.setdefault(decode.card_index_to_name(cid), set()).add(gi)

    rows = []
    for name in set(cast_ct) | set(present) | set(prio_mass):
        deltas = dv.get(name, [])
        pm = prio_mass.get(name, [0.0, 0])
        pres = present.get(name, set())
        n_present = len(pres)
        n_absent = n_games - n_present
        wr_present = (sum(1 for i in pres if games[i]["result"] > 0) / n_present
                      if n_present else None)
        wr_absent = (sum(1 for i in range(n_games)
                         if i not in pres and games[i]["result"] > 0) / n_absent
                     if n_absent else None)
        rows.append({
            "card": name,
            "n_cast": cast_ct.get(name, 0),
            "mean_dv": (float(np.mean(deltas)) if deltas else None),
            "n_dv": len(deltas),
            "priority": (pm[0] / pm[1] if pm[1] else None),
            "n_present": n_present,
            "wr_present": wr_present,
            "wr_absent": wr_absent,
            "wr_lift": (wr_present - wr_absent
                        if (wr_present is not None and wr_absent is not None) else None),
        })

    # Graded cards (enough ΔV samples) sort by mean ΔV desc; the rest by usage.
    _MIN_DV = 3

    def _key(r):
        graded = r["mean_dv"] is not None and r["n_dv"] >= _MIN_DV
        return (0 if graded else 1,
                -(r["mean_dv"] if graded else -1e9),
                -r["n_cast"])
    rows.sort(key=_key)

    if not verbose:
        return rows

    n_win = sum(1 for g in games if g["result"] > 0)
    base_wr = n_win / n_games if n_games else 0.0
    print(f"\nCard importance — {n_games} games, base win rate {base_wr:.0%}")
    print(f"  ΔV = mean change in V(s) when the card is cast/activated "
          f"(>0 raises the model's win-prob estimate).")
    print(f"  prio = mean policy probability on casting it when legal.  "
          f"ΔWR = win-rate with minus without.\n")

    graded = [r for r in rows if r["mean_dv"] is not None and r["n_dv"] >= _MIN_DV]
    maxabs = max((abs(r["mean_dv"]) for r in graded), default=0.0)

    print(f"  {'Card':<26} {'n':>4} {'ΔV':>7} {'prio':>6} {'ΔWR':>6}  value")
    print(f"  {'-'*26} {'-'*4} {'-'*7} {'-'*6} {'-'*6}  {'-'*37}")
    for r in rows[:top_n]:
        dvv = r["mean_dv"]
        dv_str = f"{dvv:+7.3f}" if dvv is not None else "    —  "
        prio = r["priority"]
        prio_str = f"{prio*100:5.0f}%" if prio is not None else "    —"
        lift = r["wr_lift"]
        lift_str = f"{lift*100:+5.0f}%" if lift is not None else "    —"
        bar = viz.diverging_bar(dvv, maxabs) if (dvv is not None and maxabs > 0) else ""
        print(f"  {r['card'][:26]:<26} {r['n_cast']:>4} {dv_str} "
              f"{prio_str} {lift_str}  {bar}")
    return rows


def _chart_cardvalue(rows, args=None, top_n=20):
    """Diverging horizontal bar chart of mean ΔV per card (helps vs hurts)."""
    graded = [r for r in rows if r["mean_dv"] is not None and r["n_dv"] >= 3][:top_n]
    if not graded:
        print("  No cards with enough cast samples to chart.")
        return None
    plt = viz.pyplot(show=viz.want_show(args))
    if plt is None:
        print("  matplotlib unavailable; skipping chart.")
        return None
    graded = list(reversed(graded))  # barh plots bottom-up
    names = [r["card"][:28] for r in graded]
    vals = [r["mean_dv"] for r in graded]
    colors = ["seagreen" if v >= 0 else "firebrick" for v in vals]
    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.35)))
    ax.barh(range(len(names)), vals, color=colors)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean ΔV(s) when cast / activated")
    ax.set_title("Card importance (value contribution)")
    ax.grid(True, axis="x", alpha=0.3)
    return viz.save_or_show(plt, fig, "cardvalue", args)


def _chart_sbvalue(sbrows, args=None, top_n=20):
    """Three-panel sideboard chart: per-direction swap preference — mean policy
    prob on each swap when offered, one panel for boarded-IN and one for
    boarded-OUT so a bar's direction is never ambiguous — plus net-impact ΔWR
    per card class (fetchlands fungible): the post-board GAME win-rate delta
    vs that card's net-zero games. Mirrors _chart_cardvalue's conventions
    (green/red poles, gray zero line, horizontal bars)."""
    pref_in = [(r["card"], r["conf"] or 0.0) for r in sbrows["in"]][:top_n]
    pref_out = [(r["card"], r["conf"] or 0.0) for r in sbrows["out"]][:top_n]
    # Only chart classes with a non-zero win-rate delta — an all-zero row is
    # visually indistinguishable from missing data and just pads the panel.
    net = [(r["card"], r["dwr_in"], r["dwr_out"]) for r in sbrows["net"]
           if any(d is not None and abs(d) > 1e-9 for d in (r["dwr_in"], r["dwr_out"]))]
    net = sorted(net, key=lambda t: -max(abs(d) for d in t[1:] if d is not None))[:top_n]
    panels = ([("Boarded IN — swap preference", "seagreen", pref_in)] if pref_in else []) + \
             ([("Boarded OUT — swap preference", "firebrick", pref_out)] if pref_out else [])
    if not panels and not net:
        print("  No sideboard swaps to chart.")
        return None
    plt = viz.pyplot(show=viz.want_show(args))
    if plt is None:
        print("  matplotlib unavailable; skipping chart.")
        return None

    n_rows = len(panels) + (1 if net else 0)
    heights = [max(2.0, len(p[2]) * 0.35) for p in panels]
    if net:
        heights.append(max(2.5, len(net) * 0.35))
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(9, sum(heights) + 0.9 * n_rows),
                             squeeze=False,
                             gridspec_kw={"height_ratios": heights}
                             if n_rows > 1 else None)
    row = 0
    for title, color, pref in panels:
        ax = axes[row, 0]; row += 1
        pref = list(reversed(pref))  # barh plots bottom-up
        names = [c[:28] for c, _ in pref]
        vals = [v for _, v in pref]
        ax.barh(range(len(names)), vals, color=color)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Mean policy prob on the swap when offered")
        ax.set_title(title)
        ax.grid(True, axis="x", alpha=0.3)
    if net:
        ax = axes[row, 0]
        net = list(reversed(net))
        names = [c[:28] for c, _, _ in net]
        ys = np.arange(len(names))
        h = 0.38
        ax.barh(ys + h / 2, [d if d is not None else 0.0 for _, d, _ in net],
                height=h, color="seagreen", label="ΔWR when net-IN")
        ax.barh(ys - h / 2, [d if d is not None else 0.0 for _, _, d in net],
                height=h, color="firebrick", label="ΔWR when net-OUT")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Post-board game win-rate delta vs net-zero games")
        ax.set_title("Net sideboard impact (fetchlands fungible)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return viz.save_or_show(plt, fig, "sbvalue", args)


def _chart_value_overview(games, args=None):
    """Faint V(s) curve per game plus the mean curve, on a shared 0–1 x-axis."""
    curves = [g["values"] for g in games if len(g.get("values", [])) >= 2]
    if not curves:
        return None
    plt = viz.pyplot(show=viz.want_show(args))
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(10, 4))
    max_len = max(len(v) for v in curves)
    resampled = []
    for g in games:
        v = g.get("values", [])
        if len(v) < 2:
            continue
        xs = np.linspace(0, 1, len(v))
        color = "seagreen" if g["result"] > 0 else ("firebrick" if g["result"] < 0 else "gray")
        ax.plot(xs, v, color=color, alpha=0.18, linewidth=0.8)
        resampled.append(np.interp(np.linspace(0, 1, max_len), xs, v))
    if resampled:
        ax.plot(np.linspace(0, 1, max_len), np.mean(resampled, axis=0),
                color="navy", linewidth=2.0, label="mean")
        ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Fraction of game elapsed")
    ax.set_ylabel("V(s)")
    ax.set_title("Value trajectories (green=win, red=loss)")
    ax.grid(True, alpha=0.3)
    return viz.save_or_show(plt, fig, "value_overview", args)




def _capture(fn, *a, **k):
    """Run a verbose analyzer, returning its printed text instead of stdout."""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


def cmd_report(args):
    """Run the standard battery once and emit a single self-contained HTML report."""
    import html as _html

    model, env, opp_model = _load_model_and_env(args)
    if getattr(model, "is_az", False):
        print("[report] AZ checkpoint: V(s) is the AZNet tanh outcome estimate in "
              "[-1, 1] (a bounded game-result prediction, not the PPO shaped-return "
              "critic). All battery analyses apply; only absolute value magnitudes "
              "differ in scale from PPO reports.")
    print(f"\nCollecting {args.n_games} game traces...")
    games = _collect_game_traces(model, env, opp_model, args.n_games)
    env.close()

    out = viz.out_dir(args)
    deck_a = getattr(args, "deck_a", None) or "?"
    deck_b = getattr(args, "deck_b", None) or args.opponent

    # Text sections (captured from the verbose analyzers). Bo3-only analyzers
    # (sideboard, boundaries, match calibration) print a one-line "no data"
    # note on bo1 traces, so they are safe to include unconditionally.
    is_bo3 = any(_match_score(g) is not None for g in games)
    sections = [
        ("Summary", _capture(_sim_summary, games)),
        ("Card importance", _capture(_analyze_cardvalue, games, args.top if hasattr(args, "top") else 30)),
        ("Targeting / hold-vs-cast", _capture(_sim_targeting, games)),
        ("Value calibration", _capture(_analyze_calibration, games)),
        ("Turning points", _capture(_analyze_turning_points, games)),
        ("Trajectory archetypes", _capture(_analyze_clusters, games)),
        ("Top value swings", _capture(lambda g: _print_swing_table(_compute_swings(g)[:10]), games)),
        ("Policy regret", _capture(_analyze_regret, games, 20)),
        ("Policy entropy", _capture(_analyze_entropy, games)),
        ("Decision consistency", _capture(_analyze_consistency, games, 10)),
    ]
    if is_bo3:
        sections[2:2] = [
            ("Sideboard decisions", _capture(_sim_sideboard_report, games)),
            ("Sideboard preference & net impact", _capture(_analyze_sbvalue, games)),
            ("Match boundaries (V(s) across games)",
             _capture(lambda g: _print_boundaries(_compute_boundaries(g)), games)),
            ("Match-score calibration", _capture(_print_match_calibration, games)),
        ]

    # Charts (saved as PNGs alongside the report; referenced by basename).
    rows = _analyze_cardvalue(games, verbose=False)
    chart_paths = [_chart_cardvalue(rows, args=args),
                   _chart_value_overview(games, args=args)]
    if is_bo3:
        chart_paths.append(_chart_sbvalue(_analyze_sbvalue(games, verbose=False),
                                          args=args))
    imgs = [os.path.basename(p) for p in chart_paths if p]

    parts = ["<!doctype html><meta charset='utf-8'>",
             "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1000px}"
             "pre{background:#f5f5f5;padding:1rem;overflow-x:auto;font-size:12px;line-height:1.3}"
             "img{max-width:100%;border:1px solid #ddd;margin:0.5rem 0}h1{font-size:1.4rem}"
             "h2{font-size:1.1rem;border-bottom:1px solid #ccc;padding-bottom:0.2rem}</style>",
             f"<h1>RoboMage analysis — {_html.escape(deck_a)} vs {_html.escape(deck_b)}</h1>",
             f"<p>{len(games)} simulated games · model "
             f"<code>{_html.escape(os.path.basename(args.model))}</code></p>"]
    for name in imgs:
        parts.append(f"<img src='{_html.escape(name)}' alt='{_html.escape(name)}'>")
    for title, text in sections:
        parts.append(f"<h2>{_html.escape(title)}</h2><pre>{_html.escape(text)}</pre>")

    report_path = os.path.join(out, "report.html")
    with open(report_path, "w") as f:
        f.write("\n".join(parts))
    print(f"\n[report] wrote {report_path}")








def cmd_interactive(args):
    """Load model, simulate games, then enter the interactive session."""
    model, env, opp_model = _load_model_and_env(args)

    games = []
    if args.n_games > 0:
        print(f"\nSimulating {args.n_games} games...")
        games = _collect_game_traces(model, env, opp_model, args.n_games)

    ctx = {
        "games": games,
        "swing_data": None,
        "shap_values": None,
        "shap_samples": None,
        "calibration_data": None,
        "turning_data": None,
        "cluster_data": None,
        "whatif_data": None,
        "model": model,
        "env": env,
        "opp_model": opp_model,
        "args": args,
    }

    try:
        _interactive_session(ctx)
    finally:
        env.close()


# ── Search vs raw comparison (AZ / PPO evaluator + MCTS) ──────────────────────
#
# Per searched (loop-safe) root, compare what the NET alone says (softmax priors,
# leaf value) against what SEARCH concludes (MCTS visit distribution, root value):
#   * how far the visit distribution moved off the prior (mean KL(priors||visits)),
#   * how often search's top move disagrees with the net's greedy move,
#   * how well the net's leaf value tracks the search's root value (MAE + corr).
# Search is where an AZ/PPO checkpoint's play differs from its raw policy, so this
# is the natural search-aware analysis view.


def _build_search_evaluator(spec):
    """(evaluator, None) for the search-compare tool.

    The second element is always ``None`` — a checkpoint no longer encodes a deck
    (one generalist), so the deck is supplied explicitly via --deck-a/--deck-b.
    An AZ spec -> AZEvaluator (falling back to a PPO warm-start); a PPO spec ->
    PPOEvaluator; ``uniform`` / ``mcts:uniform`` -> the torch-free UniformEvaluator."""
    from mcts import PPOEvaluator, UniformEvaluator
    base = _az_spec_base(spec)
    if base.lower() in ("uniform", "mcts:uniform"):
        return UniformEvaluator(), None
    if _is_az_model_spec(spec):
        from az_net import AZEvaluator, load_az, from_ppo, resolve_az_checkpoint
        az = resolve_az_checkpoint(base)
        if az is not None:
            return AZEvaluator(load_az(az)), None
        ppo = _resolve_model_path(base)
        print(f"No AZ checkpoint for {base!r}; warm-starting an AZNet from PPO {ppo}")
        return AZEvaluator(from_ppo(ppo)), None
    from opponents import _load_model
    path = _resolve_model_path(base)
    return PPOEvaluator(_load_model(path)), None


def _make_search_compare_controller(evaluator, *, sims, worlds, c_puct, rng_seed):
    """A SearchController that also RECORDS (priors, visit_dist, net_value,
    root_value, obs) for every searched root, for the search-vs-raw report."""
    from opponents import SearchController

    class _SearchCompareController(SearchController):
        def __init__(self):
            super().__init__(evaluator, sims=sims, worlds=worlds, c_puct=c_puct,
                             temperature=0.0, label="search-compare", rng_seed=rng_seed)
            self.records = []

        def choose(self, obs, num_choices, action_masks=None, decoded_actions=None):
            from mcts import run_search
            env = self._env
            searchable = (env is not None
                          and getattr(env, "last_search_safe", None)
                          and num_choices > 1)
            priors, net_value = self._evaluator.evaluate(obs, num_choices)
            if not searchable:
                self.stats["fallback"] += 1
                return int(np.argmax(priors))
            result = run_search(env, self._evaluator, sims=self._sims,
                                worlds=self._worlds, c_puct=self._c_puct, rng=self._rng)
            self.stats["searched"] += 1
            self.stats["sims"] += result.sims_run
            self.stats["sim_steps"] += result.sim_steps
            visits = result.visits.astype(np.float64)
            tot = visits.sum()
            visit_dist = (visits / tot if tot > 0
                          else np.full(num_choices, 1.0 / num_choices))
            self.records.append({
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "num_choices": int(num_choices),
                "priors": np.asarray(priors, dtype=np.float64).copy(),
                "visit_dist": visit_dist,
                "net_value": float(net_value),
                "root_value": float(result.root_value),
            })
            return result.best_action()

    return _SearchCompareController()


def _kl(p, q):
    """KL(p || q) over a menu, smoothing q off zero so an unvisited action
    doesn't blow up (p is a softmax prior, strictly positive)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.maximum(np.asarray(q, dtype=np.float64), 1e-12)
    q = q / q.sum()
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / q[nz])))


def _report_search_compare(ctrl, args):
    """Print the search-vs-raw summary from a recording controller's records."""
    recs = ctrl.records
    st = ctrl.stats
    total = st["searched"] + st["fallback"]
    print("\n" + "=" * 68)
    print("Search vs raw-net comparison")
    print("=" * 68)
    print(f"  Decisions: {st['searched']} searched, {st['fallback']} fallback "
          f"(safe fraction {st['searched'] / max(1, total):.1%}); "
          f"{st['sims']} sims, {st['sim_steps']} sim steps.")
    if not recs:
        print("  No searched roots recorded (all decisions fell back to the raw "
              "policy — try a deck/opponent with more loop-safe priority windows).")
        return

    kls = np.array([_kl(r["priors"], r["visit_dist"]) for r in recs])
    agree = np.array([int(np.argmax(r["priors"]) == np.argmax(r["visit_dist"]))
                      for r in recs])
    net_v = np.array([r["net_value"] for r in recs])
    root_v = np.array([r["root_value"] for r in recs])
    vmae = float(np.mean(np.abs(net_v - root_v)))
    if len(recs) > 1 and net_v.std() > 1e-9 and root_v.std() > 1e-9:
        vcorr = float(np.corrcoef(net_v, root_v)[0, 1])
        vcorr_s = f"{vcorr:+.3f}"
    else:
        vcorr_s = "n/a"

    print(f"  Roots analyzed: {len(recs)}")
    print(f"  mean KL(priors || visits): {kls.mean():.4f}  "
          f"(median {np.median(kls):.4f}, max {kls.max():.4f})")
    print(f"  argmax agreement (net greedy == search pick): {agree.mean():.1%}")
    print(f"  value net-vs-search:  MAE {vmae:.4f}   corr {vcorr_s}")

    top_n = max(0, int(getattr(args, "top", 8)))
    if top_n:
        order = np.argsort(-kls)[:top_n]
        print(f"\n  Top {len(order)} biggest prior-vs-visit disagreements:")
        for rank, i in enumerate(order):
            r = recs[i]
            obs = r["obs"]
            feat = _extract_interpretable(obs)
            step = _step_name_from_feat(feat)
            turn_no = 1 + int(round(feat[_FEAT["turn"]]))
            pa = int(np.argmax(r["priors"]))
            va = int(np.argmax(r["visit_dist"]))
            print(f"   [{rank}] T{turn_no} {step:<12} "
                  f"Life {feat[_FEAT['self_life']]:.0f}/{feat[_FEAT['opp_life']]:.0f}"
                  f"  KL={kls[i]:.3f}  Vnet={r['net_value']:+.3f} "
                  f"Vsearch={r['root_value']:+.3f}")
            print(f"        net greedy : {_action_desc(obs, pa)}  "
                  f"(P={r['priors'][pa]:.2f}, visits={r['visit_dist'][pa]:.2f})")
            if va != pa:
                print(f"        search pick: {_action_desc(obs, va)}  "
                      f"(P={r['priors'][va]:.2f}, visits={r['visit_dist'][va]:.2f})")
            else:
                print(f"        search pick: (same action, visit mass shifted)")


class _MergedSearchStats:
    """Duck-types the bits of ``_SearchCompareController`` that
    ``_report_search_compare`` reads (``records``/``stats``), so results
    gathered from parallel worker batches can be reported the same way as a
    single in-process controller."""

    def __init__(self):
        self.records = []
        self.stats = {"searched": 0, "fallback": 0, "sims": 0, "sim_steps": 0}

    def absorb(self, records, stats):
        self.records.extend(records)
        for k in self.stats:
            self.stats[k] += stats.get(k, 0)


def _run_search_compare_batch(payload):
    """Worker entry point (one process per batch): rebuild the evaluator/
    controller from scratch — a loaded model isn't picklable across the
    process boundary — and drive this batch's games. Returns
    ``(batch_id, n_games, records, stats, elapsed)``."""
    (batch_id, model_spec, opponent_spec, deck_a, deck_b, n_games, seed,
     sims, worlds, c_puct, binary_path, bo3) = payload
    t0 = time.time()
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    import runner
    from opponents import make_controller

    evaluator, _ = _build_search_evaluator(model_spec)
    ctrl_model = _make_search_compare_controller(
        evaluator, sims=sims, worlds=worlds, c_puct=c_puct, rng_seed=seed)
    ctrl_opp = make_controller(opponent_spec)
    runner.run_games(ctrl_model, ctrl_opp, label_a="Search", label_b="Opp",
                     binary_path=binary_path, deck_a=deck_a, deck_b=deck_b,
                     n_games=n_games, bo3=bo3, seed=seed, transcript="quiet")
    return batch_id, n_games, ctrl_model.records, dict(ctrl_model.stats), time.time() - t0


def _split_batches(n_games, n_workers, seed):
    """Contiguous, seed-disjoint batches: batch i's local seed+j lines up
    with the sequential run's seed+(global index), so results are the same
    set of (deck, seed) games regardless of worker count."""
    n_workers = max(1, min(n_workers, n_games))
    base, extra = divmod(n_games, n_workers)
    batches = []
    start = 0
    for i in range(n_workers):
        count = base + (1 if i < extra else 0)
        if count:
            batches.append((start, count))
        start += count
    return batches


def cmd_search_compare(args):
    """Drive N games with an MCTS controller and report, per searched decision,
    net priors vs MCTS visits and net value vs search root value."""
    deck_a = getattr(args, "deck_a", None)
    if not deck_a:
        print("Model deck is required — a checkpoint no longer encodes a deck; "
              "pass --deck-a", file=sys.stderr)
        sys.exit(1)
    deck_b = getattr(args, "deck_b", None)
    if not deck_b:
        # A model opponent no longer encodes a deck either; mirror by default.
        deck_b = deck_a
    args.deck_a, args.deck_b = deck_a, deck_b

    n_workers = max(1, getattr(args, "workers", 1) or 1)
    bo3 = _effective_bo3(args)

    if n_workers <= 1:
        import runner
        from opponents import make_controller

        evaluator, _ = _build_search_evaluator(args.model)
        ctrl_model = _make_search_compare_controller(
            evaluator, sims=args.sims, worlds=args.worlds, c_puct=args.c,
            rng_seed=args.seed)
        ctrl_opp = make_controller(args.opponent)

        print(f"Search-compare: {deck_a} (search {args.sims}x{args.worlds}, c={args.c}) "
              f"vs {args.opponent} [{deck_b}] over {args.n_games} game(s)...")

        t0 = time.time()
        done = 0

        def _progress(record):
            nonlocal done
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = (args.n_games - done) / rate if rate > 0 else float("inf")
            st = ctrl_model.stats
            print(f"  game {done}/{args.n_games}  "
                  f"(searched {st['searched']}, fallback {st['fallback']})  "
                  f"elapsed {elapsed:.1f}s  eta {eta:.1f}s", flush=True)

        runner.run_games(ctrl_model, ctrl_opp, label_a="Search", label_b="Opp",
                         binary_path=args.binary, deck_a=deck_a, deck_b=deck_b,
                         n_games=args.n_games, bo3=bo3,
                         seed=args.seed, transcript="quiet", on_game_end=_progress)
        _report_search_compare(ctrl_model, args)
        return

    # Parallel: split n_games across worker processes, each rebuilding its own
    # evaluator/controller (a loaded model can't cross the process boundary),
    # then merge every batch's records/stats before reporting.
    batches = _split_batches(args.n_games, n_workers, args.seed)
    payloads = [
        (i, args.model, args.opponent, deck_a, deck_b, count, args.seed + start,
         args.sims, args.worlds, args.c, args.binary, bo3)
        for i, (start, count) in enumerate(batches)
    ]
    print(f"Search-compare (parallel): {deck_a} (search {args.sims}x{args.worlds}, "
          f"c={args.c}) vs {args.opponent} [{deck_b}] over {args.n_games} game(s) "
          f"across {len(payloads)} worker(s)...")

    merged = _MergedSearchStats()
    t0 = time.time()
    done_games = 0
    with ProcessPoolExecutor(max_workers=len(payloads)) as ex:
        futs = {ex.submit(_run_search_compare_batch, p): p[0] for p in payloads}
        for fut in as_completed(futs):
            batch_id, n, records, stats, dt = fut.result()
            merged.absorb(records, stats)
            done_games += n
            elapsed = time.time() - t0
            rate = done_games / elapsed if elapsed > 0 else 0.0
            eta = (args.n_games - done_games) / rate if rate > 0 else float("inf")
            print(f"  [batch {batch_id}] {n} game(s) in {dt:.1f}s -> "
                  f"{done_games}/{args.n_games} done "
                  f"({100 * done_games / args.n_games:.0f}%)  "
                  f"elapsed {elapsed:.1f}s  eta {eta:.1f}s", flush=True)

    _report_search_compare(merged, args)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze a trained RoboMage model by simulating games")
    sub = parser.add_subparsers(dest="command", required=True)

    # All subcommands and their flags come from cli_spec.ANALYSIS_TOOL (single
    # source shared with the TUI). Dispatch below stays hand-written.
    for s in ANALYSIS_TOOL.subs:
        sp = sub.add_parser(s.name, help=s.help)
        apply_to_parser(sp, s)

    args = parser.parse_args()
    {
        "report": cmd_report,
        "interactive": cmd_interactive,
        "search": cmd_search_compare,
    }[args.command](args)


if __name__ == "__main__":
    main()
