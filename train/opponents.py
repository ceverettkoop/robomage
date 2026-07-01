"""Opponent controllers and the factory/pool that builds them.

This unifies the three historical opponent code paths (ModelVsScriptedEnv's
scripted opponent, SelfPlayEnv's scripted fallback, and observe's per-side
controller) behind a single ``Controller`` interface, so scripted agents and PPO
checkpoints are interchangeable and can be mixed per episode.

sb3/torch are imported lazily (only when a model controller is actually built) so
importing this module — and the env/scripted_agent chain — never pulls torch when
only scripted opponents are used.
"""

import glob as _glob
import math
import os
import re
from typing import Callable, Optional, Protocol, Sequence, Union

import numpy as np

from env import MAX_ACTIONS
from scripted_agent import ScriptedAgent, make_agent

# Bare suffixes (and the "scripted" prefix) that denote a scripted controller.
# Keep in sync with scripted_agent._PRESETS.
_SCRIPTED_SUFFIXES = frozenset({"scripted", "random", "greedy", "easy", "hard",
                                "heuristic", "explore", "fuzz"})

# Checkpoint format version. v2 is the deck-pilot naming ('{deck}__v{steps}.zip' /
# '{deck}__final.zip') used since the embed_dim bump; the double underscore is the
# format marker that separates these from the legacy '{a}_{b}_*.zip' matchup files.
# Bumping embed_dim already invalidates old nets, so this is a clean break with no
# back-compat shim — files that don't parse as v2 are simply skipped (see
# ``deck_snapshots``), so they never get loaded with a mismatched network.
CHECKPOINT_FORMAT_VERSION = 2

_SNAPSHOT_RE = re.compile(r"^(?P<deck>.+)__v(?P<steps>\d+)\.zip$")


def deck_snapshots(deck: Optional[str], checkpoint_dir: Optional[str]) -> list[str]:
    """All frozen snapshots that pilot ``deck`` (v2 naming).

    Matches ``{deck}__v{steps}.zip`` plus ``{deck}__final.zip``. Returns absolute
    paths sorted by training step (the ``__final`` snapshot, if present, last).
    Files that don't parse as v2 deck-pilot snapshots are skipped, so legacy
    matchup checkpoints never leak into a league pool. Empty when the deck/dir is
    unknown or nothing matches.
    """
    if not (deck and checkpoint_dir):
        return []
    # A deck name may be a path relative to decks/ (e.g. league-folder decks are
    # 'league/<stem>'); the glob is already scoped to that subdir, so compare the
    # parsed snapshot's deck against the name's basename rather than the full path.
    deck_stem = os.path.basename(deck)
    versioned: list[tuple[int, str]] = []
    for path in _glob.glob(os.path.join(checkpoint_dir, f"{deck}__v*.zip")):
        m = _SNAPSHOT_RE.match(os.path.basename(path))
        if m and m.group("deck") == deck_stem:
            versioned.append((int(m.group("steps")), path))
    versioned.sort()
    out = [p for _, p in versioned]
    final = os.path.join(checkpoint_dir, f"{deck}__final.zip")
    if os.path.exists(final):
        out.append(final)
    return out


def latest_snapshot(deck: Optional[str], checkpoint_dir: Optional[str]) -> Optional[str]:
    """Newest snapshot piloting ``deck`` (highest version, or ``__final``); None if none."""
    snaps = deck_snapshots(deck, checkpoint_dir)
    return snaps[-1] if snaps else None

# Pool token standing for a random checkpoint compatible with the current
# matchup — a generalist trained to pilot the opponent's deck. OpponentPool
# expands it into the opponent deck's pilots ('{opp_deck}__v*.zip' /
# '{opp_deck}__final.zip'), the same deck-pilot snapshots SelfPlayEnv samples.
MATCHUP_MODEL_TOKEN = "random-model"


def is_scripted_spec(spec: str) -> bool:
    """True if ``spec`` names a scripted agent rather than a checkpoint path."""
    s = (spec or "").strip().lower()
    return s.startswith("scripted") or s in _SCRIPTED_SUFFIXES


def matchup_checkpoints(model_deck: Optional[str], opp_deck: Optional[str],
                        checkpoint_dir: Optional[str]) -> list[str]:
    """Generalist checkpoints that pilot the opponent's deck.

    Models are per-deck generalists, so any pilot of ``opp_deck`` is a valid
    opponent — the deck-pilot snapshots ``{opp_deck}__v*.zip`` /
    ``{opp_deck}__final.zip`` (same set SelfPlayEnv._reload_opponent samples).
    ``model_deck`` is unused (kept for signature compatibility). Returns absolute
    paths sorted by training step; empty when the deck/dir is unknown or nothing
    matches.
    """
    return deck_snapshots(opp_deck, checkpoint_dir)


class Controller(Protocol):
    """Uniform decision interface for any opponent.

    ``decoded_actions`` is the decision's decoded action menu
    (``decode.decode_actions_from_obs`` output); the runner supplies it only
    when a controller advertises ``wants_decoded = True`` (e.g. PlayController),
    so index/model/scripted controllers ignore it.
    """

    def choose(self, obs: np.ndarray, num_choices: int,
               action_masks: Optional[np.ndarray] = None,
               decoded_actions: Optional[list] = None) -> int: ...


class ScriptedController:
    """Wraps a ScriptedAgent."""

    def __init__(self, agent: ScriptedAgent, label: str = "scripted"):
        self._agent = agent
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        return self._agent.act(obs, num_choices)


class ModelController:
    """Wraps a loaded MaskablePPO checkpoint."""

    def __init__(self, model, label: str = "model", deterministic: bool = False):
        self._model = model
        self.label = label
        self._deterministic = deterministic
        self._mask = np.zeros(MAX_ACTIONS, dtype=bool)

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        if action_masks is None:
            self._mask[:] = False
            self._mask[:num_choices] = True
            action_masks = self._mask
        action, _ = self._model.predict(obs, action_masks=action_masks,
                                        deterministic=self._deterministic)
        return int(action)


class ActionListController:
    """Plays a fixed sequence of action indices (test harness ``--actions``).

    Consumes one index per decision regardless of which side has priority (the
    sequence is global, matching the harness convention); once exhausted it
    falls back to action 0 (pass / first choice).
    """

    def __init__(self, actions: Sequence[int], label: str = "Actions"):
        self._actions = [int(a) for a in actions]
        self._i = 0
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        if self._i < len(self._actions):
            a = self._actions[self._i]
            self._i += 1
            return a
        return 0


class PlayController:
    """Plays a fixed sequence of semantic action specs (``--play``).

    Each spec (``cast:Lightning Bolt``, ``target:Grizzly Bears@opp``, ``pass``,
    ``#7`` …) is resolved against *this* decision's decoded menu via
    :mod:`action_spec`, so the sequence is robust to dynamic index reordering.
    The sequence is global (one spec consumed per decision); once exhausted it
    falls back to action ``0`` (pass / first choice — always legal) so the game
    keeps advancing to its conclusion or the decision cap.

    **Seat keys.** When the SAME controller drives both seats (the test harness's
    dual-seat ``--play`` mode), sequencing the priority hand-offs between the two
    players by hand is error-prone. So a spec may be pinned to a seat with a
    leading ``A:`` / ``B:`` key (see :mod:`action_spec`). Before applying the next
    spec this controller checks the seat that currently holds priority
    (``obs[32]`` — true = Player A): if the next spec is keyed to the *other*
    seat, the current priority holder passes (the spec is **not** consumed) and
    play advances until the keyed seat is on the clock. Unkeyed specs are applied
    to whoever has priority (the legacy behaviour), so existing scripts are
    unaffected.

    A keyed spec also **passes the keyed seat forward through its own priority
    windows** until the action becomes legal: e.g. ``A:attack:Voice`` given while A
    still holds priority in its main phase auto-passes (leaving the spec unconsumed)
    until A reaches the declare-attackers step where ``attack:`` is offered, instead
    of erroring because no attacker menu exists yet. This fires only on a genuine "not
    offered at this decision" (``no_match``) and only while a pass is itself legal (a
    priority window); an ambiguous/typo'd spec, or one that reaches a mandatory choice
    with no pass, still raises :class:`action_spec.PlayResolveError` with the menu,
    bounded by ``_MAX_WAIT`` so a never-legal spec can't silently pass the game away.

    ``wants_decoded = True`` tells the runner to hand ``choose`` the decoded menu.
    A spec that resolves to zero or multiple legal actions raises
    :class:`action_spec.PlayResolveError` — a loud failure with the legal menu,
    instead of silently playing the wrong index. ``self.resolved`` accumulates
    the integer indices chosen, so a run can be replayed as a plain ``--actions``
    list.
    """

    wants_decoded = True

    # Cap on consecutive auto-passes spent waiting for a seat-keyed spec to become
    # legal, so a spec that is never offered (a typo, or an action the seat can never
    # take) fails loudly instead of silently passing the rest of the game away.
    _MAX_WAIT = 200

    def __init__(self, specs, label: str = "Play"):
        import action_spec
        self._action_spec = action_spec
        self._specs = action_spec.parse_spec_list(specs)
        self._i = 0
        self._wait = 0
        self.label = label
        self.resolved: list[int] = []

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        if decoded_actions is None:
            raise RuntimeError("PlayController requires decoded_actions; drive it "
                               "through runner.run_games (which supplies the menu).")
        # Specs exhausted: auto-advance with action 0 (always legal), like
        # AutoPassController, so the game runs to its end / the decision cap.
        if self._i >= len(self._specs):
            self.resolved.append(0)
            return 0
        spec = self._specs[self._i]
        # Seat-keyed spec for the seat that is NOT on the clock: the current
        # priority holder passes (spec left for later) so we advance to the keyed
        # seat's decision instead of mis-applying its action to this player.
        seat = self._action_spec.spec_seat(spec)
        if seat is not None and seat != ("A" if obs[32] > 0.5 else "B"):
            idx = self._action_spec.resolve_to_index("pass", decoded_actions)
            self.resolved.append(idx)
            return idx

        r = self._action_spec.resolve(spec, decoded_actions)
        if r.ok:
            self._i += 1
            self._wait = 0
            self.resolved.append(r.index)
            return r.index

        # The keyed seat IS on the clock but its action isn't legal at THIS decision
        # (e.g. `A:attack:Voice` while A still holds priority in its main phase — the
        # attacker menu only appears at the declare-attackers step). Pass priority to
        # advance the keyed seat toward the step where the action becomes legal, leaving
        # the spec unconsumed, instead of failing. Bounded by _MAX_WAIT, and only on a
        # genuine "not offered here" (`no_match`): an ambiguous/typo'd spec, or one whose
        # decision offers no pass (a mandatory choice), still fails loudly with the menu.
        if (seat is not None and r.kind == "no_match" and self._wait < self._MAX_WAIT):
            pass_r = self._action_spec.resolve("pass", decoded_actions)
            if pass_r.ok:
                self._wait += 1
                self.resolved.append(pass_r.index)
                return pass_r.index

        raise self._action_spec.PlayResolveError(r, decoded_actions)


class InteractiveController:
    """Prompts stdin for an action index each decision (test harness ``--interactive``)."""

    def __init__(self, label: str = "Human"):
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        while True:
            try:
                raw = input("  >> Enter action index: ").strip()
                c = int(raw)
                if 0 <= c < num_choices:
                    return c
                print(f"     Invalid: must be 0-{num_choices - 1}")
            except (ValueError, EOFError):
                print("     Enter a valid integer")


class AutoPassController:
    """Always picks action 0 (pass priority / first choice) — the harness default."""

    def __init__(self, label: str = "Auto"):
        self.label = label

    def choose(self, obs, num_choices, action_masks=None, decoded_actions=None) -> int:
        return 0


def _load_model(path: str):
    """Load a checkpoint with MaskablePPO (falling back to PPO). Lazy sb3 import."""
    try:
        from sb3_contrib import MaskablePPO as _PPO
    except ImportError:
        from stable_baselines3 import PPO as _PPO
    return _PPO.load(path, device="cpu")


def make_controller(spec: str, *,
                    checkpoint_resolver: Optional[Callable[[str], str]] = None,
                    deterministic: bool = False) -> Controller:
    """Resolve a spec string into a Controller.

    Scripted specs ("scripted", "scripted:hard", "random", ...) build a
    ScriptedController; anything else is treated as a checkpoint path/shorthand,
    resolved via ``checkpoint_resolver`` (identity if not given) and loaded as a
    ModelController.
    """
    if is_scripted_spec(spec):
        return ScriptedController(make_agent(spec), label=spec)
    path = checkpoint_resolver(spec) if checkpoint_resolver else spec
    return ModelController(_load_model(path), label=spec, deterministic=deterministic)


def parse_pool_spec(spec: Union[str, Sequence]) -> list[tuple[str, float]]:
    """Parse a pool specification into a list of (spec, weight) pairs.

    Accepts a comma-separated string ("scripted:easy,scripted:hard,ckpt=2") or an
    iterable of specs / (spec, weight) tuples.  Each comma item may carry an
    optional "=<weight>" suffix (default weight 1.0).
    """
    items: list = []
    if isinstance(spec, str):
        items = [p.strip() for p in spec.split(",") if p.strip()]
    else:
        items = list(spec)

    out: list[tuple[str, float]] = []
    for item in items:
        if isinstance(item, (tuple, list)):
            s, w = item[0], float(item[1])
        elif "=" in item:
            s, _, w_str = item.rpartition("=")
            s, w = s.strip(), float(w_str)
        else:
            s, w = item, 1.0
        out.append((s, w))
    return out


class OpponentPool:
    """Per-episode weighted sampler over a mix of scripted agents and checkpoints.

    Memory bound: under SubprocVecEnv each env is its own process, so caches are
    not shared and total resident models ≈ (checkpoints loaded per process) ×
    n_envs.  We cap the number of *active* checkpoints to
    ``max(1, floor(max_checkpoint_ratio * n_envs))`` and shard them across
    processes by ``env_index``, so each process only ever loads ≈ ceil(ratio)
    distinct checkpoints (≤ 1 per process at ratio ≤ 1.0).  Scripted specs are
    stateless and free, so every process keeps the full scripted set.
    """

    _matchup_warned = False

    def __init__(self, spec: Union[str, Sequence], *,
                 checkpoint_resolver: Optional[Callable[[str], str]] = None,
                 rng: Optional[np.random.Generator] = None,
                 n_envs: int = 1, env_index: int = 0,
                 max_checkpoint_ratio: float = 1.0,
                 deterministic: bool = False,
                 model_deck: Optional[str] = None,
                 opp_deck: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None):
        self._resolver = checkpoint_resolver
        self._deterministic = deterministic
        self._rng = rng if rng is not None else np.random.default_rng()
        self._cache: dict[str, Controller] = {}

        entries = parse_pool_spec(spec) or [("scripted", 1.0)]
        entries = self._expand_matchup_tokens(entries, model_deck, opp_deck, checkpoint_dir)
        scripted = [(s, w) for s, w in entries if is_scripted_spec(s)]
        ckpts = [(s, w) for s, w in entries if not is_scripted_spec(s)]

        # Cap and shard the checkpoint set to bound per-process memory.
        max_unique = max(1, int(math.floor(max_checkpoint_ratio * max(1, n_envs))))
        active = ckpts[:max_unique]
        n_envs = max(1, n_envs)
        my_ckpts = active[env_index % n_envs::n_envs] if active else []

        self._entries = scripted + my_ckpts
        if not self._entries:
            self._entries = [("scripted", 1.0)]
        weights = np.array([w for _, w in self._entries], dtype=float)
        self._weights = weights / weights.sum()
        self._specs = [s for s, _ in self._entries]

    @classmethod
    def _expand_matchup_tokens(cls, entries, model_deck, opp_deck, checkpoint_dir):
        """Replace each MATCHUP_MODEL_TOKEN entry with the matchup's compatible
        checkpoints (absolute paths), splitting its weight evenly among them.

        A token that matches no checkpoint is dropped (with a one-time warning);
        all other entries pass through unchanged."""
        out: list[tuple[str, float]] = []
        for spec, w in entries:
            if isinstance(spec, str) and spec.strip().lower() == MATCHUP_MODEL_TOKEN:
                files = matchup_checkpoints(model_deck, opp_deck, checkpoint_dir)
                if not files:
                    if not cls._matchup_warned:
                        print(f"[opponent-pool] WARNING: '{MATCHUP_MODEL_TOKEN}' matched no "
                              f"generalist piloting {opp_deck} ({opp_deck}__*.zip) in "
                              f"{checkpoint_dir}; ignoring it.")
                        cls._matchup_warned = True
                    continue
                share = w / len(files)
                out.extend((f, share) for f in files)
            else:
                out.append((spec, w))
        return out

    def _controller_for(self, spec: str) -> Controller:
        ctrl = self._cache.get(spec)
        if ctrl is None:
            ctrl = make_controller(spec, checkpoint_resolver=self._resolver,
                                   deterministic=self._deterministic)
            self._cache[spec] = ctrl
        return ctrl

    def sample(self) -> tuple[str, Controller]:
        """Return (label, controller) for this episode (weighted random)."""
        idx = int(self._rng.choice(len(self._specs), p=self._weights))
        spec = self._specs[idx]
        return spec, self._controller_for(spec)

    @property
    def specs(self) -> list[str]:
        return list(self._specs)


class LeaguePool:
    """League opponent sampler for PFSP / softmax multi-matchup training.

    Per episode it makes the two coupled choices the league needs — which opponent
    *deck* and which *controller* pilots it — from a pool spanning:

      1. a scripted anchor piloting a sampled roster deck (collapse guard),
      2. frozen snapshots of every deck's model (cross-deck + mirror self-play),
      3. the latest snapshot of the learner's own deck (the OpenAI-Five 'play the
         latest self' slot, chosen with probability ``self_play_frac``).

    Memory is bounded exactly like :class:`OpponentPool`: historical snapshot
    checkpoints are capped at ``max(1, floor(max_checkpoint_ratio * n_envs))`` unique
    files and sharded across processes by ``env_index``; scripted agents are free so
    every process keeps the full anchor set, and the latest-self snapshot is always
    resident (one model).

    Quality weighting for the historical-snapshot branch is supplied externally by
    the PFSPCallback via :meth:`set_weights` (a global win-rate estimate broadcast to
    every process); each process applies it to whatever sharded subset it holds — an
    accepted approximation, documented in
    docs/plan_pfsp_multimatchup_training.md. The weighting *mode* (pfsp vs softmax)
    lives entirely in the callback; the pool only consumes the resulting weights and
    falls back to uniform until the first broadcast arrives.
    """

    REFRESH_EVERY = 50  # episodes between snapshot rescans (pick up new vN files)

    def __init__(self, learner_deck: str, roster: Sequence[str], checkpoint_dir: str, *,
                 self_play_frac: float = 0.8, scripted_anchor_frac: float = 0.1,
                 rng: Optional[np.random.Generator] = None,
                 n_envs: int = 1, env_index: int = 0,
                 max_checkpoint_ratio: float = 1.0,
                 deterministic: bool = False, scripted_spec: str = "scripted"):
        self._learner = learner_deck
        self._roster = list(roster) or [learner_deck]
        self._dir = checkpoint_dir
        self._self_play_frac = float(self_play_frac)
        self._anchor_frac = float(scripted_anchor_frac)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._n_envs = max(1, n_envs)
        self._env_index = env_index
        self._ratio = max_checkpoint_ratio
        self._deterministic = deterministic
        self._scripted_spec = scripted_spec
        self._cache: dict[str, Controller] = {}    # path -> ModelController
        self._weights: dict = {}                   # (opp_deck, label) -> float
        self._snap_entries: list[tuple[str, str]] = []  # [(path, deck)] this process holds
        self._latest_self: Optional[str] = None    # learner's newest snapshot
        self._total_snaps = 0                      # total snapshots across roster (auto-ramp)
        self._episode = 0
        self._scripted_controller: Optional[Controller] = None
        self.refresh()

    # ── snapshot discovery / sharding ────────────────────────────────────────
    def refresh(self):
        """Rescan the checkpoint dir for snapshots, re-shard, and re-find latest-self.

        The active pool is bounded to ``max_unique`` files (per-process memory) but
        composed for league *fairness* rather than by a raw recency slice over the
        roster-ordered snapshot list — that slice kept whichever decks happened to
        be last in the roster and silently evicted the ones listed first once the
        pool overflowed. Instead it is built in two tiers:

          * Tier 1 (guaranteed) — every roster deck's ``__final`` (or its newest
            snapshot when no ``__final`` exists yet). This keeps a low-win-rate
            deck present as an opponent no matter how many snapshots the stronger
            decks have accumulated; ``max_unique`` is raised if needed so no deck's
            anchor is ever dropped.
          * Tier 2 (discretionary) — the remaining ``__v*`` snapshots, filled
            newest-first round-robin across decks so one prolific deck can't crowd
            the others out. These intermediates are already quality-gated at *save*
            time by the promote-margin (SnapshotCallback), so re-enabling that gate
            thins this tier without ever dropping a deck's guaranteed ``__final``.
        """
        per_deck: list[tuple[str, list[str]]] = []   # (deck, [oldest..newest, __final last])
        for deck in self._roster:
            snaps = deck_snapshots(deck, self._dir)
            if snaps:
                per_deck.append((deck, snaps))
        self._total_snaps = sum(len(s) for _, s in per_deck)
        self._latest_self = latest_snapshot(self._learner, self._dir)

        # Tier 1: newest-per-deck anchor (== __final when present). Never evicted,
        # so grow the cap to fit every deck's anchor if the ratio would undercut it.
        guaranteed = [(snaps[-1], deck) for deck, snaps in per_deck]
        max_unique = max(len(guaranteed), max(1, int(math.floor(self._ratio * self._n_envs))))

        # Tier 2: remaining __v* intermediates, newest-first, round-robin by deck.
        queues = [list(reversed(snaps[:-1])) for _, snaps in per_deck]  # newest -> oldest
        decks = [deck for deck, _ in per_deck]
        discretionary: list[tuple[str, str]] = []
        remaining = max_unique - len(guaranteed)
        while remaining > 0 and any(queues):
            for qi, q in enumerate(queues):
                if not q:
                    continue
                discretionary.append((q.pop(0), decks[qi]))
                remaining -= 1
                if remaining == 0:
                    break

        # Shard the pooled set across processes (each env keeps ~max_unique/n_envs).
        active = guaranteed + discretionary
        self._snap_entries = active[self._env_index % self._n_envs::self._n_envs]

    def _maybe_refresh(self):
        self._episode += 1
        if self._episode % self.REFRESH_EVERY == 0:
            self.refresh()

    # ── weights pushed from the PFSPCallback ─────────────────────────────────
    def set_weights(self, weights):
        """Replace the historical-snapshot weighting (keyed by (opp_deck, label))."""
        if weights:
            self._weights = dict(weights)

    # ── controllers ──────────────────────────────────────────────────────────
    def _scripted(self) -> Controller:
        if self._scripted_controller is None:
            self._scripted_controller = ScriptedController(
                make_agent(self._scripted_spec), label=self._scripted_spec)
        return self._scripted_controller

    def _model_for(self, path: str) -> Controller:
        ctrl = self._cache.get(path)
        if ctrl is None:
            ctrl = ModelController(_load_model(path), label=os.path.basename(path),
                                   deterministic=self._deterministic)
            self._cache[path] = ctrl
        return ctrl

    # ── auto-ramp (bootstrap / curriculum) ───────────────────────────────────
    def _effective_self_play_frac(self) -> float:
        """Scale the self-play fraction with how full the pool is.

        No learner snapshot yet -> 0.0 (face only scripted/other-deck snapshots);
        grows linearly toward ``self_play_frac`` and saturates once there is at
        least one snapshot per roster deck. Makes the warm-start-vs-scripted
        curriculum automatic rather than a manual phase switch."""
        if self._latest_self is None:
            return 0.0
        target = max(1, len(self._roster))
        return self._self_play_frac * min(1.0, self._total_snaps / float(target))

    # ── per-episode sampling ─────────────────────────────────────────────────
    def sample_episode(self) -> tuple[str, str, Controller]:
        """Return (opp_deck, label, controller) for this episode."""
        self._maybe_refresh()
        # 1. latest-self slot (fast learning vs the current frontier).
        if (self._latest_self is not None
                and self._rng.random() < self._effective_self_play_frac()):
            return (self._learner, os.path.basename(self._latest_self),
                    self._model_for(self._latest_self))
        # 2. scripted anchor — a fixed floor carved out of the historical share so
        #    the collapse guard never vanishes (also the cold-start fallback when
        #    this process holds no snapshots).
        if (not self._snap_entries) or self._rng.random() < self._anchor_frac:
            deck = self._roster[int(self._rng.integers(len(self._roster)))]
            return deck, self._scripted_spec, self._scripted()
        # 3. weighted historical snapshot (cross-deck league + mirror self-play).
        weights = self._entry_weights()
        idx = int(self._rng.choice(len(self._snap_entries), p=weights))
        path, deck = self._snap_entries[idx]
        return deck, os.path.basename(path), self._model_for(path)

    def _entry_weights(self) -> np.ndarray:
        """Normalised PFSP/softmax weights for this process's snapshot entries.

        Unseen entries default to the current max weight so fresh snapshots get
        tried (OpenAI-Five 'init new snapshot quality to the max' rule)."""
        default = max(self._weights.values()) if self._weights else 1.0
        w = np.array([
            self._weights.get((deck, os.path.basename(path)), default)
            for path, deck in self._snap_entries], dtype=float)
        w = np.clip(w, 1e-8, None)
        return w / w.sum()
