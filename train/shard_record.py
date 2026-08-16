"""Record interactive play into AZ trainer-schema shards.

`ShardRecorder` turns a GUI/TUI play session (human vs a search opponent) into
a directory of ``shard_*.npz`` files in the exact self-play schema
(:data:`az_selfplay.SHARD_KEYS`), so every existing shard consumer works on a
recorded session unchanged: the analysis browser's shard mode
(``shard_replay.load_records`` behind ``tui_analysis --shards`` /
``gui_browser``), ``az_inspect --shards`` / ``tui_az_inspect --shards``, and
even ``az_train.load_window``.

Row sources, mirroring the two self-play precedents:
  * A SEARCHED opponent decision — fed through :meth:`on_search_result`, the
    ``SearchController.on_result`` hook signature — becomes a full row: ``pi``
    is the root visit posterior, ``q`` the search's root value,
    ``explored`` whether the played action differs from the visit argmax.
  * Every OTHER decision with >1 legal action (the human's decisions, and the
    opponent's unsearched fallback/trivial/tree-followed ones) becomes a
    behavior-cloning row exactly like ``az_selfplay.generate_expert`` writes:
    one-hot ``pi`` on the played action, ``q = NaN``, ``explored = 0`` (hence
    ``td_q = z``). This keeps BOTH seats' lines contiguous, so the browser can
    page either seat and match segmentation never sees gaps.

Commit protocol (all on the driver worker thread): ``on_search_result`` only
STASHES its row — :meth:`observe_step` (the ``GameDriver.step_observer`` hook,
called once per stepped decision with the pre-step obs) is the single commit
point. It attaches the stashed row to the decision being stepped, or builds
the one-hot row; then it reads the game/match boundaries off ``(reward,
info)`` exactly like az_selfplay's loop: ``info["game_result"]`` marks a bo3
game boundary whose winner is the reward sign (Player-A perspective), a bare
``terminated`` without one is the bo1 ending. z/td_q are backfilled per game
from each row's own mover's perspective via ``az_selfplay._backfill_and_pack``.

File protocol: ONE file per match — ``shard_{ts}_{pid}_{n}.npz`` with ``n``
the match index — atomically REWRITTEN (write ``.tmp`` + ``os.replace``) with
the match's full row prefix at every game boundary and on any explicit
:meth:`flush`. When the session wires ``replay_meta``/``seed_fn`` (see the
class docstring) each flush also rewrites a same-stem ``.rmplay`` replay
sidecar — engine seed + full action log + per-row replay prefixes — making
the recording exactly replayable for the browser's replay-to-step MCTS
analysis. Readers glob ``shard_*.npz`` and never see a partial file or a
duplicated row, and a game is never split across files (the
``shard_replay.segment_matches`` contract). Rows of a game still in progress
are flushed with ``z = 0`` (an unfinished game has no outcome; az_selfplay
drops such rows at truncation, but here they are the whole point of mid-game
analysis — the browser's net-V(s) overwrite makes them browsable, and the
``z = 0`` draws are the documented cost of pointing a TRAINER at a recorded
dir).

Threading: recording happens on the driver worker thread; :meth:`flush` may be
called from the UI thread (the "analyze mid-game" path), so all state is
guarded by one lock. Torch-free; ``az_selfplay`` (numpy-only at import) is
imported lazily on first flush.
"""

import os
import threading
import time

import numpy as np

from env import MAX_ACTIONS, _SELF_IS_A_IDX


def default_recording_dir(base_dir=None):
    """A fresh per-session recording directory:
    ``train/az_data/recorded/rec_{ts}_{pid}`` (not the training pool).
    ``ROBOMAGE_RECORD_DIR`` overrides the base (the smokes point it at a
    scratch dir so CI leaves no recordings behind)."""
    if base_dir is None:
        base_dir = os.environ.get("ROBOMAGE_RECORD_DIR") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "az_data", "recorded")
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(base_dir, f"rec_{ts}_{os.getpid()}")


class ShardRecorder:
    """Accumulate play-session decisions and write trainer-schema shards.

    With ``replay_meta`` (static session facts: ``deck_a``/``deck_b``/
    ``human_deck``/``opp_deck``/``human_is_a``/``bo3``/``opponent_spec``/
    ``binary``) and ``seed_fn`` (returns the env's ``last_engine_seed``), each
    match's shard also gets a ``.rmplay`` REPLAY SIDECAR (same stem,
    ``gui_session_io.save_replay`` format): the engine seed, every action index
    stepped this match (the recorder's own per-step tally — the
    ``step_observer`` hook sees every main-loop decision, 1-choice ones
    included), and ``row_decision_idx`` mapping each recorded shard row to its
    position in that action list. The sidecar makes a recording exactly
    replayable: ``shard_replay`` attaches seed/actions/prefixes to the
    browsable records, enabling the browser's replay-to-step MCTS analysis
    (and whatif) on recorded sessions. ``replay_prefix`` seeds the action list
    for a session continued from a saved ``.rmplay`` (whose replayed prefix
    the observer never sees). Without ``replay_meta``/``seed_fn`` no sidecar
    is written and behavior is exactly as before."""

    def __init__(self, out_dir, td_n=None, replay_meta=None, seed_fn=None,
                 replay_prefix=None):
        if td_n is None:
            from cli_spec import DEFAULT_AZ_TD_N
            td_n = DEFAULT_AZ_TD_N
        self.out_dir = out_dir
        self._td_n = int(td_n)
        self._lock = threading.Lock()
        self._ts = time.strftime("%Y%m%d_%H%M%S")
        self._match_idx = 0
        self._samples = []        # this match's rows, az_selfplay sample dicts
        self._game_winners = []   # this match's finished games: "A"/"B"/None
        self._games_done = 0      # == len(_game_winners); the next game_idx
        self._pending = None      # stashed searched row awaiting observe_step
        self._dirty = False       # rows not yet on disk
        self.rows_recorded = 0
        self._replay_meta = dict(replay_meta) if replay_meta else None
        self._seed_fn = seed_fn
        # This match's full action list (replayed prefix + every observed
        # step). The prefix applies to the FIRST match only — a continued
        # session resumes mid-match; any later match starts from its own reset.
        self._match_actions = list(replay_prefix) if replay_prefix else []
        self._match_over = False   # sidecar in_progress flag source
        os.makedirs(out_dir, exist_ok=True)

    # ----- taps (driver worker thread) -----

    def on_search_result(self, obs, num_choices, result, chosen):
        """``SearchController.on_result`` hook: stash the searched row; the
        matching :meth:`observe_step` call commits it."""
        num_choices = int(num_choices)
        pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
        pi[:num_choices] = result.policy_target(1.0).astype(np.float32)
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[:num_choices] = True
        with self._lock:
            self._pending = {
                "obs": np.asarray(obs, dtype=np.float32).copy(),
                "pi": pi, "mask": mask,
                "mover_is_a": bool(obs[_SELF_IS_A_IDX] > 0.5),
                "q": float(result.root_value),
                "explored": int(int(chosen) != int(result.best_action())),
            }

    def observe_step(self, obs, num_choices, action, reward, info, done):
        """``GameDriver.step_observer`` hook, once per stepped decision.

        Commits the row for THIS decision (the stashed searched row if the
        opponent's search produced one, else a one-hot behavior row when the
        menu offered a real choice), then processes any game/match boundary
        this step's ``(reward, info)`` carries."""
        num_choices = int(num_choices)
        action = int(action)
        with self._lock:
            self._match_actions.append(action)
            sample, self._pending = self._pending, None
            if sample is None and num_choices > 1 \
                    and 0 <= action < num_choices:
                pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
                pi[action] = 1.0
                mask = np.zeros(MAX_ACTIONS, dtype=bool)
                mask[:num_choices] = True
                sample = {
                    "obs": np.asarray(obs, dtype=np.float32).copy(),
                    "pi": pi, "mask": mask,
                    "mover_is_a": bool(obs[_SELF_IS_A_IDX] > 0.5),
                    "q": float("nan"), "explored": 0,
                }
            if sample is not None:
                sample["game_idx"] = self._games_done
                # Replay position of THIS decision: the action list up to (not
                # including) the action just appended is the prefix that parks
                # a replaying env exactly at it.
                sample["decision_idx"] = len(self._match_actions) - 1
                self._samples.append(sample)
                self.rows_recorded += 1
                self._dirty = True

            # Boundaries, az_selfplay's rules: a GAME_RESULT step ends a bo3
            # game (winner = reward sign, Player-A perspective); a terminated
            # step without one is the bo1 ending.
            boundary = bool(info.get("game_result"))
            if boundary or (done and not boundary and self._samples):
                self._game_winners.append(
                    "A" if reward > 0 else ("B" if reward < 0 else None))
                self._games_done += 1
                self._flush_locked()
            if done:
                self._match_over = True
                if self._dirty:
                    self._flush_locked()
                elif self._samples:
                    # Rows already on disk, but the sidecar's in_progress flag
                    # must still flip to a finished match.
                    self._write_sidecar_locked()
                # The match is over: the next rows open a new shard file.
                if self._samples:
                    self._match_idx += 1
                self._samples = []
                self._game_winners = []
                self._games_done = 0
                self._dirty = False
                self._match_actions = []
                self._match_over = False

    # ----- flush (any thread) -----

    def flush(self):
        """Rewrite the current match's shard with every row so far (rows of an
        unfinished game get ``z = 0``). Returns the shard path, or None when
        nothing has been recorded for this match yet. Safe from the UI thread
        (the mid-game "analyze now" path)."""
        with self._lock:
            return self._flush_locked()

    def close(self):
        """Final flush (idempotent)."""
        with self._lock:
            if self._dirty:
                self._flush_locked()

    def _flush_locked(self):
        if not self._samples:
            return None
        from az_selfplay import SHARD_KEYS, _backfill_and_pack
        # Pad winners so rows of the in-progress game price as z = 0 (winner
        # None) instead of indexing past the finished games.
        n_games = max(s["game_idx"] for s in self._samples) + 1
        winners = (self._game_winners
                   + [None] * (n_games - len(self._game_winners)))
        arrays = _backfill_and_pack(self._samples, winners, td_n=self._td_n)
        path = os.path.join(
            self.out_dir,
            f"shard_{self._ts}_{os.getpid()}_{self._match_idx}.npz")
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **{k: arrays[k] for k in SHARD_KEYS})
        os.replace(tmp, path)
        self._write_sidecar_locked()
        self._dirty = False
        return path

    def _write_sidecar_locked(self):
        """Write/refresh this match's ``.rmplay`` replay sidecar (same stem as
        the shard, atomic replace). Skipped without ``replay_meta``/``seed_fn``
        or before the env's first reset has produced a seed."""
        meta, seed_fn = self._replay_meta, self._seed_fn
        if meta is None or seed_fn is None or not self._samples:
            return
        seed = seed_fn()
        if seed is None:
            return
        import gui_session_io
        path = os.path.join(
            self.out_dir,
            f"shard_{self._ts}_{os.getpid()}_{self._match_idx}"
            + gui_session_io.PLAY_EXT)
        tmp = path + ".tmp"
        gui_session_io.save_replay(
            tmp,
            engine_seed=seed,
            actions=self._match_actions,
            deck_a=meta.get("deck_a"), deck_b=meta.get("deck_b"),
            human_deck=meta.get("human_deck"), opp_deck=meta.get("opp_deck"),
            human_is_a=bool(meta.get("human_is_a", True)),
            bo3=bool(meta.get("bo3", True)),
            opponent_spec=meta.get("opponent_spec"),
            binary=meta.get("binary"),
            in_progress=not self._match_over,
            extra={"row_decision_idx":
                   [int(s["decision_idx"]) for s in self._samples]})
        os.replace(tmp, path)
