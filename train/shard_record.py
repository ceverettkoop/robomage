"""Record interactive play into AZ trainer-schema shards.

`ShardRecorder` turns a GUI/TUI play session (human vs a search opponent) into
a directory of ``shard_*.npz`` files in the exact self-play schema
(:data:`az_selfplay.SHARD_KEYS`), so every existing shard consumer works on a
recorded session unchanged: the analysis browser's shard mode
(``shard_replay.load_records`` behind ``tui_analysis --shards`` /
``gui_browser``), ``az_inspect --shards`` / ``tui_az_inspect --shards``, and
even ``az_train.load_window``.

Row sources, mirroring the two self-play precedents — and built by the SAME
two builders self-play uses (``az_selfplay.sample_from_search_result`` /
``one_hot_sample``), so a recorded row is bit-identical to a self-play one:
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
:meth:`flush`. Readers glob ``shard_*.npz`` and never see a partial file or a
duplicated row, and a game is never split across files (the
``shard_replay.segment_matches`` contract). Rows of a game still in progress
are flushed with ``z = 0`` (an unfinished game has no outcome; az_selfplay
drops such rows at truncation via ``_drop_unfinished``, which this module
therefore deliberately does NOT call — here they are the whole point of mid-game
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
    """Accumulate play-session decisions and write trainer-schema shards."""

    def __init__(self, out_dir, td_n=None):
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
        os.makedirs(out_dir, exist_ok=True)

    # ----- taps (driver worker thread) -----

    def on_search_result(self, obs, num_choices, result, chosen):
        """``SearchController.on_result`` hook: stash the searched row; the
        matching :meth:`observe_step` call commits it."""
        from az_selfplay import sample_from_search_result
        sample = sample_from_search_result(obs, num_choices, result)
        # The builder leaves explored=0 for its caller to finalize; here the
        # played action is the controller's `chosen`.
        sample["explored"] = int(int(chosen) != int(result.best_action()))
        with self._lock:
            self._pending = sample

    def observe_step(self, obs, num_choices, action, reward, info, done):
        """``GameDriver.step_observer`` hook, once per stepped decision.

        Commits the row for THIS decision (the stashed searched row if the
        opponent's search produced one, else a one-hot behavior row when the
        menu offered a real choice), then processes any game/match boundary
        this step's ``(reward, info)`` carries."""
        from az_selfplay import one_hot_sample, winner_from_reward
        num_choices = int(num_choices)
        action = int(action)
        with self._lock:
            sample, self._pending = self._pending, None
            if sample is None and num_choices > 1 \
                    and 0 <= action < num_choices:
                sample = one_hot_sample(obs, num_choices, action)
            if sample is not None:
                sample["game_idx"] = self._games_done
                self._samples.append(sample)
                self.rows_recorded += 1
                self._dirty = True

            # Boundaries, az_selfplay's rules: a GAME_RESULT step ends a bo3
            # game (winner = reward sign, Player-A perspective); a terminated
            # step without one is the bo1 ending.
            boundary = bool(info.get("game_result"))
            if boundary or (done and not boundary and self._samples):
                self._game_winners.append(winner_from_reward(reward))
                self._games_done += 1
                self._flush_locked()
            if done:
                if self._dirty:
                    self._flush_locked()
                # The match is over: the next rows open a new shard file.
                if self._samples:
                    self._match_idx += 1
                self._samples = []
                self._game_winners = []
                self._games_done = 0
                self._dirty = False

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
        self._dirty = False
        return path
