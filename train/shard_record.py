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
    is the root visit posterior, ``q`` the search's Q of the played action
    (``az_selfplay.finalize_searched_sample``), ``explored`` whether the
    played action differs from the visit argmax.
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
drops such rows at truncation via ``_drop_unfinished``, which this module
therefore deliberately does NOT call — here they are the whole point of mid-game
analysis — the browser's net-V(s) overwrite makes them browsable, and the
``z = 0`` draws are the documented cost of pointing a TRAINER at a recorded
dir).

Search diagnostics sidecar: each shard also gets a same-stem ``.diag`` (an
npz archive — the extension avoids the readers' ``shard_*.npz`` glob;
:data:`DIAG_KEYS`, packed by :func:`pack_diag`, row index == shard row) with
the per-decision MCTS root facts the trainer schema deliberately omits — full
visit / prior / Q / W vectors, sim counts, the per-world seeds, visits and
values, and the time budget — so a recorded search can be rebuilt exactly.
``kind`` tags each row: 0 = plain one-hot row (no search), 1 = in-game PUCT
search, 2 = tree-FOLLOWED decision (answered from the most recent kind-1
row's trees — ``origin_row`` points at it, ``follow_path`` is the raw action
path from that root and ``follow_worlds`` the surviving origin worlds), 3 =
sideboard plan search. Static provenance (spec, checkpoint hash, knobs,
device) rides in the ``.rmplay`` sidecar's ``search_provenance``.

Threading: recording happens on the driver worker thread; :meth:`flush` may be
called from the UI thread (the "analyze mid-game" path), so all state is
guarded by one lock. Torch-free; ``az_selfplay`` (numpy-only at import) is
imported lazily on first flush.
"""

import os
import threading
import time

import numpy as np

from env import MAX_ACTIONS, _IS_SIDEBOARD_IDX

# The diag sidecar is an npz archive (np.load sniffs the magic bytes) but
# deliberately does NOT end in ".npz": every trainer/browser reader globs
# ``shard_*.npz`` and would otherwise ingest it as a schema-less shard.
DIAG_EXT = ".diag"

# Row kinds in the .diag sidecar.
DIAG_KIND_NONE = 0       # one-hot behavior row, no search
DIAG_KIND_SEARCH = 1     # in-game PUCT search
DIAG_KIND_FOLLOWED = 2   # answered from a previous search's trees
DIAG_KIND_PLAN = 3       # sideboard plan search

DIAG_KEYS = (
    "kind", "visits", "priors", "q_act", "w_sum", "root_value", "sims_run",
    "sim_steps", "reused_visits", "memo_hits", "stopped_early",
    "time_budget_s", "time_budget_min_s", "n_worlds", "world_seeds",
    "world_visits", "world_values", "origin_row", "follow_path",
    "follow_worlds",
)


def diag_path_for(shard_path):
    """The ``.diag`` sidecar path beside a ``shard_*.npz``."""
    stem, _ext = os.path.splitext(shard_path)
    return stem + DIAG_EXT


def _menu_vec(values, num_choices, dtype):
    """A MAX_ACTIONS-wide row from a menu-length vector (None -> zeros)."""
    row = np.zeros(MAX_ACTIONS, dtype=dtype)
    if values is not None:
        v = np.asarray(values).reshape(-1)[:num_choices]
        row[:len(v)] = v
    return row


def _world_row(values, width, dtype, fill):
    row = np.full(width, fill, dtype=dtype)
    if values is not None:
        v = np.asarray(values).reshape(-1)[:width]
        row[:len(v)] = v
    return row


def pack_diag(diags):
    """Pack a list of per-row diag dicts (None for a plain row) into the
    fixed-width :data:`DIAG_KEYS` arrays. Pure; shared with ``shard_replay``
    and the tests."""
    n = len(diags)
    w_max = max([1] + [len(d.get("seeds") or []) for d in diags if d]
                + [len(d.get("world_visits") or []) for d in diags if d]
                + [int(w) + 1 for d in diags if d
                   for w in (d.get("follow_worlds") or [])])
    p_max = max([1] + [len(d.get("follow_path") or []) for d in diags if d])
    out = {
        "kind": np.zeros(n, dtype=np.int8),
        "visits": np.zeros((n, MAX_ACTIONS), dtype=np.int32),
        "priors": np.zeros((n, MAX_ACTIONS), dtype=np.float32),
        "q_act": np.zeros((n, MAX_ACTIONS), dtype=np.float32),
        "w_sum": np.zeros((n, MAX_ACTIONS), dtype=np.float32),
        "root_value": np.zeros(n, dtype=np.float32),
        "sims_run": np.zeros(n, dtype=np.int32),
        "sim_steps": np.zeros(n, dtype=np.int32),
        "reused_visits": np.zeros(n, dtype=np.int32),
        "memo_hits": np.zeros(n, dtype=np.int32),
        "stopped_early": np.zeros(n, dtype=np.uint8),
        "time_budget_s": np.full(n, np.nan, dtype=np.float32),
        "time_budget_min_s": np.full(n, np.nan, dtype=np.float32),
        "n_worlds": np.zeros(n, dtype=np.int16),
        "world_seeds": np.full((n, w_max), -1, dtype=np.int64),
        "world_visits": np.zeros((n, w_max), dtype=np.int32),
        "world_values": np.full((n, w_max), np.nan, dtype=np.float32),
        "origin_row": np.full(n, -1, dtype=np.int32),
        "follow_path": np.full((n, p_max), -1, dtype=np.int32),
        "follow_worlds": np.zeros((n, w_max), dtype=np.uint8),
    }
    for i, d in enumerate(diags):
        if not d:
            continue
        nc = int(d.get("num_choices", MAX_ACTIONS))
        out["kind"][i] = int(d.get("kind", DIAG_KIND_NONE))
        out["visits"][i] = _menu_vec(d.get("visits"), nc, np.int32)
        out["priors"][i] = _menu_vec(d.get("priors"), nc, np.float32)
        out["q_act"][i] = _menu_vec(d.get("q_act"), nc, np.float32)
        out["w_sum"][i] = _menu_vec(d.get("w_sum"), nc, np.float32)
        out["root_value"][i] = float(d.get("root_value", 0.0))
        for k in ("sims_run", "sim_steps", "reused_visits", "memo_hits"):
            out[k][i] = int(d.get(k, 0))
        out["stopped_early"][i] = 1 if d.get("stopped_early") else 0
        for k in ("time_budget_s", "time_budget_min_s"):
            v = d.get(k)
            out[k][i] = np.nan if v is None else float(v)
        seeds = d.get("seeds") or []
        wv = d.get("world_visits")
        wvals = d.get("world_values")
        n_worlds = max(len(seeds), 0 if wv is None else len(wv),
                       0 if wvals is None else len(wvals))
        out["n_worlds"][i] = n_worlds
        out["world_seeds"][i] = _world_row(seeds, w_max, np.int64, -1)
        out["world_visits"][i] = _world_row(wv, w_max, np.int32, 0)
        out["world_values"][i] = _world_row(wvals, w_max, np.float32, np.nan)
        out["origin_row"][i] = int(d.get("origin_row", -1))
        out["follow_path"][i] = _world_row(d.get("follow_path"), p_max,
                                           np.int32, -1)
        fw = np.zeros(w_max, dtype=np.uint8)
        for w in d.get("follow_worlds") or []:
            if 0 <= int(w) < w_max:
                fw[int(w)] = 1
        out["follow_worlds"][i] = fw
    return out


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
        self._diags = []          # parallel to _samples: diag dict or None
        self._game_winners = []   # this match's finished games: "A"/"B"/None
        self._games_done = 0      # == len(_game_winners); the next game_idx
        self._pending = None      # stashed searched row awaiting observe_step
        self._pending_diag = None  # its diag dict (or a followed row's)
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
        from az_selfplay import (finalize_searched_sample,
                                 sample_from_search_result)
        sample = sample_from_search_result(obs, num_choices, result)
        # The builder leaves explored/q for its caller to finalize; here the
        # played action is the controller's `chosen`.
        finalize_searched_sample(sample, result, chosen)
        diag = self._diag_from_result(obs, num_choices, result)
        with self._lock:
            self._pending = sample
            self._pending_diag = diag

    @staticmethod
    def _diag_from_result(obs, num_choices, result):
        """The diag dict for a searched decision. Every field is read with
        getattr so a minimal duck-typed result still records."""
        g = lambda name, default=None: getattr(result, name, default)  # noqa: E731
        num_choices = int(num_choices)
        roots = g("roots")
        plan = (not roots) and float(obs[_IS_SIDEBOARD_IDX]) > 0.5
        q = g("q")
        w_sum = g("w_sum")
        seeds = g("seeds")
        world_visits = g("world_visits")
        world_values = g("world_values")
        return {
            "kind": DIAG_KIND_PLAN if plan else DIAG_KIND_SEARCH,
            "num_choices": num_choices,
            "visits": np.asarray(result.visits[:num_choices]).astype(np.int64),
            "priors": (None if g("priors") is None
                       else np.asarray(g("priors")[:num_choices],
                                       dtype=np.float32)),
            "q_act": (np.zeros(num_choices, dtype=np.float32) if q is None
                      else np.asarray(q[:num_choices], dtype=np.float32)),
            "w_sum": (np.zeros(num_choices, dtype=np.float32) if w_sum is None
                      else np.asarray(w_sum[:num_choices], dtype=np.float32)),
            "root_value": float(g("root_value", 0.0)),
            "sims_run": int(g("sims_run", 0)),
            "sim_steps": int(g("sim_steps", 0)),
            "reused_visits": int(g("reused_visits", 0)),
            "stopped_early": bool(g("stopped_early", False)),
            "memo_hits": int(g("memo_hits", 0)),
            "seeds": [int(s) for s in seeds] if seeds is not None else [],
            "world_visits": (None if world_visits is None
                             else [int(v) for v in world_visits]),
            "world_values": (None if world_values is None
                             else [float(v) for v in world_values]),
            "time_budget_s": g("time_budget_s"),
            "time_budget_min_s": g("time_budget_min_s"),
        }

    def on_followed(self, obs, num_choices, visits, path, world_idx, chosen):
        """``SearchController.on_followed`` hook: stash the diag of a
        tree-followed decision (the trainer row itself is the one-hot
        :meth:`observe_step` builds). ``origin_row`` is the most recent
        searched (kind 1) row this match, whose trees were followed."""
        num_choices = int(num_choices)
        with self._lock:
            origin = -1
            for i in range(len(self._diags) - 1, -1, -1):
                d = self._diags[i]
                if d is not None and d["kind"] == DIAG_KIND_SEARCH:
                    origin = i
                    break
            self._pending_diag = {
                "kind": DIAG_KIND_FOLLOWED,
                "num_choices": num_choices,
                "visits": np.asarray(visits[:num_choices]).astype(np.int64),
                "origin_row": origin,
                "follow_path": [int(a) for a in path],
                "follow_worlds": [int(w) for w in world_idx],
            }

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
            self._match_actions.append(action)
            sample, self._pending = self._pending, None
            diag, self._pending_diag = self._pending_diag, None
            if sample is None and num_choices > 1 \
                    and 0 <= action < num_choices:
                sample = one_hot_sample(obs, num_choices, action)
            if sample is not None:
                sample["game_idx"] = self._games_done
                # Replay position of THIS decision: the action list up to (not
                # including) the action just appended is the prefix that parks
                # a replaying env exactly at it.
                sample["decision_idx"] = len(self._match_actions) - 1
                self._samples.append(sample)
                self._diags.append(diag)
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
                self._diags = []
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
        self._write_diag_locked(path)
        self._write_sidecar_locked()
        self._dirty = False
        return path

    def _write_diag_locked(self, shard_path):
        """Write/refresh the shard's ``.diag`` search-diagnostics sidecar
        (row index == shard row; atomic replace)."""
        diag = pack_diag(self._diags)
        path = diag_path_for(shard_path)
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **{k: diag[k] for k in DIAG_KEYS})
        os.replace(tmp, path)

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
        extra = {"row_decision_idx":
                 [int(s["decision_idx"]) for s in self._samples]}
        if meta.get("search_provenance") is not None:
            extra["search_provenance"] = meta["search_provenance"]
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
            extra=extra)
        os.replace(tmp, path)
