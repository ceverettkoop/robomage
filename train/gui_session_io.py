"""Save/load for GUI sessions — Qt-free, testable headless.

Two on-disk kinds, routed by `sniff_kind`:

* ``.rmplay`` — a PLAY session as a deterministic replay spec (JSON): the
  engine seed, every action index fed to env.step, decks/seat/opponent spec,
  and an obs-hash tripwire. Small; reopening replays the game to its saved
  point (see game_driver.GameDriver's replay_actions).
* ``.rmtrace`` — an ANALYSIS session: the collected game traces
  (analysis._collect_game_traces schema) flattened into one
  ``np.savez_compressed`` following the az shard precedent, with a JSON meta
  blob for provenance and the ragged/string parts. Layout widths are stamped
  and validated on load exactly like shard_replay.load_shard_rows, so a file
  recorded under a different engine build fails loudly instead of misframing.
  ``interp_features`` are NOT stored — they are a deterministic function of
  the obs and are recomputed on load.

`trace_from_replay` converts a loaded replay into one browsable game trace
(fresh env, forced seed, the saved action list driving BOTH seats) so an
opened play session can land in the analysis browser.
"""

import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np

from env import MAX_ACTIONS, OBS_SIZE, STATE_SIZE

PLAY_FORMAT = "robomage-play-replay"
TRACE_FORMAT = "robomage-analysis-traces"
PLAY_VERSION = 1
TRACE_VERSION = 1

PLAY_EXT = ".rmplay"
TRACE_EXT = ".rmtrace"

# Per-game trace keys serialized as flat per-STEP arrays (ragged → spans).
_STEP_ARRAYS = ("values", "actions", "num_choices", "prefix_len",
                "clock_remaining")


def obs_sha256(obs):
    """Hash of an observation's float32 bytes — the replay divergence tripwire."""
    return hashlib.sha256(np.asarray(obs, dtype=np.float32).tobytes()).hexdigest()


def engine_build_stamp():
    """The live layout constants a saved session must match to replay."""
    return {"obs_size": int(OBS_SIZE), "max_actions": int(MAX_ACTIONS),
            "state_size": int(STATE_SIZE)}


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Play sessions (.rmplay) ───────────────────────────────────────────────────

def save_replay(path, *, engine_seed, actions, deck_a, deck_b, human_deck,
                opp_deck, human_is_a, bo3, opponent_spec, analysis=None,
                binary=None, final_obs=None, in_progress=True, extra=None):
    """Write a play-session replay spec. `actions` is the driver's action_log
    snapshot; `final_obs` (the obs at save time) stamps the divergence hash.
    `extra` merges additional keys into the doc (e.g. the shard recorder's
    ``row_decision_idx``); reserved keys cannot be overridden."""
    doc = {
        "format": PLAY_FORMAT, "version": PLAY_VERSION,
        "engine_seed": int(engine_seed) if engine_seed is not None else None,
        "actions": [int(a) for a in actions],
        "deck_a": deck_a, "deck_b": deck_b,
        "human_deck": human_deck, "opp_deck": opp_deck,
        "human_is_a": bool(human_is_a), "bo3": bool(bo3),
        "opponent_spec": opponent_spec,
        "analysis": analysis,
        "binary": binary,
        "engine_build": engine_build_stamp(),
        "history_len": len(actions),
        "final_obs_sha256": obs_sha256(final_obs) if final_obs is not None else None,
        "in_progress": bool(in_progress),
        "saved_at": _now(),
    }
    if extra:
        doc.update({k: v for k, v in extra.items() if k not in doc})
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def load_replay(path):
    """Read + validate a .rmplay file. Raises ValueError with a user-facing
    message on a wrong-kind file or an engine-build mismatch."""
    with open(path) as f:
        doc = json.load(f)
    if doc.get("format") != PLAY_FORMAT:
        raise ValueError(f"{path} is not a RoboMage play-session file")
    if doc.get("engine_seed") is None:
        raise ValueError(f"{path} carries no engine seed — it cannot be replayed")
    build = doc.get("engine_build") or {}
    if (build.get("obs_size") != OBS_SIZE
            or build.get("max_actions") != MAX_ACTIONS):
        raise ValueError(
            f"{path} was saved with a different engine build (obs "
            f"{build.get('obs_size')}x{build.get('max_actions')}, this build is "
            f"{OBS_SIZE}x{MAX_ACTIONS}) — check out the matching build or "
            f"re-save the session")
    return doc


# ── Analysis sessions (.rmtrace) ──────────────────────────────────────────────

def save_traces(path, games, provenance=None):
    """Write finished game traces (the _collect_game_traces per-game dict
    schema; shard records and whatif branches included) as one compressed npz.
    Live placeholders must be excluded by the caller."""
    games = list(games)
    n_steps = sum(len(g["observations"]) for g in games)
    obs = np.zeros((n_steps, OBS_SIZE), dtype=np.float32)
    probs = np.full((n_steps, MAX_ACTIONS), np.nan, dtype=np.float32)
    flat = {k: np.zeros(n_steps, dtype=np.float64) for k in _STEP_ARRAYS}
    game_span = np.zeros((len(games), 2), dtype=np.int64)
    full_actions = []
    full_span = np.zeros((len(games), 2), dtype=np.int64)
    engine_seed = np.full(len(games), -1, dtype=np.int64)
    result = np.zeros(len(games), dtype=np.float32)
    model_is_a = np.zeros(len(games), dtype=bool)
    per_game_meta = []

    row = 0
    for gi, g in enumerate(games):
        k = len(g["observations"])
        game_span[gi] = (row, row + k)
        for i in range(k):
            o = np.asarray(g["observations"][i], dtype=np.float32)
            if o.shape[0] != OBS_SIZE:
                raise ValueError(
                    f"game {gi} obs width {o.shape[0]} != OBS_SIZE {OBS_SIZE} "
                    f"— traces from a different engine build cannot be saved")
            obs[row + i] = o
            p = (g.get("action_probs") or [None] * k)[i] if g.get("action_probs") else None
            if p is not None:
                p = np.asarray(p, dtype=np.float32)
                probs[row + i, :p.shape[0]] = p
        _fill_step_arrays(flat, g, row, k)
        fa = g.get("full_actions")
        start = len(full_actions)
        if fa:
            full_actions.extend(int(a) for a in fa)
        full_span[gi] = (start, len(full_actions))
        if g.get("engine_seed") is not None:
            engine_seed[gi] = int(g["engine_seed"])
        result[gi] = float(g["result"]) if g.get("result") is not None else 0.0
        model_is_a[gi] = bool(g.get("model_is_a", True))
        per_game_meta.append({
            "opp_actions": g.get("opp_actions") or [],
            "clock_bank": g.get("clock_bank"),
            "opp_clock_bank": g.get("opp_clock_bank"),
            "shard": bool(g.get("shard")),
            "whatif": g.get("whatif"),
            "has_full_actions": g.get("full_actions") is not None,
            "has_probs": bool(g.get("action_probs")),
        })
        row += k

    meta = {
        "format": TRACE_FORMAT, "version": TRACE_VERSION,
        "obs_size": int(OBS_SIZE), "max_actions": int(MAX_ACTIONS),
        "saved_at": _now(),
        "provenance": provenance or {},
        "games": per_game_meta,
    }
    # A file object keeps the .rmtrace name (np.savez_compressed appends .npz
    # to a bare path that lacks it).
    with open(path, "wb") as f:
        np.savez_compressed(
            f, obs=obs, probs=probs,
            values=flat["values"].astype(np.float32),
            # NaN marks a missing entry (e.g. a shard record's prefix_len=None);
            # casting NaN to int is undefined, so stamp the -1 sentinel first.
            actions=np.nan_to_num(flat["actions"], nan=-1.0).astype(np.int32),
            num_choices=np.nan_to_num(flat["num_choices"],
                                      nan=-1.0).astype(np.int32),
            prefix_len=np.nan_to_num(flat["prefix_len"],
                                     nan=-1.0).astype(np.int32),
            clock_remaining=flat["clock_remaining"].astype(np.float32),
            game_span=game_span,
            full_actions=np.asarray(full_actions, dtype=np.int32),
            full_span=full_span, engine_seed=engine_seed, result=result,
            model_is_a=model_is_a, meta=np.array(json.dumps(meta)))
    return path


def _fill_step_arrays(flat, g, row, k):
    for key in _STEP_ARRAYS:
        src = g.get(key) or []
        for i in range(k):
            v = src[i] if i < len(src) else None
            flat[key][row + i] = np.nan if v is None else float(v)


def load_traces(path, interp_fn=None):
    """Read + validate a .rmtrace file back into the per-game dict list.

    `interp_fn(obs) -> array` recomputes interp_features (pass
    analysis._extract_interpretable; None leaves them empty, which only the
    shap/consistency views need). Returns (games, meta)."""
    with np.load(path, allow_pickle=False) as d:
        try:
            meta = json.loads(str(d["meta"]))
        except KeyError:
            raise ValueError(f"{path} is not a RoboMage analysis-session file")
        if meta.get("format") != TRACE_FORMAT:
            raise ValueError(f"{path} is not a RoboMage analysis-session file")
        if (d["obs"].shape[1] != OBS_SIZE
                or meta.get("max_actions") != MAX_ACTIONS):
            raise ValueError(
                f"trace layout mismatch in {path}: obs width {d['obs'].shape[1]} "
                f"(expected {OBS_SIZE}), max_actions {meta.get('max_actions')} "
                f"(expected {MAX_ACTIONS}) — re-record with this build")
        obs = d["obs"]
        probs = d["probs"]
        values = d["values"]
        actions = d["actions"]
        num_choices = d["num_choices"]
        prefix_len = d["prefix_len"]
        clock_remaining = d["clock_remaining"]
        game_span = d["game_span"]
        full_actions = d["full_actions"]
        full_span = d["full_span"]
        engine_seed = d["engine_seed"]
        result = d["result"]
        model_is_a = d["model_is_a"]

        games = []
        for gi, gm in enumerate(meta["games"]):
            lo, hi = (int(x) for x in game_span[gi])
            g_obs = [obs[i].copy() for i in range(lo, hi)]
            g_probs = None
            if gm.get("has_probs"):
                g_probs = []
                for i in range(lo, hi):
                    row = probs[i, :int(num_choices[i])]
                    g_probs.append(None if np.isnan(row).all()
                                   else row.astype(np.float64))
            flo, fhi = (int(x) for x in full_span[gi])
            # A shard record has no replay prefixes (prefix_len=None, see
            # shard_replay); the -1 sentinel restores that None on load.
            g_prefix = [int(p) for p in prefix_len[lo:hi]]
            if g_prefix and all(p < 0 for p in g_prefix):
                g_prefix = None
            clock = [None if np.isnan(c) else float(c)
                     for c in clock_remaining[lo:hi]]
            game = {
                "observations": g_obs,
                "values": [float(v) for v in values[lo:hi]],
                "interp_features": ([interp_fn(o) for o in g_obs]
                                    if interp_fn is not None else []),
                "actions": [int(a) for a in actions[lo:hi]],
                "num_choices": [int(n) for n in num_choices[lo:hi]],
                "action_probs": g_probs,
                "opp_actions": gm.get("opp_actions") or [],
                "engine_seed": (int(engine_seed[gi])
                                if engine_seed[gi] >= 0 else None),
                "full_actions": ([int(a) for a in full_actions[flo:fhi]]
                                 if gm.get("has_full_actions") else None),
                "prefix_len": g_prefix,
                "result": float(result[gi]),
                "model_is_a": bool(model_is_a[gi]),
                "clock_remaining": clock,
                "clock_bank": gm.get("clock_bank"),
                "opp_clock_bank": gm.get("opp_clock_bank"),
            }
            if gm.get("shard"):
                game["shard"] = True
            if gm.get("whatif"):
                game["whatif"] = gm["whatif"]
            games.append(game)
    return games, meta


# ── Kind sniffing ─────────────────────────────────────────────────────────────

def sniff_kind(path):
    """'play' | 'trace' | None, by extension first, content second."""
    ext = os.path.splitext(path)[1].lower()
    if ext == PLAY_EXT:
        return "play"
    if ext == TRACE_EXT:
        return "trace"
    try:
        with open(path, "rb") as f:
            head = f.read(2)
    except OSError:
        return None
    if head[:2] == b"PK":                      # npz = zip container
        return "trace"
    if head[:1] == b"{":
        try:
            with open(path) as f:
                doc = json.load(f)
            return "play" if doc.get("format") == PLAY_FORMAT else None
        except (OSError, ValueError):
            return None
    return None


# ── Replay → trace conversion ─────────────────────────────────────────────────

def trace_from_replay(replay, binary_path, value_fn=None, viewpoint=None):
    """Replay a loaded .rmplay deterministically and record it as ONE game
    trace in the _collect_game_traces schema, so an opened play session can be
    browsed in the analysis browser.

    The saved action list drives BOTH seats (it is the global env.step
    sequence). `viewpoint` picks whose decisions become the trace's model
    steps — "A"/"B", default the saved human seat. `value_fn(obs) -> float`
    supplies V(s) per recorded step (e.g. a loaded net); without one values
    are 0.0 (the histogram stays flat but every other view works; action
    probs are absent, which the probs-dependent analyses self-report)."""
    import runner
    import analysis as an
    from env import RoboMageEnv
    from opponents import ActionListController

    view_is_a = (replay["human_is_a"] if viewpoint is None
                 else viewpoint == "A")
    env = RoboMageEnv(binary_path=binary_path, deck_a=replay["deck_a"],
                      deck_b=replay["deck_b"], bo3=replay["bo3"])
    try:
        obs, _ = env.reset(options={"engine_seed": replay["engine_seed"]})
        ctrl = ActionListController(replay["actions"], label="Replay")

        trace = {"observations": [], "values": [], "interp_features": [],
                 "actions": [], "num_choices": [], "action_probs": None,
                 "opp_actions": [], "engine_seed": replay["engine_seed"],
                 "prefix_len": [], "model_is_a": view_is_a,
                 "clock_remaining": [], "clock_bank": None,
                 "opp_clock_bank": None,
                 # Absolute seat decks: the browser's replay-to-step search
                 # rebuilds an env from these (no model-seat deck swap).
                 "replay_decks": {"deck_a": replay["deck_a"],
                                  "deck_b": replay["deck_b"],
                                  "bo3": bool(replay.get("bo3", True))}}

        def on_query(d):
            if d.priority_is_a != view_is_a:
                return
            trace["observations"].append(d.obs.copy())
            trace["values"].append(float(value_fn(d.obs)) if value_fn else 0.0)
            trace["interp_features"].append(an._extract_interpretable(d.obs))
            trace["num_choices"].append(d.num_choices)
            trace["clock_remaining"].append(None)
            trace["prefix_len"].append(d.index)

        def on_action(d, action):
            if d.priority_is_a == view_is_a:
                trace["actions"].append(int(action))
            else:
                trace["opp_actions"].append({
                    "desc": an._action_desc(d.obs, action),
                    "before_model_step": len(trace["observations"]),
                    "clock": None})

        rec = runner.drive_game(env, obs, ctrl, ctrl,
                                on_query=on_query, on_action=on_action,
                                max_decisions=len(replay["actions"]))
    finally:
        env.close()

    trace["full_actions"] = rec.actions
    trace["result"] = rec.reward if view_is_a else -rec.reward
    return trace
