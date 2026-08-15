#!/usr/bin/env python3
"""GUI session save/load regression (gui_session_io.py).

Proves the on-disk session guarantees the GUI relies on:

  1. replay round-trip — a scripted game recorded through env.step, saved as
     .rmplay and replayed on a FRESH env with the saved engine seed lands on a
     byte-identical final obs, matching the saved obs_sha256 tripwire.
  2. trace round-trip — a mixed game list (normal with probs/full_actions, a
     shard-style record without either, a whatif branch) survives
     save_traces/load_traces: obs bytes, float64 probs (None steps included),
     None round-trips (engine_seed, clock entries, shard prefix_len), the
     shard/whatif markers, and seat/result fields.
  3. validation — wrong-build .rmplay, wrong-format JSON, seedless replay,
     .rmplay fed to load_traces, meta-less npz, tampered max_actions and
     garbage files all fail loudly; sniff_kind routes by extension first and
     content second, returning None for foreign files.
  4. trace_from_replay — a recorded scripted prefix converts into ONE
     browsable trace whose full_actions equal the replayed prefix, with the
     viewpoint seat's decisions as model steps.

Needs bin/robomage (legs 1 and 4); torch-free. Runnable standalone::

    train/.venv/bin/python train/test_gui_session_io.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gui_session_io as gio  # noqa: E402
from cli_spec import BINARY, BIN_DIR  # noqa: E402
from env import MAX_ACTIONS, OBS_SIZE  # noqa: E402

SEED = 123          # engine seed for the recorded games
N_RECORD = 25       # scripted actions recorded before saving


class SessionIoTestError(AssertionError):
    pass


def _check(cond, msg):
    if not cond:
        raise SessionIoTestError(msg)


def _write_decks():
    """Fast lopsided matchup (same fixture as test_analysis_session)."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "session_io_a.dk")
    b = os.path.join(d, "session_io_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\n")
    with open(b, "w") as f:
        f.write("60 Swamp\n")
    return "temp/session_io_a", "temp/session_io_b", [a, b]


def _rm(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _record_scripted(deck_a, deck_b, n_actions):
    """Play n scripted actions on a fresh env; return (actions, final_obs)."""
    from env import RoboMageEnv
    from scripted_agent import make_agent

    env = RoboMageEnv(binary_path=BINARY, deck_a=deck_a, deck_b=deck_b)
    try:
        env.reset(options={"engine_seed": SEED})
        agent = make_agent("scripted")
        agent.new_game()
        actions = []
        for _ in range(n_actions):
            a = int(agent.act(env._obs, env._num_choices))
            actions.append(a)
            _, _, term, trunc, _ = env.step(a)
            if term or trunc:
                break
        return actions, env._obs.copy()
    finally:
        env.close()


# ── 1. Replay round-trip ──────────────────────────────────────────────────────

def test_replay_roundtrip(tmp):
    from env import RoboMageEnv

    deck_a, deck_b, paths = _write_decks()
    try:
        actions, final = _record_scripted(deck_a, deck_b, N_RECORD)
        _check(len(actions) == N_RECORD,
               f"fixture game ended early ({len(actions)} actions)")
        path = os.path.join(tmp, "session" + gio.PLAY_EXT)
        gio.save_replay(path, engine_seed=SEED, actions=actions,
                        deck_a=deck_a, deck_b=deck_b, human_deck=deck_a,
                        opp_deck=deck_b, human_is_a=True, bo3=False,
                        opponent_spec="scripted", binary=BINARY,
                        final_obs=final, in_progress=True)
        doc = gio.load_replay(path)
        _check(doc["engine_seed"] == SEED and doc["actions"] == actions
               and doc["history_len"] == len(actions),
               "loaded replay spec differs from saved")
        _check(doc["engine_build"] == gio.engine_build_stamp(),
               "engine_build stamp missing")

        env = RoboMageEnv(binary_path=BINARY, deck_a=doc["deck_a"],
                          deck_b=doc["deck_b"], bo3=doc["bo3"])
        try:
            env.reset(options={"engine_seed": doc["engine_seed"]})
            for a in doc["actions"]:
                env.step(a)
            _check(np.array_equal(env._obs, final),
                   "replayed final obs not byte-identical")
            _check(gio.obs_sha256(env._obs) == doc["final_obs_sha256"],
                   "obs sha tripwire mismatch after replay")
        finally:
            env.close()
        return (f"{len(actions)} actions replayed on a fresh env, final obs "
                "byte-identical, sha match")
    finally:
        _rm(paths)


# ── 2. Trace round-trip ───────────────────────────────────────────────────────

def _synthetic_games():
    rng = np.random.default_rng(5)

    def _obs():
        # Random but float32-exact values.
        return (rng.integers(0, 256, size=OBS_SIZE) / 256.0).astype(np.float32)

    normal = {
        "observations": [_obs(), _obs(), _obs()],
        "values": [0.25, -0.5, 0.75],
        "actions": [0, 1, 2], "num_choices": [2, 3, 4],
        "action_probs": [np.array([0.75, 0.25]), None,
                         np.array([0.25, 0.25, 0.25, 0.25])],
        "opp_actions": [{"desc": "PASS", "before_model_step": 1,
                         "clock": None}],
        "engine_seed": 42, "full_actions": [0, 1, 2, 0, 1],
        "prefix_len": [0, 2, 4], "result": 1.0, "model_is_a": True,
        "clock_remaining": [None, 12.5, 10.25],
        "clock_bank": 30.0, "opp_clock_bank": None,
    }
    shard = {   # shard_replay schema: no probs/replay keys, prefix_len None
        "observations": [_obs(), _obs()],
        "values": [0.0, -0.25],
        "actions": [1, 0], "num_choices": [2, 2],
        "action_probs": None, "opp_actions": [],
        "engine_seed": None, "full_actions": None, "prefix_len": None,
        "result": -1.0, "model_is_a": False, "shard": True,
    }
    whatif = {
        "observations": [_obs()],
        "values": [0.5],
        "actions": [0], "num_choices": [1],
        "action_probs": [np.array([1.0])],
        "opp_actions": [], "engine_seed": 7, "full_actions": [0, 1],
        "prefix_len": [0], "result": 0.0, "model_is_a": True,
        "clock_remaining": [None], "clock_bank": None, "opp_clock_bank": None,
        "whatif": {"src_game": 0, "step": 2},
    }
    return [normal, shard, whatif]


def test_trace_roundtrip(tmp):
    games = _synthetic_games()
    path = os.path.join(tmp, "traces" + gio.TRACE_EXT)
    gio.save_traces(path, games, provenance={"tool": "test"})
    loaded, meta = gio.load_traces(path)
    _check(meta["format"] == gio.TRACE_FORMAT
           and meta["provenance"] == {"tool": "test"}, "meta wrong")
    _check(len(loaded) == 3, f"{len(loaded)} games loaded, want 3")

    for gi, (orig, got) in enumerate(zip(games, loaded)):
        for i, o in enumerate(orig["observations"]):
            _check(np.array_equal(got["observations"][i], o),
                   f"g{gi} obs {i} not byte-equal")
        _check(got["values"] == orig["values"], f"g{gi} values differ")
        _check(got["actions"] == orig["actions"]
               and got["num_choices"] == orig["num_choices"],
               f"g{gi} actions/num_choices differ")
        _check(got["result"] == orig["result"]
               and got["model_is_a"] == orig["model_is_a"],
               f"g{gi} result/seat differ")
        _check(got["engine_seed"] == orig["engine_seed"],
               f"g{gi} engine_seed differs (None round-trip)")
        _check(got["full_actions"] == orig["full_actions"],
               f"g{gi} full_actions differ (None round-trip)")
        _check(got["prefix_len"] == orig["prefix_len"],
               f"g{gi} prefix_len differs (shard None round-trip)")
        _check(got["opp_actions"] == orig["opp_actions"],
               f"g{gi} opp_actions differ")
        _check(got["interp_features"] == [],
               f"g{gi} interp not empty without interp_fn")

    # probs: float64 out, per-step None preserved, values exact.
    gp = loaded[0]["action_probs"]
    _check(gp[1] is None, "None probs step not preserved")
    _check(gp[0].dtype == np.float64 and gp[2].dtype == np.float64,
           "loaded probs not float64")
    _check(np.array_equal(gp[0], [0.75, 0.25])
           and np.array_equal(gp[2], [0.25] * 4), "probs values differ")
    _check(loaded[1]["action_probs"] is None, "shard probs should be None")

    # clock: None entries round-trip; shard's missing key loads as all-None.
    _check(loaded[0]["clock_remaining"] == [None, 12.5, 10.25]
           and loaded[0]["clock_bank"] == 30.0
           and loaded[0]["opp_clock_bank"] is None, "clock round-trip failed")
    _check(loaded[1]["clock_remaining"] == [None, None],
           "shard clock entries not None")

    # markers
    _check(loaded[1].get("shard") is True and "whatif" not in loaded[1],
           "shard marker lost")
    _check(loaded[2]["whatif"] == {"src_game": 0, "step": 2},
           "whatif marker lost")
    _check("shard" not in loaded[0] and "whatif" not in loaded[0],
           "markers leaked onto the normal game")

    # interp_fn recompute hook is applied per obs.
    loaded2, _ = gio.load_traces(path, interp_fn=lambda o: float(o[0]))
    _check(loaded2[0]["interp_features"]
           == [float(o[0]) for o in games[0]["observations"]],
           "interp_fn not applied")
    return "3 games (normal/shard/whatif) round-trip exact incl. None fields"


# ── 3. Validation failures + kind sniffing ────────────────────────────────────

def _expect_raises(fn, exc, needle, what):
    try:
        fn()
    except exc as e:
        if needle and needle not in str(e):
            raise SessionIoTestError(
                f"{what}: message {str(e)!r} lacks {needle!r}")
        return
    raise SessionIoTestError(f"{what}: no {exc.__name__} raised")


def test_validation(tmp):
    # A valid replay doc to corrupt.
    good = os.path.join(tmp, "good" + gio.PLAY_EXT)
    gio.save_replay(good, engine_seed=1, actions=[0, 1], deck_a="a",
                    deck_b="b", human_deck="a", opp_deck="b", human_is_a=True,
                    bo3=False, opponent_spec="scripted",
                    final_obs=np.zeros(OBS_SIZE, dtype=np.float32))
    gio.load_replay(good)   # sanity: the untampered doc loads

    with open(good) as f:
        doc = json.load(f)

    def _dump(d, name):
        p = os.path.join(tmp, name)
        with open(p, "w") as f:
            json.dump(d, f)
        return p

    bad_build = dict(doc, engine_build={"obs_size": 17, "max_actions": 2})
    _expect_raises(lambda: gio.load_replay(_dump(bad_build, "bb.rmplay")),
                   ValueError, "build", "wrong engine_build")
    bad_fmt = dict(doc, format="something-else")
    _expect_raises(lambda: gio.load_replay(_dump(bad_fmt, "bf.rmplay")),
                   ValueError, "not a RoboMage play-session",
                   "wrong format")
    seedless = dict(doc, engine_seed=None)
    _expect_raises(lambda: gio.load_replay(_dump(seedless, "ns.rmplay")),
                   ValueError, "seed", "seedless replay")

    # load_traces on a .rmplay (JSON) file and on garbage must raise.
    _expect_raises(lambda: gio.load_traces(good), Exception, None,
                   "load_traces on a replay file")
    garbage = os.path.join(tmp, "junk.rmtrace")
    with open(garbage, "wb") as f:
        f.write(b"PK\x03\x04truncated-not-a-zip")
    _expect_raises(lambda: gio.load_traces(garbage), Exception, None,
                   "load_traces on truncated garbage")

    # npz without the meta blob is not a session file.
    stray = os.path.join(tmp, "stray.rmtrace")
    with open(stray, "wb") as f:
        np.savez_compressed(f, foo=np.zeros(3))
    _expect_raises(lambda: gio.load_traces(stray), ValueError,
                   "not a RoboMage analysis-session", "meta-less npz")

    # Tampered meta max_actions: re-pack a valid trace file with a bumped
    # value and expect the layout gate to fire.
    tr = os.path.join(tmp, "t" + gio.TRACE_EXT)
    gio.save_traces(tr, _synthetic_games())
    with np.load(tr, allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    meta = json.loads(str(data["meta"]))
    meta["max_actions"] = MAX_ACTIONS + 1
    data["meta"] = np.array(json.dumps(meta))
    tampered = os.path.join(tmp, "tampered" + gio.TRACE_EXT)
    with open(tampered, "wb") as f:
        np.savez_compressed(f, **data)
    _expect_raises(lambda: gio.load_traces(tampered), ValueError,
                   "layout mismatch", "tampered max_actions")

    # sniff_kind: extension first...
    _check(gio.sniff_kind("x" + gio.PLAY_EXT) == "play"
           and gio.sniff_kind("x" + gio.TRACE_EXT) == "trace",
           "extension sniffing wrong")
    # ...content fallback for both kinds...
    as_json = _dump(doc, "renamed.json")
    _check(gio.sniff_kind(as_json) == "play", "content sniff missed a replay")
    renamed_tr = os.path.join(tmp, "renamed.bin")
    gio.save_traces(renamed_tr, _synthetic_games()[:1])
    _check(gio.sniff_kind(renamed_tr) == "trace",
           "content sniff missed a trace (PK header)")
    # ...and None for foreign/missing files.
    foreign = _dump({"format": "unrelated"}, "foreign.json")
    _check(gio.sniff_kind(foreign) is None, "foreign JSON not rejected")
    _check(gio.sniff_kind(os.path.join(tmp, "missing.file")) is None,
           "missing file not None")
    return "build/format/seed/layout gates fire; sniffing routes both kinds"


# ── 4. trace_from_replay ──────────────────────────────────────────────────────

def test_trace_from_replay(tmp):
    deck_a, deck_b, paths = _write_decks()
    try:
        actions, final = _record_scripted(deck_a, deck_b, 40)
        path = os.path.join(tmp, "tfr" + gio.PLAY_EXT)
        gio.save_replay(path, engine_seed=SEED, actions=actions,
                        deck_a=deck_a, deck_b=deck_b, human_deck=deck_a,
                        opp_deck=deck_b, human_is_a=True, bo3=False,
                        opponent_spec="scripted", final_obs=final,
                        in_progress=True)
        doc = gio.load_replay(path)
        trace = gio.trace_from_replay(doc, BINARY)

        _check(trace["full_actions"] == actions,
               "trace full_actions != replayed prefix")
        n = len(trace["observations"])
        _check(n >= 1, "no model steps recorded")
        _check(len(trace["actions"]) == n and len(trace["values"]) == n
               and len(trace["num_choices"]) == n
               and len(trace["prefix_len"]) == n,
               "per-step lists ragged")
        _check(isinstance(trace["result"], float), "result not a float")
        _check(trace["model_is_a"] == doc["human_is_a"],
               "viewpoint seat != saved human seat")
        _check(all(v == 0.0 for v in trace["values"]),
               "values not 0.0 without a value_fn")
        _check(trace["prefix_len"] == sorted(trace["prefix_len"])
               and trace["prefix_len"][-1] < len(actions),
               "prefix_len not an increasing prefix map")
        _check(all(0 <= oa["before_model_step"] <= n
                   for oa in trace["opp_actions"]),
               "opp_actions step index out of range")
        _check(n + len(trace["opp_actions"]) == len(actions),
               "model + opp decisions != total actions")

        # The trace saves and reloads like any collected game.
        tr = os.path.join(tmp, "converted" + gio.TRACE_EXT)
        gio.save_traces(tr, [trace])
        loaded, _ = gio.load_traces(tr)
        _check(loaded[0]["full_actions"] == actions
               and loaded[0]["engine_seed"] == SEED,
               "converted trace did not round-trip")
        return (f"{len(actions)}-action prefix -> {n} model steps + "
                f"{len(trace['opp_actions'])} opp actions, round-trips")
    finally:
        _rm(paths)


TESTS = [
    ("replay_roundtrip", test_replay_roundtrip),
    ("trace_roundtrip", test_trace_roundtrip),
    ("validation", test_validation),
    ("trace_from_replay", test_trace_from_replay),
]


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found at {BINARY} — run `make` first", file=sys.stderr)
        return 2
    failures = 0
    with tempfile.TemporaryDirectory(prefix="rm_session_io_") as tmp:
        for name, fn in TESTS:
            try:
                detail = fn(tmp)
                print(f"ok    {name}: {detail}", flush=True)
            except (SessionIoTestError, AssertionError, EOFError,
                    RuntimeError) as e:
                failures += 1
                print(f"FAIL  {name}: {e}", flush=True)
    print(f"\ngui session io: {len(TESTS) - failures}/{len(TESTS)} tests passed",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
