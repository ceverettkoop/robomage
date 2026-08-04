#!/usr/bin/env python3
"""Qt-free analysis-browser core regression (browse_session.py).

Proves the guarantees both trace-browser front ends rely on:

  1. histogram geometry — ``HistogramModel`` lays out columns with the SAME
     algorithm as tui_analysis's ValueHistogram (unbucketed widened bars +
     click-mapped spacers when n <= w; max-|V| bucket representatives when
     n > w; cursor columns in both modes; vmax = max(1, max|V|), growing but
     never shrinking under append()).
  2. BrowseStore event application — a GameStarted placeholder grows under
     StepAppended/OppActionAppended, is wholesale-replaced by GameFinished,
     and is removed (with cursor fixup) by GameAborted; GameAdded appends.
  3. presentation helpers — game_label/summary_line/analysis_pool exclude
     live+whatif games correctly; decision_data sorts rows most-probable-first
     with the chosen marker; opp_actions_before collapses runs; clock_line
     formats both seats' banks.
  4. live streaming end-to-end — a real engine game collected through
     EngineCore (preloaded test seam) + analysis._collect_game_traces streams
     the documented event order (one GameStarted, steps with the full step
     record, a GameFinished carrying the authoritative dict, EngineIdle last),
     and a should_stop after the 3rd step yields GameAborted with no
     GameFinished. The same events replayed through BrowseStore leave it
     consistent. This leg needs torch (self-skips without it) + bin/robomage.

Runnable standalone::

    train/.venv/bin/python train/test_browse_session.py
"""
import os
import sys
import threading
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import browse_session as bs  # noqa: E402
from cli_spec import BINARY, BIN_DIR  # noqa: E402
from env import OBS_SIZE  # noqa: E402

SEED = 11          # gymnasium seed for the live-collect env's engine seeds
COIN_SEED = 7      # np.random seed for the collector's model_is_a coin flips


class BrowseTestError(AssertionError):
    pass


def _check(cond, msg):
    if not cond:
        raise BrowseTestError(msg)


# ── 1. HistogramModel geometry ────────────────────────────────────────────────

def _expected_layout(values, w):
    """Reference port of tui_analysis.ValueHistogram._layout_cols."""
    n = len(values)
    cols, bar_w, bucketed = [], 1, False
    if not n or w <= 0:
        return cols, bar_w, bucketed
    if n <= w:
        per = max(1, min(4, w // n))
        bar_w = per - 1 if per >= 2 else 1
        for s, v in enumerate(values):
            cols.extend([(v, s)] * bar_w)
            if per >= 2:
                cols.append((None, s))
    else:
        bucketed = True
        for c in range(w):
            lo = c * n // w
            hi = max(lo + 1, (c + 1) * n // w)
            rep = max(range(lo, hi), key=lambda s: abs(values[s]))
            cols.append((values[rep], rep))
    return cols, bar_w, bucketed


def _expected_cursor_cols(values, cols, bar_w, bucketed, cursor):
    """Reference port of tui_analysis.ValueHistogram._cursor_cols."""
    n = len(values)
    if not n or not cols:
        return set()
    if bucketed:
        return {min(cursor * len(cols) // n, len(cols) - 1)}
    per = bar_w + (1 if len(cols) > n * bar_w else 0)
    start = cursor * per
    return set(range(start, min(start + bar_w, len(cols))))


def test_histogram_layout():
    cases = [
        ("widened", [0.1, -0.4, 0.9, -0.2, 0.6], 20),        # per=4, spacers
        ("exact", [(-1) ** i * (i / 10.0) for i in range(10)], 10),  # per=1
        ("narrow-spacer", [0.5, -0.5, 0.25, -0.25], 9),      # per=2, bar_w=1
        ("bucketed", [np.sin(i / 3.0) * (1 + (i % 7) / 7.0)
                      for i in range(100)], 12),
    ]
    hm = bs.HistogramModel()
    for name, values, w in cases:
        hm.set_data(values, cursor=0)
        hm.layout(w)
        exp_cols, exp_bar_w, exp_bucketed = _expected_layout(
            [float(v) for v in values], w)
        _check(hm.cols == exp_cols, f"{name}: cols diverge from reference")
        _check(hm.bar_w == exp_bar_w, f"{name}: bar_w {hm.bar_w} != {exp_bar_w}")
        _check(hm.bucketed == exp_bucketed, f"{name}: bucketed flag diverges")
        _check(hm.vmax == max(1.0, max(abs(float(v)) for v in values)),
               f"{name}: vmax {hm.vmax} wrong")
        # step_at maps every column (spacers included) to its step; out of
        # range is None.
        for c, (_, s) in enumerate(hm.cols):
            _check(hm.step_at(c) == s, f"{name}: step_at({c}) != {s}")
        _check(hm.step_at(-1) is None and hm.step_at(len(hm.cols)) is None,
               f"{name}: out-of-range step_at not None")
        # cursor_cols parity at a few cursor positions.
        for cur in (0, len(values) // 2, len(values) - 1):
            hm.set_cursor(cur)
            exp = _expected_cursor_cols([float(v) for v in values], exp_cols,
                                        exp_bar_w, exp_bucketed, cur)
            _check(hm.cursor_cols() == exp,
                   f"{name}: cursor_cols({cur}) {hm.cursor_cols()} != {exp}")

    # Spot-check the widened case's concrete shape: 5 steps x (3 bars + spacer).
    hm.set_data(cases[0][1])
    hm.layout(20)
    _check(len(hm.cols) == 20 and hm.bar_w == 3 and not hm.bucketed,
           "widened case shape wrong")
    _check(hm.cols[3] == (None, 0), "spacer column missing or mismapped")
    hm.set_cursor(2)
    _check(hm.cursor_cols() == {8, 9, 10}, "widened cursor_cols wrong")

    # vmax grows under append, never shrinks; set_data resets it.
    hm.set_data([0.2, -1.5])
    _check(hm.vmax == 1.5, "vmax not max|V|")
    hm.append(-2.0)
    _check(hm.vmax == 2.0, "append did not grow vmax")
    hm.append(0.5)
    _check(hm.vmax == 2.0 and len(hm.values) == 4, "vmax shrank on append")
    hm.set_data([0.1])
    _check(hm.vmax == 1.0, "set_data did not reset vmax floor to 1")
    # Cursor clamping.
    hm.set_data([0.1, 0.2, 0.3], cursor=99)
    _check(hm.cursor == 2, "set_data cursor not clamped")
    hm.set_cursor(-5)
    _check(hm.cursor == 0, "set_cursor not clamped to 0")
    return "4 layouts bit-equal to ValueHistogram reference, vmax/cursor ok"


# ── 2. BrowseStore event sequences ────────────────────────────────────────────

def _mkstep(i):
    return {"obs": np.zeros(8, dtype=np.float32), "value": 0.1 * i,
            "interp": None, "num_choices": 2, "probs": None, "clock": None,
            "prefix_len": i, "action": i % 2}


def _finished(n_steps, result=1.0, model_is_a=True):
    return {"observations": [np.zeros(OBS_SIZE, dtype=np.float32)] * n_steps,
            "values": [0.0] * n_steps, "interp_features": [None] * n_steps,
            "actions": [0] * n_steps, "num_choices": [2] * n_steps,
            "action_probs": None, "opp_actions": [], "engine_seed": 5,
            "full_actions": list(range(2 * n_steps)),
            "prefix_len": list(range(n_steps)), "result": result,
            "model_is_a": model_is_a, "clock_remaining": [None] * n_steps,
            "clock_bank": None, "opp_clock_bank": None}


def test_store_live_sequence():
    st = bs.BrowseStore()
    _check(st.engine_busy, "store does not start engine_busy")
    ap = st.apply(bs.GameStarted(model_is_a=True, engine_seed=99))
    _check(st.live_idx == 0 and len(st.games) == 1 and ap.game_idx == 0,
           "GameStarted did not open a live placeholder")
    g = st.games[0]
    _check(g["live"] and g["result"] is None and g["full_actions"] is None
           and g["engine_seed"] == 99, "placeholder schema wrong")
    _check(bs.result_str(g) == "LIVE", "placeholder result_str != LIVE")

    st.select_game(0)
    for i in range(3):
        ap = st.apply(bs.StepAppended(step=_mkstep(i)))
        _check(ap.game_idx == 0 and ap.selected_grew,
               "StepAppended on selected live game did not flag selected_grew")
    _check(len(g["observations"]) == 3 and g["actions"] == [0, 1, 0]
           and g["values"] == [0.0, 0.1, 0.2] and g["prefix_len"] == [0, 1, 2],
           "placeholder per-step lists did not grow correctly")

    opp = {"desc": "PASS", "before_model_step": 3, "clock": None}
    ap = st.apply(bs.OppActionAppended(opp=opp))
    _check(g["opp_actions"] == [opp] and ap.game_idx == 0,
           "OppActionAppended did not land in opp_actions")

    fin = _finished(3)
    ap = st.apply(bs.GameFinished(game=fin))
    _check(st.games[0] is fin and st.live_idx is None and ap.game_idx == 0,
           "GameFinished did not replace the placeholder wholesale")

    # Steps with no live game open are no-ops.
    ap = st.apply(bs.StepAppended(step=_mkstep(9)))
    _check(ap.game_idx is None and len(st.games[0]["observations"]) == 3,
           "StepAppended without a live game mutated the store")

    st.apply(bs.EngineIdle())
    _check(not st.engine_busy, "EngineIdle did not clear engine_busy")

    # GameFinished with no placeholder appends defensively.
    ap = st.apply(bs.GameFinished(game=_finished(1, result=-1.0)))
    _check(len(st.games) == 2 and ap.game_idx == 1,
           "defensive GameFinished did not append")
    return "placeholder grew 3 steps + opp, replaced wholesale, idle ok"


def test_store_abort():
    # (a) cursor ON the aborted live game -> deselected.
    st = bs.BrowseStore()
    st.apply(bs.GameAdded(game=_finished(2)))
    st.apply(bs.GameStarted(model_is_a=True, engine_seed=1))
    st.apply(bs.StepAppended(step=_mkstep(0)))
    st.select_game(1)
    ap = st.apply(bs.GameAborted())
    _check(ap.removed and ap.game_idx == 1, "abort Applied flags wrong")
    _check(len(st.games) == 1 and st.live_idx is None,
           "abort did not remove the placeholder")
    _check(st.cur_game is None and st.cur_step == 0,
           "abort of the selected game did not deselect")

    # (b) cursor PAST the aborted index -> decremented. (GameAdded can land
    # after the live placeholder, so cur_game > live_idx is reachable.)
    st = bs.BrowseStore()
    st.apply(bs.GameStarted(model_is_a=False, engine_seed=2))
    st.apply(bs.GameAdded(game=_finished(2)))
    st.select_game(1)
    ap = st.apply(bs.GameAborted())
    _check(st.cur_game == 0 and len(st.games) == 1 and ap.removed,
           "abort before the cursor did not decrement cur_game")

    # (c) abort with no live game is a no-op.
    ap = st.apply(bs.GameAborted())
    _check(not ap.removed and len(st.games) == 1,
           "abort without a live game mutated the store")
    return "abort removed placeholder; cursor deselected / decremented"


# ── 3. Presentation helpers ───────────────────────────────────────────────────

def test_labels_and_pool():
    win = _finished(5, result=1.0, model_is_a=True)
    loss = _finished(4, result=-1.0, model_is_a=False)
    shard = _finished(3, result=-1.0)
    shard["shard"] = True
    whatif = _finished(2, result=0.0)
    whatif["whatif"] = {"src_game": 2, "step": 17}
    live = bs._live_placeholder(True, 5)
    games = [win, loss, shard, whatif, live]

    _check(bs.analysis_pool(games) == [win, loss, shard],
           "analysis_pool did not exclude live+whatif")
    line = bs.summary_line(games)
    _check("3 games" in line and "1W/2L/0D" in line and "+1 whatif" in line
           and "1 live" in line, f"summary_line wrong: {line!r}")
    _check("simulating" in bs.summary_line(games, loading=True),
           "loading marker missing")

    _check("WIN" in bs.game_label(0, win) and bs.game_label(0, win).endswith("A"),
           "win label wrong")
    _check("LOSS" in bs.game_label(1, loss) and bs.game_label(1, loss).endswith("B"),
           "loss label wrong (seat)")
    _check(bs.game_label(2, shard).endswith("⛁"), "shard label missing ⛁")
    _check("↳g2@17" in bs.game_label(3, whatif), "whatif label missing origin")
    llab = bs.game_label(4, live)
    _check("LIVE" in llab and llab.endswith("…"), "live label wrong")
    _check(bs.result_str(whatif) == "DRAW" and bs.result_str(loss) == "LOSS",
           "result_str wrong")
    return "labels (LIVE/⛁/↳g2@17), pool and summary correct"


def test_decision_data():
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    game = {"observations": [obs, obs], "values": [0.25, 0.5],
            "num_choices": [3, 2], "actions": [1, 0],
            "action_probs": [np.array([0.2, 0.5, 0.3]), None],
            "opp_actions": [
                {"desc": "PASS", "before_model_step": 0, "clock": None},
                {"desc": "PASS", "before_model_step": 0, "clock": None},
                {"desc": "PASS", "before_model_step": 0, "clock": None},
                {"desc": "Cast Bolt", "before_model_step": 0, "clock": None},
                {"desc": "PASS", "before_model_step": 1, "clock": None}],
            "clock_remaining": [None, None],
            "clock_bank": None, "opp_clock_bank": None}
    dd = bs.decision_data(game, 0)
    _check(dd.num_choices == 3 and len(dd.rows) == 3, "row count wrong")
    _check([r.k for r in dd.rows] == [1, 2, 0],
           f"rows not sorted most-probable-first: {[r.k for r in dd.rows]}")
    _check(dd.rows[0].is_chosen and dd.rows[0].prob == 0.5,
           "chosen marker / prob wrong")
    _check(all(isinstance(r.desc, str) for r in dd.rows),
           "zero-obs action decode failed")
    _check(dd.opp_lines == ["PASS (x3)", "Cast Bolt"],
           f"opp run-collapse wrong: {dd.opp_lines}")
    _check(dd.clock == "" and not dd.shard_caveat, "clock/shard flags wrong")

    # Step without recorded probs: natural order, prob None.
    dd1 = bs.decision_data(game, 1)
    _check([r.k for r in dd1.rows] == [0, 1], "probless rows reordered")
    _check(dd1.opp_lines == ["PASS"], "single opp action wrong")

    # Clock line: both seats banked; opp remaining is the last reading at or
    # before the step.
    cg = {"clock_bank": 30.0, "clock_remaining": [25.5],
          "opp_clock_bank": 20.0,
          "opp_actions": [{"desc": "x", "before_model_step": 0, "clock": 18.0}]}
    line = bs.clock_line(cg, 0)
    _check("model 25.5s / 30s" in line and "opp 18.0s / 20s" in line,
           f"clock_line wrong: {line!r}")
    _check("?" in bs.clock_line({"clock_bank": 30.0, "clock_remaining": [],
                                 "opp_clock_bank": None, "opp_actions": []}, 0),
           "missing model reading should render '?'")
    return "rows sorted w/ chosen marker, 'PASS (x3)' collapse, clocks ok"


# ── 4. Live streaming through the real collector ──────────────────────────────

def _write_decks():
    """Fast lopsided matchup (same fixture as test_analysis_session)."""
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "browse_test_a.dk")
    b = os.path.join(d, "browse_test_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\n")
    with open(b, "w") as f:
        f.write("60 Swamp\n")
    return "temp/browse_test_a", "temp/browse_test_b", [a, b]


def _rm(paths):
    for p in paths:
        try:
            os.remove(p)
        except OSError:
            pass


def _make_stub_model(torch):
    """The minimal surface _collect_game_traces + ModelController touch:
    policy.predict_values / policy.get_distribution / predict. Plays via the
    scripted HARD agent so games are real and finish."""
    from scripted_agent import make_agent

    class _Obj:
        pass

    class _Policy:
        def predict_values(self, obs_t):
            return torch.tanh(obs_t.sum(dim=1, keepdim=True) / 5000.0)

        def get_distribution(self, obs_t, action_masks=None):
            m = torch.as_tensor(np.asarray(action_masks), dtype=torch.float32)
            probs = m / m.sum(dim=1, keepdim=True)
            dist = _Obj()
            dist.distribution = _Obj()
            dist.distribution.probs = probs
            return dist

    class _Model:
        def __init__(self):
            self.policy = _Policy()
            self._agent = make_agent("scripted")

        def predict(self, obs, action_masks=None, deterministic=False):
            num = int(np.count_nonzero(action_masks))
            return self._agent.act(obs, num), None

    return _Model()


def test_live_collect_stream():
    try:
        import torch
    except ImportError:
        return "SKIP — torch not installed (live-stream leg needs it)"
    from env import RoboMageEnv

    deck_a, deck_b, paths = _write_decks()
    try:
        env = RoboMageEnv(binary_path=BINARY, deck_a=deck_a, deck_b=deck_b)
        model = _make_stub_model(torch)
        try:
            env.reset(seed=SEED)           # seed the env's engine-seed stream
            np.random.seed(COIN_SEED)      # the collector's model_is_a coin

            # --- happy path: one full game streamed through EngineCore ---
            events = []
            core = bs.EngineCore(SimpleNamespace(), events.append,
                                 preloaded=(model, env, None))
            core.collect(1)

            _check(isinstance(events[-1], bs.EngineIdle),
                   "last event is not EngineIdle")
            starts = [e for e in events if isinstance(e, bs.GameStarted)]
            steps = [e for e in events if isinstance(e, bs.StepAppended)]
            fins = [e for e in events if isinstance(e, bs.GameFinished)]
            aborts = [e for e in events if isinstance(e, bs.GameAborted)]
            _check(len(starts) == 1, f"{len(starts)} GameStarted, want 1")
            _check(len(steps) >= 1, "no StepAppended events")
            _check(len(fins) == 1 and not aborts, "want 1 GameFinished, 0 aborts")
            _check(events.index(starts[0]) < events.index(steps[0])
                   < events.index(fins[0]),
                   "event order start < step < end violated")
            _check(starts[0].engine_seed is not None, "GameStarted seed missing")
            want_keys = {"obs", "value", "interp", "num_choices", "probs",
                         "clock", "prefix_len", "action"}
            for e in steps:
                _check(want_keys <= set(e.step), f"step keys {set(e.step)}")
                _check(len(e.step["obs"]) == OBS_SIZE, "step obs width wrong")
                p = e.step["probs"]
                _check(len(p) == e.step["num_choices"]
                       and abs(float(np.sum(p)) - 1.0) < 1e-5,
                       "step probs malformed")
            fin = fins[0].game
            _check(fin["result"] is not None and fin["full_actions"] is not None,
                   "finished game missing result/full_actions")
            _check(len(fin["observations"]) == len(steps),
                   "finished game step count != streamed steps")
            _check(fin["engine_seed"] == starts[0].engine_seed,
                   "finished seed != GameStarted seed")

            # Same events through a BrowseStore.
            st = bs.BrowseStore()
            for ev in events:
                if isinstance(ev, bs.GameFinished):
                    _check(len(st.games[0]["observations"]) == len(steps),
                           "placeholder had not grown to all streamed steps")
                st.apply(ev)
            _check(len(st.games) == 1 and st.games[0] is fin
                   and st.live_idx is None and not st.engine_busy,
                   "store inconsistent after full stream")

            # --- abort path: stop after the 3rd step event ---
            events2 = []
            stop = threading.Event()

            def emit2(ev):
                events2.append(ev)
                if (isinstance(ev, bs.StepAppended)
                        and sum(isinstance(x, bs.StepAppended)
                                for x in events2) == 3):
                    stop.set()

            core2 = bs.EngineCore(SimpleNamespace(), emit2,
                                  preloaded=(model, env, None))
            core2.collect(1, stop)
            _check(any(isinstance(e, bs.GameAborted) for e in events2),
                   "no GameAborted after should_stop")
            _check(not any(isinstance(e, bs.GameFinished) for e in events2),
                   "GameFinished emitted despite abort")
            _check(isinstance(events2[-1], bs.EngineIdle),
                   "abort run did not end EngineIdle")
            n_steps2 = sum(isinstance(e, bs.StepAppended) for e in events2)
            _check(n_steps2 == 3, f"{n_steps2} steps streamed before abort, want 3")

            st2 = bs.BrowseStore()
            for ev in events2:
                st2.apply(ev)
            _check(len(st2.games) == 0 and st2.live_idx is None,
                   "aborted placeholder not removed from store")
            return (f"1 game streamed ({len(steps)} steps) + abort after 3 "
                    "steps, store consistent")
        finally:
            env.close()
    finally:
        _rm(paths)


TESTS = [
    ("histogram_layout", test_histogram_layout),
    ("store_live_sequence", test_store_live_sequence),
    ("store_abort", test_store_abort),
    ("labels_and_pool", test_labels_and_pool),
    ("decision_data", test_decision_data),
    ("live_collect_stream", test_live_collect_stream),
]


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found at {BINARY} — run `make` first", file=sys.stderr)
        return 2
    failures = 0
    for name, fn in TESTS:
        try:
            detail = fn()
            print(f"ok    {name}: {detail}", flush=True)
        except (BrowseTestError, AssertionError, EOFError, RuntimeError) as e:
            failures += 1
            print(f"FAIL  {name}: {e}", flush=True)
    print(f"\nbrowse session: {len(TESTS) - failures}/{len(TESTS)} tests passed",
          flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
