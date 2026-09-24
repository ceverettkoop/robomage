#!/usr/bin/env python3
"""Textual analysis-browser regression (tui_analysis.AnalysisApp).

Drives the terminal board headlessly (Textual's ``run_test`` pilot) over a
synthetic saved session and proves the capabilities it shares with the GUI
pane (gui_browser.BrowserPane) are wired to browse_session:

  1. .rmtrace source — a saved session (``bs.save_session``) loads through
     the shared EngineCore/EngineWorker into the games list, the cursor
     steps, and the opened session's provenance rides on EnvReady.
  2. save — ctrl+s opens the path prompt; submitting it writes a .rmtrace
     that ``gui_session_io.load_traces`` reads back with the same games and
     the opened session's provenance (save → load round trip).
  3. net probes — a PROBE_MENU entry snapshots the browsed games and runs
     ``shard_probes.run_probe`` on the analysis worker (net loader + probe
     stubbed: this pins the glue, az_inspect's probes have their own tests),
     output landing in the Analysis-output log.
  4. tree walk — F7 refuses outside a recording; a TreeReady installs the
     per-world root rows + world picker, expanding a node submits a
     ``tree_expand`` job, the TreeNodes reply populates its children, and
     selecting the node renders its walked hypothetical board as text
     (opponent's hand hidden unless revealed).
  5. live streaming — GameStarted/StepAppended/GameFinished events grow a
     LIVE games-list row that follow mode rides, then finalize it.
  6. views — the VIEWS entries are in the analyses menu; the transcript
     view prints the selected game into the Analysis-output log and a chart
     view saves its PNG (redirected to a temp dir) and logs the path.

Needs bin/robomage (one real engine observation is the fixture); torch-free.

Runnable standalone::

    train/.venv/bin/python train/test_tui_browser.py
"""
import argparse
import asyncio
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import browse_session as bs  # noqa: E402
import gui_session_io  # noqa: E402
import shard_probes  # noqa: E402
from cli_spec import BINARY, BIN_DIR  # noqa: E402
from env import RoboMageEnv, _SELF_IS_A_IDX  # noqa: E402
from textual.widgets import Checkbox, Input, RichLog, Select, Tree  # noqa: E402

import tui_analysis  # noqa: E402
import viz  # noqa: E402

_PROV = {"player_a": "gen", "player_b": "scripted",
         "deck_a": "temp/tui_browser_a", "deck_b": "temp/tui_browser_b",
         "format": "bo1"}


class TuiBrowserError(AssertionError):
    pass


def _check(cond, msg):
    if not cond:
        raise TuiBrowserError(msg)


def _write_decks():
    d = os.path.join(BIN_DIR, "resources", "decks", "temp")
    os.makedirs(d, exist_ok=True)
    a = os.path.join(d, "tui_browser_a.dk")
    b = os.path.join(d, "tui_browser_b.dk")
    with open(a, "w") as f:
        f.write("36 Grizzly Bears\n24 Forest\n")
    with open(b, "w") as f:
        f.write("60 Swamp\n")
    return [a, b]


def _real_obs():
    """One real engine observation (the opening mulligan decision)."""
    env = RoboMageEnv(binary_path=BINARY, deck_a=_PROV["deck_a"],
                      deck_b=_PROV["deck_b"])
    try:
        obs, _info = env.reset(seed=3)
        return np.asarray(obs, dtype=np.float32), int(env._num_choices)
    finally:
        env.close()


def _game(obs, num, n_steps, result, model_is_a=True):
    probs = np.full(num, 1.0 / num)
    return {"observations": [obs.copy() for _ in range(n_steps)],
            "values": [0.1 * i * result for i in range(n_steps)],
            "interp_features": [], "actions": [0] * n_steps,
            "num_choices": [num] * n_steps,
            "action_probs": [probs.copy() for _ in range(n_steps)],
            "opp_actions": [{"before_model_step": 1, "desc": "PASS"}],
            "engine_seed": 7, "full_actions": list(range(2 * n_steps)),
            "prefix_len": list(range(n_steps)), "result": result,
            "model_is_a": model_is_a, "clock_remaining": [None] * n_steps,
            "clock_bank": None, "opp_clock_bank": None}


def _args(source):
    ns = argparse.Namespace(**bs.BROWSE_ARG_DEFAULTS)
    ns.source = source
    ns.player_a = "gen"
    return ns


def _log_text(app):
    log = app.query_one("#output", RichLog)
    return "\n".join("".join(seg.text for seg in line) for line in log.lines)


async def _wait(pilot, cond, what, timeout=15.0):
    t = 0.0
    while not cond():
        if t >= timeout:
            raise TuiBrowserError(f"timed out waiting for {what}")
        await pilot.pause(0.05)
        t += 0.05


async def _drive(trace_path, save_path, obs, num):
    app = tui_analysis.AnalysisApp(_args(trace_path))
    results = []
    async with app.run_test(size=(160, 60)) as pilot:
        st = app._store
        # 1. .rmtrace source through the shared engine worker.
        await _wait(pilot, lambda: len(st.games) == 2 and not st.engine_busy,
                    "the saved session to load")
        await _wait(pilot, lambda: app.query_one("#games").option_count == 2,
                    "the games list to fill")
        _check(app._loaded_provenance is not None
               and app._loaded_provenance.get("deck_a") == _PROV["deck_a"],
               f"opened provenance not captured: {app._loaded_provenance}")
        _check(st.cur_game == 0, "first game not auto-selected")
        await pilot.press("right")
        _check(st.cur_step == 1, f"right did not step (cur_step {st.cur_step})")
        results.append("trace source loads + steps")

        # 2. ctrl+s → prompt → .rmtrace round trip.
        await pilot.press("ctrl+s")
        await _wait(pilot, lambda: isinstance(app.screen, tui_analysis.SavePrompt),
                    "the save prompt")
        app.screen.query_one("#save-path", Input).value = save_path
        await pilot.press("enter")
        await _wait(pilot, lambda: os.path.exists(save_path), "the saved file")
        games, meta = gui_session_io.load_traces(save_path)
        _check(len(games) == 2 and len(games[1]["observations"]) == 4,
               "saved session games differ")
        _check((meta.get("provenance") or {}).get("deck_b") == _PROV["deck_b"],
               "saved provenance is not the opened session's")
        results.append("ctrl+s save → load round trip")

        # 3. net probe glue (loader + probe stubbed).
        calls = []
        real_load, real_run = shard_probes.load_probe_net, shard_probes.run_probe
        shard_probes.load_probe_net = lambda spec: ("NET", f"stub {spec}")
        shard_probes.run_probe = lambda key, net, snap: (
            calls.append((key, net, snap["sel"])) or [f"stub probe {key}"])
        try:
            app._run_menu_entry("probe_state")
            await _wait(pilot, lambda: calls and not st.analysis_busy,
                        "the probe to run")
        finally:
            shard_probes.load_probe_net, shard_probes.run_probe = \
                real_load, real_run
        _check(calls[0] == ("probe_state", "NET", (0, 1)),
               f"probe got {calls[0]}")
        _check("stub probe probe_state" in _log_text(app)
               and "probe net: stub gen" in _log_text(app),
               "probe output missing from the log")
        results.append("probe menu → shard_probes on the analysis worker")

        # 6. views: transcript + a saved chart on the selected game.
        menu_ids = {app.query_one("#analyses").get_option_at_index(i).id
                    for i in range(app.query_one("#analyses").option_count)}
        _check({k for k, *_ in bs.VIEWS} <= menu_ids,
               "VIEWS entries missing from the analyses menu")
        app._run_menu_entry("transcript_full")
        await _wait(pilot, lambda: "Game 0" in _log_text(app)
                    and not st.analysis_busy, "the transcript")
        _check("BF self" in _log_text(app), "full transcript lacks the zones")
        chart_dir = os.path.join(os.path.dirname(save_path), "charts")
        real_out = viz._DEFAULT_OUT
        viz._DEFAULT_OUT = chart_dir
        try:
            app._run_menu_entry("chart_game")
            png = os.path.join(chart_dir, "game0.png")
            await _wait(pilot, lambda: f"[chart] saved {png}" in _log_text(app)
                        and not st.analysis_busy, "the chart")
        finally:
            viz._DEFAULT_OUT = real_out
        _check(os.path.exists(png), f"{png} not written")
        results.append("views: transcript + chart_game PNG in the output log")

        # 4. tree walk glue.
        await pilot.press("f7")
        _check(not st.engine_busy, "F7 submitted a tree job on a .rmtrace")
        submitted = []
        real_submit = app._worker.submit
        app._worker.submit = submitted.append
        try:
            app._tree_open = True
            ready = bs.TreeReady(
                gn=0, step=1, root_step=1, summary="verified: stub",
                verified=True, from_cache=False, mismatch=None, worlds=2,
                root_labels=["Keep", "Mulligan"],
                root_rows=[[(0, 6, 0.25, 0.7), (1, 2, -0.1, 0.3)],
                           [(0, 5, 0.2, 0.6), (1, 3, 0.0, 0.4)]],
                merged_rows=[(0, 11, 0.23, 0.65), (1, 5, -0.04, 0.35)],
                pv_lines=[{0: "Keep → #1 (6, +0.250)", 1: "Mulligan (2, -0.100)"},
                          {0: "Keep (5, +0.200)", 1: "Mulligan (3, +0.000)"}])
            app.post_message(tui_analysis.EngineEvent(ready))
            await _wait(pilot, lambda: app.query_one(
                "#tabs").active == "tab-tree", "the Tree tab")
            tree = app.query_one("#tree-view", Tree)
            _check(len(tree.root.children) == 2
                   and tree.root.children[0].data == (0, (0,)),
                   "world-0 root rows not installed most-visited first")
            _check(app.query_one("#tree-world", Select).value == 0,
                   "world picker not on world 0")
            pane = app.query_one("#tree-pane", tui_analysis.TreePane)
            pane.request_expand(0, [0])
            await _wait(pilot, lambda: submitted, "the tree_expand job")
            _check(submitted[0] == ("tree_expand", 0, [0]),
                   f"expand submitted {submitted[0]}")
            # The walked node: the mover's own obs, and the same position
            # from the other seat (mirrored → hand hidden).
            other = obs.copy()
            other[_SELF_IS_A_IDX] = 1.0 - other[_SELF_IS_A_IDX]
            walk = [SimpleNamespace(obs=other, terminal=None)]
            app.post_message(tui_analysis.EngineEvent(bs.TreeNodes(
                0, [0], [(0, 4, 0.3, 0.5), (1, 2, 0.1, 0.5)],
                ["Pass", "Other"], walk, None)))
            app.post_message(tui_analysis.EngineEvent(bs.EngineIdle()))
            node = tree.root.children[0]
            await _wait(pilot, lambda: len(node.children) == 2,
                        "the expanded node's children")
            tree.move_cursor(node)
            tree.select_node(node)
            await pilot.pause(0.1)
            board = str(app.query_one("#tree-board").render())
            _check("YOU ♥" in board, f"walked board not rendered: {board!r}")
            _check("PV: Keep" in str(app.query_one("#tree-pv").render()),
                   "root action's PV not shown")
            model_is_a = bool(obs[_SELF_IS_A_IDX] > 0.5)
            want_hidden = "hand hidden" in "\n".join(
                bs.walk_board_lines(other, not model_is_a))
            _check(("hand hidden" in board) == want_hidden,
                   "hidden-hand rule differs from walk_board_lines")
            if want_hidden:
                app.query_one("#tree-reveal", Checkbox).value = True
                await pilot.pause(0.1)
                _check("hand hidden" not in str(
                    app.query_one("#tree-board").render()),
                    "reveal did not show the opponent's hand")
            results.append("tree pane: roots, expand job, walked text board")
        finally:
            app._worker.submit = real_submit
            app._tree_open = False

        # 5. live streaming events → LIVE row, follow, finalize.
        app._follow = True
        app.post_message(tui_analysis.EngineEvent(bs.GameStarted(True, 9)))
        for i in range(3):
            app.post_message(tui_analysis.EngineEvent(bs.StepAppended(
                {"obs": obs, "value": 0.1 * i, "interp": None,
                 "num_choices": num, "probs": None, "clock": None,
                 "prefix_len": i, "action": 0})))
        await _wait(pilot, lambda: st.live_idx == 2
                    and len(st.games[2]["observations"]) == 3
                    and app.query_one("#games").option_count == 3,
                    "the live row")
        _check(st.cur_game == 2 and st.cur_step == 2,
               f"follow did not ride the live game ({st.cur_game}, {st.cur_step})")
        _check("LIVE" in str(app.query_one("#games").get_option_at_index(2).prompt),
               "live row not labelled LIVE")
        fin = _game(obs, num, 3, -1.0)
        app.post_message(tui_analysis.EngineEvent(bs.GameFinished(fin)))
        await _wait(pilot, lambda: st.live_idx is None and st.games[2] is fin,
                    "the live game to finish")
        await pilot.pause(0.2)
        _check("LOSS" in str(app.query_one("#games").get_option_at_index(2).prompt),
               "finished row not relabelled")
        results.append("live stream → LIVE row, follow, finalize")
    return results


def main():
    if not os.path.exists(BINARY):
        print(f"binary not found at {BINARY} — run `make` first", file=sys.stderr)
        return 2
    decks = _write_decks()
    tmp = tempfile.mkdtemp(prefix="tui_browser_test_")
    try:
        obs, num = _real_obs()
        games = [_game(obs, num, 3, 1.0), _game(obs, num, 4, -1.0, False)]
        trace, n = bs.save_session(os.path.join(tmp, "session"), games, _PROV)
        if not trace.endswith(gui_session_io.TRACE_EXT) or n != 2:
            print(f"FAIL  save_session: {trace} {n}")
            return 1
        try:
            results = asyncio.run(_drive(trace, os.path.join(tmp, "resaved.rmtrace"),
                                         obs, num))
        except (TuiBrowserError, AssertionError) as e:
            print(f"FAIL  tui browser: {e}", flush=True)
            return 1
        for r in results:
            print(f"ok    {r}", flush=True)
        print(f"\ntui browser: {len(results)}/6 checks passed", flush=True)
        return 0 if len(results) == 6 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for p in decks:
            try:
                os.remove(p)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
