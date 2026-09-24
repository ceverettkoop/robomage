"""Front-end-independent core of the analysis browser.

Everything the trace-browsing front ends share, extracted from tui_analysis.py
so the Textual TUI and the Qt GUI cannot drift: the games store and step
cursor, the analyses registry and its one process-global stdout-capture lock,
the presentation-data helpers (game-list labels, decision rows, phase strip,
clock line), the V(s) histogram geometry/bucketing model, and the engine-side
job bodies (load, collect with live streaming, whatif, shard replay, and the
replay-to-step MCTS `search_step`).

Threading contract (mirrors tui_analysis's @work groups):
  * `BrowseStore` is UI-thread-only — the front end applies events to it on its
    UI thread (one-writer discipline).
  * `EngineCore` is worker-thread-only — exactly one engine worker thread ever
    touches model/env/opp_model, running one job at a time; each job streams
    `Event` dataclasses through the thread-safe `emit` callable supplied at
    construction (Textual: a post_message wrapper; Qt: a queued-signal bridge)
    and ends by emitting EngineIdle.
  * Finished game dicts are immutable once emitted — sharing them read-only
    with an analysis thread is safe.

No Textual, Qt, or Rich imports; torch is only reached through analysis.py's
lazy imports (loading a model / collecting traces), so this module and its
pure helpers stay importable everywhere.
"""

import io
import os
import threading
import traceback
from contextlib import redirect_stdout
from dataclasses import dataclass, field

import numpy as np

import analysis as an
import decode
from cli_spec import is_bo3
import tree_rebuild
from env import (STATE_SIZE, _IS_SIDEBOARD_IDX, _SELF_IS_A_IDX,
                 _STEP_ONEHOT_START,
                 _STEP_ONEHOT_SIZE)

# ── Stdout capture ────────────────────────────────────────────────────────────
# One lock around every stdout capture: engine and analysis workers run in
# separate threads and redirect_stdout swaps the process-global sys.stdout, so
# concurrent captures would interleave each other's prints. This is THE lock —
# tui_analysis re-imports it; never create a second one.
CAPTURE_LOCK = threading.Lock()


def capture(fn, *a, **k):
    """Run a printing analysis function, returning its stdout as a string."""
    buf = io.StringIO()
    with CAPTURE_LOCK, redirect_stdout(buf):
        fn(*a, **k)
    return buf.getvalue()


# ── Analyses registry ─────────────────────────────────────────────────────────

def run_shap(games, n_background=50, n_samples=200):
    """Fit the V(s) surrogate and print SHAP feature importances (the REPL's
    'shap' command, minus the chart)."""
    try:
        import shap
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError as e:
        print(f"  Missing dependency: {e}")
        return
    all_interp = np.array([f for g in games for f in g["interp_features"]])
    all_vals = np.array([v for g in games for v in g["values"]])
    print(f"Fitting surrogate on {len(all_interp)} points...")
    surrogate = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.8)
    surrogate.fit(all_interp, all_vals)
    print(f"Surrogate R^2: {surrogate.score(all_interp, all_vals):.4f}")
    bg_idx = np.random.choice(len(all_interp),
                              size=min(n_background, len(all_interp)), replace=False)
    smp_idx = np.random.choice(len(all_interp),
                               size=min(n_samples, len(all_interp)), replace=False)
    print(f"Running SHAP ({len(bg_idx)} background, {len(smp_idx)} samples)...")
    explainer = shap.KernelExplainer(surrogate.predict, all_interp[bg_idx])
    shap_vals = explainer.shap_values(all_interp[smp_idx])
    mean_abs = abs(shap_vals).mean(axis=0)
    print(f"\n{'Feature':<25} {'Mean |SHAP|':>12}")
    print("-" * 40)
    for idx in mean_abs.argsort()[::-1]:
        print(f"  {an._INTERP_FEATURE_NAMES[idx]:<23} {mean_abs[idx]:12.4f}")


def has_probs(games):
    return any(g.get("action_probs") for g in games)


def probs_guard(fn):
    """Wrap a probs-dependent analyzer with the REPL's no-data message."""
    def run(games):
        if not has_probs(games):
            print("  No action probability data in these traces.")
        else:
            fn(games)
    return run


# Analyses menu: (key, label, fn(games)). Mirrors the REPL commands (each just
# prints; the front end captures the text via `capture`).
ANALYSES = [
    ("summary", "summary — W/L/D stats", an._sim_summary),
    ("cardvalue", "cardvalue — card importance",
     lambda g: an._analyze_cardvalue(g, top_n=30)),
    ("targeting", "targeting — self/opp targets, hold vs cast", an._sim_targeting),
    ("swings", "swings — top in-game V(s) swings",
     lambda g: an._print_swing_table(an._compute_swings(g)[:15])),
    ("boundaries", "boundaries — V(s) across bo3 games",
     lambda g: an._print_boundaries(an._compute_boundaries(g))),
    ("matchcal", "matchcal — V at game start by match score",
     an._print_match_calibration),
    ("regret", "regret — high-regret decisions",
     probs_guard(lambda g: an._analyze_regret(g, top_n=20))),
    ("entropy", "entropy — policy entropy by phase/board",
     probs_guard(an._analyze_entropy)),
    ("consistency", "consistency — similar states, different actions",
     lambda g: an._analyze_consistency(g, top_n=10)),
    ("calibration", "calibration — start V(s) vs win rate", an._analyze_calibration),
    ("turning", "turning — point of no return", an._analyze_turning_points),
    ("clusters", "clusters — V(s) curve archetypes", an._analyze_clusters),
    ("sideboard", "sideboard — sideboard decisions (bo3)", an._sim_sideboard_report),
    ("sbvalue", "sbvalue — sideboard preference & impact (bo3)", an._analyze_sbvalue),
    ("shap", "shap — feature importance (slow)", run_shap),
]

# Live-env entries appended after the analyses (they need the engine worker).
ENGINE_MENU = [
    ("whatif", "whatif — branch alternatives at current step (w); "
               "adds each branch as a steppable ↳trace"),
    ("run5", "run +5 — simulate 5 more games"),
    ("run20", "run +20 — simulate 20 more games"),
]

# Replay-search entry: needs a REPLAYABLE game (recorded engine seed + action
# log — a recording's .rmplay sidecars, or a replay-enabled collector trace),
# not the session's live env, so it stays available in shard-browse mode.
REPLAY_MENU = [
    ("search", "search — replay to current step + MCTS analysis "
               "(the live window's F6 review, offline)"),
]

# Exact-tree entry: needs a recording whose step carries search diagnostics
# (a search opponent's own decision, kind 1 searched / kind 2 followed) — the
# recorded tree is rebuilt bit-for-bit and browsed in the Tree tab.
TREE_MENU = [
    ("tree", "Rebuild search tree (F7)"),
]

_MSG_NO_DECKS = ("This session does not know the game's seat decks — cannot "
                 "build a replay env.")

_TABLE_HEAD = f"   {'visits':>6} {'v%':>7} {'prior':>6} {'Q':>7}  action"


def _tree_cache_dir(game):
    """``<shard dir>/trees`` for a record that knows its shard, else None
    (TreeSession then skips the cache)."""
    stem = game.get("shard_stem")
    return os.path.join(os.path.dirname(stem), "trees") if stem else None


def _visit_table_lines(visits, priors, q, played, describe):
    """The shared visits/v%/prior/Q rows, most-visited first; ``describe(i)``
    names action ``i``."""
    visits = np.asarray(visits, dtype=np.float64)
    tot = max(float(visits.sum()), 1.0)
    lines = [_TABLE_HEAD]
    for i in np.argsort(-visits):
        i = int(i)
        mark = "▶" if i == played else " "
        lines.append(f" {mark} {int(visits[i]):>6} {visits[i] / tot:>7.1%} "
                     f"{float(priors[i]):>6.3f} {float(q[i]):>+7.3f}  [{i}] "
                     f"{describe(i)}")
    return lines


def _mover_label(obs):
    return "A" if bool(obs[_SELF_IS_A_IDX] > 0.5) else "B"


def _rebuilt_tree_text(game, step, session):
    """Display text for an exact rebuild: the verification line, the root
    table (a followed row: its origin's root plus the followed node's own
    summed table)."""
    stats = session.root_stats()
    root_step = session.root_step
    root_obs = np.asarray(game["observations"][root_step], dtype=np.float32)
    source = "cache hit" if session.from_cache else "rebuilt"
    lines = [f"verified: rebuilt visits == recorded ({source})"
             if session.verified else
             f"MISMATCH: {session.mismatch} (showing rebuilt)"]
    lines.append(session.summary())
    lines.append("")
    lines.append(f"Tree root @ step {root_step} (mover: Player "
                 f"{_mover_label(root_obs)}) — root value "
                 f"{stats.root_value:+.3f} "
                 f"(win% {50.0 * (1.0 + stats.root_value):.1f})")
    lines.append("")
    lines += _visit_table_lines(stats.visits, stats.priors, stats.q,
                                int(game["actions"][root_step]),
                                lambda i: an._action_desc(root_obs, i))
    if session.follow_path:
        obs = np.asarray(game["observations"][step], dtype=np.float32)
        num = int(game["num_choices"][step])
        n_sum = np.zeros(num, dtype=np.int64)
        w_sum = np.zeros(num, dtype=np.float64)
        p_max = np.zeros(num, dtype=np.float64)
        for w in session.follow_worlds:
            for a, n, q, p in session.node_stats(w, session.follow_path):
                if a < num:
                    n_sum[a] += n
                    w_sum[a] += q * n
                    p_max[a] = max(p_max[a], p)
        q_node = np.divide(w_sum, n_sum, out=np.zeros(num), where=n_sum > 0)
        lines.append("")
        lines.append(f"Followed node @ step {step} (mover: Player "
                     f"{_mover_label(obs)}) — path {session.follow_path} in "
                     f"worlds {session.follow_worlds}, summed over them")
        lines.append("")
        lines += _visit_table_lines(n_sum, p_max, q_node,
                                    int(game["actions"][step]),
                                    lambda i: an._action_desc(obs, i))
    lines.append("")
    lines.append("▶ = the action played in the recording. Q and the root "
                 "value are from the MOVER's perspective.")
    return "\n".join(lines)


def pv_line(pv, labels):
    """``a → b → c (N, Q)`` for a ``mcts.PVStep`` list: the root action named
    from the real menu, deeper steps by index (their menus are synthesized
    inside the simulation), then the root action's visits and Q."""
    if not pv:
        return ""
    first = pv[0]
    names = [labels[first.action] if first.action < len(labels)
             else f"#{first.action}"]
    names += [f"#{s.action}" for s in pv[1:]]
    return f"{' → '.join(names)} ({first.visits}, {first.q:+.3f})"


def _tree_ready_event(gn, step, session):
    """Snapshot an open TreeSession's root into a TreeReady (every field a
    plain Python value, so the UI thread never touches the session)."""
    stats = session.root_stats()
    labels = session.root_labels()
    merged = [(i, int(stats.visits[i]), float(stats.q[i]),
               float(stats.priors[i])) for i in range(int(stats.num_choices))]
    root_rows, pv_lines = [], []
    for w in range(session.worlds):
        rows = session.node_stats(w, [])
        root_rows.append(rows)
        pv_lines.append({a: pv_line(session.pv(a, w), labels)
                         for a, n, _q, _p in rows if n > 0})
    return TreeReady(gn=int(gn), step=int(step), root_step=session.root_step,
                     summary=session.summary(), verified=session.verified,
                     from_cache=session.from_cache, mismatch=session.mismatch,
                     worlds=session.worlds, root_labels=labels,
                     root_rows=root_rows, merged_rows=merged,
                     pv_lines=pv_lines,
                     follow_path=list(session.follow_path),
                     follow_worlds=list(session.follow_worlds))


def replay_search_decks(game, args):
    """The (deck_a, deck_b, bo3) a replay-search env must be built with:
    the record's own absolute-seat decks (a recording's sidecar), else the
    session args'. Returns None when neither knows the decks."""
    rd = game.get("replay_decks")
    if rd and rd.get("deck_a") and rd.get("deck_b"):
        return rd["deck_a"], rd["deck_b"], bool(rd.get("bo3", True))
    deck_a = getattr(args, "deck_a", None)
    deck_b = getattr(args, "deck_b", None)
    if deck_a and deck_b:
        return deck_a, deck_b, is_bo3(args)
    return None


def run_replay_search(game, step, *, binary, deck_a, deck_b, bo3,
                      eval_spec="az:gen", sims=256, worlds=4):
    """Replay ``game`` to model decision ``step`` on a fresh search env and
    run a determinized MCTS there — the offline equivalent of the live
    analysis window's F6 "opponent's last decision" review. Returns display
    text.

    ``deck_a``/``deck_b`` must be the ABSOLUTE seat decks of the recorded
    session (see :func:`replay_search_decks`) — the replay resets with the
    recorded engine seed and feeds the recorded action log, so no
    model-seat deck swap applies. The reached obs is verified against the
    recorded one before searching (divergence is reported, not hidden).

    A step that recorded search diagnostics (a search opponent's own searched
    or tree-followed decision) gets its EXACT tree instead — rebuilt with the
    recording's seeds/evaluator (or installed from the tree cache beside the
    shard) and verified against the recorded visits — so the table is the
    played search, not a fresh one."""
    import mcts
    from opponents import load_spec_evaluator

    if not an._game_is_replayable(game):
        return ("This game has no recorded seed/action log. Training-pool "
                "shards and recordings made before the replay sidecar cannot "
                "be replayed — record a new session to enable this view.")
    prefix = game["prefix_len"][step]
    if prefix is None or game["full_actions"] is None:
        return "This step has no recorded replay position."
    diag = step_diag(game, step)
    if diag is not None and diag.get("kind") in (tree_rebuild.DIAG_KIND_SEARCH,
                                                 tree_rebuild.DIAG_KIND_FOLLOWED):
        try:
            with tree_rebuild.TreeSession(
                    game, step, binary=binary, deck_a=deck_a, deck_b=deck_b,
                    bo3=bo3, cache_dir=_tree_cache_dir(game)) as session:
                return _rebuilt_tree_text(game, step, session)
        except tree_rebuild.RebuildError as exc:
            return f"Cannot rebuild the recorded tree: {exc}"
    lines = []
    try:
        env = tree_rebuild.replay_to_step(game, step, binary=binary,
                                         deck_a=deck_a, deck_b=deck_b,
                                         bo3=bo3, strict=False)
    except tree_rebuild.RebuildError as exc:
        return f"{exc} (deck files changed since the recording?)."
    try:
        obs = env._obs
        expected = np.asarray(game["observations"][step], dtype=np.float32)
        if not np.allclose(obs, expected, atol=1e-4):
            n_diff = int(np.sum(~np.isclose(obs, expected, atol=1e-4)))
            lines.append(f"WARNING: replay diverged from the recorded state "
                         f"({n_diff} obs floats differ) — the search below "
                         f"may not describe the recorded position.")
            lines.append("")
        if not getattr(env, "last_search_safe", 0):
            lines.append("This decision is not a legal search root "
                         "(safe=0 prompt) — no search possible here.")
            return "\n".join(lines)
        evaluator, label = load_spec_evaluator(eval_spec)
        is_sb = bool(expected[_IS_SIDEBOARD_IDX] > 0.5)
        if is_sb:
            # A replayed sideboard prompt gets the same flat plan search the
            # live seats use (visits there are pi-proportional, Q per first
            # pick = its best plan's cross-world mean value).
            res = mcts.run_plan_search(env, evaluator, worlds=int(worlds),
                                       rng=np.random.default_rng(0))
            effort = (f"{res.sims_run} plan evals x {worlds} worlds "
                      f"(plan search)")
        else:
            # Cross-world batching is arithmetically identical to the
            # sequential search, so the offline review keeps it on
            # unconditionally; the evaluator's device follows
            # ROBOMAGE_EVAL_DEVICE via the loader.
            res = mcts.run_search(env, evaluator, sims=int(sims),
                                  worlds=int(worlds),
                                  rng=np.random.default_rng(0),
                                  cross_world=True)
            effort = f"{res.sims_run} sims x {worlds} worlds"
        num = int(game["num_choices"][step])
        q = res.q if res.q is not None else np.zeros(num)
        lines.append(f"MCTS @ step {step} (mover: Player "
                     f"{_mover_label(expected)}) — {label}, {effort}, "
                     f"root value {res.root_value:+.3f} "
                     f"(win% {50.0 * (1.0 + res.root_value):.1f})")
        lines.append("")
        lines += _visit_table_lines(res.visits, res.priors, q,
                                    int(game["actions"][step]),
                                    lambda i: an._action_desc(expected, i))
        lines.append("")
        lines.append("▶ = the action played in the recording. Q and the root "
                     "value are from the MOVER's perspective.")
        return "\n".join(lines)
    finally:
        env.close()


# ── Presentation-data helpers (pure over game dicts) ──────────────────────────

def result_str(game):
    """WIN/LOSS/DRAW from the model's perspective; LIVE for an in-progress
    streamed game (result is None until GameFinished replaces it)."""
    if game.get("live"):
        return "LIVE"
    r = game["result"]
    return "WIN" if r > 0 else ("LOSS" if r < 0 else "DRAW")


def game_label(i, game):
    """One games-list row: index, result, bo3 score, decision count, seat —
    with the ↳ whatif-origin, ⛁ shard, and LIVE markers."""
    w = game.get("whatif")
    if w:
        # A whatif branch trace: mark its origin instead of the bo3 score.
        return (f"{i:>3}  {result_str(game):<4} {len(game['values']):>4}d  "
                f"↳g{w['src_game']}@{w['step']}")
    if game.get("live"):
        return f"{i:>3}  {result_str(game):<4} {'—':<4} {len(game['values']):>4}d  …"
    sc = an._match_score(game)
    sc_str = f"{sc[0]}-{sc[1]}" if sc is not None else " — "
    side = "A" if game["model_is_a"] else "B"
    label = (f"{i:>3}  {result_str(game):<4} {sc_str:<4} "
             f"{len(game['values']):>4}d  {side}")
    if game.get("shard"):
        label += "  ⛁"
    return label


def analysis_pool(games):
    """The games the statistical analyses may pool: finished, independent
    samples only. Whatif branch traces share their source game's prefix and
    live games have no outcome yet — both would bias every statistic."""
    return [g for g in games if not g.get("whatif") and not g.get("live")]


def summary_line(games, loading=False):
    """The sidebar W/L/D summary over finished real games."""
    real = analysis_pool(games)
    w = sum(1 for g in real if g["result"] > 0)
    l = sum(1 for g in real if g["result"] < 0)
    d = len(real) - w - l
    line = f"{len(real)} games · {w}W/{l}L/{d}D"
    n_wf = sum(1 for g in games if g.get("whatif"))
    if n_wf:
        line += f" · +{n_wf} whatif"
    if any(g.get("live") for g in games):
        line += " · 1 live"
    if loading:
        line += "  (simulating…)"
    return line


def clock_line(game, step):
    """Match-clock strip for the decision panel: each clocked seat's bank
    entering this model decision, as 'remaining / bank' seconds. Empty when
    neither seat played under a match clock (no clock= knob in its spec).
    The model's reading is recorded at each of its decisions; the opponent's
    is the last reading its actions left at or before this step (its full
    bank before it has acted)."""
    def fmt(remaining, bank):
        rem = f"{remaining:.1f}s" if remaining is not None else "?"
        return f"{rem} / {bank:g}s"
    parts = []
    if game.get("clock_bank") is not None:
        rems = game.get("clock_remaining") or []
        rem = rems[step] if step < len(rems) else None
        parts.append("model " + fmt(rem, game["clock_bank"]))
    if game.get("opp_clock_bank") is not None:
        opp_rem = game["opp_clock_bank"]
        for oa in game.get("opp_actions", []):
            if (oa["before_model_step"] <= step
                    and oa.get("clock") is not None):
                opp_rem = oa["clock"]
        parts.append("opp " + fmt(opp_rem, game["opp_clock_bank"]))
    return "⏱ clock: " + " · ".join(parts) if parts else ""


def opp_actions_before(game, step):
    """Opponent actions between model decisions step-1 and step, with runs
    of identical consecutive actions collapsed ('PASS (x19)')."""
    descs = [oa["desc"] for oa in game.get("opp_actions", [])
             if oa["before_model_step"] == step]
    lines = []
    run_desc, run_len = None, 0

    def flush():
        if run_len == 1:
            lines.append(run_desc)
        elif run_len > 1:
            lines.append(f"{run_desc} (x{run_len})")
    for desc in descs:
        if desc == run_desc:
            run_len += 1
        else:
            flush()
            run_desc, run_len = desc, 1
    flush()
    return lines


def info_line(label, p, library):
    """One player's life/counters/mana/hand/library info line. Poison (☠) /
    energy (⚡) only when set (mirrors the play boards)."""
    counters = ""
    if p.get("poison", 0) > 0:
        counters += f"  ☠ {p['poison']}"
    if p.get("energy", 0) > 0:
        counters += f"  ⚡ {p['energy']}"
    return (f"{label}  ♥ {p['life']}{counters}  "
            f"mana [{decode.fmt_mana(p['mana'])}]  "
            f"hand {p['hand_count']}  lib {library}")


@dataclass
class PhaseData:
    """The phase-strip content, renderer-agnostic: the front end highlights
    step cell `cur_step_idx` and lays the text fragments out its own way."""
    cur_step_idx: int
    header: str       # "G4 (WIN) · decision 37/141 · V=+0.312"
    match: str        # " · match 1–0 SIDEBOARD" or ""
    context: str      # "Turn 6 · Active A · Priority B"


def phase_data(game, gn, step, obs, gs):
    cur = int(np.argmax(
        obs[_STEP_ONEHOT_START:_STEP_ONEHOT_START + _STEP_ONEHOT_SIZE]))
    val = game["values"][step] if step < len(game["values"]) else None
    vstr = f" · V={val:+.3f}" if val is not None else ""
    header = (f"G{gn} ({result_str(game)}) · decision {step}/"
              f"{max(len(game['observations']) - 1, 0)}{vstr}")
    m = gs.get("match") or {}
    mstr = ""
    if any(m.get(k) for k in ("self_wins", "opp_wins", "is_sideboard")):
        mstr = f" · match {m['self_wins']}–{m['opp_wins']}"
        if m.get("is_sideboard"):
            mstr += " SIDEBOARD"
    active = "A" if gs["active_is_a"] else "B"
    context = (f"Turn {gs['turn']} · Active {active} · "
               f"Priority {gs['priority_player']}")
    return PhaseData(cur_step_idx=cur, header=header, match=mstr, context=context)


@dataclass
class DecisionRow:
    k: int                    # legal-action index
    prob: object              # float | None (no recorded policy)
    desc: str
    is_chosen: bool
    visits: object = None     # int | None — search visit count (diag rows)
    q: object = None          # float | None — search Q for this action
    prior: object = None      # float | None — net prior the search started from


@dataclass
class DecisionData:
    clock: str                # "" when neither seat is clocked
    rows: list                # [DecisionRow], sorted most-probable first
    opp_lines: list           # opponent actions since the previous decision
    shard_caveat: bool
    num_choices: int
    search_line: str = ""     # one-line search summary ("" without a diag)


def step_diag(game, step):
    """The recorded search diagnostics dict for `step` (None when the record
    carries none, e.g. a simulated trace or a plain one-hot row)."""
    diags = game.get("diag")
    if diags and step < len(diags):
        return diags[step]
    return None


def search_line_for(diag, origin_step):
    """One-line summary of a step's recorded search (``diag`` as loaded by
    ``shard_replay.diag_row``; ``origin_step`` the followed row's origin as a
    step index in the same record, None when it isn't in view). "" when there
    is no diag."""
    if not diag:
        return ""
    kind = diag.get("kind", 0)
    if kind == 2:
        if origin_step is None:
            return "followed (origin not in view)"
        path = diag.get("follow_path") or []
        visits = int(np.sum(diag.get("visits", [])))
        return (f"followed from decision {origin_step} via {len(path)} "
                f"action(s) · {visits} visits")
    if kind == 3:
        return f"sideboard plan search · {diag.get('sims_run', 0)} sims"
    parts = [f"search: {diag.get('sims_run', 0)} sims",
             f"{diag.get('n_worlds', 0)} worlds",
             f"root V {diag.get('root_value', 0.0):+.3f}"]
    tb = diag.get("time_budget_s")
    if tb is not None:
        timed = f"timed {tb:.1f}s"
        if diag.get("stopped_early"):
            timed += " (stopped early)"
        parts.append(timed)
    reused = int(diag.get("reused_visits", 0))
    if reused:
        parts.append(f"reused {reused}")
    return " · ".join(parts)


def search_caption(game, step):
    """The search-effort suffix for a step's caption (" · 512 sims", or
    " · followed" for a decision answered from an earlier search's trees);
    "" when no search ran there."""
    diag = step_diag(game, step)
    if not diag:
        return ""
    if diag.get("kind") == 2:
        return " · followed"
    return f" · {int(diag.get('sims_run', 0))} sims"


def decision_data(game, step):
    """The model's decision at `step` as renderer-agnostic rows (the recorded
    obs stays full OBS_SIZE — _action_desc reads the action-metadata blocks
    past STATE_SIZE). A recorded search's diag adds per-row visits / Q /
    prior and the summary line; searched rows sort by visits, the rest by
    probability."""
    obs = game["observations"][step]
    num_ch = game["num_choices"][step] if step < len(game["num_choices"]) else 0
    chosen = game["actions"][step] if step < len(game.get("actions", [])) else None
    probs = None
    if game.get("action_probs") and step < len(game["action_probs"]):
        probs = game["action_probs"][step]
    diag = step_diag(game, step)
    origins = game.get("origin_step") or []
    origin_step = origins[step] if step < len(origins) else None

    visits = q_act = priors = None
    if diag is not None:
        visits = diag.get("visits")
        if diag.get("kind") in (1, 3):
            q_act = diag.get("q_act")
            priors = diag.get("priors")

    def _field(vec, k, cast):
        return cast(vec[k]) if vec is not None and k < len(vec) else None

    order = range(num_ch)
    if visits is not None and diag.get("kind") in (1, 3):
        order = sorted(order, key=lambda k: -(_field(visits, k, int) or 0))
    elif probs is not None:
        order = sorted(order, key=lambda k: -probs[k])
    rows = [DecisionRow(k=k,
                        prob=float(probs[k]) if probs is not None else None,
                        desc=an._action_desc(obs, k),
                        is_chosen=(k == chosen),
                        visits=_field(visits, k, int),
                        q=_field(q_act, k, float),
                        prior=_field(priors, k, float))
            for k in order]
    return DecisionData(clock=clock_line(game, step), rows=rows,
                        opp_lines=opp_actions_before(game, step),
                        shard_caveat=bool(game.get("shard")),
                        num_choices=num_ch,
                        search_line=search_line_for(diag, origin_step))


# ── V(s) histogram geometry ───────────────────────────────────────────────────

class HistogramModel:
    """The portable half of the TUI's ValueHistogram: values → plot columns.

    Vertical scale is symmetric ±vmax where vmax = max(1, max|V|) — usually
    ±1, stretched only when a bo3 trace's V exceeds it (game rewards stack
    with the match terminal). When the game has more decisions than plot
    columns, steps are bucketed and each column carries the bucket's max-|V|
    step (preserving swings a mean would smooth away); a click on such a
    column seeks to that extreme step. The front end calls `layout(n_cols)`
    with however many columns fit its render surface and draws from `cols`
    (a list of `(value, representative_step)`; `value` None = spacer)."""

    def __init__(self):
        self.values = []
        self.cursor = 0
        self.vmax = 1.0
        self.cols = []
        self.bar_w = 1
        self.bucketed = False
        self._n_cols = 0

    def set_data(self, values, cursor=0):
        self.values = [float(v) for v in values]
        self.cursor = (max(0, min(cursor, len(self.values) - 1))
                       if self.values else 0)
        self.vmax = max(1.0, max((abs(v) for v in self.values), default=1.0))
        self._relayout()

    def append(self, v):
        """Live streaming: extend by one value (vmax only ever grows, so bars
        already drawn never rescale downward mid-game)."""
        v = float(v)
        self.values.append(v)
        if abs(v) > self.vmax:
            self.vmax = abs(v)
        self._relayout()

    def set_cursor(self, step):
        if not self.values:
            return
        self.cursor = max(0, min(step, len(self.values) - 1))

    def layout(self, n_cols):
        self._n_cols = max(0, int(n_cols))
        self._relayout()

    def _relayout(self):
        """Rebuild the column -> (value, step) mapping for the current width."""
        w = self._n_cols
        n = len(self.values)
        self.cols = []
        self.bucketed = False
        self.bar_w = 1
        if not n or w <= 0:
            return
        if n <= w:
            # Whole steps fit: widen bars up to 3 cells, with a 1-cell spacer
            # between bars when there's room (the spacer still maps to its step
            # so clicks in the gap don't dead-zone).
            per = max(1, min(4, w // n))
            self.bar_w = per - 1 if per >= 2 else 1
            for s, v in enumerate(self.values):
                self.cols.extend([(v, s)] * self.bar_w)
                if per >= 2:
                    self.cols.append((None, s))
        else:
            self.bucketed = True
            for c in range(w):
                lo = c * n // w
                hi = max(lo + 1, (c + 1) * n // w)
                rep = max(range(lo, hi), key=lambda s: abs(self.values[s]))
                self.cols.append((self.values[rep], rep))

    def cursor_cols(self):
        """Set of plot-column indices highlighted for the cursor step."""
        n = len(self.values)
        if not n or not self.cols:
            return set()
        if self.bucketed:
            return {min(self.cursor * len(self.cols) // n, len(self.cols) - 1)}
        per = self.bar_w + (1 if len(self.cols) > n * self.bar_w else 0)
        start = self.cursor * per
        return set(range(start, min(start + self.bar_w, len(self.cols))))

    def step_at(self, col):
        """Map a plot-column index to a decision step (or None)."""
        if 0 <= col < len(self.cols):
            return self.cols[col][1]
        return None


# ── Worker → UI events ────────────────────────────────────────────────────────

@dataclass
class EnvReady:
    startup_text: str
    subtitle: str = ""


@dataclass
class LoadFailed:
    text: str


@dataclass
class GameStarted:
    model_is_a: bool
    engine_seed: object


@dataclass
class StepAppended:
    step: dict


@dataclass
class OppActionAppended:
    opp: dict


@dataclass
class GameFinished:
    game: dict


@dataclass
class GameAborted:
    pass


@dataclass
class GameAdded:
    """A complete game arriving whole (shard record, whatif branch trace)."""
    game: dict


@dataclass
class EngineNote:
    text: str


@dataclass
class EngineIdle:
    pass


@dataclass
class AnalysisDone:
    title: str
    text: str


@dataclass
class TreeReady:
    """An exact rebuilt (or cached) search tree is open on the engine worker
    for browsing (EngineCore.open_tree). ``root_rows[w]`` is world ``w``'s
    root as ``(action, N, Q, P)`` rows; ``merged_rows`` the same over the
    summed root stats; ``pv_lines[w][action]`` the pre-rendered principal
    variation text. A followed row carries the path from this root to the
    followed node and the worlds whose trees reach it."""
    gn: int
    step: int
    root_step: int
    summary: str
    verified: bool
    from_cache: bool
    mismatch: object                # str | None
    worlds: int
    root_labels: list
    root_rows: list                 # per world: [(action, N, Q, P)]
    merged_rows: list               # [(action, N, Q, P)] from root_stats
    pv_lines: list                  # per world: {action: "a → b (N, Q)"}
    follow_path: list = field(default_factory=list)
    follow_worlds: list = field(default_factory=list)


@dataclass
class TreeNodes:
    """One node of the open tree, expanded (EngineCore.tree_expand): its
    ``(action, N, Q, P)`` rows and the child action labels, plus the WalkNode
    per step of the path (the last one's obs is the hypothetical board;
    ``terminal`` is set when the walk ended the game)."""
    world: int
    path: list
    rows: list
    labels: list
    walk_nodes: list
    terminal: object                # None | "A" | "B" | "DRAW"


@dataclass
class TreeClosed:
    reason: str


@dataclass
class Applied:
    """What a BrowseStore.apply changed, so the front end refreshes only what
    it must. game_idx is the affected row (None when no row changed);
    `removed` marks a deleted row (aborted live game); `selected_grew` marks a
    StepAppended landing on the currently selected game (follow-mode hook)."""
    event: object
    game_idx: object = None
    removed: bool = False
    selected_grew: bool = False


def _live_placeholder(model_is_a, engine_seed):
    """The in-progress game dict a GameStarted opens: the trace schema with
    empty per-step lists, no result, and no replay keys (full_actions=None so
    _game_is_replayable refuses it until GameFinished swaps in the real dict)."""
    return {"observations": [], "values": [], "interp_features": [],
            "actions": [], "num_choices": [], "action_probs": [],
            "opp_actions": [], "clock_remaining": [], "prefix_len": [],
            "diag": [], "origin_step": [],
            "engine_seed": engine_seed, "full_actions": None,
            "result": None, "model_is_a": model_is_a,
            "clock_bank": None, "opp_clock_bank": None, "live": True}


_STEP_KEYS = (("obs", "observations"), ("value", "values"),
              ("interp", "interp_features"), ("num_choices", "num_choices"),
              ("probs", "action_probs"), ("clock", "clock_remaining"),
              ("diag", "diag"), ("origin_step", "origin_step"),
              ("prefix_len", "prefix_len"), ("action", "actions"))


class BrowseStore:
    """The games store + cursor. UI-thread-only: the front end applies worker
    events here (one writer) and reads for rendering."""

    def __init__(self):
        self.games = []
        self.cur_game = None      # index into games
        self.cur_step = 0
        self.engine_busy = True   # startup load+collect owns the env first
        self.analysis_busy = False
        self.live_idx = None      # index of the in-progress streamed game

    # ----- event application -----

    def apply(self, ev):
        if isinstance(ev, GameStarted):
            self.games.append(_live_placeholder(ev.model_is_a, ev.engine_seed))
            self.live_idx = len(self.games) - 1
            return Applied(ev, game_idx=self.live_idx)
        if isinstance(ev, StepAppended):
            if self.live_idx is None:
                return Applied(ev)
            g = self.games[self.live_idx]
            for src, dst in _STEP_KEYS:
                g[dst].append(ev.step.get(src))
            return Applied(ev, game_idx=self.live_idx,
                           selected_grew=(self.cur_game == self.live_idx))
        if isinstance(ev, OppActionAppended):
            if self.live_idx is None:
                return Applied(ev)
            self.games[self.live_idx]["opp_actions"].append(ev.opp)
            return Applied(ev, game_idx=self.live_idx,
                           selected_grew=(self.cur_game == self.live_idx))
        if isinstance(ev, GameFinished):
            if self.live_idx is None:       # defensive: no placeholder open
                self.games.append(ev.game)
                return Applied(ev, game_idx=len(self.games) - 1)
            idx, self.live_idx = self.live_idx, None
            self.games[idx] = ev.game       # authoritative record, wholesale
            return Applied(ev, game_idx=idx)
        if isinstance(ev, GameAborted):
            if self.live_idx is None:
                return Applied(ev)
            idx, self.live_idx = self.live_idx, None
            del self.games[idx]             # a partial game poisons the pools
            if self.cur_game is not None:
                if self.cur_game == idx:
                    self.cur_game = None
                    self.cur_step = 0
                elif self.cur_game > idx:
                    self.cur_game -= 1
            return Applied(ev, game_idx=idx, removed=True)
        if isinstance(ev, GameAdded):
            self.games.append(ev.game)
            return Applied(ev, game_idx=len(self.games) - 1)
        if isinstance(ev, EngineIdle):
            self.engine_busy = False
            return Applied(ev)
        # EnvReady / LoadFailed / EngineNote / AnalysisDone: display-only.
        return Applied(ev)

    # ----- cursor -----

    def select_game(self, gn):
        if not (0 <= gn < len(self.games)):
            return False
        self.cur_game = gn
        self.cur_step = 0
        return True

    def clamp_step(self, step):
        if self.cur_game is None:
            return 0
        n = len(self.games[self.cur_game]["observations"])
        return max(0, min(step, n - 1))

    def selected(self):
        return None if self.cur_game is None else self.games[self.cur_game]


# ── Engine-side job bodies (worker-thread-only) ───────────────────────────────

class EngineCore:
    """Owns model/env/opp_model exclusively; every method is a synchronous job
    run on the front end's single engine worker thread, streaming events
    through `emit`. `args` is an ANALYSIS_TUI_TOOL-style namespace the core
    may mutate (_apply_search_knob_flags self-clears; deck_a/deck_b are
    written back) — hand it a dedicated copy. `preloaded=(model, env,
    opp_model)` skips _load_model_and_env (the test seam)."""

    def __init__(self, args, emit, preloaded=None):
        self.args = args
        self.emit = emit
        self.model = self.env = self.opp_model = None
        self._tree = None              # open tree_rebuild.TreeSession
        self._tree_key = None          # (gn, step) it was opened for
        if preloaded is not None:
            self.model, self.env, self.opp_model = preloaded

    @property
    def has_env(self):
        return self.env is not None and self.model is not None

    def _subtitle(self):
        deck_a = getattr(self.args, "deck_a", None) or "?"
        deck_b = getattr(self.args, "deck_b", None) or "?"
        return (f"{deck_a} (model)  vs  {deck_b} ({self.args.player_b})"
                + ("  · bo3" if is_bo3(self.args) else ""))

    # ----- jobs (each ends by emitting EngineIdle) -----

    def load_and_collect(self, n, stop=None):
        """Startup job: load model+env (or shard records) then stream n games."""
        try:
            if getattr(self.args, "shards", None):
                self._load_shards(n)
                return
            if not self.has_env:
                try:
                    buf = io.StringIO()
                    with CAPTURE_LOCK, redirect_stdout(buf):
                        self.model, self.env, self.opp_model = \
                            an._load_model_and_env(self.args)
                    self.emit(EnvReady(buf.getvalue(), self._subtitle()))
                except BaseException as exc:  # _load_model_and_env may sys.exit()
                    self.emit(LoadFailed(f"model/env load failed: {exc!r}\n"
                                         f"{traceback.format_exc()}"))
                    return
            else:
                self.emit(EnvReady("", self._subtitle()))
            self._collect(n, stop)
        finally:
            self.emit(EngineIdle())

    def collect(self, n, stop=None):
        """Simulate n more games, streaming each decision as it lands."""
        try:
            self._collect(n, stop)
        finally:
            self.emit(EngineIdle())

    def whatif(self, gn, step, k, game):
        """Branch k alternatives at (gn, step); graft each branch trace into
        the games list. `game` is the (immutable) finished dict."""
        try:
            buf = io.StringIO()
            with CAPTURE_LOCK, redirect_stdout(buf):
                branches = an._run_whatif(self.model, self.env, self.opp_model,
                                          game, gn, step, k, make_traces=True)
            text = buf.getvalue()
            added = 0
            for b in branches or []:
                if b.get("trace_game") is not None:
                    self.emit(GameAdded(b["trace_game"]))
                    added += 1
            if added:
                text += (f"\n  Added {added} branch trace(s) to the games list "
                         f"(marked ↳g{gn}@{step}) — select one to step through it.")
            self.emit(AnalysisDone(f"whatif — game {gn}, step {step}", text))
        except Exception:
            self.emit(AnalysisDone("whatif error", traceback.format_exc()))
        finally:
            self.emit(EngineIdle())

    def search_step(self, gn, step, game):
        """Replay-to-step MCTS analysis (REPLAY_MENU's 'search'): rebuilds the
        position from the game's recorded seed + action log on a throwaway
        search env, so it needs no live env/model — available in shard-browse
        mode (recordings with .rmplay sidecars) and for replay-enabled traces
        alike. `game` is the (immutable) finished dict."""
        try:
            params = self._replay_params(game)
            if params is None:
                self.emit(AnalysisDone(
                    f"search — game {gn}, step {step}", _MSG_NO_DECKS))
                return
            spec = getattr(self.args, "player_a", None) or "az:gen"
            text = run_replay_search(game, step, eval_spec=spec, **params)
            self.emit(AnalysisDone(f"search — game {gn}, step {step}", text))
        except Exception:
            self.emit(AnalysisDone("search error", traceback.format_exc()))
        finally:
            self.emit(EngineIdle())

    # ----- exact-tree browsing (TREE_MENU) -----

    def open_tree(self, gn, step, game):
        """Rebuild (or load from cache) the recorded search tree of (gn, step)
        and keep it open on this thread for tree_expand. Emits TreeReady, or
        AnalysisDone("tree", why) + TreeClosed when the step has no
        rebuildable tree; always ends with EngineIdle."""
        try:
            self._close_tree("replaced")
            params = self._replay_params(game)
            if params is None:
                self.emit(AnalysisDone("tree", _MSG_NO_DECKS))
                self.emit(TreeClosed("no decks"))
                return
            try:
                session = tree_rebuild.TreeSession(
                    game, step, cache_dir=_tree_cache_dir(game),
                    **params).open()
            except tree_rebuild.RebuildError as exc:
                self.emit(AnalysisDone(
                    "tree", f"Cannot rebuild the recorded tree: {exc}"))
                self.emit(TreeClosed(str(exc)))
                return
            self._tree = session
            self._tree_key = (int(gn), int(step))
            self.emit(_tree_ready_event(gn, step, session))
        except Exception:
            self._close_tree("error")
            self.emit(AnalysisDone("tree error", traceback.format_exc()))
        finally:
            self.emit(EngineIdle())

    def tree_expand(self, world, path):
        """Expand the open tree's node at ``path`` in ``world``: its rows and
        child labels, plus the walked positions along the path."""
        try:
            if self._tree is None:
                self.emit(AnalysisDone("tree", "No tree is open."))
                return
            path = [int(a) for a in path]
            rows = self._tree.node_stats(world, path)
            nodes, labels = self._tree.walk(world, path)
            terminal = nodes[-1].terminal if nodes else None
            self.emit(TreeNodes(int(world), path, rows, labels, nodes,
                                terminal))
        except Exception:
            self.emit(AnalysisDone("tree error", traceback.format_exc()))
        finally:
            self.emit(EngineIdle())

    def close_tree(self):
        self._close_tree("closed")

    def close(self):
        """Release the tree session and the env (call on the worker thread as
        its last job)."""
        self._close_tree("shutdown")
        if self.env is not None:
            self.env.close()
            self.env = None

    # ----- internals -----

    def _replay_params(self, game):
        """The kwargs a replay/rebuild env needs (binary + absolute-seat decks
        + bo3), or None when the session cannot know the decks."""
        decks = replay_search_decks(game, self.args)
        if decks is None:
            return None
        deck_a, deck_b, bo3 = decks
        binary = getattr(self.args, "binary", None)
        if not binary:
            from runner import BINARY as binary
        return {"binary": binary, "deck_a": deck_a, "deck_b": deck_b,
                "bo3": bo3}

    def _close_tree(self, reason):
        tree, self._tree = self._tree, None
        self._tree_key = None
        if tree is not None:
            tree.close()
            self.emit(TreeClosed(reason))

    def _collect(self, n, stop=None):
        should_stop = stop.is_set if stop is not None else None

        def progress(ev):
            kind = ev["kind"]
            if kind == "game_start":
                self.emit(GameStarted(ev["model_is_a"], ev["engine_seed"]))
            elif kind == "step":
                self.emit(StepAppended(ev["step"]))
            elif kind == "opp_action":
                self.emit(OppActionAppended(ev["opp"]))
            elif kind == "game_end":
                self.emit(GameFinished(ev["game"]))
            elif kind == "game_abort":
                self.emit(GameAborted())
            elif kind == "note":
                self.emit(EngineNote(ev["text"]))

        try:
            # One game per call keeps the between-games stop check and mirrors
            # the TUI's game-at-a-time cadence.
            for _ in range(n):
                if stop is not None and stop.is_set():
                    break
                an._collect_game_traces(self.model, self.env, self.opp_model, 1,
                                        verbose=False, progress=progress,
                                        should_stop=should_stop)
        except Exception:
            self.emit(AnalysisDone("simulation error", traceback.format_exc()))

    def _load_shards(self, n):
        """Shard-replay startup: reconstruct match records from recorded AZ
        self-play instead of simulating. No env is created (has_env stays
        False, so whatif/run stay gated); the model spec is loaded only as the
        V(s) net, unless --no-net keeps values at the recorded z."""
        import shard_replay
        try:
            buf = io.StringIO()
            with CAPTURE_LOCK, redirect_stdout(buf):
                model = None
                if not getattr(self.args, "no_net", False):
                    model = shard_replay.load_value_model(self.args.player_a)
                records = shard_replay.load_records(
                    self.args.shards,
                    viewpoint_is_a=getattr(self.args, "seat", "A") != "B",
                    limit=n or None,
                    interp_fn=an._extract_interpretable)
                if model is not None:
                    shard_replay.apply_net_values(model, records)
                print(f"{len(records)} match record(s) from {self.args.shards} "
                      f"(seat {getattr(self.args, 'seat', 'A')}, "
                      + ("net V(s))" if model is not None else "z values)"))
            net = ("z values" if getattr(self.args, "no_net", False)
                   else f"V(s): {self.args.player_a}")
            subtitle = (f"shard replay: {self.args.shards} · "
                        f"seat {getattr(self.args, 'seat', 'A')} · {net}")
            self.emit(EnvReady(buf.getvalue(), subtitle))
            for g in records:
                self.emit(GameAdded(g))
        except BaseException:
            self.emit(LoadFailed(f"shard load failed:\n{traceback.format_exc()}"))
