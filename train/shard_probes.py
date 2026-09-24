"""Net-probe views for the analysis browser — az_inspect over browsed records.

The browser (gui_browser / tui_analysis) pages per-game trace dicts; az_inspect
owns the "what informed this evaluation" probes (single-state block permutation
importance, card-identity swap, scalar sweeps, search-π-vs-net state detail)
and the pooled net-vs-search views (KL divergence by action category with the
biggest disagreements decoded, net V vs the search's root value, per-bucket
value calibration). ``analysis.py report`` runs the same pooled search-vs-net
views over its batch simulation, so each number has one implementation. This
module is the glue between the two: it snapshots the browser's games into
az_inspect's flat sample-dict shape (obs/pi/mask/z + a (game, step) -> row
index) and runs each view, returning the display lines the front end appends
to its output pane.

Two-phase by design, matching the browser's threading contract:
:func:`snapshot` runs on the UI thread and only takes cheap list() copies of
the per-game step lists (the ndarray rows themselves are append-only, never
mutated); :func:`build_sample` and :func:`run_probe` run on a worker thread
and do the stacking and the torch work. The probe net comes from
:func:`load_probe_net` — ``opponents.load_spec_net``: the net the browser's
model spec names (see the resolver table in opponents.py), in AZNet form, so
a PPO spec is probed through ``az_net.from_ppo`` of that same checkpoint.

Per-row π is the SEARCH posterior (:func:`step_search_pi`): a step's diag
visits (a search / plan / tree-followed decision, simulated or recorded), else
a shard record's resolved ``search_pi`` (the recorded training π for
sidecar-less pool shards). A step with neither — a raw-policy seat, a
human / behavior row, a playout-cap fast row — has no posterior, and the
π-dependent views (``probe_kl``, ``probe_state``'s search column) exclude it
and say so. The simulated trace's ``action_probs`` are the inspection net's
own softmax and are never used as π.

Per-row z: shard-mode records carry an exact per-step game outcome (the shard's
``z`` column, preserved by shard_replay); simulated traces don't, so their rows
fall back to the game's match-level result — close enough for calibration, and
rows with no outcome at all (a live game) are excluded from it.

Qt-free and torch-free at import (torch loads inside the net/probe calls).
"""

import numpy as np

from env import MAX_ACTIONS, OBS_SIZE

# Menu entries the front end appends after the engine section: (key, label).
# The first six probe THE SELECTED DECISION; the last three pool every browsed
# decision that has the needed data.
PROBE_MENU = [
    ("probe_state", "net state — search π vs net at current step"),
    ("probe_blocks", "net blocks — what this evaluation rests on"),
    ("probe_blocks_pi", "net blocks (policy) — what this policy rests on"),
    ("probe_readout", "net readout — V under every matchup column here"),
    ("probe_swap", "net swap — card-identity ΔV at current step"),
    ("probe_sweeps", "net sweeps — life/hand/turn response here"),
    ("probe_kl", "net KL — search vs net by action category (all)"),
    ("probe_value", "net V vs search root value — MAE / corr (all)"),
    ("probe_calib", "net calib — V vs realized z by bucket (all)"),
]
PROBE_KEYS = frozenset(k for k, _ in PROBE_MENU)
DECISION_PROBES = frozenset(
    ("probe_state", "probe_blocks", "probe_blocks_pi", "probe_readout",
     "probe_swap", "probe_sweeps"))

_MAX_POOL_ROWS = 2000     # cap for the pooled views / donor pool
_MAX_SWAP_SITES = 6       # card-swap sites probed per decision (cost control)
_TOP_DISAGREEMENTS = 8    # biggest KL(search‖net) decisions probe_kl decodes


def load_probe_net(model_spec):
    """(AZNet, label) for the probes: ``opponents.load_spec_net`` — the net the
    spec names (the browser's V(s) and replay search read the same one), in
    AZNet form. Raises with a readable message when torch / a checkpoint is
    unavailable (the front end shows it verbatim)."""
    from opponents import load_spec_net
    return load_spec_net(model_spec)


def snapshot(games, cur_game=None, cur_step=None):
    """UI-thread capture: shallow list() copies of each game's step data plus
    the current selection. Cheap — no array copies, no stacking."""
    caps = []
    for gn, g in enumerate(games):
        caps.append({
            "gn": gn,
            "observations": list(g.get("observations") or ()),
            "num_choices": list(g.get("num_choices") or ()),
            "diag": list(g.get("diag") or ()),
            "search_pi": (list(g["search_pi"])
                          if g.get("search_pi") is not None else None),
            "search_v": (list(g["search_v"])
                         if g.get("search_v") is not None else None),
            "z": list(g.get("z") or ()),
            "result": g.get("result"),
        })
    return {"games": caps, "sel": (cur_game, cur_step)}


def step_search_pi(game, step):
    """The search posterior over ``game``'s menu at ``step`` (a float array,
    menu-indexed, merged duplicates on their representative), or None when
    no search ran there. A shard record's resolved ``search_pi`` list wins
    (shard_replay built it from the diag or the recorded π); otherwise the
    step's diag visits. Works on a trace dict and a :func:`snapshot` cap."""
    from shard_record import diag_posterior
    resolved = game.get("search_pi")
    if resolved is not None:
        return resolved[step] if step < len(resolved) else None
    diags = game.get("diag") or ()
    return diag_posterior(diags[step]) if step < len(diags) else None


def step_search_value(game, step):
    """The search's root value at ``game``'s ``step`` (root-mover
    perspective, like the net's V), or None when no search of its own ran
    there (a tree-followed step included). A resolved ``search_v`` list (an
    .rmtrace load, which keeps no diag dicts) wins; otherwise the step's diag
    (:func:`shard_record.diag_root_value`). Works on a trace dict and a
    :func:`snapshot` cap."""
    from shard_record import diag_root_value
    resolved = game.get("search_v")
    if resolved is not None:
        return resolved[step] if step < len(resolved) else None
    diags = game.get("diag") or ()
    return diag_root_value(diags[step]) if step < len(diags) else None


def build_sample(snap):
    """Stack a snapshot into the az_inspect sample-dict shape.

    Returns ``(sample, index, z_valid, pi_valid)``: ``sample`` has
    obs/pi/mask/z (pi is the search posterior, zeros where none; z is 0 where
    unknown) plus ``search_v`` (the search root value,
    :func:`step_search_value`; NaN where none), ``index`` maps
    (game, step) -> row, ``z_valid`` marks rows whose
    z is a real outcome (shard z, or a finished game's result), ``pi_valid``
    rows that carry a search posterior (:func:`step_search_pi`)."""
    obs_rows, pi_rows, mask_rows, z_rows, z_ok, pi_ok, sv_rows, index = \
        [], [], [], [], [], [], [], {}
    for cap in snap["games"]:
        for step, o in enumerate(cap["observations"]):
            o = np.asarray(o, dtype=np.float32)
            if o.shape[0] != OBS_SIZE:
                continue
            n = int(cap["num_choices"][step]) \
                if step < len(cap["num_choices"]) else 0
            n = max(0, min(n, MAX_ACTIONS))
            mask = np.zeros(MAX_ACTIONS, dtype=bool)
            mask[:n] = True
            pi = np.zeros(MAX_ACTIONS, dtype=np.float32)
            post = step_search_pi(cap, step) if n else None
            has_pi = False
            if post is not None:
                p = np.asarray(post, dtype=np.float32).reshape(-1)[:n]
                s = float(p.sum())
                if s > 0:
                    pi[:len(p)] = p / s
                    has_pi = True
            if step < len(cap["z"]):
                z, ok = float(cap["z"][step]), True
            elif cap["result"] is not None:
                z, ok = float(cap["result"]), True
            else:
                z, ok = 0.0, False
            sv = step_search_value(cap, step)
            sv_rows.append(np.nan if sv is None else float(sv))
            index[(cap["gn"], step)] = len(obs_rows)
            obs_rows.append(o)
            pi_rows.append(pi)
            mask_rows.append(mask)
            z_rows.append(z)
            z_ok.append(ok)
            pi_ok.append(has_pi)
    if not obs_rows:
        return None, {}, None, None
    sample = {"obs": np.stack(obs_rows), "pi": np.stack(pi_rows),
              "mask": np.stack(mask_rows),
              "z": np.asarray(z_rows, dtype=np.float32),
              "search_v": np.asarray(sv_rows, dtype=np.float32)}
    return (sample, index, np.asarray(z_ok, dtype=bool),
            np.asarray(pi_ok, dtype=bool))


def _no_posterior_note(n_skipped, n_total):
    """The footer naming rows a π-dependent view left out."""
    if not n_skipped:
        return []
    return [f"({n_skipped} of {n_total} browsed decisions skipped: no search "
            "posterior — raw-policy, human / behavior, or fast-search rows)"]


def _subsample(sample, keep_mask=None, limit=_MAX_POOL_ROWS, seed=0):
    """``(sub_sample, rows)``: the rows ``keep_mask`` keeps, randomly capped
    at ``limit`` (None = no cap); ``rows`` maps each sub-sample row back to
    its full-sample row."""
    idx = np.arange(sample["obs"].shape[0])
    if keep_mask is not None:
        idx = idx[keep_mask]
    if limit is not None and len(idx) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(
            idx, size=limit, replace=False))
    return {k: v[idx] for k, v in sample.items()}, idx


def _row_labels(index, rows):
    """Sub-sample row -> "game G step S" (the browsers' game / step indices)
    for the pooled views' per-decision listings."""
    where = {row: (gn, step) for (gn, step), row in index.items()}
    return {i: f"game {where[r][0]} step {where[r][1]}"
            for i, r in enumerate(rows) if r in where}


def search_stats_lines(snap):
    """One-line tally of the searches behind the browsed decisions: roots
    searched in-game / at bo3 sideboard roots, tree-followed decisions,
    decisions with no search, and the sims / sim steps the searches ran."""
    from shard_record import (DIAG_KIND_FOLLOWED, DIAG_KIND_PLAN,
                              DIAG_KIND_SEARCH)
    counts = {DIAG_KIND_SEARCH: 0, DIAG_KIND_PLAN: 0, DIAG_KIND_FOLLOWED: 0}
    none = sims = steps = 0
    for cap in snap["games"]:
        diags = cap["diag"]
        for step in range(len(cap["observations"])):
            d = diags[step] if step < len(diags) else None
            if d is None or d.get("kind") not in counts:
                none += 1
                continue
            counts[d["kind"]] += 1
            sims += int(d.get("sims_run") or 0)
            steps += int(d.get("sim_steps") or 0)
    return [f"searches: {counts[DIAG_KIND_SEARCH]} in-game, "
            f"{counts[DIAG_KIND_PLAN]} bo3 sideboard, "
            f"{counts[DIAG_KIND_FOLLOWED]} tree-followed, {none} without a "
            f"search; {sims} sims, {steps} sim steps"]


def run_probe(key, net, snap, limit=_MAX_POOL_ROWS):
    """Run one PROBE_MENU view over a snapshot; returns display lines.
    ``limit`` caps the rows a pooled view samples (None = every row — the
    batch report's setting)."""
    import az_inspect as azi
    sample, index, z_valid, pi_valid = build_sample(snap)
    if sample is None:
        return ["no browsable decisions yet"]

    if key in DECISION_PROBES:
        gn, step = snap["sel"]
        row = index.get((gn, step))
        if row is None:
            return ["select a game and step first"]
        obs_row, mask_row = sample["obs"][row], sample["mask"][row]
        if key == "probe_state":
            return azi.render_state(sample, row, net=net,
                                    has_pi=bool(pi_valid[row]))
        if key in ("probe_blocks", "probe_blocks_pi"):
            imp = azi.state_block_importance(net, sample, row, donors=16)
            sort = "pi" if key == "probe_blocks_pi" else "v"
            return (azi.render_block_importance(imp, top_n=20, single=True,
                                                sort=sort)
                    + ["", f"(donor states drawn from the {sample['obs'].shape[0]}"
                           " browsed decisions)"])
        if key == "probe_swap":
            sites = azi.card_id_sites(obs_row)
            if not sites:
                return ["no card-identity sites in this state "
                        "(empty board and hand)"]
            lines = []
            for label, offset, _idx in sites[:_MAX_SWAP_SITES]:
                probe = azi.card_swap_probe(net, obs_row, mask_row, offset)
                lines += azi.render_card_swap(probe, label, top_n=8) + [""]
            if len(sites) > _MAX_SWAP_SITES:
                lines.append(f"({len(sites) - _MAX_SWAP_SITES} more sites not "
                             f"probed — cap {_MAX_SWAP_SITES} per run)")
            return lines
        if key == "probe_readout":
            return azi.render_value_readouts(azi.value_readouts(net, obs_row))
        if key == "probe_sweeps":
            return azi.render_sweeps(net, obs_row, mask_row)

    if key == "probe_kl":
        total = sample["obs"].shape[0]
        n_pi = int(pi_valid.sum())
        if not n_pi:
            return ["no browsed decision carries a search posterior to compare "
                    "against (simulated games need a searching seat, e.g. "
                    "az:gen or mcts:gen)"] + _no_posterior_note(total, total)
        sub, rows = _subsample(sample, pi_valid, limit)
        div = azi.policy_divergence(net, sub)
        return (azi.render_divergence(div) + [""]
                + azi.render_disagreements(sub, div, _TOP_DISAGREEMENTS,
                                           _row_labels(index, rows))
                + ["", f"(over {len(rows)} of {total} browsed decisions with "
                       "a search posterior)"]
                + _no_posterior_note(total - n_pi, total))
    if key == "probe_value":
        has_v = np.isfinite(sample["search_v"])
        if not has_v.any():
            return ["no browsed decision carries a search root value "
                    "(simulated games need a searching seat, e.g. az:gen or "
                    "mcts:gen; tree-followed and pool-shard rows have none)"]
        sub, rows = _subsample(sample, has_v, limit)
        return (azi.render_value_vs_search(azi.value_vs_search(net, sub))
                + search_stats_lines(snap)
                + ["", f"(over {len(rows)} of {sample['obs'].shape[0]} "
                       "browsed decisions with a search root value)"])
    if key == "probe_calib":
        if not z_valid.any():
            return ["no decisions with a known outcome yet"]
        sub, rows = _subsample(sample, z_valid, limit)
        cal = azi.bucket_calibration(net, sub)
        return azi.render_calibration(cal) + [
            "", f"(over {len(rows)} decisions with a realized outcome; simulated "
                "traces use the match result as z)"]
    return [f"unknown probe {key!r}"]
