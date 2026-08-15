"""Net-probe views for the analysis browser — az_inspect over browsed records.

The browser (gui_browser / tui_analysis) pages per-game trace dicts; az_inspect
owns the "what informed this evaluation" probes (single-state block permutation
importance, card-identity swap, scalar sweeps, recorded-π-vs-net state detail)
and the pooled net-vs-search views (KL divergence by action category, per-bucket
value calibration). This module is the glue between the two: it snapshots the
browser's games into az_inspect's flat sample-dict shape (obs/pi/mask/z + a
(game, step) -> row index) and runs each view, returning the display lines the
front end appends to its output pane.

Two-phase by design, matching the browser's threading contract:
:func:`snapshot` runs on the UI thread and only takes cheap list() copies of
the per-game step lists (the ndarray rows themselves are append-only, never
mutated); :func:`build_sample` and :func:`run_probe` run on a worker thread
and do the stacking and the torch work. The probe net comes from
:func:`load_probe_net` — ``opponents._load_az_evaluator``'s AZNet (the AZ
checkpoint when one exists, else the PPO warm-start), so the probes work with
only a PPO ``gen`` trained.

Per-row z: shard-mode records carry an exact per-step game outcome (the shard's
``z`` column, preserved by shard_replay); simulated traces don't, so their rows
fall back to the game's match-level result — close enough for calibration, and
rows with no outcome at all (a live game) are excluded from it.

Qt-free and torch-free at import (torch loads inside the net/probe calls).
"""

import numpy as np

from env import MAX_ACTIONS, OBS_SIZE

# Menu entries the front end appends after the engine section: (key, label).
# The first four probe THE SELECTED DECISION; the last two pool every browsed
# decision that has the needed data.
PROBE_MENU = [
    ("probe_state", "net state — recorded π vs net at current step"),
    ("probe_blocks", "net blocks — what this evaluation rests on"),
    ("probe_swap", "net swap — card-identity ΔV at current step"),
    ("probe_sweeps", "net sweeps — life/hand/turn response here"),
    ("probe_kl", "net KL — search vs net by action category (all)"),
    ("probe_calib", "net calib — V vs realized z by bucket (all)"),
]
PROBE_KEYS = frozenset(k for k, _ in PROBE_MENU)
DECISION_PROBES = frozenset(
    ("probe_state", "probe_blocks", "probe_swap", "probe_sweeps"))

_MAX_POOL_ROWS = 2000     # cap for the pooled views / donor pool
_MAX_SWAP_SITES = 6       # card-swap sites probed per decision (cost control)


def probe_model_spec(model_spec):
    """The _load_az_evaluator base for a browser model spec: strip the ?knob
    query and any az:/azraw:/mcts: wrapper prefix."""
    base = (model_spec or "gen").split("?", 1)[0].strip()
    low = base.lower()
    for pre in ("az:", "azraw:", "mcts:"):
        if low.startswith(pre):
            return base[len(pre):] or "gen"
    return base or "gen"


def load_probe_net(model_spec):
    """(AZNet, label) for the probes. Raises with a readable message when
    torch / a checkpoint is unavailable (the front end shows it verbatim)."""
    from opponents import _load_az_evaluator
    evaluator, resolved = _load_az_evaluator(probe_model_spec(model_spec))
    net = getattr(evaluator, "_net", None)   # AZEvaluator holds the AZNet
    if net is None:
        raise RuntimeError(
            f"evaluator for {model_spec!r} exposes no net — cannot probe")
    return net, str(resolved)


def snapshot(games, cur_game=None, cur_step=None):
    """UI-thread capture: shallow list() copies of each game's step data plus
    the current selection. Cheap — no array copies, no stacking."""
    caps = []
    for gn, g in enumerate(games):
        caps.append({
            "gn": gn,
            "observations": list(g.get("observations") or ()),
            "num_choices": list(g.get("num_choices") or ()),
            "action_probs": list(g.get("action_probs") or ()),
            "z": list(g.get("z") or ()),
            "result": g.get("result"),
        })
    return {"games": caps, "sel": (cur_game, cur_step)}


def build_sample(snap):
    """Stack a snapshot into the az_inspect sample-dict shape.

    Returns ``(sample, index, z_valid)``: ``sample`` has obs/pi/mask/z (z is 0
    where unknown), ``index`` maps (game, step) -> row, ``z_valid`` marks rows
    whose z is a real outcome (shard z, or a finished game's result)."""
    obs_rows, pi_rows, mask_rows, z_rows, z_ok, index = [], [], [], [], [], {}
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
            probs = (cap["action_probs"][step]
                     if step < len(cap["action_probs"]) else None)
            if probs is not None and n:
                p = np.asarray(probs, dtype=np.float32)[:n]
                s = float(p.sum())
                if s > 0:
                    pi[:n] = p / s
            if step < len(cap["z"]):
                z, ok = float(cap["z"][step]), True
            elif cap["result"] is not None:
                z, ok = float(cap["result"]), True
            else:
                z, ok = 0.0, False
            index[(cap["gn"], step)] = len(obs_rows)
            obs_rows.append(o)
            pi_rows.append(pi)
            mask_rows.append(mask)
            z_rows.append(z)
            z_ok.append(ok)
    if not obs_rows:
        return None, {}, None
    sample = {"obs": np.stack(obs_rows), "pi": np.stack(pi_rows),
              "mask": np.stack(mask_rows),
              "z": np.asarray(z_rows, dtype=np.float32)}
    return sample, index, np.asarray(z_ok, dtype=bool)


def _subsample(sample, keep_mask=None, limit=_MAX_POOL_ROWS, seed=0):
    idx = np.arange(sample["obs"].shape[0])
    if keep_mask is not None:
        idx = idx[keep_mask]
    if len(idx) > limit:
        idx = np.sort(np.random.default_rng(seed).choice(
            idx, size=limit, replace=False))
    return {k: v[idx] for k, v in sample.items()}, len(idx)


def run_probe(key, net, snap):
    """Run one PROBE_MENU view over a snapshot; returns display lines."""
    import az_inspect as azi
    sample, index, z_valid = build_sample(snap)
    if sample is None:
        return ["no browsable decisions yet"]

    if key in DECISION_PROBES:
        gn, step = snap["sel"]
        row = index.get((gn, step))
        if row is None:
            return ["select a game and step first"]
        obs_row, mask_row = sample["obs"][row], sample["mask"][row]
        if key == "probe_state":
            return azi.render_state(sample, row, net=net)
        if key == "probe_blocks":
            imp = azi.state_block_importance(net, sample, row, donors=16)
            return (azi.render_block_importance(imp, top_n=20, single=True)
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
        if key == "probe_sweeps":
            return azi.render_sweeps(net, obs_row, mask_row)

    if key == "probe_kl":
        has_pi = sample["pi"].sum(axis=1) > 0
        if not has_pi.any():
            return ["no recorded action-probability data to compare against"]
        sub, n = _subsample(sample, has_pi)
        div = azi.policy_divergence(net, sub)
        return azi.render_divergence(div) + [
            "", f"(over {n} of {sample['obs'].shape[0]} browsed decisions "
                "with a recorded posterior)"]
    if key == "probe_calib":
        if not z_valid.any():
            return ["no decisions with a known outcome yet"]
        sub, n = _subsample(sample, z_valid)
        cal = azi.bucket_calibration(net, sub)
        return azi.render_calibration(cal) + [
            "", f"(over {n} decisions with a realized outcome; simulated "
                "traces use the match result as z)"]
    return [f"unknown probe {key!r}"]
