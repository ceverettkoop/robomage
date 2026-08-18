"""AlphaZero trainer + gating for RoboMage (Phase C).

- ``train_az`` : load the last M self-play shards from the pooled ``az_data/gen``
  window, optimize the one generalist AZNet with Adam (constant lr, weight_decay
  1e-4) on the AZ loss ``CE(pi, masked_log_softmax(logits)) + c_v * MSE(value,
  z)``, snapshot the CANDIDATE to ``gen__azv{steps}.pt``. ``gen__azfinal.pt`` (the
  incumbent) is written ONLY by the ``az_eval`` promotion gate — training never
  touches it.
- ``az_eval`` : candidate-vs-incumbent gating via ``run_match`` with ``az``
  controllers at low sims over a ROSTER-WIDE panel (a mirror for every roster
  deck, so every deck the net must pilot is measured, plus direction-balanced
  cross pairings), seats alternating (half the games each way); promote the
  candidate to ``gen__azfinal.pt`` on aggregate win-rate, subject to a per-deck
  floor veto (a candidate that collapsed at piloting any one deck is rejected
  even when its aggregate clears the bar), printing a per-matchup and
  per-piloted-deck breakdown.
- ``az_cycle`` : one sequential generate -> train -> eval iteration.

Exposed to ``train.py`` as the ``az-train`` / ``az-eval`` / ``az`` subcommands.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time
from typing import Optional

# Shared crash-safe sidecar IO (write-to-temp + os.replace), same writer the PPO
# league / exploiter / curriculum drivers use.
from progress_io import write_progress_state, read_progress_state
# The bo3 sideboard-root budget has ONE home (cli_spec). These were hardcoded
# literals here, which silently pinned this module to the old values whenever the
# arg was absent — import them so a change to the default actually reaches the
# az / az-league paths.
from cli_spec import (DEFAULT_SB_SIMS, DEFAULT_SB_WORLDS, DEFAULT_SB_MAX_DEPTH,
                      DEFAULT_SB_ROLLOUT_TURNS, DEFAULT_SB_PERSIST,
                      DEFAULT_AZ_GAMES, DEFAULT_AZ_SIMS, DEFAULT_AZ_WORLDS,
                      DEFAULT_AZ_MIRROR_FRAC, DEFAULT_AZ_LR,
                      DEFAULT_AZ_WEIGHT_DECAY, DEFAULT_AZ_BATCH_SIZE,
                      DEFAULT_AZ_TRAIN_BATCHES, DEFAULT_AZ_CYCLE_BATCHES,
                      DEFAULT_AZ_WINDOW, DEFAULT_AZ_CV,
                      DEFAULT_AZ_TD_N, DEFAULT_AZ_Q_MIX,
                      DEFAULT_AZ_EVAL_GAMES, DEFAULT_AZ_EVAL_SIMS,
                      DEFAULT_AZ_EVAL_WORLDS, DEFAULT_AZ_PROMOTE_THRESHOLD,
                      DEFAULT_AZ_GATE_FLOOR, DEFAULT_AZ_GATE_FLOOR_MIN,
                      DEFAULT_AZ_GATE_CROSS_PAIRS, DEFAULT_AZ_GATE_EVERY,
                      DEFAULT_AZ_EXPERT_GAMES, DEFAULT_AZ_EXHAUSTIVE_REPEATS,
                      DEFAULT_AZ_SCRIPTED_CELLS,
                      EXPERT_DECKS_ROSTER, EXPERT_DECKS_NONE)

import numpy as np

try:
    from env import OBS_SIZE, MAX_ACTIONS
    from opponents import GEN_STEM, assert_not_reserved_deck
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS
    from train.opponents import GEN_STEM, assert_not_reserved_deck

_AZ_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "az")
_AZ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "az_data")
_DECKS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "bin", "resources", "decks")
_LEAGUE_DECKS_DIR = os.path.join(_DECKS_DIR, "league")

# P(opponent deck == focus deck) per self-play game (mirror vs cross-deck roster).
# Value lives in cli_spec's AZ-defaults block (one home); local alias kept for
# the existing internal references.
DEFAULT_MIRROR_FRAC = DEFAULT_AZ_MIRROR_FRAC

# Promotion-gate defaults. The gate panel is ROSTER-WIDE: a candidate-vs-incumbent
# mirror for every roster deck (so a piloting regression on ANY deck shows up in
# the aggregate — the old focus-only gate could not see one), plus
# DEFAULT_GATE_CROSS_PAIRS cross pairings each played in BOTH directions so a
# lopsided deck matchup cancels out of the aggregate. The per-deck floor veto is
# a LIKE-PAIRING comparison (see az_eval): a piloted deck with >=
# DEFAULT_GATE_FLOOR_MIN candidate matches is vetoed when the candidate's pooled
# win-rate sits more than (1 - 2*DEFAULT_GATE_FLOOR) below the INCUMBENT's
# win-rate piloting the same deck on the same pairings — on mirrors alone that
# is exactly "candidate mirror win-rate < DEFAULT_GATE_FLOOR", so the intended
# trigger stays a 0-for-4 wipeout; a lopsided cross matchup the incumbent also
# loses no longer counts against the deck. Deliberate: per-deck samples are
# tiny, so the floor is a catastrophic-collapse tripwire, not a fine measure.
# Values live in cli_spec's AZ-defaults block (one home); local aliases kept
# for the existing internal references.
DEFAULT_GATE_FLOOR = DEFAULT_AZ_GATE_FLOOR
DEFAULT_GATE_FLOOR_MIN = DEFAULT_AZ_GATE_FLOOR_MIN
DEFAULT_GATE_CROSS_PAIRS = DEFAULT_AZ_GATE_CROSS_PAIRS

# az-league gates (az_eval + promotion) run every DEFAULT_GATE_EVERY slots. One
# slot of training between gates is a small weight delta against a >=55%
# aggregate bar, and each gate costs real wall-clock (56 searched matches), so
# gating every K>1 slots lets the candidate accumulate K cycles of training —
# and K slots of CANDIDATE-generated self-play (resolve_source prefers the
# snapshot line) — before paying for an eval. Promotion cadence, not training,
# is all that changes: candidate snapshots still save every slot.
DEFAULT_GATE_EVERY = DEFAULT_AZ_GATE_EVERY


# ----------------------------------------------------------------------
# Shard loading
# ----------------------------------------------------------------------

def load_window(deck: str, window: int, data_dir: Optional[str] = None) -> dict:
    """Load the last ``window`` shards (by mtime) into flat arrays.

    Shards pool into ``az_data/gen/`` — self-play across every focus deck feeds
    the ONE generalist net — so ``deck`` is used only for the error message; the
    default ``data_dir`` is the shared gen pool. Pass ``data_dir`` to override
    (tests point it at a temp dir).

    Returns a dict of the :data:`az_selfplay.SHARD_KEYS` arrays plus
    ``n_shards``. Every key is REQUIRED: a shard predating the n-step TD schema
    (no ``td_q`` column) is a hard error, because silently training such a shard
    would mix two different value targets in one window."""
    from az_selfplay import SHARD_KEYS
    data_dir = data_dir or os.path.join(_AZ_DATA_DIR, GEN_STEM)
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")),
                    key=os.path.getmtime)
    if not shards:
        raise FileNotFoundError(
            f"no self-play shards in {data_dir} — run az-selfplay first")
    shards = shards[-window:]
    parts = {k: [] for k in SHARD_KEYS}
    for s in shards:
        d = np.load(s)
        missing = [k for k in SHARD_KEYS if k not in d.files]
        if missing:
            raise RuntimeError(
                f"self-play shard {s} is missing the column(s) "
                f"{', '.join(missing)} — regenerate self-play shards (schema "
                f"changed: n-step TD targets). Delete {data_dir}/shard_*.npz and "
                f"re-run az-selfplay / the az cycle.")
        for k in SHARD_KEYS:
            parts[k].append(d[k])
    out = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    # Shards are raw observation rows, so an obs-layout change (e.g. a new tail
    # block) makes older shards unusable. Say so instead of letting the net's
    # first slice fail with a bare shape error deep in the forward pass.
    if out["obs"].shape[1] != OBS_SIZE or out["mask"].shape[1] != MAX_ACTIONS:
        raise RuntimeError(
            f"self-play shards in {data_dir} were recorded against a different "
            f"observation layout (obs width {out['obs'].shape[1]}, mask "
            f"{out['mask'].shape[1]}) but this build has OBS_SIZE={OBS_SIZE}, "
            f"MAX_ACTIONS={MAX_ACTIONS} — "
            "delete/regenerate the shards (re-run az-selfplay) before training")
    out["n_shards"] = len(shards)
    return out


def prune_shards(window: int, data_dir: Optional[str] = None) -> int:
    """Delete self-play shards that neither the current nor the PREVIOUS
    training window would read: everything older than the newest ``2*window``
    shards (by mtime, matching :func:`load_window`'s ordering). Called by
    :func:`train_az` after each training pass so a long-running az-league
    (``rotations=0``) doesn't grow ``az_data/gen`` without bound. Returns the
    number of shards deleted."""
    keep = 2 * int(window)
    if keep <= 0:
        return 0
    data_dir = data_dir or os.path.join(_AZ_DATA_DIR, GEN_STEM)
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")),
                    key=os.path.getmtime)
    pruned = 0
    for s in shards[:-keep]:
        try:
            os.remove(s)
            pruned += 1
        except OSError as exc:
            print(f"[az-train] WARNING: could not prune shard {s}: {exc}")
    return pruned


# ----------------------------------------------------------------------
# Net init / resume
# ----------------------------------------------------------------------

def _init_net(from_ppo: Optional[str], fresh: bool):
    """Return (net, prior_steps, provenance-string) for the ONE generalist AZ net.

    Every slot/cycle trains THE gen net, so init resolves the gen checkpoints, not
    a per-deck file.

    Deliberately NOT the shared ladder ``opponents.load_az_evaluator``: this is
    the TRAINING contract, with different policy at every rung — it continues the
    CANDIDATE line (``prefer="snapshot"``), honours an explicit ``--fresh`` gate,
    has a random-init rung, reports resumed STEP COUNTS and a provenance string,
    and returns a bare net rather than an AZEvaluator. Keep the two in sync in
    SPIRIT (AZ checkpoint -> PPO warm-start -> …), not in code."""
    from az_net import (AZNet, load_az, from_ppo as warm_from_ppo,
                        resolve_az_checkpoint)
    from opponents import resolve_checkpoint

    if not fresh:
        # Continue the CANDIDATE line: newest gen__azv snapshot first, gen__azfinal
        # (the gate-promoted incumbent) only as a fallback. This keeps training
        # cumulative across cycles even while the gate rejects candidates.
        az = resolve_az_checkpoint(GEN_STEM, prefer="snapshot")
        if az:
            net = load_az(az)
            steps = _read_steps(az)
            return net, steps, f"resumed AZ checkpoint {az} (steps={steps})"
    if from_ppo:
        path = resolve_checkpoint(from_ppo)
        return warm_from_ppo(path), 0, f"warm-started from PPO {path}"
    if not fresh:
        # No AZ checkpoint yet: default to warm-starting the gen PPO generalist.
        ppo = resolve_checkpoint(GEN_STEM)
        if ppo and os.path.exists(ppo):
            return warm_from_ppo(ppo), 0, f"warm-started from gen PPO {ppo}"
    return AZNet().eval(), 0, "fresh random init"


def _read_steps(az_path: str) -> int:
    import json
    from az_net import _meta_path
    mp = _meta_path(az_path)
    if os.path.exists(mp):
        with open(mp) as f:
            return int(json.load(f).get("steps", 0))
    return 0


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_az(deck: str, *, batches: int = DEFAULT_AZ_TRAIN_BATCHES,
             batch_size: int = DEFAULT_AZ_BATCH_SIZE,
             lr: float = DEFAULT_AZ_LR, c_v: float = DEFAULT_AZ_CV,
             q_mix: float = DEFAULT_AZ_Q_MIX,
             window: int = DEFAULT_AZ_WINDOW,
             weight_decay: float = DEFAULT_AZ_WEIGHT_DECAY,
             from_ppo: Optional[str] = None,
             fresh: bool = False, log_every: int = 50,
             snapshot_every: int = 0, data_dir: Optional[str] = None,
             ckpt_dir: str = _AZ_CKPT_DIR, seed: int = 0) -> dict:
    """Optimize the gen AZNet on the newest ``window`` shards.

    ``batches <= 0`` resolves to AUTO — ``max(1, n_samples // batch_size)``
    optimizer updates, i.e. exactly one epoch over the loaded window — and the
    resolution is printed (``batches=auto(one-epoch)->534``). That is the az /
    az-league cycle default (``DEFAULT_AZ_CYCLE_BATCHES``); the standalone
    az-train default stays a fixed batch count.

    The value target is the MIX ``(1 - q_mix) * z + q_mix * td_q``: the shard's
    per-game outcome blended with its recorded n-step TD bootstrap (see
    :func:`az_selfplay.compute_td_targets`). ``q_mix=0`` restores the classic
    pure-outcome AlphaZero target. The periodic log line reports the value loss
    against BOTH the mixed target and the plain outcome, so a drift between them
    is visible while training."""
    import torch
    import torch.nn.functional as F
    from az_net import az_checkpoint_path, decay_exempt_param_groups

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    w = load_window(deck, window, data_dir)
    obs, pi, z, mask = w["obs"], w["pi"], w["z"], w["mask"]
    td_q, n_shards = w["td_q"], w["n_shards"]
    n = obs.shape[0]
    # Both backends already resolve every unbootstrappable row to z, so a NaN
    # here means a shard was written by something that does not honour the
    # schema — refuse it rather than propagate NaN through the value head.
    if not np.all(np.isfinite(td_q)):
        raise RuntimeError(
            f"{int((~np.isfinite(td_q)).sum())} of {n} shard rows have a "
            f"non-finite td_q — every writer must resolve an unbootstrappable "
            f"sample to its outcome z (regenerate the shards)")
    q_mix = float(q_mix)
    v_target = ((1.0 - q_mix) * z + q_mix * td_q).astype(np.float32)
    # batches <= 0 = AUTO: exactly ONE epoch over the loaded window. A fixed
    # batch count silently drifts in epochs as the window's data volume changes
    # (1000 x 256 over a ~137k-sample window was ~1.9 epochs of re-fitting one
    # small, correlated window); one pass is the same amount of fitting whatever
    # the volume.
    bs = min(batch_size, n)
    batches_txt = str(batches)
    if batches <= 0:
        batches = max(1, n // bs)
        batches_txt = f"auto(one-epoch)->{batches}"
    print(f"[az-train] gen (focus={deck}): {n} samples from {n_shards} shards; "
          f"batches={batches_txt} batch_size={batch_size} lr={lr} c_v={c_v} "
          f"q_mix={q_mix}")
    print(f"[az-train] value target: (1-q_mix)*z + q_mix*td_q; "
          f"td_q != z on {int((td_q != z).sum())}/{n} rows "
          f"(mean |td_q - z| = {float(np.abs(td_q - z).mean()):.4f})")

    net, prior_steps, prov = _init_net(from_ppo, fresh)
    print(f"[az-train] net: {prov}")
    net.train()
    # Weight decay applies to everything EXCEPT the per-sample-selected
    # parameters (the critic's archetype-bucket columns, the embedding tables).
    # Those get a task gradient only for the rows a batch actually uses, and
    # Adam's normalization turns decay-without-gradient into a ~lr/step march to
    # zero — so a single-matchup cycle would erase every other matchup's critic
    # column (and every card absent from the window) in ~100 batches.
    opt = torch.optim.Adam(decay_exempt_param_groups(net, weight_decay), lr=lr)

    obs_t = torch.as_tensor(obs)
    pi_t = torch.as_tensor(pi)
    z_t = torch.as_tensor(z)
    vt_t = torch.as_tensor(v_target)
    mask_t = torch.as_tensor(mask)

    log_path = os.path.join(ckpt_dir, f"{GEN_STEM}_az_train.log")
    os.makedirs(ckpt_dir, exist_ok=True)
    first_loss = None
    last_loss = None

    with open(log_path, "a") as logf:
        logf.write(f"# az-train {time.strftime('%Y-%m-%d %H:%M:%S')} deck={deck} "
                   f"samples={n} batches={batches} bs={bs} lr={lr}\n")
        for b in range(batches):
            idx = rng.integers(0, n, size=bs)
            bi = torch.as_tensor(idx)
            ob = obs_t[bi]; tp = pi_t[bi]; tz = z_t[bi]; mk = mask_t[bi]
            tv = vt_t[bi]

            logits, value = net(ob, mk)
            logp = F.log_softmax(logits, dim=-1)
            logp = torch.where(mk, logp, torch.zeros_like(logp))   # kill -inf*0 nan
            loss_pi = -(tp * logp).sum(dim=1).mean()
            # Optimized against the MIXED target; the pure-outcome MSE is
            # computed alongside (no grad) purely as a comparable diagnostic.
            loss_v = F.mse_loss(value, tv)
            with torch.no_grad():
                loss_vz = F.mse_loss(value, tz)
            loss = loss_pi + c_v * loss_v

            opt.zero_grad()
            loss.backward()
            opt.step()

            fl = float(loss.item())
            if first_loss is None:
                first_loss = fl
            last_loss = fl
            if b == 0 or (b + 1) % log_every == 0 or b == batches - 1:
                line = (f"[az-train] batch {b+1}/{batches} loss={fl:.4f} "
                        f"(pi={loss_pi.item():.4f} v_mix={loss_v.item():.4f} "
                        f"v_z={loss_vz.item():.4f})")
                print(line)
                logf.write(line + "\n"); logf.flush()
            if snapshot_every and (b + 1) % snapshot_every == 0:
                steps = prior_steps + (b + 1) * bs
                net.save(az_checkpoint_path(steps, ckpt_dir), steps)

    steps = prior_steps + batches * bs
    net.eval()
    # Candidate snapshot only — gen__azfinal (the incumbent) advances exclusively
    # through the az_eval promotion gate.
    snap = net.save(az_checkpoint_path(steps, ckpt_dir), steps)
    print(f"[az-train] saved candidate snapshot {snap}")
    print(f"[az-train] loss {first_loss:.4f} -> {last_loss:.4f} over {batches} batches")
    pruned = prune_shards(window, data_dir)
    if pruned:
        print(f"[az-train] pruned {pruned} stale shard(s) older than the last "
              f"2x{window}-shard window")
    return {"samples": n, "first_loss": first_loss, "last_loss": last_loss,
            "snapshot": snap, "steps": steps}


# ----------------------------------------------------------------------
# Gating (candidate vs incumbent)
# ----------------------------------------------------------------------

def _normalize_focus(deck, default_roster) -> list:
    """Coerce a focus argument (None / str / list) into a non-empty deck list.
    ``None`` or an empty list falls back to ``default_roster`` (the whole
    decks/league/ roster)."""
    if deck is None:
        return list(default_roster)
    if isinstance(deck, (list, tuple)):
        return list(deck) or list(default_roster)
    return [deck]


def _gate_matchups(focus_decks, roster, cross_pairs: int, seed: int) -> list:
    """The ROSTER-WIDE matchup panel the gate plays: a candidate-vs-incumbent
    MIRROR for every deck in focus + roster (de-duplicated, order-preserving), so
    every deck the one gen net must pilot is measured every gate — a regression
    on a non-focus deck is visible, which the old focus-only sample was blind to.
    Plus ``cross_pairs`` seeded cross pairings, each added in BOTH directions
    ((x, y) AND (y, x)): the candidate always pilots deck_x, so an unpaired cross
    matchup would credit/penalize the candidate for raw deck strength (a 90-10
    deck matchup swamps any net difference); playing both directions cancels
    that out of the aggregate. Each entry is a (deck_x, deck_y) pair; the
    candidate pilots deck_x and the incumbent deck_y, seats alternating within
    the pair."""
    decks = []
    for d in list(focus_decks or []) + list(roster or []):
        if d not in decks:
            decks.append(d)
    matchups = [(d, d) for d in decks]
    if len(decks) >= 2 and cross_pairs > 0:
        rng = np.random.default_rng(seed)
        for _ in range(int(cross_pairs)):
            i, j = rng.choice(len(decks), size=2, replace=False)
            matchups += [(decks[int(i)], decks[int(j)]),
                         (decks[int(j)], decks[int(i)])]
    seen, out = set(), []
    for m in matchups:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _like_pairing_floor(per_matchup: dict, gate_floor: float,
                        floor_min_matches: int) -> tuple:
    """The per-deck floor veto, computed on LIKE PAIRINGS (see az_eval's
    docstring). ``per_matchup`` maps (deck_x, deck_y) -> [w, l, d] from the
    CANDIDATE's view piloting deck_x. For each piloted deck, pool the
    candidate's record against the incumbent's record piloting the SAME deck on
    the same pairings, and veto on a pooled win-rate deficit below
    ``2*gate_floor - 1``. The incumbent's record needs no extra games: a mirror
    is candidate-vs-incumbent on the same deck already (flip it), and a cross
    pairing's reverse direction — which _gate_matchups always schedules — has
    the incumbent piloting this deck (flip that). A cross pairing whose reverse
    is missing is skipped rather than read one-sided. Returns
    (vetoes, deck_floor) with deck_floor[deck] = (cand_wr, inc_wr, n_cand)."""
    vetoes: list = []
    deck_floor: dict = {}
    decks = {dx for (dx, _dy) in per_matchup}
    for pd in decks:
        cw = cn = iw = inn = 0
        for (dx, dy), (mw, ml, md) in per_matchup.items():
            if dx != pd:
                continue
            n = mw + ml + md
            if n == 0:
                continue
            if dx == dy:
                cw += mw; cn += n
                iw += ml; inn += n          # incumbent-as-pd = flip of the mirror
            elif (dy, dx) in per_matchup:
                rw, rl, rd = per_matchup[(dy, dx)]
                rn = rw + rl + rd
                if rn == 0:
                    continue
                cw += mw; cn += n
                iw += rl; inn += rn         # incumbent-as-pd = flip of the reverse
        if cn == 0 or inn == 0:
            continue
        cand_wr, inc_wr = cw / cn, iw / inn
        deck_floor[pd] = (cand_wr, inc_wr, cn)
        if gate_floor > 0 and cn >= floor_min_matches and \
                cand_wr - inc_wr < 2 * gate_floor - 1:
            vetoes.append(pd)
    return vetoes, deck_floor


def _gate_worker_init():
    """Gate-pool worker initializer: pin the batch-1 CPU net eval to one math
    thread. Gate parallelism comes from the process fan-out (each worker already
    alternates with its own engine subprocess); letting torch fan tiny matmuls
    across cores would just oversubscribe the pool."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _gate_matchup_worker(mi: int, dx: str, dy: str, per: int, cand_spec: str,
                         opp_spec: str, bo3: bool, seed: int) -> tuple:
    """One gate matchup — ``per`` matches of candidate-piloting-``dx`` vs
    opponent-piloting-``dy``, seats split half/half — returning
    ``(mi, mw, ml, md)`` from the candidate's view.

    Controllers are built FRESH from the specs by each ``run_match`` call
    (never cached across matchups): a controller's internal rng advances as it
    searches, so reuse would make a matchup's result depend on which matchups
    ran before it on the same worker. Fresh construction plus seeds derived
    only from ``mi`` make every matchup's result identical to the serial
    loop's, regardless of worker count or scheduling."""
    from runner import run_match
    from cli_spec import INTERACTIVE_BINARY
    half = per // 2
    mw = ml = md = 0
    mseed = seed + mi * 100003
    if per - half:  # candidate (piloting dx) in seat A
        r = run_match(cand_spec, opp_spec, deck_a=dx, deck_b=dy,
                      games=per - half, bo3=bo3, seed=mseed, transcript="quiet",
                      binary_path=INTERACTIVE_BINARY)
        mw += r.wins; ml += r.losses; md += r.draws
    if half:        # candidate (piloting dx) in seat B — flip tally to cand view
        r = run_match(opp_spec, cand_spec, deck_a=dy, deck_b=dx,
                      games=half, bo3=bo3, seed=mseed + per, transcript="quiet",
                      binary_path=INTERACTIVE_BINARY)
        mw += r.losses; ml += r.wins; md += r.draws
    return mi, mw, ml, md


def az_eval(deck, candidate: str, incumbent: Optional[str] = None, *,
            games: int = DEFAULT_AZ_EVAL_GAMES, sims: int = DEFAULT_AZ_EVAL_SIMS,
            worlds: int = DEFAULT_AZ_EVAL_WORLDS, c_puct: float = 1.5,
            sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
            sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
            sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
            sb_persist: int = DEFAULT_SB_PERSIST,
            promote_threshold: float = DEFAULT_AZ_PROMOTE_THRESHOLD,
            promote: bool = False,
            roster: Optional[list] = None,
            cross_pairs: int = DEFAULT_GATE_CROSS_PAIRS,
            gate_floor: float = DEFAULT_GATE_FLOOR,
            floor_min_matches: int = DEFAULT_GATE_FLOOR_MIN,
            ckpt_dir: str = _AZ_CKPT_DIR, seed: int = 1, bo3: bool = True,
            workers: Optional[int] = None) -> dict:
    """ROSTER-WIDE gate: play ``candidate`` vs ``incumbent`` over a panel of
    matchups covering EVERY deck in ``roster`` (default: the whole decks/league/
    roster) as pilot — a mirror per deck plus direction-balanced cross pairings
    (see :func:`_gate_matchups`) — seats alternating. With ``bo3`` (default) each
    contest is a best-of-three MATCH and the gate is decided on the aggregate
    MATCH win-rate. Promote the candidate to ``gen__azfinal.pt`` when
    ``promote``, the AGGREGATE win-rate across all matchups >=
    ``promote_threshold``, AND no per-deck floor veto fires.

    The floor veto is a LIKE-PAIRING comparison: for each piloted deck the
    candidate's win-rate is measured against the INCUMBENT's win-rate piloting
    the same deck on the same pairings (mirror: the incumbent's record is the
    flip of the candidate's; cross pairing (x, y): the incumbent-as-x record is
    the flip of the reverse pairing (y, x), which the panel always plays). A
    deck with >= ``floor_min_matches`` candidate matches whose pooled win-rate
    deficit vs the incumbent falls below ``2*gate_floor - 1`` blocks promotion
    (``gate_floor=0`` disables the veto). On mirrors alone this reduces to the
    old absolute rule (candidate mirror win-rate < ``gate_floor``); the
    difference is that a LOPSIDED deck matchup — where the incumbent piloting
    the same deck loses just as badly — now cancels out of the deck's tally
    instead of counting as candidate collapse (the old absolute floor read one
    side of a 90-10 matchup as a regression). A veto only delays promotion by a
    gate — training continues from snapshots either way — so a rare unlucky
    veto is much cheaper than promoting a net that forgot how to pilot a deck.
    Prints per-matchup and per-piloted-deck breakdowns. The no-incumbent-yet
    fallback (vs scripted) is preserved (the like-pairing baseline is then the
    scripted opponent piloting the same deck).

    ``workers`` fans the matchup panel out over a process pool (default
    ``max(1, cpu-1)``, capped at the panel size; 1 = the old serial loop). A
    gate match is one driver process ping-ponging with one engine subprocess —
    ~one busy core — so the serial panel left the machine idle. Matchups are
    independent and every matchup's seeds derive only from its panel index
    (and its controllers are built fresh per matchup — see
    :func:`_gate_matchup_worker`), so the aggregate/breakdown/veto results are
    identical for ANY worker count; only per-matchup print order varies."""
    from az_net import az_checkpoint_path, resolve_az_checkpoint

    cand_path = resolve_az_checkpoint(candidate, prefer="snapshot") or candidate
    if incumbent is None:
        incumbent = az_checkpoint_path(None, ckpt_dir)
    inc_path = incumbent if os.path.exists(incumbent) else \
        (resolve_az_checkpoint(incumbent) or incumbent)

    # bo3 gate: sideboard roots get their own (deeper) budget, mirroring self-play.
    knobs = (f"?sims={sims}&worlds={worlds}&c={c_puct}"
             f"&sb_sims={sb_sims}&sb_worlds={sb_worlds}&sb_max_depth={sb_max_depth}"
             f"&sb_rollout_turns={sb_rollout_turns}&sb_persist={int(sb_persist)}")
    cand_spec = f"az:{cand_path}{knobs}"
    have_inc = os.path.exists(inc_path)
    opp_spec = f"az:{inc_path}{knobs}" if have_inc else "scripted"

    roster = list(roster) if roster else _default_az_league_roster()
    focus_decks = _normalize_focus(deck, roster)
    matchups = _gate_matchups(focus_decks, roster, cross_pairs, seed)
    per = max(2, games // len(matchups))   # matches per matchup (>=2 so seats alternate)
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(int(workers), len(matchups)))
    unit = "matches" if bo3 else "games"
    print(f"[az-eval] {len(matchups)} matchup(s) x {per} {unit} "
          f"({'bo3' if bo3 else 'bo1'}, seats alternating, {workers} worker(s)): "
          f"candidate={cand_path} vs "
          f"{'incumbent ' + inc_path if have_inc else 'scripted (no incumbent yet)'} "
          f"@ sims={sims} worlds={worlds}")

    def _tag(dx: str, dy: str) -> str:
        return f"{dx}(mirror)" if dx == dy else f"{dx} vs {dy}"

    # Run the panel: each matchup's result depends only on its index mi (seeds)
    # and the specs (fresh controllers per matchup), so the parallel fan-out is
    # result-identical to the serial loop for any worker count.
    tallies: dict = {}       # mi -> (mw, ml, md), candidate's view
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_gate_worker_init) as ex:
            futs = [ex.submit(_gate_matchup_worker, mi, dx, dy, per, cand_spec,
                              opp_spec, bo3, seed)
                    for mi, (dx, dy) in enumerate(matchups)]
            for fut in as_completed(futs):
                mi, mw, ml, md = fut.result()
                tallies[mi] = (mw, ml, md)
                dx, dy = matchups[mi]
                print(f"[az-eval]   {_tag(dx, dy)}: {mw}W-{ml}L-{md}D "
                      f"({len(tallies)}/{len(matchups)})", flush=True)
    else:
        for mi, (dx, dy) in enumerate(matchups):
            _, mw, ml, md = _gate_matchup_worker(mi, dx, dy, per, cand_spec,
                                                 opp_spec, bo3, seed)
            tallies[mi] = (mw, ml, md)
            print(f"[az-eval]   {_tag(dx, dy)}: {mw}W-{ml}L-{md}D")

    w = l = d = 0
    breakdown = []
    per_deck: dict = {}      # piloted deck -> [w, l, d] from the candidate's view
    per_matchup: dict = {}   # (dx, dy) -> [w, l, d], candidate piloting dx
    for mi, (dx, dy) in enumerate(matchups):
        mw, ml, md = tallies[mi]
        w += mw; l += ml; d += md
        t = per_deck.setdefault(dx, [0, 0, 0])
        t[0] += mw; t[1] += ml; t[2] += md
        per_matchup[(dx, dy)] = [mw, ml, md]
        breakdown.append((_tag(dx, dy), mw, ml, md))

    wr = w / max(1, w + l + d)
    print(f"[az-eval] AGGREGATE candidate {w}W-{l}L-{d}D (win_rate={wr:.3f})")

    vetoes, deck_floor = _like_pairing_floor(per_matchup, gate_floor,
                                             floor_min_matches)
    print("[az-eval] per piloted deck (candidate's view; floor compares vs the "
          "incumbent on like pairings):")
    for pd, (pw, pl, pdr) in per_deck.items():
        n = max(1, pw + pl + pdr)
        flag = "  FLOOR-VETO" if pd in vetoes else ""
        cand_wr, inc_wr, cn = deck_floor.get(pd, (pw / n, float("nan"), 0))
        print(f"[az-eval]   {pd}: {pw}W-{pl}L-{pdr}D ({pw / n:.2f})  "
              f"like-pairing cand {cand_wr:.2f} vs inc {inc_wr:.2f}{flag}")

    promoted = False
    if promote and wr >= promote_threshold and not vetoes:
        final = _promote_to_final(cand_path, ckpt_dir)
        promoted = True
        print(f"[az-eval] PROMOTED candidate -> {final} (>= {promote_threshold:.2f})")
    elif promote and vetoes:
        print(f"[az-eval] not promoted (per-deck floor {gate_floor:.2f} veto: "
              f"{', '.join(vetoes)}; aggregate {wr:.3f})")
    elif promote:
        print(f"[az-eval] not promoted (win_rate {wr:.3f} < {promote_threshold:.2f})")
    return {"wins": w, "losses": l, "draws": d, "win_rate": wr,
            "promoted": promoted, "breakdown": breakdown,
            "per_deck": {k: list(v) for k, v in per_deck.items()},
            "vetoes": vetoes}


def _meta_of(path: str) -> str:
    base, _ = os.path.splitext(path)
    return base + ".meta.json"


def _promote_to_final(cand_path: str, ckpt_dir: str = _AZ_CKPT_DIR) -> str:
    """Copy a candidate snapshot (and its meta sidecar) over ``gen__azfinal.pt``
    — the checkpoint the ``az:gen`` serving/eval specs resolve to. Used by the
    gate on promotion and by an ungated (``gate_every=0``) az-league run at
    completion. Returns the final path."""
    from az_net import az_checkpoint_path
    final = az_checkpoint_path(None, ckpt_dir)
    for src, dst in ((cand_path, final),
                     (_meta_of(cand_path), _meta_of(final))):
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
    return final


# ----------------------------------------------------------------------
# One full cycle: generate -> train -> eval
# ----------------------------------------------------------------------

def _resolve_expert_decks(expert_decks) -> Optional[list]:
    """Resolve an ``--expert-decks`` value to a deck list (or None = no experts).

    The ONE place the ``roster`` / ``none`` sentinels are interpreted, so every
    caller (az cycle, az-league, the resume sidecar) can carry the RAW user value
    around and resolve it at the moment it is consumed:

    * ``roster`` (the default) -> every deck in ``decks/league/``, re-read now, so
      a resumed run picks up the roster of the day;
    * ``none`` / empty / None -> None (write no expert shards);
    * anything else -> the explicit deck list (a comma-joined string is split).
    """
    if expert_decks is None:
        return None
    if isinstance(expert_decks, str):
        expert_decks = [d.strip() for d in expert_decks.split(",") if d.strip()]
    decks = [str(d).strip() for d in expert_decks if str(d).strip()]
    if not decks:
        return None
    lowered = [d.lower() for d in decks]
    if lowered == [EXPERT_DECKS_NONE]:
        return None
    if lowered == [EXPERT_DECKS_ROSTER]:
        return _default_az_league_roster() or None
    return decks


def az_cycle(deck=None, *, games: int = DEFAULT_AZ_GAMES,
             sims: int = DEFAULT_AZ_SIMS, worlds: int = DEFAULT_AZ_WORLDS,
             sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
             sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
             sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
             sb_persist: int = DEFAULT_SB_PERSIST,
             workers: Optional[int] = None, batches: int = DEFAULT_AZ_CYCLE_BATCHES,
             batch_size: int = DEFAULT_AZ_BATCH_SIZE, lr: float = DEFAULT_AZ_LR,
             td_n: int = DEFAULT_AZ_TD_N, q_mix: float = DEFAULT_AZ_Q_MIX,
             window: int = DEFAULT_AZ_WINDOW,
             eval_games: int = DEFAULT_AZ_EVAL_GAMES,
             eval_sims: int = DEFAULT_AZ_EVAL_SIMS,
             eval_worlds: int = DEFAULT_AZ_EVAL_WORLDS,
             promote_threshold: float = DEFAULT_AZ_PROMOTE_THRESHOLD, seed: int = 1,
             use_actor: Optional[bool] = None,
             mirror_frac: float = DEFAULT_MIRROR_FRAC,
             scripted_opponent_frac: float = 0.0,
             gate_floor: float = DEFAULT_GATE_FLOOR,
             expert_decks=EXPERT_DECKS_ROSTER,
             expert_games: int = DEFAULT_AZ_EXPERT_GAMES,
             expert_opponent: Optional[str] = None,
             roster: Optional[list] = None, bo3: bool = True,
             gate: bool = True, exhaustive: bool = False,
             exhaustive_selfplay: bool = False,
             exhaustive_repeats: int = DEFAULT_AZ_EXHAUSTIVE_REPEATS,
             scripted_cells: int = DEFAULT_AZ_SCRIPTED_CELLS,
             cross_world: bool = False,
             slot: int = 0) -> dict:
    """Sequential single-process cycle: cross-deck self-play (mirror + roster,
    ``mirror_frac``) -> train the ONE gen candidate -> gate it against the current
    incumbent over a matchup sample (promote on aggregate WR).

    ``exhaustive`` swaps the random self-play schedule for the exact matchup
    matrix (see :func:`az_selfplay.build_exhaustive_schedule_ex`): one bo3 match
    vs scripted:hard per ORDERED focus x opponent pair plus one pure self-play
    match per UNORDERED pair — 155 matches on the 10-deck roster. ``games``,
    ``mirror_frac`` and ``scripted_opponent_frac`` are ignored; with the actor
    built the WHOLE matrix runs on it (vs-scripted cells via the scripted
    oracle, train/scripted_oracle.py).

    ``exhaustive_selfplay`` (implies ``exhaustive``) drops the vs-scripted family:
    the cycle plays ONLY the pure self-play cells, one bo3 match per unordered
    deck pair (55 on the 10-deck roster), entirely on the C++ actor backend.
    ``exhaustive_repeats=n`` plays every cell of that matrix n times per cycle.

    ``scripted_cells=k`` (self-play matrix only) adds back k vs-scripted matches
    per cycle, rotating through the ordered (focus, opponent) pair list by
    ``slot`` — az-league passes its slot index, a standalone cycle is slot 0 —
    so coverage of every ordered pair accumulates across a run. The cells are
    marked like the full matrix's, so the actor plays them via the scripted
    oracle alongside the self-play cells.

    ``window=0`` sizes the training window automatically: 2x the shards THIS
    cycle's generation just wrote (self-play + expert), so each training pass
    always covers exactly this generation pass plus the previous one — older
    shards age out and anything newer (e.g. expert shards written just before
    the run) still lands inside the window.

    ``gate=False`` skips the eval/gate stage entirely (the returned ``eval`` is
    None) — az-league uses it to gate every K slots (``--gate-every``) instead
    of every slot, letting the candidate accumulate several cycles of training
    (and candidate-generated self-play) between promotions.

    ``expert_decks`` (the doomsday fix) additionally writes
    :func:`az_selfplay.generate_expert` demonstration shards each cycle —
    ``expert_games`` scripted:hard matches per listed deck, pi = one-hot on the
    expert's action — into the same shard pool the trainer reads, so the net
    behavior-clones hand-coded combo lines (and prices mid-combo states by games
    the combo wins) that self-play search cannot discover on its own.
    ``expert_opponent`` (a scripted spec, e.g. ``"scripted:random"``) weakens
    the expert games' opponent seat and records only the focus seat — for a
    combo deck that loses its hard-vs-hard matchups, this is what makes the
    demonstrations come from games the combo actually wins (see
    :func:`az_selfplay.generate_expert`).

    ``deck`` is the FOCUS pool — a str (single focus), a list of stems (a
    deck×opponent matrix), or None (default: the whole decks/league/ roster). Each
    game one focus deck is drawn and plays a mirror (P=``mirror_frac``) or a draw
    from ``roster`` (the opponent pool, default: the whole league roster).

    ``scripted_opponent_frac`` (0..1, default 0) hands that fraction of the
    self-play games' OPPONENT seat to the rule-based scripted:hard agent, with the
    net+MCTS on the focus seat and only the net seat's decisions sampled; 1.0
    trains the cycle entirely against the scripted agent. It forces the Python
    self-play backend (the C++ actor is pure self-play) and does NOT change the
    gate, which stays candidate-vs-incumbent.

    ``use_actor`` chooses the self-play backend (None=AUTO: the C++ actor iff
    built, else Python; see :func:`az_selfplay.generate`)."""
    import az_selfplay

    if roster is None:
        roster = _default_az_league_roster()
    focus = _normalize_focus(deck, _default_az_league_roster())
    label = focus[0] if len(focus) == 1 else f"{len(focus)}-deck matrix"

    exhaustive = bool(exhaustive) or bool(exhaustive_selfplay)
    matrix_txt = ("" if not exhaustive else
                  ", exhaustive matrix"
                  + (" (self-play only)" if exhaustive_selfplay else "")
                  + (f" x{exhaustive_repeats}" if exhaustive_repeats > 1 else ""))
    print(f"=== az cycle: self-play (cross-deck, focus={label}, "
          f"{'bo3' if bo3 else 'bo1'}{matrix_txt}) ===")
    gen = az_selfplay.generate(focus[0], games=games, sims=sims, worlds=worlds,
                               sb_sims=sb_sims, sb_worlds=sb_worlds,
                               sb_max_depth=sb_max_depth,
                               sb_rollout_turns=sb_rollout_turns,
                               sb_persist=bool(sb_persist),
                               workers=workers, seed=seed, use_actor=use_actor,
                               roster=roster, focus_decks=focus,
                               mirror_frac=mirror_frac,
                               scripted_opponent_frac=scripted_opponent_frac,
                               bo3=bo3, exhaustive=exhaustive,
                               exhaustive_selfplay=exhaustive_selfplay,
                               exhaustive_repeats=exhaustive_repeats,
                               scripted_cells=scripted_cells, slot=slot,
                               cross_world=cross_world, td_n=td_n)
    experts = _resolve_expert_decks(expert_decks)
    if experts:
        # Per-listed-deck matches so a multi-deck list doesn't dilute each deck's
        # demonstrations; written AFTER self-play so both land inside the window.
        print("=== az cycle: expert demonstrations (scripted:hard"
              + (f" vs {expert_opponent}" if expert_opponent else "") + ") ===")
        print(f"[az cycle] expert decks: {expert_decks} -> {len(experts)} deck(s) "
              f"[{', '.join(experts)}] x {expert_games} matches each")
        gen["expert"] = az_selfplay.generate_expert(
            experts, games=expert_games * len(experts),
            roster=roster, mirror_frac=mirror_frac, bo3=bo3, seed=seed,
            opponent=expert_opponent)
    else:
        print(f"[az cycle] expert decks: {expert_decks!r} -> none "
              f"(no expert shards this cycle)")
    if window == 0:
        # Auto window: 2x the shards this cycle just wrote, so training always
        # reads exactly this generation pass plus the previous one.
        n_new = len(gen["shards"]) + len((gen.get("expert") or {}).get("shards", []))
        window = max(1, 2 * n_new)
        print(f"[az cycle] auto window: {n_new} new shard(s) this cycle -> "
              f"window={window} (2x, covers this pass + the previous one)")
    print("=== az cycle: train (gen net) ===")
    tr = train_az(label, batches=batches, batch_size=batch_size, lr=lr,
                  q_mix=q_mix, window=window, seed=seed)
    if not gate:
        print("=== az cycle: eval/gate skipped (gated every K slots) ===")
        return {"generate": gen, "train": tr, "eval": None}
    print("=== az cycle: eval/gate (aggregate) ===")
    ev = az_eval(focus, candidate=tr["snapshot"], games=eval_games, sims=eval_sims,
                 worlds=eval_worlds, sb_sims=sb_sims, sb_worlds=sb_worlds,
                 sb_max_depth=sb_max_depth, sb_rollout_turns=sb_rollout_turns,
                 sb_persist=sb_persist,
                 promote_threshold=promote_threshold,
                 promote=True, seed=seed, roster=roster, gate_floor=gate_floor,
                 bo3=bo3, workers=workers)
    return {"generate": gen, "train": tr, "eval": ev}


# ----------------------------------------------------------------------
# AZ league: rotate az cycles over the decks/league/ roster
# ----------------------------------------------------------------------
#
# Mirrors the PPO league driver's rotation + resume ERGONOMICS (a roster taken
# from decks/league/, a JSON progress sidecar rewritten after each unit of work,
# and a `--resume` that restores the full run config and continues from the next
# incomplete slot) WITHOUT reusing its PPO-specific PFSP pool machinery. Each
# slot runs one self-contained `az_cycle` (self-play -> train -> gate) for one
# deck with the actor backend (AUTO), so resume granularity is one deck cycle —
# the az cycle is atomic enough that this needs no finer checkpointing.

AZ_LEAGUE_STATE_VERSION = 1

# Per-slot result records kept in memory and in the sidecar are capped so an
# indefinite (rotations=0) run can't grow either without bound.
_AZ_LEAGUE_MAX_RESULTS = 500


def _az_league_state_path(ckpt_dir: str) -> str:
    return os.path.join(ckpt_dir, "_az_league_progress.json")


def _write_az_league_state(ckpt_dir: str, state: dict) -> None:
    """Atomically persist the az-league driver's progress to its JSON sidecar."""
    write_progress_state(_az_league_state_path(ckpt_dir), state, "az-league")


def _read_az_league_state(ckpt_dir: str) -> Optional[dict]:
    return read_progress_state(_az_league_state_path(ckpt_dir), "az-league")


def _default_az_league_roster() -> list:
    """Every deck in decks/league/, referenced 'league/<stem>' (rotation order)."""
    if not os.path.isdir(_LEAGUE_DECKS_DIR):
        return []
    return sorted("league/" + os.path.splitext(p)[0]
                  for p in os.listdir(_LEAGUE_DECKS_DIR) if p.endswith(".dk"))


def az_league(*, decks=None, rotations: int = 1, cycles_per_deck: int = 1,
              games: int = DEFAULT_AZ_GAMES, sims: int = DEFAULT_AZ_SIMS,
              worlds: int = DEFAULT_AZ_WORLDS,
              sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
              sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
              sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
              sb_persist: int = DEFAULT_SB_PERSIST,
              workers: Optional[int] = None, batches: int = DEFAULT_AZ_CYCLE_BATCHES,
              batch_size: int = DEFAULT_AZ_BATCH_SIZE, lr: float = DEFAULT_AZ_LR,
              td_n: int = DEFAULT_AZ_TD_N, q_mix: float = DEFAULT_AZ_Q_MIX,
              window: int = DEFAULT_AZ_WINDOW,
              eval_games: int = DEFAULT_AZ_EVAL_GAMES,
              eval_sims: int = DEFAULT_AZ_EVAL_SIMS,
              eval_worlds: int = DEFAULT_AZ_EVAL_WORLDS,
              promote_threshold: float = DEFAULT_AZ_PROMOTE_THRESHOLD, seed: int = 1,
              mirror_frac: float = DEFAULT_MIRROR_FRAC,
              scripted_opponent_frac: float = 0.0,
              matrix: bool = False, exhaustive: bool = False,
              exhaustive_selfplay: bool = False,
              exhaustive_repeats: int = DEFAULT_AZ_EXHAUSTIVE_REPEATS,
              scripted_cells: int = DEFAULT_AZ_SCRIPTED_CELLS,
              cross_world: bool = False,
              gate_floor: float = DEFAULT_GATE_FLOOR,
              gate_every: int = DEFAULT_GATE_EVERY,
              expert_decks=EXPERT_DECKS_ROSTER,
              expert_games: int = DEFAULT_AZ_EXPERT_GAMES,
              expert_opponent: Optional[str] = None,
              use_actor: Optional[bool] = None, resume: bool = False,
              bo3: bool = True, ckpt_dir: str = _AZ_CKPT_DIR) -> dict:
    """Rotate ``az_cycle`` over the league roster.

    The unit of work is one deck cycle; slot ``i`` maps to (rotation, deck,
    cycle) by index arithmetic and the sidecar persists the index of the NEXT
    slot to run. ``resume=True`` restores the roster, budgets, and every knob
    from the sidecar and continues from that slot (all other flags are ignored
    on resume, mirroring ``league --resume``).

    ``matrix=True`` replaces the one-deck-per-slot focus rotation with the full
    deck x deck MATRIX every slot: each cycle's self-play draws its focus deck
    uniformly from the whole roster per game (``az_cycle`` with a list focus), so
    the training window stays stationary across slots instead of sweeping one
    deck's distribution at a time — the per-deck rotation is where the
    catastrophic-forgetting pressure came from. A "rotation" then counts
    ``cycles_per_deck`` matrix cycles instead of a roster pass.

    ``exhaustive=True`` goes one step further than ``matrix``: every slot's
    self-play is the EXACT matchup matrix (one vs-scripted match per ordered
    pair + one pure self-play match per unordered pair — 155 bo3 matches on a
    10-deck roster; see :func:`az_selfplay.build_exhaustive_schedule_ex`)
    instead of a random draw. Slot accounting follows the matrix rule (a
    rotation = ``cycles_per_deck`` whole-roster cycles); ``games``,
    ``mirror_frac`` and ``scripted_opponent_frac`` are ignored, and with the
    actor built the whole matrix runs on it — vs-scripted cells via the
    scripted oracle.

    ``exhaustive_selfplay=True`` (implies ``exhaustive``) keeps only the pure
    SELF-PLAY family of that matrix: every slot plays one bo3 match per
    UNORDERED deck pair, mirrors included (55 on the 10-deck roster) and no
    vs-scripted cells, so the whole slot runs on the C++ actor backend.
    ``exhaustive_repeats=n`` plays each cell n times per slot — e.g.
    ``exhaustive_selfplay=True, exhaustive_repeats=2`` is "every self-play
    matchup twice, every slot".

    ``scripted_cells=k`` (with ``exhaustive_selfplay``) additionally plays k
    vs-scripted:hard matches per slot, ROTATING through the full ordered (focus,
    opponent) pair list: slot ``s`` takes the k consecutive pairs starting at
    ``(s * k) % n_ordered``, wrapping — so ceil(n_ordered / k) slots cover every
    ordered pair, and the scripted cells never repeat the same corner of the
    matrix while the rest of the slot stays on the C++ actor.

    ``gate_every=0`` disables the eval/gate entirely: no slot is gated, and at
    run COMPLETION the newest candidate snapshot is promoted to
    ``gen__azfinal.pt`` unconditionally — otherwise ``az:gen`` serving/eval
    specs would keep resolving a stale (or missing) incumbent forever. An
    indefinite run (``rotations=0``) with ``gate_every=0`` never reaches
    completion, so it never promotes until interrupted+finished — prefer a
    finite ``rotations`` with ``gate_every=0``.

    ``rotations=0`` runs INDEFINITELY: slots keep generating until the process
    is interrupted. The sidecar still advances after every completed slot, so an
    interrupted indefinite run resumes with ``--resume`` like a finite one, and
    each training pass prunes shards outside the last two windows (see
    :func:`prune_shards`) so ``az_data/gen`` stays bounded."""
    os.makedirs(ckpt_dir, exist_ok=True)

    slot_index = 0
    results: list = []
    if resume:
        state = _read_az_league_state(ckpt_dir)
        if state is None:
            raise FileNotFoundError(
                f"--resume: no az-league progress file at "
                f"{_az_league_state_path(ckpt_dir)}. Start an az-league run first "
                f"(it writes the file as it trains), then resume it.")
        roster = list(state["roster"])
        rotations = int(state["rotations"])
        cycles_per_deck = int(state["cycles_per_deck"])
        p = state.get("params", {})
        games = int(p.get("games", games))
        sims = int(p.get("sims", sims))
        worlds = int(p.get("worlds", worlds))
        sb_sims = int(p.get("sb_sims", sb_sims))
        sb_worlds = int(p.get("sb_worlds", sb_worlds))
        sb_max_depth = int(p.get("sb_max_depth", sb_max_depth))
        # p.get default keeps pre-rollout sidecars resumable (they carry no
        # sb_rollout_turns key).
        sb_rollout_turns = int(p.get("sb_rollout_turns", sb_rollout_turns))
        sb_persist = int(p.get("sb_persist", sb_persist))
        workers = p.get("workers", workers)
        batches = int(p.get("batches", batches))
        batch_size = int(p.get("batch_size", batch_size))
        lr = float(p.get("lr", lr))
        # .get defaults keep sidecars written before the n-step TD knobs resumable.
        td_n = int(p.get("td_n", td_n))
        q_mix = float(p.get("q_mix", q_mix))
        window = int(p.get("window", window))
        eval_games = int(p.get("eval_games", eval_games))
        eval_sims = int(p.get("eval_sims", eval_sims))
        eval_worlds = int(p.get("eval_worlds", eval_worlds))
        promote_threshold = float(p.get("promote_threshold", promote_threshold))
        seed = int(p.get("seed", seed))
        mirror_frac = float(p.get("mirror_frac", mirror_frac))
        # Older sidecars predate these knobs; .get keeps them resumable.
        scripted_opponent_frac = float(
            p.get("scripted_opponent_frac", scripted_opponent_frac))
        matrix = bool(p.get("matrix", False))
        exhaustive = bool(p.get("exhaustive", False))
        exhaustive_selfplay = bool(p.get("exhaustive_selfplay", False))
        exhaustive_repeats = int(p.get("exhaustive_repeats", 1))
        scripted_cells = int(p.get("scripted_cells", 0))
        cross_world = bool(p.get("cross_world", False))
        gate_floor = float(p.get("gate_floor", gate_floor))
        gate_every = int(p.get("gate_every", gate_every))
        # The RAW user value (sentinel included) is what the sidecar carries;
        # _resolve_expert_decks interprets it at use, so a resumed run re-reads
        # decks/league/ instead of freezing yesterday's roster. Sidecars written
        # before the sentinel existed carry an explicit list or None and resolve
        # to themselves.
        expert_decks = p.get("expert_decks", None)
        expert_games = int(p.get("expert_games", expert_games))
        expert_opponent = p.get("expert_opponent", expert_opponent)
        use_actor = p.get("use_actor", use_actor)
        bo3 = bool(p.get("bo3", bo3))
        slot_index = int(state.get("slot_index", 0))
        results = list(state.get("results", []))
        print(f"[az-league] resuming from {_az_league_state_path(ckpt_dir)}: "
              f"next slot {slot_index}")
    elif decks:
        roster = ([d.strip() for d in decks.split(",") if d.strip()]
                  if isinstance(decks, str) else [str(d).strip() for d in decks if str(d).strip()])
    else:
        roster = _default_az_league_roster()
    if not roster:
        raise ValueError(
            f"No decks found for az-league (looked in {_LEAGUE_DECKS_DIR}). "
            f"Add deck files there, or pass --decks explicitly.")
    # 'gen' is the reserved generalist stem — a roster deck may not collide with it.
    for _d in roster:
        assert_not_reserved_deck(_d)
    # Normalize to a list (or None) but do NOT resolve the roster/none sentinel
    # here: az_cycle resolves it at the moment the shards are generated, and the
    # sidecar below persists this raw value.
    if expert_decks and isinstance(expert_decks, str):
        expert_decks = [d.strip() for d in expert_decks.split(",") if d.strip()]
    expert_decks = list(expert_decks) if expert_decks else None

    # rotations == 0 -> indefinite: no materialized slot list; slot i maps to
    # (rotation, deck, cycle) by index arithmetic and total stays None. In
    # matrix (and exhaustive) mode every slot is a whole-roster cycle, so a
    # rotation is just cycles_per_deck slots.
    # --exhaustive-selfplay is a NARROWING of the exhaustive matrix, so it
    # implies it; normalize before the slot accounting and the sidecar so both
    # (and a later --resume) see the effective mode.
    exhaustive = bool(exhaustive) or bool(exhaustive_selfplay)
    exhaustive_repeats = max(1, int(exhaustive_repeats))
    # The rotating vs-scripted slice only applies to the self-play-only matrix
    # (full --exhaustive already plays every ordered pair).
    if scripted_cells and not exhaustive_selfplay:
        print(f"[az-league] ignoring --scripted-cells {scripted_cells}: it "
              f"applies to --exhaustive-selfplay slots only"
              + (" (full --exhaustive already plays every ordered vs-scripted "
                 "cell)" if exhaustive else ""))
    scripted_cells = max(0, int(scripted_cells)) if exhaustive_selfplay else 0
    whole_roster_slots = matrix or exhaustive
    per_rotation = (cycles_per_deck if whole_roster_slots
                    else len(roster) * cycles_per_deck)
    total = None if rotations == 0 else rotations * per_rotation

    base_state = {
        "version": AZ_LEAGUE_STATE_VERSION,
        "roster": roster,
        "rotations": rotations,
        "cycles_per_deck": cycles_per_deck,
        "params": {
            "games": games, "sims": sims, "worlds": worlds,
            "sb_sims": sb_sims, "sb_worlds": sb_worlds, "sb_max_depth": sb_max_depth,
            "sb_rollout_turns": sb_rollout_turns, "sb_persist": sb_persist,
            "workers": workers,
            "batches": batches, "batch_size": batch_size, "lr": lr,
            "td_n": td_n, "q_mix": q_mix,
            "window": window, "eval_games": eval_games, "eval_sims": eval_sims,
            "eval_worlds": eval_worlds, "promote_threshold": promote_threshold,
            "seed": seed, "mirror_frac": mirror_frac,
            "scripted_opponent_frac": scripted_opponent_frac, "matrix": matrix,
            "exhaustive": exhaustive,
            "exhaustive_selfplay": exhaustive_selfplay,
            "exhaustive_repeats": exhaustive_repeats,
            "scripted_cells": scripted_cells,
            "cross_world": cross_world,
            "gate_floor": gate_floor, "gate_every": gate_every,
            "expert_decks": expert_decks,
            "expert_games": expert_games,
            "expert_opponent": expert_opponent, "use_actor": use_actor,
            "bo3": bo3,
        },
    }

    print(f"AZ league roster: {', '.join(roster)}")
    rotations_txt = "indefinite" if total is None else str(rotations)
    total_txt = "unbounded" if total is None else str(total)
    times_txt = ("once" if exhaustive_repeats == 1
                 else f"{exhaustive_repeats}x")
    scr_cells_txt = (f"; +{scripted_cells} rotating vs-scripted cell(s) per slot"
                     if scripted_cells else "; no vs-scripted cells")
    focus_txt = (f"exhaustive SELF-PLAY matrix (every unordered pair {times_txt}, "
                 f"every slot{scr_cells_txt})" if exhaustive_selfplay
                 else f"exhaustive matrix (every matchup cell {times_txt}, every slot)"
                 if exhaustive
                 else "matrix (whole-roster focus every slot)" if matrix
                 else "per-deck rotation")
    print(f"  rotations={rotations_txt}  cycles_per_deck={cycles_per_deck}  "
          f"slots={total_txt}  focus={focus_txt}  (starting at slot {slot_index})")
    gate_every = max(0, int(gate_every))
    gate_txt = ("OFF (ungated promotion at run completion)" if gate_every == 0
                else str(gate_every))
    window_txt = "auto(2x new shards)" if window == 0 else str(window)
    print(f"  games={games} sims={sims} worlds={worlds} mirror_frac={mirror_frac}  "
          f"sb_sims={sb_sims} sb_worlds={sb_worlds} sb_max_depth={sb_max_depth} "
          f"sb_rollout_turns={sb_rollout_turns} sb_persist={sb_persist}  "
          f"batches={batches} window={window_txt} td_n={td_n} q_mix={q_mix}  "
          f"cross_world={int(cross_world)}  "
          f"eval_games={eval_games} promote>={promote_threshold} "
          f"gate_floor={gate_floor} gate_every={gate_txt}")
    if gate_every == 0 and total is None:
        print("  WARNING: gate_every=0 with rotations=0 (indefinite) never "
              "reaches completion, so gen__azfinal is never refreshed — prefer "
              "a finite --rotations with --gate-every 0.")
    if scripted_opponent_frac:
        print(f"  scripted_opponent_frac={scripted_opponent_frac} "
              f"(scripted:hard on the opponent seat that share of self-play "
              f"games; Python backend)")
    if expert_decks:
        # Raw value (the 'roster' sentinel included) — az_cycle prints what it
        # resolves to each slot.
        print(f"  expert demonstrations: {', '.join(expert_decks)} "
              f"({expert_games} scripted:hard matches per deck per slot)")
    if scripted_cells:
        n_ordered = len(roster) * len(roster)
        print(f"  rotating vs-scripted cells: {scripted_cells} per slot, offset "
              f"(slot * {scripted_cells}) % {n_ordered} over the ordered "
              f"(focus, opponent) pair list "
              f"({-(-n_ordered // scripted_cells)} slots cover every pair)")

    def save_progress(next_slot: int):
        _write_az_league_state(ckpt_dir, {
            **base_state, "slot_index": int(next_slot), "results": results,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S")})

    # Record the starting position so an interruption before the first cycle still
    # leaves a resumable sidecar.
    save_progress(slot_index)
    if total is not None and slot_index >= total:
        print("[az-league] saved progress is already complete — nothing to resume.")
        return {"roster": roster, "slots": total, "results": results}

    si = slot_index
    last_snapshot = None
    while total is None or si < total:
        r, rem = divmod(si, per_rotation)
        if whole_roster_slots:
            c = rem
            focus = list(roster)
            mode_lbl = ("exhaustive-selfplay" if exhaustive_selfplay
                        else "exhaustive" if exhaustive else "matrix")
            deck_label = f"{mode_lbl}[{len(roster)} decks]"
        else:
            di, c = divmod(rem, cycles_per_deck)
            focus = roster[di]
            deck_label = focus
        slot_seed = seed + si
        # Gate on the LAST slot of each gate_every group (slot-index arithmetic,
        # so an interrupted run resumes onto the same cadence). gate_every == 0
        # disables gating entirely (ungated promotion at run completion instead).
        do_gate = gate_every > 0 and ((si + 1) % gate_every == 0)
        slot_txt = f"{si + 1}" if total is None else f"{si + 1}/{total}"
        rot_txt = f"{r + 1}" if total is None else f"{r + 1}/{rotations}"
        gate_note = ("" if do_gate
                     else "  [gating off]" if gate_every == 0
                     else f"  [gate deferred: every {gate_every} slots]")
        print(f"\n{'='*60}")
        print(f"[az-league slot {slot_txt}] rotation {rot_txt}  "
              f"deck={deck_label}  cycle {c + 1}/{cycles_per_deck}  "
              f"(seed={slot_seed}){gate_note}")
        print(f"{'='*60}")
        res = az_cycle(focus, games=games, sims=sims, worlds=worlds,
                       sb_sims=sb_sims, sb_worlds=sb_worlds, sb_max_depth=sb_max_depth,
                       sb_rollout_turns=sb_rollout_turns, sb_persist=sb_persist,
                       workers=workers,
                       batches=batches, batch_size=batch_size, lr=lr,
                       td_n=td_n, q_mix=q_mix, window=window,
                       eval_games=eval_games, eval_sims=eval_sims,
                       eval_worlds=eval_worlds, promote_threshold=promote_threshold,
                       seed=slot_seed, use_actor=use_actor,
                       mirror_frac=mirror_frac,
                       scripted_opponent_frac=scripted_opponent_frac,
                       gate_floor=gate_floor,
                       expert_decks=expert_decks, expert_games=expert_games,
                       expert_opponent=expert_opponent,
                       roster=roster, bo3=bo3, gate=do_gate,
                       exhaustive=exhaustive,
                       exhaustive_selfplay=exhaustive_selfplay,
                       exhaustive_repeats=exhaustive_repeats,
                       scripted_cells=scripted_cells,
                       cross_world=cross_world, slot=si)
        gen, tr, ev = res["generate"], res["train"], res["eval"]
        last_snapshot = tr.get("snapshot")
        if ev is None:
            gate_txt = "gating off" if gate_every == 0 else "gate deferred"
        else:
            veto_txt = (f" (floor-veto: {', '.join(ev['vetoes'])})"
                        if ev.get("vetoes") else "")
            gate_txt = (f"gate {ev['wins']}W-{ev['losses']}L-{ev['draws']}D "
                        f"wr={ev['win_rate']:.3f} "
                        f"{'PROMOTED' if ev['promoted'] else 'kept-incumbent' + veto_txt}")
        print(f"[az-league] slot {slot_txt} deck={deck_label}: "
              f"samples={gen['samples']} shards={len(gen['shards'])}  "
              f"train_loss {tr['first_loss']:.3f}->{tr['last_loss']:.3f}  "
              f"{gate_txt}")
        results.append({"slot": si, "deck": deck_label, "rotation": r, "cycle": c,
                        "samples": gen["samples"], "shards": len(gen["shards"]),
                        "gate_win_rate": ev["win_rate"] if ev else None,
                        "promoted": ev["promoted"] if ev else None,
                        "gate_per_deck": ev.get("per_deck") if ev else None,
                        "gate_vetoes": ev.get("vetoes") if ev else None})
        del results[:-_AZ_LEAGUE_MAX_RESULTS]
        save_progress(si + 1)
        si += 1

    if gate_every == 0:
        # Ungated run: nothing ever refreshed gen__azfinal, so az:gen specs
        # (serving, baseline, the next run's gate opponent) would resolve a
        # stale or missing incumbent. Promote the last trained candidate
        # unconditionally at completion.
        from az_net import resolve_az_checkpoint
        cand = last_snapshot or resolve_az_checkpoint("gen", prefer="snapshot")
        if cand and os.path.exists(cand):
            final = _promote_to_final(cand, ckpt_dir)
            print(f"[az-league] gating off: promoted final candidate {cand} -> "
                  f"{final} (unconditional, run complete)")
        else:
            print("[az-league] gating off: no candidate snapshot found to "
                  "promote — gen__azfinal left untouched")

    print(f"\n[az-league] complete: {total} slots over {rotations} rotations.")
    return {"roster": roster, "slots": total, "results": results}


# ----------------------------------------------------------------------
# train.py dispatch entries
# ----------------------------------------------------------------------

def _resolve_use_actor(args) -> Optional[bool]:
    """--actor -> True, --no-actor -> False, neither -> None (AUTO)."""
    if getattr(args, "actor", False):
        return True
    if getattr(args, "no_actor", False):
        return False
    return None

def run_train(args) -> None:
    train_az(args.deck, batches=args.batches, batch_size=args.batch_size,
             lr=args.lr, c_v=args.c_v,
             q_mix=getattr(args, "q_mix", DEFAULT_AZ_Q_MIX), window=args.window,
             from_ppo=args.from_ppo, fresh=args.fresh,
             snapshot_every=args.snapshot_every,
             seed=args.seed if args.seed is not None else 0)


def run_eval(args) -> None:
    # az-eval defaults to bo3 matches; --bo1 opts back into single games.
    az_eval(args.deck, candidate=args.candidate, incumbent=args.incumbent,
            games=args.games, sims=args.sims, worlds=args.worlds,
            sb_sims=getattr(args, "sb_sims", DEFAULT_SB_SIMS),
            sb_worlds=getattr(args, "sb_worlds", DEFAULT_SB_WORLDS),
            sb_max_depth=getattr(args, "sb_max_depth", DEFAULT_SB_MAX_DEPTH),
            sb_rollout_turns=getattr(args, "sb_rollout_turns",
                                     DEFAULT_SB_ROLLOUT_TURNS),
            sb_persist=getattr(args, "sb_persist", DEFAULT_SB_PERSIST),
            promote_threshold=args.promote_threshold, promote=args.promote,
            gate_floor=getattr(args, "gate_floor", DEFAULT_GATE_FLOOR),
            seed=args.seed if args.seed is not None else 1,
            bo3=not getattr(args, "bo1", False),
            workers=getattr(args, "workers", None))


def _split_decks(val) -> Optional[list]:
    """Parse a comma-joined multipick flag into a deck list (None if empty)."""
    if not val:
        return None
    return [d.strip() for d in val.split(",") if d.strip()] or None


def run_cycle(args) -> None:
    # --deck (comma-joined multipick) is the FOCUS pool and --opponents the
    # opponent pool for this cycle's self-play + gating; either default (None/empty)
    # falls back to the whole decks/league/ roster inside az_cycle. So a bare
    # `train.py az` runs the full league deck×opponent matrix; pass a single --deck
    # to fix one focus.
    focus = _split_decks(getattr(args, "deck", None))
    roster = _split_decks(getattr(args, "opponents", None))
    # az defaults to bo3 matches (per-game value target); --bo1 opts back to bo1.
    az_cycle(focus, games=args.games, sims=args.sims, worlds=args.worlds,
             sb_sims=getattr(args, "sb_sims", DEFAULT_SB_SIMS),
             sb_worlds=getattr(args, "sb_worlds", DEFAULT_SB_WORLDS),
             sb_max_depth=getattr(args, "sb_max_depth", DEFAULT_SB_MAX_DEPTH),
             sb_rollout_turns=getattr(args, "sb_rollout_turns",
                                      DEFAULT_SB_ROLLOUT_TURNS),
             sb_persist=getattr(args, "sb_persist", DEFAULT_SB_PERSIST),
             workers=args.workers, batches=args.batches, batch_size=args.batch_size,
             lr=args.lr, td_n=getattr(args, "td_n", DEFAULT_AZ_TD_N),
             q_mix=getattr(args, "q_mix", DEFAULT_AZ_Q_MIX),
             window=args.window, eval_games=args.eval_games,
             eval_sims=args.eval_sims, eval_worlds=args.eval_worlds,
             promote_threshold=args.promote_threshold,
             seed=args.seed if args.seed is not None else 1,
             mirror_frac=getattr(args, "mirror_frac", DEFAULT_MIRROR_FRAC),
             scripted_opponent_frac=getattr(args, "scripted_opponent_frac", 0.0),
             gate_floor=getattr(args, "gate_floor", DEFAULT_GATE_FLOOR),
             # Raw value: az_cycle resolves the roster/none sentinel.
             expert_decks=_split_decks(getattr(args, "expert_decks",
                                               EXPERT_DECKS_ROSTER)),
             expert_games=getattr(args, "expert_games", DEFAULT_AZ_EXPERT_GAMES),
             expert_opponent=getattr(args, "expert_opponent", None),
             roster=roster, bo3=not getattr(args, "bo1", False),
             use_actor=_resolve_use_actor(args),
             exhaustive=getattr(args, "exhaustive", False),
             exhaustive_selfplay=getattr(args, "exhaustive_selfplay", False),
             exhaustive_repeats=getattr(args, "exhaustive_repeats",
                                        DEFAULT_AZ_EXHAUSTIVE_REPEATS),
             scripted_cells=getattr(args, "scripted_cells",
                                    DEFAULT_AZ_SCRIPTED_CELLS),
             cross_world=bool(getattr(args, "cross_world", False)))


def run_league(args) -> None:
    az_league(decks=args.decks, rotations=args.rotations,
              cycles_per_deck=args.cycles_per_deck,
              games=args.games, sims=args.sims, worlds=args.worlds,
              sb_sims=getattr(args, "sb_sims", DEFAULT_SB_SIMS),
              sb_worlds=getattr(args, "sb_worlds", DEFAULT_SB_WORLDS),
              sb_max_depth=getattr(args, "sb_max_depth", DEFAULT_SB_MAX_DEPTH),
              sb_rollout_turns=getattr(args, "sb_rollout_turns",
                                       DEFAULT_SB_ROLLOUT_TURNS),
              sb_persist=getattr(args, "sb_persist", DEFAULT_SB_PERSIST),
              workers=args.workers, batches=args.batches, batch_size=args.batch_size,
              lr=args.lr, td_n=getattr(args, "td_n", DEFAULT_AZ_TD_N),
              q_mix=getattr(args, "q_mix", DEFAULT_AZ_Q_MIX),
              window=args.window, eval_games=args.eval_games,
              eval_sims=args.eval_sims, eval_worlds=args.eval_worlds,
              promote_threshold=args.promote_threshold,
              seed=args.seed if args.seed is not None else 1,
              mirror_frac=getattr(args, "mirror_frac", DEFAULT_MIRROR_FRAC),
              scripted_opponent_frac=getattr(args, "scripted_opponent_frac", 0.0),
              matrix=getattr(args, "matrix", False),
              exhaustive=getattr(args, "exhaustive", False),
              exhaustive_selfplay=getattr(args, "exhaustive_selfplay", False),
              exhaustive_repeats=getattr(args, "exhaustive_repeats",
                                         DEFAULT_AZ_EXHAUSTIVE_REPEATS),
              scripted_cells=getattr(args, "scripted_cells",
                                     DEFAULT_AZ_SCRIPTED_CELLS),
              cross_world=bool(getattr(args, "cross_world", False)),
              gate_floor=getattr(args, "gate_floor", DEFAULT_GATE_FLOOR),
              gate_every=getattr(args, "gate_every", DEFAULT_GATE_EVERY),
              # Raw value: az_cycle resolves the roster/none sentinel per slot.
              expert_decks=_split_decks(getattr(args, "expert_decks",
                                                EXPERT_DECKS_ROSTER)),
              expert_games=getattr(args, "expert_games", DEFAULT_AZ_EXPERT_GAMES),
              expert_opponent=getattr(args, "expert_opponent", None),
              use_actor=_resolve_use_actor(args), resume=args.resume,
              bo3=not getattr(args, "bo1", False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AlphaZero trainer / gating")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train AZNet on self-play shards")
    t.add_argument("--deck", default="delver")
    t.add_argument("--batches", type=int, default=DEFAULT_AZ_TRAIN_BATCHES)
    t.add_argument("--batch-size", type=int, default=DEFAULT_AZ_BATCH_SIZE)
    t.add_argument("--lr", type=float, default=DEFAULT_AZ_LR)
    t.add_argument("--c-v", type=float, default=DEFAULT_AZ_CV)
    t.add_argument("--q-mix", type=float, default=DEFAULT_AZ_Q_MIX,
                   help="Weight of the shard's n-step TD target in the value "
                        "loss: (1-q_mix)*z + q_mix*td_q (default %s)"
                        % DEFAULT_AZ_Q_MIX)
    t.add_argument("--window", type=int, default=DEFAULT_AZ_WINDOW)
    t.add_argument("--from-ppo", default=None, help="Warm-start from a PPO ckpt")
    t.add_argument("--fresh", action="store_true", help="Start from random init")
    t.add_argument("--snapshot-every", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=run_train)

    e = sub.add_parser("eval", help="Gate candidate vs incumbent")
    e.add_argument("--deck", default="delver")
    e.add_argument("--candidate", required=True)
    e.add_argument("--incumbent", default=None)
    e.add_argument("--games", type=int, default=DEFAULT_AZ_EVAL_GAMES)
    e.add_argument("--sims", type=int, default=DEFAULT_AZ_EVAL_SIMS)
    e.add_argument("--worlds", type=int, default=DEFAULT_AZ_EVAL_WORLDS)
    e.add_argument("--sb-sims", type=int, default=DEFAULT_SB_SIMS,
                   help="PUCT sims at a bo3 sideboard root (bo3 only)")
    e.add_argument("--sb-worlds", type=int, default=DEFAULT_SB_WORLDS,
                   help="Determinized worlds at a bo3 sideboard root (bo3 only)")
    e.add_argument("--sb-max-depth", type=int, default=DEFAULT_SB_MAX_DEPTH,
                   help="Descent depth cap at a bo3 sideboard root (bo3 only)")
    e.add_argument("--sb-rollout-turns", type=int,
                   default=DEFAULT_SB_ROLLOUT_TURNS,
                   help="Leaf-rollout horizon at a bo3 sideboard root, in "
                        "player turns (0 = off)")
    e.add_argument("--promote-threshold", type=float,
                   default=DEFAULT_AZ_PROMOTE_THRESHOLD)
    e.add_argument("--gate-floor", type=float, default=DEFAULT_GATE_FLOOR,
                   help="Per-piloted-deck win-rate floor; a deck below it "
                        "vetoes promotion (0 disables)")
    e.add_argument("--promote", action="store_true")
    e.add_argument("--seed", type=int, default=1)
    e.add_argument("--bo1", action="store_true",
                   help="Single-game gate (default: bo3 match win-rate)")
    e.set_defaults(func=run_eval)

    c = sub.add_parser("cycle", help="One generate->train->eval cycle")
    c.add_argument("--deck", default="delver")
    c.add_argument("--games", type=int, default=DEFAULT_AZ_GAMES)
    c.add_argument("--sims", type=int, default=DEFAULT_AZ_SIMS,
                   help="Self-play PUCT sims, TOTAL across --worlds")
    c.add_argument("--worlds", type=int, default=DEFAULT_AZ_WORLDS)
    c.add_argument("--sb-sims", type=int, default=DEFAULT_SB_SIMS,
                   help="PUCT sims at a bo3 sideboard root (bo3 only)")
    c.add_argument("--sb-worlds", type=int, default=DEFAULT_SB_WORLDS,
                   help="Determinized worlds at a bo3 sideboard root")
    c.add_argument("--sb-max-depth", type=int, default=DEFAULT_SB_MAX_DEPTH,
                   help="Descent depth cap at a bo3 sideboard root")
    c.add_argument("--sb-rollout-turns", type=int,
                   default=DEFAULT_SB_ROLLOUT_TURNS,
                   help="Leaf-rollout horizon at a bo3 sideboard root, in "
                        "player turns (0 = off)")
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--batches", type=int, default=DEFAULT_AZ_CYCLE_BATCHES)
    c.add_argument("--batch-size", type=int, default=DEFAULT_AZ_BATCH_SIZE)
    c.add_argument("--lr", type=float, default=DEFAULT_AZ_LR)
    c.add_argument("--td-n", type=int, default=DEFAULT_AZ_TD_N,
                   help="n-step TD horizon recorded in this cycle's shards "
                        "(default %d)" % DEFAULT_AZ_TD_N)
    c.add_argument("--q-mix", type=float, default=DEFAULT_AZ_Q_MIX,
                   help="Weight of td_q in the value target (default %s)"
                        % DEFAULT_AZ_Q_MIX)
    c.add_argument("--window", type=int, default=DEFAULT_AZ_WINDOW)
    c.add_argument("--eval-games", type=int, default=DEFAULT_AZ_EVAL_GAMES)
    c.add_argument("--eval-sims", type=int, default=DEFAULT_AZ_EVAL_SIMS)
    c.add_argument("--eval-worlds", type=int, default=DEFAULT_AZ_EVAL_WORLDS)
    c.add_argument("--promote-threshold", type=float,
                   default=DEFAULT_AZ_PROMOTE_THRESHOLD)
    c.add_argument("--gate-floor", type=float, default=DEFAULT_GATE_FLOOR,
                   help="Per-piloted-deck gate floor (0 disables the veto)")
    c.add_argument("--exhaustive", action="store_true",
                   help="Exact matchup matrix instead of the random draw")
    c.add_argument("--exhaustive-selfplay", action="store_true",
                   help="Exhaustive matrix, pure SELF-PLAY cells only "
                        "(implies --exhaustive)")
    c.add_argument("--exhaustive-repeats", type=int,
                   default=DEFAULT_AZ_EXHAUSTIVE_REPEATS,
                   help="Play every cell of the matrix N times (default %d)"
                        % DEFAULT_AZ_EXHAUSTIVE_REPEATS)
    c.add_argument("--scripted-cells", type=int,
                   default=DEFAULT_AZ_SCRIPTED_CELLS,
                   help="With --exhaustive-selfplay: also play K vs-scripted "
                        "matches from the rotating ordered-pair slice "
                        "(default %d; 0 disables)" % DEFAULT_AZ_SCRIPTED_CELLS)
    c.add_argument("--expert-decks", default=EXPERT_DECKS_ROSTER,
                   help="Comma-separated decks to also write scripted:hard "
                        "EXPERT demonstration shards for each cycle (BC "
                        "targets for combo lines search can't discover); "
                        "default '%s' = every decks/league/ deck, 'none' to "
                        "disable" % EXPERT_DECKS_ROSTER)
    c.add_argument("--expert-games", type=int, default=DEFAULT_AZ_EXPERT_GAMES,
                   help="Expert matches per expert deck per cycle")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--mirror-frac", type=float, default=DEFAULT_MIRROR_FRAC,
                   help="P(opponent deck == focus deck) per self-play game "
                        "(else uniform league-roster draw)")
    c.add_argument("--scripted-opponent-frac", type=float, default=0.0,
                   help="Fraction of self-play games (0..1) whose opponent seat "
                        "is piloted by scripted:hard (net+MCTS on the focus "
                        "seat, net samples only). Forces the Python backend")
    c.add_argument("--bo1", action="store_true",
                   help="Run bo1 self-play + gate (default: bo3 with per-game value)")
    cg = c.add_mutually_exclusive_group()
    cg.add_argument("--actor", action="store_true",
                    help="Force the C++ az_actor self-play backend")
    cg.add_argument("--no-actor", action="store_true",
                    help="Force the pure-Python self-play backend")
    c.set_defaults(func=run_cycle)

    lg = sub.add_parser("league",
                        help="Rotate az cycles over the decks/league/ roster")
    lg.add_argument("--resume", action="store_true",
                    help="Resume from checkpoints/_az_league_progress.json "
                         "(other flags ignored)")
    lg.add_argument("--decks", default=None,
                    help="Comma-separated roster (default: every decks/league/*.dk)")
    lg.add_argument("--rotations", type=int, default=1,
                    help="Full passes over the roster (0 = run indefinitely "
                         "until interrupted)")
    lg.add_argument("--cycles-per-deck", type=int, default=1)
    lg.add_argument("--games", type=int, default=DEFAULT_AZ_GAMES)
    lg.add_argument("--sims", type=int, default=DEFAULT_AZ_SIMS,
                    help="Self-play PUCT sims, TOTAL across --worlds")
    lg.add_argument("--worlds", type=int, default=DEFAULT_AZ_WORLDS)
    lg.add_argument("--sb-sims", type=int, default=DEFAULT_SB_SIMS,
                    help="PUCT sims at a bo3 sideboard root (bo3 only)")
    lg.add_argument("--sb-worlds", type=int, default=DEFAULT_SB_WORLDS,
                    help="Determinized worlds at a bo3 sideboard root")
    lg.add_argument("--sb-max-depth", type=int, default=DEFAULT_SB_MAX_DEPTH,
                    help="Descent depth cap at a bo3 sideboard root")
    lg.add_argument("--sb-rollout-turns", type=int,
                    default=DEFAULT_SB_ROLLOUT_TURNS,
                    help="Leaf-rollout horizon at a bo3 sideboard root, in "
                         "player turns (0 = off)")
    lg.add_argument("--workers", type=int, default=None)
    lg.add_argument("--batches", type=int, default=DEFAULT_AZ_CYCLE_BATCHES)
    lg.add_argument("--batch-size", type=int, default=DEFAULT_AZ_BATCH_SIZE)
    lg.add_argument("--lr", type=float, default=DEFAULT_AZ_LR)
    lg.add_argument("--td-n", type=int, default=DEFAULT_AZ_TD_N,
                    help="n-step TD horizon recorded in each slot's shards "
                         "(default %d)" % DEFAULT_AZ_TD_N)
    lg.add_argument("--q-mix", type=float, default=DEFAULT_AZ_Q_MIX,
                    help="Weight of td_q in the value target (default %s)"
                         % DEFAULT_AZ_Q_MIX)
    lg.add_argument("--window", type=int, default=DEFAULT_AZ_WINDOW)
    lg.add_argument("--eval-games", type=int, default=DEFAULT_AZ_EVAL_GAMES)
    lg.add_argument("--eval-sims", type=int, default=DEFAULT_AZ_EVAL_SIMS)
    lg.add_argument("--eval-worlds", type=int, default=DEFAULT_AZ_EVAL_WORLDS)
    lg.add_argument("--promote-threshold", type=float,
                    default=DEFAULT_AZ_PROMOTE_THRESHOLD)
    lg.add_argument("--gate-floor", type=float, default=DEFAULT_GATE_FLOOR,
                    help="Per-piloted-deck gate floor (0 disables the veto)")
    lg.add_argument("--matrix", action="store_true",
                    help="Whole-roster focus MATRIX every slot instead of the "
                         "per-deck focus rotation")
    lg.add_argument("--exhaustive", action="store_true",
                    help="Exact matchup matrix every slot")
    lg.add_argument("--exhaustive-selfplay", action="store_true",
                    help="Exhaustive matrix, pure SELF-PLAY cells only "
                         "(implies --exhaustive)")
    lg.add_argument("--exhaustive-repeats", type=int,
                    default=DEFAULT_AZ_EXHAUSTIVE_REPEATS,
                    help="Play every cell of the matrix N times per slot "
                         "(default %d)" % DEFAULT_AZ_EXHAUSTIVE_REPEATS)
    lg.add_argument("--scripted-cells", type=int,
                    default=DEFAULT_AZ_SCRIPTED_CELLS,
                    help="With --exhaustive-selfplay: also play K vs-scripted "
                         "matches per slot from the rotating ordered-pair "
                         "slice (default %d; 0 disables)"
                         % DEFAULT_AZ_SCRIPTED_CELLS)
    lg.add_argument("--expert-decks", default=EXPERT_DECKS_ROSTER,
                    help="Comma-separated decks to also write scripted:hard "
                         "EXPERT demonstration shards for each slot (BC "
                         "targets for combo lines search can't discover); "
                         "default '%s' = every decks/league/ deck, 'none' to "
                         "disable" % EXPERT_DECKS_ROSTER)
    lg.add_argument("--expert-games", type=int, default=DEFAULT_AZ_EXPERT_GAMES,
                    help="Expert matches per expert deck per slot")
    lg.add_argument("--seed", type=int, default=1)
    lg.add_argument("--mirror-frac", type=float, default=DEFAULT_MIRROR_FRAC,
                    help="P(opponent deck == focus deck) per self-play game "
                         "(else uniform league-roster draw)")
    lg.add_argument("--scripted-opponent-frac", type=float, default=0.0,
                    help="Fraction of self-play games (0..1) whose opponent seat "
                         "is piloted by scripted:hard (net+MCTS on the focus "
                         "seat, net samples only). Forces the Python backend")
    lg.add_argument("--bo1", action="store_true",
                    help="Run bo1 self-play + gate (default: bo3 with per-game value)")
    lgg = lg.add_mutually_exclusive_group()
    lgg.add_argument("--actor", action="store_true",
                     help="Force the C++ az_actor self-play backend")
    lgg.add_argument("--no-actor", action="store_true",
                     help="Force the pure-Python self-play backend")
    lg.set_defaults(func=run_league)

    args = ap.parse_args()
    args.func(args)
