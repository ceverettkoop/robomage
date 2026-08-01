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
                      DEFAULT_SB_ROLLOUT_TURNS, DEFAULT_SB_PERSIST)

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
DEFAULT_MIRROR_FRAC = 0.25

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
DEFAULT_GATE_FLOOR = 0.2
DEFAULT_GATE_FLOOR_MIN = 4
DEFAULT_GATE_CROSS_PAIRS = 2

# az-league gates (az_eval + promotion) run every DEFAULT_GATE_EVERY slots. One
# slot of training between gates is a small weight delta against a >=55%
# aggregate bar, and each gate costs real wall-clock (56 searched matches), so
# gating every K>1 slots lets the candidate accumulate K cycles of training —
# and K slots of CANDIDATE-generated self-play (resolve_source prefers the
# snapshot line) — before paying for an eval. Promotion cadence, not training,
# is all that changes: candidate snapshots still save every slot.
DEFAULT_GATE_EVERY = 1


# ----------------------------------------------------------------------
# Shard loading
# ----------------------------------------------------------------------

def load_window(deck: str, window: int, data_dir: Optional[str] = None):
    """Load the last ``window`` shards (by mtime) into flat arrays.

    Shards pool into ``az_data/gen/`` — self-play across every focus deck feeds
    the ONE generalist net — so ``deck`` is used only for the error message; the
    default ``data_dir`` is the shared gen pool. Pass ``data_dir`` to override
    (tests point it at a temp dir)."""
    data_dir = data_dir or os.path.join(_AZ_DATA_DIR, GEN_STEM)
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")),
                    key=os.path.getmtime)
    if not shards:
        raise FileNotFoundError(
            f"no self-play shards in {data_dir} — run az-selfplay first")
    shards = shards[-window:]
    obs, pi, z, mask = [], [], [], []
    for s in shards:
        d = np.load(s)
        obs.append(d["obs"]); pi.append(d["pi"]); z.append(d["z"]); mask.append(d["mask"])
    obs = np.concatenate(obs, axis=0)
    pi = np.concatenate(pi, axis=0)
    z = np.concatenate(z, axis=0)
    mask = np.concatenate(mask, axis=0)
    # Shards are raw observation rows, so an obs-layout change (e.g. a new tail
    # block) makes older shards unusable. Say so instead of letting the net's
    # first slice fail with a bare shape error deep in the forward pass.
    if obs.shape[1] != OBS_SIZE or mask.shape[1] != MAX_ACTIONS:
        raise RuntimeError(
            f"self-play shards in {data_dir} were recorded against a different "
            f"observation layout (obs width {obs.shape[1]}, mask {mask.shape[1]}) "
            f"but this build has OBS_SIZE={OBS_SIZE}, MAX_ACTIONS={MAX_ACTIONS} — "
            "delete/regenerate the shards (re-run az-selfplay) before training")
    return obs, pi, z, mask, len(shards)


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
    a per-deck file."""
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

def train_az(deck: str, *, batches: int = 1000, batch_size: int = 256,
             lr: float = 1e-3, c_v: float = 1.0, window: int = 50,
             weight_decay: float = 1e-4, from_ppo: Optional[str] = None,
             fresh: bool = False, log_every: int = 50,
             snapshot_every: int = 0, data_dir: Optional[str] = None,
             ckpt_dir: str = _AZ_CKPT_DIR, seed: int = 0) -> dict:
    import torch
    import torch.nn.functional as F
    from az_net import az_checkpoint_path, decay_exempt_param_groups

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs, pi, z, mask, n_shards = load_window(deck, window, data_dir)
    n = obs.shape[0]
    print(f"[az-train] gen (focus={deck}): {n} samples from {n_shards} shards; "
          f"batches={batches} batch_size={batch_size} lr={lr} c_v={c_v}")

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
    mask_t = torch.as_tensor(mask)

    log_path = os.path.join(ckpt_dir, f"{GEN_STEM}_az_train.log")
    os.makedirs(ckpt_dir, exist_ok=True)
    first_loss = None
    last_loss = None
    bs = min(batch_size, n)

    with open(log_path, "a") as logf:
        logf.write(f"# az-train {time.strftime('%Y-%m-%d %H:%M:%S')} deck={deck} "
                   f"samples={n} batches={batches} bs={bs} lr={lr}\n")
        for b in range(batches):
            idx = rng.integers(0, n, size=bs)
            bi = torch.as_tensor(idx)
            ob = obs_t[bi]; tp = pi_t[bi]; tz = z_t[bi]; mk = mask_t[bi]

            logits, value = net(ob, mk)
            logp = F.log_softmax(logits, dim=-1)
            logp = torch.where(mk, logp, torch.zeros_like(logp))   # kill -inf*0 nan
            loss_pi = -(tp * logp).sum(dim=1).mean()
            loss_v = F.mse_loss(value, tz)
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
                        f"(pi={loss_pi.item():.4f} v={loss_v.item():.4f})")
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


def az_eval(deck, candidate: str, incumbent: Optional[str] = None, *,
            games: int = 56, sims: int = 32, worlds: int = 2, c_puct: float = 1.5,
            sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
            sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
            sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
            sb_persist: int = DEFAULT_SB_PERSIST,
            promote_threshold: float = 0.55, promote: bool = False,
            roster: Optional[list] = None,
            cross_pairs: int = DEFAULT_GATE_CROSS_PAIRS,
            gate_floor: float = DEFAULT_GATE_FLOOR,
            floor_min_matches: int = DEFAULT_GATE_FLOOR_MIN,
            ckpt_dir: str = _AZ_CKPT_DIR, seed: int = 1, bo3: bool = True) -> dict:
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
    scripted opponent piloting the same deck)."""
    from runner import run_match
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
    unit = "matches" if bo3 else "games"
    print(f"[az-eval] {len(matchups)} matchup(s) x {per} {unit} "
          f"({'bo3' if bo3 else 'bo1'}, seats alternating): candidate={cand_path} vs "
          f"{'incumbent ' + inc_path if have_inc else 'scripted (no incumbent yet)'} "
          f"@ sims={sims} worlds={worlds}")

    w = l = d = 0
    breakdown = []
    per_deck: dict = {}      # piloted deck -> [w, l, d] from the candidate's view
    per_matchup: dict = {}   # (dx, dy) -> [w, l, d], candidate piloting dx
    for mi, (dx, dy) in enumerate(matchups):
        half = per // 2
        mw = ml = md = 0
        mseed = seed + mi * 100003
        if per - half:  # candidate (piloting dx) in seat A
            r = run_match(cand_spec, opp_spec, deck_a=dx, deck_b=dy,
                          games=per - half, bo3=bo3, seed=mseed, transcript="quiet")
            mw += r.wins; ml += r.losses; md += r.draws
        if half:        # candidate (piloting dx) in seat B — flip tally to cand view
            r = run_match(opp_spec, cand_spec, deck_a=dy, deck_b=dx,
                          games=half, bo3=bo3, seed=mseed + per, transcript="quiet")
            mw += r.losses; ml += r.wins; md += r.draws
        w += mw; l += ml; d += md
        t = per_deck.setdefault(dx, [0, 0, 0])
        t[0] += mw; t[1] += ml; t[2] += md
        per_matchup[(dx, dy)] = [mw, ml, md]
        tag = f"{dx}(mirror)" if dx == dy else f"{dx} vs {dy}"
        breakdown.append((tag, mw, ml, md))
        print(f"[az-eval]   {tag}: {mw}W-{ml}L-{md}D")

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
        final = az_checkpoint_path(None, ckpt_dir)
        for src, dst in ((cand_path, final),
                         (_meta_of(cand_path), _meta_of(final))):
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
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


# ----------------------------------------------------------------------
# One full cycle: generate -> train -> eval
# ----------------------------------------------------------------------

def az_cycle(deck=None, *, games: int = 50, sims: int = 256, worlds: int = 4,
             sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
             sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
             sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
             sb_persist: int = DEFAULT_SB_PERSIST,
             workers: Optional[int] = None, batches: int = 500,
             batch_size: int = 256, lr: float = 1e-3, window: int = 50,
             eval_games: int = 56, eval_sims: int = 32, eval_worlds: int = 2,
             promote_threshold: float = 0.55, seed: int = 1,
             use_actor: Optional[bool] = None,
             mirror_frac: float = DEFAULT_MIRROR_FRAC,
             scripted_opponent_frac: float = 0.0,
             gate_floor: float = DEFAULT_GATE_FLOOR,
             expert_decks: Optional[list] = None, expert_games: int = 16,
             roster: Optional[list] = None, bo3: bool = True,
             gate: bool = True) -> dict:
    """Sequential single-process cycle: cross-deck self-play (mirror + roster,
    ``mirror_frac``) -> train the ONE gen candidate -> gate it against the current
    incumbent over a matchup sample (promote on aggregate WR).

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

    print(f"=== az cycle: self-play (cross-deck, focus={label}, "
          f"{'bo3' if bo3 else 'bo1'}) ===")
    gen = az_selfplay.generate(focus[0], games=games, sims=sims, worlds=worlds,
                               sb_sims=sb_sims, sb_worlds=sb_worlds,
                               sb_max_depth=sb_max_depth,
                               sb_rollout_turns=sb_rollout_turns,
                               sb_persist=bool(sb_persist),
                               workers=workers, seed=seed, use_actor=use_actor,
                               roster=roster, focus_decks=focus,
                               mirror_frac=mirror_frac,
                               scripted_opponent_frac=scripted_opponent_frac,
                               bo3=bo3)
    if expert_decks:
        # Per-listed-deck matches so a multi-deck list doesn't dilute each deck's
        # demonstrations; written AFTER self-play so both land inside the window.
        print("=== az cycle: expert demonstrations (scripted:hard) ===")
        gen["expert"] = az_selfplay.generate_expert(
            list(expert_decks), games=expert_games * len(expert_decks),
            roster=roster, mirror_frac=mirror_frac, bo3=bo3, seed=seed)
    print("=== az cycle: train (gen net) ===")
    tr = train_az(label, batches=batches, batch_size=batch_size, lr=lr,
                  window=window, seed=seed)
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
                 bo3=bo3)
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
              games: int = 50, sims: int = 256, worlds: int = 4,
              sb_sims: int = DEFAULT_SB_SIMS, sb_worlds: int = DEFAULT_SB_WORLDS,
              sb_max_depth: int = DEFAULT_SB_MAX_DEPTH,
              sb_rollout_turns: int = DEFAULT_SB_ROLLOUT_TURNS,
              sb_persist: int = DEFAULT_SB_PERSIST,
              workers: Optional[int] = None, batches: int = 500,
              batch_size: int = 256, lr: float = 1e-3, window: int = 50,
              eval_games: int = 56, eval_sims: int = 32, eval_worlds: int = 2,
              promote_threshold: float = 0.55, seed: int = 1,
              mirror_frac: float = DEFAULT_MIRROR_FRAC,
              scripted_opponent_frac: float = 0.0,
              matrix: bool = False, gate_floor: float = DEFAULT_GATE_FLOOR,
              gate_every: int = DEFAULT_GATE_EVERY,
              expert_decks: Optional[list] = None, expert_games: int = 16,
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
        gate_floor = float(p.get("gate_floor", gate_floor))
        gate_every = int(p.get("gate_every", gate_every))
        expert_decks = p.get("expert_decks", None)
        expert_games = int(p.get("expert_games", expert_games))
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
    if expert_decks and isinstance(expert_decks, str):
        expert_decks = [d.strip() for d in expert_decks.split(",") if d.strip()]
    expert_decks = list(expert_decks) if expert_decks else None

    # rotations == 0 -> indefinite: no materialized slot list; slot i maps to
    # (rotation, deck, cycle) by index arithmetic and total stays None. In
    # matrix mode every slot is a whole-roster cycle, so a rotation is just
    # cycles_per_deck slots.
    per_rotation = cycles_per_deck if matrix else len(roster) * cycles_per_deck
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
            "window": window, "eval_games": eval_games, "eval_sims": eval_sims,
            "eval_worlds": eval_worlds, "promote_threshold": promote_threshold,
            "seed": seed, "mirror_frac": mirror_frac,
            "scripted_opponent_frac": scripted_opponent_frac, "matrix": matrix,
            "gate_floor": gate_floor, "gate_every": gate_every,
            "expert_decks": expert_decks,
            "expert_games": expert_games, "use_actor": use_actor,
            "bo3": bo3,
        },
    }

    print(f"AZ league roster: {', '.join(roster)}")
    rotations_txt = "indefinite" if total is None else str(rotations)
    total_txt = "unbounded" if total is None else str(total)
    focus_txt = ("matrix (whole-roster focus every slot)" if matrix
                 else "per-deck rotation")
    print(f"  rotations={rotations_txt}  cycles_per_deck={cycles_per_deck}  "
          f"slots={total_txt}  focus={focus_txt}  (starting at slot {slot_index})")
    gate_every = max(1, int(gate_every))
    print(f"  games={games} sims={sims} worlds={worlds} mirror_frac={mirror_frac}  "
          f"sb_sims={sb_sims} sb_worlds={sb_worlds} sb_max_depth={sb_max_depth} "
          f"sb_rollout_turns={sb_rollout_turns} sb_persist={sb_persist}  "
          f"batches={batches} window={window}  "
          f"eval_games={eval_games} promote>={promote_threshold} "
          f"gate_floor={gate_floor} gate_every={gate_every}")
    if scripted_opponent_frac:
        print(f"  scripted_opponent_frac={scripted_opponent_frac} "
              f"(scripted:hard on the opponent seat that share of self-play "
              f"games; Python backend)")
    if expert_decks:
        print(f"  expert demonstrations: {', '.join(expert_decks)} "
              f"({expert_games} scripted:hard matches per deck per slot)")

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
    while total is None or si < total:
        r, rem = divmod(si, per_rotation)
        if matrix:
            c = rem
            focus = list(roster)
            deck_label = f"matrix[{len(roster)} decks]"
        else:
            di, c = divmod(rem, cycles_per_deck)
            focus = roster[di]
            deck_label = focus
        slot_seed = seed + si
        # Gate on the LAST slot of each gate_every group (slot-index arithmetic,
        # so an interrupted run resumes onto the same cadence).
        do_gate = ((si + 1) % gate_every == 0)
        slot_txt = f"{si + 1}" if total is None else f"{si + 1}/{total}"
        rot_txt = f"{r + 1}" if total is None else f"{r + 1}/{rotations}"
        print(f"\n{'='*60}")
        print(f"[az-league slot {slot_txt}] rotation {rot_txt}  "
              f"deck={deck_label}  cycle {c + 1}/{cycles_per_deck}  (seed={slot_seed})"
              f"{'' if do_gate else f'  [gate deferred: every {gate_every} slots]'}")
        print(f"{'='*60}")
        res = az_cycle(focus, games=games, sims=sims, worlds=worlds,
                       sb_sims=sb_sims, sb_worlds=sb_worlds, sb_max_depth=sb_max_depth,
                       sb_rollout_turns=sb_rollout_turns, sb_persist=sb_persist,
                       workers=workers,
                       batches=batches, batch_size=batch_size, lr=lr, window=window,
                       eval_games=eval_games, eval_sims=eval_sims,
                       eval_worlds=eval_worlds, promote_threshold=promote_threshold,
                       seed=slot_seed, use_actor=use_actor,
                       mirror_frac=mirror_frac,
                       scripted_opponent_frac=scripted_opponent_frac,
                       gate_floor=gate_floor,
                       expert_decks=expert_decks, expert_games=expert_games,
                       roster=roster, bo3=bo3, gate=do_gate)
        gen, tr, ev = res["generate"], res["train"], res["eval"]
        if ev is None:
            gate_txt = "gate deferred"
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
             lr=args.lr, c_v=args.c_v, window=args.window,
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
            bo3=not getattr(args, "bo1", False))


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
             lr=args.lr, window=args.window, eval_games=args.eval_games,
             eval_sims=args.eval_sims, eval_worlds=args.eval_worlds,
             promote_threshold=args.promote_threshold,
             seed=args.seed if args.seed is not None else 1,
             mirror_frac=getattr(args, "mirror_frac", DEFAULT_MIRROR_FRAC),
             scripted_opponent_frac=getattr(args, "scripted_opponent_frac", 0.0),
             gate_floor=getattr(args, "gate_floor", DEFAULT_GATE_FLOOR),
             expert_decks=_split_decks(getattr(args, "expert_decks", None)),
             expert_games=getattr(args, "expert_games", 16),
             roster=roster, bo3=not getattr(args, "bo1", False),
             use_actor=_resolve_use_actor(args))


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
              lr=args.lr, window=args.window, eval_games=args.eval_games,
              eval_sims=args.eval_sims, eval_worlds=args.eval_worlds,
              promote_threshold=args.promote_threshold,
              seed=args.seed if args.seed is not None else 1,
              mirror_frac=getattr(args, "mirror_frac", DEFAULT_MIRROR_FRAC),
              scripted_opponent_frac=getattr(args, "scripted_opponent_frac", 0.0),
              matrix=getattr(args, "matrix", False),
              gate_floor=getattr(args, "gate_floor", DEFAULT_GATE_FLOOR),
              gate_every=getattr(args, "gate_every", DEFAULT_GATE_EVERY),
              expert_decks=_split_decks(getattr(args, "expert_decks", None)),
              expert_games=getattr(args, "expert_games", 16),
              use_actor=_resolve_use_actor(args), resume=args.resume,
              bo3=not getattr(args, "bo1", False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AlphaZero trainer / gating")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train AZNet on self-play shards")
    t.add_argument("--deck", default="delver")
    t.add_argument("--batches", type=int, default=1000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--c-v", type=float, default=1.0)
    t.add_argument("--window", type=int, default=50)
    t.add_argument("--from-ppo", default=None, help="Warm-start from a PPO ckpt")
    t.add_argument("--fresh", action="store_true", help="Start from random init")
    t.add_argument("--snapshot-every", type=int, default=0)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=run_train)

    e = sub.add_parser("eval", help="Gate candidate vs incumbent")
    e.add_argument("--deck", default="delver")
    e.add_argument("--candidate", required=True)
    e.add_argument("--incumbent", default=None)
    e.add_argument("--games", type=int, default=56)
    e.add_argument("--sims", type=int, default=32)
    e.add_argument("--worlds", type=int, default=2)
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
    e.add_argument("--promote-threshold", type=float, default=0.55)
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
    c.add_argument("--games", type=int, default=50)
    c.add_argument("--sims", type=int, default=256,
                   help="Self-play PUCT sims, TOTAL across --worlds")
    c.add_argument("--worlds", type=int, default=4)
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
    c.add_argument("--batches", type=int, default=500)
    c.add_argument("--batch-size", type=int, default=256)
    c.add_argument("--lr", type=float, default=1e-3)
    c.add_argument("--window", type=int, default=50)
    c.add_argument("--eval-games", type=int, default=56)
    c.add_argument("--eval-sims", type=int, default=32)
    c.add_argument("--eval-worlds", type=int, default=2)
    c.add_argument("--promote-threshold", type=float, default=0.55)
    c.add_argument("--gate-floor", type=float, default=DEFAULT_GATE_FLOOR,
                   help="Per-piloted-deck gate floor (0 disables the veto)")
    c.add_argument("--expert-decks", default=None,
                   help="Comma-separated decks to also write scripted:hard "
                        "EXPERT demonstration shards for each cycle (BC "
                        "targets for combo lines search can't discover)")
    c.add_argument("--expert-games", type=int, default=16,
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
    lg.add_argument("--games", type=int, default=50)
    lg.add_argument("--sims", type=int, default=256,
                    help="Self-play PUCT sims, TOTAL across --worlds")
    lg.add_argument("--worlds", type=int, default=4)
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
    lg.add_argument("--batches", type=int, default=500)
    lg.add_argument("--batch-size", type=int, default=256)
    lg.add_argument("--lr", type=float, default=1e-3)
    lg.add_argument("--window", type=int, default=50)
    lg.add_argument("--eval-games", type=int, default=56)
    lg.add_argument("--eval-sims", type=int, default=32)
    lg.add_argument("--eval-worlds", type=int, default=2)
    lg.add_argument("--promote-threshold", type=float, default=0.55)
    lg.add_argument("--gate-floor", type=float, default=DEFAULT_GATE_FLOOR,
                    help="Per-piloted-deck gate floor (0 disables the veto)")
    lg.add_argument("--matrix", action="store_true",
                    help="Whole-roster focus MATRIX every slot instead of the "
                         "per-deck focus rotation")
    lg.add_argument("--expert-decks", default=None,
                    help="Comma-separated decks to also write scripted:hard "
                         "EXPERT demonstration shards for each slot (BC "
                         "targets for combo lines search can't discover)")
    lg.add_argument("--expert-games", type=int, default=16,
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
