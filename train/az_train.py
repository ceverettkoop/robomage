"""AlphaZero trainer + gating for RoboMage (Phase C).

- ``train_az`` : load the last M self-play shards for a deck, optimize AZNet with
  Adam (constant lr, weight_decay 1e-4) on the AZ loss
  ``CE(pi, masked_log_softmax(logits)) + c_v * MSE(value, z)``, snapshot the
  CANDIDATE to ``{deck}__azv{steps}.pt``. ``{deck}__azfinal.pt`` (the incumbent)
  is written ONLY by the ``az_eval`` promotion gate — training never touches it.
- ``az_eval`` : candidate-vs-incumbent gating via ``run_match`` with ``az``
  controllers at low sims, seats alternating (half the games each way); promote
  the candidate to ``{deck}__azfinal.pt`` at a win-rate threshold.
- ``az_cycle`` : one sequential generate -> train -> eval iteration.

Exposed to ``train.py`` as the ``az-train`` / ``az-eval`` / ``az`` subcommands.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import time
from typing import Optional

import numpy as np

try:
    from env import OBS_SIZE, MAX_ACTIONS
except ImportError:  # pragma: no cover
    from train.env import OBS_SIZE, MAX_ACTIONS

_AZ_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints", "az")
_AZ_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "az_data")


# ----------------------------------------------------------------------
# Shard loading
# ----------------------------------------------------------------------

def load_window(deck: str, window: int, data_dir: Optional[str] = None):
    """Load the last ``window`` shards (by mtime) for ``deck`` into flat arrays."""
    data_dir = data_dir or os.path.join(_AZ_DATA_DIR, deck)
    shards = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")),
                    key=os.path.getmtime)
    if not shards:
        raise FileNotFoundError(
            f"no self-play shards in {data_dir} — run az-selfplay for '{deck}' first")
    shards = shards[-window:]
    obs, pi, z, mask = [], [], [], []
    for s in shards:
        d = np.load(s)
        obs.append(d["obs"]); pi.append(d["pi"]); z.append(d["z"]); mask.append(d["mask"])
    obs = np.concatenate(obs, axis=0)
    pi = np.concatenate(pi, axis=0)
    z = np.concatenate(z, axis=0)
    mask = np.concatenate(mask, axis=0)
    return obs, pi, z, mask, len(shards)


# ----------------------------------------------------------------------
# Net init / resume
# ----------------------------------------------------------------------

def _init_net(deck: str, from_ppo: Optional[str], fresh: bool):
    """Return (net, prior_steps, provenance-string)."""
    from az_net import (AZNet, load_az, from_ppo as warm_from_ppo,
                        resolve_az_checkpoint, az_checkpoint_path)
    from opponents import resolve_checkpoint

    if not fresh:
        # Continue the CANDIDATE line: newest __azv snapshot first, __azfinal
        # (the gate-promoted incumbent) only as a fallback. This keeps training
        # cumulative across cycles even while the gate rejects candidates.
        az = resolve_az_checkpoint(deck, prefer="snapshot")
        if az:
            net = load_az(az)
            steps = _read_steps(az)
            return net, steps, f"resumed AZ checkpoint {az} (steps={steps})"
    if from_ppo:
        path = resolve_checkpoint(from_ppo)
        return warm_from_ppo(path), 0, f"warm-started from PPO {path}"
    if not fresh:
        # No AZ checkpoint yet: default to warm-starting the deck's PPO generalist.
        ppo = resolve_checkpoint(deck)
        if ppo and os.path.exists(ppo):
            return warm_from_ppo(ppo), 0, f"warm-started from deck PPO {ppo}"
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
    from az_net import az_checkpoint_path

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    obs, pi, z, mask, n_shards = load_window(deck, window, data_dir)
    n = obs.shape[0]
    print(f"[az-train] deck={deck}: {n} samples from {n_shards} shards; "
          f"batches={batches} batch_size={batch_size} lr={lr} c_v={c_v}")

    net, prior_steps, prov = _init_net(deck, from_ppo, fresh)
    print(f"[az-train] net: {prov}")
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    obs_t = torch.as_tensor(obs)
    pi_t = torch.as_tensor(pi)
    z_t = torch.as_tensor(z)
    mask_t = torch.as_tensor(mask)

    log_path = os.path.join(ckpt_dir, f"{os.path.basename(deck)}_az_train.log")
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
                net.save(az_checkpoint_path(deck, steps, ckpt_dir), steps)

    steps = prior_steps + batches * bs
    net.eval()
    # Candidate snapshot only — __azfinal (the incumbent) advances exclusively
    # through the az_eval promotion gate.
    snap = net.save(az_checkpoint_path(deck, steps, ckpt_dir), steps)
    print(f"[az-train] saved candidate snapshot {snap}")
    print(f"[az-train] loss {first_loss:.4f} -> {last_loss:.4f} over {batches} batches")
    return {"samples": n, "first_loss": first_loss, "last_loss": last_loss,
            "snapshot": snap, "steps": steps}


# ----------------------------------------------------------------------
# Gating (candidate vs incumbent)
# ----------------------------------------------------------------------

def az_eval(deck: str, candidate: str, incumbent: Optional[str] = None, *,
            games: int = 20, sims: int = 32, worlds: int = 2, c_puct: float = 1.5,
            promote_threshold: float = 0.55, promote: bool = False,
            ckpt_dir: str = _AZ_CKPT_DIR, seed: int = 1) -> dict:
    """Play ``candidate`` vs ``incumbent`` (mirror deck) with MCTS+AZ controllers
    at low sims, seats alternating (half the games each way, since ``run_match``
    itself never swaps seats and seat A is on the play in bo1). Returns the
    candidate-perspective tally; promotes candidate -> ``{deck}__azfinal.pt``
    when ``promote`` and win-rate >= threshold."""
    from runner import run_match
    from az_net import az_checkpoint_path, resolve_az_checkpoint

    cand_path = resolve_az_checkpoint(candidate, prefer="snapshot") or candidate
    if incumbent is None:
        incumbent = az_checkpoint_path(deck, None, ckpt_dir)
    inc_path = incumbent if os.path.exists(incumbent) else \
        (resolve_az_checkpoint(incumbent) or incumbent)

    knobs = f"?sims={sims}&worlds={worlds}&c={c_puct}"
    cand_spec = f"az:{cand_path}{knobs}"
    have_inc = os.path.exists(inc_path)
    opp_spec = f"az:{inc_path}{knobs}" if have_inc else "scripted"
    print(f"[az-eval] {games} games (bo1, seats alternating): candidate={cand_path} vs "
          f"{'incumbent ' + inc_path if have_inc else 'scripted (no incumbent yet)'} "
          f"@ sims={sims} worlds={worlds}")

    half = games // 2
    w = l = d = 0
    if games - half:  # candidate in seat A
        r = run_match(cand_spec, opp_spec, deck_a=deck, deck_b=deck,
                      games=games - half, bo3=False, seed=seed, transcript="quiet")
        w += r.wins; l += r.losses; d += r.draws
    if half:          # candidate in seat B (flip the tally back to candidate view)
        r = run_match(opp_spec, cand_spec, deck_a=deck, deck_b=deck,
                      games=half, bo3=False, seed=seed + games, transcript="quiet")
        w += r.losses; l += r.wins; d += r.draws
    wr = w / max(1, w + l + d)
    print(f"[az-eval] candidate {w}W-{l}L-{d}D (win_rate={wr:.3f})")

    promoted = False
    if promote and wr >= promote_threshold:
        final = az_checkpoint_path(deck, None, ckpt_dir)
        for src, dst in ((cand_path, final),
                         (_meta_of(cand_path), _meta_of(final))):
            if os.path.exists(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copyfile(src, dst)
        promoted = True
        print(f"[az-eval] PROMOTED candidate -> {final} (>= {promote_threshold:.2f})")
    elif promote:
        print(f"[az-eval] not promoted (win_rate {wr:.3f} < {promote_threshold:.2f})")
    return {"wins": w, "losses": l, "draws": d,
            "win_rate": wr, "promoted": promoted}


def _meta_of(path: str) -> str:
    base, _ = os.path.splitext(path)
    return base + ".meta.json"


# ----------------------------------------------------------------------
# One full cycle: generate -> train -> eval
# ----------------------------------------------------------------------

def az_cycle(deck: str, *, games: int = 50, sims: int = 64, worlds: int = 4,
             workers: Optional[int] = None, batches: int = 500,
             batch_size: int = 256, lr: float = 1e-3, window: int = 50,
             eval_games: int = 20, eval_sims: int = 32, eval_worlds: int = 2,
             promote_threshold: float = 0.55, seed: int = 1) -> dict:
    """Sequential single-process cycle: self-play -> train a candidate -> gate it
    against the current incumbent (promote on success)."""
    import az_selfplay
    from az_net import az_checkpoint_path

    print("=== az cycle: self-play ===")
    gen = az_selfplay.generate(deck, games=games, sims=sims, worlds=worlds,
                               workers=workers, seed=seed)
    print("=== az cycle: train ===")
    tr = train_az(deck, batches=batches, batch_size=batch_size, lr=lr,
                  window=window, seed=seed)
    print("=== az cycle: eval/gate ===")
    ev = az_eval(deck, candidate=tr["snapshot"], games=eval_games, sims=eval_sims,
                 worlds=eval_worlds, promote_threshold=promote_threshold,
                 promote=True, seed=seed)
    return {"generate": gen, "train": tr, "eval": ev}


# ----------------------------------------------------------------------
# train.py dispatch entries
# ----------------------------------------------------------------------

def run_train(args) -> None:
    train_az(args.deck, batches=args.batches, batch_size=args.batch_size,
             lr=args.lr, c_v=args.c_v, window=args.window,
             from_ppo=args.from_ppo, fresh=args.fresh,
             snapshot_every=args.snapshot_every,
             seed=args.seed if args.seed is not None else 0)


def run_eval(args) -> None:
    az_eval(args.deck, candidate=args.candidate, incumbent=args.incumbent,
            games=args.games, sims=args.sims, worlds=args.worlds,
            promote_threshold=args.promote_threshold, promote=args.promote,
            seed=args.seed if args.seed is not None else 1)


def run_cycle(args) -> None:
    az_cycle(args.deck, games=args.games, sims=args.sims, worlds=args.worlds,
             workers=args.workers, batches=args.batches, batch_size=args.batch_size,
             lr=args.lr, window=args.window, eval_games=args.eval_games,
             eval_sims=args.eval_sims, eval_worlds=args.eval_worlds,
             promote_threshold=args.promote_threshold,
             seed=args.seed if args.seed is not None else 1)


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
    e.add_argument("--games", type=int, default=20)
    e.add_argument("--sims", type=int, default=32)
    e.add_argument("--worlds", type=int, default=2)
    e.add_argument("--promote-threshold", type=float, default=0.55)
    e.add_argument("--promote", action="store_true")
    e.add_argument("--seed", type=int, default=1)
    e.set_defaults(func=run_eval)

    c = sub.add_parser("cycle", help="One generate->train->eval cycle")
    c.add_argument("--deck", default="delver")
    c.add_argument("--games", type=int, default=50)
    c.add_argument("--sims", type=int, default=64)
    c.add_argument("--worlds", type=int, default=4)
    c.add_argument("--workers", type=int, default=None)
    c.add_argument("--batches", type=int, default=500)
    c.add_argument("--batch-size", type=int, default=256)
    c.add_argument("--lr", type=float, default=1e-3)
    c.add_argument("--window", type=int, default=50)
    c.add_argument("--eval-games", type=int, default=20)
    c.add_argument("--eval-sims", type=int, default=32)
    c.add_argument("--eval-worlds", type=int, default=2)
    c.add_argument("--promote-threshold", type=float, default=0.55)
    c.add_argument("--seed", type=int, default=1)
    c.set_defaults(func=run_cycle)

    args = ap.parse_args()
    args.func(args)
