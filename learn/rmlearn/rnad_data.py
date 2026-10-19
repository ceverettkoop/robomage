"""R-NaD's input: trajectory episodes (docs/trajectory_format.md) as the learner's [T, G] batches
(docs/rnad_training_plan.md §4 L1, L4).

    ep = read_episode("runs/r1/episodes/a3-m17.g2.rmtraj")   # numpy, every step expanded
    ep = from_rmtraj(rmtraj.read(path))                      # the same from the stdlib reader
    kept, dropped = filter_episodes(eps, missing_mu="error")
    batch = RNaDBatch(kept, chunk_len=32, policies=(1, 2), omniscient=False)
    x, flat = batch.chunk_inputs(c)                           # chunk c's valid steps, padded

**The batch** (``RNaDBatch``; the learner reads only these names, so any object with them, such as
L1's replay batch, feeds it): G full games time-major, T = the longest game rounded up to
``chunk_len``, A = the largest menu rounded up to a multiple of 8.

| name | shape, dtype | meaning |
|---|---|---|
| ``valid`` | [T, G] bool | a real step (a game's steps first, then padding) |
| ``seat`` | [T, G] i64 | the deciding seat, the reference's player id (0 = A, 1 = B; 0 at padding) |
| ``legal`` | [T, G, A] bool | the step's menu rows |
| ``mu`` | [T, G, A] f32 | the behaviour distribution over the menu rows (0 without μ / padding) |
| ``has_mu`` | [T, G] bool | the step carries μ |
| ``chosen`` | [T, G] i64 | the menu row played (−1 at padding) |
| ``reward`` | [T, G, 2] f32 | each seat's result (+1 / −1 / 0) at its game's last step, 0 elsewhere |
| ``sideboarding`` | [T, G] bool | a sideboarding decision (its own regularisation ε, plan B3) |
| ``policy_version`` | [T, G] i64 | the version of the policy μ came from (−1 none) |
| ``evals[p]`` | dict | per served policy id p in ``policies``: ``present`` [T, G] bool, ``version`` [T, G] i64 (−1 absent), ``log_pi`` [T, G, A] f32 (0 absent / pad), as the actor recorded them |
| ``G``, ``T``, ``A``, ``chunk_len``, ``chunks`` | int | |
| ``chunk_inputs(c, viewer)`` | | (the ten contract inputs of chunk c's valid steps, padded as ``batch.pad`` with A = the batch's ``A``, as numpy {name: array}; their flat indices t·G + g, time-major), (None, empty) for a chunk without a valid step; ``viewer`` "seat" or "omni" |

**Episodes** (``Episode``): the file's steps with their tensors expanded once (``batch.Decisions``,
unpadded and concatenated, the seat's and, in an omniscient episode, the omniscient
featurization's), the per-step fields, μ and the recorded evaluations. ``filter_episodes`` drops
what R-NaD cannot train on, with counts: an ``end_reason`` in ``drop_end_reasons`` (default
``cap`` and ``quit``: a truncated game's 0 / 0 is not its value), a step without μ (or raises,
``missing_mu="error"``, the default: self-play steps all carry one), a concession sentinel as the
reply (no menu row was played), a spec other than ``spec_hash``.
"""

import os
import sys

import numpy as np

from . import rmtb
from .batch import Decisions, pad, padded

KIND = "episode"
VERSION = 1
TENSOR_NAMES = ("tok_type", "tok_num", "tok_cat", "tok_card", "tok_ability", "tok_mask", "rel", "act_tok", "act_mask")


def _starts(sizes):
    s = np.zeros(len(sizes) + 1, np.int64)
    np.cumsum(sizes, out=s[1:])
    return s


def _need(arrays, name):
    a = arrays.get(name)
    if a is None:
        raise ValueError("episode: no array %s" % name)
    return a


def _expand(arrays, prefix, steps, widths, dims, menu_type):
    """One featurization's compact arrays → Decisions (docs/trajectory_format.md §3.3, vectorised)."""
    F, C, K, Q = dims
    g = lambda n: _need(arrays, prefix + n)  # noqa: E731
    tok_type = g("tok_type").astype(np.int64)
    tok_off = g("tok_offsets").astype(np.int64)
    rel_off = g("rel_offsets").astype(np.int64)
    num_off = g("tok_num_offsets").astype(np.int64)
    if len(tok_off) != steps + 1 or tok_off[-1] != len(tok_type):
        raise ValueError("episode: %stok_offsets does not tile its tokens" % prefix)
    n = np.diff(tok_off)
    total = int(tok_off[-1])
    # Numeric values: each at row tok_off[step] + index // F.
    idx = g("tok_num_index").astype(np.int64)
    val = g("tok_num_value").astype(np.float32)
    step_of = np.repeat(np.arange(steps), np.diff(num_off))
    if (idx >= n[step_of] * F).any():
        raise ValueError("episode: a numeric position past its decision's tokens")
    tok_num = np.zeros((total, F), np.float32)
    tok_num[tok_off[step_of] + idx // F, idx % F] = val
    w = widths[tok_type]
    out = {"tok_type": tok_type, "tok_num": tok_num}
    for col, name, width, fill in ((1, "tok_cat", C, 0), (2, "tok_card", K, -1), (3, "tok_ability", Q, -1)):
        ww = w[:, col].astype(np.int64)
        values = g(name).astype(np.int64)
        if int(ww.sum()) != len(values):
            raise ValueError("episode: %s%s holds %d values, its tokens' widths %d" % (prefix, name, len(values),
                                                                                       int(ww.sum())))
        a = np.full((total, width), fill, np.int64)
        a[np.repeat(np.arange(total), ww), np.arange(len(values)) - np.repeat(_starts(ww)[:-1], ww)] = values
        out[name] = a
    rel = g("rel").astype(np.int64).reshape(-1, 3)
    r = np.diff(rel_off)
    menu_pos = np.nonzero(tok_type == menu_type)[0]
    menu_step = np.searchsorted(tok_off, menu_pos, side="right") - 1
    act_tok = menu_pos - tok_off[menu_step]
    a_sizes = np.bincount(menu_step, minlength=steps).astype(np.int64)
    out.update({"tok_mask": np.ones(total, np.uint8), "rel": rel, "act_tok": act_tok,
                "act_mask": np.ones(len(act_tok), np.uint8)})
    return Decisions(out, {"n": n, "r": r, "a": a_sizes})


class Episode:
    """One episode file with its steps expanded: ``attrs``, ``spec_hash``, ``steps`` (D); per step
    ``seat``, ``game``, ``decision``, ``sideboarding``, ``chosen``, ``has_mu``, ``policy_version``,
    ``teacher`` ([D] arrays); ``menu`` [D] (menu sizes), ``menu_off`` [D + 1], ``mu`` [ΣA] f32;
    ``eval_policy``, ``eval_version`` [D, P] and ``eval_log_pi`` [ΣA, P] (P = 0 without evaluations);
    ``decs`` (the seat's tensors, batch.Decisions), ``omni`` (the omniscient one's, or None);
    ``result`` [2] f32 (A, B); ``end_reason``."""

    def __init__(self, meta, fields, menu_off, mu, evals, decs, omni):
        self.meta = meta
        self.attrs = meta.get("attrs", {})
        self.spec_hash = meta.get("spec_hash")
        for k, v in fields.items():
            setattr(self, k, v)
        self.steps = len(self.seat)
        self.menu_off = menu_off
        self.menu = np.diff(menu_off)
        self.mu = mu
        self.eval_policy, self.eval_version, self.eval_log_pi = evals
        self.decs, self.omni = decs, omni
        self.result = np.array([self.attrs.get("result_a", 0), self.attrs.get("result_b", 0)], np.float32)
        self.end_reason = self.attrs.get("end_reason", "")

    def evaluation(self, d, policy_id):
        """(version, log π over step d's menu rows) of policy `policy_id`, or None."""
        if self.eval_policy.shape[1] == 0:
            return None
        cols = np.nonzero(self.eval_policy[d] == policy_id)[0]
        if not len(cols):
            return None
        c = int(cols[0])
        return int(self.eval_version[d, c]), self.eval_log_pi[self.menu_off[d]:self.menu_off[d + 1], c]


STEP_FIELDS = ("seat", "game", "decision", "sideboarding", "chosen", "has_mu", "policy_version", "teacher")


def read_episode_bytes(data):
    meta, arrays = rmtb.read_bytes(data)
    attrs = meta.get("attrs", {})
    if meta.get("kind") != KIND:
        raise ValueError("not an episode file (kind %r)" % meta.get("kind"))
    if attrs.get("episode_version") != VERSION:
        raise ValueError("episode version %r, expected %d" % (attrs.get("episode_version"), VERSION))
    dims = tuple(int(attrs[k]) for k in ("F", "C", "K", "Q"))
    menu_type = int(attrs["menu_type"])
    widths = _need(arrays, "type_widths").astype(np.int64)
    steps = len(_need(arrays, "seat"))
    fields = {}
    for k in STEP_FIELDS:
        a = _need(arrays, k)
        if len(a) != steps:
            raise ValueError("episode: the per-decision arrays differ in length")
        fields[k] = a.astype(np.bool_ if k in ("sideboarding", "has_mu") else np.int64)
    mu = _need(arrays, "mu").astype(np.float32)
    menu_off = _need(arrays, "menu_offsets").astype(np.int64)
    if len(menu_off) != steps + 1 or menu_off[-1] != len(mu):
        raise ValueError("episode: menu_offsets does not tile mu")
    if "eval_policy" in arrays:
        evals = (arrays["eval_policy"].astype(np.int64), arrays["eval_version"].astype(np.int64),
                 arrays["eval_log_pi"].astype(np.float32).reshape(len(mu), -1))
    else:
        evals = (np.zeros((steps, 0), np.int64), np.zeros((steps, 0), np.int64), np.zeros((len(mu), 0), np.float32))
    decs = _expand(arrays, "", steps, widths, dims, menu_type)
    if not np.array_equal(decs.sizes["a"], np.diff(menu_off)):
        raise ValueError("episode: the menu tokens are not the menu_offsets rows")
    omni = _expand(arrays, "omni_", steps, widths, dims, menu_type) if attrs.get("omniscient") else None
    return Episode(meta, fields, menu_off, mu, evals, decs, omni)


def read_episode(path):
    """The ``Episode`` of the episode file at `path` (numpy)."""
    with open(path, "rb") as fp:
        ep = read_episode_bytes(fp.read())
    ep.path = path
    return ep


def _decisions_from_steps(tensor_dicts, F, C, K, Q):
    """Decisions from per-step tensor dicts of flat lists (python/rmtraj.py's Step.tensors)."""
    arrays = {}
    for name in TENSOR_NAMES:
        parts = [np.asarray(t[name]) for t in tensor_dicts]
        a = np.concatenate(parts) if parts else np.zeros(0)
        if name == "tok_num":
            a = a.astype(np.float32).reshape(-1, F)
        elif name in ("tok_cat", "tok_card", "tok_ability"):
            a = a.astype(np.int64).reshape(-1, {"tok_cat": C, "tok_card": K, "tok_ability": Q}[name])
        elif name == "rel":
            a = a.astype(np.int64).reshape(-1, 3)
        elif name in ("tok_mask", "act_mask"):
            a = a.astype(np.uint8)
        else:
            a = a.astype(np.int64)
        arrays[name] = a
    sizes = {"n": np.array([len(t["tok_type"]) for t in tensor_dicts], np.int64),
             "r": np.array([len(t["rel"]) // 3 for t in tensor_dicts], np.int64),
             "a": np.array([len(t["act_tok"]) for t in tensor_dicts], np.int64)}
    return Decisions(arrays, sizes)


def from_rmtraj(ep):
    """An ``Episode`` from python/rmtraj.py's ``Episode`` (the stdlib reader): the per-episode dicts
    of steps (seat, chosen, mu or None, policies, sideboarding, tensors, omniscient) → the same
    object ``read_episode`` gives."""
    a = ep.attrs
    F, C, K, Q = (int(a[k]) for k in ("F", "C", "K", "Q"))
    steps = ep.steps
    fields = {"seat": np.array([s.seat for s in steps], np.int64),
              "game": np.array([s.game for s in steps], np.int64),
              "decision": np.array([s.decision for s in steps], np.int64),
              "sideboarding": np.array([s.sideboarding for s in steps], np.bool_),
              "chosen": np.array([s.chosen for s in steps], np.int64),
              "has_mu": np.array([s.mu is not None for s in steps], np.bool_),
              "policy_version": np.array([s.policy_version for s in steps], np.int64),
              "teacher": np.array([s.teacher for s in steps], np.int64)}
    menus = [len(s.tensors["act_tok"]) for s in steps]
    menu_off = _starts(np.array(menus, np.int64))
    mu = np.concatenate([np.asarray(s.mu, np.float32) if s.mu is not None else np.zeros(m, np.float32)
                         for s, m in zip(steps, menus)]) if steps else np.zeros(0, np.float32)
    p = max((len(s.policies) for s in steps), default=0)
    ev_id = np.full((len(steps), p), -1, np.int64)
    ev_ver = np.full((len(steps), p), -1, np.int64)
    ev_lp = np.zeros((int(menu_off[-1]), p), np.float32)
    for d, s in enumerate(steps):
        for k, (pid, ver, _value, lp) in enumerate(s.policies):
            ev_id[d, k], ev_ver[d, k] = pid, ver
            ev_lp[menu_off[d]:menu_off[d + 1], k] = lp
    decs = _decisions_from_steps([s.tensors for s in steps], F, C, K, Q)
    omni = _decisions_from_steps([s.omniscient for s in steps], F, C, K, Q) if a.get("omniscient") else None
    return Episode(ep.meta, fields, menu_off, mu, (ev_id, ev_ver, ev_lp), decs, omni)


def filter_episodes(eps, missing_mu="error", drop_end_reasons=("cap", "quit"), spec_hash=None):
    """(the episodes R-NaD trains on, {reason: count dropped}); see the module docstring.
    ``missing_mu``: "error" raises ValueError at an episode with a step without μ, "drop" drops it."""
    if missing_mu not in ("error", "drop"):
        raise ValueError("missing_mu is 'error' or 'drop'")
    kept, dropped = [], {}

    def drop(reason):
        dropped[reason] = dropped.get(reason, 0) + 1

    for ep in eps:
        if spec_hash is not None and ep.spec_hash != spec_hash:
            drop("spec")
        elif ep.steps == 0:
            drop("empty")
        elif ep.end_reason in drop_end_reasons:
            drop("end_" + ep.end_reason)
        elif not ep.has_mu.all():
            if missing_mu == "error":
                raise ValueError("%s: %d of %d steps carry no behaviour distribution (R-NaD needs μ at every "
                                 "step; missing_mu='drop' drops such episodes)"
                                 % (getattr(ep, "path", "episode"), int((~ep.has_mu).sum()), ep.steps))
            drop("no_mu")
        elif (ep.chosen < -1).any():
            drop("concession_reply")
        else:
            kept.append(ep)
    return kept, dropped


class RNaDBatch:
    """G full games time-major, padded to the longest and split into chunks (the module docstring's
    table)."""

    def __init__(self, eps, chunk_len=32, policies=(), omniscient=False):
        if not eps:
            raise ValueError("RNaDBatch: no games")
        self.eps = list(eps)
        self.G = G = len(self.eps)
        self.chunk_len = int(chunk_len)
        self.omniscient = omniscient
        self.length = np.array([e.steps for e in self.eps], np.int64)
        self.T = T = max(1, -(-int(self.length.max()) // self.chunk_len)) * self.chunk_len
        self.chunks = T // self.chunk_len
        self.A = A = padded(max(int(e.menu.max()) for e in self.eps))
        self.valid = np.arange(T)[:, None] < self.length[None, :]
        tt, gg = np.nonzero(self.valid)                    # time-major
        self.flat = tt * G + gg
        # Each valid step's (episode, step) as an index into the episodes' concatenation.
        ep_start = _starts(self.length)
        self._row = ep_start[gg] + tt
        self.seat = np.zeros((T, G), np.int64)
        self.sideboarding = np.zeros((T, G), np.bool_)
        self.has_mu = np.zeros((T, G), np.bool_)
        self.chosen = np.full((T, G), -1, np.int64)
        self.policy_version = np.full((T, G), -1, np.int64)
        self.legal = np.zeros((T, G, A), np.bool_)
        self.mu = np.zeros((T, G, A), np.float32)
        self.reward = np.zeros((T, G, 2), np.float32)
        self.evals = {int(p): {"present": np.zeros((T, G), np.bool_), "version": np.full((T, G), -1, np.int64),
                               "log_pi": np.zeros((T, G, A), np.float32)} for p in policies}
        for g, ep in enumerate(self.eps):
            d = ep.steps
            self.seat[:d, g] = ep.seat
            self.sideboarding[:d, g] = ep.sideboarding
            self.has_mu[:d, g] = ep.has_mu
            ch = ep.chosen.copy()
            ch[ch == -1] = ep.menu[ch == -1] - 1      # -1 is the last row (the decision log's convention)
            self.chosen[:d, g] = ch
            self.policy_version[:d, g] = ep.policy_version
            rows = np.repeat(np.arange(d), ep.menu)
            cols = np.arange(int(ep.menu_off[-1])) - np.repeat(ep.menu_off[:-1], ep.menu)
            self.legal[rows, g, cols] = True
            self.mu[rows, g, cols] = ep.mu
            self.reward[d - 1, g] = ep.result
            for p, ev in self.evals.items():
                if ep.eval_policy.shape[1] == 0:
                    continue
                hit = ep.eval_policy == p                      # [D, P]
                has = hit.any(1)
                col = hit.argmax(1)
                ev["present"][:d, g] = has
                ev["version"][:d, g] = np.where(has, ep.eval_version[np.arange(d), col], -1)
                lp = ep.eval_log_pi[np.arange(int(ep.menu_off[-1])), col[rows]]
                ev["log_pi"][rows, g, cols] = np.where(has[rows], lp, 0.0)
        self._decs = _concat_decisions([e.decs for e in self.eps])
        self._omni = _concat_decisions([e.omni for e in self.eps]) if omniscient else None
        if omniscient and any(e.omni is None for e in self.eps):
            raise ValueError("RNaDBatch: omniscient=True needs omniscient episodes")
        self._cache = {}

    def prepare(self):
        """Pads every chunk's inputs now (e.g. on a loader thread), so ``chunk_inputs`` only looks
        them up during the learner step."""
        for c in range(self.chunks):
            for viewer in ("seat", "omni") if self.omniscient else ("seat",):
                self._cache[(c, viewer)] = self._chunk_inputs(c, viewer)
        return self

    def chunk_inputs(self, c, viewer="seat"):
        hit = self._cache.get((c, viewer))
        return hit if hit is not None else self._chunk_inputs(c, viewer)

    def _chunk_inputs(self, c, viewer):
        lo, hi = c * self.chunk_len * self.G, (c + 1) * self.chunk_len * self.G
        a, b = np.searchsorted(self.flat, [lo, hi])
        flat = self.flat[a:b]
        if not len(flat):
            return None, flat
        decs = self._decs if viewer == "seat" else self._omni
        x = pad(decs.select(self._row[a:b]), {"F": decs.arrays["tok_num"].shape[1],
                                               "C": decs.arrays["tok_cat"].shape[1],
                                               "K": decs.arrays["tok_card"].shape[1],
                                               "Q": decs.arrays["tok_ability"].shape[1]},
                {"A": self.A} if viewer == "seat" else None)
        return x, flat


def _concat_decisions(decs_list):
    if any(d is None for d in decs_list):
        return None
    arrays = {name: np.concatenate([d.arrays[name] for d in decs_list]) for name in TENSOR_NAMES}
    sizes = {k: np.concatenate([d.sizes[k] for d in decs_list]) for k in ("n", "r", "a")}
    return Decisions(arrays, sizes)


def rmtraj_module():
    """python/rmtraj.py (the stdlib reader), imported from the repository."""
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(here, "python")
    if path not in sys.path:
        sys.path.insert(0, path)
    import rmtraj
    return rmtraj
