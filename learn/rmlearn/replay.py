"""The learner's RAM replay window over trajectory episodes and its two batch kinds
(docs/rnad_training_plan.md §4 L1).

    win = ReplayWindow(WindowConfig(dirs=["runs/r1/episodes"], max_decisions=4_000_000))
    win.poll()                                   # ingest the episodes that appeared (or win.start())
    items = win.sample_decisions(256, rng)       # uniform over the window's labelled decisions
    db = decision_batch(items, n_cards, depth)   # DecisionBatch: inputs, labels, outcomes, targets
    eps = win.sample_games(64, rng)              # or win.take_fresh(64): each new game once
    tb = TrajectoryBatch(eps, chunk_len=64, policies=(1, 2))   # full games, time-major [T, G]
    for c in range(tb.chunks): x, index = tb.chunk_inputs(c)   # the network inputs, chunk by chunk
    pf = Prefetcher(lambda rng: decision_batch(win.sample_decisions(256, rng), ...).to_torch("cuda"))

**The window** (``ReplayWindow``) holds whole episodes in ingestion order, each as its file's bytes
(``episodes.Episode``: compact arrays, expanded per batch). ``poll`` scans the configured
directories for ``*.rmtraj`` files it has not seen (the engine writes each whole and renames it
into place, so a listed file is complete), oldest modification time first, and ingests them; then
it evicts the oldest episodes while the window holds more than ``max_decisions`` steps or
``max_bytes`` bytes (at least one stays). An evicted episode's file is deleted only with
``delete_evicted``, and only that ``.rmtraj``: decision logs (``.rmlog``) are never touched.
``start()`` polls on a background thread every ``poll_interval`` seconds. Episodes of another spec
than ``spec_hash`` (when set: one hash, or a set — after ``rmtrain retable`` the run's earlier
tables' specs, whose vocab manifests the current one extends at the same layout, so their stored
card and ability ids still read right) and episodes with no steps are skipped and counted; a file that
does not read is counted as an error, never retried.

**The decision-size cap** (``max_decision_tokens``, 0: none): an episode with a decision of more
seat tokens (menu tokens included) is left out whole, so R-NaD's games stay contiguous; it is
counted (``skipped_oversize``, the largest such decision ``oversize_max_tokens``) and logged once
with its path and largest decision. Attention is quadratic in a batch's padded length, so one such
decision pads its whole batch to it. ``set_max_decision_tokens`` changes the cap of a running
window: it applies to the episodes ingested from then on, and a lowered cap also drops the window's
episodes over it.

**Labels** (``LabelConfig``), per step at ingestion: the teacher's label when it has one (≥ 0),
else, with ``chosen_without_mu``, the reply itself when the step has no behaviour distribution (a
replayed or stream-decided seat: the scripted corpus through ``--replay``) and the seat's agent is
not one of ``exclude_agents`` (``explore``, ``random``: their replies are not labels); -1 none.

**Held-out split by game**: a game is held out when the hash of (``split_seed``, its match's
engine seed, its game index, its log's name) falls below ``heldout_fraction``; with ``heldout_by``
"match" the game index is left out, so a match's games fall on the same side. ``split="train"``
samples only the other games; ``iter_decisions("heldout")`` walks the held-out ones in order. With
``probe_fraction`` a training game whose second hash (the same key, ``probe|`` before it) falls
below it is also a probe game: ``iter_decisions("probe")`` walks those, a sample of the training
games for comparing train and held-out metrics.

**Decision batches** (BC): ``sample_decisions`` draws steps uniformly over the split's labelled
steps; ``decision_batch`` expands them (``DecisionBatch``).

**Trajectory batches** (R-NaD, L4): ``TrajectoryBatch`` lays G full games out time-major,
``[T, G]`` with T the longest game rounded up to a multiple of ``chunk_len``, both seats'
decisions interleaved in decision order (an episode's steps are in that order, the sideboarding
before the game first); the network inputs come chunk by chunk (``chunk_inputs``), since a whole
batch expanded would not fit. The terminal reward (each seat's result, +1 / -1 / 0) sits on each
game's last step; every other step's reward is 0.
"""

import collections
import hashlib
import os
import queue
import threading
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import episodes as EP


@dataclass
class LabelConfig:
    teacher: bool = True                 # the teacher's label when present
    chosen_without_mu: bool = True       # else the reply, at steps without μ
    exclude_agents: tuple = ("explore", "random")   # seat agents whose replies are never labels


@dataclass
class WindowConfig:
    dirs: list = field(default_factory=list)
    max_decisions: int = 4_000_000
    max_bytes: int = 40 << 30
    delete_evicted: bool = False
    poll_interval: float = 2.0
    spec_hash: object = None             # skip episodes of another spec: a hash or a set of them (the run's
                                         # tables and their lineage, spec.Tables.accepted_spec_hashes; None:
                                         # take the first's)
    heldout_fraction: float = 0.0
    split_seed: int = 0
    heldout_by: str = "game"             # "game", or "match": a match's games are held out together
    probe_fraction: float = 0.0          # training games also walked by iter_decisions("probe")
    max_decision_tokens: int = 0         # leave out an episode with a decision of more seat tokens (0: none)
    labels: LabelConfig = field(default_factory=LabelConfig)


def spec_accepted(ep, spec_hash):
    """Whether episode `ep` is of `spec_hash` (one hash, or a set / frozenset / list of them)."""
    return ep.spec_hash == spec_hash if isinstance(spec_hash, str) else ep.spec_hash in spec_hash


def step_labels(ep, cfg):
    """[D] i64: each step's label under `cfg` (LabelConfig), -1 none."""
    lab = ep.teacher.astype(np.int64) if cfg.teacher else np.full(ep.steps, -1, np.int64)
    if cfg.chosen_without_mu:
        agents = [str(ep.attrs.get("agent_a", "")), str(ep.attrs.get("agent_b", ""))]
        ok_seat = np.array([not any(a == x or a.startswith(x + ":") for x in cfg.exclude_agents) for a in agents])
        use = (lab < 0) & (ep.has_mu == 0) & ok_seat[ep.seat.astype(np.int64)]
        lab = np.where(use, ep.chosen.astype(np.int64), lab)
    size = np.diff(ep.menu_off)
    return np.where((lab >= 0) & (lab < size), lab, -1)


def heldout(ep, fraction, seed, by="game", salt=""):
    if fraction <= 0:
        return False
    key = ep.match_key() if by == "match" else ep.game_key()
    h = hashlib.blake2b(("%s%d|%s" % (salt, seed, key)).encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") / 2.0 ** 64 < fraction


class ReplayWindow:
    """A bounded window of episodes in RAM (see the module docstring)."""

    def __init__(self, cfg, log=None):
        self.cfg = cfg
        self.log = log or (lambda msg: None)
        self.spec_hash = cfg.spec_hash
        self._eps = collections.deque()
        self._fresh = collections.deque()
        self._seen = set()
        self._lock = threading.Lock()
        self._poll_lock = threading.Lock()      # one poll at a time (the thread's and a caller's)
        self._version = 0
        self._index = {}
        self._thread = None
        self._stop = threading.Event()
        self.decisions = 0
        self.bytes = 0
        self.counts = collections.Counter()
        self.oversize_max_tokens = 0
        self.last_error = ""

    # ── ingestion and eviction ──

    def poll(self):
        """Ingests every new episode of the directories; the number ingested. Safe to call beside
        the polling thread (``start``): one poll runs at a time."""
        with self._poll_lock:
            return self._poll()

    def _poll(self):
        new = []
        for d in self.cfg.dirs:
            try:
                it = os.scandir(d)
            except FileNotFoundError:
                continue
            with it:
                for e in it:
                    if e.name.endswith(".rmtraj") and e.path not in self._seen and e.is_file():
                        try:
                            new.append((e.stat().st_mtime, e.path))
                        except FileNotFoundError:
                            continue
        new.sort()
        n = 0
        for _, path in new:
            self._seen.add(path)
            try:
                ep = EP.read_episode(path)
            except (OSError, ValueError) as err:
                self.counts["errors"] += 1
                self.last_error = "%s: %s" % (path, err)
                continue
            if self.add(ep):
                n += 1
        return n

    def add(self, ep, fresh=True):
        """Adds one Episode (ingestion order = age); False when skipped. With `fresh` False it is not
        offered by ``take_fresh`` (a resumed learner's episodes it had already taken)."""
        if not ep.steps:
            self.counts["skipped_empty"] += 1
            return False
        if self.spec_hash is None:
            self.spec_hash = ep.spec_hash
        if not spec_accepted(ep, self.spec_hash):
            self.counts["skipped_spec"] += 1
            return False
        ep.max_tokens = int(np.diff(ep.seat_feat.tok_off).max())
        if self._over_cap(ep):
            with self._lock:
                self._note_oversize(ep)
            return False
        ep.labels = step_labels(ep, self.cfg.labels)
        ep.split = "heldout" if heldout(ep, self.cfg.heldout_fraction, self.cfg.split_seed,
                                        self.cfg.heldout_by) else "train"
        ep.probe = ep.split == "train" and heldout(ep, self.cfg.probe_fraction, self.cfg.split_seed,
                                                   self.cfg.heldout_by, "probe|")
        ep.added_at = time.time()
        with self._lock:
            self._eps.append(ep)
            if fresh:
                self._fresh.append(ep)
            self.decisions += ep.steps
            self.bytes += ep.nbytes
            self.counts["ingested"] += 1
            self._evict()
            self._version += 1
        return True

    def _evict(self):
        cfg = self.cfg
        while len(self._eps) > 1 and (self.decisions > cfg.max_decisions or self.bytes > cfg.max_bytes):
            ep = self._eps.popleft()
            self.decisions -= ep.steps
            self.bytes -= ep.nbytes
            self.counts["evicted"] += 1
            ep.evicted = True
            if cfg.delete_evicted and ep.path and ep.path.endswith(".rmtraj"):
                try:
                    os.unlink(ep.path)
                    self.counts["deleted"] += 1
                except FileNotFoundError:
                    pass
        while self._fresh and getattr(self._fresh[0], "evicted", False):
            self._fresh.popleft()

    def _over_cap(self, ep):
        cap = self.cfg.max_decision_tokens
        return cap > 0 and ep.max_tokens > cap

    def _note_oversize(self, ep):
        self.counts["skipped_oversize"] += 1
        self.oversize_max_tokens = max(self.oversize_max_tokens, ep.max_tokens)
        self.log("left out %s: a decision of %d seat tokens (replay.max_decision_tokens %d)"
                 % (ep.path or ep.name, ep.max_tokens, self.cfg.max_decision_tokens))

    def set_max_decision_tokens(self, cap):
        """The decision-size cap from now on; a lowered cap drops the window's episodes over it. The
        number dropped."""
        self.cfg.max_decision_tokens = int(cap)
        with self._lock:
            over = [e for e in self._eps if self._over_cap(e)]
            if not over:
                return 0
            for e in over:
                self.decisions -= e.steps
                self.bytes -= e.nbytes
                e.evicted = True
                self._note_oversize(e)
            self._eps = collections.deque(e for e in self._eps if not getattr(e, "evicted", False))
            self._fresh = collections.deque(e for e in self._fresh if not getattr(e, "evicted", False))
            self._version += 1
        return len(over)

    def skip(self, paths):
        """Marks `paths` as seen: ``poll`` never ingests them (a resume leaving older files out)."""
        self._seen.update(paths)

    def files(self):
        """(the window's episode paths in ingestion order, the paths not yet taken by take_fresh)."""
        with self._lock:
            return [e.path for e in self._eps], [e.path for e in self._fresh]

    def start(self):
        """Polls on a background thread every poll_interval seconds."""
        if self._thread:
            return
        self._stop.clear()

        def run():
            while not self._stop.is_set():
                try:
                    self.poll()
                except Exception as err:  # keep polling; the error is reported
                    self.counts["errors"] += 1
                    self.last_error = repr(err)
                self._stop.wait(self.cfg.poll_interval)

        self._thread = threading.Thread(target=run, name="replay-poll", daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join()
            self._thread = None

    # ── views ──

    def __len__(self):
        return len(self._eps)

    def episodes(self, split=None):
        with self._lock:
            return [e for e in self._eps if split is None or e.split == split]

    def stats(self):
        with self._lock:
            labelled = sum(int((e.labels >= 0).sum()) for e in self._eps)
            split = collections.Counter(e.split for e in self._eps)
            return {"episodes": len(self._eps), "decisions": self.decisions, "labelled": labelled,
                    "bytes": self.bytes, "train_episodes": split["train"], "heldout_episodes": split["heldout"],
                    "fresh": len(self._fresh), "max_decision_tokens": self.cfg.max_decision_tokens,
                    "oversize_max_tokens": self.oversize_max_tokens, **dict(self.counts),
                    "last_error": self.last_error}

    def _split_index(self, split, labelled):
        key = (split, labelled)
        idx = self._index.get(key)
        if idx is None or idx[0] != self._version:
            eps = [e for e in self._eps if split is None or e.split == split or (split == "probe" and e.probe)]
            steps = [np.nonzero(e.labels >= 0)[0] if labelled else np.arange(e.steps) for e in eps]
            cum = np.zeros(len(eps) + 1, np.int64)
            np.cumsum([len(s) for s in steps], out=cum[1:])
            idx = (self._version, eps, steps, cum)
            self._index[key] = idx
        return idx

    def sample_decisions(self, n, rng, split="train", labelled=True):
        """`n` steps [(Episode, step)] drawn uniformly (with replacement) over the split's steps
        (only the labelled ones with `labelled`)."""
        with self._lock:
            _, eps, steps, cum = self._split_index(split, labelled)
        total = int(cum[-1])
        if not total:
            raise ValueError("the window has no %s%s decisions" % ("labelled " if labelled else "", split or ""))
        i = rng.integers(0, total, n)
        e = np.searchsorted(cum, i, side="right") - 1
        return [(eps[a], int(steps[a][i[k] - cum[a]])) for k, a in enumerate(e)]

    def iter_decisions(self, split="heldout", labelled=True, batch=256, limit=None):
        """The split's (labelled) steps in window order, `batch` at a time (at most `limit`)."""
        with self._lock:
            _, eps, steps, _ = self._split_index(split, labelled)
        out = []
        n = 0
        for ep, st in zip(eps, steps):
            for d in st:
                if limit is not None and n >= limit:
                    break
                out.append((ep, int(d)))
                n += 1
                if len(out) == batch:
                    yield out
                    out = []
        if out:
            yield out

    def sample_games(self, n, rng, split="train", require_mu=False):
        """`n` episodes drawn uniformly (with replacement) from the split (with `require_mu`, only
        those whose every step has μ)."""
        eps = [e for e in self.episodes(split) if not require_mu or bool(e.has_mu.all())]
        if not eps:
            raise ValueError("the window has no %s games" % split)
        return [eps[i] for i in rng.integers(0, len(eps), n)]

    def take_fresh(self, n, timeout=None, split="train", require_mu=False, newest=False):
        """The `n` oldest (`newest`: the newest) episodes not yet taken (each episode is taken once),
        waiting up to `timeout` seconds (None: forever) for enough; fewer only on timeout."""
        out = []
        end = None if timeout is None else time.time() + timeout
        while True:
            with self._lock:
                while self._fresh and len(out) < n:
                    e = self._fresh.pop() if newest else self._fresh.popleft()
                    if e.split == split and (not require_mu or bool(e.has_mu.all())):
                        out.append(e)
            if len(out) >= n or (end is not None and time.time() >= end):
                return out
            if self._thread is None:
                self.poll()
            time.sleep(min(self.cfg.poll_interval, 0.2))


# ── decision batches ─────────────────────────────────────────────────────────────────────────

class DecisionBatch:
    """Expanded decisions: ``x`` (the ten padded inputs, numpy), ``label`` [B] (-1 none),
    ``outcome`` [B] f32 (the deciding seat's game result), ``seat``, ``sideboarding``, ``has_mu``,
    ``chosen`` [B], ``targets`` (aux targets, None when no step has them), ``omni`` (the
    omniscient inputs, with ``omniscient=True``), ``items``."""

    def __init__(self, items, n_cards=None, depth=None, omniscient=False):
        self.items = items
        self.x = EP.expand(items)
        self.sizes = self.x.pop("sizes")
        self.label = np.fromiter((ep.labels[d] for ep, d in items), np.int64, len(items))
        self.outcome = np.fromiter((ep.result[ep.seat[d]] for ep, d in items), np.float32, len(items))
        for k in ("seat", "sideboarding", "has_mu", "chosen"):
            setattr(self, k, EP.step_field(items, k))
        self.targets = None
        if n_cards is not None and any(ep.targets is not None for ep, _ in items):
            self.targets = EP.targets(items, n_cards, depth)
        self.omni = None
        if omniscient:
            self.omni = EP.expand(items, "omni")
            self.omni.pop("sizes")

    def __len__(self):
        return len(self.items)

    def to_torch(self, device="cpu", pin=False):
        """A TorchDecisions: every array as a tensor on `device` (pinned and copied non-blocking
        when `pin` and the device is not the CPU)."""
        return TorchDecisions(self, device, pin)


def _tensor(a, device, pin):
    t = torch.from_numpy(np.ascontiguousarray(a))
    if str(device) == "cpu":
        return t
    if pin:
        t = t.pin_memory()
    return t.to(device, non_blocking=pin)


class TorchDecisions:
    """A DecisionBatch on a device: ``inputs`` (the ten tensors in contract order), ``label``,
    ``outcome``, ``seat``, ``sideboarding``, ``targets`` (dict or None), ``omni`` (list or None),
    ``n`` (decisions)."""

    def __init__(self, db, device, pin):
        self.n = len(db)
        self.inputs = [_tensor(db.x[k], device, pin) for k in EP.INPUTS]
        self.label = _tensor(db.label, device, pin)
        self.outcome = _tensor(db.outcome, device, pin)
        self.seat = _tensor(db.seat, device, pin)
        self.sideboarding = _tensor(db.sideboarding.astype(np.bool_), device, pin)
        self.targets = None if db.targets is None else {k: _tensor(v, device, pin) for k, v in db.targets.items()}
        self.omni = None if db.omni is None else [_tensor(db.omni[k], device, pin) for k in EP.INPUTS]


def decision_batch(items, n_cards=None, depth=None, omniscient=False):
    return DecisionBatch(items, n_cards, depth, omniscient)


# ── trajectory batches ───────────────────────────────────────────────────────────────────────

class TrajectoryBatch:
    """G full games, time-major. Arrays (numpy), T = the longest game rounded up to ``chunk_len``:

    - ``valid`` [T, G] bool: a real step; ``length`` [G]; ``result`` [G, 2] (seat A, B)
    - ``seat`` [T, G] i64 (0 = A; 0 at padding), ``sideboarding``, ``has_mu`` [T, G] bool
    - ``chosen`` [T, G] i64 (the reply; -1 at padding and for a concession sentinel), ``teacher``,
      ``policy_version`` [T, G] i64
    - ``legal`` [T, G, A] bool (the menu rows), ``mu`` [T, G, A] f32 (0 without μ)
    - ``reward`` [T, G, 2] f32: each seat's result at its game's last step, 0 elsewhere
    - ``evals[p]`` per requested policy id p: ``present`` [T, G] bool, ``version`` [T, G] i64,
      ``value`` [T, G] f32, ``log_pi`` [T, G, A] f32 (0 where absent), as the actor recorded them
    - ``chunks``: T / chunk_len; ``chunk_inputs(c)`` → (the ten padded inputs of the valid steps of
      time steps [c·chunk_len, (c+1)·chunk_len), time-major then game; their flat index t·G + g)
    """

    def __init__(self, eps, chunk_len=64, policies=(), omniscient=False):
        if not eps:
            raise ValueError("TrajectoryBatch: no games")
        self.eps = list(eps)
        self.G = G = len(eps)
        self.chunk_len = chunk_len
        self.omniscient = omniscient
        self.length = np.array([e.steps for e in eps], np.int64)
        self.T = T = max(1, -(-int(self.length.max()) // chunk_len)) * chunk_len
        self.chunks = T // chunk_len
        self.valid = np.arange(T)[:, None] < self.length[None, :]
        self.result = np.array([e.result for e in eps], np.float32)
        tt, gg = np.nonzero(self.valid)                       # time-major order
        self.flat = tt * G + gg
        self.items = [(eps[g], int(t)) for t, g in zip(tt, gg)]
        A = EP.padded(max(int(np.diff(e.menu_off).max()) for e in eps))
        self.A = A

        def grid(name, dtype, fill=0):
            out = np.full((T, G), fill, dtype)
            out[tt, gg] = EP.step_field(self.items, name, dtype)
            return out

        self.seat = grid("seat", np.int64)
        self.sideboarding = grid("sideboarding", np.int64).astype(np.bool_)
        self.has_mu = grid("has_mu", np.int64).astype(np.bool_)
        self.chosen = grid("chosen", np.int64, -1)
        self.chosen[self.chosen < 0] = -1
        self.teacher = grid("teacher", np.int64, -1)
        self.policy_version = grid("policy_version", np.int64, -1)
        sizes = np.zeros((T, G), np.int64)
        sizes[tt, gg] = [ep.menu_size(d) for ep, d in self.items]
        self.legal = np.arange(A)[None, None, :] < sizes[:, :, None]
        self.mu = np.zeros((T, G, A), np.float32)
        self.mu[tt, gg] = EP.menu_rows(self.items, A)
        self.reward = np.zeros((T, G, 2), np.float32)
        self.reward[self.length - 1, np.arange(G)] = self.result
        self.evals = {}
        for p in policies:
            ev = EP.evaluations(self.items, p, A)
            out = {"present": np.zeros((T, G), np.bool_), "version": np.full((T, G), -1, np.int64),
                   "value": np.zeros((T, G), np.float32), "log_pi": np.zeros((T, G, A), np.float32)}
            for k in out:
                out[k][tt, gg] = ev[k]
            self.evals[p] = out

    def chunk_items(self, c):
        """The (Episode, step) items of chunk c (valid steps, time-major) and their flat indices."""
        lo, hi = c * self.chunk_len * self.G, (c + 1) * self.chunk_len * self.G
        a, b = np.searchsorted(self.flat, [lo, hi])
        return self.items[a:b], self.flat[a:b]

    def chunk_inputs(self, c, viewer="seat", min_dims=None):
        """(padded inputs {name: ndarray} with ``sizes``, flat index [n]) of chunk c; (None, empty)
        when the chunk holds no valid step."""
        items, flat = self.chunk_items(c)
        if not items:
            return None, flat
        return EP.expand(items, viewer, min_dims or {"A": self.A}), flat

    def chunk_targets(self, c, n_cards, depth):
        items, _ = self.chunk_items(c)
        return EP.targets(items, n_cards, depth) if items else None


# ── a background loader ──────────────────────────────────────────────────────────────────────

class Prefetcher:
    """Runs ``make(rng)`` on ``workers`` background threads (each its own generator from
    ``seed``) and keeps up to ``depth`` results queued; ``get()`` takes the next (and adds the time
    it waited to ``wait_s``). An exception in ``make`` is re-raised by ``get``."""

    def __init__(self, make, depth=4, workers=1, seed=0):
        self.make = make
        self.q = queue.Queue(maxsize=depth)
        self.stop_ev = threading.Event()
        self.wait_s = 0.0
        self.made = 0
        self.threads = []
        for k in range(workers):
            t = threading.Thread(target=self._run, args=(np.random.default_rng([seed, k]),), name="prefetch-%d" % k,
                                 daemon=True)
            t.start()
            self.threads.append(t)

    def _run(self, rng):
        while not self.stop_ev.is_set():
            try:
                item = (True, self.make(rng))
            except Exception as err:  # handed to the consumer
                item = (False, err)
            while not self.stop_ev.is_set():
                try:
                    self.q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if not item[0]:
                return

    def get(self):
        t = time.time()
        ok, item = self.q.get()
        self.wait_s += time.time() - t
        if not ok:
            raise item
        self.made += 1
        return item

    def stop(self):
        self.stop_ev.set()
        for t in self.threads:
            t.join()
