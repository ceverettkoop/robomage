"""The learner's inference server (docs/rnad_training_plan.md §4 L5): RMIP v1
(docs/inference_protocol.md) as a component inside a host process.

    srv = InferenceServer("/run/x/infer.sock", device="cuda", max_batch=512, deadline_us=2000)
    srv.set_policy(0, model, version=12)          # an RMNet (folded here), an InferencePolicy,
    srv.set_policy(1, reg_model, version=11)      # an RMNet state_dict (with template=), or a .ts.pt path
    srv.start()                                    # binds the socket, starts its threads
    ...                                            # actors connect with nn:remote:<socket>[:policies=0,1,2]
    srv.set_policy(0, model, version=13)          # swapped between batches; clients never wait on it
    srv.stats()                                    # dict: batches, rows/s, batch sizes, queue latency, ...
    srv.stop()                                     # ERROR shutdown to every client, socket removed

**Threads.** An I/O thread accepts clients, answers handshakes (refusals as the protocol's
``ERROR`` codes), reads ``EVAL`` requests from each client's shared region (validated: layout,
dims, policies, every index within its decision and every id within the model's tables, so a bad
request can never reach the device) and queues them. A compute thread forms batches by the
``rmserve`` rules: queued requests in arrival order while their decisions fit ``max_batch`` (a larger
request is a batch of its own), evaluated when the queued decisions reach ``max_batch``, when
every connected client has a request queued (nobody else can add to the batch), or
``deadline_us`` after the oldest queued request arrived. It pads the batch as
``src/featurize/batch.cpp`` does, runs each requested policy once per batch (under
``torch.inference_mode()``, fp32 or, with ``precision="fp16"|"bf16"``, autocast), and writes each
response (per policy its version, values, logits and ``log π`` over the real menu rows) into the
client's region before sending ``DONE``.

**Policies** are held by id with a version each. ``set_policy`` builds the new module on the
device outside any lock and swaps it in under a short lock; a batch already running keeps the
module it started with, and every response carries the version it was evaluated with.
"""

import collections
import copy
import json
import mmap
import os
import select
import socket
import struct
import sys
import threading
import time

import numpy as np
import torch

from .batch import Decisions, INPUTS, TENSORS, pad
from .model import InferencePolicy, RMNet

MSG = struct.Struct("<4sHHI")
REQ = struct.Struct("<4sIQQII4IQQ")
RESP = struct.Struct("<4sIQQIIIIQQ")
HELLO, WELCOME, EVAL, DONE, ERROR, BYE = 1, 2, 3, 4, 5, 6
MAX_BODY = 1 << 20
PROTOCOL = 1
# The request's tensors: name, dtype, the size array counting its rows, its row width.
ARRAYS = [("tok_type", np.int64, "n", None), ("tok_num", np.float32, "n", "F"), ("tok_cat", np.int64, "n", "C"),
          ("tok_card", np.int64, "n", "K"), ("tok_ability", np.int64, "n", "Q"), ("tok_mask", np.uint8, "n", None),
          ("rel", np.int64, "r", 3), ("act_tok", np.int64, "a", None), ("act_mask", np.uint8, "a", None)]
STAT_WINDOW = 2000  # batches kept for the rolling statistics


def _align8(x):
    return (x + 7) & ~7


class Refused(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class _Region:
    """A client's POSIX shared-memory region mapped read-write (`buf`, `size`). Dropping the last
    reference unmaps it once no array view of it remains (no close() that could fail on live views)."""

    def __init__(self, name):
        import _posixshmem
        fd = _posixshmem.shm_open(name, os.O_RDWR, mode=0o600)
        try:
            self.size = os.fstat(fd).st_size
            self.buf = mmap.mmap(fd, self.size)
        finally:
            os.close(fd)


class _Policy:
    """A served policy: its module on the device, version and the bounds requests are checked against."""

    def __init__(self, module, version):
        self.module = module
        self.version = int(version)
        self.spec_hash = module.spec_hash
        self.vocab_hash = module.vocab_hash
        self.text_model = module.text_model
        emb = module.encoder.embed
        self.bounds = {"types": emb.type_emb.num_embeddings, "cat": emb.cat_emb.num_embeddings,
                       "cards": module.none_card, "abilities": module.none_ability,
                       "rel": max(module.encoder.n_rel, 1)}


class _Client:
    def __init__(self, sock, cid):
        self.sock = sock
        self.cid = cid
        self.buf = b""
        self.shm = None
        self.hello = None
        self.lock = threading.Lock()
        self.alive = True
        self.queued = False     # a request is queued or being evaluated

    def send(self, kind, body=b""):
        with self.lock:
            if not self.alive:
                return
            try:
                self.sock.sendall(MSG.pack(b"RMIP", kind, 0, len(body)) + body)
            except OSError:
                self.alive = False


class _Request:
    __slots__ = ("client", "seq", "spec_hash", "decs", "policies", "arrived", "d", "first")

    def __init__(self, client, seq, spec_hash, decs, policies, arrived):
        self.client, self.seq, self.spec_hash, self.decs, self.policies, self.arrived = \
            client, seq, spec_hash, decs, policies, arrived
        self.d = len(decs)
        self.first = 0          # its first decision's index in the batch


class InferenceServer:
    """An RMIP v1 server running in the host process (see the module docstring)."""

    def __init__(self, socket_path, device="cuda", max_batch=512, deadline_us=2000, precision="fp32",
                 chunk=64, template=None, name="rmlearn", log=None, pause_after=None):
        if precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision is fp32, fp16 or bf16")
        self.socket_path = socket_path
        self.device = torch.device(device)
        self.max_batch = int(max_batch)
        self.deadline = deadline_us / 1e6
        self.precision = precision
        self.chunk = int(chunk)           # most decisions per forward (a batch runs in chunks)
        self.template = template          # an RMNet whose state_dicts set_policy may receive
        self.name = name
        self.pause_after = pause_after    # tests: stop answering after this many requests
        self._log = log or (lambda msg: print("rmlearn.server: " + msg, file=sys.stderr, flush=True))
        self._policies = {}
        self._plock = threading.Lock()
        self._queue = collections.deque()
        self._cv = threading.Condition()
        self._clients = {}
        self._next_cid = 0
        self._stop = threading.Event()
        self._threads = []
        self._listener = None
        self._wake_r, self._wake_w = None, None
        self._stats = {"batches": 0, "decisions": 0, "requests": 0, "refusals": 0, "errors": 0, "clients_total": 0,
                       "max_batch_seen": 0, "started": None}
        self._recent = collections.deque(maxlen=STAT_WINDOW)  # (end time, decisions, queue wait s, compute s)
        self._slock = threading.Lock()

    # ── policies ──────────────────────────────────────────────────────────────────────────

    def _module_for(self, pid, policy, version):
        if isinstance(policy, RMNet):
            m = policy.inference_module(version)
        elif isinstance(policy, InferencePolicy):
            m = copy.deepcopy(policy)
        elif isinstance(policy, str):
            m = torch.jit.load(policy, map_location="cpu")
        elif isinstance(policy, dict):
            if any(k.startswith("cards.") for k in policy):
                if self.template is None:
                    raise ValueError("an RMNet state_dict needs the server's template model")
                t = copy.deepcopy(self.template).cpu()
                t.load_state_dict(policy)
                m = t.inference_module(version)
            else:
                with self._plock:
                    cur = self._policies.get(pid)
                if cur is None:
                    raise ValueError("an InferencePolicy state_dict needs policy %d to exist already" % pid)
                m = copy.deepcopy(cur.module).cpu()
                m.load_state_dict(policy)
        else:
            raise TypeError("set_policy takes an RMNet, an InferencePolicy, a state_dict or a .ts.pt path")
        if hasattr(m, "policy_version") and not isinstance(m, torch.jit.ScriptModule):
            m.policy_version = str(int(version))
        return m.to(self.device).eval()

    def set_policy(self, pid, policy, version):
        """Serves `policy` as id `pid` at `version` from the next batch on."""
        p = _Policy(self._module_for(int(pid), policy, version), version)
        with self._plock:
            self._policies[int(pid)] = p

    def remove_policy(self, pid):
        with self._plock:
            self._policies.pop(int(pid), None)

    def policies(self):
        with self._plock:
            return {pid: p.version for pid, p in self._policies.items()}

    # ── lifecycle ─────────────────────────────────────────────────────────────────────────

    def start(self):
        path = self.socket_path
        tmp = path + ".tmp%d" % os.getpid()
        for p in (path, tmp):
            if os.path.exists(p):
                os.unlink(p)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(tmp)
        self._listener.listen(256)
        os.rename(tmp, path)
        self._wake_r, self._wake_w = os.pipe()
        self._stats["started"] = time.time()
        self._threads = [threading.Thread(target=self._io_loop, name="rmip-io", daemon=True),
                         threading.Thread(target=self._compute_loop, name="rmip-compute", daemon=True)]
        for t in self._threads:
            t.start()
        self._log("listening on %s (%s, max_batch %d, deadline %d us, %s)"
                  % (path, self.device, self.max_batch, int(self.deadline * 1e6), self.precision))

    def stop(self):
        """Stops serving: every client gets ERROR shutdown, the socket is removed."""
        if self._stop.is_set():
            return
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._wake_w is not None:
            os.write(self._wake_w, b"x")
        for t in self._threads:
            t.join(timeout=30)
        with self._cv:
            clients = list(self._clients.values())
        for c in clients:
            if c.hello is not None:
                c.send(ERROR, json.dumps({"code": "shutdown", "message": "the server is stopping"}).encode())
            self._drop(c)
        if self._listener is not None:
            self._listener.close()
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
        for fd in (self._wake_r, self._wake_w):
            if fd is not None:
                os.close(fd)
        self._wake_r = self._wake_w = None

    # ── statistics ────────────────────────────────────────────────────────────────────────

    def stats(self):
        """A snapshot: totals, connected clients, policies and versions, and over the recent batches
        (up to STAT_WINDOW, at most the last 60 s) rows/s, batch size mean / max, queue latency mean /
        p50 / p99 (ms) and compute time per batch (ms)."""
        now = time.time()
        with self._slock:
            s = dict(self._stats)
            recent = [r for r in self._recent if now - r[0] <= 60.0]
        s["uptime_s"] = now - s["started"] if s["started"] else 0.0
        with self._cv:
            s["clients"] = sum(1 for c in self._clients.values() if c.hello is not None)
        s["queued_requests"] = len(self._queue)
        s["policies"] = self.policies()
        s["precision"] = self.precision
        s["device"] = str(self.device)
        if recent:
            span = max(now - min(r[0] - r[3] for r in recent), 1e-6)
            sizes = np.array([r[1] for r in recent], np.float64)
            waits = np.array([r[2] for r in recent], np.float64) * 1e3
            comp = np.array([r[3] for r in recent], np.float64) * 1e3
            s.update({"window_batches": len(recent), "rows_per_s": float(sizes.sum() / span),
                      "batch_mean": float(sizes.mean()), "batch_max": int(sizes.max()),
                      "queue_ms_mean": float(waits.mean()), "queue_ms_p50": float(np.percentile(waits, 50)),
                      "queue_ms_p99": float(np.percentile(waits, 99)), "compute_ms_mean": float(comp.mean())})
        if self.device.type == "cuda":
            s["vram_allocated_mb"] = torch.cuda.memory_allocated(self.device) / 2 ** 20
            s["vram_peak_mb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        return s

    # ── I/O thread ────────────────────────────────────────────────────────────────────────

    def _drop(self, c):
        with c.lock:
            c.alive = False
        with self._cv:
            self._clients.pop(c.sock, None)
        try:
            c.sock.close()
        except OSError:
            pass
        c.shm = None    # unmapped once the last array view of it goes
        with self._cv:
            self._cv.notify_all()

    def _io_loop(self):
        while not self._stop.is_set():
            with self._cv:
                socks = [self._listener] + list(self._clients)
            readable, _, _ = select.select(socks + [self._wake_r], [], [], 0.5)
            for s in readable:
                if self._stop.is_set():
                    break
                if s == self._wake_r:
                    continue
                if s is self._listener:
                    conn, _ = self._listener.accept()
                    with self._cv:
                        self._clients[conn] = _Client(conn, self._next_cid)
                    self._next_cid += 1
                    with self._slock:
                        self._stats["clients_total"] += 1
                    continue
                with self._cv:
                    c = self._clients.get(s)
                if c is None:
                    continue
                try:
                    data = s.recv(1 << 16)
                except OSError:
                    data = b""
                keep = bool(data)
                c.buf += data
                while keep and len(c.buf) >= MSG.size:
                    magic, kind, _, length = MSG.unpack_from(c.buf, 0)
                    if magic != b"RMIP" or length > MAX_BODY:
                        self._refuse(c, Refused("protocol", "not an RMIP message"))
                        keep = False
                        break
                    if len(c.buf) < MSG.size + length:
                        break
                    body, c.buf = c.buf[MSG.size:MSG.size + length], c.buf[MSG.size + length:]
                    try:
                        keep = self._handle(c, kind, body)
                    except Refused as e:
                        self._refuse(c, e)
                        keep = False
                if not keep:
                    self._drop(c)

    def _refuse(self, c, e):
        with self._slock:
            self._stats["refusals"] += 1
        c.send(ERROR, json.dumps({"code": e.code, "message": str(e)}).encode())
        try:
            c.sock.shutdown(socket.SHUT_RDWR)   # the I/O thread sees the end and drops the client
        except OSError:
            pass
        self._log("refused %s: %s: %s" % (c.hello["client"] if c.hello else "client %d" % c.cid, e.code, e))

    def _handle(self, c, kind, body):
        if kind == HELLO and c.hello is None:
            self._handshake(c, body)
        elif kind == EVAL and c.hello is not None:
            if c.queued:
                raise Refused("protocol", "a second EVAL before the first one's DONE")
            if len(body) != 16:
                raise Refused("protocol", "an EVAL body is u64 seq, u64 length")
            seq, length = struct.unpack("<QQ", body)
            req = self._read_request(c, seq, length)
            c.queued = True
            with self._cv:
                self._queue.append(req)
                self._cv.notify_all()
        elif kind == BYE:
            return False
        else:
            raise Refused("protocol", "unexpected message type %d" % kind)
        return True

    def _handshake(self, c, body):
        try:
            h = json.loads(body)
        except ValueError:
            raise Refused("protocol", "the HELLO is not JSON")
        if not isinstance(h, dict) or h.get("protocol") != PROTOCOL:
            raise Refused("protocol", "this server speaks protocol %d" % PROTOCOL)
        need = ("spec_hash", "vocab_hash", "text_model", "dims", "shm", "request_bytes", "response_bytes", "policies")
        if any(k not in h for k in need) or not isinstance(h["policies"], list) or not h["policies"] or \
                len(set(h["policies"])) != len(h["policies"]) or not isinstance(h["dims"], dict):
            raise Refused("protocol", "the HELLO lacks a key or its policies are empty or repeated")
        with self._plock:
            pols = dict(self._policies)
        for pid in h["policies"]:
            if pid not in pols:
                raise Refused("policy", "policy %s is not served" % pid)
            for key in ("spec_hash", "vocab_hash", "text_model"):
                if getattr(pols[pid], key) != h[key]:
                    raise Refused(key, "the client's %s %s is not policy %d's (%s)"
                                  % (key, h[key], pid, getattr(pols[pid], key)))
        try:
            dims = {k: int(h["dims"][k]) for k in "FCKQ"}
        except (KeyError, TypeError, ValueError):
            raise Refused("protocol", "dims lacks F, C, K or Q")
        if h["request_bytes"] % 8 or h["response_bytes"] % 8 or h["request_bytes"] < 64 or h["response_bytes"] < 56:
            raise Refused("protocol", "the areas' sizes are not multiples of 8 or too small")
        try:
            c.shm = _Region(str(h["shm"]))
        except (OSError, ValueError) as e:
            raise Refused("shm", "cannot open %s: %s" % (h["shm"], e))
        if c.shm.size < h["request_bytes"] + h["response_bytes"]:
            raise Refused("shm", "the region is smaller than its areas")
        h["dims"] = dims
        c.hello = h
        welcome = {"protocol": PROTOCOL, "server": self.name, "max_batch": self.max_batch,
                   "policies": [{"id": pid, "version": pols[pid].version} for pid in h["policies"]]}
        c.send(WELCOME, json.dumps(welcome).encode())
        self._log("%s connected" % h.get("client", "?"))

    def _read_request(self, c, seq, length):
        """The request in c's region as a _Request of array views (validated)."""
        h = c.hello
        buf = c.shm.buf
        if length > h["request_bytes"] or length < 64:
            raise Refused("request", "the request's length %d is outside the request area" % length)
        magic, proto, spec_hash, rseq, d, p, f, cc, k, q, rlen, _ = REQ.unpack_from(buf, 0)
        if magic != b"RMRQ" or proto != PROTOCOL or rseq != seq or rlen != length or d < 1:
            raise Refused("request", "not a request of protocol 1 matching its EVAL (magic, protocol, seq, length)")
        if "%016x" % spec_hash != h["spec_hash"]:
            raise Refused("spec_hash", "the request's spec_hash is not the handshake's")
        dims = {"F": f, "C": cc, "K": k, "Q": q}
        if dims != h["dims"]:
            raise Refused("request", "the request's dims are not the handshake's")
        off = 64
        if off + 4 * p > length:
            raise Refused("request", "the policy ids pass the request's end")
        policies = np.frombuffer(buf, np.uint32, p, off).tolist()
        if policies != h["policies"]:
            raise Refused("request", "the request's policies are not the handshake's")
        off = _align8(off + 4 * p)
        sizes = {}
        for key in ("n", "r", "a"):
            if off + 8 * d > length:
                raise Refused("request", "the sizes pass the request's end")
            sizes[key] = np.frombuffer(buf, np.int64, d, off)
            off = _align8(off + 8 * d)
        if (sizes["n"] < 1).any() or (sizes["a"] < 1).any() or (sizes["r"] < 0).any():
            raise Refused("request", "a decision with no token, no menu row or negative relations")
        width = {"F": f, "C": cc, "K": k, "Q": q, 3: 3}
        arrays = {}
        for name, dtype, key, w in ARRAYS:
            rows = int(sizes[key].sum())
            cols = width[w] if w is not None else 1
            nbytes = rows * cols * np.dtype(dtype).itemsize
            if off + nbytes > length:
                raise Refused("request", "the request ends inside " + name)
            flat = np.frombuffer(buf, dtype, rows * cols, off)
            arrays[name] = flat.reshape(rows, cols) if w is not None else flat
            off = _align8(off + nbytes)
        if off != _align8(length):
            raise Refused("request", "the request's length %d is not its arrays' end %d" % (length, off))
        self._check_indices(h["policies"], sizes, arrays)
        decs = Decisions(arrays, {k2: v for k2, v in sizes.items()})
        return _Request(c, seq, spec_hash, decs, h["policies"], time.perf_counter())

    def _check_indices(self, pids, sizes, arrays):
        with self._plock:
            bounds = [self._policies[p].bounds for p in pids if p in self._policies]
        if len(bounds) != len(pids):
            raise Refused("policy", "a requested policy is no longer served")
        n = sizes["n"]
        n_of_act = np.repeat(n, sizes["a"])
        at = arrays["act_tok"]
        if ((at < 0) | (at >= n_of_act)).any():
            raise Refused("request", "an act_tok index outside its decision's tokens")
        rel = arrays["rel"]
        n_of_rel = np.repeat(n, sizes["r"])
        if len(rel) and (((rel[:, 0] < 0) | (rel[:, 0] >= n_of_rel)).any() or
                         ((rel[:, 1] < 0) | (rel[:, 1] >= n_of_rel)).any()):
            raise Refused("request", "a rel index outside its decision's tokens")
        for b in bounds:
            if len(rel) and ((rel[:, 2] < 0) | (rel[:, 2] >= b["rel"])).any():
                raise Refused("request", "a relation type outside the model's")
            tt = arrays["tok_type"]
            if ((tt < 0) | (tt >= b["types"])).any():
                raise Refused("request", "a token type outside the model's")
            tc = arrays["tok_cat"]
            if ((tc < 0) | (tc >= b["cat"])).any():
                raise Refused("request", "a categorical id outside the model's")
            card = arrays["tok_card"]
            if ((card < -1) | (card >= b["cards"])).any():
                raise Refused("request", "a card id outside the card table")
            ab = arrays["tok_ability"]
            if ((ab < -1) | (ab >= b["abilities"])).any():
                raise Refused("request", "an ability row outside the ability table")

    # ── compute thread ────────────────────────────────────────────────────────────────────

    def _take_batch(self):
        """Blocks until a batch is due; the requests (arrival order, fitting max_batch) or None at stop."""
        with self._cv:
            while not self._stop.is_set():
                live = [r for r in self._queue if r.client.alive]
                if len(live) != len(self._queue):
                    self._queue = collections.deque(live)
                if self._queue:
                    queued = sum(r.d for r in self._queue)
                    connected = sum(1 for c in self._clients.values() if c.alive and c.hello is not None)
                    waiting = sum(1 for c in self._clients.values() if c.alive and c.queued)
                    wait = self._queue[0].arrived + self.deadline - time.perf_counter()
                    if queued >= self.max_batch or waiting >= connected or wait <= 0:
                        batch = [self._queue.popleft()]
                        total = batch[0].d
                        while self._queue and total + self._queue[0].d <= self.max_batch and \
                                (self.pause_after is None or len(batch) + self._stats["requests"] < self.pause_after):
                            r = self._queue.popleft()
                            batch.append(r)
                            total += r.d
                        return batch
                    self._cv.wait(timeout=max(wait, 1e-4))
                else:
                    self._cv.wait(timeout=0.5)
        return None

    def _compute_loop(self):
        while True:
            if self.pause_after is not None and self._stats["requests"] >= self.pause_after:
                self._log("paused after %d requests" % self._stats["requests"])
                self._stop.wait()
                return
            batch = self._take_batch()
            if batch is None:
                return
            start = time.perf_counter()
            pos = 0
            for r in batch:
                r.first = pos
                pos += r.d
            try:
                results = self._evaluate(batch)
                err = None
            except Exception as e:  # noqa: BLE001 — every request of the batch gets `eval`
                results, err = None, "%s: %s" % (type(e).__name__, e)
            compute = time.perf_counter() - start
            us = min(int(compute * 1e6), 0xFFFFFFFF)
            d_total = sum(r.d for r in batch)
            for r in batch:
                c = r.client
                if not c.alive:
                    continue
                if err is not None:
                    with self._slock:
                        self._stats["errors"] += 1
                    c.queued = False
                    self._refuse(c, Refused("eval", err))
                    continue
                try:
                    length = self._write_response(r, results, d_total, us)
                except Refused as e:
                    c.queued = False
                    self._refuse(c, e)
                    continue
                c.queued = False
                c.send(DONE, struct.pack("<QQ", r.seq, length))
            end = time.time()
            with self._slock:
                self._stats["batches"] += 1
                self._stats["decisions"] += d_total
                self._stats["requests"] += len(batch)
                self._stats["max_batch_seen"] = max(self._stats["max_batch_seen"], d_total)
                self._recent.append((end, d_total, start - batch[0].arrived, compute))
            with self._cv:
                self._cv.notify_all()

    def _evaluate(self, batch):
        """Per policy id: (version, values [D], per batch decision its logits and log π over its
        real menu rows), for the batch's decisions that asked for it. The decisions run sorted by
        token count in chunks of at most `chunk` (less padding per forward, bounded memory); every
        chunk is queued on the device before the first result is read back."""
        with self._plock:
            pols = dict(self._policies)
        decs = _concat([r.decs for r in batch])
        dims = batch[0].client.hello["dims"]
        dev = self.device
        total = len(decs)
        wants = {}
        pos = 0
        for r in batch:
            for pid in r.policies:
                wants.setdefault(pid, np.zeros(total, np.bool_))[pos:pos + r.d] = True
            pos += r.d
        for pid in wants:
            if pid not in pols:
                raise RuntimeError("policy %d is no longer served" % pid)
        order = np.argsort(decs.sizes["n"], kind="stable")
        pending = []
        with torch.inference_mode():
            for start in range(0, total, self.chunk):
                ch = order[start:start + self.chunk]
                x = pad(decs.select(ch), dims)
                inputs = [torch.from_numpy(x[k]).to(dev, non_blocking=True) for k in INPUTS]
                for pid, want in wants.items():
                    sel = want[ch]
                    if sel.all():
                        sub, where = inputs, ch
                    else:
                        if not sel.any():
                            continue
                        rows = torch.from_numpy(np.nonzero(sel)[0]).to(dev)
                        sub, where = [t.index_select(0, rows) for t in inputs], ch[sel]
                    logits, value = self._forward(pols[pid].module, sub)
                    act_mask = sub[9]
                    lp = torch.log_softmax(logits.masked_fill(~act_mask, -float("inf")), -1)
                    pending.append((pid, where, logits, lp, value))
            menu = decs.sizes["a"]
            out = {pid: (pols[pid].version, np.zeros(total, np.float32), [None] * total, [None] * total)
                   for pid in wants}
            for pid, where, logits, lp, value in pending:
                lg, lpn, vl = logits.cpu().numpy(), lp.cpu().numpy(), value.float().cpu().numpy()
                _, vals, lgs, lps = out[pid]
                for j, b in enumerate(where):
                    a = int(menu[b])
                    vals[b] = vl[j]
                    lgs[b] = lg[j, :a]
                    lps[b] = lpn[j, :a]
        return out

    def _forward(self, module, inputs):
        if self.precision == "fp32":
            logits, value = module(*inputs)
        else:
            dt = torch.float16 if self.precision == "fp16" else torch.bfloat16
            with torch.autocast(device_type=self.device.type, dtype=dt):
                logits, value = module(*inputs)
        return logits.float(), value.float()

    def _write_response(self, r, results, d_total, us):
        c = r.client
        h = c.hello
        buf = c.shm.buf
        base = h["request_bytes"]
        lo, hi = r.first, r.first + r.d   # the request's decisions in the batch
        p = len(r.policies)
        a_rows = int(r.decs.sizes["a"].sum())
        need = 56 + 16 * p
        for _ in r.policies:
            need = _align8(need) + 4 * r.d
            need = _align8(need) + 4 * a_rows
            need = _align8(need) + 4 * a_rows
        if need > h["response_bytes"]:
            raise Refused("request", "the response (%d bytes) does not fit the response area" % need)
        off = 56 + 16 * p
        for i, pid in enumerate(r.policies):
            struct.pack_into("<IIQ", buf, base + 56 + 16 * i, pid, 0, results[pid][0])
        for pid in r.policies:
            _, vals, lgs, lps = results[pid]
            parts = (vals[lo:hi], np.concatenate(lgs[lo:hi]), np.concatenate(lps[lo:hi]))
            for arr in parts:
                off = _align8(off)
                np.frombuffer(buf, np.float32, len(arr), base + off)[:] = arr
                off += 4 * len(arr)
        RESP.pack_into(buf, base, b"RMRS", PROTOCOL, r.spec_hash, r.seq, r.d, p, d_total, us, off, 0)
        return off


def _concat(decs_list):
    """One Decisions of several (their decisions in order)."""
    if len(decs_list) == 1:
        return decs_list[0]
    arrays = {name: np.concatenate([d.arrays[name] for d in decs_list]) for name, _ in TENSORS}
    sizes = {k: np.concatenate([d.sizes[k] for d in decs_list]) for k in ("n", "r", "a")}
    return Decisions(arrays, sizes)


# ── standalone ────────────────────────────────────────────────────────────────────────────────

def _load_policy_arg(text, tables_dir, config):
    """A --policy value: a .ts.pt export, `ckpt:PATH` (an RMNet checkpoint) or `init:SEED` (a fresh
    RMNet of --config)."""
    if text.startswith("ckpt:") or text.startswith("init:"):
        from .model import ModelConfig, load_checkpoint
        from .spec import load_tables
        if not tables_dir:
            raise SystemExit("rmlearn.server: %s needs --tables" % text)
        tables = load_tables(tables_dir)
        if text.startswith("ckpt:"):
            return load_checkpoint(text[5:], tables)
        torch.manual_seed(int(text[5:]))
        cfg = ModelConfig.from_dict(json.loads(config)) if config else ModelConfig()
        return RMNet(cfg, tables)
    return text


def main(argv=None):
    import argparse
    import signal
    ap = argparse.ArgumentParser(description="The learner's RMIP v1 inference server, standalone (tests, benches).")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--policy", action="append", required=True, help="ID=MODEL: a .ts.pt, ckpt:PATH or init:SEED")
    ap.add_argument("--version", action="append", default=[], help="ID=N")
    ap.add_argument("--tables", help="rmfeat --export-tables DIR (for ckpt: and init:)")
    ap.add_argument("--config", help="ModelConfig JSON (for init:)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-batch", type=int, default=512)
    ap.add_argument("--deadline-us", type=int, default=2000)
    ap.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--pause-after", type=int, help="tests: stop answering after this many requests")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    versions = {}
    for v in args.version:
        pid, n = v.split("=", 1)
        versions[int(pid)] = int(n)
    srv = InferenceServer(args.socket, device=args.device, max_batch=args.max_batch, deadline_us=args.deadline_us,
                          precision=args.precision, pause_after=args.pause_after)
    for spec in args.policy:
        pid, what = spec.split("=", 1)
        srv.set_policy(int(pid), _load_policy_arg(what, args.tables, args.config), versions.get(int(pid), 0))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stop.set())
    srv.start()
    while not stop.wait(0.2):
        pass
    st = srv.stats()
    srv.stop()
    print("RMLEARN-SERVE-STATS: batches=%d decisions=%d requests=%d refusals=%d errors=%d mean_batch=%.2f"
          % (st["batches"], st["decisions"], st["requests"], st["refusals"], st["errors"],
             st["decisions"] / max(st["batches"], 1)), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
