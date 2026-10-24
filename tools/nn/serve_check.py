#!/usr/bin/env python3
"""A minimal inference server in Python (plan step 5d, docs/inference_protocol.md).

NEEDS torch and numpy (e.g. /root/torchenv/bin/python); `make` never runs it. It proves the protocol
is implementable outside C++ with `socket`, `struct`, `json`, `multiprocessing.shared_memory`, numpy
and torch, and the gate's `actor` tier plays one nn:remote: game against it.

    python serve_check.py --socket PATH --policy ID=MODEL [--policy ID=MODEL ...] [--version ID=N ...]

Loads each policy's TorchScript export (tools/nn/make_reference_policy.py), listens on the Unix
socket PATH (bound under a temporary name and renamed once listening) and serves any number of
clients, each request as it arrives (no batching across clients): the decisions padded as
src/featurize/batch.h pads them (make_reference_policy.pad_batch), one forward per requested policy,
log pi as the log-softmax over each decision's menu rows. Refusals follow the protocol's ERROR codes.
Logs to stderr; SIGTERM or SIGINT stops it (the socket removed; the loop polls a stop flag the
handler sets, so a signal never lands inside a destructor).
"""

import argparse
import json
import os
import select
import signal
import socket
import struct
import sys
import time
from multiprocessing import shared_memory

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_reference_policy import pad_batch  # noqa: E402

MSG = struct.Struct("<4sHHI")
REQ = struct.Struct("<4sIQQII4IQQ")
RESP = struct.Struct("<4sIQQIIIIQQ")
PROTOCOL = 2
HELLO, WELCOME, EVAL, DONE, ERROR, BYE = 1, 2, 3, 4, 5, 6
# The request's tensors in order: name, dtype, the size array counting its rows, its row width.
ARRAYS = [("tok_type", np.int64, "n", None), ("tok_num", np.float32, "n", "F"), ("tok_cat", np.int64, "n", "C"),
          ("tok_card", np.int64, "n", "K"), ("tok_ability", np.int64, "n", "Q"), ("tok_mask", np.uint8, "n", None),
          ("rel", np.int64, "r", 3), ("act_tok", np.int64, "a", None), ("act_mask", np.uint8, "a", None)]


def align8(x):
    return (x + 7) & ~7


def log(msg):
    print("serve_check: " + msg, file=sys.stderr, flush=True)


class Refused(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class Client:
    def __init__(self, sock):
        self.sock, self.buf, self.shm, self.hello = sock, b"", None, None

    def send(self, kind, body=b""):
        self.sock.sendall(MSG.pack(b"RMIP", kind, 0, len(body)) + body)


def read_request(buf, hello):
    magic, proto, spec_hash, seq, d, p, f, c, k, q, length, _ = REQ.unpack_from(buf, 0)
    if magic != b"RMRQ" or proto != PROTOCOL or d < 1 or length > hello["request_bytes"]:
        raise Refused("request", "not a request of protocol %d within the request area" % PROTOCOL)
    if "%016x" % spec_hash != hello["spec_hash"]:
        raise Refused("spec_hash", "the request's spec_hash is not the handshake's")
    width = {"F": f, "C": c, "K": k, "Q": q, 3: 3}
    if {x: width[x] for x in "FCKQ"} != hello["dims"]:
        raise Refused("request", "the request's dims are not the handshake's")
    off = 64
    policies = np.frombuffer(buf, np.uint32, p, off).tolist()
    off = align8(off + 4 * p)
    sizes = {}
    for key in ("n", "r", "a"):
        sizes[key] = np.frombuffer(buf, np.int64, d, off).copy()
        off = align8(off + 8 * d)
    if policies != hello["policies"] or (sizes["n"] < 1).any() or (sizes["a"] < 1).any() or (sizes["r"] < 0).any():
        raise Refused("request", "the request's policies or sizes are not the layout's")
    decisions = [{} for _ in range(d)]
    for name, dtype, key, w in ARRAYS:
        rows, cols = int(sizes[key].sum()), (width[w] if w is not None else 1)
        if off + rows * cols * np.dtype(dtype).itemsize > length:
            raise Refused("request", "the request ends inside " + name)
        flat = np.frombuffer(buf, dtype, rows * cols, off).copy()
        off = align8(off + rows * cols * np.dtype(dtype).itemsize)
        values = flat.reshape(rows, cols) if w is not None else flat
        bounds = np.concatenate([[0], np.cumsum(sizes[key])])
        for i in range(d):
            decisions[i][name] = values[bounds[i]:bounds[i + 1]]
    return spec_hash, seq, decisions


def evaluate(models, versions, client):
    hello, buf = client.hello, client.shm.buf
    spec_hash, seq, decisions = read_request(buf, hello)
    x = [torch.from_numpy(np.ascontiguousarray(v)) for v in pad_batch(decisions, hello["dims"])]
    rows = [len(dec["act_tok"]) for dec in decisions]
    start = time.perf_counter()
    results = []
    with torch.no_grad():
        for pid in hello["policies"]:
            logits, value = (t.numpy() for t in models[pid](*x))
            lg = np.concatenate([logits[i, :a] for i, a in enumerate(rows)]).astype(np.float32)
            lp = []
            for i, a in enumerate(rows):
                row = logits[i, :a].astype(np.float64)
                m = row.max()
                lp.append((row - (m + np.log(np.exp(row - m).sum()))).astype(np.float32))
            results.append((pid, value[:len(rows)].astype(np.float32), lg, np.concatenate(lp)))
    us = min(int((time.perf_counter() - start) * 1e6), 0xFFFFFFFF)
    base = hello["request_bytes"]
    off = 56 + 16 * len(results)
    for i, (pid, *_rest) in enumerate(results):
        struct.pack_into("<IIQ", buf, base + 56 + 16 * i, pid, 0, versions[pid])
    for _pid, *arrays in results:
        for arr in arrays:
            off = align8(off)
            buf[base + off:base + off + 4 * len(arr)] = arr.tobytes()
            off += 4 * len(arr)
    if off > hello["response_bytes"]:
        raise Refused("request", "the response does not fit the response area")
    RESP.pack_into(buf, base, b"RMRS", PROTOCOL, spec_hash, seq, len(rows), len(results), len(rows), us, off, 0)
    client.send(DONE, struct.pack("<QQ", seq, off))


def handshake(models, versions, client, body):
    h = json.loads(body)
    if h.get("protocol") != PROTOCOL:
        raise Refused("protocol", "this server speaks protocol %d" % PROTOCOL)
    for pid in h["policies"]:
        if pid not in models:
            raise Refused("policy", "policy %d is not served" % pid)
        # The export must be of the client's tables exactly (docs/featurizer.md §6.2).
        for key in ("spec_hash", "vocab_hash", "text_model", "layout_hash", "tables_digest"):
            if getattr(models[pid], key) != h[key]:
                raise Refused(key, "the client's %s %s is not policy %d's: rebuild the export" % (key, h[key], pid))
        if json.loads(models[pid].vocab) != h["vocab"]:
            raise Refused("vocab_hash", "the client's vocab manifest is not policy %d's: rebuild the export" % pid)
    client.shm = shared_memory.SharedMemory(h["shm"].lstrip("/"), track=False)
    if client.shm.size < h["request_bytes"] + h["response_bytes"]:
        raise Refused("shm", "the region is smaller than its areas")
    client.hello = h
    pols = [{"id": pid, "version": versions[pid]} for pid in h["policies"]]
    client.send(WELCOME, json.dumps({"protocol": PROTOCOL, "server": "serve_check.py", "max_batch": 1,
                                     "policies": pols}).encode())
    log("%s connected" % h["client"])


def handle(models, versions, client, kind, body):
    if kind == HELLO and client.hello is None:
        handshake(models, versions, client, body)
    elif kind == EVAL and client.hello is not None:
        evaluate(models, versions, client)
    elif kind == BYE:
        return False
    else:
        raise Refused("protocol", "unexpected message type %d" % kind)
    return True


STOP = []


def drop(clients, s):
    c = clients.pop(s)
    s.close()
    if c.shm is not None:
        try:
            c.shm.close()
        except BufferError:
            pass


def serve(path, models, versions):
    tmp = path + ".tmp%d" % os.getpid()
    for p in (path, tmp):
        if os.path.exists(p):
            os.unlink(p)
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(tmp)
    listener.listen(64)
    os.rename(tmp, path)
    log("listening on %s (%d policies)" % (path, len(models)))
    clients = {}
    try:
        while not STOP:
            readable, _, _ = select.select([listener] + list(clients), [], [], 0.2)
            for s in readable:
                if s is listener:
                    conn, _ = listener.accept()
                    clients[conn] = Client(conn)
                    continue
                c = clients[s]
                data = s.recv(1 << 16)
                c.buf += data
                keep = bool(data)
                while keep and len(c.buf) >= MSG.size:
                    magic, kind, _, length = MSG.unpack_from(c.buf, 0)
                    if magic != b"RMIP":
                        keep = False
                        break
                    if len(c.buf) < MSG.size + length:
                        break
                    body, c.buf = c.buf[MSG.size:MSG.size + length], c.buf[MSG.size + length:]
                    try:
                        keep = handle(models, versions, c, kind, body)
                    except Refused as e:
                        c.send(ERROR, json.dumps({"code": e.code, "message": str(e)}).encode())
                        log("refused: %s: %s" % (e.code, e))
                        keep = False
                if not keep:
                    drop(clients, s)
    finally:
        listener.close()
        os.unlink(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--socket", required=True)
    ap.add_argument("--policy", action="append", required=True, help="ID=MODEL (TorchScript)")
    ap.add_argument("--version", action="append", default=[], help="ID=N")
    args = ap.parse_args()
    torch.set_num_threads(1)
    models, versions = {}, {}
    for spec in args.policy:
        pid, path = spec.split("=", 1)
        models[int(pid)] = torch.jit.load(path).eval()
        versions[int(pid)] = int(models[int(pid)].policy_version or 0)
    for spec in args.version:
        pid, n = spec.split("=", 1)
        versions[int(pid)] = int(n)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: STOP.append(signum))
    serve(args.socket, models, versions)
    log("stopping (signal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
