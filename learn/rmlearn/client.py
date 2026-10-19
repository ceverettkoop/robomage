"""A Python RMIP v1 client (docs/inference_protocol.md): the protocol's other side, for checks and
benches (the actors' client is the C++ ``nn:remote:``).

    cl = Client(socket_path, hashes, dims, policies=[0, 1])   # creates the region, HELLO / WELCOME
    res = cl.evaluate(decisions)    # Decisions -> {"versions": [...], "batch": n, "value": [P][D],
                                    #   "logits": [P][ΣA], "log_pi": [P][ΣA]} (numpy)
    cl.close()

A refused handshake or request raises ``RmipError`` with the server's ``code``.
"""

import json
import os
import socket
import struct
from multiprocessing import shared_memory

import numpy as np

from .server import ARRAYS, BYE, DONE, ERROR, EVAL, HELLO, MSG, REQ, RESP, WELCOME, _align8


class RmipError(Exception):
    def __init__(self, code, message):
        super().__init__("%s: %s" % (code, message))
        self.code = code


def _recv_exact(sock, n):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RmipError("closed", "the server closed the connection")
        data += chunk
    return data


def recv_message(sock):
    magic, kind, _, length = MSG.unpack(_recv_exact(sock, MSG.size))
    if magic != b"RMIP":
        raise RmipError("protocol", "not an RMIP message")
    return kind, _recv_exact(sock, length)


def send_message(sock, kind, body=b""):
    sock.sendall(MSG.pack(b"RMIP", kind, 0, len(body)) + body)


def hello_only(socket_path, hello, timeout=30):
    """Sends `hello` (a dict) and returns ("WELCOME", body) or ("ERROR", code)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect(socket_path)
    try:
        send_message(s, HELLO, json.dumps(hello).encode())
        kind, body = recv_message(s)
        if kind == ERROR:
            return "ERROR", json.loads(body)["code"]
        return "WELCOME", json.loads(body)
    finally:
        s.close()


class Client:
    def __init__(self, socket_path, hashes, dims, policies=(0,), request_bytes=32 << 20, response_bytes=4 << 20,
                 name="rmlearn.client", timeout=60):
        self.dims = dict(dims)
        self.policies = list(policies)
        self.spec_hash = hashes["spec_hash"]
        self.request_bytes, self.response_bytes = request_bytes, response_bytes
        shm_name = "rmip-py-%d-%d" % (os.getpid(), id(self) & 0xFFFFFF)
        self.shm = shared_memory.SharedMemory(shm_name, create=True, size=request_bytes + response_bytes,
                                              track=False)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.seq = 0
        try:
            self.sock.connect(socket_path)
            hello = {"protocol": 1, "client": name, "spec_hash": hashes["spec_hash"],
                     "vocab_hash": hashes["vocab_hash"], "text_model": hashes["text_model"], "dims": self.dims,
                     "shm": "/" + shm_name, "request_bytes": request_bytes, "response_bytes": response_bytes,
                     "policies": self.policies}
            send_message(self.sock, HELLO, json.dumps(hello).encode())
            kind, body = recv_message(self.sock)
        finally:
            self.shm.unlink()
        if kind == ERROR:
            e = json.loads(body)
            self.close()
            raise RmipError(e["code"], e["message"])
        if kind != WELCOME:
            raise RmipError("protocol", "expected WELCOME, got %d" % kind)
        self.welcome = json.loads(body)

    def write_request(self, decs, tamper=None):
        """Lays `decs` out in the request area; its length. `tamper(name, array)` may alter a copy."""
        buf = self.shm.buf
        d, p = len(decs), len(self.policies)
        off = 64
        np.frombuffer(buf, np.uint32, p, off)[:] = self.policies
        off = _align8(off + 4 * p)
        for key in ("n", "r", "a"):
            np.frombuffer(buf, np.int64, d, off)[:] = decs.sizes[key]
            off = _align8(off + 8 * d)
        width = {"F": self.dims["F"], "C": self.dims["C"], "K": self.dims["K"], "Q": self.dims["Q"], 3: 3}
        for name, dtype, key, w in ARRAYS:
            arr = np.ascontiguousarray(decs.arrays[name], dtype=dtype)
            if tamper is not None:
                arr = tamper(name, arr.copy())
            raw = arr.reshape(-1)
            off = _align8(off)
            np.frombuffer(buf, dtype, len(raw), off)[:] = raw
            off += raw.nbytes
            del w
        length = off
        F, C, K, Q = width["F"], width["C"], width["K"], width["Q"]
        REQ.pack_into(buf, 0, b"RMRQ", 1, int(self.spec_hash, 16), self.seq, d, p, F, C, K, Q, length, 0)
        return length

    def evaluate(self, decs, tamper=None):
        self.seq += 1
        length = self.write_request(decs, tamper)
        send_message(self.sock, EVAL, struct.pack("<QQ", self.seq, length))
        kind, body = recv_message(self.sock)
        if kind == ERROR:
            e = json.loads(body)
            raise RmipError(e["code"], e["message"])
        if kind != DONE:
            raise RmipError("protocol", "expected DONE, got %d" % kind)
        seq, rlen = struct.unpack("<QQ", body)
        base = self.request_bytes
        buf = self.shm.buf
        magic, proto, spec_hash, rseq, d, p, batch, us, length, _ = RESP.unpack_from(buf, base)
        if magic != b"RMRS" or rseq != self.seq or seq != self.seq or d != len(decs) or p != len(self.policies):
            raise RmipError("response", "the response is not this request's")
        versions, ids = [], []
        for i in range(p):
            pid, _, ver = struct.unpack_from("<IIQ", buf, base + 56 + 16 * i)
            ids.append(pid)
            versions.append(ver)
        rows = int(decs.sizes["a"].sum())
        off = 56 + 16 * p
        out = {"ids": ids, "versions": versions, "batch": batch, "compute_us": us, "value": [], "logits": [],
               "log_pi": []}
        for _ in range(p):
            for key, count in (("value", d), ("logits", rows), ("log_pi", rows)):
                off = _align8(off)
                out[key].append(np.frombuffer(buf, np.float32, count, base + off).copy())
                off += 4 * count
        if off != length or rlen != length:
            raise RmipError("response", "the response's length is not its arrays' end")
        return out

    def close(self):
        try:
            send_message(self.sock, BYE)
        except OSError:
            pass
        self.sock.close()
        try:
            self.shm.close()
        except BufferError:
            pass
