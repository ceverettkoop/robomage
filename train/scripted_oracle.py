#!/usr/bin/env python3
"""Scripted-agent oracle: a Unix-socket server answering scripted:hard actions.

The C++ actor (bin/az_actor --scripted-seat A|B --scripted-oracle <socket>)
plays vs-scripted az-selfplay cells in-process, but the rule-based agent lives
only in Python (train/scripted_agent.py). Its ``act`` needs nothing but the
machine-mode observation — which the actor reconstructs bit-exactly
(src/actor/obs_builder) — so the actor ships each scripted-seat REAL decision
here and gets the action index back. Search simulations never consult the
oracle (tree play is net-both-seats, mirroring az_selfplay._play_match).

One oracle process serves a whole actor fleet: each actor process opens ONE
connection and each connection gets its OWN ScriptedAgent (the agent carries
per-game state — mulligan counts, fruitless-activation memory), configured by
the connection's HELLO and reset by NEW_GAME frames at exactly the points
az_selfplay's Python backend calls ``agent.new_game()`` (match start + every
GAME_RESULT).

Wire protocol (little-endian, per frame: int32 kind, int32 len, payload[len]):
  kind 1 HELLO    payload = UTF-8 JSON {"deck_a", "deck_b", "spec"};
                  reply kind 1, payload = int32 OBS_SIZE (layout handshake —
                  the client fatals on mismatch instead of feeding a stale
                  layout through the agent's obs offsets).
  kind 2 NEW_GAME no payload, no reply.
  kind 3 QUERY    payload = int32 num_choices + float32[OBS_SIZE] obs;
                  reply kind 3, payload = int32 action.

Torch-free (scripted_agent is pure numpy), so the oracle starts in ~a second
and adds no per-worker memory beyond one thread per connection.

Run: train/.venv/bin/python train/scripted_oracle.py --socket /tmp/oracle.sock
(--ready-fd N writes a newline to fd N once listening, for launchers.)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import struct
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import OBS_SIZE  # noqa: E402
from scripted_agent import make_agent  # noqa: E402

HELLO, NEW_GAME, QUERY = 1, 2, 3
_HDR = struct.Struct("<ii")


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly n bytes; b"" on a clean EOF at a frame boundary."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            if buf:
                raise ConnectionError("oracle: EOF mid-frame")
            return b""
        buf += chunk
    return buf


def _send_frame(conn: socket.socket, kind: int, payload: bytes) -> None:
    conn.sendall(_HDR.pack(kind, len(payload)) + payload)


def _serve_conn(conn: socket.socket) -> None:
    """One actor process's connection: HELLO configures the agent, then a
    stream of NEW_GAME / QUERY frames until EOF."""
    agent = None
    query_len = 4 + 4 * OBS_SIZE
    try:
        while True:
            hdr = _recv_exact(conn, _HDR.size)
            if not hdr:
                return
            kind, plen = _HDR.unpack(hdr)
            payload = _recv_exact(conn, plen) if plen else b""
            if kind == HELLO:
                h = json.loads(payload.decode("utf-8"))
                agent = make_agent(h.get("spec") or "scripted:hard")
                agent.set_deck_names(h.get("deck_a"), h.get("deck_b"))
                agent.new_game()
                _send_frame(conn, HELLO, struct.pack("<i", OBS_SIZE))
            elif kind == NEW_GAME:
                if agent is None:
                    raise ConnectionError("oracle: NEW_GAME before HELLO")
                agent.new_game()
            elif kind == QUERY:
                if agent is None:
                    raise ConnectionError("oracle: QUERY before HELLO")
                if plen != query_len:
                    raise ConnectionError(
                        f"oracle: QUERY payload {plen} bytes, expected "
                        f"{query_len} (obs layout mismatch?)")
                (nc,) = struct.unpack_from("<i", payload, 0)
                obs = np.frombuffer(payload, dtype="<f4", count=OBS_SIZE,
                                    offset=4)
                action = int(agent.act(obs, int(nc))) if nc > 1 else 0
                _send_frame(conn, QUERY, struct.pack("<i", action))
            else:
                raise ConnectionError(f"oracle: unknown frame kind {kind}")
    except (ConnectionError, OSError) as exc:
        print(f"[oracle] connection closed: {exc}", file=sys.stderr, flush=True)
    finally:
        conn.close()


def serve(sock_path: str, ready_fd: int | None = None) -> None:
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(64)
    print(f"[oracle] listening on {sock_path} (OBS_SIZE={OBS_SIZE})", flush=True)
    if ready_fd is not None:
        os.write(ready_fd, b"\n")
        os.close(ready_fd)
    try:
        while True:
            conn, _ = srv.accept()
            threading.Thread(target=_serve_conn, args=(conn,),
                             daemon=True).start()
    finally:
        srv.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--socket", required=True, help="Unix socket path to bind")
    ap.add_argument("--ready-fd", type=int, default=None,
                    help="fd to write a newline to once listening")
    args = ap.parse_args()
    serve(args.socket, args.ready_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
