#!/usr/bin/env python
"""Central AZNet inference server (Stage C of docs/gpu_selfplay_inference_plan.md).

One process owns the GPU (or a CPU copy of the net) and serves batched leaf
evaluations to the whole az_actor fleet over a Unix domain socket, so the
effective forward batch is fleet-wide (workers x worlds rows in flight) instead
of per-process K = worlds. The server returns RAW logits + value; every client
keeps its exact local double-precision prior computation, so AZEvalResultD
numerics are identical whether the forward ran locally or remotely.

Wire protocol (little-endian, length-prefixed frames; the C++ twin lives in
src/actor/az_evaluator.cpp — change them in lockstep):

  hello (once per connection, client -> server):
      uint32 len | int32 MAGIC (0x415A4556) | int32 OBS_SIZE | int32 MAX_ACTIONS
  hello reply: uint32 len | int32 1  (mismatch: server replies int32 0, then
      closes — the client fatals loudly)
  request:  uint32 len | int32 k | k*OBS_SIZE float32 obs rows
                        | k int32 num_choices
  reply:    uint32 len | k*MAX_ACTIONS float32 raw masked logits
                        | k float32 tanh values

Batching: requests from ALL connections accumulate; once one is pending the
server waits up to --timeout-ms for more (or --max-batch rows), runs ONE
forward over the concatenated rows, and slices the replies back per request.
A request's rows never split across two forwards (k <= worlds is tiny next to
--max-batch).

Lifecycle: parent (az_selfplay._generate_actor) starts this before the actor
fleet, waits for the READY line, and terminates it when generation ends. Any
error is fatal and loud — clients see a closed socket and exit(1) themselves.
"""

import argparse
import os
import selectors
import socket
import struct
import sys
import time

MAGIC = 0x415A4556  # "AZEV"


def log(msg: str) -> None:
    print(f"[az-eval-server] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Central AZNet inference server")
    ap.add_argument("--model", required=True, help="TorchScript net (.ts.pt)")
    ap.add_argument("--socket", required=True, help="Unix socket path to bind")
    ap.add_argument("--device", default="cuda",
                    help="Torch device for the forward (default cuda = the "
                         "Radeon under the ROCm build)")
    ap.add_argument("--max-batch", type=int, default=256,
                    help="Flush a forward at this many accumulated rows")
    ap.add_argument("--timeout-ms", type=float, default=1.0,
                    help="Micro-timeout: with >=1 row pending, wait this long "
                         "for more rows before forwarding")
    args = ap.parse_args()

    import torch  # noqa: PLC0415 — after argparse so --help stays instant

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from env import OBS_SIZE, MAX_ACTIONS

    torch.set_num_threads(1)
    if args.device != "cpu":
        # Same landmine as the C++ actor (see AZEvaluator::load): the NNC
        # fuser's IRSimplifier grinds for minutes recompiling this module for
        # the gpu device. GEMM-dominated net — skip fusion.
        torch._C._jit_set_texpr_fuser_enabled(False)

    net = torch.jit.load(args.model)
    net.eval()
    net = net.to(args.device)
    # Warm one max-batch forward now so lazy device init (kernel code objects,
    # BLAS workspaces) is paid before the first client, not during it.
    with torch.no_grad():
        w_obs = torch.zeros(args.max_batch, OBS_SIZE, device=args.device)
        w_mask = torch.ones(args.max_batch, MAX_ACTIONS, dtype=torch.bool,
                            device=args.device)
        net(w_obs, w_mask)
        if args.device != "cpu":
            torch.cuda.synchronize()
    log(f"model={args.model} device={args.device} OBS_SIZE={OBS_SIZE} "
        f"MAX_ACTIONS={MAX_ACTIONS} max_batch={args.max_batch} "
        f"timeout_ms={args.timeout_ms}")

    if os.path.exists(args.socket):
        os.unlink(args.socket)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(args.socket)
    srv.listen(128)
    srv.setblocking(False)
    sel = selectors.DefaultSelector()
    sel.register(srv, selectors.EVENT_READ, None)
    print(f"READY {args.socket}", flush=True)

    ROW_REQ = 4 + 4 * OBS_SIZE + 4          # per-row request bytes (+k header)
    conns = {}    # sock -> bytearray recv buffer
    hello_ok = set()
    # pending: list of (sock, k, obs_bytes, nc_list) not yet forwarded
    pending = []
    pending_rows = 0

    def close_conn(s):
        nonlocal pending, pending_rows
        try:
            sel.unregister(s)
        except Exception:
            pass
        conns.pop(s, None)
        hello_ok.discard(s)
        kept = [p for p in pending if p[0] is not s]
        pending_rows -= sum(p[1] for p in pending) - sum(p[1] for p in kept)
        pending = kept
        s.close()

    def send_all(s, payload: bytes):
        try:
            s.sendall(struct.pack("<I", len(payload)) + payload)
        except OSError:
            close_conn(s)

    def parse_frames(s):
        """Consume complete frames from s's buffer into pending/hello."""
        nonlocal pending_rows
        buf = conns[s]
        while True:
            if len(buf) < 4:
                return
            (flen,) = struct.unpack_from("<I", buf, 0)
            if len(buf) < 4 + flen:
                return
            payload = bytes(buf[4:4 + flen])
            del buf[:4 + flen]
            if s not in hello_ok:
                if flen != 12:
                    log(f"bad hello length {flen}; closing client")
                    close_conn(s)
                    return
                magic, obs_sz, max_act = struct.unpack("<iii", payload)
                if (magic, obs_sz, max_act) != (MAGIC, OBS_SIZE, MAX_ACTIONS):
                    log(f"hello mismatch: magic={magic:#x} OBS_SIZE={obs_sz} "
                        f"MAX_ACTIONS={max_act} (server: {OBS_SIZE}/{MAX_ACTIONS})")
                    send_all(s, struct.pack("<i", 0))
                    close_conn(s)
                    return
                hello_ok.add(s)
                send_all(s, struct.pack("<i", 1))
                continue
            (k,) = struct.unpack_from("<i", payload, 0)
            if k < 1 or flen != 4 + k * (4 * OBS_SIZE + 4):
                log(f"bad request: k={k} len={flen}; closing client")
                close_conn(s)
                return
            obs_bytes = payload[4:4 + 4 * k * OBS_SIZE]
            nc = list(struct.unpack_from(f"<{k}i", payload, 4 + 4 * k * OBS_SIZE))
            if any(c < 1 or c > MAX_ACTIONS for c in nc):
                log(f"bad num_choices {nc}; closing client")
                close_conn(s)
                return
            pending.append((s, k, obs_bytes, nc))
            pending_rows += k

    def flush():
        """One forward over everything pending; reply per request."""
        nonlocal pending, pending_rows
        batch, rows = [], 0
        while pending and rows + pending[0][1] <= max(args.max_batch, pending[0][1]):
            p = pending.pop(0)
            batch.append(p)
            rows += p[1]
            if rows >= args.max_batch:
                break
        pending_rows -= rows
        obs = torch.frombuffer(bytearray(b"".join(p[2] for p in batch)),
                               dtype=torch.float32).reshape(rows, OBS_SIZE)
        mask = torch.zeros(rows, MAX_ACTIONS, dtype=torch.bool)
        r = 0
        for _, k, _, nc in batch:
            for i, c in enumerate(nc):
                mask[r + i, :c] = True
            r += k
        with torch.no_grad():
            if args.device != "cpu":
                obs = obs.pin_memory().to(args.device, non_blocking=True)
                mask = mask.pin_memory().to(args.device, non_blocking=True)
            logits, value = net(obs, mask)
            logits = logits.to("cpu").contiguous()
            value = value.to("cpu").contiguous()
        lb = logits.numpy().tobytes()
        vb = value.numpy().tobytes()
        row_l = 4 * MAX_ACTIONS
        r = 0
        for s, k, _, _ in batch:
            payload = (lb[r * row_l:(r + k) * row_l]
                       + vb[4 * r:4 * (r + k)])
            send_all(s, payload)
            r += k

    served = 0
    while True:
        # With work pending, poll briefly (micro-timeout batching window);
        # otherwise block until a socket is readable.
        timeout = (args.timeout_ms / 1000.0) if pending else None
        t0 = time.monotonic()
        events = sel.select(timeout)
        for key, _ in events:
            if key.data is None:
                try:
                    c, _ = srv.accept()
                except OSError:
                    continue
                c.setblocking(True)
                sel.register(c, selectors.EVENT_READ, "conn")
                conns[c] = bytearray()
            else:
                s = key.fileobj
                try:
                    data = s.recv(1 << 20)
                except OSError:
                    close_conn(s)
                    continue
                if not data:
                    close_conn(s)
                    continue
                conns[s].extend(data)
                parse_frames(s)
        if pending and (pending_rows >= args.max_batch or not events
                        or (time.monotonic() - t0) * 1000.0 >= args.timeout_ms):
            n = len(pending)
            flush()
            served += n
            if served % 5000 < n:
                log(f"served {served} requests")


if __name__ == "__main__":
    main()
