"""The control socket (docs/rmtrain.md §2): one JSON request per connection, one JSON reply.

    srv = ControlServer(path, handler)       # handler(request dict) -> reply dict (in a thread)
    srv.start(); ...; srv.stop()
    reply = request(path, {"cmd": "status"}, timeout=10)   # ConnectionError when nothing answers

A request is ``{"cmd": NAME, ...}`` on one line; the reply is one JSON line, ``{"ok": true, ...}``
or ``{"ok": false, "error": TEXT}``. The handler runs on the server's thread, one connection at a
time per thread (each connection gets its own short-lived thread).
"""

import json
import os
import socket
import threading

MAX_REQUEST = 1 << 20


class ControlServer:
    def __init__(self, path, handler, log=None):
        self.path = path
        self.handler = handler
        self.log = log or (lambda m: None)
        self._sock = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        os.makedirs(os.path.dirname(self.path), mode=0o700, exist_ok=True)
        if os.path.exists(self.path):
            os.unlink(self.path)
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        tmp = "%s.tmp%d" % (self.path, os.getpid())
        if os.path.exists(tmp):
            os.unlink(tmp)
        s.bind(tmp)
        os.chmod(tmp, 0o600)
        s.listen(16)
        os.rename(tmp, self.path)
        s.settimeout(0.5)
        self._sock = s
        self._thread = threading.Thread(target=self._loop, name="control", daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            conn.settimeout(10)
            data = b""
            while not data.endswith(b"\n") and len(data) < MAX_REQUEST:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            try:
                req = json.loads(data.decode() or "{}")
                if not isinstance(req, dict):
                    raise ValueError("a request is a JSON object")
                reply = self.handler(req)
            except Exception as err:  # every failure is the client's answer
                reply = {"ok": False, "error": "%s: %s" % (type(err).__name__, err)}
            conn.settimeout(30)
            conn.sendall((json.dumps(reply, default=str) + "\n").encode())
        except OSError:
            pass
        finally:
            conn.close()

    def stop(self):
        self._stop.set()
        if self._sock is not None:
            self._sock.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


def request(path, req, timeout=30.0):
    """Sends `req` to the control socket at `path`; its reply (ConnectionError when none answers)."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        s.sendall((json.dumps(req) + "\n").encode())
        data = b""
        while not data.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    except (FileNotFoundError, ConnectionRefusedError, socket.timeout, OSError) as err:
        raise ConnectionError(str(err)) from err
    finally:
        s.close()
    if not data:
        raise ConnectionError("no reply")
    return json.loads(data.decode())


def alive(path, timeout=3.0):
    try:
        return bool(request(path, {"cmd": "ping"}, timeout).get("ok"))
    except (ConnectionError, ValueError):
        return False
