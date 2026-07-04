#!/usr/bin/env python3
"""League dashboard: HTTP monitor + stop/resume/restart control for
distributed league training (docs/distributed_league_training.md).

Stdlib-only. Run on EITHER machine; it reads only the local (Syncthing-synced)
checkpoint dir, so it sees every shard's progress sidecar and worker heartbeat
regardless of which host wrote them, and it controls remote workers by writing
per-shard `_control{tag}.json` files that Syncthing replicates — no listener or
open port on the workers.

  scripts/league_dashboard.py --host 100.x.y.z        # bind the Tailscale IP
  open http://100.x.y.z:8787

Endpoints:
  GET  /            self-contained HTML dashboard (auto-refreshing)
  GET  /api/status  JSON: per-shard progress + worker liveness + snapshot times
  POST /api/control JSON body {"shard": "i/n"|null, "command": "stop|resume|restart"}
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "train"))
from cli_spec import parse_shard, shard_tag  # noqa: E402

DEFAULT_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "train", "checkpoints")
# A worker heartbeats every ~30s; allow ~3 heartbeat+sync cycles (plus clock
# skew) before rendering it as down/stale.
STALE_AFTER = 180.0

_PROGRESS_RE = re.compile(r"^_league_progress(?P<tag>\.shard\d+of\d+)?\.json$")
_TAG_RE = re.compile(r"^\.shard(?P<i>\d+)of(?P<n>\d+)$")


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _shard_from_tag(tag: str | None) -> str | None:
    if not tag:
        return None
    m = _TAG_RE.match(tag)
    return f"{m.group('i')}/{m.group('n')}" if m else None


def _deck_snapshot_mtimes(checkpoint_dir: str) -> dict:
    """Newest snapshot mtime per deck (searching one subdir level, matching the
    league/<stem> checkpoint namespacing)."""
    newest: dict[str, float] = {}
    for pattern in ("*__*.zip", "*/*__*.zip"):
        for path in glob.glob(os.path.join(checkpoint_dir, pattern)):
            rel = os.path.relpath(path, checkpoint_dir)
            deck = rel.rsplit("__", 1)[0]
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime > newest.get(deck, 0.0):
                newest[deck] = mtime
    return newest


def collect_status(checkpoint_dir: str) -> dict:
    """Aggregate every shard's sidecar + worker heartbeat from the synced dir."""
    now = time.time()
    shards = {}
    for path in glob.glob(os.path.join(checkpoint_dir, "_league_progress*.json")):
        m = _PROGRESS_RE.match(os.path.basename(path))
        if not m:
            continue
        tag = m.group("tag") or ""
        prog = _read_json(path)
        if not prog:
            continue
        roster = prog.get("roster") or []
        shard = prog.get("shard") or _shard_from_tag(tag)
        split = None
        try:
            split = parse_shard(shard)
        except ValueError:
            pass
        train_decks = roster[split[0]::split[1]] if split else roster
        rotation = int(prog.get("rotation", 0))
        learner = train_decks[rotation % len(train_decks)] if train_decks else None
        steps_done = int(prog.get("steps_done", 0))
        chunk_start = int(prog.get("chunk_start", steps_done))
        shards[shard or "single"] = {
            "shard": shard,
            "roster": roster,
            "train_decks": train_decks,
            "learner": learner,
            "rotation": rotation,
            "steps_done": steps_done,
            "total_timesteps": int(prog.get("total_timesteps", 0)),
            "rotation_done": steps_done - chunk_start,
            "rotation_target": prog.get("rotation_target"),
            "deck_stats": prog.get("deck_stats") or {},
            "worker": None,
            "control": None,
        }
    for path in glob.glob(os.path.join(checkpoint_dir, "_worker_status*.json")):
        status = _read_json(path)
        if not status:
            continue
        key = status.get("shard") or "single"
        entry = shards.setdefault(key, {"shard": status.get("shard"), "worker": None,
                                        "control": None})
        age = now - float(status.get("written_at", 0))
        status["heartbeat_age"] = round(age, 1)
        status["stale"] = age > STALE_AFTER
        entry["worker"] = status
    for path in glob.glob(os.path.join(checkpoint_dir, "_control*.json")):
        ctl = _read_json(path)
        if not ctl:
            continue
        tag = os.path.basename(path)[len("_control"):-len(".json")]
        key = _shard_from_tag(tag) or "single"
        if key in shards:
            shards[key]["control"] = ctl
    return {
        "now": now,
        "checkpoint_dir": checkpoint_dir,
        "shards": shards,
        "snapshots": _deck_snapshot_mtimes(checkpoint_dir),
    }


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>RoboMage League</title>
<style>
 body{font-family:system-ui,sans-serif;margin:2em;background:#111;color:#eee}
 h1{font-size:1.3em} .cards{display:flex;flex-wrap:wrap;gap:1em}
 .card{background:#1c1c22;border:1px solid #333;border-radius:8px;padding:1em;
       min-width:22em;flex:1}
 .dot{display:inline-block;width:.7em;height:.7em;border-radius:50%;margin-right:.4em}
 .running{background:#2ecc71}.stopped{background:#e67e22}.crashed{background:#e74c3c}
 .stale{background:#777}.done{background:#3498db}
 .bar{background:#333;border-radius:4px;height:.6em;margin:.4em 0}
 .bar>div{background:#2e86de;height:100%;border-radius:4px}
 table{border-collapse:collapse;font-size:.85em;margin-top:.5em;width:100%}
 td,th{padding:.15em .5em;text-align:left;border-bottom:1px solid #2a2a30}
 button{margin-right:.4em;padding:.3em .8em;border-radius:5px;border:1px solid #555;
        background:#26262e;color:#eee;cursor:pointer}
 button:hover{background:#333340} .muted{color:#888;font-size:.8em}
 .pending{color:#f1c40f;font-size:.8em}
</style></head><body>
<h1>RoboMage league training</h1>
<div class="cards" id="cards">loading…</div>
<script>
async function ctl(shard, command){
  await fetch('/api/control',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({shard:shard==='single'?null:shard, command})});
  load();
}
function fmt(n){return n==null?'?':n.toLocaleString()}
function esc(s){const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML}
async function load(){
  const r = await fetch('/api/status'); const s = await r.json();
  const cards = [];
  for (const [key, sh] of Object.entries(s.shards)){
    const w = sh.worker || {};
    let cls = w.state || 'stale'; if (w.stale) cls = 'stale';
    let liveness = w.stale ? `stale (${Math.round(w.heartbeat_age)}s)` : (w.state || 'no worker');
    const pct = sh.total_timesteps ? (100*sh.steps_done/sh.total_timesteps).toFixed(1) : 0;
    const rotPct = sh.rotation_target ? Math.min(100, 100*sh.rotation_done/sh.rotation_target).toFixed(0) : 0;
    let decks = '';
    for (const d of (sh.train_decks||[])){
      const st = (sh.deck_stats||{})[d] || {};
      const wr = st.winrate==null ? '—' : (100*st.winrate).toFixed(1)+'%';
      const snap = s.snapshots[d] ? new Date(1000*s.snapshots[d]).toLocaleTimeString() : '—';
      decks += `<tr><td>${esc(d)}${d===sh.learner?' ⬅':''}</td><td>${wr}</td>`+
               `<td>${fmt(st.steps)}</td><td>${snap}</td></tr>`;
    }
    const pending = sh.control && (!w.last_command_ack || w.last_command_ack !== sh.control.issued_at)
      ? `<span class="pending">pending: ${esc(sh.control.command)}</span>` : '';
    cards.push(`<div class="card">
      <b><span class="dot ${cls}"></span>${esc(key)}</b>
      <span class="muted">${esc(w.hostname||'?')} · ${esc(liveness)} · restarts ${w.restarts??0}</span><br>
      <div class="muted">learner: <b>${esc(sh.learner||'?')}</b> · rotation ${sh.rotation??'?'}
        (${rotPct}% of ${fmt(sh.rotation_target)})</div>
      <div class="bar"><div style="width:${pct}%"></div></div>
      <div class="muted">${fmt(sh.steps_done)} / ${fmt(sh.total_timesteps)} steps (${pct}%)</div>
      <table><tr><th>deck</th><th>winrate</th><th>steps</th><th>last snap</th></tr>${decks}</table>
      <div style="margin-top:.6em">
        <button onclick="ctl('${esc(key)}','stop')">Stop</button>
        <button onclick="ctl('${esc(key)}','resume')">Resume</button>
        <button onclick="ctl('${esc(key)}','restart')">Restart</button> ${pending}
      </div>
      <div class="muted" style="margin-top:.5em">${esc(w.last_log_line||'')}</div>
    </div>`);
  }
  document.getElementById('cards').innerHTML =
    cards.join('') || '<i>no league runs found in the checkpoint dir yet</i>';
}
load(); setInterval(load, 5000);
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split("?")[0] == "/api/status":
            body = json.dumps(collect_status(self.checkpoint_dir)).encode()
            self._send(200, body, "application/json")
        elif self.path.split("?")[0] == "/":
            self._send(200, _PAGE.encode(), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.split("?")[0] != "/api/control":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
            command = req.get("command")
            shard = req.get("shard")
            if command not in ("stop", "resume", "restart"):
                raise ValueError(f"bad command {command!r}")
            tag = shard_tag(shard)  # validates the shard spec too
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode(),
                       "application/json")
            return
        path = os.path.join(self.checkpoint_dir, f"_control{tag}.json")
        payload = {"command": command, "issued_at": time.time()}
        _atomic_write_json(path, payload)
        self._send(200, json.dumps({"ok": True, **payload}).encode(),
                   "application/json")

    def log_message(self, fmt, *args):  # quiet the per-request stderr spam
        pass


def _detect_tailscale_ip() -> str | None:
    if shutil.which("tailscale"):
        try:
            out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                                 text=True, timeout=5)
            ip = out.stdout.strip().splitlines()
            if out.returncode == 0 and ip:
                return ip[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=None,
                    help="Bind address (default: this machine's Tailscale IP "
                         "when detectable, else 127.0.0.1 — pass 0.0.0.0 to "
                         "expose on all interfaces)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR,
                    help=f"Synced checkpoint dir (default {DEFAULT_CHECKPOINT_DIR})")
    args = ap.parse_args(argv)
    host = args.host or _detect_tailscale_ip() or "127.0.0.1"
    Handler.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    server = ThreadingHTTPServer((host, args.port), Handler)
    print(f"[dashboard] serving http://{host}:{args.port}  "
          f"(checkpoint dir: {Handler.checkpoint_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
