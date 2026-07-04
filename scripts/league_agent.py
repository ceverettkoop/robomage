#!/usr/bin/env python3
"""League agent: one per machine — web UI + trainer supervisor + peer control.

Run this on every machine (each on the tailnet, repo built, Syncthing sharing
train/checkpoints/). Open the web UI on ANY one of them; it auto-discovers the
other agents over Tailscale, lets you configure a league (the same options as
the TUI league form), pick which machines take part, adjust how the roster is
split across them, and Start — the controller fans the launch out to each
machine's agent, which runs its assigned shard supervised (auto-restart on
crash, resume from checkpoint). Monitor and Stop/Resume/Restart from the UI.

  scripts/league_agent.py                 # binds this machine's Tailscale IP
  scripts/league_agent.py --host 100.x.y.z --port 8787 [--token SECRET]
  # then open http://<that-ip>:8787 from any tailnet device

Control plane is direct HTTP between agents over Tailscale (no worker-side
polling); the checkpoint pool is still shared by Syncthing (the data plane).
This agent supersedes scripts/league_worker.py + scripts/league_dashboard.py
for the web-driven flow — see docs/distributed_league_training.md.
"""

import argparse
import glob
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "train"))
from cli_spec import (parse_shard, shard_tag, iter_args, TRAIN_TOOL,  # noqa: E402
                      BINARY)

DEFAULT_CHECKPOINT_DIR = os.path.join(_REPO_ROOT, "train", "checkpoints")
# Per-machine league session output is teed here (one file per session/shard).
DISTRIBUTED_LOG_DIR = os.path.join(_REPO_ROOT, "logs", "distributed")
DEFAULT_VENV_PY = os.path.join(_REPO_ROOT, "train", ".venv", "bin", "python")
TRAIN_PY = os.path.join(_REPO_ROOT, "train", "train.py")
LEAGUE_DECKS_DIR = os.path.join(_REPO_ROOT, "bin", "resources", "decks", "league")

RESTART_BACKOFF = 10.0   # seconds before relaunching a crashed trainer
STOP_GRACE = 60.0        # seconds to wait after SIGINT before SIGKILL
PROBE_TIMEOUT = 2.0      # seconds when probing a peer agent
# Form fields handled by the distribution UI / the agent itself, not shown as
# generic league settings inputs.
_EXCLUDED_FIELDS = {"--resume", "--shard", "--train-decks", "--binary",
                    "--decks", "--tally", "--fresh"}


# ── small json / fs helpers ───────────────────────────────────────────────────
def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _install_stignore(checkpoint_dir):
    dst = os.path.join(checkpoint_dir, ".stignore")
    src = os.path.join(_REPO_ROOT, "scripts", "checkpoints.stignore")
    if not os.path.exists(dst) and os.path.exists(src):
        try:
            shutil.copyfile(src, dst)
        except OSError:
            pass


def _scan_league_decks():
    return sorted("league/" + os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(LEAGUE_DECKS_DIR, "*.dk")))


# ── league form schema (single source of truth: the cli_spec league Sub) ──────
def _league_sub():
    for sub in TRAIN_TOOL.subs:
        if sub.name == "league":
            return sub
    raise RuntimeError("league subcommand not found in cli_spec.TRAIN_TOOL")


def league_form_schema():
    """Serializable list of the league settings fields for the web form —
    every league Arg except the ones the distribution UI/agent supply."""
    fields = []
    for a in iter_args(_league_sub()):
        if a.name in _EXCLUDED_FIELDS:
            continue
        fields.append({
            "name": a.name,
            "kind": a.kind,
            "default": a.default,
            "choices": list(a.choices),
            "help": a.help,
        })
    return fields


def settings_to_flags(settings: dict) -> list[str]:
    """Turn a {flag: value} settings dict from the UI into CLI tokens, honoring
    each league Arg's kind. Unknown/excluded flags are ignored defensively."""
    known = {a.name: a for a in iter_args(_league_sub())}
    out = []
    for name, val in settings.items():
        a = known.get(name)
        if a is None or name in _EXCLUDED_FIELDS:
            continue
        if a.kind == "flag":
            if val:
                out.append(name)
        else:
            if val is None or val == "":
                continue
            out += [name, str(val)]
    return out


# ── Tailscale discovery ───────────────────────────────────────────────────────
def _tailscale_json(*args):
    if not shutil.which("tailscale"):
        return None
    try:
        r = subprocess.run(["tailscale", *args], capture_output=True,
                           text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return json.loads(r.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        pass
    return None


def self_tailscale_ip():
    if not shutil.which("tailscale"):
        return None
    try:
        r = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                           text=True, timeout=5)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def tailnet_peer_ips():
    """[(ip, hostname)] for every online tailnet peer (excludes self)."""
    data = _tailscale_json("status", "--json")
    peers = []
    if data:
        for p in (data.get("Peer") or {}).values():
            ips = p.get("TailscaleIPs") or []
            if not ips:
                continue
            host = (p.get("HostName") or (p.get("DNSName") or "").split(".")[0]
                    or ips[0])
            peers.append((ips[0], host))
    return peers


# ── the local trainer supervisor ──────────────────────────────────────────────
class Trainer:
    """Supervises this machine's `train.py league` process for one session:
    launch from a config, auto-restart on crash with --resume, stop/resume/
    restart on demand. Runs its watch loop on a daemon thread."""

    def __init__(self, checkpoint_dir, venv_py):
        self.dir = checkpoint_dir
        self.venv_py = venv_py
        self.proc = None
        self.state = "idle"       # idle|running|stopped|crashed|done
        self.restarts = 0
        self.last_log_line = ""
        self.config = None        # last launch config (roster, train_decks, shard, settings)
        self._lock = threading.Lock()
        self._want_running = False
        self._crashed_at = None
        threading.Thread(target=self._watch, daemon=True).start()

    # -- command construction --
    def _sidecar(self):
        tag = shard_tag(self.config.get("shard")) if self.config else ""
        return os.path.join(self.dir, f"_league_progress{tag}.json")

    def _cmd(self):
        c = self.config
        binary = c.get("binary") or BINARY
        cmd = [self.venv_py, TRAIN_PY, "league", "--binary", binary,
               "--decks", ",".join(c["roster"])]
        if c.get("shard"):
            cmd += ["--shard", c["shard"]]
        if c.get("train_decks"):
            cmd += ["--train-decks", ",".join(c["train_decks"])]
        cmd += settings_to_flags(c.get("settings") or {})
        if os.path.exists(self._sidecar()):
            cmd += ["--resume"]
        return cmd

    def log_path(self):
        """Per-session, per-machine log file under logs/distributed/. One file
        per Start (session_id); crash-restarts and Resume of the same session
        append, so it's the continuous transcript for this machine's shard."""
        c = self.config or {}
        sid = c.get("session_id") or "session"
        tag = shard_tag(c.get("shard")) or ".single"
        host = socket.gethostname()
        safe = f"{sid}_{host}{tag}".replace("/", "-")
        return os.path.join(DISTRIBUTED_LOG_DIR, safe + ".log")

    def _spawn(self):
        cmd = self._cmd()
        print(f"[agent] launching: {' '.join(cmd)}", flush=True)
        os.makedirs(DISTRIBUTED_LOG_DIR, exist_ok=True)
        logf = open(self.log_path(), "a", buffering=1)  # line-buffered
        logf.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} : "
                   f"{' '.join(cmd)} =====\n")
        self.proc = subprocess.Popen(
            cmd, cwd=_REPO_ROOT, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        self.state = "running"
        threading.Thread(target=self._drain, args=(self.proc, logf),
                         daemon=True).start()

    def _drain(self, proc, logf):
        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                logf.write(line)
                s = line.strip()
                if s:
                    self.last_log_line = s[:300]
        finally:
            logf.close()

    def _kill(self):
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGINT)
            deadline = time.monotonic() + STOP_GRACE
            while self.proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.5)
            if self.proc.poll() is None:
                os.killpg(pgid, signal.SIGKILL)
                self.proc.wait()
        except ProcessLookupError:
            pass

    # -- public control (thread-safe) --
    def launch(self, config):
        with self._lock:
            self._kill()
            self.config = config
            self.restarts = 0
            self._crashed_at = None
            self._want_running = True
            self._spawn()

    def stop(self):
        with self._lock:
            self._want_running = False
            self._crashed_at = None
            self._kill()
            self.state = "stopped"

    def resume(self):
        with self._lock:
            if self.config is None:
                return False
            self._want_running = True
            self._crashed_at = None
            self._spawn()
            return True

    def restart(self):
        with self._lock:
            if self.config is None:
                return False
            self._kill()
            self._want_running = True
            self._crashed_at = None
            self._spawn()
            return True

    def _watch(self):
        while True:
            time.sleep(1.0)
            with self._lock:
                if not self._want_running or self.proc is None:
                    continue
                if self.proc.poll() is None:
                    continue
                rc = self.proc.returncode
                if rc == 0:
                    self.state = "done"
                    self._want_running = False
                    continue
                if self.state != "crashed":
                    print(f"[agent] trainer exited rc={rc}; restarting with "
                          f"--resume in {RESTART_BACKOFF:.0f}s", flush=True)
                    self.state = "crashed"
                    self._crashed_at = time.monotonic()
                elif self._crashed_at and \
                        time.monotonic() - self._crashed_at >= RESTART_BACKOFF:
                    self.restarts += 1
                    self._crashed_at = None
                    self._spawn()

    # -- self status for /api/agent/info --
    def info(self):
        prog = _read_json(self._sidecar()) if self.config else None
        learner = rotation = steps_done = total = None
        train_decks = self.config.get("train_decks") if self.config else None
        if prog:
            roster = prog.get("roster") or []
            td = train_decks or roster
            rotation = int(prog.get("rotation", 0))
            learner = td[rotation % len(td)] if td else None
            steps_done = int(prog.get("steps_done", 0))
            total = int(prog.get("total_timesteps", 0))
        return {
            "hostname": socket.gethostname(),
            "cores": os.cpu_count(),
            "state": self.state,
            "shard": self.config.get("shard") if self.config else None,
            "train_decks": train_decks,
            "session_id": self.config.get("session_id") if self.config else None,
            "learner": learner,
            "rotation": rotation,
            "steps_done": steps_done,
            "total_timesteps": total,
            "restarts": self.restarts,
            "last_log_line": self.last_log_line,
            "log": self.log_path() if self.config else None,
        }


# ── web UI ────────────────────────────────────────────────────────────────────
_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>RoboMage League</title>
<style>
 body{font-family:system-ui,sans-serif;margin:1.5em;background:#111;color:#eee}
 h1{font-size:1.3em} h2{font-size:1em;margin:1.2em 0 .4em;color:#9db}
 .row{display:flex;flex-wrap:wrap;gap:1.5em}
 .col{flex:1;min-width:20em}
 .card{background:#1c1c22;border:1px solid #333;border-radius:8px;padding:.8em 1em;margin:.5em 0}
 label{display:block;font-size:.8em;color:#9aa;margin:.5em 0 .1em}
 input,select{background:#26262e;color:#eee;border:1px solid #555;border-radius:4px;padding:.25em .4em;width:100%}
 input[type=checkbox]{width:auto}
 button{padding:.35em .9em;border-radius:5px;border:1px solid #555;background:#26262e;color:#eee;cursor:pointer;margin:.2em .3em .2em 0}
 button:hover{background:#33333e} button.primary{background:#2e6fde;border-color:#2e6fde}
 table{border-collapse:collapse;width:100%;font-size:.85em}
 td,th{padding:.2em .5em;border-bottom:1px solid #2a2a30;text-align:left}
 .dot{display:inline-block;width:.7em;height:.7em;border-radius:50%;margin-right:.4em}
 .running{background:#2ecc71}.stopped{background:#e67e22}.crashed{background:#e74c3c}
 .idle{background:#777}.done{background:#3498db}
 .bar{background:#333;border-radius:4px;height:.55em;margin:.3em 0}.bar>div{background:#2e86de;height:100%;border-radius:4px}
 .muted{color:#889;font-size:.8em} .err{color:#e74c3c;font-size:.8em}
 .decks{max-height:11em;overflow:auto;border:1px solid #333;border-radius:5px;padding:.3em}
 .decks label{display:flex;align-items:center;gap:.4em;color:#eee;margin:.1em 0}
</style></head><body>
<h1>RoboMage league — distributed control</h1>
<div class="row">
 <div class="col">
  <h2>1 · Machines</h2>
  <div id="machines" class="card">discovering…</div>
  <div class="card">
    <label>Add a peer agent by IP (if auto-discovery misses it)</label>
    <div style="display:flex;gap:.4em"><input id="peerip" placeholder="100.x.y.z">
    <button onclick="addPeer()">Add</button></div>
  </div>

  <h2>2 · League settings</h2>
  <div id="settings" class="card">loading…</div>

  <h2>3 · Roster</h2>
  <div id="roster" class="card decks">loading…</div>
 </div>

 <div class="col">
  <h2>4 · Distribution</h2>
  <div class="card">
    <button onclick="autobalance('cores')">Auto-balance by cores</button>
    <button onclick="autobalance('even')">Even split</button>
    <div class="muted">Assign each roster deck to a machine. Opponents always
      span the full roster; only the training work is split.</div>
    <div id="dist"></div>
  </div>
  <button class="primary" onclick="start()">▶ Start distributed league</button>
  <span id="startmsg" class="muted"></span>

  <h2>5 · Live status</h2>
  <div style="margin:.3em 0">
    <button onclick="sessionControl('stop')">Stop all</button>
    <button onclick="sessionControl('resume')">Resume all</button>
    <button onclick="sessionControl('restart')">Restart all</button>
  </div>
  <div id="status">—</div>
 </div>
</div>
<script>
let SPEC=null, MACHINES=[], SEL={}, ASSIGN={}, NENVS={};
const $=id=>document.getElementById(id);
const esc=s=>{const d=document.createElement('div');d.textContent=s==null?'':String(s);return d.innerHTML};
const fmt=n=>n==null?'?':Number(n).toLocaleString();

async function jget(u){return (await fetch(u)).json()}
async function jpost(u,b){return (await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)})).json()}

async function loadSpec(){
  SPEC=await jget('/api/spec');
  // settings form
  let h='';
  for(const f of SPEC.fields){
    const id='f_'+f.name.replace(/\W/g,'');
    if(f.kind==='flag'){
      h+=`<label><input type=checkbox id=${id} ${f.default?'checked':''}> ${esc(f.name)}</label>`;
    }else if(f.kind==='choice'){
      h+=`<label>${esc(f.name)}</label><select id=${id}>`+
         f.choices.map(c=>`<option ${c===f.default?'selected':''}>${esc(c)}</option>`).join('')+`</select>`;
    }else{
      const t=(f.kind==='int'||f.kind==='float')?'number':'text';
      h+=`<label title="${esc(f.help)}">${esc(f.name)}</label>`+
         `<input id=${id} type=${t} value="${f.default==null?'':esc(f.default)}">`;
    }
  }
  $('settings').innerHTML=h;
  // roster
  $('roster').innerHTML=SPEC.decks.map(d=>
    `<label><input type=checkbox class=rk value="${esc(d)}" checked onchange="rebuildDist()"> ${esc(d)}</label>`).join('');
  rebuildDist();
}
function selectedDecks(){return [...document.querySelectorAll('.rk:checked')].map(x=>x.value)}
function selectedMachines(){return MACHINES.filter(m=>SEL[m.hostname])}

async function loadMachines(){
  const r=await jget('/api/machines'); MACHINES=r.machines;
  for(const m of MACHINES){ if(SEL[m.hostname]===undefined) SEL[m.hostname]=true; }
  $('machines').innerHTML=MACHINES.map(m=>
    `<div style="display:flex;align-items:center;gap:.5em;margin:.15em 0">
      <label style="flex:1;margin:0"><input type=checkbox ${SEL[m.hostname]?'checked':''} onchange="SEL['${esc(m.hostname)}']=this.checked;rebuildDist()">
        <b>${esc(m.hostname)}</b> <span class=muted>${esc(m.ip)} · ${m.cores} cores · ${esc(m.state)}${m.is_self?' · self':''}</span></label>
      <input type=number min=1 style="width:5.5em" title="envs on this machine (blank = global --n-envs)"
        placeholder="envs" value="${NENVS[m.hostname]??''}"
        onchange="NENVS['${esc(m.hostname)}']=this.value">
    </div>`).join('');
  rebuildDist();
}
async function addPeer(){await jpost('/api/peers/add',{ip:$('peerip').value});$('peerip').value='';loadMachines();}

function autobalance(mode){
  const decks=selectedDecks(), ms=selectedMachines();
  if(!ms.length)return;
  const weights=ms.map(m=>mode==='cores'?Math.max(1,m.cores||1):1);
  const tot=weights.reduce((a,b)=>a+b,0);
  // largest-remainder allocation of deck counts, then slice in order
  let counts=weights.map(w=>Math.floor(decks.length*w/tot));
  let rem=decks.length-counts.reduce((a,b)=>a+b,0);
  const order=weights.map((w,i)=>[decks.length*w/tot-counts[i],i]).sort((a,b)=>b[0]-a[0]);
  for(let k=0;k<rem;k++)counts[order[k%order.length][1]]++;
  ASSIGN={}; let p=0;
  ms.forEach((m,i)=>{for(let k=0;k<counts[i];k++){ASSIGN[decks[p++]]=m.hostname;}});
  rebuildDist();
}
function rebuildDist(){
  const decks=selectedDecks(), ms=selectedMachines();
  const names=ms.map(m=>m.hostname);
  // default: strided assignment if a deck is unassigned or its machine deselected
  decks.forEach((d,i)=>{ if(!ASSIGN[d]||!names.includes(ASSIGN[d])) ASSIGN[d]=names.length?names[i%names.length]:''; });
  let h='<table><tr><th>deck</th><th>machine</th></tr>';
  for(const d of decks){
    h+=`<tr><td>${esc(d)}</td><td><select onchange="ASSIGN['${esc(d)}']=this.value">`+
       names.map(n=>`<option ${n===ASSIGN[d]?'selected':''}>${esc(n)}</option>`).join('')+`</select></td></tr>`;
  }
  $('dist').innerHTML=h+'</table>';
}
function gatherSettings(){
  const s={};
  for(const f of SPEC.fields){
    const el=$('f_'+f.name.replace(/\W/g,'')); if(!el)continue;
    if(f.kind==='flag')s[f.name]=el.checked;
    else if(el.value!=='')s[f.name]=el.value;
  }
  return s;
}
async function start(){
  const roster=selectedDecks(), ms=selectedMachines();
  const byHost={}; ms.forEach(m=>byHost[m.hostname]=m);
  const amap={};
  for(const d of roster){const h=ASSIGN[d]; if(!h)continue; (amap[h]=amap[h]||[]).push(d);}
  const assignments=Object.entries(amap).map(([host,decks])=>(
    {host,ip:byHost[host]?.ip,train_decks:decks,n_envs:NENVS[host]||undefined}));
  $('startmsg').textContent='starting…';
  const r=await jpost('/api/session/start',{settings:gatherSettings(),roster,assignments});
  $('startmsg').textContent=r.error?('error: '+r.error):('started session '+r.session_id);
  loadStatus();
}
async function sessionControl(cmd){await jpost('/api/session/control',{command:cmd});loadStatus();}

async function loadStatus(){
  const r=await jget('/api/status');
  $('status').innerHTML=r.machines.map(m=>{
    const cls=m.state||'idle';
    const pct=m.total_timesteps?(100*m.steps_done/m.total_timesteps).toFixed(1):0;
    const decks=(m.train_decks||[]).join(', ')||'—';
    return `<div class="card"><b><span class="dot ${cls}"></span>${esc(m.hostname)}</b>
      <span class=muted>${esc(m.state)} · restarts ${m.restarts||0}</span><br>
      <span class=muted>learner: <b>${esc(m.learner||'—')}</b> · rotation ${m.rotation??'—'} · decks: ${esc(decks)}</span>
      <div class=bar><div style="width:${pct}%"></div></div>
      <span class=muted>${fmt(m.steps_done)} / ${fmt(m.total_timesteps)} (${pct}%)</span>
      <div class=muted>${esc(m.last_log_line||'')}</div>
      ${m.log?`<div class=muted>log: ${esc(m.log)}</div>`:''}</div>`;
  }).join('')||'<span class=muted>no agents found</span>';
}
loadSpec(); loadMachines(); loadStatus();
setInterval(loadMachines, 8000); setInterval(loadStatus, 5000);
</script></body></html>
"""


# ── HTTP server ───────────────────────────────────────────────────────────────
def _probe_agent(ip, port, timeout=PROBE_TIMEOUT):
    try:
        with urllib.request.urlopen(f"http://{ip}:{port}/api/agent/info",
                                    timeout=timeout) as r:
            info = json.loads(r.read())
            info["ip"] = ip
            info["reachable"] = True
            return info
    except (OSError, ValueError):
        return None


def _post_agent(ip, port, path, body, token, timeout=15):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-League-Token"] = token
    req = urllib.request.Request(f"http://{ip}:{port}{path}", data=data,
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}", "detail": e.read().decode(errors="replace")}
    except (OSError, ValueError) as e:
        return {"error": str(e)}


def make_handler(trainer, port, token):
    self_ip = self_tailscale_ip() or "127.0.0.1"

    def discover():
        """All reachable agents (self + tailnet peers + manual), keyed by host."""
        machines = {}
        me = trainer.info()
        me["ip"] = self_ip
        me["reachable"] = True
        me["is_self"] = True
        machines[me["hostname"]] = me
        candidates = list(tailnet_peer_ips()) + [(ip, ip) for ip in Handler.manual_peers]
        for ip, _host in candidates:
            if ip == self_ip:
                continue
            info = _probe_agent(ip, port)
            if info and info.get("hostname") not in machines:
                info["is_self"] = False
                machines[info["hostname"]] = info
        return list(machines.values())

    class Handler(BaseHTTPRequestHandler):
        manual_peers = []

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authed(self):
            if not token:
                return True
            return self.headers.get("X-League-Token") == token

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        # -- GET --
        def do_GET(self):
            try:
                self._route_get()
            except Exception as exc:  # never drop the connection on a bug
                self._send(500, {"error": repr(exc)})

        def _route_get(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, _PAGE, "text/html; charset=utf-8")
            elif path == "/api/spec":
                self._send(200, {"fields": league_form_schema(),
                                 "decks": _scan_league_decks(),
                                 "self": {"hostname": socket.gethostname(),
                                          "cores": os.cpu_count()}})
            elif path == "/api/agent/info":
                self._send(200, trainer.info())
            elif path == "/api/machines":
                self._send(200, {"machines": discover(), "now": time.time()})
            elif path == "/api/status":
                self._send(200, {"machines": discover(), "now": time.time()})
            else:
                self._send(404, {"error": "not found"})

        # -- POST --
        def do_POST(self):
            try:
                self._route_post()
            except Exception as exc:  # never drop the connection on a bug
                self._send(500, {"error": repr(exc)})

        def _route_post(self):
            path = self.path.split("?")[0]
            if not self._authed():
                self._send(403, {"error": "bad or missing X-League-Token"})
                return
            try:
                body = self._body()
            except (ValueError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"bad json: {e}"})
                return
            if path == "/api/agent/launch":
                trainer.launch(body)
                self._send(200, {"ok": True, "info": trainer.info()})
            elif path == "/api/agent/control":
                self._send(200, self._local_control(body.get("command")))
            elif path == "/api/session/start":
                self._send(200, self._session_start(body))
            elif path == "/api/session/control":
                self._send(200, self._session_control(body))
            elif path == "/api/peers/add":
                ip = (body.get("ip") or "").strip()
                if ip and ip not in Handler.manual_peers:
                    Handler.manual_peers.append(ip)
                self._send(200, {"ok": True, "manual_peers": Handler.manual_peers})
            else:
                self._send(404, {"error": "not found"})

        def _local_control(self, command):
            if command == "stop":
                trainer.stop()
            elif command == "resume":
                if not trainer.resume():
                    return {"error": "no session to resume"}
            elif command == "restart":
                if not trainer.restart():
                    return {"error": "no session to restart"}
            else:
                return {"error": f"bad command {command!r}"}
            return {"ok": True, "info": trainer.info()}

        def _session_start(self, body):
            """Controller: assign shards and fan launches out to each machine.

            body = {settings, roster:[...], assignments:[{host, ip, train_decks:[...]}]}
            """
            settings = body.get("settings") or {}
            roster = body.get("roster") or []
            assignments = [a for a in (body.get("assignments") or [])
                           if a.get("train_decks")]
            if not roster:
                return {"error": "empty roster"}
            if not assignments:
                return {"error": "no machines assigned any decks"}
            n = len(assignments)
            session_id = body.get("session_id") or f"s{int(time.time())}"
            results = []
            for i, a in enumerate(assignments):
                # Per-machine settings: a machine's --n-envs override (from the UI)
                # wins over the shared global value, so each host can run an env
                # count matched to its core budget.
                machine_settings = dict(settings)
                if a.get("n_envs") not in (None, "", 0):
                    machine_settings["--n-envs"] = a["n_envs"]
                config = {
                    "session_id": session_id,
                    "roster": roster,
                    "train_decks": a["train_decks"],
                    "shard": f"{i}/{n}",
                    "settings": machine_settings,
                    "binary": body.get("binary") or BINARY,
                }
                ip = a.get("ip") or self_ip
                if ip == self_ip or a.get("host") == socket.gethostname():
                    trainer.launch(config)
                    res = {"ok": True, "info": trainer.info()}
                else:
                    res = _post_agent(ip, port, "/api/agent/launch", config, token)
                results.append({"host": a.get("host"), "ip": ip,
                                "shard": config["shard"], "result": res})
            return {"ok": True, "session_id": session_id, "results": results}

        def _session_control(self, body):
            command = body.get("command")
            targets = body.get("machines")
            if targets is None:
                targets = [{"host": m["hostname"], "ip": m.get("ip")}
                           for m in discover()]
            results = []
            for t in targets:
                ip = t.get("ip") or self_ip
                if ip == self_ip or t.get("host") == socket.gethostname():
                    res = self._local_control(command)
                else:
                    res = _post_agent(ip, port, "/api/agent/control",
                                      {"command": command}, token)
                results.append({"host": t.get("host"), "result": res})
            return {"ok": True, "results": results}

        def log_message(self, *a):
            pass

    Handler.manual_peers = list(Handler.manual_peers)
    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default=None,
                    help="Bind address (default: this machine's Tailscale IP, "
                         "else 127.0.0.1; pass 0.0.0.0 to expose on all NICs)")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR)
    ap.add_argument("--venv-python", default=DEFAULT_VENV_PY,
                    help="Python that runs train.py (default: train/.venv)")
    ap.add_argument("--token", default=os.environ.get("LEAGUE_TOKEN"),
                    help="Shared secret required on control/launch endpoints "
                         "(all agents must match; or set LEAGUE_TOKEN)")
    ap.add_argument("--peer", action="append", default=[],
                    help="Extra peer agent IP to include beyond tailnet "
                         "auto-discovery (repeatable)")
    args = ap.parse_args(argv)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    _install_stignore(args.checkpoint_dir)
    trainer = Trainer(args.checkpoint_dir, args.venv_python)
    Handler = make_handler(trainer, args.port, args.token)
    Handler.manual_peers = list(args.peer)
    host = args.host or self_tailscale_ip() or "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), Handler)
    print(f"[agent] {socket.gethostname()} serving http://{host}:{args.port}  "
          f"(checkpoints: {args.checkpoint_dir}"
          + (", token required" if args.token else "") + ")", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[agent] shutting down; stopping trainer", flush=True)
        trainer.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
