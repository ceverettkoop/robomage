# Distributed league training (two machines over Tailscale)

Run one league driver per machine, each training a **disjoint slice of the
roster** (`--shard i/n`) while sampling opponents from the **full** roster, with
`train/checkpoints/` kept in sync by **Syncthing** over the tailnet. Because
deck ownership is disjoint, the two drivers never write the same file — no
locks, no coordinator. Each machine's `LeaguePool` rescans the checkpoint dir
every 50 episodes and automatically ingests the peer's `{deck}__v*.zip`
snapshots as opponents.

A stdlib-only supervisor (`scripts/league_worker.py`) auto-restarts a crashed
trainer with `--resume`, and a web dashboard (`scripts/league_dashboard.py`)
on either machine monitors both and issues stop/resume/restart — status and
control both ride the synced folder, so workers need no open ports.

## One-time setup

1. **Tailscale** on both machines (already assumed); note each machine's
   tailnet IP/hostname (`tailscale ip -4`).
2. **Syncthing** on both machines:
   - Install (`apt install syncthing` or per platform), run it as a service
     (`systemctl enable --now syncthing@<user>`).
   - In each Syncthing GUI (`http://127.0.0.1:8384`), *Add Remote Device* using
     the other machine's device ID; set its address to
     `tcp://<tailscale-ip>:22000` so sync rides the tailnet.
   - Share the folder `<repo>/train/checkpoints` from one machine, accept it on
     the other, **Send & Receive on both sides**.
   - The first worker start copies `scripts/checkpoints.stignore` to
     `train/checkpoints/.stignore` (never replicate `*.tmp` atomic-write
     intermediates). Verify it's there after the first run.
3. Both repos built (`make`) and the training venv installed on each machine.

## Starting a run

Same full roster on both machines; only `--shard` differs. With 7 decks,
shard `0/2` trains decks 0,2,4,6 and shard `1/2` trains 1,3,5 (strided).

Machine 1:
```bash
scripts/league_worker.py --shard 0/2 -- \
  train/.venv/bin/python train/train.py league \
  --binary bin/robomage \
  --decks league/bug,league/tron,league/car_doomsday,league/wrb_energy,league/ur_delver,league/gw_maverick,league/bw_dnt \
  --total-timesteps 8000000 --bo3
```

Machine 2: identical command with `--shard 1/2` (and its own
`--total-timesteps`).

Notes:
- **Do not pass `--shard`/`--resume` in the base command** — the worker appends
  them (it adds `--resume` automatically whenever the shard's sidecar exists,
  which is what makes crash restarts and reboots hands-off).
- **`--total-timesteps` is per-machine**: it is the compute *that machine*
  spends on *its* decks. There is deliberately no cross-machine step counter.
  Size each budget to its shard (deck count × steps you want per deck).
- Running bare `train.py league --shard i/n ...` without the supervisor also
  works (you just lose auto-restart, the heartbeat, and dashboard control).
- Each shard writes its own sidecar
  (`_league_progress.shard{i}of{n}.json`), so a manual resume is
  `train.py league --resume --shard i/n`.

## Dashboard

On either machine (it only reads/writes the local synced folder, so it sees and
controls both shards from anywhere):

```bash
scripts/league_dashboard.py            # binds the Tailscale IP if detectable
# or explicitly: scripts/league_dashboard.py --host 100.x.y.z --port 8787
```

Open `http://<tailscale-ip>:8787` from any tailnet device. Per shard it shows:
worker liveness (green running / orange stopped / red crashed / grey stale),
hostname, current learner deck, rotation progress, total-step progress bar,
per-deck win-rate + last-snapshot table, the trainer's last log line, and
**Stop / Resume / Restart** buttons.

- Buttons write `_control.shard{i}of{n}.json` into the synced folder; the
  target worker applies the command on its next poll and echoes an ack. Expect
  a few seconds of latency (a Syncthing round-trip) for a *remote* worker — the
  UI shows the command as *pending* until acked. A command issued while a
  worker is offline applies when it comes back.
- **Stop** SIGINTs the trainer process group. Progress is safe: the sidecar is
  rewritten at every snapshot, so at most one snapshot interval of training is
  lost, and **Resume** relaunches with `--resume`.
- The dashboard is stateless — if its host goes down, just start it on the
  other machine.

## Failure behavior

| Event | What happens |
|---|---|
| Trainer process crashes | Worker relaunches it with `--resume` after ~10 s (restart count on the dashboard). |
| Whole machine goes down | Its shard pauses; the other machine keeps training against the last-synced snapshots. On reboot, the systemd unit (below) or a manual worker start resumes the shard. |
| Sync link down | Both keep training; snapshots exchange when the link returns (Syncthing retries forever). |
| Dashboard host down | Web UI only; training unaffected. |

## Boot-time start (optional)

`scripts/league-worker@.service` is a systemd template (instance name = shard
with `of` for `/`, e.g. `0of2`). Edit the `User=`, `ROBOMAGE=` path, and
`WORKER_ARGS=` league flags in the file (or via `systemctl edit`), then:

```bash
sudo cp scripts/league-worker@.service /etc/systemd/system/
sudo systemctl enable --now league-worker@0of2     # machine 1
sudo systemctl enable --now league-worker@1of2     # machine 2
```

## Caveats (known, accepted)

- **Stale `__final` in a long-running peer pool**: `LeaguePool` caches loaded
  models by path and `{deck}__final.zip` is overwritten in place, so a pool
  that already loaded a peer deck's `__final` won't reload the newer weights.
  The version-stamped `{deck}__v*.zip` snapshots (fresh filenames every time)
  are ingested continuously, so opponent freshness is unaffected in practice.
- **Sync lag** (seconds) on opponents and on dashboard control — irrelevant for
  training quality, visible as *pending* in the UI.
- **Adaptive rotations** (`--adaptive-boost`) compare cumulative steps against
  the *full* roster via snapshot filenames, so the "catch-up" signal correctly
  spans both machines once sync is flowing.
- Do not run two workers with the **same** shard spec against one synced folder
  (they would race on the same sidecar/checkpoints; nothing detects this).
