# Distributed league training (quickstart)

Train one league across several machines: each trains a disjoint slice of the
roster while sampling opponents from the full roster, with `train/checkpoints/`
synced by Syncthing over Tailscale. You drive the whole thing from a **web UI** —
start one agent per machine, open the page on any of them, configure the league,
choose how to split the roster across machines, and hit Start.

All machines train the **one generalist model** (`gen__final.zip` +
`gen__v{steps}.zip` snapshots; there is no longer a separate `{deck}__final.zip`
per deck) — a machine's deck slice only decides which decks *its* learner pilots
this rotation, not which checkpoint it writes. Keep each machine on a disjoint
deck slice (per the last note below) so their snapshot streams don't collide as
Syncthing replicates the shared `gen` checkpoints.

## Setup (once)

1. **Tailscale** running on every machine.
2. **Syncthing** on every machine, sharing `<repo>/train/checkpoints`
   (**Send & Receive** on all), peer address `tcp://<tailscale-ip>:22000`.
   (The agent drops in `scripts/checkpoints.stignore` on first run so `*.tmp`
   files aren't replicated.)
3. Each machine: repo built (`make`) and the training venv installed.

## Run it from the browser

Start the agent on **every** machine (keep it running, e.g. in tmux):
```bash
scripts/league_agent.py                     # binds this machine's Tailscale IP, port 8787
# optional shared secret so only you can launch jobs:
scripts/league_agent.py --token MYSECRET    # (use the SAME token on every machine)
```

Open `http://<any-machine-tailscale-ip>:8787` in a browser on the tailnet. The
page walks top to bottom:

1. **Machines** — auto-discovered agents on the tailnet (hostname, IP, cores,
   state). Untick any you don't want to use; use *Add a peer* if one isn't
   found automatically. Each machine has an **envs** box to set its own
   `--n-envs` (parallel game processes) — leave blank to use the global value,
   or size it to that machine's cores.
2. **League settings** — the same options as the TUI league form
   (total-timesteps, rotate-every, self-play-frac, pfsp-p, adaptive-boost,
   bo3, …), pre-filled with defaults.
3. **Roster** — the decks to train (all league decks ticked by default).
4. **Distribution** — a deck→machine table. Hit **Auto-balance by cores** or
   **Even split**, or set each deck's machine by hand. Opponents always span the
   full roster; only the training work is divided.
5. **Start** — the controller assigns each machine its shard and launches it.
   Below, **Live status** shows every machine's current learner deck, rotation,
   step progress, and last log line, with **Stop / Resume / Restart** (per the
   whole session). Each machine auto-restarts its own trainer on a crash.

`--total-timesteps` is per-machine — the compute that machine spends on its
decks. Machines assigned no decks simply don't start.

Each machine tees its session output to `logs/distributed/<session>_<host><shard>.log`
(shown in the status card and reported in `/api/agent/info`); crash-restarts and
Resume of the same session append to it.

## Notes

- Keep each `league_agent.py` running (tmux/screen). If an agent is stopped,
  its training stops too but the checkpoint sidecar persists — relaunch the
  agent and hit Start/Resume to pick up where it left off.
- A crashed *trainer* is auto-restarted by its agent with `--resume`; a whole
  machine going down only pauses its own decks, and the others keep training
  against the last-synced snapshots.
- Bind to the Tailscale IP (the default) so the UI is reachable only on your
  tailnet. Add `--token` for an extra shared-secret check on launch/control.
- Don't point two agents at the same machine's checkpoint dir with overlapping
  decks.

## Manual / headless alternative (no web UI)

You can still drive shards from the command line with `train.py league --shard
i/n` (strided split) or `--train-decks A,B,...` (explicit subset), optionally
under `scripts/league_worker.py` for crash-restart. Example on machine 1:
```bash
scripts/league_worker.py --shard 0/2 -- \
  train/.venv/bin/python train/train.py league --binary bin/robomage \
  --decks league/bug,league/tron,league/car_doomsday,league/wrb_energy,league/ur_delver,league/gw_maverick,league/bw_dnt \
  --total-timesteps 8000000 --bo3
```
Machine 2: same with `--shard 1/2`. The agent flow above is just this wired to a
UI, so the two interoperate (both write the same `_league_progress.shard*.json`
sidecars and share the Syncthing pool).
