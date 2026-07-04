# Distributed league training (quickstart)

Run one league driver per machine, each training a disjoint slice of the
roster (`--shard i/n`) while sampling opponents from the full roster, with
`train/checkpoints/` synced by Syncthing over Tailscale. A stdlib web
dashboard on either machine lets you monitor and control both by hand — no
auto-start, just glance at the page and click Stop/Resume/Restart as needed.

## Setup (once)

1. **Tailscale** running on both machines (`tailscale ip -4` to get each
   machine's tailnet IP).
2. **Syncthing** on both machines: add each other as a remote device
   (address `tcp://<tailscale-ip>:22000`), then share the folder
   `<repo>/train/checkpoints`, **Send & Receive on both sides**. (The worker
   auto-installs `scripts/checkpoints.stignore` into that folder on first run.)
3. Both machines: repo built (`make`) and the training venv installed.

## Run (one worker per machine)

Same full `--decks` roster on both machines; only `--shard` differs. Don't
add `--shard`/`--resume` to the base command yourself — the worker appends
them (and adds `--resume` automatically once a sidecar exists, so a crash
restarts hands-off).

Machine 1:
```bash
scripts/league_worker.py --shard 0/2 -- \
  train/.venv/bin/python train/train.py league \
  --binary bin/robomage \
  --decks league/bug,league/tron,league/car_doomsday,league/wrb_energy,league/ur_delver,league/gw_maverick,league/bw_dnt \
  --total-timesteps 8000000 --bo3
```

Machine 2: same command with `--shard 1/2` (and its own `--total-timesteps`
— the budget is per-machine, not global).

## Monitor & control (web interface)

On either machine:
```bash
scripts/league_dashboard.py            # auto-binds this machine's Tailscale IP
# or: scripts/league_dashboard.py --host <tailscale-ip> --port 8787
```
Open `http://<tailscale-ip>:8787` from any device on the tailnet. It shows a
card per shard — liveness, current learner deck, rotation/step progress,
per-deck win-rates, last log line — with **Stop / Resume / Restart** buttons.
Button clicks write into the synced folder, so expect a few seconds of
latency for a remote worker (shown as *pending* until acked). Stop is
snapshot-safe (at most one snapshot interval of progress is lost).

## Notes

- A crashed trainer is auto-restarted by its own worker with `--resume` — no
  action needed.
- If a machine goes down, its shard just pauses; the other keeps training.
  Restart that machine's worker command when it's back to resume its shard.
- Don't run two workers with the same `--shard` against one synced folder.
