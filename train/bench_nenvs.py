#!/usr/bin/env python3
"""Benchmark training throughput vs. n_envs to find this machine's optimum.

Parallel data collection in ``train.py`` uses ``SubprocVecEnv`` — one OS process
per environment, each running the C++ ``bin/robomage`` binary plus its Python
wrapper. The best ``--n-envs`` is machine-specific: it climbs with spare physical
cores, then flattens once CPU/memory-bandwidth saturates, and eventually *drops*
when envs exceed cores or RAM starts to swap. Model-vs-model (self-play / league)
is the expensive case the built-in ``N_ENVS_SELF_PLAY = 10`` default is sized for,
because every env additionally loads a torch opponent policy — so RAM and thread
contention bite far sooner than in scripted play.

This tool sweeps a set of ``n_envs`` values through the *real* training path and
reports, for each:
  * steady-state throughput (env steps / second) — the metric to maximize
  * per-env throughput (steps/s ÷ n_envs) — shows where parallel scaling stalls
  * peak resident memory of the whole process tree (the hard RAM ceiling)
  * measured-phase wall-clock

For each ``n_envs`` it builds the same vec-env + ``MaskablePPO`` that ``train.py``
uses, runs one **warmup** rollout (pays process spawn, torch import, and the
opponent-model load) and then a **timed** measured phase of ``--rollouts``
rollouts. Nothing is saved and no checkpoint is written or overwritten — this is
a pure timing harness.

Usage (run from the repo root):

    train/.venv/bin/python train/bench_nenvs.py --deck delver --opponent delver
    train/.venv/bin/python train/bench_nenvs.py --mode self-play --envs 4,6,8,10,12,16
    train/.venv/bin/python train/bench_nenvs.py --mode scripted --deck burn --opponent mav

Notes:
  * ``--mode self-play`` (default) benchmarks the model-vs-model path and needs an
    opponent checkpoint (``{opponent}__final.zip`` / newest ``{opponent}__v*.zip``)
    to exist — otherwise SelfPlayEnv silently falls back to the scripted agent and
    the numbers understate real self-play cost. The tool warns when that happens.
  * ``--mode scripted`` benchmarks the cheaper scripted-opponent path (sized around
    the ``N_ENVS = 32`` default) and needs no checkpoints.
"""

import argparse
import os
import sys
import threading
import time

# Run everything relative to train/ so train.py's flat imports (``from cli_spec
# import ...``) resolve, exactly as when train.py is invoked directly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _page_size():
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (ValueError, AttributeError, OSError):
        return 4096


_PAGE = _page_size()


def _rss_tree_bytes(root_pid):
    """Sum resident memory (bytes) of ``root_pid`` and all its descendants.

    Reads /proc directly so it works without psutil. Returns None if /proc is
    unavailable (non-Linux), in which case the caller falls back to getrusage.
    """
    try:
        all_pids = [int(p) for p in os.listdir("/proc") if p.isdigit()]
    except (FileNotFoundError, NotADirectoryError):
        return None
    children = {}
    for pid in all_pids:
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read()
            # comm (field 2) is parenthesized and may contain spaces/parens;
            # everything after the last ')' is space-delimited: state ppid ...
            rp = data.rindex(b")")
            fields = data[rp + 2:].split()
            ppid = int(fields[1])
        except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError,
                IndexError):
            continue
        children.setdefault(ppid, []).append(pid)

    total = 0
    stack = [root_pid]
    seen = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            with open(f"/proc/{pid}/statm") as f:
                resident_pages = int(f.read().split()[1])
            total += resident_pages * _PAGE
        except (FileNotFoundError, ProcessLookupError, ValueError, PermissionError,
                IndexError):
            pass
        stack.extend(children.get(pid, []))
    return total


class _PeakRSSSampler(threading.Thread):
    """Background thread tracking peak process-tree RSS while a phase runs."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self._interval = interval
        self._stop_evt = threading.Event()
        self._root = os.getpid()
        self.peak_bytes = 0
        self.supported = _rss_tree_bytes(self._root) is not None

    def run(self):
        if not self.supported:
            return
        while not self._stop_evt.is_set():
            cur = _rss_tree_bytes(self._root)
            if cur is not None and cur > self.peak_bytes:
                self.peak_bytes = cur
            self._stop_evt.wait(self._interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=2.0)


def _default_env_sweep():
    """A reasonable default sweep derived from this machine's core count."""
    cpu = os.cpu_count() or 8
    candidates = {2, 4, 6, 8, 10, 12, 16, max(1, cpu - 1), cpu}
    cap = int(cpu * 1.5)
    return sorted(v for v in candidates if 1 <= v <= cap)


def _build_vec_env(mode, n_envs, deck, opp_deck, env_kwargs):
    """Build the same SubprocVecEnv train.py would, for the chosen opponent mode."""
    import train as T
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if mode == "self-play":
        factories = [
            T.make_self_play_env(T._CHECKPOINT_ABS, i, deck, opp_deck, **env_kwargs)
            for i in range(n_envs)
        ]
    else:  # scripted
        factories = [
            T.make_env(i, deck, opp_deck, opponent_pool=None, opp_ckpt_ratio=1.0,
                       n_envs=n_envs, **env_kwargs)
            for i in range(n_envs)
        ]
    return SubprocVecEnv(factories)


def _make_model(vec_env, embed_dim):
    """Fresh MaskablePPO matching train.py's hyperparameters (no checkpoint I/O)."""
    import train as T
    policy_kwargs = dict(
        features_extractor_class=T.CardGameExtractor,
        features_extractor_kwargs=dict(embed_dim=embed_dim),
        net_arch=[256, 256],
    )
    return T.MaskablePPO(
        "MlpPolicy",
        vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=1024,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.25,
        ent_coef=0.12,
        verbose=0,
    )


def _bench_one(mode, n_envs, deck, opp_deck, steps_per_env, rollouts, embed_dim):
    """Return a result dict for a single n_envs value (or an 'error' key)."""
    vec_env = None
    sampler = _PeakRSSSampler()
    try:
        sampler.start()
        vec_env = _build_vec_env(mode, n_envs, deck, opp_deck,
                                 {"binary_path": _BINARY})
        model = _make_model(vec_env, embed_dim)

        # Warmup: pays subprocess spawn, torch/JIT warmup, and (self-play) the
        # opponent-model load at first reset. reset_num_timesteps=True.
        warmup_steps = steps_per_env * n_envs
        model.learn(total_timesteps=warmup_steps, reset_num_timesteps=True)

        # Measured phase: steady-state throughput over `rollouts` rollouts.
        measured_steps = steps_per_env * n_envs * rollouts
        t0 = time.perf_counter()
        model.learn(total_timesteps=measured_steps, reset_num_timesteps=False)
        elapsed = time.perf_counter() - t0

        steps_per_s = measured_steps / elapsed if elapsed > 0 else 0.0
        return {
            "n_envs": n_envs,
            "elapsed": elapsed,
            "steps_per_s": steps_per_s,
            "steps_per_s_per_env": steps_per_s / n_envs,
            "peak_rss_gb": (sampler.peak_bytes / 1e9) if sampler.supported else None,
            "measured_steps": measured_steps,
        }
    except Exception as e:  # noqa: BLE001 — report and continue the sweep
        return {"n_envs": n_envs, "error": f"{type(e).__name__}: {e}"}
    finally:
        sampler.stop()
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:  # noqa: BLE001
                pass


def _print_table(results, ram_budget_gb):
    ok = [r for r in results if "error" not in r]
    print("\n" + "=" * 78)
    print("n_envs benchmark results")
    print("=" * 78)
    header = f"{'n_envs':>7} {'steps/s':>10} {'steps/s/env':>12} {'peak RAM':>10} {'wall (s)':>9}"
    print(header)
    print("-" * 78)
    for r in results:
        if "error" in r:
            print(f"{r['n_envs']:>7}   ERROR: {r['error']}")
            continue
        ram = f"{r['peak_rss_gb']:.2f} GB" if r["peak_rss_gb"] is not None else "n/a"
        over = ""
        if ram_budget_gb and r["peak_rss_gb"] and r["peak_rss_gb"] > ram_budget_gb:
            over = "  (over budget)"
        print(f"{r['n_envs']:>7} {r['steps_per_s']:>10.1f} "
              f"{r['steps_per_s_per_env']:>12.1f} {ram:>10} {r['elapsed']:>9.1f}{over}")
    print("-" * 78)

    if not ok:
        print("No successful runs — check the opponent checkpoint / mode and try again.")
        return

    # Candidates within RAM budget (if given), else all successful runs.
    feasible = ok
    if ram_budget_gb:
        feasible = [r for r in ok
                    if r["peak_rss_gb"] is None or r["peak_rss_gb"] <= ram_budget_gb] or ok

    best = max(feasible, key=lambda r: r["steps_per_s"])
    print(f"\nRecommended --n-envs = {best['n_envs']}  "
          f"({best['steps_per_s']:.1f} steps/s"
          + (f", {best['peak_rss_gb']:.2f} GB peak" if best["peak_rss_gb"] is not None else "")
          + ")")
    if ram_budget_gb:
        print(f"(highest throughput with peak RAM <= {ram_budget_gb:.1f} GB)")
    print("Tip: the optimum is the 'knee' — where steps/s stops climbing. If the top\n"
          "of the sweep is still the fastest, extend --envs higher to find the plateau.")


_BINARY = None


def main(argv=None):
    global _BINARY
    from cli_spec import BINARY, N_ENVS, N_ENVS_SELF_PLAY, EMBED_DIM

    p = argparse.ArgumentParser(
        description="Benchmark training throughput vs. n_envs to size --n-envs for this machine.")
    p.add_argument("--mode", choices=("self-play", "scripted"), default="self-play",
                   help="Opponent path to benchmark (default: self-play = model-vs-model).")
    p.add_argument("--deck", default="delver", help="Deck the learner pilots (.dk stem).")
    p.add_argument("--opponent", default=None,
                   help="Opponent deck (.dk stem). Default: mirror (--deck).")
    p.add_argument("--envs", default=None,
                   help="Comma-separated n_envs values to sweep (default: derived from CPU count).")
    p.add_argument("--steps-per-env", type=int, default=4096,
                   help="Env steps per rollout per env (matches train.py n_steps; default 4096).")
    p.add_argument("--rollouts", type=int, default=2,
                   help="Rollouts in the timed measured phase after warmup (default 2).")
    p.add_argument("--embed-dim", type=int, default=EMBED_DIM,
                   help="Feature-extractor embed dim for the throwaway model.")
    p.add_argument("--ram-budget-gb", type=float, default=None,
                   help="If set, recommend the fastest n_envs whose peak RAM stays under this.")
    p.add_argument("--binary", default=BINARY, help="Path to the robomage binary.")
    args = p.parse_args(argv)

    _BINARY = args.binary
    deck = args.deck
    opp_deck = args.opponent or args.deck
    sweep = ([int(x) for x in args.envs.split(",") if x.strip()]
             if args.envs else _default_env_sweep())

    if not os.path.exists(args.binary):
        p.error(f"binary not found: {args.binary} — build it first (make HEADLESS=TRUE)")

    print(f"Machine: {os.cpu_count()} logical CPUs")
    print(f"Mode: {args.mode}  |  deck={deck}  opponent={opp_deck}")
    print(f"Sweep n_envs: {sweep}")
    print(f"Per point: 1 warmup rollout + {args.rollouts} measured rollouts "
          f"({args.steps_per_env} steps/env/rollout)")
    print(f"Built-in defaults for reference: N_ENVS={N_ENVS} (scripted), "
          f"N_ENVS_SELF_PLAY={N_ENVS_SELF_PLAY} (self-play)")

    if args.mode == "self-play":
        import train as T
        resolved = T._resolve_model(opp_deck)
        if not os.path.exists(resolved):
            print(f"\n[warn] No opponent checkpoint for '{opp_deck}' "
                  f"({resolved}). SelfPlayEnv will fall back to the scripted agent, "
                  f"so these numbers understate real model-vs-model cost.\n"
                  f"       Train a snapshot first, or benchmark --mode scripted.")

    results = []
    for n in sweep:
        print(f"\n--- n_envs={n} ---", flush=True)
        r = _bench_one(args.mode, n, deck, opp_deck,
                       args.steps_per_env, args.rollouts, args.embed_dim)
        if "error" in r:
            print(f"    ERROR: {r['error']}")
        else:
            ram = f"{r['peak_rss_gb']:.2f} GB" if r["peak_rss_gb"] is not None else "n/a"
            print(f"    {r['steps_per_s']:.1f} steps/s "
                  f"({r['steps_per_s_per_env']:.1f}/env), peak {ram}, "
                  f"{r['elapsed']:.1f}s")
        results.append(r)

    _print_table(results, args.ram_budget_gb)


if __name__ == "__main__":
    main()
