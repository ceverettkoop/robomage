"""The run's configuration (docs/rmtrain.md §3): defaults for everything, a TOML or JSON file over
them, ``--set section.key=value`` overrides, validation, and the parameters that may change while
the run is running (``rmtrain set``).

    cfg = resolve(load_file("my.toml"), ["actors.count=4", "model.d=64"])   # a nested dict
    validate(cfg)                     # ValueError naming the key
    print(defaults_toml())            # the defaults as a commented TOML file
    value = parse_live("rnad.eta", "0.1")   # (key, value) or ValueError

The config is nested: ``{section: {key: value}}``. Every key has a default (``SPEC``); a file or an
override may set only known keys, with the default's type (an int where a float is expected is
accepted; a list's items are not typed beyond being scalars or lists). ``LIVE`` names the keys
``rmtrain set`` may change on a running run; the others are read once at start (a resumed run reads
its resolved ``config.json``, so ``start --set`` changes them for the resumed run).
"""

import copy
import json
import os
import re
import tomllib

# The default league.decks: the ten decks the existing runs train on, named, not read from
# decks/league/, so a deck onboarded there later (docs/open_vocab_plan.md S7: eldrazi_aggro) joins a run
# only through its league.decks (S8).
LEAGUE = ["bug", "burn", "bw_dnt", "gw_maverick", "lands", "reanimator", "ur_delver", "uw_control", "wrb_energy",
          "wubg_doomsday"]
EVAL_PAIRS = [["ur_delver", "burn"], ["bug", "bw_dnt"], ["uw_control", "wrb_energy"], ["gw_maverick", "lands"]]

# section -> [(key, default, doc)]
SPEC = {
    "run": [
        ("seed", 1, "the run seed: actors' --actor-seed, the model's initialisation, the held-out split"),
        ("bin_dir", "bin", "the engine's working directory (relative to the repository)"),
        ("build", "release", "the engine build: bin/<build>/robomage_nn and rmfeat"),
        ("tables", "", "an existing rmfeat --export-tables directory, e.g. another run's tables/ (empty: export into the run's tables/; a relative path is relative to the repository)"),
        ("memory_max", "100G", "systemd MemoryMax of the whole run (orchestrator, GPU process, actors)"),
        ("restart_on_failure", True, "systemd restarts a run that failed (a GPU-process crash, a kill) from its checkpoint"),
        ("restart_delay_s", 60, "the delay before that restart"),
        ("crash_loop_fatals", 3, "fatal stops in a row without progress (the same phase and stage, its step and "
         "collected decisions not past the last fatal stop's) after which the daemon exits with a status systemd "
         "does not restart (4); `rmtrain start` resets the count (0: never)"),
        ("device", "cuda", "the GPU process's torch device (cpu for tests)"),
        ("hsa_override_gfx_version", "10.3.0", "HSA_OVERRIDE_GFX_VERSION of the GPU process (empty: unset)"),
        ("gpu_lock", "/tmp/rm-gpu.lock", "the flock the GPU process holds while it runs (empty: none)"),
        ("gpu_lock_wait", False, "wait for the GPU lock (false: fail at once when another process holds it)"),
        ("torch_threads", 2, "the GPU process's torch CPU threads"),
        ("status_every_s", 2.0, "how often status.json is rewritten"),
        ("start_from", "", "a BC checkpoint (model.save_checkpoint, e.g. another run's ckpt/bc-final.pt): the run "
         "starts at B3, R-NaD seeded from it (B0–B2 skipped; the model's config is the checkpoint's)"),
    ],
    "league": [
        ("decks", list(LEAGUE), "deck stems under bin/resources/decks/league/ (never heldout/); a deck is added to a "
         "run with `rmtrain add-decks` (docs/rmtrain.md §1.1)"),
        ("pairs", [], "[[deck_a, deck_b, weight], ...] over exactly league.decks; empty: every pair of decks, weight 1"),
        ("format", "bo3", "bo3 or bo1"),
        ("knowledge", "expert", "both seats' knowledge level (off, expert)"),
        ("max_game_decisions", 5000, "the per-game decision cap"),
        ("omniscient", True, "episodes carry the omniscient featurization (aux targets; the omniscient critic)"),
    ],
    "actors": [
        ("count", 26, "training actors (engine processes)"),
        ("teacher_tier", "hard", "the scripted teacher's tier (python/teacher.py --tier; hard+sb: hard that sideboards "
         "per the league guides): both scripted seats of B0's corpus, the scripted seat of its explore matches and "
         "every DAgger label; evaluations play scripted:hard whatever this is"),
        ("restart_delay_s", 2.0, "delay before restarting a crashed actor"),
        ("crash_loop_restarts", 5, "restarts within crash_loop_window_s that count as a crash loop"),
        ("crash_loop_window_s", 300.0, "the crash-loop window"),
        ("max_backoff_s", 600.0, "the longest delay before restarting a crash-looping actor"),
        ("stop_timeout_s", 30.0, "how long a stopping actor gets before SIGKILL"),
        ("policy_eval", "local", "how the training actors (DAgger, self-play) evaluate the policy: remote "
         "(nn:remote through the GPU process's inference server) or local (nn:<export> in process, one CPU thread "
         "each; the current policy exported on a cadence; the learner computes the regularisation policies' log π)"),
        ("local_count", 16, "local: the training actors while they play the network in process (DAgger, self-play; "
         "0: actors.count): with the network's forward on the actors' cores, more than ~16 on this 16-core / "
         "32-thread box play fewer decisions in all (docs/rmtrain.md §5.1) (live)"),
        ("export_format", "ts", "local: the actors' export of the current policy (ts: TorchScript, ~0.1 s to export; "
         "pt2: AOTInductor, ~16 s, ~7% faster on the CPU)"),
        ("export_every_s", 30.0, "local: seconds between exports of the current policy for the actors (when the "
         "learner stepped) (live)"),
        ("export_every_steps", 0, "local: also after this many learner steps (0: off) (live)"),
    ],
    "model": [
        ("d", 192, "token width"),
        ("heads", 4, "attention heads"),
        ("layers", 4, "transformer layers"),
        ("ff_mult", 4, "feed-forward width multiple"),
        ("id_dropout", 0.1, "card id-dropout: the probability a card slot reads the unknown-card vector, the path "
         "a card no training batch has held plays (docs/open_vocab_plan.md S4); 0.2-0.25 for runs meant to "
         "generalise to unseen decks, to be decided by the held-out benchmark (S8). The run's value applies to "
         "a resumed BC or R-NaD model and to a start_from checkpoint"),
        ("aux_heads", True, "hidden-information heads"),
        ("omniscient_critic", False, "the omniscient critic (an A/B experiment, plan §4 L4)"),
        ("critic_layers", 2, "the omniscient critic's depth"),
    ],
    "replay": [
        ("max_decisions", 4_000_000, "the window's decisions"),
        ("max_gb", 40.0, "the window's bytes (GiB)"),
        ("heldout_fraction", 0.05, "BC's held-out games"),
        ("max_decision_tokens", 640, "leave out of the window (BC's and R-NaD's) a whole episode with a decision of "
         "more seat tokens, menu tokens included: one pads its whole batch to it, and attention is quadratic in "
         "the padded length (0: no cap; a change applies to episodes ingested from then on, and a lowered cap "
         "drops the window's episodes over it) (live)"),
        ("delete_evicted_selfplay", True, "delete self-play .rmtraj files evicted from the window (RMLOGs are kept)"),
        ("poll_interval_s", 2.0, "the window's directory polling interval"),
    ],
    "server": [
        ("max_batch", 512, "the inference server's batch"),
        ("deadline_us", 2000, "its micro-timeout"),
        ("precision", "fp32", "fp32, fp16 or bf16 (autocast)"),
        ("attention_budget", 1 << 26, "the most attention cells (rows × padded tokens², per head and layer) of one "
         "forward of a batch, so a few long decisions run in forwards of their own (~2 GiB of scores at fp32 with 4 "
         "heads)"),
    ],
    "b0": [
        ("corpus_dir", "", "reuse an existing corpus's episode directory (skips B0)"),
        ("decisions", 5_000_000, "B0 ends after this many decisions played"),
        ("explore_frac", 0.2, "the share of scripted-vs-explore matches"),
    ],
    "bc": [
        ("batch_size", 128, ""),
        ("lr", 3e-4, "learning rate (live)"),
        ("weight_decay", 0.0, ""),
        ("warmup", 300, ""),
        ("decay_steps", 0, "cosine decay steps after warm-up (0: constant)"),
        ("lr_min", 3e-5, ""),
        ("label_smoothing", 0.05, ""),
        ("value_weight", 0.5, ""),
        ("aux_weight", 0.05, ""),
        ("grad_clip", 1.0, ""),
        ("sideboard_weight", 1.0, "the policy loss's weight of a sideboarding decision against 1 for any other (a "
         "weighted mean over the batch; value and aux losses unweighted)"),
        ("amp", "fp16", "fp16, bf16 or off"),
        ("loader_workers", 2, ""),
        ("sort_pool", 4, ""),
        ("prefetch", 8, ""),
        ("eval_limit", 30000, "held-out decisions per evaluation"),
        ("eval_batch", 256, ""),
        ("unit_steps", 25, "steps between command checks in the GPU process"),
        ("report_every", 250, "steps per metrics line"),
        ("eval_every", 2000, "steps per held-out evaluation"),
    ],
    "b1": [
        ("min_steps", 2000, "BC steps before the plateau test"),
        ("max_steps", 30000, "the step cap"),
        ("patience", 3, "held-out evaluations without an improvement of min_delta that end B1"),
        ("min_delta", 0.002, "the held-out agreement improvement that counts"),
        ("play_eval", True, "play an evaluation at the end of B1"),
    ],
    "b2": [
        ("rounds_min", 3, "DAgger rounds before the stop test"),
        ("rounds_max", 5, "the round cap (0: skip B2)"),
        ("teacher_mix_start", 0.5, "teacher_mix of the first round"),
        ("teacher_mix_end", 0.0, "teacher_mix of the last round (rounds_max); linear in between"),
        ("decisions_per_round", 1_000_000, "decisions played per round"),
        ("min_steps_per_round", 2000, "BC steps per round (the round waits for both)"),
        ("nn_options", "", "the playing network's nn:remote options (empty: temp 1 sampling)"),
        ("min_improvement", 0.01, "a round whose win rate vs scripted:hard is not this much above the best ends B2"),
    ],
    "rnad": [
        ("eta", 0.2, "the reward transform's η (live)"),
        ("learning_rate", 5e-5, "(live)"),
        ("entropy_schedule_sizes", [10000], "Δm: iteration sizes (live)"),
        ("entropy_schedule_repeats", [1], "their repeats (the last size repeats forever) (live)"),
        ("target_network_avg", 0.001, ""),
        ("clip_gradient", 10000.0, ""),
        ("nerd_beta", 2.0, ""),
        ("nerd_clip", 10000.0, ""),
        ("c_vtrace", 1.0, ""),
        ("chunk_len", 32, "time steps per chunk"),
        ("games_per_step", 32, "G: games per learner step (live)"),
        ("forward_tokens", 65536, "the most padded tokens (rows × tokens) of one forward of a chunk"),
        ("attention_budget", 1 << 26, "the most attention cells (rows × padded tokens², per head and layer) of one "
         "forward of a chunk; a chunk whose forwards hold more together checkpoints them"),
        ("checkpoint_tokens", 1 << 17, "a chunk whose forwards hold more padded tokens together checkpoints them: "
         "each forward's activations recomputed in the backward, one forward at a time (the same gradients, one "
         "more forward; a chunk of 1024 decisions of ~150 tokens peaks at ~7.6 GiB without, ~2.8 GiB with)"),
        ("precision", "fp16", "fp32, fp16 or bf16"),
        ("reg_eps", 0.1, "π_0,reg's smoothing ε (live)"),
        ("reg_eps_sideboarding", 0.25, "π_0,reg's smoothing ε at sideboarding steps; about 0 with a sideboarding "
         "teacher (actors.teacher_tier hard+sb: exploration belongs to the actors, plan §6.2) (live)"),
        ("stale_reg", "recompute", "recompute or drop"),
        ("finetune_threshold", 0.03, "B5's threshold"),
        ("finetune_discretization", 16, "B5's discretisation"),
        ("push_every", 50, "learner steps between pushes of the current policy to the server (live)"),
        ("report_every", 10, "learner steps per metrics line"),
        ("value_after_every", 10, "learner steps between the value diagnostic's after-update forwards (0: never) "
         "(live)"),
        ("max_policy_lag", 0, "drop a self-play game whose oldest step was played by a policy more than this many "
         "learner steps old (0: off; v-trace corrects the lag) (live)"),
        ("fresh_order", "newest", "which fresh self-play games a step takes first: oldest (every game in order) or "
         "newest (the least policy lag when the actors play more than the learner takes; the older ones wait, and "
         "rnad.max_policy_lag drops them) (live)"),
    ],
    "b4": [
        ("iterations", 0, "R-NaD iterations before B5 (0: stay in B4 until phase next / the step cap)"),
        ("max_learner_steps", 0, "learner steps before B5 (0: no cap)"),
        ("nn_options", "", "self-play nn:remote options (empty: temp 1 sampling) (live)"),
    ],
    "b5": [
        ("max_learner_steps", 0, "learner steps at which the run ends (0: never)"),
    ],
    "sb_explore": [
        ("p_start", 0.0, "p_explore, the probability that a self-play seat's sideboarding phase explores "
         "(docs/actor.md sb_explore), from B4's start until hold_frac of b4.max_learner_steps (live)"),
        ("p_end", 0.0, "p_explore at b4.max_learner_steps, linear from p_start after the hold (0 in B5, in other "
         "phases and in every evaluation) (live)"),
        ("hold_frac", 0.5, "the share of b4.max_learner_steps held at p_start (b4.max_learner_steps 0: p_start "
         "throughout B4) (live)"),
        ("count_weights", [1.0] * 11, "the weights of an exploring phase's swap count k = 0 ... 10 (live)"),
        ("coverage", True, "weight cards by coverage: a card's in- (out-) weight in a matchup is "
         "(1 + n) ** -coverage_power, n the explored phases of that matchup that moved it in (out); false: "
         "uniform (live)"),
        ("coverage_power", 1.0, "the coverage weights' power (0: uniform) (live)"),
        ("rewrite_every_s", 600.0, "seconds between the schedule's rewrites with the current counts (also when p "
         "moves by 0.01, the phase changes or a key here is set) (live)"),
    ],
    "eval": [
        ("evaluators", ["sprt", "fixed", "ladder"], "the evaluators of an evaluation (the registry's names: sprt, "
         "fixed, ladder, exploiter; sprt and exploiter run as evaluations of their own)"),
        ("opponents", ["scripted", "explore", "random"], "the fixed opponents (scripted = scripted:hard, "
         "whatever actors.teacher_tier is; scripted:<tier> another tier)"),
        ("pairs", copy.deepcopy(EVAL_PAIRS), "evaluation deck pairs (both decks and seats each)"),
        ("matches", 48, "matches per fixed opponent (live)"),
        ("ladder", 3, "past R-NaD fixed points in the ladder (plus the BC policy) (live)"),
        ("ladder_matches", 24, "matches per ladder opponent (live)"),
        ("actors", 4, "evaluation actors (live)"),
        ("every_minutes", 60.0, "B4/B5 evaluation cadence in minutes (0: off) (live)"),
        ("per_iteration", True, "evaluate at every R-NaD iteration boundary (live)"),
        ("nn_options_bc", "argmax", "the evaluated policy's options in B1–B2 (the primary result)"),
        ("nn_options_rnad", "argmax", "in B4 (empty: temp 1 sampling)"),
        ("nn_options_finetune", "thresh=0.03:disc=16", "in B5 (the fine-tuning projection)"),
        ("also_sampled", False, "the fixed evaluator also plays the policy with the actors' sampling options "
         "(b2.nn_options in B1–B2, b4.nn_options in B4–B5): groups <opponent>@sampled, a secondary result (live)"),
        ("opponent_nn_options", "argmax", "ladder opponents' options"),
        ("policy_eval", "local", "how the evaluation actors play the evaluated policy and the ladder rungs: remote "
         "(server policies) or local (nn:<export> in process: each one exported first); the exploiter is always "
         "remote"),
        ("sprt_opponents", ["scripted"], "the sprt evaluator's opponents (scripted = scripted:hard)"),
        ("sprt_pairs", [], "its deck pairs ([[a, b], ...]; empty: every league pair, both decks and seats each)"),
        ("sprt_p0", 0.45, "H0: the per-game win rate p0"),
        ("sprt_p1", 0.55, "H1: p1 (> p0)"),
        ("sprt_alpha", 0.05, "the test's error rates"),
        ("sprt_beta", 0.05, ""),
        ("sprt_max_games", 400, "games per opponent at which the test stops undecided (0: no sprt) (live)"),
        ("exploiter_every_minutes", 0.0, "B4/B5 exploiter cadence in minutes (0: off; `rmtrain eval --evaluators "
         "exploiter` runs one) (live)"),
        ("exploiter_minutes", 20.0, "an exploiter's training time (live)"),
        ("exploiter_actors", 8, "its actors (beside the training actors)"),
        ("exploiter_pause_learner", True, "the learner idles while the exploiter trains (one GPU process: they "
         "would otherwise share the GPU)"),
        ("exploiter_baseline_games", 100, "games of the initial exploiter (= the policy) before its first update"),
        ("exploiter_final_games", 100, "games of the frozen final exploiter: its reported win rate"),
        ("exploiter_pairs", [], "its deck pairs (empty: eval.pairs)"),
        ("exploiter_victim_options", "", "the victim's nn: options (empty: temperature 1; argmax)"),
        ("exploiter_lr", 3e-5, "the exploiter's PPO learning rate"),
        ("exploiter_rollout_decisions", 4096, "exploiter decisions per PPO update"),
    ],
    "checkpoint": [
        ("every_minutes", 15.0, "checkpoint cadence (live)"),
        ("keep", 3, "periodic checkpoints kept (phase-end checkpoints and fixed points are kept)"),
        ("export_formats", ["ts", "pt2"], "the exports written at phase ends and fixed points (ts, pt2, onnx)"),
        ("export", True, "write exports"),
    ],
}

CHOICES = {
    "league.format": ("bo1", "bo3"),
    "league.knowledge": ("off", "expert"),
    "server.precision": ("fp32", "fp16", "bf16"),
    "bc.amp": ("fp16", "bf16", "off"),
    "rnad.precision": ("fp32", "fp16", "bf16"),
    "rnad.stale_reg": ("recompute", "drop"),
    "rnad.fresh_order": ("oldest", "newest"),
    "actors.policy_eval": ("remote", "local"),
    "actors.export_format": ("ts", "pt2"),
    "eval.policy_eval": ("remote", "local"),
}

LIVE = {
    "bc.lr", "bc.report_every", "bc.eval_every",
    "b0.decisions", "b1.min_steps", "b1.max_steps", "b1.patience", "b1.min_delta",
    "b2.rounds_min", "b2.rounds_max", "b2.decisions_per_round", "b2.min_steps_per_round", "b2.min_improvement",
    "rnad.eta", "rnad.learning_rate", "rnad.entropy_schedule_sizes", "rnad.entropy_schedule_repeats",
    "rnad.reg_eps", "rnad.reg_eps_sideboarding", "rnad.push_every", "rnad.report_every", "rnad.games_per_step",
    "rnad.value_after_every", "rnad.max_policy_lag", "rnad.fresh_order", "actors.export_every_s",
    "actors.export_every_steps", "actors.local_count", "b4.iterations", "b4.max_learner_steps", "b4.nn_options", "b5.max_learner_steps",
    "actors.count", "eval.matches", "eval.ladder", "eval.ladder_matches", "eval.actors", "eval.every_minutes",
    "eval.per_iteration", "eval.also_sampled", "eval.sprt_max_games", "eval.exploiter_every_minutes",
    "eval.exploiter_minutes", "checkpoint.every_minutes", "sb_explore.p_start", "sb_explore.p_end",
    "sb_explore.hold_frac", "sb_explore.count_weights", "sb_explore.coverage", "sb_explore.coverage_power",
    "sb_explore.rewrite_every_s", "replay.max_decision_tokens", "league.decks", "league.pairs",
}

# Live keys the GPU process applies (the others are the orchestrator's).
GPU_LIVE = {"bc.lr", "bc.report_every", "bc.eval_every", "rnad.eta", "rnad.learning_rate",
            "rnad.entropy_schedule_sizes", "rnad.entropy_schedule_repeats", "rnad.reg_eps",
            "rnad.reg_eps_sideboarding", "rnad.push_every", "rnad.report_every", "rnad.games_per_step",
            "rnad.value_after_every", "rnad.max_policy_lag", "rnad.fresh_order", "actors.export_every_s",
            "actors.export_every_steps", "replay.max_decision_tokens"}


def defaults():
    return {sec: {k: copy.deepcopy(v) for k, v, _ in keys} for sec, keys in SPEC.items()}


def load_file(path):
    """A config file's dict: TOML (``.toml``) or JSON."""
    with open(path, "rb") as fp:
        data = fp.read()
    if path.endswith(".json"):
        return json.loads(data)
    return tomllib.loads(data.decode())


def parse_value(text):
    """A command-line value: JSON when it parses (numbers, true/false, lists, quoted strings), else
    the text itself."""
    try:
        return json.loads(text)
    except ValueError:
        low = text.strip().lower()
        if low in ("true", "false"):
            return low == "true"
        return text


def _default_of(key):
    sec, _, name = key.partition(".")
    for k, v, _ in SPEC.get(sec, []):
        if k == name:
            return v
    raise ValueError("no config key %s" % key)


def coerce(key, value):
    """`value` as `key`'s type (ValueError when it does not fit)."""
    d = _default_of(key)
    if isinstance(d, bool):
        if isinstance(value, bool):
            return value
        raise ValueError("%s is a boolean, not %r" % (key, value))
    if isinstance(d, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != int(value):
            raise ValueError("%s is an integer, not %r" % (key, value))
        return int(value)
    if isinstance(d, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("%s is a number, not %r" % (key, value))
        return float(value)
    if isinstance(d, str):
        if not isinstance(value, str):
            raise ValueError("%s is a string, not %r" % (key, value))
        return value
    if isinstance(d, list):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and d and isinstance(d[0], int):
            return [int(value)]
        if not isinstance(value, list):
            raise ValueError("%s is a list, not %r" % (key, value))
        return value
    return value


def merge(cfg, overlay, where="config"):
    """`overlay` ({section: {key: value}}) over `cfg`, every key checked and coerced."""
    out = copy.deepcopy(cfg)
    for sec, keys in overlay.items():
        if sec not in SPEC:
            raise ValueError("%s: no config section [%s]" % (where, sec))
        if not isinstance(keys, dict):
            raise ValueError("%s: [%s] is not a table" % (where, sec))
        for k, v in keys.items():
            out[sec][k] = coerce("%s.%s" % (sec, k), v)
    return out


def apply_sets(cfg, sets):
    """``section.key=value`` overrides over `cfg`."""
    overlay = {}
    for s in sets or []:
        key, eq, text = s.partition("=")
        if not eq or "." not in key:
            raise ValueError("--set takes section.key=value, not %r" % s)
        sec, _, name = key.strip().partition(".")
        overlay.setdefault(sec, {})[name] = parse_value(text)
    return merge(cfg, overlay, "--set")


def resolve(file_cfg=None, sets=None):
    cfg = defaults()
    if file_cfg:
        cfg = merge(cfg, file_cfg)
    cfg = apply_sets(cfg, sets)
    validate(cfg)
    return cfg


def absolute_paths(cfg):
    """`cfg` with ``run.tables`` and ``run.start_from`` absolute: a relative path is relative to the
    repository, whatever the working directory of the process that reads the config."""
    from .rundir import REPO_ROOT
    for key in ("tables", "start_from"):
        v = cfg["run"].get(key)
        if v and not os.path.isabs(v):
            cfg["run"][key] = os.path.normpath(os.path.join(REPO_ROOT, v))
    return cfg


def fill_defaults(cfg):
    """A resumed run's config with the keys added since it was created set to their defaults:
    (config, the added keys)."""
    out = copy.deepcopy(cfg)
    added = []
    for sec, keys in SPEC.items():
        for k, default, _ in keys:
            if k not in out.setdefault(sec, {}):
                out[sec][k] = copy.deepcopy(default)
                added.append("%s.%s" % (sec, k))
    return out, added


def validate(cfg):
    """ValueError naming the first bad key."""
    for sec, keys in SPEC.items():
        for k, _, _ in keys:
            if k not in cfg.get(sec, {}):
                raise ValueError("config: %s.%s is missing" % (sec, k))
            coerce("%s.%s" % (sec, k), cfg[sec][k])
    for key, choices in CHOICES.items():
        sec, _, k = key.partition(".")
        if cfg[sec][k] not in choices:
            raise ValueError("%s is one of %s, not %r" % (key, ", ".join(choices), cfg[sec][k]))

    def need(cond, msg):
        if not cond:
            raise ValueError(msg)

    lg = cfg["league"]
    need(all(isinstance(d, str) and d for d in lg["decks"]), "league.decks are deck stems")
    need(len(lg["decks"]) >= 1, "league.decks is empty")
    for p in lg["pairs"]:
        need(isinstance(p, list) and len(p) in (2, 3) and all(isinstance(x, str) for x in p[:2])
             and (len(p) == 2 or (isinstance(p[2], (int, float)) and p[2] >= 0)),
             "league.pairs entries are [deck_a, deck_b] or [deck_a, deck_b, weight], not %r" % (p,))
    need(not lg["pairs"] or len(lg["decks"]) >= 1, "league.pairs needs decks")
    need(lg["pairs"] or len(lg["decks"]) >= 2, "league.decks needs two decks (or league.pairs)")
    held = [d for d in lg["decks"] + [x for p in lg["pairs"] for x in p[:2]] if d.startswith("heldout/")]
    need(not held, "league: %s is held out (decks/heldout/: never trained on)" % (held[0] if held else ""))
    if lg["pairs"]:
        paired = {x for p in lg["pairs"] for x in p[:2]}
        need(not paired - set(lg["decks"]), "league.pairs names %s, not in league.decks"
             % ", ".join(sorted(paired - set(lg["decks"]))))
        need(not set(lg["decks"]) - paired, "league.decks' %s is in no league.pairs entry, so it would never be "
             "played (`rmtrain add-decks` adds a deck with its pairs)" % ", ".join(sorted(set(lg["decks"]) - paired)))
    need(lg["max_game_decisions"] >= 0, "league.max_game_decisions >= 0")
    need(cfg["run"]["crash_loop_fatals"] >= 0, "run.crash_loop_fatals >= 0")
    need(cfg["actors"]["count"] >= 0 and cfg["actors"]["local_count"] >= 0, "actors.count, local_count >= 0")
    need(re.fullmatch(r"[a-z][a-z0-9+:_-]*", cfg["actors"]["teacher_tier"]) is not None,
         "actors.teacher_tier is a scripted tier name (hard, hard+sb, ...), not %r" % cfg["actors"]["teacher_tier"])
    m = cfg["model"]
    need(m["d"] > 0 and m["heads"] > 0 and m["d"] % m["heads"] == 0, "model.d is a positive multiple of model.heads")
    need(m["layers"] > 0, "model.layers > 0")
    need(0 <= m["id_dropout"] < 1, "model.id_dropout in [0, 1)")
    need(cfg["replay"]["max_decisions"] > 0 and cfg["replay"]["max_gb"] > 0, "replay limits are positive")
    need(0 <= cfg["replay"]["heldout_fraction"] < 1, "replay.heldout_fraction in [0, 1)")
    need(cfg["replay"]["max_decision_tokens"] >= 0, "replay.max_decision_tokens >= 0")
    need(cfg["b0"]["decisions"] >= 0, "b0.decisions >= 0")
    need(0 <= cfg["b0"]["explore_frac"] < 1, "b0.explore_frac in [0, 1)")
    need(cfg["bc"]["lr"] > 0 and cfg["bc"]["batch_size"] > 0, "bc.lr and bc.batch_size are positive")
    need(cfg["bc"]["sideboard_weight"] >= 0, "bc.sideboard_weight >= 0")
    need(cfg["bc"]["unit_steps"] > 0 and cfg["bc"]["report_every"] > 0 and cfg["bc"]["eval_every"] > 0,
         "bc.unit_steps, report_every, eval_every are positive")
    b1 = cfg["b1"]
    need(0 <= b1["min_steps"] <= b1["max_steps"] and b1["max_steps"] > 0, "0 <= b1.min_steps <= b1.max_steps")
    need(b1["patience"] >= 1, "b1.patience >= 1")
    b2 = cfg["b2"]
    need(0 <= b2["rounds_min"] <= max(b2["rounds_max"], 0), "0 <= b2.rounds_min <= b2.rounds_max")
    for k in ("teacher_mix_start", "teacher_mix_end"):
        need(0 <= b2[k] <= 1, "b2.%s in [0, 1]" % k)
    need(b2["decisions_per_round"] > 0, "b2.decisions_per_round > 0")
    r = cfg["rnad"]
    need(r["eta"] >= 0 and r["learning_rate"] > 0, "rnad.eta >= 0, rnad.learning_rate > 0")
    sizes, reps = r["entropy_schedule_sizes"], r["entropy_schedule_repeats"]
    need(sizes and all(isinstance(x, int) and x > 0 for x in sizes), "rnad.entropy_schedule_sizes are positive ints")
    need(len(reps) == len(sizes) and all(isinstance(x, int) and x > 0 for x in reps),
         "rnad.entropy_schedule_repeats has one positive int per size")
    need(0 <= r["reg_eps"] <= 1 and 0 <= r["reg_eps_sideboarding"] <= 1, "rnad.reg_eps* in [0, 1]")
    need(r["chunk_len"] > 0 and r["games_per_step"] > 0, "rnad.chunk_len and games_per_step are positive")
    need(r["forward_tokens"] > 0 and r["attention_budget"] > 0 and r["checkpoint_tokens"] > 0 and
         cfg["server"]["attention_budget"] > 0,
         "rnad.forward_tokens, attention_budget, checkpoint_tokens and server.attention_budget are positive")
    need(r["push_every"] >= 0 and r["report_every"] > 0, "rnad.push_every >= 0, report_every > 0")
    need(r["value_after_every"] >= 0, "rnad.value_after_every >= 0")
    need(r["max_policy_lag"] >= 0, "rnad.max_policy_lag >= 0")
    a = cfg["actors"]
    need(a["export_every_s"] > 0 and a["export_every_steps"] >= 0, "actors.export_every_s > 0, export_every_steps >= 0")
    need(0 <= r["finetune_threshold"] < 1 and r["finetune_discretization"] >= 0, "rnad.finetune_* ranges")
    e = cfg["eval"]
    need(all(x in ("fixed", "ladder") or isinstance(x, str) for x in e["evaluators"]), "eval.evaluators are names")
    need(e["actors"] >= 1 and e["matches"] >= 0 and e["ladder_matches"] >= 0, "eval.actors >= 1, matches >= 0")
    need(all(isinstance(p, list) and len(p) == 2 for p in e["pairs"]) and e["pairs"], "eval.pairs are [a, b] pairs")
    need(e["every_minutes"] >= 0, "eval.every_minutes >= 0")
    need(0 < e["sprt_p0"] < e["sprt_p1"] < 1, "eval.sprt_p0 < eval.sprt_p1, both in (0, 1)")
    need(0 < e["sprt_alpha"] < 0.5 and 0 < e["sprt_beta"] < 0.5, "eval.sprt_alpha, sprt_beta in (0, 0.5)")
    need(e["sprt_max_games"] >= 0, "eval.sprt_max_games >= 0")
    need(e["exploiter_every_minutes"] >= 0 and e["exploiter_minutes"] >= 0 and e["exploiter_actors"] >= 1,
         "eval.exploiter_every_minutes, exploiter_minutes >= 0, exploiter_actors >= 1")
    for k in ("sprt_pairs", "exploiter_pairs"):
        need(all(isinstance(p, list) and len(p) == 2 for p in e[k]), "eval.%s are [a, b] pairs" % k)
    sx = cfg["sb_explore"]
    need(0 <= sx["p_start"] <= 1 and 0 <= sx["p_end"] <= 1, "sb_explore.p_start, p_end in [0, 1]")
    need(0 <= sx["hold_frac"] <= 1, "sb_explore.hold_frac in [0, 1]")
    cw = sx["count_weights"]
    need(len(cw) == 11 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 for x in cw)
         and any(x > 0 for x in cw), "sb_explore.count_weights: 11 numbers >= 0 (k = 0 ... 10), not all 0")
    need(sx["coverage_power"] >= 0, "sb_explore.coverage_power >= 0")
    need(sx["rewrite_every_s"] > 0, "sb_explore.rewrite_every_s > 0")
    c = cfg["checkpoint"]
    need(c["every_minutes"] > 0 and c["keep"] >= 1, "checkpoint.every_minutes > 0, keep >= 1")
    need(all(f in ("ts", "pt2", "onnx") for f in c["export_formats"]), "checkpoint.export_formats: ts, pt2, onnx")
    return cfg


def parse_live(key, text):
    """(key, value) of ``rmtrain set key value``; ValueError when the key is unknown, not live or the
    value does not fit."""
    if "." not in key:
        raise ValueError("set takes section.key (e.g. rnad.eta)")
    _default_of(key)
    if key not in LIVE:
        raise ValueError("%s is not settable while running (live keys: %s); use start --set on a stopped run"
                         % (key, ", ".join(sorted(LIVE))))
    return key, coerce(key, parse_value(text))


def set_key(cfg, key, value):
    return set_keys(cfg, {key: value})


def set_keys(cfg, changes):
    """`cfg` with every ``section.key`` of `changes` set (coerced), validated as one config."""
    out = copy.deepcopy(cfg)
    for key, value in changes.items():
        sec, _, k = key.partition(".")
        out[sec][k] = coerce(key, value)
    validate(out)
    return out


def get_key(cfg, key):
    sec, _, k = key.partition(".")
    return cfg[sec][k]


def _toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(v)


def to_toml(cfg, docs=True):
    """`cfg` as TOML (with each key's documentation as a comment)."""
    lines = []
    for sec, keys in SPEC.items():
        lines.append("[%s]" % sec)
        for k, _, doc in keys:
            live = " (live)" if "%s.%s" % (sec, k) in LIVE and "(live)" not in doc else ""
            text = "%s = %s" % (k, _toml_value(cfg[sec][k]))
            if docs and (doc or live):
                text = "%-44s # %s%s" % (text, doc, live)
            lines.append(text)
        lines.append("")
    return "\n".join(lines)


def defaults_toml():
    return to_toml(defaults())

