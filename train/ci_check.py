#!/usr/bin/env python3
"""Standardized engine test gate — the single command CI and developers run.

Replaces the disparate ad-hoc harness invocations (test_harness scenarios,
fuzz_campaign transcript dumps, replay_diff, engine-sanity-check greps) with one
entry point that runs a fixed set of tiers and exits nonzero on any finding.
``make check`` wraps this; CI runs the same command.

The standard it enforces: every card in the league/ decks behaves properly —
0 missing vocab entries, deterministic games complete, no engine crashes, no
non-fatal errors / anomalies in the transcripts.

Tiers (run cheap-to-expensive; all requested tiers run even if an earlier one
fails, so one invocation reports every finding):

  pygen   Regenerate train/_enums.py + train/card_costs.py and fail if the
          committed copies are stale (someone changed a C++ input without
          committing the regenerated Python).
  vocab   Every card referenced by the top-level and league/ decks resolves to a
          card_vocab.h entry (result-level league-coverage gate). DFC deck names
          resolve through their script's front face, mirroring the engine.
  obsinv  Structural per-decision invariants on the raw machine-mode observation
          vector across a few seeded scripted games (train/test_obs_invariants.py):
          card-id / entity-ref floats decode in range, recency-packed zones have
          no holes, one-hots are one-hot, etc. Catches a silent serialize_state
          layout/encoding regression.
  snapshot The --search-server snapshot/restore/determinize protocol behaves:
          round-trip byte identity, RNG isolation, determinize efficacy, the
          SIM_RESULT terminal intercept, and determinize invariants
          (train/test_snapshot.py). Catches a regression in the in-process
          MCTS search primitives.
  sbselfplay The AZ self-play collector turns bo3 sideboard search roots into
          samples priced by the UPCOMING game's winner: one small bo3 match
          through the real _play_match, then assert a sideboard sample exists,
          gates the next game, and its z is +/-1 per that game's result
          (train/test_sideboard_selfplay.py). Torch-free; needs bin/robomage.
  mirror  World-parallel mirror-pool search for interactive play: a plain
          run_search and a run_search_parallel over primary+mirror (worlds split
          across processes) merge to bit-identical visits; a mirror stays in
          lockstep through a full bo3; a desynced mirror disables the pool and
          the primary plays on (train/test_mirror_search.py). Torch-free.
  replay  The byte-identical replay-diff corpus (delver/doomsday/mav) still
          matches — catches unintended narrative/behavior drift.
  smoke   Deterministic league games with the scripted *hard* agent (realistic
          lines) across the league mirrors + a ring of crosses. Fixed seeds.
  fuzz    Short random coverage fuzz (the 'explore' agent) over the league
          mirrors. Fixed seeds on PR (a red run is the diff's fault); the nightly
          workflow rotates the seed and widens to every matchup.

Opt-in tiers (valid for --tier, NOT part of the default run):

  actor   Phase-D AZ actor gate: obs bit-parity (test_actor_parity.py), MCTS
          visit-count parity (test_mcts_parity.py), and self-play shard schema /
          trainer ingest (test_actor_shards.py). Self-skips with a message when
          bin/az_actor is not built or torch is unavailable.
  analysis The GUI analysis window's engine core: chunked IncrementalSearch is
          bit-identical to run_search (pinned world seeds), pv/walk browse and
          replay lines then hand the engine back clean (driver discipline), an
          AnalysisSession's detached mirror stays in lockstep across a bo3
          sideboard boundary by delta replay, and a cross-thread stop event
          cancels within one chunk (train/test_analysis_session.py).
          Torch-free; needs bin/robomage.

Draw classification (per repo policy — draws are not acceptable, but the two
causes differ in severity):
  * A game that ends with no winner because the engine hit its internal step cap
    (a stall) is a WARNING — flagged for review, does not fail the gate.
  * A game whose engine process crashed (nonzero exit / EOF mid-game — surfaced
    as an exception from run_games) is an ERROR — fails the gate.
  * A standalone non-fatal 'ERROR:' / 'FATAL:' / assert / etc. line in any
    transcript is an ERROR regardless of the game's outcome.

Exit code: 1 if any tier reported an error, else 0 (warnings alone pass, with a
WARNINGS summary line). Transcripts and relocated draw logs land in --out-dir for
artifact upload.
"""
import argparse
import contextlib
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runner
import missing_cards
from opponents import make_controller

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DECKS_ROOT = os.path.join(_REPO_ROOT, "bin", "resources", "decks")

# The PFSP league roster — the decks whose cards must all behave properly.
# Discovered from the .dk files actually present under decks/league/, so a
# deck removed from (or added to) the roster doesn't require an edit here.
# Deck specs are given as the engine expects them (relative to decks/, so
# "league/x").
LEAGUE = sorted(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(os.path.join(_DECKS_ROOT, "league", "*.dk"))
)
LEAGUE_SPECS = [f"league/{d}" for d in LEAGUE]

ALL_TIERS = ["pygen", "vocab", "obsinv", "snapshot", "sbselfplay", "mirror",
             "replay", "smoke", "fuzz"]

# Opt-in tiers: valid for --tier but NOT part of the default run. `actor` gates
# the Phase-D AZ actor (bin/az_actor) — it needs the actor binary + torch, and
# self-skips cleanly when either is absent (so it never breaks a stock build).
OPT_IN_TIERS = ["actor", "analysis"]
KNOWN_TIERS = ALL_TIERS + OPT_IN_TIERS

# Transcript scan (stdout narrative + captured engine stderr). Two severities:
#   ERROR  — genuine failures that fail the gate: the engine's own ERROR:
#            (non_fatal_error) / FATAL: markers, a C/C++ assert or crash, or a
#            Python traceback. Per CLAUDE.md a non-fatal ERROR is not acceptable.
#   WARNING — flagged for review but non-blocking: an engine WARNING: line OTHER
#            than the routine "Unrecognized ability param" parse notice. That
#            notice fires in bulk for cosmetic / AI-hint params on common cards
#            (Brainstorm's Reorder$, Green Sun's Zenith's AIXMax$, ...) that do
#            not affect the modeled rules, so — matching the engine-sanity-check
#            skill, which counts them for periodic review rather than acting per
#            run — it is left in the transcript (greppable) but not surfaced, so a
#            reported warning stays meaningful. A genuinely new unimplemented
#            mechanic still shows up in the transcript for bulk review.
_ERROR_RE = re.compile(
    r"^(ERROR|FATAL):|Traceback \(most recent call last\)|Segmentation fault|"
    r"Assertion .* failed|\bAborted\b|core dumped")
_WARNING_RE = re.compile(r"^WARNING:")
_WARNING_IGNORE_RE = re.compile(r"Unrecognized ability param")


def scan_transcript(path):
    """Return (errors, warnings), each a list of (lineno, line), for a transcript."""
    errors, warnings = [], []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f, 1):
                s = line.rstrip("\n")
                if _ERROR_RE.search(s):
                    errors.append((n, s))
                elif _WARNING_RE.search(s) and not _WARNING_IGNORE_RE.search(s):
                    warnings.append((n, s))
    except OSError as e:
        errors.append((0, f"(could not read transcript: {e})"))
    return errors, warnings


def _matchups(preset):
    """Resolve a matchup preset to a list of (deck_a_spec, deck_b_spec) pairs."""
    n = len(LEAGUE_SPECS)
    mirrors = [(d, d) for d in LEAGUE_SPECS]
    ring = [(LEAGUE_SPECS[i], LEAGUE_SPECS[(i + 1) % n]) for i in range(n)]
    if preset == "mirrors":
        return mirrors
    if preset == "ring":
        return ring
    if preset == "mirrors+ring":
        return mirrors + ring
    if preset == "all":
        return [(a, b) for a in LEAGUE_SPECS for b in LEAGUE_SPECS]
    # Explicit "a:b,c:d" list.
    pairs = []
    for item in preset.split(","):
        item = item.strip()
        if not item:
            continue
        a, _, b = item.partition(":")
        pairs.append((a.strip(), b.strip() or a.strip()))
    return pairs


def _short(spec):
    """Bare deck stem for filenames/labels (drop the 'league/' prefix)."""
    return spec.rsplit("/", 1)[-1]


class Report:
    """Collects errors and warnings across tiers and prints a final summary."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def error(self, tier, msg):
        self.errors.append((tier, msg))
        print(f"  [ERROR] {tier}: {msg}", flush=True)

    def warn(self, tier, msg):
        self.warnings.append((tier, msg))
        print(f"  [WARN]  {tier}: {msg}", flush=True)


# ── Tiers ───────────────────────────────────────────────────────────────────

def tier_pygen(rep):
    """Regenerate the codegen outputs and fail if the committed copies are stale.

    Both are deterministic given the committed C++ inputs AND a fully provisioned
    card set: _enums.py derives from action.h/game.h; card_costs.py additionally
    reads each vocab card's ManaCost from its script — and provision_decks.py
    fetches the whole vocab, so every cost is reproducible here and in CI. The
    working tree is restored afterward so a stale result is reported, not left
    half-regenerated (the developer runs `make pygen` to actually update them)."""
    gen_files = ["train/_enums.py", "train/card_costs.py",
                 "src/gen/card_costs_gen.h"]
    for gen in ("train/gen_enums.py", "train/gen_card_costs.py"):
        r = subprocess.run([sys.executable, gen], cwd=_REPO_ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            rep.error("pygen", f"{gen} failed: {r.stderr.strip()}")
            subprocess.run(["git", "checkout", "--", *gen_files], cwd=_REPO_ROOT,
                           capture_output=True)
            return
    diff = subprocess.run(["git", "diff", "--", *gen_files], cwd=_REPO_ROOT,
                          capture_output=True, text=True)
    # Restore the committed copies regardless — the tier only reports staleness.
    subprocess.run(["git", "checkout", "--", *gen_files], cwd=_REPO_ROOT,
                   capture_output=True)
    if diff.stdout.strip():
        rep.error("pygen",
                  "generated files are stale — run `make pygen` and commit "
                  f"{', '.join(gen_files)} (ensure the full card set is provisioned "
                  f"first: tools/forge_fetch/provision_decks.py):\n{diff.stdout}")


def tier_vocab(rep):
    """Every top-level + league deck card must resolve to a vocab entry."""
    vocab, _ = missing_cards.parse_vocab(missing_cards.VOCAB_H)
    for label, decks_dir in (("top-level", _DECKS_ROOT),
                             ("league", os.path.join(_DECKS_ROOT, "league"))):
        refs, _ = missing_cards.collect(decks_dir)
        missing = [r["name"] for uid, r in sorted(refs.items())
                   if missing_cards.vocab_identity(uid, vocab) is None]
        if missing:
            rep.error("vocab", f"{label} decks reference {len(missing)} card(s) "
                               f"not in card_vocab.h: {', '.join(missing)}")


def tier_obsinv(rep):
    """Structural invariants on the raw machine-mode observation vector.

    Drives a few seeded scripted games and asserts per-decision that the
    observation encoding stays self-consistent (see train/test_obs_invariants.py).
    A cheap gate that catches a silent serialize_state regression."""
    r = subprocess.run([sys.executable, "train/test_obs_invariants.py"],
                       cwd=_REPO_ROOT, capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        rep.error("obsinv", "observation invariant violation "
                            f"(test_obs_invariants.py exit {r.returncode}):\n"
                            f"{r.stdout}{r.stderr}")


def tier_snapshot(rep):
    """The --search-server snapshot/restore/determinize protocol regression.

    Drives the raw machine protocol over a subprocess and asserts the search
    primitives' guarantees (round-trip byte identity, RNG isolation, determinize
    efficacy/invariants, terminal intercept — see train/test_snapshot.py). A gate
    on the in-process MCTS search subsystem."""
    r = subprocess.run([sys.executable, "train/test_snapshot.py"],
                       cwd=_REPO_ROOT, capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        rep.error("snapshot", "search-server protocol violation "
                             f"(test_snapshot.py exit {r.returncode}):\n"
                             f"{r.stdout}{r.stderr}")


def tier_sbselfplay(rep):
    """AZ self-play persists searched bo3 sideboard samples priced by next game.

    Runs one small bo3 match through the real self-play collector and asserts the
    sideboard-decision data path (see train/test_sideboard_selfplay.py). Torch-free
    and quick; gates the Stage-4 sideboard-search plumbing."""
    r = subprocess.run([sys.executable, "train/test_sideboard_selfplay.py"],
                       cwd=_REPO_ROOT, capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        rep.error("sbselfplay", "sideboard self-play data-path violation "
                                f"(test_sideboard_selfplay.py exit {r.returncode}):\n"
                                f"{r.stdout}{r.stderr}")


def tier_mirror(rep):
    """World-parallel mirror-pool search regression (interactive play only).

    Runs train/test_mirror_search.py: bit-exact single-vs-parallel visit merge,
    lockstep across a bo3, and graceful pool-disable on mirror drift. Torch-free
    and quick; needs bin/robomage."""
    r = subprocess.run([sys.executable, "train/test_mirror_search.py"],
                       cwd=_REPO_ROOT, capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        rep.error("mirror", "mirror-pool search violation "
                            f"(test_mirror_search.py exit {r.returncode}):\n"
                            f"{r.stdout}{r.stderr}")


def tier_analysis(rep):
    """Analysis-window engine core regression (opt-in; interactive play only).

    Runs train/test_analysis_session.py: chunked-vs-plain search bit-parity,
    pv/walk driver discipline, AnalysisSession detached-mirror lockstep across
    a bo3 sideboard boundary, and cancellation. Torch-free; needs bin/robomage."""
    r = subprocess.run([sys.executable, "train/test_analysis_session.py"],
                       cwd=_REPO_ROOT, capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        rep.error("analysis", "analysis-session violation "
                              f"(test_analysis_session.py exit {r.returncode}):\n"
                              f"{r.stdout}{r.stderr}")


def tier_replay(rep):
    """Run the byte-identical replay-diff corpus check."""
    r = subprocess.run([sys.executable, "train/regression/replay_diff.py", "check"],
                       cwd=_REPO_ROOT, capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.returncode != 0:
        rep.error("replay", f"corpus drift (replay_diff check exit {r.returncode}); "
                            "re-record intentionally with `replay_diff.py record` "
                            "if the behavior change is expected")


def _run_capturing(out_path, run_fn):
    """Run run_fn() capturing both Python stdout AND the engine subprocess's
    stderr (fd 2, which NarrativeEnv inherits) into out_path. Returns whatever
    run_fn returns; a raised exception propagates after capture is torn down."""
    with open(out_path, "w") as fh:
        # Engine stderr goes to a temp file via an fd-2 dup, then is appended to
        # the transcript (fd-level and Python-level writes to one file would race
        # on the offset, so keep them separate and merge afterward).
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errf:
            saved_fd = os.dup(2)
            os.dup2(errf.fileno(), 2)
            try:
                with contextlib.redirect_stdout(fh):
                    return run_fn()
            finally:
                sys.stderr.flush()
                os.dup2(saved_fd, 2)
                os.close(saved_fd)
                errf.seek(0)
                data = errf.read()
                if data.strip():
                    fh.write("\n=== engine stderr ===\n")
                    fh.write(data)


def _run_matchups(rep, tier, pairs, mode, n_games, base_seed, out_dir):
    """Run each matchup as scripted games, classify draws, scan transcripts."""
    for k, (deck_a, deck_b) in enumerate(pairs):
        seed = base_seed + 1000 * k
        out_path = os.path.join(out_dir, f"{tier}_{_short(deck_a)}__{_short(deck_b)}.txt")
        result = {}

        def run_fn():
            result["wld"] = runner.run_games(
                make_controller(mode), make_controller(mode),
                label_a=f"A:{mode}", label_b=f"B:{mode}",
                deck_a=deck_a, deck_b=deck_b, n_games=n_games, seed=seed,
                verbose=True)

        crashed = None
        try:
            _run_capturing(out_path, run_fn)
        except Exception as e:  # engine crash: nonzero exit / EOF mid-game
            crashed = e
        wins, losses, draws = result.get("wld", (0, 0, 0))
        # Relocate any draw_<n>.txt run_games wrote into cwd, so they're captured
        # as artifacts and don't litter the tree.
        for dl in glob.glob(os.path.join(os.getcwd(), "draw_*.txt")):
            shutil.move(dl, os.path.join(out_dir, f"{tier}_{_short(deck_a)}__"
                                                  f"{_short(deck_b)}_{os.path.basename(dl)}"))

        label = f"{_short(deck_a)} vs {_short(deck_b)} [{mode}] seed {seed}"
        if crashed is not None:
            rep.error(tier, f"{label}: engine crashed — {crashed} (see {out_path})")
            print(f"  {label}: CRASH -> {out_path}", flush=True)
            continue
        if wins + losses + draws < n_games:
            rep.error(tier, f"{label}: only {wins + losses + draws}/{n_games} games "
                           f"completed (see {out_path})")
        if draws > 0:
            # A returned draw is a clean-exit stall (a crash would have raised).
            rep.warn(tier, f"{label}: {draws} step-cap-stall draw(s) — review {out_path}")
        errs, warns = scan_transcript(out_path)
        if errs:
            first = errs[0]
            rep.error(tier, f"{label}: {len(errs)} error line(s) in transcript, first "
                           f"at {out_path}:{first[0]}: {first[1]!r}")
        if warns:
            rep.warn(tier, f"{label}: {len(warns)} engine WARNING line(s) — review "
                          f"{out_path} (e.g. {out_path}:{warns[0][0]}: {warns[0][1]!r})")
        print(f"  {label}: {wins}W / {losses}L / {draws}D -> {out_path}", flush=True)


def tier_smoke(rep, args, out_dir):
    """Deterministic league smoke with the scripted hard agent."""
    pairs = _matchups(args.matchups or "mirrors+ring")
    _run_matchups(rep, "smoke", pairs, "scripted", args.smoke_games, args.seed, out_dir)


def tier_fuzz(rep, args, out_dir):
    """Short fixed-seed coverage fuzz over league matchups."""
    pairs = _matchups(args.matchups or "mirrors")
    _run_matchups(rep, "fuzz", pairs, args.mode, args.fuzz_games, args.seed, out_dir)


def tier_actor(rep):
    """Phase-D AZ actor gate (opt-in). Self-skips when bin/az_actor is missing or
    torch is unavailable; otherwise runs the actor/MCTS/shard parity tests as
    subprocesses and reports each PASS/FAIL through the Report."""
    actor_bin = os.path.join(_REPO_ROOT, "bin", "az_actor")
    if not os.path.exists(actor_bin):
        print(f"  [skip] actor: {actor_bin} not built "
              "(build with `make actor`)", flush=True)
        return
    try:
        import torch  # noqa: F401
    except Exception as e:
        print(f"  [skip] actor: torch not importable ({e})", flush=True)
        return

    tests = ["train/test_actor_parity.py",   # M5: obs bit-parity
             "train/test_mcts_parity.py",    # M6: MCTS visit-count parity
             "train/test_actor_shards.py"]   # M7: shard schema / trainer ingest
    for t in tests:
        r = subprocess.run([sys.executable, t], cwd=_REPO_ROOT,
                           capture_output=True, text=True)
        print(r.stdout, end="", flush=True)
        name = os.path.basename(t)
        if r.returncode != 0:
            rep.error("actor", f"{name} failed (exit {r.returncode}):\n"
                               f"{r.stdout}{r.stderr}")
        else:
            print(f"  {name}: PASS", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", default=",".join(ALL_TIERS),
                   help="comma list of tiers to run (default: all default tiers): "
                        + ",".join(ALL_TIERS) + "; opt-in: "
                        + ",".join(OPT_IN_TIERS))
    p.add_argument("--smoke-games", type=int, default=2,
                   help="games per matchup in the smoke tier (default 2)")
    p.add_argument("--fuzz-games", type=int, default=8,
                   help="games per matchup in the fuzz tier (default 8)")
    p.add_argument("--seed", type=int, default=1,
                   help="base RNG seed (fixed => deterministic; default 1)")
    p.add_argument("--mode", default="explore",
                   help="fuzz agent tier: 'explore' (default) or 'explore:patient'")
    p.add_argument("--matchups", default=None,
                   help="matchup set: mirrors | ring | mirrors+ring | all | "
                        "'a:b,c:d' (default: smoke=mirrors+ring, fuzz=mirrors)")
    p.add_argument("--out-dir", default="ci_out",
                   help="directory for transcripts + draw logs (default ci_out/)")
    args = p.parse_args(argv)

    tiers = [t.strip() for t in args.tier.split(",") if t.strip()]
    unknown = [t for t in tiers if t not in KNOWN_TIERS]
    if unknown:
        print(f"unknown tier(s): {', '.join(unknown)}; valid: {', '.join(KNOWN_TIERS)}",
              file=sys.stderr)
        return 2

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Game tiers need a built binary and provisioned card scripts.
    game_tiers = {"smoke", "fuzz", "replay", "obsinv", "snapshot",
                  "sbselfplay", "mirror", "analysis"} & set(tiers)
    if game_tiers and not os.path.exists(runner.BINARY):
        print(f"binary not found at {runner.BINARY} — run `make` first", file=sys.stderr)
        return 2
    if {"smoke", "fuzz", "vocab", "obsinv", "snapshot", "sbselfplay",
        "mirror", "analysis"} & set(tiers):
        cards_dir = os.path.join(_REPO_ROOT, "bin", "resources", "cardsfolder")
        if not glob.glob(os.path.join(cards_dir, "*", "*.txt")):
            print(f"no card scripts under {cards_dir} — run "
                  "tools/forge_fetch/provision_decks.py first", file=sys.stderr)
            return 2

    rep = Report()
    for t in tiers:
        print(f"\n=== tier: {t} ===", flush=True)
        if t == "pygen":
            tier_pygen(rep)
        elif t == "vocab":
            tier_vocab(rep)
        elif t == "obsinv":
            tier_obsinv(rep)
        elif t == "snapshot":
            tier_snapshot(rep)
        elif t == "sbselfplay":
            tier_sbselfplay(rep)
        elif t == "mirror":
            tier_mirror(rep)
        elif t == "replay":
            tier_replay(rep)
        elif t == "smoke":
            tier_smoke(rep, args, out_dir)
        elif t == "fuzz":
            tier_fuzz(rep, args, out_dir)
        elif t == "actor":
            tier_actor(rep)
        elif t == "analysis":
            tier_analysis(rep)

    print("\n" + "=" * 60, flush=True)
    print(f"ci_check: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s)",
          flush=True)
    for tier, msg in rep.warnings:
        print(f"  WARNING [{tier}]: {msg.splitlines()[0]}", flush=True)
    for tier, msg in rep.errors:
        print(f"  ERROR   [{tier}]: {msg.splitlines()[0]}", flush=True)
    print(f"\nReproduce: train/.venv/bin/python train/ci_check.py "
          f"--tier {args.tier} --seed {args.seed} --mode {args.mode}"
          + (f" --matchups {args.matchups}" if args.matchups else ""), flush=True)
    return 1 if rep.errors else 0


if __name__ == "__main__":
    sys.exit(main())
