# Continuous integration & the test standard

The standard for changes to this repo: **`make check` must pass.** It is the one
command that stands in for the old grab-bag of harness invocations, and CI runs
exactly it. Run it locally before pushing.

```bash
make check          # builds (headless debug) + runs train/ci_check.py (all tiers)
```

`make check` requires the Forge card scripts to be provisioned first (the
SessionStart hook does this on a fresh clone / cloud session):

```bash
train/.venv/bin/python tools/forge_fetch/provision_decks.py
```

`provision_decks.py` fetches the script for **every card in `src/card_vocab.h`**
plus every card named by the decks (top-level, `meta/`, `league/`) and the tokens
they create. Fetching the whole vocab — not just deck cards — is what keeps the
codegen (`train/card_costs.py`) deterministic, so the `pygen` tier below is a
reliable gate rather than an environment-dependent one.

## The tiers (`train/ci_check.py`)

`ci_check.py` runs cheap-to-expensive; every requested tier runs even if an
earlier one fails, so one invocation reports every finding. It exits nonzero on
any **error** (warnings alone still pass).

| Tier | What it checks | Fails on |
|---|---|---|
| `pygen` | `train/_enums.py` and `train/card_costs.py` are in sync with their C++ inputs | a stale committed copy (someone changed a C++ input without running `make pygen`) |
| `vocab` | every card in the top-level and `league/` decks resolves to a `card_vocab.h` entry (DFC deck names resolve through their script's front face) | any deck card missing from the vocab |
| `obsinv` | structural per-decision invariants on the raw machine-mode observation across a few seeded scripted games (`train/test_obs_invariants.py`): card-id / entity-ref floats decode in range, recency-packed zones (GY/exile) have no holes, one-hots are one-hot, player counts non-negative, a declared companion is revealed to the opponent | any observation-encoding invariant violation (a silent `serialize_state` layout/encoding regression) |
| `replay` | the byte-identical replay corpus (`train/regression/corpus/`, decks delver/doomsday/mav in every seating) still reproduces exactly | any transcript drift |
| `smoke` | deterministic league games with the scripted **hard** agent across the league mirrors + a ring of crosses (fixed seeds) | a crash, an incomplete game, or an `ERROR:`/`FATAL:` line |
| `fuzz` | short random coverage fuzz (the `explore` agent) over the league mirrors (fixed seeds on PR) | same as smoke |

Useful flags: `--tier pygen,vocab` (subset), `--smoke-games N` / `--fuzz-games N`
(depth), `--seed N`, `--matchups mirrors|ring|mirrors+ring|all|"a:b,c:d"`,
`--out-dir DIR` (transcripts + relocated draw logs; `ci_out/` by default).

### Draw and warning classification

Draws are not acceptable, but the two causes differ:

- A game that ends with no winner because the engine hit its internal step cap (a
  **stall**) is a **warning** — flagged for review, does not fail the gate.
- A game whose engine process **crashed** (nonzero exit / EOF mid-game, surfaced
  as an exception) is an **error** — fails the gate. So is any `ERROR:`/`FATAL:`
  / assert / traceback line in a transcript (a non-fatal engine `ERROR:` is not
  acceptable per `CLAUDE.md`).

Routine `WARNING: Unrecognized ability param` lines are left in the transcript
(greppable) but not surfaced — they fire in bulk for cosmetic / AI-hint params
and would drown the signal. The keys currently suppressed at the source (and the
ones still needing a real handler) are tracked in `todo.md`.

## Workflows

- **`.github/workflows/ci.yml`** — on push / PR to `main` (and `workflow_dispatch`).
  Runs all tiers with the fixed default seed, so a red run is the diff's fault,
  not RNG. Transcripts are uploaded as an artifact (`if: always()`), so
  warning-level stall-draw logs are reviewable even on green runs.
- **`.github/workflows/nightly-fuzz.yml`** — scheduled (08:00 UTC) + manual. Runs
  the `fuzz` tier over **all** league matchups in both `explore` and
  `explore:patient` modes, with a **rotating** seed (derived from the run number,
  printed to the job summary with the exact local reproduce command).

Both share `.github/actions/setup-robomage` (Python venv, pip / card-script /
ccache caching, provisioning, headless debug build). The card-script cache is
populated with a split restore/save so the first run leaves a warm cache even if
a later step fails; the fetch tool is add-only, so a stale cache only re-fetches
the delta.

## Reproducing a CI failure locally

1. **PR gate failure:** run `make check` (same command, same fixed seed). The
   failing tier names the transcript file under `ci_out/`; open it. Download the
   run's `ci-transcripts-*` artifact if you want the exact CI transcripts.
   - An **`obsinv`** failure prints the exact violated invariant (decision index,
     seat, block, slot, raw value); reproduce it directly with
     `train/.venv/bin/python train/test_obs_invariants.py` (fixed internal seeds,
     so it replays identically).
2. **Nightly-fuzz failure:** the job summary prints the seed and the exact
   `ci_check.py --tier fuzz ... --seed <SEED>` line — run it locally to replay
   the same games. The `nightly-fuzz-*` artifact has the transcripts.

## Intentionally changing behavior

Some tiers pin current behavior, so an **intended** change must update them in
the same commit (the diff is part of the review).

**The one-command path is `make regen`** — it builds, provisions the card set,
force-reruns both codegen generators, and re-records the replay corpus; then you
review and commit the changed files. Prefer it over the individual steps below
(which document what it does):

- **`replay` drift** from a deliberate engine/agent/card-data change:
  re-record the corpus and commit it.
  ```bash
  train/.venv/bin/python train/regression/replay_diff.py record
  ```
  The corpus is **platform-portable**: all gameplay randomness goes through
  `src/stable_rng.h` (hand-rolled Fisher–Yates + rejection sampling over the
  raw mt19937 stream), so Mac and Linux produce byte-identical transcripts
  from the same seed. Keep it that way — never call `std::shuffle`,
  `std::uniform_int_distribution`, or `rand()` with `cur_game.gen` (or for any
  gameplay decision); their output is implementation-defined and differs
  between libstdc++ (CI) and libc++ (macOS), which silently re-locks the
  corpus, RMLOG replays, and bug-repro seeds to one platform.
- **`pygen` staleness** after editing `src/card_vocab.h` (or a C++ codegen
  input): regenerate and commit the outputs (provision the full card set first so
  `card_costs.py` is complete).
  ```bash
  train/.venv/bin/python tools/forge_fetch/provision_decks.py
  train/.venv/bin/python train/gen_enums.py
  train/.venv/bin/python train/gen_card_costs.py
  ```
  Run the generators directly (or use `make regen`) rather than `make pygen`
  here: ci_check's pygen tier restores the committed copies with `git
  checkout`, which bumps their mtimes past the C++ inputs, so the incremental
  `pygen` file targets can report "nothing to be done" on stale content.

`train/fuzz_campaign.py` remains the manual, exploratory fuzz-campaign tool (dumps
a transcript for review; always exits 0). `ci_check.py` is the gating wrapper.
