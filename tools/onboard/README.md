# onboard — the deck onboarding pipeline

One command takes a scraped meta deck to a playable, fuzzed deck in `decks/league/` (training) or
`decks/heldout/` (never trained on: the generalisation test). The plan it implements is
`docs/open_vocab_plan.md` S7. Standard library only (`python3`, no venv); run from anywhere.

```bash
python3 tools/onboard/onboard.py scrape --top 25 --meta 39   # stage 1 (network)
python3 tools/onboard/onboard.py all                         # stages 2-6 for every deck
python3 tools/onboard/onboard.py report                      # one table of every status file
python3 tools/onboard/onboard.py vocab --deck bant_control --dry-run
```

## Deck directories and status files

| directory | holds |
|---|---|
| `bin/resources/decks/candidates/` | scraped decks, not yet playable; **every deck's status file** `<stem>.status.json`; `REPORT.md` (the capability report) |
| `bin/resources/decks/league/` | training decks (`rmtrain`'s `league.decks` names stems here) |
| `bin/resources/decks/heldout/` | playable decks never trained on |

A deck moves from `candidates/` to `league/` or `heldout/` by `git mv` (stage 10); its status file
stays in `candidates/` (the audit trail, committed). `robomage --check-decks` and the gate's `vocab`
tier cover `league/` and `heldout/`; the `fuzz` and `smoke` tiers play every deck of both in their
mirrors and ring (`tools/robomage_check/gate.h`, `playable_decks`), so an onboarded deck gets CI
games (8 bo1 mirror games, a bo3 and 2 `--no-auto-mana` ring games, 4 scripted smoke games) with no
further wiring. `candidates/` is checked by `onboard.py check` (below), not by `--check-decks`.

A status file records the deck's source (MTGTop8 name, rank, meta share, section, ids, scrape date
and window) and per stage `{result, time, inputs, summary, detail}`: `result` is `ok`, `fail` or
`pending` (stage 6 while mechanics are missing; stage 7 under `--dry-run`); `inputs` the sha256 of
what the result depends on (the deck file, `src/card_vocab.h`, the scripts the deck's cards resolve
to, the engine binary, the token scripts, `mechanic_sizes.json`, the text table). A stage is skipped
while its inputs are unchanged and its result `ok`; `--force` reruns it.

## Stages

| # | stage | what it does | runs in `all` |
|---|---|---|---|
| 1 | `scrape` | `tools/meta_scraper/scrape_meta.py --top N --meta M --manifest` into a scratch directory (or `--from DIR`, a scraper output with its `scrape.json`), then each deck into `candidates/` with its status file. A stem naming a deck in `league/` or `heldout/` gets the suffix `_top8`; a designated deck is not overwritten | no (network) |
| 2 | `diff` | `robomage --check-decks --check-decks-dir candidates/<stem>.dk`: each card in vocab / script but not in vocab / no script | yes |
| 3 | `provision` | `tools/forge_fetch/fetch_script.py` for the cards without a script and the token scripts the new cards make (`provision_decks.collect_tokens`), at the pinned Forge commit. A card Forge lacks at the pin is recorded (and whether Forge's master has it) and becomes the capability item `script:<card>` | yes (fetches only what is missing; `--offline` fetches nothing) |
| 4 | `parse` | `robomage --parse-report @<names>` over the deck's cards not in the vocab (`docs/game_running.md`, `src/parse_report.h`): every finding per card | yes |
| 5 | `capability` | per card clean / needs mechanics; per deck the new cards and mechanics with a size; writes `candidates/REPORT.md` over every status file: decks cheapest first, mechanics ranked by the decks that need them (and the decks each alone unlocks), a greedy unlock order | yes |
| 6 | `implement` | `ok` once the deck's new cards need no mechanic; else `pending` with the list. The engine work is a developer's, one mechanic per commit, honouring the scripts' tags, a rules scenario each, `docs/rules_deviations.md` for gaps | yes |
| 7 | `vocab` | appends the deck's new cards to `card_vocab_entries` at the next free ids below 900 (each face of a double-faced card its own id; a split card's halves and `A/B` one id) and the token scripts they make to `token_vocab_entries` (900-1022), then `make registry` and prints `git diff --stat src/card_vocab.h` for review. `--dry-run` only reports | no |
| 8 | `text` | `make text-table`: the new cards' oracle-text rows (fetches the pinned model, needs `make rmembed`) | no |
| 9 | `checks` | `make check` (unless `--skip-make-check`), then the campaign `robomage_check --tier fuzz --matchups <dir>/<stem>:* --fuzz-games 100 --manual-mana-games 10 --out-dir ci_out/onboard_<stem>`: bo1 explore, bo3 `explore:patient` and `--no-auto-mana` games against every playable deck. Any error or drawn game fails it | no |
| 10 | `designate` | `--to league` or `--to heldout`: `git mv` the deck there | no |
| 11 | `baseline` | `rm_driver.py` `scripted:hard` with the deck against every league deck (`--baseline-games` bo1 each), then `learn/rmtrain bench <run> --decks <dir>/<stem>` (`--run`). Fails with a message until `rmtrain bench` has `--decks` (plan S8) | no |

Rules (plan S7): a stage never runs past an earlier one that is not `ok` (stages 2-6 rerun
themselves first when their inputs changed; 7-10 need only a recorded `ok`); `vocab` needs zero
findings and zero missing mechanics (stage 6 `ok`); `designate` needs a clean campaign (stage 9
`ok`); at least one fair and, when available, one combo deck stay in `heldout/` permanently.

## Mechanics and sizes

`mechanic_sizes.json` groups parse-report findings (`kind:name`, e.g. `trigger_mode:LandPlayed`,
`ability_param:Charm.CanRepeatModes`, `script:*`) into the mechanics an engine change implements,
each with a size (S ≤ ½ day, M ≈ 1 day, L 2-3 days, XL a week or more) and a note. A finding no group
lists is its own mechanic, sized by its kind. Edit the file when a review shows a better grouping
or estimate; stage 5 reruns (the file is one of its inputs).

## The gate's part

`onboard.py check` reruns stages 2, 4 and 5 for every deck in `candidates/` in memory — no file
written, no network — and compares the results with the committed status files: a tool failure
exits 1, a status file whose recorded result the current engine and vocab no longer give is
printed as `STALE` (rerun `onboard.py all` and commit the status files). The gate's `vocab` tier
runs it (`tools/robomage_check/parse_report_check.h`): errors fail, stale files warn. CI has the
candidates' scripts because `tools/forge_fetch/provision_decks.py` provisions the `candidates/` and
`heldout/` decks with the vocab and `league/`.
