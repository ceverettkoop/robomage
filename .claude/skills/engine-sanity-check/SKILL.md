---
name: engine-sanity-check
description: Run scripted-vs-scripted games for the fully-implemented decks and review the transcripts for incorrect or suspicious engine behavior, then flag findings. Use after an engine/parser/effects rework, after adding cards or mechanics, or whenever you want a regression pass over real games. Distinguishes genuine engine bugs from scripted-agent suboptimality.
---

# Engine sanity check

Plays full scripted-vs-scripted games with the decks whose cards are **all** implemented,
scans + reads the transcripts, and flags anything that looks like incorrect engine behavior.
The goal is a fast, repeatable regression pass after a rework. **Flag findings for the user;
do not fix anything unless asked.** Always separate genuine **engine bugs** from **scripted-agent
suboptimality** (the rule-based agent plays badly on purpose-built combos and X-spells — that
is not an engine fault).

**Ground truth for "correct" behavior is the Comprehensive Rules** at
`docs/mtg_comprehensive_rules.txt` (navigation guide: `docs/mtg_comprehensive_rules.md`). Before
classifying any suspicious event as an engine bug, confirm what the rules actually require for
that mechanic — grep the relevant numbered rule (e.g. rule 702 for keyword abilities, 704 for
state-based actions, 5xx for combat/timing) instead of relying on memory.

## 1. Pick the fully-implemented decks

A deck is testable only if every card it lists is in `src/card_vocab.h`. The curated training
decks live in `bin/resources/decks/` (currently `delver`, `doomsday`, `mav`); the scraped
metagame decks in `bin/resources/decks/meta/` usually still have missing cards. Confirm coverage:

```bash
train/.venv/bin/python -c "
import os, sys; sys.path.insert(0, 'train'); import missing_cards as mc
vocab,_ = mc.parse_vocab(mc.VOCAB_H)
for fn in ['delver.dk','doomsday.dk','mav.dk']:
    miss=[n for q,n in mc.parse_deck('bin/resources/decks/'+fn) if mc.name_to_uid(n) not in vocab]
    print(fn, 'COMPLETE' if not miss else miss)"
```

Note any deck that only fails coverage because of a combined double-faced name
(e.g. delver lists `Delver of Secrets Insectile Aberration` while the vocab has the faces
separately) — that is a naming artifact, not a missing card; the engine still loads it.

## 2. Generate transcripts

Run **mirror** matchups (each deck vs itself — exercises its whole card pool on both sides) and
**cross** matchups (deck interactions), a couple of games each, with `train.py observe` driving
both sides (scripted vs scripted is the default) in `--verbose` mode so the full per-decision
transcript is captured. Save each to a file. `observe --games N` plays N games and prints a
per-game result line plus a final `W / L / D` summary; `--seed` makes the run reproducible
(game k uses seed+k).

```bash
mkdir -p /tmp/rmtx
run(){ train/.venv/bin/python train/train.py observe --deck "$1" --opponent "$2" \
       --verbose --games 2 --seed 1 > "/tmp/rmtx/$1__$2.txt" 2>&1
       tail -1 "/tmp/rmtx/$1__$2.txt"; }
run delver delver;  run doomsday doomsday;  run mav mav
run delver mav;     run delver doomsday;    run doomsday mav
```

Every game must end with a winner — `observe` prints `game k/N: <Scripted/X> wins` per game and a
`W / L / D` summary line. Any **draw** (D > 0), or a game `observe` reports as stalled (the engine
step cap is hit with no winner — it counts as a draw and dumps the full log to `draw_<n>.txt`), is
itself a finding (per CLAUDE.md draws are unacceptable). Review the verbose output, not just the
summary.

Transcript format (`observe --verbose`): `--- Decision N ---` blocks — decoded state (life, hands,
battlefield with `[P/T]`, `(T)`=tapped, `(SICK)`=summoning-sick, stack, graveyards), then an
`Actions:` menu of every legal choice, then the chosen `>> <Scripted/Side>: i (desc)` line —
interleaved with `--- Narrative ---` lines (the engine game log: casts, damage, zone moves,
triggers, combat). Turn banners (`--- Turn N (Player X) ---` + each side's known hand) mark turn
changes. Review the Narrative lines and the state deltas around them.

## 3. Automated anomaly scan

```bash
cd /tmp/rmtx
grep -rhiE "error|exception|traceback|assert|abort|segfault|non-fatal" *.txt | grep -vi "Unholy"
grep -rhE "Unrecognized ability param" *.txt | sort | uniq -c          # card-load warnings
grep -rhiE "fizzle|fails to find|illegal" *.txt | sort | uniq -c
grep -rhE "^[0-9]+W / [0-9]+L / [0-9]+D" *.txt                          # per-file W/L/D summary — D must be 0
grep -rlE "DRAW \(should not occur\)|stopped after" *.txt              # files with a draw / stalled game
```
(Case-sensitive `DRAW` on purpose — a case-insensitive match would hit every
"Player X draws ..." line. The draw/stall banners are the only real signals.)

- **Unrecognized ability param** warnings are card-load cosmetics (a sub-param the parser
  ignores). Acceptable when behavior is inferable; suspicious only if a card visibly misbehaves.
- **fizzle / fails to find** are usually scripted-agent suboptimality (over-activated abilities
  whose target was already consumed; X-spells cast for X=0; tutors where the agent declines).
  Investigate context before deciding — quote the surrounding lines.

## 4. Deep narrative review

Read the transcripts (or fan out one subagent per deck for parallelism) against a per-deck
mechanic checklist. For each deck, scrutinise the cards/mechanics it actually exercises, e.g.:
- **delver**: DFC transform (Delver→Insectile Aberration on instant/sorcery reveal at upkeep),
  delve (Murktide P/T), delirium (Dragon's Rage Channeler, Unholy Heat damage mode), alt costs
  (Force of Will pitch, Daze bounce), tokens (Cori-Steel Cutter), fetchlands/Wasteland.
- **doomsday**: Thassa's Oracle win condition + timing, decking out (empty-library draw = loss,
  never a draw), Doomsday pile, rituals (Dark Ritual/Lotus Petal/Lion's Eye Diamond), discard
  (Thoughtseize life loss), flashback (Deep Analysis), Street Wraith cycling.
- **mav**: Green Sun's Zenith X value + fetch, Knight of the Reliquary P/T growth, Gaea's Cradle
  (G per creature), exalted (Hierarchs), Swords to Plowshares (the *exiled creature's controller*
  gains life = its power), Sylvan Library, Thalia tax, Wasteland, evoke (Endurance).

For every suspicious event, capture **file:line + a quoted snippet** and classify it:
- **LIKELY ENGINE BUG** — an effect resolves to the wrong amount/zone/target/player, a P/T or
  life delta that doesn't match the card, a trigger double-firing/not firing, an impossible board
  state, or a stall/draw. Verify the root cause against the card script and the parser/effect
  source before asserting (a `Resolving ability ... amount: 0` line is only a bug if the observed
  effect is actually wrong — many abilities print amount:0 yet function).
- **scripted-agent suboptimality** — bad-but-legal play (casts X-spells too cheaply, declines
  good blocks, over-activates an ability, can't pilot a combo). Note it once; do not treat as an
  engine fault.

Common false alarms (verify, usually NOT bugs): Swords life gain goes to the exiled creature's
controller; Doomsday "lose half your life rounded up"; Daze/Force of Will targeting a spell on
the stack; summoning-sick creatures being offered as blockers (legal).

## 5. Report

Give the user a single consolidated list at the end: engine bugs first (with file:line, quoted
evidence, and confirmed root cause), then scripted-agent suboptimalities, then cosmetic
warnings, then anything left untested (cards that never resolved in the sample). Do not fix code
unless asked — if the user wants fixes deferred, file them in `todo.md`.
