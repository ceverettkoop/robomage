---
name: demonstrate-session-cards
description: End-of-session demonstration pass for a batch of newly-implemented cards. Builds temporary decks (a base curated deck + the session's new cards), runs the engine-sanity-check over real scripted games, proves every new card was actually demonstrated (cast + its defining clause resolved) — filling gaps with targeted test-harness scenarios — then reviews the verbose output for engine bugs. Use at the END of an autonomous-implement-missing-cards session (or any batch card-implementation session) to verify the batch in real games before handing back.
---

# Demonstrate session cards

This is the **verification capstone** that runs after a batch of cards has been implemented
(e.g. by `autonomous-implement-missing-cards`). Each card was already unit-tested in isolation
during implementation; this pass proves the whole batch works **together, in real games**, and
that **every** new card is actually exercised — then it runs the standard engine-sanity-check
narrative review over those games.

It composes two existing skills: it **builds the test vehicle** (temp decks containing the new
cards) and then **invokes the `engine-sanity-check` methodology** on them, plus a per-card
coverage guarantee backed by `test_harness.py`.

**Flag findings for the user; do not fix code unless asked** (same rule as engine-sanity-check).
Separate genuine **engine bugs** from **scripted-agent suboptimality**. Ground truth for correct
behavior is the Comprehensive Rules at `docs/mtg_comprehensive_rules.txt` (navigation guide
`docs/mtg_comprehensive_rules.md`) — grep the numbered rule before classifying anything.

## 1. Enumerate the session's new cards

The cards implemented in the branch each have a design doc committed under
`docs/card_implementations/`. List them (and their commits):

```bash
git log main..HEAD --oneline | grep -i "^.* Implement "
git diff --name-only main..HEAD -- docs/card_implementations/ | sort
```

For each card, read its doc and extract: exact **vocab name string** (confirm it is in
`src/card_vocab.h`), **mana cost**, **color**, **types**, the **defining clause** (the unique
effect/trigger that must be seen to call it "demonstrated"), and any **special requirement** to
trigger that clause (needs a sacrifice, needs other creatures, alt/pitch cost, X cost, needs
graveyard/library cards, timing window). A subagent is good for this extraction — ask for a
table plus a separate list of **cards whose cost contains a colorless `{C}` symbol** (see §3).

## 2. Build temporary decks (base deck + new cards)

Write `temp/t<base>.dk` files under `bin/resources/decks/temp/`, one per base curated deck
(`delver` = U/R + artifacts + colorless-Eldrazi, `mav` = G/W + lands, `doomsday` = U/B). Start
from the base deck's mana/engine and **distribute the new cards by color** into the deck that can
actually cast them; swap out base spells to keep each deck at **60 cards** (and a `SIDEBOARD:`
line). The goal is *coverage*, not a competitive list — make the new cards a heavy fraction so
they get drawn, but keep enough lands/cantrips that the deck functions.

Each new card must land in a deck whose mana base supports it. Lands (duals, fetches, utility
lands) go in whatever deck shares their colors. A card that is awkward in every base deck (an
off-color creature) still goes in the closest deck — §4 will demonstrate it via the harness if it
never gets cast organically.

**Colorless `{C}` costs need a colorless mana source — they are NOT castable with generic mana.**
The engine enforces this: in `src/mana_system.cpp` `pay_from_pool`, a `{C}` pip (the `COLORLESS`
color) requires an exact `COLORLESS` mana in the pool; only `GENERIC` pips eat any mana. So any
new card whose cost contains `{C}` (e.g. Glaring Fleshraker `{2}{C}`, Eldrazi Linebreaker
`{1}{C}{R}`, Kozilek's Command `{X}{C}{C}`) must be placed in a deck that contains a **colorless
source** — **Wasteland** (`{T}: add {C}`), **Ancient Tomb** (`{C}{C}`), or **Eldrazi Temple**
(`{C}`, or `{C}{C}` for colorless-Eldrazi spells). The base `delver` and `mav` decks already run
4× Wasteland; add Ancient Tomb / Eldrazi Temple when a deck needs more `{C}` (e.g. Kozilek's
Command needs two `{C}` in one turn → Ancient Tomb supplies both). In §4 you will **verify from
the transcript that the `{C}` pip was actually paid by a colorless source, not generic.**

Validate every temp deck (60 cards, all names in vocab) before running:

```bash
train/.venv/bin/python -c "
import sys; sys.path.insert(0,'train'); import missing_cards as mc
vocab,_ = mc.parse_vocab(mc.VOCAB_H)
for fn in ['temp/tdelver.dk','temp/tmav.dk','temp/tdoomsday.dk']:
    cards=mc.parse_deck('bin/resources/decks/'+fn); total=sum(q for q,n in cards)
    miss=[n for q,n in cards if mc.name_to_uid(n) not in vocab]
    print(fn,'count=',total,'COMPLETE' if not miss else ('MISSING: '+str(miss)))"
```

## 3. Run the engine-sanity-check games on the temp decks

Run scripted-vs-scripted `observe --verbose` games — **mirror** matchups (each temp deck vs
itself, exercising its whole pool on both sides) and **cross** matchups — saving each transcript.
Mirrors get more games (richer pool coverage); crosses fewer. This is the `engine-sanity-check`
generation step pointed at the temp decks:

```bash
D=<scratchpad>/rmtx; mkdir -p "$D"
run(){ train/.venv/bin/python train/train.py observe --deck "$1" --opponent "$2" \
       --verbose --games "${3:-2}" --seed 1 > "$D/$(basename $1)__$(basename $2).txt" 2>&1
       tail -1 "$D/$(basename $1)__$(basename $2).txt"; }
run temp/tdelver temp/tdelver 3;  run temp/tmav temp/tmav 3;  run temp/tdoomsday temp/tdoomsday 3
run temp/tdelver temp/tmav;  run temp/tdelver temp/tdoomsday;  run temp/tmav temp/tdoomsday
```

Run the engine-sanity-check automated anomaly scan over the transcripts (errors/asserts, draws,
`W / L / D` summaries — **D must be 0**, fizzles, card-load warnings). A **DRAW or stalled game is
a finding** (per CLAUDE.md draws are unacceptable) and is often the visible symptom of a deeper
bug — e.g. an engine crash mid-game makes the harness report a draw and dump `draw_<n>.txt`.

## 4. Prove every new card was demonstrated

"In the deck" and "appeared in the transcript" are **not** "demonstrated." A raw name grep counts
hand/menu/board listings:

```bash
cd "$D"; for c in "Card One" "Card Two" ...; do printf "%-28s %s\n" "$c" "$(grep -rhoF "$c" *.txt | wc -l)"; done
```

…so every card will likely show non-zero, but that only tells you which cards **never appeared at
all**. To classify each card you must read the **Narrative** lines: a card is

- **DEMONSTRATED** — cast/played/activated AND its defining clause resolved (quoted evidence:
  cast line + the effect/trigger/zone-change/damage line, with `file:decision`).
- **PARTIAL** — cast/entered, but its defining clause never fired (e.g. Chalice cast for X=0 so
  its counter never triggered; Containment Priest in play but no uncast creature ever entered;
  Aether Vial accrued counters but never put a creature in).
- **NOT-DEMONSTRATED** — only ever sat in a hand or a menu.

Fan out one subagent per deck transcript (as engine-sanity-check suggests) to produce the
demonstration table with quoted evidence, the colorless-mana confirmations (§2), and the bug scan
(§5) in one pass.

**For every PARTIAL or NOT-DEMONSTRATED card, force its defining clause with a targeted
`test_harness.py` scenario** (this is how you *guarantee* coverage rather than hoping the scripted
agent stumbles into it). Use stacked hands / pre-set zones and `--play` seat keys:

- `--battlefield-a/-b`, `--graveyard-a/-b`, `--hand-a/-b`, `--library-a/-b` set up the exact
  board (e.g. start Arclight Phoenix in `--graveyard-a` then cast 3 instants to trigger its
  begin-combat return; pre-place Kappa Cannoneer then cast another artifact to fire its +1/+1).
- `--play "A:…,B:…"` with seat keys scripts both sides' lines; `#<n>` is the index escape for X
  choices / mode picks; `--no-shuffle` is implied with stacked hands.
- **Colorless `{C}` cards:** in the targeted run, confirm the `{C}` pip is paid by a colorless
  source — the transcript shows e.g. `activated Wasteland for 1(C)` / `Ancient Tomb for 2(C)` /
  `Eldrazi Temple for 2(C)` next to the cast. If a `{C}` is ever paid from a generic/colored
  source, that is a **LIKELY ENGINE BUG**.

Iterate until **every** new card has either a DEMONSTRATED row from the games or a passing
targeted-harness demonstration. Read the menu the harness prints on a failed `--play` spec and fix
the spec.

## 5. Deep narrative review (engine-sanity-check §4)

Apply the full `engine-sanity-check` deep-review methodology to the transcripts, focused on the
new cards' mechanics and their cross-interactions (e.g. Containment Priest exiling a searched
creature, Leyline exiling opponent cards, a symmetric static like Clarion Conqueror hitting both
players' creature abilities but not lands, a self-damage land like Ancient Tomb). For each
suspicious event capture `file:decision` + a quoted snippet and classify it: **LIKELY ENGINE BUG**
(verify root cause against the card script + parser/effect source and the CR before asserting) vs
**scripted-agent suboptimality** (X-spells for X=0, casting into an empty stack, over-activating)
vs **cosmetic** (`Unrecognized ability param` card-load warnings — acceptable when behavior is
inferable). Watch specifically for: a missing **legal-target guard** on a new activated ability
(offering it with no legal target can crash the engine on an empty action menu → reported as a
draw); P/T or life deltas that don't match the card; triggers double-firing or not firing.

## 6. Report

Give the user one consolidated report:
1. **Per-card demonstration table** — all session cards, DEMONSTRATED / PARTIAL→harness-confirmed
   / NOT-DEMONSTRATED→harness-confirmed, each with quoted evidence.
2. **Colorless-mana confirmations** — for every `{C}`-cost card, the line showing a colorless
   source paid the `{C}`.
3. **Classified findings** — engine bugs first (file:decision, quoted evidence, confirmed root
   cause), then scripted-agent suboptimalities, then cosmetic warnings, then anything still
   untested. Note any **draw/stall** prominently. Do not fix code unless asked; if fixes are
   deferred, file them in `todo.md`.

Temp deck files in `bin/resources/decks/temp/` are scratch — leave them, or note they can be
removed. Remove any stray `draw_<n>.txt` crash dumps left in the repo root by a stalled game.
