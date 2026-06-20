# Corpus review — broken core functionality

## STATUS: FIXED (2026-06-20). Baseline re-recorded on the corrected engine.

**Fix:** `parse.cpp` `parse_one_trigger` — when resolving a trigger's `Execute$`
SVar it copied only a hand-picked subset of effect fields, dropping
`Origin$`/`Destination$`/`ValidTgts$`/etc. Now it takes the full parsed effect
and restores only the trigger metadata. Plus `resolve_change_zone_all`
(`ability.cpp`) now honors the target player, library-bottom placement
(`LibraryPosition$`), and `RandomOrder$`, with accurate logging; `RandomOrder$`
is now parsed.

**Post-fix corpus (re-recorded):**
- Pump/PutCounter/MultiplyCounter: 80 each → **1 each**. Landfall trigger storm gone.
- Endurance now correctly bottoms the target's graveyard ("moves 0 card(s) to the
  bottom of their library" when graveyard empty) instead of dumping the library
  onto the battlefield.
- mav games: ~200 decisions → ~100, all end with a real winner (no deck-out loop).
- BUG 3 "gains (always)": 80×-inflated → **2 total** (was a storm side-effect).
- New `thassas_oracle_win` scenario added → covers WinsGame + Dig + SylvanLibrary.
- Determinism: all 11 scenarios byte-identical on `check`. No CRASH lines.
- Remaining "ChangeZone fizzles" (×43): the **scripted agent** spam-activates
  Keen-Eyed Curator's {1} graveyard-exile at an already-exiled target — correct
  rules behavior (illegal target → fizzle), agent inefficiency, NOT an engine bug.

BUGs 1/2/3/4 below were all resolved by the single BUG 1 fix (2/3/4 were
downstream of it). Original catalog retained below for reference.

---


Review of `train/regression/corpus/*.txt` (10 scripted games, decks delver/doomsday/mav)
for behavior that is wrong **independent of the dispatch refactor**. These are
pre-existing engine bugs surfaced by the corpus. Fix engine-side only — do NOT
modify card scripts.

Healthy: all **delver** games (189/178/62/63 decisions) and **doomsday** games
(56/56/63) — reasonable Draw/ChangeZone/PeekAndReveal/Counter counts, no runaway.
All **mav** games (193–203 decisions) are broken — same root cause below.

---

## BUG 1 — `ChangeZoneAll` ignores `Origin$` and `Destination$` (Endurance)  [SEVERITY: HIGH]

**Card:** Endurance (`cardsfolder/e/endurance.txt`). Intended ETB (script line 8-9):
> up to one target player puts all the cards **from their graveyard** on the **bottom of their library** in a random order.
> `DB$ ChangeZoneAll | ChangeType$ Card | Origin$ Graveyard | Destination$ Library | LibraryPosition$ -1 | RandomOrder$ True`

**Observed** (`mav_mirror.txt:103-160`):
```
Player A casts Endurance
Endurance triggered
Resolving ability (category: ChangeZoneAll, amount: 0)
Player A exiles Green Sun's Zenith
... (53 lines) ...
Player A exiles 53 card(s) from library and graveyard
```
It pulls from **library AND graveyard** (Origin should be Graveyard only) and the
move does not honor `Destination$ Library` (bottom). `resolve_change_zone_all`
in `src/components/ability.cpp` (~line 486) hardcodes the "exiles … from library
and graveyard" log and does not respect the parsed `origin`/`destination`.

**Expected:** move only the target player's **graveyard** to the **bottom of
their library**, random order. Nothing should leave the library; nothing should
be exiled.

**Fix area:** `Ability::resolve_change_zone_all` (origin/destination handling +
log message); verify `Origin$ Graveyard` / `Destination$ Library` /
`LibraryPosition$ -1` / `RandomOrder$` are parsed and applied. Confirm the
`RandomOrder$ True` param is consumed (it currently logs "Unrecognized ability
param: RandomOrder$ True" at parse).

---

## BUG 2 — Landfall trigger storm (Pump → PutCounter → MultiplyCounter ×80)  [SEVERITY: HIGH, user-flagged]

**Observed** (`mav_mirror.txt:167-266`, repeats in every mav game):
```
Icetill Explorer triggered      (×20)
Scythecat Cub triggered         (×80)
...
Resolving ability (category: Pump, amount: 0)
Choose a creature for Pump:
Resolving ability (category: PutCounter, amount: 0)
Resolving ability (category: MultiplyCounter, amount: 0)
   ... repeats 80× ...
```
Counts are deterministic across all 4 mav scenarios. The math: BUG 1 moves ~20
lands through a zone change, and each land's zone-change fires the
`ChangesZone | Destination$ Battlefield | ValidCard$ Land.YouCtrl` landfall
trigger on every landfall creature in play — 1 Icetill Explorer → 20, 4 Scythecat
Cubs → 80 (20 × 4). The resulting Pump chain is the repeating block the user
flagged.

**Root cause:** primarily downstream of BUG 1 — the lands should never have been
moving (graveyard→library bottom doesn't put lands onto the battlefield). The
destination filter IS checked (`state_manager.cpp:1087-1088`), so the storm means
BUG 1's mass move is landing cards in a zone that matches `Destination$
Battlefield`. **To verify during fix:** whether `ChangeZoneAll` is moving the
cards to BATTLEFIELD (despite logging "exiles"), which would make landfall fire
legitimately on garbage input. Re-check after BUG 1 is fixed; the storm likely
disappears. If landfall still over-fires, inspect how `ChangeZoneAll`/mass moves
emit `CARD_CHANGED_ZONE` events (per-card destination correctness).

---

## BUG 3 — Static "gains (always)" re-application spam  [SEVERITY: MEDIUM]

**Observed** (`mav_mirror.txt:161-166`):
```
Icetill Explorer gains (always)
Icetill Explorer gains (always)
Knight of the Reliquary gains (always)   (×4)
```
Continuous/static abilities log "gains (always)" repeatedly (once per permanent
per SBE pass, it appears). Likely benign correctness-wise but indicates static
effects are re-applied/re-logged each pass rather than idempotently. Confirm it
is not doing real duplicate work (e.g. stacking bonuses). Lower priority than
BUG 1/2.

---

## BUG 4 — mav games run ~200 decisions and end by decking out  [SEVERITY: MEDIUM — likely downstream]

All mav games reach 193–203 decisions and terminate with "Player A decked -
Player B wins!" after a tail of repeated `Mill` resolutions
(`mav_mirror.txt` last ~10 lines). Likely a downstream consequence of the
board/zone corruption from BUG 1/2 (e.g. a card stuck re-milling). Re-evaluate
after BUG 1/2 are fixed; if mav games then end normally, no separate fix needed.

---

## Suggested fix order
1. **BUG 1** (`resolve_change_zone_all` origin/destination) — root cause; fixing
   it most likely resolves BUG 2 and BUG 4 as well.
2. Re-record the corpus expectations only after fixes land (current
   `corpus/*.txt` baselines encode the buggy behavior — they must be regenerated
   once the engine is correct, otherwise the replay-diff would lock in the bugs).
3. **BUG 3** if it proves to do real duplicate work.

> NOTE: the replay-diff safety net currently encodes the BUGGY transcripts as
> "expected". Do not treat a passing `check` as correctness here — it only proves
> no *new* drift. After fixing BUGs 1/2/4, re-run `replay_diff.py record` on the
> corrected engine to reset the baseline, then resume the dispatch refactor.
