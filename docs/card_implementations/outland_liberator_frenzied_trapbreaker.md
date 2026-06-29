# Outland Liberator // Frenzied Trapbreaker

Combined transforming double-faced card (uid: `outland_liberator_frenzied_trapbreaker`).
Vocab indices: **294** (front, Outland Liberator) and **295** (back, Frenzied Trapbreaker).

## Oracle

**Outland Liberator** — {1}{G}, Creature — Human Werewolf, 2/2
- {1}, Sacrifice Outland Liberator: Destroy target artifact or enchantment.
- Daybound (If a player casts no spells during their own turn, it becomes night next turn.)

**Frenzied Trapbreaker** (back) — Creature — Werewolf, 3/3, green
- {1}, Sacrifice Frenzied Trapbreaker: Destroy target artifact or enchantment.
- Whenever Frenzied Trapbreaker attacks, destroy target artifact or enchantment defending player controls.
- Nightbound (If a player casts at least two spells during their own turn, it becomes day next turn.)

## Forge source

Pre-existing local combined script `bin/resources/cardsfolder/o/outland_liberator_frenzied_trapbreaker.txt`
(`AlternateMode:DoubleFaced`). Key tags:
- Both faces: `A:AB$ Destroy | Cost$ 1 Sac<1/CARDNAME> | ValidTgts$ Artifact,Enchantment` (sac-to-destroy).
- Front: `K:Daybound`. Back: `K:Nightbound`.
- Back: `T:Mode$ Attacks | ValidCard$ Card.Self | Execute$ TrigDestroy` with
  `SVar:TrigDestroy:DB$ Destroy | ValidTgts$ Artifact.ControlledBy TriggeredDefendingPlayer,Enchantment.ControlledBy TriggeredDefendingPlayer`.

`K:Daybound`/`K:Nightbound` already flow through the parser's generic keyword fallback
(`split_keywords`), so each face stores its keyword in `CardData::keywords`; no parser change was
needed to mark the faces.

## Engine work

### New general mechanic: Day/Night + Daybound/Nightbound (CR 731, 702.145, 502.2)
- `Game::day_night` (`DayNight {DN_NEITHER, DN_DAY, DN_NIGHT}`, src/classes/game.h) — the game-wide
  designation, starting at neither (CR 731.1). Reset per fresh Game.
- **src/day_night.{h,cpp}** — the single subsystem that changes the designation and performs the
  transforms it drives, reusable by any daybound/nightbound card:
  - `become_day()` / `become_night()` → `set_day_night()` set the designation and immediately
    transform (CR 702.145c/f): becoming night flips every front-face-up daybound permanent to its
    back face; becoming day flips every back-face-up nightbound permanent to its front face. No-op
    if already that designation (CR 731.1).
  - `card_has_daybound(front)` — front-face Daybound keyword marks a participating DFC.
  - `day_night_untap_transition()` — the CR 502.2 / 731.2 turn-based check.
- **Turn-based transition (CR 502.2 / 731.2), src/classes/game.cpp `advance_step` UNTAP case:**
  `day_night_untap_transition()` runs as the second part of the untap step (after phasing, before
  untap), keyed on the turn that just ended. If it's day and the previous turn's active player cast
  no spells that turn → night (731.2a); if it's night and they cast two or more → day (731.2b);
  neither → no check (731.2c).
- **Own-turn spell count.** `Player::spells_cast_this_turn` accumulates a player's spells over a
  turn; to isolate "spells the active player cast during their own turn" (which is what 731.2
  needs), `advance_step` snapshots the active player's counter at the start of their turn
  (`Game::active_spells_at_turn_start`) and stores `current - snapshot` into
  `Game::prev_turn_active_spell_count` at cleanup **before** the per-turn reset. The day/night untap
  check reads only that snapshot, never a live counter.
  - **Follow-up resolved (Flusterstorm storm-count fix).** The per-turn spell counts are now zeroed
    for **both** players at every cleanup (the opponent's numeric counts were previously left
    standing, which over-counted Storm — see `flusterstorm.md`). Because the day/night check uses
    `prev_turn_active_spell_count` (the active player's pre-reset snapshot) and not the live
    opponent counter, this added reset does **not** affect the day↔night transitions. The day→night
    test below (the opponent takes a spell-less turn — the exact cleanup path the fix touches) was
    re-run and still transforms Outland Liberator → Frenzied Trapbreaker; the night→day and
    opponent-turn leak-guard cases read the same pre-reset snapshot and are unaffected by
    construction.
- **Enters on the correct face (CR 702.145b/d), src/systems/state_manager_statics.cpp:**
  - At permanent creation, a daybound DFC entering while it's night is marked into
    `pending_enters_transformed` so it enters back-face-up (reusing the existing transformed-entry
    path) — CR 702.145b.
  - The continuous CR 702.145d check: while a front-face-up daybound permanent is on the
    battlefield and it's neither day nor night, `become_day()` (self-limiting once day is set).

### Reused (verified, not rebuilt)
- Transform DFC machinery (`src/transform.*`, combined-script loading) flips the faces.
- Sac-to-destroy activated ability ({1}, Sac → targeted Destroy) — works on both faces.
- Attacks-trigger-with-target Destroy (Cityscape-Leveler-style) — the trigger fires and resolves.

### Fixes (general)
- **Back-face triggered abilities now fire (src/systems/state_manager_triggers.cpp).** The trigger
  scan previously always read the FRONT `CardData::abilities` and then blanket-skipped every
  ability when `perm.transformed`, so a transformed DFC fired no triggers at all and its back
  face's own triggers (Frenzied Trapbreaker's Attacks trigger) never fired. The scan now pulls
  innate abilities from whichever face is up (`backside->abilities` when transformed), so a
  transformed permanent functions with its active face's triggers (CR 712.4) — general to all DFCs.
- **`ControlledBy TriggeredDefendingPlayer` target filter (src/game_queries.cpp `eval_qualifier`).**
  Mapped to OppCtrl semantics: the attack trigger is controlled by the attacking player, and in the
  two-player engine the defending player is that controller's sole opponent.
- **Player-target false positive (src/components/ability.cpp `is_legal_target`).** `inc_players`
  was computed via `vt.find("Player")`, which matched "Player" inside the qualifier
  "TriggeredDefendingPlayer" and wrongly offered both players as targets for the attack trigger.
  Replaced with a whole-word check (`valid_tgts_names_word`), so a control qualifier embedding
  "Player" is no longer read as the Player target type.

## Behavioral decisions
- **731.2 timing:** implemented exactly as written — checked in the untap step on the previous
  turn (the just-ended turn's active player and their own-turn spell count). The own-turn delta
  (above) is the deliberate reading of "cast … during that turn": instants the player cast on the
  opponent's preceding turn do NOT count.
- Two-player scope only: the defending player / "their opponent" is the single opponent.

## Tests (train/test_harness.py, semantic `--play`)
1. **Enters → day:** casting Outland Liberator while neither → "It becomes day.", 2/2 front face. PASS
2. **Day → night (no spells):** opponent takes a turn casting nothing → next untap "It becomes
   night.", Outland Liberator transforms into Frenzied Trapbreaker (3/3, attack trigger + Nightbound). PASS
3. **Night → day (2+ own-turn spells):** controller casts two Veil of Summer on its own turn while
   night → next untap "It becomes day.", Frenzied Trapbreaker transforms into Outland Liberator. PASS
   - Leak guard: two instants cast on the OPPONENT's turn do NOT flip night→day. PASS
4. **Back attack trigger:** Frenzied Trapbreaker attacking destroys Null Rod the defender controls;
   target menu offers only the artifact (not players). "Null Rod is destroyed". PASS
5. **Sac ability:** {1}, Sac Outland Liberator → "Null Rod is destroyed". PASS
6. **Real-game scripted regression** (deck with Outland Liberator, seeds 1–4): decisive results,
   no draws, no non-fatal errors; day/night cycling + transforms occur in play. PASS
7. **Delver regression** (delver vs mav, seeds 1–2): clean decisive games — the trigger-scan
   restructure leaves non-transformed front-face triggers unchanged. PASS

## Result
Implemented at full rules fidelity. New general Day/Night + Daybound/Nightbound subsystem (CR 731 /
702.145 / 502.2) reusable by any daybound/nightbound card; back-face triggered abilities, the
defending-player target filter, and the player-target word-boundary fix are all general engine
improvements. Both faces registered in the vocab (294/295).
