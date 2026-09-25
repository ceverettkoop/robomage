# Observation rework plan (draft, 2026-09-24)

Status: **design under refinement — nothing implemented.** This is a single `STATE_SIZE` layout
break: every checkpoint and shard is invalidated, so land all items together.

Goal: train on the board as observed. Remove the action-history block, which lets the net
memorize sequences and leaks hidden information. Replace the few real facts it carried with
explicit, rules-grounded fields.

---

## 1. Remove the action-history block

**Current:** `[HIST_START, HIST_START + 512)` = 128 entries × {category, card id, is_self,
turn/50}, newest first. Written by `record_chosen_action` (`src/input_logger.cpp:293`) →
`Game::record_action` (`src/classes/game.cpp:67`); serialized in `src/machine_io.cpp:~742`. The
extractor embeds only the newest 32 (`_HIST_RECENT_K`, `train/extractor.py:265`, `hist_recent`).

**Why remove it — it leaks hidden information.** Entries are not masked per viewer, so the
opponent's private choices reach the net with card ids:

| Opp action (category) | Leak | Emitted at |
|---|---|---|
| `SEARCH_LIBRARY` | the card they tutored (non-revealing tutors included) | `components/ability.cpp:203` |
| `BOTTOM_DECK_CARD` (mulligan) | the exact cards they bottomed | `game_driver.cpp:803` |
| scry keep/bottom, surveil, `TOP_LIBRARY` put-backs (Brainstorm/Ponder) | those card ids | `effect_scry.cpp:77`, `effect_surveil.cpp:77`, `effect_rearrange_top_of_library.cpp:60` |
| `DIG_CHOICE` | the card they picked | `effect_dig.cpp:165` |
| `SIDEBOARD_IN/OUT` | their sideboard swaps (the history range is kept for sideboard context, `actor/obs_builder.cpp:152`) | `game_driver.cpp:1101` |
| an opp `PASS_PRIORITY` entry existing at all | that they had a legal non-pass action (forced passes at `game_driver.cpp:626` are never recorded) | — |

The history also contradicts MCTS determinization: each simulated world re-samples the
opponent's hidden cards, but the history keeps the real game's ids.

**What it carried that isn't on the board** (each gets a replacement below, or is accepted as lost):
- per-turn counters (spells cast, cards drawn, life lost/gained) → §4
- pending delayed triggers (Mishra's Bauble draw, etc.) → §5
- whether the opponent already passed priority this round → §2
- accepted as lost (minor):
  - tokens that died without a trace
  - bounce/blink provenance
  - mulligan count after turn 1
  - the viewer's own bottomed cards (the known-top-5 block covers only the top)
  - the opponent's "held mana and passed" tells

**Remove from:**
- `machine_io.h`: `HIST_ENTRY_SIZE`, `HIST_START`, the layout comment
- `gamestate.h`: `action_history[]`, `action_history_len`
- the serializer
- `env.py`, `extractor.py` (`hist_recent` and its category embedding use; the per-action
  encoder still uses the category embedding)
- `obs_builder.cpp` (`keep_range(HIST_START, …)`)

**Delete the engine-side ring too (decided).** The obs serializer
(`machine_io.cpp:742-764`) is its only reader; the only writer is `record_chosen_action`
(`input_logger.cpp:293-303`). Delete:
- `ActionHistoryEntry` (`game.h:145`)
- the ring fields (`game.h:777-779`)
- `Game::record_action` (`game.cpp:67`)
- `record_chosen_action` and its two call sites (`commit_choice` and the replay path)
- the `GameState` copy (`gamestate.h:190-191`)
- the `ACTION_HISTORY_SIZE` constant, if nothing else uses it

The Python `_action_history` list in `search_env.py` is a different thing (the action-index
replay log used by the analysis session, opponents and the tree-follow code) and is unaffected.

**Sideboard-phase obs: the history goes too (decided).** Drop the history range from the
sideboard keep lists: `actor/obs_builder.cpp:152` (`keep_range`) and its Python twin
`env.py:637`. Accepted loss: the previous game's tempo/sequence. Its graveyards and exile are
still kept.

**Other obs-block history consumers to update:**
- `az_net.py:149-332, 511` — ScriptTrunk `hist_recent`
- `test_obs_invariants.py:136`
- `ci_check.py:914` — the `HIST_START` parity row
- `az_inspect.py:807, 1625-1633` — block labels / size table
- `scripted_agent.py:2450` — see §6

---

## 2. Priority pass flags

`Game` already tracks `a_has_passed` / `b_has_passed` (`game.h:199`). They're set by
`pass_priority()` (`game.cpp:91`), including the forced auto-pass path, cleared by
`take_action()` and after each resolution / step change.

- `self_has_passed` — the viewer's flag.
- `opp_has_passed` — the other seat's flag. Meaning: if I pass now, the top of the stack
  resolves, or the step ends if the stack is empty.

Rules for the flags:
- **Must not distinguish forced passes from voluntary ones.** In paper Magic every pass looks
  the same, and distinguishing them would reintroduce the leak above.
- **Emit 0 outside real priority windows.** `advance_step` sets both flags true at
  UNTAP/CLEANUP (`game.cpp:558`, "hacky"), and a cleanup discard would otherwise read 1. Also
  zero them whenever a mandatory choice is pending.

`opp_has_passed` is not derivable from the current obs: with a non-empty stack, the non-active
player holding priority is ambiguous between "the active player passed" and "I just acted and
kept priority".

## 3. `is_priority_window`

1.0 when the current decision is an ordinary priority window (pass = pass priority), 0.0 for
mandatory choices and mid-resolution choices. The engine reuses `PASS_PRIORITY` as the action
type for scry and dig choices, so today this is inferred indirectly from the MandatoryChoice
one-hot plus the pending-decision source.

Notes:
- Retained priority (CR 117.3c) = `opp_has_passed == 0` and the top of the stack is mine, so
  it needs no field of its own.
- `viewer_has_priority` (extras) carries almost nothing because the obs is always serialized
  from the priority holder's view. Consider dropping or repurposing it in this same break.

## 4. Per-turn counters (both players)

These already exist on `Player` (`src/components/player.h:15-35`) but are not serialized:
- `spells_cast_this_turn`
- `noncreature_spells_cast_this_turn`
- `instant_sorcery_spells_cast_this_turn`
- `cards_drawn_this_turn` (size of the vector)
- `life_gained_this_turn`
- `life_lost_this_turn`

That's 6 per player, 12 floats. Vocab cards that depend on them: Cori-Steel Cutter (second
spell), Orcish Bowmasters (non-first draws), Tamiyo (third draw), Spectacle / "opponent lost
life this turn", Ocelot Pride (life gained this turn). Normalizers (decided): counts /10, life /20;
add them as int constants in `machine_io.h` so `gen_enums.py` mirrors them.

**Spell colors cast this turn: added** (decided 2026-09-24; Veil of Summer). 5 floats per
player (W/U/B/R/G multi-hot from `Player::spell_colors_cast_this_turn`), 10 in total.

**Two per-permanent-slot fields: added** (decided):
- `entered_this_turn` — a bit. Needed by Ocelot Pride (city's blessing copies tokens that
  entered this turn) and Phelia. Summoning sickness is only a proxy for creatures.
- `ability_resolutions_this_turn` — a count. Needed by Scythecat Cub
  (`Count$ResolvedThisTurn`). Read from the stack_manager's resolution tracking
  (`stack_manager.cpp:32`, `ability.cpp:1039`).

`PERM_SLOT_SIZE` 38 → 41 (with `pending_delayed_subject` from §5), +288 floats in total.

## 5. Pending delayed triggers

**Requirement:**
- An entry persists from registration **until it resolves** (or is countered, fizzles, or
  expires unfired).
- It is associated with the entity involved.

**Current lifecycle:**
1. Registered at six sites into `cur_game.delayed_triggers`:
   - `effect_delayed_trigger.cpp` (Bauble, Flickerwisp, Phelia)
   - `effect_token.cpp:122` (exile-at-end-of-combat tokens)
   - `effect_mobilize.cpp:76` (sacrifice at end step)
   - `effect_change_zone.cpp:147` (exile until the host leaves)
   - `effect_earthbend.cpp:85`
2. Fired in `state_manager_triggers.cpp:177`. The record is **erased** and the ability goes on
   the stack, where it looks like any other triggered ability. The association is lost here.
3. Expired unfired: `ThisTurn$` at cleanup (`game.cpp:531`), or an earthbend departure to a
   zone the trigger doesn't watch.

**Design: a derived view, not a second store.** The block is built at serialization time as the
union of:
- *waiting* — records in `delayed_triggers`;
- *on stack* — stack ability entities whose `Ability::delayed_seq != 0`.

An entry disappears when its stack entity is destroyed (resolve, counter, fizzle), so there is
no parallel list to clean up.

**Engine changes:**
- Add a single `register_delayed_trigger(DelayedTrigger dt, Entity creator)` (next to the
  shared helpers; all six sites call it). It stamps:
  - `creator` — the card whose ability set it up. This must be stored explicitly because
    `dt.ability.source` varies by site; for exile-until-host-leaves it is the *exiled card*,
    not the host.
  - `seq` — from a monotonic `Game` counter. Used for stable slot order and the stack link.
- `Ability` gains `delayed_seq` (0 = not delayed) and `delayed_creator`, copied in the fire
  block.
- Confirm snapshot/restore covers the new `Game` counter and the new `Ability` fields.

**Creator vs subject per site:**

| Site | Creator | Subject |
|---|---|---|
| Mishra's Bauble | Bauble (in the graveyard) | none (the controller draws) |
| Flickerwisp / Phelia | the creature | the exiled card (`remembered_objects`) |
| Mobilize / exile-at-end-of-combat tokens | the creating card | the created tokens (`ability.targets`) |
| Exile until host leaves | the host | the exiled card; fires on `watch_entity` = host |
| Earthbend | the earthbend card | the watched land (`watch_entity`) |

**Block:** 16 slots × 13 floats = 208, ordered by `seq`. Overflow truncates with a
debug-build stderr WARNING (see Decisions).

| Field | Meaning |
|---|---|
| `present` | slot is filled |
| `controller_is_self` | whose trigger it is |
| `state` | 0 = waiting, 1 = on the stack |
| `stack_ref` | `norm_ref` of its stack slot when on the stack, else 0 |
| `creator_card_id` | `norm_card_id` |
| `creator_ref` | `norm_ref` if the creator is on the battlefield or stack, else 0 |
| `subject_ref` | first watched / affected permanent's slot ref |
| `subject_card_id` | watched or remembered card id (sentinel if none) |
| `fire_on` one-hot ×4 | upkeep, end step, end of combat, leaves the battlefield |
| `fires_this_turn` | `fire_on_turn <= turn` |

All fields are public information.

Open: for multi-subject triggers (several mobilized tokens) only the first subject is
referenced. Optionally add a per-permanent-slot bit `doomed_by_delayed_trigger` (+96 floats);
deferred unless needed.

## 6. Complete the pending-decision source; rework the scripted Wrath check

**Decided.** The pending-decision block (`[PENDING_DECISION_START]`: source card id + source
controller is viewer) should name the asking card at every mid-flow prompt. Several prompts
deliberately run with `decision_source = 0` today. That was only for replay-corpus byte-compat:
the corpus transcript decodes the field into a `Pending:` line. The corpus is re-recorded in
this break anyway, so fill them all.

Known source-less sites (grep `decision_source = 0` / `source-less`):

| Site | Prompts | Source to use |
|---|---|---|
| `action_processor.cpp:1555-1610` (`arm_flow_query`, CAST) | the LINEAR cast prompts before announce: **X ladder** (`CHOOSE_X`), delve count, cast variants, additional/alt costs | the spell being cast |
| `action_processor.cpp:1660-2060` (ACTIVATION flow) | X ladders (`:1878`, `:1912`), TAP_PAY, sacrifice/return cost picks, ninjutsu return | the ability's source |
| `action_processor.cpp:466` | combat target sub-prompts (attack target / block target) | the chosen attacker / blocker |
| `action_processor.cpp:3688` | combat damage assignment | the attacking creature |
| `state_manager_triggers.cpp:1302` | (check) | the trigger source |
| `components/ability.cpp:448`, `:527`, `:630`, `:1641` | PAY_UNLESS, optional-trigger yes/no, etc. — pass the ambient value through | the resolving ability's source |
| `game_driver.cpp:377`, `:1060` | explicit resets — verify they are intentional (the sideboard/pregame boundary) | — |

Rule: every `ask` / `arm_*` either runs under a `PendingDecisionScope(source)` or passes an
explicit source. Add an invariant to `test_obs_invariants.py`: at any non-priority,
non-mulligan, non-sideboard decision the pending source is non-null. Enumerate any allowed
exceptions explicitly.

**Scripted agent rework.** `scripted_agent.py:~2436-2452` identifies Wrath of the Skies' cast-time
X ladder from the newest history entry (our own `CAST` of Wrath). Replace that clause with the
existing pending-source check, `_slot_card_idx(obs, _PENDING_DECISION_START) ==
_WRATH_OF_SKIES_VOCAB_IDX`, which then covers both the cast-time X ladder and the
resolution-time pay menu. Also drop `_HIST_START` from the imports (`:87`), and update the
comment. Verify with a Wrath harness line (cast → X = planned energy).

---

## 7. Re-review follow-ups (decided 2026-09-24)

### Redundancy removals

- **R1 — drop the per-permanent `[7] controller_is_self`.** The permanent blocks are split by
  controller (`machine_io.cpp:670`), so it is constant within each block.
- **R2 — drop `max_affordable_cmc_proxy`** from both mana-dev halves: `MANA_DEV_SELF_SIZE`
  11 → 10, `MANA_DEV_OPP_SIZE` 10 → 9. Python: `env.py:568-575` (`_MD_*_MAX_CMC`).
- **R3 — drop `is_post_board`** (= `game_number > 1`): `LIBRARY_CTX_SIZE` 3 → 2. Python:
  `decode.py:68, 961` (display — derive it from `game_number` instead);
  `test_obs_invariants.py:653-788` uses a post-board sample threshold — re-derive.
- **R4 — replace the 1024-float opponent revealed multi-hot** with a `revealed` bit on each
  opponent registered-decklist slot: the opp main (48) and side (16) slots go from
  (card_id, count) to (card_id, count, revealed).
  - Introduce `OPP_DECKLIST_SLOT_SIZE = 3`; the self blocks keep `DECKLIST_SLOT_SIZE = 2`.
  - The engine source (the match-scoped reveal set in `match_state`) is unchanged; only the
    serialization moves.
  - Remove `REVEALED_START` / the `N_CARD_TYPES`-wide block and the extractor's
    `revealed_encoder`; the decklist-slot encoder picks up the extra bit.
  - Consumers: `env.py`, `extractor.py`, `az_net.py`, `decode.py`, `az_inspect.py`,
    `ci_check.py`, `test_obs_invariants.py`, `actor/obs_builder.cpp`.
  - Invariant: `revealed = 1` only on filled slots.
- **R5 — hide `self_is_A` (`[34]`) from the network.** Keep it in the state vector (the Python
  drivers read `_SELF_IS_A_IDX`: `env.py:1836, 2021, 2178`; `game_driver.py:519`). Zero or mask
  it in the extractor's `global_ctx` and ScriptTrunk (`az_net.py`) so the net can't learn
  seat-specific behavior. Keep the mask in one shared place both trunks import.
- Not doing: R6 (sideboard drift encoding stays).

### Additions

- **M1 — mulligan state** (with §1 — the history was its only carrier), 3 floats:
  - `self_mulligans_taken / 7`
  - `opp_mulligans_taken / 7` (public)
  - `self_bottom_remaining / 7` (cards still to bottom; 0 outside the bottoming stage)
  
  Source: `Game::pregame` (`bottom_remaining`, the per-player mulligan counts; add the counts if
  they aren't stored). A new int constant `MULLIGAN_NORMALIZER = 7`.
- **M2 + M3 — widen the graveyard and exile slots** from a bare card id to:
  - graveyard slot (4 floats): `card_id`, `playable_by_self`, `playable_by_opp`,
    `play_expires_this_turn`
  - exile slot (5 floats): the same 4 + `counters / 10` (the counters on the exiled card:
    suspend time counters — Rift Bolt, `Game::suspend_time_counters` — and void counters —
    Dauthi Voidwalker)
  - "playable" = some play/cast permission covers the card for that player: effect-granted
    (Light Up the Stage, Emry, Ugin, Mole Man, Dauthi's chosen card, `Game::may_cast_this_turn`
    …) or static (Icetill Explorer's lands from the graveyard). Timing and cost are ignored.
  - Implement it as ONE shared predicate next to the `game_queries.h` accessors, used by both
    the legal-action enumeration and the serializer, so the obs can't disagree with the menu.
    If the enumeration currently open-codes these checks, refactor them onto the predicate.
  - `play_expires_this_turn`: 1 if the permission ends at this turn's cleanup, 0 if it lasts
    longer (Light Up the Stage "until the end of your next turn") or is static.
  - `GY_SLOT_SIZE` stops being `CARD_ID_SLOT_SIZE`: add `GY_SLOT_SIZE = 4` and
    `EXILE_SLOT_SIZE = 5`. The recency-packed invariants apply per slot.
- **M4 — player-effects block**, both players, 13 floats each:
  - `protection_from_everything` (The One Ring)
  - `cant_gain_life` (Roiling Vortex; `cant_gain_life_this_turn`)
  - hexproof-from-colors ×5 (Veil of Summer; `hexproof_from_colors_this_turn`)
  - `spells_cant_be_countered` (Veil)
  - `may_cast_sorceries_as_flash` (Teferi +1)
  - `restricted_to_sorcery_speed` (Teferi static on the opponent)
  - emblem card ids ×2 (Kaito, Tamiyo)
  - `floating_trigger_source` card id (Tamiyo +2, Forth Eorlingas!)
  
  **Prerequisite:** inventory every player-scoped / duration effect `Game` tracks (e.g.
  `cant_gain_life_this_turn`, `hexproof_from_colors_this_turn`, the `until_your_next_turn`
  grants at `game.h:651-670`, emblems, floating triggers) and map each to a field. Any effect
  found that doesn't fit the list above is **reported to the user for review** before a field
  is added or it is skipped.
- **M5 — per-permanent granted statuses** (+2 floats per slot): `cant_be_blocked_this_turn`
  (Kappa Cannoneer, Manifold Key) and `combat_damage_prevented` (Maze of Ith;
  `combat_damage_prevention_shields`, either direction).
- **M6 — per-permanent `activations_this_turn / 10`** (+1 per slot): loyalty abilities used
  (Jace, Kaito, Tamiyo, Karn, Teferi, Ugin) and `ActivationLimit$` (Scryb Ranger). It is kept
  separate from `ability_resolutions_this_turn` (§4), which counts resolutions.
- **M9 — verify only:** Thespian's Stage copying a land (e.g. Dark Depths) serializes the copied
  card's id and characteristics. If it doesn't, report it before fixing.
- Not doing: M7 (Cavern's chosen type), M8 (what the opponent knows about my hand).

### Resulting per-permanent slot

38 − 1 (R1) + 3 (§4 `entered_this_turn`, `ability_resolutions_this_turn`; §5
`pending_delayed_subject`) + 2 (M5) + 1 (M6) = **43 floats** (`PERM_SLOT_SIZE` 38 → 43). The card
id stays LAST. Update the slot comment and the `PERM_SLOT_SIZE` breakdown comment.

Out of scope, noted: the matchup tail in `OBS_SIZE` (archetype identity) is an explicit
matchup-memorization channel, kept by design for the per-deck value heads.

## Size impact (approximate)

| Change | Floats |
|---|---|
| history removed (§1) | −512 |
| pass flags (+2) + `is_priority_window` (+1) (§2–3) | +3 |
| per-turn counters (6 × 2) + spell colors (5 × 2) (§4) | +22 |
| delayed-trigger block, 16 × 13 (§5) | +208 |
| permanent slot 38 → 43, × 96 (§4, §5, R1, M5, M6) | +480 |
| `viewer_has_priority` dropped | −1 |
| R2 (−2) + R3 (−1) | −3 |
| R4: revealed multi-hot −1024, opp decklist `revealed` bits +64 | −960 |
| M1 mulligan state | +3 |
| M2/M3: graveyard slots 1 → 4 (128 × 3), exile slots 1 → 5 (128 × 4) | +896 |
| M4 player effects, 13 × 2 | +26 |
| **Net** | **≈ +162** |

## Lockstep checklist (per CLAUDE.md "Source of truth" + memory "AZ obs 4-way sync")

- [ ] `src/machine_io.h`: layout comment, widths, `OFFSET_CHAIN`, `STATE_SIZE`,
      normalizer int constants
- [ ] `src/classes/gamestate.h` + `src/machine_io.cpp` serializer (`populate_gamestate`,
      `serialize_state`)
- [ ] `src/actor/obs_builder.cpp` (sideboard `keep_range` list)
- [ ] `train/_enums.py` via `gen_enums.py` (`make`)
- [ ] `train/env.py` offset chain; `train/extractor.py` (drop `hist_recent`; add encoders for
      the new scalars and the delayed-trigger slots, with card ids through the embedding like
      the stack slots); `az_net.py` ScriptTrunk if it mirrors the extractor
- [ ] R5 seat mask shared by `extractor.py` and `az_net.py`
- [ ] `decode.py` (history, revealed, post-board, new blocks — the transcript decoder); `az_inspect.py`
      block-label table
- [ ] `train/test_obs_invariants.py`:
  - pass flags are 0 outside priority windows
  - counters are ≥ 0
  - delayed slots are packed with no holes
  - every `state=1` entry's `stack_ref` points at a stack slot whose card id is the creator's,
    or the source for the change-zone case — decide which
- [ ] replay corpus: **re-record** — the §6 pending-source fill changes the `Pending:`
      transcript lines (intentional; see `docs/ci.md` for the re-record recipe)
- [ ] `make check`, plus `ci_check --tier actor` (obs parity C++ vs Python)
- [ ] docs: CLAUDE.md bo3 state-vector notes, `docs/alphazero_status.md` if it mentions history

## Decisions so far

- Remove the obs action-history block, the engine ring, and the history in the sideboard-phase obs.
- Add `self_has_passed` / `opp_has_passed`: no forced/voluntary distinction, 0 outside
  priority windows.
- Add `is_priority_window`.
- Serialize the per-turn counters for both players.
- Add the delayed-trigger block: a derived view (waiting records ∪ tagged stack abilities);
  creator + subject; one shared registration function; entries persist until they resolve.
- Name the pending-decision source at every mid-flow prompt; the scripted Wrath check uses it.
- One layout break for everything; the corpus is re-recorded.
- §7 re-review: do R1–R5 and M1–M6, verify M9; skip R6, M7 and M8.
- Add `spell_colors_cast_this_turn` for both players (Veil of Summer).
- Delayed-trigger `creator` = the **host** for exile-until-host-leaves. The stack-link
  invariant accepts a match on the creator OR the subject card id.
- Per-turn counts are plain scaled floats: **counts /10, life gained/lost /20**
  (`LIFE_NORMALIZER`); `ability_resolutions_this_turn` /10. Thermometer ("at least N")
  encodings were considered and rejected for now; revisit only if analysis shows threshold
  misjudgment. Add a `PER_TURN_COUNT_NORMALIZER = 10` int constant in `machine_io.h`.
- All per-turn facts in the table below, including the two per-permanent fields
  (`entered_this_turn`, `ability_resolutions_this_turn`).
- **Drop `viewer_has_priority`** (extras `[EXTRAS_START + 2]`). Verified nothing relies on it:
  - C++: written only at `machine_io.cpp:463` and serialized at `:1025`.
  - Python: `env.py:503` `_EXTRAS_HAS_PRIORITY` → `decode.py:916` `_decode_extras`, which
    sets a display-only `has_priority` key that nothing reads.
  - The Python priority checks use `_SELF_IS_A_IDX`, a different field.
  - In machine queries the flag is always 1 (the obs is serialized from the priority holder).
    The exception is the sideboard phase (viewer = the sideboarding seat), where it is leftover
    noise from whoever held priority at game end.
  - Who holds priority is implied by construction; who has passed moves to the §2 flags.
  - `EXTRAS_SCALARS` 13 → 12.
- **Add the per-permanent bit `pending_delayed_subject`** (+1 float per permanent slot, 96
  total): set when the permanent is the subject of a waiting delayed trigger (watched
  permanent, or a token in the trigger's `ability.targets`, e.g. Mobilize tokens).
- **Delayed-trigger slots: 16.** Overflow truncates and prints a stderr WARNING (not fatal),
  following the `MAX_ACTIONS` truncation precedent (`machine_io.cpp:801-810`, `#ifndef
  NDEBUG`). For reference, the stack block is `MAX_STACK_DISPLAY = 12` slots
  (`gamestate.h:14`); the engine's stack itself is unbounded and truncates silently, and the
  header `stack_size/10` carries the true size.
- Pending-source exceptions (null source allowed): mulligan and mulligan bottoming,
  sideboarding, cleanup discard, declare-attackers / declare-blockers menus. **Any further
  exception found during implementation is reported to the user for review, not silently
  whitelisted.**

## Open questions for refinement

None currently. Anything found during implementation that isn't covered here — a missing
pending-source exception (§6), a player effect that doesn't fit M4, an M9 discrepancy — is
reported to the user for review.

### Per-turn facts in the vocab (from scanning the scripts of vocab cards) — all decided IN

| Fact | Readers | Where |
|---|---|---|
| spells cast this turn (per player) | Cori-Steel Cutter (2nd spell), Damping Sphere (own), Mindbreak Trap (opp ≥3), Storm (Flusterstorm: both players' sum) | §4 |
| noncreature spells cast | engine statics / triggers (`rules_modifying.cpp:106`, `state_manager_triggers.cpp:746`) | §4 |
| instant/sorcery spells cast | Arclight Phoenix | §4 |
| cards drawn this turn | Orcish Bowmasters (non-first draw-step draw), Tamiyo (3rd), Sylvan Library | §4 |
| life gained this turn | Ocelot Pride | §4 |
| life lost this turn | Kaito (opponent lost life), Spectacle (Light Up the Stage, Skewer) | §4 |
| spell colors cast this turn | Veil of Summer | §4 |
| permanent entered this turn | Ocelot Pride (blessing copies tokens), Phelia | per-permanent `entered_this_turn` |
| ability resolved N times this turn | Scythecat Cub (`Count$ResolvedThisTurn`) | per-permanent `ability_resolutions_this_turn` |
| revolt | Fatal Push | extras (present) |
| delirium / threshold | DRC, Unholy Heat, Cabal Ritual | derivable from the graveyard blocks |

## Queued follow-ups (run after the plan units, before the final `make check`)

Found while reviewing unit A's "FOR USER REVIEW" item 5 (Tabernacle at Pendrell Vale). Offering the
pay-unless choice after its object has left the battlefield is CORRECT per CR 118.12/118.12a and
608.2b–d (an untargeted ability still resolves, and its unless-cost is still offered even though the
effect can no longer find the object, CR 400.7). Do NOT suppress that choice. Two real defects to fix:

1. **Unless-cost decline labels describe the actual effect, consistently.** The decline option
   reads "Don't pay (spell is countered)" for Tabernacle's DESTROY-unless. Fix this with one shared
   label builder used by EVERY unless-cost prompt (`run_unless_loop`, `run_discard_unless` and any
   other pay/decline menu in `components/ability.cpp` and elsewhere). The builder describes what
   declining causes from the ability's category and its affected object, e.g. "Don't pay (Grizzly
   Bears is destroyed)", "Don't pay (Counterspell target is countered)". Audit every existing
   pay/decline and yes/no label for the same copy-paste mismatch and route all of them through the
   builder, rather than patching this one string.
2. **A token that ceased to exist keeps its identity at the prompt.** When the unless-cost's object
   (or the ability's source) was a token that left the battlefield, the prompt shows `<unknown>` and
   the pending-decision source serializes as null. Use the last-known information (name, and the
   token vocab id via `token_vocab_idx` / LKI) for both the prompt text and the pending-decision
   source. Then REMOVE the broad "pay-unless with no source" allowance from
   `test_obs_invariants.py`'s `_SOURCELESS_PROVISIONAL` table.

Repro (non-token variant): `test_harness.py --format bo1 --battlefield-a "The Tabernacle at Pendrell
Vale,Grizzly Bears,Forest" --hand-a "Forest" --hand-b "Swords to Plowshares" --battlefield-b
"Plains" --play "A:keep,B:keep,B:cast:Swords to Plowshares,B:target:Grizzly Bears@opp"
--max-decisions 12`. For the token variant, get a token under Tabernacle (e.g. an Urza's Saga
construct, or a token maker in the vocab) and remove it in response to its upkeep trigger.

3. **"Activate only once each turn" resets at every turn, not just the controller's untap.**
   The `game.cpp` untap loop (~214) clears `Ability::activations_this_turn` and
   `Permanent::loyalty_ability_activated_this_turn` only for the ACTIVE player's permanents. So
   Scryb Ranger (`ActivationLimit$ 1`, "Activate only once each turn") used on its controller's
   turn stays locked through the opponent's turn, and the obs `activations_this_turn` shows
   stale loyalty use on the opponent's turn. Fix: reset both for ALL permanents at each turn
   boundary. Loyalty (CR 606.3) is only ever activated on its controller's turn, so a per-turn
   reset is behavior-identical for it. Add a harness regression: Scryb Ranger activated on its
   controller's turn is offered again on the opponent's turn.

4. **Actor parity: replay the actor's actions and allow near-ties** (`test_actor_parity.py` only;
   runs after the 1000-decision cap change).
   - The `az_actor --dump-obs` record becomes (int32 num_choices, int32 chosen action, float32[OBS]).
     Grep for any other reader of the dump format and update it.
   - The Python controller plays the actor's recorded action at decision i, so both sides stay on
     one trajectory and obs parity covers the whole game. It still computes its own masked logits
     and records its top pick and the top-2 gap.
   - `_compare`, per decision:
     - obs must be bit-exact and num_choices equal (checked before the replay step, so an
       out-of-range recorded index fails cleanly);
     - a top pick that differs from the actor's with top-2 gap >= 1e-6 FAILS;
     - a differing pick with gap < 1e-6 is a tolerated near-tie, and the PASS line prints the
       near-tie count.
   - `test_mcts_parity` is unchanged (it compares visit counts and hasn't shown the near-tie issue).

5. **`spells_cant_be_countered` from the engine's own counter check.** The M4 flag reads only
   the Veil-style player grant (`Game::cant_counter_spells_of`). Hexing Squelcher's battlefield
   replacement `R:Event$ Counter | ValidSA$ Spell.YouCtrl | ActiveZones$ Battlefield` ("Spells
   you control can't be countered", league/wrb_energy) is the same player-level result but reads
   0 today.
   - Fix: derive the flag from ONE shared query ("is a spell controlled by this player protected
     from being countered?"), shared with the counter-resolution path if possible, so the obs and
     the rules agree.
   - Include player grants and unfiltered `Spell.YouCtrl` battlefield statics.
   - Exclude per-card or filtered statics (a card's own "This spell can't be countered", or
     type-filtered ones).
   - No layout change. Add an obs-invariant probe with Hexing Squelcher.
6. **Mirrored board view swaps every per-player block.** `game_driver.decode_human_frame` doesn't
   swap `self_this_turn` / the per-turn counters or the exile blocks (display only, not the obs).
   Audit every self/opp pair the decoder emits and swap them all; add a unit check.
7. **Monarch trigger ordering is an approved sourceless exception** (user, 2026-09-25). Move it
   from `_SOURCELESS_PROVISIONAL` to the approved list in `test_obs_invariants.py`.

Declined for now (user, 2026-09-25): a `companion_available` field; an emblem-count field.
