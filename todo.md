TODO:

## Open engine correctness issues

- Dauthi Voidwalker (vocab 88) does not work exactly as written — casts immediately (should be
  the exile-a-card-then-cast-from-exile ability with its "can't be cast unless" restriction).

- Keen-Eyed Curator (vocab 40): buff from the types among exiled cards — untested.

- Protection from a color (pro-color) — untested.

- rules_modifying::may_play_lands_from_graveyard ignores Affected$ filters (rules_modifying.cpp:136):
  returns true if the player controls ANY `may_play_from_graveyard` static, without consulting
  WHICH cards that static permits. Latent — current vocab only has Crucible-style "play all lands
  from your graveyard", so the boolean is correct today; wrong if a filtered "play only <X> from
  graveyard" card is ever added.

- delve_exiled is a game-GLOBAL vector (cur_game.delve_exiled), written during a spell's cost
  payment and cleared only at a delve creature's ETB. Two delve spells on the stack at once (cast
  one in response to another) would share/miscount the exile set — the later spell over-counts
  (its ETB sees both spells' exiles), the earlier under-counts (its set was already cleared). A
  plain reset-at-cast-start is INSUFFICIENT (it wipes the earlier spell's record); the real fix is
  per-spell scoping (store the exile set keyed by the spell entity). Latent — every current vocab
  delve card is a sorcery-speed creature (Murktide, Barrowgoyf), so two can't be on the stack at
  once.

- Surgical Extraction script carries a single B/P pip (needs an upstream refetch of the correct
  Forge script).

## Cosmetic / logging

- The One Ring damage-prevention double-logs: a prevented Ancient Tomb self-damage prints BOTH
  "N damage prevented" AND "Dealt N damage (now at X)" for the same event (life is correctly
  unchanged). Also planeswalker loyalty prints negative on lethal damage ("loyalty now -10")
  though the SBA correctly destroys it — floor-0 display only (CR 122.1b). Surfaced in fuzz
  campaign #2 (car_doomsday vs tron).

- Parse and display ability descriptions and targeting prompts (more readable human menus).

## Deferred — bigger, needs its own session

### Modal spell mode & target selection at CAST (CR 601.2b/c) — DEFERRED (known bug)

STATUS: known bug, deliberately deferred to a dedicated session (user decision 2026-07-02).

CURRENT BEHAVIOR: modal ("Choose one/two —", `SP$ Charm` / `CharmNum$`) spells choose BOTH their
mode(s) AND their target(s) at RESOLUTION, not when cast. See the explicit note in
src/effects/effect_charm.cpp:28-30 ("This engine chooses modes at resolution rather than on cast —
a simplification shared by every Charm card here"). `effects::charm` loops CharmNum$ times; each
iteration prompts CHOOSE_MODE, then selects that mode's target(s) via `select_target`, then
resolves it immediately — all during resolution.

CORRECT BEHAVIOR:
- CR 601.2b: as the spell is put on the stack (cast), its controller chooses the mode(s) —
  different modes, CharmNum$ of them.
- CR 601.2c: immediately after modes, targets are chosen (also at cast, and become public info).
  Target legality is then re-checked at resolution (CR 608.2b): a spell whose targets are all
  illegal is countered; otherwise it resolves affecting only the still-legal targets.

OBSERVABLE CONSEQUENCES OF THE CURRENT SIMPLIFICATION:
- Targets are locked in too late: an opponent cannot respond to the specific chosen mode/target
  (e.g. cannot save the creature Prismari Charm will target, because the target isn't announced
  until the spell resolves). Priority/interaction is wrong.
- Mode/target choices are hidden from the opponent until resolution (should be public on cast).

WHAT THE FIX REQUIRES:
1. Move mode + target selection into the cast-announcement flow (src/action_processor.cpp,
   alongside the existing non-modal target selection). Record which mode(s) were chosen and each
   chosen sub-ability's target/targets on the spell's Ability (the `charm_choices` already exist;
   need a "chosen" marker + populated targets per chosen mode).
2. `effects::charm` becomes a RESOLVER only: for each pre-chosen mode, re-verify target legality
   (608.2b) and resolve — no prompting at resolution.
3. Handle modal-spell COPIES (CR 707.10): a copy keeps the same modes but its controller may choose
   new targets — the copy path (src/effects/effect_copy_spell.cpp) must carry the modes and
   re-select targets.
4. Handle the "fewer legal modes than required" case at cast (CR 601.2b/e).

WHY DEFERRED (not a quick fix):
- Changes the ML DECISION SCHEDULE: the CHOOSE_MODE and SELECT_TARGET decisions move from
  resolution-time to cast-time, shifting the BQUERY decision sequence. This affects replay
  fidelity and every per-deck checkpoint (all trained against the resolution-time order). Must be
  done deliberately with the ML pipeline in mind (the state-vector/action-encoding SHAPE is
  unchanged, but the ORDER of emitted decisions changes).
- Cross-cutting: cast flow + charm resolver + copy path + legality re-check.

VOCAB MODAL CARDS TO REGRESSION-TEST: Prismari Charm (Choose one — note its damage mode "1 damage
to each of one or two targets"; the multi-target DealDamage itself was fixed separately on branch
fix/campaign-2-engine-bugs, commit b1f9eee), Prismari Command (Choose two), and any other
`SP$ Charm` / `CharmNum$` card. Ref: effect_charm.cpp:28-30.

### T3.2 cleanup-step trigger priority (rule 514.3a) — DEFERRED
No card in the current vocab has a cleanup-step trigger that uses the stack, so the "no priority
window when triggers fire during cleanup" bug cannot be observed or tested with the present card
pool (the Forge `Phase$ Cleanup` triggers in the DB are mostly `Static$ True` bookkeeping that
bypasses the stack). Deferred until a real cleanup-trigger card is added — Thawing Glaciers
(delayed "return to hand at the next cleanup step") or a discard/madness card are the natural test
vehicles. When implementing: entering CLEANUP force-sets both pass flags (game.cpp), and a
cleanup-triggered ability then resolves before any player gets priority — the fix is to reset the
pass flags and give the active player priority when triggers fire / SBAs occur in cleanup,
distinguishing the no-trigger fast path from the trigger-present path.

### Aura-on-PLAYER state-based action (CR 704.5n / 704.5m) — DEFERRED, latent
The 704.5n Aura fall-off SBA (state_manager.cpp, added with Sheltered by Ghosts) treats an Aura as
illegally-attached and destroys it whenever its `Permanent::equipped_to == 0`. But `equipped_to` is
only written when the enchant target has a Permanent component (apply_permanent_components,
state_manager_statics.cpp), so an Aura that enchants a PLAYER (a Curse, or any "Enchant player"
Aura) resolves with equipped_to == 0 and is immediately put into the graveyard — it can never stay
on the battlefield. No player-enchanting Aura is in the vocab yet, so this is latent. Proper fix
needs player-attachment modeling, which the engine deliberately lacks (see The One Ring notes: "no
player-attach in engine"). When a Curse/Enchant-player Aura is added: record the enchanted PLAYER
(not just a Permanent) as the attach target, and make the 704.5n SBA treat a legally
player-attached Aura as attached. Until then, do NOT add an "Enchant player" Aura to the vocab
without this, or it will self-destruct on resolution.

## Harness / tooling

- test_harness raises PlayResolveError when a seat-keyed `--play` spec lands on a mandatory
  cleanup-discard decision.
- Engine `--battlefield-a/-b` (and other preset list flags) split on commas → assert-crash
  (reported as DRAW) on comma-named cards. Workaround: pass comma-free names (name_to_uid strips
  punctuation, so the same card loads). The underlying assert is unfixed.
- Old-format / pre-fix machine-mode logs are NOT replayable (cleanup candidate; the new RMLOG v2
  logs replay fine — see the replay-fidelity work).

## ML / observation

- ML can only pay for spells AFTER choosing them — precludes some rare optimal lines (e.g. floating
  mana) but reduces noise. LED is the written exception (allows ML to float mana).
- ML does not know what's in exile.

## Engine robustness

### MAX_ENTITIES exhaustion / unbounded-loop detection
- MAX_ENTITIES is 1000 (lowered from 5000 to cut per-sim component-array memory; Ability=992B +
  CardData=856B per slot dominate). The cap bounds CONCURRENT living entities (IDs recycle via the
  free queue), not lifetime total. Typical peak ~250-350; token storms maybe 400-500.
- RISK: RELEASE builds compile -DNDEBUG, so the CreateEntity assert (mLivingEntityCount <
  MAX_ENTITIES) is gone. On exhaustion CreateEntity does mAvailableEntities.front() on an empty
  queue -> UB / crash. Need a real runtime guard: detect cap exhaustion and fail the game cleanly
  (loss/draw-abort with a diagnostic) instead of UB.
- Also guard against an unbounded rules-engine loop creating entities without end (token/trigger
  loop, replacement-effect ping-pong): detect runaway creation (living-entity high-water tripwire
  or per-step creation cap) and abort with a logged diagnostic so it surfaces in training.
- Before trusting 1000: instrument a debug build to record peak mLivingEntityCount across the
  engine-sanity-check decks and set the cap from the measured high-water mark (peak x2).

## Recently fixed / triaged (branch fix/campaign-2-engine-bugs, not yet merged — do NOT re-open)

- FIXED: Amped Raptor reads wasCastFromYourHandByYou via LKI (1834b69); Phyrexian pip counts 1
  toward mana value (532523d); Trinisphere {3} floor applies to impulse/free casts (43624df);
  DealDamage damages every target (Prismari Charm multi-target, b1f9eee); dies-trigger suppressed
  under ability removal via LKI (f63ac02).
- VERIFIED NOT BUGS: Goblin Bombardment "fizzle vs legal target" (false positive — legitimate
  608.2b self/dead-target fizzles); Static Prison "upkeep-after-draw" (its trigger is Phase$ Main1,
  correct); delve paying the Trinisphere floor (rules-correct payment at 601.2h); Ajani, Nacatl
  Avenger [0] damage (works — deals damage = creatures you control, not "counters added").
- Ward on spell copies: real code gap (spawn_spell_copies skips fire_targeting_hooks) but
  UNREACHABLE in current vocab (copy producers Flusterstorm/Consign target stack objects; Ward is
  only on permanents). Route copies through fire_targeting_hooks when a permanent-targeting
  copy-spell card is added.

## Dead-code / duplication cleanup (docs/dead_code_audit.md)

Tier 1 DONE (each committed separately): is_battlefield_permanent() scan helper;
target_display_name() entity->display-name; action_card_vocab_idx() entity->vocab-index; phasing
"phased-out = not on battlefield" baked into is_battlefield_permanent() + battlefield_permanents()
accessor. Tier 2/3 (safe deletions + localized dup) deferred — see the audit doc.
