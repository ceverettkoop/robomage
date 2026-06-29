# Shadowspear (vocab index 233)

## Oracle text
Equipped creature gets +1/+1 and has trample and lifelink.
{1}: Permanents your opponents control lose hexproof and indestructible until end of turn.
Equip {2}

## Forge script
Source: pre-existing local (`bin/resources/cardsfolder/s/shadowspear.txt`)

Key tags:
- `Types:Legendary Artifact Equipment`, `ManaCost:1`
- `S:Mode$ Continuous | Affected$ Creature.EquippedBy | AddPower$ 1 | AddToughness$ 1 | AddKeyword$ Trample & Lifelink` — the equip static.
- `A:AB$ AnimateAll | Cost$ 1 | ValidCards$ Permanent.OppCtrl | RemoveKeywords$ Hexproof & Indestructible` — the activated keyword-removal ability (until end of turn).
- `K:Equip:2`

## Engine work
The equip static (+1/+1, Trample & Lifelink via `Creature.EquippedBy`) and `Equip:2` were
already covered by the equipment subsystem. The gap was **`AB$ AnimateAll`**, which had no
handler. Added a real, general `AnimateAll` effect:

- `src/effects/effect_animate_all.cpp` — new handler `animate_all` + parse hook
  `parse_animate_all`. On resolution it scans every battlefield permanent matching
  `ValidCards$` (via the shared `permanent_matches_filter`, so `Permanent.OppCtrl` controller
  scoping comes for free) and, for the removal direction (`RemoveKeywords$`), inserts each named
  keyword into that permanent's `removed_keywords_eot` set. A symmetric grant direction
  (`AddKeyword$`/`Keywords$`, until end of turn via the creature `eot_keywords` bucket) is wired
  for generality but unexercised by Shadowspear.
- `EffectKind::AnimateAll` added to `effect_kind.{h,cpp}`, dispatched in `effect_table.cpp`
  (handler + parse hook), declared in `effects.h`.
- `Ability::remove_keywords` / `Ability::add_keywords` (plain members, `ability.h`).
- `Permanent::removed_keywords_eot` (`permanent.h`) — the per-permanent set of keywords
  suppressed until end of turn (CR 613, layer 6 keyword removal).
- The effective-keyword accessors in `src/game_queries.h` (`permanent_has_keyword`,
  `is_indestructible`, via the new `keyword_removed_eot` gate) treat a keyword in
  `removed_keywords_eot` as **absent**. Because Shroud/Hexproof targeting enforcement
  (`is_legal_target` in `ability.cpp`) and the indestructible checks (lethal-damage SBA in
  `state_manager.cpp`, `effect_destroy.cpp`) already route through these accessors, the loss has
  a real consequence for both creatures and noncreature permanents with no further call-site
  changes.
- `gather_active_statics` (`state_manager_statics.cpp`) also subtracts `removed_keywords_eot`
  from the rebuilt `Creature::keywords` after every grant is merged, so a direct reader of
  `cr.keywords` agrees with the accessors.
- Cleanup (`classes/game.cpp`, CLEANUP step) clears `removed_keywords_eot` for every permanent
  so the removal lapses at end of turn (CR 514.2), mirroring the existing `eot_keywords` clear.

## Rules basis
- CR 611 — continuous effects; the {1} ability creates a one-shot continuous effect lasting
  "until end of turn".
- CR 613 — layer system; keyword removal is layer 6, applied after all keyword grants.
- CR 702.11 (Hexproof) / 702.12 (Indestructible) — the removed keywords; both are honored by
  the engine (targeting and the destroy/lethal-damage SBA), so the removal is meaningful.
- CR 514.2 — "until end of turn" effects end during the cleanup step.

## Behavioral decisions
- `Permanent.OppCtrl` is matched relative to the ability's controller, so "your opponents'
  permanents" is correct in the 2-player engine.
- Removal is modeled as a per-permanent suppression set consulted by the keyword accessors,
  rather than mutating printed keywords, so it is uniform across creatures (whose `cr.keywords`
  is rebuilt each static pass) and noncreature permanents (whose keywords the accessors read
  off `CardData`/`Token`).

## Limitations / notes
None affecting Shadowspear: indestructible IS enforced by the engine (verified below), so its
removal has a real effect. Hexproof removal is likewise enforced through `is_legal_target`.

## Tests
Verified via `train/test_harness.py` (`--play`), seed 1:

1. **Equip static.** Shadowspear cast and equipped to Voice of Victory (1/3):
   "Voice of Victory gains 1/1 Trample & Lifelink" → it becomes 2/4. Attacking unblocked:
   "Voice of Victory deals 2 damage to Player B" → "Player A gains 2 life (lifelink)"
   (life 20→22). +1/+1 and lifelink confirmed; Trample keyword granted. PASS.
2. **Indestructible removal (control).** Opponent's Silverbluff Bridge (indestructible
   nonbasic land); A's Wasteland targets it → "Silverbluff Bridge is indestructible — not
   destroyed". PASS (baseline).
3. **Indestructible removal (effect).** A activates Shadowspear {1} → "1 permanent(s) lose
   Hexproof, Indestructible until end of turn"; A's Wasteland then targets the Bridge →
   "Silverbluff Bridge is destroyed". The keyword loss has a real consequence. PASS.
4. **Until-EOT expiry.** Activate {1} on turn 1 without destroying; on turn 2 (after the
   cleanup step) Wasteland targets the Bridge → "Silverbluff Bridge is indestructible — not
   destroyed". The removal correctly lapsed at cleanup; no cross-turn leak. PASS.
5. **Real-game regression.** Scripted Shadowspear deck vs delver, seeds 1/2/3,
   `--max-decisions` up to 2000: every game terminates with a decisive winner, no draws, no
   fatal/non-fatal errors; Shadowspear cast, equipped, and AnimateAll activated repeatedly
   across many turns with no crash or leak. PASS.

## Result
Done — registered in vocab (233), `train/card_costs.py` regenerated, clean headless build, all
scenarios verified.
