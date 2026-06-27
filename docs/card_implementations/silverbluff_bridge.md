# Silverbluff Bridge (vocab index 166)

## Oracle text
Silverbluff Bridge enters tapped.
Indestructible
{T}: Add {U} or {R}.

## Forge script (Source: pre-existing local `bin/resources/cardsfolder/s/silverbluff_bridge.txt`)
Key tags:
- `Types:Artifact Land`
- `R:Event$ Moved | ... | ReplaceWith$ ETBTapped` + `SVar:ETBTapped:DB$ Tap | Defined$ Self | ETB$ True` — enters tapped (already handled by the ETBTapped replacement).
- `K:Indestructible` — the keyword this implementation adds general support for.
- `A:AB$ Mana | Cost$ T | Produced$ Combo U R` — tap for {U} or {R} (Combo mana already handled).

## Engine work
The only gap was the **Indestructible** keyword (CR 702.12), which had no handler.
Implemented generally (not card-specific), honoring the script's `K:Indestructible` tag
(parsed into the keyword list by the generic `split_keywords` fallthrough in `parse.cpp`,
so no parser change was needed).

- **Shared predicate** `is_indestructible(Entity)` in `src/game_queries.h`. Reads the
  object's effective keyword list: `Creature::keywords` for a creature (rebuilt each static
  pass from `CardData::keywords` in `state_manager_statics.cpp`, so granted keywords count),
  otherwise the printed `CardData::keywords` (or `Token::keywords`) — covering a non-creature
  permanent such as this artifact land.
- **Destroy effect** `effects::destroy` / `destroy_single` (`src/effects/effect_destroy.cpp`):
  skips an indestructible target (effect resolves, permanent stays on the battlefield).
- **Mass destroy** `effects::destroy_all` (`src/effects/effect_destroy_all.cpp`): skips any
  indestructible permanent in the matched set.
- **Lethal-damage SBA** (`src/systems/state_manager.cpp`, 704.5 block): the lethal-damage /
  deathtouch branch (704.5g/h) is gated on `!is_indestructible(entity)`. The zero-toughness
  branch (704.5f) is deliberately NOT gated — it still applies to indestructible creatures.

## Behavioral decisions
- CR 702.12b: a permanent with indestructible can't be destroyed; "destroy" effects don't
  destroy it and it ignores the lethal-damage SBA (704.5g) and deathtouch SBA (704.5h).
- Indestructible does NOT prevent, and these call sites intentionally do not consult the
  predicate: sacrifice, exile, "put into graveyard", or the 0-toughness state-based action
  (CR 704.5f). Verified by test (Dismember -5/-5 and Sheoldred's Edict both still removed an
  indestructible creature).

## Tests (`train/test_harness.py`, inline hands/battlefields, semantic `--play`)
- (a) **Destroy immunity + ETB tapped + mana.** Player A's Wasteland (Destroy target nonbasic
  land) targets Player B's Silverbluff Bridge → "Silverbluff Bridge is indestructible — not
  destroyed"; Bridge stays on battlefield (Wasteland sacrificed to graveyard). Played from
  hand it enters tapped (shown "(T)").
- (b) **Lethal-damage immunity.** A 2/2 indestructible creature (temporary test card, removed
  after testing) took 3 damage from Lightning Bolt → survived with damage marked
  (`[2/2, 3dmg]`), not destroyed.
- (c) **Does NOT over-apply.** Sheoldred's Edict forced the same indestructible creature to be
  sacrificed (went to graveyard). Dismember's -5/-5 killed it via the 0-toughness SBA.
- **Regression:** scripted full games, seeds 1/2/3, UR (Silverbluff Bridge) vs a GW
  Wasteland/Abrade removal deck — all three produced a decisive winner (no draws), no
  non-fatal errors, no failed scripts. Bridge's indestructibility fired (and was logged)
  multiple times per game against opposing land destruction.

## Result
Indestructible implemented generally and correctly; Silverbluff Bridge fully functional
(enters tapped, taps for {U}/{R}, cannot be destroyed). Build clean (`make HEADLESS=TRUE`).
