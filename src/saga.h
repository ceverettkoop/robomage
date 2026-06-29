#ifndef SAGA_H
#define SAGA_H

#include "components/zone.h"
#include "ecs/entity.h"

struct CardData;

// CR 714 — Saga support. A Saga uses lore counters (CR 714.2) to track its progress: it enters
// with one lore counter (714.3a) and gains one more as its controller's precombat main phase
// begins each of their turns (714.3c). When the lore counters reach a chapter number, that
// chapter's ability triggers (714.3); when they reach the Saga's final chapter number and no
// chapter ability of that Saga is still on the stack, its controller sacrifices it (714.4 SBA).
// The chapter abilities themselves live on CardData::saga_chapters (parsed from K:Chapter); the
// trigger system (state_manager_triggers.cpp) places them on the stack from the SAGA_CHAPTER
// events this module emits, and the sacrifice SBA lives in StateManager::state_based_effects.

// True if `cd` is a Saga (it has chapter abilities).
bool card_is_saga(const CardData &cd);

// Put `n` lore counters (CR 714.2) on the Saga `saga`, firing the chapter ability (CR 714.3) for
// every chapter number newly reached (was below it, now at or above it). For each fired chapter it
// increments the Saga's saga_chapters_in_flight and emits a SAGA_CHAPTER event; the trigger system
// turns that into the chapter's triggered ability on the stack. No-op if `saga` isn't a Saga.
void saga_add_lore_counters(Entity saga, int n);

// CR 714.3c turn-based action: as `active`'s precombat main phase begins, put one lore counter on
// each Saga they control. Called from the draw→first-main step transition.
void saga_put_precombat_lore_counters(Zone::Ownership active);

#endif /* SAGA_H */
