#ifndef DAY_NIGHT_H
#define DAY_NIGHT_H

#include "ecs/entity.h"

struct CardData;

// Day/Night designation + Daybound/Nightbound (CR 731 and 702.145). The game-wide day/night
// value lives on the Game (Game::day_night), starting at "neither". This subsystem is the single
// place that changes that value and performs the daybound/nightbound transforms it drives, so any
// daybound/nightbound card participates without card-specific code.

// True if the card's FRONT face carries the Daybound keyword — i.e. it is a transforming DFC that
// participates in the day/night cycle (daybound is on the front face, nightbound on the back —
// CR 702.145a/b/e). The front-face check identifies the whole card.
bool card_has_daybound(const CardData &front);

// "It becomes day" / "it becomes night" (CR 731.1). Sets the game's day/night designation and
// immediately performs the daybound/nightbound transforms (CR 702.145c/f): becoming night flips
// every front-face-up daybound permanent to its back (night) face; becoming day flips every
// back-face-up nightbound permanent to its front (day) face. No-op if already that designation.
void become_day();
void become_night();

// Turn-based check run as the second part of the untap step (CR 502.2 / 731.2), based on the turn
// that just ended: if it's day and the previous turn's active player cast no spells, it becomes
// night; if it's night and they cast two or more, it becomes day. Reads the spell count captured
// at the previous turn's cleanup (Game::prev_turn_active_spell_count). No-op when it's neither.
void day_night_untap_transition();

#endif /* DAY_NIGHT_H */
