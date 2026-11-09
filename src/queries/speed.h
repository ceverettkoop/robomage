#ifndef QUERIES_SPEED_H
#define QUERIES_SPEED_H

#include "../components/player.h"
#include "../components/zone.h"

// ── Speed (CR 702.179) ───────────────────────────────────────────────────────────────────────────
//
// Player::speed, a designation (designations.def). A player has none until a rule or effect sets it
// (CR 702.179b); having none is not having 0, though an effect reading it reads 0 (CR 702.179f).

constexpr int kMaxSpeed = 4;  // CR 702.179e

inline bool has_speed(const Player &pl) { return pl.speed >= 0; }
// The speed an effect that refers to it reads (CR 702.179f).
inline int speed_of(const Player &pl) { return has_speed(pl) ? pl.speed : 0; }
inline bool has_max_speed(const Player &pl) { return pl.speed == kMaxSpeed; }

// `who` gets a speed of 1 from start your engines! (CR 702.179a).
void start_speed(Zone::Ownership who);
// `who`'s speed increases by `by`, from 0 if they have none (CR 702.179c), to at most 4.
void increase_speed(Zone::Ownership who, int by);

#endif /* QUERIES_SPEED_H */
