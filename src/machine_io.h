#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include <vector>

// Fixed-size state vector layout (STATE_SIZE floats):
//
//  [0]     Player A life / 20.0
//  [1]     Player A hand size / 10.0
//  [2]     Player A poison / 10.0
//  [3-8]   Player A mana (W, U, B, R, G, C) — each count / 10.0
//  [9]     Player B life / 20.0
//  [10]    Player B hand size / 10.0
//  [11]    Player B poison / 10.0
//  [12-17] Player B mana (W, U, B, R, G, C)
//  [18-29] Current step one-hot (12 steps: UNTAP..CLEANUP)
//  [30]    Active player A's turn (1.0 = A, 0.0 = B)
//  [31]    Priority player (1.0 = A has priority, 0.0 = B)
//  [32]    Stack size / 10.0
//  [33-192] Battlefield slots — 20 slots × 8 floats each
//           Slots 0-9:  Player A's creatures (sorted by entity id, padded with 0s)
//           Slots 10-19: Player B's creatures
//           Per slot: power/10, toughness/10, is_tapped, is_attacking,
//                     is_blocking, has_summoning_sickness, damage/10, controller_is_A

static constexpr int STATE_SIZE = 193;
static constexpr int MAX_BATTLEFIELD_SLOTS = 10;  // per player

std::vector<float> serialize_state();

#endif /* MACHINE_IO_H */
