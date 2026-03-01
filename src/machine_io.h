#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include <vector>

// QUERY line format (machine mode):
//   "QUERY: <num_choices> <f0>...<f2757> <cat0>...<cat_{N-1}> <id0>...<id_{N-1}>"
//   where N = num_choices.
//
// The state vector (STATE_SIZE floats) is followed by:
//   - N ActionCategory integers (values 0-21, see ActionCategory enum).
//   - N card vocab index floats: card_vocab_index / N_CARD_TYPES for card
//     entities, or -1.0 / N_CARD_TYPES (-0.03125) as a null sentinel for
//     non-card entities (players, confirm slots, fail-to-find, empty).
//
// The Python env pads both arrays to MAX_ACTIONS slots so the full observation
// is STATE_SIZE + MAX_ACTIONS + MAX_ACTIONS floats (plus cost features).
//
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
//
//  [33-832]    Creature slots — 20 slots × 40 floats (unchanged)
//              Slots 0-9:  Player A's creatures (sorted by entity id)
//              Slots 10-19: Player B's creatures
//              Per slot: power/10, toughness/10, is_tapped, is_attacking,
//                        is_blocking, has_summoning_sickness, damage/10,
//                        controller_is_A, card_id one-hot (N_CARD_TYPES floats)
//
//  [833-1632]  Land slots — 20 slots × 40 floats (same format as creature slots)
//              Slots 0-9:  Player A's lands (sorted by entity id)
//              Slots 10-19: Player B's lands
//              Per slot: 0(power), 0(toughness), is_tapped, 0(attacking),
//                        0(blocking), 0(sickness), 0(damage),
//                        controller_is_A, card_id one-hot (N_CARD_TYPES floats)
//
//  [1633-1797] Stack slots — 5 slots × 33 floats, top first
//              Per slot: controller_is_A(1), card_id one-hot(N_CARD_TYPES)
//              Empty slots are all zeros.  Activated ability entities use
//              the source permanent's card identity.
//
//  [1798-2437] Graveyard slots — 20 slots × 32 floats
//              Slots 0-9:  Player A's graveyard (sorted by entity id)
//              Slots 10-19: Player B's graveyard
//              Per slot: card_id one-hot (N_CARD_TYPES floats)
//
//  [2438-2757] Priority player's hand — 10 slots × N_CARD_TYPES floats
//              Each slot is a one-hot card identity vector (all zeros = empty).

static constexpr int STATE_SIZE             = 2758;
static constexpr int N_CARD_TYPES           = 32;
static constexpr int MAX_BATTLEFIELD_SLOTS  = 10;  // creatures, per player
static constexpr int MAX_LAND_SLOTS         = 10;  // per player
static constexpr int MAX_STACK_DISPLAY      = 5;
static constexpr int MAX_GY_SLOTS           = 10;  // per player
static constexpr int MAX_HAND_SLOTS         = 10;
static constexpr int PERM_SLOT_SIZE         = 40;  // creature AND land slots
static constexpr int STACK_SLOT_SIZE        = 33;  // controller_is_A + card one-hot
static constexpr int GY_SLOT_SIZE           = 32;  // card one-hot only

std::vector<float> serialize_state();

#endif /* MACHINE_IO_H */
