#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include <vector>

// QUERY line format (machine mode):
//   "QUERY: <num_choices> <f0>...<f8684> <cat0>...<cat_{N-1}> <id0>...<id_{N-1}> <ctrl0>...<ctrl_{N-1}>"
//   where N = num_choices.
//
// The state vector (STATE_SIZE floats) is followed by:
//   - N ActionCategory integers (values 0-21, see ActionCategory enum).
//   - N card vocab index floats: card_vocab_index / N_CARD_TYPES for card
//     entities, or -1.0 / N_CARD_TYPES (-0.03125) as a null sentinel for
//     non-card entities (players, confirm slots, fail-to-find, empty).
//   - N controller_is_self floats: 1.0 if entity is controlled by the priority
//     player, 0.0 if controlled by the opponent, -0.03125 null sentinel for
//     non-entity actions (pass priority, confirm slots, etc.).
//
// The Python env pads all three arrays to MAX_ACTIONS slots so the full
// observation is STATE_SIZE + 3*MAX_ACTIONS floats (plus cost features).
//
// State is always serialized from the PRIORITY PLAYER'S perspective ("self").
// "Self" refers to the player who currently holds priority.
//
// Fixed-size state vector layout (STATE_SIZE floats):
//
//  [0]     Self (priority player) life / 20.0
//  [1]     Self hand size / 10.0
//  [2]     Self poison / 10.0
//  [3-8]   Self mana (W, U, B, R, G, C) — each count / 10.0
//  [9]     Opponent life / 20.0
//  [10]    Opponent hand size / 10.0
//  [11]    Opponent poison / 10.0
//  [12-17] Opponent mana (W, U, B, R, G, C)
//  [18-29] Current step one-hot (12 steps: UNTAP..CLEANUP)
//  [30]    1.0 if priority player is the active player (self's turn), 0.0 otherwise
//  [31]    1.0 if self is Player A (priority player identity), 0.0 if self is Player B
//  [32]    Stack size / 10.0
//
//  [33-1952]   Creature slots — 48 slots × 40 floats (24 self + 24 opp)
//              Slots 0-23:  Self's creatures (sorted by entity id)
//              Slots 24-47: Opponent's creatures
//              Per slot: power/10, toughness/10, is_tapped, is_attacking,
//                        is_blocking, has_summoning_sickness, damage/10,
//                        controller_is_self, card_id one-hot (N_CARD_TYPES floats)
//
//  [1953-3872] Land slots — 48 slots × 40 floats (same format as creature slots)
//              Slots 0-23:  Self's lands (sorted by entity id)
//              Slots 24-47: Opponent's lands
//              Per slot: 0(power), 0(toughness), is_tapped, 0(attacking),
//                        0(blocking), 0(sickness), 0(damage),
//                        controller_is_self, card_id one-hot (N_CARD_TYPES floats)
//
//  [3873-4268] Stack slots — 12 slots × 33 floats, top first
//              Per slot: controller_is_self(1), card_id one-hot(N_CARD_TYPES)
//              Empty slots are all zeros.  Activated ability entities use
//              the source permanent's card identity.
//
//  [4269-8364] Graveyard slots — 128 slots × 32 floats (64 self + 64 opp)
//              Slots 0-63:   Self's graveyard (sorted by entity id)
//              Slots 64-127: Opponent's graveyard
//              Per slot: card_id one-hot (N_CARD_TYPES floats)
//
//  [8365-8684] Priority player's hand — 10 slots × N_CARD_TYPES floats
//              Each slot is a one-hot card identity vector (all zeros = empty).

static constexpr int STATE_SIZE             = 8685;
static constexpr int N_CARD_TYPES           = 32;
static constexpr int MAX_BATTLEFIELD_SLOTS  = 24;  // creatures, per player
static constexpr int MAX_LAND_SLOTS         = 24;  // per player
static constexpr int MAX_STACK_DISPLAY      = 12;
static constexpr int MAX_GY_SLOTS           = 64;  // per player
static constexpr int MAX_HAND_SLOTS         = 10;
static constexpr int PERM_SLOT_SIZE         = 40;  // creature AND land slots
static constexpr int STACK_SLOT_SIZE        = 33;  // controller_is_self + card one-hot
static constexpr int GY_SLOT_SIZE           = 32;  // card one-hot only

std::vector<float> serialize_state();

#endif /* MACHINE_IO_H */
