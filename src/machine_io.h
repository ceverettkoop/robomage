#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include "classes/gamestate.h"
#include "classes/action.h"
#include <vector>

// QUERY line format (machine mode):
//   "QUERY: <num_choices> <f0>...<f8876> <cat0>...<cat_{N-1}> <id0>...<id_{N-1}> <ctrl0>...<ctrl_{N-1}>"
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
// NOTE: ActionChoice.description is NOT emitted in the QUERY line.
// It is stored in Query for human-readable display (GUI/CLI) only.
//
// The Python env pads all three arrays to MAX_ACTIONS slots so the full
// observation is STATE_SIZE + 3*MAX_ACTIONS floats (plus cost features).
//
// State is always serialized from the PRIORITY PLAYER'S perspective ("self").
// "Self" refers to the player who currently holds priority.
//
// NOTE: Exile zones are populated in GameState but NOT serialized.
// Add them back once cards that use exile are implemented.
//
// Fixed-size state vector layout (STATE_SIZE = 8889 floats):
//
//  [0-8]    Self player block (9 floats):
//             life/20, hand_ct/10, poison/10, mana[W,U,B,R,G,C]/10
//  [9-17]   Opponent player block (9 floats, same layout)
//  [18-29]  Current step one-hot (12 steps: UNTAP..CLEANUP)
//  [30]     1.0 if priority player is the active player (self's turn), 0.0 otherwise
//  [31]     1.0 if self is Player A, 0.0 if self is Player B
//  [32]     Stack size / 10.0
//
//  [33-2048]   Self permanents: 48 slots x 42 floats = 2016
//  [2049-4064] Opp permanents:  48 slots x 42 floats = 2016
//              Per slot: power/10, toughness/10, is_tapped, is_attacking,
//                        is_blocking, has_summoning_sickness, damage/10,
//                        controller_is_self, is_creature, is_land,
//                        card_id one-hot (N_CARD_TYPES floats)
//              Empty slots (card_vocab_idx == -1) are all zeros.
//
//  [4065-4472] Stack: 12 slots x 34 floats = 408
//              Per slot: controller_is_self(1), card_id one-hot(32), is_spell(1)
//              is_spell=1.0 for a cast spell; 0.0 for a triggered/activated ability
//
//  [4473-6520] Self graveyard: 64 slots x 32 floats = 2048
//  [6521-8568] Opp graveyard:  64 slots x 32 floats = 2048
//              Per slot: card_id one-hot (all zeros = empty)
//
//  [8569-8888] Self hand: 10 slots x 32 floats = 320
//              Per slot: card_id one-hot (all zeros = empty)

static constexpr int STATE_SIZE      = 8889;
static constexpr int N_CARD_TYPES    = 32;
static constexpr int PERM_SLOT_SIZE  = 42;  // 8 stat/combat + 2 type flags + N_CARD_TYPES
static constexpr int STACK_SLOT_SIZE = 34;  // controller_is_self(1) + card one-hot(32) + is_spell(1)
static constexpr int GY_SLOT_SIZE    = 32;  // card one-hot only

// viewer: which player's perspective to fill from. Zone::UNKNOWN defaults to the priority player.
void populate_gamestate(GameState* gs, Zone::Ownership viewer = Zone::UNKNOWN);
void populate_query(Query* q, const std::vector<LegalAction>& actions);
std::vector<float> serialize_state(const GameState* gs);

#endif /* MACHINE_IO_H */
