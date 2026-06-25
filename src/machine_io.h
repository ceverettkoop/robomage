#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include "classes/gamestate.h"
#include "classes/action.h"
#include <vector>

// BQUERY format (machine mode): a text header line "BQUERY: <num_choices>\n"
// followed immediately by a binary payload (see cli_emit_machine_query):
//   float32[STATE_SIZE] state, int32[MAX_ACTIONS] cats,
//   float32[MAX_ACTIONS] ids, float32[MAX_ACTIONS] ctrl,
//   float32[MAX_ACTIONS] pub.
// Per-action metadata is padded to MAX_ACTIONS; only the first num_choices
// entries are meaningful.
//
// The state vector (STATE_SIZE floats) is followed by:
//   - N ActionCategory integers (values 0-45, see ActionCategory enum).
//   - N card vocab index floats: card_vocab_index / N_CARD_TYPES for card
//     entities, or -1.0 / N_CARD_TYPES (-0.0078125) as a null sentinel for
//     non-card entities (players, confirm slots, fail-to-find, empty).
//   - N controller_is_self floats: 1.0 if entity is controlled by the priority
//     player, 0.0 if controlled by the opponent, -0.0078125 null sentinel for
//     non-entity actions (pass priority, confirm slots, etc.).
//   - N card_is_public floats: 1.0 if the choice's card identity is public
//     knowledge to all players (a revealed tutor, e.g. Personal Tutor), else 0.0.
//     Lets observers show the card name for an otherwise-private choice.
//
// NOTE: ActionChoice.description is NOT emitted in the BQUERY payload.
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
// Fixed-size state vector layout (STATE_SIZE = 2919 floats):
// Card identity is a single normalized id float per slot (see norm_card_id):
// idx/N_CARD_TYPES, or -1/N_CARD_TYPES for empty/unknown. The id is NOT a one-hot.
//
//  [0-8]      Self player block (9 floats):
//               life/20, hand_ct/10, poison/10, mana[W,U,B,R,G,C]/10
//  [9-17]     Opponent player block (9 floats, same layout)
//  [18-30]    Current step one-hot (13 steps: UNTAP..CLEANUP, includes FIRST_STRIKE_DAMAGE)
//  [31]       1.0 if priority player is the active player (self's turn), 0.0 otherwise
//  [32]       1.0 if self is Player A, 0.0 if self is Player B
//  [33]       Stack size / 10.0
//
//  [34-609]      Self permanents: 48 slots x 12 floats = 576
//  [610-1185]    Opp permanents:  48 slots x 12 floats = 576
//                Per slot: power/10, toughness/10, is_tapped, is_attacking,
//                          is_blocking, has_summoning_sickness, damage/10,
//                          controller_is_self, is_creature, is_land, loyalty/10, card_id
//                Empty slots: 11 zeros + card_id sentinel (-1/N_CARD_TYPES).
//
//  [1186-1221]   Stack: 12 slots x 3 floats = 36
//                Per slot: controller_is_self(1), card_id(1), is_spell(1)
//                is_spell=1.0 for a cast spell; 0.0 for a triggered/activated ability
//
//  [1222-1285]   Self graveyard: 64 slots x 1 float = 64
//  [1286-1349]   Opp graveyard:  64 slots x 1 float = 64
//                Per slot: card_id (sentinel = empty)
//
//  [1350-1359]   Self hand: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty)
//
//  [1360-1871]   Action history: 128 entries x 4 floats = 512 (newest first)
//                Per entry: category / ACTION_CATEGORY_MAX,
//                           card_vocab_idx / N_CARD_TYPES (or -1/N_CARD_TYPES sentinel),
//                           is_self (1.0 = viewer's action, 0.0 = opponent's),
//                           turn / 50.0 (the game turn when the action was taken)
//                Empty entries (beyond action_history_len) are all zeros.
//
//  [1872-1875]   Match context (4 floats, all 0.0 in single-game mode):
//                game_number / 3.0, self_match_wins / 2.0,
//                opp_match_wins / 2.0, is_sideboard_phase (0.0 or 1.0)
//
//  [1876-1878]   Library & post-board context (3 floats):
//                self_library_ct / 60.0, opp_library_ct / 60.0,
//                is_post_board (1.0 if game 2+ of bo3, else 0.0)
//
//  [1879]        Current game turn / 50.0
//
//  [1880-1884]   Known top-5 library cards for the viewer: 5 slots x 1 float = 5
//                Per slot: card_id (sentinel = unknown). Index 0 is the top of
//                the library. Entries are set when a card is placed on top (e.g.
//                Ponder, Brainstorm, Rearrange) and cleared when shuffled.
//
//  [1885-2908]   Opponent revealed-cards multi-hot (N_CARD_TYPES floats, zeros =
//                none seen yet). Binary "has the opponent-of-viewer ever revealed
//                card X this match"; accumulated across the games of a bo3 and
//                persists over the per-game ECS reset. Set whenever an opponent
//                card enters a public zone (battlefield/stack/graveyard/exile) or
//                is revealed by a tutor. This is the only vocab-width block.
//
//  [2909-2918]   Known opponent-hand cards: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty/unknown). The specific
//                identities of opponent-hand cards the viewer has had revealed
//                (Duress/Thoughtseize/tutor) and that are still in hand. Unlike
//                the multi-hot above this tracks the exact card and a slot clears
//                when that card leaves the hand for another zone.

static constexpr int STATE_SIZE             = 2919;
static constexpr int N_CARD_TYPES      = 1024; // embedding vocab size (card identity is emitted as a normalized id, not a one-hot)
static constexpr int PERM_SLOT_SIZE    = 12;   // 8 stat/combat + 2 type flags + loyalty + 1 card-id float
static constexpr int STACK_SLOT_SIZE   = 3;    // controller_is_self(1) + card-id float(1) + is_spell(1)
static constexpr int GY_SLOT_SIZE      = 1;    // card-id float only
static constexpr float TURN_NORMALIZER = 50.0f; // divisor for turn fields

// Card identity is serialized as a single normalized id float per slot:
//   idx >= 0 -> idx / N_CARD_TYPES ;  empty/unknown -> -1.0 / N_CARD_TYPES.
// The policy network (extractor.py) maps these back to ids and looks them up in
// a learned nn.Embedding, so the observation cost is decoupled from vocab size.
inline float norm_card_id(int idx) {
    return (idx >= 0 ? static_cast<float>(idx) : -1.0f) / static_cast<float>(N_CARD_TYPES);
}

// viewer: which player's perspective to fill from. Zone::UNKNOWN defaults to the priority player.
void populate_gamestate(GameState* gs, Zone::Ownership viewer = Zone::UNKNOWN);
void populate_query(Query* q, const std::vector<LegalAction>& actions);
std::vector<float> serialize_state(const GameState* gs);

#endif /* MACHINE_IO_H */
