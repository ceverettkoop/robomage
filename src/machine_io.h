#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include "classes/gamestate.h"
#include "classes/action.h"
#include <vector>

// BQUERY format (machine mode): a text header line "BQUERY: <num_choices>\n"
// followed immediately by a binary payload (see cli_emit_machine_query):
//   float32[STATE_SIZE] state, int32[MAX_ACTIONS] cats,
//   float32[MAX_ACTIONS] ids, float32[MAX_ACTIONS] ctrl,
//   float32[MAX_ACTIONS] pub, int32[MAX_ACTIONS] zone.
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
//   - N zone_ref integers (ActionRefZone enum): which zone/side the choice's
//     entity lives in (self/opp battlefield, hand, stack, GY, exile, the
//     player objects themselves; REF_NONE = no referenced entity). Lets the
//     policy distinguish e.g. "target the opponent" from "target their creature".
//
// NOTE: ActionChoice.description is NOT emitted in the BQUERY payload.
// It is stored in Query for human-readable display (CLI) only.
//
// The Python env pads the per-action arrays to MAX_ACTIONS slots; cats, ids,
// ctrl, and zone go into the observation (STATE_SIZE + 4*MAX_ACTIONS floats
// plus cost features); pub stays a side-channel for observers.
//
// State is always serialized from the PRIORITY PLAYER'S perspective ("self").
// "Self" refers to the player who currently holds priority.
//
// NOTE: Exile zones are populated in GameState but NOT serialized.
// Add them back once cards that use exile are implemented.
//
// Fixed-size state vector layout (STATE_SIZE = 3187 floats):
// Card identity is a single normalized id float per slot (see norm_card_id):
// idx/N_CARD_TYPES, or -1/N_CARD_TYPES for empty/unknown. The id is NOT a one-hot.
//
//  [0-9]      Self player block (10 floats):
//               life/20, hand_ct/10, poison/10, mana[W,U,B,R,G,C]/10, energy/10
//  [10-19]    Opponent player block (10 floats, same layout)
//  [20-32]    Current step one-hot (13 steps: UNTAP..CLEANUP, includes FIRST_STRIKE_DAMAGE)
//  [33]       1.0 if priority player is the active player (self's turn), 0.0 otherwise
//  [34]       1.0 if self is Player A, 0.0 if self is Player B
//  [35]       Stack size / 10.0
//
//  [36-611]      Self permanents: 48 slots x 12 floats = 576
//  [612-1187]    Opp permanents:  48 slots x 12 floats = 576
//                Per slot: power/10, toughness/10, is_tapped, is_attacking,
//                          is_blocking, has_summoning_sickness, damage/10,
//                          controller_is_self, is_creature, is_land, loyalty/10, card_id
//                Empty slots: 11 zeros + card_id sentinel (-1/N_CARD_TYPES).
//
//  [1188-1487]   Stack: 12 slots x 25 floats = 300
//                Per slot: controller_is_self(1), card_id(1), is_spell(1),
//                          chosen-mode multi-hot(6), 4 target sub-slots x 4 floats(16)
//                is_spell=1.0 for a cast spell; 0.0 for a triggered/activated ability.
//                Chosen-mode multi-hot: 1.0 at index i if modal mode i (of the spell's
//                charm_choices) was announced at cast (CR 601.2b); all zeros when the
//                object is not modal. Target sub-slots carry the object's ANNOUNCED
//                targets (public info, CR 601.2c) in announcement order — primary
//                ability targets, then targeting sub-abilities', then each chosen
//                mode's — truncated at 4. Per target sub-slot:
//                  present(1: a target occupies this sub-slot),
//                  target_is_player(1),
//                  target_controller_is_self(1: controller of the targeted permanent /
//                    the targeted player himself; owner for a non-permanent card),
//                  target_card_id(1: norm_card_id; -1 sentinel for players).
//                Empty sub-slots: 3 zeros + card_id sentinel.
//
//  [1488-1551]   Self graveyard: 64 slots x 1 float = 64
//  [1552-1615]   Opp graveyard:  64 slots x 1 float = 64
//                Per slot: card_id (sentinel = empty)
//
//  [1616-1625]   Self hand: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty)
//
//  [1626-2137]   Action history: 128 entries x 4 floats = 512 (newest first)
//                Per entry: category / ACTION_CATEGORY_MAX,
//                           card_vocab_idx / N_CARD_TYPES (or -1/N_CARD_TYPES sentinel),
//                           is_self (1.0 = viewer's action, 0.0 = opponent's),
//                           turn / 50.0 (the game turn when the action was taken)
//                Empty entries (beyond action_history_len) are all zeros.
//
//  [2138-2141]   Match context (4 floats, all 0.0 in single-game mode):
//                game_number / 3.0, self_match_wins / 2.0,
//                opp_match_wins / 2.0, is_sideboard_phase (0.0 or 1.0)
//
//  [2142-2144]   Library & post-board context (3 floats):
//                self_library_ct / 60.0, opp_library_ct / 60.0,
//                is_post_board (1.0 if game 2+ of bo3, else 0.0)
//
//  [2145]        Current game turn / 50.0
//
//  [2146-2150]   Known top-5 library cards for the viewer: 5 slots x 1 float = 5
//                Per slot: card_id (sentinel = unknown). Index 0 is the top of
//                the library. Entries are set when a card is placed on top (e.g.
//                Ponder, Brainstorm, Rearrange) and cleared when shuffled.
//
//  [2151-3174]   Opponent revealed-cards multi-hot (N_CARD_TYPES floats, zeros =
//                none seen yet). Binary "has the opponent-of-viewer ever revealed
//                card X this match"; accumulated across the games of a bo3 and
//                persists over the per-game ECS reset. Set whenever an opponent
//                card enters a public zone (battlefield/stack/graveyard/exile) or
//                is revealed by a tutor. This is the only vocab-width block.
//
//  [3175-3184]   Known opponent-hand cards: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty/unknown). The specific
//                identities of opponent-hand cards the viewer has had revealed
//                (Duress/Thoughtseize/tutor) and that are still in hand. Unlike
//                the multi-hot above this tracks the exact card and a slot clears
//                when that card leaves the hand for another zone.
//
//  [3185-3186]   Pending decision context: 2 floats.
//                [3185] card_id of the spell/ability currently making a
//                mid-resolution choice (target select, dig/scry/surveil pick,
//                search, discard, modal, ...; sentinel = none). Set via
//                PendingDecisionScope — the source may not be on the stack yet,
//                since targets are announced before the spell moves there
//                (CR 601.2b/c), so this is the only place the observation shows
//                WHAT is asking for the current choice.
//                [3186] 1.0 if that source's controller is the viewer, else 0.0
//                (e.g. 0.0 while choosing a card for the opponent's Thoughtseize).

static constexpr int STATE_SIZE             = 3187;
static constexpr int N_CARD_TYPES      = 1024; // embedding vocab size (card identity is emitted as a normalized id, not a one-hot)
static constexpr int PERM_SLOT_SIZE    = 12;   // 8 stat/combat + 2 type flags + loyalty + 1 card-id float
static constexpr int STACK_MODE_SLOTS  = MAX_STACK_MODES; // chosen-mode multi-hot width per stack slot
static constexpr int STACK_TGT_SLOTS   = MAX_STACK_TGTS;  // serialized targets per stack slot (truncated)
static constexpr int STACK_TGT_FIELDS  = 4;    // present + is_player + controller_is_self + card-id
static constexpr int STACK_SLOT_SIZE   = 3 + STACK_MODE_SLOTS + STACK_TGT_SLOTS * STACK_TGT_FIELDS;
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

// Card vocab index for an action's source entity (or a stack entity): the card's
// vocab index, TOKEN_SENTINEL for a token, or -1 for a non-card source. Walks the
// Permanent(token) -> CardData -> Ability.source(Permanent(token)/CardData) chain.
// Single source so the index logged for a chosen action (input_logger) matches the
// index emitted for that same action in the BQUERY (populate_query).
int action_card_vocab_idx(Entity e);
// Same, but honoring a modal-DFC back-face play: such an action's source entity
// carries the FRONT face's CardData, so the entity overload above would report
// the front face. When the action plays the back face, this resolves the back
// face's name instead. Use this for action emission/logging so the emitted id
// matches the face actually being played.
int action_card_vocab_idx(const LegalAction& la);
// Returns a reference to a reused internal scratch buffer (valid until the next
// serialize_state call) to avoid a ~135 KB heap allocation on every machine-mode
// decision. Consume it (e.g. fwrite) before calling serialize_state again.
const std::vector<float>& serialize_state(const GameState* gs);

#endif /* MACHINE_IO_H */
