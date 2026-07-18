#ifndef MACHINE_IO_H
#define MACHINE_IO_H

#include "classes/gamestate.h"
#include "classes/action.h"
#include <vector>

// BQUERY format (machine mode): a text header line
// "BQUERY: <num_choices> <STATE_SIZE> <MAX_ACTIONS>\n"
// (the two trailing sizes are a runtime layout handshake — the Python driver
// asserts them against its own imported constants so a C++ layout change without
// regenerated Python constants fails loudly instead of misframing the payload)
// followed immediately by a binary payload (see cli_emit_machine_query):
//   float32[STATE_SIZE] state, int32[MAX_ACTIONS] cats,
//   float32[MAX_ACTIONS] ids, float32[MAX_ACTIONS] ctrl,
//   float32[MAX_ACTIONS] pub, int32[MAX_ACTIONS] zone,
//   int32[MAX_ACTIONS] refs.
// Per-action metadata is padded to MAX_ACTIONS; only the first num_choices
// entries are meaningful.
//
// The state vector (STATE_SIZE floats) is followed by:
//   - N ActionCategory integers (values 0-46, see ActionCategory enum).
//   - N card vocab index floats: card_vocab_index / N_CARD_TYPES for card
//     entities, or -1.0 / N_CARD_TYPES (-0.0009765625) as a null sentinel for
//     non-card entities (players, confirm slots, fail-to-find, empty).
//   - N controller_is_self floats: 1.0 if entity is controlled by the priority
//     player, 0.0 if controlled by the opponent, null sentinel for
//     non-entity actions (pass priority, confirm slots, etc.).
//   - N card_is_public floats: 1.0 if the choice's card identity is public
//     knowledge to all players (a revealed tutor, e.g. Personal Tutor), else 0.0.
//     Lets observers show the card name for an otherwise-private choice.
//   - N zone_ref integers (ActionRefZone enum): which zone/side the choice's
//     entity lives in (self/opp battlefield, hand, stack, GY, exile, the
//     player objects themselves; REF_NONE = no referenced entity). Lets the
//     policy distinguish e.g. "target the opponent" from "target their creature".
//   - N slot_ref integers: the choice's source entity resolved into the unified
//     entity-reference slot space (see below; -1 = no serialized entity). This is
//     the action<->entity join — the policy can tell WHICH of two same-name
//     permanents a targeting action refers to.
//
// NOTE: ActionChoice.description is NOT emitted in the BQUERY payload.
// It is stored in Query for human-readable display (CLI) only.
//
// The Python env pads the per-action arrays to MAX_ACTIONS slots; cats, ids,
// ctrl, zone, and refs go into the observation (STATE_SIZE + 5*MAX_ACTIONS floats
// plus cost features); pub stays a side-channel for observers.
//
// State is always serialized from the PRIORITY PLAYER'S perspective ("self").
// "Self" refers to the player who currently holds priority.
//
// ── Unified entity-reference slot space ──────────────────────────────────────
// Several fields point at OTHER serialized objects (attachments, combat pairing,
// stack targets, action sources). They all share one viewer-relative index space:
//   0-47   self permanent slots (pack order of the self-permanents block)
//   48-95  opp permanent slots
//   96-107 stack slots (96 = top of stack)
//   -1     none / not serialized (player targets, hand/GY entities, truncation)
// N_ENTITY_REF_SLOTS = 108. In the float state vector a ref is normalized via
// norm_ref(idx) = (idx + 1) / 108 — 0.0 = none, real refs in (0, 1] (avoids a
// sentinel/slot-0 collision); decode with round(v * 108) - 1. The BQUERY per-action
// refs array stays raw int32 with -1 sentinel; env.py normalizes.
//
// Fixed-size state vector layout (STATE_SIZE = 6196 floats):
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
//  [36-1859]     Self permanents: 48 slots x 38 floats = 1824
//  [1860-3683]   Opp permanents:  48 slots x 38 floats = 1824
//                Per slot (offsets within the slot):
//                  [0]  power / 10
//                  [1]  toughness / 10
//                  [2]  is_tapped
//                  [3]  is_attacking
//                  [4]  is_blocking
//                  [5]  has_summoning_sickness
//                  [6]  damage / 10
//                  [7]  controller_is_self
//                  [8]  is_creature
//                  [9]  is_land
//                  [10] loyalty / 10 (planeswalkers; 0 otherwise)
//                  [11] p1p1_net / 10 — net (+1/+1 minus -1/-1) counters, SIGNED
//                       (distinguishes persistent counters from EOT pumps; both are
//                       already folded into effective P/T above)
//                  [12] other_counters / 10 — total counters of every other kind
//                       (excl. P1P1/M1M1/LOYALTY; the card-id embedding disambiguates
//                       the kind: Saga -> lore, Chalice -> charge, ...)
//                  [13] attached_to_ref (norm_ref) — for equipment/auras: the slot of
//                       the permanent this is attached to
//                  [14] attached_by_ref (norm_ref) — for creatures: the slot of the
//                       equipment/aura attached to this
//                  [15] attack_target_ref (norm_ref) — attacked planeswalker's slot;
//                       0.0 while is_attacking means "attacking the player"
//                  [16] blocking_target_ref (norm_ref) — the attacker this blocker blocks
//                  [17] is_blocked — attacker was blocked at declare-blockers; stays
//                       blocked even if all blockers leave (CR 509.1h)
//                  [18] is_phased_out — phased-out permanents ARE serialized (with this
//                       flag set) even though the rules treat them as nonexistent
//                       (CR 702.26e), so the model can anticipate the phase-in
//                  [19-34] effective keyword multi-hot x16, OBS_KEYWORDS order
//                       (post-layer, via permanent_has_keyword)
//                  [35] chosen_name_id — normalized vocab id of Permanent::chosen_name,
//                       the card name chosen on this permanent (Pithing Needle /
//                       Disruptor Flute named card, Petrified Hamlet named land);
//                       -1/N_CARD_TYPES sentinel when no name is chosen
//                  [36] returnable_exile_id — normalized vocab id of the most recently
//                       exiled card linked to this permanent that STILL has a return path
//                       (Static Prison holding a real card, Flickerwisp/Phelia EOT blink);
//                       -1/N_CARD_TYPES sentinel when none (no return, e.g. Skyclave
//                       Apparition, or nothing exiled). See returnable_exiled_card().
//                  [37] card_id (LAST)
//                Empty slots: 35 zeros + chosen_name sentinel + returnable_exile sentinel
//                + card_id sentinel (all three -1/N_CARD_TYPES).
//
//  [3684-4127]   Stack: 12 slots x 37 floats = 444 (slot 0 = top of stack)
//                Per slot (offsets within the slot):
//                  [0]  controller_is_self
//                  [1]  card_id
//                  [2]  is_spell — 1.0 for a cast spell; 0.0 for a triggered/activated ability
//                  [3]  x_or_amount / 10 — spell: X paid at cast (Spell::x_paid);
//                       ability: Ability::amount
//                  [4-10] cast qualifiers (all 0.0 for abilities): is_copy, kicked_any,
//                       cast_with_flashback, cast_with_evoke, cast_with_escape,
//                       cast_with_offspring, cast_with_impending
//                  [11-16] chosen-mode multi-hot: 1.0 at index i if modal mode i (of the
//                       spell's charm_choices) was announced at cast (CR 601.2b); all
//                       zeros when the object is not modal
//                  [17-36] 4 target sub-slots x 5 floats. Target sub-slots carry the
//                       object's ANNOUNCED targets (public info, CR 601.2c) in
//                       announcement order — primary ability targets, then targeting
//                       sub-abilities', then each chosen mode's — truncated at 4.
//                       Per target sub-slot:
//                         [+0] present (a target occupies this sub-slot)
//                         [+1] target_is_player
//                         [+2] target_controller_is_self (controller of the targeted
//                              permanent / the targeted player himself; owner for a
//                              non-permanent card)
//                         [+3] target_slot_ref (norm_ref; 0.0 for players / non-serialized)
//                         [+4] target_card_id (LAST; -1 sentinel for players)
//                       Empty sub-slots: 4 zeros + card_id sentinel.
//
//  [4128-4191]   Self graveyard: 64 slots x 1 float = 64
//  [4192-4255]   Opp graveyard:  64 slots x 1 float = 64
//                Per slot: card_id (sentinel = empty). RECENCY order: slot 0 is the
//                most recent arrival (sorted by Zone::distance_from_top).
//
//  [4256-4319]   Self exile: 64 slots x 1 float = 64
//  [4320-4383]   Opp exile:  64 slots x 1 float = 64
//                Per slot: card_id (sentinel = empty). RECENCY order like the
//                graveyards (slot 0 = most recent arrival). All exile is public in
//                this engine, so both zones are fully serialized.
//
//  [4384-4393]   Self hand: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty)
//
//  [4394-4905]   Action history: 128 entries x 4 floats = 512 (newest first)
//                Per entry: category / ACTION_CATEGORY_MAX,
//                           card_vocab_idx / N_CARD_TYPES (or -1/N_CARD_TYPES sentinel),
//                           is_self (1.0 = viewer's action, 0.0 = opponent's),
//                           turn / 50.0 (the game turn when the action was taken)
//                Empty entries (beyond action_history_len) are all zeros.
//
//  [4906-4909]   Match context (4 floats, all 0.0 in single-game mode):
//                game_number / 3.0, self_match_wins / 2.0,
//                opp_match_wins / 2.0, is_sideboard_phase (0.0 or 1.0)
//
//  [4910-4912]   Library & post-board context (3 floats):
//                self_library_ct / 60.0, opp_library_ct / 60.0,
//                is_post_board (1.0 if game 2+ of bo3, else 0.0)
//
//  [4913]        Current game turn / 50.0
//
//  [4914-4918]   Known top-5 library cards for the viewer: 5 slots x 1 float = 5
//                Per slot: card_id (sentinel = unknown). Index 0 is the top of
//                the library. Entries are set when a card is placed on top (e.g.
//                Ponder, Brainstorm, Rearrange) and cleared when shuffled.
//
//  [4919-5942]   Opponent revealed-cards multi-hot (N_CARD_TYPES floats, zeros =
//                none seen yet). Binary "has the opponent-of-viewer ever revealed
//                card X this match"; accumulated across the games of a bo3 and
//                persists over the per-game ECS reset. Set whenever an opponent
//                card enters a public zone (battlefield/stack/graveyard/exile) or
//                is revealed by a tutor. This is the only vocab-width block.
//
//  [5943-5952]   Known opponent-hand cards: 10 slots x 1 float = 10
//                Per slot: card_id (sentinel = empty/unknown). The specific
//                identities of opponent-hand cards the viewer has had revealed
//                (Duress/Thoughtseize/tutor) and that are still in hand. Unlike
//                the multi-hot above this tracks the exact card and a slot clears
//                when that card leaves the hand for another zone.
//
//  [5953-5954]   Pending decision context: 2 floats.
//                [5953] card_id of the spell/ability currently making a
//                mid-resolution choice (target select, dig/scry/surveil pick,
//                search, discard, modal, ...; sentinel = none). Set via
//                PendingDecisionScope — the source may not be on the stack yet,
//                since targets are announced before the spell moves there
//                (CR 601.2b/c), so this is the only place the observation shows
//                WHAT is asking for the current choice.
//                [5954] 1.0 if that source's controller is the viewer, else 0.0
//                (e.g. 0.0 while choosing a card for the opponent's Thoughtseize).
//
//  [5955-5973]   Global extras (19 floats):
//                  [5955] self lands_played_this_turn / 10
//                  [5956] opp  lands_played_this_turn / 10
//                  [5957] viewer_has_priority (1.0 = the viewer holds priority)
//                  [5958] self is_monarch (CR 725)
//                  [5959] opp  is_monarch
//                  [5960] self city's blessing (CR 702.131c)
//                  [5961] opp  city's blessing
//                  [5962] self revolt (a permanent self controlled left the battlefield this turn)
//                  [5963] opp  revolt
//                  [5964] self pending extra turns / 3
//                  [5965] opp  pending extra turns / 3
//                  [5966] is_day  (CR 731.1; both 0.0 = neither)
//                  [5967] is_night
//                  [5968-5973] MandatoryChoice one-hot x6 (NONE at index 0, then
//                       DECLARE_ATTACKERS_CHOICE, DECLARE_BLOCKERS_CHOICE,
//                       CLEANUP_DISCARD, CHOOSE_ENTITY, ASSIGN_COMBAT_DAMAGE_CHOICE)
//
//  ── Deck-identity tail blocks ────────────────────────────────────────────────
//  Each slot is (card_id, count): card_id via norm_card_id (empty slot = -1
//  sentinel), count normalized /4.0 (basics may exceed 1.0). Slots are packed
//  ascending by vocab index with NO holes (a testable invariant that keeps the
//  encoding byte-stable across actors). Overflow (more distinct names than slots)
//  or a name absent from the vocab is a fatal_error, never a silent truncation.
//
//  [5974-6069]   Self LIVE library: 48 slots x (card_id, count) = 96.
//                The viewer's LIBRARY zone tallied live at serialization time, so
//                cards leaving/returning to the library are always reflected.
//                Viewer-only — the opponent's live library stays hidden.
//
//  [6070-6165]   Opponent-of-viewer STATIC maindeck: 48 slots x (card_id, count) = 96.
//  [6166-6195]   Opponent-of-viewer STATIC sideboard: 15 slots x (card_id, count) = 30.
//                The opponent's configured decklist (open-decklist ruleset), read
//                from deck_state — the post-sideboard config for game 2+ of a bo3.
//                Refreshed at game start; during the between-games sideboard phase
//                itself the lists still show the just-played game's config.

static constexpr int STATE_SIZE             = 6196;
static constexpr int N_CARD_TYPES      = 1024; // embedding vocab size (card identity is emitted as a normalized id, not a one-hot)
static constexpr int OPTION_ORDINAL_MAX = 63;  // normalizer for the per-action option_ordinal scalar
                                               // (see LegalAction::option_ordinal): mode index, X
                                               // value, color index, cast variant, top-of-library
                                               // depth, binary pole, activated-ability index within
                                               // the source's ability list (so same-permanent
                                               // activations — e.g. planeswalker loyalty abilities —
                                               // are distinguishable; synthesised equip/unattach use
                                               // 32/33); -1 = not applicable
// Number of per-action metadata arrays folded into the RL observation vector,
// in order: cats | ids | ctrl | zone_ref | slot_ref | option_ordinal. Each is
// MAX_ACTIONS wide. The `pub` array is ALSO emitted in the BQUERY payload but is a
// side-channel (observer-only), so it is NOT counted here. This is the ONE source
// of truth for the block count: src/actor/obs_builder.h derives ACTOR_OBS_SIZE from
// it, and train/env.py imports it (via _enums.py codegen) for OBS_SIZE — so the two
// obs reconstructions can never disagree on how many action blocks there are.
static constexpr int N_ACTION_OBS_BLOCKS = 6;
static constexpr int PERM_SLOT_SIZE    = 38;   // 11 stat/combat/type + 2 counters + 4 refs + 2 flags + 16 keywords + chosen-name id + returnable-exile id + card-id float
static constexpr int STACK_MODE_SLOTS  = MAX_STACK_MODES; // chosen-mode multi-hot width per stack slot
static constexpr int STACK_TGT_SLOTS   = MAX_STACK_TGTS;  // serialized targets per stack slot (truncated)
static constexpr int STACK_TGT_FIELDS  = 5;    // present + is_player + controller_is_self + slot_ref + card-id
static constexpr int STACK_SLOT_SIZE   = 3 + 1 + 7 + STACK_MODE_SLOTS + STACK_TGT_SLOTS * STACK_TGT_FIELDS;
static constexpr int GY_SLOT_SIZE      = 1;    // card-id float only
static constexpr float TURN_NORMALIZER = 50.0f; // divisor for turn fields

// Unified entity-reference slot space width (see the layout comment above):
// 48 self perms + 48 opp perms + 12 stack slots.
static constexpr int N_ENTITY_REF_SLOTS = 2 * MAX_BATTLEFIELD_SLOTS + MAX_STACK_DISPLAY;
static_assert(N_ENTITY_REF_SLOTS == 108, "entity-ref space width documented as 108");
static_assert(STACK_SLOT_SIZE == 37, "stack slot layout documented as 37 floats");

// Effective-keyword multi-hot vocabulary for the permanent slots (offsets [19-34]),
// in serialized order. Exactly the engine-implemented keyword set; queried per slot
// via permanent_has_keyword (post-layer, so granted/removed keywords are honored).
static constexpr const char* OBS_KEYWORDS[] = {
    "Flying", "Reach", "First Strike", "Double Strike", "Deathtouch", "Lifelink",
    "Trample", "Vigilance", "Menace", "Haste", "Defender", "Indestructible",
    "Hexproof", "Shroud", "Ward", "Flash",
};
static_assert(sizeof(OBS_KEYWORDS) / sizeof(OBS_KEYWORDS[0]) == N_OBS_KEYWORDS,
              "OBS_KEYWORDS must have exactly N_OBS_KEYWORDS entries");

// Card identity is serialized as a single normalized id float per slot:
//   idx >= 0 -> idx / N_CARD_TYPES ;  empty/unknown -> -1.0 / N_CARD_TYPES.
// The policy network (extractor.py) maps these back to ids and looks them up in
// a learned nn.Embedding, so the observation cost is decoupled from vocab size.
inline float norm_card_id(int idx) {
    return (idx >= 0 ? static_cast<float>(idx) : -1.0f) / static_cast<float>(N_CARD_TYPES);
}

// Normalize an entity-reference slot index (see the layout comment): -1/none -> 0.0,
// slot idx -> (idx + 1) / N_ENTITY_REF_SLOTS, so real refs live in (0, 1] and can
// never collide with the "none" sentinel. Decode: round(v * 108) - 1.
inline float norm_ref(int idx) {
    return static_cast<float>(idx + 1) / static_cast<float>(N_ENTITY_REF_SLOTS);
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
