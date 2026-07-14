#include "obs_builder.h"

#include <array>
#include <cmath>
#include <cstring>

#include "classes/game.h"  // MandatoryChoice enum (mandatory-choice one-hot width)
#include "components/zone.h"
#include "error.h"
#include "game_driver.h"  // sideboard_phase / sideboard_phase_player globals
#include "gen/card_costs_gen.h"  // CARD_COST_MATRIX, CARD_ABILITY_COST_MATRIX, N_COST_FEATS

// ── Derived state-vector offsets ────────────────────────────────────────────
// Every offset below is DERIVED from the engine layout constants (never a bare
// literal), mirroring the derivation chain in train/env.py so the two stay
// locked together. The static_asserts pin the derived absolute indices against
// the values documented in src/machine_io.h.
static constexpr int SELF_PERM_START = ACTOR_STATE_HEADER_SIZE;
static constexpr int OPP_PERM_START = SELF_PERM_START + MAX_BATTLEFIELD_SLOTS * PERM_SLOT_SIZE;
static constexpr int STACK_START = OPP_PERM_START + MAX_BATTLEFIELD_SLOTS * PERM_SLOT_SIZE;
static constexpr int GY_START = STACK_START + MAX_STACK_DISPLAY * STACK_SLOT_SIZE;
static constexpr int EXILE_START = GY_START + 2 * MAX_GY_SLOTS;
static constexpr int HAND_START = EXILE_START + 2 * MAX_GY_SLOTS;
// The three id-family floats sit LAST in a permanent slot (chosen-name id, then
// returnable-exile id, then card id) — mirror env.py's _PERM_*_OFF.
static constexpr int PERM_CHOSEN_NAME_OFF = PERM_SLOT_SIZE - 3;
static constexpr int PERM_RETURNABLE_OFF = PERM_SLOT_SIZE - 2;
static constexpr int PERM_CARD_OFF = PERM_SLOT_SIZE - 1;  // card id is LAST in a perm slot
// Stack-slot layout (mirror env.py): [1] is the object card id; the announced
// target sub-slots start after ctrl+id+is_spell(3) + x/quals(1+7) + mode multi-hot.
static constexpr int STACK_TGT_START = 4 + 7 + STACK_MODE_SLOTS;  // == 17
static constexpr int STACK_TGT_ID_OFF = STACK_TGT_FIELDS - 1;     // card id is LAST per sub-slot
// Match-context block starts right after the action-history ring; its 4th float
// (offset +3) is is_sideboard_phase, which triggers the sideboard obs mask below.
static constexpr int HIST_START = HAND_START + MAX_HAND_SLOTS;
static constexpr int HIST_END = HIST_START + ACTION_HISTORY_SIZE * 4;
static constexpr int MATCH_CTX_START = HIST_END;
// Remaining tail-block offsets (mirror the derivation chain in train/env.py and
// the byte ranges in src/machine_io.h). Needed by the sideboard mask below.
static constexpr int MATCH_CTX_SIZE = 4;                       // game#, self/opp wins, is_sideboard
static constexpr int LIBRARY_CTX_START = MATCH_CTX_START + MATCH_CTX_SIZE;
static constexpr int LIBRARY_CTX_SIZE = 3;                     // self_lib/60, opp_lib/60, is_post_board
static constexpr int CUR_TURN_IDX = LIBRARY_CTX_START + LIBRARY_CTX_SIZE;
static constexpr int KNOWN_TOP_LIB_START = CUR_TURN_IDX + 1;
static constexpr int KNOWN_TOP_LIB_END = KNOWN_TOP_LIB_START + KNOWN_TOP_LIBRARY_SIZE;
static constexpr int REVEALED_START = KNOWN_TOP_LIB_END;
static constexpr int REVEALED_END = REVEALED_START + N_CARD_TYPES;
static constexpr int OPP_KNOWN_HAND_START = REVEALED_END;
static constexpr int OPP_KNOWN_HAND_END = OPP_KNOWN_HAND_START + MAX_HAND_SLOTS;
static constexpr int PENDING_DECISION_START = OPP_KNOWN_HAND_END;
static constexpr int PENDING_DECISION_END = PENDING_DECISION_START + 2;
// Global extras: 13 scalar/flag floats then the MandatoryChoice one-hot. Derive
// the one-hot width from the enum (ASSIGN_COMBAT_DAMAGE_CHOICE is the last value)
// so a new mandatory-choice kind moves this offset automatically.
static constexpr int N_MANDATORY_CHOICES = ASSIGN_COMBAT_DAMAGE_CHOICE + 1;  // == 6
static constexpr int EXTRAS_START = PENDING_DECISION_END;
static constexpr int EXTRAS_END = EXTRAS_START + 13 + N_MANDATORY_CHOICES;
// Deck-identity tail blocks: each slot is (card_id, count).
static constexpr int DECKLIST_SLOT_SIZE = 2;
static constexpr int SELF_LIVE_LIB_START = EXTRAS_END;
static constexpr int SELF_LIVE_LIB_END = SELF_LIVE_LIB_START + DECKLIST_MAIN_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_MAIN_START = SELF_LIVE_LIB_END;
static constexpr int OPP_DECK_MAIN_END = OPP_DECK_MAIN_START + DECKLIST_MAIN_SLOTS * DECKLIST_SLOT_SIZE;
static constexpr int OPP_DECK_SIDE_START = OPP_DECK_MAIN_END;
static constexpr int OPP_DECK_SIDE_END = OPP_DECK_SIDE_START + DECKLIST_SIDE_SLOTS * DECKLIST_SLOT_SIZE;

static_assert(SELF_PERM_START == 36, "self permanents start at float 36");
static_assert(HAND_START == 4384, "self hand starts at float 4384 (machine_io.h)");
static_assert(MATCH_CTX_START == 4906, "match context starts at float 4906 (machine_io.h)");
static_assert(STACK_TGT_START == 17, "stack target sub-slots start at slot offset 17");
static_assert(KNOWN_TOP_LIB_START == 4914, "known top-of-library starts at float 4914");
static_assert(REVEALED_START == 4919, "opponent revealed multi-hot starts at float 4919");
static_assert(PENDING_DECISION_START == 5953, "pending-decision context starts at float 5953");
static_assert(SELF_LIVE_LIB_START == 5974, "self live library starts at float 5974");
static_assert(OPP_DECK_MAIN_START == 6070, "opp maindeck starts at float 6070");
static_assert(OPP_DECK_SIDE_END == STATE_SIZE, "deck-identity tail must end exactly at STATE_SIZE");

// REF_ZONE_MAX (env.py / _enums.py) = highest ActionRefZone value = REF_PLAYER_OPP.
static constexpr int ACTOR_REF_ZONE_MAX = static_cast<int>(REF_PLAYER_OPP);
static_assert(ACTOR_REF_ZONE_MAX == 11, "REF_ZONE_MAX documented as 11");

// obs sub-block start offsets (must mirror env.py's writes into self._obs).
static constexpr int ACT_CATS_START = STATE_SIZE;
static constexpr int ACT_IDS_START = ACT_CATS_START + MAX_ACTIONS;
static constexpr int ACT_CTRL_START = ACT_IDS_START + MAX_ACTIONS;
static constexpr int ACT_ZONE_START = ACT_CTRL_START + MAX_ACTIONS;
static constexpr int ACT_REFS_START = ACT_ZONE_START + MAX_ACTIONS;
static constexpr int HAND_COST_START = ACT_REFS_START + MAX_ACTIONS;
static constexpr int BF_COST_START = HAND_COST_START + MAX_HAND_SLOTS * ACTOR_N_COST_FEATS;

static_assert(N_COST_FEATS == ACTOR_N_COST_FEATS, "cost matrix width mismatch");

// ── Sideboard observation mask ──────────────────────────────────────────────
// Bit-exact C++ twin of train/env.py::_build_sideboard_mask. During the bo3
// between-games sideboard phase the engine keeps the ended game's ECS alive, so
// the raw state vector describes the STALE terminal board — noise for a
// sideboarding decision. env.py zeroes every state block except the ones that
// inform sideboarding (graveyards+exile, action history, match/library/turn ctx,
// the opponent revealed multi-hot, the pending-decision context, and BOTH opponent
// static-decklist blocks), plus the self-is-A seat flag; card-id positions inside
// masked blocks are filled with the empty sentinel (-1/N_CARD_TYPES), NOT 0.0 (0.0
// decodes to real vocab index 0). We build the identical keep/fill pair once.
struct SideboardMask {
    std::array<bool, STATE_SIZE> keep{};   // value-initialized: all false
    std::array<float, STATE_SIZE> fill{};  // value-initialized: all 0.0f
};

static const SideboardMask& sideboard_mask() {
    static const SideboardMask mask = [] {
        SideboardMask m;
        auto keep_range = [&](int lo, int hi) {
            for (int i = lo; i < hi; i++) m.keep[static_cast<size_t>(i)] = true;
        };
        keep_range(GY_START, HAND_START);                   // graveyards + exile (self + opp)
        keep_range(HIST_START, HIST_END);                   // action history ring
        keep_range(MATCH_CTX_START, KNOWN_TOP_LIB_START);   // match + library ctx + current turn
        keep_range(REVEALED_START, REVEALED_END);           // opponent revealed multi-hot
        keep_range(PENDING_DECISION_START, PENDING_DECISION_END);  // pending-decision context
        keep_range(OPP_DECK_MAIN_START, OPP_DECK_SIDE_END); // opponent static decklist (both blocks)
        // The "self is Player A" seat flag must survive (it is live during the
        // sideboard phase, set from sideboard_phase_player, and every obs consumer
        // routes seats by it).
        m.keep[static_cast<size_t>(ACTOR_SELF_IS_A_IDX)] = true;

        const float null_id = -1.0f / static_cast<float>(N_CARD_TYPES);
        auto card_id_slot = [&](int i) {
            if (!m.keep[static_cast<size_t>(i)]) m.fill[static_cast<size_t>(i)] = null_id;
        };
        // All three id-family floats per permanent slot (self + opp).
        for (int s = 0; s < MAX_BATTLEFIELD_SLOTS; s++) {
            for (int off : {PERM_CHOSEN_NAME_OFF, PERM_RETURNABLE_OFF, PERM_CARD_OFF}) {
                card_id_slot(SELF_PERM_START + s * PERM_SLOT_SIZE + off);
                card_id_slot(OPP_PERM_START + s * PERM_SLOT_SIZE + off);
            }
        }
        // Stack: object card id + each announced-target sub-slot's card id.
        for (int s = 0; s < MAX_STACK_DISPLAY; s++) {
            int base = STACK_START + s * STACK_SLOT_SIZE;
            card_id_slot(base + 1);
            int tgt0 = base + STACK_TGT_START;
            for (int t = 0; t < STACK_TGT_SLOTS; t++)
                card_id_slot(tgt0 + t * STACK_TGT_FIELDS + STACK_TGT_ID_OFF);
        }
        for (int i = HAND_START; i < HAND_START + MAX_HAND_SLOTS; i++) card_id_slot(i);       // self hand
        for (int i = KNOWN_TOP_LIB_START; i < KNOWN_TOP_LIB_END; i++) card_id_slot(i);        // known top-5
        for (int i = OPP_KNOWN_HAND_START; i < OPP_KNOWN_HAND_END; i++) card_id_slot(i);      // known opp hand
        for (int s = 0; s < DECKLIST_MAIN_SLOTS; s++)                                         // self live library
            card_id_slot(SELF_LIVE_LIB_START + s * DECKLIST_SLOT_SIZE);
        return m;
    }();
    return mask;
}

// Apply the sideboard mask in place over the state region [0, STATE_SIZE): mirror
// env.py's `np.copyto(state_arr, _SB_MASK_FILL, where=~_SB_MASK_KEEP)`.
static void apply_sideboard_mask(float* o) {
    const SideboardMask& m = sideboard_mask();
    for (int i = 0; i < STATE_SIZE; i++)
        if (!m.keep[static_cast<size_t>(i)]) o[i] = m.fill[static_cast<size_t>(i)];
}

// Decode a normalized card-id float back to a vocab index. env.py uses np.rint
// (round-half-even); ids are exact multiples of 1/N_CARD_TYPES so llrintf (which
// also rounds half-to-even under the default mode) reproduces it exactly.
static int decode_card_id(float f) {
    return static_cast<int>(std::llrintf(f * static_cast<float>(N_CARD_TYPES)));
}

// Write one cost row (N_COST_FEATS floats) from `matrix` for vocab `id`; a
// negative id (empty slot) writes a zero row — exactly _gather_costs in env.py.
static void write_cost_row(float* dst, const float matrix[][N_COST_FEATS], int id) {
    if (id < 0) {
        for (int k = 0; k < N_COST_FEATS; k++) dst[k] = 0.0f;
        return;
    }
    int safe = id;
    if (safe < 0) safe = 0;
    if (safe > N_CARD_TYPES - 1) safe = N_CARD_TYPES - 1;
    for (int k = 0; k < N_COST_FEATS; k++) dst[k] = matrix[safe][k];
}

ActorObs build_obs(const std::vector<LegalAction>& actions) {
    // Mirror the machine-branch order in input_logger.cpp: populate the game
    // state (which also rebuilds the entity->slot map that populate_query and
    // serialize_state both consult), then the query, then serialize the state.
    // bo1 never sideboards, so the viewer is always the priority player.
    Zone::Ownership viewer = sideboard_phase ? sideboard_phase_player : Zone::UNKNOWN;
    static thread_local GameState local_gs;
    populate_gamestate(&local_gs, viewer);
    Query q;
    populate_query(&q, actions);
    const std::vector<float>& state = serialize_state(&local_gs);

    ActorObs result;
    result.num_choices = q.num_choices;
    result.obs.assign(static_cast<size_t>(ACTOR_OBS_SIZE), 0.0f);
    float* o = result.obs.data();

    // [0, STATE_SIZE): state vector verbatim.
    std::memcpy(o, state.data(), sizeof(float) * static_cast<size_t>(STATE_SIZE));

    // During the bo3 between-games sideboard phase the state vector describes the
    // stale terminal board; mask it down to the sideboard-relevant blocks exactly
    // like env.py (applied to [0, STATE_SIZE) BEFORE the cost gathers below so the
    // derived hand/bf cost rows zero out from the sentinel-filled ids, matching the
    // env). is_sideboard_phase is MATCH_CTX_START + 3.
    if (o[MATCH_CTX_START + 3] > 0.5f) apply_sideboard_mask(o);

    // Per-action metadata. Reproduce EXACTLY the padded arrays cli_output.cpp
    // writes into the BQUERY payload, then apply the same normalization env.py
    // applies when it folds those arrays into the obs:
    //   cats : int32 (pad 0)  -> value / ACTION_CATEGORY_MAX   (float64 divide)
    //   ids  : float32 (pad -1/N)  -> verbatim
    //   ctrl : float32 (pad -1/N)  -> verbatim
    //   zone : int32 (pad 0)  -> value / REF_ZONE_MAX          (float64 divide)
    //   refs : int32 (pad -1) -> (value + 1) / N_ENTITY_REF_SLOTS (float64 divide)
    const float id_null = -1.0f / static_cast<float>(N_CARD_TYPES);
    const float ctrl_null = -1.0f / static_cast<float>(N_CARD_TYPES);
    for (int i = 0; i < MAX_ACTIONS; i++) {
        int cat = 0;
        float idf = id_null;
        float ctrlf = ctrl_null;
        int zoneval = 0;
        int refval = -1;
        if (i < q.num_choices) {
            const ActionChoice& ac = q.choices[i];
            cat = ac.category;
            idf = ac.card_vocab_idx >= 0
                      ? static_cast<float>(ac.card_vocab_idx) / static_cast<float>(N_CARD_TYPES)
                      : id_null;
            ctrlf = (ac.zone_ref != REF_NONE)
                        ? (ac.controller_is_self ? 1.0f : 0.0f)
                        : ctrl_null;
            zoneval = static_cast<int>(ac.zone_ref);
            refval = ac.slot_ref;
        }
        // int-array normalizers go through float64 (numpy int/int -> float64 ->
        // cast to float32) so the last-ULP rounding matches env.py exactly.
        o[ACT_CATS_START + i] =
            static_cast<float>(static_cast<double>(cat) / static_cast<double>(ACTION_CATEGORY_MAX));
        o[ACT_IDS_START + i] = idf;
        o[ACT_CTRL_START + i] = ctrlf;
        o[ACT_ZONE_START + i] =
            static_cast<float>(static_cast<double>(zoneval) / static_cast<double>(ACTOR_REF_ZONE_MAX));
        o[ACT_REFS_START + i] =
            static_cast<float>(static_cast<double>(refval + 1) / static_cast<double>(N_ENTITY_REF_SLOTS));
    }

    // Hand cast-cost rows: gather by each self-hand slot's card id.
    for (int i = 0; i < MAX_HAND_SLOTS; i++) {
        int id = decode_card_id(o[HAND_START + i]);
        write_cost_row(o + HAND_COST_START + i * N_COST_FEATS, CARD_COST_MATRIX, id);
    }

    // Battlefield activated-ability cost rows: gather by each of the 48 self
    // permanent slots' card ids (the card id is LAST within each perm slot).
    for (int i = 0; i < MAX_BATTLEFIELD_SLOTS; i++) {
        int id = decode_card_id(o[SELF_PERM_START + i * PERM_SLOT_SIZE + PERM_CARD_OFF]);
        write_cost_row(o + BF_COST_START + i * N_COST_FEATS, CARD_ABILITY_COST_MATRIX, id);
    }

    return result;
}
