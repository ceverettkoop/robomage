#include "obs_builder.h"

#include <cmath>
#include <cstring>

#include "components/zone.h"
#include "error.h"
#include "game_driver.h"  // sideboard_phase / sideboard_phase_player globals
#include "gen/card_costs_gen.h"  // CARD_COST_MATRIX, CARD_ABILITY_COST_MATRIX, N_COST_FEATS

// ── Derived state-vector offsets ────────────────────────────────────────────
// Every offset below is DERIVED from the engine layout constants (never a bare
// literal), mirroring the derivation chain in train/env.py so the two stay
// locked together. The static_asserts pin the derived absolute indices against
// the values documented in src/machine_io.h.
static constexpr int HEADER_SIZE = 2 * 10 + 13 + 3;  // self(10)+opp(10)+step(13)+flags(3)
static constexpr int SELF_PERM_START = HEADER_SIZE;
static constexpr int OPP_PERM_START = SELF_PERM_START + MAX_BATTLEFIELD_SLOTS * PERM_SLOT_SIZE;
static constexpr int STACK_START = OPP_PERM_START + MAX_BATTLEFIELD_SLOTS * PERM_SLOT_SIZE;
static constexpr int GY_START = STACK_START + MAX_STACK_DISPLAY * STACK_SLOT_SIZE;
static constexpr int EXILE_START = GY_START + 2 * MAX_GY_SLOTS;
static constexpr int HAND_START = EXILE_START + 2 * MAX_GY_SLOTS;
static constexpr int PERM_CARD_OFF = PERM_SLOT_SIZE - 1;  // card id is LAST in a perm slot
// Match-context block starts right after the action-history ring; its 4th float
// (offset +3) is is_sideboard_phase. Used only to assert bo1 never sideboards.
static constexpr int HIST_START = HAND_START + MAX_HAND_SLOTS;
static constexpr int HIST_END = HIST_START + ACTION_HISTORY_SIZE * 4;
static constexpr int MATCH_CTX_START = HIST_END;

static_assert(SELF_PERM_START == 36, "self permanents start at float 36");
static_assert(HAND_START == 4384, "self hand starts at float 4384 (machine_io.h)");
static_assert(MATCH_CTX_START == 4906, "match context starts at float 4906 (machine_io.h)");

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

    // The env masks the (stale) state during bo3 sideboarding; the actor is bo1
    // only, so instead of porting that mask we hard-assert the flag is clear.
    if (o[MATCH_CTX_START + 3] > 0.5f)
        fatal_error("az_actor build_obs: is_sideboard_phase set — the sideboard "
                    "observation mask is not ported (bo1 actor should never sideboard)");

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
