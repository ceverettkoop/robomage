#ifndef ACTOR_MENU_MERGE_H
#define ACTOR_MENU_MERGE_H

// Duplicate-edge merge partition for the in-process MCTS — the C++ twin of
// train/decode.py::menu_merge_reps / MENU_MERGE_WHITELIST. The two MUST stay
// in exact lockstep: the actor parity gate (ci_check --tier actor /
// train/test_mcts_parity.py) asserts bit-identical visit counts, and a
// whitelist or decode divergence shifts every visit after the first merged
// root. Any change here must land in decode.py identically (and vice versa).
//
// Semantics: rep[i] is the LOWEST menu index whose whitelisted merge key
// equals action i's (rep[i] == i marks a representative; rep[0] == 0 always).
// Key = (category, card id, ctrl, zone_ref, ordinal), a pure integer decode of
// the six per-action obs metadata blocks — the whitelisted zones (hand /
// library / graveyard / exile) always carry slot_ref == -1 (the entity-ref
// slot map covers only battlefield + stack), so no state reads are needed.
// Cast-from-exile / play-free edges are deliberately NOT whitelisted: two
// same-name exile cards can carry different hidden ImpulseCastPermissions
// (free vs pay-life vs energy, from_suspend timing) the obs cannot see.

#include <cmath>
#include <cstdint>
#include <vector>

#include "actor/obs_builder.h"  // ACT_*_START, ACTOR_REF_ZONE_MAX
#include "classes/action.h"     // ActionCategory, ACTION_CATEGORY_MAX
#include "machine_io.h"         // N_CARD_TYPES, N_ENTITY_REF_SLOTS, OPTION_ORDINAL_MAX

// (category, zone_ref) pairs whose serialized-identical actions are
// known-interchangeable — mirror of decode.py::MENU_MERGE_WHITELIST.
// Zone values are ActionRefZone: REF_SELF_HAND=3, REF_NONE=0 (library picks
// have no zone value), REF_SELF_GY=6 / REF_OPP_GY=7, REF_SELF_EXILE=8 /
// REF_OPP_EXILE=9.
inline bool menu_merge_whitelisted(int cat, int zone) {
    using AC = ActionCategory;
    switch (static_cast<AC>(cat)) {
    case AC::CAST_SPELL:  // hand casts + GY flashback/escape/permission casts
        return zone == REF_SELF_HAND || zone == REF_SELF_GY ||
               zone == REF_OPP_GY;
    case AC::PLAY_LAND:
    case AC::DISCARD:
    case AC::BOTTOM_DECK_CARD:
    case AC::PAYING_COSTS:
        return zone == REF_SELF_HAND;
    case AC::SEARCH_LIBRARY:
    case AC::TOP_LIBRARY:
        return zone == REF_NONE;
    case AC::SELECT_TARGET:
    case AC::EXILE_FROM_YARD:
        return zone == REF_SELF_GY || zone == REF_OPP_GY;
    case AC::CHOOSE_CARD:
        return zone == REF_SELF_GY || zone == REF_OPP_GY ||
               zone == REF_SELF_EXILE || zone == REF_OPP_EXILE;
    default:
        return false;
    }
}

// Fill `out` with the merge partition for the obs's legal menu. Decodes with
// lround over float64 products, the same rounding fill_pick_meta uses (the
// normalized values are within ~1e-7 of integers, so lround and numpy's
// np.round agree). Mirror any change into decode.py::menu_merge_reps.
inline void menu_merge_reps(const float* o, int nc,
                            std::vector<int16_t>& out) {
    out.resize(static_cast<size_t>(nc));
    for (int i = 0; i < nc; i++) out[static_cast<size_t>(i)] = static_cast<int16_t>(i);
    if (nc <= 1) return;
    // Null ctrl sentinel is -1/N_CARD_TYPES; anything below half of it is
    // "no owner" (decode.py's _NULL_SENTINEL / 2 threshold).
    const double ctrl_null_thresh = (-1.0 / N_CARD_TYPES) / 2.0;
    std::vector<int> cat(static_cast<size_t>(nc)), cid(static_cast<size_t>(nc)),
        ctrl(static_cast<size_t>(nc)), zone(static_cast<size_t>(nc)),
        ord(static_cast<size_t>(nc));
    std::vector<bool> mergeable(static_cast<size_t>(nc));
    for (int i = 0; i < nc; i++) {
        const size_t s = static_cast<size_t>(i);
        cat[s] = static_cast<int>(
            std::lround(double(o[ACT_CATS_START + i]) * ACTION_CATEGORY_MAX));
        cid[s] = static_cast<int>(
            std::lround(double(o[ACT_IDS_START + i]) * N_CARD_TYPES));
        const double cv = double(o[ACT_CTRL_START + i]);
        ctrl[s] = cv < ctrl_null_thresh ? 2 : (cv > 0.5 ? 1 : 0);
        zone[s] = static_cast<int>(
            std::lround(double(o[ACT_ZONE_START + i]) * ACTOR_REF_ZONE_MAX));
        const int slot = static_cast<int>(std::lround(
                             double(o[ACT_REFS_START + i]) * N_ENTITY_REF_SLOTS)) - 1;
        ord[s] = static_cast<int>(std::lround(
                     double(o[ACT_ORDS_START + i]) * (OPTION_ORDINAL_MAX + 1))) - 1;
        const bool is_lib =
            cat[s] == static_cast<int>(ActionCategory::SEARCH_LIBRARY) ||
            cat[s] == static_cast<int>(ActionCategory::TOP_LIBRARY);
        mergeable[s] =
            cid[s] >= 0 &&
            cat[s] != static_cast<int>(ActionCategory::OTHER_CHOICE) &&
            slot == -1 && menu_merge_whitelisted(cat[s], zone[s]) &&
            (!is_lib || ctrl[s] == 2);
    }
    for (int i = 1; i < nc; i++) {
        const size_t si = static_cast<size_t>(i);
        if (!mergeable[si]) continue;
        for (int j = 0; j < i; j++) {
            const size_t sj = static_cast<size_t>(j);
            if (mergeable[sj] && cat[sj] == cat[si] && cid[sj] == cid[si] &&
                ctrl[sj] == ctrl[si] && zone[sj] == zone[si] &&
                ord[sj] == ord[si]) {
                out[si] = out[sj];
                break;
            }
        }
    }
}

#endif /* ACTOR_MENU_MERGE_H */
