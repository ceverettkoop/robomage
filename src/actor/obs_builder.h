#ifndef ACTOR_OBS_BUILDER_H
#define ACTOR_OBS_BUILDER_H

// Bit-exact, engine-side reconstruction of the Python RL observation vector.
//
// The gym env (train/env.py::_parse_bquery_payload) builds a 6700-float obs by
// concatenating the machine-mode BQUERY payload (the STATE_SIZE state vector +
// the padded per-action metadata arrays) and appending gathered cast/ability
// cost rows. The in-process AlphaZero actor never speaks the stdio BQUERY
// protocol, so it reconstructs the identical obs here from the same primitives
// (populate_gamestate / populate_query / serialize_state) plus the generated
// cost matrices — see obs_builder.cpp for the exact per-block contract, which
// MIRRORS env.py so the two paths are bit-for-bit identical.

#include <vector>

#include "classes/action.h"     // LegalAction, ACTION_CATEGORY_MAX
#include "classes/gamestate.h"  // MAX_ACTIONS, MAX_HAND_SLOTS, MAX_BATTLEFIELD_SLOTS
#include "machine_io.h"         // STATE_SIZE, N_CARD_TYPES, N_ENTITY_REF_SLOTS

// Cost-feature width (W,U,B,R,G,C,generic) — mirrors card_costs.py::_N_COST_FEATS
// and src/gen/card_costs_gen.h::N_COST_FEATS.
constexpr int ACTOR_N_COST_FEATS = 7;

// obs = state | cats | ids | ctrl | zone | refs | hand_costs | bf_ability_costs.
// Must equal train/env.py::OBS_SIZE (6700). Derived from the engine layout
// constants so a layout change is caught by the static_assert, never a literal.
constexpr int ACTOR_OBS_SIZE =
    STATE_SIZE + 5 * MAX_ACTIONS +
    MAX_HAND_SLOTS * ACTOR_N_COST_FEATS +
    MAX_BATTLEFIELD_SLOTS * ACTOR_N_COST_FEATS;
static_assert(ACTOR_OBS_SIZE == 6700, "actor obs size must match env.py OBS_SIZE");

struct ActorObs {
    std::vector<float> obs;  // ACTOR_OBS_SIZE floats
    int num_choices;         // legal-action count for this decision (mask width)
};

// Build the bit-exact observation for the decision whose legal menu is `actions`.
// MUST be called at a machine-mode decision, mirroring the order the stdio path
// uses (populate_gamestate -> populate_query -> serialize_state) so the shared
// entity->slot map is consistent. Fatal-errors if a sideboard phase is active
// (bo1 never sideboards; the env's stale-board mask is intentionally not ported).
ActorObs build_obs(const std::vector<LegalAction>& actions);

#endif /* ACTOR_OBS_BUILDER_H */
