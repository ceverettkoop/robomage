#ifndef ACTOR_OBS_BUILDER_H
#define ACTOR_OBS_BUILDER_H

// Bit-exact, engine-side reconstruction of the Python RL observation vector.
//
// The gym env (train/env.py::_parse_bquery_payload) builds a 6922-float obs by
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

// State-vector header (machine_io.h floats [0-35]): self player block (10) |
// opp player block (10) | step one-hot (13) | is_active | self_is_a | stack_size.
// Derived here, mirroring train/env.py's _IS_ACTIVE_IDX/_SELF_IS_A_IDX chain, so
// a header layout change moves every C++ consumer together instead of leaving a
// stale bare literal behind.
constexpr int ACTOR_PLAYER_BLOCK_SIZE = 10;
constexpr int ACTOR_STEP_ONEHOT_SIZE = 13;
constexpr int ACTOR_IS_ACTIVE_IDX = 2 * ACTOR_PLAYER_BLOCK_SIZE + ACTOR_STEP_ONEHOT_SIZE;
constexpr int ACTOR_SELF_IS_A_IDX = ACTOR_IS_ACTIVE_IDX + 1;
constexpr int ACTOR_STATE_HEADER_SIZE = ACTOR_SELF_IS_A_IDX + 2;  // + stack_size
static_assert(ACTOR_SELF_IS_A_IDX == 34, "self_is_a documented at float 34 (machine_io.h)");

// obs = state | cats | ids | ctrl | zone | refs | hand_costs | bf_ability_costs.
// Must equal train/env.py::OBS_SIZE (6922). Derived from the engine layout
// constants so a layout change is caught by the static_assert, never a literal.
constexpr int ACTOR_OBS_SIZE =
    STATE_SIZE + 5 * MAX_ACTIONS +
    MAX_HAND_SLOTS * ACTOR_N_COST_FEATS +
    MAX_BATTLEFIELD_SLOTS * ACTOR_N_COST_FEATS;
static_assert(ACTOR_OBS_SIZE == 6922, "actor obs size must match env.py OBS_SIZE");

struct ActorObs {
    std::vector<float> obs;  // ACTOR_OBS_SIZE floats
    int num_choices;         // legal-action count for this decision (mask width)
};

// Build the bit-exact observation for the decision whose legal menu is `actions`.
// MUST be called at a machine-mode decision, mirroring the order the stdio path
// uses (populate_gamestate -> populate_query -> serialize_state) so the shared
// entity->slot map is consistent. During a bo3 between-games sideboard phase it
// applies the same stale-board observation mask env.py applies, so the obs stays
// bit-for-bit identical to the Python pipeline in bo3.
ActorObs build_obs(const std::vector<LegalAction>& actions);

#endif /* ACTOR_OBS_BUILDER_H */
