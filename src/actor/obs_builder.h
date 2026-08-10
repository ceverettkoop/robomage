#ifndef ACTOR_OBS_BUILDER_H
#define ACTOR_OBS_BUILDER_H

// Bit-exact, engine-side reconstruction of the Python RL observation vector.
//
// The gym env (train/env.py::_parse_bquery_payload) builds an OBS_SIZE-float obs
// by concatenating the machine-mode BQUERY payload (the STATE_SIZE state vector +
// the padded per-action metadata arrays), appending gathered cast/ability cost
// rows, and finally appending the matchup tail (value-bucket index + the two
// archetype one-hots). The in-process AlphaZero actor never speaks the stdio
// BQUERY protocol, so it reconstructs the identical obs here from the same
// primitives (populate_gamestate / populate_query / serialize_state) plus the
// generated cost/archetype tables — see obs_builder.cpp for the exact per-block
// contract, which MIRRORS env.py so the two paths are bit-for-bit identical.

#include <vector>

#include "classes/action.h"     // LegalAction, ACTION_CATEGORY_MAX
#include "classes/gamestate.h"  // MAX_ACTIONS, MAX_HAND_SLOTS, MAX_BATTLEFIELD_SLOTS
#include "gen/archetypes_gen.h" // ARCH_N (matchup-tail width)
#include "machine_io.h"         // STATE_SIZE, N_CARD_TYPES, N_ENTITY_REF_SLOTS

// Cost-feature width (W,U,B,R,G,C,generic) — mirrors card_costs.py::_N_COST_FEATS
// and src/gen/card_costs_gen.h::N_COST_FEATS.
constexpr int ACTOR_N_COST_FEATS = 7;

// State-vector header (machine_io.h floats [0-35]): self player block | opp
// player block | step one-hot | is_active | self_is_a | stack_size. The widths
// come from machine_io.h (PLAYER_BLOCK_SIZE / STEP_ONEHOT_SIZE / HEADER_FLAGS,
// the same constants train/env.py mirrors via codegen), so a header layout change
// moves every consumer together instead of leaving a stale bare literal behind.
constexpr int ACTOR_IS_ACTIVE_IDX = 2 * PLAYER_BLOCK_SIZE + STEP_ONEHOT_SIZE;
constexpr int ACTOR_SELF_IS_A_IDX = ACTOR_IS_ACTIVE_IDX + 1;
static_assert(ACTOR_SELF_IS_A_IDX == 34, "self_is_a documented at float 34 (machine_io.h)");
static_assert(ACTOR_IS_ACTIVE_IDX + HEADER_FLAGS == STATE_HEADER_SIZE,
              "is_active/self_is_a/stack_size are the HEADER_FLAGS trailing scalars");

// Matchup tail (train/env.py's _MATCHUP_TAIL_FEATS): the raw value-bucket index
// (self_arch * ARCH_N + opp_arch) followed by one-hot(self arch) + one-hot(opp
// arch). ARCH_N comes from the generated archetype mirror, so the width tracks
// train/archetypes.py automatically.
constexpr int ACTOR_MATCHUP_TAIL_FEATS = 1 + 2 * ARCH_N;

// obs = state | cats | ids | ctrl | zone | refs | ords | hand_costs |
//       bf_ability_costs | matchup_tail.
// Must equal train/env.py::OBS_SIZE (7161). Derived from the engine layout
// constants so a layout change is caught by the static_assert, never a literal.
// The per-action metadata blocks (N_ACTION_OBS_BLOCKS: cats | ids | ctrl | zone_ref
// | slot_ref | option_ordinal; pub stays a side-channel, not in the obs) are counted
// from the shared machine_io.h constant that env.py also imports, so the block count
// has one source of truth across the C++/Python boundary.
constexpr int ACTOR_OBS_SIZE =
    STATE_SIZE + N_ACTION_OBS_BLOCKS * MAX_ACTIONS +
    MAX_HAND_SLOTS * ACTOR_N_COST_FEATS +
    MAX_BATTLEFIELD_SLOTS * ACTOR_N_COST_FEATS +
    ACTOR_MATCHUP_TAIL_FEATS;
static_assert(ACTOR_OBS_SIZE == 7161, "actor obs size must match env.py OBS_SIZE");

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
