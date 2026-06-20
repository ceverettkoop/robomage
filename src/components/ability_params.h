#ifndef ABILITY_PARAMS_H
#define ABILITY_PARAMS_H

#include <cstddef>
#include <string>

// Per-effect parameter blocks held by Ability's `params` variant (see ability.h).
//
// Phase 3 of the effect-dispatch refactor decomposes the Ability god struct:
// fields that are exclusive to a single effect move into one of these structs;
// fields shared across effects (amount, target(s), controller, costs, valid_tgts,
// ...) stay as direct Ability members. Effects whose data is entirely shared use
// std::monostate.
//
// NOTE: the zone/library-manipulation family (ChangeZone, ChangeZoneAll, Counter,
// Dig, Rearrange) reads an overlapping set of fields that do NOT partition per
// kind (e.g. Counter reads `destination`, Rearrange reads `may_shuffle`,
// ChangeZoneAll reads `rest_random_order`/`dig_library_position`). Those fields
// therefore remain shared on Ability rather than being split into per-kind
// alternatives. Only cleanly-exclusive groups become variant alternatives.

struct PumpParams {
    int att = 0;  // NumAtt$ — power modifier (can be negative)
    int def = 0;  // NumDef$ — toughness modifier (can be negative)
};

// Delirium-conditional damage (Unholy Heat). The base damage stays in the
// shared Ability::amount field; these capture only the delirium upgrade.
struct DamageParams {
    bool is_delirium_scale = false;  // use delirium_amount when delirium is active
    size_t delirium_amount = 0;      // damage dealt when the caster has delirium
};

// DestroyAll (e.g. Meltdown). Filter spec like "Artifact.cmcLEX".
struct DestroyAllParams {
    std::string filter = "";  // ValidCards$ — type/CMC filter for what to destroy
};

// Token creation (e.g. Cori-Steel Cutter). TokenScript$ string parsed at resolve.
struct TokenParams {
    std::string script = "";  // TokenScript$ e.g. "w_1_1_monk_prowess"
};

#endif /* ABILITY_PARAMS_H */
