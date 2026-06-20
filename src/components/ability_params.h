#ifndef ABILITY_PARAMS_H
#define ABILITY_PARAMS_H

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

#endif /* ABILITY_PARAMS_H */
