#ifndef QUERIES_AFFECTED_H
#define QUERIES_AFFECTED_H

#include <string>
#include <vector>
#include "../components/object_spec.h"
#include "../components/zone.h"
#include "../ecs/entity.h"
#include "../object_ref.h"

struct Ability;
struct AbilityDef;

// ── What an effect acts on (CR 608.2c) ──────────────────────────────────────────
// Each instruction of a resolving spell or ability refers to objects and players by its own
// definition: its chosen targets, or a Defined$ reference (the source, the objects remembered so
// far, the card its source exiled, "you", ...). The reference is an ObjectSpec, parsed once from
// the one table in affected.cpp (name, kind, what it resolves to, object or player, who reads it);
// the parser, --parse-report and the resolvers below all read that table, so a new Defined$ name
// is one row.

// What a reference is resolved against: the "you" and the "this" of the reading ability or
// static, the ability instance itself (its chosen and inherited targets; null for a static) and
// what it bound when it triggered. Built where it is read, never kept: the source stays an
// ObjectRef, so an object that became a new one (CR 400.7) resolves to nothing.
struct ResolveCtx {
    Zone::Ownership controller = Zone::UNKNOWN;
    ObjectRef source;
    const Ability *ability = nullptr;
    // The players bound as the ability triggered (UNKNOWN when it bound none).
    Zone::Ownership triggered_activator = Zone::UNKNOWN;
    Zone::Ownership triggered_player = Zone::UNKNOWN;

    static ResolveCtx of(const Ability &ab);
    // A static ability's context: its controller and source, with no instance.
    static ResolveCtx of_static(Zone::Ownership controller, Entity source);
};

// The ObjectSpec a Defined$ / DefinedPlayer$ value names: its row's kind, UNREAD for a name with
// no row, DEFAULT for an empty value.
ObjectSpec parse_object_spec(const std::string &value);

// The ObjectSpec a static's AffectedDefined$ `value` names, UNREAD for a name no static reads.
ObjectSpec parse_affected_defined(const std::string &value);

// Does `ability` act on Defined$ `value`? The table's answer for --parse-report: a name without a
// row is not read, and a row limited to some abilities is read by those.
bool defined_name_read(const AbilityDef &ability, const std::string &value);

// The objects `spec` refers to, in order. An object that is gone (CR 400.7) is left out, and a
// name that refers to players, or that the reading ability does not act on, is no objects.
std::vector<Entity> resolve_objects(const ObjectSpec &spec, const ResolveCtx &ctx);

// The players `spec` refers to, active player first when it names both. A player it cannot name
// right now (a target that never was, a trigger that bound no one) is left out; a name that the
// reading ability does not act on is no players.
std::vector<Zone::Ownership> resolve_players(const ObjectSpec &spec, const ResolveCtx &ctx);

// True for the references a sub-ability takes from its parent's chosen target, which
// bind_sub_target hands down (CR 608.2c).
bool inherits_parent_target(const ObjectSpec &spec);

// True if the ability's player is a Defined$ player reference it acts on, which resolve_players
// reads: a row that names only a player, unlike Defined$ Targeted, whose target may be an object.
bool names_defined_player(const Ability &ab);

// The player an effect acts on: its Defined$ player when it names one; else the player it
// targets, when the ability itself declared that target (its own ValidTgts$, or Defined$
// Targeted / ParentTarget / Parent naming its parent's) rather than inheriting one down the
// sub-ability chain; else its controller ("you", CR 109.5).
Zone::Ownership affected_player(const Ability &ab);

// The objects an effect acts on: its chosen targets when the ability targets; else its Defined$
// objects — the source for Defined$ Self or no Defined$ (Forge's default), the inherited target(s)
// for Defined$ Targeted / ParentTarget / Parent, the triggering object bound when it triggered
// (TriggeredAttacker, TriggeredSpellAbility, TriggeredSourceSA), the remembered objects for
// Defined$ Remembered, the card its source exiled for Defined$ ExiledWith. An object that is gone
// (CR 400.7) is left out.
std::vector<Entity> affected_objects(const Ability &ab);

#endif /* QUERIES_AFFECTED_H */
