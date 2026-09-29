#ifndef QUERIES_AFFECTED_H
#define QUERIES_AFFECTED_H

#include <vector>
#include "../components/zone.h"
#include "../ecs/entity.h"

struct Ability;

// ── What an effect acts on (CR 608.2c) ──────────────────────────────────────────
// Each instruction of a resolving spell or ability refers to objects and players by its own
// definition: its chosen targets, or a Defined$ reference (the source, the objects remembered so
// far, the card its source exiled, "you", ...). These resolve that reference once, for every
// effect handler.

// True if the ability's player is a Defined$ player reference (You, Player.Opponent,
// TargetedController, TriggeredActivator, TriggeredPlayer, TriggeredCardController), which
// resolve_defined_player reads.
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
