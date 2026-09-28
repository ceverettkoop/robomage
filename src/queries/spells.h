#ifndef QUERIES_SPELLS_H
#define QUERIES_SPELLS_H

#include <set>
#include <string>
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"

struct CardData;

// ── Spells (CR 112 / 601) ─────────────────────────────────────────────────────

// True if `e` is a spell that was cast via flashback. Such a spell is exiled
// (rather than sent to the graveyard) when it leaves the stack — whether it
// resolves or is countered. Single source for that "leaves-stack → exile" rule.
inline bool spell_cast_with_flashback(Entity e) {
    return global_coordinator.entity_has_component<Spell>(e) &&
           global_coordinator.GetComponent<Spell>(e).cast_with_flashback;
}

// The additional Sacrifice-a-<type> cost a spell pays as it is cast, taken from the
// card's SPELL ability `Cost$` (e.g. Natural Order's "Sac<1/Creature.Green>" →
// "Creature.Green"). Empty when the spell has no additional sacrifice cost. Mirrors how
// an activated ability stores its Sac cost on the Ability; reading it from the SPELL
// ability keeps the parser's real Cost$ tag authoritative (no retag). The cast path pays
// it with the same SACRIFICE_PERMANENT machinery activated abilities use, and cast
// legality requires a matching permanent (CR 601.2f).
std::string spell_additional_sac_spec(const CardData &cd);

// True if the spell's SPELL ability carries a VARIABLE life cost (Cost$ ... PayLife<X>): the
// amount of life paid IS the spell's X (Count$xPaid), chosen as an additional cost while casting
// (Toxic Deluge). Distinct from a fixed PayLife<N>, which is paid as a flat life cost. Reading it
// off the SPELL ability keeps the parser's real Cost$ tag authoritative (no retag); the cast path
// prompts for X, sets cur_game.x_paid, and pays that much life.
bool spell_has_variable_life_cost(const CardData &cd);

// True if the spell `spell` (an entity on the stack) can't be countered because some live
// battlefield permanent has a continuous "spells … can't be countered" replacement that covers
// it (CR 614.13/CantHappen) — e.g. Hexing Squelcher's "Spells you control can't be countered."
// Scans battlefield permanents for a battlefield-scoped CANT_BE_COUNTERED replacement and tests
// the spell against its ValidSA$ filter. The controller scope (YouCtrl/OppCtrl) is read from the
// spell's caster relative to the replacement source's controller, because a stack spell has no
// Permanent controller for the generic filter matcher to read (its YouCtrl token is a no-op off
// the battlefield). Consulted at counter-resolution time; reusable by any future can't-be-countered
// permanent. `entities` is the iterating system's mEntities (e.g. orderer->mEntities).
bool spell_uncounterable_by_static(Entity spell, const std::set<Entity> &entities);

// True if EVERY spell `player` controls is protected from being countered by an effect covering
// the player as a whole (CR 614.13/CantHappen, "spells you control can't be countered"): a
// Game::resolved_effects.cant_counter_spells_of grant (Veil of Summer), or a live battlefield CANT_BE_COUNTERED
// replacement whose ValidSA$ filter is the bare controller-scoped spell filter — "Spell.YouCtrl" on
// a permanent `player` controls (Hexing Squelcher) or "Spell.OppCtrl" on one the opponent controls.
// A spell's own "This spell can't be countered" and a type/color-narrowed filter cover only some
// spells, so they are not player-level protection (the counter-resolution path checks those per
// spell). Single source for the counter-resolution check on a spell and the observation's
// spells_cant_be_countered flag. `entities` must hold the battlefield permanents (e.g. the
// iterating system's mEntities). Defined in spells.cpp.
bool player_spells_cant_be_countered(Zone::Ownership player, const std::set<Entity> &entities);

#endif /* QUERIES_SPELLS_H */
