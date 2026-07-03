#ifndef ACTION_PROCESSOR_H
#define ACTION_PROCESSOR_H

#include <memory>
#include <vector>
#include "classes/action.h"
#include "classes/game.h"
#include "components/ability.h"
#include "components/zone.h"
#include "ecs/entity.h"

// Forward declarations
class Orderer;

// Process a legal action selected by the user
void process_action(const LegalAction& action, Game& game, std::shared_ptr<Orderer> orderer);

// Handle the current mandatory choice (declare attackers, blockers, etc.)
void proc_mandatory_choice(Game& game, std::shared_ptr<Orderer> orderer);

// T3.10: true if some attacker this combat-damage step needs its controller to divide damage
// among 2+ blockers it cannot all kill (and hasn't already been asked). When true, the combat
// step requests ASSIGN_COMBAT_DAMAGE_CHOICE before dealing damage.
bool any_attacker_needs_damage_assignment(Game& game, std::shared_ptr<Orderer> orderer,
                                          bool first_strike_only);

// Returns true if the ability has no targeting requirement or at least one legal target exists.
bool has_legal_targets(const Ability& ability, std::shared_ptr<Orderer> orderer);

// CR 601.2c cast-legality target check across a spell's reachable modes. Returns true if every
// required target (of the primary spell ability and any targeting sub-ability) can be legally
// chosen for at least one reachable set of choices — in particular, for a Gift spell, the
// not-promised OR the promised mode (which switch which ability actually requires a target).
bool spell_has_castable_targets(const Ability& primary, std::shared_ptr<Orderer> orderer,
                                Zone::Ownership caster, bool has_gift);

// Stamp the real casting source/controller onto a spell-ability TEMPLATE (from CardData::abilities,
// whose source is 0) and its targeting sub-abilities, returning the stamped copy for the
// cast-legality target probe. Cast-time source-dependent target restrictions — protection from the
// spell's color (CR 702.16e; e.g. Emrakul vs a white spell, Scryb Ranger vs a blue spell), and the
// OppCtrl/YouCtrl perspective — are evaluated off the ability's source/controller. Without this the
// gate probes with source 0, so has_protection_from(cand, 0) is vacuously false and a protected
// creature is offered as a legal target; select_target then re-checks with the real source, finds
// none, and aborts (CR 601.2c). `card_entity` is the card being cast, which is the SAME entity that
// becomes the spell on the stack, so its source matches select_target's exactly.
Ability cast_gate_probe(const Ability& tmpl, Entity card_entity, Zone::Ownership caster);

// Prompts the active player to choose a target and sets ability.target.
// Targets are presented opponent-first so action index 0 always refers to an
// opponent entity (player or permanent), regardless of which player is casting.
// Caller must ensure has_legal_targets() is true before calling.
void select_target(Ability& ability, std::shared_ptr<Orderer> orderer, Zone::Ownership priority_player);

// General "copy a spell on the stack" routine (CR 707.10 / 707.12). Creates `count` independent
// copies of the spell entity `original` (which must be a spell currently on the stack) on top of
// the stack, controlled by `controller`. Each copy is a copy of the spell's characteristics
// (CardData/color/Ability), is NOT cast (pays no costs, fires no cast triggers), and may CHOOSE
// NEW TARGETS — each copy re-runs target selection (illegal-by-default copies with no legal
// target are simply not created). The copies are marked Spell::is_copy so they cease to exist on
// resolution. Reusable by any copy-spell effect (Replicate, storm, fork). No-op if count <= 0.
void copy_spell_on_stack(Entity original, int count, Zone::Ownership controller,
                         std::shared_ptr<Orderer> orderer);

// Evaluates ability.condition_present against ability.condition_compare for `controller`.
// Domain is battlefield permanents matching the filter's type and YouCtrl/OppCtrl qualifier,
// unless ability.condition_on_remembered is set, in which case it counts the remembered
// entities (cur_game.remembered_entities). An empty condition_present returns true; an empty
// condition_compare defaults to ">= 1". Shared by spell castability, trigger intervening-ifs
// (603.4), and ConditionDefined$ Remembered subability gates.
bool evaluate_present_condition(const Ability& ability, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer);

#endif
