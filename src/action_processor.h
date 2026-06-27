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
