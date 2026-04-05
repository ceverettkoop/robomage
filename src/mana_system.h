#ifndef MANA_SYSTEM_H
#define MANA_SYSTEM_H

#include <cstddef>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include "classes/action.h"
#include "classes/colors.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;

// Get player entity from ownership
Entity get_player_entity(Zone::Ownership player);

// Check if a given mana pool can afford a cost (does not read player state)
bool can_afford_pool(const std::multiset<Colors>& pool, const std::multiset<Colors>& cost);

// Check if player can afford a mana cost
bool can_afford(Zone::Ownership player, const std::multiset<Colors>& cost);

// Check if player can afford a cost considering current pool plus all activatable mana sources
// exclude_entity: if nonzero, skip this entity when counting available sources (used when
// the ability being checked will tap the source as part of its own cost)
bool can_afford_with_sources(Zone::Ownership player, const std::multiset<Colors>& cost,
                             std::shared_ptr<Orderer> orderer, Entity exclude_entity = 0);

// Return the maximum total mana producible (pool + untapped sources) minus colored base cost
// obligations; used for computing max X value
size_t max_available_mana(Zone::Ownership player, const ManaValue& base_cost,
                          std::shared_ptr<Orderer> orderer);

// Spend mana from player's pool (assumes can_afford was checked)
// paid_for: the entity being paid for (used for diagnostics on insufficient mana)
void spend_mana(Zone::Ownership player, const std::multiset<Colors>& cost, Entity paid_for);

// Add mana to player's pool
void add_mana(Zone::Ownership player, Colors mana_color, size_t amount);

// Pay as much of cost as possible from player's pool, return the unpaid remainder
ManaValue pay_partial(Zone::Ownership player, const ManaValue& cost);

// Empty player's mana pool (called at step transitions)
void empty_mana_pool(Zone::Ownership player);

// Snapshot of mana-related state for rewind on payment failure
struct ManaPaymentSnapshot {
    std::multiset<Colors> player_mana;
    std::vector<std::pair<Entity, bool>> tapped_state;  // entity, was_tapped
    std::vector<std::tuple<Entity, size_t, int>> activation_counts;  // entity, ability_idx, old count
    std::vector<Entity> delve_exiled;  // snapshot of cur_game.delve_exiled for delve rewind
};

ManaPaymentSnapshot snapshot_mana_state(Zone::Ownership player, std::shared_ptr<Orderer> orderer);
void restore_mana_state(Zone::Ownership player, const ManaPaymentSnapshot& snap,
                        std::shared_ptr<Orderer> orderer);

// Map a mana color to its ActionCategory (MANA_W, MANA_U, etc.)
ActionCategory mana_action_category(Colors color);

// Collect legal actions for all activatable mana sources a player controls.
// Each action has a color-specific ActionCategory (MANA_W, MANA_U, etc.).
// Sources with activation_mana_cost are included only if the cost is affordable.
std::vector<LegalAction> collect_mana_legal_actions(
    Zone::Ownership player, std::shared_ptr<Orderer> orderer, Entity paid_for = 0);

// Prompt the player to activate mana abilities to pay a cost. Returns true if cost was fully paid.
// On false, caller must restore from snapshot.
// If has_delve is true, delve exile options are included alongside mana abilities for generic costs.
bool prompt_mana_payment(Zone::Ownership controller, const ManaValue& cost,
                         Entity paid_for, std::shared_ptr<Orderer> orderer,
                         bool has_delve = false);

#endif
