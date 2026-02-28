#ifndef ABILITY_H
#define ABILITY_H

#include "../classes/colors.h"
#include "../ecs/entity.h"
#include "zone.h"
#include <memory>
#include <string>
#include <vector>

class Orderer;

struct Ability{

    enum AbilityType{
        TRIGGERED,
        ACTIVATED,
        SPELL
    };

    AbilityType ability_type = SPELL;
    std::string category = "";
    // TODO: support multiple targets (e.g. "deal 1 damage to each of up to two targets")
    std::string valid_tgts = "N_A";  // Value of ValidTgts$ param; "N_A" if no targeting required
    Entity source = 0;
    Entity target = 0;
    // TODO: support multiple effects per ability (e.g. "deal 3 damage and gain 3 life")
    size_t amount = 0;
    Colors color = NO_COLOR; //for mana ability

    // Activated ability costs
    bool tap_cost = false;              // {T} is part of the activation cost
    ManaValue activation_mana_cost;     // Mana that must be paid to activate
    int life_cost = 0;                  // PayLife<N> — life paid at activation
    bool sac_self = false;              // Sac<1/CARDNAME> — sacrifice source permanent as cost
    std::string change_type = "";        // ChangeType$ — comma-separated subtypes to search
    Zone::ZoneValue origin = Zone::LIBRARY;          // Origin$ — zone to search
    Zone::ZoneValue destination = Zone::BATTLEFIELD; // Destination$ — zone to move card to
    bool mandatory = false;              // Mandatory$ True — player must choose; suppresses fail-to-find when zone non-empty
    bool may_shuffle = false;            // MayShuffle$ True — player may optionally shuffle after
    size_t unless_generic_cost = 0;      // UnlessCost$ N — target controller pays {N} to prevent counter
    std::string target_type = "";        // TargetType$ Spell — restricts targeting to stack spells
    //for each AB on a card script there may be multiple SubAbility$, would get parsed into vector below
    std::vector<Ability> subabilities; // additional abilities resolved at same time this resolves, stored in order

    void resolve(std::shared_ptr<Orderer> orderer);
    bool identical_activated_ability(const Ability& other);
private:
    void resolve_change_zone(std::shared_ptr<Orderer> orderer);
    void resolve_destroy(std::shared_ptr<Orderer> orderer);
    void resolve_rearrange_top_of_library(std::shared_ptr<Orderer> orderer);

};

// Search a zone for cards matching the comma-separated type list in change_type
// (empty change_type matches all cards in the zone).
// When mandatory=true, "fail to find" is suppressed unless the zone is empty.
// Returns the chosen Entity, or 0 if the player fails to find / zone is empty.
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
                   Zone::ZoneValue zone, const std::string& change_type,
                   bool mandatory = false,
                   Zone::ZoneValue destination = Zone::GRAVEYARD);

#endif /* ABILITY_H */
