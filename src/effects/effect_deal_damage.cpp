#include "effects.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>

#include "../cli_output.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../error.h"
#include "../game_queries.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

namespace effects {

bool deal_damage(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Delirium-conditional damage (Unholy Heat)
    size_t dmg = ab.amount;
    if (ab.amount_is_delirium_scale) {
        Zone::Ownership caster = global_coordinator.entity_has_component<Permanent>(ab.source)
                                     ? global_coordinator.GetComponent<Permanent>(ab.source).controller
                                     : global_coordinator.GetComponent<Zone>(ab.source).owner;
        if (check_delirium(caster, orderer->mEntities)) dmg = ab.amount_delirium;
    }
    if (global_coordinator.entity_has_component<Player>(ab.target)) {
        auto &player = global_coordinator.GetComponent<Player>(ab.target);
        player.life_total -= static_cast<int32_t>(dmg);
        game_log("Dealt %zu damage to player (now at %d life)\n", dmg, player.life_total);
    } else {
        if (::deal_damage(ab.source, ab.target, dmg)) {
            game_log("Dealt %zu damage to creature\n", dmg);
        } else {
#ifndef NDEBUG
            fprintf(stderr, "SOURCE:");
            dump_entity(ab.source);
            fprintf(stderr, "TARGET:");
            dump_entity(ab.target);
#endif
            non_fatal_error("Damage should have fizzled prior to this");
        }
    }
    return true;
}

bool parse_deal_damage(Ability &ab, const std::string &key, const std::string &value) {
    if (key != "NumDmg") return false;
    // Check if value is numeric; if not, store as SVar key for resolution later
    if (!value.empty() && (std::isdigit(static_cast<unsigned char>(value[0])) ||
                           (value[0] == '-' && value.size() > 1 &&
                            std::isdigit(static_cast<unsigned char>(value[1]))))) {
        ab.amount = static_cast<size_t>(std::stoi(value));
    } else {
        ab.amount_svar = value;
    }
    return true;
}

}  // namespace effects
