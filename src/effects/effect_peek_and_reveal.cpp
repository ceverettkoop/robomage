#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool peek_and_reveal(Ability &ab, std::shared_ptr<Orderer> orderer) {
    const PeekParams *pp = std::get_if<PeekParams>(&ab.params);
    if (pp && pp->no_reveal) {
        // Mishra's Bauble: look at target player's top card privately, no reveal choice
        Zone::Ownership peek_owner = global_coordinator.entity_has_component<Player>(ab.target)
                                         ? (ab.target == cur_game.player_a_entity ? Zone::PLAYER_A : Zone::PLAYER_B)
                                         : ab.controller;
        Entity top_card = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location == Zone::LIBRARY && z.owner == peek_owner && z.distance_from_top == 0) {
                top_card = e;
                break;
            }
        }
        if (top_card == 0) {
            game_log("%s's library is empty — nothing to peek.\n", player_name(peek_owner).c_str());
        } else if (global_coordinator.entity_has_component<CardData>(top_card)) {
            auto &top_cd = global_coordinator.GetComponent<CardData>(top_card);
            game_log_private(ab.controller, "%s looks at top of %s's library: %s\n",
                player_name(ab.controller).c_str(), player_name(peek_owner).c_str(), top_cd.name.c_str());
        }
        // fall through to subabilities (DelayedTrigger sub-ability fires next upkeep)
        return true;
    }

    // Delver of Secrets: peek own library top, optionally reveal
    if (!global_coordinator.entity_has_component<Permanent>(ab.source)) {
        game_log("%s fizzles\n", ab.category.c_str());
        return false;
    }
    auto &src_perm = global_coordinator.GetComponent<Permanent>(ab.source);
    Entity top_card = 0;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::LIBRARY && z.owner == ab.controller && z.distance_from_top == 0) {
            top_card = e;
            break;
        }
    }

    if (top_card == 0) {
        game_log("Library is empty — nothing to peek.\n");
        return false;
    }
    auto &top_cd = global_coordinator.GetComponent<CardData>(top_card);
    game_log_private(ab.controller, "Top card of library: %s\n", top_cd.name.c_str());
    std::vector<LegalAction> reveal_actions = {
        LegalAction(PASS_PRIORITY, top_card, std::string("Don't reveal")),
        LegalAction(PASS_PRIORITY, top_card, std::string("Reveal")),
    };
    int reveal_choice = InputLogger::instance().get_input(reveal_actions);

    if (reveal_choice == 1) {
        game_log("Revealed: %s\n", top_cd.name.c_str());
        bool is_instant_or_sorcery = false;
        for (auto &t : top_cd.types) {
            if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) {
                is_instant_or_sorcery = true;
                break;
            }
        }
        if (is_instant_or_sorcery && global_coordinator.entity_has_component<CardData>(ab.source)) {
            auto &src_cd = global_coordinator.GetComponent<CardData>(ab.source);
            if (src_cd.backside && !src_perm.transformed) {
                src_perm.transformed = true;
                if (global_coordinator.entity_has_component<Creature>(ab.source))
                    global_coordinator.RemoveComponent<Creature>(ab.source);
                if (global_coordinator.entity_has_component<Damage>(ab.source))
                    global_coordinator.RemoveComponent<Damage>(ab.source);
                Creature back_creature;
                back_creature.base_power = static_cast<int>(src_cd.backside->power);
                back_creature.base_toughness = static_cast<int>(src_cd.backside->toughness);
                back_creature.keywords = src_cd.backside->keywords;
                recompute_pt(back_creature);
                global_coordinator.AddComponent(ab.source, back_creature);
                Damage dmg;
                dmg.damage_counters = 0;
                global_coordinator.AddComponent(ab.source, dmg);
                game_log("%s transforms into %s!\n", src_perm.name.c_str(), src_cd.backside->name.c_str());
            }
        }
    }
    return false;  // transform logic handled inline; skip subabilities loop
}

bool parse_peek_and_reveal(Ability &ab, const std::string &key, const std::string &value) {
    if (key != "NoReveal") return false;
    effect_params<PeekParams>(ab).no_reveal = (value == "True");
    return true;
}

}  // namespace effects
