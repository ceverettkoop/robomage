#include "card_db.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "parse.h"
#include "error.h"

extern Coordinator global_coordinator;

std::unordered_map<std::string, Entity> card_db;

Entity load_card(std::string card_name) {
    //search for card based on normalized name string
    auto uid = name_to_uid(card_name);
    auto itr = card_db.find(uid);
    //check if already loaded
    if(itr != card_db.end()) return itr->second;
    //load script
    std::string path = RESOURCE_DIR + "/cardsfolder/" + uid[0] + "/" + uid + ".txt";
    Entity parsed_card_eid = parse_card_script(path);
    if(parsed_card_eid < 0){
        non_fatal_error("Failed to parse card " + card_name);
        return parsed_card_eid;
    }
    //success
    card_db.emplace(uid, parsed_card_eid);

    // For DFCs, also store aliases so front/back face names resolve in GUI lookups
    if (global_coordinator.entity_has_component<CardData>(parsed_card_eid)) {
        const auto& cd = global_coordinator.GetComponent<CardData>(parsed_card_eid);
        auto front_uid = name_to_uid(cd.name);
        if (front_uid != uid && card_db.find(front_uid) == card_db.end())
            card_db.emplace(front_uid, parsed_card_eid);
        if (cd.backside) {
            auto back_uid = name_to_uid(cd.backside->name);
            if (back_uid != uid && card_db.find(back_uid) == card_db.end())
                card_db.emplace(back_uid, parsed_card_eid);
        }
    }

    return parsed_card_eid;
}
