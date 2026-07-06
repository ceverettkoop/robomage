#include "card_db.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "parse.h"
#include "error.h"

#include <dirent.h>
#include <fstream>

extern Coordinator global_coordinator;

std::unordered_map<std::string, Entity> card_db;

static bool script_file_exists(const std::string& path);
static std::string resolve_dfc_script_path(const std::string& dir, const std::string& uid);

Entity load_card(std::string card_name) {
    //search for card based on normalized name string
    auto uid = name_to_uid(card_name);
    auto itr = card_db.find(uid);
    //check if already loaded
    if(itr != card_db.end()) return itr->second;
    //load script
    std::string dir = RESOURCE_DIR + "/cardsfolder/" + uid[0] + "/";
    std::string path = dir + uid + ".txt";
    //A double-faced card referenced by its front-face name has no "<uid>.txt" of
    //its own: Forge stores it under the combined "<front>_<back>.txt" filename.
    //Fall back to that combined script when the direct file is absent.
    if(!script_file_exists(path)){
        std::string dfc_path = resolve_dfc_script_path(dir, uid);
        if(!dfc_path.empty()) path = dfc_path;
    }
    Entity parsed_card_eid = parse_card_script(path);
    if(parsed_card_eid < 0){
        non_fatal_error("Failed to parse card " + card_name);
        return parsed_card_eid;
    }
    //success
    card_db.emplace(uid, parsed_card_eid);

    // For DFCs, also store aliases so front/back face names resolve in name lookups
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

static bool script_file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.is_open();
}

//Scan a cardsfolder letter directory for a double-faced card's combined script
//file "<uid>_*.txt" and return its full path, or "" if none exists. Only called
//on the miss path (no exact "<uid>.txt"), so single-faced cards never reach here.
static std::string resolve_dfc_script_path(const std::string& dir, const std::string& uid) {
    std::string prefix = uid + "_";
    const std::string suffix = ".txt";
    DIR* d = opendir(dir.c_str());
    if(!d) return "";
    std::string result;
    struct dirent* ent;
    while((ent = readdir(d)) != nullptr){
        std::string fname = ent->d_name;
        if(fname.size() > prefix.size() + suffix.size() &&
           fname.compare(0, prefix.size(), prefix) == 0 &&
           fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) == 0){
            result = dir + fname;
            break;
        }
    }
    closedir(d);
    return result;
}
