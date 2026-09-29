#include "card_db.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "parse.h"
#include "error.h"

#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <vector>

extern Coordinator global_coordinator;

std::unordered_map<std::string, Entity> card_db;

static bool script_file_exists(const std::string& path);
static std::vector<std::string> sorted_dir_entries(const std::string& dir);
static bool ends_with(const std::string& s, const std::string& suffix);
static std::string script_face_name(const std::string& path, bool back);
static std::string resolve_dfc_script_path(const std::string& dir, const std::string& uid);
static std::string resolve_back_face_script_path(const std::string& uid);

Entity load_card(std::string card_name) {
    //search for card based on normalized name string
    auto uid = name_to_uid(card_name);
    auto itr = card_db.find(uid);
    //check if already loaded
    if(itr != card_db.end()) return itr->second;
    //load script
    std::string dir = RESOURCE_DIR + "/cardsfolder/" + uid[0] + "/";
    std::string path = dir + uid + ".txt";
    //A double-faced card has no "<uid>.txt" of its own: Forge stores it under the
    //combined "<front>_<back>.txt" filename. Fall back to that combined script when
    //the direct file is absent — matched on its front face, else on its back face
    //(a card named by its back-face name, e.g. a test preset of a transformed DFC).
    if(!script_file_exists(path)){
        std::string dfc_path = resolve_dfc_script_path(dir, uid);
        if(dfc_path.empty()) dfc_path = resolve_back_face_script_path(uid);
        if(!dfc_path.empty()) path = dfc_path;
    }
    Entity parsed_card_eid = parse_card_script(path);
    if(parsed_card_eid == 0) fatal_error("Failed to parse card " + card_name + " (" + path + ")");
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

bool names_back_face(const std::string& card_name, const CardData& cd) {
    if(!cd.backside) return false;
    auto uid = name_to_uid(card_name);
    return uid != name_to_uid(cd.name) && uid == name_to_uid(cd.backside->name);
}

static bool script_file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.is_open();
}

//The entry names in `dir`, sorted, so a scan never depends on readdir order.
static std::vector<std::string> sorted_dir_entries(const std::string& dir) {
    std::vector<std::string> names;
    DIR* d = opendir(dir.c_str());
    if(!d) return names;
    struct dirent* ent;
    while((ent = readdir(d)) != nullptr) names.push_back(ent->d_name);
    closedir(d);
    std::sort(names.begin(), names.end());
    return names;
}

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

//The "Name:" of a script's front face (back=false) or of the face after its
//ALTERNATE separator (back=true); "" when the script has no such face.
static std::string script_face_name(const std::string& path, bool back) {
    std::ifstream f(path);
    std::string line;
    bool in_back = false;
    while(std::getline(f, line)){
        if(!line.empty() && line.back() == '\r') line.pop_back();
        if(line == "ALTERNATE"){
            if(!back) return "";
            in_back = true;
            continue;
        }
        if(in_back == back && line.compare(0, 5, "Name:") == 0) return line.substr(5);
    }
    return "";
}

//Scan a cardsfolder letter directory for a double-faced card's combined script
//"<uid>_*.txt" whose front face is named `uid`, and return its full path, or ""
//if none exists. Candidates are taken in sorted order and name-checked, so an
//unrelated "<uid>_more_words.txt" single-faced card never matches. Only called
//on the miss path (no exact "<uid>.txt"), so single-faced cards never reach here.
static std::string resolve_dfc_script_path(const std::string& dir, const std::string& uid) {
    const std::string prefix = uid + "_";
    for(const auto& fname : sorted_dir_entries(dir)){
        if(fname.size() <= prefix.size() + 4 || fname.compare(0, prefix.size(), prefix) != 0 ||
           !ends_with(fname, ".txt"))
            continue;
        std::string path = dir + fname;
        if(name_to_uid(script_face_name(path, false)) == uid) return path;
    }
    return "";
}

//Find the combined "*_<uid>.txt" script (in any letter directory) whose BACK face
//is named `uid` — a DFC back face or a split card's second half — or "".
static std::string resolve_back_face_script_path(const std::string& uid) {
    const std::string root = RESOURCE_DIR + "/cardsfolder/";
    const std::string suffix = "_" + uid + ".txt";
    for(const auto& letter : sorted_dir_entries(root)){
        if(letter.empty() || letter[0] == '.') continue;
        std::string dir = root + letter + "/";
        for(const auto& fname : sorted_dir_entries(dir)){
            if(fname.size() <= suffix.size() || !ends_with(fname, suffix)) continue;
            std::string path = dir + fname;
            if(name_to_uid(script_face_name(path, true)) == uid) return path;
        }
    }
    return "";
}
