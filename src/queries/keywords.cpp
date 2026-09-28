#include "keywords.h"

#include <algorithm>

#include "../components/carddata.h"
#include "../components/token.h"
#include "characteristics.h"

void add_keyword_instance(std::vector<std::string> &keywords, const std::string &kw) {
    if (keyword_instances_stack(kw) || std::find(keywords.begin(), keywords.end(), kw) == keywords.end())
        keywords.push_back(kw);
}

bool is_indestructible(Entity e) {
    if (keyword_removed_eot(e, "Indestructible")) return false;
    if (global_coordinator.entity_has_component<Creature>(e)) {
        return creature_has_keyword(global_coordinator.GetComponent<Creature>(e), "Indestructible");
    }
    if (global_coordinator.entity_has_component<CardData>(e)) {
        for (const auto &k : active_face(e, global_coordinator.GetComponent<CardData>(e)).keywords)
            if (k == "Indestructible") return true;
        return false;
    }
    if (global_coordinator.entity_has_component<Token>(e)) {
        for (const auto &k : global_coordinator.GetComponent<Token>(e).keywords)
            if (k == "Indestructible") return true;
    }
    return false;
}

bool permanent_has_keyword(Entity e, const char *kw) {
    if (keyword_removed_eot(e, kw)) return false;
    if (global_coordinator.entity_has_component<Creature>(e))
        return creature_has_keyword(global_coordinator.GetComponent<Creature>(e), kw);
    if (global_coordinator.entity_has_component<CardData>(e)) {
        for (const auto &k : active_face(e, global_coordinator.GetComponent<CardData>(e)).keywords)
            if (k == kw) return true;
        return false;
    }
    if (global_coordinator.entity_has_component<Token>(e)) {
        for (const auto &k : global_coordinator.GetComponent<Token>(e).keywords)
            if (k == kw) return true;
    }
    return false;
}

std::vector<std::string> permanent_keywords(Entity e) {
    std::vector<std::string> out;
    if (global_coordinator.entity_has_component<Creature>(e))
        out = global_coordinator.GetComponent<Creature>(e).keywords;
    else if (global_coordinator.entity_has_component<CardData>(e))
        out = active_face(e, global_coordinator.GetComponent<CardData>(e)).keywords;
    else if (global_coordinator.entity_has_component<Token>(e))
        out = global_coordinator.GetComponent<Token>(e).keywords;
    out.erase(std::remove_if(out.begin(), out.end(),
                             [e](const std::string &k) { return keyword_removed_eot(e, k.c_str()); }),
              out.end());
    return out;
}
