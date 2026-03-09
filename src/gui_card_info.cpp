#include "gui.h"

#include "card_db.h"
#include "card_vocab.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"

#include <cstring>

extern Coordinator global_coordinator;

extern "C" {

const char* gui_card_name(int vocab_idx) {
    return card_index_to_name(vocab_idx);
}

const char* gui_card_oracle(int vocab_idx) {
    static char buf[1024];
    const char* name = card_index_to_name(vocab_idx);
    if (name[0] == '?') return "";
    auto it = card_db.find(name);
    if (it == card_db.end()) return "";
    if (!global_coordinator.entity_has_component<CardData>(it->second)) return "";
    const CardData& cd = global_coordinator.GetComponent<CardData>(it->second);
    strncpy(buf, cd.oracle_text.c_str(), sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    return buf;
}

const char* gui_card_type_line(int vocab_idx) {
    static char buf[256];
    const char* name = card_index_to_name(vocab_idx);
    if (name[0] == '?') return "";
    auto it = card_db.find(name);
    if (it == card_db.end()) return "";
    if (!global_coordinator.entity_has_component<CardData>(it->second)) return "";
    const CardData& cd = global_coordinator.GetComponent<CardData>(it->second);
    buf[0] = '\0';
    bool has_subtype = false;
    for (const auto& t : cd.types) {
        if (t.kind == SUBTYPE) { has_subtype = true; continue; }
        if (buf[0] != '\0') strncat(buf, " ", sizeof(buf) - strlen(buf) - 1);
        strncat(buf, t.name.c_str(), sizeof(buf) - strlen(buf) - 1);
    }
    if (has_subtype) {
        strncat(buf, " - ", sizeof(buf) - strlen(buf) - 1);
        bool first = true;
        for (const auto& t : cd.types) {
            if (t.kind != SUBTYPE) continue;
            if (!first) strncat(buf, " ", sizeof(buf) - strlen(buf) - 1);
            strncat(buf, t.name.c_str(), sizeof(buf) - strlen(buf) - 1);
            first = false;
        }
    }
    return buf;
}

const char* gui_step_name(int step) {
    static const char* names[] = {
        "UNTAP", "UPKEEP", "DRAW", "FIRST MAIN",
        "BEGIN COMBAT", "DECLARE ATTACKERS", "DECLARE BLOCKERS",
        "COMBAT DAMAGE", "END OF COMBAT", "SECOND MAIN",
        "END STEP", "CLEANUP"
    };
    if (step < 0 || step >= 12) return "UNKNOWN";
    return names[step];
}

int gui_card_base_power(int vocab_idx) {
    const char* name = card_index_to_name(vocab_idx);
    if (name[0] == '?') return 0;
    auto it = card_db.find(name);
    if (it == card_db.end()) return 0;
    if (!global_coordinator.entity_has_component<CardData>(it->second)) return 0;
    return (int)global_coordinator.GetComponent<CardData>(it->second).power;
}

int gui_card_base_toughness(int vocab_idx) {
    const char* name = card_index_to_name(vocab_idx);
    if (name[0] == '?') return 0;
    auto it = card_db.find(name);
    if (it == card_db.end()) return 0;
    if (!global_coordinator.entity_has_component<CardData>(it->second)) return 0;
    return (int)global_coordinator.GetComponent<CardData>(it->second).toughness;
}

} // extern "C"
