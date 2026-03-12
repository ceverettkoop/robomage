#include "gui.h"

#include "card_db.h"
#include "card_vocab.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "parse.h"

#include <cstring>

extern Coordinator global_coordinator;

extern "C" {

const char* gui_card_name(int vocab_idx) {
    return card_index_to_name(vocab_idx);
}

//convoluated way to find it here, TOOD revisit
const char* gui_card_oracle(int vocab_idx) {
    static char buf[1024];
    auto uid = name_to_uid(card_index_to_name(vocab_idx)) ;
    if (uid[0] == '?') return "";
    auto it = card_db.find(uid);
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

const char* gui_card_mana_cost(int vocab_idx) {
    static char buf[128];
    buf[0] = '\0';
    auto uid = name_to_uid(card_index_to_name(vocab_idx));
    if (uid[0] == '?') return "";
    auto it = card_db.find(uid);
    if (it == card_db.end()) return "";
    if (!global_coordinator.entity_has_component<CardData>(it->second)) return "";
    const CardData& cd = global_coordinator.GetComponent<CardData>(it->second);
    if (cd.mana_cost.empty()) return "";
    int generic = 0;
    int pos = 0;
    for (Colors c : cd.mana_cost) {
        if (c == GENERIC) { generic++; continue; }
        const char* sym = "";
        switch (c) {
            case WHITE:     sym = "W"; break;
            case BLUE:      sym = "U"; break;
            case BLACK:     sym = "B"; break;
            case RED:       sym = "R"; break;
            case GREEN:     sym = "G"; break;
            case COLORLESS: sym = "C"; break;
            default: break;
        }
        pos += snprintf(buf + pos, sizeof(buf) - (size_t)pos, "{%s}", sym);
    }
    if (generic > 0) {
        // prepend generic cost
        char tmp[128];
        snprintf(tmp, sizeof(tmp), "{%d}", generic);
        strncat(tmp, buf, sizeof(tmp) - strlen(tmp) - 1);
        strncpy(buf, tmp, sizeof(buf) - 1);
        buf[sizeof(buf) - 1] = '\0';
    }
    return buf;
}

int gui_card_color_identity(int vocab_idx) {
    auto uid = name_to_uid(card_index_to_name(vocab_idx));
    if (uid[0] == '?') return 0;
    auto it = card_db.find(uid);
    if (it == card_db.end()) return 0;
    if (!global_coordinator.entity_has_component<CardData>(it->second)) return 0;
    const CardData& cd = global_coordinator.GetComponent<CardData>(it->second);
    int mask = 0;
    for (Colors c : cd.mana_cost) {
        switch (c) {
            case WHITE: mask |= 1; break;
            case BLUE:  mask |= 2; break;
            case BLACK: mask |= 4; break;
            case RED:   mask |= 8; break;
            case GREEN: mask |= 16; break;
            default: break;
        }
    }
    return mask;
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
