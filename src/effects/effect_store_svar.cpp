#include "effects.h"

#include <cctype>
#include <string>

#include "../components/permanent.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

namespace effects {

// DB$ StoreSVar | SVar$ <name> | Type$ Number | Expression$ <int>
//
// Latch a named integer onto the SOURCE permanent's per-permanent scratch store
// (Permanent::stored_svars), where a CheckSVar$ <name> | SVarCompare$ ... trigger gate reads it
// back (see Ability::stored_svar_gate_name and stored_svar_gate_passes). Carpet of Flowers uses
// CarpetX as a once-per-turn latch: its CheckPlus sub-ability stores 1 after the mana is added,
// the Static$ True cleanup reset stores 0 to re-arm next turn, and the gate (EQ0) blocks the
// second main-phase fire after the first one used it.
//
// No-op when the source is not a battlefield permanent — e.g. the ChangesZone Battlefield→Any
// reset, whose Permanent has already been destroyed by the time it would resolve (the store
// resets naturally because a fresh Permanent on re-entry starts with an empty stored_svars map).
bool store_svar(Ability &ab, std::shared_ptr<Orderer> /*orderer*/) {
    if (ab.stored_svar_set_name.empty()) return true;
    if (ab.source != 0 && global_coordinator.entity_has_component<Permanent>(ab.source)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(ab.source);
        perm.stored_svars[ab.stored_svar_set_name] = ab.stored_svar_set_value;
    }
    return true;
}

bool parse_store_svar(Ability &ab, const std::string &key, const std::string &value) {
    if (ab.category != "StoreSVar") return false;
    if (key == "SVar") {
        ab.stored_svar_set_name = value;
        return true;
    }
    if (key == "Expression") {
        // The value to latch. Forge only ever stores a plain integer literal here for the
        // scratch-int idiom (Carpet of Flowers: Expression$ 0 / 1). Parse it by hand — exceptions
        // are disabled, so std::stoi on a non-numeric value would terminate. A non-integer
        // expression leaves the default 0.
        int sign = 1;
        size_t i = 0;
        if (i < value.size() && (value[i] == '-' || value[i] == '+')) {
            if (value[i] == '-') sign = -1;
            ++i;
        }
        int acc = 0;
        bool any = false;
        for (; i < value.size() && std::isdigit(static_cast<unsigned char>(value[i])); ++i) {
            acc = acc * 10 + (value[i] - '0');
            any = true;
        }
        ab.stored_svar_set_value = any ? sign * acc : 0;
        return true;
    }
    if (key == "Type") {
        // Type$ Number — the only supported scratch-SVar kind; consumed (no extra state needed).
        return true;
    }
    return false;
}

}  // namespace effects
