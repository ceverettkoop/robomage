#include "colors.h"

std::string mana_symbol(Colors color) {
    switch (color) {
        case WHITE:
            return "W";
            break;
        case BLUE:
            return "U";
            break;
        case BLACK:
            return "B";
            break;
        case RED:
            return "R";
            break;
        case GREEN:
            return "G";
            break;
        case COLORLESS:
            return "C";
            break;
        default:
            return "?";
            break;
    }
}

const char *mana_symbol_str(Colors color) {
    switch (color) {
        case WHITE:     return "W";
        case BLUE:      return "U";
        case BLACK:     return "B";
        case RED:       return "R";
        case GREEN:     return "G";
        case COLORLESS: return "C";
        default:        return "?";
    }
}

std::string mana_value_text(const ManaValue &mana) {
    size_t generic = mana.count(GENERIC);
    std::string out;
    if (generic > 0 || mana.empty()) out += "{" + std::to_string(generic) + "}";
    for (Colors c : mana) {
        if (c == GENERIC) continue;
        out += "{";
        out += mana_symbol_str(c);
        out += "}";
    }
    return out;
}
