#include "creature.h"
#include "color_identity.h"
#include "../ecs/coordinator.h"

static const char *PROT_PREFIX = "Protection from ";
static const size_t PROT_PREFIX_LEN = 16; // strlen("Protection from ")

std::set<Colors> get_protection_colors(const Creature &cr) {
    std::set<Colors> result;
    for (const auto &kw : cr.keywords) {
        if (kw.compare(0, PROT_PREFIX_LEN, PROT_PREFIX) != 0) continue;
        std::string color_word = kw.substr(PROT_PREFIX_LEN);
        if      (color_word == "white") result.insert(WHITE);
        else if (color_word == "blue")  result.insert(BLUE);
        else if (color_word == "black") result.insert(BLACK);
        else if (color_word == "red")   result.insert(RED);
        else if (color_word == "green") result.insert(GREEN);
        else if (color_word == "each color") {
            result.insert(WHITE);
            result.insert(BLUE);
            result.insert(BLACK);
            result.insert(RED);
            result.insert(GREEN);
        }
    }
    return result;
}

bool has_protection_from(const Creature &cr, Entity source) {
    Coordinator &coordinator = Coordinator::global();
    if (!coordinator.entity_has_component<ColorIdentity>(source)) return false;
    const auto &ci = coordinator.GetComponent<ColorIdentity>(source);
    std::set<Colors> prot = get_protection_colors(cr);
    for (Colors c : ci.colors) {
        if (prot.count(c)) return true;
    }
    return false;
}
