#include "creature.h"
#include "color_identity.h"
#include "../ecs/coordinator.h"
#include <algorithm>

// Apply layer-7 P/T effects to a single creature in sublayer order (rule 613.4):
//   7a CDA -> 7b set -> 7c modify (counters & +N/+N) -> 7d switch.
// Each contribution is stored on the Creature; the layer engine
// (StateManager::apply_continuous_effects) rebuilds them every SBE pass so this
// recompute applies the full layer-7 result, not an out-of-band delta. Signed
// arithmetic floors at 0 so a transient negative can never underflow the uint32_t
// cached fields read by the ~75 combat/damage/ML consumers.
void recompute_pt(Creature &cr) {
    // 7a — characteristic-defining base, folded into base_* by the CDA pass.
    long p = static_cast<long>(cr.base_power);
    long t = static_cast<long>(cr.base_toughness);

    // 7b — "set power/toughness to N" overrides the base (inert today: has_set_pt
    // is never set by any current-vocab effect, so 7a flows straight to 7c).
    if (cr.has_set_pt) {
        p = static_cast<long>(cr.set_power);
        t = static_cast<long>(cr.set_toughness);
    }

    // 7c — additive modifications and counters. These are commutative, so the
    // flat sum equals any timestamp-ordered application within the sublayer.
    p += cr.counter_pt_bonus + cr.prowess_bonus + cr.eot_power_bonus + cr.static_power_bonus;
    t += cr.counter_pt_bonus + cr.prowess_bonus + cr.eot_toughness_bonus + cr.static_toughness_bonus;

    // 7d — switch power and toughness (inert today).
    if (cr.switch_pt) std::swap(p, t);

    cr.power     = static_cast<uint32_t>(std::max(0L, p));
    cr.toughness = static_cast<uint32_t>(std::max(0L, t));
}

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

// True if this creature has "protection from colored spells" (Emrakul: K:Protection:
// Spell.nonColorless). CR 702.16a: the quality is "colored spell". The caller (is_legal_target)
// applies the spell-ness and color of the source, since whether the targeting object is a SPELL
// (vs an ability) is known there from the Ability's type, not from the candidate's Creature.
bool has_protection_from_colored_spells(const Creature &cr) {
    for (const auto &kw : cr.keywords)
        if (kw == "Protection from colored spells") return true;
    return false;
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
