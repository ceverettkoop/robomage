#include "cost_part.h"

std::string hybrid_pip_text(const HybridPip &pip) {
    std::string out = "{";
    if (pip.generic_alt > 0) out += std::to_string(pip.generic_alt);
    for (size_t i = 0; i < pip.colors.size(); ++i) {
        if (i > 0 || pip.generic_alt > 0) out += "/";
        out += mana_symbol_str(pip.colors[i]);
    }
    return out + "}";
}

std::string mana_cost_text(const CostParts &cost) {
    std::string out = cost.mana().empty() && !cost.hybrid().empty() ? "" : mana_value_text(cost.mana());
    for (const HybridPip &pip : cost.hybrid()) out += hybrid_pip_text(pip);
    return out;
}

void CostParts::add(const CostPart &part) {
    if (part.kind == CostPart::MANA && part.mana.empty() && part.hybrid.empty()) return;
    const bool merges = part.kind == CostPart::MANA || part.kind == CostPart::X_MANA;
    for (CostPart &own : parts_) {
        if (!merges || own.kind != part.kind) continue;
        own.mana.insert(part.mana.begin(), part.mana.end());
        own.hybrid.insert(own.hybrid.end(), part.hybrid.begin(), part.hybrid.end());
        own.count += part.count;
        return;
    }
    parts_.push_back(part);
    kinds_ = static_cast<uint16_t>(kinds_ | kind_bit(part.kind));
}

const CostPart *CostParts::find(CostPart::Kind kind) const {
    if (!has(kind)) return nullptr;
    for (const CostPart &part : parts_)
        if (part.kind == kind) return &part;
    return nullptr;
}

const ManaValue &CostParts::mana() const {
    static const ManaValue kNoMana;
    const CostPart *part = find(CostPart::MANA);
    return part ? part->mana : kNoMana;
}

const std::vector<HybridPip> &CostParts::hybrid() const {
    static const std::vector<HybridPip> kNoPips;
    const CostPart *part = find(CostPart::MANA);
    return part ? part->hybrid : kNoPips;
}

bool CostParts::discards_hand() const {
    const CostPart *part = find(CostPart::DISCARD);
    return part && part->whole_hand;
}

int CostParts::exile_min_types() const {
    const CostPart *part = find(CostPart::EXILE_FROM_GRAVEYARD);
    return part ? part->min_types : 0;
}

int CostParts::count(CostPart::Kind kind) const {
    const CostPart *part = find(kind);
    return part ? part->count : 0;
}

bool CostParts::variable(CostPart::Kind kind) const {
    const CostPart *part = find(kind);
    return part && part->count_is_x;
}

bool CostParts::self(CostPart::Kind kind) const {
    if (!has(kind)) return false;
    for (const CostPart &part : parts_)
        if (part.kind == kind && part.is_self()) return true;
    return false;
}

const FilterSpec &CostParts::filter(CostPart::Kind kind) const {
    static const FilterSpec kNoFilter;
    if (!has(kind)) return kNoFilter;
    for (const CostPart &part : parts_)
        if (part.kind == kind && !part.filter.empty()) return part.filter;
    return kNoFilter;
}
