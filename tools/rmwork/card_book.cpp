#include "card_book.h"

static const char *const TYPE_ORDER[] = {
    "Kindred", "Tribal", "Battle", "Enchantment", "Artifact", "Land", "Creature", "Planeswalker", "Instant", "Sorcery"};

static uint32_t type_bit(const agents::SeatStream &s, const char *name);
static bool in_order(const std::string &name);

static uint32_t type_bit(const agents::SeatStream &s, const char *name) {
    const int v = s.enum_value("card_type", name);
    return v >= 0 && v < 32 ? 1u << v : 0u;
}

// Whether `name` is one of TYPE_ORDER's types (the others follow them alphabetically).
static bool in_order(const std::string &name) {
    for (const char *t : TYPE_ORDER)
        if (name == t) return true;
    return false;
}

namespace rmwork {

uint8_t TypeBits::wubrg(uint64_t mask) const {
    uint8_t out = 0;
    for (size_t i = 0; i < color_bit.size() && i < 64; ++i)
        if (mask & (1ull << i)) out |= color_bit[i];
    return out;
}

void TypeBits::read(const agents::SeatStream &s) {
    land = type_bit(s, "Land");
    creature = type_bit(s, "Creature");
    planeswalker = type_bit(s, "Planeswalker");
    color_bit.assign(s.enum_values("color").size(), 0);
    static const char *const NAMES[] = {"WHITE", "BLUE", "BLACK", "RED", "GREEN"};
    for (int i = 0; i < 5; ++i) {
        const int v = s.enum_value("color", NAMES[i]);
        if (v >= 0 && static_cast<size_t>(v) < color_bit.size())
            color_bit[static_cast<size_t>(v)] = static_cast<uint8_t>(1u << i);
    }
}

std::string type_line_of(
    const agents::SeatStream &s, uint64_t supertypes, uint64_t types, const std::vector<std::string> &subtypes) {
    std::string out;
    const auto add = [&out](const std::string &word) {
        if (!out.empty()) out += ' ';
        out += word;
    };
    const std::vector<std::string> &supers = s.enum_values("supertype");
    for (size_t i = 0; i < supers.size() && i < 64; ++i)
        if (supertypes & (1ull << i)) add(supers[i]);
    const std::vector<std::string> &names = s.enum_values("card_type");
    for (const char *t : TYPE_ORDER) {
        const int v = s.enum_value("card_type", t);
        if (v >= 0 && v < 64 && (types & (1ull << v))) add(t);
    }
    for (size_t i = 0; i < names.size() && i < 64; ++i)
        if ((types & (1ull << i)) && !in_order(names[i])) add(names[i]);
    if (!subtypes.empty()) {
        out += " \xE2\x80\x94";
        for (const std::string &st : subtypes) add(st);
    }
    return out;
}

void CardBook::build(const agents::SeatStream &s) {
    faces_.clear();
    stream_ = &s;
    bits_.read(s);
    const rmstate::DecodedFrame &frame = s.cards_frame();
    const agents::Table card(frame, s.schema(), "card");
    const agents::Table sub(frame, s.schema(), "card_subtype");
    const int back = s.enum_value("face", "BACK");
    std::map<std::pair<int, bool>, std::vector<std::string>> subtypes;
    const int c_card = sub.col("card"), c_face = sub.col("face"), c_sub = sub.col("subtype");
    for (size_t r = 0; r < sub.rows(); ++r)
        subtypes[{sub.card(r, c_card), static_cast<int>(sub.u(r, c_face)) == back}].push_back(
            s.enum_name("subtype", sub.u(r, c_sub)));
    const int k_card = card.col("card"), k_face = card.col("face"), k_name = card.col("name"),
              k_token = card.col("token"), k_cost = card.col("mana_cost"), k_colors = card.col("colors"),
              k_types = card.col("types"), k_super = card.col("supertypes"), k_power = card.col("power"),
              k_tough = card.col("toughness"), k_loyalty = card.col("loyalty"), k_oracle = card.col("oracle");
    for (size_t r = 0; r < card.rows(); ++r) {
        const int cid = card.card(r, k_card);
        const bool is_back = static_cast<int>(card.u(r, k_face)) == back;
        if (cid < 0 || faces_.count({cid, is_back})) continue;
        CardFace f;
        f.card = cid;
        f.back = is_back;
        f.name = card.text(r, k_name);
        f.mana_cost = card.text(r, k_cost);
        f.oracle = card.text(r, k_oracle);
        const uint64_t types = card.u(r, k_types);
        f.land = (types & bits_.land) != 0;
        f.creature = (types & bits_.creature) != 0;
        f.planeswalker = (types & bits_.planeswalker) != 0;
        f.token = card.u(r, k_token) != 0;
        f.colors = bits_.wubrg(card.u(r, k_colors));
        f.power = static_cast<int>(card.s(r, k_power));
        f.toughness = static_cast<int>(card.s(r, k_tough));
        f.loyalty = static_cast<int>(card.u(r, k_loyalty));
        auto st = subtypes.find({cid, is_back});
        if (st != subtypes.end()) f.subtypes = st->second;
        bool vehicle = false;
        for (const std::string &t : f.subtypes) vehicle = vehicle || t == "Vehicle";
        f.has_pt = f.creature || (vehicle && (f.power != 0 || f.toughness != 0));
        f.type_line = type_line_of(s, card.u(r, k_super), types, f.subtypes);
        faces_.emplace(std::make_pair(cid, is_back), std::move(f));
    }
}

const CardFace *CardBook::face(int card, bool back) const {
    auto it = faces_.find({card, back});
    return it == faces_.end() ? nullptr : &it->second;
}

std::vector<std::string> mana_symbols(const std::string &cost) {
    std::vector<std::string> out;
    size_t i = 0;
    while ((i = cost.find('{', i)) != std::string::npos) {
        const size_t j = cost.find('}', i);
        if (j == std::string::npos) break;
        out.push_back(cost.substr(i + 1, j - i - 1));
        i = j + 1;
    }
    return out;
}

}  // namespace rmwork
