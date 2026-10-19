#include "targets.h"

#include <algorithm>
#include <map>
#include <utility>

#include "columns.h"

namespace {

// One registered or reference list: its name and per card its (main, side) counts.
struct ListCounts {
        std::string name;
        std::map<uint64_t, std::pair<uint64_t, uint64_t>> cards;
};

}  // namespace

static const rmstate::DecodedTable *frame_table(
    const rmstate::Schema &schema, const rmstate::DecodedFrame &frame, const char *name, const rmstate::TableDef **def);
static bool header_of(const featurize::FeatureSpec &spec, const rmstate::DecodedFrame &frame, int64_t *game,
    int64_t *decision, bool *sideboarding);
static int64_t enum_value(const rmstate::Schema &schema, const char *enum_name, const char *value);
static uint64_t overlap(const ListCounts &a, const ListCounts &b);
static int64_t classify(const ListCounts &registered, const std::vector<ListCounts> &reference);
static void append_lists(featurize::RmtbFile *file, const char *name, const std::vector<featurize::Targets> &list,
    const std::vector<int64_t> featurize::Targets::*member);

static const rmstate::DecodedTable *frame_table(const rmstate::Schema &schema, const rmstate::DecodedFrame &frame,
    const char *name, const rmstate::TableDef **def) {
    *def = featurize::table_named(schema, name);
    return *def ? frame.table(static_cast<uint16_t>((*def)->id)) : nullptr;
}

static bool header_of(const featurize::FeatureSpec &spec, const rmstate::DecodedFrame &frame, int64_t *game,
    int64_t *decision, bool *sideboarding) {
    const rmstate::DecodedTable *h = frame.table(spec.types[static_cast<size_t>(spec.header_type)].table);
    if (!h || h->rows.empty()) return false;
    *game = static_cast<int64_t>(h->rows[0][static_cast<size_t>(spec.header_game)].bits);
    *decision = static_cast<int64_t>(h->rows[0][static_cast<size_t>(spec.header_decision)].bits);
    *sideboarding = h->rows[0][static_cast<size_t>(spec.header_sideboarding)].bits != 0;
    return true;
}

static int64_t enum_value(const rmstate::Schema &schema, const char *enum_name, const char *value) {
    const std::vector<std::string> &values = featurize::enum_values(schema, enum_name);
    for (size_t i = 0; i < values.size(); ++i)
        if (values[i] == value) return static_cast<int64_t>(i);
    return -1;
}

// Σ over cards of the lesser main count plus the lesser side count.
static uint64_t overlap(const ListCounts &a, const ListCounts &b) {
    uint64_t n = 0;
    for (const auto &c : a.cards) {
        auto it = b.cards.find(c.first);
        if (it == b.cards.end()) continue;
        n += std::min(c.second.first, it->second.first) + std::min(c.second.second, it->second.second);
    }
    return n;
}

// docs/knowledge_store.md §10.4: the list of the same name, else the most cards in common (ties to the
// lower id), else -1.
static int64_t classify(const ListCounts &registered, const std::vector<ListCounts> &reference) {
    for (size_t i = 0; i < reference.size(); ++i)
        if (!registered.name.empty() && reference[i].name == registered.name) return static_cast<int64_t>(i);
    int64_t best = -1;
    uint64_t best_overlap = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const uint64_t o = overlap(reference[i], registered);
        if (o > best_overlap) {
            best = static_cast<int64_t>(i);
            best_overlap = o;
        }
    }
    return best;
}

static void append_lists(featurize::RmtbFile *file, const char *name, const std::vector<featurize::Targets> &list,
    const std::vector<int64_t> featurize::Targets::*member) {
    std::vector<int64_t> values, offsets = {0};
    for (const featurize::Targets &t : list) {
        const std::vector<int64_t> &v = t.*member;
        values.insert(values.end(), v.begin(), v.end());
        offsets.push_back(static_cast<int64_t>(values.size()));
    }
    file->arrays.push_back(featurize::rmtb_ints(name, featurize::Dtype::I64, {values.size()}, values));
    file->arrays.push_back(
        featurize::rmtb_ints(std::string(name) + "_offsets", featurize::Dtype::I64, {offsets.size()}, offsets));
}

namespace featurize {

bool registered_lists(const rmstate::Schema &schema, const rmstate::DecodedFrame &decklists,
    const rmstate::DecodedFrame &metagame, RegisteredLists *out, std::string *err) {
    *out = RegisteredLists();
    const rmstate::TableDef *dl = nullptr, *dc = nullptr, *ml = nullptr, *mc = nullptr;
    const rmstate::DecodedTable *t_dl = frame_table(schema, decklists, "decklist", &dl);
    const rmstate::DecodedTable *t_dc = frame_table(schema, decklists, "decklist_card", &dc);
    const rmstate::DecodedTable *t_ml = frame_table(schema, metagame, "meta_list", &ml);
    const rmstate::DecodedTable *t_mc = frame_table(schema, metagame, "meta_list_card", &mc);
    if (!dl || !dc || !ml || !mc) {
        *err = "the schema lacks decklist, decklist_card, meta_list or meta_list_card";
        return false;
    }
    const int dl_player = column_index(*dl, "player"), dl_name = column_index(*dl, "name");
    const int dc_player = column_index(*dc, "player"), dc_card = column_index(*dc, "card"),
              dc_main = column_index(*dc, "main"), dc_side = column_index(*dc, "side");
    const int ml_name = column_index(*ml, "name"), ml_arch = column_index(*ml, "archetype");
    const int mc_list = column_index(*mc, "list"), mc_card = column_index(*mc, "card"),
              mc_main = column_index(*mc, "main"), mc_side = column_index(*mc, "side");
    std::vector<ListCounts> reference(t_ml ? t_ml->rows.size() : 0);
    for (size_t l = 0; l < reference.size(); ++l) reference[l].name = t_ml->rows[l][static_cast<size_t>(ml_name)].text;
    if (t_mc)
        for (const std::vector<rmstate::Value> &r : t_mc->rows) {
            const uint64_t l = r[static_cast<size_t>(mc_list)].bits;
            if (l < reference.size())
                reference[l].cards[r[static_cast<size_t>(mc_card)].bits] = {
                    r[static_cast<size_t>(mc_main)].bits, r[static_cast<size_t>(mc_side)].bits};
        }
    ListCounts seats[2];
    if (t_dl)
        for (const std::vector<rmstate::Value> &r : t_dl->rows) {
            const uint64_t p = r[static_cast<size_t>(dl_player)].bits;
            if (p == 1 || p == 2) seats[p - 1].name = r[static_cast<size_t>(dl_name)].text;
        }
    if (t_dc)
        for (const std::vector<rmstate::Value> &r : t_dc->rows) {
            const uint64_t p = r[static_cast<size_t>(dc_player)].bits;
            if (p != 1 && p != 2) continue;
            std::pair<uint64_t, uint64_t> &c = seats[p - 1].cards[r[static_cast<size_t>(dc_card)].bits];
            c.first += r[static_cast<size_t>(dc_main)].bits;
            c.second += r[static_cast<size_t>(dc_side)].bits;
        }
    for (int s = 0; s < 2; ++s) {
        if (seats[s].cards.empty()) continue;
        out->list[s] = classify(seats[s], reference);
        if (out->list[s] < 0) continue;
        const rmstate::Value &a = t_ml->rows[static_cast<size_t>(out->list[s])][static_cast<size_t>(ml_arch)];
        out->archetype[s] = rmstate::value_is_all_ones(a, ml->columns[static_cast<size_t>(ml_arch)].storage)
                                ? -1
                                : static_cast<int64_t>(a.bits);
    }
    return true;
}

bool decision_targets(const FeatureSpec &spec, const rmstate::Schema &schema, const rmstate::DecodedFrame &seat_state,
    const rmstate::DecodedFrame &omniscient_state, const RegisteredLists &lists, Targets *out, std::string *err) {
    const uint8_t seat = seat_state.header.viewer;
    if (seat != static_cast<uint8_t>(rmstate::Viewer::A) && seat != static_cast<uint8_t>(rmstate::Viewer::B)) {
        *err = "the targets' seat frame is not a seat's";
        return false;
    }
    if (omniscient_state.header.viewer != static_cast<uint8_t>(rmstate::Viewer::OMNISCIENT)) {
        *err = "the targets' truth frame is not the omniscient viewer's";
        return false;
    }
    *out = Targets();
    out->seat = seat;
    int64_t og = 0, od = 0;
    bool osb = false;
    if (!header_of(spec, seat_state, &out->game, &out->decision, &out->sideboarding) ||
        !header_of(spec, omniscient_state, &og, &od, &osb) || og != out->game || od != out->decision ||
        osb != out->sideboarding) {
        *err = "the omniscient frame is not of the seat frame's decision";
        return false;
    }
    const size_t opp = seat == static_cast<uint8_t>(rmstate::Viewer::A) ? 1 : 0;
    out->opp_list = lists.list[opp];
    out->opp_archetype = lists.archetype[opp];
    const rmstate::TableDef *def = nullptr;
    const rmstate::DecodedTable *objects = frame_table(schema, omniscient_state, "object", &def);
    if (!def) {
        *err = "the schema has no object table";
        return false;
    }
    if (!objects) return true;
    const int c_zone = column_index(*def, "zone"), c_owner = column_index(*def, "owner"),
              c_pos = column_index(*def, "pos"), c_card = column_index(*def, "card");
    const uint64_t hand = static_cast<uint64_t>(enum_value(schema, "zone", "HAND")),
                   library = static_cast<uint64_t>(enum_value(schema, "zone", "LIBRARY"));
    const rmstate::Storage card_storage = def->columns[static_cast<size_t>(c_card)].storage;
    // enum:seat: 1 = A, 2 = B (Appendix A.1).
    const uint64_t own_seat = seat + 1u, opp_seat = opp + 1u;
    std::vector<std::pair<uint64_t, int64_t>> own_lib, opp_lib;
    for (const std::vector<rmstate::Value> &r : objects->rows) {
        const rmstate::Value &card = r[static_cast<size_t>(c_card)];
        if (rmstate::value_is_all_ones(card, card_storage)) continue;
        const uint64_t zone = r[static_cast<size_t>(c_zone)].bits, owner = r[static_cast<size_t>(c_owner)].bits,
                       pos = r[static_cast<size_t>(c_pos)].bits;
        if (zone == hand && owner == opp_seat) out->opp_hand.push_back(static_cast<int64_t>(card.bits));
        if (zone == library && pos < spec.library_depth && (owner == own_seat || owner == opp_seat))
            (owner == own_seat ? own_lib : opp_lib).emplace_back(pos, static_cast<int64_t>(card.bits));
    }
    std::sort(out->opp_hand.begin(), out->opp_hand.end());
    std::sort(own_lib.begin(), own_lib.end());
    std::sort(opp_lib.begin(), opp_lib.end());
    for (const auto &c : own_lib) out->own_library.push_back(c.second);
    for (const auto &c : opp_lib) out->opp_library.push_back(c.second);
    return true;
}

RmtbFile targets_file(const FeatureSpec &spec, const std::vector<Targets> &list) {
    RmtbFile f;
    f.kind = "targets";
    f.has_vocab_hash = f.has_spec_hash = true;
    f.vocab_hash = spec.vocab_hash;
    f.spec_hash = spec.hash;
    std::vector<int64_t> seat, game, decision, sideboarding, opp_list, opp_archetype;
    for (const Targets &t : list) {
        seat.push_back(t.seat);
        game.push_back(t.game);
        decision.push_back(t.decision);
        sideboarding.push_back(t.sideboarding ? 1 : 0);
        opp_list.push_back(t.opp_list);
        opp_archetype.push_back(t.opp_archetype);
    }
    const uint64_t d = list.size();
    f.arrays.push_back(rmtb_ints("seat", Dtype::I64, {d}, seat));
    f.arrays.push_back(rmtb_ints("game", Dtype::I64, {d}, game));
    f.arrays.push_back(rmtb_ints("decision", Dtype::I64, {d}, decision));
    f.arrays.push_back(rmtb_ints("sideboarding", Dtype::I64, {d}, sideboarding));
    f.arrays.push_back(rmtb_ints("opp_list", Dtype::I64, {d}, opp_list));
    f.arrays.push_back(rmtb_ints("opp_archetype", Dtype::I64, {d}, opp_archetype));
    append_lists(&f, "opp_hand", list, &Targets::opp_hand);
    append_lists(&f, "own_library", list, &Targets::own_library);
    append_lists(&f, "opp_library", list, &Targets::opp_library);
    return f;
}

}  // namespace featurize
