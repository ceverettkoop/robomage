#include "targets.h"

#include <algorithm>
#include <utility>

#include "columns.h"

static const rmstate::DecodedTable *frame_table(
    const rmstate::Schema &schema, const rmstate::DecodedFrame &frame, const char *name, const rmstate::TableDef **def);
static bool header_of(const featurize::FeatureSpec &spec, const rmstate::DecodedFrame &frame, int64_t *game,
    int64_t *decision, bool *sideboarding);
static int64_t enum_value(const rmstate::Schema &schema, const char *enum_name, const char *value);
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

bool decision_targets(const FeatureSpec &spec, const rmstate::Schema &schema, const rmstate::DecodedFrame &seat_state,
    const rmstate::DecodedFrame &omniscient_state, Targets *out, std::string *err) {
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
    std::vector<int64_t> seat, game, decision, sideboarding;
    for (const Targets &t : list) {
        seat.push_back(t.seat);
        game.push_back(t.game);
        decision.push_back(t.decision);
        sideboarding.push_back(t.sideboarding ? 1 : 0);
    }
    const uint64_t d = list.size();
    f.arrays.push_back(rmtb_ints("seat", Dtype::I64, {d}, seat));
    f.arrays.push_back(rmtb_ints("game", Dtype::I64, {d}, game));
    f.arrays.push_back(rmtb_ints("decision", Dtype::I64, {d}, decision));
    f.arrays.push_back(rmtb_ints("sideboarding", Dtype::I64, {d}, sideboarding));
    append_lists(&f, "opp_hand", list, &Targets::opp_hand);
    append_lists(&f, "own_library", list, &Targets::own_library);
    append_lists(&f, "opp_library", list, &Targets::opp_library);
    return f;
}

}  // namespace featurize
