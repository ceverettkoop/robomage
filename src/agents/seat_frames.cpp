#include "seat_frames.h"

namespace agents {

static const rmstate::TableDef *table_def_named(const rmstate::Schema &schema, const char *name);

static const rmstate::TableDef *table_def_named(const rmstate::Schema &schema, const char *name) {
    for (const rmstate::TableDef &def : schema.tables)
        if (def.name == name) return &def;
    return nullptr;
}

Table::Table(const rmstate::DecodedFrame &frame, const rmstate::Schema &schema, const char *name)
    : def_(table_def_named(schema, name)) {
    if (def_) table_ = frame.table(static_cast<uint16_t>(def_->id));
}

int Table::col(const char *name) const {
    if (!def_) return -1;
    for (size_t i = 0; i < def_->columns.size(); ++i)
        if (def_->columns[i].name == name) return static_cast<int>(i);
    return -1;
}

uint64_t Table::u(size_t row, int col) const {
    if (!table_ || col < 0) return 0;
    return table_->rows[row][static_cast<size_t>(col)].bits;
}

int64_t Table::s(size_t row, int col) const {
    if (!table_ || col < 0) return 0;
    return rmstate::value_signed(table_->rows[row][static_cast<size_t>(col)], def_->columns[col].storage);
}

const std::string &Table::text(size_t row, int col) const {
    static const std::string empty;
    if (!table_ || col < 0) return empty;
    return table_->rows[row][static_cast<size_t>(col)].text;
}

int Table::card(size_t row, int col) const {
    if (!table_ || col < 0) return -1;
    const rmstate::Value &v = table_->rows[row][static_cast<size_t>(col)];
    return rmstate::value_is_all_ones(v, def_->columns[col].storage) ? -1 : static_cast<int>(v.bits);
}

void CardNames::read(const rmstate::DecodedFrame &cards, const rmstate::Schema &schema) {
    front_.clear();
    back_.clear();
    back_card_.clear();
    mana_value_.clear();
    face_name_.clear();
    const Table t(cards, schema, "card");
    const int c_card = t.col("card"), c_face = t.col("face"), c_face_card = t.col("face_card"), c_name = t.col("name"),
              c_mv = t.col("mana_value");
    for (size_t r = 0; r < t.rows(); ++r) {
        const int cid = t.card(r, c_card);
        const std::string &name = t.text(r, c_name);
        if (t.u(r, c_face) != 0) {
            back_.emplace(cid, name);
            back_card_[cid] = t.card(r, c_face_card);
        } else {
            front_.emplace(cid, name);
        }
        mana_value_.emplace(name, static_cast<double>(t.u(r, c_mv)));
    }
    // Front rows first: a split card's second half names the card's own id, which is its front row's.
    for (int pass = 0; pass < 2; ++pass)
        for (size_t r = 0; r < t.rows(); ++r)
            if ((t.u(r, c_face) != 0) == (pass == 1)) {
                const int fid = t.card(r, c_face_card);
                if (fid >= 0) face_name_.emplace(fid, t.text(r, c_name));
            }
}

std::string CardNames::name(int cid, bool back) const {
    if (back) {
        auto it = back_.find(cid);
        if (it != back_.end()) return it->second;
    }
    auto it = front_.find(cid);
    return it != front_.end() ? it->second : "#" + std::to_string(cid);
}

int CardNames::back_face(int cid) const {
    auto it = back_card_.find(cid);
    return it != back_card_.end() ? it->second : cid;
}

std::string CardNames::face_name(int face_card) const {
    auto it = face_name_.find(face_card);
    return it != face_name_.end() ? it->second : "#" + std::to_string(face_card);
}

double CardNames::mana_value(const std::string &name) const {
    auto it = mana_value_.find(name);
    return it != mana_value_.end() ? it->second : 0.0;
}

bool SeatStream::feed_match_frame(const std::string &payload, std::string *err) {
    if (!have_schema_) {
        const rmstate::DecodeResult res = rmstate::decode_schema_frame(payload, &schema_);
        if (!res.ok()) {
            *err = "the SCHEMA frame does not decode: " + res.message;
            return false;
        }
        have_schema_ = true;
        return true;
    }
    rmstate::DecodedFrame frame;
    const rmstate::DecodeResult res = rmstate::decode_frame(payload, schema_, &frame);
    if (!res.ok()) {
        *err = "a match frame does not decode: " + res.message;
        return false;
    }
    if (frame.header.kind == static_cast<uint8_t>(rmstate::FrameKind::CARDS)) cards_.read(frame, schema_);
    return true;
}

bool SeatStream::decode_state(const std::string &payload, rmstate::DecodedFrame *out, std::string *err) const {
    const rmstate::DecodeResult res = rmstate::decode_frame(payload, schema_, out);
    if (!res.ok()) {
        *err = "a STATE frame does not decode: " + res.message;
        return false;
    }
    return true;
}

int SeatStream::enum_value(const char *enum_name, const char *name) const {
    const std::vector<std::string> &values = enum_values(enum_name);
    for (size_t v = 0; v < values.size(); ++v)
        if (values[v] == name) return static_cast<int>(v);
    return -1;
}

const std::string &SeatStream::enum_name(const char *enum_name, uint64_t v) const {
    static const std::string unknown = "?";
    const std::vector<std::string> &values = enum_values(enum_name);
    return v < values.size() ? values[v] : unknown;
}

const std::vector<std::string> &SeatStream::enum_values(const char *enum_name) const {
    static const std::vector<std::string> none;
    const rmstate::EnumTable *t = schema_.find_enum(enum_name);
    return t ? t->values : none;
}

}  // namespace agents
