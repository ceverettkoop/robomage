#include "frames.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <tuple>

#include "../../src/agents/py_random.h"
#include "util.h"

// A STATE frame's number in its stream: (viewer, game, sideboarding, decision, terminal).
using FrameKey = std::tuple<int, uint64_t, uint64_t, uint64_t, uint64_t>;

static bool cells_equal(const std::vector<rmstate::Value> &a, const std::vector<rmstate::Value> &b, int skip);
static bool tables_equal_without_events(
    const rmstate::DecodedFrame &a, const rmstate::DecodedFrame &b, const rmstate::Schema &schema, std::string *diff);
static void append_events(const rmstate::DecodedFrame &f, const rmstate::Schema &schema, std::vector<std::string> *out);
static std::string cells_text(const std::vector<rmstate::Value> &cells, int skip);
static FrameKey frame_key(int viewer, const rmstate::DecodedFrame &f, const rmstate::Schema &schema);
static std::string key_text(const FrameKey &k);

static bool cells_equal(const std::vector<rmstate::Value> &a, const std::vector<rmstate::Value> &b, int skip) {
    if (a.size() != b.size()) return false;
    for (size_t c = 0; c < a.size(); ++c) {
        if (static_cast<int>(c) == skip) continue;
        if (a[c].bits != b[c].bits || a[c].text != b[c].text) return false;
    }
    return true;
}

// The frames' tables (apart from those named event*) hold the same rows; `diff` lists the tables
// that differ.
static bool tables_equal_without_events(
    const rmstate::DecodedFrame &a, const rmstate::DecodedFrame &b, const rmstate::Schema &schema, std::string *diff) {
    std::set<uint16_t> ids;
    for (const auto &t : a.tables) ids.insert(t.id);
    for (const auto &t : b.tables) ids.insert(t.id);
    std::vector<std::string> differ;
    for (uint16_t id : ids) {
        const rmstate::TableDef *def = schema.find_table(id);
        const std::string name = def ? def->name : std::to_string(id);
        if (rmcheck::starts_with(name, "event")) continue;
        const rmstate::DecodedTable *ta = a.table(id), *tb = b.table(id);
        bool same = ta && tb && ta->rows.size() == tb->rows.size();
        for (size_t r = 0; same && r < ta->rows.size(); ++r) same = cells_equal(ta->rows[r], tb->rows[r], -1);
        if (!same) differ.push_back(name);
    }
    *diff = rmcheck::join(differ, ", ");
    return differ.empty();
}

static std::string cells_text(const std::vector<rmstate::Value> &cells, int skip) {
    std::string out;
    for (size_t c = 0; c < cells.size(); ++c) {
        if (static_cast<int>(c) == skip) continue;
        out += std::to_string(cells[c].bits);
        out += '|';
        out += cells[c].text;
        out += '\x1f';
    }
    return out;
}

// One line per event of the frame: its row, then its cards' and players' rows with their event
// links dropped.
static void append_events(
    const rmstate::DecodedFrame &f, const rmstate::Schema &schema, std::vector<std::string> *out) {
    const rmcheck::Rows events(f, schema, "event"), cards(f, schema, "event_card"), players(f, schema, "event_player");
    const int c_ev = cards.col("event"), p_ev = players.col("event");
    for (size_t e = 0; e < events.size(); ++e) {
        std::string line = "E" + cells_text(events.row(e), -1);
        for (size_t r = 0; r < cards.size(); ++r)
            if (cards.u(r, c_ev) == e) line += "C" + cells_text(cards.row(r), c_ev);
        for (size_t r = 0; r < players.size(); ++r)
            if (players.u(r, p_ev) == e) line += "P" + cells_text(players.row(r), p_ev);
        out->push_back(line);
    }
}

static FrameKey frame_key(int viewer, const rmstate::DecodedFrame &f, const rmstate::Schema &schema) {
    return FrameKey(viewer, rmcheck::header_value(f, schema, "game"), rmcheck::header_value(f, schema, "sideboarding"),
        rmcheck::header_value(f, schema, "decision"), rmcheck::header_value(f, schema, "terminal"));
}

static std::string key_text(const FrameKey &k) {
    return std::string("(") + rmcheck::viewer_name(std::get<0>(k)) + ", game " + std::to_string(std::get<1>(k)) +
           ", sideboarding " + std::to_string(std::get<2>(k)) + ", decision " + std::to_string(std::get<3>(k)) +
           ", terminal " + std::to_string(std::get<4>(k)) + ")";
}

namespace rmcheck {

int frame_kind(const std::string &payload) {
    return payload.size() < 8 ? -1 : static_cast<unsigned char>(payload[6]);
}

int frame_viewer(const std::string &payload) {
    return payload.size() < 8 ? -1 : static_cast<unsigned char>(payload[7]);
}

const char *kind_name(int kind) {
    static const char *names[] = {"SCHEMA", "CARDS", "DECKLISTS", "STATE"};
    return kind >= 0 && kind < 4 ? names[kind] : "?";
}

const char *viewer_name(int viewer) {
    static const char *names[] = {"A", "B", "OMNISCIENT"};
    return viewer >= 0 && viewer < 3 ? names[viewer] : "?";
}

bool read_records(const std::string &path, std::vector<std::string> *payloads, std::string *err) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        *err = path + ": cannot open";
        return false;
    }
    rmstate::RecordReader reader(f);
    std::string payload;
    rmstate::DecodeResult res;
    payloads->clear();
    while (reader.next(&payload, &res)) payloads->push_back(payload);
    std::fclose(f);
    if (!res.ok()) {
        *err = path + ": " + res.message;
        return false;
    }
    return true;
}

bool write_records(const std::string &path, const std::vector<std::string> &payloads) {
    std::string data;
    for (const std::string &p : payloads) data += "RMSTATE " + std::to_string(p.size()) + "\n" + p;
    return write_file(path, data);
}

bool FrameDecoder::feed(const std::string &payload, rmstate::DecodedFrame *out, std::string *err) {
    const int viewer = frame_viewer(payload);
    if (viewer < 0 || viewer > 2) {
        *err = "a frame with no viewer";
        return false;
    }
    if (frame_kind(payload) == KIND_SCHEMA) {
        rmstate::DecodeResult r = rmstate::decode_schema_frame(payload, &schemas_[viewer]);
        if (!r.ok()) {
            *err = "SCHEMA frame: " + r.message;
            return false;
        }
        have_[viewer] = true;
    }
    const rmstate::Schema *schema = this->schema(viewer);
    if (!schema) {
        *err = std::string("a ") + kind_name(frame_kind(payload)) + " frame before any SCHEMA frame";
        return false;
    }
    rmstate::DecodeResult r = rmstate::decode_frame(payload, *schema, out);
    if (!r.ok()) {
        *err = std::string(kind_name(frame_kind(payload))) + " frame: " + r.message;
        return false;
    }
    return true;
}

const rmstate::Schema *FrameDecoder::schema(int viewer) const {
    if (viewer >= 0 && viewer < 3 && have_[viewer]) return &schemas_[viewer];
    for (int v = 0; v < 3; ++v)
        if (have_[v]) return &schemas_[v];
    return nullptr;
}

const rmstate::TableDef *table_named(const rmstate::Schema &schema, const char *name) {
    for (const rmstate::TableDef &def : schema.tables)
        if (def.name == name) return &def;
    return nullptr;
}

Rows::Rows(const rmstate::DecodedFrame &frame, const rmstate::Schema &schema, const char *table)
    : schema_(schema), def_(table_named(schema, table)) {
    if (def_) table_ = frame.table(static_cast<uint16_t>(def_->id));
}

int Rows::col(const char *name) const {
    if (!def_) return -1;
    for (size_t c = 0; c < def_->columns.size(); ++c)
        if (def_->columns[c].name == name) return static_cast<int>(c);
    return -1;
}

bool Rows::null(size_t row, int col) const {
    if (!table_ || col < 0) return true;
    const rmstate::ColumnDef &c = def_->columns[static_cast<size_t>(col)];
    const bool nullable =
        c.tag == "card" || c.tag == "index" || starts_with(c.tag, "row:") || starts_with(c.tag, "enum:");
    return nullable && rmstate::value_is_all_ones(table_->rows[row][static_cast<size_t>(col)], c.storage);
}

uint64_t Rows::u(size_t row, int col) const {
    if (!table_ || col < 0) return 0;
    return table_->rows[row][static_cast<size_t>(col)].bits;
}

int64_t Rows::i(size_t row, int col) const {
    if (!table_ || col < 0) return 0;
    return rmstate::value_signed(
        table_->rows[row][static_cast<size_t>(col)], def_->columns[static_cast<size_t>(col)].storage);
}

const std::string &Rows::text(size_t row, int col) const {
    static const std::string empty;
    if (!table_ || col < 0) return empty;
    return table_->rows[row][static_cast<size_t>(col)].text;
}

std::string Rows::name(size_t row, int col) const {
    if (null(row, col)) return "";
    const rmstate::ColumnDef &c = def_->columns[static_cast<size_t>(col)];
    const uint64_t v = u(row, col);
    if (starts_with(c.tag, "enum:")) {
        const rmstate::EnumTable *e = schema_.find_enum(c.tag.substr(5));
        if (e && v < e->values.size()) return e->values[static_cast<size_t>(v)];
    }
    return std::to_string(v);
}

uint64_t header_value(const rmstate::DecodedFrame &frame, const rmstate::Schema &schema, const char *col) {
    const Rows h(frame, schema, "header");
    return h.size() ? h.u(0, h.col(col)) : 0;
}

StreamRun run_stream(const ProcSpec &spec, const std::function<int(const StreamDecision &)> &reply) {
    StreamRun run;
    Session s;
    std::string err;
    if (!s.start(spec, &err)) {
        run.error = "could not start the engine: " + err;
        run.proc.err = err;
        return run;
    }
    FrameDecoder dec;
    std::string payload;
    while (s.next_record(&payload, &err)) {
        run.payloads.push_back(payload);
        rmstate::DecodedFrame frame;
        if (!dec.feed(payload, &frame, &err)) {
            run.error = "record " + std::to_string(run.payloads.size()) + ": " + err;
            break;
        }
        if (frame_kind(payload) != KIND_STATE) continue;
        const int viewer = frame_viewer(payload);
        const rmstate::Schema &schema = *dec.schema(viewer);
        if (header_value(frame, schema, "terminal")) {
            ++run.terminals;
            continue;
        }
        StreamDecision d;
        d.viewer = viewer;
        d.decision = header_value(frame, schema, "decision");
        d.game = header_value(frame, schema, "game");
        d.sideboarding = header_value(frame, schema, "sideboarding") != 0;
        d.frame = &frame;
        d.schema = &schema;
        const Rows menu(frame, schema, "menu");
        const int c_desc = menu.col("description");
        for (size_t r = 0; r < menu.size(); ++r) d.descriptions.push_back(menu.text(r, c_desc));
        if (d.descriptions.empty()) {
            run.error = "a decision frame with no menu (decision " + std::to_string(d.decision) + ")";
            break;
        }
        const int answer = reply(d);
        run.replies.push_back(answer);
        if (!s.write_line(std::to_string(answer))) break;
    }
    if (run.error.empty() && !err.empty()) run.error = err;
    run.proc = s.finish();
    return run;
}

std::function<int(const StreamDecision &)> random_replies(uint64_t seed, uint64_t cap) {
    std::shared_ptr<agents::PyRandom> rng = std::make_shared<agents::PyRandom>(seed);
    return [rng, cap](const StreamDecision &d) -> int {
        if (d.decision > cap) return -2;
        if (rng->random() < 0.05) return -1;
        return static_cast<int>(rng->randrange(d.descriptions.size()));
    };
}

std::vector<std::string> check_stream(
    const std::vector<std::string> &stream, const std::vector<std::string> &dump, const std::string &label) {
    std::vector<std::string> fails;
    std::string err;
    FrameDecoder ddec;
    std::map<FrameKey, rmstate::DecodedFrame> dump_state;
    std::map<std::pair<int, int>, std::string> dump_match;
    std::map<int, std::vector<std::string>> dump_events;
    for (const std::string &p : dump) {
        rmstate::DecodedFrame f;
        if (!ddec.feed(p, &f, &err)) {
            fails.push_back(label + ": the dump does not decode: " + err);
            return fails;
        }
        const int viewer = frame_viewer(p);
        if (frame_kind(p) == KIND_STATE) {
            const FrameKey key = frame_key(viewer, f, *ddec.schema(viewer));
            if (dump_state.count(key)) fails.push_back(label + ": two dump frames are numbered " + key_text(key));
            append_events(f, *ddec.schema(viewer), &dump_events[viewer]);
            dump_state[key] = std::move(f);
        } else {
            dump_match.emplace(std::make_pair(viewer, frame_kind(p)), p);
        }
    }
    FrameDecoder sdec;
    std::set<std::pair<int, int>> seen_match;
    std::set<FrameKey> stream_keys;
    std::map<int, std::vector<std::string>> stream_events;
    std::map<int, size_t> last_terminal;
    for (size_t i = 0; i < stream.size(); ++i) {
        const std::string &p = stream[i];
        rmstate::DecodedFrame f;
        if (!sdec.feed(p, &f, &err)) {
            fails.push_back(label + ": stream record " + std::to_string(i + 1) + " does not decode: " + err);
            return fails;
        }
        const int viewer = frame_viewer(p), kind = frame_kind(p);
        if (viewer != VIEWER_A && viewer != VIEWER_B) {
            fails.push_back(label + ": record " + std::to_string(i + 1) + " is for viewer " + viewer_name(viewer));
            continue;
        }
        if (kind != KIND_STATE) {
            seen_match.insert({viewer, kind});
            auto it = dump_match.find({viewer, kind});
            if (it == dump_match.end() || it->second != p)
                fails.push_back(
                    label + ": " + viewer_name(viewer) + "'s " + kind_name(kind) + " frame differs from the dump's");
            continue;
        }
        for (int k : {KIND_SCHEMA, KIND_CARDS, KIND_DECKLISTS})
            if (!seen_match.count({viewer, k})) {
                fails.push_back(label + ": record " + std::to_string(i + 1) + ": " + viewer_name(viewer) +
                                "'s STATE frame before its match frames");
                break;
            }
        const rmstate::Schema &schema = *sdec.schema(viewer);
        const FrameKey key = frame_key(viewer, f, schema);
        if (!stream_keys.insert(key).second)
            fails.push_back(label + ": two stream frames are numbered " + key_text(key));
        auto d = dump_state.find(key);
        std::string diff;
        if (d == dump_state.end())
            fails.push_back(label + ": no dump frame for " + key_text(key));
        else if (!tables_equal_without_events(d->second, f, schema, &diff))
            fails.push_back(label + ": frame " + key_text(key) + " differs from the dump's in " + diff);
        append_events(f, schema, &stream_events[viewer]);
        if (std::get<4>(key)) last_terminal[viewer] = i;
    }
    for (int seat : {VIEWER_A, VIEWER_B}) {
        if (!last_terminal.count(seat)) fails.push_back(label + ": no terminal frame for " + viewer_name(seat));
        if (stream_events[seat] != dump_events[seat])
            fails.push_back(label + ": " + viewer_name(seat) + "'s stream events differ from the dump's");
    }
    if (!last_terminal.empty()) {
        size_t lo = stream.size();
        for (const auto &t : last_terminal) lo = std::min(lo, t.second);
        if (lo + 2 < stream.size()) fails.push_back(label + ": a terminal frame is not at the end of the stream");
    }
    return fails;
}

}  // namespace rmcheck
