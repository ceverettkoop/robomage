#include <algorithm>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <tuple>

#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/json.h"
#include "../../src/featurize/setup.h"
#include "../../src/featurize/targets.h"
#include "corpus.h"
#include "frames.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "sessions.h"
#include "util.h"

using featurize::Featurization;
using featurize::Tensors;

namespace {

// One seat frame's facts its targets are checked against.
struct SeatFacts {
        bool valid = false, has_menu = false;
        int64_t game = 0, decision = 0;
        bool sideboarding = false;
        rmstate::DecodedFrame lite;  // the frame header and its header table
        uint64_t opp_hand = 0, own_library = 0, opp_library = 0;
        bool has_templates = false;
        int64_t registered = -1;  // the k_template row marking the seat's own list registered
};

// What one dump's checks found; with `keep`, every decision's tensors and targets.
struct DumpResult {
        size_t decisions = 0, seat = 0, omniscient = 0, permuted = 0, targets = 0, tokens = 0, relations = 0;
        double featurize_s = 0;
        std::vector<std::string> fails;
        std::vector<Tensors> tensors[3];
        std::vector<featurize::Targets> targets_of[2];
};

// Every this many decisions of a dump, the determinism and row-order checks.
const size_t SAMPLE_EVERY = 3;
const size_t MAX_FAILS = 20;

}  // namespace

static std::vector<rmcheck::FuzzGame> featurize_games();
static std::string out_path(const rmcheck::Options &opt, const std::string &name);
static bool load_inputs(const rmcheck::Options &opt, rmcheck::Report &rep, Featurization *f, std::string *cards);
static void tables_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::string &cards, const std::string &tables_dir);
static bool same_tensors(const Tensors &a, const Tensors &b);
static std::string check_shapes(
    const featurize::FeatureSpec &spec, const featurize::CardTables &tables, const Tensors &t, size_t menu_rows);
static size_t table_rows(const rmstate::DecodedFrame &frame, uint16_t id);
static std::vector<int64_t> row_cat_ids(
    const featurize::FeatureSpec &spec, const featurize::TokenType &type, const std::vector<rmstate::Value> &row);
static int32_t type_named(const featurize::FeatureSpec &spec, const std::string &name);
static std::string check_deltas(
    const Featurization &f, const rmstate::DecodedFrame &state, const Tensors &t, const std::vector<int64_t> &first);
static std::string check_ledger(const Featurization &f, const rmstate::Schema &schema,
    const rmstate::DecodedFrame &state, const Tensors &t, const std::vector<int64_t> &first);
static std::string check_pivots(const Featurization &f, const rmstate::DecodedFrame &state, const Tensors &t);
static std::string check_folds(
    const Featurization &f, const rmstate::Schema &schema, const rmstate::DecodedFrame &state, const Tensors &t);
static std::string check_permuted(const Featurization &f, const rmstate::Schema &schema,
    const rmstate::DecodedFrame &state, const Tensors &t, uint32_t seed);
static SeatFacts seat_facts(
    const Featurization &f, const rmstate::Schema &schema, const rmstate::DecodedFrame &state, size_t menu_rows);
static std::string check_targets(const featurize::Targets &t, const SeatFacts &own, const SeatFacts &opp, size_t depth);
static void add_fail(DumpResult *r, const std::string &label, const std::string &msg);
static DumpResult check_dump(const Featurization &f, const std::string &path, const std::string &label, bool keep);
static void report_dump(rmcheck::Report &rep, const std::string &label, const DumpResult &r);
static void add_totals(DumpResult *total, const DumpResult &r);
static bool same_decision(const SeatFacts &a, const SeatFacts &b);
static void corpus_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f, DumpResult *total);
static std::string games_part(
    const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f, DumpResult *total, DumpResult *kept);
static std::string file_bytes(const std::string &path);
static void cli_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::string &cards, const std::string &tables_dir, const std::string &dump, const DumpResult &kept);
static std::string digests_json(const std::string &path, const featurize::RmtbFile &file);
static void python_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::string &tables_dir, const DumpResult &kept);

// Explore games under --selfcheck-featurize, with dumps: bo1 and bo3, a closed decklist, and seat A at
// level expert+belief (the b_* token types).
static std::vector<rmcheck::FuzzGame> featurize_games() {
    struct Row {
            const char *a, *b;
            bool bo3;
            long seed;
            const char *flag;
    };
    static const Row rows[] = {
        {"ur_delver", "bug", false, 301, nullptr},
        {"reanimator", "wrb_energy", true, 302, nullptr},
        {"lands", "bw_dnt", false, 303, "--closed-decklist"},
        {"gw_maverick", "uw_control", false, 304, "expert+belief"},
    };
    std::vector<rmcheck::FuzzGame> out;
    for (const Row &r : rows) {
        rmcheck::FuzzGame g;
        g.deck_a = std::string("league/") + r.a;
        g.deck_b = std::string("league/") + r.b;
        g.bo3 = r.bo3;
        g.seed = r.seed;
        g.flags = {"--selfcheck-featurize"};
        if (r.flag && std::string(r.flag) == "expert+belief") {
            g.flags.insert(g.flags.end(), {"--knowledge-a", "expert+belief"});
        } else if (r.flag) {
            g.flags.push_back(r.flag);
        }
        g.dump = true;
        out.push_back(g);
    }
    return out;
}

static std::string out_path(const rmcheck::Options &opt, const std::string &name) {
    return rmcheck::path_join(opt.out("featurize"), name);
}

// The whole-vocab cards file (robomage --cards-frame) and the featurizer's inputs built from it.
static bool load_inputs(const rmcheck::Options &opt, rmcheck::Report &rep, Featurization *f, std::string *cards) {
    *cards = out_path(opt, "cards.rmst");
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--cards-frame", *cards};
    spec.cwd = opt.bin_dir;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    std::vector<rmcheck::ScanLine> errors;
    rmcheck::scan_transcript(r.out + r.err, &errors, nullptr);
    if (!r.ok() || !errors.empty()) {
        rep.error(
            "featurize", "robomage --cards-frame " + r.status_text() + (errors.empty() ? "" : ": " + errors[0].line));
        return false;
    }
    std::string err;
    if (!f->load(*cards, rmcheck::path_join(opt.bin_dir, "resources/embeddings/oracle_text.rmtb"), "", &err)) {
        rep.error("featurize", "the featurizer's inputs: " + err);
        return false;
    }
    return true;
}

// The exported tables read back give the same tables and spec; rmfeat's --spec and --export-tables
// write the same bytes.
static void tables_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::string &cards, const std::string &tables_dir) {
    std::string err;
    rmcheck::remove_tree(tables_dir);
    rmcheck::make_dirs(tables_dir);
    if (!f.export_tables(tables_dir, &err)) {
        rep.error("featurize", "export: " + err);
        return;
    }
    Featurization back;
    const std::string text = rmcheck::path_join(opt.bin_dir, "resources/embeddings/oracle_text.rmtb");
    if (!back.load(cards, text, tables_dir, &err)) {
        rep.error("featurize", "the exported tables do not load: " + err);
        return;
    }
    const featurize::CardTableFiles a = featurize::card_table_files(f.tables, f.spec.hash),
                                    b = featurize::card_table_files(back.tables, back.spec.hash);
    if (back.spec.json != f.spec.json || featurize::rmtb_encode(a.card) != featurize::rmtb_encode(b.card) ||
        featurize::rmtb_encode(a.ability) != featurize::rmtb_encode(b.ability) ||
        featurize::rmtb_encode(a.part) != featurize::rmtb_encode(b.part))
        rep.error("featurize", "the exported tables read back as other tables or another spec");
    rmcheck::ProcSpec spec;
    spec.argv = {opt.rmfeat, "--spec", "--cards", cards, "--text-table", text};
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    if (!r.ok() || r.out != f.spec.json + "\n")
        rep.error("featurize", "rmfeat --spec differs (" + r.status_text() + ")");
    const std::string cli_dir = out_path(opt, "tables_cli");
    rmcheck::remove_tree(cli_dir);
    rmcheck::make_dirs(cli_dir);
    spec.argv = {opt.rmfeat, "--export-tables", cards, cli_dir, "--text-table", text};
    const rmcheck::ProcResult e = rmcheck::run_process(spec);
    for (const char *name :
        {Featurization::CARD_FILE, Featurization::ABILITY_FILE, Featurization::PART_FILE, Featurization::SPEC_FILE})
        if (!e.ok() ||
            file_bytes(rmcheck::path_join(cli_dir, name)) != file_bytes(rmcheck::path_join(tables_dir, name)))
            rep.error(
                "featurize", std::string("rmfeat --export-tables: ") + name + " differs (" + e.status_text() + ")");
    rmcheck::remove_tree(cli_dir);
    const featurize::FeatureSpec &s = f.spec;
    rep.info("OK   spec " + featurize::hex16(s.hash) + ": " + std::to_string(s.types.size()) +
             " token types, F=" + std::to_string(s.F) + " C=" + std::to_string(s.C) + " K=" + std::to_string(s.K) +
             " Q=" + std::to_string(s.Q) + ", " + std::to_string(s.relations.size()) +
             " relation types, categorical size " + std::to_string(s.categorical_size));
    rep.info("OK   tables: card [" + std::to_string(f.tables.card_rows) + ", " +
             std::to_string(f.tables.card_columns.size()) + "], ability [" + std::to_string(f.tables.ability_rows()) +
             ", " + std::to_string(f.tables.ability_columns.size()) + "] (" + std::to_string(f.tables.ab_keys.size()) +
             " keys), part [" + std::to_string(f.tables.part_rows()) + ", " +
             std::to_string(f.tables.part_columns.size()) + "], text [" + std::to_string(f.tables.text_rows) + ", " +
             std::to_string(f.tables.text_width) + "]; exported, read back and rmfeat's the same");
}

static bool same_tensors(const Tensors &a, const Tensors &b) {
    return a.viewer == b.viewer && a.game == b.game && a.decision == b.decision && a.sideboarding == b.sideboarding &&
           a.tok_type == b.tok_type && a.tok_num == b.tok_num && a.tok_cat == b.tok_cat && a.tok_card == b.tok_card &&
           a.tok_ability == b.tok_ability && a.tok_mask == b.tok_mask && a.rel == b.rel && a.act_tok == b.act_tok &&
           a.act_mask == b.act_mask;
}

// Shapes and indices against the spec and the tables; "" when they agree.
static std::string check_shapes(
    const featurize::FeatureSpec &spec, const featurize::CardTables &tables, const Tensors &t, size_t menu_rows) {
    const size_t n = t.n();
    if (t.tok_num.size() != n * spec.F || t.tok_cat.size() != n * spec.C || t.tok_card.size() != n * spec.K ||
        t.tok_ability.size() != n * spec.Q || t.tok_mask.size() != n || t.rel.size() % 3 != 0)
        return "a tensor's width is not the spec's";
    if (t.a() != menu_rows || t.act_mask.size() != menu_rows) return "act_tok has not one entry per menu row";
    for (size_t i = 0; i < n; ++i) {
        const int64_t ty = t.tok_type[i];
        if (ty < 0 || static_cast<size_t>(ty) >= spec.types.size()) return "tok_type out of range";
        const featurize::TokenType &type = spec.types[static_cast<size_t>(ty)];
        for (size_t c = 0; c < spec.C; ++c) {
            const int64_t id = t.tok_cat[i * spec.C + c];
            if (c >= type.categorical.size()
                    ? id != 0
                    : id < type.categorical[c].offset || id >= type.categorical[c].offset + type.categorical[c].size)
                return type.name + ": tok_cat slot " + std::to_string(c) + " out of its range";
        }
        for (size_t k = 0; k < spec.K; ++k) {
            const int64_t c = t.tok_card[i * spec.K + k];
            if (c < -1 || c >= static_cast<int64_t>(tables.card_rows) || (k >= type.cards.size() && c != -1))
                return type.name + ": tok_card out of the card table";
        }
        for (size_t q = 0; q < spec.Q; ++q) {
            const int64_t a = t.tok_ability[i * spec.Q + q];
            if (a < -1 || a >= static_cast<int64_t>(tables.ability_rows()) || (q >= type.abilities.size() && a != -1))
                return type.name + ": tok_ability out of the ability table";
        }
    }
    for (size_t r = 0; r < t.r(); ++r) {
        const int64_t s = t.rel[3 * r], d = t.rel[3 * r + 1], ty = t.rel[3 * r + 2];
        if (s < 0 || d < 0 || static_cast<size_t>(s) >= n || static_cast<size_t>(d) >= n || ty < 0 ||
            static_cast<size_t>(ty) >= spec.relations.size() || s == d)
            return "a relation's token or type is out of range";
        const featurize::RelationType &rt = spec.relations[static_cast<size_t>(ty)];
        if (spec.types[static_cast<size_t>(t.tok_type[static_cast<size_t>(s)])].name != rt.table)
            return "a relation of type " + rt.table + "." + rt.column + " from another table's token";
    }
    for (const int64_t a : t.act_tok)
        if (a < 0 || static_cast<size_t>(a) >= n || t.tok_type[static_cast<size_t>(a)] != spec.menu_type)
            return "act_tok is not a menu token";
    return "";
}

static size_t table_rows(const rmstate::DecodedFrame &frame, uint16_t id) {
    const rmstate::DecodedTable *t = frame.table(id);
    return t ? t->rows.size() : 0;
}

// A row's categorical ids under a type's CATEGORICAL and HASHED columns, in slot order (the gate's
// own reading of the column rules, docs/featurizer.md §3.2, §3.4).
static std::vector<int64_t> row_cat_ids(
    const featurize::FeatureSpec &spec, const featurize::TokenType &type, const std::vector<rmstate::Value> &row) {
    std::vector<int64_t> out;
    for (const featurize::ColumnOp &op : type.ops) {
        const rmstate::Value &v = row[op.col];
        if (op.kind == featurize::OpKind::CATEGORICAL)
            out.push_back(op.offset + (v.bits < op.width ? static_cast<int64_t>(v.bits) + 1 : 0));
        if (op.kind == featurize::OpKind::HASHED)
            out.push_back(
                op.offset + (v.text.empty() ? 0
                                            : 1 + static_cast<int64_t>(rmstate::fnv1a64(v.text.data(), v.text.size()) %
                                                                       spec.hash_buckets)));
    }
    return out;
}

static int32_t type_named(const featurize::FeatureSpec &spec, const std::string &name) {
    for (size_t i = 0; i < spec.types.size(); ++i)
        if (spec.types[i].name == name) return static_cast<int32_t>(i);
    return -1;
}

// Every object's rows of each delta type, rebuilt from the tensors (its current face's printed rows
// when the object token's face card has a card row, plus the delta tokens related to it, less those
// marked removed; the tokens related to no object for handles no object row has), are the frame's.
static std::string check_deltas(
    const Featurization &f, const rmstate::DecodedFrame &state, const Tensors &t, const std::vector<int64_t> &first) {
    const featurize::FeatureSpec &spec = f.spec;
    if (spec.object_type < 0) return "";
    const featurize::TokenType &ot = spec.types[static_cast<size_t>(spec.object_type)];
    const rmstate::DecodedTable *objects = state.table(ot.table);
    int face_slot = -1;
    for (size_t c = 0; c < ot.cards.size(); ++c)
        if (ot.cards[c].current_face) face_slot = static_cast<int>(c);
    std::map<uint64_t, int64_t> row_of;  // handle -> object row
    for (size_t r = 0; objects && r < objects->rows.size(); ++r)
        row_of[objects->rows[r][static_cast<size_t>(spec.object_handle)].bits] = static_cast<int64_t>(r);
    for (size_t d = 0; d < spec.types.size(); ++d) {
        const featurize::TokenType &dt = spec.types[d];
        if (dt.kind != featurize::TypeKind::DELTA) continue;
        int64_t rel_id = -1;
        for (size_t i = 0; i < spec.relations.size(); ++i)
            if (spec.relations[i].table == dt.name && spec.relations[i].column == "obj")
                rel_id = static_cast<int64_t>(i);
        typedef std::map<int64_t, std::multiset<std::vector<int64_t>>> PerObject;
        PerObject want, got;
        if (const rmstate::DecodedTable *child = state.table(dt.table))
            for (const std::vector<rmstate::Value> &row : child->rows) {
                auto it = row_of.find(row[dt.obj_col].bits);
                want[it == row_of.end() ? -1 : it->second].insert(row_cat_ids(spec, dt, row));
            }
        const featurize::PrintedRows *printed = f.tables.printed_rows(dt.printed);
        for (size_t r = 0; objects && printed && face_slot >= 0 && r < objects->rows.size(); ++r) {
            const size_t tok = static_cast<size_t>(first[static_cast<size_t>(spec.object_type)]) + r;
            const int64_t face = t.tok_card[tok * spec.K + static_cast<size_t>(face_slot)];
            if (face < 0 || static_cast<size_t>(face) >= f.tables.present.size() ||
                !f.tables.present[static_cast<size_t>(face)])
                continue;
            for (uint64_t p = printed->offsets[static_cast<size_t>(face)];
                p < printed->offsets[static_cast<size_t>(face) + 1]; ++p) {
                std::vector<rmstate::Value> row(dt.columns);
                for (size_t i = 0; i < dt.value_cols.size(); ++i) row[dt.value_cols[i]] = printed->rows[p][i];
                got[static_cast<int64_t>(r)].insert(row_cat_ids(spec, dt, row));
            }
        }
        std::map<int64_t, int64_t> object_of;  // delta token -> object row
        for (size_t r = 0; r < t.r(); ++r)
            if (t.rel[3 * r + 2] == rel_id)
                object_of[t.rel[3 * r]] = t.rel[3 * r + 1] - first[static_cast<size_t>(spec.object_type)];
        for (size_t k = 0; k < t.n(); ++k) {
            if (t.tok_type[k] != static_cast<int64_t>(d)) continue;
            const std::vector<int64_t> ids(t.tok_cat.begin() + static_cast<long>(k * spec.C),
                t.tok_cat.begin() + static_cast<long>(k * spec.C + dt.categorical.size()));
            auto it = object_of.find(static_cast<int64_t>(k));
            const int64_t obj = it == object_of.end() ? -1 : it->second;
            const bool removed = t.tok_num[k * spec.F + static_cast<size_t>(dt.removed_slot)] != 0.0f;
            if (!removed) {
                got[obj].insert(ids);
                continue;
            }
            auto at = got[obj].find(ids);
            if (at == got[obj].end()) return dt.name + ": a removed token names no printed row";
            got[obj].erase(at);
        }
        for (PerObject *m : {&want, &got})
            for (auto it = m->begin(); it != m->end();) it = it->second.empty() ? m->erase(it) : std::next(it);
        if (want != got) return dt.name + ": an object's rows rebuilt from the delta tokens are not the frame's";
    }
    return "";
}

// Every folded row's numbers are in its ledger token (folds.h): own_deck, k_seen and k_known_hand
// rows, the k_pool_card rows of HAND and SIDEBOARD pools and of an owner's only library pool (with
// the owner's exact k_position rows of the card); the other k_pool_card rows are tokens; one ledger
// token per (player_rel, card) of those rows.
static std::string check_ledger(const Featurization &f, const rmstate::Schema &schema,
    const rmstate::DecodedFrame &state, const Tensors &t, const std::vector<int64_t> &first) {
    const featurize::FeatureSpec &spec = f.spec;
    const int32_t lt = spec.ledger.type, pt = type_named(spec, "k_pool_card");
    if (lt < 0) return "";
    const featurize::TokenType &ledger = spec.types[static_cast<size_t>(lt)];
    std::map<std::pair<uint64_t, uint64_t>, size_t> tok;  // (player_rel, card) -> token
    for (size_t k = 0; k < t.n(); ++k)
        if (t.tok_type[k] == lt)
            tok[{static_cast<uint64_t>(t.tok_cat[k * spec.C] - ledger.rel_offset - 1),
                static_cast<uint64_t>(t.tok_card[k * spec.K])}] = k;
    std::set<std::pair<uint64_t, uint64_t>> keys;
    std::string bad;
    const auto expect = [&](uint64_t rel, uint64_t card, size_t slot, float value) {
        keys.insert({rel, card});
        auto it = tok.find({rel, card});
        if (it == tok.end()) {
            bad = "no ledger token for a folded row";
        } else if (t.tok_num[it->second * spec.F + slot] != value && bad.empty()) {
            bad = "ledger slot " + ledger.numeric[slot].name + " is not its folded rows' value";
        }
    };
    const auto lg = [](uint64_t x) { return std::log1p(static_cast<float>(x)); };
    const rmcheck::Rows deck(state, schema, "own_deck"), seen(state, schema, "k_seen"),
        known(state, schema, "k_known_hand"), pools(state, schema, "k_pool"), pc(state, schema, "k_pool_card"),
        pos(state, schema, "k_position");
    for (size_t r = 0; r < deck.size(); ++r) {
        const uint64_t rel = deck.u(r, deck.col("player_rel")), card = deck.u(r, deck.col("card"));
        expect(rel, card, featurize::L_DECK, 1.0f);
        expect(rel, card, featurize::L_MAIN, lg(deck.u(r, deck.col("main"))));
        expect(rel, card, featurize::L_SIDE, lg(deck.u(r, deck.col("side"))));
    }
    for (size_t r = 0; r < seen.size(); ++r) {
        const uint64_t rel = seen.u(r, seen.col("player_rel")), card = seen.u(r, seen.col("card"));
        expect(rel, card, featurize::L_SEEN_GAME, lg(seen.u(r, seen.col("game"))));
        expect(rel, card, featurize::L_SEEN_MATCH, lg(seen.u(r, seen.col("match"))));
    }
    for (size_t r = 0; r < known.size(); ++r)
        expect(2, known.u(r, known.col("card")), featurize::L_KNOWN_HAND, lg(known.u(r, known.col("copies"))));
    const std::vector<std::string> &kinds = featurize::enum_values(schema, "pool_kind");
    std::map<uint64_t, int> libraries;
    for (size_t r = 0; r < pools.size(); ++r)
        if (kinds[pools.u(r, pools.col("kind"))] == "LIBRARY_RANGE") ++libraries[pools.u(r, pools.col("owner_rel"))];
    std::map<std::pair<uint64_t, uint64_t>, uint64_t> exact;  // (owner_rel, card) -> exact positions
    for (size_t r = 0; r < pos.size(); ++r)
        if (!pos.null(r, pos.col("card"))) ++exact[{pos.u(r, pos.col("owner_rel")), pos.u(r, pos.col("card"))}];
    size_t kept = 0;
    for (size_t r = 0; r < pc.size(); ++r) {
        const uint64_t p = pc.u(r, pc.col("pool")), card = pc.u(r, pc.col("card"));
        const uint64_t lo = pc.u(r, pc.col("min")), hi = pc.u(r, pc.col("max"));
        const std::string &kind = kinds[pools.u(static_cast<size_t>(p), pools.col("kind"))];
        const uint64_t rel = pools.u(static_cast<size_t>(p), pools.col("owner_rel"));
        keys.insert({rel, card});
        if (kind == "HAND") {
            expect(rel, card, featurize::L_HAND_MIN, lg(lo));
            expect(rel, card, featurize::L_HAND_MAX, lg(hi));
        } else if (kind == "SIDEBOARD") {
            expect(rel, card, featurize::L_SIDEBOARD_MIN, lg(lo));
            expect(rel, card, featurize::L_SIDEBOARD_MAX, lg(hi));
        } else if (kind == "LIBRARY_RANGE" && libraries[rel] == 1) {
            const uint64_t e = exact.count({rel, card}) ? exact[{rel, card}] : 0;
            expect(rel, card, featurize::L_LIBRARY_MIN, lg(lo + e));
            expect(rel, card, featurize::L_LIBRARY_MAX, lg(hi + e));
        } else {
            ++kept;
        }
    }
    for (const auto &e : exact) keys.insert(e.first);
    if (!bad.empty()) return bad;
    const size_t pc_tokens =
        pt >= 0 ? static_cast<size_t>(first[static_cast<size_t>(pt) + 1] - first[static_cast<size_t>(pt)]) : 0;
    if (pt >= 0 && pc_tokens != kept) return "the k_pool_card tokens are not the rows of the unfolded pools";
    if (tok.size() != keys.size()) return "the ledger has a token per (player, card) of no folded row";
    return "";
}

// Every pivot source row's columns are in its player's token at its key's slots.
static std::string check_pivots(const Featurization &f, const rmstate::DecodedFrame &state, const Tensors &t) {
    const featurize::FeatureSpec &spec = f.spec;
    for (size_t p = 0; p < spec.types.size(); ++p) {
        const featurize::TokenType &pt = spec.types[p];
        if (pt.kind != featurize::TypeKind::PIVOT) continue;
        std::map<uint64_t, size_t> tok;
        for (size_t k = 0; k < t.n(); ++k)
            if (t.tok_type[k] == static_cast<int64_t>(p))
                tok[static_cast<uint64_t>(t.tok_cat[k * spec.C] - pt.rel_offset - 1)] = k;
        std::set<uint64_t> rels;
        for (const featurize::PivotSource &src : pt.pivot) {
            const rmstate::DecodedTable *tab = state.table(src.table_id);
            for (size_t r = 0; tab && r < tab->rows.size(); ++r) {
                const std::vector<rmstate::Value> &row = tab->rows[r];
                const uint64_t rel = row[src.rel_col].bits, key = row[src.key_col].bits;
                rels.insert(rel);
                auto it = tok.find(rel);
                if (it == tok.end()) return pt.name + ": no token for a player of " + src.table;
                for (size_t i = 0; key < src.keys && i < src.cols.size(); ++i)
                    if (t.tok_num[it->second * spec.F + src.first_slot + key * src.cols.size() + i] !=
                        featurize::cell_value(src.transforms[i], 0, row[src.cols[i]], src.storages[i]))
                        return pt.name + ": a slot is not its " + src.table + " row's value";
            }
        }
        if (rels.size() != tok.size()) return pt.name + ": a token for a player with no row";
    }
    return "";
}

// The folds lose nothing (docs/featurizer.md §3.6): the deltas, the ledger and the pivots read back.
static std::string check_folds(
    const Featurization &f, const rmstate::Schema &schema, const rmstate::DecodedFrame &state, const Tensors &t) {
    std::vector<int64_t> first(f.spec.types.size() + 1, 0);  // per type its first token (types are contiguous)
    for (size_t k = 0; k < t.n(); ++k) ++first[static_cast<size_t>(t.tok_type[k]) + 1];
    for (size_t i = 0; i < f.spec.types.size(); ++i) first[i + 1] += first[i];
    std::string e = check_deltas(f, state, t, first);
    if (e.empty()) e = check_ledger(f, schema, state, t, first);
    if (e.empty()) e = check_pivots(f, state, t);
    return e;
}

// The row-order check: every token table but the menu (whose row is the reply), the folded ones
// included, with its rows shuffled and every row: reference into it renumbered must featurize to the
// same tensors up to the matching token permutation (each token found by its origin, folds.h: a table
// row moved with its row, a folded token in its canonical place; the same token rows, the same
// relation multiset, act_tok the menu's).
static std::string check_permuted(const Featurization &f, const rmstate::Schema &schema,
    const rmstate::DecodedFrame &state, const Tensors &t, uint32_t seed) {
    const featurize::FeatureSpec &spec = f.spec;
    const uint16_t menu = spec.types[static_cast<size_t>(spec.menu_type)].table;
    std::vector<featurize::TokenSource> src, psrc;
    Tensors again, p;
    std::string err;
    if (!featurize::featurize_traced(spec, f.tables, state, &again, &src, &err)) return "the traced frame: " + err;
    if (!same_tensors(t, again) || src.size() != t.n()) return "the traced featurization differs";
    rmstate::DecodedFrame copy = state;
    std::map<uint16_t, std::vector<int64_t>> inv;  // table id -> old row -> new row
    std::mt19937 gen(seed);
    for (rmstate::DecodedTable &tab : copy.tables) {
        const bool token_table = tab.def->frame_kind == rmstate::FrameKind::STATE &&
                                 ((tab.id >= 20 && tab.id <= 48) || (tab.id >= 90 && tab.id <= 99));
        if (!token_table || tab.id == menu || tab.rows.size() < 2) continue;
        std::vector<size_t> order(tab.rows.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), gen);
        std::vector<std::vector<rmstate::Value>> rows(order.size());
        std::vector<int64_t> &back = inv[tab.id];
        back.assign(order.size(), 0);
        for (size_t j = 0; j < order.size(); ++j) {
            rows[j] = tab.rows[order[j]];
            back[order[j]] = static_cast<int64_t>(j);
        }
        tab.rows = std::move(rows);
    }
    for (rmstate::DecodedTable &tab : copy.tables) {
        for (size_t c = 0; c < tab.def->columns.size(); ++c) {
            const featurize::Tag tag = featurize::parse_tag(tab.def->columns[c].tag);
            const rmstate::TableDef *target =
                tag.kind == featurize::TagKind::ROW ? featurize::table_named(schema, tag.arg) : nullptr;
            auto it = target ? inv.find(static_cast<uint16_t>(target->id)) : inv.end();
            if (it == inv.end()) continue;
            for (std::vector<rmstate::Value> &row : tab.rows)
                if (row[c].bits < it->second.size()) row[c].bits = static_cast<uint64_t>(it->second[row[c].bits]);
        }
    }
    if (!featurize::featurize_traced(spec, f.tables, copy, &p, &psrc, &err)) return "the permuted frame: " + err;
    if (p.n() != t.n() || p.r() != t.r() || p.a() != t.a()) return "the permuted frame has other tensor sizes";
    std::map<std::tuple<int32_t, int32_t, int64_t, int64_t>, int64_t> where;
    for (size_t m = 0; m < psrc.size(); ++m)
        where[std::make_tuple(psrc[m].type, psrc[m].table, psrc[m].row, psrc[m].sub)] = static_cast<int64_t>(m);
    std::vector<int64_t> pi(t.n());
    for (size_t k = 0; k < t.n(); ++k) {
        const featurize::TokenSource &s = src[k];
        auto it = s.table >= 0 ? inv.find(static_cast<uint16_t>(s.table)) : inv.end();
        const int64_t row = it == inv.end() ? s.row : it->second[static_cast<size_t>(s.row)];
        auto at = where.find(std::make_tuple(s.type, s.table, row, s.sub));
        if (at == where.end())
            return spec.types[static_cast<size_t>(s.type)].name + " token " + std::to_string(k) +
                   " has no counterpart once the rows are reordered";
        pi[k] = at->second;
    }
    for (size_t k = 0; k < t.n(); ++k) {
        const size_t m = static_cast<size_t>(pi[k]);
        if (p.tok_type[m] != t.tok_type[k] ||
            !std::equal(t.tok_num.begin() + static_cast<long>(k * spec.F),
                t.tok_num.begin() + static_cast<long>((k + 1) * spec.F),
                p.tok_num.begin() + static_cast<long>(m * spec.F)) ||
            !std::equal(t.tok_cat.begin() + static_cast<long>(k * spec.C),
                t.tok_cat.begin() + static_cast<long>((k + 1) * spec.C),
                p.tok_cat.begin() + static_cast<long>(m * spec.C)) ||
            !std::equal(t.tok_card.begin() + static_cast<long>(k * spec.K),
                t.tok_card.begin() + static_cast<long>((k + 1) * spec.K),
                p.tok_card.begin() + static_cast<long>(m * spec.K)) ||
            !std::equal(t.tok_ability.begin() + static_cast<long>(k * spec.Q),
                t.tok_ability.begin() + static_cast<long>((k + 1) * spec.Q),
                p.tok_ability.begin() + static_cast<long>(m * spec.Q)))
            return spec.types[static_cast<size_t>(t.tok_type[k])].name + " token " + std::to_string(k) +
                   " differs once its table's rows are reordered";
    }
    std::vector<std::tuple<int64_t, int64_t, int64_t>> want, got;
    for (size_t r = 0; r < t.r(); ++r) {
        want.emplace_back(
            pi[static_cast<size_t>(t.rel[3 * r])], pi[static_cast<size_t>(t.rel[3 * r + 1])], t.rel[3 * r + 2]);
        got.emplace_back(p.rel[3 * r], p.rel[3 * r + 1], p.rel[3 * r + 2]);
    }
    std::sort(want.begin(), want.end());
    std::sort(got.begin(), got.end());
    if (want != got) return "the relations differ once the rows are reordered";
    for (size_t a = 0; a < t.a(); ++a)
        if (p.act_tok[a] != pi[static_cast<size_t>(t.act_tok[a])]) return "act_tok differs once the rows are reordered";
    return "";
}

// The facts of a seat frame its targets (and its opponent's) are checked against.
static SeatFacts seat_facts(
    const Featurization &f, const rmstate::Schema &schema, const rmstate::DecodedFrame &state, size_t menu_rows) {
    SeatFacts s;
    s.valid = true;
    s.has_menu = menu_rows > 0;
    const featurize::FeatureSpec &spec = f.spec;
    const uint16_t header_id = spec.types[static_cast<size_t>(spec.header_type)].table;
    s.lite.header = state.header;
    if (const rmstate::DecodedTable *h = state.table(header_id)) {
        s.lite.tables.push_back(*h);
        s.game = static_cast<int64_t>(h->rows[0][static_cast<size_t>(spec.header_game)].bits);
        s.decision = static_cast<int64_t>(h->rows[0][static_cast<size_t>(spec.header_decision)].bits);
        s.sideboarding = h->rows[0][static_cast<size_t>(spec.header_sideboarding)].bits != 0;
    }
    const rmcheck::Rows player(state, schema, "player"), templates(state, schema, "k_template");
    const int p_rel = player.col("player_rel"), p_hand = player.col("hand"), p_lib = player.col("library");
    for (size_t r = 0; r < player.size(); ++r) {
        if (player.u(r, p_rel) == 1) s.own_library = player.u(r, p_lib);
        if (player.u(r, p_rel) == 2) {
            s.opp_hand = player.u(r, p_hand);
            s.opp_library = player.u(r, p_lib);
        }
    }
    const int t_rel = templates.col("player_rel"), t_list = templates.col("list"), t_reg = templates.col("registered");
    s.has_templates = templates.size() > 0;
    for (size_t r = 0; r < templates.size(); ++r)
        if (templates.u(r, t_rel) == 1 && templates.u(r, t_reg) == 1)
            s.registered = static_cast<int64_t>(templates.u(r, t_list));
    return s;
}

// A decision's targets against the seat's frame (the opponent's hand size, the libraries' sizes) and
// the opponent's own frame (the k_template row it marks registered).
static std::string check_targets(
    const featurize::Targets &t, const SeatFacts &own, const SeatFacts &opp, size_t depth) {
    if (t.opp_hand.size() != own.opp_hand) return "opp_hand is not the opponent's hand size";
    if (t.own_library.size() != std::min<uint64_t>(own.own_library, depth) ||
        t.opp_library.size() != std::min<uint64_t>(own.opp_library, depth))
        return "a library target is not the library's top";
    if (opp.has_templates && t.opp_list != opp.registered)
        return "opp_list " + std::to_string(t.opp_list) + " is not the list the opponent's frame marks registered (" +
               std::to_string(opp.registered) + ")";
    return "";
}

static bool same_decision(const SeatFacts &a, const SeatFacts &b) {
    return a.valid && b.valid && a.game == b.game && a.decision == b.decision && a.sideboarding == b.sideboarding;
}

static void add_fail(DumpResult *r, const std::string &label, const std::string &msg) {
    if (r->fails.size() < MAX_FAILS) r->fails.push_back(label + ": " + msg);
}

// Every decision of a dump with a menu: the spec's shapes, the actor entry point's refusal of the
// omniscient frame, each seat decision's targets, and every SAMPLE_EVERY'th decision deterministic
// and invariant to row order.
static DumpResult check_dump(const Featurization &f, const std::string &path, const std::string &label, bool keep) {
    DumpResult r;
    SeatFacts facts[2];
    featurize::RegisteredLists lists;
    bool have_lists = false;
    Tensors t, again;
    std::string err;
    const bool ok = featurize::read_stream(
        path,
        [&](const featurize::ViewerFrames &v, const rmstate::DecodedFrame &state, std::string *e) {
            const size_t menu = table_rows(state, f.spec.types[static_cast<size_t>(f.spec.menu_type)].table);
            const uint8_t viewer = state.header.viewer;
            const bool seat = viewer != static_cast<uint8_t>(rmstate::Viewer::OMNISCIENT);
            if (seat) facts[viewer] = seat_facts(f, v.schema, state, menu);
            if (menu == 0) return true;
            const std::string at = "decision " + std::to_string(r.decisions + 1) + " viewer " + std::to_string(viewer);
            const double t0 = rmcheck::now_seconds();
            bool fed = seat ? featurize::featurize_seat(f.spec, f.tables, state, &t, e)
                            : featurize::featurize_viewer(f.spec, f.tables, state, &t, e);
            r.featurize_s += rmcheck::now_seconds() - t0;
            if (!fed) return false;
            ++r.decisions;
            ++(seat ? r.seat : r.omniscient);
            r.tokens += t.n();
            r.relations += t.r();
            const std::string shapes = check_shapes(f.spec, f.tables, t, menu);
            if (!shapes.empty()) add_fail(&r, label, at + ": " + shapes);
            const std::string folds = check_folds(f, v.schema, state, t);
            if (!folds.empty()) add_fail(&r, label, at + ": " + folds);
            if (!seat && featurize::featurize_seat(f.spec, f.tables, state, &again, &err))
                add_fail(&r, label, at + ": the actor entry point featurized the omniscient frame");
            if (r.decisions % SAMPLE_EVERY == 0) {
                if (!featurize::featurize_viewer(f.spec, f.tables, state, &again, e) || !same_tensors(t, again))
                    add_fail(&r, label, at + ": featurizing twice gives other tensors");
                ++r.permuted;
                const std::string p = check_permuted(f, v.schema, state, t, static_cast<uint32_t>(r.decisions));
                if (!p.empty()) add_fail(&r, label, at + ": " + p);
            }
            if (keep) r.tensors[viewer].push_back(t);
            if (seat) return true;
            if (!have_lists && !featurize::registered_lists(v.schema, v.decklists, v.metagame, &lists, e)) return false;
            have_lists = true;
            for (int s = 0; s < 2; ++s) {
                const SeatFacts &own = facts[s], &opp = facts[1 - s];
                if (!own.has_menu || !same_decision(own, opp)) continue;
                featurize::Targets tg;
                if (!featurize::decision_targets(f.spec, v.schema, own.lite, state, lists, &tg, e)) return false;
                ++r.targets;
                const std::string c = check_targets(tg, own, opp, f.spec.library_depth);
                if (!c.empty()) add_fail(&r, label, at + " seat " + std::to_string(s) + ": " + c);
                if (keep) r.targets_of[s].push_back(tg);
            }
            return true;
        },
        &err);
    if (!ok) add_fail(&r, label, err);
    return r;
}

static void report_dump(rmcheck::Report &rep, const std::string &label, const DumpResult &r) {
    for (const std::string &m : r.fails) rep.error("featurize", m);
    if (r.decisions == 0) rep.error("featurize", label + ": no decision featurized");
}

static void add_totals(DumpResult *total, const DumpResult &r) {
    total->decisions += r.decisions;
    total->seat += r.seat;
    total->omniscient += r.omniscient;
    total->permuted += r.permuted;
    total->targets += r.targets;
    total->tokens += r.tokens;
    total->relations += r.relations;
    total->featurize_s += r.featurize_s;
}

// The replay corpus, each game replayed with a dump and checked.
static void corpus_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f, DumpResult *total) {
    const std::vector<std::string> names = rmcheck::corpus_names(opt);
    std::vector<DumpResult> results(names.size());
    rmcheck::parallel_for(names.size(), opt.jobs, [&](size_t i) {
        const std::string dump = out_path(opt, "corpus_" + names[i] + ".rms");
        rmcheck::ProcSpec spec;
        spec.argv = {opt.binary, "--replay", rmcheck::corpus_log(opt, names[i]), "--machine", "--rmstate-dump", dump};
        spec.cwd = opt.bin_dir;
        spec.timeout_s = 900;
        const rmcheck::ProcResult pr = rmcheck::run_process(spec);
        if (!pr.ok()) {
            add_fail(&results[i], names[i], "replay " + pr.status_text());
        } else {
            results[i] = check_dump(f, dump, "corpus " + names[i], false);
        }
        rmcheck::remove_tree(dump);
    });
    DumpResult sum;
    for (size_t i = 0; i < names.size(); ++i) {
        report_dump(rep, "corpus " + names[i], results[i]);
        add_totals(&sum, results[i]);
    }
    add_totals(total, sum);
    rep.info("featurize: corpus " + std::to_string(names.size()) + " games, " + std::to_string(sum.decisions) +
             " decisions (" + std::to_string(sum.seat) + " seat, " + std::to_string(sum.omniscient) + " omniscient), " +
             std::to_string(sum.permuted) + " reordered, " + std::to_string(sum.targets) + " targets");
}

// The explore games under --selfcheck-featurize: clean self-check summaries, then their dumps checked.
// Returns the first game's dump (kept, its tensors in `kept`) for the CLI and Python parts.
static std::string games_part(
    const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f, DumpResult *total, DumpResult *kept) {
    const std::vector<rmcheck::FuzzGame> games = featurize_games();
    const std::string dir = opt.out("featurize");
    std::vector<rmcheck::FuzzOutcome> outs(games.size());
    std::vector<DumpResult> results(games.size());
    rmcheck::parallel_for(games.size(), opt.jobs, [&](size_t i) {
        outs[i] = rmcheck::play_fuzz_game(opt, games[i], dir);
        if (outs[i].result.ok() && rmcheck::file_exists(outs[i].dump))
            results[i] = check_dump(f, outs[i].dump, outs[i].label, i == 0);
    });
    for (size_t i = 0; i < games.size(); ++i) {
        const rmcheck::FuzzOutcome &o = outs[i];
        if (!o.result.ok())
            rep.error("featurize", o.label + ": engine " + o.result.status_text() + " (see " + o.err_path + ")");
        for (const std::string &m :
            rmcheck::selfcheck_failures(o.result.err, o.err_path, o.label, o.finished ? 1 : -1, {"STATE", "FEATURIZE"}))
            rep.error("featurize", m);
        report_dump(rep, o.label, results[i]);
        add_totals(total, results[i]);
        if (i > 0) rmcheck::remove_tree(o.dump);
    }
    rep.info("featurize: " + std::to_string(games.size()) +
             " explore games under --selfcheck-featurize (scramble, "
             "permutation and isolation of each seat's tensors in the engine) and their dumps checked");
    *kept = std::move(results[0]);
    return outs[0].dump;
}

static std::string file_bytes(const std::string &path) {
    std::string s;
    rmcheck::read_file(path, &s);
    return s;
}

// rmfeat over the kept dump (with the exported tables) writes the in-process tensors and targets.
static void cli_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::string &cards, const std::string &tables_dir, const std::string &dump, const DumpResult &kept) {
    const std::string text = rmcheck::path_join(opt.bin_dir, "resources/embeddings/oracle_text.rmtb");
    const char *viewers[] = {"A", "B", "omniscient"};
    size_t files = 0;
    for (int v = 0; v < 3; ++v) {
        const std::string out = out_path(opt, std::string("tensors_") + viewers[v] + ".rmtb");
        rmcheck::ProcSpec spec;
        spec.argv = {opt.rmfeat, "--featurize", dump, "--viewer", viewers[v], "--out", out, "--cards", cards,
            "--tables", tables_dir, "--text-table", text};
        const rmcheck::ProcResult r = rmcheck::run_process(spec);
        if (!r.ok() || file_bytes(out) != featurize::rmtb_encode(featurize::tensors_file(f.spec, kept.tensors[v])))
            rep.error("featurize", std::string("rmfeat --featurize --viewer ") + viewers[v] +
                                       " differs from the in-process tensors (" + r.status_text() + ")");
        ++files;
        if (v < 2) {
            const std::string tout = out_path(opt, std::string("targets_") + viewers[v] + ".rmtb");
            spec.argv = {opt.rmfeat, "--targets", dump, "--seat", viewers[v], "--out", tout, "--cards", cards,
                "--text-table", text};
            const rmcheck::ProcResult t = rmcheck::run_process(spec);
            if (!t.ok() ||
                file_bytes(tout) != featurize::rmtb_encode(featurize::targets_file(f.spec, kept.targets_of[v])))
                rep.error("featurize", std::string("rmfeat --targets --seat ") + viewers[v] +
                                           " differs from the in-process targets (" + t.status_text() + ")");
            ++files;
        }
    }
    rep.info("OK   rmfeat: " + std::to_string(files) + " files equal the in-process featurization of " +
             rmcheck::basename_of(dump));
}

// {"path":…,"kind":…,"arrays":{name:{"fnv":<FNV-1a 64 of the values' bytes>,"shape":[…]}}}.
static std::string digests_json(const std::string &path, const featurize::RmtbFile &file) {
    std::string out =
        "{\"path\":" + featurize::json_quote(path) + ",\"kind\":" + featurize::json_quote(file.kind) + ",\"arrays\":{";
    for (size_t i = 0; i < file.arrays.size(); ++i) {
        const featurize::RmtbArray &a = file.arrays[i];
        out += std::string(i ? "," : "") + featurize::json_quote(a.name) + ":{\"fnv\":\"" +
               featurize::hex16(rmstate::fnv1a64(a.data.data(), a.data.size())) + "\",\"shape\":[";
        for (size_t d = 0; d < a.shape.size(); ++d) out += (d ? "," : "") + std::to_string(a.shape[d]);
        out += "]}";
    }
    return out + "}}";
}

// python/pycheck.py --part featurize reads the exported tables and the kept dump's tensors and
// targets with python/rmtb.py: the values the C++ side built (their digests), shapes against the spec.
static void python_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::string &tables_dir, const DumpResult &kept) {
    if (opt.python.empty()) {
        rep.info("featurize: pycheck skipped (no python3 on PATH)");
        return;
    }
    const featurize::CardTableFiles tf = featurize::card_table_files(f.tables, f.spec.hash);
    const std::string tensors = out_path(opt, "tensors_A.rmtb"), targets = out_path(opt, "targets_A.rmtb");
    std::string json =
        "{\"spec\":" + featurize::json_quote(rmcheck::path_join(tables_dir, Featurization::SPEC_FILE)) + ",\"files\":[";
    json += digests_json(rmcheck::path_join(tables_dir, Featurization::CARD_FILE), tf.card) + ",";
    json += digests_json(rmcheck::path_join(tables_dir, Featurization::ABILITY_FILE), tf.ability) + ",";
    json += digests_json(rmcheck::path_join(tables_dir, Featurization::PART_FILE), tf.part) + ",";
    json += digests_json(tensors, featurize::tensors_file(f.spec, kept.tensors[0])) + ",";
    json += digests_json(targets, featurize::targets_file(f.spec, kept.targets_of[0])) + "]}\n";
    const std::string expected = out_path(opt, "pycheck_featurize.json");
    rmcheck::write_file(expected, json);
    rmcheck::ProcSpec spec;
    spec.argv = {opt.python, rmcheck::path_join(opt.repo, "python/pycheck.py"), "--part", "featurize",
        "--featurize-expected", expected};
    spec.cwd = opt.repo;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    for (const std::string &line : rmcheck::split_lines(r.out)) {
        if (rmcheck::starts_with(line, "FAIL: ")) rep.error("featurize", "pycheck: " + line.substr(6));
        if (rmcheck::starts_with(line, "  OK   ")) rep.info("OK   pycheck: " + line.substr(7));
    }
    if (!r.ok()) rep.error("featurize", "python/pycheck.py --part featurize " + r.status_text() + ": " + r.err);
}

namespace rmcheck {

// The featurize tier (docs/featurizer.md §7): the card tables and the spec from the whole-vocab
// cards file, exported and read back (C++, rmfeat and Python alike); the corpus's and four explore
// games' dumps featurized with the checks of check_dump; the engine's --selfcheck-featurize on the
// explore games; rmfeat's tensors and targets against the in-process ones.
void tier_featurize(const Options &opt, Report &rep) {
    const double t0 = now_seconds();
    Featurization f;
    std::string cards;
    if (!load_inputs(opt, rep, &f, &cards)) return;
    const std::string tables_dir = out_path(opt, "tables");
    tables_part(opt, rep, f, cards, tables_dir);
    const double t1 = now_seconds();
    DumpResult total, kept;
    corpus_part(opt, rep, f, &total);
    const double t2 = now_seconds();
    const std::string dump = games_part(opt, rep, f, &total, &kept);
    const double t3 = now_seconds();
    if (file_exists(dump)) {
        cli_part(opt, rep, f, cards, tables_dir, dump, kept);
        python_part(opt, rep, f, tables_dir, kept);
        remove_tree(dump);
    }
    const double d = static_cast<double>(std::max<size_t>(total.decisions, 1));
    rep.info("featurize: " + std::to_string(total.decisions) + " decisions, " +
             fmt1(static_cast<double>(total.tokens) / d) + " tokens and " +
             fmt1(static_cast<double>(total.relations) / d) + " relations per decision, " +
             std::to_string(static_cast<long>(d / std::max(total.featurize_s, 1e-9))) +
             " decisions/s featurized (this build, one thread)");
    rep.info("featurize: tables " + fmt1(t1 - t0) + "s, corpus " + fmt1(t2 - t1) + "s, games " + fmt1(t3 - t2) +
             "s, rmfeat + pycheck " + fmt1(now_seconds() - t3) + "s");
}

}  // namespace rmcheck
