#include <algorithm>
#include <cstring>
#include <map>
#include <set>

#include "corpus.h"
#include "frames.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "scenarios.h"
#include "sessions.h"
#include "util.h"

using rmcheck::FuzzGame;
using rmcheck::StreamGame;

// One table block of a frame payload, by its framing alone (Appendix A.2).
struct Block {
        uint16_t id;
        uint32_t width, rows;
        std::string body;
};

static std::vector<std::string> selfcheck_flags(bool every_decision);
static std::vector<FuzzGame> fuzz_games();
static std::vector<StreamGame> stream_games();
static void corpus_part(const rmcheck::Options &opt, rmcheck::Report &rep);
static std::vector<std::string> fuzz_part(const rmcheck::Options &opt, rmcheck::Report &rep);
static void scenarios_part(const rmcheck::Options &opt, rmcheck::Report &rep);
static void decode_part(const rmcheck::Options &opt, rmcheck::Report &rep, const std::vector<std::string> &dumps);
static uint64_t read_le(const std::string &s, size_t at, int n);
static void put_le(std::string *s, size_t at, uint64_t v, int n);
static bool split_blocks(const std::string &payload, std::string *header, std::vector<Block> *blocks);
static std::string rebuild(const std::string &header, const std::vector<Block> &blocks);
static void check_refusals(const rmcheck::Options &opt, rmcheck::Report &rep, const std::string &dump);
static void check_name_card(const rmcheck::Options &opt, rmcheck::Report &rep);
static void isolation_part(const rmcheck::Options &opt, rmcheck::Report &rep);
static std::string normpath(const std::string &path);
static void check_card_params(const rmcheck::Options &opt, rmcheck::Report &rep);
static int64_t card_id_named(const rmcheck::Rows &card, const std::string &name);
static long first_ability_row(const rmcheck::Rows &ab, int64_t card);
static std::vector<size_t> ability_part_rows(const rmcheck::Rows &part, int64_t card, uint64_t ability);
static uint64_t enum_mask(const rmstate::Schema &schema, const char *enum_name, const std::vector<std::string> &names);
static void expect_cell(const rmcheck::Rows &t, size_t row, const char *col, const std::string &want,
    const std::string &what, std::vector<std::string> *fails);
static void check_part_structure(const rmcheck::Rows &ab, const rmcheck::Rows &part, std::vector<std::string> *fails);

static std::vector<std::string> selfcheck_flags(bool every_decision) {
    if (every_decision) return {"--selfcheck-state", "--selfcheck-state-sample", "1"};
    return {"--selfcheck-state"};
}

// The state tier's fuzz games.
static std::vector<FuzzGame> fuzz_games() {
    struct Row {
            const char *a, *b;
            bool bo3;
            long seed;
            bool every, dump;
    };
    static const Row rows[] = {
        {"reanimator", "wrb_energy", false, 101, false, true},
        {"wrb_energy", "bw_dnt", true, 102, false, true},
        {"bw_dnt", "wubg_doomsday", false, 103, false, false},
        {"reanimator", "uw_control", true, 104, false, false},
        {"wubg_doomsday", "wrb_energy", false, 105, false, false},
        {"burn", "reanimator", false, 106, false, false},
        {"ur_delver", "gw_maverick", false, 107, true, false},
        {"lands", "bw_dnt", false, 108, true, false},
        {"bug", "reanimator", true, 109, false, false},
        {"wrb_energy", "lands", true, 110, false, false},
        {"uw_control", "bw_dnt", false, 111, false, false},
        {"gw_maverick", "wrb_energy", false, 112, false, false},
        {"reanimator", "reanimator", false, 113, true, false},
        {"wubg_doomsday", "burn", true, 114, false, false},
        {"bw_dnt", "ur_delver", true, 115, false, false},
        {"wrb_energy", "wrb_energy", false, 116, true, false},
    };
    std::vector<FuzzGame> out;
    for (const Row &r : rows) {
        FuzzGame g;
        g.deck_a = std::string("league/") + r.a;
        g.deck_b = std::string("league/") + r.b;
        g.bo3 = r.bo3;
        g.seed = r.seed;
        g.flags = selfcheck_flags(r.every);
        g.dump = r.dump;
        // The first dump also renders its live JSON, checked against --rmstate-read's.
        if (r.seed == 101) {
            g.flags.push_back("--state-dump");
            g.flags.push_back("json");
        }
        out.push_back(g);
    }
    return out;
}

// The state tier's RMSTATE sessions.
static std::vector<StreamGame> stream_games() {
    StreamGame a;
    a.deck_a = "league/ur_delver";
    a.deck_b = "league/bug";
    a.seed = 21;
    a.flags = {"--selfcheck-state", "--narrative"};
    StreamGame b;
    b.deck_a = "league/wrb_energy";
    b.deck_b = "league/bw_dnt";
    b.bo3 = true;
    b.seed = 23;
    b.flags = {"--selfcheck-state", "--closed-decklist"};
    return {a, b};
}

// The corpus replayed with --selfcheck-knowledge: the same output as its baseline and a clean
// SELFCHECK-STATE and SELFCHECK-KNOWLEDGE summary per game.
static void corpus_part(const rmcheck::Options &opt, rmcheck::Report &rep) {
    const std::vector<std::string> names = rmcheck::corpus_names(opt);
    if (names.empty()) {
        rep.error("state", "corpus: no decision log in " + opt.corpus_dir());
        return;
    }
    const std::string dir = opt.out("state");
    std::vector<int> ok(names.size(), 0);
    rmcheck::parallel_for(names.size(), opt.jobs, [&](size_t i) {
        const std::string &name = names[i];
        const rmcheck::ProcResult r = rmcheck::replay_corpus_game(opt, name, {"--selfcheck-knowledge"});
        const std::string err_path = rmcheck::path_join(dir, "corpus_" + name + ".err");
        rmcheck::write_file(err_path, r.err);
        std::vector<std::string> fails;
        if (!r.ok()) fails.push_back("corpus " + name + ": the replay " + r.status_text());
        std::string expected;
        rmcheck::read_file(rmcheck::corpus_baseline(opt, name), &expected);
        const std::string diff =
            rmcheck::first_difference(expected, rmcheck::baseline_text(r.out, rmcheck::corpus_log(opt, name), name));
        if (!diff.empty())
            fails.push_back("corpus " + name + ": the replay drifts with --selfcheck-knowledge on: " + diff);
        for (const std::string &f :
            rmcheck::selfcheck_failures(r.err, err_path, "corpus " + name, 1, {"STATE", "KNOWLEDGE"}))
            fails.push_back(f);
        for (const std::string &f : fails) rep.error("state", f);
        ok[i] = fails.empty();
    });
    size_t n = 0;
    for (int o : ok) n += static_cast<size_t>(o);
    rep.info("corpus: " + std::to_string(n) + "/" + std::to_string(names.size()) +
             " games replay byte-identical with the self-checks clean (state and knowledge)");
}

// The fuzz games; returns their dumps.
static std::vector<std::string> fuzz_part(const rmcheck::Options &opt, rmcheck::Report &rep) {
    const std::vector<FuzzGame> games = fuzz_games();
    const std::string dir = opt.out("state");
    std::vector<rmcheck::FuzzOutcome> outs(games.size());
    rmcheck::parallel_for(
        games.size(), opt.jobs, [&](size_t i) { outs[i] = rmcheck::play_fuzz_game(opt, games[i], dir); });
    std::vector<std::string> dumps;
    for (const rmcheck::FuzzOutcome &o : outs) {
        std::vector<std::string> fails;
        if (!o.result.ok())
            fails.push_back(o.label + ": engine " + o.result.status_text() + " (see " + o.err_path + ")");
        for (const std::string &f :
            rmcheck::selfcheck_failures(o.result.err, o.err_path, o.label, o.finished ? 1 : -1, {"STATE"}))
            fails.push_back(f);
        if (!o.dump.empty()) {
            if (rmcheck::file_exists(o.dump))
                dumps.push_back(o.dump);
            else
                fails.push_back(o.label + ": no --rmstate-dump file was written");
        }
        for (const std::string &f : fails) rep.error("state", f);
        if (fails.empty()) rep.info("OK   " + o.label + (o.finished ? "" : " (stopped at the decision cap)"));
    }
    return dumps;
}

// Every scenario with the self-checks at every decision (a scenario stops at its decision cap, so
// it prints no summary; its ERROR: lines fail it).
static void scenarios_part(const rmcheck::Options &opt, rmcheck::Report &rep) {
    const std::vector<std::string> paths = rmcheck::scenario_paths(opt);
    rmcheck::prevalidate_scenarios(paths);
    std::vector<rmcheck::ScenarioRun> runs(paths.size());
    rmcheck::parallel_for(paths.size(), opt.jobs,
        [&](size_t i) { runs[i] = rmcheck::run_scenario(opt, paths[i], selfcheck_flags(true)); });
    size_t ok = 0;
    for (const rmcheck::ScenarioRun &r : runs) {
        if (r.fails.empty()) {
            ++ok;
            continue;
        }
        rep.error("state", "scenario " + r.name + " with --selfcheck-state --selfcheck-state-sample 1: " +
                               rmcheck::join(r.fails, "; ") + " (reproduce: " + r.reproduce + ")");
    }
    rep.info("scenarios: " + std::to_string(ok) + "/" + std::to_string(runs.size()) +
             " pass with --selfcheck-state --selfcheck-state-sample 1");
}

static uint64_t read_le(const std::string &s, size_t at, int n) {
    uint64_t v = 0;
    for (int i = n - 1; i >= 0; --i) v = (v << 8) | static_cast<unsigned char>(s[at + static_cast<size_t>(i)]);
    return v;
}

static void put_le(std::string *s, size_t at, uint64_t v, int n) {
    for (int i = 0; i < n; ++i) (*s)[at + static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xFF);
}

static bool split_blocks(const std::string &payload, std::string *header, std::vector<Block> *blocks) {
    if (payload.size() < 24) return false;
    *header = payload.substr(0, 24);
    const uint64_t n = read_le(payload, 16, 4);
    size_t off = 24;
    for (uint64_t k = 0; k < n; ++k) {
        if (off + 16 > payload.size()) return false;
        Block b;
        b.id = static_cast<uint16_t>(read_le(payload, off, 2));
        b.width = static_cast<uint32_t>(read_le(payload, off + 4, 4));
        b.rows = static_cast<uint32_t>(read_le(payload, off + 8, 4));
        const size_t len = static_cast<size_t>(b.width) * b.rows;
        if (off + 16 + len > payload.size()) return false;
        b.body = payload.substr(off + 16, len);
        off = (off + 16 + len + 7) / 8 * 8;
        blocks->push_back(b);
    }
    return true;
}

// A frame payload from a 24-byte header (its table count replaced) and blocks.
static std::string rebuild(const std::string &header, const std::vector<Block> &blocks) {
    std::string out = header;
    put_le(&out, 16, blocks.size(), 4);
    for (const Block &b : blocks) {
        std::string h(16, '\0');
        put_le(&h, 0, b.id, 2);
        put_le(&h, 4, b.width, 4);
        put_le(&h, 8, b.rows, 4);
        out += h + b.body;
        out.append((8 - out.size() % 8) % 8, '\0');
    }
    return out;
}

// Each mutation of a dump's first STATE frame is refused by the engine's decoder (--rmstate-read
// exit 1 naming it), an unknown table skipped and listed (the engine's half; python/pycheck.py
// checks the Python decoder's).
static void check_refusals(const rmcheck::Options &opt, rmcheck::Report &rep, const std::string &dump) {
    std::vector<std::string> payloads, head;
    std::string err, state;
    if (!rmcheck::read_records(dump, &payloads, &err)) {
        rep.error("state", "refusals: " + err);
        return;
    }
    for (const std::string &p : payloads) {
        if (rmcheck::frame_kind(p) == rmcheck::KIND_STATE) {
            state = p;
            break;
        }
        head.push_back(p);
    }
    std::string header;
    std::vector<Block> blocks;
    if (state.empty() || !split_blocks(state, &header, &blocks)) {
        rep.error("state", "refusals: " + rmcheck::basename_of(dump) + ": no STATE frame to mutate");
        return;
    }
    struct Mutant {
            std::string label, payload, needle;
            bool refused;
    };
    std::vector<Mutant> mutants;
    std::string h = header;
    put_le(&h, 4, read_le(header, 4, 2) + 1, 2);
    mutants.push_back({"version", rebuild(h, blocks), "format version", true});
    h = header;
    h[8] = static_cast<char>(h[8] ^ 1);
    mutants.push_back({"hash", rebuild(h, blocks), "schema hash differs", true});
    h = header;
    h[6] = static_cast<char>(200);
    mutants.push_back({"kind", rebuild(h, blocks), "unknown frame kind", true});
    std::vector<Block> wider = blocks;
    for (Block &b : wider) {
        if (b.id == 1) continue;
        std::string wide;
        for (uint32_t r = 0; r < b.rows; ++r) wide += b.body.substr(static_cast<size_t>(r) * b.width, b.width) + '\0';
        b.body = wide;
        ++b.width;
        break;
    }
    mutants.push_back({"row width", rebuild(header, wider), "has row width", true});
    std::vector<Block> extra = blocks;
    extra.push_back({0xFFFE, 8, 1, std::string(8, '\1')});
    mutants.push_back({"unknown table", rebuild(header, extra), "blocks the decoding schema does not know", false});
    const std::string dir = rmcheck::make_temp_dir(opt.out("state"), "mutants_");
    bool all_ok = true;
    for (const Mutant &m : mutants) {
        std::vector<std::string> recs = head;
        recs.push_back(m.payload);
        const std::string path = rmcheck::path_join(dir, "mutant.rms");
        rmcheck::write_records(path, recs);
        rmcheck::ProcSpec spec;
        spec.argv = {opt.binary, "--rmstate-read", path, "--state-dump", "json"};
        spec.cwd = opt.bin_dir;
        const rmcheck::ProcResult r = rmcheck::run_process(spec);
        if (r.exit_code != 1 || !rmcheck::contains(r.err, m.needle.c_str())) {
            all_ok = false;
            rep.error("state", "mutant " + m.label + ": the engine should report " + rmcheck::quoted(m.needle) + " (" +
                                   r.status_text() + "): " + r.err.substr(r.err.size() > 300 ? r.err.size() - 300 : 0));
        }
        if (!m.refused && !rmcheck::contains(r.out, "\"skipped\":[65534]")) {
            all_ok = false;
            rep.error("state", "mutant " + m.label + ": the engine's JSON does not list the block as skipped");
        }
    }
    rmcheck::remove_tree(dir);
    if (all_ok) rep.info("OK   refusals (" + std::to_string(mutants.size()) + " mutations)");
}

// Disruptor Flute's name-a-card menu with open and closed decklists.
static void check_name_card(const rmcheck::Options &opt, rmcheck::Report &rep) {
    for (bool closed : {false, true}) {
        const std::string label = std::string("name-a-card ") + (closed ? "closed" : "open") + " decklist";
        StreamGame g;
        g.deck_a = "league/bw_dnt";
        g.deck_b = "league/burn";
        g.seed = 31;
        g.flags = {"--battlefield-a", "Disruptor Flute", "--selfcheck-state", "--selfcheck-state-sample", "1"};
        if (closed) g.flags.push_back("--closed-decklist");
        const rmcheck::StreamRun run = rmcheck::play_stream_session(opt, g, "", 20);
        if (!run.error.empty()) {
            rep.error("state", label + ": " + run.error);
            continue;
        }
        rmcheck::FrameDecoder dec;
        std::string err, cards, decklists;
        rmstate::DecodedFrame state_frame;
        bool found = false;
        std::vector<size_t> rows;
        for (const std::string &p : run.payloads) {
            rmstate::DecodedFrame f;
            if (!dec.feed(p, &f, &err)) break;
            if (rmcheck::frame_viewer(p) != rmcheck::VIEWER_A) continue;
            const int kind = rmcheck::frame_kind(p);
            if (kind == rmcheck::KIND_CARDS) cards = p;
            if (kind == rmcheck::KIND_DECKLISTS) decklists = p;
            if (kind != rmcheck::KIND_STATE) continue;
            const rmcheck::Rows menu(f, *dec.schema(rmcheck::VIEWER_A), "menu");
            for (size_t r = 0; r < menu.size(); ++r)
                if (menu.name(r, menu.col("category")) == "NAME_CARD") rows.push_back(r);
            if (!rows.empty()) {
                state_frame = std::move(f);
                found = true;
                break;
            }
        }
        if (!found) {
            rep.error("state", label + ": Player A was never offered a name-a-card menu");
            continue;
        }
        const rmstate::Schema &schema = *dec.schema(rmcheck::VIEWER_A);
        rmstate::DecodedFrame cf, df;
        rmstate::decode_frame(cards, schema, &cf);
        rmstate::decode_frame(decklists, schema, &df);
        const rmcheck::Rows menu(state_frame, schema, "menu"), card(cf, schema, "card"),
            dl(df, schema, "decklist_card");
        std::vector<uint64_t> named;
        std::vector<std::string> descs;
        bool has_source = false;
        for (size_t r : rows) {
            named.push_back(menu.u(r, menu.col("source_card")));
            descs.push_back(menu.text(r, menu.col("description")));
            has_source = has_source || menu.u(r, menu.col("source")) != 0;
        }
        std::map<std::string, std::set<uint64_t>> listed;
        for (size_t r = 0; r < dl.size(); ++r) listed[dl.name(r, dl.col("player"))].insert(dl.u(r, dl.col("card")));
        std::vector<uint64_t> fronts;
        for (size_t r = 0; r < card.size(); ++r)
            if (card.name(r, card.col("face")) == "FRONT" && !card.u(r, card.col("token")))
                fronts.push_back(card.u(r, card.col("card")));
        std::sort(fronts.begin(), fronts.end());
        std::vector<std::string> fails;
        if (closed) {
            std::vector<uint64_t> sorted_named = named;
            std::sort(sorted_named.begin(), sorted_named.end());
            if (sorted_named != fronts)
                fails.push_back(label + ": " + std::to_string(named.size()) + " candidates, CARDS lists " +
                                std::to_string(fronts.size()) + " vocab cards");
            if (has_source) fails.push_back(label + ": a candidate has an object handle");
            if (!std::is_sorted(descs.begin(), descs.end()))
                fails.push_back(label + ": the candidates are not in name order");
            if (listed.size() != 1 || !listed.count("PLAYER_A"))
                fails.push_back(label + ": DECKLISTS lists other than only A's own list");
        } else {
            const std::set<uint64_t> &opp = listed["PLAYER_B"];
            const std::set<uint64_t> unique(named.begin(), named.end());
            bool inside = unique.size() == named.size();
            for (uint64_t c : named) inside = inside && opp.count(c);
            if (!inside) fails.push_back(label + ": candidates outside the opponent's registered list");
        }
        for (const std::string &l : rmcheck::split_lines(run.proc.err))
            if (rmcheck::starts_with(l, "ERROR:") || rmcheck::starts_with(l, "FATAL:")) {
                fails.push_back(label + ": " + rmcheck::trim(l));
                break;
            }
        for (const std::string &f : fails) rep.error("state", f);
        if (fails.empty()) rep.info("OK   " + label + ": " + std::to_string(rows.size()) + " candidates");
    }
}

// The dumps' re-encoding, the decoders' refusals, the RMSTATE sessions (each against its dump, with
// and without one) and the name-a-card menus.
static void decode_part(const rmcheck::Options &opt, rmcheck::Report &rep, const std::vector<std::string> &dumps) {
    if (dumps.empty()) rep.error("state", "decode: no --rmstate-dump file to check");
    for (const std::string &d : dumps) {
        std::string sum;
        const std::vector<std::string> fails = rmcheck::recode_dump(opt, d, &sum);
        for (const std::string &f : fails) rep.error("state", f);
        if (fails.empty()) rep.info("OK   " + rmcheck::basename_of(d) + ": " + sum);
    }
    if (!dumps.empty()) check_refusals(opt, rep, dumps[0]);
    const std::string dir = opt.out("state");
    const std::vector<StreamGame> games = stream_games();
    std::vector<std::vector<std::string>> fails(games.size());
    // Session 2i is game i with a dump, 2i + 1 without.
    std::vector<rmcheck::StreamRun> runs(2 * games.size());
    rmcheck::parallel_for(runs.size(), opt.jobs, [&](size_t k) {
        const StreamGame &g = games[k / 2];
        const std::string dump = rmcheck::path_join(dir, "stream_" + std::to_string(g.seed) + ".rms");
        runs[k] = rmcheck::play_stream_session(opt, g, k % 2 ? "" : dump);
    });
    rmcheck::parallel_for(games.size(), opt.jobs, [&](size_t i) {
        const StreamGame &g = games[i];
        const std::string label = rmcheck::stream_label("stream", g);
        const std::string dump = rmcheck::path_join(dir, "stream_" + std::to_string(g.seed) + ".rms");
        const rmcheck::StreamRun &with = runs[2 * i], &alone = runs[2 * i + 1];
        const std::string err_path = rmcheck::path_join(dir, "stream_" + std::to_string(g.seed) + ".err");
        rmcheck::write_file(err_path, with.proc.err);
        if (!with.error.empty() || !alone.error.empty()) {
            fails[i].push_back(label + ": " + with.error + alone.error + " (see " + err_path + ")");
            return;
        }
        std::vector<std::string> dump_payloads;
        std::string err;
        if (!rmcheck::read_records(dump, &dump_payloads, &err)) fails[i].push_back(label + ": " + err);
        for (const std::string &f : rmcheck::check_stream(with.payloads, dump_payloads, label)) fails[i].push_back(f);
        bool narrative = false;
        for (const std::string &f : g.flags) narrative = narrative || f == "--narrative";
        if (!rmcheck::contains(with.proc.err, "GAME_RESULT:") ||
            (narrative && !rmcheck::contains(with.proc.err, "-------- TURN")))
            fails[i].push_back(label + ": the results / narrative did not reach stderr");
        if (with.payloads != alone.payloads)
            fails[i].push_back(label + ": the stream differs with and without --rmstate-dump");
        std::string sum;
        for (const std::string &f : rmcheck::recode_dump(opt, dump, &sum)) fails[i].push_back(f);
        for (const std::string &f : rmcheck::selfcheck_failures(with.proc.err, err_path, label, 1, {"STATE"}))
            fails[i].push_back(f);
        if (fails[i].empty()) rep.info("OK   " + label + ": " + std::to_string(with.payloads.size()) + " records");
        rmcheck::remove_tree(dump);
    });
    for (const auto &fs : fails)
        for (const std::string &f : fs) rep.error("state", f);
    check_name_card(opt, rep);
}

// path with "." and ".." components resolved (Python's os.path.normpath on a relative path).
static std::string normpath(const std::string &path) {
    std::vector<std::string> parts;
    for (const std::string &p : rmcheck::split(path, '/')) {
        if (p.empty() || p == ".") continue;
        if (p == ".." && !parts.empty() && parts.back() != "..")
            parts.pop_back();
        else
            parts.push_back(p);
    }
    return rmcheck::join(parts, "/");
}

// The id of the card whose front row is named `name` in a CARDS frame's card table, or -1.
static int64_t card_id_named(const rmcheck::Rows &card, const std::string &name) {
    for (size_t r = 0; r < card.size(); ++r)
        if (card.name(r, card.col("face")) == "FRONT" && card.text(r, card.col("name")) == name)
            return static_cast<int64_t>(card.u(r, card.col("card")));
    return -1;
}

// The card_ability row of `card`'s ability 0, or -1.
static long first_ability_row(const rmcheck::Rows &ab, int64_t card) {
    for (size_t r = 0; r < ab.size(); ++r)
        if (static_cast<int64_t>(ab.u(r, ab.col("card"))) == card && ab.u(r, ab.col("index")) == 0)
            return static_cast<long>(r);
    return -1;
}

// The card_ability_part rows of `card`'s ability `ability`, in frame order.
static std::vector<size_t> ability_part_rows(const rmcheck::Rows &part, int64_t card, uint64_t ability) {
    std::vector<size_t> out;
    for (size_t r = 0; r < part.size(); ++r)
        if (static_cast<int64_t>(part.u(r, part.col("card"))) == card && part.u(r, part.col("ability")) == ability)
            out.push_back(r);
    return out;
}

// The mask over enum `enum_name` with the bits of `names` set (0 for a name it lacks).
static uint64_t enum_mask(const rmstate::Schema &schema, const char *enum_name, const std::vector<std::string> &names) {
    const rmstate::EnumTable *e = schema.find_enum(enum_name);
    uint64_t m = 0;
    if (!e) return 0;
    for (const std::string &n : names)
        for (size_t i = 0; i < e->values.size(); ++i)
            if (e->values[i] == n) m |= uint64_t{1} << i;
    return m;
}

// Fails unless cell `col` of `row` reads `want` (an enum column by its value name, a null cell as
// "null", anything else as its number).
static void expect_cell(const rmcheck::Rows &t, size_t row, const char *col, const std::string &want,
    const std::string &what, std::vector<std::string> *fails) {
    const int c = t.col(col);
    if (c < 0) {
        fails->push_back(what + ": no column " + col);
        return;
    }
    const std::string got = t.null(row, c) ? "null" : t.name(row, c);
    if (got != want) fails->push_back(what + ": " + col + " is " + got + ", expected " + want);
}

// Every card_ability row's n_parts equals its card_ability_part rows, which number 0, 1, … in
// order, each with a parent before it (or none at depth 1); charm_num is nonzero exactly for a
// Charm.
static void check_part_structure(const rmcheck::Rows &ab, const rmcheck::Rows &part, std::vector<std::string> *fails) {
    std::map<std::pair<uint64_t, uint64_t>, std::vector<size_t>> by_ability;
    for (size_t r = 0; r < part.size(); ++r)
        by_ability[{part.u(r, part.col("card")), part.u(r, part.col("ability"))}].push_back(r);
    for (size_t r = 0; r < ab.size(); ++r) {
        const uint64_t card = ab.u(r, ab.col("card")), index = ab.u(r, ab.col("index"));
        const std::string what = "card " + std::to_string(card) + " ability " + std::to_string(index);
        const auto it = by_ability.find({card, index});
        const std::vector<size_t> rows = it == by_ability.end() ? std::vector<size_t>() : it->second;
        if (ab.u(r, ab.col("n_parts")) != rows.size()) fails->push_back(what + ": n_parts disagrees with its parts");
        const bool charm = ab.name(r, ab.col("category")) == "Charm";
        if (charm != (ab.u(r, ab.col("charm_num")) != 0)) fails->push_back(what + ": charm_num on a non-Charm");
        for (size_t k = 0; k < rows.size(); ++k) {
            const size_t p = rows[k];
            const bool top = part.null(p, part.col("parent"));
            if (part.u(p, part.col("part")) != k || (!top && part.u(p, part.col("parent")) >= k) ||
                top != (part.u(p, part.col("depth")) == 1))
                fails->push_back(what + ": part " + std::to_string(k) + " is out of order");
        }
    }
}

// The CARDS frame's ability parameters (register 41) over every vocab card (--closed-decklist):
// Lightning Bolt's damage and targets, Brainstorm's SubAbility$ part, Kozilek's Command's modes,
// and every ability's parts in order.
static void check_card_params(const rmcheck::Options &opt, rmcheck::Report &rep) {
    const std::string label = "card ability parameters";
    StreamGame g;
    g.deck_a = "league/burn";
    g.deck_b = "league/burn";
    g.seed = 1;
    g.flags = {"--closed-decklist"};
    const rmcheck::StreamRun run = rmcheck::play_stream_session(opt, g, "", 1);
    if (!run.error.empty()) {
        rep.error("state", label + ": " + run.error);
        return;
    }
    rmcheck::FrameDecoder dec;
    rmstate::DecodedFrame cards;
    bool found = false;
    std::string err;
    for (const std::string &p : run.payloads) {
        rmstate::DecodedFrame f;
        if (!dec.feed(p, &f, &err)) {
            rep.error("state", label + ": " + err);
            return;
        }
        if (rmcheck::frame_viewer(p) != rmcheck::VIEWER_A || rmcheck::frame_kind(p) != rmcheck::KIND_CARDS) continue;
        cards = std::move(f);
        found = true;
        break;
    }
    if (!found) {
        rep.error("state", label + ": Player A got no CARDS frame");
        return;
    }
    const rmstate::Schema &schema = *dec.schema(rmcheck::VIEWER_A);
    const rmcheck::Rows card(cards, schema, "card"), ab(cards, schema, "card_ability"),
        part(cards, schema, "card_ability_part");
    std::vector<std::string> fails;
    const int64_t bolt = card_id_named(card, "Lightning Bolt"), brainstorm = card_id_named(card, "Brainstorm"),
                  command = card_id_named(card, "Kozilek's Command");
    const long bolt_row = first_ability_row(ab, bolt), brainstorm_row = first_ability_row(ab, brainstorm),
               command_row = first_ability_row(ab, command);
    if (bolt_row < 0 || brainstorm_row < 0 || command_row < 0) {
        rep.error("state", label + ": CARDS lacks Lightning Bolt, Brainstorm or Kozilek's Command");
        return;
    }
    const size_t b = static_cast<size_t>(bolt_row);
    expect_cell(ab, b, "category", "DealDamage", "Lightning Bolt", &fails);
    expect_cell(ab, b, "amount", "3", "Lightning Bolt", &fails);
    expect_cell(ab, b, "amount_kind", "FIXED", "Lightning Bolt", &fails);
    expect_cell(ab, b, "target_min", "1", "Lightning Bolt", &fails);
    expect_cell(ab, b, "target_max", "1", "Lightning Bolt", &fails);
    expect_cell(ab, b, "n_parts", "0", "Lightning Bolt", &fails);
    expect_cell(ab, b, "charm_num", "0", "Lightning Bolt", &fails);
    const uint64_t any_target = enum_mask(schema, "target_class", {"CREATURE", "PLANESWALKER", "BATTLE", "PLAYER"});
    if (ab.u(b, ab.col("targets")) != any_target) fails.push_back("Lightning Bolt: targets is not any target");
    const std::vector<size_t> bs = ability_part_rows(part, brainstorm, 0);
    if (bs.size() != 1) {
        fails.push_back("Brainstorm: " + std::to_string(bs.size()) + " parts, expected 1");
    } else {
        expect_cell(part, bs[0], "kind", "SUB", "Brainstorm part 0", &fails);
        expect_cell(part, bs[0], "category", "ChangeZone", "Brainstorm part 0", &fails);
        expect_cell(part, bs[0], "amount", "2", "Brainstorm part 0", &fails);
        expect_cell(part, bs[0], "parent", "null", "Brainstorm part 0", &fails);
    }
    const size_t c = static_cast<size_t>(command_row);
    expect_cell(ab, c, "category", "Charm", "Kozilek's Command", &fails);
    expect_cell(ab, c, "charm_num", "2", "Kozilek's Command", &fails);
    expect_cell(ab, c, "n_parts", "5", "Kozilek's Command", &fails);
    const std::vector<size_t> cs = ability_part_rows(part, command, 0);
    const char *kinds[] = {"MODE", "MODE", "SUB", "MODE", "MODE"};
    for (size_t k = 0; k < cs.size() && k < 5; ++k)
        expect_cell(part, cs[k], "kind", kinds[k], "Kozilek's Command part " + std::to_string(k), &fails);
    if (cs.size() == 5) {
        expect_cell(part, cs[2], "parent", "1", "Kozilek's Command part 2", &fails);
        expect_cell(part, cs[2], "depth", "2", "Kozilek's Command part 2", &fails);
        expect_cell(part, cs[0], "amount_kind", "VARIABLE", "Kozilek's Command part 0", &fails);
        expect_cell(part, cs[4], "target_min", "0", "Kozilek's Command part 4", &fails);
        expect_cell(part, cs[4], "target_max", "255", "Kozilek's Command part 4", &fails);
        if (part.u(cs[4], part.col("targets")) != enum_mask(schema, "target_class", {"CARD_GRAVEYARD"}))
            fails.push_back("Kozilek's Command part 4: targets is not a card in a graveyard");
    }
    check_part_structure(ab, part, &fails);
    for (const std::string &f : fails) rep.error("state", label + ": " + f);
    if (fails.empty())
        rep.info("OK   " + label + ": " + std::to_string(ab.size()) + " abilities, " + std::to_string(part.size()) +
                 " parts");
}

// The knowledge-side sources include only the standard library and the std-only headers.
static void isolation_part(const rmcheck::Options &opt, rmcheck::Report &rep) {
    static const std::set<std::string> std_headers = {"algorithm", "any", "array", "atomic", "bitset", "cassert",
        "cctype", "cerrno", "cfloat", "charconv", "chrono", "cinttypes", "climits", "cmath", "complex", "cstdarg",
        "cstddef", "cstdint", "cstdio", "cstdlib", "cstring", "ctime", "deque", "functional", "initializer_list",
        "iomanip", "ios", "iosfwd", "iostream", "istream", "iterator", "limits", "list", "map", "memory", "mutex",
        "new", "numeric", "optional", "ostream", "queue", "random", "ratio", "set", "sstream", "stack", "stdexcept",
        "string", "string_view", "system_error", "tuple", "type_traits", "typeinfo", "unordered_map", "unordered_set",
        "utility", "variant", "vector"};
    static const std::set<std::string> allowed = {
        "src/state_format/decode.h", "src/state_format/schema.h", "src/state_format/json.h"};
    std::vector<std::string> files;
    for (const char *stem : {"decode", "schema", "json", "recode"})
        for (const char *ext : {".h", ".cpp"}) {
            const std::string rel = std::string("src/state_format/") + stem + ext;
            if (rmcheck::file_exists(opt.repo + "/" + rel)) files.push_back(rel);
        }
    for (const char *ext : {".h", ".cpp"})
        for (const std::string &f : rmcheck::list_dir(opt.repo + "/src/knowledge", ext))
            files.push_back("src/knowledge/" + f);
    std::sort(files.begin(), files.end());
    size_t bad = 0;
    for (const std::string &rel : files) {
        std::string text;
        rmcheck::read_file(opt.repo + "/" + rel, &text);
        const std::string own = rmcheck::strip_ext(rel) + ".h";
        const bool in_knowledge = rmcheck::starts_with(rel, "src/knowledge/");
        const std::vector<std::string> lines = rmcheck::split_lines(text);
        for (size_t n = 0; n < lines.size(); ++n) {
            // ^\s*#\s*include\s*([<"])([^>"]+)[>"]
            const std::string t = rmcheck::trim(lines[n]);
            if (t.empty() || t[0] != '#') continue;
            const std::string rest = rmcheck::trim(t.substr(1));
            if (!rmcheck::starts_with(rest, "include")) continue;
            const std::string spec = rmcheck::trim(rest.substr(7));
            if (spec.empty() || (spec[0] != '<' && spec[0] != '"')) continue;
            const size_t close = spec.find_first_of(">\"", 1);
            if (close == std::string::npos || close == 1) continue;
            const std::string name = spec.substr(1, close - 1);
            const std::string where = rel + ":" + std::to_string(n + 1);
            if (spec[0] == '<') {
                if (!std_headers.count(name)) {
                    ++bad;
                    rep.error("state", "isolation: " + where + " includes <" + name +
                                           ">, not a standard library header this check allows");
                }
                continue;
            }
            const std::string target = normpath(rmcheck::dirname_of(rel) + "/" + name);
            const bool ok =
                allowed.count(target) || target == own ||
                (in_knowledge && rmcheck::starts_with(target, "src/knowledge/") && rmcheck::ends_with(target, ".h"));
            if (!ok) {
                ++bad;
                rep.error("state",
                    "isolation: " + where + " includes \"" + name + "\" (" + target +
                        "): only the standard library, its own header and the std-only decoder headers are allowed");
            }
        }
    }
    if (files.empty()) rep.error("state", "isolation: no knowledge-side source found");
    if (!bad && !files.empty())
        rep.info("OK   isolation: " + std::to_string(files.size()) +
                 " file(s) include only the standard library and std-only headers");
}

namespace rmcheck {

// The state tier: the step-1b self-checks over the replayed corpus, explore fuzz games and the scenarios; the dumps'
// re-encoding and the decoders' refusals; RMSTATE sessions against their dumps; the name-a-card
// candidate menus; the CARDS frame's ability parameters; the knowledge-side include isolation.
void tier_state(const Options &opt, Report &rep) {
    const double t0 = now_seconds();
    corpus_part(opt, rep);
    const double t1 = now_seconds();
    const std::vector<std::string> dumps = fuzz_part(opt, rep);
    const double t2 = now_seconds();
    scenarios_part(opt, rep);
    const double t3 = now_seconds();
    decode_part(opt, rep, dumps);
    check_card_params(opt, rep);
    isolation_part(opt, rep);
    for (const std::string &d : dumps) {
        remove_tree(d);
        remove_tree(d + ".jsonl");
    }
    rep.info("state: corpus " + fmt1(t1 - t0) + "s, fuzz " + fmt1(t2 - t1) + "s, scenarios " + fmt1(t3 - t2) +
             "s, decode + isolation " + fmt1(now_seconds() - t3) + "s");
}

}  // namespace rmcheck
