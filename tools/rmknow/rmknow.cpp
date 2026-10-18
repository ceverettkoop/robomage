// rmknow: one seat's knowledge from a recorded RMSTATE stream (docs/knowledge_store.md §2, §7).
//
//   rmknow [--viewer A|B] [--level off|expert|expert+belief] [--in FILE] [--out FILE] [--text | --quiet]
//          [--truth] [--isolation] [--stats] [--belief] [--hide-lists] < frames.bin > knowledge.bin
//   rmknow --selftest
//   rmknow --exactness N
//   rmknow --belief-exactness N
//
// --stats prints, after the summary, one RMKNOW_STATS line: frames, exact frames (k_meta.exact),
// the most nodes and junctions either player's network held, the largest and mean serialised state
// (Knowledge::serialize) and the percentiles and histogram of k_meta's solver steps.
// Reads RMSTATE records (an --rmstate-dump file, or a saved --machine stream), keeps the
// chosen seat's frames (SCHEMA, CARDS, DECKLISTS, METAGAME, STATE) and runs the knowledge sidecar on them at
// --level (default expert). With several viewers in the stream --viewer picks one (default: the
// stream's first frame's seat). A STATE frame's own k_* tables (the engine's, tables 90 and up)
// are split off before it is fed, so the sidecar sees each frame exactly as the engine's instance
// for that seat and writer did.
// Output, per STATE frame of that seat:
//   - default: one RMSTATE record holding the frame's k_* tables (the stream starts with the seat's
//     SCHEMA frame, so the output decodes like any RMSTATE stream);
//   - --text: a human-readable knowledge summary per frame instead;
//   - --quiet: nothing per frame.
// Every model inconsistency (the per-frame consistency check, an event the model cannot apply) is
// an `ERROR: knowledge…` line on stderr. --truth also checks every reported fact against the
// omniscient viewer's frames of the same stream (a --rmstate-dump holds them): the true world must
// satisfy each fact knowledge/truth.h lists (pool sizes, k_pool_card ranges, k_position cards and
// candidates, k_known_hand counts, k_seen counts, the reference lists equal to a registered list and
// their archetypes, from the omniscient DECKLISTS frame; face-down pools are not checked: the
// omniscient frames' handles are not the seat's); a violation is an `ERROR: rmknow truth…` line. The last
// stderr line is the summary
//   RMKNOW: viewer=<A|B> frames=<n> errors=<n>[ truth_frames=<n> truth_facts=<n> truth_errors=<n>]
// --isolation (implies --quiet unless --text) compares, for every STATE frame of the seat, the k_*
// tables this run computes with the ones the frame carries, byte for byte (docs/knowledge_store.md
// §7, isolation: the engine's instance is a pure function of the seat's frames from that writer).
// A difference, or a frame without k_* tables, is an `ERROR: rmknow isolation…` line naming the
// first differing table and row, and the summary adds isolation_frames=<n> isolation_errors=<n>.
// Exit status 0 when there was no error.
//
// --belief (implies --quiet; --level expert+belief) prints, per STATE frame of the seat in a live game
// whose opponent's list it does not see, one line for calibration (docs/knowledge_store.md §10.6):
//   RMKNOW_BELIEF: game=<g> decision=<d> true=<list ids equal to the opponent's registered list, from
//   the omniscient DECKLISTS frame, comma-separated, or -> exact=<0|1> prob=<b_list prob per list>
// --hide-lists feeds the seat's DECKLISTS frame without the opponent's list, as --closed-decklist
// would have sent it: the belief layer's calibration over games played with open decklists; the
// events are the same either way.
// It does not combine with --isolation (the frames' own tables were built with the list).
// --belief-exactness N checks the belief layer against the weighted enumeration of the hypothesis
// networks and the model played out directly on N simulated games (exactness.cpp); the last line is
// RMKNOW_BELIEF_EXACTNESS: cases=… enum_checks=… model_checks=… skipped=… failed=….
// --selftest runs the update-rule unit tests on synthetic event streams (selftest.cpp).
// --exactness N checks the exact solver and compaction against an independent enumerator of the
// worlds on N synthetic histories and N random event streams plus sculpted cases (exactness.cpp);
// the last line is RMKNOW_EXACTNESS: cases=… checks=… exact=… skipped=… failed=….

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "../../src/knowledge/belief.h"
#include "../../src/knowledge/knowledge.h"
#include "../../src/knowledge/truth.h"
#include "../../src/state_format/decode.h"
#include "../../src/state_format/schema.h"
#include "selftest.h"

namespace {

struct Options {
        int viewer = knowledge::NO_SEAT;
        rmstate::KnowledgeLevel level = rmstate::KnowledgeLevel::EXPERT;
        std::string in;
        std::string out;
        bool text = false;
        bool quiet = false;
        bool truth = false;
        bool isolation = false;
        bool selftest = false;
        int exactness = 0;
        int belief_exactness = 0;
        bool belief = false;
        bool hide_lists = false;
        bool stats = false;
};

// Per-frame cost and size figures (--stats).
struct Stats {
        size_t frames = 0;
        size_t exact_frames = 0;
        size_t max_nodes[2] = {0, 0};
        size_t max_junctions[2] = {0, 0};
        size_t max_blob = 0;
        uint64_t blob_sum = 0;
        std::vector<uint32_t> steps;
};

}  // namespace

static bool parse_args(int argc, char **argv, Options *opt);
static size_t check_frame_truth(const knowledge::Knowledge &know, const knowledge::KRows &rows,
    const knowledge::Truth &truth, const knowledge::Snapshot &omni, size_t *facts);
static void truth_error(const std::string &what, const knowledge::Snapshot &omni, size_t *errors);
static void add_stats(const knowledge::Knowledge &know, const knowledge::KRows &rows, Stats *st);
static void print_belief(const knowledge::Knowledge &know, const knowledge::Decklists *truth, FILE *out);
static void feed_hidden_lists(knowledge::Knowledge &know, const std::string &payload);
static void print_stats(Stats *st);
static std::string tables_frame(const std::string &blocks, uint32_t n, rmstate::Viewer viewer, uint64_t hash);
static std::string describe_k_difference(const knowledge::Knowledge &know, const std::string &engine, uint32_t n_engine,
    const std::string &mine, uint32_t n_mine, rmstate::Viewer viewer);

static bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == "--viewer" && has_value) {
            const std::string v = argv[++i];
            if (v == "A" || v == "a")
                opt->viewer = 0;
            else if (v == "B" || v == "b")
                opt->viewer = 1;
            else {
                std::fprintf(stderr, "rmknow: --viewer takes A or B, got '%s'\n", v.c_str());
                return false;
            }
        } else if (a == "--level" && has_value) {
            const std::string v = argv[++i];
            if (v == "off")
                opt->level = rmstate::KnowledgeLevel::OFF;
            else if (v == "expert")
                opt->level = rmstate::KnowledgeLevel::EXPERT;
            else if (v == "expert+belief")
                opt->level = rmstate::KnowledgeLevel::EXPERT_BELIEF;
            else {
                std::fprintf(stderr, "rmknow: --level takes off, expert or expert+belief, got '%s'\n", v.c_str());
                return false;
            }
        } else if (a == "--isolation") {
            opt->isolation = true;
        } else if (a == "--in" && has_value) {
            opt->in = argv[++i];
        } else if (a == "--out" && has_value) {
            opt->out = argv[++i];
        } else if (a == "--text") {
            opt->text = true;
        } else if (a == "--quiet") {
            opt->quiet = true;
        } else if (a == "--truth") {
            opt->truth = true;
        } else if (a == "--stats") {
            opt->stats = true;
        } else if (a == "--selftest") {
            opt->selftest = true;
        } else if (a == "--exactness" && has_value) {
            opt->exactness = std::atoi(argv[++i]);
            if (opt->exactness <= 0) {
                std::fprintf(stderr, "rmknow: --exactness takes a positive count\n");
                return false;
            }
        } else if (a == "--belief-exactness" && has_value) {
            opt->belief_exactness = std::atoi(argv[++i]);
            if (opt->belief_exactness <= 0) {
                std::fprintf(stderr, "rmknow: --belief-exactness takes a positive count\n");
                return false;
            }
        } else if (a == "--belief") {
            opt->belief = true;
        } else if (a == "--hide-lists") {
            opt->hide_lists = true;
        } else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                "usage: rmknow [--viewer A|B] [--level off|expert|expert+belief] [--in FILE] [--out FILE] [--text | "
                "--quiet] "
                "[--truth]\n"
                "              [--isolation] [--stats] [--belief] [--hide-lists] < frames > knowledge\n"
                "       rmknow --selftest\n"
                "       rmknow --exactness N\n"
                "       rmknow --belief-exactness N\n");
            return false;
        } else {
            std::fprintf(stderr, "rmknow: unknown argument '%s' (see --help)\n", a.c_str());
            return false;
        }
    }
    if (opt->isolation && !opt->text) opt->quiet = true;
    if (opt->hide_lists && opt->isolation) {
        std::fprintf(stderr, "rmknow: --hide-lists does not combine with --isolation\n");
        return false;
    }
    if (opt->belief) {
        opt->quiet = true;
        opt->level = rmstate::KnowledgeLevel::EXPERT_BELIEF;
    }
    return true;
}

// A STATE frame holding only `blocks` (n table blocks) beside an empty string pool, for decoding.
static std::string tables_frame(const std::string &blocks, uint32_t n, rmstate::Viewer viewer, uint64_t hash) {
    std::string frame = rmstate::FrameWriter(rmstate::FrameKind::STATE, viewer, hash).finish();
    if (!rmstate::append_frame_blocks(&frame, blocks, n).ok()) return std::string();
    return frame;
}

// Where the frame's own k_* tables (`engine`) and this run's (`mine`) first differ: the table and
// row, with both rows' cells.
static std::string describe_k_difference(const knowledge::Knowledge &know, const std::string &engine, uint32_t n_engine,
    const std::string &mine, uint32_t n_mine, rmstate::Viewer viewer) {
    rmstate::DecodedFrame a, b;
    if (!rmstate::decode_frame(tables_frame(engine, n_engine, viewer, know.schema_hash()), know.schema(), &a).ok() ||
        !rmstate::decode_frame(tables_frame(mine, n_mine, viewer, know.schema_hash()), know.schema(), &b).ok())
        return "the k_* tables differ (and do not decode)";
    const auto row_text = [](const rmstate::DecodedTable &t, size_t r) {
        std::string out;
        for (size_t c = 0; c < t.def->columns.size(); ++c)
            out += (c ? " " : "") + t.def->columns[c].name + "=" + std::to_string(t.rows[r][c].bits);
        return out;
    };
    for (const rmstate::TableDef &def : know.schema().tables) {
        const uint16_t id = static_cast<uint16_t>(def.id);
        if (id < static_cast<uint16_t>(rmstate::TableId::K_META)) continue;
        const rmstate::DecodedTable *x = a.table(id);
        const rmstate::DecodedTable *y = b.table(id);
        const size_t nx = x ? x->rows.size() : 0, ny = y ? y->rows.size() : 0;
        for (size_t r = 0; r < std::max(nx, ny); ++r) {
            const std::string ex = r < nx ? row_text(*x, r) : "(none)";
            const std::string my = r < ny ? row_text(*y, r) : "(none)";
            if (ex != my) return def.name + " row " + std::to_string(r) + ": frame has " + ex + ", rmknow has " + my;
        }
    }
    return "the k_* tables differ in their layout";
}

// The seat's DECKLISTS frame with the opponent's list left out (--hide-lists).
static void feed_hidden_lists(knowledge::Knowledge &know, const std::string &payload) {
    rmstate::DecodedFrame frame;
    knowledge::Decklists lists;
    std::string err;
    if (!rmstate::decode_frame(payload, know.schema(), &frame).ok() ||
        !knowledge::read_decklists(frame, know.codes(), &lists, &err)) {
        know.feed(payload);
        return;
    }
    const int opp = 1 - know.viewer();
    lists.present[opp] = false;
    lists.name[opp].clear();
    std::vector<knowledge::DeckEntry> kept;
    for (const knowledge::DeckEntry &e : lists.cards)
        if (e.player != opp) kept.push_back(e);
    lists.cards = kept;
    know.begin_match(know.viewer(), know.codes(), lists);
}

// One RMKNOW_BELIEF line (--belief): the opponent's b_list probabilities and which lists are truly
// theirs.
static void print_belief(const knowledge::Knowledge &know, const knowledge::Decklists *truth, FILE *out) {
    const int opp = 1 - know.viewer();
    std::string true_lists;
    if (truth) {
        std::vector<std::pair<int, int>> main, side;
        knowledge::registered_counts(*truth, opp, &main, &side);
        for (int l : knowledge::matching_lists(know.metagame(), main, side))
            true_lists += (true_lists.empty() ? "" : ",") + std::to_string(l);
    }
    std::string probs;
    bool exact = true;
    for (const knowledge::BListRow &b : know.rows().belief_lists) {
        if (b.player != opp) continue;
        probs += (probs.empty() ? "" : ",") + std::to_string(b.prob);
        if (b.prob > 0 && !b.exact) exact = false;
    }
    std::fprintf(out, "RMKNOW_BELIEF: game=%u decision=%u true=%s exact=%d prob=%s\n", know.frame_game(),
        know.frame_decision(), true_lists.empty() ? "-" : true_lists.c_str(), exact ? 1 : 0, probs.c_str());
}

static void add_stats(const knowledge::Knowledge &know, const knowledge::KRows &rows, Stats *st) {
    ++st->frames;
    if (rows.meta.exact) ++st->exact_frames;
    st->steps.push_back(rows.meta.steps);
    for (int p = 0; p < 2; ++p) {
        st->max_nodes[p] = std::max(st->max_nodes[p], know.player(p).net.node_count());
        st->max_junctions[p] = std::max(st->max_junctions[p], know.player(p).net.junction_count());
    }
    std::string blob;
    know.serialize(blob);
    st->max_blob = std::max(st->max_blob, blob.size());
    st->blob_sum += blob.size();
}

// RMKNOW_STATS: frames, exact frames, the most nodes and junctions per player, the largest and
// mean serialised state, the k_meta step counts' percentiles and their histogram (frames with at
// most 10^2, 10^3, 10^4, 10^5, 10^6 steps, and more).
static void print_stats(Stats *st) {
    std::vector<uint32_t> &v = st->steps;
    std::sort(v.begin(), v.end());
    const auto pct = [&v](double q) {
        return v.empty() ? 0u : v[static_cast<size_t>(q * static_cast<double>(v.size() - 1))];
    };
    size_t hist[6] = {0, 0, 0, 0, 0, 0};
    for (uint32_t x : v) {
        size_t b = 0;
        for (uint64_t lim = 100; b < 5 && x > lim; lim *= 10) ++b;
        ++hist[b];
    }
    std::fprintf(stderr,
        "RMKNOW_STATS: frames=%zu exact=%zu nodes=%zu/%zu junctions=%zu/%zu blob_max=%zu blob_mean=%llu "
        "steps_p50=%u p90=%u p99=%u max=%u hist=%zu/%zu/%zu/%zu/%zu/%zu\n",
        st->frames, st->exact_frames, st->max_nodes[0], st->max_nodes[1], st->max_junctions[0], st->max_junctions[1],
        st->max_blob, static_cast<unsigned long long>(st->frames ? st->blob_sum / st->frames : 0), pct(0.5), pct(0.9),
        pct(0.99), v.empty() ? 0u : v.back(), hist[0], hist[1], hist[2], hist[3], hist[4], hist[5]);
}

static void truth_error(const std::string &what, const knowledge::Snapshot &omni, size_t *errors) {
    ++*errors;
    if (*errors > 50) return;
    std::fprintf(stderr, "ERROR: rmknow truth: game %u decision %u: %s\n", omni.game, omni.decision, what.c_str());
}

// Every fact of `rows` holds in the true world (knowledge/truth.h). Returns the violations.
static size_t check_frame_truth(const knowledge::Knowledge &know, const knowledge::KRows &rows,
    const knowledge::Truth &truth, const knowledge::Snapshot &omni, size_t *facts) {
    const knowledge::Universe *const universe[2] = {&know.player(0).full.universe, &know.player(1).full.universe};
    knowledge::TruthCheck check;
    knowledge::check_truth(rows, truth, know.viewer(), universe, &check);
    *facts += check.facts;
    size_t errors = 0;
    for (const std::string &e : check.errors) truth_error(e, omni, &errors);
    return errors;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return 2;
    if (opt.selftest) return run_selftest();
    if (opt.exactness) return run_exactness(opt.exactness);
    if (opt.belief_exactness) return run_belief_exactness(opt.belief_exactness);
    FILE *in = stdin;
    if (!opt.in.empty() && opt.in != "-") {
        in = std::fopen(opt.in.c_str(), "rb");
        if (!in) {
            std::fprintf(stderr, "rmknow: cannot open %s\n", opt.in.c_str());
            return 2;
        }
    }
    FILE *out = stdout;
    if (!opt.out.empty() && opt.out != "-") {
        out = std::fopen(opt.out.c_str(), "wb");
        if (!out) {
            std::fprintf(stderr, "rmknow: cannot open %s\n", opt.out.c_str());
            return 2;
        }
    }

    knowledge::Knowledge know(opt.level);
    rmstate::RecordReader reader(in);
    std::string payload;
    rmstate::DecodeResult res;
    rmstate::Schema omni_schema;
    bool have_omni_schema = false;
    knowledge::Codes omni_codes;
    knowledge::Decklists omni_lists;  // both registered lists, from the omniscient DECKLISTS frame
    bool have_omni_lists = false;
    size_t frames = 0, errors = 0, truth_frames = 0, truth_facts = 0, truth_errors = 0, isolation_errors = 0;
    knowledge::KRows last_rows;
    knowledge::Snapshot last_snap;
    bool have_last = false;
    bool wrote_schema = false;
    Stats stats;

    while (reader.next(&payload, &res)) {
        rmstate::FrameKind kind;
        rmstate::Viewer viewer;
        if (!knowledge::peek_frame_header(payload, &kind, &viewer)) {
            std::fprintf(stderr, "ERROR: rmknow: record %zu is not an RMSTATE frame\n", reader.count());
            ++errors;
            continue;
        }
        if (viewer == rmstate::Viewer::OMNISCIENT) {
            if (!opt.truth && !opt.belief) continue;
            if (kind == rmstate::FrameKind::SCHEMA) {
                std::string err;
                have_omni_schema = rmstate::decode_schema_frame(payload, &omni_schema).ok() &&
                                   knowledge::read_codes(omni_schema, &omni_codes, &err);
                continue;
            }
            if (kind == rmstate::FrameKind::DECKLISTS && have_omni_schema) {
                rmstate::DecodedFrame frame;
                std::string err;
                have_omni_lists = rmstate::decode_frame(payload, omni_schema, &frame).ok() &&
                                  knowledge::read_decklists(frame, omni_codes, &omni_lists, &err);
                continue;
            }
            if (kind != rmstate::FrameKind::STATE || !have_omni_schema || !have_last || !opt.truth) continue;
            rmstate::DecodedFrame frame;
            knowledge::Snapshot omni;
            std::string err;
            if (!rmstate::decode_frame(payload, omni_schema, &frame).ok() ||
                !knowledge::read_snapshot(frame, omni_codes, &omni, &err)) {
                std::fprintf(stderr, "ERROR: rmknow: an omniscient frame does not decode\n");
                ++errors;
                continue;
            }
            if (omni.game != last_snap.game || omni.sideboard_phase != last_snap.sideboard_phase ||
                omni.decision != last_snap.decision || omni.terminal)
                continue;
            knowledge::Truth truth;
            if (!knowledge::truth_from_snapshot(omni, &truth)) {
                truth_error("the omniscient frame's library positions are not 0..n-1", omni, &truth_errors);
                continue;
            }
            if (have_omni_lists) {
                const bool visible[2] = {know.decklists().present[0], know.decklists().present[1]};
                knowledge::set_truth_lists(know.metagame(), omni_lists, visible, &truth);
            }
            ++truth_frames;
            truth_errors += check_frame_truth(know, last_rows, truth, omni, &truth_facts);
            have_last = false;
            continue;
        }
        const int seat = viewer == rmstate::Viewer::A ? 0 : 1;
        if (opt.viewer == knowledge::NO_SEAT) opt.viewer = seat;
        if (seat != opt.viewer) continue;
        // A STATE frame is fed without the k_* tables the engine appended to it.
        std::string fed = payload, embedded;
        uint32_t n_embedded = 0;
        if (kind == rmstate::FrameKind::STATE) {
            const rmstate::DecodeResult split = rmstate::split_frame_blocks(
                payload, static_cast<uint16_t>(rmstate::TableId::K_META), &fed, &embedded, &n_embedded);
            if (!split.ok()) {
                std::fprintf(stderr, "ERROR: rmknow: record %zu: %s\n", reader.count(), split.message.c_str());
                ++errors;
                continue;
            }
        }
        if (opt.hide_lists && kind == rmstate::FrameKind::DECKLISTS && know.started())
            feed_hidden_lists(know, fed);
        else
            know.feed(fed);
        for (const std::string &e : know.take_errors()) {
            ++errors;
            if (errors <= 200) std::fprintf(stderr, "%s\n", e.c_str());
        }
        if (kind == rmstate::FrameKind::SCHEMA && !opt.text && !opt.quiet && !wrote_schema) {
            const std::string rec = rmstate::frame_record(payload);
            std::fwrite(rec.data(), 1, rec.size(), out);
            wrote_schema = true;
        }
        if (kind != rmstate::FrameKind::STATE) continue;
        ++frames;
        if (opt.isolation) {
            std::string mine;
            uint32_t n_mine = 0;
            std::string what;
            if (n_embedded == 0)
                what = "the frame carries no k_* tables";
            else if (!know.table_blocks(&mine, &n_mine))
                what = "no k_* tables before the SCHEMA frame";
            else if (mine != embedded || n_mine != n_embedded)
                what = describe_k_difference(know, embedded, n_embedded, mine, n_mine, viewer);
            if (!what.empty()) {
                ++isolation_errors;
                if (isolation_errors <= 50)
                    std::fprintf(stderr, "ERROR: rmknow isolation: frame %zu (game %u decision %u): %s\n", frames,
                        know.frame_game(), know.frame_decision(), what.c_str());
            }
        }
        if (opt.stats && know.game_active()) add_stats(know, know.rows(), &stats);
        if (opt.belief && know.game_active() && !know.decklists().present[1 - opt.viewer])
            print_belief(know, have_omni_lists ? &omni_lists : nullptr, out);
        if (opt.truth) {
            rmstate::DecodedFrame frame;
            std::string err;
            last_rows = know.rows();
            have_last = rmstate::decode_frame(fed, know.schema(), &frame).ok() &&
                        knowledge::read_snapshot(frame, know.codes(), &last_snap, &err) && know.game_active();
        }
        if (opt.quiet) continue;
        if (opt.text) {
            std::fprintf(out, "frame %zu (game %u decision %u):\n%s", frames, know.frame_game(), know.frame_decision(),
                know.text().c_str());
            continue;
        }
        rmstate::FrameWriter w(rmstate::FrameKind::STATE, viewer, know.schema_hash());
        know.write_tables(w);
        const std::string rec = rmstate::frame_record(w.finish());
        std::fwrite(rec.data(), 1, rec.size(), out);
    }
    if (!res.ok()) {
        std::fprintf(stderr, "ERROR: rmknow: %s\n", res.message.c_str());
        ++errors;
    }
    std::fprintf(stderr, "RMKNOW: viewer=%s frames=%zu errors=%zu", opt.viewer == 1 ? "B" : "A", frames, errors);
    if (opt.truth)
        std::fprintf(
            stderr, " truth_frames=%zu truth_facts=%zu truth_errors=%zu", truth_frames, truth_facts, truth_errors);
    if (opt.isolation) std::fprintf(stderr, " isolation_frames=%zu isolation_errors=%zu", frames, isolation_errors);
    std::fprintf(stderr, "\n");
    if (opt.stats) print_stats(&stats);
    if (in != stdin) std::fclose(in);
    if (out != stdout) std::fclose(out);
    return errors == 0 && truth_errors == 0 && isolation_errors == 0 ? 0 : 1;
}
