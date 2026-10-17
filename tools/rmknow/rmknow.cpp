// rmknow: one seat's knowledge from a recorded RMSTATE stream (docs/knowledge_store.md §2, §7).
//
//   rmknow [--viewer A|B] [--in FILE] [--out FILE] [--text | --quiet] [--truth] < frames.bin > knowledge.bin
//   rmknow --selftest
//   rmknow --exactness N
//
// --stats prints, after the summary, one RMKNOW_STATS line: frames, exact frames (k_meta.exact),
// the most nodes and junctions either player's network held, the largest and mean serialised state
// (Knowledge::serialize) and the percentiles and histogram of k_meta's solver steps.
// Reads RMSTATE records (an --rmstate-dump file, or a --machine-format rmstate stream), keeps the
// chosen seat's frames (SCHEMA, CARDS, DECKLISTS, STATE) and runs the knowledge sidecar on them.
// With several viewers in the stream --viewer picks one (default: the stream's first frame's seat).
// Output, per STATE frame of that seat:
//   - default: one RMSTATE record holding the frame's k_* tables (the stream starts with the seat's
//     SCHEMA frame, so the output decodes like any RMSTATE stream);
//   - --text: a human-readable knowledge summary per frame instead;
//   - --quiet: nothing per frame.
// Every model inconsistency (the per-frame consistency check, an event the model cannot apply) is
// an `ERROR: knowledge…` line on stderr. --truth also checks every reported fact against the
// omniscient viewer's frames of the same stream (a --rmstate-dump holds them): the true world must
// satisfy each k_pool_card range, k_position card, k_known_hand count and pool size, and each
// k_seen game count must not exceed the copies the player really has; a violation is an
// `ERROR: rmknow truth…` line. The last stderr line is the summary
//   RMKNOW: viewer=<A|B> frames=<n> errors=<n>[ truth_frames=<n> truth_errors=<n>]
// Exit status 0 when there was no error.
//
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

#include "../../src/knowledge/knowledge.h"
#include "../../src/state_format/decode.h"
#include "../../src/state_format/schema.h"
#include "selftest.h"

namespace {

struct Options {
        int viewer = knowledge::NO_SEAT;
        std::string in;
        std::string out;
        bool text = false;
        bool quiet = false;
        bool truth = false;
        bool selftest = false;
        int exactness = 0;
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

// The true world of one decision, from the omniscient frame.
struct Truth {
        std::map<int, int> hand[2];
        std::map<int, int> sideboard[2];
        std::map<int, int> owned[2];  // every non-token card object of the player
        std::vector<int> library[2];  // top first
};

}  // namespace

static bool parse_args(int argc, char **argv, Options *opt);
static bool read_truth(const knowledge::Snapshot &snap, Truth *out);
static size_t check_truth(const knowledge::Knowledge &know, const knowledge::KRows &rows, const Truth &truth,
    const knowledge::Snapshot &omni, size_t frame);
static void truth_error(const std::string &what, const knowledge::Snapshot &omni, size_t *errors);
static int count_of(const std::map<int, int> &m, int card);
static void add_stats(const knowledge::Knowledge &know, const knowledge::KRows &rows, Stats *st);
static void print_stats(Stats *st);

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
        } else if (a == "--help" || a == "-h") {
            std::fprintf(stderr,
                "usage: rmknow [--viewer A|B] [--in FILE] [--out FILE] [--text | --quiet] [--truth] [--stats] < frames "
                "> "
                "knowledge\n"
                "       rmknow --selftest\n"
                "       rmknow --exactness N\n");
            return false;
        } else {
            std::fprintf(stderr, "rmknow: unknown argument '%s' (see --help)\n", a.c_str());
            return false;
        }
    }
    return true;
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

static int count_of(const std::map<int, int> &m, int card) {
    const auto it = m.find(card);
    return it == m.end() ? 0 : it->second;
}

static bool read_truth(const knowledge::Snapshot &snap, Truth *out) {
    *out = Truth{};
    std::map<int, int> lib_pos[2];
    for (const knowledge::SnapObject &o : snap.objects) {
        if (o.owner == knowledge::NO_SEAT || o.token) continue;
        if (o.card != knowledge::NO_CARD) ++out->owned[o.owner][o.card];
        if (o.zone == knowledge::Zn::HAND) ++out->hand[o.owner][o.card];
        if (o.zone == knowledge::Zn::SIDEBOARD) ++out->sideboard[o.owner][o.card];
        if (o.zone == knowledge::Zn::LIBRARY) lib_pos[o.owner][o.pos] = o.card;
    }
    for (int p = 0; p < 2; ++p) {
        for (const auto &e : lib_pos[p]) {
            if (e.first != static_cast<int>(out->library[p].size())) return false;
            out->library[p].push_back(e.second);
        }
    }
    return true;
}

static void truth_error(const std::string &what, const knowledge::Snapshot &omni, size_t *errors) {
    ++*errors;
    if (*errors > 50) return;
    std::fprintf(stderr, "ERROR: rmknow truth: game %u decision %u: %s\n", omni.game, omni.decision, what.c_str());
}

// Every fact of `rows` holds in the true world. Returns the violations.
static size_t check_truth(const knowledge::Knowledge &know, const knowledge::KRows &rows, const Truth &truth,
    const knowledge::Snapshot &omni, size_t frame) {
    (void)frame;
    size_t errors = 0;
    // The true contents of each pool.
    std::vector<std::map<int, int>> contents(rows.pools.size());
    std::vector<bool> checkable(rows.pools.size(), true);
    for (size_t i = 0; i < rows.pools.size(); ++i) {
        const knowledge::KPoolRow &pool = rows.pools[i];
        const int p = pool.owner;
        size_t true_size = 0;
        switch (pool.kind) {
            case rmstate::PoolKind::HAND:
                contents[i] = truth.hand[p];
                break;
            case rmstate::PoolKind::SIDEBOARD:
                contents[i] = truth.sideboard[p];
                break;
            case rmstate::PoolKind::LIBRARY_RANGE:
                for (size_t k = pool.range.pos; k < static_cast<size_t>(pool.range.pos) + pool.size; ++k)
                    if (k < truth.library[p].size()) ++contents[i][truth.library[p][k]];
                break;
            default:
                checkable[i] = false;  // face-down and set-aside units: no handle shared with the truth
                break;
        }
        if (!checkable[i]) continue;
        for (const auto &c : contents[i]) true_size += static_cast<size_t>(c.second);
        if (true_size != pool.size)
            truth_error("pool " + std::to_string(i) + " has size " + std::to_string(pool.size) + ", truly " +
                            std::to_string(true_size),
                omni, &errors);
    }
    std::vector<std::map<int, bool>> listed(rows.pools.size());
    for (const knowledge::KPoolCardRow &c : rows.pool_cards) {
        if (c.pool >= rows.pools.size() || !checkable[c.pool]) continue;
        listed[c.pool][c.card] = true;
        const int n = count_of(contents[c.pool], c.card);
        if (n < c.min || n > c.max)
            truth_error("pool " + std::to_string(c.pool) + " holds " + std::to_string(n) + " of card " +
                            std::to_string(c.card) + ", reported " + std::to_string(c.min) + "-" +
                            std::to_string(c.max),
                omni, &errors);
    }
    for (size_t i = 0; i < rows.pools.size(); ++i) {
        if (!checkable[i]) continue;
        const knowledge::Universe &uni = know.player(rows.pools[i].owner).full.universe;
        for (const auto &c : contents[i])
            if (!listed[i][c.first] && uni.contains(c.first))
                truth_error(
                    "pool " + std::to_string(i) + " holds card " + std::to_string(c.first) + ", reported impossible",
                    omni, &errors);
    }
    for (const knowledge::KPositionRow &pos : rows.positions) {
        if (pos.card == knowledge::NO_CARD) continue;
        const std::vector<int> &lib = truth.library[pos.owner];
        const int at = pos.at.pos;
        if (at >= static_cast<int>(lib.size()) || lib[static_cast<size_t>(at)] != pos.card)
            truth_error("library position " + std::to_string(at) + " is reported card " + std::to_string(pos.card) +
                            ", truly " +
                            (at < static_cast<int>(lib.size()) ? std::to_string(lib[static_cast<size_t>(at)]) : "none"),
                omni, &errors);
    }
    const int opp = 1 - know.viewer();
    for (const knowledge::KKnownHandRow &k : rows.known_hand)
        if (count_of(truth.hand[opp], k.card) < k.copies)
            truth_error("the opponent's hand is reported to hold " + std::to_string(k.copies) + " of card " +
                            std::to_string(k.card) + ", truly " + std::to_string(count_of(truth.hand[opp], k.card)),
                omni, &errors);
    for (const knowledge::KSeenRow &s : rows.seen)
        if (s.game > count_of(truth.owned[s.player], s.card))
            truth_error("seen " + std::to_string(s.game) + " of card " + std::to_string(s.card) +
                            " this game, the player has " + std::to_string(count_of(truth.owned[s.player], s.card)),
                omni, &errors);
    return errors;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return 2;
    if (opt.selftest) return run_selftest();
    if (opt.exactness) return run_exactness(opt.exactness);
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

    knowledge::Knowledge know;
    rmstate::RecordReader reader(in);
    std::string payload;
    rmstate::DecodeResult res;
    rmstate::Schema omni_schema;
    bool have_omni_schema = false;
    knowledge::Codes omni_codes;
    size_t frames = 0, errors = 0, truth_frames = 0, truth_errors = 0;
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
            if (!opt.truth) continue;
            if (kind == rmstate::FrameKind::SCHEMA) {
                std::string err;
                have_omni_schema = rmstate::decode_schema_frame(payload, &omni_schema).ok() &&
                                   knowledge::read_codes(omni_schema, &omni_codes, &err);
                continue;
            }
            if (kind != rmstate::FrameKind::STATE || !have_omni_schema || !have_last) continue;
            rmstate::DecodedFrame frame;
            knowledge::Snapshot omni;
            std::string err;
            if (!rmstate::decode_frame(payload, omni_schema, &frame).ok() ||
                !knowledge::read_snapshot(frame, omni_codes, &omni, &err)) {
                std::fprintf(stderr, "ERROR: rmknow: an omniscient frame does not decode\n");
                ++errors;
                continue;
            }
            if (omni.game != last_snap.game || omni.decision != last_snap.decision || omni.terminal) continue;
            Truth truth;
            if (!read_truth(omni, &truth)) {
                truth_error("the omniscient frame's library positions are not 0..n-1", omni, &truth_errors);
                continue;
            }
            ++truth_frames;
            truth_errors += check_truth(know, last_rows, truth, omni, frames);
            have_last = false;
            continue;
        }
        const int seat = viewer == rmstate::Viewer::A ? 0 : 1;
        if (opt.viewer == knowledge::NO_SEAT) opt.viewer = seat;
        if (seat != opt.viewer) continue;
        know.feed(payload);
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
        if (opt.stats && know.game_active()) add_stats(know, know.rows(), &stats);
        if (opt.truth) {
            rmstate::DecodedFrame frame;
            std::string err;
            last_rows = know.rows();
            have_last = rmstate::decode_frame(payload, know.schema(), &frame).ok() &&
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
    if (opt.truth) std::fprintf(stderr, " truth_frames=%zu truth_errors=%zu", truth_frames, truth_errors);
    std::fprintf(stderr, "\n");
    if (opt.stats) print_stats(&stats);
    if (in != stdin) std::fclose(in);
    if (out != stdout) std::fclose(out);
    return errors == 0 && truth_errors == 0 ? 0 : 1;
}
