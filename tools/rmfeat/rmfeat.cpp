// rmfeat: the featurizer over recorded RMSTATE frames (docs/featurizer.md §3-4, §6).
//
//   rmfeat --spec --cards CARDS
//   rmfeat --export-tables CARDS OUTDIR
//   rmfeat --featurize DUMP --viewer A|B|omniscient --out FILE --cards CARDS [--stats]
//   rmfeat --targets DUMP --seat A|B --out FILE --cards CARDS
//   rmfeat --stats DUMP --cards CARDS
//
// Every mode also takes --text-table FILE (default bin/resources/embeddings/oracle_text.rmtb, else
// resources/embeddings/oracle_text.rmtb: its row count is N_CARD_TYPES + 1 and its vocab_hash the
// vocab's) and the featurizing modes --tables DIR (read the exported card tables instead of building
// them; they must carry the spec's hash) and --history N (the spec's recent-event history tokens, at
// most N events: docs/featurizer.md §3.7; a seat's frames are fed to its event history in order).
// CARDS is robomage --cards-frame's file (SCHEMA, the whole-vocab CARDS frame), DUMP an
// --rmstate-dump file.
//
// --spec prints FEATURE_SPEC (one line of JSON). --export-tables writes OUTDIR/card.rmtb,
// ability.rmtb, part.rmtb and spec.json and prints one RMFEAT line per file. --featurize writes one
// RMTB tensors file with an entry per decision of the viewer that has a menu (featurize_viewer;
// featurize_seat for a seat). --targets writes the learner targets of every decision of the seat that
// has a menu, from the omniscient frame of the same decision (targets.h). --stats featurizes every
// decision of every viewer with a menu (with --featurize: of that viewer) and prints token counts
// per token type (mean and most per decision) and the timing (RMFEAT_STATS lines). Errors go to stderr as `rmfeat: …`;
// exit 0, 1 on an error in the data, 2 on bad arguments or files.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/rmtb.h"
#include "../../src/featurize/setup.h"
#include "../../src/featurize/targets.h"

namespace {

struct Options {
        bool spec = false;
        std::string export_cards, export_dir;
        std::string featurize, targets, stats_dump;
        bool stats = false;
        int viewer = -1;  // rmstate::Viewer
        std::string out, cards, text_table, tables;
        uint32_t history = 0;
};

// Per token type rows, and the decisions' totals, for --stats.
struct Stats {
        size_t decisions = 0, tokens = 0, max_tokens = 0, relations = 0, actions = 0;
        std::vector<size_t> rows, max_rows;  // per token type: tokens over all decisions, most in one
        double featurize_s = 0, total_s = 0;
};

const char *const TEXT_TABLES[] = {
    "bin/resources/embeddings/oracle_text.rmtb", "resources/embeddings/oracle_text.rmtb"};

}  // namespace

static int usage();
static int viewer_of(const char *s);
static bool parse_args(int argc, char **argv, Options *opt);
static bool file_exists(const std::string &path);
static std::string text_table_path(const Options &opt);
static bool load(const Options &opt, featurize::Featurization *f);
static size_t menu_rows(const featurize::FeatureSpec &spec, const rmstate::DecodedFrame &state);
static double seconds_since(std::chrono::steady_clock::time_point t0);
static void count(const featurize::Tensors &t, Stats *s);
static void print_stats(const featurize::FeatureSpec &spec, const Stats &s);
static int run_featurize(const Options &opt, const featurize::Featurization &f, const std::string &dump, bool write);
static int run_targets(const Options &opt, const featurize::Featurization &f);

static int usage() {
    std::fprintf(stderr,
        "usage: rmfeat --spec --cards CARDS\n"
        "       rmfeat --export-tables CARDS OUTDIR\n"
        "       rmfeat --featurize DUMP --viewer A|B|omniscient --out FILE --cards CARDS [--stats]\n"
        "       rmfeat --targets DUMP --seat A|B --out FILE --cards CARDS\n"
        "       rmfeat --stats DUMP --cards CARDS\n"
        "  (each also --text-table FILE and --history N; the featurizing modes --tables DIR)\n");
    return 2;
}

static int viewer_of(const char *s) {
    if (!std::strcmp(s, "A") || !std::strcmp(s, "a")) return static_cast<int>(rmstate::Viewer::A);
    if (!std::strcmp(s, "B") || !std::strcmp(s, "b")) return static_cast<int>(rmstate::Viewer::B);
    if (!std::strcmp(s, "omniscient") || !std::strcmp(s, "O")) return static_cast<int>(rmstate::Viewer::OMNISCIENT);
    return -1;
}

static bool parse_args(int argc, char **argv, Options *opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has1 = i + 1 < argc, has2 = i + 2 < argc;
        if (a == "--spec") {
            opt->spec = true;
        } else if (a == "--export-tables" && has2) {
            opt->export_cards = argv[++i];
            opt->export_dir = argv[++i];
        } else if (a == "--featurize" && has1) {
            opt->featurize = argv[++i];
        } else if (a == "--targets" && has1) {
            opt->targets = argv[++i];
        } else if (a == "--stats") {
            opt->stats = true;
            if (has1 && argv[i + 1][0] != '-') opt->stats_dump = argv[++i];
        } else if ((a == "--viewer" || a == "--seat") && has1) {
            opt->viewer = viewer_of(argv[++i]);
            if (opt->viewer < 0 || (a == "--seat" && opt->viewer > 1)) return false;
        } else if (a == "--out" && has1) {
            opt->out = argv[++i];
        } else if (a == "--cards" && has1) {
            opt->cards = argv[++i];
        } else if (a == "--text-table" && has1) {
            opt->text_table = argv[++i];
        } else if (a == "--tables" && has1) {
            opt->tables = argv[++i];
        } else if (a == "--history" && has1) {
            opt->history = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        } else {
            return false;
        }
    }
    if (!opt->export_cards.empty()) opt->cards = opt->export_cards;
    const int modes = (opt->spec ? 1 : 0) + (opt->export_cards.empty() ? 0 : 1) + (opt->featurize.empty() ? 0 : 1) +
                      (opt->targets.empty() ? 0 : 1) + (opt->stats_dump.empty() ? 0 : 1);
    if (modes != 1 || opt->cards.empty()) return false;
    if (!opt->featurize.empty() && (opt->viewer < 0 || (opt->out.empty() && !opt->stats))) return false;
    if (!opt->targets.empty() && (opt->viewer < 0 || opt->viewer > 1 || opt->out.empty())) return false;
    return true;
}

static bool file_exists(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (f) std::fclose(f);
    return f != nullptr;
}

static std::string text_table_path(const Options &opt) {
    if (!opt.text_table.empty()) return opt.text_table;
    for (const char *p : TEXT_TABLES)
        if (file_exists(p)) return p;
    return TEXT_TABLES[0];
}

static bool load(const Options &opt, featurize::Featurization *f) {
    std::string err;
    f->options.history_events = opt.history;
    if (f->load(opt.cards, text_table_path(opt), opt.tables, &err)) return true;
    std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
    return false;
}

static size_t menu_rows(const featurize::FeatureSpec &spec, const rmstate::DecodedFrame &state) {
    const rmstate::DecodedTable *m = state.table(spec.types[static_cast<size_t>(spec.menu_type)].table);
    return m ? m->rows.size() : 0;
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

static void count(const featurize::Tensors &t, Stats *s) {
    ++s->decisions;
    s->tokens += t.n();
    s->max_tokens = std::max(s->max_tokens, t.n());
    s->relations += t.r();
    s->actions += t.a();
    std::vector<size_t> here(s->rows.size(), 0);
    for (const int64_t ty : t.tok_type) ++here[static_cast<size_t>(ty)];
    for (size_t i = 0; i < here.size(); ++i) {
        s->rows[i] += here[i];
        s->max_rows[i] = std::max(s->max_rows[i], here[i]);
    }
}

static void print_stats(const featurize::FeatureSpec &spec, const Stats &s) {
    const double d = s.decisions ? static_cast<double>(s.decisions) : 1.0;
    std::printf("RMFEAT_STATS: decisions=%zu tokens_mean=%.1f tokens_max=%zu relations_mean=%.1f actions_mean=%.1f\n",
        s.decisions, static_cast<double>(s.tokens) / d, s.max_tokens, static_cast<double>(s.relations) / d,
        static_cast<double>(s.actions) / d);
    for (size_t i = 0; i < spec.types.size(); ++i)
        if (s.rows[i])
            std::printf("RMFEAT_STATS: type %s rows_mean=%.2f rows_max=%zu\n", spec.types[i].name.c_str(),
                static_cast<double>(s.rows[i]) / d, s.max_rows[i]);
    std::printf("RMFEAT_STATS: featurize_s=%.3f decode_s=%.3f featurize_per_s=%.0f end_to_end_per_s=%.0f\n",
        s.featurize_s, s.total_s - s.featurize_s, s.featurize_s > 0 ? d / s.featurize_s : 0.0,
        s.total_s > 0 ? d / s.total_s : 0.0);
}

// --featurize (write: the tensors file) and --stats.
static int run_featurize(const Options &opt, const featurize::Featurization &f, const std::string &dump, bool write) {
    std::vector<featurize::Tensors> list;
    Stats stats;
    stats.rows.assign(f.spec.types.size(), 0);
    stats.max_rows.assign(f.spec.types.size(), 0);
    featurize::Tensors t;
    featurize::EventHistory history[2];  // per seat
    const auto t0 = std::chrono::steady_clock::now();
    std::string err;
    const bool ok = featurize::read_stream(
        dump,
        [&](const featurize::ViewerFrames &, const rmstate::DecodedFrame &state, std::string *e) {
            const bool seat = state.header.viewer != static_cast<uint8_t>(rmstate::Viewer::OMNISCIENT);
            if (seat) history[state.header.viewer == static_cast<uint8_t>(rmstate::Viewer::B)].observe(f.spec, state);
            if ((opt.viewer >= 0 && state.header.viewer != opt.viewer) || menu_rows(f.spec, state) == 0) return true;
            const auto t1 = std::chrono::steady_clock::now();
            const featurize::EventHistory *h =
                seat ? &history[state.header.viewer == static_cast<uint8_t>(rmstate::Viewer::B)] : nullptr;
            if (!(seat ? featurize::featurize_seat(f.spec, f.tables, state, h, &t, e)
                       : featurize::featurize_viewer(f.spec, f.tables, state, &t, e)))
                return false;
            stats.featurize_s += seconds_since(t1);
            count(t, &stats);
            if (write) list.push_back(t);
            return true;
        },
        &err);
    stats.total_s = seconds_since(t0);
    if (!ok) {
        std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
        return 1;
    }
    if (write && !featurize::rmtb_write_file(opt.out, featurize::tensors_file(f.spec, list), &err)) {
        std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
        return 2;
    }
    if (write) std::printf("RMFEAT: wrote %s (%zu decisions)\n", opt.out.c_str(), list.size());
    if (opt.stats) print_stats(f.spec, stats);
    return 0;
}

static int run_targets(const Options &opt, const featurize::Featurization &f) {
    std::vector<featurize::Targets> list;
    rmstate::DecodedFrame seat_frame;
    bool pending = false;
    std::string err;
    const bool ok = featurize::read_stream(
        opt.targets,
        [&](const featurize::ViewerFrames &v, const rmstate::DecodedFrame &state, std::string *e) {
            if (state.header.viewer == opt.viewer) {
                pending = menu_rows(f.spec, state) > 0;
                if (pending) seat_frame = state;
                return true;
            }
            if (state.header.viewer != static_cast<uint8_t>(rmstate::Viewer::OMNISCIENT) || !pending) return true;
            pending = false;
            list.emplace_back();
            return featurize::decision_targets(f.spec, v.schema, seat_frame, state, &list.back(), e);
        },
        &err);
    if (!ok) {
        std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
        return 1;
    }
    if (!featurize::rmtb_write_file(opt.out, featurize::targets_file(f.spec, list), &err)) {
        std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
        return 2;
    }
    std::printf("RMFEAT: wrote %s (%zu decisions)\n", opt.out.c_str(), list.size());
    return 0;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return usage();
    featurize::Featurization f;
    if (!load(opt, &f)) return 2;
    if (opt.spec) {
        std::printf("%s\n", f.spec.json.c_str());
        return 0;
    }
    if (!opt.export_dir.empty()) {
        std::string err;
        if (!f.export_tables(opt.export_dir, &err)) {
            std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
            return 2;
        }
        std::printf("RMFEAT: card [%llu, %zu], ability [%zu, %zu], part [%zu, %zu], spec %s in %s\n",
            static_cast<unsigned long long>(f.tables.card_rows), f.tables.card_columns.size(), f.tables.ability_rows(),
            f.tables.ability_columns.size(), f.tables.part_rows(), f.tables.part_columns.size(),
            featurize::hex16(f.spec.hash).c_str(), opt.export_dir.c_str());
        return 0;
    }
    if (!opt.targets.empty()) return run_targets(opt, f);
    if (!opt.stats_dump.empty()) return run_featurize(opt, f, opt.stats_dump, false);
    return run_featurize(opt, f, opt.featurize, !opt.out.empty());
}
