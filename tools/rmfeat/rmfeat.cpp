// rmfeat: the featurizer over recorded RMSTATE frames (docs/featurizer.md §3-4, §6).
//
//   rmfeat --spec --cards CARDS
//   rmfeat --export-tables CARDS OUTDIR
//   rmfeat --featurize DUMP --viewer A|B|omniscient --out FILE --cards CARDS [--stats]
//   rmfeat --targets DUMP --seat A|B --out FILE --cards CARDS
//   rmfeat --stats DUMP --cards CARDS
//   rmfeat --registry-update FILE --cards CARDS
//   rmfeat --registry-check FILE --cards CARDS [--base FILE]
//   rmfeat --registry-raise FILE ENUM=CAPACITY[,ENUM=CAPACITY…]
//   rmfeat --digest-update FILE --cards CARDS
//   rmfeat --digest-check FILE --cards CARDS
//   rmfeat --table-drift OLD_TABLES NEW_TABLES
//   rmfeat --compat MODEL NOW [--exact]
//
// Every mode also takes --text-table FILE (default bin/resources/embeddings/oracle_text.rmtb, else
// resources/embeddings/oracle_text.rmtb: its row count is N_CARD_TYPES + 1 and its vocab_hash the
// vocab's) and the featurizing modes --tables DIR (read the exported card tables instead of building
// them; they must carry the spec's hash) and --history N (the spec's recent-event history tokens, at
// most N events: docs/featurizer.md §3.7; a seat's frames are fed to its event history in order).
// CARDS is robomage --cards-frame's file (SCHEMA, the whole-vocab CARDS frame), DUMP an
// --rmstate-dump file. --registry FILE lays the tables and the spec out by that layout registry
// instead of the compiled-in one (src/featurize/registry.json, docs/featurizer.md §6.1).
//
// --registry-update (`make registry`) adds to the registry FILE (a fresh one when FILE does not
// exist) every entry the cards file's schema and tables need that it lacks, at the next free
// positions, and rewrites FILE when anything was added (REGISTRY lines name each); exit 1 past a
// capacity. --registry-check (the gate's vocab tier) requires FILE to parse and be in its canonical
// form, to be the registry this binary was built with, to need no entry the cards file's schema and
// tables make, and, with --base (the committed registry), to move no entry of the base unless
// layout_version was raised. --registry-raise is a layout bump: each named enum's capacity raised
// (and the enum made open; Registry::raise_capacities: the positions after a widened block move, the
// spaces grow as needed), layout_version raised by one, FILE rewritten; then `make registry` appends
// the new values and every checkpoint is migrated by name (rmlearn.migrate, docs/rmtrain.md §1.2).
//
// --digest-update (`make registry`) writes the card digest FILE (src/featurize/card_digest.json,
// drift.h): per card the cards file describes a hash of its card, ability and part rows.
// --digest-check (the gate's vocab tier) fails (DIGEST-FAIL lines) when a known card's rows no
// longer match FILE, or a card is new or gone: a change to a known card's inputs is then made on
// purpose (`make registry`, and the digest's diff in the commit). --table-drift compares two
// exported tables directories (an old run's and today's: rmtrain retable) by card id and prints
// a TABLE-DRIFT summary and one line per changed, new or removed card (the columns that changed,
// old and new values); exit 0 either way. --compat checks the model manifest MODEL against the
// tables' manifest NOW (JSON files: compat.h's manifest_json, `registry` included when carried;
// a tables directory's manifest.json and registry.json, or a checkpoint's), exact (--exact) for an
// export: COMPAT: ok, exit 0, or COMPAT-REFUSED: <code>: <why>, exit 1.
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
#include <map>
#include <string>
#include <vector>

#include "../../src/featurize/compat.h"
#include "../../src/featurize/drift.h"
#include "../../src/featurize/featurizer.h"
#include "../../src/featurize/registry.h"
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
        std::string registry, registry_update, registry_check, registry_raise, raises, base;
        std::string digest_update, digest_check, drift_old, drift_new, compat_model, compat_now;
        bool exact = false;
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
static bool read_text(const std::string &path, std::string *out);
static bool write_text(const std::string &path, const std::string &text);
static bool load_registry(const std::string &path, featurize::Registry *out);
static bool append_pass(const Options &opt, featurize::Registry *reg);
static int run_registry_update(const Options &opt);
static int run_registry_check(const Options &opt);
static int run_registry_raise(const Options &opt);
static bool manifest_of_dir(const std::string &dir, featurize::Manifest *out);
static int run_digest(const Options &opt, const featurize::Featurization &f);
static int run_table_drift(const Options &opt);
static bool read_manifest(const std::string &path, featurize::Manifest *out);
static int run_compat(const Options &opt);

static int usage() {
    std::fprintf(stderr,
        "usage: rmfeat --spec --cards CARDS\n"
        "       rmfeat --export-tables CARDS OUTDIR\n"
        "       rmfeat --featurize DUMP --viewer A|B|omniscient --out FILE --cards CARDS [--stats]\n"
        "       rmfeat --targets DUMP --seat A|B --out FILE --cards CARDS\n"
        "       rmfeat --stats DUMP --cards CARDS\n"
        "       rmfeat --registry-update FILE --cards CARDS\n"
        "       rmfeat --registry-check FILE --cards CARDS [--base FILE]\n"
        "       rmfeat --registry-raise FILE ENUM=CAPACITY[,ENUM=CAPACITY...]\n"
        "       rmfeat --digest-update FILE --cards CARDS\n"
        "       rmfeat --digest-check FILE --cards CARDS\n"
        "       rmfeat --table-drift OLD_TABLES NEW_TABLES\n"
        "       rmfeat --compat MODEL NOW [--exact]\n"
        "  (each also --text-table FILE, --registry FILE and --history N; the featurizing modes --tables DIR)\n");
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
        } else if (a == "--registry" && has1) {
            opt->registry = argv[++i];
        } else if (a == "--registry-update" && has1) {
            opt->registry_update = argv[++i];
        } else if (a == "--registry-check" && has1) {
            opt->registry_check = argv[++i];
        } else if (a == "--registry-raise" && has2) {
            opt->registry_raise = argv[++i];
            opt->raises = argv[++i];
        } else if (a == "--base" && has1) {
            opt->base = argv[++i];
        } else if (a == "--digest-update" && has1) {
            opt->digest_update = argv[++i];
        } else if (a == "--digest-check" && has1) {
            opt->digest_check = argv[++i];
        } else if (a == "--table-drift" && has2) {
            opt->drift_old = argv[++i];
            opt->drift_new = argv[++i];
        } else if (a == "--compat" && has2) {
            opt->compat_model = argv[++i];
            opt->compat_now = argv[++i];
        } else if (a == "--exact") {
            opt->exact = true;
        } else {
            return false;
        }
    }
    if (!opt->export_cards.empty()) opt->cards = opt->export_cards;
    const int modes = (opt->spec ? 1 : 0) + (opt->export_cards.empty() ? 0 : 1) + (opt->featurize.empty() ? 0 : 1) +
                      (opt->targets.empty() ? 0 : 1) + (opt->stats_dump.empty() ? 0 : 1) +
                      (opt->registry_update.empty() ? 0 : 1) + (opt->registry_check.empty() ? 0 : 1) +
                      (opt->registry_raise.empty() ? 0 : 1) + (opt->digest_update.empty() ? 0 : 1) +
                      (opt->digest_check.empty() ? 0 : 1) + (opt->drift_old.empty() ? 0 : 1) +
                      (opt->compat_model.empty() ? 0 : 1);
    const bool no_cards = !opt->drift_old.empty() || !opt->compat_model.empty() || !opt->registry_raise.empty();
    if (modes != 1 || opt->cards.empty() != no_cards) return false;
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
    if (!opt.registry.empty() && !load_registry(opt.registry, &f->registry)) return false;
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

static bool read_text(const std::string &path, std::string *out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    out->clear();
    char buf[65536];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) out->append(buf, n);
    std::fclose(f);
    return true;
}

// `text` into `path` through a temporary file renamed into place.
static bool write_text(const std::string &path, const std::string &text) {
    const std::string tmp = path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (std::fclose(f) != 0 || !ok) return false;
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

static bool load_registry(const std::string &path, featurize::Registry *out) {
    std::string text, err;
    if (!read_text(path, &text)) {
        std::fprintf(stderr, "rmfeat: cannot read %s\n", path.c_str());
        return false;
    }
    if (featurize::Registry::parse(text, out, &err)) return true;
    std::fprintf(stderr, "rmfeat: %s: %s\n", path.c_str(), err.c_str());
    return false;
}

// The tables and a spec with the history types built over `reg` in append mode, so it gains every
// entry the cards file makes; then the new types and tables placed. False (the problems printed)
// past a capacity or on an error in the inputs.
static bool append_pass(const Options &opt, featurize::Registry *reg) {
    featurize::Featurization f;
    f.options.history_events = 1;
    f.registry = *reg;
    f.registry.append = true;
    std::string err;
    const bool ok = f.load(opt.cards, text_table_path(opt), "", &err);
    f.registry.finish();
    for (const std::string &p : f.registry.problems) std::fprintf(stderr, "rmfeat: %s\n", p.c_str());
    if (!ok && f.registry.problems.empty()) std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
    if (!ok || !f.registry.ok()) return false;
    *reg = f.registry;
    return true;
}

static int run_registry_update(const Options &opt) {
    featurize::Registry reg = featurize::Registry::fresh();
    std::string before;
    if (read_text(opt.registry_update, &before) && !load_registry(opt.registry_update, &reg)) return 2;
    if (!append_pass(opt, &reg)) return 1;
    for (const std::string &a : reg.added) std::printf("REGISTRY: added %s\n", a.c_str());
    const std::string text = reg.json();
    if (text == before) {
        std::printf("REGISTRY: %s is current\n", opt.registry_update.c_str());
        return 0;
    }
    if (!write_text(opt.registry_update, text)) {
        std::fprintf(stderr, "rmfeat: cannot write %s\n", opt.registry_update.c_str());
        return 2;
    }
    std::printf("REGISTRY: wrote %s (%zu entries added); rebuild (make) to compile it in\n",
        opt.registry_update.c_str(), reg.added.size());
    return 0;
}

static int run_registry_check(const Options &opt) {
    std::string text;
    featurize::Registry reg;
    if (!read_text(opt.registry_check, &text) || !load_registry(opt.registry_check, &reg)) return 2;
    std::vector<std::string> fails;
    if (reg.json() != text)
        fails.push_back(opt.registry_check + " is not in the canonical form `make registry` writes");
    if (text != featurize::builtin_registry_text())
        fails.push_back("this binary was built with another registry than " + opt.registry_check + " (rebuild: make)");
    featurize::Registry grown = reg;
    if (!append_pass(opt, &grown)) return 1;
    for (const std::string &a : grown.added)
        fails.push_back("the registry lacks " + a + " (run `make registry` and commit the file)");
    for (const uint32_t history : {0u, 1u}) {
        featurize::Featurization f;
        f.options.history_events = history;
        f.registry = reg;
        std::string err;
        if (grown.added.empty() && !f.load(opt.cards, text_table_path(opt), "", &err))
            fails.push_back(
                "the spec at the registry's positions" + std::string(history ? " (history)" : "") + ": " + err);
        if (history == 0 && fails.empty())
            std::printf(
                "REGISTRY: layout_version %d, layout_hash %s, spec %s; %zu enums, %zu token types, %zu "
                "numeric slots, %zu categorical slots, %zu relation types, %zu table columns, %zu keys\n",
                reg.layout_version, featurize::hex16(f.spec.layout_hash).c_str(), featurize::hex16(f.spec.hash).c_str(),
                reg.enums.size(), reg.types.size(), reg.numeric.size(), reg.categorical.size(), reg.relations.size(),
                reg.columns.size(), reg.ab_keys.size());
    }
    if (!opt.base.empty()) {
        featurize::Registry base;
        if (!load_registry(opt.base, &base)) return 2;
        if (base.layout_version > reg.layout_version) {
            fails.push_back("layout_version " + std::to_string(reg.layout_version) + " is below the base's " +
                            std::to_string(base.layout_version));
        } else if (base.layout_version < reg.layout_version) {
            std::printf("REGISTRY: layout bump %d -> %d against %s: entries may move (S5-style migration)\n",
                base.layout_version, reg.layout_version, opt.base.c_str());
        } else {
            for (const std::string &m : featurize::registry_moves(base, reg))
                fails.push_back("against " + opt.base + ": " + m + " (raise layout_version for a deliberate bump)");
            if (fails.empty()) std::printf("REGISTRY: no entry of %s moved\n", opt.base.c_str());
        }
    }
    for (const std::string &f : fails) std::printf("REGISTRY-FAIL: %s\n", f.c_str());
    if (fails.empty()) std::printf("REGISTRY: %s ok\n", opt.registry_check.c_str());
    return fails.empty() ? 0 : 1;
}

// A tables directory's manifest.json and registry.json (each optional: a directory exported before
// manifests has neither).
static int run_registry_raise(const Options &opt) {
    featurize::Registry reg;
    if (!load_registry(opt.registry_raise, &reg)) return 2;
    std::map<std::string, uint32_t> caps;
    std::string list = opt.raises;
    while (!list.empty()) {
        const size_t comma = list.find(',');
        const std::string item = list.substr(0, comma);
        list = comma == std::string::npos ? "" : list.substr(comma + 1);
        const size_t eq = item.find('=');
        char *end = nullptr;
        const unsigned long cap = eq == std::string::npos ? 0 : std::strtoul(item.c_str() + eq + 1, &end, 10);
        if (eq == std::string::npos || !end || *end || cap == 0 || cap > 0xFFFFFFFFul) {
            std::fprintf(stderr, "rmfeat: --registry-raise: %s is not ENUM=CAPACITY\n", item.c_str());
            return 2;
        }
        caps[item.substr(0, eq)] = static_cast<uint32_t>(cap);
    }
    std::map<std::string, uint32_t> before;
    for (const auto &kv : caps) {
        const featurize::RegEnum *e = reg.find_enum(kv.first);
        before[kv.first] = e ? e->capacity : 0;
    }
    std::string err;
    if (!reg.raise_capacities(caps, &err)) {
        std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
        return 1;
    }
    for (const auto &kv : caps)
        std::printf("REGISTRY: enum %s capacity %u -> %u (open)\n", kv.first.c_str(), before[kv.first], kv.second);
    ++reg.layout_version;
    const std::string text = reg.json();
    featurize::Registry back;
    if (!featurize::Registry::parse(text, &back, &err)) {
        std::fprintf(stderr, "rmfeat: the raised registry does not parse: %s\n", err.c_str());
        return 1;
    }
    if (!write_text(opt.registry_raise, text)) {
        std::fprintf(stderr, "rmfeat: cannot write %s\n", opt.registry_raise.c_str());
        return 2;
    }
    std::printf("REGISTRY: wrote %s: layout_version %d, G %u, categorical %u", opt.registry_raise.c_str(),
        reg.layout_version, reg.dims.G, reg.dims.categorical);
    for (const featurize::RegTable &t : reg.tables) std::printf(", %s %u", t.name.c_str(), t.capacity);
    std::printf("; run `make registry`, then migrate the checkpoints\n");
    return 0;
}

static bool manifest_of_dir(const std::string &dir, featurize::Manifest *out) {
    std::string text, err;
    *out = featurize::Manifest();
    if (read_text(dir + "/" + featurize::Featurization::MANIFEST_FILE, &text) &&
        !featurize::parse_manifest_json(text, out, &err)) {
        std::fprintf(stderr, "rmfeat: %s: %s\n", dir.c_str(), err.c_str());
        return false;
    }
    if (read_text(dir + "/" + featurize::Featurization::REGISTRY_FILE, &text)) out->registry = text;
    return true;
}

static int run_digest(const Options &opt, const featurize::Featurization &f) {
    const std::vector<featurize::CardDigest> now = featurize::card_digests(f.tables, f.tables.names);
    const std::string text = featurize::digest_json(now);
    if (!opt.digest_update.empty()) {
        std::string before;
        if (read_text(opt.digest_update, &before) && before == text) {
            std::printf("DIGEST: %s is current (%zu cards)\n", opt.digest_update.c_str(), now.size());
            return 0;
        }
        if (!write_text(opt.digest_update, text)) {
            std::fprintf(stderr, "rmfeat: cannot write %s\n", opt.digest_update.c_str());
            return 2;
        }
        std::printf("DIGEST: wrote %s (%zu cards)\n", opt.digest_update.c_str(), now.size());
        return 0;
    }
    std::string committed_text, err;
    std::vector<featurize::CardDigest> committed;
    if (!read_text(opt.digest_check, &committed_text) ||
        !featurize::parse_digest_json(committed_text, &committed, &err)) {
        std::fprintf(stderr, "rmfeat: %s: %s\n", opt.digest_check.c_str(), err.empty() ? "cannot read" : err.c_str());
        return 2;
    }
    const featurize::Drift d = featurize::digest_drift(committed, now);
    const char *const hint = " (if intended, `make registry` records it; commit the digest)";
    for (const std::string &c : d.changed) std::printf("DIGEST-FAIL: known %s%s\n", c.c_str(), hint);
    for (const std::string &c : d.added) std::printf("DIGEST-FAIL: %s is not in the digest%s\n", c.c_str(), hint);
    for (const std::string &c : d.removed) std::printf("DIGEST-FAIL: %s is gone%s\n", c.c_str(), hint);
    if (d.empty() && committed_text != text)
        std::printf("DIGEST-FAIL: %s is not in the canonical form `make registry` writes\n", opt.digest_check.c_str());
    const bool ok = d.empty() && committed_text == text;
    if (ok) std::printf("DIGEST: %s ok (%zu cards)\n", opt.digest_check.c_str(), now.size());
    return ok ? 0 : 1;
}

static int run_table_drift(const Options &opt) {
    featurize::CardTables before, after;
    featurize::Manifest mb, ma;
    uint64_t spec_b = 0, spec_a = 0;
    std::string err;
    if (!featurize::Featurization::read_tables_dir(opt.drift_old, &before, &spec_b, &err) ||
        !featurize::Featurization::read_tables_dir(opt.drift_new, &after, &spec_a, &err)) {
        std::fprintf(stderr, "rmfeat: %s\n", err.c_str());
        return 2;
    }
    if (!manifest_of_dir(opt.drift_old, &mb) || !manifest_of_dir(opt.drift_new, &ma)) return 2;
    std::vector<std::string> names_b(before.card_rows), names_a(after.card_rows);
    for (const auto &e : mb.vocab)
        if (e.first < names_b.size()) names_b[e.first] = e.second;
    for (const auto &e : ma.vocab)
        if (e.first < names_a.size()) names_a[e.first] = e.second;
    const featurize::Drift d = featurize::table_drift(before, names_b, after, names_a);
    std::printf("TABLE-DRIFT: %zu known cards changed, %zu new, %zu removed (spec %s -> %s)\n", d.changed.size(),
        d.added.size(), d.removed.size(), featurize::hex16(spec_b).c_str(), featurize::hex16(spec_a).c_str());
    for (const std::string &c : d.changed) std::printf("  changed %s\n", c.c_str());
    for (const std::string &c : d.added) std::printf("  new %s\n", c.c_str());
    for (const std::string &c : d.removed) std::printf("  removed %s\n", c.c_str());
    return 0;
}

static bool read_manifest(const std::string &path, featurize::Manifest *out) {
    std::string text, err;
    if (!read_text(path, &text)) {
        std::fprintf(stderr, "rmfeat: cannot read %s\n", path.c_str());
        return false;
    }
    if (featurize::parse_manifest_json(text, out, &err)) return true;
    std::fprintf(stderr, "rmfeat: %s: %s\n", path.c_str(), err.c_str());
    return false;
}

static int run_compat(const Options &opt) {
    featurize::Manifest model, now;
    if (!read_manifest(opt.compat_model, &model) || !read_manifest(opt.compat_now, &now)) return 2;
    const featurize::Refusal r = featurize::check_compatible(model, now, opt.exact);
    if (!r.refused()) {
        std::printf("COMPAT: ok\n");
        return 0;
    }
    std::printf("COMPAT-REFUSED: %s: %s\n", r.code.c_str(), r.message.c_str());
    return 1;
}

int main(int argc, char **argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return usage();
    if (!opt.registry_update.empty()) return run_registry_update(opt);
    if (!opt.registry_check.empty()) return run_registry_check(opt);
    if (!opt.registry_raise.empty()) return run_registry_raise(opt);
    if (!opt.drift_old.empty()) return run_table_drift(opt);
    if (!opt.compat_model.empty()) return run_compat(opt);
    featurize::Featurization f;
    if (!load(opt, &f)) return 2;
    if (!opt.digest_update.empty() || !opt.digest_check.empty()) return run_digest(opt, f);
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
