#include "cli.h"

#include <signal.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "gate.h"
#include "util.h"

struct TierEntry {
        const char *name;
        void (*run)(const rmcheck::Options &, rmcheck::Report &);
};

static const TierEntry TIERS[] = {
    {"vocab", rmcheck::tier_vocab},
    {"concede", rmcheck::tier_concede},
    {"scenarios", rmcheck::tier_scenarios},
    {"replay", rmcheck::tier_replay},
    {"state", rmcheck::tier_state},
    {"knowledge", rmcheck::tier_knowledge},
    {"fuzz", rmcheck::tier_fuzz},
    {"bench", rmcheck::tier_bench},
    {"smoke", rmcheck::tier_smoke},
};

static std::string self_dir();
static std::string find_python();
static const TierEntry *find_tier(const std::string &name);
static std::string all_tier_names();
static void usage();
static bool parse_int(const char *s, long *out);

// The directory of this executable (bin/<config>/).
static std::string self_dir() {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n <= 0) return ".";
    buf[n] = '\0';
    return rmcheck::dirname_of(buf);
}

// The interpreter for the smoke tier (the kept Python is stdlib only): the first python3 on $PATH,
// else a system one.
static std::string find_python() {
    std::vector<std::string> dirs;
    if (const char *path = std::getenv("PATH"))
        for (const std::string &d : rmcheck::split(path, ':'))
            if (!d.empty()) dirs.push_back(d);
    for (const char *d : {"/usr/bin", "/usr/local/bin", "/bin"}) dirs.push_back(d);
    for (const std::string &dir : dirs) {
        const std::string p = dir + "/python3";
        if (access(p.c_str(), X_OK) == 0) return p;
    }
    return "";
}

static const TierEntry *find_tier(const std::string &name) {
    for (const TierEntry &t : TIERS)
        if (name == t.name) return &t;
    return nullptr;
}

static std::string all_tier_names() {
    std::vector<std::string> names;
    for (const TierEntry &t : TIERS) names.push_back(t.name);
    return rmcheck::join(names, ",");
}

static void usage() {
    std::printf(
        "usage: robomage_check [--tier a,b,...] [-j N] [--seed N] [--binary PATH] [--out-dir DIR]\n"
        "                      [--fuzz-games N] [--smoke-games N] [--matchups PRESET] [--mode MODE]\n"
        "                      [--scenario GLOB ...] [--show output|narrative|dump] [--record]\n"
        "tiers: %s\n",
        all_tier_names().c_str());
}

static bool parse_int(const char *s, long *out) {
    char *end = nullptr;
    *out = std::strtol(s, &end, 10);
    return end != s && *end == '\0';
}

namespace rmcheck {

int gate_main(int argc, char **argv) {
    signal(SIGPIPE, SIG_IGN);
    rmcheck::Options opt;
    const std::string here = self_dir();
    opt.repo = rmcheck::dirname_of(rmcheck::dirname_of(here));
    opt.bin_dir = opt.repo + "/bin";
    opt.binary = here + "/robomage";
    opt.out_dir = opt.repo + "/ci_out";
    const unsigned hw = std::thread::hardware_concurrency();
    opt.jobs = hw ? static_cast<int>(hw) : 4;
    std::string tiers_arg = all_tier_names();
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        long n = 0;
        if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else if (a == "--tier" && has_value) {
            tiers_arg = argv[++i];
        } else if ((a == "-j" || a == "--jobs") && has_value && parse_int(argv[i + 1], &n) && n > 0) {
            opt.jobs = static_cast<int>(n);
            ++i;
        } else if (a == "--seed" && has_value && parse_int(argv[i + 1], &n)) {
            opt.seed = n;
            ++i;
        } else if (a == "--binary" && has_value) {
            opt.binary = argv[++i];
        } else if (a == "--out-dir" && has_value) {
            opt.out_dir = argv[++i];
        } else if (a == "--fuzz-games" && has_value && parse_int(argv[i + 1], &n) && n >= 0) {
            opt.fuzz_games = static_cast<int>(n);
            ++i;
        } else if (a == "--smoke-games" && has_value && parse_int(argv[i + 1], &n) && n >= 0) {
            opt.smoke_games = static_cast<int>(n);
            ++i;
        } else if (a == "--matchups" && has_value) {
            opt.matchups = argv[++i];
        } else if (a == "--mode" && has_value) {
            opt.mode = argv[++i];
        } else if (a == "--scenario" && has_value) {
            opt.scenario_patterns.push_back(argv[++i]);
        } else if (a == "--show" && has_value) {
            opt.show = argv[++i];
        } else if (a == "--record") {
            opt.record = true;
        } else {
            std::fprintf(stderr, "robomage_check: bad argument '%s'\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (opt.binary[0] != '/') {
        char cwd[4096];
        if (getcwd(cwd, sizeof cwd)) opt.binary = rmcheck::path_join(cwd, opt.binary);
    }
    if (opt.out_dir[0] != '/') {
        char cwd[4096];
        if (getcwd(cwd, sizeof cwd)) opt.out_dir = rmcheck::path_join(cwd, opt.out_dir);
    }
    opt.rmknow = rmcheck::path_join(rmcheck::dirname_of(opt.binary), "rmknow");
    opt.python = find_python();
    std::vector<const TierEntry *> tiers;
    for (const std::string &name : rmcheck::split(tiers_arg, ',')) {
        const std::string t = rmcheck::trim(name);
        if (t.empty()) continue;
        const TierEntry *e = find_tier(t);
        if (!e) {
            std::fprintf(stderr, "robomage_check: unknown tier '%s'; valid: %s\n", t.c_str(), all_tier_names().c_str());
            return 2;
        }
        tiers.push_back(e);
    }
    if (access(opt.binary.c_str(), X_OK) != 0) {
        std::fprintf(stderr, "robomage_check: engine binary not found at %s (run `make`)\n", opt.binary.c_str());
        return 2;
    }
    bool have_scripts = false;
    const std::string cards = opt.bin_dir + "/resources/cardsfolder";
    for (const std::string &letter : rmcheck::list_dir(cards, ""))
        have_scripts = have_scripts || !rmcheck::list_dir(cards + "/" + letter, ".txt").empty();
    if (!have_scripts) {
        std::fprintf(stderr,
            "robomage_check: no card scripts under %s/resources/cardsfolder (run "
            "tools/forge_fetch/provision_decks.py)\n",
            opt.bin_dir.c_str());
        return 2;
    }
    rmcheck::make_dirs(opt.out_dir);
    std::printf("robomage_check: engine %s, %d job(s), seed %ld, output in %s\n", opt.binary.c_str(), opt.jobs,
        opt.seed, opt.out_dir.c_str());

    rmcheck::Report rep;
    std::vector<std::pair<std::string, double>> times;
    const double t_start = rmcheck::now_seconds();
    for (const TierEntry *t : tiers) {
        std::printf("\n=== tier: %s ===\n", t->name);
        std::fflush(stdout);
        const double t0 = rmcheck::now_seconds();
        const size_t errors_before = rep.error_count();
        t->run(opt, rep);
        times.emplace_back(t->name, rmcheck::now_seconds() - t0);
        std::printf(
            "  (%s: %s, %.1fs)\n", t->name, rep.error_count() == errors_before ? "PASS" : "FAIL", times.back().second);
        std::fflush(stdout);
    }
    const double total = rmcheck::now_seconds() - t_start;

    std::printf("\n============================================================\ntier wall times:\n");
    for (const auto &t : times) {
        size_t errs = 0;
        for (const auto &e : rep.errors()) errs += e.first == t.first ? 1 : 0;
        std::printf("  %-12s %7.1fs  %s\n", t.first.c_str(), t.second, errs ? "FAIL" : "PASS");
    }
    std::printf("  %-12s %7.1fs\n", "total", total);
    std::printf("\n============================================================\n");
    std::printf("robomage_check: %zu error(s), %zu warning(s)\n", rep.errors().size(), rep.warnings().size());
    for (const auto &w : rep.warnings())
        std::printf("  WARNING [%s]: %s\n", w.first.c_str(), w.second.substr(0, w.second.find('\n')).c_str());
    for (const auto &e : rep.errors())
        std::printf("  ERROR   [%s]: %s\n", e.first.c_str(), e.second.substr(0, e.second.find('\n')).c_str());
    std::vector<std::string> names;
    for (const TierEntry *t : tiers) names.push_back(t->name);
    std::printf("\nReproduce: %s --tier %s --seed %ld --mode %s%s%s\n", argv[0], rmcheck::join(names, ",").c_str(),
        opt.seed, opt.mode.c_str(), opt.matchups.empty() ? "" : " --matchups ", opt.matchups.c_str());
    return rep.errors().empty() ? 0 : 1;
}

}  // namespace rmcheck
