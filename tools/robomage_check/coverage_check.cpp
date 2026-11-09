#include "coverage_check.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>

#include "proc.h"
#include "util.h"

// `<kind>:<name>` to its hits.
using CoverageCounts = std::map<std::string, uint64_t>;

static const char *const FIXTURE_DIR = "tests";
static const char *const FIXTURE_PREFIX = "mech_";

static bool is_fixture_spec(const std::string &spec);
static bool add_coverage_file(const std::string &path, CoverageCounts *counts);
static bool read_expect(const std::string &path, std::vector<std::string> *keys, std::string *err);
static std::string render_counts(const CoverageCounts &counts);

static bool is_fixture_spec(const std::string &spec) {
    return rmcheck::starts_with(spec, (std::string(FIXTURE_DIR) + "/" + FIXTURE_PREFIX).c_str());
}

// Adds a --coverage file's `<kind> <name> <count>` lines; false when it cannot be read or a line
// has another shape.
static bool add_coverage_file(const std::string &path, CoverageCounts *counts) {
    std::string text;
    if (!rmcheck::read_file(path, &text)) return false;
    for (const std::string &line : rmcheck::split_lines(text)) {
        const std::vector<std::string> parts = rmcheck::split(line, ' ');
        if (parts.size() != 3 || parts[2].empty()) return false;
        (*counts)[parts[0] + ":" + parts[1]] += std::strtoull(parts[2].c_str(), nullptr, 10);
    }
    return true;
}

// An .expect file's keys in file order. False with `err` when it cannot be read, a line is not a
// `<kind>:<name>` key, or it lists none.
static bool read_expect(const std::string &path, std::vector<std::string> *keys, std::string *err) {
    std::string text;
    if (!rmcheck::read_file(path, &text)) {
        *err = "no " + path;
        return false;
    }
    for (const std::string &raw : rmcheck::split_lines(text)) {
        const std::string line = rmcheck::trim(raw.substr(0, raw.find('#')));
        if (line.empty()) continue;
        const size_t colon = line.find(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 == line.size() ||
            line.find(' ') != std::string::npos) {
            *err = path + ": " + rmcheck::quoted(line) + " is not a <kind>:<name> key";
            return false;
        }
        keys->push_back(line);
    }
    if (keys->empty()) *err = path + " lists no key";
    return !keys->empty();
}

// The counts as the engine's --coverage writes them.
static std::string render_counts(const CoverageCounts &counts) {
    std::string text;
    for (const auto &[key, count] : counts) {
        std::string line = key;
        line[line.find(':')] = ' ';
        text += line + " " + std::to_string(count) + "\n";
    }
    return text;
}

namespace rmcheck {

long fixture_seed_slot(const std::string &fixture) {
    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV-1a
    for (unsigned char c : fixture) hash = (hash ^ c) * 0x100000001b3ULL;
    return static_cast<long>(hash % 1000);
}

std::vector<std::string> fuzz_fixtures(const Options &opt) {
    std::vector<std::string> out;
    if (!opt.matchups.empty() && matchups_explicit(opt.matchups)) {
        for (const auto &[a, b] : matchups(opt, opt.matchups))
            for (const std::string &spec : {a, b})
                if (is_fixture_spec(spec) && std::find(out.begin(), out.end(), spec) == out.end()) out.push_back(spec);
        return out;
    }
    const std::string dir = path_join(opt.decks_dir(), FIXTURE_DIR);
    for (const char *suffix : {".dk", ".expect"})
        for (const std::string &name : list_dir(dir, suffix)) {
            const std::string spec = std::string(FIXTURE_DIR) + "/" + strip_ext(name);
            if (is_fixture_spec(spec) && std::find(out.begin(), out.end(), spec) == out.end()) out.push_back(spec);
        }
    std::sort(out.begin(), out.end());
    return out;
}

void check_fixture_coverage(
    const Options &opt, Report &rep, const std::string &fixture, const std::vector<std::string> &coverage_files) {
    ProcSpec check;
    check.argv = {opt.binary, "--check-decks", "--check-decks-dir", fixture + ".dk"};
    check.cwd = opt.bin_dir;
    const ProcResult checked = run_process(check);
    if (!checked.ok()) rep.error("fuzz", fixture + ": --check-decks failed:\n" + checked.out + checked.err);

    CoverageCounts counts;
    for (const std::string &path : coverage_files)
        if (!add_coverage_file(path, &counts)) rep.error("fuzz", fixture + ": no coverage from " + path);
    const std::string summed = path_join(opt.out("coverage"), deck_stem(fixture) + ".txt");
    write_file(summed, render_counts(counts));

    std::vector<std::string> keys;
    std::string err;
    if (!read_expect(path_join(opt.decks_dir(), fixture + ".expect"), &keys, &err)) {
        rep.error("fuzz", fixture + ": " + err);
        return;
    }
    size_t hit = 0;
    std::string rarest;
    for (const std::string &key : keys) {
        const auto it = counts.find(key);
        if (it == counts.end() || it->second == 0) {
            rep.error("fuzz", fixture + ": .expect key " + key + " has no hits over " +
                                  std::to_string(coverage_files.size()) + " games (all counts: " + summed + ")");
            continue;
        }
        ++hit;
        if (rarest.empty() || it->second < counts[rarest]) rarest = key;
    }
    rep.info("fuzz: " + fixture + ": " + std::to_string(hit) + "/" + std::to_string(keys.size()) +
             " expected keys hit over " + std::to_string(coverage_files.size()) + " games" +
             (rarest.empty() ? "" : ", rarest " + rarest + " (" + std::to_string(counts[rarest]) + ")") + " -> " +
             summed);
}

}  // namespace rmcheck
