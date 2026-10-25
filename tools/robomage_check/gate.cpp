#include "gate.h"

#include <cstdio>

#include "util.h"

namespace rmcheck {

std::string Options::out(const std::string &sub) const {
    const std::string dir = sub.empty() ? out_dir : path_join(out_dir, sub);
    make_dirs(dir);
    return dir;
}

void Report::error(const std::string &tier, const std::string &msg) {
    std::lock_guard<std::mutex> lock(mu_);
    errors_.emplace_back(tier, msg);
    std::printf("  [ERROR] %s: %s\n", tier.c_str(), msg.c_str());
    std::fflush(stdout);
}

void Report::warn(const std::string &tier, const std::string &msg) {
    std::lock_guard<std::mutex> lock(mu_);
    warnings_.emplace_back(tier, msg);
    std::printf("  [WARN]  %s: %s\n", tier.c_str(), msg.c_str());
    std::fflush(stdout);
}

void Report::info(const std::string &msg) {
    std::lock_guard<std::mutex> lock(mu_);
    std::printf("  %s\n", msg.c_str());
    std::fflush(stdout);
}

size_t Report::error_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return errors_.size();
}

std::vector<std::string> playable_decks(const Options &opt) {
    std::vector<std::string> out;
    for (const char *dir : {"league", "heldout"})
        for (const std::string &name : list_dir(path_join(opt.decks_dir(), dir), ".dk"))
            out.push_back(std::string(dir) + "/" + name.substr(0, name.size() - 3));
    return out;
}

bool matchups_explicit(const std::string &preset) {
    return preset != "mirrors" && preset != "ring" && preset != "mirrors+ring" && preset != "all";
}

std::vector<std::pair<std::string, std::string>> matchups(const Options &opt, const std::string &preset) {
    const std::vector<std::string> specs = playable_decks(opt);
    const size_t n = specs.size();
    std::vector<std::pair<std::string, std::string>> mirrors, ring, out;
    for (size_t i = 0; i < n; ++i) {
        mirrors.emplace_back(specs[i], specs[i]);
        ring.emplace_back(specs[i], specs[(i + 1) % n]);
    }
    if (preset == "mirrors") return mirrors;
    if (preset == "ring") return ring;
    if (preset == "mirrors+ring") {
        out = mirrors;
        out.insert(out.end(), ring.begin(), ring.end());
        return out;
    }
    if (preset == "all") {
        for (const std::string &a : specs)
            for (const std::string &b : specs) out.emplace_back(a, b);
        return out;
    }
    for (const std::string &item : split(preset, ',')) {
        const std::string t = trim(item);
        if (t.empty()) continue;
        const size_t colon = t.find(':');
        const std::string a = trim(t.substr(0, colon));
        std::string b = colon == std::string::npos ? a : trim(t.substr(colon + 1));
        if (b.empty()) b = a;
        if (b == "*") {
            for (const std::string &other : specs) out.emplace_back(a, other);
            continue;
        }
        out.emplace_back(a, b);
    }
    return out;
}

std::string deck_stem(const std::string &spec) {
    return basename_of(spec);
}

}  // namespace rmcheck
