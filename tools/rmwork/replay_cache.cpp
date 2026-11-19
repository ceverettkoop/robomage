#include "replay_cache.h"

#include <dirent.h>
#include <signal.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../src/featurize/json.h"
#include "files.h"

namespace {

struct Entry {
        std::string dir;
        int64_t mtime_ns = 0;
        uint64_t bytes = 0;
};

}  // namespace

static std::string hex64(uint64_t v);
static uint64_t tree_bytes(const std::string &path);
static bool stale_tmp(const std::string &name);

static std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

static uint64_t tree_bytes(const std::string &path) {
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) return 0;
    if (!S_ISDIR(st.st_mode)) return static_cast<uint64_t>(st.st_size);
    uint64_t total = 0;
    if (DIR *d = opendir(path.c_str())) {
        while (const dirent *e = readdir(d)) {
            const std::string name = e->d_name;
            if (name != "." && name != "..") total += tree_bytes(path + "/" + name);
        }
        closedir(d);
    }
    return total;
}

// "<key>.tmp-<pid>" of a process that no longer runs.
static bool stale_tmp(const std::string &name) {
    const size_t at = name.rfind(".tmp-");
    if (at == std::string::npos) return false;
    const long pid = std::strtol(name.c_str() + at + 5, nullptr, 10);
    return pid > 0 && kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH;
}

namespace rmwork {

EngineIdentity engine_identity(const std::string &binary) {
    EngineIdentity id;
    char buf[PATH_MAX];
    id.path = realpath(binary.c_str(), buf) ? std::string(buf) : binary;
    const FileStamp st = file_stamp(id.path);
    if (!st.found) return id;
    id.found = st.regular;
    id.size = st.size;
    id.mtime_ns = st.mtime_ns;
    return id;
}

std::string replay_cache_root() {
    const char *xdg = std::getenv("XDG_CACHE_HOME");
    const char *home = std::getenv("HOME");
    const std::string base = xdg && *xdg ? std::string(xdg) : std::string(home && *home ? home : ".") + "/.cache";
    return base + "/robomage/workbench/replays";
}

uint64_t fnv1a64(const std::string &data, uint64_t h) {
    for (unsigned char c : data) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return h;
}

std::string replay_cache_key(const std::string &log_bytes, const EngineIdentity &engine) {
    const std::string id = engine.path + '\n' + std::to_string(engine.size) + '\n' + std::to_string(engine.mtime_ns);
    return hex64(fnv1a64(log_bytes)) + "-" + hex64(fnv1a64(id));
}

CacheEntry cache_entry(const std::string &root, const std::string &key) {
    CacheEntry e;
    e.dir = root + "/" + key;
    e.dump = e.dir + "/dump.rmst";
    e.narrative = e.dir + "/narrative.txt";
    e.stderr_path = e.dir + "/stderr.txt";
    e.entry_json = e.dir + "/entry.json";
    e.index = e.dir + "/index.bin";
    return e;
}

bool cache_entry_complete(const CacheEntry &e) {
    return path_kind(e.entry_json) == PathKind::FILE && path_kind(e.dump) == PathKind::FILE &&
           path_kind(e.narrative) == PathKind::FILE;
}

bool write_cache_entry(const CacheEntry &e, const std::string &log_path, const EngineIdentity &engine, size_t decisions,
    std::string *err) {
    using featurize::json_quote;
    const std::string json = "{\"log\": " + json_quote(log_path) + ", \"engine\": " + json_quote(engine.path) +
                             ", \"engine_size\": " + std::to_string(engine.size) +
                             ", \"engine_mtime_ns\": " + std::to_string(engine.mtime_ns) +
                             ", \"decisions\": " + std::to_string(decisions) + "}\n";
    return write_file_atomic(e.entry_json, json, err);
}

void prune_replay_cache(const std::string &root, uint64_t max_bytes, const std::string &keep) {
    std::vector<Entry> entries;
    uint64_t total = 0;
    DIR *d = opendir(root.c_str());
    if (!d) return;
    std::vector<std::string> names;
    while (const dirent *e = readdir(d)) {
        const std::string name = e->d_name;
        if (name != "." && name != "..") names.push_back(name);
    }
    closedir(d);
    for (const std::string &name : names) {
        const std::string dir = root + "/" + name;
        if (stale_tmp(name)) {
            remove_tree(dir);
            continue;
        }
        if (name.find(".tmp-") != std::string::npos) continue;
        Entry en;
        en.dir = dir;
        en.bytes = tree_bytes(dir);
        total += en.bytes;
        en.mtime_ns = file_stamp(dir + "/entry.json").mtime_ns;
        if (dir != keep) entries.push_back(en);
    }
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.mtime_ns < b.mtime_ns; });
    for (const Entry &en : entries) {
        if (total <= max_bytes) break;
        remove_tree(en.dir);
        total -= std::min(total, en.bytes);
    }
}

}  // namespace rmwork
