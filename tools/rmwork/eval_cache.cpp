#include "eval_cache.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "files.h"
#include "model_zip.h"
#include "replay_cache.h"

namespace {

const char MAGIC[4] = {'R', 'M', 'E', 'V'};
const uint32_t FORMAT_VERSION = 1;
// A `.tmp` file older than this belongs to no write in progress.
const int64_t STALE_TMP_NS = 600ll * 1000000000ll;

struct Entry {
        std::string path;
        int64_t mtime_ns = 0;
        uint64_t bytes = 0;
};

// Reads the fixed-size values of an entry in order.
struct Reader {
        const std::string &b;
        size_t at = 0;
        bool ok = true;
        bool take(void *dst, size_t n) {
            if (!ok || b.size() - at < n) return ok = false;
            std::memcpy(dst, b.data() + at, n);
            at += n;
            return true;
        }
};

}  // namespace

static std::string hex64(uint64_t v);
static std::string resolved(const std::string &path);
static void put(std::string *out, const void *src, size_t n);
static int64_t now_ns();

static std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

static std::string resolved(const std::string &path) {
    char buf[PATH_MAX];
    return realpath(path.c_str(), buf) ? std::string(buf) : path;
}

static void put(std::string *out, const void *src, size_t n) {
    out->append(static_cast<const char *>(src), n);
}

static int64_t now_ns() {
    timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return static_cast<int64_t>(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
}

namespace rmwork {

std::string eval_cache_root() {
    const char *xdg = std::getenv("XDG_CACHE_HOME");
    const char *home = std::getenv("HOME");
    const std::string base = xdg && *xdg ? std::string(xdg) : std::string(home && *home ? home : ".") + "/.cache";
    return base + "/robomage/workbench/evaluations";
}

std::string model_identity(const std::string &model_ref) {
    std::string zip, member;
    const bool zipped = split_zip_model(model_ref, &zip, &member);
    const std::string file = resolved(zipped ? zip : model_ref);
    const FileStamp st = file_stamp(file);
    if (!st.found || !st.regular) return "missing\n" + model_ref;
    std::string id = file + '\n' + std::to_string(st.size) + '\n' + std::to_string(st.mtime_ns);
    if (zipped) id += '\n' + member;
    return id;
}

std::string eval_cache_path(const std::string &root, uint64_t log_hash, const std::string &identity, int seat) {
    return root + "/" + hex64(log_hash) + "-" + hex64(fnv1a64(identity)) + "-" + (seat == 0 ? "a" : "b") + ".rmev";
}

std::string eval_cache_bytes(const CachedEvals &c) {
    std::string out;
    put(&out, MAGIC, 4);
    put(&out, &FORMAT_VERSION, 4);
    put(&out, &c.log_hash, 8);
    const uint32_t id_len = static_cast<uint32_t>(c.identity.size());
    put(&out, &id_len, 4);
    out += c.identity;
    const uint8_t seat = static_cast<uint8_t>(c.seat);
    put(&out, &seat, 1);
    put(&out, &c.policy, 4);
    put(&out, &c.version, 8);
    const uint32_t n = static_cast<uint32_t>(c.decisions.size());
    put(&out, &n, 4);
    for (const CachedDecision &d : c.decisions) {
        put(&out, &d.index, 4);
        put(&out, &d.game, 4);
        const uint8_t sb = d.sideboarding ? 1 : 0;
        put(&out, &sb, 1);
        put(&out, &d.decision, 8);
        put(&out, &d.value, 4);
        const uint32_t rows = static_cast<uint32_t>(d.log_pi.size());
        put(&out, &rows, 4);
        if (rows) put(&out, d.log_pi.data(), 4 * static_cast<size_t>(rows));
    }
    const uint64_t h = fnv1a64(out);
    put(&out, &h, 8);
    return out;
}

bool parse_eval_cache(const std::string &bytes, CachedEvals *out, std::string *err) {
    if (bytes.size() < 16) {
        *err = "too short for an entry";
        return false;
    }
    uint64_t stored = 0;
    std::memcpy(&stored, bytes.data() + bytes.size() - 8, 8);
    if (fnv1a64(bytes.substr(0, bytes.size() - 8)) != stored) {
        *err = "its hash does not match (damaged or truncated)";
        return false;
    }
    const std::string body = bytes.substr(0, bytes.size() - 8);
    Reader r{body};
    char magic[4];
    uint32_t version = 0, id_len = 0, n = 0;
    r.take(magic, 4);
    r.take(&version, 4);
    if (!r.ok || std::memcmp(magic, MAGIC, 4) != 0 || version != FORMAT_VERSION) {
        *err = "not a re-evaluation entry of format " + std::to_string(FORMAT_VERSION);
        return false;
    }
    CachedEvals c;
    r.take(&c.log_hash, 8);
    r.take(&id_len, 4);
    if (!r.ok || body.size() - r.at < id_len) {
        *err = "truncated";
        return false;
    }
    c.identity = body.substr(r.at, id_len);
    r.at += id_len;
    uint8_t seat = 0;
    r.take(&seat, 1);
    c.seat = seat;
    r.take(&c.policy, 4);
    r.take(&c.version, 8);
    r.take(&n, 4);
    for (uint32_t i = 0; r.ok && i < n; ++i) {
        CachedDecision d;
        uint8_t sb = 0;
        uint32_t rows = 0;
        r.take(&d.index, 4);
        r.take(&d.game, 4);
        r.take(&sb, 1);
        r.take(&d.decision, 8);
        r.take(&d.value, 4);
        r.take(&rows, 4);
        d.sideboarding = sb != 0;
        if (!r.ok || (body.size() - r.at) / 4 < rows) {
            r.ok = false;
            break;
        }
        d.log_pi.resize(rows);
        r.take(d.log_pi.data(), 4 * static_cast<size_t>(rows));
        c.decisions.push_back(std::move(d));
    }
    if (!r.ok || r.at != body.size() || seat > 1) {
        *err = "malformed";
        return false;
    }
    *out = std::move(c);
    return true;
}

bool write_eval_cache(const std::string &path, const CachedEvals &c, std::string *err) {
    if (!make_dirs(parent_dir(path))) {
        *err = "cannot make " + parent_dir(path);
        return false;
    }
    return write_file_atomic(path, eval_cache_bytes(c), err);
}

bool read_eval_cache(const std::string &path, CachedEvals *out, std::string *err) {
    std::string bytes;
    if (!read_file(path, &bytes)) {
        *err = "no entry";
        return false;
    }
    return parse_eval_cache(bytes, out, err);
}

bool eval_cache_fits(const CachedEvals &c, const Match &m, int seat, std::string *why) {
    if (c.log_hash != m.log_hash || c.seat != seat) {
        *why = "of another log or seat";
        return false;
    }
    size_t k = 0;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const DecisionInfo &d = m.decisions[i];
        if (d.decider != seat) continue;
        if (k >= c.decisions.size()) {
            *why = "fewer decisions than the match's";
            return false;
        }
        const CachedDecision &e = c.decisions[k++];
        if (e.index != i || e.game != d.game || e.sideboarding != d.sideboarding || e.decision != d.decision ||
            static_cast<int>(e.log_pi.size()) != d.menu_rows) {
            *why = "decision " + std::to_string(i + 1) + " is not the match's";
            return false;
        }
    }
    if (k != c.decisions.size()) {
        *why = "more decisions than the match's";
        return false;
    }
    return true;
}

void prune_eval_cache(const std::string &root, uint64_t max_bytes, const std::string &keep) {
    DIR *d = opendir(root.c_str());
    if (!d) return;
    std::vector<Entry> entries;
    uint64_t total = 0;
    const int64_t now = now_ns();
    while (const dirent *e = readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        const std::string path = root + "/" + name;
        const FileStamp st = file_stamp(path);
        if (!st.found || !st.regular) continue;
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            if (now - st.mtime_ns > STALE_TMP_NS) unlink(path.c_str());
            continue;
        }
        total += st.size;
        if (path != keep) entries.push_back({path, st.mtime_ns, st.size});
    }
    closedir(d);
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) { return a.mtime_ns < b.mtime_ns; });
    for (const Entry &en : entries) {
        if (total <= max_bytes) break;
        unlink(en.path.c_str());
        total -= std::min(total, en.bytes);
    }
}

}  // namespace rmwork
