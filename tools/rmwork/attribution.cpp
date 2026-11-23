#include "attribution.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../src/featurize/frame_stream.h"
#include "eval_cache.h"
#include "files.h"
#include "replay_cache.h"
#include "subprocess.h"

namespace {

const char MAGIC[4] = {'R', 'M', 'A', 'T'};
const uint32_t FORMAT_VERSION = 1;
// Log π this close to the argmax's is the same choice (learn/rmlearn/ablation.py's TIE).
const double TIE = 1e-3;
// The deepest row of the tree (a group, its token, the token's cells).
const int MAX_DEPTH = 3;

// Writes an entry's values in order.
struct Writer {
        std::string out;
        void raw(const void *p, size_t n) { out.append(static_cast<const char *>(p), n); }
        template <typename T>
        void put(T v) {
            raw(&v, sizeof v);
        }
        void str(const std::string &s) {
            put<uint32_t>(static_cast<uint32_t>(s.size()));
            out += s;
        }
        void eval(const rmwork::AttrEval &e) {
            put<float>(e.value);
            put<uint32_t>(static_cast<uint32_t>(e.log_pi.size()));
            if (!e.log_pi.empty()) raw(e.log_pi.data(), 4 * e.log_pi.size());
        }
};

// Reads them back; `ok` false once anything is short.
struct Reader {
        const std::string &b;
        size_t at = 0;
        bool ok = true;
        bool raw(void *p, size_t n) {
            if (!ok || b.size() - at < n) return ok = false;
            std::memcpy(p, b.data() + at, n);
            at += n;
            return true;
        }
        template <typename T>
        T get() {
            T v{};
            raw(&v, sizeof v);
            return v;
        }
        std::string str() {
            const uint32_t n = get<uint32_t>();
            if (!ok || b.size() - at < n) {
                ok = false;
                return "";
            }
            std::string s = b.substr(at, n);
            at += n;
            return s;
        }
        rmwork::AttrEval eval() {
            rmwork::AttrEval e;
            e.value = get<float>();
            const uint32_t n = get<uint32_t>();
            if (!ok || (b.size() - at) / 4 < n) {
                ok = false;
                return e;
            }
            e.log_pi.resize(n);
            if (n) raw(e.log_pi.data(), 4 * static_cast<size_t>(n));
            return e;
        }
};

struct CacheFile {
        std::string path;
        int64_t mtime_ns = 0;
        uint64_t bytes = 0;
};

}  // namespace

static std::string hex64(uint64_t v);
static void put_unit(Writer *w, const featurize::Unit &u);
static featurize::Unit get_unit(Reader *r);
static std::vector<double> softmax(const std::vector<float> &log_pi);
static double target_scalar(const rmwork::AttrEval &e, rmwork::AttrTarget t, int focus, int rival, bool negate);
static int menu_rows_of(const rmstate::DecodedFrame &f);
static void collect_files(const std::string &dir, int depth, std::vector<CacheFile> *out, uint64_t *total);
static uint64_t mix(uint64_t x);

static std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

static void put_unit(Writer *w, const featurize::Unit &u) {
    w->put<uint8_t>(static_cast<uint8_t>(u.kind));
    w->str(u.key);
    w->put<uint32_t>(static_cast<uint32_t>(u.tokens.size()));
    for (const int64_t t : u.tokens) w->put<int64_t>(t);
    w->put<int64_t>(u.card);
    w->put<int32_t>(u.type);
    w->put<int32_t>(u.slot);
    w->put<int32_t>(u.relation);
    w->put<int32_t>(u.table);
    w->put<int64_t>(u.row);
    w->put<uint64_t>(u.handle);
    w->put<uint8_t>(u.owns ? 1 : 0);
    w->put<int32_t>(u.seat);
    w->str(u.column);
    w->str(u.group);
    w->str(u.label);
    w->str(u.tensors);
    w->put<uint8_t>(u.coarse ? 1 : 0);
}

static featurize::Unit get_unit(Reader *r) {
    featurize::Unit u;
    u.kind = static_cast<featurize::UnitKind>(r->get<uint8_t>());
    u.key = r->str();
    const uint32_t n = r->get<uint32_t>();
    if (!r->ok || (r->b.size() - r->at) / 8 < n) {
        r->ok = false;
        return u;
    }
    for (uint32_t i = 0; i < n; ++i) u.tokens.push_back(r->get<int64_t>());
    u.card = r->get<int64_t>();
    u.type = r->get<int32_t>();
    u.slot = r->get<int32_t>();
    u.relation = r->get<int32_t>();
    u.table = r->get<int32_t>();
    u.row = r->get<int64_t>();
    u.handle = r->get<uint64_t>();
    u.owns = r->get<uint8_t>() != 0;
    u.seat = r->get<int32_t>();
    u.column = r->str();
    u.group = r->str();
    u.label = r->str();
    u.tensors = r->str();
    u.coarse = r->get<uint8_t>() != 0;
    if (static_cast<int>(u.kind) > static_cast<int>(featurize::UnitKind::RELATION)) r->ok = false;
    return u;
}

static std::vector<double> softmax(const std::vector<float> &log_pi) {
    std::vector<double> p(log_pi.size());
    for (size_t i = 0; i < p.size(); ++i) p[i] = std::exp(static_cast<double>(log_pi[i]));
    return p;
}

// The scalar Shapley shares out: the value (on the chosen scale), log π(focus), the margin.
static double target_scalar(const rmwork::AttrEval &e, rmwork::AttrTarget t, int focus, int rival, bool negate) {
    const auto lp = [&e](int row) {
        return row >= 0 && static_cast<size_t>(row) < e.log_pi.size() ? static_cast<double>(e.log_pi[row]) : 0.0;
    };
    switch (t) {
        case rmwork::AttrTarget::V:
            return negate ? -e.value : e.value;
        case rmwork::AttrTarget::LOGP:
            return lp(focus);
        case rmwork::AttrTarget::MARGIN:
            return lp(focus) - lp(rival);
        case rmwork::AttrTarget::TV:
            return 0;
    }
    return 0;
}

static int menu_rows_of(const rmstate::DecodedFrame &f) {
    for (const rmstate::DecodedTable &t : f.tables)
        if (t.def && t.def->name == "menu") return static_cast<int>(t.rows.size());
    return 0;
}

// The regular files under `dir` (and its directories, `depth` levels down), `.tmp` files older than ten
// minutes removed.
static void collect_files(const std::string &dir, int depth, std::vector<CacheFile> *out, uint64_t *total) {
    DIR *d = opendir(dir.c_str());
    if (!d) return;
    std::vector<std::string> subdirs;
    while (const dirent *e = readdir(d)) {
        const std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        const std::string path = dir + "/" + name;
        const rmwork::FileStamp st = rmwork::file_stamp(path);
        if (!st.found) continue;
        if (!st.regular) {
            if (depth > 0) subdirs.push_back(path);
            continue;
        }
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            const int64_t now = static_cast<int64_t>(ts.tv_sec) * 1000000000ll + ts.tv_nsec;
            if (now - st.mtime_ns > 600ll * 1000000000ll) unlink(path.c_str());
            continue;
        }
        *total += st.size;
        out->push_back({path, st.mtime_ns, st.size});
    }
    closedir(d);
    for (const std::string &s : subdirs) collect_files(s, depth - 1, out, total);
}

static uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return x;
}

namespace rmwork {

const char *attr_target_name(AttrTarget t) {
    switch (t) {
        case AttrTarget::V:
            return "V";
        case AttrTarget::TV:
            return "\xCF\x80: TV";
        case AttrTarget::LOGP:
            return "log \xCF\x80(focus)";
        case AttrTarget::MARGIN:
            return "focus vs rival";
    }
    return "?";
}

double AttrMetrics::get(AttrTarget t) const {
    switch (t) {
        case AttrTarget::V:
            return dv;
        case AttrTarget::TV:
            return tv;
        case AttrTarget::LOGP:
            return dlogp;
        case AttrTarget::MARGIN:
            return dmargin;
    }
    return 0;
}

int attr_argmax(const std::vector<float> &log_pi) {
    int best = -1;
    for (size_t i = 0; i < log_pi.size(); ++i)
        if (best < 0 || log_pi[i] > log_pi[static_cast<size_t>(best)]) best = static_cast<int>(i);
    return best;
}

int attr_runner_up(const std::vector<float> &log_pi, int best) {
    int second = -1;
    for (size_t i = 0; i < log_pi.size(); ++i) {
        if (static_cast<int>(i) == best) continue;
        if (second < 0 || log_pi[i] > log_pi[static_cast<size_t>(second)]) second = static_cast<int>(i);
    }
    return second;
}

AttrMetrics attr_metrics(
    const AttrEval &base, const std::vector<AttrEval> &changed, int focus, int rival, bool negate) {
    AttrMetrics m;
    if (changed.empty()) return m;
    m.valid = true;
    const size_t a = base.log_pi.size();
    m.policy = a >= 1 && focus >= 0 && static_cast<size_t>(focus) < a;
    const std::vector<double> p0 = softmax(base.log_pi);
    const int top = attr_argmax(base.log_pi);
    for (const AttrEval &c : changed) {
        m.dv += static_cast<double>(c.value) - base.value;
        if (!m.policy || c.log_pi.size() != a) continue;
        const std::vector<double> p1 = softmax(c.log_pi);
        double tv = 0, kl = 0;
        for (size_t i = 0; i < a; ++i) {
            tv += std::fabs(p0[i] - p1[i]);
            const double d = static_cast<double>(base.log_pi[i]) - c.log_pi[i];
            if (std::isfinite(d)) kl += p0[i] * d;
        }
        m.tv += 0.5 * tv;
        m.kl += kl;
        const int top1 = attr_argmax(c.log_pi);
        if (base.log_pi[static_cast<size_t>(top1)] < base.log_pi[static_cast<size_t>(top)] - TIE) m.flip += 1;
        m.dlogp += static_cast<double>(c.log_pi[static_cast<size_t>(focus)]) - base.log_pi[static_cast<size_t>(focus)];
        if (rival >= 0 && static_cast<size_t>(rival) < a)
            m.dmargin +=
                (static_cast<double>(c.log_pi[static_cast<size_t>(focus)]) - c.log_pi[static_cast<size_t>(rival)]) -
                (static_cast<double>(base.log_pi[static_cast<size_t>(focus)]) -
                    base.log_pi[static_cast<size_t>(rival)]);
    }
    const double n = static_cast<double>(changed.size());
    m.dv /= n;
    if (negate) m.dv = -m.dv;
    m.tv /= n;
    m.kl /= n;
    m.flip /= n;
    m.dlogp /= n;
    m.dmargin /= n;
    return m;
}

std::string attr_cache_root() {
    const char *xdg = std::getenv("XDG_CACHE_HOME");
    const char *home = std::getenv("HOME");
    const std::string base = xdg && *xdg ? std::string(xdg) : std::string(home && *home ? home : ".") + "/.cache";
    return base + "/robomage/workbench/attribution";
}

std::string attr_cache_dir(const std::string &root, uint64_t log_hash, bool live, const std::string &log_path) {
    if (!live && log_hash) return root + "/" + hex64(log_hash);
    return root + "/live-" + hex64(fnv1a64(absolute_path(log_path)));
}

bool AttrKey::operator==(const AttrKey &o) const {
    return dir == o.dir && game == o.game && sideboarding == o.sideboarding && decision == o.decision &&
           seat == o.seat && policy == o.policy && model == o.model && baseline == o.baseline && method == o.method &&
           shapley_k == o.shapley_k;
}

std::string AttrKey::method_name() const {
    const std::string b = featurize::baseline_name(baseline);
    if (method == AttrMethod::OCCLUSION) return "occ-" + b;
    return "shap-" + b + "-k" + std::to_string(shapley_k) + "-m" + std::to_string(SHAPLEY_PERMUTATIONS);
}

std::string AttrKey::path() const {
    return dir + "/g" + std::to_string(game) + (sideboarding ? "s" : "") + "d" + std::to_string(decision) + "-" +
           (seat == 0 ? "a" : "b") + "-" + hex64(fnv1a64(model)) + "-" + method_name() + ".bin";
}

const featurize::Unit *AttrEntry::unit(const std::string &key) const {
    for (const featurize::Unit &u : units)
        if (u.key == key) return &u;
    return nullptr;
}

std::string attr_entry_bytes(const AttrEntry &e) {
    Writer w;
    w.raw(MAGIC, 4);
    w.put<uint32_t>(FORMAT_VERSION);
    w.put<int32_t>(e.key.game);
    w.put<uint8_t>(e.key.sideboarding ? 1 : 0);
    w.put<int64_t>(e.key.decision);
    w.put<uint8_t>(static_cast<uint8_t>(e.key.seat));
    w.put<uint32_t>(e.key.policy);
    w.str(e.key.model);
    w.str(e.key.method_name());
    w.put<uint64_t>(e.frame_hash);
    w.put<uint64_t>(e.spec_hash);
    w.put<uint64_t>(e.version);
    w.put<int32_t>(e.menu_rows);
    w.put<uint8_t>(e.have_base ? 1 : 0);
    w.eval(e.base);
    w.put<uint32_t>(static_cast<uint32_t>(e.units.size()));
    for (const featurize::Unit &u : e.units) {
        put_unit(&w, u);
        auto it = e.evals.find(u.key);
        const uint32_t draws = it == e.evals.end() ? 0 : static_cast<uint32_t>(it->second.size());
        w.put<uint32_t>(draws);
        for (uint32_t d = 0; d < draws; ++d) w.eval(it->second[d]);
    }
    w.put<uint32_t>(static_cast<uint32_t>(e.children.size()));
    for (const auto &kv : e.children) {
        w.str(kv.first);
        w.put<uint32_t>(static_cast<uint32_t>(kv.second.size()));
        for (const std::string &c : kv.second) w.str(c);
    }
    w.put<uint32_t>(static_cast<uint32_t>(e.players.size()));
    for (const std::string &p : e.players) w.str(p);
    w.put<uint32_t>(static_cast<uint32_t>(e.coalitions.size()));
    for (const auto &kv : e.coalitions) {
        w.put<uint32_t>(kv.first);
        w.eval(kv.second);
    }
    const uint64_t h = fnv1a64(w.out);
    w.put<uint64_t>(h);
    return w.out;
}

bool parse_attr_entry(const std::string &bytes, AttrEntry *out, std::string *err) {
    if (bytes.size() < 16) {
        *err = "too short for an entry";
        return false;
    }
    uint64_t stored = 0;
    std::memcpy(&stored, bytes.data() + bytes.size() - 8, 8);
    const std::string body = bytes.substr(0, bytes.size() - 8);
    if (fnv1a64(body) != stored) {
        *err = "its hash does not match (damaged or truncated)";
        return false;
    }
    Reader r{body};
    char magic[4];
    r.raw(magic, 4);
    if (!r.ok || std::memcmp(magic, MAGIC, 4) != 0 || r.get<uint32_t>() != FORMAT_VERSION) {
        *err = "not an attribution entry of format " + std::to_string(FORMAT_VERSION);
        return false;
    }
    AttrEntry e;
    e.key.game = r.get<int32_t>();
    e.key.sideboarding = r.get<uint8_t>() != 0;
    e.key.decision = r.get<int64_t>();
    e.key.seat = r.get<uint8_t>();
    e.key.policy = r.get<uint32_t>();
    e.key.model = r.str();
    const std::string method = r.str();
    e.frame_hash = r.get<uint64_t>();
    e.spec_hash = r.get<uint64_t>();
    e.version = r.get<uint64_t>();
    e.menu_rows = r.get<int32_t>();
    e.have_base = r.get<uint8_t>() != 0;
    e.base = r.eval();
    const uint32_t units = r.get<uint32_t>();
    for (uint32_t i = 0; r.ok && i < units; ++i) {
        featurize::Unit u = get_unit(&r);
        const uint32_t draws = r.get<uint32_t>();
        std::vector<AttrEval> evals;
        for (uint32_t d = 0; r.ok && d < draws; ++d) evals.push_back(r.eval());
        if (draws) e.evals[u.key] = std::move(evals);
        e.units.push_back(std::move(u));
    }
    const uint32_t parents = r.get<uint32_t>();
    for (uint32_t i = 0; r.ok && i < parents; ++i) {
        const std::string key = r.str();
        const uint32_t n = r.get<uint32_t>();
        std::vector<std::string> kids;
        for (uint32_t k = 0; r.ok && k < n; ++k) kids.push_back(r.str());
        e.children[key] = std::move(kids);
    }
    const uint32_t players = r.get<uint32_t>();
    for (uint32_t i = 0; r.ok && i < players; ++i) e.players.push_back(r.str());
    const uint32_t coalitions = r.get<uint32_t>();
    for (uint32_t i = 0; r.ok && i < coalitions; ++i) {
        const uint32_t mask = r.get<uint32_t>();
        e.coalitions[mask] = r.eval();
    }
    if (!r.ok || r.at != body.size() || e.key.seat > 1) {
        *err = "malformed";
        return false;
    }
    (void)method;
    *out = std::move(e);
    return true;
}

bool write_attr_entry(const AttrEntry &e, const std::string &root, std::string *err) {
    const std::string path = e.key.path();
    if (!make_dirs(parent_dir(path))) {
        *err = "cannot make " + parent_dir(path);
        return false;
    }
    if (!write_file_atomic(path, attr_entry_bytes(e), err)) return false;
    prune_attr_cache(root, ATTR_CACHE_MAX_BYTES, path);
    return true;
}

bool read_attr_entry(const std::string &path, AttrEntry *out, std::string *err) {
    std::string bytes;
    if (!read_file(path, &bytes)) {
        *err = "no entry";
        return false;
    }
    return parse_attr_entry(bytes, out, err);
}

void prune_attr_cache(const std::string &root, uint64_t max_bytes, const std::string &keep) {
    std::vector<CacheFile> files;
    uint64_t total = 0;
    collect_files(root, 1, &files, &total);
    std::sort(
        files.begin(), files.end(), [](const CacheFile &a, const CacheFile &b) { return a.mtime_ns < b.mtime_ns; });
    for (const CacheFile &f : files) {
        if (total <= max_bytes) break;
        if (f.path == keep) continue;
        unlink(f.path.c_str());
        total -= std::min(total, f.bytes);
    }
}

std::vector<std::vector<int>> shapley_permutations(int k, uint64_t seed) {
    std::vector<std::vector<int>> out;
    uint64_t s = mix(seed + 0x9E3779B97F4A7C15ull);
    for (int p = 0; p < SHAPLEY_PERMUTATIONS; ++p) {
        std::vector<int> order(static_cast<size_t>(k));
        for (int i = 0; i < k; ++i) order[static_cast<size_t>(i)] = i;
        for (int i = k - 1; i > 0; --i) {
            s = mix(s + 1);
            std::swap(order[static_cast<size_t>(i)], order[static_cast<size_t>(s % static_cast<uint64_t>(i + 1))]);
        }
        out.push_back(order);
    }
    return out;
}

ShapleyResult shapley_values(const AttrEntry &e, AttrTarget t, int focus, int rival, bool negate) {
    ShapleyResult r;
    const int k = static_cast<int>(e.players.size());
    if (k == 0 || !e.have_base || t == AttrTarget::TV) return r;
    r.phi.assign(static_cast<size_t>(k), 0.0);
    const double base = target_scalar(e.base, t, focus, rival, negate);
    const std::vector<std::vector<int>> perms = shapley_permutations(k, fnv1a64(e.key.path()));
    for (const std::vector<int> &order : perms) {
        uint32_t mask = 0;
        double prev = base;
        for (const int u : order) {
            mask |= 1u << u;
            auto it = e.coalitions.find(mask);
            if (it == e.coalitions.end()) return ShapleyResult();
            const double cur = target_scalar(it->second, t, focus, rival, negate);
            r.phi[static_cast<size_t>(u)] += cur - prev;
            prev = cur;
        }
    }
    for (double &v : r.phi) v /= static_cast<double>(perms.size());
    for (const double v : r.phi) r.sum += v;
    auto all = e.coalitions.find((k >= 32 ? 0u : (1u << k)) - 1u);
    if (all == e.coalitions.end()) return ShapleyResult();
    r.total = target_scalar(all->second, t, focus, rival, negate) - base;
    r.ready = true;
    return r;
}

std::map<uint64_t, uint64_t> map_handles(const rmstate::DecodedFrame &from, const rmstate::DecodedFrame &to) {
    std::map<uint64_t, uint64_t> out;
    const auto object_table = [](const rmstate::DecodedFrame &f) -> const rmstate::DecodedTable * {
        for (const rmstate::DecodedTable &t : f.tables)
            if (t.def && t.def->name == "object") return &t;
        return nullptr;
    };
    const auto stack_table = [](const rmstate::DecodedFrame &f) -> const rmstate::DecodedTable * {
        for (const rmstate::DecodedTable &t : f.tables)
            if (t.def && t.def->name == "stack") return &t;
        return nullptr;
    };
    const rmstate::DecodedTable *a = object_table(from), *b = object_table(to);
    if (a && b) {
        const char *const cols[] = {"zone", "owner", "controller", "card", "timestamp"};
        std::vector<int> ca, cb;
        for (const char *c : cols) {
            ca.push_back(featurize::column_index(*a->def, c));
            cb.push_back(featurize::column_index(*b->def, c));
        }
        const int card = 3;
        const auto key_of = [&](const std::vector<rmstate::Value> &row, const std::vector<int> &c) {
            std::vector<uint64_t> k;
            for (const int i : c) k.push_back(i >= 0 ? row[static_cast<size_t>(i)].bits : 0);
            return k;
        };
        std::map<std::vector<uint64_t>, std::vector<uint64_t>> by_key;  // key → `to`'s handles in order
        for (const std::vector<rmstate::Value> &row : b->rows) by_key[key_of(row, cb)].push_back(row[0].bits);
        std::map<std::vector<uint64_t>, size_t> used;
        for (const std::vector<rmstate::Value> &row : a->rows) {
            if (ca[card] < 0 || rmstate::value_is_all_ones(row[static_cast<size_t>(ca[card])],
                                    a->def->columns[static_cast<size_t>(ca[card])].storage))
                continue;
            const std::vector<uint64_t> k = key_of(row, ca);
            auto it = by_key.find(k);
            if (it == by_key.end()) continue;
            size_t &n = used[k];
            if (n < it->second.size()) out[row[0].bits] = it->second[n++];
        }
    }
    const rmstate::DecodedTable *sa = stack_table(from), *sb = stack_table(to);
    if (sa && sb) {
        const int pa = featurize::column_index(*sa->def, "pos"), pb = featurize::column_index(*sb->def, "pos");
        for (const std::vector<rmstate::Value> &ra : sa->rows)
            for (const std::vector<rmstate::Value> &rb : sb->rows)
                if (pa >= 0 && pb >= 0 && ra[static_cast<size_t>(pa)].bits == rb[static_cast<size_t>(pb)].bits &&
                    out.find(ra[0].bits) == out.end())
                    out[ra[0].bits] = rb[0].bits;
    }
    return out;
}

// ── Attribution ──

Attribution::~Attribution() {
    for (Job &j : jobs_)
        if (ev_) ev_->forget(j.id);
}

void Attribution::fail(const std::string &why, bool no_server) {
    state_ = State::FAILED;
    error_ = why;
    no_server_ = no_server;
}

void Attribution::clear() {
    cancel();
    have_key_ = false;
    entry_.reset();
    rows_.clear();
    frame_.reset();
    have_units_ = false;
    state_ = State::IDLE;
}

void Attribution::cancel() {
    for (Job &j : jobs_) {
        if (!j.current) continue;
        j.current = false;
        if (ev_) ev_->cancel(j.id);
    }
    if (state_ == State::RUNNING) state_ = State::CANCELLED;
}

bool Attribution::busy() const {
    return !jobs_.empty();
}

void Attribution::start(ReplayEvaluator *ev, const Request &r) {
    if (!r.match || r.index >= r.match->decisions.size() || r.seat < 0 || r.seat > 1) return;
    const DecisionInfo &d = r.match->decisions[r.index];
    AttrKey k;
    const std::string root = r.cache_root.empty() ? attr_cache_root() : r.cache_root;
    k.dir = attr_cache_dir(root, r.match->log_hash, r.match->live, r.log_path);
    k.game = d.game;
    k.sideboarding = d.sideboarding;
    k.decision = d.decision;
    k.seat = r.seat;
    k.policy = r.policy;
    k.model = model_identity(r.model);
    k.baseline = r.baseline;
    k.method = r.method;
    if (have_key_ && k == key_) {
        const bool again = state_ == State::FAILED || state_ == State::CANCELLED || !entry_ || !entry_->have_base;
        const bool server_now = r.use_server && !req_.use_server;
        if (!(r.force && again) && !server_now) return;
    }
    cancel();
    ev_ = ev;
    req_ = r;
    req_.cache_root = root;
    key_ = k;
    have_key_ = true;
    index_ = r.index;
    error_.clear();
    no_server_ = false;
    from_cache_ = false;
    evaluated_ = 0;
    focus_ = rival_ = -1;
    second_done_ = shapley_asked_ = false;
    have_units_ = false;
    features_.reset();
    rows_.clear();
    stage_total_ = stage_done_ = 0;
    stage_ = 0;
    entry_.reset(new AttrEntry());
    entry_->key = k;
    std::string err;
    frame_ = r.match->frame(r.index, r.seat, &err);
    std::string payload;
    if (!frame_ || !r.match->read_payload(d.frames[static_cast<size_t>(r.seat)], &payload, &err)) {
        frame_.reset();
        fail(std::string("decision ") + std::to_string(r.index + 1) + " has no frame of seat " +
                 (r.seat == 0 ? "A" : "B") + (err.empty() ? "" : ": " + err),
            false);
        return;
    }
    schema_ = &r.match->stream(r.seat).schema();
    frame_hash_ = fnv1a64(payload);
    menu_rows_ = menu_rows_of(*frame_);
    chosen_ = d.decider == r.seat && d.reply >= 0 && d.reply < menu_rows_ ? d.reply : -1;
    entry_->frame_hash = frame_hash_;
    entry_->menu_rows = menu_rows_;
    // The cache: this directory's entry, else (Replay) the live game's of the same log; for Shapley the
    // occlusion entry as its start.
    std::vector<std::string> paths = {k.path()};
    if (!r.match->live) {
        AttrKey live = k;
        live.dir = attr_cache_dir(root, 0, true, r.log_path);
        paths.push_back(live.path());
    }
    if (k.method == AttrMethod::SHAPLEY) {
        AttrKey occ = k;
        occ.method = AttrMethod::OCCLUSION;
        paths.push_back(occ.path());
    }
    for (const std::string &p : paths) {
        AttrEntry cached;
        if (!read_attr_entry(p, &cached, &err) || cached.frame_hash != frame_hash_ || cached.key.seat != k.seat ||
            cached.menu_rows != menu_rows_)
            continue;
        if (p != k.path() && k.method == AttrMethod::SHAPLEY) {
            cached.players.clear();
            cached.coalitions.clear();
        }
        cached.key = k;
        *entry_ = std::move(cached);
        from_cache_ = true;
        break;
    }
    rebuild_rows();
    if (!r.use_server || !ev_) {
        state_ = entry_->have_base ? State::DONE : State::IDLE;
        return;
    }
    state_ = State::RUNNING;
    features_ = ev_->features();
    if (features_ && entry_->spec_hash && entry_->spec_hash != features_->spec.hash) {
        AttrEntry fresh;
        fresh.key = k;
        fresh.frame_hash = frame_hash_;
        fresh.menu_rows = menu_rows_;
        *entry_ = std::move(fresh);
        from_cache_ = false;
        rebuild_rows();
    }
    if (features_ && !build_units()) return;
    if (from_cache_) {
        submit_version_check();
    } else if (!have_units_) {
        submit(0, {}, true);
    } else {
        advance(1);
    }
}

// The decision featurized with the evaluator's inputs and its units listed; false (FAILED) when not.
bool Attribution::build_units() {
    if (have_units_) return true;
    if (!features_ || !frame_ || !schema_) return false;
    std::vector<featurize::TokenSource> sources;
    std::string err;
    if (!featurize::featurize_traced(features_->spec, features_->tables, *frame_, nullptr, &tensors_, &sources, &err) ||
        !featurize::list_units(
            features_->spec, features_->tables, *schema_, *frame_, tensors_, sources, &list_, &err)) {
        fail("the decision's units: " + err, false);
        return false;
    }
    slots_ = featurize::type_slots(features_->spec);
    entry_->spec_hash = features_->spec.hash;
    have_units_ = true;
    return true;
}

int Attribution::draws(const featurize::Unit &u) const {
    const bool slot = u.kind == featurize::UnitKind::SLOT || u.kind == featurize::UnitKind::CELL;
    return slot && key_.baseline == featurize::Baseline::RESAMPLE ? RESAMPLE_DRAWS : 1;
}

void Attribution::submit_version_check() {
    EvalJob job;
    job.source = req_.source;
    job.policies = {key_.policy};
    job.priority = ATTR_PRIORITY;
    Job j;
    j.entry = entry_;
    j.stage = 5;
    j.id = ev_->submit(std::move(job));
    jobs_.push_back(std::move(j));
    stage_ = 5;
}

void Attribution::submit(int stage, const std::vector<const featurize::Unit *> &units, bool with_base) {
    EvalJob job;
    job.source = req_.source;
    job.policies = {key_.policy};
    job.priority = ATTR_PRIORITY;
    Job j;
    j.entry = entry_;
    j.stage = stage;
    if (with_base) {
        job.items.push_back({index_, key_.seat, nullptr});
        j.items.push_back({"", 0});
    }
    // What every change needs, shared by the items.
    struct Ctx {
            std::shared_ptr<const featurize::Featurization> features;
            std::vector<std::vector<featurize::SlotDef>> slots;
            std::shared_ptr<const featurize::SlotStats> stats;
            featurize::Baseline baseline = featurize::Baseline::NONE;
            size_t n = 0;
    };
    std::shared_ptr<Ctx> ctx(new Ctx());
    ctx->features = features_;
    ctx->slots = slots_;
    ctx->stats = req_.stats;
    ctx->baseline = key_.baseline;
    ctx->n = tensors_.n();
    for (const featurize::Unit *u : units) {
        if (!entry_->unit(u->key)) entry_->units.push_back(*u);
        const size_t have = entry_->evals.count(u->key) ? entry_->evals[u->key].size() : 0;
        for (int d = static_cast<int>(have); d < draws(*u); ++d) {
            std::shared_ptr<const featurize::Unit> unit(new featurize::Unit(*u));
            const uint64_t draw = fnv1a64(u->key) + static_cast<uint64_t>(d);
            job.items.push_back({index_, key_.seat, [ctx, unit, draw](featurize::Tensors *t, std::string *err) {
                                     if (t->n() != ctx->n) {
                                         *err = "the decision's tensors are not those its units were listed from";
                                         return false;
                                     }
                                     featurize::BaselineChoice b;
                                     b.kind = ctx->baseline;
                                     b.stats = ctx->stats.get();
                                     b.draw = draw;
                                     return featurize::apply_units(
                                         ctx->features->spec, ctx->slots, {unit.get()}, b, t, err);
                                 }});
            j.items.push_back({u->key, d});
        }
    }
    if (job.items.empty()) return;
    stage_ = stage;
    stage_total_ = job.items.size();
    stage_done_ = 0;
    j.id = ev_->submit(std::move(job));
    jobs_.push_back(std::move(j));
}

bool Attribution::submit_shapley() {
    if (!entry_->have_base) return false;
    // The players: the evaluated units (no cells) of the largest |target| (V, or TV's ranking by V).
    if (entry_->players.empty()) {
        std::vector<std::pair<double, std::string>> ranked;
        for (const featurize::Unit &u : entry_->units) {
            if (u.kind == featurize::UnitKind::CELL || !entry_->evals.count(u.key)) continue;
            const AttrMetrics m = attr_metrics(entry_->base, entry_->evals[u.key], focus(), rival(), negate());
            ranked.push_back({std::fabs(m.dv) + std::fabs(m.dlogp), u.key});
        }
        std::sort(ranked.begin(), ranked.end(),
            [](const std::pair<double, std::string> &a, const std::pair<double, std::string> &b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
        const size_t k = std::min<size_t>(ranked.size(), static_cast<size_t>(key_.shapley_k));
        for (size_t i = 0; i < k; ++i) entry_->players.push_back(ranked[i].second);
    }
    const int k = static_cast<int>(entry_->players.size());
    if (k == 0) return false;
    std::vector<featurize::Unit> players;
    for (const std::string &key : entry_->players) {
        const featurize::Unit *u = entry_->unit(key);
        if (!u) return false;
        players.push_back(*u);
    }
    struct Ctx {
            std::shared_ptr<const featurize::Featurization> features;
            std::vector<std::vector<featurize::SlotDef>> slots;
            std::shared_ptr<const featurize::SlotStats> stats;
            featurize::Baseline baseline = featurize::Baseline::NONE;
            std::vector<featurize::Unit> players;
            size_t n = 0;
    };
    std::shared_ptr<Ctx> ctx(new Ctx());
    ctx->features = features_;
    ctx->slots = slots_;
    ctx->stats = req_.stats;
    ctx->baseline = key_.baseline;
    ctx->players = players;
    ctx->n = tensors_.n();
    EvalJob job;
    job.source = req_.source;
    job.policies = {key_.policy};
    job.priority = ATTR_PRIORITY;
    Job j;
    j.entry = entry_;
    j.stage = 4;
    std::set<uint32_t> asked;
    for (const std::vector<int> &order : shapley_permutations(k, fnv1a64(key_.path()))) {
        uint32_t mask = 0;
        for (const int u : order) {
            mask |= 1u << u;
            if (entry_->coalitions.count(mask) || !asked.insert(mask).second) continue;
            job.items.push_back({index_, key_.seat, [ctx, mask](featurize::Tensors *t, std::string *err) {
                                     if (t->n() != ctx->n) {
                                         *err = "the decision's tensors are not those its units were listed from";
                                         return false;
                                     }
                                     std::vector<const featurize::Unit *> on;
                                     for (size_t i = 0; i < ctx->players.size(); ++i)
                                         if (mask & (1u << i)) on.push_back(&ctx->players[i]);
                                     featurize::BaselineChoice b;
                                     b.kind = ctx->baseline;
                                     b.stats = ctx->stats.get();
                                     return featurize::apply_units(ctx->features->spec, ctx->slots, on, b, t, err);
                                 }});
            j.items.push_back({"#" + std::to_string(mask), 0});
        }
    }
    if (job.items.empty()) return false;
    stage_ = 4;
    stage_total_ = job.items.size();
    stage_done_ = 0;
    j.id = ev_->submit(std::move(job));
    jobs_.push_back(std::move(j));
    return true;
}

std::vector<const featurize::Unit *> Attribution::missing(int stage) {
    std::vector<const featurize::Unit *> out;
    if (!have_units_) return out;
    const auto lacks = [this](const featurize::Unit &u) {
        auto it = entry_->evals.find(u.key);
        return it == entry_->evals.end() || static_cast<int>(it->second.size()) < draws(u);
    };
    if (stage == 1) {
        for (const featurize::Unit &u : list_.units)
            if (u.coarse && lacks(u)) out.push_back(&u);
        return out;
    }
    if (stage == 2) {
        // The groups that moved the target most: by |dv| + tv (both targets' sense of "moved").
        std::vector<std::pair<double, const featurize::Unit *>> groups;
        for (const featurize::Unit &u : list_.units) {
            if (u.kind != featurize::UnitKind::GROUP || !entry_->evals.count(u.key) || !entry_->have_base) continue;
            const AttrMetrics m = attr_metrics(entry_->base, entry_->evals[u.key], focus(), rival(), false);
            groups.push_back({std::fabs(m.dv) + m.tv, &u});
        }
        std::stable_sort(groups.begin(), groups.end(),
            [](const std::pair<double, const featurize::Unit *> &a,
                const std::pair<double, const featurize::Unit *> &b) { return a.first > b.first; });
        for (size_t g = 0; g < groups.size() && g < SECOND_PASS_GROUPS; ++g) {
            const featurize::Unit &grp = *groups[g].second;
            std::vector<std::string> &kids = entry_->children[grp.key];
            for (const featurize::Unit &child :
                featurize::unit_children(features_->spec, slots_, list_, tensors_, grp)) {
                if (std::find(kids.begin(), kids.end(), child.key) == kids.end()) kids.push_back(child.key);
                const int at = list_.find(child.key);
                if (at >= 0 && lacks(list_.units[static_cast<size_t>(at)]))
                    out.push_back(&list_.units[static_cast<size_t>(at)]);
            }
        }
        for (const featurize::Unit &u : list_.units)
            if ((u.kind == featurize::UnitKind::SLOT || u.kind == featurize::UnitKind::RELATION) && lacks(u))
                out.push_back(&u);
    }
    return out;
}

void Attribution::advance(int stage) {
    if (!have_units_) return;
    for (int s = stage; s <= 4; ++s) {
        if (s == 1) {
            const std::vector<const featurize::Unit *> u = missing(1);
            if (!u.empty() || !entry_->have_base) {
                submit(1, u, !entry_->have_base);
                return;
            }
        } else if (s == 2) {
            const std::vector<const featurize::Unit *> u = missing(2);
            rebuild_rows();
            if (!u.empty()) {
                submit(2, u, false);
                return;
            }
            second_done_ = true;
        } else if (s == 4 && key_.method == AttrMethod::SHAPLEY) {
            if (submit_shapley()) return;
        }
    }
    second_done_ = true;
    stage_ = 0;
    state_ = State::DONE;
}

void Attribution::take_results(Job &j) {
    for (EvalResult &r : ev_->take(j.id)) {
        if (r.item >= j.items.size() || r.policies.empty()) continue;
        const ItemRef &ref = j.items[r.item];
        AttrEval e;
        e.value = r.policies[0].value;
        e.log_pi = r.policies[0].log_pi;
        AttrEntry &entry = *j.entry;
        if (ref.key.empty()) {
            entry.base = e;
            entry.have_base = true;
        } else if (ref.key[0] == '#') {
            entry.coalitions[static_cast<uint32_t>(std::strtoul(ref.key.c_str() + 1, nullptr, 10))] = e;
        } else {
            std::vector<AttrEval> &v = entry.evals[ref.key];
            if (static_cast<int>(v.size()) == ref.draw) v.push_back(e);
        }
        ++j.done;
        if (j.current) {
            ++evaluated_;
            ++stage_done_;
        }
    }
}

void Attribution::finish_job(Job &j, const EvalJobStatus &st) {
    if (!st.versions.empty() && j.stage != 5) j.entry->version = st.versions[0];
    if (j.done > 0) write(*j.entry);
    if (!j.current) return;
    if (st.state == EvalJobStatus::State::FAILED) {
        fail(st.error, st.no_server);
        return;
    }
    if (st.state != EvalJobStatus::State::DONE) return;
    switch (j.stage) {
        case 0:
            features_ = ev_->features();
            if (!build_units()) return;
            rebuild_rows();
            advance(1);
            return;
        case 1:
            rebuild_rows();
            advance(2);
            return;
        case 2:
            second_done_ = true;
            rebuild_rows();
            advance(4);
            return;
        case 3:
            rebuild_rows();
            if (state_ == State::RUNNING && stage_ == 3) state_ = State::DONE;
            return;
        case 4:
            rebuild_rows();
            stage_ = 0;
            state_ = State::DONE;
            return;
        case 5: {
            features_ = ev_->features();
            if (features_ && entry_->spec_hash && entry_->spec_hash != features_->spec.hash) entry_->have_base = false;
            const uint64_t now = st.versions.empty() ? 0 : st.versions[0];
            if (entry_->have_base && now != entry_->version) {
                // Another version: the cached evaluations are another model's.
                AttrEntry fresh;
                fresh.key = key_;
                fresh.frame_hash = frame_hash_;
                fresh.menu_rows = menu_rows_;
                *entry_ = std::move(fresh);
                from_cache_ = false;
            }
            if (!build_units()) return;
            rebuild_rows();
            advance(1);
            if (state_ == State::DONE && from_cache_) stage_ = 0;
            return;
        }
        default:
            return;
    }
}

void Attribution::poll() {
    if (!ev_) return;
    bool changed = false;
    for (size_t i = 0; i < jobs_.size();) {
        Job &j = jobs_[i];
        const size_t before = j.done;
        take_results(j);
        changed = changed || (j.current && j.done != before);
        const EvalJobStatus st = ev_->status(j.id);
        if (st.finished() || st.state == EvalJobStatus::State::UNKNOWN) {
            take_results(j);
            Job done = std::move(j);
            jobs_.erase(jobs_.begin() + static_cast<long>(i));
            ev_->forget(done.id);
            finish_job(done, st);
            changed = changed || done.current;
            continue;
        }
        ++i;
    }
    if (changed) rebuild_rows();
}

void Attribution::expand(int row) {
    if (row < 0 || static_cast<size_t>(row) >= rows_.size() || !entry_ || !ev_) return;
    AttrRow &r = rows_[static_cast<size_t>(row)];
    if (r.asked || !r.expandable) return;
    const featurize::Unit *u = entry_->unit(r.key);
    if (!u || !have_units_) return;
    r.asked = true;
    const std::vector<featurize::Unit> kids = featurize::unit_children(features_->spec, slots_, list_, tensors_, *u);
    std::vector<std::string> &keys = entry_->children[u->key];
    std::vector<featurize::Unit> asked;
    for (const featurize::Unit &k : kids) {
        if (std::find(keys.begin(), keys.end(), k.key) == keys.end()) keys.push_back(k.key);
        if (!entry_->unit(k.key)) entry_->units.push_back(k);
        auto it = entry_->evals.find(k.key);
        if (it == entry_->evals.end() || static_cast<int>(it->second.size()) < draws(k)) asked.push_back(k);
    }
    rebuild_rows();
    if (asked.empty() || !req_.use_server) return;
    std::vector<const featurize::Unit *> ptrs;
    for (const featurize::Unit &k : asked) ptrs.push_back(&k);
    if (state_ != State::RUNNING) state_ = State::RUNNING;
    submit(3, ptrs, false);
}

std::string Attribution::phase() const {
    switch (stage_) {
        case 0:
            return have_units_ || !running() ? "" : "Attribution: the unchanged evaluation";
        case 1:
            return "Attribution: coarse units " + std::to_string(stage_done_) + " of " + std::to_string(stage_total_);
        case 2:
            return "Attribution: second pass " + std::to_string(stage_done_) + " of " + std::to_string(stage_total_);
        case 3:
            return "Attribution: children " + std::to_string(stage_done_) + " of " + std::to_string(stage_total_);
        case 4:
            return "Attribution: Shapley coalitions " + std::to_string(stage_done_) + " of " +
                   std::to_string(stage_total_);
        case 5:
            return "Attribution: checking the cached version";
        default:
            return "";
    }
}

double Attribution::fraction() const {
    return stage_total_ ? std::min(1.0, static_cast<double>(stage_done_) / static_cast<double>(stage_total_)) : -1.0;
}

void Attribution::set_focus(int focus, int rival) {
    if (focus == focus_ && rival == rival_) return;
    focus_ = focus;
    rival_ = rival;
    recompute();
}

int Attribution::focus() const {
    if (focus_ >= 0 && focus_ < menu_rows_) return focus_;
    if (chosen_ >= 0) return chosen_;
    return entry_ && entry_->have_base ? attr_argmax(entry_->base.log_pi) : -1;
}

int Attribution::rival() const {
    const int f = focus();
    if (rival_ >= 0 && rival_ < menu_rows_ && rival_ != f) return rival_;
    return entry_ && entry_->have_base ? attr_runner_up(entry_->base.log_pi, f) : -1;
}

void Attribution::set_scale_a(bool on) {
    if (on == scale_a_) return;
    scale_a_ = on;
    recompute();
}

bool Attribution::negate() const {
    return scale_a_ && key_.seat == 1;
}

const featurize::Unit *Attribution::unit(const AttrRow &r) const {
    return entry_ ? entry_->unit(r.key) : nullptr;
}

void Attribution::add_row(const std::string &key, int parent, int depth) {
    const int at = static_cast<int>(rows_.size());
    AttrRow r;
    r.key = key;
    r.parent = parent;
    r.depth = depth;
    auto kids = entry_->children.find(key);
    r.asked = kids != entry_->children.end();
    const featurize::Unit *u = entry_->unit(key);
    r.expandable =
        depth + 1 < MAX_DEPTH && u && u->kind != featurize::UnitKind::CELL && u->kind != featurize::UnitKind::RELATION;
    rows_.push_back(r);
    if (parent >= 0) rows_[static_cast<size_t>(parent)].children.push_back(at);
    if (kids == entry_->children.end() || depth + 1 >= MAX_DEPTH) return;
    for (const std::string &k : kids->second)
        if (entry_->unit(k)) add_row(k, at, depth + 1);
}

void Attribution::rebuild_rows() {
    rows_.clear();
    if (!entry_) return;
    std::set<std::string> child;
    for (const auto &kv : entry_->children)
        for (const std::string &k : kv.second) child.insert(k);
    for (const featurize::Unit &u : entry_->units) {
        const bool root = u.coarse || u.kind == featurize::UnitKind::SLOT || u.kind == featurize::UnitKind::RELATION ||
                          !child.count(u.key);
        if (root) add_row(u.key, -1, 0);
    }
    recompute();
}

void Attribution::recompute() {
    if (!entry_) return;
    const int f = focus(), r = rival();
    for (AttrRow &row : rows_) {
        auto it = entry_->evals.find(row.key);
        row.m = entry_->have_base && it != entry_->evals.end() ? attr_metrics(entry_->base, it->second, f, r, negate())
                                                               : AttrMetrics();
        row.shapley = false;
    }
    if (key_.method != AttrMethod::SHAPLEY || entry_->players.empty()) return;
    const ShapleyResult s = shapley_values(*entry_, AttrTarget::V, f, r, negate());
    if (!s.ready) return;
    for (AttrRow &row : rows_)
        for (size_t p = 0; p < entry_->players.size(); ++p)
            if (entry_->players[p] == row.key) {
                row.shapley = true;
                row.phi = s.phi[p];
            }
}

ShapleyResult Attribution::shapley(AttrTarget t) const {
    if (!entry_ || key_.method != AttrMethod::SHAPLEY) return ShapleyResult();
    return shapley_values(*entry_, t, focus(), rival(), negate());
}

double Attribution::row_value(const AttrRow &r, AttrTarget t) const {
    return r.m.valid ? r.m.get(t) : 0.0;
}

std::vector<int> Attribution::ranked_roots(AttrTarget t) const {
    std::vector<int> out;
    for (size_t i = 0; i < rows_.size(); ++i)
        if (rows_[i].parent < 0) out.push_back(static_cast<int>(i));
    std::stable_sort(out.begin(), out.end(), [this, t](int a, int b) {
        const AttrRow &ra = rows_[static_cast<size_t>(a)], &rb = rows_[static_cast<size_t>(b)];
        if (ra.m.valid != rb.m.valid) return ra.m.valid;
        return std::fabs(row_value(ra, t)) > std::fabs(row_value(rb, t));
    });
    return out;
}

AttrTints Attribution::tints(AttrTarget t, const rmstate::DecodedFrame *view_frame, int view) const {
    AttrTints out;
    out.target = t;
    if (!entry_ || !entry_->have_base || !frame_) return out;
    out.on = true;
    out.same_frame = view == key_.seat;
    if (!out.same_frame && view_frame) out.handles = map_handles(*frame_, *view_frame);
    const auto handle_in_view = [&out](uint64_t h) { return out.view_handle(h); };
    const int f = focus(), r = rival();
    const auto value = [&](const std::string &key, double *v) {
        auto it = entry_->evals.find(key);
        if (it == entry_->evals.end()) return false;
        const AttrMetrics m = attr_metrics(entry_->base, it->second, f, r, negate());
        *v = m.get(t);
        return true;
    };
    std::map<int64_t, double> by_card;
    std::map<uint64_t, bool> from_token;
    // A delta token (an object's subtype or keyword it differs by) names its object's row, not its own.
    int object_table = -1;
    for (const rmstate::DecodedTable &tab : frame_->tables)
        if (tab.def && tab.def->name == "object") object_table = tab.id;
    for (const featurize::Unit &u : entry_->units) {
        double v = 0;
        if (!value(u.key, &v)) continue;
        switch (u.kind) {
            case featurize::UnitKind::TOKEN:
                if (u.table >= 0 && (u.owns || u.table != object_table)) out.rows.insert({{u.table, u.row}, v});
                if (u.handle && u.owns) {
                    const uint64_t h = handle_in_view(u.handle);
                    if (h) {
                        out.objects[h] = v;
                        from_token[h] = true;
                    }
                }
                break;
            case featurize::UnitKind::GROUP:
                if (u.seat >= 0 && u.group.size() > 2 && u.group.compare(2, std::string::npos, "player") == 0) {
                    out.players[static_cast<size_t>(u.seat)] = v;
                    out.has_player[static_cast<size_t>(u.seat)] = true;
                }
                break;
            case featurize::UnitKind::CARD:
                by_card[u.card] = v;
                break;
            case featurize::UnitKind::CELL:
                if (u.table >= 0) out.cells[std::make_tuple(u.table, u.row, u.column)] = v;
                if (!u.tokens.empty()) out.token_cells[{u.tokens[0], u.slot}] = v;
                break;
            default:
                break;
        }
    }
    // An object without its own token's value takes its card's.
    for (const rmstate::DecodedTable &tab : frame_->tables) {
        if (!tab.def || tab.def->name != "object") continue;
        const int card = featurize::column_index(*tab.def, "card");
        if (card < 0) break;
        for (const std::vector<rmstate::Value> &row : tab.rows) {
            const uint64_t h = handle_in_view(row[0].bits);
            if (!h || from_token.count(h)) continue;
            auto it = by_card.find(static_cast<int64_t>(row[static_cast<size_t>(card)].bits));
            if (it != by_card.end()) out.objects[h] = it->second;
        }
    }
    for (const auto &kv : out.objects) out.scale = std::max(out.scale, std::fabs(kv.second));
    for (int s = 0; s < 2; ++s)
        if (out.has_player[static_cast<size_t>(s)])
            out.scale = std::max(out.scale, std::fabs(out.players[static_cast<size_t>(s)]));
    for (const auto &kv : out.rows) out.scale = std::max(out.scale, std::fabs(kv.second));
    for (const auto &kv : out.cells) out.scale = std::max(out.scale, std::fabs(kv.second));
    for (const auto &kv : out.token_cells) out.cell_scale = std::max(out.cell_scale, std::fabs(kv.second));
    out.legend = std::string(attr_target_name(t)) + " \xC2\xB7 decision " + std::to_string(index_ + 1) + ", seat " +
                 (key_.seat == 0 ? "A" : "B") + " \xC2\xB7 p" + std::to_string(key_.policy);
    if (!out.same_frame) out.rows.clear(), out.cells.clear();
    return out;
}

void Attribution::write(const AttrEntry &e) {
    std::string err;
    write_attr_entry(e, req_.cache_root, &err);
}

// ── The slot statistics ──

SlotStatsSource::~SlotStatsSource() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
}

bool SlotStatsSource::building() const {
    return running_.load();
}

std::string SlotStatsSource::note() const {
    std::lock_guard<std::mutex> lock(mu_);
    return note_;
}

std::shared_ptr<const featurize::SlotStats> SlotStatsSource::get(
    const std::shared_ptr<const featurize::Featurization> &f, const std::string &engine, const std::string &workdir,
    const std::string &corpus, const std::string &dir) {
    if (!f) return nullptr;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stats_ && spec_ == f->spec.hash) return stats_;
        if (running_.load() || (spec_ == f->spec.hash && !note_.empty())) return nullptr;
    }
    const std::string path = dir + "/" + featurize::hex16(f->spec.hash) + ".rmtb";
    featurize::RmtbFile file;
    featurize::SlotStats s;
    std::string err;
    if (featurize::rmtb_read_file(path, &file, &err) && featurize::parse_slot_stats(file, &s, &err) &&
        s.spec_hash == f->spec.hash) {
        std::lock_guard<std::mutex> lock(mu_);
        stats_.reset(new featurize::SlotStats(std::move(s)));
        spec_ = f->spec.hash;
        note_.clear();
        return stats_;
    }
    if (thread_.joinable()) thread_.join();
    {
        std::lock_guard<std::mutex> lock(mu_);
        spec_ = f->spec.hash;
        note_ = "Slot statistics: replaying the corpus";
    }
    running_.store(true);
    stop_.store(false);
    thread_ = std::thread([this, f, engine, workdir, corpus, path] { build(f, engine, workdir, corpus, path); });
    return nullptr;
}

void SlotStatsSource::build(std::shared_ptr<const featurize::Featurization> f, std::string engine, std::string workdir,
    std::string corpus, std::string path) {
    std::vector<std::string> logs;
    if (DIR *d = opendir(corpus.c_str())) {
        while (const dirent *e = readdir(d)) {
            const std::string n = e->d_name;
            if (n.size() > 6 && n.compare(n.size() - 6, 6, ".rmlog") == 0) logs.push_back(corpus + "/" + n);
        }
        closedir(d);
    }
    std::sort(logs.begin(), logs.end());
    if (logs.size() > SLOT_STATS_LOGS) logs.resize(SLOT_STATS_LOGS);
    featurize::SlotStatsBuilder builder(f->spec);
    const std::string tmp = make_temp_dir("rmwork_slot_stats_");
    std::string why;
    for (size_t i = 0; i < logs.size() && why.empty() && !stop_.load(); ++i) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            note_ = "Slot statistics: replaying the corpus (" + std::to_string(i + 1) + " of " +
                    std::to_string(logs.size()) + ")";
        }
        const std::string dump = tmp + "/dump.rmst";
        ChildProcess child;
        std::string err;
        if (!child.start({engine, "--replay", logs[i], "--rmstate-dump", dump}, workdir, tmp + "/out.txt",
                tmp + "/err.txt", &err)) {
            why = err;
            break;
        }
        while (child.poll() && !stop_.load()) usleep(20000);
        if (stop_.load()) child.kill_and_reap();
        if (!child.ok()) {
            why = "the engine's replay of " + logs[i] + ": " + child.status_text();
            break;
        }
        const bool ok = featurize::read_stream(
            dump,
            [&](const featurize::ViewerFrames &, const rmstate::DecodedFrame &state, std::string *e) {
                if (state.header.viewer > 1) return true;
                featurize::Tensors t;
                if (!featurize::featurize_seat(f->spec, f->tables, state, &t, e)) return false;
                builder.add(t);
                return true;
            },
            &err);
        if (!ok) why = logs[i] + ": " + err;
    }
    std::string err;
    if (why.empty() && logs.empty()) why = "no decision log in " + corpus;
    if (why.empty() && !stop_.load()) {
        featurize::SlotStats s = builder.finish();
        if (!make_dirs(parent_dir(path)) || !featurize::rmtb_write_file(path, featurize::slot_stats_file(s), &err)) {
            why = "cannot write " + path + ": " + err;
        } else {
            std::lock_guard<std::mutex> lock(mu_);
            stats_.reset(new featurize::SlotStats(std::move(s)));
            note_.clear();
        }
    }
    remove_tree(tmp);
    if (!why.empty()) {
        std::lock_guard<std::mutex> lock(mu_);
        note_ = "Slot statistics not built: " + why;
    }
    running_.store(false);
}

}  // namespace rmwork
