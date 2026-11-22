#include "policy_catalog.h"

#include <dirent.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <utility>

#include "../../src/featurize/json.h"
#include "file_picker.h"
#include "files.h"
#include "model_zip.h"

namespace {

const char *const LATEST = "latest";
const char *const LATEST_INFO = "latest.json";
const char *const RUNS_PREFIX = "runs/";
const char *const EXPORTS_DIR = "/exports/";
const char *const DOT = " \xC2\xB7 ";
// The metrics events that name a policy.
const char *const NAMING_EVENTS[] = {"rnad_iteration", "checkpoint", "export", "latest"};
const size_t READ_BLOCK = 1 << 20;

// A policy's earliest metrics line and the learner steps one gave.
struct Mention {
        double time = 0;
        int64_t steps = -1;
};

using Mentions = std::map<std::string, Mention>;

// A metrics file read up to `offset` (the end of its last whole line), and what it said there.
struct MetricsRead {
        uint64_t size = 0;
        int64_t mtime_ns = 0;
        uint64_t offset = 0;
        Mentions named;
};

// A file of a policy: its format bit and path.
struct PolicyFile {
        unsigned format = 0;
        std::string path;
};

std::mutex metrics_mutex;
std::map<std::string, MetricsRead> metrics_cache;

}  // namespace

static std::vector<std::string> names_in(const std::string &dir, bool dirs);
static bool ends_with(const std::string &s, const char *suffix);
static std::string stem_of(const std::string &file, const char *suffix);
static bool is_periodic(const std::string &stem);
static std::string checkpoint_row_name(const std::string &stem);
static bool split_export(const std::string &file, std::string *stem, unsigned *format);
static void note_mention(Mentions *m, const std::string &name, double time, int64_t steps);
static std::string event_of(const std::string &line);
static void read_metrics_line(const std::string &line, Mentions *m);
static Mentions metrics_mentions(const std::string &path);
static void add_file(rmwork::PolicyRow *row, unsigned format, const std::string &path);
static void finish_row(rmwork::PolicyRow *row);
static void scan_run(const std::string &dir, const std::string &run, std::vector<rmwork::PolicyRow> *out);
static bool latest_row(const std::string &dir, rmwork::PolicyRow *row);
static void zip_rows(const std::string &models_dir, std::vector<rmwork::PolicyRow> *out);
static bool newer_first(const rmwork::PolicyRow &a, const rmwork::PolicyRow &b);

// The entries of `dir` that are directories (`dirs`) or regular files, in natural order, hidden ones
// and temporary files left out.
static std::vector<std::string> names_in(const std::string &dir, bool dirs) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    while (const dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.empty() || n[0] == '.' || n.find(".tmp") != std::string::npos) continue;
        const rmwork::PathKind k = rmwork::path_kind(dir + "/" + n);
        if (dirs ? k == rmwork::PathKind::DIRECTORY : k == rmwork::PathKind::FILE) out.push_back(n);
    }
    closedir(d);
    std::sort(out.begin(), out.end(), rmwork::natural_less);
    return out;
}

static bool ends_with(const std::string &s, const char *suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() > n && s.compare(s.size() - n, n, suffix) == 0;
}

// `file` without `suffix` ("" when it does not end with it).
static std::string stem_of(const std::string &file, const char *suffix) {
    return ends_with(file, suffix) ? file.substr(0, file.size() - std::strlen(suffix)) : "";
}

// A periodic checkpoint's stem: `<kind>-p<step>` (rnad-p1200, bc-p300) or `p<step>`.
static bool is_periodic(const std::string &stem) {
    const size_t p = stem.rfind('p');
    if (p == std::string::npos || p + 1 >= stem.size() || (p > 0 && stem[p - 1] != '-')) return false;
    for (size_t i = p + 1; i < stem.size(); ++i)
        if (!std::isdigit(static_cast<unsigned char>(stem[i]))) return false;
    return true;
}

// The row of a checkpoint in ckpt/: a periodic one by its stem, else the stem without `rnad-`
// (rnad-final → final, rnad-seed → seed; bc-final stays).
static std::string checkpoint_row_name(const std::string &stem) {
    if (is_periodic(stem)) return stem;
    return stem.compare(0, 5, "rnad-") == 0 && stem.size() > 5 ? stem.substr(5) : stem;
}

// An export's name and format by its suffix (.ts.pt, .pt2, .onnx).
static bool split_export(const std::string &file, std::string *stem, unsigned *format) {
    const std::pair<const char *, unsigned> kinds[] = {
        {".ts.pt", rmwork::FORMAT_TS}, {".pt2", rmwork::FORMAT_PT2}, {".onnx", rmwork::FORMAT_ONNX}};
    for (const auto &k : kinds) {
        *stem = stem_of(file, k.first);
        if (!stem->empty()) {
            *format = k.second;
            return true;
        }
    }
    return false;
}

static void note_mention(Mentions *m, const std::string &name, double time, int64_t steps) {
    if (name.empty()) return;
    Mention &x = (*m)[name];
    if (time > 0 && (x.time == 0 || time < x.time)) x.time = time;
    if (x.steps < 0 && steps >= 0) x.steps = steps;
}

// A metrics line's event ("" when it has none), read without parsing the line.
static std::string event_of(const std::string &line) {
    size_t at = line.find("\"event\"");
    if (at == std::string::npos) return "";
    at += 7;
    while (at < line.size() && (line[at] == ' ' || line[at] == ':')) ++at;
    if (at >= line.size() || line[at] != '"') return "";
    const size_t end = line.find('"', at + 1);
    return end == std::string::npos ? "" : line.substr(at + 1, end - at - 1);
}

// The policy a naming metrics line names, noted in `m` (other lines are not parsed).
static void read_metrics_line(const std::string &line, Mentions *m) {
    const std::string event = event_of(line);
    if (std::find(std::begin(NAMING_EVENTS), std::end(NAMING_EVENTS), event) == std::end(NAMING_EVENTS)) return;
    featurize::Json doc;
    std::string err;
    if (!featurize::Json::parse(line, &doc, &err) || !doc.is_object()) return;
    const featurize::Json *t = doc.get("time"), *steps = doc.get("learner_steps");
    const double time = t && t->is_number() ? t->as_number() : 0;
    const int64_t n = steps && steps->is_number() ? steps->as_int() : -1;
    if (event == "rnad_iteration") {
        const featurize::Json *k = doc.get("reg_updates"), *fp = doc.get("fixed_point");
        if (k && k->is_number()) {
            note_mention(m, "m" + std::to_string(k->as_int()), time, n);
        } else if (fp && fp->is_string()) {
            const std::string &p = fp->as_string();
            note_mention(m, stem_of(p.substr(p.rfind('/') + 1), ".pt"), time, n);
        }
    } else if (event == "checkpoint") {
        const featurize::Json *p = doc.get("path"), *at = doc.get("at");
        if (!p || !p->is_string()) return;
        const std::string stem = stem_of(p->as_string().substr(p->as_string().rfind('/') + 1), ".pt");
        if (!stem.empty())
            note_mention(m, checkpoint_row_name(stem), at && at->is_number() ? at->as_number() : time, n);
    } else {
        const featurize::Json *name = doc.get("name");
        if (name && name->is_string()) note_mention(m, name->as_string(), time, n);
    }
}

// The policies a run's metrics.jsonl names. Read from where the last read of the file stopped while
// it only grew; again from its start when it shrank.
static Mentions metrics_mentions(const std::string &path) {
    const rmwork::FileStamp st = rmwork::file_stamp(path);
    std::lock_guard<std::mutex> lock(metrics_mutex);
    if (!st.regular) {
        metrics_cache.erase(path);
        return Mentions();
    }
    MetricsRead &r = metrics_cache[path];
    if (r.size == st.size && r.mtime_ns == st.mtime_ns) return r.named;
    if (st.size < r.offset) r = MetricsRead();
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return r.named;
    if (std::fseek(f, static_cast<long>(r.offset), SEEK_SET) == 0) {
        std::string block(READ_BLOCK, '\0'), line;
        size_t got;
        while ((got = std::fread(&block[0], 1, block.size(), f)) > 0) {
            size_t from = 0;
            for (size_t nl; (nl = block.find('\n', from)) < got; from = nl + 1) {
                line.append(block, from, nl - from);
                read_metrics_line(line, &r.named);
                r.offset += line.size() + 1;
                line.clear();
            }
            line.append(block, from, got - from);
        }
    }
    std::fclose(f);
    r.size = st.size;
    r.mtime_ns = st.mtime_ns;
    return r.named;
}

static void add_file(rmwork::PolicyRow *row, unsigned format, const std::string &path) {
    row->formats |= format;
    switch (format) {
        case rmwork::FORMAT_TS: row->ts = path; break;
        case rmwork::FORMAT_PT2: row->pt2 = path; break;
        case rmwork::FORMAT_ONNX: row->onnx = path; break;
        default: row->checkpoint = path; break;
    }
}

// The export a row plays and its label.
static void finish_row(rmwork::PolicyRow *row) {
    row->play = !row->ts.empty() ? row->ts : row->pt2;
    if (row->latest) {
        row->label = LATEST;
        if (!row->run.empty() || !row->name.empty())
            row->label += DOT + row->run + (row->run.empty() ? "" : " ") + row->name;
    } else if (row->archive.empty()) {
        row->label = rmwork::policy_date_text(row->trained_at) + DOT + row->run + "/" + row->name;
    }
}

// The policies of the run directory `dir` (named `run`), unsorted.
static void scan_run(const std::string &dir, const std::string &run, std::vector<rmwork::PolicyRow> *out) {
    std::map<std::string, std::vector<PolicyFile>> files;
    for (const std::string &n : names_in(dir + "/ckpt/fixed", false)) {
        const std::string stem = stem_of(n, ".pt");
        if (!stem.empty()) files[stem].push_back({rmwork::FORMAT_CHECKPOINT, dir + "/ckpt/fixed/" + n});
    }
    for (const std::string &n : names_in(dir + "/ckpt", false)) {
        const std::string stem = stem_of(n, ".pt");
        if (!stem.empty()) files[checkpoint_row_name(stem)].push_back({rmwork::FORMAT_CHECKPOINT, dir + "/ckpt/" + n});
    }
    std::string stem;
    unsigned format = 0;
    for (const std::string &n : names_in(dir + "/exports", false))
        if (split_export(n, &stem, &format)) files[stem].push_back({format, dir + "/exports/" + n});
    for (const std::string &n : names_in(dir + "/exports/live", false))
        if (split_export(n, &stem, &format)) files["live"].push_back({format, dir + "/exports/live/" + n});
    if (files.empty()) return;
    const Mentions named = metrics_mentions(dir + "/metrics.jsonl");
    const std::string manifest = dir + "/tables/manifest.json";
    const bool has_manifest = rmwork::path_kind(manifest) == rmwork::PathKind::FILE;
    for (const auto &kv : files) {
        rmwork::PolicyRow row;
        row.run = run;
        row.name = kv.first;
        row.hidden = is_periodic(kv.first) || kv.first == "seed" || kv.first == "live";
        int64_t oldest = 0;
        for (const PolicyFile &f : kv.second) {
            add_file(&row, f.format, f.path);
            const int64_t t = rmwork::file_stamp(f.path).mtime_ns;
            if (oldest == 0 || t < oldest) oldest = t;
        }
        const auto m = named.find(kv.first);
        if (m != named.end() && m->second.time > 0 && kv.first != "live") {
            row.trained_at = m->second.time;
            row.dated_by = rmwork::PolicyDateSource::METRICS;
        } else {
            row.trained_at = static_cast<double>(oldest) / 1e9;
            row.dated_by = rmwork::PolicyDateSource::MTIME;
        }
        if (m != named.end()) row.learner_steps = m->second.steps;
        if (has_manifest) row.tables_manifest = manifest;
        finish_row(&row);
        out->push_back(std::move(row));
    }
}

// The runs root's latest/ as the pinned row: false when it holds neither latest.json nor a policy file.
static bool latest_row(const std::string &dir, rmwork::PolicyRow *row) {
    if (rmwork::path_kind(dir) != rmwork::PathKind::DIRECTORY) return false;
    row->latest = true;
    row->name.clear();
    const std::pair<const char *, unsigned> kinds[] = {{"policy.ts.pt", rmwork::FORMAT_TS},
        {"policy.pt2", rmwork::FORMAT_PT2}, {"policy.onnx", rmwork::FORMAT_ONNX},
        {"model.pt", rmwork::FORMAT_CHECKPOINT}};
    int64_t oldest = 0;
    for (const auto &k : kinds) {
        const std::string p = dir + "/" + k.first;
        const rmwork::FileStamp st = rmwork::file_stamp(p);
        if (!st.regular) continue;
        add_file(row, k.second, p);
        if (oldest == 0 || st.mtime_ns < oldest) oldest = st.mtime_ns;
    }
    const std::string info = dir + "/" + LATEST_INFO;
    std::string text, err;
    featurize::Json doc;
    const bool has_info = rmwork::read_file(info, &text) && featurize::Json::parse(text, &doc, &err) && doc.is_object();
    if (!has_info && row->formats == 0) return false;
    if (has_info) {
        row->tables_manifest = info;
        const featurize::Json *run = doc.get("run"), *name = doc.get("name"), *steps = doc.get("learner_steps"),
                              *at = doc.get("trained_at");
        if (run && run->is_string()) row->run = run->as_string();
        if (name && name->is_string()) row->name = name->as_string();
        if (steps && steps->is_number()) row->learner_steps = steps->as_int();
        if (at && at->is_number() && at->as_number() > 0) {
            row->trained_at = at->as_number();
            row->dated_by = rmwork::PolicyDateSource::LATEST;
        }
    }
    if (row->dated_by != rmwork::PolicyDateSource::LATEST && oldest != 0) {
        row->trained_at = static_cast<double>(oldest) / 1e9;
        row->dated_by = rmwork::PolicyDateSource::MTIME;
    }
    finish_row(row);
    return true;
}

// The models directory's zipped exports, a row per archive and export name (its formats merged).
static void zip_rows(const std::string &models_dir, std::vector<rmwork::PolicyRow> *out) {
    for (const std::string &zip : names_in(models_dir, false)) {
        if (!ends_with(zip, ".zip")) continue;
        std::string why, stem;
        unsigned format = 0;
        std::vector<std::string> order;
        std::map<std::string, rmwork::PolicyRow> rows;
        for (const rmwork::ZipEntry &e : rmwork::zip_model_entries(models_dir + "/" + zip, &why)) {
            if (!split_export(e.name, &stem, &format)) continue;
            if (!rows.count(stem)) order.push_back(stem);
            rmwork::PolicyRow &row = rows[stem];
            add_file(&row, format, models_dir + "/" + zip + "!" + e.name);
        }
        for (const std::string &s : order) {
            rmwork::PolicyRow &row = rows[s];
            row.archive = zip;
            row.name = s.substr(s.rfind('/') + 1);
            const size_t exports = s.find(EXPORTS_DIR);
            if (s.compare(0, std::strlen(RUNS_PREFIX), RUNS_PREFIX) == 0 && exports != std::string::npos)
                row.run = s.substr(std::strlen(RUNS_PREFIX), exports - std::strlen(RUNS_PREFIX));
            row.label = rmwork::zip_model_label(zip, s);
            finish_row(&row);
            out->push_back(std::move(row));
        }
    }
}

// The newer policy first; then by run, then by name in natural order, the later first (m10 before m9).
static bool newer_first(const rmwork::PolicyRow &a, const rmwork::PolicyRow &b) {
    if (a.trained_at != b.trained_at) return a.trained_at > b.trained_at;
    if (a.run != b.run) return rmwork::natural_less(a.run, b.run);
    return rmwork::natural_less(b.name, a.name);
}

namespace rmwork {

std::vector<PolicyRow> scan_policies(const std::string &runs, const std::string &models_dir) {
    std::vector<PolicyRow> out, dated;
    PolicyRow latest;
    if (!runs.empty() && latest_row(runs + "/" + LATEST, &latest)) out.push_back(std::move(latest));
    if (!runs.empty())
        for (const std::string &run : names_in(runs, true))
            if (run != LATEST) scan_run(runs + "/" + run, run, &dated);
    std::stable_sort(dated.begin(), dated.end(), newer_first);
    for (PolicyRow &r : dated) out.push_back(std::move(r));
    if (!models_dir.empty()) zip_rows(models_dir, &out);
    return out;
}

std::string policy_formats_text(unsigned formats) {
    const std::pair<unsigned, const char *> names[] = {
        {FORMAT_TS, "ts"}, {FORMAT_PT2, "pt2"}, {FORMAT_ONNX, "onnx"}, {FORMAT_CHECKPOINT, "checkpoint"}};
    std::string out;
    for (const auto &n : names)
        if (formats & n.first) out += (out.empty() ? "" : ", ") + std::string(n.second);
    return out;
}

std::string policy_date_text(double seconds) {
    if (seconds <= 0) return "";
    const std::time_t t = static_cast<std::time_t>(seconds);
    std::tm tm;
    localtime_r(&t, &tm);
    char buf[16];
    std::strftime(buf, sizeof buf, "%Y-%m-%d", &tm);
    return buf;
}

}  // namespace rmwork
