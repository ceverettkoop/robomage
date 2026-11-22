#include "model_manager.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <set>

#include "../../src/featurize/json.h"
#include "../../src/featurize/setup.h"
#include "eval_connection.h"
#include "file_picker.h"
#include "files.h"

namespace {

const char *const FIXED_ENGINE = "(given)";

}  // namespace

static std::string lower(const std::string &s);
static bool row_matches(const rmwork::PolicyRow &r, const rmwork::ModelFilter &f, const std::string &needle);
static int compat_rank(rmwork::PolicyCompat c);
static int compare_rows(const rmwork::PolicyRow &a, const rmwork::PolicyRow &b, rmwork::ModelColumn column);
static bool string_field(const featurize::Json &o, const char *key, std::string *out, std::string *why);
static bool manifest_from(const featurize::Json &o, featurize::Manifest *out, std::string *why);
static std::string engine_key_of(const std::string &engine);
static bool engine_manifest(const rmwork::EngineTables &t, featurize::Manifest *out, std::string *why);

static std::string lower(const std::string &s) {
    std::string out = s;
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// `needle` is the filter's text in lower case.
static bool row_matches(const rmwork::PolicyRow &r, const rmwork::ModelFilter &f, const std::string &needle) {
    if (r.hidden && !f.show_all) return false;
    if (!f.run.empty() && r.run != f.run) return false;
    return needle.empty() || lower(r.run).find(needle) != std::string::npos ||
           lower(r.name).find(needle) != std::string::npos;
}

static int compat_rank(rmwork::PolicyCompat c) {
    switch (c) {
        case rmwork::PolicyCompat::COMPATIBLE:
            return 0;
        case rmwork::PolicyCompat::UNKNOWN:
            return 1;
        case rmwork::PolicyCompat::INCOMPATIBLE:
            return 2;
    }
    return 1;
}

// <0, 0, >0 as `a` sorts before, with or after `b` by `column` ascending.
static int compare_rows(const rmwork::PolicyRow &a, const rmwork::PolicyRow &b, rmwork::ModelColumn column) {
    switch (column) {
        case rmwork::ModelColumn::DATE:
            return a.trained_at < b.trained_at ? -1 : a.trained_at > b.trained_at ? 1 : 0;
        case rmwork::ModelColumn::RUN:
            return a.run == b.run ? 0 : rmwork::natural_less(a.run, b.run) ? -1 : 1;
        case rmwork::ModelColumn::NAME:
            return a.name == b.name ? 0 : rmwork::natural_less(a.name, b.name) ? -1 : 1;
        case rmwork::ModelColumn::STEPS:
            return a.learner_steps < b.learner_steps ? -1 : a.learner_steps > b.learner_steps ? 1 : 0;
        case rmwork::ModelColumn::FORMATS: {
            const std::string x = rmwork::policy_formats_text(a.formats), y = rmwork::policy_formats_text(b.formats);
            return x < y ? -1 : x > y ? 1 : 0;
        }
        case rmwork::ModelColumn::COMPAT:
            return compat_rank(a.compat) - compat_rank(b.compat);
    }
    return 0;
}

static bool string_field(const featurize::Json &o, const char *key, std::string *out, std::string *why) {
    const featurize::Json *v = o.get(key);
    if (!v || v->is_null()) return true;
    if (!v->is_string()) {
        *why = std::string("the manifest's ") + key + " is not a string";
        return false;
    }
    *out = v->as_string();
    return true;
}

// The manifest an object holds (compat.h's keys; a missing one left empty).
static bool manifest_from(const featurize::Json &o, featurize::Manifest *out, std::string *why) {
    *out = featurize::Manifest();
    for (const auto &f : {std::make_pair("spec_hash", &out->spec_hash), std::make_pair("vocab_hash", &out->vocab_hash),
             std::make_pair("text_model", &out->text_model), std::make_pair("layout_hash", &out->layout_hash),
             std::make_pair("tables_digest", &out->tables_digest), std::make_pair("registry", &out->registry)})
        if (!string_field(o, f.first, f.second, why)) return false;
    const featurize::Json *v = o.get("vocab");
    return !v || featurize::vocab_from_json(*v, &out->vocab, why);
}

// The engine binary's path and modification time.
static std::string engine_key_of(const std::string &engine) {
    return engine + '\n' + std::to_string(rmwork::file_stamp(engine).mtime_ns);
}

// The engine's tables' manifest: its cards file written, then loaded with the text table.
static bool engine_manifest(const rmwork::EngineTables &t, featurize::Manifest *out, std::string *why) {
    if (!rmwork::is_executable(t.engine)) {
        *why = "no engine at " + t.engine;
        return false;
    }
    featurize::Featurization features;
    std::string err;
    const bool loaded = rmwork::load_engine_features(t.engine, t.workdir, t.cards_path, nullptr, &features, &err);
    unlink(t.cards_path.c_str());
    unlink((t.cards_path + ".err").c_str());
    if (!loaded) {
        *why = "the engine's tables: " + err;
        return false;
    }
    *out = features.manifest();
    return true;
}

namespace rmwork {

std::vector<size_t> model_view(
    const std::vector<PolicyRow> &rows, const ModelFilter &f, ModelColumn column, bool ascending) {
    const std::string needle = lower(f.text);
    std::vector<size_t> out;
    for (size_t i = 0; i < rows.size(); ++i)
        if (row_matches(rows[i], f, needle)) out.push_back(i);
    std::stable_sort(out.begin(), out.end(), [&](size_t x, size_t y) {
        const PolicyRow &a = rows[x], &b = rows[y];
        if (a.latest != b.latest) return a.latest;
        const int c = compare_rows(a, b, column);
        if (c != 0) return ascending ? c < 0 : c > 0;
        return a.trained_at > b.trained_at;
    });
    return out;
}

std::vector<std::string> model_runs(const std::vector<PolicyRow> &rows) {
    std::set<std::string> seen;
    std::vector<std::string> out;
    for (const PolicyRow &r : rows)
        if (!r.run.empty() && seen.insert(r.run).second) out.push_back(r.run);
    std::sort(out.begin(), out.end(), natural_less);
    return out;
}

bool read_model_manifest(const std::string &path, featurize::Manifest *out, std::string *why) {
    std::string text;
    if (!read_file(path, &text)) {
        *why = "cannot read " + path;
        return false;
    }
    featurize::Json doc;
    std::string err;
    if (!featurize::Json::parse(text, &doc, &err) || !doc.is_object()) {
        *why = path + " is not a JSON object" + (err.empty() ? "" : ": " + err);
        return false;
    }
    // latest.json names its run directory and holds the manifest as a member.
    const featurize::Json *inner = doc.get("manifest");
    const bool latest = inner || doc.has("run_dir");
    if (latest && (!inner || !inner->is_object())) {
        *why = "latest.json carries no manifest (its tables were exported before model manifests)";
        return false;
    }
    if (manifest_from(latest ? *inner : doc, out, &err)) return true;
    *why = path + ": " + err;
    return false;
}

CompatVerdict compat_verdict(featurize::Manifest model, const featurize::Manifest &now, bool exact) {
    CompatVerdict v;
    if (model.vocab.empty() && !model.vocab_hash.empty()) {
        if (model.vocab_hash == now.vocab_hash) {
            model.vocab = now.vocab;
        } else if (exact) {
            v.compat = PolicyCompat::INCOMPATIBLE;
            v.why = "its vocab_hash " + model.vocab_hash + " is not the engine's " + now.vocab_hash;
            return v;
        }
    }
    const featurize::Refusal r = featurize::check_compatible(model, now, exact);
    v.compat = r.refused() ? PolicyCompat::INCOMPATIBLE : PolicyCompat::COMPATIBLE;
    v.why = r.message;
    return v;
}

std::string retable_hint(const PolicyRow &row) {
    return "rmtrain retable " + (row.run.empty() ? std::string("<run>") : row.run);
}

CompatChecker::~CompatChecker() {
    if (running_ && running_->thread.joinable()) running_->thread.join();
}

std::unique_ptr<CompatChecker::Job> CompatChecker::make_job(
    const EngineTables &e, const std::vector<PolicyRow> &rows) const {
    std::unique_ptr<Job> job(new Job());
    job->tables = e;
    job->engine_key = fixed_now_ ? engine_key_ : engine_key_of(e.engine);
    job->need_engine = !fixed_now_ && job->engine_key != engine_key_;
    if (!job->need_engine && !have_now_) return nullptr;
    job->now = now_;
    std::set<std::string> seen;
    for (const PolicyRow &r : rows) {
        if (r.tables_manifest.empty() || !seen.insert(r.tables_manifest).second) continue;
        const int64_t t = file_stamp(r.tables_manifest).mtime_ns;
        const auto it = verdicts_.find(r.tables_manifest);
        if (job->need_engine || it == verdicts_.end() || it->second.mtime_ns != t)
            job->files.emplace_back(r.tables_manifest, t);
    }
    if (!job->need_engine && job->files.empty()) return nullptr;
    return job;
}

void CompatChecker::request(const EngineTables &e, const std::vector<PolicyRow> &rows) {
    std::unique_ptr<Job> job = make_job(e, rows);
    if (!job) return;
    if (running_) {
        waiting_ = std::move(job);
        return;
    }
    start(std::move(job));
}

void CompatChecker::start(std::unique_ptr<Job> job) {
    running_ = std::move(job);
    Job *j = running_.get();
    j->thread = std::thread([j] { run(j); });
}

void CompatChecker::run(Job *job) {
    if (job->need_engine) {
        job->now_ok = engine_manifest(job->tables, &job->now, &job->now_error);
        if (!job->now_ok) {
            job->done.store(true);
            return;
        }
    }
    for (const auto &f : job->files) {
        FileVerdicts &fv = job->verdicts[f.first];
        fv.mtime_ns = f.second;
        featurize::Manifest model;
        fv.ok = read_model_manifest(f.first, &model, &fv.error);
        if (!fv.ok) continue;
        fv.exact = compat_verdict(model, job->now, true);
        fv.loose = compat_verdict(model, job->now, false);
    }
    job->done.store(true);
}

bool CompatChecker::poll() {
    if (!running_ || !running_->done.load()) return false;
    running_->thread.join();
    std::unique_ptr<Job> job = std::move(running_);
    if (job->need_engine) {
        engine_key_ = job->engine_key;
        have_now_ = job->now_ok;
        now_ = job->now_ok ? job->now : featurize::Manifest();
        engine_error_ = job->now_error;
        verdicts_ = job->verdicts;
    } else if (job->engine_key == engine_key_) {
        for (auto &kv : job->verdicts) verdicts_[kv.first] = kv.second;
    }
    if (waiting_) {
        // Made before these results: the engine's manifest it was to read may be the one just read.
        if (waiting_->need_engine && waiting_->engine_key == engine_key_) {
            waiting_->need_engine = false;
            waiting_->now = now_;
        }
        if (waiting_->need_engine || have_now_) start(std::move(waiting_));
        waiting_.reset();
    }
    return true;
}

void CompatChecker::forget_files() {
    verdicts_.clear();
    if (!have_now_ && !fixed_now_) engine_key_.clear();
}

CompatVerdict CompatChecker::verdict(const PolicyRow &row) const {
    CompatVerdict v;
    if (row.tables_manifest.empty()) {
        v.why = row.archive.empty() ? "no tables manifest beside it (tables/manifest.json, latest.json)"
                                    : "a zipped export carries no tables manifest";
        return v;
    }
    if (!have_now_) {
        v.checking = engine_error_.empty();
        v.why = v.checking ? "reading the engine's tables" : engine_error_;
        return v;
    }
    const auto it = verdicts_.find(row.tables_manifest);
    if (it == verdicts_.end()) {
        v.checking = true;
        v.why = "reading " + row.tables_manifest;
        return v;
    }
    if (!it->second.ok) {
        v.why = it->second.error;
        return v;
    }
    return row.play.empty() ? it->second.loose : it->second.exact;
}

void CompatChecker::set_engine_manifest(const featurize::Manifest &now) {
    fixed_now_ = true;
    have_now_ = true;
    now_ = now;
    engine_key_ = FIXED_ENGINE;
    engine_error_.clear();
    verdicts_.clear();
}

}  // namespace rmwork
