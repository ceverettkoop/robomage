#include "parse_report_check.h"

#include <map>
#include <set>

#include "../../src/featurize/json.h"
#include "proc.h"
#include "util.h"

namespace rmcheck {

using CardFindings = std::map<std::string, std::set<std::string>>;

static bool read_report(
    const std::string &out, CardFindings *cards, std::vector<std::string> *missing, std::string *err);
static bool read_baseline(const std::string &path, CardFindings *cards, std::string *err);
static std::string baseline_text(const CardFindings &cards);

void parse_report_part(const Options &opt, Report &rep) {
    ProcSpec spec;
    spec.argv = {opt.binary, "--parse-report", "@vocab"};
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 300;
    const ProcResult r = run_process(spec);
    if (!r.started || (r.exit_code != 0 && r.exit_code != 1)) {
        rep.error("vocab", "robomage --parse-report @vocab " + r.status_text() + ": " + r.err);
        return;
    }
    CardFindings now;
    std::vector<std::string> missing;
    std::string err;
    if (!read_report(r.out, &now, &missing, &err)) {
        rep.error("vocab", "robomage --parse-report @vocab: " + err);
        return;
    }
    for (const std::string &name : missing) rep.error("vocab", "parse report: vocab card " + name + " has no script");
    const std::string path = path_join(opt.repo, "tests/parse_report/vocab.json");
    if (opt.record) {
        if (!write_file(path, baseline_text(now)))
            rep.error("vocab", "parse report: cannot write " + path);
        else
            rep.info(
                "RECORDED parse report: " + std::to_string(now.size()) + " vocab card(s) with findings -> " + path);
        return;
    }
    CardFindings base;
    if (!read_baseline(path, &base, &err)) {
        rep.error("vocab", "parse report: " + path + ": " + err);
        return;
    }
    size_t fresh = 0, gone = 0, total = 0;
    std::set<std::string> names;
    for (const auto &kv : now) names.insert(kv.first);
    for (const auto &kv : base) names.insert(kv.first);
    for (const std::string &name : names) {
        const std::set<std::string> &a = now[name], &b = base[name];
        for (const std::string &f : a) {
            ++total;
            if (b.count(f)) continue;
            ++fresh;
            rep.error("vocab", "parse report: " + name + ": new finding " + f +
                                   " (a parser or evaluator change, or the report's tables; "
                                   "robomage --parse-report \"" +
                                   name + "\")");
        }
        for (const std::string &f : b)
            if (!a.count(f)) {
                ++gone;
                rep.error("vocab", "parse report: " + name + ": finding " + f +
                                       " is gone: rerecord tests/parse_report/vocab.json (robomage_check --tier vocab "
                                       "--record) and update docs/rules_deviations.md");
            }
    }
    if (fresh == 0 && gone == 0 && missing.empty())
        rep.info("OK   parse report: every vocab card parses as tests/parse_report/vocab.json records (" +
                 std::to_string(now.size()) + " card(s), " + std::to_string(total) + " known finding(s))");
}

void onboard_check_part(const Options &opt, Report &rep) {
    if (opt.python.empty()) {
        rep.info("skipped: onboard check (no python3)");
        return;
    }
    ProcSpec spec;
    spec.argv = {opt.python, path_join(opt.repo, "tools/onboard/onboard.py"), "check", "--binary", opt.binary};
    spec.cwd = opt.repo;
    spec.timeout_s = 600;
    const ProcResult r = run_process(spec);
    std::string summary;
    for (const std::string &line : split_lines(r.out)) {
        if (starts_with(line, "onboard check: STALE ")) rep.warn("vocab", line.substr(15));
        if (starts_with(line, "onboard check: ERROR ")) rep.error("vocab", line.substr(15));
        if (starts_with(line, "onboard check: ") && contains(line, " candidate(s), ")) summary = line.substr(15);
    }
    if (!r.ok())
        rep.error("vocab", "tools/onboard/onboard.py check " + r.status_text() + ":\n" + r.out + r.err);
    else
        rep.info("OK   onboard: " + summary);
}

// The report's JSON lines as card -> {kind:name}, the cards with findings only; `missing` the names
// with no script.
static bool read_report(
    const std::string &out, CardFindings *cards, std::vector<std::string> *missing, std::string *err) {
    for (const std::string &line : split_lines(out)) {
        if (trim(line).empty()) continue;
        featurize::Json j;
        if (!featurize::Json::parse(line, &j, err)) return false;
        const featurize::Json *card = j.get("card"), *findings = j.get("findings");
        if (!card || card->is_null()) {
            const featurize::Json *q = j.get("query");
            missing->push_back(q ? q->as_string() : "?");
            continue;
        }
        if (!findings) continue;
        for (const featurize::Json &f : findings->items())
            (*cards)[card->as_string()].insert(f.get("kind")->as_string() + ":" + f.get("name")->as_string());
    }
    return true;
}

static bool read_baseline(const std::string &path, CardFindings *cards, std::string *err) {
    std::string text;
    if (!read_file(path, &text)) {
        *err = "cannot read it (robomage_check --tier vocab --record writes it)";
        return false;
    }
    featurize::Json j;
    if (!featurize::Json::parse(text, &j, err)) return false;
    const featurize::Json *c = j.get("cards");
    if (!c || !c->is_object()) {
        *err = "no \"cards\" object";
        return false;
    }
    for (const auto &kv : c->members())
        for (const featurize::Json &f : kv.second.items()) (*cards)[kv.first].insert(f.as_string());
    return true;
}

static std::string baseline_text(const CardFindings &cards) {
    std::string out =
        "{\"_doc\": \"robomage --parse-report @vocab: the vocab cards' known parser gaps (kind:name; "
        "src/parse_report.h), checked by the vocab tier (tools/robomage_check/parse_report_check.h); "
        "robomage_check --tier vocab --record rewrites it. Each is listed in docs/rules_deviations.md.\",\n \"cards\": "
        "{";
    bool first = true;
    for (const auto &kv : cards) {
        if (kv.second.empty()) continue;
        out += std::string(first ? "\n" : ",\n") + "  " + featurize::json_quote(kv.first) + ": [";
        bool f1 = true;
        for (const std::string &f : kv.second) {
            out += std::string(f1 ? "" : ", ") + featurize::json_quote(f);
            f1 = false;
        }
        out += "]";
        first = false;
    }
    return out + "\n }\n}\n";
}

}  // namespace rmcheck
