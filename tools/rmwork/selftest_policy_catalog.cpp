#include "selftest_policy_catalog.h"

#include <sys/time.h>

#include <cstdio>
#include <set>
#include <vector>

#include "files.h"
#include "new_game_dialog.h"
#include "policy_catalog.h"
#include "settings.h"

namespace {

// The fixture's base time (2023-11-14) and the files' modification time, newer than every metrics date.
const double T0 = 1700000000.0;
const double FILE_TIME = T0 + 30 * 86400.0;

}  // namespace

static bool put(const std::string &path, const std::string &data, double mtime);
static std::string metrics_line(double time, const std::string &event, const std::string &fields);
static const rmwork::PolicyRow *find_row(
    const std::vector<rmwork::PolicyRow> &rows, const std::string &run, const std::string &name);
static std::string row_keys(const std::vector<rmwork::PolicyRow> &rows, bool hidden);

// `data` written at `path` (its directory made) with modification time `mtime` (0: now).
static bool put(const std::string &path, const std::string &data, double mtime) {
    if (!rmwork::make_dirs(rmwork::parent_dir(path))) return false;
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    std::fclose(f);
    if (!ok || mtime == 0) return ok;
    struct timeval tv[2];
    tv[0].tv_sec = tv[1].tv_sec = static_cast<time_t>(mtime);
    tv[0].tv_usec = tv[1].tv_usec = 0;
    return utimes(path.c_str(), tv) == 0;
}

// A metrics.jsonl line as rmtrain's emit writes it.
static std::string metrics_line(double time, const std::string &event, const std::string &fields) {
    char head[160];
    std::snprintf(
        head, sizeof head, "{\"time\": %.3f, \"phase\": \"B4\", \"stage\": \"selfplay\", \"step\": 1, ", time);
    return head + std::string("\"event\": \"") + event + "\"" + (fields.empty() ? "" : ", " + fields) + "}\n";
}

static const rmwork::PolicyRow *find_row(
    const std::vector<rmwork::PolicyRow> &rows, const std::string &run, const std::string &name) {
    for (const rmwork::PolicyRow &r : rows)
        if (!r.latest && r.archive.empty() && r.run == run && r.name == name) return &r;
    return nullptr;
}

// "run/name" of the runs' rows that are (`hidden`) or are not hidden, in order, space-separated.
static std::string row_keys(const std::vector<rmwork::PolicyRow> &rows, bool hidden) {
    std::string out;
    for (const rmwork::PolicyRow &r : rows)
        if (!r.latest && r.archive.empty() && r.hidden == hidden)
            out += (out.empty() ? "" : " ") + r.run + "/" + r.name;
    return out;
}

namespace rmwork {

bool write_policy_fixture(const std::string &runs) {
    const std::string a = runs + "/alpha", b = runs + "/beta";
    bool ok = true;
    // alpha: trained first (its metrics), its files the newest.
    std::string m = metrics_line(T0 + 10, "rnad_step", "\"learner_steps\": 10, \"loss\": 0.5");
    m += metrics_line(T0 + 20, "checkpoint",
        "\"kind\": \"rnad\", \"learner_steps\": 0, \"version\": 0, \"path\": \"/old/place/alpha/ckpt/rnad-seed.pt\", "
        "\"at\": " + std::to_string(T0 + 20));
    m += metrics_line(T0 + 100, "rnad_iteration",
        "\"reg_updates\": 1, \"learner_steps\": 1000, \"version\": 1000, \"fixed_point\": \"/x/ckpt/fixed/m1.pt\"");
    m += metrics_line(T0 + 150, "export", "\"name\": \"m1\", \"version\": 1000, \"seconds\": 4.0");
    m += metrics_line(T0 + 200, "rnad_iteration",
        "\"reg_updates\": 2, \"learner_steps\": 2000, \"version\": 2000, \"fixed_point\": \"/x/ckpt/fixed/m2.pt\"");
    m += metrics_line(T0 + 250, "checkpoint",
        "\"kind\": \"rnad\", \"learner_steps\": 2400, \"version\": 2400, \"path\": \"/x/ckpt/rnad-p2400.pt\"");
    m += metrics_line(T0 + 300, "checkpoint",
        "\"kind\": \"rnad\", \"learner_steps\": 2500, \"version\": 2500, \"path\": \"/x/ckpt/rnad-final.pt\"");
    m += metrics_line(T0 + 310, "export", "\"name\": \"final\", \"version\": 2500, \"seconds\": 4.0");
    // a retable's export made again, later: the policy keeps its first date
    m += metrics_line(T0 + 20 * 86400.0, "export", "\"name\": \"final\", \"version\": 2500, \"seconds\": 4.0");
    ok &= put(a + "/metrics.jsonl", m, 0);
    for (const char *f : {"ckpt/fixed/m1.pt", "ckpt/fixed/m2.pt", "ckpt/rnad-final.pt", "ckpt/rnad-p2400.pt",
             "ckpt/rnad-seed.pt", "ckpt/bc-p100.pt", "exports/m1.ts.pt", "exports/m1.pt2", "exports/m2.ts.pt",
             "exports/final.ts.pt", "exports/final.pt2", "exports/final.onnx", "exports/live/current.ts.pt",
             "ckpt.v1/fixed/m1.pt", "ckpt.layout1/rnad-final.pt", "exports.v1/m1.ts.pt", "exports.layout1/final.pt2",
             "exports/eval/it1-v1000.ts.pt", "exports/m2.pt2.tmp123.4", "tables/manifest.json"})
        ok &= put(a + "/" + f, "x", FILE_TIME);
    // beta: trained after alpha; bc-final named by no metrics line, its file the oldest; a torn last line.
    std::string mb = metrics_line(T0 + 86400, "rnad_iteration",
        "\"reg_updates\": 1, \"learner_steps\": 500, \"version\": 500, \"fixed_point\": \"/y/ckpt/fixed/m1.pt\"");
    mb += "{\"time\": 1700090000.0, \"event\": \"rnad_iteration\", \"reg_upd";
    ok &= put(b + "/metrics.jsonl", mb, 0);
    for (const char *f : {"ckpt/fixed/m1.pt", "exports/m1.ts.pt"}) ok &= put(b + "/" + f, "x", FILE_TIME);
    ok &= put(b + "/exports/bc-final.pt2", "x", T0 + 50);
    // latest/: beta's m1
    ok &= put(runs + "/latest/latest.json",
        "{\"run\": \"beta\", \"run_dir\": \"/y\", \"name\": \"m1\", \"learner_steps\": 500, \"version\": 500, "
        "\"trained_at\": " + std::to_string(T0 + 86400) + ", \"written_at\": 1, \"files\": {}}\n", 0);
    ok &= put(runs + "/latest/policy.ts.pt", "x", 0) && put(runs + "/latest/model.pt", "x", 0);
    return ok;
}

void check_policy_catalog_units(const CheckFn &check, const std::string &tmp) {
    const std::string runs = tmp + "/policy_catalog/runs", a = runs + "/alpha", b = runs + "/beta";
    const bool ok = write_policy_fixture(runs);
    check(ok, "Policy catalog: the fixture cannot be written");

    std::vector<PolicyRow> rows = scan_policies(runs, "");
    check(!rows.empty() && rows[0].latest && rows[0].run == "beta" && rows[0].name == "m1" &&
              rows[0].learner_steps == 500 && rows[0].dated_by == PolicyDateSource::LATEST &&
              rows[0].play == runs + "/latest/policy.ts.pt" &&
              rows[0].formats == (FORMAT_TS | FORMAT_CHECKPOINT) && rows[0].label == "latest \xC2\xB7 beta m1",
        "Policy catalog: latest is not the pinned first row with latest.json's run, name and steps (" +
            (rows.empty() ? std::string("none") : rows[0].label) + ")");
    int latest_rows = 0;
    for (const PolicyRow &r : rows) latest_rows += r.latest || r.run == "latest";
    check(latest_rows == 1, "Policy catalog: latest/ is listed as a run");
    check(row_keys(rows, false) == "beta/m1 alpha/final alpha/m2 alpha/m1 beta/bc-final",
        "Policy catalog: the shown rows are not newest first by training date: " + row_keys(rows, false));
    const std::set<std::string> hidden = {"alpha/live", "alpha/rnad-p2400", "alpha/seed", "alpha/bc-p100"};
    std::set<std::string> got;
    for (const PolicyRow &r : rows)
        if (r.hidden) got.insert(r.run + "/" + r.name);
    check(got == hidden, "Policy catalog: the hidden rows are " + row_keys(rows, true));
    bool never = true;
    for (const PolicyRow &r : rows)
        for (const std::string *p : {&r.ts, &r.pt2, &r.onnx, &r.checkpoint})
            never &= p->find(".v1/") == std::string::npos && p->find(".layout1/") == std::string::npos &&
                     p->find("/eval/") == std::string::npos && p->find(".tmp") == std::string::npos;
    check(never, "Policy catalog: an old layout's copy, an evaluation export or a temporary file is listed");

    const PolicyRow *fin = find_row(rows, "alpha", "final");
    check(fin && fin->formats == (FORMAT_TS | FORMAT_PT2 | FORMAT_ONNX | FORMAT_CHECKPOINT) &&
              fin->play == a + "/exports/final.ts.pt" && fin->checkpoint == a + "/ckpt/rnad-final.pt" &&
              fin->learner_steps == 2500 && fin->dated_by == PolicyDateSource::METRICS && fin->trained_at == T0 + 300 &&
              fin->tables_manifest == a + "/tables/manifest.json" &&
              fin->label == policy_date_text(T0 + 300) + " \xC2\xB7 alpha/final" &&
              policy_formats_text(fin->formats) == "ts, pt2, onnx, checkpoint",
        "Policy catalog: alpha's final is not one row of its exports and checkpoint, dated by its checkpoint line");
    const PolicyRow *m1 = find_row(rows, "alpha", "m1");
    check(m1 && m1->formats == (FORMAT_TS | FORMAT_PT2 | FORMAT_CHECKPOINT) && m1->learner_steps == 1000 &&
              m1->trained_at == T0 + 100 && m1->checkpoint == a + "/ckpt/fixed/m1.pt",
        "Policy catalog: alpha's m1 is not its fixed point and exports, dated by its rnad_iteration line");
    const PolicyRow *seed = find_row(rows, "alpha", "seed");
    check(seed && seed->play.empty() && seed->trained_at == T0 + 20,
        "Policy catalog: the seed is not a checkpoint-only row dated by its checkpoint line");
    const PolicyRow *live = find_row(rows, "alpha", "live");
    check(live && live->dated_by == PolicyDateSource::MTIME && live->play == a + "/exports/live/current.ts.pt",
        "Policy catalog: the live export is not dated by its file");
    const PolicyRow *bc = find_row(rows, "beta", "bc-final");
    check(bc && bc->dated_by == PolicyDateSource::MTIME && bc->trained_at == T0 + 50 &&
              bc->play == b + "/exports/bc-final.pt2" && bc->learner_steps == -1 && bc->tables_manifest.empty(),
        "Policy catalog: a policy no metrics line names is not dated by its file's modification time");

    // the metrics file grows: read from where the last read stopped, the torn line completed
    FILE *f = std::fopen((b + "/metrics.jsonl").c_str(), "ab");
    if (f) {
        std::fputs("dates\": 9}\n", f);
        const std::string more = metrics_line(T0 + 60, "export", "\"name\": \"bc-final\", \"version\": 7");
        std::fputs(more.c_str(), f);
        std::fclose(f);
    }
    rows = scan_policies(runs, "");
    bc = find_row(rows, "beta", "bc-final");
    check(f && bc && bc->dated_by == PolicyDateSource::METRICS && bc->trained_at == T0 + 60,
        "Policy catalog: a metrics line appended after the first scan is not read");

    // New Game's list and the Add row's suggestions
    Settings s = default_settings();
    s.models.run_dir = runs;
    s.models.policies.push_back({"0", "/somewhere/p.ts.pt"});
    const std::vector<ModelChoice> models = NewGameDialog::list_models(s), exports = NewGameDialog::list_exports(s);
    check(models.size() >= 7 && models[0].latest && models[0].path == runs + "/latest/policy.ts.pt" &&
              models[1].label.compare(0, 8, "policy 0") == 0 && models[2].label.find("beta/m1") != std::string::npos &&
              models[3].path == a + "/exports/final.ts.pt",
        "Policy catalog: New Game's models are not latest, the policies, then the runs' newest first");
    bool shown_only = true;
    for (const ModelChoice &c : exports)
        shown_only &= c.path.find("/live/") == std::string::npos && c.path.find("/ckpt/") == std::string::npos;
    check(exports.size() >= 6 && exports[0].latest && shown_only,
        "Policy catalog: the Add row's suggestions list a hidden or checkpoint-only row");
    NewGameDialog dialog;
    NewGameSettings g;
    g.model.clear();
    dialog.open(g, s, "");
    check(dialog.choices().model == runs + "/latest/policy.ts.pt",
        "Policy catalog: New Game's default model is not latest");
    g.model = a + "/exports/m1.pt2";
    dialog.open(g, s, "");
    check(dialog.choices().model == g.model, "Policy catalog: New Game drops the model chosen before");
}

}  // namespace rmwork
