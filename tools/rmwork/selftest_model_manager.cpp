#include "selftest_model_manager.h"

#include <unistd.h>

#include <string>
#include <vector>

#include "app.h"
#include "files.h"
#include "imgui_internal.h"
#include "model_manager.h"
#include "policy_catalog.h"
#include "selftest_policy_catalog.h"

namespace {

// The engine's manifest in the checks, and a model's that differs from it by `field`.
const char *const NOW_JSON =
    "{\"spec_hash\": \"5555555555555555\", \"vocab_hash\": \"7777777777777777\", \"text_model\": \"m@1\", "
    "\"layout_hash\": \"1111111111111111\", \"tables_digest\": \"dddddddddddddddd\", "
    "\"vocab\": [[0, \"Island\"], [1, \"Lightning Bolt\"]]}";

}  // namespace

static featurize::Manifest now_manifest();
static std::string json_with(const std::string &key, const std::string &value, bool vocab);
static std::string keys_of(const std::vector<rmwork::PolicyRow> &rows, const std::vector<size_t> &order);
static const rmwork::PolicyRow *row_named(const std::vector<rmwork::PolicyRow> &rows, const std::string &key);
static void frames_of(rmwork::App &app, int n);
static bool settle(rmwork::App &app);

static featurize::Manifest now_manifest() {
    featurize::Manifest m;
    std::string err;
    featurize::parse_manifest_json(NOW_JSON, &m, &err);
    return m;
}

// NOW_JSON's fields with `key` set to `value` (none changed for ""), with or without the vocab.
static std::string json_with(const std::string &key, const std::string &value, bool vocab) {
    const featurize::Manifest m = now_manifest();
    const std::pair<const char *, std::string> fields[] = {{"spec_hash", m.spec_hash}, {"vocab_hash", m.vocab_hash},
        {"text_model", m.text_model}, {"layout_hash", m.layout_hash}, {"tables_digest", m.tables_digest}};
    std::string out = "{";
    for (const auto &f : fields)
        out += std::string(out.size() > 1 ? ", " : "") + "\"" + f.first + "\": \"" +
               (key == f.first ? value : f.second) + "\"";
    if (vocab) out += ", \"vocab\": " + featurize::vocab_json(m.vocab);
    return out + "}";
}

// "latest beta/m1 …" of the rows in `order` (the zipped exports of the repository's models/ left out).
static std::string keys_of(const std::vector<rmwork::PolicyRow> &rows, const std::vector<size_t> &order) {
    std::string out;
    for (size_t i : order)
        if (rows[i].archive.empty())
            out += (out.empty() ? "" : " ") + (rows[i].latest ? "latest" : rows[i].run + "/" + rows[i].name);
    return out;
}

static const rmwork::PolicyRow *row_named(const std::vector<rmwork::PolicyRow> &rows, const std::string &key) {
    for (const rmwork::PolicyRow &r : rows)
        if ((r.latest && key == "latest") || (!r.latest && r.run + "/" + r.name == key)) return &r;
    return nullptr;
}

static void frames_of(rmwork::App &app, int n) {
    for (int i = 0; i < n; ++i) {
        app.begin_frame();
        ImGui::NewFrame();
        app.frame();
        ImGui::Render();
    }
}

// Frames until the tool's checker is idle (at most ~5 s); false when it is not.
static bool settle(rmwork::App &app) {
    for (int i = 0; i < 250; ++i) {
        frames_of(app, 1);
        if (!app.model_manager().busy()) {
            frames_of(app, 1);
            return true;
        }
        usleep(20000);
    }
    return false;
}

namespace rmwork {

void check_model_manager_units(const CheckFn &check, const std::string &tmp) {
    const std::string runs = tmp + "/model_manager/runs";
    check(write_policy_fixture(runs), "Models: the fixture cannot be written");
    const std::vector<PolicyRow> rows = scan_policies(runs, "");
    ModelFilter f;
    const std::string by_date = keys_of(rows, model_view(rows, f, ModelColumn::DATE, false));
    check(by_date == "latest beta/m1 alpha/final alpha/m2 alpha/m1 beta/bc-final",
        "Models: the default view is not latest, then newest first: " + by_date);
    const std::string oldest = keys_of(rows, model_view(rows, f, ModelColumn::DATE, true));
    check(oldest == "latest beta/bc-final alpha/m1 alpha/m2 alpha/final beta/m1",
        "Models: date ascending does not keep latest first: " + oldest);
    const std::string by_name = keys_of(rows, model_view(rows, f, ModelColumn::NAME, true));
    check(by_name == "latest beta/bc-final alpha/final beta/m1 alpha/m1 alpha/m2",
        "Models: by name (ties newest first) is not " + by_name);
    const std::string by_steps = keys_of(rows, model_view(rows, f, ModelColumn::STEPS, false));
    check(by_steps.compare(0, 20, "latest alpha/final a") == 0,
        "Models: by learner steps, most first, is not latest then alpha/final: " + by_steps);
    f.text = "M2";
    check(keys_of(rows, model_view(rows, f, ModelColumn::DATE, false)) == "alpha/m2",
        "Models: the filter does not match a name in any case");
    f.text = "beta";
    check(keys_of(rows, model_view(rows, f, ModelColumn::RUN, true)) == "latest beta/m1 beta/bc-final",
        "Models: the filter does not match a run (latest's included)");
    f.text.clear();
    f.run = "alpha";
    check(keys_of(rows, model_view(rows, f, ModelColumn::DATE, false)) == "alpha/final alpha/m2 alpha/m1",
        "Models: the run does not narrow the rows to its own");
    f.show_all = true;
    const std::vector<size_t> all = model_view(rows, f, ModelColumn::DATE, false);
    check(all.size() == 7 && keys_of(rows, all).find("alpha/seed") != std::string::npos &&
              keys_of(rows, all).find("alpha/live") != std::string::npos,
        "Models: Show all does not show the hidden rows: " + keys_of(rows, all));
    check(model_runs(rows) == std::vector<std::string>{"alpha", "beta"}, "Models: the runs listed are not alpha, beta");

    // The verdicts on manifest pairs.
    const featurize::Manifest now = now_manifest();
    const std::string dir = tmp + "/model_manager/manifests";
    make_dirs(dir);
    std::string err;
    featurize::Manifest m;
    const auto verdict_of = [&](const std::string &name, const std::string &text, bool exact) {
        write_file_atomic(dir + "/" + name, text, &err);
        CompatVerdict v;
        if (!read_model_manifest(dir + "/" + name, &m, &v.why)) return v;
        return compat_verdict(m, now, exact);
    };
    CompatVerdict v = verdict_of("same.json", json_with("", "", true), true);
    check(v.compat == PolicyCompat::COMPATIBLE && v.why.empty(),
        "Models: the same manifest is not compatible: " + v.why);
    v = verdict_of("spec.json", json_with("spec_hash", "0000000000000000", true), true);
    check(v.compat == PolicyCompat::INCOMPATIBLE && v.why.find("spec_hash") != std::string::npos,
        "Models: an export of another spec is not refused by its spec_hash: " + v.why);
    v = verdict_of("spec.json", json_with("spec_hash", "0000000000000000", true), false);
    check(v.compat == PolicyCompat::COMPATIBLE, "Models: a checkpoint of another spec is refused: " + v.why);
    v = verdict_of("layout.json", json_with("layout_hash", "0000000000000000", true), false);
    check(v.compat == PolicyCompat::INCOMPATIBLE && v.why.find("layout_hash") != std::string::npos,
        "Models: a checkpoint of another layout is not refused: " + v.why);
    PolicyRow alpha;
    alpha.run = "alpha";
    check(retable_hint(alpha) == "rmtrain retable alpha", "Models: the hint is " + retable_hint(alpha));
    v = verdict_of("latest.json",
        "{\"run\": \"beta\", \"run_dir\": \"/y\", \"manifest\": " + json_with("", "", false) + "}", true);
    check(v.compat == PolicyCompat::COMPATIBLE,
        "Models: latest.json's manifest (no vocab) is not compatible: " + v.why);
    v = verdict_of("latest.json",
        "{\"run\": \"beta\", \"run_dir\": \"/y\", \"manifest\": " + json_with("vocab_hash", "0000000000000000", false) +
            "}",
        true);
    check(v.compat == PolicyCompat::INCOMPATIBLE && v.why.find("vocab_hash") != std::string::npos,
        "Models: latest.json of another vocab is not refused by its vocab_hash: " + v.why);
    v = verdict_of("latest.json", "{\"run\": \"beta\", \"run_dir\": \"/y\", \"manifest\": null}", true);
    check(v.compat == PolicyCompat::UNKNOWN && v.why.find("no manifest") != std::string::npos,
        "Models: latest.json without a manifest is not unknown: " + v.why);

    // The checker over the fixture: alpha's tables of another spec, latest's the engine's.
    write_file_atomic(runs + "/alpha/tables/manifest.json", json_with("spec_hash", "0000000000000000", true), &err);
    write_file_atomic(runs + "/latest/latest.json",
        "{\"run\": \"beta\", \"run_dir\": \"/y\", \"name\": \"m1\", \"learner_steps\": 500, "
        "\"trained_at\": 1700086400, "
        "\"manifest\": " + json_with("", "", false) + "}",
        &err);
    std::vector<PolicyRow> fixture = scan_policies(runs, "");
    CompatChecker checker;
    checker.set_engine_manifest(now);
    EngineTables t;
    t.engine = tmp + "/no-engine";
    check(checker.verdict(*row_named(fixture, "alpha/final")).checking,
        "Models: a row not checked yet is not checking");
    checker.request(t, fixture);
    for (int i = 0; i < 500 && checker.busy(); ++i) {
        checker.poll();
        usleep(10000);
    }
    check(!checker.busy(), "Models: the checker does not finish");
    const CompatVerdict fin = checker.verdict(*row_named(fixture, "alpha/final"));
    const CompatVerdict seed = checker.verdict(*row_named(fixture, "alpha/seed"));
    const CompatVerdict latest = checker.verdict(*row_named(fixture, "latest"));
    const CompatVerdict beta = checker.verdict(*row_named(fixture, "beta/m1"));
    check(fin.compat == PolicyCompat::INCOMPATIBLE && seed.compat == PolicyCompat::COMPATIBLE &&
              latest.compat == PolicyCompat::COMPATIBLE && beta.compat == PolicyCompat::UNKNOWN && !beta.checking,
        "Models: the checker's verdicts are not alpha's exports incompatible, its seed and latest compatible, beta "
        "unknown");
}

void check_model_manager_app(const CheckFn &check, const ConfigPaths &paths, const std::string &tmp) {
    const std::string runs = tmp + "/model_manager_app/runs";
    std::string err;
    check(write_policy_fixture(runs) &&
              write_file_atomic(runs + "/alpha/tables/manifest.json", json_with("layout_hash", "0", true), &err),
        "Models: the App's fixture cannot be written");
    Settings s = default_settings();
    s.models.run_dir = runs;
    s.engine.binary = tmp + "/no-engine";
    s.models.policies.push_back({"0", "/somewhere/p.ts.pt"});
    App app(s, paths, false);
    ModelManagerView &mm = app.model_manager();
    mm.checker().set_engine_manifest(now_manifest());
    app.set_mode(Mode::REPLAY);
    app.set_tool_open(Mode::REPLAY, Tool::MODELS, true);
    app.focus_tool(Tool::MODELS);
    frames_of(app, 3);
    const ImGuiWindow *w = ImGui::FindWindowByName(tool_window_name(Mode::REPLAY, Tool::MODELS).c_str());
    check(w && w->Active, "Models: the tool's window is not drawn");
    check(settle(app), "Models: the App's checker does not finish");
    check(keys_of(mm.rows(), mm.shown()) == "latest beta/m1 alpha/final alpha/m2 alpha/m1 beta/bc-final" &&
              mm.sort_column() == ModelColumn::DATE && !mm.sort_ascending(),
        "Models: the tool does not open on latest, then newest first: " + keys_of(mm.rows(), mm.shown()));
    const PolicyRow *fin = row_named(mm.rows(), "alpha/final");
    const std::string play = fin ? fin->play : "";
    check(fin && fin->compat == PolicyCompat::INCOMPATIBLE, "Models: alpha's final is not shown incompatible");
    const PolicyRow *beta = row_named(mm.rows(), "beta/m1");
    check(beta && beta->compat == PolicyCompat::UNKNOWN, "Models: beta's m1 (no manifest) is not unknown");

    mm.filter().text = "alpha";
    frames_of(app, 2);
    check(keys_of(mm.rows(), mm.shown()) == "alpha/final alpha/m2 alpha/m1", "Models: the filter does not narrow");
    mm.filter().text.clear();
    mm.filter().run = "beta";
    frames_of(app, 2);
    check(keys_of(mm.rows(), mm.shown()) == "latest beta/m1 beta/bc-final", "Models: the run does not narrow");
    mm.filter().run.clear();
    mm.filter().show_all = true;
    frames_of(app, 2);
    size_t shown_runs = 0;
    for (size_t i : mm.shown()) shown_runs += mm.rows()[i].archive.empty();
    check(shown_runs == 10, "Models: Show all does not show every row: " + keys_of(mm.rows(), mm.shown()));
    mm.filter().show_all = false;

    // Opened again: scanned again.
    app.set_tool_open(Mode::REPLAY, Tool::MODELS, false);
    frames_of(app, 2);
    make_dirs(runs + "/gamma/exports");
    write_file_atomic(runs + "/gamma/exports/final.ts.pt", "x", &err);
    app.set_tool_open(Mode::REPLAY, Tool::MODELS, true);
    app.focus_tool(Tool::MODELS);
    frames_of(app, 3);
    check(row_named(mm.rows(), "gamma/final") != nullptr,
        "Models: a run written while closed is not listed on opening: " + keys_of(mm.rows(), mm.shown()));

    // Serve: once, under the next free id; the server turned on.
    check(mm.select_path(play) && mm.selected() && mm.selected()->play == play,
        "Models: alpha's final cannot be selected by its path");
    ModelRequest serve;
    serve.kind = ModelRequest::Kind::SERVE;
    serve.path = play;
    app.run_model_request(serve);
    app.run_model_request(serve);
    const ModelSettings &ms = app.settings().models;
    check(ms.policies.size() == 2 && ms.policies[1].id == "1" && ms.policies[1].model == play && ms.start_server,
        "Models: Serve does not add the policy once as policy 1 and start the server");
    ModelRequest stop;
    stop.kind = ModelRequest::Kind::STOP_SERVER;
    app.run_model_request(stop);
    check(!app.settings().models.start_server, "Models: Stop server does not turn the server off");
    stop.kind = ModelRequest::Kind::START_SERVER;
    app.run_model_request(stop);
    check(app.settings().models.start_server, "Models: Start server does not turn the server on");

    // Play against: New Game with the model as the opponent's.
    ModelRequest against;
    against.kind = ModelRequest::Kind::PLAY_AGAINST;
    against.path = play;
    app.run_model_request(against);
    frames_of(app, 2);
    check(app.new_game_dialog().is_open() && app.new_game_dialog().choices().model == play &&
              app.new_game_dialog().choices().opponent == opponent_key(Opponent::MODEL),
        "Models: Play against does not open New Game with the model as the opponent's");
    ModelRequest settings;
    settings.kind = ModelRequest::Kind::SETTINGS;
    app.run_model_request(settings);
    frames_of(app, 2);
    check(app.settings_open(), "Models: Settings \xE2\x86\x92 Models does not open Settings");
}

}  // namespace rmwork
