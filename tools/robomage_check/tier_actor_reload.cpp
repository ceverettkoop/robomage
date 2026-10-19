#include <cstdio>
#include <map>

#include "../../src/featurize/json.h"
#include "actor.h"
#include "proc.h"
#include "scan.h"
#include "traj.h"
#include "util.h"

using featurize::Json;

static bool versioned_copy(
    const rmcheck::ActorSetup &a, const std::string &from, const std::string &to, int version, std::string *err);
static std::string versions_of(const Json &line);

// The reference policy's TorchScript export saved again with policy_version `version`.
static bool versioned_copy(
    const rmcheck::ActorSetup &a, const std::string &from, const std::string &to, int version, std::string *err) {
    rmcheck::ProcSpec spec;
    spec.argv = {a.python, "-c",
        "import sys, torch\n"
        "m = torch.jit.load(sys.argv[1])\n"
        "m.policy_version = sys.argv[3]\n"
        "torch.jit.save(m, sys.argv[2])\n",
        from, to, std::to_string(version)};
    spec.env = {"HIP_VISIBLE_DEVICES=-1", "CUDA_VISIBLE_DEVICES=-1"};
    spec.merge_stderr = true;
    spec.timeout_s = 300;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    if (r.ok()) return true;
    *err = "saving a copy with policy_version " + std::to_string(version) + " " + r.status_text() + ": " +
           r.out.substr(0, 400);
    return false;
}

// A results line's policy_versions_a as text ("3", "3,5").
static std::string versions_of(const Json &line) {
    std::string out;
    if (const Json *arr = line.get("policy_versions_a"))
        for (const Json &x : arr->items()) out += (out.empty() ? "" : ",") + x.text();
    return out;
}

namespace rmcheck {

// An actor (robomage_nn --actor) whose entry plays nn:<model> (in process) against explore over
// three bo1 matches; after match 0 the model file is replaced (renamed over) by a copy with another
// policy_version: match 0 records the first version, match 2 the second, no match mixes them, the
// episodes' policy versions agree, the status counts both and the agent's NN-STATS line one reload.
void actor_reload_check(const Options &opt, Report &rep, const ActorSetup &a) {
    const std::string dir = path_join(a.dir, "reload");
    remove_tree(dir);
    make_dirs(dir);
    const std::string model = path_join(dir, "model.ts.pt"), next = path_join(dir, "v5.ts.pt");
    std::string err;
    if (!versioned_copy(a, a.ts(), model, 3, &err) || !versioned_copy(a, a.ts(), next, 5, &err)) {
        rep.error("actor", "hot reload: " + err);
        return;
    }
    const std::string schedule = path_join(dir, "schedule.json"), out = path_join(dir, "run");
    write_file(schedule,
        "{\"entries\": [{\"id\": \"nn\", \"format\": \"bo1\", \"deck_a\": \"league/burn\", "
        "\"deck_b\": \"league/bug\", \"agent_a\": " +
            featurize::json_quote("nn:" + model + ":stats") + ", \"agent_b\": \"explore\"}]}\n");
    ProcSpec spec;
    spec.argv = {a.nn_binary, "--actor", "--actor-schedule", schedule, "--actor-out", out, "--actor-seed",
        std::to_string(opt.seed + 940), "--actor-matches", "3"};
    spec.cwd = opt.bin_dir;
    spec.env = {"HIP_VISIBLE_DEVICES=-1", "CUDA_VISIBLE_DEVICES=-1"};
    Session s;
    if (!s.start(spec, &err)) {
        rep.error("actor", "hot reload: the actor does not start: " + err);
        return;
    }
    s.set_timeout(1800);
    std::string text;
    bool replaced = false;
    while (s.fill_line()) {
        const size_t nl = s.pending().find('\n');
        const std::string line = s.pending().substr(0, nl);
        s.take(nl + 1);
        text += line + "\n";
        if (!replaced && starts_with(line, "GAME_RESULT:")) replaced = std::rename(next.c_str(), model.c_str()) == 0;
    }
    ProcResult r = s.finish();
    r.out = text + r.out;
    write_file(out + ".txt", r.out + "\n=== engine stderr ===\n" + r.err);
    std::vector<std::string> fails;
    std::vector<ScanLine> errors;
    scan_transcript(r.out + "\n" + r.err, &errors, nullptr);
    if (!r.ok()) fails.push_back("the actor " + r.status_text());
    for (const ScanLine &l : errors) fails.push_back("an error line: " + l.line);
    if (!replaced) fails.push_back("the model file was not replaced");
    std::vector<Json> lines;
    Json status;
    if (!read_file(path_join(out, "a0.results.jsonl"), &text)) fails.push_back("no results file");
    for (const std::string &l : split_lines(text)) {
        Json j;
        if (Json::parse(l, &j, &err)) lines.push_back(j);
    }
    if (lines.size() != 3) fails.push_back(std::to_string(lines.size()) + " results lines, not 3");
    if (lines.size() == 3) {
        if (versions_of(lines[0]) != "3") fails.push_back("match 0 recorded versions " + versions_of(lines[0]));
        if (versions_of(lines[2]) != "5") fails.push_back("match 2 recorded versions " + versions_of(lines[2]));
        if (versions_of(lines[1]) != "3" && versions_of(lines[1]) != "5")
            fails.push_back("match 1 recorded versions " + versions_of(lines[1]));
    }
    std::vector<EpisodeFile> eps;
    if (!read_episodes(path_join(out, "train/traj"), &eps, &err)) fails.push_back(err);
    std::map<std::string, std::string> ep_versions;
    for (const EpisodeFile &e : eps)
        if (const featurize::RmtbAttr *v = e.ep.file.attr("policy_versions_a")) ep_versions[e.path] = v->s;
    for (const Json &l : lines) {
        const Json *ep = l.get("episode");
        if (!ep || !ep->is_string() || ep_versions[ep->as_string()] != versions_of(l))
            fails.push_back("an episode's policy_versions_a is not its results line's");
    }
    if (!read_file(path_join(out, "a0.status.json"), &text) || !Json::parse(text, &status, &err) ||
        !status.get("policy_versions") || status.get("policy_versions")->get("count") == nullptr ||
        status.get("policy_versions")->get("count")->as_int() != 2 ||
        status.get("policy_versions")->get("max")->as_int() != 5)
        fails.push_back("the status does not count versions 3 and 5");
    if (!contains(r.out, "reloads=1")) fails.push_back("the NN-STATS line does not show one reload");
    for (size_t i = 0; i < fails.size() && i < 20; ++i) rep.error("actor", "hot reload: " + fails[i]);
    if (fails.empty())
        rep.info("hot reload: robomage_nn --actor reloaded nn:<model> between matches (versions " +
                 versions_of(lines[0]) + ", " + versions_of(lines[1]) + ", " + versions_of(lines[2]) + ")");
}

}  // namespace rmcheck
