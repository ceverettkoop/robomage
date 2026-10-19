#include <signal.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>

#include "../../src/featurize/json.h"
#include "actor.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "traj.h"
#include "util.h"

using featurize::Json;

namespace {

// One actor process's run: its directory, its result and the files it wrote.
struct ActorRun {
        std::string dir;  // <tier out>/<part>
        std::string out;  // the actor's --actor-out
        std::string schedule;
        std::string binary;  // the engine ("" = the gate's; robomage_nn for nn: agents)
        rmcheck::ProcResult result;
        std::vector<Json> lines;  // the results file's lines, parsed
        Json status;
};

const size_t MAX_FAILS = 20;
const char *const TIER = "actorloop";

}  // namespace

static std::vector<std::string> actor_args(const rmcheck::Options &opt, const ActorRun &a, long matches, long seed);
static bool write_schedule(const std::string &path, const std::string &text);
static void run_actor(const rmcheck::Options &opt, ActorRun *a, long matches, long seed);
static bool read_outputs(ActorRun *a, std::vector<std::string> *fails);
static std::string jstr(const Json &o, const char *key);
static int64_t jint(const Json &o, const char *key);
static std::string canonical(const Json &v);
static std::string versions_text(const Json &arr);
static std::string stem_of(const std::string &rmlog);
static bool episodes_by_stem(
    const std::string &dir, std::map<std::string, std::vector<rmcheck::EpisodeFile>> *out, std::string *err);
static void check_run(const ActorRun &a, long matches, const char *state, std::vector<std::string> *fails);
static void check_footers(const ActorRun &a, const std::map<std::string, std::vector<rmcheck::EpisodeFile>> &eps,
    std::vector<std::string> *fails);
static void check_replays(const rmcheck::Options &opt, const ActorRun &a,
    const std::map<std::string, std::vector<rmcheck::EpisodeFile>> &eps, const std::string &teacher,
    std::vector<std::string> *fails);
static void report(
    rmcheck::Report &rep, const char *part, const std::vector<std::string> &fails, const char *tier = TIER);
static void part_main(const rmcheck::Options &opt, rmcheck::Report &rep);
static void part_cap(const rmcheck::Options &opt, rmcheck::Report &rep);
static void part_teacher(const rmcheck::Options &opt, rmcheck::Report &rep);
static void part_change_stop(const rmcheck::Options &opt, rmcheck::Report &rep);
static void check_smoke_steps(const std::map<std::string, std::vector<rmcheck::EpisodeFile>> &eps,
    const std::vector<Json> &lines, std::vector<std::string> *fails);

// The actor's command line: its schedule, out dir, id 2, run seed `seed` and `matches` (< 0: until
// stopped).
static std::vector<std::string> actor_args(const rmcheck::Options &opt, const ActorRun &a, long matches, long seed) {
    std::vector<std::string> argv = {a.binary.empty() ? opt.binary : a.binary, "--actor", "--actor-schedule",
        a.schedule, "--actor-out", a.out,
        "--actor-id", "2", "--actor-seed", std::to_string(seed)};
    if (matches >= 0) argv.insert(argv.end(), {"--actor-matches", std::to_string(matches)});
    return argv;
}

// Writes a schedule whole (a temporary file renamed over it), as an orchestrator does.
static bool write_schedule(const std::string &path, const std::string &text) {
    const std::string tmp = path + ".new";
    return rmcheck::write_file(tmp, text) && std::rename(tmp.c_str(), path.c_str()) == 0;
}

static void run_actor(const rmcheck::Options &opt, ActorRun *a, long matches, long seed) {
    rmcheck::ProcSpec spec;
    spec.argv = actor_args(opt, *a, matches, seed);
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 900;
    if (!a->binary.empty()) spec.env = {"HIP_VISIBLE_DEVICES=-1", "CUDA_VISIBLE_DEVICES=-1"};
    a->result = rmcheck::run_process(spec);
    rmcheck::write_file(a->out + ".txt", a->result.out + "\n=== engine stderr ===\n" + a->result.err);
}

// The results file's lines and the status file, parsed.
static bool read_outputs(ActorRun *a, std::vector<std::string> *fails) {
    std::string text, err;
    if (!rmcheck::read_file(rmcheck::path_join(a->out, "a2.results.jsonl"), &text)) {
        fails->push_back(a->out + ": no results file");
        return false;
    }
    a->lines.clear();
    for (const std::string &line : rmcheck::split_lines(text)) {
        Json j;
        if (!Json::parse(line, &j, &err)) {
            fails->push_back("a results line does not parse: " + err);
            return false;
        }
        a->lines.push_back(j);
    }
    if (!rmcheck::read_file(rmcheck::path_join(a->out, "a2.status.json"), &text) ||
        !Json::parse(text, &a->status, &err)) {
        fails->push_back(a->out + ": the status file does not parse: " + err);
        return false;
    }
    return true;
}

static std::string jstr(const Json &o, const char *key) {
    const Json *v = o.get(key);
    return v && v->is_string() ? v->as_string() : "";
}

static int64_t jint(const Json &o, const char *key) {
    const Json *v = o.get(key);
    return v && v->is_number() ? v->as_int() : -999999;
}

// A JSON value's text in one canonical form (objects in key order as written).
static std::string canonical(const Json &v) {
    if (v.is_object()) {
        std::string out = "{";
        for (const auto &m : v.members())
            out += (out.size() > 1 ? "," : "") + featurize::json_quote(m.first) + ":" + canonical(m.second);
        return out + "}";
    }
    if (v.is_array()) {
        std::string out = "[";
        for (const Json &x : v.items()) out += (out.size() > 1 ? "," : "") + canonical(x);
        return out + "]";
    }
    if (v.is_null()) return "null";
    if (v.is_bool()) return v.as_bool() ? "true" : "false";
    return v.is_string() ? featurize::json_quote(v.as_string()) : v.text();
}

// A results line's policy versions as an episode's policy_versions_* attr writes them.
static std::string versions_text(const Json &arr) {
    std::string out;
    for (const Json &x : arr.items()) out += (out.empty() ? "" : ",") + x.text();
    return out;
}

static std::string stem_of(const std::string &rmlog) {
    return rmcheck::strip_ext(rmcheck::basename_of(rmlog));
}

// The episodes of `dir`, by their decision log's stem, each list in game order.
static bool episodes_by_stem(
    const std::string &dir, std::map<std::string, std::vector<rmcheck::EpisodeFile>> *out, std::string *err) {
    std::vector<rmcheck::EpisodeFile> all;
    out->clear();
    if (!rmcheck::is_dir(dir)) return true;
    if (!rmcheck::read_episodes(dir, &all, err)) return false;
    for (rmcheck::EpisodeFile &e : all) {
        const size_t g = e.name.rfind(".g");
        (*out)[e.name.substr(0, g)].push_back(std::move(e));
    }
    return true;
}

// The run exited 0 with no error line, its status says `state` and agrees with its results lines,
// and (`matches` >= 0) it played that many matches, each with at least one line.
static void check_run(const ActorRun &a, long matches, const char *state, std::vector<std::string> *fails) {
    std::vector<rmcheck::ScanLine> errors;
    rmcheck::scan_transcript(a.result.out + "\n" + a.result.err, &errors, nullptr);
    if (!a.result.ok()) fails->push_back("the actor " + a.result.status_text() + " (see " + a.out + ".txt)");
    for (const rmcheck::ScanLine &l : errors) fails->push_back("an error line: " + l.line);
    const Json &st = a.status;
    if (jstr(st, "state") != state) fails->push_back("status state '" + jstr(st, "state") + "', not " + state);
    int64_t decisions = 0, caps = 0, draws = 0, concessions = 0;
    std::set<int64_t> match_ids;
    std::set<std::string> versions;
    for (const Json &l : a.lines) {
        decisions += jint(l, "decisions");
        const std::string reason = jstr(l, "end_reason");
        caps += reason == "cap";
        draws += reason == "draw";
        concessions += reason == "concession";
        match_ids.insert(jint(l, "match"));
        for (const char *k : {"policy_versions_a", "policy_versions_b"})
            if (const Json *arr = l.get(k))
                for (const Json &x : arr->items()) versions.insert(x.text());
        if (jint(l, "decisions") != jint(l, "decisions_a") + jint(l, "decisions_b"))
            fails->push_back("results line of match " + std::to_string(jint(l, "match")) +
                             ": decisions is not decisions_a + decisions_b");
    }
    const bool stopped = std::string(state) == "stopped";
    if (jint(st, "games") != static_cast<int64_t>(a.lines.size()))
        fails->push_back(
            "status games " + std::to_string(jint(st, "games")) + ", results lines " + std::to_string(a.lines.size()));
    if (stopped ? jint(st, "decisions") < decisions : jint(st, "decisions") != decisions)
        fails->push_back("status decisions " + std::to_string(jint(st, "decisions")) + ", the results lines' " +
                         std::to_string(decisions));
    if (jint(st, "caps") != caps || jint(st, "draws") != draws || jint(st, "concessions") != concessions)
        fails->push_back("status caps / draws / concessions disagree with the results lines");
    if (jint(st, "errors") != 0) fails->push_back("status errors " + std::to_string(jint(st, "errors")));
    const Json *pv = st.get("policy_versions");
    if (!pv || jint(*pv, "count") != static_cast<int64_t>(versions.size()))
        fails->push_back("status policy_versions.count disagrees with the results lines' versions");
    if (matches >= 0) {
        if (jint(st, "matches") != matches || jint(st, "next_match") != matches)
            fails->push_back("status matches / next_match not " + std::to_string(matches));
        if (static_cast<long>(match_ids.size()) != matches ||
            (!match_ids.empty() && *match_ids.rbegin() != matches - 1))
            fails->push_back("the results lines name " + std::to_string(match_ids.size()) + " matches, not 0.." +
                             std::to_string(matches - 1));
    }
}

// Every results line with an episode agrees with that episode's footer; every episode has a line.
static void check_footers(const ActorRun &a, const std::map<std::string, std::vector<rmcheck::EpisodeFile>> &eps,
    std::vector<std::string> *fails) {
    size_t with_episode = 0, episodes = 0;
    for (const auto &kv : eps) episodes += kv.second.size();
    for (const Json &l : a.lines) {
        const Json *ep = l.get("episode");
        if (!ep || ep->is_null()) continue;
        ++with_episode;
        const std::string name = rmcheck::basename_of(ep->as_string());
        const rmcheck::EpisodeFile *e = nullptr;
        auto it = eps.find(stem_of(jstr(l, "rmlog")));
        if (it != eps.end())
            for (const rmcheck::EpisodeFile &f : it->second)
                if (f.name == name) e = &f;
        if (!e) {
            fails->push_back("results line names " + name + ", which was not written");
            continue;
        }
        const featurize::RmtbFile &f = e->ep.file;
        auto attr_s = [&f](const char *k) { return f.attr(k) ? f.attr(k)->s : std::string("?"); };
        auto attr_i = [&f](const char *k) { return f.attr(k) ? f.attr(k)->i : int64_t{-999999}; };
        std::vector<std::string> bad;
        if (attr_i("game") != jint(l, "game") - 1) bad.push_back("game");
        if (attr_s("first") != jstr(l, "first")) bad.push_back("first");
        if (attr_i("result_a") != jint(l, "result_a") || attr_i("result_b") != jint(l, "result_b"))
            bad.push_back("result");
        if (attr_s("end_reason") != jstr(l, "end_reason")) bad.push_back("end_reason");
        if (attr_i("game_decisions") != jint(l, "decisions")) bad.push_back("decisions");
        if (attr_i("wins_a") != jint(l, "wins_a") || attr_i("wins_b") != jint(l, "wins_b")) bad.push_back("wins");
        if (attr_s("deck_a") != jstr(l, "deck_a") || attr_s("deck_b") != jstr(l, "deck_b")) bad.push_back("decks");
        if (attr_s("agent_a") != jstr(l, "agent_a") || attr_s("agent_b") != jstr(l, "agent_b")) bad.push_back("agents");
        if (attr_i("game_seed") != jint(l, "game_seed") || attr_i("engine_seed") != jint(l, "engine_seed"))
            bad.push_back("seeds");
        if (attr_s("rmlog") != jstr(l, "rmlog")) bad.push_back("rmlog");
        const std::string seats = attr_s("seats");
        for (int s = 0; s < 2; ++s) {
            const char *key = s == 0 ? "policy_versions_a" : "policy_versions_b";
            if (seats.find(s == 0 ? 'A' : 'B') != std::string::npos && l.get(key) &&
                attr_s(key) != versions_text(*l.get(key)))
                bad.push_back(key);
        }
        if (!bad.empty())
            fails->push_back(name + ": its footer and results line differ in " + rmcheck::join(bad, ", "));
    }
    if (with_episode != episodes)
        fails->push_back(std::to_string(episodes) + " episodes written, " + std::to_string(with_episode) +
                         " results lines name one");
}

// Each match's decision log replayed alone (with `teacher` on the taught seat when given): the
// same episodes (compare_replayed; the labels too under a teacher) and the same game ends.
static void check_replays(const rmcheck::Options &opt, const ActorRun &a,
    const std::map<std::string, std::vector<rmcheck::EpisodeFile>> &eps, const std::string &teacher,
    std::vector<std::string> *fails) {
    std::map<std::string, std::vector<const Json *>> matches;  // rmlog -> its lines
    for (const Json &l : a.lines) matches[jstr(l, "rmlog")].push_back(&l);
    std::vector<std::pair<std::string, std::vector<const Json *>>> list(matches.begin(), matches.end());
    std::vector<std::vector<std::string>> part(list.size());
    rmcheck::parallel_for(list.size(), opt.jobs, [&](size_t i) {
        const std::string &rmlog = list[i].first;
        const std::string stem = stem_of(rmlog);
        const std::string dir = rmcheck::path_join(rmcheck::path_join(a.dir, "replay"), stem);
        rmcheck::remove_tree(dir);
        rmcheck::ProcSpec spec;
        const Json &first = *list[i].second[0];
        spec.argv = {opt.binary, "--replay", rmlog, "--trajectory-out", dir, "--knowledge-a",
            jstr(first, "knowledge_a"), "--knowledge-b", jstr(first, "knowledge_b")};
        if (!teacher.empty())
            spec.argv.insert(
                spec.argv.end(), {jstr(first, "teacher_a").empty() ? "--teacher-b" : "--teacher-a", teacher});
        spec.cwd = opt.bin_dir;
        spec.timeout_s = 600;
        spec.merge_stderr = true;
        const rmcheck::ProcResult r = rmcheck::run_process(spec);
        std::vector<rmcheck::ScanLine> errors;
        rmcheck::scan_transcript(r.out, &errors, nullptr);
        if (!r.ok() || !errors.empty()) {
            part[i].push_back(stem + ": --replay " + r.status_text() + (errors.empty() ? "" : ": " + errors[0].line));
            return;
        }
        const std::map<int, int> results = rmcheck::game_results(r.out);
        for (const Json *l : list[i].second) {
            const int g = static_cast<int>(jint(*l, "game"));
            const bool capped = jstr(*l, "end_reason") == "cap";
            const bool ok = capped ? rmcheck::contains(r.out, ("GAME_CAP: " + std::to_string(g) + " ").c_str())
                                   : results.count(g) && results.at(g) == jint(*l, "result_a");
            if (!ok) part[i].push_back(stem + ": the replay's game " + std::to_string(g) + " ends otherwise");
        }
        auto live = eps.find(stem);
        std::map<std::string, std::vector<rmcheck::EpisodeFile>> replayed;
        std::string err;
        if (!episodes_by_stem(dir, &replayed, &err)) {
            part[i].push_back(stem + ": the replay's episodes: " + err);
            return;
        }
        const std::vector<rmcheck::EpisodeFile> none;
        const std::vector<rmcheck::EpisodeFile> &l_eps = live == eps.end() ? none : live->second;
        const std::vector<rmcheck::EpisodeFile> &r_eps = replayed.count(stem) ? replayed.at(stem) : none;
        for (const std::string &f : rmcheck::compare_replayed(l_eps, r_eps)) part[i].push_back(f);
        if (teacher.empty() || l_eps.size() != r_eps.size()) return;
        for (size_t k = 0; k < l_eps.size(); ++k) {
            const std::vector<featurize::EpisodeStep> &ls = l_eps[k].ep.steps, &rs = r_eps[k].ep.steps;
            bool same = ls.size() == rs.size();
            for (size_t d = 0; same && d < ls.size(); ++d) same = ls[d].teacher == rs[d].teacher;
            if (!same) part[i].push_back(l_eps[k].name + ": its labels differ from a fresh teacher's on its replay");
        }
    });
    for (const std::vector<std::string> &p : part) fails->insert(fails->end(), p.begin(), p.end());
}

static void report(
    rmcheck::Report &rep, const char *part, const std::vector<std::string> &fails, const char *tier) {
    for (size_t i = 0; i < fails.size() && i < MAX_FAILS; ++i) rep.error(tier, std::string(part) + ": " + fails[i]);
    if (fails.size() > MAX_FAILS)
        rep.error(tier, std::string(part) + ": " + std::to_string(fails.size() - MAX_FAILS) + " more failures");
}

// Six matches of a two-entry schedule (a bo1 entry swapping sides, a bo3 entry swapping decks): the
// run, its status and results against the episodes, every match's decision log replayed alone to
// the same episodes, and a rerun with the same run seed and actor id writing the same decision logs
// and results.
static void part_main(const rmcheck::Options &opt, rmcheck::Report &rep) {
    std::vector<std::string> fails;
    const std::string base = rmcheck::path_join(opt.out(TIER), "main");
    rmcheck::remove_tree(base);
    rmcheck::make_dirs(base);
    ActorRun a, b;
    a.dir = b.dir = base;
    a.schedule = b.schedule = rmcheck::path_join(base, "schedule.json");
    a.out = rmcheck::path_join(base, "run1");
    b.out = rmcheck::path_join(base, "run2");
    write_schedule(a.schedule,
        "{\"version\": \"main\", \"entries\": [\n"
        " {\"id\": \"bo1_sides\", \"format\": \"bo1\", \"swap\": \"sides\", \"deck_a\": \"league/burn\", "
        "\"deck_b\": \"league/bug\", \"agent_a\": \"random\", \"agent_b\": \"explore\"},\n"
        " {\"id\": \"bo3_decks\", \"format\": \"bo3\", \"swap\": \"decks\", \"deck_a\": \"league/ur_delver\", "
        "\"deck_b\": \"league/lands\", \"agent_a\": \"explore\", \"agent_b\": \"random\", \"knowledge_b\": "
        "\"off\"}]}\n");
    const long seed = opt.seed + 900;
    std::vector<ActorRun *> runs = {&a, &b};
    rmcheck::parallel_for(2, opt.jobs, [&](size_t i) { run_actor(opt, runs[i], 6, seed); });
    if (!read_outputs(&a, &fails) || !read_outputs(&b, &fails)) return report(rep, "main", fails);
    check_run(a, 6, "done", &fails);
    std::map<std::string, std::vector<rmcheck::EpisodeFile>> eps;
    std::string err;
    if (!episodes_by_stem(rmcheck::path_join(a.out, "train/traj"), &eps, &err)) fails.push_back(err);
    check_footers(a, eps, &fails);
    std::map<std::string, int> entries;
    int swapped = 0;
    for (const Json &l : a.lines) {
        ++entries[jstr(l, "entry")];
        if (jint(l, "game") != 1) continue;
        const bool sw = l.get("swapped") && l.get("swapped")->as_bool();
        swapped += sw;
        const bool sides = jstr(l, "entry") == "bo1_sides";
        const std::string want_a =
            sides ? (sw ? "league/bug" : "league/burn") : (sw ? "league/lands" : "league/ur_delver");
        const std::string want_agent = sides ? (sw ? "explore" : "random") : "explore";
        if (jstr(l, "deck_a") != want_a || jstr(l, "agent_a") != want_agent)
            fails.push_back("match " + std::to_string(jint(l, "match")) + ": seat A has the wrong deck or agent");
    }
    if (entries.size() != 2) fails.push_back("the six matches did not play both entries");
    check_replays(opt, a, eps, "", &fails);
    // The rerun: the same decision logs byte for byte and the same results lines but the paths and
    // the timing.
    for (const char *purpose : {"train", "eval"}) {
        const std::string d1 = rmcheck::path_join(a.out, std::string(purpose) + "/rmlog"),
                          d2 = rmcheck::path_join(b.out, std::string(purpose) + "/rmlog");
        const std::vector<std::string> l1 = rmcheck::list_dir(d1, ".rmlog"), l2 = rmcheck::list_dir(d2, ".rmlog");
        if (l1 != l2) fails.push_back("the rerun wrote other decision logs");
        for (const std::string &name : l1) {
            std::string x, y;
            rmcheck::read_file(rmcheck::path_join(d1, name), &x);
            rmcheck::read_file(rmcheck::path_join(d2, name), &y);
            if (x != y || x.empty()) fails.push_back("the rerun's " + name + " differs");
        }
    }
    if (a.lines.size() != b.lines.size()) fails.push_back("the rerun wrote another number of results lines");
    for (size_t i = 0; i < a.lines.size() && i < b.lines.size(); ++i) {
        Json x = a.lines[i], y = b.lines[i];
        for (Json *j : {&x, &y})
            for (const char *k : {"seconds", "rmlog", "episode"}) j->erase(k);
        if (canonical(x) != canonical(y)) fails.push_back("the rerun's results line " + std::to_string(i) + " differs");
    }
    if (fails.empty())
        rep.info("actorloop main: 6 matches, " + std::to_string(a.lines.size()) + " games (" +
                 std::to_string(entries["bo1_sides"]) + " bo1, " + std::to_string(entries["bo3_decks"]) + " bo3), " +
                 std::to_string(swapped) + " swapped; replays and rerun identical");
    report(rep, "main", fails);
}

// A bo3 entry capped at 25 decisions a game: each match's game 1 ends with end reason cap, the
// match stops there and the actor plays the next one; the replays cap at the same decision.
static void part_cap(const rmcheck::Options &opt, rmcheck::Report &rep) {
    std::vector<std::string> fails;
    ActorRun a;
    a.dir = rmcheck::path_join(opt.out(TIER), "cap");
    rmcheck::remove_tree(a.dir);
    rmcheck::make_dirs(a.dir);
    a.schedule = rmcheck::path_join(a.dir, "schedule.json");
    a.out = rmcheck::path_join(a.dir, "run");
    write_schedule(a.schedule,
        "{\"entries\": [{\"id\": \"capped\", \"deck_a\": \"league/gw_maverick\", \"deck_b\": \"league/uw_control\", "
        "\"agent_a\": \"random\", \"agent_b\": \"explore\", \"max_game_decisions\": 25}]}\n");
    run_actor(opt, &a, 3, opt.seed + 910);
    if (!read_outputs(&a, &fails)) return report(rep, "cap", fails);
    check_run(a, 3, "done", &fails);
    if (a.lines.size() != 3) fails.push_back("expected one game per match, got " + std::to_string(a.lines.size()));
    for (const Json &l : a.lines)
        if (jstr(l, "end_reason") != "cap" || jint(l, "decisions") != 25 || jint(l, "result_a") != 0 ||
            jint(l, "game") != 1)
            fails.push_back("match " + std::to_string(jint(l, "match")) + ": not a game 1 capped at 25 decisions");
    if (jint(a.status, "caps") != 3) fails.push_back("status caps is not 3");
    if (rmcheck::contains(a.result.out, "MATCH_RESULT")) fails.push_back("a capped match printed a MATCH_RESULT");
    std::map<std::string, std::vector<rmcheck::EpisodeFile>> eps;
    std::string err;
    if (!episodes_by_stem(rmcheck::path_join(a.out, "train/traj"), &eps, &err)) fails.push_back(err);
    check_footers(a, eps, &fails);
    check_replays(opt, a, eps, "", &fails);
    if (fails.empty()) rep.info("actorloop cap: 3 matches, each game 1 capped at 25 decisions; the actor went on");
    report(rep, "cap", fails);
}

// A bo1 entry whose side a is taught (scripted:hard through python/teacher.py, mix 0) and swaps
// sides, over three matches: one teacher process per taught seat across matches; each match's
// labels equal a fresh teacher's on that match's replay.
static void part_teacher(const rmcheck::Options &opt, rmcheck::Report &rep) {
    if (opt.python.empty()) {
        rep.info("actorloop teacher: skipped: no python3 on PATH");
        return;
    }
    std::vector<std::string> fails;
    ActorRun a;
    a.dir = rmcheck::path_join(opt.out(TIER), "teacher");
    rmcheck::remove_tree(a.dir);
    rmcheck::make_dirs(a.dir);
    a.schedule = rmcheck::path_join(a.dir, "schedule.json");
    a.out = rmcheck::path_join(a.dir, "run");
    const std::string teacher =
        "'" + opt.python + "' '" + rmcheck::path_join(opt.repo, "python/teacher.py") + "' --seed 11";
    write_schedule(a.schedule,
        "{\"entries\": [{\"id\": \"taught\", \"format\": \"bo1\", \"swap\": \"sides\", \"deck_a\": \"league/bug\", "
        "\"deck_b\": \"league/burn\", \"agent_a\": \"explore\", \"agent_b\": \"random\", \"teacher_a\": " +
            featurize::json_quote(teacher) + ", \"teacher_mix\": 0}]}\n");
    run_actor(opt, &a, 3, opt.seed + 920);
    if (!read_outputs(&a, &fails)) return report(rep, "teacher", fails);
    check_run(a, 3, "done", &fails);
    std::map<std::string, std::vector<rmcheck::EpisodeFile>> eps;
    std::string err;
    if (!episodes_by_stem(rmcheck::path_join(a.out, "train/traj"), &eps, &err)) fails.push_back(err);
    check_footers(a, eps, &fails);
    int taught_b = 0;
    for (const Json &l : a.lines) {
        const int seat = jstr(l, "teacher_a").empty() ? 1 : 0;
        taught_b += seat;
        auto it = eps.find(stem_of(jstr(l, "rmlog")));
        if (it == eps.end()) continue;
        for (const rmcheck::EpisodeFile &e : it->second)
            for (const featurize::EpisodeStep &s : e.ep.steps)
                if ((s.seat == seat) != (s.teacher >= 0)) {
                    fails.push_back(e.name + ": a step of seat " + std::string(s.seat ? "B" : "A") +
                                    (s.seat == seat ? " (taught) has no label" : " (untaught) has a label"));
                    break;
                }
    }
    check_replays(opt, a, eps, teacher, &fails);
    if (fails.empty())
        rep.info("actorloop teacher: 3 matches labelled (" + std::to_string(3 - taught_b) + " on seat A, " +
                 std::to_string(taught_b) + " on seat B), the labels a fresh teacher's on each replay");
    report(rep, "teacher", fails);
}

// An actor run until stopped: an eval schedule (no episodes) rewritten mid-run to a train bo3
// schedule, picked up at the next match; SIGTERM after a new match's game 1 stops it in that match:
// exit 0, state stopped, its decision log ending with QUIT (a replay stops there), game 1's episode
// and no other, no temporary file.
static void part_change_stop(const rmcheck::Options &opt, rmcheck::Report &rep) {
    std::vector<std::string> fails;
    ActorRun a;
    a.dir = rmcheck::path_join(opt.out(TIER), "change_stop");
    rmcheck::remove_tree(a.dir);
    rmcheck::make_dirs(a.dir);
    a.schedule = rmcheck::path_join(a.dir, "schedule.json");
    a.out = rmcheck::path_join(a.dir, "run");
    write_schedule(a.schedule,
        "{\"version\": \"old\", \"entries\": [{\"id\": \"old\", \"purpose\": \"eval\", \"format\": \"bo1\", "
        "\"deck_a\": \"league/burn\", \"deck_b\": \"league/bug\", \"agent_a\": \"explore\", \"agent_b\": "
        "\"explore\"}]}\n");
    rmcheck::ProcSpec spec;
    spec.argv = actor_args(opt, a, -1, opt.seed + 930);
    spec.cwd = opt.bin_dir;
    rmcheck::Session session;
    std::string err;
    if (!session.start(spec, &err)) {
        fails.push_back("the actor does not start: " + err);
        return report(rep, "change_stop", fails);
    }
    session.set_timeout(300);
    std::string out, line;
    bool rewrote = false, in_new = false, signalled = false;
    while (!signalled && session.fill_line()) {
        const size_t nl = session.pending().find('\n');
        line = session.pending().substr(0, nl);
        session.take(nl + 1);
        out += line + "\n";
        if (!rewrote && rmcheck::starts_with(line, "GAME_RESULT:")) {
            rewrote = write_schedule(a.schedule,
                "{\"version\": \"new\", \"entries\": [{\"id\": \"new\", \"format\": \"bo3\", \"deck_a\": "
                "\"league/lands\", \"deck_b\": \"league/bug\", \"agent_a\": \"explore\", \"agent_b\": \"random\"}]}\n");
            if (!rewrote) fails.push_back("cannot rewrite the schedule");
        }
        if (rmcheck::starts_with(line, "ACTOR-MATCH:")) in_new = rmcheck::contains(line, "entry=new");
        if (in_new && rmcheck::starts_with(line, "GAME_RESULT: 1 ")) {
            session.send_signal(SIGTERM);
            signalled = true;
        }
    }
    a.result = session.finish();
    a.result.out = out + a.result.out;
    rmcheck::write_file(a.out + ".txt", a.result.out + "\n=== engine stderr ===\n" + a.result.err);
    if (!signalled) fails.push_back("no match of the rewritten schedule finished its game 1 (see " + a.out + ".txt)");
    if (!read_outputs(&a, &fails)) return report(rep, "change_stop", fails);
    check_run(a, -1, "stopped", &fails);
    size_t old_lines = 0, new_lines = 0;
    for (const Json &l : a.lines) {
        const bool is_new = jstr(l, "entry") == "new";
        if (!is_new && new_lines > 0) fails.push_back("a match of the old schedule after one of the new");
        (is_new ? new_lines : old_lines) += 1;
        const Json *ep = l.get("episode");
        if (is_new != (ep && !ep->is_null()))
            fails.push_back("match " + std::to_string(jint(l, "match")) + ": " +
                            (is_new ? "a train game without an episode" : "an eval game with an episode"));
        if (jstr(l, "purpose") != (is_new ? "train" : "eval")) fails.push_back("a results line's purpose");
    }
    if (old_lines == 0 || new_lines == 0) fails.push_back("expected games of both schedules");
    if (rmcheck::is_dir(rmcheck::path_join(a.out, "eval/traj"))) fails.push_back("eval matches wrote episodes");
    const Json *sched = a.status.get("schedule");
    if (!sched || jstr(*sched, "version") != "new" || jint(*sched, "loads") != 2)
        fails.push_back("the status does not show the rewritten schedule loaded once");
    if (!rmcheck::contains(jstr(a.status, "exit_reason"), "in match"))
        fails.push_back("the actor did not stop within a match: " + jstr(a.status, "exit_reason"));
    const Json *cur = a.status.get("current");
    const std::string rmlog = cur && cur->is_object() ? jstr(*cur, "rmlog") : "";
    std::string text;
    if (rmlog.empty() || !rmcheck::read_file(rmlog, &text) || !rmcheck::ends_with(text, "\nQUIT\n"))
        fails.push_back("the stopped match's decision log does not end with QUIT");
    std::map<std::string, std::vector<rmcheck::EpisodeFile>> eps;
    if (!episodes_by_stem(rmcheck::path_join(a.out, "train/traj"), &eps, &err)) fails.push_back(err);
    if (!rmlog.empty() && (!eps.count(stem_of(rmlog)) || eps.at(stem_of(rmlog)).size() != 1))
        fails.push_back("the stopped match has other episodes than its game 1's");
    if (!rmcheck::list_dir(a.out, ".tmp").empty()) fails.push_back("a temporary file was left in " + a.out);
    check_footers(a, eps, &fails);
    if (!rmlog.empty()) {
        rmcheck::ProcSpec replay;
        replay.argv = {opt.binary, "--replay", rmlog};
        replay.cwd = opt.bin_dir;
        replay.merge_stderr = true;
        const rmcheck::ProcResult r = rmcheck::run_process(replay);
        if (!r.ok() || !rmcheck::contains(r.out, "QUIT: the recorded session quit"))
            fails.push_back("--replay of the stopped match's log " + r.status_text() + " without its QUIT line");
    }
    if (fails.empty())
        rep.info("actorloop change_stop: " + std::to_string(old_lines) + " eval games, then " +
                 std::to_string(new_lines) + " of the rewritten schedule; SIGTERM stopped match " +
                 std::to_string(jint(*cur, "match")) + " cleanly");
    report(rep, "change_stop", fails);
}

// The smoke run's steps: every step carries a behaviour distribution summing to 1 with mass on
// the reply and the model's policy version (the results lines' one version); side a's steps (taught
// at mix 0.5) are labelled with mu(label) >= 0.5, side b's (thresh=0.03:disc=16) unlabelled,
// multiples of 1/16, no row below 0.03 with mass.
static void check_smoke_steps(const std::map<std::string, std::vector<rmcheck::EpisodeFile>> &eps,
    const std::vector<Json> &lines, std::vector<std::string> *fails) {
    std::set<std::string> versions;
    for (const Json &l : lines)
        for (const char *k : {"policy_versions_a", "policy_versions_b"})
            if (const Json *arr = l.get(k)) versions.insert(versions_text(*arr));
    if (versions.size() != 1 || versions.begin()->find(',') != std::string::npos)
        fails->push_back("the results lines do not name one policy version for both seats");
    const int64_t version = versions.size() == 1 ? std::atoll(versions.begin()->c_str()) : -1;
    size_t steps = 0;
    for (const Json &l : lines) {
        auto it = eps.find(stem_of(jstr(l, "rmlog")));
        if (it == eps.end()) continue;
        const int taught = jstr(l, "side_a_seat") == "A" ? 0 : 1;
        for (const rmcheck::EpisodeFile &e : it->second) {
            if (e.game_number != jint(l, "game")) continue;
            std::string bad;
            for (const featurize::EpisodeStep &st : e.ep.steps) {
                ++steps;
                double sum = 0;
                bool sixteenths = true, below = false;
                for (const float m : st.mu) {
                    sum += m;
                    sixteenths = sixteenths && std::fabs(m * 16 - std::round(m * 16)) < 1e-4;
                    below = below || (m > 0 && m < 0.03 - 1e-6);
                }
                const bool reply_ok = st.chosen >= 0 && st.chosen < static_cast<int64_t>(st.mu.size()) &&
                                      st.mu[static_cast<size_t>(st.chosen)] > 0;
                if (!st.has_mu || std::fabs(sum - 1) > 1e-4 || !reply_ok)
                    bad = "a step's mu is not a distribution with mass on the reply";
                else if (st.policy_version != version)
                    bad = "a step's policy version is not the results lines'";
                else if (st.seat == taught && (st.teacher < 0 || st.mu[static_cast<size_t>(st.teacher)] < 0.5 - 1e-5))
                    bad = "a taught step without a label carrying half the mass";
                else if (st.seat != taught && (st.teacher >= 0 || !sixteenths || below))
                    bad = "a projected step labelled, or not multiples of 1/16 above 0.03";
                if (!bad.empty()) break;
            }
            if (!bad.empty()) fails->push_back(e.name + ": " + bad);
        }
    }
    if (steps == 0) fails->push_back("no recorded step");
}

namespace rmcheck {

// The actor tier's continuous smoke run (plan T8; actor.h).
void actor_smoke_check(const Options &opt, Report &rep, const ActorSetup &setup) {
    const char *part = "actor smoke";
    if (opt.python.empty()) {
        rep.info("actor smoke: skipped: no python3 on PATH");
        return;
    }
    std::vector<std::string> fails;
    ActorRun a;
    a.dir = path_join(setup.dir, "smoke");
    remove_tree(a.dir);
    make_dirs(a.dir);
    a.schedule = path_join(a.dir, "schedule.json");
    a.out = path_join(a.dir, "run");
    a.binary = setup.nn_binary;
    const std::string teacher = "'" + opt.python + "' '" + path_join(opt.repo, "python/teacher.py") + "' --seed 11";
    ::write_schedule(a.schedule,
        "{\"version\": \"smoke\", \"entries\": [{\"id\": \"nn_taught\", \"format\": \"bo3\", \"swap\": "
        "\"sides\", \"deck_a\": \"league/burn\", \"deck_b\": \"league/ur_delver\", \"agent_a\": " +
            featurize::json_quote("nn:" + setup.ts()) + ", \"agent_b\": " +
            featurize::json_quote("nn:" + setup.ts() + ":thresh=0.03:disc=16") + ", \"teacher_a\": " +
            featurize::json_quote(teacher) + ", \"teacher_mix\": 0.5}]}\n");
    ::run_actor(opt, &a, 4, opt.seed + 950);
    if (!::read_outputs(&a, &fails)) return ::report(rep, part, fails, "actor");
    ::check_run(a, 4, "done", &fails);
    std::map<std::string, std::vector<EpisodeFile>> eps;
    std::string err;
    if (!::episodes_by_stem(path_join(a.out, "train/traj"), &eps, &err)) fails.push_back(err);
    ::check_footers(a, eps, &fails);
    ::check_smoke_steps(eps, a.lines, &fails);
    ::check_replays(opt, a, eps, teacher, &fails);
    int sb_steps = 0, steps = 0;
    for (const auto &kv : eps)
        for (const EpisodeFile &e : kv.second)
            for (const featurize::EpisodeStep &st : e.ep.steps) {
                ++steps;
                sb_steps += st.sideboarding;
            }
    if (fails.empty())
        rep.info("actor smoke: robomage_nn --actor, 4 bo3 matches of nn:<reference policy> (taught at mix 0.5) vs "
                 "nn:<it>:thresh=0.03:disc=16, " + std::to_string(a.lines.size()) + " games, " +
                 std::to_string(steps) + " recorded steps (" + std::to_string(sb_steps) +
                 " sideboarding), mu checked, replays and labels identical");
    ::report(rep, part, fails, "actor");
}

void tier_actorloop(const Options &opt, Report &rep) {
    const std::vector<void (*)(const Options &, Report &)> parts = {
        part_main, part_cap, part_teacher, part_change_stop};
    parallel_for(parts.size(), opt.jobs, [&](size_t i) { parts[i](opt, rep); });
}

}  // namespace rmcheck
