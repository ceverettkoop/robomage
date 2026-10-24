#include <cmath>
#include <cstdio>
#include <map>

#include "../../src/featurize/json.h"
#include "../../src/featurize/setup.h"
#include "frames.h"
#include "gate.h"
#include "proc.h"
#include "scan.h"
#include "traj.h"
#include "util.h"

using featurize::Featurization;

namespace {

// One engine run with --trajectory-out and its --replay.
struct TrajGame {
        std::string name;
        std::vector<std::string> args;  // agents, script, decks and caps
        bool omniscient = false;
        bool dump = false;
        bool seats_ab = true;
        std::vector<int> mu_seats;             // the seats every step of which carries μ (the random agent's)
        std::vector<std::string> end_reasons;  // the episodes' end reasons in game order
        // Filled by the run.
        std::string dir, live_dir, replay_dir, rmlog, dump_path;
        rmcheck::ProcResult live, replay;
};

// A decision as the dump shows it: the deciding seat's frame and the omniscient frame.
struct DumpDecision {
        int seat = -1;
        rmstate::DecodedFrame seat_frame, omni_frame;
};

const size_t MAX_FAILS = 20;

}  // namespace

static std::vector<TrajGame> traj_games(const rmcheck::Options &opt);
static bool load_featurization(const rmcheck::Options &opt, rmcheck::Report &rep, Featurization *f);
static void run_game(const rmcheck::Options &opt, TrajGame *g);
static void check_steps(const TrajGame &g, const rmcheck::EpisodeFile &e, std::vector<std::string> *fails);
static void check_game(
    const TrajGame &g, const std::vector<rmcheck::EpisodeFile> &eps, std::vector<std::string> *fails);
static bool dump_decisions(
    const std::string &path, std::vector<DumpDecision> *out, rmstate::Schema *omni_schema, std::string *err);
static void check_against_frames(const Featurization &f, const TrajGame &g,
    const std::vector<rmcheck::EpisodeFile> &eps, std::vector<std::string> *fails, size_t *checked);
static std::string step_json(const featurize::EpisodeStep &s);
static std::string float_json(float v);
static void python_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::vector<std::pair<std::string, const rmcheck::EpisodeFile *>> &eps);
static void report_sizes(rmcheck::Report &rep, const rmcheck::EpisodeFile &e);
static void driver_part(const rmcheck::Options &opt, rmcheck::Report &rep);

// The runs: a bo1 game with the omniscient featurization and a dump; a bo3 match with
// sideboarding and a dump; a bo3 match whose game 1 seat A concedes and whose match it concedes in
// sideboarding; a bo3 match of seat A's decisions only, stopped at --max-decisions. Seat A or B is
// the random agent in each (μ uniform), the other explore (no μ).
static std::vector<TrajGame> traj_games(const rmcheck::Options &opt) {
    const std::string s = std::to_string(opt.seed + 400);
    std::vector<TrajGame> out(4);
    out[0].name = "bo1_omniscient";
    out[0].args = {"--agent-a", "random", "--agent-b", "explore", "--deck-a", "league/ur_delver", "--deck-b",
        "league/bug", "--game-seed", s, "--trajectory-omniscient"};
    out[0].omniscient = out[0].dump = true;
    out[0].mu_seats = {0};
    out[0].end_reasons = {"win"};
    out[1].name = "bo3";
    out[1].args = {"--agent-a", "explore", "--agent-b", "random", "--deck-a", "league/reanimator", "--deck-b",
        "league/wrb_energy", "--bo3", "--game-seed", s};
    out[1].dump = true;
    out[1].mu_seats = {1};
    out[2].name = "concede";
    out[2].args = {"--agent-a", "random", "--agent-b", "explore", "--deck-a", "league/burn", "--deck-b", "league/lands",
        "--bo3", "--game-seed", s, "--play", "A:keep,B:keep,A:concede,A:concede:match"};
    out[2].end_reasons = {"concession", "concession"};
    out[3].name = "cap_seat_a";
    out[3].args = {"--agent-a", "random", "--agent-b", "explore", "--deck-a", "league/gw_maverick", "--deck-b",
        "league/uw_control", "--bo3", "--game-seed", s, "--max-decisions", "40", "--trajectory-seats", "A"};
    out[3].seats_ab = false;
    out[3].mu_seats = {0};
    out[3].end_reasons = {"cap"};
    return out;
}

// The whole-vocab cards file (robomage --cards-frame) and the featurizer's inputs built from it.
static bool load_featurization(const rmcheck::Options &opt, rmcheck::Report &rep, Featurization *f) {
    const std::string cards = rmcheck::path_join(opt.out("traj"), "cards.rmst");
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary, "--cards-frame", cards};
    spec.cwd = opt.bin_dir;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    std::string err;
    if (!r.ok()) {
        rep.error("traj", "robomage --cards-frame " + r.status_text());
        return false;
    }
    if (!f->load(cards, rmcheck::path_join(opt.bin_dir, "resources/embeddings/oracle_text.rmtb"), "", &err)) {
        rep.error("traj", "the featurizer's inputs: " + err);
        return false;
    }
    return true;
}

// The live run into <dir>/live (its decision log there, by default), then the --replay of that
// log into <dir>/replay.
static void run_game(const rmcheck::Options &opt, TrajGame *g) {
    g->dir = rmcheck::path_join(opt.out("traj"), g->name);
    rmcheck::remove_tree(g->dir);
    g->live_dir = rmcheck::path_join(g->dir, "live");
    g->replay_dir = rmcheck::path_join(g->dir, "replay");
    g->dump_path = rmcheck::path_join(g->dir, "live.rms");
    rmcheck::make_dirs(g->dir);
    rmcheck::ProcSpec spec;
    spec.argv = {opt.binary};
    spec.argv.insert(spec.argv.end(), g->args.begin(), g->args.end());
    spec.argv.insert(spec.argv.end(), {"--trajectory-out", g->live_dir});
    if (g->dump) spec.argv.insert(spec.argv.end(), {"--rmstate-dump", g->dump_path});
    spec.cwd = opt.bin_dir;
    spec.timeout_s = 900;
    spec.merge_stderr = true;
    g->live = rmcheck::run_process(spec);
    const std::vector<std::string> logs = rmcheck::list_dir(g->live_dir, ".rmlog");
    if (logs.size() != 1) return;
    g->rmlog = rmcheck::path_join(g->live_dir, logs[0]);
    rmcheck::ProcSpec replay;
    replay.argv = {opt.binary, "--replay", g->rmlog, "--trajectory-out", g->replay_dir};
    if (g->omniscient) replay.argv.push_back("--trajectory-omniscient");
    if (!g->seats_ab) replay.argv.insert(replay.argv.end(), {"--trajectory-seats", "A"});
    replay.cwd = opt.bin_dir;
    replay.timeout_s = 900;
    replay.merge_stderr = true;
    g->replay = rmcheck::run_process(replay);
}

// One episode's steps: the game and seats, sideboarding before the game, μ a distribution with
// mass at the reply, the replies menu rows or sentinels, the omniscient part as requested.
static void check_steps(const TrajGame &g, const rmcheck::EpisodeFile &e, std::vector<std::string> *fails) {
    const featurize::Episode &ep = e.ep;
    const featurize::RmtbAttr *game = ep.file.attr("game");
    const std::string where = g.name + "/" + e.name + ": ";
    if (!game || game->i + 1 != e.game_number) fails->push_back(where + "attr game is not the file's game number");
    if (ep.omniscient != g.omniscient || (ep.file.find("omni_tok_type") != nullptr) != g.omniscient ||
        (ep.file.find("opp_hand") != nullptr) != g.omniscient)
        fails->push_back(where + "the omniscient arrays are " + (g.omniscient ? "absent" : "present"));
    bool in_game = false;
    for (size_t i = 0; i < ep.steps.size() && fails->size() < MAX_FAILS; ++i) {
        const featurize::EpisodeStep &s = ep.steps[i];
        const std::string at = where + "step " + std::to_string(i) + ": ";
        const int64_t a = static_cast<int64_t>(s.tensors.a());
        if (game && s.game != game->i) fails->push_back(at + "game " + std::to_string(s.game) + " in this episode");
        if (s.seat > 1 || (!g.seats_ab && s.seat != 0)) fails->push_back(at + "a seat that is not recorded");
        if (s.sideboarding && (in_game || e.game_number == 1))
            fails->push_back(at + "a sideboarding decision after the game began, or before game 1");
        in_game = in_game || !s.sideboarding;
        if (!(s.chosen >= 0 && s.chosen < a) && s.chosen != -2 && s.chosen != -3)
            fails->push_back(at + "reply " + std::to_string(s.chosen) + " is no menu row or sentinel");
        bool mu_seat = false;
        for (int m : g.mu_seats) mu_seat = mu_seat || m == s.seat;
        if (mu_seat != s.has_mu && g.name != "concede")
            fails->push_back(at + (s.has_mu ? "a mu from an agent with none" : "no mu from the random agent"));
        if (s.has_mu) {
            double sum = 0;
            bool nonneg = true;
            for (float p : s.mu) {
                sum += p;
                nonneg = nonneg && p >= 0;
            }
            if (std::fabs(sum - 1.0) > 1e-5 || !nonneg) fails->push_back(at + "mu is not a distribution");
            if (s.chosen >= 0 && s.chosen < a && !(s.mu[static_cast<size_t>(s.chosen)] > 0))
                fails->push_back(at + "the reply's row has no mass under mu");
        }
        if (s.teacher != -1) fails->push_back(at + "a teacher label with no teacher");
    }
}

// The run's episodes together: one per game played (and per stopped or conceded game), the
// footers as the GAME_RESULT lines say, the replies those of the decision log.
static void check_game(
    const TrajGame &g, const std::vector<rmcheck::EpisodeFile> &eps, std::vector<std::string> *fails) {
    const std::map<int, int> results = rmcheck::game_results(g.live.out);
    for (size_t i = 0; i < eps.size(); ++i) {
        const rmcheck::EpisodeFile &e = eps[i];
        const featurize::RmtbFile &f = e.ep.file;
        const std::string where = g.name + "/" + e.name + ": ";
        if (e.game_number != static_cast<int>(i) + 1) fails->push_back(where + "a game number out of sequence");
        const featurize::RmtbAttr *ra = f.attr("result_a"), *rb = f.attr("result_b"), *reason = f.attr("end_reason"),
                                  *first = f.attr("first"), *rmlog = f.attr("rmlog"), *dec = f.attr("decisions");
        if (!ra || !rb || !reason || !first || !rmlog || !dec || ra->i != -rb->i ||
            dec->i != static_cast<int64_t>(e.ep.steps.size())) {
            fails->push_back(where + "the footer attrs are missing or inconsistent");
            continue;
        }
        const auto r = results.find(e.game_number);
        if (r != results.end() && r->second != ra->i)
            fails->push_back(
                where + "result_a " + std::to_string(ra->i) + ", GAME_RESULT says " + std::to_string(r->second));
        if (r == results.end() && reason->s != "cap" && !(reason->s == "concession" && first->s.empty()))
            fails->push_back(where +
                             "an episode of a game with no GAME_RESULT that was neither capped nor conceded "
                             "before it began");
        if (i < g.end_reasons.size() && reason->s != g.end_reasons[i])
            fails->push_back(where + "end_reason " + reason->s + ", expected " + g.end_reasons[i]);
        if (g.end_reasons.empty() && reason->s != "win" && reason->s != "draw")
            fails->push_back(where + "end_reason " + reason->s);
        if (rmlog->s != g.rmlog) fails->push_back(where + "rmlog names " + rmlog->s);
        check_steps(g, e, fails);
    }
    for (const auto &r : results)
        if (r.first < 1 || r.first > static_cast<int>(eps.size()))
            fails->push_back(g.name + ": GAME_RESULT " + std::to_string(r.first) + " has no episode");
    const std::vector<int64_t> logged = rmcheck::logged_replies(g.rmlog);
    std::vector<int64_t> recorded;
    int64_t game_decisions = 0;
    for (const rmcheck::EpisodeFile &e : eps) {
        for (const featurize::EpisodeStep &s : e.ep.steps) recorded.push_back(s.chosen);
        if (const featurize::RmtbAttr *gd = e.ep.file.attr("game_decisions")) game_decisions += gd->i;
    }
    if (g.seats_ab && recorded != logged)
        fails->push_back(g.name + ": the episodes' replies are not the decision log's (" +
                         std::to_string(recorded.size()) + " vs " + std::to_string(logged.size()) + ")");
    if (game_decisions != static_cast<int64_t>(logged.size()))
        fails->push_back(g.name + ": the episodes' game_decisions sum to " + std::to_string(game_decisions) +
                         ", the decision log holds " + std::to_string(logged.size()));
}

// The dump's decisions in order (the seat frame with a menu, and the omniscient frame of the same
// decision), and the omniscient viewer's schema.
static bool dump_decisions(
    const std::string &path, std::vector<DumpDecision> *out, rmstate::Schema *omni_schema, std::string *err) {
    std::vector<std::string> payloads;
    if (!rmcheck::read_records(path, &payloads, err)) return false;
    rmcheck::FrameDecoder dec;
    DumpDecision cur;
    for (const std::string &p : payloads) {
        rmstate::DecodedFrame frame;
        if (!dec.feed(p, &frame, err)) return false;
        const int viewer = frame.header.viewer;
        const int kind = frame.header.kind;
        if (kind != rmcheck::KIND_STATE) continue;
        const rmstate::Schema &schema = *dec.schema(viewer);
        if (rmcheck::header_value(frame, schema, "terminal")) continue;
        if (viewer == rmcheck::VIEWER_OMNISCIENT) {
            if (cur.seat < 0) {
                *err = "a decision of the dump with no seat frame holding a menu";
                return false;
            }
            cur.omni_frame = std::move(frame);
            out->push_back(std::move(cur));
            cur = DumpDecision();
        } else if (rmcheck::Rows(frame, schema, "menu").size() > 0) {
            cur.seat = viewer;
            cur.seat_frame = std::move(frame);
        }
    }
    *omni_schema = *dec.schema(rmcheck::VIEWER_OMNISCIENT);
    return true;
}

// Every recorded step against the dump written beside the run: the same (seat, game, sideboarding,
// decision) sequence, the seat's tensors those featurize_seat gives on the dump's seat frame and the
// omniscient ones (and targets) those of its omniscient frame.
static void check_against_frames(const Featurization &f, const TrajGame &g,
    const std::vector<rmcheck::EpisodeFile> &eps, std::vector<std::string> *fails, size_t *checked) {
    std::vector<DumpDecision> decisions;
    rmstate::Schema schema;
    std::string err;
    if (!dump_decisions(g.dump_path, &decisions, &schema, &err)) {
        fails->push_back(g.name + ": the dump: " + err);
        return;
    }
    size_t d = 0;
    for (const rmcheck::EpisodeFile &e : eps)
        for (size_t i = 0; i < e.ep.steps.size() && fails->size() < MAX_FAILS; ++i, ++d) {
            const featurize::EpisodeStep &s = e.ep.steps[i];
            const std::string at = g.name + "/" + e.name + " step " + std::to_string(i) + ": ";
            if (d >= decisions.size()) {
                fails->push_back(at + "past the dump's " + std::to_string(decisions.size()) + " decisions");
                return;
            }
            const DumpDecision &dd = decisions[d];
            featurize::Tensors t;
            if (dd.seat != s.seat || !featurize::featurize_seat(f.spec, f.tables, dd.seat_frame, &t, &err)) {
                fails->push_back(at + "the dump's decision is seat " + std::to_string(dd.seat) + "'s " + err);
                continue;
            }
            std::string diff = rmcheck::compare_tensors(s.tensors, t);
            if (!diff.empty()) fails->push_back(at + diff + " differs from the dump's frame featurized");
            if (g.omniscient) {
                featurize::Targets tg;
                if (!featurize::featurize_viewer(f.spec, f.tables, dd.omni_frame, &t, &err) ||
                    !featurize::decision_targets(f.spec, schema, dd.seat_frame, dd.omni_frame, &tg, &err)) {
                    fails->push_back(at + "the dump's omniscient frame: " + err);
                    continue;
                }
                diff = rmcheck::compare_tensors(s.omniscient, t);
                if (!diff.empty()) fails->push_back(at + "omniscient " + diff + " differs from the dump's frame");
                if (tg.opp_hand != s.targets.opp_hand || tg.own_library != s.targets.own_library ||
                    tg.opp_library != s.targets.opp_library)
                    fails->push_back(at + "the targets differ from the dump's");
            }
            ++*checked;
        }
    if (d != decisions.size())
        fails->push_back(
            g.name + ": " + std::to_string(d) + " steps, the dump " + std::to_string(decisions.size()) + " decisions");
}

static std::string float_json(float v) {
    char buf[40];
    std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v));
    return buf;
}

// One step's scalars, μ and evaluations as JSON (the tensors go in the tensors file).
static std::string step_json(const featurize::EpisodeStep &s) {
    std::string out =
        "{\"seat\":" + std::to_string(s.seat) + ",\"game\":" + std::to_string(s.game) +
        ",\"decision\":" + std::to_string(s.decision) + ",\"sideboarding\":" + (s.sideboarding ? "true" : "false") +
        ",\"chosen\":" + std::to_string(s.chosen) + ",\"policy_version\":" + std::to_string(s.policy_version) +
        ",\"teacher\":" + std::to_string(s.teacher) + ",\"mu\":";
    if (!s.has_mu) {
        out += "null";
    } else {
        out += "[";
        for (size_t i = 0; i < s.mu.size(); ++i) out += (i ? "," : "") + float_json(s.mu[i]);
        out += "]";
    }
    out += ",\"policies\":[";
    for (size_t k = 0; k < s.policies.size(); ++k) {
        const featurize::PolicyEvalRecord &p = s.policies[k];
        out += std::string(k ? "," : "") + "[" + std::to_string(p.id) + "," + std::to_string(p.version) + "," +
               float_json(p.value) + ",[";
        for (size_t r = 0; r < p.log_pi.size(); ++r) out += (r ? "," : "") + float_json(p.log_pi[r]);
        out += "]]";
    }
    return out + "]}";
}

// python/pycheck.py --part traj reads the episodes with python/rmtraj.py: every step's scalars and μ,
// and its tensors (and the omniscient tensors and targets) as tensors and targets files the C++
// reader's view was written to.
static void python_part(const rmcheck::Options &opt, rmcheck::Report &rep, const Featurization &f,
    const std::vector<std::pair<std::string, const rmcheck::EpisodeFile *>> &eps) {
    if (opt.python.empty()) {
        rep.info("traj: pycheck skipped (no python3 on PATH)");
        return;
    }
    const std::string dir = opt.out("traj");
    std::string json = "{\"episodes\":[";
    for (size_t i = 0; i < eps.size(); ++i) {
        const rmcheck::EpisodeFile &e = *eps[i].second;
        std::vector<featurize::Tensors> seat, omni;
        std::vector<featurize::Targets> targets;
        for (const featurize::EpisodeStep &s : e.ep.steps) {
            seat.push_back(s.tensors);
            if (!e.ep.omniscient) continue;
            omni.push_back(s.omniscient);
            targets.push_back(s.targets);
        }
        const std::string stem = rmcheck::path_join(dir, "py_" + eps[i].first);
        std::string err;
        featurize::rmtb_write_file(stem + "_tensors.rmtb", featurize::tensors_file(f.spec, seat), &err);
        json += std::string(i ? "," : "") + "{\"path\":" + featurize::json_quote(e.path) +
                ",\"tensors\":" + featurize::json_quote(stem + "_tensors.rmtb");
        if (e.ep.omniscient) {
            featurize::rmtb_write_file(stem + "_omni.rmtb", featurize::tensors_file(f.spec, omni), &err);
            featurize::rmtb_write_file(stem + "_targets.rmtb", featurize::targets_file(f.spec, targets), &err);
            json += ",\"omniscient\":" + featurize::json_quote(stem + "_omni.rmtb") +
                    ",\"targets\":" + featurize::json_quote(stem + "_targets.rmtb");
        }
        json += ",\"steps\":[";
        for (size_t k = 0; k < e.ep.steps.size(); ++k) json += (k ? "," : "") + step_json(e.ep.steps[k]);
        json += "]}";
    }
    json += "]}\n";
    const std::string expected = rmcheck::path_join(dir, "pycheck_traj.json");
    rmcheck::write_file(expected, json);
    rmcheck::ProcSpec spec;
    spec.argv = {opt.python, rmcheck::path_join(opt.repo, "python/pycheck.py"), "--binary", opt.binary, "--part",
        "traj", "--traj-expected", expected};
    spec.timeout_s = 900;
    spec.merge_stderr = true;
    const rmcheck::ProcResult r = rmcheck::run_process(spec);
    const std::string log = rmcheck::path_join(dir, "pycheck_traj.txt");
    rmcheck::write_file(log, r.out);
    for (const std::string &line : rmcheck::split_lines(r.out)) {
        const std::string t = rmcheck::trim(line);
        if (rmcheck::starts_with(t, "FAIL:")) rep.error("traj", "pycheck: " + t.substr(6));
        if (rmcheck::starts_with(t, "OK ")) rep.info("traj: pycheck: " + t.substr(5));
    }
    if (!r.ok()) rep.error("traj", "python/pycheck.py --part traj " + r.status_text() + " (see " + log + ")");
}

// Bytes per decision of an episode, by array (the seat's tensors and the omniscient ones).
static void report_sizes(rmcheck::Report &rep, const rmcheck::EpisodeFile &e) {
    const double d = static_cast<double>(std::max<size_t>(e.ep.steps.size(), 1));
    std::string seat, omni;
    double seat_total = 0, omni_total = 0, total = 0;
    for (const featurize::RmtbArray &a : e.ep.file.arrays) {
        const double per = static_cast<double>(a.data.size()) / d;
        total += per;
        const bool o = rmcheck::starts_with(a.name, "omni_") || a.name == "opp_hand" || a.name == "own_library" ||
                       a.name == "opp_library" || rmcheck::ends_with(a.name, "library_offsets") ||
                       a.name == "opp_hand_offsets";
        (o ? omni_total : seat_total) += per;
        if (per >= 20) (o ? omni : seat) += " " + a.name + "=" + std::to_string(static_cast<long>(per));
    }
    rep.info("traj: " + e.name + ": " + std::to_string(static_cast<long>(total)) + " bytes/decision: seat " +
             std::to_string(static_cast<long>(seat_total)) + " (" + rmcheck::trim(seat) + "), omniscient " +
             std::to_string(static_cast<long>(omni_total)) + " (" + rmcheck::trim(omni) + ")");
}

// The behaviour-cloning corpus's path (training plan B0): the scripted agent (Python, over the
// RMSTATE stream) against explore in a bo3 match through python/rm_driver.py, its decision log kept
// (--rmlog-dir), then re-featurized by --replay --trajectory-out: μ-absent episodes, one per game,
// the replies the log's and the footers the replay's GAME_RESULT lines.
static void driver_part(const rmcheck::Options &opt, rmcheck::Report &rep) {
    if (opt.python.empty()) {
        rep.info("traj: the rm_driver.py corpus path skipped (no python3 on PATH)");
        return;
    }
    TrajGame g;
    g.name = "rm_driver";
    g.dir = rmcheck::path_join(opt.out("traj"), g.name);
    rmcheck::remove_tree(g.dir);
    const std::string logs = rmcheck::path_join(g.dir, "rmlogs");
    g.replay_dir = rmcheck::path_join(g.dir, "episodes");
    rmcheck::ProcSpec drive;
    drive.argv = {opt.python, rmcheck::path_join(opt.repo, "python/rm_driver.py"), "--player-a", "scripted",
        "--player-b", "explore", "--deck-a", "league/bug", "--deck-b", "league/burn", "--format", "bo3", "--games", "1",
        "--seed", std::to_string(opt.seed + 400), "--quiet", "--rmlog-dir", logs, "--binary", opt.binary};
    drive.timeout_s = 900;
    drive.merge_stderr = true;
    const rmcheck::ProcResult d = rmcheck::run_process(drive);
    const std::vector<std::string> names = rmcheck::list_dir(logs, ".rmlog");
    if (!d.ok() || names.size() != 1) {
        rep.error("traj", "rm_driver.py --rmlog-dir " + d.status_text() + ", " + std::to_string(names.size()) +
                              " decision logs kept (" + g.dir + ")");
        return;
    }
    g.rmlog = rmcheck::path_join(logs, names[0]);
    rmcheck::ProcSpec replay;
    replay.argv = {opt.binary, "--replay", g.rmlog, "--trajectory-out", g.replay_dir};
    replay.cwd = opt.bin_dir;
    replay.timeout_s = 900;
    replay.merge_stderr = true;
    g.live = rmcheck::run_process(replay);
    std::vector<rmcheck::ScanLine> errors;
    rmcheck::scan_transcript(g.live.out, &errors, nullptr);
    std::vector<rmcheck::EpisodeFile> eps;
    std::string err;
    if (!g.live.ok() || !errors.empty() || !rmcheck::read_episodes(g.replay_dir, &eps, &err) || eps.empty()) {
        rep.error("traj", "rm_driver: the replay " + g.live.status_text() + " " +
                              (errors.empty() ? err : errors[0].line) + " (" + g.dir + ")");
        return;
    }
    std::vector<std::string> fails;
    check_game(g, eps, &fails);
    const std::string stem = rmcheck::strip_ext(names[0]);
    for (const rmcheck::EpisodeFile &e : eps)
        if (!rmcheck::starts_with(e.name, (stem + ".g").c_str()))
            fails.push_back("rm_driver: " + e.name + " is not named after its decision log " + names[0]);
    for (size_t k = 0; k < fails.size() && k < MAX_FAILS; ++k) rep.error("traj", fails[k]);
    if (fails.empty())
        rep.info("traj: rm_driver: a scripted vs explore match's kept log re-featurized: " +
                 std::to_string(eps.size()) + " episodes");
}

namespace rmcheck {

// The traj tier (docs/ci.md): trajectory episodes (--trajectory-out) of in-process agents' runs and
// their --replay: the replay writes the same episodes (every tensor byte for byte), the replies are
// the decision log's, the decision keys and seat tensors those of the run's dump featurized (the
// omniscient ones and the targets too, present only when asked), μ a distribution with mass at the
// reply, one episode per game with the footers the GAME_RESULT lines give (a conceded game, a match
// conceded in sideboarding, a capped run), python/rmtraj.py reading what the C++ reader reads, and
// a scripted match's decision log kept by python/rm_driver.py re-featurized (driver_part).
void tier_traj(const Options &opt, Report &rep) {
    const double t0 = now_seconds();
    Featurization f;
    if (!load_featurization(opt, rep, &f)) return;
    std::vector<TrajGame> games = traj_games(opt);
    parallel_for(games.size(), opt.jobs, [&](size_t i) { run_game(opt, &games[i]); });
    std::vector<std::vector<EpisodeFile>> live(games.size());
    std::vector<std::pair<std::string, const EpisodeFile *>> for_python;
    size_t episodes = 0, steps = 0, framed = 0;
    for (size_t i = 0; i < games.size(); ++i) {
        TrajGame &g = games[i];
        std::vector<std::string> fails;
        std::vector<ScanLine> errors;
        scan_transcript(g.live.out + g.replay.out, &errors, nullptr);
        if (!g.live.ok() || !g.replay.ok() || !errors.empty() || g.rmlog.empty()) {
            rep.error("traj", g.name + ": the run " + g.live.status_text() + ", its replay " + g.replay.status_text() +
                                  (errors.empty() ? "" : ": " + errors[0].line) + " (" + g.dir + ")");
            continue;
        }
        std::vector<EpisodeFile> replayed;
        std::string err;
        if (!read_episodes(g.live_dir, &live[i], &err) || !read_episodes(g.replay_dir, &replayed, &err)) {
            rep.error("traj", g.name + ": " + err);
            continue;
        }
        fails = compare_replayed(live[i], replayed);
        check_game(g, live[i], &fails);
        if (g.dump) check_against_frames(f, g, live[i], &fails, &framed);
        for (size_t k = 0; k < fails.size() && k < MAX_FAILS; ++k) rep.error("traj", fails[k]);
        episodes += live[i].size();
        for (const EpisodeFile &e : live[i]) steps += e.ep.steps.size();
        if (fails.empty())
            rep.info("traj: " + g.name + ": " + std::to_string(live[i].size()) + " episodes, replayed identically");
    }
    if (!live[0].empty()) {
        report_sizes(rep, live[0][0]);
        for_python.emplace_back("bo1", &live[0][0]);
    }
    if (live[1].size() > 1) for_python.emplace_back("bo3_g2", &live[1][1]);
    python_part(opt, rep, f, for_python);
    driver_part(opt, rep);
    traj_teacher_part(opt, rep);
    rep.info("traj: " + std::to_string(episodes) + " episodes, " + std::to_string(steps) + " decisions (" +
             std::to_string(framed) + " against the dumps' frames) in " + fmt1(now_seconds() - t0) + "s");
}

}  // namespace rmcheck
