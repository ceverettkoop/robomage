#include "frames.h"
#include "gate.h"
#include "knowledge_meta.h"
#include "proc.h"
#include "scan.h"
#include "sessions.h"
#include "util.h"

using rmcheck::FuzzGame;
using rmcheck::StreamGame;

// One stream session of the tier (the isolation and belief parts) and what came of it.
struct KnowledgeSession {
        StreamGame game;
        bool belief = false;
        std::string label, dump, saved, err_path;
        std::vector<std::string> fails;
};

static std::vector<std::string> knowledge_flags(bool every_decision);
static std::vector<FuzzGame> fuzz_games();
static std::vector<KnowledgeSession> sessions();
static void selftest_part(const rmcheck::Options &opt, rmcheck::Report &rep);
static void play_session(const rmcheck::Options &opt, const std::string &dir, KnowledgeSession *s);
static bool has_flag(const std::vector<std::string> &flags, const char *flag);

static std::vector<std::string> knowledge_flags(bool every_decision) {
    if (every_decision) return {"--selfcheck-knowledge", "--selfcheck-state-sample", "1"};
    return {"--selfcheck-knowledge"};
}

static bool has_flag(const std::vector<std::string> &flags, const char *flag) {
    for (const std::string &f : flags)
        if (f == flag) return true;
    return false;
}

// The knowledge tier's fuzz games.
static std::vector<FuzzGame> fuzz_games() {
    struct Row {
            const char *a, *b;
            bool bo3;
            long seed;
            bool every, dump;
    };
    static const Row rows[] = {
        {"league/reanimator", "league/uw_control", true, 201, false, true},
        {"mav", "league/ur_delver", false, 202, false, true},
        {"league/bug", "league/wrb_energy", true, 203, false, false},
        {"league/wubg_doomsday", "league/bw_dnt", false, 204, true, false},
        {"mav", "league/reanimator", true, 205, false, false},
        {"league/ur_delver", "league/bug", false, 206, false, false},
    };
    std::vector<FuzzGame> out;
    for (const Row &r : rows) {
        FuzzGame g;
        g.deck_a = r.a;
        g.deck_b = r.b;
        g.bo3 = r.bo3;
        g.seed = r.seed;
        g.flags = knowledge_flags(r.every);
        g.dump = r.dump;
        out.push_back(g);
    }
    return out;
}

// The knowledge tier's RMSTATE sessions (expert, and expert+belief).
static std::vector<KnowledgeSession> sessions() {
    struct Row {
            const char *a, *b;
            bool bo3;
            long seed;
            bool closed, belief, selfcheck;
    };
    static const Row rows[] = {
        {"league/ur_delver", "league/bug", false, 31, false, false, true},
        {"league/reanimator", "league/uw_control", true, 32, false, false, true},
        {"league/bug", "league/wubg_doomsday", true, 33, true, false, true},
        {"league/reanimator", "league/ur_delver", false, 51, true, true, true},
        {"league/burn", "mav", false, 52, false, true, false},
    };
    std::vector<KnowledgeSession> out;
    for (const Row &r : rows) {
        KnowledgeSession s;
        s.game.deck_a = r.a;
        s.game.deck_b = r.b;
        s.game.bo3 = r.bo3;
        s.game.seed = r.seed;
        if (r.belief) s.game.flags = {"--knowledge-a", "expert+belief", "--knowledge-b", "expert+belief"};
        if (r.selfcheck) s.game.flags.push_back("--selfcheck-knowledge");
        if (r.closed) s.game.flags.push_back("--closed-decklist");
        s.belief = r.belief;
        s.label = rmcheck::stream_label(r.belief ? "belief" : "stream", s.game);
        out.push_back(s);
    }
    return out;
}

// rmknow's own checks: the update rules, the exact solver and the belief layer's posterior.
static void selftest_part(const rmcheck::Options &opt, rmcheck::Report &rep) {
    struct Run {
            std::vector<std::string> args;
            const char *summary;
            bool fail_word;  // a line holding FAIL fails it too
    };
    const std::vector<Run> runs = {{{"--selftest"}, "RMKNOW_SELFTEST:", true},
        {{"--exactness", "40"}, "RMKNOW_EXACTNESS:", true},
        {{"--belief-exactness", "10"}, "RMKNOW_BELIEF_EXACTNESS:", false}};
    std::vector<rmcheck::ProcResult> results(runs.size());
    rmcheck::parallel_for(runs.size(), opt.jobs, [&](size_t i) {
        rmcheck::ProcSpec spec;
        spec.argv = {opt.rmknow};
        spec.argv.insert(spec.argv.end(), runs[i].args.begin(), runs[i].args.end());
        spec.timeout_s = 1200;
        results[i] = rmcheck::run_process(spec);
    });
    for (size_t i = 0; i < runs.size(); ++i) {
        const std::string label = "rmknow " + rmcheck::join(runs[i].args, " ");
        std::string last;
        std::vector<std::string> bad;
        for (const std::string &l : rmcheck::split_lines(results[i].out + results[i].err)) {
            if (rmcheck::starts_with(l, runs[i].summary)) last = l;
            if (rmcheck::contains(l, "ERROR") || (runs[i].fail_word && rmcheck::contains(l, "FAIL"))) bad.push_back(l);
        }
        if (!results[i].ok() || last.empty() || !bad.empty())
            rep.error("knowledge", label + ": " + results[i].status_text() + ": " + (bad.empty() ? last : bad[0]));
        else
            rep.info("OK   " + label + ": " + last);
    }
}

// One stream session with its dump: the stream against the dump, saved; the self-check summaries;
// a bo3 session's sideboarding; a belief session's rows, isolation and truth at expert+belief.
static void play_session(const rmcheck::Options &opt, const std::string &dir, KnowledgeSession *s) {
    const std::string stem = std::string(s->belief ? "belief_" : "stream_") + std::to_string(s->game.seed);
    s->dump = rmcheck::path_join(dir, stem + ".rms");
    s->saved = rmcheck::path_join(dir, stem + ".stream.rms");
    s->err_path = rmcheck::path_join(dir, stem + ".err");
    const rmcheck::StreamRun run = rmcheck::play_stream_session(opt, s->game, s->dump);
    rmcheck::write_file(s->err_path, run.proc.err);
    if (!run.error.empty()) {
        s->fails.push_back(s->label + ": " + run.error + " (see " + s->err_path + ")");
        s->dump.clear();
        s->saved.clear();
        return;
    }
    std::vector<std::string> dump;
    std::string err;
    if (!rmcheck::read_records(s->dump, &dump, &err)) s->fails.push_back(s->label + ": " + err);
    for (const std::string &f : rmcheck::check_stream(run.payloads, dump, s->label)) s->fails.push_back(f);
    rmcheck::write_records(s->saved, run.payloads);
    const bool selfcheck = has_flag(s->game.flags, "--selfcheck-knowledge");
    for (const std::string &f : rmcheck::selfcheck_failures(run.proc.err, s->err_path, s->label, selfcheck ? 1 : -1,
             selfcheck ? std::vector<std::string>{"STATE", "KNOWLEDGE"} : std::vector<std::string>{}))
        s->fails.push_back(f);
    rmcheck::FrameDecoder dec;
    bool sideboarding = false;
    size_t b_rows = 0;
    for (const std::string &p : run.payloads) {
        rmstate::DecodedFrame f;
        if (!dec.feed(p, &f, &err)) break;
        if (rmcheck::frame_kind(p) != rmcheck::KIND_STATE) continue;
        const rmstate::Schema &schema = *dec.schema(rmcheck::frame_viewer(p));
        sideboarding = sideboarding || rmcheck::header_value(f, schema, "sideboarding");
        b_rows += rmcheck::Rows(f, schema, "b_list").size();
    }
    if (s->game.bo3 && !sideboarding)
        s->fails.push_back(s->label + ": no sideboarding frame (the session should cover sideboarding)");
    if (!s->belief) return;
    if (!b_rows) s->fails.push_back(s->label + ": no b_list rows at level expert+belief");
    for (const std::string &path : {s->dump, s->saved})
        for (const std::string &f : rmcheck::rmknow_isolation(opt, path, "expert+belief", nullptr))
            s->fails.push_back(s->label + ": " + f);
    for (const char *seat : {"A", "B"})
        for (const std::string &f : rmcheck::rmknow_truth(opt, s->dump, seat, "expert+belief", nullptr))
            s->fails.push_back(s->label + ": " + f);
}

namespace rmcheck {

// The knowledge tier, on the engine and rmknow: selftest (rmknow
// --selftest, --exactness 40, --belief-exactness 10), fuzz (explore games under
// --selfcheck-knowledge), isolation (RMSTATE sessions against their dumps, k_* tables included,
// and rmknow --isolation on every dump and saved stream), truth (rmknow --truth on every dump),
// the belief layer's structure (expert+belief sessions: stream against dump, isolation and truth at that level),
// archetype inference and the belief calibration report (knowledge_meta.h).
void tier_knowledge(const Options &opt, Report &rep) {
    const std::string dir = opt.out("knowledge");
    const double t0 = now_seconds();
    selftest_part(opt, rep);
    // The fuzz games and every stream session in one pool.
    const std::vector<FuzzGame> games = fuzz_games();
    std::vector<FuzzOutcome> outs(games.size());
    std::vector<KnowledgeSession> sess = sessions();
    parallel_for(games.size() + sess.size(), opt.jobs, [&](size_t i) {
        if (i < games.size())
            outs[i] = play_fuzz_game(opt, games[i], dir);
        else
            play_session(opt, dir, &sess[i - games.size()]);
    });
    const double t1 = now_seconds();
    std::vector<std::string> dumps, isolation_files;
    for (const FuzzOutcome &o : outs) {
        std::vector<std::string> fails;
        if (!o.result.ok())
            fails.push_back(o.label + ": engine " + o.result.status_text() + " (see " + o.err_path + ")");
        for (const std::string &f :
            selfcheck_failures(o.result.err, o.err_path, o.label, o.finished ? 1 : -1, {"STATE", "KNOWLEDGE"}))
            fails.push_back(f);
        if (!o.dump.empty()) {
            if (file_exists(o.dump)) {
                dumps.push_back(o.dump);
                isolation_files.push_back(o.dump);
            } else {
                fails.push_back(o.label + ": no --rmstate-dump file was written");
            }
        }
        for (const std::string &f : fails) rep.error("knowledge", f);
        if (fails.empty())
            rep.info("OK   " + o.label + (o.finished ? "" : " (stopped at the decision cap)") + ": " +
                     knowledge_summary(o.result.err));
    }
    for (const KnowledgeSession &s : sess) {
        for (const std::string &f : s.fails) rep.error("knowledge", f);
        if (s.fails.empty()) rep.info("OK   " + s.label + ": stream frames equal the dump's (k_* included)");
        if (s.belief || s.dump.empty()) continue;
        dumps.push_back(s.dump);
        isolation_files.push_back(s.dump);
        isolation_files.push_back(s.saved);
    }
    // rmknow --isolation on every dump and saved stream, --truth on every dump.
    struct Job {
            std::string path, seat;
            bool truth;
            std::vector<std::string> fails;
            std::string summary;
    };
    std::vector<Job> jobs;
    for (const std::string &p : isolation_files) jobs.push_back({p, "", false, {}, ""});
    for (const std::string &p : dumps)
        for (const char *seat : {"A", "B"}) jobs.push_back({p, seat, true, {}, ""});
    parallel_for(jobs.size(), opt.jobs, [&](size_t i) {
        Job &j = jobs[i];
        j.fails = j.truth ? rmknow_truth(opt, j.path, j.seat.c_str(), "expert", &j.summary)
                          : rmknow_isolation(opt, j.path, "expert", &j.summary);
    });
    for (const Job &j : jobs) {
        for (const std::string &f : j.fails) rep.error("knowledge", f);
        if (j.fails.empty())
            rep.info(std::string("OK   ") + (j.truth ? "truth " : "isolation ") + basename_of(j.path) +
                     (j.truth ? " viewer " + j.seat : "") + ": " + j.summary);
    }
    if (isolation_files.empty()) rep.error("knowledge", "isolation: no dump or saved stream to check");
    const double t2 = now_seconds();
    archetype_part(opt, rep, dir);
    calibration_part(opt, rep, dir);
    for (const KnowledgeSession &s : sess) {
        remove_tree(s.dump);
        remove_tree(s.saved);
    }
    for (const FuzzOutcome &o : outs)
        if (!o.dump.empty()) remove_tree(o.dump);
    rep.info("knowledge: selftest + games " + fmt1(t1 - t0) + "s, isolation + truth " + fmt1(t2 - t1) +
             "s, archetype + calibration " + fmt1(now_seconds() - t2) + "s");
}

}  // namespace rmcheck
