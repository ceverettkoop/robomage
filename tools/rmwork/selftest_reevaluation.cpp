#include "selftest_reevaluation.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <thread>

#include "eval_cache.h"
#include "eval_connection.h"
#include "files.h"
#include "inference_server.h"
#include "paths.h"
#include "reevaluation.h"
#include "selftest_fake_server.h"
#include "selftest_tools.h"

namespace {

// How long a re-evaluation or a server may take in the checks.
const double REEVAL_TIMEOUT_S = 600;

}  // namespace

static void synthetic_match(uint64_t log_hash, rmwork::Match *out);
static std::vector<std::string> entries_in(const std::string &root);
static bool wait_reeval(rmwork::App &app, const rmwork::ReevalHooks &h);
static std::map<std::pair<size_t, int64_t>, featurize::PolicyEvalRecord> reeval_records(const rmwork::Match &m);
static bool same_records(const std::map<std::pair<size_t, int64_t>, featurize::PolicyEvalRecord> &a,
    const std::map<std::pair<size_t, int64_t>, featurize::PolicyEvalRecord> &b);
static void check_merged_values(const rmwork::CheckFn &check, const rmwork::App &app, const rmwork::ReevalHooks &h,
    const std::vector<uint32_t> &policies);
static rmwork::ModelSettings models_with(
    const std::string &socket, const std::vector<std::pair<std::string, std::string>> &policies, bool on_open);
static bool flip_byte(const std::string &path, size_t at);

// A bo1-like match of six decisions (seats A B A B A B, menus of 2 and 3 rows), game 1 won by A;
// decisions 0 and 1 carry a recorded episode step of policy 0.
static void synthetic_match(uint64_t log_hash, rmwork::Match *out) {
    rmwork::Match &m = *out;
    m.log_hash = log_hash;
    m.dump_path = "/nonexistent/dump.rmst";
    for (int i = 0; i < 6; ++i) {
        rmwork::DecisionInfo d;
        d.decision = i;
        d.decider = i % 2;
        d.menu_rows = 2 + i % 2;
        d.reply = 0;
        d.phase = "First main";
        d.turn = i / 2;
        m.decisions.push_back(d);
    }
    for (int i = 0; i < 2; ++i) {
        rmwork::EpisodeStepInfo e;
        e.file = "recorded.rmtraj";
        e.seat = i % 2;
        e.policy_version = 4;
        featurize::PolicyEvalRecord r;
        r.id = 0;
        r.version = 4;
        r.value = 0.25f;
        r.log_pi.assign(static_cast<size_t>(m.decisions[static_cast<size_t>(i)].menu_rows), -1.0f);
        e.policies.push_back(r);
        m.episode_steps.push_back(e);
        m.decisions[static_cast<size_t>(i)].episode_step = i;
    }
    rmwork::GameSpan g;
    g.first = 0;
    g.count = 6;
    g.label = "Game 1";
    m.games.push_back(g);
    m.result.game_winners = {0};
}

// The entries (`.rmev`) under `root`.
static std::vector<std::string> entries_in(const std::string &root) {
    std::vector<std::string> out;
    DIR *d = opendir(root.c_str());
    if (!d) return out;
    while (const dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".rmev") == 0) out.push_back(root + "/" + n);
    }
    closedir(d);
    return out;
}

// Frames until the App's re-evaluation is no longer running; false past the timeout.
static bool wait_reeval(rmwork::App &app, const rmwork::ReevalHooks &h) {
    const auto start = std::chrono::steady_clock::now();
    while (app.reevaluation().running()) {
        h.frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > REEVAL_TIMEOUT_S)
            return false;
    }
    h.frames(2);
    return true;
}

// Every re-evaluated record of the match, by (decision, id).
static std::map<std::pair<size_t, int64_t>, featurize::PolicyEvalRecord> reeval_records(const rmwork::Match &m) {
    std::map<std::pair<size_t, int64_t>, featurize::PolicyEvalRecord> out;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const int s = m.decisions[i].episode_step;
        if (s < 0) continue;
        for (const featurize::PolicyEvalRecord &r : m.episode_steps[static_cast<size_t>(s)].policies)
            if (rmwork::is_reevaluated(r.id)) out[{i, r.id}] = r;
    }
    return out;
}

static bool same_records(const std::map<std::pair<size_t, int64_t>, featurize::PolicyEvalRecord> &a,
    const std::map<std::pair<size_t, int64_t>, featurize::PolicyEvalRecord> &b) {
    if (a.size() != b.size()) return false;
    for (const auto &kv : a) {
        auto it = b.find(kv.first);
        if (it == b.end() || it->second.value != kv.second.value || it->second.log_pi != kv.second.log_pi ||
            it->second.version != kv.second.version)
            return false;
    }
    return true;
}

// Every decision of the open match re-evaluated by each of `policies`, in its deciding seat's own
// frame: the record's value the stand-in's of that frame's token count, π over its menu.
static void check_merged_values(const rmwork::CheckFn &check, const rmwork::App &app, const rmwork::ReevalHooks &h,
    const std::vector<uint32_t> &policies) {
    using namespace rmwork;
    const Match &m = *app.replay().match();
    featurize::Featurization f;
    std::string why;
    if (!load_engine_features(h.engine, h.workdir, h.tmp + "/check_cards.rmst", nullptr, &f, &why)) {
        check(false, "Re-evaluation: the check's featurizer inputs: " + why);
        return;
    }
    size_t wrong_value = 0, missing = 0, wrong_seat = 0, not_pi = 0, seats_seen = 0;
    std::set<int> seats;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const DecisionInfo &d = m.decisions[i];
        std::string err;
        const auto frame = m.frame(i, d.decider, &err);
        featurize::Tensors t;
        if (!frame || !featurize::featurize_seat(f.spec, f.tables, *frame, &t, &err)) {
            check(false, "Re-evaluation: decision " + std::to_string(i + 1) + " does not featurize: " + err);
            return;
        }
        const int s = d.episode_step;
        if (s < 0) {
            ++missing;
            continue;
        }
        const EpisodeStepInfo &e = m.episode_steps[static_cast<size_t>(s)];
        if (e.seat != d.decider) ++wrong_seat;
        seats.insert(e.seat);
        for (uint32_t p : policies) {
            const featurize::PolicyEvalRecord *r = find_policy(e, reevaluated_id(p));
            if (!r) {
                ++missing;
                continue;
            }
            if (std::fabs(r->value - FakeServer::value_of(t, p)) > 1e-6f) ++wrong_value;
            double sum = 0;
            for (float l : r->log_pi) sum += std::exp(static_cast<double>(l));
            if (static_cast<int>(r->log_pi.size()) != d.menu_rows || std::fabs(sum - 1) > 1e-4) ++not_pi;
        }
    }
    seats_seen = seats.size();
    check(missing == 0, "Re-evaluation: " + std::to_string(missing) + " decisions or records missing");
    check(wrong_seat == 0, "Re-evaluation: " + std::to_string(wrong_seat) + " steps of another seat than the decider");
    check(wrong_value == 0,
        "Re-evaluation: " + std::to_string(wrong_value) + " values are not those of the deciding seat's own frame");
    check(not_pi == 0, "Re-evaluation: " + std::to_string(not_pi) + " \xCF\x80 not a distribution over the menu");
    check(seats_seen == 2, "Re-evaluation: not both seats' decisions evaluated");
}

static rmwork::ModelSettings models_with(
    const std::string &socket, const std::vector<std::pair<std::string, std::string>> &policies, bool on_open) {
    rmwork::ModelSettings m;
    m.socket = socket;
    for (const auto &p : policies) m.policies.push_back({p.first, p.second});
    m.reevaluate_on_open = on_open;
    return m;
}

// Flips byte `at` of the file; false when it cannot.
static bool flip_byte(const std::string &path, size_t at) {
    std::string bytes, err;
    if (!rmwork::read_file(path, &bytes) || at >= bytes.size()) return false;
    bytes[at] = static_cast<char>(bytes[at] ^ 0x5a);
    return rmwork::write_file_atomic(path, bytes, &err);
}

namespace rmwork {

void check_reevaluation_units(const CheckFn &check, const std::string &tmp) {
    // The cache's entries.
    CachedEvals c;
    c.log_hash = 0x1234;
    c.identity = "/x/model.ts.pt\n10\n20";
    c.seat = 1;
    c.policy = 3;
    c.version = 77;
    for (uint32_t i = 1; i < 6; i += 2) {
        CachedDecision d;
        d.index = i;
        d.decision = i;
        d.value = 0.125f * static_cast<float>(i);
        d.log_pi.assign(3, -1.0986f);
        c.decisions.push_back(d);
    }
    const std::string bytes = eval_cache_bytes(c);
    CachedEvals back;
    std::string err;
    check(parse_eval_cache(bytes, &back, &err) && back.identity == c.identity && back.seat == 1 && back.policy == 3 &&
              back.version == 77 && back.decisions.size() == 3 && back.decisions[1].index == 3 &&
              back.decisions[2].value == c.decisions[2].value && back.decisions[0].log_pi == c.decisions[0].log_pi,
        "Re-evaluation cache: an entry does not round-trip: " + err);
    std::string damaged = bytes;
    damaged[damaged.size() / 2] = static_cast<char>(damaged[damaged.size() / 2] ^ 1);
    check(!parse_eval_cache(damaged, &back, &err), "Re-evaluation cache: a damaged entry reads");
    check(!parse_eval_cache(bytes.substr(0, bytes.size() - 9), &back, &err),
        "Re-evaluation cache: a truncated entry reads");
    check(!parse_eval_cache("RMIX" + bytes.substr(4), &back, &err), "Re-evaluation cache: a foreign file reads");
    const std::string root = tmp + "/evals";
    const std::string id_a = "/m/a\n1\n2", id_b = "/m/b\n1\n2";
    const std::string pa = eval_cache_path(root, 0x1234, id_a, 0), pb = eval_cache_path(root, 0x1234, id_b, 0);
    check(pa != pb && pa != eval_cache_path(root, 0x1234, id_a, 1) && pa != eval_cache_path(root, 0x1235, id_a, 0),
        "Re-evaluation cache: the paths do not differ per log, model and seat");
    check(write_eval_cache(pa, c, &err) && read_eval_cache(pa, &back, &err) && back.version == 77,
        "Re-evaluation cache: an entry is not written and read back: " + err);
    check(!read_eval_cache(pb, &back, &err), "Re-evaluation cache: a missing entry reads");
    // The model's identity follows its file.
    const std::string model = tmp + "/model.ts.pt";
    write_file_atomic(model, "weights", &err);
    const std::string i1 = model_identity(model);
    write_file_atomic(model, "other weights", &err);
    const std::string i2 = model_identity(model);
    check(i1 != i2 && i1.compare(0, 6, "missing") != 0, "Re-evaluation cache: the model's identity ignores its file");
    check(model_identity(tmp + "/none.pt").compare(0, 8, "missing\n") == 0,
        "Re-evaluation cache: a missing model has an identity of a file");
    check(model_identity(tmp + "/m.zip!a/final.ts.pt") != model_identity(tmp + "/m.zip!b/final.ts.pt"),
        "Re-evaluation cache: a zipped model's member is not part of its identity");
    // Fitting a match.
    Match m;
    synthetic_match(0x1234, &m);
    CachedEvals fit;
    fit.log_hash = 0x1234;
    fit.seat = 1;
    for (uint32_t i = 1; i < 6; i += 2) {
        CachedDecision d;
        d.index = i;
        d.decision = i;
        d.log_pi.assign(static_cast<size_t>(m.decisions[i].menu_rows), -1.0f);
        fit.decisions.push_back(d);
    }
    std::string why;
    check(eval_cache_fits(fit, m, 1, &why), "Re-evaluation cache: a fitting entry does not fit: " + why);
    check(!eval_cache_fits(fit, m, 0, &why), "Re-evaluation cache: an entry fits the other seat");
    CachedEvals bad = fit;
    bad.decisions[1].log_pi.push_back(-2);
    check(!eval_cache_fits(bad, m, 1, &why), "Re-evaluation cache: an entry of another menu size fits");
    bad = fit;
    bad.decisions[2].decision = 9;
    check(!eval_cache_fits(bad, m, 1, &why), "Re-evaluation cache: an entry of another decision key fits");
    bad = fit;
    bad.decisions.pop_back();
    check(!eval_cache_fits(bad, m, 1, &why), "Re-evaluation cache: an entry missing a decision fits");
    bad = fit;
    bad.log_hash = 1;
    check(!eval_cache_fits(bad, m, 1, &why), "Re-evaluation cache: an entry of another log fits");
    // Pruning: the oldest go first, never the one kept; a stale .tmp goes.
    const std::string proot = tmp + "/prune";
    make_dirs(proot);
    std::vector<std::string> files;
    for (int i = 0; i < 4; ++i) {
        files.push_back(proot + "/e" + std::to_string(i) + ".rmev");
        write_file_atomic(files.back(), std::string(1000, 'x'), &err);
        const struct timespec times[2] = {{1000 + i, 0}, {1000 + i, 0}};
        utimensat(AT_FDCWD, files.back().c_str(), times, 0);
    }
    write_file_atomic(proot + "/old.rmev.tmp", "x", &err);
    const struct timespec old_times[2] = {{1000, 0}, {1000, 0}};
    utimensat(AT_FDCWD, (proot + "/old.rmev.tmp").c_str(), old_times, 0);
    prune_eval_cache(proot, 2500, files[0]);
    check(file_exists(files[0]) && !file_exists(files[1]) && !file_exists(files[2]) && file_exists(files[3]) &&
              !file_exists(proot + "/old.rmev.tmp"),
        "Re-evaluation cache: pruning does not remove the oldest entries but the one kept, and stale .tmp files");
    // Merging into a match.
    Match mm;
    synthetic_match(0x1234, &mm);
    std::vector<std::pair<size_t, featurize::PolicyEvalRecord>> recs;
    for (size_t i = 0; i < 6; i += 2) {
        featurize::PolicyEvalRecord r;
        r.version = 12;
        r.value = 0.1f * static_cast<float>(i);
        r.log_pi.assign(2, std::log(0.5f));
        recs.emplace_back(i, r);
    }
    const unsigned changed = mm.evaluations_changed;
    merge_evaluations(&mm, reevaluated_id(0), recs);
    check(mm.evaluations_changed == changed + 1, "Re-evaluation: a merge does not count as a change");
    check(mm.decisions[0].episode_step == 0 && mm.episode_steps[0].policies.size() == 2 &&
              mm.episode_steps[0].policies[0].id == 0 && mm.episode_steps[0].policies[1].id == reevaluated_id(0),
        "Re-evaluation: a recorded step does not keep its record first with the re-evaluated one after it");
    const int s2 = mm.decisions[2].episode_step;
    check(s2 >= 2 && mm.episode_steps[static_cast<size_t>(s2)].file == REEVAL_STEP_FILE &&
              mm.episode_steps[static_cast<size_t>(s2)].seat == 0 &&
              mm.episode_steps[static_cast<size_t>(s2)].chosen == 0,
        "Re-evaluation: a decision with no step does not get one of its own for its deciding seat");
    MatchValues v = match_values(mm);
    check(v.policies == std::vector<int64_t>{0, reevaluated_id(0)} && v.find(0, reevaluated_id(0)) &&
              v.find(0, reevaluated_id(0))->points.size() == 3 && v.find(0, 0)->points.size() == 1 &&
              !v.find(1, reevaluated_id(0)),
        "Re-evaluation: the merged series is not a series of its own beside the recorded one");
    // A second policy merged before a lower one lands after it; a merge again replaces.
    merge_evaluations(&mm, reevaluated_id(5), recs);
    merge_evaluations(&mm, reevaluated_id(2), recs);
    const std::vector<featurize::PolicyEvalRecord> &ps = mm.episode_steps[0].policies;
    check(ps.size() == 4 && ps[1].id == reevaluated_id(0) && ps[2].id == reevaluated_id(2) &&
              ps[3].id == reevaluated_id(5),
        "Re-evaluation: the re-evaluated records are not in id order after the recorded ones");
    recs[0].second.value = -0.75f;
    merge_evaluations(&mm, reevaluated_id(0), {recs[0]});
    check(mm.episode_steps[0].policies.size() == 4 &&
              find_policy(mm.episode_steps[0], reevaluated_id(0))->value == -0.75f,
        "Re-evaluation: merging a policy again does not replace its record");
    check(policy_label(reevaluated_id(3), 9) == "re p3 v9" && policy_label(3, 9) == "p3 v9" &&
              policy_name(reevaluated_id(0)) == "re p0" && !is_reevaluated(0xFFFFFFFFll) &&
              is_reevaluated(reevaluated_id(0)),
        "Re-evaluation: the policies' labels do not tell recorded from re-evaluated");
    // The served policies.
    ModelSettings ms;
    ms.policies = {{"0", "a.ts.pt"}, {"latest", "b.ts.pt"}, {"2", "c.ts.pt"}, {"02", "d.ts.pt"}};
    const std::vector<ServedPolicy> sp = served_policies(ms);
    check(sp.size() == 2 && sp[0].id == 0 && sp[1].id == 2 && sp[1].model == policy_model_path(ms.policies[2]),
        "Re-evaluation: the served policies are not the numeric ids with their models");
    // From the cache alone; with no server; Play's live match.
    {
        ReplayEvaluator ev;
        ReplayEvaluator::Config cfg;
        cfg.socket = tmp + "/no_server.sock";
        ev.configure(cfg);
        Match cm;
        synthetic_match(0x99, &cm);
        const std::string croot = tmp + "/evals_cm";
        const std::string mpath = tmp + "/cached_model.ts.pt";
        write_file_atomic(mpath, "cached", &err);
        for (int seat = 0; seat < 2; ++seat) {
            CachedEvals e;
            e.log_hash = 0x99;
            e.identity = model_identity(mpath);
            e.seat = seat;
            e.version = 31;
            for (uint32_t i = static_cast<uint32_t>(seat); i < 6; i += 2) {
                CachedDecision d;
                d.index = i;
                d.decision = i;
                d.value = 0.5f;
                d.log_pi.assign(static_cast<size_t>(cm.decisions[i].menu_rows), -0.5f);
                e.decisions.push_back(d);
            }
            write_eval_cache(eval_cache_path(croot, 0x99, e.identity, seat), e, &err);
        }
        Reevaluation r;
        ReevalRequest req;
        req.policies = {{4, mpath}};
        req.cache_root = croot;
        req.use_server = false;
        r.start(&cm, &ev, req);
        r.poll(&cm);
        check(r.state() == Reevaluation::State::DONE && r.merged().size() == 2 && r.merged().begin()->second,
            "Re-evaluation: the cache alone does not merge its entries (" + r.summary() + ")");
        const MatchValues cv = match_values(cm);
        check(cv.find(0, reevaluated_id(4)) && cv.find(1, reevaluated_id(4)) &&
                  cv.find(1, reevaluated_id(4))->points.size() == 3 && cv.versions.at(reevaluated_id(4)) == 31,
            "Re-evaluation: the cached series are not both seats'");
        // Another model: nothing cached, and no server to ask.
        Match nm;
        synthetic_match(0x99, &nm);
        req.policies = {{4, tmp + "/model.ts.pt"}};
        req.use_server = true;
        r.start(&nm, &ev, req);
        const auto start = std::chrono::steady_clock::now();
        while (r.running() && std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < 30) {
            r.poll(&nm);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        check(r.state() == Reevaluation::State::FAILED && r.no_server() && nm.evaluations_changed == 0 &&
                  entries_in(croot).size() == 2,
            "Re-evaluation: with no server it does not fail as no server, merging and writing nothing (" + r.summary() +
                ")");
        Match live;
        synthetic_match(0x99, &live);
        live.live = true;
        r.start(&live, &ev, req);
        check(r.state() == Reevaluation::State::FAILED && live.evaluations_changed == 0,
            "Re-evaluation: Play's live match is re-evaluated");
    }
}

void check_reevaluation_data(const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log) {
    const std::string models = h.tmp + "/reeval_models";
    make_dirs(models);
    std::string err;
    const std::string model_a = models + "/a.ts.pt", model_b = models + "/b.ts.pt";
    write_file_atomic(model_a, "model a", &err);
    write_file_atomic(model_b, "model b", &err);
    const std::string root = eval_cache_root();
    const std::vector<std::pair<std::string, std::string>> two = {{"0", model_a}, {"1", model_b}};
    const ModelSettings saved = app.settings().models;
    // No server: on open nothing is asked and no dialog opens; asked, the dialog says so.
    app.set_models(models_with(h.tmp + "/none.sock", two, true));
    if (!h.open(log)) return;
    check(!app.error_open() && app.reevaluation().state() == Reevaluation::State::DONE &&
              !app.replay().match()->evaluations_changed,
        "Re-evaluation: opening with no server and nothing cached does not end quietly (" +
            app.reevaluation().summary() + ")");
    check(app.open_reevaluate() && app.reevaluate_dialog().is_open(), "Re-evaluation: the dialog does not open");
    h.frames(2);
    check(app.reevaluate_dialog().chosen().size() == 2 && app.reevaluate_dialog().seats()[0] &&
              app.reevaluate_dialog().seats()[1],
        "Re-evaluation: the dialog does not start with every served policy and both seats");
    h.escape();
    check(!app.reevaluate_dialog().is_open(), "Re-evaluation: Esc does not close the dialog");
    ReevalRequest both;
    both.policies = served_policies(app.settings().models);
    app.reevaluate(both, true);
    wait_reeval(app, h);
    check(app.reevaluation().state() == Reevaluation::State::FAILED && app.reevaluation().no_server() &&
              app.error_open() && app.error_title() == "No inference server",
        "Re-evaluation: asked with no server it does not say so (" + app.error_title() + ")");
    h.escape();
    // The stand-in server.
    FakeServer fake;
    FakeServer::Options o;
    o.versions = {{0, 5}, {1, 9}};
    const std::string sock = h.tmp + "/fake.sock";
    if (!fake.start(sock, o)) {
        check(false, "Re-evaluation: the stand-in server does not listen at " + sock);
        app.set_models(saved);
        return;
    }
    app.set_models(models_with(sock, two, false));
    check(h.open(log), "Re-evaluation: the log does not open again");
    const auto t0 = std::chrono::steady_clock::now();
    app.reevaluate(both, true);
    check(wait_reeval(app, h), "Re-evaluation: it does not finish");
    const double took = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const Match &m = *app.replay().match();
    // The match is replaced at each open below: what the checks need of it is kept here.
    const size_t decisions = m.decisions.size();
    const uint64_t log_hash = m.log_hash;
    FakeServer::Counts n1 = fake.counts();
    check(app.reevaluation().state() == Reevaluation::State::DONE && app.reevaluation().merged().size() == 4,
        "Re-evaluation: with the server it does not finish both policies for both seats (" +
            app.reevaluation().summary() + ")");
    for (const auto &kv : app.reevaluation().merged())
        check(!kv.second, "Re-evaluation: a first re-evaluation came from the cache");
    check(n1.requests > 1 && n1.largest <= o.max_batch && n1.last_policies == std::vector<uint32_t>{0, 1},
        "Re-evaluation: the requests are not batched within the server's max_batch (" + std::to_string(n1.requests) +
            " requests, largest " + std::to_string(n1.largest) + ")");
    check(n1.decisions == m.decisions.size() && app.evaluator().stats().featurized == m.decisions.size(),
        "Re-evaluation: " + std::to_string(n1.decisions) + " decisions evaluated, not the match's " +
            std::to_string(m.decisions.size()) + " each once");
    check(!app.error_open(), "Re-evaluation: a successful one opened the error dialog: " + app.error_title());
    check_merged_values(check, app, h, {0, 1});
    check(entries_in(root).size() == 4,
        "Re-evaluation: " + std::to_string(entries_in(root).size()) + " cache entries, not one per policy and seat");
    check(app.status_message().find("re p0 v5") != std::string::npos &&
              app.status_message().find("re p1 v9") != std::string::npos,
        "Re-evaluation: the status bar does not name the policies: " + app.status_message());
    std::printf("Re-evaluation: %zu decisions, %zu requests (largest %zu) in %.2f s through the stand-in server\n",
        m.decisions.size(), n1.requests, n1.largest, took);
    // The tools on the merged evaluations, and what each view sees.
    {
        const CheckFn fn = [&check](bool ok, const std::string &what) { check(ok, "Re-evaluation tools: " + what); };
        ValuesExpect want;
        want.policies = 2;
        want.from_match = true;
        ToolsCoverage cov;
        check_tools_values(fn, app, h.frames, "re-evaluated", want, &cov);
        check_tools_frames(fn, app, "re-evaluated", &cov);
        check(cov.episode_decisions > 0, "Re-evaluation: no decision checked against its re-evaluation");
    }
    const auto first = reeval_records(m);
    // Reopened: everything from the cache, the versions checked by a connection, nothing evaluated.
    app.set_models(models_with(sock, two, true));
    check(h.open(log) && wait_reeval(app, h), "Re-evaluation: the reopened log does not re-evaluate");
    FakeServer::Counts n2 = fake.counts();
    bool all_cached = app.reevaluation().merged().size() == 4;
    for (const auto &kv : app.reevaluation().merged()) all_cached = all_cached && kv.second;
    check(all_cached && n2.requests == n1.requests && n2.hellos == n1.hellos + 1 &&
              same_records(first, reeval_records(*app.replay().match())),
        "Re-evaluation: reopening is not the cache's results with one version check (" + app.reevaluation().summary() +
            ")");
    // A version the server changed: that policy evaluated again, the other from the cache.
    o.versions[1] = 10;
    fake.set(o);
    check(h.open(log) && wait_reeval(app, h), "Re-evaluation: the reopened log does not re-evaluate");
    FakeServer::Counts n3 = fake.counts();
    const std::map<std::pair<uint32_t, int>, bool> &mg = app.reevaluation().merged();
    check(mg.size() == 4 && mg.at({0, 0}) && mg.at({0, 1}) && !mg.at({1, 0}) && !mg.at({1, 1}) &&
              n3.last_policies == std::vector<uint32_t>{1} && n3.decisions - n2.decisions == decisions &&
              reeval_records(*app.replay().match()).at({0, reevaluated_id(1)}).version == 10,
        "Re-evaluation: a version the server changed is not evaluated again alone (" + app.reevaluation().summary() +
            ")");
    // A damaged entry: that one evaluated again.
    const std::string damaged = eval_cache_path(root, log_hash, model_identity(model_a), 0);
    check(flip_byte(damaged, 40), "Re-evaluation: the entry to damage is not there");
    check(h.open(log) && wait_reeval(app, h), "Re-evaluation: the reopened log does not re-evaluate");
    {
        const std::map<std::pair<uint32_t, int>, bool> &g = app.reevaluation().merged();
        check(g.size() == 4 && !g.at({0, 0}) && g.at({0, 1}) && g.at({1, 0}) && g.at({1, 1}),
            "Re-evaluation: a damaged entry is not evaluated again alone (" + app.reevaluation().summary() + ")");
    }
    // A replaced model: its entries are another model's.
    write_file_atomic(model_b, "model b, trained further", &err);
    check(h.open(log) && wait_reeval(app, h), "Re-evaluation: the reopened log does not re-evaluate");
    {
        const std::map<std::pair<uint32_t, int>, bool> &g = app.reevaluation().merged();
        check(g.size() == 4 && g.at({0, 0}) && g.at({0, 1}) && !g.at({1, 0}) && !g.at({1, 1}),
            "Re-evaluation: a replaced model's entries are used (" + app.reevaluation().summary() + ")");
    }
    // Cancelled mid-way: no entry, no series.
    app.set_models(models_with(sock, two, false));
    remove_tree(root);
    o.delay_ms = 25;
    o.max_batch = 2;
    fake.set(o);
    check(h.open(log), "Re-evaluation: the log does not open again");
    app.reevaluate(both, true);
    const auto c0 = std::chrono::steady_clock::now();
    while (app.reevaluation().running() && app.reevaluation().fraction() <= 0 &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count() < 120) {
        h.frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    check(app.reevaluation().running() && app.reevaluation().fraction() > 0 && app.reevaluation().fraction() < 1,
        "Re-evaluation: no progress to cancel at");
    app.reevaluation().cancel();
    const auto c1 = std::chrono::steady_clock::now();
    while (app.evaluator().busy() && std::chrono::duration<double>(std::chrono::steady_clock::now() - c1).count() < 60)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    h.frames(2);
    check(app.reevaluation().state() == Reevaluation::State::CANCELLED && entries_in(root).empty() &&
              app.replay().match()->evaluations_changed == 0 && !app.evaluator().busy(),
        "Re-evaluation: a cancel leaves an entry or a series (" + app.reevaluation().summary() + ")");
    // The server restarted, refusing the handshake: the kept connection fails, a new one is refused;
    // the error dialog, nothing written.
    o.delay_ms = 0;
    o.max_batch = 7;
    o.refuse = true;
    fake.stop();
    check(fake.start(sock, o), "Re-evaluation: the stand-in server does not listen again");
    app.reevaluate(both, true);
    wait_reeval(app, h);
    check(app.reevaluation().state() == Reevaluation::State::FAILED && !app.reevaluation().no_server() &&
              app.error_open() && app.error_title() == "Re-evaluation failed" &&
              app.reevaluation().error().find("not served") != std::string::npos && entries_in(root).empty(),
        "Re-evaluation: a refused handshake does not fail with the server's refusal: " + app.reevaluation().error());
    h.escape();
    // The server lost after one request: the error dialog, nothing written; then a retry succeeds.
    o.refuse = false;
    o.lose_after = 1;
    fake.set(o);
    app.reevaluate(both, true);
    wait_reeval(app, h);
    check(app.reevaluation().state() == Reevaluation::State::FAILED && app.error_open() &&
              app.error_title() == "Re-evaluation failed" && entries_in(root).empty() &&
              app.replay().match()->evaluations_changed == 0,
        "Re-evaluation: a server lost mid-way does not fail cleanly: " + app.reevaluation().error());
    h.escape();
    o.lose_after = -1;
    fake.set(o);
    app.reevaluate(both, true);
    wait_reeval(app, h);
    check(app.reevaluation().state() == Reevaluation::State::DONE && entries_in(root).size() == 4,
        "Re-evaluation: a retry after a lost server does not succeed (" + app.reevaluation().summary() + ")");
    fake.stop();
    app.set_models(saved);
}

void check_reevaluation_real(
    const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log, const std::string &model) {
    // rmserve serving the reference policy as policy 0, at a short socket path.
    ServerConfig sc;
    sc.binary = parent_dir(h.engine) + "/rmserve";
    sc.socket = h.tmp + "/rmserve.sock";
    sc.cwd = repo_root();
    sc.log_dir = h.tmp + "/rmserve";
    sc.policies = {{"0", model}};
    InferenceServer server;
    server.start(sc);
    const auto s0 = std::chrono::steady_clock::now();
    while (server.state() == InferenceServer::State::STARTING &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count() < 300) {
        server.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (server.state() != InferenceServer::State::RUNNING) {
        check(false, "Re-evaluation: rmserve does not start: " + server.message() + "\n" + server.detail());
        return;
    }
    const ModelSettings saved = app.settings().models;
    app.set_models(models_with(sc.socket, {{"0", model}}, false));
    ReevalRequest r;
    r.policies = served_policies(app.settings().models);
    app.reevaluate(r, true);
    check(wait_reeval(app, h), "Re-evaluation: the real re-evaluation does not finish");
    check(app.reevaluation().state() == Reevaluation::State::DONE,
        "Re-evaluation: through rmserve it does not finish (" + app.reevaluation().summary() + ")");
    const Match &m = *app.replay().match();
    size_t compared = 0, off = 0, missing = 0;
    std::set<int> seats;
    double worst = 0;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const int s = m.decisions[i].episode_step;
        if (s < 0) {
            ++missing;
            continue;
        }
        const EpisodeStepInfo &e = m.episode_steps[static_cast<size_t>(s)];
        const featurize::PolicyEvalRecord *rec = find_policy(e, 0);
        const featurize::PolicyEvalRecord *re = find_policy(e, reevaluated_id(0));
        if (!rec || !re || rec->log_pi.size() != re->log_pi.size()) {
            ++missing;
            continue;
        }
        double d = std::fabs(static_cast<double>(rec->value) - re->value);
        for (size_t k = 0; k < rec->log_pi.size(); ++k)
            d = std::max(d, std::fabs(static_cast<double>(rec->log_pi[k]) - re->log_pi[k]));
        worst = std::max(worst, d);
        if (d > 1e-5) ++off;
        seats.insert(e.seat);
        ++compared;
    }
    check(missing == 0 && compared == m.decisions.size(),
        "Re-evaluation: " + std::to_string(missing) + " decisions without both the recorded and re-evaluated policy");
    check(off == 0, "Re-evaluation: " + std::to_string(off) +
                        " decisions' value or log \xCF\x80 differ from the episode's by more than 1e-5 (worst " +
                        std::to_string(worst) + ")");
    check(seats.size() == 2, "Re-evaluation: not both seats' decisions compared");
    std::printf(
        "Re-evaluation: %zu decisions of both seats through rmserve, largest difference from the episodes "
        "%.3g\n",
        compared, worst);
    const CheckFn fn = [&check](bool ok, const std::string &what) { check(ok, "Re-evaluation tools: " + what); };
    ValuesExpect want;
    want.policies = 2;
    want.agents = {"nn:" + model, "nn:" + model};
    want.every_decision = true;
    want.from_match = true;
    ToolsCoverage cov;
    check_tools_values(fn, app, h.frames, "re-evaluated nn: match", want, &cov);
    // Reopened: from the cache, nothing evaluated.
    const ReplayEvaluator::Stats before = app.evaluator().stats();
    app.set_models(models_with(sc.socket, {{"0", model}}, true));
    check(h.open(log) && wait_reeval(app, h), "Re-evaluation: the nn: match does not reopen");
    bool cached = app.reevaluation().merged().size() == 2;
    for (const auto &kv : app.reevaluation().merged()) cached = cached && kv.second;
    check(cached && app.evaluator().stats().decisions == before.decisions,
        "Re-evaluation: reopening the nn: match evaluates again (" + app.reevaluation().summary() + ")");
    app.set_models(saved);
    server.stop();
}

}  // namespace rmwork
