#include "selftest_policy_compare.h"

#include <fcntl.h>
#include <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

#include "files.h"
#include "inference_server.h"
#include "paths.h"
#include "policy_compare.h"
#include "selftest_fake_server.h"

namespace {

// How long a re-evaluation, a bring-in or a server may take in the checks.
const double COMPARE_TIMEOUT_S = 600;

// What a decision's metrics are, computed here from the records alone.
struct Expected {
        double kl = 0, tv = 0;
        int ref_argmax = -1, argmax = -1;
};

}  // namespace

static void add_step(rmwork::Match *m, size_t index, const std::vector<std::pair<int64_t, std::vector<double>>> &pis);
static void synthetic_match(rmwork::Match *out);
static bool near(double a, double b, double tol);
static Expected expected_metrics(const featurize::PolicyEvalRecord &ref, const featurize::PolicyEvalRecord &p);
static bool wait_idle(rmwork::App &app, const rmwork::ReevalHooks &h);
static void touch_at(const std::string &path, long seconds);

// Decision `index`'s episode step with each listed policy's π (its log stored, as an episode holds it).
static void add_step(rmwork::Match *m, size_t index, const std::vector<std::pair<int64_t, std::vector<double>>> &pis) {
    rmwork::EpisodeStepInfo e;
    e.file = "made-up.rmtraj";
    e.seat = m->decisions[index].decider;
    for (const auto &p : pis) {
        featurize::PolicyEvalRecord r;
        r.id = p.first;
        r.version = 7;
        r.value = 0.1f * static_cast<float>(index) - 0.2f;
        for (double x : p.second) r.log_pi.push_back(static_cast<float>(std::log(x)));
        e.policies.push_back(r);
    }
    m->decisions[index].episode_step = static_cast<int>(m->episode_steps.size());
    m->episode_steps.push_back(e);
}

// Six decisions (seats A B A B A B, menus of 2 and 3 rows, turns 1 1 2 2 3 3). Policy 0 (recorded) evaluated
// decisions 1 and 3 (seat B's), re p1 every decision, re p2 all but the last.
static void synthetic_match(rmwork::Match *out) {
    using namespace rmwork;
    Match &m = *out;
    for (int i = 0; i < 6; ++i) {
        DecisionInfo d;
        d.decision = i;
        d.decider = i % 2;
        d.menu_rows = 2 + i % 2;
        d.reply = 1;
        d.phase = "First main";
        d.turn = i / 2;
        m.decisions.push_back(d);
    }
    const int64_t r1 = reevaluated_id(1), r2 = reevaluated_id(2);
    add_step(&m, 0, {{r1, {0.9, 0.1}}, {r2, {0.5, 0.5}}});
    add_step(&m, 1, {{0, {0.65, 0.2, 0.15}}, {r1, {0.2, 0.5, 0.3}}, {r2, {0.2, 0.5, 0.3}}});
    add_step(&m, 2, {{r1, {0.6, 0.4}}, {r2, {0.3, 0.7}}});
    add_step(&m, 3, {{0, {1.0 / 3, 1.0 / 3, 1.0 / 3}}, {r1, {0.1, 0.1, 0.8}}, {r2, {1.0 / 3, 1.0 / 3, 1.0 / 3}}});
    add_step(&m, 4, {{r1, {0.8, 0.2}}, {r2, {0.8, 0.2}}});
    add_step(&m, 5, {{r1, {0.1, 0.2, 0.7}}});
    GameSpan g;
    g.count = 6;
    g.label = "Game 1";
    m.games.push_back(g);
    m.result.game_winners = {0};
}

static bool near(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

static Expected expected_metrics(const featurize::PolicyEvalRecord &ref, const featurize::PolicyEvalRecord &p) {
    Expected e;
    for (size_t i = 0; i < ref.log_pi.size() && i < p.log_pi.size(); ++i) {
        const double r = std::exp(static_cast<double>(ref.log_pi[i])), q = std::exp(static_cast<double>(p.log_pi[i]));
        e.kl += r * (static_cast<double>(ref.log_pi[i]) - static_cast<double>(p.log_pi[i]));
        e.tv += std::fabs(r - q) / 2;
        if (e.ref_argmax < 0 || ref.log_pi[i] > ref.log_pi[static_cast<size_t>(e.ref_argmax)])
            e.ref_argmax = static_cast<int>(i);
        if (e.argmax < 0 || p.log_pi[i] > p.log_pi[static_cast<size_t>(e.argmax)]) e.argmax = static_cast<int>(i);
    }
    return e;
}

// Frames until no bring-in waits and no re-evaluation runs; false past the timeout.
static bool wait_idle(rmwork::App &app, const rmwork::ReevalHooks &h) {
    const auto start = std::chrono::steady_clock::now();
    h.frames(2);
    while (app.compare_waiting() || app.reevaluation().running()) {
        h.frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > COMPARE_TIMEOUT_S)
            return false;
    }
    h.frames(3);
    return true;
}

// Sets the file's modification time.
static void touch_at(const std::string &path, long seconds) {
    const struct timespec times[2] = {{seconds, 0}, {seconds, 0}};
    utimensat(AT_FDCWD, path.c_str(), times, 0);
}

namespace rmwork {

void check_policy_compare_units(const CheckFn &check) {
    // The metrics on made-up distributions.
    const std::vector<float> half = {std::log(0.5f), std::log(0.5f)}, skew = {std::log(0.9f), std::log(0.1f)};
    check(near(kl_divergence(half, skew), 0.5108256, 1e-5) && near(kl_divergence(half, half), 0, 1e-12) &&
              near(kl_divergence(skew, half), 0.3680642, 1e-5),
        "Policy Compare: KL is not 0.5 ln(0.5/0.9) + 0.5 ln(0.5/0.1) for (0.5, 0.5) against (0.9, 0.1)");
    check(near(total_variation(half, skew), 0.4, 1e-6) && near(total_variation(skew, half), 0.4, 1e-6),
        "Policy Compare: TV is not 0.4 between (0.5, 0.5) and (0.9, 0.1)");
    check(argmax_row(half) == 0 && argmax_row(skew) == 0 && argmax_row({-2.0f, -0.1f, -0.1f}) == 1 &&
              argmax_row({}) == -1,
        "Policy Compare: the argmax is not the first row of the largest \xCF\x80");
    check(std::isfinite(kl_divergence(half, {0.0f, -1e30f})), "Policy Compare: KL is not finite at a row q excludes");
    check(metric_text(0.03125) == "0.031" && metric_text(0.0001) == "<0.001" && metric_text(0) == "0.000",
        "Policy Compare: the metrics' text");
    // A synthetic match against re p2.
    Match m;
    synthetic_match(&m);
    const int64_t r1 = reevaluated_id(1), r2 = reevaluated_id(2);
    const MatchValues v = match_values(m);
    check(comparable_policies(v, VIEW_OMNI) == std::vector<int64_t>{0, r1, r2} &&
              comparable_policies(v, VIEW_A) == std::vector<int64_t>{r1, r2} &&
              comparable_policies(v, VIEW_B) == std::vector<int64_t>{0, r1, r2},
        "Policy Compare: the candidates are not the policies with a series of a seat the viewer sees");
    check(default_reference({0, r1, r2}) == r2 && default_reference({}) == -1,
        "Policy Compare: the default reference is not the last policy");
    const MatchComparison c = compare_match(m, VIEW_OMNI, {0, r1, r2}, r2);
    check(c.ready() && c.decisions.size() == 5 && !c.find(5) && c.find(3) && c.position(r1) == 1,
        "Policy Compare: the comparison does not hold the five decisions re p2 evaluated");
    struct Want {
            size_t index;
            double kl, tv;
            bool agrees;
    };
    // re p1 against re p2, hand-computed.
    const Want want1[] = {{0, 0.5108256, 0.4, true}, {1, 0, 0, true}, {2, 0.1837869, 0.3, false},
        {3, 0.5108256, 0.4666667, false}, {4, 0, 0, true}};
    for (const Want &w : want1) {
        const DecisionCompare *d = c.find(w.index);
        const PolicyAt *p = d ? &d->at[1] : nullptr;
        check(p && p->has && near(p->kl, w.kl, 1e-5) && near(p->tv, w.tv, 1e-5) && p->agrees == w.agrees,
            "Policy Compare: re p1 at decision " + std::to_string(w.index + 1) +
                " is not KL / TV / agreement as computed by hand");
    }
    const DecisionCompare *d1 = c.find(1);
    check(d1 && d1->at[0].has && near(d1->at[0].kl, 0.4303586, 1e-5) && near(d1->at[0].tv, 0.45, 1e-5) &&
              !d1->at[0].agrees && d1->ref == 2 && d1->at[2].kl == 0 && d1->differs && near(d1->max_tv, 0.45, 1e-5),
        "Policy Compare: policy 0 at decision 2 is not KL 0.430, TV 0.45, another argmax");
    check(c.find(0) && !c.find(0)->at[0].has && !c.find(0)->differs,
        "Policy Compare: a policy that did not evaluate a decision counts there");
    // Agreement.
    check(c.agreement.size() == 2 && c.agreement[0].policy == 0 && c.agreement[1].policy == r1,
        "Policy Compare: the agreement is not per policy but the reference");
    if (c.agreement.size() == 2) {
        const PolicyAgreement &a0 = c.agreement[0], &a1 = c.agreement[1];
        check(a0.all.n == 2 && a0.all.agree == 1 && a0.seat[0].n == 0 && a0.seat[1].n == 2,
            "Policy Compare: policy 0's agreement is not 1 of 2, seat B's");
        check(a1.all.n == 5 && a1.all.agree == 3 && a1.seat[0].n == 3 && a1.seat[0].agree == 2 && a1.seat[1].n == 2 &&
                  a1.seat[1].agree == 1 && near(a1.all.kl, 1.2054381, 1e-4) && near(a1.all.rate(), 0.6, 1e-12),
            "Policy Compare: re p1's agreement is not 3 of 5 (A 2 of 3, B 1 of 2) with KL summing to 1.205");
        check(c.bands.size() == 3 && a1.bands.size() == 3 && a1.bands[0].n == 2 && a1.bands[0].agree == 2 &&
                  a1.bands[1].n == 2 && a1.bands[1].agree == 0 && a1.bands[2].n == 1,
            "Policy Compare: re p1's agreement per turn is not T1 2/2, T2 0/2, T3 1/1");
        check(agreement_text(a1.all) == "60% (3/5)" && agreement_text(Agreement()) == "\xE2\x80\x94",
            "Policy Compare: the agreement's text");
    }
    // The ranked list.
    const std::vector<RankedDecision> tv = rank_disagreements(c, -1, CompareMetric::TV, false, 10);
    const std::vector<size_t> tv_order = {3, 1, 0, 2, 4};
    bool order_ok = tv.size() == tv_order.size();
    for (size_t i = 0; order_ok && i < tv.size(); ++i) order_ok = tv[i].index == tv_order[i];
    check(order_ok && tv[1].policy == 0 && tv[0].policy == r1,
        "Policy Compare: ranked by TV the decisions are not 4, 2, 1, 3, 5 (the most diverging policy each)");
    const std::vector<RankedDecision> flips = rank_disagreements(c, -1, CompareMetric::TV, true, 10);
    check(flips.size() == 3 && flips[0].index == 3 && flips[1].index == 1 && flips[2].index == 2 && flips[0].differs &&
              flips[0].ref_argmax == 0 && flips[0].argmax == 2 && near(flips[2].shift, 0.3, 1e-5),
        "Policy Compare: \"argmax differs only\" does not keep decisions 4, 2 and 3 with their rows and shift");
    const std::vector<RankedDecision> only1 = rank_disagreements(c, r1, CompareMetric::KL, false, 2);
    check(only1.size() == 2 && only1[0].policy == r1 && only1[1].policy == r1 &&
              ((only1[0].index == 0 && only1[1].index == 3) || (only1[0].index == 3 && only1[1].index == 0)),
        "Policy Compare: re p1 alone by KL, at most two, is not decisions 1 and 4");
    // A seat view: only its seat's decisions.
    const MatchComparison ca = compare_match(m, VIEW_A, {r1, r2}, r2);
    bool seat_ok = ca.decisions.size() == 3;
    for (const DecisionCompare &d : ca.decisions) seat_ok = seat_ok && d.seat == 0;
    check(seat_ok && !ca.find(1), "Policy Compare: seat A's view compares decisions other than its own");
    check(!compare_match(m, VIEW_OMNI, {r1}, r2).ready() && !compare_match(m, VIEW_OMNI, {0, r1}, r2).ready(),
        "Policy Compare: a comparison without its reference, or of one policy, is ready");
}

void check_policy_compare_data(const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log) {
    const ModelSettings saved = app.settings().models;
    ToolFlags was{};
    for (int t = 0; t < TOOL_COUNT; ++t)
        was[static_cast<size_t>(t)] = app.tool_open(Mode::REPLAY, static_cast<Tool>(t));
    PolicyCompareView &v = app.compare_view();
    const auto restore = [&] {
        v.set_reference(-1);
        v.force_hover(-1);
        v.set_argmax_only(false);
        v.set_metric(CompareMetric::TV);
        for (int t = 0; t < TOOL_COUNT; ++t)
            app.set_tool_open(Mode::REPLAY, static_cast<Tool>(t), was[static_cast<size_t>(t)]);
        app.focus_tool(Tool::TABLE);
        app.set_models(saved);
        h.frames(3);
    };
    const std::string models = h.tmp + "/compare_models";
    make_dirs(models);
    std::string err;
    std::vector<std::pair<std::string, std::string>> three;
    for (int i = 0; i < 3; ++i) {
        const std::string path = models + "/p" + std::to_string(i) + ".ts.pt";
        write_file_atomic(path, "compare model " + std::to_string(i), &err);
        three.push_back({std::to_string(i), path});
    }
    FakeServer fake;
    FakeServer::Options o;
    o.versions = {{0, 11}, {1, 12}, {2, 13}};
    const std::string sock = h.tmp + "/compare.sock";
    if (!fake.start(sock, o)) {
        check(false, "Policy Compare: the stand-in server does not listen at " + sock);
        return;
    }
    ModelSettings ms;
    ms.socket = sock;
    for (const auto &p : three) ms.policies.push_back({p.first, p.second});
    ms.reevaluate_on_open = false;
    ms.run_dir = h.tmp + "/compare_runs";
    app.set_models(ms);
    if (!h.open(log)) {
        restore();
        return;
    }
    app.set_tool_open(Mode::REPLAY, Tool::COMPARE, true);
    app.focus_tool(Tool::COMPARE);
    ReplaySession &s = app.replay();
    s.set_view(VIEW_OMNI);
    h.frames(4);
    check(v.state() == PolicyCompareView::State::NO_POLICIES && v.candidates().empty(),
        "Policy Compare: a replay with no evaluation is not \"no evaluations to compare\"");
    // One policy.
    const std::vector<ServedPolicy> served = served_policies(app.settings().models);
    ReevalRequest r;
    r.policies = {served.front()};
    app.reevaluate(r, true);
    check(wait_idle(app, h) && app.reevaluation().state() == Reevaluation::State::DONE,
        "Policy Compare: the re-evaluation with one policy does not finish (" + app.reevaluation().summary() + ")");
    check(v.state() == PolicyCompareView::State::ONE_POLICY && v.candidates().size() == 1,
        "Policy Compare: one evaluated policy is not \"one policy\"");
    // Three.
    r.policies = served;
    app.reevaluate(r, true);
    check(wait_idle(app, h) && app.reevaluation().state() == Reevaluation::State::DONE,
        "Policy Compare: the re-evaluation with three policies does not finish (" + app.reevaluation().summary() + ")");
    const int64_t r0 = reevaluated_id(0), r1 = reevaluated_id(1), r2 = reevaluated_id(2);
    const Match &m = *s.match();
    check(v.state() == PolicyCompareView::State::READY && v.candidates() == std::vector<int64_t>{r0, r1, r2} &&
              v.comparison().ref == r2,
        "Policy Compare: three re-evaluated policies are not compared against the last");
    // Every decision against values computed from the merged records.
    const MatchComparison &c = v.comparison();
    size_t compared = 0, wrong = 0, missing = 0, flips = 0;
    std::array<size_t, 2> agree{{0, 0}}, total{{0, 0}};
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const int step = m.decisions[i].episode_step;
        const featurize::PolicyEvalRecord *ref =
            step >= 0 ? find_policy(m.episode_steps[static_cast<size_t>(step)], r2) : nullptr;
        if (!ref) {
            ++missing;
            continue;
        }
        const DecisionCompare *d = c.find(i);
        if (!d || d->at.size() != 3) {
            ++wrong;
            continue;
        }
        for (size_t k = 0; k < 2; ++k) {
            const featurize::PolicyEvalRecord *p =
                find_policy(m.episode_steps[static_cast<size_t>(step)], k == 0 ? r0 : r1);
            if (!p) {
                ++missing;
                continue;
            }
            const Expected e = expected_metrics(*ref, *p);
            const PolicyAt &a = d->at[k];
            if (!a.has || !near(a.kl, e.kl, 1e-6) || !near(a.tv, e.tv, 1e-6) || a.argmax != e.argmax ||
                a.agrees != (e.argmax == e.ref_argmax))
                ++wrong;
            if (k == 1) {
                total[static_cast<size_t>(d->seat)]++;
                if (e.argmax == e.ref_argmax) agree[static_cast<size_t>(d->seat)]++;
                flips += e.argmax == e.ref_argmax ? 0 : 1;
            }
        }
        ++compared;
    }
    check(missing == 0 && compared == m.decisions.size() && c.decisions.size() == compared,
        "Policy Compare: " + std::to_string(missing) + " records missing, " + std::to_string(compared) + " of " +
            std::to_string(m.decisions.size()) + " decisions compared");
    check(wrong == 0, "Policy Compare: " + std::to_string(wrong) +
                          " decisions' KL, TV or argmax are not those computed from the merged records");
    check(flips > 0 && flips < compared,
        "Policy Compare: the stand-in's policies 1 and 2 never (or always) disagree: " + std::to_string(flips));
    if (c.agreement.size() == 2) {
        const PolicyAgreement &a1 = c.agreement[1];
        check(a1.policy == r1 && a1.all.n == compared && a1.all.agree == compared - flips && a1.seat[0].n == total[0] &&
                  a1.seat[0].agree == agree[0] && a1.seat[1].agree == agree[1],
            "Policy Compare: re p1's agreement counts are not those computed from the records");
    } else {
        check(false, "Policy Compare: the agreement is not two policies'");
    }
    std::printf("Policy Compare: %zu decisions, re p1 against re p2 agrees at %zu (%zu argmax differences)\n", compared,
        compared - flips, flips);
    // The ranked list, filtered, and a jump.
    v.set_metric(CompareMetric::TV);
    v.set_argmax_only(false);
    h.frames(2);
    const std::vector<RankedDecision> ranked = v.ranked();
    bool sorted = !ranked.empty();
    for (size_t i = 1; i < ranked.size(); ++i) sorted = sorted && ranked[i - 1].score >= ranked[i].score;
    check(sorted && ranked.size() == std::min<size_t>(200, compared),
        "Policy Compare: the ranked list is not every decision by TV, largest first");
    v.set_argmax_only(true);
    h.frames(2);
    bool all_differ = !v.ranked().empty();
    for (const RankedDecision &x : v.ranked()) all_differ = all_differ && x.differs && x.argmax != x.ref_argmax;
    check(all_differ && v.ranked().size() <= ranked.size(),
        "Policy Compare: \"argmax differs only\" keeps a decision where the argmax agrees");
    if (!v.ranked().empty()) {
        const size_t target = v.ranked().front().index;
        s.go_to(target == 0 ? 1 : 0);
        h.frames(2);
        v.force_jump(0);
        h.frames(3);
        check(s.cursor() == target, "Policy Compare: a click on the ranked list's first row does not go to decision " +
                                        std::to_string(target + 1));
        // The decision panel's row hovered links its menu row.
        v.force_hover(1);
        h.frames(3);
        check(v.current() && v.current()->index == target && app.links().hovered().kind == LinkItem::Kind::MENU_ROW &&
                  app.links().hovered().row == 1,
            "Policy Compare: hovering a decision panel row does not link its menu row");
        v.force_hover(-1);
    }
    v.set_argmax_only(false);
    // Every viewer.
    for (const int view : {VIEW_A, VIEW_B, VIEW_OMNI}) {
        s.set_view(view);
        h.frames(3);
        size_t want = 0;
        for (const DecisionInfo &d : m.decisions) want += view == VIEW_OMNI || d.decider == view ? 1 : 0;
        bool own = v.state() == PolicyCompareView::State::READY && v.comparison().decisions.size() == want;
        for (const DecisionCompare &d : v.comparison().decisions) own = own && (view == VIEW_OMNI || d.seat == view);
        check(
            own, std::string("Policy Compare: ") + view_name(view) + "'s view does not compare exactly its decisions");
        if (view == VIEW_OMNI) continue;
        for (size_t i = 0; i < m.decisions.size(); ++i)
            if (m.decisions[i].decider != view) {
                s.go_to(i);
                h.frames(2);
                check(!v.current(),
                    std::string("Policy Compare: ") + view_name(view) + "'s view shows the other seat's decision");
                break;
            }
    }
    // The catalog: a made-up runs directory, two of its policies brought in.
    const std::string runs = ms.run_dir;
    const std::string alpha = runs + "/alpha/exports/final.ts.pt", beta = runs + "/beta/exports/final.ts.pt";
    make_dirs(parent_dir(alpha));
    make_dirs(parent_dir(beta));
    write_file_atomic(alpha, "alpha final", &err);
    write_file_atomic(beta, "beta final", &err);
    touch_at(alpha, 1700000000);
    touch_at(beta, 1750000000);
    v.open_picker();
    h.frames(3);
    std::string alpha_play, beta_play;
    for (const PolicyRow &row : v.picker_rows()) {
        if (row.run == "alpha" && row.name == "final") alpha_play = row.play;
        if (row.run == "beta" && row.name == "final") beta_play = row.play;
    }
    check(v.picker_open() && !alpha_play.empty() && !beta_play.empty(),
        "Policy Compare: the catalog popup does not list the runs directory's two policies");
    h.escape();
    check(!v.picker_open(), "Policy Compare: Esc does not close the catalog popup");
    const size_t requests = fake.counts().requests;
    check(app.compare_bring_in({alpha_play, beta_play}), "Policy Compare: bringing two checkpoints in is refused");
    check(wait_idle(app, h), "Policy Compare: the bring-in does not finish");
    const ModelSettings &now = app.settings().models;
    std::string alpha_id, beta_id;
    for (const PolicyEntry &p : now.policies) {
        if (policy_model_path(p) == resolve_path(alpha_play, "")) alpha_id = p.id;
        if (policy_model_path(p) == resolve_path(beta_play, "")) beta_id = p.id;
    }
    check(now.policies.size() == 5 && alpha_id == "3" && beta_id == "4" && now.start_server,
        "Policy Compare: the checkpoints are not served under the next ids, the older first");
    check(app.reevaluation().state() == Reevaluation::State::DONE && fake.counts().requests > requests &&
              fake.counts().last_policies == std::vector<uint32_t>{3, 4},
        "Policy Compare: the replay is not re-evaluated with the brought-in policies alone (" +
            app.reevaluation().summary() + ")");
    const int64_t r3 = reevaluated_id(3), r4 = reevaluated_id(4);
    const std::vector<int64_t> &cand = v.candidates();
    check(v.state() == PolicyCompareView::State::READY && std::count(cand.begin(), cand.end(), r3) == 1 &&
              std::count(cand.begin(), cand.end(), r4) == 1 && v.comparison().ref == r4 &&
              v.comparison().decisions.size() == m.decisions.size() && app.compare_note().empty(),
        "Policy Compare: the brought-in policies are not compared, the newest the reference (" + app.compare_note() +
            ")");
    // Not in Play: refused.
    app.set_mode(Mode::PLAY);
    check(!app.compare_bring_in({alpha_play}), "Policy Compare: a bring-in in Play is not refused");
    app.set_mode(Mode::REPLAY);
    fake.stop();
    restore();
}

void check_policy_compare_real(
    const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log, const std::string &model) {
    ServerConfig sc;
    sc.binary = parent_dir(h.engine) + "/rmserve";
    sc.socket = h.tmp + "/rmserve_compare.sock";
    sc.cwd = repo_root();
    sc.log_dir = h.tmp + "/rmserve_compare";
    sc.policies = {{"0", model}, {"1", model}};
    InferenceServer server;
    server.start(sc);
    const auto s0 = std::chrono::steady_clock::now();
    while (server.state() == InferenceServer::State::STARTING &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - s0).count() < 300) {
        server.poll();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    if (server.state() != InferenceServer::State::RUNNING) {
        check(false, "Policy Compare: rmserve does not start: " + server.message() + "\n" + server.detail());
        return;
    }
    const ModelSettings saved = app.settings().models;
    ModelSettings ms;
    ms.socket = sc.socket;
    ms.policies = {{"0", model}, {"1", model}};
    ms.reevaluate_on_open = false;
    app.set_models(ms);
    if (h.open(log)) {
        ReevalRequest r;
        r.policies = served_policies(app.settings().models);
        app.reevaluate(r, true);
        check(wait_idle(app, h) && app.reevaluation().state() == Reevaluation::State::DONE,
            "Policy Compare: the real re-evaluation with two ids does not finish (" + app.reevaluation().summary() +
                ")");
        PolicyCompareView &v = app.compare_view();
        app.replay().set_view(VIEW_OMNI);
        v.set_reference(reevaluated_id(1));
        h.frames(2);
        v.update(app.replay(), app.tools().data());
        const MatchComparison &c = v.comparison();
        const Match &m = *app.replay().match();
        check(v.state() == PolicyCompareView::State::READY &&
                  c.policies == std::vector<int64_t>{0, reevaluated_id(0), reevaluated_id(1)} &&
                  c.ref == reevaluated_id(1) && c.decisions.size() == m.decisions.size(),
            "Policy Compare: the recorded and the two re-evaluated policies are not compared at every decision");
        double worst_same = 0, worst_recorded = 0;
        size_t off = 0;
        for (const DecisionCompare &d : c.decisions) {
            if (!d.at[0].has || !d.at[1].has || !d.at[0].agrees || !d.at[1].agrees) ++off;
            worst_recorded = std::max(worst_recorded, d.at[0].kl);
            worst_same = std::max(worst_same, d.at[1].kl);
        }
        check(off == 0 && worst_same < 1e-6 && worst_recorded < 1e-6,
            "Policy Compare: the same model served twice does not agree everywhere (" + std::to_string(off) +
                " decisions off, KL up to " + std::to_string(worst_same) + ", recorded " +
                std::to_string(worst_recorded) + ")");
        check(c.agreement.size() == 2 && c.agreement[0].all.rate() == 1.0 && c.agreement[1].all.rate() == 1.0 &&
                  c.agreement[1].all.n == m.decisions.size(),
            "Policy Compare: the agreement of the same model served twice is not 100%");
        std::printf(
            "Policy Compare: %zu decisions, the reference policy served twice: KL at most %.3g (the recorded "
            "one %.3g), argmax agreement 100%%\n",
            c.decisions.size(), worst_same, worst_recorded);
        v.set_reference(-1);
    }
    app.set_models(saved);
    server.stop();
}

}  // namespace rmwork
