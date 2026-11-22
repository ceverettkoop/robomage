#include "selftest_attribution.h"

#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

#include "../../src/inference/protocol.h"
#include "eval_cache.h"
#include "files.h"
#include "inference_server.h"
#include "paths.h"
#include "reevaluation.h"
#include "replay_cache.h"
#include "selftest_fake_server.h"
#include "selftest_live.h"

namespace {

// How long an attribution may take in the checks.
const double ATTR_TIMEOUT_S = 600;

}  // namespace

static bool wait_attr(rmwork::App &app, const rmwork::ReevalHooks &h);
static bool wait_reeval_done(rmwork::App &app, const rmwork::ReevalHooks &h);
static rmwork::ModelSettings served(const std::string &socket, const std::string &model);
static bool expected_eval(
    const rmwork::Attribution &a, const featurize::Unit &u, uint32_t policy, rmwork::AttrEval *out);
static long decision_of(const rmwork::Match &m, int seat, size_t from);
static std::vector<std::string> files_in(const std::string &dir);
static const featurize::PolicyEvalRecord *record_of(const rmwork::Match &m, size_t index, int64_t id);
static int row_of_kind(const rmwork::Attribution &a, featurize::UnitKind kind, bool child);
static bool same_eval(const rmwork::AttrEval &a, const featurize::PolicyEvalRecord &r, double tol);

// Frames until the App's attribution is neither running nor draining a job; false past the timeout.
static bool wait_attr(rmwork::App &app, const rmwork::ReevalHooks &h) {
    const auto start = std::chrono::steady_clock::now();
    h.frames(2);
    while (app.attribution().running() || app.attribution().busy()) {
        h.frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > ATTR_TIMEOUT_S)
            return false;
    }
    h.frames(2);
    return true;
}

static bool wait_reeval_done(rmwork::App &app, const rmwork::ReevalHooks &h) {
    const auto start = std::chrono::steady_clock::now();
    while (app.reevaluation().running()) {
        h.frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > ATTR_TIMEOUT_S)
            return false;
    }
    h.frames(2);
    return true;
}

static rmwork::ModelSettings served(const std::string &socket, const std::string &model) {
    rmwork::ModelSettings m;
    m.socket = socket;
    m.policies.push_back({"0", model});
    m.reevaluate_on_open = false;
    return m;
}

// What the stand-in server answers for the decision's tensors changed by `u`.
static bool expected_eval(
    const rmwork::Attribution &a, const featurize::Unit &u, uint32_t policy, rmwork::AttrEval *out) {
    if (!a.tensors() || !a.spec()) return false;
    featurize::Tensors t = *a.tensors();
    featurize::BaselineChoice b;
    std::string err;
    if (!featurize::apply_units(*a.spec(), a.slots(), {&u}, b, &t, &err)) return false;
    out->value = rmwork::FakeServer::value_of(t, policy);
    const std::vector<float> logits = rmwork::FakeServer::logits_of(t, policy);
    out->log_pi.assign(logits.size(), 0.0f);
    if (!logits.empty()) inference::log_softmax(logits.data(), logits.size(), out->log_pi.data());
    return true;
}

// The first decision of `seat` from `from` on with a menu of two rows or more and a permanent on the
// battlefield of each seat, else of `seat` with such a menu; -1 none.
static long decision_of(const rmwork::Match &m, int seat, size_t from) {
    long fallback = -1;
    for (size_t i = from; i < m.decisions.size(); ++i) {
        const rmwork::DecisionInfo &d = m.decisions[i];
        if (d.decider != seat || d.menu_rows < 2 || d.sideboarding) continue;
        if (fallback < 0) fallback = static_cast<long>(i);
        if (d.turn >= 4) return static_cast<long>(i);
    }
    return fallback;
}

static std::vector<std::string> files_in(const std::string &dir) {
    std::vector<std::string> out;
    if (DIR *d = opendir(dir.c_str())) {
        while (const dirent *e = readdir(d)) {
            const std::string n = e->d_name;
            if (n != "." && n != "..") out.push_back(n);
        }
        closedir(d);
    }
    return out;
}

static const featurize::PolicyEvalRecord *record_of(const rmwork::Match &m, size_t index, int64_t id) {
    const int s = m.decisions[index].episode_step;
    return s < 0 ? nullptr : rmwork::find_policy(m.episode_steps[static_cast<size_t>(s)], id);
}

// The first row of a unit of `kind` (a child row when `child`), -1 none.
static int row_of_kind(const rmwork::Attribution &a, featurize::UnitKind kind, bool child) {
    for (size_t i = 0; i < a.rows().size(); ++i) {
        const featurize::Unit *u = a.unit(a.rows()[i]);
        if (u && u->kind == kind && (a.rows()[i].parent >= 0) == child) return static_cast<int>(i);
    }
    return -1;
}

static bool same_eval(const rmwork::AttrEval &a, const featurize::PolicyEvalRecord &r, double tol) {
    if (a.log_pi.size() != r.log_pi.size() || std::fabs(static_cast<double>(a.value) - r.value) > tol) return false;
    for (size_t k = 0; k < a.log_pi.size(); ++k)
        if (std::fabs(static_cast<double>(a.log_pi[k]) - r.log_pi[k]) > tol) return false;
    return true;
}

namespace rmwork {

void check_attribution_units(const CheckFn &check, const std::string &tmp) {
    // The metrics against hand-computed ones.
    AttrEval base;
    base.value = 0.2f;
    base.log_pi = {std::log(0.5f), std::log(0.3f), std::log(0.2f)};
    AttrEval moved;
    moved.value = -0.1f;
    moved.log_pi = {std::log(0.2f), std::log(0.6f), std::log(0.2f)};
    AttrMetrics m = attr_metrics(base, {moved}, 0, 1, false);
    check(m.valid && m.policy && std::fabs(m.dv + 0.3) < 1e-6 && std::fabs(m.tv - 0.3) < 1e-6,
        "Attribution: dv or tv is not the moved evaluation's");
    const double kl = 0.5 * std::log(0.5 / 0.2) + 0.3 * std::log(0.3 / 0.6);
    check(std::fabs(m.kl - kl) < 1e-5 && m.flip == 1.0, "Attribution: kl or flip is not the moved evaluation's");
    check(std::fabs(m.dlogp - std::log(0.2 / 0.5)) < 1e-5 &&
              std::fabs(m.dmargin - (std::log(0.2 / 0.6) - std::log(0.5 / 0.3))) < 1e-5,
        "Attribution: dlogp or dmargin is not the moved evaluation's");
    check(std::fabs(attr_metrics(base, {moved}, 0, 1, true).dv - 0.3) < 1e-6,
        "Attribution: dv on the other seat's scale is not negated");
    check(
        attr_metrics(base, {moved, base}, 0, 1, false).flip == 0.5, "Attribution: the draws' metrics are not averaged");
    AttrEval v_only;
    v_only.value = 0.5f;
    AttrEval v_moved;
    v_moved.value = 0.25f;
    const AttrMetrics mv = attr_metrics(v_only, {v_moved}, -1, -1, false);
    check(mv.valid && !mv.policy && std::fabs(mv.dv + 0.25) < 1e-6 && mv.tv == 0,
        "Attribution: a frame without a menu is not V only");
    check(attr_argmax(base.log_pi) == 0 && attr_runner_up(base.log_pi, 0) == 1 && attr_argmax({}) == -1,
        "Attribution: the argmax or runner-up is wrong");
    // The cache's entries.
    const std::string root = tmp + "/attribution";
    AttrEntry e;
    e.key.dir = attr_cache_dir(root, 0x1234, false, "/x/game.rmlog");
    e.key.game = 1;
    e.key.decision = 17;
    e.key.seat = 1;
    e.key.policy = 2;
    e.key.model = "/models/a.ts.pt\n10\n20";
    e.frame_hash = 99;
    e.spec_hash = 77;
    e.version = 5;
    e.menu_rows = 3;
    e.have_base = true;
    e.base = base;
    featurize::Unit u;
    u.kind = featurize::UnitKind::GROUP;
    u.key = "grp:B:hand";
    u.tokens = {3, 4, 9};
    u.label = "Player B \xC2\xB7 hand";
    u.seat = 1;
    u.coarse = true;
    e.units.push_back(u);
    featurize::Unit t = u;
    t.kind = featurize::UnitKind::TOKEN;
    t.key = "tok:4:24:3:0:3";
    t.tokens = {3};
    t.handle = 41;
    t.owns = true;
    t.coarse = false;
    e.units.push_back(t);
    e.evals[u.key] = {moved};
    e.evals[t.key] = {moved, base};
    e.children[u.key] = {t.key};
    e.players = {u.key, t.key};
    e.coalitions[1] = moved;
    e.coalitions[3] = base;
    const std::string bytes = attr_entry_bytes(e);
    AttrEntry back;
    std::string err;
    check(parse_attr_entry(bytes, &back, &err) && attr_entry_bytes(back) == bytes && back.units.size() == 2 &&
              back.units[1].handle == 41 && back.units[1].owns && back.evals[t.key].size() == 2 &&
              back.children[u.key] == std::vector<std::string>{t.key} && back.coalitions.size() == 2,
        "Attribution: a cache entry does not round trip: " + err);
    std::string damaged = bytes;
    damaged[damaged.size() / 2] = static_cast<char>(damaged[damaged.size() / 2] ^ 0x21);
    check(!parse_attr_entry(damaged, &back, &err) && !parse_attr_entry(bytes.substr(0, bytes.size() - 3), &back, &err),
        "Attribution: a damaged or truncated entry is read");
    check(write_attr_entry(e, root, &err) && file_exists(e.key.path()), "Attribution: an entry is not written: " + err);
    bool tmp_left = false;
    for (const std::string &n : files_in(e.key.dir)) tmp_left = tmp_left || n.find(".tmp") != std::string::npos;
    check(!tmp_left, "Attribution: a written entry leaves a .tmp file");
    check(
        e.key.path() == root + "/0000000000001234/g1d17-b-" + featurize::hex16(fnv1a64(e.key.model)) + "-occ-none.bin",
        "Attribution: the entry's path is not its key's: " + e.key.path());
    check(attr_cache_dir(root, 0x1234, true, "/x/game.rmlog").find(root + "/live-") == 0 &&
              attr_cache_dir(root, 0, false, "/x/game.rmlog") == attr_cache_dir(root, 7, true, "/x/game.rmlog"),
        "Attribution: a live game's entries are not under its log path's directory");
    AttrKey shap = e.key;
    shap.method = AttrMethod::SHAPLEY;
    check(shap.method_name() == "shap-none-k12-m8" && shap != e.key, "Attribution: Shapley's method name");
    // Pruning: the oldest go, the one kept stays.
    AttrEntry other = e;
    other.key.decision = 18;
    write_attr_entry(other, root, &err);
    prune_attr_cache(root, 1, other.key.path());
    check(file_exists(other.key.path()) && !file_exists(e.key.path()),
        "Attribution: pruning does not keep the kept entry and drop the others");
    // Shapley on an additive game: φ exact and efficient.
    AttrEntry g;
    g.key = e.key;
    g.have_base = true;
    g.base.value = 0;
    g.players = {"a", "b", "c"};
    const float w[3] = {0.5f, -0.25f, 0.125f};
    for (uint32_t mask = 1; mask < 8; ++mask) {
        AttrEval x;
        for (int i = 0; i < 3; ++i)
            if (mask & (1u << i)) x.value += w[i];
        g.coalitions[mask] = x;
    }
    const ShapleyResult s = shapley_values(g, AttrTarget::V, -1, -1, false);
    check(s.ready && s.efficient() && std::fabs(s.phi[0] - 0.5) < 1e-6 && std::fabs(s.phi[1] + 0.25) < 1e-6 &&
              std::fabs(s.phi[2] - 0.125) < 1e-6,
        "Attribution: Shapley on an additive game is not exact");
    check(shapley_permutations(5, 42) == shapley_permutations(5, 42) &&
              shapley_permutations(5, 42).size() == static_cast<size_t>(SHAPLEY_PERMUTATIONS),
        "Attribution: Shapley's permutations are not deterministic");
    g.coalitions.erase(7);
    check(!shapley_values(g, AttrTarget::V, -1, -1, false).ready,
        "Attribution: Shapley is ready without every coalition");
}

void check_attribution_data(const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log) {
    const ModelSettings saved = app.settings().models;
    ToolFlags was{};
    for (int t = 0; t < TOOL_COUNT; ++t)
        was[static_cast<size_t>(t)] = app.tool_open(Mode::REPLAY, static_cast<Tool>(t));
    const std::string root = h.tmp + "/attr_cache";
    app.set_attribution_cache_root(root);
    std::string err;
    const std::string model = h.tmp + "/attr_model.ts.pt";
    write_file_atomic(model, "attribution model", &err);
    const auto restore = [&] {
        app.attribution().clear();
        app.attribution_view().set_method(AttrMethod::OCCLUSION);
        app.attribution_view().force_hover("");
        app.set_attribution_layer(true);
        for (int t = 0; t < TOOL_COUNT; ++t)
            app.set_tool_open(Mode::REPLAY, static_cast<Tool>(t), was[static_cast<size_t>(t)]);
        app.focus_tool(Tool::TABLE);
        h.frames(3);
        app.set_models(saved);
        app.set_attribution_cache_root("");
    };
    // No server, nothing cached: nothing attributed, the tool says why.
    app.set_models(served(h.tmp + "/none.sock", model));
    if (!h.open(log)) return;
    ReplaySession &s = app.replay();
    const Match *m = s.match();
    const long ia = decision_of(*m, 0, 4), ib = decision_of(*m, 1, 4);
    check(ia >= 0 && ib >= 0, "Attribution: the log has no decision of each seat to attribute");
    if (ia < 0 || ib < 0) {
        restore();
        return;
    }
    app.set_tool_open(Mode::REPLAY, Tool::ATTRIBUTION, true);
    s.set_view(VIEW_OMNI);
    s.go_to(static_cast<size_t>(ia));
    wait_attr(app, h);
    check(app.attribution_current() && !app.attribution().base() && !app.attribution().running(),
        "Attribution: with no server and nothing cached something is attributed");
    // The stand-in, and a re-evaluation of policy 0 to compare with.
    FakeServer fake;
    FakeServer::Options o;
    o.versions = {{0, 3}};
    const std::string sock = h.tmp + "/attr.sock";
    if (!fake.start(sock, o)) {
        check(false, "Attribution: the stand-in server does not listen at " + sock);
        restore();
        return;
    }
    app.set_models(served(sock, model));
    ReevalRequest rr;
    rr.policies = served_policies(app.settings().models);
    app.reevaluate(rr, true);
    check(wait_reeval_done(app, h) && app.reevaluation().state() == Reevaluation::State::DONE,
        "Attribution: the re-evaluation to compare with does not finish (" + app.reevaluation().summary() + ")");
    m = s.match();
    size_t units_checked = 0, units_wrong = 0;
    for (const long i : {ia, ib}) {
        const int seat = m->decisions[static_cast<size_t>(i)].decider;
        const std::string at =
            std::string("Attribution at decision ") + std::to_string(i + 1) + " (" + seat_name(seat) + ")";
        s.set_view(VIEW_OMNI);
        s.go_to(static_cast<size_t>(i));
        check(wait_attr(app, h), at + ": it does not finish");
        const Attribution &a = app.attribution();
        check(a.state() == Attribution::State::DONE && app.attribution_current() && a.key().seat == seat,
            at + ": not done in the decider's frame: " + a.error());
        check(a.frame() && a.frame()->header.viewer == seat, at + ": the frame attributed is not the decider's own");
        const featurize::PolicyEvalRecord *re = record_of(*m, static_cast<size_t>(i), reevaluated_id(0));
        check(a.base() && re && same_eval(*a.base(), *re, 0),
            at + ": the unchanged evaluation is not the re-evaluation's of the decision");
        size_t slots = 0, slot_units = 0, second = 0;
        if (a.units())
            for (const featurize::Unit &u : a.units()->units) slot_units += u.kind == featurize::UnitKind::SLOT;
        for (const featurize::Unit &u : a.entry()->units) {
            auto it = a.entry()->evals.find(u.key);
            if (it == a.entry()->evals.end()) continue;
            AttrEval want;
            ++units_checked;
            if (!expected_eval(a, u, 0, &want) || it->second.size() != 1 || it->second[0].value != want.value ||
                it->second[0].log_pi != want.log_pi)
                ++units_wrong;
            slots += u.kind == featurize::UnitKind::SLOT;
            second += !u.coarse && u.kind == featurize::UnitKind::TOKEN;
        }
        check(slots == slot_units && slots > 0 && second > 0,
            at + ": the second pass did not evaluate every slot (" + std::to_string(slots) + " of " +
                std::to_string(slot_units) + ") and the top groups' tokens (" + std::to_string(second) + ")");
        std::printf("Attribution: decision %ld (%s) with the stand-in server: %zu units\n", i + 1, seat_name(seat),
            a.entry()->units.size());
    }
    check(units_wrong == 0, "Attribution: " + std::to_string(units_wrong) + " of " + std::to_string(units_checked) +
                                " units' evaluations are not those of the decision's tensors changed by the unit");
    // The cache: the decision visited again is read from it, with a version check only.
    {
        const AttrKey key = app.attribution().key();
        AttrEntry cached;
        check(read_attr_entry(key.path(), &cached, &err) &&
                  cached.units.size() == app.attribution().entry()->units.size(),
            "Attribution: the entry is not cached whole: " + err);
        const FakeServer::Counts before = fake.counts();
        s.go_to(static_cast<size_t>(ia));
        wait_attr(app, h);
        s.go_to(static_cast<size_t>(ib));
        wait_attr(app, h);
        const FakeServer::Counts after = fake.counts();
        check(app.attribution().from_cache() && after.requests == before.requests && after.hellos == before.hellos + 2,
            "Attribution: a decision visited again is not the cache's with a version check (" +
                std::to_string(after.requests - before.requests) + " requests, " +
                std::to_string(after.hellos - before.hellos) + " handshakes)");
    }
    // A row's children asked on expanding it: a token's cells.
    s.go_to(static_cast<size_t>(ia));
    wait_attr(app, h);
    {
        Attribution &a = app.attribution();
        const int tok = row_of_kind(a, featurize::UnitKind::TOKEN, true);
        check(tok >= 0, "Attribution: no token row under a group after the second pass");
        if (tok >= 0) {
            const std::string key = a.rows()[static_cast<size_t>(tok)].key;
            const int32_t type =
                a.unit(a.rows()[static_cast<size_t>(tok)])->tokens.empty()
                    ? -1
                    : a.units()
                          ->tokens[static_cast<size_t>(a.unit(a.rows()[static_cast<size_t>(tok)])->tokens[0])]
                          .type;
            a.expand(tok);
            wait_attr(app, h);
            size_t cells = 0, valid = 0;
            for (const AttrRow &r : a.rows()) {
                const featurize::Unit *u = a.unit(r);
                if (!u || u->kind != featurize::UnitKind::CELL || r.parent < 0 ||
                    a.rows()[static_cast<size_t>(r.parent)].key != key)
                    continue;
                ++cells;
                valid += r.m.valid;
            }
            check(cells > 0 && valid == cells, "Attribution: expanding a token does not evaluate its cells (" +
                                                   std::to_string(valid) + " of " + std::to_string(cells) + ")");
            // The Tokens tool's heat on those cells.
            app.set_tool_open(Mode::REPLAY, Tool::TOKENS, true);
            app.focus_tool(Tool::TOKENS);
            app.tokens_view().set_type(type);
            h.frames(4);
            check(app.tokens_view().heat_cells() > 0,
                "Attribution: the Tokens tool shows no heat on the cells evaluated");
        }
    }
    // The Table layer in the decider's own view: tints on the cards and chips the units name.
    {
        const int seat = m->decisions[static_cast<size_t>(ia)].decider;
        s.set_view(seat);
        app.set_tool_open(Mode::REPLAY, Tool::TABLE, true);
        app.focus_tool(Tool::TABLE);
        h.frames(4);
        const AttrTints &t = app.attribution_tints();
        const std::map<uint64_t, ImU32> &tinted = app.table().tinted_objects();
        size_t wrong = 0;
        for (const auto &kv : tinted)
            if (!t.objects.count(kv.first)) ++wrong;
        size_t untinted = 0, on_table = 0;
        if (const TableModel *tm = app.table().model())
            for (const PlacedCard &c : app.table().layout().cards) {
                const uint64_t hnd = tm->objects[static_cast<size_t>(c.object)].handle;
                if (!t.objects.count(hnd) || !c.front) continue;
                ++on_table;
                if (!tinted.count(hnd)) ++untinted;
            }
        check(t.on && t.same_frame && !tinted.empty() && wrong == 0 && untinted == 0,
            "Attribution: the Table layer's tints do not land on the cards the units name (" +
                std::to_string(tinted.size()) + " tinted, " + std::to_string(wrong) + " without a value, " +
                std::to_string(untinted) + " of " + std::to_string(on_table) + " valued cards untinted)");
        check(app.table().tinted_players()[0] != 0 && app.table().tinted_players()[1] != 0,
            "Attribution: the player chips are not tinted by the players' units");
        app.set_attribution_layer(false);
        h.frames(2);
        check(app.table().tinted_objects().empty() && app.table().attr_legend().w() > 0,
            "Attribution: the layer turned off still tints, or its legend's toggle is gone");
        app.set_attribution_layer(true);
        // A hovered row links its object (its group's row opened, so it is drawn).
        uint64_t handle = 0;
        std::string key;
        for (const AttrRow &r : app.attribution().rows()) {
            const featurize::Unit *u = app.attribution().unit(r);
            if (!u || u->kind != featurize::UnitKind::TOKEN || !u->owns || !u->handle || !r.m.valid) continue;
            key = u->key;
            handle = u->handle;
            for (int p = r.parent; p >= 0; p = app.attribution().rows()[static_cast<size_t>(p)].parent)
                app.attribution_view().open().insert(app.attribution().rows()[static_cast<size_t>(p)].key);
            break;
        }
        app.set_tool_open(Mode::REPLAY, Tool::ATTRIBUTION, true);
        app.focus_tool(Tool::ATTRIBUTION);
        app.attribution_view().force_hover(key);
        h.frames(4);
        check(!key.empty() && app.links().hovered().kind == LinkItem::Kind::OBJECT &&
                  app.links().hovered().handle == handle,
            "Attribution: hovering a token's row does not link its object");
        app.attribution_view().force_hover("");
        // The State Tree's impact: its object's node, and the sort.
        app.set_tool_open(Mode::REPLAY, Tool::TREE, true);
        app.tools().tree().expand_overview();
        app.tools().tree().set_sort_by_impact(true);
        h.frames(3);
        size_t with = 0, unsorted = 0;
        // Per depth the impacts of the current node's children so far (a node's children are checked once
        // the walk leaves them); the roots keep their order.
        std::vector<std::vector<std::pair<bool, double>>> siblings;
        const auto check_list = [&unsorted](const std::vector<std::pair<bool, double>> &l) {
            for (size_t k = 1; k < l.size(); ++k)
                if ((!l[k - 1].first && l[k].first) ||
                    (l[k - 1].first && l[k].first && std::fabs(l[k].second) > std::fabs(l[k - 1].second) + 1e-12))
                    ++unsorted;
        };
        for (const TreeRow &r : app.tools().tree().rows()) {
            double v = 0;
            const bool has = app.tools().tree().impact_of(r, &v);
            with += has;
            const size_t d = static_cast<size_t>(r.depth);
            while (siblings.size() > d + 1) {
                if (siblings.size() > 1) check_list(siblings.back());
                siblings.pop_back();
            }
            while (siblings.size() < d + 1) siblings.emplace_back();
            siblings[d].push_back({has, v});
        }
        while (siblings.size() > 1) {
            check_list(siblings.back());
            siblings.pop_back();
        }
        check(with > 0, "Attribution: the State Tree shows no impact");
        check(unsorted == 0, "Attribution: the State Tree's children are not sorted by impact (" +
                                 std::to_string(unsorted) + " out of order)");
        app.tools().tree().set_sort_by_impact(false);
        // In the omniscient view the decider's frame's handles are mapped onto the Table's.
        s.set_view(VIEW_OMNI);
        app.focus_tool(Tool::TABLE);
        h.frames(4);
        check(!app.attribution_tints().same_frame && !app.table().tinted_objects().empty(),
            "Attribution: the omniscient view's Table shows no tint of the decider's frame's units");
    }
    // The seat-view rule: a seat view at the other seat's decision sees its own frame, which has no menu, and
    // RMIP v2 evaluates only decisions with a menu row: nothing is attributed or sent, and the tool says why.
    {
        const int decider = m->decisions[static_cast<size_t>(ia)].decider, other = 1 - decider;
        const FakeServer::Counts before = fake.counts();
        s.set_view(other);
        s.go_to(static_cast<size_t>(ia));
        wait_attr(app, h);
        const FakeServer::Counts after = fake.counts();
        check(app.attribution_seat() == other && app.attribution_no_menu() && !app.attribution_current() &&
                  after.requests == before.requests && after.hellos == before.hellos,
            "Attribution: a seat view at the other seat's decision is attributed or sent to the server");
    }
    // Shapley: efficient.
    s.set_view(VIEW_OMNI);
    s.go_to(static_cast<size_t>(ia));
    app.attribution_view().set_method(AttrMethod::SHAPLEY);
    check(wait_attr(app, h), "Attribution: Shapley does not finish");
    {
        const ShapleyResult v = app.attribution().shapley(AttrTarget::V),
                            lp = app.attribution().shapley(AttrTarget::LOGP);
        check(v.ready && v.efficient() && lp.ready && lp.efficient(),
            "Attribution: Shapley is not efficient (V: " + std::to_string(v.sum) + " vs " + std::to_string(v.total) +
                ")");
        std::printf("Attribution: Shapley over %zu units: sum %.6f, all changed %.6f\n",
            app.attribution().entry()->players.size(), v.sum, v.total);
    }
    app.attribution_view().set_method(AttrMethod::OCCLUSION);
    // A cancel mid-way: a whole entry or none, and no .tmp.
    o.delay_ms = 30;
    o.max_batch = 2;
    fake.set(o);
    const long ic = decision_of(*m, 0, static_cast<size_t>(ia) + 1);
    if (ic >= 0) {
        s.go_to(static_cast<size_t>(ic));
        h.frames(2);
        const auto c0 = std::chrono::steady_clock::now();
        while ((app.attribution().index() != static_cast<size_t>(ic) || app.attribution().evaluated() < 3) &&
               std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count() < 120)
            h.frames(1);
        const AttrKey key = app.attribution().key();
        check(app.attribution().running(), "Attribution: nothing running to cancel (state " +
                                               std::to_string(static_cast<int>(app.attribution().state())) + " " +
                                               app.attribution().error() + ", " +
                                               std::to_string(app.attribution().evaluated()) + " evaluated)");
        app.attribution().cancel();
        wait_attr(app, h);
        bool tmp_left = false;
        for (const std::string &n : files_in(key.dir)) tmp_left = tmp_left || n.find(".tmp") != std::string::npos;
        AttrEntry e;
        const bool whole = !file_exists(key.path()) || read_attr_entry(key.path(), &e, &err);
        check(
            app.attribution().state() == Attribution::State::CANCELLED && whole && !tmp_left && !app.evaluator().busy(),
            "Attribution: a cancel leaves a partial file or a running job");
    }
    fake.stop();
    restore();
}

void check_attribution_play(const CheckFn &check, App &app, const ReevalHooks &h) {
    const ModelSettings saved = app.settings().models;
    const std::string root = h.tmp + "/attr_play_cache";
    app.set_attribution_cache_root(root);
    std::string err;
    const std::string model = h.tmp + "/attr_model.ts.pt";
    write_file_atomic(model, "attribution model", &err);
    FakeServer fake;
    FakeServer::Options o;
    o.versions = {{0, 3}};
    const std::string sock = h.tmp + "/attr_play.sock";
    if (!fake.start(sock, o)) {
        check(false, "Attribution Play: the stand-in server does not listen");
        return;
    }
    app.set_models(served(sock, model));
    const bool play_tool = app.tool_open(Mode::PLAY, Tool::ATTRIBUTION);
    app.set_mode(Mode::PLAY);
    const live::Frames frames = h.frames;
    if (live::start_sculpted(check, app, true,
            {"--hand-a", "Mountain,Lightning Bolt,Mountain", "--battlefield-a", "Mountain", "--battlefield-b",
                "Grizzly Bears"},
            "Attribution Play", 1) &&
        live::wait_decision(frames, app.play())) {
        const size_t index = static_cast<size_t>(app.play().pending_index());
        app.explain();
        check(wait_attr(app, h), "Attribution Play: Explain does not finish");
        const Attribution &a = app.attribution();
        check(a.state() == Attribution::State::DONE && app.attribution_current() && a.key().seat == 0 && a.base() &&
                  a.frame() && a.frame()->header.viewer == 0 && a.index() == index &&
                  a.key().dir.find(root + "/live-") == 0 && file_exists(a.key().path()),
            "Attribution Play: Explain does not attribute the human's live decision in its own frame: " + a.error());
        const AttrKey key = a.key();
        const size_t units = a.entry() ? a.entry()->units.size() : 0;
        fake.stop();
        // The game opened in Replay: the same decision from the live game's entry, with no server.
        if (live::open_log_in_replay(check, app, frames, "Attribution Play")) {
            const int at = app.replay().match()->find_decision(key.game, key.sideboarding, key.decision);
            check(at >= 0, "Attribution Play: the decision is not in the replay");
            if (at >= 0) {
                app.replay().set_view(VIEW_A);
                app.replay().go_to(static_cast<size_t>(at));
                app.set_tool_open(Mode::REPLAY, Tool::ATTRIBUTION, true);
                wait_attr(app, h);
                check(app.attribution().from_cache() && app.attribution().base() &&
                          app.attribution().entry()->units.size() == units,
                    "Attribution Play: the replay does not open the live game's attribution");
                std::printf(
                    "Attribution Play: live decision %zu explained (%zu units), found again in Replay at "
                    "decision %d\n",
                    index + 1, units, at + 1);
                app.set_tool_open(Mode::REPLAY, Tool::ATTRIBUTION, false);
                app.focus_tool(Tool::TABLE);
            }
        }
    }
    fake.stop();
    app.attribution().clear();
    // Play's tools as they were, the Table in front.
    app.set_mode(Mode::PLAY);
    app.set_tool_open(Mode::PLAY, Tool::ATTRIBUTION, play_tool);
    app.focus_tool(Tool::TABLE);
    h.frames(3);
    app.set_mode(Mode::REPLAY);
    app.set_models(saved);
    app.set_attribution_cache_root("");
    h.frames(2);
}

void check_attribution_real(
    const CheckFn &check, App &app, const ReevalHooks &h, const std::string &log, const std::string &model) {
    ServerConfig sc;
    sc.binary = parent_dir(h.engine) + "/rmserve";
    sc.socket = h.tmp + "/rmserve_attr.sock";
    sc.cwd = repo_root();
    sc.log_dir = h.tmp + "/rmserve_attr";
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
        check(false, "Attribution: rmserve does not start: " + server.message());
        return;
    }
    if (!app.replay().loaded() && !h.open(log)) {
        server.stop();
        return;
    }
    const ModelSettings saved = app.settings().models;
    ToolFlags was{};
    for (int t = 0; t < TOOL_COUNT; ++t)
        was[static_cast<size_t>(t)] = app.tool_open(Mode::REPLAY, static_cast<Tool>(t));
    app.set_models(served(sc.socket, model));
    app.set_attribution_cache_root(h.tmp + "/attr_real_cache");
    ReplaySession &s = app.replay();
    const Match &m = *s.match();
    app.set_tool_open(Mode::REPLAY, Tool::ATTRIBUTION, true);
    s.set_view(VIEW_OMNI);
    for (const int seat : {0, 1}) {
        const long i = decision_of(m, seat, 4);
        if (i < 0) continue;
        const std::string at = std::string("Attribution (rmserve) at decision ") + std::to_string(i + 1);
        s.go_to(static_cast<size_t>(i));
        const auto t0 = std::chrono::steady_clock::now();
        double coarse_s = -1;
        h.frames(1);
        while (app.attribution().running() || app.attribution().busy()) {
            h.frames(1);
            if (coarse_s < 0 && app.attribution().stage() >= 2)
                coarse_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > ATTR_TIMEOUT_S) break;
        }
        const double total_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        h.frames(2);
        const Attribution &a = app.attribution();
        check(a.state() == Attribution::State::DONE, at + ": not done: " + a.error());
        const featurize::PolicyEvalRecord *rec = record_of(m, static_cast<size_t>(i), 0);
        const featurize::PolicyEvalRecord *re = record_of(m, static_cast<size_t>(i), reevaluated_id(0));
        check(a.base() && rec && same_eval(*a.base(), *rec, 1e-5),
            at + ": the unchanged evaluation is not the episode's recorded one within 1e-5");
        check(!re || (a.base() && same_eval(*a.base(), *re, 1e-5)),
            at + ": the unchanged evaluation is not the re-evaluation's within 1e-5");
        size_t coarse = 0, second = 0;
        for (const featurize::Unit &u : a.entry()->units) (u.coarse ? coarse : second) += 1;
        if (coarse_s < 0) coarse_s = total_s;
        std::printf(
            "ATTRIBUTION-COST: decision %ld (%s, %zu tokens): coarse pass %zu units in %.2f s (%.1f ms/unit), "
            "second pass %zu units in %.2f s (%.1f ms/unit), %.2f s in all\n",
            i + 1, seat_name(seat), a.tensors() ? a.tensors()->n() : 0, coarse, coarse_s,
            coarse ? 1000 * coarse_s / static_cast<double>(coarse) : 0.0, second, total_s - coarse_s,
            second ? 1000 * (total_s - coarse_s) / static_cast<double>(second) : 0.0, total_s);
        if (seat != 0) continue;
        app.attribution_view().set_method(AttrMethod::SHAPLEY);
        const auto t1 = std::chrono::steady_clock::now();
        check(wait_attr(app, h), at + ": Shapley does not finish");
        for (const AttrTarget t : {AttrTarget::V, AttrTarget::LOGP, AttrTarget::MARGIN}) {
            const ShapleyResult r = app.attribution().shapley(t);
            check(r.ready && r.efficient(), at + ": Shapley is not efficient for " + attr_target_name(t) + " (" +
                                                std::to_string(r.sum) + " vs " + std::to_string(r.total) + ")");
            std::printf("Attribution (rmserve): Shapley %s over %zu units: sum %.6f, all changed %.6f\n",
                attr_target_name(t), app.attribution().entry()->players.size(), r.sum, r.total);
        }
        std::printf("ATTRIBUTION-COST: Shapley's %zu coalitions in %.2f s\n",
            app.attribution().entry()->coalitions.size(),
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count());
        app.attribution_view().set_method(AttrMethod::OCCLUSION);
    }
    app.attribution().clear();
    for (int t = 0; t < TOOL_COUNT; ++t)
        app.set_tool_open(Mode::REPLAY, static_cast<Tool>(t), was[static_cast<size_t>(t)]);
    app.focus_tool(Tool::TABLE);
    h.frames(3);
    app.set_models(saved);
    app.set_attribution_cache_root("");
    server.stop();
}

}  // namespace rmwork
