#include "selftest_sideboard.h"

#include <chrono>
#include <cstdio>
#include <thread>

#include "../../src/featurize/json.h"
#include "files.h"
#include "paths.h"
#include "sideboard_plan.h"

namespace {

using rmwork::CheckFn;
using Frames = std::function<void(int)>;

const double GAME_TIMEOUT_S = 300;

}  // namespace

static bool run_until(const Frames &frames, const std::function<bool()> &done, double timeout_s);
static void press_escape(const Frames &frames);
static std::string number_name(int card);
static int card_named(const rmwork::DeckCounts &d, const agents::CardNames &names, const std::string &name);
static std::map<std::string, int> json_counts(const featurize::Json *v);
static void check_steps(const CheckFn &check);
static void check_plan_rules(const CheckFn &check);
static void check_guides_file(const CheckFn &check);
static bool reach_sideboarding(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_live_guide(const CheckFn &check, rmwork::App &app);
static void check_live_mismatch(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_live_swap(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_replay_tool(const CheckFn &check, rmwork::App &app, const Frames &frames);

// Frames until `done`; false past the timeout.
static bool run_until(const Frames &frames, const std::function<bool()> &done, double timeout_s) {
    const auto start = std::chrono::steady_clock::now();
    while (!done()) {
        frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > timeout_s) return false;
    }
    frames(2);
    return true;
}

static void press_escape(const Frames &frames) {
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Escape, true);
    frames(1);
    ImGui::GetIO().AddKeyEvent(ImGuiKey_Escape, false);
    frames(1);
}

static std::string number_name(int card) {
    return "#" + std::to_string(card);
}

// The card of `d` (main deck or sideboard) named `name`, -1 none.
static int card_named(const rmwork::DeckCounts &d, const agents::CardNames &names, const std::string &name) {
    for (const std::map<int, int> *m : {&d.main, &d.side})
        for (const auto &kv : *m)
            if (names.name(kv.first) == name) return kv.first;
    return -1;
}

static std::map<std::string, int> json_counts(const featurize::Json *v) {
    std::map<std::string, int> out;
    if (v && v->is_object())
        for (const auto &m : v->members()) out[m.first] = static_cast<int>(m.second.as_int());
    return out;
}

// next_sideboard_step on made-up decisions.
static void check_steps(const CheckFn &check) {
    using namespace rmwork;
    SideboardFacts f;
    f.sideboarding = true;
    f.deck.main = {{1, 4}, {2, 4}};
    f.deck.side = {{3, 2}};
    f.menu = {{"SIDEBOARD_DONE", -1}, {"SIDEBOARD_IN", 3}};
    const std::map<int, int> target = {{1, 4}, {2, 3}, {3, 1}};
    SideboardStep st = next_sideboard_step(f, target, number_name);
    check(st.row == 1 && !st.done && st.error.empty(),
        "Sideboard steps: a balanced deck short of the target: not its IN row");
    f.drift = 1;
    f.deck.main = {{1, 4}, {2, 4}, {3, 1}};
    f.menu = {{"SIDEBOARD_OUT", 1}, {"SIDEBOARD_OUT", 2}};
    st = next_sideboard_step(f, target, number_name);
    check(st.row == 1 && st.what == "out: #2",
        "Sideboard steps: a cut owed: not the OUT row of the card over its target");
    f.drift = 0;
    f.deck.main = target;
    f.menu = {{"SIDEBOARD_DONE", -1}, {"SIDEBOARD_IN", 1}};
    st = next_sideboard_step(f, target, number_name);
    check(st.row == 0 && st.done, "Sideboard steps: the deck at its target: not Done");
    // The menu lacks what the plan needs.
    f.deck.main = {{1, 4}, {2, 4}};
    st = next_sideboard_step(f, {{1, 4}, {2, 3}, {4, 1}}, number_name);
    check(st.row < 0 && st.error.find("#4") != std::string::npos,
        "Sideboard steps: a menu without the plan's IN row is not refused by name: " + st.error);
    f.drift = 1;
    f.deck.main = {{1, 4}, {2, 4}, {3, 1}};
    f.menu = {{"SIDEBOARD_OUT", 1}};
    st = next_sideboard_step(f, target, number_name);
    check(st.row < 0 && st.error.find("#2") != std::string::npos,
        "Sideboard steps: a menu without the plan's OUT row is not refused: " + st.error);
    f.sideboarding = false;
    check(next_sideboard_step(f, target, number_name).row < 0, "Sideboard steps: a decision of a game is answered");
}

// The plan's rules over a phase with one swap made (3 in, 2 out).
static void check_plan_rules(const CheckFn &check) {
    using namespace rmwork;
    SideboardState s;
    s.start.main = {{1, 4}, {2, 4}};
    s.start.side = {{3, 2}, {4, 1}};
    s.now.main = {{1, 4}, {2, 3}, {3, 1}};
    s.now.side = {{2, 1}, {3, 1}, {4, 1}};
    s.swaps = 1;
    SideboardPlan p;
    std::string why;
    check(!p.click_main(s, 3, &why), "Sideboard plan: a card brought in this phase can be cut");
    check(!p.click_side(s, 2, &why), "Sideboard plan: a card cut this phase can come back");
    check(!p.click_main(s, 1, &why), "Sideboard plan: a cut with nothing to balance is planned");
    check(p.click_side(s, 4, &why) && !p.balanced(s), "Sideboard plan: a sideboard card does not come in");
    check(!p.click_side(s, 4, &why), "Sideboard plan: a card with no copy left comes in");
    check(p.click_main(s, 1, &why) && p.balanced(s), "Sideboard plan: the balancing cut is refused: " + why);
    const DeckCounts r = p.result(s);
    check(r.main_total() == 8 && r.side_total() == 3 && r.main_of(4) == 1 && r.main_of(1) == 3 && p.swaps_after(s) == 2,
        "Sideboard plan: the result's counts");
    check(p.undo() && !p.balanced(s) && p.outs() == 0, "Sideboard plan: Undo does not take back the cut");
    check(p.click_main(s, 4, &why) && p.ins() == 0, "Sideboard plan: a click on a planned card does not take it back");
    s.swaps = 14;
    p.clear();
    check(p.click_side(s, 4, &why) && !p.click_side(s, 3, &why),
        "Sideboard plan: the 15-swap cap is not kept (" + why + ")");
    p.clear();
    s.swaps = 1;
    check(p.set_target(s, {{1, 3}, {2, 3}, {3, 1}, {4, 1}}, &why) && p.ins() == 1 && p.outs() == 1,
        "Sideboard plan: a reachable target is refused: " + why);
    check(!p.set_target(s, {{1, 4}, {2, 4}}, &why), "Sideboard plan: a target cutting a card brought in is taken");
}

// The guides file against its JSON.
static void check_guides_file(const CheckFn &check) {
    using namespace rmwork;
    const std::string path = repo_root() + "/python/sideboard_guides.json";
    SideboardGuides g;
    std::string err, text;
    check(g.load(path, &err), "Sideboard guides: " + err);
    featurize::Json doc;
    if (!read_file(path, &text) || !featurize::Json::parse(text, &doc, &err)) return;
    const featurize::Json *row = doc.get("guides") ? doc.get("guides")->get("burn") : nullptr;
    row = row ? row->get("bug") : nullptr;
    const GuideRow *r = g.row("burn", "bug");
    check(r && row, "Sideboard guides: no burn vs bug row");
    if (r && row) {
        std::map<std::string, int> in, out;
        for (const auto &c : r->in) in[c.first] = c.second;
        for (const auto &c : r->out) out[c.first] = c.second;
        check(in == json_counts(row->get("in")) && out == json_counts(row->get("out")) &&
                  r->draw == row->get("draw")->as_string(),
            "Sideboard guides: burn vs bug is not the file's row");
    }
    check(g.draw_land("burn") == doc.get("draw_land")->get("burn")->as_string() && !g.draw_land("burn").empty(),
        "Sideboard guides: burn's draw land");
    check(g.holdout("burn", "uw_control") && !g.holdout("burn", "bug"), "Sideboard guides: the held-out pairs");
    check(league_stem("league/burn") == "burn" && league_stem("heldout/storm").empty() && league_stem("delver").empty(),
        "Sideboard guides: a deck's league stem");
    check(card_key("Dead/Gone") == card_key("Dead // Gone") && card_key("L\xC3\xB3rien Revealed") == "lorienrevealed",
        "Sideboard guides: card names across spellings");
    SideboardGuides none;
    check(!none.load(repo_root() + "/python/no_such_guides.json", &err) && !none.loaded(),
        "Sideboard guides: a missing file reads");
}

// A bo3 as seat A to the human's first sideboarding decision.
static bool reach_sideboarding(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    NewGameSettings g;
    g.opponent = "explore";
    g.deck = "league/burn";
    g.opponent_deck = "league/bug";
    g.seat = 0;
    g.bo3 = true;
    g.seed = 3;
    check(app.start_game(g), "Sideboard: the bo3 does not start");
    PlaySession &s = app.play();
    app.auto_play(-1, "sideboard");
    const auto at_sideboarding = [&s] {
        return !s.running() ||
               (s.deciding() && s.match()->decisions[static_cast<size_t>(s.pending_index())].sideboarding);
    };
    run_until(frames, at_sideboarding, GAME_TIMEOUT_S);
    frames(3);
    const bool reached = s.running() && s.deciding();
    check(reached, "Sideboard: the bo3 ended before the human's sideboarding");
    if (!reached) return false;
    SideboardView &v = app.tools().sideboard();
    v.set_choosing(app.choosing());
    v.update(app.tools().data(), app.session());
    const SideboardModel &md = v.model();
    check(app.tool_open(Mode::PLAY, Tool::SIDEBOARD), "Sideboard: the tool is not opened at the sideboarding phase");
    check(md.ok && md.phase && md.seat == 0 && md.seat_decides && v.interactive() && md.facts.has_deck &&
              md.state.now.main_total() == 60 && md.state.now.side_total() == 15 && md.facts.drift == 0,
        "Sideboard: the tool's model at the phase's first decision: " + md.error);
    check(md.lists.name[0] == "league/burn" && md.lists.name[1] == "league/bug" && md.opponent == "explore",
        "Sideboard: the decks' and opponent's names");
    return true;
}

// The guide's plan for burn against bug, on the play or the draw as the frame says, and Use This Plan.
static void check_live_guide(const CheckFn &check, rmwork::App &app) {
    using namespace rmwork;
    SideboardView &v = app.tools().sideboard();
    const SideboardModel &md = v.model();
    const GuidePlan &gp = md.guide;
    check(gp.found && gp.own == "burn" && gp.opp == "bug" && gp.unresolved.empty() && !gp.holdout &&
              gp.on_play == (md.facts.plays_first == 0),
        "Sideboard: the guide's plan for burn vs bug is not found: " + gp.why + gp.unresolved);
    const SideboardGuides &g = v.guides();
    const GuideRow *row = g.row("burn", "bug");
    if (!gp.found || !row) return;
    size_t want_in = row->in.size() + (gp.on_play ? 0 : 1), want_out = row->out.size() + (gp.on_play ? 0 : 1);
    check(gp.in.size() == want_in && gp.out.size() == want_out && !gp.other.empty(),
        "Sideboard: the guide's lines are not the row's (with the draw swap on the draw)");
    int total = 0;
    for (const auto &kv : gp.target) total += kv.second;
    check(total == 60, "Sideboard: the guide's target deck has " + std::to_string(total) + " cards");
    std::string why;
    check(v.use_guide(&why) && main_matches(v.plan().result(md.state), gp.target) && v.plan().balanced(md.state),
        "Sideboard: Use This Plan does not plan the guide's deck: " + why);
    v.reset();
    check(v.plan().empty(), "Sideboard: Reset leaves a plan");
}

// A target the menu cannot reach (a main-deck-only card brought in): stopped, nothing sent.
static void check_live_mismatch(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    SideboardView &v = app.tools().sideboard();
    PlaySession &s = app.play();
    const SideboardModel &md = v.model();
    const agents::CardNames &names = s.match()->stream(0).cards();
    const int mountain = card_named(md.state.now, names, "Mountain");
    const int bolt = card_named(md.state.now, names, "Lightning Bolt");
    check(mountain >= 0 && bolt >= 0 && md.state.now.side_of(mountain) == 0, "Sideboard: burn's Mountain and Bolt");
    std::map<int, int> target = md.state.now.main;
    ++target[mountain];
    --target[bolt];
    const long at = s.pending_index();
    std::string why;
    check(v.send_target(target, &why), "Sideboard: an unreachable target is not taken to send: " + why);
    run_until(frames, [&v] { return !v.sending(); }, 30);
    check(!v.error().empty() && v.error().find("menu") != std::string::npos &&
              v.error().find("Mountain") != std::string::npos && v.replies_sent() == 0 && s.deciding() &&
              s.pending_index() == at,
        "Sideboard: a menu without the planned row did not stop the send at once: " + v.error());
    v.dismiss();
}

// A planned swap sent through the session; the next game's frame shows it.
static void check_live_swap(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    SideboardView &v = app.tools().sideboard();
    PlaySession &s = app.play();
    const SideboardModel &md = v.model();
    const agents::CardNames &names = s.match()->stream(0).cards();
    const int smash = card_named(md.state.now, names, "Smash to Smithereens");
    const int stage = card_named(md.state.now, names, "Light Up the Stage");
    check(smash >= 0 && stage >= 0, "Sideboard: burn's Smash to Smithereens and Light Up the Stage");
    if (smash < 0 || stage < 0) return;
    const int smash_main = md.state.now.main_of(smash), stage_main = md.state.now.main_of(stage);
    std::string why;
    check(v.click_side(smash, &why) && v.click_main(stage, &why) && v.plan().balanced(md.state),
        "Sideboard: the planned swap is refused: " + why);
    const DeckCounts planned = v.plan().result(md.state);
    const int phase_game = md.facts.game;
    const size_t phase_first = static_cast<size_t>(s.pending_index());
    frames(2);  // the tool drawn with its plan
    check(v.confirm(&why), "Sideboard: Confirm is refused: " + why);
    run_until(frames, [&v] { return !v.sending(); }, GAME_TIMEOUT_S);
    run_until(frames, [&s] { return !s.running() || s.deciding(); }, GAME_TIMEOUT_S);
    check(v.error().empty() && !v.result().empty() && v.replies_sent() == 3,
        "Sideboard: the plan was not sent as IN, OUT, Done (" + std::to_string(v.replies_sent()) +
            " replies): " + v.error());
    if (!s.deciding()) return;
    const Match *m = s.match();
    std::string err;
    const std::shared_ptr<const rmstate::DecodedFrame> f = m->frame(static_cast<size_t>(s.pending_index()), 0, &err);
    check(f != nullptr, "Sideboard: the next game's frame: " + err);
    if (!f) return;
    const SideboardFacts next = read_sideboard_facts(m->stream(0), *f, 0);
    check(!next.sideboarding && next.game == phase_game && next.deck.main_of(smash) == smash_main + 1 &&
              next.deck.main_of(stage) == stage_main - 1 && next.deck.main_total() == 60 &&
              next.deck.side_total() == 15 && main_matches(next.deck, planned.main),
        "Sideboard: game " + std::to_string(phase_game + 1) + "'s frame does not show the planned deck");
    // The phase's replies: IN, OUT, Done, each the row of its category.
    std::vector<std::string> cats;
    for (size_t i = phase_first; i < m->decisions.size(); ++i) {
        const DecisionInfo &d = m->decisions[i];
        if (!d.sideboarding || d.game != phase_game) break;
        const std::shared_ptr<const rmstate::DecodedFrame> df = m->frame(i, 0, &err);
        if (!df || d.reply < 0) continue;
        const SideboardFacts at = read_sideboard_facts(m->stream(0), *df, 0);
        if (static_cast<size_t>(d.reply) < at.menu.size())
            cats.push_back(at.menu[static_cast<size_t>(d.reply)].category);
    }
    check(cats == std::vector<std::string>({"SIDEBOARD_IN", "SIDEBOARD_OUT", "SIDEBOARD_DONE"}),
        "Sideboard: the phase's replies are not IN, OUT, Done");
    std::printf("rmwork: Sideboard: sent %d replies, game %d's deck +1 Smash to Smithereens, -1 Light Up the Stage\n",
        v.replies_sent(), phase_game + 1);
}

// The match in Replay: the read-only tool in seat A's view and seat B's.
static void check_replay_tool(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    check(app.open_replay(app.play().log_path()), "Sideboard: the match does not open in Replay");
    run_until(frames, [&app] { return !app.loading(); }, GAME_TIMEOUT_S);
    if (!app.replay().loaded()) return;
    const Match &rm = *app.replay().match();
    long at = -1;
    for (size_t i = 0; i < rm.decisions.size() && at < 0; ++i)
        if (rm.decisions[i].sideboarding && rm.decisions[i].decider == 0) at = static_cast<long>(i);
    check(at >= 0, "Sideboard: Replay has no sideboarding decision of seat A");
    if (at < 0) return;
    app.set_tool_open(Mode::REPLAY, Tool::SIDEBOARD, true);
    app.focus_tool(Tool::SIDEBOARD);
    app.replay().set_view(VIEW_A);
    app.replay().go_to(static_cast<size_t>(at));
    frames(3);
    SideboardView &v = app.tools().sideboard();
    v.set_choosing(false);
    v.update(app.tools().data(), app.replay());
    const SideboardModel &a = v.model();
    const agents::CardNames &names = rm.stream(VIEW_A).cards();
    const int smash = card_named(a.state.now, names, "Smash to Smithereens");
    const int stage = card_named(a.state.now, names, "Light Up the Stage");
    check(a.ok && a.phase && a.seat == 0 && a.seat_decides && !v.interactive() && a.have_end &&
              a.end.main_of(smash) == a.state.start.main_of(smash) + 1 &&
              a.end.main_of(stage) == a.state.start.main_of(stage) - 1 && a.end.main_total() == 60,
        "Sideboard: Replay's tool in seat A's view does not show the phase's swap");
    app.replay().set_view(VIEW_B);
    frames(3);
    v.update(app.tools().data(), app.replay());
    const SideboardModel &b = v.model();
    std::string err;
    const std::shared_ptr<const rmstate::DecodedFrame> fb = rm.frame(static_cast<size_t>(at), VIEW_B, &err);
    check(b.ok && b.seat == 1 && !b.seat_decides && !b.note.empty() && b.facts.has_deck &&
              b.state.now.main_total() == 60 && fb && !read_sideboard_facts(rm.stream(VIEW_B), *fb, 0).has_deck,
        "Sideboard: Replay's tool in seat B's view at seat A's sideboarding shows other than seat B's own 75");
    app.set_mode(Mode::PLAY);
    frames(2);
}

namespace rmwork {

void check_sideboard(const CheckFn &check, App &app, const std::function<void(int)> &frames, const PlayTestEnv &) {
    check_steps(check);
    check_plan_rules(check);
    check_guides_file(check);
    if (!reach_sideboarding(check, app, frames)) {
        app.play().stop();
        return;
    }
    check_live_guide(check, app);
    check_live_mismatch(check, app, frames);
    check_live_swap(check, app, frames);
    app.auto_play(-1, "end");
    run_until(frames, [&app] { return !app.play().running(); }, GAME_TIMEOUT_S);
    check(app.play().state() == PlaySession::State::FINISHED, "Sideboard: the match did not finish");
    press_escape(frames);
    check_replay_tool(check, app, frames);
}

}  // namespace rmwork
