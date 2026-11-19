#include "selftest_decisions.h"

#include <chrono>
#include <cstdio>
#include <memory>
#include <set>
#include <thread>

#include "../../src/agents/decision_view.h"
#include "../../src/state_format/decode.h"

namespace {

using rmwork::CheckFn;
using Frames = std::function<void(int)>;
using RowTest = std::function<bool(const std::string &category, const std::string &text)>;

const double DECISION_TIMEOUT_S = 120;
// Decisions drive_to answers with row 0 before giving up.
const int DRIVE_LIMIT = 40;

// The human's pending decision: its frame and the view over it.
struct Pending {
        std::shared_ptr<const rmstate::DecodedFrame> frame;
        std::unique_ptr<agents::DecisionView> view;
        size_t index = 0;
        bool ok() const { return view != nullptr; }
};

}  // namespace

static bool wait_decision(const Frames &frames, rmwork::PlaySession &s);
static Pending pending(rmwork::PlaySession &s);
static int find_row(const Pending &p, const RowTest &test);
static int drive_to(rmwork::App &app, const Frames &frames, const RowTest &test);
static int row_naming(const Pending &p, uint64_t handle, int seat, const std::string &containing);
static uint64_t object_named(const rmwork::App &app, const std::string &name, bool untapped);
static uint64_t object_of(const rmwork::App &app, const std::string &name, int controller);
static ImVec2 screen_point(rmwork::Mode m, const rmwork::TRect &r);
static bool card_rect(const rmwork::App &app, uint64_t handle, rmwork::TRect *out);
static int click_reply(rmwork::App &app, const Frames &frames, ImVec2 p, size_t index);
static void check_picks_shown(const CheckFn &check, const rmwork::App &app, const std::string &at);
static bool start_sculpted(const CheckFn &check, rmwork::App &app, bool auto_mana,
    const std::vector<std::string> &setup, const std::string &at);
static bool click_object(
    const CheckFn &check, rmwork::App &app, const Frames &frames, uint64_t handle, int want, const std::string &at);
static bool click_button(
    const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &text, const std::string &at);
static bool press_button(
    const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &text, const std::string &at);
static bool card_point(const rmwork::App &app, uint64_t handle, ImVec2 *out);
static void click_card(rmwork::App &app, const Frames &frames, uint64_t handle);
static bool wait_sent(const Frames &frames, rmwork::App &app);
static bool combat_has(const Pending &p, uint64_t attacker, uint64_t target, int target_seat, uint64_t blocker);
static size_t combat_attackers(const Pending &p);
static Pending frame_after(const rmwork::App &app, size_t index, const std::function<bool(const Pending &)> &test);
static void check_attackers(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_damage(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_blockers(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_combat_replay(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_targeting(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_targeting_replay(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_mana(const CheckFn &check, rmwork::App &app, const Frames &frames);

// Frames until the human decides or the game is over; whether the human decides.
static bool wait_decision(const Frames &frames, rmwork::PlaySession &s) {
    const auto start = std::chrono::steady_clock::now();
    while (!s.deciding() && s.running()) {
        frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > DECISION_TIMEOUT_S)
            return false;
    }
    frames(2);
    return s.deciding();
}

static Pending pending(rmwork::PlaySession &s) {
    Pending p;
    if (!s.deciding() || !s.match()) return p;
    p.index = static_cast<size_t>(s.pending_index());
    std::string err;
    p.frame = s.match()->frame(p.index, s.config().seat, &err);
    if (p.frame) p.view.reset(new agents::DecisionView(s.match()->stream(s.config().seat), *p.frame, s.config().seat));
    return p;
}

static int find_row(const Pending &p, const RowTest &test) {
    if (!p.ok()) return -1;
    for (size_t i = 0; i < p.view->n(); ++i)
        if (test(p.view->category(i), p.view->description(i))) return static_cast<int>(i);
    return -1;
}

// Answers the human's decisions with row 0 until one has a row `test` holds for; that row (the decision
// left pending), -1 when none came.
static int drive_to(rmwork::App &app, const Frames &frames, const RowTest &test) {
    for (int k = 0; k < DRIVE_LIMIT; ++k) {
        if (!wait_decision(frames, app.play())) return -1;
        const int row = find_row(pending(app.play()), test);
        if (row >= 0) return row;
        app.play_reply(0);
    }
    return -1;
}

// The one row naming object `handle` (or player `seat`) whose description contains `containing`; -1
// when none or several do.
static int row_naming(const Pending &p, uint64_t handle, int seat, const std::string &containing) {
    if (!p.ok()) return -1;
    const agents::Table menu = p.view->table("menu");
    const int c_src = menu.col("source"), c_sp = menu.col("source_player");
    int found = -1;
    for (size_t i = 0; i < p.view->n(); ++i) {
        const uint64_t sp = menu.u(i, c_sp);
        const bool names = handle ? menu.u(i, c_src) == handle : (sp == 1 ? 0 : sp == 2 ? 1 : -1) == seat;
        if (!names || p.view->description(i).find(containing) == std::string::npos) continue;
        if (found >= 0) return -1;
        found = static_cast<int>(i);
    }
    return found;
}

// The first object on the Table named `name` (untapped only when `untapped`), 0 when none.
static uint64_t object_named(const rmwork::App &app, const std::string &name, bool untapped) {
    const rmwork::TableModel *m = const_cast<rmwork::App &>(app).table().model();
    if (!m) return 0;
    for (const rmwork::TableObject &o : m->objects)
        if (o.name == name && !o.in_hand && (!untapped || !o.tapped)) return o.handle;
    return 0;
}

// The centre of `r` (the Table's panel coordinates) on screen.
static ImVec2 screen_point(rmwork::Mode m, const rmwork::TRect &r) {
    const ImVec2 o = rmwork::table_window_origin(m);
    return ImVec2(o.x + (r.x0 + r.x1) / 2, o.y + (r.y0 + r.y1) / 2);
}

// The first battlefield object named `name` controlled by seat `controller`, 0 when none.
static uint64_t object_of(const rmwork::App &app, const std::string &name, int controller) {
    const rmwork::TableModel *m = const_cast<rmwork::App &>(app).table().model();
    if (!m) return 0;
    for (const rmwork::TableObject &o : m->objects)
        if (o.name == name && !o.in_hand && o.controller == controller) return o.handle;
    return 0;
}

// The rectangle object `handle`'s card is drawn in (its pile's front card).
static bool card_rect(const rmwork::App &app, uint64_t handle, rmwork::TRect *out) {
    rmwork::TableView &t = const_cast<rmwork::App &>(app).table();
    const rmwork::TableModel *m = t.model();
    if (!m) return false;
    const int obj = m->find(handle);
    for (const rmwork::PlacedCard &c : t.layout().cards)
        if (c.object == obj && c.front) {
            *out = c.rect;
            return true;
        }
    return false;
}

// A click at `p`; the reply then recorded at decision `index` (REPLY_PENDING when none was sent).
static int click_reply(rmwork::App &app, const Frames &frames, ImVec2 p, size_t index) {
    rmwork::mouse_click(p, frames);
    const rmwork::Match *m = app.play().match();
    return m && index < m->decisions.size() ? m->decisions[index].reply : rmwork::REPLY_PENDING;
}

// Every pick of the Table's decision is an object the frame shows on the Table (the bottom seat's
// battlefield and hand, the stack) or a player.
static void check_picks_shown(const CheckFn &check, const rmwork::App &app, const std::string &at) {
    rmwork::TableView &t = const_cast<rmwork::App &>(app).table();
    const rmwork::TableModel *m = t.model();
    if (!m) return;
    for (const rmwork::DecisionPick &p : t.decision().picks) {
        if (p.handle == 0) {
            check(p.seat == 0 || p.seat == 1, at + ": a pick names neither an object nor a player");
            continue;
        }
        bool shown = false;
        const int obj = m->find(p.handle);
        if (obj >= 0)
            shown = !m->objects[static_cast<size_t>(obj)].in_hand ||
                    m->objects[static_cast<size_t>(obj)].owner == m->bottom;
        for (const rmwork::TableStackItem &s : m->stack) shown = shown || s.handle == p.handle;
        check(shown, at + ": a pick (handle " + std::to_string(p.handle) + ") is no object the Table shows");
    }
}

static bool start_sculpted(const CheckFn &check, rmwork::App &app, bool auto_mana,
    const std::vector<std::string> &setup, const std::string &at) {
    app.set_tool_open(rmwork::Mode::PLAY, rmwork::Tool::TABLE, true);
    app.focus_tool(rmwork::Tool::TABLE);
    rmwork::NewGameSettings g;
    g.opponent = "explore";
    g.deck = "league/burn";
    g.opponent_deck = "league/bug";
    g.seat = 0;
    g.seed = 5;
    g.auto_mana = auto_mana;
    const bool ok = app.start_game(g, setup);
    check(ok, at + ": the sculpted game does not start");
    return ok;
}

// A click on object `handle`'s card answers with row `want`.
static bool click_object(
    const CheckFn &check, rmwork::App &app, const Frames &frames, uint64_t handle, int want, const std::string &at) {
    rmwork::TRect r;
    const bool placed = handle != 0 && want >= 0 && card_rect(app, handle, &r);
    check(placed, at + ": the object to click is not on the Table, or no row names it");
    if (!placed) return false;
    const size_t index = static_cast<size_t>(app.play().pending_index());
    const int got = click_reply(app, frames, screen_point(rmwork::Mode::PLAY, r), index);
    check(got == want, at + ": the click replied " + std::to_string(got) + ", not row " + std::to_string(want));
    return got == want;
}

// A click on the prompt line's button whose label contains `text` answers with its row.
static bool click_button(
    const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &text, const std::string &at) {
    int want = -1;
    for (const rmwork::PromptButton &b : app.table().prompt_buttons())
        if (b.label.find(text) != std::string::npos) {
            want = b.row;
            break;
        }
    const size_t index = static_cast<size_t>(app.play().pending_index());
    if (!press_button(check, app, frames, text, at)) return false;
    const rmwork::Match *m = app.play().match();
    const int got = m && index < m->decisions.size() ? m->decisions[index].reply : rmwork::REPLY_PENDING;
    check(got == want, at + ": the button replied " + std::to_string(got) + ", not row " + std::to_string(want));
    return got == want;
}

// A click on the prompt line's button whose label contains `text`; false when there is none.
static bool press_button(
    const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &text, const std::string &at) {
    const std::vector<rmwork::PromptButton> &buttons = app.table().prompt_buttons();
    const std::vector<rmwork::TRect> &rects = app.table().layout().prompt_buttons;
    for (size_t i = 0; i < buttons.size() && i < rects.size(); ++i)
        if (buttons[i].label.find(text) != std::string::npos) {
            rmwork::mouse_click(screen_point(rmwork::Mode::PLAY, rects[i]), frames);
            return true;
        }
    std::string have;
    for (const rmwork::PromptButton &b : buttons) have += " [" + b.label + "]";
    check(false, at + ": the prompt line has no button \"" + text + "\" (it has" + have + ")");
    return false;
}

// Where object `handle`'s card is on screen (Play's Table).
static bool card_point(const rmwork::App &app, uint64_t handle, ImVec2 *out) {
    rmwork::TRect r;
    if (!handle || !card_rect(app, handle, &r)) return false;
    *out = screen_point(rmwork::Mode::PLAY, r);
    return true;
}

static void click_card(rmwork::App &app, const Frames &frames, uint64_t handle) {
    ImVec2 p;
    if (card_point(app, handle, &p)) rmwork::mouse_click(p, frames);
}

// Frames until the Table's plan is sent (or stopped), then until the human decides; whether it was.
static bool wait_sent(const Frames &frames, rmwork::App &app) {
    const auto start = std::chrono::steady_clock::now();
    while (app.table().plan_sending() && app.play().running()) {
        frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > DECISION_TIMEOUT_S)
            return false;
    }
    return !app.table().plan_sending() && wait_decision(frames, app.play());
}

// The pending frame's combat table has `attacker` attacking the planeswalker `target` (or, target 0,
// the player `target_seat`; -1 either), blocked by `blocker` (0: any or none).
static bool combat_has(const Pending &p, uint64_t attacker, uint64_t target, int target_seat, uint64_t blocker) {
    if (!p.ok()) return false;
    const agents::Table c = p.view->table("combat");
    for (size_t r = 0; r < c.rows(); ++r) {
        if (c.u(r, c.col("attacker")) != attacker) continue;
        if (blocker && c.u(r, c.col("blocker")) != blocker) continue;
        if (target && c.u(r, c.col("target")) != target) continue;
        const uint64_t tp = c.u(r, c.col("target_player"));
        if (!target && target_seat >= 0 && (tp == 1 ? 0 : tp == 2 ? 1 : -1) != target_seat) continue;
        return true;
    }
    return false;
}

// The attackers in the frame's combat table (a row per blocker of each).
static size_t combat_attackers(const Pending &p) {
    if (!p.ok()) return 0;
    const agents::Table c = p.view->table("combat");
    std::set<uint64_t> out;
    for (size_t r = 0; r < c.rows(); ++r) out.insert(c.u(r, c.col("attacker")));
    return out.size();
}

// The first of the human's decisions after `index` (the stops answer some unasked; their frames are
// recorded too) whose frame `test` holds for; not ok() when none.
static Pending frame_after(const rmwork::App &app, size_t index, const std::function<bool(const Pending &)> &test) {
    const rmwork::PlaySession &s = const_cast<rmwork::App &>(app).play();
    const rmwork::Match *m = s.match();
    const int seat = s.config().seat;
    for (size_t i = index + 1; m && i < m->decisions.size(); ++i) {
        if (m->decisions[i].decider != seat) continue;
        Pending p;
        std::string err;
        p.frame = m->frame(i, seat, &err);
        if (!p.frame) continue;
        p.index = i;
        p.view.reset(new agents::DecisionView(m->stream(seat), *p.frame, seat));
        if (test(p)) return p;
    }
    return Pending();
}

// Lightning Bolt at a creature, Prismari Charm at the opponent and Done; then the game in Replay.
static void check_targeting(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table targeting";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Lightning Bolt,Prismari Charm", "--library-a", "Mountain", "--battlefield-a",
                "Volcanic Island,Volcanic Island,Mountain,Mountain", "--hand-b", "Island", "--library-b", "Island",
                "--battlefield-b", "Grizzly Bears,Delver of Secrets"},
            at))
        return;
    const int cast = drive_to(app, frames, [](const std::string &c, const std::string &t) {
        return c == "CAST_SPELL" && t.find("Lightning Bolt") != std::string::npos;
    });
    check(cast >= 0, at + ": no decision offers to cast Lightning Bolt");
    if (cast < 0 || !app.play_reply(cast) || !wait_decision(frames, s)) return;
    Pending p = pending(s);
    const TableDecision &d = app.table().decision();
    const uint64_t bears = object_named(app, "Grizzly Bears", false);
    bool source_on_stack = false;
    for (const TableStackItem &it : app.table().model()->stack)
        source_on_stack = source_on_stack || it.handle == d.source;
    check(d.kind == DecisionKind::TARGET && source_on_stack && d.pick_of_object(bears) && d.pick_of_seat(0) &&
              d.pick_of_seat(1) && d.buttons.empty(),
        at + ": Lightning Bolt's target decision is not lit on the Table (its spell, the bears, both players)");
    check_picks_shown(check, app, at + " (Bolt)");
    if (!click_object(check, app, frames, bears, row_naming(p, bears, -1, ""), at + ": Bolt at the bears")) return;

    const int charm = drive_to(app, frames, [](const std::string &c, const std::string &t) {
        return c == "CAST_SPELL" && t.find("Prismari Charm") != std::string::npos;
    });
    check(charm >= 0, at + ": no decision offers to cast Prismari Charm");
    if (charm < 0 || !app.play_reply(charm)) return;
    const int mode = drive_to(app, frames, [](const std::string &c, const std::string &t) {
        return (c == "CHOOSE_MODE" && t.find("damage") != std::string::npos) || c == "SELECT_TARGET";
    });
    if (mode >= 0 && pending(s).view->category(static_cast<size_t>(mode)) == "CHOOSE_MODE") {
        app.play_reply(mode);
        wait_decision(frames, s);
    }
    p = pending(s);
    check(p.ok() && app.table().decision().kind == DecisionKind::TARGET && app.table().decision().pick_of_seat(1),
        at + ": Prismari Charm's first target decision does not light the opponent");
    if (!p.ok() || app.table().decision().kind != DecisionKind::TARGET) return;
    check_picks_shown(check, app, at + " (Charm, first target)");
    const int want = row_naming(p, 0, 1, "");
    const size_t index = p.index;
    const int got = click_reply(app, frames, screen_point(Mode::PLAY, app.table().chip_rect(1)), index);
    check(want >= 0 && got == want,
        at + ": a click on the opponent's chip replied " + std::to_string(got) + ", not row " + std::to_string(want));
    if (got != want || !wait_decision(frames, s)) return;
    const TableDecision &d2 = app.table().decision();
    check(d2.kind == DecisionKind::TARGET && d2.chosen.size() == 1 && d2.chosen[0].seat == 1 &&
              d2.chosen[0].order == 1 && !d2.pick_of_seat(1),
        at + ": the second target decision does not mark the opponent chosen first");
    check_picks_shown(check, app, at + " (Charm, second target)");
    if (!click_button(check, app, frames, "Done selecting targets", at + ": Done")) return;
    std::printf("rmwork: %s: a creature, a player chip and Done answered on the Table\n", at.c_str());
    check_targeting_replay(check, app, frames);
}

// The targeting game in Replay: the highlights in seat A's view, none in B's; a click selects only.
static void check_targeting_replay(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table targeting in Replay";
    const std::string log = app.play().log_path();
    app.play().stop();
    check(app.open_replay(log), at + ": the decision log does not open");
    const auto start = std::chrono::steady_clock::now();
    while (app.loading() &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < DECISION_TIMEOUT_S)
        frames(1);
    frames(2);
    if (!app.replay().loaded()) {
        check(false, at + ": the log is not loaded");
        return;
    }
    // The first decision of seat A choosing a target.
    long target = -1;
    const Match &m = *app.replay().match();
    for (size_t i = 0; i < m.decisions.size() && target < 0; ++i) {
        if (m.decisions[i].decider != 0) continue;
        std::string err;
        const auto f = m.frame(i, 0, &err);
        if (!f) continue;
        const agents::DecisionView v(m.stream(0), *f, 0);
        if (v.n() && v.category(0) == "SELECT_TARGET") target = static_cast<long>(i);
    }
    check(target >= 0, at + ": the replay has no target decision of seat A");
    if (target < 0) return;
    app.set_tool_open(Mode::REPLAY, Tool::TABLE, true);
    app.focus_tool(Tool::TABLE);
    app.replay().set_view(0);
    app.replay().go_to(static_cast<size_t>(target));
    frames(3);
    const uint64_t bears = object_named(app, "Grizzly Bears", false);
    check(app.table().decision().kind == DecisionKind::TARGET && app.table().decision().pick_of_object(bears),
        at + ": seat A's view does not light the target decision");
    TRect r;
    if (card_rect(app, bears, &r)) {
        mouse_click(screen_point(Mode::REPLAY, r), frames);
        check(app.links().selected().kind == LinkItem::Kind::OBJECT && app.links().selected().handle == bears &&
                  !app.table().pick_popup_open() && app.table().take_answer() < 0 &&
                  app.replay().cursor() == static_cast<size_t>(target),
            at + ": a click on a target does more than select it");
    }
    app.replay().set_view(1);
    frames(3);
    check(app.table().decision().kind == DecisionKind::NONE && app.table().decision().picks.empty(),
        at + ": seat B's view of seat A's target decision lights its targets");
    mouse_to(ImVec2(-1000, -1000), frames);
    app.set_mode(Mode::PLAY);
    frames(2);
}

// --no-auto-mana: a dual land's popup, Phyrexian pips by button, lands clicked to pay and for mana.
static void check_mana(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table mana payment";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, false,
            {"--hand-a", "Lightning Bolt,Dismember", "--library-a", "Mountain", "--battlefield-a",
                "Volcanic Island,Swamp,Swamp,Mountain,Mountain", "--hand-b", "Island", "--library-b", "Island",
                "--battlefield-b", "Grizzly Bears,Delver of Secrets"},
            at))
        return;
    const int cast = drive_to(app, frames, [](const std::string &c, const std::string &t) {
        return c == "CAST_SPELL" && t.find("Lightning Bolt") != std::string::npos;
    });
    check(cast >= 0, at + ": no decision offers to cast Lightning Bolt");
    if (cast < 0 || !app.play_reply(cast) || !wait_decision(frames, s)) return;
    const uint64_t bears = object_named(app, "Grizzly Bears", false);
    const int target = row_naming(pending(s), bears, -1, "");
    if (target < 0 || !app.play_reply(target) || !wait_decision(frames, s)) {
        check(false, at + ": Lightning Bolt's target decision did not come");
        return;
    }
    // Paying {R}: the volcanic island's two rows in a popup.
    Pending p = pending(s);
    const TableDecision &d = app.table().decision();
    const uint64_t volcanic = object_named(app, "Volcanic Island", true);
    const DecisionPick *vp = d.pick_of_object(volcanic);
    bool cancel = false;
    for (int b : d.buttons) cancel = cancel || d.row_text[static_cast<size_t>(b)] == "Cancel casting";
    check(d.kind == DecisionKind::PAY && d.pips == std::vector<std::string>{"R"} && vp && vp->rows.size() == 2 &&
              cancel && app.table().layout().prompt.w() > 0,
        at + ": paying {R} does not show the pip, the dual land's two rows and Cancel casting");
    check_picks_shown(check, app, at + " ({R})");
    TRect r;
    if (!vp || !card_rect(app, volcanic, &r)) return;
    const int want_r = row_naming(p, volcanic, -1, "(R)");
    const int none = click_reply(app, frames, screen_point(Mode::PLAY, r), p.index);
    check(none == REPLY_PENDING && app.table().pick_popup_open() && app.table().pick_popup_rows() == vp->rows,
        at + ": a click on the dual land does not open the popup of its two rows");
    const std::vector<int> &rows = app.table().pick_popup_rows();
    const std::vector<TRect> &items = app.table().pick_popup_items();
    int item = -1;
    for (size_t k = 0; k < rows.size() && k < items.size(); ++k)
        if (rows[k] == want_r) item = static_cast<int>(k);
    check(item >= 0, at + ": the popup has no (R) item");
    if (item < 0) return;
    const TRect it = items[static_cast<size_t>(item)];
    rmwork::mouse_click(ImVec2((it.x0 + it.x1) / 2, (it.y0 + it.y1) / 2), frames);
    const int got = s.match()->decisions[p.index].reply;
    check(got == want_r,
        at + ": the popup's (R) item replied " + std::to_string(got) + ", not row " + std::to_string(want_r));
    if (got != want_r) return;

    // Dismember: its Phyrexian pips by the prompt's buttons, then lands.
    const int dis = drive_to(app, frames, [](const std::string &c, const std::string &t) {
        return c == "CAST_SPELL" && t.find("Dismember") != std::string::npos;
    });
    check(dis >= 0, at + ": no decision offers to cast Dismember");
    if (dis < 0 || !app.play_reply(dis) || !wait_decision(frames, s)) return;
    bool phyrexian = false;
    for (const std::string &pip : app.table().decision().pips) phyrexian = phyrexian || pip == "B/P";
    check(app.table().decision().kind == DecisionKind::PAY && phyrexian && app.table().decision().picks.empty(),
        at + ": Dismember's Phyrexian pip is not on the prompt line");
    if (!click_button(check, app, frames, "life", at + ": the pay-life button") || !wait_decision(frames, s)) return;
    if (!click_button(check, app, frames, "Pay {B}", at + ": the pay-{B} button") || !wait_decision(frames, s)) return;
    // Cast while Lightning Bolt is still on the stack, Dismember has two creatures to choose from.
    p = pending(s);
    if (p.ok() && p.view->n() && p.view->category(0) == "SELECT_TARGET") {
        const uint64_t delver = object_named(app, "Delver of Secrets", false);
        if (!click_object(
                check, app, frames, delver, row_naming(p, delver, -1, ""), at + ": Dismember at the Delver") ||
            !wait_decision(frames, s))
            return;
    }
    // {1}{B} with the lands: a swamp, then the mountain.
    for (const char *land : {"Swamp", "Mountain"}) {
        p = pending(s);
        const uint64_t h = object_named(app, land, true);
        const DecisionPick *lp = app.table().decision().pick_of_object(h);
        check(app.table().decision().kind == DecisionKind::PAY && lp && lp->rows.size() == 1,
            at + ": paying Dismember does not light the " + land + " (one row)");
        check_picks_shown(check, app, at + " (Dismember)");
        if (!lp || !click_object(check, app, frames, h, lp->rows[0], at + ": a click on the " + land) ||
            !wait_decision(frames, s))
            return;
    }
    // A priority decision with Dismember on the stack: the untapped swamp taps for mana, and with the
    // last mountain untapped the next decision is asked, the {B} floating.
    p = pending(s);
    const uint64_t swamp = object_named(app, "Swamp", true);
    const TableDecision &dm = app.table().decision();
    check(dm.kind == DecisionKind::MANA && dm.pick_of_object(swamp) && !dm.prompted() &&
              app.table().layout().prompt.w() == 0,
        at + ": a priority decision with a mana ability does not light the untapped swamp");
    const int mana_row = row_naming(p, swamp, -1, "(B)");
    if (!click_object(check, app, frames, swamp, mana_row, at + ": the swamp for mana") || !wait_decision(frames, s))
        return;
    const TableModel *m = app.table().model();
    check(m && pool_text(m->players[0]) == "{B}", at + ": the floating {B} is not on the player");
    std::printf(
        "rmwork: %s: a dual land's popup, Phyrexian pips by button, lands paying and tapping for mana\n", at.c_str());
    mouse_to(ImVec2(-1000, -1000), frames);
    s.stop();
}

// A sculpted combat as seat A against the `first` agent (it attacks with every creature and blocks
// with each of its creatures the first attacker): A's Goblin Guide, Monastery Swiftspear and Grizzly
// Bears against B's Grizzly Bears, Baleful Strix and Kaito, Bane of Nightmares (a planeswalker to
// attack); A's Lightning Bolt and untapped Mountain give it priority decisions in B's combat, whose
// frames show the declared blocks. Attackers planned by clicks and sent, the damage of the double-blocked Goblin Guide
// assigned, B's attack blocked by a plan (a drag among the clicks); then the game in Replay.
static void check_attackers(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table attackers";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Lightning Bolt,Mountain", "--library-a", "Mountain", "--battlefield-a",
                "Goblin Guide,Monastery Swiftspear,Grizzly Bears,Mountain", "--hand-b", "Island", "--library-b",
                "Island", "--battlefield-b", "Grizzly Bears,Baleful Strix,Kaito Bane of Nightmares,Island", "--agent-b",
                "first"},
            at))
        return;
    const int row =
        drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "SELECT_ATTACKER"; });
    check(row >= 0, at + ": no decision declares attackers");
    if (row < 0) return;
    const uint64_t guide = object_of(app, "Goblin Guide", 0), spear = object_of(app, "Monastery Swiftspear", 0),
                   bears = object_of(app, "Grizzly Bears", 0), kaito = object_of(app, "Kaito, Bane of Nightmares", 1);
    const TableDecision &d = app.table().decision();
    check(d.kind == DecisionKind::ATTACKERS && d.pick_of_object(guide) && d.pick_of_object(spear) &&
              d.pick_of_object(bears) && d.confirm >= 0 && kaito,
        at + ": the declaration does not light the three creatures that can attack");
    check_picks_shown(check, app, at);
    const size_t index = static_cast<size_t>(s.pending_index());
    const CombatPlan &plan = app.table().plan();
    // The guide at Kaito: planned against the player, armed, then aimed by a click on Kaito.
    click_card(app, frames, guide);
    check(plan.attack_of(guide) && plan.attack_of(guide)->target == 0 && plan.armed() == guide,
        at + ": a click on Goblin Guide does not plan it (armed, the player its target)");
    click_card(app, frames, kaito);
    check(plan.attack_of(guide) && plan.attack_of(guide)->target == kaito && plan.armed() == 0,
        at + ": a click on Kaito does not aim the armed Goblin Guide at it");
    // The swiftspear at the player: armed, then the opponent's chip.
    click_card(app, frames, spear);
    mouse_click(screen_point(Mode::PLAY, app.table().chip_rect(1)), frames);
    check(plan.attack_of(spear) && plan.attack_of(spear)->target == 0 && plan.attack_of(spear)->target_seat == 1 &&
              plan.armed() == 0,
        at + ": the swiftspear is not planned at the player by a click on the opponent's chip");
    // The bears planned and taken back.
    click_card(app, frames, bears);
    click_card(app, frames, bears);
    check(plan.attack_of(bears) == nullptr && plan.attacks().size() == 2,
        at + ": a second click on the bears does not take its planned attack back");
    const TableModel *m = app.table().model();
    const int gi = m ? m->find(guide) : -1;
    check(gi >= 0 && m->objects[static_cast<size_t>(gi)].planned_attack,
        at + ": the planned Goblin Guide does not stand forward");
    check(s.match()->decisions[index].reply == REPLY_PENDING, at + ": planning sent a reply");
    check(app.table().prompt_buttons().size() == 3 && app.table().prompt_buttons()[0].label == "Attack with 2",
        at + ": the prompt line does not offer Attack with 2, Attack with all and Clear");
    if (!press_button(check, app, frames, "Attack with 2", at)) return;
    if (!wait_sent(frames, app)) {
        check(false, at + ": the plan was not sent");
        return;
    }
    const Pending p = frame_after(app, index, [](const Pending &f) {
        const agents::Table t = f.view->table("turn");
        return t.rows() && t.u(0, t.col("attackers_declared")) != 0;
    });
    check(combat_has(p, guide, kaito, -1, 0) && combat_has(p, spear, 0, 1, 0) && !combat_has(p, bears, 0, -1, 0) &&
              combat_attackers(p) == 2,
        at + ": the combat table after the declaration is not Goblin Guide at Kaito and the swiftspear at Player B");
    std::printf(
        "rmwork: %s: planned by clicks (one taken back, one aimed at a planeswalker), sent, declared\n", at.c_str());
    check_damage(check, app, frames);
}

// The guide blocked by B's bears and the strix: the strix given its lethal damage by a click.
static void check_damage(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table damage assignment";
    PlaySession &s = app.play();
    const int row =
        drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "ASSIGN_DAMAGE"; });
    check(row >= 0, at + ": no decision assigns combat damage");
    if (row < 0) return;
    const uint64_t guide = object_of(app, "Goblin Guide", 0), bears = object_of(app, "Grizzly Bears", 1),
                   strix = object_of(app, "Baleful Strix", 1);
    const TableDecision &d = app.table().decision();
    const DecisionPick *pb = d.pick_of_object(bears), *ps = d.pick_of_object(strix);
    check(d.kind == DecisionKind::DAMAGE && d.source == guide && pb && pb->lethal == 2 && ps && ps->lethal == 1 &&
              d.buttons.size() == 1 && d.owed.find("2 of 2") != std::string::npos,
        at + ": the assignment does not light the blockers with their lethal damage and Done");
    check_picks_shown(check, app, at);
    if (!ps || !click_object(check, app, frames, strix, ps->rows[0], at + ": the strix") || !wait_decision(frames, s))
        return;
    const TableModel *m = app.table().model();
    const int bi = m ? m->find(bears) : -1;
    check(m && m->find(strix) < 0 && bi >= 0 && m->objects[static_cast<size_t>(bi)].damage == 0,
        at + ": the strix did not die with the bears unharmed");
    std::printf("rmwork: %s: the lethal blocker clicked\n", at.c_str());
    check_blockers(check, app, frames);
}

// B attacks with its bears and Kaito: A's bears planned onto Kaito (a click, taken back, then a drag).
static void check_blockers(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table blockers";
    PlaySession &s = app.play();
    const int row =
        drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "SELECT_BLOCKER"; });
    check(row >= 0, at + ": no decision declares blockers");
    if (row < 0) return;
    const uint64_t mine = object_of(app, "Grizzly Bears", 0), theirs = object_of(app, "Grizzly Bears", 1),
                   kaito = object_of(app, "Kaito, Bane of Nightmares", 1);
    const TableDecision &d = app.table().decision();
    check(d.kind == DecisionKind::BLOCKERS && d.pick_of_object(mine) && d.block.attackers.size() == 2,
        at + ": the declaration does not light the bears against two attackers");
    const size_t index = static_cast<size_t>(s.pending_index());
    const CombatPlan &plan = app.table().plan();
    click_card(app, frames, mine);
    check(plan.armed() == mine, at + ": a click on the bears does not arm it");
    click_card(app, frames, theirs);
    check(plan.block_of(mine) && plan.block_of(mine)->attacker == theirs && plan.armed() == 0,
        at + ": a click on an attacker does not plan the armed bears' block");
    click_card(app, frames, mine);
    check(plan.block_of(mine) == nullptr, at + ": a click on the planned blocker does not take its block back");
    ImVec2 from, to;
    if (card_point(app, mine, &from) && card_point(app, kaito, &to)) mouse_drag(from, to, frames);
    check(plan.block_of(mine) && plan.block_of(mine)->attacker == kaito,
        at + ": a drag from the bears onto Kaito does not plan the block");
    check(s.match()->decisions[index].reply == REPLY_PENDING, at + ": planning sent a reply");
    if (!press_button(check, app, frames, "Block with 1", at)) return;
    if (!wait_sent(frames, app)) {
        check(false, at + ": the plan was not sent");
        return;
    }
    const Pending p = frame_after(app, index, [mine](const Pending &f) {
        const agents::Table c = f.view->table("combat");
        for (size_t r = 0; r < c.rows(); ++r)
            if (c.u(r, c.col("blocker")) == mine) return true;
        return false;
    });
    check(combat_has(p, kaito, 0, -1, mine) && !combat_has(p, theirs, 0, -1, mine),
        at + ": the combat table after the declaration does not have the bears blocking Kaito");
    std::printf("rmwork: %s: planned by click and by drag, taken back, sent, declared\n", at.c_str());
    check_combat_replay(check, app, frames);
}

// The combat game in Replay: the attacker declaration lit in seat A's view, a click selects only.
static void check_combat_replay(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table combat in Replay";
    const std::string log = app.play().log_path();
    app.play().stop();
    check(app.open_replay(log), at + ": the decision log does not open");
    const auto start = std::chrono::steady_clock::now();
    while (app.loading() &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < DECISION_TIMEOUT_S)
        frames(1);
    frames(2);
    if (!app.replay().loaded()) {
        check(false, at + ": the log is not loaded");
        return;
    }
    long declare = -1;
    const Match &m = *app.replay().match();
    for (size_t i = 0; i < m.decisions.size() && declare < 0; ++i) {
        if (m.decisions[i].decider != 0) continue;
        std::string err;
        const auto f = m.frame(i, 0, &err);
        if (!f) continue;
        const agents::DecisionView v(m.stream(0), *f, 0);
        if (v.n() && v.category(0) == "SELECT_ATTACKER") declare = static_cast<long>(i);
    }
    check(declare >= 0, at + ": the replay has no attacker declaration of seat A");
    if (declare < 0) return;
    app.set_tool_open(Mode::REPLAY, Tool::TABLE, true);
    app.focus_tool(Tool::TABLE);
    app.replay().set_view(0);
    app.replay().go_to(static_cast<size_t>(declare));
    frames(3);
    const uint64_t guide = object_named(app, "Goblin Guide", false);
    check(app.table().decision().kind == DecisionKind::ATTACKERS && app.table().decision().pick_of_object(guide),
        at + ": seat A's view does not light the declaration");
    TRect r;
    if (card_rect(app, guide, &r)) {
        mouse_click(screen_point(Mode::REPLAY, r), frames);
        const TableModel *tm = app.table().model();
        const int gi = tm ? tm->find(guide) : -1;
        check(app.links().selected().kind == LinkItem::Kind::OBJECT && app.links().selected().handle == guide &&
                  app.table().plan().empty() && gi >= 0 && !tm->objects[static_cast<size_t>(gi)].planned_attack &&
                  app.table().take_answer() < 0 && app.replay().cursor() == static_cast<size_t>(declare),
            at + ": a click on a creature that can attack does more than select it");
    }
    mouse_to(ImVec2(-1000, -1000), frames);
    app.set_mode(Mode::PLAY);
    frames(2);
}

namespace rmwork {

void check_table_decisions(const CheckFn &check, App &app, const std::function<void(int)> &frames) {
    check_targeting(check, app, frames);
    check_mana(check, app, frames);
    check_attackers(check, app, frames);
}

}  // namespace rmwork
