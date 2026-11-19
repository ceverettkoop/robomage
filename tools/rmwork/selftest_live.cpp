#include "selftest_live.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace rmwork {
namespace live {

// Frames until the human decides or the game is over; whether the human decides.
bool wait_decision(const Frames &frames, rmwork::PlaySession &s) {
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

Pending pending(rmwork::PlaySession &s) {
    Pending p;
    if (!s.deciding() || !s.match()) return p;
    p.index = static_cast<size_t>(s.pending_index());
    std::string err;
    p.frame = s.match()->frame(p.index, s.config().seat, &err);
    if (p.frame) p.view.reset(new agents::DecisionView(s.match()->stream(s.config().seat), *p.frame, s.config().seat));
    return p;
}

int find_row(const Pending &p, const RowTest &test) {
    if (!p.ok()) return -1;
    for (size_t i = 0; i < p.view->n(); ++i)
        if (test(p.view->category(i), p.view->description(i))) return static_cast<int>(i);
    return -1;
}

// Answers the human's decisions with row 0 until one has a row `test` holds for; that row (the decision
// left pending), -1 when none came.
int drive_to(rmwork::App &app, const Frames &frames, const RowTest &test) {
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
int row_naming(const Pending &p, uint64_t handle, int seat, const std::string &containing) {
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
uint64_t object_named(const rmwork::App &app, const std::string &name, bool untapped) {
    const rmwork::TableModel *m = const_cast<rmwork::App &>(app).table().model();
    if (!m) return 0;
    for (const rmwork::TableObject &o : m->objects)
        if (o.name == name && !o.in_hand && (!untapped || !o.tapped)) return o.handle;
    return 0;
}

// The first battlefield object named `name` controlled by seat `controller`, 0 when none.
uint64_t object_of(const rmwork::App &app, const std::string &name, int controller) {
    const rmwork::TableModel *m = const_cast<rmwork::App &>(app).table().model();
    if (!m) return 0;
    for (const rmwork::TableObject &o : m->objects)
        if (o.name == name && !o.in_hand && o.controller == controller) return o.handle;
    return 0;
}

// The centre of `r` (the Table's panel coordinates) on screen.
ImVec2 screen_point(rmwork::Mode m, const rmwork::TRect &r) {
    const ImVec2 o = rmwork::table_window_origin(m);
    return ImVec2(o.x + (r.x0 + r.x1) / 2, o.y + (r.y0 + r.y1) / 2);
}

// The rectangle object `handle`'s card is drawn in (its pile's front card).
bool card_rect(const rmwork::App &app, uint64_t handle, rmwork::TRect *out) {
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
int click_reply(rmwork::App &app, const Frames &frames, ImVec2 p, size_t index) {
    rmwork::mouse_click(p, frames);
    const rmwork::Match *m = app.play().match();
    return m && index < m->decisions.size() ? m->decisions[index].reply : rmwork::REPLY_PENDING;
}

// Every pick of the Table's decision is an object the frame shows on the Table (the bottom seat's
// battlefield and hand, the stack) or a player.
void check_picks_shown(const CheckFn &check, const rmwork::App &app, const std::string &at) {
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

bool start_sculpted(const CheckFn &check, rmwork::App &app, bool auto_mana, const std::vector<std::string> &setup,
    const std::string &at, uint64_t seed) {
    app.set_tool_open(rmwork::Mode::PLAY, rmwork::Tool::TABLE, true);
    app.focus_tool(rmwork::Tool::TABLE);
    rmwork::NewGameSettings g;
    g.opponent = "explore";
    g.deck = "league/burn";
    g.opponent_deck = "league/bug";
    g.seat = 0;
    g.seed = seed;
    g.auto_mana = auto_mana;
    const bool ok = app.start_game(g, setup);
    check(ok, at + ": the sculpted game does not start");
    return ok;
}

// A click on object `handle`'s card answers with row `want`.
bool click_object(
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
bool click_button(
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
bool press_button(
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
bool card_point(const rmwork::App &app, uint64_t handle, ImVec2 *out) {
    rmwork::TRect r;
    if (!handle || !card_rect(app, handle, &r)) return false;
    *out = screen_point(rmwork::Mode::PLAY, r);
    return true;
}

void click_card(rmwork::App &app, const Frames &frames, uint64_t handle) {
    ImVec2 p;
    if (card_point(app, handle, &p)) rmwork::mouse_click(p, frames);
}

// Frames until the Table's plan is sent (or stopped), then until the human decides; whether it was.
bool wait_sent(const Frames &frames, rmwork::App &app) {
    const auto start = std::chrono::steady_clock::now();
    while (app.table().plan_sending() && app.play().running()) {
        frames(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() > DECISION_TIMEOUT_S)
            return false;
    }
    return !app.table().plan_sending() && wait_decision(frames, app.play());
}

// The first of the human's decisions after `index` (the stops answer some unasked; their frames are
// recorded too) whose frame `test` holds for; not ok() when none.
Pending frame_after(const rmwork::App &app, size_t index, const std::function<bool(const Pending &)> &test) {
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

// The picker's card item named `name` (in zone `zone`, -1 any), as last drawn; nullptr when none.
const rmwork::CardPickerView::Item *picker_card(rmwork::App &app, const std::string &name, int zone) {
    const rmwork::CardPicker &p = app.table().picker();
    for (const rmwork::CardPickerView::Item &it : app.table().picker_view().items()) {
        if (!it.card || it.label != name || it.group >= p.groups.size()) continue;
        if (zone < 0 || static_cast<int>(p.groups[it.group].zone) == zone) return &it;
    }
    return nullptr;
}

// The picker's button labelled `label`: a placement button of the card named `entry`, or ("" entry) a
// button under the cards; nullptr when none.
const rmwork::CardPickerView::Item *picker_button(
    rmwork::App &app, const std::string &entry, const std::string &label) {
    const rmwork::CardPicker &p = app.table().picker();
    for (const rmwork::CardPickerView::Item &it : app.table().picker_view().items()) {
        if (it.card || it.label != label) continue;
        const bool placement = it.group < p.groups.size() && it.entry < p.groups[it.group].entries.size() &&
                               std::find(p.buttons.begin(), p.buttons.end(), it.row) == p.buttons.end();
        if (entry.empty() ? !placement : placement && p.groups[it.group].entries[it.entry].name == entry) return &it;
    }
    return nullptr;
}

// A click on the picker's item `it` answers with row `want`.
bool click_item(const CheckFn &check, rmwork::App &app, const Frames &frames, const rmwork::CardPickerView::Item *it,
    int want, const std::string &at) {
    check(it != nullptr && want >= 0, at + ": the picker does not show the item to click, or no row names it");
    if (!it || want < 0) return false;
    const size_t index = static_cast<size_t>(app.play().pending_index());
    const int got = click_reply(app, frames, ImVec2((it->min.x + it->max.x) / 2, (it->min.y + it->max.y) / 2), index);
    check(got == want, at + ": the click replied " + std::to_string(got) + ", not row " + std::to_string(want));
    return got == want;
}

std::string card_name(const Pending &p, int card) {
    return card >= 0 ? p.view->cards().name(card) : std::string("?");
}

// Stops Play's game and opens its decision log in Replay; whether it loaded.
bool open_log_in_replay(const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &at) {
    const std::string log = app.play().log_path();
    app.play().stop();
    check(app.open_replay(log), at + ": the decision log does not open");
    const auto start = std::chrono::steady_clock::now();
    while (app.loading() &&
           std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < DECISION_TIMEOUT_S)
        frames(1);
    frames(2);
    check(app.replay().loaded(), at + ": the log is not loaded");
    if (!app.replay().loaded()) return false;
    app.set_tool_open(rmwork::Mode::REPLAY, rmwork::Tool::TABLE, true);
    app.focus_tool(rmwork::Tool::TABLE);
    return true;
}

// The first decision of seat `seat` whose frame `test` holds for, -1 none.
long first_decision(const rmwork::Match &m, int seat, const std::function<bool(const agents::DecisionView &)> &test) {
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        if (m.decisions[i].decider != seat) continue;
        std::string err;
        const auto f = m.frame(i, seat, &err);
        if (!f) continue;
        const agents::DecisionView v(m.stream(seat), *f, seat);
        if (test(v)) return static_cast<long>(i);
    }
    return -1;
}

}  // namespace live
}  // namespace rmwork
