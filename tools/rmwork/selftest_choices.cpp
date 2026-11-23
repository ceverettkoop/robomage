#include "selftest_choices.h"

#include <algorithm>
#include <cstdio>
#include <set>

#include "selftest_live.h"

using rmwork::CheckFn;
using namespace rmwork::live;

static bool is_category(const std::string &c, const char *want);
static RowTest category_with(const char *category, const std::string &containing);
static const rmwork::ChoicePanelView::Item *panel_item(rmwork::App &app, const std::string &text, bool button);
static bool click_panel(const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &text, int want,
    const std::string &at);
static void type_filter(rmwork::App &app, const Frames &frames, const std::string &text);
static bool cast_and_wait(const CheckFn &check, rmwork::App &app, const Frames &frames, const char *category,
    const std::string &containing, const std::string &at);
static bool target_player_b(const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &at);
static int life_of(const Pending &p, int seat);
static uint64_t hand_card(const rmwork::App &app, const std::string &name);
static int tokens_of(const Pending &p, int seat);
static bool in_zone(const Pending &p, const std::string &name, const char *zone, int owner);
static bool object_tapped(const Pending &p, const std::string &name, const char *zone);
static void check_kozilek(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_kozilek_replay(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_brutality(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_fury(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_therapy(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_daze_force(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_delve(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_cleanup_discard(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_thoughtseize(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_yesno_legend(const CheckFn &check, rmwork::App &app, const Frames &frames);

static bool is_category(const std::string &c, const char *want) {
    return c == want;
}

// A row of `category` whose text contains `containing`.
static RowTest category_with(const char *category, const std::string &containing) {
    return [category, containing](const std::string &c, const std::string &t) {
        return is_category(c, category) && t.find(containing) != std::string::npos;
    };
}

// The choice panel's item (or, `button`, foot button) whose label contains `text`, as last drawn.
static const rmwork::ChoicePanelView::Item *panel_item(rmwork::App &app, const std::string &text, bool button) {
    for (const rmwork::ChoicePanelView::Item &it : app.table().panel_view().items())
        if (it.button == button && it.label.find(text) != std::string::npos) return &it;
    return nullptr;
}

// A click on the choice panel's item labelled with `text` answers with row `want`.
static bool click_panel(const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &text, int want,
    const std::string &at) {
    const rmwork::ChoicePanelView::Item *it = panel_item(app, text, false);
    check(it != nullptr && want >= 0, at + ": the choice panel shows no \"" + text + "\", or no row is it");
    if (!it || want < 0) return false;
    const size_t index = static_cast<size_t>(app.play().pending_index());
    const int got = click_reply(app, frames, ImVec2((it->min.x + it->max.x) / 2, (it->min.y + it->max.y) / 2), index);
    check(got == want, at + ": the click replied " + std::to_string(got) + ", not row " + std::to_string(want));
    return got == want;
}

// Types `text` into the choice panel's search box.
static void type_filter(rmwork::App &app, const Frames &frames, const std::string &text) {
    rmwork::ChoicePanelView &v = app.table().panel_view();
    rmwork::mouse_click(
        ImVec2((v.filter_min().x + v.filter_max().x) / 2, (v.filter_min().y + v.filter_max().y) / 2), frames);
    ImGui::GetIO().AddInputCharactersUTF8(text.c_str());
    frames(3);
}

// Drives to the decision offering a row of `category` containing `containing`, answers it and waits for
// the next decision.
static bool cast_and_wait(const CheckFn &check, rmwork::App &app, const Frames &frames, const char *category,
    const std::string &containing, const std::string &at) {
    const int row = drive_to(app, frames, category_with(category, containing));
    check(row >= 0, at + ": no decision offers \"" + containing + "\"");
    if (row < 0 || !app.play_reply(row) || !wait_decision(frames, app.play())) return false;
    frames(2);
    return true;
}

// A click on Player B's chip answers a target decision with B's row.
static bool target_player_b(const CheckFn &check, rmwork::App &app, const Frames &frames, const std::string &at) {
    const Pending p = pending(app.play());
    const int want = row_naming(p, 0, 1, "");
    const int got = click_reply(app, frames, screen_point(rmwork::Mode::PLAY, app.table().chip_rect(1)), p.index);
    check(want >= 0 && got == want,
        at + ": a click on Player B's chip replied " + std::to_string(got) + ", not row " + std::to_string(want));
    return want >= 0 && got == want && wait_decision(frames, app.play());
}

// The first card named `name` in the bottom seat's hand on the Table, 0 when none.
static uint64_t hand_card(const rmwork::App &app, const std::string &name) {
    const rmwork::TableModel *m = const_cast<rmwork::App &>(app).table().model();
    if (!m) return 0;
    for (int i : m->hand)
        if (m->objects[static_cast<size_t>(i)].name == name) return m->objects[static_cast<size_t>(i)].handle;
    return 0;
}

// The tokens on the battlefield seat `seat` controls.
static int tokens_of(const Pending &p, int seat) {
    if (!p.ok()) return 0;
    const agents::Table &o = p.view->objects();
    int n = 0;
    for (size_t r = 0; r < o.rows(); ++r)
        n += p.view->stream().enum_name("zone", o.u(r, o.col("zone"))) == "BATTLEFIELD" && o.u(r, o.col("token")) &&
                     o.u(r, o.col("controller")) == static_cast<uint64_t>(seat + 1)
                 ? 1
                 : 0;
    return n;
}

static int life_of(const Pending &p, int seat) {
    if (!p.ok()) return -1;
    const agents::Table t = p.view->table("player");
    for (size_t r = 0; r < t.rows(); ++r)
        if (t.u(r, t.col("player")) == static_cast<uint64_t>(seat + 1)) return static_cast<int>(t.s(r, t.col("life")));
    return -1;
}

// Whether the frame lists a card named `name` in zone `zone` ("GRAVEYARD", …) of seat `owner`.
static bool in_zone(const Pending &p, const std::string &name, const char *zone, int owner) {
    if (!p.ok()) return false;
    const agents::Table &o = p.view->objects();
    for (size_t r = 0; r < o.rows(); ++r)
        if (p.view->stream().enum_name("zone", o.u(r, o.col("zone"))) == zone &&
            o.u(r, o.col("owner")) == static_cast<uint64_t>(owner + 1) && !o.none(r, o.col("card")) &&
            card_name(p, o.card(r, o.col("card"))) == name)
            return true;
    return false;
}

// Whether the first object named `name` in zone `zone` is tapped (false when there is none).
static bool object_tapped(const Pending &p, const std::string &name, const char *zone) {
    if (!p.ok()) return false;
    const agents::Table &o = p.view->objects();
    for (size_t r = 0; r < o.rows(); ++r)
        if (p.view->stream().enum_name("zone", o.u(r, o.col("zone"))) == zone && !o.none(r, o.col("card")) &&
            card_name(p, o.card(r, o.col("card"))) == name)
            return o.u(r, o.col("tapped")) != 0;
    return false;
}

// Kozilek's Command: X by the stepper, two modes in the panel (the second marking the first chosen), the
// spell on the stack with X 2 and two modes; then the game in Replay.
static void check_kozilek(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: Kozilek's Command";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Kozilek's Command,Mountain", "--library-a", "Mountain", "--battlefield-a",
                "Ancient Tomb,Ancient Tomb,Mountain", "--hand-b", "Island", "--library-b", "Island", "--battlefield-b",
                "Grizzly Bears"},
            at) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Kozilek's Command", at))
        return;
    const TableDecision &d = app.table().decision();
    check(d.kind == DecisionKind::NUMBER && d.numbers == std::vector<int>{0, 1, 2, 3} && !d.delve &&
              app.table().prompt_buttons().size() == 3 && app.table().number_value() == 0,
        at + ": choosing X does not offer the stepper over 0 to 3");
    for (int k = 0; k < 2; ++k)
        if (!press_button(check, app, frames, "+", at)) return;
    check(app.table().number_value() == 2 &&
              number_pips(app.table().decision(), 2) == std::vector<std::string>{"2", "C", "C"},
        at + ": + twice does not set X 2, its cost {2}{C}{C}");
    const Pending px = pending(s);
    const int x2 =
        find_row(px, [](const std::string &c, const std::string &t) { return c == "CHOOSE_X" && t == "X = 2"; });
    if (!click_button(check, app, frames, "X = 2", at + ": X = 2") || !wait_decision(frames, s)) return;
    frames(2);
    (void)x2;
    // The first mode, X 2 on the spell.
    Pending p = pending(s);
    bool x_two = false;
    if (p.ok())
        for (size_t r = 0; r < p.view->stack().rows(); ++r)
            x_two = x_two || p.view->stack().s(r, p.view->stack().col("x")) == 2;
    check(x_two, at + ": after X = 2 the spell on the stack has not X 2");
    const ChoicePanel &panel = app.table().panel();
    check(app.table().decision().kind == DecisionKind::MODES && panel.active && panel.kind == DecisionKind::MODES &&
              app.table().panel_view().shown() && panel.items.size() >= 3,
        at + ": the mode decision does not show its modes in the choice panel");
    if (!click_panel(check, app, frames, "Eldrazi Spawn", find_row(p, category_with("CHOOSE_MODE", "Eldrazi Spawn")),
            at + ": the Spawn mode") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    if (!target_player_b(check, app, frames, at + ": the Spawn mode's target")) return;
    frames(2);
    p = pending(s);
    const ChoicePanel &second = app.table().panel();
    bool marked = false;
    for (const ChoiceItem &it : second.items)
        marked = marked || (it.chosen_before && it.row < 0 && it.text.find("Eldrazi Spawn") != std::string::npos &&
                               it.note.find("Player B") != std::string::npos);
    check(second.active && marked && !second.item_of_row(find_row(p, category_with("CHOOSE_MODE", "Eldrazi Spawn"))),
        at + ": the second mode decision does not mark the Spawn mode chosen, with its target");
    const size_t index = p.index;
    if (!click_panel(check, app, frames, "scries", find_row(p, category_with("CHOOSE_MODE", "scries")),
            at + ": the scry mode") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    if (app.table().decision().kind == DecisionKind::TARGET && !target_player_b(check, app, frames, at)) return;
    // Resolved: Player B has X (2) Eldrazi Spawn tokens.
    const Pending after = frame_after(app, index, [](const Pending &q) { return tokens_of(q, 1) > 0; });
    check(tokens_of(after, 1) == 2, at + ": once the spell resolved Player B has not its two Spawn tokens");
    std::printf("rmwork: %s: X by the stepper, two modes in the panel (the first marked chosen)\n", at.c_str());
    check_kozilek_replay(check, app, frames);
}

// The Kozilek's Command game in Replay: the stepper at X 2, the panel read-only with the mode chosen marked;
// nothing in seat B's view.
static void check_kozilek_replay(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices in Replay";
    if (!open_log_in_replay(check, app, frames, at)) return;
    const Match &m = *app.replay().match();
    const auto has = [](const char *category) {
        return [category](const agents::DecisionView &v) {
            for (size_t r = 0; r < v.n(); ++r)
                if (v.category(r) == category) return true;
            return false;
        };
    };
    const long x = first_decision(m, 0, has("CHOOSE_X"));
    const long mode = first_decision(m, 0, has("CHOOSE_MODE"));
    check(x >= 0 && mode >= 0, at + ": the replay has no X or mode decision of seat A");
    if (x < 0 || mode < 0) return;
    app.replay().set_view(0);
    app.replay().go_to(static_cast<size_t>(x));
    frames(3);
    check(app.table().decision().kind == DecisionKind::NUMBER && app.table().number_value() == 2,
        at + ": the X decision's stepper is not at the X chosen (2)");
    app.replay().go_to(static_cast<size_t>(mode));
    frames(3);
    const ChoicePanel &p = app.table().panel();
    const int chosen = m.decisions[static_cast<size_t>(mode)].reply;
    check(p.active && p.chosen_row == chosen && p.item_of_row(chosen) &&
              p.item_of_row(chosen)->text.find("Eldrazi Spawn") != std::string::npos &&
              app.table().panel_view().shown(),
        at + ": the mode decision's panel does not mark the Spawn mode chosen");
    const ChoicePanelView::Item *it = panel_item(app, "scries", false);
    if (it) {
        mouse_click(ImVec2((it->min.x + it->max.x) / 2, (it->min.y + it->max.y) / 2), frames);
        check(app.table().take_answer() < 0 && app.replay().cursor() == static_cast<size_t>(mode),
            at + ": a click on a mode in Replay does more than nothing");
    }
    app.replay().set_view(1);
    frames(3);
    check(!app.table().panel().active && !app.table().panel_view().shown(),
        at + ": seat B's view of seat A's mode decision shows the panel");
    mouse_to(ImVec2(-1000, -1000), frames);
    app.set_mode(Mode::PLAY);
    frames(2);
    std::printf("rmwork: %s: the stepper at the X chosen, the mode chosen marked, read-only\n", at.c_str());
}

// Collective Brutality: one mode (its target clicked), then Done by the prompt line's button.
static void check_brutality(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: Collective Brutality";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Collective Brutality,Lightning Bolt,Ponder", "--library-a", "Mountain", "--battlefield-a",
                "Badlands,Badlands", "--hand-b", "Island", "--library-b", "Island", "--battlefield-b", "Grizzly Bears"},
            at) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Collective Brutality", at))
        return;
    Pending p = pending(s);
    if (!click_panel(check, app, frames, "-2/-2", find_row(p, category_with("CHOOSE_MODE", "-2/-2")), at) ||
        !wait_decision(frames, s))
        return;
    frames(2);
    const uint64_t bears = object_of(app, "Grizzly Bears", 1);
    p = pending(s);
    if (app.table().decision().kind == DecisionKind::TARGET) {
        if (!click_object(check, app, frames, bears, row_naming(p, bears, -1, ""), at + ": the bears") ||
            !wait_decision(frames, s))
            return;
        frames(2);
    }
    p = pending(s);
    const int done = find_row(p, [](const std::string &c, const std::string &) { return c == "CONFIRM_MODES"; });
    check(done >= 0 && panel_item(app, "Done choosing modes", true) != nullptr,
        at + ": the second mode decision does not offer Done in the panel's foot");
    const size_t index = p.index;
    if (!click_button(check, app, frames, "Done choosing modes", at + ": Done") || !wait_decision(frames, s)) return;
    // Resolved with its one mode: -2/-2 killed the bears.
    const Pending after =
        frame_after(app, index, [](const Pending &q) { return in_zone(q, "Grizzly Bears", "GRAVEYARD", 1); });
    check(after.ok(), at + ": after Done the spell did not resolve with its one mode (the bears alive)");
    std::printf("rmwork: %s: a mode in the panel, Done by the prompt line's button\n", at.c_str());
}

// Fury: the evoke cost's exile clicked on the hand, the triggers ordered in the panel, two targets, a point of
// divided damage clicked.
static void check_fury(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: Fury";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Fury,Lightning Bolt,Goblin Guide", "--library-a", "Mountain", "--battlefield-a", "Mountain",
                "--hand-b", "Island", "--library-b", "Island", "--battlefield-b",
                "Grizzly Bears,Thalia Guardian of Thraben"},
            at) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Fury (alternate cost)", at))
        return;
    Pending p = pending(s);
    const uint64_t bolt = hand_card(app, "Lightning Bolt");
    check(app.table().decision().kind == DecisionKind::PAY && app.table().decision().cost_part == "alternative cost",
        at + ": the evoke cost is not a payment of the alternative cost on the Table");
    if (!click_object(check, app, frames, bolt, row_naming(p, bolt, -1, "Exile"), at + ": Lightning Bolt exiled") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    p = pending(s);
    check(app.table().decision().kind == DecisionKind::ORDER && app.table().panel().items.size() == 2,
        at + ": the two triggers are not in the choice panel");
    if (!click_panel(check, app, frames, "DealDamage", find_row(p, category_with("ORDER_TRIGGERS", "DealDamage")),
            at + ": the damage trigger first") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    const uint64_t bears = object_of(app, "Grizzly Bears", 1),
                   thalia = object_of(app, "Thalia, Guardian of Thraben", 1);
    p = pending(s);
    if (!click_object(check, app, frames, bears, row_naming(p, bears, -1, ""), at + ": the bears") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    p = pending(s);
    if (!click_object(check, app, frames, thalia, row_naming(p, thalia, -1, ""), at + ": Thalia") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    if (!click_button(check, app, frames, "Done selecting targets", at + ": Done") || !wait_decision(frames, s)) return;
    frames(2);
    p = pending(s);
    const TableDecision &d = app.table().decision();
    const DecisionPick *pb = d.pick_of_object(bears);
    check(d.kind == DecisionKind::TARGET && pb && pb->so_far == 1 && d.owed.find("given so far") != std::string::npos,
        at + ": the divided damage decision does not count a point on each target");
    if (!click_object(check, app, frames, bears, row_naming(p, bears, -1, "one more"), at + ": a point to the bears") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    const DecisionPick *pb2 = app.table().decision().pick_of_object(bears);
    check(app.table().decision().kind == DecisionKind::TARGET && pb2 && pb2->so_far == 2,
        at + ": the next point's decision does not count 2 on the bears");
    std::printf(
        "rmwork: %s: the evoke exile, the triggers ordered, the targets and a point of damage clicked\n", at.c_str());
}

// Cabal Therapy: the name chosen in the panel after a search, the opponent discarding it; the flashback's
// sacrifice by a click.
static void check_therapy(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: Cabal Therapy";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Cabal Therapy", "--library-a", "Swamp", "--battlefield-a", "Swamp,Grizzly Bears,Goblin Guide",
                "--hand-b", "Brainstorm,Ponder,Fatal Push,Island", "--library-b", "Island"},
            at, 1) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Cabal Therapy", at))
        return;
    if (app.table().decision().kind == DecisionKind::TARGET && !target_player_b(check, app, frames, at)) return;
    if (drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "NAME_CARD"; }) < 0) {
        check(false, at + ": the name decision did not come");
        return;
    }
    frames(2);
    const ChoicePanel &p = app.table().panel();
    std::set<std::string> names;
    for (const ChoiceItem &it : p.items) names.insert(it.text);
    check(app.table().decision().kind == DecisionKind::NAME && p.active && p.searchable &&
              names == std::set<std::string>{"Brainstorm", "Ponder", "Fatal Push"} && !p.groups.empty() &&
              p.items[0].group == 0,
        at + ": the name decision's panel does not list Player B's three nonland cards, grouped");
    type_filter(app, frames, "pon");
    check(app.table().panel_view().filter() == "pon" && app.table().panel_view().items_shown() == 1,
        at + ": typing \"pon\" does not filter the names to one");
    Pending pn = pending(s);
    const size_t index = pn.index;
    if (!click_panel(
            check, app, frames, "Ponder", find_row(pn, category_with("NAME_CARD", "Ponder")), at + ": Ponder") ||
        !wait_decision(frames, s))
        return;
    const Pending after = frame_after(app, index, [](const Pending &) { return true; });
    check(in_zone(after, "Ponder", "GRAVEYARD", 1), at + ": Player B did not discard the Ponder named");
    // The flashback: sacrifice a creature.
    if (!cast_and_wait(check, app, frames, "CAST_SPELL", "flashback", at)) return;
    if (app.table().decision().kind == DecisionKind::TARGET && !target_player_b(check, app, frames, at)) return;
    frames(2);
    const uint64_t bears = object_of(app, "Grizzly Bears", 0);
    const Pending ps = pending(s);
    check(app.table().decision().kind == DecisionKind::CARDS && app.table().decision().pick_of_object(bears) &&
              !app.table().picker().active,
        at + ": the flashback's sacrifice does not light the bears on the Table");
    check_picks_shown(check, app, at + " (sacrifice)");
    if (!click_object(check, app, frames, bears, row_naming(ps, bears, -1, "Sacrifice"), at + ": the bears sacrificed"))
        return;
    std::printf("rmwork: %s: a name searched and chosen in the panel, a creature sacrificed by a click\n", at.c_str());
}

// Lightning Bolt then Daze: the Island returned by a click, the unless cost declined by its button; a second
// Bolt then Force of Will: a blue card exiled by a click.
static void check_daze_force(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: Daze and Force of Will";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Lightning Bolt,Lightning Bolt,Daze,Force of Will,Brainstorm,Ponder", "--library-a",
                "Mountain", "--battlefield-a", "Volcanic Island,Island,Mountain,Mountain", "--hand-b", "Island",
                "--library-b", "Island"},
            at) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Lightning Bolt", at))
        return;
    if (app.table().decision().kind == DecisionKind::TARGET && !target_player_b(check, app, frames, at)) return;
    if (!cast_and_wait(check, app, frames, "CAST_SPELL", "Daze (alternate cost)", at)) return;
    Pending p = pending(s);
    const uint64_t island = object_named(app, "Island", false);
    check(app.table().decision().kind == DecisionKind::CARDS && app.table().decision().pick_of_object(island) &&
              app.table().decision().cost_part == "alternative cost",
        at + ": Daze's return does not light the Island on the Table");
    if (!click_object(check, app, frames, island, row_naming(p, island, -1, "Return"), at + ": the Island returned"))
        return;
    if (drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "PAY_UNLESS"; }) < 0) {
        check(false, at + ": Daze's unless cost was not offered");
        return;
    }
    frames(2);
    p = pending(s);
    check(app.table().decision().kind == DecisionKind::UNLESS && app.table().decision().prompted(),
        at + ": the unless cost is not on the prompt line");
    const size_t index = p.index;
    if (!click_button(check, app, frames, "Don't pay", at + ": Don't pay") || !wait_decision(frames, s)) return;
    const Pending after = frame_after(app, index, [](const Pending &) { return true; });
    check(in_zone(after, "Lightning Bolt", "GRAVEYARD", 0) && in_zone(after, "Island", "HAND", 0) &&
              life_of(after, 1) == 20,
        at + ": after declining, the Bolt is not countered (or the Island not in hand)");
    // The second Bolt, then Force of Will's pitch.
    if (!cast_and_wait(check, app, frames, "CAST_SPELL", "Lightning Bolt", at)) return;
    if (app.table().decision().kind == DecisionKind::TARGET && !target_player_b(check, app, frames, at)) return;
    if (!cast_and_wait(check, app, frames, "CAST_SPELL", "Force of Will (alternate cost)", at)) return;
    if (app.table().decision().kind == DecisionKind::TARGET) {
        // Force of Will's target: the Bolt on the stack.
        p = pending(s);
        const TableModel *tm = app.table().model();
        uint64_t spell = 0;
        for (const TableStackItem &it : tm->stack)
            if (it.name == "Lightning Bolt") spell = it.handle;
        const int want = row_naming(p, spell, -1, "");
        const size_t ti = p.index;
        int got = REPLY_PENDING;
        for (size_t k = 0; k < app.table().layout().stack_items.size(); ++k)
            if (tm->stack[k].handle == spell)
                got = click_reply(app, frames, screen_point(Mode::PLAY, app.table().layout().stack_items[k]), ti);
        check(want >= 0 && got == want, at + ": a click on the Bolt on the stack does not target it");
        if (!wait_decision(frames, s)) return;
        frames(2);
    }
    p = pending(s);
    const uint64_t brainstorm = hand_card(app, "Brainstorm");
    const uint64_t ponder = hand_card(app, "Ponder");
    const TableDecision &d = app.table().decision();
    check(d.kind == DecisionKind::PAY && d.cost_part == "alternative cost" && d.pick_of_object(brainstorm) &&
              d.pick_of_object(ponder) && d.picks.size() == 2,
        at + ": Force of Will's pitch does not light the blue cards alone");
    const size_t fi = p.index;
    if (!click_object(
            check, app, frames, brainstorm, row_naming(p, brainstorm, -1, "Exile"), at + ": Brainstorm exiled") ||
        !wait_decision(frames, s))
        return;
    const Pending after2 = frame_after(app, fi, [](const Pending &) { return true; });
    check(in_zone(after2, "Brainstorm", "EXILE", 0), at + ": Brainstorm is not in exile after the pitch");
    std::printf("rmwork: %s: the Island returned, the unless cost declined, a blue card pitched\n", at.c_str());
}

// Murktide Regent: delve 5 by the stepper, then a graveyard card exiled in the picker.
static void check_delve(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: delve";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Murktide Regent", "--library-a", "Island", "--battlefield-a", "Island,Island,Island",
                "--graveyard-a", "Ponder,Brainstorm,Lightning Bolt,Mountain,Daze", "--hand-b", "Island", "--library-b",
                "Island"},
            at) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Murktide Regent", at))
        return;
    const TableDecision &d = app.table().decision();
    check(d.kind == DecisionKind::NUMBER && d.delve && d.numbers == std::vector<int>{4, 5},
        at + ": the delve count is not a stepper over 4 and 5");
    if (!press_button(check, app, frames, "+", at)) return;
    check(
        app.table().number_value() == 5 && number_pips(app.table().decision(), 5) == std::vector<std::string>{"U", "U"},
        at + ": + does not make it 5, leaving {U}{U}");
    if (!press_button(check, app, frames, "+", at) || !press_button(check, app, frames, "\xE2\x88\x92", at)) return;
    check(app.table().number_value() == 4 &&
              number_pips(app.table().decision(), 4) == std::vector<std::string>{"1", "U", "U"},
        at + ": + past the last row and \xE2\x88\x92 do not make it 4, leaving {1}{U}{U}");
    // Four of the five: the picks are asked.
    if (!click_button(check, app, frames, "Exile 4", at + ": Exile 4") || !wait_decision(frames, s)) return;
    frames(2);
    Pending p = pending(s);
    check(app.table().decision().kind == DecisionKind::CARDS && app.table().picker().active,
        at + ": delve's picks are not in the card picker");
    if (!click_item(check, app, frames, picker_card(app, "Ponder", static_cast<int>(ZoneKind::GRAVEYARD)),
            find_row(p, category_with("CHOOSE_CARD", "Ponder")), at + ": Ponder exiled") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    const PickerPile &pile = app.table().picker().pile;
    check(pile.chosen.size() == 1 && pile.chosen[0].name == "Ponder",
        at + ": the next pick's picker does not list Ponder chosen");
    std::printf("rmwork: %s: the count by the stepper, a card picked from the graveyard\n", at.c_str());
}

// Lórien Revealed, then at cleanup two cards to discard planned on the hand and sent.
static void check_cleanup_discard(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: cleanup discard";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Lorien Revealed,Mountain,Mountain,Lightning Bolt,Lava Spike,Fireblast,Chain Lightning",
                "--library-a", "Goblin Guide,Monastery Swiftspear,Price of Progress,Mountain", "--battlefield-a",
                "Island,Island,Island,Island,Island", "--hand-b", "Island", "--library-b", "Island"},
            at) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Revealed", at))
        return;
    if (drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "DISCARD"; }) < 0) {
        check(false, at + ": the cleanup discard did not come");
        return;
    }
    frames(2);
    const TableDecision &d = app.table().decision();
    check(d.kind == DecisionKind::CARDS && d.plan_owed == 2 && d.owed.find("9 in hand, 2 to discard") == 0 &&
              d.picks.size() == 9 && !app.table().picker().active,
        at + ": the cleanup discard does not light the 9 hand cards with 2 to discard");
    if (d.picks.size() < 3) return;
    const uint64_t x = d.picks[0].handle, y = d.picks[2].handle;
    const size_t index = static_cast<size_t>(s.pending_index());
    const CardPlan &plan = app.table().card_plan();
    click_card(app, frames, x);
    click_card(app, frames, y);
    check(plan.cards() == std::vector<uint64_t>{x, y} && s.match()->decisions[index].reply == REPLY_PENDING,
        at + ": two clicks do not plan two cards, or planning sent a reply");
    const int want_x = row_naming(pending(s), x, -1, "");
    if (!press_button(check, app, frames, "Discard 2", at)) return;
    if (!wait_sent(frames, app)) {
        check(false, at + ": the plan was not sent");
        return;
    }
    const Match &m = *s.match();
    int second = -1;
    for (size_t i = index + 1; i < m.decisions.size() && second < 0; ++i)
        if (m.decisions[i].decider == 0) second = m.decisions[i].reply;
    const Pending now = frame_after(app, index + 1, [](const Pending &) { return true; });
    bool gone = now.ok();
    if (now.ok()) {
        const agents::Table &o = now.view->objects();
        for (size_t r = 0; r < o.rows(); ++r)
            gone = gone && ((o.u(r, o.col("handle")) != x && o.u(r, o.col("handle")) != y) ||
                               now.view->stream().enum_name("zone", o.u(r, o.col("zone"))) != "HAND");
    }
    check(m.decisions[index].reply == want_x && second >= 0 && gone,
        at + ": the plan's replies are not the planned cards' rows, or the cards stay in the hand");
    std::printf("rmwork: %s: two cards planned on the hand and discarded on confirm\n", at.c_str());
}

// Thoughtseize: Player B's revealed hand in the picker, its lands dimmed; a card picked.
static void check_thoughtseize(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: Thoughtseize";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Thoughtseize", "--library-a", "Swamp", "--battlefield-a", "Swamp", "--hand-b",
                "Lightning Bolt,Lava Spike,Mountain,Fireblast,Mountain", "--library-b", "Mountain"},
            at, 1) ||
        !cast_and_wait(check, app, frames, "CAST_SPELL", "Thoughtseize", at))
        return;
    if (app.table().decision().kind == DecisionKind::TARGET && !target_player_b(check, app, frames, at)) return;
    if (drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "DISCARD"; }) < 0) {
        check(false, at + ": Thoughtseize's pick did not come");
        return;
    }
    frames(2);
    const CardPicker &pk = app.table().picker();
    std::set<std::string> lit, dim;
    for (const CardPickGroup &g : pk.groups)
        for (const CardPickEntry &e : g.entries)
            if (g.seat == 1 && g.zone == ZoneKind::HAND) (e.pickable() ? lit : dim).insert(e.name);
    check(app.table().decision().kind == DecisionKind::CARDS && pk.active && app.table().picker_view().shown() &&
              lit == std::set<std::string>{"Lightning Bolt", "Lava Spike", "Fireblast"} && dim.count("Mountain"),
        at + ": the picker does not show Player B's hand with the nonland cards lit, the Mountains dimmed");
    const Pending p = pending(s);
    const size_t index = p.index;
    if (!click_item(check, app, frames, picker_card(app, "Lava Spike", static_cast<int>(ZoneKind::HAND)),
            find_row(p, category_with("DISCARD", "Lava Spike")), at + ": Lava Spike") ||
        !wait_decision(frames, s))
        return;
    const Pending after = frame_after(app, index, [](const Pending &) { return true; });
    check(in_zone(after, "Lava Spike", "GRAVEYARD", 1), at + ": Player B did not discard the Lava Spike picked");
    std::printf("rmwork: %s: the revealed hand in the picker, the lands dimmed, a card picked\n", at.c_str());
}

// Watery Grave's 2 life accepted by the prompt line's button; the legend rule's Thalia kept by a click.
static void check_yesno_legend(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Choices: yes/no and legend";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Watery Grave,Thalia Guardian of Thraben", "--library-a", "Plains", "--battlefield-a",
                "Thalia Guardian of Thraben,Plains,Plains", "--hand-b", "Island", "--library-b", "Island"},
            at) ||
        !cast_and_wait(check, app, frames, "PLAY_LAND", "Watery Grave", at))
        return;
    Pending p = pending(s);
    const TableDecision &d = app.table().decision();
    check(d.kind == DecisionKind::CHOICE && d.buttons.size() == 2 && app.table().prompt_buttons().size() == 2,
        at + ": Watery Grave's choice is not two buttons on the prompt line");
    const size_t index = p.index;
    if (!click_button(check, app, frames, "Accept", at + ": Accept") || !wait_decision(frames, s)) return;
    const Pending after = frame_after(app, index, [](const Pending &) { return true; });
    check(life_of(after, 0) == 18 && in_zone(after, "Watery Grave", "BATTLEFIELD", 0) &&
              !object_tapped(after, "Watery Grave", "BATTLEFIELD"),
        at + ": after accepting, Player A is not at 18 life with Watery Grave untapped");
    if (!cast_and_wait(check, app, frames, "CAST_SPELL", "Thalia", at)) return;
    if (drive_to(app, frames, [](const std::string &c, const std::string &) { return c == "KEEP_LEGEND"; }) < 0) {
        check(false, at + ": the legend rule's choice did not come");
        return;
    }
    frames(2);
    p = pending(s);
    const uint64_t kept = object_named(app, "Thalia, Guardian of Thraben", false);
    check(app.table().decision().kind == DecisionKind::CARDS && app.table().decision().picks.size() == 2,
        at + ": the legend rule's choice does not light the two Thalias");
    if (!click_object(check, app, frames, kept, row_naming(p, kept, -1, "Keep"), at + ": a Thalia kept")) return;
    std::printf("rmwork: %s: an optional cost accepted by button, a legend kept by a click\n", at.c_str());
}

namespace rmwork {

void check_table_choices(const CheckFn &check, App &app, const std::function<void(int)> &frames) {
    check_kozilek(check, app, frames);
    check_brutality(check, app, frames);
    check_fury(check, app, frames);
    check_therapy(check, app, frames);
    check_daze_force(check, app, frames);
    check_delve(check, app, frames);
    check_cleanup_discard(check, app, frames);
    check_thoughtseize(check, app, frames);
    check_yesno_legend(check, app, frames);
}

}  // namespace rmwork
