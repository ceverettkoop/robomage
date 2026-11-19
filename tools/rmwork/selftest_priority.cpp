#include "selftest_priority.h"

#include <algorithm>
#include <cstdio>

#include "autopass.h"
#include "priority_costs.h"
#include "selftest_live.h"

using rmwork::CheckFn;
using namespace rmwork::live;

static RowTest category_with(const char *category, const std::string &containing);
static int exact_row(const Pending &p, const std::string &description);
static uint64_t hand_card(const rmwork::App &app, const std::string &name);
static int button_row(rmwork::App &app, const std::string &label);
static void check_ways_shown(const CheckFn &check, rmwork::App &app, const std::string &at);
static bool choose_in_popup(const CheckFn &check, rmwork::App &app, const Frames &frames, uint64_t handle,
    const std::vector<int> &rows, int want, const std::string &at);
static void press_space(rmwork::App &app, const Frames &frames);
static void check_land_bolt(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_land_bolt_replay(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_daze(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_fetchland(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_wasteland(const CheckFn &check, rmwork::App &app, const Frames &frames);
static void check_flashback(const CheckFn &check, rmwork::App &app, const Frames &frames);

// A row of `category` whose text contains `containing`.
static RowTest category_with(const char *category, const std::string &containing) {
    return [category, containing](const std::string &c, const std::string &t) {
        return c == category && t.find(containing) != std::string::npos;
    };
}

// The row whose description is `description`, -1 none.
static int exact_row(const Pending &p, const std::string &description) {
    return find_row(p, [&description](const std::string &, const std::string &t) { return t == description; });
}

// The first card named `name` in the bottom seat's hand on the Table, 0 when none.
static uint64_t hand_card(const rmwork::App &app, const std::string &name) {
    const rmwork::TableModel *m = const_cast<rmwork::App &>(app).table().model();
    if (!m) return 0;
    for (int i : m->hand)
        if (m->objects[static_cast<size_t>(i)].name == name) return m->objects[static_cast<size_t>(i)].handle;
    return 0;
}

// The row of the prompt line's button labelled `label`, -1 when there is none.
static int button_row(rmwork::App &app, const std::string &label) {
    for (const rmwork::PromptButton &b : app.table().prompt_buttons())
        if (b.label == label) return b.row;
    return -1;
}

// At a priority decision: every pick a card the Table shows, named by priority rows only; a cost per row.
static void check_ways_shown(const CheckFn &check, rmwork::App &app, const std::string &at) {
    const rmwork::TableDecision &d = app.table().decision();
    check_picks_shown(check, app, at);
    static const char *const WAYS[] = {"CAST_SPELL", "PLAY_LAND", "ACTIVATE_ABILITY", "TURN_FACE_UP", "COMPANION"};
    bool ways = true;
    const Pending pd = pending(app.play());
    for (const rmwork::DecisionPick &p : d.picks)
        for (int row : p.rows) {
            const std::string c = pd.ok() ? pd.view->category(static_cast<size_t>(row)) : "";
            ways =
                ways && (c.rfind("MANA_", 0) == 0 || std::find(std::begin(WAYS), std::end(WAYS), c) != std::end(WAYS));
        }
    check(ways && d.costs.size() == d.row_text.size(),
        at + ": a pick names a row that is no way to act, or a row has no cost");
}

// A click on object `handle` opens the popup of `rows`, and a click on its item for `want` answers with it.
static bool choose_in_popup(const CheckFn &check, rmwork::App &app, const Frames &frames, uint64_t handle,
    const std::vector<int> &rows, int want, const std::string &at) {
    rmwork::TRect r;
    if (!card_rect(app, handle, &r)) {
        check(false, at + ": the card is not on the Table");
        return false;
    }
    const size_t index = static_cast<size_t>(app.play().pending_index());
    const int none = click_reply(app, frames, screen_point(rmwork::Mode::PLAY, r), index);
    check(none == rmwork::REPLY_PENDING && app.table().pick_popup_open() && app.table().pick_popup_rows() == rows,
        at + ": a click does not open the popup of the card's ways");
    const std::vector<int> &shown = app.table().pick_popup_rows();
    const std::vector<rmwork::TRect> &items = app.table().pick_popup_items();
    int item = -1;
    for (size_t k = 0; k < shown.size() && k < items.size(); ++k)
        if (shown[k] == want) item = static_cast<int>(k);
    check(item >= 0, at + ": the popup has no item for row " + std::to_string(want));
    if (item < 0) return false;
    const rmwork::TRect it = items[static_cast<size_t>(item)];
    rmwork::mouse_click(ImVec2((it.x0 + it.x1) / 2, (it.y0 + it.y1) / 2), frames);
    const rmwork::Match *m = app.play().match();
    const int got = m && index < m->decisions.size() ? m->decisions[index].reply : rmwork::REPLY_PENDING;
    check(got == want, at + ": the popup's item replied " + std::to_string(got) + ", not row " + std::to_string(want));
    return got == want;
}

static void press_space(rmwork::App &app, const Frames &frames) {
    const ImGuiKeyChord chord = app.settings().keys[rmwork::action_index(rmwork::Action::PLAY_PASS)];
    const ImGuiKey key = static_cast<ImGuiKey>(chord & ~ImGuiMod_Mask_);
    ImGui::GetIO().AddKeyEvent(key, true);
    frames(1);
    ImGui::GetIO().AddKeyEvent(key, false);
    frames(1);
}

// A Mountain played and Lightning Bolt cast by clicks on the hand, the Bolt's target clicked; Pass by the
// prompt line's button with the Bolt on the stack, then by Space at the next priority decision.
static void check_land_bolt(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table priority: a land and Lightning Bolt";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Mountain,Lightning Bolt,Brainstorm", "--library-a", "Mountain", "--battlefield-a", "Island",
                "--hand-b", "Island", "--library-b", "Island", "--battlefield-b", "Grizzly Bears"},
            at))
        return;
    const int land = drive_to(app, frames, category_with("PLAY_LAND", "Mountain"));
    check(land >= 0, at + ": no decision offers to play a Mountain");
    if (land < 0) return;
    frames(2);
    Pending p = pending(s);
    const TableDecision &d = app.table().decision();
    const uint64_t mountain = hand_card(app, "Mountain");
    const uint64_t brainstorm = hand_card(app, "Brainstorm");
    const int pass = find_row(p, category_with("PASS_PRIORITY", ""));
    check(d.kind == DecisionKind::PRIORITY && d.pick_of_object(mountain) && d.pick_of_object(brainstorm) && pass >= 0 &&
              d.pass == pass && button_row(app, "Pass") == pass && d.prompt.rfind("Play a land", 0) == 0 &&
              app.table().layout().prompt.w() > 0,
        at +
            ": the priority decision does not light the Mountain and Brainstorm, say what may be done, or offer "
            "Pass (prompt \"" +
            d.prompt + "\")");
    check_ways_shown(check, app, at);
    const int play = row_naming(p, mountain, -1, "Play");
    check(play >= 0 && static_cast<size_t>(play) < d.costs.size() && d.costs[static_cast<size_t>(play)].empty(),
        at + ": playing a land costs something");
    if (!click_object(check, app, frames, mountain, play, at + ": the Mountain played") || !wait_decision(frames, s))
        return;
    // Lightning Bolt, {R}.
    const int cast = drive_to(app, frames, category_with("CAST_SPELL", "Lightning Bolt"));
    check(cast >= 0, at + ": no decision offers to cast Lightning Bolt");
    if (cast < 0) return;
    frames(2);
    p = pending(s);
    const uint64_t bolt = hand_card(app, "Lightning Bolt");
    const DecisionPick *bp = app.table().decision().pick_of_object(bolt);
    const TableDecision &dc = app.table().decision();
    check(bp && bp->rows == std::vector<int>{cast} && static_cast<size_t>(cast) < dc.costs.size() &&
              dc.costs[static_cast<size_t>(cast)].pips == std::vector<std::string>{"R"} &&
              way_label(dc.row_text[static_cast<size_t>(cast)], dc.costs[static_cast<size_t>(cast)]) ==
                  "Cast Lightning Bolt {R}",
        at + ": Lightning Bolt is not lit with its one row, or its cost is not {R}");
    if (!click_object(check, app, frames, bolt, cast, at + ": Lightning Bolt cast") || !wait_decision(frames, s))
        return;
    // Its target, clicked on the Table.
    p = pending(s);
    const uint64_t bears = object_of(app, "Grizzly Bears", 1);
    check(app.table().decision().kind == DecisionKind::TARGET, at + ": the Bolt's target decision does not follow");
    if (app.table().decision().kind != DecisionKind::TARGET ||
        !click_object(check, app, frames, bears, row_naming(p, bears, -1, ""), at + ": the bears targeted") ||
        !wait_decision(frames, s))
        return;
    // The Bolt on the stack: Pass by the prompt line's button.
    frames(2);
    p = pending(s);
    const int pass2 = find_row(p, category_with("PASS_PRIORITY", ""));
    check(app.table().decision().kind == DecisionKind::PRIORITY &&
              app.table().decision().prompt.rfind("Respond to your Lightning Bolt", 0) == 0 &&
              button_row(app, "Pass") == pass2 && pass2 >= 0,
        at + ": with the Bolt on the stack the prompt does not say what is responded to, or Pass is not offered (\"" +
            app.table().decision().prompt + "\")");
    if (!click_button(check, app, frames, "Pass", at + ": Pass") || !wait_decision(frames, s)) return;
    // A later plain priority decision: Space passes.
    const int later = drive_to(app, frames, category_with("PASS_PRIORITY", ""));
    p = pending(s);
    const int plain = p.ok() ? read_pass_facts(*p.view).pass_row : -1;
    check(later >= 0 && plain == later, at + ": no plain priority decision follows");
    if (plain < 0) return;
    const size_t index = p.index;
    press_space(app, frames);
    const int got = s.match()->decisions[index].reply;
    check(got == plain, at + ": Space replied " + std::to_string(got) + ", not the pass row " + std::to_string(plain));
    std::printf("rmwork: %s: a land played and a spell cast by clicks, its target clicked, Pass by button and Space\n",
        at.c_str());
}

// The land-and-Bolt game in Replay: the same lights, read-only, the row chosen marked.
static void check_land_bolt_replay(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table priority in Replay";
    if (!open_log_in_replay(check, app, frames, at)) return;
    const Match &m = *app.replay().match();
    long cast = -1, passed = -1;
    for (size_t i = 0; i < m.decisions.size(); ++i) {
        const DecisionInfo &di = m.decisions[i];
        if (di.decider != 0 || di.reply < 0) continue;
        std::string err;
        const auto f = m.frame(i, 0, &err);
        if (!f) continue;
        const agents::DecisionView v(m.stream(0), *f, 0);
        if (static_cast<size_t>(di.reply) >= v.n()) continue;
        const std::string &c = v.category(static_cast<size_t>(di.reply));
        if (cast < 0 && c == "CAST_SPELL" && v.description(static_cast<size_t>(di.reply)) == "Cast Lightning Bolt")
            cast = static_cast<long>(i);
        if (passed < 0 && c == "PASS_PRIORITY" && v.stack().rows() > 0) passed = static_cast<long>(i);
    }
    check(cast >= 0 && passed >= 0, at + ": the replay has no Bolt cast or pass with the stack full by seat A");
    if (cast < 0 || passed < 0) return;
    app.replay().set_view(0);
    app.replay().go_to(static_cast<size_t>(cast));
    frames(3);
    const uint64_t bolt = hand_card(app, "Lightning Bolt");
    const DecisionPick *bp = app.table().decision().pick_of_object(bolt);
    check(app.table().decision().kind == DecisionKind::PRIORITY && bp && app.table().decision().holds_chosen(*bp),
        at + ": seat A's view does not light the Bolt with its cast marked chosen");
    TRect r;
    if (card_rect(app, bolt, &r)) {
        mouse_click(screen_point(Mode::REPLAY, r), frames);
        check(app.links().selected().kind == LinkItem::Kind::OBJECT && app.links().selected().handle == bolt &&
                  !app.table().pick_popup_open() && app.table().take_answer() < 0 &&
                  app.replay().cursor() == static_cast<size_t>(cast),
            at + ": a click on the Bolt does more than select it");
    }
    app.replay().go_to(static_cast<size_t>(passed));
    frames(3);
    bool marked = false;
    for (const PromptButton &b : app.table().prompt_buttons())
        marked = marked || (b.label == "Pass" && b.row == app.table().decision().chosen_row && b.row >= 0);
    check(marked, at + ": the Pass chosen is not the prompt line's Pass");
    app.replay().go_to(static_cast<size_t>(cast));
    app.replay().set_view(1);
    frames(3);
    check(app.table().decision().kind == DecisionKind::NONE && app.table().decision().picks.empty(),
        at + ": seat B's view of seat A's priority decision lights its ways");
    mouse_to(ImVec2(-1000, -1000), frames);
    app.set_mode(Mode::PLAY);
    frames(2);
    std::printf("rmwork: %s: read-only lights, the cast and the pass marked chosen\n", at.c_str());
}

// Daze with Lightning Bolt on the stack: its two ways in a popup with their costs, the alternative chosen.
static void check_daze(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table priority: Daze's two ways";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Lightning Bolt,Daze", "--library-a", "Island", "--battlefield-a", "Mountain,Island,Island",
                "--hand-b", "Island", "--library-b", "Island"},
            at))
        return;
    const int bolt = drive_to(app, frames, category_with("CAST_SPELL", "Lightning Bolt"));
    if (bolt < 0 || !app.play_reply(bolt) || !wait_decision(frames, s)) {
        check(false, at + ": Lightning Bolt was not cast");
        return;
    }
    Pending p = pending(s);
    if (p.ok() && p.view->n() && p.view->category(0) == "SELECT_TARGET") {
        const int b = row_naming(p, 0, 1, "");
        if (b < 0 || !app.play_reply(b) || !wait_decision(frames, s)) {
            check(false, at + ": the Bolt's target was not chosen");
            return;
        }
    }
    const int alt = drive_to(app, frames, category_with("CAST_SPELL", "Daze (alternate cost)"));
    check(alt >= 0, at + ": no decision offers Daze's alternative cost");
    if (alt < 0) return;
    frames(2);
    p = pending(s);
    const int normal = exact_row(p, "Cast Daze");
    const uint64_t daze = hand_card(app, "Daze");
    const TableDecision &d = app.table().decision();
    const DecisionPick *dp = d.pick_of_object(daze);
    check(dp && dp->rows.size() == 2 && normal >= 0 && d.costs.size() == d.row_text.size() &&
              d.costs[static_cast<size_t>(normal)].pips == std::vector<std::string>{"1", "U"} &&
              d.costs[static_cast<size_t>(alt)].text == "return an Island you control to its owner's hand",
        at + ": Daze's two rows or their costs are not on its pick");
    check_ways_shown(check, app, at);
    if (!dp || !choose_in_popup(check, app, frames, daze, dp->rows, alt, at + ": the alternative cost") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    check(app.table().decision().kind == DecisionKind::CARDS && app.table().decision().cost_part == "alternative cost",
        at + ": Daze's Island return does not follow");
    std::printf("rmwork: %s: a popup of its two casts with their costs, the alternative chosen\n", at.c_str());
}

// Polluted Delta's ability by a click on it: {T}, pay 1 life, sacrifice it; its search follows.
static void check_fetchland(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table priority: a fetchland";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Mountain,Mountain,Mountain,Mountain,Mountain,Mountain,Mountain", "--library-a",
                "Island,Swamp,Mountain", "--battlefield-a", "Polluted Delta", "--hand-b", "Island", "--library-b",
                "Island"},
            at))
        return;
    const int row = drive_to(app, frames, category_with("ACTIVATE_ABILITY", "Polluted Delta"));
    check(row >= 0, at + ": no decision offers Polluted Delta's ability");
    if (row < 0) return;
    frames(2);
    const uint64_t delta = object_of(app, "Polluted Delta", 0);
    const TableDecision &d = app.table().decision();
    const DecisionPick *dp = d.pick_of_object(delta);
    const WayCost want{{"T"}, "pay 1 life, sacrifice it"};
    check(dp && dp->rows == std::vector<int>{row} && static_cast<size_t>(row) < d.costs.size() &&
              d.costs[static_cast<size_t>(row)].pips == want.pips &&
              d.costs[static_cast<size_t>(row)].text == want.text,
        at + ": the fetchland is not lit with its ability, or its cost is not {T}, pay 1 life, sacrifice it");
    check_ways_shown(check, app, at);
    if (!click_object(check, app, frames, delta, row, at + ": the ability activated") || !wait_decision(frames, s))
        return;
    // The ability on the stack, passed; its search at resolution.
    const int search = drive_to(app, frames, category_with("SEARCH_LIBRARY", ""));
    frames(2);
    check(search >= 0 && app.table().decision().kind == DecisionKind::CARDS && app.table().picker().active,
        at + ": the search does not follow in the card picker");
    std::printf("rmwork: %s: activated by a click, its cost read from the script\n", at.c_str());
}

// --no-auto-mana: Wasteland's mana and its ability in one popup; the ability chosen, its target clicked.
static void check_wasteland(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table priority: Wasteland (auto-mana off)";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, false,
            {"--hand-a", "Mountain", "--library-a", "Mountain", "--battlefield-a", "Wasteland", "--hand-b", "Island",
                "--library-b", "Island", "--battlefield-b", "Volcanic Island"},
            at))
        return;
    const int row = drive_to(app, frames, category_with("ACTIVATE_ABILITY", "Wasteland"));
    check(row >= 0, at + ": no decision offers Wasteland's ability");
    if (row < 0) return;
    frames(2);
    Pending p = pending(s);
    const uint64_t waste = object_of(app, "Wasteland", 0);
    const DecisionPick *wp = app.table().decision().pick_of_object(waste);
    bool mana = false;
    for (int r : wp ? wp->rows : std::vector<int>())
        mana = mana || p.view->category(static_cast<size_t>(r)) == "MANA_C";
    check(wp && wp->rows.size() == 2 && mana, at + ": Wasteland's pick does not hold its mana row and its ability");
    if (!wp || !choose_in_popup(check, app, frames, waste, wp->rows, row, at + ": the ability") ||
        !wait_decision(frames, s))
        return;
    frames(2);
    p = pending(s);
    const uint64_t volcanic = object_of(app, "Volcanic Island", 1);
    check(app.table().decision().kind == DecisionKind::TARGET, at + ": the target decision does not follow");
    if (!click_object(
            check, app, frames, volcanic, row_naming(p, volcanic, -1, ""), at + ": the Volcanic Island targeted"))
        return;
    std::printf(
        "rmwork: %s: a popup of its mana and its ability, the ability chosen, its target clicked\n", at.c_str());
}

// Cabal Therapy's flashback from the graveyard: a button of the prompt line.
static void check_flashback(const CheckFn &check, rmwork::App &app, const Frames &frames) {
    using namespace rmwork;
    const std::string at = "Table priority: a flashback";
    PlaySession &s = app.play();
    if (!start_sculpted(check, app, true,
            {"--hand-a", "Mountain", "--library-a", "Mountain", "--graveyard-a", "Cabal Therapy", "--battlefield-a",
                "Swamp,Grizzly Bears", "--hand-b", "Brainstorm,Ponder,Island", "--library-b", "Island"},
            at, 1))
        return;
    const int row = drive_to(app, frames, category_with("CAST_SPELL", "flashback"));
    check(row >= 0, at + ": no decision offers Cabal Therapy's flashback");
    if (row < 0) return;
    frames(2);
    const TableDecision &d = app.table().decision();
    const std::string label = static_cast<size_t>(row) < d.row_text.size() ? d.row_text[static_cast<size_t>(row)] : "";
    check(d.kind == DecisionKind::PRIORITY && button_row(app, label) == row &&
              static_cast<size_t>(row) < d.costs.size() &&
              d.costs[static_cast<size_t>(row)].text == "sacrifice a creature",
        at + ": the flashback is not a prompt-line button, or its cost is not \"sacrifice a creature\"");
    if (!click_button(check, app, frames, "flashback", at + ": the flashback") || !wait_decision(frames, s)) return;
    std::printf("rmwork: %s: cast from the graveyard by the prompt line's button\n", at.c_str());
}

namespace rmwork {

void check_priority_units(const CheckFn &check) {
    const auto same = [](const WayCost &c, const std::vector<std::string> &pips, const std::string &text) {
        return c.pips == pips && c.text == text;
    };
    check(same(ability_cost("T PayLife<1> Sac<1/CARDNAME>"), {"T"}, "pay 1 life, sacrifice it"), "a fetchland's cost");
    check(same(ability_cost("2 Sac<1/CARDNAME/this token>"), {"2"}, "sacrifice this token"), "a Clue's cost");
    check(same(ability_cost("1 R"), {"1", "R"}, ""), "an equip cost");
    check(same(ability_cost("SubCounter<3/LOYALTY>"), {},
              "\xE2\x88\x92"
              "3 loyalty"),
        "a loyalty cost");
    check(same(ability_cost("Sac<1/Land>"), {}, "sacrifice a land"), "a cycling cost");
    check(same(ability_cost("Sac<2/Mountain>"), {}, "sacrifice two Mountains"), "two of a type");
    const std::string uro =
        "When Uro enters, sacrifice it unless it escaped.\nEscape\xE2\x80\x94{G}{G}{U}{U}, Exile five other cards from "
        "your graveyard. (You may cast this card from your graveyard for its escape cost.)";
    check(same(keyword_cost(uro, "Escape"), {"G", "G", "U", "U"}, "exile five other cards from your graveyard"),
        "Uro's escape");
    check(same(keyword_cost("Flashback\xE2\x80\x94Sacrifice a creature. (You may cast …)", "Flashback"), {},
              "sacrifice a creature"),
        "a flashback without mana");
    check(same(keyword_cost("Deal 2 damage.\nFlashback {R} (You may cast …)", "Flashback"), {"R"}, ""),
        "a flashback for mana");
    check(same(keyword_cost("Suspend 4\xE2\x80\x94{1}{U} (Rather than …)", "Suspend"), {"1", "U"},
              "with 4 time counters"),
        "suspend");
    check(keyword_cost("Flash\nLifelink", "Flashback").empty(), "no line of the keyword");
    check(alternative_cost("You may return an Island you control to its owner's hand rather than pay this spell's mana "
                           "cost.\nCounter target spell unless its controller pays {1}.") ==
              "return an Island you control to its owner's hand",
        "Daze's alternative cost");
    check(alternative_cost("If you control a Forest, you may pay 1 life and exile a blue card from your hand rather "
                           "than pay this spell's mana cost.") == "pay 1 life and exile a blue card from your hand",
        "an alternative cost after a condition");
    check(way_label("Cast Lightning Bolt", WayCost{{"R"}, ""}) == "Cast Lightning Bolt {R}", "a cast's label");
    check(way_label("Activate Polluted Delta (ChangeZone)", WayCost{{"T"}, "pay 1 life, sacrifice it"}) ==
              "Activate Polluted Delta (ChangeZone) \xE2\x80\x94 {T}, pay 1 life, sacrifice it",
        "an ability's label");
    check(way_label("Play Mountain", WayCost()) == "Play Mountain", "a land's label");
}

void check_table_priority(const CheckFn &check, App &app, const std::function<void(int)> &frames) {
    check_land_bolt(check, app, frames);
    check_land_bolt_replay(check, app, frames);
    check_daze(check, app, frames);
    check_fetchland(check, app, frames);
    check_wasteland(check, app, frames);
    check_flashback(check, app, frames);
}

}  // namespace rmwork
