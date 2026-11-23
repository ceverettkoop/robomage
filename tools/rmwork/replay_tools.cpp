#include "replay_tools.h"

#include <map>

namespace rmwork {

void ReplayTools::begin_frame(ReplaySession &s, TableView &table, LinkState &links) {
    if (!data_.refresh(s, table)) return;
    const FrameModels &m = data_.models();
    links.set_menu(m.table.menu);
    std::map<uint64_t, std::vector<StackTarget>> targets;
    for (const TableStackItem &it : m.table.stack) targets[it.handle] = it.targets;
    links.set_stack_targets(targets);
}

void ReplayTools::draw(Tool t, ReplaySession &s, LinkState *links) {
    int seat = 0, zone = 0;
    if (zone_tool(t, &seat, &zone)) {
        players_.draw_zone_tool(data_, seat, static_cast<ZoneKind>(zone), links, magnify_);
        return;
    }
    switch (t) {
        case Tool::PLAYERS:
            players_.draw(data_, s, links);
            return;
        case Tool::CHOICES:
            choices_.draw(data_, links, choosing_);
            return;
        case Tool::INSPECTOR:
            inspector_.draw(data_, links);
            return;
        case Tool::TREE:
            tree_.draw(data_, links);
            return;
        case Tool::SWINGS:
            swings_.draw(data_, s);
            return;
        case Tool::GRAPH:
            graph_.draw(data_, s);
            return;
        case Tool::MODEL:
            model_.draw(data_, s);
            return;
        case Tool::SIDEBOARD:
            sideboard_.draw(data_, s, choosing_, magnify_);
            return;
        default:  // the Table, the Game Log and the zone tools: not here
            return;
    }
}

void ReplayTools::draw_windows(LinkState *links, ImGuiKeyChord magnify) {
    players_.draw_zone_windows(data_, links, magnify);
}

long ReplayTools::next_swing_target(const ReplaySession &s) const {
    if (!s.loaded()) return -1;
    return rmwork::next_swing_target(data_.swings(s.view(), SwingsView::ROWS), s.cursor());
}

bool ReplayTools::next_swing(ReplaySession &s) {
    const long t = next_swing_target(s);
    if (t < 0) return false;
    s.go_to(static_cast<size_t>(t));
    return true;
}

std::string ReplayTools::status_value() const {
    if (!data_.ok()) return "V \xE2\x80\x94";
    const DecisionFrame &f = data_.frame();
    const EpisodeStepInfo *e = f.visible_episode();
    const featurize::PolicyEvalRecord *r = e ? find_policy(*e, data_.shown_policy()) : nullptr;
    if (!r) return "V \xE2\x80\x94";
    return "V " + signed_text(r->value) + " (" + (e->seat == 0 ? "A" : "B") + ")";
}

}  // namespace rmwork
