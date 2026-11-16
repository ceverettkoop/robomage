#include "game_log.h"

#include <algorithm>

#include "imgui.h"
#include "theme.h"

static size_t first_line_at_or_after(const std::vector<rmwork::LogLine> &lines, int decision);
static ImU32 line_color(const rmwork::LogLine &l, int cursor);

static size_t first_line_at_or_after(const std::vector<rmwork::LogLine> &lines, int decision) {
    return static_cast<size_t>(std::lower_bound(lines.begin(), lines.end(), decision,
                                   [](const rmwork::LogLine &l, int d) { return l.decision < d; }) -
                               lines.begin());
}

// The current decision's lines and the past in the text colour, the future dimmed; headers in the
// accent colour and the chosen rows in their seat's series colour while not in the future.
static ImU32 line_color(const rmwork::LogLine &l, int cursor) {
    using rmwork::tok;
    if (l.decision > cursor) return tok().DIM;
    if (l.kind == rmwork::LogLine::Kind::HEADER) return tok().ACCENT;
    if (l.kind == rmwork::LogLine::Kind::CHOICE) return l.seat == 1 ? tok().S_B : tok().S_A;
    return tok().TEXT;
}

namespace rmwork {

const std::vector<LogLine> &GameLogView::lines(const Match &m, int view) {
    return m.logs[static_cast<size_t>(view < 0 || view >= VIEW_COUNT ? VIEW_OMNI : view)];
}

std::string GameLogView::caption(const Match &m, int view) {
    if (view == VIEW_OMNI)
        return std::string("Narrative (omniscient)") +
               (m.narrative_mapped ? ""
                                   : ": its decision markers do not match the decisions, so lines may sit at the "
                                     "wrong decision");
    return std::string(view == VIEW_A ? "Seat A" : "Seat B") + "'s own log: the events in its frames";
}

void GameLogView::draw(ReplaySession &s) {
    const Match &m = *s.match();
    const std::vector<LogLine> &ls = lines(m, s.view());
    const int cursor = static_cast<int>(s.cursor());
    ImGui::PushStyleColor(ImGuiCol_Text, m.narrative_mapped || s.view() != VIEW_OMNI ? tok().DIM : tok().BAD);
    ImGui::TextUnformatted(caption(m, s.view()).c_str());
    ImGui::PopStyleColor();
    ImGui::Separator();
    ImGui::BeginChild("##log_lines", ImVec2(0, 0), ImGuiChildFlags_None, ImGuiWindowFlags_HorizontalScrollbar);
    const float lh = ImGui::GetTextLineHeightWithSpacing();
    if (s.generation() != drawn_generation_) {
        drawn_generation_ = s.generation();
        size_t first = first_line_at_or_after(ls, cursor);
        size_t end = first_line_at_or_after(ls, cursor + 1);
        if (first == end && first > 0) first = end - 1;
        if (end == first) end = first + 1;
        const float top = static_cast<float>(first) * lh, bottom = static_cast<float>(end) * lh;
        const float y = ImGui::GetScrollY(), h = ImGui::GetWindowHeight();
        if (top < y || bottom > y + h) ImGui::SetScrollY(std::max(0.0f, top - h * 0.25f));
    }
    if (ls.empty()) ImGui::TextDisabled("This viewer's log is empty.");
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(ls.size()), lh);
    int go = -1;
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const LogLine &l = ls[static_cast<size_t>(i)];
            const bool current = l.decision == cursor;
            ImGui::PushID(i);
            if (l.decision > cursor && (i == 0 || ls[static_cast<size_t>(i) - 1].decision <= cursor)) {
                // The boundary between the current decision and what comes after it.
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const float w = ImGui::GetContentRegionAvail().x;
                ImGui::GetWindowDrawList()->AddLine(ImVec2(p.x, p.y - 1), ImVec2(p.x + w, p.y - 1), tok().ACCENT);
            }
            ImGui::PushStyleColor(ImGuiCol_Text, line_color(l, cursor));
            ImGui::PushStyleColor(ImGuiCol_Header, tok().SEL);
            const std::string text = l.kind == LogLine::Kind::EVENT ? "  " + l.text : l.text;
            if (ImGui::Selectable(text.c_str(), current, ImGuiSelectableFlags_None)) go = l.decision;
            ImGui::PopStyleColor(2);
            if (ImGui::IsItemHovered() && l.decision != cursor)
                ImGui::SetTooltip("Decision %d: click to go there", l.decision + 1);
            ImGui::PopID();
        }
    }
    ImGui::EndChild();
    if (go >= 0) s.go_to(static_cast<size_t>(go));
}

}  // namespace rmwork
