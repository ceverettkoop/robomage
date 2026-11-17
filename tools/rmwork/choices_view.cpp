#include "choices_view.h"

#include <cstdio>

#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

static std::string number(double v, int decimals);

static std::string number(double v, int decimals) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.*f", decimals, v);
    return buf;
}

namespace rmwork {

const featurize::PolicyEvalRecord *ChoicesView::shown_record(const ToolData &data) {
    const EpisodeStepInfo *e = data.frame().visible_episode();
    if (!e) return nullptr;
    const featurize::PolicyEvalRecord *r = find_policy(*e, data.shown_policy());
    if (!r) r = find_policy(*e, -1);
    if (!r || r->log_pi.size() != data.models().menu.rows.size()) return nullptr;
    return r;
}

std::vector<double> ChoicesView::shown_pi(const ToolData &data) {
    const featurize::PolicyEvalRecord *r = shown_record(data);
    return r ? policy_pi(*r) : std::vector<double>();
}

int ChoicesView::take_submitted() {
    const int r = submitted_;
    submitted_ = -1;
    return r;
}

void ChoicesView::draw(ToolData &data, LinkState *links, bool play) {
    const float width = 46.0f * ImGui::CalcTextSize("M").x;
    if (!data.ok()) {
        centered_text_block("Frame not readable", data.error().c_str(), width);
        return;
    }
    const DecisionFrame &f = data.frame();
    const MenuModel &menu = data.models().menu;
    if (data.generation() != generation_) {
        generation_ = data.generation();
        highlight_ = menu.chosen;
        scroll_to_ = true;
    }
    if (!menu.shown) {
        const std::string who = std::string(seat_name(menu.decider)) + " decides";
        const std::string body = std::string("The ") + view_name(f.view) + " view shows neither " +
                                 seat_name(menu.decider) + "'s choices nor its evaluations. Switch to " +
                                 (menu.decider == 0 ? "A" : "B") + " or Omni to see them.";
        centered_text_block(who.c_str(), body.c_str(), width);
        return;
    }
    const size_t n = menu.rows.size();
    if (force_hover_ >= 0 && links && static_cast<size_t>(force_hover_) < n) links->hover_menu_row(force_hover_);
    const std::vector<double> pi = shown_pi(data);
    const EpisodeStepInfo *e = f.visible_episode();
    const bool mu = e && e->has_mu && e->mu.size() == n;
    // ↑ / ↓ move the highlight, Enter submits it, while the tool has focus.
    if (ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) && !ImGui::GetIO().WantTextInput && n) {
        if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) {
            highlight_ = highlight_ < 0 ? 0 : std::min(static_cast<int>(n) - 1, highlight_ + 1);
            scroll_to_ = true;
        }
        if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) {
            highlight_ = highlight_ < 0 ? 0 : std::max(0, highlight_ - 1);
            scroll_to_ = true;
        }
        if (play && highlight_ >= 0 &&
            (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)))
            submitted_ = highlight_;
    }
    const int columns = 4 + (mu ? 1 : 0);
    const float foot = ImGui::GetTextLineHeightWithSpacing() + 4;
    if (begin_grid("##choices", columns, ImVec2(0, -foot), true)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(" ", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("\xE2\x96\xBA").x + 4);
        ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("000").x);
        ImGui::TableSetupColumn("Choice", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("\xCF\x80", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0.00").x + 4);
        if (mu)
            ImGui::TableSetupColumn("\xCE\xBC", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0.000").x + 4);
        ImGui::TableHeadersRow();
        for (size_t r = 0; r < n; ++r) {
            const int row = static_cast<int>(r);
            ImGui::TableNextRow(ImGuiTableRowFlags_None, ImGui::GetTextLineHeight() + 6);
            if (links && links->row_linked(row)) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, linked_row_color());
            ImGui::TableNextColumn();
            ImGui::PushID(row);
            const bool selected = row == highlight_;
            if (ImGui::Selectable("##row", selected,
                    ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap,
                    ImVec2(0, ImGui::GetTextLineHeight() + 4))) {
                highlight_ = row;
                if (links) {
                    LinkItem item;
                    item.kind = LinkItem::Kind::MENU_ROW;
                    item.row = row;
                    links->select(item);
                }
                if (play) submitted_ = row;
            }
            if (ImGui::IsItemHovered() && links) links->hover_menu_row(row);
            if (selected && scroll_to_) {
                ImGui::SetScrollHereY(0.5f);
                scroll_to_ = false;
            }
            ImGui::SameLine(0, 0);
            if (!play && row == menu.chosen) {
                ImGui::PushStyleColor(ImGuiCol_Text, tok().ACCENT);
                ImGui::TextUnformatted("\xE2\x96\xBA");
                ImGui::PopStyleColor();
            }
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%d", row);
            ImGui::TableNextColumn();
            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::TextUnformatted(menu.rows[r].text.c_str());
            if (!pi.empty()) {
                const float w = ImGui::GetContentRegionAvail().x;
                const float y = p.y + ImGui::GetTextLineHeight() + 1;
                ImGui::GetWindowDrawList()->AddRectFilled(
                    ImVec2(p.x, y), ImVec2(p.x + std::max(1.0f, w * static_cast<float>(pi[r])), y + 3), tok().ACCENT);
            }
            ImGui::TableNextColumn();
            if (!pi.empty()) {
                ImGui::TextUnformatted(number(pi[r], 2).c_str());
            } else {
                ImGui::TextDisabled("\xE2\x80\x94");
            }
            if (mu) {
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(number(e->mu[r], 3).c_str());
            }
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    std::string note;
    if (play) {
        note = "Click a row, or \xE2\x86\x91 / \xE2\x86\x93 and Enter, to choose \xC2\xB7 F2 / F6 pass";
    } else {
        note = menu.chosen >= 0 ? std::string("\xE2\x96\xBA the row ") + seat_name(menu.decider) + " chose" : "";
        if (const featurize::PolicyEvalRecord *r = shown_record(data)) {
            note +=
                std::string(note.empty() ? "" : " \xC2\xB7 ") + "bars: \xCF\x80 of " + policy_label(r->id, r->version);
        } else if (!e) {
            note += std::string(note.empty() ? "" : " \xC2\xB7 ") + "no evaluations for this decision";
        }
    }
    ImGui::TextDisabled("%s", note.c_str());
}

}  // namespace rmwork
