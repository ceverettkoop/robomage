#include "state_tree_view.h"

#include "imgui_internal.h"
#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

namespace rmwork {

StateTreeView::StateTreeView()
    : open_({"player/A", "player/B", "player/A/battlefield", "player/B/battlefield", "player/A/hand", "player/B/hand",
          "stack"}) {}

void StateTreeView::refresh(const ToolData &data) {
    if (data.generation() == generation_ && model_.ready() == data.ok()) return;
    generation_ = data.generation();
    if (data.ok()) {
        model_.build(data.frame(), data.models().table);
    } else {
        model_ = StateTreeModel();
    }
    rows_dirty_ = true;
}

void StateTreeView::expand_overview() {
    for (const char *k : {"player/A", "player/B", "player/A/battlefield", "player/B/battlefield", "player/A/hand",
             "player/B/hand", "player/A/graveyard", "player/B/graveyard", "stack", "tables", "tables/object"})
        open_.insert(k);
    rows_dirty_ = true;
}

const std::vector<TreeRow> &StateTreeView::rows() {
    if (rows_dirty_) {
        rows_ = model_.ready() ? model_.flatten(open_) : std::vector<TreeRow>();
        rows_dirty_ = false;
    }
    return rows_;
}

void StateTreeView::draw(ToolData &data, LinkState *links) {
    refresh(data);
    if (!data.ok()) {
        centered_text_block("Frame not readable", data.error().c_str(), 46.0f * ImGui::CalcTextSize("M").x);
        return;
    }
    const std::vector<TreeRow> &shown = rows();
    std::string toggle;
    if (begin_grid("##tree", 5, ImVec2(0, 0), true)) {
        const float em = ImGui::CalcTextSize("M").x;
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableSetupColumn("Col", ImGuiTableColumnFlags_WidthFixed, em * 3.5f);
        ImGui::TableSetupColumn("Zone", ImGuiTableColumnFlags_WidthFixed, em * 10);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableHeadersRow();
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(shown.size()));
        const float indent = em * 1.6f;
        while (clipper.Step()) {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                const TreeRow &r = shown[static_cast<size_t>(i)];
                ImGui::TableNextRow();
                const bool linked = links && ((r.handle && links->object_linked(r.handle)) ||
                                                 (r.kind == TreeRow::Kind::PLAYER && links->player_linked(r.seat)));
                if (linked) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, linked_row_color());
                ImGui::TableNextColumn();
                ImGui::PushID(i);
                if (ImGui::Selectable(
                        "##node", false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap) &&
                    r.expandable)
                    toggle = r.key;
                if (ImGui::IsItemHovered() && links) {
                    if (r.handle) links->hover_object(r.handle);
                    if (r.kind == TreeRow::Kind::PLAYER) links->hover_player(r.seat);
                }
                ImGui::SameLine(0, 0);
                const ImVec2 p = ImGui::GetCursorScreenPos();
                const float x = p.x + indent * static_cast<float>(r.depth);
                if (r.expandable)
                    ImGui::RenderArrow(ImGui::GetWindowDrawList(), ImVec2(x, p.y + ImGui::GetTextLineHeight() * 0.15f),
                        tok().DIM, open_.count(r.key) ? ImGuiDir_Down : ImGuiDir_Right, 0.7f);
                ImGui::SetCursorScreenPos(ImVec2(x + em * 1.5f, p.y));
                ImGui::TextUnformatted(r.name.c_str());
                ImGui::TableNextColumn();
                if (r.swatch) card_swatch(r.colors, r.land, em * 2.6f);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.zone.c_str());
                ImGui::TableNextColumn();
                ImGui::PushStyleColor(ImGuiCol_Text, r.card_type ? tok().ACCENT : tok().GOOD);
                ImGui::TextUnformatted(r.type.c_str());
                ImGui::PopStyleColor();
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(r.value.c_str());
                ImGui::PopID();
            }
        }
        ImGui::EndTable();
    }
    if (!toggle.empty()) {
        if (open_.count(toggle)) {
            open_.erase(toggle);
        } else {
            open_.insert(toggle);
        }
        rows_dirty_ = true;
    }
}

}  // namespace rmwork
