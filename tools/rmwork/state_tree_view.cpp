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
    if (data.generation() != generation_) {
        followed_ = own_hover_ = LinkItem();
        scroll_key_.clear();
    }
    generation_ = data.generation();
    if (data.ok()) {
        model_.build(data.frame(), data.models().table);
    } else {
        model_ = StateTreeModel();
    }
    rows_dirty_ = true;
}

// The keys down to the row of what `item` links: an object, a player, a menu row's source object,
// else its target, else a player it names.
std::vector<std::string> StateTreeView::link_path(const LinkItem &item, const std::vector<MenuLink> &menu) const {
    switch (item.kind) {
        case LinkItem::Kind::NONE:
            return {};
        case LinkItem::Kind::OBJECT:
            return model_.object_path(item.handle);
        case LinkItem::Kind::PLAYER:
            return {StateTreeModel::player_key(item.seat)};
        case LinkItem::Kind::MENU_ROW: {
            if (item.row < 0 || static_cast<size_t>(item.row) >= menu.size()) return {};
            const MenuLink &m = menu[static_cast<size_t>(item.row)];
            std::vector<std::string> path = model_.object_path(m.source);
            if (path.empty()) path = model_.object_path(m.target);
            if (path.empty() && m.source_seat >= 0) path = {StateTreeModel::player_key(m.source_seat)};
            if (path.empty() && m.target_seat >= 0) path = {StateTreeModel::player_key(m.target_seat)};
            return path;
        }
    }
    return {};
}

// Whether the last item drawn is wholly inside the tree's scrolled area (below the frozen header).
bool StateTreeView::row_on_screen() {
    const ImGuiWindow *w = ImGui::GetCurrentWindow();
    const float top = w->InnerRect.Min.y + w->DecoInnerSizeY1;
    return ImGui::GetItemRectMin().y >= top - 1 && ImGui::GetItemRectMax().y <= w->InnerRect.Max.y + 1;
}

void StateTreeView::follow_links(const LinkState &links) {
    const LinkItem &item = links.active();
    if (item == followed_) return;
    followed_ = item;
    if (!model_.ready() || item == own_hover_) return;
    const std::vector<std::string> path = link_path(item, links.menu());
    if (path.empty()) return;
    for (size_t i = 0; i + 1 < path.size(); ++i)
        if (open_.insert(path[i]).second) rows_dirty_ = true;
    scroll_key_ = path.back();
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
    if (links) follow_links(*links);
    own_hover_ = LinkItem();
    const std::vector<TreeRow> &shown = rows();
    // The row to scroll to, and the last one revealed (whether it is on screen).
    int scroll_row = -1, revealed_row = -1;
    if (!scroll_key_.empty() || !revealed_key_.empty())
        for (size_t i = 0; i < shown.size(); ++i) {
            if (!scroll_key_.empty() && shown[i].key == scroll_key_) scroll_row = static_cast<int>(i);
            if (!revealed_key_.empty() && shown[i].key == revealed_key_) revealed_row = static_cast<int>(i);
        }
    if (!scroll_key_.empty()) {
        revealed_key_ = scroll_key_;
        revealed_row = scroll_row;
        scroll_key_.clear();
    }
    revealed_on_screen_ = false;
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
        if (scroll_to_end_) {
            ImGui::SetScrollY(ImGui::GetScrollMaxY());
            scroll_to_end_ = false;
        }
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(shown.size()));
        if (scroll_row >= 0) clipper.IncludeItemByIndex(scroll_row);
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
                    if (r.handle) {
                        links->hover_object(r.handle);
                        own_hover_.kind = LinkItem::Kind::OBJECT;
                        own_hover_.handle = r.handle;
                    }
                    if (r.kind == TreeRow::Kind::PLAYER) {
                        links->hover_player(r.seat);
                        own_hover_.kind = LinkItem::Kind::PLAYER;
                        own_hover_.seat = r.seat;
                    }
                }
                if (i == scroll_row) ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleCenterY);
                if (i == revealed_row) revealed_on_screen_ = row_on_screen();
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
