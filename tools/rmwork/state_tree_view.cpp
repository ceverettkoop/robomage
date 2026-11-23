#include "state_tree_view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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
    table_ids_.clear();
    if (data.ok()) {
        model_.build(data.frame(), data.models().table);
        for (const rmstate::DecodedTable &t : data.frame().frame->tables) table_ids_.push_back(t.id);
    } else {
        model_ = StateTreeModel();
    }
    rows_dirty_ = true;
}

void StateTreeView::set_impact(const AttrTints *t) {
    const bool had = impact_.on;
    impact_ = t ? *t : AttrTints();
    if (sort_impact_ && (had || impact_.on)) rows_dirty_ = true;
}

void StateTreeView::set_sort_by_impact(bool on) {
    if (on == sort_impact_) return;
    sort_impact_ = on;
    rows_dirty_ = true;
}

bool StateTreeView::impact_of(const TreeRow &r, double *v) const {
    if (!impact_.on) return false;
    if (r.kind == TreeRow::Kind::PLAYER && r.seat >= 0 && r.seat < 2 &&
        impact_.has_player[static_cast<size_t>(r.seat)]) {
        *v = impact_.players[static_cast<size_t>(r.seat)];
        return true;
    }
    if (r.handle && (r.kind == TreeRow::Kind::OBJECT || r.kind == TreeRow::Kind::STACK_ITEM)) {
        auto it = impact_.objects.find(r.handle);
        if (it != impact_.objects.end()) {
            *v = it->second;
            return true;
        }
    }
    if (!impact_.same_frame || r.table < 0 || static_cast<size_t>(r.table) >= table_ids_.size()) return false;
    const int id = table_ids_[static_cast<size_t>(r.table)];
    if (r.kind == TreeRow::Kind::ROW || r.kind == TreeRow::Kind::OBJECT) {
        auto it = impact_.rows.find({id, r.row});
        if (it != impact_.rows.end()) {
            *v = it->second;
            return true;
        }
    }
    if (r.kind == TreeRow::Kind::FIELD) {
        auto it = impact_.cells.find(std::make_tuple(id, static_cast<int64_t>(r.row), r.name));
        if (it != impact_.cells.end()) {
            *v = it->second;
            return true;
        }
    }
    return false;
}

// Each node's children (with their subtrees) ordered by |impact|, those with none after, stably.
void StateTreeView::sort_rows() {
    struct Sorter {
            const StateTreeView *view;
            std::vector<TreeRow> *rows;
            // Sorts the children at depth `depth` in [begin, end); returns the rows in their new order.
            std::vector<TreeRow> run(size_t begin, size_t end, int depth) {
                std::vector<std::pair<std::pair<int, double>, std::vector<TreeRow>>> chunks;
                size_t i = begin;
                while (i < end) {
                    size_t j = i + 1;
                    while (j < end && (*rows)[j].depth > depth) ++j;
                    std::vector<TreeRow> chunk = {(*rows)[i]};
                    if (j > i + 1) {
                        const std::vector<TreeRow> kids = run(i + 1, j, depth + 1);
                        chunk.insert(chunk.end(), kids.begin(), kids.end());
                    }
                    double v = 0;
                    const bool has = view->impact_of((*rows)[i], &v);
                    chunks.push_back({{has ? 0 : 1, -std::fabs(v)}, std::move(chunk)});
                    i = j;
                }
                std::stable_sort(chunks.begin(), chunks.end(),
                    [](const std::pair<std::pair<int, double>, std::vector<TreeRow>> &a,
                        const std::pair<std::pair<int, double>, std::vector<TreeRow>> &b) {
                        return a.first < b.first;
                    });
                std::vector<TreeRow> out;
                for (auto &c : chunks) out.insert(out.end(), c.second.begin(), c.second.end());
                return out;
            }
    };
    if (rows_.empty()) return;
    Sorter s{this, &rows_};
    std::vector<TreeRow> sorted;
    size_t i = 0;
    // The roots keep their order; their children are sorted.
    while (i < rows_.size()) {
        size_t j = i + 1;
        while (j < rows_.size() && rows_[j].depth > rows_[i].depth) ++j;
        sorted.push_back(rows_[i]);
        if (j > i + 1) {
            const std::vector<TreeRow> kids = s.run(i + 1, j, rows_[i].depth + 1);
            sorted.insert(sorted.end(), kids.begin(), kids.end());
        }
        i = j;
    }
    rows_ = std::move(sorted);
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
        if (sort_impact_ && impact_.on) sort_rows();
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
    if (impact_.on) {
        bool sort = sort_impact_;
        if (ImGui::Checkbox("Sort children by impact", &sort)) set_sort_by_impact(sort);
        ImGui::SameLine();
        ImGui::TextDisabled("Impact: %s", impact_.legend.c_str());
    }
    const int columns = impact_.on ? 6 : 5;
    if (begin_grid("##tree", columns, ImVec2(0, 0), true)) {
        const float em = ImGui::CalcTextSize("M").x;
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        ImGui::TableSetupColumn("Col", ImGuiTableColumnFlags_WidthFixed, em * 3.5f);
        ImGui::TableSetupColumn("Zone", ImGuiTableColumnFlags_WidthFixed, em * 10);
        ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 2.4f);
        ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch, 3.0f);
        if (impact_.on) ImGui::TableSetupColumn("Impact", ImGuiTableColumnFlags_WidthFixed, em * 8);
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
                if (impact_.on) {
                    ImGui::TableNextColumn();
                    double v = 0;
                    const bool has = impact_of(r, &v);
                    heat_swatch(has, v, r.kind == TreeRow::Kind::FIELD ? impact_.cell_scale : impact_.scale,
                        attr_target_signed(impact_.target), em * 2.2f);
                    if (has) {
                        ImGui::SameLine();
                        char buf[24];
                        std::snprintf(buf, sizeof buf, attr_target_signed(impact_.target) ? "%+.3f" : "%.3f", v);
                        ImGui::TextUnformatted(buf);
                    }
                }
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
