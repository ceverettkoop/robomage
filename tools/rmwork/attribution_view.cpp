#include "attribution_view.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "imgui_internal.h"
#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

namespace {

// The metric columns, in order: their header and the target (or -1) they show.
struct MetricColumn {
        const char *name;
        int target;  // an AttrTarget, -1 for kl / flip
};
const MetricColumn METRICS[] = {{"dv", 0}, {"tv", 1}, {"kl", -1}, {"flip", -1}, {"dlogp", 2}, {"dmargin", 3}};
const int METRIC_COUNT = 6;

}  // namespace

static double metric_of(const rmwork::AttrMetrics &m, int metric);
static std::string number(double v, bool sign);
static std::string base_name(const std::string &path);

// The last component of a path.
static std::string base_name(const std::string &path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static double metric_of(const rmwork::AttrMetrics &m, int metric) {
    switch (metric) {
        case 0:
            return m.dv;
        case 1:
            return m.tv;
        case 2:
            return m.kl;
        case 3:
            return m.flip;
        case 4:
            return m.dlogp;
        case 5:
            return m.dmargin;
        default:
            return 0;
    }
}

static std::string number(double v, bool sign) {
    char buf[32];
    std::snprintf(buf, sizeof buf, sign ? "%+.3f" : "%.3f", v);
    return buf;
}

namespace rmwork {

bool AttributionView::take_explain() {
    const bool e = explain_;
    explain_ = false;
    return e;
}

std::string AttributionView::label(const Attribution &a, const AttrRow &r) const {
    const featurize::Unit *u = a.unit(r);
    if (!u) return r.key;
    return tensors_ ? u->tensors : u->label;
}

std::vector<int> AttributionView::sorted(const Attribution &a, const std::vector<int> &rows, int metric) const {
    std::vector<int> out = rows;
    std::stable_sort(out.begin(), out.end(), [&a, metric](int x, int y) {
        const AttrRow &rx = a.rows()[static_cast<size_t>(x)], &ry = a.rows()[static_cast<size_t>(y)];
        if (rx.m.valid != ry.m.valid) return rx.m.valid;
        return std::fabs(metric_of(rx.m, metric)) > std::fabs(metric_of(ry.m, metric));
    });
    return out;
}

void AttributionView::draw(const AttributionContext &c) {
    const float width = 46.0f * ImGui::CalcTextSize("M").x;
    if (!c.att || !c.data || !c.data->ok()) {
        centered_text_block(
            "No decision", "Open a replay or start a game: Attribution explains a decision's V and \xCF\x80.", width);
        return;
    }
    draw_header(c);
    const Attribution &a = *c.att;
    if (!c.current || !a.has_key()) {
        const char *title = c.waiting ? "Waiting for Model's Read" : "Not attributed";
        std::string body;
        if (c.no_menu) {
            title = "No menu in this frame";
            body = std::string("At this decision ") + seat_name(c.seat) +
                   "'s frame has no menu (the other seat decides), and the inference server evaluates only a "
                   "decision with menu rows. Switch to the deciding seat's view or Omni to attribute it.";
        } else if (c.policies.empty()) {
            body =
                "Settings \xE2\x86\x92 Models lists no served policy (a numeric id): Attribution evaluates the "
                "decision through the inference server.";
        } else if (c.waiting) {
            body = "Attribution runs once Model's Read has evaluated this decision.";
        } else if (c.mode == Mode::PLAY) {
            body = "Press Explain (or its key) to attribute this decision, or tick Auto.";
        } else {
            body = c.server ? "Press Explain to attribute this decision."
                            : "No inference server listens at the configured socket, and the cache has nothing for "
                              "this decision. Start one (Settings \xE2\x86\x92 Models).";
        }
        centered_text_block(title, body.c_str(), width);
        shown_.clear();
        return;
    }
    if (!a.base() && !a.running() && a.state() != Attribution::State::FAILED) {
        centered_text_block("Not attributed",
            "No inference server listens at the configured socket, and the cache has nothing for this decision. "
            "Start one (Settings \xE2\x86\x92 Models), then Explain.",
            width);
        shown_.clear();
        return;
    }
    if (a.state() == Attribution::State::FAILED && a.rows().empty()) {
        centered_text_block(a.no_server() ? "No inference server" : "Attribution failed", a.error().c_str(), width);
        shown_.clear();
        return;
    }
    draw_rows(c);
}

void AttributionView::draw_header(const AttributionContext &c) {
    const Attribution &a = *c.att;
    const DecisionFrame &f = c.data->frame();
    std::string head = "Decision " + std::to_string(f.index + 1);
    if (c.seat >= 0) head += std::string(" \xC2\xB7 ") + seat_name(c.seat) + "'s frame";
    if (c.current && a.has_key() && a.frame() && !a.base()) head += " \xC2\xB7 evaluating";
    if (c.current && a.base() && a.base()->log_pi.empty()) head += " \xC2\xB7 no menu: V only";
    if (c.current && a.from_cache()) head += " \xC2\xB7 from the cache";
    ImGui::TextUnformatted(head.c_str());
    ImGui::SameLine();
    if (ImGui::SmallButton("Explain")) explain_ = true;
    if (c.mode == Mode::PLAY) {
        ImGui::SameLine();
        ImGui::Checkbox("Auto", &auto_);
    }
    if (c.layer) {
        ImGui::SameLine();
        ImGui::Checkbox("Table layer", c.layer);
    }
    if (c.current && a.running()) {
        ImGui::SameLine();
        const double fr = a.fraction();
        ImGui::ProgressBar(
            fr < 0 ? 0.0f : static_cast<float>(fr), ImVec2(ImGui::CalcTextSize("M").x * 8, 0), a.phase().c_str());
        ImGui::SameLine();
        if (ImGui::SmallButton("Cancel")) c.att->cancel();
    }
    // The choices.
    const float em = ImGui::CalcTextSize("M").x;
    ImGui::SetNextItemWidth(em * 12);
    std::string policy_text = "p" + std::to_string(policy_);
    if (policy_ < 0) policy_text = c.policies.empty() ? "none served" : "shown policy";
    if (ImGui::BeginCombo("##policy", policy_text.c_str())) {
        if (ImGui::Selectable("shown policy", policy_ < 0)) policy_ = -1;
        for (const ServedPolicy &p : c.policies) {
            const std::string name = "p" + std::to_string(p.id) + "  " + base_name(p.model);
            if (ImGui::Selectable(name.c_str(), policy_ == static_cast<int64_t>(p.id))) policy_ = p.id;
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(em * 13);
    if (ImGui::BeginCombo("##target", attr_target_name(target_))) {
        for (int t = 0; t < ATTR_TARGET_COUNT; ++t)
            if (ImGui::Selectable(
                    attr_target_name(static_cast<AttrTarget>(t)), target_ == static_cast<AttrTarget>(t))) {
                target_ = static_cast<AttrTarget>(t);
                sort_metric_ = -1;
            }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(em * 10);
    if (ImGui::BeginCombo("##baseline", featurize::baseline_name(baseline_))) {
        for (const featurize::Baseline b :
            {featurize::Baseline::NONE, featurize::Baseline::MEAN, featurize::Baseline::RESAMPLE})
            if (ImGui::Selectable(featurize::baseline_name(b), baseline_ == b)) baseline_ = b;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    int method = method_ == AttrMethod::SHAPLEY ? 1 : 0;
    ImGui::SetNextItemWidth(em * 11);
    if (ImGui::Combo("##method", &method, "occlusion\0shapley\0"))
        method_ = method ? AttrMethod::SHAPLEY : AttrMethod::OCCLUSION;
    ImGui::SameLine();
    int level = tensors_ ? 1 : 0;
    if (ImGui::RadioButton("state", &level, 0)) tensors_ = false;
    ImGui::SameLine();
    if (ImGui::RadioButton("tensors", &level, 1)) tensors_ = true;
    if (c.stats_missing) ImGui::TextDisabled("%s", c.stats_note.c_str());
    if (!c.current || !a.base()) return;
    // The focus and rival rows.
    const MenuModel &menu = c.data->models().menu;
    const auto row_text = [&menu](int r) {
        if (r < 0) return std::string("\xE2\x80\x94");
        std::string t = "row " + std::to_string(r);
        if (menu.shown && static_cast<size_t>(r) < menu.rows.size())
            t += " \"" + menu.rows[static_cast<size_t>(r)].text + "\"";
        return t;
    };
    if (!a.base()->log_pi.empty())
        ImGui::TextDisabled("focus %s \xC2\xB7 rival %s (a Choices click sets the focus, Shift+click the rival)",
            row_text(a.focus()).c_str(), row_text(a.rival()).c_str());
    if (a.key().method == AttrMethod::SHAPLEY) {
        const ShapleyResult s = a.shapley(target_);
        if (s.ready)
            ImGui::TextDisabled(
                "Shapley over %zu units, %d permutations: \xCE\xA3\xCF\x86 %+.4f, all changed %+.4f (%s)",
                a.entry()->players.size(), SHAPLEY_PERMUTATIONS, s.sum, s.total,
                s.efficient() ? "efficient" : "NOT efficient");
        else if (target_ == AttrTarget::TV)
            ImGui::TextDisabled("Shapley: TV is not additive; choose V, log \xCF\x80 or the margin");
    }
}

void AttributionView::draw_rows(const AttributionContext &c) {
    const Attribution &a = *c.att;
    const int metric = sort_metric_ >= 0                ? sort_metric_
                       : static_cast<int>(target_) == 0 ? 0
                       : static_cast<int>(target_) == 1 ? 1
                       : static_cast<int>(target_) == 2 ? 4
                                                        : 5;
    double scale = 0;
    for (const AttrRow &r : a.rows())
        if (r.m.valid) scale = std::max(scale, std::fabs(r.m.get(target_)));
    const bool shap = a.key().method == AttrMethod::SHAPLEY;
    const int columns = 3 + METRIC_COUNT + (shap ? 1 : 0);
    shown_.clear();
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter |
                                  ImGuiTableFlags_Resizable | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Sortable |
                                  ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##attr", columns, flags)) return;
    const float em = ImGui::CalcTextSize("M").x;
    ImGui::TableSetupScrollFreeze(1, 1);
    ImGui::TableSetupColumn("Unit", ImGuiTableColumnFlags_WidthStretch | ImGuiTableColumnFlags_NoSort);
    ImGui::TableSetupColumn("Kind", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort, em * 4.5f);
    ImGui::TableSetupColumn(
        attr_target_name(target_), ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort, em * 7);
    for (int m = 0; m < METRIC_COUNT; ++m)
        ImGui::TableSetupColumn(METRICS[m].name,
            ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_PreferSortDescending |
                (m == metric ? ImGuiTableColumnFlags_DefaultSort : 0),
            em * 4.6f);
    if (shap)
        ImGui::TableSetupColumn("\xCF\x86", ImGuiTableColumnFlags_WidthFixed | ImGuiTableColumnFlags_NoSort, em * 4.6f);
    ImGui::TableHeadersRow();
    if (ImGuiTableSortSpecs *specs = ImGui::TableGetSortSpecs()) {
        if (specs->SpecsDirty && specs->SpecsCount > 0) {
            const int col = specs->Specs[0].ColumnIndex - 3;
            if (col >= 0 && col < METRIC_COUNT) sort_metric_ = col;
        }
        specs->SpecsDirty = false;
    }
    for (const int r : sorted(a, a.ranked_roots(target_), metric)) draw_row(c, r, scale, metric);
    ImGui::EndTable();
    // Asked after the table: expanding rebuilds the rows.
    if (expand_.empty()) return;
    for (size_t i = 0; i < a.rows().size(); ++i)
        if (a.rows()[i].key == expand_) {
            c.att->expand(static_cast<int>(i));
            break;
        }
    expand_.clear();
}

void AttributionView::draw_row(const AttributionContext &c, int index, double scale, int metric) {
    const Attribution &a = *c.att;
    const AttrRow &r = a.rows()[static_cast<size_t>(index)];
    const featurize::Unit *u = a.unit(r);
    shown_.push_back(index);
    const bool open = open_.count(r.key) > 0;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::PushID(index);
    const float em = ImGui::CalcTextSize("M").x;
    // The linked object or player: the object the unit is of (in the viewer's frame), the player.
    uint64_t handle = u && u->handle && c.tints ? c.tints->view_handle(u->handle) : 0;
    const int seat = u ? u->seat : -1;
    const bool linked =
        c.links && ((handle && c.links->object_linked(handle)) ||
                       (!handle && seat >= 0 && u->kind == featurize::UnitKind::GROUP && c.links->player_linked(seat)));
    if (linked) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, linked_row_color());
    const bool clicked =
        ImGui::Selectable("##row", false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap);
    if ((ImGui::IsItemHovered() || r.key == force_hover_) && c.links) {
        if (handle)
            c.links->hover_object(handle);
        else if (seat >= 0 && u && u->kind == featurize::UnitKind::GROUP)
            c.links->hover_player(seat);
        if (u) ImGui::SetItemTooltip("%s\n%s", u->label.c_str(), u->tensors.c_str());
    }
    if (clicked) {
        if (r.expandable) {
            if (open) {
                open_.erase(r.key);
            } else {
                open_.insert(r.key);
                if (!r.asked) expand_ = r.key;
            }
        }
        if (c.links && (handle || seat >= 0)) {
            LinkItem item;
            item.kind = handle ? LinkItem::Kind::OBJECT : LinkItem::Kind::PLAYER;
            item.handle = handle;
            item.seat = seat;
            c.links->select(item);
        }
    }
    ImGui::SameLine(0, 0);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float x = p.x + em * 1.4f * static_cast<float>(r.depth);
    if (r.expandable)
        ImGui::RenderArrow(ImGui::GetWindowDrawList(), ImVec2(x, p.y + ImGui::GetTextLineHeight() * 0.15f), tok().DIM,
            open ? ImGuiDir_Down : ImGuiDir_Right, 0.7f);
    ImGui::SetCursorScreenPos(ImVec2(x + em * 1.3f, p.y));
    ImGui::TextUnformatted(label(a, r).c_str());
    ImGui::TableNextColumn();
    ImGui::TextDisabled("%s", u ? featurize::unit_kind_name(u->kind) : "");
    ImGui::TableNextColumn();
    // The target's bar: from the middle for a signed target, from the left for TV.
    const bool sign = attr_target_signed(target_);
    if (r.m.valid && (target_ == AttrTarget::V || r.m.policy)) {
        const double v = r.m.get(target_);
        const float w = ImGui::GetContentRegionAvail().x, h = ImGui::GetTextLineHeight() * 0.6f;
        const ImVec2 q = ImGui::GetCursorScreenPos();
        const float y = q.y + (ImGui::GetTextLineHeight() - h) / 2;
        const float f = scale > 0 ? static_cast<float>(std::min(1.0, std::fabs(v) / scale)) : 0.0f;
        ImDrawList *dl = ImGui::GetWindowDrawList();
        if (sign) {
            const float mid = q.x + w / 2;
            const float end = v >= 0 ? mid + f * w / 2 : mid - f * w / 2;
            dl->AddRectFilled(
                ImVec2(std::min(mid, end), y), ImVec2(std::max(mid, end) + 1, y + h), heat_color(v, scale, true, 255));
            dl->AddLine(ImVec2(mid, y - 1), ImVec2(mid, y + h + 1), tok().LINE);
        } else {
            dl->AddRectFilled(ImVec2(q.x, y), ImVec2(q.x + 1 + f * w, y + h), heat_color(v, scale, false, 255));
        }
        ImGui::Dummy(ImVec2(w, ImGui::GetTextLineHeight()));
    } else {
        ImGui::TextDisabled("%s", r.m.valid ? "\xE2\x80\x94" : "\xE2\x80\xA6");
    }
    for (int m = 0; m < METRIC_COUNT; ++m) {
        ImGui::TableNextColumn();
        const bool pi = METRICS[m].target != 0;
        if (!r.m.valid || (pi && !r.m.policy)) {
            ImGui::TextDisabled("\xE2\x80\x94");
            continue;
        }
        const double v = metric_of(r.m, m);
        const bool signed_metric = m == 0 || m == 4 || m == 5;
        if (signed_metric)
            signed_value(v, number(v, true));
        else
            ImGui::TextUnformatted(number(v, false).c_str());
    }
    if (a.key().method == AttrMethod::SHAPLEY) {
        ImGui::TableNextColumn();
        if (r.shapley)
            signed_value(r.phi, number(r.phi, true));
        else
            ImGui::TextDisabled(" ");
    }
    ImGui::PopID();
    if (!open) return;
    for (const int k : sorted(a, r.children, metric)) draw_row(c, k, scale, metric);
}

}  // namespace rmwork
