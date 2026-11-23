#include "policy_compare_view.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>

#include "implot.h"
#include "paths.h"
#include "policy_catalog.h"
#include "theme.h"
#include "tool_widgets.h"
#include "value_views.h"
#include "widgets.h"

namespace {

// The ranked list's rows at most.
const size_t RANK_ROWS = 200;
// From this width (in ems) the tool draws two columns.
const float WIDE_EMS = 110.0f;
// How often the catalog popup asks the compatibility checker again while it is open.
const double COMPAT_EVERY_S = 1.0;

}  // namespace

static ImVec4 color4(ImU32 c);
static void wrap_for(float width);
static std::string base_name(const std::string &path);
static std::string pi_text(double p);
static void cursor_line(size_t cursor);
static bool row_selectable(const char *id, float height);

static ImVec4 color4(ImU32 c) {
    return ImGui::ColorConvertU32ToFloat4(c);
}

// Starts a new line unless an item `width` wide fits after the last one.
static void wrap_for(float width) {
    const float right = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    if (ImGui::GetCursorScreenPos().x + width > right) ImGui::NewLine();
}

static std::string base_name(const std::string &path) {
    const size_t slash = path.find_last_of("/!");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// π with two decimals, three below 0.1.
static std::string pi_text(double p) {
    char buf[32];
    std::snprintf(buf, sizeof buf, p < 0.1 ? "%.3f" : "%.2f", p);
    return buf;
}

// The session's cursor as a dashed vertical line across the current plot.
static void cursor_line(size_t cursor) {
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    ImPlot::PushPlotClipRect();
    const ImVec2 pos = ImPlot::GetPlotPos(), size = ImPlot::GetPlotSize();
    const float x = ImPlot::PlotToPixels(static_cast<double>(cursor), 0).x;
    for (float y = pos.y; y < pos.y + size.y; y += 10)
        dl->AddLine(ImVec2(x, y), ImVec2(x, std::min(pos.y + size.y, y + 6)), rmwork::tok().TEXT);
    ImPlot::PopPlotClipRect();
}

// A row-spanning selectable `height` tall, other items allowed over it; true when clicked.
static bool row_selectable(const char *id, float height) {
    return ImGui::Selectable(
        id, false, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap, ImVec2(0, height));
}

namespace rmwork {

ImU32 PolicyCompareView::policy_color(int k) {
    const Palette &p = tok();
    const ImU32 colors[] = {p.S_A, p.S_B, p.ACCENT, p.LIFE_A, p.LIFE_B, p.GOOD, p.S_A6, p.BAD};
    return colors[static_cast<size_t>(std::max(0, k)) % (sizeof colors / sizeof colors[0])];
}

void PolicyCompareView::set_included(int64_t id, bool on) {
    if (on)
        excluded_.erase(id);
    else
        excluded_.insert(id);
}

void PolicyCompareView::update(const ReplaySession &s, const ToolData &data) {
    current_ = nullptr;
    if (!s.loaded()) {
        state_ = State::NO_MATCH;
        candidates_.clear();
        cmp_ = MatchComparison();
        ranked_.clear();
        key_.clear();
        return;
    }
    const Match &m = *s.match();
    candidates_ = comparable_policies(data.values(), s.view());
    versions_ = data.values().versions;
    for (int64_t p : candidates_)
        if (!ref_name_.empty() && policy_name(p) == ref_name_) {
            ref_ = p;
            ref_name_.clear();
        }
    std::vector<int64_t> chosen;
    for (int64_t p : candidates_)
        if (included(p)) chosen.push_back(p);
    const int64_t ref =
        std::find(chosen.begin(), chosen.end(), ref_) != chosen.end() ? ref_ : default_reference(chosen);
    state_ = candidates_.empty()       ? State::NO_POLICIES
             : candidates_.size() == 1 ? State::ONE_POLICY
             : chosen.size() < 2       ? State::ONE_CHOSEN
                                       : State::READY;
    if (state_ != State::READY) {
        cmp_ = MatchComparison();
        ranked_.clear();
        key_.clear();
        return;
    }
    std::string key = std::to_string(m.serial) + '|' + std::to_string(m.evaluations_changed) + '|' +
                      std::to_string(m.episode_steps.size()) + '|' + std::to_string(m.decisions.size()) + '|' +
                      std::to_string(s.view()) + '|' + std::to_string(ref);
    for (int64_t p : chosen) key += ',' + std::to_string(p);
    if (key != key_) {
        cmp_ = compare_match(m, s.view(), chosen, ref);
        key_ = key;
        rank_key_.clear();
    }
    if (texts_key_ != std::to_string(m.serial) + '|' + std::to_string(s.view())) {
        texts_key_ = std::to_string(m.serial) + '|' + std::to_string(s.view());
        texts_.clear();
    }
    const int64_t ranked_policy = rank_policy_ != ref && cmp_.position(rank_policy_) >= 0 ? rank_policy_ : -1;
    const std::string rk = key_ + '|' + compare_metric_name(metric_) + '|' + (argmax_only_ ? "1" : "0") + '|' +
                           std::to_string(ranked_policy);
    if (rk != rank_key_) {
        ranked_ = rank_disagreements(cmp_, ranked_policy, metric_, argmax_only_, RANK_ROWS);
        rank_key_ = rk;
    }
    current_ = cmp_.find(s.cursor());
}

std::string PolicyCompareView::label_of(int64_t policy) const {
    auto it = versions_.find(policy);
    return policy_label(policy, it == versions_.end() ? 0 : it->second);
}

ImU32 PolicyCompareView::color_of(int64_t policy) const {
    for (size_t k = 0; k < candidates_.size(); ++k)
        if (candidates_[k] == policy) return policy_color(static_cast<int>(k));
    return tok().DIM;
}

// The model a re-evaluated policy's served id names ("" for a recorded policy or one no longer served).
std::string PolicyCompareView::model_of(const std::vector<ServedPolicy> &served, int64_t policy) const {
    if (!is_reevaluated(policy)) return "";
    for (const ServedPolicy &p : served)
        if (reevaluated_id(p.id) == policy) return p.model;
    return "";
}

bool PolicyCompareView::go_to(ReplaySession *s, size_t index) {
    if (!s || !s->loaded()) return false;
    s->set_playing(false, ImGui::GetTime());
    return s->go_to(index);
}

const std::vector<std::string> &PolicyCompareView::menu_texts(const Match &m, size_t index, int view) {
    auto it = texts_.find(index);
    if (it != texts_.end()) return it->second;
    std::string err;
    return texts_[index] = menu_descriptions(m, index, view, &err);
}

std::string PolicyCompareView::row_text(const Match &m, size_t index, int view, int row) {
    if (row < 0) return "\xE2\x80\x94";
    const std::vector<std::string> &t = menu_texts(m, index, view);
    const std::string head = "#" + std::to_string(row);
    return static_cast<size_t>(row) < t.size() ? head + " " + t[static_cast<size_t>(row)] : head;
}

std::string PolicyCompareView::report(
    const ReplaySession &s, const ToolData &data, const std::vector<ServedPolicy> &served, size_t n) {
    update(s, data);
    if (state_ != State::READY) return "POLICY-COMPARE: nothing to compare\n";
    const Match &m = *s.match();
    const auto named = [&](int64_t p) {
        const std::string model = model_of(served, p);
        return label_of(p) + (model.empty() ? "" : " (" + model + ")");
    };
    std::string out = "POLICY-COMPARE: " + m.title() + ", " + view_name(s.view()) + "'s view, " +
                      std::to_string(cmp_.decisions.size()) + " decisions compared against " + named(cmp_.ref) + "\n";
    for (const PolicyAgreement &a : cmp_.agreement)
        out += "  " + named(a.policy) + ": argmax agrees " + agreement_text(a.all) + ", A " +
               agreement_text(a.seat[0]) + ", B " + agreement_text(a.seat[1]) + "; mean KL " +
               metric_text(a.all.mean_kl()) + ", mean TV " + metric_text(a.all.mean_tv()) + "\n";
    out += std::string("  the decisions where they diverge most by ") + compare_metric_name(metric_) +
           (argmax_only_ ? ", argmax differing only" : "") + ":\n";
    for (size_t i = 0; i < ranked_.size() && i < n; ++i) {
        const RankedDecision &r = ranked_[i];
        char buf[96];
        std::snprintf(buf, sizeof buf, "  #%zu %s %s %s: KL %s TV %s", r.index + 1,
            turn_label(m.decisions[r.index]).c_str(), m.decisions[r.index].decider == 0 ? "A" : "B",
            label_of(r.policy).c_str(), metric_text(r.kl).c_str(), metric_text(r.tv).c_str());
        out += buf;
        out += "; reference " + row_text(m, r.index, s.view(), r.ref_argmax);
        if (r.differs)
            out +=
                ", it " + row_text(m, r.index, s.view(), r.argmax) + " (\xCE\x94\xCF\x80 " + signed_text(r.shift) + ")";
        out += "\n";
    }
    return out;
}

CompareRequest PolicyCompareView::draw(const CompareContext &c) {
    CompareRequest req;
    const float width = 46.0f * ImGui::CalcTextSize("M").x;
    if (!c.session || !c.data) return req;
    update(*c.session, *c.data);
    if (state_ == State::NO_MATCH) {
        centered_text_block("No game loaded",
            "Open a replay (or start a game): Policy Compare sets several policies' \xCF\x80 and V side by side.",
            width);
        return req;
    }
    draw_header(c, &req);
    if (state_ != State::READY) {
        const bool seat_view = c.session->view() != VIEW_OMNI;
        std::string title, body;
        if (state_ == State::NO_POLICIES) {
            title = "No evaluations to compare";
            body = std::string(seat_view ? "This seat's decisions carry no evaluation (a seat view sees only its own "
                                           "seat's). "
                                         : "The match carries no evaluation. ") +
                   "Add checkpoints\xE2\x80\xA6 serves policies of the runs and re-evaluates the match with them; "
                   "Analysis \xE2\x86\x92 Re-evaluate uses the served ones.";
        } else if (state_ == State::ONE_POLICY) {
            title = "One policy";
            body = "Only " + label_of(candidates_.front()) +
                   " evaluated the decisions this viewer sees: add a checkpoint to compare it with (Add "
                   "checkpoints\xE2\x80\xA6), or re-evaluate with another served policy.";
        } else {
            title = "Tick two policies";
            body = "Policy Compare needs two policies or more ticked, one of them the reference.";
        }
        if (c.mode == Mode::PLAY)
            body +=
                " In Play the tool compares what Model's Read evaluated (every served policy with a numeric id); "
                "a whole match is re-evaluated in Replay.";
        centered_text_block(title.c_str(), body.c_str(), width);
        draw_picker(c, &req);
        return req;
    }
    // Two columns when the tool is wide (the decision and the agreement; the plots and the ranked list), else
    // one above the other.
    const float line = ImGui::GetTextLineHeightWithSpacing(), em = ImGui::CalcTextSize("M").x;
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float spacing = ImGui::GetStyle().ItemSpacing.y;
    if (avail.x >= em * WIDE_EMS) {
        if (ImGui::BeginChild("##cmp_left", ImVec2(avail.x * 0.46f, avail.y))) {
            draw_decision(c, std::max(line * 5, avail.y - agreement_height() - spacing));
            draw_agreement(c);
        }
        ImGui::EndChild();
        ImGui::SameLine();
        if (ImGui::BeginChild("##cmp_right", ImVec2(0, avail.y))) {
            draw_plots(c, std::max(line * 6, avail.y * 0.45f));
            draw_ranked(c);
        }
        ImGui::EndChild();
    } else {
        const float rest = std::max(line * 6, avail.y - line * 0.5f);
        draw_decision(c, std::max(line * 5, std::min(decision_height(), rest * 0.34f)));
        draw_plots(c, std::max(line * 6, rest * 0.30f));
        draw_agreement(c);
        draw_ranked(c);
    }
    draw_picker(c, &req);
    return req;
}

float PolicyCompareView::table_row_height() {
    return ImGui::GetTextLineHeight() + ImGui::GetStyle().CellPadding.y * 2;
}

float PolicyCompareView::agreement_height() const {
    return table_row_height() * static_cast<float>(1 + cmp_.agreement.size()) + 4;
}

float PolicyCompareView::decision_height() const {
    const size_t rows = current_ ? current_->rows : 1;
    return ImGui::GetTextLineHeightWithSpacing() + table_row_height() * 3 +
           (ImGui::GetTextLineHeight() + 5 + ImGui::GetStyle().CellPadding.y * 2) * static_cast<float>(rows) + 8;
}

void PolicyCompareView::draw_header(const CompareContext &c, CompareRequest *req) {
    (void)req;
    const float em = ImGui::CalcTextSize("M").x;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 1));
    if (!candidates_.empty()) {
        ImGui::TextDisabled("Compare");
        ImGui::SameLine();
    }
    const int64_t ref = cmp_.ref >= 0 ? cmp_.ref : ref_;
    for (size_t k = 0; k < candidates_.size(); ++k) {
        const int64_t p = candidates_[k];
        const std::string label = label_of(p);
        const bool is_ref = p == ref && state_ == State::READY;
        const std::string shown = label + (is_ref ? " (reference)" : "");
        wrap_for(ImGui::GetFrameHeight() * 2 + em * 1.5f + ImGui::CalcTextSize(shown.c_str()).x + 16);
        ImGui::PushID(static_cast<int>(k));
        bool on = included(p);
        if (ImGui::Checkbox("##on", &on)) set_included(p, on);
        ImGui::SetItemTooltip("Compare %s", label.c_str());
        ImGui::SameLine(0, 2);
        if (ImGui::RadioButton("##ref", is_ref)) {
            ref_ = p;
            set_included(p, true);
        }
        ImGui::SetItemTooltip("Make %s the reference: KL, TV and agreement are against it", label.c_str());
        ImGui::SameLine(0, 4);
        ImGui::PushStyleColor(ImGuiCol_Text, policy_color(static_cast<int>(k)));
        ImGui::TextUnformatted("\xE2\x96\xA0");
        ImGui::PopStyleColor();
        ImGui::SameLine(0, 3);
        if (is_ref)
            ImGui::TextUnformatted(shown.c_str());
        else
            ImGui::TextDisabled("%s", shown.c_str());
        const std::string model = model_of(c.served, p);
        if (!model.empty()) ImGui::SetItemTooltip("%s", model.c_str());
        ImGui::SameLine(0, 14);
        ImGui::PopID();
    }
    wrap_for(em * 30);
    const bool can_add = c.mode == Mode::REPLAY && c.session->loaded() && !c.session->match()->live;
    ImGui::BeginDisabled(!can_add);
    if (ImGui::SmallButton("Add checkpoints\xE2\x80\xA6")) open_picker_ = true;
    ImGui::EndDisabled();
    if (!can_add) ImGui::SetItemTooltip("Bringing checkpoints in re-evaluates a match: open it in Replay");
    ImGui::SameLine();
    int metric = metric_ == CompareMetric::KL ? 0 : 1;
    ImGui::SetNextItemWidth(em * 5);
    if (ImGui::Combo("##metric", &metric, "KL\0TV\0")) metric_ = metric == 0 ? CompareMetric::KL : CompareMetric::TV;
    ImGui::SetItemTooltip(
        "The divergence the plot and the ranked list use: KL(ref \xE2\x80\x96 p) in nats, or TV (the "
        "probability moved)");
    if (c.reevaluating) {
        ImGui::SameLine();
        ImGui::ProgressBar(c.reeval_fraction < 0 ? 0.0f : static_cast<float>(c.reeval_fraction), ImVec2(em * 10, 0),
            c.reeval_phase.c_str());
    }
    ImGui::PopStyleVar();
    if (!c.note.empty()) ImGui::TextWrapped("%s", c.note.c_str());
}

void PolicyCompareView::draw_decision(const CompareContext &c, float height) {
    if (!ImGui::BeginChild("##cmp_decision", ImVec2(0, height))) {
        ImGui::EndChild();
        return;
    }
    const DecisionFrame &f = c.data->frame();
    const MenuModel &menu = c.data->models().menu;
    const DecisionInfo *d = f.info;
    std::string head = "Decision " + std::to_string(f.index + 1);
    if (d) head += " \xC2\xB7 " + turn_label(*d) + " \xC2\xB7 " + seat_name(d->decider) + " decides";
    ImGui::TextUnformatted(head.c_str());
    if (!c.data->ok() || !menu.shown) {
        ImGui::TextDisabled("%s", d ? (std::string(seat_name(d->decider)) +
                                          " decides: a seat view sees only its own seat's decisions and evaluations")
                                          .c_str()
                                    : "");
        ImGui::EndChild();
        return;
    }
    if (!current_) {
        ImGui::TextDisabled("The reference, %s, did not evaluate this decision", label_of(cmp_.ref).c_str());
        ImGui::EndChild();
        return;
    }
    const DecisionCompare &dc = *current_;
    const float em = ImGui::CalcTextSize("M").x;
    const int columns = 1 + static_cast<int>(dc.at.size());
    const float table_w = ImGui::GetContentRegionAvail().x;
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter |
                                  ImGuiTableFlags_ScrollY | ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##cmp_rows", columns, flags, ImVec2(0, ImGui::GetContentRegionAvail().y))) {
        ImGui::EndChild();
        return;
    }
    ImGui::TableSetupScrollFreeze(1, 3);
    // The rows' text takes what the policies' columns leave, at least 16 ems.
    float policy_w = 0;
    for (const PolicyAt &p : dc.at)
        policy_w +=
            std::max(ImGui::CalcTextSize("0.000 / 0.000").x, ImGui::CalcTextSize(label_of(p.policy).c_str()).x) + em;
    ImGui::TableSetupColumn("Row", ImGuiTableColumnFlags_WidthFixed,
        std::max(em * 16, table_w - policy_w - em * static_cast<float>(dc.at.size())));
    for (const PolicyAt &p : dc.at)
        ImGui::TableSetupColumn(label_of(p.policy).c_str(), ImGuiTableColumnFlags_WidthFixed,
            std::max(ImGui::CalcTextSize("0.000 / 0.000").x, ImGui::CalcTextSize(label_of(p.policy).c_str()).x) +
                em * 0.5f);
    ImGui::TableHeadersRow();
    const bool scale_a = c.data->scale_a;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled(scale_a ? "V (Player A's scale)" : "V (the deciding seat's scale)");
    for (const PolicyAt &p : dc.at) {
        ImGui::TableNextColumn();
        if (!p.has) {
            ImGui::TextDisabled("\xE2\x80\x94");
            continue;
        }
        const double v = scale_a ? on_a_scale(dc.seat, p.value) : p.value;
        signed_value(v, signed_text(v, 3));
    }
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::TextDisabled("KL / TV against the reference");
    for (size_t k = 0; k < dc.at.size(); ++k) {
        const PolicyAt &p = dc.at[k];
        ImGui::TableNextColumn();
        if (static_cast<int>(k) == dc.ref) {
            ImGui::TextUnformatted("reference");
        } else if (!p.has) {
            ImGui::TextDisabled("not evaluated");
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, p.agrees ? tok().TEXT : tok().BAD);
            ImGui::TextUnformatted((metric_text(p.kl) + " / " + metric_text(p.tv)).c_str());
            ImGui::PopStyleColor();
            ImGui::SetItemTooltip("KL(ref \xE2\x80\x96 p) %.4f nats \xC2\xB7 TV %.4f \xC2\xB7 %s", p.kl, p.tv,
                p.agrees ? "the same argmax" : "another argmax");
        }
    }
    const float row_h = ImGui::GetTextLineHeight() + 5;
    for (size_t r = 0; r < dc.rows; ++r) {
        const int row = static_cast<int>(r);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushID(row);
        if (c.links && c.links->row_linked(row)) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, linked_row_color());
        const bool clicked = row_selectable("##row", row_h);
        if ((ImGui::IsItemHovered() || row == force_hover_) && c.links) c.links->hover_menu_row(row);
        if (clicked && c.links) {
            LinkItem item;
            item.kind = LinkItem::Kind::MENU_ROW;
            item.row = row;
            c.links->select(item);
        }
        ImGui::SameLine(0, 0);
        if (row == menu.chosen) {
            ImGui::PushStyleColor(ImGuiCol_Text, tok().ACCENT);
            ImGui::TextUnformatted("\xE2\x96\xBA");
            ImGui::PopStyleColor();
            ImGui::SameLine();
        }
        ImGui::TextDisabled("%d", row);
        ImGui::SameLine();
        ImGui::TextUnformatted(r < menu.rows.size() ? menu.rows[r].text.c_str() : "");
        for (size_t k = 0; k < dc.at.size(); ++k) {
            const PolicyAt &p = dc.at[k];
            ImGui::TableNextColumn();
            if (!p.has || r >= p.pi.size()) {
                ImGui::TextDisabled("\xE2\x80\x94");
                continue;
            }
            const ImU32 color = color_of(p.policy);
            const ImVec2 q = ImGui::GetCursorScreenPos();
            const float w = ImGui::GetContentRegionAvail().x;
            if (p.argmax == row) {
                ImGui::PushStyleColor(ImGuiCol_Text, color);
                ImGui::TextUnformatted("\xE2\x97\x8F");
                ImGui::PopStyleColor();
                ImGui::SetItemTooltip("%s's argmax", label_of(p.policy).c_str());
                ImGui::SameLine(0, 4);
            } else {
                ImGui::SetCursorScreenPos(ImVec2(q.x + ImGui::CalcTextSize("\xE2\x97\x8F").x + 4, q.y));
            }
            ImGui::TextUnformatted(pi_text(p.pi[r]).c_str());
            const float y = q.y + ImGui::GetTextLineHeight() + 1;
            ImGui::GetWindowDrawList()->AddRectFilled(
                ImVec2(q.x, y), ImVec2(q.x + std::max(1.0f, w * static_cast<float>(p.pi[r])), y + 3), color);
        }
        ImGui::PopID();
    }
    ImGui::EndTable();
    ImGui::EndChild();
}

void PolicyCompareView::draw_plots(const CompareContext &c, float height) {
    const Match &m = *c.session->match();
    const double n = static_cast<double>(m.decisions.size());
    const ImPlotSubplotFlags sub = ImPlotSubplotFlags_NoTitle | ImPlotSubplotFlags_LinkAllX |
                                   ImPlotSubplotFlags_NoResize | ImPlotSubplotFlags_NoMenus;
    float ratios[2] = {0.5f, 0.5f};
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) dragging_ = false;
    if (!ImPlot::BeginSubplots("##cmp_plots", 2, 1, ImVec2(-1, height), sub, ratios)) return;
    const ImPlotFlags flags = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText |
                              ImPlotFlags_NoLegend | ImPlotFlags_NoFrame;
    const bool kl = metric_ == CompareMetric::KL;
    // The divergence of each policy from the reference.
    double top = kl ? 0.05 : 0.0;
    for (const DecisionCompare &d : cmp_.decisions) top = std::max(top, kl ? d.max_kl : d.max_tv);
    if (ImPlot::BeginPlot("##div", ImVec2(-1, 0), flags)) {
        ImPlot::SetupAxis(
            ImAxis_X1, nullptr, ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_Lock);
        ImPlot::SetupAxis(ImAxis_Y1, kl ? "KL vs ref" : "TV vs ref", ImPlotAxisFlags_Lock);
        ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, n + 0.5, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, kl ? top * 1.1 : 1.05, ImPlotCond_Always);
        for (size_t k = 0; k < cmp_.policies.size(); ++k) {
            const int64_t p = cmp_.policies[k];
            if (p == cmp_.ref) continue;
            std::vector<double> xs, ys, dx, dy;
            for (const DecisionCompare &d : cmp_.decisions) {
                const PolicyAt &a = d.at[k];
                if (!a.has) continue;
                xs.push_back(static_cast<double>(d.index));
                ys.push_back(kl ? a.kl : a.tv);
                if (!a.agrees) {
                    dx.push_back(xs.back());
                    dy.push_back(ys.back());
                }
            }
            const std::string id = "##d" + std::to_string(k);
            ImPlot::SetNextLineStyle(color4(color_of(p)), 1.3f);
            ImPlot::PlotLine(id.c_str(), xs.data(), ys.data(), static_cast<int>(xs.size()));
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Square, 2.5f, color4(tok().BAD), 0, color4(tok().BAD));
            ImPlot::PlotScatter((id + "x").c_str(), dx.data(), dy.data(), static_cast<int>(dx.size()));
        }
        cursor_line(c.session->cursor());
        const bool hovered = ImPlot::IsPlotHovered();
        if (hovered) {
            const long i = std::lround(ImPlot::GetPlotMousePos().x);
            const DecisionCompare *d = i >= 0 ? cmp_.find(static_cast<size_t>(i)) : nullptr;
            if (d) {
                ImGui::BeginTooltip();
                ImGui::Text("#%ld \xC2\xB7 %s \xC2\xB7 %s decides", i + 1, turn_label(m.decisions[d->index]).c_str(),
                    seat_name(d->seat));
                for (const PolicyAt &a : d->at)
                    if (a.has && a.policy != cmp_.ref)
                        ImGui::Text("%s  KL %s  TV %s%s", label_of(a.policy).c_str(), metric_text(a.kl).c_str(),
                            metric_text(a.tv).c_str(), a.agrees ? "" : "  (another argmax)");
                ImGui::EndTooltip();
            }
        }
        plot_scrub(*c.session, hovered, &dragging_);
        ImPlot::EndPlot();
    }
    // Each policy's value on Player A's scale.
    if (ImPlot::BeginPlot("##val", ImVec2(-1, 0), flags)) {
        ImPlot::SetupAxis(
            ImAxis_X1, nullptr, ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_Lock);
        ImPlot::SetupAxis(ImAxis_Y1, "V (A)", ImPlotAxisFlags_Lock);
        ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, n + 0.5, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.12, 1.12, ImPlotCond_Always);
        for (size_t k = 0; k < cmp_.policies.size(); ++k) {
            std::vector<double> xs, ys;
            for (const DecisionCompare &d : cmp_.decisions) {
                const PolicyAt &a = d.at[k];
                if (!a.has) continue;
                xs.push_back(static_cast<double>(d.index));
                ys.push_back(on_a_scale(d.seat, a.value));
            }
            const std::string id = "##v" + std::to_string(k);
            ImPlot::SetNextLineStyle(color4(color_of(cmp_.policies[k])), cmp_.policies[k] == cmp_.ref ? 2.0f : 1.3f);
            ImPlot::PlotLine(id.c_str(), xs.data(), ys.data(), static_cast<int>(xs.size()));
        }
        cursor_line(c.session->cursor());
        const bool hovered = ImPlot::IsPlotHovered();
        if (hovered) {
            const long i = std::lround(ImPlot::GetPlotMousePos().x);
            const DecisionCompare *d = i >= 0 ? cmp_.find(static_cast<size_t>(i)) : nullptr;
            if (d) {
                ImGui::BeginTooltip();
                ImGui::Text("#%ld \xC2\xB7 V on Player A's scale", i + 1);
                for (const PolicyAt &a : d->at)
                    if (a.has)
                        ImGui::Text(
                            "%s  %s", label_of(a.policy).c_str(), signed_text(on_a_scale(d->seat, a.value)).c_str());
                ImGui::EndTooltip();
            }
        }
        plot_scrub(*c.session, hovered, &dragging_);
        ImPlot::EndPlot();
    }
    ImPlot::EndSubplots();
}

void PolicyCompareView::draw_agreement(const CompareContext &c) {
    const float em = ImGui::CalcTextSize("M").x;
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter |
                                  ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##cmp_agree", 6, flags)) return;
    float label_w = ImGui::CalcTextSize("Policy").x;
    for (const PolicyAgreement &a : cmp_.agreement)
        label_w = std::max(label_w, ImGui::CalcTextSize(label_of(a.policy).c_str()).x);
    // The seats' column: both seats' rates in the omniscient view, else the viewer's own seat's.
    const int view = c.session->view();
    ImGui::TableSetupColumn("Policy", ImGuiTableColumnFlags_WidthFixed, label_w + em * 1.6f);
    ImGui::TableSetupColumn("argmax agrees", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("100% (999/999)").x);
    ImGui::TableSetupColumn(view == VIEW_OMNI ? "A \xC2\xB7 B"
                            : view == VIEW_A  ? "A"
                                              : "B",
        ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("100% \xC2\xB7 100%").x);
    ImGui::TableSetupColumn("mean KL", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("mean KL").x);
    ImGui::TableSetupColumn("mean TV", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("mean TV").x);
    ImGui::TableSetupColumn("per turn", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableHeadersRow();
    for (const PolicyAgreement &a : cmp_.agreement) {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::PushStyleColor(ImGuiCol_Text, color_of(a.policy));
        ImGui::TextUnformatted("\xE2\x96\xA0");
        ImGui::PopStyleColor();
        ImGui::SameLine(0, 3);
        ImGui::TextUnformatted(label_of(a.policy).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(agreement_text(a.all).c_str());
        ImGui::TableNextColumn();
        std::string seats;
        for (int seat = 0; seat < 2; ++seat) {
            if (view != VIEW_OMNI && view != seat) continue;
            const Agreement &g = a.seat[static_cast<size_t>(seat)];
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.0f%%", 100.0 * g.rate());
            seats += (seats.empty() ? "" : " \xC2\xB7 ") + (g.n ? std::string(buf) : std::string("\xE2\x80\x94"));
        }
        ImGui::TextUnformatted(seats.c_str());
        ImGui::SetItemTooltip(
            "Player A: %s\nPlayer B: %s", agreement_text(a.seat[0]).c_str(), agreement_text(a.seat[1]).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(metric_text(a.all.mean_kl()).c_str());
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(metric_text(a.all.mean_tv()).c_str());
        ImGui::TableNextColumn();
        // The strip: a cell per turn band the policy was compared in.
        std::vector<size_t> cells;
        for (size_t b = 0; b < a.bands.size(); ++b)
            if (a.bands[b].n) cells.push_back(b);
        const float w = ImGui::GetContentRegionAvail().x, h = ImGui::GetTextLineHeight();
        if (cells.empty() || w < 4) {
            ImGui::TextDisabled("\xE2\x80\x94");
            continue;
        }
        const float cw = std::min(em * 0.9f, w / static_cast<float>(cells.size()));
        const ImVec2 q = ImGui::GetCursorScreenPos();
        ImDrawList *dl = ImGui::GetWindowDrawList();
        for (size_t i = 0; i < cells.size(); ++i) {
            const Agreement &g = a.bands[cells[i]];
            const float x = q.x + cw * static_cast<float>(i);
            dl->AddRectFilled(
                ImVec2(x, q.y + 1), ImVec2(x + cw - 1, q.y + h - 1), heat_color(1.0 - g.rate(), 1.0, false, 230));
            dl->AddRect(ImVec2(x, q.y + 1), ImVec2(x + cw - 1, q.y + h - 1), tok().LINE);
        }
        ImGui::Dummy(ImVec2(cw * static_cast<float>(cells.size()), h));
        if (ImGui::IsItemHovered()) {
            const size_t i = static_cast<size_t>(std::max(0.0f, (ImGui::GetMousePos().x - q.x) / cw));
            if (i < cells.size()) {
                const TurnBand &band = cmp_.bands[cells[i]];
                const Agreement &g = a.bands[cells[i]];
                ImGui::SetTooltip("%s of game %d: argmax agrees at %s; mean KL %s, TV %s", band.label.c_str(),
                    band.game + 1, agreement_text(g).c_str(), metric_text(g.mean_kl()).c_str(),
                    metric_text(g.mean_tv()).c_str());
            }
        }
    }
    ImGui::EndTable();
}

void PolicyCompareView::draw_ranked(const CompareContext &c) {
    const float em = ImGui::CalcTextSize("M").x;
    const Match &m = *c.session->match();
    const int view = c.session->view();
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 1));
    ImGui::TextDisabled("Where");
    ImGui::SameLine();
    const bool all = rank_policy_ < 0 || rank_policy_ == cmp_.ref || cmp_.position(rank_policy_) < 0;
    ImGui::SetNextItemWidth(em * 12);
    if (ImGui::BeginCombo("##rank_policy", all ? "any policy" : label_of(rank_policy_).c_str())) {
        if (ImGui::Selectable("any policy", all)) rank_policy_ = -1;
        for (const PolicyAgreement &a : cmp_.agreement)
            if (ImGui::Selectable(label_of(a.policy).c_str(), !all && rank_policy_ == a.policy))
                rank_policy_ = a.policy;
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::Text("diverges most from %s by %s", label_of(cmp_.ref).c_str(), compare_metric_name(metric_));
    ImGui::SameLine(0, 12);
    ImGui::Checkbox("argmax differs only", &argmax_only_);
    ImGui::SetItemTooltip(
        "Only the decisions where the policy's preferred row is not the reference's: where a newer "
        "checkpoint changed its mind, ranked by the probability moved when the metric is TV");
    ImGui::PopStyleVar();
    // Applied after the controls: the list follows them from this frame on.
    update(*c.session, *c.data);
    if (force_jump_ >= 0) {
        if (static_cast<size_t>(force_jump_) < ranked_.size())
            go_to(c.session, ranked_[static_cast<size_t>(force_jump_)].index);
        force_jump_ = -1;
    }
    if (ranked_.empty()) {
        ImGui::TextDisabled(argmax_only_ ? "The argmax agrees at every decision compared" : "Nothing to rank");
        return;
    }
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter |
                                  ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
    if (!ImGui::BeginTable("##cmp_ranked", 8, flags, ImVec2(0, ImGui::GetContentRegionAvail().y))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Decision", ImGuiTableColumnFlags_WidthFixed, em * 7.5f);
    ImGui::TableSetupColumn("Seat", ImGuiTableColumnFlags_WidthFixed, seat_chip_width() + 4);
    float label_w = ImGui::CalcTextSize("Policy").x;
    for (const PolicyAgreement &a : cmp_.agreement)
        label_w = std::max(label_w, ImGui::CalcTextSize(label_of(a.policy).c_str()).x);
    ImGui::TableSetupColumn("Policy", ImGuiTableColumnFlags_WidthFixed, label_w);
    ImGui::TableSetupColumn("KL", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0.000").x);
    ImGui::TableSetupColumn("TV", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("0.000").x);
    ImGui::TableSetupColumn("Reference prefers", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("It prefers", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("\xCE\x94\xCF\x80", ImGuiTableColumnFlags_WidthFixed, em * 3.6f);
    ImGui::TableHeadersRow();
    ImGuiListClipper clip;
    clip.Begin(static_cast<int>(ranked_.size()));
    while (clip.Step()) {
        for (int i = clip.DisplayStart; i < clip.DisplayEnd; ++i) {
            const RankedDecision &r = ranked_[static_cast<size_t>(i)];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(i);
            if (r.index == c.session->cursor()) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, tok().SEL);
            if (row_selectable("##rank", 0)) go_to(c.session, r.index);
            ImGui::SetItemTooltip("Go to decision %zu", r.index + 1);
            ImGui::SameLine(0, 0);
            ImGui::Text("#%zu %s", r.index + 1, turn_label(m.decisions[r.index]).c_str());
            ImGui::TableNextColumn();
            seat_chip(m.decisions[r.index].decider);
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, color_of(r.policy));
            ImGui::TextUnformatted(label_of(r.policy).c_str());
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(metric_text(r.kl).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(metric_text(r.tv).c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(row_text(m, r.index, view, r.ref_argmax).c_str());
            ImGui::TableNextColumn();
            if (r.differs) {
                ImGui::PushStyleColor(ImGuiCol_Text, tok().BAD);
                ImGui::TextUnformatted(row_text(m, r.index, view, r.argmax).c_str());
                ImGui::PopStyleColor();
            } else {
                ImGui::TextDisabled("the same");
            }
            ImGui::TableNextColumn();
            if (r.differs)
                signed_value(r.shift, signed_text(r.shift, 2));
            else
                ImGui::TextDisabled(" ");
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
}

void PolicyCompareView::scan_catalog(const CompareContext &c) {
    rows_.clear();
    if (!c.settings) return;
    for (PolicyRow &r : scan_policies(run_dir_path(*c.settings), default_models_dir()))
        if (!r.hidden && !r.play.empty()) rows_.push_back(std::move(r));
    requested_at_ = -1;
}

void PolicyCompareView::draw_picker(const CompareContext &c, CompareRequest *req) {
    const char *const name = "Add checkpoints##policy_compare";
    // Opened once the tool's window has the focus (a popup closes when another window takes it): asked for first.
    // Forced open, it opens again whenever something else closed it.
    if (force_picker_ && !ImGui::IsPopupOpen(name)) open_picker_ = true;
    if (open_picker_ && !ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows)) {
        ImGui::SetWindowFocus();
    } else if (open_picker_) {
        open_picker_ = false;
        if (!force_picker_ || rows_.empty()) scan_catalog(c);
        ImGui::OpenPopup(name);
    }
    picker_shown_ = false;
    const float em = ImGui::CalcTextSize("M").x;
    ImGui::SetNextWindowSize(ImVec2(em * 64, 0), ImGuiCond_Appearing);
    if (!ImGui::BeginPopup(name)) return;
    picker_shown_ = true;
    if (c.settings) {
        EngineTables t;
        t.engine = c.engine;
        t.workdir = engine_workdir_path(*c.settings);
        t.cards_path = workbench_cache_dir() + "/policy_compare/cards-" + std::to_string(getpid()) + ".rmst";
        const double now = ImGui::GetTime();
        if (requested_at_ < 0 || now - requested_at_ >= COMPAT_EVERY_S) {
            requested_at_ = now;
            checker_.request(t, rows_);
        }
    }
    checker_.poll();
    ImGui::TextUnformatted("Policies of the runs to serve and re-evaluate this match with (both seats)");
    ImGui::TextDisabled(
        "They are added to Settings \xE2\x86\x92 Models' served policies, oldest first, so the newest "
        "becomes the reference.");
    ImGui::SetNextItemWidth(em * 20);
    ImGui::InputTextWithHint("##filter", "filter: run or name", filter_, sizeof filter_);
    const std::string filter = filter_;
    const auto lower = [](std::string s) {
        for (char &ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        return s;
    };
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter |
                                  ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit;
    if (rows_.empty()) {
        ImGui::TextDisabled("No policy with an export in the runs directory (%s) or the models directory",
            c.settings ? run_dir_path(*c.settings).c_str() : "");
    } else if (ImGui::BeginTable("##catalog", 5, flags, ImVec2(0, ImGui::GetTextLineHeightWithSpacing() * 16))) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFrameHeight());
        ImGui::TableSetupColumn("Date", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("2026-11-22").x);
        ImGui::TableSetupColumn("Policy", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Steps", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Steps 0").x);
        ImGui::TableSetupColumn(
            "Engine's tables", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("Engine's tables").x);
        ImGui::TableHeadersRow();
        for (size_t i = 0; i < rows_.size(); ++i) {
            const PolicyRow &r = rows_[i];
            if (!filter.empty() &&
                lower(r.run + " " + r.name + " " + r.archive).find(lower(filter)) == std::string::npos)
                continue;
            const CompatVerdict v = checker_.verdict(r);
            const bool bad = v.compat == PolicyCompat::INCOMPATIBLE;
            ImGui::TableNextRow();
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextColumn();
            bool on = ticked_.count(r.play) > 0 && !bad;
            ImGui::BeginDisabled(bad);
            if (ImGui::Checkbox("##tick", &on)) {
                if (on)
                    ticked_.insert(r.play);
                else
                    ticked_.erase(r.play);
            }
            ImGui::EndDisabled();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(r.trained_at > 0 ? policy_date_text(r.trained_at).c_str() : "");
            ImGui::TableNextColumn();
            const std::string what = r.latest             ? "latest \xC2\xB7 " + r.run + " " + r.name
                                     : !r.archive.empty() ? r.archive + " \xC2\xB7 " + base_name(r.play)
                                                          : r.run + "/" + r.name;
            if (bad)
                ImGui::TextDisabled("%s", what.c_str());
            else
                ImGui::TextUnformatted(what.c_str());
            ImGui::SetItemTooltip("%s", r.play.c_str());
            ImGui::TableNextColumn();
            if (r.learner_steps >= 0) ImGui::Text("%lld", static_cast<long long>(r.learner_steps));
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, bad                                    ? tok().BAD
                                                 : v.compat == PolicyCompat::COMPATIBLE ? tok().GOOD
                                                                                        : tok().DIM);
            ImGui::TextUnformatted(compat_verdict_text(v));
            ImGui::PopStyleColor();
            if (!v.why.empty()) ImGui::SetItemTooltip("%s", v.why.c_str());
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    // The ticked rows, oldest first.
    std::vector<const PolicyRow *> picked;
    for (const PolicyRow &r : rows_)
        if (ticked_.count(r.play) && checker_.verdict(r).compat != PolicyCompat::INCOMPATIBLE) picked.push_back(&r);
    std::stable_sort(picked.begin(), picked.end(),
        [](const PolicyRow *a, const PolicyRow *b) { return a->trained_at < b->trained_at; });
    const std::string serve = "Serve and re-evaluate (" + std::to_string(picked.size()) + ")";
    ImGui::BeginDisabled(picked.empty());
    if (ImGui::Button(serve.c_str())) {
        req->kind = CompareRequest::Kind::BRING_IN;
        for (const PolicyRow *r : picked) req->models.push_back(r->play);
        ticked_.clear();
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Settings \xE2\x86\x92 Models")) {
        req->kind = CompareRequest::Kind::SETTINGS;
        ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

}  // namespace rmwork
