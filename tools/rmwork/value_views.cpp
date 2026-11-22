#include "value_views.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "implot.h"
#include "theme.h"
#include "tool_widgets.h"
#include "widgets.h"

static ImU32 with_alpha(ImU32 c, int alpha);
static void dashed_polyline(ImDrawList *dl, const std::vector<ImVec2> &pts, ImU32 color, float thickness, int dash);
static void legend_sample(ImU32 color, int dash);
static double clamp_unit(double v);
static void legend_wrap(float width);

static ImU32 with_alpha(ImU32 c, int alpha) {
    return (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(alpha) << IM_COL32_A_SHIFT);
}

// A polyline through `pts`: solid (dash 0), dashed (1) or dotted (2).
static void dashed_polyline(ImDrawList *dl, const std::vector<ImVec2> &pts, ImU32 color, float thickness, int dash) {
    if (pts.size() < 2) return;
    if (dash == 0) {
        dl->AddPolyline(pts.data(), static_cast<int>(pts.size()), color, ImDrawFlags_None, thickness);
        return;
    }
    const float on = dash == 1 ? 6.0f : 2.0f, off = dash == 1 ? 4.0f : 3.0f;
    float phase = 0;  // distance into the current on + off period
    for (size_t i = 0; i + 1 < pts.size(); ++i) {
        const ImVec2 a = pts[i], b = pts[i + 1];
        const float dx = b.x - a.x, dy = b.y - a.y, len = std::sqrt(dx * dx + dy * dy);
        float t = 0;
        while (t < len) {
            const float in_period = std::fmod(phase, on + off);
            const bool drawing = in_period < on;
            const float step = std::min(len - t, drawing ? on - in_period : on + off - in_period);
            if (drawing) {
                const ImVec2 p(a.x + dx * t / len, a.y + dy * t / len);
                const ImVec2 q(a.x + dx * (t + step) / len, a.y + dy * (t + step) / len);
                dl->AddLine(p, q, color, thickness);
            }
            t += step;
            phase += step;
        }
    }
}

// A short line of the series' colour and style, as an item.
static void legend_sample(ImU32 color, int dash) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float y = p.y + ImGui::GetTextLineHeight() / 2;
    dashed_polyline(ImGui::GetWindowDrawList(), {ImVec2(p.x, y), ImVec2(p.x + 20, y)}, color, 2.0f, dash);
    ImGui::Dummy(ImVec2(22, ImGui::GetTextLineHeight()));
}

// Starts a new legend line unless an entry `width` wide fits after the last one.
static void legend_wrap(float width) {
    const float right = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;
    if (ImGui::GetCursorScreenPos().x + width > right) ImGui::NewLine();
}

static double clamp_unit(double v) {
    return std::max(-1.0, std::min(1.0, v));
}

namespace rmwork {

std::vector<GraphSeries> ValueGraphView::series_for(const ToolData &data, int view) {
    std::vector<GraphSeries> out;
    const MatchValues &v = data.values();
    for (int seat = 0; seat < 2; ++seat) {
        if (!ToolData::sees_seat(view, seat)) continue;
        int k = 0;
        for (int64_t p : v.policies) {
            const ValueSeries *s = v.find(seat, p);
            if (!s) continue;
            GraphSeries g;
            g.seat = seat;
            g.policy = p;
            const std::string pl = policy_label(p, s->version);
            g.label = seat == 0 ? "V (A) \xC2\xB7 " + pl : "\xE2\x88\x92V of B seat \xC2\xB7 " + pl;
            if (seat == 0) {
                g.color = k == 0 ? tok().S_A : tok().S_A6;
                g.dash = k == 0 ? 0 : k == 1 ? 1 : 2;
            } else {
                g.color = tok().S_B;
                g.dash = k == 0 ? 2 : 1;
            }
            out.push_back(g);
            ++k;
        }
    }
    return out;
}

std::vector<std::pair<size_t, double>> ValueGraphView::points(const ToolData &data, int seat, int64_t policy) {
    std::vector<std::pair<size_t, double>> out;
    const ValueSeries *s = data.values().find(seat, policy);
    if (!s) return out;
    for (const ValuePoint &p : s->points) out.emplace_back(p.decision, on_a_scale(seat, p.value));
    return out;
}

bool ValueGraphView::shown(int seat, int64_t policy) const {
    auto it = hidden_.find({seat, policy});
    return it == hidden_.end() || !it->second;
}

void ValueGraphView::draw_legend(const std::vector<GraphSeries> &series, bool any) {
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(2, 1));
    const float fixed = ImGui::GetFrameHeight() + 4 + 22 + 2;
    for (size_t i = 0; i < series.size(); ++i) {
        const GraphSeries &g = series[i];
        if (i) legend_wrap(fixed + ImGui::CalcTextSize(g.label.c_str()).x);
        ImGui::PushID(static_cast<int>(i));
        bool on = shown(g.seat, g.policy);
        if (ImGui::Checkbox("##on", &on)) set_shown(g.seat, g.policy, on);
        ImGui::SameLine(0, 4);
        legend_sample(g.color, g.dash);
        ImGui::SameLine(0, 2);
        ImGui::TextUnformatted(g.label.c_str());
        ImGui::SameLine(0, 16);
        ImGui::PopID();
    }
    if (!series.empty()) legend_wrap(fixed + 24 + ImGui::CalcTextSize("Life (A, B)").x);
    ImGui::Checkbox("##life", &life_);
    ImGui::SameLine(0, 4);
    legend_sample(tok().LIFE_A, 0);
    ImGui::SameLine(0, 2);
    legend_sample(tok().LIFE_B, 0);
    ImGui::SameLine(0, 2);
    ImGui::TextUnformatted("Life (A, B)");
    if (!any) {
        ImGui::SameLine(0, 16);
        legend_wrap(ImGui::CalcTextSize("Re-evaluate, or load episodes, for values").x);
        ImGui::TextDisabled("Re-evaluate, or load episodes, for values");
    }
    ImGui::PopStyleVar();
}

void ValueGraphView::draw(ToolData &data, ReplaySession &s) {
    if (!s.loaded()) return;
    const std::vector<GraphSeries> series = series_for(data, s.view());
    draw_legend(series, data.values().any());
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) dragging_ = false;
    const ImPlotSubplotFlags sub = ImPlotSubplotFlags_NoTitle | ImPlotSubplotFlags_LinkAllX |
                                   ImPlotSubplotFlags_NoResize | ImPlotSubplotFlags_NoMenus;
    float ratios[2] = {0.78f, 0.22f};
    if (!ImPlot::BeginSubplots("##value_graph", life_ ? 2 : 1, 1, ImGui::GetContentRegionAvail(), sub, ratios)) return;
    draw_values(data, s, series);
    if (life_) draw_life(s);
    ImPlot::EndSubplots();
}

// Moves the cursor to the decision under the mouse while the left button is held after a click in
// a plot.
void ValueGraphView::scrub(ReplaySession &s, bool hovered) {
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) dragging_ = true;
    if (!dragging_ || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) return;
    const double x = ImPlot::GetPlotMousePos().x;
    const long i = std::lround(x);
    const long n = static_cast<long>(s.count());
    if (n == 0) return;
    s.set_playing(false, ImGui::GetTime());
    s.go_to(static_cast<size_t>(std::max(0L, std::min(n - 1, i))));
}

void ValueGraphView::draw_values(ToolData &data, ReplaySession &s, const std::vector<GraphSeries> &series) {
    const Match &m = *s.match();
    const double n = static_cast<double>(m.decisions.size());
    const ImPlotFlags flags = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText |
                              ImPlotFlags_NoLegend | ImPlotFlags_NoFrame;
    if (!ImPlot::BeginPlot("##v", ImVec2(-1, 0), flags)) return;
    static const double v_ticks[] = {-1.0, -0.5, 0.0, 0.5, 1.0};
    static const char *const v_labels[] = {
        "\xE2\x88\x92"
        "1",
        "\xE2\x88\x92"
        "0.5",
        "0", "+0.5", "+1"};
    ImPlot::SetupAxis(
        ImAxis_X1, nullptr, ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_Lock);
    ImPlot::SetupAxis(ImAxis_Y1, "V (A)", ImPlotAxisFlags_Lock);
    ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, n + 0.5, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, -1.12, 1.12, ImPlotCond_Always);
    ImPlot::SetupAxisTicks(ImAxis_Y1, v_ticks, 5, v_labels);
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    ImPlot::PushPlotClipRect();
    const ImVec2 top_left = ImPlot::GetPlotPos(), size = ImPlot::GetPlotSize();
    const float y0 = top_left.y, y1 = top_left.y + size.y;
    // Turn bands, every other one shaded, labelled where they are wide enough.
    const std::vector<TurnBand> bands = turn_bands(m);
    for (size_t k = 0; k < bands.size(); ++k) {
        const float xa = ImPlot::PlotToPixels(static_cast<double>(bands[k].first) - 0.5, 0).x;
        const float xb = ImPlot::PlotToPixels(static_cast<double>(bands[k].last) + 0.5, 0).x;
        if (k % 2) dl->AddRectFilled(ImVec2(xa, y0), ImVec2(xb, y1), with_alpha(tok().TEXT, 9));
        const ImVec2 ts = ImGui::CalcTextSize(bands[k].label.c_str());
        if (xb - xa > ts.x + 4) dl->AddText(ImVec2((xa + xb - ts.x) / 2, y0 + 2), tok().DIM, bands[k].label.c_str());
    }
    // The zero line, the games' boundaries and results.
    const float zero = ImPlot::PlotToPixels(0, 0).y;
    dl->AddLine(ImVec2(top_left.x, zero), ImVec2(top_left.x + size.x, zero), tok().LINE);
    std::vector<GraphSeries> shown_series;
    for (const GraphSeries &g : series)
        if (shown(g.seat, g.policy)) shown_series.push_back(g);
    for (size_t g = 0; g < m.games.size(); ++g) {
        const GameSpan &span = m.games[g];
        const float xa = ImPlot::PlotToPixels(static_cast<double>(span.first) - 0.5, 0).x;
        const float xb = ImPlot::PlotToPixels(static_cast<double>(span.first + span.count) - 0.5, 0).x;
        if (g > 0) dashed_polyline(dl, {ImVec2(xa, y0), ImVec2(xa, y1)}, tok().DIM, 1.0f, 2);
        if (span.sideboarding) {
            dl->AddRectFilled(ImVec2(xa, y0), ImVec2(xb, y1), with_alpha(tok().DIM, 40));
        } else if (m.games.size() > 1) {
            const ImVec2 ts = ImGui::CalcTextSize(span.label.c_str());
            if (xb - xa > ts.x + 8) dl->AddText(ImVec2(xa + 4, y1 - ts.y - 2), tok().DIM, span.label.c_str());
        }
        if (span.sideboarding || span.count == 0) continue;
        if (static_cast<size_t>(span.game) >= m.result.game_winners.size()) continue;
        const double z = game_result(m, span.game, 0);
        const double xz = static_cast<double>(span.first + span.count) - 0.5;
        const ImVec2 pz = ImPlot::PlotToPixels(xz, z);
        const ImU32 zc = z > 0 ? tok().GOOD : z < 0 ? tok().BAD : tok().DIM;
        // A dashed line from the first shown series' last value in the game to the result.
        for (const GraphSeries &gs : shown_series) {
            const std::vector<std::pair<size_t, double>> pts = points(data, gs.seat, gs.policy);
            const std::pair<size_t, double> *last = nullptr;
            for (const auto &p : pts)
                if (p.first >= span.first && p.first < span.first + span.count) last = &p;
            if (!last) continue;
            dashed_polyline(
                dl, {ImPlot::PlotToPixels(static_cast<double>(last->first), last->second), pz}, zc, 1.2f, 1);
            break;
        }
        dl->AddRectFilled(ImVec2(pz.x - 3, pz.y - 3), ImVec2(pz.x + 3, pz.y + 3), zc);
        const std::string label = "z " + signed_text(z, 0);
        const ImVec2 ts = ImGui::CalcTextSize(label.c_str());
        const float lx = pz.x + 6 + ts.x > top_left.x + size.x ? pz.x - 6 - ts.x : pz.x + 6;
        dl->AddText(ImVec2(lx, z > 0 ? pz.y + 2 : pz.y - ts.y - 2), zc, label.c_str());
    }
    // The series.
    for (const GraphSeries &g : shown_series) {
        const std::vector<std::pair<size_t, double>> pts = points(data, g.seat, g.policy);
        std::vector<ImVec2> px;
        // A line per game: no line across a game boundary.
        int game = -1;
        for (const auto &p : pts) {
            const int pg = m.decisions[p.first].game;
            if (pg != game && !px.empty()) {
                dashed_polyline(dl, px, g.color, 1.6f, g.dash);
                px.clear();
            }
            game = pg;
            px.push_back(ImPlot::PlotToPixels(static_cast<double>(p.first), clamp_unit(p.second)));
        }
        dashed_polyline(dl, px, g.color, 1.6f, g.dash);
        for (const auto &p : pts) {
            const ImVec2 q = ImPlot::PlotToPixels(static_cast<double>(p.first), clamp_unit(p.second));
            const float r = p.first == s.cursor() ? 4.0f : 1.6f;
            dl->AddRectFilled(ImVec2(q.x - r, q.y - r), ImVec2(q.x + r, q.y + r), g.color);
        }
    }
    if (!data.values().any())
        ImPlot::PlotText(m.live ? "Model's Read is off or not connected: no values"
                         : m.episode_files.empty()
                             ? "No evaluations: Analysis \xE2\x86\x92 Re-evaluate, or load the match's episodes"
                             : "The episodes carry no evaluations: Analysis \xE2\x86\x92 Re-evaluate",
            n / 2, 0.5);
    // The cursor.
    const float cx = ImPlot::PlotToPixels(static_cast<double>(s.cursor()), 0).x;
    dashed_polyline(dl, {ImVec2(cx, y0), ImVec2(cx, y1)}, tok().TEXT, 1.0f, 1);
    ImPlot::PopPlotClipRect();
    const bool hovered = ImPlot::IsPlotHovered();
    if (hovered) {
        const long i = std::lround(ImPlot::GetPlotMousePos().x);
        if (i >= 0 && i < static_cast<long>(m.decisions.size())) {
            const DecisionInfo &d = m.decisions[static_cast<size_t>(i)];
            ImGui::BeginTooltip();
            ImGui::Text("#%ld \xC2\xB7 %s \xC2\xB7 %s decides", i + 1, turn_label(d).c_str(), seat_name(d.decider));
            for (const GraphSeries &g : shown_series)
                for (const auto &p : points(data, g.seat, g.policy))
                    if (p.first == static_cast<size_t>(i))
                        ImGui::Text("%s  %s", g.label.c_str(), signed_text(p.second).c_str());
            ImGui::EndTooltip();
        }
    }
    scrub(s, hovered);
    ImPlot::EndPlot();
}

void ValueGraphView::draw_life(ReplaySession &s) {
    const Match &m = *s.match();
    const size_t n = m.decisions.size();
    int64_t top = 20;
    for (const DecisionInfo &d : m.decisions) top = std::max(top, std::max(d.life[0], d.life[1]));
    const ImPlotFlags flags = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText |
                              ImPlotFlags_NoLegend | ImPlotFlags_NoFrame;
    if (!ImPlot::BeginPlot("##life", ImVec2(-1, 0), flags)) return;
    const double life_ticks[] = {0.0, static_cast<double>(top)};
    ImPlot::SetupAxis(
        ImAxis_X1, nullptr, ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoGridLines | ImPlotAxisFlags_Lock);
    ImPlot::SetupAxis(ImAxis_Y1, "Life", ImPlotAxisFlags_Lock);
    ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, static_cast<double>(n) + 0.5, ImPlotCond_Always);
    ImPlot::SetupAxisLimits(ImAxis_Y1, -1, static_cast<double>(top) + 2, ImPlotCond_Always);
    ImPlot::SetupAxisTicks(ImAxis_Y1, life_ticks, 2);
    ImDrawList *dl = ImPlot::GetPlotDrawList();
    ImPlot::PushPlotClipRect();
    for (int seat = 0; seat < 2; ++seat) {
        std::vector<ImVec2> px;
        for (size_t i = 0; i < n; ++i) {
            const double v = static_cast<double>(m.decisions[i].life[static_cast<size_t>(seat)]);
            px.push_back(ImPlot::PlotToPixels(static_cast<double>(i) - 0.5, v));
            px.push_back(ImPlot::PlotToPixels(static_cast<double>(i) + 0.5, v));
        }
        dashed_polyline(dl, px, seat == 0 ? tok().LIFE_A : tok().LIFE_B, 1.4f, 0);
    }
    if (s.cursor() < n) {
        const DecisionInfo &d = m.decisions[s.cursor()];
        const float cx = ImPlot::PlotToPixels(static_cast<double>(s.cursor()), 0).x;
        const ImVec2 pos = ImPlot::GetPlotPos(), size = ImPlot::GetPlotSize();
        dashed_polyline(dl, {ImVec2(cx, pos.y), ImVec2(cx, pos.y + size.y)}, tok().TEXT, 1.0f, 1);
        for (int seat = 0; seat < 2; ++seat) {
            const std::string t =
                std::string(seat == 0 ? "A " : "B ") + std::to_string(d.life[static_cast<size_t>(seat)]);
            const ImVec2 q = ImPlot::PlotToPixels(
                static_cast<double>(s.cursor()), static_cast<double>(d.life[static_cast<size_t>(seat)]));
            const ImVec2 ts = ImGui::CalcTextSize(t.c_str());
            const float x = cx + 6 + ts.x > pos.x + size.x ? cx - 6 - ts.x : cx + 6;
            dl->AddText(ImVec2(x, seat == 0 ? q.y - ts.y : q.y), seat == 0 ? tok().LIFE_A : tok().LIFE_B, t.c_str());
        }
    }
    ImPlot::PopPlotClipRect();
    scrub(s, ImPlot::IsPlotHovered());
    ImPlot::EndPlot();
}

void SwingsView::draw(ToolData &data, ReplaySession &s) {
    const float width = 46.0f * ImGui::CalcTextSize("M").x;
    if (!s.loaded()) return;
    const Match &m = *s.match();
    if (!data.values().any()) {
        centered_text_block("No evaluations",
            m.episode_files.empty()
                ? "Swings come from the evaluations: re-evaluate the replay through the inference server (Analysis "
                  "\xE2\x86\x92 Re-evaluate), or open the match's episodes (File \xE2\x86\x92 Open Trajectory "
                  "Episode), or a run's log, whose episodes join by name."
                : "The joined episodes carry no evaluations (only an nn: agent records them): re-evaluate the replay "
                  "through the inference server (Analysis \xE2\x86\x92 Re-evaluate).",
            width);
        return;
    }
    const int view = s.view();
    const int64_t policy = data.shown_policy();
    const std::string pl = policy_label(policy, data.values().versions.at(policy));
    const std::vector<Swing> swings = data.swings(view, ROWS);
    const std::string who = view == VIEW_OMNI ? std::string("a seat's") : std::string(seat_name(view)) + "'s";
    ImGui::TextDisabled("Largest changes in %s V between its decisions (%s)", who.c_str(), pl.c_str());
    const bool omni = view == VIEW_OMNI;
    const float rows_h =
        (ImGui::GetTextLineHeightWithSpacing() + 2) * static_cast<float>(std::min<size_t>(swings.size(), ROWS) + 1) + 8;
    if (begin_grid("##swings", omni ? 6 : 5, ImVec2(0, omni ? rows_h : 0), true)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn(
            "Steps", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("#000\xE2\x86\x92#000").x);
        if (omni) ImGui::TableSetupColumn("Seat", ImGuiTableColumnFlags_WidthFixed, seat_chip_width() + 4);
        ImGui::TableSetupColumn("Turn", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("G2 T00").x);
        ImGui::TableSetupColumn(
            "V", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("+0.00 \xE2\x86\x92 z +1").x);
        ImGui::TableSetupColumn("\xCE\x94V", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("+0.00").x);
        ImGui::TableSetupColumn("Between", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (size_t k = 0; k < swings.size(); ++k) {
            const Swing &w = swings[k];
            ImGui::TableNextRow();
            if (swing_target(w) == s.cursor()) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, tok().SEL);
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(k));
            const std::string steps = "#" + std::to_string(w.from + 1) + "\xE2\x86\x92" +
                                      (w.to_result ? std::string("end") : "#" + std::to_string(w.to + 1));
            if (ImGui::Selectable(steps.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
                s.set_playing(false, ImGui::GetTime());
                s.go_to(swing_target(w));
            }
            ImGui::PopID();
            if (omni) {
                ImGui::TableNextColumn();
                seat_chip(w.seat);
            }
            ImGui::TableNextColumn();
            const DecisionInfo &d = m.decisions[w.from];
            ImGui::Text("%s%s", m.bo3() ? ("G" + std::to_string(d.game + 1) + " ").c_str() : "", turn_label(d).c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%s \xE2\x86\x92 %s", signed_text(w.v0).c_str(),
                w.to_result ? ("z " + signed_text(w.v1, 0)).c_str() : signed_text(w.v1).c_str());
            ImGui::TableNextColumn();
            signed_value(w.dv, signed_text(w.dv));
            ImGui::TableNextColumn();
            const std::string between = between_text(m, view, w.from, w.to_result, w.to, 3);
            ImGui::TextUnformatted(between.empty() ? "\xE2\x80\x94" : between.c_str());
            if (!between.empty())
                ImGui::SetItemTooltip("%s", between_text(m, view, w.from, w.to_result, w.to, 40).c_str());
        }
        ImGui::EndTable();
    }
    if (!omni) {
        return;
    }
    ImGui::Spacing();
    ImGui::TextDisabled("Seat disagreement: A's V against \xE2\x88\x92V of B's seat (%s)", pl.c_str());
    const std::vector<Disagreement> gaps = top_disagreements(m, data.values(), policy, ROWS);
    if (gaps.empty()) {
        ImGui::TextDisabled("Both seats are not evaluated by this policy at adjacent decisions.");
        return;
    }
    if (begin_grid("##gaps", 5,
            ImVec2(0, std::max(80.0f, ImGui::GetContentRegionAvail().y - ImGui::GetTextLineHeightWithSpacing())),
            true)) {
        ImGui::TableSetupScrollFreeze(0, 1);
        ImGui::TableSetupColumn("Steps", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("#000/#000").x);
        ImGui::TableSetupColumn("Turn", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("G2 T00").x);
        ImGui::TableSetupColumn("V (A)", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("+0.00 ").x);
        ImGui::TableSetupColumn("\xE2\x88\x92V (B)", ImGuiTableColumnFlags_WidthFixed, ImGui::CalcTextSize("+0.00 ").x);
        ImGui::TableSetupColumn("Gap", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableHeadersRow();
        for (size_t k = 0; k < gaps.size(); ++k) {
            const Disagreement &g = gaps[k];
            const size_t at = std::min(g.a, g.b);
            ImGui::TableNextRow();
            if (at == s.cursor()) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, tok().SEL);
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(k));
            const std::string steps = "#" + std::to_string(g.a + 1) + "/#" + std::to_string(g.b + 1);
            if (ImGui::Selectable(steps.c_str(), false, ImGuiSelectableFlags_SpanAllColumns)) {
                s.set_playing(false, ImGui::GetTime());
                s.go_to(at);
            }
            ImGui::PopID();
            ImGui::TableNextColumn();
            const DecisionInfo &d = m.decisions[at];
            ImGui::Text("%s%s", m.bo3() ? ("G" + std::to_string(d.game + 1) + " ").c_str() : "", turn_label(d).c_str());
            ImGui::TableNextColumn();
            signed_value(g.va, signed_text(g.va));
            ImGui::TableNextColumn();
            signed_value(g.vb_on_a, signed_text(g.vb_on_a));
            ImGui::TableNextColumn();
            char buf[16];
            std::snprintf(buf, sizeof buf, "%.2f", g.gap);
            ImGui::PushStyleColor(ImGuiCol_Text, tok().LIFE_B);
            ImGui::TextUnformatted(buf);
            ImGui::PopStyleColor();
        }
        ImGui::EndTable();
    }
    ImGui::TextDisabled("Click a row to go to that decision.");
}

int ModelReadView::seat_of(const ToolData &data) {
    const DecisionFrame &f = data.frame();
    if (f.view != VIEW_OMNI) return f.view;
    return f.info ? f.info->decider : 0;
}

std::vector<double> ModelReadView::history(const ToolData &data, size_t *last_decision) {
    std::vector<double> out;
    *last_decision = 0;
    const ValueSeries *s = data.values().find(seat_of(data), data.shown_policy());
    if (!s) return out;
    for (const ValuePoint &p : s->points) {
        if (p.decision > data.frame().index) break;
        out.push_back(p.value);
        *last_decision = p.decision;
    }
    return out;
}

void ModelReadView::draw(ToolData &data, ReplaySession &s) {
    (void)s;
    if (!data.ok()) {
        centered_text_block("Frame not readable", data.error().c_str(), 46.0f * ImGui::CalcTextSize("M").x);
        return;
    }
    const DecisionFrame &f = data.frame();
    const int seat = seat_of(data);
    const bool deciding = f.info && f.info->decider == seat;
    size_t last = 0;
    const std::vector<double> hist = history(data, &last);
    const int64_t policy = data.shown_policy();
    const ImGuiTableFlags flags = ImGuiTableFlags_SizingFixedFit;
    if (ImGui::BeginTable("##model", 2, flags)) {
        ImGui::TableSetupColumn("k", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("v", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("Seat");
        ImGui::TableNextColumn();
        seat_chip(seat);
        ImGui::SameLine();
        ImGui::Text("%s%s", seat_name(seat), deciding ? " \xC2\xB7 deciding" : "");
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("V");
        ImGui::TableNextColumn();
        if (hist.empty()) {
            ImGui::TextDisabled("%s", data.values().any() ? "not evaluated yet" : "no evaluations");
        } else {
            signed_value(hist.back(), signed_text(hist.back(), 3));
            if (last != f.index) {
                ImGui::SameLine();
                ImGui::TextDisabled("(at decision %zu)", last + 1);
            }
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("Would play");
        ImGui::TableNextColumn();
        const EpisodeStepInfo *e = f.visible_episode();
        const featurize::PolicyEvalRecord *r = e && deciding ? find_policy(*e, policy) : nullptr;
        const MenuModel &menu = data.models().menu;
        if (r && r->log_pi.size() == menu.rows.size() && !menu.rows.empty()) {
            const std::vector<double> pi = policy_pi(*r);
            const size_t best = static_cast<size_t>(std::max_element(pi.begin(), pi.end()) - pi.begin());
            char buf[16];
            std::snprintf(buf, sizeof buf, " (%.2f)", pi[best]);
            ImGui::TextWrapped("%s%s%s", menu.rows[best].text.c_str(), buf,
                static_cast<int>(best) == menu.chosen ? " \xE2\x80\x94 chosen" : "");
        } else {
            ImGui::TextDisabled(
                "%s", deciding ? "no evaluation of this decision"
                               : (std::string("\xE2\x80\x94 (") + seat_name(f.info->decider) + " decides)").c_str());
        }
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextDisabled("Model");
        ImGui::TableNextColumn();
        // Play's evaluations are Model's Read's of the human's seat, not the seat's agent.
        // A re-evaluated policy is not the seat's agent.
        const std::string agent = f.match->live                           ? "Model's Read"
                                  : policy >= 0 && is_reevaluated(policy) ? "Re-evaluation"
                                                                          : agent_label(*f.match, seat);
        std::string model = agent.empty() ? std::string("\xE2\x80\x94") : agent;
        if (policy >= 0) model += " \xC2\xB7 " + policy_label(policy, data.values().versions.at(policy));
        if (f.match->live && policy < 0) model += " off or not connected";
        ImGui::TextWrapped("%s", model.c_str());
        ImGui::EndTable();
    }
    sparkline("##spark", hist, ImVec2(0, std::max(40.0f, std::min(90.0f, ImGui::GetContentRegionAvail().y - 4))),
        seat == 0 ? tok().S_A : tok().LIFE_B);
}

}  // namespace rmwork
