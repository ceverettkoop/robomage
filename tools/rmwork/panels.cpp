#include "panels.h"

#include "implot.h"
#include "theme.h"
#include "widgets.h"

static void legend_entry(const char *label, ImU32 color, bool dashed);
static void draw_value_graph();

// A short line in the series' colour and its label, on the current line.
static void legend_entry(const char *label, ImU32 color, bool dashed) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float y = p.y + ImGui::GetTextLineHeight() / 2;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    if (dashed) {
        for (float x = p.x; x < p.x + 18; x += 5) dl->AddLine(ImVec2(x, y), ImVec2(x + 3, y), color, 2.0f);
    } else {
        dl->AddLine(ImVec2(p.x, y), ImVec2(p.x + 18, y), color, 2.0f);
    }
    ImGui::Dummy(ImVec2(22, ImGui::GetTextLineHeight()));
    ImGui::SameLine(0, 0);
    ImGui::TextUnformatted(label);
    ImGui::SameLine(0, 14);
}

// V from −1 to +1 above a life strip from 0 to 20, decisions along the shared x axis.
static void draw_value_graph() {
    using namespace rmwork;
    legend_entry("V (A)", tok().S_A, false);
    legend_entry("V (A, re-eval)", tok().S_A6, true);
    legend_entry("−V of B seat", tok().S_B, true);
    legend_entry("Life A", tok().LIFE_A, false);
    legend_entry("Life B", tok().LIFE_B, false);
    ImGui::NewLine();
    float ratios[2] = {0.78f, 0.22f};
    const ImPlotSubplotFlags sub_flags = ImPlotSubplotFlags_NoTitle | ImPlotSubplotFlags_LinkAllX |
                                         ImPlotSubplotFlags_NoResize | ImPlotSubplotFlags_NoMenus;
    if (!ImPlot::BeginSubplots("##value_graph", 2, 1, ImGui::GetContentRegionAvail(), sub_flags, ratios)) return;
    const ImPlotFlags flags = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText |
                              ImPlotFlags_NoLegend | ImPlotFlags_NoInputs;
    if (ImPlot::BeginPlot("##v", ImVec2(-1, 0), flags)) {
        static const double v_ticks[] = {-1.0, -0.5, 0.0, 0.5, 1.0};
        static const char *const v_labels[] = {"−1", "−0.5", "0", "+0.5", "+1"};
        ImPlot::SetupAxis(ImAxis_X1, nullptr, ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, "V (A)");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, -1.08, 1.08, ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_Y1, v_ticks, 5, v_labels);
        ImPlot::PlotText("No game loaded", 0.5, 0.5);
        ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("##life", ImVec2(-1, 0), flags)) {
        static const double life_ticks[] = {0.0, 20.0};
        ImPlot::SetupAxis(ImAxis_X1, "decision", ImPlotAxisFlags_NoTickLabels);
        ImPlot::SetupAxis(ImAxis_Y1, "Life");
        ImPlot::SetupAxisLimits(ImAxis_X1, 0, 1, ImPlotCond_Always);
        ImPlot::SetupAxisLimits(ImAxis_Y1, 0, 20, ImPlotCond_Always);
        ImPlot::SetupAxisTicks(ImAxis_Y1, life_ticks, 2);
        ImPlot::EndPlot();
    }
    ImPlot::EndSubplots();
}

namespace rmwork {

void draw_tool_body(Tool t) {
    if (t == Tool::GRAPH) {
        draw_value_graph();
        return;
    }
    centered_text_block("No game loaded", tool_info(t).blurb, 46.0f * ImGui::CalcTextSize("M").x);
}

}  // namespace rmwork
