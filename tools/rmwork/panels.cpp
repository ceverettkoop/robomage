#include "panels.h"

#include <algorithm>

#include "implot.h"
#include "theme.h"
#include "widgets.h"

static void legend_entry(const char *label, ImU32 color, bool dashed);
static void draw_value_graph(const char *note);
static std::string loaded_body(const rmwork::ReplaySession &s, rmwork::Tool t);
static bool replay_tool(rmwork::Tool t);
static void draw_new_game_prompt(bool *pressed, float width);

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

// V from −1 to +1 above a life strip from 0 to 20, decisions along the shared x axis; `note` in the
// middle of the V plot.
static void draw_value_graph(const char *note) {
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
        ImPlot::PlotText(note, 0.5, 0.5);
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

// "<deck> vs <deck>, decision n of N (<game, turn, step>), viewer …", then what the tool will show.
static std::string loaded_body(const rmwork::ReplaySession &s, rmwork::Tool t) {
    const rmwork::Match &m = *s.match();
    const rmwork::DecisionInfo *d = s.current();
    std::string out = m.title() + ": decision " + std::to_string(s.cursor() + 1) + " of " + std::to_string(s.count()) +
                      (d ? " (" + d->header + ")" : "") + ", " + rmwork::view_name(s.view()) + "'s frames.\n\n" +
                      rmwork::tool_info(t).blurb;
    return out;
}

// One of the tools ReplayTools draws.
static bool replay_tool(rmwork::Tool t) {
    return t != rmwork::Tool::TABLE && t != rmwork::Tool::LOG && t != rmwork::Tool::MODELS;
}

// Play with no game: what to do, and a New Game button under it.
static void draw_new_game_prompt(bool *pressed, float width) {
    const char *body =
        "Play a game against a model, the scripted agent, explore or random: you choose with the Choices tool "
        "(click a row, or \xE2\x86\x91 / \xE2\x86\x93 and Enter), F2 passes to the end of the turn and F6 skips it. "
        "Game \xE2\x86\x92 New Game vs Model (Ctrl+N) opens the same dialog.";
    rmwork::centered_text_block("No game", body, width);
    ImGui::Spacing();
    const char *label = "New Game\xE2\x80\xA6";
    const float bw = ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2;
    ImGui::SetCursorPosX(std::max(0.0f, (ImGui::GetWindowWidth() - bw) / 2));
    if (ImGui::Button(label) && pressed) *pressed = true;
}

namespace rmwork {

void draw_tool_body(Tool t, const ToolContext &ctx) {
    const bool loaded = ctx.replay && ctx.replay->loaded();
    if (loaded && replay_tool(t) && ctx.tools) {
        ctx.tools->draw(t, *ctx.replay, ctx.links);
        return;
    }
    if (t == Tool::GRAPH) {
        draw_value_graph(ctx.mode == Mode::PLAY ? "No game: Game \xE2\x86\x92 New Game vs Model" : "No game loaded");
        return;
    }
    if (loaded && t == Tool::LOG && ctx.log) {
        ctx.log->draw(*ctx.replay);
        return;
    }
    if (loaded && t == Tool::TABLE && ctx.table && ctx.settings) {
        TableDraw d;
        d.replay = ctx.replay;
        d.background = &ctx.settings->table;
        d.links = ctx.links;
        d.magnify = ctx.settings->keys[action_index(Action::MAGNIFY)];
        d.play = ctx.mode == Mode::PLAY;
        d.stops = ctx.stops;
        d.answer = ctx.choosing;
        d.pass = ctx.settings->keys[action_index(Action::PLAY_PASS)];
        d.to_end_of_turn = ctx.settings->keys[action_index(Action::PLAY_TO_END_OF_TURN)];
        d.tints = ctx.tints;
        d.attr_layer = ctx.attr_layer;
        ctx.table->draw(d);
        return;
    }
    const float width = 46.0f * ImGui::CalcTextSize("M").x;
    if (loaded) {
        centered_text_block("Game loaded", loaded_body(*ctx.replay, t).c_str(), width);
    } else if (!ctx.loading.empty()) {
        centered_text_block(ctx.loading.c_str(), tool_info(t).blurb, width);
    } else if (ctx.mode == Mode::PLAY && t == Tool::TABLE) {
        draw_new_game_prompt(ctx.new_game, width);
    } else {
        centered_text_block("No game loaded", tool_info(t).blurb, width);
    }
}

}  // namespace rmwork
