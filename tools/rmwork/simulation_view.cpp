#include "simulation_view.h"

#include <algorithm>
#include <cstdio>

#include "imgui.h"
#include "modal.h"
#include "start_data.h"
#include "theme.h"
#include "tool_widgets.h"

namespace {

const char *const WINDOW = "Simulations###rmwork.simulations";
// The past simulations listed.
const size_t PAST_MAX = 30;

}  // namespace

static std::string base_name(const std::string &path);
static std::string percent(double share);
static ImU32 state_color(const rmwork::SimMatch &m);
static std::string seconds_text(double s);

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// "60%".
static std::string percent(double share) {
    char buf[16];
    std::snprintf(buf, sizeof buf, "%.0f%%", share * 100);
    return buf;
}

// A match's result colour: running in the accent, a failure in red, cancelled and waiting dim.
static ImU32 state_color(const rmwork::SimMatch &m) {
    switch (m.state) {
        case rmwork::SimState::RUNNING:
            return rmwork::tok().ACCENT;
        case rmwork::SimState::FAILED:
            return rmwork::tok().BAD;
        case rmwork::SimState::WAITING:
        case rmwork::SimState::CANCELLED:
            return rmwork::tok().DIM;
        case rmwork::SimState::FINISHED:
            break;
    }
    return rmwork::tok().TEXT;
}

static std::string seconds_text(double s) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f s", s);
    return buf;
}

namespace rmwork {

void SimulationView::open(const std::string &root) {
    root_ = root;
    open_ = true;
    appearing_ = true;
    refresh();
}

void SimulationView::refresh() {
    past_ = list_simulations(root_, PAST_MAX);
}

std::vector<std::string> SimulationView::summary_lines(const Simulation &sim) {
    std::vector<std::string> out;
    const SimTally t = tally_simulation(sim.matches());
    const SimConfig &c = sim.config();
    for (int e = 0; e < 2; ++e)
        out.push_back("Entrant " + std::to_string(e + 1) + " (" + sim_entrant_text(c.entrants[static_cast<size_t>(e)]) +
                      "): " + std::to_string(t.wins[e]) + " matches, " + std::to_string(t.game_wins[e]) + " games");
    out.push_back(std::to_string(t.draws) + " drawn, " + std::to_string(t.failed) + " failed, " +
                  std::to_string(t.cancelled) + " cancelled");
    if (t.finished == 0) return out;
    if (c.alternate) {
        out.push_back("Entrant 1 scores " + percent(t.share) + " of " + std::to_string(t.finished) +
                      " finished matches (95% interval " + percent(t.low) + "\xE2\x80\x93" + percent(t.high) +
                      "; a draw counts half)");
    } else {
        out.push_back("Seat A won " + std::to_string(t.seat_wins[0]) + ", seat B " + std::to_string(t.seat_wins[1]) +
                      " (the seats do not alternate: a seat's advantage is in these numbers)");
    }
    return out;
}

SimulationRequest SimulationView::draw(const Simulation &sim) {
    SimulationRequest req;
    if (!open_) return req;
    if (appearing_) {
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(
            ImVec2(vp->WorkPos.x + vp->WorkSize.x / 2, vp->WorkPos.y + vp->WorkSize.y / 2), ImGuiCond_Always,
            ImVec2(0.5f, 0.5f));
        const float k = em_scale();
        ImGui::SetNextWindowSize(
            ImVec2(std::min(1040 * k, vp->WorkSize.x - 40), std::min(600 * k, vp->WorkSize.y - 40)), ImGuiCond_Always);
        ImGui::SetNextWindowFocus();
        appearing_ = false;
    }
    if (ImGui::Begin(WINDOW, &open_, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
                                         ImGuiWindowFlags_NoCollapse)) {
        draw_header(sim, &req);
        if (sim.loaded()) {
            draw_summary(sim);
            draw_matches(sim, &req);
        } else {
            ImGui::Spacing();
            note("No simulation open: New\xE2\x80\xA6 runs one, and a past one opens from the list.");
        }
    }
    ImGui::End();
    return req;
}

// The past simulations, New…, Open Folder…, Cancel; the entrants and settings; the progress.
void SimulationView::draw_header(const Simulation &sim, SimulationRequest *req) {
    const std::string current = sim.loaded() ? base_name(sim.dir()) : "(no simulation open)";
    ImGui::SetNextItemWidth(std::max(20 * char_width(), ImGui::GetContentRegionAvail().x - 36 * char_width()));
    if (ImGui::BeginCombo("##past", current.c_str(), ImGuiComboFlags_HeightLarge)) {
        for (size_t i = 0; i < past_.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::Selectable(past_[i].label.c_str(), past_[i].dir == sim.dir())) {
                req->kind = SimulationRequest::Kind::OPEN_PAST;
                req->path = past_[i].dir;
            }
            ImGui::SetItemTooltip("%s\n%d of %d matches finished", past_[i].dir.c_str(), past_[i].finished,
                past_[i].matches);
            ImGui::PopID();
        }
        if (past_.empty()) ImGui::TextDisabled("No simulations yet under %s", display_path(root_).c_str());
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("The past simulations, newest first");
    ImGui::SameLine();
    if (ImGui::Button("New\xE2\x80\xA6")) req->kind = SimulationRequest::Kind::NEW;
    ImGui::SameLine();
    if (ImGui::Button("Open Folder\xE2\x80\xA6")) req->kind = SimulationRequest::Kind::PICK;
    ImGui::SetItemTooltip("A simulation's directory (its simulation.json)");
    if (sim.running()) {
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) req->kind = SimulationRequest::Kind::CANCEL;
        ImGui::SetItemTooltip("Kill the running engines; the finished matches' logs stay");
    }
    if (!sim.loaded()) return;
    const SimConfig &c = sim.config();
    ImGui::Spacing();
    ImGui::TextUnformatted(
        (sim_entrant_text(c.entrants[0]) + "   vs   " + sim_entrant_text(c.entrants[1])).c_str());
    const std::string line = std::string(c.bo3 ? "best of three" : "best of one") + " \xC2\xB7 " +
                             std::to_string(c.matches) + (c.matches == 1 ? " match" : " matches") + " from seed " +
                             std::to_string(c.seed) + (c.alternate ? " \xC2\xB7 seats alternate" : "") +
                             (c.auto_mana ? "" : " \xC2\xB7 --no-auto-mana") + " \xC2\xB7 " +
                             std::to_string(c.jobs) + (c.jobs == 1 ? " job" : " jobs") + " \xC2\xB7 " +
                             base_name(sim.engine()) + " \xC2\xB7 " + display_path(sim.dir());
    ImGui::PushStyleColor(ImGuiCol_Text, tok().DIM);
    ImGui::TextWrapped("%s", line.c_str());
    ImGui::PopStyleColor();
    ImGui::SetItemTooltip("%s", sim.summary_path().c_str());
    ImGui::PushStyleColor(ImGuiCol_PlotHistogram, tok().ACCENT);
    ImGui::ProgressBar(static_cast<float>(sim.fraction()), ImVec2(-1, 0), sim.progress_text().c_str());
    ImGui::PopStyleColor();
}

void SimulationView::draw_summary(const Simulation &sim) {
    const SimTally t = tally_simulation(sim.matches());
    const SimConfig &c = sim.config();
    const std::vector<std::string> lines = summary_lines(sim);
    if (begin_grid("##sim_summary", 4)) {
        ImGui::TableSetupColumn("Entrant", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("Matches won");
        ImGui::TableSetupColumn("Games won");
        ImGui::TableSetupColumn("Share");
        ImGui::TableHeadersRow();
        for (int e = 0; e < 2; ++e) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("%d  %s", e + 1, sim_entrant_text(c.entrants[static_cast<size_t>(e)]).c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%d", t.wins[e]);
            ImGui::TableNextColumn();
            ImGui::Text("%d", t.game_wins[e]);
            ImGui::TableNextColumn();
            if (t.finished > 0)
                ImGui::TextUnformatted(percent(e == 0 ? t.share : 1 - t.share).c_str());
            else
                ImGui::TextDisabled("\xE2\x80\xA6");
        }
        ImGui::EndTable();
    }
    for (size_t i = 2; i < lines.size(); ++i) {
        ImGui::PushStyleColor(ImGuiCol_Text, i == 2 ? tok().DIM : tok().TEXT);
        ImGui::TextWrapped("%s", lines[i].c_str());
        ImGui::PopStyleColor();
    }
    ImGui::Spacing();
}

void SimulationView::draw_matches(const Simulation &sim, SimulationRequest *req) {
    const std::vector<SimMatch> &matches = sim.matches();
    const SimConfig &c = sim.config();
    if (!begin_grid("##sim_matches", 8, ImVec2(0, ImGui::GetContentRegionAvail().y), true)) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    const float cw = char_width();
    ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 4 * cw);
    ImGui::TableSetupColumn("Seed", ImGuiTableColumnFlags_WidthFixed, 8 * cw);
    ImGui::TableSetupColumn("Seat A", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Seat B", ImGuiTableColumnFlags_WidthStretch);
    ImGui::TableSetupColumn("Result", ImGuiTableColumnFlags_WidthFixed, 11 * cw);
    ImGui::TableSetupColumn("Games", ImGuiTableColumnFlags_WidthFixed, 6 * cw);
    ImGui::TableSetupColumn("Decisions", ImGuiTableColumnFlags_WidthFixed, 10 * cw);
    ImGui::TableSetupColumn("Time", ImGuiTableColumnFlags_WidthFixed, 8 * cw);
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(matches.size()));
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const SimMatch &m = matches[static_cast<size_t>(i)];
            ImGui::TableNextRow();
            ImGui::PushID(i);
            ImGui::TableNextColumn();
            const bool openable = m.state == SimState::FINISHED || m.state == SimState::FAILED;
            const std::string label = std::to_string(m.index + 1);
            if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_SpanAllColumns) && openable) {
                req->kind = m.state == SimState::FINISHED ? SimulationRequest::Kind::OPEN_LOG
                                                          : SimulationRequest::Kind::FAILURE;
                req->path = m.log;
                req->match = static_cast<size_t>(i);
            }
            if (m.state == SimState::FINISHED)
                ImGui::SetItemTooltip("%s\nA click opens it in Replay", m.log.c_str());
            else if (m.state == SimState::FAILED)
                ImGui::SetItemTooltip("%s\n%s\nA click shows the engine's stderr", m.error.c_str(), m.stderr_path.c_str());
            ImGui::TableNextColumn();
            ImGui::Text("%llu", static_cast<unsigned long long>(m.seed));
            for (int seat = 0; seat < 2; ++seat) {
                ImGui::TableNextColumn();
                const int e = m.entrant(seat);
                const bool won = m.state == SimState::FINISHED && m.winner_seat == seat;
                ImGui::PushStyleColor(ImGuiCol_Text, won ? tok().GOOD : tok().TEXT);
                ImGui::Text("%d %s", e + 1, sim_entrant_text(c.entrants[static_cast<size_t>(e)]).c_str());
                ImGui::PopStyleColor();
            }
            ImGui::TableNextColumn();
            ImGui::PushStyleColor(ImGuiCol_Text, state_color(m));
            ImGui::TextUnformatted(sim_result_text(m).c_str());
            ImGui::PopStyleColor();
            ImGui::TableNextColumn();
            if (m.games() > 0) ImGui::Text("%d", m.games());
            ImGui::TableNextColumn();
            if (m.decisions > 0) ImGui::Text("%zu", m.decisions);
            ImGui::TableNextColumn();
            if (m.seconds > 0) ImGui::TextDisabled("%s", seconds_text(m.seconds).c_str());
            ImGui::PopID();
        }
    }
    ImGui::EndTable();
}

}  // namespace rmwork
