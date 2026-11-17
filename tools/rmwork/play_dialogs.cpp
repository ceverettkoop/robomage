#include "play_dialogs.h"

#include "evaluations.h"
#include "imgui.h"
#include "implot.h"
#include "modal.h"
#include "theme.h"
#include "widgets.h"

namespace {

const char *const CONCEDE_POPUP = "Concede###rmwork.concede";
const char *const GAME_OVER_POPUP = "Match Over###rmwork.game_over";

}  // namespace

static std::string game_line(int game, int winner, int seat);

// "Game 2: you won" / "you lost" / "a draw".
static std::string game_line(int game, int winner, int seat) {
    const std::string who = winner < 0 ? "a draw" : winner == seat ? "you won" : "you lost";
    return "Game " + std::to_string(game + 1) + ": " + who;
}

namespace rmwork {

void ConcedeDialog::open(bool match) {
    match_ = match;
    want_open_ = true;
}

bool ConcedeDialog::draw(bool *match) {
    if (want_open_) {
        ImGui::OpenPopup(CONCEDE_POPUP);
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(460, 0));
    bool keep = true;
    open_ = ImGui::BeginPopupModal(
        CONCEDE_POPUP, &keep, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleColor(3);
    if (!open_) return false;
    ImGui::TextUnformatted("Concede to the opponent?");
    ImGui::Spacing();
    if (ImGui::RadioButton("Concede this game (-2)", !match_)) match_ = false;
    if (ImGui::RadioButton("Concede the match (-3)", match_)) match_ = true;
    note(
        "A concession is sent at one of your decisions: while the opponent thinks, it waits for your next one. In "
        "sideboarding only the match can be conceded.");
    ImGui::Separator();
    bool close = !keep || close_requested(), confirmed = false;
    if (ImGui::Button(match_ ? "Concede Match" : "Concede Game")) {
        *match = match_;
        confirmed = close = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) close = true;
    if (close) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
    return confirmed;
}

MatchSummary summarize_match(const Match &m, int seat) {
    MatchSummary s;
    const MatchResult &r = m.result;
    s.title = !r.known ? "The match ended" : r.winner < 0 ? "A draw" : r.winner == seat ? "You win" : "You lose";
    if (m.bo3() && r.known)
        s.score = std::to_string(r.wins[static_cast<size_t>(seat)]) + "-" +
                  std::to_string(r.wins[static_cast<size_t>(1 - seat)]);
    for (size_t g = 0; g < r.game_winners.size(); ++g)
        s.games.push_back(game_line(static_cast<int>(g), r.game_winners[g], seat));
    s.log_path = m.log_path;
    const MatchValues v = match_values(m);
    const int64_t policy = v.resolve(-1);
    const ValueSeries *series = policy < 0 ? nullptr : v.find(seat, policy);
    if (series) {
        for (const ValuePoint &p : series->points) s.values.push_back({static_cast<double>(p.decision), p.value});
    } else {
        s.values_note = "Model's Read did not run in this game, so there is no value curve.";
    }
    return s;
}

void GameOverDialog::open(const MatchSummary &s) {
    s_ = s;
    want_open_ = true;
}

GameOverDialog::Choice GameOverDialog::draw() {
    if (want_open_) {
        ImGui::OpenPopup(GAME_OVER_POPUP);
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(560, 0));
    bool keep = true;
    open_ = ImGui::BeginPopupModal(
        GAME_OVER_POPUP, &keep, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleColor(3);
    if (!open_) return Choice::NONE;
    const float width = 52 * char_width();
    ImGui::Dummy(ImVec2(width, 0));
    const ImU32 title_color = s_.title == "You win" ? tok().GOOD : s_.title == "You lose" ? tok().BAD : tok().TEXT;
    ImGui::PushStyleColor(ImGuiCol_Text, title_color);
    ImGui::PushFont(nullptr, ImGui::GetStyle().FontSizeBase * 1.6f);
    ImGui::TextUnformatted(s_.title.c_str());
    ImGui::PopFont();
    ImGui::PopStyleColor();
    if (!s_.score.empty()) {
        ImGui::SameLine();
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%s", s_.score.c_str());
    }
    for (const std::string &g : s_.games) ImGui::BulletText("%s", g.c_str());
    ImGui::Spacing();
    if (s_.values.empty()) {
        ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + width);
        note(s_.values_note.c_str());
        ImGui::PopTextWrapPos();
    } else {
        ImGui::TextDisabled("Your value (Model's Read) over your decisions");
        const ImPlotFlags flags = ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect | ImPlotFlags_NoMouseText |
                                  ImPlotFlags_NoLegend | ImPlotFlags_NoInputs;
        if (ImPlot::BeginPlot("##game_over_values", ImVec2(width, 9 * ImGui::GetFontSize()), flags)) {
            ImPlot::SetupAxis(ImAxis_X1, "decision");
            ImPlot::SetupAxis(ImAxis_Y1, "V");
            ImPlot::SetupAxisLimits(ImAxis_X1, 0, s_.values.back().first + 1, ImPlotCond_Always);
            ImPlot::SetupAxisLimits(ImAxis_Y1, -1.05, 1.05, ImPlotCond_Always);
            std::vector<double> xs, ys;
            for (const auto &p : s_.values) {
                xs.push_back(p.first);
                ys.push_back(p.second);
            }
            ImPlot::PushStyleColor(ImPlotCol_Line, ImGui::ColorConvertU32ToFloat4(tok().ACCENT));
            ImPlot::PlotLine("V", xs.data(), ys.data(), static_cast<int>(xs.size()));
            ImPlot::PopStyleColor();
            ImPlot::EndPlot();
        }
    }
    const size_t slash = s_.log_path.rfind('/');
    ImGui::TextDisabled("Decision log: %s", s_.log_path.substr(slash == std::string::npos ? 0 : slash + 1).c_str());
    ImGui::SetItemTooltip("%s", s_.log_path.c_str());
    ImGui::Separator();
    Choice choice = Choice::NONE;
    bool close = !keep || close_requested();
    if (ImGui::Button("Open in Replay")) choice = Choice::OPEN_REPLAY;
    ImGui::SetItemTooltip("Replay the game's decision log with every viewer's frames");
    ImGui::SameLine();
    if (ImGui::Button("New Game")) choice = Choice::NEW_GAME;
    ImGui::SameLine();
    if (ImGui::Button("Close")) close = true;
    if (close || choice != Choice::NONE) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
    return choice;
}

}  // namespace rmwork
