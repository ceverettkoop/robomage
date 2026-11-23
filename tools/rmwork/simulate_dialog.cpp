#include "simulate_dialog.h"

#include <algorithm>
#include <cstdio>
#include <thread>

#include "files.h"
#include "imgui.h"
#include "modal.h"
#include "paths.h"
#include "play_session.h"
#include "simulation.h"
#include "theme.h"
#include "widgets.h"

namespace {

const char *const POPUP = "Simulate Match###rmwork.simulate";
const float FIELD_CHARS = 16;

}  // namespace

namespace rmwork {

std::string SimulateDialog::check(const SimulateSettings &g, const std::string &engine_configured) {
    Opponent nn = Opponent::EXPLORE;
    for (size_t i = 0; i < 2; ++i) {
        const SimulateSeatSettings &s = g.seats[i];
        const std::string seat = i == 0 ? "Seat A" : "Seat B";
        const int o = find_opponent(s.agent);
        if (o < 0) return seat + ": choose an agent.";
        if (s.deck.empty()) return seat + ": choose a deck.";
        const std::string problem = agent_problem(s.agent, s.model, s.server);
        if (!problem.empty()) return seat + ": " + problem;
        if (opponent_needs_nn(static_cast<Opponent>(o))) nn = static_cast<Opponent>(o);
    }
    if (g.matches < 1) return "Run at least one match.";
    std::string why;
    const std::string engine = play_engine_binary(engine_configured, nn, &why);
    if (engine.empty()) return why;
    if (!is_executable(engine)) return "No engine at " + engine + " (Settings \xE2\x86\x92 Engine).";
    return "";
}

void SimulateDialog::open(const SimulateSettings &from, const Settings &s, const std::string &engine_configured) {
    g_ = from;
    decks_ = NewGameDialog::list_decks(engine_workdir_path(s));
    models_ = NewGameDialog::list_models(s);
    socket_ = socket_path(s);
    for (SimulateSeatSettings &seat : g_.seats) {
        if (find_opponent(seat.agent) < 0) seat.agent = opponent_key(Opponent::EXPLORE);
        if (seat.model.empty() && !models_.empty() && models_[0].latest) seat.model = models_[0].path;
        if (seat.server.empty()) seat.server = socket_;
    }
    engine_ = engine_configured;
    want_open_ = true;
}

// A seat's agent, its model or server and options, and its deck.
void SimulateDialog::draw_seat(int seat) {
    SimulateSeatSettings &s = g_.seats[static_cast<size_t>(seat)];
    const float field_x = FIELD_CHARS * char_width();
    ImGui::PushID(seat);
    ImGui::SeparatorText(seat == 0 ? "Seat A (the first entrant)" : "Seat B (the second entrant)");
    field_label("Agent", field_x);
    agent_kind_combo("##agent", &s.agent);
    char buf[2048];
    const int kind = find_opponent(s.agent);
    if (s.agent == opponent_key(Opponent::MODEL)) {
        field_label("Model", field_x);
        model_combo("##model", models_, &s.model, field_x);
    } else if (s.agent == opponent_key(Opponent::SERVER)) {
        field_label("Server socket", field_x);
        std::snprintf(buf, sizeof buf, "%s", s.server.c_str());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputTextWithHint("##server", "rmserve --socket PATH", buf, sizeof buf)) s.server = buf;
    } else if (s.agent == opponent_key(Opponent::SCRIPTED)) {
        field_label("", field_x);
        ImGui::TextDisabled("python/teacher.py --tier hard, through the teacher channel at mix 1");
    }
    if (kind >= 0 && opponent_needs_nn(static_cast<Opponent>(kind))) {
        field_label("Options", field_x);
        ImGui::Checkbox("Argmax", &s.argmax);
        ImGui::SetItemTooltip("Always the most likely row (the nn: agent's argmax), else a sample of \xCF\x80");
        ImGui::SameLine(0, 16);
        ImGui::BeginDisabled(s.argmax);
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted("Temperature");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(9 * char_width());
        if (ImGui::InputDouble("##temp", &s.temperature, 0, 0, "%.2f"))
            s.temperature = std::min(100.0, std::max(0.01, s.temperature));
        ImGui::SetItemTooltip("Sampling temperature (temp=); 1 samples \xCF\x80 as it is");
        ImGui::EndDisabled();
    }
    field_label("Deck", field_x);
    deck_combo("##deck", decks_, &s.deck);
    ImGui::PopID();
}

bool SimulateDialog::draw(SimulateSettings *out) {
    if (want_open_) {
        ImGui::OpenPopup(POPUP);
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(680, 0));
    bool keep = true;
    open_ = ImGui::BeginPopupModal(POPUP, &keep, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleColor(3);
    if (!open_) return false;
    const float field_x = FIELD_CHARS * char_width();
    ImGui::PushItemWidth(40 * char_width());
    ImGui::Dummy(ImVec2(60 * char_width(), 0));
    draw_seat(0);
    ImGui::Spacing();
    draw_seat(1);
    ImGui::Spacing();
    ImGui::SeparatorText("Matches");
    field_label("Format", field_x);
    const Segment formats[] = {
        {"Best of one", false, Icon::LOGO, "One game a match"}, {"Best of three", false, Icon::LOGO, "Matches with sideboarding"}};
    int format = g_.bo3 ? 1 : 0;
    if (segmented("##format", formats, 2, &format)) g_.bo3 = format == 1;
    field_label("Matches", field_x);
    ImGui::SetNextItemWidth(12 * char_width());
    if (ImGui::InputInt("##matches", &g_.matches)) g_.matches = std::max(1, std::min(100000, g_.matches));
    ImGui::SameLine(0, 16);
    ImGui::Checkbox("Alternate seats", &g_.alternate);
    ImGui::SetItemTooltip("Odd matches swap the entrants' agents and decks, so each plays both seats");
    field_label("First seed", field_x);
    unsigned long long seed = g_.seed;
    ImGui::SetNextItemWidth(14 * char_width());
    if (ImGui::InputScalar("##seed", ImGuiDataType_U64, &seed)) g_.seed = seed;
    ImGui::SameLine();
    if (ImGui::Button("Random")) g_.seed = random_seed();
    ImGui::SameLine();
    ImGui::TextDisabled("match i plays --game-seed %llu + i", static_cast<unsigned long long>(g_.seed));
    field_label("Auto-mana", field_x);
    ImGui::Checkbox("##auto_mana", &g_.auto_mana);
    ImGui::SameLine();
    ImGui::TextDisabled(g_.auto_mana ? "costs paid by the engine" : "--no-auto-mana: the agents tap for mana");
    field_label("Parallel jobs", field_x);
    ImGui::SetNextItemWidth(12 * char_width());
    if (ImGui::InputInt("##jobs", &g_.jobs)) g_.jobs = std::max(0, std::min(1024, g_.jobs));
    ImGui::SameLine();
    const unsigned hw = std::thread::hardware_concurrency();
    ImGui::TextDisabled("%s", g_.jobs == 0 ? ("0: the default, " + std::to_string(default_simulation_jobs()) + " of " +
                                                 std::to_string(hw) + " hardware threads")
                                                    .c_str()
                                           : ("engines at once (" + std::to_string(hw) + " hardware threads)").c_str());
    ImGui::PopItemWidth();
    problem_ = check(g_, engine_);
    ImGui::Spacing();
    if (!problem_.empty()) colored_note(problem_.c_str(), tok().BAD);
    ImGui::Separator();
    bool close = !keep || close_requested(), started = false;
    ImGui::BeginDisabled(!problem_.empty());
    const std::string run = g_.matches == 1 ? "Run Match" : "Run " + std::to_string(g_.matches) + " Matches";
    if (ImGui::Button(run.c_str())) {
        *out = g_;
        started = close = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) close = true;
    if (close) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
    return started;
}

}  // namespace rmwork
