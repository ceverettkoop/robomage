#include "new_game_dialog.h"

#include <dirent.h>

#include <algorithm>
#include <cstdio>
#include <random>
#include <utility>

#include "file_picker.h"
#include "files.h"
#include "imgui.h"
#include "modal.h"
#include "model_zip.h"
#include "paths.h"
#include "play_session.h"
#include "policy_catalog.h"
#include "theme.h"
#include "widgets.h"

namespace {

const char *const POPUP = "New Game vs Model###rmwork.new_game";
// The subdirectories of resources/decks whose decks are offered, after the top level.
const char *const DECK_DIRS[] = {"", "league", "heldout"};

}  // namespace

static std::vector<std::string> names_in(const std::string &dir, bool dirs);
static bool has_suffix(const std::string &s, const char *suffix);

// The entries of `dir` that are directories (`dirs`) or files, in natural order, hidden ones left out.
static std::vector<std::string> names_in(const std::string &dir, bool dirs) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    while (const dirent *e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.empty() || n[0] == '.') continue;
        if ((rmwork::path_kind(dir + "/" + n) == rmwork::PathKind::DIRECTORY) == dirs) out.push_back(n);
    }
    closedir(d);
    std::sort(out.begin(), out.end(), rmwork::natural_less);
    return out;
}

static bool has_suffix(const std::string &s, const char *suffix) {
    const std::string x = suffix;
    return s.size() > x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}

namespace rmwork {

uint64_t random_seed() {
    std::random_device rd;
    return 1 + static_cast<uint64_t>(rd()) % 999999u;
}

void agent_kind_combo(const char *id, std::string *key) {
    const int current = std::max(0, find_opponent(*key));
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo(id, opponent_label(static_cast<Opponent>(current)))) {
        for (int o = 0; o < OPPONENT_COUNT; ++o)
            if (ImGui::Selectable(opponent_label(static_cast<Opponent>(o)), o == current))
                *key = opponent_key(static_cast<Opponent>(o));
        ImGui::EndCombo();
    }
}

void deck_combo(const char *id, const std::vector<std::string> &decks, std::string *deck) {
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo(id, deck->empty() ? "(choose a deck)" : deck->c_str(), ImGuiComboFlags_HeightLarge)) {
        std::string group = "-";
        for (const std::string &d : decks) {
            const size_t slash = d.find('/');
            const std::string g = slash == std::string::npos ? "" : d.substr(0, slash);
            if (g != group) {
                group = g;
                ImGui::TextDisabled("%s", g.empty() ? "decks/" : (g + "/").c_str());
            }
            if (ImGui::Selectable(d.c_str(), d == *deck)) *deck = d;
            if (d == *deck && ImGui::IsWindowAppearing()) ImGui::SetScrollHereY(0.5f);
        }
        if (decks.empty()) ImGui::TextDisabled("No decks under the engine's resources/decks");
        ImGui::EndCombo();
    }
}

void model_combo(const char *id, const std::vector<ModelChoice> &models, std::string *model, float field_x) {
    ImGui::PushID(id);
    std::string preview = model->empty() ? "(choose a model)" : *model;
    for (const ModelChoice &m : models)
        if (m.path == *model) preview = m.label;
    ImGui::SetNextItemWidth(-1);
    if (ImGui::BeginCombo("##model", preview.c_str(), ImGuiComboFlags_HeightLarge)) {
        for (size_t i = 0; i < models.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::Selectable(models[i].label.c_str(), models[i].path == *model)) *model = models[i].path;
            ImGui::SetItemTooltip("%s", models[i].path.c_str());
            ImGui::PopID();
        }
        if (models.empty())
            ImGui::TextDisabled(
                "No policies in Settings \xE2\x86\x92 Models, no exports under its run directory and no "
                "zipped exports in models/");
        ImGui::EndCombo();
    }
    field_label("", field_x);
    char buf[2048];
    std::snprintf(buf, sizeof buf, "%s", model->c_str());
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint(
            "##model_path", "or the export's path (archive.zip!member for a zipped one)", buf, sizeof buf))
        *model = buf;
    ImGui::PopID();
}

std::string agent_problem(const std::string &key, const std::string &model, const std::string &server) {
    const std::string m = resolve_path(model, ""), sock = resolve_path(server, "");
    if (key == opponent_key(Opponent::MODEL)) {
        if (m.empty()) return "Choose the model the agent plays.";
        return model_problem(m);
    }
    if (key == opponent_key(Opponent::SERVER) && (sock.empty() || path_kind(sock) != PathKind::SOCKET))
        return sock.empty() ? "Give the inference server's socket." : "No server at " + sock + ".";
    return "";
}

std::vector<std::string> NewGameDialog::list_decks(const std::string &workdir) {
    std::vector<std::string> out;
    for (const char *sub : DECK_DIRS) {
        const std::string dir = workdir + "/resources/decks" + (*sub ? std::string("/") + sub : "");
        for (const std::string &n : names_in(dir, false))
            if (has_suffix(n, ".dk")) out.push_back((*sub ? std::string(sub) + "/" : "") + n.substr(0, n.size() - 3));
    }
    return out;
}

std::vector<ModelChoice> NewGameDialog::list_models(const Settings &s) {
    std::vector<ModelChoice> out, exports = list_exports(s);
    size_t i = 0;
    if (!exports.empty() && exports[0].latest) out.push_back(std::move(exports[i++]));
    for (const PolicyEntry &p : s.models.policies)
        out.push_back({"policy " + p.id + " \xC2\xB7 " + p.model, policy_model_path(p)});
    for (; i < exports.size(); ++i) out.push_back(std::move(exports[i]));
    return out;
}

std::vector<ModelChoice> NewGameDialog::list_exports(const Settings &s) {
    std::vector<ModelChoice> out;
    for (PolicyRow &r : scan_policies(run_dir_path(s), default_models_dir()))
        if (!r.hidden && !r.play.empty()) out.push_back({std::move(r.label), std::move(r.play), r.latest});
    return out;
}

std::string NewGameDialog::check(const NewGameSettings &g, const std::string &engine_configured) {
    const int o = find_opponent(g.opponent);
    if (o < 0) return "Choose an opponent.";
    if (g.deck.empty() || g.opponent_deck.empty()) return "Choose both decks.";
    const Opponent opp = static_cast<Opponent>(o);
    const std::string problem = agent_problem(g.opponent, g.model, g.server);
    if (!problem.empty()) return problem;
    std::string why;
    const std::string engine = play_engine_binary(engine_configured, opp, &why);
    if (engine.empty()) return why;
    if (!is_executable(engine)) return "No engine at " + engine + " (Settings \xE2\x86\x92 Engine).";
    return "";
}

void NewGameDialog::open(const NewGameSettings &from, const Settings &s, const std::string &engine_configured) {
    g_ = from;
    if (find_opponent(g_.opponent) < 0) g_.opponent = opponent_key(Opponent::EXPLORE);
    decks_ = list_decks(engine_workdir_path(s));
    models_ = list_models(s);
    if (g_.model.empty() && !models_.empty() && models_[0].latest) g_.model = models_[0].path;
    engine_ = engine_configured;
    socket_ = socket_path(s);
    if (g_.server.empty()) g_.server = socket_;
    want_open_ = true;
}

// The opponent's kind and, for a model, its export or server.
void NewGameDialog::draw_opponent() {
    const float field_x = 16 * char_width();
    field_label("Opponent", field_x);
    agent_kind_combo("##opponent", &g_.opponent);
    char buf[2048];
    if (g_.opponent == opponent_key(Opponent::MODEL)) {
        field_label("Model", field_x);
        model_combo("##model", models_, &g_.model, field_x);
    } else if (g_.opponent == opponent_key(Opponent::SERVER)) {
        field_label("Server socket", field_x);
        std::snprintf(buf, sizeof buf, "%s", g_.server.c_str());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputTextWithHint("##server", "rmserve --socket PATH", buf, sizeof buf)) g_.server = buf;
        note("The engine asks the server for policy 0 (nn:remote:<socket>).");
    } else if (g_.opponent == opponent_key(Opponent::SCRIPTED)) {
        note(
            "python/teacher.py --tier hard plays the seat through the engine's teacher channel (--teacher-mix 1); "
            "it needs python3.");
    }
}

bool NewGameDialog::draw(NewGameSettings *out) {
    if (want_open_) {
        ImGui::OpenPopup(POPUP);
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(640, 0));
    bool keep = true;
    open_ = ImGui::BeginPopupModal(POPUP, &keep, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleColor(3);
    if (!open_) return false;
    const float field_x = 16 * char_width();
    ImGui::PushItemWidth(40 * char_width());
    ImGui::Dummy(ImVec2(56 * char_width(), 0));
    field_label("Your deck", field_x);
    deck_combo("##deck", decks_, &g_.deck);
    field_label("Your seat", field_x);
    const Segment seats[] = {
        {"Player A", false, Icon::LOGO, "Seat A (the engine's player A)"}, {"Player B", false, Icon::LOGO, "Seat B"}};
    int seat = g_.seat == 1 ? 1 : 0;
    if (segmented("##seat", seats, 2, &seat)) g_.seat = seat;
    ImGui::Spacing();
    draw_opponent();
    field_label("Opponent's deck", field_x);
    deck_combo("##opp_deck", decks_, &g_.opponent_deck);
    ImGui::Spacing();
    field_label("Format", field_x);
    const Segment formats[] = {
        {"Best of one", false, Icon::LOGO, "One game"}, {"Best of three", false, Icon::LOGO, "A match, sideboarding"}};
    int format = g_.bo3 ? 1 : 0;
    if (segmented("##format", formats, 2, &format)) g_.bo3 = format == 1;
    field_label("Seed", field_x);
    unsigned long long seed = g_.seed;
    ImGui::SetNextItemWidth(14 * char_width());
    if (ImGui::InputScalar("##seed", ImGuiDataType_U64, &seed)) g_.seed = seed;
    ImGui::SameLine();
    if (ImGui::Button("Random")) g_.seed = random_seed();
    ImGui::SetItemTooltip("A random seed: the engine's --game-seed (the shuffles, and the opponent's choices)");
    field_label("Auto-mana", field_x);
    ImGui::Checkbox("##auto_mana", &g_.auto_mana);
    ImGui::SameLine();
    ImGui::TextDisabled(g_.auto_mana ? "costs paid for you" : "--no-auto-mana: you tap for mana");
    field_label("Model's Read", field_x);
    ImGui::Checkbox("##model_read", &g_.model_read);
    ImGui::SameLine();
    const bool server = !socket_.empty() && path_kind(socket_) == PathKind::SOCKET;
    ImGui::TextDisabled("%s", !g_.model_read    ? "off: an unaided game"
                              : server          ? ("your decisions evaluated through " + socket_).c_str()
                              : socket_.empty() ? "no server socket in Settings \xE2\x86\x92 Models: not connected"
                                                : ("no server at " + socket_ + ": not connected").c_str());
    ImGui::PopItemWidth();
    problem_ = check(g_, engine_);
    ImGui::Spacing();
    if (!problem_.empty()) colored_note(problem_.c_str(), tok().BAD);
    ImGui::Separator();
    bool close = !keep || close_requested(), started = false;
    ImGui::BeginDisabled(!problem_.empty());
    if (ImGui::Button("Start Game")) {
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
