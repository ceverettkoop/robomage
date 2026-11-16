#include "run_browser.h"

#include <dirent.h>

#include <algorithm>
#include <cstdlib>
#include <unordered_map>

#include "episodes.h"
#include "file_picker.h"
#include "files.h"
#include "imgui.h"
#include "modal.h"
#include "theme.h"

namespace {

const char *const RUN_POPUP = "###rmwork.run_browser";

}  // namespace

using rmwork::RunLog;
using rmwork::RunPurpose;

static bool ends_with(const std::string &s, const char *suffix);
static std::vector<std::string> list_names(const std::string &dir, bool dirs);
static RunPurpose read_purpose(const std::string &name, const std::string &log_dir);
static std::string games_text(const std::vector<int> &games);

static bool ends_with(const std::string &s, const char *suffix) {
    const std::string x = suffix;
    return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}

// The names in `dir`: its subdirectories (`dirs`) or its files.
static std::vector<std::string> list_names(const std::string &dir, bool dirs) {
    std::vector<std::string> out;
    DIR *h = opendir(dir.c_str());
    if (!h) return out;
    while (const dirent *e = readdir(h)) {
        const std::string name = e->d_name;
        if (name.empty() || name[0] == '.') continue;
        const bool is_dir =
            e->d_type == DT_DIR || ((e->d_type == DT_LNK || e->d_type == DT_UNKNOWN) &&
                                       rmwork::path_kind(dir + "/" + name) == rmwork::PathKind::DIRECTORY);
        if (is_dir == dirs) out.push_back(name);
    }
    closedir(h);
    return out;
}

// The logs in `log_dir` and the games their episodes cover.
static RunPurpose read_purpose(const std::string &name, const std::string &log_dir) {
    RunPurpose p;
    p.name = name;
    std::unordered_map<std::string, std::vector<int>> episodes;
    for (const std::string &dir : rmwork::episode_dirs_for_log(log_dir + "/x.rmlog")) {
        for (const std::string &f : list_names(dir, false)) {
            if (!ends_with(f, ".rmtraj")) continue;
            const size_t g = f.rfind(".g");
            if (g == std::string::npos) continue;
            const int n = std::atoi(f.c_str() + g + 2);
            if (n <= 0) continue;
            std::vector<int> &games = episodes[f.substr(0, g)];
            if (std::find(games.begin(), games.end(), n) == games.end()) games.push_back(n);
        }
    }
    for (const std::string &f : list_names(log_dir, false)) {
        if (!ends_with(f, ".rmlog")) continue;
        RunLog log;
        log.path = log_dir + "/" + f;
        log.name = f.substr(0, f.size() - 6);
        const auto it = episodes.find(log.name);
        if (it != episodes.end()) {
            log.games = it->second;
            std::sort(log.games.begin(), log.games.end());
            ++p.with_episodes;
        }
        p.logs.push_back(std::move(log));
    }
    std::sort(p.logs.begin(), p.logs.end(),
        [](const RunLog &a, const RunLog &b) { return rmwork::natural_less(a.name, b.name); });
    return p;
}

static std::string games_text(const std::vector<int> &games) {
    std::string out;
    for (int g : games) out += (out.empty() ? "g" : " g") + std::to_string(g);
    return out;
}

namespace rmwork {

std::vector<RunPurpose> scan_run_directory(const std::string &dir) {
    std::vector<RunPurpose> out;
    const std::string rmlog = dir + "/rmlog";
    if (path_kind(rmlog) == PathKind::DIRECTORY) {
        std::vector<std::string> subs = list_names(rmlog, true);
        std::sort(subs.begin(), subs.end(), natural_less);
        for (const std::string &s : subs) {
            RunPurpose p = read_purpose(s, rmlog + "/" + s);
            if (!p.logs.empty()) out.push_back(std::move(p));
        }
        RunPurpose top = read_purpose(".", rmlog);
        if (!top.logs.empty()) out.push_back(std::move(top));
    }
    RunPurpose here = read_purpose(".", dir);
    if (!here.logs.empty()) out.push_back(std::move(here));
    return out;
}

RunBrowser::~RunBrowser() {
    if (thread_.joinable()) thread_.join();
}

void RunBrowser::open(const std::string &dir) {
    if (thread_.joinable()) thread_.join();
    dir_ = dir;
    done_.store(false);
    collected_ = false;
    purposes_.clear();
    scanned_.clear();
    shown_.clear();
    shown_key_.clear();
    purpose_ = 0;
    selected_ = -1;
    thread_ = std::thread([this] {
        scanned_ = scan_run_directory(dir_);
        done_.store(true);
    });
    want_open_ = true;
}

void RunBrowser::collect() {
    if (collected_ || !done_.load()) return;
    thread_.join();
    purposes_ = std::move(scanned_);
    collected_ = true;
}

bool RunBrowser::draw(std::string *log_path) {
    collect();
    const std::string popup = "Run: " + dir_ + RUN_POPUP;
    if (want_open_) {
        ImGui::OpenPopup(popup.c_str());
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(720, 560));
    bool keep = true;
    open_ = ImGui::BeginPopupModal(popup.c_str(), &keep, ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleColor(3);
    if (!open_) return false;
    bool chosen = false;
    if (!collected_) {
        ImGui::TextDisabled("Reading %s …", dir_.c_str());
    } else if (purposes_.empty()) {
        colored_note("No decision logs here: a run directory holds them under rmlog/<purpose>/.", tok().BAD);
    } else {
        if (purpose_ >= static_cast<int>(purposes_.size())) purpose_ = 0;
        const RunPurpose &p = purposes_[static_cast<size_t>(purpose_)];
        field_label("Logs", 9 * char_width());
        ImGui::SetNextItemWidth(30 * char_width());
        const std::string label = p.name + " (" + std::to_string(p.logs.size()) + ")";
        if (ImGui::BeginCombo("##purpose", label.c_str())) {
            for (size_t i = 0; i < purposes_.size(); ++i) {
                const std::string l = purposes_[i].name + " (" + std::to_string(purposes_[i].logs.size()) + ", " +
                                      std::to_string(purposes_[i].with_episodes) + " with episodes)";
                if (ImGui::Selectable(l.c_str(), static_cast<int>(i) == purpose_)) {
                    purpose_ = static_cast<int>(i);
                    selected_ = -1;
                }
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::Checkbox("With episodes only", &episodes_only_);
        field_label("Filter", 9 * char_width());
        ImGui::SetNextItemWidth(30 * char_width());
        ImGui::InputTextWithHint("##filter", "part of a name", filter_, sizeof filter_);
        const RunPurpose &cur = purposes_[static_cast<size_t>(purpose_)];
        const std::string key = std::to_string(purpose_) + "\n" + filter_ + (episodes_only_ ? "\n1" : "\n0");
        if (key != shown_key_) {
            shown_key_ = key;
            shown_.clear();
            for (size_t i = 0; i < cur.logs.size(); ++i) {
                if (episodes_only_ && cur.logs[i].games.empty()) continue;
                if (filter_[0] && cur.logs[i].name.find(filter_) == std::string::npos) continue;
                shown_.push_back(i);
            }
            selected_ = -1;
        }
        const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_BordersOuter;
        if (ImGui::BeginTable("##logs", 2, flags, ImVec2(0, -footer_height() - ImGui::GetFrameHeightWithSpacing()))) {
            ImGui::TableSetupScrollFreeze(0, 1);
            ImGui::TableSetupColumn("Decision log", ImGuiTableColumnFlags_WidthStretch, 3.0f);
            ImGui::TableSetupColumn("Episodes", ImGuiTableColumnFlags_WidthStretch, 1.0f);
            ImGui::TableHeadersRow();
            ImGuiListClipper clipper;
            clipper.Begin(static_cast<int>(shown_.size()));
            while (clipper.Step()) {
                for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                    const RunLog &log = cur.logs[shown_[static_cast<size_t>(i)]];
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::PushID(i);
                    if (ImGui::Selectable(log.name.c_str(), selected_ == i,
                            ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick)) {
                        selected_ = i;
                        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                            *log_path = log.path;
                            chosen = true;
                        }
                    }
                    ImGui::PopID();
                    ImGui::TableNextColumn();
                    ImGui::TextDisabled("%s", log.games.empty() ? "—" : games_text(log.games).c_str());
                }
            }
            ImGui::EndTable();
        }
        ImGui::TextDisabled("%zu shown of %zu", shown_.size(), cur.logs.size());
    }
    ImGui::Separator();
    const bool can_open =
        collected_ && selected_ >= 0 && selected_ < static_cast<int>(shown_.size()) && !purposes_.empty();
    ImGui::BeginDisabled(!can_open);
    if (ImGui::Button("Open") && can_open) {
        *log_path = purposes_[static_cast<size_t>(purpose_)].logs[shown_[static_cast<size_t>(selected_)]].path;
        chosen = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (right_aligned_button("Close") || close_requested() || !keep || chosen) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
    return chosen;
}

}  // namespace rmwork
