#include "file_picker.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include "files.h"
#include "imgui.h"
#include "modal.h"
#include "paths.h"
#include "theme.h"

namespace {

const char *const PICKER_POPUP = "###rmwork.file_picker";
const char *const FILTER_LABELS[rmwork::FILE_FILTER_COUNT] = {"Decision logs (*.rmlog)",
    "Trajectory episodes (*.rmtraj)", "Decision logs and episodes", "Directories", "All files"};

}  // namespace

static bool ends_with(const std::string &s, const char *suffix);
static std::string home();
static std::string join_path(const std::string &dir, const std::string &name);
static std::string normalize(const std::string &path);
static std::string base_name(const std::string &path);
static std::string size_text(uint64_t bytes);
static std::string time_text(int64_t t);
static void fill_stat(const std::string &dir, const rmwork::PickerEntry &e);

static bool ends_with(const std::string &s, const char *suffix) {
    const std::string x = suffix;
    return s.size() >= x.size() && s.compare(s.size() - x.size(), x.size(), x) == 0;
}

static std::string home() {
    const char *h = std::getenv("HOME");
    return h && *h ? h : "/";
}

static std::string join_path(const std::string &dir, const std::string &name) {
    return dir == "/" ? "/" + name : dir + "/" + name;
}

// An absolute path with "." and ".." resolved by name and no trailing slash.
static std::string normalize(const std::string &path) {
    std::string p = path;
    if (p.empty()) return "/";
    if (p == "~" || p.compare(0, 2, "~/") == 0) p = home() + p.substr(1);
    if (p[0] != '/') p = rmwork::current_dir() + "/" + p;
    std::vector<std::string> parts;
    size_t start = 0;
    while (start <= p.size()) {
        size_t slash = p.find('/', start);
        if (slash == std::string::npos) slash = p.size();
        const std::string part = p.substr(start, slash - start);
        if (part == "..") {
            if (!parts.empty()) parts.pop_back();
        } else if (!part.empty() && part != ".") {
            parts.push_back(part);
        }
        start = slash + 1;
    }
    std::string out;
    for (const std::string &part : parts) out += "/" + part;
    return out.empty() ? "/" : out;
}

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string size_text(uint64_t bytes) {
    char buf[32];
    if (bytes < 1024) {
        std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(bytes));
    } else if (bytes < 1024 * 1024) {
        std::snprintf(buf, sizeof buf, "%.1f KB", static_cast<double>(bytes) / 1024.0);
    } else {
        std::snprintf(buf, sizeof buf, "%.1f MB", static_cast<double>(bytes) / (1024.0 * 1024.0));
    }
    return buf;
}

static std::string time_text(int64_t t) {
    const std::time_t tt = static_cast<std::time_t>(t);
    std::tm tm;
    localtime_r(&tt, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M", &tm);
    return buf;
}

static void fill_stat(const std::string &dir, const rmwork::PickerEntry &e) {
    if (e.stat_done) return;
    e.stat_done = true;
    struct stat st;
    if (stat(join_path(dir, e.name).c_str(), &st) != 0) return;
    e.size = static_cast<uint64_t>(st.st_size);
    e.mtime = static_cast<int64_t>(st.st_mtime);
}

namespace rmwork {

const char *file_filter_label(FileFilter f) {
    return FILTER_LABELS[static_cast<int>(f)];
}

bool file_filter_admits(FileFilter f, const std::string &name) {
    switch (f) {
        case FileFilter::DECISION_LOGS:
            return ends_with(name, ".rmlog");
        case FileFilter::EPISODES:
            return ends_with(name, ".rmtraj");
        case FileFilter::LOGS_AND_EPISODES:
            return ends_with(name, ".rmlog") || ends_with(name, ".rmtraj");
        case FileFilter::DIRECTORIES:
            return false;
        case FileFilter::ALL:
            return true;
    }
    return true;
}

bool natural_less(const std::string &a, const std::string &b) {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        const bool da = std::isdigit(static_cast<unsigned char>(a[i])) != 0;
        const bool db = std::isdigit(static_cast<unsigned char>(b[j])) != 0;
        if (da && db) {
            size_t ie = i, je = j;
            while (ie < a.size() && std::isdigit(static_cast<unsigned char>(a[ie]))) ++ie;
            while (je < b.size() && std::isdigit(static_cast<unsigned char>(b[je]))) ++je;
            size_t is = i, js = j;
            while (is + 1 < ie && a[is] == '0') ++is;
            while (js + 1 < je && b[js] == '0') ++js;
            if (ie - is != je - js) return ie - is < je - js;
            const int c = a.compare(is, ie - is, b, js, je - js);
            if (c != 0) return c < 0;
            i = ie;
            j = je;
            continue;
        }
        if (a[i] != b[j]) return a[i] < b[j];
        ++i;
        ++j;
    }
    return a.size() - i < b.size() - j;
}

void FilePicker::open(const std::string &title, int purpose, FileFilter filter, const std::string &dir,
    const std::vector<std::string> &recent) {
    title_ = title;
    purpose_ = purpose;
    filter_ = filter;
    recent_ = recent;
    selected_.clear();
    error_.clear();
    save_ = false;
    replace_path_.clear();
    std::string start = normalize(dir.empty() ? current_dir() : dir);
    if (path_kind(start) == PathKind::FILE) {
        selected_ = base_name(start);
        start = parent_dir(start);
    }
    if (path_kind(start) != PathKind::DIRECTORY || !navigate(start)) navigate(home());
    want_open_ = true;
}

void FilePicker::open_save(const std::string &title, int purpose, FileFilter filter, const std::string &dir,
    const std::string &name, const std::vector<std::string> &recent) {
    open(title, purpose, filter, dir, recent);
    save_ = true;
    save_name_ = name;
}

void FilePicker::try_save(const std::string &path, std::string *chosen, bool *picked) {
    if (path_kind(path) == PathKind::DIRECTORY) {
        error_ = path + " is a directory";
        return;
    }
    if (path_kind(path) != PathKind::MISSING && replace_path_ != path) {
        replace_path_ = path;
        error_ = base_name(path) + " exists: press Replace to overwrite it";
        return;
    }
    *chosen = path;
    *picked = true;
}

// The file name field and Save (Replace once a name that exists was asked about).
void FilePicker::draw_save_name(std::string *chosen, bool *picked) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("File name");
    ImGui::SameLine();
    char buf[1024];
    std::snprintf(buf, sizeof buf, "%s", save_name_.c_str());
    const float bw =
        ImGui::CalcTextSize("Replace").x + ImGui::CalcTextSize("Cancel").x + ImGui::GetStyle().FramePadding.x * 4 + 36;
    ImGui::SetNextItemWidth(std::max(10 * char_width(), ImGui::GetContentRegionAvail().x - bw));
    const bool enter = ImGui::InputText("##save_name", buf, sizeof buf, ImGuiInputTextFlags_EnterReturnsTrue);
    if (std::string(buf) != save_name_) {
        save_name_ = buf;
        replace_path_.clear();
        error_.clear();
    }
    ImGui::SameLine();
    const std::string path = join_path(dir_, save_name_);
    ImGui::BeginDisabled(save_name_.empty());
    if (ImGui::Button(replace_path_ == path && !replace_path_.empty() ? "Replace" : "Save") ||
        (enter && !save_name_.empty()))
        try_save(path, chosen, picked);
    ImGui::EndDisabled();
}

bool FilePicker::navigate(const std::string &dir) {
    const std::string d = normalize(dir);
    DIR *h = opendir(d.c_str());
    if (!h) {
        error_ = "cannot open " + d;
        return false;
    }
    all_.clear();
    while (const dirent *e = readdir(h)) {
        const std::string name = e->d_name;
        if (name == "." || name == ".." || name[0] == '.') continue;
        PickerEntry pe;
        pe.name = name;
        if (e->d_type == DT_DIR) {
            pe.dir = true;
        } else if (e->d_type == DT_LNK || e->d_type == DT_UNKNOWN) {
            pe.dir = path_kind(join_path(d, name)) == PathKind::DIRECTORY;
        }
        all_.push_back(pe);
    }
    closedir(h);
    std::sort(all_.begin(), all_.end(), [](const PickerEntry &a, const PickerEntry &b) {
        if (a.dir != b.dir) return a.dir;
        return natural_less(a.name, b.name);
    });
    if (d != dir_) selected_.clear();
    dir_ = d;
    path_field_ = d;
    error_.clear();
    scroll_top_ = true;
    list();
    return true;
}

void FilePicker::set_filter(FileFilter f) {
    filter_ = f;
    list();
}

void FilePicker::list() {
    entries_.clear();
    for (const PickerEntry &e : all_)
        if (e.dir || file_filter_admits(filter_, e.name)) entries_.push_back(e);
}

void FilePicker::up() {
    navigate(parent_dir(dir_).empty() ? "/" : parent_dir(dir_));
}

bool FilePicker::draw(std::string *chosen) {
    const std::string popup = title_ + PICKER_POPUP;
    if (want_open_) {
        ImGui::OpenPopup(popup.c_str());
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(940, 600));
    bool keep = true;
    open_ = ImGui::BeginPopupModal(popup.c_str(), &keep, ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleColor(3);
    if (!open_) return false;
    bool picked = false;
    if (ImGui::Button("Up")) up();
    ImGui::SetItemTooltip("The parent directory");
    ImGui::SameLine();
    char buf[4096];
    std::snprintf(buf, sizeof buf, "%s", path_field_.c_str());
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputText("##path", buf, sizeof buf, ImGuiInputTextFlags_EnterReturnsTrue)) {
        const std::string p = normalize(buf);
        if (path_kind(p) == PathKind::DIRECTORY) {
            navigate(p);
        } else if (save_ && path_kind(parent_dir(p)) == PathKind::DIRECTORY) {
            try_save(p, chosen, &picked);
        } else if (path_kind(p) == PathKind::FILE) {
            *chosen = p;
            picked = true;
        } else {
            error_ = "no such file or directory: " + p;
        }
    } else {
        path_field_ = buf;
    }
    const float places_w = 22 * char_width();
    ImGui::BeginChild(
        "##places", ImVec2(places_w, -footer_height() - ImGui::GetFrameHeightWithSpacing()), ImGuiChildFlags_Borders);
    draw_places(chosen, &picked);
    ImGui::EndChild();
    ImGui::SameLine();
    ImGui::BeginChild(
        "##listing", ImVec2(0, -footer_height() - ImGui::GetFrameHeightWithSpacing()), ImGuiChildFlags_Borders);
    draw_listing(chosen, &picked);
    ImGui::EndChild();
    if (!error_.empty()) {
        colored_note(error_.c_str(), tok().BAD);
    } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextDisabled("%zu shown of %zu in %s", entries_.size(), all_.size(), dir_.c_str());
    }
    ImGui::Separator();
    if (save_) {
        draw_save_name(chosen, &picked);
        ImGui::SameLine();
        if (ImGui::Button("Cancel") || close_requested() || !keep) {
            ImGui::CloseCurrentPopup();
            open_ = false;
        }
        if (picked) {
            ImGui::CloseCurrentPopup();
            open_ = false;
        }
        ImGui::EndPopup();
        return picked;
    }
    ImGui::SetNextItemWidth(30 * char_width());
    if (ImGui::BeginCombo("##filter", file_filter_label(filter_))) {
        for (int f = 0; f < FILE_FILTER_COUNT; ++f)
            if (ImGui::Selectable(FILTER_LABELS[f], f == static_cast<int>(filter_)))
                set_filter(static_cast<FileFilter>(f));
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    const bool dirs = filter_ == FileFilter::DIRECTORIES;
    const char *open_label = dirs ? "Choose this directory" : "Open";
    const float bw =
        ImGui::CalcTextSize(open_label).x + ImGui::CalcTextSize("Cancel").x + ImGui::GetStyle().FramePadding.x * 4 + 36;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - bw));
    const bool can_open = dirs || !selected_.empty();
    ImGui::BeginDisabled(!can_open);
    const bool enter = ImGui::IsKeyPressed(ImGuiKey_Enter, false) && !ImGui::GetIO().WantTextInput;
    if (ImGui::Button(open_label) || (enter && can_open)) {
        *chosen = dirs ? (selected_.empty() ? dir_ : join_path(dir_, selected_)) : join_path(dir_, selected_);
        picked = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Cancel") || close_requested() || !keep) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    if (picked) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
    return picked;
}

// Home, the repository's directories that hold games, and the recent files.
void FilePicker::draw_places(std::string *chosen, bool *picked) {
    ImGui::TextDisabled("Places");
    const std::string repo = repo_root();
    const struct {
            const char *label;
            std::string path;
    } places[] = {{"Home", home()}, {"Repository", repo}, {"Runs", repo + "/runs"},
        {"Replay corpus", repo + "/tests/corpus"}, {"Current directory", current_dir()}};
    for (const auto &p : places) {
        if (path_kind(p.path) != PathKind::DIRECTORY) continue;
        if (ImGui::Selectable(p.label, dir_ == p.path)) navigate(p.path);
        ImGui::SetItemTooltip("%s", p.path.c_str());
    }
    if (recent_.empty()) return;
    ImGui::Spacing();
    ImGui::TextDisabled("Recent");
    for (size_t i = 0; i < recent_.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        const bool exists = path_kind(recent_[i]) != PathKind::MISSING;
        ImGui::BeginDisabled(!exists);
        if (ImGui::Selectable(base_name(recent_[i]).c_str())) {
            *chosen = recent_[i];
            *picked = true;
        }
        ImGui::EndDisabled();
        ImGui::SetItemTooltip("%s%s", recent_[i].c_str(), exists ? "" : " (missing)");
        ImGui::PopID();
    }
}

void FilePicker::draw_listing(std::string *chosen, bool *picked) {
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingStretchProp;
    if (!ImGui::BeginTable("##files", 3, flags)) return;
    if (scroll_top_) {
        ImGui::SetScrollY(0);
        scroll_top_ = false;
    }
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 6.0f);
    ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthStretch, 1.2f);
    ImGui::TableSetupColumn("Modified", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableHeadersRow();
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(entries_.size()));
    std::string enter_dir;
    while (clipper.Step()) {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
            const PickerEntry &e = entries_[static_cast<size_t>(i)];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::PushID(i);
            const std::string label = e.dir ? e.name + "/" : e.name;
            if (e.dir) ImGui::PushStyleColor(ImGuiCol_Text, tok().FOLDER);
            const bool clicked = ImGui::Selectable(label.c_str(), selected_ == e.name,
                ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowDoubleClick);
            if (e.dir) ImGui::PopStyleColor();
            if (clicked) {
                selected_ = e.name;
                if (save_ && !e.dir) {
                    save_name_ = e.name;
                    error_.clear();
                }
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    if (e.dir) {
                        enter_dir = join_path(dir_, e.name);
                    } else if (save_) {
                        try_save(join_path(dir_, e.name), chosen, picked);
                    } else {
                        *chosen = join_path(dir_, e.name);
                        *picked = true;
                    }
                }
            }
            ImGui::PopID();
            if (e.dir) {
                ImGui::TableNextColumn();
                ImGui::TableNextColumn();
                continue;
            }
            fill_stat(dir_, e);
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", size_text(e.size).c_str());
            ImGui::TableNextColumn();
            ImGui::TextDisabled("%s", time_text(e.mtime).c_str());
        }
    }
    ImGui::EndTable();
    if (!enter_dir.empty()) navigate(enter_dir);
}

}  // namespace rmwork
