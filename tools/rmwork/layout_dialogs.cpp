#include "layout_dialogs.h"

#include <cstdio>

#include "modal.h"
#include "theme.h"
#include "widgets.h"

namespace {

const char *const SAVE_POPUP = "Save Layout As###rmwork.save_layout";
const char *const MANAGE_POPUP = "Manage Layouts###rmwork.manage_layouts";

}  // namespace

namespace rmwork {

void LayoutDialogs::open_save(Mode m) {
    mode_ = m;
    want_save_ = true;
    name_[0] = '\0';
    confirm_overwrite_ = false;
    focus_name_ = true;
    error_.clear();
}

void LayoutDialogs::open_manage(Mode m) {
    mode_ = m;
    want_manage_ = true;
    renaming_ = confirm_delete_ = -1;
    manage_error_.clear();
}

void LayoutDialogs::set_save_name(const std::string &name) {
    std::snprintf(name_, sizeof name_, "%s", name.c_str());
    confirm_overwrite_ = false;
    error_.clear();
}

bool LayoutDialogs::draw(Settings *s, Mode *mode, std::string *name) {
    if (want_save_) {
        ImGui::OpenPopup(SAVE_POPUP);
        want_save_ = false;
    }
    if (want_manage_) {
        ImGui::OpenPopup(MANAGE_POPUP);
        want_manage_ = false;
    }
    draw_save(s);
    std::string show;
    if (!draw_manage(s, &show)) return false;
    *mode = mode_;
    *name = show;
    return true;
}

// A name field, Save and Cancel; a name already taken turns Save into an Overwrite question.
void LayoutDialogs::draw_save(Settings *s) {
    begin_modal_frame(ImVec2(520, 0));
    bool keep = true;
    save_open_ =
        ImGui::BeginPopupModal(SAVE_POPUP, &keep, ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleColor(3);
    if (!save_open_) return;
    ImGui::Text("Save the %s layout as it is now (splits, tab groups, open tools).", mode_title(mode_));
    ImGui::Spacing();
    field_label("Name", 6 * char_width());
    ImGui::SetNextItemWidth(36 * char_width());
    if (focus_name_) {
        ImGui::SetKeyboardFocusHere();
        focus_name_ = false;
    }
    const bool enter = ImGui::InputText("##name", name_, sizeof name_, ImGuiInputTextFlags_EnterReturnsTrue);
    if (ImGui::IsItemEdited()) {
        confirm_overwrite_ = false;
        error_.clear();
    }
    if (!error_.empty()) colored_note(error_.c_str(), confirm_overwrite_ ? tok().DIM : tok().BAD);
    ImGui::Spacing();
    ImGui::Separator();
    bool close = !keep || close_requested();
    if (confirm_overwrite_) {
        if (ImGui::Button("Overwrite")) {
            commit_save(s, true);
            close = error_.empty();
        }
    } else if (ImGui::Button("Save") || enter) {
        commit_save(s, false);
        close = error_.empty();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel")) close = true;
    if (close) {
        ImGui::CloseCurrentPopup();
        save_open_ = false;
    }
    ImGui::EndPopup();
}

// Saves the mode's dockspace under the typed name; leaves a message in error_ when it does not
// (a taken name sets confirm_overwrite_ to ask).
void LayoutDialogs::commit_save(Settings *s, bool overwrite) {
    const LayoutNode root = capture_layout(mode_, s->tools[mode_index(mode_)]);
    std::string why;
    const SaveResult r = save_named_layout(&s->layouts[mode_index(mode_)], name_, root, overwrite, &why);
    error_.clear();
    if (r == SaveResult::EXISTS) {
        confirm_overwrite_ = true;
        error_ = why + ". Overwrite it with the layout as it is now?";
    } else if (r == SaveResult::INVALID) {
        error_ = why;
    }
}

// The saved layouts of the chosen mode, each with Show, Rename and Delete. True when Show was
// pressed (`*show` names the layout; the dialog closes).
bool LayoutDialogs::draw_manage(Settings *s, std::string *show) {
    begin_modal_frame(ImVec2(620, 380));
    bool keep = true;
    manage_open_ = ImGui::BeginPopupModal(MANAGE_POPUP, &keep, ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleColor(3);
    if (!manage_open_) return false;
    const Segment modes[] = {
        {"Play", false, Icon::LOGO, "Play's saved layouts"}, {"Replay", false, Icon::LOGO, "Replay's saved layouts"}};
    int sel = mode_index(mode_);
    if (segmented("##mode", modes, 2, &sel)) {
        mode_ = static_cast<Mode>(sel);
        renaming_ = confirm_delete_ = -1;
        manage_error_.clear();
    }
    ImGui::SameLine();
    ImGui::AlignTextToFramePadding();
    ImGui::TextDisabled("View → Save Layout As… adds the current one.");
    ImGui::Spacing();
    bool shown = false;
    NamedLayouts &all = s->layouts[mode_index(mode_)];
    ImGui::BeginChild("##list", ImVec2(0, -footer_height()));
    if (all.empty()) ImGui::TextDisabled("No saved %s layouts.", mode_title(mode_));
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter;
    if (!all.empty() && ImGui::BeginTable("##layouts", 2, flags)) {
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("##actions", ImGuiTableColumnFlags_WidthFixed, 22 * char_width());
        ImGui::TableHeadersRow();
        for (int i = 0; i < static_cast<int>(all.size()) && !shown; ++i) shown = draw_manage_row(&all, i, show);
        ImGui::EndTable();
    }
    if (!manage_error_.empty()) colored_note(manage_error_.c_str(), tok().BAD);
    ImGui::EndChild();
    ImGui::Separator();
    if (shown || right_aligned_button("Close") || !keep || (renaming_ < 0 && close_requested())) {
        ImGui::CloseCurrentPopup();
        manage_open_ = false;
    }
    ImGui::EndPopup();
    return shown;
}

// One saved layout: its name (or the rename field) and its buttons. True when Show was pressed.
bool LayoutDialogs::draw_manage_row(NamedLayouts *all, int i, std::string *show) {
    NamedLayout &l = (*all)[static_cast<size_t>(i)];
    ImGui::PushID(i);
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    bool shown = false;
    if (renaming_ == i) {
        ImGui::SetNextItemWidth(-1);
        const bool enter = ImGui::InputText("##rename", rename_, sizeof rename_, ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::TableNextColumn();
        if (ImGui::SmallButton("OK") || enter) {
            if (rename_named_layout(all, l.name, rename_, &manage_error_)) {
                renaming_ = -1;
                manage_error_.clear();
            }
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            renaming_ = -1;
            manage_error_.clear();
        }
    } else if (confirm_delete_ == i) {
        ImGui::AlignTextToFramePadding();
        ImGui::Text("Delete \"%s\"?", l.name.c_str());
        ImGui::TableNextColumn();
        if (ImGui::SmallButton("Delete")) {
            delete_named_layout(all, l.name);
            confirm_delete_ = -1;
            ImGui::PopID();
            return false;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Keep")) confirm_delete_ = -1;
    } else {
        ImGui::AlignTextToFramePadding();
        ImGui::TextUnformatted(l.name.c_str());
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", describe_layout(l.root).c_str());
        ImGui::TableNextColumn();
        if (ImGui::SmallButton("Show")) {
            *show = l.name;
            shown = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Rename")) {
            renaming_ = i;
            confirm_delete_ = -1;
            std::snprintf(rename_, sizeof rename_, "%s", l.name.c_str());
            manage_error_.clear();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Delete…")) {
            confirm_delete_ = i;
            renaming_ = -1;
        }
    }
    ImGui::PopID();
    return shown;
}

}  // namespace rmwork
