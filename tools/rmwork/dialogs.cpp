#include "dialogs.h"

#include "modal.h"
#include "theme.h"

namespace {

const char *const ERROR_POPUP_ID = "###rmwork.error";

}  // namespace

namespace rmwork {

void ErrorDialog::open(const std::string &title, const std::string &message, const std::string &detail) {
    title_ = title;
    message_ = message;
    detail_ = detail;
    want_open_ = true;
}

std::string ErrorDialog::diagnostics() const {
    return title_ + "\n\n" + message_ + "\n\n" + detail_ + (detail_.empty() || detail_.back() == '\n' ? "" : "\n");
}

bool ErrorDialog::draw() {
    const std::string name = title_ + ERROR_POPUP_ID;
    if (want_open_) {
        ImGui::OpenPopup(name.c_str());
        want_open_ = false;
    }
    // Without a detail the dialog is the message and the footer, its height fitted to them.
    begin_modal_frame(ImVec2(720, detail_.empty() ? 0 : 460));
    bool keep = true;
    const bool visible = ImGui::BeginPopupModal(name.c_str(), &keep, ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleColor(3);
    open_ = visible;
    if (!visible) return false;
    ImGui::TextWrapped("%s", message_.c_str());
    ImGui::Spacing();
    if (!detail_.empty()) {
        ImGui::TextDisabled("Details");
        ImGui::PushStyleColor(ImGuiCol_FrameBg, tok().PLOT_BG);
        ImGui::InputTextMultiline("##detail", detail_.data(), detail_.size() + 1, ImVec2(-1, -footer_height()),
            ImGuiInputTextFlags_ReadOnly);
        ImGui::PopStyleColor();
    }
    ImGui::Separator();
    const bool copied = ImGui::Button("Copy diagnostics");
    if (copied) ImGui::SetClipboardText(diagnostics().c_str());
    ImGui::SameLine();
    if (right_aligned_button("Close") || close_requested() || !keep) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
    return copied;
}

}  // namespace rmwork
