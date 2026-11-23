#include "modal.h"

#include <algorithm>

#include "fonts.h"
#include "theme.h"

namespace rmwork {

float em_scale() {
    return ImGui::GetFontSize() / static_cast<float>(FontChoice().size);
}

void begin_modal_frame(ImVec2 size) {
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    const float k = em_scale();
    const ImVec2 scaled(std::min(size.x * k, vp->WorkSize.x - 16), std::min(size.y * k, vp->WorkSize.y - 16));
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(scaled, ImGuiCond_Appearing);
    ImGui::PushStyleColor(ImGuiCol_TitleBg, tok().TITLE_ACTIVE);
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, tok().TITLE_ACTIVE);
    ImGui::PushStyleColor(ImGuiCol_Text, tok().TITLE_TEXT);
}

bool close_requested() {
    return ImGui::IsKeyPressed(ImGuiKey_Escape, false);
}

float footer_height() {
    return ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y + 1;
}

bool right_aligned_button(const char *label) {
    const float w = ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2 + 12;
    const float avail = ImGui::GetContentRegionAvail().x;
    if (avail > w) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - w);
    return ImGui::Button(label, ImVec2(w, 0));
}

void note(const char *text) {
    ImGui::PushStyleColor(ImGuiCol_Text, tok().DIM);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

void field_label(const char *label, float field_x) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(field_x);
}

float char_width() {
    return ImGui::CalcTextSize("M").x;
}

void colored_note(const char *text, ImU32 color) {
    ImGui::PushStyleColor(ImGuiCol_Text, color);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

}  // namespace rmwork
