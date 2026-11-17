#include "magnifier.h"

#include <algorithm>

#include "table_layout.h"
#include "theme.h"

namespace rmwork {

void MagnifierState::update(uint64_t under, bool over, ImGuiKeyChord key) {
    const ImGuiKey k = static_cast<ImGuiKey>(key & ~ImGuiMod_Mask_);
    const bool key_down = key != 0 && !ImGui::GetIO().WantTextInput && ImGui::IsKeyDown(k);
    if (key_down && under) {
        handle_ = under;
        by_click_ = false;
    } else if (!by_click_ && (!key_down || !under)) {
        handle_ = 0;
    }
    if (under && over && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        handle_ = under;
        by_click_ = true;
    } else if (by_click_ && (under != handle_ || ImGui::IsMouseClicked(ImGuiMouseButton_Left) ||
                                ImGui::IsKeyPressed(ImGuiKey_Escape, false))) {
        handle_ = 0;
        by_click_ = false;
    }
}

void MagnifierState::open(uint64_t handle) {
    handle_ = handle;
    by_click_ = true;
}

void MagnifierState::close() {
    handle_ = 0;
    by_click_ = false;
}

void draw_magnified_card(const CardRenderer &renderer, const CardLook &look, const std::vector<std::string> &lines,
    ImVec2 min, ImVec2 max, const std::function<void(ImDrawList *, ImVec2, ImVec2)> &overlay) {
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    const float mw = std::max(200.0f, std::min(340.0f, vp->Size.y * 0.42f / CARD_ASPECT));
    const float mh = mw * CARD_ASPECT;
    const float lh = ImGui::GetTextLineHeight();
    float box_h = 0;
    std::vector<std::vector<std::string>> wrapped;
    for (const std::string &l : lines) {
        wrapped.push_back(wrap_text(ImGui::GetFont(), ImGui::GetFontSize(), l, mw - 16, 3));
        box_h += lh * static_cast<float>(wrapped.back().size());
    }
    box_h += 12;
    float left = max.x + 12;
    if (left + mw > vp->Pos.x + vp->Size.x - 8) left = min.x - mw - 12;
    if (left < vp->Pos.x + 8) left = vp->Pos.x + (vp->Size.x - mw) / 2;
    const float total = mh + 6 + box_h;
    float top = (min.y + max.y) / 2 - total / 2;
    top = std::max(vp->Pos.y + 8, std::min(top, vp->Pos.y + vp->Size.y - total - 8));
    ImDrawList *fg = ImGui::GetForegroundDrawList();
    fg->AddRectFilled(
        ImVec2(left + 3, top + 5), ImVec2(left + mw + 3, top + mh + 5), hex_color(0x000000, 120), mw * 0.05f);
    renderer.draw(fg, look, ImVec2(left, top), ImVec2(left + mw, top + mh), false, 1.0f);
    if (overlay) overlay(fg, ImVec2(left, top), ImVec2(left + mw, top + mh));
    const ImVec2 ba(left, top + mh + 6), bz(left + mw, top + mh + 6 + box_h);
    fg->AddRectFilled(ba, bz, tok().DROP, 2);
    fg->AddRect(ba, bz, tok().LINE, 2);
    float y = ba.y + 6;
    for (size_t i = 0; i < wrapped.size(); ++i)
        for (const std::string &l : wrapped[i]) {
            fg->AddText(ImVec2(ba.x + 8, y), i == 0 ? tok().TEXT : tok().DIM, l.c_str());
            y += lh;
        }
}

}  // namespace rmwork
