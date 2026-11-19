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

MagnifierRects magnifier_rects(
    ImVec2 min, ImVec2 max, ImVec2 vp_min, ImVec2 vp_max, int faces, float caption_h, float box_h) {
    MagnifierRects r;
    r.faces = faces == 2 ? 2 : 1;
    const float gap = 8, margin = 8, below = 6;
    const float vw = vp_max.x - vp_min.x, vh = vp_max.y - vp_min.y;
    const float cap = r.faces == 2 ? caption_h : 0;
    float mw = std::max(200.0f, std::min(340.0f, vh * 0.42f / CARD_ASPECT));
    // Shrunk to the screen: the faces across its width, the faces, captions and box down its height.
    const float fit_w = (vw - 2 * margin - gap * static_cast<float>(r.faces - 1)) / static_cast<float>(r.faces);
    const float fit_h = (vh - 2 * margin - cap - below - box_h) / CARD_ASPECT;
    mw = std::max(40.0f, std::min(mw, std::min(fit_w, fit_h)));
    const float mh = mw * CARD_ASPECT;
    const float total_w = static_cast<float>(r.faces) * mw + gap * static_cast<float>(r.faces - 1);
    float left = max.x + 12;
    if (left + total_w > vp_max.x - margin) left = min.x - total_w - 12;
    if (left < vp_min.x + margin) left = vp_min.x + (vw - total_w) / 2;
    const float total = cap + mh + below + box_h;
    float top = (min.y + max.y) / 2 - total / 2;
    top = std::max(vp_min.y + margin, std::min(top, vp_max.y - total - margin));
    for (int i = 0; i < r.faces; ++i) {
        const float x = left + static_cast<float>(i) * (mw + gap);
        r.caption_min[i] = ImVec2(x, top);
        r.face_min[i] = ImVec2(x, top + cap);
        r.face_max[i] = ImVec2(x + mw, top + cap + mh);
    }
    r.box_min = ImVec2(left, top + cap + mh + below);
    r.box_max = ImVec2(left + total_w, r.box_min.y + box_h);
    return r;
}

void draw_magnified_card(const CardRenderer &renderer, const CardLook &look, const std::vector<std::string> &lines,
    ImVec2 min, ImVec2 max, const std::function<void(ImDrawList *, ImVec2, ImVec2)> &overlay) {
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    const ImVec2 vp_max(vp->Pos.x + vp->Size.x, vp->Pos.y + vp->Size.y);
    const bool both = renderer.other_face_differs(look);
    const float lh = ImGui::GetTextLineHeight();
    // The box's lines wrap to the width the faces take; their count fixes its height.
    const MagnifierRects probe = magnifier_rects(min, max, vp->Pos, vp_max, both ? 2 : 1, lh + 4, 0);
    const float box_w = probe.box_max.x - probe.box_min.x;
    float box_h = 12;
    std::vector<std::vector<std::string>> wrapped;
    for (const std::string &l : lines) {
        wrapped.push_back(wrap_text(ImGui::GetFont(), ImGui::GetFontSize(), l, box_w - 16, 3));
        box_h += lh * static_cast<float>(wrapped.back().size());
    }
    const MagnifierRects r = magnifier_rects(min, max, vp->Pos, vp_max, both ? 2 : 1, lh + 4, box_h);
    ImDrawList *fg = ImGui::GetForegroundDrawList();
    for (int i = 0; i < r.faces; ++i) {
        CardLook face = look;
        if (i == 1) {
            face.face = look.other;
            face.other = look.face;
            face.colors = look.other->colors;
            face.land = look.other->land;
            face.land_mana = look.other->land ? basic_land_mana(look.other->subtypes) : 0;
        }
        const ImVec2 a = r.face_min[i], b = r.face_max[i];
        const float mw = b.x - a.x;
        fg->AddRectFilled(ImVec2(a.x + 3, a.y + 5), ImVec2(b.x + 3, b.y + 5), hex_color(0x000000, 120), mw * 0.05f);
        renderer.draw(fg, face, a, b, false, 1.0f);
        if (i == 0 && overlay) overlay(fg, a, b);
        if (r.faces < 2) continue;
        // The captions: the face up first and outlined, then the other face.
        const bool back = face.face && face.face->back;
        const std::string caption = i == 0 ? std::string("Face up \xC2\xB7 ") + (back ? "back face" : "front face")
                                           : std::string(back ? "Back face" : "Front face");
        const ImVec2 c = r.caption_min[i];
        const float tw = std::min(mw - 12, ImGui::CalcTextSize(caption.c_str()).x);
        fg->AddRectFilled(c, ImVec2(c.x + tw + 12, c.y + lh + 2), tok().DROP, 2);
        fg->AddRect(c, ImVec2(c.x + tw + 12, c.y + lh + 2), i == 0 ? tok().GLOW : tok().LINE, 2);
        draw_fitted_text(fg, ImGui::GetFont(), ImGui::GetFontSize(), ImVec2(c.x + 6, c.y + 1),
            i == 0 ? tok().TEXT : tok().DIM, caption, mw - 12);
        if (i == 0)
            fg->AddRect(ImVec2(a.x - 2, a.y - 2), ImVec2(b.x + 2, b.y + 2), tok().GLOW, mw * 0.05f + 2, 0, 2.0f);
    }
    fg->AddRectFilled(r.box_min, r.box_max, tok().DROP, 2);
    fg->AddRect(r.box_min, r.box_max, tok().LINE, 2);
    float y = r.box_min.y + 6;
    for (size_t i = 0; i < wrapped.size(); ++i)
        for (const std::string &l : wrapped[i]) {
            fg->AddText(ImVec2(r.box_min.x + 8, y), i == 0 ? tok().TEXT : tok().DIM, l.c_str());
            y += lh;
        }
}

}  // namespace rmwork
