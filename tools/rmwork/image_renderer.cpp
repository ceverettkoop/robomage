#include "image_renderer.h"

#include <algorithm>

namespace rmwork {

ImTextureData *ImageCardRenderer::texture(const CardLook &look, float width) const {
    if (!look.face || !images_) return nullptr;
    const std::string file = images_->file_for(*look.face);
    if (file.empty()) return nullptr;
    const float scale = std::max(1.0f, ImGui::GetIO().DisplayFramebufferScale.x);
    return images_->textures().get(file, width * scale);
}

bool ImageCardRenderer::draws_image(const CardLook &look, float width) const {
    return texture(look, width) != nullptr;
}

bool ImageCardRenderer::other_face_differs(const CardLook &look) const {
    if (!look.other || !look.face || !images_) return look.other != nullptr;
    const std::string a = images_->file_for(*look.face), b = images_->file_for(*look.other);
    return a.empty() || b.empty() || a != b;
}

void ImageCardRenderer::draw(
    ImDrawList *dl, const CardLook &look, ImVec2 slot_min, ImVec2 slot_max, bool rotated, float alpha) const {
    const float sw = slot_max.x - slot_min.x, sh = slot_max.y - slot_min.y;
    const float w = rotated ? sh : sw, h = rotated ? sw : sh;
    ImTextureData *tex = texture(look, w);
    if (!tex) {
        text_.draw(dl, look, slot_min, slot_max, rotated, alpha);
        return;
    }
    const ImVec2 center((slot_min.x + slot_max.x) / 2, (slot_min.y + slot_max.y) / 2);
    const ImVec2 a(center.x - w / 2, center.y - h / 2), b(center.x + w / 2, center.y + h / 2);
    const CardPass pass = begin_card_pass(dl, rotated);
    dl->AddImageRounded(tex->GetTexRef(), a, b, ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, w * 0.045f);
    end_card_pass(dl, pass, center, alpha);
}

}  // namespace rmwork
