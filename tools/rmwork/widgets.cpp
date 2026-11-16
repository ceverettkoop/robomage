#include "widgets.h"

#include <algorithm>
#include <cstdio>

#include "theme.h"

namespace {

const float HALF_PI = 1.57079632679f;
const int PALETTE_STRIP_SWATCHES = 8;

}  // namespace

static ImU32 fade(ImU32 color, float alpha);
static ImVec2 at(ImVec2 pos, float x, float y);
static void triangle(ImDrawList *dl, ImVec2 pos, ImVec2 a, ImVec2 b, ImVec2 c, ImU32 color);
static void polyline(ImDrawList *dl, ImVec2 pos, const ImVec2 *pts, int n, ImU32 color, bool closed, float width);
static void draw_logo(ImDrawList *dl, ImVec2 pos, float alpha);
static void noise(ImDrawList *dl, ImVec2 min, ImVec2 max, float alpha);
static void swatch_frame(ImDrawList *dl, ImVec2 min, ImVec2 max, bool selected, bool hovered);

static ImU32 fade(ImU32 color, float alpha) {
    const ImU32 a = (color >> IM_COL32_A_SHIFT) & 0xFF;
    const ImU32 faded = static_cast<ImU32>(static_cast<float>(a) * alpha);
    return (color & ~IM_COL32_A_MASK) | (faded << IM_COL32_A_SHIFT);
}

static ImVec2 at(ImVec2 pos, float x, float y) {
    return ImVec2(pos.x + x, pos.y + y);
}

static void triangle(ImDrawList *dl, ImVec2 pos, ImVec2 a, ImVec2 b, ImVec2 c, ImU32 color) {
    dl->AddTriangleFilled(at(pos, a.x, a.y), at(pos, b.x, b.y), at(pos, c.x, c.y), color);
}

static void polyline(ImDrawList *dl, ImVec2 pos, const ImVec2 *pts, int n, ImU32 color, bool closed, float width) {
    for (int i = 0; i < n; ++i) dl->PathLineTo(at(pos, pts[i].x, pts[i].y));
    dl->PathStroke(color, width, closed ? ImDrawFlags_Closed : ImDrawFlags_None);
}

// A rounded square with an "R" (the mock-up's logo), in the theme's logo colours.
static void draw_logo(ImDrawList *dl, ImVec2 pos, float alpha) {
    dl->AddRectFilled(at(pos, 1, 1), at(pos, 17, 17), fade(rmwork::tok().LOGO_BG, alpha), 2.0f);
    const ImU32 letter = fade(rmwork::tok().LOGO_FG, alpha);
    dl->PathLineTo(at(pos, 5, 13));
    dl->PathLineTo(at(pos, 5, 5));
    dl->PathLineTo(at(pos, 9, 5));
    dl->PathArcTo(at(pos, 9, 7.5f), 2.5f, -HALF_PI, HALF_PI, 10);
    dl->PathLineTo(at(pos, 5, 10));
    dl->PathStroke(letter, 1.6f);
    dl->AddLine(at(pos, 9, 10), at(pos, 12, 13), letter, 1.6f);
}

// Sparse dark specks, deterministic in the rectangle's size.
static void noise(ImDrawList *dl, ImVec2 min, ImVec2 max, float alpha) {
    const float w = max.x - min.x, h = max.y - min.y;
    const int n = std::min(2000, static_cast<int>(w * h / 30.0f));
    uint32_t s = 2463534242u;
    for (int i = 0; i < n; ++i) {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        const float x = min.x + static_cast<float>(s % 10000) / 10000.0f * w;
        const float y = min.y + static_cast<float>((s / 10000) % 10000) / 10000.0f * h;
        dl->AddRectFilled(ImVec2(x, y), ImVec2(x + 1, y + 1), fade(IM_COL32_BLACK, alpha));
    }
}

static void swatch_frame(ImDrawList *dl, ImVec2 min, ImVec2 max, bool selected, bool hovered) {
    if (selected)
        dl->AddRect(ImVec2(min.x - 2, min.y - 2), ImVec2(max.x + 2, max.y + 2), rmwork::tok().ACCENT, 0.0f, 2.0f);
    else
        dl->AddRect(min, max, hovered ? rmwork::tok().DIM : rmwork::tok().LINE);
}

namespace rmwork {

ImVec2 icon_size(Icon icon) {
    switch (icon) {
        case Icon::LOGO:
            return ImVec2(18, 18);
        case Icon::MODE_PLAY:
            return ImVec2(10, 10);
        case Icon::MODE_REPLAY:
            return ImVec2(12, 10);
        case Icon::OPEN:
        case Icon::SAVE:
            return ImVec2(16, 16);
        case Icon::SWING:
            return ImVec2(16, 14);
        case Icon::FIRST:
        case Icon::PREV:
        case Icon::PLAY:
        case Icon::PAUSE:
        case Icon::NEXT:
        case Icon::LAST:
            break;
    }
    return ImVec2(14, 14);
}

void draw_icon(ImDrawList *dl, Icon icon, ImVec2 pos, float alpha) {
    const ImU32 text = fade(tok().TEXT, alpha);
    switch (icon) {
        case Icon::LOGO:
            draw_logo(dl, pos, alpha);
            break;
        case Icon::OPEN: {
            const ImVec2 p[] = {{1.5f, 4.5f}, {6.5f, 4.5f}, {7.5f, 6}, {14.5f, 6}, {14.5f, 13.5f}, {1.5f, 13.5f}};
            polyline(dl, pos, p, 6, fade(tok().FOLDER, alpha), true, 1.0f);
            break;
        }
        case Icon::SAVE: {
            const ImU32 c = fade(tok().S_A, alpha);
            const ImVec2 body[] = {{2.5f, 2.5f}, {11.5f, 2.5f}, {13.5f, 4.5f}, {13.5f, 13.5f}, {2.5f, 13.5f}};
            const ImVec2 shutter[] = {{5, 2.5f}, {5, 6.5f}, {11, 6.5f}, {11, 2.5f}};
            const ImVec2 label[] = {{5, 13.5f}, {5, 9.5f}, {11, 9.5f}, {11, 13.5f}};
            polyline(dl, pos, body, 5, c, true, 1.0f);
            polyline(dl, pos, shutter, 4, c, false, 1.0f);
            polyline(dl, pos, label, 4, c, false, 1.0f);
            break;
        }
        case Icon::MODE_PLAY:
            triangle(dl, pos, {1, 1}, {1, 9}, {8, 5}, fade(tok().GOOD, alpha));
            break;
        case Icon::MODE_REPLAY:
            triangle(dl, pos, {6, 1}, {6, 9}, {1, 5}, fade(tok().S_A, alpha));
            triangle(dl, pos, {11, 1}, {11, 9}, {6, 5}, fade(tok().S_A, alpha));
            break;
        case Icon::FIRST:
            dl->AddRectFilled(at(pos, 1.5f, 2), at(pos, 3, 12), text);
            triangle(dl, pos, {12, 2}, {12, 12}, {4, 7}, text);
            break;
        case Icon::PREV:
            triangle(dl, pos, {10, 2}, {10, 12}, {3, 7}, text);
            break;
        case Icon::PLAY:
            triangle(dl, pos, {4, 2}, {4, 12}, {11, 7}, fade(tok().GOOD, alpha));
            break;
        case Icon::PAUSE:
            dl->AddRectFilled(at(pos, 3.5f, 2), at(pos, 6, 12), fade(tok().GOOD, alpha));
            dl->AddRectFilled(at(pos, 8.5f, 2), at(pos, 11, 12), fade(tok().GOOD, alpha));
            break;
        case Icon::NEXT:
            triangle(dl, pos, {4, 2}, {4, 12}, {11, 7}, text);
            break;
        case Icon::LAST:
            triangle(dl, pos, {2, 2}, {2, 12}, {10, 7}, text);
            dl->AddRectFilled(at(pos, 11, 2), at(pos, 12.5f, 12), text);
            break;
        case Icon::SWING: {
            const ImVec2 p[] = {{1, 3}, {6, 4}, {9, 12}, {15, 11}};
            polyline(dl, pos, p, 4, fade(tok().BAD, alpha), false, 2.0f);
            break;
        }
    }
}

bool icon_button(const char *id, Icon icon, const char *tooltip, bool enabled) {
    const ImVec2 size(24, ImGui::GetFrameHeight());
    ImGui::BeginDisabled(!enabled);
    const bool clicked = ImGui::InvisibleButton(id, size);
    ImGui::EndDisabled();
    const ImVec2 min = ImGui::GetItemRectMin();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const bool hovered = ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled);
    if (hovered && enabled) dl->AddRectFilled(min, ImGui::GetItemRectMax(), tok().HEADER, 2.0f);
    const ImVec2 is = icon_size(icon);
    draw_icon(dl, icon, ImVec2(min.x + (size.x - is.x) / 2, min.y + (size.y - is.y) / 2), enabled ? 1.0f : 0.35f);
    if (hovered && tooltip) ImGui::SetTooltip("%s", tooltip);
    return clicked && enabled;
}

void toolbar_separator() {
    ImGui::SameLine(0, 5);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetFrameHeight();
    const float rule = std::min(18.0f, h);
    ImGui::GetWindowDrawList()->AddLine(
        ImVec2(p.x, p.y + (h - rule) / 2), ImVec2(p.x, p.y + (h + rule) / 2), tok().LINE);
    ImGui::Dummy(ImVec2(1, h));
    ImGui::SameLine(0, 5);
}

bool segmented(const char *id, const Segment *segments, int count, int *selected) {
    bool changed = false;
    ImGui::PushID(id);
    ImDrawList *dl = ImGui::GetWindowDrawList();
    const float h = ImGui::GetFrameHeight();
    const ImVec2 start = ImGui::GetCursorScreenPos();
    ImVec2 end = start;
    for (int i = 0; i < count; ++i) {
        const Segment &s = segments[i];
        const float icon_w = s.has_icon ? icon_size(s.icon).x + 5 : 0;
        const float w = ImGui::CalcTextSize(s.label).x + icon_w + 20;
        if (i) ImGui::SameLine(0, 0);
        ImGui::PushID(i);
        if (ImGui::InvisibleButton("##seg", ImVec2(w, h)) && i != *selected) {
            *selected = i;
            changed = true;
        }
        ImGui::PopID();
        const bool hovered = ImGui::IsItemHovered();
        const ImVec2 a = ImGui::GetItemRectMin(), b = ImGui::GetItemRectMax();
        const ImU32 bg = i == *selected ? tok().SEL : hovered ? tok().FIELD_HOVER : tok().FIELD;
        ImDrawFlags corners = ImDrawFlags_RoundCornersNone;
        if (i == 0) corners |= ImDrawFlags_RoundCornersLeft;
        if (i == count - 1) corners |= ImDrawFlags_RoundCornersRight;
        dl->AddRectFilled(a, b, bg, 3.0f, corners);
        if (i) dl->AddLine(ImVec2(a.x, a.y), ImVec2(a.x, b.y), tok().LINE);
        float x = a.x + 10;
        if (s.has_icon) {
            const ImVec2 is = icon_size(s.icon);
            draw_icon(dl, s.icon, ImVec2(x, a.y + (h - is.y) / 2));
            x += icon_w;
        }
        dl->AddText(ImVec2(x, a.y + (h - ImGui::GetFontSize()) / 2), tok().TEXT, s.label);
        if (hovered && s.tooltip) ImGui::SetTooltip("%s", s.tooltip);
        end = b;
    }
    dl->AddRect(start, end, tok().LINE, 3.0f);
    ImGui::PopID();
    return changed;
}

bool texture_swatch(const char *id, const std::string &key, bool selected) {
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(56, 36));
    const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(min, max, tok().FIELD);
    draw_table_texture(dl, ImVec2(min.x + 3, min.y + 3), ImVec2(max.x - 3, max.y - 3), key);
    swatch_frame(dl, min, max, selected, ImGui::IsItemHovered());
    return clicked;
}

bool color_swatch(const char *id, ImU32 color, bool selected) {
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(56, 36));
    const ImVec2 min = ImGui::GetItemRectMin(), max = ImGui::GetItemRectMax();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(min, max, tok().FIELD);
    dl->AddRectFilled(ImVec2(min.x + 3, min.y + 3), ImVec2(max.x - 3, max.y - 3), color);
    swatch_frame(dl, min, max, selected, ImGui::IsItemHovered());
    return clicked;
}

void draw_table_texture(ImDrawList *dl, ImVec2 min, ImVec2 max, const std::string &key) {
    dl->PushClipRect(min, max, true);
    if (key == "felt") {
        dl->AddRectFilled(min, max, hex_color(0x0b6b2a));
        noise(dl, min, max, 0.32f);
    } else if (key == "linen") {
        dl->AddRectFilled(min, max, hex_color(0xb9ad92));
        for (float y = min.y; y < max.y; y += 3)
            dl->AddLine(ImVec2(min.x, y), ImVec2(max.x, y), IM_COL32(255, 255, 255, 31));
        for (float x = min.x; x < max.x; x += 3) dl->AddLine(ImVec2(x, min.y), ImVec2(x, max.y), IM_COL32(0, 0, 0, 23));
    } else if (key == "tiles") {
        for (float y = min.y; y < max.y; y += 8) {
            for (float x = min.x; x < max.x; x += 8) {
                const bool odd = (static_cast<int>((x - min.x) / 8) + static_cast<int>((y - min.y) / 8)) % 2 != 0;
                dl->AddRectFilled(ImVec2(x, y), ImVec2(x + 8, y + 8), hex_color(odd ? 0x007b7b : 0x006b6b));
            }
        }
    } else if (key == "slate") {
        const ImU32 light = hex_color(0x3c4650), dark = hex_color(0x262c33);
        dl->AddRectFilledMultiColor(min, max, light, light, dark, dark);
        noise(dl, min, max, 0.32f);
    } else {
        dl->AddRectFilled(min, max, hex_color(0x6e4526));
        for (float x = min.x; x < max.x; x += 11) {
            dl->AddLine(ImVec2(x, min.y), ImVec2(x, max.y), IM_COL32(0, 0, 0, 26));
            dl->AddLine(ImVec2(x + 5, min.y), ImVec2(x + 5, max.y), IM_COL32(255, 255, 255, 13));
        }
        for (float x = min.x + 158; x < max.x; x += 160)
            dl->AddRectFilled(ImVec2(x, min.y), ImVec2(x + 2, max.y), IM_COL32(0, 0, 0, 115));
        noise(dl, min, max, 0.32f);
    }
    dl->PopClipRect();
}

float palette_strip_width(float height) {
    return PALETTE_STRIP_SWATCHES * height + 2;
}

void palette_strip(const Palette &p, float height) {
    const ImU32 colors[PALETTE_STRIP_SWATCHES] = {
        p.BG, p.PANEL, p.TAB_ACTIVE, p.ACCENT, p.TEXT, p.S_A, p.S_B, p.LIFE_B};
    const ImVec2 min = ImGui::GetCursorScreenPos();
    ImGui::Dummy(ImVec2(palette_strip_width(height), height));
    ImDrawList *dl = ImGui::GetWindowDrawList();
    for (int i = 0; i < PALETTE_STRIP_SWATCHES; ++i) {
        const float x = min.x + 1 + static_cast<float>(i) * height;
        dl->AddRectFilled(ImVec2(x, min.y), ImVec2(x + height, min.y + height), colors[i]);
    }
    dl->AddRect(min, ImVec2(min.x + palette_strip_width(height), min.y + height), tok().LINE);
}

bool parse_hex_color(const std::string &text, ImU32 *out) {
    if (text.size() != 7 || text[0] != '#') return false;
    unsigned int rgb = 0;
    for (size_t i = 1; i < 7; ++i) {
        const char c = text[i];
        unsigned int d = 0;
        if (c >= '0' && c <= '9')
            d = static_cast<unsigned int>(c - '0');
        else if (c >= 'a' && c <= 'f')
            d = static_cast<unsigned int>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F')
            d = static_cast<unsigned int>(c - 'A' + 10);
        else
            return false;
        rgb = rgb * 16 + d;
    }
    *out = hex_color(rgb);
    return true;
}

std::string format_hex_color(ImU32 color) {
    char buf[8];
    std::snprintf(buf, sizeof buf, "#%02x%02x%02x", (color >> IM_COL32_R_SHIFT) & 0xFF,
        (color >> IM_COL32_G_SHIFT) & 0xFF, (color >> IM_COL32_B_SHIFT) & 0xFF);
    return buf;
}

void centered_text_block(const char *title, const char *body, float max_width) {
    const ImVec2 avail = ImGui::GetContentRegionAvail();
    const float wrap = std::max(40.0f, std::min(max_width, avail.x - 12));
    const ImVec2 title_size = ImGui::CalcTextSize(title);
    const ImVec2 body_size = ImGui::CalcTextSize(body, nullptr, false, wrap);
    const float block_w = std::max(title_size.x, body_size.x);
    const float block_h = title_size.y + ImGui::GetStyle().ItemSpacing.y + body_size.y;
    const ImVec2 origin = ImGui::GetCursorPos();
    const float left = origin.x + std::max(0.0f, (avail.x - block_w) / 2);
    ImGui::SetCursorPos(ImVec2(
        origin.x + std::max(0.0f, (avail.x - title_size.x) / 2), origin.y + std::max(0.0f, (avail.y - block_h) / 2)));
    ImGui::TextUnformatted(title);
    ImGui::SetCursorPosX(left);
    ImGui::PushTextWrapPos(left + wrap);
    ImGui::PushStyleColor(ImGuiCol_Text, tok().DIM);
    ImGui::TextUnformatted(body);
    ImGui::PopStyleColor();
    ImGui::PopTextWrapPos();
}

}  // namespace rmwork
