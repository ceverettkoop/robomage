#include "tool_widgets.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "card_renderer.h"
#include "theme.h"

namespace {

// The library icon's card back (card content, as the Table's card backs).
const ImU32 BACK_FILL = rmwork::hex_color(0x8a2a2a);
const ImU32 BACK_OVAL = rmwork::hex_color(0xc08040);

}  // namespace

static ImVec2 icon_point(ImVec2 pos, float k, float x, float y);
static void rotated_card(ImDrawList *dl, ImVec2 pos, float k, float x, float y, float w, float h, float deg,
    ImVec2 pivot, ImU32 fill, ImU32 stroke);
static ImU32 with_alpha(ImU32 c, int alpha);

// A point of the 16-unit icon grid at scale `k`.
static ImVec2 icon_point(ImVec2 pos, float k, float x, float y) {
    return ImVec2(pos.x + x * k, pos.y + y * k);
}

// A card rectangle of the icon grid turned `deg` degrees about `pivot` (grid units).
static void rotated_card(ImDrawList *dl, ImVec2 pos, float k, float x, float y, float w, float h, float deg,
    ImVec2 pivot, ImU32 fill, ImU32 stroke) {
    const float r = deg * 3.14159265f / 180.0f, c = std::cos(r), s = std::sin(r);
    ImVec2 p[4] = {{x, y}, {x + w, y}, {x + w, y + h}, {x, y + h}};
    for (ImVec2 &q : p) {
        const float dx = q.x - pivot.x, dy = q.y - pivot.y;
        q = icon_point(pos, k, pivot.x + dx * c - dy * s, pivot.y + dx * s + dy * c);
    }
    dl->AddQuadFilled(p[0], p[1], p[2], p[3], fill);
    dl->AddQuad(p[0], p[1], p[2], p[3], stroke, std::max(1.0f, k));
}

static ImU32 with_alpha(ImU32 c, int alpha) {
    return (c & ~IM_COL32_A_MASK) | (static_cast<ImU32>(alpha) << IM_COL32_A_SHIFT);
}

namespace rmwork {

bool begin_grid(const char *id, int columns, ImVec2 size, bool scroll) {
    ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter |
                            ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingFixedFit;
    if (scroll) flags |= ImGuiTableFlags_ScrollY;
    return ImGui::BeginTable(id, columns, flags, size);
}

float seat_chip_width() {
    return ImGui::CalcTextSize("B").x + 12;
}

void seat_chip(int seat) {
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight();
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(p, ImVec2(p.x + seat_chip_width(), p.y + h), seat == 0 ? tok().SEAT_A : tok().SEAT_B, 2);
    dl->AddText(ImVec2(p.x + 6, p.y), tok().TEXT, seat == 0 ? "A" : "B");
    ImGui::Dummy(ImVec2(seat_chip_width(), h));
}

void draw_zone_icon(ImDrawList *dl, ZoneKind zone, ImVec2 pos, float size) {
    const float k = size / 16.0f;
    const ImU32 ink = tok().PANEL, light = tok().TEXT, mid = tok().DIM;
    const float t = std::max(1.0f, k);
    switch (zone) {
        case ZoneKind::HAND:
            rotated_card(dl, pos, k, 1.5f, 4.5f, 7, 10, -14, ImVec2(5, 10), light, ink);
            rotated_card(dl, pos, k, 4.5f, 3.5f, 7, 10, 0, ImVec2(8, 8), light, ink);
            rotated_card(dl, pos, k, 7.5f, 4.5f, 7, 10, 14, ImVec2(11, 10), mid, ink);
            return;
        case ZoneKind::LIBRARY:
            rotated_card(dl, pos, k, 4.5f, 1.5f, 9, 12, 0, ImVec2(0, 0), mid, ink);
            rotated_card(dl, pos, k, 3.5f, 2.5f, 9, 12, 0, ImVec2(0, 0), light, ink);
            rotated_card(dl, pos, k, 2.5f, 3.5f, 9, 12, 0, ImVec2(0, 0), BACK_FILL, ink);
            dl->AddEllipseFilled(icon_point(pos, k, 7, 9.5f), ImVec2(2.5f * k, 3.5f * k), BACK_OVAL);
            return;
        case ZoneKind::GRAVEYARD: {
            const ImVec2 c = icon_point(pos, k, 8, 6);
            dl->PathArcTo(c, 4.5f * k, 3.14159265f, 2 * 3.14159265f);
            dl->PathLineTo(icon_point(pos, k, 12.5f, 15.5f));
            dl->PathLineTo(icon_point(pos, k, 3.5f, 15.5f));
            dl->PathFillConvex(light);
            dl->PathArcTo(c, 4.5f * k, 3.14159265f, 2 * 3.14159265f);
            dl->PathLineTo(icon_point(pos, k, 12.5f, 15.5f));
            dl->PathLineTo(icon_point(pos, k, 3.5f, 15.5f));
            dl->PathStroke(ink, ImDrawFlags_Closed, t);
            dl->AddLine(icon_point(pos, k, 8, 5), icon_point(pos, k, 8, 11), ink, t);
            dl->AddLine(icon_point(pos, k, 6, 7), icon_point(pos, k, 10, 7), ink, t);
            dl->AddLine(icon_point(pos, k, 1, 15.5f), icon_point(pos, k, 15, 15.5f), tok().GOOD, 1.5f * k);
            return;
        }
        case ZoneKind::EXILE: {
            const ImVec2 c = icon_point(pos, k, 8, 8);
            dl->AddCircleFilled(c, 6.5f * k, light);
            dl->AddCircle(c, 6.5f * k, ink, 0, t);
            // A spiral inward from the right.
            const int n = 28;
            for (int i = 0; i < n; ++i) {
                const float a = static_cast<float>(i) / n * 2.6f * 3.14159265f;
                const float r = (4.6f - 3.4f * static_cast<float>(i) / n) * k;
                dl->PathLineTo(ImVec2(c.x + r * std::cos(a), c.y + r * std::sin(a)));
            }
            dl->PathStroke(tok().S_A6, 0, 1.4f * k);
            return;
        }
    }
}

bool zone_button(const char *id, ZoneKind zone, int count, const std::string &tooltip, bool active) {
    const float h = ImGui::GetTextLineHeight();
    const float icon = std::round(h * 16.0f / 17.0f);
    const std::string n = std::to_string(count);
    const float w = icon + 4 + ImGui::CalcTextSize(n.c_str()).x + 6;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const bool clicked = ImGui::InvisibleButton(id, ImVec2(w, h + 2));
    ImDrawList *dl = ImGui::GetWindowDrawList();
    if (ImGui::IsItemHovered()) dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h + 2), tok().HEADER, 2);
    if (active) dl->AddRect(p, ImVec2(p.x + w, p.y + h + 2), tok().ACCENT, 2.0f, ImDrawFlags_None, 2.0f);
    draw_zone_icon(dl, zone, ImVec2(p.x + 3, p.y + 1 + (h - icon) / 2), icon);
    dl->AddText(ImVec2(p.x + 3 + icon + 4, p.y + 1), tok().TEXT, n.c_str());
    if (!tooltip.empty()) ImGui::SetItemTooltip("%s", tooltip.c_str());
    return clicked;
}

void draw_bar(ImDrawList *dl, ImVec2 pos, float width, float height, float fraction, ImU32 fill) {
    fraction = std::max(0.0f, std::min(1.0f, fraction));
    dl->AddRectFilled(pos, ImVec2(pos.x + width, pos.y + height), tok().FIELD);
    if (fraction > 0) dl->AddRectFilled(pos, ImVec2(pos.x + std::max(1.0f, width * fraction), pos.y + height), fill);
}

void value_bar(float fraction, float width, ImU32 fill) {
    const float h = ImGui::GetTextLineHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float bh = std::round(h * 0.6f);
    draw_bar(ImGui::GetWindowDrawList(), ImVec2(p.x, p.y + (h - bh) / 2), width, bh, fraction, fill);
    ImGui::Dummy(ImVec2(width, h));
}

ImU32 signed_color(double v) {
    if (std::fabs(v) < 0.005) return tok().TEXT;
    return v > 0 ? tok().S_A : tok().LIFE_B;
}

ImU32 heat_color(double v, double scale, bool is_signed, int max_alpha) {
    const double f = scale > 0 ? std::min(1.0, std::fabs(v) / scale) : 0.0;
    const ImU32 base = !is_signed ? tok().ACCENT : v >= 0 ? tok().S_A : tok().LIFE_B;
    return with_alpha(base, static_cast<int>(20 + f * (max_alpha - 20)));
}

void heat_swatch(bool has, double v, double scale, bool is_signed, float width) {
    const float h = ImGui::GetTextLineHeight() * 0.7f;
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float y = p.y + (ImGui::GetTextLineHeight() - h) / 2;
    ImDrawList *dl = ImGui::GetWindowDrawList();
    if (has) dl->AddRectFilled(ImVec2(p.x, y), ImVec2(p.x + width, y + h), heat_color(v, scale, is_signed, 255), 2);
    dl->AddRect(ImVec2(p.x, y), ImVec2(p.x + width, y + h), tok().LINE, 2);
    ImGui::Dummy(ImVec2(width, ImGui::GetTextLineHeight()));
}

float draw_heat_legend(
    ImDrawList *dl, ImVec2 pos, float width, double scale, bool is_signed, const std::string &label) {
    const float lh = ImGui::GetTextLineHeight();
    dl->AddText(pos, tok().TEXT, label.c_str());
    const float y = pos.y + lh + 2, h = lh * 0.6f;
    const int steps = 24;
    for (int i = 0; i < steps; ++i) {
        const double t = is_signed ? -1.0 + 2.0 * (i + 0.5) / steps : (i + 0.5) / steps;
        const float x0 = pos.x + width * static_cast<float>(i) / static_cast<float>(steps),
                    x1 = pos.x + width * static_cast<float>(i + 1) / static_cast<float>(steps);
        dl->AddRectFilled(ImVec2(x0, y), ImVec2(x1, y + h), heat_color(t, 1.0, is_signed, 255));
    }
    dl->AddRect(ImVec2(pos.x, y), ImVec2(pos.x + width, y + h), tok().LINE);
    char lo[32], hi[32];
    std::snprintf(lo, sizeof lo, is_signed ? "%+.3f" : "%.3f", is_signed ? -scale : 0.0);
    std::snprintf(hi, sizeof hi, is_signed ? "%+.3f" : "%.3f", scale);
    dl->AddText(ImVec2(pos.x, y + h + 1), tok().DIM, lo);
    dl->AddText(ImVec2(pos.x + width - ImGui::CalcTextSize(hi).x, y + h + 1), tok().DIM, hi);
    return lh * 2 + h + 3;
}

void signed_value(double v, const std::string &text) {
    ImGui::PushStyleColor(ImGuiCol_Text, signed_color(v));
    ImGui::TextUnformatted(text.c_str());
    ImGui::PopStyleColor();
}

void card_swatch(uint8_t colors, bool land, float width) {
    const float h = ImGui::GetTextLineHeight();
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float sh = std::round(h * 0.8f);
    const ImVec2 a(p.x, p.y + (h - sh) / 2), b(p.x + width, p.y + (h - sh) / 2 + sh);
    draw_card_swatch(ImGui::GetWindowDrawList(), a, b, colors, land);
    ImGui::GetWindowDrawList()->AddRect(a, b, tok().LINE);
    ImGui::Dummy(ImVec2(width, h));
}

ImU32 linked_row_color() {
    return with_alpha(tok().LINK, 56);
}

void sparkline(const char *id, const std::vector<double> &values, ImVec2 size, ImU32 color) {
    ImGui::PushID(id);
    const ImVec2 a = ImGui::GetCursorScreenPos();
    if (size.x <= 0) size.x = std::max(40.0f, ImGui::GetContentRegionAvail().x);
    ImGui::Dummy(size);
    ImGui::PopID();
    const ImVec2 b(a.x + size.x, a.y + size.y);
    ImDrawList *dl = ImGui::GetWindowDrawList();
    dl->AddRectFilled(a, b, tok().PLOT_BG);
    dl->AddRect(a, b, tok().LINE);
    const float pad = 4;
    const auto y_of = [&](double v) {
        const double c = std::max(-1.0, std::min(1.0, v));
        return a.y + pad + static_cast<float>((1 - c) / 2) * (size.y - 2 * pad);
    };
    for (float x = a.x + 1; x < b.x - 1; x += 4) dl->AddLine(ImVec2(x, y_of(0)), ImVec2(x + 2, y_of(0)), tok().GRID);
    const size_t n = values.size();
    if (n == 0) return;
    const auto x_of = [&](size_t i) {
        return n == 1 ? (a.x + b.x) / 2
                      : a.x + pad + static_cast<float>(i) * (size.x - 2 * pad) / static_cast<float>(n - 1);
    };
    for (size_t i = 0; i < n; ++i) dl->PathLineTo(ImVec2(x_of(i), y_of(values[i])));
    dl->PathStroke(color, 0, 1.5f);
    const ImVec2 last(x_of(n - 1), y_of(values.back()));
    dl->AddRectFilled(ImVec2(last.x - 2.5f, last.y - 2.5f), ImVec2(last.x + 2.5f, last.y + 2.5f), color);
}

}  // namespace rmwork
