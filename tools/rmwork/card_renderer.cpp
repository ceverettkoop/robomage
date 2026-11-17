#include "card_renderer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "fonts.h"
#include "imgui_internal.h"
#include "theme.h"

namespace {

// One drawing pass of a card: the draw list's state at its start, to rotate and fade what it adds.
struct CardPass {
        int vtx = 0;
        int cmd = 0;
        ImVec4 clip;
        bool rotated = false;
};

// An upright card's rectangle and the scale its parts are sized by.
struct Box {
        float x, y, w, h;
};

const ImVec4 UNCLIPPED(-16384.0f, -16384.0f, 16384.0f, 16384.0f);

// Card content colours (not theme): frames, pips, text.
const ImU32 BORDER = rmwork::hex_color(0x161412);
const ImU32 TOKEN_BORDER = rmwork::hex_color(0xb9bcc0);
const ImU32 INK = rmwork::hex_color(0x141414);
const ImU32 PARCHMENT = rmwork::hex_color(0xf1ebdc);
const ImU32 LAND_PARCHMENT = rmwork::hex_color(0xe4d3b5);
const ImU32 FRAME_W = rmwork::hex_color(0xd9d0ae);
const ImU32 FRAME_U = rmwork::hex_color(0x3f7fc0);
const ImU32 FRAME_B = rmwork::hex_color(0x5b524d);
const ImU32 FRAME_R = rmwork::hex_color(0xc8553b);
const ImU32 FRAME_G = rmwork::hex_color(0x3d8a50);
const ImU32 FRAME_GOLD = rmwork::hex_color(0xc9a54a);
const ImU32 FRAME_GREY = rmwork::hex_color(0xa3a8ad);
const ImU32 FRAME_LAND = rmwork::hex_color(0x9b7a52);
const ImU32 BACK_EDGE = rmwork::hex_color(0x1b120b);
const ImU32 BACK_FILL = rmwork::hex_color(0x5b3a1e);
const ImU32 BACK_INNER = rmwork::hex_color(0x3d2412);
const ImU32 BACK_OVAL = rmwork::hex_color(0x8a5a2b);
const ImU32 BACK_CORE = rmwork::hex_color(0x23304f);
const ImU32 BACK_TEXT = rmwork::hex_color(0xf0e6d2);

}  // namespace

static CardPass begin_pass(ImDrawList *dl, bool rotated);
static void end_pass(ImDrawList *dl, const CardPass &p, ImVec2 center, float alpha);
static ImU32 mix(ImU32 a, ImU32 b, float t);
static ImU32 color_frame(uint8_t bit);
static int color_count(uint8_t colors);
static ImFont *card_font(bool heavy);
static void draw_frame(ImDrawList *dl, const Box &b, const rmwork::CardLook &look, float inset, float rounding);
static ImU32 main_frame(const rmwork::CardLook &look);
static std::vector<std::string> mana_symbols(const std::string &cost);
static ImU32 pip_color(const std::string &sym);
static float pips_width(size_t count, float d);
static void draw_pips(ImDrawList *dl, const std::vector<std::string> &syms, ImVec2 right_center, float d);
static void draw_land_glyph(ImDrawList *dl, ImVec2 center, float size, ImU32 color);
static void draw_land_mana(ImDrawList *dl, uint8_t mana, ImVec2 center, float d);
static std::string type_text(const rmwork::CardLook &look);
static void draw_compact(ImDrawList *dl, const Box &b, const rmwork::CardLook &look);
static void draw_full(ImDrawList *dl, const Box &b, const rmwork::CardLook &look);
static void draw_back(ImDrawList *dl, const Box &b, const rmwork::CardLook &look);
static void draw_lines(ImDrawList *dl, ImFont *font, float size, const std::vector<std::string> &lines, float x,
    float y, float w, ImU32 color, bool centred);

// A rotated card is drawn upright first, unclipped (the font's line culling and the window's clip
// rectangle are in upright coordinates), then turned and given the clip rectangle back.
static CardPass begin_pass(ImDrawList *dl, bool rotated) {
    CardPass p;
    p.vtx = dl->VtxBuffer.Size;
    p.cmd = dl->CmdBuffer.Size;
    p.clip = dl->_ClipRectStack.back();
    p.rotated = rotated;
    if (rotated) dl->PushClipRect(ImVec2(UNCLIPPED.x, UNCLIPPED.y), ImVec2(UNCLIPPED.z, UNCLIPPED.w), false);
    return p;
}

static void end_pass(ImDrawList *dl, const CardPass &p, ImVec2 center, float alpha) {
    if (p.rotated) {
        dl->PopClipRect();
        for (int i = p.vtx; i < dl->VtxBuffer.Size; ++i) {
            ImVec2 &pos = dl->VtxBuffer[i].pos;
            const float dx = pos.x - center.x, dy = pos.y - center.y;
            pos = ImVec2(center.x - dy, center.y + dx);
        }
        for (int c = std::max(0, p.cmd - 1); c < dl->CmdBuffer.Size; ++c) {
            ImVec4 &r = dl->CmdBuffer[c].ClipRect;
            if (r.x == UNCLIPPED.x && r.y == UNCLIPPED.y && r.z == UNCLIPPED.z && r.w == UNCLIPPED.w) r = p.clip;
        }
    }
    if (alpha < 1.0f) {
        for (int i = p.vtx; i < dl->VtxBuffer.Size; ++i) {
            ImU32 &col = dl->VtxBuffer[i].col;
            const ImU32 a = static_cast<ImU32>(static_cast<float>((col >> IM_COL32_A_SHIFT) & 0xFF) * alpha);
            col = (col & ~IM_COL32_A_MASK) | (a << IM_COL32_A_SHIFT);
        }
    }
}

static ImU32 mix(ImU32 a, ImU32 b, float t) {
    const ImVec4 x = ImGui::ColorConvertU32ToFloat4(a), y = ImGui::ColorConvertU32ToFloat4(b);
    return ImGui::ColorConvertFloat4ToU32(
        ImVec4(x.x + (y.x - x.x) * t, x.y + (y.y - x.y) * t, x.z + (y.z - x.z) * t, x.w + (y.w - x.w) * t));
}

static ImU32 color_frame(uint8_t bit) {
    switch (bit) {
        case rmwork::COLOR_W:
            return FRAME_W;
        case rmwork::COLOR_U:
            return FRAME_U;
        case rmwork::COLOR_B:
            return FRAME_B;
        case rmwork::COLOR_R:
            return FRAME_R;
        default:
            return FRAME_G;
    }
}

static int color_count(uint8_t colors) {
    int n = 0;
    for (int i = 0; i < 5; ++i) n += (colors >> i) & 1;
    return n;
}

// Inter, the card text's face (its heavier weight for names and the type line).
static ImFont *card_font(bool heavy) {
    static const int inter = rmwork::find_font_family("inter");
    return rmwork::ui_font(inter < 0 ? 0 : inter, heavy ? rmwork::FontWeight::HEAVY : rmwork::FontWeight::REGULAR);
}

// The colour a part of the frame takes when it has one (a split frame's first colour).
static ImU32 main_frame(const rmwork::CardLook &look) {
    if (look.land && look.colors == 0) return FRAME_LAND;
    const int n = color_count(look.colors);
    if (n == 0) return FRAME_GREY;
    if (n >= 3) return FRAME_GOLD;
    for (int i = 0; i < 5; ++i)
        if (look.colors & (1 << i)) return color_frame(static_cast<uint8_t>(1 << i));
    return FRAME_GREY;
}

// The frame inside the border: one colour, two split down the middle, gold, grey or land tan.
static void draw_frame(ImDrawList *dl, const Box &b, const rmwork::CardLook &look, float inset, float rounding) {
    const ImVec2 a(b.x + inset, b.y + inset), z(b.x + b.w - inset, b.y + b.h - inset);
    if (color_count(look.colors) == 2) {
        uint8_t first = 0, second = 0;
        for (int i = 0; i < 5; ++i) {
            if (!(look.colors & (1 << i))) continue;
            if (!first) {
                first = static_cast<uint8_t>(1 << i);
            } else {
                second = static_cast<uint8_t>(1 << i);
            }
        }
        const float mid = (a.x + z.x) / 2;
        dl->AddRectFilled(a, ImVec2(mid, z.y), color_frame(first), rounding, ImDrawFlags_RoundCornersLeft);
        dl->AddRectFilled(ImVec2(mid, a.y), z, color_frame(second), rounding, ImDrawFlags_RoundCornersRight);
        return;
    }
    dl->AddRectFilled(a, z, main_frame(look), rounding);
}

// "{2}{U}{U}" -> {"2", "U", "U"}; "{W/U}" -> {"W/U"}.
static std::vector<std::string> mana_symbols(const std::string &cost) {
    std::vector<std::string> out;
    size_t i = 0;
    while ((i = cost.find('{', i)) != std::string::npos) {
        const size_t j = cost.find('}', i);
        if (j == std::string::npos) break;
        out.push_back(cost.substr(i + 1, j - i - 1));
        i = j + 1;
    }
    return out;
}

static ImU32 pip_color(const std::string &sym) {
    switch (sym.empty() ? '?' : sym[0]) {
        case 'W':
            return rmwork::hex_color(0xf8f4d8);
        case 'U':
            return rmwork::hex_color(0xa9d8f2);
        case 'B':
            return rmwork::hex_color(0xc9bfbb);
        case 'R':
            return rmwork::hex_color(0xf4a98c);
        case 'G':
            return rmwork::hex_color(0x9cd2ac);
        default:
            return rmwork::hex_color(0xd2cac7);
    }
}

static float pips_width(size_t count, float d) {
    return static_cast<float>(count) * d + static_cast<float>(count > 0 ? count - 1 : 0) * d * 0.12f;
}

// Mana pips, right-aligned to `right_center`, `d` across.
static void draw_pips(ImDrawList *dl, const std::vector<std::string> &syms, ImVec2 right_center, float d) {
    float x = right_center.x - pips_width(syms.size(), d);
    ImFont *font = card_font(true);
    for (const std::string &sym : syms) {
        const ImVec2 c(x + d / 2, right_center.y);
        const size_t slash = sym.find('/');
        if (slash != std::string::npos && slash + 1 < sym.size() && sym[slash + 1] != 'P') {
            dl->AddCircleFilled(c, d / 2, pip_color(sym.substr(slash + 1)));
            dl->PathArcTo(c, d / 2, IM_PI * 0.75f, IM_PI * 1.75f);
            dl->PathFillConvex(pip_color(sym));
        } else {
            dl->AddCircleFilled(c, d / 2, pip_color(sym));
        }
        dl->AddCircle(c, d / 2, rmwork::hex_color(0x000000, 140), 0, std::max(1.0f, d * 0.06f));
        std::string label;
        for (char ch : sym)
            if (ch != '/') label += ch;
        if (label.size() > 2) label = label.substr(0, 2);
        const float fs = d * (label.size() > 1 ? 0.55f : 0.78f);
        const ImVec2 ts = font->CalcTextSizeA(fs, FLT_MAX, 0, label.c_str());
        dl->AddText(font, fs, ImVec2(c.x - ts.x / 2, c.y - ts.y / 2), INK, label.c_str());
        x += d * 1.12f;
    }
}

// Two mountains: the land glyph.
static void draw_land_glyph(ImDrawList *dl, ImVec2 c, float s, ImU32 color) {
    dl->AddTriangleFilled(ImVec2(c.x - s * 0.55f, c.y + s * 0.35f), ImVec2(c.x - s * 0.1f, c.y - s * 0.4f),
        ImVec2(c.x + s * 0.3f, c.y + s * 0.35f), color);
    dl->AddTriangleFilled(ImVec2(c.x - s * 0.05f, c.y + s * 0.35f), ImVec2(c.x + s * 0.3f, c.y - s * 0.15f),
        ImVec2(c.x + s * 0.6f, c.y + s * 0.35f), mix(color, rmwork::hex_color(0x000000), 0.2f));
}

// A land's basic types as pips, centred on `c`.
static void draw_land_mana(ImDrawList *dl, uint8_t mana, ImVec2 c, float d) {
    static const char *const SYMS[] = {"W", "U", "B", "R", "G"};
    std::vector<std::string> syms;
    for (int i = 0; i < 5; ++i)
        if (mana & (1 << i)) syms.push_back(SYMS[i]);
    if (syms.empty()) return;
    draw_pips(dl, syms, ImVec2(c.x + pips_width(syms.size(), d) / 2, c.y), d);
}

static std::string type_text(const rmwork::CardLook &look) {
    const std::string &t = look.face->type_line;
    return look.token ? (t.compare(0, 6, "Token ") == 0 ? t : "Token " + t) : t;
}

static void draw_lines(ImDrawList *dl, ImFont *font, float size, const std::vector<std::string> &lines, float x,
    float y, float w, ImU32 color, bool centred) {
    const float lh = size * 1.18f;
    for (const std::string &l : lines) {
        float lx = x;
        if (centred) lx = x + (w - font->CalcTextSizeA(size, FLT_MAX, 0, l.c_str()).x) / 2;
        dl->AddText(font, size, ImVec2(lx, y), color, l.c_str());
        y += lh;
    }
}

// A table-sized card: the name wrapped in a pale panel with the cost under it, then the type line
// (and a land's glyph and pips) on parchment.
static void draw_compact(ImDrawList *dl, const Box &b, const rmwork::CardLook &look) {
    const float border = std::max(1.0f, b.w * 0.035f);
    const float inner = border + std::max(1.0f, b.w * 0.035f);
    const ImU32 frame = main_frame(look);
    const float fs = std::max(5.0f, std::min(15.0f, b.w * 0.125f));
    ImFont *heavy = card_font(true), *regular = card_font(false);
    const float x = b.x + inner, w = b.w - 2 * inner;
    const float pad = std::max(1.0f, b.w * 0.025f);
    const std::vector<std::string> name = rmwork::wrap_text(heavy, fs, look.face->name, w - 2 * pad, 3);
    const std::vector<std::string> cost = mana_symbols(look.face->mana_cost);
    const float d = std::max(4.0f, std::min(13.0f, b.w * 0.115f));
    const bool show_cost = !cost.empty() && b.w >= 36;
    const float name_h = static_cast<float>(name.size()) * fs * 1.18f + (show_cost ? d + pad : 0) + 2 * pad;
    const float y0 = b.y + inner;
    dl->AddRectFilled(
        ImVec2(x, y0), ImVec2(x + w, y0 + name_h), mix(frame, rmwork::hex_color(0xffffff), 0.74f), b.w * 0.025f);
    draw_lines(dl, heavy, fs, name, x + pad, y0 + pad, w - 2 * pad, INK, false);
    if (show_cost) {
        const float avail = w - 2 * pad;
        float dd = d;
        if (pips_width(cost.size(), dd) > avail) dd = avail / (static_cast<float>(cost.size()) * 1.12f);
        draw_pips(dl, cost, ImVec2(x + w - pad, y0 + name_h - pad - d / 2), dd);
    }
    const float ty0 = y0 + name_h + inner * 0.6f, ty1 = b.y + b.h - inner;
    if (ty1 - ty0 < fs) return;
    dl->AddRectFilled(ImVec2(x, ty0), ImVec2(x + w, ty1), look.land ? LAND_PARCHMENT : PARCHMENT, b.w * 0.02f);
    const float ts = std::min(12.0f, b.w * 0.095f);
    float ty = ty0 + pad;
    if (look.land) {
        const float g = std::min(w * 0.45f, (ty1 - ty0) * 0.4f);
        draw_land_glyph(
            dl, ImVec2(x + w / 2, ty0 + pad + g * 0.5f), g, mix(FRAME_LAND, rmwork::hex_color(0x000000), 0.25f));
        ty += g + pad;
        if (look.land_mana) {
            draw_land_mana(dl, look.land_mana, ImVec2(x + w / 2, ty + d / 2), d);
            ty += d + pad;
        }
    }
    // Below 6.5 px the type line is not legible: the card shows its name alone.
    const int lines = ts < 6.5f ? 0 : std::max(0, std::min(3, static_cast<int>((ty1 - ty - pad) / (ts * 1.18f))));
    if (lines > 0)
        draw_lines(dl, regular, ts, rmwork::wrap_text(regular, ts, type_text(look), w - 2 * pad, lines), x + pad, ty,
            w - 2 * pad, INK, true);
}

// A large card: a name bar with the cost, an art box (from 170 px: the frame's colours, a land's
// glyph and pips), the type bar and the oracle text.
static void draw_full(ImDrawList *dl, const Box &b, const rmwork::CardLook &look) {
    const float border = std::max(2.0f, b.w * 0.035f);
    const float inner = border + b.w * 0.03f;
    const ImU32 frame = main_frame(look);
    const ImU32 bar = mix(frame, rmwork::hex_color(0xffffff), 0.72f);
    const ImU32 bar_edge = mix(frame, rmwork::hex_color(0x000000), 0.35f);
    ImFont *heavy = card_font(true), *regular = card_font(false);
    const float fs = std::max(8.0f, std::min(17.0f, b.w * 0.075f));
    const float x = b.x + inner, w = b.w - 2 * inner;
    float y = b.y + inner;
    const float nb = fs * 1.7f;
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + w, y + nb), bar, nb * 0.25f);
    dl->AddRect(ImVec2(x, y), ImVec2(x + w, y + nb), bar_edge, nb * 0.25f);
    const std::vector<std::string> cost = mana_symbols(look.face->mana_cost);
    const float d = fs * 1.05f;
    const float cost_w = cost.empty() ? 0 : pips_width(cost.size(), d) + fs * 0.4f;
    rmwork::draw_fitted_text(
        dl, heavy, fs, ImVec2(x + fs * 0.45f, y + (nb - fs) / 2 - 1), INK, look.face->name, w - fs * 0.9f - cost_w);
    if (!cost.empty()) draw_pips(dl, cost, ImVec2(x + w - fs * 0.35f, y + nb / 2), d);
    y += nb + inner * 0.4f;
    const bool art = b.w >= 170;
    if (art) {
        const float ah = b.h * 0.24f;
        const ImU32 light = mix(frame, rmwork::hex_color(0xffffff), 0.35f),
                    dark = mix(frame, rmwork::hex_color(0x000000), 0.3f);
        dl->AddRectFilledMultiColor(ImVec2(x + w * 0.02f, y), ImVec2(x + w * 0.98f, y + ah), light, light, dark, dark);
        dl->AddRect(ImVec2(x + w * 0.02f, y), ImVec2(x + w * 0.98f, y + ah), bar_edge);
        if (look.land) {
            draw_land_glyph(
                dl, ImVec2(x + w / 2, y + ah * 0.45f), ah * 0.55f, mix(FRAME_LAND, rmwork::hex_color(0x000000), 0.35f));
            draw_land_mana(dl, look.land_mana, ImVec2(x + w / 2, y + ah * 0.82f), d);
        }
        y += ah + inner * 0.4f;
    }
    const float tb = fs * 1.55f;
    dl->AddRectFilled(ImVec2(x, y), ImVec2(x + w, y + tb), bar, tb * 0.25f);
    dl->AddRect(ImVec2(x, y), ImVec2(x + w, y + tb), bar_edge, tb * 0.25f);
    const float ts = fs * 0.86f;
    rmwork::draw_fitted_text(
        dl, heavy, ts, ImVec2(x + fs * 0.45f, y + (tb - ts) / 2 - 1), INK, type_text(look), w - fs * 0.9f);
    if (!art && look.land && look.land_mana)
        draw_land_mana(dl, look.land_mana, ImVec2(x + w - d * 1.5f, y + tb / 2), d * 0.8f);
    y += tb + inner * 0.4f;
    const float bottom = b.y + b.h - inner - b.h * 0.07f;
    if (bottom - y < fs) return;
    dl->AddRectFilled(ImVec2(x + w * 0.02f, y), ImVec2(x + w * 0.98f, b.y + b.h - inner),
        look.land ? LAND_PARCHMENT : PARCHMENT, fs * 0.2f);
    const float os = std::max(7.0f, std::min(14.0f, b.w * 0.052f));
    const float lh = os * 1.18f;
    const int max_lines = static_cast<int>((bottom - y - os * 0.4f) / lh);
    if (max_lines <= 0) return;
    // Paragraphs keep their breaks; the whole is cut at the box's last line.
    std::vector<std::string> lines;
    const std::string &text = look.face->oracle;
    size_t start = 0;
    while (start <= text.size() && static_cast<int>(lines.size()) < max_lines) {
        size_t nl = text.find('\n', start);
        if (nl == std::string::npos) nl = text.size();
        const std::string para = text.substr(start, nl - start);
        const int left = max_lines - static_cast<int>(lines.size());
        std::vector<std::string> wrapped = rmwork::wrap_text(regular, os, para, w * 0.96f - os, left);
        const bool more = nl < text.size();
        if (more && static_cast<int>(wrapped.size()) == left && !wrapped.empty() &&
            wrapped.back().find("\xE2\x80\xA6") == std::string::npos)
            wrapped.back() += "\xE2\x80\xA6";
        for (std::string &l : wrapped) lines.push_back(std::move(l));
        start = nl + 1;
    }
    draw_lines(dl, regular, os, lines, x + w * 0.02f + os * 0.5f, y + os * 0.35f, w, INK, false);
}

// The back: brown, an inner panel and an oval; the note (a face-down card's known identity) on it.
static void draw_back(ImDrawList *dl, const Box &b, const rmwork::CardLook &look) {
    const float r = b.w * 0.045f;
    dl->AddRectFilled(ImVec2(b.x, b.y), ImVec2(b.x + b.w, b.y + b.h), BACK_EDGE, r);
    const float i1 = std::max(1.0f, b.w * 0.04f);
    dl->AddRectFilled(ImVec2(b.x + i1, b.y + i1), ImVec2(b.x + b.w - i1, b.y + b.h - i1), BACK_FILL, r * 0.7f);
    const float i2 = b.w * 0.11f;
    dl->AddRectFilled(ImVec2(b.x + i2, b.y + i2), ImVec2(b.x + b.w - i2, b.y + b.h - i2), BACK_INNER, r * 0.5f);
    const ImVec2 c(b.x + b.w / 2, b.y + b.h / 2);
    dl->AddEllipseFilled(c, ImVec2(b.w * 0.30f, b.h * 0.34f), BACK_OVAL);
    dl->AddEllipseFilled(c, ImVec2(b.w * 0.24f, b.h * 0.28f), BACK_CORE);
    if (look.note.empty() || b.w < 56) return;
    ImFont *font = card_font(true);
    const float fs = std::max(7.0f, std::min(14.0f, b.w * 0.09f));
    const std::vector<std::string> lines = rmwork::wrap_text(font, fs, look.note, b.w - 2 * i2, 3);
    const float th = static_cast<float>(lines.size()) * fs * 1.18f;
    const float ty = b.y + b.h - i2 - th - fs * 0.3f;
    dl->AddRectFilled(ImVec2(b.x + i2 * 0.6f, ty - fs * 0.2f), ImVec2(b.x + b.w - i2 * 0.6f, ty + th + fs * 0.1f),
        rmwork::hex_color(0x000000, 170), fs * 0.2f);
    draw_lines(dl, font, fs, lines, b.x + i2, ty, b.w - 2 * i2, BACK_TEXT, true);
}

namespace rmwork {

uint8_t basic_land_mana(const std::vector<std::string> &subtypes) {
    uint8_t out = 0;
    for (const std::string &t : subtypes) {
        if (t == "Plains") out |= COLOR_W;
        if (t == "Island") out |= COLOR_U;
        if (t == "Swamp") out |= COLOR_B;
        if (t == "Mountain") out |= COLOR_R;
        if (t == "Forest") out |= COLOR_G;
    }
    return out;
}

void draw_card_swatch(ImDrawList *dl, ImVec2 min, ImVec2 max, uint8_t colors, bool land) {
    CardLook look;
    look.colors = colors;
    look.land = land;
    draw_frame(dl, Box{min.x, min.y, max.x - min.x, max.y - min.y}, look, 0, 0);
}

float draw_fitted_text(
    ImDrawList *dl, ImFont *font, float size, ImVec2 pos, ImU32 color, const std::string &text, float max_w) {
    if (max_w <= 0) return 0;
    const float full = font->CalcTextSizeA(size, FLT_MAX, 0, text.c_str()).x;
    if (full <= max_w) {
        dl->AddText(font, size, pos, color, text.c_str());
        return full;
    }
    static const char *const ELLIPSIS = "\xE2\x80\xA6";
    const float ew = font->CalcTextSizeA(size, FLT_MAX, 0, ELLIPSIS).x;
    std::string cut = text;
    while (!cut.empty()) {
        // Drop one UTF-8 character.
        size_t n = cut.size() - 1;
        while (n > 0 && (static_cast<unsigned char>(cut[n]) & 0xC0) == 0x80) --n;
        cut.resize(n);
        if (font->CalcTextSizeA(size, FLT_MAX, 0, cut.c_str()).x + ew <= max_w) break;
    }
    cut += ELLIPSIS;
    dl->AddText(font, size, pos, color, cut.c_str());
    return font->CalcTextSizeA(size, FLT_MAX, 0, cut.c_str()).x;
}

std::vector<std::string> wrap_text(ImFont *font, float size, const std::string &text, float max_w, int max_lines) {
    std::vector<std::string> lines;
    if (max_lines <= 0) return lines;
    std::vector<std::string> words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && (text[i] == ' ' || text[i] == '\n')) ++i;
        size_t j = i;
        while (j < text.size() && text[j] != ' ' && text[j] != '\n') ++j;
        if (j > i) words.push_back(text.substr(i, j - i));
        i = j;
    }
    const auto width = [&](const std::string &s) { return font->CalcTextSizeA(size, FLT_MAX, 0, s.c_str()).x; };
    std::string cur;
    size_t w = 0;
    for (; w < words.size(); ++w) {
        const std::string next = cur.empty() ? words[w] : cur + " " + words[w];
        if (cur.empty() || width(next) <= max_w) {
            cur = next;
            continue;
        }
        lines.push_back(cur);
        cur = words[w];
        if (static_cast<int>(lines.size()) == max_lines) break;
    }
    const bool more = w < words.size();
    if (!more && !cur.empty()) lines.push_back(cur);
    if (more || (!lines.empty() && width(lines.back()) > max_w)) {
        std::string &last = lines.back();
        static const char *const ELLIPSIS = "\xE2\x80\xA6";
        if (more) last += " ";
        while (!last.empty() && width(last + ELLIPSIS) > max_w) {
            size_t n = last.size() - 1;
            while (n > 0 && (static_cast<unsigned char>(last[n]) & 0xC0) == 0x80) --n;
            last.resize(n);
        }
        while (!last.empty() && last.back() == ' ') last.pop_back();
        last += ELLIPSIS;
    }
    return lines;
}

void TextCardRenderer::draw(
    ImDrawList *dl, const CardLook &look, ImVec2 slot_min, ImVec2 slot_max, bool rotated, float alpha) const {
    const float sw = slot_max.x - slot_min.x, sh = slot_max.y - slot_min.y;
    const float w = rotated ? sh : sw, h = rotated ? sw : sh;
    const ImVec2 center((slot_min.x + slot_max.x) / 2, (slot_min.y + slot_max.y) / 2);
    const Box b{center.x - w / 2, center.y - h / 2, w, h};
    const CardPass pass = begin_pass(dl, rotated);
    if (!look.face) {
        draw_back(dl, b, look);
    } else {
        const float r = w * 0.045f;
        dl->AddRectFilled(ImVec2(b.x, b.y), ImVec2(b.x + w, b.y + h), look.token ? TOKEN_BORDER : BORDER, r);
        const float inset = std::max(1.0f, w * 0.035f);
        draw_frame(dl, b, look, inset, r * 0.6f);
        if (w >= 120) {
            draw_full(dl, b, look);
        } else {
            draw_compact(dl, b, look);
        }
    }
    end_pass(dl, pass, center, alpha);
}

}  // namespace rmwork
