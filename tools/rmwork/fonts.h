#ifndef RMWORK_FONTS_H
#define RMWORK_FONTS_H

#include <string>

#include "imgui.h"

// ── The UI fonts: eight embedded families, each in a regular and a heavier weight ──────────────
//
// Every family × weight is added to the atlas when the context is created (ImGui 1.92 bakes glyphs
// on demand, at whatever size is in use, so an unused font costs nothing). The chosen one becomes
// io.FontDefault and its size style.FontSizeBase (apply_font, between frames); the size passed to
// AddFont only matters while FontSizeBase is unset. Families other than JetBrains Mono fall back to
// it for a glyph they lack.

namespace rmwork {

enum class FontWeight {
    REGULAR,
    HEAVY  // the family's heavier weight: Medium, SemiBold or Bold
};

struct FontFamily {
        const char *id;          // settings key ("jetbrains-mono")
        const char *name;        // display name ("JetBrains Mono")
        const char *heavy_name;  // the heavier weight's name ("Medium")
        bool proportional;
};

inline constexpr int FONT_FAMILY_COUNT = 8;
inline constexpr int FONT_SIZE_MIN = 11;
inline constexpr int FONT_SIZE_MAX = 20;

// The UI font a user chose (settings.json "font"); the default is JetBrains Mono Medium 14.
struct FontChoice {
        std::string family = "jetbrains-mono";
        FontWeight weight = FontWeight::HEAVY;
        int size = 14;
        bool operator==(const FontChoice &o) const {
            return family == o.family && weight == o.weight && size == o.size;
        }
        bool operator!=(const FontChoice &o) const { return !(*this == o); }
};

const FontFamily &font_family(int index);
// The family with settings key `id`, or -1.
int find_font_family(const std::string &id);
// "regular" or the heavier weight's name in lower case ("medium"), as settings.json spells it.
std::string font_weight_key(int family, FontWeight w);
// "Regular" or the heavier weight's name ("Medium").
const char *font_weight_name(int family, FontWeight w);
// The weight `key` names for `family` (font_weight_key's spelling); false for neither of its weights.
bool parse_font_weight(int family, const std::string &key, FontWeight *out);
bool font_size_valid(int size);
// `c` with an unknown family or an out-of-range size replaced by the default's.
FontChoice sanitize_font_choice(const FontChoice &c);
// "JetBrains Mono Medium 14".
std::string font_choice_label(const FontChoice &c);

// Adds every family × weight to the current context's atlas (the atlas does not own the data).
void load_ui_fonts();
// The loaded font of a family and weight.
ImFont *ui_font(int family, FontWeight w);
// Makes `c` (sanitized) the UI font from the next ImGui::NewFrame on; call between frames.
void apply_font(const FontChoice &c);

}  // namespace rmwork

#endif /* RMWORK_FONTS_H */
