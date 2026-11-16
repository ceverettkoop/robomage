#include "fonts.h"

#include <cctype>

#include "imgui_internal.h"

// The fonts (third_party/fonts/<family>/), compiled in: the assembler's .incbin copies each file's
// bytes into read-only data between two symbols, <sym> and <sym>_end (the Makefile passes the
// directory as RM_FONT_DIR and makes this object depend on every file).
#ifndef RM_FONT_DIR
#define RM_FONT_DIR "third_party/fonts"
#endif
#ifdef __APPLE__
#define RM_SYM_PREFIX "_"
#define RM_RODATA ".const\n"
#define RM_RODATA_END ".text\n"
#else
#define RM_SYM_PREFIX ""
#define RM_RODATA ".pushsection .rodata\n"
#define RM_RODATA_END ".popsection\n"
#endif
// clang-format off
#define RM_EMBED_FONT(sym, file)                                                                     \
    __asm__(RM_RODATA ".globl " RM_SYM_PREFIX #sym "\n"                                              \
            ".globl " RM_SYM_PREFIX #sym "_end\n"                                                    \
            ".p2align 4\n" RM_SYM_PREFIX #sym ":\n"                                                  \
            ".incbin \"" RM_FONT_DIR "/" file "\"\n" RM_SYM_PREFIX #sym "_end:\n" RM_RODATA_END);    \
    extern "C" const unsigned char sym[];                                                            \
    extern "C" const unsigned char sym##_end[]

RM_EMBED_FONT(rm_font_jetbrains_regular, "jetbrains-mono/JetBrainsMono-Regular.ttf");
RM_EMBED_FONT(rm_font_jetbrains_heavy, "jetbrains-mono/JetBrainsMono-Medium.ttf");
RM_EMBED_FONT(rm_font_fira_regular, "fira-code/FiraCode-Regular.ttf");
RM_EMBED_FONT(rm_font_fira_heavy, "fira-code/FiraCode-Medium.ttf");
RM_EMBED_FONT(rm_font_plex_mono_regular, "ibm-plex-mono/IBMPlexMono-Regular.ttf");
RM_EMBED_FONT(rm_font_plex_mono_heavy, "ibm-plex-mono/IBMPlexMono-Medium.ttf");
RM_EMBED_FONT(rm_font_source_regular, "source-code-pro/SourceCodePro-Regular.ttf");
RM_EMBED_FONT(rm_font_source_heavy, "source-code-pro/SourceCodePro-Medium.ttf");
RM_EMBED_FONT(rm_font_cascadia_regular, "cascadia-mono/CascadiaMono-Regular.ttf");
RM_EMBED_FONT(rm_font_cascadia_heavy, "cascadia-mono/CascadiaMono-SemiBold.ttf");
RM_EMBED_FONT(rm_font_hack_regular, "hack/Hack-Regular.ttf");
RM_EMBED_FONT(rm_font_hack_heavy, "hack/Hack-Bold.ttf");
RM_EMBED_FONT(rm_font_inter_regular, "inter/Inter-Regular.ttf");
RM_EMBED_FONT(rm_font_inter_heavy, "inter/Inter-Medium.ttf");
RM_EMBED_FONT(rm_font_plex_sans_regular, "ibm-plex-sans/IBMPlexSans-Regular.ttf");
RM_EMBED_FONT(rm_font_plex_sans_heavy, "ibm-plex-sans/IBMPlexSans-Medium.ttf");
// clang-format on

namespace {

// One embedded file.
struct Blob {
        const unsigned char *begin;
        const unsigned char *end;
};

struct FamilyData {
        rmwork::FontFamily info;
        Blob regular;
        Blob heavy;
};

const FamilyData FAMILIES[rmwork::FONT_FAMILY_COUNT] = {
    {{"jetbrains-mono", "JetBrains Mono", "Medium", false}, {rm_font_jetbrains_regular, rm_font_jetbrains_regular_end},
        {rm_font_jetbrains_heavy, rm_font_jetbrains_heavy_end}},
    {{"fira-code", "Fira Code", "Medium", false}, {rm_font_fira_regular, rm_font_fira_regular_end},
        {rm_font_fira_heavy, rm_font_fira_heavy_end}},
    {{"ibm-plex-mono", "IBM Plex Mono", "Medium", false}, {rm_font_plex_mono_regular, rm_font_plex_mono_regular_end},
        {rm_font_plex_mono_heavy, rm_font_plex_mono_heavy_end}},
    {{"source-code-pro", "Source Code Pro", "Medium", false}, {rm_font_source_regular, rm_font_source_regular_end},
        {rm_font_source_heavy, rm_font_source_heavy_end}},
    {{"cascadia-mono", "Cascadia Mono", "SemiBold", false}, {rm_font_cascadia_regular, rm_font_cascadia_regular_end},
        {rm_font_cascadia_heavy, rm_font_cascadia_heavy_end}},
    {{"hack", "Hack", "Bold", false}, {rm_font_hack_regular, rm_font_hack_regular_end},
        {rm_font_hack_heavy, rm_font_hack_heavy_end}},
    {{"inter", "Inter", "Medium", true}, {rm_font_inter_regular, rm_font_inter_regular_end},
        {rm_font_inter_heavy, rm_font_inter_heavy_end}},
    {{"ibm-plex-sans", "IBM Plex Sans", "Medium", true}, {rm_font_plex_sans_regular, rm_font_plex_sans_regular_end},
        {rm_font_plex_sans_heavy, rm_font_plex_sans_heavy_end}},
};

}  // namespace

static std::string lower(const char *s);
static ImFont *add_blob(const Blob &b, const char *name, bool merge);

static std::string lower(const char *s) {
    std::string out = s;
    for (char &c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// Adds an embedded file to the atlas as a new font, or merged into the last one (`merge`: glyphs
// the last font lacks). The size given is only the font's LegacySize: apply_font sets the size.
static ImFont *add_blob(const Blob &b, const char *name, bool merge) {
    ImFontConfig cfg;
    cfg.FontDataOwnedByAtlas = false;
    cfg.MergeMode = merge;
    ImFormatString(cfg.Name, sizeof cfg.Name, "%s", name);
    // ImGui takes a mutable pointer but only reads the data when the atlas does not own it.
    void *data = const_cast<unsigned char *>(b.begin);
    return ImGui::GetIO().Fonts->AddFontFromMemoryTTF(
        data, static_cast<int>(b.end - b.begin), static_cast<float>(rmwork::FontChoice().size), &cfg);
}

namespace rmwork {

const FontFamily &font_family(int index) {
    return FAMILIES[index].info;
}

int find_font_family(const std::string &id) {
    for (int i = 0; i < FONT_FAMILY_COUNT; ++i)
        if (id == FAMILIES[i].info.id) return i;
    return -1;
}

std::string font_weight_key(int family, FontWeight w) {
    return lower(font_weight_name(family, w));
}

const char *font_weight_name(int family, FontWeight w) {
    return w == FontWeight::REGULAR ? "Regular" : FAMILIES[family].info.heavy_name;
}

bool parse_font_weight(int family, const std::string &key, FontWeight *out) {
    for (FontWeight w : {FontWeight::REGULAR, FontWeight::HEAVY}) {
        if (key == font_weight_key(family, w)) {
            *out = w;
            return true;
        }
    }
    return false;
}

bool font_size_valid(int size) {
    return size >= FONT_SIZE_MIN && size <= FONT_SIZE_MAX;
}

FontChoice sanitize_font_choice(const FontChoice &c) {
    FontChoice out = c;
    const FontChoice def;
    if (find_font_family(c.family) < 0) out.family = def.family;
    if (!font_size_valid(c.size)) out.size = def.size;
    return out;
}

std::string font_choice_label(const FontChoice &c) {
    const FontChoice s = sanitize_font_choice(c);
    const int f = find_font_family(s.family);
    return std::string(FAMILIES[f].info.name) + " " + font_weight_name(f, s.weight) + " " + std::to_string(s.size);
}

void load_ui_fonts() {
    const FamilyData &fallback = FAMILIES[0];
    for (int i = 0; i < FONT_FAMILY_COUNT; ++i) {
        for (FontWeight w : {FontWeight::REGULAR, FontWeight::HEAVY}) {
            const bool heavy = w == FontWeight::HEAVY;
            const FamilyData &f = FAMILIES[i];
            add_blob(heavy ? f.heavy : f.regular, f.info.name, false);
            if (i != 0) add_blob(heavy ? fallback.heavy : fallback.regular, fallback.info.name, true);
        }
    }
}

ImFont *ui_font(int family, FontWeight w) {
    return ImGui::GetIO().Fonts->Fonts[family * 2 + (w == FontWeight::HEAVY ? 1 : 0)];
}

// NewFrame takes the size from style.FontSizeBase (and _NextFrameFontSizeBase, which survives a
// call made inside a frame, where ImGui rewrites FontSizeBase whenever the font stack changes).
void apply_font(const FontChoice &c) {
    const FontChoice s = sanitize_font_choice(c);
    ImGuiStyle &style = ImGui::GetStyle();
    ImGui::GetIO().FontDefault = ui_font(find_font_family(s.family), s.weight);
    style.FontSizeBase = static_cast<float>(s.size);
    style._NextFrameFontSizeBase = style.FontSizeBase;
}

}  // namespace rmwork
