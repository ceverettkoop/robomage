#include "dialogs.h"

#include <algorithm>
#include <cfloat>
#include <cstring>

#include "theme.h"
#include "widgets.h"

namespace {

const char *const SETTINGS_POPUP = "Settings###rmwork.settings";
const char *const TAB_NAMES[] = {"Table", "Card Printings", "Keys", "Appearance"};
const char *const TAB_KEYS[] = {"table", "printings", "keys", "appearance"};
// The Appearance tab's preview lines.
const char *const FONT_PREVIEW[] = {"Player A casts Brainstorm. Turn 4 · Decision 212 → 213 · V −0.42",
    "{1}{U}{U}  3/3  0123456789  ( ) [ ] { } < > = + − * / % & | ! ? @ # …",
    "The quick brown fox jumps over the lazy dog. Il1| O0 rn m"};
const char *const ERROR_POPUP_ID = "###rmwork.error";

// The mock-up's colour swatches for the table.
const ImU32 TABLE_COLORS[] = {rmwork::hex_color(0x6e4526), rmwork::hex_color(0x0b6b2a), rmwork::hex_color(0x008080),
    rmwork::hex_color(0x1d2836), rmwork::hex_color(0x3a3f47), rmwork::hex_color(0x101215)};

}  // namespace

static float em_scale();
static void begin_modal_frame(ImVec2 size);
static bool close_requested();
static float footer_height();
static bool right_aligned_button(const char *label);
static void note(const char *text);
static void field_label(const char *label, float field_x);
static float char_width();

// The current font size over the default's, for sizes laid out at the default.
static float em_scale() {
    return ImGui::GetFontSize() / static_cast<float>(rmwork::FontChoice().size);
}

// Centres the next modal, `size` (at the default font size) scaled to the font and kept inside the
// viewport, and gives it the theme's dialog title bar (pop with ImGui::PopStyleColor(3) right after
// the Begin, which draws the title).
static void begin_modal_frame(ImVec2 size) {
    const ImGuiViewport *vp = ImGui::GetMainViewport();
    const float k = em_scale();
    const ImVec2 scaled(std::min(size.x * k, vp->WorkSize.x - 16), std::min(size.y * k, vp->WorkSize.y - 16));
    ImGui::SetNextWindowPos(vp->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(scaled, ImGuiCond_Appearing);
    ImGui::PushStyleColor(ImGuiCol_TitleBg, rmwork::tok().TITLE_ACTIVE);
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, rmwork::tok().TITLE_ACTIVE);
    ImGui::PushStyleColor(ImGuiCol_Text, rmwork::tok().TITLE_TEXT);
}

// Esc closes the current modal.
static bool close_requested() {
    return ImGui::IsKeyPressed(ImGuiKey_Escape, false);
}

static float footer_height() {
    return ImGui::GetFrameHeightWithSpacing() + ImGui::GetStyle().ItemSpacing.y + 1;
}

// A button ending at the right edge of the content region, from the cursor's line.
static bool right_aligned_button(const char *label) {
    const float w = ImGui::CalcTextSize(label).x + ImGui::GetStyle().FramePadding.x * 2 + 12;
    const float avail = ImGui::GetContentRegionAvail().x;
    if (avail > w) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - w);
    return ImGui::Button(label, ImVec2(w, 0));
}

static void note(const char *text) {
    ImGui::PushStyleColor(ImGuiCol_Text, rmwork::tok().DIM);
    ImGui::TextWrapped("%s", text);
    ImGui::PopStyleColor();
}

// A field's label on the frame's text line, the field starting `field_x` from the line's start.
static void field_label(const char *label, float field_x) {
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::SameLine(field_x);
}

// The width of a character of the current font ("M").
static float char_width() {
    return ImGui::CalcTextSize("M").x;
}

namespace rmwork {

int SettingsDialog::find_tab(const std::string &key) {
    for (int i = 0; i < TAB_COUNT; ++i)
        if (key == TAB_KEYS[i]) return i;
    return -1;
}

void SettingsDialog::open(Tab tab) {
    want_open_ = true;
    tab_ = tab;
    select_tab_ = true;
}

void SettingsDialog::sync_from(const Settings &s) {
    switch (s.table.kind) {
        case TableBackground::Kind::COLOR:
            color_ = s.table.value;
            break;
        case TableBackground::Kind::TEXTURE:
            texture_ = s.table.value;
            break;
        case TableBackground::Kind::IMAGE:
            std::snprintf(image_, sizeof image_, "%s", s.table.value.c_str());
            break;
    }
}

void SettingsDialog::draw(Settings *s, const std::string &settings_path) {
    if (want_open_) {
        sync_from(*s);
        ImGui::OpenPopup(SETTINGS_POPUP);
        want_open_ = false;
    }
    begin_modal_frame(ImVec2(760, 470));
    bool keep = true;
    const bool visible = ImGui::BeginPopupModal(SETTINGS_POPUP, &keep, ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleColor(3);
    open_ = visible;
    if (!visible) return;
    ImGui::BeginChild("##settings_body", ImVec2(0, -footer_height()));
    if (ImGui::BeginTabBar("##settings_tabs")) {
        for (int i = 0; i < TAB_COUNT; ++i) {
            const ImGuiTabItemFlags f = select_tab_ && i == tab_ ? ImGuiTabItemFlags_SetSelected : 0;
            if (!ImGui::BeginTabItem(TAB_NAMES[i], nullptr, f)) continue;
            ImGui::Spacing();
            if (i == TABLE) draw_table_tab(s, settings_path);
            if (i == PRINTINGS) draw_printings_tab();
            if (i == KEYS) draw_keys_tab(*s);
            if (i == APPEARANCE) draw_appearance_tab(s, settings_path);
            ImGui::EndTabItem();
        }
        select_tab_ = false;
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
    ImGui::Separator();
    if (right_aligned_button("Close") || close_requested() || !keep) {
        ImGui::CloseCurrentPopup();
        open_ = false;
    }
    ImGui::EndPopup();
}

void SettingsDialog::draw_table_tab(Settings *s, const std::string &settings_path) {
    using Kind = TableBackground::Kind;
    TableBackground &bg = s->table;
    ImGui::TextDisabled("Table background");
    ImGui::Spacing();
    if (ImGui::RadioButton("Colour", bg.kind == Kind::COLOR)) bg = {Kind::COLOR, color_};
    ImGui::Indent(22);
    for (size_t i = 0; i < sizeof TABLE_COLORS / sizeof TABLE_COLORS[0]; ++i) {
        const std::string hex = format_hex_color(TABLE_COLORS[i]);
        ImGui::PushID(static_cast<int>(i));
        if (color_swatch("##color", TABLE_COLORS[i], bg.kind == Kind::COLOR && bg.value == hex)) {
            color_ = hex;
            bg = {Kind::COLOR, color_};
        }
        ImGui::PopID();
        ImGui::SameLine(0, 8);
    }
    ImU32 other = hex_color(0x6e4526);
    parse_hex_color(color_, &other);
    ImVec4 rgb = ImGui::ColorConvertU32ToFloat4(other);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Other");
    ImGui::SameLine();
    if (ImGui::ColorEdit3("##other_color", &rgb.x, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel)) {
        color_ = format_hex_color(ImGui::ColorConvertFloat4ToU32(rgb));
        bg = {Kind::COLOR, color_};
    }
    ImGui::Unindent(22);
    ImGui::Spacing();

    if (ImGui::RadioButton("Texture", bg.kind == Kind::TEXTURE)) bg = {Kind::TEXTURE, texture_};
    ImGui::Indent(22);
    for (int i = 0; i < TEXTURE_COUNT; ++i) {
        const TextureInfo &t = TABLE_TEXTURES[i];
        ImGui::BeginGroup();
        ImGui::PushID(i);
        if (texture_swatch("##texture", t.key, bg.kind == Kind::TEXTURE && bg.value == t.key)) {
            texture_ = t.key;
            bg = {Kind::TEXTURE, texture_};
        }
        ImGui::PopID();
        const float label_w = ImGui::CalcTextSize(t.name).x;
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + (56 - label_w) / 2);
        ImGui::TextDisabled("%s", t.name);
        ImGui::EndGroup();
        ImGui::SameLine(0, 10);
    }
    ImGui::NewLine();
    ImGui::Unindent(22);
    ImGui::Spacing();

    if (ImGui::RadioButton("Image", bg.kind == Kind::IMAGE)) bg = {Kind::IMAGE, image_};
    ImGui::Indent(22);
    ImGui::SetNextItemWidth(-1);
    if (ImGui::InputTextWithHint("##image", "path to an image file", image_, sizeof image_)) bg = {Kind::IMAGE, image_};
    ImGui::Unindent(22);
    ImGui::Spacing();
    ImGui::Separator();
    note(("Saved to " + settings_path +
          ". Nothing draws the table yet: the Table tool renders the background when it arrives (plan §2.5).")
            .c_str());
}

void SettingsDialog::draw_printings_tab() {
    note(
        "Each card will be drawn with its oldest paper printing unless you choose another here. This tab arrives "
        "with card images (docs/workbench_plan.md §2.4): the choices are kept in ~/.config/robomage/printings.json "
        "and the images cached under ~/.cache/robomage/scryfall/.");
}

void SettingsDialog::draw_keys_tab(const Settings &s) {
    note(
        "Rebinding from here comes later; until then edit \"keys\" in settings.json (e.g. \"replay.next\": "
        "\"RightArrow\") and restart.");
    ImGui::Spacing();
    const ImGuiTableFlags flags =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_ScrollY;
    if (!ImGui::BeginTable("##keys", 4, flags, ImVec2(0, -1))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
    const float cw = char_width();
    ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, 13 * cw);
    ImGui::TableSetupColumn("Where", ImGuiTableColumnFlags_WidthFixed, 9 * cw);
    ImGui::TableSetupColumn("Settings key", ImGuiTableColumnFlags_WidthFixed, 23 * cw);
    ImGui::TableHeadersRow();
    for (int a = 0; a < ACTION_COUNT; ++a) {
        const ActionInfo &info = action_info(static_cast<Action>(a));
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(info.label);
        ImGui::TableNextColumn();
        const std::string chord = chord_name(s.keys[a]);
        ImGui::TextUnformatted(chord.empty() ? "—" : chord.c_str());
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", scope_name(info.scope));
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", info.key);
    }
    ImGui::EndTable();
}

// The colour theme, then the font's family, weight and size, each font choice drawn in its own font;
// a change shows from the next frame.
void SettingsDialog::draw_appearance_tab(Settings *s, const std::string &settings_path) {
    FontChoice *font = &s->font;
    *font = sanitize_font_choice(*font);
    const int family = find_font_family(font->family);
    const float field_x = 9 * char_width();
    const float field_w = std::min(22 * char_width(), ImGui::GetContentRegionAvail().x - field_x);
    ImGui::TextDisabled("Colours");
    ImGui::Spacing();
    draw_theme_choice(&s->theme, field_x, field_w);
    ImGui::Spacing();
    ImGui::TextDisabled("Interface font");
    ImGui::Spacing();

    field_label("Family", field_x);
    ImGui::SetNextItemWidth(field_w);
    if (ImGui::BeginCombo("##family", font_family(family).name, ImGuiComboFlags_HeightLarge)) {
        for (int i = 0; i < FONT_FAMILY_COUNT; ++i) {
            const FontFamily &f = font_family(i);
            const bool selected = i == family;
            ImGui::PushID(i);
            ImGui::PushFont(ui_font(i, font->weight), 0.0f);
            if (ImGui::Selectable(f.name, selected)) font->family = f.id;
            if (f.proportional) {
                const char *tag = "proportional";
                ImGui::SameLine(ImGui::GetContentRegionAvail().x + ImGui::GetCursorPosX() - ImGui::CalcTextSize(tag).x);
                ImGui::TextDisabled("%s", tag);
            }
            ImGui::PopFont();
            if (selected) ImGui::SetItemDefaultFocus();
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }

    field_label("Weight", field_x);
    ImGui::SetNextItemWidth(field_w);
    if (ImGui::BeginCombo("##weight", font_weight_name(family, font->weight))) {
        for (FontWeight w : {FontWeight::REGULAR, FontWeight::HEAVY}) {
            ImGui::PushFont(ui_font(family, w), 0.0f);
            if (ImGui::Selectable(font_weight_name(family, w), w == font->weight)) font->weight = w;
            ImGui::PopFont();
            if (w == font->weight) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    field_label("Size", field_x);
    ImGui::SetNextItemWidth(field_w);
    int size = font->size;
    if (ImGui::SliderInt("##size", &size, FONT_SIZE_MIN, FONT_SIZE_MAX, "%d px", ImGuiSliderFlags_AlwaysClamp))
        font->size = size;

    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + field_x);
    const FontChoice def;
    ImGui::BeginDisabled(*font == def);
    if (ImGui::Button("Default##font")) *font = def;
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("%s", font_choice_label(def).c_str());
    ImGui::Spacing();

    ImGui::TextDisabled("Preview");
    ImGui::PushStyleColor(ImGuiCol_ChildBg, tok().PLOT_BG);
    const float lines = static_cast<float>(sizeof FONT_PREVIEW / sizeof FONT_PREVIEW[0]);
    const float h = lines * ImGui::GetTextLineHeightWithSpacing() + 2 * ImGui::GetStyle().WindowPadding.y;
    if (ImGui::BeginChild("##preview", ImVec2(0, h), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar))
        for (const char *line : FONT_PREVIEW) ImGui::TextUnformatted(line);
    ImGui::EndChild();
    ImGui::PopStyleColor();
    ImGui::Spacing();
    ImGui::Separator();
    note(("The whole workbench uses these colours and this font; a change applies at once and is saved to " +
          settings_path + ".")
            .c_str());
}

// The theme combo box: each theme with a strip of its colours, the current one's strip beside it.
void SettingsDialog::draw_theme_choice(std::string *theme, float field_x, float field_w) {
    const int current = std::max(0, find_theme(*theme));
    const float strip_h = ImGui::GetTextLineHeight();
    const float strip_w = palette_strip_width(strip_h);
    const ImGuiStyle &style = ImGui::GetStyle();
    float name_w = 0;
    for (int i = 0; i < THEME_COUNT; ++i) name_w = std::max(name_w, ImGui::CalcTextSize(theme_info(i).name).x);
    const float strip_x = name_w + 2 * style.ItemSpacing.x;
    field_label("Theme", field_x);
    ImGui::SetNextItemWidth(field_w);
    // The list is wide enough for every name and its strip, whatever the box's width.
    ImGui::SetNextWindowSizeConstraints(
        ImVec2(strip_x + strip_w + 2 * style.WindowPadding.x + style.ScrollbarSize, 0), ImVec2(FLT_MAX, FLT_MAX));
    if (ImGui::BeginCombo("##theme", theme_info(current).name, ImGuiComboFlags_HeightLarge)) {
        for (int i = 0; i < THEME_COUNT; ++i) {
            const bool selected = i == current;
            ImGui::PushID(i);
            const float x = ImGui::GetCursorPosX();
            if (ImGui::Selectable(theme_info(i).name, selected)) *theme = theme_info(i).id;
            ImGui::SameLine(x + strip_x);
            palette_strip(theme_palette(i), strip_h);
            if (selected) ImGui::SetItemDefaultFocus();
            ImGui::PopID();
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() + ImGui::GetStyle().FramePadding.y);
    palette_strip(theme_palette(current), strip_h);
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + field_x);
    ImGui::BeginDisabled(*theme == DEFAULT_THEME);
    if (ImGui::Button("Default##theme")) *theme = DEFAULT_THEME;
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("%s", theme_info(find_theme(DEFAULT_THEME)).name);
}

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
    begin_modal_frame(ImVec2(720, 460));
    bool keep = true;
    const bool visible = ImGui::BeginPopupModal(name.c_str(), &keep, ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleColor(3);
    open_ = visible;
    if (!visible) return false;
    ImGui::TextWrapped("%s", message_.c_str());
    ImGui::Spacing();
    ImGui::TextDisabled("Details");
    ImGui::PushStyleColor(ImGuiCol_FrameBg, tok().PLOT_BG);
    ImGui::InputTextMultiline(
        "##detail", detail_.data(), detail_.size() + 1, ImVec2(-1, -footer_height()), ImGuiInputTextFlags_ReadOnly);
    ImGui::PopStyleColor();
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
