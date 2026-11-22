#include "settings_dialog.h"

#include <algorithm>
#include <cfloat>
#include <cctype>
#include <cstdio>
#include <cstring>

#include "modal.h"
#include "model_zip.h"
#include "paths.h"
#include "theme.h"
#include "widgets.h"

namespace {

const char *const SETTINGS_POPUP = "Settings###rmwork.settings";
const char *const TAB_NAMES[] = {"Table", "Card Printings", "Keys", "Appearance", "Engine", "Models", "Stops"};
const char *const TAB_KEYS[] = {"table", "printings", "keys", "appearance", "engine", "models", "stops"};
// The Appearance tab's preview lines.
const char *const FONT_PREVIEW[] = {"Player A casts Brainstorm. Turn 4 · Decision 212 → 213 · V −0.42",
    "{1}{U}{U}  3/3  0123456789  ( ) [ ] { } < > = + − * / % & | ! ? @ # …",
    "The quick brown fox jumps over the lazy dog. Il1| O0 rn m"};
// The mock-up's colour swatches for the table.
const ImU32 TABLE_COLORS[] = {rmwork::hex_color(0x6e4526), rmwork::hex_color(0x0b6b2a), rmwork::hex_color(0x008080),
    rmwork::hex_color(0x1d2836), rmwork::hex_color(0x3a3f47), rmwork::hex_color(0x101215)};

}  // namespace

namespace rmwork {

int SettingsDialog::find_tab(const std::string &key) {
    for (int i = 0; i < TAB_COUNT; ++i)
        if (key == TAB_KEYS[i]) return i;
    return -1;
}

void SettingsDialog::open(Tab tab) {
    want_open_ = !open_;
    tab_ = tab;
    select_tab_ = true;
}

void SettingsDialog::begin_key_capture(Action a) {
    open(KEYS);
    capture_ = action_index(a);
    key_message_.clear();
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
    begin_modal_frame(ImVec2(860, 600));
    bool keep = true;
    const bool visible = ImGui::BeginPopupModal(SETTINGS_POPUP, &keep, ImGuiWindowFlags_NoSavedSettings);
    ImGui::PopStyleColor(3);
    open_ = visible;
    if (!visible) {
        capture_ = -1;
        return;
    }
    // Esc cancels a key capture rather than closing the dialog.
    const bool capturing_now = capture_ >= 0;
    ImGui::BeginChild("##settings_body", ImVec2(0, -footer_height()));
    if (ImGui::BeginTabBar("##settings_tabs")) {
        bool keys_shown = false;
        for (int i = 0; i < TAB_COUNT; ++i) {
            const ImGuiTabItemFlags f = select_tab_ && i == tab_ ? ImGuiTabItemFlags_SetSelected : 0;
            if (!ImGui::BeginTabItem(TAB_NAMES[i], nullptr, f)) continue;
            ImGui::Spacing();
            if (i == TABLE) draw_table_tab(s, settings_path);
            if (i == PRINTINGS) draw_printings_tab(s);
            if (i == KEYS) {
                keys_shown = true;
                draw_keys_tab(s);
            }
            if (i == APPEARANCE) draw_appearance_tab(s, settings_path);
            if (i == ENGINE) draw_engine_tab(s);
            if (i == MODELS) draw_models_tab(s);
            if (i == STOPS) draw_stops_tab(s);
            ImGui::EndTabItem();
        }
        if (!keys_shown && !select_tab_) capture_ = -1;
        select_tab_ = false;
        ImGui::EndTabBar();
    }
    ImGui::EndChild();
    ImGui::Separator();
    if (right_aligned_button("Close") || (!capturing_now && close_requested()) || !keep) {
        ImGui::CloseCurrentPopup();
        open_ = false;
        capture_ = -1;
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

void SettingsDialog::draw_printings_tab(Settings *s) {
    note(
        "Each card is drawn with its oldest paper printing unless you choose another here; the choices are kept in "
        "printings.json beside the settings.");
    path_field("Image cache", &s->image_cache, default_image_cache(), Want::DIRECTORY, 11 * char_width(),
        "missing: tools/scryfall/fetch_images.py creates it when it fetches the first image");
    printings_.draw(s, images_);
}

// The bindings, a row each: click a row and press the new chord (Esc cancels); each row resets to
// its default, Reset All resets every one.
void SettingsDialog::draw_keys_tab(Settings *s) {
    if (capture_ >= 0) capture_key(s);
    note("Click an action, then press its new key (with Ctrl, Shift, Alt or Super if you like); Esc cancels.");
    if (!key_message_.empty()) colored_note(key_message_.c_str(), key_message_error_ ? tok().BAD : tok().DIM);
    const KeyBindings defaults = default_bindings();
    ImGui::BeginDisabled(s->keys == defaults);
    if (ImGui::Button("Reset All")) reset_all_keys(s);
    ImGui::EndDisabled();
    ImGui::Spacing();
    const ImGuiTableFlags flags =
        ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_ScrollY;
    if (!ImGui::BeginTable("##keys", 5, flags, ImVec2(0, -1))) return;
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthStretch);
    const float cw = char_width();
    ImGui::TableSetupColumn("Key", ImGuiTableColumnFlags_WidthFixed, 15 * cw);
    ImGui::TableSetupColumn("Where", ImGuiTableColumnFlags_WidthFixed, 9 * cw);
    ImGui::TableSetupColumn("Settings key", ImGuiTableColumnFlags_WidthFixed, 23 * cw);
    ImGui::TableSetupColumn("##reset", ImGuiTableColumnFlags_WidthFixed, 7 * cw);
    ImGui::TableHeadersRow();
    for (int a = 0; a < ACTION_COUNT; ++a) {
        const Action action = static_cast<Action>(a);
        const ActionInfo &info = action_info(action);
        const bool capturing_row = capture_ == a;
        ImGui::PushID(a);
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        if (capturing_row) ImGui::TableSetBgColor(ImGuiTableBgTarget_RowBg1, tok().SEL);
        if (ImGui::Selectable(
                info.label, capturing_row, ImGuiSelectableFlags_SpanAllColumns | ImGuiSelectableFlags_AllowOverlap)) {
            capture_ = capturing_row ? -1 : a;
            key_message_.clear();
        }
        ImGui::TableNextColumn();
        if (capturing_row) {
            ImGui::TextColored(ImGui::ColorConvertU32ToFloat4(tok().ACCENT), "press a key…");
        } else {
            const std::string chord = chord_label(s->keys[a]);
            ImGui::TextUnformatted(chord.empty() ? "—" : chord.c_str());
        }
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", scope_name(info.scope));
        ImGui::TableNextColumn();
        ImGui::TextDisabled("%s", info.key);
        ImGui::TableNextColumn();
        if (s->keys[a] != defaults[a] && ImGui::SmallButton("Reset")) reset_key(s, action);
        ImGui::PopID();
    }
    ImGui::EndTable();
}

bool SettingsDialog::reset_key(Settings *s, Action a) {
    const ImGuiKeyChord def = default_bindings()[action_index(a)];
    std::string why;
    key_message_error_ = !rebind(&s->keys, a, def, &why);
    key_message_ =
        key_message_error_ ? why : std::string(action_info(a).label) + " is back to " + chord_label(def) + ".";
    capture_ = -1;
    return !key_message_error_;
}

void SettingsDialog::reset_all_keys(Settings *s) {
    s->keys = default_bindings();
    capture_ = -1;
    key_message_ = "Every key is back to its default.";
    key_message_error_ = false;
}

// The chord pressed this frame becomes the captured action's, unless another action answers to it
// where both work; Esc gives up.
void SettingsDialog::capture_key(Settings *s) {
    const Action action = static_cast<Action>(capture_);
    if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
        capture_ = -1;
        key_message_ = "Kept the key of " + std::string(action_info(action).label) + ".";
        key_message_error_ = false;
        return;
    }
    const ImGuiKeyChord chord = pressed_chord();
    if (chord == 0) return;
    capture_ = -1;
    std::string why;
    if (rebind(&s->keys, action, chord, &why)) {
        key_message_ = std::string(action_info(action).label) + " is now " + chord_label(chord) + ".";
        key_message_error_ = false;
    } else {
        key_message_ = why;
        key_message_error_ = true;
    }
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

// The engine binary and its working directory.
void SettingsDialog::draw_engine_tab(Settings *s) {
    const float field_x = 16 * char_width();
    note(
        "The engine the workbench starts for Play, for replaying a decision log in Replay and for Replay's "
        "re-evaluation. A replay runs the plain robomage beside a robomage_nn named here (a replay needs no learned "
        "policy). Leave a field empty for its default; a relative path is under the repository, ~/ under your home "
        "directory.");
    ImGui::Spacing();
    path_field("Engine binary", &s->engine.binary, default_engine_binary(), Want::EXECUTABLE, field_x,
        "missing: build it with make actor (robomage_nn) or make (robomage)");
    ImGui::Spacing();
    path_field("Working dir", &s->engine.workdir, default_engine_workdir(), Want::DIRECTORY, field_x,
        "missing: the engine reads resources/ under this directory");
}

// The inference server's socket, the policies it serves and where training runs are kept.
void SettingsDialog::draw_models_tab(Settings *s) {
    const float field_x = 16 * char_width();
    ImGui::Checkbox("Start the server with the workbench", &s->models.start_server);
    ImGui::SetItemTooltip("Runs rmserve (make actor) with the policies below while the workbench is open, "
                          "unless a server already listens at the socket");
    if (s->models.start_server) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Restart server")) server_restart_ = true;
        if (!server_status_.empty()) colored_note(server_status_.c_str(), server_failed_ ? tok().BAD : tok().DIM);
    }
    ImGui::Spacing();
    path_field("Server socket", &s->models.socket, s->models.start_server ? default_server_socket() : "",
        Want::SOCKET, field_x,
        s->models.start_server ? "not listening yet: the workbench starts the server at this socket"
                               : "start rmserve --socket PATH; the status bar then shows the socket present");
    ImGui::Spacing();
    path_field("Run directory", &s->models.run_dir, default_run_dir(), Want::DIRECTORY, field_x,
        "missing: rmtrain start creates runs here");
    ImGui::Spacing();
    ImGui::Separator();
    ImGui::TextDisabled("Policies (rmserve --policy ID=MODEL)");
    ImGui::Spacing();
    draw_policies(*s, &s->models, field_x);
}

// The policy rows (each removable) and a row to add one, refused with a message when its id is
// malformed or taken or its model path empty. While the model field is edited, the exports under the
// run directory and the models directory that match it are listed under the row.
void SettingsDialog::draw_policies(const Settings &s, ModelSettings *m, float field_x) {
    const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter;
    const float cw = char_width();
    if (ImGui::BeginTable("##policies", 4, flags)) {
        ImGui::TableSetupColumn("Id", ImGuiTableColumnFlags_WidthFixed, 12 * cw);
        ImGui::TableSetupColumn("Model", ImGuiTableColumnFlags_WidthStretch);
        ImGui::TableSetupColumn("##found", ImGuiTableColumnFlags_WidthFixed, 9 * cw);
        ImGui::TableSetupColumn("##remove", ImGuiTableColumnFlags_WidthFixed, 8 * cw);
        ImGui::TableHeadersRow();
        int remove = -1;
        for (size_t i = 0; i < m->policies.size(); ++i) {
            const PolicyEntry &p = m->policies[i];
            ImGui::PushID(static_cast<int>(i));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(p.id.c_str());
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(p.model.c_str());
            ImGui::TableNextColumn();
            const bool found = model_problem(policy_model_path(p)).empty();
            ImGui::TextColored(
                ImGui::ColorConvertU32ToFloat4(found ? tok().GOOD : tok().BAD), "%s", found ? "found" : "missing");
            ImGui::TableNextColumn();
            if (ImGui::SmallButton("Remove")) remove = static_cast<int>(i);
            ImGui::PopID();
        }
        if (m->policies.empty()) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextDisabled("none");
        }
        ImGui::EndTable();
        if (remove >= 0) m->policies.erase(m->policies.begin() + remove);
    }
    ImGui::Spacing();
    field_label("Add", 0);
    ImGui::SetCursorPosX(field_x);
    ImGui::SetNextItemWidth(12 * cw);
    ImGui::InputTextWithHint("##policy_id", "id", policy_id_, sizeof policy_id_);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(10 * cw, ImGui::GetContentRegionAvail().x - 8 * cw));
    ImGui::InputTextWithHint("##policy_model", "model (.pt2, TorchScript, or archive.zip!member)", policy_model_,
        sizeof policy_model_);
    if (ImGui::IsItemActivated()) {
        model_choices_ = NewGameDialog::list_exports(s);
        suggest_open_ = true;
    }
    const bool model_editing = ImGui::IsItemActive();
    const float model_w = ImGui::GetItemRectSize().x;
    ImGui::SameLine();
    if (ImGui::Button("Add")) {
        if (add_policy(m, policy_id_, policy_model_, &policy_error_)) {
            policy_id_[0] = policy_model_[0] = '\0';
            policy_error_.clear();
        }
    }
    if (!policy_error_.empty()) colored_note(policy_error_.c_str(), tok().BAD);
    if (suggest_open_ && (model_editing || suggest_hovered_))
        draw_model_suggestions(m, field_x, model_w);
    else
        suggest_hovered_ = false;
}

// The Add row's suggestions under it: a click puts the export's path in the model field, and the
// next free number in an empty id field.
void SettingsDialog::draw_model_suggestions(ModelSettings *m, float field_x, float width) {
    const std::vector<const ModelChoice *> shown = matching_models(model_choices_, policy_model_);
    const float row = ImGui::GetTextLineHeightWithSpacing();
    const int rows = std::max(1, std::min(8, static_cast<int>(shown.size())));
    ImGui::SetCursorPosX(field_x + 12 * char_width() + ImGui::GetStyle().ItemSpacing.x);
    ImGui::BeginChild("##model_suggestions", ImVec2(width, rows * row + ImGui::GetStyle().WindowPadding.y * 2),
        ImGuiChildFlags_Borders);
    suggest_hovered_ = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem);
    for (size_t i = 0; i < shown.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        if (ImGui::Selectable(shown[i]->label.c_str())) {
            std::snprintf(policy_model_, sizeof policy_model_, "%s", shown[i]->path.c_str());
            if (!policy_id_[0]) std::snprintf(policy_id_, sizeof policy_id_, "%s", next_policy_id(*m).c_str());
            suggest_open_ = suggest_hovered_ = false;
        }
        ImGui::SetItemTooltip("%s", shown[i]->path.c_str());
        ImGui::PopID();
    }
    if (shown.empty())
        ImGui::TextDisabled(model_choices_.empty() ? "No exports under the run directory or in models/"
                                                   : "No export under the run directory or in models/ matches");
    ImGui::EndChild();
}

std::vector<const ModelChoice *> SettingsDialog::matching_models(const std::vector<ModelChoice> &all, const char *text) {
    std::string want = text;
    std::transform(want.begin(), want.end(), want.begin(), [](unsigned char c) { return std::tolower(c); });
    std::vector<const ModelChoice *> out;
    for (const ModelChoice &c : all) {
        std::string hay = c.label + '\n' + c.path;
        std::transform(hay.begin(), hay.end(), hay.begin(), [](unsigned char ch) { return std::tolower(ch); });
        if (hay.find(want) != std::string::npos) out.push_back(&c);
    }
    return out;
}

// Play's default stops: in which steps priority stops when the stack is empty, in your turn and in
// the opponent's.
void SettingsDialog::draw_stops_tab(Settings *s) {
    note(
        "In Play, priority passes by itself in a step without a stop while the stack is empty; F2 and F6 pass "
        "further (§2.3). A click on a step of the Table's step bar in Play toggles the same stop, for the side "
        "whose turn it is. A stop at combat damage covers first-strike damage too.");
    ImGui::Spacing();
    const ImGuiTableFlags flags = ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_RowBg;
    if (ImGui::BeginTable("##stops", 1 + STOP_STEP_COUNT, flags)) {
        ImGui::TableSetupColumn("##turn", ImGuiTableColumnFlags_WidthFixed, 16 * char_width());
        for (int i = 0; i < STOP_STEP_COUNT; ++i) ImGui::TableSetupColumn(stop_step(i).abbrev);
        ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
        for (int c = 0; c <= STOP_STEP_COUNT; ++c) {
            ImGui::TableSetColumnIndex(c);
            ImGui::TableHeader(ImGui::TableGetColumnName(c));
            if (c > 0 && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", stop_step(c - 1).name);
        }
        stop_row("Your turn", &s->stops.own);
        stop_row("Opponent's turn", &s->stops.opponent);
        ImGui::EndTable();
    }
    ImGui::Spacing();
    const StopSettings def = default_stops();
    ImGui::BeginDisabled(s->stops.own == def.own && s->stops.opponent == def.opponent);
    if (ImGui::Button("Default")) s->stops = def;
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextDisabled("upkeep, main 1, declare attackers, declare blockers, end step");
}

// A row of the stops table: the turn, then a checkbox per step; true when one changed.
bool SettingsDialog::stop_row(const char *label, std::array<bool, STOP_STEP_COUNT> *row) {
    bool changed = false;
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(label);
    ImGui::PushID(label);
    for (int i = 0; i < STOP_STEP_COUNT; ++i) {
        ImGui::TableNextColumn();
        ImGui::PushID(i);
        bool on = (*row)[static_cast<size_t>(i)];
        if (ImGui::Checkbox("##stop", &on)) {
            (*row)[static_cast<size_t>(i)] = on;
            changed = true;
        }
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s: %s", label, stop_step(i).name);
        ImGui::PopID();
    }
    ImGui::PopID();
    return changed;
}

void SettingsDialog::path_field(const char *label, std::string *value, const std::string &fallback, Want want,
    float field_x, const char *missing_note) {
    ImGui::PushID(label);
    field_label(label, field_x);
    char buf[1024];
    std::snprintf(buf, sizeof buf, "%s", value->c_str());
    ImGui::SetNextItemWidth(-1);
    const std::string hint = fallback.empty() ? std::string("none") : fallback;
    if (ImGui::InputTextWithHint("##path", hint.c_str(), buf, sizeof buf)) *value = buf;
    ImGui::SetCursorPosX(ImGui::GetCursorPosX() + field_x);
    path_mark(resolve_path(*value, fallback), want, missing_note);
    ImGui::PopID();
}

// "● found  /path" or "● missing  /path: what to do", on the current line.
void SettingsDialog::path_mark(const std::string &resolved, Want want, const char *missing_note) {
    const bool found = path_found(resolved, want);
    const ImVec2 p = ImGui::GetCursorScreenPos();
    const float h = ImGui::GetTextLineHeight();
    const ImU32 dot = resolved.empty() ? tok().DIM : found ? tok().GOOD : tok().BAD;
    ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(p.x + 4, p.y + h / 2), 3.5f, dot);
    ImGui::Dummy(ImVec2(12, h));
    ImGui::SameLine(0, 0);
    if (resolved.empty()) {
        ImGui::TextDisabled("%s", missing_note);
        return;
    }
    ImGui::TextColored(
        ImGui::ColorConvertU32ToFloat4(found ? tok().GOOD : tok().BAD), "%s", found ? "found" : "missing");
    ImGui::SameLine();
    const std::string text = found ? resolved : resolved + " — " + missing_note;
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%s", text.c_str());
    ImGui::PopTextWrapPos();
}

bool SettingsDialog::path_found(const std::string &resolved, Want want) {
    switch (want) {
        case Want::EXECUTABLE:
            return is_executable(resolved);
        case Want::DIRECTORY:
            return path_kind(resolved) == PathKind::DIRECTORY;
        case Want::SOCKET:
            return path_kind(resolved) == PathKind::SOCKET;
        case Want::FILE:
            break;
    }
    return path_kind(resolved) == PathKind::FILE;
}

}  // namespace rmwork
