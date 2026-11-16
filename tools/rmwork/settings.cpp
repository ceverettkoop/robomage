#include "settings.h"

#include <cstdlib>

#include "../../src/featurize/json.h"
#include "files.h"

namespace {

const char *const BACKGROUND_KINDS[] = {"color", "texture", "image"};

}  // namespace

static const char *background_kind_key(rmwork::TableBackground::Kind k);
static bool read_mode(const featurize::Json &v, rmwork::Mode *out, std::string *err);
static bool read_tools(
    const featurize::Json &v, std::array<rmwork::ToolFlags, rmwork::MODE_COUNT> *out, std::string *err);
static bool read_background(const featurize::Json &v, rmwork::TableBackground *out, std::string *err);
static bool read_font(const featurize::Json &v, rmwork::FontChoice *out, std::string *err);
static bool read_keys(const featurize::Json &v, rmwork::KeyBindings *out, std::string *err);

static const char *background_kind_key(rmwork::TableBackground::Kind k) {
    return BACKGROUND_KINDS[static_cast<int>(k)];
}

static bool read_mode(const featurize::Json &v, rmwork::Mode *out, std::string *err) {
    for (int m = 0; m < rmwork::MODE_COUNT; ++m) {
        if (v.is_string() && v.as_string() == rmwork::mode_key(static_cast<rmwork::Mode>(m))) {
            *out = static_cast<rmwork::Mode>(m);
            return true;
        }
    }
    *err = "\"mode\" is not \"play\" or \"replay\"";
    return false;
}

static bool read_tools(
    const featurize::Json &v, std::array<rmwork::ToolFlags, rmwork::MODE_COUNT> *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"tools\" is not an object";
        return false;
    }
    for (int m = 0; m < rmwork::MODE_COUNT; ++m) {
        const featurize::Json *mode = v.get(rmwork::mode_key(static_cast<rmwork::Mode>(m)));
        if (!mode) continue;
        if (!mode->is_object()) {
            *err = std::string("\"tools\".\"") + rmwork::mode_key(static_cast<rmwork::Mode>(m)) + "\" is not an object";
            return false;
        }
        for (int t = 0; t < rmwork::TOOL_COUNT; ++t) {
            const char *key = rmwork::tool_info(static_cast<rmwork::Tool>(t)).key;
            const featurize::Json *open = mode->get(key);
            if (!open) continue;
            if (!open->is_bool()) {
                *err = std::string("tool \"") + key + "\" is not true or false";
                return false;
            }
            (*out)[m][t] = open->as_bool();
        }
    }
    return true;
}

static bool read_background(const featurize::Json &v, rmwork::TableBackground *out, std::string *err) {
    const featurize::Json *kind = v.get("kind");
    const featurize::Json *value = v.get("value");
    if (!v.is_object() || !kind || !kind->is_string() || !value || !value->is_string()) {
        *err = "\"table_background\" is not {\"kind\": …, \"value\": …}";
        return false;
    }
    for (int k = 0; k < 3; ++k) {
        if (kind->as_string() == BACKGROUND_KINDS[k]) {
            out->kind = static_cast<rmwork::TableBackground::Kind>(k);
            out->value = value->as_string();
            return true;
        }
    }
    *err = "\"table_background\".\"kind\" is not color, texture or image";
    return false;
}

// {"family": …, "weight": …, "size": …}, each optional; a value of a known kind but unknown (a
// family not embedded, a weight the family lacks, a size out of range or fractional) is the
// default's.
static bool read_font(const featurize::Json &v, rmwork::FontChoice *out, std::string *err) {
    using namespace rmwork;
    const featurize::Json *family = v.get("family");
    const featurize::Json *weight = v.get("weight");
    const featurize::Json *size = v.get("size");
    if (!v.is_object() || (family && !family->is_string()) || (weight && !weight->is_string()) ||
        (size && !size->is_number())) {
        *err = "\"font\" is not {\"family\": \"…\", \"weight\": \"…\", \"size\": N}";
        return false;
    }
    const FontChoice def;
    *out = def;
    if (family && find_font_family(family->as_string()) >= 0) out->family = family->as_string();
    if (weight && !parse_font_weight(find_font_family(out->family), weight->as_string(), &out->weight))
        out->weight = def.weight;
    if (size && size->is_integer() && size->as_int() >= FONT_SIZE_MIN && size->as_int() <= FONT_SIZE_MAX)
        out->size = static_cast<int>(size->as_int());
    return true;
}

static bool read_keys(const featurize::Json &v, rmwork::KeyBindings *out, std::string *err) {
    if (!v.is_object()) {
        *err = "\"keys\" is not an object";
        return false;
    }
    for (const auto &member : v.members()) {
        const int a = rmwork::find_action(member.first);
        if (a < 0) continue;
        ImGuiKeyChord chord = 0;
        if (!member.second.is_string() || !rmwork::parse_chord(member.second.as_string(), &chord)) {
            *err = "key binding \"" + member.first + "\" is not a key such as \"Ctrl+O\" or \"RightArrow\"";
            return false;
        }
        (*out)[a] = chord;
    }
    return true;
}

namespace rmwork {

const TextureInfo TABLE_TEXTURES[TEXTURE_COUNT] = {
    {"wood", "Wood"}, {"felt", "Felt"}, {"linen", "Linen"}, {"tiles", "Tiles"}, {"slate", "Slate"}};

Settings default_settings() {
    Settings s;
    for (int m = 0; m < MODE_COUNT; ++m) s.tools[m] = preset_tools(presets()[default_preset(static_cast<Mode>(m))]);
    s.keys = default_bindings();
    return s;
}

ConfigPaths config_paths(const std::string &dir) {
    ConfigPaths p;
    if (!dir.empty()) {
        p.dir = dir;
    } else {
        const char *xdg = std::getenv("XDG_CONFIG_HOME");
        const char *home = std::getenv("HOME");
        if (xdg && *xdg)
            p.dir = std::string(xdg) + "/robomage";
        else
            p.dir = std::string(home && *home ? home : ".") + "/.config/robomage";
    }
    p.settings = p.dir + "/settings.json";
    p.imgui_ini = p.dir + "/workbench/imgui.ini";
    return p;
}

std::string settings_to_json(const Settings &s) {
    using featurize::json_quote;
    std::string out = "{\n  \"version\": 1,\n";
    out += "  \"mode\": " + json_quote(mode_key(s.mode)) + ",\n";
    out += "  \"tools\": {\n";
    for (int m = 0; m < MODE_COUNT; ++m) {
        out += "    " + json_quote(mode_key(static_cast<Mode>(m))) + ": {";
        for (int t = 0; t < TOOL_COUNT; ++t) {
            out += std::string(t ? ", " : "") + json_quote(tool_info(static_cast<Tool>(t)).key) + ": " +
                   (s.tools[m][t] ? "true" : "false");
        }
        out += m + 1 < MODE_COUNT ? "},\n" : "}\n";
    }
    out += "  },\n";
    out += "  \"table_background\": {\"kind\": " + json_quote(background_kind_key(s.table.kind)) +
           ", \"value\": " + json_quote(s.table.value) + "},\n";
    out += "  \"theme\": " + json_quote(find_theme(s.theme) >= 0 ? s.theme : DEFAULT_THEME) + ",\n";
    const FontChoice font = sanitize_font_choice(s.font);
    out += "  \"font\": {\"family\": " + json_quote(font.family) +
           ", \"weight\": " + json_quote(font_weight_key(find_font_family(font.family), font.weight)) +
           ", \"size\": " + std::to_string(font.size) + "},\n";
    out += "  \"keys\": {\n";
    for (int a = 0; a < ACTION_COUNT; ++a) {
        out += "    " + json_quote(action_info(static_cast<Action>(a)).key) + ": " + json_quote(chord_name(s.keys[a])) +
               (a + 1 < ACTION_COUNT ? ",\n" : "\n");
    }
    out += "  }\n}\n";
    return out;
}

bool settings_from_json(const std::string &text, Settings *out, std::string *err) {
    *out = default_settings();
    featurize::Json doc;
    if (!featurize::Json::parse(text, &doc, err)) return false;
    if (!doc.is_object()) {
        *err = "not a JSON object";
        return false;
    }
    if (const featurize::Json *v = doc.get("mode"))
        if (!read_mode(*v, &out->mode, err)) return false;
    if (const featurize::Json *v = doc.get("tools"))
        if (!read_tools(*v, &out->tools, err)) return false;
    if (const featurize::Json *v = doc.get("table_background"))
        if (!read_background(*v, &out->table, err)) return false;
    if (const featurize::Json *v = doc.get("theme")) {
        if (!v->is_string()) {
            *err = "\"theme\" is not a theme name such as \"imhex-dark\"";
            return false;
        }
        if (find_theme(v->as_string()) >= 0) out->theme = v->as_string();
    }
    if (const featurize::Json *v = doc.get("font"))
        if (!read_font(*v, &out->font, err)) return false;
    if (const featurize::Json *v = doc.get("keys"))
        if (!read_keys(*v, &out->keys, err)) return false;
    return true;
}

bool load_settings(const std::string &path, Settings *out, std::string *err) {
    *out = default_settings();
    if (!file_exists(path)) return true;
    std::string text;
    if (!read_file(path, &text)) {
        *err = "cannot read " + path;
        return false;
    }
    std::string why;
    if (!settings_from_json(text, out, &why)) {
        *out = default_settings();
        *err = path + ": " + why;
        return false;
    }
    return true;
}

bool save_settings(const std::string &path, const Settings &s, std::string *err) {
    const std::string dir = parent_dir(path);
    if (!dir.empty() && !make_dirs(dir)) {
        *err = "cannot create " + dir;
        return false;
    }
    return write_file_atomic(path, settings_to_json(s), err);
}

}  // namespace rmwork
