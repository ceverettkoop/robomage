#ifndef RMWORK_SETTINGS_H
#define RMWORK_SETTINGS_H

#include <array>
#include <string>

#include "fonts.h"
#include "keys.h"
#include "layout.h"
#include "theme.h"

// ── The settings file: ~/.config/robomage/settings.json ($XDG_CONFIG_HOME respected) ───────────
//
// Read at start, written whole (atomically) whenever a setting changes. ImGui's own layout file
// lives beside it, in workbench/imgui.ini. A key the reader does not know is ignored and a missing
// one keeps its default, so an older or newer file still loads.

namespace rmwork {

struct TableBackground {
        enum class Kind {
            COLOR,
            TEXTURE,
            IMAGE
        };
        Kind kind = Kind::TEXTURE;
        std::string value = "wood";  // "#rrggbb", a texture key or an image path
};

struct Settings {
        Mode mode = Mode::REPLAY;
        std::array<ToolFlags, MODE_COUNT> tools{};  // per mode, which tools are open
        TableBackground table;
        std::string theme = DEFAULT_THEME;  // a theme id (theme.h); an unknown one reads as the default
        FontChoice font;                    // always sanitized: a known family, a size in range
        KeyBindings keys{};
};

// The table textures, by settings key, with their display names.
struct TextureInfo {
        const char *key;
        const char *name;
};
inline constexpr int TEXTURE_COUNT = 5;
extern const TextureInfo TABLE_TEXTURES[TEXTURE_COUNT];

struct ConfigPaths {
        std::string dir;        // <config>/robomage
        std::string settings;   // <dir>/settings.json
        std::string imgui_ini;  // <dir>/workbench/imgui.ini
};

// Each mode's default preset's tools open, the default bindings, a wood table.
Settings default_settings();
// `dir` when given, else $XDG_CONFIG_HOME/robomage, else $HOME/.config/robomage.
ConfigPaths config_paths(const std::string &dir);
std::string settings_to_json(const Settings &s);
// Starts from the defaults; false with `err` on malformed JSON or a value of the wrong kind. An
// unknown theme, font family or weight, or an out-of-range size is the default's (no error).
bool settings_from_json(const std::string &text, Settings *out, std::string *err);
// A missing file gives the defaults (true); an unreadable or malformed one false with `err`.
bool load_settings(const std::string &path, Settings *out, std::string *err);
// Creates the directory and writes the file atomically.
bool save_settings(const std::string &path, const Settings &s, std::string *err);

}  // namespace rmwork

#endif /* RMWORK_SETTINGS_H */
