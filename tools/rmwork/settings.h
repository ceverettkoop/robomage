#ifndef RMWORK_SETTINGS_H
#define RMWORK_SETTINGS_H

#include <array>
#include <string>
#include <vector>

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

// The engine the workbench runs (Play, Replay's replays of decision logs and re-evaluation); "" is the
// default (paths.h).
struct EngineSettings {
        std::string binary;   // default bin/<config>/robomage_nn, else robomage
        std::string workdir;  // default bin/
};

// One policy rmserve serves: `rmserve --policy <id>=<model>`.
struct PolicyEntry {
        std::string id;
        std::string model;
};

struct ModelSettings {
        std::string socket;  // the inference server's socket ("" = none)
        std::vector<PolicyEntry> policies;
        std::string run_dir;  // where training runs live; default runs/
        bool start_server = false;  // the workbench runs rmserve with these policies (inference_server.h)
};

// The twelve steps a stop can be set in (a stop at combat damage also covers first-strike damage).
inline constexpr int STOP_STEP_COUNT = 12;
struct StepInfo {
        const char *key;     // settings key ("declare_attackers")
        const char *name;    // "Declare attackers"
        const char *abbrev;  // "Atk", as the Table's step strip shows it
};
const StepInfo &stop_step(int index);
// The step with settings key `key`, or -1.
int find_stop_step(const std::string &key);
// The stop step of the state format's `step` enum value name ("FIRST_MAIN" -> main1; first-strike
// and regular combat damage share one), or -1.
int stop_step_of(const std::string &step_name);

// Play's default stops: per step, whether priority stops there with an empty stack.
struct StopSettings {
        std::array<bool, STOP_STEP_COUNT> own{};       // in your turn
        std::array<bool, STOP_STEP_COUNT> opponent{};  // in the opponent's turn
};
// The mock-up's: upkeep, main 1, declare attackers, declare blockers and end step, in both turns.
StopSettings default_stops();
// The stop of step `step` (0 .. STOP_STEP_COUNT - 1) in your turn (`own_turn`) or the opponent's.
bool *stop_flag(StopSettings *s, bool own_turn, int step);

// The New Game dialog's last choices (Play; play_session.h names the opponents).
struct NewGameSettings {
        std::string deck = "league/burn";  // your deck, relative to the engine's resources/decks (no .dk)
        int seat = 0;                      // your seat: 0 A, 1 B
        std::string opponent = "explore";  // the opponent's kind (its key: model, server, explore, random, scripted)
        std::string model;                 // model: the export (nn:<model>)
        std::string server;                // server: the inference server's socket (nn:remote:<socket>)
        std::string opponent_deck = "league/bug";
        bool bo3 = false;
        uint64_t seed = 1;
        bool auto_mana = true;
        bool model_read = false;  // evaluate your decisions through the inference server
};

struct Settings {
        Mode mode = Mode::REPLAY;
        std::array<ToolFlags, MODE_COUNT> tools{};     // per mode, which tools are open
        std::array<NamedLayouts, MODE_COUNT> layouts;  // per mode, the user's saved layouts
        TableBackground table;
        std::string theme = DEFAULT_THEME;  // a theme id (theme.h); an unknown one reads as the default
        FontChoice font;                    // always sanitized: a known family, a size in range
        KeyBindings keys{};
        EngineSettings engine;
        ModelSettings models;
        StopSettings stops = default_stops();
        std::string image_cache;          // the card-image cache; "" = default_image_cache()
        bool card_images = true;          // cards drawn as their images where the cache has them
        std::vector<std::string> recent;  // files and run directories opened, the latest first
        NewGameSettings new_game;
        bool start_page = true;  // the start page shows at start and when the open game or replay is closed
};

// How many recent files and directories are kept.
inline constexpr size_t RECENT_MAX = 12;
// Puts `path` first in `recent` (once), keeping at most RECENT_MAX.
void add_recent(std::vector<std::string> *recent, const std::string &path);

// The path settings resolved (paths.h): the setting, else its default; the socket's default is
// default_server_socket() while the workbench starts the server, else none ("").
std::string engine_binary_path(const Settings &s);
std::string engine_workdir_path(const Settings &s);
std::string run_dir_path(const Settings &s);
std::string image_cache_path(const Settings &s);
std::string socket_path(const Settings &s);
std::string policy_model_path(const PolicyEntry &p);

// A policy id rmserve takes: letters, digits, '_', '-' and '.'; false with `why`.
bool policy_id_valid(const std::string &id, std::string *why);
// Appends a policy with a valid id no other policy has and a model path; false with `why`.
bool add_policy(ModelSettings *m, const std::string &id, const std::string &model, std::string *why);

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

// Each mode's default preset's tools open, the default bindings, a wood table, no saved layouts,
// the default paths and stops.
Settings default_settings();
// `dir` when given, else $XDG_CONFIG_HOME/robomage, else $HOME/.config/robomage.
ConfigPaths config_paths(const std::string &dir);
std::string settings_to_json(const Settings &s);
// Starts from the defaults; false with `err` on malformed JSON, a value of the wrong kind, an
// unknown key name or step, two actions bound to one key where both work, a malformed or misnamed
// layout, or a bad policy row. An unknown theme, font family or weight, or an out-of-range size is
// the default's (no error).
bool settings_from_json(const std::string &text, Settings *out, std::string *err);
// A missing file gives the defaults (true); an unreadable or malformed one false with `err`.
bool load_settings(const std::string &path, Settings *out, std::string *err);
// Creates the directory and writes the file atomically.
bool save_settings(const std::string &path, const Settings &s, std::string *err);

}  // namespace rmwork

#endif /* RMWORK_SETTINGS_H */
