#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <string>

#include "app.h"
#include "files.h"
#include "platform.h"
#include "selftest.h"
#include "theme.h"

namespace {

struct CliOptions {
        bool selftest = false;
        bool open_settings = false;
        int settings_tab = rmwork::SettingsDialog::TABLE;
        // The font flags, each empty / 0 when not given.
        std::string theme;
        std::string font;
        std::string font_weight;
        int font_size = 0;
        bool demo_error = false;
        std::string mode;
        std::string preset;
        std::string config_dir;
        rmwork::WindowOptions window;
};

}  // namespace

static void usage();
static bool parse_size(const std::string &text, int *w, int *h);
static bool parse_args(int argc, char **argv, CliOptions *opt, int *exit_code);
static bool check_font_flags(const CliOptions &opt);
static rmwork::FontChoice font_override(const CliOptions &opt, const rmwork::FontChoice &from);
static void show_demo_error(rmwork::App &app);

static void usage() {
    std::printf(
        "usage: rmwork [--mode play|replay] [--preset NAME] [--config-dir DIR] [--size WxH]\n"
        "       rmwork --screenshot FILE.png [--mode …] [--preset …] [--size WxH] [--config-dir DIR] "
        "[--open-settings] [--settings-tab TAB] [--theme NAME] [--font FAMILY] [--font-weight W] [--font-size N]\n"
        "       rmwork --selftest\n"
        "The Robomage workbench (docs/workbench_plan.md). Settings: $XDG_CONFIG_HOME/robomage/settings.json\n"
        "(default ~/.config/robomage), the layout beside it in workbench/imgui.ini.\n"
        "  --mode MODE        start in Play or Replay\n"
        "  --preset NAME      apply a preset layout to the starting mode: replay (Replay (ImHex)), play,\n"
        "                     analysis, table (Table only)\n"
        "  --config-dir DIR   keep the settings and layout in DIR instead\n"
        "  --size WxH         the window's size (default 1600x1000)\n"
        "  --screenshot FILE  render 30 frames, write the last to FILE (PNG) and exit; reads the settings\n"
        "                     and layout but writes neither\n"
        "  --open-settings    open Edit → Settings at start\n"
        "  --settings-tab TAB open Settings at a tab: table, printings, keys, appearance\n"
        "  --theme NAME       the colour theme: imhex-dark, imhex-light, imgui-dark, imgui-classic,\n"
        "                     solarized-dark, solarized-light, nord, windows-95\n"
        "  --font FAMILY      the UI font: jetbrains-mono, fira-code, ibm-plex-mono, source-code-pro,\n"
        "                     cascadia-mono, hack, inter, ibm-plex-sans (as Settings → Appearance sets it)\n"
        "  --font-weight W    regular or heavy (the family's Medium, SemiBold or Bold)\n"
        "  --font-size N      the UI font's size in pixels, 11 to 20\n"
        "  --selftest         headless checks of the presets, modes and settings file; prints\n"
        "                     RMWORK-SELFTEST: N checks, M failures\n");
}

static bool parse_size(const std::string &text, int *w, int *h) {
    char *end = nullptr;
    const long a = std::strtol(text.c_str(), &end, 10);
    if (!end || *end != 'x') return false;
    const char *rest = end + 1;
    const long b = std::strtol(rest, &end, 10);
    if (end == rest || *end != '\0' || a < 320 || b < 240 || a > 16384 || b > 16384) return false;
    *w = static_cast<int>(a);
    *h = static_cast<int>(b);
    return true;
}

// False when the process should exit with `*exit_code` (help, or a bad argument).
static bool parse_args(int argc, char **argv, CliOptions *opt, int *exit_code) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const bool has_value = i + 1 < argc;
        if (a == "-h" || a == "--help") {
            usage();
            *exit_code = 0;
            return false;
        } else if (a == "--selftest") {
            opt->selftest = true;
        } else if (a == "--open-settings") {
            opt->open_settings = true;
        } else if (a == "--settings-tab" && has_value) {
            opt->open_settings = true;
            opt->settings_tab = rmwork::SettingsDialog::find_tab(argv[++i]);
            if (opt->settings_tab < 0) {
                std::fprintf(stderr, "rmwork: no settings tab '%s' (table, printings, keys, appearance)\n", argv[i]);
                *exit_code = 2;
                return false;
            }
        } else if (a == "--theme" && has_value) {
            opt->theme = argv[++i];
            if (rmwork::find_theme(opt->theme) < 0) {
                std::fprintf(stderr, "rmwork: no theme '%s'; one of:", argv[i]);
                for (int t = 0; t < rmwork::THEME_COUNT; ++t) std::fprintf(stderr, " %s", rmwork::theme_info(t).id);
                std::fprintf(stderr, "\n");
                *exit_code = 2;
                return false;
            }
        } else if (a == "--font" && has_value) {
            opt->font = argv[++i];
        } else if (a == "--font-weight" && has_value) {
            opt->font_weight = argv[++i];
        } else if (a == "--font-size" && has_value) {
            opt->font_size = std::atoi(argv[++i]);
            if (opt->font_size == 0) opt->font_size = -1;
        } else if (a == "--demo-error") {
            opt->demo_error = true;
        } else if (a == "--mode" && has_value) {
            opt->mode = argv[++i];
        } else if (a == "--preset" && has_value) {
            opt->preset = argv[++i];
        } else if (a == "--config-dir" && has_value) {
            opt->config_dir = argv[++i];
        } else if (a == "--screenshot" && has_value) {
            opt->window.screenshot = argv[++i];
        } else if (a == "--size" && has_value && parse_size(argv[i + 1], &opt->window.width, &opt->window.height)) {
            ++i;
        } else {
            std::fprintf(stderr, "rmwork: bad argument '%s'\n", a.c_str());
            usage();
            *exit_code = 2;
            return false;
        }
    }
    if (!opt->mode.empty() && opt->mode != "play" && opt->mode != "replay") {
        std::fprintf(stderr, "rmwork: --mode is play or replay, not '%s'\n", opt->mode.c_str());
        *exit_code = 2;
        return false;
    }
    if (!check_font_flags(*opt)) {
        *exit_code = 2;
        return false;
    }
    if (!opt->preset.empty() && rmwork::find_preset(opt->preset) < 0) {
        std::fprintf(stderr, "rmwork: no preset '%s' (replay, play, analysis, table)\n", opt->preset.c_str());
        *exit_code = 2;
        return false;
    }
    return true;
}

// The font flags name a family, a weight and a size the workbench has; false after a message.
static bool check_font_flags(const CliOptions &opt) {
    if (!opt.font.empty() && rmwork::find_font_family(opt.font) < 0) {
        std::fprintf(stderr, "rmwork: no font family '%s'; one of:", opt.font.c_str());
        for (int i = 0; i < rmwork::FONT_FAMILY_COUNT; ++i) std::fprintf(stderr, " %s", rmwork::font_family(i).id);
        std::fprintf(stderr, "\n");
        return false;
    }
    if (!opt.font_weight.empty() && opt.font_weight != "regular" && opt.font_weight != "heavy") {
        std::fprintf(stderr, "rmwork: --font-weight is regular or heavy, not '%s'\n", opt.font_weight.c_str());
        return false;
    }
    if (opt.font_size != 0 && !rmwork::font_size_valid(opt.font_size)) {
        std::fprintf(stderr, "rmwork: --font-size is %d to %d\n", rmwork::FONT_SIZE_MIN, rmwork::FONT_SIZE_MAX);
        return false;
    }
    return true;
}

// `from` with the font flags given applied.
static rmwork::FontChoice font_override(const CliOptions &opt, const rmwork::FontChoice &from) {
    rmwork::FontChoice c = from;
    if (!opt.font.empty()) c.family = opt.font;
    if (!opt.font_weight.empty())
        c.weight = opt.font_weight == "regular" ? rmwork::FontWeight::REGULAR : rmwork::FontWeight::HEAVY;
    if (opt.font_size != 0) c.size = opt.font_size;
    return c;
}

// The error dialog with sample content (--demo-error), for screenshots.
static void show_demo_error(rmwork::App &app) {
    app.report_error("Engine exited",
        "The engine exited with status 1 while replaying game_7.rmlog, after decision 212 of 340. The replay stops "
        "at the last decoded decision. (A demonstration opened by --demo-error: nothing failed.)",
        "$ bin/debug/robomage --replay game_7.rmlog --rmstate-dump game_7.rmst --narrative\n"
        "Player A casts Brainstorm.\n"
        "Brainstorm resolves. Player A draws three cards.\n"
        "Player A puts Force of Will and Island on top of their library.\n"
        "FATAL: --replay: decision 213: the log's reply 7 is outside the menu (5 rows)\n"
        "exit status 1\n");
}

namespace rmwork {

int workbench_main(int argc, char **argv) {
    CliOptions opt;
    int exit_code = 0;
    if (!parse_args(argc, argv, &opt, &exit_code)) return exit_code;
    if (opt.selftest) return run_selftest();

    create_ui_context();
    const ConfigPaths paths = config_paths(opt.config_dir);
    const bool persist = opt.window.screenshot.empty();
    if (persist) {
        make_dirs(parent_dir(paths.imgui_ini));
        ImGui::GetIO().IniFilename = paths.imgui_ini.c_str();
    } else if (file_exists(paths.imgui_ini)) {
        ImGui::LoadIniSettingsFromDisk(paths.imgui_ini.c_str());
    }
    Settings settings;
    std::string err;
    const bool loaded = load_settings(paths.settings, &settings, &err);
    App app(settings, paths, persist);
    if (!loaded)
        app.report_error("Settings not loaded",
            "The settings file could not be read, so the workbench started with the default settings. The file is "
            "left as it is until a setting changes.",
            err);
    if (!opt.mode.empty()) app.set_mode(opt.mode == "play" ? Mode::PLAY : Mode::REPLAY);
    if (!opt.preset.empty()) app.request_preset(find_preset(opt.preset));
    app.set_font(font_override(opt, app.settings().font));
    if (!opt.theme.empty()) app.set_theme(opt.theme);
    if (opt.open_settings) app.open_settings(static_cast<SettingsDialog::Tab>(opt.settings_tab));
    if (opt.demo_error) show_demo_error(app);
    const int status = run_window(app, opt.window);
    destroy_ui_context();
    return status;
}

}  // namespace rmwork
