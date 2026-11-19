#include "cli.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "app.h"
#include "files.h"
#include "platform.h"
#include "selftest.h"
#include "selftest_data.h"
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
        std::string card_faces;      // "images" / "text" ("" the setting's)
        std::string printings_card;  // Settings → Card Printings: the card shown
        bool demo_error = false;
        bool save_layout = false;
        bool manage_layouts = false;
        std::string layout;       // a saved layout to show at start
        std::string capture_key;  // an action whose key Settings → Keys waits for
        std::string mode;
        std::string preset;
        std::string config_dir;
        // Replay's data.
        std::string open_log;
        std::vector<std::string> episodes;
        std::string run_dir;
        int decision = 0;  // 1-based; 0 = the first
        int view = -1;
        std::string engine;
        bool selftest_data = false;
        bool open_picker = false;
        bool game_combo = false;
        int table_hover_stack = -1;
        int table_hover_step = -1;
        std::string table_magnify;
        std::vector<std::string> focus_tools;
        int choices_hover = -1;
        bool tree_expand = false;
        std::vector<std::pair<int, rmwork::ZoneKind>> zones;
        std::string zone_magnify;
        // Play.
        std::string new_game;  // the --new-game spec ("" none)
        long play_decisions = 0;
        std::string play_until;
        std::vector<std::string> play_setup;  // --play-setup's engine flags
        std::vector<int> play_rows;
        std::string table_pick;
        std::string table_plan;
        std::string sideboard_plan;
        bool open_new_game = false;
        bool game_over = false;
        bool start_page = false;  // --start-page
        rmwork::WindowOptions window;
};

}  // namespace

static void usage();
static bool parse_size(const std::string &text, int *w, int *h);
static bool parse_args(int argc, char **argv, CliOptions *opt, int *exit_code);
static bool check_font_flags(const CliOptions &opt);
static rmwork::FontChoice font_override(const CliOptions &opt, const rmwork::FontChoice &from);
static void show_demo_error(rmwork::App &app);
static int parse_view(const std::string &text);
static bool parse_zone(const std::string &text, std::pair<int, rmwork::ZoneKind> *out);
static bool parse_new_game(const std::string &spec, rmwork::NewGameSettings *g, std::string *why);
static std::vector<std::string> split_list(const std::string &text, char sep);
static void open_data(rmwork::App &app, const CliOptions &opt);
static void open_play(rmwork::App &app, const CliOptions &opt);
static bool start_page_at_start(const CliOptions &opt, const rmwork::Settings &s);
static int run_app(const CliOptions &opt, const rmwork::ConfigPaths &paths, bool persist);

// `text`'s items between `sep`s (empty items dropped).
static std::vector<std::string> split_list(const std::string &text, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t end = text.find(sep, start);
        const std::string item = text.substr(start, end == std::string::npos ? std::string::npos : end - start);
        if (!item.empty()) out.push_back(item);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return out;
}

static void usage() {
    std::printf(
        "usage: rmwork [--mode play|replay] [--preset NAME] [--config-dir DIR] [--size WxH]\n"
        "       rmwork --screenshot FILE.png [--mode …] [--preset …] [--size WxH] [--config-dir DIR] "
        "[--open-settings] [--settings-tab TAB] [--capture-key ACTION] [--save-layout] [--manage-layouts]\n"
        "         [--layout NAME] [--theme NAME] [--font FAMILY] [--font-weight W] [--font-size N]\n"
        "         [--card-faces images|text] [--printings-card NAME]\n"
        "       rmwork [--open LOG.rmlog] [--episode FILE.rmtraj]… [--run DIR] [--decision N] [--view a|b|omni]\n"
        "         [--engine PATH] [--open-picker] [--game-combo] [--table-hover-stack N] [--table-magnify NAME]\n"
        "         [--tool NAME]… [--choices-hover N] [--tree-expand] [--open-zone SEAT:ZONE]… [--zone-magnify NAME]\n"
        "         (with --screenshot too)\n"
        "       rmwork [--new-game SPEC [--play-decisions N]\n"
        "              [--play-until stack|combat|attack|block|damage|sideboard|target|mana|end]\n"
        "              [--play-setup FLAGS] [--play-rows R,R,...] [--table-pick NAME] [--table-plan SPEC]\n"
        "              [--game-over]]\n"
        "         [--sideboard-plan SPEC]\n"
        "         [--open-new-game] [--table-hover-step N] [--engine PATH] (with --screenshot too)\n"
        "       rmwork [--start-page] (with --screenshot too)\n"
        "       rmwork --selftest\n"
        "       rmwork --selftest-data [--engine PATH]\n"
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
        "  --settings-tab TAB open Settings at a tab: table, printings, keys, appearance, engine, models, stops\n"
        "  --card-faces F     draw cards as images or text, as Settings → Card Printings sets it\n"
        "  --printings-card NAME  open Settings → Card Printings showing the first card whose name contains NAME\n"
        "  --capture-key ACTION  open Settings → Keys waiting for ACTION's new key (a settings key such as\n"
        "                     replay.next)\n"
        "  --save-layout      open View → Save Layout As… at start\n"
        "  --manage-layouts   open View → Manage Layouts… at start\n"
        "  --layout NAME      show the starting mode's saved layout NAME\n"
        "  --theme NAME       the colour theme: imhex-dark, imhex-light, imgui-dark, imgui-classic,\n"
        "                     solarized-dark, solarized-light, nord, windows-95\n"
        "  --font FAMILY      the UI font: jetbrains-mono, fira-code, ibm-plex-mono, source-code-pro,\n"
        "                     cascadia-mono, hack, inter, ibm-plex-sans (as Settings → Appearance sets it)\n"
        "  --font-weight W    regular or heavy (the family's Medium, SemiBold or Bold)\n"
        "  --font-size N      the UI font's size in pixels, 11 to 20\n"
        "  --open LOG         open a decision log in Replay (replayed by the engine, cached under\n"
        "                     $XDG_CACHE_HOME/robomage/workbench/replays)\n"
        "  --episode FILE     open a trajectory episode (with its decision log); repeatable\n"
        "  --run DIR          list a run directory's decision logs\n"
        "  --decision N       show decision N (1-based) of the opened log\n"
        "  --view V           Replay's viewer: a, b or omni\n"
        "  --engine PATH      replay and play with this engine binary instead of Settings → Engine's (an nn:\n"
        "                     opponent plays in the robomage_nn beside it)\n"
        "  --open-picker      open File → Open Replay's file picker at start\n"
        "  --game-combo       open the toolbar's game list once the game is loaded\n"
        "  --table-hover-stack N  the Table: hover stack object N (0 = top) every frame (its targets lit)\n"
        "  --table-magnify NAME   the Table: open the magnifier on the first card whose name contains NAME\n"
        "  --tool NAME        bring a tool's tab to the front (table, players, choices, inspector, tree, log,\n"
        "                     swings, graph, model, sideboard); repeatable\n"
        "  --choices-hover N  Choices: hover menu row N (0-based) every frame (its objects lit on the Table)\n"
        "  --tree-expand      the State Tree: expand the players' zones, the stack, the frame's tables and its\n"
        "                     object table\n"
        "  --open-zone SEAT:ZONE  open a zone window: a or b, then hand, library, graveyard or exile; repeatable\n"
        "  --zone-magnify NAME    the zone windows: open the magnifier on the first card whose name contains NAME\n"
        "  --new-game SPEC    start a Play game: the New Game dialog's last choices changed by comma-separated items\n"
        "                     opp=explore|random|scripted|model|server, model=EXPORT, server=SOCKET, deck=DECK,\n"
        "                     opp-deck=DECK, seat=a|b, format=bo1|bo3, seed=N, auto-mana=on|off, read=on|off\n"
        "                     (decks relative to the engine's resources/decks, e.g. league/burn)\n"
        "  --play-decisions N answer your first N decisions with a seeded random row (the game's seed)\n"
        "  --table-hover-step N  Play's Table: hover step N (0 = untap) of the step bar every frame (its stop's\n"
        "                     tooltip)\n"
        "  --play-until COND  ... until a decision where COND holds: stack (the stack is not empty), combat (an\n"
        "                     attacker or blocker declaration), attack (an attacker declaration), block (a\n"
        "                     blocker declaration), damage (combat damage to assign among blockers),\n"
        "                     sideboard (a sideboarding decision), target (a\n"
        "                     target or a point of divided damage to choose), mana (a cost to pay with a mana\n"
        "                     ability: auto-mana=off), end (none: to the match's end)\n"
        "  --play-setup FLAGS the engine's test-setup flags for the --new-game game, ';'-separated (e.g.\n"
        "                     --hand-a;Lightning Bolt;--battlefield-b;Grizzly Bears,Delver of Secrets)\n"
        "  --play-rows R,R,...  answer your first decisions with these menu rows, before --play-decisions or\n"
        "                     --play-until's rule\n"
        "  --table-plan SPEC  Play's Table, at your attacker or blocker declaration: plan it, comma-separated\n"
        "                     NAME (attacks the player / a blocker armed), NAME>TARGET (attacks the\n"
        "                     planeswalker TARGET or \"player\" / blocks the attacker TARGET) or NAME>? (planned\n"
        "                     and armed); NAME and TARGET name substrings\n"
        "  --table-pick NAME  Play's Table: open the popup of the first card whose name contains NAME that\n"
        "                     the decision's rows name more than once (a dual land's colours)\n"
        "  --sideboard-plan SPEC  at your first sideboarding decision, plan in the Sideboard tool: guide (the\n"
        "                     league guide's plan) or ';'-separated +CARD (bring in) and -CARD (cut) items\n"
        "  --game-over        play the --new-game to its end with that rule: the match result screen opens\n"
        "  --open-new-game    open Game \xE2\x86\x92 New Game vs Model at start\n"
        "  --start-page       show the start page at start, whatever its setting (it shows by default when no\n"
        "                     flag opens a game, replay or run or names a mode, preset, layout or tool)\n"
        "  --selftest         headless checks of the presets, modes and settings file; prints\n"
        "                     RMWORK-SELFTEST: N checks, M failures\n"
        "  --selftest-data    headless checks of Replay's data on corpus logs, replayed by the engine\n"
        "                     beside rmwork (or --engine), and of Play's games against it; prints\n"
        "                     RMWORK-SELFTEST-DATA: N checks, M failures\n");
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
        } else if (a == "--selftest-data") {
            opt->selftest_data = true;
        } else if (a == "--open" && has_value) {
            opt->open_log = argv[++i];
        } else if (a == "--episode" && has_value) {
            opt->episodes.push_back(argv[++i]);
        } else if (a == "--run" && has_value) {
            opt->run_dir = argv[++i];
        } else if (a == "--decision" && has_value) {
            opt->decision = std::atoi(argv[++i]);
            if (opt->decision < 1) {
                std::fprintf(stderr, "rmwork: --decision is a decision number from 1\n");
                *exit_code = 2;
                return false;
            }
        } else if (a == "--view" && has_value) {
            opt->view = parse_view(argv[++i]);
            if (opt->view < 0) {
                std::fprintf(stderr, "rmwork: --view is a, b or omni, not '%s'\n", argv[i]);
                *exit_code = 2;
                return false;
            }
        } else if (a == "--engine" && has_value) {
            opt->engine = argv[++i];
        } else if (a == "--open-picker") {
            opt->open_picker = true;
        } else if (a == "--game-combo") {
            opt->game_combo = true;
        } else if (a == "--table-hover-stack" && has_value) {
            opt->table_hover_stack = std::atoi(argv[++i]);
        } else if (a == "--table-hover-step" && has_value) {
            opt->table_hover_step = std::atoi(argv[++i]);
        } else if (a == "--table-magnify" && has_value) {
            opt->table_magnify = argv[++i];
        } else if (a == "--tool" && has_value) {
            opt->focus_tools.push_back(argv[++i]);
            if (rmwork::find_tool(opt->focus_tools.back()) < 0) {
                std::fprintf(stderr, "rmwork: no tool '%s'\n", argv[i]);
                *exit_code = 2;
                return false;
            }
        } else if (a == "--choices-hover" && has_value) {
            opt->choices_hover = std::atoi(argv[++i]);
        } else if (a == "--tree-expand") {
            opt->tree_expand = true;
        } else if (a == "--open-zone" && has_value) {
            std::pair<int, rmwork::ZoneKind> z;
            if (!parse_zone(argv[++i], &z)) {
                std::fprintf(stderr,
                    "rmwork: --open-zone is SEAT:ZONE (a or b; hand, library, graveyard, exile), not '%s'\n", argv[i]);
                *exit_code = 2;
                return false;
            }
            opt->zones.push_back(z);
        } else if (a == "--zone-magnify" && has_value) {
            opt->zone_magnify = argv[++i];
        } else if (a == "--new-game" && has_value) {
            opt->new_game = argv[++i];
        } else if (a == "--play-decisions" && has_value) {
            opt->play_decisions = std::atol(argv[++i]);
            if (opt->play_decisions < 1) {
                std::fprintf(stderr, "rmwork: --play-decisions is a number of decisions from 1\n");
                *exit_code = 2;
                return false;
            }
        } else if (a == "--play-until" && has_value) {
            opt->play_until = argv[++i];
            static const char *const UNTIL[] = {
                "stack", "combat", "attack", "block", "damage", "sideboard", "target", "mana", "end"};
            bool known = false;
            for (const char *u : UNTIL) known = known || opt->play_until == u;
            if (!known) {
                std::fprintf(stderr,
                    "rmwork: --play-until is stack, combat, attack, block, damage, sideboard, target, mana or end, "
                    "not '%s'\n",
                    argv[i]);
                *exit_code = 2;
                return false;
            }
        } else if (a == "--play-setup" && has_value) {
            opt->play_setup = split_list(argv[++i], ';');
        } else if (a == "--play-rows" && has_value) {
            for (const std::string &r : split_list(argv[++i], ',')) opt->play_rows.push_back(std::atoi(r.c_str()));
        } else if (a == "--table-pick" && has_value) {
            opt->table_pick = argv[++i];
        } else if (a == "--table-plan" && has_value) {
            opt->table_plan = argv[++i];
        } else if (a == "--sideboard-plan" && has_value) {
            opt->sideboard_plan = argv[++i];
        } else if (a == "--game-over") {
            opt->game_over = true;
        } else if (a == "--open-new-game") {
            opt->open_new_game = true;
        } else if (a == "--start-page") {
            opt->start_page = true;
        } else if (a == "--open-settings") {
            opt->open_settings = true;
        } else if (a == "--settings-tab" && has_value) {
            opt->open_settings = true;
            opt->settings_tab = rmwork::SettingsDialog::find_tab(argv[++i]);
            if (opt->settings_tab < 0) {
                std::fprintf(stderr,
                    "rmwork: no settings tab '%s' (table, printings, keys, appearance, engine, models, stops)\n",
                    argv[i]);
                *exit_code = 2;
                return false;
            }
        } else if (a == "--card-faces" && has_value) {
            opt->card_faces = argv[++i];
            if (opt->card_faces != "images" && opt->card_faces != "text") {
                std::fprintf(stderr, "rmwork: --card-faces takes images or text, not '%s'\n", argv[i]);
                *exit_code = 2;
                return false;
            }
        } else if (a == "--printings-card" && has_value) {
            opt->printings_card = argv[++i];
            opt->open_settings = true;
            opt->settings_tab = rmwork::SettingsDialog::PRINTINGS;
        } else if (a == "--capture-key" && has_value) {
            opt->capture_key = argv[++i];
            if (rmwork::find_action(opt->capture_key) < 0) {
                std::fprintf(stderr, "rmwork: no action '%s' (Settings → Keys lists them)\n", argv[i]);
                *exit_code = 2;
                return false;
            }
        } else if (a == "--save-layout") {
            opt->save_layout = true;
        } else if (a == "--manage-layouts") {
            opt->manage_layouts = true;
        } else if (a == "--layout" && has_value) {
            opt->layout = argv[++i];
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
    if ((opt->play_decisions || !opt->play_until.empty() || opt->game_over || !opt->play_setup.empty() ||
            !opt->play_rows.empty()) &&
        opt->new_game.empty()) {
        std::fprintf(stderr,
            "rmwork: --play-decisions, --play-until, --play-setup, --play-rows and --game-over need --new-game\n");
        *exit_code = 2;
        return false;
    }
    if (!opt->new_game.empty()) {
        rmwork::NewGameSettings g;
        std::string why;
        if (!parse_new_game(opt->new_game, &g, &why)) {
            std::fprintf(stderr, "rmwork: --new-game: %s\n", why.c_str());
            *exit_code = 2;
            return false;
        }
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

// "a", "b", "omni" (or "o", "omniscient") -> the viewer, else -1.
static int parse_view(const std::string &text) {
    if (text == "a" || text == "A") return rmwork::VIEW_A;
    if (text == "b" || text == "B") return rmwork::VIEW_B;
    if (text == "o" || text == "O" || text == "omni" || text == "omniscient") return rmwork::VIEW_OMNI;
    return -1;
}

// "a:graveyard" -> (0, GRAVEYARD); false for anything else.
static bool parse_zone(const std::string &text, std::pair<int, rmwork::ZoneKind> *out) {
    const size_t colon = text.find(':');
    if (colon == std::string::npos) return false;
    const int view = parse_view(text.substr(0, colon));
    if (view != rmwork::VIEW_A && view != rmwork::VIEW_B) return false;
    const std::string zone = text.substr(colon + 1);
    for (int z = 0; z < rmwork::ZONE_KINDS; ++z)
        if (zone == rmwork::zone_key(static_cast<rmwork::ZoneKind>(z))) {
            *out = {view, static_cast<rmwork::ZoneKind>(z)};
            return true;
        }
    return false;
}

// "opp=explore,deck=league/burn,…" over `g` (the New Game dialog's keys); false with `why`.
static bool parse_new_game(const std::string &spec, rmwork::NewGameSettings *g, std::string *why) {
    size_t start = 0;
    while (start < spec.size()) {
        size_t comma = spec.find(',', start);
        if (comma == std::string::npos) comma = spec.size();
        const std::string item = spec.substr(start, comma - start);
        start = comma + 1;
        if (item.empty()) continue;
        const size_t eq = item.find('=');
        const std::string key = item.substr(0, eq), value = eq == std::string::npos ? "" : item.substr(eq + 1);
        const bool on = value == "on" || value == "true" || value == "1";
        const bool off = value == "off" || value == "false" || value == "0";
        if (key == "opp" && rmwork::find_opponent(value) >= 0) {
            g->opponent = value;
        } else if (key == "model" && !value.empty()) {
            g->model = rmwork::absolute_path(value);
        } else if (key == "server" && !value.empty()) {
            g->server = rmwork::absolute_path(value);
        } else if (key == "deck" && !value.empty()) {
            g->deck = value;
        } else if (key == "opp-deck" && !value.empty()) {
            g->opponent_deck = value;
        } else if (key == "seat" && (value == "a" || value == "b")) {
            g->seat = value == "a" ? 0 : 1;
        } else if (key == "format" && (value == "bo1" || value == "bo3")) {
            g->bo3 = value == "bo3";
        } else if (key == "seed" && !value.empty() && value.find_first_not_of("0123456789") == std::string::npos) {
            g->seed = std::strtoull(value.c_str(), nullptr, 10);
        } else if (key == "auto-mana" && (on || off)) {
            g->auto_mana = on;
        } else if (key == "read" && (on || off)) {
            g->model_read = on;
        } else {
            *why = "bad item '" + item + "'";
            return false;
        }
    }
    return true;
}

// Play's flags: the dialog, or a game started at once and answered by the seeded rule.
static void open_play(rmwork::App &app, const CliOptions &opt) {
    if (opt.open_new_game) app.open_new_game();
    if (opt.new_game.empty()) return;
    rmwork::NewGameSettings g = app.settings().new_game;
    std::string why;
    if (!parse_new_game(opt.new_game, &g, &why) || !app.start_game(g, opt.play_setup)) return;
    if (!opt.sideboard_plan.empty()) app.set_sideboard_spec(opt.sideboard_plan);
    app.script_rows(opt.play_rows);
    if (opt.game_over) {
        app.auto_play(-1, "end");
    } else if (opt.play_decisions || !opt.play_until.empty()) {
        app.auto_play(opt.play_decisions ? opt.play_decisions : -1, opt.play_until);
    }
}

// The data flags: a log (with the episodes), else an episode, a run directory, the picker.
static void open_data(rmwork::App &app, const CliOptions &opt) {
    if (!opt.engine.empty()) app.set_engine_override(opt.engine);
    app.set_load_target(opt.decision - 1, opt.view);
    if (!opt.open_log.empty()) {
        app.open_replay(opt.open_log, opt.episodes);
    } else if (!opt.episodes.empty()) {
        app.open_episode(opt.episodes.front());
    } else if (opt.view >= 0) {
        app.replay().set_view(opt.view);
    }
    if (!opt.run_dir.empty()) app.open_run(opt.run_dir);
    if (opt.open_picker) app.open_file_picker(rmwork::App::PICK_REPLAY);
    if (opt.game_combo) app.open_game_combo();
    app.table().force_stack_hover(opt.table_hover_stack);
    app.table().force_step_hover(opt.table_hover_step);
    app.table().force_magnify(opt.table_magnify);
    app.table().force_pick(opt.table_pick);
    app.table().force_plan(opt.table_plan);
    for (const std::string &t : opt.focus_tools) app.focus_tool(static_cast<rmwork::Tool>(rmwork::find_tool(t)));
    app.tools().choices().force_hover(opt.choices_hover);
    if (opt.tree_expand) app.tools().tree().expand_overview();
    for (const auto &z : opt.zones) app.tools().players().open_zone(z.first, z.second);
    app.tools().players().force_magnify(opt.zone_magnify);
}

// The start page shows at start when asked for, else when its setting is on and no flag opens a game,
// replay or run, or names a mode, preset, layout or tool to look at.
static bool start_page_at_start(const CliOptions &opt, const rmwork::Settings &s) {
    if (opt.start_page) return true;
    const bool names = !opt.mode.empty() || !opt.preset.empty() || !opt.layout.empty() || opt.save_layout ||
                       opt.manage_layouts || !opt.focus_tools.empty() || !opt.open_log.empty() ||
                       !opt.episodes.empty() || !opt.run_dir.empty() || !opt.new_game.empty();
    return s.start_page && !names;
}

// The workbench's window until it closes; the App is gone before the ImGui context is destroyed
// (its textures unregister themselves).
static int run_app(const CliOptions &opt, const rmwork::ConfigPaths &paths, bool persist) {
    using namespace rmwork;
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
    if (!opt.card_faces.empty()) app.set_card_faces(opt.card_faces == "images");
    if (!opt.printings_card.empty()) app.settings_dialog().printings().select(opt.printings_card);
    if (opt.demo_error) show_demo_error(app);
    if (!opt.capture_key.empty()) app.begin_key_capture(static_cast<Action>(find_action(opt.capture_key)));
    if (opt.save_layout) app.layout_dialogs().open_save(app.mode());
    if (opt.manage_layouts) app.layout_dialogs().open_manage(app.mode());
    open_data(app, opt);
    open_play(app, opt);
    if (start_page_at_start(opt, app.settings())) app.show_start_page(true);
    if (!opt.layout.empty() && !app.request_named_layout(opt.layout))
        app.report_error("No such layout",
            "The " + std::string(mode_title(app.mode())) + " mode has no saved layout named \"" + opt.layout +
                "\". View lists the saved layouts.",
            "");
    return run_window(app, opt.window);
}

namespace rmwork {

int workbench_main(int argc, char **argv) {
    CliOptions opt;
    int exit_code = 0;
    if (!parse_args(argc, argv, &opt, &exit_code)) return exit_code;
    if (opt.selftest) return run_selftest();
    if (opt.selftest_data) return run_selftest_data(opt.engine);

    create_ui_context();
    const ConfigPaths paths = config_paths(opt.config_dir);
    const bool persist = opt.window.screenshot.empty();
    if (persist) {
        make_dirs(parent_dir(paths.imgui_ini));
        ImGui::GetIO().IniFilename = paths.imgui_ini.c_str();
    } else if (file_exists(paths.imgui_ini)) {
        ImGui::LoadIniSettingsFromDisk(paths.imgui_ini.c_str());
    }
    const int status = run_app(opt, paths, persist);
    destroy_ui_context();
    return status;
}

}  // namespace rmwork
