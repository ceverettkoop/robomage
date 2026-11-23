#include "paths.h"

#include <cstdlib>

#include "files.h"

namespace {

// Where the rmwork binary says it lives.
struct Install {
        std::string repo;
        std::string config;
};

}  // namespace

static const Install &install();
static std::string base_name(const std::string &path);
static std::string home_dir();

// <repo>/bin/<config>/rmwork, else the current directory and "debug"; worked out once.
static const Install &install() {
    static const Install found = [] {
        Install in;
        const std::string config_dir = rmwork::parent_dir(rmwork::executable_path());
        const std::string bin_dir = rmwork::parent_dir(config_dir);
        if (!config_dir.empty() && base_name(bin_dir) == "bin") {
            in.repo = rmwork::parent_dir(bin_dir);
            in.config = base_name(config_dir);
        } else {
            in.repo = rmwork::current_dir();
            in.config = "debug";
        }
        return in;
    }();
    return found;
}

static std::string base_name(const std::string &path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

static std::string home_dir() {
    const char *home = std::getenv("HOME");
    return home && *home ? home : ".";
}

namespace rmwork {

std::string repo_root() {
    return install().repo;
}

std::string build_config() {
    return install().config;
}

std::string default_engine_binary() {
    const std::string dir = repo_root() + "/bin/" + build_config() + "/";
    return is_executable(dir + "robomage_nn") ? dir + "robomage_nn" : dir + "robomage";
}

std::string default_engine_workdir() {
    return repo_root() + "/bin";
}

std::string default_run_dir() {
    return repo_root() + "/runs";
}

std::string default_server_socket() {
    return workbench_cache_dir() + "/rmserve.sock";
}

std::string default_models_dir() {
    return repo_root() + "/models";
}

std::string default_image_cache() {
    const char *xdg = std::getenv("XDG_CACHE_HOME");
    return (xdg && *xdg ? std::string(xdg) : home_dir() + "/.cache") + "/robomage/scryfall";
}

std::string default_games_dir() {
    const char *xdg = std::getenv("XDG_DATA_HOME");
    return (xdg && *xdg ? std::string(xdg) : home_dir() + "/.local/share") + "/robomage/workbench/games";
}

std::string default_simulations_dir() {
    return parent_dir(default_games_dir()) + "/simulations";
}

std::string workbench_cache_dir() {
    const char *xdg = std::getenv("XDG_CACHE_HOME");
    return (xdg && *xdg ? std::string(xdg) : home_dir() + "/.cache") + "/robomage/workbench";
}

std::string resolve_path(const std::string &text, const std::string &fallback) {
    if (text.empty()) return fallback;
    if (text == "~") return home_dir();
    if (text.compare(0, 2, "~/") == 0) return home_dir() + text.substr(1);
    if (text[0] == '/') return text;
    return repo_root() + "/" + text;
}

}  // namespace rmwork
