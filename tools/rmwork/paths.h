#ifndef RMWORK_PATHS_H
#define RMWORK_PATHS_H

#include <string>

// ── Where the workbench finds the engine, models and caches ─────────────────────────────────────
//
// The repository is found from the rmwork binary's own path (<repo>/bin/<config>/rmwork), else the
// current directory. A path setting left empty means its default; a relative one is relative to
// the repository and "~/" is the home directory.

namespace rmwork {

// The repository's root directory.
std::string repo_root();
// The build configuration the rmwork binary is from ("debug", "release", "pgo"; "debug" when it
// does not run from bin/<config>/).
std::string build_config();
// bin/<config>/robomage_nn when it exists, else bin/<config>/robomage (absolute).
std::string default_engine_binary();
// <repo>/bin: the engine runs from there (it reads resources/ under its working directory).
std::string default_engine_workdir();
// <repo>/runs.
std::string default_run_dir();
// workbench_cache_dir()/rmserve.sock: the socket of the server the workbench starts, when none is set.
std::string default_server_socket();
// <repo>/models: model exports packaged as zip archives (model_zip.h).
std::string default_models_dir();
// $XDG_CACHE_HOME/robomage/scryfall, else ~/.cache/robomage/scryfall.
std::string default_image_cache();
// $XDG_DATA_HOME/robomage/workbench/games, else ~/.local/share/…: Play's games (decision logs).
std::string default_games_dir();
// $XDG_DATA_HOME/robomage/workbench/simulations, else ~/.local/share/…: Simulate's batches (simulation.h).
std::string default_simulations_dir();
// $XDG_CACHE_HOME/robomage/workbench, else ~/.cache/robomage/workbench.
std::string workbench_cache_dir();
// `text` as a path: `fallback` when empty, "~/…" under $HOME, a relative path under repo_root().
std::string resolve_path(const std::string &text, const std::string &fallback);

}  // namespace rmwork

#endif /* RMWORK_PATHS_H */
