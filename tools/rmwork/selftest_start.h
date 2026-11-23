#ifndef RMWORK_SELFTEST_START_H
#define RMWORK_SELFTEST_START_H

#include <functional>
#include <string>

#include "app.h"
#include "selftest_table.h"

// ── The start page's checks: --selftest (synthetic files) and --selftest-data (Play's games) ─────
//
// Every file is made under a temporary directory: never the user's games, settings or caches.

namespace rmwork {

// The result lines, the sidecar, and scan_start over a synthetic games directory (a finished best of
// one and best of three, a game that stopped, a malformed log, the running game, one with nothing to
// read, more logs than the page lists), recent replays (one with a cached replay's result, one
// without, recent paths that are not listed), the runs (a recent one, the runs directory's with and
// without status.json, a malformed status.json, a directory that is no run) and the replay cache's
// size; the status blocks with and without an engine, robomage_nn, a working directory, card images
// and a socket. `tmp` is a scratch directory.
void check_start_units(const CheckFn &check, const std::string &tmp);

// In an ImGui context the caller made headless: an App showing the page over a synthetic games
// directory ($XDG_DATA_HOME pointed into `tmp` for the call): the page's window in place of the
// dockspace and its list, the Start tab's ✕ giving the dockspace back, the Start Page key, File →
// Close bringing the page back (or not, with its setting off), the setting's file round trip.
void check_start_app(const CheckFn &check, const ConfigPaths &paths, const std::string &tmp);

// --selftest-data, after Play's bo1 and bo3: the page lists both games of the games directory with
// their seats, opponents, formats, decision counts and results (the bo3's the live match's), and a
// double-click's request on the bo1 opens it in Replay at that log, the page gone; File → Close
// brings the page back with nothing loaded.
void check_start_games(const CheckFn &check, App &app, const std::function<void(int)> &frames);

// --selftest-data, after the engine was killed mid-game: the page lists that game as stopped, and it
// opens in Replay.
void check_start_stopped(const CheckFn &check, App &app, const std::function<void(int)> &frames);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_START_H */
