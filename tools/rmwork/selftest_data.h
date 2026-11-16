#ifndef RMWORK_SELFTEST_DATA_H
#define RMWORK_SELFTEST_DATA_H

#include <string>

// ── rmwork --selftest-data: Replay's data, headless, on real replays ─────────────────────────────
//
// Runs the engine (`engine`, else the robomage beside rmwork) in a temporary $XDG_CACHE_HOME and
// $XDG_CONFIG_HOME, through the App as the UI drives it (headless frames, the loader's thread):
//   - corpus logs (tests/corpus/: a bo1 game and a --no-auto-mana one) and a bo3 match the engine
//     plays for the test (explore vs random, --trajectory-out, so its episodes lie beside its log):
//     the decision count is the log's, every decision has A, B and omniscient STATE frames that
//     decode with their viewer's schema and agree on the decision's key, the menu is in the deciding
//     seat's frame and the omniscient one only, the frame cache returns what it decoded;
//   - the bo3 match's games and sideboarding phases, its result, the game list jumping to a game;
//   - the narrative cut at every decision; each seat's log exactly its own frames' events (frame by
//     frame, the terminal frame's at the last decision), the opponent's opening draws unnamed;
//   - the cache: a second open replays nothing, an engine of another identity replays again, a
//     cached dump that no longer decodes reaches the error dialog;
//   - episodes: a --replay --trajectory-out episode opened on its own finds its log and joins every
//     decision (seat and reply), the bo3 match's own episodes join by name, μ where the random seat
//     gave one; a run directory's listing and the log ↔ episode correspondence by name;
//   - navigation (the keys, autoplay, the game list) and viewer switching update the status bar's
//     fields and the Game Log's lines;
//   - a malformed log, a diverging replay (the engine exits 1), a missing engine and a cancelled load
//     reach the error path (or none), leaving no temporary entry;
//   - the file picker's listing and filters, natural order, the recent files.
// Prints "RMWORK-SELFTEST-DATA: N checks, M failures"; nonzero exit on a failure.

namespace rmwork {

int run_selftest_data(const std::string &engine);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_DATA_H */
