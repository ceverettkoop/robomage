#ifndef RMWORK_SELFTEST_DATA_H
#define RMWORK_SELFTEST_DATA_H

#include <string>

// ── rmwork --selftest-data: Replay's data, headless, on real replays ─────────────────────────────
//
// Runs the engine (`engine`, else the robomage beside rmwork) in a temporary $XDG_CACHE_HOME,
// $XDG_CONFIG_HOME and $XDG_DATA_HOME, through the App as the UI drives it (headless frames, the
// loader's thread):
//   - corpus logs (tests/corpus/: a bo1 game and a --no-auto-mana one) and a bo3 match the engine
//     plays for the test (explore vs random, --trajectory-out, so its episodes lie beside its log):
//     the decision count is the log's, every decision has A, B and omniscient STATE frames that
//     decode with their viewer's schema and agree on the decision's key, the menu is in the deciding
//     seat's frame and the omniscient one only, the frame cache returns what it decoded;
//   - the bo3 match's games and sideboarding phases, its result, the game list jumping to a game;
//   - the narrative cut at every decision; each seat's log exactly its own frames' events (frame by
//     frame, the terminal frame's at the last decision), the opponent's opening draws unnamed;
//   - the cache: a second open replays nothing, an engine of another identity replays again, a
//     cached dump that no longer decodes reaches the error dialog; the match index kept beside the
//     cached dump (selftest_index.h);
//   - episodes: a --replay --trajectory-out episode opened on its own finds its log and joins every
//     decision (seat and reply), the bo3 match's own episodes join by name, μ where the random seat
//     gave one; a run directory's listing and the log ↔ episode correspondence by name;
//   - navigation (the keys, autoplay, the game list) and viewer switching update the status bar's
//     fields and the Game Log's lines;
//   - a malformed log, a diverging replay (the engine exits 1), a missing engine and a cancelled load
//     reach the error path (or none), leaving no temporary entry;
//   - the file picker's listing and filters, natural order, the recent files;
//   - the Table and the other tools on every match (selftest_table.h, selftest_tools.h), and the
//     evaluations' tools on the bo3 match with made-up evaluations added to its episodes;
//   - Play (selftest_play.h): games against the engine's agents and the scripted agent played to their
//     end, the passing, concessions, the decision logs in Replay, the engine's failures;
//   - Simulate (selftest_simulation.h): matches with no human seat, their results against the engine's
//     replays, a batch with alternating seats, a failure and a cancel, a match opened in Replay.
// Prints "RMWORK-SELFTEST-DATA: N checks, M failures"; nonzero exit on a failure.

namespace rmwork {

int run_selftest_data(const std::string &engine);

// rmwork --selftest-evals: the evaluations' tools on real evaluations. The robomage_nn beside the
// engine plays a bo3 match with the nn: agent of `model` (a TorchScript or .pt2 export, such as the
// gate's reference policy) on both seats, recorded with --trajectory-out, on the CPU; Replay opens it
// with its episodes (every decision joined) and check_tools_values (selftest_tools.h) runs on it:
// every decision evaluated, π a distribution over its menu, the Value Graph's series a point per
// decision of the seat, Swings, seat disagreement and next swing recomputed from the values, the
// Inspector, Choices' π and the Model's Read; then Simulate's matches of the model against itself and
// against the scripted agent (selftest_simulation.h). Same temporary directories as --selftest-data. Prints
// "RMWORK-SELFTEST-EVALS: N checks, M failures"; nonzero exit on a failure or a missing
// robomage_nn or model.
int run_selftest_evals(const std::string &engine, const std::string &model);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_DATA_H */
