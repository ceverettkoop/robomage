#ifndef RMWORK_SELFTEST_PLAY_H
#define RMWORK_SELFTEST_PLAY_H

#include <functional>
#include <string>

#include "app.h"
#include "selftest_table.h"

// ── rmwork --selftest-data's Play checks ─────────────────────────────────────────────────────────
//
// Through the App as the UI drives it (headless frames), with the engine `engine` in `workdir` and
// Play's games under $XDG_DATA_HOME (the caller's temporary one):
//   - the engine's arguments for every opponent kind, the engine chosen for one (robomage_nn only for
//     an nn: opponent, its absence named), the New Game dialog's decks and checks, its opening;
//   - a bo1 game as seat A against explore played to its end by auto_play's seeded rule, Model's Read
//     asked for with no server (not connected, the game goes on): every record the session kept is
//     the human seat's (its stream file read again), the other seat's records dropped unread and
//     only after the human's last decision, the stops answering exactly the plain priority passes
//     with an empty stack in steps without a stop, the result and the result screen; its decision
//     log opened in Replay: the log's decision count, the human's decisions in the same order with
//     the same keys, replies and frames (event tables aside: the stream batches the events since the
//     seat's previous frame); File → Save Decision Log's copy;
//   - F2, F6 and the stops (autopass.h) on that game's recorded frames and on made-up facts;
//   - a bo3 match as seat B against random to its end, its sideboarding decisions in Choices' list;
//   - the start page (selftest_start.h) listing those two games, the bo1 opened from it in Replay;
//   - the Sideboard tool (selftest_sideboard.h): its plan, guides and steps, a bo3 sideboarded through
//     it, and that match in Replay;
//   - decisions answered on the Table (selftest_decisions.h): targets and mana payment on sculpted games;
//   - the concession dialog, a game conceded (-2) and the match conceded while the opponent thinks
//     (-3, queued to the next decision), both in the decision log;
//   - the scripted opponent through the teacher channel, when python3 runs;
//   - the engine killed mid-game, a stream that does not decode and a record that is not a frame:
//     the error dialog, the session ended; the killed game on the start page as stopped, opening in
//     Replay.

namespace rmwork {

struct PlayTestEnv {
        std::string engine;   // the engine binary (robomage)
        std::string workdir;  // its working directory
        std::string tmp;      // a scratch directory
};

void check_play(const CheckFn &check, App &app, const std::function<void(int)> &frames, const PlayTestEnv &env);

}  // namespace rmwork

#endif /* RMWORK_SELFTEST_PLAY_H */
