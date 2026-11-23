#ifndef RMWORK_EPISODES_H
#define RMWORK_EPISODES_H

#include <string>
#include <vector>

#include "match.h"

// ── Trajectory episodes and their decision logs (docs/trajectory_format.md, docs/actor.md) ──────
//
// An episode is one game of a match: `<stem>.g<n>.rmtraj`, `<stem>` the decision log's file name
// without `.rmlog`. Where they live:
//   - beside the log (`--trajectory-out DIR` writes DIR/s<seed>.rmlog and DIR/s<seed>.g<n>.rmtraj);
//   - a run directory: `<run>/rmlog/<purpose>/<stem>.rmlog` and `<run>/episodes/<purpose>/<stem>.g<n>.rmtraj`;
//   - an actor's own output: `<out>/<purpose>/rmlog/<stem>.rmlog` and `<out>/<purpose>/traj/<stem>.g<n>.rmtraj`.
// An episode names its log in its `rmlog` attribute (relative to the engine's working directory
// unless absolute), which is tried first when an episode is opened on its own.
//
// Joining: each step's (game, sideboarding, decision) key finds the match's decision of that key;
// the step's seat must be the decision's deciding seat and its reply the log's.

namespace rmwork {

// The directories an episode of the log at `log_path` may be in (the log's own first).
std::vector<std::string> episode_dirs_for_log(const std::string &log_path);
// The episode files of the log at `log_path`, by game.
std::vector<std::string> episode_files_for_log(const std::string &log_path);
// The decision log of the episode at `episode_path`: its `rmlog` attribute resolved against
// `engine_workdir` when that file exists, else the corresponding name; "" with `err` when none
// is found.
std::string log_for_episode(const std::string &episode_path, const std::string &engine_workdir, std::string *err);

struct JoinReport {
        size_t steps = 0;   // steps in the episode
        size_t joined = 0;  // joined to a decision with the same seat and reply
        std::vector<std::string> problems;
};

// Reads the episode at `path` and joins its steps to `m`'s decisions (Match::episode_steps,
// DecisionInfo::episode_step); the seats' agents it names go to Match::agents. False with `err` when the file is not an
// episode or is of another match (its engine_seed differs from the log's SEED); a step with no decision of its key, or
// of another seat or reply, is a problem in `report` and is not joined.
bool join_episode(Match *m, const std::string &path, JoinReport *report, std::string *err);

}  // namespace rmwork

#endif /* RMWORK_EPISODES_H */
