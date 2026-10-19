#ifndef AGENTS_ACTOR_SCHEDULE_H
#define AGENTS_ACTOR_SCHEDULE_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// ── The actor's schedule: matchups, seats, agents, teachers, purposes (training plan T5) ──────
//
// Standard library and featurize/json.h only. docs/actor.md §2 is the reference. A JSON object:
//
//   version   optional, any JSON value: echoed (as text) in the actor's status and results files;
//             absent, the FNV-1a 64 of the file's bytes (16 hex digits)
//   outputs   optional, per purpose ("train", "eval"): {"rmlog_dir": DIR, "trajectory_dir": DIR or
//             null}; the actor's defaults are <actor out>/<purpose>/rmlog and, for train only,
//             <actor out>/train/traj (eval writes no episodes unless given a trajectory_dir)
//   defaults  optional, an entry's keys (not id) applied under every entry
//   entries   the entries, at least one with a positive weight
//
// An entry (keys ending in _a / _b name its two sides; a side sits in the seat of its letter unless
// `swap` moves it):
//
//   id                     required, unique: names the entry in the status and results files
//   weight                 >= 0 (default 1): the entry's share of the matches
//   purpose                "train" (default) or "eval": which outputs the match writes
//   deck_a, deck_b         deck stems relative to resources/decks/ (league/bug), files that exist
//   format                 "bo3" (default) or "bo1"
//   swap                   "none" (default), "decks" (the decks change seats when the match's coin
//                          says so) or "sides" (everything of side a: deck, agent, teacher,
//                          knowledge level and its letter in trajectory_seats, changes seats with
//                          side b's when the coin says so)
//   agent_a, agent_b       agent specs (agents/agent.h): explore, random, nn:<model>,
//                          nn:remote:<socket>:policies=0,1,2, ...
//   teacher_a, teacher_b   a teacher command (agents/teacher.h) or null (none)
//   teacher_mix            in [0, 1] (default 0): --teacher-mix
//   knowledge_a, knowledge_b  off | expert | expert+belief
//   max_game_decisions     >= 0: the per-game decision cap (--max-game-decisions)
//   trajectory_seats       A | B | AB (default AB): the recorded sides
//   trajectory_omniscient  bool
//   rmlog_dir, trajectory_dir  override the purpose's outputs (trajectory_dir null: no episodes)
//
// An unknown key, a value of the wrong type or out of range, a missing deck file, an entry without
// both decks and both agents (after the defaults and the command line's) is an error naming the
// entry. Relative paths are relative to the engine's working directory (bin/).

namespace agents {

enum class SeatSwap {
    NONE,
    DECKS,
    SIDES
};

// Where a purpose's matches write: their decision logs, and their episodes ("" none).
struct PurposeOutputs {
        bool rmlog_set = false, trajectory_set = false;
        std::string rmlog_dir, trajectory_dir;
};

// One entry, every field resolved against the defaults (by side: [0] = side a).
struct ScheduleEntry {
        std::string id;
        double weight = 1;
        std::string purpose = "train";
        std::string deck[2];
        bool bo3 = true;
        SeatSwap swap = SeatSwap::NONE;
        std::string agent[2];
        std::string teacher[2];
        double teacher_mix = 0;
        int knowledge[2] = {1, 1};  // rmstate::KnowledgeLevel values: 0 off, 1 expert, 2 expert+belief
        int64_t max_game_decisions = -1;
        std::string trajectory_seats = "AB";
        bool trajectory_omniscient = false;
        PurposeOutputs outputs;  // the entry's own rmlog_dir / trajectory_dir
};

struct Schedule {
        std::string version;
        std::map<std::string, PurposeOutputs> outputs;
        std::vector<ScheduleEntry> entries;
        double total_weight = 0;
};

// Parses a schedule file's text; every entry starts from `base` (the command line's values), then
// the file's defaults, then its own keys. `decks_dir` is resources/decks (deck files are checked
// there). False with `err` on any error.
bool parse_schedule(
    const std::string &text, const ScheduleEntry &base, const std::string &decks_dir, Schedule *out, std::string *err);

// SplitMix64's output for state `x` (agents/actor.h actor_game_seed).
uint64_t splitmix64(uint64_t x);

// One match's plan, drawn from its game seed g: the entry with u = (splitmix64(g ^
// 0x6A09E667F3BCC909) >> 11) / 2^53 in its share of the cumulative weights (entries in file order),
// and the swap coin (splitmix64(g ^ 0xBB67AE8584CAA73B) >> 63) == 1 (drawn for every entry, used by
// a swapping one).
struct MatchPlan {
        size_t entry = 0;
        bool swapped = false;          // the coin said swap and the entry swaps
        int side_of_seat[2] = {0, 1};  // [seat] = the side (0 a, 1 b) whose agent, teacher, ... sit there
        int deck_of_seat[2] = {0, 1};  // [seat] = the side whose deck is played there
};
MatchPlan plan_match(const Schedule &schedule, uint64_t game_seed);

}  // namespace agents

#endif /* AGENTS_ACTOR_SCHEDULE_H */
