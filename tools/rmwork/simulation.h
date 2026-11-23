#ifndef RMWORK_SIMULATION_H
#define RMWORK_SIMULATION_H

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "play_session.h"
#include "settings.h"
#include "start_data.h"
#include "subprocess.h"

// ── Simulate: matches with no human seat (docs/workbench_plan.md §2.9) ──────────────────────────
//
// A simulation is a batch of N matches between two entrants, each an agent (play_session.h's kinds:
// a model export, a served policy, the scripted agent through the teacher channel at mix 1, explore,
// random) with a deck. Match i (from 0) plays `--game-seed` seed + i; with `alternate` the odd matches
// swap the entrants' seats, so entrant 0 plays seat B there. Each match is one headless engine run
// (robomage_nn when an entrant is a model or a served policy, else the plain robomage; play_session.h's
// play_engine_binary) in the engine's working directory, its own process group (the teachers it starts
// die with it), stdin from /dev/null, its decision log and output in the simulation's directory (the
// `.stderr` file holds stdout and stderr together: with no seat on a stream the engine writes its
// result lines to stdout, a FATAL to stderr):
//
//   <root>/<timestamp>-<agent>-<deck>-vs-<agent>-<deck>/   (root: default_simulations_dir())
//     m001.rmlog, m001.stderr, m002.rmlog, …
//     simulation.json   the configuration, the engine and per match its seed, seats, state, result,
//                       games, decisions, time, log and stderr (relative to the directory) and error
//
// The Simulation runs on the UI thread: poll() (once a frame) reaps the engines that exited, judges
// each from its exit status and stderr (a FATAL line, or no GAME_RESULT / MATCH_RESULT line, is a
// failure of that match alone, never of the batch), counts its log's decisions and starts waiting
// matches while fewer than `jobs` run; simulation.json is rewritten (atomically) whenever a match's
// state changes. cancel() kills the running engines (their partial logs are removed; finished logs
// stay) and marks every unfinished match cancelled. open() reads a past simulation's simulation.json;
// a match it lists as running or waiting (the workbench quit meanwhile) reads as cancelled. Replay
// opens a match's log with the plain engine: its decisions are all in the log.

namespace rmwork {

// One side of a simulation: an agent and its deck.
struct SimEntrant {
        SeatAgent agent;  // MODEL: the export as chosen (a zipped one's `<archive>.zip!<member>`); SERVER: the socket
        std::string deck;  // relative to resources/decks (no .dk)
};

struct SimConfig {
        std::array<SimEntrant, 2> entrants;
        bool bo3 = false;
        bool auto_mana = true;
        uint64_t seed = 1;
        int matches = 1;
        bool alternate = false;
        int jobs = 1;
};

// The configuration of the Simulate dialog's choices (model paths resolved as Settings' are; jobs 0
// the default); false with `why` for an unknown agent kind.
bool sim_config_from(const SimulateSettings &s, SimConfig *out, std::string *why);
// The nn: options of a seat's choices (":argmax", ":temp=0.5", "" for sampling at temperature 1).
std::string nn_options(bool argmax, double temperature);
// Engines at once by default: a quarter of the hardware threads, at least 1.
int default_simulation_jobs();
// An agent as the results name it: "explore", "scripted (hard)", "nn: policy.ts.pt", "nn: policy.ts.pt argmax",
// "nn:remote: s.sock".
std::string sim_agent_text(const SeatAgent &a);
// "<agent> · <deck>".
std::string sim_entrant_text(const SimEntrant &e);

enum class SimState {
    WAITING,
    RUNNING,
    FINISHED,
    FAILED,
    CANCELLED
};
// "waiting", "running", "finished", "failed", "cancelled".
const char *sim_state_key(SimState s);

struct SimMatch {
        int index = 0;  // from 0
        uint64_t seed = 0;
        bool swapped = false;  // entrant 1 plays seat A
        SimState state = SimState::WAITING;
        std::string log;          // the decision log ("" once a cancelled match's partial log is removed)
        std::string stderr_path;  // the engine's output (stdout and stderr)
        LogResult result;         // its result lines
        int winner_seat = -1;     // 0 A, 1 B, -1 a draw or no result
        size_t decisions = 0;     // the log's decisions
        double seconds = 0;       // the engine's run time
        std::string error;        // FAILED: why (the FATAL line, the exit status, the missing result)
        // The entrant on seat `seat` (0 A, 1 B).
        int entrant(int seat) const { return swapped ? 1 - seat : seat; }
        // The winning entrant, -1 a draw or no result.
        int winner() const { return winner_seat < 0 ? -1 : entrant(winner_seat); }
        // The games played (its GAME_RESULT lines).
        int games() const { return static_cast<int>(result.games.size()); }
};

// The matches' counts and results.
struct SimTally {
        int waiting = 0, running = 0, finished = 0, failed = 0, cancelled = 0;
        int wins[2] = {0, 0};  // finished matches won, by entrant
        int draws = 0;         // finished matches without a winner
        int game_wins[2] = {0, 0}, game_draws = 0;  // games, by entrant
        int seat_wins[2] = {0, 0};                  // finished matches won, by seat (A, B)
        // Entrant 0's share of the finished matches (a draw counts half) and its 95% Wilson score
        // interval; -1 with no finished match.
        double share = -1, low = -1, high = -1;
};
SimTally tally_simulation(const std::vector<SimMatch> &matches);
// The match's result as the results show it: "A wins", "B wins 2-1", "draw", "failed", "running", ….
std::string sim_result_text(const SimMatch &m);
// What a match's engine run left: exit status (`exited_ok`, `status` as status_text names it) and its
// stderr judged: the result lines, the winning seat, a failure's error.
void judge_match(SimMatch *m, bool exited_ok, const std::string &status, const std::string &stderr_text, bool bo3);
// The engine's arguments (after its path) for match `m` of `c`, its log at `log`; `models` the files
// the entrants' MODEL agents load (a zipped export extracted).
std::vector<std::string> simulation_arguments(const SimConfig &c, const SimMatch &m, const std::string &log,
    const std::array<std::string, 2> &models, const std::string &repo, const std::string &python);
// "<agent kind>-<deck>-vs-<agent kind>-<deck>", file-name safe.
std::string simulation_description(const SimConfig &c);

// Where a simulation runs.
struct SimPaths {
        std::string engine;   // the configured engine binary (play_engine_binary chooses robomage or robomage_nn)
        std::string workdir;  // its working directory (bin/)
        std::string root;     // the simulations directory
        std::string repo;     // the repository (python/teacher.py)
        std::string python = "python3";
};

// simulation.json and its reader (paths in it relative to `dir`).
std::string simulation_json(const SimConfig &c, const std::vector<SimMatch> &matches, const std::string &engine,
    const std::string &state, const std::string &dir);
bool parse_simulation(const std::string &text, const std::string &dir, SimConfig *c, std::vector<SimMatch> *matches,
    std::string *engine, std::string *state, std::string *err);

// A past simulation, as File → Recent Simulations lists it.
struct SimulationEntry {
        std::string dir;
        std::string label;  // "11-22 15:02 · explore · league/burn vs random · league/bug · 10 bo1"
        int matches = 0, finished = 0;
};
// The simulations under `root` with a simulation.json, newest first, at most `max`.
std::vector<SimulationEntry> list_simulations(const std::string &root, size_t max);

class Simulation {
    public:
        enum class State {
            NONE,      // nothing started or opened
            RUNNING,   // engines run or wait
            DONE,      // every match ended (some may have failed)
            CANCELLED  // cancelled, or opened unfinished
        };

        Simulation() = default;
        Simulation(const Simulation &) = delete;
        Simulation &operator=(const Simulation &) = delete;
        ~Simulation();

        // Starts a batch (a running one is cancelled first): the directory made, the models extracted,
        // the engine chosen; false with `err` when none of that can be done. Its first engines start
        // here.
        bool start(const SimConfig &c, const SimPaths &p, std::string *err);
        // The UI thread, once a frame: the engines that exited judged, waiting matches started.
        void poll();
        // Kills the running engines; every unfinished match cancelled.
        void cancel();
        // A past simulation's directory; false with `err` (no simulation.json, malformed).
        bool open(const std::string &dir, std::string *err);
        // Forgets it (the files stay); a running batch is cancelled first.
        void close();
        // Taken once: the batch ended (every match finished or failed; not a cancel).
        bool take_finished();

        State state() const { return state_; }
        bool running() const { return state_ == State::RUNNING; }
        bool loaded() const { return state_ != State::NONE; }
        const SimConfig &config() const { return config_; }
        const std::vector<SimMatch> &matches() const { return matches_; }
        const std::string &dir() const { return dir_; }
        std::string summary_path() const { return dir_ + "/simulation.json"; }
        // The engine binary the matches run.
        const std::string &engine() const { return engine_; }
        // A match's engine command line (empty for one not started in this session).
        const std::vector<std::string> &command(size_t i) const { return commands_[i]; }
        // The ended matches' share of all of them.
        double fraction() const;
        // "7 / 10 matches · 3 running · 0:42".
        std::string progress_text() const;
        // The last `max_bytes` of match `i`'s stderr.
        std::string stderr_tail(size_t i, size_t max_bytes = 6000) const;

    private:
        void launch(size_t i);
        void reap(size_t i);
        void write_summary();
        const char *state_key() const;

        State state_ = State::NONE;
        SimConfig config_;
        SimPaths paths_;
        std::string dir_;
        std::string engine_;
        std::array<std::string, 2> models_;  // the files MODEL entrants load
        std::vector<SimMatch> matches_;
        std::vector<std::unique_ptr<ChildProcess>> children_;
        std::vector<std::vector<std::string>> commands_;
        std::vector<std::chrono::steady_clock::time_point> started_;
        std::chrono::steady_clock::time_point began_;
        double elapsed_ = 0;  // the batch's run time once it ended
        size_t next_ = 0;     // the next waiting match
        bool finished_event_ = false;
};

}  // namespace rmwork

#endif /* RMWORK_SIMULATION_H */
