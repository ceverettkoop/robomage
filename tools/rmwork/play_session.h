#ifndef RMWORK_PLAY_SESSION_H
#define RMWORK_PLAY_SESSION_H

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "autopass.h"
#include "match.h"
#include "model_read.h"
#include "replay_session.h"
#include "settings.h"
#include "subprocess.h"

// ── Play: a live game against an in-process agent, the human's seat on the RMSTATE stream ───────
//
// start() runs the engine (`play_command`: `--machine`, the human's seat with no agent so it reads
// the stream, the opponent's seat an in-process agent, `--decision-log` into the games directory) in
// the engine's working directory with its stdin and stdout on pipes and its stderr in a file
// (subprocess.h's start_piped: its own process group, killed whole). A thread reads the stdout's
// RMSTATE records (docs/state_format.md §9); the UI thread's poll() takes them, appends the human
// seat's to the live match (match.h) and keeps the session's ReplaySession (the tools' source in
// Play) at the newest decision. At the match's end the engine writes every seat's terminal frame to
// the stream, the other seat's with its once-per-match frames: a record whose header names another
// viewer is dropped unread (withheld()), so the tools never see another viewer's frame.
// reply() writes one integer line: a menu row (the confirm row of an attacker or blocker declaration
// is the last row, which -1 also names), -2 or -3 to concede (only at an asked decision: a
// concession asked for while the opponent thinks waits for the human's next decision).
//
// Passing (autopass.h): F2 / F6 and the per-step stops answer a decision as it arrives. Model's Read
// (model_read.h), when asked for, gets every record of the stream and its values join the decisions
// as episode steps (file "Model's Read"), which the Value Graph, Choices' π and the Model's Read
// tool show; without a server it says so (a note) and the game goes on, and a failure of it is
// reported once (take_model_failure) while the game goes on without it.
//
// Beside the decision log (<stem>.rmlog) go the engine's stderr (<stem>.stderr), the human seat's
// stream (<stem>.seat-a.rmstate / .seat-b.rmstate) and the game's sidecar (<stem>.game.json: the
// human's seat, the opponent, the decks, the format and the seed, start_data.h), which the start
// page reads.
//
// The engine's end: once it has exited and the stream has ended, a match with its terminal frame is
// finished (take_finished); anything else (an exit mid-game, a frame that does not decode, a frame
// of another viewer, a failed write) ends the session with a failure (take_failure: title, message
// and the stderr tail). stop() kills and reaps the engine and joins the thread; so does the
// destructor. A game that ends without its result (stopped, or the engine gone) gets the QUIT line
// appended to its decision log, as a human seat's quit writes it, so Replay opens it to the decision
// it stopped at.

namespace rmwork {

enum class Opponent {
    MODEL,     // nn:<model>, in robomage_nn
    SERVER,    // nn:remote:<socket>, in robomage_nn
    EXPLORE,   // explore
    RANDOM,    // random
    SCRIPTED,  // the scripted agent (python/teacher.py --tier hard) through the teacher channel at mix 1
};
inline constexpr int OPPONENT_COUNT = 5;
// "model", "server", "explore", "random", "scripted".
const char *opponent_key(Opponent o);
// "Model (nn:)", …
const char *opponent_label(Opponent o);
// The opponent of key `key`, or -1.
int find_opponent(const std::string &key);
// Whether the opponent runs in robomage_nn.
inline bool opponent_needs_nn(Opponent o) {
    return o == Opponent::MODEL || o == Opponent::SERVER;
}

struct PlayConfig {
        int seat = 0;      // the human's seat: 0 A, 1 B
        std::string deck;  // the human's deck, relative to resources/decks (no .dk)
        Opponent opponent = Opponent::EXPLORE;
        std::string model;  // MODEL: the export's path; SERVER: the socket
        std::string opponent_deck;
        bool bo3 = false;
        uint64_t seed = 1;  // the engine's --game-seed
        bool auto_mana = true;
        bool model_read = false;
        // The engine's test-setup flags after the others ("--hand-a", "Lightning Bolt", "--battlefield-b",
        // …: a sculpted game, which the decision log's header records), for the self-test; none from the
        // New Game dialog.
        std::vector<std::string> setup;
};

// The configuration of the New Game dialog's choices (false with `why` for an unknown opponent).
bool play_config_from(const NewGameSettings &g, PlayConfig *out, std::string *why);

// Where a session runs.
struct PlayPaths {
        std::string engine;     // the engine binary to run (play_engine_binary applied)
        std::string workdir;    // its working directory (bin/)
        std::string games_dir;  // where the decision log, the seat's stream and stderr go
        std::string repo;       // the repository (python/teacher.py)
        std::string python = "python3";
        ModelReadConfig model_read;  // Model's Read's server and policies (engine, workdir, cards path set by start)
};

// The engine a session with opponent `o` runs: robomage_nn for an nn: opponent (`configured`, or the
// robomage_nn beside it), else the plain robomage beside a configured robomage_nn (no learned policy
// needed), else `configured`. "" with `why` when an nn: opponent has no robomage_nn.
std::string play_engine_binary(const std::string &configured, Opponent o, std::string *why);
// The opponent seat's agent spec ("explore", "nn:<model>", "nn:remote:<socket>", "first" for the
// scripted agent's seat, which its teacher plays).
std::string opponent_agent(const PlayConfig &c);
// The engine's arguments after its path.
std::vector<std::string> play_arguments(
    const PlayConfig &c, const std::string &log_path, const std::string &repo, const std::string &python);

class PlaySession {
    public:
        enum class State {
            NONE,      // no game started
            RUNNING,   // the engine runs
            FINISHED,  // the match ended
            FAILED,    // the session ended with a failure
            STOPPED    // stopped by the workbench
        };

        PlaySession() = default;
        PlaySession(const PlaySession &) = delete;
        PlaySession &operator=(const PlaySession &) = delete;
        ~PlaySession();

        // Starts a game (a running one is stopped first); false with `err` when the engine cannot
        // start or the files cannot be made.
        bool start(const PlayConfig &c, const PlayPaths &p, std::string *err);
        // The UI thread, once a frame: the frames read so far, the passing, Model's Read's values,
        // the engine's end.
        void poll(const StopSettings &stops);
        // Kills and reaps the engine, joins the reader (no failure reported).
        void stop();
        // Stops it and forgets the game: no game started (File → Close; the files stay).
        void close_game();

        State state() const { return state_; }
        bool running() const { return state_ == State::RUNNING; }
        // A game was started (its match may still have no decision).
        bool started() const { return state_ != State::NONE; }
        // The human has an unanswered decision.
        bool deciding() const { return pending_ >= 0; }
        long pending_index() const { return pending_; }
        // The engine runs and has not asked the human yet since the last reply.
        bool waiting() const { return running() && pending_ < 0; }
        // Sends `reply` (a row of the pending decision, -1, -2 or -3); false with `err` when no
        // decision is pending, the row is not on its menu, or the write fails (a failure is then
        // reported too).
        bool reply(int reply, std::string *err);
        // Concedes the game (-2) or the match (-3): now when deciding, else at the next decision.
        void concede(bool match);
        bool concede_queued() const { return queued_concede_ != 0; }
        // Starts F2 / F6 from the newest decision (answering it when it waits); false with `why` when
        // there is no running game.
        bool pass(PassMode mode, const StopSettings &stops, std::string *why);
        PassMode pass_mode() const { return autopass_.mode; }

        // The tools' source: the live match (loaded once its first decision arrived), the human's view.
        ReplaySession &view() { return view_; }
        const ReplaySession &view() const { return view_; }
        // The live match (nullptr before start).
        const Match *match() const;
        const PlayConfig &config() const { return config_; }
        const std::string &log_path() const { return log_path_; }
        const std::string &stderr_path() const { return stderr_path_; }
        const std::string &stream_path() const { return stream_path_; }
        const std::vector<std::string> &command() const { return command_; }
        // The decisions the workbench answered itself (F2, F6, stops), by index.
        const std::vector<size_t> &auto_answered() const { return auto_answered_; }
        // Records read off the stream, and those of another viewer dropped unread.
        size_t records() const { return records_; }
        size_t withheld() const { return withheld_; }
        // The record number (from 1) of the first record withheld and of the last decision's frame
        // (0 none).
        size_t first_withheld_record() const { return first_withheld_record_; }
        size_t last_decision_record() const { return last_decision_record_; }
        const ModelReader &model_read() const { return model_; }

        // Taken once: a failure (title, message, detail with the stderr tail).
        bool take_failure(std::string *title, std::string *message, std::string *detail);
        // Taken once: the match ended.
        bool take_finished();
        // Taken once: Model's Read failed (a refused handshake, a lost server); the game goes on.
        bool take_model_failure(std::string *message);
        // Taken once: what the status bar may say ("F2 stopped: the turn ended").
        std::vector<std::string> take_notes();

        // The last `max_bytes` of the engine's stderr.
        std::string stderr_tail(size_t max_bytes = 6000) const;
        // Kills the engine without stopping the session, as if it had died (the self-test's
        // engine-death path).
        void kill_engine() { child_.kill_and_reap(); }

    private:
        void reader_main(int fd);
        void take_records();
        void add_record(const std::string &payload);
        void answer_pending(const StopSettings &stops);
        void attach_values();
        void check_exit();
        void fail(const std::string &title, const std::string &message);
        void close_pipes();
        bool write_reply(int reply, std::string *err);
        Match *live();

        State state_ = State::NONE;
        PlayConfig config_;
        PlayPaths paths_;
        ChildProcess child_;
        int to_child_ = -1;
        std::thread reader_;
        std::mutex mu_;
        std::deque<std::string> inbox_;
        bool stream_done_ = false;
        std::string stream_error_;
        std::unique_ptr<Match> starting_;  // the match until its first decision
        ReplaySession view_;
        long pending_ = -1;
        bool considered_ = false;  // the pending decision went through the passing
        int queued_concede_ = 0;
        AutoPass autopass_;
        std::vector<size_t> auto_answered_;
        size_t records_ = 0;
        size_t withheld_ = 0;
        size_t first_withheld_record_ = 0, last_decision_record_ = 0;
        std::string log_path_, stderr_path_, stream_path_;
        std::vector<std::string> command_;
        bool finished_event_ = false;
        bool failure_event_ = false;
        std::string fail_title_, fail_message_, fail_detail_;
        std::vector<std::string> notes_;
        ModelReader model_;
        bool model_reported_ = false;  // Model's Read's failure or absence was reported
        bool model_failure_event_ = false;
};

}  // namespace rmwork

#endif /* RMWORK_PLAY_SESSION_H */
