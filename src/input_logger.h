#ifndef INPUT_LOGGER_H
#define INPUT_LOGGER_H

#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "classes/action.h"
#include "classes/deck.h"
#include "components/zone.h"

// Out-of-band values accepted BY get_input() as input. Negative integers that do
// not correspond to any legal action index. -1 is the confirm-slot sentinel (the
// Python env remaps the confirm slot to -1; get_input remaps it back to the last
// action index), so the concession sentinels start at -2.
//
// CONCEDE_GAME / CONCEDE_MATCH are protocol INPUT on every path (machine mode,
// replay, in-process agents and the CLI's human seats): the seat the pending
// decision belongs to concedes per CR 104.3a.
#define CONCEDE_GAME -2
#define CONCEDE_MATCH -3

// The decision log's last line when a seat's in-process decider ended the run mid-decision with
// no result (a human seat's quit, or the end of its input): --replay stops cleanly there.
#define QUIT_MARKER "QUIT"

// Everything the RMLOG v3 decision-log header records beyond the seed: the
// state-affecting launch flags and both decklists (verbatim file text), so a
// log is a self-contained replay — `--replay <file>` needs no other arguments.
struct DecisionLogHeader {
    std::string flags;  // space-separated tokens; empty = none (written as "-")
    std::string deck_a_name;
    std::string deck_a_text;
    std::string deck_b_name;
    std::string deck_b_text;
    void add_flag(const std::string& token);
    void add_flag(const std::string& key, const std::string& value);
    // key=comma,joined,list — no-op when the list is empty
    void add_flag_list(const std::string& key, const std::vector<std::string>& values);
};

class InputLogger {
   public:
    static InputLogger& instance();

    void init_logging(unsigned int seed, const std::string& resource_dir, const DecisionLogHeader& header);
    void init_replay(const std::string& log_path);
    void init_machine(unsigned int seed, const std::string& resource_dir, bool enable_log,
                      const DecisionLogHeader& header);
    // A process that plays several matches (agents/actor.h): closes the decision log of the last
    // match, if one is open, and starts the next one's at `path` (its header for `seed`), in machine
    // mode, with the run's decision counts back at 0. False (with a non-fatal error printed) when
    // the log cannot be opened.
    bool begin_match_log(unsigned int seed, const std::string &path, const DecisionLogHeader &header);
    // Closes the decision log being written, if one is.
    void end_match_log();
    // A game starts (play_single_game): its decision count, which --max-game-decisions caps, is 0.
    void begin_game();
    // --decision-log FILE: the decision log init_logging / init_machine open is FILE instead of
    // resources/logs/game_<seed>.log, and no "Logging inputs to" line is printed (the caller
    // named the file). Call before either init.
    void set_log_path(const std::string& path);
    bool is_replay_mode() const;
    // Machine mode is an I/O mode only (the RMSTATE stream, the in-process agents, output
    // suppression): the decision schedule — which decisions are queried and what each menu
    // contains — is the same for every consumer, so nothing that shapes a menu may read this.
    bool is_machine_mode() const;
    unsigned int get_replay_seed() const;
    // The decision log being written (empty when none is).
    const std::string &log_file_name() const;
    // Parsed from the RMLOG v3 header (valid after init_replay)
    const DecisionLogHeader& get_replay_header() const;
    const Deck& get_replay_deck_a() const;
    const Deck& get_replay_deck_b() const;
    const std::vector<std::string>& get_replay_flags() const;

    // One decision: presents `actions` — with each choice interchangeable with an earlier one
    // left out (two identical cards in a graveyard or library, see interchangeable_cards) — to the
    // deciding player or agent, and returns the chosen action's index in `actions` (or a
    // concession / unwind sentinel). A menu left with one row is not presented: its row is taken
    // (no frame, no decision-log line, no decision number), so a seat can concede only at a
    // decision with a choice.
    int get_input(const std::vector<LegalAction>& actions);

    // In-process decision hook. When set, the decision read (outside --replay) of a seat the
    // provider covers delegates the choice to it instead of the RMSTATE stream: no frame is
    // written to stdout and stdin is not read by the engine. The provider receives the deciding
    // seat (the sideboarding player between games, else the priority holder:
    // rmstate_deciding_seat) and the same `actions` vector, and returns an action index (with the
    // same -1 → last-slot confirm remap and concession sentinels the stream accepts); the choice
    // still flows through commit_choice so the decision log stays complete. A seat it does not
    // cover reads the stream. Set by the in-process agents and the CLI's human seats
    // (agents/driver.h); unset by default. `real_decisions` is false for a provider that also
    // answers decisions no game makes (a search agent's simulations): the --selfcheck-state checks
    // skip every decision such a provider covers.
    using InputProvider = std::function<int(Zone::Ownership seat, const std::vector<LegalAction> &)>;
    void set_input_provider(
        InputProvider provider, bool covers_a = true, bool covers_b = true, bool real_decisions = false);
    void clear_input_provider();

    // The in-process decider of the decision being read ended the run there, with no result (a
    // human seat's quit or the end of its input): writes QUIT_MARKER as the decision log's last
    // line, so --replay of the log stops cleanly at this decision (end_run_at_recorded_quit), and
    // exits with status 0.
    [[noreturn]] void end_run_by_quit();
    // A stop was requested (an actor's SIGTERM / SIGINT, agents/actor.h) and the decision being
    // read has not been decided: writes QUIT_MARKER as the decision log's last line (a replay stops
    // cleanly there), writes no episode for the unfinished game, lets the actor record its stop and
    // exits with status 0.
    [[noreturn]] void end_run_by_signal();

   private:
    InputLogger() = default;
    // Write the RMLOG v3 header (seed, flags, embedded decklists) to the open log file.
    void write_header(unsigned int seed, const DecisionLogHeader& header);
    // Persist a committed choice: echo it to the log file (if open). Shared by the
    // provider and stream input paths.
    void commit_choice(const std::vector<LegalAction>& actions, int choice);
    // An integer a machine-side source (the in-process provider, an RMSTATE reply) answered with:
    // a concession sentinel is committed and applied; -1 (confirm) becomes the last slot; any
    // other value must index `actions`. Returns the choice, committed.
    int accept_machine_choice(const std::vector<LegalAction>& actions, int choice);
    // get_input over a menu with its interchangeable choices already left out.
    int choose(const std::vector<LegalAction>& actions);
    bool replay_mode = false;
    bool machine_mode = false;
    // 1-based count of decisions read from the replay log (for divergence reports)
    size_t replay_decision_no = 0;
    unsigned int replay_seed = 0;
    std::ofstream log_file;
    std::ifstream replay_file;
    std::string log_path;
    // Set by set_log_path; empty = the seed-named default.
    std::string named_log_path;
    // Opens the decision log (the named path, else the default under resource_dir) and writes its
    // header; false when it cannot be opened.
    bool open_log(unsigned int seed, const std::string& resource_dir, const DecisionLogHeader& header);
    DecisionLogHeader replay_header;
    Deck replay_deck_a;
    Deck replay_deck_b;
    std::vector<std::string> replay_flags;
    // Unset by default (see set_input_provider).
    InputProvider input_provider;
    bool provider_covers[2] = {false, false};  // seats A, B
    bool provider_real_decisions = false;
    // Whether the provider decides `seat`'s decisions.
    bool provider_decides(Zone::Ownership seat) const;
    // The input read for one decision once the frame writers have run (the replay, provider and
    // stream paths).
    int read_choice(const std::vector<LegalAction> &actions, Zone::Ownership seat);
    // Decisions read so far in the run (for --max-decisions and the decision trace).
    size_t decisions_read = 0;
    // Decisions read so far in the current game, its sideboarding before it left out (for
    // --max-game-decisions).
    size_t game_decisions_read = 0;
    // Writes QUIT_MARKER as the decision log's last line, if one is open.
    void write_quit_marker();
    // The reply of the decision just read, as the decision log holds it: a menu row of the menu as
    // offered, or a concession sentinel (read_choice returns the unwind's first choice for one).
    int last_reply = 0;
};

// Ask `chooser` an optional yes/no question with a single decline (index 0) and accept
// (index 1) action, both categorised OPTIONAL_YESNO. Returns true if they accept. Temporarily
// points priority at the chooser so the decision is logged from their seat, then restores it.
// Shared by every "you may ..." resolution-time confirmation (the OptionalDecider triggers,
// the ImmediateTrigger optional cost, etc.) so the yes/no menu is built one way.
bool request_optional_yesno(Zone::Ownership chooser, const std::string& prompt);

#endif
