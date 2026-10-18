#ifndef INPUT_LOGGER_H
#define INPUT_LOGGER_H

#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "classes/action.h"
#include "classes/deck.h"
#include "components/zone.h"

// Out-of-band values accepted BY get_input() as input, or returned by
// get_int_input(). Negative integers that do not correspond to any legal action
// index. -1 is the confirm-slot sentinel (the Python env remaps the confirm slot
// to -1; get_input remaps it back to the last action index), so the concession
// sentinels start at -2.
//
// CONCEDE_GAME / CONCEDE_MATCH are protocol INPUT on every path (machine mode,
// replay, CLI): the seat the pending decision belongs to concedes per CR 104.3a.
// PASS_TURN_CMD / FLAG_QUIT are CLI-only markers produced by the 'z'/'q' keys
// and never appear in an input stream, so they sit far out of the way of any
// value a driver could send.
#define CONCEDE_GAME -2
#define CONCEDE_MATCH -3
#define PASS_TURN_CMD -100
#define FLAG_QUIT -101

// Everything the RMLOG v2 decision-log header records beyond the seed: the
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
    // --decision-log FILE: the decision log init_logging / init_machine open is FILE instead of
    // resources/logs/game_<seed>.log, and no "Logging inputs to" line is printed (the caller
    // named the file). Call before either init.
    void set_log_path(const std::string& path);
    bool is_replay_mode() const;
    bool is_machine_mode() const;
    // True when the game must follow the machine-mode DECISION SCHEDULE: machine mode
    // itself, or a replay of a log recorded in machine mode (FLAGS token "machine").
    // Machine mode auto-resolves some choices (mana payment, hybrid pips) and masks
    // some menu entries that the CLI path presents interactively, so a machine log
    // only replays faithfully when those same code paths are taken. Use this (not
    // is_machine_mode) in any conditional that changes WHICH decisions are queried
    // or WHAT the menu contains; keep is_machine_mode for I/O concerns (output
    // suppression).
    bool is_machine_schedule() const;
    unsigned int get_replay_seed() const;
    // Parsed from the RMLOG v2 header (valid after init_replay)
    const DecisionLogHeader& get_replay_header() const;
    const Deck& get_replay_deck_a() const;
    const Deck& get_replay_deck_b() const;
    const std::vector<std::string>& get_replay_flags() const;

    // One decision: presents `actions` — with each choice interchangeable with an earlier one
    // left out (two identical cards in a graveyard or library, see interchangeable_cards) — to the
    // deciding player or agent, and returns the chosen action's index in `actions` (or a
    // concession / unwind sentinel).
    int get_input(const std::vector<LegalAction>& actions);

    // In-process decision hook. When set, the machine-mode decision read (machine_mode &&
    // !human_has_priority) of a seat the provider covers delegates the choice to it instead of
    // emitting a query and blocking on stdin: no query is emitted and stdin/stdout are not
    // touched. The provider receives the deciding seat (the sideboarding player between games,
    // else the priority holder: rmstate_deciding_seat) and the same `actions` vector, and returns
    // an action index (with the same -1 → last-slot confirm remap and concession sentinels the
    // stdio path accepts); the choice still flows through commit_choice so the decision log stays
    // complete. A seat it does not cover keeps the stdio path. Set by the in-process agents
    // (agents/driver.h); unset by default. `real_decisions` is false for a provider that also
    // answers decisions no game makes (a search agent's simulations): the --selfcheck-state checks
    // skip every decision such a provider covers.
    using InputProvider = std::function<int(Zone::Ownership seat, const std::vector<LegalAction> &)>;
    void set_input_provider(
        InputProvider provider, bool covers_a = true, bool covers_b = true, bool real_decisions = false);
    void clear_input_provider();

   private:
    InputLogger() = default;
    // Write the RMLOG v2 header (seed, flags, embedded decklists) to the open log file.
    void write_header(unsigned int seed, const DecisionLogHeader& header);
    // Persist a committed choice: echo it to the log file (if open). Shared by the
    // machine / auto-pass / CLI input paths.
    void commit_choice(const std::vector<LegalAction>& actions, int choice);
    // An integer a machine-side source (the in-process provider, an RMSTATE reply) answered with:
    // a concession sentinel is committed and applied; -1 (confirm) becomes the last slot; any
    // other value must index `actions`. Returns the choice, committed.
    int accept_machine_choice(const std::vector<LegalAction>& actions, int choice);
    // get_input over a menu with its interchangeable choices already left out.
    int choose(const std::vector<LegalAction>& actions);
    bool replay_mode = false;
    bool machine_mode = false;
    // Replayed log was recorded in machine mode (see is_machine_schedule)
    bool replay_machine_schedule = false;
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
    int auto_pass_until_turn = -1;
    // Unset by default (see set_input_provider).
    InputProvider input_provider;
    bool provider_covers[2] = {false, false};  // seats A, B
    bool provider_real_decisions = false;
    // Whether the provider decides `seat`'s decisions.
    bool provider_decides(Zone::Ownership seat) const;
    // The input read for one decision once the frame writers have run (the replay, machine and
    // CLI paths).
    int read_choice(const std::vector<LegalAction> &actions, Zone::Ownership seat);
    // Decisions read so far in the run (for --max-decisions and the decision trace).
    size_t decisions_read = 0;
};

// Ask `chooser` an optional yes/no question with a single decline (index 0) and accept
// (index 1) action, both categorised OPTIONAL_YESNO. Returns true if they accept. Temporarily
// points priority at the chooser so the decision is logged from their seat, then restores it.
// Shared by every "you may ..." resolution-time confirmation (the OptionalDecider triggers,
// the ImmediateTrigger optional cost, etc.) so the yes/no menu is built one way.
bool request_optional_yesno(Zone::Ownership chooser, const std::string& prompt);

#endif
