#ifndef INPUT_LOGGER_H
#define INPUT_LOGGER_H

#include <fstream>
#include <functional>
#include <string>
#include <vector>
#include "classes/action.h"
#include "classes/deck.h"
#include "components/zone.h"

// Out-of-band flag values returned by get_input() / get_int_input(). Negative
// integers that do not correspond to any legal action index.
#define PASS_TURN_CMD -2
#define FLAG_QUIT -3

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
    bool is_replay_mode() const;
    bool is_machine_mode() const;
    // True when the game must follow the machine-mode DECISION SCHEDULE: machine mode
    // itself, or a replay of a log recorded in machine mode (FLAGS token "machine").
    // Machine mode auto-resolves some choices (mana payment, hybrid pips) and masks
    // some menu entries that the CLI path presents interactively, so a machine log
    // only replays faithfully when those same code paths are taken. Use this (not
    // is_machine_mode) in any conditional that changes WHICH decisions are queried
    // or WHAT the menu contains; keep is_machine_mode for I/O concerns (BQUERY
    // emission, output suppression, training-signal prints).
    bool is_machine_schedule() const;
    unsigned int get_replay_seed() const;
    // Parsed from the RMLOG v2 header (valid after init_replay)
    const DecisionLogHeader& get_replay_header() const;
    const Deck& get_replay_deck_a() const;
    const Deck& get_replay_deck_b() const;
    const std::vector<std::string>& get_replay_flags() const;

    int get_input(const std::vector<LegalAction>& actions);

    // In-process decision hook (Phase D in-process actor). When set, the machine-mode
    // decision read (machine_mode && !human_has_priority) delegates the choice to this
    // provider instead of populating a GameState/Query and blocking on stdin — no query
    // is emitted and stdin/stdout are never touched. The provider receives the same
    // `actions` vector and returns an action index (with the same -1 → last-slot confirm
    // remap the stdio path uses); the choice still flows through commit_choice so the
    // action-history observation block stays correct. The provider is set by the future
    // C++ actor binary (which links the engine objects minus main.o); nothing in
    // bin/robomage ever sets it, so unset is the default and the stdio path is unchanged.
    // The cooperative-unwind short-circuit (search_restore_pending) runs BEFORE this hook,
    // so search RESTORE semantics are unaffected.
    void set_input_provider(std::function<int(const std::vector<LegalAction>&)> provider);
    void clear_input_provider();

   private:
    InputLogger() = default;
    // Write the RMLOG v2 header (seed, flags, embedded decklists) to the open log file.
    void write_header(unsigned int seed, const DecisionLogHeader& header);
    // Persist a committed choice: echo it to the log file (if open) and record it in
    // the action history. Shared by the machine / auto-pass / CLI input paths.
    void commit_choice(const std::vector<LegalAction>& actions, int choice);
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
    DecisionLogHeader replay_header;
    Deck replay_deck_a;
    Deck replay_deck_b;
    std::vector<std::string> replay_flags;
    int auto_pass_until_turn = -1;
    // Unset by default (see set_input_provider); only the Phase D in-process actor sets it.
    std::function<int(const std::vector<LegalAction>&)> input_provider;
};

// Ask `chooser` an optional yes/no question with a single decline (index 0) and accept
// (index 1) action, both categorised OPTIONAL_YESNO. Returns true if they accept. Temporarily
// points priority at the chooser so the decision is logged from their seat, then restores it.
// Shared by every "you may ..." resolution-time confirmation (the OptionalDecider triggers,
// the ImmediateTrigger optional cost, etc.) so the yes/no menu is built one way.
bool request_optional_yesno(Zone::Ownership chooser, const std::string& prompt);

#endif
