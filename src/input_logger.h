#ifndef INPUT_LOGGER_H
#define INPUT_LOGGER_H

#include <fstream>
#include <string>
#include <vector>
#include "classes/action.h"
#include "components/zone.h"
#include "gui_flags.h"

class InputLogger {
   public:
    static InputLogger& instance();

    void init_logging(unsigned int seed, const std::string& resource_dir);
    void init_replay(const std::string& log_path);
    void init_machine(unsigned int seed, const std::string& resource_dir);
    bool is_replay_mode() const;
    bool is_machine_mode() const;
    unsigned int get_replay_seed() const;

    int get_input(const std::vector<LegalAction>& actions);

   private:
    InputLogger() = default;
    // Persist a committed choice: echo it to the log file (if open) and record it in
    // the action history. Shared by the machine / auto-pass / CLI input paths.
    void commit_choice(const std::vector<LegalAction>& actions, int choice);
    bool replay_mode = false;
    bool machine_mode = false;
    unsigned int replay_seed = 0;
    std::ofstream log_file;
    std::ifstream replay_file;
    std::string log_path;
    int auto_pass_until_turn = -1;
};

// Ask `chooser` an optional yes/no question with a single decline (index 0) and accept
// (index 1) action, both categorised OPTIONAL_YESNO. Returns true if they accept. Temporarily
// points priority at the chooser so the decision is logged from their seat, then restores it.
// Shared by every "you may ..." resolution-time confirmation (the OptionalDecider triggers,
// the ImmediateTrigger optional cost, etc.) so the yes/no menu is built one way.
bool request_optional_yesno(Zone::Ownership chooser, const std::string& prompt);

#endif
