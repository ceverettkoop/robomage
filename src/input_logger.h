#ifndef INPUT_LOGGER_H
#define INPUT_LOGGER_H

#include <fstream>
#include <string>
#include <vector>
#include "classes/action.h"

class InputLogger {
   public:
    static InputLogger& instance();

    void init_logging(unsigned int seed, const std::string& resource_dir);
    void init_replay(const std::string& log_path);
    void init_machine(unsigned int seed, const std::string& resource_dir);
    bool is_replay_mode() const;
    bool is_machine_mode() const;
    unsigned int get_replay_seed() const;

    // Primary overload: takes full LegalAction list for richest query data.
    int get_logged_input(size_t cur_turn, const std::vector<LegalAction>& actions);

    // Backward-compat overload: builds trivial LegalActions from category+entity
    // pairs and delegates to the primary overload.
    int get_logged_input(size_t cur_turn,
                         const std::vector<ActionCategory>& action_categories,
                         const std::vector<Entity>& entities = {});

   private:
    InputLogger() = default;
    bool replay_mode = false;
    bool machine_mode = false;
    unsigned int replay_seed = 0;
    std::ofstream log_file;
    std::ifstream replay_file;
    std::string log_path;
    int auto_pass_until_turn = -1;
};

#endif
