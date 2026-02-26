#ifndef INPUT_LOGGER_H
#define INPUT_LOGGER_H

#include <fstream>
#include <string>

class InputLogger {
   public:
    static InputLogger& instance();

    void init_logging(unsigned int seed, const std::string& resource_dir);
    void init_replay(const std::string& log_path);
    void init_machine(unsigned int seed, const std::string& resource_dir);
    bool is_replay_mode() const;
    bool is_machine_mode() const;
    unsigned int get_replay_seed() const;

    // num_choices: how many valid actions exist at this decision point (-1 = unspecified).
    // In machine mode this is emitted in the QUERY so the Python side can mask actions.
    int get_logged_input(size_t cur_turn, int num_choices = -1);

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
