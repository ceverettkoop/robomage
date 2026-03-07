#ifndef CLI_OUTPUT_H
#define CLI_OUTPUT_H

#include <string>
#include "classes/gamestate.h"
#include "classes/game.h"

// Startup / meta
void cli_print_version(const char* version);
void cli_print_help(const char* program, const char* version);
void cli_print_seed(unsigned int seed);
void cli_print_turn_header(size_t turn, bool player_a_turn);
void cli_print_invalid_action();
void cli_print_gui_exit();

// Game state display (reads from GameState only — no ECS access)
void print_game_state(const GameState* gs);

// Choice display
void print_query(const Query* q, bool player_a_has_priority);

// Errors (replaces error.cpp printfs)
void cli_error(const std::string& msg);
[[noreturn]] void cli_fatal_error(const std::string& msg);

// Utility
const char* step_to_string(Step step);

#endif /* CLI_OUTPUT_H */
