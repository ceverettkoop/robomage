#ifndef CLI_OUTPUT_H
#define CLI_OUTPUT_H

#include <cstdarg>
#include <string>
#include "classes/gamestate.h"
#include "classes/game.h"
#include "components/zone.h"

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
std::string player_name(Zone::Ownership owner);

// Logging: routes to GUI buffer in GUI mode, printf in CLI; no-op in machine mode
void game_log(const char* fmt, ...);
// Like game_log, but suppressed when a human player is designated and private_to is their opponent
void game_log_private(Zone::Ownership private_to, const char* fmt, ...);

// Machine query emitter: raw printf of QUERY line (called only in machine mode)
void cli_emit_machine_query(const Query* q, const GameState* gs);

// C-API buffer accessors (callable from C, including gui.c)
#ifdef __cplusplus
extern "C" {
#endif
int gui_log_line_count(void);
const char* gui_log_get_line(int idx);
void gui_query_clear(void);
int gui_query_line_count(void);
const char* gui_query_get_line(int idx);
#ifdef __cplusplus
}
#endif

#endif /* CLI_OUTPUT_H */
