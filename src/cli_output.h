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
void cli_print_pregame_header();
void cli_print_turn_header(size_t turn, bool player_a_turn);
void cli_print_invalid_action();

// Game state display (reads from GameState only — no ECS access)
void print_game_state(const GameState* gs);

// Choice display
void print_query(const Query* q, bool player_a_has_priority);

// Errors (replaces error.cpp printfs)
void cli_warning(const std::string& msg);
void cli_error(const std::string& msg);
[[noreturn]] void cli_fatal_error(const std::string& msg);

// Utility
const char* step_to_string(Step step);
std::string player_name(Zone::Ownership owner);

// Logging: printf in CLI; no-op in machine mode
void game_log(const char* fmt, ...);
// Like game_log, but suppressed when a human player is designated and private_to is their opponent
void game_log_private(Zone::Ownership private_to, const char* fmt, ...);
// Complement of game_log_private: the generic line shown only to an opponent
// viewer (in place of owner's hidden card). Pair with game_log_private per event.
void game_log_redacted(Zone::Ownership owner, const char* fmt, ...);
// True (setting *owner) when narrative should be limited to one player's view
// (--player human seat, or --log-viewer spectator seat); false for full narrative
bool resolve_narrative_viewer(Zone::Ownership* owner);

// Machine query emitter: BQUERY header line + binary payload (called only in machine mode)
void cli_emit_machine_query(const Query* q, const GameState* gs);
void cli_emit_machine_bstate(const Query* q, const GameState* gs);

#endif /* CLI_OUTPUT_H */
