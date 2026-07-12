#ifndef SEARCH_SERVER_H
#define SEARCH_SERVER_H

#include <functional>

// --search-server: machine-protocol extension for MCTS tree search.
//
// With the flag on, the machine-mode decision read accepts text commands in
// addition to the usual integer action index:
//
//   SNAPSHOT <slot>      deep-copy the full game state into a snapshot slot
//                        (legal only at a loop-safe decision; see below)
//   RESTORE <slot>       roll the game back to a saved slot
//   DETERMINIZE <seed>   reshuffle the hidden zones (unknown library cards and
//                        the opponent's unknown hand cards) from the priority
//                        player's perspective, using a world-local RNG
//                        (legal only at a loop-safe decision)
//   RELEASE              drop all snapshots (also leaves the terminal intercept)
//
// A decision is "loop-safe" when the whole game state lives in cur_game + the
// ECS and re-entering the main loop from that data re-derives the identical
// decision: the priority decision in play_single_game, the attacker/blocker
// SELECT prompts, and the cleanup discard. Mid-resolution prompts (targets,
// dig/discard picks, the attack/block TARGET sub-prompts, combat damage
// assignment) sit above live C++ stack frames and are not restorable. Every
// query is preceded by a "SEARCHINFO safe=<0|1>" line so the driver knows.
//
// RESTORE works by cooperative unwind: the engine cannot unwind the C++ stack
// (-fno-exceptions), so get_input hands back choice 0 at every decision --
// emitting no query and consuming no stdin -- until control returns to the top
// of the main loop, where the pending restore is applied and the loop
// re-derives (and re-emits) the restored decision. The state the auto-0
// choices mutate is about to be overwritten, so they are harmless.
//
// A game that ENDS during a simulated line must not exit the process: the main
// loop's end-of-game checks route through search_intercept_game_end(), which
// (when a snapshot is live) emits "SIM_RESULT: <A|B|DRAW>" and blocks for
// RESTORE or RELEASE instead of finishing the game.

extern bool search_server_mode;

// Loop-safety marker, set around the loop-safe get_input call sites.
void search_set_loop_safe(bool safe);
bool search_loop_safe();

// Cooperative-unwind state (see above). search_unwind_choice() consumes one
// unwind step and fatals past a runaway cap.
bool search_restore_pending();
int search_unwind_choice();
void search_apply_pending_restore();

// Latch a restore for the main loop to apply (the in-process equivalent of the
// stdio RESTORE command): validates the slot is live, then arms the pending
// restore so search_restore_pending()/search_apply_pending_restore() unwind and
// roll the game back at the top of the loop. Reuses the exact path the stdio
// RESTORE handler uses. Provided for the Phase D in-process game-end hook.
void search_request_restore(int slot);

// Game-end intercept hook for the Phase D in-process actor. When set (and a
// snapshot slot is live), search_intercept_game_end() delegates a simulated
// line's game-over to this hook instead of the stdio SIM_RESULT/stdin branch:
// the hook is passed cur_game.winner (0=none/draw, 1=A, 2=B), records the
// simulated result, latches a restore via search_request_restore(), and returns
// true to keep the loop running (the pending restore unwinds at the loop top) or
// false to let the game end. It is set by the future C++ actor binary (which
// links the engine objects minus main.o); nothing in bin/robomage sets it, so
// unset is the default and the stdio search-server behavior is byte-identical.
// With the hook set, search_server_mode stays false, so no SEARCHINFO/SIM_RESULT
// lines are printed.
void search_set_game_end_hook(std::function<bool(int winner)> hook);
void search_clear_game_end_hook();

// Handle one non-integer machine-protocol line. Returns true if it was a
// recognized command (RESTORE leaves the restore pending for the main loop;
// the others reply and return). Unknown commands are fatal.
bool search_handle_command(const char *line);

// Terminal intercept: called wherever the main loop would end the game.
// Returns true when the "game over" belonged to a simulated line and play
// should continue (state already restored); false when the game really ends.
bool search_intercept_game_end();

// Reshuffle the hidden information from the current priority player's
// perspective using a world-local RNG (never cur_game.gen, so the real game's
// RNG stream is untouched). Exported for the future in-process search actor.
void determinize_hidden_state(unsigned int world_seed);

#endif /* SEARCH_SERVER_H */
