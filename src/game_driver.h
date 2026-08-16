#ifndef GAME_DRIVER_H
#define GAME_DRIVER_H

// Reusable game-driver code shared by the robomage front-end (main.cpp) and any
// other binary that needs to set up and run games in-process (e.g. az_actor).
// Everything here lives outside main.cpp so a second binary can link all engine
// objects EXCEPT obj/main.o and still have the globals, ECS setup, and per-game
// loop available.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "classes/deck.h"
#include "classes/game.h"
#include "classes/gamestate.h"
#include "classes/match_context.h"
#include "components/zone.h"
#include "ecs/coordinator.h"

// Forward declarations for the systems held by EcsSystems.
class Orderer;
class StateManager;
class StackManager;

// ---- Global game state (definitions live in game_driver.cpp) ----
extern std::string RESOURCE_DIR;
extern Coordinator global_coordinator;
extern Deck DEFAULT_DECK_ONE;
extern Deck DEFAULT_DECK_TWO;
extern Game cur_game;
extern bool has_human_player;
extern bool human_player_is_a;
extern bool log_viewer_set;
extern Zone::Ownership log_viewer_owner;

extern GameState gs;
extern const GameState *gs_ptr;
extern std::string replay_file_path;
extern bool replay_mode;
extern bool machine_mode;
extern std::string deck_a_name;
extern std::string deck_b_name;
extern bool seed_override;
extern unsigned int seed_value;
extern bool no_shuffle;
extern bool narrative_mode;
extern bool bo3_mode;
extern bool log_decisions_flag;
extern bool broadcast_steps_mode;
extern std::vector<std::string> battlefield_a_cards;
extern std::vector<std::string> battlefield_b_cards;
extern std::vector<std::string> graveyard_a_cards;
extern std::vector<std::string> graveyard_b_cards;
extern std::vector<std::string> exile_a_cards;
extern std::vector<std::string> exile_b_cards;
extern std::vector<std::string> sideboard_a_cards;
extern std::vector<std::string> sideboard_b_cards;
extern int life_a_override;
extern int life_b_override;

extern int match_game_number;
extern int match_wins_a;
extern int match_wins_b;
extern bool sideboard_phase;
extern Zone::Ownership sideboard_phase_player;
// The phase state being driven right now, or null outside the phase. The state
// serializer reads its progress scalars (swaps completed, maindeck drift) rather
// than keeping mirror globals that could drift from it.
extern const SideboardPhaseState *sideboard_phase_state;

// The whole bo3 match's between-game state (dispatched over by play_bo3_match).
// Exposed so the match-scoped game snapshot (snapshot.cpp) can capture/restore
// it across an init_ecs() teardown — a sideboard-rooted MCTS search snapshots
// the match, rolls forward into the next game, and restores this wholesale.
extern MatchContext g_match_ctx;

// Bumped every init_ecs() (next to card_db.clear()). The snapshot captures it so
// snapshot_restore knows a card_db.clear()+reload happened across the rollback
// (a coincidental size match would otherwise skip the card_db copy and leave the
// name->entity map pointing at torn-down prototype entities).
extern uint64_t g_card_db_generation;

// True while play_single_game's decision loop is running. Gates blocking
// fallbacks for prompts that fire with no loop top to suspend to. Since the
// pregame gate (Batch 13) the whole pregame runs inside the loop, so the
// fallbacks are defensive only.
bool in_main_loop();

// ── Concession (CR 104.3a) ──────────────────────────────────────────────────
// A player may concede at any time; they lose the game immediately. The engine
// reads a concession as an out-of-band decision INPUT: the sentinels
// CONCEDE_GAME (-2) and CONCEDE_MATCH (-3) (see input_logger.h) are accepted
// wherever an action index is read, and are attributed to the seat the pending
// decision belongs to.
//
// concede_current_game() ends the live game through the ordinary player-loss
// path (Game::player_loses — the same one the life/deck-out state-based actions
// use), so everything downstream (GAME_RESULT, loser-on-the-play, sideboarding,
// the RL reward) behaves exactly as for any other loss. It then arms a
// COOPERATIVE UNWIND modelled on the search-server RESTORE unwind: the engine
// cannot unwind the C++ stack (-fno-exceptions), so every get_input below the
// concede hands back choice 0 — emitting no query, consuming no stdin,
// committing nothing — until control returns to the main loop, whose
// already-decided-game check exits it. Those auto-choices run under an ended
// game (state-based effects early-return, so nothing can change the winner).
//
// `whole_match` (the -3 sentinel) additionally latches match_conceded(): the
// bo3 dispatcher stops after the current game and reports the conceder's
// opponent as the match winner whatever the game score is. Outside a live game
// (between games, during sideboarding) there is nothing to concede: -2 is
// ignored with a log line, -3 still latches the match concession.
void concede_current_game(Zone::Ownership conceder, bool whole_match);
bool concede_unwind_pending();
int concede_unwind_choice();
// Drop the unwind latch. Called at both ends of a game's decision loop so the
// auto-choice mode can never leak into the between-games sideboard phase or the
// next game (each of which must emit its own queries again).
void concede_reset_unwind();
// True iff a concession sentinel was read at the most recent decision (set in
// concede_current_game). Query-and-clear. run_sideboard_phase uses it to treat
// the handed-back choice 0 as a safe no-op rather than an arbitrary swap;
// concede_clear_read_flag() drops it without reading (phase entry).
bool concede_read_this_decision();
void concede_clear_read_flag();
// Latched by a CONCEDE_MATCH sentinel; cleared at the start of every match.
bool match_conceded();
Zone::Ownership match_conceder();

struct EcsSystems {
    std::shared_ptr<Orderer> orderer;
    std::shared_ptr<StateManager> state_manager;
    std::shared_ptr<StackManager> stack_manager;
};

EcsSystems init_ecs();
int play_single_game(EcsSystems &sys, const Deck &deck_a, const Deck &deck_b,
                     bool player_a_goes_first, unsigned int seed);
void run_sideboard_phase(Deck &deck, SideboardPhaseState &st);

// Drive a full best-of-three MATCH — the single source of bo3 sequencing shared
// by main.cpp (the robomage front end) and bin/az_actor. Mirrors the tabletop
// flow: for each game it sets match_game_number, calls `before_game`, spins a
// fresh ECS, plays the game via play_single_game (Player A on the play in game 1,
// the loser on the play thereafter), records the winner, prints
// GAME_RESULT/MATCH_RESULT, calls `after_game`, then runs BOTH players' sideboard
// phases (unless the match is already decided). `deck_a`/`deck_b` are taken BY
// VALUE because sideboarding mutates them across games. `match_reset_revealed()`
// is called once at match start (the revealed accumulator spans the whole match).
// Callers must `std::srand(seed)` before calling (main.cpp does; the actor does
// per match) — the per-game engine RNG is seeded from `seed + game_num` inside.
// `before_game`/`after_game` may be empty. Returns the match winner
// (Zone::PLAYER_A or Zone::PLAYER_B).
int play_bo3_match(Deck deck_a, Deck deck_b, unsigned int seed,
                   const std::function<void(int game_num, bool a_goes_first)> &before_game = {},
                   const std::function<void(int game_num, int winner)> &after_game = {});

#endif /* GAME_DRIVER_H */
