#ifndef GAME_DRIVER_H
#define GAME_DRIVER_H

// Reusable game-driver code shared by the robomage front-end (main.cpp) and any
// other binary that needs to set up and run games in-process (e.g. az_actor).
// Everything here lives outside main.cpp so a second binary can link all engine
// objects EXCEPT obj/main.o and still have the globals, ECS setup, and per-game
// loop available.

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "classes/deck.h"
#include "classes/game.h"
#include "classes/gamestate.h"
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

struct EcsSystems {
    std::shared_ptr<Orderer> orderer;
    std::shared_ptr<StateManager> state_manager;
    std::shared_ptr<StackManager> stack_manager;
};

EcsSystems init_ecs();
int play_single_game(EcsSystems &sys, const Deck &deck_a, const Deck &deck_b,
                     bool player_a_goes_first, unsigned int seed);
void run_sideboard_phase(Deck &deck, Zone::Ownership player);

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
