#ifndef GAME_DRIVER_H
#define GAME_DRIVER_H

// Reusable game-driver code shared by the robomage front-end (main.cpp) and any
// other binary that needs to set up and run games in-process.
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
// The human seats (the C++ CLI, cli/human.h), [0] = Player A: set once the options are parsed
// (cli_args.cpp).
extern bool human_seat[2];
extern bool log_viewer_set;
extern Zone::Ownership log_viewer_owner;

extern std::string replay_file_path;
extern bool replay_mode;
extern bool machine_mode;
extern std::string deck_a_name;
extern std::string deck_b_name;
extern bool seed_override;
extern unsigned int seed_value;
extern bool no_shuffle;
extern bool offer_cast_cancel;
extern bool auto_mana;
extern bool narrative_mode;
extern bool bo3_mode;
extern bool log_decisions_flag;
extern std::string decision_log_path;
extern bool trace_visibility;
extern bool trace_handles;
extern bool trace_events;
extern bool closed_decklist;
extern std::string rmstate_dump_path;
extern bool rmstate_card_scripts;
extern bool state_dump_json;
extern std::string rmstate_read_path;
extern int knowledge_level_a;
extern int knowledge_level_b;
extern bool selfcheck_state;
extern int selfcheck_state_sample;
extern bool selfcheck_knowledge;
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
extern std::vector<std::string> hand_a_cards;
extern std::vector<std::string> hand_b_cards;
extern std::vector<std::string> library_a_cards;
extern std::vector<std::string> library_b_cards;
extern long max_decisions;
extern std::string agent_a_spec;
extern std::string agent_b_spec;
extern bool play_script_given;
extern std::string play_script_specs;
extern bool actions_script_given;
extern std::string actions_script;
extern bool trace_decisions;
extern bool game_seed_given;
extern uint64_t game_seed_value;
extern bool check_decks_flag;
extern std::vector<std::string> check_decks_dirs;

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

// True only while play_single_game's decision loop is emitting an ordinary
// priority decision — the determine_legal_actions menu. False for every other
// decision kind: pregame (keep/mulligan, bottoming, opening-hand actions),
// mandatory choices, parked mid-flow queries, blocking mid-cast / mid-resolution
// prompts, and sideboarding. The state format's encoder gates the pass flags and
// the priority-window flag on it, so the two always agree.
bool priority_window_open();
// Sets what priority_window_open answers. Only the state-format self-check uses it, to encode a
// restored snapshot as of the decision it was saved at (state_format/selfcheck.h).
void set_priority_window_open(bool open);

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
// COOPERATIVE UNWIND: the engine
// cannot unwind the C++ stack (-fno-exceptions), so every get_input below the
// concede hands back choice 0 — emitting no query, consuming no stdin,
// committing nothing — until control returns to the main loop. Those
// auto-choices run under an ended game (state-based effects early-return, so
// nothing can change the winner). A concession read at a loop-level decision (a
// parked query, a priority window) acts on nothing; at the loop top the
// conceded game's suspended state — the parked query, the resolution frame,
// the cast / activation / draw continuations, an unfinished pregame — is
// discarded and the game's decision loop exits.
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

// The deck `seat` starts the run with: its --hand-a/-b + --library-a/-b stack when a hand is given
// (the hand's cards first, then the library padded to 15 cards with copies of its last card, else
// of a basic land in the hand, else Mountains; no sideboard; the deck is named stacked_a /
// stacked_b), else the .dk file --deck-a/-b names (resources/decks/<name>.dk).
Deck load_launch_deck(Zone::Ownership seat);

// --max-decisions: ends the run at the first decision past the cap, before it is decided: prints
// "MAX_DECISIONS: stopped after <n> decisions (--max-decisions; no result)" and exits with status
// 0 (no GAME_RESULT for the unfinished game).
[[noreturn]] void end_run_at_decision_cap(size_t decisions);

// A best-of-three match that reaches this many games without a match winner is a fatal error
// (only drawn games, which count for neither player, can lengthen a match past three).
constexpr int MAX_MATCH_GAMES = 10;

// Print game `game_number`'s (1-based) machine-readable result line: "GAME_RESULT: <n> Player A
// wins" / "GAME_RESULT: <n> Player B wins", or "GAME_RESULT: <n> draw" when `winner` is neither
// player (a drawn game, CR 104.4a). Printed once per real game — bo1 and every bo3 game.
void print_game_result(int game_number, int winner);

// Drive a full best-of-three MATCH — the single source of bo3 sequencing shared
// by main.cpp (the robomage front end) and any in-process driver. Mirrors the tabletop
// flow: for each game it sets match_game_number, calls `before_game`, spins a
// fresh ECS, plays the game via play_single_game (Player A on the play in game 1,
// the loser on the play thereafter), records the winner, prints
// GAME_RESULT/MATCH_RESULT, calls `after_game`, then runs BOTH players' sideboard
// phases (unless the match is already decided). A drawn game counts for neither
// player (`after_game` gets winner 0): the match plays on until a player has won
// two games, and the player who went first in the drawn game goes first again. `deck_a`/`deck_b` are taken BY
// VALUE because sideboarding mutates them across games.
// Callers must `std::srand(seed)` before calling (main.cpp does) — the per-game
// engine RNG is seeded from `seed + game_num` inside.
// `before_game`/`after_game` may be empty. Returns the match winner
// (Zone::PLAYER_A or Zone::PLAYER_B).
int play_bo3_match(Deck deck_a, Deck deck_b, unsigned int seed,
                   const std::function<void(int game_num, bool a_goes_first)> &before_game = {},
                   const std::function<void(int game_num, int winner)> &after_game = {});

#endif /* GAME_DRIVER_H */
