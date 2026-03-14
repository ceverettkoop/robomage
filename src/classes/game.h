#ifndef GAME_H
#define GAME_H

//passing Step enum to C for GUI
#define ACTION_HISTORY_SIZE 15

#ifdef __cplusplus
extern "C" {
#endif

typedef enum Step {
    UNTAP,
    UPKEEP,
    DRAW,
    FIRST_MAIN,
    BEGIN_COMBAT,
    DECLARE_ATTACKERS,
    DECLARE_BLOCKERS,
    COMBAT_DAMAGE,
    END_OF_COMBAT,
    SECOND_MAIN,
    END_STEP,
    CLEANUP
}Step;

#ifdef __cplusplus
}  // end extern "C"
#endif

#ifdef __cplusplus

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "../ecs/entity.h"
#include "../components/ability.h"

struct Deck;
struct Game;
struct DelayedTrigger;

extern Game cur_game;

struct DelayedTrigger {
    Ability ability;        // what to push onto the stack when it fires
    uint32_t fire_on;       // event ID (e.g. Events::UPKEEP_BEGAN)
    Entity owner_entity;    // player entity who controls it
    size_t fire_on_turn;    // game.turn value at which to fire (cur_game.turn + 1 at registration)
};

enum MandatoryChoice {
    NONE,
    DECLARE_ATTACKERS_CHOICE,
    DECLARE_BLOCKERS_CHOICE,
    CLEANUP_DISCARD,
    CHOOSE_ENTITY  // Legend rule, replacement effect, choose card name, choose permanent
};

struct ActionHistoryEntry {
    int category;        // ActionCategory value
    int card_vocab_idx;  // -1 for non-card entities
    bool player_a;       // true if Player A took this action
};

struct Game {
        Game() {};
        Game(size_t _seed) {
            seed = _seed;
            gen = std::mt19937(seed);
        };
        size_t seed;
        size_t timestamp = 0;
        size_t turn = 0;
        Step cur_step = UNTAP;
        Entity player_a_entity;
        Entity player_b_entity;
        std::mt19937 gen;
        bool ended = false;
        bool player_a_active = true;
        bool player_a_turn = true;
        bool player_a_has_priority = true;
        bool a_has_passed = false;
        bool b_has_passed = false;
        MandatoryChoice pending_choice = NONE;
        bool attackers_declared = false;
        bool blockers_declared = false;
        bool combat_damage_dealt = false;
        std::vector<DelayedTrigger> delayed_triggers;
        std::vector<Entity> delve_exiled;   // entities exiled during current delve cast; cleared after ETB
        Entity remembered_entity = 0;       // Defined$ Remembered — used by Attach sub-ability

        // Recent action history ring buffer for ML observation
        ActionHistoryEntry action_history[ACTION_HISTORY_SIZE] = {};
        int action_history_write = 0;  // next write position (circular)
        int action_history_count = 0;  // total entries written (capped at ACTION_HISTORY_SIZE)

        // Starting decklist snapshots for ML observation (count per card vocab slot).
        // Size 128 must equal N_CARD_TYPES in machine_io.h.
        int starting_decklist_a[128] = {};
        int starting_decklist_b[128] = {};
        void record_action(int category, int card_vocab_idx, bool player_a);

        bool ready_to_resolve();
        bool is_mandatory_choice_pending() const;
        void generate_players(const Deck &deck_a, const Deck &deck_b);
        bool advance_step(std::shared_ptr<class StackManager> stack_manager, std::shared_ptr<class Orderer> orderer);
        void pass_priority();
        void take_action();  // resets last_player_passed since an action was taken

    private:
        Entity gen_player(const Deck &deck);
};

#endif // __cplusplus

#endif /* GAME_H */
