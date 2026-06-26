#ifndef GAME_H
#define GAME_H

//passing Step enum to C for GUI
#define ACTION_HISTORY_SIZE 128
#define KNOWN_TOP_LIBRARY_SIZE 5

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
    FIRST_STRIKE_DAMAGE,
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
#include <map>
#include <memory>
#include <random>
#include <set>
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
    CHOOSE_ENTITY,  // Legend rule, replacement effect, choose card name, choose permanent
    ASSIGN_COMBAT_DAMAGE_CHOICE  // T3.10: attacker divides damage among 2+ blockers it can't all kill
};

struct ActionHistoryEntry {
    int category;        // ActionCategory value
    int card_vocab_idx;  // -1 for non-card entities
    bool player_a;       // true if Player A took this action
    int turn;            // cur_game.turn when the action was taken
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
        int winner = 0;  // 0=none, 1=PLAYER_A, 2=PLAYER_B (Zone::Ownership values)
        bool player_a_active = true;
        bool player_a_turn = true;
        bool player_a_has_priority = true;
        bool a_has_passed = false;
        bool b_has_passed = false;
        MandatoryChoice pending_choice = NONE;
        bool attackers_declared = false;
        bool blockers_declared = false;
        bool combat_damage_dealt = false;
        bool has_first_strikers = false;
        // T3.10: attacker -> (blocker -> damage assigned). Populated by assign_combat_damage()
        // only for attackers that required a controller choice this strike step; deal_combat_damage()
        // reads it and auto-assigns any attacker absent from the map. Cleared at handler entry
        // (per strike step) and in the END_OF_COMBAT cleanup.
        std::map<Entity, std::map<Entity, uint32_t>> combat_damage_assignment;
        std::vector<DelayedTrigger> delayed_triggers;
        std::vector<Entity> delve_exiled;   // entities exiled during current delve cast; cleared after ETB
        size_t x_paid = 0;                  // X value chosen at cast time for X-cost spells
        std::vector<Entity> remembered_entities;  // Defined$ Remembered — used by Attach sub-ability, Doomsday remember-changed
        std::map<Entity, int> ability_resolution_counts;  // Count$ResolvedThisTurn: incremented per triggered-ability resolve
        std::map<Entity, int> payment_fail_counts;  // machine mode: block casting after 2 failed payments
        bool pending_cant_be_countered = false;  // set during mana payment when Cavern restricted mana used
        bool revolt_player_a = false;  // a permanent Player A controlled left the battlefield this turn
        bool revolt_player_b = false;  // a permanent Player B controlled left the battlefield this turn
        std::set<Entity> void_countered;  // entities exiled with void counters (Dauthi Voidwalker)
        std::set<Entity> chosen_cards;  // permanents chosen/kept by a ChooseCard effect (Ajani -4); read by SacrificeAll's nonChosenCard filter, cleared by Cleanup ClearChosenCard$
        std::map<Entity, std::vector<std::string>> lk_battlefield_types;  // last-known type/subtype names of a permanent captured as it leaves the battlefield (603.10 look-back), so a "dies"/leaves trigger can match a token that has already ceased to exist by the time triggers are checked
        std::set<Entity> pending_enters_tapped;  // one-shot: a ChangeZone effect put this card onto the battlefield tapped; consumed when its Permanent is created
        std::set<Entity> pending_enters_transformed;  // one-shot: a ChangeZone effect (Transformed$ True) put this card onto the battlefield showing its DFC back face; consumed when its Permanent is created
        std::set<Entity> pending_evoked;  // one-shot: a spell cast for its evoke cost is resolving; consumed when its Permanent is created (sets Permanent::evoked)

        // Recent action history ring buffer for ML observation
        ActionHistoryEntry action_history[ACTION_HISTORY_SIZE] = {};
        int action_history_write = 0;  // next write position (circular)
        int action_history_count = 0;  // total entries written (capped at ACTION_HISTORY_SIZE)

        // Known top-of-library cards (one array per player). Index 0 is the top of the
        // library. -1 = unknown (default). Updated when a card is placed on top of a
        // library or when a card is removed from the top; cleared to all -1 on shuffle.
        int known_top_library_a[KNOWN_TOP_LIBRARY_SIZE] = {-1, -1, -1, -1, -1};
        int known_top_library_b[KNOWN_TOP_LIBRARY_SIZE] = {-1, -1, -1, -1, -1};

        void record_action(int category, int card_vocab_idx, bool player_a);
        void clear_known_top_library(bool player_a_owner);
        void known_top_library_push(bool player_a_owner, int card_vocab_idx);
        void known_top_library_remove_pos(bool player_a_owner, int pos);

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
