#ifndef STATE_MANAGER_H
#define STATE_MANAGER_H

#include "../ecs/system.h"
#include "../ecs/entity.h"
#include "../components/zone.h"
#include "../components/static_ability.h"
#include "../classes/action.h"
#include <vector>
#include <string>
#include <memory>

struct Deck;
struct Game;

class Orderer;
class StackManager;

// Cached snapshot of an active static ability on the battlefield.
// Rebuilt each SBE pass by apply_static_ability_effects(); queried by
// determine_legal_actions, check_triggered_abilities, mana_system, game.cpp untap, etc.
struct ActiveStatic {
    Entity            entity = 0;
    StaticAbility    *sa = nullptr;
    Zone::Ownership   controller = Zone::PLAYER_A;
};

// Global cached list of active static abilities on battlefield permanents.
// Rebuilt every SBE pass. Consumers read this instead of scanning all permanents.
extern std::vector<ActiveStatic> g_active_statics;

class StateManager : public System {

public:
    static void init();
    void state_based_effects(Game& game, std::shared_ptr<Orderer> orderer);
    void process_turn_based_actions(Game& game, std::shared_ptr<Orderer> orderer);
    std::vector<LegalAction> determine_legal_actions(const Game& game, std::shared_ptr<Orderer> orderer,
                                                      std::shared_ptr<StackManager> stack_manager);

private:
    void apply_permanent_components(Game& game);
    void apply_land_abilities(Entity entity);
    void apply_keyword_abilities(Entity entity);
    void apply_type_changing_effects();
    void recompute_battlefield_pt();
    void deal_combat_damage(Game& game, bool first_strike_only);
    void check_triggered_abilities(Game& game, std::shared_ptr<Orderer> orderer);
    void apply_static_ability_effects();
};

#endif /* STATE_MANAGER_H */
