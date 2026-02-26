#include "state_manager.h"

#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/damage.h"
#include "../components/effect.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../error.h"
#include "../classes/game.h"
#include "../mana_system.h"
#include "../systems/orderer.h"
#include "../systems/stack_manager.h"

static Colors mana_color_for_subtype(const std::string& subtype) {
    if (subtype == "Mountain") return RED;
    if (subtype == "Forest")   return GREEN;
    if (subtype == "Plains")   return WHITE;
    if (subtype == "Island")   return BLUE;
    if (subtype == "Swamp")    return BLACK;
    return COLORLESS;
}

// Called on every SBA check. Land types can change due to effects, so this
// runs every time. Only skips adding an ability if an identical one (same
// color) is already present — does not remove stale abilities when types change.
static void apply_land_abilities() {
    for (Entity entity = 0; entity < MAX_ENTITIES; ++entity) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        //TODO make this work for tokens
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;

        auto& zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        auto& card_data = global_coordinator.GetComponent<CardData>(entity);

        bool is_land = false;
        bool is_basic = false;
        std::string land_subtype;
        for (auto& type : card_data.types) {
            if (type.kind == TYPE      && type.name == "Land")  is_land = true;
            if (type.kind == SUPERTYPE && type.name == "Basic") is_basic = true;
            if (type.kind == SUBTYPE   && (type.name == "Mountain" || type.name == "Forest" ||
                                           type.name == "Plains"   || type.name == "Island"  ||
                                           type.name == "Swamp")) {
                land_subtype = type.name;
            }
        }
        if (!is_land || !is_basic || land_subtype.empty()) continue;

        Colors required_color = mana_color_for_subtype(land_subtype);

        // Skip only if this exact color ability already exists
        bool already_present = false;
        for (auto ability_entity : card_data.abilities) {
            if (!global_coordinator.entity_has_component<Ability>(ability_entity)) continue;
            auto& ab = global_coordinator.GetComponent<Ability>(ability_entity);
            if (ab.category == "AddMana" && static_cast<Colors>(ab.amount) == required_color) {
                already_present = true;
                break;
            }
        }
        if (already_present) continue;

        Ability mana_ability;
        mana_ability.ability_type = Ability::ACTIVATED;
        mana_ability.category = "AddMana";
        mana_ability.amount = static_cast<size_t>(required_color);

        Entity ability_id = global_coordinator.CreateEntity();
        mana_ability.source = ability_id;
        global_coordinator.AddComponent(ability_id, mana_ability);
        card_data.abilities.insert(ability_id);
    }
}

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    signature.set(global_coordinator.GetComponentType<Effect>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

//layers / timestamps would be implemented here; for now order is arbitrary
void StateManager::state_based_effects(Game& game) {
    // Reset pending choice
    game.pending_choice = NONE;

    // Check for player death (0 or less life) - this ends the game immediately
    auto& player_a = global_coordinator.GetComponent<Player>(game.player_a_entity);
    auto& player_b = global_coordinator.GetComponent<Player>(game.player_b_entity);

    if (player_a.life_total <= 0) {
        printf("\nPlayer A has %d life - Player B wins!\n", player_a.life_total);
        game.ended = true;
        return;
    }
    if (player_b.life_total <= 0) {
        printf("\nPlayer B has %d life - Player A wins!\n", player_b.life_total);
        game.ended = true;
        return;
    }

    // Add Creature component to any battlefield creature that doesn't have one yet
    for (Entity entity = 0; entity < MAX_ENTITIES; ++entity) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        auto& zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        if (global_coordinator.entity_has_component<Creature>(entity)) continue;

        auto& card_data = global_coordinator.GetComponent<CardData>(entity);
        bool is_creature = false;
        for (auto& type : card_data.types) {
            if (type.kind == TYPE && type.name == "Creature") {
                is_creature = true;
                break;
            }
        }
        if (is_creature) {
            Creature creature;
            creature.power = card_data.power;
            creature.toughness = card_data.toughness;
            global_coordinator.AddComponent(entity, creature);
        }
    }

    // Apply mana abilities to basic land permanents that don't have one yet
    apply_land_abilities();

    // Check for lethal damage on creatures
    std::vector<Entity> creatures_to_destroy;
    for (Entity entity = 0; entity < MAX_ENTITIES; ++entity) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto& zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        if (!global_coordinator.entity_has_component<Damage>(entity)) continue;
        auto& creature = global_coordinator.GetComponent<Creature>(entity);
        auto& damage = global_coordinator.GetComponent<Damage>(entity);
        if (damage.damage_counters >= creature.toughness) {
            creatures_to_destroy.push_back(entity);
        }
    }

    // Move destroyed creatures to graveyard
    for (auto entity : creatures_to_destroy) {
        auto& zone = global_coordinator.GetComponent<Zone>(entity);
        auto& card_data = global_coordinator.GetComponent<CardData>(entity);
        printf("%s is destroyed (lethal damage)\n", card_data.name.c_str());

        zone.location = Zone::GRAVEYARD;

        if (global_coordinator.entity_has_component<Permanent>(entity)) {
            global_coordinator.RemoveComponent<Permanent>(entity);
        }
        global_coordinator.RemoveComponent<Creature>(entity);
        global_coordinator.RemoveComponent<Damage>(entity);

        Event death_event(Events::CREATURE_DIED);
        death_event.SetParam(Params::ENTITY, entity);
        global_coordinator.SendEvent(death_event);
    }

    // Check for mandatory choices based on game step
    // These must be resolved before priority-based actions can occur

    // Declare attackers step - active player must declare attackers
    if (game.cur_step == DECLARE_ATTACKERS && !game.attackers_declared) {
        game.pending_choice = DECLARE_ATTACKERS_CHOICE;
        return;
    }

    // Declare blockers step - defending player must declare blockers
    if (game.cur_step == DECLARE_BLOCKERS && !game.blockers_declared) {
        game.pending_choice = DECLARE_BLOCKERS_CHOICE;
        return;
    }

    // Cleanup step - check if active player exceeds maximum hand size
    if (game.cur_step == CLEANUP) {
        // TODO: Check actual hand size vs maximum hand size
        // For now, assume no discard needed
        // If hand_size > max_hand_size:
        //     game.pending_choice = CLEANUP_DISCARD;
        //     return;
    }

    // TODO: Check for legend rule violations
    // If multiple legendary permanents with same name exist:
    //     game.pending_choice = CHOOSE_ENTITY;
    //     return;

    // Process continuous effects
    for (auto &&entity : mEntities) {
        // if we are dealing with an effect
        if (global_coordinator.entity_has_component<Effect>(entity)) {
            // TODO: Apply continuous effects based on layers/timestamps
        }
    }
}

std::vector<LegalAction> StateManager::determine_legal_actions(const Game& game,
                                                               std::shared_ptr<Orderer> orderer,
                                                               std::shared_ptr<StackManager> stack_manager) {
    std::vector<LegalAction> actions;

    // Determine whose turn/priority it is
    Zone::Ownership priority_player = game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity priority_player_entity = get_player_entity(priority_player);

    // Pass priority is always legal - because mandatory decisions happen in a different function
    actions.push_back(LegalAction(PASS_PRIORITY, "Pass priority"));

    // Check for special action: play land
    if ((game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
        game.player_a_turn == game.player_a_has_priority &&  // It's their turn
        global_coordinator.entity_has_component<Player>(priority_player_entity)) {

        auto& player = global_coordinator.GetComponent<Player>(priority_player_entity);

        if (player.lands_played_this_turn == 0) {
            // Check hand for lands
            auto hand = orderer->get_hand(priority_player);
            for (auto card_entity : hand) {
                auto& card_data = global_coordinator.GetComponent<CardData>(card_entity);
                bool is_land = false;
                for (auto& type : card_data.types) {
                    if (type.kind == TYPE && type.name == "Land") {
                        is_land = true;
                        break;
                    }
                }
                if (is_land) {
                    std::string desc = "Play " + card_data.name;
                    actions.push_back(LegalAction(SPECIAL_ACTION, card_entity, desc));
                }
            }
        }
    }

    // checking for spells to cast from hand
    // TODO spells cast from elsewhere
    bool stack_empty = stack_manager->is_empty();

    auto hand = orderer->get_hand(priority_player);
    for (auto card_entity : hand) {
        auto& card_data = global_coordinator.GetComponent<CardData>(card_entity);

        // Check if it's a spell (not a land)
        bool is_instant = false, is_sorcery = false, is_creature = false;
        for (auto& type : card_data.types) {
            if (type.kind == TYPE) {
                if (type.name == "Instant") {
                    is_instant = true;
                } else if (type.name == "Sorcery") {
                    is_sorcery = true;
                } else if (type.name == "Creature") {
                    is_creature = true;
                }
            }
        }

        // Timing restrictions
        bool can_cast_now = false;
        if (is_instant) {
            can_cast_now = true;  // Can cast anytime you have priority
        } else if (is_sorcery || is_creature) {
            // Sorcery speed: main phase, your turn, stack empty
            can_cast_now = (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                           (game.player_a_turn == game.player_a_has_priority) && stack_empty;
        }

        if (can_cast_now && can_afford(priority_player, card_data.mana_cost)) {
            std::string desc = "Cast " + card_data.name;
            actions.push_back(LegalAction(CAST_SPELL, card_entity, desc));
        }
    }

    //checking permanents for activated abilities
    //TODO timing restrictions 
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;

        auto& zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        auto& permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != priority_player) continue;
        if (permanent.is_tapped) continue;  // Can't tap already-tapped permanent

        // Check for mana abilities
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto& card_data = global_coordinator.GetComponent<CardData>(entity);
            for (auto ability_entity : card_data.abilities) {
                auto& ability = global_coordinator.GetComponent<Ability>(ability_entity);
                if (ability.category == "AddMana" && ability.ability_type == Ability::ACTIVATED) {
                    Colors mana_color = static_cast<Colors>(ability.amount);
                    std::string mana_symbol;
                    switch (mana_color) {
                        case WHITE:
                            mana_symbol = "W";
                            break;
                        case BLUE:
                            mana_symbol = "U";
                            break;
                        case BLACK:
                            mana_symbol = "B";
                            break;
                        case RED:
                            mana_symbol = "R";
                            break;
                        case GREEN:
                            mana_symbol = "G";
                            break;
                        case COLORLESS:
                            mana_symbol = "C";
                            break;
                        default:
                            mana_symbol = "?";
                            break;
                    }
                    std::string desc = "Tap " + card_data.name + " for {" + mana_symbol + "}";
                    actions.push_back(LegalAction(ACTIVATE_ABILITY, entity, ability_entity, desc));
                }
            }
        }
    } 

    return actions;
}
