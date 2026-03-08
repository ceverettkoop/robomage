#include "state_manager.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../classes/game.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/creature.h"
#include "../components/static_ability.h"
#include "../components/damage.h"
#include "../components/effect.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../mana_system.h"
#include "../systems/stack_manager.h"
#include "orderer.h"

static Colors mana_color_for_subtype(const std::string &subtype) {
    if (subtype == "Mountain") return RED;
    if (subtype == "Forest") return GREEN;
    if (subtype == "Plains") return WHITE;
    if (subtype == "Island") return BLUE;
    if (subtype == "Swamp") return BLACK;
    if (subtype == "Wastes") return COLORLESS;
    return NO_COLOR;
}

static bool check_delirium(Zone::Ownership owner, const std::set<Entity> &entities) {
    std::set<std::string> type_names;
    for (auto entity : entities) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        if (z.location != Zone::GRAVEYARD || z.owner != owner) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        for (auto &t : global_coordinator.GetComponent<CardData>(entity).types)
            if (t.kind == TYPE) type_names.insert(t.name);
    }
    return type_names.size() >= 4;
}

// Permanents on battlefield set to have appropriate components
// if they are in a different zone these are removed as no longer applicable
void StateManager::apply_permanent_components(Game &game) {
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        // TODO handle tokens w/o carddata
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location == Zone::BATTLEFIELD) {  // on battlefield, check to add components
            // check types
            // TODO planeswalker here
            bool is_creature = false;
            bool is_land = false;
            auto &card_data = global_coordinator.GetComponent<CardData>(entity);
            for (auto &t : card_data.types) {
                if (t.kind == TYPE && t.name == "Creature") {
                    is_creature = true;
                }  // can be creature and land
                if (t.kind == TYPE && t.name == "Land") {
                    is_land = true;
                }
            }
            // providing permanent component if doesn't have
            if (!global_coordinator.entity_has_component<Permanent>(entity)) {
                Permanent perm;
                perm.controller = zone.controller;
                perm.has_summoning_sickness = is_creature;
                perm.is_tapped = false;
                perm.timestamp_entered_battlefield = game.timestamp++;
                global_coordinator.AddComponent(entity, perm);
                if (is_creature) {
                    Event etb(Events::CREATURE_ENTERED);
                    etb.SetParam(Params::ENTITY, entity);
                    global_coordinator.SendEvent(etb);
                }
            }
            // copy activated abilities from card_data to permanent; incl mana abilities altough mana abilities innate to basic land types
            // added elsewhere
            for (auto ab : card_data.abilities) {
                if (ab.ability_type != Ability::ACTIVATED) continue;
                auto &perm_abilities = global_coordinator.GetComponent<Permanent>(entity).abilities;
                bool already_present = false;
                for (auto &existing : perm_abilities) {
                    if (existing.identical_activated_ability(ab)) {
                        already_present = true;
                        break;
                    }
                }
                if (already_present) continue;
                ab.source = entity;
                perm_abilities.push_back(ab);
            }

            // copy static abilities from card_data to permanent (applied = false by default)
            if (global_coordinator.GetComponent<Permanent>(entity).static_abilities.empty() &&
                !card_data.static_abilities.empty()) {
                auto &perm_sa = global_coordinator.GetComponent<Permanent>(entity).static_abilities;
                for (auto &sa : card_data.static_abilities)
                    perm_sa.push_back(sa);
            }

            // providing creature related components if applicable
            if (is_creature && !global_coordinator.entity_has_component<Creature>(entity)) {
                Creature creature;
                creature.power = card_data.power;
                creature.toughness = card_data.toughness;
                creature.keywords = card_data.keywords;
                global_coordinator.AddComponent(entity, creature);
                // damage component
                Damage damage;
                damage.damage_counters = 0;
                global_coordinator.AddComponent(entity, damage);
            }
            if (is_land) {
                apply_land_abilities(entity);
            }

        } else {  // off battlefield, check to remove
            if (global_coordinator.entity_has_component<Permanent>(entity)) {
                global_coordinator.RemoveComponent<Permanent>(entity);
            }
            if (global_coordinator.entity_has_component<Creature>(entity)) {
                global_coordinator.RemoveComponent<Creature>(entity);
            }
            if (global_coordinator.entity_has_component<Damage>(entity)) {
                global_coordinator.RemoveComponent<Damage>(entity);
            }
        }
    }
}

// Applies abilities to lands based on the land subtypes
// TODO recheck in case blood moon or similar nuked one; rn blood moon on a tundra would just add an ability, even if
// types are successfully replaced
void StateManager::apply_land_abilities(Entity entity) {
    // assumes called with entity that has permanent component and is on battlefield and is land
    auto &card_data = global_coordinator.GetComponent<CardData>(entity);
    std::vector<std::string> land_subtypes;
    for (auto &type : card_data.types) {
        if (type.kind == SUBTYPE && (type.name == "Mountain" || type.name == "Forest" || type.name == "Plains" ||
                                        type.name == "Island" || type.name == "Swamp" || type.name == "Wastes")) {
            land_subtypes.push_back(type.name);
        }
    }
    if (land_subtypes.empty()) return;
    // find mana abilities for corresponding land subtype
    for (auto subtype : land_subtypes) {
        Colors required_color = mana_color_for_subtype(subtype);
        if (required_color == NO_COLOR) continue;

        // Skip only if this exact color ability already exists
        auto &perm_abilities = global_coordinator.GetComponent<Permanent>(entity).abilities;
        bool already_present = false;
        for (auto ab : perm_abilities) {
            if (ab.category == "AddMana" && ab.color == required_color && ab.amount == 1) {
                already_present = true;
                break;
            }
        }
        if (already_present) continue;

        Ability mana_ability;
        mana_ability.ability_type = Ability::ACTIVATED;
        mana_ability.category = "AddMana";
        mana_ability.color = required_color;
        mana_ability.amount = 1;
        mana_ability.tap_cost = true;

        mana_ability.source = entity;
        perm_abilities.push_back(mana_ability);
    }
}

void StateManager::apply_static_ability_effects() {
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;

        auto &perm = global_coordinator.GetComponent<Permanent>(entity);

        // Front-face static abilities don't apply while transformed
        if (perm.transformed) {
            for (auto &sa : perm.static_abilities)
                sa.applied = false;
            continue;
        }

        bool delirium_now = check_delirium(perm.controller, mEntities);

        for (auto &sa : perm.static_abilities) {
            bool condition_met = false;
            if (sa.condition.empty()) {
                condition_met = true;
            } else if (sa.condition == "Delirium") {
                condition_met = delirium_now;
            }

            if (sa.category == "Continuous") {
                if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
                auto &cr = global_coordinator.GetComponent<Creature>(entity);

                if (condition_met && !sa.applied) {
                    if (sa.add_power != 0)     cr.power     += static_cast<uint32_t>(sa.add_power);
                    if (sa.add_toughness != 0) cr.toughness += static_cast<uint32_t>(sa.add_toughness);
                    if (!sa.add_keyword.empty()) cr.keywords.push_back(sa.add_keyword);
                    sa.applied = true;
                    auto &cd = global_coordinator.GetComponent<CardData>(entity);
                    game_log("%s gains %s%s%s(Delirium)\n", cd.name.c_str(),
                           sa.add_power != 0 ? (std::to_string(sa.add_power) + "/" + std::to_string(sa.add_toughness) + " ").c_str() : "",
                           !sa.add_keyword.empty() ? (sa.add_keyword + " ").c_str() : "",
                           !sa.condition.empty() ? "" : "");
                } else if (!condition_met && sa.applied) {
                    if (sa.add_power != 0)     cr.power     -= static_cast<uint32_t>(sa.add_power);
                    if (sa.add_toughness != 0) cr.toughness -= static_cast<uint32_t>(sa.add_toughness);
                    if (!sa.add_keyword.empty()) {
                        auto it = std::find(cr.keywords.begin(), cr.keywords.end(), sa.add_keyword);
                        if (it != cr.keywords.end()) cr.keywords.erase(it);
                    }
                    sa.applied = false;
                    auto &cd = global_coordinator.GetComponent<CardData>(entity);
                    game_log("%s loses Delirium bonus\n", cd.name.c_str());
                }
            }
            // MustAttack: enforcement deferred (future work)
        }
    }
}

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

// layers / timestamps would be implemented here; for now order is arbitrary
// TODO clean this up - more concise code
void StateManager::state_based_effects(Game &game, std::shared_ptr<Orderer> orderer) {
    // Reset pending choice
    game.pending_choice = NONE;

    // Check for player death (0 or less life) - this ends the game immediately
    auto &player_a = global_coordinator.GetComponent<Player>(game.player_a_entity);
    auto &player_b = global_coordinator.GetComponent<Player>(game.player_b_entity);

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
    // Add/remove Permanent component as entities enter/leave the battlefield
    apply_permanent_components(game);

    // Apply/revert static ability effects (e.g. Delirium bonuses)
    apply_static_ability_effects();

    // Check for lethal damage on creatures
    std::vector<Entity> creatures_to_destroy;
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        if (!global_coordinator.entity_has_component<Damage>(entity)) continue;
        auto &creature = global_coordinator.GetComponent<Creature>(entity);
        auto &damage = global_coordinator.GetComponent<Damage>(entity);
        if (damage.damage_counters >= creature.toughness) {
            creatures_to_destroy.push_back(entity);
        }
    }

    // Move destroyed creatures to graveyard
    for (auto entity : creatures_to_destroy) {
        auto &card_data = global_coordinator.GetComponent<CardData>(entity);
        game_log("%s is destroyed (lethal damage)\n", card_data.name.c_str());

        orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
        // components will be removed by state based effects
        Event death_event(Events::CREATURE_DIED);
        death_event.SetParam(Params::ENTITY, entity);
        global_coordinator.SendEvent(death_event);
    }

    // Deal combat damage
    if (game.cur_step == COMBAT_DAMAGE && !game.combat_damage_dealt) {
        game_log("\n--- Combat Damage ---\n");

        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            if (!cr.is_attacking) continue;

            auto &cd = global_coordinator.GetComponent<CardData>(entity);

            // Collect blockers for this attacker
            std::vector<Entity> blockers;
            for (Entity b = 0; b < MAX_ENTITIES; ++b) {
                if (!global_coordinator.entity_has_component<Creature>(b)) continue;
                auto &bcr = global_coordinator.GetComponent<Creature>(b);
                if (bcr.is_blocking && bcr.blocking_target == entity) {
                    blockers.push_back(b);
                }
            }

            if (blockers.empty()) {
                // Unblocked — deal damage to attack target
                uint32_t dmg = cr.power;
                if (dmg > 0) {
                    deal_damage(entity, cr.attack_target, dmg);
                    if (global_coordinator.entity_has_component<Player>(cr.attack_target)) {
                        auto &target_player = global_coordinator.GetComponent<Player>(cr.attack_target);
                        target_player.life_total -= static_cast<int>(dmg);
                        const char *tname = (cr.attack_target == game.player_a_entity) ? "Player A" : "Player B";
                        game_log("  %s deals %u damage to %s\n", cd.name.c_str(), dmg, tname);
                    }
                }
            } else {
                // Blocked — assign damage to blockers in order, blockers deal damage back
                uint32_t remaining = cr.power;
                for (auto blocker : blockers) {
                    auto &bcr = global_coordinator.GetComponent<Creature>(blocker);
                    auto &bcd = global_coordinator.GetComponent<CardData>(blocker);

                    // Blocker deals damage to attacker
                    if (bcr.power > 0) {
                        deal_damage(blocker, entity, bcr.power);
                        game_log("  %s deals %u damage to %s\n", bcd.name.c_str(), bcr.power, cd.name.c_str());
                    }

                    // Attacker deals damage to blocker (lethal to each in order, overflow to next)
                    if (remaining > 0) {
                        uint32_t assigned = (remaining >= bcr.toughness) ? bcr.toughness : remaining;
                        deal_damage(entity, blocker, assigned);
                        game_log("  %s deals %u damage to %s\n", cd.name.c_str(), assigned, bcd.name.c_str());
                        remaining -= assigned;
                    }
                }
            }
        }

        game.combat_damage_dealt = true;
        game_log("--- End Combat Damage ---\n\n");
        return;  // Re-enter loop so SBAs process deaths from damage
    }

    // Check for mandatory choices based on game step
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
    // Cleanup step - active player must discard down to 7 cards
    if (game.cur_step == CLEANUP) {
        Zone::Ownership active_player = game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
        size_t hand_size = 0;
        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            if (zone.location == Zone::HAND && zone.owner == active_player) hand_size++;
        }
        if (hand_size > 7) {
            game.pending_choice = CLEANUP_DISCARD;
            return;
        }
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

    check_triggered_abilities(game, orderer);
}

// Drains all buffered events since the last call and puts any triggered abilities
// from battlefield permanents whose trigger condition matches onto the stack.
void StateManager::check_triggered_abilities(Game &game, std::shared_ptr<Orderer> orderer) {
    auto events = global_coordinator.drain_pending_events();
    if (events.empty()) return;

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        if (!global_coordinator.entity_has_component<CardData>(entity)) continue;

        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        auto &card_data = global_coordinator.GetComponent<CardData>(entity);

        for (const auto &ev : events) {
            for (const auto &ab : card_data.abilities) {
                if (ab.ability_type != Ability::TRIGGERED) continue;
                if (ab.trigger_on == 0 || ab.trigger_on != ev.GetType()) continue;
                // "another" check: skip if the entering entity is the source itself
                if (ab.trigger_self_excluded && ev.HasParam(Params::ENTITY) &&
                    ev.GetParam<Entity>(Params::ENTITY) == entity) continue;
                // Don't fire front-face triggers on a transformed permanent
                if (perm.transformed) continue;
                // ValidPlayer$ You: only fire when the active player is the permanent's controller
                if (ab.trigger_valid_player_is_controller && ev.HasParam(Params::PLAYER)) {
                    Entity event_player = ev.GetParam<Entity>(Params::PLAYER);
                    Entity ctrl_entity = (perm.controller == Zone::PLAYER_A)
                                         ? game.player_a_entity : game.player_b_entity;
                    if (event_player != ctrl_entity) continue;
                }

                // Push the triggered ability onto the stack as a standalone entity
                Entity trigger_entity = global_coordinator.CreateEntity();
                Zone ab_zone(Zone::HAND, perm.controller, perm.controller);
                global_coordinator.AddComponent(trigger_entity, ab_zone);
                orderer->add_to_zone(false, trigger_entity, Zone::STACK);

                Ability trigger_ab = ab;
                trigger_ab.source = entity;
                global_coordinator.AddComponent(trigger_entity, trigger_ab);

                game_log("%s triggered\n", card_data.name.c_str());
            }
        }
    }
}

static bool can_afford_alt(const AltCost& alt_cost, Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer) {
    if (!alt_cost.has_alt_cost) return false;

    if (alt_cost.return_to_hand_count > 0) {
        int matching = 0;
        const std::string& sub = alt_cost.return_to_hand_type;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            auto& perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != priority_player) continue;
            auto& cd2 = global_coordinator.GetComponent<CardData>(e);
            for (auto& t : cd2.types) {
                if (t.kind == SUBTYPE && t.name == sub) { matching++; break; }
            }
        }
        return matching >= alt_cost.return_to_hand_count;
    }

    if (alt_cost.life_cost > 0) {
        Entity pp_entity = (priority_player == Zone::PLAYER_A)
            ? cur_game.player_a_entity : cur_game.player_b_entity;
        if (global_coordinator.GetComponent<Player>(pp_entity).life_total < alt_cost.life_cost)
            return false;
    }

    if (alt_cost.exile_blue_from_hand > 0) {
        bool has_blue = false;
        for (auto e : orderer->get_hand(priority_player)) {
            if (e == card_entity) continue;
            if (global_coordinator.entity_has_component<ColorIdentity>(e) &&
                global_coordinator.GetComponent<ColorIdentity>(e).colors.count(BLUE)) {
                has_blue = true; break;
            }
        }
        if (!has_blue) return false;
    }

    return true;
}

std::vector<LegalAction> StateManager::determine_legal_actions(
    const Game &game, std::shared_ptr<Orderer> orderer, std::shared_ptr<StackManager> stack_manager) {
    std::vector<LegalAction> actions;          // return value
    std::vector<LegalAction> pending_actions;  // non mana-ability actions possible if costs could be paid; used to
                                               // check what mana abilities can be rationally activated

    // Determine whose turn/priority it is
    Zone::Ownership priority_player = game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    Entity priority_player_entity = get_player_entity(priority_player);

    // PASS PRIORITY
    LegalAction la(PASS_PRIORITY, "Pass priority");
    la.category = ActionCategory::PASS_PRIORITY;
    actions.push_back(la);

    // LAND FROM HAND
    if ((game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
        game.player_a_turn == game.player_a_has_priority &&
        global_coordinator.entity_has_component<Player>(priority_player_entity)) {
        auto &player = global_coordinator.GetComponent<Player>(priority_player_entity);

        if (player.lands_played_this_turn == 0) {
            // Check hand for lands
            auto hand = orderer->get_hand(priority_player);
            for (auto card_entity : hand) {
                auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
                bool is_land = false;
                for (auto &type : card_data.types) {
                    if (type.kind == TYPE && type.name == "Land") {
                        is_land = true;
                        break;
                    }
                }
                if (is_land) {
                    std::string desc = "Play " + card_data.name;
                    LegalAction la(SPECIAL_ACTION, card_entity, desc);
                    la.category = ActionCategory::PLAY_LAND;
                    actions.push_back(la);
                }
            }
        }
    }
    // checking for spells to cast from hand
    // TODO spells cast from elsewhere
    bool stack_empty = stack_manager->is_empty();
    auto hand = orderer->get_hand(priority_player);
    for (auto card_entity : hand) {
        auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
        // TODO handle flash
        bool is_instant = false;
        bool is_land = false;
        for (auto &type : card_data.types) {
            if (type.kind == TYPE) {
                if (type.name == "Instant") {
                    is_instant = true;
                } else if (type.name == "Land") {
                    is_land = true;  // can't cast land
                    break;
                }
            }
        }
        if (is_land) continue;
        // Timing restrictions
        bool can_cast_now = false;
        if (is_instant) {
            can_cast_now = true;  // cast anytime you have priority... TODO handle edge cases
        } else {
            // Sorcery speed: main phase, your turn, stack empty
            can_cast_now = (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                           (game.player_a_turn == game.player_a_has_priority) && stack_empty;
        }
        // Check that at least one legal target exists for any targeting requirement
        bool tgt_ok = true;
        for (const auto &ab : card_data.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            break;
        }
        if (can_cast_now && tgt_ok) {
            std::string desc = "Cast " + card_data.name;
            LegalAction la(CAST_SPELL, card_entity, desc);
            la.category = ActionCategory::CAST_SPELL;

            bool can_regular = can_afford(priority_player, card_data.mana_cost);

            bool can_alt = can_afford_alt(card_data.alt_cost, priority_player, card_entity, orderer);

            if (can_regular) actions.push_back(la);
            if (can_alt) {
                LegalAction alt_la = la;
                alt_la.use_alt_cost = true;
                alt_la.description = "Cast " + card_data.name + " (alternate cost)";
                actions.push_back(alt_la);
            }
            if (!can_regular && !can_alt) pending_actions.push_back(la);
        }
    }
    // checking permanents for activated abilities
    // mana abilities parsed last, after pending_actions complete
    std::vector<LegalAction> legal_mana_abilities;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != priority_player) continue;
        for (auto ab : permanent.abilities) {
            if (ab.ability_type != Ability::ACTIVATED) continue;
            // todo handle this elswewhere, tapping check
            if (ab.tap_cost && permanent.is_tapped) continue;
            if (ab.category == "AddMana") {  //
                auto &card_data = global_coordinator.GetComponent<CardData>(ab.source);
                std::string desc = "Tap " + card_data.name + " for {" + mana_symbol(ab.color) + "}";
                LegalAction la(ACTIVATE_ABILITY, ab.source, ab, desc);
                switch (ab.color) {
                    case WHITE:
                        la.category = ActionCategory::MANA_W;
                        break;
                    case BLUE:
                        la.category = ActionCategory::MANA_U;
                        break;
                    case BLACK:
                        la.category = ActionCategory::MANA_B;
                        break;
                    case RED:
                        la.category = ActionCategory::MANA_R;
                        break;
                    case GREEN:
                        la.category = ActionCategory::MANA_G;
                        break;
                    default:
                        la.category = ActionCategory::MANA_C;
                        break;
                }
                legal_mana_abilities.push_back(la);
                continue;
            } else {
                // Non-mana activated ability (e.g. ChangeZone for fetch lands, Destroy for Wasteland)
                if (ab.valid_tgts != "N_A" && !has_legal_targets(ab, orderer)) continue;
                auto &ab_card_data = global_coordinator.GetComponent<CardData>(ab.source);
                std::string desc = "Activate " + ab_card_data.name + " (" + ab.category + ")";
                LegalAction non_mana_la(ACTIVATE_ABILITY, ab.source, ab, desc);
                non_mana_la.category = ActionCategory::ACTIVATE_ABILITY;
                actions.push_back(non_mana_la);
            }
        }
    }
    // not filtering mana abilities based on if they contribute to a spell- will revisit this if it makes ML harder
    /*
    for (auto &ma : useful_mana_abilities(legal_mana_abilities, pending_actions)) {
        actions.push_back(ma);
    }
    */
    for (auto &ma : legal_mana_abilities) {
        actions.push_back(ma);
    }
    return actions;
}