#include "mana_system.h"

#include <cstddef>
#include <cstdio>
#include <tuple>

#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "components/carddata.h"
#include "components/creature.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/static_ability.h"
#include "components/types.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "input_logger.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static const char *mana_symbol_str(Colors color);
static bool is_simple_mana_source(const Ability &ab);
static bool permanent_cant_activate(Entity entity, std::shared_ptr<Orderer> orderer);
static size_t eval_mana_amount(const Ability &ab, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer);

Entity get_player_entity(Zone::Ownership player) {
    return (player == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
}

bool can_afford_pool(const std::multiset<Colors> &pool, const std::multiset<Colors> &cost) {
    auto remaining = pool;

    // Pay specific colors first
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = remaining.find(color);
        if (it != remaining.end()) {
            remaining.erase(it);
        } else {
            return false;
        }
    }

    size_t generic_needed = cost.count(GENERIC);
    return remaining.size() >= generic_needed;
}

bool can_afford(Zone::Ownership player_owner, const std::multiset<Colors> &cost) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) {
        return false;
    }
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    return can_afford_pool(player.mana, cost);
}

void spend_mana(Zone::Ownership player_owner, const std::multiset<Colors> &cost, Entity paid_for) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    if (!can_afford_pool(player.mana, cost)) {
        non_fatal_error("spend_mana called with insufficient mana in pool");
        #ifndef NDEBUG
        dump_entity(paid_for);
        #endif
    }

    // Pay specific colors first
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = player.mana.find(color);
        if (it != player.mana.end()) {
            player.mana.erase(it);
        }
    }

    // Pay generic with any remaining mana
    size_t generic_needed = cost.count(GENERIC);
    for (size_t i = 0; i < generic_needed; i++) {
        if (!player.mana.empty()) {
            auto it = player.mana.begin();
            player.mana.erase(it);
        }
    }
}

void add_mana(Zone::Ownership player_owner, Colors mana_color, size_t amount) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    for (size_t i = 0; i < amount; i++) {
        player.mana.insert(mana_color);
    }
}

ManaValue pay_partial(Zone::Ownership player_owner, const ManaValue &cost) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    ManaValue remaining;

    // Pay specific colors first
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = player.mana.find(color);
        if (it != player.mana.end()) {
            player.mana.erase(it);
        } else {
            remaining.insert(color);
        }
    }

    // Pay generic with any remaining mana
    size_t generic_needed = cost.count(GENERIC);
    for (size_t i = 0; i < generic_needed; i++) {
        if (!player.mana.empty()) {
            auto it = player.mana.begin();
            player.mana.erase(it);
        } else {
            remaining.insert(GENERIC);
        }
    }

    return remaining;
}

void empty_mana_pool(Zone::Ownership player_owner) {
    Entity player_entity = get_player_entity(player_owner);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);
    if (!player.mana.empty() && InputLogger::instance().is_machine_mode()) {
        // Signal the Python env to apply a shaping penalty to the model if this
        // player's side is the one being trained.
        const char* side = (player_owner == Zone::PLAYER_A) ? "A" : "B";
        printf("MANA_WASTED: %s\n", side);
        fflush(stdout);
    }
    player.mana.clear();
}

static const char *mana_symbol_str(Colors color) {
    switch (color) {
        case WHITE:    return "W";
        case BLUE:     return "U";
        case BLACK:    return "B";
        case RED:      return "R";
        case GREEN:    return "G";
        case COLORLESS: return "C";
        default:       return "?";
    }
}

// A mana ability is "simple" if it only requires tapping (no life, sac, return, or mana cost)
static bool is_simple_mana_source(const Ability &ab) {
    return ab.category == "AddMana" &&
           ab.ability_type == Ability::ACTIVATED &&
           ab.tap_cost &&
           ab.life_cost == 0 &&
           !ab.sac_self &&
           ab.sac_cost_spec.empty() &&
           ab.return_cost_type.empty() &&
           ab.activation_mana_cost.empty();
}

// Check if a permanent's abilities are suppressed by a CantBeActivated static
static bool permanent_cant_activate(Entity entity, std::shared_ptr<Orderer> orderer) {
    auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
    for (auto e2 : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(e2)) continue;
        auto &z2 = global_coordinator.GetComponent<Zone>(e2);
        if (z2.location != Zone::BATTLEFIELD) continue;
        auto &p2 = global_coordinator.GetComponent<Permanent>(e2);
        if (p2.is_phased_out) continue;
        for (auto &sa : p2.static_abilities) {
            if (sa.category != "CantBeActivated" || sa.cant_activate_card_filter.empty()) continue;
            if (sa.cant_activate_card_filter == "Artifact") {
                for (auto &t : permanent.types)
                    if (t.kind == TYPE && t.name == "Artifact") return true;
            }
        }
    }
    return false;
}

// Evaluate the mana amount a source produces (handles dynamic amounts like Gaea's Cradle)
static size_t eval_mana_amount(const Ability &ab, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer) {
    if (!ab.dynamic_amount_expr.empty() &&
        ab.dynamic_amount_expr.find("Count$Valid Creature.YouCtrl") != std::string::npos) {
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (!global_coordinator.entity_has_component<Creature>(e)) continue;
            auto &sz = global_coordinator.GetComponent<Zone>(e);
            if (sz.location != Zone::BATTLEFIELD) continue;
            if (global_coordinator.GetComponent<Permanent>(e).controller == controller) count++;
        }
        return count;
    }
    return ab.amount;
}

// Collect all mana abilities a player could activate (untapped, simple tap-only sources)
static std::vector<std::pair<Entity, Ability>> collect_available_mana_sources(
    Zone::Ownership player, std::shared_ptr<Orderer> orderer) {
    std::vector<std::pair<Entity, Ability>> sources;
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != player) continue;
        if (permanent.is_tapped) continue;
        if (permanent.is_phased_out) continue;
        if (permanent_cant_activate(entity, orderer)) continue;
        // Summoning sickness check for creatures with tap cost
        if (permanent.has_summoning_sickness &&
            global_coordinator.entity_has_component<Creature>(entity)) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            bool has_haste = false;
            for (const auto &kw : cr.keywords)
                if (kw == "Haste") { has_haste = true; break; }
            if (!has_haste) continue;
        }

        for (const auto &ab : permanent.abilities) {
            if (!is_simple_mana_source(ab)) continue;
            if (ab.activation_limit > 0 && ab.activations_this_turn >= ab.activation_limit) continue;
            if (!ab.mana_choices.empty()) {
                // Multi-color source: add one entry per color choice
                for (Colors choice_color : ab.mana_choices) {
                    Ability choice_ab = ab;
                    choice_ab.color = choice_color;
                    sources.push_back({entity, choice_ab});
                }
            } else {
                sources.push_back({entity, ab});
            }
        }
    }
    return sources;
}

bool can_afford_with_sources(Zone::Ownership player_owner, const std::multiset<Colors> &cost,
                             std::shared_ptr<Orderer> orderer, Entity exclude_entity) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return false;
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // Fast path: pool alone is enough
    if (can_afford_pool(player.mana, cost)) return true;

    // Build a hypothetical pool: current mana + all available tap-only sources
    auto hypothetical = player.mana;
    // Track which entities we've already counted (each source taps once)
    std::set<Entity> counted_entities;
    // Separate flexible (multi-color) mana count
    size_t flexible_count = 0;

    auto sources = collect_available_mana_sources(player_owner, orderer);
    for (auto &[entity, ab] : sources) {
        if (entity == exclude_entity) continue;
        if (counted_entities.count(entity)) continue;
        size_t amount = eval_mana_amount(ab, player_owner, orderer);
        if (!ab.mana_choices.empty()) {
            // Multi-color: count as flexible mana (can be any color)
            flexible_count += amount;
            counted_entities.insert(entity);
        } else {
            for (size_t i = 0; i < amount; i++) hypothetical.insert(ab.color);
            counted_entities.insert(entity);
        }
    }

    // Try to pay colored costs from the hypothetical pool
    auto remaining = hypothetical;
    size_t flexible_used = 0;
    for (auto color : cost) {
        if (color == GENERIC) continue;
        auto it = remaining.find(color);
        if (it != remaining.end()) {
            remaining.erase(it);
        } else if (flexible_used < flexible_count) {
            flexible_used++;
        } else {
            return false;
        }
    }

    size_t generic_needed = cost.count(GENERIC);
    size_t available_for_generic = remaining.size() + (flexible_count - flexible_used);
    return available_for_generic >= generic_needed;
}

size_t max_available_mana(Zone::Ownership player_owner, const ManaValue &base_cost,
                          std::shared_ptr<Orderer> orderer) {
    Entity player_entity = get_player_entity(player_owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return 0;
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // Count total mana available: pool + all sources
    size_t total = player.mana.size();
    std::set<Entity> counted;
    auto sources = collect_available_mana_sources(player_owner, orderer);
    for (auto &[entity, ab] : sources) {
        if (counted.count(entity)) continue;
        total += eval_mana_amount(ab, player_owner, orderer);
        counted.insert(entity);
    }

    // Subtract colored requirements from base cost (generic is what X adds to)
    size_t colored_obligations = 0;
    for (auto c : base_cost) {
        if (c != GENERIC) colored_obligations++;
    }
    size_t generic_in_base = base_cost.count(GENERIC);
    size_t fixed_cost = colored_obligations + generic_in_base;
    return (total > fixed_cost) ? total - fixed_cost : 0;
}

ManaPaymentSnapshot snapshot_mana_state(Zone::Ownership player, std::shared_ptr<Orderer> orderer) {
    ManaPaymentSnapshot snap;
    Entity player_entity = get_player_entity(player);
    snap.player_mana = global_coordinator.GetComponent<Player>(player_entity).mana;
    snap.delve_exiled = cur_game.delve_exiled;

    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != player) continue;

        snap.tapped_state.push_back({entity, permanent.is_tapped});
        for (size_t i = 0; i < permanent.abilities.size(); i++) {
            if (permanent.abilities[i].category == "AddMana") {
                snap.activation_counts.push_back({entity, i, permanent.abilities[i].activations_this_turn});
            }
        }
    }
    return snap;
}

void restore_mana_state(Zone::Ownership player, const ManaPaymentSnapshot &snap,
                        std::shared_ptr<Orderer> orderer) {
    Entity player_entity = get_player_entity(player);
    global_coordinator.GetComponent<Player>(player_entity).mana = snap.player_mana;

    for (auto &[entity, was_tapped] : snap.tapped_state) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        global_coordinator.GetComponent<Permanent>(entity).is_tapped = was_tapped;
    }
    for (auto &[entity, idx, old_count] : snap.activation_counts) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (idx < permanent.abilities.size()) {
            permanent.abilities[idx].activations_this_turn = old_count;
        }
    }
    // Undo delve exiles: move cards exiled since snapshot back to graveyard
    for (size_t i = snap.delve_exiled.size(); i < cur_game.delve_exiled.size(); i++) {
        Entity exiled = cur_game.delve_exiled[i];
        orderer->add_to_zone(false, exiled, Zone::GRAVEYARD);
    }
    cur_game.delve_exiled = snap.delve_exiled;
}

bool prompt_mana_payment(Zone::Ownership controller, const ManaValue &cost,
                         Entity paid_for, std::shared_ptr<Orderer> orderer,
                         bool has_delve) {
    Entity player_entity = get_player_entity(controller);
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // If pool already covers it, just spend
    if (can_afford_pool(player.mana, cost)) {
        spend_mana(controller, cost, paid_for);
        return true;
    }

    bool is_machine = InputLogger::instance().is_machine_mode();

    // Drain existing pool toward the cost first
    ManaValue remaining = pay_partial(controller, cost);

    // Payment loop: prompt player to tap sources or delve until cost is paid
    while (!remaining.empty()) {
        // Check if pool now covers remaining cost (from prior mana activations or delve)
        if (can_afford_pool(player.mana, remaining)) {
            spend_mana(controller, remaining, paid_for);
            return true;
        }

        // Build list of available payment options
        std::vector<LegalAction> pay_actions;

        // Mana abilities
        for (auto entity : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            if (zone.location != Zone::BATTLEFIELD) continue;
            auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
            if (permanent.controller != controller) continue;
            if (permanent.is_tapped) continue;
            if (permanent.is_phased_out) continue;
            if (permanent_cant_activate(entity, orderer)) continue;
            if (permanent.has_summoning_sickness &&
                global_coordinator.entity_has_component<Creature>(entity)) {
                auto &cr = global_coordinator.GetComponent<Creature>(entity);
                bool has_haste = false;
                for (const auto &kw : cr.keywords)
                    if (kw == "Haste") { has_haste = true; break; }
                if (!has_haste) continue;
            }

            for (const auto &ab : permanent.abilities) {
                if (!is_simple_mana_source(ab)) continue;
                if (ab.activation_limit > 0 && ab.activations_this_turn >= ab.activation_limit) continue;

                std::string src_name = permanent.name;
                if (!ab.mana_choices.empty()) {
                    for (Colors choice_color : ab.mana_choices) {
                        Ability choice_ab = ab;
                        choice_ab.color = choice_color;
                        std::string desc = "Tap " + src_name + " for {" +
                                           mana_symbol_str(choice_color) + "}";
                        LegalAction la(ACTIVATE_ABILITY, entity, choice_ab, desc);
                        la.category = ActionCategory::PAYING_COSTS;
                        pay_actions.push_back(la);
                    }
                } else {
                    std::string desc = "Tap " + src_name + " for {" +
                                       mana_symbol_str(ab.color) + "}";
                    LegalAction la(ACTIVATE_ABILITY, entity, ab, desc);
                    la.category = ActionCategory::PAYING_COSTS;
                    pay_actions.push_back(la);
                }
            }
        }

        // Delve: exile instants/sorceries from graveyard to pay generic costs
        size_t delve_action_start = pay_actions.size();
        if (has_delve && remaining.count(GENERIC) > 0) {
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                auto &ez = global_coordinator.GetComponent<Zone>(e);
                if (ez.location != Zone::GRAVEYARD || ez.owner != controller) continue;
                if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                auto &ecd = global_coordinator.GetComponent<CardData>(e);
                bool is_instant_or_sorcery = false;
                for (auto &t : ecd.types)
                    if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) {
                        is_instant_or_sorcery = true;
                        break;
                    }
                if (!is_instant_or_sorcery) continue;
                LegalAction la(PASS_PRIORITY, e, "Exile " + ecd.name + " (Delve)");
                la.category = ActionCategory::PAYING_COSTS;
                pay_actions.push_back(la);
            }
        }

        if (pay_actions.empty()) {
            return false;
        }

        // Add cancel option in non-machine mode
        if (!is_machine) {
            LegalAction cancel(PASS_PRIORITY, "Cancel casting");
            cancel.category = ActionCategory::PAYING_COSTS;
            pay_actions.push_back(cancel);
        }

        // Show remaining cost
        game_log("Pay mana costs (");
        bool first = true;
        for (auto it = remaining.begin(); it != remaining.end(); ++it) {
            if (!first) game_log(",");
            game_log("{%s}", mana_symbol_str(*it));
            first = false;
        }
        game_log(" remaining):\n");

        int choice = InputLogger::instance().get_input(pay_actions);

        // Check for cancel
        if (!is_machine && choice == static_cast<int>(pay_actions.size()) - 1) {
            return false;
        }

        auto &chosen = pay_actions[static_cast<size_t>(choice)];

        // Is this a delve exile action?
        if (static_cast<size_t>(choice) >= delve_action_start &&
            (!has_delve || chosen.type == PASS_PRIORITY)) {
            Entity exiled = chosen.source_entity;
            orderer->add_to_zone(false, exiled, Zone::EXILE);
            cur_game.delve_exiled.push_back(exiled);
            // Remove one generic from remaining
            auto git = remaining.find(GENERIC);
            if (git != remaining.end()) remaining.erase(git);
            auto &ecd = global_coordinator.GetComponent<CardData>(exiled);
            game_log("%s exiles %s via Delve.\n", player_name(controller).c_str(), ecd.name.c_str());
        } else {
            // Mana ability activation
            Entity source_entity = chosen.source_entity;
            const Ability &chosen_ab = chosen.ability;

            auto &perm = global_coordinator.GetComponent<Permanent>(source_entity);
            perm.is_tapped = true;

            size_t mana_amount = eval_mana_amount(chosen_ab, controller, orderer);
            add_mana(controller, chosen_ab.color, mana_amount);

            game_log("%s tapped %s for %zu{%s}\n", player_name(controller).c_str(),
                     perm.name.c_str(), mana_amount, mana_symbol_str(chosen_ab.color));

            // Increment activation counter if limited
            if (chosen_ab.activation_limit > 0) {
                for (auto &perm_ab : perm.abilities) {
                    if (perm_ab.category == chosen_ab.category &&
                        perm_ab.tap_cost == chosen_ab.tap_cost &&
                        perm_ab.color == chosen_ab.color) {
                        perm_ab.activations_this_turn++;
                        break;
                    }
                }
            }

        }
    }

    return true;
}
