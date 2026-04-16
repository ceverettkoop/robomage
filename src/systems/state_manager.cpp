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
#include "../components/token.h"
#include "../components/types.h"
#include "../type_constants.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../cli_output.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../systems/stack_manager.h"
#include "orderer.h"

std::vector<ActiveStatic> g_active_statics;

static bool compare_svar(int value, const std::string &compare);
static bool check_condition_present(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer);

static std::string entity_name(Entity e) {
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        return perm.is_token ? perm.name + " token" : perm.name;
    }
    if (global_coordinator.entity_has_component<CardData>(e))
        return global_coordinator.GetComponent<CardData>(e).name;
    if (global_coordinator.entity_has_component<Token>(e))
        return global_coordinator.GetComponent<Token>(e).name + " token";
    return "<unknown>";
}

static Colors mana_color_for_subtype(const std::string &subtype) {
    if (subtype == "Mountain") return RED;
    if (subtype == "Forest") return GREEN;
    if (subtype == "Plains") return WHITE;
    if (subtype == "Island") return BLUE;
    if (subtype == "Swamp") return BLACK;
    if (subtype == "Wastes") return COLORLESS;
    return NO_COLOR;
}

// Permanents on battlefield set to have appropriate components
// if they are in a different zone these are removed as no longer applicable
void StateManager::apply_permanent_components(Game &game) {
    // Collect token entities that have left the battlefield for destruction after iteration.
    std::vector<Entity> tokens_to_destroy;

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;

        // Handle token entities (no CardData)
        if (!global_coordinator.entity_has_component<CardData>(entity)) {
            if (!global_coordinator.entity_has_component<Token>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            auto &token = global_coordinator.GetComponent<Token>(entity);
            if (zone.location == Zone::BATTLEFIELD) {
                if (!global_coordinator.entity_has_component<Permanent>(entity)) {
                    Permanent perm;
                    perm.name = token.name;
                    perm.types = token.types;
                    perm.is_token = true;
                    perm.controller = zone.controller;
                    perm.has_summoning_sickness = true;
                    perm.is_tapped = false;
                    perm.timestamp_entered_battlefield = game.timestamp++;
                    global_coordinator.AddComponent(entity, perm);
                }
                if (!global_coordinator.entity_has_component<Creature>(entity)) {
                    Creature creature;
                    creature.power = token.power;
                    creature.toughness = token.toughness;
                    creature.keywords = token.keywords;
                    global_coordinator.AddComponent(entity, creature);
                    Damage damage;
                    damage.damage_counters = 0;
                    global_coordinator.AddComponent(entity, damage);
                }
                apply_keyword_abilities(entity);
            } else {
                // Token has left the battlefield — schedule for destruction
                if (global_coordinator.entity_has_component<Permanent>(entity))
                    global_coordinator.RemoveComponent<Permanent>(entity);
                if (global_coordinator.entity_has_component<Creature>(entity))
                    global_coordinator.RemoveComponent<Creature>(entity);
                if (global_coordinator.entity_has_component<Damage>(entity))
                    global_coordinator.RemoveComponent<Damage>(entity);
                tokens_to_destroy.push_back(entity);
            }
            continue;
        }

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
                perm.name = card_data.name;
                perm.types = card_data.types;
                perm.controller = zone.controller;
                perm.has_summoning_sickness = is_creature;
                perm.is_tapped = false;
                // Apply replacement effects at the point the permanent enters (rule 614)
                for (const auto &r : card_data.replacement_effects) {
                    switch (r.kind) {
                        case Effect::Replacement::ENTERS_TAPPED:
                            perm.is_tapped = true;
                            game_log("%s enters tapped.\n", perm.name.c_str());
                            break;
                        case Effect::Replacement::CANT_BE_COUNTERED:
                            break;  // handled at cast time, not ETB
                    }
                }
                perm.timestamp_entered_battlefield = game.timestamp++;
                global_coordinator.AddComponent(entity, perm);
            }
            // copy activated abilities from card_data to permanent; incl mana abilities although mana abilities innate to basic land types
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

                // Apply "enters with" counters from static abilities
                for (auto &sa : card_data.static_abilities) {
                    if (sa.category != "EtbCounter") continue;
                    if (sa.counter_type != "P1P1") continue;
                    int n = 0;
                    if (sa.counter_count_from_delve) {
                        n = static_cast<int>(cur_game.delve_exiled.size());
                        cur_game.delve_exiled.clear();
                    }
                    if (n <= 0) continue;
                    auto &cr = global_coordinator.GetComponent<Creature>(entity);
                    cr.plus_one_counters += n;
                    cr.power     += static_cast<uint32_t>(n);
                    cr.toughness += static_cast<uint32_t>(n);
                    game_log("%s enters with %d +1/+1 counter(s) (%u/%u).\n",
                        card_data.name.c_str(), n, cr.power, cr.toughness);
                }
            }
            if (is_land) {
                apply_land_abilities(entity);
            }
            apply_keyword_abilities(entity);

            // ETBReplacement: choose creature type (Cavern of Souls)
            if (card_data.has_etb_choose_creature_type) {
                auto &perm_ref = global_coordinator.GetComponent<Permanent>(entity);
                if (perm_ref.chosen_type.empty()) {
                Entity player_entity = (perm_ref.controller == Zone::PLAYER_A)
                    ? cur_game.player_a_entity : cur_game.player_b_entity;
                auto &player = global_coordinator.GetComponent<Player>(player_entity);

                if (!player.creature_subtypes.empty()) {
                    // Build subtype name list from all_subtypes index
                    std::vector<std::string> subtype_names;
                    for (auto &pair : player.creature_subtypes) {
                        auto it = all_subtypes.begin();
                        std::advance(it, pair.second);
                        subtype_names.push_back(*it);
                    }

                    // Count frequency of each subtype among owned creatures for sorting
                    std::vector<int> freq(subtype_names.size(), 0);
                    for (auto e2 : mEntities) {
                        if (!global_coordinator.entity_has_component<CardData>(e2)) continue;
                        auto &z2 = global_coordinator.GetComponent<Zone>(e2);
                        if (z2.owner != perm_ref.controller) continue;
                        auto &cd2 = global_coordinator.GetComponent<CardData>(e2);
                        bool is_cr = false;
                        for (auto &t : cd2.types)
                            if (t.kind == TYPE && t.name == "Creature") { is_cr = true; break; }
                        if (!is_cr) continue;
                        for (auto &t : cd2.types) {
                            if (t.kind != SUBTYPE) continue;
                            for (size_t i = 0; i < subtype_names.size(); i++) {
                                if (t.name == subtype_names[i]) freq[i]++;
                            }
                        }
                    }

                    // Sort by frequency descending (most prominent first)
                    std::vector<size_t> order(subtype_names.size());
                    for (size_t i = 0; i < order.size(); i++) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                        return freq[a] > freq[b];
                    });

                    std::vector<LegalAction> type_choices;
                    for (size_t idx : order) {
                        LegalAction la(PASS_PRIORITY, entity, "Choose creature type: " + subtype_names[idx]);
                        la.category = ActionCategory::OTHER_CHOICE;
                        type_choices.push_back(la);
                    }

                    bool prev_priority = cur_game.player_a_has_priority;
                    cur_game.player_a_has_priority = (perm_ref.controller == Zone::PLAYER_A);
                    game_log("Choose a creature type for %s:\n", perm_ref.name.c_str());
                    int choice = InputLogger::instance().get_input(type_choices);
                    cur_game.player_a_has_priority = prev_priority;
                    perm_ref.chosen_type = subtype_names[order[static_cast<size_t>(choice)]];
                    game_log("%s chose creature type: %s\n",
                             player_name(perm_ref.controller).c_str(), perm_ref.chosen_type.c_str());
                }
                }
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

    // Destroy token entities that left the battlefield (done after iteration to avoid invalidating iterators)
    for (auto e : tokens_to_destroy) {
        game_log("Token is destroyed.\n");
        global_coordinator.DestroyEntity(e);
    }

}

// Applies mana abilities to lands based on the land subtypes in perm.types.
// Type-changing effects (Blood Moon, etc.) modify perm.types before this runs.
void StateManager::apply_land_abilities(Entity entity) {
    // assumes called with entity that has permanent component and is on battlefield and is land
    auto &perm = global_coordinator.GetComponent<Permanent>(entity);
    std::vector<std::string> land_subtypes;
    for (auto &type : perm.types) {
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
        auto &perm_abilities = perm.abilities;
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
        mana_ability.subtype_derived = true;

        mana_ability.source = entity;
        perm_abilities.push_back(mana_ability);
    }
}

static Ability keyword_triggered_ability(const std::string &keyword);

void StateManager::apply_keyword_abilities(Entity entity) {
    if (!global_coordinator.entity_has_component<Creature>(entity)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(entity);
    auto &perm_abilities = global_coordinator.GetComponent<Permanent>(entity).abilities;

    for (const auto &kw : cr.keywords) {
        Ability ab = keyword_triggered_ability(kw);
        if (ab.trigger_on == 0) continue;

        bool already_present = false;
        for (const auto &existing : perm_abilities) {
            if (existing.ability_type == Ability::TRIGGERED &&
                existing.category == ab.category &&
                existing.trigger_on == ab.trigger_on) {
                already_present = true;
                break;
            }
        }
        if (already_present) continue;

        ab.source = entity;
        perm_abilities.push_back(ab);
    }
}

// Maps keywords to their corresponding triggered abilities.
// Returns an ability with trigger_on == 0 if the keyword has no triggered ability.
static Ability keyword_triggered_ability(const std::string &keyword) {
    Ability ab;
    if (keyword == "Prowess") {
        ab.ability_type = Ability::TRIGGERED;
        ab.trigger_on = Events::NONCREATURE_SPELL_CAST;
        ab.trigger_valid_player_is_controller = true;
        ab.category = "ProwessBonus";
        ab.amount = 1;
    } else if (keyword == "Exalted") {
        ab.ability_type = Ability::TRIGGERED;
        ab.trigger_on = Events::CREATURE_ATTACKED_ALONE;
        ab.trigger_valid_player_is_controller = true;
        ab.category = "ExaltedBonus";
        ab.amount = 1;
    }
    return ab;
}

// Evaluate a StaticAbility SVar expression such as "Count$TypeInYourYard.Land".
// Returns the computed integer value.
static int evaluate_sa_svar(const std::string &expr, Zone::Ownership controller) {
    // Handle /Plus.N suffix: strip it, evaluate the base, then add N
    size_t plus_pos = expr.find("/Plus.");
    if (plus_pos != std::string::npos) {
        std::string base = expr.substr(0, plus_pos);
        int offset = std::stoi(expr.substr(plus_pos + 6));
        return evaluate_sa_svar(base, controller) + offset;
    }

    // Count$TypeInYourYard.<TypeName> — count cards of that type in controller's graveyard
    if (expr.rfind("Count$TypeInYourYard.", 0) == 0) {
        std::string type_name = expr.substr(21);  // after "Count$TypeInYourYard."
        int count = 0;
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; ++e) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::GRAVEYARD || z.owner != controller) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &t : cd.types) {
                if (t.name == type_name) { count++; break; }
            }
        }
        return count;
    }

    // Count$ValidGraveyard Card$CardTypes — count distinct card types (Creature, Instant, etc.)
    // across both players' graveyards (Barrowgoyf)
    if (expr == "Count$CardTypesInAllGraveyards" ||
        expr == "Count$ValidGraveyard Card$CardTypes") {
        std::set<std::string> type_names;
        Entity max_e = global_coordinator.GetMaxIssuedEntity();
        for (Entity e = 0; e < max_e; ++e) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::GRAVEYARD) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            for (auto &t : cd.types) {
                if (t.kind == TYPE) type_names.insert(t.name);
            }
        }
        return static_cast<int>(type_names.size());
    }

    return 0;
}

// Rule 613.1d — Layer 4: type-changing continuous effects.
// Resets land types to CardData originals, then applies type-changing effects
// sorted by timestamp (rule 613.7: later timestamp wins within the same layer).
// Regenerates subtype-derived mana abilities after types are finalized.
void StateManager::apply_type_changing_effects() {
    // Collect type-changing statics from the already-populated g_active_statics.
    struct TypeChanger {
        ActiveStatic *as;
        size_t timestamp;  // source permanent's ETB timestamp
    };
    std::vector<TypeChanger> changers;
    for (auto &a : g_active_statics) {
        if (a.sa->add_type.empty()) continue;
        if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
        auto &src_perm = global_coordinator.GetComponent<Permanent>(a.entity);
        changers.push_back({&a, src_perm.timestamp_entered_battlefield});
    }

    if (changers.empty()) return;

    // Sort by timestamp ascending — later entries override earlier ones on the same permanent.
    std::sort(changers.begin(), changers.end(),
              [](const TypeChanger &a, const TypeChanger &b) { return a.timestamp < b.timestamp; });

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        if (perm.is_phased_out) continue;

        bool is_land = false;
        bool is_basic = false;
        for (auto &t : perm.types) {
            if (t.kind == TYPE && t.name == "Land") is_land = true;
            if (t.kind == SUPERTYPE && t.name == "Basic") is_basic = true;
        }
        if (!is_land || is_basic) continue;

        // Find the winning (latest-timestamp) type-changing effect that affects this land.
        // Because changers is sorted ascending, the last match wins.
        const TypeChanger *winner = nullptr;
        for (auto &tc : changers) {
            if (tc.as->sa->affected == "Land.nonBasic") {
                winner = &tc;  // later entry overwrites
            }
        }
        if (!winner) continue;

        // Reset land subtypes to CardData originals
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            auto &card_data = global_coordinator.GetComponent<CardData>(entity);
            // Restore subtypes from CardData
            std::set<Type> new_types;
            for (auto &t : card_data.types) {
                if (t.kind != SUBTYPE) new_types.insert(t);
            }
            // Non-subtype types come from CardData; subtypes are replaced below
            perm.types = new_types;
        } else {
            // Token land — strip existing land subtypes
            std::set<Type> new_types;
            for (auto &t : perm.types) {
                if (t.kind == SUBTYPE &&
                    (t.name == "Mountain" || t.name == "Forest" || t.name == "Plains" ||
                     t.name == "Island" || t.name == "Swamp" || t.name == "Wastes"))
                    continue;
                new_types.insert(t);
            }
            perm.types = new_types;
        }

        // Apply the winning type
        if (winner->as->sa->remove_land_types) {
            // Already stripped above; add the new subtype
            perm.types.insert({SUBTYPE, winner->as->sa->add_type});
        }

        // Strip subtype-derived mana abilities and regenerate from new types
        auto &abilities = perm.abilities;
        abilities.erase(std::remove_if(abilities.begin(), abilities.end(),
                                       [](const Ability &ab) { return ab.subtype_derived; }),
                        abilities.end());
        apply_land_abilities(entity);
    }
}

void StateManager::apply_static_ability_effects() {
    // Phase 1: gather active static abilities from all battlefield permanents into the
    // global g_active_statics cache. Other systems read this instead of scanning permanents.
    g_active_statics.clear();

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        if (perm.is_phased_out) continue;
        if (perm.transformed) {
            for (auto &sa : perm.static_abilities) sa.applied = false;
            continue;
        }
        for (auto &sa : perm.static_abilities)
            g_active_statics.push_back({entity, &sa, perm.controller});
    }

    // Layer 4 (rule 613.1d): type-changing continuous effects applied before mana
    // abilities are regenerated. Must run even if no other statics are active, but
    // only the type-changing subset matters here — the rest is handled below.
    apply_type_changing_effects();

    // Alias for local readability
    auto &active = g_active_statics;

    if (active.empty()) return;

    // Phase 2: evaluate only the conditions actually referenced by gathered abilities.
    // Each condition is computed at most once per player rather than once per permanent.
    bool need_delirium_a = false, need_delirium_b = false;
    for (auto &a : active) {
        if (a.sa->condition == "Delirium") {
            if (a.controller == Zone::PLAYER_A) need_delirium_a = true;
            else                                need_delirium_b = true;
        }
    }
    bool delirium_a = need_delirium_a ? check_delirium(Zone::PLAYER_A, mEntities) : false;
    bool delirium_b = need_delirium_b ? check_delirium(Zone::PLAYER_B, mEntities) : false;

    // Phase 3: apply or revert effects based on condition results.
    for (auto &a : active) {
        bool condition_met;
        if (a.sa->condition.empty() && a.sa->check_svar_expr.empty()) {
            condition_met = true;
        } else if (a.sa->condition == "Delirium") {
            condition_met = (a.controller == Zone::PLAYER_A) ? delirium_a : delirium_b;
        } else if (!a.sa->check_svar_expr.empty()) {
            // SVar-based condition (e.g. Keen-Eyed Curator: GE4 distinct card types among exiled_with)
            int svar_val = 0;
            if (a.sa->check_svar_expr.find("Count$ValidExile") != std::string::npos &&
                a.sa->check_svar_expr.find("CardTypes") != std::string::npos) {
                // Count distinct card types among entities in this permanent's exiled_with
                if (global_coordinator.entity_has_component<Permanent>(a.entity)) {
                    auto &eperm = global_coordinator.GetComponent<Permanent>(a.entity);
                    std::set<std::string> type_names;
                    for (auto ex_e : eperm.exiled_with) {
                        if (!global_coordinator.entity_has_component<CardData>(ex_e)) continue;
                        for (auto &t : global_coordinator.GetComponent<CardData>(ex_e).types)
                            if (t.kind == TYPE) type_names.insert(t.name);
                    }
                    svar_val = static_cast<int>(type_names.size());
                }
            } else {
                svar_val = evaluate_sa_svar(a.sa->check_svar_expr, a.controller);
            }
            condition_met = compare_svar(svar_val, a.sa->svar_compare);
        } else {
            condition_met = false;  // unrecognised condition — treat as unmet
        }

        if (a.sa->category == "Continuous") {
            // Characteristic-defining ability (rule 604.3): sets base P/T each SBE pass.
            // Handled separately because it replaces rather than modifies P/T.
            if (a.sa->characteristic_defining &&
                (!a.sa->set_power_svar.empty() || !a.sa->set_toughness_svar.empty())) {
                if (!global_coordinator.entity_has_component<Creature>(a.entity)) continue;
                auto &cr = global_coordinator.GetComponent<Creature>(a.entity);
                int base_p = !a.sa->set_power_svar.empty()
                    ? evaluate_sa_svar(a.sa->set_power_svar, a.controller) : 0;
                int base_t = !a.sa->set_toughness_svar.empty()
                    ? evaluate_sa_svar(a.sa->set_toughness_svar, a.controller) : 0;
                // Preserve counters and temporary bonuses on top of the new base
                int counters = cr.plus_one_counters;
                int prowess = cr.prowess_bonus;
                cr.power = static_cast<uint32_t>(std::max(0, base_p + counters + prowess));
                cr.toughness = static_cast<uint32_t>(std::max(0, base_t + counters));
                a.sa->applied = true;
                continue;
            }

            // Determine which entity receives the buff (source or equipped creature)
            Entity target_entity = a.entity;
            if (a.sa->affected == "EquippedBy") {
                if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
                target_entity = global_coordinator.GetComponent<Permanent>(a.entity).equipped_to;
            }

            // If applied to a different entity than before, revert from the previous one
            if (a.sa->applied && a.sa->last_applied_entity != target_entity) {
                Entity prev = static_cast<Entity>(a.sa->last_applied_entity);
                if (prev != 0 && global_coordinator.entity_has_component<Creature>(prev)) {
                    auto &pcr = global_coordinator.GetComponent<Creature>(prev);
                    int rev_p = !a.sa->add_power_svar.empty()     ? a.sa->last_applied_power     : a.sa->add_power;
                    int rev_t = !a.sa->add_toughness_svar.empty() ? a.sa->last_applied_toughness : a.sa->add_toughness;
                    if (rev_p != 0) pcr.power     = static_cast<uint32_t>(static_cast<int>(pcr.power)     - rev_p);
                    if (rev_t != 0) pcr.toughness = static_cast<uint32_t>(static_cast<int>(pcr.toughness) - rev_t);
                    if (!a.sa->add_keyword.empty()) {
                        const std::string &kws = a.sa->add_keyword;
                        size_t p = 0;
                        while (p < kws.size()) {
                            size_t sep = kws.find(" & ", p);
                            if (sep == std::string::npos) sep = kws.size();
                            std::string kw = kws.substr(p, sep - p);
                            if (!kw.empty()) {
                                auto it = std::find(pcr.keywords.begin(), pcr.keywords.end(), kw);
                                if (it != pcr.keywords.end()) pcr.keywords.erase(it);
                            }
                            p = (sep < kws.size()) ? sep + 3 : sep;
                        }
                    }
                }
                a.sa->last_applied_power = 0;
                a.sa->last_applied_toughness = 0;
                a.sa->applied = false;
            }

            if (target_entity == 0 || !global_coordinator.entity_has_component<Creature>(target_entity)) {
                // No valid target; revert if currently applied
                if (a.sa->applied) {
                    a.sa->applied = false;
                }
                continue;
            }

            auto &cr = global_coordinator.GetComponent<Creature>(target_entity);
            const std::string name_for_log = entity_name(target_entity);

            // Dynamic svar P/T: re-evaluate every SBE pass and apply the delta.
            bool has_dynamic_pt = !a.sa->add_power_svar.empty() || !a.sa->add_toughness_svar.empty();
            if (has_dynamic_pt) {
                if (condition_met) {
                    int new_p = a.sa->add_power_svar.empty()
                                    ? a.sa->add_power
                                    : evaluate_sa_svar(a.sa->add_power_svar, a.controller);
                    int new_t = a.sa->add_toughness_svar.empty()
                                    ? a.sa->add_toughness
                                    : evaluate_sa_svar(a.sa->add_toughness_svar, a.controller);
                    int dp = new_p - a.sa->last_applied_power;
                    int dt = new_t - a.sa->last_applied_toughness;
                    if (dp != 0) cr.power     = static_cast<uint32_t>(static_cast<int>(cr.power)     + dp);
                    if (dt != 0) cr.toughness = static_cast<uint32_t>(static_cast<int>(cr.toughness) + dt);
                    a.sa->last_applied_power     = new_p;
                    a.sa->last_applied_toughness = new_t;
                    a.sa->applied = true;
                    a.sa->last_applied_entity = static_cast<uint32_t>(target_entity);
                } else if (a.sa->applied) {
                    if (a.sa->last_applied_power != 0)
                        cr.power     = static_cast<uint32_t>(static_cast<int>(cr.power)     - a.sa->last_applied_power);
                    if (a.sa->last_applied_toughness != 0)
                        cr.toughness = static_cast<uint32_t>(static_cast<int>(cr.toughness) - a.sa->last_applied_toughness);
                    a.sa->last_applied_power = 0;
                    a.sa->last_applied_toughness = 0;
                    a.sa->applied = false;
                }
                continue;
            }

            if (condition_met && !a.sa->applied) {
                if (a.sa->add_power     != 0) cr.power     += static_cast<uint32_t>(a.sa->add_power);
                if (a.sa->add_toughness != 0) cr.toughness += static_cast<uint32_t>(a.sa->add_toughness);
                if (!a.sa->add_keyword.empty()) {
                    // Split multi-keywords on " & " and add each separately
                    const std::string &kws = a.sa->add_keyword;
                    size_t p = 0;
                    while (p < kws.size()) {
                        size_t sep = kws.find(" & ", p);
                        if (sep == std::string::npos) sep = kws.size();
                        std::string kw = kws.substr(p, sep - p);
                        if (!kw.empty()) cr.keywords.push_back(kw);
                        p = (sep < kws.size()) ? sep + 3 : sep;
                    }
                }
                a.sa->applied = true;
                a.sa->last_applied_entity = static_cast<uint32_t>(target_entity);
                game_log("%s gains %s%s(%s)\n", name_for_log.c_str(),
                         a.sa->add_power != 0 ? (std::to_string(a.sa->add_power) + "/" +
                                                  std::to_string(a.sa->add_toughness) + " ").c_str() : "",
                         !a.sa->add_keyword.empty() ? (a.sa->add_keyword + " ").c_str() : "",
                         a.sa->condition.empty() ? "always" : a.sa->condition.c_str());
            } else if (!condition_met && a.sa->applied) {
                if (a.sa->add_power     != 0) cr.power     -= static_cast<uint32_t>(a.sa->add_power);
                if (a.sa->add_toughness != 0) cr.toughness -= static_cast<uint32_t>(a.sa->add_toughness);
                if (!a.sa->add_keyword.empty()) {
                    const std::string &kws = a.sa->add_keyword;
                    size_t p = 0;
                    while (p < kws.size()) {
                        size_t sep = kws.find(" & ", p);
                        if (sep == std::string::npos) sep = kws.size();
                        std::string kw = kws.substr(p, sep - p);
                        if (!kw.empty()) {
                            auto it = std::find(cr.keywords.begin(), cr.keywords.end(), kw);
                            if (it != cr.keywords.end()) cr.keywords.erase(it);
                        }
                        p = (sep < kws.size()) ? sep + 3 : sep;
                    }
                }
                a.sa->applied = false;
                game_log("%s loses %s bonus\n", name_for_log.c_str(),
                         a.sa->condition.empty() ? "static" : a.sa->condition.c_str());
            }
        }

        if (a.sa->category == "MustAttack") {
            if (!global_coordinator.entity_has_component<Creature>(a.entity)) continue;
            auto &cr = global_coordinator.GetComponent<Creature>(a.entity);
            cr.must_attack = condition_met;
            a.sa->applied = condition_met;
        }
    }
}

void StateManager::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<StateManager>(signature);
}

// Returns true if the creature has the given keyword
static bool creature_has_keyword(const Creature &cr, const char *kw) {
    for (const auto &k : cr.keywords)
        if (k == kw) return true;
    return false;
}

// Should this creature deal damage during this combat damage step?
static bool should_deal_damage(const Creature &cr, bool first_strike_only) {
    bool has_fs = creature_has_keyword(cr, "First Strike");
    bool has_ds = creature_has_keyword(cr, "Double Strike");
    if (first_strike_only) return has_fs || has_ds;
    // Regular damage step: skip first-strikers (they already dealt), but double strikers hit again
    if (has_fs && !has_ds) return false;
    return true;
}

// Accumulate combat damage and lifelink gain in a single pass so that rule 119.3
// (simultaneous damage/life-gain) applies: a controller only dies if the NET
// life change leaves them at 0 or less after both effects resolve together.
static void apply_lifelink_if_any(Entity source, size_t amount,
                                  int &life_delta_a, int &life_delta_b,
                                  Game &game) {
    if (amount == 0) return;
    if (!global_coordinator.entity_has_component<Creature>(source)) return;
    auto &cr = global_coordinator.GetComponent<Creature>(source);
    bool has_lifelink = false;
    for (const auto &k : cr.keywords)
        if (k == "Lifelink") { has_lifelink = true; break; }
    if (!has_lifelink) return;
    if (!global_coordinator.entity_has_component<Permanent>(source)) return;
    Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(source).controller;
    if (ctrl == Zone::PLAYER_A)      life_delta_a += static_cast<int>(amount);
    else if (ctrl == Zone::PLAYER_B) life_delta_b += static_cast<int>(amount);
    (void)game;
}

// Combat damage is a turn-based action (rule 510.2), not a state-based action
void StateManager::deal_combat_damage(Game &game, bool first_strike_only) {
    game_log("\n--- %sCombat Damage ---\n", first_strike_only ? "First Strike " : "");

    // Accumulate lifelink life gains; apply at end so they are simultaneous with damage.
    int life_delta_a = 0;
    int life_delta_b = 0;

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(entity);
        if (!cr.is_attacking) continue;
        if (!should_deal_damage(cr, first_strike_only)) continue;

        std::string attacker_name = entity_name(entity);

        // Collect blockers for this attacker
        std::vector<Entity> blockers;
        for (auto b : mEntities) {
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
                    game_log("  %s deals %u damage to %s\n", attacker_name.c_str(), dmg, tname);

                    Event ev(Events::COMBAT_DAMAGE_TO_PLAYER);
                    ev.SetParam(Params::ENTITY, entity);
                    ev.SetParam(Params::PLAYER, cr.attack_target);
                    ev.SetParam(Params::AMOUNT, dmg);
                    global_coordinator.SendEvent(ev);
                }
                apply_lifelink_if_any(entity, dmg, life_delta_a, life_delta_b, game);
            }
        } else {
            // Blocked — assign damage to blockers in order, blockers deal damage back
            uint32_t remaining = cr.power;
            for (auto blocker : blockers) {
                auto &bcr = global_coordinator.GetComponent<Creature>(blocker);
                std::string blocker_name = entity_name(blocker);

                // Blocker deals damage to attacker (only if the blocker qualifies for this step)
                if (bcr.power > 0 && should_deal_damage(bcr, first_strike_only)) {
                    deal_damage(blocker, entity, bcr.power);
                    game_log("  %s deals %u damage to %s\n", blocker_name.c_str(), bcr.power, attacker_name.c_str());
                    apply_lifelink_if_any(blocker, bcr.power, life_delta_a, life_delta_b, game);
                }

                // Attacker deals damage to blocker. Deathtouch on the attacker means
                // 1 damage is considered lethal (702.2c), so we don't have to stack
                // toughness worth onto the first blocker.
                if (remaining > 0) {
                    bool attacker_dt = creature_has_keyword(cr, "Deathtouch");
                    uint32_t needed = attacker_dt ? 1u : bcr.toughness;
                    uint32_t assigned = (remaining >= needed) ? needed : remaining;
                    deal_damage(entity, blocker, assigned);
                    game_log("  %s deals %u damage to %s\n", attacker_name.c_str(), assigned, blocker_name.c_str());
                    remaining -= assigned;
                    apply_lifelink_if_any(entity, assigned, life_delta_a, life_delta_b, game);
                }
            }
            // Trample: excess damage goes to attack target
            if (remaining > 0) {
                bool has_trample = creature_has_keyword(cr, "Trample");
                if (has_trample && global_coordinator.entity_has_component<Player>(cr.attack_target)) {
                    deal_damage(entity, cr.attack_target, remaining);
                    auto &target_player = global_coordinator.GetComponent<Player>(cr.attack_target);
                    target_player.life_total -= static_cast<int>(remaining);
                    const char *tname = (cr.attack_target == game.player_a_entity) ? "Player A" : "Player B";
                    game_log("  %s tramples %u damage to %s\n", attacker_name.c_str(), remaining, tname);

                    Event ev(Events::COMBAT_DAMAGE_TO_PLAYER);
                    ev.SetParam(Params::ENTITY, entity);
                    ev.SetParam(Params::PLAYER, cr.attack_target);
                    ev.SetParam(Params::AMOUNT, remaining);
                    global_coordinator.SendEvent(ev);
                    apply_lifelink_if_any(entity, remaining, life_delta_a, life_delta_b, game);
                }
            }
        }
    }

    // Apply lifelink life gains simultaneously with damage taken (rule 119.3).
    if (life_delta_a != 0) {
        auto &pa = global_coordinator.GetComponent<Player>(game.player_a_entity);
        pa.life_total += life_delta_a;
        game_log("  Player A gains %d life (lifelink)\n", life_delta_a);
    }
    if (life_delta_b != 0) {
        auto &pb = global_coordinator.GetComponent<Player>(game.player_b_entity);
        pb.life_total += life_delta_b;
        game_log("  Player B gains %d life (lifelink)\n", life_delta_b);
    }

    game.combat_damage_dealt = true;
    game_log("--- End %sCombat Damage ---\n\n", first_strike_only ? "First Strike " : "");
}

// Turn-based actions happen at the start of specific steps (rules 508, 509, 510, 514)
void StateManager::process_turn_based_actions(Game &game, std::shared_ptr<Orderer> orderer) {
    game.pending_choice = NONE;

    // First strike combat damage (rule 510.1)
    if (game.cur_step == FIRST_STRIKE_DAMAGE && !game.combat_damage_dealt) {
        deal_combat_damage(game, true);
    }
    // Regular combat damage (rule 510.2)
    if (game.cur_step == COMBAT_DAMAGE && !game.combat_damage_dealt) {
        deal_combat_damage(game, false);
    }

    // Declare attackers (rule 508.1)
    if (game.cur_step == DECLARE_ATTACKERS && !game.attackers_declared) {
        game.pending_choice = DECLARE_ATTACKERS_CHOICE;
        return;
    }
    // Declare blockers (rule 509.1)
    if (game.cur_step == DECLARE_BLOCKERS && !game.blockers_declared) {
        game.pending_choice = DECLARE_BLOCKERS_CHOICE;
        return;
    }
    // Cleanup discard (rule 514.1)
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
}

// State-based actions are checked simultaneously and loop until stable (rule 704.3)
void StateManager::state_based_effects(Game &game, std::shared_ptr<Orderer> orderer) {
    for (;;) {
        // Continuous effects define the game state that SBAs evaluate
        apply_permanent_components(game);
        apply_static_ability_effects();

        bool any_applied = false;

        // 704.5a - player with 0 or less life loses
        auto &player_a = global_coordinator.GetComponent<Player>(game.player_a_entity);
        auto &player_b = global_coordinator.GetComponent<Player>(game.player_b_entity);
        if (player_a.life_total <= 0) {
            printf("\nPlayer A has %d life - Player B wins!\n", player_a.life_total);
            game.ended = true;
            game.winner = Zone::PLAYER_B;
            return;
        }
        if (player_b.life_total <= 0) {
            printf("\nPlayer B has %d life - Player A wins!\n", player_b.life_total);
            game.ended = true;
            game.winner = Zone::PLAYER_A;
            return;
        }

        // 704.5d - tokens in zones other than battlefield cease to exist
        // (handled by apply_permanent_components above)

        // 704.5f - creature with toughness 0 or less goes to graveyard
        // 704.5g - creature with lethal damage is destroyed
        std::vector<Entity> creatures_to_destroy;
        for (auto entity : mEntities) {
            if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
            if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
            auto &zone = global_coordinator.GetComponent<Zone>(entity);
            if (zone.location != Zone::BATTLEFIELD) continue;
            if (global_coordinator.entity_has_component<Permanent>(entity) &&
                global_coordinator.GetComponent<Permanent>(entity).is_phased_out) continue;

            auto &creature = global_coordinator.GetComponent<Creature>(entity);
            if (creature.toughness == 0) {
                creatures_to_destroy.push_back(entity);
            } else if (global_coordinator.entity_has_component<Damage>(entity)) {
                auto &damage = global_coordinator.GetComponent<Damage>(entity);
                // 702.2b: any nonzero damage from a deathtouch source is lethal.
                bool deathtouched = damage.has_deathtouch_damage && damage.damage_counters > 0;
                if (deathtouched || damage.damage_counters >= creature.toughness) {
                    creatures_to_destroy.push_back(entity);
                }
            }
        }

        for (auto entity : creatures_to_destroy) {
            std::string name = entity_name(entity);
            auto &creature = global_coordinator.GetComponent<Creature>(entity);
            if (creature.toughness == 0)
                game_log("%s dies (zero toughness)\n", name.c_str());
            else
                game_log("%s is destroyed (lethal damage)\n", name.c_str());
            orderer->add_to_zone(false, entity, Zone::GRAVEYARD);
            any_applied = true;
        }

        // TODO 704.5j - legend rule

        if (!any_applied) break;
    }

    // SBA loop settled; triggered abilities go on the stack (rule 704.3)
    check_triggered_abilities(game, orderer);
}

// Drains all buffered events since the last call and puts any triggered abilities
// from battlefield permanents whose trigger condition matches onto the stack.
void StateManager::check_triggered_abilities(Game &game, std::shared_ptr<Orderer> orderer) {
    auto events = global_coordinator.drain_pending_events();

    // Fire any delayed triggers that match current events
    {
        std::vector<size_t> to_remove;
        for (size_t i = 0; i < game.delayed_triggers.size(); i++) {
            auto &dt = game.delayed_triggers[i];
            bool matched = false;
            for (const auto &ev : events) {
                if (ev.GetType() != dt.fire_on) continue;
                if (dt.fire_on == Events::UPKEEP_BEGAN && game.turn < dt.fire_on_turn) continue;
                // Owner check: only fire on the correct player's upkeep
                if (ev.HasParam(Params::PLAYER) &&
                    ev.GetParam<Entity>(Params::PLAYER) != dt.owner_entity) continue;
                matched = true;
                break;
            }
            if (matched) {
                // Determine controller from owner_entity
                Zone::Ownership ctrl = (dt.owner_entity == game.player_a_entity)
                                       ? Zone::PLAYER_A : Zone::PLAYER_B;
                Entity trigger_entity = global_coordinator.CreateEntity();
                Zone ab_zone(Zone::HAND, ctrl, ctrl);
                global_coordinator.AddComponent(trigger_entity, ab_zone);
                orderer->add_to_zone(false, trigger_entity, Zone::STACK);
                Ability trigger_ab = dt.ability;
                trigger_ab.controller = ctrl;
                global_coordinator.AddComponent(trigger_entity, trigger_ab);
                game_log("Delayed trigger fires.\n");
                to_remove.push_back(i);
            }
        }
        for (auto it = to_remove.rbegin(); it != to_remove.rend(); ++it)
            game.delayed_triggers.erase(game.delayed_triggers.begin() + static_cast<ptrdiff_t>(*it));
    }

    if (events.empty()) return;

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        if (perm.is_phased_out) continue;

        // Gather triggered abilities from all sources:
        // CardData/Token for innate abilities, Permanent for keyword-granted abilities
        std::vector<const std::vector<Ability>*> ab_sources;
        if (global_coordinator.entity_has_component<CardData>(entity))
            ab_sources.push_back(&global_coordinator.GetComponent<CardData>(entity).abilities);
        if (global_coordinator.entity_has_component<Token>(entity))
            ab_sources.push_back(&global_coordinator.GetComponent<Token>(entity).abilities);
        ab_sources.push_back(&perm.abilities);
        if (ab_sources.empty()) continue;

        const std::string ent_name = entity_name(entity);

        for (const auto &ev : events) {
            for (const auto *src : ab_sources) {
            for (const auto &ab : *src) {
                if (ab.ability_type != Ability::TRIGGERED) continue;
                if (ab.trigger_on == 0 || ab.trigger_on != ev.GetType()) continue;
                // "another" check: skip if the event entity is the triggering permanent itself
                if (ab.trigger_self_excluded && ev.HasParam(Params::ENTITY) &&
                    ev.GetParam<Entity>(Params::ENTITY) == entity) continue;
                // Card.Self: only fire when the event entity is the triggering permanent itself
                if (ab.trigger_only_self && ev.HasParam(Params::ENTITY) &&
                    ev.GetParam<Entity>(Params::ENTITY) != entity) continue;
                // Don't fire front-face triggers on a transformed permanent
                if (perm.transformed) continue;
                // ValidPlayer$ You: only fire when the event's player matches the permanent's controller
                if (ab.trigger_valid_player_is_controller && ev.HasParam(Params::PLAYER)) {
                    Entity event_player = ev.GetParam<Entity>(Params::PLAYER);
                    Entity ctrl_entity = (perm.controller == Zone::PLAYER_A)
                                         ? game.player_a_entity : game.player_b_entity;
                    if (event_player != ctrl_entity) continue;
                }
                // DisableTriggers check (Doorkeeper Thrull): suppress ETB triggers caused by matching card types
                if (ev.GetType() == Events::CARD_CHANGED_ZONE &&
                    ev.GetParam<Zone::ZoneValue>(Params::DESTINATION) == Zone::BATTLEFIELD) {
                    bool suppressed = false;
                    Entity entering = ev.HasParam(Params::ENTITY) ? ev.GetParam<Entity>(Params::ENTITY) : 0;
                    for (const auto &as : g_active_statics) {
                        if (as.sa->category != "DisableTriggers") continue;
                        if (entering != 0 && global_coordinator.entity_has_component<CardData>(entering)) {
                            auto &ecd = global_coordinator.GetComponent<CardData>(entering);
                            for (auto &t : ecd.types) {
                                if (as.sa->disable_triggers_cause.find(t.name) != std::string::npos) {
                                    suppressed = true; break;
                                }
                            }
                        }
                        if (suppressed) break;
                    }
                    if (suppressed) continue;
                }

                // CARD_CHANGED_ZONE filters: origin, destination, card type
                if (ev.GetType() == Events::CARD_CHANGED_ZONE) {
                    Zone::ZoneValue ev_origin = ev.GetParam<Zone::ZoneValue>(Params::ORIGIN);
                    Zone::ZoneValue ev_dest   = ev.GetParam<Zone::ZoneValue>(Params::DESTINATION);
                    if (ab.trigger_zone_origin >= 0 &&
                        ev_origin != static_cast<Zone::ZoneValue>(ab.trigger_zone_origin)) continue;
                    if (ab.trigger_zone_destination >= 0 &&
                        ev_dest != static_cast<Zone::ZoneValue>(ab.trigger_zone_destination)) continue;
                    // ValidCard$ Creature filter
                    if (ab.trigger_valid_card_is_creature && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        bool is_creature = global_coordinator.entity_has_component<Token>(ev_card);
                        if (!is_creature && global_coordinator.entity_has_component<CardData>(ev_card)) {
                            for (auto &t : global_coordinator.GetComponent<CardData>(ev_card).types)
                                if (t.kind == TYPE && t.name == "Creature") { is_creature = true; break; }
                        }
                        if (!is_creature) continue;
                    }
                    // ValidCard$ Instant/Sorcery filter (Murktide Regent)
                    if (ab.trigger_valid_card_is_instant_or_sorcery && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        if (!global_coordinator.entity_has_component<CardData>(ev_card)) continue;
                        bool ok = false;
                        for (auto &t : global_coordinator.GetComponent<CardData>(ev_card).types)
                            if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) { ok = true; break; }
                        if (!ok) continue;
                    }
                    // ValidCard$ Land.* filter (landfall)
                    if (ab.trigger_valid_card_is_land && ev.HasParam(Params::ENTITY)) {
                        Entity ev_card = ev.GetParam<Entity>(Params::ENTITY);
                        bool is_land = false;
                        if (global_coordinator.entity_has_component<CardData>(ev_card)) {
                            for (auto &t : global_coordinator.GetComponent<CardData>(ev_card).types)
                                if (t.kind == TYPE && t.name == "Land") { is_land = true; break; }
                        }
                        if (!is_land) continue;
                    }
                }
                // Spell count filter (Cori-Steel Cutter)
                if (ab.trigger_spell_count_eq > 0 && ev.HasParam(Params::PLAYER)) {
                    Entity ev_player = ev.GetParam<Entity>(Params::PLAYER);
                    if (!global_coordinator.entity_has_component<Player>(ev_player)) continue;
                    auto &pl = global_coordinator.GetComponent<Player>(ev_player);
                    if (pl.spells_cast_this_turn != ab.trigger_spell_count_eq) continue;
                }

                // Push the triggered ability onto the stack as a standalone entity
                Entity trigger_entity = global_coordinator.CreateEntity();
                Zone ab_zone(Zone::HAND, perm.controller, perm.controller);
                global_coordinator.AddComponent(trigger_entity, ab_zone);
                orderer->add_to_zone(false, trigger_entity, Zone::STACK);

                Ability trigger_ab = ab;
                trigger_ab.source = entity;
                trigger_ab.controller = perm.controller;
                // For exalted, target the sole attacker from the event
                if (trigger_ab.category == "ExaltedBonus" && ev.HasParam(Params::ENTITY))
                    trigger_ab.target = ev.GetParam<Entity>(Params::ENTITY);
                // For combat damage triggers, capture the damage amount
                if (ev.GetType() == Events::COMBAT_DAMAGE_TO_PLAYER && ev.HasParam(Params::AMOUNT))
                    trigger_ab.trigger_damage_amount = ev.GetParam<uint32_t>(Params::AMOUNT);
                global_coordinator.AddComponent(trigger_entity, trigger_ab);

                game_log("%s triggered\n", ent_name.c_str());
            }
            }
        }
    }
}

// Simple SVar comparison logic (shared between statics and alt costs)
static bool compare_svar(int value, const std::string &compare) {
    if (compare.rfind("EQ", 0) == 0)  return value == std::stoi(compare.substr(2));
    if (compare.rfind("NE", 0) == 0)  return value != std::stoi(compare.substr(2));
    if (compare.rfind("GE", 0) == 0)  return value >= std::stoi(compare.substr(2));
    if (compare.rfind("LE", 0) == 0)  return value <= std::stoi(compare.substr(2));
    if (compare.rfind("GT", 0) == 0)  return value >  std::stoi(compare.substr(2));
    if (compare.rfind("LT", 0) == 0)  return value <  std::stoi(compare.substr(2));
    return false;
}

static bool can_afford_alt(const AltCost& alt_cost, Zone::Ownership priority_player,
                           Entity card_entity, std::shared_ptr<Orderer> orderer) {
    if (!alt_cost.has_alt_cost) return false;

    // Check SVar condition (e.g. Once Upon a Time: free only if first spell this game)
    if (!alt_cost.condition_svar.empty()) {
        int svar_value = 0;
        if (alt_cost.condition_svar.find("Count$YouCastThisGame") != std::string::npos) {
            Entity pp_entity = (priority_player == Zone::PLAYER_A)
                ? cur_game.player_a_entity : cur_game.player_b_entity;
            svar_value = static_cast<int>(global_coordinator.GetComponent<Player>(pp_entity).spells_cast_this_game);
        }
        if (!compare_svar(svar_value, alt_cost.condition_compare)) return false;
    }

    // Free alt cost: no further affordability checks needed
    if (alt_cost.is_free) return true;

    if (alt_cost.return_to_hand_count > 0) {
        int matching = 0;
        const std::string& sub = alt_cost.return_to_hand_type;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            auto& perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != priority_player) continue;
            for (auto& t : perm.types) {
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

    // Condition: not your turn (Force of Negation, Force of Vigor)
    if (alt_cost.condition_not_your_turn) {
        bool is_my_turn = (priority_player == Zone::PLAYER_A) ? cur_game.player_a_turn : !cur_game.player_a_turn;
        if (is_my_turn) return false;
    }

    if (alt_cost.exile_from_hand_count > 0) {
        Colors required_color = alt_cost.exile_from_hand_color;
        bool has_match = false;
        for (auto e : orderer->get_hand(priority_player)) {
            if (e == card_entity) continue;
            if (required_color != NO_COLOR && global_coordinator.entity_has_component<ColorIdentity>(e) &&
                global_coordinator.GetComponent<ColorIdentity>(e).colors.count(required_color)) {
                has_match = true; break;
            }
        }
        if (!has_match) return false;
    }

    return true;
}

// Check ConditionPresent$ / ConditionCompare$ castability condition.
// Counts battlefield permanents matching the filter and compares against the threshold.
// Filter format: "Type.YouCtrl" or "Type.OppCtrl" (e.g. "Land.YouCtrl").
static bool check_condition_present(const Ability &ab, Zone::Ownership caster, std::shared_ptr<Orderer> orderer) {
    if (ab.condition_present.empty()) return true;

    // Parse filter: "Land.YouCtrl" → type_filter="Land", controller check
    std::string filter = ab.condition_present;
    std::string type_filter;
    bool you_ctrl = false;
    bool opp_ctrl = false;
    size_t dot = filter.find('.');
    if (dot != std::string::npos) {
        type_filter = filter.substr(0, dot);
        std::string qualifier = filter.substr(dot + 1);
        if (qualifier == "YouCtrl") you_ctrl = true;
        else if (qualifier == "OppCtrl") opp_ctrl = true;
    } else {
        type_filter = filter;
    }

    Zone::Ownership required_ctrl = you_ctrl ? caster :
        opp_ctrl ? (caster == Zone::PLAYER_A ? Zone::PLAYER_B : Zone::PLAYER_A) :
        Zone::UNKNOWN;

    size_t count = 0;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::BATTLEFIELD) continue;
        if (required_ctrl != Zone::UNKNOWN) {
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != required_ctrl) continue;
        }
        if (!type_filter.empty() && global_coordinator.entity_has_component<CardData>(e)) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            bool match = false;
            for (const auto &t : cd.types) {
                if (t.name == type_filter) { match = true; break; }
            }
            if (!match) continue;
        }
        count++;
    }

    return compare_svar(static_cast<int>(count), ab.condition_compare);
}

// Count instants/sorceries in the player's graveyard that can be exiled for Delve
static size_t count_delve_fuel(Zone::Ownership player, std::shared_ptr<Orderer> orderer) {
    size_t count = 0;
    for (auto e : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &ez = global_coordinator.GetComponent<Zone>(e);
        if (ez.location != Zone::GRAVEYARD || ez.owner != player) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        for (auto &t : global_coordinator.GetComponent<CardData>(e).types)
            if (t.kind == TYPE && (t.name == "Instant" || t.name == "Sorcery")) { count++; break; }
    }
    return count;
}

// Check if player can afford cost using mana pool + Delve (exiling graveyard instants/sorceries)
static bool can_afford_with_delve(Zone::Ownership player, const ManaValue &cost,
                                  std::shared_ptr<Orderer> orderer, Entity paid_for = 0) {
    size_t generic_in_cost = cost.count(GENERIC);
    if (generic_in_cost == 0) return can_afford_with_sources(player, cost, orderer, 0, paid_for);
    // Reduce generic by however many cards can be exiled
    size_t fuel = count_delve_fuel(player, orderer);
    size_t to_exile = std::min(generic_in_cost, fuel);
    ManaValue reduced_cost = cost;
    for (size_t i = 0; i < to_exile; i++) reduced_cost.erase(reduced_cost.find(GENERIC));
    return can_afford_with_sources(player, reduced_cost, orderer, 0, paid_for);
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

    // LAND FROM HAND — requires empty stack (sorcery-speed)
    if ((game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
        game.player_a_turn == game.player_a_has_priority && stack_manager->is_empty() &&
        global_coordinator.entity_has_component<Player>(priority_player_entity)) {
        auto &player = global_coordinator.GetComponent<Player>(priority_player_entity);

        // Compute effective land play limit (base 1 + AdjustLandPlays statics)
        int land_play_limit = 1;
        bool may_play_from_graveyard = false;
        for (const auto &as : g_active_statics) {
            if (as.controller != priority_player) continue;
            if (as.sa->adjust_land_plays > 0) land_play_limit += as.sa->adjust_land_plays;
            if (as.sa->may_play_from_graveyard) may_play_from_graveyard = true;
        }

        if (player.lands_played_this_turn < land_play_limit) {
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
            // Check graveyard for lands if MayPlay from graveyard is active
            if (may_play_from_graveyard) {
                Entity max_e = global_coordinator.GetMaxIssuedEntity();
                for (Entity gy_e = 0; gy_e < max_e; ++gy_e) {
                    if (!global_coordinator.entity_has_component<Zone>(gy_e)) continue;
                    auto &gz = global_coordinator.GetComponent<Zone>(gy_e);
                    if (gz.location != Zone::GRAVEYARD || gz.owner != priority_player) continue;
                    if (!global_coordinator.entity_has_component<CardData>(gy_e)) continue;
                    auto &gcd = global_coordinator.GetComponent<CardData>(gy_e);
                    bool is_land = false;
                    for (auto &t : gcd.types)
                        if (t.kind == TYPE && t.name == "Land") { is_land = true; break; }
                    if (is_land) {
                        std::string desc = "Play " + gcd.name + " (from graveyard)";
                        LegalAction la(SPECIAL_ACTION, gy_e, desc);
                        la.category = ActionCategory::PLAY_LAND;
                        actions.push_back(la);
                    }
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
        // Flash keyword grants instant-speed casting
        if (!is_instant) {
            for (const auto &kw : card_data.keywords) {
                if (kw == "Flash") { is_instant = true; break; }
            }
        }
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
        // and that any ConditionPresent$ castability condition is met
        bool tgt_ok = true;
        bool condition_ok = true;
        for (const auto &ab : card_data.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            if (!ab.condition_present.empty())
                condition_ok = check_condition_present(ab, priority_player, orderer);
            break;
        }
        // Machine mode: don't offer spells with conditional destroy if no target would pass
        // (e.g. Fatal Push: only show if a creature with CMC <= revolt threshold exists)
        if (InputLogger::instance().is_machine_mode() && tgt_ok && condition_ok) {
            for (const auto &ab : card_data.abilities) {
                if (ab.ability_type != Ability::SPELL) continue;
                if (ab.condition_present.find("cmcLEX") != std::string::npos &&
                    !ab.dynamic_amount_expr.empty()) {
                    // Evaluate Revolt threshold inline
                    int threshold = 2;
                    if (ab.dynamic_amount_expr.find("Count$Revolt.") != std::string::npos) {
                        size_t dot1 = ab.dynamic_amount_expr.find("Revolt.") + 7;
                        size_t dot2 = ab.dynamic_amount_expr.find('.', dot1);
                        int high_val = std::stoi(ab.dynamic_amount_expr.substr(dot1, dot2 - dot1));
                        int low_val = std::stoi(ab.dynamic_amount_expr.substr(dot2 + 1));
                        bool revolt = (priority_player == Zone::PLAYER_A)
                            ? cur_game.revolt_player_a : cur_game.revolt_player_b;
                        threshold = revolt ? high_val : low_val;
                    }
                    bool any_valid = false;
                    for (auto ce : mEntities) {
                        if (!global_coordinator.entity_has_component<Creature>(ce)) continue;
                        if (!global_coordinator.entity_has_component<Zone>(ce)) continue;
                        auto &cz = global_coordinator.GetComponent<Zone>(ce);
                        if (cz.location != Zone::BATTLEFIELD) continue;
                        if (!global_coordinator.entity_has_component<CardData>(ce)) continue;
                        int cmc = static_cast<int>(global_coordinator.GetComponent<CardData>(ce).mana_cost.size());
                        if (cmc <= threshold) { any_valid = true; break; }
                    }
                    if (!any_valid) tgt_ok = false;
                }
                break;
            }
        }

        auto pf_it = cur_game.payment_fail_counts.find(card_entity);
        bool payment_blocked = pf_it != cur_game.payment_fail_counts.end() && pf_it->second >= 2;
        if (can_cast_now && tgt_ok && condition_ok && !payment_blocked) {
            std::string desc = "Cast " + card_data.name;
            LegalAction la(CAST_SPELL, card_entity, desc);
            la.category = ActionCategory::CAST_SPELL;

            // Check RaiseCost and CantBeCast statics from cached active_statics
            bool card_is_creature = false;
            for (auto &t : card_data.types)
                if (t.kind == TYPE && t.name == "Creature") { card_is_creature = true; break; }
            int raise_total = 0;
            bool cast_blocked = false;
            for (const auto &as : g_active_statics) {
                if (as.sa->category == "RaiseCost") {
                    if (as.sa->raise_cost_filter == "nonCreature" && card_is_creature) continue;
                    raise_total += as.sa->raise_cost;
                } else if (as.sa->category == "CantBeCast") {
                    // Skip if the spell doesn't match the filter (creatures are unaffected by nonCreature restriction)
                    if (as.sa->cant_cast_filter.find("nonCreature") != std::string::npos && card_is_creature)
                        continue;
                    Entity pp_entity = (priority_player == Zone::PLAYER_A)
                        ? cur_game.player_a_entity : cur_game.player_b_entity;
                    auto &pp = global_coordinator.GetComponent<Player>(pp_entity);
                    if (as.sa->cant_cast_limit_per_turn > 0 &&
                        static_cast<int>(pp.noncreature_spells_cast_this_turn) >= as.sa->cant_cast_limit_per_turn) {
                        cast_blocked = true;
                    }
                }
            }
            if (cast_blocked) continue;

            ManaValue effective_cost = card_data.mana_cost;
            for (int ri = 0; ri < raise_total; ri++) effective_cost.insert(GENERIC);

            // X-cost spells: base cost (without X) is enough to be castable;
            // X value is chosen at cast time in action_processor
            bool can_regular = card_data.has_delve
                ? can_afford_with_delve(priority_player, effective_cost, orderer, card_entity)
                : can_afford_with_sources(priority_player, effective_cost, orderer, 0, card_entity);

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
    // checking graveyard for flashback spells
    for (auto gy_entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Zone>(gy_entity)) continue;
        auto &gz = global_coordinator.GetComponent<Zone>(gy_entity);
        if (gz.location != Zone::GRAVEYARD || gz.owner != priority_player) continue;
        if (!global_coordinator.entity_has_component<CardData>(gy_entity)) continue;
        auto &gcd = global_coordinator.GetComponent<CardData>(gy_entity);
        if (!gcd.has_flashback) continue;

        bool is_instant = false;
        for (auto &type : gcd.types) {
            if (type.kind == TYPE && type.name == "Instant") { is_instant = true; break; }
        }
        bool can_cast_now = false;
        if (is_instant) {
            can_cast_now = true;
        } else {
            can_cast_now = (game.cur_step == FIRST_MAIN || game.cur_step == SECOND_MAIN) &&
                           (game.player_a_turn == game.player_a_has_priority) && stack_empty;
        }
        if (!can_cast_now) continue;

        bool tgt_ok = true;
        for (const auto &ab : gcd.abilities) {
            if (ab.ability_type != Ability::SPELL) continue;
            tgt_ok = has_legal_targets(ab, orderer);
            break;
        }
        if (!tgt_ok) continue;

        // Check affordability: flashback mana cost + life cost
        bool can_afford_fb = can_afford_with_sources(priority_player, gcd.flashback_mana_cost, orderer, 0, gy_entity);
        if (can_afford_fb && gcd.flashback_alt_cost.life_cost > 0) {
            Entity pp_entity = (priority_player == Zone::PLAYER_A)
                ? cur_game.player_a_entity : cur_game.player_b_entity;
            if (global_coordinator.GetComponent<Player>(pp_entity).life_total < gcd.flashback_alt_cost.life_cost)
                can_afford_fb = false;
        }
        if (!can_afford_fb) continue;

        LegalAction fb_la(CAST_SPELL, gy_entity, "Cast " + gcd.name + " (flashback)");
        fb_la.category = ActionCategory::CAST_SPELL;
        fb_la.use_flashback = true;
        actions.push_back(fb_la);
    }
    // checking permanents for activated abilities
    // mana abilities parsed last, after pending_actions complete
    // Simple tap-only mana sources collected via shared function
    std::vector<LegalAction> legal_mana_abilities = collect_mana_legal_actions(priority_player, orderer);
    for (auto entity : orderer->mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;
        auto &permanent = global_coordinator.GetComponent<Permanent>(entity);
        if (permanent.controller != priority_player) continue;
        if (permanent.is_phased_out) continue;

        // Check if any CantBeActivated static suppresses this permanent's abilities
        bool cant_activate = false;
        for (const auto &as : g_active_statics) {
            if (as.sa->category != "CantBeActivated" || as.sa->cant_activate_card_filter.empty()) continue;
            if (as.sa->cant_activate_card_filter == "Artifact") {
                for (auto &t : permanent.types)
                    if (t.kind == TYPE && t.name == "Artifact") { cant_activate = true; break; }
            }
            if (cant_activate) break;
        }
        if (cant_activate) continue;

        for (auto ab : permanent.abilities) {
            if (ab.ability_type != Ability::ACTIVATED) continue;
            if (ab.activation_zone == Zone::HAND) continue;  // hand-only ability, not usable from battlefield
            // todo handle this elswewhere, tapping check
            if (ab.tap_cost && permanent.is_tapped) continue;
            if (ab.tap_cost && permanent.has_summoning_sickness &&
                global_coordinator.entity_has_component<Creature>(entity)) {
                auto &cr = global_coordinator.GetComponent<Creature>(entity);
                bool has_haste = false;
                for (const auto &kw : cr.keywords) {
                    if (kw == "Haste") { has_haste = true; break; }
                }
                if (!has_haste) continue;
            }
            // Activation limit check
            if (ab.activation_limit > 0 && ab.activations_this_turn >= ab.activation_limit) continue;
            // sac_cost_spec: require controller has a permanent matching type
            if (!ab.sac_cost_spec.empty()) {
                bool found_sac = false;
                for (auto e2 : orderer->mEntities) {
                    if (!global_coordinator.entity_has_component<Permanent>(e2)) continue;
                    auto &sz = global_coordinator.GetComponent<Zone>(e2);
                    if (sz.location != Zone::BATTLEFIELD) continue;
                    auto &sp = global_coordinator.GetComponent<Permanent>(e2);
                    if (sp.controller != priority_player) continue;
                    // match semicolon-separated subtypes in sac_cost_spec
                    const std::string &spec = ab.sac_cost_spec;
                    size_t pp = 0;
                    while (pp <= spec.size()) {
                        size_t sc = spec.find(';', pp);
                        if (sc == std::string::npos) sc = spec.size();
                        std::string sub = spec.substr(pp, sc - pp);
                        for (auto &t2 : sp.types) {
                            if (t2.name == sub) { found_sac = true; break; }
                        }
                        if (found_sac) break;
                        pp = sc + 1;
                    }
                    if (found_sac) break;
                }
                if (!found_sac) continue;
            }
            // Return cost: require controller has a land of given subtype
            if (!ab.return_cost_type.empty()) {
                bool found_ret = false;
                for (auto e2 : orderer->mEntities) {
                    if (!global_coordinator.entity_has_component<Permanent>(e2)) continue;
                    auto &sz = global_coordinator.GetComponent<Zone>(e2);
                    if (sz.location != Zone::BATTLEFIELD) continue;
                    auto &sp = global_coordinator.GetComponent<Permanent>(e2);
                    if (sp.controller != priority_player) continue;
                    for (auto &t2 : sp.types) {
                        if (t2.name == ab.return_cost_type) { found_ret = true; break; }
                    }
                    if (found_ret) break;
                }
                if (!found_ret) continue;
            }
            if (ab.category == "AddMana" && !ab.instant_speed) {
                // Normal mana abilities collected via collect_mana_legal_actions above
                // InstantSpeed$ abilities (e.g. LED) are not mana abilities and go on the stack
                continue;
            } else {
                // Non-mana activated ability (e.g. ChangeZone for fetch lands, Destroy for Wasteland)
                if (!ab.activation_mana_cost.empty() && !can_afford_with_sources(priority_player, ab.activation_mana_cost, orderer)) continue;
                if (ab.valid_tgts != "N_A" && !has_legal_targets(ab, orderer)) continue;
                { auto it = cur_game.payment_fail_counts.find(ab.source);
                  if (it != cur_game.payment_fail_counts.end() && it->second >= 2) continue; }
                std::string src_name = entity_name(ab.source);
                std::string desc = "Activate " + src_name + " (" + ab.category + ")";
                LegalAction non_mana_la(ACTIVATE_ABILITY, ab.source, ab, desc);
                non_mana_la.category = ActionCategory::ACTIVATE_ABILITY;
                actions.push_back(non_mana_la);
            }
        }
    }
    // Check hand for cards with ActivationZone$ Hand abilities (e.g. Talon Gates of Madara)
    for (auto card_entity : hand) {
        auto &card_data = global_coordinator.GetComponent<CardData>(card_entity);
        for (const auto &ab : card_data.abilities) {
            if (ab.ability_type != Ability::ACTIVATED) continue;
            if (ab.activation_zone != Zone::HAND) continue;
            // Check mana affordability
            if (!ab.activation_mana_cost.empty() && !can_afford_with_sources(priority_player, ab.activation_mana_cost, orderer)) continue;
            // Check target legality
            if (ab.valid_tgts != "N_A" && ab.target_min > 0 && !has_legal_targets(ab, orderer)) continue;
            // sac_cost_spec: require controller has a permanent matching type
            if (!ab.sac_cost_spec.empty()) {
                bool found_sac = false;
                for (auto e2 : orderer->mEntities) {
                    if (!global_coordinator.entity_has_component<Permanent>(e2)) continue;
                    auto &sz = global_coordinator.GetComponent<Zone>(e2);
                    if (sz.location != Zone::BATTLEFIELD) continue;
                    auto &sp = global_coordinator.GetComponent<Permanent>(e2);
                    if (sp.controller != priority_player) continue;
                    const std::string &spec = ab.sac_cost_spec;
                    size_t pp = 0;
                    while (pp <= spec.size()) {
                        size_t sc = spec.find(';', pp);
                        if (sc == std::string::npos) sc = spec.size();
                        std::string sub = spec.substr(pp, sc - pp);
                        for (auto &t2 : sp.types) {
                            if (t2.name == sub) { found_sac = true; break; }
                        }
                        if (found_sac) break;
                        pp = sc + 1;
                    }
                    if (found_sac) break;
                }
                if (!found_sac) continue;
            }
            { auto it = cur_game.payment_fail_counts.find(card_entity);
              if (it != cur_game.payment_fail_counts.end() && it->second >= 2) continue; }
            std::string desc = "Activate " + card_data.name + " from hand (" + ab.category + ")";
            LegalAction la(ACTIVATE_ABILITY, card_entity, ab, desc);
            la.category = ActionCategory::ACTIVATE_ABILITY;
            actions.push_back(la);
        }
    }

    // not filtering mana abilities based on if they contribute to a spell- will revisit this if it makes ML harder
    /*
    for (auto &ma : useful_mana_abilities(legal_mana_abilities, pending_actions)) {
        actions.push_back(ma);
    }
    */
    if (!InputLogger::instance().is_machine_mode()) {
        for (auto &ma : legal_mana_abilities) {
            actions.push_back(ma);
        }
    }
    return actions;
}