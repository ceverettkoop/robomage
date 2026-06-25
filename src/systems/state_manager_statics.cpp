#include "state_manager.h"
#include "state_manager_internal.h"

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "../action_processor.h"
#include "../card_vocab.h"
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
#include "../svar_eval.h"
#include "../systems/stack_manager.h"
#include "replacement_effects.h"
#include "continuous_effects.h"
#include "orderer.h"

int active_raise_cost_for(const CardData &card_data) {
    bool is_creature = is_creature_card(card_data);
    int total = 0;
    for (const auto &as : g_active_statics) {
        if (as.sa->category != "RaiseCost") continue;
        if (as.sa->raise_cost_filter == "nonCreature" && is_creature) continue;
        if (as.sa->match_named_card) {
            if (!global_coordinator.entity_has_component<Permanent>(as.entity)) continue;
            auto &src = global_coordinator.GetComponent<Permanent>(as.entity);
            if (src.chosen_name.empty() || src.chosen_name != card_data.name) continue;
        }
        total += as.sa->raise_cost;
    }
    return total;
}

ManaValue effective_base_cost(const CardData &card_data) {
    ManaValue cost = card_data.mana_cost;
    int raise_total = active_raise_cost_for(card_data);
    for (int ri = 0; ri < raise_total; ri++) cost.insert(GENERIC);
    return cost;
}

static void add_keywords_from_spec(Creature &cr, const std::string &spec);
static void remove_keywords_from_spec(Creature &cr, const std::string &spec);

// Apply/remove a " & "-delimited keyword spec (e.g. "Flying & Trample") to a creature.
static void add_keywords_from_spec(Creature &cr, const std::string &spec) {
    size_t p = 0;
    while (p < spec.size()) {
        size_t sep = spec.find(" & ", p);
        if (sep == std::string::npos) sep = spec.size();
        std::string kw = spec.substr(p, sep - p);
        if (!kw.empty()) cr.keywords.push_back(kw);
        p = (sep < spec.size()) ? sep + 3 : sep;
    }
}

static void remove_keywords_from_spec(Creature &cr, const std::string &spec) {
    size_t p = 0;
    while (p < spec.size()) {
        size_t sep = spec.find(" & ", p);
        if (sep == std::string::npos) sep = spec.size();
        std::string kw = spec.substr(p, sep - p);
        if (!kw.empty()) {
            auto it = std::find(cr.keywords.begin(), cr.keywords.end(), kw);
            if (it != cr.keywords.end()) cr.keywords.erase(it);
        }
        p = (sep < spec.size()) ? sep + 3 : sep;
    }
}

std::string entity_name(Entity e) {
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
                bootstrap_token_components(entity, token, zone.controller, game.timestamp);
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
            auto &card_data = global_coordinator.GetComponent<CardData>(entity);
            bool is_creature = is_creature_card(card_data);  // can be creature and land
            bool is_land = is_land_card(card_data);
            int etb_p1p1 = 0;  // +1/+1 counters this permanent enters with (614.1c), applied once the Creature exists
            // providing permanent component if doesn't have
            if (!global_coordinator.entity_has_component<Permanent>(entity)) {
                Permanent perm;
                perm.name = card_data.name;
                perm.types = card_data.types;
                perm.controller = zone.controller;
                perm.has_summoning_sickness = is_creature;
                perm.is_tapped = false;
                // Replacement effects at the point the permanent enters (rule 614): "enters
                // tapped" (a self-replacement, or a fetch's "onto the battlefield tapped") and
                // "enters with +1/+1 counters". The counters are applied once the Creature
                // component exists (below).
                {
                    ReplacementEvent rev;
                    rev.type = ReplacementEvent::ENTERS_BATTLEFIELD;
                    rev.entity = entity;
                    rev.affected_player = zone.controller;  // 616.1: the permanent's controller chooses
                    replacement::dispatch(rev);
                    perm.is_tapped = rev.enters_tapped;
                    etb_p1p1 = rev.etb_p1p1;
                }
                if (perm.is_tapped) game_log("%s enters tapped.\n", perm.name.c_str());
                // Spell was cast for its evoke cost — mark the permanent so its evoke
                // self-sacrifice ETB trigger fires (consumed one-shot here).
                if (game.pending_evoked.erase(entity)) perm.evoked = true;
                // Planeswalkers enter with loyalty counters equal to printed loyalty (306.5b).
                if (is_planeswalker_card(card_data)) perm.counters["LOYALTY"] = card_data.starting_loyalty;
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
                creature.base_power = static_cast<int>(card_data.power);
                creature.base_toughness = static_cast<int>(card_data.toughness);
                creature.keywords = card_data.keywords;
                recompute_pt(creature);
                global_coordinator.AddComponent(entity, creature);
                // damage component
                Damage damage;
                damage.damage_counters = 0;
                global_coordinator.AddComponent(entity, damage);

                // Apply the "enters with" +1/+1 counters chosen by the ETB replacement
                // dispatch above (rule 614.1c), now that the Creature component exists.
                if (etb_p1p1 > 0) {
                    add_counters(entity, "P1P1", etb_p1p1);
                    auto &cr = global_coordinator.GetComponent<Creature>(entity);
                    game_log("%s enters with %d +1/+1 counter(s) (%u/%u).\n",
                        card_data.name.c_str(), etb_p1p1, cr.power, cr.toughness);
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
                Entity player_entity = get_player_entity(perm_ref.controller);
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
                        if (!is_creature_card(cd2)) continue;
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

            // ETBReplacement: choose a card name (Disruptor Flute). Candidates are the
            // distinct vocab cards in the opponent's deck; the chosen name keys this
            // permanent's Card.NamedCard RaiseCost / CantBeActivated statics.
            if (card_data.has_etb_name_card) {
                auto &perm_ref = global_coordinator.GetComponent<Permanent>(entity);
                if (perm_ref.chosen_name.empty()) {
                    Zone::Ownership opp = (perm_ref.controller == Zone::PLAYER_A)
                        ? Zone::PLAYER_B : Zone::PLAYER_A;
                    // Distinct opponent-owned vocab card names, each with a representative
                    // entity (so the per-action card id encodes the candidate) and a copy
                    // count for ordering.
                    std::vector<std::string> names;
                    std::vector<Entity> reps;
                    std::vector<int> copies;
                    for (auto e2 : mEntities) {
                        if (!global_coordinator.entity_has_component<CardData>(e2)) continue;
                        if (!global_coordinator.entity_has_component<Zone>(e2)) continue;
                        if (global_coordinator.GetComponent<Zone>(e2).owner != opp) continue;
                        auto &cd2 = global_coordinator.GetComponent<CardData>(e2);
                        if (card_name_to_index(cd2.name) < 0) continue;  // restrict to vocab cards
                        bool found = false;
                        for (size_t i = 0; i < names.size(); i++)
                            if (names[i] == cd2.name) { copies[i]++; found = true; break; }
                        if (!found) { names.push_back(cd2.name); reps.push_back(e2); copies.push_back(1); }
                    }
                    if (!names.empty()) {
                        // Order by copies desc, then name asc (deterministic for replay)
                        std::vector<size_t> order(names.size());
                        for (size_t i = 0; i < order.size(); i++) order[i] = i;
                        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
                            if (copies[a] != copies[b]) return copies[a] > copies[b];
                            return names[a] < names[b];
                        });
                        if (order.size() > static_cast<size_t>(MAX_ACTIONS))
                            order.resize(MAX_ACTIONS);
                        std::vector<LegalAction> name_choices;
                        for (size_t idx : order) {
                            LegalAction la(PASS_PRIORITY, reps[idx], "Name card: " + names[idx]);
                            la.category = ActionCategory::OTHER_CHOICE;
                            la.card_is_public = true;
                            name_choices.push_back(la);
                        }
                        bool prev_priority = cur_game.player_a_has_priority;
                        cur_game.player_a_has_priority = (perm_ref.controller == Zone::PLAYER_A);
                        game_log("Choose a card name for %s:\n", perm_ref.name.c_str());
                        int choice = InputLogger::instance().get_input(name_choices);
                        cur_game.player_a_has_priority = prev_priority;
                        perm_ref.chosen_name = names[order[static_cast<size_t>(choice)]];
                        game_log("%s names card: %s\n",
                                 player_name(perm_ref.controller).c_str(), perm_ref.chosen_name.c_str());
                    }
                }
            }

        } else {  // off battlefield, check to remove
            if (global_coordinator.entity_has_component<Permanent>(entity)) {
                // Clear any equipment attachment links before the Permanent is removed, so no
                // dangling reference survives (704.5n). A creature leaving the battlefield
                // unattaches its equipment (which stays on the battlefield); an equipment
                // leaving clears the host creature's back-link.
                auto &perm = global_coordinator.GetComponent<Permanent>(entity);
                if (perm.equipped_by != 0 &&
                    global_coordinator.entity_has_component<Permanent>(perm.equipped_by)) {
                    global_coordinator.GetComponent<Permanent>(perm.equipped_by).equipped_to = 0;
                }
                if (perm.equipped_to != 0 &&
                    global_coordinator.entity_has_component<Permanent>(perm.equipped_to)) {
                    global_coordinator.GetComponent<Permanent>(perm.equipped_to).equipped_by = 0;
                }
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
        if (type.kind == SUBTYPE && is_basic_land_subtype(type.name)) {
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
                if (t.kind == SUBTYPE && is_basic_land_subtype(t.name)) continue;
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

// Preamble for the continuous-effects engine: rebuild the g_active_statics cache from
// every battlefield permanent and evaluate each gathered static's condition exactly
// once (stored in ActiveStatic::condition_met) so every layer applier reads the same
// result. Resetting each creature's static_*_bonus here means continuous buffs are
// rebuilt from scratch each pass and need no per-ability revert.
void StateManager::gather_active_statics(Game &game) {
    (void)game;
    g_active_statics.clear();

    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Permanent>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        auto &zone = global_coordinator.GetComponent<Zone>(entity);
        if (zone.location != Zone::BATTLEFIELD) continue;

        auto &perm = global_coordinator.GetComponent<Permanent>(entity);
        if (global_coordinator.entity_has_component<Creature>(entity)) {
            auto &cr = global_coordinator.GetComponent<Creature>(entity);
            cr.static_power_bonus = 0;
            cr.static_toughness_bonus = 0;
        }
        if (perm.is_phased_out) continue;
        if (perm.transformed) {
            for (auto &sa : perm.static_abilities) sa.applied = false;
            continue;
        }
        for (auto &sa : perm.static_abilities)
            g_active_statics.push_back({entity, &sa, perm.controller, false});
    }

    // Evaluate only the conditions actually referenced; compute each at most once per
    // player rather than once per permanent.
    bool need_delirium_a = false, need_delirium_b = false;
    for (auto &a : g_active_statics) {
        if (a.sa->condition == "Delirium") {
            if (a.controller == Zone::PLAYER_A) need_delirium_a = true;
            else                                need_delirium_b = true;
        }
    }
    bool delirium_a = need_delirium_a ? check_delirium(Zone::PLAYER_A, mEntities) : false;
    bool delirium_b = need_delirium_b ? check_delirium(Zone::PLAYER_B, mEntities) : false;

    for (auto &a : g_active_statics) {
        if (a.sa->condition.empty() && a.sa->check_svar_expr.empty()) {
            a.condition_met = true;
        } else if (a.sa->condition == "Delirium") {
            a.condition_met = (a.controller == Zone::PLAYER_A) ? delirium_a : delirium_b;
        } else if (!a.sa->check_svar_expr.empty()) {
            // SVar-based condition (e.g. Keen-Eyed Curator: GE4 distinct card types
            // among exiled_with). a.entity is the source permanent the SVar belongs to.
            int svar_val = evaluate_sa_svar(a.sa->check_svar_expr, a.controller, a.entity);
            a.condition_met = compare_svar(svar_val, a.sa->svar_compare);
        } else {
            a.condition_met = false;  // unrecognised condition — treat as unmet
        }
    }
}

// Layer 6 (rule 613.1f): ability-adding/removing continuous effects. Today this is
// keyword grants from "Continuous" statics, granted/removed on condition transitions
// (tracked by StaticAbility::applied). P/T is handled entirely in layer 7, so this
// pass never touches static_*_bonus — only keywords and the applied/last_applied state.
void StateManager::apply_layer6_ability_effects() {
    for (auto &a : g_active_statics) {
        if (a.sa->category != "Continuous") continue;
        // Characteristic-defining P/T statics are pure layer 7a (no keyword); skip them
        // here exactly as the original combined loop did before the keyword branch.
        if (a.sa->characteristic_defining &&
            (!a.sa->set_power_svar.empty() || !a.sa->set_toughness_svar.empty()))
            continue;

        // Determine which entity receives the grant (source or equipped creature).
        // Affected$ is stored verbatim (e.g. "Creature.EquippedBy"), so match by substring.
        Entity target_entity = a.entity;
        if (a.sa->affected.find("EquippedBy") != std::string::npos) {
            if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
            target_entity = global_coordinator.GetComponent<Permanent>(a.entity).equipped_to;
        }

        // If the grant moved to a different creature (e.g. equipment re-attached),
        // strip granted keywords from the previous one. P/T needs no manual revert:
        // layer 7 rebuilds every static bonus from zero each pass.
        if (a.sa->applied && a.sa->last_applied_entity != target_entity) {
            Entity prev = static_cast<Entity>(a.sa->last_applied_entity);
            if (prev != 0 && global_coordinator.entity_has_component<Creature>(prev) &&
                !a.sa->add_keyword.empty()) {
                remove_keywords_from_spec(global_coordinator.GetComponent<Creature>(prev),
                                          a.sa->add_keyword);
            }
            a.sa->applied = false;
        }

        if (target_entity == 0 || !global_coordinator.entity_has_component<Creature>(target_entity)) {
            // No valid target; mark unapplied so keywords re-grant when one appears.
            if (a.sa->applied) a.sa->applied = false;
            continue;
        }

        auto &cr = global_coordinator.GetComponent<Creature>(target_entity);
        const std::string name_for_log = entity_name(target_entity);

        if (a.condition_met && !a.sa->applied) {
            if (!a.sa->add_keyword.empty()) add_keywords_from_spec(cr, a.sa->add_keyword);
            a.sa->applied = true;
            a.sa->last_applied_entity = static_cast<uint32_t>(target_entity);
            game_log("%s gains %s%s(%s)\n", name_for_log.c_str(),
                     a.sa->add_power != 0 ? (std::to_string(a.sa->add_power) + "/" +
                                              std::to_string(a.sa->add_toughness) + " ").c_str() : "",
                     !a.sa->add_keyword.empty() ? (a.sa->add_keyword + " ").c_str() : "",
                     a.sa->condition.empty() ? "always" : a.sa->condition.c_str());
        } else if (!a.condition_met && a.sa->applied) {
            if (!a.sa->add_keyword.empty()) remove_keywords_from_spec(cr, a.sa->add_keyword);
            a.sa->applied = false;
            game_log("%s loses %s bonus\n", name_for_log.c_str(),
                     a.sa->condition.empty() ? "static" : a.sa->condition.c_str());
        }
    }
}

// Layer 7 (rule 613.4): power/toughness continuous effects, in sublayer order.
//   7a — characteristic-defining abilities set base P/T (rule 604.3), applied first
//        and unconditionally; recompute_pt then layers counters/pumps/etc. on top.
//   7c — additive +N/+N statics, collected as timestamp-tagged ContinuousEffects,
//        ordered per 613.7/613.8, then accumulated into static_*_bonus. Accumulation
//        is commutative so the result matches the old fixed-order sum; the ordering
//        machinery is wired in so future non-commutative layer-7 effects order
//        correctly. (Sublayers 7b "set" and 7d "switch" have storage slots on Creature
//        but no current-vocab source — see recompute_pt.)
void StateManager::apply_layer7_pt_effects() {
    // 7a — characteristic-defining base P/T.
    for (auto &a : g_active_statics) {
        if (a.sa->category != "Continuous") continue;
        if (!(a.sa->characteristic_defining &&
              (!a.sa->set_power_svar.empty() || !a.sa->set_toughness_svar.empty())))
            continue;
        if (!global_coordinator.entity_has_component<Creature>(a.entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(a.entity);
        cr.base_power = !a.sa->set_power_svar.empty()
            ? evaluate_sa_svar(a.sa->set_power_svar, a.controller) : 0;
        cr.base_toughness = !a.sa->set_toughness_svar.empty()
            ? evaluate_sa_svar(a.sa->set_toughness_svar, a.controller) : 0;
        a.sa->applied = true;
    }

    // 7c — additive modifications, gathered then ordered before accumulation.
    std::vector<ContinuousEffect> mods;
    for (auto &a : g_active_statics) {
        if (a.sa->category != "Continuous") continue;
        if (a.sa->characteristic_defining &&
            (!a.sa->set_power_svar.empty() || !a.sa->set_toughness_svar.empty()))
            continue;  // handled in 7a
        if (!a.condition_met) continue;

        Entity target_entity = a.entity;
        if (a.sa->affected.find("EquippedBy") != std::string::npos) {
            if (!global_coordinator.entity_has_component<Permanent>(a.entity)) continue;
            target_entity = global_coordinator.GetComponent<Permanent>(a.entity).equipped_to;
        }
        if (target_entity == 0 || !global_coordinator.entity_has_component<Creature>(target_entity)) continue;

        ContinuousEffect e;
        e.layer    = Layer::PT;
        e.sublayer = PTSublayer::MODIFY;
        e.source   = a.entity;
        e.affected = target_entity;
        e.delta_power = a.sa->add_power_svar.empty()
                            ? a.sa->add_power
                            : evaluate_sa_svar(a.sa->add_power_svar, a.controller);
        e.delta_toughness = a.sa->add_toughness_svar.empty()
                            ? a.sa->add_toughness
                            : evaluate_sa_svar(a.sa->add_toughness_svar, a.controller);
        // 613.7a — a static ability's effect has its source object's timestamp.
        e.timestamp = global_coordinator.entity_has_component<Permanent>(a.entity)
            ? global_coordinator.GetComponent<Permanent>(a.entity).timestamp_entered_battlefield
            : 0;
        mods.push_back(e);
    }

    order_continuous_effects(mods);
    for (const auto &e : mods) {
        if (!global_coordinator.entity_has_component<Creature>(e.affected)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(e.affected);
        cr.static_power_bonus     += e.delta_power;
        cr.static_toughness_bonus += e.delta_toughness;
    }
}

// Rule 613.11: continuous effects that modify the rules rather than characteristics,
// applied after all other continuous effects. Today: MustAttack.
void StateManager::apply_rules_modifying_effects() {
    for (auto &a : g_active_statics) {
        if (a.sa->category != "MustAttack") continue;
        if (!global_coordinator.entity_has_component<Creature>(a.entity)) continue;
        auto &cr = global_coordinator.GetComponent<Creature>(a.entity);
        cr.must_attack = a.condition_met;
        a.sa->applied = a.condition_met;
    }
}

// Recompute cached effective P/T from contributions for every battlefield creature.
void StateManager::recompute_battlefield_pt() {
    for (auto entity : mEntities) {
        if (!global_coordinator.entity_has_component<Creature>(entity)) continue;
        if (!global_coordinator.entity_has_component<Zone>(entity)) continue;
        if (global_coordinator.GetComponent<Zone>(entity).location != Zone::BATTLEFIELD) continue;
        recompute_pt(global_coordinator.GetComponent<Creature>(entity));
    }
}

