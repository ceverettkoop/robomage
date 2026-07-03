#include "ability.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/creature.h"
#include "../components/token.h"
#include "../components/types.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../error.h"
#include "../action_processor.h"
#include "../game_queries.h"
#include "../input_logger.h"
#include "../mana_system.h"
#include "../parse.h"
#include "../str_util.h"
#include "../svar_eval.h"
#include "../effects/effects.h"
#include "../systems/orderer.h"
#include "creature.h"
#include "damage.h"
#include "permanent.h"
#include "player.h"
#include "spell.h"

extern Coordinator global_coordinator;
extern Game cur_game;

// evaluate_dynamic_amount, search_multi_zone, and run_unless_loop are declared in
// effects.h (de-static'd so effect handlers in src/effects/ can share them);
// their definitions remain below.

// Bind a chained sub-ability's target before it resolves, reading the script's stated
// Defined$ intent rather than blanket-inheriting the parent's target. See definition below
// (CR 608.2c). Forward-declared per CLAUDE.md.
static void bind_sub_target(const Ability &parent, Ability &sub);

// Category-aware detail suffix for the "Resolving ability" log line. Display-only.
// Forward-declared per CLAUDE.md.
static std::string resolving_log_detail(const Ability &ab, std::shared_ptr<Orderer> orderer);

// ── Stack-object target matching (TargetType$) ──────────────────────────────
// TargetType$ is a comma-separated list of OR alternatives, each restricting the chosen
// target to a kind of object ON THE STACK: a "Spell[.quals]" alternative matches a spell
// (with optional color / type qualifiers), an "Activated"/"Triggered" alternative matches a
// standalone ability of that kind (CR 113.7). Consign to Memory's
// "Spell.Colorless,Triggered" is one such disjunction. Forward-declared per CLAUDE.md.
static bool stack_spell_alt_matches(const std::string &alt, Entity cand);
static bool stack_ability_alt_matches(const std::string &alt, Entity cand);
static bool target_type_matches_stack_object(const std::string &target_type, Entity cand);

// edge case of two identical abilities being applied from two sources not handled
bool Ability::identical_activated_ability(const Ability &other) {
    if (other.category != this->category) return false;
    if (other.valid_tgts != this->valid_tgts) return false;
    if (other.amount != this->amount) return false;
    // A Metalcraft-/condition-gated ability and an otherwise-identical ungated one are
    // distinct abilities (Urza's Workshop: plain "{T}: Add {C}" vs the Metalcraft
    // "{T}: Add {C} for each Urza's land"). Likewise two mana abilities that produce
    // different dynamic amounts are distinct. Without these the second is wrongly deduped.
    if (other.activation_condition != this->activation_condition) return false;
    if (other.dynamic_amount_expr != this->dynamic_amount_expr) return false;
    if (other.tap_cost != this->tap_cost) return false;
    if (other.activation_mana_cost != this->activation_mana_cost) return false;
    if (other.sac_self != this->sac_self) return false;
    if (other.change_type != this->change_type) return false;
    if (other.origin != this->origin) return false;
    if (other.destination != this->destination) return false;
    if (other.color != this->color) return false;
    if (other.mana_choices != this->mana_choices) return false;
    if (other.reflected_mana_filter != this->reflected_mana_filter) return false;
    if (other.restrict_to_chosen_type_creature != this->restrict_to_chosen_type_creature) return false;
    if (other.restrict_to_creature != this->restrict_to_creature) return false;
    if (other.adds_no_counter != this->adds_no_counter) return false;
    return true;
};

// A zone-search candidate is a card object, matched by its PRINTED characteristics through the
// shared filter matcher (game_queries.h). Thin local adapter so the search_zone call sites keep
// their (entity, spec, cmc_bound, cmc_op) shape; the dynamic mana-value bound flows through the
// MatchCtx. All qualifier grammar (colors, Colorless, Basic, P/T, subtypes, cmcLEX, …) now lives
// in the one shared evaluator.
static bool matches_filter_spec(Entity entity, const std::string &spec, int cmc_bound = -1,
    const std::string &cmc_op = "", Zone::Ownership you = Zone::UNKNOWN) {
    MatchCtx ctx;
    ctx.cmc_bound = cmc_bound;
    ctx.cmc_op = cmc_op;
    ctx.controller = you;  // the "you" reference for YouOwn/YouCtrl/OppOwn/OppCtrl in the filter
    return card_matches_filter(entity, spec, ctx);
}

// Searches a zone for cards whose types match any entry in the comma-separated
// change_type string. Presents all matches plus a "fail to find" option (index 0).
// Returns the chosen Entity, or 0 for fail to find.
// 0 is a valid entity but will always be player a  so is never correct
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner, Zone::ZoneValue zone,
    const std::string &change_type, bool mandatory, Zone::ZoneValue destination, bool reveal,
    int cmc_bound, const std::string &cmc_op) {
    //  comma-separated subtypes
    std::vector<std::string> subtypes = split(change_type, ',');

    // Collect zone contents
    std::vector<Entity> zone_contents;
    if (zone == Zone::LIBRARY) {
        zone_contents = orderer->get_library_contents(owner);
    } else if (zone == Zone::HAND) {
        zone_contents = orderer->get_hand(owner);
    } else if (zone == Zone::GRAVEYARD || zone == Zone::EXILE || zone == Zone::SIDEBOARD) {
        // Graveyard / face-up exile / sideboard ("outside the game") picks (Karn, the Great
        // Creator -2: choose an artifact card you own in exile or your sideboard). These zones
        // hold their cards as entities tagged by Zone owner, so enumerate by owner like the
        // graveyard. (Sideboard entities are instantiated at game start by generate_libraries
        // from the deck's SIDEBOARD: section, plus any test-harness --sideboard presets.)
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location == zone && z.owner == owner) zone_contents.push_back(e);
        }
    }

    // Filter to matching cards; empty change_type means all cards match
    std::vector<Entity> choices;
    if (change_type.empty()) {
        choices = zone_contents;
    } else {
        // Check if any filter spec uses extended syntax (dot/plus qualifiers)
        bool has_extended = false;
        for (auto &st : subtypes) {
            if (st.find('.') != std::string::npos || st.find('+') != std::string::npos) {
                has_extended = true;
                break;
            }
        }
        // A dynamic mana-value bound (Aether Vial) forces the extended path so each
        // candidate is gated by both type and mana value.
        if (cmc_bound >= 0) has_extended = true;

        for (auto entity : zone_contents) {
            bool matches = false;
            if (has_extended) {
                for (auto &st : subtypes) {
                    if (matches_filter_spec(entity, st, cmc_bound, cmc_op, owner)) {
                        matches = true;
                        break;
                    }
                }
            } else {
                auto &cd = global_coordinator.GetComponent<CardData>(entity);
                for (auto &t : cd.types) {
                    for (auto &st : subtypes) {
                        if (t.name == st) {
                            matches = true;
                            break;
                        }
                    }
                    if (matches) break;
                }
            }
            if (matches) choices.push_back(entity);
        }
    }

    const char *zone_name = (zone == Zone::LIBRARY)     ? "library"
                            : (zone == Zone::HAND)      ? "hand"
                            : (zone == Zone::GRAVEYARD) ? "graveyard"
                            : (zone == Zone::EXILE)     ? "exile"
                            : (zone == Zone::SIDEBOARD) ? "sideboard"
                                                        : "zone";
    // Determine category: library searches going to top of library use TOP_LIBRARY,
    // other library searches use SEARCH_LIBRARY, non-library zone picks use CHOOSE_CARD
    ActionCategory cat = (destination == Zone::LIBRARY && (zone == Zone::LIBRARY || zone == Zone::HAND))
                             ? ActionCategory::TOP_LIBRARY
                         : (zone == Zone::LIBRARY) ? ActionCategory::SEARCH_LIBRARY
                                                   : ActionCategory::CHOOSE_CARD;

    // Fail-to-find is shown when: not mandatory, OR zone is empty (nothing else to choose)
    bool show_fail_to_find = !mandatory || choices.empty();

    if (mandatory && choices.empty()) {
        // Nothing left to move; return immediately without prompting
        return 0;
    }

    if (zone == Zone::LIBRARY) {
        game_log("Searching %s's %s:\n", player_name(owner).c_str(), zone_name);
    } else {
        game_log("%s chooses a card from %s %s:\n", player_name(owner).c_str(), player_name(owner).c_str(), zone_name);
    }

    std::vector<LegalAction> search_actions;
    if (show_fail_to_find) {
        LegalAction ftf(PASS_PRIORITY, Entity(0), std::string("Fail to find"));
        ftf.category = cat;
        search_actions.push_back(ftf);
    }
    for (auto entity : choices) {
        auto &cd = global_coordinator.GetComponent<CardData>(entity);
        LegalAction la(PASS_PRIORITY, entity, cd.name);
        la.category = cat;
        la.card_is_public = reveal;
        search_actions.push_back(la);
    }

    int choice = InputLogger::instance().get_input(search_actions);
    // Map choice back: if fail-to-find is shown, index 0 = fail-to-find, 1..N = choices
    // If fail-to-find suppressed, index 0..N-1 = choices directly
    if (show_fail_to_find) {
        if (choice >= 1 && choice <= static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice - 1)];
        return 0;
    } else {
        if (choice >= 0 && choice < static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice)];
        return 0;
    }
}

// Searches multiple zones combined for cards matching change_type.
// Used by Doomsday (Origin$ Graveyard,Library).
Entity search_multi_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner,
    const std::vector<Zone::ZoneValue> &zones, const std::string &change_type, bool mandatory,
    Zone::ZoneValue destination, bool reveal) {
    // Collect contents from all zones
    std::vector<Entity> zone_contents;
    for (auto zone : zones) {
        if (zone == Zone::LIBRARY) {
            auto lib = orderer->get_library_contents(owner);
            zone_contents.insert(zone_contents.end(), lib.begin(), lib.end());
        } else if (zone == Zone::HAND) {
            auto hand = orderer->get_hand(owner);
            zone_contents.insert(zone_contents.end(), hand.begin(), hand.end());
        } else if (zone == Zone::GRAVEYARD || zone == Zone::EXILE || zone == Zone::SIDEBOARD) {
            // Graveyard / face-up exile / sideboard ("outside the game"), enumerated by Zone
            // owner — Karn, the Great Creator -2 searches Origin$ Sideboard,Exile.
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                auto &z = global_coordinator.GetComponent<Zone>(e);
                if (z.location == zone && z.owner == owner) zone_contents.push_back(e);
            }
        }
    }

    // Exclude already-remembered entities (e.g. Doomsday picking 5 cards one at a time)
    if (!cur_game.remembered_entities.empty()) {
        std::vector<Entity> filtered;
        for (auto e : zone_contents) {
            bool already = false;
            for (auto re : cur_game.remembered_entities) {
                if (re == e) {
                    already = true;
                    break;
                }
            }
            if (!already) filtered.push_back(e);
        }
        zone_contents = filtered;
    }

    // Filter by change_type — "Card" matches everything
    std::vector<Entity> choices;
    if (change_type.empty() || change_type == "Card") {
        choices = zone_contents;
    } else {
        // Parse comma-separated subtypes
        std::vector<std::string> subtypes = split(change_type, ',');
        bool has_extended = false;
        for (auto &st : subtypes) {
            if (st.find('.') != std::string::npos || st.find('+') != std::string::npos) {
                has_extended = true;
                break;
            }
        }
        for (auto entity : zone_contents) {
            bool matches = false;
            if (has_extended) {
                for (auto &st : subtypes) {
                    if (matches_filter_spec(entity, st, -1, "", owner)) {
                        matches = true;
                        break;
                    }
                }
            } else {
                auto &cd = global_coordinator.GetComponent<CardData>(entity);
                for (auto &t : cd.types) {
                    for (auto &st : subtypes) {
                        if (t.name == st) {
                            matches = true;
                            break;
                        }
                    }
                    if (matches) break;
                }
            }
            if (matches) choices.push_back(entity);
        }
    }

    bool show_fail_to_find = !mandatory || choices.empty();
    if (mandatory && choices.empty()) return 0;

    bool searches_library = false;
    std::string zone_list;
    for (auto zone : zones) {
        if (zone == Zone::LIBRARY) searches_library = true;
        const char *zn = (zone == Zone::LIBRARY)     ? "library"
                         : (zone == Zone::GRAVEYARD)  ? "graveyard"
                         : (zone == Zone::HAND)       ? "hand"
                         : (zone == Zone::EXILE)      ? "exile"
                         : (zone == Zone::SIDEBOARD)  ? "sideboard"
                                                      : "zone";
        if (!zone_list.empty()) zone_list += " and ";
        zone_list += zn;
    }
    game_log("Searching %s's %s:\n", player_name(owner).c_str(), zone_list.c_str());

    // A library search uses SEARCH_LIBRARY/TOP_LIBRARY; a pick from only non-library
    // zones (e.g. Karn's -2 over Sideboard,Exile) is a CHOOSE_CARD decision.
    ActionCategory cat = (destination == Zone::LIBRARY) ? ActionCategory::TOP_LIBRARY
                         : searches_library             ? ActionCategory::SEARCH_LIBRARY
                                                        : ActionCategory::CHOOSE_CARD;

    std::vector<LegalAction> search_actions;
    if (show_fail_to_find) {
        LegalAction ftf(PASS_PRIORITY, Entity(0), std::string("Fail to find"));
        ftf.category = cat;
        search_actions.push_back(ftf);
    }
    for (auto entity : choices) {
        auto &cd = global_coordinator.GetComponent<CardData>(entity);
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        const char *zone_label = (z.location == Zone::GRAVEYARD)  ? " (graveyard)"
                                 : (z.location == Zone::EXILE)     ? " (exile)"
                                 : (z.location == Zone::SIDEBOARD) ? " (sideboard)"
                                 : (z.location == Zone::HAND)      ? " (hand)"
                                                                   : " (library)";
        LegalAction la(PASS_PRIORITY, entity, cd.name + zone_label);
        la.category = cat;
        la.card_is_public = reveal;
        search_actions.push_back(la);
    }

    int choice = InputLogger::instance().get_input(search_actions);
    if (show_fail_to_find) {
        if (choice >= 1 && choice <= static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice - 1)];
        return 0;
    } else {
        if (choice >= 0 && choice < static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice)];
        return 0;
    }
}




// Offer `controller` the choice to discard `count` card(s) of their choice from hand to pay an
// unless-cost (CR 701.8). Returns true if the cost was NOT paid (they declined or have too few
// cards), i.e. the spell should be countered. Reusable by any "unless they discard N cards"
// effect; the chosen cards go to the graveyard (a public zone — record the reveal in the belief
// state). The caller has already set cur_game.player_a_has_priority to `controller`.
static bool run_discard_unless(size_t count, Zone::Ownership controller,
                               std::shared_ptr<Orderer> orderer) {
    std::vector<Entity> hand = orderer->get_hand(controller);
    bool can_pay = hand.size() >= count;  // CR 701.8: must discard the full count or none

    std::vector<LegalAction> actions;
    size_t pay_idx = actions.size();
    if (can_pay) {
        LegalAction pay(PASS_PRIORITY,
            std::string("Discard ") + std::to_string(count) +
            (count == 1 ? " card (spell is not countered)" : " cards (spell is not countered)"));
        pay.category = ActionCategory::PAY_UNLESS;
        actions.push_back(pay);
    }
    size_t decline_idx = actions.size();
    LegalAction decline(PASS_PRIORITY, std::string("Don't discard (spell is countered)"));
    decline.category = ActionCategory::PAY_UNLESS;
    actions.push_back(decline);

    int choice = InputLogger::instance().get_input(actions);
    if (!can_pay || choice == static_cast<int>(decline_idx)) {
        (void)pay_idx;
        return true;  // countered
    }

    // Pay: the payer chooses `count` distinct cards from hand to discard.
    for (size_t i = 0; i < count; i++) {
        std::vector<Entity> cur_hand = orderer->get_hand(controller);
        if (cur_hand.empty()) break;
        std::vector<LegalAction> dactions;
        for (auto e : cur_hand) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::DISCARD;
            dactions.push_back(la);
        }
        int dchoice = InputLogger::instance().get_input(dactions);
        Entity chosen = dactions[static_cast<size_t>(dchoice)].source_entity;
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        game_log("%s discards %s — spell is not countered\n",
                 player_name(controller).c_str(), cd.name.c_str());
        // The discarded card enters a public zone — record its identity in the belief state.
        mark_card_revealed(chosen, controller);
        orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
    }
    return false;  // not countered
}

// Returns true if the spell should be countered (controller declined or couldn't pay).
// kind: how the unless-cost is paid — {cost} generic mana (default), `cost` life (Ward—Pay N life,
// CR 702.21), or discard `cost` card(s) (Reality Smasher, CR 701.8). A life payment is only offered
// when the payer's life total >= cost (CR 119.4 — a player can't pay more life than they have).
bool run_unless_loop(
    size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for,
    UnlessPayKind kind) {
    // the target's controller decides whether to pay, not the Daze caster
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (controller == Zone::PLAYER_A);

    if (kind == UnlessPayKind::DISCARD) {
        bool countered = run_discard_unless(cost, controller, orderer);
        cur_game.player_a_has_priority = prev_priority;
        return countered;
    }

    if (kind == UnlessPayKind::LIFE) {
        auto &payer = global_coordinator.GetComponent<Player>(get_player_entity(controller));
        bool can_pay = payer.life_total >= static_cast<int>(cost);  // CR 119.4

        std::vector<LegalAction> unless_actions;
        size_t pay_idx = unless_actions.size();
        if (can_pay) {
            LegalAction pay(PASS_PRIORITY,
                std::string("Pay ") + std::to_string(cost) + " life (spell is not countered)");
            pay.category = ActionCategory::PAY_UNLESS;
            unless_actions.push_back(pay);
        }
        size_t decline_idx = unless_actions.size();
        LegalAction decline(PASS_PRIORITY, std::string("Don't pay (spell is countered)"));
        decline.category = ActionCategory::PAY_UNLESS;
        unless_actions.push_back(decline);

        int choice = InputLogger::instance().get_input(unless_actions);
        cur_game.player_a_has_priority = prev_priority;
        if (can_pay && choice == static_cast<int>(pay_idx)) {
            payer.life_total -= static_cast<int>(cost);
            game_log("%s pays %zu life — spell is not countered\n",
                player_name(controller).c_str(), cost);
            return false;
        }
        (void)decline_idx;
        return true;
    }

    if (kind == UnlessPayKind::ENERGY) {
        // Pay N energy ({E}, CR 122.1c) or the prevented effect happens (Static Prison's
        // "sacrifice CARDNAME unless you pay {E}"). Energy lives as an "ENERGY" counter on the
        // payer; pay_energy gates on having enough (CR 119.4-analogue for a resource cost).
        auto &payer = global_coordinator.GetComponent<Player>(get_player_entity(controller));
        bool can_pay = player_energy(payer) >= static_cast<int>(cost);

        std::vector<LegalAction> unless_actions;
        size_t pay_idx = unless_actions.size();
        if (can_pay) {
            LegalAction pay(PASS_PRIORITY,
                std::string("Pay {E} x") + std::to_string(cost));
            pay.category = ActionCategory::PAY_UNLESS;
            unless_actions.push_back(pay);
        }
        size_t decline_idx = unless_actions.size();
        LegalAction decline(PASS_PRIORITY, std::string("Don't pay"));
        decline.category = ActionCategory::PAY_UNLESS;
        unless_actions.push_back(decline);

        int choice = InputLogger::instance().get_input(unless_actions);
        cur_game.player_a_has_priority = prev_priority;
        if (can_pay && choice == static_cast<int>(pay_idx)) {
            pay_energy(payer, static_cast<int>(cost));
            game_log("%s pays %zu energy.\n", player_name(controller).c_str(), cost);
            return false;
        }
        (void)decline_idx;
        return true;
    }

    std::multiset<Colors> cond_cost;
    for (size_t i = 0; i < cost; i++) cond_cost.insert(GENERIC);

    while (true) {
        std::vector<LegalAction> unless_actions = collect_mana_legal_actions(controller, orderer);
        // Drop cost-bearing mana sources (Talon Gates' {1}{T}) whose activation cost the
        // FLOATING pool can't cover right now: the tap-a-source loop below pays activation
        // costs only from mana already floating, so offering such a source would either
        // produce free mana or (refused) re-offer the same menu forever to a deterministic
        // machine-mode agent. The menu is rebuilt each iteration, so the source reappears
        // as soon as enough mana has been floated.
        unless_actions.erase(
            std::remove_if(unless_actions.begin(), unless_actions.end(),
                [&](const LegalAction &la) {
                    return !la.ability.activation_mana_cost.empty() &&
                           !can_afford(controller, la.ability.activation_mana_cost);
                }),
            unless_actions.end());

        bool can_pay = can_afford(controller, cond_cost);
        size_t pay_idx = unless_actions.size();
        if (can_pay) {
            LegalAction pay(PASS_PRIORITY, std::string("Pay {") + std::to_string(cost) + "} (spell is not countered)");
            pay.category = ActionCategory::PAY_UNLESS;
            unless_actions.push_back(pay);
        }
        size_t decline_idx = unless_actions.size();
        {
            LegalAction decline(PASS_PRIORITY, std::string("Don't pay (spell is countered)"));
            decline.category = ActionCategory::PAY_UNLESS;
            unless_actions.push_back(decline);
        }

        int choice = InputLogger::instance().get_input(unless_actions);

        if (choice == static_cast<int>(decline_idx)) {
            cur_game.player_a_has_priority = prev_priority;
            return true;
        }

        if (can_pay && choice == static_cast<int>(pay_idx)) {
            spend_mana(controller, cond_cost, paid_for);
            game_log("%s pays {%zu} — spell is not countered\n", player_name(controller).c_str(), cost);
            cur_game.player_a_has_priority = prev_priority;
            return false;
        }

        if (choice >= 0 && choice < static_cast<int>(pay_idx)) {
            auto &chosen = unless_actions[static_cast<size_t>(choice)];
            Entity land = chosen.source_entity;
            auto &perm = global_coordinator.GetComponent<Permanent>(land);
            // A cost-bearing mana source (Talon Gates' {1}{T}) must pay its activation cost
            // from the floating pool before producing; refuse the activation (no tap, no
            // mana) while the pool can't cover it, and re-prompt so the player can float
            // mana from a cost-free source first.
            if (!chosen.ability.activation_mana_cost.empty()) {
                if (!can_afford(controller, chosen.ability.activation_mana_cost)) {
                    game_log("Cannot pay that ability's activation cost — tap another source for mana first.\n");
                    continue;
                }
                spend_mana(controller, chosen.ability.activation_mana_cost, land);
            }
            perm.is_tapped = true;
            add_mana(controller, chosen.ability.color, chosen.ability.amount);
            game_log("%s tapped %s for {%s}\n", player_name(controller).c_str(), perm.name.c_str(),
                mana_symbol(chosen.ability.color).c_str());
        }
    }
}

void Ability::fizzle(std::shared_ptr<Orderer> orderer) {
    // stack manager present behavior moves everything to graveyard or destroys it
    // so for now this is a stub
    game_log("%s fizzles\n", this->category.c_str());
    return;
}

// Does one "Spell[.quals]" TargetType alternative match a spell on the stack? Checks the
// stack-spell preconditions plus the alternative's own qualifiers: type negations
// (nonCreature / Instant|Sorcery-only), a positive color restriction (.Blue — Red Elemental
// Blast, CR 115.1), and Colorless (Consign to Memory: a spell with no color). `alt` is the
// single alternative (e.g. "Spell.Colorless"). Returns false for a candidate that is not a
// spell on the stack.
static bool stack_spell_alt_matches(const std::string &alt, Entity cand) {
    if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
    if (global_coordinator.GetComponent<Zone>(cand).location != Zone::STACK) return false;
    if (!global_coordinator.entity_has_component<Spell>(cand)) return false;
    bool non_creature_only = alt.find("nonCreature") != std::string::npos;
    bool instant_sorcery_only =
        (alt.find("Instant") != std::string::npos || alt.find("Sorcery") != std::string::npos) &&
        alt.find("Creature") == std::string::npos;
    if ((non_creature_only || instant_sorcery_only) &&
        global_coordinator.entity_has_component<CardData>(cand)) {
        auto &cd = global_coordinator.GetComponent<CardData>(cand);
        bool is_creature = false, is_instant = false, is_sorcery = false;
        for (auto &t : cd.types) {
            if (t.name == "Creature") is_creature = true;
            if (t.name == "Instant") is_instant = true;
            if (t.name == "Sorcery") is_sorcery = true;
        }
        if (non_creature_only && is_creature) return false;
        if (instant_sorcery_only && !is_instant && !is_sorcery) return false;
    }
    // Positive color restriction (.Blue — Red Elemental Blast, CR 115.1).
    if (global_coordinator.entity_has_component<CardData>(cand) &&
        !color_set_passes(alt, effective_colors(cand)))
        return false;
    // Colorless restriction (Consign to Memory): the spell must have NO color (CR 105.2c).
    if (alt.find("Colorless") != std::string::npos && !effective_colors(cand).empty())
        return false;
    return true;
}

// Does one "Activated"/"Triggered" TargetType alternative match a standalone ability on the
// stack (Stifle, Consign to Memory)? Spells have a Spell component and are excluded.
static bool stack_ability_alt_matches(const std::string &alt, Entity cand) {
    if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
    if (global_coordinator.GetComponent<Zone>(cand).location != Zone::STACK) return false;
    if (global_coordinator.entity_has_component<Spell>(cand)) return false;  // spells aren't abilities
    if (!global_coordinator.entity_has_component<Ability>(cand)) return false;
    auto &ab = global_coordinator.GetComponent<Ability>(cand);
    if (alt.find("Activated") != std::string::npos && ab.ability_type == Ability::ACTIVATED)
        return true;
    if (alt.find("Triggered") != std::string::npos && ab.ability_type == Ability::TRIGGERED)
        return true;
    return false;
}

// TargetType$ is an OR list of stack-object alternatives — split on commas and accept the
// candidate if ANY alternative matches it (CR 115.1 target restrictions are satisfied by any
// one named kind). Drives counterspells (Spell), Stifle (Activated,Triggered) and Consign to
// Memory (Spell.Colorless,Triggered) off one matcher.
static bool target_type_matches_stack_object(const std::string &target_type, Entity cand) {
    for (const std::string &alt : split(target_type, ',', /*skip_empty=*/true)) {
        bool is_ability_alt = (alt.find("Activated") != std::string::npos ||
                               alt.find("Triggered") != std::string::npos);
        bool is_spell_alt = (alt.find("Spell") != std::string::npos);
        if (is_ability_alt && stack_ability_alt_matches(alt, cand)) return true;
        if (is_spell_alt && stack_spell_alt_matches(alt, cand)) return true;
    }
    return false;
}

// "Hexproof from <color>" (CR 702.11e): a candidate protected by a turn-long
// cur_game.hexproof_from_colors_this_turn grant can't be targeted by a spell/ability an OPPONENT
// of the protected player controls whose SOURCE is one of the granted colors. Covers both the
// protected player (the player object) and any permanent that player controls. `source` is the
// targeting object (a spell card on the stack, or an ability's source permanent) whose
// ColorIdentity gives the color of the spell/ability for the comparison; `caster` is its
// controller. Returns true when the candidate is protected (so the target is illegal).
static bool target_has_color_hexproof(Entity cand, Entity source, Zone::Ownership caster) {
    if (cur_game.hexproof_from_colors_this_turn.empty()) return false;
    if (!global_coordinator.entity_has_component<ColorIdentity>(source)) return false;
    const auto &src_colors = global_coordinator.GetComponent<ColorIdentity>(source).colors;
    for (const auto &h : cur_game.hexproof_from_colors_this_turn) {
        // Only protects against an opponent's spell/ability (two-player: caster != protected player).
        if (caster == h.player) continue;
        // The candidate must be the protected player, or a permanent that player controls.
        bool is_protected_player = (cand == get_player_entity(h.player));
        bool is_protected_perm = global_coordinator.entity_has_component<Permanent>(cand) &&
                                 is_battlefield_permanent(cand, h.player);
        if (!is_protected_player && !is_protected_perm) continue;
        // The source must be one of the granted colors.
        for (Colors c : h.colors)
            if (src_colors.count(c)) return true;
    }
    return false;
}

// "Protection from everything" for a player (CR 702.16; The One Ring). A player covered by a
// cur_game.player_protection_from_everything grant can't be the target of a spell/ability an
// OPPONENT controls. `caster` is the targeting object's controller. Returns true when the
// candidate is the protected player and the targeting object belongs to their opponent.
static bool player_has_protection_from_everything(Entity cand, Zone::Ownership caster) {
    if (cur_game.player_protection_from_everything.empty()) return false;
    for (const auto &p : cur_game.player_protection_from_everything) {
        if (caster == p.player) continue;  // own spells/abilities can still target the player
        if (cand == get_player_entity(p.player)) return true;
    }
    return false;
}

// True if `needle` occurs in `hay` as a WHOLE word — not as a substring of a longer alphabetic
// token. A ValidTgts spec names a player target with the head type "Player", but control
// qualifiers embed it inside longer words (e.g. "ControlledBy TriggeredDefendingPlayer" on
// Frenzied Trapbreaker), where a raw substring scan would wrongly read it as a player target.
static bool valid_tgts_names_word(const std::string &hay, const char *needle) {
    const size_t nlen = std::char_traits<char>::length(needle);
    for (size_t p = hay.find(needle); p != std::string::npos; p = hay.find(needle, p + 1)) {
        const bool left_ok = (p == 0) || !std::isalpha(static_cast<unsigned char>(hay[p - 1]));
        const size_t end = p + nlen;
        const bool right_ok = (end == hay.size()) || !std::isalpha(static_cast<unsigned char>(hay[end]));
        if (left_ok && right_ok) return true;
    }
    return false;
}

// Single source of truth for target legality (see header). build_valid_targets
// enumerates candidates and filters them through this; is_target_valid re-runs the
// chosen target(s) through it at resolution. Keeping both on one predicate is what
// prevents the enumeration and re-verification rules from drifting apart.
bool Ability::is_legal_target(Entity cand, Zone::Ownership caster) const {
    if (cand == 0) return false;

    // Hexproof from <color> (Veil of Summer) — applies to players and permanents alike, so it is
    // checked up front before the type-specific branches below.
    if (target_has_color_hexproof(cand, source, caster)) return false;

    // Protection from everything for a player (The One Ring) — the protected player can't be
    // targeted by an opponent's spell/ability (CR 702.16e). Checked up front like hexproof.
    if (player_has_protection_from_everything(cand, caster)) return false;

    // ValidTgts$ ...Other (e.g. Solitude/Flickerwisp "another"/"other" target): the source
    // of the ability cannot be chosen as its own target (CR 115.1; "other" is a target
    // restriction). Enforced uniformly here so it applies to every target type. Match
    // "Other" only as a complete dot/plus-delimited qualifier token (Creature.Other,
    // Permanent.Other+nonLand), never as a substring of a longer subtype/name (so a future
    // "Brotherhood"/"Otherworldly" token can't spuriously forbid self-targeting).
    if (cand == source) {
        for (size_t p = valid_tgts.find("Other"); p != std::string::npos;
             p = valid_tgts.find("Other", p + 1)) {
            bool left_ok = (p > 0) && (valid_tgts[p - 1] == '.' || valid_tgts[p - 1] == '+');
            size_t end = p + 5;  // length of "Other"
            bool right_ok = (end == valid_tgts.size()) ||
                            valid_tgts[end] == '.' || valid_tgts[end] == '+';
            if (left_ok && right_ok) return false;
        }
    }

    // NOTE: Pyroblast/Hydroblast (ValidTgts$ Card + ConditionPresent$ <type>.<Color>)
    // intentionally do NOT restrict target legality by color — they may target any
    // spell/permanent and their counter/destroy effect is conditional on the target's
    // color (enforced in effects::counter / effects::destroy via
    // target_color_condition_met). By contrast Red Elemental Blast (ValidTgts$ Card.Blue /
    // Permanent.Blue) bakes blue into the target restriction itself, enforced below through
    // the shared filter evaluator on both the stack-spell and battlefield-permanent paths
    // (CR 115.1: target restrictions are checked when chosen, against the candidate).

    const std::string &vt = valid_tgts;

    // Stack-object targets (counterspells, Stifle, Consign to Memory): TargetType$ names one or
    // more kinds of stack object — "Spell[.quals]", "Activated", "Triggered" — as an OR list.
    if (!target_type.empty() &&
        (target_type.find("Spell") != std::string::npos ||
         target_type.find("Activated") != std::string::npos ||
         target_type.find("Triggered") != std::string::npos)) {
        if (!target_type_matches_stack_object(target_type, cand)) return false;
        // A bare "TargetType$ Spell" carries its restriction in ValidTgts$, written against the
        // CARD on the stack — Red Elemental Blast "Card.Blue", Force of Negation
        // "Card.nonCreature", Flusterstorm "Instant,Sorcery" (CR 115.1: the restriction is part
        // of choosing the target, and re-checked at resolution via is_target_valid). Match it
        // through the shared comma-OR card filter. A standalone ability entity on the stack (an
        // Activated/Triggered alternative the script names explicitly — Stifle, Consign to
        // Memory) is not a card, so a card-shaped ValidTgts ("Card,Emblem") never filters it.
        if (vt != "N_A" && !vt.empty() &&
            global_coordinator.entity_has_component<Spell>(cand) &&
            !card_matches_any(cand, vt, MatchCtx{caster, source}))
            return false;
        return true;
    }

    // Card in a non-battlefield zone (e.g. Faerie Macabre targeting graveyard cards)
    if (vt == "Card" && category == "ChangeZone" && origin == Zone::GRAVEYARD) {
        return global_coordinator.entity_has_component<Zone>(cand) &&
               global_coordinator.GetComponent<Zone>(cand).location == Zone::GRAVEYARD;
    }

    bool any = (vt == "Any");
    bool opp_only = (vt == "Opponent");
    bool inc_players = any || opp_only || valid_tgts_names_word(vt, "Player");
    bool inc_creatures = any || vt.find("Creature") != std::string::npos;
    bool inc_lands = vt.find("Land") != std::string::npos;
    bool nonbasic_only = vt.find("nonBasic") != std::string::npos;
    // "Any" is "any target" (creature/player/planeswalker), NOT non-creature
    // artifacts/enchantments — those require the type named explicitly.
    bool inc_artifacts = vt.find("Artifact") != std::string::npos;
    bool inc_enchantments = vt.find("Enchantment") != std::string::npos;
    int cmc_le = -1;
    {
        size_t cmc_pos = vt.find("cmcLE");
        if (cmc_pos != std::string::npos) {
            std::string bound = vt.substr(cmc_pos + 5);
            // Trim to the leading token (stop at the next '.'/'+' qualifier).
            size_t end = bound.find_first_of(".+");
            if (end != std::string::npos) bound = bound.substr(0, end);
            if (!bound.empty() && std::isdigit(static_cast<unsigned char>(bound[0])))
                cmc_le = std::stoi(bound);
            else
                // cmcLEX (Kozilek's Command): the bound is the X paid at cast time.
                cmc_le = static_cast<int>(cur_game.x_paid);
        }
    }

    // Card in a graveyard targeted by a ChangeZone with a type filter. Covers both
    // graveyard→non-battlefield moves (e.g. Life from the Loam: ValidTgts$ Land.YouCtrl,
    // Origin$ Graveyard, Destination$ Hand) and targeted reanimation graveyard→battlefield
    // (e.g. Lorehold Charm's "return target artifact or creature with mana value 2 or less
    // from your graveyard to the battlefield"). In every case the target is the card sitting
    // in the graveyard, so it is matched there regardless of destination. Filter by zone,
    // owner (YouCtrl/OppCtrl/YouOwn), card type, mana value (cmcLE), and Basic supertype.
    if (target_in_graveyard ||
        (category == "ChangeZone" && origin == Zone::GRAVEYARD)) {
        if (!global_coordinator.entity_has_component<Zone>(cand)) return false;
        auto &cz = global_coordinator.GetComponent<Zone>(cand);
        if (cz.location != Zone::GRAVEYARD) return false;
        // For a card in a graveyard, its owner is also its controller, so YouCtrl/OppCtrl
        // and YouOwn/OppOwn restrict the same way (Emry: ValidTgts$ Artifact.YouOwn).
        bool you_ctrl = vt.find("YouCtrl") != std::string::npos || vt.find("YouOwn") != std::string::npos;
        bool opp_ctrl = vt.find("OppCtrl") != std::string::npos || vt.find("OppOwn") != std::string::npos;
        if (you_ctrl && cz.owner != caster) return false;
        if (opp_ctrl && cz.owner == caster) return false;
        if (!global_coordinator.entity_has_component<CardData>(cand)) return false;
        auto &cd = global_coordinator.GetComponent<CardData>(cand);
        // Instant/Sorcery graveyard targets (Mystic Sanctuary: ValidTgts$
        // Instant.YouOwn,Sorcery.YouOwn) filter alongside the permanent types — without
        // these the type gate fell open and offered the whole graveyard.
        bool inc_instants  = vt.find("Instant") != std::string::npos;
        bool inc_sorceries = vt.find("Sorcery") != std::string::npos;
        bool type_ok = !(inc_creatures || inc_lands || inc_artifacts || inc_enchantments ||
                         inc_instants || inc_sorceries);
        for (auto &t : cd.types) {
            if (t.kind != TYPE) continue;
            if (inc_creatures    && t.name == "Creature")    type_ok = true;
            if (inc_lands        && t.name == "Land")        type_ok = true;
            if (inc_artifacts    && t.name == "Artifact")    type_ok = true;
            if (inc_enchantments && t.name == "Enchantment") type_ok = true;
            if (inc_instants     && t.name == "Instant")     type_ok = true;
            if (inc_sorceries    && t.name == "Sorcery")     type_ok = true;
        }
        if (!type_ok) return false;
        // Mana-value bound (e.g. Lorehold Charm's cmcLE2): a graveyard card has no live MV
        // layer, so read its printed mana value.
        if (cmc_le >= 0 && static_cast<int>(cd.mana_cost.size()) > cmc_le) return false;
        if (nonbasic_only && has_basic_supertype(cd.types)) return false;
        return true;
    }

    // Player target
    if (global_coordinator.entity_has_component<Player>(cand)) {
        if (!inc_players) return false;
        if (opp_only) {
            Zone::Ownership opp = (caster == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
            return cand == get_player_entity(opp);
        }
        return true;
    }

    // Battlefield permanent target (phased-out permanents can't be targeted, 702.26e)
    if (!is_battlefield_permanent(cand)) return false;

    // Match the ValidTgts spec against the permanent through the shared filter evaluator
    // (game_queries), per OR-clause. This replaces the old whole-string substring scans
    // (a comma-joined "…YouCtrl,…OppCtrl" set both flags and rejected everything; any text
    // containing "token" was read as token-only). Two normalizations bridge the ValidTgts
    // grammar to the evaluator's: Forge separates OR alternatives with ',' while the evaluator
    // uses ';'; and "YouDontCtrl" is Forge's spelling of OppCtrl (a permanent you don't control)
    // — the evaluator only knows OppCtrl. "Any" as a permanent target is "any creature or
    // planeswalker" (CR 115.4 / 306.7; players are handled in the branch above).
    {
        MatchCtx ctx;
        ctx.controller = caster;
        ctx.source = source;
        // The dynamic mana-value bound (cmcLE<n>/cmcLEX, e.g. Abrupt Decay's cmcLE3) is parsed
        // above into cmc_le; feed it to the evaluator so the bound is actually enforced.
        if (cmc_le >= 0) { ctx.cmc_bound = cmc_le; ctx.cmc_op = "LE"; }
        std::string spec = (vt == "Any") ? std::string("Creature;Planeswalker") : vt;
        std::replace(spec.begin(), spec.end(), ',', ';');
        for (size_t pos = spec.find("YouDontCtrl"); pos != std::string::npos;
             pos = spec.find("YouDontCtrl", pos))
            spec.replace(pos, std::string("YouDontCtrl").size(), "OppCtrl");
        if (!permanent_matches_filter(cand, spec, ctx)) return false;
    }

    // Protection (CR 702.16e): a creature with protection from the source's color/quality can't
    // be targeted by it. The filter evaluator doesn't model protection, so check it separately.
    if (global_coordinator.entity_has_component<Creature>(cand)) {
        const Creature &cand_cr = global_coordinator.GetComponent<Creature>(cand);
        if (has_protection_from(cand_cr, source)) return false;
        // Protection from colored spells (Emrakul: K:Protection:Spell.nonColorless, CR 702.16b/e):
        // a creature with this protection can't be the target of a SPELL that is one or more
        // colors. The "is a spell" half is known here from this Ability's type (a SPELL ability,
        // not an activated/triggered ability), and the "is colored" half from the source's
        // effective colors — so a colorless spell, or any ability, may still target it.
        if (ability_type == Ability::SPELL && has_protection_from_colored_spells(cand_cr) &&
            !effective_colors(source).empty())
            return false;
    }

    // Shroud (CR 702.18e) and Hexproof (CR 702.11b): targeting restrictions read off the
    // candidate's EFFECTIVE keyword set (printed + granted via Pump/effects/keyword counter,
    // through permanent_has_keyword), so a creature granted Shroud (Sylvan Safekeeper) is
    // untargetable while the grant lasts.
    //   • Shroud: can't be the target of ANY spell or ability (yours OR opponents').
    //   • Hexproof: can't be the target of spells/abilities an OPPONENT controls — legal for
    //     the controller of the targeting spell/ability, illegal for the other player. `caster`
    //     is the controller of this ability/spell; the candidate's controller is its Permanent.
    if (permanent_has_keyword(cand, "Shroud")) return false;
    if (permanent_has_keyword(cand, "Hexproof") &&
        global_coordinator.entity_has_component<Permanent>(cand) &&
        global_coordinator.GetComponent<Permanent>(cand).controller != caster)
        return false;
    return true;
}

bool Ability::is_target_valid() const {
    // Optional targeting: no target chosen is valid
    if (target == 0 && targets.empty() && target_min == 0) return true;

    // Multi-target abilities (target_max > 1) populate `targets`. CR 608.2b: the spell or
    // ability is countered on resolution only if ALL of its targets are illegal; with at
    // least one still-legal target it resolves, affecting only the legal ones (resolve()
    // prunes the illegal targets before dispatching to the effect handler).
    if (!targets.empty()) {
        for (Entity t : targets)
            if (is_legal_target(t, controller)) return true;
        return false;
    }

    return is_legal_target(target, controller);
}

// Evaluates a condition SVar expression against cur_game state.
static int evaluate_condition_svar(const std::string &expr, Entity src, Zone::Ownership ctrl = Zone::PLAYER_A,
    std::shared_ptr<Orderer> orderer = nullptr) {
    if (expr == "Count$ResolvedThisTurn") {
        auto it = cur_game.ability_resolution_counts.find(src);
        return (it != cur_game.ability_resolution_counts.end()) ? it->second : 0;
    }
    // Delegate to evaluate_dynamic_amount for Count$ expressions
    if (orderer && expr.find("Count$") != std::string::npos) {
        return static_cast<int>(evaluate_dynamic_amount(expr, ctrl, orderer, 0));
    }
    return 0;
}

// Returns true if val passes the compare spec (e.g. "EQ2", "NE2", "GE1", "LE3").
// When svar_rhs is non-empty, it is evaluated as the RHS instead of parsing an int
// from spec. Unlike svar_eval::compare_svar this is permissive — a missing/unknown
// operator passes — so the operator application (but not the defaulting) is shared
// via apply_svar_op.
static bool compare_svar(int val, const std::string &spec, const std::string &svar_rhs = "", Entity src = 0,
    Zone::Ownership ctrl = Zone::PLAYER_A, std::shared_ptr<Orderer> orderer = nullptr) {
    if (spec.size() < 2) return true;
    std::string op = spec.substr(0, 2);
    int rhs;
    if (!svar_rhs.empty() && orderer) {
        rhs = evaluate_condition_svar(svar_rhs, src, ctrl, orderer);
    } else {
        if (spec.size() < 3) return true;
        rhs = std::stoi(spec.substr(2));
    }
    if (op == "EQ" || op == "NE" || op == "GE" ||
        op == "LE" || op == "GT" || op == "LT") return apply_svar_op(val, op, rhs);
    return true;  // unknown operator passes (permissive)
}

// Evaluates a dynamic_amount_expr at runtime for the given controller.
// Supports: Count$InYourLibrary, Count$YourLifeTotal, Count$YourLifeTotal/HalfUp,
//           Count$Valid Creature.YouCtrl, Targeted$CardPower.
size_t evaluate_dynamic_amount(
    const std::string &expr, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer, Entity target,
    Entity source) {
    // Count$CardCounters.<TYPE> — the number of <TYPE> counters on the ability's SOURCE permanent
    // (The One Ring: X = Count$CardCounters.BURDEN, read by its upkeep life-loss and its draw).
    // The counter type is the substring after the dot, up to any further qualifier delimiter.
    if (expr.rfind("Count$CardCounters.", 0) == 0 && source != 0 &&
        global_coordinator.entity_has_component<Permanent>(source)) {
        std::string ctype = expr.substr(std::string("Count$CardCounters.").size());
        size_t end = ctype.find_first_of(".+ ");
        if (end != std::string::npos) ctype = ctype.substr(0, end);
        const auto &counters = global_coordinator.GetComponent<Permanent>(source).counters;
        auto it = counters.find(ctype);
        return (it != counters.end() && it->second > 0) ? static_cast<size_t>(it->second) : 0;
    }
    if (expr.find("Count$Devotion.") != std::string::npos) {
        // Count mana symbols of a given color in mana costs of permanents you control
        Colors devotion_color = NO_COLOR;
        if (expr.find("Devotion.Blue") != std::string::npos)
            devotion_color = BLUE;
        else if (expr.find("Devotion.Black") != std::string::npos)
            devotion_color = BLACK;
        else if (expr.find("Devotion.Red") != std::string::npos)
            devotion_color = RED;
        else if (expr.find("Devotion.Green") != std::string::npos)
            devotion_color = GREEN;
        else if (expr.find("Devotion.White") != std::string::npos)
            devotion_color = WHITE;
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::BATTLEFIELD) continue;
            auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (perm.controller != ctrl) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            count += cd.mana_cost.count(devotion_color);
        }
        return count;
    }
    // Count$xPaid — the value of X chosen for an X cost when this spell/ability was
    // cast/activated (Kozilek's Command: X = Count$xPaid feeds the token count, scry
    // count and graveyard-exile cap). cur_game.x_paid is recorded at cast time.
    if (expr.find("xPaid") != std::string::npos) {
        return cur_game.x_paid;
    }
    if (expr.find("Count$InYourLibrary") != std::string::npos ||
        expr.find("Count$ValidLibrary Card.YouOwn") != std::string::npos) {
        size_t lib = orderer->get_library_contents(ctrl).size();
        // /HalfUp — half the count, rounded up (Tamiyo, Seasoned Scholar's -7: "draw cards
        // equal to half the number of cards in your library, rounded up"). ceil(N/2).
        if (expr.find("/HalfUp") != std::string::npos) return (lib + 1) / 2;
        return lib;
    }
    if (expr.find("Count$YourLifeTotal") != std::string::npos) {
        Entity ctrl_entity = (ctrl == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
        auto &player = global_coordinator.GetComponent<Player>(ctrl_entity);
        int life = player.life_total;
        if (life < 0) life = 0;
        if (expr.find("/HalfUp") != std::string::npos) {
            return static_cast<size_t>((life + 1) / 2);
        }
        return static_cast<size_t>(life);
    }
    // Count$ValidStack <filter> — number of stack objects matching a card filter (Mindbreak
    // Trap: TargetMax$ MaxTgts, MaxTgts = Count$ValidStack Card — the cap on "exile any number
    // of target spells" is the number of spell cards on the stack). Card-shaped filters are
    // matched through the shared comma-OR card filter against each spell's printed
    // characteristics; standalone ability entities (no CardData) are not cards and don't count.
    // The evaluating ability's own source is excluded — a spell can never target itself, so
    // counting it would only inflate the cap past the real candidate pool.
    if (expr.rfind("Count$ValidStack ", 0) == 0) {
        std::string spec = expr.substr(std::string("Count$ValidStack ").size());
        size_t count = 0;
        for (auto e : orderer->get_stack()) {
            if (e == source) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (!card_matches_any(e, spec, MatchCtx{ctrl, source})) continue;
            count++;
        }
        return count;
    }
    if (expr.find("Count$Valid Creature.YouCtrl") != std::string::npos) {
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Creature>(e)) continue;
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location != Zone::BATTLEFIELD) continue;
            if (!global_coordinator.entity_has_component<Permanent>(e)) continue;
            if (global_coordinator.GetComponent<Permanent>(e).controller != ctrl) continue;
            count++;
        }
        return count;
    }
    // Count$Valid <filter>$CardManaCost — the SUM of mana values of battlefield permanents matching
    // the filter, rather than their count (Summon: Bahamut's Mega Flare: X = Count$Valid
    // Permanent.YouCtrl+Other$CardManaCost = the total mana value of OTHER permanents you control).
    // The "+Other" qualifier excludes the ability's own source (the Bahamut); `source` is threaded
    // into the match context for it. A token / costless permanent contributes mana value 0.
    if (expr.rfind("Count$Valid ", 0) == 0 &&
        expr.find("$CardManaCost") != std::string::npos) {
        std::string rest = expr.substr(std::string("Count$Valid ").size());
        size_t dollar = rest.rfind("$CardManaCost");
        std::string spec = rest.substr(0, dollar);  // the filter, e.g. "Permanent.YouCtrl+Other"
        if (!spec.empty()) {
            MatchCtx mctx;
            mctx.controller = ctrl;  // "you" reference for YouCtrl/OppCtrl
            mctx.source = source;    // for the +Other qualifier (exclude the source)
            size_t total = 0;
            for (auto e : orderer->mEntities) {
                if (!is_battlefield_permanent(e)) continue;
                if (!permanent_matches_filter(e, spec, mctx)) continue;
                if (global_coordinator.entity_has_component<CardData>(e))
                    total += static_cast<size_t>(
                        card_mana_value(global_coordinator.GetComponent<CardData>(e)));
            }
            return total;
        }
    }
    // Count$Valid <filter>.TargetedPlayerCtrl — number of battlefield permanents matching the
    // filter controlled by the PLAYER this ability targets (Carpet of Flowers: Islands the target
    // opponent controls, via the curse-Pump's ValidTgts$ Opponent inherited by the Mana sub).
    // `target` is that Player entity; evaluate the filter from its perspective by rewriting
    // TargetedPlayerCtrl → YouCtrl and counting with the target player's ownership. Handled before
    // the generic Count$Valid branch (which would treat TargetedPlayerCtrl as an unknown qualifier).
    if (expr.rfind("Count$Valid ", 0) == 0 &&
        expr.find("TargetedPlayerCtrl") != std::string::npos) {
        std::string spec = expr.substr(std::string("Count$Valid ").size());
        size_t pos = spec.find("TargetedPlayerCtrl");
        spec.replace(pos, std::string("TargetedPlayerCtrl").size(), "YouCtrl");
        Zone::Ownership tgt_ctrl = Zone::UNKNOWN;
        if (target == cur_game.player_a_entity)      tgt_ctrl = Zone::PLAYER_A;
        else if (target == cur_game.player_b_entity) tgt_ctrl = Zone::PLAYER_B;
        if (tgt_ctrl == Zone::UNKNOWN) return 0;
        return static_cast<size_t>(count_battlefield_matching(spec, tgt_ctrl, source));
    }
    // Count$Valid <Filter> — number of battlefield permanents matching the full Forge filter
    // spec (e.g. Eldrazi Linebreaker: "Count$Valid Eldrazi.YouCtrl"; Eiganjo's Channel
    // ReduceCost: "Count$Valid Creature.Legendary+YouCtrl" = legendary creatures you control).
    // The Creature-specific branch above is kept for its common case; this generic branch
    // routes the whole spec (head type + '.'/'+'-joined qualifiers like Legendary/YouCtrl/
    // colors) through the shared permanent_matches_filter so supertype/color/etc. qualifiers
    // are honored, not just the head type. The RememberedPlayerCtrl form is excluded so it
    // falls through to its dedicated handler below (it needs the remembered-player reference
    // and the /Times multiplier, neither of which permanent_matches_filter understands).
    if (expr.rfind("Count$Valid ", 0) == 0 &&
        expr.find("RememberedPlayerCtrl") == std::string::npos &&
        expr.find("$CardManaCost") == std::string::npos) {
        std::string spec = expr.substr(std::string("Count$Valid ").size());  // full filter spec
        if (!spec.empty())
            return static_cast<size_t>(count_battlefield_matching(spec, ctrl, source));
    }
    // Count$Revolt.high.low — returns high if revolt active for controller, low otherwise
    if (expr.find("Count$Revolt.") != std::string::npos) {
        size_t dot1 = expr.find("Revolt.") + 7;
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        bool revolt = (ctrl == Zone::PLAYER_A) ? cur_game.revolt_player_a : cur_game.revolt_player_b;
        return static_cast<size_t>(revolt ? high_val : low_val);
    }
    // Count$PromisedGift.high.low — Gift (CR 702.176): returns high if the spell currently being
    // cast/resolved promised its gift to an opponent, low otherwise. Into the Flood Maw drives its
    // two ChangeZone abilities' TargetMin$/TargetMax$ off this (X = .0.1, Y = .1.0): not promised →
    // the creature-bounce targets 1 and the nonland-bounce targets 0; promised → the reverse, so
    // the spell instead bounces any nonland permanent. Read from the cast-time pending flag (the
    // target counts are evaluated as targets are chosen, before the Spell component exists).
    if (expr.find("Count$PromisedGift.") != std::string::npos) {
        size_t dot1 = expr.find("PromisedGift.") + std::string("PromisedGift.").size();
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        return static_cast<size_t>(cur_game.pending_gift_promised ? high_val : low_val);
    }
    // Count$Threshold.high.low — Threshold (CR 702.27 historical keyword action; modern cards
    // spell the condition out): returns high if the controller has seven or more cards in their
    // graveyard, low otherwise (Cabal Ritual: Count$Threshold.5.3 → 5 black mana with threshold,
    // else 3). General for any card scaling a dynamic amount by the threshold condition.
    if (expr.find("Count$Threshold.") != std::string::npos) {
        size_t dot1 = expr.find("Threshold.") + std::string("Threshold.").size();
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        bool threshold = orderer->get_graveyard(ctrl).size() >= 7;
        return static_cast<size_t>(threshold ? high_val : low_val);
    }
    // Count$UrzaLands.high.low — the "Tron" mana lands (Urza's Mine/Power Plant/Tower): returns
    // high if the controller controls at least one Urza's Mine AND one Urza's Power-Plant AND one
    // Urza's Tower (a complete set), low otherwise. Per CR 205.3i these are LAND TYPES, so the
    // check reads each permanent's effective type line (Permanent::types — includes types added
    // by continuous effects, e.g. Planar Nexus's AllNonBasicLandType self-CDA), not card names.
    // One permanent with several of the subtypes (Nexus) satisfies each it carries. Each land's
    // own ability scales its colorless output (Mine/Power Plant: .2.1 → {C}{C} assembled / {C}
    // alone; Tower: .3.1 → {C}{C}{C} / {C}). General over any card scaling a dynamic amount by
    // Tron assembly.
    if (expr.find("Count$UrzaLands.") != std::string::npos) {
        size_t dot1 = expr.find("UrzaLands.") + std::string("UrzaLands.").size();
        size_t dot2 = expr.find('.', dot1);
        int high_val = std::stoi(expr.substr(dot1, dot2 - dot1));
        int low_val = std::stoi(expr.substr(dot2 + 1));
        bool mine = false, plant = false, tower = false;
        for (auto e : battlefield_permanents(orderer->mEntities, ctrl)) {
            const auto &perm = global_coordinator.GetComponent<Permanent>(e);
            if (!permanent_has_type(perm, "Urza's")) continue;
            if (permanent_has_type(perm, "Mine")) mine = true;
            if (permanent_has_type(perm, "Power-Plant")) plant = true;
            if (permanent_has_type(perm, "Tower")) tower = true;
        }
        return static_cast<size_t>((mine && plant && tower) ? high_val : low_val);
    }
    if (expr.find("Targeted$CardPower") != std::string::npos) {
        // CR 608.2h: effective power, read live while the creature is in play (counters/buffs
        // included), else its last-known value once it has left (e.g. Swords to Plowshares
        // reads the power of the creature it just exiled). Single unified accessor.
        int p = effective_power(target);
        return static_cast<size_t>(p < 0 ? 0 : p);
    }
    if (expr.find("Targeted$CardManaCost") != std::string::npos) {
        // The target's mana value (CR 202.3 / 107.14). Used by Karn, the Great Creator's +1
        // Animate (Power$/Toughness$ X, X = Targeted$CardManaCost): the animated permanent
        // becomes a creature whose P/T equal its own mana value, snapshotted at resolution.
        int mv = 0;
        if (target != 0 && global_coordinator.entity_has_component<CardData>(target))
            mv = static_cast<int>(global_coordinator.GetComponent<CardData>(target).mana_cost.size());
        return static_cast<size_t>(mv < 0 ? 0 : mv);
    }
    // Remembered$Valid <comma-OR-filter> — number of remembered cards (e.g. cards just moved
    // by a RememberChanged$ ChangeZoneAll) matching ANY of the comma-separated filters (Canoptek
    // Scarab Swarm: X = Remembered$Valid Land,Artifact, "for each artifact or land card exiled
    // this way"). These are now in their destination zone (e.g. exile), so match by printed
    // characteristics via the shared card_matches_filter; control qualifiers resolve against ctrl.
    if (expr.rfind("Remembered$Valid ", 0) == 0) {
        std::string filters = expr.substr(std::string("Remembered$Valid ").size());
        MatchCtx mctx;
        mctx.controller = ctrl;  // "you" reference for YouCtrl/OppCtrl in any filter
        size_t count = 0;
        for (auto e : cur_game.remembered_entities) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (card_matches_any(e, filters, mctx)) count++;  // ',' = OR over the filters
        }
        return count;
    }
    // Remembered$CardManaCost[/Plus.N] — mana value of the first remembered card (Birthing
    // Ritual: X = 1 plus the sacrificed creature's mana value).
    if (expr.find("Remembered$CardManaCost") != std::string::npos) {
        int base = 0;
        if (!cur_game.remembered_entities.empty()) {
            Entity r = cur_game.remembered_entities[0];
            if (global_coordinator.entity_has_component<CardData>(r))
                base = static_cast<int>(global_coordinator.GetComponent<CardData>(r).mana_cost.size());
        }
        size_t plus = expr.find("/Plus.");
        if (plus != std::string::npos) base += std::stoi(expr.substr(plus + 6));
        return static_cast<size_t>(base < 0 ? 0 : base);
    }
    // Count$Valid Land.nonBasic+RememberedPlayerCtrl[/Times.N] — number of nonbasic
    // lands controlled by the remembered player (Price of Progress, evaluated once per
    // player by the RepeatEach loop), optionally multiplied by N. The remembered player
    // is cur_game.remembered_entities[0] (a Player entity set by the repeat_each handler).
    if (expr.find("Count$Valid Land.nonBasic+RememberedPlayerCtrl") != std::string::npos) {
        Zone::Ownership remembered_ctrl = ctrl;
        if (!cur_game.remembered_entities.empty()) {
            Entity rp = cur_game.remembered_entities[0];
            if (rp == cur_game.player_a_entity) remembered_ctrl = Zone::PLAYER_A;
            else if (rp == cur_game.player_b_entity) remembered_ctrl = Zone::PLAYER_B;
        }
        size_t count = 0;
        for (auto e : orderer->mEntities) {
            if (!is_battlefield_permanent(e, remembered_ctrl)) continue;
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            bool is_land = false;
            for (auto &t : cd.types)
                if (t.name == "Land") { is_land = true; break; }
            if (!is_land) continue;
            if (has_basic_supertype(cd.types)) continue;  // nonBasic only
            count++;
        }
        size_t mult = 1;
        size_t times_pos = expr.find("/Times.");
        if (times_pos != std::string::npos)
            mult = static_cast<size_t>(std::stoi(expr.substr(times_pos + 7)));
        return count * mult;
    }
    // Count$ThisTurnCast_Card.<Ctrl>+<Color>[,Card.<Ctrl>+<Color>...] — has a player (relative to
    // ctrl) cast a spell of one of the named colors this turn? (Veil of Summer:
    // Count$ThisTurnCast_Card.OppCtrl+Blue,Card.OppCtrl+Black, the gate for its conditional draw.)
    // Reads Player::spell_colors_cast_this_turn (presence-tracked per color); returns 1 if any
    // requested color was cast by the relevant player this turn, else 0 — sufficient for the GE1
    // conditions that consume it.
    if (expr.find("Count$ThisTurnCast_") != std::string::npos) {
        Zone::Ownership opp = (ctrl == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
        // The clause controller token is read per-expression (Veil uses OppCtrl); YouCtrl (or no
        // controller token) means the source's controller.
        Zone::Ownership who = (expr.find("OppCtrl") != std::string::npos) ? opp : ctrl;
        Entity pe = get_player_entity(who);
        if (global_coordinator.entity_has_component<Player>(pe)) {
            const auto &colors = global_coordinator.GetComponent<Player>(pe).spell_colors_cast_this_turn;
            bool hit = (expr.find("Blue") != std::string::npos && colors.count(BLUE)) ||
                       (expr.find("Black") != std::string::npos && colors.count(BLACK)) ||
                       (expr.find("Red") != std::string::npos && colors.count(RED)) ||
                       (expr.find("Green") != std::string::npos && colors.count(GREEN)) ||
                       (expr.find("White") != std::string::npos && colors.count(WHITE));
            return hit ? 1 : 0;
        }
        return 0;
    }
    // Fall back to the shared static-ability SVar evaluator for graveyard-count
    // expressions (Count$TypeInYourYard / Count$ValidGraveyard / CardTypes). It
    // returns 0 for anything it doesn't recognise, so this preserves the prior
    // default while making one set of Count$ handlers serve both paths.
    int sa_val = evaluate_sa_svar(expr, ctrl);
    return sa_val > 0 ? static_cast<size_t>(sa_val) : 0;
}

// Bind a chained sub-ability's target before it resolves. CR 608.2c: as a spell/ability
// resolves it follows its instructions in order, and each instruction references objects by
// its own definition. The Forge scripts already encode that intent in the sub's Defined$:
//   * The sub targets independently (its own ValidTgts$, a target was chosen for it at cast /
//     activation time, e.g. Cabal Therapy's DB$ Discard ValidTgts$ Player) -> keep that
//     target untouched.
//   * The sub references the PARENT's chosen target: Defined$ {Targeted, ParentTarget, Parent},
//     Defined$ TargetedController (Swords to Plowshares / Solitude GainLife and Smash to
//     Smithereens DealDamage read ab.target to find the targeted permanent's controller / power),
//     or no Defined$ at all (legacy implicit inherit) -> inherit parent.target.
//   * Any other explicit Defined$ that names an INDEPENDENT reference (You / Opponent /
//     Remembered / Self / TriggeredActivator / ...) -> leave sub.target alone; the effect
//     resolves that reference from its own Defined flag and never reads ab.target. Not
//     overwriting here is behavior-preserving (the old blanket sentinel set ab.target too, but
//     those handlers return before touching it).
static void bind_sub_target(const Ability &parent, Ability &sub) {
    if (sub.valid_tgts != "N_A") return;  // independently targeted at cast/activation — keep it
    const std::string &d = sub.defined;
    if (d.empty() || d == "Targeted" || d == "ParentTarget" || d == "Parent" ||
        d == "TargetedController")
        sub.target = parent.target;  // inherit the parent's chosen target (or its controller)
    // else: independent Defined$ reference — leave sub.target alone (effect resolves its own ref)
}

// Detail suffix for the "Resolving ability (category: ...)" log line. Ability::amount is
// only what the effect actually uses for some categories, so a blanket ", amount: N" was
// often a meaningless 0 or an un-evaluated base. Policy: Pump shows its effective +A/+D
// (via the same resolve_pump_amounts the handler uses); categories whose amount IS the
// effect magnitude show the effective amount (the dynamic Count$ expression evaluated the
// same side-effect-free way the handler will); everything else shows nothing. Display-only.
static std::string resolving_log_detail(const Ability &ab, std::shared_ptr<Orderer> orderer) {
    auto signed_str = [](int v) {
        return (v >= 0 ? std::string("+") : std::string()) + std::to_string(v);
    };
    if (ab.category == "Pump" || ab.category == "PumpAll") {
        const PumpParams *pp = std::get_if<PumpParams>(&ab.params);
        if (!pp || (pp->att == 0 && pp->def == 0 && pp->att_expr.empty() && pp->def_expr.empty()))
            return "";  // keyword-grant-only pump — no P/T change to report
        int att = 0, def = 0;
        effects::resolve_pump_amounts(pp, ab.controller, orderer, ab.target, att, def);
        return ", " + signed_str(att) + "/" + signed_str(def);
    }
    // Counter effects keep their count in CounterParams (CounterNum$/its SVar), not
    // Ability::amount — read it the way the handlers do.
    if (ab.category == "PutCounter" || ab.category == "PutCounterAll" ||
        ab.category == "RemoveCounter") {
        const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
        if (!cp) return "";
        int n = cp->count;
        if (!cp->count_expr.empty())
            n = static_cast<int>(
                evaluate_dynamic_amount(cp->count_expr, ab.controller, orderer, ab.target));
        return ", amount: " + std::to_string(n);
    }
    // Discard only counts by Ability::amount in Random mode (Hymn to Tourach); the
    // reveal-and-choose / discard-all modes (Thoughtseize, Cabal Therapy) have no fixed count.
    if (ab.category == "Discard") {
        const DiscardParams *dp = std::get_if<DiscardParams>(&ab.params);
        if (dp && dp->mode == "Random") return ", amount: " + std::to_string(ab.amount);
        return "";
    }
    // Categories where Ability::amount (or its dynamic Count$ expression) is the effect's
    // magnitude, so printing it is informative.
    static const std::set<std::string> kAmountIsAuthoritative = {
        "DealDamage", "DamageAll", "Draw", "Mill", "GainLife", "LoseLife",
        "Scry", "Surveil"};
    if (kAmountIsAuthoritative.count(ab.category)) {
        size_t amt = ab.amount;
        // Draw/Mill treat a 0 amount as the "draw/mill a card" default — mirror it.
        if (amt == 0 && (ab.category == "Draw" || ab.category == "Mill")) amt = 1;
        if (!ab.dynamic_amount_expr.empty())
            amt = evaluate_dynamic_amount(ab.dynamic_amount_expr, ab.controller, orderer,
                                          ab.target, ab.source);
        return ", amount: " + std::to_string(amt);
    }
    return "";
}

// The remembered set (cur_game.remembered_entities) is per-resolution scope in Forge — each
// resolving spell/ability instance has its OWN Remembered list (CR 608.2). This engine backs it
// with one global vector, so a top-level ability's resolution must not inherit remembered objects
// left behind by an EARLIER, unrelated ability that never cleared them (Skyclave Apparition's ETB
// exile pushes its target via RememberChanged and only clears on its own leaves-battlefield
// trigger). Without scoping, the next RememberChanged effect (Phelia's attack exile) appended to
// that stale set and its delayed-return trigger snapshotted BOTH cards, returning Skyclave's
// permanently-exiled target too (CR 603.7c — a delayed trigger references only the objects it was
// set up over). This RAII guard gives each TOP-LEVEL resolve a clean remembered set and restores
// the prior contents on the way out; sub-abilities (resolved recursively within the same call)
// keep sharing the parent's accumulated set, as their chained Remembered$ readers require.
// Destructors run on every return path under -fno-exceptions, so all early-outs are covered.
namespace {
int g_resolve_depth = 0;
struct RememberedResolutionScope {
    bool top_level;
    std::vector<Entity> saved;
    RememberedResolutionScope() : top_level(g_resolve_depth == 0) {
        if (top_level) {
            saved = cur_game.remembered_entities;
            cur_game.remembered_entities.clear();
        }
        ++g_resolve_depth;
    }
    ~RememberedResolutionScope() {
        --g_resolve_depth;
        if (top_level) cur_game.remembered_entities = saved;
    }
};
}  // namespace

void Ability::resolve(std::shared_ptr<Orderer> orderer) {
    RememberedResolutionScope remembered_scope;
    // Leaves-the-battlefield ability whose body references the cards its source had exiled
    // (Skyclave Apparition's TrigToken): restore those entities into the remembered set before
    // any gate or SVar runs, so ConditionPresent$ Card.ExiledWithSource, Remembered$CardManaCost
    // (token P/T), and TokenOwner$ RememberedOwner all read the exiled card (CR 608.2h). The
    // exiled_with snapshot was captured at the source's departure into last-known info.
    if (!restore_remembered_exiled_with.empty()) {
        cur_game.remembered_entities = restore_remembered_exiled_with;
    }

    // OptionalDecider$ You ("you may ..."): the controller may decline the whole
    // triggered ability as it resolves (Ajani's exile-and-return-transformed).
    if (trigger_optional) {
        std::vector<LegalAction> yn;
        LegalAction decline(PASS_PRIORITY, std::string("Decline"));
        decline.category = ActionCategory::OPTIONAL_YESNO;
        yn.push_back(decline);
        LegalAction accept(PASS_PRIORITY, std::string("Accept"));
        accept.category = ActionCategory::OPTIONAL_YESNO;
        yn.push_back(accept);
        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (controller == Zone::PLAYER_A);
        int yc = InputLogger::instance().get_input(yn);
        cur_game.player_a_has_priority = prev_priority;
        if (yc == 0) {
            game_log("%s declines the optional triggered ability.\n", player_name(controller).c_str());
            return;
        }
    }
    // Reflexive "you may sacrifice CARDNAME. If you do, ..." cost on a TRIGGERED ability (The
    // Fantasticar's fourth-noncreature-spell trigger: Execute AB$ Token | Cost$ Sac<1/CARDNAME>).
    // Unlike an activated ability — whose sac cost is paid up front at activation — a triggered
    // ability pays its cost as it resolves (CR 603.2), and the Sac<.../CARDNAME> cost makes the
    // whole effect optional: prompt the controller, sacrifice the source on accept, and do nothing
    // (skip the effect and its subabilities) on decline. Activated abilities never reach here with
    // ability_type == TRIGGERED, so their already-paid sac is not double-charged.
    if (ability_type == TRIGGERED && sac_self) {
        std::string sname = global_coordinator.entity_has_component<Permanent>(source)
                                ? global_coordinator.GetComponent<Permanent>(source).name
                                : std::string("it");
        std::vector<LegalAction> yn;
        LegalAction decline(PASS_PRIORITY, std::string("Decline"));
        decline.category = ActionCategory::OPTIONAL_YESNO;
        yn.push_back(decline);
        LegalAction accept(PASS_PRIORITY, std::string("Sacrifice ") + sname);
        accept.category = ActionCategory::OPTIONAL_YESNO;
        yn.push_back(accept);
        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (controller == Zone::PLAYER_A);
        int yc = InputLogger::instance().get_input(yn);
        cur_game.player_a_has_priority = prev_priority;
        if (yc == 0) {
            game_log("%s declines to sacrifice %s.\n", player_name(controller).c_str(), sname.c_str());
            return;
        }
        orderer->add_to_zone(false, source, Zone::GRAVEYARD);
        game_log("%s sacrifices %s.\n", player_name(controller).c_str(), sname.c_str());
    }
    // 603.4 intervening-if: re-check the trigger's "if" condition on resolution. If it is no
    // longer true the ability is removed from the stack and does nothing — not even its
    // subabilities fire (unlike a ConditionCheckSVar gate).
    if (intervening_if && !evaluate_present_condition(*this, controller, orderer)) {
        game_log("Triggered ability's intervening-if condition is no longer true; it does nothing.\n");
        return;
    }
    // Per-permanent stored-SVar gate (Carpet of Flowers' "if you haven't added mana with this
    // ability this turn", CheckSVar$ CarpetX | SVarCompare$ EQ0). Like the intervening-if it is
    // re-checked at resolution (CR 603.4): if the source permanent's latched scratch int no longer
    // satisfies the comparison, the ability does nothing (CheckPlus sets the latch to 1 only after
    // the mana resolves, so the gate still reads 0 here for a legitimate fire).
    if (!stored_svar_gate_name.empty() &&
        !stored_svar_gate_passes(source, stored_svar_gate_name, stored_svar_gate_compare)) {
        game_log("Triggered ability's stored-SVar gate is no longer satisfied; it does nothing.\n");
        return;
    }
    // Pre-resolve target validity check (CR 608.2b). A Pump that reaches resolution with no
    // pre-chosen target selects its own target inside the handler (an immediate-trigger
    // sub-ability — see effects::pump / effect_immediate_trigger.cpp), so only that case is
    // exempt; a Pump whose target was chosen at cast/trigger placement is verified like any
    // other targeted effect and fizzles if the target became illegal (it does NOT retarget).
    bool pump_selects_own_target = (category == "Pump" && target == 0 && targets.empty());
    if (valid_tgts != "N_A" && !pump_selects_own_target) {
        if (!is_target_valid()) {
            fizzle(orderer);
            return;  // subabilities do not fire; TODO revisit this in light of cards e.g. k-command
        }
        // CR 608.2b: a multi-target spell/ability whose targets are only PARTLY illegal still
        // resolves, affecting only the still-legal targets (a spell that left the stack, a
        // permanent that left the battlefield or gained protection, ...). Prune the illegal
        // ones here so every effect handler downstream sees legal targets only; the all-illegal
        // case was already countered by the is_target_valid gate above.
        if (!targets.empty()) {
            for (auto it = targets.begin(); it != targets.end();) {
                if (!is_legal_target(*it, controller)) {
                    std::string tname = entity_name(*it);
                    game_log("%s is no longer a legal target; it is unaffected\n", tname.c_str());
                    it = targets.erase(it);
                } else {
                    ++it;
                }
            }
            target = targets.empty() ? 0 : targets[0];
        }
    }
    // RememberTargets/RememberObjects: stash the target(s) so chained
    // ChangeType$ Remembered.sameName subabilities can match by name (Surgical Extraction).
    if (remember_targeted) {
        cur_game.remembered_entities.clear();
        if (!targets.empty())
            for (auto t : targets) cur_game.remembered_entities.push_back(t);
        else if (target != 0)
            cur_game.remembered_entities.push_back(target);
    }
    game_log("Resolving ability (category: %s%s)\n", category.c_str(),
             resolving_log_detail(*this, orderer).c_str());

    // Gift (CR 702.176c): if this spell promised its gift, the promised opponent receives the gift
    // BEFORE the spell's other effects. The gift effect(s) are carried on the primary (spell)
    // ability; run them first when Spell::gift_promised is set on the source spell. The token's
    // TokenOwner$ Promised routes it to the opponent of this ability's controller (effects::token).
    if (!gift_abilities.empty() && source != 0 &&
        global_coordinator.entity_has_component<Spell>(source) &&
        global_coordinator.GetComponent<Spell>(source).gift_promised) {
        for (Ability gift : gift_abilities) {
            gift.source = source;
            gift.controller = controller;
            gift.resolve(orderer);
        }
    }

    // Conditional execution: if condition fails, skip this ability's body but still chain subabilities
    bool condition_passed = true;
    if (!condition_check_svar.empty()) {
        int val = evaluate_condition_svar(condition_check_svar, source, controller, orderer);
        condition_passed =
            compare_svar(val, condition_svar_compare, condition_compare_svar_expr, source, controller, orderer);
    }
    // ConditionDefined$ Remembered gate (Birthing Ritual): the dig only happens if a creature
    // was sacrificed (remembered count satisfies condition_present/condition_compare). Like the
    // SVar gate, failure skips this body but still chains subabilities.
    if (condition_passed && condition_on_remembered)
        condition_passed = evaluate_present_condition(*this, controller, orderer);
    // ConditionDefined$ TriggeredCard gate (Amped Raptor): the dig only happens if the card
    // that triggered this ability was cast from its controller's hand. evaluate_present_condition
    // reads the property off the source's permanent state. Failure skips this body but still
    // chains subabilities.
    if (condition_passed && condition_on_triggered_card)
        condition_passed = evaluate_present_condition(*this, controller, orderer);
    // Condition$ Blessing (Ocelot Pride's CopyPermanent): the body runs only if the
    // controller has the city's blessing (702.131). Failure still chains subabilities.
    if (condition_passed && condition_city_blessing) {
        Entity pe = get_player_entity(controller);
        condition_passed = global_coordinator.entity_has_component<Player>(pe) &&
                           global_coordinator.GetComponent<Player>(pe).has_city_blessing;
    }
    if (!condition_passed) {
        for (auto sub_ab : this->subabilities) {
            sub_ab.source = this->source;
            bind_sub_target(*this, sub_ab);  // CR 608.2c — Defined$-driven (see helper)
            sub_ab.controller = this->controller;
            sub_ab.resolve(orderer);
        }
        return;
    }

    // Table-driven dispatch: every effect category resolves through its handler
    // in src/effects/. handler_for() returns nullptr only for categories with no
    // resolve-time handler (e.g. "Equip", handled at activation) — those simply
    // chain subabilities, matching the legacy chain's fall-through behavior.
    effects::EffectHandler handler = effects::handler_for(effect_kind_from_string(category));
    bool run_subs = handler ? handler(*this, orderer) : true;

    // Chain subabilities unless the handler opted out (Charm/WinsGame and the
    // non-peek PeekAndReveal path return false to handle their own resolution).
    if (run_subs) {
        for (auto sub_ab : this->subabilities) {
            sub_ab.source = this->source;
            bind_sub_target(*this, sub_ab);  // CR 608.2c — Defined$-driven (see helper)
            sub_ab.controller = this->controller;
            sub_ab.resolve(orderer);
        }
    }
    // Clear the named card once the whole spell/ability has finished resolving, so it
    // doesn't leak into an unrelated later Card.NamedCard check (CR 201.4 — the name is
    // chosen for this effect only). Only the top-level resolve clears it; sub-abilities
    // (ability_type SPELL parent vs. its DB$ children) are resolved within this call.
    if (effect_kind_from_string(category) == EffectKind::NameCard)
        cur_game.named_card.clear();
}







