#include "zone_search.h"

#include <string>
#include <vector>

#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "components/carddata.h"
#include "ecs/coordinator.h"
#include "input_logger.h"
#include "queries/battlefield.h"
#include "queries/characteristics.h"
#include "queries/filters.h"
#include "queries/players.h"
#include "resolution_frame.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static bool search_candidate_matches(Entity entity, const std::string &change_type, int cmc_bound,
                                     const std::string &cmc_op, Zone::Ownership you,
                                     Entity chain_target);


// Does a zone-search candidate match the search's ChangeType$ (a comma-OR filter spec)? Matched
// through the shared filter matcher in whatever zone the candidate sits: a card by its printed
// characteristics, a battlefield permanent (a multi-zone search's Battlefield origin) by its live
// ones. An empty spec or the catch-all "Card" (a bare "search for a card", Demonic Tutor) matches
// every candidate. `you` is the YouOwn/YouCtrl reference, `chain_target` the targetedBy card, and
// a dynamic mana-value bound (Aether Vial) flows in through cmc_bound / cmc_op.
static bool search_candidate_matches(Entity entity, const std::string &change_type, int cmc_bound,
                                     const std::string &cmc_op, Zone::Ownership you,
                                     Entity chain_target) {
    if (change_type.empty() || change_type == "Card") return true;
    MatchCtx ctx;
    ctx.cmc_bound = cmc_bound;
    ctx.cmc_op = cmc_op;
    ctx.controller = you;
    ctx.chain_target = chain_target;
    return object_matches_filter(entity, change_type, ctx);
}

// Searches a zone for cards whose types match any entry in the comma-separated
// change_type string. Presents all matches plus a "fail to find" option (index 0).
// Returns the chosen Entity, or 0 for fail to find.
// 0 is a valid entity but will always be player a  so is never correct
Entity search_zone(std::shared_ptr<Orderer> orderer, Zone::Ownership owner, Zone::ZoneValue zone,
    const std::string &change_type, bool mandatory, Zone::ZoneValue destination, bool reveal,
    int cmc_bound, const std::string &cmc_op,
    FrameCtx &ctx, Entity decision_source, bool &suspended, Entity chain_target) {
    suspended = false;
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

    std::vector<Entity> choices;
    for (auto entity : zone_contents)
        if (search_candidate_matches(entity, change_type, cmc_bound, cmc_op, owner, chain_target))
            choices.push_back(entity);

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

    // Arm-only log: the resume rebuilds the identical menu (the candidates are
    // covered by the parked menu's determinize pins) without re-logging.
    if (!ctx.resuming()) {
        if (zone == Zone::LIBRARY) {
            game_log("Searching %s's %s:\n", player_name(owner).c_str(), zone_name);
        } else {
            game_log("%s chooses a card from %s %s:\n", player_name(owner).c_str(), player_name(owner).c_str(), zone_name);
        }
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

    // The calling handler has already seated priority on the choosing player
    // (change_zone's search-seat repoint), so asking on the ambient seat is a
    // no-op swap — the exact seat today's inline get_input read from.
    Zone::Ownership chooser = priority_seat();
    int choice = ctx.ask(search_actions, chooser, decision_source);
    if (choice < 0 && decision_suspended()) {
        suspended = true;
        return 0;
    }
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
    Zone::ZoneValue destination, bool reveal,
    FrameCtx &ctx, Entity decision_source, bool &suspended, Entity chain_target) {
    suspended = false;
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
        } else if (zone == Zone::BATTLEFIELD) {
            // Origin$ ...,Battlefield (Cloak and Dagger, Entwined: exile the chosen creature OR
            // a nonland hand card): candidates are the searched player's battlefield permanents.
            auto bf = battlefield_permanents(orderer->mEntities, owner);
            zone_contents.insert(zone_contents.end(), bf.begin(), bf.end());
        }
    }

    // Exclude already-remembered entities (e.g. Doomsday picking 5 cards one at a time)
    if (!cur_game.resolution.memory.remembered.empty()) {
        std::vector<Entity> filtered;
        for (auto e : zone_contents)
            if (!refs_contain(cur_game.resolution.memory.remembered, e)) filtered.push_back(e);
        zone_contents = filtered;
    }

    std::vector<Entity> choices;
    for (auto entity : zone_contents)
        if (search_candidate_matches(entity, change_type, -1, "", owner, chain_target))
            choices.push_back(entity);

    bool show_fail_to_find = !mandatory || choices.empty();
    if (mandatory && choices.empty()) return 0;

    bool searches_library = false;
    std::string zone_list;
    for (auto zone : zones) {
        if (zone == Zone::LIBRARY) searches_library = true;
        const char *zn = (zone == Zone::LIBRARY)      ? "library"
                         : (zone == Zone::GRAVEYARD)   ? "graveyard"
                         : (zone == Zone::HAND)        ? "hand"
                         : (zone == Zone::EXILE)       ? "exile"
                         : (zone == Zone::SIDEBOARD)   ? "sideboard"
                         : (zone == Zone::BATTLEFIELD) ? "battlefield"
                                                       : "zone";
        if (!zone_list.empty()) zone_list += " and ";
        zone_list += zn;
    }
    // Arm-only log (see search_zone above).
    if (!ctx.resuming())
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
        auto &z = global_coordinator.GetComponent<Zone>(entity);
        const char *zone_label = (z.location == Zone::GRAVEYARD)    ? " (graveyard)"
                                 : (z.location == Zone::EXILE)       ? " (exile)"
                                 : (z.location == Zone::SIDEBOARD)   ? " (sideboard)"
                                 : (z.location == Zone::HAND)        ? " (hand)"
                                 : (z.location == Zone::BATTLEFIELD) ? " (battlefield)"
                                                                     : " (library)";
        // entity_name, not CardData: a battlefield candidate may be a token (no CardData).
        LegalAction la(PASS_PRIORITY, entity, entity_name(entity) + zone_label);
        la.category = cat;
        la.card_is_public = reveal;
        search_actions.push_back(la);
    }

    // Seat convention identical to search_zone: the caller already repointed
    // priority at the choosing player.
    Zone::Ownership chooser = priority_seat();
    int choice = ctx.ask(search_actions, chooser, decision_source);
    if (choice < 0 && decision_suspended()) {
        suspended = true;
        return 0;
    }
    if (show_fail_to_find) {
        if (choice >= 1 && choice <= static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice - 1)];
        return 0;
    } else {
        if (choice >= 0 && choice < static_cast<int>(choices.size())) return choices[static_cast<size_t>(choice)];
        return 0;
    }
}
