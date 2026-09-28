#include "zone_search.h"

#include <algorithm>
#include <string>
#include <vector>

#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
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

static std::vector<Entity> zone_contents(const std::shared_ptr<Orderer> &orderer,
                                         Zone::Ownership owner, Zone::ZoneValue zone);
static bool search_candidate_matches(Entity entity, const ZoneSearch &search);
static const char *zone_word(Zone::ZoneValue zone);
static bool searches_zone(const ZoneSearch &search, Zone::ZoneValue zone);
static void log_search(const ZoneSearch &search);
static ActionCategory search_menu_category(const ZoneSearch &search);

// The cards `owner` has in `zone`, in the zone's own order. Graveyard / face-up exile / sideboard
// ("outside the game") cards are enumerated by Zone owner (Karn, the Great Creator -2: choose an
// artifact card you own in exile or your sideboard; sideboard entities are instantiated at game
// start by generate_libraries from the deck's SIDEBOARD: section, plus any test-harness
// --sideboard presets). A Battlefield origin (Cloak and Dagger, Entwined: exile the chosen
// creature OR a nonland hand card) offers the searched player's battlefield permanents.
static std::vector<Entity> zone_contents(const std::shared_ptr<Orderer> &orderer,
                                         Zone::Ownership owner, Zone::ZoneValue zone) {
    if (zone == Zone::LIBRARY) return orderer->get_library_contents(owner);
    if (zone == Zone::HAND) return orderer->get_hand(owner);
    if (zone == Zone::BATTLEFIELD) return battlefield_permanents(orderer->mEntities, owner);
    std::vector<Entity> out;
    if (zone == Zone::GRAVEYARD || zone == Zone::EXILE || zone == Zone::SIDEBOARD) {
        for (auto e : orderer->mEntities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.location == zone && z.owner == owner) out.push_back(e);
        }
    }
    return out;
}

// Does a zone-search candidate match the search's ChangeType$ (a comma-OR filter spec)? Matched
// through the shared filter matcher in whatever zone the candidate sits: a card by its printed
// characteristics, a battlefield permanent (a multi-zone search's Battlefield origin) by its live
// ones. An empty spec or the catch-all "Card" (a bare "search for a card", Demonic Tutor) matches
// every candidate. The owner is the YouOwn/YouCtrl reference, `chain_target` the targetedBy card,
// and a dynamic mana-value bound (Aether Vial) flows in through cmc_bound / cmc_op.
static bool search_candidate_matches(Entity entity, const ZoneSearch &search) {
    if (search.change_type.empty() || search.change_type == "Card") return true;
    MatchCtx ctx;
    ctx.cmc_bound = search.cmc_bound;
    ctx.cmc_op = search.cmc_op;
    ctx.controller = search.owner;
    ctx.chain_target = search.chain_target;
    return object_matches_filter(entity, search.change_type, ctx);
}

static const char *zone_word(Zone::ZoneValue zone) {
    switch (zone) {
    case Zone::LIBRARY:     return "library";
    case Zone::HAND:        return "hand";
    case Zone::GRAVEYARD:   return "graveyard";
    case Zone::EXILE:       return "exile";
    case Zone::SIDEBOARD:   return "sideboard";
    case Zone::BATTLEFIELD: return "battlefield";
    default:                return "zone";
    }
}

static bool searches_zone(const ZoneSearch &search, Zone::ZoneValue zone) {
    return std::find(search.zones.begin(), search.zones.end(), zone) != search.zones.end();
}

// The narrative line opening the search: a library search (or several zones searched together)
// is "Searching <player>'s <zones>", a pick from one other zone "<player> chooses a card from
// <player> <zone>".
static void log_search(const ZoneSearch &search) {
    const std::string who = player_name(search.owner);
    if (search.zones.size() == 1 && search.zones[0] != Zone::LIBRARY) {
        game_log("%s chooses a card from %s %s:\n", who.c_str(), who.c_str(), zone_word(search.zones[0]));
        return;
    }
    std::string zone_list;
    for (auto zone : search.zones) {
        if (!zone_list.empty()) zone_list += " and ";
        zone_list += zone_word(zone);
    }
    game_log("Searching %s's %s:\n", who.c_str(), zone_list.c_str());
}

// Library searches putting the card on top of a library (from the library or a hand) are
// TOP_LIBRARY decisions, other library searches SEARCH_LIBRARY, and a pick from only non-library
// zones (Karn's -2 over Sideboard,Exile) a CHOOSE_CARD decision.
static ActionCategory search_menu_category(const ZoneSearch &search) {
    bool library = searches_zone(search, Zone::LIBRARY);
    if (search.destination == Zone::LIBRARY && (library || searches_zone(search, Zone::HAND)))
        return ActionCategory::TOP_LIBRARY;
    return library ? ActionCategory::SEARCH_LIBRARY : ActionCategory::CHOOSE_CARD;
}

Entity search_zones(std::shared_ptr<Orderer> orderer, const ZoneSearch &search, FrameCtx &ctx,
                    Entity decision_source, bool &suspended) {
    suspended = false;
    const std::vector<ObjectRef> &remembered = cur_game.resolution.memory.remembered;
    std::vector<Entity> choices;
    for (auto zone : search.zones)
        for (auto entity : zone_contents(orderer, search.owner, zone)) {
            if (search.exclude_remembered && refs_contain(remembered, entity)) continue;
            if (search_candidate_matches(entity, search)) choices.push_back(entity);
        }

    // Nothing left to move; return immediately without prompting
    if (search.mandatory && choices.empty()) return 0;
    // Fail-to-find is shown when: not mandatory, OR zone is empty (nothing else to choose)
    bool show_fail_to_find = !search.mandatory || choices.empty();

    // Arm-only log: the resume rebuilds the identical menu (the candidates are
    // covered by the parked menu's determinize pins) without re-logging.
    if (!ctx.resuming()) log_search(search);

    ActionCategory cat = search_menu_category(search);
    const bool label_zone = search.zones.size() > 1;
    std::vector<LegalAction> search_actions;
    if (show_fail_to_find) {
        LegalAction ftf(PASS_PRIORITY, Entity(0), std::string("Fail to find"));
        ftf.category = cat;
        search_actions.push_back(ftf);
    }
    for (auto entity : choices) {
        // entity_name, not CardData: a battlefield candidate may be a token (no CardData).
        std::string label = entity_name(entity);
        if (label_zone)
            label += std::string(" (") +
                     zone_word(global_coordinator.GetComponent<Zone>(entity).location) + ")";
        LegalAction la(PASS_PRIORITY, entity, label);
        la.category = cat;
        la.card_is_public = search.reveal;
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
    int first = show_fail_to_find ? 1 : 0;
    int idx = choice - first;
    if (idx >= 0 && idx < static_cast<int>(choices.size())) return choices[static_cast<size_t>(idx)];
    return 0;
}
