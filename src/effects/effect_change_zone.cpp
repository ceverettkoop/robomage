#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static bool search_reveals_card(const Ability &ab);
static Zone::ZoneValue change_zone_move(const std::shared_ptr<Orderer> &orderer, Entity e,
                                        Zone::ZoneValue dest);

// Move a card for a ChangeZone effect and report the zone it actually landed in. A
// replacement effect can divert the move during add_to_zone — Containment Priest redirects an
// uncast creature to exile (614.1a), Grafdigger's Cage prevents the entry so the card stays in
// its origin zone (614.13) — so the result may differ from `dest`. Callers gate their
// battlefield-only bookkeeping (controller / enters-tapped / enters-transformed) and their
// "moved to <dest>" log on the returned zone; when it differs from `dest` the replacement
// dispatcher has already logged the reason for the divert, so no generic line is emitted.
static Zone::ZoneValue change_zone_move(const std::shared_ptr<Orderer> &orderer, Entity e,
                                        Zone::ZoneValue dest) {
    orderer->add_to_zone(false, e, dest);
    return global_coordinator.GetComponent<Zone>(e).location;
}

// A library search reveals the chosen card when it must satisfy a restriction
// more specific than "any card" — the searcher proves the card qualifies (e.g.
// Personal Tutor: "search for a sorcery card, reveal it"). An unrestricted
// "Card" search (Vampiric/Demonic Tutor, Doomsday) reveals nothing. Searches of
// other zones are public already, so this only matters for the library.
static bool search_reveals_card(const Ability &ab) {
    bool from_library = (ab.origin == Zone::LIBRARY);
    for (auto z : ab.origins) {
        if (z == Zone::LIBRARY) from_library = true;
    }
    bool specific_type = !ab.change_type.empty() && ab.change_type != "Card";
    return from_library && specific_type;
}

bool change_zone(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Same-name search/move (Surgical Extraction, Infernal Tutor, Secret Salvage, Pack
    // Hunt, ...): ChangeType$ Remembered.sameName / Targeted.sameName.
    if (ab.change_type.find("sameName") != std::string::npos)
        return change_zone_same_name(ab, orderer, /*force_all=*/false);

    Zone::Ownership owner = global_coordinator.GetComponent<Zone>(ab.source).owner;

    // DefinedPlayer$ TargetedController (Erode: "Its controller may search their library
    // for a basic land card, put it onto the battlefield tapped, then shuffle."): the
    // searching/owning player for a search-based ChangeZone is the targeted card's
    // controller, not the spell's caster (CR 109.5). The target may already have left the
    // battlefield (Erode's Destroy sub-ability ran first), but Zone.controller persists
    // through the move to the graveyard, so it still names the last controller. This only
    // redirects the search path; the targeted-move branch below is skipped because such a
    // sub-ability carries no target of its own (ValidTgts$ N_A).
    if (ab.defined_targeted_controller && ab.target != 0) {
        Zone::Ownership tc = last_known_controller(ab.target);  // CR 608.2g/h, robust to post-move reads
        if (tc != Zone::UNKNOWN) owner = tc;
    }

    const char *dest_str = ab.destination == Zone::BATTLEFIELD ? "the battlefield"
                           : ab.destination == Zone::LIBRARY   ? "top of library"
                           : ab.destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.destination == Zone::HAND      ? "hand"
                                                               : "exile";

    // Targeted ChangeZone (e.g. Swords to Plowshares, Faerie Macabre, Life from the
    // Loam): move target(s) directly. A targeted ability with no chosen targets (e.g.
    // "up to three" with zero picked) does nothing rather than searching.
    if (ab.valid_tgts != "N_A") {
        std::vector<Entity> to_move;
        if (!ab.targets.empty()) {
            to_move = ab.targets;
        } else if (ab.target != 0) {
            to_move.push_back(ab.target);
        }
        for (auto tgt : to_move) {
            if (!global_coordinator.entity_has_component<Zone>(tgt)) continue;
            std::string tname = global_coordinator.entity_has_component<CardData>(tgt)
                                    ? global_coordinator.GetComponent<CardData>(tgt).name
                                    : (global_coordinator.entity_has_component<Permanent>(tgt)
                                              ? global_coordinator.GetComponent<Permanent>(tgt).name
                                              : "<unknown>");
            // A target that is a spell/ability on the stack (Mindbreak Trap: "exile any
            // number of target spells") is being removed from the stack. Strip its Spell/
            // Ability components — like effects::counter does — so the stack no longer
            // treats it as a live object to resolve (CR 701.5a / 702.59c). A standalone
            // ability entity (no card / CardData) is destroyed after leaving the stack.
            bool on_stack = global_coordinator.GetComponent<Zone>(tgt).location == Zone::STACK;
            bool standalone_ability = on_stack &&
                                      !global_coordinator.entity_has_component<CardData>(tgt) &&
                                      global_coordinator.entity_has_component<Ability>(tgt);
            if (on_stack) {
                if (global_coordinator.entity_has_component<Spell>(tgt))
                    global_coordinator.RemoveComponent<Spell>(tgt);
                if (global_coordinator.entity_has_component<Ability>(tgt))
                    global_coordinator.RemoveComponent<Ability>(tgt);
            }
            Zone::ZoneValue landed = change_zone_move(orderer, tgt, ab.destination);
            if (standalone_ability) {
                global_coordinator.DestroyEntity(tgt);
                game_log("%s is exiled from the stack\n", tname.c_str());
                continue;
            }
            if (ab.destination == Zone::EXILE && ab.source != 0 &&
                global_coordinator.entity_has_component<Permanent>(ab.source)) {
                global_coordinator.GetComponent<Permanent>(ab.source).exiled_with.push_back(tgt);
            }
            if (landed == ab.destination)
                game_log("%s is moved to %s\n", tname.c_str(), dest_str);
        }
        return true;
    }

    // Defined$ Self — move the source card directly (e.g. Talon Gates putting itself onto battlefield from hand)
    if (ab.defined_self && ab.source != 0) {
        std::string sname = global_coordinator.entity_has_component<CardData>(ab.source)
                                ? global_coordinator.GetComponent<CardData>(ab.source).name
                                : "<unknown>";
        Zone::ZoneValue landed = change_zone_move(orderer, ab.source, ab.destination);
        if (landed == Zone::BATTLEFIELD)
            global_coordinator.GetComponent<Zone>(ab.source).controller = owner;
        if (landed == ab.destination)
            game_log("%s is moved to %s\n", sname.c_str(), dest_str);
        return true;
    }

    // Defined$ Remembered — move the remembered card(s) directly. Ajani's exile-and-
    // return chain uses this: TrigExile remembers the exiled permanent, DBReturn brings
    // that same card back to the battlefield (transformed) under its owner's control.
    if (ab.defined_remembered) {
        for (auto e : cur_game.remembered_entities) {
            if (!global_coordinator.entity_has_component<Zone>(e)) continue;
            std::string nm = global_coordinator.entity_has_component<CardData>(e)
                                 ? global_coordinator.GetComponent<CardData>(e).name
                                 : "<unknown>";
            Zone::ZoneValue landed = change_zone_move(orderer, e, ab.destination);
            if (landed == Zone::BATTLEFIELD) {
                global_coordinator.GetComponent<Zone>(e).controller = owner;
                if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(e);
                if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(e);
            }
            if (landed == ab.destination)
                game_log("%s is moved to %s\n", nm.c_str(), dest_str);
        }
        return true;
    }

    // A ChangeZone leaving the battlefield with no target and no search filter operates
    // on the ability's own source (Forge's Defined$ Self default; e.g. Ajani's TrigExile
    // exiles himself). search_zone can't enumerate the battlefield, so a self-move is the
    // only sensible reading of a battlefield origin here.
    if (ab.origin == Zone::BATTLEFIELD && ab.change_type.empty() && ab.source != 0 &&
        global_coordinator.entity_has_component<Zone>(ab.source)) {
        std::string nm = global_coordinator.entity_has_component<CardData>(ab.source)
                             ? global_coordinator.GetComponent<CardData>(ab.source).name
                             : "<unknown>";
        Zone::ZoneValue landed = change_zone_move(orderer, ab.source, ab.destination);
        if (ab.remember_changed) cur_game.remembered_entities.push_back(ab.source);
        if (landed == Zone::BATTLEFIELD) {
            global_coordinator.GetComponent<Zone>(ab.source).controller = owner;
            if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(ab.source);
        }
        if (landed == ab.destination)
            game_log("%s is moved to %s\n", nm.c_str(), dest_str);
        return true;
    }

    // Search-based ChangeZone (e.g. fetch lands, Green Sun's Zenith)
    size_t num_to_move = (ab.amount > 0) ? ab.amount : 1;
    bool multi_zone = ab.origins.size() > 1;
    bool reveal = search_reveals_card(ab);

    // Dynamic mana-value bound on the search filter (Aether Vial: "Creature.cmcEQX",
    // X = charge counters on this Aether Vial). Resolve against the ability's source so
    // the hand search only offers creatures of the matching mana value (CR 122.1).
    int cmc_bound = -1;
    if (!ab.change_type_cmc_expr.empty())
        cmc_bound = evaluate_sa_svar(ab.change_type_cmc_expr, owner, ab.source);

    for (size_t i = 0; i < num_to_move; i++) {
        Entity chosen = 0;
        if (multi_zone) {
            chosen = search_multi_zone(orderer, owner, ab.origins, ab.change_type, ab.mandatory, ab.destination,
                reveal);
        } else {
            chosen = search_zone(orderer, owner, ab.origin, ab.change_type, ab.mandatory, ab.destination,
                reveal, cmc_bound, ab.change_type_cmc_op);
        }

        // after we have chosen but before we place it where it goes, if we messed with library shuffle it
        if (ab.origin == Zone::LIBRARY) {
            orderer->shuffle_library(owner);
            game_log("%s shuffles their library\n", player_name(owner).c_str());
        }

        if (chosen != 0) {
            auto &chosen_cd = global_coordinator.GetComponent<CardData>(chosen);
            auto &chosen_zone = global_coordinator.GetComponent<Zone>(chosen);
            Zone::ZoneValue landed = change_zone_move(orderer, chosen, ab.destination);
            if (landed == Zone::BATTLEFIELD) {
                chosen_zone.controller = owner;
                if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(chosen);
                if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(chosen);
            }
            if (ab.remember_changed) {
                cur_game.remembered_entities.push_back(chosen);
            }
            bool dest_public =
                (ab.destination == Zone::BATTLEFIELD || ab.destination == Zone::GRAVEYARD || ab.destination == Zone::EXILE);
            if (landed != ab.destination) {
                // A replacement effect diverted the move (Containment Priest → exile,
                // Grafdigger's Cage → prevented) and already logged its reason; emit nothing.
            } else if (dest_public) {
                game_log("%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
            } else if (reveal) {
                // Hidden destination, but the card was revealed — it's public knowledge.
                // The orderer reveal hook only fires for public zones, so mark it here.
                mark_card_revealed(chosen, owner);
                game_log("%s reveals %s and puts it to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(),
                    dest_str);
            } else {
                game_log_private(
                    owner, "%s puts %s to %s\n", player_name(owner).c_str(), chosen_cd.name.c_str(), dest_str);
                game_log_redacted(owner, "%s puts a card to %s\n", player_name(owner).c_str(), dest_str);
            }
        } else {
            game_log("%s fails to find\n", player_name(owner).c_str());
            break;
        }
    }
    return true;
}

// Owns the zone-movement param keys shared by the ChangeZone family (ChangeZone
// and ChangeZoneAll both consume origin/destination/etc. at resolve).
bool parse_change_zone(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "ChangeType") {
        ab.change_type = value;
        return true;
    } else if (key == "RememberChanged") {
        ab.remember_changed = (value == "True");
        return true;
    } else if (key == "Origin") {
        // Handle comma-separated origins (e.g. "Graveyard,Library")
        auto parse_zone = [](const std::string &s) -> Zone::ZoneValue {
            if (s == "Library")        return Zone::LIBRARY;
            if (s == "Hand")           return Zone::HAND;
            if (s == "Graveyard")      return Zone::GRAVEYARD;
            if (s == "Exile")          return Zone::EXILE;
            if (s == "Stack")          return Zone::STACK;
            if (s == "Battlefield")    return Zone::BATTLEFIELD;
            return Zone::LIBRARY;
        };
        ab.origins.clear();
        size_t zp = 0;
        while (true) {
            size_t comma = value.find(',', zp);
            if (comma == std::string::npos) {
                ab.origins.push_back(parse_zone(value.substr(zp)));
                break;
            }
            ab.origins.push_back(parse_zone(value.substr(zp, comma - zp)));
            zp = comma + 1;
        }
        ab.origin = ab.origins[0];  // backward compat
        return true;
    } else if (key == "Destination") {
        if (value == "Battlefield")    ab.destination = Zone::BATTLEFIELD;
        else if (value == "Library")   ab.destination = Zone::LIBRARY;
        else if (value == "Hand")      ab.destination = Zone::HAND;
        else if (value == "Graveyard") ab.destination = Zone::GRAVEYARD;
        else if (value == "Exile")     ab.destination = Zone::EXILE;
        return true;
    } else if (key == "MayShuffle") {
        ab.may_shuffle = (value == "True");
        return true;
    } else if (key == "Tapped") {
        ab.enters_tapped = (value == "True");
        return true;
    } else if (key == "Transformed") {
        ab.enters_transformed = (value == "True");
        return true;
    }
    return false;
}

// Shared handler for ChangeType$ Remembered.sameName / Targeted.sameName (Surgical
// Extraction, Extirpate, Infernal Tutor, Secret Salvage, Pack Hunt, ...). Builds the set
// of cards sharing a name with the referenced card across the ability's Origin zone(s),
// belonging to the caster (default) or the targeted card's owner (DefinedPlayer$ /
// Defined$ TargetedController), and moves up to the allowed number to the Destination.
//
// Quantity follows the agreed simplification: an "up to N" / count-SVar / ChangeZoneAll
// quantity moves the maximum available (force_all, an empty amount_svar with a numeric
// cap moves min(cap, found)). Searching an opponent's hand/library reveals those zones,
// recorded into the match-scoped belief state (mark_card_revealed) — the only
// opponent-card-identity channel in the observation.
bool change_zone_same_name(Ability &ab, std::shared_ptr<Orderer> orderer, bool force_all) {
    // The reference card whose name we match: the target (Targeted.sameName) or the
    // remembered object (Remembered.sameName, set by RememberTargets/RememberObjects).
    Entity ref = 0;
    if (ab.change_type.find("Targeted") != std::string::npos)
        ref = !ab.targets.empty() ? ab.targets[0] : ab.target;
    else
        ref = !cur_game.remembered_entities.empty() ? cur_game.remembered_entities[0]
              : (!ab.targets.empty() ? ab.targets[0] : ab.target);
    if (ref == 0 || !global_coordinator.entity_has_component<CardData>(ref)) return true;
    std::string name = global_coordinator.GetComponent<CardData>(ref).name;

    // Whose zones to search: the caster by default, or the owner of the referenced card
    // (DefinedPlayer$/Defined$ TargetedController). Derive from `ref` (the remembered/
    // targeted card) rather than ab.target, which a Pump vehicle may have overwritten.
    Zone::Ownership caster = global_coordinator.GetComponent<Zone>(ab.source).owner;
    Zone::Ownership searched = caster;
    if (ab.defined_targeted_controller && global_coordinator.entity_has_component<Zone>(ref))
        searched = global_coordinator.GetComponent<Zone>(ref).owner;

    std::vector<Zone::ZoneValue> zones =
        ab.origins.size() > 1 ? ab.origins : std::vector<Zone::ZoneValue>{ ab.origin };

    bool searched_library = false;
    std::vector<Entity> matches;
    auto collect = [&](const std::vector<Entity> &cards) {
        for (auto e : cards) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            if (global_coordinator.GetComponent<CardData>(e).name == name) matches.push_back(e);
        }
    };
    for (auto z : zones) {
        if (z == Zone::LIBRARY)        { searched_library = true; collect(orderer->get_library_contents(searched)); }
        else if (z == Zone::HAND)      collect(orderer->get_hand(searched));
        else if (z == Zone::GRAVEYARD) collect(orderer->get_graveyard(searched));
    }

    // Cap: ChangeZoneAll / count-SVar / unset → all matches; numeric ChangeNum → min(N, found).
    size_t cap = matches.size();
    if (!force_all && ab.amount_svar.empty() && ab.amount > 0)
        cap = std::min(static_cast<size_t>(ab.amount), matches.size());

    const char *dest_str = ab.destination == Zone::EXILE       ? "exile"
                           : ab.destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.destination == Zone::HAND      ? "hand"
                           : ab.destination == Zone::BATTLEFIELD ? "the battlefield"
                                                                 : "library";
    for (size_t i = 0; i < cap; i++) {
        Zone::ZoneValue landed = change_zone_move(orderer, matches[i], ab.destination);
        if (landed == Zone::BATTLEFIELD)
            global_coordinator.GetComponent<Zone>(matches[i]).controller = caster;
        mark_card_revealed(matches[i], searched);  // moved card is now public / revealed
    }
    game_log("%s moves %zu card(s) named %s from %s's %s to %s\n", player_name(caster).c_str(),
        cap, name.c_str(), player_name(searched).c_str(),
        zones.size() > 1 ? "zones" : "zone", dest_str);

    // Searching an opponent's hidden zones reveals their full contents to the searcher;
    // record them into the belief state.
    if (searched != caster) {
        for (auto z : zones) {
            if (z == Zone::HAND)    for (auto e : orderer->get_hand(searched))             mark_card_revealed(e, searched);
            if (z == Zone::LIBRARY) for (auto e : orderer->get_library_contents(searched)) mark_card_revealed(e, searched);
        }
    }

    if (searched_library) {
        orderer->shuffle_library(searched);
        game_log("%s shuffles their library\n", player_name(searched).c_str());
    }
    return true;
}

}  // namespace effects
