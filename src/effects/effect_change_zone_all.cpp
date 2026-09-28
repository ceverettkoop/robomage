#include "effects.h"

#include <random>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../stable_rng.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Echoing Truth family: return the chosen TARGET nonland permanent AND every OTHER battlefield
// permanent sharing its name to their owners' hands (ChangeType$ TargetedCard.Self,Permanent...
// +sharesNameWith Targeted; Origin$ Battlefield, Destination$ Hand). The target itself is included
// because it is a battlefield permanent whose name trivially matches. Ownership is fixed (CR
// 108.3 / 400.3), so a HAND destination routes each permanent to ITS OWNER's hand automatically —
// this correctly handles copies split across both players. A unique-named target returns alone.
// Names are the permanents' current ones (Permanent::name): a transformed Insectile Aberration is
// named Aberration, not Delver of Secrets (CR 712.8e); a token carries its token name.
static bool change_zone_shares_name_battlefield(Ability &ab, std::shared_ptr<Orderer> orderer) {
    Entity ref = !ab.targets.empty() ? ab.targets[0].get() : ab.target.get();
    if (ref == 0 || !is_battlefield_permanent(ref)) return true;
    const std::string name = global_coordinator.GetComponent<Permanent>(ref).name;
    if (name.empty()) return true;

    std::vector<Entity> to_move;
    for (auto e : battlefield_permanents(orderer->mEntities))
        if (global_coordinator.GetComponent<Permanent>(e).name == name) to_move.push_back(e);
    for (auto e : to_move) {
        Zone::Ownership owner = global_coordinator.GetComponent<Zone>(e).owner;
        std::string ename = global_coordinator.GetComponent<Permanent>(e).name;
        orderer->add_to_zone(false, e, ab.destination);
        game_log("%s returns to %s's hand\n", ename.c_str(), player_name(owner).c_str());
    }
    return true;
}

HandlerResult change_zone_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Same-name move-all (e.g. Extirpate's ExileYard): ChangeType$ Remembered.sameName.
    if (ab.change_type.find("sameName") != std::string::npos)
        return change_zone_same_name(ab, orderer, /*force_all=*/true)
                   ? HandlerResult::DONE_RUN_SUBS
                   : HandlerResult::DONE_NO_SUBS;

    // Echoing Truth family: ChangeType$ ...+sharesNameWith Targeted over the battlefield — return
    // the target and all other permanents with its name to their owners' hands.
    if (ab.change_type.find("sharesNameWith") != std::string::npos && ab.origin == Zone::BATTLEFIELD)
        return change_zone_shares_name_battlefield(ab, orderer)
                   ? HandlerResult::DONE_RUN_SUBS
                   : HandlerResult::DONE_NO_SUBS;

    // A targeted ability resolved with no target chosen (e.g. Endurance's "up to
    // one target player", TargetMin$ 0) affects no one — do nothing rather than
    // falling back to the controller's own zones. Untargeted ChangeZoneAll
    // (valid_tgts "N_A", e.g. Doomsday) still operates on the controller below.
    if (ab.valid_tgts != "N_A" && ab.target.empty() && ab.targets.empty())
        return HandlerResult::DONE_RUN_SUBS;

    Zone::Ownership owner = ab.controller;
    // If this targets a player (e.g. Endurance: "target player puts the cards
    // from their graveyard on the bottom of their library"), operate on the
    // targeted player's zones rather than the controller's.
    const Entity tgt_player = ab.target.get();
    if (tgt_player != 0 && global_coordinator.entity_has_component<Player>(tgt_player)) {
        owner = seat_of_player(tgt_player);
    }
    // Defined$ TriggeredCardOwner (Emrakul's death trigger: "its owner shuffles their graveyard
    // into their library"): operate on the OWNER of the card that triggered this ability (the
    // source), not the last controller. Read the owner off the source's Zone (it is in the
    // graveyard by now). CR 608.2g/400.3: ownership is fixed regardless of who controlled it.
    const Entity trig_card = ab.source.lki_entity();  // ownership never changes (CR 108.3)
    if (ab.defined == "TriggeredCardOwner" && trig_card != 0 &&
        global_coordinator.entity_has_component<Zone>(trig_card)) {
        Zone::Ownership src_owner = global_coordinator.GetComponent<Zone>(trig_card).owner;
        if (src_owner != Zone::UNKNOWN) owner = src_owner;
    }

    // Determine which zones to search
    std::vector<Zone::ZoneValue> search_zones;
    if (ab.origin_any) {
        // Origin$ All/Any: every owned-card zone this collector reads.
        search_zones = {Zone::LIBRARY, Zone::HAND, Zone::GRAVEYARD, Zone::EXILE};
    } else if (ab.origins.size() > 1) {
        search_zones = ab.origins;
    } else {
        search_zones.push_back(ab.origin);
    }

    // Collect all cards from the specified zones
    std::vector<Entity> zone_contents;
    for (auto zone : search_zones) {
        if (zone == Zone::LIBRARY) {
            auto lib = orderer->get_library_contents(owner);
            zone_contents.insert(zone_contents.end(), lib.begin(), lib.end());
        } else if (zone == Zone::HAND) {
            auto hand = orderer->get_hand(owner);
            zone_contents.insert(zone_contents.end(), hand.begin(), hand.end());
        } else if (zone == Zone::GRAVEYARD) {
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                auto &z = global_coordinator.GetComponent<Zone>(e);
                if (z.location == Zone::GRAVEYARD && z.owner == owner) zone_contents.push_back(e);
            }
        } else if (zone == Zone::EXILE) {
            // Origin$ Exile — the owner's exiled cards (Triumph of Saint Katherine's recursion:
            // move the self-exiled pile back onto the library). Usually paired with a
            // Card.IsRemembered filter so only the intended pile (not every exiled card) moves.
            for (auto e : orderer->mEntities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                auto &z = global_coordinator.GetComponent<Zone>(e);
                if (z.location == Zone::EXILE && z.owner == owner) zone_contents.push_back(e);
            }
        }
    }

    // Filter by ChangeType$ through the shared filter matcher: Doomsday's Card.!IsRemembered
    // (exile everything but the cards just stacked), Atraxa's Card.IsImprinted+!IsRemembered
    // (bottom the revealed cards not taken), Triumph of Saint Katherine's Card.IsRemembered (only
    // the remembered pile). Absent a ChangeType$, every card in the zones moves.
    MatchCtx mctx;
    mctx.controller = owner;
    mctx.source = ab.source.lki_entity();
    std::vector<Entity> to_move;
    for (auto entity : zone_contents)
        if (ab.change_type.empty() || card_matches_filter(entity, ab.change_type, mctx))
            to_move.push_back(entity);

    // RandomOrder$ — randomize the moved cards with the seeded RNG (deterministic
    // per game seed, platform-stable — see stable_rng.h). Used for "in a random
    // order" library placement (Endurance).
    if (ab.rest_random_order) {
        stable_shuffle(to_move, cur_game.gen);
    }

    // Library destinations honor LibraryPosition$ (-1 / unset = bottom).
    bool on_bottom = (ab.destination == Zone::LIBRARY && ab.dig_library_position != 0);
    const char *dest_str = ab.destination == Zone::EXILE       ? "exile"
                           : ab.destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.destination == Zone::HAND      ? "hand"
                           : ab.destination == Zone::BATTLEFIELD ? "the battlefield"
                           : ab.destination == Zone::LIBRARY   ? (on_bottom ? "the bottom of their library"
                                                                            : "the top of their library")
                                                               : "zone";

    // Cards put into a library in a random order land in positions no player knows (Triumph of
    // Saint Katherine's shuffled pile), so the known-top record keeps them unknown.
    const LibraryTopView top_view =
        ab.rest_random_order ? LibraryTopView::NOBODY : LibraryTopView::OWNER;
    size_t moved = 0;
    for (auto entity : to_move) {
        // The shared uncast battlefield entry first: an Aura picks what it enchants, or stays
        // where it is without a legal object (CR 303.4f/g), and then isn't moved.
        if (ab.destination == Zone::BATTLEFIELD &&
            put_onto_battlefield(orderer, FrameCtx::blocking(), entity) != Zone::BATTLEFIELD)
            continue;
        if (global_coordinator.entity_has_component<CardData>(entity)) {
            // A face-down exiled card's identity is hidden from every player (CR 406.3).
            if (global_coordinator.GetComponent<Zone>(entity).is_face_down) {
                game_log("%s moves a face-down card to %s\n", player_name(owner).c_str(), dest_str);
            } else {
                auto &cd = global_coordinator.GetComponent<CardData>(entity);
                game_log("%s moves %s to %s\n", player_name(owner).c_str(), cd.name.c_str(),
                         dest_str);
            }
        }
        if (ab.destination != Zone::BATTLEFIELD)
            orderer->add_to_zone(on_bottom, entity, ab.destination, top_view);
        // RememberChanged$ True: stash every moved card in the remembered set, mirroring the
        // single-target ChangeZone path (effect_change_zone.cpp). A later SVar can then count
        // these cards (Canoptek Scarab Swarm: X = Remembered$Valid Land,Artifact, "for each
        // artifact or land card exiled this way"); cleared by the paired DBCleanup ClearRemembered$.
        if (ab.remember_changed) cur_game.remembered_entities.push_back(entity);
        moved++;
    }
    game_log("%s moves %zu card(s) to %s\n", player_name(owner).c_str(), moved, dest_str);

    // Shuffle$ True (Emrakul's death trigger: "shuffle their graveyard into their library"): after
    // moving the cards into the library, shuffle it. shuffle_library also clears the known-top-of-
    // library tracking for that player (CR 701.20).
    if (ab.shuffle_after && ab.destination == Zone::LIBRARY) {
        orderer->shuffle_library(owner);
        game_log("%s shuffles their library.\n", player_name(owner).c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
