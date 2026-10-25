#include "effects.h"

#include <algorithm>
#include <memory>
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
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/filters.h"
#include "../queries/players.h"
#include "../stable_rng.h"
#include "../state_format/events.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static bool change_zone_shares_name_battlefield(Ability &ab, std::shared_ptr<Orderer> orderer);
static size_t move_all_cards(Ability &ab, std::shared_ptr<Orderer> orderer, const std::vector<Entity> &cards,
                             Zone::Ownership owner);

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
        orderer->add_to_zone(false, e, ab.def->destination);
        game_log("%s returns to %s's hand\n", ename.c_str(), player_name(owner).c_str());
    }
    return true;
}

HandlerResult change_zone_all(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Same-name move-all (e.g. Extirpate's ExileYard): ChangeType$ Remembered.sameName.
    if (ab.def->change_type.find("sameName") != std::string::npos)
        return change_zone_same_name(ab, orderer, /*force_all=*/true)
                   ? HandlerResult::DONE_RUN_SUBS
                   : HandlerResult::DONE_NO_SUBS;

    // Echoing Truth family: ChangeType$ ...+sharesNameWith Targeted over the battlefield — return
    // the target and all other permanents with its name to their owners' hands.
    if (ab.def->change_type.find("sharesNameWith") != std::string::npos && ab.def->origin == Zone::BATTLEFIELD)
        return change_zone_shares_name_battlefield(ab, orderer)
                   ? HandlerResult::DONE_RUN_SUBS
                   : HandlerResult::DONE_NO_SUBS;

    // A targeted ability resolved with no target chosen (e.g. Endurance's "up to
    // one target player", TargetMin$ 0) affects no one — do nothing rather than
    // falling back to the controller's own zones. Untargeted ChangeZoneAll
    // (valid_tgts "N_A", e.g. Doomsday) still operates on the controller below.
    if (ab.def->valid_tgts != "N_A" && ab.target.empty() && ab.targets.empty())
        return HandlerResult::DONE_RUN_SUBS;

    Zone::Ownership owner = ab.controller;
    bool player_defined = false;  // the effect acts on one player's zones (targeted or defined)
    // If this targets a player (e.g. Endurance: "target player puts the cards
    // from their graveyard on the bottom of their library"), operate on the
    // targeted player's zones rather than the controller's.
    const Entity tgt_player = ab.target.get();
    if (tgt_player != 0 && global_coordinator.entity_has_component<Player>(tgt_player)) {
        owner = seat_of_player(tgt_player);
        player_defined = true;
    }
    // Defined$ TriggeredCardOwner (Emrakul's death trigger: "its owner shuffles their graveyard
    // into their library"): operate on the OWNER of the card that triggered this ability (the
    // source), not the last controller. Read the owner off the source's Zone (it is in the
    // graveyard by now). CR 608.2g/400.3: ownership is fixed regardless of who controlled it.
    const Entity trig_card = ab.source.lki_entity();  // ownership never changes (CR 108.3)
    if (ab.def->defined == "TriggeredCardOwner" && trig_card != 0 &&
        global_coordinator.entity_has_component<Zone>(trig_card)) {
        Zone::Ownership src_owner = global_coordinator.GetComponent<Zone>(trig_card).owner;
        if (src_owner != Zone::UNKNOWN) owner = src_owner;
        player_defined = true;
    }

    // Determine which zones to search
    std::vector<Zone::ZoneValue> search_zones;
    if (ab.def->origin_any) {
        // Origin$ All/Any: every owned-card zone this collector reads.
        search_zones = {Zone::LIBRARY, Zone::HAND, Zone::GRAVEYARD, Zone::EXILE};
    } else if (ab.def->origins.size() > 1) {
        search_zones = ab.def->origins;
    } else {
        search_zones.push_back(ab.def->origin);
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
        } else if (zone == Zone::BATTLEFIELD) {
            // Origin$ Battlefield (Terminus: "put all creatures on the bottom of their owners'
            // libraries"): every permanent on the battlefield, or the defined player's alone.
            for (auto e : battlefield_permanents(orderer->mEntities))
                if (!player_defined || global_coordinator.GetComponent<Zone>(e).controller == owner)
                    zone_contents.push_back(e);
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
    // the remembered pile). A permanent is matched by its characteristics on the battlefield
    // (Terminus' Creature: tokens and animated permanents too). Absent a ChangeType$, every card in
    // the zones moves.
    MatchCtx mctx;
    mctx.controller = owner;
    mctx.source = ab.source.lki_entity();
    std::vector<Entity> to_move;
    for (auto entity : zone_contents)
        if (ab.def->change_type.empty() || object_matches_filter(entity, ab.def->change_type, mctx))
            to_move.push_back(entity);

    // RandomOrder$ — randomize the moved cards with the seeded RNG (deterministic
    // per game seed, platform-stable — see stable_rng.h). Used for "in a random
    // order" library placement (Endurance).
    if (ab.def->rest_random_order) {
        stable_shuffle(to_move, cur_game.rng.engine);
    }

    // A permanent goes to its owner's zone (CR 400.3), so the cards moved from the battlefield
    // are moved owner by owner, each owner's cards into their library as one batch.
    if (std::find(search_zones.begin(), search_zones.end(), Zone::BATTLEFIELD) != search_zones.end()) {
        for (Zone::Ownership o : {Zone::PLAYER_A, Zone::PLAYER_B}) {
            std::vector<Entity> owned;
            for (auto e : to_move)
                if (global_coordinator.GetComponent<Zone>(e).owner == o) owned.push_back(e);
            if (!owned.empty()) move_all_cards(ab, orderer, owned, o);
        }
    } else {
        move_all_cards(ab, orderer, to_move, owner);
    }

    // Shuffle$ True (Emrakul's death trigger: "shuffle their graveyard into their library"): after
    // moving the cards into the library, shuffle it.
    if (ab.def->shuffle_after && ab.def->destination == Zone::LIBRARY) {
        orderer->shuffle_library(owner);
        game_log("%s shuffles their library.\n", player_name(owner).c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

// Moves `cards` (owned by `owner`) to the ability's destination, logging each and the count:
// cards put into `owner`'s library land as one batch whose order only `owner` chose (or, in a
// random order, nobody saw). Returns the number moved.
static size_t move_all_cards(Ability &ab, std::shared_ptr<Orderer> orderer, const std::vector<Entity> &cards,
                             Zone::Ownership owner) {
    // Library destinations honor LibraryPosition$ (-1 / unset = bottom).
    bool on_bottom = (ab.def->destination == Zone::LIBRARY && ab.def->dig_library_position != 0);
    const char *dest_str = ab.def->destination == Zone::EXILE       ? "exile"
                           : ab.def->destination == Zone::GRAVEYARD ? "graveyard"
                           : ab.def->destination == Zone::HAND      ? "hand"
                           : ab.def->destination == Zone::BATTLEFIELD ? "the battlefield"
                           : ab.def->destination == Zone::LIBRARY   ? (on_bottom ? "the bottom of their library"
                                                                            : "the top of their library")
                                                               : "zone";

    // Cards put into a library together land in an order only their owner chose (or, in a random
    // order, nobody saw). A token is no card: it goes first, outside that batch, and ceases to exist
    // there (CR 111.7, 704.5d).
    std::vector<Entity> ordered;
    for (auto entity : cards)
        if (global_coordinator.entity_has_component<Token>(entity)) ordered.push_back(entity);
    const size_t tokens = ordered.size();
    for (auto entity : cards)
        if (!global_coordinator.entity_has_component<Token>(entity)) ordered.push_back(entity);
    std::unique_ptr<ObsLibraryBatch> batch;
    size_t moved = 0, visited = 0;
    for (auto entity : ordered) {
        if (!batch && visited++ >= tokens)
            batch = std::make_unique<ObsLibraryBatch>(owner, ab.def->rest_random_order ? Zone::UNKNOWN : owner);
        const bool from_battlefield = global_coordinator.GetComponent<Zone>(entity).location == Zone::BATTLEFIELD;
        // The shared uncast battlefield entry first: an Aura picks what it enchants, or stays
        // where it is without a legal object (CR 303.4f/g), and then isn't moved.
        if (ab.def->destination == Zone::BATTLEFIELD &&
            put_onto_battlefield(orderer, FrameCtx::blocking(), entity) != Zone::BATTLEFIELD)
            continue;
        if (from_battlefield) {
            // A permanent (a token too) by its name on the battlefield.
            game_log("%s moves %s to %s\n", player_name(owner).c_str(), entity_name(entity).c_str(), dest_str);
        } else if (global_coordinator.entity_has_component<CardData>(entity)) {
            // A face-down exiled card's identity is hidden from every player (CR 406.3).
            if (global_coordinator.GetComponent<Zone>(entity).is_face_down) {
                game_log("%s moves a face-down card to %s\n", player_name(owner).c_str(), dest_str);
            } else {
                auto &cd = global_coordinator.GetComponent<CardData>(entity);
                game_log("%s moves %s to %s\n", player_name(owner).c_str(), cd.name.c_str(),
                         dest_str);
            }
        }
        if (ab.def->destination != Zone::BATTLEFIELD)
            orderer->add_to_zone(on_bottom, entity, ab.def->destination);
        // RememberChanged$ True: stash every moved card in the remembered set, mirroring the
        // single-target ChangeZone path (effect_change_zone.cpp). A later SVar can then count
        // these cards (Canoptek Scarab Swarm: X = Remembered$Valid Land,Artifact, "for each
        // artifact or land card exiled this way"); cleared by the paired DBCleanup ClearRemembered$.
        if (ab.def->remember_changed) cur_game.resolution.memory.remembered.push_back(ObjectRef::of(entity));
        moved++;
    }
    if (batch) batch->close();
    game_log("%s moves %zu card(s) to %s\n", player_name(owner).c_str(), moved, dest_str);
    return moved;
}

}  // namespace effects
