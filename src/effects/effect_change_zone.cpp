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
#include "../components/player.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../classes/action.h"
#include "../input_logger.h"
#include "../game_queries.h"
#include "../mana_system.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static bool search_reveals_card(const Ability &ab);
static Zone::ZoneValue change_zone_move(const std::shared_ptr<Orderer> &orderer, Entity e,
                                        Zone::ZoneValue dest);
static void register_exile_until_host_leaves(Entity host, Entity card, Zone::ZoneValue origin);

// Move a card for a ChangeZone effect and report the zone it actually landed in. A
// replacement effect can divert the move during add_to_zone — Containment Priest redirects an
// uncast creature to exile (614.1a), Grafdigger's Cage prevents the entry so the card stays in
// its origin zone (614.13) — so the result may differ from `dest`. Callers gate their
// battlefield-only bookkeeping (controller / enters-tapped / enters-transformed) and their
// "moved to <dest>" log on the returned zone; when it differs from `dest` the replacement
// dispatcher has already logged the reason for the divert, so no generic line is emitted.
static Zone::ZoneValue change_zone_move(const std::shared_ptr<Orderer> &orderer, Entity e,
                                        Zone::ZoneValue dest) {
    // CR 110.4a / 712.10: only permanents exist on the battlefield. An effect that would put a
    // non-permanent card onto the battlefield can't — the card stays in its current zone. The
    // case that reaches here is a double-faced card returning from exile via a flicker (e.g.
    // Flickerwisp/Phelia on a modal DFC like Fell the Profane // Fell Mire): a DFC returns with
    // its FRONT face up (the card already shows its front face once off the battlefield), so the
    // entity's current CardData decides. If that front face is an instant/sorcery it can't enter,
    // and the card remains exiled. Permanent-faced cards and tokens (always permanents) are
    // unaffected; like a Grafdigger's Cage divert, the caller's battlefield bookkeeping/log is
    // gated on the returned zone, so nothing is logged as having entered.
    if (dest == Zone::BATTLEFIELD &&
        global_coordinator.entity_has_component<CardData>(e) &&
        !is_permanent_card(global_coordinator.GetComponent<CardData>(e))) {
        return global_coordinator.GetComponent<Zone>(e).location;
    }
    orderer->add_to_zone(false, e, dest);
    return global_coordinator.GetComponent<Zone>(e).location;
}

// CR 603.6e linked exile-and-return ("exile ... until [host] leaves the battlefield"). Records
// that `host` exiled `card` from `origin` under a Duration$ UntilHostLeavesPlay, and registers a
// delayed trigger that RETURNS the card when the host leaves the battlefield. The return goes
// back to `origin`: a HAND card to its owner's hand, a BATTLEFIELD permanent onto the battlefield
// under its owner's control (a fresh object — re-entry triggers fire normally). Reused by Cloak
// and Dagger (a revealed hand card or the chosen creature) and by the single-target-permanent
// exilers Sheltered by Ghosts / Static Prison (origin BATTLEFIELD). One trigger is registered per
// exiled card, so each card carries its own origin even when a host exiles several.
static void register_exile_until_host_leaves(Entity host, Entity card, Zone::ZoneValue origin) {
    if (host == 0 || card == 0) return;
    // Track the exiled card on the host's Permanent (snapshotted into last-known info when the
    // host leaves; also the channel Keen-Eyed Curator-style "cards exiled with this" effects read).
    if (global_coordinator.entity_has_component<Permanent>(host))
        global_coordinator.GetComponent<Permanent>(host).exiled_with.push_back(card);

    // The fire ability returns this one card from exile to its origin zone. Seeding
    // restore_remembered_exiled_with makes resolve() set the remembered set to exactly this card,
    // and the Defined$ Remembered move (source = card) sends it to fire_ab.destination under its
    // owner's control (owner = the card's Zone.owner).
    Ability fire_ab;
    fire_ab.ability_type = Ability::TRIGGERED;
    fire_ab.category = "ChangeZone";
    fire_ab.defined_remembered = true;
    fire_ab.restore_remembered_exiled_with = {card};
    fire_ab.source = card;
    fire_ab.origin = Zone::EXILE;
    fire_ab.destination = origin;

    DelayedTrigger dt;
    dt.ability = fire_ab;
    dt.fire_on = Events::CARD_CHANGED_ZONE;
    dt.owner_entity = get_player_entity(source_controller(host));
    dt.fire_on_turn = cur_game.turn;
    dt.watch_entity = host;
    dt.fire_on_leave_battlefield = true;
    cur_game.delayed_triggers.push_back(dt);
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

    // DefinedPlayer$ Targeted with a chosen PLAYER target (Thought-Knot Seer: the sub-ability's
    // target is the opponent the parent RevealHand targeted, bound by bind_sub_target). The
    // searched/owning zone is that targeted player's, so a Hand search picks from THEIR hand. The
    // targeted-move branch below is skipped because this sub-ability carries no target of its own
    // (ValidTgts$ N_A); the bound ab.target is only used to name the searched player here.
    if (ab.defined == "Targeted" && ab.target != 0 &&
        global_coordinator.entity_has_component<Player>(ab.target)) {
        owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
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
            // Pre-move origin zone — captured before the move for a Duration$ UntilHostLeavesPlay
            // exile so the return knows where the card came from (Sheltered by Ghosts / Static
            // Prison target a battlefield permanent: origin BATTLEFIELD).
            Zone::ZoneValue tgt_origin = global_coordinator.GetComponent<Zone>(tgt).location;
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
            // Targeted reanimation (Lorehold Charm: graveyard→battlefield). A permanent
            // entering this way comes under the controller's control (CR 608.2; the spell's
            // controller is the one returning it from their own graveyard). enters_tapped/
            // enters_transformed honour the same flags the search/defined paths use.
            if (ab.destination == Zone::BATTLEFIELD && ab.origin != Zone::BATTLEFIELD) {
                if (ab.enters_tapped) cur_game.pending_enters_tapped.insert(tgt);
                if (ab.enters_transformed) cur_game.pending_enters_transformed.insert(tgt);
            }
            Zone::ZoneValue landed = change_zone_move(orderer, tgt, ab.destination);
            if (landed == Zone::BATTLEFIELD && ab.origin != Zone::BATTLEFIELD)
                global_coordinator.GetComponent<Zone>(tgt).controller = ab.controller;
            if (standalone_ability) {
                global_coordinator.DestroyEntity(tgt);
                game_log("%s is exiled from the stack\n", tname.c_str());
                continue;
            }
            if (ab.destination == Zone::EXILE && ab.source != 0 &&
                global_coordinator.entity_has_component<Permanent>(ab.source)) {
                // Duration$ UntilHostLeavesPlay: register the linked return (which also records
                // exiled_with). Otherwise just record the exile for "cards exiled with this"
                // readers (Skyclave Apparition / Keen-Eyed Curator).
                if (ab.duration_until_host_leaves)
                    register_exile_until_host_leaves(ab.source, tgt, tgt_origin);
                else
                    global_coordinator.GetComponent<Permanent>(ab.source).exiled_with.push_back(tgt);
            }
            // RememberChanged$ True (Skyclave Apparition's TrigExile): stash the moved card in
            // the remembered set so a later SVar (Remembered$CardManaCost) and a paired
            // leaves-the-battlefield ability (TrigToken sizing/owning the Illusion) can read it.
            if (ab.remember_changed || ab.remember_lki) cur_game.remembered_entities.push_back(tgt);
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
        if (ab.enters_tapped && ab.destination == Zone::BATTLEFIELD)
            cur_game.pending_enters_tapped.insert(ab.source);
        Zone::ZoneValue landed = change_zone_move(orderer, ab.source, ab.destination);
        if (landed == Zone::BATTLEFIELD)
            global_coordinator.GetComponent<Zone>(ab.source).controller = owner;
        if (landed == ab.destination)
            game_log("%s is moved to %s\n", sname.c_str(), dest_str);
        return true;
    }

    // Defined$ Remembered with a bounded/optional selection (Cloak and Dagger's DBChangeZone:
    // "you MAY exile up to one of the remembered candidates"). Unlike the blanket defined_remembered
    // move below (Ajani, which moves EVERY remembered object), this PRESENTS A CHOICE of which
    // remembered object(s) to move: ChangeNum$ N caps the count and Optional$ True permits
    // declining. Candidates are the remembered objects currently in an eligible Origin zone (Hand
    // or Battlefield); nonland-only, honoring the oracle "exile a nonland card from their hand or
    // the chosen creature" (the creature is itself nonland). Only reached with an explicit
    // count/optionality, so the Ajani blanket path is unaffected.
    if (ab.defined_remembered && (ab.optional_choice || ab.change_num >= 0)) {
        int cap = (ab.change_num >= 0) ? ab.change_num
                                       : (ab.amount > 0 ? static_cast<int>(ab.amount) : 1);
        bool optional = ab.optional_choice;
        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
        for (int picked = 0; picked < cap; picked++) {
            // Rebuild the candidate list each pick (a card already moved leaves the eligible zones).
            std::vector<Entity> cands;
            for (auto e : cur_game.remembered_entities) {
                if (!global_coordinator.entity_has_component<Zone>(e)) continue;
                if (!global_coordinator.entity_has_component<CardData>(e)) continue;
                Zone::ZoneValue loc = global_coordinator.GetComponent<Zone>(e).location;
                bool zone_ok = false;
                for (auto z : ab.origins) if (z == loc) zone_ok = true;
                if (!zone_ok) continue;
                if (is_land_card(global_coordinator.GetComponent<CardData>(e))) continue;  // nonland (oracle)
                cands.push_back(e);
            }
            if (cands.empty()) break;
            std::vector<LegalAction> picks;
            for (auto e : cands) {
                auto &cd = global_coordinator.GetComponent<CardData>(e);
                Zone::ZoneValue loc = global_coordinator.GetComponent<Zone>(e).location;
                const char *where = (loc == Zone::BATTLEFIELD) ? " (the chosen creature)" : " (from hand)";
                LegalAction la(PASS_PRIORITY, e, std::string("Exile ") + cd.name + where);
                la.category = ActionCategory::CHOOSE_CARD;
                la.card_is_public = true;
                picks.push_back(la);
            }
            if (optional) {
                LegalAction none(PASS_PRIORITY, std::string("Exile nothing"));
                none.category = ActionCategory::CHOOSE_CARD;
                picks.push_back(none);
            }
            game_log("%s may exile a card until %s leaves the battlefield:\n",
                player_name(ab.controller).c_str(),
                global_coordinator.entity_has_component<CardData>(ab.source)
                    ? global_coordinator.GetComponent<CardData>(ab.source).name.c_str() : "the source");
            int choice = InputLogger::instance().get_input(picks);
            if (choice < 0 || choice >= static_cast<int>(cands.size())) break;  // declined
            Entity chosen = cands[static_cast<size_t>(choice)];
            Zone::ZoneValue card_origin = global_coordinator.GetComponent<Zone>(chosen).location;
            Zone::Ownership card_owner = global_coordinator.GetComponent<Zone>(chosen).owner;
            std::string cname = global_coordinator.GetComponent<CardData>(chosen).name;
            change_zone_move(orderer, chosen, ab.destination);
            mark_card_revealed(chosen, card_owner);
            game_log("%s exiles %s\n", player_name(ab.controller).c_str(), cname.c_str());
            if (ab.duration_until_host_leaves)
                register_exile_until_host_leaves(ab.source, chosen, card_origin);
        }
        cur_game.player_a_has_priority = prev_priority;
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

    // Search-based ChangeZone (e.g. fetch lands, Green Sun's Zenith). The number to move is
    // normally a fixed numeric ChangeNum$. A dynamic count-SVar ChangeNum (Ugin, Eye of the
    // Storms' -11: "Search ... for any number of colorless nonland cards", ChangeNum$ X, X =
    // Count of those cards you own) means "any number up to all": resolve the cap from the
    // count expression and let a fail-to-find stop the search early (CR 701.19, a "search for
    // any number" lets the player choose fewer). dynamic_amount_expr is only set when ChangeNum$
    // itself is a count-SVar, so fixed-count fetches (and Green Sun's Zenith, whose X bounds the
    // filter, not the count) are unaffected.
    size_t num_to_move;
    if (!ab.dynamic_amount_expr.empty())
        num_to_move = evaluate_dynamic_amount(ab.dynamic_amount_expr, owner, orderer, 0);
    else
        num_to_move = (ab.amount > 0) ? ab.amount : 1;
    bool multi_zone = ab.origins.size() > 1;
    bool reveal = search_reveals_card(ab);

    // Chooser$ You — the ability's controller makes the selection from `owner`'s zone (Thought-Knot
    // Seer: you pick a nonland card from the targeted opponent's revealed hand to exile). Switch
    // priority to the controller so the choice prompt is offered to them, not the searched player,
    // and restore it after. The searched cards are public knowledge here (the hand was revealed by
    // the parent RevealHand), so the picks carry card_is_public — flag reveal so the chosen card's
    // identity is shown even into a hidden destination and recorded in the belief state.
    bool prev_priority = cur_game.player_a_has_priority;
    if (ab.chooser_is_controller) {
        cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
        reveal = true;
    }

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
    if (ab.chooser_is_controller) cur_game.player_a_has_priority = prev_priority;
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
            if (s == "Sideboard")      return Zone::SIDEBOARD;
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
