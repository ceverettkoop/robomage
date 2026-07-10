#include "orderer.h"

#include <algorithm>
#include <numeric>

#include "../stable_rng.h"

#include "../card_db.h"
#include "../card_vocab.h"
#include "../classes/deck.h"
#include "../classes/game.h"
#include "../classes/action.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../game_queries.h"
#include "../components/player.h"
#include "../components/effect.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../input_logger.h"
#include "replacement_effects.h"
#include "../machine_io.h"
#include "../type_constants.h"
#include "../components/types.h"

// --- file-local helpers (forward declarations) ---
static ColorIdentity color_identity_from(const CardData &cd);

// orderer cares about anything that has a zone
void Orderer::init() {
    Signature signature;
    signature.set(global_coordinator.GetComponentType<Zone>());
    global_coordinator.SetSystemSignature<Orderer>(signature);
}

Entity Orderer::push_ability_onto_stack(const Ability &ability, Zone::Ownership controller) {
    // Initialize the entity's Zone as HAND so add_to_zone's origin-zone removal is a
    // harmless no-op, then move it onto the stack and attach the (already-populated) ability.
    Entity ability_entity = global_coordinator.CreateEntity();
    Zone ab_zone(Zone::HAND, controller, controller);
    global_coordinator.AddComponent(ability_entity, ab_zone);
    add_to_zone(false, ability_entity, Zone::STACK);
    global_coordinator.AddComponent(ability_entity, ability);
    return ability_entity;
}

void Orderer::place_created_on_stack(Entity target, Zone::Ownership controller) {
    // The object comes into existence on the stack — no origin zone, so no MOVE_TO_ZONE
    // replacement (614) and no CARD_CHANGED_ZONE event (which would model a transition the
    // copy never made). Just register it as the new top and shift the existing stack down.
    Zone z(Zone::STACK, controller, controller);
    z.distance_from_top = 0;
    z.identity_known = false;
    for (auto &&card : mEntities) {
        if (card == target) continue;
        auto &cmp_zone = global_coordinator.GetComponent<Zone>(card);
        if (cmp_zone.location == Zone::STACK) cmp_zone.distance_from_top++;
    }
    global_coordinator.AddComponent(target, z);
    // A stack object is public information (CR 400.2): record it in the owner's revealed set,
    // the same chokepoint add_to_zone uses when a card enters a public zone.
    mark_card_revealed(target, controller);
}

void Orderer::add_to_zone(bool on_bottom, Entity target, Zone::ZoneValue destination,
                          bool top_seen_by_owner) {
    size_t back = 0;
    auto &target_zone = global_coordinator.GetComponent<Zone>(target);

    // Replacement effects (rule 614): redirect graveyard → exile when Dauthi Voidwalker etc.
    // apply, or prevent a creature card from entering the battlefield out of a graveyard/library
    // (Grafdigger's Cage, 614.13) — in which case the move doesn't happen and the card stays put.
    {
        ReplacementEvent rev;
        rev.type = ReplacementEvent::MOVE_TO_ZONE;
        rev.entity = target;
        rev.affected_player = target_zone.owner;  // 616.1: the card's owner is the affected player
        rev.origin = target_zone.location;
        rev.destination = destination;
        replacement::dispatch(rev);
        if (rev.prevented) return;  // 614.13 — the move is prevented; the card remains in its origin zone
        destination = rev.destination;
    }

    // Unearth (CR 702.84): a permanent returned to the battlefield with its unearth ability is
    // exiled instead of going anywhere else if it would leave the battlefield. Redirect any move
    // off the battlefield (to graveyard/hand/library) to exile. A move already headed to exile is
    // unchanged (so the end-step delayed exile is a no-op redirect, not a loop).
    if (target_zone.location == Zone::BATTLEFIELD && destination != Zone::EXILE &&
        global_coordinator.entity_has_component<Permanent>(target) &&
        global_coordinator.GetComponent<Permanent>(target).unearthed) {
        destination = Zone::EXILE;
    }

    // Fire CARD_CHANGED_ZONE on every zone transition so any parsed ChangesZone trigger can match.
    {
        Entity owner_entity = target_zone.owner == Zone::PLAYER_A
                              ? cur_game.player_a_entity : cur_game.player_b_entity;
        Event ev(Events::CARD_CHANGED_ZONE);
        ev.SetParam(Params::ENTITY,      target);
        ev.SetParam(Params::PLAYER,      owner_entity);
        ev.SetParam(Params::ORIGIN,      target_zone.location);
        ev.SetParam(Params::DESTINATION, destination);
        global_coordinator.SendEvent(ev);
    }

    // Track revolt: a permanent leaving the battlefield sets revolt for its controller
    if (target_zone.location == Zone::BATTLEFIELD &&
        global_coordinator.entity_has_component<Permanent>(target)) {
        Zone::Ownership ctrl = global_coordinator.GetComponent<Permanent>(target).controller;
        if (ctrl == Zone::PLAYER_A) cur_game.revolt_player_a = true;
        else                        cur_game.revolt_player_b = true;

        // 603.10 look-back: snapshot the permanent's type/subtype names as it leaves the
        // battlefield so a "dies"/leaves-the-battlefield trigger can still match it after a
        // token has ceased to exist (and after CardData/Permanent are stripped). Consumed
        // and cleared by check_triggered_abilities.
        std::vector<std::string> &names = cur_game.lk_battlefield_types[target];
        names.clear();
        for (const auto &t : global_coordinator.GetComponent<Permanent>(target).types)
            names.push_back(t.name);

        // Full last-known-information snapshot (CR 608.2h / 112.7a): the permanent's effective
        // characteristics as it last existed in play, so the effective_* accessors can answer a
        // post-departure read (e.g. Swords to Plowshares' "gains life equal to its power" for the
        // creature it just exiled) with its in-play values. Captured here while the Creature/
        // CardData components are still intact (they are stripped later, by the SBA pass).
        LastKnownInfo &lki = cur_game.last_known_info[target];
        lki = LastKnownInfo{};
        lki.type_names = names;
        lki.controller = global_coordinator.GetComponent<Permanent>(target).controller;
        // Snapshot the cards this permanent had exiled (CR 608.2h last-known info): a
        // leaves-the-battlefield ability that creates a token sized/owned by an exiled card
        // (Skyclave Apparition) still needs them after the Permanent component is stripped.
        lki.exiled_with = global_coordinator.GetComponent<Permanent>(target).exiled_with;
        // How-it-entered markers: an ETB trigger of a permanent that leaves again before trigger
        // collection (legend rule, 0-toughness SBA) is fired by the look-back scan in
        // check_triggered_abilities, which needs these gates after Permanent is stripped.
        {
            const auto &p = global_coordinator.GetComponent<Permanent>(target);
            lki.entered_by_cast = p.entered_by_cast;
            lki.evoked = p.evoked;
            lki.entered_with_offspring = p.entered_with_offspring;
            lki.transformed = p.transformed;
            // Layer-6 ability removal (Humility, CR 613.1f): a permanent whose abilities were
            // removed has no triggered abilities, so its own leaves/dies look-back trigger
            // (CR 603.10) must not fire — capture the flag before Permanent is stripped.
            lki.abilities_removed = p.abilities_removed;
            lki.cast_from_hand_by_controller = p.cast_from_hand_by_controller;
        }
        if (global_coordinator.entity_has_component<Creature>(target)) {
            auto &cr = global_coordinator.GetComponent<Creature>(target);
            lki.power = static_cast<int>(cr.power);
            lki.toughness = static_cast<int>(cr.toughness);
        }
        if (global_coordinator.entity_has_component<CardData>(target))
            lki.colors = card_colors(global_coordinator.GetComponent<CardData>(target));
    }

    // If the entity is leaving an ordered zone, close the gap it leaves behind.
    // LIBRARY, STACK, and GRAVEYARD are ordered zones where distance_from_top is meaningful.
    Zone::ZoneValue origin = target_zone.location;
    if (origin == Zone::LIBRARY || origin == Zone::STACK || origin == Zone::GRAVEYARD) {
        size_t departing_pos = target_zone.distance_from_top;
        Zone::Ownership owner = target_zone.owner;
        for (auto &&card : mEntities) {
            if (card == target) continue;
            auto &cmp_zone = global_coordinator.GetComponent<Zone>(card);
            if (cmp_zone.location != origin) continue;
            // Library and graveyard are per-player; stack is shared
            if ((origin == Zone::LIBRARY || origin == Zone::GRAVEYARD) && cmp_zone.owner != owner) continue;
            if (cmp_zone.distance_from_top > departing_pos) {
                cmp_zone.distance_from_top--;
            }
        }

        // If a card left the library within the tracked top window, drop it
        // from the known-top cache and shift the rest up.
        if (origin == Zone::LIBRARY && departing_pos < static_cast<size_t>(KNOWN_TOP_LIBRARY_SIZE)) {
            cur_game.known_top_library_remove_pos(owner == Zone::PLAYER_A, static_cast<int>(departing_pos));
        }
    }

    if (!on_bottom) {
        target_zone.distance_from_top = 0;
    }

    // If a card is being placed on top of a library, it becomes the new known top.
    // The push is unconditional even for a fateseal (top_seen_by_owner == false): the owner's
    // previously-known top entries must still shift one position deeper. But when the owner does
    // NOT see the card (the looker is the opponent), record an UNKNOWN marker (-1) instead of the
    // real identity, so the cache positions stay honest without leaking a card they never saw.
    if (!on_bottom && destination == Zone::LIBRARY) {
        int vocab_idx = -1;
        if (top_seen_by_owner && global_coordinator.entity_has_component<CardData>(target)) {
            vocab_idx = card_name_to_index(
                global_coordinator.GetComponent<CardData>(target).name);
        }
        cur_game.known_top_library_push(target_zone.owner == Zone::PLAYER_A, vocab_idx);
    }

    for (auto &&card : mEntities) {
        if (card == target) continue;
        auto &cmp_zone = global_coordinator.GetComponent<Zone>(card);
        if (cmp_zone.location != destination) continue;
        // Library and graveyard are per-player; only shift cards belonging to the same owner
        if ((destination == Zone::LIBRARY || destination == Zone::GRAVEYARD) && cmp_zone.owner != target_zone.owner)
            continue;
        if (!on_bottom) {
            // placing on top: shift everything else down one
            cmp_zone.distance_from_top++;
        } else {
            // placing on bottom: find the current bottom position
            if (cmp_zone.distance_from_top > back) back = cmp_zone.distance_from_top;
        }
    }

    if (on_bottom) target_zone.distance_from_top = back + 1;
    target_zone.location = destination;

    // CR 400.7: a card returning to the battlefield is a NEW object. Its previous battlefield
    // components are normally gone already — stripped by the state-based pass while it was away —
    // but a card that leaves AND returns within a single resolution (same-resolution flicker,
    // Ajani's exile-and-return transform) re-enters before that pass could run and still carries
    // its stale Permanent/Creature/Damage (tapped state, summoning sickness, counters,
    // attachments, damage). Strip them here, at entry, AFTER the departure-side LKI snapshot
    // above captured last-known info; the next state-based pass then rebuilds the permanent fresh
    // exactly like any other entry — including the ENTERS_BATTLEFIELD replacement dispatch
    // (enters tapped / with counters) and the pending_enters_tapped/_transformed one-shots.
    // (Phasing never passes through add_to_zone; a phased-out permanent keeps its state, 702.26.)
    if (destination == Zone::BATTLEFIELD && origin != Zone::BATTLEFIELD &&
        global_coordinator.entity_has_component<Permanent>(target)) {
        strip_permanent_components(target);
    }

    // A zone change re-derives visibility from the new zone: any prior "identity
    // known to the opponent while hidden in hand" belief no longer applies (the
    // card is now either public, or a fresh hidden object). Reveal sites set this
    // flag again if the destination is a revealed hidden zone.
    target_zone.identity_known = false;

    // Match-scoped reveal tracking: any card entering a PUBLIC zone becomes known
    // to both players, so accumulate it in the owner's revealed multi-hot. This
    // single chokepoint covers casts (→STACK), ETB (→BATTLEFIELD), and
    // deaths/discards (→GRAVEYARD/EXILE). Moves to HAND or within LIBRARY are
    // hidden and intentionally skipped (tutor reveals are marked at their site).
    if (destination == Zone::BATTLEFIELD || destination == Zone::STACK ||
        destination == Zone::GRAVEYARD || destination == Zone::EXILE) {
        mark_card_revealed(target, target_zone.owner);
    }
}

// TODO MERGE THESE INTO A GENERIC GETTER
std::vector<Entity> Orderer::get_library_contents(Zone::Ownership owner) {
    std::vector<Entity> contents;

    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if ((card_zone.location == Zone::LIBRARY) && (card_zone.owner == owner)) {
            contents.push_back(card);
        }
    }
    return contents;
}

std::vector<Entity> Orderer::get_library_top(Zone::Ownership owner, size_t n) {
    std::vector<Entity> lib = get_library_contents(owner);
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    if (lib.size() > n) lib.resize(n);
    return lib;
}

std::vector<Entity> Orderer::get_hand(Zone::Ownership owner) {
    std::vector<Entity> contents;

    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if ((card_zone.location == Zone::HAND) && (card_zone.owner == owner)) {
            contents.push_back(card);
        }
    }
    return contents;
}

void Orderer::shuffle_library(Zone::Ownership owner) {
    auto contents = get_library_contents(owner);
    size_t n = contents.size();
    std::vector<int> placements(n);
    std::iota(placements.begin(), placements.end(), 0);
    // stable_shuffle, not std::shuffle: std::shuffle output differs between
    // libstdc++ and libc++ for the same seed (see stable_rng.h).
    stable_shuffle(placements, cur_game.gen);

    size_t i = 0;
    for (auto &&card : contents) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        card_zone.distance_from_top = placements[i];
        i++;
    }

    // Shuffling destroys any knowledge of which cards are on top of the library
    cur_game.clear_known_top_library(owner == Zone::PLAYER_A);
}

extern bool no_shuffle;

// Derive a card's color identity from its mana cost, honoring an explicit
// color-identity override (e.g. Dryad Arbor) when present. Routed through the shared
// card_colors() (game_queries.h) so hybrid and Phyrexian pips (CR 202.2d: {B/P} makes the
// card black however it is paid) color the card here the same way they do everywhere else.
static ColorIdentity color_identity_from(const CardData &cd) {
    ColorIdentity ci;
    ci.colors = card_colors(cd);
    return ci;
}

void Orderer::generate_libraries(const Deck &deck_a, const Deck &deck_b) {
    Zone::Ownership owner = Zone::PLAYER_A;
    auto target_deck = deck_a;
    Coordinator &coordinator = Coordinator::global();

    // outer loop to assign each deck to proper player
    for (size_t i = 0; i < 2; i++) {
        if (i == 1) {
            owner = Zone::PLAYER_B;
            target_deck = deck_b;
        }
        size_t deck_position = 0;
        // loop through each card and create an entity in appropriate library per qty
        for (auto &&card_name : target_deck.main_deck) {
            for (size_t i = 0; i < card_name.first; i++) {  // qty
                // TODO this will probably need to be made a function for when it is repeated in the case of token
                // creation
                Entity card_id = coordinator.CreateEntity();
                auto card_data_id = load_card(card_name.second);
                coordinator.AddComponent(card_id, coordinator.GetComponent<CardData>(card_data_id));
                coordinator.AddComponent(card_id, Zone(Zone::LIBRARY, owner, owner));
                if (no_shuffle) {
                    auto &z = coordinator.GetComponent<Zone>(card_id);
                    z.distance_from_top = deck_position++;
                }
                auto &cd = coordinator.GetComponent<CardData>(card_id);
                coordinator.AddComponent(card_id, color_identity_from(cd));
            }
        }
    }

    if (!no_shuffle) {
        shuffle_library(Zone::PLAYER_A);
        shuffle_library(Zone::PLAYER_B);
    }

    // Build creature subtype lists for ChooseType (Cavern of Souls)
    for (size_t pi = 0; pi < 2; pi++) {
        Zone::Ownership player_owner = (pi == 0) ? Zone::PLAYER_A : Zone::PLAYER_B;
        Entity player_entity = (pi == 0) ? cur_game.player_a_entity : cur_game.player_b_entity;
        auto &player = coordinator.GetComponent<Player>(player_entity);

        std::set<int> seen_subtype_indices;
        for (auto e : mEntities) {
            auto &z = coordinator.GetComponent<Zone>(e);
            if (z.owner != player_owner) continue;
            if (!coordinator.entity_has_component<CardData>(e)) continue;
            auto &cd = coordinator.GetComponent<CardData>(e);
            bool is_creature = false;
            for (auto &t : cd.types)
                if (t.kind == TYPE && t.name == "Creature") { is_creature = true; break; }
            if (!is_creature) continue;
            for (auto &t : cd.types) {
                if (t.kind != SUBTYPE) continue;
                auto it = all_subtypes.find(t.name);
                if (it == all_subtypes.end()) continue;
                int idx = static_cast<int>(std::distance(all_subtypes.begin(), it));
                seen_subtype_indices.insert(idx);
            }
        }
        int list_idx = 0;
        for (int subtype_idx : seen_subtype_indices) {
            player.creature_subtypes.push_back({list_idx, subtype_idx});
            list_idx++;
        }
    }

    // Sideboard cards ("outside the game"): instantiate each deck's SIDEBOARD: section as
    // SIDEBOARD-zone entities so wish effects (Karn, the Great Creator -2, Origin$ Sideboard)
    // can find them in every game, not just via test-harness --sideboard presets. SIDEBOARD is
    // never serialized to the ML state (populate_gamestate ignores it), and this runs after the
    // creature-subtype scan above so ChooseType menus are unchanged. setup_companions reuses an
    // already-instantiated sideboard entity instead of creating its own, so no duplication.
    for (size_t i = 0; i < 2; i++) {
        Zone::Ownership sb_owner = (i == 0) ? Zone::PLAYER_A : Zone::PLAYER_B;
        const Deck &d = (i == 0) ? deck_a : deck_b;
        std::vector<std::string> sb_names;
        for (const auto &entry : d.sideboard)
            for (size_t n = 0; n < entry.first; n++) sb_names.push_back(entry.second);
        if (!sb_names.empty()) place_in_zone(sb_names, sb_owner, Zone::SIDEBOARD);
    }
}

void Orderer::draw_hands() {
    draw(Zone::PLAYER_A, 7, false);
    draw(Zone::PLAYER_B, 7, false);
}

// Draw `ct` cards one at a time. Each individual draw is a separate "would draw a
// card" event, so the dredge replacement effect is offered before each one.
void Orderer::draw(Zone::Ownership player, size_t ct, bool fire_draw_event) {
    for (size_t i = 0; i < ct; i++) {
        if (cur_game.ended) return;
        draw_one(player, fire_draw_event);
    }
}

// actual effect here; replacement (dredge) handled here, triggers handled by callers
void Orderer::draw_one(Zone::Ownership player, bool fire_draw_event) {
    // Dredge replacement (rule 702.52a / 614.1a): the player may replace this draw
    // with a dredge from their graveyard. If they do, no card is drawn.
    {
        Entity player_entity = (player == Zone::PLAYER_A) ? cur_game.player_a_entity
                                                          : cur_game.player_b_entity;
        ReplacementEvent rev;
        rev.type = ReplacementEvent::DRAW_CARD;
        rev.entity = player_entity;
        rev.affected_player = player;
        replacement::dispatch(rev);
        if (rev.draw_replaced) {
            // Mill N, then return the dredge card from graveyard to hand. The dredge card is
            // in the graveyard (not the library), so milling never touches it.
            std::string dname = global_coordinator.GetComponent<CardData>(rev.dredge_source).name;
            mill(player, static_cast<size_t>(rev.dredge_mill));
            add_to_zone(false, rev.dredge_source, Zone::HAND);
            game_log("%s dredges %s (milled %d)\n", player_name(player).c_str(), dname.c_str(),
                     rev.dredge_mill);
            return;
        }
    }

    // Find the single top card of the player's library.
    Entity top = 0;
    size_t best = 0;
    bool found = false;
    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if (card_zone.location != Zone::LIBRARY || card_zone.owner != player) continue;
        if (!found || card_zone.distance_from_top < best) {
            best = card_zone.distance_from_top;
            top = card;
            found = true;
        }
    }

    if (!found) {
        // Drew from an empty library: the drawing player loses.
        if (player == Zone::PLAYER_A) {
            printf("\nPlayer A decked - Player B wins!\n");
            cur_game.winner = Zone::PLAYER_B;
        } else {
            printf("\nPlayer B decked - Player A wins!\n");
            cur_game.winner = Zone::PLAYER_A;
        }
        cur_game.ended = true;
        return;
    }

    // Track drawn cards on the player's cards_drawn_this_turn list (for Sylvan Library)
    Entity player_entity = (player == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    auto &pl = global_coordinator.GetComponent<Player>(player_entity);
    game_log_private(player, "%s draws %s\n", player_name(player).c_str(),
             global_coordinator.GetComponent<CardData>(top).name.c_str());
    game_log_redacted(player, "%s draws a card\n", player_name(player).c_str());
    // Route through the canonical zone mover so a draw fires CARD_CHANGED_ZONE,
    // closes the library gap, and updates the known-top-of-library cache.
    add_to_zone(false, top, Zone::HAND);
    pl.cards_drawn_this_turn.push_back(top);

    // Fire PLAYER_DREW_CARD for this individual draw. The "first card in the
    // drawer's draw step" is flagged so triggers like Orcish Bowmasters can
    // ignore the turn-based draw while punishing every extra draw.
    Zone::Ownership active = cur_game.player_a_turn ? Zone::PLAYER_A : Zone::PLAYER_B;
    bool first_in_draw_step = false;
    if (cur_game.cur_step == DRAW && player == active) {
        first_in_draw_step = (pl.cards_drawn_this_draw_step == 0);
        pl.cards_drawn_this_draw_step++;
    }
    if (!fire_draw_event) return;
    Event draw_event(Events::PLAYER_DREW_CARD);
    draw_event.SetParam(Params::PLAYER, player_entity);
    draw_event.SetParam(Params::ENTITY, top);
    draw_event.SetParam(Params::FIRST_IN_STEP, first_in_draw_step ? 1 : 0);
    // Running 1-based count of cards this player has drawn this turn (cards_drawn_this_turn was
    // just pushed above, so its size includes this card). Read by a Mode$ Drawn | Number$ N
    // trigger (Tamiyo, Inquisitive Student's "your third card in a turn") to fire on exactly the
    // Nth draw — robust even when several cards are drawn in one batch (each event carries its own
    // ordinal), unlike reading the live counter at trigger-scan time.
    draw_event.SetParam(Params::AMOUNT, static_cast<uint32_t>(pl.cards_drawn_this_turn.size()));
    global_coordinator.SendEvent(draw_event);
}

std::vector<Entity> Orderer::mill(Zone::Ownership player, size_t ct) {
    std::vector<Entity> lib = get_library_contents(player);
    std::sort(lib.begin(), lib.end(), [](Entity a, Entity b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });
    std::vector<Entity> milled;
    for (size_t i = 0; i < ct && i < lib.size(); i++) {
        std::string cname = global_coordinator.entity_has_component<CardData>(lib[i])
                                ? global_coordinator.GetComponent<CardData>(lib[i]).name
                                : "card";
        // add_to_zone keeps distance_from_top and the known-top-of-library cache correct.
        add_to_zone(false, lib[i], Zone::GRAVEYARD);
        game_log("%s mills %s.\n", player_name(player).c_str(), cname.c_str());
        milled.push_back(lib[i]);
    }
    return milled;
}

std::vector<Entity> Orderer::get_graveyard(Zone::Ownership owner) {
    std::vector<Entity> contents;
    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if (card_zone.location == Zone::GRAVEYARD && card_zone.owner == owner) {
            contents.push_back(card);
        }
    }
    return contents;
}

// ordered stack, top first
std::vector<Entity> Orderer::get_stack() {
    std::vector<Entity> on_stack;
    for (auto &&card : mEntities) {
        auto &card_zone = global_coordinator.GetComponent<Zone>(card);
        if (card_zone.location == Zone::STACK) {
            on_stack.push_back(card);
        }
    }
    std::sort(on_stack.begin(), on_stack.end(), [](Entity const &a, Entity const &b) {
        return global_coordinator.GetComponent<Zone>(a).distance_from_top <
               global_coordinator.GetComponent<Zone>(b).distance_from_top;
    });

    return on_stack;
}


void Orderer::do_london_mulligan(bool player_a_goes_first) {
    int mulligans_a = 0;
    int mulligans_b = 0;
    bool a_kept = false;
    bool b_kept = false;

    // Keep/mulligan outcomes are public knowledge (both players see them), so
    // log via game_log. On a London-mulligan keep the final hand is 7 minus the
    // number of mulligans taken (the rest get bottomed).
    auto log_mull_decision = [&](Zone::Ownership owner, bool kept, int mulls) {
        if (kept)
            game_log("%s keeps a %d-card hand.\n", player_name(owner).c_str(), 7 - mulls);
        else
            game_log("%s mulligans.\n", player_name(owner).c_str());
    };

    auto do_bottom_deck = [&](Zone::Ownership owner, int count) {
        std::string pname = player_name(owner);
        for (int i = 0; i < count; i++) {
            auto hand = this->get_hand(owner);
            if (hand.empty()) break;
            game_log("%s: Choose card to put on library bottom (%d remaining):\n", pname.c_str(), count - i);
            std::vector<LegalAction> btm_actions;
            for (auto card : hand) {
                auto &cd = global_coordinator.GetComponent<CardData>(card);
                LegalAction la(PASS_PRIORITY, card, cd.name);
                la.category = ActionCategory::BOTTOM_DECK_CARD;
                btm_actions.push_back(la);
            }
            int choice = InputLogger::instance().get_input(btm_actions);
            this->add_to_zone(true, hand[static_cast<size_t>(choice)], Zone::LIBRARY);
        }
    };

    // One seat's keep/mulligan decision. The two seats are identical apart from
    // which player owns the hand and who holds priority while deciding.
    auto decide_for = [&](Zone::Ownership owner, bool &kept, int &mulligans, bool priority_is_a) {
        cur_game.player_a_has_priority = priority_is_a;
        {
            auto hand_display = this->get_hand(owner);
            game_log_private(owner, "%s hand:\n", player_name(owner).c_str());
            for (auto card : hand_display) {
                auto &data = global_coordinator.GetComponent<CardData>(card);
                game_log_private(owner, "%s\n", data.name.c_str());
            }
        }
        std::vector<LegalAction> mull_actions = {
            LegalAction(PASS_PRIORITY, std::string("Keep")),
            LegalAction(PASS_PRIORITY, std::string("Mulligan")),
        };
        mull_actions[0].category = ActionCategory::MULLIGAN;
        mull_actions[1].category = ActionCategory::MULLIGAN;
        int choice = InputLogger::instance().get_input(mull_actions);
        if (choice == 0) {
            kept = true;
            log_mull_decision(owner, true, mulligans);
            do_bottom_deck(owner, mulligans);
        } else {
            log_mull_decision(owner, false, mulligans);
            mulligans++;
            if (mulligans == 7) {
                kept = true;
                log_mull_decision(owner, true, mulligans);
                do_bottom_deck(owner, mulligans);
            } else {
                if (mulligans >= 3 && InputLogger::instance().is_machine_mode()) {
                    printf("MULLIGAN_PENALTY: %s\n", owner == Zone::PLAYER_A ? "A" : "B");
                    fflush(stdout);
                }
                auto hand = this->get_hand(owner);
                for (auto card : hand) {
                    this->add_to_zone(false, card, Zone::LIBRARY);
                }
                this->shuffle_library(owner);
                this->draw(owner, 7, false);
            }
        }
    };

    // CR 103.5: each round of keep/mulligan decisions is announced in turn
    // order, starting player first. That matters in bo3 games 2-3, where the
    // loser of the previous game (possibly B) is on the play — always letting
    // A decide first handed B a systematic information edge in those games.
    while (!a_kept || !b_kept) {
        if (player_a_goes_first) {
            if (!a_kept) decide_for(Zone::PLAYER_A, a_kept, mulligans_a, true);
            if (!b_kept) decide_for(Zone::PLAYER_B, b_kept, mulligans_b, false);
        } else {
            if (!b_kept) decide_for(Zone::PLAYER_B, b_kept, mulligans_b, false);
            if (!a_kept) decide_for(Zone::PLAYER_A, a_kept, mulligans_a, true);
        }
    }
}

bool Orderer::do_opening_hand_actions(bool player_a_goes_first) {
    bool any_ran = false;
    const Zone::Ownership order[2] = {player_a_goes_first ? Zone::PLAYER_A : Zone::PLAYER_B,
                                      player_a_goes_first ? Zone::PLAYER_B : Zone::PLAYER_A};
    for (auto player : order) {
        bool is_starting_player = (player == order[0]);
        // Snapshot the kept hand up front: a resolved ability moves its card out of the hand,
        // and no new card can join the opening hand mid-phase (CR 103.5: the opening hand is
        // the hand kept after mulligan resolution).
        auto hand = this->get_hand(player);
        for (auto card : hand) {
            if (!global_coordinator.entity_has_component<CardData>(card)) continue;
            auto &cd = global_coordinator.GetComponent<CardData>(card);
            if (cd.opening_hand_abilities.empty()) continue;
            // :!PlayFirst (Gemstone Caverns): only offered to a player NOT going first.
            if (cd.opening_hand_not_first && is_starting_player) continue;
            const Ability &first_ab = cd.opening_hand_abilities.front();
            std::string prompt =
                (first_ab.category == "ChangeZone" && first_ab.destination == Zone::BATTLEFIELD)
                    ? "begin the game with " + cd.name + " on the battlefield"
                    : "use " + cd.name + "'s opening-hand ability";
            // request_optional_yesno seats the decision on `player` (priority save/set/restore).
            if (!request_optional_yesno(player, prompt)) continue;
            // Same instantiation pattern as gift_abilities in Ability::resolve: copy the parsed
            // template, wire this card as the source and its holder as controller, and run it
            // through the normal resolve pipeline so zone-change replacements/ETB machinery apply.
            for (Ability ab : cd.opening_hand_abilities) {
                ab.source = card;
                ab.controller = player;
                ab.resolve(shared_from_this());
            }
            any_ran = true;
        }
    }
    return any_ran;
}

std::vector<Entity> Orderer::place_on_battlefield(const std::vector<std::string> &card_names,
                                                   Zone::Ownership owner) {
    Coordinator &coordinator = Coordinator::global();
    std::vector<Entity> placed;

    for (const auto &name : card_names) {
        Entity card_id = coordinator.CreateEntity();
        auto card_data_id = load_card(name);
        coordinator.AddComponent(card_id, coordinator.GetComponent<CardData>(card_data_id));
        coordinator.AddComponent(card_id, Zone(Zone::BATTLEFIELD, owner, owner));
        auto &z = coordinator.GetComponent<Zone>(card_id);
        z.controller = owner;

        auto &cd = coordinator.GetComponent<CardData>(card_id);
        coordinator.AddComponent(card_id, color_identity_from(cd));
        placed.push_back(card_id);
    }

    return placed;
}

std::vector<Entity> Orderer::place_in_graveyard(const std::vector<std::string> &card_names,
                                                Zone::Ownership owner) {
    return place_in_zone(card_names, owner, Zone::GRAVEYARD);
}

std::vector<Entity> Orderer::place_in_zone(const std::vector<std::string> &card_names,
                                           Zone::Ownership owner, Zone::ZoneValue zone) {
    Coordinator &coordinator = Coordinator::global();
    std::vector<Entity> placed;

    for (const auto &name : card_names) {
        Entity card_id = coordinator.CreateEntity();
        auto card_data_id = load_card(name);
        coordinator.AddComponent(card_id, coordinator.GetComponent<CardData>(card_data_id));
        coordinator.AddComponent(card_id, Zone(zone, owner, owner));
        auto &cd = coordinator.GetComponent<CardData>(card_id);
        coordinator.AddComponent(card_id, color_identity_from(cd));
        placed.push_back(card_id);
    }

    return placed;
}
