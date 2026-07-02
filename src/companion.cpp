#include "companion.h"

#include <string>
#include <vector>

#include "card_db.h"
#include "components/carddata.h"
#include "components/player.h"
#include "components/zone.h"
#include "ecs/coordinator.h"
#include "mana_system.h"
#include "systems/orderer.h"

extern Coordinator global_coordinator;

// The minimum constructed deck size (CR 100.2a). The engine does not enforce a deck size, so the
// companion deckbuilding restriction is evaluated against the actual main deck card count.
static constexpr size_t MINIMUM_DECK_SIZE = 60;

static size_t main_deck_card_count(const Deck &deck);
static void setup_companion_for(Zone::Ownership owner, const Deck &deck,
                                const std::shared_ptr<Orderer> &orderer);

static size_t main_deck_card_count(const Deck &deck) {
    size_t total = 0;
    for (const auto &entry : deck.main_deck) total += entry.first;
    return total;
}

bool deck_meets_companion_restriction(const std::string &restriction, const Deck &deck) {
    // "DeckSizePlus<N>" (Yorion): the starting deck must contain at least N cards more than the
    // minimum deck size. Generalized over N so a future companion sharing this restriction shape
    // reuses the same evaluator; an unrecognized restriction fails closed.
    const std::string prefix = "DeckSizePlus";
    if (restriction.rfind(prefix, 0) == 0) {
        std::string digits = restriction.substr(prefix.size());
        if (digits.empty()) return false;
        size_t n = 0;
        for (char c : digits) {
            if (c < '0' || c > '9') return false;  // not a plain number
            n = n * 10 + static_cast<size_t>(c - '0');
        }
        return main_deck_card_count(deck) >= MINIMUM_DECK_SIZE + n;
    }
    return false;
}

static void setup_companion_for(Zone::Ownership owner, const Deck &deck,
                                const std::shared_ptr<Orderer> &orderer) {
    Entity player_entity = get_player_entity(owner);
    if (!global_coordinator.entity_has_component<Player>(player_entity)) return;
    auto &player = global_coordinator.GetComponent<Player>(player_entity);

    // 1) Prefer a Companion already instantiated in the SIDEBOARD zone (test-harness --sideboard
    //    preset, or a bo3 sideboard). The entity is the one the special action will move to hand.
    Entity comp = 0;
    std::string restriction;
    Entity max_e = global_coordinator.GetMaxIssuedEntity();
    for (Entity e = 0; e < max_e; ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        const auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::SIDEBOARD || z.owner != owner) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        const auto &cd = global_coordinator.GetComponent<CardData>(e);
        if (cd.is_companion) {
            comp = e;
            restriction = cd.companion_restriction;
            break;
        }
    }

    // 2) Otherwise look for a Companion named in the deck's sideboard list (fallback for callers
    //    that haven't instantiated sideboard entities) and instantiate it below if eligible.
    std::string comp_name;
    if (comp == 0) {
        for (const auto &entry : deck.sideboard) {
            Entity tmpl = load_card(entry.second);
            if (tmpl == 0 || !global_coordinator.entity_has_component<CardData>(tmpl)) continue;
            const auto &cd = global_coordinator.GetComponent<CardData>(tmpl);
            if (cd.is_companion) {
                comp_name = entry.second;
                restriction = cd.companion_restriction;
                break;
            }
        }
        if (comp_name.empty()) return;  // no companion in this player's sideboard
    }

    // Deckbuilding gate (CR 702.139c): the companion is only usable if the starting deck satisfies
    // its restriction. When it isn't met the card stays outside the game with no special action.
    if (!deck_meets_companion_restriction(restriction, deck)) return;

    if (comp == 0) {
        auto placed = orderer->place_in_zone({comp_name}, owner, Zone::SIDEBOARD);
        if (placed.empty()) return;
        comp = placed.front();
    }
    player.chosen_companion = comp;
}

void setup_companions(const Deck &deck_a, const Deck &deck_b,
                      const std::shared_ptr<Orderer> &orderer) {
    setup_companion_for(Zone::PLAYER_A, deck_a, orderer);
    setup_companion_for(Zone::PLAYER_B, deck_b, orderer);
}
