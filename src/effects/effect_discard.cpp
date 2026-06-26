#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Forward declaration (see definition below).
static bool discard_filter_matches(Entity e, const std::string &discard_valid);

// True if the card entity `e` matches a DiscardValid$ filter spec — a "Card." head
// followed by '+'-delimited constraints. Supported constraints: non<Type> (the card must
// not have that card type) and NamedCard (the card's name must equal the name chosen by a
// preceding NameCard effect, cur_game.named_card; CR 201.4). An empty filter matches all.
static bool discard_filter_matches(Entity e, const std::string &discard_valid) {
    if (discard_valid.empty()) return true;
    auto &cd = global_coordinator.GetComponent<CardData>(e);
    std::string filter = discard_valid;
    if (filter.rfind("Card.", 0) == 0) filter = filter.substr(5);
    size_t fp = 0;
    while (fp < filter.size()) {
        size_t plus = filter.find('+', fp);
        if (plus == std::string::npos) plus = filter.size();
        std::string constraint = filter.substr(fp, plus - fp);
        if (constraint == "NamedCard") {
            // No name has been chosen → nothing matches (the named-card discard does nothing).
            if (cur_game.named_card.empty() || cd.name != cur_game.named_card) return false;
        } else if (constraint.rfind("non", 0) == 0) {
            std::string excluded_type = constraint.substr(3);
            for (auto &t : cd.types)
                if (t.name == excluded_type) return false;
        }
        fp = plus + 1;
    }
    return true;
}

bool discard(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Target player reveals hand, then either the controller picks ONE matching card
    // (RevealYouChoose — Thoughtseize/Duress) or all matching cards are discarded
    // (RevealDiscardAll — Cabal Therapy).
    Zone::Ownership tgt_owner = Zone::PLAYER_A;
    if (global_coordinator.entity_has_component<Player>(ab.target)) {
        tgt_owner = (ab.target == cur_game.player_a_entity) ? Zone::PLAYER_A : Zone::PLAYER_B;
    }
    std::vector<Entity> hand = orderer->get_hand(tgt_owner);
    game_log("%s reveals their hand:\n", player_name(tgt_owner).c_str());
    for (auto e : hand) {
        auto &cd = global_coordinator.GetComponent<CardData>(e);
        game_log("  %s\n", cd.name.c_str());
        // The whole hand is revealed to the caster: record each card's identity in
        // the belief state (match-scoped multi-hot + per-card known-in-hand flag).
        mark_card_revealed(e, tgt_owner);
    }

    // Filter by DiscardValid$ — "Card.nonLand", "Card.nonCreature+nonLand", "Card.NamedCard"
    const DiscardParams *dp = std::get_if<DiscardParams>(&ab.params);
    std::string discard_valid = dp ? dp->valid : std::string();
    std::string mode = dp ? dp->mode : std::string();
    std::vector<Entity> valid;
    for (auto e : hand)
        if (discard_filter_matches(e, discard_valid)) valid.push_back(e);

    // RevealDiscardAll (Cabal Therapy): the target player discards EVERY matching card; no choice.
    if (mode == "RevealDiscardAll") {
        if (valid.empty()) {
            game_log("No matching cards to discard.\n");
        } else {
            for (auto chosen : valid) {
                auto &cd = global_coordinator.GetComponent<CardData>(chosen);
                game_log("%s discards %s\n", player_name(tgt_owner).c_str(), cd.name.c_str());
                orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
            }
        }
        return true;
    }

    if (valid.empty()) {
        game_log("No valid cards to discard.\n");
    } else {
        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (ab.controller == Zone::PLAYER_A);
        game_log("%s chooses a card to discard:\n", player_name(ab.controller).c_str());
        std::vector<LegalAction> discard_actions;
        for (auto e : valid) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::DISCARD;
            discard_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(discard_actions);
        Entity chosen = valid[static_cast<size_t>(choice)];
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        game_log("%s discards %s\n", player_name(tgt_owner).c_str(), cd.name.c_str());
        orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
        cur_game.player_a_has_priority = prev_priority;
    }
    return true;
}

bool parse_discard(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "DiscardValid") { effect_params<DiscardParams>(ab).valid = value; return true; }
    // Discard Mode$ — only the discard modes are claimed here (other effects, e.g.
    // SetState's Mode$ Transform, use the same key with a different meaning).
    if (key == "Mode" && (value == "RevealYouChoose" || value == "RevealDiscardAll")) {
        effect_params<DiscardParams>(ab).mode = value;
        return true;
    }
    return false;
}

}  // namespace effects
