#include "effects.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../game_queries.h"
#include "../classes/action.h"
#include "../classes/game.h"
#include "../classes/match_state.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/player.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../input_logger.h"
#include "../stable_rng.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// Forward declaration (see definition below).
static bool discard_filter_matches(Entity e, const std::string &discard_valid);

// True if the card entity `e` matches a DiscardValid$ filter spec (Thoughtseize's
// Card.nonLand, Cabal Therapy's Card.NamedCard, Mox Diamond's Land), matched by the shared filter
// matcher against the card's characteristics. An empty filter matches every card.
static bool discard_filter_matches(Entity e, const std::string &discard_valid) {
    return discard_valid.empty() || card_matches_filter(e, discard_valid);
}

HandlerResult discard(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // Target player reveals hand, then either the controller picks ONE matching card
    // (RevealYouChoose — Thoughtseize/Duress) or all matching cards are discarded
    // (RevealDiscardAll — Cabal Therapy).
    // Default to the ability's controller so a `Defined$ You` self-discard with no player target
    // (e.g. Careful Study's "then discard two cards") discards from the CASTER's hand, not a
    // hard-coded Player A (which was wrong for the non-A seat). A targeted discard (Thoughtseize,
    // Hymn, Cabal Therapy) overrides this with the actual target player below.
    Zone::Ownership tgt_owner = ab.controller;
    if (global_coordinator.entity_has_component<Player>(ab.target.get())) {
        tgt_owner = seat_of_player(ab.target.get());
    }
    std::vector<Entity> hand = orderer->get_hand(tgt_owner);

    const DiscardParams *dp = std::get_if<DiscardParams>(&ab.params);
    std::string discard_valid = dp ? dp->valid : std::string();
    std::string mode = dp ? dp->mode : std::string();

    // Random (Hymn to Tourach): the target player discards NumCards$ cards chosen uniformly at
    // random from their hand — no reveal and no choice by any player (CR 701.8e/f). If the hand
    // holds fewer cards than requested, discard them all. Uses the game's seeded RNG so replays
    // are deterministic. This path runs before the hand is revealed because a random discard
    // does not reveal the hand.
    if (mode == "Random") {
        size_t count = ab.amount;  // NumCards$ N (do not hardcode); 0 means none.
        if (count > hand.size()) count = hand.size();
        if (count == 0) {
            game_log("No cards to discard at random.\n");
            return HandlerResult::DONE_RUN_SUBS;
        }
        // stable_shuffle, not std::shuffle: platform-stable given the seed
        // (see stable_rng.h).
        stable_shuffle(hand, cur_game.rng.engine);
        for (size_t i = 0; i < count; ++i) {
            Entity chosen = hand[i];
            auto &cd = global_coordinator.GetComponent<CardData>(chosen);
            game_log("%s discards %s at random\n", player_name(tgt_owner).c_str(), cd.name.c_str());
            // The discarded card enters a public zone — record its identity in the belief state.
            mark_card_revealed(chosen, tgt_owner);
            orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Mode$ TgtChoose (Archon of Cruelty: "target opponent ... discards a card"): the TARGET
    // player chooses one of their own cards to discard, and the hand is NOT revealed to the
    // controller (unlike RevealYouChoose, where the caster sees the hand and picks). Everything
    // else — the DiscardValid$ filter and the pick loop — is shared with the default branch.
    bool tgt_chooses = (mode == "TgtChoose");
    Zone::Ownership chooser = tgt_chooses ? tgt_owner : ab.controller;

    // Arm-only reveal: the logs and the belief-state recording ran when the pick
    // below was armed; a resume must not repeat them. TgtChoose does not reveal the hand.
    if (!ctx.resuming() && !tgt_chooses) {
        game_log("%s reveals their hand:\n", player_name(tgt_owner).c_str());
        for (auto e : hand) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            game_log("  %s\n", cd.name.c_str());
            // The whole hand is revealed to the caster: record each card's identity in
            // the belief state (match-scoped multi-hot + per-card known-in-hand flag).
            mark_card_revealed(e, tgt_owner);
        }
    }

    // RevealDiscardAll (Cabal Therapy): the target player discards EVERY matching card; no choice.
    if (mode == "RevealDiscardAll") {
        std::vector<Entity> valid;
        for (auto e : hand)
            if (discard_filter_matches(e, discard_valid)) valid.push_back(e);
        if (valid.empty()) {
            game_log("No matching cards to discard.\n");
        } else {
            for (auto chosen : valid) {
                auto &cd = global_coordinator.GetComponent<CardData>(chosen);
                game_log("%s discards %s\n", player_name(tgt_owner).c_str(), cd.name.c_str());
                orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
            }
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Player choice (RevealYouChoose — Thoughtseize; TgtChoose — Archon of Cruelty / Careful
    // Study): the chooser discards NumCards$ card(s) of their choice, ONE per query so each
    // pick round-trips in machine mode. NumCards$ 0 (unset) means one card (the common
    // single-discard case); Careful Study's NumCards$ 2 discards two (CR 701.8b). If the hand
    // runs out of matching cards, discard as many as possible. The DiscardValid$ pool is
    // rebuilt from the LIVE hand each pick (a picked card left the hand), and the pick count
    // persists in the level's DiscardRt so a machine-mode suspension resumes at the next pick.
    size_t count = ab.amount > 0 ? ab.amount : 1;
    DiscardRt local_rt;
    DiscardRt &rt = ctx.can_suspend() ? ctx.rt<DiscardRt>() : local_rt;
    for (; rt.discards_done < count; ++rt.discards_done) {
        std::vector<Entity> valid;
        for (auto e : orderer->get_hand(tgt_owner))
            if (discard_filter_matches(e, discard_valid)) valid.push_back(e);
        if (valid.empty()) {
            if (rt.discards_done == 0 && !ctx.resuming())
                game_log("No valid cards to discard.\n");
            break;
        }
        if (!ctx.resuming())
            game_log("%s chooses a card to discard:\n", player_name(chooser).c_str());
        std::vector<LegalAction> discard_actions;
        for (auto e : valid) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, cd.name);
            la.category = ActionCategory::DISCARD;
            discard_actions.push_back(la);
        }
        int choice = ctx.ask(std::move(discard_actions), chooser, ab.source.lki_entity());
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        Entity chosen = valid[static_cast<size_t>(choice)];
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        game_log("%s discards %s\n", player_name(tgt_owner).c_str(), cd.name.c_str());
        orderer->add_to_zone(false, chosen, Zone::GRAVEYARD);
    }
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_discard(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "DiscardValid") { effect_params<DiscardParams>(ab).valid = value; return true; }
    // Discard Mode$ — only the discard modes are claimed here (other effects, e.g.
    // SetState's Mode$ Transform, use the same key with a different meaning).
    if (key == "Mode" &&
        (value == "RevealYouChoose" || value == "RevealDiscardAll" || value == "Random" ||
         value == "TgtChoose")) {
        effect_params<DiscardParams>(ab).mode = value;
        return true;
    }
    return false;
}

}  // namespace effects
