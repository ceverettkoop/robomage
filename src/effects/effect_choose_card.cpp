#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../input_logger.h"
#include "../queries/affected.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/filters.h"
#include "../queries/players.h"
#include "../queries/types.h"
#include "../resolution_memory.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static std::vector<Entity> choose_card_candidates(const Ability &ab,
                                                 std::shared_ptr<Orderer> orderer);

namespace effects {

HandlerResult choose_card(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // ChooseCard | Choices$ Card.ChosenType+YouOwn+IsImprinted (Atraxa, Grand Unifier): from the
    // imprinted cards (still in the controller's library after the reveal), choose one of the
    // current cur_game.resolution.memory.chosen_type to take. A "you may" choice — the controller may decline. A
    // chosen card is appended to the remembered set (RememberChosen$ True) so the trailing
    // Defined$ Remembered ChangeZone moves it to hand. CR 300/401.
    if (ab.def->choose_imprinted) {
        Zone::Ownership you = ab.controller;
        std::vector<Entity> cands;
        for (Entity e : live_entities(cur_game.resolution.memory.imprinted)) {
            if (!global_coordinator.entity_has_component<CardData>(e)) continue;
            auto &z = global_coordinator.GetComponent<Zone>(e);
            if (z.owner != you || z.location != Zone::LIBRARY) continue;  // YouOwn + still in library
            // A card already chosen for an earlier card type this resolution is no longer "among
            // them" (CR: one card per card type) — exclude it even though it hasn't physically left
            // the library yet (the trailing Defined$ Remembered move runs after the whole type loop).
            if (refs_contain(cur_game.resolution.memory.remembered, e)) continue;
            if (!cur_game.resolution.memory.chosen_type.empty() &&
                !card_has_type(global_coordinator.GetComponent<CardData>(e), cur_game.resolution.memory.chosen_type))
                continue;
            cands.push_back(e);
        }
        if (cands.empty()) return HandlerResult::DONE_RUN_SUBS;  // no imprinted card of this type left

        std::vector<LegalAction> picks;
        for (auto e : cands) {
            const std::string &nm = global_coordinator.GetComponent<CardData>(e).name;
            LegalAction la(PASS_PRIORITY, e, "Put " + nm + " (" + cur_game.resolution.memory.chosen_type + ") into hand");
            la.category = ActionCategory::CHOOSE_CARD;
            la.card_is_public = true;
            picks.push_back(la);
        }
        LegalAction none(PASS_PRIORITY, std::string("Put no ") + cur_game.resolution.memory.chosen_type + " card into hand");
        none.category = ActionCategory::CHOOSE_CARD;
        picks.push_back(none);

        if (!ctx.resuming())
            game_log("%s may put a %s card from among the revealed cards into their hand:\n",
                     player_name(you).c_str(), cur_game.resolution.memory.chosen_type.c_str());
        int choice = ctx.ask(std::move(picks), you, ab.source.lki_entity());
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        if (choice >= 0 && choice < static_cast<int>(cands.size())) {
            Entity chosen = cands[static_cast<size_t>(choice)];
            report_object(ab, EffectObjects::CHOSEN_IMPRINTED, chosen);
            game_log("%s chooses %s\n", player_name(you).c_str(),
                     global_coordinator.GetComponent<CardData>(chosen).name.c_str());
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // ChooseEach (Ajani -4): each opponent keeps one of their nonland permanents of each
    // listed type; the kept permanents go into cur_game.resolution.memory.chosen_cards and a SubAbility$
    // SacrificeAll then sacrifices the rest (ValidCards$ ...+nonChosenCard).
    if (!ab.def->choose_each.empty()) {
        Zone::Ownership opp = opponent_of(ab.controller);

        // Split the "Artifact & Creature & Enchantment & Planeswalker" type list.
        std::vector<std::string> types;
        size_t tp = 0;
        while (tp <= ab.def->choose_each.size()) {
            size_t amp = ab.def->choose_each.find('&', tp);
            if (amp == std::string::npos) amp = ab.def->choose_each.size();
            std::string t = ab.def->choose_each.substr(tp, amp - tp);
            while (!t.empty() && t.front() == ' ') t.erase(t.begin());
            while (!t.empty() && t.back() == ' ') t.pop_back();
            if (!t.empty()) types.push_back(t);
            tp = amp + 1;
        }

        // One ask per type — the loop index persists in the frame rt so a
        // resume re-enters the suspended type's pick (its candidate pool
        // rebuilds identically; already-answered types were applied at
        // consume time and are skipped by the persisted index).
        ChooseCardRt local_rt;
        ChooseCardRt &rt = ctx.can_suspend() ? ctx.rt<ChooseCardRt>() : local_rt;
        for (; rt.type_idx < types.size(); ++rt.type_idx) {
            const auto &type = types[rt.type_idx];
            std::vector<Entity> cands;
            for (auto e : orderer->mEntities) {
                if (!is_battlefield_permanent(e, opp)) continue;
                auto &perm = global_coordinator.GetComponent<Permanent>(e);
                if (permanent_has_type(perm, "Land")) continue;   // nonland pool
                if (!permanent_has_type(perm, type)) continue;
                cands.push_back(e);
            }
            if (cands.empty()) continue;

            std::vector<LegalAction> picks;
            for (auto e : cands) {
                const std::string &nm = global_coordinator.GetComponent<Permanent>(e).name;
                LegalAction la(PASS_PRIORITY, e, "Keep " + nm + " (" + type + ")");
                la.category = ActionCategory::CHOOSE_CARD;
                la.card_is_public = true;
                picks.push_back(la);
            }
            if (!ctx.resuming())
                game_log("%s chooses a %s to keep:\n", player_name(opp).c_str(), type.c_str());
            int choice = ctx.ask(std::move(picks), opp, ab.source.lki_entity());
            if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
            Entity kept = cands[static_cast<size_t>(choice)];
            report_object(ab, EffectObjects::KEPT, kept);
            game_log("%s keeps %s.\n", player_name(opp).c_str(),
                     global_coordinator.GetComponent<Permanent>(kept).name.c_str());
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // ChooseCard | Choices$ <filter> | ChoiceZone$ <zone> (Dauthi Voidwalker: "Choose an exiled
    // card an opponent owns with a void counter on it."): the player it names or targets, else its
    // controller, chooses one card in that zone matching the filter (Strategic Betrayal: "target
    // opponent exiles a creature they control", CR 608.2d). The choice becomes the resolution's chosen card
    // (cur_game.resolution.memory.chosen_cards), which a chained RememberObjects$ ChosenCard Effect acts on (Dauthi:
    // "You may play it this turn without paying its mana cost"). Mandatory$ True leaves no
    // "choose nothing" option; with no matching card nothing is chosen.
    std::vector<Entity> cands = choose_card_candidates(ab, orderer);
    if (cands.empty()) {
        if (!ctx.resuming()) game_log("There is no card to choose.\n");
        report_effect(ab, EffectObjects::CHOSEN);
        return HandlerResult::DONE_RUN_SUBS;
    }
    std::vector<LegalAction> picks;
    for (auto e : cands) {
        LegalAction la(PASS_PRIORITY, e, "Choose " + entity_name(e));
        la.category = ActionCategory::CHOOSE_CARD;
        // Cards in a public zone (exile is face up, CR 406.3) are public; a hand or library
        // choice is not.
        la.card_is_public = ab.def->choose_card_zone != Zone::HAND && ab.def->choose_card_zone != Zone::LIBRARY;
        picks.push_back(la);
    }
    if (!ab.def->mandatory) {
        LegalAction none(PASS_PRIORITY, std::string("Choose no card"));
        none.category = ActionCategory::CHOOSE_CARD;
        picks.push_back(none);
    }
    const Zone::Ownership chooser = affected_player(ab);
    if (!ctx.resuming()) game_log("%s chooses a card:\n", player_name(chooser).c_str());
    int choice = ctx.ask(std::move(picks), chooser, ab.source.lki_entity());
    if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
    report_effect(ab, EffectObjects::CHOSEN);
    if (choice >= 0 && choice < static_cast<int>(cands.size())) {
        Entity chosen = cands[static_cast<size_t>(choice)];
        report_object(ab, EffectObjects::CHOSEN, chosen);
        game_log("%s chooses %s.\n", player_name(chooser).c_str(), entity_name(chosen).c_str());
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects

// The objects a generic ChooseCard chooses among: those in its ChoiceZone$ matching its Choices$
// filter, read relative to the ability's controller (YouOwn / OppOwn) and against its targets. On
// the battlefield they are permanents, tokens too; elsewhere they are cards.
static std::vector<Entity> choose_card_candidates(const Ability &ab,
                                                 std::shared_ptr<Orderer> orderer) {
    const ResolveCtx rctx = ResolveCtx::of(ab);
    MatchCtx mctx;
    mctx.controller = ab.controller;
    mctx.source = ab.source.lki_entity();
    mctx.resolve = &rctx;
    std::vector<Entity> cands;
    for (auto e : orderer->mEntities) {
        if (global_coordinator.GetComponent<Zone>(e).location != ab.def->choose_card_zone) continue;
        const bool candidate = ab.def->choose_card_zone == Zone::BATTLEFIELD
                                   ? is_battlefield_permanent(e)
                                   : global_coordinator.entity_has_component<CardData>(e);
        if (!candidate) continue;
        if (!object_matches_filter(e, ab.def->choose_card_filter, mctx)) continue;
        cands.push_back(e);
    }
    return cands;
}
