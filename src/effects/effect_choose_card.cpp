#include "effects.h"

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/spell.h"
#include "../components/types.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"
#include "../ecs/entity.h"
#include "../input_logger.h"
#include "../action_processor.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool choose_card(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // ChooseEach (Ajani -4): each opponent keeps one of their nonland permanents of each
    // listed type; the kept permanents go into cur_game.chosen_cards and a SubAbility$
    // SacrificeAll then sacrifices the rest (ValidCards$ ...+nonChosenCard).
    if (!ab.choose_each.empty()) {
        Zone::Ownership opp = (ab.controller == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;

        // Split the "Artifact & Creature & Enchantment & Planeswalker" type list.
        std::vector<std::string> types;
        size_t tp = 0;
        while (tp <= ab.choose_each.size()) {
            size_t amp = ab.choose_each.find('&', tp);
            if (amp == std::string::npos) amp = ab.choose_each.size();
            std::string t = ab.choose_each.substr(tp, amp - tp);
            while (!t.empty() && t.front() == ' ') t.erase(t.begin());
            while (!t.empty() && t.back() == ' ') t.pop_back();
            if (!t.empty()) types.push_back(t);
            tp = amp + 1;
        }

        bool prev_priority = cur_game.player_a_has_priority;
        cur_game.player_a_has_priority = (opp == Zone::PLAYER_A);
        for (const auto &type : types) {
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
            game_log("%s chooses a %s to keep:\n", player_name(opp).c_str(), type.c_str());
            int choice = InputLogger::instance().get_input(picks);
            Entity kept = cands[static_cast<size_t>(choice)];
            cur_game.chosen_cards.insert(kept);
            game_log("%s keeps %s.\n", player_name(opp).c_str(),
                     global_coordinator.GetComponent<Permanent>(kept).name.c_str());
        }
        cur_game.player_a_has_priority = prev_priority;
        return true;
    }

    // Dauthi Voidwalker: choose an exiled card owned by opponent with a void counter,
    // then play it without paying its mana cost.
    Zone::Ownership ctrl = ab.controller;
    Zone::Ownership opp = (ctrl == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;

    std::vector<Entity> choices;
    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
        if (!global_coordinator.entity_has_component<Zone>(e)) continue;
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location != Zone::EXILE) continue;
        if (z.owner != opp) continue;
        if (cur_game.void_countered.find(e) == cur_game.void_countered.end()) continue;
        if (!global_coordinator.entity_has_component<CardData>(e)) continue;
        choices.push_back(e);
    }

    if (!choices.empty()) {
        std::vector<LegalAction> pick_actions;
        for (auto e : choices) {
            auto &cd = global_coordinator.GetComponent<CardData>(e);
            LegalAction la(PASS_PRIORITY, e, "Play " + cd.name + " (exiled, free)");
            la.category = ActionCategory::PLAY_FREE;
            pick_actions.push_back(la);
        }

        game_log("Choose an exiled card with a void counter:\n");
        int choice = InputLogger::instance().get_input(pick_actions);
        Entity chosen = choices[static_cast<size_t>(choice)];
        auto &cd = global_coordinator.GetComponent<CardData>(chosen);
        cur_game.void_countered.erase(chosen);

        // Determine if permanent or instant/sorcery
        bool is_permanent = false;
        for (auto &t : cd.types) {
            if (t.kind == TYPE && (t.name == "Creature" || t.name == "Artifact" ||
                t.name == "Enchantment" || t.name == "Planeswalker" || t.name == "Land")) {
                is_permanent = true;
                break;
            }
        }

        if (is_permanent) {
            orderer->add_to_zone(false, chosen, Zone::BATTLEFIELD);
            auto &cz = global_coordinator.GetComponent<Zone>(chosen);
            cz.controller = ctrl;
            game_log("%s plays %s from exile (Dauthi Voidwalker).\n",
                     player_name(ctrl).c_str(), cd.name.c_str());
        } else {
            // Instant/Sorcery: put on stack as a spell, let normal resolution handle it
            Spell sp;
            sp.caster = ctrl;
            global_coordinator.AddComponent(chosen, sp);
            for (auto &spell_ab : cd.abilities) {
                if (spell_ab.ability_type != Ability::SPELL) continue;
                Ability cast_ab = spell_ab;
                cast_ab.source = chosen;
                cast_ab.controller = ctrl;
                // Announce cast-time choices now (modal modes + all targets, CR 601.2b/c) as
                // casting normally would, so the spell doesn't fizzle for want of a target on
                // resolution. Guarded: this forced cast wasn't vetted by the offer-time cast
                // gate, so a spell with no castable target set skips announcement and simply
                // fizzles at resolution instead of forcing an empty target menu.
                if (spell_has_castable_targets(cast_ab, orderer, ctrl, cd.has_gift)) {
                    announce_spell_targets(cast_ab, orderer, ctrl);
                }
                global_coordinator.AddComponent(chosen, cast_ab);
                break;
            }
            orderer->add_to_zone(false, chosen, Zone::STACK);
            game_log("%s casts %s from exile (Dauthi Voidwalker).\n",
                     player_name(ctrl).c_str(), cd.name.c_str());
        }
    } else {
        game_log("No exiled cards with void counters to choose.\n");
    }
    return true;
}

}  // namespace effects
