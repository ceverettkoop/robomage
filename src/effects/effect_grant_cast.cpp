#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// AB$ Effect granting "you may cast that card this turn" (Emry, Lurker of the Loch).
//
// Forge models this as a transient continuous Effect object whose static ability
// (MayPlay$ True, AffectedZone$ Graveyard) lets the remembered card be cast from the
// graveyard until end of turn. Rather than instantiate a stack/effect object, we record
// the targeted card in cur_game.may_cast_this_turn — a per-turn cast-permission set
// (CR 601.3e) consumed by the casting path in determine_legal_actions and cleared each
// cleanup. The target's legality (an artifact card in the controller's own graveyard) is
// already enforced when the ability is put on the stack and re-verified at resolution;
// here we only grant the permission for a target that is still in a graveyard.
HandlerResult grant_cast(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;

    // Emblem (CR 114): an AB$ Effect | StaticAbilities$ <SVar> | Duration$ Permanent creates a
    // player-owned emblem carrying the named permanent continuous static(s) — Kaito's [+1] "You
    // get an emblem with 'Ninjas you control get +1/+1.'" The emblem is an unremovable, zoneless
    // source; its statics are gathered into g_active_statics every SBA pass with the controller as
    // their owner (see gather_active_statics), so they apply through the normal layer engine. The
    // statics were parsed onto the ability at parse time. General over any emblem-making Effect.
    if (!ab.effect_emblem_statics.empty()) {
        Emblem emblem;
        emblem.controller = ab.controller;
        emblem.statics = ab.effect_emblem_statics;
        cur_game.emblems.push_back(std::move(emblem));
        game_log("%s gets an emblem.\n", player_name(ab.controller).c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

    // DB$ Effect | Triggers$ <SVar> — register a transient until-end-of-turn floating triggered
    // ability (Forth Eorlingas!'s "Whenever one or more creatures you control deal combat damage
    // to one or more players this turn, you become the monarch", CR 603.7e-style). Each parsed
    // trigger is bound to this effect's controller and pushed into cur_game.floating_triggers,
    // where the trigger scan fires it like any triggered ability; it lapses at cleanup. General
    // over any DB$ Effect that names a Triggers$ SVar.
    if (!ab.effect_floating_triggers.empty()) {
        // Duration$ UntilYourNextTurn (Tamiyo, Seasoned Scholar's +2 Effect) extends the floating
        // trigger past the end of this turn — it persists until the start of the controller's next
        // turn (removed at their untap, see game.cpp). The Forge default (no flag) lapses at cleanup.
        bool until_next_turn = ab.duration_until_your_next_turn;
        for (const auto &trig : ab.effect_floating_triggers) {
            Ability ft = trig;
            ft.controller = ab.controller;
            ft.duration_until_your_next_turn = until_next_turn;
            cur_game.floating_triggers.push_back(ft);
        }
        game_log("A floating triggered ability is created%s.\n",
                 until_next_turn ? " until your next turn" : " until end of turn");
        return HandlerResult::DONE_RUN_SUBS;
    }

    // DB$ Effect | StaticAbilities$ <SVar(MayPlay+MayPlayWithoutManaCost, AffectedZone$ Exile)>
    // | RememberObjects$ Remembered (Ugin, Eye of the Storms' -11): "Until end of turn, you may
    // cast those cards without paying their mana costs." The "those cards" are the colorless
    // nonland cards the preceding RememberChanged$ ChangeZone just exiled — still sitting in
    // cur_game.remembered_entities (DBCleanup clears them only AFTER this sub-ability). Rather
    // than instantiate a continuous-effect object, record a FREE cast-from-exile permission for
    // each remembered exiled card in cur_game.impulse_cast_permission (the same per-turn map the
    // alt-cost impulse cast uses), good until cleanup (CR 118.9 / 601.2f). The casting path
    // offers these from EXILE while they remain there (ForgetOnMoved$ Exile = the permission
    // lapses once a card leaves exile), pays no cost, and clears the map each cleanup.
    if (ab.effect_grant_free_cast_from_exile) {
        for (Entity card : cur_game.remembered_entities) {
            if (!global_coordinator.entity_has_component<Zone>(card)) continue;
            auto &cz = global_coordinator.GetComponent<Zone>(card);
            if (cz.location != Zone::EXILE) continue;
            if (!global_coordinator.entity_has_component<CardData>(card)) continue;
            Game::ImpulseCastPermission perm;
            perm.resource = Game::ImpulseCastPermission::FREE;
            perm.amount = 0;
            perm.caster = ab.controller;
            cur_game.impulse_cast_permission[card] = perm;
            game_log("%s may cast %s from exile without paying its mana cost this turn.\n",
                     player_name(ab.controller).c_str(),
                     global_coordinator.GetComponent<CardData>(card).name.c_str());
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // DB$ Effect | StaticAbilities$ <SVar(MayPlay$ True, AffectedZone$ Exile — NO
    // MayPlayWithoutManaCost)> | RememberObjects$ Remembered (Light Up the Stage): "Until the
    // end of your next turn, you may play those cards." The "those cards" are the two cards the
    // preceding RememberChanged$ Dig just exiled (still in cur_game.remembered_entities). Record a
    // NORMAL-cost play-from-exile permission for each remembered exiled card in
    // cur_game.impulse_cast_permission — paid for its normal cost (unlike Ugin's free grant),
    // LANDS permitted (it's "play", not "cast"), and persisting until the end of the caster's next
    // turn (Duration$ UntilTheEndOfYourNextTurn). ForgetOnMoved$ Exile: the permission lapses once
    // a card leaves exile (is played), which the casting/play path enforces by offering it only
    // while the card is still in exile.
    if (ab.effect_grant_play_from_exile) {
        for (Entity card : cur_game.remembered_entities) {
            if (!global_coordinator.entity_has_component<Zone>(card)) continue;
            if (global_coordinator.GetComponent<Zone>(card).location != Zone::EXILE) continue;
            if (!global_coordinator.entity_has_component<CardData>(card)) continue;
            Game::ImpulseCastPermission perm;
            perm.resource = Game::ImpulseCastPermission::NORMAL;
            perm.amount = 0;
            perm.caster = ab.controller;
            perm.allow_land = true;
            perm.persist_until_end_of_next_turn = ab.duration_until_end_of_your_next_turn;
            perm.grant_turn = cur_game.turn;
            cur_game.impulse_cast_permission[card] = perm;
            game_log("%s may play %s from exile%s.\n", player_name(ab.controller).c_str(),
                     global_coordinator.GetComponent<CardData>(card).name.c_str(),
                     perm.persist_until_end_of_next_turn ? " until the end of their next turn" : " this turn");
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // DB$ Effect | StaticAbilities$ Unblockable | RememberObjects$ Self — a transient
    // continuous effect that makes the source unblockable until end of turn (Kappa
    // Cannoneer, CR 509.1b / 702.x). Modeled as a per-turn "can't be blocked" mark on the
    // remembered creature (the source), set here and cleared at the cleanup step (514.2),
    // rather than instantiating a continuous-effect object. The mark is read by the combat
    // blocker-legality check so the creature is removed from every blocker's legal list.
    if (ab.effect_static_ability == "Unblockable") {
        Entity who = ab.effect_remember_self ? ab.source : ab.target;
        if (who != 0 && global_coordinator.entity_has_component<Creature>(who) &&
            is_battlefield_permanent(who)) {
            global_coordinator.GetComponent<Creature>(who).cant_be_blocked_this_turn = true;
            const char *nm = global_coordinator.entity_has_component<Permanent>(who)
                                 ? global_coordinator.GetComponent<Permanent>(who).name.c_str()
                                 : "Creature";
            game_log("%s can't be blocked this turn.\n", nm);
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // DB$ Effect | ReplacementEffects$ <CantHappen Counter on Spell.YouCtrl> (Veil of Summer:
    // "Spells you control can't be countered this turn"). Record the effect's controller in the
    // turn-long can't-be-countered set; consulted by effects::counter and cleared at cleanup.
    // A sourceless turn-long grant (the instant resolves to the graveyard), unlike Hexing
    // Squelcher's battlefield static.
    if (ab.effect_spells_uncounterable_this_turn) {
        cur_game.cant_counter_spells_of.insert(ab.controller);
        game_log("Spells %s controls can't be countered this turn.\n", player_name(ab.controller).c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

    // DB$ Effect | ReplacementEffects$ <DamageDone/Prevent shields> | RememberObjects$ Targeted
    // (Maze of Ith): register a turn-scoped combat-damage prevention shield on the remembered
    // creature — all combat damage it would deal (ValidSource$ Card.IsRemembered) and/or be dealt
    // (ValidTarget$ Card.IsRemembered) this turn is prevented (CR 615). The remembered creature is
    // the ability's inherited target (RememberObjects$ Targeted also stashed it in
    // remembered_entities). Sourceless turn-long grant, cleared at cleanup. General over any such
    // Effect (reusable by future fog/prevention cards).
    if (ab.effect_prevent_combat_damage_by_remembered || ab.effect_prevent_combat_damage_to_remembered) {
        Entity who = ab.target;
        if (who == 0 && !cur_game.remembered_entities.empty()) who = cur_game.remembered_entities.front();
        if (who != 0 && global_coordinator.entity_has_component<Creature>(who)) {
            Game::CombatDamagePreventionShield shield;
            shield.creature = who;
            shield.prevent_as_source = ab.effect_prevent_combat_damage_by_remembered;
            shield.prevent_as_target = ab.effect_prevent_combat_damage_to_remembered;
            cur_game.combat_damage_prevention_shields.push_back(shield);
            game_log("All combat damage dealt to and dealt by %s is prevented this turn.\n",
                     entity_name(who).c_str());
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // AB$ Effect | StaticAbilities$ <SVar(Mode$ CantGainLife | ValidPlayer$ ...)> (Roiling Vortex's
    // {R}: "Your opponents can't gain life this turn."). Register the affected player(s) in the
    // turn-long can't-gain-life set (CR 119.x); consulted centrally in player_gain_life and cleared
    // at cleanup. Scope is resolved relative to this effect's controller. A sourceless turn-long
    // grant, unlike a battlefield static.
    if (ab.effect_cant_gain_life != Ability::CantGainLifeScope::NONE) {
        Zone::Ownership me = ab.controller;
        Zone::Ownership opp = (me == Zone::PLAYER_A) ? Zone::PLAYER_B : Zone::PLAYER_A;
        switch (ab.effect_cant_gain_life) {
            case Ability::CantGainLifeScope::OPPONENTS:
                cur_game.cant_gain_life_this_turn.insert(opp);
                game_log("%s can't gain life this turn.\n", player_name(opp).c_str());
                break;
            case Ability::CantGainLifeScope::YOU:
                cur_game.cant_gain_life_this_turn.insert(me);
                game_log("%s can't gain life this turn.\n", player_name(me).c_str());
                break;
            case Ability::CantGainLifeScope::ALL:
                cur_game.cant_gain_life_this_turn.insert(me);
                cur_game.cant_gain_life_this_turn.insert(opp);
                game_log("No player can gain life this turn.\n");
                break;
            default:
                break;
        }
        return HandlerResult::DONE_RUN_SUBS;
    }

    // AB$ Effect | StaticAbilities$ <SVar(Mode$ CastWithFlash | ValidCard$ <filter> | Caster$ You)>
    // (Teferi, Time Raveler's +1: "Until your next turn, you may cast sorcery spells as though they
    // had flash."). Record a cast-timing permission bound to this effect's controller for the named
    // filter, good for the effect's Duration — until the controller's next turn (Duration$
    // UntilYourNextTurn) or, absent that, until cleanup. Consulted by the cast-speed gate
    // (rules_mod::cast_with_flash_active). A sourceless turn-scoped grant, like the ones above.
    if (ab.effect_cast_with_flash) {
        Game::CastWithFlashPermission perm;
        perm.controller = ab.controller;
        perm.filter = ab.effect_cast_with_flash_filter;
        perm.until_your_next_turn = ab.duration_until_your_next_turn;
        cur_game.cast_with_flash_permissions.push_back(std::move(perm));
        game_log("%s may cast %s spells as though they had flash%s.\n",
                 player_name(ab.controller).c_str(),
                 ab.effect_cast_with_flash_filter.empty() ? "" : ab.effect_cast_with_flash_filter.c_str(),
                 ab.duration_until_your_next_turn ? " until their next turn" : " this turn");
        return HandlerResult::DONE_RUN_SUBS;
    }

    Entity tgt = ab.target;
    if (tgt == 0 || !global_coordinator.entity_has_component<Zone>(tgt)) return HandlerResult::DONE_RUN_SUBS;
    if (global_coordinator.GetComponent<Zone>(tgt).location != Zone::GRAVEYARD) return HandlerResult::DONE_RUN_SUBS;

    cur_game.may_cast_this_turn.insert(tgt);

    std::string tname = global_coordinator.entity_has_component<CardData>(tgt)
        ? global_coordinator.GetComponent<CardData>(tgt).name : "card";
    game_log("%s may be cast from the graveyard this turn\n", tname.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
