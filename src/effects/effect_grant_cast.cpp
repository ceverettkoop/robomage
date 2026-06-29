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
bool grant_cast(Ability &ab, std::shared_ptr<Orderer> orderer) {
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
        return true;
    }

    // DB$ Effect | Triggers$ <SVar> — register a transient until-end-of-turn floating triggered
    // ability (Forth Eorlingas!'s "Whenever one or more creatures you control deal combat damage
    // to one or more players this turn, you become the monarch", CR 603.7e-style). Each parsed
    // trigger is bound to this effect's controller and pushed into cur_game.floating_triggers,
    // where the trigger scan fires it like any triggered ability; it lapses at cleanup. General
    // over any DB$ Effect that names a Triggers$ SVar.
    if (!ab.effect_floating_triggers.empty()) {
        for (const auto &trig : ab.effect_floating_triggers) {
            Ability ft = trig;
            ft.controller = ab.controller;
            cur_game.floating_triggers.push_back(ft);
        }
        game_log("A floating triggered ability is created until end of turn.\n");
        return true;
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
        return true;
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
        return true;
    }

    // DB$ Effect | ReplacementEffects$ <CantHappen Counter on Spell.YouCtrl> (Veil of Summer:
    // "Spells you control can't be countered this turn"). Record the effect's controller in the
    // turn-long can't-be-countered set; consulted by effects::counter and cleared at cleanup.
    // A sourceless turn-long grant (the instant resolves to the graveyard), unlike Hexing
    // Squelcher's battlefield static.
    if (ab.effect_spells_uncounterable_this_turn) {
        cur_game.cant_counter_spells_of.insert(ab.controller);
        game_log("Spells %s controls can't be countered this turn.\n", player_name(ab.controller).c_str());
        return true;
    }

    Entity tgt = ab.target;
    if (tgt == 0 || !global_coordinator.entity_has_component<Zone>(tgt)) return true;
    if (global_coordinator.GetComponent<Zone>(tgt).location != Zone::GRAVEYARD) return true;

    cur_game.may_cast_this_turn.insert(tgt);

    std::string tname = global_coordinator.entity_has_component<CardData>(tgt)
        ? global_coordinator.GetComponent<CardData>(tgt).name : "card";
    game_log("%s may be cast from the graveyard this turn\n", tname.c_str());
    return true;
}

}  // namespace effects
