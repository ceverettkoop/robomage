#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../coverage.h"
#include "../ecs/coordinator.h"
#include "../queries/affected.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/players.h"

extern Coordinator global_coordinator;
extern Game cur_game;

static EffectObject::Duration effect_duration(const AbilityDef &def);
static EffectObject effect_object_of(const Ability &ab);
static void log_effect_object(const Ability &ab, const EffectObject &object);
static bool grant_may_play_from_exile(const Ability &ab);
static bool grant_cant_be_blocked(const Ability &ab);
static bool grant_cant_gain_life(const Ability &ab);
static bool grant_cast_with_flash(const Ability &ab);
static bool remembered_shield_object(const EffectObject &object);

// How long the object of `def` lasts: Duration$ Permanent is an emblem (CR 114), Duration$
// UntilYourNextTurn ends at the controller's untap (game.cpp), and the Forge default is cleanup.
static EffectObject::Duration effect_duration(const AbilityDef &def) {
    if (def.duration_permanent) return EffectObject::Duration::GAME;
    return def.duration_until_your_next_turn ? EffectObject::Duration::UNTIL_YOUR_NEXT_TURN
                                             : EffectObject::Duration::END_OF_TURN;
}

// The one effect object `ab` creates: what the Effect carries, its controller and source. Each
// trigger is bound to the controller, so the trigger scan fires it like any triggered ability.
static EffectObject effect_object_of(const Ability &ab) {
    EffectObject object;
    object.controller = ab.controller;
    object.duration = effect_duration(*ab.def);
    object.source = ab.source.lki_entity();
    object.source_vocab_idx = source_vocab_idx(object.source);
    object.statics = ab.def->effect_statics;
    object.replacements = ab.def->effect_replacements;
    if (!ab.def->effect_remember.is(ObjectSpec::DEFAULT))
        for (Entity e : resolve_filter_objects(ab.def->effect_remember, ResolveCtx::of(ab)))
            object.remembered.push_back(ObjectRef::of(e));
    for (const AbilityDef &trig : ab.def->effect_floating_triggers) {
        Ability ft(&trig);
        ft.controller = ab.controller;
        ft.floating_creator = object.source;
        ft.floating_creator_vocab_idx = object.source_vocab_idx;
        object.triggers.push_back(std::move(ft));
    }
    return object;
}

// Narrates what `object` is.
static void log_effect_object(const Ability &ab, const EffectObject &object) {
    if (object.is_emblem()) {
        game_log("%s gets an emblem.\n", player_name(ab.controller).c_str());
        return;
    }
    if (!object.triggers.empty())
        game_log("A floating triggered ability is created%s.\n",
                 object.duration == EffectObject::Duration::UNTIL_YOUR_NEXT_TURN ? " until your next turn"
                                                                                 : " until end of turn");
    if (!object.remembered.empty()) return;
    if (ab.def->has_effect_replacement(Effect::Replacement::CANT_BE_COUNTERED))
        game_log("Spells %s controls can't be countered this turn.\n", player_name(ab.controller).c_str());
}

// DB$ Effect | ReplacementEffects$ <DamageDone/Prevent shields> | RememberObjects$ Targeted (Maze
// of Ith): all combat damage the remembered creature would deal (ValidSource$ Card.IsRemembered)
// and/or be dealt (ValidTarget$ Card.IsRemembered) this turn is prevented (CR 615). False, with no
// shield to make, when no such creature is left.
static bool remembered_shield_object(const EffectObject &object) {
    const Entity who = object.remembered.empty() ? 0 : object.remembered.front().get();
    if (who == 0 || !global_coordinator.entity_has_component<Creature>(who)) return false;
    game_log("All combat damage dealt to and dealt by %s is prevented this turn.\n", entity_name(who).c_str());
    return true;
}

// DB$ Effect | StaticAbilities$ <SVar(MayPlay$ True, AffectedZone$ Exile)> — "you may play /
// cast those cards" for the exiled cards the Effect remembers: the cards the preceding
// RememberChanged$ exile just moved (RememberObjects$ Remembered: Light Up the Stage, Ugin -11)
// or the card a preceding ChooseCard chose (RememberObjects$ ChosenCard: Dauthi Voidwalker).
// Rather than instantiate a continuous-effect object, record a play-from-exile permission for
// each such card still in exile in cur_game.resolved_effects.impulse_cast_permission (CR 601.2, 305.1). The
// permission is used at priority through the ordinary cast and land-play actions, so the card's
// timing, the land-drop limit and every cast trigger apply as usual (CR 601.3, 305.2). With
// MayPlayWithoutManaCost$ it is cast without paying its mana cost (CR 118.9), otherwise for its
// normal costs; a land may be played unless the static's Affected$ is nonLand (Ugin: "cast").
// It lasts this turn, or until the end of the caster's next turn (Light Up the Stage), and
// lapses once the card leaves exile (ForgetOnMoved$ Exile).
static bool grant_may_play_from_exile(const Ability &ab) {
    if (!ab.def->effect_may_play_from_exile) return false;
    const std::vector<Entity> cards = ab.def->effect_remember_chosen_card
                                          ? cur_game.resolution.memory.chosen_cards.live()
                                          : live_entities(cur_game.resolution.memory.remembered);
    for (Entity card : cards) {
        if (!global_coordinator.entity_has_component<Zone>(card)) continue;
        if (global_coordinator.GetComponent<Zone>(card).location != Zone::EXILE) continue;
        if (!global_coordinator.entity_has_component<CardData>(card)) continue;
        Game::ImpulseCastPermission perm;
        perm.resource = ab.def->effect_may_play_free ? Game::ImpulseCastPermission::FREE
                                                : Game::ImpulseCastPermission::NORMAL;
        perm.caster = ab.controller;
        perm.allow_land = ab.def->effect_may_play_lands;
        perm.persist_until_end_of_next_turn = ab.def->duration_until_end_of_your_next_turn;
        perm.grant_turn = cur_game.turn_state.turn;
        cur_game.resolved_effects.impulse_cast_permission[card] = perm;
        game_log("%s may %s %s from exile%s%s.\n", player_name(ab.controller).c_str(),
                 perm.allow_land ? "play" : "cast",
                 global_coordinator.GetComponent<CardData>(card).name.c_str(),
                 ab.def->effect_may_play_free ? " without paying its mana cost" : "",
                 perm.persist_until_end_of_next_turn ? " until the end of their next turn"
                                                     : " this turn");
    }
    return true;
}

// DB$ Effect | StaticAbilities$ Unblockable | RememberObjects$ Self — a transient
// continuous effect that makes the source unblockable until end of turn (Kappa
// Cannoneer, CR 509.1b / 702.x). Modeled as a per-turn "can't be blocked" mark on the
// remembered creature (the source), set here and cleared at the cleanup step (514.2),
// rather than instantiating a continuous-effect object. The mark is read by the combat
// blocker-legality check so the creature is removed from every blocker's legal list.
static bool grant_cant_be_blocked(const Ability &ab) {
    if (ab.def->effect_static_ability != "Unblockable") return false;
    Entity who = ab.def->effect_remember_self ? ab.source.get() : ab.target.get();
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

// AB$ Effect | StaticAbilities$ <SVar(Mode$ CantGainLife | ValidPlayer$ ...)> (Roiling Vortex's
// {R}: "Your opponents can't gain life this turn."). Register the affected player(s) in the
// turn-long can't-gain-life set (CR 119.x); consulted centrally in player_gain_life and cleared
// at cleanup. Scope is resolved relative to this effect's controller. A sourceless turn-long
// grant, unlike a battlefield static.
static bool grant_cant_gain_life(const Ability &ab) {
    if (ab.def->effect_cant_gain_life == AbilityDef::CantGainLifeScope::NONE) return false;
    Zone::Ownership me = ab.controller;
    Zone::Ownership opp = opponent_of(me);
    switch (ab.def->effect_cant_gain_life) {
        case AbilityDef::CantGainLifeScope::OPPONENTS:
            cur_game.resolved_effects.cant_gain_life_this_turn.insert(opp);
            game_log("%s can't gain life this turn.\n", player_name(opp).c_str());
            break;
        case AbilityDef::CantGainLifeScope::YOU:
            cur_game.resolved_effects.cant_gain_life_this_turn.insert(me);
            game_log("%s can't gain life this turn.\n", player_name(me).c_str());
            break;
        case AbilityDef::CantGainLifeScope::ALL:
            cur_game.resolved_effects.cant_gain_life_this_turn.insert(me);
            cur_game.resolved_effects.cant_gain_life_this_turn.insert(opp);
            game_log("No player can gain life this turn.\n");
            break;
        default:
            break;
    }
    return true;
}

// AB$ Effect | StaticAbilities$ <SVar(Mode$ CastWithFlash | ValidCard$ <filter> | Caster$ You)>
// (Teferi, Time Raveler's +1: "Until your next turn, you may cast sorcery spells as though they
// had flash."). Record a cast-timing permission bound to this effect's controller for the named
// filter, good for the effect's Duration — until the controller's next turn (Duration$
// UntilYourNextTurn) or, absent that, until cleanup. Consulted by the cast-speed gate
// (rules_mod::cast_with_flash_active). A sourceless turn-scoped grant, like the ones above.
static bool grant_cast_with_flash(const Ability &ab) {
    if (!ab.def->effect_cast_with_flash) return false;
    Game::CastWithFlashPermission perm;
    perm.controller = ab.controller;
    perm.filter = ab.def->effect_cast_with_flash_filter;
    perm.until_your_next_turn = ab.def->duration_until_your_next_turn;
    cur_game.resolved_effects.cast_with_flash_permissions.push_back(std::move(perm));
    game_log("%s may cast %s spells as though they had flash%s.\n",
             player_name(ab.controller).c_str(),
             ab.def->effect_cast_with_flash_filter.empty() ? "" : ab.def->effect_cast_with_flash_filter.text().c_str(),
             ab.def->duration_until_your_next_turn ? " until their next turn" : " this turn");
    return true;
}

namespace effects {

// DB$ Effect: what a resolved spell or ability leaves behind (CR 611.2). Everything it carries that
// is a static, a replacement effect or a trigger goes on one effect object: its statics are gathered
// into g_active_statics every SBA pass with the controller as their owner (gather_active_statics),
// its replacements are walked by replacement::dispatch and its triggers by the trigger scan. An
// Effect also grants what the engine keeps by its own rows (each grant_* above).
//
// An Effect with none of these names "you may cast that card this turn" (Emry, Lurker of the Loch):
// its static (MayPlay$ True, AffectedZone$ Graveyard) lets the remembered card be cast from the
// graveyard until end of turn, kept in cur_game.resolved_effects.may_cast_this_turn, a per-turn
// cast-permission set (CR 601.3e) consumed by the casting path in determine_legal_actions and
// cleared each cleanup. The target's legality (an artifact card in the controller's own graveyard)
// is already enforced when the ability is put on the stack and re-verified at resolution; here we
// only grant the permission for a target that is still in a graveyard.
HandlerResult grant_cast(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;

    EffectObject object = effect_object_of(ab);
    const bool shield = ab.def->has_effect_replacement(Effect::Replacement::PREVENT_COMBAT_DAMAGE);
    const bool holds_abilities = !object.statics.empty() || !object.triggers.empty() || !object.replacements.empty();
    if (holds_abilities && (!shield || remembered_shield_object(object))) {
        if (!object.statics.empty() && !object.is_emblem()) coverage_row("effect_static");
        if (!object.triggers.empty() && object.is_emblem()) coverage_row("emblem_trigger");
        log_effect_object(ab, object);
        cur_game.resolved_effects.add(std::move(object));
    }
    bool granted = grant_may_play_from_exile(ab);
    granted = grant_cant_be_blocked(ab) || granted;
    granted = grant_cant_gain_life(ab) || granted;
    granted = grant_cast_with_flash(ab) || granted;
    if (holds_abilities || granted) return HandlerResult::DONE_RUN_SUBS;

    Entity tgt = ab.target.get();
    if (tgt == 0 || !global_coordinator.entity_has_component<Zone>(tgt)) return HandlerResult::DONE_RUN_SUBS;
    if (global_coordinator.GetComponent<Zone>(tgt).location != Zone::GRAVEYARD) return HandlerResult::DONE_RUN_SUBS;

    cur_game.resolved_effects.may_cast_this_turn.insert(tgt);

    std::string tname = global_coordinator.entity_has_component<CardData>(tgt)
        ? global_coordinator.GetComponent<CardData>(tgt).name : "card";
    game_log("%s may be cast from the graveyard this turn\n", tname.c_str());
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
