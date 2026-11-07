#include "effects.h"

#include <memory>
#include <vector>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../queries/affected.h"
#include "../queries/battlefield.h"
#include "../queries/copies.h"
#include "../queries/filters.h"
#include "../systems/orderer.h"
#include "../unless_payment.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

static Entity put_token_copy(const Token &tok, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer);

// Creates the token `tok`, a copy, under `ctrl`'s control.
static Entity put_token_copy(const Token &tok, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer) {
    Entity tok_entity = global_coordinator.CreateEntity();
    global_coordinator.AddComponent(tok_entity, Zone(Zone::HAND, ctrl, ctrl));
    global_coordinator.AddComponent(tok_entity, tok);
    orderer->add_to_zone(false, tok_entity, Zone::BATTLEFIELD);
    bootstrap_token_components(tok_entity, tok, ctrl, cur_game.timestamp);
    return tok_entity;
}

// CopyPermanent (Ocelot Pride): "for each token you control that entered this turn, create a
// token that's a copy of it." The set of permanents matching Defined$ Valid <filter> is
// snapshotted *before* any copy is made, so the freshly created copies — which also entered
// this turn — are not themselves re-copied. Each copy is a new token under the controller's
// control.
HandlerResult copy_permanent(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    Zone::Ownership ctrl = ab.controller;
    const Entity self = ab.source.get();
    if (self != 0 && global_coordinator.entity_has_component<Permanent>(self))
        ctrl = global_coordinator.GetComponent<Permanent>(self).controller;

    // Offspring (CR 702.175a): "create a token that's a copy of it, except it's 1/1."
    // Copy the source permanent itself, then override the copy's P/T to 1/1.
    if (ab.def->is_offspring_token) {
        Token tok = copied_token(ab.source.lki_entity(), ab.def->copy_overlay);
        if (tok.name.empty()) return HandlerResult::DONE_RUN_SUBS;
        put_token_copy(tok, ctrl, orderer);
        game_log("Offspring token copy created: %u/%u %s\n", tok.power, tok.toughness, tok.name.c_str());
        return HandlerResult::DONE_RUN_SUBS;
    }

    // Snapshot the matching permanents first (707.2 / "for each ... that entered this turn"). A
    // Defined$ Remembered card may sit in any zone (Lazotep Quarry's, in exile): copied_token reads
    // it there. ValidTgts$ names its target (Reflection of Kiki-Jiki's).
    std::vector<Entity> sources;
    if (ab.def->defined.is(ObjectSpec::VALID) || ab.def->defined.is(ObjectSpec::ENCHANTED) || ab.def->defined.is_remembered()) {
        ResolveCtx copied = ResolveCtx::of(ab);
        copied.controller = ctrl;
        sources = resolve_objects(ab.def->defined, copied);
    } else if (ab.def->valid_tgts != "N_A") {
        sources = affected_objects(ab);
    }

    // UnlessCost$ (CR 118.12): Springheart Nantuko's "you may pay {1}{G}. If you do, create a token
    // that's a copy", UnlessSwitched$ making it happen only if paid. UnlessResolveSubs$ WhenNotPaid
    // keeps what is chained for when it isn't.
    bool paid = false;
    const size_t unless = unless_amount(ab.def->unless_cost);
    if (unless > 0) {
        const Entity from = ab.source.lki_entity();
        bool suspended = false;
        const bool declined =
            run_unless_loop(unless, ctrl, orderer, from, from, ctx, suspended,
                            UnlessSubject{UnlessEffect::COPY, sources.empty() ? from : sources.front(), ab.def->unless_switched},
                            unless_pay_kind(ab.def->unless_cost), &ab.def->unless_cost.mana());
        if (suspended) return HandlerResult::SUSPENDED;
        paid = !declined;
        if (paid != ab.def->unless_switched) return HandlerResult::DONE_RUN_SUBS;
    }

    std::vector<ObjectRef> created;
    for (auto src : sources) {
        Token tok = copied_token(src, ab.def->copy_overlay);
        if (tok.name.empty()) continue;
        created.push_back(ObjectRef::of(put_token_copy(tok, ctrl, orderer)));
        game_log("Token copy created: %u/%u %s\n", tok.power, tok.toughness, tok.name.c_str());
    }
    register_at_eot_move(ab, created);
    return paid && ab.def->unless_subs_when_not_paid ? HandlerResult::DONE_NO_SUBS : HandlerResult::DONE_RUN_SUBS;
}

// AB$ Clone (Thespian's Stage: "{2}, {T}: CARDNAME becomes a copy of target land, except it has
// this ability."). CR 707.2 / 707.9b in-place copy: the SOURCE permanent (Thespian's Stage) becomes
// a copy of the target land while staying the SAME object (become_copy). Because a copied land is a
// fresh set of copiable values, any counters already on the source remain: a Stage cloning Dark
// Depths becomes a Dark Depths with ZERO ice counters (the ice counters come from Dark Depths'
// etbCounter, which applies only on ENTRY, never on becoming a copy), so its Mode$ Always state
// trigger sees no ice counters and fires. Orderer::add_to_zone ends the copy effect when the
// permanent leaves the battlefield (CR 400.7). CloneTarget$ Targeted (The Myriad Pools) is the
// other way round: the target becomes a copy of the spell that fired the trigger, whether or not
// that spell is still on the stack (Defined$ TriggeredCardLKICopy), and a target left unchosen
// ("up to one") copies nothing.
HandlerResult clone(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    (void)orderer;
    (void)ctx;
    const Entity copier = ab.def->clone_target_targeted ? ab.target.get() : ab.source.get();
    const Entity original = ab.def->clone_target_targeted ? ab.triggered_card.lki_entity() : ab.target.get();
    if (!is_battlefield_permanent(copier)) {
        game_log("Clone: nothing is left to become a copy\n");
        return HandlerResult::DONE_RUN_SUBS;
    }
    // CR 608.2b re-verify the target: a land still on the battlefield. The Myriad Pools' spell may
    // have left the stack: its card holds the values.
    if (original == 0 || (!ab.def->clone_target_targeted && !is_battlefield_permanent(original))) {
        game_log("Clone: target land is no longer valid\n");
        return HandlerResult::DONE_RUN_SUBS;
    }

    become_copy(copier, original, ab.def->copy_overlay, ab.def->clone_until_eot);
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
