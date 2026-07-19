// Storm (CR 702.40) — resolution of the self-cast "copy this spell" triggered ability.
//
// K:Storm is parsed (src/parse.cpp) into a triggered ability that fires "When you cast this
// spell" (Events::SPELL_CAST, trigger_only_self). When that trigger fires, the storm count —
// the number of OTHER spells cast before the storm spell this turn, by EITHER player (CR
// 702.40a) — is locked into ab.amount (src/systems/state_manager_triggers.cpp). The trigger
// goes on the stack above its own spell; when it resolves here we put ab.amount COPIES of the
// source spell onto the stack. A copy is NOT cast (it pays no costs and triggers no cast
// triggers — so a copy storms nothing further, CR 707.10), and its controller MAY choose new
// targets (CR 702.40a / 707.10c). The copies sit above the original and resolve first; each
// resolves independently (for Flusterstorm: each copy counters its target unless that target's
// controller pays {1}).
//
// This handler is GENERAL over any Storm card: it owns no Flusterstorm-specific behavior. It
// reuses the shared copy-a-spell-on-stack mechanism (effect_copy_spell.cpp), the same primitive
// Replicate uses, rather than open-coding a parallel copy/targeting path.

#include "effects.h"

#include "../action_processor.h"   // copy_spell_begin / run_copy_spell
#include "../components/spell.h"
#include "../ecs/coordinator.h"
#include "../resolution_frame.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

HandlerResult effects::storm(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    // The Storm triggered ability is a separate object on the stack, independent of the spell
    // that created it (CR 113.7a / 702.40a). It still makes its copies even if that spell has
    // already left the stack — e.g. it was countered (Daze) before this trigger resolved. The
    // shared copy machine handles both cases: it copies the live spell if it's still on the
    // stack, else it rebuilds the copies from the spell's last-known copiable characteristics
    // (which survive on the card entity in the graveyard). CR 707.10.2: the copies are created
    // on the stack even though the original is gone.
    //
    // Resumable (Batch 10): the machine's progress persists as this handler's EffectRuntime
    // (CopySpellRT — its own rt, with the shared tsel a member field per the Batch 4 finding),
    // so a copy's target pick suspends as a loop-top pending decision through the
    // ResolutionTargetAsker (tag RESOLUTION — storm runs under a resolution frame).
    if (ab.source == 0) return HandlerResult::DONE_RUN_SUBS;
    if (ab.amount <= 0) return HandlerResult::DONE_RUN_SUBS;
    CopySpellRT local_rt;
    CopySpellRT &rt = ctx.can_suspend() ? ctx.rt<CopySpellRT>() : local_rt;
    if (!rt.active) {
        copy_spell_begin(rt, ab.source, static_cast<int>(ab.amount), ab.controller);
        if (!rt.active) return HandlerResult::DONE_RUN_SUBS;  // nothing to copy (no CardData)
    }
    ResolutionTargetAsker asker(ctx);
    if (run_copy_spell(rt, asker, orderer) != TargetStatus::DONE)
        return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}
