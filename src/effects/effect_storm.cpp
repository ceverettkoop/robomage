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

#include "../action_processor.h"   // copy_spell_on_stack
#include "../components/spell.h"
#include "../ecs/coordinator.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;

bool effects::storm(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // The source storm spell must still be on the stack (the trigger resolves above it). If it's
    // already gone (e.g. countered in response), there is nothing to copy.
    if (ab.source == 0) return true;
    if (!global_coordinator.entity_has_component<Spell>(ab.source)) return true;
    if (ab.amount > 0)
        copy_spell_on_stack(ab.source, static_cast<int>(ab.amount), ab.controller, orderer);
    return true;
}
