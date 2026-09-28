#ifndef RESOLUTION_H
#define RESOLUTION_H

#include <memory>

#include "components/ability.h"
#include "resolution_frame.h"

class Orderer;

// Resolving a spell or ability (CR 608.2): the target recheck, the condition gates, the effect
// handler (src/effects/) and the sub-ability chain, run as a phase-tagged state machine whose
// resume point lives in the persisted resolution frame, so a decision inside it can suspend.
// Only StackManager::resolve_top passes FrameCtx::root() (the suspendable path); every other
// caller uses the blocking form below.
ResolveStatus resolve_ability(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx ctx);

// Resolve `ab` inline with a blocking context (sub-ability recursion, charm modes, repeat_each,
// immediate and off-stack triggers, mana-ability riders). A blocking context never suspends.
void resolve_ability(Ability &ab, std::shared_ptr<Orderer> orderer);

#endif /* RESOLUTION_H */
