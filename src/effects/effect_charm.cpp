#include "effects.h"

#include <string>
#include <vector>

#include "../classes/game.h"
#include "../error.h"
#include "../queries/characteristics.h"
#include "../resolution.h"

namespace effects {

// The bind applied once to each mode's persisted resolve_child copy (and the
// stamping the blocking path applies to the stored entry before its by-ref
// resolve). Forward-declared per CLAUDE.md.
static void stamp_mode(const Ability &parent, Ability &mode);

static void stamp_mode(const Ability &parent, Ability &mode) {
    mode.source = parent.source;
    mode.controller = parent.controller;
}

// Modal spell or ability (CR 700.2): "Choose one/two —". The mode(s) and their targets were
// announced as the object was put on the stack — cast (CR 601.2b/c), activated (602.2b) or
// triggered (603.3c/d), all through run_announce — and recorded in charm_chosen, in printed
// order. Resolution follows those modes in that order (CR 608.2c): each mode's own resolve()
// re-verifies its targets (CR 608.2b), so a mode whose targets became illegal fizzles
// individually without any prompting here. In a suspendable context each mode resolves as a
// persisted CHARM_MODE FrameLevel (a COPY of the stored charm_choices entry — the stored
// entry is never read again after its mode resolves), with CharmRt.announced_idx as the
// persisted loop cursor.
HandlerResult charm(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source.lki_entity());
    if (ab.charm_chosen.empty())
        fatal_error("modal ability of " + entity_name(ab.source.lki_entity()) +
                    " reached resolution with no announced mode (CR 700.2)");
    if (ctx.can_suspend()) {
        CharmRt &rt = ctx.rt<CharmRt>();
        for (; rt.announced_idx < static_cast<int>(ab.charm_chosen.size()); ++rt.announced_idx) {
            int idx = ab.charm_chosen[static_cast<size_t>(rt.announced_idx)];
            if (idx < 0 || static_cast<size_t>(idx) >= ab.charm_choices.size()) continue;
            Ability *parent = &ab;
            auto bind = [parent](Ability &mode) { stamp_mode(*parent, mode); };
            if (ctx.resolve_child(ab.charm_choices[static_cast<size_t>(idx)],
                                  FrameLevel::CHARM_MODE, idx, rt.announced_idx, bind,
                                  orderer) == ResolveStatus::SUSPENDED)
                return HandlerResult::SUSPENDED;
        }
    } else {
        for (int idx : ab.charm_chosen) {
            if (idx < 0 || static_cast<size_t>(idx) >= ab.charm_choices.size()) continue;
            Ability &chosen = ab.charm_choices[static_cast<size_t>(idx)];
            stamp_mode(ab, chosen);
            resolve_ability(chosen, orderer);
        }
    }
    // Skip subabilities — charm handles its own resolution
    return HandlerResult::DONE_NO_SUBS;
}

}  // namespace effects
