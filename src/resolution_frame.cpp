#include "resolution_frame.h"

#include "classes/game.h"
#include "error.h"
#include "input_logger.h"
#include "pending_query.h"

extern Game cur_game;

FrameCtx FrameCtx::root() { return FrameCtx(true); }

FrameCtx FrameCtx::blocking() { return FrameCtx(false); }

bool FrameCtx::can_suspend() const {
    // Blocking contexts never suspend. Root contexts CAN, but every call site
    // starts blocking-equivalent and is flipped suspendable per conversion
    // batch — so for now the root mode blocks too (zero behavior change).
    return false;
}

FrameLevel &FrameCtx::current_level() {
    if (!root_mode || !cur_game.resolution.active || cur_game.resolution.levels.empty())
        fatal_error("FrameCtx: no active resolution frame level");
    return cur_game.resolution.levels.back();
}

int FrameCtx::ask(std::vector<LegalAction> menu, Zone::Ownership chooser, Entity decision_source) {
    if (can_suspend()) {
        // Suspend: park the query, persist priority at the chooser, and hand
        // -1 back so the caller returns SUSPENDED mutating nothing. The main
        // loop re-emits the stored menu and resumes with the latched answer.
        // (Unreachable until a site is flipped suspendable.)
        PendingQuery &pq = cur_game.pending_query;
        pq = PendingQuery{};
        pq.tag = PendingQuery::RESOLUTION;
        pq.active = true;
        pq.menu = std::move(menu);
        pq.chooser_is_a = (chooser == Zone::PLAYER_A);
        pq.decision_source = decision_source;
        cur_game.player_a_has_priority = pq.chooser_is_a;
        return -1;
    }
    // Blocking path — exactly today's mid-resolution prompt convention: repoint
    // priority at the chooser, expose the asking source as the pending-decision
    // context, read one choice inline, restore priority.
    bool prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = (chooser == Zone::PLAYER_A);
    int choice;
    {
        PendingDecisionScope pending(decision_source);
        choice = InputLogger::instance().get_input(menu);
    }
    cur_game.player_a_has_priority = prev_priority;
    return choice;
}

ResolveStatus FrameCtx::resolve_child(const Ability &child_template, FrameLevel::ChildKind kind,
                                      int child_index, int iter_index, void (*bind)(Ability &child),
                                      std::shared_ptr<Orderer> orderer) {
    (void)child_template;
    (void)kind;
    (void)child_index;
    (void)iter_index;
    (void)bind;
    (void)orderer;
    fatal_error("FrameCtx::resolve_child not implemented (container batches)");
}

std::set<Entity> collect_pending_pins() {
    // Real pin collection (menu entities, per-level work targets, EffectRuntime
    // visitor, remembered set) lands with the frozen-pool batch.
    return {};
}
