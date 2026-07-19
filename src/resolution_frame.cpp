#include "resolution_frame.h"

#include <string>

#include "classes/game.h"
#include "ecs/coordinator.h"
#include "error.h"
#include "game_driver.h"
#include "input_logger.h"
#include "pending_query.h"

extern Game cur_game;
extern Coordinator global_coordinator;

FrameCtx FrameCtx::root() { return FrameCtx(true, 0); }

FrameCtx FrameCtx::blocking() { return FrameCtx(false, 0); }

bool FrameCtx::can_suspend() const {
    // Blocking contexts never suspend. Root contexts can, but only once the
    // main decision loop is running — a resolve reached outside it (pregame
    // SBE on test-harness preplaced permanents) has no loop top to park a
    // query at, so it blocks inline exactly as before.
    return root_mode && in_main_loop();
}

bool FrameCtx::resuming() const { return can_suspend() && cur_game.pending_query.active; }

FrameLevel &FrameCtx::current_level() {
    if (!root_mode || !cur_game.resolution.active ||
        depth_ >= cur_game.resolution.levels.size())
        fatal_error("FrameCtx: no active resolution frame level at this depth");
    return cur_game.resolution.levels[depth_];
}

int FrameCtx::ask(std::vector<LegalAction> menu, Zone::Ownership chooser, Entity decision_source) {
    if (can_suspend()) {
        PendingQuery &pq = cur_game.pending_query;
        if (pq.active) {
            // Consume the latched answer for THIS ask. The re-entered handler
            // must have rebuilt the identical menu (menu builds are pure and
            // nothing runs between suspend and resume) — a size mismatch means
            // it diverged, which would misapply the answer.
            if (!pq.answered)
                fatal_error("FrameCtx::ask re-entered with an unanswered pending query");
            if (pq.menu.size() != menu.size())
                fatal_error("FrameCtx::ask: menu size changed between arm and resume (" +
                            std::to_string(pq.menu.size()) + " vs " +
                            std::to_string(menu.size()) + ")");
            int answer = pq.answer;
            // Restore the pre-prompt priority, exactly as the blocking path's
            // post-get_input restore does, so code after the resumed ask sees
            // the same ambient priority it would have inline.
            cur_game.player_a_has_priority = pq.prev_priority;
            pq = PendingQuery{};
            return answer;
        }
        // Suspend: park the query, persist priority at the chooser, and hand
        // -1 back so the caller returns SUSPENDED mutating nothing. The main
        // loop re-emits the stored menu (loop-safe) and the resume path's
        // re-entered ask consumes the latched answer above.
        pq = PendingQuery{};
        pq.tag = PendingQuery::RESOLUTION;
        pq.active = true;
        pq.menu = std::move(menu);
        pq.chooser_is_a = (chooser == Zone::PLAYER_A);
        pq.decision_source = decision_source;
        pq.prev_priority = cur_game.player_a_has_priority;
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

// ── SBE_LATCHED helpers (pending_query.h) ───────────────────────────────────

uint64_t pq_key(SbeSite site, bool chooser_is_a, uint64_t context, size_t menu_size) {
    // Deterministic fold of the site identity. Only ever compared for equality
    // within one process (the key lives in cur_game and snapshots in-process),
    // so the exact mixing constants are irrelevant — distinctness across the
    // handful of simultaneously-derivable questions is all that matters.
    uint64_t h = static_cast<uint64_t>(site);
    h = h * 1000003ull ^ (chooser_is_a ? 1ull : 2ull);
    h = h * 1000003ull ^ context;
    h = h * 1000003ull ^ static_cast<uint64_t>(menu_size);
    return h;
}

bool pq_take_latched(uint64_t key, int *choice) {
    PendingQuery &pq = cur_game.pending_query;
    if (!pq.active) return false;  // first arrival — the site arms (or blocks)
    // An SBE prompt site is only ever reached with a query parked when the
    // main loop's SBE_LATCHED dispatch fell into the normal flow carrying an
    // answered latch — anything else parked here is a protocol violation.
    if (pq.tag != PendingQuery::SBE_LATCHED || !pq.answered)
        fatal_error("SBE prompt site reached with a foreign/unanswered pending query parked");
    if (pq.key != key)
        fatal_error("SBE latched-answer key mismatch: the resumed scan derived a different question");
    *choice = pq.answer;
    // Restore the pre-arm priority, exactly as the blocking path's
    // post-get_input restore does — the site never self-restores.
    cur_game.player_a_has_priority = pq.prev_priority;
    pq = PendingQuery{};
    return true;
}

void pq_arm_sbe(uint64_t key, std::vector<LegalAction> menu, Zone::Ownership chooser,
                Entity decision_source) {
    PendingQuery &pq = cur_game.pending_query;
    if (pq.active)
        fatal_error("pq_arm_sbe: a pending query is already parked");
    pq = PendingQuery{};
    pq.tag = PendingQuery::SBE_LATCHED;
    pq.active = true;
    pq.menu = std::move(menu);
    pq.chooser_is_a = (chooser == Zone::PLAYER_A);
    pq.decision_source = decision_source;
    pq.key = key;
    // Persist priority at the chooser (the loop-top emitter asserts it); the
    // pre-arm seat is restored by pq_take_latched at consume time.
    pq.prev_priority = cur_game.player_a_has_priority;
    cur_game.player_a_has_priority = pq.chooser_is_a;
}

ResolveStatus FrameCtx::resolve_child(const Ability &child_template, FrameLevel::ChildKind kind,
                                      int child_index, int iter_index,
                                      const std::function<void(Ability &child)> &bind,
                                      std::shared_ptr<Orderer> orderer) {
    if (!can_suspend())
        fatal_error("FrameCtx::resolve_child in a non-suspendable context — "
                    "blocking callers keep direct recursion through the blocking shim");
    ResolutionFrame &fr = cur_game.resolution;
    const size_t child_depth = depth_ + 1;
    if (fr.levels.size() < child_depth)
        fatal_error("resolve_child: parent level missing from the resolution frame");
    if (fr.levels.size() == child_depth) {
        // First entry: push the persisted in-flight copy and bind it ONCE from
        // the parent's CURRENT state. A resume never re-binds (risk R3):
        // earlier children may have mutated the parent state the bind reads,
        // so re-binding would diverge from the suspended in-flight copy.
        FrameLevel lv;
        lv.kind = kind;
        lv.child_index = child_index;
        lv.iter_index = iter_index;
        lv.work = child_template;
        fr.levels.push_back(std::move(lv));
        bind(fr.levels.back().work);
    } else {
        // Resume: the level at depth+1 must be THIS child. Every container
        // persists its loop progress (next_sub, CharmRt/RepeatRt/ImmediateRt
        // indices), so the resumed caller re-derives the identical call; a
        // mismatch means the resume path diverged from the suspended one.
        const FrameLevel &deeper = fr.levels[child_depth];
        if (deeper.kind != kind || deeper.child_index != child_index ||
            deeper.iter_index != iter_index)
            fatal_error("resolve_child resume mismatch: persisted child (kind " +
                        std::to_string(deeper.kind) + ", idx " +
                        std::to_string(deeper.child_index) + ", iter " +
                        std::to_string(deeper.iter_index) + ") vs re-derived (kind " +
                        std::to_string(kind) + ", idx " + std::to_string(child_index) +
                        ", iter " + std::to_string(iter_index) + ")");
    }
    FrameCtx child_ctx(true, child_depth);
    ResolveStatus st = fr.levels[child_depth].work.resolve(orderer, child_ctx);
    if (st == ResolveStatus::DONE) {
        // The completed child must be the deepest level (its own children pop
        // before it returns DONE).
        if (fr.levels.size() != child_depth + 1)
            fatal_error("resolve_child: child completed with orphaned deeper levels");
        fr.levels.pop_back();
    }
    return st;
}

// ── RESOLUTION-tag target asker (Batch 8) ───────────────────────────────────
// One target pick through the suspension machinery: consume-or-park via
// FrameCtx::ask on the AMBIENT seat (the call site already seated the chooser,
// so the ask's repoint/restore is a no-op — the asker itself never moves
// priority, per the TargetAsker contract).

int ResolutionTargetAsker::ask(const std::vector<LegalAction> &menu, Entity decision_source) {
    Zone::Ownership seat = cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
    return ctx.ask(menu, seat, decision_source);
}

bool ResolutionTargetAsker::resuming() const { return ctx.resuming(); }

static void pin_all(const std::vector<Entity> &entities, std::set<Entity> &pins) {
    for (auto e : entities)
        if (e != 0) pins.insert(e);
}

// pinned_entities() visitor over EffectRuntime: the per-handler pool slices a
// suspended frozen-pool loop references. A dig pins its WHOLE revealed slice
// (already-picked cards and the to-bottom remainder alike — the epilogue moves
// both), scry/rearrange pin their looked-at library slices, surveil its
// unassigned + kept-on-top cards, sylvan its drawn/chosen set (cards it may
// still put back on top). SacrificeRt/ChooseCardRt/UnlessRt/ChangeZoneSearchRt
// pin nothing extra: their menus are live battlefield/hand scans, and a
// search's whole candidate pool is already listed in the parked menu (menu
// pins), so there is no off-menu revealed state to protect.
static void pin_effect_runtime(const EffectRuntime &rt, std::set<Entity> &pins) {
    if (const auto *d = std::get_if<DigRt>(&rt)) {
        pin_all(d->lib, pins);
    } else if (const auto *s = std::get_if<ScryRt>(&rt)) {
        pin_all(s->lib, pins);
    } else if (const auto *sv = std::get_if<SurveilRt>(&rt)) {
        pin_all(sv->remaining, pins);
        pin_all(sv->to_top, pins);
    } else if (const auto *r = std::get_if<RearrangeRt>(&rt)) {
        pin_all(r->lib, pins);
    } else if (const auto *sy = std::get_if<SylvanRt>(&rt)) {
        pin_all(sy->drawn_in_hand, pins);
        pin_all(sy->chosen, pins);
    } else if (const auto *rp = std::get_if<RepeatRt>(&rt)) {
        // The outer remembered set repeat_each will restore at completion — it
        // may reference cards a determinize would otherwise resample (mirrors
        // the cur_game.remembered_entities pin below). CharmRt/ImmediateRt
        // hold no entity state of their own: in-flight mode/sub targets live
        // on the persisted parent work / child levels (pinned generically).
        pin_all(rp->saved_remembered, pins);
    }
}

std::set<Entity> collect_pending_pins() {
    std::set<Entity> pins;
    const PendingQuery &pq = cur_game.pending_query;
    if (pq.active) {
        // The parked menu's entities: a determinized world must keep them where
        // the menu (and the handler's pure rebuild on resume) expects them —
        // e.g. the library top card a suspended peek/reveal is looking at.
        for (const auto &la : pq.menu) {
            if (la.source_entity != 0) pins.insert(la.source_entity);
            if (la.target_entity != 0) pins.insert(la.target_entity);
        }
    }
    const ResolutionFrame &fr = cur_game.resolution;
    if (fr.active) {
        // In-flight nested levels carry their own by-value ability copies; the
        // ROOT level's work is unused — the resolving ability lives in the
        // stack entity's component, so read its targets from there.
        for (const auto &lv : fr.levels) {
            if (lv.work.source != 0) pins.insert(lv.work.source);
            if (lv.work.target != 0) pins.insert(lv.work.target);
            for (auto t : lv.work.targets)
                if (t != 0) pins.insert(t);
            // The level's suspended handler runtime: revealed/looked-at pool
            // slices (dig, scry, surveil, rearrange, sylvan) that must survive
            // a world resample in place.
            pin_effect_runtime(lv.rt, pins);
        }
        if (fr.stack_entity != 0 &&
            global_coordinator.entity_has_component<Ability>(fr.stack_entity)) {
            auto &ab = global_coordinator.GetComponent<Ability>(fr.stack_entity);
            if (ab.source != 0) pins.insert(ab.source);
            if (ab.target != 0) pins.insert(ab.target);
            for (auto t : ab.targets)
                if (t != 0) pins.insert(t);
        }
    }
    // A suspended trigger placement: the queued-but-not-yet-offered triggers'
    // sources and any already-bound targets (the parked menu's pins cover only
    // the currently offered choices; the rest of the queue is what the resumed
    // placement will 603.3d-check, target, and push).
    const TriggerPlacementRT &tp = cur_game.trigger_placement;
    if (tp.active) {
        for (const auto &pt : tp.queue) {
            if (pt.source != 0) pins.insert(pt.source);
            if (pt.ab.source != 0) pins.insert(pt.ab.source);
            if (pt.ab.target != 0) pins.insert(pt.ab.target);
            for (auto t : pt.ab.targets)
                if (t != 0) pins.insert(t);
        }
    }
    // A suspended cast (tag CAST): the spell being cast — its zone row (hand /
    // graveyard / exile) must survive a world resample so the resumed flow
    // (and its FINISH move-to-stack) finds it where it left it — plus any
    // already-announced targets on the half-built ability. This batch's
    // converted prompts all precede target announcement (have_ability is
    // false at every suspension) and their menus reference no hidden-zone
    // entities (the kicker/replicate/gift y/n and X ladders are entity-less;
    // the spell-sac menu lists battlefield permanents, covered by the menu
    // pins above) — the spell itself is the one hidden-zone reference.
    const Game::PendingCast &pcst = cur_game.pending_cast;
    if (pcst.active) {
        if (pcst.spell_entity != 0) pins.insert(pcst.spell_entity);
        if (pcst.have_ability) {
            if (pcst.ability.source != 0) pins.insert(pcst.ability.source);
            if (pcst.ability.target != 0) pins.insert(pcst.ability.target);
            for (auto t : pcst.ability.targets)
                if (t != 0) pins.insert(t);
        }
    }
    // The remembered set: a suspended resolution's accumulated Remembered$
    // references (Doomsday piles, RememberChanged) must survive a determinize.
    for (auto e : cur_game.remembered_entities)
        if (e != 0) pins.insert(e);
    return pins;
}
