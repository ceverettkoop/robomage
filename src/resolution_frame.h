#ifndef RESOLUTION_FRAME_H
#define RESOLUTION_FRAME_H

#include <memory>
#include <set>
#include <variant>
#include <vector>

#include "classes/action.h"
#include "components/ability.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;

// ── Suspension framework core ───────────────────────────────────────────────
//
// With -fno-exceptions the C++ stack under StackManager::resolve_top cannot be
// unwound or rebuilt, so a mid-resolution prompt can never be a search root.
// The suspension protocol replaces those live frames with a value-typed
// continuation persisted on Game: a handler that needs a decision asks through
// FrameCtx; in a suspendable context the query is parked (PendingQuery) and
// SUSPENDED propagates up through resolve_top to the main loop, which emits the
// stored menu at the loop top (loop-safe) and re-enters resolve_top with the
// latched answer. All state lives in cur_game / the ECS — never in statics —
// so a snapshot copy of cur_game covers the whole continuation.

// What a (possibly nested) resolve() call reports upward.
enum class ResolveStatus { DONE, SUSPENDED };

// What an effect handler reports to resolve(): completed (with or without the
// standard subability-chaining loop afterward), or suspended on a query.
enum class HandlerResult { DONE_RUN_SUBS, DONE_NO_SUBS, SUSPENDED };

// Per-handler runtime state for a suspended effect (dig pools, unless-loop
// counters, ...). Grown one alternative per converted handler in later batches;
// std::monostate covers handlers with no suspended state. The Batch 3 Shape A
// single-prompt handlers mostly need none (their menus rebuild purely and the
// pending query itself carries the answer); only the multi-ask loops persist
// their progress here.
struct SacrificeRt {
    bool pre_done = false;  // remembered-reset + unless-energy gate already ran
    size_t iter = 0;        // next sac_count iteration
};
struct ChooseCardRt {
    size_t type_idx = 0;    // next ChooseEach type index (Ajani -4 per-type loop)
};
using EffectRuntime = std::variant<std::monostate, SacrificeRt, ChooseCardRt>;

// One level of the persisted resolve() continuation: the ROOT is the stack
// object's own ability; nested levels hold the BY-VALUE in-flight copy of a
// child (sub-abilities resolve as copies today, so the frame must own the copy
// — resuming must NOT re-bind from mutated parent state).
struct FrameLevel {
    enum ChildKind { ROOT, SUB, GIFT, CHARM_MODE, REPEAT_SUB, IMMEDIATE };
    ChildKind kind = ROOT;
    int child_index = 0;   // index into the parent's child list (subabilities, charm modes, ...)
    int iter_index = 0;    // iteration counter for repeated children (repeat_each, gift loop)
    Ability work;          // in-flight copy for nested levels (unused at ROOT — the
                           // component itself is resolved there)
    int phase = 0;         // resume point in the phase-tagged resolve() (Batch 3+)
    int next_sub = 0;      // next subability to chain after the handler
    EffectRuntime rt;      // handler-specific suspended state
    bool saved_priority = false;  // priority to restore when this level completes
};

// The whole persisted resolution: armed by resolve_top on first entry, cleared
// by its completion epilogue. While active, advance_step short-circuits back to
// resolve_top (the resume path) instead of resetting pass tracking.
struct ResolutionFrame {
    bool active = false;
    Entity stack_entity = 0;         // the stack object being resolved (resume verifies it)
    bool prev_priority = false;      // player_a_has_priority to restore on completion
    bool counted_resolution = false; // ability_resolution_counts++ already applied (first entry)
    std::vector<Entity> saved_remembered;  // remembered set to restore on completion
                                           // (saved+cleared by frame_enter, restored by
                                           // frame_finish in stack_manager.cpp)
    std::vector<FrameLevel> levels;
};

// Handler-facing context threaded through resolve()/effect handlers. root()
// binds to cur_game.resolution and may suspend (per-site opt-in, later
// batches); blocking() reproduces today's inline get_input for every
// non-stack caller (sub-ability recursion, triggers resolved off-stack, ...).
// Converted and unconverted sites coexist: an unconverted prompt under a
// converted frame just blocks mid-iteration exactly like today.
class FrameCtx {
    public:
        static FrameCtx root();
        static FrameCtx blocking();
        bool is_root() const { return root_mode; }
        bool can_suspend() const;
        // True when re-entering suspended code with a latched answer pending —
        // the next ask() reached will consume it. Pre-ask side effects
        // (narrative logs, reveal bookkeeping) already ran when the query was
        // armed, so converted handlers guard them with !resuming() so a resume
        // never duplicates them.
        bool resuming() const;
        // One decision: latched/blocking paths return the chosen index; the
        // suspend path parks the query and returns -1 (the caller returns
        // SUSPENDED mutating nothing — check decision_suspended() to tell a
        // suspension apart from a CLI flag value).
        int ask(std::vector<LegalAction> menu, Zone::Ownership chooser, Entity decision_source);
        // The active level of the persisted frame (the ROOT level this batch).
        // The phase-tagged resolve() persists its resume point (phase/next_sub)
        // here. Only valid when can_suspend() — fatal otherwise.
        FrameLevel &level() { return current_level(); }
        // Drive a nested resolve as a persisted FrameLevel: first entry copies
        // `child_template` and binds it via `bind`; a resume re-enters the
        // persisted copy WITHOUT re-binding. (Placeholder until the container
        // batches — nothing calls it suspendably yet.)
        ResolveStatus resolve_child(const Ability &child_template, FrameLevel::ChildKind kind,
                                    int child_index, int iter_index,
                                    void (*bind)(Ability &child), std::shared_ptr<Orderer> orderer);
        // The active level's handler runtime state, default-constructing (and
        // switching the variant) on first access.
        template <class RT>
        RT &rt() {
            FrameLevel &lv = current_level();
            if (!std::holds_alternative<RT>(lv.rt)) lv.rt = RT{};
            return std::get<RT>(lv.rt);
        }

    private:
        explicit FrameCtx(bool is_root) : root_mode(is_root) {}
        FrameLevel &current_level();  // fatal_error outside an armed root frame
        bool root_mode = false;
};

// Entities a suspended decision references, which determinization must pin in
// place (a DETERMINIZE at a suspended root must not shuffle a card the parked
// menu / the resolving ability refers to). Batch 3 slice: the union of pending
// query menu entities, each FrameLevel's in-flight work targets/source (plus
// the ROOT ability component's), and cur_game.remembered_entities. The full
// per-handler EffectRuntime visitor (dig pools etc.) lands with the frozen-pool
// batch.
std::set<Entity> collect_pending_pins();

#endif /* RESOLUTION_FRAME_H */
