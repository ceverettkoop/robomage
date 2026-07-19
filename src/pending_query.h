#ifndef PENDING_QUERY_H
#define PENDING_QUERY_H

#include <cstdint>
#include <vector>

#include "classes/action.h"
#include "components/zone.h"
#include "ecs/entity.h"

// A mid-flow decision parked for the main loop to emit (the suspension half of
// the snapshot-safe decision protocol). The suspending site stores the EXACT
// menu it would have prompted with — the loop top re-emits it verbatim rather
// than re-deriving it, so re-emission after a snapshot restore is trivially
// byte-identical (nothing mutates between suspend and resume). Lives on Game by
// value so a cur_game copy covers it for free. At most one is active game-wide.
struct PendingQuery {
    // Which flow parked the decision — the loop-top dispatcher switches on this
    // after the answer is read to hand control back to the right resume path.
    enum Tag {
        NONE,
        RESOLUTION,     // effect handler under StackManager::resolve_top
        SBE_LATCHED,    // SBE-pass prompt re-derived on resume (legend rule, ETB choices)
        TURN_DRAW,      // draw-step replacement (dredge) mini-frame
        CAST,           // PendingCast state machine (cast-time prompts)
        ACTIVATION,     // PendingActivation state machine
        ATTACK_TARGET,  // declare-attackers target sub-prompt
        BLOCK_TARGET,   // declare-blockers target sub-prompt
        DAMAGE_ASSIGN,  // combat damage assignment order
        TRIGGER_PLACE   // trigger ordering / trigger target selection
    };
    Tag tag = NONE;
    bool active = false;
    std::vector<LegalAction> menu;  // the suspended menu, re-emitted verbatim
    bool chooser_is_a = true;       // who answers (priority is persisted here on suspend)
    Entity decision_source = 0;     // pending-decision context for the observation
    uint64_t key = 0;               // site key for latched-answer re-convergence (SBE flavor)
    bool answered = false;
    int answer = -1;
    // player_a_has_priority at arm time (before the repoint at the chooser).
    // FrameCtx::ask's consume path restores it, reproducing the blocking path's
    // post-get_input priority restore — so code after a resumed ask sees the
    // exact ambient priority it would have seen inline.
    bool prev_priority = false;
};

// ── SBE_LATCHED helpers (latched-answer re-derivation flavor) ───────────────
//
// An SBE-pass prompt (legend-rule keep, ETB choose-a-creature-type / name-a-
// card) has no persisted continuation struct: the suspended site is re-FOUND
// deterministically by re-running state_based_effects from scratch on resume
// (already-applied SBAs are silent no-ops on the frozen state, so the deriving
// scan reaches the identical question). The site key proves that identity:
// pq_key folds the site tag, the chooser, a stable context (the conflicting
// legend name / the entering entity) and the menu size, and pq_take_latched
// fatals on any mismatch — a re-run that derives a different question while an
// answer is latched is an invariant violation, never a recoverable state.

// Which SBE prompt site computed a key (folded into the key itself).
enum class SbeSite : uint32_t {
    LEGEND_KEEP = 1,       // 704.5j keep choice (context: hash of the conflicting name)
    ETB_CHOOSE_TYPE = 2,   // ETB choose-a-creature-type (context: the entering entity)
    ETB_NAME_CARD = 3,     // ETB name-a-card (context: the entering entity)
    ETB_TAPPED_LIFE = 4,   // "enters tapped unless you pay N life" y/n (context: the entering entity)
};

uint64_t pq_key(SbeSite site, bool chooser_is_a, uint64_t context, size_t menu_size);

// First-arrival test + consume, called at the prompt site itself. Returns
// false when no query is parked (first arrival — the site logs its arm-time
// lines and either arms via pq_arm_sbe or blocks inline outside the main
// loop). Returns true after consuming a matching latched answer into *choice:
// the query is cleared and priority is restored from pq.prev_priority (the
// site never self-restores), exactly reproducing the blocking path's
// post-get_input restore. Key mismatch, a foreign tag, or an unanswered query
// reaching an SBE site is fatal.
bool pq_take_latched(uint64_t key, int *choice);

// Park an SBE prompt for the main loop to emit: tag SBE_LATCHED, the site key,
// and the exact menu; persists priority at the chooser (prev saved in the
// query for pq_take_latched's restore). The caller then cooperatively
// early-returns out of the SBA pass / apply_permanent_components /
// state_based_effects, deferring the pass's trigger drain to the resumed run.
void pq_arm_sbe(uint64_t key, std::vector<LegalAction> menu, Zone::Ownership chooser,
                Entity decision_source);

#endif /* PENDING_QUERY_H */
