#ifndef RESOLUTION_FRAME_H
#define RESOLUTION_FRAME_H

#include <deque>
#include <functional>
#include <memory>
#include <set>
#include <string>
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
// ── Batch 4: frozen-pool multi-pick loops (Shape B) ─────────────────────────
// Each rt persists the pool/counters its handler computed ONCE before its ask
// loop, so a resume re-enters the suspended pick against the exact same pool.
// The library slices these rts hold are what pinned_entities() feeds into
// determinization pinning: a world sampled at a suspended root must keep every
// revealed/looked-at card exactly where the resumed handler (and its epilogue)
// will look for it.
struct DigRt {
    bool init = false;            // slice/pool/counts computed + pre-loop log emitted
    std::vector<Entity> lib;      // the WHOLE revealed slice (picked + to-bottom); all pinned
    std::vector<Entity> pool;     // filter-matching candidates not yet picked
    std::vector<Entity> chosen;   // picks so far, in pick order
    size_t pick = 0;              // next pick index
    size_t take_count = 0;        // how many may be taken (resolved once)
    bool optional = false;        // "Take nothing" offered
};
struct ScryRt {
    bool init = false;            // slice fetched + "scries N" log emitted
    std::vector<Entity> lib;      // looked-at top slice, top-first; all pinned
    size_t idx = 0;               // next per-card keep/bottom decision
};
struct SurveilRt {
    bool init = false;            // slice fetched + look logs emitted
    std::vector<Entity> remaining;  // looked-at cards not yet assigned; pinned
    std::vector<Entity> to_top;     // chosen to stay on top, in choice order; pinned
};
struct RearrangeRt {
    bool init = false;            // slice fetched + look log emitted
    std::vector<Entity> lib;      // looked-at slice, top-first; all pinned
    std::vector<Entity> remaining;    // not yet slotted
    std::vector<Entity> chosen_order; // slot picks so far (deepest first)
    size_t pick = 0;              // next slot pick
    bool placed = false;          // put-back epilogue already ran (exactly once)
};
struct SylvanRt {
    bool init = false;            // draw-2 + drawn-in-hand scan done
    size_t draws_done = 0;        // draw-2 progress (a dredge question may suspend mid-batch)
    std::vector<Entity> drawn_in_hand;  // choose-from candidates; pinned
    std::vector<Entity> chosen;         // the (up to) 2 chosen cards; pinned
    size_t to_choose = 0;         // how many to choose (resolved once)
    size_t pick = 0;              // next choose-a-card pick
    size_t resolved = 0;          // next pay-4-life-or-top decision
};
// run_discard_unless (Reality Smasher's "unless they discard a card"): the
// yes/no answer and the count of discards already made. The per-discard menu is
// intentionally rebuilt from the LIVE hand each ask — the hand only shrinks by
// the discards themselves between asks. Shares the level's rt slot with the
// calling handler; safe because its only caller (effects::counter) holds no
// competing rt.
struct UnlessRt {
    bool pay_answered = false;    // "Discard N cards" chosen; now picking the cards
    size_t discards_done = 0;
};
// effects::discard's choose-N-to-discard loop (Careful Study's "discard two cards";
// also the single-card Thoughtseize/Archon path with count 1): the chooser picks one
// card per query so each round-trips in machine mode, and this persists how many picks
// are done so a suspension resumes at the next unmade pick. The per-pick menu is rebuilt
// from the LIVE hand each ask (a picked card left the hand), so nothing else persists and
// the visitor pins nothing extra for it.
struct DiscardRt {
    size_t discards_done = 0;
};
// change_zone's search loop (fetch lands, Doomsday's 5-pick multi-zone tutor):
// the outer pick count + the resolved dynamic bounds, computed once — earlier
// picks mutate the counted state (cards moved, remembered grown), so a resume
// must not re-evaluate them — plus the pre-repoint priority seat to restore in
// the epilogue. The search menus themselves rebuild purely from the live zones
// (the searchable pool IS the parked menu, so the menu pins cover it).
struct ChangeZoneSearchRt {
    bool init = false;            // counts resolved + prev seat saved
    bool prev_priority = false;   // player_a_has_priority to restore after the loop
    size_t num_to_move = 0;       // resolved ChangeNum (may be a count-SVar)
    int cmc_bound = -1;           // resolved dynamic mana-value bound (Aether Vial)
    size_t iter = 0;              // next pick index
};
// ── Batch 7: live-menu loops (Shape C) ──────────────────────────────────────
// change_zone's Defined$ Remembered bounded/optional selection (Cloak and
// Dagger's DBChangeZone: "you may exile up to one of the remembered
// candidates"): the candidate pool — and hence the menu — is rebuilt from the
// remembered set's CURRENT zones at every arm and every consume (a card
// already exiled left the eligible zones), so only the count of picks already
// made persists; it caps the remaining picks after a resume. The other two
// Shape C loops (change_zone's Yorion battlefield multi-select and
// run_unless_loop's MANA kind) persist nothing at all: their loops carry no
// counter — every pass is fully re-derived from live components (battlefield
// contents / floated mana), so a resume just rebuilds the identical menu.
struct ChangeZoneRememberedRt {
    int picked = 0;               // picks completed so far (bounds the cap)
};
// ── Shared target-selection sub-machine (Batch 5) ───────────────────────────
// Suspension-aware form of select_target/select_single_target
// (run_target_select, action_processor.cpp): the dynamic TargetMin$/TargetMax$
// bounds are stamped ONCE at entry, then one ask per pick; the candidate pool
// is re-derived on every pick as build_valid_targets(ability) minus the
// already-chosen targets (byte-identical to the blocking erase-chosen pool —
// nothing mutates between picks). Embedded BY VALUE as a member field of each
// host rt/state struct (trigger placement, the Batch 8 container rts below;
// cast announce in later batches) — NEVER its own EffectRuntime alternative:
// FrameCtx::rt<RT>() switches the variant on first access, so a separate
// alternative would dangle the host's rt reference (Batch 4 finding).
struct TargetSelectRT {
    bool active = false;     // bounds stamped; a pick is in flight
    int effective_min = 0;   // resolved TargetMin$ (dynamic stamping done once)
    int effective_max = 0;   // resolved TargetMax$
    int picked = 0;          // multi-target picks completed so far
};

// ── Batch 8: container effects (nested resolves as persisted FrameLevels) ───
// These rts persist a CONTAINER handler's own loop progress; the children they
// drive resolve as persisted deeper FrameLevels via FrameCtx::resolve_child
// (each child's in-flight state lives in ITS level, never here). The shared
// TargetSelectRT machines are member fields per the Batch 4 finding (a variant
// alternative of their own would dangle the host's rt reference).
struct CharmRt {
    // Announced path (modes + targets chosen at cast): next charm_chosen
    // position to resolve.
    int announced_idx = 0;
    // Resolution-time fallback (a charm that reached the stack unannounced):
    bool init = false;            // "(modes were not announced...)" log printed; taken sized
    std::vector<char> taken;      // CR 601.2b: a mode can be chosen only once
    int pick = 0;                 // mode picks completed
    int chosen_idx = -1;          // mode chosen for the CURRENT pick (-1 = ask pending)
    bool targets_done = false;    // current pick's target selection completed
    TargetSelectRT tsel;          // current mode's in-flight target selection
};
struct RepeatRt {
    bool init = false;            // saved_remembered captured
    bool player_setup = false;    // current player's remembered-set swap done
    int player_idx = 0;           // 0 = active player, 1 = non-active (APNAP); reused as the type index in the per-type path
    int sub_idx = 0;              // next RepeatSubAbility for the current player/type
    int trailing_idx = 0;         // per-type path: next trailing SubAbility$ resolved once after the loop
    std::vector<Entity> saved_remembered;  // outer remembered set to restore at completion
};
struct ImmediateRt {
    bool init = false;            // fire condition + optional energy cost resolved
    bool fire = false;            // reflexive effect fires (condition met, cost paid)
    int sub_idx = 0;              // next Execute/Cleanup sub-ability
    bool tsel_done = false;       // current sub's target selection completed
    TargetSelectRT tsel;          // current sub's in-flight target selection
};
// ── Batch 10: spell copies (CR 707.10 / 707.12) ─────────────────────────────
// The resumable form of copy-a-spell-on-stack (effect_copy_spell.cpp), shared
// by Replicate (cast FINISH — embedded in Game::PendingCast, driven with the
// CAST-tag asker) and Storm (resolution — this struct IS the storm handler's
// EffectRuntime, driven with the ResolutionTargetAsker). The current copy is
// built incrementally: the entity (CardData/ColorIdentity/Spell, deliberately
// NO Zone until placement) persists in the ECS across a suspension, and its
// in-flight ability lives here BY VALUE until it is complete enough to become
// the copy's Ability component. The no-legal-target path DESTROYS the copy
// entity mid-loop before any ask, so cur_copy is dropped (0) in the same
// stride — a parked pick always references a live copy.
struct CopySpellRT {
    bool active = false;
    Entity original = 0;      // the spell being copied (copiable characteristics re-read here)
    int remaining = 0;        // copies still to create, INCLUDING the one in flight
    bool controller_is_a = true;  // who controls (and targets) the copies
    Entity cur_copy = 0;      // the partially built copy entity (0 = none in flight)
    Ability work;             // the current copy's in-flight ability (targets accumulate here)
    bool have_ability = false;    // the original had a resolving ability to copy
    int phase = 0;            // within the current copy: 0 primary target, 1 subs, 2 charm modes
    size_t sub_idx = 0;       // next sub-ability to retarget (phase 1)
    size_t mode_pos = 0;      // next work.charm_chosen position to retarget (phase 2)
    TargetSelectRT tsel;      // the in-flight pick (member field per the Batch 4 finding)
};
// ── Batch 14: resolution-time draws (effects::draw) ─────────────────────────
// The drawing player, the resolved draw count (a dynamic NumCards$ is
// evaluated ONCE — never re-evaluated after dredge mills mutate the counted
// state), and completed draws, so a resume re-enters the parked dredge
// question mid-batch. Holds no entities — the parked dredge menu's graveyard
// cards are covered by the generic menu pins.
struct DrawRt {
    bool init = false;
    Zone::Ownership owner = Zone::PLAYER_A;
    size_t total = 0;
    size_t done = 0;
};
// change_zone's DefinedPlayer$ Player put-from-hand (Show and Tell: "Each player MAY put an
// artifact, creature, enchantment, or land card from their hand onto the battlefield"). Each
// player decides in APNAP order (CR 101.4 / 405.6, active player first); only the loop index
// persists across a suspension — the per-player candidate menu is re-derived from that player's
// live hand every pass (a live-menu loop; the parked menu pins cover the hand cards).
struct EachPlayerPutRt {
    int player_idx = 0;           // 0 = active player, 1 = non-active (APNAP)
};
using EffectRuntime = std::variant<std::monostate, SacrificeRt, ChooseCardRt, DigRt, ScryRt,
                                   SurveilRt, RearrangeRt, SylvanRt, UnlessRt, ChangeZoneSearchRt,
                                   ChangeZoneRememberedRt, CharmRt, RepeatRt, ImmediateRt,
                                   CopySpellRT, DrawRt, DiscardRt, EachPlayerPutRt>;

// What one run_target_select call reports: the ability's targets are fully
// chosen, or an ask parked a pending query (caller returns/suspends, mutating
// nothing further; re-enter with the same rt to resume at the parked pick).
enum class TargetStatus { DONE, SUSPENDED };

// The one-decision primitive run_target_select asks through, so the SAME
// machine serves every family: an implementation either reads the choice
// inline (blocking — today's select_target behavior) or arms
// cur_game.pending_query with its family's tag and reports suspension by
// returning a negative value. resuming() is true when re-entering with a
// latched answer pending — the next ask consumes it, and arm-time side effects
// (the "Choose target for ..." log) already ran, so they are guarded on it.
class TargetAsker {
    public:
        virtual ~TargetAsker() = default;
        // Returns the chosen index, or a negative value when the query was
        // parked (the caller distinguishes a real suspension from a CLI flag
        // via decision_suspended()).
        virtual int ask(const std::vector<LegalAction> &menu, Entity decision_source) = 0;
        virtual bool resuming() const = 0;
};

// ── Trigger placement (Batch 5) ─────────────────────────────────────────────
// One collected trigger waiting to be put on the stack (the persisted form of
// state_manager_triggers.cpp's PendingTrigger; Ability is a pure value type).
struct PendingTriggerRT {
    Ability ab;                 // fully prepared (source / controller / event-derived fields set)
    Zone::Ownership controller; // whose trigger this is (drives APNAP partitioning)
    Entity source = 0;          // source permanent (for logging)
    std::string label;          // choice label when its controller orders simultaneous triggers
    std::string log_line;       // narrative line emitted when it is placed on the stack
    bool needs_target = false;  // select a target at placement time if it still has legal targets
};

// The whole persisted APNAP placement (CR 603.3b): armed by place_triggers_apnap
// when a batch of triggers is collected, driven to completion by
// resume_trigger_placement (synchronously when nothing prompts; across main-loop
// iterations via TRIGGER_PLACE pending queries otherwise). The queue is the
// APNAP-flattened order — active player's group first, collection order within
// groups — and the current group is the leading run sharing queue.front()'s
// controller. Game value member, so a snapshot covers the parked placement.
struct TriggerPlacementRT {
    bool active = false;
    std::vector<PendingTriggerRT> queue;
    bool saved_priority = false;   // player_a_has_priority to restore at completion
    bool target_in_flight = false; // queue.front() is mid-target-selection (tsel live)
    TargetSelectRT tsel;           // the front trigger's in-flight target selection
};

// One level of the persisted resolve() continuation: the ROOT is the stack
// object's own ability; nested levels hold the BY-VALUE in-flight copy of a
// child (sub-abilities resolve as copies today, so the frame must own the copy
// — resuming must NOT re-bind from mutated parent state). Levels are stored in
// a deque, NOT a vector: a parent resolve holds live references into its level
// (phase/next_sub/rt — and for nested parents `this` IS levels[d].work) across
// resolve_child's push of the child level, and deque push_back/pop_back never
// invalidate references to other elements.
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
    std::deque<FrameLevel> levels;
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
        // THIS ctx's level of the persisted frame (levels[depth] — the ROOT at
        // depth 0, a resolve_child copy at depth > 0). The phase-tagged
        // resolve() persists its resume point (phase/next_sub) here. Only
        // valid when can_suspend() — fatal otherwise.
        FrameLevel &level() { return current_level(); }
        // Drive a nested resolve as a persisted FrameLevel at depth+1. First
        // entry (no level exists at depth+1): push a fresh FrameLevel holding
        // a COPY of `child_template`, then apply `bind` ONCE to that copy
        // (source/controller stamps, sub-target inheritance) while the parent
        // state it reads is current. Resume (a level already exists at
        // depth+1): it must match (kind, child_index, iter_index) — the
        // caller's persisted progress re-derives the same call — and the
        // PERSISTED copy is re-entered WITHOUT re-copying or re-binding
        // (risk R3: re-binding would re-read parent state mutated by earlier
        // children and diverge); any mismatch is fatal. The level is popped
        // when the child's resolve returns DONE; SUSPENDED propagates with the
        // level left in place. Only legal when can_suspend() — blocking
        // contexts keep today's direct recursion through the blocking shim.
        ResolveStatus resolve_child(const Ability &child_template, FrameLevel::ChildKind kind,
                                    int child_index, int iter_index,
                                    const std::function<void(Ability &child)> &bind,
                                    std::shared_ptr<Orderer> orderer);
        // This level's handler runtime state, default-constructing (and
        // switching the variant) on first access. Binds strictly to
        // levels[depth], so a parent's rt and a child's rt never share a slot.
        template <class RT>
        RT &rt() {
            FrameLevel &lv = current_level();
            if (!std::holds_alternative<RT>(lv.rt)) lv.rt = RT{};
            return std::get<RT>(lv.rt);
        }

    private:
        FrameCtx(bool is_root, size_t depth) : root_mode(is_root), depth_(depth) {}
        FrameLevel &current_level();  // fatal_error outside an armed root frame
        bool root_mode = false;
        size_t depth_ = 0;  // which FrameLevel this ctx binds to (root() = 0)
};

// TargetAsker with the RESOLUTION family tag (Batch 8): the suspendable form
// of a resolution-time select_target (charm's fallback mode targeting,
// immediate_trigger's per-sub selection). Delegates each pick straight to
// FrameCtx::ask — consume-or-park with tag RESOLUTION — asking on the AMBIENT
// seat, so it never repoints priority itself (Batch 5 finding: the call site
// seats the chooser, exactly as blocking select_target expects; during a root
// resolution the seat is already the resolving controller).
class ResolutionTargetAsker final : public TargetAsker {
    public:
        explicit ResolutionTargetAsker(FrameCtx &ctx) : ctx(ctx) {}
        int ask(const std::vector<LegalAction> &menu, Entity decision_source) override;
        bool resuming() const override;

    private:
        FrameCtx &ctx;
};

// Entities a suspended decision references, which determinization must pin in
// place (a DETERMINIZE at a suspended root must not shuffle a card the parked
// menu / the resolving ability refers to). The union of: pending query menu
// entities, each FrameLevel's in-flight work targets/source (plus the ROOT
// ability component's), cur_game.remembered_entities, and each level's
// EffectRuntime pool slices (dig/scry/surveil/rearrange lib slices, sylvan's
// drawn/chosen set) via the per-handler visitor in resolution_frame.cpp.
std::set<Entity> collect_pending_pins();

#endif /* RESOLUTION_FRAME_H */
