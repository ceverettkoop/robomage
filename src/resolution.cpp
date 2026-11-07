#include "resolution.h"

#include <set>
#include <string>
#include <vector>

#include "action_processor.h"
#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "coverage.h"
#include "components/permanent.h"
#include "components/player.h"
#include "components/spell.h"
#include "cost_payment.h"
#include "ecs/coordinator.h"
#include "effects/effects.h"
#include "input_logger.h"
#include "queries/activation.h"
#include "queries/affected.h"
#include "queries/battlefield.h"
#include "queries/characteristics.h"
#include "queries/filters.h"
#include "queries/lki.h"
#include "queries/players.h"
#include "queries/spells.h"
#include "queries/visibility.h"
#include "resolution_memory.h"
#include "svar_eval.h"
#include "systems/orderer.h"
#include "targeting.h"

extern Coordinator global_coordinator;
extern Game cur_game;

// ConditionCheckSVar$ / ConditionSVarCompare$ (CR 608.2c): does the checked SVar pass its
// comparison, against a number or another SVar? Forward-declared per CLAUDE.md.
static bool condition_svar_passes(const Ability &ab);
// Bind a chained sub-ability's target before it resolves, reading the script's stated
// Defined$ intent rather than blanket-inheriting the parent's target. See definition below
// (CR 608.2c). Forward-declared per CLAUDE.md.
static void bind_sub_target(const Ability &parent, Ability &sub);
// The subability-chaining loop shared by resolve_ability() phases 5 and 7: resolves
// parent.subabilities[next_sub..] in order, each as a persisted FrameLevel via
// resolve_child in a suspendable context (the per-sub CR 608.2c binding runs
// once, at push time) or as today's by-value blocking recursion otherwise.
// next_sub is the caller's persisted cursor (++ on each completed sub).
// Forward-declared per CLAUDE.md.
static ResolveStatus chain_subabilities(Ability &parent, std::shared_ptr<Orderer> orderer,
                                        FrameCtx &ctx, int &next_sub);
// Category-aware detail suffix for the "Resolving ability" log line. Display-only.
// Forward-declared per CLAUDE.md.
static std::string resolving_log_detail(const Ability &ab, std::shared_ptr<Orderer> orderer);
// CR 608.2b: every target of `ab` became illegal; it is removed from the stack without effect.
static void fizzle(const Ability &ab, const FrameCtx &ctx);
static void prune_illegal_targets(Ability &ab);

static bool condition_svar_passes(const Ability &ab) {
    const Entity src = ab.source.lki_entity();
    int val = evaluate_svar(ab.def->condition_check_svar, ab.controller, src);
    if (ab.def->condition_compare_svar_expr.empty())
        return compare_svar(val, ab.def->condition_svar_compare);
    // An SVar right-hand side (Thassa's Oracle: LEX, X = Count$Devotion.Blue).
    return apply_svar_op(val, ab.def->condition_svar_compare,
                         evaluate_svar(ab.def->condition_compare_svar_expr, ab.controller, src));
}

// Bind a chained sub-ability's target before it resolves. CR 608.2c: as a spell/ability
// resolves it follows its instructions in order, and each instruction references objects by
// its own definition. The Forge scripts already encode that intent in the sub's Defined$:
//   * The sub targets independently (its own ValidTgts$, a target was chosen for it at cast /
//     activation time, e.g. Cabal Therapy's DB$ Discard ValidTgts$ Player) -> keep that
//     target untouched.
//   * The sub references the PARENT's chosen target: Defined$ {Targeted, ParentTarget, Parent},
//     Defined$ TargetedController (Swords to Plowshares / Solitude GainLife and Smash to
//     Smithereens DealDamage read ab.target to find the targeted permanent's controller / power),
//     or no Defined$ at all (legacy implicit inherit) -> inherit parent.target.
//   * Any other explicit Defined$ that names an INDEPENDENT reference (You / Opponent /
//     Remembered / Self / TriggeredActivator / ...) -> leave sub.target alone; the effect
//     resolves that reference from its own Defined flag and never reads ab.target. Not
//     overwriting here is behavior-preserving (the old blanket sentinel set ab.target too, but
//     those handlers return before touching it).
static void bind_sub_target(const Ability &parent, Ability &sub) {
    // Propagate the chain's PLAYER target for DefinedPlayer$ Targeted reads — through EVERY
    // sub, including independently-targeted ones, whose own (card) target must not erase the
    // outer player target (see Ability::targeted_player).
    sub.targeted_player = parent.player_target_for_subs();
    if (sub.def->valid_tgts != "N_A") return;  // independently targeted at cast/activation — keep it
    const ObjectSpec &d = sub.def->defined;
    if (d.is(ObjectSpec::DEFAULT) || inherits_parent_target(d))
        sub.target = parent.target;  // inherit the parent's chosen target (or its controller)
    // else: independent Defined$ reference — leave sub.target alone (effect resolves its own ref)
}

void bind_chained_sub(const Ability &parent, Ability &sub) {
    sub.source = parent.source;
    sub.source_transforms = parent.source_transforms;
    bind_sub_target(parent, sub);  // CR 608.2c — Defined$-driven (see helper)
    sub.controller = parent.controller;
}

// See forward declaration at top of file. The bind closure holds the exact
// per-sub setup the old inline loop ran on its by-value copy (source stamp,
// CR 608.2c target binding, controller stamp, in that order); resolve_child
// applies it ONCE to the pushed copy and never on resume, so a suspended sub
// keeps its in-flight state even if earlier siblings mutated the parent.
static ResolveStatus chain_subabilities(Ability &parent, std::shared_ptr<Orderer> orderer,
                                        FrameCtx &ctx, int &next_sub) {
    for (; next_sub < static_cast<int>(parent.subabilities.size()); ++next_sub) {
        const Ability &sub_template = parent.subabilities[static_cast<size_t>(next_sub)];
        if (ctx.can_suspend()) {
            Ability *pp = &parent;
            auto bind = [pp](Ability &sub) { bind_chained_sub(*pp, sub); };
            if (ctx.resolve_child(sub_template, FrameLevel::SUB, next_sub, -1, bind, orderer) ==
                ResolveStatus::SUSPENDED)
                return ResolveStatus::SUSPENDED;
        } else {
            Ability sub_ab = sub_template;
            bind_chained_sub(parent, sub_ab);
            resolve_ability(sub_ab, orderer);
        }
    }
    return ResolveStatus::DONE;
}

// Detail suffix for the "Resolving ability (category: ...)" log line. Ability::amount is
// only what the effect actually uses for some categories, so a blanket ", amount: N" was
// often a meaningless 0 or an un-evaluated base. Policy: Pump shows its effective +A/+D
// (via the same resolve_pump_amounts the handler uses); categories whose amount IS the
// effect magnitude show the effective amount (the dynamic Count$ expression evaluated the
// same side-effect-free way the handler will); everything else shows nothing. Display-only.
static std::string resolving_log_detail(const Ability &ab, std::shared_ptr<Orderer> orderer) {
    auto signed_str = [](int v) {
        return (v >= 0 ? std::string("+") : std::string()) + std::to_string(v);
    };
    if (ab.def->kind == EffectKind::Pump || ab.def->kind == EffectKind::PumpAll) {
        const PumpParams *pp = std::get_if<PumpParams>(&ab.def->params);
        if (!pp || (pp->att == 0 && pp->def == 0 && pp->att_expr.empty() && pp->def_expr.empty()))
            return "";  // keyword-grant-only pump — no P/T change to report
        int att = 0, def = 0;
        effects::resolve_pump_amounts(pp, ab.controller, orderer, ab.target.get(), att, def);
        return ", " + signed_str(att) + "/" + signed_str(def);
    }
    // Counter effects keep their count in CounterParams (CounterNum$/its SVar), not
    // Ability::amount — read it the way the handlers do.
    if (ab.def->kind == EffectKind::PutCounter || ab.def->kind == EffectKind::PutCounterAll ||
        ab.def->kind == EffectKind::RemoveCounter) {
        const CounterParams *cp = std::get_if<CounterParams>(&ab.def->params);
        if (!cp) return "";
        int n = cp->count;
        if (!cp->count_expr.empty())
            n = static_cast<int>(
                evaluate_amount(cp->count_expr, ab.controller, 0, ab.target.get()));
        return ", amount: " + std::to_string(n);
    }
    // Discard only counts by Ability::amount in Random mode (Hymn to Tourach); the
    // reveal-and-choose / discard-all modes (Thoughtseize, Cabal Therapy) have no fixed count.
    if (ab.def->kind == EffectKind::Discard) {
        const DiscardParams *dp = std::get_if<DiscardParams>(&ab.def->params);
        if (dp && dp->mode == "Random") return ", amount: " + std::to_string(ab.def->amount);
        return "";
    }
    // Categories where Ability::amount (or its dynamic Count$ expression) is the effect's
    // magnitude, so printing it is informative.
    static const std::set<EffectKind> kAmountIsAuthoritative = {
        EffectKind::DealDamage, EffectKind::DamageAll, EffectKind::Draw,     EffectKind::Mill,
        EffectKind::GainLife,   EffectKind::LoseLife,  EffectKind::Scry,     EffectKind::Surveil};
    if (kAmountIsAuthoritative.count(ab.def->kind)) {
        size_t amt = ab.def->amount;
        // Draw/Mill treat a 0 amount as the "draw/mill a card" default — mirror it.
        if (amt == 0 && (ab.def->kind == EffectKind::Draw || ab.def->kind == EffectKind::Mill)) amt = 1;
        if (!ab.def->dynamic_amount_expr.empty())
            amt = evaluate_amount(ab.def->dynamic_amount_expr, ResolveCtx::of(ab));
        return ", amount: " + std::to_string(amt);
    }
    return "";
}

// The remembered set (cur_game.resolution.memory.remembered) is per-resolution scope in Forge — each
// resolving spell/ability instance has its OWN Remembered list (CR 608.2). This engine backs it
// with one global vector, so a top-level ability's resolution must not inherit remembered objects
// left behind by an EARLIER, unrelated ability that never cleared them (Skyclave Apparition's ETB
// exile pushes its target via RememberChanged and only clears on its own leaves-battlefield
// trigger). Without scoping, the next RememberChanged effect (Phelia's attack exile) appended to
// that stale set and its delayed-return trigger snapshotted BOTH cards, returning Skyclave's
// permanently-exiled target too (CR 603.7c — a delayed trigger references only the objects it was
// set up over).
//
// The ROOT (stack) resolve's scoping now lives in the persisted resolution frame
// (frame_enter saves+clears, frame_finish restores — stack_manager.cpp), because a
// suspension must keep the mid-resolution accumulations rather than unwinding them.
// This RAII guard covers only the remaining top-level BLOCKING resolves — mana-ability
// riders (mana_system.cpp), opening-hand abilities (orderer.cpp), and Static$ True
// off-stack triggers (state_manager_triggers.cpp) — which can never suspend, so
// destructor unwinding stays safe there. Top-level = blocking depth 0 AND no active
// resolution frame; nested blocking resolves (a parent's sub-abilities, or any blocking
// call under an active frame) keep sharing the parent's accumulated set, as their
// chained Remembered$ readers require. The depth counter tracks only synchronous
// blocking calls, so it is always 0 at any loop-top decision — never suspended state.
namespace {
int g_blocking_resolve_depth = 0;
struct BlockingRememberedScope {
    bool participates;
    bool top_level = false;
    std::vector<ObjectRef> saved;
    explicit BlockingRememberedScope(bool blocking) : participates(blocking) {
        if (!participates) return;  // root resolves: the frame owns the scoping
        top_level = (g_blocking_resolve_depth == 0 && !cur_game.resolution.active);
        if (top_level) {
            saved = cur_game.resolution.memory.remembered;
            cur_game.resolution.memory.remembered.clear();
        }
        ++g_blocking_resolve_depth;
    }
    ~BlockingRememberedScope() {
        if (!participates) return;
        --g_blocking_resolve_depth;
        if (top_level) cur_game.resolution.memory.remembered = saved;
    }
};
}  // namespace

static void fizzle(const Ability &ab, const FrameCtx &ctx) {
    if (ctx.is_stack_object()) cur_game.resolution.fizzled = true;
    game_log("%s fizzles\n", ab.def->category.c_str());
}

// CR 608.2b: a multi-target instruction whose targets are only PARTLY illegal still resolves,
// affecting only the still-legal targets (a spell that left the stack, a permanent that left the
// battlefield or gained protection, ...). Prune the illegal ones so every effect handler
// downstream sees legal targets only.
static void prune_illegal_targets(Ability &ab) {
    if (ab.targets.empty()) return;
    for (size_t i = 0; i < ab.targets.size();) {
        if (!is_legal_target(ab, ab.targets[i].get(), ab.controller)) {
            std::string tname = entity_name(ab.targets[i].lki_entity());
            game_log("%s is no longer a legal target; it is unaffected\n", tname.c_str());
            ab.targets.erase(ab.targets.begin() + i);
        } else {
            ++i;
        }
    }
    ab.target = ab.targets.empty() ? ObjectRef{} : ab.targets[0];
}

// Transitional blocking shim: every non-stack caller (sub-ability recursion, the
// gift loop, charm's chosen modes, repeat_each, immediate/delayed triggers, the
// mana-system riders) resolves inline exactly as before. A blocking ctx can
// never suspend, so the discarded status is always DONE. A top-level blocking resolve is one
// effect, so it gets its own CR 400.7j follow window (a nested one shares the open window), and
// the cards it revealed or looked at stop being shown when it finishes (CR 701.20a).
void resolve_ability(Ability &ab, std::shared_ptr<Orderer> orderer) {
    const bool top_level = !follow_window_open();
    {
        FollowWindowScope follow_window;
        (void)resolve_ability(ab, orderer, FrameCtx::blocking());
    }
    if (top_level) close_resolution_windows(0);
}

// resolve() as a staged machine: the body is a phase-tagged fall-through so a
// suspension (a converted prompt parking its query for the loop top) can be
// re-entered at the exact point it left off. The phase lives in the persisted
// ROOT FrameLevel for a suspendable (root) resolve, or in a local for blocking
// resolves — which run every phase in one pass, in EXACTLY the pre-refactor
// order (pure control-flow restructure; the never-suspending path is
// behavior-identical). Phases: 0 restore-remembered + OptionalDecider y/n;
// 1 reflexive sac; 2 one-shot gates/prune/log (skipped on resume so prune logs
// never duplicate) + the gift loop (each gift a persisted GIFT FrameLevel via
// resolve_child); 3 condition gates (evaluated once — failure routes to phase
// 7); 4 handler dispatch; 5 subability chaining (next_sub-driven; each sub a
// persisted SUB FrameLevel via resolve_child in a suspendable context);
// 6 NameCard clear + DONE; 7 the condition-failed sub chain (no NameCard
// clear, matching the old early return).
ResolveStatus resolve_ability(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx ctx) {
    BlockingRememberedScope remembered_scope(!ctx.is_root());
    int local_phase = 0;
    int local_next_sub = 0;
    int &phase = ctx.can_suspend() ? ctx.level().phase : local_phase;
    int &next_sub = ctx.can_suspend() ? ctx.level().next_sub : local_next_sub;

    if (phase == 0) {
        // Leaves-the-battlefield ability whose body references the cards its source had exiled
        // (Skyclave Apparition's TrigToken): restore those entities into the remembered set before
        // any gate or SVar runs, so ConditionPresent$ Card.ExiledWithSource, Remembered$CardManaCost
        // (token P/T), and TokenOwner$ RememberedOwner all read the exiled card (CR 608.2h). The
        // exiled_with snapshot was captured at the source's departure into last-known info.
        if (!ab.restore_remembered_exiled_with.empty()) {
            cur_game.resolution.memory.remembered = ab.restore_remembered_exiled_with;
        }

        // ResolvedLimit$ (CR 603.4 intervening-if, re-checked on resolution): an ability that
        // has already resolved its limit's number of times this turn does nothing.
        if (trigger_resolved_limit_reached(ab)) {
            game_log("Triggered ability has already resolved its limit this turn; it does nothing.\n");
            return ResolveStatus::DONE;
        }
        // OptionalDecider$ You ("you may ..."): the controller may decline the whole
        // triggered ability as it resolves (Ajani's exile-and-return-transformed).
        if (ab.def->trigger_optional && !trigger_cost_offered(ab.def->cost)) {
            int yc = ctx.ask(
                optional_yesno_menu("use " + entity_name(ab.source.lki_entity()) + "'s triggered ability"),
                ab.controller, ab.source.lki_entity());
            if (yc < 0 && decision_suspended()) return ResolveStatus::SUSPENDED;
            if (yc == 0) {
                game_log("%s declines the optional triggered ability.\n", player_name(ab.controller).c_str());
                return ResolveStatus::DONE;
            }
        }
        phase = 1;
    }
    if (phase == 1) {
        // A triggered ability pays its cost as it resolves (CR 603.2), and the cost makes the whole
        // effect conditional on it: on no sacrifice, discard or mana, the effect and its subabilities
        // are skipped. Activated abilities never reach here with ability_type == TRIGGERED, so
        // their already-paid sac is not double-charged.
        if (ab.def->ability_type == AbilityDef::TRIGGERED &&
            (ab.def->cost.self(CostPart::SACRIFICE) || trigger_cost_offered(ab.def->cost))) {
            const PayStatus paid = pay_trigger_cost(ab, ctx, orderer);
            if (paid == PayStatus::SUSPENDED) return ResolveStatus::SUSPENDED;
            if (paid != PayStatus::PAID) return ResolveStatus::DONE;
        }
        phase = 2;
    }
    if (phase == 2) {
        // The one-shot gates/prune/log section runs exactly once: a re-entry at
        // phase 2 (ctx.resuming() — a suspension inside the gift loop below
        // parked a query and the latch is still unconsumed) skips straight to
        // the gift loop, whose persisted next_sub resumes the suspended child.
        // Nothing here asks, so phase 2 with resuming() true always means a
        // gift suspension; the gates all passed on first entry.
        if (!ctx.resuming()) {
            // 603.4 intervening-if: re-check the trigger's "if" condition on resolution. If it is no
            // longer true the ability is removed from the stack and does nothing — not even its
            // subabilities fire (unlike a ConditionCheckSVar gate). EXCEPTION: a Mode$ Always
            // state-triggered ability (CR 603.8) uses its IsPresent$ as the TRIGGER condition,
            // evaluated by the state-trigger scan when it fires — it is NOT an intervening-if to be
            // re-checked at resolution. Dark Depths' "sacrifice it, if you do create Marit Lage" must
            // still create the token even though the sacrifice it just performed makes the "no ice
            // counters" condition read false (the source has left the battlefield). Neither this
            // check nor the stored-SVar gate below applies to a NoResolvingCheck$ trigger, whose
            // condition is part of the trigger event (Moonshadow's "while this creature has a
            // -1/-1 counter on it").
            const bool recheck_condition = !ab.def->trigger_no_resolving_check;
            if (recheck_condition && ab.def->intervening_if && !ab.def->trigger_state_condition &&
                !evaluate_present_condition(ab, ab.controller, orderer)) {
                game_log("Triggered ability's intervening-if condition is no longer true; it does nothing.\n");
                return ResolveStatus::DONE;
            }
            // Per-permanent stored-SVar gate (CheckSVar$ NAME | SVarCompare$ EQ0 on a Number$ scratch
            // int). Like the intervening-if it is re-checked at resolution (CR 603.4): if the source
            // permanent's latched scratch int no longer satisfies the comparison, the ability does
            // nothing (a latch the ability's own effect sets is written only after this check, so the
            // gate still reads its old value here for a legitimate fire).
            if (recheck_condition && !ab.def->stored_svar_gate_name.empty() &&
                !stored_svar_gate_passes(
                    ab.source.get(), ab.def->stored_svar_gate_name, ab.def->stored_svar_gate_compare)) {
                game_log("Triggered ability's stored-SVar gate is no longer satisfied; it does nothing.\n");
                return ResolveStatus::DONE;
            }
            // Pre-resolve target validity check (CR 608.2b): every targeted effect, a Pump
            // included, was targeted as it was put on the stack and fizzles here if its target
            // became illegal (it does NOT retarget). A stack object fizzles only when all its
            // targets are illegal, whichever of its instructions chose them: with one of a chained
            // instruction's still legal it resolves, and this instruction alone does nothing
            // (phase 3).
            const bool targets = ab.def->valid_tgts != "N_A";
            if (ctx.is_stack_object() ? every_target_illegal(ab) : targets && !targets_still_legal(ab)) {
                fizzle(ab, ctx);
                return ResolveStatus::DONE;  // subabilities do not fire; TODO revisit this in light of cards e.g. k-command
            }
            if (targets) prune_illegal_targets(ab);
            // The ability resolves: count it toward its ResolvedLimit$ and, as the stack object,
            // its source's Count$ResolvedThisTurn, before its effects read the count (a declined
            // optional ability, an unmet intervening-if or illegal targets returned above
            // uncounted, CR 603.4 / 608.2b).
            count_resolution(ab, ctx.is_stack_object());
            // RememberTargets/RememberObjects: stash the target(s) so chained
            // ChangeType$ Remembered.sameName subabilities can match by name (Surgical Extraction).
            report_targets(ab);
            apply_memory_ops(ab);
            if (!ab.replacing_an_event)
                game_log("Resolving ability (category: %s%s)\n", ab.def->category.c_str(),
                         resolving_log_detail(ab, orderer).c_str());
            next_sub = 0;  // gift-loop cursor (fresh levels start at 0; explicit for clarity)
        }  // end of the one-shot (non-resuming) section

        // Gift (CR 702.176c): if this spell promised its gift, the promised opponent receives the gift
        // BEFORE the spell's other effects. The gift effect(s) are carried on the primary (spell)
        // ability; run them first when ChosenCost::gift_promised is set on the source spell. The token's
        // TokenOwner$ Promised routes it to the opponent of this ability's controller (effects::token).
        // Each gift resolves as a persisted GIFT FrameLevel (next_sub is the loop cursor — free
        // until phase 4 resets it for the phase 5 sub chain) so a prompt inside a gift suspends.
        const Entity spell_src = ab.source.get();
        if (!ab.gift_abilities.empty() && spell_src != 0 &&
            global_coordinator.entity_has_component<Spell>(spell_src) &&
            global_coordinator.GetComponent<Spell>(spell_src).chosen.gift_promised) {
            for (; next_sub < static_cast<int>(ab.gift_abilities.size()); ++next_sub) {
                const Ability &gift_template = ab.gift_abilities[static_cast<size_t>(next_sub)];
                if (ctx.can_suspend()) {
                    Ability *self = &ab;
                    auto bind = [self](Ability &g) {
                        g.source = self->source;
                        g.controller = self->controller;
                    };
                    if (ctx.resolve_child(gift_template, FrameLevel::GIFT, next_sub, -1, bind,
                                          orderer) == ResolveStatus::SUSPENDED)
                        return ResolveStatus::SUSPENDED;
                } else {
                    Ability gift = gift_template;
                    gift.source = ab.source;
                    gift.controller = ab.controller;
                    resolve_ability(gift, orderer);
                }
            }
        }
        phase = 3;
    }
    if (phase == 3) {
        // Conditional execution: if condition fails, skip this ability's body but still chain subabilities
        bool condition_passed = ab.def->valid_tgts == "N_A" || !ctx.is_stack_object() || targets_still_legal(ab);
        if (condition_passed && !ab.def->condition_check_svar.empty()) condition_passed = condition_svar_passes(ab);
        // ConditionPresent$ / ConditionCompare$ gate (CR 608.2c): the "if ..." clause is checked
        // as the ability resolves. Covers the plain board-presence form (Edge of Autumn: "If you
        // control four or fewer lands"), ConditionDefined$ Remembered (Birthing Ritual: the dig
        // only happens if a creature was sacrificed), TriggeredCard (Amped Raptor: the card that
        // triggered this was cast from its controller's hand) and ExiledWith (The Creation of
        // Avacyn II & III: the card the Saga exiled is a creature). Like the SVar gate, failure
        // skips this body but still chains subabilities. An intervening-if was already checked in
        // phase 2, and a ConditionDefined$ Targeted condition is applied by the effect handler
        // to its target.
        if (condition_passed && !ab.def->condition_present.empty() && !ab.def->intervening_if && !ab.def->condition_on_target)
            condition_passed = evaluate_present_condition(ab, ab.controller, orderer);
        // Condition$ Blessing (Ocelot Pride's CopyPermanent): the body runs only if the
        // controller has the city's blessing (702.131). Failure still chains subabilities.
        if (condition_passed && ab.def->condition_city_blessing) {
            // 702.131b applies "any time", so re-latch the blessing before reading the
            // flag: a token created by an EARLIER sub-ability of this same resolution
            // (Ocelot Pride's first clause making the 10th permanent) must count — the
            // SBA-pass refresh alone would lag until after the whole trigger resolved
            // and this gate would read a stale false.
            if (orderer) refresh_city_blessing(orderer->mEntities);
            Entity pe = get_player_entity(ab.controller);
            condition_passed = global_coordinator.entity_has_component<Player>(pe) &&
                               global_coordinator.GetComponent<Player>(pe).has_city_blessing;
        }
        if (!condition_passed) {
            // Condition-failed body: skip the handler but still chain the
            // subabilities, then finish WITHOUT the phase-6 NameCard clear
            // (matching the pre-refactor early return). Routed to its own
            // phase (7) so a suspension inside a chained sub resumes there —
            // never re-evaluating the condition gates above, which could read
            // state an earlier sub mutated and diverge.
            next_sub = static_cast<int>(ab.def->repeat_sub_count);
            phase = 7;
        } else {
            phase = 4;
        }
    }
    if (phase == 4) {
        // Table-driven dispatch: every effect category resolves through its handler
        // in src/effects/. handler_for() returns nullptr only for kinds with no
        // resolve-time handler (None, EFFECT_KIND_ELSEWHERE) — those simply chain
        // subabilities.
        effects::EffectHandler handler = effects::handler_for(ab.def->kind);
        HandlerResult hres = handler ? handler(ab, orderer, ctx) : HandlerResult::DONE_RUN_SUBS;
        // A suspended handler parked its decision; propagate WITHOUT chaining subs
        // or clearing the named card — the re-entry finishes both.
        if (hres == HandlerResult::SUSPENDED) return ResolveStatus::SUSPENDED;
        // The effect is done: what it reported goes into the resolution's memory, once, before
        // anything chained after it can read it.
        report_effect(ab, EffectObjects::FINISHED);
        apply_memory_ops(ab);
        coverage_effect(ab.def->kind);
        // A repeating effect's body is its own to run; what is chained follows it.
        next_sub = static_cast<int>(ab.def->repeat_sub_count);
        // Chain subabilities unless the handler opted out (Charm/WinsGame and the
        // non-peek PeekAndReveal path return DONE_NO_SUBS to handle their own resolution).
        phase = (hres == HandlerResult::DONE_RUN_SUBS) ? 5 : 6;
    }
    if (phase == 5) {
        if (chain_subabilities(ab, orderer, ctx, next_sub) == ResolveStatus::SUSPENDED)
            return ResolveStatus::SUSPENDED;
        phase = 6;
    }
    if (phase == 7) {
        // Condition-failed sub chain (routed from phase 3): same loop, but no
        // phase-6 NameCard clear afterward — today's early-return behavior.
        if (chain_subabilities(ab, orderer, ctx, next_sub) == ResolveStatus::SUSPENDED)
            return ResolveStatus::SUSPENDED;
        return ResolveStatus::DONE;
    }
    // Phase 6: clear the named card once the whole spell/ability has finished resolving, so it
    // doesn't leak into an unrelated later Card.NamedCard check (CR 201.4 — the name is
    // chosen for this effect only). Only the top-level resolve clears it; sub-abilities
    // (ability_type SPELL parent vs. its DB$ children) are resolved within this call.
    if (ab.def->kind == EffectKind::NameCard)
        cur_game.resolution.memory.named_card.clear();
    return ResolveStatus::DONE;
}
