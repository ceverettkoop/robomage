// General "copy a spell on the stack" mechanism (CR 707.10 / 707.12).
//
// A copy of a spell is a new object created ON THE STACK that copies the original spell's
// copiable characteristics (its CardData, color, and resolving ability). A copy is NOT cast:
// it pays no costs, is put directly onto the stack, and does not trigger "when you cast"
// abilities (so a copy with Replicate replicates nothing further). The controller of a copy
// MAY choose new targets for it (CR 707.12); here each copy re-runs the normal target
// selection. A copy is not a card — when it leaves the stack (resolves or is countered) it
// ceases to exist instead of moving to a zone (handled in the stack manager / effects::counter
// via Spell::is_copy).
//
// This routine is reusable by any copy-spell effect (Replicate at cast FINISH, Storm at
// resolution; fork would reuse it). It deliberately reuses the existing entity/component
// construction, the Spell/Ability components, and the shared run_target_select targeting path
// rather than open-coding a parallel copy or targeting UI.
//
// RESUMABLE (Batch 10): the whole loop is driven by a persisted CopySpellRT so a copy's
// target pick can suspend as a loop-top pending decision. The current copy's entity
// (CardData/ColorIdentity/Spell, deliberately no Zone until placement) persists in the ECS
// across the suspension — snapshot-covered — and its in-flight ability accumulates in
// rt.work. The no-legal-target path DESTROYS the copy entity mid-loop; that happens strictly
// before any ask for that copy, and rt.cur_copy is dropped in the same stride, so a parked
// pick always references a live copy and a resume re-validates nothing stale.

#include <memory>

#include "../action_processor.h"
#include "../cli_output.h"
#include "../components/ability.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../game_queries.h"
#include "../resolution_frame.h"
#include "../systems/orderer.h"
#include "effects.h"

extern Coordinator global_coordinator;

// Seed the copy machine. Leaves rt inactive — making run_copy_spell a no-op — when there is
// nothing to copy (count <= 0) or the original's copiable characteristics are gone entirely
// (no CardData), reproducing the old blocking routine's early-outs.
void copy_spell_begin(CopySpellRT &rt, Entity original, int count, Zone::Ownership controller) {
    rt = CopySpellRT{};
    if (count <= 0) return;
    if (!global_coordinator.entity_has_component<CardData>(original)) return;
    rt.active = true;
    rt.original = original;
    rt.remaining = count;
    rt.controller_is_a = (controller == Zone::PLAYER_A);
}

TargetStatus run_copy_spell(CopySpellRT &rt, TargetAsker &asker, std::shared_ptr<Orderer> orderer) {
    if (!rt.active) return TargetStatus::DONE;

    // Copiable characteristics: the card's printed values and color identity (CR 707.2),
    // re-derived from the original on every (re-)entry — pure reads, and nothing runs between
    // suspend and resume, so a resume re-derives them identically. These survive on the card
    // entity even after the spell has left the stack (a countered spell keeps its CardData/
    // ColorIdentity in the graveyard; only its Spell/Ability components are stripped).
    const auto &orig_card = global_coordinator.GetComponent<CardData>(rt.original);
    const ColorIdentity *orig_color =
        global_coordinator.entity_has_component<ColorIdentity>(rt.original)
            ? &global_coordinator.GetComponent<ColorIdentity>(rt.original)
            : nullptr;
    const Ability *orig_ability = nullptr;
    bool cant_be_countered = false;
    int x_paid = 0;
    if (global_coordinator.entity_has_component<Spell>(rt.original)) {
        // The original is still a live spell on the stack (Replicate / an uncountered Storm
        // spell): copy its resolving Ability and Spell flags directly.
        const Spell &orig_spell = global_coordinator.GetComponent<Spell>(rt.original);
        cant_be_countered = orig_spell.cant_be_countered;
        x_paid = orig_spell.x_paid;  // copies copy the X chosen for the original (CR 707.10b)
        if (global_coordinator.entity_has_component<Ability>(rt.original))
            orig_ability = &global_coordinator.GetComponent<Ability>(rt.original);
    } else {
        // The original has already left the stack (e.g. a Storm spell countered before its Storm
        // triggered ability resolved — CR 702.40a / 113.7a: the Storm ability is a separate object,
        // independent of the spell that created it, so countering that spell doesn't stop the
        // copies). Rebuild the copies from the spell's last-known copiable characteristics: the
        // card's SPELL ability template reproduces the resolving ability, and the copies choose
        // their own new targets anyway, so the template's (absent) targets don't matter.
        for (const auto &tmpl : orig_card.abilities)
            if (tmpl.ability_type == Ability::SPELL) { orig_ability = &tmpl; break; }
    }
    Zone::Ownership controller = rt.controller_is_a ? Zone::PLAYER_A : Zone::PLAYER_B;

    while (rt.remaining > 0) {
        if (rt.cur_copy == 0) {
            // Start a fresh copy: entity + copied components. The copy's Zone is added by
            // place_created_on_stack() at the end (CR 707.10: the copy is created on the
            // stack, not moved there from any zone). Until then it has no Zone, so it can't
            // appear as a target candidate during its own targeting below, and target
            // perspective falls back to the ability's controller.
            Entity copy = global_coordinator.CreateEntity();
            global_coordinator.AddComponent(copy, orig_card);
            if (orig_color) global_coordinator.AddComponent(copy, *orig_color);

            Spell copy_spell;
            copy_spell.caster = controller;
            copy_spell.cant_be_countered = cant_be_countered;
            copy_spell.x_paid = x_paid;
            copy_spell.is_copy = true;               // ceases to exist when it leaves the stack
            // A copy is not cast: it has no replicate count of its own and copies no further.
            global_coordinator.AddComponent(copy, copy_spell);

            rt.cur_copy = copy;
            rt.have_ability = (orig_ability != nullptr);
            rt.phase = 0;
            rt.sub_idx = 0;
            rt.mode_pos = 0;
            rt.tsel = TargetSelectRT{};
            if (orig_ability) {
                // Copy the resolving spell ability; the controller chooses new targets below
                // (CR 707.12), so the original's primary/sub targets are cleared up front.
                rt.work = *orig_ability;
                rt.work.source = copy;
                rt.work.controller = controller;
                rt.work.target = 0;
                rt.work.targets.clear();
                for (auto &sub : rt.work.subabilities) {
                    sub.source = copy;
                    sub.controller = controller;
                    sub.target = 0;
                    sub.targets.clear();
                }
            }
        }

        if (rt.have_ability) {
            if (rt.phase == 0) {
                if (rt.work.valid_tgts != "N_A") {
                    // If no legal target remains for this copy, it can't be put on the stack
                    // (CR 707.10c-ish: a copy that requires a target with none available is not
                    // created). Skip it cleanly rather than placing an untargeted copy. Checked
                    // only before the pick begins (tsel inactive) — a resumed pick already
                    // passed it, on state nothing has mutated since.
                    if (!rt.tsel.active && !has_legal_targets(rt.work, orderer)) {
                        global_coordinator.DestroyEntity(rt.cur_copy);
                        game_log("A copy of %s has no legal target — not created\n",
                                 orig_card.name.c_str());
                        rt.cur_copy = 0;
                        rt.remaining--;
                        continue;
                    }
                    if (run_target_select(rt.work, rt.tsel, asker, orderer, controller) !=
                        TargetStatus::DONE)
                        return TargetStatus::SUSPENDED;
                }
                rt.phase = 1;
            }
            if (rt.phase == 1) {
                while (rt.sub_idx < rt.work.subabilities.size()) {
                    // Guard subability targeting the same way the main ability is guarded
                    // above: selecting on an empty candidate pool (target required, none
                    // legal) would index an empty action list. A copy whose subability has no
                    // legal target simply leaves it untargeted (it does nothing at
                    // resolution) rather than crashing.
                    Ability &sub = rt.work.subabilities[rt.sub_idx];
                    if (sub.valid_tgts != "N_A" &&
                        (rt.tsel.active || has_legal_targets(sub, orderer))) {
                        if (run_target_select(sub, rt.tsel, asker, orderer, controller) !=
                            TargetStatus::DONE)
                            return TargetStatus::SUSPENDED;
                    }
                    rt.sub_idx++;
                }
                rt.phase = 2;
            }
            // Modal spell copy (CR 707.10c): the copy has the SAME chosen modes — they can't
            // be changed — but its controller may choose new targets for each chosen mode.
            // Re-select a chosen mode's targets when a legal candidate exists for this copy;
            // otherwise keep the original's targets (re-verified at resolution, CR 608.2b,
            // fizzling that mode naturally if they're illegal).
            while (rt.mode_pos < rt.work.charm_chosen.size()) {
                int idx = rt.work.charm_chosen[rt.mode_pos];
                if (idx >= 0 && static_cast<size_t>(idx) < rt.work.charm_choices.size()) {
                    Ability &mode = rt.work.charm_choices[static_cast<size_t>(idx)];
                    mode.source = rt.cur_copy;
                    mode.controller = controller;
                    if (mode.valid_tgts != "N_A" &&
                        (rt.tsel.active || has_legal_targets(mode, orderer))) {
                        if (!rt.tsel.active) {
                            mode.target = 0;
                            mode.targets.clear();
                        }
                        if (run_target_select(mode, rt.tsel, asker, orderer, controller) !=
                            TargetStatus::DONE)
                            return TargetStatus::SUSPENDED;
                    }
                }
                rt.mode_pos++;
            }
            global_coordinator.AddComponent(rt.cur_copy, rt.work);
        }

        // Bring the copy into existence on top of the stack (above the original, so it resolves
        // first) without firing a zone-change event/replacement it never earned.
        orderer->place_created_on_stack(rt.cur_copy, controller);
        game_log("%s copies %s\n", player_name(controller).c_str(), orig_card.name.c_str());
        rt.cur_copy = 0;
        rt.remaining--;
    }
    rt = CopySpellRT{};
    return TargetStatus::DONE;
}

// DB$ CopySpellAbility (Chain Lightning): after the parent spell's effect resolves, a player MAY
// pay an unless-cost ({R}{R}) to copy the parent spell and choose new targets for the copy.
// UnlessSwitched$ True means paying ENABLES the copy (copy only if paid), the inverse of the usual
// "do it unless paid". The payer/copier is UnlessPayer$ TargetedOrController — the targeted PLAYER,
// or the targeted permanent's controller. General over any "then a player may pay X to copy this
// spell (and may choose new targets)". CR 707.10 / 707.12.
//
// Two suspendable stages share one persisted CopySpellRT: (1) the unless-cost payment loop runs
// while rt is INACTIVE (a resume re-enters the idempotent Shape-C loop and consumes the latched
// answer — the copy hasn't begun); (2) copy_spell_begin() activates rt, after which a resume goes
// straight to run_copy_spell (the pay question is never re-asked).
HandlerResult effects::copy_spell_ability(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    CopySpellRT local_rt;
    CopySpellRT &rt = ctx.can_suspend() ? ctx.rt<CopySpellRT>() : local_rt;

    if (!rt.active) {
        // Determine the payer/copier. UnlessPayer$ TargetedOrController: the targeted player, or —
        // when the parent targeted a permanent — that permanent's controller. The target is the
        // parent effect's target, inherited onto this sub-ability (Defined$ Parent) by bind_sub_target.
        Zone::Ownership payer = ab.controller;
        if (ab.unless_payer_is_targeted_or_controller) {
            Entity tgt = ab.target;
            if (tgt != 0 && global_coordinator.entity_has_component<Player>(tgt)) {
                payer = (tgt == get_player_entity(Zone::PLAYER_A)) ? Zone::PLAYER_A : Zone::PLAYER_B;
            } else if (tgt != 0 && global_coordinator.entity_has_component<Permanent>(tgt)) {
                payer = global_coordinator.GetComponent<Permanent>(tgt).controller;
            }
        } else if (ab.unless_payer != Zone::UNKNOWN) {
            payer = ab.unless_payer;
        }

        // Offer the optional unless-cost. UnlessSwitched inverts the meaning of "paid".
        bool do_copy;
        if (ab.unless_generic_cost > 0) {
            if (!ctx.resuming())
                game_log("%s may pay to copy %s:\n", player_name(payer).c_str(),
                         entity_name(ab.source).c_str());
            bool suspended = false;
            bool prevented = run_unless_loop(ab.unless_generic_cost, payer, orderer, ab.source, ctx,
                                             suspended, UnlessPayKind::MANA, &ab.unless_cost_pips);
            if (suspended) return HandlerResult::SUSPENDED;
            bool paid = !prevented;
            // UnlessSwitched$ True: copy only if paid. Otherwise copy unless paid.
            do_copy = ab.unless_switched ? paid : prevented;
        } else {
            do_copy = true;  // no cost — always copy
        }
        if (!do_copy) return HandlerResult::DONE_RUN_SUBS;

        copy_spell_begin(rt, ab.source, 1, payer);
        if (!rt.active) return HandlerResult::DONE_RUN_SUBS;  // nothing to copy (original gone)
    }

    ResolutionTargetAsker asker(ctx);
    if (run_copy_spell(rt, asker, orderer) != TargetStatus::DONE)
        return HandlerResult::SUSPENDED;
    return HandlerResult::DONE_RUN_SUBS;
}
