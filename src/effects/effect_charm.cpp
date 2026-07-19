#include "effects.h"

#include <string>
#include <vector>

#include "../action_processor.h"
#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../input_logger.h"
#include "../systems/orderer.h"

namespace effects {

// The bind applied once to each mode's persisted resolve_child copy (and the
// stamping the blocking path applies to the stored entry before its by-ref
// resolve). Forward-declared per CLAUDE.md.
static void stamp_mode(const Ability &parent, Ability &mode);

static void stamp_mode(const Ability &parent, Ability &mode) {
    mode.source = parent.source;
    mode.controller = parent.controller;
}

// Modal spell (CR 700.2): "Choose one/two —". The mode(s) and their targets were announced
// when the spell was CAST (CR 601.2b/c) — see announce_spell_targets — and recorded in
// charm_chosen. Resolution only replays those picks in order: each mode's own resolve()
// re-verifies its targets (CR 608.2b), so a mode whose targets became illegal fizzles
// individually without any prompting here. In a suspendable context each mode resolves as a
// persisted CHARM_MODE FrameLevel (a COPY of the stored charm_choices entry — the stored
// entry is never read again after its mode resolves, so the old by-reference resolution and
// the copy are indistinguishable), with CharmRt.announced_idx as the persisted loop cursor.
// The choose-at-resolution loop below remains as a FALLBACK for a charm that reached the
// stack without an announcement (a cast path not routed through announce_spell_targets).
HandlerResult charm(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    PendingDecisionScope pending_scope(ab.source);
    if (!ab.charm_chosen.empty()) {
        if (ctx.can_suspend()) {
            CharmRt &rt = ctx.rt<CharmRt>();
            for (; rt.announced_idx < static_cast<int>(ab.charm_chosen.size());
                 ++rt.announced_idx) {
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
                chosen.resolve(orderer);
            }
        }
        // Skip subabilities — charm handles its own resolution
        return HandlerResult::DONE_NO_SUBS;
    }

    // Resolution-time fallback: choose the mode(s) now. A staged machine over
    // CharmRt so every prompt — the mode pick, a chosen mode's target pick
    // (the shared run_target_select with a RESOLUTION asker), and any prompt
    // inside the mode's own resolution — can suspend and resume.
    CharmRt local_rt;
    CharmRt &rt = ctx.can_suspend() ? ctx.rt<CharmRt>() : local_rt;
    if (!rt.init) {
        game_log("(modes were not announced at cast — choosing at resolution)\n");
        rt.taken.assign(ab.charm_choices.size(), 0);  // CR 601.2b: a mode only once
        rt.pick = 0;
        rt.init = true;
    }
    const int to_pick = ab.charm_num < 1 ? 1 : ab.charm_num;

    while (rt.pick < to_pick) {
        if (rt.chosen_idx < 0) {
            // Arm-only log: a resume re-enters with the mode ask latched.
            if (!ctx.resuming()) game_log("Choose mode:\n");
            std::vector<LegalAction> mode_actions;
            std::vector<size_t> mode_indices;  // map action index -> charm_choices index
            for (size_t i = 0; i < ab.charm_choices.size(); i++) {
                if (rt.taken[i]) continue;
                Ability &candidate = ab.charm_choices[i];
                stamp_mode(ab, candidate);
                // Skip modes that require targets but have none available.
                if (candidate.valid_tgts != "N_A" && candidate.target_min > 0 &&
                    !has_legal_targets(candidate, orderer)) {
                    continue;
                }
                std::string desc =
                    (i < ab.charm_choice_descriptions.size() && !ab.charm_choice_descriptions[i].empty())
                        ? ab.charm_choice_descriptions[i]
                        : ("Mode " + std::to_string(i + 1));
                LegalAction la(PASS_PRIORITY, desc);
                la.category = ActionCategory::CHOOSE_MODE;
                la.option_ordinal = static_cast<int>(i);  // mode index (into charm_choices)
                mode_actions.push_back(la);
                mode_indices.push_back(i);
            }
            if (mode_actions.empty()) {
                // No further legal mode (all taken or none with legal targets). A modal
                // spell with too few legal modes simply resolves with what it could pick.
                if (rt.pick == 0) game_log("No valid modes — charm fizzles\n");
                break;
            }
            // Asked on the ambient seat (the resolving controller — a no-op
            // repoint, exactly the seat the old inline get_input read from),
            // under the handler's pending scope source.
            Zone::Ownership seat =
                cur_game.player_a_has_priority ? Zone::PLAYER_A : Zone::PLAYER_B;
            int choice = ctx.ask(mode_actions, seat, ab.source);
            if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
            rt.chosen_idx = static_cast<int>(mode_indices[static_cast<size_t>(choice)]);
            rt.taken[static_cast<size_t>(rt.chosen_idx)] = 1;
        }

        // Resolve the chosen mode: select its target(s) if any — on the STORED
        // charm_choices entry, matching the old by-reference resolve_chosen_mode
        // (the parent ability persists, so an in-flight selection survives a
        // suspension) — then resolve it (as a persisted CHARM_MODE child when
        // suspendable; the copy carries the just-chosen targets).
        Ability &chosen = ab.charm_choices[static_cast<size_t>(rt.chosen_idx)];
        stamp_mode(ab, chosen);
        if (!rt.targets_done && chosen.valid_tgts != "N_A") {
            ResolutionTargetAsker asker(ctx);
            if (run_target_select(chosen, rt.tsel, asker, orderer, ab.controller) ==
                TargetStatus::SUSPENDED)
                return HandlerResult::SUSPENDED;
        }
        rt.targets_done = true;
        if (ctx.can_suspend()) {
            Ability *parent = &ab;
            auto bind = [parent](Ability &mode) { stamp_mode(*parent, mode); };
            if (ctx.resolve_child(chosen, FrameLevel::CHARM_MODE, rt.chosen_idx, rt.pick, bind,
                                  orderer) == ResolveStatus::SUSPENDED)
                return HandlerResult::SUSPENDED;
        } else {
            chosen.resolve(orderer);
        }
        rt.chosen_idx = -1;
        rt.targets_done = false;
        rt.tsel = TargetSelectRT{};
        rt.pick++;
    }
    // Skip subabilities — charm handles its own resolution
    return HandlerResult::DONE_NO_SUBS;
}

}  // namespace effects
