#include "announce.h"

#include <algorithm>
#include <string>
#include <vector>

#include "action_processor.h"
#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "coverage.h"
#include "systems/state_manager_internal.h"
#include "targeting.h"

// The display label for mode `idx` of `modal`: its script description, else a positional fallback.
static std::string mode_desc(const Ability &modal, size_t idx);

// Record the pick of printed mode `idx` in modal.charm_chosen and return the mode instance its
// targets are chosen on: the printed entry itself, or under CanRepeatModes$ a fresh copy of it
// appended to charm_choices, so each pick of a repeated mode is its own instance (CR 700.2d).
static Ability &record_mode_pick(Ability &modal, size_t idx);

// Put modal.charm_chosen in the order the modes resolve: printed order, the instances of a mode
// chosen more than once in sequence in the order picked (CR 608.2c, 700.2d).
static void order_chosen_modes(Ability &modal);

// One mode pick's menu (CR 700.2a/b): the printed modes that can be chosen now, less those
// already chosen (a mode is chosen at most once, CR 700.2d, unless the ability may choose the
// same mode more than once, CanRepeatModes$), with `mode_indices` mapping action index ->
// printed mode index. Pure ECS reads, so a resumed pick re-derives the identical menu. Stamps
// each mode with the modal object's source and the chooser, which its target pick then reads.
static std::vector<LegalAction> build_mode_menu(Ability &modal, std::shared_ptr<Orderer> orderer,
                                                Zone::Ownership chooser,
                                                std::vector<size_t> &mode_indices);

// The row that ends a "choose one or more" mode choice (CR 700.2) once the fewest modes are chosen.
static LegalAction confirm_modes_action(const Ability &modal, int chosen);

// Does `ab` still have its own targets to choose — it targets, and none was bound to it (a
// trigger's target can come from its event)?
static bool targets_unchosen(const Ability &ab);

// The target chosen last before chained instruction `subs[idx]` of `ab`.
static ObjectRef target_before(const Ability &ab, const std::vector<Ability *> &subs, size_t idx);

// The targets `ab` chose, in the order of its target_shares.
static std::vector<ObjectRef> chosen_target_list(const Ability &ab);

// Open the division of what `ab` divides among its chosen targets (CR 601.2d, 603.3d): each holds
// one, and with a single target it holds all. True when points are left to place.
static bool begin_division(Ability &ab, AnnounceRT &rt, Zone::Ownership controller);

// The menu of one point: a row per target (CR 601.2d), so its length is the number of targets
// whatever the total. Each row's ordinal is what its target holds so far.
static std::vector<LegalAction> division_menu(const Ability &ab);

// Place the points left, one decision each; a resumed pick re-derives the identical menu.
static AnnounceStatus announce_division(Ability &ab, AnnounceRT &rt, TargetAsker &asker);

// The target pick of `ab` for the current announce stage, and the division of what it divides. With
// `remove_unchoosable`, a required target with no legal choice is checked once, before the pick
// begins, and reported as REMOVED. `target_gate` is run_target_select's.
static AnnounceStatus announce_targets(Ability &ab, AnnounceRT &rt, TargetAsker &asker,
    std::shared_ptr<Orderer> orderer, Zone::Ownership controller, bool remove_unchoosable,
    const TargetPickGate *target_gate);

static std::string mode_desc(const Ability &modal, size_t idx) {
    return (idx < modal.def->charm_choice_descriptions.size() &&
            !modal.def->charm_choice_descriptions[idx].empty())
               ? modal.def->charm_choice_descriptions[idx]
               : ("Mode " + std::to_string(idx + 1));
}

static Ability &record_mode_pick(Ability &modal, size_t idx) {
    if (!modal.def->charm_repeat_modes) {
        modal.charm_chosen.push_back(static_cast<int>(idx));
        return modal.charm_choices[idx];
    }
    Ability instance = modal.charm_choices[idx];
    modal.charm_choices.push_back(std::move(instance));
    modal.charm_chosen.push_back(static_cast<int>(modal.charm_choices.size() - 1));
    return modal.charm_choices.back();
}

static void order_chosen_modes(Ability &modal) {
    std::stable_sort(modal.charm_chosen.begin(), modal.charm_chosen.end(), [&modal](int a, int b) {
        return modal.charm_printed_mode(static_cast<size_t>(a)) <
               modal.charm_printed_mode(static_cast<size_t>(b));
    });
}

static std::vector<LegalAction> build_mode_menu(Ability &modal, std::shared_ptr<Orderer> orderer,
                                                Zone::Ownership chooser,
                                                std::vector<size_t> &mode_indices) {
    std::vector<LegalAction> mode_actions;
    mode_indices.clear();
    for (size_t i = 0; i < modal.charm_printed_count(); i++) {
        if (!modal.def->charm_repeat_modes &&
            std::find(modal.charm_chosen.begin(), modal.charm_chosen.end(), static_cast<int>(i)) !=
                modal.charm_chosen.end())
            continue;
        Ability &candidate = modal.charm_choices[i];
        candidate.source = modal.source;
        candidate.controller = chooser;
        if (!mode_choosable(modal, i, orderer, chooser, true)) continue;
        // Ground every mode to the modal object's source card so the serialized action
        // carries that card's id/zone/controller instead of the null-source
        // sentinel — otherwise each mode emits card_id -1 and the modes differ
        // only by the raw option_ordinal scalar, which reads as "all modes
        // identical" to the policy/search. The distinct option_ordinal still
        // separates the modes from one another.
        LegalAction la(PASS_PRIORITY, modal.source.lki_entity(), mode_desc(modal, i));
        la.category = ActionCategory::CHOOSE_MODE;
        la.option_ordinal = static_cast<int>(i);  // mode index (into charm_choices)
        mode_actions.push_back(la);
        mode_indices.push_back(i);
    }
    return mode_actions;
}

static LegalAction confirm_modes_action(const Ability &modal, int chosen) {
    LegalAction la(
        PASS_PRIORITY, modal.source.lki_entity(), "Done choosing modes (" + std::to_string(chosen) + " chosen)");
    la.category = ActionCategory::CONFIRM_MODES;
    la.option_ordinal = chosen;  // how many modes are chosen
    return la;
}

static bool targets_unchosen(const Ability &ab) {
    return ab.def->valid_tgts != "N_A" && ab.target.empty() && ab.targets.empty();
}

static AnnounceStatus announce_targets(Ability &ab, AnnounceRT &rt, TargetAsker &asker,
    std::shared_ptr<Orderer> orderer, Zone::Ownership controller, bool remove_unchoosable,
    const TargetPickGate *target_gate) {
    if (!rt.pick_in_flight) {
        if (remove_unchoosable && !has_legal_targets(ab, orderer)) return AnnounceStatus::REMOVED;
        rt.pick_in_flight = true;
        rt.tsel = TargetSelectRT{};
    }
    if (!rt.dividing) {
        if (run_target_select(ab, rt.tsel, asker, orderer, controller, target_gate) == TargetStatus::SUSPENDED)
            return AnnounceStatus::SUSPENDED;
        rt.dividing = begin_division(ab, rt, controller);
    }
    if (rt.dividing && announce_division(ab, rt, asker) == AnnounceStatus::SUSPENDED)
        return AnnounceStatus::SUSPENDED;
    rt.dividing = false;
    rt.pick_in_flight = false;
    return AnnounceStatus::DONE;
}

static ObjectRef target_before(const Ability &ab, const std::vector<Ability *> &subs, size_t idx) {
    while (idx-- > 0)
        if (!subs[idx]->target.empty()) return subs[idx]->target;
    return ab.target;
}

static std::vector<ObjectRef> chosen_target_list(const Ability &ab) {
    if (!ab.targets.empty()) return ab.targets;
    if (!ab.target.empty()) return {ab.target};
    return {};
}

static bool begin_division(Ability &ab, AnnounceRT &rt, Zone::Ownership controller) {
    if (!divides_among_targets(*ab.def)) return false;
    const size_t chosen = chosen_target_list(ab).size();
    if (chosen == 0) return false;
    const int total = division_total(ab, controller);
    coverage_row(chosen == 1 ? "divide_one_target" : "divide_among_targets");
    ab.target_shares.assign(chosen, 1);
    if (chosen == 1) ab.target_shares[0] = total;
    rt.divide_left = chosen == 1 ? 0 : total - static_cast<int>(chosen);
    return rt.divide_left > 0;
}

static std::vector<LegalAction> division_menu(const Ability &ab) {
    const std::vector<ObjectRef> chosen = chosen_target_list(ab);
    std::vector<LegalAction> menu;
    for (size_t i = 0; i < chosen.size(); i++) {
        const Entity target = chosen[i].get();
        const int share = ab.target_shares[i];
        LegalAction la(PASS_PRIORITY, target,
            target_display_name(cur_game, target) + ": one more damage (" + std::to_string(share) + " so far)");
        la.category = ActionCategory::DIVIDE_DAMAGE;
        la.option_ordinal = share;
        menu.push_back(la);
    }
    return menu;
}

static AnnounceStatus announce_division(Ability &ab, AnnounceRT &rt, TargetAsker &asker) {
    while (rt.divide_left > 0) {
        if (!asker.resuming()) game_log("Divide the rest (%d):\n", rt.divide_left);
        const int choice = asker.ask(division_menu(ab), ab.source.lki_entity());
        if (choice < 0 && decision_suspended()) return AnnounceStatus::SUSPENDED;
        ab.target_shares[static_cast<size_t>(choice)]++;
        rt.divide_left--;
    }
    return AnnounceStatus::DONE;
}

AnnounceStatus run_announce(Ability &ab, AnnounceRT &rt, TargetAsker &asker, std::shared_ptr<Orderer> orderer,
    Zone::Ownership controller, bool remove_unchoosable, const ModePickGate *mode_gate,
    const TargetPickGate *target_gate) {
    if (!rt.active) {
        rt = AnnounceRT{};
        rt.active = true;
        rt.stage = is_modal(ab) ? AnnounceRT::MODES : AnnounceRT::PRIMARY;
    }
    for (;;) switch (rt.stage) {
        case AnnounceRT::MODES: {
            // CR 700.2a/b: one mode pick per pass; the just-picked mode's targets are chosen
            // (MODE_TARGETS, CR 601.2c) before the next pick.
            // "Choose one or more" (MinCharmNum$): past its fewest modes the choice may end.
            const int to_pick = charm_max_picks(*ab.def);
            const int min_picks = charm_min_picks(*ab.def);
            std::vector<LegalAction> menu;
            std::vector<size_t> mode_indices;
            if (rt.mode_picks < to_pick && (rt.mode_picks == 0 || !mode_gate || (*mode_gate)(rt.mode_picks))) {
                menu = build_mode_menu(ab, orderer, controller, mode_indices);
                if (!menu.empty() && !asker.resuming()) game_log("Choose mode:\n");
                if (menu.empty()) {
                    // CR 603.3c: a triggered ability none of whose modes can be chosen is
                    // removed. A spell or activated ability was gated on its required modes
                    // being choosable, so this is reachable for them only once an earlier
                    // pick's targets changed what is legal: keep the modes chosen so far.
                    if (rt.mode_picks == 0 && remove_unchoosable) {
                        rt = AnnounceRT{};
                        return AnnounceStatus::REMOVED;
                    }
                    game_log("No further legal mode — %d chosen\n", rt.mode_picks);
                }
            }
            const bool may_stop = !menu.empty() && rt.mode_picks >= min_picks;
            if (may_stop) menu.push_back(confirm_modes_action(ab, rt.mode_picks));
            int choice = -1;
            if (!menu.empty()) {
                choice = asker.ask(menu, ab.source.lki_entity());
                if (choice < 0 && decision_suspended()) return AnnounceStatus::SUSPENDED;
                if (may_stop && static_cast<size_t>(choice) == mode_indices.size()) {
                    game_log(
                        "%s chooses no further mode — %d chosen\n", player_name(controller).c_str(), rt.mode_picks);
                    menu.clear();
                }
            }
            if (menu.empty()) {
                // The modes resolve in the order they are printed (CR 608.2c, 700.2d).
                order_chosen_modes(ab);
                rt.stage = AnnounceRT::PRIMARY;
                break;
            }
            const size_t idx = mode_indices[static_cast<size_t>(choice)];
            const Ability &picked = record_mode_pick(ab, idx);
            game_log("%s chooses mode — %s\n", player_name(controller).c_str(),
                     mode_desc(ab, idx).c_str());
            if (picked.def->valid_tgts != "N_A")
                rt.stage = AnnounceRT::MODE_TARGETS;
            else
                rt.mode_picks++;
            break;
        }

        case AnnounceRT::MODE_TARGETS: {
            Ability &mode = ab.charm_choices[static_cast<size_t>(ab.charm_chosen.back())];
            // Its choosability was just checked by the menu, so no removal check here.
            if (announce_targets(mode, rt, asker, orderer, controller, false, target_gate) == AnnounceStatus::SUSPENDED)
                return AnnounceStatus::SUSPENDED;
            rt.mode_picks++;
            rt.stage = AnnounceRT::MODES;
            break;
        }

        case AnnounceRT::PRIMARY: {
            if (rt.pick_in_flight || targets_unchosen(ab)) {
                AnnounceStatus st =
                    announce_targets(ab, rt, asker, orderer, controller, remove_unchoosable, target_gate);
                if (st == AnnounceStatus::SUSPENDED) return st;
                if (st == AnnounceStatus::REMOVED) {
                    rt = AnnounceRT{};
                    return st;
                }
            }
            rt.sub_idx = 0;
            rt.stage = AnnounceRT::SUBS;
            break;
        }

        case AnnounceRT::SUBS: {
            // A chained sub-ability that targets chooses its target now too (CR 601.2c): Cabal
            // Therapy's DB$ Discard, Cloak and Dagger's "up to one target creature", Galvanic
            // Discharge's DB$ DealDamage two links down, with a "ParentTarget" player bound from
            // the ability's chosen player target first. An Execute$ body (a reflexive or delayed
            // trigger's effect) belongs to the ability that triggers later and chooses its targets
            // then (CR 603.12, 603.7).
            const std::vector<Ability *> subs = ab.chained_subs();
            for (; rt.sub_idx < subs.size(); ++rt.sub_idx) {
                Ability &sub = *subs[rt.sub_idx];
                if (!rt.pick_in_flight) {
                    if (!targets_unchosen(sub)) continue;
                    sub.source = ab.source;
                    sub.controller = controller;
                    sub.targeted_player = ab.player_target_for_subs();
                    sub.parent_target = target_before(ab, subs, rt.sub_idx);
                }
                AnnounceStatus st =
                    announce_targets(sub, rt, asker, orderer, controller, remove_unchoosable, target_gate);
                if (st == AnnounceStatus::SUSPENDED) return st;
                if (st == AnnounceStatus::REMOVED) {
                    rt = AnnounceRT{};
                    return st;
                }
            }
            rt = AnnounceRT{};
            return AnnounceStatus::DONE;
        }
    }
}
