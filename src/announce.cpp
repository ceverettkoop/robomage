#include "announce.h"

#include <algorithm>
#include <string>
#include <vector>

#include "action_processor.h"
#include "classes/action.h"
#include "classes/game.h"
#include "cli_output.h"
#include "targeting.h"

// The display label for mode `idx` of `modal`: its script description, else a positional fallback.
static std::string mode_desc(const Ability &modal, size_t idx);

// One mode pick's menu (CR 700.2a/b): the not-yet-chosen modes that can be chosen now (a mode
// is chosen at most once, CR 700.2d), with `mode_indices` mapping action index -> charm_choices
// index. Pure ECS reads, so a resumed pick re-derives the identical menu. Stamps each mode with
// the modal object's source and the chooser, which its target pick then reads.
static std::vector<LegalAction> build_mode_menu(Ability &modal, std::shared_ptr<Orderer> orderer,
                                                Zone::Ownership chooser,
                                                std::vector<size_t> &mode_indices);

// Does `ab` still have its own targets to choose — it targets, and none was bound to it (a
// trigger's target can come from its event)?
static bool targets_unchosen(const Ability &ab);

// The target pick of `ab` for the current announce stage. With `remove_unchoosable`, a required
// target with no legal choice is checked once, before the pick begins, and reported as REMOVED.
static AnnounceStatus announce_targets(Ability &ab, AnnounceRT &rt, TargetAsker &asker,
                                       std::shared_ptr<Orderer> orderer,
                                       Zone::Ownership controller, bool remove_unchoosable);

static std::string mode_desc(const Ability &modal, size_t idx) {
    return (idx < modal.def->charm_choice_descriptions.size() &&
            !modal.def->charm_choice_descriptions[idx].empty())
               ? modal.def->charm_choice_descriptions[idx]
               : ("Mode " + std::to_string(idx + 1));
}

static std::vector<LegalAction> build_mode_menu(Ability &modal, std::shared_ptr<Orderer> orderer,
                                                Zone::Ownership chooser,
                                                std::vector<size_t> &mode_indices) {
    std::vector<LegalAction> mode_actions;
    mode_indices.clear();
    for (size_t i = 0; i < modal.charm_choices.size(); i++) {
        if (std::find(modal.charm_chosen.begin(), modal.charm_chosen.end(), static_cast<int>(i)) !=
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

static bool targets_unchosen(const Ability &ab) {
    return ab.def->valid_tgts != "N_A" && ab.target.empty() && ab.targets.empty();
}

static AnnounceStatus announce_targets(Ability &ab, AnnounceRT &rt, TargetAsker &asker,
                                       std::shared_ptr<Orderer> orderer,
                                       Zone::Ownership controller, bool remove_unchoosable) {
    if (!rt.pick_in_flight) {
        if (remove_unchoosable && !has_legal_targets(ab, orderer)) return AnnounceStatus::REMOVED;
        rt.pick_in_flight = true;
        rt.tsel = TargetSelectRT{};
    }
    if (run_target_select(ab, rt.tsel, asker, orderer, controller) == TargetStatus::SUSPENDED)
        return AnnounceStatus::SUSPENDED;
    rt.pick_in_flight = false;
    return AnnounceStatus::DONE;
}

AnnounceStatus run_announce(Ability &ab, AnnounceRT &rt, TargetAsker &asker,
                            std::shared_ptr<Orderer> orderer, Zone::Ownership controller,
                            bool remove_unchoosable) {
    if (!rt.active) {
        rt = AnnounceRT{};
        rt.active = true;
        rt.stage = is_modal(ab) ? AnnounceRT::MODES : AnnounceRT::PRIMARY;
    }
    for (;;) switch (rt.stage) {
        case AnnounceRT::MODES: {
            // CR 700.2a/b: one mode pick per pass; the just-picked mode's targets are chosen
            // (MODE_TARGETS, CR 601.2c) before the next pick.
            const int to_pick = ab.def->charm_num < 1 ? 1 : ab.def->charm_num;
            std::vector<LegalAction> menu;
            std::vector<size_t> mode_indices;
            if (rt.mode_picks < to_pick) {
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
            if (menu.empty()) {
                // The modes resolve in the order they are printed (CR 608.2c, 700.2d).
                std::sort(ab.charm_chosen.begin(), ab.charm_chosen.end());
                rt.stage = AnnounceRT::PRIMARY;
                break;
            }
            int choice = asker.ask(menu, ab.source.lki_entity());
            if (choice < 0 && decision_suspended()) return AnnounceStatus::SUSPENDED;
            const size_t idx = mode_indices[static_cast<size_t>(choice)];
            ab.charm_chosen.push_back(static_cast<int>(idx));
            game_log("%s chooses mode — %s\n", player_name(controller).c_str(),
                     mode_desc(ab, idx).c_str());
            if (ab.charm_choices[idx].def->valid_tgts != "N_A")
                rt.stage = AnnounceRT::MODE_TARGETS;
            else
                rt.mode_picks++;
            break;
        }

        case AnnounceRT::MODE_TARGETS: {
            Ability &mode = ab.charm_choices[static_cast<size_t>(ab.charm_chosen.back())];
            // Its choosability was just checked by the menu, so no removal check here.
            if (announce_targets(mode, rt, asker, orderer, controller, false) ==
                AnnounceStatus::SUSPENDED)
                return AnnounceStatus::SUSPENDED;
            rt.mode_picks++;
            rt.stage = AnnounceRT::MODES;
            break;
        }

        case AnnounceRT::PRIMARY: {
            if (rt.pick_in_flight || targets_unchosen(ab)) {
                AnnounceStatus st =
                    announce_targets(ab, rt, asker, orderer, controller, remove_unchoosable);
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
            // Therapy's DB$ Discard, Cloak and Dagger's "up to one target creature", with a
            // "ParentTarget" player bound from the ability's chosen player target first. An
            // Execute$ body (a reflexive or delayed trigger's effect) belongs to the ability
            // that triggers later and chooses its targets then (CR 603.12, 603.7).
            for (; rt.sub_idx < ab.subabilities.size(); ++rt.sub_idx) {
                Ability &sub = ab.subabilities[rt.sub_idx];
                if (!rt.pick_in_flight) {
                    if (sub.def->from_delayed_execute || !targets_unchosen(sub)) continue;
                    sub.source = ab.source;
                    sub.controller = controller;
                    sub.targeted_player = ab.player_target_for_subs();
                }
                AnnounceStatus st =
                    announce_targets(sub, rt, asker, orderer, controller, remove_unchoosable);
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
