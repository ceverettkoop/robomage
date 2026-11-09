#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../components/ability.h"
#include "../queries/characteristics.h"
#include "../queries/filters.h"
#include "../queries/spells.h"
#include "../svar_eval.h"
#include "../systems/orderer.h"
#include "../systems/replacement_effects.h"
#include "../unless_payment.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool target_color_condition_met(const Ability &ab, Entity target) {
    if (ab.def->condition_present.empty()) return true;
    const std::string &c = ab.def->condition_present;
    Colors required = NO_COLOR;
    if (c.find(".Red") != std::string::npos) required = RED;
    else if (c.find(".Blue") != std::string::npos) required = BLUE;
    else if (c.find(".Green") != std::string::npos) required = GREEN;
    else if (c.find(".White") != std::string::npos) required = WHITE;
    else if (c.find(".Black") != std::string::npos) required = BLACK;
    if (required == NO_COLOR) return true;  // non-color condition (e.g. cmcLEX) — not handled here
    // The target's current color (CR 105.2 / 613.1e / 712.8e): a token's color indicator, a
    // transformed face's colors, or a SetColor override.
    return effective_colors(target).count(required) > 0;
}

HandlerResult counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    const Entity tgt = ab.target.get();
    if (global_coordinator.entity_has_component<Zone>(tgt)) {
        auto &tz = global_coordinator.GetComponent<Zone>(tgt);
        if (tz.location == Zone::STACK) {
            Zone::Ownership target_controller = global_coordinator.entity_has_component<Spell>(tgt)
                                                    ? global_coordinator.GetComponent<Spell>(tgt).caster
                                                    : tz.owner;

            bool do_counter = true;
            // Pyroblast/Hydroblast: only counter if the target is the required color.
            // The spell still resolves (and is put to the graveyard) doing nothing otherwise.
            if (!target_color_condition_met(ab, tgt)) {
                std::string tname = entity_name(tgt);
                game_log("%s is not the required color — not countered\n", tname.c_str());
                do_counter = false;
            }
            const size_t unless_cost = unless_amount(ab.def->unless_cost);
            if (do_counter && unless_cost > 0) {
                std::string tname = entity_name(tgt);
                // UnlessPayer$ (Reality Smasher: TriggeredSourceSAController) selects WHO pays —
                // the controller of the spell that targeted the source. When unset, default to the
                // countered spell's controller (Ward / Mana Leak / Daze).
                Zone::Ownership payer = (ab.unless_payer != Zone::UNKNOWN) ? ab.unless_payer
                                                                          : target_controller;
                UnlessPayKind kind = unless_pay_kind(ab.def->unless_cost);
                // Arm-only log: a resume re-enters the suspended unless prompt
                // without re-announcing it.
                if (!ctx.resuming()) {
                    if (kind == UnlessPayKind::DISCARD)
                        game_log("%s may discard %zu card%s to save %s:\n", player_name(payer).c_str(),
                                 unless_cost, unless_cost == 1 ? "" : "s", tname.c_str());
                    else if (kind == UnlessPayKind::LIFE)
                        game_log("%s's controller may pay %zu life to save it:\n", tname.c_str(), unless_cost);
                    else
                        game_log("%s's controller may pay {%zu} to save it:\n", tname.c_str(), unless_cost);
                }
                bool suspended = false;
                do_counter = run_unless_loop(unless_cost, payer, orderer, tgt, ab.source.lki_entity(), ctx,
                                             suspended, UnlessSubject{UnlessEffect::COUNTER, tgt, false},
                                             kind);
                if (suspended) return HandlerResult::SUSPENDED;
            }

            // Can't be countered (CR 614.13): the replacement dispatcher says what covers it.
            if (do_counter) {
                ReplacementEvent rev;
                rev.type = ReplacementEvent::COUNTER;
                rev.entity = tgt;
                rev.affected_player = target_controller;
                replacement::dispatch(rev);
                if (rev.prevented) {
                    std::string name = entity_name(tgt);
                    game_log("%s can't be countered\n", name.c_str());
                    do_counter = false;
                }
            }

            if (do_counter) {
                std::string name = entity_name(tgt);
                // A countered card goes to its owner's graveyard, or to exile when the counter
                // spell says so ("counter, then exile"); a copy or an ability ceases to exist.
                orderer->remove_from_stack(tgt, ab.def->destination == Zone::EXILE ? Zone::EXILE
                                                                                   : Zone::GRAVEYARD);
                game_log("%s is countered\n", name.c_str());
            }
        } else {
            // Target still exists but has left the stack (e.g. it was already countered by an
            // earlier counter that resolved first). Its only target is illegal, so the counter
            // does nothing and is put into its graveyard (CR 608.2b). The pre-resolve target check
            // (resolve_ability / targets_still_legal) normally fizzles this first; this is a defensive
            // clean fizzle for any path that reaches the effect with a stale target.
            game_log("Counter fizzles (target no longer on the stack)\n");
        }
    } else {
        game_log("Counter fizzles (target no longer on the stack)\n");
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
