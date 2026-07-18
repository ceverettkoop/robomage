#include "effects.h"

#include <string>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/color_identity.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../components/ability.h"
#include "../game_queries.h"
#include "../saga.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

bool target_color_condition_met(const Ability &ab, Entity target) {
    if (ab.condition_present.empty()) return true;
    const std::string &c = ab.condition_present;
    Colors required = NO_COLOR;
    if (c.find(".Red") != std::string::npos) required = RED;
    else if (c.find(".Blue") != std::string::npos) required = BLUE;
    else if (c.find(".Green") != std::string::npos) required = GREEN;
    else if (c.find(".White") != std::string::npos) required = WHITE;
    else if (c.find(".Black") != std::string::npos) required = BLACK;
    if (required == NO_COLOR) return true;  // non-color condition (e.g. cmcLEX) — not handled here
    if (!global_coordinator.entity_has_component<ColorIdentity>(target)) return false;
    return global_coordinator.GetComponent<ColorIdentity>(target).colors.count(required) > 0;
}

HandlerResult counter(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    if (global_coordinator.entity_has_component<Zone>(ab.target)) {
        auto &tz = global_coordinator.GetComponent<Zone>(ab.target);
        if (tz.location == Zone::STACK) {
            Zone::Ownership target_controller = global_coordinator.entity_has_component<Spell>(ab.target)
                                                    ? global_coordinator.GetComponent<Spell>(ab.target).caster
                                                    : tz.owner;

            bool do_counter = true;
            // Pyroblast/Hydroblast: only counter if the target is the required color.
            // The spell still resolves (and is put to the graveyard) doing nothing otherwise.
            if (!target_color_condition_met(ab, ab.target)) {
                std::string tname = entity_name(ab.target);
                game_log("%s is not the required color — not countered\n", tname.c_str());
                do_counter = false;
            }
            if (do_counter && ab.unless_generic_cost > 0) {
                std::string tname = entity_name(ab.target);
                // UnlessPayer$ (Reality Smasher: TriggeredSourceSAController) selects WHO pays —
                // the controller of the spell that targeted the source. When unset, default to the
                // countered spell's controller (Ward / Mana Leak / Daze).
                Zone::Ownership payer = (ab.unless_payer != Zone::UNKNOWN) ? ab.unless_payer
                                                                          : target_controller;
                UnlessPayKind kind = ab.unless_cost_is_discard ? UnlessPayKind::DISCARD
                                   : ab.unless_cost_is_life     ? UnlessPayKind::LIFE
                                                                : UnlessPayKind::MANA;
                // Arm-only log: a resume re-enters the suspended unless prompt
                // without re-announcing it.
                if (!ctx.resuming()) {
                    if (kind == UnlessPayKind::DISCARD)
                        game_log("%s may discard %zu card%s to save %s:\n", player_name(payer).c_str(),
                                 ab.unless_generic_cost, ab.unless_generic_cost == 1 ? "" : "s", tname.c_str());
                    else if (kind == UnlessPayKind::LIFE)
                        game_log("%s's controller may pay %zu life to save it:\n", tname.c_str(), ab.unless_generic_cost);
                    else
                        game_log("%s's controller may pay {%zu} to save it:\n", tname.c_str(), ab.unless_generic_cost);
                }
                bool suspended = false;
                do_counter = run_unless_loop(ab.unless_generic_cost, payer, orderer, ab.target,
                                             ctx, suspended, kind);
                if (suspended) return HandlerResult::SUSPENDED;
            }

            // Can't be countered check — either a cast-time stamp (Cavern of Souls / a "this spell
            // can't be countered" self replacement) or a continuous battlefield static covering the
            // spell (Hexing Squelcher: "Spells you control can't be countered", CR 614.13).
            if (do_counter &&
                ((global_coordinator.entity_has_component<Spell>(ab.target) &&
                  global_coordinator.GetComponent<Spell>(ab.target).cant_be_countered) ||
                 spell_uncounterable_by_static(ab.target, orderer->mEntities) ||
                 cur_game.cant_counter_spells_of.count(target_controller) > 0)) {
                std::string name = entity_name(ab.target);
                game_log("%s can't be countered\n", name.c_str());
                do_counter = false;
            }

            if (do_counter) {
                std::string name = entity_name(ab.target);
                bool is_standalone_ability = !global_coordinator.entity_has_component<Spell>(ab.target) &&
                                            global_coordinator.entity_has_component<Ability>(ab.target);
                // A COPY of a spell (CR 707.10c) is not a card: a countered copy ceases to exist
                // rather than going to a graveyard. Capture before the Spell component is removed.
                bool target_is_copy = global_coordinator.entity_has_component<Spell>(ab.target) &&
                                      global_coordinator.GetComponent<Spell>(ab.target).is_copy;
                // Capture flashback status before the Spell component (which carries it) is removed.
                bool was_flashback = spell_cast_with_flashback(ab.target);
                // CR 714.4: if the countered object is a Saga chapter ability, it is leaving the
                // stack WITHOUT resolving — release the same sacrifice gate the resolve path
                // releases, so a completed Saga isn't stranded on the battlefield forever. Read it
                // before the Ability component (which carries is_saga_chapter/source) is removed.
                if (global_coordinator.entity_has_component<Ability>(ab.target))
                    decrement_saga_in_flight(global_coordinator.GetComponent<Ability>(ab.target));
                if (global_coordinator.entity_has_component<Ability>(ab.target))
                    global_coordinator.RemoveComponent<Ability>(ab.target);
                if (global_coordinator.entity_has_component<Spell>(ab.target))
                    global_coordinator.RemoveComponent<Spell>(ab.target);
                if (is_standalone_ability || target_is_copy) {
                    // Standalone ability entities (activated/triggered) and copies of spells
                    // have no card to send to a zone — remove from stack and destroy
                    // (rule 701.5a / 707.10c).
                    orderer->add_to_zone(false, ab.target, Zone::EXILE);
                    global_coordinator.DestroyEntity(ab.target);
                } else {
                    // Exile if the counter spell says so (e.g. "counter, then exile")
                    // or if the countered spell was cast via flashback (it would be
                    // exiled when it left the stack anyway). Otherwise → graveyard.
                    bool to_exile = (ab.destination == Zone::EXILE) || was_flashback;
                    orderer->add_to_zone(false, ab.target, to_exile ? Zone::EXILE : Zone::GRAVEYARD);
                }
                game_log("%s is countered\n", name.c_str());
            }
        } else {
            // Target still exists but has left the stack (e.g. it was already countered by an
            // earlier counter that resolved first). Its only target is illegal, so the counter
            // does nothing and is put into its graveyard (CR 608.2b). The pre-resolve target check
            // (Ability::resolve / is_target_valid) normally fizzles this first; this is a defensive
            // clean fizzle for any path that reaches the effect with a stale target.
            game_log("Counter fizzles (target no longer on the stack)\n");
        }
    } else {
        game_log("Counter fizzles (target no longer on the stack)\n");
    }
    return HandlerResult::DONE_RUN_SUBS;
}

}  // namespace effects
