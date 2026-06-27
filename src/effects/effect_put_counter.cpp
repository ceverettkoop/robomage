#include "effects.h"

#include <cctype>

#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/carddata.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/player.h"
#include "../ecs/coordinator.h"
#include "../ecs/events.h"
#include "../game_queries.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// CR 701.37: a Monstrosity$ ability that resolves while its source isn't yet monstrous puts its
// +1/+1 counters on the source, sets the monstrous designation, and fires BECAME_MONSTROUS (which
// puts any "becomes monstrous" triggers on the stack). If the source is already monstrous (701.37a
// — the activation gate normally prevents this), the whole ability does nothing. Returns true when
// the counter placement should proceed, false when it must be skipped (already monstrous).
static bool apply_monstrosity(Ability &ab) {
    if (!global_coordinator.entity_has_component<Permanent>(ab.source)) return false;
    auto &perm = global_coordinator.GetComponent<Permanent>(ab.source);
    if (perm.is_monstrous) return false;  // 701.37a: monstrosity does nothing if already monstrous
    perm.is_monstrous = true;
    game_log("%s becomes monstrous.\n", perm.name.c_str());
    Entity ctrl_entity =
        (perm.controller == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
    Event ev(Events::BECAME_MONSTROUS);
    ev.SetParam(Params::ENTITY, ab.source);
    ev.SetParam(Params::PLAYER, ctrl_entity);
    global_coordinator.SendEvent(ev);
    return true;
}

bool put_counter(Ability &ab, std::shared_ptr<Orderer> orderer) {
    // Monstrosity$ (CR 701.37): set the monstrous designation + fire the event first, then fall
    // through to place the N +1/+1 counters on the source via the normal counter path below. If
    // already monstrous, the ability does nothing at all.
    if (ab.is_monstrosity && !apply_monstrosity(ab)) return true;
    // Defined$ You — the counters go on the controlling PLAYER, not a permanent (CR 122.1c:
    // a player can have counters too, e.g. energy {E}, poison, experience). Guide of Souls'
    // "get {E}" puts an ENERGY counter on the source's controller.
    if (ab.defined_you) {
        const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
        if (!cp || cp->type.empty()) return true;
        // A dynamic CounterNum$ (Wrath of the Skies: CounterNum$ X, X = Count$xPaid → the
        // player gets X {E}) is evaluated at resolution; otherwise use the static count.
        int n = cp->count;
        if (!cp->count_expr.empty())
            n = static_cast<int>(evaluate_dynamic_amount(cp->count_expr, ab.controller, orderer, ab.target));
        if (n <= 0) return true;
        Entity ctrl_entity =
            (ab.controller == Zone::PLAYER_A) ? cur_game.player_a_entity : cur_game.player_b_entity;
        auto &pl = global_coordinator.GetComponent<Player>(ctrl_entity);
        int total = pl.add_counters(cp->type, n);
        game_log("%s gets %d %s counter(s) (now %d).\n", player_name(ab.controller).c_str(),
                 n, cp->type.c_str(), total);
        return true;
    }
    (void)orderer;
    // Use target if set (e.g. from a Pump parent), otherwise put counters on source
    // (Defined$ Self — e.g. Aether Vial's upkeep "put a charge counter on it"). Counters
    // can go on any permanent, not just creatures (CR 122.1), so gate on Permanent: a
    // creature gets +1/+1-style P/T resync via add_counters, a non-creature (Aether Vial,
    // an artifact) just accrues the typed counter in its counter map.
    Entity counter_tgt =
        (ab.target != 0 && global_coordinator.entity_has_component<Permanent>(ab.target)) ? ab.target : ab.source;
    if (!global_coordinator.entity_has_component<Permanent>(counter_tgt)) return true;
    const CounterParams *cp = std::get_if<CounterParams>(&ab.params);
    if (cp && !cp->type.empty()) {
        // A dynamic CounterNum$ (count_expr, e.g. CounterNum$ X = Count$xPaid) is evaluated at
        // resolution; otherwise the static count is used. Mirrors the Defined$ You / PutCounterAll
        // paths so a targeted counter with a runtime count no longer places zero.
        int n = cp->count;
        if (!cp->count_expr.empty())
            n = static_cast<int>(evaluate_dynamic_amount(cp->count_expr, ab.controller, orderer, ab.target));
        if (n > 0) {
            int total = add_counters(counter_tgt, cp->type, n);
            if (global_coordinator.entity_has_component<Creature>(counter_tgt)) {
                auto &cr = global_coordinator.GetComponent<Creature>(counter_tgt);
                if (cp->type == "P1P1")
                    game_log("Put %d +1/+1 counter(s) on creature (now %u/%u).\n", n, cr.power, cr.toughness);
                else
                    game_log("Put %d %s counter(s) on creature (now %u/%u).\n", n, cp->type.c_str(), cr.power, cr.toughness);
            } else {
                const char *nm = global_coordinator.entity_has_component<CardData>(counter_tgt)
                                     ? global_coordinator.GetComponent<CardData>(counter_tgt).name.c_str()
                                     : "permanent";
                game_log("Put %d %s counter(s) on %s (now %d).\n", n, cp->type.c_str(), nm, total);
            }
        }
        // Optional second counter kind (CounterType2$/CounterNum2$), as in PutCounterAll.
        if (!cp->type2.empty() && cp->count2 > 0) {
            add_counters(counter_tgt, cp->type2, cp->count2);
            const char *nm = global_coordinator.entity_has_component<CardData>(counter_tgt)
                                 ? global_coordinator.GetComponent<CardData>(counter_tgt).name.c_str()
                                 : "permanent";
            game_log("Put %d %s counter(s) on %s.\n", cp->count2, cp->type2.c_str(), nm);
        }
    }
    return true;
}

bool parse_put_counter(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "CounterType") {
        auto &cp = effect_params<CounterParams>(ab);
        cp.type = value;
        // Forge defaults CounterNum$ to 1 when omitted (e.g. Kappa Cannoneer's
        // "put a +1/+1 counter"). A later explicit CounterNum$ overrides this.
        if (cp.count == 0) cp.count = 1;
        return true;
    }
    if (key == "CounterNum")  {
        // A numeric CounterNum$ is used directly. A non-numeric value is an SVar key (Wrath
        // of the Skies: CounterNum$ X, X = Count$xPaid) — stash the raw token; parse_abilities
        // resolves it through the SVar map into a runtime Count$ expression on count_expr.
        auto &cp = effect_params<CounterParams>(ab);
        if (!value.empty() && (std::isdigit(static_cast<unsigned char>(value[0])) || value[0] == '-'))
            cp.count = std::stoi(value);
        else
            cp.count_expr = value;
        return true;
    }
    if (key == "CounterType2") {
        auto &cp = effect_params<CounterParams>(ab);
        cp.type2 = value;
        if (cp.count2 == 0) cp.count2 = 1;  // default to 1 like CounterNum
        return true;
    }
    if (key == "CounterNum2") { effect_params<CounterParams>(ab).count2 = std::stoi(value); return true; }
    if (key == "Monstrosity") {
        // Monstrosity$ N (CR 701.37): "Monstrosity N" = if this permanent isn't monstrous, put
        // N +1/+1 counters on it and it becomes monstrous. Modeled as a self-targeting +1/+1
        // PutCounter with a marker flag; the flag drives the became-monstrous side effects at
        // resolution and installs the "activate only while not monstrous" gate (CR 701.37a).
        auto &cp = effect_params<CounterParams>(ab);
        cp.type = "P1P1";
        cp.count = std::stoi(value);
        ab.is_monstrosity = true;
        ab.activation_condition = "NotMonstrous";
        return true;
    }
    return false;
}

}  // namespace effects
