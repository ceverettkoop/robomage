#include "effects.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../cli_output.h"
#include "../components/creature.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"
#include "../ecs/entity.h"
#include "../game_queries.h"
#include "../input_logger.h"

extern Coordinator global_coordinator;

namespace effects {

bool pump(Ability &ab, std::shared_ptr<Orderer> orderer) {
    (void)orderer;
    // Pump used purely as a targeting vehicle for a graveyard card (Surgical Extraction's
    // SP$ Pump | TgtZone$ Graveyard): the target was already chosen at cast and the
    // subabilities do the work — don't re-pick a battlefield creature here.
    if (ab.target_in_graveyard) return true;

    // Present target selection, then chain subabilities with that target
    Zone::Ownership ctrl = ab.controller;
    std::vector<Entity> pump_targets;
    for (Entity e = 0; e < global_coordinator.GetMaxIssuedEntity(); ++e) {
        if (!is_battlefield_permanent(e)) continue;
        if (!global_coordinator.entity_has_component<Creature>(e)) continue;
        auto &p = global_coordinator.GetComponent<Permanent>(e);
        if (ab.valid_tgts.find("YouCtrl") != std::string::npos && p.controller != ctrl) continue;
        pump_targets.push_back(e);
    }
    if (pump_targets.empty()) {
        game_log("Pump: no valid targets.\n");
        // still chain subabilities with no target
    } else {
        game_log("Choose a creature for Pump:\n");
        std::vector<LegalAction> tgt_actions;
        for (auto te : pump_targets) {
            std::string ename = global_coordinator.GetComponent<Permanent>(te).name;
            auto &tcr = global_coordinator.GetComponent<Creature>(te);
            LegalAction la(PASS_PRIORITY, te,
                ename + " [" + std::to_string(tcr.power) + "/" + std::to_string(tcr.toughness) + "]");
            la.category = ActionCategory::SELECT_TARGET;
            tgt_actions.push_back(la);
        }
        int choice = InputLogger::instance().get_input(tgt_actions);
        if (choice >= 0 && choice < static_cast<int>(pump_targets.size()))
            ab.target = pump_targets[static_cast<size_t>(choice)];
    }
    // Apply P/T modification if NumAtt$/NumDef$ were set. A count-SVar NumAtt$/NumDef$
    // (e.g. Eldrazi Linebreaker's "+X" where X = number of Eldrazi you control) is
    // evaluated now against the ability's controller.
    const PumpParams *pp = std::get_if<PumpParams>(&ab.params);
    int pump_att = pp ? pp->att : 0;
    int pump_def = pp ? pp->def : 0;
    if (pp && !pp->att_expr.empty())
        pump_att = pp->att_sign * static_cast<int>(evaluate_dynamic_amount(pp->att_expr, ctrl, orderer, ab.target));
    if (pp && !pp->def_expr.empty())
        pump_def = pp->def_sign * static_cast<int>(evaluate_dynamic_amount(pp->def_expr, ctrl, orderer, ab.target));
    bool grants_kw = pp && !pp->grant_keywords.empty();
    if ((pump_att != 0 || pump_def != 0 || grants_kw) && ab.target != 0 &&
        global_coordinator.entity_has_component<Creature>(ab.target)) {
        auto &cr = global_coordinator.GetComponent<Creature>(ab.target);
        // "Until end of turn" pump: store in the EOT bonus bucket so cleanup (514.2/611.2b)
        // reverts it. Signed + floored at 0 by recompute_pt so a -X/-X pump (e.g. Dismember)
        // can never underflow the uint32_t effective field.
        cr.eot_power_bonus += pump_att;
        cr.eot_toughness_bonus += pump_def;
        recompute_pt(cr);
        std::string tname = global_coordinator.entity_has_component<Permanent>(ab.target)
            ? global_coordinator.GetComponent<Permanent>(ab.target).name : "<unknown>";
        if (pump_att != 0 || pump_def != 0)
            game_log("%s gets %+d/%+d (now %u/%u)\n", tname.c_str(), pump_att, pump_def, cr.power, cr.toughness);
        // Grant "until end of turn" keyword(s) (e.g. Haste). Stored in the eot_keywords
        // bucket; the static pass re-merges them onto cr.keywords each pass and cleanup
        // clears them (514.2). De-dup so repeated grants don't pile up.
        for (const auto &kw : pp->grant_keywords) {
            if (std::find(cr.eot_keywords.begin(), cr.eot_keywords.end(), kw) == cr.eot_keywords.end())
                cr.eot_keywords.push_back(kw);
            if (std::find(cr.keywords.begin(), cr.keywords.end(), kw) == cr.keywords.end())
                cr.keywords.push_back(kw);
            game_log("%s gains %s until end of turn.\n", tname.c_str(), kw.c_str());
        }
    }
    return true;
}

// Parse a NumAtt$/NumDef$ value that is either a literal integer (e.g. "2", "-1")
// or a signed count-SVar reference (e.g. "+X", "-X"). For the SVar form the static
// magnitude stays 0 and the SVar key + sign are recorded for runtime resolution; the
// caller resolves the SVar key (e.g. "X") to its Count$ expression. Returns true if a
// non-numeric SVar reference was stored (so the parser knows to resolve the expr).
static void parse_pump_amount(const std::string &value, int &out_static, std::string &out_expr, int &out_sign) {
    if (value.empty()) return;
    int sign = 1;
    std::string body = value;
    if (body[0] == '+') { body = body.substr(1); }
    else if (body[0] == '-') { sign = -1; body = body.substr(1); }
    bool numeric = !body.empty();
    for (char c : body) if (!std::isdigit(static_cast<unsigned char>(c))) { numeric = false; break; }
    if (numeric) {
        out_static = sign * std::stoi(body);
        out_expr.clear();
    } else {
        // Non-numeric body (e.g. "X") is an SVar key resolved later to a Count$ expression.
        out_static = 0;
        out_expr = body;
        out_sign = sign;
    }
}

bool parse_pump(Ability &ab, const std::string &key, const std::string &value) {
    if (key == "NumAtt") {
        PumpParams &pp = effect_params<PumpParams>(ab);
        parse_pump_amount(value, pp.att, pp.att_expr, pp.att_sign);
        return true;
    }
    if (key == "NumDef") {
        PumpParams &pp = effect_params<PumpParams>(ab);
        parse_pump_amount(value, pp.def, pp.def_expr, pp.def_sign);
        return true;
    }
    if (key == "KW") {
        // KW$ — keyword(s) granted to the pumped creature until end of turn. Forge
        // separates multiple keywords with " & " (e.g. "Flying & Haste").
        PumpParams &pp = effect_params<PumpParams>(ab);
        size_t start = 0;
        while (start < value.size()) {
            size_t amp = value.find(" & ", start);
            std::string kw = (amp == std::string::npos) ? value.substr(start)
                                                        : value.substr(start, amp - start);
            // trim
            size_t b = kw.find_first_not_of(" \t");
            size_t e = kw.find_last_not_of(" \t");
            if (b != std::string::npos) pp.grant_keywords.push_back(kw.substr(b, e - b + 1));
            if (amp == std::string::npos) break;
            start = amp + 3;
        }
        return true;
    }
    return false;
}

}  // namespace effects
