#include "effects.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "../classes/action.h"
#include "../classes/game.h"
#include "../cli_output.h"
#include "../components/permanent.h"
#include "../components/token.h"
#include "../components/zone.h"
#include "../coverage.h"
#include "../ecs/coordinator.h"
#include "../parse.h"
#include "../queries/battlefield.h"
#include "../queries/characteristics.h"
#include "../queries/counters.h"
#include "../queries/players.h"
#include "../queries/types.h"
#include "../systems/orderer.h"

extern Coordinator global_coordinator;
extern Game cur_game;

namespace effects {

// What an empower makes when its controller has none: a Type$ is a row of this table, and one with
// no row a parse finding.
struct EmpowerToken {
        const char *subtype;  // Type$, the token's subtype
        const char *script;   // its token script's stem
};

static const EmpowerToken kEmpowerTokens[] = {
    {"Jace", "u_empower_jace"},
};

static const EmpowerToken *empower_row(const std::string &subtype);
static std::vector<Entity> empower_tokens(Zone::Ownership ctrl, const std::string &subtype, const EntitySet &entities);
static Entity create_empower_token(const std::string &script, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer);
static std::vector<LegalAction> empower_menu(const std::vector<Entity> &tokens, int n);

static const EmpowerToken *empower_row(const std::string &subtype) {
    for (const EmpowerToken &row : kEmpowerTokens)
        if (subtype == row.subtype) return &row;
    return nullptr;
}

// The tokens of `subtype` `ctrl` controls. A phased-out one doesn't exist (CR 702.26b), so an empower
// makes another and the two can be there together once it phases in.
static std::vector<Entity> empower_tokens(Zone::Ownership ctrl, const std::string &subtype, const EntitySet &entities) {
    std::vector<Entity> tokens;
    for (Entity e : entities) {
        if (!is_battlefield_permanent(e, ctrl)) continue;
        const Permanent &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.is_token && permanent_has_type(perm, subtype)) tokens.push_back(e);
    }
    return tokens;
}

static Entity create_empower_token(const std::string &script, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer) {
    Token tok = parse_token_script(script);
    if (tok.name.empty()) return 0;
    Entity token = global_coordinator.CreateEntity();
    global_coordinator.AddComponent(token, Zone(Zone::HAND, ctrl, ctrl));
    global_coordinator.AddComponent(token, tok);
    orderer->add_to_zone(false, token, Zone::BATTLEFIELD);
    bootstrap_token_components(token, tok, ctrl, cur_game.timestamp);
    game_log("Token created: %s\n", tok.name.c_str());
    return token;
}

static std::vector<LegalAction> empower_menu(const std::vector<Entity> &tokens, int n) {
    std::vector<LegalAction> menu;
    for (Entity e : tokens) {
        LegalAction la(PASS_PRIORITY, e,
            "Put " + std::to_string(n) + " loyalty counters on " + entity_name(e) + " (loyalty " +
                std::to_string(get_counters(e, "LOYALTY")) + ")");
        la.category = ActionCategory::CHOOSE_CARD;
        la.card_is_public = true;
        menu.push_back(la);
    }
    return menu;
}

// "Empower <subtype> N" (not in the checked-in CR; docs/rules_deviations.md): put N loyalty counters
// on a <subtype> token you control, and if you control none, first create one. The token and its
// counters arrive in one resolution, so no state-based action sees it at 0 loyalty (CR 704.5i).
HandlerResult empower(Ability &ab, std::shared_ptr<Orderer> orderer, FrameCtx &ctx) {
    const int n = static_cast<int>(ab.def->amount);
    const std::string subtype = empower_subtype(*ab.def);
    const Zone::Ownership ctrl = ab.controller;  // "you" is the ability's controller (CR 109.5)

    const std::vector<Entity> tokens = empower_tokens(ctrl, subtype, orderer->mEntities);
    Entity token = 0;
    if (tokens.size() > 1) {
        if (!ctx.resuming()) coverage_row("empower_choice");
        const int choice = ctx.ask(empower_menu(tokens, n), ctrl, ab.source.lki_entity());
        if (choice < 0 && decision_suspended()) return HandlerResult::SUSPENDED;
        token = tokens[static_cast<size_t>(choice < 0 ? 0 : choice)];
    } else if (!tokens.empty()) {
        coverage_row("empower_existing");
        token = tokens[0];
    } else {
        token = create_empower_token(empower_token_script(subtype), ctrl, orderer);
    }
    if (token == 0) return HandlerResult::DONE_RUN_SUBS;

    const int loyalty = add_counters(token, "LOYALTY", n);
    game_log("Empower %s %d: %s's %s now has %d loyalty.\n", subtype.c_str(), n, player_name(ctrl).c_str(),
        entity_name(token).c_str(), loyalty);
    return HandlerResult::DONE_RUN_SUBS;
}

bool parse_empower(AbilityDef &ab, const std::string &key, const std::string &value) {
    if (ab.category != "Empower") return false;
    if (key == "Type") {
        if (!empower_row(value)) return false;
        effect_params<EmpowerParams>(ab).subtype = value;
        return true;
    }
    if (key == "Num") {
        if (value.empty() || !std::all_of(value.begin(), value.end(), [](unsigned char c) { return std::isdigit(c); }))
            return false;
        ab.amount = static_cast<size_t>(std::stoi(value));
        return true;
    }
    return false;
}

std::string empower_token_script(const std::string &subtype) {
    const EmpowerToken *row = empower_row(subtype);
    return row ? row->script : "";
}

std::string empower_subtype(const AbilityDef &def) {
    const EmpowerParams *ep = std::get_if<EmpowerParams>(&def.params);
    return ep ? ep->subtype : "";
}

}  // namespace effects
