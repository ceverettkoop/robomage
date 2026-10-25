#include "tokens.h"

#include <variant>

#include "../components/ability.h"
#include "../components/carddata.h"
#include "../effects/effects.h"

void ability_token_scripts(const AbilityDef &def, std::set<std::string> &out) {
    if (const TokenParams *tp = std::get_if<TokenParams>(&def.params))
        if (!tp->script.empty()) out.insert(tp->script);
    if (def.kind == EffectKind::Investigate) out.insert(effects::INVESTIGATE_TOKEN_SCRIPT);
    if (def.kind == EffectKind::Amass) out.insert(effects::amass_army_token_script(effects::amass_subtype(def)));
    if (def.kind == EffectKind::Mobilize) out.insert(effects::MOBILIZE_TOKEN_SCRIPT);
    for (const AbilityDef &sub : def.subabilities) ability_token_scripts(sub, out);
    for (const AbilityDef &mode : def.charm_choices) ability_token_scripts(mode, out);
    for (const AbilityDef &trig : def.effect_floating_triggers) ability_token_scripts(trig, out);
    for (const AbilityDef &granted : def.animate_granted_abilities) ability_token_scripts(granted, out);
}

void keyword_token_scripts(const CardData &cd, std::set<std::string> &out) {
    const CardData *faces[] = {&cd, cd.backside.get()};
    for (const CardData *face : faces) {
        if (!face) continue;
        for (const std::string &kw : face->keywords)
            if (kw.rfind("Mobilize", 0) == 0) out.insert(effects::MOBILIZE_TOKEN_SCRIPT);
    }
}
