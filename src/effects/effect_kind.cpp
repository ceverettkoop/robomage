#include "effect_kind.h"

#include <unordered_map>

EffectKind effect_kind_from_string(const std::string &category) {
    static const std::unordered_map<std::string, EffectKind> table = {
        {"AddMana", EffectKind::AddMana},
        {"GainLife", EffectKind::GainLife},
        {"LoseLife", EffectKind::LoseLife},
        {"Discard", EffectKind::Discard},
        {"Draw", EffectKind::Draw},
        {"ChangeZone", EffectKind::ChangeZone},
        {"ChangeZoneAll", EffectKind::ChangeZoneAll},
        {"RearrangeTopOfLibrary", EffectKind::RearrangeTopOfLibrary},
        {"DealDamage", EffectKind::DealDamage},
        {"PutCounter", EffectKind::PutCounter},
        {"RemoveCounter", EffectKind::RemoveCounter},
        {"ProwessBonus", EffectKind::ProwessBonus},
        {"ExaltedBonus", EffectKind::ExaltedBonus},
        {"Token", EffectKind::Token},
        {"Attach", EffectKind::Attach},
        {"Mill", EffectKind::Mill},
        {"Pump", EffectKind::Pump},
        {"MultiplyCounter", EffectKind::MultiplyCounter},
        {"ChooseCard", EffectKind::ChooseCard},
        {"Cleanup", EffectKind::Cleanup},
        {"DelayedTrigger", EffectKind::DelayedTrigger},
        {"Untap", EffectKind::Untap},
        {"Destroy", EffectKind::Destroy},
        {"DestroyAll", EffectKind::DestroyAll},
        {"Counter", EffectKind::Counter},
        {"Surveil", EffectKind::Surveil},
        {"Scry", EffectKind::Scry},
        {"PeekAndReveal", EffectKind::PeekAndReveal},
        {"Phases", EffectKind::Phases},
        {"Dig", EffectKind::Dig},
        {"SylvanLibrary", EffectKind::SylvanLibrary},
        {"WinsGame", EffectKind::WinsGame},
        {"Charm", EffectKind::Charm},
        {"Amass", EffectKind::Amass},
        {"Sacrifice", EffectKind::Sacrifice},
        {"PutCounterAll", EffectKind::PutCounterAll},
        {"SacrificeAll", EffectKind::SacrificeAll},
        {"ImmediateTrigger", EffectKind::ImmediateTrigger},
        {"CopyPermanent", EffectKind::CopyPermanent},
        {"Mobilize", EffectKind::Mobilize},
        {"SacrificeTokens", EffectKind::SacrificeTokens},
        {"RepeatEach", EffectKind::RepeatEach},
        // AB$ Effect that grants "you may cast that card this turn" (Emry, Lurker of the
        // Loch). The transient continuous Effect object is modeled as a per-turn cast
        // permission rather than a stack object; see effect_grant_cast.cpp.
        {"Effect", EffectKind::GrantCast},
    };
    auto it = table.find(category);
    return (it != table.end()) ? it->second : EffectKind::None;
}
