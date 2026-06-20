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
        {"PeekAndReveal", EffectKind::PeekAndReveal},
        {"Phases", EffectKind::Phases},
        {"Dig", EffectKind::Dig},
        {"SylvanLibrary", EffectKind::SylvanLibrary},
        {"WinsGame", EffectKind::WinsGame},
        {"Charm", EffectKind::Charm},
    };
    auto it = table.find(category);
    return (it != table.end()) ? it->second : EffectKind::None;
}
