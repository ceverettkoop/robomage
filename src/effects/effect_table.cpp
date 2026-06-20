#include "effects.h"

// Dispatch table for effect resolution. As each effect category is migrated out
// of Ability::resolve()'s legacy if/else chain into its own src/effects/
// translation unit, add its case here. Categories with no case fall through to
// nullptr, and Ability::resolve() runs its legacy branch for them. When the
// legacy chain is empty, this switch becomes the sole dispatch.
namespace effects {

EffectHandler handler_for(EffectKind kind) {
    switch (kind) {
        case EffectKind::DealDamage:      return &deal_damage;
        case EffectKind::Draw:            return &draw;
        case EffectKind::GainLife:        return &gain_life;
        case EffectKind::LoseLife:        return &lose_life;
        case EffectKind::Mill:            return &mill;
        case EffectKind::Untap:           return &untap;
        case EffectKind::Cleanup:         return &cleanup;
        case EffectKind::MultiplyCounter: return &multiply_counter;
        case EffectKind::Phases:          return &phases;
        case EffectKind::WinsGame:        return &wins_game;
        case EffectKind::ProwessBonus:    return &prowess_bonus;
        case EffectKind::ExaltedBonus:    return &exalted_bonus;
        case EffectKind::Attach:          return &attach;
        case EffectKind::ChooseCard:      return &choose_card;
        case EffectKind::DestroyAll:      return &destroy_all;
        default:                          return nullptr;
    }
}

}  // namespace effects
