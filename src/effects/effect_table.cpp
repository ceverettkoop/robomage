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
        case EffectKind::Destroy:         return &destroy;
        case EffectKind::Token:           return &token;
        case EffectKind::Surveil:         return &surveil;
        case EffectKind::DelayedTrigger:  return &delayed_trigger;
        case EffectKind::PutCounter:      return &put_counter;
        case EffectKind::RearrangeTopOfLibrary: return &rearrange_top_of_library;
        case EffectKind::ChangeZone:      return &change_zone;
        case EffectKind::ChangeZoneAll:   return &change_zone_all;
        case EffectKind::Counter:         return &counter;
        case EffectKind::Charm:           return &charm;
        case EffectKind::AddMana:         return &add_mana;
        case EffectKind::Discard:         return &discard;
        case EffectKind::Pump:            return &pump;
        case EffectKind::PeekAndReveal:   return &peek_and_reveal;
        case EffectKind::Dig:             return &dig;
        case EffectKind::SylvanLibrary:   return &sylvan_library;
        default:                          return nullptr;
    }
}

}  // namespace effects
