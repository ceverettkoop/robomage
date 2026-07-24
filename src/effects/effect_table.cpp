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
        case EffectKind::UntapAll:        return &untap_all;
        case EffectKind::Cleanup:         return &cleanup;
        case EffectKind::MultiplyCounter: return &multiply_counter;
        case EffectKind::Phases:          return &phases;
        case EffectKind::WinsGame:        return &wins_game;
        case EffectKind::ProwessBonus:    return &prowess_bonus;
        case EffectKind::ExaltedBonus:    return &exalted_bonus;
        case EffectKind::Attach:          return &attach;
        case EffectKind::ChooseCard:      return &choose_card;
        case EffectKind::DestroyAll:      return &destroy_all;
        case EffectKind::DamageAll:       return &damage_all;
        case EffectKind::Destroy:         return &destroy;
        case EffectKind::Token:           return &token;
        case EffectKind::Investigate:     return &investigate;
        case EffectKind::Surveil:         return &surveil;
        case EffectKind::Scry:            return &scry;
        case EffectKind::DelayedTrigger:  return &delayed_trigger;
        case EffectKind::PutCounter:      return &put_counter;
        case EffectKind::RemoveCounter:   return &remove_counter;
        case EffectKind::RearrangeTopOfLibrary: return &rearrange_top_of_library;
        case EffectKind::ChangeZone:      return &change_zone;
        case EffectKind::ChangeZoneAll:   return &change_zone_all;
        case EffectKind::Counter:         return &counter;
        case EffectKind::Charm:           return &charm;
        case EffectKind::AddMana:         return &add_mana;
        case EffectKind::Discard:         return &discard;
        case EffectKind::Pump:            return &pump;
        case EffectKind::PumpAll:         return &pump_all;
        case EffectKind::PeekAndReveal:   return &peek_and_reveal;
        case EffectKind::Reveal:          return &reveal;
        case EffectKind::Dig:             return &dig;
        case EffectKind::SylvanLibrary:   return &sylvan_library;
        case EffectKind::Amass:           return &amass;
        case EffectKind::Sacrifice:       return &sacrifice;
        case EffectKind::PutCounterAll:   return &put_counter_all;
        case EffectKind::SacrificeAll:    return &sacrifice_all;
        case EffectKind::ImmediateTrigger: return &immediate_trigger;
        case EffectKind::CopyPermanent:   return &copy_permanent;
        case EffectKind::Clone:           return &clone;
        case EffectKind::Mobilize:        return &mobilize;
        case EffectKind::SacrificeTokens: return &sacrifice_tokens;
        case EffectKind::ExileTokens:     return &exile_tokens;
        case EffectKind::RepeatEach:      return &repeat_each;
        case EffectKind::GrantCast:       return &grant_cast;
        case EffectKind::NameCard:        return &name_card;
        case EffectKind::Animate:         return &animate;
        case EffectKind::AnimateAll:      return &animate_all;
        case EffectKind::ChooseNumber:    return &choose_number;
        case EffectKind::DigUntil:        return &dig_until;
        case EffectKind::Play:            return &play;
        case EffectKind::Earthbend:       return &earthbend;
        case EffectKind::Tap:             return &tap;
        case EffectKind::RevealHand:      return &reveal_hand;
        case EffectKind::BecomeMonarch:   return &become_monarch;
        case EffectKind::Vote:            return &vote;
        case EffectKind::Storm:           return &storm;
        case EffectKind::AddTurn:         return &add_turn;
        case EffectKind::StoreSVar:       return &store_svar;
        case EffectKind::CopySpellAbility: return &copy_spell_ability;
        case EffectKind::SuspendTick:     return &suspend_tick;
        case EffectKind::SetState:        return &set_state;
        default:                          return nullptr;
    }
}

// Tries each effect's parse hook in turn. Keys are partitioned so at most one
// hook claims any given (key, value); order is therefore irrelevant. Returns
// true if some hook consumed the key.
bool apply_parse_hook(Ability &ab, const std::string &key, const std::string &value) {
    return parse_deal_damage(ab, key, value)
        || parse_pump(ab, key, value)
        || parse_token(ab, key, value)
        || parse_add_mana(ab, key, value)
        || parse_destroy_all(ab, key, value)
        || parse_change_zone(ab, key, value)
        || parse_put_counter(ab, key, value)
        || parse_dig(ab, key, value)
        || parse_delayed_trigger(ab, key, value)
        || parse_discard(ab, key, value)
        || parse_mill(ab, key, value)
        || parse_peek_and_reveal(ab, key, value)
        || parse_reveal(ab, key, value)
        || parse_amass(ab, key, value)
        || parse_choose_number(ab, key, value)
        || parse_dig_until(ab, key, value)
        || parse_play(ab, key, value)
        || parse_animate_all(ab, key, value)
        || parse_store_svar(ab, key, value)
        || parse_set_state(ab, key, value);
}

}  // namespace effects
