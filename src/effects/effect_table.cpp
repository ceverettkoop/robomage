#include "effects.h"

// Dispatch table for effect resolution: each effect category's handler lives in its own
// src/effects/ translation unit (the list is effect_kinds.def). A kind with no handler returns
// nullptr and resolve_ability() just chains its subabilities.
namespace effects {

EffectHandler handler_for(EffectKind kind) {
    switch (kind) {
#define EFFECT_KIND(kind, category, handler) \
    case EffectKind::kind:                   \
        return &handler;
#include "effect_kinds.def"
        default: return nullptr;
    }
}

// Tries each effect's parse hook in turn. Keys are partitioned so at most one
// hook claims any given (key, value); order is therefore irrelevant. Returns
// true if some hook consumed the key.
bool apply_parse_hook(AbilityDef &ab, const std::string &key, const std::string &value) {
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
