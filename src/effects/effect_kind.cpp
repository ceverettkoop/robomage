#include "effect_kind.h"

#include <unordered_map>

bool effect_kind_from_category(const std::string &category, EffectKind &out) {
    static const std::unordered_map<std::string, EffectKind> table = {
#define EFFECT_KIND(kind, category, handler) {category, EffectKind::kind},
#define EFFECT_KIND_ELSEWHERE(kind, category) {category, EffectKind::kind},
#include "effect_kinds.def"
    };
    auto it = table.find(category);
    out = (it != table.end()) ? it->second : EffectKind::None;
    return it != table.end();
}
