#ifndef EFFECT_KIND_H
#define EFFECT_KIND_H

#include <string>

// Which effect-resolution handler an ability dispatches to, from its category (the list is
// effect_kinds.def). `None` is an ability with no category (an engine-built targeting probe, a
// cost-only ability) or an unknown one; it resolves as a no-op that chains its sub-abilities.
enum class EffectKind {
    None,
#define EFFECT_KIND(kind, category, handler) kind,
#define EFFECT_KIND_ELSEWHERE(kind, category) kind,
#include "effect_kinds.def"
};

// The EffectKind of a category string; false (with `out` = None) for an unknown category.
bool effect_kind_from_category(const std::string &category, EffectKind &out);
// A kind's name: its category, as the state format's effect_kind table holds it ("None" for None).
const char *effect_kind_name(EffectKind kind);

#endif /* EFFECT_KIND_H */
