#ifndef EFFECT_KIND_H
#define EFFECT_KIND_H

#include <string>

// Identifies which effect-resolution handler an Ability dispatches to. Mirrors
// the 32 effect-category strings that Ability::resolve() historically branched
// on. `None` covers Ability categories that have no resolve-time handler (e.g.
// "Equip", which is handled at activation time and never reaches resolve), plus
// any unrecognized string — both fall through to a no-op + subability chaining,
// matching the legacy if/else chain's behavior for unmatched categories.
enum class EffectKind {
    None,
    AddMana,
    GainLife,
    LoseLife,
    Discard,
    Draw,
    ChangeZone,
    ChangeZoneAll,
    RearrangeTopOfLibrary,
    DealDamage,
    PutCounter,
    ProwessBonus,
    ExaltedBonus,
    Token,
    Attach,
    Mill,
    Pump,
    MultiplyCounter,
    ChooseCard,
    Cleanup,
    DelayedTrigger,
    Untap,
    Destroy,
    DestroyAll,
    Counter,
    Surveil,
    PeekAndReveal,
    Phases,
    Dig,
    SylvanLibrary,
    WinsGame,
    Charm,
    Amass,
    Sacrifice,
    PutCounterAll,
    SacrificeAll,
    ImmediateTrigger,
};

// Maps a normalized category string to its EffectKind. Unknown strings → None.
EffectKind effect_kind_from_string(const std::string &category);

#endif /* EFFECT_KIND_H */
