#ifndef EFFECT_OBJECT_H
#define EFFECT_OBJECT_H

#include <cstdint>
#include <vector>

#include "../ecs/entity.h"
#include "../object_ref.h"
#include "ability.h"
#include "effect.h"
#include "static_ability.h"
#include "zone.h"

// What a resolved spell or ability leaves in the game that belongs to no permanent (CR 611.2), or
// an emblem (CR 114): a source of continuous effects, replacement effects and triggered abilities
// that exists outside any zone and can't be removed, read through Game::resolved_effects. Its
// controller is who it acts for, so its statics are gathered into g_active_statics every SBA pass
// with that player as their controller and apply through the normal layer engine, without the
// object being a real (targetable, counted, destructible) permanent. It holds no entity id:
// issuing one would move every later object's, and an effect is never in a zone.
//
// It lasts as `duration` says and is gone with its game: a fresh Game starts with none.
struct EffectObject {
    enum class Duration : uint8_t {
        END_OF_TURN,           // ends at cleanup (CR 514.2)
        UNTIL_YOUR_NEXT_TURN,  // ends as its controller's next turn begins (CR 611.2)
        GAME,                  // the rest of the game: an emblem (CR 114.1)
    };

    // Issued by ResolvedEffects::add, the handle a gathered static keeps: ending another object
    // moves this one within the list, so an index would not stay.
    int id = -1;
    Zone::Ownership controller = Zone::PLAYER_A;
    Duration duration = Duration::END_OF_TURN;
    // The card whose ability created it and its vocab idx captured at creation: its identity in
    // the state format. Display-only (the vocab idx is what the frames read), so a plain Entity.
    Entity source = 0;
    int source_vocab_idx = -1;
    std::vector<StaticAbility> statics;
    std::vector<Effect::Replacement> replacements;
    std::vector<Ability> triggers;  // each a TRIGGERED ability, controller bound
    std::vector<ObjectRef> remembered;  // RememberObjects$: what its abilities' IsRemembered reads

    bool is_emblem() const { return duration == Duration::GAME; }
    // Does it remember `e`, still the object it was (CR 400.7)?
    bool remembers(Entity e) const {
        for (const ObjectRef &ref : remembered)
            if (e != 0 && ref.get() == e) return true;
        return false;
    }
    // Does it hold a static or a replacement no row of the state format says by kind, so that a
    // frame can only name the card that made it (Sundering Eruption, Gaea's Will)?
    bool reported_by_card_alone() const {
        if (!statics.empty()) return true;
        for (const Effect::Replacement &r : replacements)
            if ((r.kind != Effect::Replacement::CANT_BE_COUNTERED && r.kind != Effect::Replacement::PREVENT_COMBAT_DAMAGE) ||
                r.protects_remembered)
                return true;
        return false;
    }
    // Is all combat damage dealt by (or to) what it remembers prevented (CR 615)?
    bool prevents_combat_damage_by() const {
        for (const Effect::Replacement &r : replacements)
            if (r.kind == Effect::Replacement::PREVENT_COMBAT_DAMAGE && r.prevent_by_remembered) return true;
        return false;
    }
    bool prevents_combat_damage_to() const {
        for (const Effect::Replacement &r : replacements)
            if (r.kind == Effect::Replacement::PREVENT_COMBAT_DAMAGE && r.prevent_to_remembered) return true;
        return false;
    }
};

#endif /* EFFECT_OBJECT_H */
