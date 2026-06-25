#ifndef CONTINUOUS_EFFECTS_H
#define CONTINUOUS_EFFECTS_H

#include <cstddef>
#include <vector>
#include "../ecs/entity.h"

// Data model for the MTG layer system (comprehensive rule 613). Continuous
// effects are applied to objects' characteristics in a fixed series of layers,
// each layer in timestamp order (613.7), with a dependency override (613.8).
//
// This header defines the registration/ordering vocabulary the continuous-effects
// engine (StateManager::apply_continuous_effects, src/systems/state_manager_layers.cpp)
// sorts and dispatches on. Today only the layers that have implemented behavior are
// populated (4 type, 6 abilities, 7 P/T); the rest are documented extension points.

// Rule 613.1 layer order.
enum class Layer {
    COPY    = 1,  // 613.1a — copy effects, merges, face-down (extension point)
    CONTROL = 2,  // 613.1b — control-changing effects (extension point)
    TEXT    = 3,  // 613.1c — text-changing effects (extension point)
    TYPE    = 4,  // 613.1d — type/subtype/supertype changes (implemented)
    COLOR   = 5,  // 613.1e — color-changing effects (extension point)
    ABILITY = 6,  // 613.1f — ability add/remove, keyword counters (implemented: keyword grants)
    PT      = 7,  // 613.1g — power/toughness changes (implemented)
};

// Rule 613.4 sublayers within layer 7. Applied 7a -> 7b -> 7c -> 7d, each in
// timestamp order. recompute_pt() (src/components/creature.cpp) applies these
// sublayers to a single creature's stored contributions in this order.
enum class PTSublayer {
    NONE   = 0,
    CDA    = 1,  // 7a — characteristic-defining P/T (rule 604.3 / 613.4a)
    SET    = 2,  // 7b — "set power/toughness to N" (613.4b)
    MODIFY = 3,  // 7c — +N/+N effects and counters (613.4c)
    SWITCH = 4,  // 7d — switch power and toughness (613.4d) (extension point)
};

// One continuous-effect contribution, tagged with the layer/sublayer it applies
// in and the timestamp that orders it against peers (613.7). `affected` is the
// resolved object the effect modifies; the payload fields carry the modification
// for the layer it belongs to (start minimal: enough for layer-7 P/T). Future
// layers add their own payload as they are implemented.
struct ContinuousEffect {
    Layer       layer     = Layer::PT;
    PTSublayer  sublayer  = PTSublayer::NONE;
    size_t      timestamp = 0;        // 613.7 — earlier timestamp applies first
    bool        is_cda    = false;    // characteristic-defining: applies first within layer (613.3/613.4a)
    Entity      source    = 0;        // permanent/effect generating the contribution
    Entity      affected  = 0;        // resolved object the contribution modifies

    // Layer-7 payload.
    int delta_power     = 0;          // 7c additive power
    int delta_toughness = 0;          // 7c additive toughness
};

// Order a within-(sub)layer batch of effects per rule 613.7: characteristic-
// defining effects first (613.3/613.4a), then by ascending timestamp. The 613.8
// dependency system, when it overrides timestamps, is resolved by
// resolve_dependencies() below before this ordering is consumed.
//
// Ordering is stable (equal timestamps keep gather/insertion order), so it is
// deterministic. Every consumer today applies commutative (additive) 7c contributions,
// so the post-sort accumulation result is timestamp-independent; when a non-commutative
// effect (e.g. a 7b "set") is added, this ordering becomes observable and already does
// the right thing.
void order_continuous_effects(std::vector<ContinuousEffect> &effects);

// Rule 613.8 dependency override (extension point). A dependency, when present,
// makes one effect wait until another in the same (sub)layer has applied,
// overriding timestamp order. No card in the current vocab forms a dependency, so
// this is a documented no-op that falls through to timestamp order; implement the
// 613.8a–c algorithm here when the first dependent pair is added.
void resolve_dependencies(std::vector<ContinuousEffect> &effects);

#endif /* CONTINUOUS_EFFECTS_H */
