#ifndef EFFECTS_H
#define EFFECTS_H

#include <memory>

#include "../components/ability.h"
#include "effect_kind.h"

class Orderer;

// ── Effect handler dispatch ────────────────────────────────────────────────
//
// Each effect category resolves through a free-function handler with this
// signature. The bool return is "run the standard subability-chaining loop
// afterward?" — almost every effect returns true; the handful that manage their
// own subability resolution or short-circuit the game (Charm, WinsGame, the
// non-peek PeekAndReveal path) return false to suppress it, exactly as the
// legacy if/else chain did via early `return`.
namespace effects {

using EffectHandler = bool (*)(Ability &, std::shared_ptr<Orderer>);

// Returns the handler for `kind`, or nullptr if no resolve-time handler exists
// (None / not-yet-migrated). Ability::resolve() falls back to its legacy chain
// when this returns nullptr.
EffectHandler handler_for(EffectKind kind);

// Per-effect handlers (defined one per src/effects/effect_*.cpp).
bool deal_damage(Ability &ab, std::shared_ptr<Orderer> orderer);
bool draw(Ability &ab, std::shared_ptr<Orderer> orderer);
bool gain_life(Ability &ab, std::shared_ptr<Orderer> orderer);
bool lose_life(Ability &ab, std::shared_ptr<Orderer> orderer);
bool mill(Ability &ab, std::shared_ptr<Orderer> orderer);
bool untap(Ability &ab, std::shared_ptr<Orderer> orderer);
bool cleanup(Ability &ab, std::shared_ptr<Orderer> orderer);
bool multiply_counter(Ability &ab, std::shared_ptr<Orderer> orderer);
bool phases(Ability &ab, std::shared_ptr<Orderer> orderer);
bool wins_game(Ability &ab, std::shared_ptr<Orderer> orderer);
bool prowess_bonus(Ability &ab, std::shared_ptr<Orderer> orderer);
bool exalted_bonus(Ability &ab, std::shared_ptr<Orderer> orderer);
bool attach(Ability &ab, std::shared_ptr<Orderer> orderer);
bool choose_card(Ability &ab, std::shared_ptr<Orderer> orderer);
bool destroy_all(Ability &ab, std::shared_ptr<Orderer> orderer);
bool destroy(Ability &ab, std::shared_ptr<Orderer> orderer);
bool token(Ability &ab, std::shared_ptr<Orderer> orderer);
bool surveil(Ability &ab, std::shared_ptr<Orderer> orderer);
bool delayed_trigger(Ability &ab, std::shared_ptr<Orderer> orderer);
bool put_counter(Ability &ab, std::shared_ptr<Orderer> orderer);
bool rearrange_top_of_library(Ability &ab, std::shared_ptr<Orderer> orderer);

}  // namespace effects

// ── Shared resolution helpers ───────────────────────────────────────────────
// De-static'd from ability.cpp so effect handlers in separate translation units
// can reuse them. Definitions still live in ability.cpp for now.
// (search_zone is declared in ability.h.)
size_t evaluate_dynamic_amount(
    const std::string &expr, Zone::Ownership ctrl, std::shared_ptr<Orderer> orderer, Entity target);

#endif /* EFFECTS_H */
