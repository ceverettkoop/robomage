#ifndef QUERIES_PLAYER_RESOURCES_H
#define QUERIES_PLAYER_RESOURCES_H

#include <cstdint>
#include "../components/player.h"
#include "../ecs/entity.h"

// ── Life (CR 119) and energy (CR 122.1c) ─────────────────────────────────────

// True if a turn-long "can't gain life" prohibition (CR 119.x) currently applies to this player
// (Roiling Vortex's {R} ability, cur_game.resolved_effects.cant_gain_life_this_turn). Defined out-of-line in
// player_resources.cpp (needs cur_game). Consulted by player_gain_life so every life-gain site obeys
// the prohibition.
bool player_cant_gain_life(Entity player_entity);

// Single source for "a player gains life": raises their life total and accumulates
// life_gained_this_turn (118.9 / the "if you gained life this turn" check on cards like
// Ocelot Pride). Every life-gain site (lifelink, GainLife effects, combat lifelink) routes
// through this so the per-turn counter cannot drift from life_total. Pass the player's
// Player-component entity. No-op for amount <= 0.
void player_gain_life(Entity player_entity, int32_t amount);

// Single source for "a player loses life": lowers their life total and accumulates
// life_lost_this_turn (the mirror of player_gain_life). Read by Spectacle (CR 702.107a,
// "you may cast this spell for its spectacle cost … if an opponent lost life this turn").
// Damage dealt to a player IS a loss of life (CR 120.3), so the damage-to-player sites
// (combat + noncombat DealDamage) and the explicit "lose life" effects route through here
// so the per-turn counter cannot drift from life_total. Life PAID as a cost goes through
// pay_life below. Pass the player's Player-component entity. No-op for amount <= 0.
void player_lose_life(Entity player_entity, int32_t amount);

// ── Paying life (CR 119.4) ──────────────────────────────────────────────────
// The single check and payment path for every life cost (fetch/horizon-land PayLife, Phyrexian
// pips, activation/alternative/deferred life costs, "unless you pay N life", Sylvan Library,
// enters-tapped-unless-you-pay-life).

// Can `pl` pay `n` life? Only with a life total of at least `n` (CR 119.4); 0 life can always be
// paid (CR 119.4b).
inline bool can_pay_life(const Player &pl, int n) { return n <= 0 || pl.life_total >= n; }

// Pay `n` life from `pl`. Returns false and changes nothing if the player can't pay it;
// otherwise subtracts it and returns true. Paying life is losing that much life (CR 119.4), so it
// accumulates life_lost_this_turn like player_lose_life (Spectacle sees an opponent's payment).
bool pay_life(Player &pl, int n);

// ── Player energy ({E}, CR 122.1c) ──────────────────────────────────────────
// Energy is stored as an "ENERGY" counter in Player::counters. These are the single
// read/spend path so every energy producer/consumer (Guide of Souls, Wrath of the Skies,
// Amped Raptor) agrees on the key and the "can't pay if insufficient" rule.

// Amount of energy ({E}) the player currently has.
inline int player_energy(const Player &pl) { return pl.counter_count("ENERGY"); }

// Pay `n` energy from `pl` (CR 122.1c / 118.x — paying {E} is a cost). Returns false and
// leaves the pool untouched if the player has fewer than `n`; otherwise deducts and returns
// true. `n <= 0` is a trivially-payable no-op (returns true). Reusable by every "pay {E}"
// cost / optional payment.
bool pay_energy(Player &pl, int n);

#endif /* QUERIES_PLAYER_RESOURCES_H */
