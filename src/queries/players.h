#ifndef QUERIES_PLAYERS_H
#define QUERIES_PLAYERS_H

#include <string>
#include "../components/zone.h"
#include "../ecs/entity.h"

struct Ability;

// ── Seats (two-player game, CR 102.2) ───────────────────────────────────────
// The engine seats exactly two players, so "the opponent" is always the other seat. These are the
// single conversions between a seat (Zone::Ownership) and its player entity; do not open-code the
// PLAYER_A/PLAYER_B ternaries at call sites. Defined in players.cpp (they read cur_game).

// The other seat: a player's sole opponent (CR 102.2). UNKNOWN stays UNKNOWN.
inline Zone::Ownership opponent_of(Zone::Ownership p) {
    if (p == Zone::PLAYER_A) return Zone::PLAYER_B;
    if (p == Zone::PLAYER_B) return Zone::PLAYER_A;
    return Zone::UNKNOWN;
}

// The player entity sitting in `player`'s seat.
Entity get_player_entity(Zone::Ownership player);
// Revolt: a permanent `player` controlled left the battlefield this turn.
bool revolt_this_turn(Zone::Ownership player);

// The seat of a player entity; UNKNOWN when `player_entity` is not a player.
Zone::Ownership seat_of_player(Entity player_entity);

// The seat holding priority, and the active player's seat (the player whose turn it is).
Zone::Ownership priority_seat();
Zone::Ownership active_seat();

// ── Player targets (CR 115.1 / 109.5) ───────────────────────────────────────
// The player half of a Forge ValidTgts$ spec: its comma-OR alternatives that name a player —
// "Any" (any target), "Player" (either player), "Opponent" / "Player.Opponent" (only an opponent
// of `you`), "Player.You" (only `you`). `names_players` says whether any alternative names a
// player at all; `player_matches_target_spec` whether `player` satisfies one of them.
bool target_spec_names_players(const std::string &valid_tgts);
bool player_matches_target_spec(const std::string &valid_tgts, Entity player, Zone::Ownership you);

// ── Defined$ player resolution (CR 109.5 / 608.2g) ──────────────────────────
// Who controls an object: its live Permanent.controller while on the battlefield, a spell's
// caster while on the stack (CR 110.2), else the owner of its current zone (CR 108.4: an object
// outside the battlefield and stack is its owner's), else a vanished token's last-known
// controller; a player entity reports its own seat. The one "who controls entity e" query; do
// not re-derive it from Zone::owner at call sites. For "you" in a resolving ability use the
// ability's own controller (Ability::controller, CR 109.5 / 608.2g), which stays fixed if the
// source changes control or leaves play.
Zone::Ownership source_controller(Entity source);

// Last-known controller of an object for a "that permanent's controller" effect
// (CR 608.2g/h): its Zone.controller while it still records one, else the live
// Permanent.controller (mid-resolution before the SBA strips it), else the controller
// captured in its last-known info as it left the battlefield. UNKNOWN if never controlled.
Zone::Ownership last_known_controller(Entity e);

// Resolve the Defined$ player an ability designates, or UNKNOWN when it names none (the
// caller then falls back to its own chosen target):
//   Defined$ You                -> the ability's controller (CR 109.5)
//   Defined$ Player.Opponent    -> that controller's single opponent (2-player; CR 109.5)
//   Defined$ TargetedController -> the last-known controller of ab.target
//   Defined$ TriggeredActivator -> the player bound when the trigger fired
Zone::Ownership resolve_defined_player(const Ability &ab);

#endif /* QUERIES_PLAYERS_H */
