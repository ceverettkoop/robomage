#ifndef QUERIES_PLAYER_EFFECTS_H
#define QUERIES_PLAYER_EFFECTS_H

#include <set>
#include <vector>
#include "../components/zone.h"
#include "../ecs/entity.h"

// ── Player-scoped effects (the observation's PLAYER EFFECTS block) ───────────
// The continuous effects currently applying to one player as a whole, each read from the
// same state the rules consult:
//   protection_from_everything  — a Game::resolved_effects.player_protection_from_everything grant (The One Ring)
//   cant_gain_life              — player_cant_gain_life (Roiling Vortex's {R})
//   hexproof_from[W,U,B,R,G]    — the colors of this player's Game::resolved_effects.hexproof_from_colors_this_turn
//                                 grants (Veil of Summer)
//   spells_cant_be_countered    — player_spells_cant_be_countered(): a Veil of Summer grant or an
//                                 unfiltered "spells you control can't be countered" battlefield
//                                 static (Hexing Squelcher). A per-card or type-filtered form covers
//                                 only some spells and stays on its visible card/permanent.
//   may_cast_sorceries_as_flash — a Game::resolved_effects.cast_with_flash_permissions entry the player controls
//                                 (Teferi, Time Raveler's +1)
//   restricted_to_sorcery_speed — rules_mod::opponent_sorcery_speed_locked, derived from the live
//                                 static (Teferi, Time Raveler on the opponent's battlefield)
//   emblem_vocab_idx            — the distinct creating cards of the player's emblems, in creation
//                                 order (the serializer keeps the first MAX_EMBLEM_SLOTS)
//   floating_trigger_vocab_idx  — the creating card of the player's first live floating trigger
//                                 (Tamiyo, Seasoned Scholar's +2, Forth Eorlingas!); -1 = none
struct PlayerEffects {
    bool protection_from_everything = false;
    bool cant_gain_life = false;
    bool hexproof_from[5] = {false, false, false, false, false};
    bool spells_cant_be_countered = false;
    bool may_cast_sorceries_as_flash = false;
    bool restricted_to_sorcery_speed = false;
    std::vector<int> emblem_vocab_idx;
    int floating_trigger_vocab_idx = -1;
};
// `entities` must hold the battlefield permanents (e.g. the iterating system's mEntities).
PlayerEffects player_effects(Zone::Ownership player, const std::set<Entity> &entities);

#endif /* QUERIES_PLAYER_EFFECTS_H */
