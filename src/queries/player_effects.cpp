#include "player_effects.h"

#include <algorithm>

#include "../classes/game.h"
#include "../systems/rules_modifying.h"
#include "player_resources.h"
#include "players.h"
#include "spells.h"

PlayerEffects player_effects(Zone::Ownership player, const EntitySet &entities) {
    PlayerEffects fx;
    for (const auto &p : cur_game.resolved_effects.player_protection_from_everything)
        if (p.player == player) fx.protection_from_everything = true;
    fx.cant_gain_life = player_cant_gain_life(get_player_entity(player));
    const Colors wubrg[5] = {WHITE, BLUE, BLACK, RED, GREEN};
    for (const auto &h : cur_game.resolved_effects.hexproof_from_colors_this_turn) {
        if (h.player != player) continue;
        for (int i = 0; i < 5; i++)
            if (h.colors.count(wubrg[i])) fx.hexproof_from[i] = true;
    }
    fx.spells_cant_be_countered = player_spells_cant_be_countered(player, entities);
    for (const auto &perm : cur_game.resolved_effects.cast_with_flash_permissions)
        if (perm.controller == player) fx.may_cast_sorceries_as_flash = true;
    fx.restricted_to_sorcery_speed = rules_mod::opponent_sorcery_speed_locked(player);
    for (const EffectObject &object : cur_game.resolved_effects.effect_objects) {
        if (object.controller != player || object.source_vocab_idx < 0) continue;
        if (object.is_emblem() && std::find(fx.emblem_vocab_idx.begin(), fx.emblem_vocab_idx.end(),
                                            object.source_vocab_idx) == fx.emblem_vocab_idx.end())
            fx.emblem_vocab_idx.push_back(object.source_vocab_idx);
        if (fx.floating_trigger_vocab_idx < 0 && !object.triggers.empty())
            fx.floating_trigger_vocab_idx = object.source_vocab_idx;
    }
    return fx;
}
