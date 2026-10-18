#include "reveals.h"

#include "../card_vocab.h"
#include "../classes/game.h"
#include "../components/carddata.h"
#include "../components/spell.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;
extern Game cur_game;

void mark_card_revealed(Entity e, Zone::Ownership owner) {
    if (owner != Zone::PLAYER_A && owner != Zone::PLAYER_B) return;
    if (global_coordinator.entity_has_component<Spell>(e) && global_coordinator.GetComponent<Spell>(e).is_copy) return;
    if (!global_coordinator.entity_has_component<Zone>(e)) return;
    const Zone &z = global_coordinator.GetComponent<Zone>(e);
    if (z.location != Zone::LIBRARY) return;
    cur_game.revealed_in_library.insert(e);
    if (global_coordinator.entity_has_component<CardData>(e)) {
        const bool known = card_name_to_index(global_coordinator.GetComponent<CardData>(e).name) != -1;
        cur_game.opp_known_top_set(z.owner, static_cast<int>(z.distance_from_top), known);
    }
}
