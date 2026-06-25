#include "match_state.h"

#include "../card_vocab.h"
#include "../components/carddata.h"
#include "../components/permanent.h"
#include "../components/zone.h"
#include "../ecs/coordinator.h"

extern Coordinator global_coordinator;

unsigned char g_revealed_by_a[REVEALED_SIZE] = {};
unsigned char g_revealed_by_b[REVEALED_SIZE] = {};

void match_reset_revealed() {
    for (int i = 0; i < REVEALED_SIZE; i++) {
        g_revealed_by_a[i] = 0;
        g_revealed_by_b[i] = 0;
    }
}

void mark_card_revealed(Entity e, Zone::Ownership owner) {
    if (owner != Zone::PLAYER_A && owner != Zone::PLAYER_B) return;

    // Per-card, per-game belief: if this card is revealed while still in a hidden
    // zone (hand), record that its specific identity is now known to the non-owner
    // so the observation can carry the exact card, not just "seen once this match".
    // Cleared by Orderer::add_to_zone when the card next changes zones.
    if (global_coordinator.entity_has_component<Zone>(e)) {
        auto &z = global_coordinator.GetComponent<Zone>(e);
        if (z.location == Zone::HAND) z.identity_known = true;
    }

    // Resolve the card's vocab index the same way machine_io does: a battlefield
    // permanent carries its name on Permanent; everything else on CardData.
    int idx = -1;
    if (global_coordinator.entity_has_component<Permanent>(e)) {
        auto &perm = global_coordinator.GetComponent<Permanent>(e);
        if (perm.is_token) return;  // tokens are not deck cards
        idx = card_name_to_index(perm.name);
    } else if (global_coordinator.entity_has_component<CardData>(e)) {
        idx = card_name_to_index(global_coordinator.GetComponent<CardData>(e).name);
    }

    // Ignore unregistered cards (-1) and the token sentinel slot.
    if (idx < 0 || idx >= REVEALED_SIZE || idx == TOKEN_SENTINEL) return;

    unsigned char *arr = (owner == Zone::PLAYER_A) ? g_revealed_by_a : g_revealed_by_b;
    arr[idx] = 1;
}
