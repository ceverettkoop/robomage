#include "snapshot.h"

#include <string>
#include <unordered_map>

#include "card_db.h"
#include "classes/game.h"
#include "classes/match_state.h"
#include "ecs/coordinator.h"
#include "ecs/entity.h"

extern Game cur_game;
extern Coordinator global_coordinator;
// card_db is declared in card_db.h.

namespace {

// One saved game state. Holds a deep copy of every mutable per-game store: the
// value-typed Game, the ECS (via EcsSnapshot), the card name->entity map, and the
// match-scoped revealed arrays. card_db MUST be part of the snapshot: a card first
// loaded during a simulated line adds a prototype entity to the ECS and a
// name->entity mapping to card_db, so restoring the ECS without also restoring
// card_db would leave that mapping pointing at an entity that no longer exists —
// a later load_card/GetData would then abort.
struct GameSnapshot {
    bool live = false;
    Game game{};
    EcsSnapshot ecs;
    std::unordered_map<std::string, Entity> card_db_copy;
    unsigned char revealed_a[REVEALED_SIZE] = {};
    unsigned char revealed_b[REVEALED_SIZE] = {};
};

GameSnapshot g_slots[N_SNAPSHOT_SLOTS];

static bool valid_slot(int slot);

static bool valid_slot(int slot) { return slot >= 0 && slot < N_SNAPSHOT_SLOTS; }

}  // namespace

bool snapshot_save(int slot) {
    if (!valid_slot(slot)) return false;
    GameSnapshot &s = g_slots[slot];
    s.game = cur_game;
    global_coordinator.snapshot_to(s.ecs);
    s.card_db_copy = card_db;
    for (int i = 0; i < REVEALED_SIZE; ++i) {
        s.revealed_a[i] = g_revealed_by_a[i];
        s.revealed_b[i] = g_revealed_by_b[i];
    }
    s.live = true;
    return true;
}

bool snapshot_restore(int slot) {
    if (!valid_slot(slot)) return false;
    GameSnapshot &s = g_slots[slot];
    if (!s.live) return false;
    cur_game = s.game;
    global_coordinator.restore_from(s.ecs);
    // card_db is add-only during play (load_card only inserts), so a size match
    // means nothing was loaded since the save and the maps are identical — skip
    // the copy rather than reallocating every string key once per sim restore.
    if (card_db.size() != s.card_db_copy.size()) card_db = s.card_db_copy;
    for (int i = 0; i < REVEALED_SIZE; ++i) {
        g_revealed_by_a[i] = s.revealed_a[i];
        g_revealed_by_b[i] = s.revealed_b[i];
    }
    return true;
}

bool snapshot_slot_live(int slot) { return valid_slot(slot) && g_slots[slot].live; }

void snapshot_release_all() {
    for (int i = 0; i < N_SNAPSHOT_SLOTS; ++i) {
        g_slots[i] = GameSnapshot{};  // drops the held copies and clears the live flag
    }
}
