#ifndef SNAPSHOT_H
#define SNAPSHOT_H
// In-process deep-copy snapshot/restore of the complete per-game state
// (cur_game + ECS + card_db + match revealed arrays), for MCTS search.
inline constexpr int N_SNAPSHOT_SLOTS = 4;
bool snapshot_save(int slot);       // deep copy current state into slot; false on bad slot
bool snapshot_restore(int slot);    // restore state from slot; false if slot empty/bad
bool snapshot_slot_live(int slot);  // slot holds a snapshot
void snapshot_release_all();        // drop all snapshots
#endif
