#ifndef SNAPSHOT_H
#define SNAPSHOT_H
// In-process deep-copy snapshot/restore of the complete per-game state
// (cur_game + ECS + card_db + match revealed arrays), for MCTS search.
//
// A snapshot has a SCOPE. A GAME snapshot is the usual in-game search root; it
// rolls back within the current game. A MATCH snapshot is taken at a bo3
// sideboard prompt: it additionally captures the whole between-game MatchContext,
// the deck_state store, the match globals, and the card_db generation, so a
// sideboard-rooted search can roll FORWARD across an init_ecs() teardown into the
// next game and still restore the sideboard-root state byte-identically. The
// scope is chosen automatically in snapshot_save (MATCH iff sideboard_phase).
inline constexpr int N_SNAPSHOT_SLOTS = 4;
bool snapshot_save(int slot);       // deep copy current state into slot; false on bad slot
bool snapshot_restore(int slot);    // restore state from slot; false if slot empty/bad
bool snapshot_slot_live(int slot);  // slot holds a snapshot
bool snapshot_slot_is_match(int slot);      // slot holds a live MATCH-scoped snapshot
bool snapshot_any_match_scope_live();       // any slot holds a live MATCH-scoped snapshot
void snapshot_release_all();        // drop all snapshots

// Post-restore hook, run at the end of every successful snapshot_restore().
// The snapshot covers only AUTHORITATIVE state; DERIVED global state
// (g_active_statics — the rule-613 continuous-effects registry) is normally
// re-derived by the next SBE pass, but a restore that lands on a parked
// pending-query re-emits the suspended flow BEFORE any SBE pass runs, so the
// resumed cast/payment would read the PREVIOUS simulation's statics (e.g. a
// "can't activate" hoser another sim resolved) and diverge — the az_mcts
// world-consistency violation. init_ecs registers apply_continuous_effects
// here so every restore re-derives the statics at the same choke point.
void snapshot_set_post_restore_hook(void (*hook)());
#endif
