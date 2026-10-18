#ifndef SNAPSHOT_H
#define SNAPSHOT_H

#include <string>

// In-process deep-copy snapshot/restore of the complete per-game state
// (cur_game + ECS + card_db + the match stores), for search agents and the
// --selfcheck-state snapshot check.
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
// snapshot_restore without the post-restore hook: the derived state the hook would re-derive
// (g_active_statics) is left as it is, for a caller that saved and puts back its own copy (the
// --selfcheck-state snapshot check, which must see the saved decision exactly as it was).
bool snapshot_restore_exact(int slot);
bool snapshot_slot_live(int slot);  // slot holds a snapshot
bool snapshot_slot_is_match(int slot);      // slot holds a live MATCH-scoped snapshot
bool snapshot_any_match_scope_live();       // any slot holds a live MATCH-scoped snapshot
void snapshot_release_all();        // drop all snapshots
// Debug builds print a [save] / [restore] trace line on stderr for every save and restore; off
// while `quiet` is set (the --selfcheck-state snapshot check saves and restores at decisions).
void snapshot_set_quiet(bool quiet);

// Loop-safety marker, set around the loop-safe get_input call sites: a decision is loop-safe when
// the whole game state lives in cur_game and the ECS, and re-entering the main loop from that data
// re-derives the identical decision — the priority decision in play_single_game, the pregame and
// sideboard decisions, a parked pending query, the attacker/blocker SELECT prompts and the cleanup
// discard. Mid-resolution prompts that sit above live C++ stack frames are not. A snapshot saved
// at a loop-safe decision can be restored there (the --selfcheck-state snapshot check saves only
// at one).
void snapshot_set_loop_safe(bool safe);
bool snapshot_loop_safe();

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

// Sidecar state hooks: state kept outside cur_game, the ECS and the match stores that a restored
// decision must find as it was (the RMSTATE knowledge instances, state_format/knowledge_feed.h,
// which register them with their first instance). Every snapshot_save calls `save` for a blob the
// slot keeps, and every restore of that slot calls `restore` with it; a slot saved before the
// hooks were registered restores an empty blob. Unregistered (the default), nothing is saved.
void snapshot_set_sidecar_hooks(void (*save)(std::string &blob), void (*restore)(const std::string &blob));
#endif
