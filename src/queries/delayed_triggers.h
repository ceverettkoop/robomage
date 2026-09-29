#ifndef QUERIES_DELAYED_TRIGGERS_H
#define QUERIES_DELAYED_TRIGGERS_H

#include "../ecs/entity.h"
#include "../object_ref.h"

// ── Delayed triggers (CR 603.7) ──────────────────────────────────────────────
struct DelayedTrigger;

// THE registration path for every delayed triggered ability: stamps the fire ability's
// DelayedTriggerLink (a fresh seq from Game::next_delayed_seq, `creator` = the card whose
// ability set the trigger up, the fire kind, and the subjects with both vocab ids captured
// now) and appends the trigger to cur_game.delayed_triggers. The subjects are the fire
// ability's delayed_link.subjects when the caller pre-set them, else the first non-empty of
// dt.remembered_objects, the fire ability's non-player targets, its
// restore_remembered_exiled_with, and dt.watched. Defined in delayed_triggers.cpp.
void register_delayed_trigger(DelayedTrigger dt, const ObjectRef &creator);

// True when `e` is the watched object or one of the subjects of a delayed trigger still
// waiting in cur_game.delayed_triggers (not yet fired). Defined in delayed_triggers.cpp.
bool is_waiting_delayed_trigger_subject(Entity e);

// True when the waiting delayed trigger `dt` is scheduled to fire later in the current turn:
// a phase trigger whose fire_on_turn has arrived, whose ValidPlayer$ restriction (if any)
// names the active player, and whose step is still ahead. A leaves-the-battlefield watch is
// not turn-scheduled and returns false. Defined in delayed_triggers.cpp.
bool delayed_trigger_fires_this_turn(const DelayedTrigger &dt);

#endif /* QUERIES_DELAYED_TRIGGERS_H */
