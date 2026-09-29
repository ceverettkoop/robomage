#ifndef ANNOUNCE_H
#define ANNOUNCE_H

#include <memory>

#include "components/ability.h"
#include "components/zone.h"
#include "resolution_frame.h"

class Orderer;

// Announcing a stack object's choices as it is put on the stack: a modal object's modes, each
// chosen mode's targets right after it, then the object's own targets and those of its chained
// sub-abilities. One routine for every kind of stack object — a spell being cast (CR 601.2b/c),
// an ability being activated (CR 602.2b) and a triggered ability being put on the stack
// (CR 603.3c/d) — so the modes and targets are public, and Ward / becomes-target abilities see
// them, before any player can respond.

// DONE: every choice is made. SUSPENDED: a choice parked a pending query through the asker —
// re-enter with the same AnnounceRT once it is answered. REMOVED: `remove_unchoosable` was set
// and no mode could be chosen (CR 603.3c) or a required target has no legal choice (CR 603.3d);
// the object is removed from the stack.
enum class AnnounceStatus { DONE, SUSPENDED, REMOVED };

// Run (or resume) the announcement of `ab`, controlled by `controller`, asking each choice
// through `asker`. The picked modes are recorded in ab.charm_chosen in printed order (the order
// they resolve in, CR 608.2c / 700.2d). Without `remove_unchoosable` (a spell or activated
// ability, whose legality was gated before the announcement began) a mode menu that runs dry
// keeps the modes chosen so far, and target picks are not pre-checked.
AnnounceStatus run_announce(Ability &ab, AnnounceRT &rt, TargetAsker &asker,
                            std::shared_ptr<Orderer> orderer, Zone::Ownership controller,
                            bool remove_unchoosable);

#endif /* ANNOUNCE_H */
