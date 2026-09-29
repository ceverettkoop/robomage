#ifndef UNLESS_PAYMENT_H
#define UNLESS_PAYMENT_H

#include <cstddef>
#include <memory>

#include "choice_labels.h"
#include "classes/colors.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;
class FrameCtx;

// Offer an unless-cost (CR 118.12) to `controller`: pay `cost` of `kind` — {cost} generic mana or
// the exact `cost_pips` (Chain Lightning: {R}{R}), `cost` life (Ward—Pay life, CR 702.21), discard
// `cost` card(s) from hand (Reality Smasher, CR 701.8), or `cost` energy ({E}, CR 122.1c — Static
// Prison) — or decline. Returns true if the payer declined or couldn't pay. `subject` names the
// governed effect and its object, which the shared choice_labels builders word into the pay and
// decline entries. The LIFE/ENERGY yes-no and the DISCARD flow ask through `ctx` and may suspend
// (`suspended` set, return value meaningless — check it FIRST); the MANA tap-for-mana loop is a
// live-menu loop (Shape C). `decision_source` (the resolving ability's source) is the
// pending-decision context of every ask.
bool run_unless_loop(size_t cost, Zone::Ownership controller, std::shared_ptr<Orderer> orderer, Entity paid_for,
                     Entity decision_source, FrameCtx &ctx, bool &suspended, const UnlessSubject &subject,
                     UnlessPayKind kind = UnlessPayKind::MANA, const ManaValue *cost_pips = nullptr);

#endif /* UNLESS_PAYMENT_H */
