#ifndef TARGETING_H
#define TARGETING_H

#include <memory>
#include <vector>

#include "components/ability.h"
#include "components/zone.h"
#include "ecs/entity.h"

class Orderer;

// Target legality (CR 115 / 601.2c / 608.2b): which objects and players an ability may target,
// and whether the targets it chose are still legal as it resolves.

// Single source of truth for target legality. Returns true if `cand` is a legal target for `ab`
// when it is controlled by `caster`. Used both to enumerate legal targets (build_valid_targets)
// and to re-verify chosen targets at resolution (targets_still_legal).
bool is_legal_target(const Ability &ab, Entity cand, Zone::Ownership caster);

// CR 608.2b: true unless every target `ab` chose has become illegal (an optional target left
// unchosen counts as legal). A target that changed zones since it was chosen is a new object and
// illegal (CR 400.7).
bool targets_still_legal(const Ability &ab);

// The legal targets for `ability` from `priority_player`'s perspective, in menu order: opponent
// entities first (opponent player, then opponent's permanents in entity-ID order), then own
// entities. This keeps action index 0 pointing at the opponent for burn spells regardless of who
// casts, which makes the action space symmetric. Legality of each candidate is decided by
// is_legal_target; this function only chooses the candidate set and its order.
std::vector<Entity> build_valid_targets(const Ability &ability, std::shared_ptr<Orderer> orderer,
                                        Zone::Ownership priority_player);

// Returns true if the ability has no targeting requirement or at least one legal target exists.
// "Requirement" is the ability's minimum target count (CR 601.2c) as effective_target_min reads it
// before X is announced: an X-driven minimum counts as 0, since X may legally be 0. A modal
// ability (CR 700.2) needs its required number of choosable modes (has_choosable_modes).
bool has_legal_targets(const Ability &ability, std::shared_ptr<Orderer> orderer);

// ── Modes (CR 700.2) ─────────────────────────────────────────────────────────
// A modal spell or ability ("Choose one —", DB$/SP$/AB$ Charm): its modes live in
// charm_choices, and the ones announced as it was put on the stack in charm_chosen.
bool is_modal(const Ability &ab);

// Can mode `idx` of `modal` be chosen by `chooser` now (CR 700.2a/b, 603.3c)? A mode can't be
// chosen only when it requires a target and none is legal. The required minimum is
// effective_target_min's, reading the announced X when `x_announced`.
bool mode_choosable(const Ability &modal, size_t idx, std::shared_ptr<Orderer> orderer,
                    Zone::Ownership chooser, bool x_announced);

// Can `modal`'s required number of different modes (CharmNum$, default one) be chosen now?
// The legality gate for casting a modal spell or activating a modal ability (CR 601.2b, 602.2b).
bool has_choosable_modes(const Ability &modal, std::shared_ptr<Orderer> orderer,
                         Zone::Ownership chooser, bool x_announced);

// The minimum number of targets `ab` requires (CR 601.2c), the one rule behind the cast- and
// activation-legality gates (has_legal_targets), the charm-mode filter and target selection. A
// static TargetMin$ is its literal value. A non-xPaid count-SVar min (Into the Flood Maw:
// TargetMin$ X = Count$PromisedGift.0.1) is evaluated now against the current game state (which
// reads the pending gift-promise flag).
// An xPaid-driven min ("exactly X targets", Hide on the Ceiling; "up to X", Kozilek's Command)
// reads the X announced for the spell when `x_announced`; before X is chosen (the cast-legality
// gate) it counts as 0 — X may legally be 0, so it must not gate castability.
int effective_target_min(const Ability &ab, Zone::Ownership perspective,
                         std::shared_ptr<Orderer> orderer, bool x_announced);

#endif /* TARGETING_H */
