#ifndef NAME_CARD_CHOICES_H
#define NAME_CARD_CHOICES_H

#include <set>
#include <vector>

#include "classes/action.h"
#include "components/zone.h"
#include "ecs/entity.h"

// Build the deterministic NAME_CARD candidate menu used by every "name a card"
// decision (Cabal Therapy's SP$ NameCard, Disruptor Flute's ETB name-a-card,
// Petrified Hamlet's ETB name-a-land).
//
// The candidate set is a deliberately LIMITED, context-driven slice of the match
// (a documented deviation from CR 201.4's "name any card"; see CLAUDE.md). The
// `scope` selects whose deck the candidates are drawn from:
//   CHOOSER_ONLY  — distinct vocab cards owned by `owner` across all zones (that
//                   one player's whole deck). Used by Cabal Therapy (the target
//                   player) and Disruptor Flute (the opponent).
//   BOTH_PLAYERS  — distinct vocab cards owned by EITHER player A or player B,
//                   de-duped by name. Used by land-naming cards (Petrified
//                   Hamlet / Alpine Moon-style "name a land"), so a land that
//                   exists only in the opponent's deck is still nameable. `owner`
//                   is ignored in this scope.
//
// When `exclude_lands` is true, land cards are filtered out (Cabal Therapy's
// ValidCards$ Card.nonLand); Disruptor Flute passes false to offer everything.
// When `only_lands` is true, ONLY land cards are offered (Petrified Hamlet's
// ValidCards$ Land — "name a land card"); exclude_lands and only_lands are mutually
// exclusive (only_lands wins if both are set).
//
// Each returned LegalAction carries a representative owner-owned entity for the
// named card (so the per-action card id encodes the candidate), category
// NAME_CARD, and card_is_public = true. The list is ordered by copy count
// descending then name ascending (deterministic for replay) and capped at
// MAX_ACTIONS. The names parallel-array (chosen[i] is the name for choice i) is
// returned via `out_names` so the caller can record the picked name in its own
// field (cur_game.named_card vs Permanent::chosen_name).
enum class NameCardScope { CHOOSER_ONLY, BOTH_PLAYERS };

std::vector<LegalAction> build_name_card_choices(const std::set<Entity> &entities,
                                                 Zone::Ownership owner, bool exclude_lands,
                                                 std::vector<std::string> &out_names,
                                                 bool only_lands = false,
                                                 NameCardScope scope = NameCardScope::CHOOSER_ONLY);

#endif /* NAME_CARD_CHOICES_H */
