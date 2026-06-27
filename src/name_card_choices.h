#ifndef NAME_CARD_CHOICES_H
#define NAME_CARD_CHOICES_H

#include <set>
#include <vector>

#include "classes/action.h"
#include "components/zone.h"
#include "ecs/entity.h"

// Build the deterministic NAME_CARD candidate menu used by every "name a card"
// decision (Cabal Therapy's SP$ NameCard, Disruptor Flute's ETB name-a-card).
//
// Candidates are the distinct vocab cards (card_name_to_index >= 0) owned by
// `owner` across all zones — i.e. that player's whole deck, not just their hand.
// When `exclude_lands` is true, land cards are filtered out (Cabal Therapy's
// ValidCards$ Card.nonLand); Disruptor Flute passes false to offer everything.
//
// Each returned LegalAction carries a representative owner-owned entity for the
// named card (so the per-action card id encodes the candidate), category
// NAME_CARD, and card_is_public = true. The list is ordered by copy count
// descending then name ascending (deterministic for replay) and capped at
// MAX_ACTIONS. The names parallel-array (chosen[i] is the name for choice i) is
// returned via `out_names` so the caller can record the picked name in its own
// field (cur_game.named_card vs Permanent::chosen_name).
std::vector<LegalAction> build_name_card_choices(const std::set<Entity> &entities,
                                                 Zone::Ownership owner, bool exclude_lands,
                                                 std::vector<std::string> &out_names);

#endif /* NAME_CARD_CHOICES_H */
