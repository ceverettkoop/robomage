#ifndef NAME_CARD_CHOICES_H
#define NAME_CARD_CHOICES_H

#include <set>
#include <string>
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
// `valid_filter` is the ability's ValidCards$ spec, applied to each candidate's
// printed characteristics through the unified matcher (card_matches_any), so every
// filter token works uniformly — Cabal Therapy's `Card.nonLand` drops lands,
// Petrified Hamlet's `Land` offers only lands. An empty filter offers everything
// (Disruptor Flute's unfiltered "choose a card name").
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
                                                 Zone::Ownership owner,
                                                 const std::string &valid_filter,
                                                 std::vector<std::string> &out_names,
                                                 NameCardScope scope = NameCardScope::CHOOSER_ONLY);

#endif /* NAME_CARD_CHOICES_H */
