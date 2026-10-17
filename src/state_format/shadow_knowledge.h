#ifndef STATE_FORMAT_SHADOW_KNOWLEDGE_H
#define STATE_FORMAT_SHADOW_KNOWLEDGE_H

#include "../components/zone.h"
#include "../knowledge/frame_view.h"
#include "../knowledge/k_rows.h"
#include "../knowledge/truth.h"

// ── The shadow comparison: k_* against the old vector's memory (docs/knowledge_store.md §7) ────
//
// Part of --selfcheck-knowledge. At every decision, for each seat, the knowledge sidecar's implied
// view (the seat's k_* tables as its self-check frame carries them) is compared with what the BQUERY
// state vector shows that seat, read through populate_gamestate (machine_io.h), the serializer's
// own source, and opp_known_hand_cards:
//
//   top   the known top positions 0..KNOWN_TOP_LIBRARY_SIZE-1 of each library: the k_position rows
//         with a card (positions from the top) against known_top_library_self / _opp;
//   hand  the opponent's cards definitely in hand: k_known_hand copies against the old memory
//         behind the opp_known_hand slots (every opponent-hand card with Zone::identity_known,
//         opp_known_hand_cards; the slots show the first MAX_HAND_SLOTS of them), per card;
//   seen  the opponent's cards seen this match: k_seen rows with match > 0 against the opponent's
//         registered-decklist revealed bits (opp_deck_main_revealed / opp_deck_side_revealed), per
//         card.
//
// A mismatch is not an engine error: train/test_knowledge.py classifies it against
// train/regression/knowledge_divergences.json. A mismatch is printed to stderr when it appears
// (one already there at the seat's previous decision, with the same kind, direction, owner and
// card, is only counted, whatever its values now are), as one line:
//
//   SHADOW: <kind> <direction> viewer=<A|B> owner=<self|opp> card=<name> old=<..> new=<..> <detail>
//           truth=<1|0|?> game=<n> turn=<n> decision=<n> sideboarding=<0|1> events=<…>
//
// direction: old_more (the old vector reports something the k_* tables do not), new_more (the
// other way) or conflict (both name a card, different ones). For top, card is the card the side
// that knows more names (the old vector's for a conflict), old / new are the cards each side names
// (`-` = unknown) and the detail is at=<positions>; for hand and seen, old / new are copy counts
// (seen: 0 or 1), and a hand line's detail is left=<n> entered=<n> slots=<n>: the copies of the
// card the frame's moves show leaving and entering the opponent's hand, and the copies of it the
// opp_known_hand slots show. truth says whether the old vector's
// claim holds in the true world (top: the card is there; hand: the hand holds that many; seen and
// new_more top: `?`). events lists the frame's events (those since the previous decision) as
// `kind:zone.owner>to_zone.to_owner@source` with owners relative to the viewer, distinct ones in
// order, `×n` for repeats. A `SHADOW-KNOWLEDGE: decisions=… compared=… mismatches=… shown=…` summary
// goes to stderr at match end (mismatches counts the mismatches present at each decision, a top
// mismatch once per position; shown counts the lines printed).

// One seat's comparison at one decision: `rows` are the seat's k_* tables, `snap` its frame's
// events and header facts, `truth` the true world of the decision.
void shadow_knowledge_seat(Zone::Ownership viewer, const knowledge::KRows &rows, const knowledge::Snapshot &snap,
    const knowledge::Truth &truth);

// The match is over: the summary line.
void shadow_knowledge_match_over();

#endif /* STATE_FORMAT_SHADOW_KNOWLEDGE_H */
