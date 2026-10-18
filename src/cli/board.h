#ifndef CLI_BOARD_H
#define CLI_BOARD_H

#include <cstdint>
#include <string>
#include <vector>

#include "../agents/decision_view.h"

// ── The C++ CLI's renderer: one seat's decoded RMSTATE frames as text (plan step 4b) ──────────
//
// Reads only the deciding seat's decoded frames (agents/decision_view.h: its STATE frame with the
// k_* knowledge tables appended to it, and the stream's SCHEMA, CARDS, DECKLISTS and METAGAME
// frames), never the engine, so it shows exactly what that seat may see: a display bug is a format
// bug. Standard library and the agents' frame reader only. The human seat (cli/human.h) and
// --trace-decisions (agents/trace.h) print it.
//
// Objects are labelled by their handle in the viewer's frame, `#<handle>`: it stays the same
// while the object stays where it is (a zone change makes a new object with a new handle).
// Players are named absolutely ("Player A"); the viewer's own seat is marked "(you)" where both
// players are listed. Turns are shown from 1 (the frame's turn counter is 0-based). A card is
// named as a whole card (a split card by its two names, "Dead // Gone", CR 709.4), except a split
// card on the stack, which is the half cast (CR 709.3b).

namespace cli {

// "Game 1, turn 3, First main: Player A's turn; Player B has priority" (the opening hands and the
// sideboarding phase have their own wording).
std::string board_header(const agents::DecisionView &v);

// The board, top to bottom: the players (life, zone sizes, mana pool, counters, designations),
// each battlefield (the opponent's first: P/T, statuses, counters, attachments, combat), the stack
// (top first, with targets), the viewer's hand, graveyards and exile, open reveal windows, player
// effects and delayed triggers, and the knowledge summary; in the sideboarding phase, the players
// and the viewer's current deck and sideboard. Sections with nothing to show are left out (an
// empty battlefield or graveyard says so). The pending decision is pending_lines'.
std::vector<std::string> board_lines(const agents::DecisionView &v);

// The pending decision: what is being decided ("Decision: casting Force of Will: pay the cost")
// and, at a cost-payment decision, what is still owed by pip and what it pays for.
std::vector<std::string> pending_lines(const agents::DecisionView &v);

// The menu, one "<i>: <description>" line per row (row i = reply i). With `annotate`, the
// indices are right-aligned and a last row confirming attackers or blockers is marked as the one
// -1 also selects.
std::vector<std::string> menu_lines(const agents::DecisionView &v, bool annotate);

// The knowledge panel from the k_* (and b_*) tables. `full`: every fact (pools with their card
// bounds, known positions, copies seen, consistent reference lists, archetypes, beliefs); else a
// short "you know / you can deduce" summary (cards known in the opponent's hand, known library
// positions, the opponent's archetype), empty when there is nothing to say or knowledge is off.
std::vector<std::string> knowledge_lines(const agents::DecisionView &v, bool full);

// One zone as the viewer sees it: `zone` is one of zone_names(), of the opponent's side when
// `opponent` (the stack has no side). Hidden zones show their size, the cards the frame
// identifies in them and what the knowledge tables say. An unknown zone name gives one line
// saying so.
std::vector<std::string> zone_lines(const agents::DecisionView &v, const std::string &zone, bool opponent);
// The zone names zone_lines accepts.
const std::vector<std::string> &zone_names();

// A card's details from the CARDS frame (each face's name, cost, type line, P/T or loyalty and
// oracle text). `query` is an object's handle ("12" or "#12": the object as the frame shows it
// now, then its card) or a card name (exact, else a unique substring; case-insensitive).
std::vector<std::string> card_lines(const agents::DecisionView &v, const std::string &query);

// The frame's observation events (what the seat saw happen since its previous frame), one line
// each, prefixed with the turn ("T3: Player B casts Lightning Bolt from hand"); consecutive
// identical lines are merged ("... (x7)"), consecutive draws of one player are one line, and the
// end of a reveal window has none.
std::vector<std::string> event_lines(const agents::DecisionView &v);

// "Player A" / "Player B" for a `seat` enum value.
std::string seat_name(const agents::DecisionView &v, uint64_t seat);

}  // namespace cli

#endif /* CLI_BOARD_H */
