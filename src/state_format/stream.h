#ifndef STATE_FORMAT_STREAM_H
#define STATE_FORMAT_STREAM_H

#include <vector>

#include "../classes/action.h"
#include "../components/zone.h"

// ── RMSTATE frame outputs: shared bookkeeping and --machine-format rmstate (§9) ───────────────
//
// Two writers produce RMSTATE frames: --rmstate-dump FILE (state_format/dump.h: every viewer at
// every decision, beside the normal protocol) and --machine-format rmstate (here: the deciding
// seat's frame on stdout in place of BQUERY). Each drains its own copy of the event queues
// (ObsConsumer, state_format/events.h). At every decision, while either writer is on, the decision
// number advances and every viewer's handles are issued canonically before any frame is built, so
// handle numbers do not depend on which writers are on.
//
// --machine-format rmstate (implies --machine). stdout carries only RMSTATE records
// ("RMSTATE <nbytes>\n" + payload); everything else the engine prints (narrative under
// --narrative, GAME_RESULT / MATCH_RESULT, the CLI prompt of a --player seat) goes to stderr. At
// each machine decision the deciding seat (the sideboarding player between games, else the
// priority holder) gets, the first time it decides in the match, its SCHEMA, CARDS and DECKLISTS
// frames, then its STATE frame with the menu (ending with the k_* tables of the stream's own
// knowledge instance for that seat, state_format/knowledge_feed.h); the engine then reads one integer line from stdin,
// exactly as for BQUERY: a menu row index, -1 (the last row: confirm), -2 / -3 (concede game /
// match); end of input or an unparseable line is -1. Menu row i is the action reply i selects
// (the menu as offered, after the interchangeable-choice collapse). When the match is over each
// seat (A, then B) gets a terminal STATE frame (header.terminal = 1, no menu) carrying its last
// events (game_end, match_end); no reply is read. Passive BSTATE frames (--broadcast-steps) are
// not written in this mode, and it cannot be combined with --search-server.

// Called once at startup, before anything is printed: under --machine-format rmstate, moves
// stdout's descriptor aside for the frames and points stdout at stderr. Fatal with
// --search-server.
void rmstate_outputs_init();

// Whether --machine-format rmstate is on.
bool rmstate_stream_enabled();

// The seat a decision belongs to: the sideboarding player between games, else the priority
// holder (every mid-flow prompt seats priority at its chooser).
Zone::Ownership rmstate_deciding_seat();

// Called as each decision is read (InputLogger::choose), before either writer builds a frame for
// it: advances the decision number and issues every viewer's handles. No-op unless a writer (or
// --selfcheck-state, state_format/selfcheck.h) is on.
void rmstate_begin_decision();

// --machine-format rmstate: writes the deciding seat's frames for this decision (`actions` as
// offered) and returns the integer read from stdin (-1 at end of input or for an unparseable line).
int rmstate_stream_exchange(const std::vector<LegalAction> &actions);

// The match is over (after its match_end event): each writer's terminal frames, and the
// self-check's.
void rmstate_match_over();

#endif /* STATE_FORMAT_STREAM_H */
