#ifndef AGENTS_TRAJECTORY_H
#define AGENTS_TRAJECTORY_H

#include <string>
#include <vector>

#include "../classes/action.h"
#include "../featurize/episode.h"
#include "../state_format/decode.h"
#include "agent.h"

// ── The trajectory writer: one episode file per game (docs/trajectory_format.md) ──────────────
//
// Engine side, like agents/driver.h (the episode format itself is featurize/episode.h, std only).
//
// Options (cli_args.cpp; not recorded in the decision log, honored under --replay):
//   --trajectory-out DIR     write every game's episode into DIR (created)
//   --trajectory-seats A|B|AB  the seats whose decisions are recorded (default AB)
//   --trajectory-omniscient  each decision also carries the omniscient frame's featurization and
//                            the learner targets (featurize/targets.h); learner-side data, which
//                            no agent receives
//
// An episode is one game with the sideboarding phase before it (a sideboarding frame's game is
// already the upcoming one), so a bo3 match writes one episode per game played. Each recorded
// seat's decisions are featurized from the decoded frame the in-process feed built for the
// decision (agents/driver.h serves every recorded seat), through the actor's entry point
// (featurize_seat), or taken from the deciding nn: agent, which featurized that same frame
// (Behaviour::tensors). Under --feature-history each recorded seat's frames feed that seat's event
// history (featurize/history.h), kept for the match. The behaviour distribution μ, the policy
// version and every policy's evaluation come from the deciding agent's last_behaviour (absent for
// an agent with none: explore, first, a script's own rows, a stream seat, every decision of
// --replay), or under --teacher-mix from the teacher channel's mixture (agents/teacher.h); the
// reply is the decision log's (a menu row or a concession sentinel); the teacher label the
// channel's for a taught seat, else -1.
//
// Outside --replay a decision log is always written: --decision-log when given, else
// DIR/s<engine seed>.rmlog. Files: DIR/<stem>.g<n>.rmtraj for game n (1-based) of the match, <stem>
// the decision log's file name without its extension (the --replay log's under --replay), each
// written whole through a temporary file renamed over it when the game ends (or the run stops at
// --max-decisions, or a human seat quits), so a reader polling DIR never sees part of one. Every
// episode names its log, and replaying that log with --trajectory-out writes the same episodes
// (μ, the policy versions and evaluations absent; the agent fields "replay").

// The options.
extern std::string trajectory_out_dir;
extern std::string trajectory_seats_option;
extern bool trajectory_omniscient;

// Whether --trajectory-out was given.
bool trajectory_enabled();
// Whether seat `s` (0 = A, 1 = B) is recorded.
bool trajectory_records_seat(int s);

// Before the decision log is opened (game_loop): outside --replay, points it at DIR/s<seed>.rmlog
// when no --decision-log was given and turns logging on. No-op unless --trajectory-out is given.
void trajectory_configure_decision_log(unsigned int engine_seed);

// Before the match (agents_install; an actor's every match, whose options may differ): validates
// the options, creates DIR and loads the featurizer's inputs (once per process); fatal on a bad
// option. `agent_specs` describe each seat's decider for the header. No episode is open after it.
void trajectory_install(unsigned int engine_seed, const std::string agent_specs[2]);

// Every decision, as it is read (agents_begin_decision), after the deciding seat's frame was
// decoded: `seat` (0, 1, or -1 for none) and its decoded STATE frame (null when the seat is not
// recorded), with the menu as offered. Builds the omniscient featurization when asked.
void trajectory_begin_decision(int seat, const rmstate::DecodedFrame *frame, const std::vector<LegalAction> &actions);
// After the decision: the reply as the decision log holds it (a menu row or a concession sentinel),
// the behaviour distribution it was drawn from (the deciding agent's last_behaviour or the teacher
// channel's mixture, agents/teacher.h; null for none, a stream seat or --replay) and the teacher's
// label (-1 none). Appends the record of a recorded seat's decision.
void trajectory_end_decision(int reply, const agents::Behaviour *behaviour, int teacher);

// Game boundaries (game_driver.cpp): a game starts (play_single_game: who plays first and its
// engine seed; the match score and game number from the match globals), a game ended with
// `winner` (Zone::PLAYER_A / PLAYER_B, else a draw; end reason cap, results 0, when
// --max-game-decisions stopped it): its episode is written.
void trajectory_game_start(bool a_goes_first, unsigned int game_engine_seed);
void trajectory_game_end(int winner);
// The match is over: an episode left open (sideboarding before a game the match was conceded
// before) is written, a concession.
void trajectory_match_over();
// The run ends with no result (`reason`: CAP at --max-decisions, QUIT): the open episode is
// written with that end reason and results 0.
void trajectory_end_run(featurize::EndReason reason);

#endif /* AGENTS_TRAJECTORY_H */
