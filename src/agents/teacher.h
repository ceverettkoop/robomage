#ifndef AGENTS_TEACHER_H
#define AGENTS_TEACHER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "agent.h"

// ── The teacher channel: a seat's labels from an external agent (training plan T4) ────────────
//
// Engine side, like agents/trajectory.h. Options (cli_args.cpp; not recorded in the decision log):
//   --teacher-a <command> / --teacher-b <command>
//                        start <command> (through /bin/sh -c, in the engine's working directory)
//                        and stream it that seat's frames: exactly the bytes the --machine
//                        protocol would send the seat (docs/state_format.md §9), from the match's
//                        first frame (SCHEMA, CARDS, DECKLISTS the first time the seat
//                        decides), a STATE frame at each of its decisions (sideboarding included),
//                        and its terminal frame at the end of the match. The command answers each
//                        decision frame with one integer line on its stdout: its label. Its stdin
//                        and stdout are one Unix stream socket; its stderr is the engine's.
//                        python/teacher.py is the scripted agent as such a command.
//   --teacher-mix <p>    each decision of a taught seat plays the teacher's label with probability
//                        p (default 0: labels only; 1: the teacher plays the seat)
//   --teacher-timeout <s>  how long the engine waits for a teacher to read its frames or reply
//                        (seconds, default 60)
//
// The label is what the trajectory writer records as the step's `teacher` (-1 none): a menu row,
// with -1 (the last row: confirm) mapped to that row. A concession sentinel (-2 / -3) from a
// teacher is recorded as -1 and never played. Any other reply, a teacher that exits or closes its
// output, or one that does not read its frames or reply within the timeout ends the run with a
// fatal error naming the seat; there is no default label.
//
// Mixing. At each decision of a taught seat that an in-process agent or script decides, the agent
// decides first (as without a teacher); then one coin u = random() is drawn from the teacher's own
// generator, Python's random.Random(s + 2**32) for s the game seed (--game-seed, else the engine
// seed), drawn in decision order over the match and from no other generator. When u < p and the
// label is a menu row the label is played, else the agent's reply. The decision log records the
// reply played, so a --replay needs no teacher. The step's behaviour distribution is the mixture
// the reply was drawn from: μ = p·onehot(label) + (1 − p)·μ_agent when the agent gave μ_agent, and
// onehot(label) at p = 1 (policy version 0 when the agent gave none); μ_agent unchanged when the
// label is -1 or p = 0; absent otherwise. --teacher-mix > 0 needs every taught seat to be decided
// in process (an agent or a script, not a human seat, the RMSTATE stream or --replay).
//
// The labels need no in-process decider: a stream seat, a human seat or a seat of --replay is
// labelled too (its frames are built for the teacher by the in-process feed). The teacher process
// lives for the run; it gets end of input when the run ends, and a teacher that then exits with a
// status other than 0 is a fatal error too. An actor (agents/actor.h) installs the channel again
// before each match: the coin is reseeded with that match's game seed, and a seat's teacher keeps
// running across matches (it sees each match's frames from its SCHEMA frame to its terminal frame)
// until a match names another command for the seat, or none; it then gets end of input and must
// exit with status 0 like at the end of the run, and the new command starts.

// The options.
extern std::string teacher_command[2];
extern std::string teacher_mix_option;
extern std::string teacher_timeout_option;

// Whether a --teacher-a / --teacher-b was given.
bool teacher_enabled();
// Whether seat `s` (0 = A, 1 = B) is taught.
bool teacher_on(int s);
// Whether --teacher-mix is above 0 (then every taught seat is decided in process).
bool teacher_mixing();

// Before the match (agents_install; an actor's every match): validates the options (`in_process[s]`:
// seat s is decided by an in-process agent or script that is not a human seat; null for none) and
// starts each teacher not already running with its command; `coin_seed` is the game seed (the
// engine seed without one). Fatal on a bad option.
void teacher_install(uint64_t coin_seed, const bool in_process[2]);

// At a decision of taught seat `s`: writes its frames for this decision (the once-per-match frames
// the first time, then the STATE frame; `menu_rows` the menu's size, `number` the run's decision
// number for messages). The reply is read by teacher_label.
void teacher_send(
    int s, const std::vector<std::string> &match_frames, const std::string &state, size_t menu_rows, size_t number);
// The current decision's label: the taught seat's reply, read (waiting for it) the first time it is
// asked for; -1 when the decision is not a taught seat's, or the teacher conceded.
int teacher_label();
// After the decision: the label for the record (teacher_label), and the decision is closed.
int teacher_take_label();

// Mixing at the current decision, a taught seat's under --teacher-mix: draws the coin and returns
// the reply to play (the label or `agent_reply`); `*recorded` is set to the behaviour distribution
// the reply was drawn from (`agent_behaviour` itself, the mixture in storage the channel owns, or
// nullptr) for `menu_rows` rows.
int teacher_mix(
    int agent_reply, const agents::Behaviour *agent_behaviour, size_t menu_rows, const agents::Behaviour **recorded);

// The match is over (after the stream's terminal frames): each taught seat's terminal frame (and
// its once-per-match frames when it never decided).
void teacher_match_over();
// The run is over: each teacher gets end of input and must exit with status 0.
void teacher_finish();

#endif /* AGENTS_TEACHER_H */
