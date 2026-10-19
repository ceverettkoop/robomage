#ifndef AGENTS_ACTOR_H
#define AGENTS_ACTOR_H

#include <cstdint>
#include <string>

#include "agent.h"

// ── The actor: one process playing match after match from a schedule (training plan T3, T5) ───
//
// Engine side, like agents/driver.h. docs/actor.md is the reference (the schedule file, the status
// and results files, seeding, signals); the schedule itself is agents/actor_schedule.h.
//
// Options (cli_args.cpp):
//   --actor                 play matches until stopped (or --actor-matches), each drawn from the
//                           schedule; implies --machine
//   --actor-schedule FILE   the schedule (JSON, agents/actor_schedule.h); re-read between matches
//                           when the file changes or on SIGHUP
//   --actor-out DIR         the actor's own files: DIR/a<k>.status.json, DIR/a<k>.results.jsonl
//                           and the default output directories DIR/<purpose>/rmlog and
//                           DIR/train/traj
//   --actor-id K            the actor's id (default 0): its file names and its seeds
//   --actor-seed S          the run seed (default 0)
//   --actor-start I         the first match index (default 0): an actor restarted with I = the
//                           status file's next_match continues its sequence instead of replaying it
//   --actor-matches N       stop after N matches (default: until stopped)
// The command line's --deck-a/-b, --bo3, --agent-a/-b, --teacher-a/-b, --teacher-mix,
// --teacher-timeout, --knowledge-a/-b, --max-game-decisions, --trajectory-seats,
// --trajectory-omniscient, --no-auto-mana and --closed-decklist are defaults under every schedule
// entry; options that fix one game (--seed, --game-seed, --replay, --play, --actions, --player,
// --decision-log, --log-decisions, --max-decisions, the zone presets, --life-a/-b, --hand-a/-b,
// --library-a/-b), --trajectory-out (the schedule's outputs name the directories) and the frame
// writers beside the agents' feed (--rmstate-dump, --selfcheck-*) are refused.
//
// Match i of actor k under run seed S: game seed g = actor_game_seed(S, k, i), engine seed
// agents::engine_seed_for_game(g) (the seed the drivers' --game-seed g plays), the agents'
// generator seeded with g; the entry and the seat swap are drawn from g (actor_schedule.h). Its
// decision log <rmlog dir>/a<k>-m<i>.rmlog replays it alone (--replay), and rerunning the actor with
// the same S, k and schedule plays the same matches (with the same policies: a remote server's
// versions can differ, and are recorded).
//
// Every match runs the normal per-game reset; between matches the match-scoped state of the
// process is reset too (agents/driver.h agents_install: the agents' generator, the seat streams,
// the in-process feed's once-per-match frames, the knowledge instances, the teacher channel and the
// trajectory writer; game_driver.h reset_match_state), and each agent's begin_match runs (an
// in-process nn: model is reloaded when its file changed).

// The options.
extern bool actor_mode;
extern std::string actor_schedule_path;
extern std::string actor_out_dir;
extern int64_t actor_id;
extern uint64_t actor_run_seed;
extern int64_t actor_start;
extern int64_t actor_matches;

// The game seed of match `match` of actor `actor` under run seed `run_seed`: three rounds of
// SplitMix64, h = mix(run_seed), h = mix(h ^ actor), h = mix(h ^ match), shifted right by 2 (62 bits,
// a non-negative --game-seed). mix(x) is SplitMix64's output for state x: z = x + 0x9E3779B97F4A7C15,
// z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9, z = (z ^ (z >> 27)) * 0x94D049BB133111EB, z ^ (z >> 31).
uint64_t actor_game_seed(uint64_t run_seed, uint64_t actor, uint64_t match);

// The actor loop (engine_main.cpp, under --actor): plays the matches and returns the exit status
// (0: done or stopped by a signal). A fatal error exits 1 with the reason in the status file.
int actor_main();

// Whether a SIGTERM / SIGINT asked the actor to stop (false outside --actor).
bool actor_stop_requested();
// The run stops at a decision (InputLogger::end_run_by_signal): the status file says so.
void actor_stopped_at_decision();

// After each decision (agents_end_decision): the deciding seat (0 A, 1 B, -1 none), the reply as
// the decision log holds it and the behaviour distribution it was drawn from (null for none).
void actor_note_decision(int seat, int reply, const agents::Behaviour *behaviour);

#endif /* AGENTS_ACTOR_H */
