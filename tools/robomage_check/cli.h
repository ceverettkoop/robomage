#ifndef ROBOMAGE_CHECK_CLI_H
#define ROBOMAGE_CHECK_CLI_H

// robomage_check: the engine's test gate in C++ (docs/engine_fork_plan.md step 3a, docs/ci.md).
//
//   robomage_check [--tier a,b,...] [-j N] [--seed N] [--binary PATH] [--out-dir DIR]
//                  [--fuzz-games N] [--smoke-games N] [--matchups PRESET] [--mode explore|explore:patient]
//                  [--scenario GLOB ...] [--show output|narrative|dump] [--record]
//
// Runs engine subprocesses (in parallel, -j) and links no engine translation unit: only the
// std-only RMSTATE decoder and the agents' frame reader and Python-compatible generator. Tiers
// (default: all of them, in this order):
//
//   vocab      the engine's --check-decks over the top-level and league/ decks
//   concede    the CR 104.3a concession sentinels (-2 / -3) piped at bo1 / bo3 decisions over the
//              RMSTATE stream and BQUERY, results and --replay of the logged decisions
//   scenarios  every rules-regression scenario (train/regression/scenarios/*.json)
//   replay     the replay corpus: each decision log (train/regression/corpus/<name>.rmlog) replayed,
//              its output against <name>.replay byte for byte (--record rewrites the baselines)
//   state      the state format's self-checks over the replayed corpus, explore fuzz games and the
//              scenarios; dump re-encoding, decoder refusals, RMSTATE sessions, name-a-card menus
//   knowledge  rmknow --selftest / --exactness / --belief-exactness; explore games under
//              --selfcheck-knowledge; RMSTATE sessions against their dumps; rmknow --isolation and
//              --truth; the shadow classification and its scenarios; the belief layer's structure;
//              archetype inference; the belief calibration report
//   fuzz       C++ explore vs explore over league matchups (bo1, and bo3 explore:patient matches);
//              draws are findings
//   bench      decisions per second of a fixed explore workload
//   smoke      scripted hard vs hard league games through the Python driver (train/rm_driver.py)
//              and the kept Python's own checks (train/pycheck.py: the Python decoder against the
//              engine's, the driver's imports); skipped with a notice when no Python interpreter is
//              found
//
// Exit status 0 when no tier reported an error (warnings alone pass), 1 otherwise, 2 for bad usage
// or a missing binary.

namespace rmcheck {

// The gate's command line: parses it, runs the tiers, prints the report; returns the exit status.
int gate_main(int argc, char **argv);

}  // namespace rmcheck

#endif /* ROBOMAGE_CHECK_CLI_H */
